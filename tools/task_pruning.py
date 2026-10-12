"""Experiment 17: prune silent edges, silent neurons and both from a flyvis network, read on
the optic-flow task's end-point error (EPE). See experiments/exp17_task_pruning_epe.md.

    python tools/task_pruning.py --model gt                     # flyvis flow/0000/000
    python tools/task_pruning.py --model cv00 [--device cuda:1] # experiment 2's learned circuit
    python tools/task_pruning.py --gates                        # the three gates only
    python tools/task_pruning.py --model gt --epe-bound 5.3 --out <dir> --mask-out <file.npz>
                                                                # stricter bound + the masks, 217-column too
    python tools/task_pruning.py --report                       # figure + table in the experiment file

Every model is the 721-column (extent 15) network of flow/0000/000 with 000's decoder; the
learned models only replace its shared parameters (syn_strength per type pair, time_const
and bias per cell type) with what experiment 2's general-form GNN recovered on the central
217 columns. Silence is measured on the TRAINING clips; the EPE on the held-out clips.
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import torch

logging.disable(logging.WARNING)
# the env's activate.d exports this; a python started without activation would otherwise
# resolve the data root inside site-packages and try to download Sintel (5.6 GB)
os.environ.setdefault("FLYVIS_ROOT_DIR", "/groups/saalfeld/home/allierc/GraphData/flyvis")

from flyvis import Network, NetworkView  # noqa: E402
from flyvis.network.dynamics import PPNeuronIGRSynapses  # noqa: E402
from flyvis.task.objectives import epe as epe_fn  # noqa: E402
from flyvis.task.tasks import Task  # noqa: E402
from flyvis.utils.dataset_utils import IndexSampler  # noqa: E402
from flyvis.utils.config_utils import CONFIG_PATH, get_default_config  # noqa: E402

GT_NAME = "flow/0000/000"
GT_STORED_EPE = 5.1348                       # validation/epe.h5 of 0000/000, last entry
EPE_BOUND = 5.4                              # the bound given for 000 (median of the 50 members: 5.4031)
# The bound as a fraction of the TASK GAIN, the EPE a model saves over predicting zero flow (Z, 5.7742
# on the held-out clips): 5.4 keeps KEEP = (Z - 5.4) / (Z - 5.1348) = 58.5 % of 000's gain, and every
# model must keep that fraction of its own: bound = Z - KEEP * (Z - unpruned EPE). A fixed tolerance in
# EPE (+0.265) put fold cv04's bound (5.847) above Z itself, where removing 90 % of the neurons passed.
RUN = "/groups/saalfeld/home/allierc/GraphData/log/fly/flyvis_noise_005_blank50_condl25_{fold}"
OUT = "/groups/saalfeld/home/allierc/GraphData/log/fly/exp17_task_pruning"
RUNGS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]   # 0.0 = exact zeros only
INPUT_TYPES = {f"R{k}" for k in range(1, 9)}                                 # the photoreceptors, never removed


class PrunablePPNeuronIGRSynapses(PPNeuronIGRSynapses):
    """flyvis's current-based dynamics with every edge weight multiplied by `edge_mask`
    (None = no mask). Defined here and selected by class name: flyvis resolves the dynamics
    against the live subclass tree, so nothing in flyvis is patched."""

    edge_mask = None

    def write_derived_params(self, params, **kwargs):
        super().write_derived_params(params, **kwargs)
        if self.edge_mask is not None:
            params.edges.weight = params.edges.weight * self.edge_mask


def A(t):
    return t.detach().cpu().numpy()


def build(device):
    nv = NetworkView(GT_NAME)
    cfg = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
    cfg.connectome.extent = 15
    cfg.dynamics.type = PrunablePPNeuronIGRSynapses.__name__
    net = Network(**cfg)
    # forward_subclass warns and falls back to the base class (all methods `pass`) on an
    # unknown name: the check that the mask is actually applied
    assert type(net.dynamics).__name__ == PrunablePPNeuronIGRSynapses.__name__, type(net.dynamics)
    net.load_state_dict(nv.init_network(checkpoint=0).state_dict())
    net = net.to(device).eval()
    decoder = nv.init_decoder(checkpoint=0)["flow"].to(device).eval()
    tc = dict(nv.dir.config.task)
    tc.pop("type", None)
    if "task_weight" in tc:
        tc["task_weights"] = tc.pop("task_weight")
    # the published config predates the key: without it flyvis draws a random split sharing
    # 6 of the 16 validation sequences with the one 000 was trained and scored on
    tc["original_split"] = True
    return nv, net, decoder, Task(**tc)


def shared(net):
    """Per-edge and per-node views of the shared parameters, as numpy."""
    ep, npar = net.edge_params, net.node_params
    return dict(
        c=A(ep.sign.semantic_values[ep.sign.readers["edges"]]) * A(ep.syn_count.semantic_values[ep.syn_count.readers["edges"]]),
        pair=A(ep.syn_strength.readers["edges"]), strength=A(ep.syn_strength.semantic_values),
        ttype=A(npar.time_const.readers["nodes"]), time_const=A(npar.time_const.semantic_values),
        btype=A(npar.bias.readers["nodes"]), bias=A(npar.bias.semantic_values),
    )


def fit_shared(w, tau, vrest, sh8):
    """The shared flyvis parameters from per-edge W and per-neuron tau, V_rest of the 217-column
    (extent-8) network, whose edge order is flyvis's (gate 2). syn_strength: the least-squares
    fit of W_e = sign_e * syn_count_e * strength_pair over each pair's edges, i.e. weighted by
    the edge's count (the median of W_e / (sign_e * count_e) is dominated by low-count edges);
    time_const, bias: the median over each cell type's neurons."""
    P, T = len(sh8["strength"]), len(sh8["time_const"])
    c, p = sh8["c"], sh8["pair"]
    s = np.bincount(p, c * w, P) / np.bincount(p, c * c, P)
    t = np.array([np.median(tau[sh8["ttype"] == k]) for k in range(T)])
    b = np.array([np.median(vrest[sh8["btype"] == k]) for k in range(T)])
    return s, t, b


def set_shared(net, strength, time_const, bias):
    sd = net.state_dict()
    for key, v in (("edges_syn_strength", strength), ("nodes_time_const", time_const), ("nodes_bias", bias)):
        sd[key] = torch.as_tensor(v, dtype=sd[key].dtype)
    net.load_state_dict(sd)
    sh = shared(net)
    for got, want in ((sh["strength"], strength), (sh["time_const"], time_const), (sh["bias"], bias)):
        assert np.allclose(got, want, atol=1e-6), "a raw parameter is not its semantic value"


def net8_shared():
    cfg = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
    cfg.connectome.extent = 8
    n8 = Network(**cfg)
    n8.load_state_dict(NetworkView(GT_NAME).init_network(checkpoint=0).state_dict())
    return shared(n8)


def _batches(net, task, loader):
    states = {}
    with torch.no_grad(), task.dataset.augmentation(False):
        for data in loader:
            lum = data["lum"]
            n = lum.shape[0]
            if n not in states:
                states[n] = net.steady_state(t_pre=0.25, dt=task.dataset.dt, batch_size=n, value=0.5)
            net.stimulus.zero(n, lum.shape[1])
            net.stimulus.add_input(lum)
            yield data, net(net.stimulus(), task.dataset.dt, state=states[n])


def evaluate(net, decoder, task, mask=None):
    """Held-out EPE, the loop of MultiTaskSolver.test; mean over batches."""
    net.dynamics.edge_mask = None if mask is None else torch.as_tensor(mask, dtype=torch.float32, device=net._source_indices.device)
    e = []
    for data, act in _batches(net, task, task.val_data):
        y_est = decoder(act)
        e.append(float(epe_fn(y_est, data["flow"].to(y_est.device))))
    net.dynamics.edge_mask = None
    return float(np.mean(e))


def silence(net, task, loader=None):
    """Mean rectified output relu(v_i) per neuron over everything the network simulates for a
    clip: the initial state (activity = bias), the 0.25-s grey-screen warm-up (12 steps) that sets
    the state every clip starts from, then the clip; over the training clips (or `loader`). A neuron silent on the clips but
    active in the warm-up still sets its targets' starting point: 10,562 of the 14,760 sources
    silent on every held-out clip frame are active there."""
    acc, n, warm = 0.0, 0, {}
    sh = shared(net)
    # the warm-up starts from flyvis's initial state, activity = bias, and its first step reads
    # it: a neuron with a positive bias sends during that step whatever it does afterwards
    init = torch.relu(torch.as_tensor(sh["bias"][sh["btype"]], device=net._source_indices.device))
    if loader is None:
        # all 51 training clips in a fixed order: task.train_data shuffles and drops the last
        # partial batch, so it would leave out a different 3 clips on every call
        from torch.utils.data import DataLoader
        loader = DataLoader(task.dataset, batch_size=4, sampler=IndexSampler(task.train_seq_index))
    for _, act in _batches(net, task, loader):
        b = act.shape[0]
        if b not in warm:
            with torch.no_grad():
                states = net.steady_state(t_pre=0.25, dt=task.dataset.dt, batch_size=b, value=0.5, return_last=False)
            warm[b] = (b * init + sum(torch.relu(st.nodes.activity).sum(dim=0) for st in states), len(states) + 1)
        acc = acc + torch.relu(act).sum(dim=(0, 1)) + warm[b][0]
        n += b * act.shape[1] + b * warm[b][1]
    return A(acc / n)


def prune_sets(net, s, q):
    """Edge masks (1 = kept) of the three analyses at rung q: the exact zeros, plus for q > 0 the
    lowest fraction q by rank. The zeros are in every rung so the ladder is monotone: a learned
    model's strengths clamped to 0 make ~12 % of its edges exactly silent, more than the 5 % rung."""
    sh = shared(net)
    w = sh["c"] * sh["strength"][sh["pair"]]
    src, tgt = A(net._source_indices), A(net._target_indices)
    m = np.abs(w) * s[src]                                       # edge traffic: mean |message|
    types = np.asarray(net.connectome.nodes.type[:]).astype(str)
    cand = ~np.isin(types, list(INPUT_TYPES))

    def lowest(x, ok, frac):
        out = np.zeros(len(x), bool)
        out[ok] = x[ok] == 0
        if frac > 0.0:
            idx = np.flatnonzero(ok)
            out[idx[np.argsort(x[idx], kind="stable")[: int(round(frac * len(idx)))]]] = True
        return out

    dead_e = lowest(m, np.ones(len(m), bool), q)
    dead_n = lowest(s, cand, q)
    dead_ne = dead_n[src] | dead_n[tgt]
    return {"edges": (~dead_e, dead_e.mean(), 0.0),
            "neurons": (~dead_ne, dead_ne.mean(), dead_n.sum() / cand.sum()),
            "both": (~(dead_e | dead_ne), (dead_e | dead_ne).mean(), dead_n.sum() / cand.sum()),
            "neuron_keep": ~dead_n}


def zero_epe(task):
    """The EPE of predicting zero flow on the held-out clips: the score of having learned nothing."""
    with torch.no_grad(), task.dataset.augmentation(False):
        return float(np.mean([float(epe_fn(torch.zeros_like(d["flow"]), d["flow"])) for d in task.val_data]))


def ladder(net, decoder, task, s, analysis, bound, refine=3):
    rows = []
    def run(q):
        mask, fe, fn = prune_sets(net, s, q)[analysis]
        t0 = time.time()
        e = evaluate(net, decoder, task, mask)
        rows.append({"q": q, "edges_removed": float(fe), "neurons_removed": float(fn), "epe": e})
        print(f"  {analysis:7s} q {q:5.3f}  edges removed {fe:6.1%}  neurons removed {fn:6.1%}  "
              f"EPE {e:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        return e

    lo = None
    for q in RUNGS:
        if run(q) > bound:
            hi = q
            break
        lo = q
    else:
        hi = None
    # bisection between the last kept rung and the first rejected one
    if lo is not None and hi is not None:
        for _ in range(refine):
            mid = (lo + hi) / 2
            if run(mid) <= bound:
                lo = mid
            else:
                hi = mid
    kept = [r for r in rows if r["epe"] <= bound]
    best = max(kept, key=lambda r: r["q"]) if kept else None
    return {"rows": sorted(rows, key=lambda r: r["q"]), "kept": best, "first_rejected_q": hi}


def gates(device):
    nv, net, decoder, task = build(device)
    e0 = evaluate(net, decoder, task)
    print(f"gate 1: recomputed EPE of {GT_NAME} {e0:.4f}, stored {GT_STORED_EPE}")
    sh8 = net8_shared()
    gt = torch.load(RUN.format(fold="cv00") + "/gt_weights.pt", map_location="cpu").numpy()
    s, _, _ = fit_shared(gt, np.ones(len(sh8["ttype"])), np.ones(len(sh8["btype"])), sh8)
    d = np.abs(s - sh8["strength"]).max()
    print(f"gate 2: strengths rebuilt from the true per-edge weights, max |diff| {d:.2e}")
    e1 = evaluate(net, decoder, task, np.ones(net.n_edges))
    # the mask mechanism: edges whose source sends exactly nothing on the HELD-OUT clips
    sv = silence(net, task, task.val_data)
    mask, fe, _ = prune_sets(net, sv, 0.0)["edges"]
    e2 = evaluate(net, decoder, task, mask)
    print(f"gate 3: unmasked {e0:.6f}; mask of ones {e1:.6f}; exact-zero edges removed ({fe:.1%}) {e2:.6f}")
    # 1e-6: GPU reductions are not bit-reproducible from one pass to the next
    ok = abs(e0 - GT_STORED_EPE) < 1e-3 and d < 1e-6 and abs(e1 - e0) < 1e-6 and abs(e2 - e0) < 1e-6
    print("gates", "PASS" if ok else "FAIL")
    return ok


def node_ids(net):
    """(type, u, v) of every node, in the network's node order."""
    nodes = net.connectome.nodes
    return list(zip(np.asarray(nodes.type[:]).astype(str).tolist(), np.asarray(nodes.u[:]).tolist(),
                    np.asarray(nodes.v[:]).tolist()))


def map_to_extent8(net15):
    """Index of every extent-8 node and edge in the extent-15 network, matched by identity:
    node (type, u, v); edge (source node, target node). Asserts each extent-8 element matches
    exactly one extent-15 element, and that matched edges carry the same sign * syn_count."""
    cfg = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
    cfg.connectome.extent = 8
    net8 = Network(**cfg)
    ids15, ids8 = node_ids(net15), node_ids(net8)
    where15 = {}
    for i, k in enumerate(ids15):
        where15.setdefault(k, []).append(i)
    assert all(len(v) == 1 for v in where15.values()), "duplicate (type, u, v) in the extent-15 network"
    miss = [k for k in ids8 if k not in where15]
    assert not miss, f"{len(miss)} extent-8 nodes have no extent-15 match"
    n_map = np.array([where15[k][0] for k in ids8])
    assert len(np.unique(n_map)) == len(n_map)
    N15 = len(ids15)
    key15 = A(net15._source_indices).astype(np.int64) * N15 + A(net15._target_indices)
    order = np.argsort(key15, kind="stable")
    sk = key15[order]
    assert (np.diff(sk) > 0).all(), "duplicate (source, target) edge in the extent-15 network"
    key8 = n_map[A(net8._source_indices)].astype(np.int64) * N15 + n_map[A(net8._target_indices)]
    pos = np.searchsorted(sk, key8)
    pos[pos >= len(sk)] = 0
    assert (sk[pos] == key8).all(), f"{(sk[pos] != key8).sum()} extent-8 edges have no extent-15 match"
    e_map = order[pos]
    assert len(np.unique(e_map)) == len(e_map)
    c15, c8 = shared(net15)["c"], shared(net8)["c"]
    # float32 rounding only: syn_count is stored per lattice (max |diff| 3.8e-6 of counts ~1-100)
    assert np.allclose(c15[e_map], c8, rtol=1e-5, atol=1e-5), "matched edges differ in sign * syn_count"
    return n_map, e_map


def save_masks(net, s_out, an, info, path):
    """The silent sets at the kept rungs (edges: the edges ladder's; neurons: the neurons
    ladder's) on the 721-column network, and mapped onto the 217-column network of the GNN
    datasets (flyvis extent-8 order)."""
    ke, kn = an["edges"]["kept"], an["neurons"]["kept"]
    assert ke and kn, "no kept rung"
    keep_e15, fe, _ = prune_sets(net, s_out, ke["q"])["edges"]
    keep_n15 = prune_sets(net, s_out, kn["q"])["neuron_keep"]
    n_map, e_map = map_to_extent8(net)
    keep_e8, keep_n8 = keep_e15[e_map], keep_n15[n_map]
    assert keep_e8.shape == (434112,) and keep_n8.shape == (13741,), (keep_e8.shape, keep_n8.shape)
    readme = (
        "Silent (~null) edges and neurons of flyvis flow/0000/000 on the optic-flow task, from "
        "tools/task_pruning.py (experiment 17). Silence on the 51 training clips: neuron output "
        "s_i = mean relu(v_i) over initial state + 12-step warm-up + clip frames; edge traffic "
        "m_e = |w_e| * s_source. A ladder removes the exact zeros plus the lowest fraction q (edges: of m "
        "over all edges; neurons: of s over the non-photoreceptor neurons), bisected 3 times; the kept "
        "rung is the largest q whose held-out Sintel EPE stays <= epe_bound. True = KEPT, False = silent "
        "(removed). Keys: keep_edges_217 (bool, 434112) and keep_neurons_217 (bool, 13741): the masks on "
        "the central 217 columns (flyvis extent 8), in flyvis's extent-8 edge/node order = the order of "
        "the connectome-gnn datasets (gt_weights.pt); keep_edges_721 (bool, 1513231) and keep_neurons_721 "
        "(bool, 45669): the same on the 721-column (extent 15) network the ladders ran on; "
        "edge_map_217_to_721 / node_map_217_to_721: index of each extent-8 element in extent 15, matched "
        "by identity (node: type,u,v; edge: source node + target node); q_edges, q_neurons: the kept "
        "rungs; epe_edges, epe_neurons: their held-out EPE; epe_bound: the bound used (for 000); "
        "epe_unpruned, epe_zero: 000 unpruned and the zero-flow prediction; silence_721: s_i per neuron. "
        "The edges mask comes from the edges ladder, the neurons mask from the neurons ladder: removing "
        "a neuron (the neurons ladder) also removes all its edges, which keep_edges_217 does NOT include.")
    np.savez(path, keep_edges_217=keep_e8, keep_neurons_217=keep_n8, keep_edges_721=keep_e15,
             keep_neurons_721=keep_n15, edge_map_217_to_721=e_map, node_map_217_to_721=n_map,
             q_edges=ke["q"], q_neurons=kn["q"], epe_edges=ke["epe"], epe_neurons=kn["epe"],
             epe_bound=info["epe_bound"], epe_unpruned=info["epe_unpruned"], epe_zero=info["epe_zero"],
             silence_721=s_out, readme=readme)
    info["mask_file"] = path
    for lat, e, n in (("721 columns", keep_e15, keep_n15), ("217 columns", keep_e8, keep_n8)):
        print(f"  masks, {lat}: {len(e):,} edges, {(~e).sum():,} removed ({(~e).mean():.2%}); "
              f"{len(n):,} neurons, {(~n).sum():,} removed ({(~n).mean():.2%})", flush=True)
    print("wrote", path, flush=True)


MODELS = ["gt", "cv00", "cv01", "cv02", "cv03", "cv04"]
MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "exp17_task_pruning_epe.md")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "presentation", "Fig", "exp17_pruning.png")


def report():
    """The report figure and the results table in the experiment file, both from pruning.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {m: json.load(open(os.path.join(OUT, m, "pruning.json"))) for m in MODELS
           if os.path.exists(os.path.join(OUT, m, "pruning.json"))}
    titles = {"edges": "silent edges", "neurons": "silent neurons", "both": "both, same rung"}
    xkey = {"edges": "edges_removed", "neurons": "neurons_removed", "both": "edges_removed"}
    xlab = {"edges": "fraction of edges removed", "neurons": "fraction of neurons removed",
            "both": "fraction of edges removed"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), facecolor="white")
    for ax, (a, letter) in zip(axes, zip(("edges", "neurons", "both"), "abc")):
        for m, r in res.items():
            rows = r["analyses"][a]["rows"]
            x = [row[xkey[a]] for row in rows]
            y = [(r["epe_zero"] - row["epe"]) / (r["epe_zero"] - r["epe_unpruned"]) for row in rows]
            gt = m == "gt"
            # green = the ground-truth network, black = the learned circuits (one line per fold)
            ax.plot(x, y, color="#4daf4a" if gt else "black", lw=2.2 if gt else 0.7, marker="o",
                    ms=3 if gt else 1.5, zorder=3 if not gt else 2,
                    label="flyvis 000" if gt else ("GNN, 5 folds" if m == "cv00" else None))
        keep = next(iter(res.values()))["keep_fraction"]
        ax.axhline(keep, ls="--", lw=0.8, color="0.45")
        ax.axhline(0.0, ls=":", lw=0.8, color="0.45")
        ax.set_ylim(-0.3, 1.15)
        ax.set_xlabel(xlab[a], color="0.15")
        ax.set_ylabel("task gain kept", color="0.15")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("0.45")
        ax.text(0.0, 1.04, f"{letter}  {titles[a]}", transform=ax.transAxes, va="bottom", ha="left", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[2].text(0.02, keep + 0.02, f"bound: {keep:.1%} of the gain kept (5.4 for 000)",
                 transform=axes[2].get_yaxis_transform(), ha="left", va="bottom", fontsize=7, color="0.45")
    axes[2].text(0.02, 0.02, "zero-flow prediction", transform=axes[2].get_yaxis_transform(),
                 ha="left", va="bottom", fontsize=7, color="0.45")
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=200)
    print("wrote", FIG)

    def kept(r, a):
        k = r["analyses"][a]["kept"]
        return (f"{k['edges_removed']:.1%} / {k['neurons_removed']:.1%} / {k['epe']:.3f}" if k else "none")
    lines = ["| model | EPE unpruned | task gain vs 000 | bound | edges: % edges / % neurons / EPE | neurons "
             "| both, same rung | union of the kept sets | strengths < 0 |", "|---|---|---|---|---|---|---|---|---|"]
    for m, r in res.items():
        u = r.get("union_of_kept")
        us = (f"{u['edges_removed']:.1%} / {u['neurons_removed']:.1%} / {u['epe']:.3f}"
              f"{'' if u['within_bound'] else ' (over)'}") if u else "--"
        neg = f"{r['negative_strengths']} of {r['n_strengths']}" if "negative_strengths" in r else "--"
        lines.append(f"| {'flyvis 000' if m == 'gt' else m} | {r['epe_unpruned']:.4f} | {r['gain_vs_000']:.0%} | "
                     f"{r['epe_bound']:.4f} | "
                     f"{kept(r, 'edges')} | {kept(r, 'neurons')} | {kept(r, 'both')} | {us} | {neg} |")
    cliff = [f"- {'flyvis 000' if m == 'gt' else m}: " + ", ".join(f"{t} {c}" for t, c in r["cliff_types"].items())
             for m, r in res.items() if r.get("cliff_types")]
    block = ("<!-- PRUNING:BEGIN -->\n\n## Results (written by `tools/task_pruning.py --report`)\n\n"
             "Largest kept rung per analysis: % of the 1,513,231 edges removed / % of the non-photoreceptor "
             "neurons removed / held-out EPE. Task gain = the EPE saved over predicting zero flow (" +
             f"{next(iter(res.values()))['epe_zero']:.4f}" + "); bound = keep " +
             f"{next(iter(res.values()))['keep_fraction']:.1%}" + " of the model's own gain, 5.4 for 000.\n\n"
             + "\n".join(lines) + "\n\nCell types of the neurons between the neuron ladder's kept rung and its "
             "first rejected rung (count of neurons):\n\n" + "\n".join(cliff) + "\n\n<!-- PRUNING:END -->")
    md = open(MD).read()
    if "<!-- PRUNING:BEGIN -->" in md:
        md = md[:md.index("<!-- PRUNING:BEGIN -->")] + block + md[md.index("<!-- PRUNING:END -->") + len("<!-- PRUNING:END -->"):]
    else:
        md = md.replace("<!-- STATUS:BEGIN -->", block + "\n\n<!-- STATUS:BEGIN -->", 1)
    open(MD, "w").write(md)
    print("wrote the results table into", os.path.normpath(MD))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="store_true", help="figure + results table from the pruning.json files")
    ap.add_argument("--model", default="gt", help="gt, or a fold of experiment 2 (cv00..cv04)")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epe-bound", type=float, default=EPE_BOUND,
                    help=f"the EPE bound for 000 (default {EPE_BOUND}); every model keeps the same fraction of its own task gain")
    ap.add_argument("--out", default=OUT, help="output folder (pruning.json under <out>/<model>/)")
    ap.add_argument("--mask-out", default=None,
                    help="also write the silent edge/neuron masks at the kept rungs, mapped onto the 217-column "
                         "(extent 8) network, to this .npz")
    args = ap.parse_args()
    if args.report:
        return report()
    torch.cuda.set_device(torch.device(args.device))
    if args.gates:
        raise SystemExit(0 if gates(args.device) else 1)
    nv, net, decoder, task = build(args.device)
    info = {"model": args.model, "network": GT_NAME}
    if args.model != "gt":
        run = RUN.format(fold=args.model)
        lp = torch.load(run + "/results/learned_ode_params.pt", map_location="cpu", weights_only=False)
        sh8 = net8_shared()
        s, t, b = fit_shared(lp["W"].numpy(), lp["tau_i"].numpy(), lp["V_i_rest"].numpy(), sh8)
        info.update(run=run, negative_strengths=int((s < 0).sum()), n_strengths=len(s),
                    r_strength=float(np.corrcoef(s, sh8["strength"])[0, 1]),
                    r_time_const=float(np.corrcoef(t, sh8["time_const"])[0, 1]),
                    r_bias=float(np.corrcoef(b, sh8["bias"])[0, 1]))
        set_shared(net, np.clip(s, 0, None), t, b)
        print(f"{args.model}: {info['negative_strengths']} of {len(s)} strengths negative, clamped to 0; "
              f"r(strength) {info['r_strength']:.3f}  r(time_const) {info['r_time_const']:.3f}  r(bias) {info['r_bias']:.3f}")
    info["epe_unpruned"] = e0 = evaluate(net, decoder, task)
    info["epe_zero"] = z = zero_epe(task)
    info["epe_bound_000"] = args.epe_bound
    info["keep_fraction"] = keep = (z - args.epe_bound) / (z - GT_STORED_EPE)
    info["gain_vs_000"] = (z - e0) / (z - GT_STORED_EPE)
    info["epe_bound"] = bound = z - keep * (z - e0)
    print(f"{args.model}: held-out EPE unpruned {e0:.4f}, zero prediction {z:.4f}, task gain {info['gain_vs_000']:.1%} "
          f"of 000's, bound {bound:.4f} (keeps {keep:.1%} of the gain)", flush=True)
    s_out = silence(net, task)
    info["neurons_exactly_silent"] = float((s_out == 0).mean())
    info["analyses"] = an = {a: ladder(net, decoder, task, s_out, a, bound) for a in ("edges", "neurons", "both")}
    # both, each at its own kept rung: the union of the two kept sets
    if an["edges"]["kept"] and an["neurons"]["kept"]:
        qe, qn = an["edges"]["kept"]["q"], an["neurons"]["kept"]["q"]
        me, _, _ = prune_sets(net, s_out, qe)["edges"]
        mn, _, fn = prune_sets(net, s_out, qn)["neurons"]
        m = me & mn
        e = evaluate(net, decoder, task, m)
        info["union_of_kept"] = {"edges_q": qe, "neurons_q": qn, "edges_removed": float(1 - m.mean()),
                                 "neurons_removed": float(fn), "epe": e, "within_bound": e <= bound}
        print(f"  union of the kept sets: edges q {qe:.3f} + neurons q {qn:.3f}: edges removed {1 - m.mean():.1%}, "
              f"neurons removed {fn:.1%}, EPE {e:.4f}", flush=True)
    # the cell types first removed past the neuron ladder's kept rung
    if an["neurons"]["kept"] and an["neurons"]["first_rejected_q"] is not None:
        types = np.asarray(net.connectome.nodes.type[:]).astype(str)
        cand = ~np.isin(types, list(INPUT_TYPES))
        idx = np.flatnonzero(cand)
        order = idx[np.argsort(s_out[idx], kind="stable")]
        a, b = (int(round(q * len(idx))) for q in (an["neurons"]["kept"]["q"], an["neurons"]["first_rejected_q"]))
        tt, cc = np.unique(types[order[a:b]], return_counts=True)
        info["cliff_types"] = {str(t): int(c) for t, c in sorted(zip(tt, cc), key=lambda x: -x[1])}
        print("  cell types removed at the neuron cliff:", info["cliff_types"], flush=True)
    if args.mask_out:
        save_masks(net, s_out, an, info, args.mask_out)
    os.makedirs(os.path.join(args.out, args.model), exist_ok=True)
    out = os.path.join(args.out, args.model, "pruning.json")
    json.dump(info, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
