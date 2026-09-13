"""Read a closed form out of a learned function with PySR 2, in three stages.

WHY A TEMPLATE AND NOT A FREE SEARCH. The per-edge message is identified only up
to a degeneracy: anything that does not vanish with presynaptic activity can be
moved between the message and the leak without changing dv/dt. Measured on the
trained conductance GNNs, 65-80% of the message error is exactly such a term, and
only 42-60% of the true message's variance is identifiable at all. A free search
would transcribe that contamination into a formula. Inside the family

    msg_ij = f(v_j) * (E - v_i)

it cannot be written at all: when f(v_j) is zero the product is zero. The
template is the gauge fix, moved out of the loss and into the search space.

THE STAGES, each with a known answer, so a failure is attributable.

  --source synthetic   The generator's own message, built here from the true
                       reversals: relu(v_j) * (E[cat] - v_i). No model, no data
                       loading. Tests the PySR mechanics alone: if the search
                       does not return relu and the true constants on exact
                       noiseless data, nothing downstream is trustworthy.
  --source knownode    The same functional form, but with the known-ODE
                       student's LEARNED reversals, on the real voltage
                       distribution. Tests the readout on a real model whose
                       parameters we can check one by one.
  --source gphi        The trained conductance GNN's g_phi: the real
                       measurement. Expect it to be limited by the line-fit
                       gate, which on these runs is 0.30 to 0.47.
  --source ftheta      The same GNN's f_theta, whose closed form is also known:
                       the generator integrates
                       dv/dt = (V_rest - v + msg + stim) / tau, so f_theta
                       should be LINEAR, with one 1/tau and one V_rest per cell
                       type. Template `T[cat] * (V[cat] - v_i + f(msg, stim))`,
                       where the answer for f is msg + stim.

TEMPLATES. `--template cat` hands the search the postsynaptic cell type and lets
it learn one constant per type (65 chloride reversals plus one shared cation
reversal): an oracle upper bound. `--template emb` replaces that with a
sub-expression over the learned embeddings, telling the search nothing about cell
types, so it also tests whether the GNN discovered them. `--free` drops the
template entirely, as the control showing what the structure buys.

Usage:
    python tools/sr_readout.py --source synthetic
    python tools/sr_readout.py --source gphi --log-dir <run> --template emb
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from connectome_gnn.utils import set_data_root, to_numpy  # noqa: E402

_root = os.environ.get("GNN_OUTPUT_ROOT")
if _root:
    set_data_root(_root)

SYNAPSE_FEATURES = ["v_j", "v_i", "a_i0", "a_i1", "a_j0", "a_j1", "cat"]
NEURON_FEATURES = ["v_i", "msg", "stim", "a_i0", "a_i1", "cat"]


def _r2(y, pred):
    y, pred = np.asarray(y).ravel(), np.asarray(pred).ravel()
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")


# ------------------------------------------------------------------ #
#  Stage 1: the generator's message, built here. No model, no dataset.
# ------------------------------------------------------------------ #

def table_synthetic(n_rows=20000, n_types=65, seed=0, reversals_csv=None):
    """relu(v_j) * (E[cat] - v_i) on plausible voltages, with the true reversals.

    Voltages are drawn over the range the real recording visits (about -7 to +7)
    and the reversals are read from a dataset's reversals.csv when one is given,
    so the constants the search has to find are the ones that matter rather than
    round numbers it could stumble on.
    """
    rng = np.random.default_rng(seed)
    if reversals_csv and os.path.isfile(reversals_csv):
        rows = [ln.split(",") for ln in open(reversals_csv).read().strip().split("\n")[1:]]
        rows = [r for r in rows if int(r[0]) >= 0]
        E_inh = np.array([float(r[4]) for r in rows])
        E_exc = float(rows[0][7])
        n_types = E_inh.size
    else:
        E_inh = rng.uniform(-6.0, -1.5, n_types)
        E_exc = 10.37
    E_by_cat = np.concatenate([E_inh, [E_exc]])          # 65 chloride + 1 cation

    cat = rng.integers(0, E_by_cat.size, n_rows)
    v_j = rng.uniform(-7.0, 7.0, n_rows)
    v_i = rng.uniform(-7.0, 7.0, n_rows)
    y = np.maximum(v_j, 0.0) * (E_by_cat[cat] - v_i)
    a = rng.normal(size=(n_rows, 4)) * 0.1
    X = np.column_stack([v_j, v_i, a[:, 0], a[:, 1], a[:, 2], a[:, 3], cat + 1])
    return X, y, {"n_categories": int(E_by_cat.size), "features": SYNAPSE_FEATURES,
                  "kind": "synapse", "label": "synthetic",
                  "truth": {"E": E_by_cat}}


# ------------------------------------------------------------------ #
#  Stages 2 and 3: from a trained run.
# ------------------------------------------------------------------ #

def _load_run(log_dir, device):
    from message_decomposition import _load
    return _load(log_dir, device)


def table_from_run(log_dir, source, n_rows=20000, n_frames=32, device="cpu", seed=0):
    config, data, model, ckpt = _load_run(log_dir, device)
    op, x_ts = data.ode_params, data.x_ts
    edges = data.edges.to(device)
    n = int(data.n_neurons)
    rng = np.random.default_rng(seed)
    core = getattr(model, "_orig_mod", model)

    type_index = getattr(core, "type_index", None)
    types = (to_numpy(type_index) if type_index is not None
             else to_numpy(data.type_list)).ravel().astype(int)[:n]
    n_types = int(types.max()) + 1

    frames = np.linspace(0, int(x_ts.n_frames) - 1, n_frames).astype(int)
    per_frame = max(1, n_rows // len(frames))
    data_id = torch.zeros((n, 1), dtype=torch.int, device=device)

    if source == "ftheta":
        emb = (core.a.detach() if getattr(core, "a", None) is not None
               else torch.zeros(n, 2, device=device))
        tau = np.asarray(op.gt_tau(n), dtype=np.float64)
        vrest = np.asarray(op.gt_vrest(n), dtype=np.float64)
        X, Y = [], []
        with torch.no_grad():
            for k in frames:
                state = x_ts.frame(int(k)).to(device)
                pred, feats, msg = model(state, edges, data_id=data_id, return_all=True)
                sel = rng.choice(n, min(per_frame, n), replace=False)
                sel_t = torch.as_tensor(sel, device=device)
                v = feats[sel_t, 0]
                m = feats[sel_t, 1 + int(core.a.shape[1])]
                s = feats[sel_t, 2 + int(core.a.shape[1])]
                X.append(np.column_stack([to_numpy(v), to_numpy(m), to_numpy(s),
                                          to_numpy(emb[sel_t]), types[sel] + 1]))
                Y.append(to_numpy(pred).ravel()[sel])
        X = np.concatenate(X).astype(np.float64)
        y = np.concatenate(Y).astype(np.float64)
        # f_theta = (V_rest - v + msg + stim) / tau, so the template's per-type
        # constants are T = 1/tau and V = V_rest.
        T_by_type = np.array([1.0 / tau[types == t].mean() if (types == t).any() else np.nan
                              for t in range(n_types)])
        V_by_type = np.array([vrest[types == t].mean() if (types == t).any() else np.nan
                              for t in range(n_types)])
        return X, y, {"n_categories": n_types, "features": NEURON_FEATURES,
                      "kind": "neuron", "label": os.path.basename(log_dir.rstrip("/")),
                      "checkpoint": ckpt,
                      "truth": {"T": T_by_type, "V": V_by_type}}

    # --- synapse: the per-edge message function ---
    W = torch.as_tensor(to_numpy(op.W).ravel(), dtype=torch.float32, device=device)
    E_edge = op.reversal_per_edge().to(device).float().ravel()
    is_inh = op.edge_is_inh.to(device).bool().ravel()
    src, dst = edges[0], edges[1]
    n_edges = int(src.numel())
    # A known-ODE student has no embedding; its message is analytic and the
    # embedding columns are unused by the category template, so they are zeros.
    emb = (core.a.detach() if getattr(core, "a", None) is not None
           else torch.zeros(n, 2, device=device))

    X, Y, ET = [], [], []
    with torch.no_grad():
        for k in frames:
            state = x_ts.frame(int(k)).to(device)
            v = state.voltage.float().ravel()[:n]
            sel = torch.as_tensor(rng.choice(n_edges, per_frame, replace=False), device=device)
            s_i, d_i = src[sel], dst[sel]
            vj, vi = v[s_i], v[d_i]
            if source == "knownode":
                # The generator's own message on the REAL voltage distribution.
                # The one question this stage answers that stage 1 cannot: is the
                # rectification identifiable when v_j is almost never negative?
                g = torch.relu(vj) * (E_edge[sel] - vi)
            else:                                   # gphi: the GNN's own function
                from connectome_gnn.models.utils import pad_g_phi_input
                feat = torch.cat([vj[:, None], emb[s_i], vi[:, None], emb[d_i]], dim=1)
                g = core.g_phi(pad_g_phi_input(feat, core)).ravel()
                if getattr(core, "g_phi_positive", False):
                    g = g ** 2
            cat = np.where(to_numpy(is_inh[sel]), types[to_numpy(d_i)], n_types)
            X.append(np.column_stack([to_numpy(vj), to_numpy(vi),
                                      to_numpy(emb[d_i]), to_numpy(emb[s_i]), cat + 1]))
            Y.append(to_numpy(g).ravel())
            ET.append(to_numpy(E_edge[sel]).ravel())
    X = np.concatenate(X).astype(np.float64)
    y = np.concatenate(Y).astype(np.float64)
    # The true constant behind each category, to score the recovered ones
    # against: the chloride reversal of a neuron of that type for categories
    # 0..n_types-1, and the one shared cation reversal for the last.
    E_inh_n = to_numpy(op.E_inh).ravel()[:n]
    E_exc_n = to_numpy(op.E_exc).ravel()[:n]
    E_by_cat = np.array([E_inh_n[types == t].mean() if (types == t).any() else np.nan
                         for t in range(n_types)] + [float(np.mean(E_exc_n))])
    return X, y, {"n_categories": n_types + 1, "features": SYNAPSE_FEATURES,
                  "kind": "synapse", "label": os.path.basename(log_dir.rstrip("/")),
                  "E_true_per_row": np.concatenate(ET), "checkpoint": ckpt,
                  "truth": {"E": E_by_cat}}


# ------------------------------------------------------------------ #
#  The search
# ------------------------------------------------------------------ #

def build_spec(info, template, free):
    from pysr import TemplateExpressionSpec
    if free:
        return None
    feats = info["features"]
    if info["kind"] == "neuron":
        # dv/dt = (V_rest - v + msg + stim) / tau, per cell type. The search has
        # to find that msg and stim enter additively with coefficient one.
        return TemplateExpressionSpec(
            combine="T[cat] * (V[cat] - v_i + f(msg, stim))",
            expressions=["f"],
            parameters={"T": info["n_categories"], "V": info["n_categories"]},
            variable_names=feats,
        )
    if template == "cat":
        return TemplateExpressionSpec(
            combine="f(v_j) * (E[cat] - v_i)",
            expressions=["f"],
            parameters={"E": info["n_categories"]},
            variable_names=feats,
        )
    return TemplateExpressionSpec(
        combine="f(v_j) * (g(a_i0, a_i1, a_j0, a_j1) - v_i)",
        expressions=["f", "g"],
        variable_names=feats,
    )


def _learned_parameters(equation: str) -> dict:
    """The per-category constants out of a template equation string.

    PySR prints them as `f = relu(#1); E = [-5.24, -3.76, ...]`. Parsed rather
    than read off the Julia object because the printed form is stable across the
    expression types and the object's layout is not.
    """
    out = {}
    for name, body in re.findall(r"(\w+)\s*=\s*\[([^\]]*)\]", equation):
        try:
            out[name] = np.array([float(x) for x in body.replace(";", ",").split(",")
                                  if x.strip()])
        except ValueError:
            pass
    return out


def _score_parameters(equation, truth, lines):
    """Compare every recovered constant vector with its known value."""
    got = _learned_parameters(str(equation))
    for name, true_v in (truth or {}).items():
        if name not in got:
            lines.append(f"{name}: not recovered")
            continue
        g, t = got[name], np.asarray(true_v, dtype=float)
        k = min(g.size, t.size)
        g, t = g[:k], t[:k]
        ok = np.isfinite(g) & np.isfinite(t)
        if ok.sum() < 2:
            continue
        err = np.abs(g[ok] - t[ok])
        lines.append(f"{name}: n={ok.sum()} R2={_r2(t[ok], g[ok]):.4f} "
                     f"max|err|={err.max():.4g} median|err|={np.median(err):.4g}")
    return lines


def run(X, y, info, template="cat", free=False, niterations=40, out_dir=None,
        guesses=True):
    from pysr import PySRRegressor
    spec = build_spec(info, template, free)
    kw = dict(
        niterations=niterations,
        operators={2: ["+", "-", "*", "/"], 1: ["relu", "exp", "tanh", "square"]},
        maxsize=20,
        progress=False,
        temp_equation_file=True,
        verbosity=0,
    )
    if spec is not None:
        kw["expression_spec"] = spec
        # A template guess names a sub-expression's arguments POSITIONALLY, as
        # #1, #2, ...; PySR rejects real variable names here.
        if guesses and info["kind"] == "synapse" and template == "cat":
            kw["guesses"] = [{"f": "relu(#1)"}]          # f(v_j) = relu(v_j)
        elif guesses and info["kind"] == "neuron":
            kw["guesses"] = [{"f": "#1 + #2"}]           # f(msg, stim) = msg + stim
    model = PySRRegressor(**kw)
    model.fit(X, y, variable_names=info["features"])
    pred = np.asarray(model.predict(X)).ravel()
    r2 = _r2(y, pred)
    best = model.get_best()
    equation = str(best["equation"])
    lines = [f"R2 {r2:.6f}", f"equation {equation}"]
    _score_parameters(equation, info.get("truth"), lines)
    print(f"\n=== {info['label']} | {info['kind']} | "
          f"{'free' if free else template} | R2 = {r2:.5f}")
    for ln in lines:
        print("   " + ln)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tag = f"{info['label']}_{info['kind']}_{'free' if free else template}"
        with open(os.path.join(out_dir, f"sr_{tag}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        np.savez_compressed(os.path.join(out_dir, f"sr_{tag}.npz"), X=X, y=y, pred=pred)
    return model, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=["synthetic", "knownode", "gphi", "ftheta"])
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--template", choices=["cat", "emb"], default="cat")
    ap.add_argument("--free", action="store_true")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--niterations", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reversals", default=None, help="a dataset's reversals.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.source == "synthetic":
        X, y, info = table_synthetic(args.rows, seed=0, reversals_csv=args.reversals)
    else:
        if not args.log_dir:
            ap.error(f"--source {args.source} needs --log-dir")
        X, y, info = table_from_run(args.log_dir, args.source, args.rows,
                                    args.frames, args.device)
    print(f"table: {X.shape[0]} rows x {X.shape[1]} features | target std {y.std():.4f} "
          f"| {info['n_categories']} categories")
    run(X, y, info, args.template, args.free, args.niterations, args.out)


if __name__ == "__main__":
    main()
