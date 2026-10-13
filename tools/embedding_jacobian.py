"""How the embedding a_i enters the two MLPs of a trained general-form GNN: Jacobians of f_theta
and g_phi with respect to their inputs, on real frames.

    python tools/embedding_jacobian.py [run] [--device cuda:1] [--frames 32]

f_theta(v_i, a_i0, a_i1, msg_i, I_i) -> dv_i/dt and g_phi(v_j, a_j0, a_j1, v_i, a_i0, a_i1) (squared
when g_phi_positive, the factor that multiplies W_ij in the message) are evaluated at the inputs the
model's own forward pass builds, on `--frames` frames drawn uniformly from the training trajectory,
g_phi on 20,000 random edges per frame. Printed:

  sensitivity  mean |d out / d x_k| * SD(x_k): how much the output moves over the spread x_k
               actually takes, comparable across inputs of different scales;
  constancy    for f_theta's a_i0, a_i1: the SD over a neuron's frames of d f / d a_ik, relative to
               its mean |d f / d a_ik|; near 0 = the coordinate shifts f by a constant (an offset),
               large = it changes how f depends on v, msg, I;
  affine fit   per neuron, f = alpha0 + alpha1 v + alpha2 msg + alpha3 I over the frames (the
               template readout's update fit); Spearman, over cell-type centres, of a_i0 and a_i1
               with the leak -alpha1, the resting level -alpha0/alpha1 and the message gain alpha2.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.registry import create_model
from connectome_gnn.utils import migrate_state_dict, sort_key
from connectome_gnn.zarr_io import load_simulation_data

LOG = "/groups/saalfeld/home/allierc/GraphData/log/fly"


def load(run, device):
    log_dir = f"{LOG}/{run}"
    config = NeuralGraphConfig.from_yaml(f"{log_dir}/config.yaml")
    # the run's config names the dataset without its domain folder (GNN_Main adds it)
    x_ts = load_simulation_data(f"/groups/saalfeld/home/allierc/GraphData/graphs_data/fly/{config.dataset}/x_list_train",
                                fields=["index", "voltage", "stimulus", "neuron_type", "group_type"])
    edges = torch.load(f"{log_dir}/training_edges.pt", map_location=device, weights_only=False)
    files = sorted(glob.glob(f"{log_dir}/models/best_model_with_*"), key=sort_key)
    sd = torch.load(files[-1], map_location=device, weights_only=False)
    migrate_state_dict(sd)
    config.simulation.n_edges = sd["model_state_dict"]["W"].shape[0]
    config.simulation.n_extra_null_edges = 0
    model = create_model(config.graph_model.signal_model_name, aggr_type=config.graph_model.aggr_type,
                         config=config, device=device)
    model.load_state_dict(sd["model_state_dict"], strict=False)
    model.eval()
    return config, x_ts, edges, model, files[-1]


def jac(fn, x):
    """Per-row Jacobian of a row-wise function: (n, d) in, (n, 1) out -> (n, d)."""
    x = x.detach().clone().requires_grad_(True)
    y = fn(x)
    (g,) = torch.autograd.grad(y.sum(), x)
    return y.detach(), g.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="flyvis_noise_005_blank50_condl25_cv00")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--frames", type=int, default=32)
    args = ap.parse_args()
    dev = torch.device(args.device)
    config, x_ts, edges, model, ckpt = load(args.run, dev)
    core = getattr(model, "_orig_mod", model)
    n = x_ts.n_neurons
    emb = int(core.a.shape[1])
    assert emb == 2, emb
    rng = np.random.default_rng(0)
    frames = rng.choice(min(int(x_ts.n_frames), 64000), args.frames, replace=False)
    did = torch.zeros((n, 1), dtype=torch.int, device=dev)
    src, dst = edges[0], edges[1]
    sq = bool(getattr(core, "g_phi_positive", False))

    F_in, F_out, F_jac, G_in, G_jac = [], [], [], [], []
    for k in frames:
        st = x_ts.frame(int(k)).to(dev)
        with torch.no_grad():
            _pred, feats, _msg = core(st, edges, data_id=did, return_all=True)
        feats = feats[:n].float()
        y, g = jac(lambda x: core._run_mlp(core.f_theta, x), feats)
        F_in.append(feats.cpu().numpy()); F_out.append(y.cpu().numpy().ravel()); F_jac.append(g.cpu().numpy())
        e = torch.as_tensor(rng.choice(src.shape[0], 20000, replace=False), device=dev)
        v = feats[:, 0:1]; a = core.a[:n].detach().float()
        xg = torch.cat([v[src[e]], a[src[e]], v[dst[e]], a[dst[e]]], dim=1)
        if xg.shape[1] < core.g_phi.layers[0].weight.shape[1]:
            raise SystemExit("g_phi has extra (noise) input columns; not handled here")
        _y, gg = jac(lambda x: core._run_mlp(core.g_phi, x) ** (2 if sq else 1), xg)
        G_in.append(xg.cpu().numpy()); G_jac.append(gg.cpu().numpy())
    F_in, F_out, F_jac = np.stack(F_in), np.stack(F_out), np.stack(F_jac)       # (frames, n, 5)
    G_in, G_jac = np.concatenate(G_in), np.concatenate(G_jac)                     # (frames*20000, 6)

    out = {"run": args.run, "checkpoint": os.path.basename(ckpt), "frames": int(len(frames)),
           "g_phi_squared": sq}
    f_names = ["v_i", "a_i0", "a_i1", "msg_i", "I_i"]
    g_names = ["v_j", "a_j0", "a_j1", "v_i", "a_i0", "a_i1"]
    sd_f = F_in.reshape(-1, 5).std(0); sd_g = G_in.std(0)
    out["f_sensitivity"] = {nm: float(np.abs(F_jac[..., c]).mean() * sd_f[c]) for c, nm in enumerate(f_names)}
    out["g_sensitivity"] = {nm: float(np.abs(G_jac[:, c]).mean() * sd_g[c]) for c, nm in enumerate(g_names)}
    W0 = core.g_phi.layers[0].weight.detach().cpu().numpy()
    out["g_first_layer_column_norm"] = {nm: float(np.linalg.norm(W0[:, c])) for c, nm in enumerate(g_names)}
    for c, nm in ((1, "a_i0"), (2, "a_i1")):
        d = F_jac[..., c]                                                      # (frames, n)
        out[f"f_constancy_{nm}"] = float(np.median(d.std(0) / np.maximum(np.abs(d).mean(0), 1e-12)))

    # per-neuron affine fit of f over the frames
    X = np.stack([np.ones_like(F_in[..., 0]), F_in[..., 0], F_in[..., 3], F_in[..., 4]], -1)  # (frames, n, 4)
    alpha = np.stack([np.linalg.lstsq(X[:, i], F_out[:, i], rcond=None)[0] for i in range(n)])
    leak = -alpha[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        level = -alpha[:, 0] / alpha[:, 1]
    gain = alpha[:, 2]
    t = x_ts.neuron_type.cpu().numpy().astype(int).ravel()[:n]
    a = core.a[:n].detach().cpu().numpy()
    T = np.unique(t)
    med = lambda z: np.array([np.nanmedian(z[t == k]) for k in T])
    C0, C1 = med(a[:, 0]), med(a[:, 1])
    out["affine_vs_embedding_spearman"] = {
        f"{an} vs {qn}": float(spearmanr(ac, q, nan_policy="omit").statistic)
        for an, ac in (("a_i0", C0), ("a_i1", C1))
        for qn, q in (("leak -alpha1", med(leak)), ("level -alpha0/alpha1", med(level)), ("gain alpha2", med(gain)))}
    print(json.dumps(out, indent=1))
    # per-neuron arrays for the slide figure (tools/embedding_structure.py panels g, h)
    np.savez(f"{LOG}/{args.run}/results/extras/embedding_jacobian.npz", alpha=alpha,
             dfdv=F_jac[..., 0].mean(0), dfda0=F_jac[..., 1].mean(0), dfda1=F_jac[..., 2].mean(0),
             dfda1_sd=F_jac[..., 2].std(0), type_ids=t,
             g_names=np.array(g_names), g_sensitivity=np.array([out["g_sensitivity"][k] for k in g_names]),
             f_names=np.array(f_names), f_sensitivity=np.array([out["f_sensitivity"][k] for k in f_names]))
    res = f"{LOG}/{args.run}/results/extras/embedding_jacobian.json"
    json.dump(out, open(res, "w"), indent=1)
    print("wrote", res)


if __name__ == "__main__":
    main()
