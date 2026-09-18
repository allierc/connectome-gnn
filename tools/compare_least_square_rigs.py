#!/usr/bin/env python
"""Fit the per-edge template under three sampling rigs and compare what it recovers.

The readout's constants come from a least squares over (v_i, v_j) samples, so
WHICH samples are drawn is part of the estimator. This runs the same three-column
fit on one checkpoint under three rigs and writes the W_ij and E_ij panels for
each side by side.

    frames        NOMINAL, exactly what production does: real co-occurring
                  (v_i(t), v_j(t)) pairs from the trajectory, frames chosen by
                  choose_active_frames -- a uniform base draw topped up per
                  presynaptic cell. Nothing here can change it; this script
                  imports the production sampler rather than reimplementing it.
    grid_minmax   v_i and v_j drawn INDEPENDENTLY and uniformly over each
                  neuron's own observed [min, max].
    grid_zeromax  the same over [0, max].

THE TWO GRID RIGS ARE NOT FRAME CHOICES, and the difference is the point. The
nominal draws pairs the network actually visits; a grid draws pairs it may never
see. Connected cells are correlated, so the region of the (v_i, v_j) plane the
data occupy is far from the rectangle spanned by their marginals. A grid
therefore conditions the fit much better -- v_i sweeps its full range on every
edge, so the u*v_i column is well separated from u and b_2 is identified -- while
evaluating g_phi off the distribution it was trained on. Which of those dominates
is the empirical question this script exists to answer.

Usage:
    python tools/compare_least_square_rigs.py [RUN] [--neuron 2895] [--frames 256]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
from connectome_gnn.generators.ode_params import load_ode_params_for_run  # noqa: E402
from connectome_gnn.metrics import (  # noqa: E402
    G_PHI_EVAL_CHUNK,
    choose_active_frames,
    is_conductance_gnn,
    pad_g_phi_input,
    sample_g_phi_vi_vj_observed,
)
from connectome_gnn.models.registry import create_model  # noqa: E402
from connectome_gnn.plot import INDEX_TO_NAME, plot_recovery_panels  # noqa: E402
from connectome_gnn.utils import (  # noqa: E402
    graphs_data_path,
    migrate_state_dict,
    set_data_root,
    to_numpy,
)
from connectome_gnn.zarr_io import load_simulation_data  # noqa: E402

LOG = "/groups/saalfeld/home/allierc/GraphData/log/fly"
RIGS = ("frames", "grid_minmax", "grid_zeromax")


def _eval_g_phi(model, config, vi, vj, src, dst, device):
    """g_phi at arbitrary (v_i, v_j) pairs, chunked exactly as the sampler does.

    Same input layout and same g_phi_positive handling as
    sample_g_phi_vi_vj_observed, so the only thing that differs between rigs is
    where the pairs came from.
    """
    n_e, n_f = vi.shape
    emb = model.a.shape[1]
    ai = model.a[dst].unsqueeze(1).expand(-1, n_f, -1).reshape(-1, emb)
    aj = model.a[src].unsqueeze(1).expand(-1, n_f, -1).reshape(-1, emb)
    vi_f, vj_f = vi.reshape(-1, 1), vj.reshape(-1, 1)
    cond = is_conductance_gnn(config.graph_model.signal_model_name)
    outs = []
    with torch.no_grad():
        for lo in range(0, vj_f.shape[0], G_PHI_EVAL_CHUNK):
            hi = lo + G_PHI_EVAL_CHUNK
            parts = [vj_f[lo:hi], aj[lo:hi]] + ([vi_f[lo:hi], ai[lo:hi]] if cond else [])
            o = model.g_phi(pad_g_phi_input(torch.cat(parts, dim=1).float(), model))
            if config.graph_model.g_phi_positive:
                o = o ** 2
            outs.append(o)
    return torch.cat(outs, dim=0).reshape(n_e, n_f)


def draw(rig, model, config, ode_params, edges, x_ts, device, n_frames, seed=0):
    """(vi, vj, msg, edge_idx) under one rig. `msg` is W * g_phi, the edge message."""
    n_edges = int(edges.shape[1])
    if rig == "frames":
        # The production path, imported not reimplemented.
        probe = sample_g_phi_vi_vj_observed(model, config, edges, x_ts,
                                            n_edges=n_edges, n_frames=64, seed=seed)
        u0 = np.asarray(ode_params.gt_g_phi_func(probe["vj"].astype(np.float64)))
        pos = u0[u0 > 0]
        floor = float(np.quantile(pos, 0.5)) if pos.size else 0.0
        idx = choose_active_frames(x_ts, to_numpy(edges).reshape(2, -1)[0], floor,
                                   base=n_frames, per_neuron=8,
                                   max_frames=max(4 * n_frames, n_frames), seed=seed)
        res = sample_g_phi_vi_vj_observed(model, config, edges, x_ts, n_edges=n_edges,
                                          n_frames=n_frames, seed=seed, frame_idx=idx)
        vi, vj, g = res["vi"], res["vj"], res["g_phi"]
        eid = res["edge_idx"]
    else:
        # Independent uniform draws over each neuron's OWN observed range.
        volt = to_numpy(x_ts.voltage)                       # (T, N)
        lo_n, hi_n = volt.min(axis=0), volt.max(axis=0)
        if rig == "grid_zeromax":
            # A neuron whose voltage never rises above zero has an empty [0, max]
            # range; clamping to lo collapses it to the single point 0 rather
            # than inverting the interval. Those neurons contribute u = relu = 0
            # rows, which the floor discards anyway.
            lo_n = np.zeros_like(lo_n)
            hi_n = np.maximum(hi_n, lo_n)
        e = to_numpy(edges).reshape(2, -1)
        src_i, dst_i = e[0], e[1]
        rng = np.random.default_rng(seed)
        shape = (n_edges, n_frames)
        vj = rng.uniform(lo_n[src_i, None], hi_n[src_i, None], shape)
        vi = rng.uniform(lo_n[dst_i, None], hi_n[dst_i, None], shape)
        g = to_numpy(_eval_g_phi(
            model, config,
            torch.as_tensor(vi, dtype=torch.float32, device=device),
            torch.as_tensor(vj, dtype=torch.float32, device=device),
            torch.as_tensor(src_i, device=device).long(),
            torch.as_tensor(dst_i, device=device).long(), device))
        eid = np.arange(n_edges)
    from connectome_gnn.metrics import get_model_W
    W = to_numpy(get_model_W(model)).ravel()[eid]
    return (vi.astype(np.float64), vj.astype(np.float64),
            W[:, None] * g.astype(np.float64), eid)


def fit(vi, vj, msg, ode_params, min_points=8, t_slope=3.0):
    """The three-column fit, per edge: msg = b1*u + b2*(u*vi) + b3."""
    u = np.asarray(ode_params.gt_g_phi_func(vj), dtype=np.float64).reshape(vj.shape)
    pos = u[u > 0]
    floor = float(np.quantile(pos, 0.5)) if pos.size else 0.0
    keep = u > max(floor, 1e-6)
    y = np.where(keep, msg, 0.0)
    a1 = np.where(keep, u, 0.0)
    a2 = np.where(keep, u * vi, 0.0)
    a3 = keep.astype(np.float64)
    S = dict(n=keep.sum(1).astype(np.float64),
             S11=(a1 * a1).sum(1), S12=(a1 * a2).sum(1), S22=(a2 * a2).sum(1),
             S13=(a1 * a3).sum(1), S23=(a2 * a3).sum(1), S33=keep.sum(1).astype(np.float64),
             S1y=(a1 * y).sum(1), S2y=(a2 * y).sum(1), S3y=(a3 * y).sum(1),
             Syy=(y * y).sum(1))
    M = np.stack([np.stack([S["S11"], S["S12"], S["S13"]], -1),
                  np.stack([S["S12"], S["S22"], S["S23"]], -1),
                  np.stack([S["S13"], S["S23"], S["S33"]], -1)], -2)
    r = np.stack([S["S1y"], S["S2y"], S["S3y"]], -1)
    det = np.linalg.det(M)
    ok = (S["n"] >= min_points) & np.isfinite(det) & (np.abs(det) > 1e-18)
    b = np.full(r.shape, np.nan)
    if ok.any():
        b[ok] = np.linalg.solve(M[ok], r[ok][:, :, None])[:, :, 0]
    b1, b2 = b[:, 0], b[:, 1]
    ss = S["Syy"] - b1 * S["S1y"] - b2 * S["S2y"] - b[:, 2] * S["S3y"]
    sig2 = np.where(S["n"] > 3, ss / np.maximum(S["n"] - 3, 1), np.nan)
    v22 = np.full(len(b1), np.nan)
    if ok.any():
        v22[ok] = np.linalg.inv(M[ok])[:, 1, 1]
    se = np.sqrt(np.maximum(sig2 * v22, 0.0))
    t = np.where(se > 0, np.abs(b2) / se, np.nan)
    W = -b2
    E = np.where((np.abs(W) > 1e-12) & (t >= t_slope), b1 / W, np.nan)
    r2 = np.where(ok & (S["Syy"] > 0), 1.0 - ss / S["Syy"], np.nan)
    return W, E, r2, S["n"], ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="?", default="flyvis_flowcond_noise_005_gnn_nosq_cv00")
    ap.add_argument("--neuron", type=int, default=2895)
    ap.add_argument("--frames", type=int, default=256)
    a = ap.parse_args(argv)

    set_data_root("/groups/saalfeld/home/allierc/GraphData")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = NeuralGraphConfig.from_yaml(f"config/fly/{a.run}.yaml")
    cks = sorted(glob.glob(f"{LOG}/{a.run}/models/*graphs_0_*.pt"),
                 key=lambda f: int(re.findall(r"_(\d+)\.pt$", f)[0]))
    if not cks:
        print(f"{a.run}: no checkpoint"); return 1
    it = int(re.findall(r"_(\d+)\.pt$", cks[-1])[0])
    sd = torch.load(cks[-1], map_location=dev, weights_only=False)
    migrate_state_dict(sd)
    cfg.simulation.n_edges = sd["model_state_dict"]["W"].shape[0]
    cfg.simulation.n_extra_null_edges = 0
    model = create_model(cfg.graph_model.signal_model_name,
                         aggr_type=cfg.graph_model.aggr_type, config=cfg, device=dev)
    model.load_state_dict(sd["model_state_dict"], strict=False)
    model.eval()
    cfg.dataset = "fly/" + cfg.dataset
    op = load_ode_params_for_run(cfg, device=dev)
    x_ts = load_simulation_data(graphs_data_path(cfg.dataset, "x_list_train")).truncate_frames(4000)
    edges = torch.load(f"{LOG}/{a.run}/training_edges.pt", map_location=dev, weights_only=False)
    model.edges = edges

    out = os.path.join(LOG, a.run, "comparison_least_square")
    os.makedirs(out, exist_ok=True)
    n_neurons = cfg.simulation.n_neurons
    gt_W = np.asarray(op.effective_true_weights(
        to_numpy(op.W), to_numpy(edges), n_neurons))
    gt_E = np.asarray(to_numpy(op.reversal_per_edge())).ravel()
    types = to_numpy(x_ts.neuron_type).ravel().astype(int) if hasattr(x_ts, "neuron_type") else None
    dst = to_numpy(edges).reshape(2, -1)[1]

    lines = [f"run {a.run}   checkpoint {it}   frames {a.frames}", ""]
    for rig in RIGS:
        vi, vj, msg, eid = draw(rig, model, cfg, op, edges, x_ts, dev, a.frames)
        W, E, r2, n, ok = fit(vi, vj, msg, op)
        tW, tE = gt_W[eid], gt_E[eid]
        fin = np.isfinite(W) & np.isfinite(tW)
        gain = (float(np.dot(tW[fin], W[fin]) / np.dot(tW[fin], tW[fin]))
                if fin.any() else float("nan"))
        efin = np.isfinite(E) & np.isfinite(tE)
        grp = types[dst[eid]] if types is not None else None
        plot_recovery_panels(tW, np.abs(W), os.path.join(out, f"Wij_{rig}.png"),
                             symbol="|W_{ij}|", groups=grp, group_names=INDEX_TO_NAME,
                             outlier_threshold=1.0, violin_log_y=True)
        plot_recovery_panels(tE, E, os.path.join(out, f"Eij_{rig}.png"),
                             symbol="E_{ij}", groups=grp, group_names=INDEX_TO_NAME,
                             outlier_threshold=5.0, scatter_ylim=(-10.0, 10.0))
        lines.append(
            f"{rig:13s} fitted {int(ok.sum()):7d}/{len(W)}  E identified {int(efin.sum()):7d}"
            f"  |W| gain {gain:7.3f}  median fit R2 {np.nanmedian(r2):6.3f}"
            f"  W<0 {100 * np.mean(W[fin] < 0):5.1f}%")
        # the panel's per-synapse table, for one neuron
        rows = np.where(dst[eid] == a.neuron)[0]
        order = rows[np.argsort(-np.abs(tW[rows]))][:8]
        lines.append(f"    neuron {a.neuron}: " + "  ".join(
            f"[W {tW[i]:.3f}->{W[i]:+.3f} | E {tE[i]:+.2f}->{E[i]:+.2f}]" for i in order))
        lines.append("")
    txt = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")
    print(txt)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
