#!/usr/bin/env python
"""Run the production readout under three sampling rigs and compare what it recovers.

Which (v_i, v_j) samples the per-edge fit sees is part of the estimator, so this
re-runs the SAME readout on one checkpoint under three rigs and writes each one's
figures side by side in <run>/comparison_least_square.

    frames        the nominal: real co-occurring (v_i(t), v_j(t)) pairs from the
                  trajectory, frames chosen by choose_active_frames.
    grid_minmax   v_i and v_j drawn independently and uniformly over each
                  neuron's own observed [min, max].
    grid_zeromax  v_j over [0, max], v_i still over [min, max]. relu(v_j) is zero
                  below zero, so rows with v_j < 0 carry no drive; v_i keeps its
                  range because it is the u*v_i column that identifies b_2, and
                  3.3% of these neurons never rise above zero.

NOTHING IS REIMPLEMENTED HERE. An earlier version carried its own copy of the
three-column solve and reported W four times too large, because that copy omitted
the gauge correction W <- k_i * W_fit which extract_template_params applies --
numbers that disagreed with tmp_training/Wij for a reason having nothing to do
with the rigs. Now the rig chooses the samples and everything after that is the
production path: extract_recovered_params, extract_template_params,
score_recovery, _plot_recovered_scatter and analyse_neurons, the same calls the
trainer and `-o plot` make. The figures and the numbers are comparable to a run's
own by construction.

The grid rigs are injected by substituting the sampler extract_template_params
calls, which is the only way to change the samples without editing it.

Usage:
    python tools/compare_least_square_rigs.py [RUN] [--neuron 2895] [--frames 256]
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import logging
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from connectome_gnn import metrics as M  # noqa: E402
from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
from connectome_gnn.generators.ode_params import load_ode_params_for_run  # noqa: E402
from connectome_gnn.models.registry import create_model  # noqa: E402
from connectome_gnn.models.training_utils import init_training_data  # noqa: E402
from connectome_gnn.neuron_panels import analyse_neurons  # noqa: E402
from connectome_gnn.recovery_figures import _plot_recovered_scatter  # noqa: E402
from connectome_gnn.utils import migrate_state_dict, set_data_root, to_numpy  # noqa: E402

LOG = "/groups/saalfeld/home/allierc/GraphData/log/fly"
RIGS = ("frames", "grid_minmax", "grid_zeromax")


def _g_phi_at(model, config, vi, vj, src, dst):
    """g_phi at arbitrary (v_i, v_j), in metrics' own layout and chunk size.

    Kept here rather than in metrics.py so the production module is untouched.
    Same input order and same g_phi_positive handling as the real sampler, so a
    grid sample and a frame sample differ only in where the pairs came from.
    """
    n_e, n_f = vi.shape
    emb = model.a.shape[1]
    ai = model.a[dst].unsqueeze(1).expand(-1, n_f, -1).reshape(-1, emb)
    aj = model.a[src].unsqueeze(1).expand(-1, n_f, -1).reshape(-1, emb)
    vi_f, vj_f = vi.reshape(-1, 1), vj.reshape(-1, 1)
    cond = M.is_conductance_gnn(config.graph_model.signal_model_name)
    outs = []
    with torch.no_grad():
        for lo in range(0, vj_f.shape[0], M.G_PHI_EVAL_CHUNK):
            hi = lo + M.G_PHI_EVAL_CHUNK
            parts = [vj_f[lo:hi], aj[lo:hi]] + ([vi_f[lo:hi], ai[lo:hi]] if cond else [])
            o = model.g_phi(M.pad_g_phi_input(torch.cat(parts, dim=1).float(), model))
            if config.graph_model.g_phi_positive:
                o = o ** 2
            outs.append(o)
    return torch.cat(outs, dim=0).reshape(n_e, n_f)


@contextlib.contextmanager
def rig_sampler(rig, x_ts, device):
    """Substitute the (v_i, v_j) sampler for the duration of one readout.

    `frames` yields unchanged, so the nominal runs exactly as it does in
    training. The grid rigs replace vi and vj with independent uniform draws over
    each neuron's own observed range and re-evaluate g_phi there, leaving the
    returned dict's keys, shapes and edge ordering identical so everything
    downstream is none the wiser.
    """
    if rig == "frames":
        yield
        return
    original = M.sample_g_phi_vi_vj_observed
    volt = to_numpy(x_ts.voltage)
    lo_n, hi_n = volt.min(axis=0), volt.max(axis=0)

    def patched(model, config, edges, x_ts_, n_edges=16, n_frames=2000, seed=0,
                frame_idx=None):
        res = original(model, config, edges, x_ts_, n_edges=n_edges,
                       n_frames=n_frames, seed=seed, frame_idx=frame_idx)
        n_e, n_f = res["vi"].shape
        dst, src = res["edge_ij"][:, 0], res["edge_ij"][:, 1]
        rng = np.random.default_rng(seed + 991)
        lo_j = np.zeros_like(lo_n) if rig == "grid_zeromax" else lo_n
        hi_j = np.maximum(hi_n, lo_j)
        vj = rng.uniform(lo_j[src, None], hi_j[src, None], (n_e, n_f))
        vi = rng.uniform(lo_n[dst, None], hi_n[dst, None], (n_e, n_f))
        g = _g_phi_at(model, config,
                      torch.as_tensor(vi, dtype=torch.float32, device=device),
                      torch.as_tensor(vj, dtype=torch.float32, device=device),
                      torch.as_tensor(src, device=device).long(),
                      torch.as_tensor(dst, device=device).long())
        res["vi"], res["vj"], res["g_phi"] = vi, vj, to_numpy(g)
        return res

    M.sample_g_phi_vi_vj_observed = patched
    try:
        yield
    finally:
        M.sample_g_phi_vi_vj_observed = original


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="?", default="flyvis_flowcond_noise_005_gnn_nosq_cv00")
    ap.add_argument("--neuron", type=int, default=2895)
    ap.add_argument("--frames", type=int, default=256)
    a = ap.parse_args(argv)

    set_data_root("/groups/saalfeld/home/allierc/GraphData")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = os.path.join(LOG, a.run)
    cfg = NeuralGraphConfig.from_yaml(f"config/fly/{a.run}.yaml")
    cks = sorted(glob.glob(f"{log_dir}/models/*graphs_0_*.pt"),
                 key=lambda f: int(re.findall(r"_(\d+)\.pt$", f)[0]))
    if not cks:
        print(f"{a.run}: no checkpoint")
        return 1
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
    data = init_training_data(cfg, dev, log_dir, logging.getLogger("rigs"))
    model.edges = data.edges
    x_ts = data.x_list[0] if hasattr(data, "x_list") else data.x_ts
    n_neurons = cfg.simulation.n_neurons

    out = os.path.join(log_dir, "comparison_least_square")
    os.makedirs(out, exist_ok=True)
    lines = [f"run {a.run}   checkpoint {it}   frames {a.frames}",
             "score_recovery's own numbers, on the readout the trainer and "
             "-o plot use", ""]

    for rig in RIGS:
        with rig_sampler(rig, x_ts, dev):
            rec = M.extract_recovered_params(
                model, op, cfg, edges=data.edges, x_ts=x_ts, device=dev,
                n_neurons=n_neurons, need=("W", "tau", "V_rest", "E_ij", "msg_i"))
            rec = M.extract_template_params(
                model, op, config=cfg, edges=data.edges, x_ts=x_ts, device=dev,
                n_neurons=n_neurons, base=rec, n_frames=a.frames)
            scored = M.score_recovery(rec, cfg)
            for q, key in (("W", "Wij"), ("E_ij", "Eij")):
                try:
                    _plot_recovered_scatter(
                        rec, scored, q, out, config=cfg,
                        out_path=os.path.join(out, f"{key}_{rig}.png"))
                except Exception as exc:
                    lines.append(f"    {key} figure skipped: {type(exc).__name__}: {exc}")
            try:
                analyse_neurons(cfg, model, data, log_dir, device=dev, out_dir=out,
                                tag=rig, sr_enabled=False, use_rollout=False, quiet=True)
            except Exception as exc:
                lines.append(f"    panel skipped: {type(exc).__name__}: {exc}")
        g = scored.get
        nan = float("nan")
        lines.append(
            f"{rig:13s} Wij_R2 {g('Wij_R2', nan):+7.3f}  gain {g('Wij_gain', nan):6.3f}"
            f"  pearson {g('Wij_pearson', nan):+6.3f}  |  "
            f"Eij_R2 {g('Eij_R2', nan):+8.3f}  n {int(g('Eij_n', 0) or 0):7d}"
            f"  wrong {g('Eij_pct_wrong_slope', nan):5.1f}%  |  "
            f"msg_form_r2 {g('msg_form_r2_median', nan):.3f}")
    txt = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")
    print(txt)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
