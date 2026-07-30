"""Observability control for the per-frame HD-decode failure.

The companion result (fig_zebrafish_calcium_baseline) shows a decoder cannot
read heading frame-by-frame from the REAL ZAPBench calcium. A referee asks:
is that the brain (no legible HD code) or the OBSERVABLE (GCaMP low-pass +
1.09 Hz aliasing of an ~8 s revolution + only 300/839 cells)? This control
settles it by running the SAME decoder on the MODEL --- which provably encodes
HD in its readout --- pushed through the SAME observation pipeline:

  voltage -> GCaMP convolution -> imaging-frame sampling -> 300 bump cells.

Three decodes of the Rotations block (held-out tail, circular correlation):
  * model readout      (from the trained 2-D (cos,sin) output of the latent
                        state) --- the model's own HD estimate, should be ~1.
  * model observable   (MLP decode of the model's GCaMP 300-cell calcium).
  * real observable    (MLP decode of the recorded 300-cell dF/F).

If model-observable ~ real-observable ~ readout, then the degraded observable
is the culprit, not the brain.

  python figures/zebrafish/fig_zebrafish_obs_control.py \
      --config zebrafish/zebrafish_hd_si_ipn12_v1_cv0
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "papers", "fishFuncEM"))

_PANEL_PY = os.path.join(_REPO, "figures", "zebrafish",
                         "zebrafish_functional_traces_panel.py")
_spec = importlib.util.spec_from_file_location("_ftp", _PANEL_PY)
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)


def _circ_r(a, b):
    """Centred circular correlation between two angle series (rad)."""
    a = a - np.angle(np.mean(np.exp(1j * a)))
    b = b - np.angle(np.mean(np.exp(1j * b)))
    return float(np.sum(np.sin(a) * np.sin(b)) /
                 np.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2) + 1e-9))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="zebrafish/zebrafish_hd_si_ipn12_v1_cv0")
    p.add_argument("--gcamp", default=None)
    p.add_argument("--connectome",
                   default=os.path.join(_REPO, "figures", "zebrafish",
                                        "zebrafish_connectome_HD_IPN12"))
    p.add_argument("--circuit", default="zebrafish_HD_IPN12_839_v1")
    p.add_argument("--fishfuncem-data",
                   default=os.path.join(_REPO, "papers", "fishFuncEM", "data"))
    p.add_argument("--warmup-s", type=float, default=10.0)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--display-s", type=float, default=300.0,
                   help="x-axis display crop for the heading time-series panel "
                        "(seconds; <=0 shows the full block). Correlations are "
                        "still computed on the full held-out tail.")
    p.add_argument("--out", default=os.path.join(_REPO, "figures", "zebrafish",
                                                 "fig_zebrafish_obs_control.png"))
    args = p.parse_args()

    import torch
    from connectome_gnn.config import NeuralGraphConfig
    from connectome_gnn.models.registry import create_model
    from connectome_gnn.models.gcamp import create_gcamp
    from connectome_gnn.generators.circuits import get_circuit
    from connectome_gnn.generators.zapbench_stimulus import heading_to_drive
    from connectome_gnn.plot_anatomy_voltage import run_task_rollout
    from connectome_gnn.utils import migrate_state_dict, set_data_root
    set_data_root(os.environ.get("GNN_OUTPUT_ROOT",
                                 "/groups/saalfeld/home/allierc/GraphData"))

    root = os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
    cfg_path = os.path.join(root, "config", args.config + ".yaml")
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(_REPO, "config", args.config + ".yaml")
    log_dir = cfg_path.replace(os.sep + "config" + os.sep, os.sep + "log" + os.sep)[:-5]
    cfg = NeuralGraphConfig.from_yaml(cfg_path)
    gcamp_name = args.gcamp or getattr(cfg.simulation, "gcamp_kernel", "gcamp7f")
    gm = create_gcamp(gcamp_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- fixed 300-neuron bump row set (rastermap order) ------------------ #
    rows, _ = panel.build_rows(args.connectome, args.circuit)
    panel_rows = panel.sort_rows_rastermap(
        rows[rows["matched"]].reset_index(drop=True))
    col_of_body = {int(b): i for i, b in enumerate(panel_rows["bodyId"].to_numpy())}

    # ---- REAL observable: recorded dF/F over the Rotations block ----------- #
    z = np.load(os.path.join(args.connectome, "functional",
                             "circuit_functional_traces.npz"), allow_pickle=True)
    traces = z["traces"]; col_of = {int(b): i for i, b in enumerate(z["bodyId"].astype(int))}
    theta_hr, ts, theta_frame = panel.rotation_headings(
        args.fishfuncem_data, args.connectome, 0.01)
    keep = ts < traces.shape[0]; ts = ts[keep]; theta_frame = theta_frame[: len(ts)]
    drive = heading_to_drive(theta_frame, dt=0.915, src_dt=0.915)
    theta_true = drive["theta"]                      # (n_frames,) rad
    n_frames = len(theta_true)

    real_kino = np.full((len(panel_rows), n_frames), np.nan, np.float32)
    for ri, b in enumerate(panel_rows["bodyId"].to_numpy()):
        if int(b) in col_of:
            real_kino[ri] = traces[ts, col_of[int(b)]]

    # ---- MODEL observable: voltage -> GCaMP -> imaging-sample -> 300 cells - #
    model = create_model(cfg.graph_model.signal_model_name,
                         aggr_type=cfg.graph_model.aggr_type,
                         config=cfg, device=device).to(device)
    ck = max(glob.glob(f"{log_dir}/models/best_model_with_*.pt"), key=os.path.getmtime)
    sd = torch.load(ck, map_location=device, weights_only=False)
    migrate_state_dict(sd); model.load_state_dict(sd["model_state_dict"], strict=False)
    model.eval(); dt = float(model.dt)
    cc = getattr(getattr(cfg, "circuit", None), "name", None) or args.circuit
    idx_of = {int(b): i for i, b in enumerate(
        np.asarray(get_circuit(cc).body_ids, dtype=np.int64))}

    pc = cfg.plotting.model_copy(update=dict(
        anatomy_voltage_pattern="zapbench_rotation",
        anatomy_voltage_warmup_s=args.warmup_s,
        anatomy_voltage_zapbench_connectome=args.connectome,
        anatomy_voltage_zapbench_fishfuncem_data=args.fishfuncem_data))
    h, y_hat, _th, _lab, extra = run_task_rollout(model, pc, device)
    warm, ss = extra["warm_steps"], extra["sample_steps"]
    nf = min(n_frames, len(ss))
    calcium = gm(h, dt_in=dt)[warm:][ss[:nf]]        # (nf, N) model observable
    readout = np.arctan2(y_hat[warm:, 1], y_hat[warm:, 0])[ss[:nf]]  # model readout

    model_kino = np.full((len(panel_rows), nf), np.nan, np.float32)
    for ri, b in enumerate(panel_rows["bodyId"].to_numpy()):
        mi = idx_of.get(int(b))
        if mi is not None and mi < calcium.shape[1]:
            model_kino[ri] = calcium[:, mi]

    # ---- three decodes, same MLP decoder, same train/test split ----------- #
    theta = theta_true[:nf]
    real_ok = np.where(np.isfinite(real_kino[:, 0]))[0]
    model_ok = np.where(np.isfinite(model_kino[:, 0]))[0]
    dec_real, n_tr, r_real = panel.mlp_decode_hd(
        np.nan_to_num(real_kino[real_ok, :nf].T), theta, train_frac=args.train_frac)
    dec_mobs, _, r_mobs = panel.mlp_decode_hd(
        np.nan_to_num(model_kino[model_ok].T), theta, train_frac=args.train_frac)
    r_read = _circ_r(readout[n_tr:], theta[n_tr:])
    split_t = n_tr * 0.915

    print(f"[obs-control] held-out circular r:  model readout = {r_read:+.3f}  "
          f"model observable = {r_mobs:+.3f}  real observable = {r_real:+.3f}")

    # ---- figure ----------------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(nf) * 0.915
    wrap = panel._wrap_hd_deg
    fig, (axT, axB) = plt.subplots(1, 2, figsize=(13, 4.2),
                                   gridspec_kw=dict(width_ratios=[2.4, 1]))
    axT.plot(t, wrap(theta), color=(0, 0.7, 0.25), lw=1.6, label="true heading")
    axT.plot(t, wrap(np.asarray(readout)), color="black", lw=1.0,
             label=f"model readout (r={r_read:.2f})")
    axT.plot(t, wrap(np.asarray(dec_mobs)), color="#d62728", lw=1.0,
             label=f"decode ← model observable (r={r_mobs:.2f})")
    axT.plot(t, wrap(np.asarray(dec_real)), color="#1f77b4", lw=1.0,
             label=f"decode ← real observable (r={r_real:.2f})")
    # The held-out split sits at train_frac of the full block; only mark it if
    # it falls inside the displayed window (else the crop hides it).
    if not (args.display_s and args.display_s > 0) or split_t <= args.display_s:
        axT.axvline(split_t, color="0.6", ls="--", lw=0.8)
        axT.text(split_t, 182, " held-out →", fontsize=8, color="0.5", va="bottom")
    axT.set_ylim(-185, 185); axT.set_yticks([-180, 0, 180])
    if args.display_s and args.display_s > 0:
        axT.set_xlim(0, args.display_s)
    axT.set_xlabel("time (s)"); axT.set_ylabel("heading (°)")
    axT.legend(fontsize=8, loc="lower right", framealpha=0.9)
    for sp in ("top", "right"):
        axT.spines[sp].set_visible(False)

    names = ["model\nreadout", "model\nobservable", "real\nobservable"]
    vals = [r_read, r_mobs, r_real]
    axB.bar(range(3), vals, color=["black", "#d62728", "#1f77b4"])
    axB.axhline(0, color="0.6", lw=0.6)
    axB.set_xticks(range(3)); axB.set_xticklabels(names, fontsize=8)
    axB.set_ylabel("held-out circular $r$"); axB.set_ylim(min(0, min(vals)) - 0.05, 1.05)
    for i, v in enumerate(vals):
        axB.text(i, v + (0.02 if v >= 0 else -0.06), f"{v:.2f}", ha="center",
                 fontsize=8)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)
    fig.tight_layout()
    from _despine import open_axes
    open_axes(fig)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
