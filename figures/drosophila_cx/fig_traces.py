"""Build the two voltage-trace 2x2 figures from a single entry point:

    - fig_traces_combined.png         (natural OU velocity stimulus)
    - fig_traces_const_omega.png      (constant omega = 60 deg/s, 15 s)

Each per-model panel uses the same layout as the pipeline's
``_cx_voltage_sanity_combined_plot``: stacked raw mean-subtracted
traces (5 neurons per cell type) with PEN L / R stim overlay on the
first PEN_a / PEN_b row of each cluster, and a wrapped HD GT-vs-decode
strip below. The four panels are rendered directly into one figure
(no PNG round-trip), so the output is vector-clean at high dpi.

Output:
    docs/figure/fig_traces_combined.png
    docs/figure/fig_traces_const_omega.png

CLI:
    python docs/figure/fig_traces.py
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from connectome_gnn.utils import log_path, load_data_root_from_json, set_data_root
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.models.registry import create_model
from connectome_gnn.generators.utils import generate_path_integration_batch
from connectome_gnn.task_state import TaskTrials
from connectome_gnn.generators.connconstr_data import load_drosophila_cx_connectome


MODELS = [
    ("drosophila_cx_pi_epg_no_tv_cv0",        "Known-ODE no-TV"),
    ("drosophila_cx_pi_epg_tv_cv0",           "Known-ODE $+$TV"),
    ("drosophila_cx_pi_gnn_epg_no_tv_cv0",    "GNN no-TV"),
    ("drosophila_cx_pi_gnn_epg_tv_cv0",       "GNN $+$TV"),
    ("drosophila_cx_pi_fc_epg_cv0",           "fully connected"),
    ("drosophila_cx_pi_frozen_Wrec_epg_cv0",  "frozen $W^{\\mathrm{rec}}$"),
]

# Pipeline constants (mirroring `_cx_voltage_sanity_combined_plot`).
_GT_COLOR   = "#7ec97e"
_PRED_COLOR = "black"
_GT_MS      = 2.6
_PRED_MS    = 1.5
_TICK_FS    = 8
_LABEL_FS   = 9
STIM_L_COLOR = "#f08080"
STIM_R_COLOR = "#87cefa"


def _load(config_name, device, prefer_epoch=None):
    config, _ = load_run_config(config_name, explicit_output_root=False, task="train")
    ckpt_dir = os.path.join(log_path(config.config_file), "models")
    cands = sorted(
        glob.glob(os.path.join(ckpt_dir, "best_model_with_0_graphs_*.pt")),
        key=lambda p_: int(p_.rsplit("_", 1)[1].rstrip(".pt")),
    )
    if not cands:
        raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
    if prefer_epoch is None and "gnn_tailloss" in config_name:
        prefer_epoch = 5
    elif prefer_epoch is None and "gnn_epg" in config_name:
        prefer_epoch = 3
    ckpt_path = cands[-1]
    if prefer_epoch is not None:
        match = [p_ for p_ in cands
                 if int(p_.rsplit("_", 1)[1].rstrip(".pt")) == prefer_epoch]
        if match:
            ckpt_path = match[0]
    model = create_model(
        config.graph_model.signal_model_name,
        aggr_type=config.graph_model.aggr_type,
        config=config, device=device,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"loaded {config_name}: {ckpt_path}")
    return model, config


def _build_const_batch(n_steps, dt, omega_deg, theta0, device):
    T = int(n_steps)
    omega = np.full((1, T), float(omega_deg), dtype=np.float32)
    omega_rad = np.deg2rad(omega)
    theta_hd = float(theta0) + np.cumsum(omega_rad, axis=1) * dt
    theta_hd[:, 0] = float(theta0)
    u = np.zeros((1, T, 3), dtype=np.float32)
    u[:, :, 0] = omega
    u[:, 0, 1] = math.cos(float(theta0))
    u[:, 0, 2] = math.sin(float(theta0))
    y = np.stack([np.cos(theta_hd), np.sin(theta_hd)], axis=-1).astype(np.float32)
    return TaskTrials(
        task_family='path_integration',
        n_input=3, n_output=2, dt=float(dt),
        stimulus=torch.from_numpy(u).to(device),
        target  =torch.from_numpy(y).to(device),
        theta_hd=torch.from_numpy(theta_hd).to(device),
        is_stop =torch.from_numpy(np.zeros((1, T), dtype=np.float32)).to(device),
        omega   =torch.from_numpy(omega).to(device),
    )


def _per_neuron_drive(net, u_t):
    T = u_t.shape[1]
    with torch.no_grad():
        drives = [net._project_in(u_t[:, t, :]) for t in range(T)]
    return torch.stack(drives, dim=1)[0].cpu().numpy()


def _run_one(net, batch):
    u_t = batch.stimulus
    with torch.no_grad():
        y_pred, h = net(u_t)
    return (
        batch.theta_hd[0].cpu().numpy(),
        h[0].cpu().numpy(),
        _per_neuron_drive(net, u_t),
        y_pred[0].cpu().numpy(),
    )


def _draw_panel(ax_raw, ax_hd, voltage, drive, theta_gt, y_pred, cx,
                dt, n_show):
    """Replicates `_cx_voltage_sanity_combined_plot` layout into two
    pre-existing axes (traces ax_raw, HD strip ax_hd)."""
    T = min(n_show, voltage.shape[0])
    t = np.arange(T) * dt

    neuron_types_np = np.asarray(cx["neuron_types"], dtype=np.int64)
    type_names = list(cx["type_names"])

    n_per_type = 5
    type_blocks: list[tuple[str, list[int]]] = []
    chosen: list[tuple[str, int]] = []
    for ti, name in enumerate(type_names):
        idx = np.where(neuron_types_np == ti)[0]
        if idx.size == 0:
            continue
        take = idx[:n_per_type].tolist()
        type_blocks.append((name, take))
        for ix in take:
            chosen.append((name, int(ix)))

    if not chosen:
        return

    raw = np.stack(
        [voltage[:T, ix] - voltage[:T, ix].mean() for _, ix in chosen],
        axis=0,
    )
    step_raw = 3.0 * float(raw.std()) if raw.size else 1.0
    step_raw = max(step_raw, 1e-6)

    pen_subpop = cx.get("pen_subpop_ix", {})

    def _first_ix(key: str):
        vals = pen_subpop.get(key, [])
        return int(vals[0]) if len(vals) > 0 else None

    pena_l, pena_r = _first_ix("PENa_L"), _first_ix("PENa_R")
    penb_l, penb_r = _first_ix("PENb_L"), _first_ix("PENb_R")

    slot = 0
    block_centres: list[tuple[str, float]] = []
    for name, idx_list in type_blocks:
        block_start = slot
        for ix in idx_list:
            base = slot * step_raw
            ax_raw.plot(t, raw[slot] + base, color="black", lw=0.6)
            nm = name.replace("_", "")
            if slot == block_start:
                if nm.startswith("PENa"):
                    if pena_l is not None:
                        s = drive[:T, pena_l]
                        ax_raw.plot(t, (s - s.mean()) + base,
                                    color=STIM_L_COLOR, lw=0.6, alpha=0.75)
                    if pena_r is not None:
                        s = drive[:T, pena_r]
                        ax_raw.plot(t, (s - s.mean()) + base,
                                    color=STIM_R_COLOR, lw=0.6, alpha=0.75)
                elif nm.startswith("PENb"):
                    if penb_l is not None:
                        s = drive[:T, penb_l]
                        ax_raw.plot(t, (s - s.mean()) + base,
                                    color=STIM_L_COLOR, lw=0.6, alpha=0.75)
                    if penb_r is not None:
                        s = drive[:T, penb_r]
                        ax_raw.plot(t, (s - s.mean()) + base,
                                    color=STIM_R_COLOR, lw=0.6, alpha=0.75)
            slot += 1
        block_end = slot - 1
        block_centres.append((name, ((block_start + block_end) / 2) * step_raw))

    ax_raw.set_xlim(0, float(t[-1]) if T > 0 else 1.0)
    ax_raw.set_yticks([y for _, y in block_centres])
    ax_raw.set_yticklabels([n for n, _ in block_centres], fontsize=_TICK_FS)
    ax_raw.tick_params(axis="x", labelsize=_TICK_FS)
    ax_raw.set_ylim(-step_raw, slot * step_raw)

    true_hd_wrap = np.angle(np.exp(1j * theta_gt[:T]))
    decoded_hd = np.arctan2(y_pred[:T, 1], y_pred[:T, 0])
    ax_hd.plot(t, true_hd_wrap, color=_GT_COLOR, lw=0.0, marker=".", ms=_GT_MS)
    ax_hd.plot(t, decoded_hd,   color=_PRED_COLOR, lw=0.0, marker=".", ms=_PRED_MS)
    ax_hd.set_yticks([-np.pi, 0, np.pi])
    ax_hd.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=_TICK_FS)
    ax_hd.set_ylabel("HD (rad)", fontsize=_LABEL_FS)
    ax_hd.set_xlabel("time (s)", fontsize=_LABEL_FS)
    ax_hd.tick_params(axis="x", labelsize=_TICK_FS)
    ax_hd.axhline(0, color="0.5", lw=0.5)


def _build_figure(mode, n_steps, omega_deg, theta0, seed, device, out_path):
    """Render a two-column trace figure (one tile per parameterisation,
    stacked into ``ceil(len(MODELS)/2)`` rows x 2 cols)."""
    n_cols = 2
    n_rows = (len(MODELS) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(10 * n_cols, 6.75 * n_rows))
    outer = GridSpec(n_rows, n_cols, figure=fig, hspace=0.32, wspace=0.18,
                     left=0.05, right=0.99, top=0.97, bottom=0.04)
    letters = "abcdefghij"

    for k, (cfg, title) in enumerate(MODELS):
        row, col = divmod(k, n_cols)
        net, config = _load(cfg, device)
        sim = config.simulation
        cx = load_drosophila_cx_connectome(sim.connconstr_datapath)
        dt = float(net.dt)

        if mode == "ou":
            rng = np.random.default_rng(seed)
            batch = generate_path_integration_batch(
                batch_size=1, n_steps=n_steps, dt=dt, device=device, rng=rng,
            )
        else:
            batch = _build_const_batch(n_steps, dt, omega_deg, theta0, device)

        theta_gt, voltage, drive, y_pred = _run_one(net, batch)

        inner = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row, col], height_ratios=[4.0, 1.0],
            hspace=0.18,
        )
        ax_raw = fig.add_subplot(inner[0, 0])
        ax_hd  = fig.add_subplot(inner[1, 0], sharex=ax_raw)

        _draw_panel(ax_raw, ax_hd, voltage, drive, theta_gt, y_pred,
                    cx=cx, dt=dt, n_show=n_steps)

        ax_raw.set_title(title, fontsize=12, pad=6)
        ax_raw.text(-0.06, 1.04, letters[k], transform=ax_raw.transAxes,
                    fontsize=18, fontweight="bold", va="bottom", ha="right")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_steps_ou",    type=int,   default=10000)
    p.add_argument("--n_steps_const", type=int,   default=1500)
    p.add_argument("--omega-deg",     type=float, default=60.0)
    p.add_argument("--theta0",        type=float, default=0.0)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--out-dir",
                   default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--output-root",   default=None)
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass

    device = torch.device(args.device)

    _build_figure("ou",    args.n_steps_ou,    args.omega_deg, args.theta0,
                  args.seed, device,
                  out_path=os.path.join(args.out_dir, "fig_traces_combined.png"))

    _build_figure("const", args.n_steps_const, args.omega_deg, args.theta0,
                  args.seed, device,
                  out_path=os.path.join(args.out_dir, "fig_traces_const_omega.png"))


if __name__ == "__main__":
    main()
