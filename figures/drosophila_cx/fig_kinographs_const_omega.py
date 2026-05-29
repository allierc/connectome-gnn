"""3 × 4 figure mirroring `fig_kinographs_both_sorts.py`, but driven by a
deterministic *constant angular velocity* stimulus instead of the natural
OU rollout. The constant-omega sweep maps directly to the canonical
bump-migration test used for the rollout-r metric (omega = 60 deg/s,
T = 1500 frames at dt = 0.01 s -> 15 s -> 2.5 full turns), so the diagonals
in the preferred-phase rows have a fixed slope set by omega.

Row layout (identical to the OU figure):
    Row 1 (a--d):  cell-type sort (ER6 -> EPG top-to-bottom).
    Row 2 (e--h):  cell-type primary, preferred-phase secondary within type.
    Row 3 (i--l):  pure preferred-phase sort.

Each panel is a single per-neuron z-scored kinograph with fixed +-3
colormap.

Output:
    docs/figure/fig_kinographs_const_omega.png

CLI:
    python docs/figure/fig_kinographs_const_omega.py [--omega-deg 60]
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

from connectome_gnn.utils import log_path, load_data_root_from_json, set_data_root
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.models.registry import create_model
from connectome_gnn.task_state import TaskTrials


MODELS = [
    ("drosophila_cx_pi_epg_no_tv_cv0",        "Known-ODE no-TV"),
    ("drosophila_cx_pi_epg_tv_cv0",           "Known-ODE $+$TV"),
    ("drosophila_cx_pi_gnn_epg_no_tv_cv0",    "GNN no-TV"),
    ("drosophila_cx_pi_gnn_epg_tv_cv0",       "GNN $+$TV"),
    ("drosophila_cx_pi_fc_epg_cv0",           "fully connected"),
    ("drosophila_cx_pi_frozen_Wrec_epg_cv0",  "frozen $W^{\\mathrm{rec}}$"),
]


def _load(config_name, device, prefer_epoch=None):
    config, _ = load_run_config(config_name, explicit_output_root=False, task="train")
    ckpt_dir = os.path.join(log_path(config.config_file), "models")
    cands = sorted(
        glob.glob(os.path.join(ckpt_dir, "best_model_with_0_graphs_*.pt")),
        key=lambda p_: int(p_.rsplit("_", 1)[1].rstrip(".pt")),
    )
    if not cands:
        raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
    # Per-config epoch override: the GNN is best at epoch 5 (the soft-curriculum
    # tail-loss training overfits past that point).
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
    return model


def _build_const_omega_batch(n_steps, dt, omega_deg, theta0, device):
    """Deterministic constant-omega TaskTrials (B=1)."""
    T = int(n_steps)
    omega = np.full((1, T), float(omega_deg), dtype=np.float32)
    omega_rad = np.deg2rad(omega)
    theta_hd = float(theta0) + np.cumsum(omega_rad, axis=1) * dt
    theta_hd[:, 0] = float(theta0)
    u = np.zeros((1, T, 3), dtype=np.float32)
    u[:, :, 0] = omega                                  # deg/s
    u[:, 0, 1] = math.cos(float(theta0))
    u[:, 0, 2] = math.sin(float(theta0))
    y = np.stack([np.cos(theta_hd), np.sin(theta_hd)], axis=-1).astype(np.float32)
    is_stop = np.zeros((1, T), dtype=np.float32)
    return TaskTrials(
        task_family='path_integration',
        n_input=3, n_output=2, dt=float(dt),
        stimulus=torch.from_numpy(u).to(device),
        target  =torch.from_numpy(y).to(device),
        theta_hd=torch.from_numpy(theta_hd).to(device),
        is_stop =torch.from_numpy(is_stop).to(device),
        omega   =torch.from_numpy(omega).to(device),
    )


def _run_const(net, n_steps, dt, omega_deg, theta0, device):
    batch = _build_const_omega_batch(n_steps, dt, omega_deg, theta0, device)
    theta = batch.theta_hd[0].cpu().numpy()
    with torch.no_grad():
        _, h = net(batch.stimulus)
    return h[0].cpu().numpy(), theta


def _zscore_per_neuron(h_traj):
    mu = h_traj.mean(axis=0, keepdims=True)
    sd = h_traj.std(axis=0,  keepdims=True) + 1e-8
    return (h_traj - mu) / sd


def _preferred_phase(h_traj, theta):
    act = h_traj - h_traj.mean(axis=0, keepdims=True)
    cos_h = np.cos(theta)[:, None]
    sin_h = np.sin(theta)[:, None]
    wcos = (act * cos_h).sum(axis=0)
    wsin = (act * sin_h).sum(axis=0)
    return np.arctan2(wsin, wcos)


def _order_by_type_descending(neuron_types):
    return np.argsort(-neuron_types, kind="stable")


def _order_within_type_by_phase(neuron_types, pref_phase):
    return np.lexsort((pref_phase, -neuron_types))


def _order_by_preferred_phase(pref_phase):
    return np.argsort(pref_phase, kind="stable")


def _plot_kinograph(ax, V_sorted, dt, title, vmax=3.0):
    T = V_sorted.shape[1]
    t_end = T * dt
    im = ax.imshow(
        V_sorted, aspect="auto", interpolation="nearest",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=(0.0, float(t_end), float(V_sorted.shape[0]), 0.0),
    )
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10, pad=4)
    ax.tick_params(axis="x", labelsize=8)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_steps",   type=int,   default=1500,
                   help="number of frames (default 1500 = 15 s at dt=0.01)")
    p.add_argument("--omega-deg", type=float, default=60.0,
                   help="constant angular velocity in deg/s (default 60)")
    p.add_argument("--theta0",    type=float, default=0.0,
                   help="initial heading in radians (default 0)")
    p.add_argument("--device",  default="cpu")
    p.add_argument("--output",  default=None)
    p.add_argument("--output-root", default=None)
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass

    device = torch.device(args.device)

    n_cols = len(MODELS)
    fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 14.5))
    ims_per_row = [None, None, None]

    for k, (cfg, title) in enumerate(MODELS):
        net = _load(cfg, device)
        h, theta = _run_const(net, args.n_steps, float(net.dt),
                              args.omega_deg, args.theta0, device)
        neuron_types = np.asarray(net.neuron_types, dtype=np.int64)
        H_z = _zscore_per_neuron(h)
        pref_phase = _preferred_phase(h, theta)

        ord_type       = _order_by_type_descending(neuron_types)
        ord_type_phase = _order_within_type_by_phase(neuron_types, pref_phase)
        ord_phase      = _order_by_preferred_phase(pref_phase)

        V_type       = H_z[:, ord_type].T
        V_type_phase = H_z[:, ord_type_phase].T
        V_phase      = H_z[:, ord_phase].T

        ims_per_row[0] = _plot_kinograph(axes[0, k], V_type,
                                          dt=float(net.dt), title=title)
        ims_per_row[1] = _plot_kinograph(axes[1, k], V_type_phase,
                                          dt=float(net.dt), title="")
        ims_per_row[2] = _plot_kinograph(axes[2, k], V_phase,
                                          dt=float(net.dt), title="")

    axes[0, 0].set_ylabel("neuron (cell-type sort)", fontsize=11)
    axes[1, 0].set_ylabel(r"neuron (cell-type, then $\varphi_i$ within type)",
                          fontsize=11)
    axes[2, 0].set_ylabel(r"neuron (preferred-phase sort $\varphi_i$)",
                          fontsize=11)

    letters = "abcdefghijkl"
    flat = list(axes.flatten())
    for k, ax in enumerate(flat[: len(letters)]):
        ax.text(-0.05, 1.04, letters[k], transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="right")

    for r in range(3):
        cb = fig.colorbar(ims_per_row[r], ax=axes[r, :],
                          fraction=0.012, pad=0.01, label="z-score")
        cb.ax.tick_params(labelsize=8)

    out = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fig_kinographs_const_omega.png",
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
