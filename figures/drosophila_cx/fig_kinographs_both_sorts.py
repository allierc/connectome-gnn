"""3 × 4 figure showing the all-neurons-by-time kinographs of the four
CX models under the natural OU velocity stimulus, in three row-wise sort
conventions:

    Row 1 (panels a–d):  cell-type sort, ER6 → EPG top-to-bottom
                          (same convention as fig:task_traces).
    Row 2 (panels e–h):  within-cell-type preferred-phase sort —
                          neuron_type primary, preferred-phase secondary.
                          Each cell-type block is itself ring-ordered.
    Row 3 (panels i–l):  pure preferred-phase sort,
                          φ_i = arg(Σ_t (h_i(t) - <h_i>) e^{iθ(t)}),
                          low-φ → high-φ top-to-bottom; reveals the
                          ring-attractor bump-migration diagonal that the
                          cell-type sort scrambles.

Each panel is a single full-train-split kinograph (≈64 000 frames at
dt = 0.01 s, ≈640 s); per-neuron z-score, fixed ±3 colormap.

Output:
    docs/figure/fig_kinographs_both_sorts.png

CLI:
    python docs/figure/fig_kinographs_both_sorts.py
"""
from __future__ import annotations

import argparse
import glob
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
from connectome_gnn.generators.utils import generate_path_integration_batch


MODELS = [
    ("drosophila_cx_pi",                       "Known-ODE RNN"),
    ("drosophila_cx_pi_fc",                    "fully connected RNN"),
    ("drosophila_cx_pi_gnn_tailloss_unsquared","GNN"),
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
    # Per-config epoch override: GNN best at epoch 5.
    if prefer_epoch is None and "gnn_tailloss" in config_name:
        prefer_epoch = 5
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


def _run_ou(net, n_steps, device, seed):
    rng = np.random.default_rng(seed)
    batch = generate_path_integration_batch(
        batch_size=1, n_steps=n_steps,
        dt=float(net.dt), device=device, rng=rng,
    )
    u_t = batch.stimulus
    theta = batch.theta_hd[0].cpu().numpy()      # (T,)
    with torch.no_grad():
        _, h = net(u_t)
    return h[0].cpu().numpy(), theta              # (T, N), (T,)


def _zscore_per_neuron(h_traj):
    mu = h_traj.mean(axis=0, keepdims=True)
    sd = h_traj.std(axis=0,  keepdims=True) + 1e-8
    return (h_traj - mu) / sd


def _preferred_phase(h_traj, theta):
    """φ_i = arg(Σ_t (h_i - <h_i>) e^{iθ(t)}). Returns (N,) in (-π, π]."""
    act = h_traj - h_traj.mean(axis=0, keepdims=True)
    cos_h = np.cos(theta)[:, None]
    sin_h = np.sin(theta)[:, None]
    wcos = (act * cos_h).sum(axis=0)
    wsin = (act * sin_h).sum(axis=0)
    return np.arctan2(wsin, wcos)


def _order_by_type_descending(neuron_types):
    """Descending type index so ER6 (high) sits at top of imshow."""
    return np.argsort(-neuron_types, kind="stable")


def _order_within_type_by_phase(neuron_types, pref_phase):
    """Type primary (descending → ER6 top), phase secondary (ascending)."""
    # np.lexsort: last key is the primary sort key.
    return np.lexsort((pref_phase, -neuron_types))


def _order_by_preferred_phase(pref_phase):
    return np.argsort(pref_phase, kind="stable")


def _plot_kinograph(ax, V_sorted, dt, title, vmax=3.0):
    """V_sorted: (N, T) already z-scored and reordered. Plot one tile."""
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
    p.add_argument("--n_steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=None)
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

    # 3 rows × len(MODELS) cols.
    n_cols = len(MODELS)
    fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 14.5))

    ims_per_row = [None, None, None]

    for k, (cfg, title) in enumerate(MODELS):
        net = _load(cfg, device)
        h, theta = _run_ou(net, args.n_steps, device, args.seed)
        neuron_types = np.asarray(net.neuron_types, dtype=np.int64)
        H_z = _zscore_per_neuron(h)          # (T, N)
        pref_phase = _preferred_phase(h, theta)

        ord_type        = _order_by_type_descending(neuron_types)
        ord_type_phase  = _order_within_type_by_phase(neuron_types, pref_phase)
        ord_phase       = _order_by_preferred_phase(pref_phase)

        V_type        = H_z[:, ord_type].T
        V_type_phase  = H_z[:, ord_type_phase].T
        V_phase       = H_z[:, ord_phase].T

        # Title only on the top row.
        ims_per_row[0] = _plot_kinograph(axes[0, k], V_type,
                                          dt=float(net.dt), title=title)
        ims_per_row[1] = _plot_kinograph(axes[1, k], V_type_phase,
                                          dt=float(net.dt), title="")
        ims_per_row[2] = _plot_kinograph(axes[2, k], V_phase,
                                          dt=float(net.dt), title="")

    # Row labels on the left.
    axes[0, 0].set_ylabel("neuron (cell-type sort)", fontsize=11)
    axes[1, 0].set_ylabel(r"neuron (cell-type, then $\varphi_i$ within type)",
                          fontsize=11)
    axes[2, 0].set_ylabel(r"neuron (preferred-phase sort $\varphi_i$)",
                          fontsize=11)

    # Panel labels a–l in figure-relative coords.
    letters = "abcdefghijkl"
    flat = list(axes.flatten())
    for k, ax in enumerate(flat[: len(letters)]):
        ax.text(-0.05, 1.04, letters[k], transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="right")

    # Per-row colorbars on the right edge.
    for r in range(3):
        cb = fig.colorbar(ims_per_row[r], ax=axes[r, :],
                          fraction=0.012, pad=0.01, label="z-score")
        cb.ax.tick_params(labelsize=8)

    out = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fig_kinographs_both_sorts.png",
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
