"""Zebrafish all-neuron kinographs: 3 row orderings × 4 model columns, the
zebrafish analogue of the drosophila_cx Figs 13/14
(fig_kinographs_const_omega.py / fig_kinographs_both_sorts.py).

Columns (the no-observation vs +observation contrast, RNN and GNN):
    RNN no-obs   zebrafish_hd_si_ipn12_c0_gcamp      (λ_obs = 0)
    RNN +obs     zebrafish_hd_si_ipn12_c3_gcamp      (λ_obs = 1)
    GNN no-obs   zebrafish_hd_si_gnn_ipn12_c0_gcamp  (λ_obs = 0)
    GNN +obs     zebrafish_hd_si_gnn_ipn12_c3_gcamp  (λ_obs = 1)

Row orderings (identical to the drosophila figure):
    Row 1 (a–d): cell-type sort.
    Row 2 (e–h): cell-type primary, preferred-phase φ secondary within type.
    Row 3 (i–l): pure preferred-phase sort φ — the bump-migration view.

Each panel is a per-neuron z-scored kinograph, fixed ±3 colormap.

Two stimuli, via --pattern (run twice for the figure pair):
    const      constant ω (default 60°/s) — clean bump migration, fixed slope.
    swim_left  leftward swim-impulse integration (the natural task drive).

  python figures/zebrafish/fig_zebrafish_kinographs.py --pattern const
  python figures/zebrafish/fig_zebrafish_kinographs.py --pattern swim_left
writes figures/zebrafish/fig_zebrafish_kinographs_<pattern>.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

from connectome_gnn.config import NeuralGraphConfig                 # noqa: E402
from connectome_gnn.models.registry import create_model            # noqa: E402
from connectome_gnn.plot_anatomy_voltage import run_task_rollout   # noqa: E402
from connectome_gnn.utils import migrate_state_dict, set_data_root  # noqa: E402

_ROOT = os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")

# GNN +obs (zebrafish_hd_si_gnn_ipn12_c3_gcamp) is excluded: every GNN run
# with the observation loss (c1–c5) diverged (const-ω decode r≈0, exploding
# activity). Only the three converged models are shown.
MODELS = [
    ("zebrafish_hd_si_ipn12_c0_gcamp",     "RNN no-obs"),
    ("zebrafish_hd_si_ipn12_c3_gcamp",     "RNN $+$obs"),
    ("zebrafish_hd_si_gnn_ipn12_c0_gcamp", "GNN no-obs"),
]


def _resolve_config(name):
    for d in (os.path.join(_ROOT, "config", "zebrafish"),
              os.path.join(_REPO, "config", "zebrafish")):
        p = os.path.join(d, name + ".yaml")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"config not found for {name}")


def _load(name, device):
    cfg_path = _resolve_config(name)
    cfg = NeuralGraphConfig.from_yaml(cfg_path)
    log_dir = os.path.join(_ROOT, "log", "zebrafish", name)
    model = create_model(cfg.graph_model.signal_model_name,
                         aggr_type=cfg.graph_model.aggr_type,
                         config=cfg, device=device).to(device)
    ck = max(glob.glob(f"{log_dir}/models/best_model_with_*.pt"),
             key=os.path.getmtime)
    sd = torch.load(ck, map_location=device, weights_only=False)
    migrate_state_dict(sd)
    model.load_state_dict(sd["model_state_dict"], strict=False)
    model.eval()
    print(f"loaded {name}: {os.path.basename(ck)}  N={model.n_units}")
    return cfg, model


def _zscore_per_neuron(h):
    mu = h.mean(0, keepdims=True)
    sd = h.std(0, keepdims=True) + 1e-8
    return (h - mu) / sd


def _preferred_phase(h, theta):
    act = h - h.mean(0, keepdims=True)
    wcos = (act * np.cos(theta)[:, None]).sum(0)
    wsin = (act * np.sin(theta)[:, None]).sum(0)
    return np.arctan2(wsin, wcos)


def _plot_kinograph(ax, V, dt, title, vmax=3.0):
    im = ax.imshow(V, aspect="auto", interpolation="nearest", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax,
                   extent=(0.0, V.shape[1] * dt, V.shape[0], 0.0))
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11, pad=4)
    ax.tick_params(axis="x", labelsize=8)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pattern", default="const",
                   choices=["const", "swim_left", "swim_right"])
    p.add_argument("--n_steps", type=int, default=1500)
    p.add_argument("--omega-deg", type=float, default=60.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    set_data_root(_ROOT)
    device = torch.device(args.device)

    n_cols = len(MODELS)
    fig, axes = plt.subplots(3, n_cols, figsize=(5.2 * n_cols, 13.5))
    ims = [None, None, None]

    for k, (name, title) in enumerate(MODELS):
        cfg, model = _load(name, device)
        pc = cfg.plotting.model_copy(update=dict(
            anatomy_voltage_pattern=args.pattern,
            anatomy_voltage_n_steps=int(args.n_steps),
            anatomy_voltage_stride=1,
            anatomy_voltage_omega_deg=float(args.omega_deg),
            anatomy_voltage_theta0_rad=0.0))
        h, _y, theta, label, _ = run_task_rollout(model, pc, device)
        nt = np.asarray(model.neuron_types, dtype=np.int64)
        Hz = _zscore_per_neuron(h)
        phi = _preferred_phase(h, theta)
        dt = float(model.dt)

        ord_type = np.argsort(-nt, kind="stable")
        ord_type_phi = np.lexsort((phi, -nt))
        ord_phi = np.argsort(phi, kind="stable")

        ims[0] = _plot_kinograph(axes[0, k], Hz[:, ord_type].T, dt, title)
        ims[1] = _plot_kinograph(axes[1, k], Hz[:, ord_type_phi].T, dt, "")
        ims[2] = _plot_kinograph(axes[2, k], Hz[:, ord_phi].T, dt, "")
        print(f"  [{name}] {label}; h={h.shape}")

    axes[0, 0].set_ylabel("neuron (cell-type sort)", fontsize=11)
    axes[1, 0].set_ylabel(r"neuron (cell-type, then $\varphi_i$ within type)",
                          fontsize=11)
    axes[2, 0].set_ylabel(r"neuron (preferred-phase sort $\varphi_i$)",
                          fontsize=11)

    letters = "abcdefghijkl"
    for k, ax in enumerate(axes.flatten()[:len(letters)]):
        ax.text(-0.05, 1.04, letters[k], transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="bottom", ha="right")

    for r in range(3):
        cb = fig.colorbar(ims[r], ax=axes[r, :], fraction=0.012, pad=0.01,
                          label="z-score")
        cb.ax.tick_params(labelsize=8)

    out = args.output or os.path.join(
        _HERE, f"fig_zebrafish_kinographs_{args.pattern}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
