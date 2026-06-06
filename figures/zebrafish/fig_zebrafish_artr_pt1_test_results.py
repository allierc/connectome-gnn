"""Per-model 2-row test composite to stack below the evolution figure.

For each trained ARTR / pt-IPN1 variant, renders a 2-row composite
that occupies just the last two rows of the merged paper figure
(panels i and j, following a–h in the evolution figure):

    Row i — random held-out trials: 5 columns, each showing the
            integrated quantity (true green / decoded black). No
            grey background, no per-column title, only the
            row-level letter label on the left.
    Row j — constant-input deterministic sweep, same layout.

Integrated quantity per variant:
    rotation, both, both_leaky     : wrapped HD (rad)
    translation, translation_leaky : forward distance d
    position_2d, position_2d_leaky : 2-D path (x, y)

Output: figures/zebrafish/fig_test_<run>.png — referenced by the
LaTeX figure block, one composite per trained model.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from connectome_gnn.utils import (  # noqa: E402
    load_data_root_from_json, set_data_root, graphs_data_path,
)
from connectome_gnn.zarr_io import load_raw_array  # noqa: E402
from connectome_gnn.models.utils import load_run_config  # noqa: E402
from connectome_gnn.models.registry import create_model  # noqa: E402
from connectome_gnn.models.bump_attractor_eval import (  # noqa: E402
    _deterministic_sweep_rollout,
)


PREFIX = "zebrafish_hd_si_ipn12_artr_pt1_"
VARIANTS = [
    # suffix, kind: "hd" | "d" | "xy"
    ("selfmotion_rotation",          "hd"),
    ("selfmotion_translation",       "d"),
    ("selfmotion_translation_leaky", "d"),
    ("selfmotion_both",              "hd"),
    ("selfmotion_both_leaky",        "hd"),
    ("position_2d",                  "xy"),
    ("position_2d_leaky",            "xy"),
]

GT_COLOR = "#4daf4a"
PRED_COLOR = "black"
TICK_FS = 9
LABEL_FS = 11
LETTER_FS = 18
GT_LW = 2.4
PRED_LW = 0.6

_PROFILE_BY_TARGET = {
    (3, ("rotation",)):                ([0, 2, 3],    [0, 1]),
    (3, ("translation",)):             ([1],          [2]),
    (3, ("rotation", "translation")):  ([0, 1, 2, 3], [0, 1, 2]),
    (4, ("rotation",)):                ([0, 2, 3],    [0, 1]),
    (4, ("position_2d",)):             ([0, 1, 2, 3], [0, 1, 2, 3]),
}
_RECOGNISED = ("rotation", "translation", "position_2d")


def _load_run(run_dir: str, device):
    config_name = os.path.basename(run_dir)
    config, _ = load_run_config(config_name, explicit_output_root=False,
                                 task="train")
    net = create_model(
        config.graph_model.signal_model_name,
        aggr_type=config.graph_model.aggr_type,
        config=config, device=device,
    )
    ckpts = sorted(glob.glob(os.path.join(run_dir, "models",
                                          "best_model_with_*.pt")))
    if not ckpts:
        sys.exit(f"no checkpoint in {run_dir}")
    sd = torch.load(ckpts[-1], map_location=device, weights_only=False)
    sd = sd if isinstance(sd, dict) else sd.state_dict()
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net, config


def _slice_test(u_test, y_test, config):
    raw = list(getattr(config.training, "task_targets", None) or [])
    key = tuple(t for t in _RECOGNISED if t in raw)
    if u_test.shape[-1] >= 4 and key:
        prof = (int(y_test.shape[-1]), key)
        if prof in _PROFILE_BY_TARGET:
            ic, oc = _PROFILE_BY_TARGET[prof]
            u_test = u_test[..., ic]
            y_test = y_test[..., oc]
    return u_test, y_test


def _pick_trials(rng, y_test, n_show=5, k_cand=32):
    n_test = y_test.shape[0]
    K = int(min(k_cand, n_test))
    cand = np.sort(rng.choice(n_test, size=K, replace=False))
    y_cand = np.asarray(y_test[cand])
    y_c = y_cand - y_cand.mean(axis=1, keepdims=True)
    scores = np.abs(y_c).max(axis=(1, 2))
    order = np.argsort(-scores)
    picks = cand[order[:n_show]]
    return np.sort(picks)


def _forward_trials(net, u_test, idx, device):
    u = torch.from_numpy(np.asarray(u_test[idx])).to(device)
    with torch.no_grad():
        y_pred, _ = net(u)
    return y_pred.cpu().numpy()


def _plot_row_hd(axes, t, y_true, y_pred):
    """Wrapped HD (-π, π] per column, GT green vs decoded black."""
    for col, ax in enumerate(axes):
        ax.set_facecolor("white")
        true_hd = np.arctan2(y_true[col, :, 1], y_true[col, :, 0])
        pred_hd = np.arctan2(y_pred[col, :, 1], y_pred[col, :, 0])
        ax.plot(t, true_hd, color=GT_COLOR, lw=GT_LW)
        ax.plot(t, pred_hd, color=PRED_COLOR, lw=PRED_LW)
        ax.set_ylim(-np.pi - 0.2, np.pi + 0.2)
        ax.tick_params(labelsize=TICK_FS)
        if col == 0:
            ax.set_ylabel("HD (rad)", fontsize=LABEL_FS)
        ax.set_xlabel("time (s)", fontsize=LABEL_FS)


def _plot_row_d(axes, t, y_true, y_pred, d_col):
    for col, ax in enumerate(axes):
        ax.set_facecolor("white")
        ax.plot(t, y_true[col, :, d_col], color=GT_COLOR, lw=GT_LW)
        ax.plot(t, y_pred[col, :, d_col], color=PRED_COLOR, lw=PRED_LW)
        ax.tick_params(labelsize=TICK_FS)
        if col == 0:
            ax.set_ylabel(r"$d$", fontsize=LABEL_FS)
        ax.set_xlabel("time (s)", fontsize=LABEL_FS)


def _plot_row_xy(axes, y_true, y_pred, xy_cols):
    for col, ax in enumerate(axes):
        ax.set_facecolor("white")
        tx = y_true[col, :, xy_cols[0]]; ty = y_true[col, :, xy_cols[1]]
        dx = y_pred[col, :, xy_cols[0]]; dy = y_pred[col, :, xy_cols[1]]
        ax.plot(tx, ty, color=GT_COLOR, lw=GT_LW)
        ax.plot(dx, dy, color=PRED_COLOR, lw=PRED_LW)
        ax.plot([0], [0], 'o', color='0.4', ms=4, zorder=5)
        ax.set_aspect('equal', adjustable='datalim')
        ax.tick_params(labelsize=TICK_FS)
        ax.set_xlabel(r"$x$", fontsize=LABEL_FS)
        if col == 0:
            ax.set_ylabel(r"$y$", fontsize=LABEL_FS)


def _build(run_dir: str, kind: str, out_path: str, device,
            n_show: int = 5) -> None:
    if not os.path.isdir(run_dir):
        print(f"  SKIP {os.path.basename(out_path)}: missing {run_dir}")
        return
    net, config = _load_run(run_dir, device)
    root = graphs_data_path(config.dataset)
    u_test = load_raw_array(f"{root}/test/stimulus.zarr")
    y_test = load_raw_array(f"{root}/test/target.zarr")
    u_test, y_test = _slice_test(u_test, y_test, config)
    dt = float(config.task.swim_integration.dt)

    # Random held-out trials (top row).
    rng = np.random.default_rng(int(config.training.seed) + 17)
    idx_r = _pick_trials(rng, y_test, n_show=n_show, k_cand=32)
    y_pred_r = _forward_trials(net, u_test, idx_r, device)
    T = u_test.shape[1]
    t = np.arange(T) * dt

    # Deterministic sweep (bottom row).
    T_sweep = 2000
    if kind == "hd":
        omega_set = [-120.0, -60.0, 60.0, 120.0, 180.0]
        rollouts = [_deterministic_sweep_rollout(
            net, n_steps=T_sweep, omega_deg_per_s=om, device=device,
        ) for om in omega_set]
    elif kind == "d":
        v_set = [-2.0, -1.0, 0.5, 1.0, 2.0]
        rollouts = [_deterministic_sweep_rollout(
            net, n_steps=T_sweep, v_fwd_per_s=v, device=device,
        ) for v in v_set]
    else:  # xy
        omega_v_set = [(-120.0, 1.0), (-60.0, 1.0), (30.0, 1.0),
                       (60.0, 1.0), (120.0, 1.0)]
        rollouts = [_deterministic_sweep_rollout(
            net, n_steps=T_sweep, omega_deg_per_s=om, v_fwd_per_s=v,
            device=device,
        ) for om, v in omega_v_set]
    t_sweep = np.arange(T_sweep) * dt

    fig, axes = plt.subplots(2, n_show,
                              figsize=(2.6 * n_show, 5.4))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    # ---- top row (i): random trials -----------------------------------
    if kind == "hd":
        # heading column indices are 0, 1 (cos, sin)
        _plot_row_hd(axes[0], t, y_test[idx_r], y_pred_r)
    elif kind == "d":
        d_col = 0 if y_test.shape[-1] == 1 else 2
        _plot_row_d(axes[0], t, y_test[idx_r], y_pred_r, d_col)
    else:
        xy_cols = (2, 3)
        _plot_row_xy(axes[0], y_test[idx_r], y_pred_r, xy_cols)

    # ---- bottom row (j): deterministic sweep --------------------------
    if kind == "hd":
        # y_true cosine/sine from true_theta in the rollout
        y_true_sweep = np.stack(
            [np.stack([np.cos(np.asarray(ro["true_theta"])),
                       np.sin(np.asarray(ro["true_theta"]))], axis=-1)
             for ro in rollouts], axis=0)
        y_pred_sweep = np.stack(
            [np.asarray(ro["y_pred"])[..., :2] for ro in rollouts], axis=0)
        _plot_row_hd(axes[1], t_sweep, y_true_sweep, y_pred_sweep)
    elif kind == "d":
        true_d = np.stack([np.asarray(ro["true_xi"])
                           for ro in rollouts], axis=0)[..., None]
        pred_d = np.stack([np.asarray(ro["decoded_xi"])
                           for ro in rollouts], axis=0)[..., None]
        _plot_row_d(axes[1], t_sweep, true_d, pred_d, 0)
    else:
        true_xy = np.stack([np.asarray(ro["true_xy"])
                            for ro in rollouts], axis=0)  # (5, T, 2)
        pred_xy = np.stack([np.asarray(ro["decoded_xy"])
                            for ro in rollouts], axis=0)
        # repack as (5, T, 4) with x,y at columns 2,3 for _plot_row_xy
        zeros = np.zeros_like(true_xy)
        true_pad = np.concatenate([zeros, true_xy], axis=-1)
        pred_pad = np.concatenate([zeros, pred_xy], axis=-1)
        _plot_row_xy(axes[1], true_pad, pred_pad, (2, 3))

    # Panel letters on the left of each row.
    axes[0, 0].text(-0.30, 1.02, "i", transform=axes[0, 0].transAxes,
                    fontsize=LETTER_FS, fontweight="bold",
                    ha="left", va="top")
    axes[1, 0].text(-0.30, 1.02, "j", transform=axes[1, 0].transAxes,
                    fontsize=LETTER_FS, fontweight="bold",
                    ha="left", va="top")

    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.10,
                         hspace=0.32, wspace=0.30)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root",
                   default="/groups/saalfeld/home/allierc/GraphData")
    p.add_argument("--out_dir", default=HERE)
    p.add_argument("--device", default="cpu")
    p.add_argument("--only", nargs="+", default=None,
                   help="render only these suffixes")
    args = p.parse_args()

    try:
        set_data_root(load_data_root_from_json())
    except FileNotFoundError:
        pass

    device = torch.device(args.device)
    wanted = set(args.only) if args.only else None
    for suffix, kind in VARIANTS:
        if wanted is not None and suffix not in wanted:
            continue
        run = PREFIX + suffix
        run_dir = os.path.join(args.data_root, "log", "zebrafish", run)
        out_path = os.path.join(args.out_dir, f"fig_test_{run}.png")
        try:
            _build(run_dir, kind, out_path, device)
        except Exception as e:
            print(f"  FAIL {suffix}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
