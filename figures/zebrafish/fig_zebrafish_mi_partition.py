"""MI-based partition of the recurrent pool, four joint-task models.

Run across four trained joint-task models (rotation+distance and 2-D
path, each cumulative and leaky). For each recurrent neuron i — the
whole GABAergic substrate dIPN + dsIPN + IPN12 + IPN-core
(IPN28/29/31-36); afferents RIPN*/pt-IPN* are excluded by construction —
we compute the per-neuron MI on the integrated heading θ and on the
forward displacement d, then split into four pools:

    angular      :  I(h_i;θ) ≥ τθ, I(h_i;d) < τd
    displacement :  I(h_i;θ) < τθ, I(h_i;d) ≥ τd
    shared       :  I(h_i;θ) ≥ τθ, I(h_i;d) ≥ τd
    neither      :  low on both

The thresholds (τθ, τd) default to the per-axis medians, so each pool
is a quadrant of the MI scatter.

Four rows × four columns (one column per model):
    row 1  MI scatter I(h;θ) vs I(h;d), coloured by functional pool.
    row 2  the SAME scatter, coloured by anatomical family
           (dIPN, dsIPN, IPN12, IPN-core).
    row 3  per-family functional-pool composition (stacked bars).
    row 4  per-cell-type functional-pool composition (stacked bars):
           does the functional split line up with anatomy?

Usage:
    python figures/zebrafish/fig_zebrafish_mi_partition.py
    python figures/zebrafish/fig_zebrafish_mi_partition.py \\
        --runs RUN1 RUN2 RUN3 RUN4 --device cuda
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
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from connectome_gnn.utils import (  # noqa: E402
    load_data_root_from_json, set_data_root,
)
from connectome_gnn.models.utils import load_run_config  # noqa: E402
from connectome_gnn.models.registry import create_model  # noqa: E402
from connectome_gnn.zarr_io import load_raw_array  # noqa: E402
from connectome_gnn.utils import graphs_data_path  # noqa: E402

# Recurrent-pool cell-type prefixes (everything that's *not* an
# afferent: dIPN, IPNds, IPN12).
RECURRENT_PREFIXES = ("IPNd", "IPNds", "IPN12", "IPN-core")
POOL_ORDER = ("angular", "shared", "displacement", "neither")
# Okabe-Ito-derived qualitative palette (colour-blind safe, prints well).
POOL_COLOR = {
    "angular":      "#0072b2",   # blue          — heading-tuned (bottom-right)
    "displacement": "#e69f00",   # orange        — distance-tuned (top-left)
    "shared":       "#009e73",   # bluish green  — both (top-right)
    "neither":      "#cccccc",   # light grey    — low on both (bottom-left)
}


# Coarse recurrent-substrate families (for the cell-type colouring of
# the bottom row). dsIPN (IPNds*) is split out from dIPN (IPNd*) so the
# three anatomical families of the ring substrate are distinguishable.
REC_FAMILY_ORDER = ("dIPN", "dsIPN", "IPN12", "IPN-core")
REC_FAMILY_COLOR = {
    "dIPN":     "#d29922",   # amber
    "dsIPN":    "#2aa198",   # teal
    "IPN12":    "#c9468a",   # rose
    "IPN-core": "#6a51a3",   # purple
}


def _rec_family(name: str) -> str:
    """Map a recurrent cell-type name to its coarse anatomical family."""
    if name.startswith("IPNds"):
        return "dsIPN"
    if name.startswith("IPN12"):
        return "IPN12"
    if name.startswith("IPNd"):
        return "dIPN"
    return "IPN-core"      # IPN28/29/31-36


def _is_recurrent(name: str) -> bool:
    # The recurrent GABAergic substrate is every IPN* cell type: dIPN
    # (IPNd*), dsIPN (IPNds*), the IPN12 pool, and the IPN-core families
    # (IPN28/29/31-36). The afferents are RIPN* and pt-IPN*, neither of
    # which starts with "IPN", so a single prefix test suffices.
    return name.startswith("IPN")


def _mi_plugin(x: np.ndarray, y: np.ndarray, n_bins_x=32, n_bins_y=20) -> float:
    """Plug-in histogram MI(x; y) in bits. x: continuous scalar over T
    timepoints, y: continuous scalar over T timepoints (same length)."""
    x = np.asarray(x); y = np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]; y = y[finite]
    if x.size < 100:
        return 0.0
    # Bin both axes.
    xb = np.linspace(x.min(), x.max() + 1e-9, n_bins_x + 1)
    yb = np.linspace(y.min(), y.max() + 1e-9, n_bins_y + 1)
    H, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    Pxy = H / H.sum()
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = Pxy * (np.log2(Pxy + 1e-12) - np.log2(Px + 1e-12) - np.log2(Py + 1e-12))
    return float(np.nansum(term))


def _circular_mi(x: np.ndarray, theta: np.ndarray, n_bins_x=32, n_bins_theta=24) -> float:
    """MI(x; θ) where θ is a circular variable in (-π, π]."""
    x = np.asarray(x); theta = np.asarray(theta)
    finite = np.isfinite(x) & np.isfinite(theta)
    x = x[finite]; theta = ((theta[finite] + np.pi) % (2 * np.pi)) - np.pi
    if x.size < 100:
        return 0.0
    xb = np.linspace(x.min(), x.max() + 1e-9, n_bins_x + 1)
    tb = np.linspace(-np.pi, np.pi + 1e-9, n_bins_theta + 1)
    H, _, _ = np.histogram2d(x, theta, bins=[xb, tb])
    Pxy = H / H.sum()
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = Pxy * (np.log2(Pxy + 1e-12) - np.log2(Px + 1e-12) - np.log2(Py + 1e-12))
    return float(np.nansum(term))


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


def _run_trial(net, u, device):
    """Forward pass on one (T, n_in) trial. Returns h: (T, N) numpy."""
    with torch.no_grad():
        u_t = torch.from_numpy(u[None]).to(device)
        _, h_buf = net(u_t)
    return h_buf[0].cpu().numpy()


def _accumulate_hidden(net, u_test, y_test, device, n_trials):
    """Concatenate hidden state, true heading, true d across n_trials."""
    hs, thetas, ds = [], [], []
    n_trials = min(n_trials, u_test.shape[0])
    has_d_target = y_test.shape[-1] >= 3
    for k in range(n_trials):
        h = _run_trial(net, u_test[k], device)  # (T, N)
        y = y_test[k]                            # (T, n_out)
        # heading: atan2(sinθ, cosθ) from y[:, :2]
        theta = np.arctan2(y[:, 1], y[:, 0])
        hs.append(h); thetas.append(theta)
        if has_d_target:
            # in scalar_xi 3-col target, d is col 2; in position_2d 4-col it's missing.
            if y.shape[-1] == 3:
                ds.append(y[:, 2])
            elif y.shape[-1] == 4:
                # use cumulative travelled distance as a proxy:
                # √(Δx² + Δy²) accumulated
                dx = np.diff(y[:, 2], prepend=y[0, 2])
                dy = np.diff(y[:, 3], prepend=y[0, 3])
                ds.append(np.cumsum(np.sqrt(dx**2 + dy**2)))
            else:
                ds.append(np.zeros_like(theta))
        else:
            ds.append(np.zeros_like(theta))
    H = np.concatenate(hs, axis=0)            # (n_trials*T, N)
    theta = np.concatenate(thetas)
    d = np.concatenate(ds)
    return H, theta, d


# The four joint-target models compared across the columns. Each carries
# both a heading target θ and a forward-displacement target (scalar d for
# the "both" task, the 2-D path for "position_2d"), so the I(h;θ) vs I(h;d)
# partition is well defined for all four.
DEFAULT_RUNS = (
    "zebrafish_hd_si_ipn_917_v1_selfmotion_both",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_both_leaky",
    "zebrafish_hd_si_ipn_917_v1_position_2d",
    "zebrafish_hd_si_ipn_917_v1_position_2d_leaky",
)

_PROFILE_BY_TARGET = {
    (3, ("rotation",)):                ([0, 2, 3],    [0, 1]),
    (3, ("translation",)):             ([1],          [2]),
    (3, ("rotation", "translation")):  ([0, 1, 2, 3], [0, 1, 2]),
    (4, ("rotation",)):                ([0, 2, 3],    [0, 1]),
    (4, ("position_2d",)):             ([0, 1, 2, 3], [0, 1, 2, 3]),
}
_RECOGNISED = ("rotation", "translation", "position_2d")


def _compute_run(run_basename: str, data_root: str, n_trials: int, device):
    """Load one trained run, roll out, and return the per-recurrent-neuron
    MI partition plus the cell-type family of each neuron."""
    run_dir = os.path.join(data_root, "log", "zebrafish", run_basename)
    print(f"  loading {run_dir}")
    net, config = _load_run(run_dir, device)
    type_names = list(net.type_names)
    nt = np.asarray(net.neuron_types).astype(int)
    is_rec = np.array([_is_recurrent(type_names[int(t)]) for t in nt])
    rec_ix = np.where(is_rec)[0]
    fam = np.array([_rec_family(type_names[int(nt[i])]) for i in rec_ix],
                   dtype=object)
    ctype = np.array([type_names[int(nt[i])] for i in rec_ix], dtype=object)

    root = graphs_data_path(config.dataset)
    u_test = load_raw_array(f"{root}/test/stimulus.zarr")
    y_test = load_raw_array(f"{root}/test/target.zarr")
    _task_raw = list(getattr(config.training, "task_targets", None) or [])
    _task_key = tuple(t for t in _RECOGNISED if t in _task_raw)
    if u_test.shape[-1] >= 4 and _task_key:
        key = (int(y_test.shape[-1]), _task_key)
        if key in _PROFILE_BY_TARGET:
            in_cols, out_cols = _PROFILE_BY_TARGET[key]
            u_test = u_test[..., in_cols]
            y_test = y_test[..., out_cols]

    H, theta, d = _accumulate_hidden(net, u_test, y_test, device, n_trials)
    I_theta = np.zeros(rec_ix.size)
    I_d = np.zeros(rec_ix.size)
    for k, i in enumerate(rec_ix):
        x = H[:, i]
        I_theta[k] = _circular_mi(x, theta)
        I_d[k] = _mi_plugin(x, d)

    tau_theta = float(np.median(I_theta))
    tau_d = float(np.median(I_d))
    pool = np.full(rec_ix.size, "neither", dtype=object)
    pool[(I_theta >= tau_theta) & (I_d <  tau_d)] = "angular"
    pool[(I_theta <  tau_theta) & (I_d >= tau_d)] = "displacement"
    pool[(I_theta >= tau_theta) & (I_d >= tau_d)] = "shared"
    counts = {p: int(np.sum(pool == p)) for p in POOL_ORDER}
    fam_counts = {f: int(np.sum(fam == f)) for f in REC_FAMILY_ORDER}
    print(f"    n_rec={rec_ix.size}  τθ={tau_theta:.3f} τd={tau_d:.3f}  "
          f"pools={counts}  families={fam_counts}")
    return dict(I_theta=I_theta, I_d=I_d, pool=pool, counts=counts,
                fam=fam, ctype=ctype, tau_theta=tau_theta, tau_d=tau_d)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS),
                   help="four trained run directory basenames (one per column)")
    p.add_argument("--data_root",
                   default="/groups/saalfeld/home/allierc/GraphData")
    p.add_argument("--n_trials", type=int, default=64,
                   help="how many test trials to concatenate for the MI estimate")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=os.path.join(
        HERE, "fig_zebrafish_mi_partition.png"))
    args = p.parse_args()

    try:
        set_data_root(load_data_root_from_json())
    except FileNotFoundError:
        pass

    device = torch.device(args.device)
    results = []
    for j, run in enumerate(args.runs):
        print(f"[{j + 1}/{len(args.runs)}] {run}")
        results.append(_compute_run(run, args.data_root, args.n_trials, device))

    # ---- plot: 4 rows x N cols -------------------------------------------
    # Row 1: MI scatter coloured by functional pool.
    # Row 2: the SAME scatter coloured by anatomical FAMILY (4 families).
    # Row 3: per-family functional-pool composition (stacked bars).
    # Row 4: per-cell-type functional-pool composition (stacked bars).
    # Column identity (the four models) is given in the caption.
    from matplotlib.lines import Line2D
    ncol = len(results)
    LF, TF, LET = 13, 11, 14
    fig, axes = plt.subplots(4, ncol, figsize=(4.6 * ncol, 17.2))
    letters = "abcdefghijklmnopqrstuvwxyz"
    xmax = max(float(r["I_theta"].max()) for r in results) * 1.05
    ymax = max(float(r["I_d"].max()) for r in results) * 1.05
    all_types = sorted({t for r in results for t in set(r["ctype"])})

    def _stacked_pool_bars(ax, groups, group_of):
        """Stacked per-group functional-pool fractions on ``ax``.
        ``group_of`` maps each recurrent neuron to its group key."""
        x = np.arange(len(groups))
        bottom = np.zeros(len(groups))
        gk = group_of
        for pl in POOL_ORDER:
            frac = np.array([
                float(np.mean((gk == g) & (r["pool"] == pl))
                      / max(np.mean(gk == g), 1e-9))
                if (gk == g).any() else 0.0 for g in groups])
            ax.bar(x, frac, bottom=bottom, width=0.82,
                   color=POOL_COLOR[pl], edgecolor="none")
            bottom += frac
        ax.set_xticks(x)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        return x

    for j, r in enumerate(results):
        # --- row 1: scatter coloured by functional pool ---
        ax = axes[0, j]
        for pl in POOL_ORDER:
            m = r["pool"] == pl
            if not m.any():
                continue
            ax.scatter(r["I_theta"][m], r["I_d"][m], s=18,
                       color=POOL_COLOR[pl], edgecolor="none", alpha=0.85,
                       label=f"{pl} (n={r['counts'][pl]})")
        ax.axvline(r["tau_theta"], color="0.4", lw=0.7, ls="--")
        ax.axhline(r["tau_d"],     color="0.4", lw=0.7, ls="--")
        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
        if j == 0:
            # Axis titles written once, on panel a only.
            ax.set_xlabel(r"$I(\hat h_i;\,\theta)$  (bits, heading)", fontsize=LF)
            ax.set_ylabel(r"$I(\hat h_i;\,d)$  (bits, translation)", fontsize=LF)
            ax.legend(fontsize=9, loc="upper right", frameon=False)
        ax.tick_params(labelsize=TF)
        ax.text(-0.10, 1.04, letters[j], transform=ax.transAxes,
                fontsize=LET, fontweight="bold")

        # --- row 2: scatter coloured by anatomical family ---
        ax = axes[1, j]
        for famk in REC_FAMILY_ORDER:
            m = r["fam"] == famk
            if not m.any():
                continue
            ax.scatter(r["I_theta"][m], r["I_d"][m], s=18,
                       color=REC_FAMILY_COLOR[famk], edgecolor="none",
                       alpha=0.85, label=f"{famk} (n={int(m.sum())})")
        ax.axvline(r["tau_theta"], color="0.4", lw=0.7, ls="--")
        ax.axhline(r["tau_d"],     color="0.4", lw=0.7, ls="--")
        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
        if j == 0:
            ax.legend(fontsize=9, loc="upper right", frameon=False)
        ax.tick_params(labelsize=TF)
        ax.text(-0.10, 1.04, letters[ncol + j], transform=ax.transAxes,
                fontsize=LET, fontweight="bold")

        # --- row 3: per-family pool composition ---
        ax = axes[2, j]
        _stacked_pool_bars(ax, REC_FAMILY_ORDER, r["fam"])
        ax.set_xticklabels(REC_FAMILY_ORDER, rotation=30, ha="right", fontsize=8)
        if j == 0:
            ax.set_ylabel("pool fraction", fontsize=LF)
        ax.tick_params(axis="y", labelsize=TF)
        ax.text(-0.10, 1.04, letters[2 * ncol + j], transform=ax.transAxes,
                fontsize=LET, fontweight="bold")

        # --- row 4: per-cell-type pool composition ---
        ax = axes[3, j]
        _stacked_pool_bars(ax, all_types, r["ctype"])
        ax.set_xticklabels(all_types, rotation=90, fontsize=6)
        if j == 0:
            ax.set_ylabel("pool fraction", fontsize=LF)
        ax.tick_params(axis="y", labelsize=TF)
        ax.text(-0.10, 1.04, letters[3 * ncol + j], transform=ax.transAxes,
                fontsize=LET, fontweight="bold")

    plt.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_mi_partition] wrote {args.out}")


if __name__ == "__main__":
    main()
