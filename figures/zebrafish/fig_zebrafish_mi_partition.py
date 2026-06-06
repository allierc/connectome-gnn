"""MI-based partition of the recurrent pool + W^rec edge-block analysis.

Run on a trained joint-task model (rotation + scalar-d or
rotation + 2-D position). For each recurrent neuron i (dIPN + IPN12
pool, n=551 of 839 — afferents are excluded by construction) we
compute the per-neuron MI on the integrated heading θ and on the
forward distance d, then split into three pools:

    angular      :  I(h_i;θ) ≥ τθ, I(h_i;d) < τd
    displacement :  I(h_i;θ) < τθ, I(h_i;d) ≥ τd
    shared       :  I(h_i;θ) ≥ τθ, I(h_i;d) ≥ τd

The thresholds (τθ, τd) default to the per-axis medians, so each pool
is a quadrant of the MI scatter. The fourth quadrant (low on both)
is the residual "neither" group, kept on the scatter but not in the
edge analysis.

Two panels:
    (a) MI scatter I(h;θ) vs I(h;d), one dot per recurrent neuron,
        coloured by pool. Thresholds drawn as dashed lines.
    (b) Edge-block magnitude
        avg |W^rec_{p→q}|  over recurrent→recurrent edges only,
        between the three pools. A block-diagonal pattern would
        indicate two disjoint micro-circuits; a dense cross-block
        would indicate distributed computation.

Usage:
    python figures/zebrafish/fig_zebrafish_mi_partition.py
    python figures/zebrafish/fig_zebrafish_mi_partition.py \\
        --run zebrafish_hd_si_ipn12_artr_pt1_position_2d_leaky
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
RECURRENT_PREFIXES = ("IPNd", "IPNds", "IPN12")
POOL_ORDER = ("angular", "shared", "displacement", "neither")
POOL_COLOR = {
    "angular":      "#1f6fb3",   # blue — heading-tuned
    "displacement": "#e07b1a",   # orange — distance-tuned
    "shared":       "#2a9d3d",   # green — both
    "neither":      "#888888",   # grey — low on both
}


def _is_recurrent(name: str) -> bool:
    # Order matters: IPN12 starts with IPN1 which would prefix-match
    # 'IPNd' if we weren't careful — use exact prefix tests.
    if name.startswith("IPN12"):
        return True
    if name.startswith("IPNds"):
        return True
    if name.startswith("IPNd"):
        return True
    return False


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",
                   default="zebrafish_hd_si_ipn12_artr_pt1_position_2d_leaky",
                   help="trained run directory basename")
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
    run_dir = os.path.join(args.data_root, "log", "zebrafish", args.run)
    print(f"[1/4] loading {run_dir}")
    net, config = _load_run(run_dir, device)
    type_names = list(net.type_names)
    nt = np.asarray(net.neuron_types).astype(int)
    N = nt.size
    is_rec = np.array([_is_recurrent(type_names[int(t)]) for t in nt])
    rec_ix = np.where(is_rec)[0]
    print(f"      N={N}, recurrent neurons: {rec_ix.size}")

    # Load test trials with task_targets slicing.
    print(f"[2/4] loading test trials and applying task_targets slicing")
    root = graphs_data_path(config.dataset)
    u_test = load_raw_array(f"{root}/test/stimulus.zarr")
    y_test = load_raw_array(f"{root}/test/target.zarr")
    _PROFILE_BY_TARGET = {
        (3, ("rotation",)):                ([0, 2, 3],    [0, 1]),
        (3, ("translation",)):             ([1],          [2]),
        (3, ("rotation", "translation")):  ([0, 1, 2, 3], [0, 1, 2]),
        (4, ("rotation",)):                ([0, 2, 3],    [0, 1]),
        (4, ("position_2d",)):             ([0, 1, 2, 3], [0, 1, 2, 3]),
    }
    _RECOGNISED = ("rotation", "translation", "position_2d")
    _task_raw = list(getattr(config.training, "task_targets", None) or [])
    _task_key = tuple(t for t in _RECOGNISED if t in _task_raw)
    if u_test.shape[-1] >= 4 and _task_key:
        key = (int(y_test.shape[-1]), _task_key)
        if key in _PROFILE_BY_TARGET:
            in_cols, out_cols = _PROFILE_BY_TARGET[key]
            u_test = u_test[..., in_cols]
            y_test = y_test[..., out_cols]
    print(f"      u={u_test.shape}  y={y_test.shape}")

    print(f"[3/4] rollout {args.n_trials} trials, computing MI per recurrent neuron")
    H, theta, d = _accumulate_hidden(net, u_test, y_test, device,
                                       args.n_trials)
    print(f"      H={H.shape}, θ samples={theta.size}, d range={d.min():.3f}..{d.max():.3f}")

    I_theta = np.zeros(rec_ix.size)
    I_d = np.zeros(rec_ix.size)
    for k, i in enumerate(rec_ix):
        x = H[:, i]
        I_theta[k] = _circular_mi(x, theta)
        I_d[k] = _mi_plugin(x, d)

    tau_theta = float(np.median(I_theta))
    tau_d = float(np.median(I_d))
    print(f"      thresholds: τθ={tau_theta:.3f}, τd={tau_d:.3f}")

    pool = np.full(rec_ix.size, "neither", dtype=object)
    pool[(I_theta >= tau_theta) & (I_d <  tau_d)] = "angular"
    pool[(I_theta <  tau_theta) & (I_d >= tau_d)] = "displacement"
    pool[(I_theta >= tau_theta) & (I_d >= tau_d)] = "shared"
    pool[(I_theta <  tau_theta) & (I_d <  tau_d)] = "neither"
    counts = {p: int(np.sum(pool == p)) for p in POOL_ORDER}
    print(f"      pool counts: {counts}")

    # ---- panel (b): edge-block magnitudes on recurrent→recurrent edges ----
    print(f"[4/4] computing edge-block magnitudes (recurrent→recurrent)")
    W_rec = net.W_rec.detach().cpu().numpy()        # (N, N), row=post, col=pre
    # Restrict to recurrent×recurrent block.
    W_rr = W_rec[np.ix_(rec_ix, rec_ix)]            # (n_rec, n_rec)
    blocks = {p: np.where(pool == p)[0] for p in POOL_ORDER}
    M = np.zeros((3, 3))                            # angular/shared/displacement
    pools_3 = ("angular", "shared", "displacement")
    for i, p_pre in enumerate(pools_3):
        for j, p_post in enumerate(pools_3):
            pre = blocks[p_pre]; post = blocks[p_post]
            if pre.size == 0 or post.size == 0:
                M[j, i] = np.nan
                continue
            sub = np.abs(W_rr[np.ix_(post, pre)])
            # Mean over non-zero (connectome support) entries.
            nz = sub[sub > 0]
            M[j, i] = float(nz.mean()) if nz.size else 0.0

    # ---- plot --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    # (a) MI scatter
    ax = axes[0]
    for p in POOL_ORDER:
        mask = pool == p
        ax.scatter(I_theta[mask], I_d[mask], s=22,
                   color=POOL_COLOR[p], edgecolor="black", linewidth=0.3,
                   alpha=0.85, label=f"{p} (n={counts[p]})")
    ax.axvline(tau_theta, color="0.4", lw=0.7, linestyle="--")
    ax.axhline(tau_d,     color="0.4", lw=0.7, linestyle="--")
    ax.set_xlabel(r"$I(\hat h_i;\,\theta)$  (bits)", fontsize=13)
    ax.set_ylabel(r"$I(\hat h_i;\,d)$  (bits)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="upper right", frameon=False)
    ax.text(-0.10, 1.04, "a", transform=ax.transAxes,
            fontsize=20, fontweight="bold")

    # (b) edge-block matrix
    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(M, cmap=cmap, vmin=0, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(pools_3, fontsize=12)
    ax.set_yticks(range(3)); ax.set_yticklabels(pools_3, fontsize=12)
    ax.set_xlabel("pre-synaptic pool", fontsize=13)
    ax.set_ylabel("post-synaptic pool", fontsize=13)
    for i in range(3):
        for j in range(3):
            val = M[j, i]
            if np.isfinite(val):
                ax.text(i, j, f"{val:.3f}",
                        ha="center", va="center", fontsize=11,
                        color="white" if val < M.max() * 0.5 else "black")
    cb = plt.colorbar(im, ax=ax, fraction=0.045)
    cb.set_label(r"mean $|\hat W^{rec}_{p\to q}|$ over edges",
                 fontsize=11)
    cb.ax.tick_params(labelsize=10)
    ax.text(-0.10, 1.04, "b", transform=ax.transAxes,
            fontsize=20, fontweight="bold")

    plt.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_mi_partition] wrote {args.out}")
    print(f"[fig_mi_partition] pool counts: {counts}")
    print(f"[fig_mi_partition] edge-block matrix (rows=post, cols=pre):")
    for row, p in zip(M, pools_3):
        print(f"  {p:13s}  {row}")


if __name__ == "__main__":
    main()
