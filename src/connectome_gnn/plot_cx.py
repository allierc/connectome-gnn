"""CX-specific visualisations: compass / PVA, EB ring fluorescence, 3-D anatomy.

These mirror the canonical fly-CX imaging panels:
  - Polar bump with PVA arrow (Hulse Fig. 1c middle, Seelig & Jayaraman 2015)
  - 2-D EB ring fluorescence donut (the GCaMP7f panel in Hulse Fig. 1c top)
  - Kinograph with overlaid HD trace (Hulse Fig. 1e)
  - Optional 3-D neuron-skeleton rendering (Hulse Fig. 1c left)

Each function is self-contained and uses only numpy / matplotlib. They are
hooked into the data-generation pipeline via plot.plot_connconstr_diagnostics,
and are also reusable from the teacher-training diagnostics script
(teachers/janelia_cx_diagnostic.py).
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Preferred-direction helpers
# ---------------------------------------------------------------------------


def cx_epg_directions(epg_ix: list[int] | np.ndarray, n_glom: int = 16) -> np.ndarray:
    """Map each EPG neuron index to its preferred direction theta in [-pi, pi).

    Args:
        epg_ix: per-EPG-neuron mapping into [0..n_glom-1] glomeruli.
        n_glom: total glomerulus count (default 16, fly EB convention).

    Returns:
        theta: (len(epg_ix),) float array in radians, in [-pi, pi).
    """
    epg_ix = np.asarray(epg_ix, dtype=int)
    return (epg_ix / float(n_glom)) * 2.0 * np.pi - np.pi


def cx_glomerulus_centres(n_glom: int = 16) -> np.ndarray:
    """Angular position of each glomerulus centre."""
    return (np.arange(n_glom) / float(n_glom)) * 2.0 * np.pi - np.pi


def cx_population_vector(
    epg_activity: np.ndarray,
    epg_theta: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute the population-vector average (PVA) over a single time frame.

    Args:
        epg_activity: (n_epg,) firing rate / fluorescence per EPG neuron.
        epg_theta:    (n_epg,) preferred direction in radians.

    Returns:
        pva_x, pva_y, pva_angle, pva_magnitude. pva_x, pva_y are the raw
        unnormalised vector components.
    """
    r = np.clip(epg_activity, 0.0, None)  # rectify to interpret as firing rate
    px = float(np.sum(r * np.cos(epg_theta)))
    py = float(np.sum(r * np.sin(epg_theta)))
    if r.sum() < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    px_norm = px / r.sum()
    py_norm = py / r.sum()
    angle = float(np.arctan2(py_norm, px_norm))
    mag = float(np.hypot(px_norm, py_norm))
    return px_norm, py_norm, angle, mag


# ---------------------------------------------------------------------------
# Compass plot — polar EPG bump + PVA arrow
# ---------------------------------------------------------------------------


def plot_cx_compass(
    voltage_history: np.ndarray,
    epg_indices: np.ndarray,
    epg_theta: np.ndarray,
    output_path: str,
    *,
    n_panels: int = 9,
    frame_indices: Optional[list[int]] = None,
    activation: str = "sigmoid",
    title: str = "EPG compass",
) -> None:
    """Render a grid of polar EPG-bump panels with PVA arrows.

    Args:
        voltage_history: (T_sampled, N) per-frame subthreshold voltage.
        epg_indices:     (n_epg,) neuron indices that are EPG.
        epg_theta:       (n_epg,) preferred direction per EPG neuron.
        output_path:     where to save the figure (.png).
        n_panels:        number of frames to render (uniformly spaced over T).
        frame_indices:   optional global frame indices for the per-panel titles.
        activation:      'sigmoid' to apply 1/(1+e^-h) to voltage before binning,
                         'relu' for max(0, h), 'none' to use raw voltage.
        title:           super-title for the figure.
    """
    voltage_history = np.asarray(voltage_history)
    if voltage_history.ndim != 2:
        raise ValueError(f"voltage_history must be (T, N); got shape {voltage_history.shape}")
    T = voltage_history.shape[0]
    n_panels = min(n_panels, T)
    panel_idx = np.linspace(0, T - 1, n_panels, dtype=int)

    epg_indices = np.asarray(epg_indices)
    epg_theta = np.asarray(epg_theta)
    if epg_indices.shape != epg_theta.shape:
        raise ValueError("epg_indices and epg_theta must have the same shape")

    # Per-neuron activity over time (only EPG rows).
    h_epg = voltage_history[:, epg_indices]
    if activation == "sigmoid":
        r_epg = 1.0 / (1.0 + np.exp(-h_epg))
    elif activation == "relu":
        r_epg = np.maximum(h_epg, 0.0)
    elif activation == "none":
        r_epg = h_epg
    else:
        raise ValueError(f"activation={activation!r} not in {{'sigmoid','relu','none'}}")

    # Bin into 16 glomeruli for the polar bar chart.
    n_glom = 16
    glom_theta = cx_glomerulus_centres(n_glom)
    # Build glomerulus assignment: each EPG -> closest glomerulus.
    glom_assign = np.argmin(
        np.abs(np.angle(np.exp(1j * (epg_theta[:, None] - glom_theta[None, :])))),
        axis=1,
    )
    # Sum activity per glomerulus, normalised by count.
    glom_act_full = np.zeros((T, n_glom), dtype=np.float32)
    for g in range(n_glom):
        mask = glom_assign == g
        if mask.sum() > 0:
            glom_act_full[:, g] = r_epg[:, mask].mean(axis=1)

    n_cols = int(np.ceil(np.sqrt(n_panels)))
    n_rows = int(np.ceil(n_panels / n_cols))
    fig = plt.figure(figsize=(2.4 * n_cols, 2.8 * n_rows))
    fig.suptitle(title, fontsize=12, y=0.98)
    width = 2.0 * np.pi / n_glom

    for k, t in enumerate(panel_idx):
        ax = fig.add_subplot(n_rows, n_cols, k + 1, projection="polar")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)

        # Polar bar chart of activity per glomerulus.
        bars = ax.bar(
            glom_theta,
            glom_act_full[t],
            width=width,
            bottom=0.0,
            color=plt.cm.Blues(np.clip(glom_act_full[t] / (glom_act_full[t].max() + 1e-9), 0, 1)),
            edgecolor="white",
            linewidth=0.5,
        )
        del bars  # quiet linter

        # PVA arrow.
        _, _, pva_angle, pva_mag = cx_population_vector(r_epg[t], epg_theta)
        if pva_mag > 1e-8:
            ax.annotate(
                "",
                xy=(pva_angle, pva_mag * glom_act_full[t].max() * 0.95),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.6),
            )

        ax.set_thetalim(-np.pi, np.pi)
        ax.set_rlim(0, max(glom_act_full[t].max() * 1.05, 1e-3))
        ax.set_xticks([0, np.pi / 2, np.pi, -np.pi / 2])
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$-\pi/2$"], fontsize=7)
        ax.set_yticklabels([])
        if frame_indices is not None and t < len(frame_indices):
            ax.set_title(f"t={frame_indices[t]}", fontsize=8, pad=4)
        else:
            ax.set_title(f"frame {t}", fontsize=8, pad=4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2-D EB ring fluorescence (donut view)
# ---------------------------------------------------------------------------


def plot_cx_eb_ring(
    voltage_history: np.ndarray,
    epg_indices: np.ndarray,
    epg_theta: np.ndarray,
    output_path: str,
    *,
    n_panels: int = 9,
    frame_indices: Optional[list[int]] = None,
    activation: str = "sigmoid",
    cmap: str = "hot",
) -> None:
    """Render a grid of 2-D EB donut panels, one per sampled frame.

    Each donut is a 16-bin annular heatmap where colour intensity = mean
    EPG activity in that glomerulus. Mirrors the GCaMP7f panel in
    Hulse Fig. 1c (top).
    """
    voltage_history = np.asarray(voltage_history)
    T = voltage_history.shape[0]
    n_panels = min(n_panels, T)
    panel_idx = np.linspace(0, T - 1, n_panels, dtype=int)

    epg_theta = np.asarray(epg_theta)
    epg_indices = np.asarray(epg_indices)
    n_glom = 16
    glom_theta = cx_glomerulus_centres(n_glom)
    glom_assign = np.argmin(
        np.abs(np.angle(np.exp(1j * (epg_theta[:, None] - glom_theta[None, :])))),
        axis=1,
    )

    h_epg = voltage_history[:, epg_indices]
    if activation == "sigmoid":
        r_epg = 1.0 / (1.0 + np.exp(-h_epg))
    elif activation == "relu":
        r_epg = np.maximum(h_epg, 0.0)
    elif activation == "none":
        r_epg = h_epg
    else:
        raise ValueError(f"activation={activation!r}")

    glom_act_full = np.zeros((T, n_glom), dtype=np.float32)
    for g in range(n_glom):
        mask = glom_assign == g
        if mask.sum() > 0:
            glom_act_full[:, g] = r_epg[:, mask].mean(axis=1)

    vmax = float(glom_act_full.max() + 1e-9)
    n_cols = int(np.ceil(np.sqrt(n_panels)))
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).ravel()

    # Donut geometry
    r_inner, r_outer = 0.55, 1.0
    theta_grid = np.linspace(-np.pi, np.pi, n_glom + 1)

    for k, t in enumerate(panel_idx):
        ax = axes[k]
        ax.set_aspect("equal")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axis("off")
        norm_v = glom_act_full[t] / vmax
        for g in range(n_glom):
            t0, t1 = theta_grid[g], theta_grid[g + 1]
            wedge = plt.matplotlib.patches.Wedge(
                center=(0, 0),
                r=r_outer,
                theta1=np.degrees(t0),
                theta2=np.degrees(t1),
                width=r_outer - r_inner,
                facecolor=plt.get_cmap(cmap)(norm_v[g]),
                edgecolor="white",
                linewidth=0.5,
            )
            ax.add_patch(wedge)
        if frame_indices is not None and t < len(frame_indices):
            ax.set_title(f"t={frame_indices[t]}", fontsize=8)
        else:
            ax.set_title(f"frame {t}", fontsize=8)

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Kinograph with HD overlay
# ---------------------------------------------------------------------------


def plot_cx_kinograph_pva(
    voltage_history: np.ndarray,
    epg_indices: np.ndarray,
    epg_theta: np.ndarray,
    output_path: str,
    *,
    activation: str = "sigmoid",
    cmap: str = "Blues",
    dt_s: float = 0.01,
    n_bins: int = 64,
    true_theta_hd: Optional[np.ndarray] = None,
    subtract_mean: bool = True,
) -> None:
    """Kinograph: time (vertical) vs orientation (horizontal) heatmap of EPG
    fluorescence + decoded PVA trace (black) and optional ground-truth HD
    trace (red), mirroring Hulse Fig. 1e and the right panels of the
    user's screenshot.

    Args:
        voltage_history: (T, N) subthreshold voltage history.
        epg_indices:     EPG neuron indices.
        epg_theta:       (n_epg,) preferred direction.
        output_path:     where to save the figure (.png).
        activation:      'sigmoid' (Hulse), 'relu', or 'none'.
        cmap:            matplotlib colormap name. Default 'Blues' for
                         raw activity; switches automatically to 'RdBu_r'
                         (divergent) when subtract_mean=True.
        dt_s:            seconds per frame (default Hulse 0.01).
        n_bins:          number of angular bins (default 64; matches Hulse panel).
        true_theta_hd:   optional (T,) ground-truth heading for overlay.
        subtract_mean:   if True (default), subtract the per-bin temporal
                         mean before colouring. Stationary baselines around
                         the ring (caused by sigmoid floor + uneven EPG
                         distribution) vanish; only the *moving* bump
                         survives.
    """
    voltage_history = np.asarray(voltage_history)
    T = voltage_history.shape[0]
    epg_theta = np.asarray(epg_theta)
    epg_indices = np.asarray(epg_indices)

    h_epg = voltage_history[:, epg_indices]
    if activation == "sigmoid":
        r_epg = 1.0 / (1.0 + np.exp(-h_epg))
    elif activation == "relu":
        r_epg = np.maximum(h_epg, 0.0)
    elif activation == "none":
        r_epg = h_epg
    else:
        raise ValueError(f"activation={activation!r}")

    # Bin into n_bins angular cells using soft assignment (Gaussian window).
    bin_centres = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
    diff = np.angle(np.exp(1j * (epg_theta[:, None] - bin_centres[None, :])))
    sigma = 2.0 * np.pi / n_bins
    weights = np.exp(-0.5 * (diff / sigma) ** 2)
    weights /= weights.sum(axis=0, keepdims=True) + 1e-12
    binned = r_epg @ weights  # (T, n_bins)

    # Decoded HD via PVA.
    decoded = np.zeros(T)
    for t in range(T):
        _, _, ang, _ = cx_population_vector(r_epg[t], epg_theta)
        decoded[t] = ang

    # Optionally subtract the per-bin temporal mean so stationary
    # baseline streaks (caused by the sigmoid floor and uneven EPG
    # distribution across glomeruli) vanish.
    if subtract_mean:
        binned_plot = binned - binned.mean(axis=0, keepdims=True)
        # Divergent colormap centred at 0.
        active_cmap = "RdBu_r" if cmap == "Blues" else cmap
        absmax = float(np.percentile(np.abs(binned_plot), 99) + 1e-9)
        vmin_p, vmax_p = -absmax, absmax
    else:
        binned_plot = binned
        active_cmap = cmap
        vmin_p, vmax_p = 0.0, float(np.percentile(binned, 99) + 1e-9)

    fig, ax = plt.subplots(figsize=(4.5, 6.0))
    ax.imshow(
        binned_plot,
        aspect="auto",
        origin="upper",
        cmap=active_cmap,
        vmin=vmin_p,
        vmax=vmax_p,
        extent=[-np.pi, np.pi, T * dt_s, 0],
        interpolation="nearest",
    )
    # Overlay decoded HD as red trace.
    ax.plot(decoded, np.arange(T) * dt_s, color="red", linewidth=1.2,
            label="decoded HD (PVA)")
    if true_theta_hd is not None:
        true_wrapped = np.angle(np.exp(1j * np.asarray(true_theta_hd)))
        ax.plot(true_wrapped, np.arange(T) * dt_s, color="black",
                linewidth=0.8, linestyle="--", label="true HD")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(T * dt_s, 0)
    ax.set_xlabel("orientation (rad)")
    ax.set_ylabel("time (s)")
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training-time snapshots: weight matrix + kinograph with HD curves
# ---------------------------------------------------------------------------


def plot_cx_matrix(
    J: np.ndarray,
    neuron_types: np.ndarray,
    type_names: list[str],
    output_path: str,
    *,
    title: str = "",
    transpose_to_post_pre: bool = True,
) -> None:
    """Plot a CX connectivity matrix with cell-type annotations.

    Args:
        J: (N, N) dense connectivity. By default treated as (pre, post)
            and transposed to (post, pre) for the plot (neuroscience
            convention: rows = postsynaptic, cols = presynaptic).
        neuron_types: (N,) int per-neuron type indices.
        type_names: type-name strings indexed by neuron_types values.
        output_path: PNG file to write.
        title: optional super-title.
        transpose_to_post_pre: if True (default), plot J.T.
    """
    J_plot = J.T if transpose_to_post_pre else J
    N = J_plot.shape[0]
    nonzero = np.abs(J_plot)[np.abs(J_plot) > 0]
    vmax = float(np.percentile(nonzero, 98)) if nonzero.size else 1.0

    # Cell-type boundary detection (assumes neurons are grouped contiguously).
    bounds, centres, labels = [0], [], []
    cur_t, cur_start = int(neuron_types[0]), 0
    for i, t in enumerate(neuron_types):
        t = int(t)
        if t != cur_t:
            bounds.append(i)
            centres.append((cur_start + i - 1) / 2.0)
            labels.append(type_names[cur_t])
            cur_t, cur_start = t, i
    bounds.append(len(neuron_types))
    centres.append((cur_start + len(neuron_types) - 1) / 2.0)
    labels.append(type_names[cur_t])

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    im = ax.imshow(J_plot, cmap="bwr_r", vmin=-vmax, vmax=vmax,
                   aspect="equal", interpolation="nearest", origin="upper")
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color="k", linewidth=0.4, alpha=0.6)
        ax.axvline(b - 0.5, color="k", linewidth=0.4, alpha=0.6)
    ax.set_xticks(centres)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(centres)
    ax.set_yticklabels(labels, fontsize=7)
    nnz = int((np.abs(J_plot) > 0).sum())
    sub = f"N={N}, nonzero={nnz}, vmax={vmax:.3f}"
    ax.set_title((title + "\n" if title else "") + sub, fontsize=10)
    ax.set_xlabel("presynaptic")
    ax.set_ylabel("postsynaptic")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cx_training_snapshot(
    W_rec: np.ndarray,
    rollout: dict,
    epg_theta: np.ndarray,
    output_path: str,
    *,
    neuron_types: Optional[np.ndarray] = None,
    type_names: Optional[list[str]] = None,
    step: Optional[int] = None,
    dt_s: float = 0.01,
    n_bins: int = 32,
    mat_vmax: float = 1.0,
    fwhm_z_thresh: float = 1.0,
) -> None:
    """Combined W_rec matrix + kinograph for one training-time snapshot.

    Args:
        W_rec: (N, N) current learned recurrent matrix.
        rollout: dict produced by save_rollout (must include
            r_epg, true_theta, decoded_theta).
        epg_theta: (n_epg,) preferred direction per EPG neuron.
        output_path: PNG file to write.
        neuron_types, type_names: optional, for matrix annotation.
        step: optional training step (for the title).
        dt_s: seconds per frame in the rollout.
        n_bins: number of angular bins for the kinograph (default 32, Hulse).
        mat_vmax: fixed colorbar limit for the W_rec matrix (default 1.0,
            so colors are comparable across snapshots during training).
        fwhm_z_thresh: per-frame z-score threshold used to compute the bump
            FWHM annotation on the angle-binned panel (default 1.0).
    """
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0],
                          height_ratios=[1.0, 1.0],
                          hspace=0.22, wspace=0.22,
                          left=0.06, right=0.94, top=0.93, bottom=0.05)
    ax_mat = fig.add_subplot(gs[0, 0])
    ax_kin = fig.add_subplot(gs[0, 1])
    ax_neu = fig.add_subplot(gs[1, 0])
    # Bottom-right cell becomes a 3x3 of polar axes (defaults: full 360°).
    gs_polar = gs[1, 1].subgridspec(3, 3, hspace=0.45, wspace=0.30)
    polar_axes = [
        fig.add_subplot(gs_polar[i, j], projection="polar")
        for i in range(3) for j in range(3)
    ]

    # ---- LEFT: W_rec matrix ----
    J = W_rec.T if W_rec.shape[0] == W_rec.shape[1] else W_rec
    im = ax_mat.imshow(J, cmap="bwr_r", vmin=-mat_vmax, vmax=mat_vmax,
                       aspect="equal", interpolation="nearest", origin="upper")
    if neuron_types is not None and type_names is not None:
        bounds, centres, labels = [0], [], []
        cur_t, cur_start = int(neuron_types[0]), 0
        for i, t in enumerate(neuron_types):
            t = int(t)
            if t != cur_t:
                bounds.append(i)
                centres.append((cur_start + i - 1) / 2.0)
                labels.append(type_names[cur_t])
                cur_t, cur_start = t, i
        bounds.append(len(neuron_types))
        centres.append((cur_start + len(neuron_types) - 1) / 2.0)
        labels.append(type_names[cur_t])
        for b in bounds[1:-1]:
            ax_mat.axhline(b - 0.5, color="k", linewidth=0.4, alpha=0.5)
            ax_mat.axvline(b - 0.5, color="k", linewidth=0.4, alpha=0.5)
        ax_mat.set_xticks(centres)
        ax_mat.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax_mat.set_yticks(centres)
        ax_mat.set_yticklabels(labels, fontsize=7)
    ax_mat.set_title(f"learned W_rec   vmax=±{mat_vmax:.3f}", fontsize=10)
    ax_mat.set_xlabel("presyn"); ax_mat.set_ylabel("postsyn")
    cb_mat = fig.colorbar(im, ax=ax_mat, fraction=0.04, pad=0.02, shrink=0.8)
    cb_mat.ax.tick_params(labelsize=7)

    # ---- RIGHT: kinograph with HD curves ----
    # We plot the per-frame z-scored angular bump (mean=0, std=1 across
    # the n_bins angular bins at each timestep). This makes the colorbar
    # directly interpretable — yellow ≈ "k σ above the trial-mean" — so
    # the FWHM threshold (a horizontal mark on the colorbar) shows
    # visually where the bump-edge cutoff lies.
    r_epg = rollout["r_epg"]  # (T, n_epg)
    T = r_epg.shape[0]
    bin_centres = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
    diff = np.angle(np.exp(1j * (epg_theta[:, None] - bin_centres[None, :])))
    sigma = 2 * np.pi / n_bins
    w = np.exp(-0.5 * (diff / sigma) ** 2)
    w /= w.sum(axis=0, keepdims=True) + 1e-12
    binned = r_epg @ w                                    # (T, n_bins)
    mu = binned.mean(axis=1, keepdims=True)
    sd = binned.std(axis=1, keepdims=True) + 1e-12
    z = (binned - mu) / sd                                # (T, n_bins)
    z_max = float(max(np.percentile(np.abs(z), 99), fwhm_z_thresh + 0.5))
    im_kin = ax_kin.imshow(z, aspect="auto", origin="upper", cmap="RdBu_r",
                           vmin=-z_max, vmax=z_max,
                           extent=[-np.pi, np.pi, T * dt_s, 0],
                           interpolation="nearest")
    cb_kin = fig.colorbar(im_kin, ax=ax_kin, fraction=0.04, pad=0.02, shrink=0.85)
    cb_kin.ax.tick_params(labelsize=7)
    cb_kin.set_label("z-score (per-frame, across angular bins)", fontsize=7)
    # Mark the FWHM threshold on the colorbar so "z>1" has a visual anchor.
    cb_kin.ax.axhline(fwhm_z_thresh, color="black", linewidth=0.8)

    # Scatter overlay: dense dots avoid the horizontal-jump artefact at ±π
    # without needing wrap-aware NaN insertion.
    def _scatter(theta, time, color, size, label):
        theta = np.angle(np.exp(1j * np.asarray(theta)))  # ensure (-π, π]
        ax_kin.scatter(theta, time, s=size, c=color, marker=".",
                       linewidths=0, label=label)
    t_axis = np.arange(T) * dt_s
    _scatter(rollout["true_theta"], t_axis, "black", 4, "true HD")
    _scatter(rollout["decoded_theta"], t_axis, "red", 3, "decoded HD (W_out)")
    ax_kin.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax_kin.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax_kin.set_xlabel("orientation (rad)")
    ax_kin.set_ylabel("time (s)")

    # FWHM = mean (over time) of the angular width where the per-frame
    # z-scored bump exceeds `fwhm_z_thresh`. Computed on the same z-scored
    # signal that's plotted, so the annotation matches the panel.
    bin_rad = 2 * np.pi / n_bins
    widths = []
    c = n_bins // 2
    for t in range(T):
        v = z[t]
        peak = int(np.argmax(v))
        if v[peak] <= fwhm_z_thresh:
            continue
        v_rolled = np.roll(v, c - peak)
        left = c
        while left - 1 >= 0 and v_rolled[left - 1] > fwhm_z_thresh:
            left -= 1
        right = c
        while right + 1 < n_bins and v_rolled[right + 1] > fwhm_z_thresh:
            right += 1
        widths.append((right - left + 1) * bin_rad)
    fwhm_rad = float(np.mean(widths)) if widths else float("nan")
    fwhm_str = (f"bump width={np.degrees(fwhm_rad):.0f}°"
                if widths else "bump width=n/a")
    # pi_acc on the snapshot rollout = mean cos(decoded - true) after a
    # short warmup (matches `path_integration_accuracy()` definition).
    warmup = min(10, T // 4)
    diff = np.angle(np.exp(1j * (np.asarray(rollout["decoded_theta"][warmup:])
                                 - np.asarray(rollout["true_theta"][warmup:]))))
    pi_acc = float(np.cos(diff).mean()) if diff.size else float("nan")
    ax_kin.set_title(
        f"EPG kinograph (z-scored)  —  "
        f"{fwhm_str} above z={fwhm_z_thresh:g}  —  pi_acc={pi_acc:.3f}",
        fontsize=8,
    )

    # ---- RIGHT: per-neuron EPG kinograph (sorted by preferred HD) ----
    # Rows are individual EPG neurons (no angular smoothing). This panel
    # exposes synchrony within Hulse-style "dynamical clones" — neurons
    # whose preferred HD differs by < 5° (Hulse Methods, Fig. 2 S3) form
    # a clone group and should fire together. Thin separators mark the
    # boundaries between groups.
    n_epg = r_epg.shape[1]
    vmax_neu = float(np.percentile(r_epg, 99) + 1e-9)
    # Time on y-axis (matches ax_kin) so both panels can be read together.
    ax_neu.imshow(
        r_epg, aspect="auto", origin="upper", cmap="viridis",
        vmin=0.0, vmax=vmax_neu,
        extent=[-0.5, n_epg - 0.5, T * dt_s, 0], interpolation="nearest",
    )
    ax_neu.set_xlabel("EPG neuron index")
    ax_neu.set_ylabel("time (s)")
    ax_neu.set_title("per-neuron EPG (raw firing rate)", fontsize=10)

    # ---- BOTTOM-RIGHT: 3x3 polar snapshots at 9 sampled HD values ----
    # For each target HD, find the rollout frame whose true_theta is
    # closest, then plot EPG (outer ring, r=1.0) and PEN (inner ring,
    # r=0.6) firing rates as colored dots placed at the neuron's
    # preferred angle. PEN preferred angles are unknown — we use a
    # uniform ring in connectome-order as a stand-in (matches the
    # ordering used by the circular-TV regulariser).
    true_theta_arr = np.angle(np.exp(1j * np.asarray(rollout["true_theta"])))
    target_angles = np.linspace(-np.pi, np.pi, 9, endpoint=False)
    r_pen = rollout.get("r_pen")  # (T, n_pen) or None
    n_pen = r_pen.shape[1] if r_pen is not None else 0
    pen_theta = (np.linspace(-np.pi, np.pi, n_pen, endpoint=False)
                 if n_pen > 0 else None)
    epg_theta_wrapped = np.angle(np.exp(1j * np.asarray(epg_theta)))

    # Shared colour scale across all 9 cells so brightness is comparable.
    polar_vmax = float(np.percentile(
        np.concatenate([r_epg.ravel(), r_pen.ravel()]) if r_pen is not None
        else r_epg.ravel(), 99) + 1e-9)
    # Use scatter-with-cmap; we need a mappable for the shared colourbar.
    polar_scatter = None
    epg_radius, pen_radius = 1.0, 0.6
    for k, target in enumerate(target_angles):
        ax = polar_axes[k]
        # Closest frame in the rollout to this target HD.
        dist = np.abs(np.angle(np.exp(1j * (true_theta_arr - target))))
        t_idx = int(np.argmin(dist))
        # EPG outer ring.
        sc_e = ax.scatter(
            epg_theta_wrapped, np.full(n_epg, epg_radius),
            c=r_epg[t_idx], cmap="viridis", vmin=0.0, vmax=polar_vmax,
            s=14, edgecolors="none",
        )
        # PEN inner ring (if available).
        if r_pen is not None:
            ax.scatter(
                pen_theta, np.full(n_pen, pen_radius),
                c=r_pen[t_idx], cmap="viridis", vmin=0.0, vmax=polar_vmax,
                s=10, edgecolors="none",
            )
        # Reference line at the actual frame HD (not the target).
        actual = true_theta_arr[t_idx]
        ax.plot([actual, actual], [0, 1.15], color="black", linewidth=0.8)
        ax.set_ylim(0, 1.25)
        ax.set_yticks([])
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels(["0°", "90°", "±180°", "−90°"], fontsize=6)
        ax.tick_params(pad=0)
        ax.set_title(f"θ={np.degrees(actual):+.0f}°", fontsize=8, pad=4)
        ax.grid(alpha=0.25, linewidth=0.4)
        if polar_scatter is None:
            polar_scatter = sc_e
    # Single colorbar anchored to the right column of the 3x3 grid.
    cbar = fig.colorbar(polar_scatter, ax=[polar_axes[2], polar_axes[5], polar_axes[8]],
                        fraction=0.05, pad=0.04, shrink=0.85)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("firing rate", fontsize=7)
    # Group title: position above the top row of the polar grid using the
    # gridspec coordinates of the upper-right block.
    pos = gs[1, 1].get_position(fig)
    fig.text(
        (pos.x0 + pos.x1) / 2.0, pos.y1 + 0.005,
        f"EPG (outer) + PEN (inner) at 9 HDs"
        + (f"  —  n_pen={n_pen}" if n_pen else "  —  no PEN data"),
        ha="center", fontsize=9,
    )

    title = f"training snapshot   step={step}" if step is not None else "training snapshot"
    fig.suptitle(title, fontsize=12)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3-D anatomy (cached-skeletons)
# ---------------------------------------------------------------------------


def plot_cx_anatomy_3d(
    output_path: str,
    *,
    neuron_types: Optional[np.ndarray] = None,
    type_names: Optional[list[str]] = None,
    epg_ix: Optional[list[int]] = None,
    anatomy_dir: str = "papers/janelia_cx/anatomy",
    elev: float = 22.0,
    azim: float = -55.0,
    edge_index: Optional[np.ndarray] = None,
    edge_weights: Optional[np.ndarray] = None,
    n_edge_draw: int = 200,
) -> bool:
    """3-D view of the CX neurons coloured by cell type.

    Two rendering paths:
    (1) If `anatomy_dir` contains per-type `<type_name>.npz` cache files
        with a `'coords'` (P, 3) array, render real skeleton points.
    (2) Otherwise, render a **synthetic schematic** built from the
        connectome assignments: EPG neurons on a ring (EB), PEN on a
        bar above (PB), Delta7 / PEG / EPGt as stylised side groups.
        This always succeeds and gives a useful figure offline.

    Args:
        output_path:    where to save the .png.
        neuron_types:   (N,) int array of per-neuron type indices. Required
                        for the synthetic path.
        type_names:     list mapping type index -> name. Required for
                        synthetic path.
        epg_ix:         (n_epg,) glomerulus assignment for EPG neurons
                        (0..15). Used to place EPGs on the EB ring.
        edge_index, edge_weights: optional (2, E) / (E,) tensors. If
                        provided, draw up to `n_edge_draw` strongest
                        edges as semi-transparent lines (red=excitatory,
                        blue=inhibitory).

    Returns:
        True if real cached skeletons were used; False if the synthetic
        schematic was rendered instead.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — register 3D projection

    fig = plt.figure(figsize=(6.0, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    # --- Path 1: real cached skeletons ----------------------------------
    cached_path = (
        os.path.isdir(anatomy_dir)
        and any(f.endswith(".npz") for f in os.listdir(anatomy_dir))
    )
    if cached_path:
        if type_names is None:
            type_names = [
                os.path.splitext(f)[0] for f in sorted(os.listdir(anatomy_dir))
                if f.endswith(".npz")
            ]
        colours = plt.cm.tab10(np.linspace(0, 1, max(1, len(type_names))))
        plotted_any = False
        for i, tn in enumerate(type_names):
            path = os.path.join(anatomy_dir, f"{tn}.npz")
            if not os.path.isfile(path):
                continue
            coords = np.load(path)["coords"]
            if coords.size == 0:
                continue
            ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2],
                s=1.5, c=[colours[i]], alpha=0.6, label=tn, depthshade=True,
            )
            plotted_any = True
        if plotted_any:
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
            plt.tight_layout()
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return True

    # --- Path 2: synthetic schematic -----------------------------------
    if neuron_types is None or type_names is None:
        fig.clf()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5,
                "3D anatomy unavailable\n"
                "(pass neuron_types + type_names, "
                "or cache skeletons in papers/janelia_cx/anatomy/)",
                ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return False

    nt = np.asarray(neuron_types).astype(int)
    n_glom = 16
    coords = np.zeros((nt.size, 3), dtype=np.float32)
    rng = np.random.default_rng(0)

    # Helper to find type indices by name fragment.
    def _idx_of(frag: str) -> list[int]:
        return [i for i, n in enumerate(type_names) if frag in n]

    epg_types = _idx_of("EPG")
    pen_types = _idx_of("PEN")
    d7_types = _idx_of("Delta7") + _idx_of("Δ7")
    peg_types = _idx_of("PEG") if all("PE" in type_names[i] for i in []) else _idx_of("PEG")
    # PEG/EPG name collision guard: PEG must not include EPG matches.
    peg_types = [i for i in peg_types if "EPG" not in type_names[i]]
    epg_types = [i for i in epg_types if "PEG" not in type_names[i]]

    # --- EPG on the EB ring (radius R_eb in xy at z=0) -------------------
    R_eb = 4.5
    eb_jitter = 0.25
    epg_mask = np.isin(nt, epg_types)
    epg_indices = np.where(epg_mask)[0]
    if epg_ix is not None and len(epg_ix) == len(epg_indices):
        theta_epg = cx_epg_directions(epg_ix, n_glom=n_glom)
    elif len(epg_indices) > 0:
        # Fall back: spread EPGs uniformly on the ring.
        theta_epg = np.linspace(-np.pi, np.pi, len(epg_indices), endpoint=False)
    else:
        theta_epg = np.array([])
    for j, idx in enumerate(epg_indices):
        th = float(theta_epg[j])
        r = R_eb + rng.normal(0, eb_jitter)
        coords[idx] = [r * np.cos(th), r * np.sin(th),
                       rng.normal(0, eb_jitter)]

    # --- PEN on a horizontal bar (PB) -----------------------------------
    PB_y = 7.5
    PB_z = 4.0
    pen_mask = np.isin(nt, pen_types)
    pen_indices = np.where(pen_mask)[0]
    n_pen = pen_indices.size
    if n_pen > 0:
        for j, idx in enumerate(pen_indices):
            # Half left, half right of midline.
            side = -1.0 if j < n_pen // 2 else 1.0
            inner = (j % (n_pen // 2 + 1)) / max(1, n_pen // 2)
            x = side * (1.5 + inner * 6.0)
            coords[idx] = [x, PB_y + rng.normal(0, 0.2),
                           PB_z + rng.normal(0, 0.2)]

    # --- Delta7 across the PB midline -----------------------------------
    d7_mask = np.isin(nt, d7_types)
    d7_indices = np.where(d7_mask)[0]
    for j, idx in enumerate(d7_indices):
        x = np.linspace(-7.5, 7.5, max(1, len(d7_indices)))[j]
        coords[idx] = [x, PB_y - 1.0, PB_z + 1.5]

    # --- PEG between PB and EB ------------------------------------------
    peg_mask = np.isin(nt, peg_types)
    peg_indices = np.where(peg_mask)[0]
    for j, idx in enumerate(peg_indices):
        th = (j / max(1, len(peg_indices))) * 2.0 * np.pi - np.pi
        r = R_eb * 0.7
        coords[idx] = [r * np.cos(th), r * np.sin(th) + PB_y * 0.45,
                       PB_z * 0.5]

    # --- Anything else: stack in a small cloud near the origin ----------
    other_mask = ~(epg_mask | pen_mask | d7_mask | peg_mask)
    other_indices = np.where(other_mask)[0]
    for idx in other_indices:
        coords[idx] = rng.normal(0, 1.5, size=3)
        coords[idx, 1] += PB_y * 0.5

    # --- Scatter, coloured by cell type --------------------------------
    type_colours = plt.cm.tab10(np.linspace(0, 1, max(1, len(type_names))))
    for t_idx in range(len(type_names)):
        mask = nt == t_idx
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            s=55, c=[type_colours[t_idx]], edgecolors="white",
            linewidths=0.5, alpha=0.95, depthshade=True,
            label=type_names[t_idx],
        )

    # --- Optional edge overlay (strongest only) -------------------------
    if edge_index is not None and edge_weights is not None:
        ei = np.asarray(edge_index)
        ew = np.asarray(edge_weights)
        # Pick the top-|w| edges.
        order = np.argsort(-np.abs(ew))[:n_edge_draw]
        for k in order:
            src, dst = int(ei[0, k]), int(ei[1, k])
            w = float(ew[k])
            colour = "tab:red" if w > 0 else "tab:blue"
            ax.plot(
                [coords[src, 0], coords[dst, 0]],
                [coords[src, 1], coords[dst, 1]],
                [coords[src, 2], coords[dst, 2]],
                color=colour, alpha=0.12, linewidth=0.6,
            )

    # --- Annotate EB / PB --------------------------------------------
    ax.text(0, 0, -1.5, "EB", fontsize=10, ha="center")
    if n_pen > 0 or d7_indices.size > 0:
        ax.text(0, PB_y, PB_z + 2.0, "PB", fontsize=10, ha="center")

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_box_aspect((1, 1, 0.65))
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return False
