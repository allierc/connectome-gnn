"""CX-specific visualisations: compass / PVA, EB ring fluorescence, 3-D anatomy.

These mirror the canonical fly-CX imaging panels (polar bump with PVA arrow,
2-D EB ring fluorescence donut, kinograph with overlaid HD trace, optional
3-D neuron-skeleton rendering).

Each function is self-contained and uses only numpy / matplotlib. They are
hooked into the data-generation pipeline via plot.plot_connconstr_diagnostics,
and are also reusable from the teacher-training diagnostics script
(teachers/janelia_cx_diagnostic.py).
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec


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
    EPG activity in that glomerulus.
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
    trace (red).

    Args:
        voltage_history: (T, N) subthreshold voltage history.
        epg_indices:     EPG neuron indices.
        epg_theta:       (n_epg,) preferred direction.
        output_path:     where to save the figure (.png).
        activation:      'sigmoid', 'relu', or 'none'.
        cmap:            matplotlib colormap name. Default 'Blues' for
                         raw activity; switches automatically to 'RdBu_r'
                         (divergent) when subtract_mean=True.
        dt_s:            seconds per frame (default 0.01).
        n_bins:          number of angular bins (default 64).
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
    # Overlay true HD (thick light green) under decoded HD (thin black).
    if true_theta_hd is not None:
        true_wrapped = np.angle(np.exp(1j * np.asarray(true_theta_hd)))
        ax.plot(true_wrapped, np.arange(T) * dt_s, color="#4daf4a",
                linewidth=2.4, label="true HD")
    ax.plot(decoded, np.arange(T) * dt_s, color="black", linewidth=0.6,
            label="decoded HD (PVA)")
    if true_theta_hd is not None:
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


def render_cx_snapshot_into_axes(
    fig,
    ax_gt,
    ax_mat,
    ax_kin,
    ax_neu,
    ax_pen,
    ax_hd,
    *,
    W_rec: np.ndarray,
    rollout: dict,
    epg_theta: np.ndarray,
    W_con: Optional[np.ndarray] = None,
    neuron_types: Optional[np.ndarray] = None,
    type_names: Optional[list[str]] = None,
    pen_neuron_types: Optional[np.ndarray] = None,
    dt_s: float = 0.01,
    n_bins: int = 32,
    fwhm_z_thresh: float = 1.0,
) -> None:
    """Render the 6 CX training-snapshot panels into externally-provided axes.

    Same layout/contents as `plot_cx_training_snapshot` but driven by the
    caller's figure and axes — used to embed the snapshot inside a larger
    composite figure.
    """
    # ---- shared matrix renderer (used for W_con and W_rec) ----
    # Z-scored over the non-zero entries (raw |W| spans 4 orders of
    # magnitude — fixed ±vmax makes the small entries invisible). Clipped
    # to ±3 σ so the colour scale is comparable across snapshots and
    # between W_con and W_rec.
    def _render_matrix(ax, M, title, tick_fs: int = 7):
        # M is [post, pre] (loader convention: J_effective[post, pre], with
        # Dale enforced on COLS = pre). Display without transpose so the
        # axes match the xlabel/ylabel below and Beiran fig 5d:
        #   y = row of M = post, x = col of M = pre, Dale visible on cols.
        J = M
        J_arr = np.asarray(J, dtype=np.float32)
        nz = np.abs(J_arr[J_arr != 0])
        # Zero-centred, sign-preserving scale so EVERY non-zero edge gets a
        # clear colour (red = excitatory, blue = inhibitory) instead of the
        # bulk washing out to white. A z-score centres the *most common*
        # weight at white and the 5x-amplified inhibitory tail inflates the
        # spread, so the matrix looked empty. The sqrt compresses that heavy
        # tail so small and large edges are both visible; the 90th-percentile
        # magnitude maps to full saturation.
        scale = float(np.percentile(nz, 90)) if nz.size else 1.0
        Z = np.sign(J_arr) * np.sqrt(np.abs(J_arr) / (scale + 1e-12))
        z_max = 1.0
        Z = np.clip(Z, -z_max, z_max)
        # Dilate each 1-px edge into a blob so individual synapses are visible
        # at panel resolution (a 731-839 node matrix shown this small renders
        # single edges sub-pixel and washes out). The blob size scales with
        # the matrix so the edges stay visible regardless of N. Signed:
        # positive entries via max-filter, negative via min-filter, larger
        # magnitude wins where blobs overlap (as in fig_connectome_summary).
        from scipy.ndimage import maximum_filter, minimum_filter
        blob = max(5, int(round(J_arr.shape[0] / 130.0)))  # ~5-7 px for N=731-839
        Zpos = maximum_filter(np.where(Z > 0, Z, 0.0), size=blob)
        Zneg = minimum_filter(np.where(Z < 0, Z, 0.0), size=blob)
        Zvis = np.where(np.abs(Zpos) >= np.abs(Zneg), Zpos, Zneg)
        im = ax.imshow(Zvis, cmap="RdBu_r", vmin=-z_max, vmax=z_max,
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
                ax.axhline(b - 0.5, color="k", linewidth=0.4, alpha=0.5)
                ax.axvline(b - 0.5, color="k", linewidth=0.4, alpha=0.5)
            ax.set_xticks(centres)
            ax.set_xticklabels(labels, fontsize=tick_fs, rotation=45, ha="right")
            ax.set_yticks(centres)
            ax.set_yticklabels(labels, fontsize=tick_fs)
        if title:
            ax.set_title(title, fontsize=8)
        ax.set_xlabel("presynaptic"); ax.set_ylabel("postsynaptic")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.8)
        cb.set_label(r"sign$\cdot\sqrt{|W|}$ (norm.)", fontsize=10)
        cb.ax.tick_params(labelsize=9)

    # ---- (0,0) GT W_con (reference) + (0,1) learned W_rec ----
    if W_con is not None:
        _render_matrix(ax_gt, W_con, "GT W_con (signed, $\\sqrt{}$-scaled)")
    else:
        ax_gt.text(0.5, 0.5, "no W_con provided", ha="center", va="center",
                   transform=ax_gt.transAxes, fontsize=11, color="0.5")
        ax_gt.set_xticks([]); ax_gt.set_yticks([])
        ax_gt.set_title("GT W_con", fontsize=8)
    _render_matrix(ax_mat, W_rec, "", tick_fs=7)

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
    # Fixed ±3 σ across snapshots so colour-shift between snapshots reflects
    # actual dynamics, not changing percentile floors.
    z_max = 3.0
    z_clipped = np.clip(z, -z_max, z_max)
    im_kin = ax_kin.imshow(z_clipped.T, aspect="auto", origin="lower", cmap="RdBu_r",
                           vmin=-z_max, vmax=z_max,
                           extent=[0, T * dt_s, -np.pi, np.pi],
                           interpolation="nearest")
    cb_kin = fig.colorbar(im_kin, ax=ax_kin, fraction=0.04, pad=0.02, shrink=0.85)
    cb_kin.ax.tick_params(labelsize=9)
    cb_kin.set_label("z-score", fontsize=11)
    # Mark the FWHM threshold on the colorbar so "z>1" has a visual anchor.
    cb_kin.ax.axhline(fwhm_z_thresh, color="black", linewidth=0.8)

    # Scatter overlay: dense dots avoid the horizontal-jump artefact at ±π
    # without needing wrap-aware NaN insertion.
    def _scatter(theta, time, color, size, label):
        theta = np.angle(np.exp(1j * np.asarray(theta)))  # ensure (-π, π]
        ax_kin.scatter(time, theta, s=size, c=color, marker=".",
                       linewidths=0, label=label)
    t_axis = np.arange(T) * dt_s
    _scatter(rollout["true_theta"], t_axis, "#4daf4a", 6, "true HD")
    _scatter(rollout["decoded_theta"], t_axis, "black", 2, "decoded HD (W_out)")
    ax_kin.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax_kin.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax_kin.set_xlabel("time (s)")
    ax_kin.set_ylabel("orientation (rad)")

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
    # Rows are individual EPG neurons (no angular smoothing). Exposes
    # synchrony within "dynamical clone" groups (neurons whose preferred
    # HD differs by < 5°), which should fire together. Thin separators
    # mark the boundaries between groups.
    n_epg = r_epg.shape[1]
    # Per-frame z-score across the n_epg neurons (matches the kinograph's
    # per-frame normalisation so the two panels are directly comparable).
    epg_mu = r_epg.mean(axis=1, keepdims=True)
    epg_sd = r_epg.std(axis=1, keepdims=True) + 1e-12
    z_epg = np.clip((r_epg - epg_mu) / epg_sd, -3.0, 3.0)
    im_neu = ax_neu.imshow(
        z_epg.T, aspect="auto", origin="lower", cmap="RdBu_r",
        vmin=-3.0, vmax=3.0,
        extent=[0, T * dt_s, -0.5, n_epg - 0.5], interpolation="nearest",
    )
    cb_neu = fig.colorbar(im_neu, ax=ax_neu, fraction=0.04, pad=0.02, shrink=0.85)
    cb_neu.set_label("z-score", fontsize=11)
    cb_neu.ax.tick_params(labelsize=9)
    ax_neu.set_xlabel("time (s)")
    ax_neu.set_ylabel("EPG neuron index")
    ax_neu.set_title("per-neuron EPG (z-scored, $\\pm 3\\,\\sigma$)", fontsize=8)

    # ---- BOTTOM-RIGHT: per-neuron PEN raster (mirror of bottom-left) ----
    # Twin of the EPG raster on the left so the user can read both bumps on
    # the same time axis. PEN_a / PEN_b receive ω from the noduli and re-enter
    # the EB shifted by ~one PB glomerulus (Turner-Evans 2017),
    # so the bump should *track* EPG with a velocity-dependent offset that
    # this side-by-side view exposes directly.
    #
    # Caveat on x-ordering: PEN preferred angles depend on the empirical
    # PEN-PB-glomerulus map, which we don't load here. We sort by connectome
    # index — the same order the circular-TV regulariser already uses
    # (Beiran's loader sorts by neuPrint instance, which is approximately
    # PB-glomerulus-ordered). An "outer ring / inner ring" polar overlay would
    # be misleading without the proper angular calibration; the raster is
    # honest about the ordering.
    r_pen = rollout.get("r_pen")  # (T, n_pen) or None
    if r_pen is not None and r_pen.shape[1] > 0:
        n_pen = r_pen.shape[1]
        pen_mu = r_pen.mean(axis=1, keepdims=True)
        pen_sd = r_pen.std(axis=1, keepdims=True) + 1e-12
        z_pen = np.clip((r_pen - pen_mu) / pen_sd, -3.0, 3.0)
        im_pen = ax_pen.imshow(
            z_pen.T, aspect="auto", origin="lower", cmap="RdBu_r",
            vmin=-3.0, vmax=3.0,
            extent=[0, T * dt_s, -0.5, n_pen - 0.5], interpolation="nearest",
        )
        cb_pen = fig.colorbar(im_pen, ax=ax_pen, fraction=0.04, pad=0.02, shrink=0.85)
        cb_pen.set_label("z-score", fontsize=11)
        cb_pen.ax.tick_params(labelsize=9)
        ax_pen.set_xlabel("time (s)")
        # Y-axis: cell-type-name ticks at each block centre if PEN subtypes
        # are provided (e.g. PEN_a(PEN1) / PEN_b(PEN2)). Falls back to a
        # plain "neuron index" label.
        if (pen_neuron_types is not None and type_names is not None
                and len(pen_neuron_types) == n_pen):
            pt = np.asarray(pen_neuron_types).astype(np.int64)
            bounds, centres, labels = [0], [], []
            cur_t, cur_start = int(pt[0]), 0
            for i, t in enumerate(pt):
                t = int(t)
                if t != cur_t:
                    bounds.append(i)
                    centres.append((cur_start + i - 1) / 2.0)
                    labels.append(type_names[cur_t])
                    cur_t, cur_start = t, i
            bounds.append(n_pen)
            centres.append((cur_start + n_pen - 1) / 2.0)
            labels.append(type_names[cur_t])
            for b in bounds[1:-1]:
                ax_pen.axhline(b - 0.5, color="k", linewidth=0.4, alpha=0.5)
            ax_pen.set_yticks(centres)
            ax_pen.set_yticklabels(labels, fontsize=7)
            ax_pen.set_ylabel("")
        else:
            ax_pen.set_ylabel("PEN neuron index (connectome order ≈ PB glomerulus)")
        ax_pen.set_title(
            f"per-neuron PEN (z-scored, $\\pm 3\\,\\sigma$,  n_pen={n_pen})",
            fontsize=8,
        )
    else:
        ax_pen.text(0.5, 0.5, "no PEN data", ha="center", va="center",
                    transform=ax_pen.transAxes, fontsize=11, color="0.5")
        ax_pen.set_xticks([]); ax_pen.set_yticks([])
        ax_pen.set_title("per-neuron PEN", fontsize=8)

    # ---- (2,1) decoded vs true HD + residual error ----
    # Previously this panel used a twin y-axis with unwrapped HD on the
    # right — unwrapped HD grows linearly with ω·t and trivially dominated
    # the axis, hiding the omega trace and any tracking error. Replaced
    # with a single bounded axis showing wrapped HD in (−π, π) and the
    # circular residual (decoded − true) so the heading error is directly
    # readable on the same scale.
    t_axis = np.arange(T) * dt_s
    true_hd = np.angle(np.exp(1j * np.asarray(rollout["true_theta"])))
    dec_hd  = np.angle(np.exp(1j * np.asarray(rollout["decoded_theta"])))
    err_hd  = np.angle(np.exp(1j * (np.asarray(rollout["decoded_theta"])
                                     - np.asarray(rollout["true_theta"]))))
    ax_hd.plot(t_axis, true_hd, color="#4daf4a", lw=0.0,
                marker=".", ms=3.0, ls="", label="true HD")
    ax_hd.plot(t_axis, dec_hd, color="black", lw=0.0,
                marker=".", ms=1.0, ls="", label="decoded HD")
    ax_hd.plot(t_axis, err_hd, color="C0", lw=0.8, alpha=0.45,
                label="error (dec − true)")
    ax_hd.axhline(0.0, color="0.6", lw=0.4)
    ax_hd.set_xlabel("time (s)")
    ax_hd.set_ylabel("heading (rad, wrapped)")
    ax_hd.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax_hd.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax_hd.set_ylim(-np.pi - 0.15, np.pi + 0.15)
    rmse_deg = float(np.degrees(np.sqrt(np.mean(err_hd ** 2))))
    # Pearson over the plotted rollout (skip the first 10 frames to match the
    # `_rollout_heading_metrics` warmup convention).
    _warm = 10
    _true_full = np.asarray(rollout["true_theta"])
    _dec_full = np.asarray(rollout["decoded_theta"])
    if _true_full.size > _warm:
        _dec_unwrap = np.unwrap(_dec_full[_warm:])
        _true_post = _true_full[_warm:]
        if _dec_unwrap.std() > 1e-8 and _true_post.std() > 1e-8:
            r_panel = float(np.corrcoef(_dec_unwrap, _true_post)[0, 1])
            r_str = f"r = {r_panel:.3f}"
        else:
            r_str = "r = n/a"
    else:
        r_str = "r = n/a"
    ax_hd.set_title(
        "heading tracking on snapshot rollout   "
        f"({r_str}, RMSE = {rmse_deg:.1f}°)",
        fontsize=8,
    )


def plot_cx_training_snapshot(
    W_rec: np.ndarray,
    rollout: dict,
    epg_theta: np.ndarray,
    output_path: str,
    *,
    W_con: Optional[np.ndarray] = None,
    neuron_types: Optional[np.ndarray] = None,
    type_names: Optional[list[str]] = None,
    pen_neuron_types: Optional[np.ndarray] = None,
    step: Optional[int] = None,
    dt_s: float = 0.01,
    n_bins: int = 32,
    mat_vmax: float = 1.0,
    fwhm_z_thresh: float = 1.0,
    pi_acc_history: Optional[tuple] = None,
    rmse_history: Optional[tuple] = None,
    wrec_param: str = "edge_magnitude",
) -> None:
    """2 × 4 training-snapshot figure — writes a PNG.

    Row 1: GT W_con | learned W_rec | per-neuron EPG | per-neuron PEN
    Row 2: EPG kinograph | HD tracking | pi_acc trace | GT vs learned scatter

    pi_acc_history / rmse_history are each an optional (iterations, values)
    tuple of 1-D arrays. `rmse_history` is drawn on a twin y-axis on the
    right of the pi_acc panel. The bottom-right panel is a scatter of
    GT vs learned recurrent weights (excludes diagonal and zero entries),
    annotated with the linear-fit slope and R². The scatter is suppressed
    when `wrec_param == "column_dale"` (dense mode — learned W_rec has
    entries outside the connectome support, so per-edge GT comparison is
    not meaningful).
    """
    fig, axes = plt.subplots(
        2, 4, figsize=(22, 10),
        gridspec_kw=dict(hspace=0.40, wspace=0.35,
                         left=0.05, right=0.98, top=0.93, bottom=0.08),
    )
    ax_gt, ax_mat, ax_neu, ax_pen = axes[0]
    ax_kin, ax_hd, ax_pi, ax_fw = axes[1]

    render_cx_snapshot_into_axes(
        fig, ax_gt, ax_mat, ax_kin, ax_neu, ax_pen, ax_hd,
        W_rec=W_rec, rollout=rollout, epg_theta=epg_theta,
        W_con=W_con, neuron_types=neuron_types, type_names=type_names,
        pen_neuron_types=pen_neuron_types,
        dt_s=dt_s, n_bins=n_bins, fwhm_z_thresh=fwhm_z_thresh,
    )

    if pi_acc_history is not None and len(pi_acc_history[0]) > 0:
        it, pi = pi_acc_history
        ax_pi.plot(it, pi, color="C0", lw=1.6)
        ax_pi.axhline(0.95, color="r", ls=":", lw=0.8)
        ax_pi.set_ylim(-0.05, 1.05)
        ax_pi.set_xlabel("iteration", fontsize=10)
        ax_pi.set_ylabel("pi_acc", color="C0", fontsize=10)
        ax_pi.tick_params(axis="y", labelcolor="C0", labelsize=8)
        ax_pi.tick_params(axis="x", labelsize=8)
        ax_pi.set_title("path-integration accuracy", fontsize=8)
        if rmse_history is not None and len(rmse_history[0]) > 0:
            it_r, rmse = rmse_history
            ax_pi_r = ax_pi.twinx()
            ax_pi_r.plot(it_r, rmse, color="C3", lw=1.2)
            ax_pi_r.set_ylabel("rmse", color="C3", fontsize=10)
            ax_pi_r.tick_params(axis="y", labelcolor="C3", labelsize=8)
    else:
        ax_pi.axis("off")

    if wrec_param == "column_dale":
        ax_fw.axis("off")
    elif W_con is not None:
        mask = (W_con != 0)
        np.fill_diagonal(mask, False)
        x = np.asarray(W_con[mask], dtype=np.float32)
        y = np.asarray(W_rec[mask], dtype=np.float32)
        if x.size >= 2 and x.std() > 0:
            slope, intercept = np.polyfit(x, y, 1)
            r = float(np.corrcoef(x, y)[0, 1])
            r2 = r * r
            ax_fw.scatter(x, y, s=30, c="0.15", alpha=0.55, edgecolors="none")
            lo, hi = float(x.min()), float(x.max())
            xline = np.array([lo, hi])
            ax_fw.plot(xline, slope * xline + intercept, color="C3", lw=1.0)
            ax_fw.axhline(0, color="0.6", lw=0.3)
            ax_fw.axvline(0, color="0.6", lw=0.3)
            ax_fw.set_xlabel(r"GT $W_{rec}$", fontsize=10)
            ax_fw.set_ylabel(r"learned $\hat W_{rec}$", fontsize=10)
            ax_fw.set_title(f"slope = {slope:.3f},  $R^2$ = {r2:.3f}",
                            fontsize=8)
            ax_fw.tick_params(labelsize=8)
        else:
            ax_fw.axis("off")
    else:
        ax_fw.axis("off")

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


# ===========================================================================
# Evolution figure — paper figure + training-time snapshot. Moved here from
# figures/drosophila_cx/fig_evolution.py so the trainer (bump_attractor_eval._
# save_training_snapshot) can import it via the normal package path instead
# of the importlib hack it used to do. The standalone CLI in
# figures/drosophila_cx/fig_evolution.py now imports plot_cx_evolution and
# keeps only the data-loading + argparse code.
#
# Public entry point: plot_cx_evolution(data, out_path, run_dir, n_rows).
# All _panel_* helpers below are private to this module.
# ===========================================================================

PANEL_LABEL_FS = 16
TITLE_FS = 12
LABEL_FS = 11
TICK_FS = 9
GT_COLOR = "#4daf4a"
PRED_COLOR = "black"


def _type_tick_fs(n_labels: int) -> float:
    """Auto-scale cell-type tick fontsize so 30+ types stay legible.

    Fly CX has 7 types -> returns TICK_FS (9pt, unchanged).
    Zebrafish HD has 31 types -> returns ~4pt so labels don't overlap.
    """
    return max(4.0, min(float(TICK_FS), 130.0 / max(int(n_labels), 1)))


def _panel_label(ax, letter: str):
    ax.text(-0.12, 1.02, letter, transform=ax.transAxes,
            fontsize=PANEL_LABEL_FS, fontweight="bold",
            va="bottom", ha="right")


def _panel_matrix(ax, M: np.ndarray, neuron_types, type_names, title: str):
    """Type-pair grouped W matrix, zero-centred signed-sqrt scaled and
    edge-dilated so individual synapses are visible at panel resolution."""
    if M is None:
        ax.text(0.5, 0.5, "no matrix", ha="center", va="center",
                transform=ax.transAxes); ax.axis("off"); return
    Marr = np.asarray(M, dtype=np.float32)
    nz = np.abs(Marr[Marr != 0])
    # Zero-centred, sign-preserving scale (red = excitatory, blue =
    # inhibitory) so EVERY non-zero edge gets a clear colour. A mean-
    # subtracted z-score centres the most common weight at white and the
    # 5x-amplified inhibitory tail inflates the spread, washing the matrix
    # out; sqrt compresses that tail (90th-percentile magnitude saturates).
    scale = float(np.percentile(nz, 90)) if nz.size else 1.0
    Z = np.sign(Marr) * np.sqrt(np.abs(Marr) / (scale + 1e-12))
    Z = np.clip(Z, -1.0, 1.0)
    # Dilate each 1-px edge into a blob (size scales with N) so individual
    # synapses are visible; signed max/min filters, larger magnitude wins.
    from scipy.ndimage import maximum_filter, minimum_filter
    blob = max(5, int(round(Marr.shape[0] / 130.0)))
    Zpos = maximum_filter(np.where(Z > 0, Z, 0.0), size=blob)
    Zneg = minimum_filter(np.where(Z < 0, Z, 0.0), size=blob)
    Zvis = np.where(np.abs(Zpos) >= np.abs(Zneg), Zpos, Zneg)
    im = ax.imshow(Zvis, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                    interpolation="nearest", aspect="equal")
    nt = np.asarray(neuron_types)
    if nt.size:
        order = np.argsort(nt, kind="stable")
        b = np.where(np.diff(nt[order]) != 0)[0] + 0.5
        for x in b:
            ax.axvline(x, color="k", lw=0.3, alpha=0.5)
            ax.axhline(x, color="k", lw=0.3, alpha=0.5)
        boundaries = np.concatenate([[0], b + 0.5, [nt.size]])
        centres = (boundaries[:-1] + boundaries[1:]) / 2 - 0.5
        labels = [type_names[int(nt[order[int(c)]])] for c in centres]
        type_fs = _type_tick_fs(len(labels))
        ax.set_xticks(centres); ax.set_xticklabels(labels, fontsize=type_fs,
                                                     rotation=45, ha="right")
        ax.set_yticks(centres); ax.set_yticklabels(labels, fontsize=type_fs)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel("presynaptic", fontsize=LABEL_FS)
    ax.set_ylabel("postsynaptic", fontsize=LABEL_FS)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.set_label(r"sign$\cdot\sqrt{|W|}$ (norm.)", fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)


def _panel_weight_scatter(ax, W_con, W_rec):
    """Scatter of true (W_con, panel a) vs learned (W_rec, panel b) weights
    over the connectome edges. Excludes the diagonal and zero entries; y=x
    reference line and Pearson r annotated. Under the sign-lock the points
    stay in matching-sign quadrants, so the spread off y=x shows how training
    rescaled the connectome magnitudes."""
    if W_con is None or W_rec is None:
        ax.text(0.5, 0.5, "no weights", ha="center", va="center",
                transform=ax.transAxes); ax.axis("off"); return
    A = np.asarray(W_con, dtype=np.float64)
    B = np.asarray(W_rec, dtype=np.float64)
    mask = A != 0
    np.fill_diagonal(mask, False)
    x, y = A[mask], B[mask]
    if x.size == 0:
        ax.text(0.5, 0.5, "no edges", ha="center", va="center",
                transform=ax.transAxes); ax.axis("off"); return
    # Training rescales the connectome by a large global gain, so the raw
    # cloud lies on a steep line and the per-edge deviations are invisible.
    # Correct for that slope: fit a through-origin least-squares gain
    # (sign-locked weights pass through 0) and divide it out, so the cloud
    # centres on y=x and the residual scatter is the genuine per-edge
    # reweighting. The removed slope is the mean connectome amplification.
    denom = float(np.sum(x * x))
    slope = float(np.sum(x * y) / denom) if denom > 0 else 1.0
    if slope != 0:
        y = y / slope
    r = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else float("nan")
    ax.scatter(x, y, s=4, alpha=0.25, color="0.25", edgecolors="none",
               rasterized=True)
    lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color="0.55", lw=0.8, ls="--", zorder=0)
    ax.axhline(0, color="0.8", lw=0.4); ax.axvline(0, color="0.8", lw=0.4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(rf"true vs. learned  ($r={r:.3f}$)", fontsize=TITLE_FS)
    ax.set_xlabel(r"GT $W^{\mathrm{con}}_{ij}$", fontsize=LABEL_FS)
    ax.set_ylabel(r"learned $\hat W_{ij}\,/\,m$", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)


def _panel_neuron_kinograph(ax, r_pop, neuron_types_sub, type_names,
                             dt_s: float, ylabel: str):
    """Per-neuron z-scored firing-rate kinograph, no title."""
    if r_pop is None or r_pop.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes); ax.axis("off"); return
    T = r_pop.shape[0]
    z = (r_pop - r_pop.mean(axis=0, keepdims=True))
    sd = r_pop.std(axis=0, keepdims=True); sd[sd < 1e-8] = 1.0
    z = (z / sd).clip(-3.0, 3.0)
    im = ax.imshow(z.T, aspect="auto", origin="lower", cmap="RdBu_r",
                    vmin=-3.0, vmax=3.0,
                    extent=[0, T * dt_s, 0, z.shape[1]],
                    interpolation="nearest")
    if neuron_types_sub is not None and neuron_types_sub.size:
        nt = np.asarray(neuron_types_sub)
        order = np.argsort(nt, kind="stable")
        boundaries = np.where(np.diff(nt[order]) != 0)[0] + 0.5
        for b in boundaries:
            ax.axhline(b, color="k", lw=0.3, alpha=0.6)
    ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=TICK_FS)


def _panel_all_neurons_kinograph(ax, r_full: np.ndarray, neuron_types,
                                   type_names, dt_s: float):
    """Per-neuron firing-rate kinograph for ALL neurons.

    Neurons are reordered by neuron type so cell-type blocks are visible.
    Z-scored per neuron (column-wise), clipped to ±3.
    """
    if r_full is None or r_full.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes); ax.axis("off"); return
    nt = np.asarray(neuron_types)
    order = np.argsort(nt, kind="stable")
    r_sorted = r_full[:, order]
    T = r_sorted.shape[0]
    mu = r_sorted.mean(axis=0, keepdims=True)
    sd = r_sorted.std(axis=0, keepdims=True); sd[sd < 1e-8] = 1.0
    z = ((r_sorted - mu) / sd).clip(-3.0, 3.0)
    im = ax.imshow(z.T, aspect="auto", origin="lower", cmap="RdBu_r",
                    vmin=-3.0, vmax=3.0,
                    extent=[0, T * dt_s, 0, z.shape[1]],
                    interpolation="nearest")
    nt_sorted = nt[order]
    boundaries = np.where(np.diff(nt_sorted) != 0)[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="k", lw=0.3, alpha=0.6)
    bounds_full = np.concatenate([[0], boundaries + 0.5, [nt_sorted.size]])
    centres = (bounds_full[:-1] + bounds_full[1:]) / 2 - 0.5
    labels = [type_names[int(nt_sorted[int(c)])] for c in centres]
    ax.set_yticks(centres)
    ax.set_yticklabels(labels, fontsize=_type_tick_fs(len(labels)))
    ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax.set_ylabel("neuron type", fontsize=LABEL_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=TICK_FS)


def _panel_population_kinograph(ax, rollout: dict, epg_theta: np.ndarray,
                                 dt_s: float, n_bins: int = 32):
    """Population EPG kinograph (orientation × time), no overlay."""
    r_epg = np.asarray(rollout["r_epg"])
    T = r_epg.shape[0]
    theta = np.angle(np.exp(1j * np.asarray(epg_theta)))
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(theta, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    grid = np.zeros((T, n_bins), dtype=np.float32)
    cnt = np.zeros(n_bins, dtype=np.float32)
    for k, b in enumerate(bin_idx):
        grid[:, b] += r_epg[:, k]
        cnt[b] += 1.0
    cnt[cnt < 1.0] = 1.0
    grid /= cnt[None, :]
    z = (grid - grid.mean(axis=1, keepdims=True))
    sd = grid.std(axis=1, keepdims=True); sd[sd < 1e-8] = 1.0
    z = (z / sd).clip(-3.0, 3.0)
    im = ax.imshow(z.T, aspect="auto", origin="lower", cmap="RdBu_r",
                    vmin=-3.0, vmax=3.0,
                    extent=[0, T * dt_s, -np.pi, np.pi],
                    interpolation="nearest")
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=TICK_FS)
    ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax.set_ylabel("orientation (rad)", fontsize=LABEL_FS)
    ax.set_title("EPG bump", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=TICK_FS)


def _hd_track_r_str(true_theta, dec_theta, warmup):
    """Pearson r between unwrapped decoded and GT heading after warmup."""
    if true_theta.size <= warmup:
        return "r = n/a"
    d_uw = np.unwrap(dec_theta[warmup:])
    if d_uw.std() < 1e-8 or true_theta[warmup:].std() < 1e-8:
        return "r = n/a"
    r = float(np.corrcoef(d_uw, true_theta[warmup:])[0, 1])
    return f"r = {r:.3f}"


def _xi_track_r_str(true_xi, dec_xi, warmup):
    """Pearson r between decoded ξ and GT ξ after warmup."""
    if true_xi.size <= warmup:
        return "r = n/a"
    if dec_xi[warmup:].std() < 1e-8 or true_xi[warmup:].std() < 1e-8:
        return "r = n/a"
    r = float(np.corrcoef(dec_xi[warmup:], true_xi[warmup:])[0, 1])
    return f"r = {r:.3f}"


def _draw_rotation_track(ax_drive, ax_track, t_axis, u_col, true_theta,
                         dec_theta, warmup, title_prefix=""):
    """Render the rotation half of the tracking panel into two given axes:
    top axis = ω(t), bottom axis = wrapped heading (GT vs decoded)."""
    ax_drive.plot(t_axis, u_col, color=GT_COLOR, lw=1.2)
    ax_drive.axhline(0, color="0.7", lw=0.3)
    ax_drive.set_ylabel("ω (°/s)", fontsize=LABEL_FS)
    ax_drive.tick_params(labelsize=TICK_FS, labelbottom=False)
    err = np.angle(np.exp(1j * (dec_theta - true_theta)))
    rmse_deg = float(np.degrees(np.sqrt(np.mean(err ** 2))))
    r_str = _hd_track_r_str(true_theta, dec_theta, warmup)
    ax_drive.set_title(
        f"{title_prefix}constant-ω rollout  ({r_str},  RMSE = {rmse_deg:.1f}°)",
        fontsize=TITLE_FS,
    )

    true_wrap = np.angle(np.exp(1j * true_theta))
    dec_wrap = np.angle(np.exp(1j * dec_theta))
    ax_track.plot(t_axis, true_wrap, color=GT_COLOR, lw=0.0, marker=".", ms=2.5)
    ax_track.plot(t_axis, dec_wrap, color=PRED_COLOR, lw=0.0, marker=".", ms=0.8)
    ax_track.set_yticks([-np.pi, 0, np.pi])
    ax_track.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=TICK_FS)
    ax_track.set_ylim(-np.pi - 0.15, np.pi + 0.15)
    ax_track.set_ylabel("HD (rad)", fontsize=LABEL_FS)
    ax_track.tick_params(labelsize=TICK_FS)


def _draw_translation_track(ax_drive, ax_track, t_axis, u_col, true_xi,
                            dec_xi, warmup, title_prefix=""):
    """Render the translation half of the tracking panel into two given axes:
    top axis = v_fwd(t), bottom axis = ξ (GT vs decoded), linear unbounded."""
    ax_drive.plot(t_axis, u_col, color=GT_COLOR, lw=1.2)
    ax_drive.axhline(0, color="0.7", lw=0.3)
    ax_drive.set_ylabel(r"$v_{\mathrm{fwd}}$", fontsize=LABEL_FS)
    ax_drive.tick_params(labelsize=TICK_FS, labelbottom=False)
    rmse = float(np.sqrt(np.mean((dec_xi - true_xi) ** 2)))
    r_str = _xi_track_r_str(true_xi, dec_xi, warmup)
    ax_drive.set_title(
        f"{title_prefix}constant-$v_{{\\mathrm{{fwd}}}}$ rollout  "
        f"({r_str},  RMSE = {rmse:.2f})",
        fontsize=TITLE_FS,
    )
    ax_track.plot(t_axis, true_xi, color=GT_COLOR, lw=1.2)
    ax_track.plot(t_axis, dec_xi,  color=PRED_COLOR, lw=0.8)
    ax_track.set_ylabel(r"$\xi$", fontsize=LABEL_FS)
    ax_track.tick_params(labelsize=TICK_FS)


def _panel_hd_tracking_stacked(fig, subplotspec, rollout: dict, dt_s: float,
                                warmup: int = 10):
    """Mode-aware deterministic-rollout panel.

    Picks the layout from what the rollout dict contains (set by
    ``bump_attractor_eval._deterministic_sweep_rollout`` from
    ``net.n_input``):

      rotation in rollout + translation in rollout  → 4 stacked sub-axes
          (ω, v_fwd, heading, ξ) — n_input == 4 mode.
      rotation only                                 → 2 sub-axes
          (ω, heading) — n_input == 3 (legacy CX) — byte-identical to the
          previous implementation.
      translation only                              → 2 sub-axes
          (v_fwd, ξ)   — n_input == 1.

    Returns the top axis (to attach the panel label).
    """
    has_rot = "true_theta" in rollout
    has_trans = "true_xi" in rollout
    has_xy = "true_xy" in rollout
    u = np.asarray(rollout["u"])
    T = u.shape[0]
    t_axis = np.arange(T) * dt_s

    if has_xy:
        # position_2d mode (n_input=4, n_output=4). Drives are ω and v_fwd
        # (compact time traces); the main panel is a square 2D path plot
        # comparing GT (x, y) in green to decoded (x̂, ŷ) in black. The
        # ξ trace is replaced by the 2D path because in 2D PI the
        # informative quantity is the *spatial* trajectory rather than a
        # 1D time series.
        sub = GridSpecFromSubplotSpec(
            3, 1, subplot_spec=subplotspec,
            height_ratios=[0.5, 0.5, 2.5], hspace=0.30,
        )
        ax_w   = fig.add_subplot(sub[0])
        ax_vf  = fig.add_subplot(sub[1], sharex=ax_w)
        ax_xy  = fig.add_subplot(sub[2])
        # Drives.
        ax_w.plot(t_axis, u[:, 0], color=GT_COLOR, lw=1.0)
        ax_w.axhline(0, color="0.7", lw=0.3)
        ax_w.set_ylabel("ω (°/s)", fontsize=LABEL_FS)
        ax_w.tick_params(labelsize=TICK_FS, labelbottom=False)
        ax_vf.plot(t_axis, u[:, 1], color=GT_COLOR, lw=1.0)
        ax_vf.axhline(0, color="0.7", lw=0.3)
        ax_vf.set_ylabel(r"$v_{\mathrm{fwd}}$", fontsize=LABEL_FS)
        ax_vf.tick_params(labelsize=TICK_FS)
        ax_vf.set_xlabel("time (s)", fontsize=LABEL_FS)
        # 2D path.
        true_xy = np.asarray(rollout["true_xy"])
        dec_xy = np.asarray(rollout["decoded_xy"])
        if true_xy.shape[0] > warmup:
            err = dec_xy[warmup:] - true_xy[warmup:]
            rmse_xy = float(np.sqrt(np.mean(err ** 2)))
            rs = []
            for axis in range(2):
                a = dec_xy[warmup:, axis]
                b = true_xy[warmup:, axis]
                if a.std() > 1e-8 and b.std() > 1e-8:
                    rs.append(float(np.corrcoef(a, b)[0, 1]))
            r_str = f"r̄ = {np.mean(rs):.3f}" if rs else "r̄ = n/a"
            ax_xy.set_title(
                f"2D path  ({r_str},  RMSE = {rmse_xy:.2f})",
                fontsize=TITLE_FS,
            )
        else:
            ax_xy.set_title("2D path", fontsize=TITLE_FS)
        ax_xy.plot(true_xy[:, 0], true_xy[:, 1], color=GT_COLOR, lw=1.0,
                   label="GT")
        ax_xy.plot(dec_xy[:, 0],  dec_xy[:, 1],  color=PRED_COLOR, lw=0.8,
                   label="decoded")
        # Origin marker (trial start).
        ax_xy.plot([0.0], [0.0], "o", color="0.4", ms=4, zorder=5)
        ax_xy.set_xlabel("x", fontsize=LABEL_FS)
        ax_xy.set_ylabel("y", fontsize=LABEL_FS)
        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.legend(loc="best", fontsize=TICK_FS, frameon=False)
        ax_xy.tick_params(labelsize=TICK_FS)
        return ax_w

    if has_rot and has_trans:
        # Both: 4 stacked sub-axes. Drives compact on top (h.ratios 0.6),
        # integrators larger (1.2) so the GT/decoded comparison is readable.
        sub = GridSpecFromSubplotSpec(
            4, 1, subplot_spec=subplotspec,
            height_ratios=[0.6, 0.6, 1.2, 1.2], hspace=0.22,
        )
        ax_w  = fig.add_subplot(sub[0])
        ax_vf = fig.add_subplot(sub[1], sharex=ax_w)
        ax_hd = fig.add_subplot(sub[2], sharex=ax_w)
        ax_xi = fig.add_subplot(sub[3], sharex=ax_w)
        ax_w.tick_params(labelbottom=False)
        ax_vf.tick_params(labelbottom=False)
        ax_hd.tick_params(labelbottom=False)
        _draw_rotation_track(
            ax_w, ax_hd, t_axis, u[:, 0],
            np.asarray(rollout["true_theta"]),
            np.asarray(rollout["decoded_theta"]),
            warmup,
        )
        # u[:, 1] is v_fwd in both mode; no separate title on the v_fwd row
        # — the rotation row's title carries the headline metrics, the xi
        # row's title carries the translation metrics.
        _draw_translation_track(
            ax_vf, ax_xi, t_axis, u[:, 1],
            np.asarray(rollout["true_xi"]),
            np.asarray(rollout["decoded_xi"]),
            warmup,
        )
        ax_xi.set_xlabel("time (s)", fontsize=LABEL_FS)
        return ax_w
    if has_rot:
        sub = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=subplotspec,
            height_ratios=[1.0, 1.8], hspace=0.18,
        )
        ax_top = fig.add_subplot(sub[0])
        ax_bot = fig.add_subplot(sub[1], sharex=ax_top)
        _draw_rotation_track(
            ax_top, ax_bot, t_axis, u[:, 0],
            np.asarray(rollout["true_theta"]),
            np.asarray(rollout["decoded_theta"]),
            warmup,
        )
        ax_bot.set_xlabel("time (s)", fontsize=LABEL_FS)
        return ax_top
    if has_trans:
        sub = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=subplotspec,
            height_ratios=[1.0, 1.8], hspace=0.18,
        )
        ax_top = fig.add_subplot(sub[0])
        ax_bot = fig.add_subplot(sub[1], sharex=ax_top)
        # u[:, 0] is v_fwd in translation-only mode.
        _draw_translation_track(
            ax_top, ax_bot, t_axis, u[:, 0],
            np.asarray(rollout["true_xi"]),
            np.asarray(rollout["decoded_xi"]),
            warmup,
        )
        ax_bot.set_xlabel("time (s)", fontsize=LABEL_FS)
        return ax_top
    # Defensive fallback — empty rollout, blank panel rather than crash.
    sub = GridSpecFromSubplotSpec(1, 1, subplot_spec=subplotspec)
    ax = fig.add_subplot(sub[0])
    ax.axis("off")
    return ax


def _panel_trial_rollout(fig, subplotspec, test_trial: dict):
    """One-trial rollout panel — mirrors the deterministic-rollout panel f
    layout, but on a single OU / swim trial drawn from u_test:

      rotation-only  (y_true.shape[-1] == 2) → 2 sub-axes  (ω, HD)
      translation-only (== 1)                 → 2 sub-axes  (v_fwd, ξ)
      both (== 3)                             → 4 sub-axes  (ω, v_fwd, HD, ξ)

    Mode is detected from the trial's u / y_true shapes, which the trainer
    already slices to the active task channels in
    ``graph_trainer._data_train_drosophila_cx_task`` (via the
    task_targets-driven projection at the top of training).

    Returns the *top* axis (used to attach the panel label).
    """
    u = np.asarray(test_trial["u"])
    y_true = np.asarray(test_trial["y_true"])
    y_pred = np.asarray(test_trial["y_pred"])
    dt = float(test_trial["dt"])
    T = u.shape[0]
    t_axis = np.arange(T) * dt

    n_in = int(u.shape[-1]) if u.ndim >= 2 else 0
    n_out = int(y_true.shape[-1]) if y_true.ndim >= 2 else 0
    has_rot = n_out >= 2   # cos, sin always lead the target when rotation is on
    has_trans = (n_in == 1) or (n_in == 4 and n_out == 3)
    has_xy = (n_in == 4 and n_out == 4)   # position_2d: heading + (x, y)

    trial_label = str(test_trial.get("label", "test trial"))
    title_prefix = f"{trial_label} #{int(test_trial['idx'])}  "

    def _draw_rot(ax_drive, ax_track, drive_col, is_top):
        ax_drive.plot(t_axis, drive_col, color=GT_COLOR, lw=0.8)
        ax_drive.axhline(0, color="0.7", lw=0.3)
        ax_drive.set_ylabel("ω (°/s)", fontsize=LABEL_FS)
        ax_drive.tick_params(labelsize=TICK_FS, labelbottom=False)
        if is_top:
            ax_drive.set_title(title_prefix.rstrip(), fontsize=TITLE_FS)
        # Heading in y[:, 0:2] in both rotation-only and both modes.
        theta_true = np.arctan2(y_true[:, 1], y_true[:, 0])
        theta_pred = np.arctan2(y_pred[:, 1], y_pred[:, 0])
        ax_track.plot(t_axis, theta_true, color=GT_COLOR, lw=0.0,
                      marker=".", ms=2.0)
        ax_track.plot(t_axis, theta_pred, color=PRED_COLOR, lw=0.0,
                      marker=".", ms=0.6)
        ax_track.set_yticks([-np.pi, 0, np.pi])
        ax_track.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=TICK_FS)
        ax_track.set_ylim(-np.pi - 0.15, np.pi + 0.15)
        ax_track.set_ylabel("HD (rad)", fontsize=LABEL_FS)
        ax_track.tick_params(labelsize=TICK_FS)

    def _draw_trans(ax_drive, ax_track, drive_col, xi_col_idx, is_top):
        ax_drive.plot(t_axis, drive_col, color=GT_COLOR, lw=0.8)
        ax_drive.axhline(0, color="0.7", lw=0.3)
        ax_drive.set_ylabel(r"$v_{\mathrm{fwd}}$", fontsize=LABEL_FS)
        ax_drive.tick_params(labelsize=TICK_FS, labelbottom=False)
        if is_top:
            ax_drive.set_title(title_prefix.rstrip(), fontsize=TITLE_FS)
        ax_track.plot(t_axis, y_true[:, xi_col_idx], color=GT_COLOR, lw=1.2)
        ax_track.plot(t_axis, y_pred[:, xi_col_idx], color=PRED_COLOR, lw=0.8)
        ax_track.set_ylabel(r"$\xi$", fontsize=LABEL_FS)
        ax_track.tick_params(labelsize=TICK_FS)

    if has_xy:
        # position_2d: ω(t) + v_fwd(t) on small upper traces, square 2D path
        # below (GT green, decoded black). The trial's u is time-varying so
        # the drives need their full time series; the path plot replaces
        # the standard HD/ξ time-series since the informative quantity in
        # 2D PI is the spatial trajectory.
        sub = GridSpecFromSubplotSpec(
            3, 1, subplot_spec=subplotspec,
            height_ratios=[0.5, 0.5, 2.5], hspace=0.30,
        )
        ax_w   = fig.add_subplot(sub[0])
        ax_vf  = fig.add_subplot(sub[1], sharex=ax_w)
        ax_xy  = fig.add_subplot(sub[2])
        ax_w.plot(t_axis, u[:, 0], color=GT_COLOR, lw=0.8)
        ax_w.axhline(0, color="0.7", lw=0.3)
        ax_w.set_ylabel("ω (°/s)", fontsize=LABEL_FS)
        ax_w.set_title(title_prefix.rstrip(), fontsize=TITLE_FS)
        ax_w.tick_params(labelsize=TICK_FS, labelbottom=False)
        ax_vf.plot(t_axis, u[:, 1], color=GT_COLOR, lw=0.8)
        ax_vf.axhline(0, color="0.7", lw=0.3)
        ax_vf.set_ylabel(r"$v_{\mathrm{fwd}}$", fontsize=LABEL_FS)
        ax_vf.tick_params(labelsize=TICK_FS)
        ax_vf.set_xlabel("time (s)", fontsize=LABEL_FS)
        # 2D path — y_true / y_pred columns [2, 3].
        true_xy = y_true[:, 2:4]
        dec_xy = y_pred[:, 2:4]
        ax_xy.plot(true_xy[:, 0], true_xy[:, 1], color=GT_COLOR, lw=1.0,
                   label="GT")
        ax_xy.plot(dec_xy[:, 0],  dec_xy[:, 1],  color=PRED_COLOR, lw=0.8,
                   label="decoded")
        ax_xy.plot([0.0], [0.0], "o", color="0.4", ms=4, zorder=5)
        ax_xy.set_xlabel("x", fontsize=LABEL_FS)
        ax_xy.set_ylabel("y", fontsize=LABEL_FS)
        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.legend(loc="best", fontsize=TICK_FS, frameon=False)
        ax_xy.tick_params(labelsize=TICK_FS)
        return ax_w

    if has_rot and has_trans:
        sub = GridSpecFromSubplotSpec(
            4, 1, subplot_spec=subplotspec,
            height_ratios=[0.6, 0.6, 1.2, 1.2], hspace=0.22,
        )
        ax_w  = fig.add_subplot(sub[0])
        ax_vf = fig.add_subplot(sub[1], sharex=ax_w)
        ax_hd = fig.add_subplot(sub[2], sharex=ax_w)
        ax_xi = fig.add_subplot(sub[3], sharex=ax_w)
        ax_w.tick_params(labelbottom=False)
        ax_vf.tick_params(labelbottom=False)
        ax_hd.tick_params(labelbottom=False)
        _draw_rot(ax_w, ax_hd, u[:, 0], is_top=True)
        _draw_trans(ax_vf, ax_xi, u[:, 1], xi_col_idx=2, is_top=False)
        ax_xi.set_xlabel("time (s)", fontsize=LABEL_FS)
        return ax_w
    if has_rot:
        sub = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=subplotspec,
            height_ratios=[1.0, 1.8], hspace=0.18,
        )
        ax_top = fig.add_subplot(sub[0])
        ax_bot = fig.add_subplot(sub[1], sharex=ax_top)
        _draw_rot(ax_top, ax_bot, u[:, 0], is_top=True)
        ax_bot.set_xlabel("time (s)", fontsize=LABEL_FS)
        return ax_top
    if has_trans:
        sub = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=subplotspec,
            height_ratios=[1.0, 1.8], hspace=0.18,
        )
        ax_top = fig.add_subplot(sub[0])
        ax_bot = fig.add_subplot(sub[1], sharex=ax_top)
        # Translation-only: u[:, 0] is v_fwd, y_true[:, 0] is ξ.
        _draw_trans(ax_top, ax_bot, u[:, 0], xi_col_idx=0, is_top=True)
        ax_bot.set_xlabel("time (s)", fontsize=LABEL_FS)
        return ax_top
    # Defensive fallback — empty trial / unexpected shape.
    sub = GridSpecFromSubplotSpec(1, 1, subplot_spec=subplotspec)
    ax = fig.add_subplot(sub[0])
    ax.axis("off")
    return ax


def _is_gnn(net) -> bool:
    return all(hasattr(net, n) for n in ("a", "f_theta", "g_phi"))


def _compute_tuning_data(gain_data, n_neurons, n_bins=16, warmup=10):
    """Concatenate constant-omega rollouts and build per-neuron HD curves."""
    all_r, all_dec, all_om = [], [], []
    for omega, ro in gain_data:
        T = ro["r"].shape[0]
        if T <= warmup:
            continue
        all_r.append(ro["r"][warmup:])
        dec = np.angle(np.exp(1j * ro["decoded_theta"][warmup:]))
        all_dec.append(dec)
        all_om.append(np.full(T - warmup, omega))
    r_all = np.concatenate(all_r, axis=0)
    dec_all = np.concatenate(all_dec)
    om_all = np.concatenate(all_om)

    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.clip(np.digitize(dec_all, bins) - 1, 0, n_bins - 1)
    curves = np.zeros((n_neurons, n_bins))
    counts = np.zeros(n_bins)
    for b in range(n_bins):
        m = bin_idx == b
        counts[b] = int(m.sum())
        if m.any():
            curves[:, b] = r_all[m].mean(axis=0)
    centres = (bins[:-1] + bins[1:]) / 2
    preferred = centres[np.argmax(curves, axis=1)]
    return curves, preferred, om_all, r_all


def _panel_preferred_direction_polar(ax, curves, preferred,
                                       neuron_types, type_names):
    """Polar scatter of preferred HD per neuron, coloured by cell type."""
    nt = np.asarray(neuron_types).astype(int)
    hd_max = curves.max(axis=1)
    hd_min = curves.min(axis=1)
    strength = (hd_max - hd_min) / np.maximum(hd_max, 1e-8)
    palette = plt.get_cmap("tab10").colors
    for t in sorted(set(nt.tolist())):
        m = nt == t
        if not m.any():
            continue
        col = palette[t % len(palette)]
        ax.scatter(preferred[m], strength[m],
                    c=[col], s=24, alpha=0.85,
                    edgecolors="none", label=type_names[t])
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids([0, 90, 180, 270],
                       [r"$0$", r"$\pi/2$", r"$\pi$", r"$-\pi/2$"],
                       fontsize=TICK_FS)
    ax.set_rlim(0, 1.05)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_rlabel_position(135)
    ax.tick_params(labelsize=TICK_FS - 1)
    ax.set_title("preferred HD vs tuning strength",
                  fontsize=TITLE_FS, pad=15)
    ax.legend(fontsize=TICK_FS - 1, loc="upper right",
              bbox_to_anchor=(1.30, 1.10),
              framealpha=0.85, ncol=1, handletextpad=0.3)


def _panel_tuning_scatter(ax, curves, om_all, r_all,
                           neuron_types, type_names):
    """HD vs velocity tuning scatter (Hulse Fig 2g analogue)."""
    N = curves.shape[0]
    hd_max = curves.max(axis=1)
    hd_min = curves.min(axis=1)
    hd_strength = (hd_max - hd_min) / np.maximum(hd_max, 1e-8)

    x = om_all - om_all.mean()
    x_var = (x ** 2).sum()
    vel_slope = np.zeros(N)
    if x_var > 1e-8:
        for i in range(N):
            y = r_all[:, i] - r_all[:, i].mean()
            vel_slope[i] = (x * y).sum() / x_var
    vel_slope_scaled = vel_slope * 1000.0

    nt = np.asarray(neuron_types).astype(int)
    palette = plt.get_cmap("tab10").colors
    for t in sorted(set(nt.tolist())):
        m = nt == t
        col = palette[t % len(palette)]
        ax.scatter(hd_strength[m], vel_slope_scaled[m],
                    c=[col], s=18, alpha=0.85, edgecolors="none",
                    label=type_names[t])
    ax.axhline(0, color="0.7", lw=0.4)
    ax.set_xlabel("HD-tuning strength", fontsize=LABEL_FS)
    ax.set_ylabel(r"velocity tuning ($\times 10^3$)", fontsize=LABEL_FS)
    ax.set_title("HD vs velocity tuning", fontsize=TITLE_FS)
    ax.legend(fontsize=TICK_FS - 1, loc="best", framealpha=0.85, ncol=2,
              handletextpad=0.3, columnspacing=0.4)
    ax.tick_params(labelsize=TICK_FS)


def _panel_phase_shift_histogram(ax, preferred, edge_index,
                                   neuron_types, type_names):
    """Per-edge phase shift histogram (Hulse Fig 2i analogue)."""
    src, dst = edge_index[0], edge_index[1]
    delta = preferred[dst] - preferred[src]
    delta = np.angle(np.exp(1j * delta))
    pre_types = np.asarray(neuron_types)[src]

    palette = plt.get_cmap("tab10").colors
    bins = np.linspace(-np.pi, np.pi, 36)
    for t in sorted(set(pre_types.tolist())):
        m = pre_types == t
        col = palette[t % len(palette)]
        ax.hist(np.asarray(delta)[m], bins=bins, alpha=0.55, color=col,
                 edgecolor="0.3", linewidth=0.3,
                 label=type_names[int(t)])
    ax.axvline(0, color="0.7", lw=0.4)
    ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0",
                          r"$\pi/2$", r"$\pi$"], fontsize=TICK_FS)
    ax.set_xlabel(r"phase shift $\delta$ (rad)", fontsize=LABEL_FS)
    ax.set_ylabel("edge count", fontsize=LABEL_FS)
    ax.set_title("per-edge phase shift (pre $\\to$ post)",
                  fontsize=TITLE_FS)
    ax.legend(fontsize=TICK_FS - 1, loc="best", framealpha=0.85, ncol=2,
              handletextpad=0.3, columnspacing=0.4)
    ax.tick_params(labelsize=TICK_FS)


def _panel_bump_fwhm(ax, rollout, epg_theta, dt_s,
                      n_bins=32, fwhm_z_thresh=1.0):
    """EPG bump FWHM (degrees) over time on the constant-omega rollout."""
    r_epg = np.asarray(rollout["r_epg"])
    T = r_epg.shape[0]
    theta = np.angle(np.exp(1j * np.asarray(epg_theta)))
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.clip(np.digitize(theta, edges) - 1, 0, n_bins - 1)
    bin_rad = 2 * np.pi / n_bins
    fwhms = np.full(T, np.nan)
    for t in range(T):
        grid = np.zeros(n_bins)
        cnt = np.zeros(n_bins)
        for k, b in enumerate(bin_idx):
            grid[b] += r_epg[t, k]
            cnt[b] += 1
        cnt[cnt < 1] = 1
        grid /= cnt
        if grid.std() < 1e-8:
            continue
        z = (grid - grid.mean()) / grid.std()
        peak = int(np.argmax(z))
        z_rolled = np.roll(z, n_bins // 2 - peak)
        c = n_bins // 2
        left, right = c, c
        while left - 1 >= 0 and z_rolled[left - 1] > fwhm_z_thresh:
            left -= 1
        while right + 1 < n_bins and z_rolled[right + 1] > fwhm_z_thresh:
            right += 1
        fwhms[t] = (right - left + 1) * bin_rad
    t_axis = np.arange(T) * dt_s
    ax.plot(t_axis, np.degrees(fwhms), color="black", lw=0.8)
    ax.axhline(80, color=GT_COLOR, lw=0.7, ls="--", alpha=0.7,
                label=r"~80$^\circ$ (Hulse target)")
    ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax.set_ylabel("EPG bump FWHM (deg)", fontsize=LABEL_FS)
    ax.set_title("bump width on constant-$\\omega$ rollout",
                  fontsize=TITLE_FS)
    ax.legend(fontsize=TICK_FS - 1, loc="upper right", framealpha=0.85)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_ylim(0, 360)


def _panel_voltage_distribution(ax, h_rollout, neuron_types, type_names):
    """Per-cell-type distribution of subthreshold $\\hat h_i(t)$."""
    nt = np.asarray(neuron_types).astype(int)
    type_ids = sorted(set(nt.tolist()))
    data = [h_rollout[:, nt == t].ravel() for t in type_ids]
    parts = ax.violinplot(data, positions=range(len(type_ids)),
                           widths=0.7, showmeans=False, showextrema=False)
    palette = plt.get_cmap("tab10").colors
    for i, p in enumerate(parts["bodies"]):
        p.set_facecolor(palette[i % len(palette)])
        p.set_edgecolor("0.3")
        p.set_alpha(0.7)
    means = [float(np.mean(d)) for d in data]
    stds  = [float(np.std(d))  for d in data]
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.errorbar(i, m, yerr=s, fmt="o", color="black",
                     markersize=3, capsize=3, lw=1.0)
    ax.axhline(0, color="0.6", lw=0.4)
    ax.set_xticks(range(len(type_ids)))
    ax.set_xticklabels([type_names[t] for t in type_ids],
                        rotation=45, ha="right",
                        fontsize=_type_tick_fs(len(type_ids)))
    ax.set_ylabel(r"$\hat h_i(t)$", fontsize=LABEL_FS)
    ax.set_title("subthreshold $h$ distribution by cell type",
                  fontsize=TITLE_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)


def calcium_zmse_ssim(real, learned):
    """Per-neuron z-scored MSE + SSIM between two (T, K) calcium kinographs.

    Returns ``(z_mse, ssim, Rz, Lz)`` where ``Rz``/``Lz`` are the per-neuron
    (over time) z-scored arrays. ``z_mse`` is exactly the per-neuron z-scored
    MSE used in the training observation loss (= 2(1-corr) for unit-variance
    signals; lower is better); ``ssim`` is the structural similarity of the two
    z-scored kinograph images (``nan`` if scikit-image is unavailable).
    """
    real = np.asarray(real, dtype=np.float32)
    learned = np.asarray(learned, dtype=np.float32)

    def _z(M):
        mu = M.mean(0, keepdims=True)
        sd = M.std(0, keepdims=True)
        return (M - mu) / np.where(sd > 1e-6, sd, 1.0)

    Rz = _z(real)                                           # (T, K), per-neuron z
    Lz = _z(learned)
    z_mse = float(np.mean((Rz - Lz) ** 2))
    try:
        from skimage.metrics import structural_similarity as _ssim
        dr = float(max(Rz.max(), Lz.max()) - min(Rz.min(), Lz.min()))
        ssim = float(_ssim(Rz, Lz, data_range=dr if dr > 0 else 1.0))
    except Exception:
        ssim = float("nan")
    return z_mse, ssim, Rz, Lz


def _panel_calcium_compare(ax, cp):
    """Real-vs-learned calcium kinograph for one trial (zebrafish obs runs).

    ``cp`` carries ``real`` / ``learned`` arrays of shape (T, K) over the same
    K bump-pool neurons in the same (rastermap) row order. Both are per-neuron
    z-scored for display and stacked — real on top, learned below — so the two
    kinographs can be compared row-for-row.
    """
    z_mse, ssim, Rz, Lz = calcium_zmse_ssim(cp["real"], cp["learned"])
    R = Rz.T                                                 # (K, T)
    L = Lz.T
    K, T = R.shape

    gap = np.full((max(2, K // 20), T), np.nan, dtype=np.float32)
    img = np.concatenate([R, gap, L], axis=0)
    dt = float(cp.get("dt", 0.01))
    t1 = T * dt
    ax.imshow(img, aspect="auto", cmap="viridis", vmin=-2, vmax=3,
              extent=[0, t1, img.shape[0], 0], interpolation="nearest")
    ax.text(0.01, 0.99, "real", transform=ax.transAxes, va="top", ha="left",
            color="w", fontsize=TICK_FS)
    ax.text(0.01, 0.01, "learned", transform=ax.transAxes, va="bottom",
            ha="left", color="w", fontsize=TICK_FS)
    ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax.set_ylabel(f"bump-pool neuron (n={K})", fontsize=LABEL_FS)
    ax.set_title(f"calcium  z-MSE={z_mse:.3f}  SSIM={ssim:.3f}",
                 fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)


def plot_calcium_reconstruction(groups, dt, out_path, title=None,
                                omega=None, hd=None, trial_s=None):
    """Full-block (~600 s) real-vs-learned calcium reconstruction figure.

    Panels, top→bottom (all sharing the time axis):
      * ω drive (deg/s)        — recorded angular-velocity stimulus (if ``omega``).
      * per neuron-set GROUP, in order: real ΔF/F, learned per-trial stitch,
        learned continuous. The observation loss supervises ALL observed neurons
        (bump-pool + afferents), so each group (e.g. ``bump-pool``, ``afferent``)
        gets its own real/stitch/continuous kinograph triplet.
      * HD true-vs-predicted   — one panel per learned rollout (green = true,
        black = readout decode), only for rollouts present in ``hd`` (heading is
        group-independent, so these sit once at the bottom).

    ``groups`` is a list of dicts ``{"name": str, "real": (T,K),
    "stitch": (T,K), "continuous": (T,K)|None}``; per-trial stitch re-anchors
    every 10 s (the trained regime), continuous is one unbroken rollout (drift).
    Kinographs are per-neuron z-scored on a shared viridis scale.
    ``omega``/``hd[<key>]['true'|'pred']`` are degree series on the model grid.
    ``trial_s`` (e.g. 10) draws a scale bar of that many seconds on the top
    panel — the per-trial stitch window length, so the eye can gauge how short
    one re-anchoring window is against the 600 s block.
    Returns ``{group_name: {stitch:{z_mse,ssim,T}, continuous:{…}}}``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GREEN = (0.0, 0.7, 0.25)
    _ROLLOUTS = [("stitch", "per-trial stitch (training regime)"),
                 ("continuous", "continuous 600 s rollout (drift)")]
    hd = hd or {}

    # Common time window: the shortest of every array supplied.
    lens = []
    for g in groups:
        lens.append(np.asarray(g["real"]).shape[0])
        for key, _ in _ROLLOUTS:
            if g.get(key) is not None:
                lens.append(np.asarray(g[key]).shape[0])
    T = int(min(lens))
    t = np.arange(T) * dt

    def _z(M):
        mu = M.mean(0, keepdims=True)
        sd = M.std(0, keepdims=True)
        return (M - mu) / np.where(sd > 1e-6, sd, 1.0)

    def _wrap(a):
        w = ((np.asarray(a, np.float64) + 180.0) % 360.0) - 180.0
        if w.size > 1:
            w[1:][np.abs(np.diff(w)) > 180.0] = np.nan
        return w

    # Build the panel stack: (kind, payload, height_ratio).
    specs = []
    if omega is not None:
        specs.append(("omega", None, 1.0))
    metrics = {}
    for g in groups:
        name = g["name"]
        real = np.asarray(g["real"], np.float32)[:T]
        metrics[name] = {}
        specs.append(("kino", (name, f"{name}: real ΔF/F (recorded)", real),
                      2.6))
        for key, sub in _ROLLOUTS:
            if g.get(key) is None:
                continue
            L = np.asarray(g[key], np.float32)[:T]
            z_mse, ssim, _, _ = calcium_zmse_ssim(real, L)
            metrics[name][key] = {"z_mse": z_mse, "ssim": ssim, "T": T}
            specs.append(("kino", (name, f"{name}: learned {sub} — "
                                   f"z-MSE={z_mse:.3f}  SSIM={ssim:.3f}", L),
                          2.6))
    for key, sub in _ROLLOUTS:
        if key in hd:
            specs.append(("hd", (sub, hd[key]), 1.4))

    ratios = [r for _, _, r in specs]
    fig, axs = plt.subplots(len(specs), 1, sharex=True,
                            figsize=(13, 0.92 * sum(ratios)),
                            gridspec_kw=dict(height_ratios=ratios),
                            squeeze=False)
    axl = axs[:, 0]
    for i, (ax, (kind, payload, _r)) in enumerate(zip(axl, specs)):
        bottom = (i == len(specs) - 1)
        if kind == "omega":
            ax.plot(t, np.asarray(omega)[:T], color="0.2", lw=0.8)
            ax.axhline(0, color="0.7", lw=0.4)
            ax.set_ylabel("ω (°/s)", fontsize=LABEL_FS)
            ax.set_title("angular-velocity drive", fontsize=TITLE_FS)
        elif kind == "kino":
            name, sub, M = payload
            ax.imshow(_z(M).T, aspect="auto", cmap="viridis", vmin=-2, vmax=3,
                      extent=[0, T * dt, M.shape[1], 0],
                      interpolation="nearest")
            ax.set_ylabel(f"{name}\nneuron (n={M.shape[1]})", fontsize=LABEL_FS)
            ax.set_title(sub, fontsize=TITLE_FS)
        else:  # hd true-vs-predicted
            sub, hdk = payload
            ax.plot(t, _wrap(np.asarray(hdk["true"])[:T]), color=GREEN, lw=1.0,
                    label="true")
            ax.plot(t, _wrap(np.asarray(hdk["pred"])[:T]), color="black",
                    lw=0.8, label="predicted")
            ax.set_ylim(-185, 185); ax.set_yticks([-180, 0, 180])
            ax.set_ylabel("HD (°)", fontsize=LABEL_FS)
            ax.set_title(f"heading — {sub}", fontsize=TITLE_FS)
            ax.legend(fontsize=TICK_FS, loc="upper right", framealpha=0.5)
        ax.tick_params(labelsize=TICK_FS, labelbottom=bottom)
        if bottom:
            ax.set_xlabel("time (s)", fontsize=LABEL_FS)

    # 10 s (one stitch trial) scale bar, lower-right of the bottom panel — x in
    # data (seconds), y in axes fraction. A white halo keeps it legible whether
    # the bottom panel is a (dark) kinograph or a (white) HD trace.
    if trial_s:
        import matplotlib.patheffects as pe
        ax = axl[-1]
        tr = ax.get_xaxis_transform()
        x1 = T * dt * 0.99
        x0 = x1 - trial_s
        ax.plot([x0, x1], [0.10, 0.10], transform=tr, color="black", lw=3,
                clip_on=False, solid_capstyle="butt",
                path_effects=[pe.withStroke(linewidth=5, foreground="white")])
        ax.text((x0 + x1) / 2, 0.17, f"{trial_s:.0f} s (1 trial)", transform=tr,
                ha="center", va="bottom", fontsize=TICK_FS, color="black",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    if title:
        fig.suptitle(title, fontsize=TITLE_FS + 1, y=0.997)
        fig.tight_layout(rect=[0, 0, 1, 0.975])
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return metrics


def _panel_image_from_png(ax, png_path):
    """Embed a PNG file as a borderless axis."""
    if not os.path.isfile(png_path):
        ax.text(0.5, 0.5, "snapshot missing", ha="center", va="center",
                 transform=ax.transAxes, fontsize=10, color="0.5")
        ax.axis("off")
        return
    img = plt.imread(png_path)
    ax.imshow(img, interpolation="bilinear")
    ax.axis("off")


def _latest_training_snapshot(run_dir, subdir):
    """Return the highest-step training-snapshot PNG under
    `run_dir/tmp_training/<subdir>/step_*.png`, or None if missing."""
    import re as _re
    pat = os.path.join(run_dir, "tmp_training", subdir, "step_*.png")
    files = glob.glob(pat)
    if not files:
        return None

    def _step_of(p):
        m = _re.search(r"step_(\d+)\.png$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    files.sort(key=_step_of)
    return files[-1]


def _panel_embedding(ax, net, neuron_types, type_names):
    """Scatter of the per-neuron latent embedding $\\mathbf{a}_i$."""
    emb = net.a.detach().cpu().numpy()
    nt = np.asarray(neuron_types).astype(int)
    n_types = len(type_names)
    palette = plt.get_cmap("tab10").colors
    for t in range(n_types):
        mask = (nt == t)
        if not mask.any():
            continue
        col = palette[t % len(palette)]
        ax.scatter(emb[mask, 0], emb[mask, 1],
                    c=[col], s=14, edgecolors="none",
                    alpha=0.9, label=type_names[t])
    ax.set_xlabel(r"$a_0$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$a_1$", fontsize=LABEL_FS)
    ax.set_title(r"embedding $\mathbf{a}_i$", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=3.2, loc="best", framealpha=0.6,
              ncol=2, markerscale=0.5, handletextpad=0.3, columnspacing=0.6)


def _panel_function_curves(ax, net, mlp_name: str, h_rollout: np.ndarray,
                            neuron_types, type_names, *,
                            square_output: bool, xlabel: str, ylabel: str,
                            title: str):
    """Mean ± SD per cell type of an MLP (f_theta or g_phi) over v ∈ [-3, 3]."""
    import torch

    device = next(getattr(net, mlp_name).parameters()).device
    n_pts = 400
    v_grid = torch.linspace(-3.0, 3.0, n_pts, device=device)
    a = net.a.to(device)
    N, emb_dim = a.shape
    rr = v_grid.unsqueeze(0).expand(N, -1)
    rr_flat = rr.reshape(-1, 1)
    a_flat = a.unsqueeze(1).expand(-1, n_pts, -1).reshape(-1, emb_dim)
    if mlp_name == "g_phi":
        feat = torch.cat([rr_flat, a_flat], dim=1)
    else:
        feat = torch.cat([rr_flat, a_flat, torch.zeros_like(rr_flat)], dim=1)
    mlp = getattr(net, mlp_name)
    with torch.no_grad():
        out = mlp(feat).reshape(N, n_pts, -1).squeeze(-1)
    if square_output and bool(getattr(net, "_g_phi_positive", True)):
        out = out.pow(2)
    v_np = v_grid.cpu().numpy()
    out_np = out.cpu().numpy()
    nt = np.asarray(neuron_types).astype(int)
    n_types = len(type_names)
    palette = plt.get_cmap("tab10").colors
    for t in range(n_types):
        mask = (nt == t)
        if not mask.any():
            continue
        col = palette[t % len(palette)]
        curves = out_np[mask]
        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        ax.plot(v_np, mean, color=col, lw=1.4, label=type_names[t])
        if std.max() > 1e-6:
            ax.fill_between(v_np, mean - std, mean + std,
                             color=col, alpha=0.15)
    ax.axhline(0, color="0.6", lw=0.4)
    ax.set_xlim(-3.0, 3.0)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    # Many cell types (33 for zebrafish) -> keep the legend tiny so it does
    # not swamp the curves. Small font, more columns, short handles.
    n_types = len(getattr(ax, "lines", [])) or 1
    leg_ncol = 3 if n_types <= 12 else 4
    ax.legend(fontsize=3.2, loc="best", framealpha=0.6, ncol=leg_ncol,
              handlelength=0.8, handletextpad=0.2, columnspacing=0.3,
              labelspacing=0.18, borderpad=0.2)


def _panel_integration_gain(ax, gain_data, dt: float, warmup: int = 10):
    """Hulse-style scatter: measured slope (deg/s) vs true ω (deg/s).

    No-op (axes hidden) when the rollout does not decode heading
    (e.g. translation-only models, where the output is forward distance
    only and the rotation gain is undefined)."""
    if not gain_data or "decoded_theta" not in (gain_data[0][1] or {}):
        ax.set_axis_off()
        return
    omegas, slopes = [], []
    for omega, ro in gain_data:
        dec = np.asarray(ro["decoded_theta"])
        T = dec.size
        t = np.arange(T) * dt
        if T <= warmup:
            continue
        d_uw = np.unwrap(dec[warmup:])
        t_post = t[warmup:]
        if d_uw.std() < 1e-8 or t_post.size < 2:
            slope = 0.0
        else:
            slope, _ = np.polyfit(t_post, d_uw, 1)
        omegas.append(float(omega))
        slopes.append(float(np.degrees(slope)))
    omegas = np.array(omegas); slopes = np.array(slopes)
    lim = max(float(np.abs(omegas).max()),
              float(np.abs(slopes).max()) if slopes.size else 1.0,
              1.0) * 1.10
    ax.plot([-lim, lim], [-lim, lim], color="0.5", lw=0.8, ls="--")
    ax.axhline(0, color="0.8", lw=0.4)
    ax.axvline(0, color="0.8", lw=0.4)

    linearity_tol = 0.25
    valid = np.abs(omegas) > 1e-8
    gains = np.full_like(omegas, np.nan)
    gains[valid] = slopes[valid] / omegas[valid]
    linear_mask = np.isfinite(gains) & (np.abs(gains - 1.0) <= linearity_tol)
    if linear_mask.sum() >= 2:
        om_ok = omegas[linear_mask]
        om_lo, om_hi = float(om_ok.min()), float(om_ok.max())
        ax.axvspan(om_lo, om_hi, color="0.6", alpha=0.18, zorder=0)
        domain_str = (f"linear: $[{om_lo:+.0f}, {om_hi:+.0f}]$"
                       r"$^\circ\!/\mathrm{s}$")
    else:
        domain_str = "linear: none"

    ax.scatter(omegas, slopes, s=10, c=PRED_COLOR, zorder=3)
    ax.text(0.03, 0.97, domain_str, transform=ax.transAxes,
             va="top", ha="left", fontsize=TICK_FS,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                        boxstyle="round,pad=0.2"))

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("true ω (°/s)", fontsize=LABEL_FS)
    ax.set_ylabel("measured slope (°/s)", fontsize=LABEL_FS)
    ax.set_title("integration gain  (target: y = x)", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)


def plot_cx_evolution(data: dict, out_path: str, *,
                       run_dir: str | None = None, n_rows: int = 3):
    """Render the drosophila CX evolution figure.

    Public entry point used by both the standalone CLI
    (``figures/drosophila_cx/fig_evolution.py``) and the training-time
    snapshot (``bump_attractor_eval._save_training_snapshot``). Previously
    lived as ``build_figure`` in the CLI script and was loaded via
    ``importlib`` from the trainer; centralised here so the trainer can
    just ``from connectome_gnn.plot_cx import plot_cx_evolution``.

    ``n_rows = 3``: full paper figure with panels a–l.
    ``n_rows = 2``: training-time snapshot — panels a–h only.

    ``data["test_trial"]`` may be None when n_rows=2: panel g is hidden.
    """
    plt.style.use("default")
    is_gnn = _is_gnn(data["net"])
    if n_rows == 2:
        figsize = (20, 9.5)
    else:
        figsize = (20, 14)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, 4, hspace=0.55, wspace=0.42,
                          left=0.05, right=0.97, top=0.96, bottom=0.05)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    ax_e = fig.add_subplot(gs[1, 0])
    ax_f_top = _panel_hd_tracking_stacked(
        fig, gs[1, 1], data["rollout"], data["dt_s"])
    if data.get("test_trial") is not None:
        ax_g_top = _panel_trial_rollout(fig, gs[1, 2], data["test_trial"])
    else:
        ax_g_top = fig.add_subplot(gs[1, 2])
        ax_g_top.axis("off")
    ax_h = fig.add_subplot(gs[1, 3])

    _panel_matrix(ax_a, data["W_con"],
                   data["neuron_types"], data["type_names"],
                   "GT $W_{\\mathrm{con}}$")
    _panel_label(ax_a, "a")

    _panel_matrix(ax_b, data["W_rec"],
                   data["neuron_types"], data["type_names"],
                   "learned $\\hat W_{\\mathrm{rec}}$")
    _panel_label(ax_b, "b")

    nt = np.asarray(data["neuron_types"])

    # (c) scatter of true (W_con) vs learned (W_rec) edge weights.
    _panel_weight_scatter(ax_c, data["W_con"], data["W_rec"])
    _panel_label(ax_c, "c")

    bump_label = data.get("bump_label", "EPG")
    # (d) all-neuron kinograph (previously panel c).
    _panel_all_neurons_kinograph(
        ax_d, np.asarray(data["rollout"]["r"]),
        neuron_types=data["neuron_types"], type_names=data["type_names"],
        dt_s=data["dt_s"],
    )
    _panel_label(ax_d, "d")

    epg_indices = data["net"].epg_indices
    _panel_neuron_kinograph(
        ax_e, np.asarray(data["rollout"]["r_epg"]),
        neuron_types_sub=nt[epg_indices], type_names=data["type_names"],
        dt_s=data["dt_s"], ylabel=f"{bump_label} neuron",
    )
    _panel_label(ax_e, "e")

    _panel_label(ax_f_top, "f")
    _panel_label(ax_g_top, "g")

    if data.get("calcium_panel") is not None:
        # Real-calcium training (zebrafish): panel h compares the recorded ΔF/F
        # against the model's voltage→GCaMP calcium on one held-out trial,
        # row-for-row in the same rastermap order.
        _panel_calcium_compare(ax_h, data["calcium_panel"])
    else:
        h_rollout = np.asarray(data["rollout"]["h"])
        _panel_voltage_distribution(
            ax_h, h_rollout,
            neuron_types=data["neuron_types"],
            type_names=data["type_names"],
        )
    _panel_label(ax_h, "h")

    if n_rows < 3:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    ax_i = fig.add_subplot(gs[2, 0])
    ax_j = fig.add_subplot(gs[2, 1])
    ax_k = fig.add_subplot(gs[2, 2])
    ax_l = fig.add_subplot(gs[2, 3])

    skip_extras = bool(run_dir and "frozen" in os.path.basename(
        os.path.abspath(run_dir)).lower())

    if skip_extras:
        for ax in (ax_i, ax_j, ax_k, ax_l):
            ax.axis("off")
    elif is_gnn:
        _panel_integration_gain(
            ax_i, data["gain_data"], data["test_trial"]["dt"],
        )
        _panel_label(ax_i, "i")
        _panel_embedding(
            ax_j, data["net"], data["neuron_types"], data["type_names"],
        )
        _panel_label(ax_j, "j")
        _panel_function_curves(
            ax_k, data["net"], "f_theta", h_rollout,
            neuron_types=data["neuron_types"],
            type_names=data["type_names"],
            square_output=False,
            xlabel=r"$\hat{h}_i$",
            ylabel=r"$f_\theta(\hat{h}_i, \mathbf{a}_i, m{=}0)$",
            title=r"$f_\theta$ (mean $\pm$ SD per type)",
        )
        _panel_label(ax_k, "k")
        _g_phi_pos = bool(getattr(data["net"], "_g_phi_positive", True))
        _panel_function_curves(
            ax_l, data["net"], "g_phi", h_rollout,
            neuron_types=data["neuron_types"],
            type_names=data["type_names"],
            square_output=True,
            xlabel=r"$\hat{h}_j$",
            ylabel=(r"$g_\phi(\hat{h}_j, \mathbf{a}_j)^2$" if _g_phi_pos
                     else r"$g_\phi(\hat{h}_j, \mathbf{a}_j)$"),
            title=(r"$g_\phi^2$ (mean $\pm$ SD per type)" if _g_phi_pos
                    else r"$g_\phi$ (mean $\pm$ SD per type)"),
        )
        _panel_label(ax_l, "l")
    else:
        _panel_integration_gain(
            ax_i, data["gain_data"], data["test_trial"]["dt"],
        )
        _panel_label(ax_i, "i")
        ax_j.axis("off")
        ax_k.axis("off")
        ax_l.axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_cx_evolution] wrote {out_path}")
