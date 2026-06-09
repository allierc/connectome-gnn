"""fig_zebrafish_swim_task_modes.py
=====================================

Explanatory figure for the zebrafish swim-integration task dataset and
its three sub-task projections (rotation / translation / both), plus the
leaky vs cumulative ξ-integrator variants.

What it shows
-------------
The generator (``connectome_gnn.generators.graph_data_generator``) writes
ONE on-disk dataset — a 4-channel input superset + 3-column target
superset — and the trainer projects it onto the active sub-task at
load time according to ``training.task_targets`` (see commit 5456da5).
The figure illustrates that projection plus the leaky-integrator option
(commit 5f1655d):

    (a) Swim event raster for a handful of trials — coloured ticks
        mark L / R / F / B onsets, drawn at swim_rate_hz=0.5,
        swim_duration_s=0.3, fractions L0.3/R0.3/F0.2/B0.2 (the
        production selfmotion recipe).
    (b) Stimulus (the 4-channel input superset) for one trial:
        u[:, 0] = ω, u[:, 1] = v_fwd, u[:, 2] = cosθ₀·δ_{t=0},
        u[:, 3] = sinθ₀·δ_{t=0}.
    (c) Target (the 3-column output superset) for the same trial:
        y[:, 0] = cosθ, y[:, 1] = sinθ, y[:, 2] = ξ. Panel c overlays
        the leaky ξ at τ = 0.5 s on top of the cumulative cumsum.
    (d) Channel-selection table — which input columns each mode keeps
        and which target columns it supervises:
            rotation     → in [0, 2, 3]    / out [0, 1]   (3-in / 2-out)
            translation  → in [1]           / out [2]      (1-in / 1-out)
            both         → in [0, 1, 2, 3]  / out [0, 1, 2] (4-in / 3-out)
    (e) Integrator τ sweep on the same trial: ξ(t) for
        τ ∈ {∞ (cumulative), 2.0 s, 1.0 s, 0.5 s}. Steady-state goes from
        unbounded (linear ramp) to bounded ≈ τ·v̄_fwd, illustrating why
        the leaky variant keeps the loss bounded across the curriculum.

The figure regenerates a small batch in memory with a fixed seed so it
is portable and reproducible — no on-disk dataset needed.

Usage
-----
    python figures/zebrafish/fig_zebrafish_swim_task_modes.py
    python figures/zebrafish/fig_zebrafish_swim_task_modes.py --out my.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# --- Production swim recipe (matches the artr_pt1_selfmotion configs) --------
DT = 0.01
T = 1000
SWIM_RATE_HZ = 0.5
SWIM_DURATION_S = 0.3
PHASE_IMPULSE_MEAN_RAD = math.radians(45.0)
PHASE_IMPULSE_STD_RAD = 0.4
FORWARD_VEL_MEAN = 1.0
FORWARD_VEL_STD = 0.4
L_FRAC, R_FRAC, F_FRAC, B_FRAC = 0.30, 0.30, 0.20, 0.20
LABEL_L, LABEL_R, LABEL_F, LABEL_B = 1, 2, 3, 4

# --- Style ------------------------------------------------------------------
GT_COLOR = "#2a9d3d"          # green (ground truth)
PRED_COLOR = "#000000"        # black (decoded)
LEAKY_COLOR = "#cc3333"       # red (leaky variant)
CHANNEL_COLORS = ("#1f77b4", "#ff7f0e", "#9467bd", "#7f7f7f")
SWIM_COLORS = {
    LABEL_L: "#d62728",       # red    — left turn
    LABEL_R: "#1f77b4",       # blue   — right turn
    LABEL_F: "#2ca02c",       # green  — forward
    LABEL_B: "#9467bd",       # purple — backward
}
SWIM_NAMES = {LABEL_L: "L", LABEL_R: "R", LABEL_F: "F", LABEL_B: "B"}

PANEL_LABEL_FS = 14   # panel-letter size unchanged; only the
                      # in-panel text scales up so the figure reads
                      # at \textwidth without crowding the labels.
TITLE_FS = 13
LABEL_FS = 12
TICK_FS = 11
LEGEND_FS = 13


def _panel_label(ax, letter: str, dx: float = -0.12, dy: float = 1.02):
    """Bold panel letter in the upper-left corner — matches plot_cx."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=PANEL_LABEL_FS, fontweight="bold",
            va="bottom", ha="right")


# ---------------------------------------------------------------------------
# Data generation — mirrors graph_data_generator._generate_swim_integration_task
# ---------------------------------------------------------------------------

def generate_swim_batch(B: int, *, seed: int = 0):
    """Generate one batch of swim-integration trials in memory.

    Returns
    -------
    stimulus : (B, T, 4) float32 — [ω, v_fwd, cosθ₀·δ, sinθ₀·δ]
    target   : (B, T, 3) float32 — [cosθ, sinθ, ξ_cumulative]
    swim_label_onset : (B, T) int8 — non-zero at onsets, value ∈ {L,R,F,B}
    omega    : (B, T) float32 — same as stimulus[..., 0]
    v_fwd    : (B, T) float32 — same as stimulus[..., 1]
    """
    rng = np.random.default_rng(seed)
    L = max(1, int(round(SWIM_DURATION_S / DT)))  # boxcar length in frames
    # Per-frame onset mask: each frame is a swim onset with prob = rate·dt
    onset = rng.random((B, T)) < (SWIM_RATE_HZ * DT)
    # Refractory period of L frames so two swim events don't overlap.
    onset_keep = np.zeros_like(onset)
    for b in range(B):
        last = -L
        for t in range(T):
            if onset[b, t] and (t - last) >= L:
                onset_keep[b, t] = True
                last = t
    onset = onset_keep

    # Category per onset (multinomial L/R/F/B).
    cat = np.zeros((B, T), dtype=np.int8)
    u = rng.random((B, T))
    is_on = onset
    p_L = L_FRAC
    p_R = L_FRAC + R_FRAC
    p_F = L_FRAC + R_FRAC + F_FRAC
    cat[is_on & (u <  p_L)] = LABEL_L
    cat[is_on & (u >= p_L) & (u < p_R)] = LABEL_R
    cat[is_on & (u >= p_R) & (u < p_F)] = LABEL_F
    cat[is_on & (u >= p_F)] = LABEL_B

    # Per-event magnitudes — lognormal on the configured means.
    sigma_log_LR = PHASE_IMPULSE_STD_RAD / max(PHASE_IMPULSE_MEAN_RAD, 1e-6)
    mag_LR = rng.lognormal(
        mean=math.log(max(PHASE_IMPULSE_MEAN_RAD, 1e-6)),
        sigma=sigma_log_LR, size=(B, T),
    )
    sigma_log_F = FORWARD_VEL_STD / max(FORWARD_VEL_MEAN, 1e-6)
    mag_F = rng.lognormal(
        mean=math.log(max(FORWARD_VEL_MEAN, 1e-6)),
        sigma=sigma_log_F, size=(B, T),
    )

    # Per-event signed Δθ at onset (rotation drive) + Δs_fwd at onset
    # (translation drive). L/R drive heading; F/B drive v_fwd (signed).
    delta_theta = np.zeros((B, T), dtype=np.float32)
    delta_fwd = np.zeros((B, T), dtype=np.float32)
    delta_theta[(cat == LABEL_L) & onset] = (+mag_LR[(cat == LABEL_L) & onset]).astype(np.float32)
    delta_theta[(cat == LABEL_R) & onset] = (-mag_LR[(cat == LABEL_R) & onset]).astype(np.float32)
    delta_fwd[(cat == LABEL_F) & onset] = (+mag_F[(cat == LABEL_F) & onset]).astype(np.float32)
    delta_fwd[(cat == LABEL_B) & onset] = (-mag_F[(cat == LABEL_B) & onset]).astype(np.float32)

    # Boxcar stretch over L frames so each onset persists for swim_duration_s.
    omega_rad = np.zeros((B, T), dtype=np.float32)
    vfwd = np.zeros((B, T), dtype=np.float32)
    for k in range(L):
        omega_rad[:, k:] += delta_theta[:, : T - k] / (L * DT)
        vfwd[:, k:] += delta_fwd[:, : T - k] / (L * DT)
    omega_deg = np.rad2deg(omega_rad)

    # Heading integration.
    theta0 = rng.uniform(0.0, 2.0 * math.pi, size=B).astype(np.float32)
    theta_hd = theta0[:, None] + np.cumsum(omega_rad, axis=1) * DT
    theta_hd[:, 0] = theta0

    # Perfect ξ for the on-disk target.
    disp = (np.cumsum(vfwd, axis=1) * DT).astype(np.float32)
    disp[:, 0] = 0.0

    target = np.stack(
        [np.cos(theta_hd), np.sin(theta_hd), disp], axis=-1
    ).astype(np.float32)
    stimulus = np.zeros((B, T, 4), dtype=np.float32)
    stimulus[:, :, 0] = omega_deg
    stimulus[:, :, 1] = vfwd
    stimulus[:, 0, 2] = np.cos(theta0)
    stimulus[:, 0, 3] = np.sin(theta0)

    swim_label_onset = np.where(onset, cat, np.int8(0))
    return stimulus, target, swim_label_onset, omega_deg, vfwd, theta_hd


def leaky_integrate(drive: np.ndarray, tau_s: "float | None") -> np.ndarray:
    """Forward-Euler integrator on `drive` (shape (B, T)).

    `tau_s` None / ≤ 0 → cumulative cumsum; finite τ > 0 → leaky recurrence
    with α = 1 − dt/τ. Initial condition is zero.
    """
    if tau_s is None or tau_s <= 0:
        out = (np.cumsum(drive, axis=1) * DT).astype(np.float32)
    else:
        alpha = max(0.0, min(1.0 - DT / float(tau_s), 1.0))
        out = np.zeros_like(drive, dtype=np.float32)
        for t in range(1, drive.shape[1]):
            out[:, t] = alpha * out[:, t - 1] + drive[:, t] * DT
    out[:, 0] = 0.0
    return out


def position_2d(vfwd: np.ndarray, theta_hd: np.ndarray,
                tau_s: "float | None"):
    """2D path integration: (x, y) = ∫ v_fwd · (cosθ, sinθ) dt.

    `tau_s` None → perfect 2D integrator (unbounded). Finite τ → leaky
    2D recurrence — both axes share the same time constant.
    Returns (x, y) each of shape (B, T).
    """
    vx = vfwd * np.cos(theta_hd)
    vy = vfwd * np.sin(theta_hd)
    return leaky_integrate(vx, tau_s), leaky_integrate(vy, tau_s)


# Back-compat alias used in earlier panels.
def leaky_xi(vfwd: np.ndarray, tau_s: "float | None") -> np.ndarray:
    return leaky_integrate(vfwd, tau_s)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _panel_swim_raster(ax, swim_label_onset: np.ndarray):
    """Coloured ticks at swim onsets, one row per trial."""
    B, T = swim_label_onset.shape
    t_axis = np.arange(T) * DT
    for b in range(B):
        for code in (LABEL_L, LABEL_R, LABEL_F, LABEL_B):
            ts = t_axis[swim_label_onset[b] == code]
            if ts.size:
                ax.vlines(ts, b - 0.4, b + 0.4,
                          colors=SWIM_COLORS[code], lw=1.4)
    ax.set_ylim(-0.6, B - 0.4)
    ax.set_yticks(range(B))
    ax.set_yticklabels([f"trial {i}" for i in range(B)], fontsize=TICK_FS)
    ax.set_xlim(0, T * DT)
    ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax.set_title("swim-event raster — colours: L / R / F / B",
                 fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    # Mini legend inline (no separate legend box — keeps the figure compact).
    for i, code in enumerate((LABEL_L, LABEL_R, LABEL_F, LABEL_B)):
        ax.text(0.02 + 0.06 * i, 1.04, SWIM_NAMES[code],
                color=SWIM_COLORS[code], transform=ax.transAxes,
                fontweight="bold", fontsize=LABEL_FS, ha="left", va="bottom")


def _draw_4ch_stack(fig, gs_cell, stim: np.ndarray, title: str):
    """Render 4 stacked sub-axes inside `gs_cell` for the 4 stim channels.

    Channels 2 and 3 (cos/sin θ₀ impulses) are delta-like — non-zero only
    at t=0 — so we render them with a stem marker + value annotation so
    the reader can see the impulse height instead of a single invisible
    line segment.
    """
    sub = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_cell, hspace=0.18)
    T = stim.shape[0]
    t_axis = np.arange(T) * DT
    labels = [r"$\omega$ (°/s)", r"$v_{\mathrm{fwd}}$",
              r"$\cos\theta_0\,\delta_{t=0}$",
              r"$\sin\theta_0\,\delta_{t=0}$"]
    ax0 = None
    for k in range(4):
        ax = fig.add_subplot(sub[k], sharex=ax0 if ax0 is not None else None)
        if ax0 is None:
            ax0 = ax
        if k < 2:
            # Continuous channels (ω, v_fwd) — normal line plot.
            ax.plot(t_axis, stim[:, k], color=CHANNEL_COLORS[k], lw=0.7)
        else:
            # Impulse channels — show t=0 value as a stem marker + label.
            v0 = float(stim[0, k])
            ax.plot(t_axis, stim[:, k], color=CHANNEL_COLORS[k], lw=0.0,
                    marker=".", ms=1.5)
            ax.vlines([0.0], 0.0, v0, colors=CHANNEL_COLORS[k], lw=1.8)
            ax.plot([0.0], [v0], "o", color=CHANNEL_COLORS[k], ms=5)
            ax.annotate(f"{v0:+.2f}", xy=(0.0, v0),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=TICK_FS, va="center", ha="left",
                        color=CHANNEL_COLORS[k])
            # Make sure y-axis spans the impulse value even when ≈0.
            pad = max(0.2, 0.25 * abs(v0))
            ax.set_ylim(min(-pad, v0 - pad), max(pad, v0 + pad))
        ax.axhline(0, color="0.7", lw=0.3)
        ax.set_ylabel(labels[k], fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS, labelbottom=(k == 3))
        if k == 3:
            ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    return ax0


def _draw_target_stack(fig, gs_cell, target: np.ndarray,
                       vfwd: np.ndarray, title: str):
    """Render the 3-col target stack with the leaky d overlay on column 2."""
    sub = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_cell, hspace=0.18)
    T = target.shape[0]
    t_axis = np.arange(T) * DT
    labels = [r"$\cos\theta$", r"$\sin\theta$", r"$d$"]
    ax0 = None
    for k in range(3):
        ax = fig.add_subplot(sub[k], sharex=ax0 if ax0 is not None else None)
        if ax0 is None:
            ax0 = ax
        ax.plot(t_axis, target[:, k], color=GT_COLOR, lw=1.0,
                label="perfect")
        if k == 2:
            xi_leaky = leaky_xi(vfwd[None, :], tau_s=0.5)[0]
            ax.plot(t_axis, xi_leaky, color=LEAKY_COLOR, lw=1.0, ls="--",
                    label=r"leaky $\tau=0.5\,$s")
            ax.legend(loc="upper left", fontsize=LEGEND_FS, frameon=False)
        ax.axhline(0, color="0.7", lw=0.3)
        ax.set_ylabel(labels[k], fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS, labelbottom=(k == 2))
        if k == 2:
            ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    return ax0


def _panel_mode_table(ax):
    """Static table — which input/output columns each mode supervises."""
    ax.axis("off")
    rows = [
        ("rotation",                 "[0, 2, 3]",   r"$[\omega,\cos\theta_0,\sin\theta_0]$",  "3", "[0, 1]",      r"$[\cos\theta,\sin\theta]$",        "2"),
        ("translation",              "[1]",         r"$[v_{\mathrm{fwd}}]$",                  "1", "[2]",         r"$[d]$",                            "1"),
        ("rotation + translation",   "[0,1,2,3]",   r"$[\omega,v_{\mathrm{fwd}},\cos\theta_0,\sin\theta_0]$", "4", "[0,1,2]",     r"$[\cos\theta,\sin\theta,d]$",      "3"),
    ]
    # Manual table layout — easier than matplotlib.table to format math.
    col_titles = ["mode (task_targets)", "in_cols", "model sees", r"$n_{\mathrm{in}}$",
                  "out_cols", "model predicts", r"$n_{\mathrm{out}}$"]
    col_x = [0.00, 0.25, 0.34, 0.61, 0.66, 0.76, 0.96]
    row_y = [0.78, 0.55, 0.32]
    # Header.
    for x, t in zip(col_x, col_titles):
        ax.text(x, 0.93, t, transform=ax.transAxes, fontsize=LABEL_FS,
                fontweight="bold", ha="left", va="bottom")
    # Separator line under header — drawn in axes coordinates.
    ax.plot([0.0, 1.0], [0.91, 0.91], color="0.2", lw=0.8,
            transform=ax.transAxes, clip_on=False)
    # Rows.
    for y, row in zip(row_y, rows):
        for x, cell in zip(col_x, row):
            ax.text(x, y, cell, transform=ax.transAxes, fontsize=LABEL_FS,
                    ha="left", va="center")
    ax.set_title("on-disk superset → per-mode I/O projection "
                 "(applied at trainer load time)", fontsize=TITLE_FS)


def _draw_position_2d_stack(fig, gs_cell, target: np.ndarray,
                             vfwd: np.ndarray, theta_hd: np.ndarray,
                             title: str):
    """Render the 4-stack 2D-PI target (cos θ, sin θ, x, y) with the leaky
    variant of (x, y) overlaid in red dashed."""
    sub = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_cell, hspace=0.18)
    T = target.shape[0]
    t_axis = np.arange(T) * DT
    x_perf, y_perf = position_2d(vfwd[None, :], theta_hd[None, :], tau_s=None)
    x_leak, y_leak = position_2d(vfwd[None, :], theta_hd[None, :], tau_s=0.5)
    rows = [
        (r"$\cos\theta$", target[:, 0], None,        None),
        (r"$\sin\theta$", target[:, 1], None,        None),
        (r"$x$",          x_perf[0],    x_leak[0],   r"leaky $\tau=0.5\,$s"),
        (r"$y$",          y_perf[0],    y_leak[0],   r"leaky $\tau=0.5\,$s"),
    ]
    ax0 = None
    for k, (lab, perfect, leaky, leaky_label) in enumerate(rows):
        ax = fig.add_subplot(sub[k], sharex=ax0 if ax0 is not None else None)
        if ax0 is None:
            ax0 = ax
        ax.plot(t_axis, perfect, color=GT_COLOR, lw=1.0,
                label="perfect" if leaky is not None else None)
        if leaky is not None:
            ax.plot(t_axis, leaky, color=LEAKY_COLOR, lw=1.0, ls="--",
                    label=leaky_label)
            if k == 2:
                ax.legend(loc="upper left", fontsize=LEGEND_FS, frameon=False)
        ax.axhline(0, color="0.7", lw=0.3)
        ax.set_ylabel(lab, fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS, labelbottom=(k == 3))
        if k == 3:
            ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    return ax0


def _panel_2d_path_tau_sweep(ax, vfwd: np.ndarray, theta_hd: np.ndarray):
    """(x, y) trajectory for two integrator recipes on the same trial:
    perfect (green solid) and leaky τ=0.5 s (red dashed) — same colour /
    style convention as panel (c).
    """
    x_perf, y_perf = position_2d(vfwd[None, :], theta_hd[None, :], tau_s=None)
    x_leak, y_leak = position_2d(vfwd[None, :], theta_hd[None, :], tau_s=0.5)
    ax.plot(x_perf[0], y_perf[0], color=GT_COLOR, lw=1.2, label="perfect")
    ax.plot(x_leak[0], y_leak[0], color=LEAKY_COLOR, lw=1.0, ls="--",
            label=r"leaky $\tau=0.5\,$s")
    ax.plot([0.0], [0.0], "o", color="0.4", ms=5, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axhline(0, color="0.85", lw=0.3, zorder=0)
    ax.axvline(0, color="0.85", lw=0.3, zorder=0)
    ax.set_xlabel("x", fontsize=LABEL_FS)
    ax.set_ylabel("y", fontsize=LABEL_FS)
    ax.legend(loc="best", fontsize=LEGEND_FS, frameon=False)
    ax.tick_params(labelsize=TICK_FS)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure(out_path: str, seed: int = 3):
    # Generate a small batch — only trial 0 is shown; B=5 just lets the
    # event sampler average out a bit per-trial so trial 0 isn't a
    # degenerate empty trial.
    stim, target, sw_label, omega_deg, vfwd, theta_hd = generate_swim_batch(
        B=5, seed=seed)

    fig = plt.figure(figsize=(13, 11))
    gs = GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.2, 1.2],
        width_ratios=[1.0, 1.0],
        hspace=0.42, wspace=0.30,
        left=0.07, right=0.97, top=0.97, bottom=0.06,
    )

    # (a) Stimulus — 4 stacked channels (trial 0).
    ax_a_top = _draw_4ch_stack(
        fig, gs[0, 0], stim[0],
        title="stimulus — 4-channel input superset (trial 0)",
    )
    _panel_label(ax_a_top, "a", dx=-0.16)

    # (b) scalar_d target — 3 stacked columns, d overlays perfect + leaky.
    ax_b_top = _draw_target_stack(
        fig, gs[0, 1], target[0], vfwd[0],
        title=r"scalar-$d$ target — $[\cos\theta,\sin\theta,d]$ (trial 0)",
    )
    _panel_label(ax_b_top, "b", dx=-0.16)

    # (c) position_2d target — 4 stacked columns, (x, y) overlay perfect +
    # leaky on the same trial.
    ax_c_top = _draw_position_2d_stack(
        fig, gs[1, 0], target[0], vfwd[0], theta_hd[0],
        title=r"position-2D target — $[\cos\theta,\sin\theta,x,y]$ (trial 0)",
    )
    _panel_label(ax_c_top, "c", dx=-0.16)

    # (d) 2D path τ sweep — square spatial plot.
    ax_d = fig.add_subplot(gs[1, 1])
    _panel_2d_path_tau_sweep(ax_d, vfwd[0], theta_hd[0])
    _panel_label(ax_d, "d", dx=-0.12)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fig_zebrafish_swim_task_modes.png",
    )
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    build_figure(args.out, seed=args.seed)


if __name__ == "__main__":
    main()
