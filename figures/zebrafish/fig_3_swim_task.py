"""fig_3_swim_task.py
=====================================

Explanatory figure for the zebrafish swim-integration task (scalar-d
variant) and the leaky vs cumulative integrator.

What it shows
-------------
The figure regenerates a single trial in memory with a fixed seed so it
is portable and reproducible — no on-disk dataset needed.

    (a) Stimulus (the 4-channel input superset) for one trial:
        u[:, 0] = ω, u[:, 1] = v_ext, u[:, 2] = cosθ₀·δ_{t=0},
        u[:, 3] = sinθ₀·δ_{t=0}.
    (b) Scalar-d target for the same trial:
        y[:, 0] = cosθ, y[:, 1] = sinθ, y[:, 2] = d (forward distance).
        Panel b overlays the leaky d at τ = 0.5 s on the perfect cumsum.
    (c) 2D trajectory in spatial coordinates for the same trial,
        perfect (green) vs leaky τ = 0.5 s (red). The trajectory is
        computed from the same ω / v_ext inputs shown in panel a; the
        leaky path is bounded ≈ τ · v̄_ext, while the cumulative path
        drifts unboundedly from the origin.
    (d) Proprioceptive-gain MISMATCH task for the same trial's ω:
        a piecewise-constant gain g(t) ∈ [0, 1.5] sets the
        proprioceptive copy ω_proprio = g·ω (routed to motor_efferent),
        and the supervised scalar is the integrated mismatch
        ∫(ω − ω_proprio) dt.

Usage
-----
    python figures/zebrafish/fig_3_swim_task.py
    python figures/zebrafish/fig_3_swim_task.py --out my.png
"""

from __future__ import annotations

import argparse
import math
import os

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
GT_COLOR = "#2a9d3d"          # green (perfect / ground-truth integral)
LEAKY_COLOR = "#cc3333"       # red (leaky variant)
CHANNEL_COLORS = ("#1f77b4", "#ff7f0e", "#9467bd", "#7f7f7f")

# Font sizes bumped uniformly so the figure reads at \textwidth with
# the new 1×3 layout (the previous 2×2 had more vertical room per panel).
PANEL_LABEL_FS = 18
LABEL_FS = 15
TICK_FS = 13
LEGEND_FS = 14


def _panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.04):
    """Bold panel letter in the upper-left corner."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=PANEL_LABEL_FS, fontweight="bold",
            va="bottom", ha="right")


# ---------------------------------------------------------------------------
# Data generation — mirrors graph_data_generator._generate_swim_integration_task
# ---------------------------------------------------------------------------

def generate_swim_batch(B: int, *, seed: int = 0):
    """Generate one batch of swim-integration trials in memory."""
    rng = np.random.default_rng(seed)
    L = max(1, int(round(SWIM_DURATION_S / DT)))
    onset = rng.random((B, T)) < (SWIM_RATE_HZ * DT)
    onset_keep = np.zeros_like(onset)
    for b in range(B):
        last = -L
        for t in range(T):
            if onset[b, t] and (t - last) >= L:
                onset_keep[b, t] = True
                last = t
    onset = onset_keep

    cat = np.zeros((B, T), dtype=np.int8)
    u = rng.random((B, T))
    p_L = L_FRAC
    p_R = L_FRAC + R_FRAC
    p_F = L_FRAC + R_FRAC + F_FRAC
    is_on = onset
    cat[is_on & (u <  p_L)] = LABEL_L
    cat[is_on & (u >= p_L) & (u < p_R)] = LABEL_R
    cat[is_on & (u >= p_R) & (u < p_F)] = LABEL_F
    cat[is_on & (u >= p_F)] = LABEL_B

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

    delta_theta = np.zeros((B, T), dtype=np.float32)
    delta_fwd = np.zeros((B, T), dtype=np.float32)
    delta_theta[(cat == LABEL_L) & onset] = (+mag_LR[(cat == LABEL_L) & onset]).astype(np.float32)
    delta_theta[(cat == LABEL_R) & onset] = (-mag_LR[(cat == LABEL_R) & onset]).astype(np.float32)
    delta_fwd[(cat == LABEL_F) & onset] = (+mag_F[(cat == LABEL_F) & onset]).astype(np.float32)
    delta_fwd[(cat == LABEL_B) & onset] = (-mag_F[(cat == LABEL_B) & onset]).astype(np.float32)

    omega_rad = np.zeros((B, T), dtype=np.float32)
    vfwd = np.zeros((B, T), dtype=np.float32)
    for k in range(L):
        omega_rad[:, k:] += delta_theta[:, : T - k] / (L * DT)
        vfwd[:, k:] += delta_fwd[:, : T - k] / (L * DT)
    omega_deg = np.rad2deg(omega_rad)

    theta0 = rng.uniform(0.0, 2.0 * math.pi, size=B).astype(np.float32)
    theta_hd = theta0[:, None] + np.cumsum(omega_rad, axis=1) * DT
    theta_hd[:, 0] = theta0

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
    return stimulus, target, omega_deg, vfwd, theta_hd


def leaky_integrate(drive: np.ndarray, tau_s):
    if tau_s is None or tau_s <= 0:
        out = (np.cumsum(drive, axis=1) * DT).astype(np.float32)
    else:
        alpha = max(0.0, min(1.0 - DT / float(tau_s), 1.0))
        out = np.zeros_like(drive, dtype=np.float32)
        for t in range(1, drive.shape[1]):
            out[:, t] = alpha * out[:, t - 1] + drive[:, t] * DT
    out[:, 0] = 0.0
    return out


def position_2d(vfwd, theta_hd, tau_s):
    vx = vfwd * np.cos(theta_hd)
    vy = vfwd * np.sin(theta_hd)
    return leaky_integrate(vx, tau_s), leaky_integrate(vy, tau_s)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _draw_4ch_stack(fig, gs_cell, stim: np.ndarray):
    """4 stacked sub-axes for the 4 stim channels."""
    sub = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_cell, hspace=0.20)
    T_ = stim.shape[0]
    t_axis = np.arange(T_) * DT
    labels = [r"$\omega$ (°/s)", r"$v_{\mathrm{ext}}$",
              r"$\cos\theta_0\,\delta_{t=0}$",
              r"$\sin\theta_0\,\delta_{t=0}$"]
    ax0 = None
    for k in range(4):
        ax = fig.add_subplot(sub[k], sharex=ax0 if ax0 is not None else None)
        if ax0 is None:
            ax0 = ax
        if k < 2:
            ax.plot(t_axis, stim[:, k], color=CHANNEL_COLORS[k], lw=0.8)
        else:
            v0 = float(stim[0, k])
            ax.plot(t_axis, stim[:, k], color=CHANNEL_COLORS[k], lw=0.0,
                    marker=".", ms=1.5)
            ax.vlines([0.0], 0.0, v0, colors=CHANNEL_COLORS[k], lw=1.8)
            ax.plot([0.0], [v0], "o", color=CHANNEL_COLORS[k], ms=5)
            ax.annotate(f"{v0:+.2f}", xy=(0.0, v0),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=TICK_FS, va="center", ha="left",
                        color=CHANNEL_COLORS[k])
            pad = max(0.2, 0.25 * abs(v0))
            ax.set_ylim(min(-pad, v0 - pad), max(pad, v0 + pad))
        ax.axhline(0, color="0.7", lw=0.3)
        ax.set_ylabel(labels[k], fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS, labelbottom=(k == 3))
        if k == 3:
            ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    return ax0


def _draw_target_stack(fig, gs_cell, target: np.ndarray, vfwd: np.ndarray):
    """3-col target stack: cosθ, sinθ, d; perfect vs leaky on d."""
    sub = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_cell, hspace=0.20)
    T_ = target.shape[0]
    t_axis = np.arange(T_) * DT
    labels = [r"$\cos\theta$", r"$\sin\theta$", r"$d$"]
    ax0 = None
    for k in range(3):
        ax = fig.add_subplot(sub[k], sharex=ax0 if ax0 is not None else None)
        if ax0 is None:
            ax0 = ax
        ax.plot(t_axis, target[:, k], color=GT_COLOR, lw=1.0,
                label="perfect")
        if k == 2:
            d_leaky = leaky_integrate(vfwd[None, :], tau_s=0.5)[0]
            ax.plot(t_axis, d_leaky, color=LEAKY_COLOR, lw=1.0, ls="--",
                    label=r"leaky $\tau=0.5\,$s")
            ax.legend(loc="upper left", fontsize=LEGEND_FS, frameon=False)
        ax.axhline(0, color="0.7", lw=0.3)
        ax.set_ylabel(labels[k], fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS, labelbottom=(k == 2))
        if k == 2:
            ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    return ax0


def _panel_2d_path(ax, vfwd: np.ndarray, theta_hd: np.ndarray):
    """(x, y) trajectory for the same trial — perfect vs leaky τ=0.5 s.

    Computed from the same ω / v_ext inputs shown in panel a, so this
    panel is the spatial-coordinate view of the same trial whose scalar
    d-target is shown in panel b.
    """
    x_perf, y_perf = position_2d(vfwd[None, :], theta_hd[None, :], tau_s=None)
    x_leak, y_leak = position_2d(vfwd[None, :], theta_hd[None, :], tau_s=0.5)
    ax.plot(x_perf[0], y_perf[0], color=GT_COLOR, lw=1.4, label="perfect")
    ax.plot(x_leak[0], y_leak[0], color=LEAKY_COLOR, lw=1.2, ls="--",
            label=r"leaky $\tau=0.5\,$s")
    ax.plot([0.0], [0.0], "o", color="0.4", ms=6, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axhline(0, color="0.85", lw=0.3, zorder=0)
    ax.axvline(0, color="0.85", lw=0.3, zorder=0)
    ax.set_xlabel("x", fontsize=LABEL_FS)
    ax.set_ylabel("y", fontsize=LABEL_FS)
    ax.legend(loc="best", fontsize=LEGEND_FS, frameon=False)
    ax.tick_params(labelsize=TICK_FS)


# --- Proprioceptive-gain mismatch task (target_kind = rotation_mismatch) -----
PROPRIO_COLOR = "#e8820c"     # orange (proprioceptive copy ω_proprio)
GAIN_COLOR = "#9467bd"        # purple (gain g(t))


def mismatch_signals(omega_deg, *, seed=0, g_min=0.0, g_max=1.5, seg_s=2.0):
    """Piecewise-constant gain g(t), proprioceptive copy ω_proprio = g·ω,
    and the integrated mismatch ∫(ω − ω_proprio) dt (radians)."""
    rng = np.random.default_rng(seed)
    T_ = omega_deg.shape[0]
    n_seg = max(1, int(round((T_ * DT) / max(seg_s, DT))))
    seg_gains = rng.uniform(g_min, g_max, size=n_seg).astype(np.float32)
    g = np.zeros(T_, dtype=np.float32)
    bounds = np.linspace(0, T_, n_seg + 1).astype(int)
    for s in range(n_seg):
        g[bounds[s]:bounds[s + 1]] = seg_gains[s]
    omega_pro = (g * omega_deg).astype(np.float32)
    mismatch = (np.cumsum(np.deg2rad(omega_deg - omega_pro)) * DT).astype(np.float32)
    mismatch[0] = 0.0
    return g, omega_pro, mismatch


def _draw_mismatch_stack(fig, gs_cell, omega_deg, *, seed=0):
    """3 stacked sub-axes depicting the mismatch task: g(t); ω vs ω_proprio;
    the integrated-mismatch target."""
    sub = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_cell, hspace=0.20)
    T_ = omega_deg.shape[0]
    t_axis = np.arange(T_) * DT
    g, omega_pro, mismatch = mismatch_signals(omega_deg, seed=seed)

    ax0 = fig.add_subplot(sub[0])
    ax0.plot(t_axis, g, color=GAIN_COLOR, lw=1.3)
    ax0.axhline(1.0, color="0.7", lw=0.5, ls=":")
    ax0.set_ylim(-0.1, 1.65)
    ax0.set_ylabel(r"$g(t)$", fontsize=LABEL_FS)
    ax0.tick_params(labelsize=TICK_FS, labelbottom=False)

    ax1 = fig.add_subplot(sub[1], sharex=ax0)
    ax1.plot(t_axis, omega_deg, color=GT_COLOR, lw=0.8, label=r"$\omega$")
    ax1.plot(t_axis, omega_pro, color=PROPRIO_COLOR, lw=0.8,
             label=r"$\omega_{\mathrm{proprio}}=g\,\omega$")
    ax1.axhline(0, color="0.7", lw=0.3)
    ax1.set_ylabel(r"°/s", fontsize=LABEL_FS)
    ax1.legend(loc="upper right", fontsize=LEGEND_FS - 3, frameon=False)
    ax1.tick_params(labelsize=TICK_FS, labelbottom=False)

    ax2 = fig.add_subplot(sub[2], sharex=ax0)
    ax2.plot(t_axis, mismatch, color=GT_COLOR, lw=1.2)
    ax2.axhline(0, color="0.7", lw=0.3)
    ax2.set_ylabel(r"$\int(\omega-\omega_{\mathrm{proprio}})\,dt$",
                   fontsize=LABEL_FS - 2)
    ax2.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax2.tick_params(labelsize=TICK_FS)
    return ax0


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure(out_path: str, seed: int = 3):
    stim, target, omega_deg, vfwd, theta_hd = generate_swim_batch(
        B=5, seed=seed)

    fig = plt.figure(figsize=(22.5, 6.0))
    gs = GridSpec(
        1, 4, figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 1.0],
        wspace=0.34,
        left=0.04, right=0.985, top=0.95, bottom=0.10,
    )

    ax_a_top = _draw_4ch_stack(fig, gs[0, 0], stim[0])
    _panel_label(ax_a_top, "a")

    ax_b_top = _draw_target_stack(fig, gs[0, 1], target[0], vfwd[0])
    _panel_label(ax_b_top, "b")

    ax_c = fig.add_subplot(gs[0, 2])
    _panel_2d_path(ax_c, vfwd[0], theta_hd[0])
    _panel_label(ax_c, "c", dx=-0.12)

    ax_d_top = _draw_mismatch_stack(fig, gs[0, 3], omega_deg[0], seed=seed)
    _panel_label(ax_d_top, "d")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fig_3_swim_task.png",
    )
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    build_figure(args.out, seed=args.seed)


if __name__ == "__main__":
    main()
