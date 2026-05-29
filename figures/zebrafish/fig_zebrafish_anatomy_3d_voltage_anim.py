"""Voltage animation on the 3-D zebrafish HD anatomy.

Runs the trained zebrafish_hd_si RNN under a rollout (constant-omega or
swim-integration), computes per-neuron z-scored activation, and renders
one PNG every K frames showing every HD skeleton in dark grey overlaid
with a green tint whose alpha is the current z-score. Output:
figures/zebrafish/3D_voltage_<suffix>/frame_NNNN.png.

The geometry comes from the SWC pull in zebrafish_anatomy_HD/ (produced
by fetch_zebrafish_anatomy_HD.py). The model -> bodyId mapping replays
load_zebrafish_hd_connectome's neuron ordering so model index i lines
up with the correct skeleton.

Cell-type categories (4 groups, mirroring the static anatomy script):
    IPNd    dorsal IPN (the HD ring per Petrucco et al. 2023)
    IPNds   dorsal-subset IPN
    RIPN    habenula -> IPN afferents
    pt-IPN  pretectum -> IPN afferents
"""
from __future__ import annotations

import argparse
import glob as _glob
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm

from connectome_gnn.utils import log_path, load_data_root_from_json, set_data_root
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.models.registry import create_model
from connectome_gnn.task_state import TaskTrials
from connectome_gnn.generators.connconstr_data import load_zebrafish_hd_connectome

from fig_zebrafish_anatomy_3d_HD import (
    TYPE_COLOR, TYPE_ORDER, _load_rois, _project_2d, _type_to_category,
    CORE_ROIS,
)


# Four anatomical ROIs (Fiji rectangles) drawn over the dorsal anatomy
# panel of the rendered frame. Coordinates are in *Fiji image pixels*
# (top-left origin, x_topleft, y_topleft, width, height) measured on
# a no-swim full anim render with figsize=(10.0, 14.6) at dpi=300
# (≈ 3000 × 4380 pixels — calibrated against the user-supplied
# top-left makeRectangle(6, 0, 330, 198) and bottom-right
# makeRectangle(2664, 4176, 330, 198) ROIs).
ZHD_ROI_RECTS_FIJI = [
    # name,   x_topleft, y_topleft, w, h
    ("dIPN-L",  1404, 2508, 456, 180),  # dIPN ring, left hemisphere
    ("dIPN-R",  1404, 2688, 456, 180),  # dIPN ring, right hemisphere  (x aligned with L)
    ("dsIPN-L", 1182, 2472, 330, 198),  # dorsal-subset IPN, left
    ("dsIPN-R", 1182, 2682, 330, 198),  # dorsal-subset IPN, right     (x aligned with L)
]
ZHD_ROI_REF_W = 3000   # canonical render width (10.0 * dpi 300)
ZHD_ROI_REF_H = 4380   # canonical render height (14.6 * dpi 300)
ZHD_ROI_COLORS = [
    (0.95, 0.30, 0.30),   # dIPN-L  red
    (0.30, 0.55, 0.95),   # dIPN-R  blue
    (0.95, 0.65, 0.20),   # dsIPN-L orange
    (0.40, 0.85, 0.40),   # dsIPN-R green
]


# Dorsal panel position in the no-swim reference render (figsize 10×14.6,
# GridSpec [5, 5, 1, 1], top=0.995, bottom=0.04, hspace=0, left=0.10,
# right=0.97). The Fiji ROIs were measured on that render so we convert
# them to dorsal-axis-fraction once and then draw with ax.transAxes,
# which makes the overlay follow the dorsal panel regardless of whether
# the surrounding figure has the no-swim trace strip (4 trace rows of 1)
# or the with-swim layout (also adds L/R + F/B trace rows).
ZHD_ROI_REF_DORSAL_BBOX = dict(x0=0.10, x1=0.97, y0=0.1991, y1=0.5971)


def _fiji_rects_to_dorsal_axis_fracs():
    """Convert ZHD_ROI_RECTS_FIJI to (x, y, w, h) in dorsal-axis fraction
    coords (origin at bottom-left of the dorsal panel)."""
    bb = ZHD_ROI_REF_DORSAL_BBOX
    bb_w = bb["x1"] - bb["x0"]
    bb_h = bb["y1"] - bb["y0"]
    out = []
    for name, x, y, w, h in ZHD_ROI_RECTS_FIJI:
        # First: Fiji-pixel → figure-fraction in the reference render
        fx_l = x / ZHD_ROI_REF_W
        fy_b = 1.0 - (y + h) / ZHD_ROI_REF_H      # flip y top→bottom
        fw_f = w / ZHD_ROI_REF_W
        fh_f = h / ZHD_ROI_REF_H
        # Then: figure-fraction → dorsal-axis-fraction
        ax_x = (fx_l - bb["x0"]) / bb_w
        ax_y = (fy_b - bb["y0"]) / bb_h
        ax_w = fw_f / bb_w
        ax_h = fh_f / bb_h
        out.append((name, ax_x, ax_y, ax_w, ax_h))
    return out


def _extract_zhd_roi_intensities(fig, channel=1, force_draw=True):
    """Mean channel intensity inside each ROI rectangle from the figure's
    current canvas. Mirrors drosophila's _extract_roi_intensities but for
    axis-aligned rectangles defined in dorsal-axis fractions instead of
    3-D wedge polygons. Returns (n_rois,) float array in [0, 255]."""
    if len(fig.axes) < 2:
        return np.zeros(len(ZHD_ROI_RECTS_FIJI), dtype=np.float32)
    dorsal_ax = fig.axes[1]
    if force_draw:
        fig.canvas.draw()
    Wpx, Hpx = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    if buf.shape[0] != Hpx or buf.shape[1] != Wpx:
        buf = buf.reshape(Hpx, Wpx, 4)
    img = buf[..., :3]
    fracs = _fiji_rects_to_dorsal_axis_fracs()
    intens = np.zeros(len(fracs), dtype=np.float32)
    for k, (name, ax_x, ax_y, ax_w, ax_h) in enumerate(fracs):
        x_disp_lo, y_disp_lo = dorsal_ax.transAxes.transform((ax_x, ax_y))
        x_disp_hi, y_disp_hi = dorsal_ax.transAxes.transform(
            (ax_x + ax_w, ax_y + ax_h))
        x0 = int(max(np.floor(min(x_disp_lo, x_disp_hi)), 0))
        x1 = int(min(np.ceil(max(x_disp_lo, x_disp_hi)), Wpx))
        # display origin is bottom-left; image origin is top-left → flip y
        y0 = int(max(np.floor(Hpx - max(y_disp_lo, y_disp_hi)), 0))
        y1 = int(min(np.ceil(Hpx - min(y_disp_lo, y_disp_hi)), Hpx))
        if x1 <= x0 or y1 <= y0:
            continue
        intens[k] = float(img[y0:y1, x0:x1, channel].mean())
    return intens


def _paint_zhd_roi_kinograph(ax, intens_buf, t_buf, scroll_window=10.0,
                              bg="black", init_skip_s=1.0):
    """Paint ΔF/F0 kinograph + image-based PVA on a single axes.
    Adapted verbatim from drosophila's _paint_roi_kinograph (4 ROIs
    instead of 16; ROI labels from ZHD_ROI_RECTS_FIJI).

    ``init_skip_s`` discards the network's start-up transient when
    computing the F0 baseline, the vmax for the green colormap, and the
    PVA trace, so the steady-state dynamics aren't compressed by the
    init spike. The kinograph image itself still spans the full time."""
    ax.clear()
    ax.set_facecolor(bg)
    txt_color = "white" if bg == "black" else "black"
    if intens_buf.shape[0] == 0:
        return

    n_w = intens_buf.shape[1]
    t_now = float(t_buf[-1])
    if t_now < scroll_window:
        x_lo, x_hi = 0.0, scroll_window
    else:
        x_lo, x_hi = t_now - scroll_window, t_now

    # F0 / vmax / PVA exclude the first init_skip_s seconds. Fall back
    # to the full buffer when we haven't yet accumulated enough post-init
    # samples (early frames).
    post = t_buf >= float(init_skip_s)
    if int(post.sum()) >= 4:
        F0 = intens_buf[post].mean(axis=0) + 1e-6
    else:
        F0 = intens_buf.mean(axis=0) + 1e-6
    dff = (intens_buf - F0[None, :]) / F0[None, :]

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "BlackGreen", [(0.0, 0.0, 0.0), (0.0, 1.0, 0.3)],
    )
    cmap.set_bad((0, 0, 0))
    if int(post.sum()) >= 4:
        vmax = float(np.percentile(np.clip(dff[post], 0, None), 95))
    else:
        vmax = float(np.percentile(np.clip(dff, 0, None), 95))
    if vmax <= 0:
        vmax = max(float(np.abs(dff).max()), 1e-3)
    ax.imshow(
        dff.T,
        aspect="auto", origin="lower",
        extent=(float(t_buf[0]), float(t_buf[-1]), 0.5, n_w + 0.5),
        cmap=cmap, vmin=0.0, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(n_w + 0.5, 0.5)
    roi_names = [n for n, *_ in ZHD_ROI_RECTS_FIJI]
    ax.set_yticks(np.arange(1, n_w + 1))
    ax.set_yticklabels(roi_names[:n_w], color=txt_color, fontsize=8)
    ax.set_ylabel("dIPN ROI", color=txt_color, fontsize=10, labelpad=2)
    ax.tick_params(axis="x", colors=txt_color, labelsize=9, length=3)
    ax.tick_params(axis="y", colors=txt_color, length=3)
    ax.set_xlabel("time (s)", color=txt_color, fontsize=10, labelpad=2)
    ax.spines[:].set_visible(False)

    # Per-ROI PVA: one trace per ROI, drawn on its own kinograph row in
    # the ROI's overlay color. The trace is the post-init ΔF/F0
    # normalised to its 95th-percentile and pinned to the row band
    # (offset ranges within ±0.45 of the row centre), so each row's line
    # shows the ROI's relative time-course at a glance.
    if int(post.sum()) >= 2:
        t_pva = t_buf[post]
        dff_pva = dff[post]
    else:
        t_pva = t_buf
        dff_pva = dff
    for i in range(n_w):
        row_y = i + 1
        d_i = dff_pva[:, i]
        d_max = max(float(np.percentile(np.clip(d_i, 0.0, None), 95)), 1e-6)
        norm_i = np.clip(d_i / d_max, 0.0, 1.0)
        ax.plot(t_pva, row_y - 0.45 * norm_i,
                color=ZHD_ROI_COLORS[i], lw=1.2, zorder=4)


def _draw_zhd_roi_overlay(fig, lw=1.4):
    """Draw the four anatomical ROI rectangles on the dorsal panel in
    axis-fraction coords. axes[1] is the dorsal panel (axes[0] is the
    frontal anatomy panel)."""
    from matplotlib.patches import Rectangle
    if len(fig.axes) < 2:
        return
    dorsal_ax = fig.axes[1]
    for (name, ax_x, ax_y, ax_w, ax_h), color in zip(
            _fiji_rects_to_dorsal_axis_fracs(), ZHD_ROI_COLORS):
        rect = Rectangle((ax_x, ax_y), ax_w, ax_h,
                          transform=dorsal_ax.transAxes,
                          fill=False, edgecolor=color,
                          linewidth=lw, zorder=10000,
                          clip_on=False)
        dorsal_ax.add_patch(rect)
        dorsal_ax.text(ax_x + ax_w / 2, ax_y + ax_h + 0.01, name,
                       transform=dorsal_ax.transAxes,
                       ha="center", va="bottom",
                       color=color, fontsize=8, fontweight="bold",
                       zorder=10001, clip_on=False)

# ── model loading (same pattern as fig_kinographs_const_omega._load) ──────

def _load(config_name, device, prefer_epoch=None):
    config, _ = load_run_config(config_name, explicit_output_root=False,
                                task="train")
    ckpt_dir = os.path.join(log_path(config.config_file), "models")
    cands = sorted(
        _glob.glob(os.path.join(ckpt_dir, "best_model_with_0_graphs_*.pt")),
        key=lambda p_: int(p_.rsplit("_", 1)[1].rstrip(".pt")),
    )
    if not cands:
        raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
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
    return model, config


# ── rollout helpers ───────────────────────────────────────────────────────

def _build_const_omega_batch(n_steps, dt, omega_deg, theta0, device):
    """Deterministic constant-omega TaskTrials (B=1)."""
    T = int(n_steps)
    omega = np.full((1, T), float(omega_deg), dtype=np.float32)
    omega_rad = np.deg2rad(omega)
    theta_hd = float(theta0) + np.cumsum(omega_rad, axis=1) * dt
    theta_hd[:, 0] = float(theta0)
    u = np.zeros((1, T, 3), dtype=np.float32)
    u[:, :, 0] = omega
    u[:, 0, 1] = math.cos(float(theta0))
    u[:, 0, 2] = math.sin(float(theta0))
    y = np.stack([np.cos(theta_hd), np.sin(theta_hd)],
                 axis=-1).astype(np.float32)
    is_stop = np.zeros((1, T), dtype=np.float32)
    return TaskTrials(
        task_family="path_integration",
        n_input=3, n_output=2, dt=float(dt),
        stimulus=torch.from_numpy(u).to(device),
        target=torch.from_numpy(y).to(device),
        theta_hd=torch.from_numpy(theta_hd).to(device),
        is_stop=torch.from_numpy(is_stop).to(device),
        omega=torch.from_numpy(omega).to(device),
    )


def _run_const(net, n_steps, dt, omega_deg, theta0, device):
    batch = _build_const_omega_batch(n_steps, dt, omega_deg, theta0, device)
    theta = batch.theta_hd[0].cpu().numpy()
    omega = batch.omega[0].cpu().numpy()
    with torch.no_grad():
        y_hat, h = net(batch.stimulus)
    decoded_hd = np.arctan2(y_hat[0, :, 1].cpu().numpy(),
                            y_hat[0, :, 0].cpu().numpy())
    return h[0].cpu().numpy(), theta, omega, decoded_hd, None, None


def _build_swim_batch(n_steps, dt, device, seed=0,
                      swim_rate_hz=0.5, swim_duration_s=0.3,
                      phase_impulse_mean_rad=0.785,
                      phase_impulse_std_rad=0.40,
                      backward_phase_mean_rad=3.14,
                      backward_phase_std_rad=0.30,
                      left_fraction=0.40, right_fraction=0.40,
                      forward_fraction=0.15, backward_fraction=0.05):
    """Single-trial swim-integration stimulus (B=1).

    Default fractions are the original turn-heavy training distribution
    (L/R = 80%) used by ``--swim``. ``--swim2`` overrides these with
    a Petrucco/larval-zebrafish-realistic mix (~65% forward, ~17% L/R,
    ~1% backward).
    """
    rng = np.random.default_rng(seed)
    T = int(n_steps)
    L = max(1, int(round(swim_duration_s / dt)))
    p_swim = swim_rate_hz * dt

    cdf = np.cumsum([left_fraction, right_fraction,
                     forward_fraction, backward_fraction])

    onset = rng.uniform(size=T) < p_swim
    u_type = rng.uniform(size=T)
    cat = np.digitize(u_type, cdf[:-1]) + 1  # 1=L, 2=R, 3=F, 4=B

    sigma_log_LR = phase_impulse_std_rad / max(phase_impulse_mean_rad, 1e-6)
    sigma_log_B = backward_phase_std_rad / max(backward_phase_mean_rad, 1e-6)
    mag_LR = rng.lognormal(
        mean=math.log(max(phase_impulse_mean_rad, 1e-6)),
        sigma=sigma_log_LR, size=T).astype(np.float32)
    mag_B = rng.lognormal(
        mean=math.log(max(backward_phase_mean_rad, 1e-6)),
        sigma=sigma_log_B, size=T).astype(np.float32)

    delta_theta = np.zeros(T, dtype=np.float32)
    m_left = (cat == 1) & onset
    m_right = (cat == 2) & onset
    m_fwd = (cat == 3) & onset
    m_back = (cat == 4) & onset
    delta_theta[m_left] = +mag_LR[m_left]
    delta_theta[m_right] = -mag_LR[m_right]
    bw_sign = np.where(rng.uniform(size=T) < 0.5, +1.0, -1.0)
    delta_theta[m_back] = (bw_sign[m_back] * mag_B[m_back]).astype(np.float32)

    omega_rad = np.zeros(T, dtype=np.float32)
    for k in range(L):
        omega_rad[k:] += delta_theta[:T - k] / (L * dt)
    omega = np.rad2deg(omega_rad).astype(np.float32)

    # Per-frame swim label (onset only): 1=L, 2=R, 3=F, 4=B, 0=none
    swim_label = np.where(onset, cat, 0).astype(np.int8)

    # Combined traces: L/R turn rate (positive=L, negative=R) and F/B impulses
    turn_lr = np.zeros(T, dtype=np.float32)
    # Display convention for the L/R panel: right swims point UP (red),
    # left swims point DOWN (blue). Independent of the actual angular
    # velocity delta_theta that drives the network, which keeps the
    # math sign convention (CCW positive).
    turn_lr[m_left] = -np.rad2deg(mag_LR[m_left])
    turn_lr[m_right] = +np.rad2deg(mag_LR[m_right])
    swim_fb = np.zeros(T, dtype=np.float32)
    swim_fb[m_fwd] = +1.0
    swim_fb[m_back] = -1.0

    theta0 = rng.uniform(0, 2 * math.pi)
    theta_hd = theta0 + np.cumsum(np.deg2rad(omega)) * dt
    theta_hd[0] = theta0

    u = np.zeros((1, T, 3), dtype=np.float32)
    u[0, :, 0] = omega
    u[0, 0, 1] = math.cos(theta0)
    u[0, 0, 2] = math.sin(theta0)
    y = np.stack([np.cos(theta_hd), np.sin(theta_hd)],
                 axis=-1).astype(np.float32)[None]
    batch = TaskTrials(
        task_family="swim_integration",
        n_input=3, n_output=2, dt=float(dt),
        stimulus=torch.from_numpy(u).to(device),
        target=torch.from_numpy(y).to(device),
        theta_hd=torch.from_numpy(theta_hd[None].astype(np.float32)).to(device),
        is_stop=torch.from_numpy((omega == 0).astype(np.float32)[None]).to(device),
        omega=torch.from_numpy(omega[None]).to(device),
    )
    return batch, turn_lr, swim_fb


def _build_single_impulse_batch(n_steps, dt, device,
                                 direction="L",
                                 magnitude_rad=0.785,
                                 t_event_s=0.0,
                                 interval_s=2.0,
                                 theta0=0.0,
                                 swim_duration_s=0.3):
    """Periodic single-direction swim stimulus (B=1). Fires typed swim
    impulses every ``interval_s`` seconds, starting at ``t_event_s``,
    each with phase magnitude ``magnitude_rad`` (default π/4 = Petrucco
    median). Pass ``interval_s <= 0`` for a single-shot impulse at
    ``t_event_s`` only (initialisation / persistence test).

    ``direction`` ∈ {"L", "R"} sets the sign of delta_theta and the
    L/R display trace channel; the F/B channel is left at zero.
    """
    T = int(n_steps)
    L = max(1, int(round(swim_duration_s / dt)))
    k0 = int(round(max(0.0, t_event_s) / dt))
    k0 = min(k0, T - 1)
    sign = +1.0 if direction.upper() == "L" else -1.0
    mag = float(abs(magnitude_rad))

    if interval_s and interval_s > 0:
        step_k = max(1, int(round(interval_s / dt)))
        event_ks = np.arange(k0, T, step_k, dtype=np.int64)
    else:
        event_ks = np.array([k0], dtype=np.int64)

    delta_theta = np.zeros(T, dtype=np.float32)
    delta_theta[event_ks] = sign * mag

    omega_rad = np.zeros(T, dtype=np.float32)
    for k in range(L):
        omega_rad[k:] += delta_theta[:T - k] / (L * dt)
    omega = np.rad2deg(omega_rad).astype(np.float32)

    # Display traces — L/R panel: left swims point DOWN, right UP
    # (matches _build_swim_batch's display convention).
    turn_lr = np.zeros(T, dtype=np.float32)
    if direction.upper() == "L":
        turn_lr[event_ks] = -np.rad2deg(mag)
    else:
        turn_lr[event_ks] = +np.rad2deg(mag)
    swim_fb = np.zeros(T, dtype=np.float32)

    theta0_f = float(theta0)
    theta_hd = theta0_f + np.cumsum(np.deg2rad(omega)) * dt
    theta_hd[0] = theta0_f

    u = np.zeros((1, T, 3), dtype=np.float32)
    u[0, :, 0] = omega
    u[0, 0, 1] = math.cos(theta0_f)
    u[0, 0, 2] = math.sin(theta0_f)
    y = np.stack([np.cos(theta_hd), np.sin(theta_hd)],
                 axis=-1).astype(np.float32)[None]
    batch = TaskTrials(
        task_family="swim_integration",
        n_input=3, n_output=2, dt=float(dt),
        stimulus=torch.from_numpy(u).to(device),
        target=torch.from_numpy(y).to(device),
        theta_hd=torch.from_numpy(theta_hd[None].astype(np.float32)).to(device),
        is_stop=torch.from_numpy((omega == 0).astype(np.float32)[None]).to(device),
        omega=torch.from_numpy(omega[None]).to(device),
    )
    return batch, turn_lr, swim_fb


def _run_single_impulse(net, n_steps, dt, device, **kw):
    batch, turn_lr, swim_fb = _build_single_impulse_batch(
        n_steps, dt, device, **kw)
    theta = batch.theta_hd[0].cpu().numpy()
    omega = batch.omega[0].cpu().numpy()
    with torch.no_grad():
        y_hat, h = net(batch.stimulus)
    decoded_hd = np.arctan2(y_hat[0, :, 1].cpu().numpy(),
                            y_hat[0, :, 0].cpu().numpy())
    return h[0].cpu().numpy(), theta, omega, decoded_hd, turn_lr, swim_fb


def _run_swim(net, n_steps, dt, device, seed=0, **swim_kw):
    batch, turn_lr, swim_fb = _build_swim_batch(
        n_steps, dt, device, seed=seed, **swim_kw)
    theta = batch.theta_hd[0].cpu().numpy()
    omega = batch.omega[0].cpu().numpy()
    with torch.no_grad():
        y_hat, h = net(batch.stimulus)
    decoded_hd = np.arctan2(y_hat[0, :, 1].cpu().numpy(),
                            y_hat[0, :, 0].cpu().numpy())
    return h[0].cpu().numpy(), theta, omega, decoded_hd, turn_lr, swim_fb


# ── model-index → skeleton mapping ───────────────────────────────────────

def _model_index_to_bodyid(connconstr_datapath: str) -> np.ndarray:
    """Replay load_zebrafish_hd_connectome's neuron ordering and return
    the bodyId array in model order."""
    cx = load_zebrafish_hd_connectome(connconstr_datapath)
    return cx["bodyId"], cx["category"]


def _load_skeletons_in_model_order(anatomy_dir: str, model_bodyids: np.ndarray,
                                   model_categories: np.ndarray,
                                   downsample: int = 10):
    """Load SWC skeletons for each model neuron that has a skeleton.

    Returns:
        neurons: list of navis TreeNeurons (one per model neuron with a
            skeleton; None for missing ones)
        categories: list of category strings parallel to neurons
        has_skel: (N_model,) bool mask — True where a skeleton was found
    """
    import navis

    index = pd.read_csv(os.path.join(anatomy_dir, "index.csv"))
    bid_to_swc = {int(r.bodyId): r.swc for _, r in index.iterrows()}

    neurons = []
    categories = []
    has_skel = np.zeros(len(model_bodyids), dtype=bool)

    for i, (bid, cat) in enumerate(zip(model_bodyids, model_categories)):
        swc_rel = bid_to_swc.get(int(bid))
        if swc_rel is None:
            neurons.append(None)
            categories.append(str(cat))
            continue
        path = os.path.join(anatomy_dir, swc_rel)
        if not os.path.isfile(path):
            neurons.append(None)
            categories.append(str(cat))
            continue
        n = navis.read_swc(path)
        if downsample and downsample > 1:
            n = navis.downsample_neuron(n, downsampling_factor=downsample,
                                        preserve_nodes=None)
        neurons.append(n)
        categories.append(str(cat))
        has_skel[i] = True

    return neurons, categories, has_skel


# ── segment extraction ───────────────────────────────────────────────────

def _extract_per_neuron_segments(neurons, has_skel):
    """Return:
      seg_arrays: list of (E_i, 2, 3) arrays per neuron (empty for missing)
      seg_owner:  flat (E_total,) int array, model-neuron index per segment
      all_segs:   stacked (E_total, 2, 3) array
    """
    seg_arrays = []
    for i, n in enumerate(neurons):
        if n is None:
            seg_arrays.append(np.zeros((0, 2, 3), dtype=np.float32))
            continue
        nodes = n.nodes
        child = nodes[nodes.parent_id != -1]
        if len(child) == 0:
            seg_arrays.append(np.zeros((0, 2, 3), dtype=np.float32))
            continue
        parent_xyz = nodes.set_index("node_id").loc[
            child.parent_id.values, ["x", "y", "z"]
        ].values
        child_xyz = child[["x", "y", "z"]].values
        seg_arrays.append(np.stack([parent_xyz, child_xyz], axis=1)
                          .astype(np.float32))
    counts = np.array([len(s) for s in seg_arrays])
    seg_owner = np.repeat(np.arange(len(neurons)), counts)
    all_segs = (np.concatenate(seg_arrays, axis=0) if seg_arrays
                else np.zeros((0, 2, 3), dtype=np.float32))
    return seg_arrays, seg_owner, all_segs


def _extract_soma_positions(neurons):
    """Use the largest-radius node as soma proxy. Returns:
      soma_xyz: (N, 3) float (NaN for missing neurons)
      soma_r:   (N,)   float
    """
    soma_xyz = np.full((len(neurons), 3), np.nan, dtype=np.float32)
    soma_r = np.zeros(len(neurons), dtype=np.float32)
    for i, n in enumerate(neurons):
        if n is None:
            continue
        nodes = n.nodes
        idx = int(nodes.radius.idxmax())
        row = nodes.loc[idx]
        soma_xyz[i] = [float(row.x), float(row.y), float(row.z)]
        soma_r[i] = float(row.radius)
    return soma_xyz, soma_r


# ── rendering ────────────────────────────────────────────────────────────

def _paint_panel(ax, segs2d, seg_owner, rates_t, mesh_segs2d,
                 soma_2d, soma_valid,
                 xlim, ylim, bg, green, alpha_max,
                 base_color, lw_base, lw_top, soma_size,
                 show_base=True):
    """Draw one view panel (base skeleton + green overlay + soma dots).

    ``show_base=False`` skips the dark-grey baseline skeleton and dim
    soma markers (kept only for the active green overlay), so that
    pixel-based ROI sampling on this panel isn't biased by the
    baseline ink."""
    ax.set_facecolor(bg)

    if mesh_segs2d is not None and len(mesh_segs2d):
        ax.add_collection(LineCollection(
            mesh_segs2d, colors=("0.85" if bg == "black" else "0.45",),
            linewidths=0.25, alpha=0.12,
        ))

    if show_base:
        ax.add_collection(LineCollection(
            segs2d, colors=[base_color], linewidths=lw_base, alpha=0.5,
        ))

    alpha = rates_t[seg_owner] * alpha_max
    keep = alpha > 0.02
    if keep.any():
        rgba = np.tile(np.array([*green, 1.0], dtype=np.float32),
                       (int(keep.sum()), 1))
        rgba[:, 3] = alpha[keep]
        ax.add_collection(LineCollection(
            segs2d[keep], colors=rgba, linewidths=lw_top,
        ))

    if soma_2d is not None and soma_valid is not None:
        sv = soma_valid
        if show_base:
            ax.scatter(soma_2d[sv, 0], soma_2d[sv, 1],
                       s=soma_size * 0.5, c=[base_color], edgecolors="none",
                       alpha=0.7, zorder=0)
        keep_n = (rates_t > 0.02) & sv
        if keep_n.any():
            rgba_s = np.tile(np.array([*green, 1.0], dtype=np.float32),
                              (int(keep_n.sum()), 1))
            rgba_s[:, 3] = rates_t[keep_n]
            ax.scatter(soma_2d[keep_n, 0], soma_2d[keep_n, 1],
                       s=soma_size, c=rgba_s, edgecolors="none",
                       zorder=1)

    if xlim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()


# Dorsal (top-down) view of a larval zebrafish, nose to the LEFT at
# theta=0. Outline: round head, pectoral fins, narrowing trunk, spread
# caudal fin. Same head-on-the-left convention as the dorsal anatomy
# panel; with the 180-deg trajectory flip below, motion and orientation
# stay consistent.
_FISH_SILHOUETTE_X = np.array([
    -1.20, -0.95, -0.50, -0.30, -0.10,
     0.30,  0.80,  1.30,  1.40,  1.30,
     0.80,  0.30, -0.10, -0.30, -0.50, -0.95,
])
_FISH_SILHOUETTE_Y = np.array([
     0.00,  0.30,  0.35,  0.55,  0.30,
     0.20,  0.10,  0.40,  0.00, -0.40,
    -0.10, -0.20, -0.30, -0.55, -0.35, -0.30,
])
# Two eyes on either side of the head (dorsal view).
_FISH_EYES_X = np.array([-0.85, -0.85])
_FISH_EYES_Y = np.array([ 0.20, -0.20])

# Body-coord convention here: nose is at NEGATIVE x (head-on-the-left),
# tail is at POSITIVE x. Vertices behind the waist (x > waist) get a
# sinusoidal y displacement scaled by how far back along the body they
# sit, so the caudal fin swishes side-to-side over time.
_FISH_WAIST_X = 0.30
_FISH_TAIL_TIP_X = 1.40


def _fish_tail_swish(frame_t, phase_per_frame=0.18, amp=0.25):
    """Per-vertex y offsets for the fish silhouette polygon.

    Vertices forward of the waist (x < waist) are unaffected; tail
    vertices get a sin(phase) displacement scaled by `((x - waist) /
    tail_length) ** 1.5` so the caudal-fin tip swings the most."""
    span = _FISH_TAIL_TIP_X - _FISH_WAIST_X
    t_norm = np.clip((_FISH_SILHOUETTE_X - _FISH_WAIST_X) / span, 0.0, 1.0)
    if frame_t is None:
        return np.zeros_like(_FISH_SILHOUETTE_Y)
    phase = phase_per_frame * float(frame_t)
    return amp * (t_norm ** 1.5) * math.sin(phase)


def _draw_fish_icon(ax, theta_rad, body_color="white", eye_color="black",
                    frame_t=None):
    """Draw a small fish silhouette in ``ax``, pointing at ``theta_rad``.

    Convention: theta=0 -> fish nose to the right (east), theta=pi/2 ->
    nose up, matching the HD ground-truth convention used elsewhere in
    this script. Caller must give an axes with equal aspect, axis off.
    """
    ax.clear()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    c, s = math.cos(float(theta_rad)), math.sin(float(theta_rad))
    sx = _FISH_SILHOUETTE_X
    sy = _FISH_SILHOUETTE_Y + _fish_tail_swish(frame_t)
    fx = c * sx - s * sy
    fy = s * sx + c * sy
    ax.fill(fx, fy, color=body_color, edgecolor="none", linewidth=0,
            zorder=2)
    # Two eyes (dorsal view).
    ex = c * _FISH_EYES_X - s * _FISH_EYES_Y
    ey = s * _FISH_EYES_X + c * _FISH_EYES_Y
    ax.plot(ex, ey, linestyle="", marker="o", markersize=3.0,
            color=eye_color, markeredgewidth=0, zorder=3)


def _fish_twitch_body(frame_t, dt, turn_lr, swim_fb,
                       amp_lateral=0.12, amp_axial=0.60, decay_s=0.20):
    """Body-frame twitch (dx_body, dy_body) decaying after a recent
    trace-strip tick. Conventions: +y_body = fish's left flank,
    -x_body = nose-ward (forward), +x_body = tail-ward (backward).

    Sign mapping matches the trace-strip colors:
      turn_lr > 0  (right swim, red tick)   → nudge to -y (fish-right)
      turn_lr < 0  (left swim, blue tick)   → nudge to +y (fish-left)
      swim_fb > 0  (forward, grey tick)     → nudge to -x (head-ward)
      swim_fb < 0  (backward, orange tick)  → nudge to +x (tail-ward)
    """
    if frame_t is None:
        return 0.0, 0.0
    ft = int(frame_t)
    if ft < 0:
        return 0.0, 0.0
    tau = max(decay_s / 3.0, 1e-6)
    n_lookback = max(1, int(round(decay_s / max(dt, 1e-6))))
    lo = max(0, ft - n_lookback)
    dx = dy = 0.0
    for i in range(lo, ft + 1):
        age_s = (ft - i) * dt
        decay = math.exp(-age_s / tau)
        if turn_lr is not None and i < len(turn_lr) and turn_lr[i] != 0.0:
            sign = -1.0 if turn_lr[i] > 0 else +1.0
            dy += sign * amp_lateral * decay
        if swim_fb is not None and i < len(swim_fb) and swim_fb[i] != 0.0:
            sign = -1.0 if swim_fb[i] > 0 else +1.0
            dx += sign * amp_axial * decay
    return dx, dy


def _draw_fish_with_trail(ax, theta_rad, swim_x, swim_y, frame_t,
                          body_color="white", eye_color="black",
                          bg="black", twitch_body=(0.0, 0.0)):
    """Draw the fish at swim_x[frame_t], swim_y[frame_t] with the
    trajectory it has swum so far as a faint trail.

    Panel limits are fixed to the full trajectory so the fish moves
    smoothly within the same frame across frames. Fish size is set to
    a constant fraction of the panel span so it stays visible whatever
    the path length."""
    ax.clear()
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)

    x_lo, x_hi = float(swim_x.min()), float(swim_x.max())
    y_lo, y_hi = float(swim_y.min()), float(swim_y.max())
    span = max(x_hi - x_lo, y_hi - y_lo, 1.0)
    pad = 0.12 * span
    span = span + 2 * pad
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)

    # Trail up to current frame
    trail_color = ((1.0, 1.0, 1.0, 0.35) if bg == "black"
                   else (0.0, 0.0, 0.0, 0.35))
    n = int(frame_t) + 1
    if n > 1:
        ax.plot(swim_x[:n], swim_y[:n], color=trail_color, lw=0.7,
                zorder=1)

    # Fish at the current position (with optional swim-tick twitch in
    # body coords, rotated by theta and scaled into panel units).
    fx_c = float(swim_x[int(frame_t)])
    fy_c = float(swim_y[int(frame_t)])
    fish_scale = span * 0.06
    c, s = math.cos(float(theta_rad)), math.sin(float(theta_rad))
    tbx, tby = float(twitch_body[0]), float(twitch_body[1])
    fx_c += (c * tbx - s * tby) * fish_scale
    fy_c += (s * tbx + c * tby) * fish_scale
    sx = _FISH_SILHOUETTE_X
    sy = _FISH_SILHOUETTE_Y + _fish_tail_swish(frame_t)
    fx = (c * sx - s * sy) * fish_scale + fx_c
    fy = (s * sx + c * sy) * fish_scale + fy_c
    ax.fill(fx, fy, color=body_color, edgecolor="none", linewidth=0,
            zorder=2)
    ex = (c * _FISH_EYES_X - s * _FISH_EYES_Y) * fish_scale + fx_c
    ey = (s * _FISH_EYES_X + c * _FISH_EYES_Y) * fish_scale + fy_c
    ax.plot(ex, ey, linestyle="", marker="o", markersize=2.5,
            color=eye_color, markeredgewidth=0, zorder=3)


def _style_trace_ax(ax, bg, ylabel, fontsize=11, bottom_labels=False):
    """Minimal styling shared by all trace sub-axes."""
    txt_color = "white" if bg == "black" else "black"
    ax.set_facecolor(bg)
    ax.set_ylabel(ylabel, color=txt_color, fontsize=fontsize, labelpad=2)
    ax.tick_params(axis="y", colors=txt_color, labelsize=10, length=3)
    ax.tick_params(axis="x", colors=txt_color, labelsize=10, length=3,
                   labelbottom=bottom_labels)
    ax.spines[:].set_visible(False)


def _paint_traces(trace_axes, t_sec, frame_t, trace_data, bg="black",
                  frame_label_idx=None, hd_deg=None):
    """Grow-then-scroll trace strip.

    Phase 1 (t_now < scroll_window): x-axis spans [0, scroll_window],
        trace grows rightward from t=0.
    Phase 2 (t_now >= scroll_window): x-axis slides so t_now is at the
        right edge; older data scrolls off the left.

    trace_axes: list of Axes for [ω, HD] or [ω, HD, L/R, F/B].
    """
    txt_color = "white" if bg == "black" else "black"
    dim = "0.35" if bg == "black" else "0.70"

    omega_full = trace_data["omega"]
    theta_full = trace_data["theta"]
    decoded_hd_full = trace_data["decoded_hd"]
    turn_lr = trace_data.get("turn_lr")
    swim_fb = trace_data.get("swim_fb")
    has_swim = turn_lr is not None
    win = trace_data.get("scroll_window", 10.0)

    n_now = frame_t + 1
    t_now_val = t_sec[frame_t]

    # Visible x-range: grow until scroll_window, then slide
    if t_now_val < win:
        x_lo, x_hi = 0.0, win
    else:
        x_lo, x_hi = t_now_val - win, t_now_val

    # Indices within the visible window (for bar plots)
    vis_mask = (t_sec[:n_now] >= x_lo) & (t_sec[:n_now] <= x_hi)

    ax_idx = 0

    # ── ω panel ──────────────────────────────────────────────────────
    ax = trace_axes[ax_idx]; ax_idx += 1
    ax.plot(t_sec[:n_now], omega_full[:n_now],
            color=(0.0, 0.85, 0.4), lw=1.4)
    ax.axhline(0, color=dim, lw=0.3, alpha=0.4)
    ax.set_xlim(x_lo, x_hi)
    o_abs = max(np.abs(omega_full).max(), 1.0)
    ax.set_ylim(-o_abs * 1.15, o_abs * 1.15)
    _style_trace_ax(ax, bg, "ω (°/s)")
    if frame_label_idx is not None:
        label = f"t = {frame_label_idx:04d}"
        if hd_deg is not None:
            label += f"   HD = {hd_deg:+.0f}°"
        ax.text(0.98, 0.95, label, color=txt_color, fontsize=11,
                family="monospace", ha="right", va="top",
                transform=ax.transAxes)

    # ── HD panel ─────────────────────────────────────────────────────
    ax = trace_axes[ax_idx]; ax_idx += 1

    def _wrap_hd_deg(a_rad):
        # Wrap to [-180, 180] and insert NaN at ±180 jumps so the line
        # does not draw a vertical jump across the panel.
        w = (((np.rad2deg(a_rad) + 180.0) % 360.0) - 180.0).astype(np.float32)
        if w.size > 1:
            jump = np.abs(np.diff(w)) > 180.0
            w[1:][jump] = np.nan
        return w

    target_deg = _wrap_hd_deg(theta_full)
    decoded_deg = _wrap_hd_deg(decoded_hd_full)
    ax.plot(t_sec[:n_now], target_deg[:n_now], color=(0.0, 0.85, 0.4),
            lw=1.4)
    ax.plot(t_sec[:n_now], decoded_deg[:n_now],
            color="white" if bg == "black" else "black",
            lw=1.4)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-180, 180)
    ax.set_yticks([-180, 0, 180])
    is_last = not has_swim
    _style_trace_ax(ax, bg, "HD (°)", bottom_labels=is_last)
    if is_last:
        ax.set_xlabel("time (s)", color=txt_color, fontsize=11, labelpad=2)

    if not has_swim:
        return

    # ── L/R turn panel ───────────────────────────────────────────────
    ax = trace_axes[ax_idx]; ax_idx += 1
    bar_w = (x_hi - x_lo) * 0.008
    vis_idx_lr = np.nonzero((turn_lr[:n_now] != 0) & vis_mask)[0]
    if vis_idx_lr.size:
        vals = turn_lr[vis_idx_lr]
        # Red = LEFT swim (positive turn_lr), blue = RIGHT swim (negative).
        colors_lr = np.where(vals > 0, "#ff4444", "#4488ff")
        ax.bar(t_sec[vis_idx_lr], vals, width=bar_w,
               color=colors_lr, alpha=0.9, linewidth=0)
    ax.axhline(0, color=dim, lw=0.3, alpha=0.4)
    ax.set_xlim(x_lo, x_hi)
    lr_abs = max(np.abs(turn_lr).max(), 1.0)
    ax.set_ylim(-lr_abs * 1.2, lr_abs * 1.2)
    _style_trace_ax(ax, bg, "L / R (°)")

    # ── F/B swim panel ───────────────────────────────────────────────
    ax = trace_axes[ax_idx]; ax_idx += 1
    vis_idx_fb = np.nonzero((swim_fb[:n_now] != 0) & vis_mask)[0]
    if vis_idx_fb.size:
        vals_fb = swim_fb[vis_idx_fb]
        colors_fb = np.where(vals_fb > 0, "#888888", "#ff9922")
        ax.bar(t_sec[vis_idx_fb], vals_fb, width=bar_w,
               color=colors_fb, alpha=0.9, linewidth=0)
    ax.axhline(0, color=dim, lw=0.3, alpha=0.4)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-1.5, 1.5)
    _style_trace_ax(ax, bg, "F / B", bottom_labels=True)
    ax.set_xlabel("time (s)", color=txt_color, fontsize=11, labelpad=4)


def _render_frame(out_path, view_data, seg_owner, rates_t,
                  bg="black", lw_base=0.18, lw_top=0.45,
                  base_color=(0.25, 0.25, 0.25), green=(0.0, 1.0, 0.3),
                  alpha_max=1.0,
                  frame_idx=None, hd_deg=None,
                  fig_ref=None, axes_ref=None,
                  soma_valid=None, soma_size=18.0,
                  trace_data=None,
                  roi_overlay=None):
    """Render three rows: frontal anatomy, trace strip, dorsal anatomy.

    view_data: list of dicts (frontal, dorsal).
    trace_data: dict with keys omega, theta, decoded_hd, dt, t_sec
                (None = no trace strip, just the two anatomy panels).
    """
    has_traces = trace_data is not None
    has_swim = (has_traces and trace_data.get("turn_lr") is not None)
    n_trace_rows = (4 if has_swim else 2) if has_traces else 0
    has_kinograph = bool(roi_overlay is not None
                          and roi_overlay.get("kinograph_enabled", False))

    if fig_ref is None:
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
        if has_traces:
            # Side-by-side layout (landscape): anatomy (frontal + dorsal,
            # stacked) on the left, trace strip + kinograph on the right.
            # Replaces the previous stack-everything-vertically layout that
            # made the figure too tall to view at a glance.
            n_right_rows = n_trace_rows + (1 if has_kinograph else 0)
            # kinograph row is ~2× a trace row (4 ROIs stay legible)
            right_h = [1] * n_trace_rows + ([2] if has_kinograph else [])
            total_w, total_h = 14.0, 9.0
            fig = plt.figure(figsize=(total_w, total_h), facecolor=bg)
            outer = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1.0],
                             top=0.985, bottom=0.06,
                             left=0.020, right=0.985, wspace=0.06)
            left_gs = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[0],
                height_ratios=[1, 1], hspace=0.02)
            right_gs = GridSpecFromSubplotSpec(
                n_right_rows, 1, subplot_spec=outer[1],
                height_ratios=right_h, hspace=0.0)
            axes = [fig.add_subplot(left_gs[0]),   # frontal
                    fig.add_subplot(left_gs[1])]   # dorsal
            for i in range(n_trace_rows):
                axes.append(fig.add_subplot(right_gs[i]))
            if has_kinograph:
                fig._kin_ax = fig.add_subplot(right_gs[n_right_rows - 1])
                fig._kin_ax.set_facecolor(bg)
        else:
            # No trace strip: two anatomy panels stacked, kinograph (if
            # any) at the bottom — original layout suffices.
            kino_h_in = 1.75 if has_kinograph else 0.0
            base_total_h = 11.0
            total_h = base_total_h + kino_h_in
            fig = plt.figure(figsize=(10.0, total_h), facecolor=bg)
            gs_bottom = 0.005 + (kino_h_in / total_h)
            gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1],
                         hspace=0.03,
                         top=0.995, bottom=gs_bottom, left=0.005, right=0.995)
            axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
            if has_kinograph:
                kin_pad_bottom = 0.50 / total_h
                kin_pad_top = 0.20 / total_h
                kin_h_frac = (kino_h_in / total_h) - kin_pad_bottom - kin_pad_top
                fig._kin_ax = fig.add_axes(
                    [0.07, kin_pad_bottom, 0.90, kin_h_frac])
                fig._kin_ax.set_facecolor(bg)
        # Fish-orientation overlay with motion trail: anchored to the
        # upper-right corner of the dorsal anatomy panel so the fish
        # heading and the brain top-down view share a frame of reference.
        # Stashed on the figure so it survives the per-frame ax.clear()
        # loop.
        dorsal_ax = axes[1] if len(axes) > 1 else axes[0]
        dorsal_bbox = dorsal_ax.get_position()
        fig_w_inch, fig_h_inch = fig.get_size_inches()
        fish_size_in = 1.70  # inches square (room for trail)
        fish_w = fish_size_in / fig_w_inch
        fish_h = fish_size_in / fig_h_inch
        fish_x = dorsal_bbox.x1 - fish_w - 0.005
        fish_y = dorsal_bbox.y1 - fish_h - 0.005
        fish_ax = fig.add_axes([fish_x, fish_y, fish_w, fish_h])
        fish_ax.set_aspect("equal")
        fish_ax.set_axis_off()
        fish_ax.patch.set_alpha(0.0)
        fig._fish_ax = fish_ax
    else:
        fig, axes = fig_ref, axes_ref
        for a in axes:
            a.clear()
        for txt in list(fig.texts):
            txt.remove()

    txt_color = "white" if bg == "black" else "black"

    # Anatomy panels: first two axes (frontal, dorsal). The dorsal panel
    # (i=1) drops the dark-grey baseline skeleton so ROI pixel-sampling
    # isn't biased by the static ink.
    for i, vd in enumerate(view_data):
        ax = axes[i]
        _paint_panel(ax, vd["segs2d"], seg_owner, rates_t,
                     vd["mesh_segs2d"], vd["soma_2d"], soma_valid,
                     vd["xlim"], vd["ylim"], bg, green, alpha_max,
                     base_color, lw_base, lw_top, soma_size,
                     show_base=(i != 1))
        ax.text(0.02, 0.97, vd["title"], color=txt_color, fontsize=9,
                family="monospace", ha="left", va="top",
                transform=ax.transAxes)

    # Trace strip (axes after the two anatomy panels)
    if has_traces and frame_idx is not None:
        trace_axes = axes[2:]
        _paint_traces(trace_axes, trace_data["t_sec"], frame_idx,
                      trace_data, bg=bg,
                      frame_label_idx=frame_idx, hd_deg=hd_deg)

    # Fish silhouette moving along the integrated swim trajectory, with
    # orientation matching the current ground-truth HD direction.
    if hd_deg is not None and getattr(fig, "_fish_ax", None) is not None:
        swim_x = trace_data.get("swim_x") if trace_data else None
        swim_y = trace_data.get("swim_y") if trace_data else None
        if (swim_x is not None and swim_y is not None
                and frame_idx is not None
                and 0 <= int(frame_idx) < len(swim_x)):
            twitch_xy = _fish_twitch_body(
                int(frame_idx), float(trace_data.get("dt", 0.05)),
                trace_data.get("turn_lr"), trace_data.get("swim_fb"),
            )
            _draw_fish_with_trail(
                fig._fish_ax, math.radians(float(hd_deg)),
                swim_x, swim_y, int(frame_idx),
                body_color=txt_color, eye_color=(0.30, 0.30, 0.30),
                bg=bg, twitch_body=twitch_xy,
            )
        else:
            _draw_fish_icon(
                fig._fish_ax, math.radians(float(hd_deg)),
                body_color=txt_color, eye_color=(0.30, 0.30, 0.30),
                frame_t=frame_idx,
            )

    # ROI sampling + kinograph paint. Sample on the clean canvas before
    # the overlay rectangles are drawn so the green-channel mean isn't
    # skewed by the box borders.
    if has_kinograph and frame_idx is not None:
        try:
            intens = _extract_zhd_roi_intensities(fig, channel=1)
        except Exception as e:
            print(f"      [kinograph] intensity extraction failed: {e}")
            intens = None
        if intens is not None:
            buf = roi_overlay.setdefault("intens_buf", [])
            t_buf = roi_overlay.setdefault("intens_t_buf", [])
            buf.append(intens)
            t_buf.append(float(frame_idx) * roi_overlay.get("dt", 0.01))
            _paint_zhd_roi_kinograph(
                fig._kin_ax, np.asarray(buf), np.asarray(t_buf),
                scroll_window=roi_overlay.get("scroll_window", 10.0),
                bg=bg,
            )

    # ROI rectangles. Gate on first_only (drosophila convention): when
    # set, the overlay is a frame-0 anatomical reference and the rest of
    # the animation shows only neural activity dynamics.
    if roi_overlay is not None:
        show = (not roi_overlay.get("first_only", False)
                or int(frame_idx or 0) == 0)
        if show:
            _draw_zhd_roi_overlay(fig)

    fig.savefig(out_path, dpi=300, facecolor=bg)
    return fig, axes


MONTAGE_TYPES = ["IPNd", "IPNds", "RIPN", "pt-IPN", "all"]


def _render_montage_frame(out_path, view_data, seg_owner, types_str, rates_t,
                          soma_valid=None,
                          frame_idx=None, hd_deg=None,
                          bg="black",
                          green=(0.0, 1.0, 0.3),
                          alpha_max=1.0,
                          base_color=(0.22, 0.22, 0.22),
                          lw_base=0.10, lw_top=0.40, soma_size=10.0,
                          fig_ref=None, axes_ref=None,
                          trace_data=None):
    """Two-view montage with trace strip between the two anatomy rows.

    Layout (GridSpec rows):
        row 0: frontal view  (ncols anatomy panels)
        rows 1..K: trace sub-plots (ω, HD, [L/R, F/B]) spanning full width
        row K+1: dorsal view (ncols anatomy panels)

    axes_ref structure (dict):
        "frontal": list of ncols Axes
        "dorsal":  list of ncols Axes
        "traces":  list of n_trace Axes
    """
    from matplotlib.gridspec import GridSpec

    types_arr = np.asarray(types_str)
    ncols = len(MONTAGE_TYPES)
    has_traces = trace_data is not None
    has_swim = has_traces and trace_data.get("turn_lr") is not None
    n_trace = (4 if has_swim else 2) if has_traces else 0

    if fig_ref is None:
        trace_h = [1] * n_trace
        ratios = [5, 5] + trace_h if has_traces else [5, 5]
        n_gs_rows = len(ratios)
        total_h = 9.0 + (n_trace * 1.0 if has_traces else 0.0)
        fig = plt.figure(figsize=(ncols * 3.2, total_h), facecolor=bg)
        gs = GridSpec(n_gs_rows, ncols, figure=fig,
                      height_ratios=ratios, hspace=0.04, wspace=0.02)
        frontal_axes = [fig.add_subplot(gs[0, c]) for c in range(ncols)]
        dorsal_axes = [fig.add_subplot(gs[1, c]) for c in range(ncols)]
        trace_axes = []
        for tr in range(n_trace):
            ax_tr = fig.add_subplot(gs[2 + tr, :])
            trace_axes.append(ax_tr)
        axes_dict = {"frontal": frontal_axes, "dorsal": dorsal_axes,
                     "traces": trace_axes}
    else:
        fig = fig_ref
        axes_dict = axes_ref
        for a in axes_dict["frontal"] + axes_dict["dorsal"]:
            a.clear()
        for a in axes_dict["traces"]:
            a.clear()
        for txt in list(fig.texts):
            txt.remove()

    txt_color = "white" if bg == "black" else "black"
    mesh_color = "0.85" if bg == "black" else "0.45"

    view_keys = ["frontal", "dorsal"]
    for vi, vd in enumerate(view_data):
        segs2d = vd["segs2d"]
        mesh_segs2d = vd["mesh_segs2d"]
        soma_2d = vd["soma_2d"]
        xlim, ylim = vd["xlim"], vd["ylim"]
        view_title = vd["title"]
        panel_axes = axes_dict[view_keys[vi]]

        backdrop_color = "0.45" if bg == "black" else "0.55"
        for col, ct in enumerate(MONTAGE_TYPES):
            ax = panel_axes[col]
            ax.set_facecolor(bg)

            if mesh_segs2d is not None and len(mesh_segs2d):
                ax.add_collection(LineCollection(
                    mesh_segs2d, colors=(mesh_color,),
                    linewidths=0.2, alpha=0.10,
                ))

            # Full-skeleton backdrop: shows where the per-type neurons sit
            # within the broader circuit. Skip the "all" panel since it
            # already draws every segment at full intensity.
            if ct != "all":
                ax.add_collection(LineCollection(
                    segs2d, colors=(backdrop_color,),
                    linewidths=lw_base * 0.6, alpha=0.20,
                ))

            mask_n = (np.ones(len(types_arr), dtype=bool) if ct == "all"
                      else (types_arr == ct))
            mask = mask_n[seg_owner]

            if mask.any():
                ax.add_collection(LineCollection(
                    segs2d[mask], colors=[base_color],
                    linewidths=lw_base, alpha=0.55,
                ))

            alpha = rates_t[seg_owner] * mask * alpha_max
            keep = alpha > 0.02
            if keep.any():
                rgba = np.tile(np.array([*green, 1.0], dtype=np.float32),
                               (int(keep.sum()), 1))
                rgba[:, 3] = alpha[keep]
                ax.add_collection(LineCollection(
                    segs2d[keep], colors=rgba, linewidths=lw_top,
                ))

            if soma_2d is not None and soma_valid is not None:
                sv_ct = mask_n & soma_valid
                ax.scatter(soma_2d[sv_ct, 0], soma_2d[sv_ct, 1],
                           s=soma_size * 0.5, c=[base_color],
                           edgecolors="none", alpha=0.7, zorder=0)
                lit_n = (rates_t > 0.02) & sv_ct
                if lit_n.any():
                    rgba_s = np.tile(
                        np.array([*green, 1.0], dtype=np.float32),
                        (int(lit_n.sum()), 1))
                    rgba_s[:, 3] = rates_t[lit_n]
                    ax.scatter(soma_2d[lit_n, 0], soma_2d[lit_n, 1],
                               s=soma_size, c=rgba_s,
                               edgecolors="none", zorder=1)

            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_aspect("equal"); ax.set_axis_off()

            n_count = int(mask_n.sum())
            label_ct = (f"all ({n_count})" if ct == "all"
                        else f"{ct}  (n={n_count})")
            if col == 0:
                label_ct = f"{view_title}\n{label_ct}"
            ax.text(0.02, 0.97, label_ct, color=txt_color, fontsize=8,
                    family="monospace", ha="left", va="top",
                    transform=ax.transAxes)

    # Trace strip
    if has_traces and frame_idx is not None:
        _paint_traces(axes_dict["traces"], trace_data["t_sec"],
                      frame_idx, trace_data, bg=bg,
                      frame_label_idx=frame_idx, hd_deg=hd_deg)

    fig.savefig(out_path, dpi=360, facecolor=bg)
    return fig, axes_dict


# ── main ─────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anatomy_dir",
                   default=os.path.join(here, "zebrafish_anatomy_HD"))
    p.add_argument("--connconstr_datapath",
                   default=os.path.join(here, "zebrafish_connectome_HD"))
    p.add_argument("--model", default="zebrafish_hd_si_dipn",
                   help="config name for the trained checkpoint")
    p.add_argument("--n_steps", type=int, default=10000)
    p.add_argument("--stride", type=int, default=10,
                   help="render every Nth frame")
    p.add_argument("--omega_deg", type=float, default=60.0)
    p.add_argument("--theta0", type=float, default=0.0)
    p.add_argument("--elev_front", type=float, default=5.7,
                   help="elevation for the frontal (lateral) view")
    p.add_argument("--azim_front", type=float, default=-92.4,
                   help="azimuth for the frontal (lateral) view")
    p.add_argument("--elev_top", type=float, default=90.0,
                   help="elevation for the dorsal (top) view")
    p.add_argument("--azim_top", type=float, default=-85.5,
                   help="azimuth for the dorsal (top) view")
    p.add_argument("--downsample", type=int, default=10)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--max_frames", type=int, default=None,
                   help="stop after N rendered frames (smoke-test)")
    p.add_argument("--z_lo", type=float, default=0.0)
    p.add_argument("--z_hi", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--montage", action="store_true",
                   help="render a cell-type panel montage (frontal view) "
                        "instead of the two-panel frontal+dorsal view")
    p.add_argument("--swim", action="store_true",
                   help="use the original turn-heavy swim-integration "
                        "stimulus (L/R = 80%, the training distribution)")
    p.add_argument("--swim2", action="store_true",
                   help="use a larval-zebrafish-realistic swim "
                        "stimulus (Petrucco-like: ~65%% forward, "
                        "~17%% L/R each, ~1%% backward)")
    p.add_argument("--swim_left", action="store_true",
                   help="deterministic train of LEFT impulses every "
                        "--swim_interval seconds. Pass --swim_interval 0 "
                        "for a single-shot impulse (bump init / "
                        "persistence test).")
    p.add_argument("--swim_right", action="store_true",
                   help="deterministic train of RIGHT impulses every "
                        "--swim_interval seconds.")
    p.add_argument("--swim_interval", type=float, default=1.0,
                   help="for --swim_left/--swim_right: seconds between "
                        "successive deterministic impulses (0 = single "
                        "shot). For --swim/--swim2: mean inter-swim "
                        "period (s), so rate = 1/swim_interval. Default 1s.")
    p.add_argument("--swim_magnitude_rad", type=float, default=0.393,
                   help="magnitude of each deterministic impulse (rad); "
                        "default π/8 ≈ 22.5° per turn (half the "
                        "Petrucco median).")
    p.add_argument("--swim_t_event_s", type=float, default=0.0,
                   help="time of the first deterministic impulse (s)")
    p.add_argument("--scroll_window", type=float, default=10.0,
                   help="trace window width in seconds; traces grow "
                        "left-to-right then scroll once this width is "
                        "reached")
    p.add_argument("--show_roi_overlay", action="store_true",
                   help="draw the four anatomical ROI rectangles (dIPN-L/R, "
                        "dsIPN-L/R) over the dorsal panel. Calibrated for "
                        "the no-swim figsize=(10, 14.6) at dpi=300 render.")
    p.add_argument("--show_roi_kinograph", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="sample mean intensity inside each ROI from the "
                        "rendered canvas per frame, then append a "
                        "black-green kinograph with per-ROI ΔF/F0 traces "
                        "(one colored line per ROI). Implies "
                        "--show_roi_overlay. Pass --no-show_roi_kinograph "
                        "to disable. (default: on)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=None)
    p.add_argument("--bg", default="black", choices=["black", "white"])
    p.add_argument("--prefer_epoch", type=int, default=None)
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass

    if args.swim_left and args.swim_right:
        raise SystemExit("--swim_left and --swim_right are mutually exclusive")
    swim_init_dir = ("L" if args.swim_left
                     else "R" if args.swim_right
                     else None)
    use_swim = args.swim or args.swim2
    if args.swim2:
        # Petrucco-like larval-zebrafish swim statistics.
        swim_kwargs = dict(
            forward_fraction=0.65,
            left_fraction=0.17,
            right_fraction=0.17,
            backward_fraction=0.01,
        )
    else:
        swim_kwargs = {}
    # --swim_interval also tunes the random --swim / --swim2 rate
    # (swim_rate_hz = 1 / swim_interval).
    if use_swim and args.swim_interval and args.swim_interval > 0:
        swim_kwargs["swim_rate_hz"] = 1.0 / float(args.swim_interval)

    if args.out_dir is None:
        if swim_init_dir == "L":
            suffix = f"swim_left_{args.swim_interval:g}s"
        elif swim_init_dir == "R":
            suffix = f"swim_right_{args.swim_interval:g}s"
        elif args.swim2:
            suffix = f"swim2_{args.swim_interval:g}s"
        elif args.swim:
            suffix = f"swim_{args.swim_interval:g}s"
        else:
            suffix = "const"
        args.out_dir = os.path.join(here, f"3D_voltage_{suffix}")
        print(f"[out_dir] auto: {args.out_dir}")

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    stale = _glob.glob(os.path.join(args.out_dir, "frame_*.png"))
    for fp in stale:
        os.remove(fp)
    if stale:
        print(f"      cleared {len(stale)} prior frames from {args.out_dir}/")

    # ── 1. load model ────────────────────────────────────────────────────
    print(f"[1/4] loading model {args.model} ...")
    t0 = time.time()
    net, config = _load(args.model, device, prefer_epoch=args.prefer_epoch)
    dt = float(net.dt)
    print(f"      done ({time.time() - t0:.1f}s)  "
          f"N={net.n_units}  dt={dt}")

    # ── 2. rollout ───────────────────────────────────────────────────────
    # preliminary diagnostics (500 frames)
    print(f"[1.5/4] preliminary 500-frame rollout for LUT diagnostics ...")
    h_prev, *_ = _run_const(net, 500, dt, args.omega_deg, args.theta0,
                            device)
    mu_p = h_prev.mean(axis=0, keepdims=True)
    sd_p = h_prev.std(axis=0, keepdims=True) + 1e-6
    z_p = (h_prev - mu_p) / sd_p
    z_ss = z_p[50:]
    pos_ss = z_ss[z_ss > 0]
    p95 = float(np.percentile(pos_ss, 95)) if pos_ss.size else 1.0
    p99 = float(np.percentile(pos_ss, 99)) if pos_ss.size else 1.0
    z_hi_sat = max(round(p99 * 1.05, 2), args.z_lo + 0.5)
    print(f"      z range: full [{z_p.min():.2f}, {z_p.max():.2f}]; "
          f"steady-state t>=50 [{z_ss.min():.2f}, {z_ss.max():.2f}]")
    print(f"      steady-state z>0 percentiles: 95={p95:.2f}  99={p99:.2f}")
    print(f"      current LUT:    --z_lo={args.z_lo} --z_hi={args.z_hi}")
    print(f"      saturated bump: --z_lo=0.5 --z_hi={z_hi_sat}")

    if swim_init_dir is not None:
        mode = ("single-shot" if args.swim_interval <= 0
                else f"periodic Δt={args.swim_interval:.2f}s")
        print(f"[2/4] running impulse rollout [{mode}]: "
              f"{swim_init_dir} swim from t={args.swim_t_event_s:.2f}s, "
              f"mag={args.swim_magnitude_rad:.3f} rad, "
              f"n_steps={args.n_steps}")
        h_traj, theta, omega_trace, decoded_hd, turn_lr, swim_fb = \
            _run_single_impulse(
                net, args.n_steps, dt, device,
                direction=swim_init_dir,
                magnitude_rad=args.swim_magnitude_rad,
                t_event_s=args.swim_t_event_s,
                interval_s=args.swim_interval,
                theta0=args.theta0,
            )
    elif use_swim:
        flavour = "swim2 (realistic)" if args.swim2 else "swim (turn-heavy)"
        print(f"[2/4] running swim-integration rollout [{flavour}], "
              f"n_steps={args.n_steps} seed={args.seed}")
        h_traj, theta, omega_trace, decoded_hd, turn_lr, swim_fb = \
            _run_swim(net, args.n_steps, dt, device, seed=args.seed,
                      **swim_kwargs)
    else:
        print(f"[2/4] running constant-omega rollout, "
              f"n_steps={args.n_steps} omega={args.omega_deg}")
        h_traj, theta, omega_trace, decoded_hd, turn_lr, swim_fb = \
            _run_const(net, args.n_steps, dt, args.omega_deg, args.theta0,
                       device)

    mu = h_traj.mean(axis=0, keepdims=True)
    sd = h_traj.std(axis=0, keepdims=True) + 1e-6
    z = (h_traj - mu) / sd
    rng_z = max(args.z_hi - args.z_lo, 1e-6)
    rates_lit = np.clip((z - args.z_lo) / rng_z, 0.0, 1.0)
    print(f"      done ({time.time() - t0:.1f}s); "
          f"z range = [{z.min():.2f}, {z.max():.2f}]; "
          f"lit median {np.median(rates_lit):.3f}")

    # ── 3. load skeletons ────────────────────────────────────────────────
    print(f"[3/4] loading skeletons + meshes (downsample={args.downsample}) ...")
    t0 = time.time()
    model_bodyids, model_categories = _model_index_to_bodyid(
        args.connconstr_datapath)
    assert len(model_bodyids) == rates_lit.shape[1], \
        f"model N={len(model_bodyids)} != rollout N={rates_lit.shape[1]}"

    neurons, types_str, has_skel = _load_skeletons_in_model_order(
        args.anatomy_dir, model_bodyids, model_categories,
        downsample=args.downsample,
    )
    n_with = int(has_skel.sum())
    n_without = len(has_skel) - n_with
    print(f"      {n_with} neurons with skeletons, "
          f"{n_without} without (will be invisible)")

    rois = _load_rois(args.anatomy_dir)
    seg_arrays, seg_owner, all_segs = _extract_per_neuron_segments(
        neurons, has_skel)
    soma_xyz, soma_r = _extract_soma_positions(neurons)
    soma_valid = ~np.isnan(soma_xyz[:, 0])
    print(f"      done ({time.time() - t0:.1f}s); "
          f"{all_segs.shape[0]:,} skeleton segments")

    # ── project to 2D (both views) ─────────────────────────────────────
    views = [
        {"elev": args.elev_front, "azim": args.azim_front,
         "title": "frontal"},
        {"elev": args.elev_top, "azim": args.azim_top,
         "title": "dorsal"},
    ]

    # Mesh outline segments (shared 3D data, projected per view)
    mesh_segs = []
    for mesh in rois.values():
        try:
            outline = mesh.outline().entities
            for ent in outline:
                pts = mesh.vertices[ent.points]
                mesh_segs.extend([(pts[i], pts[i + 1])
                                  for i in range(len(pts) - 1)])
        except Exception:
            pass
    mesh_segs3d = np.array(mesh_segs) if mesh_segs else None

    soma_xyz_clean = np.nan_to_num(soma_xyz, nan=0.0)

    view_data = []
    for v in views:
        elev, azim = v["elev"], v["azim"]
        s2d = _project_2d(all_segs.reshape(-1, 3),
                          elev, azim).reshape(-1, 2, 2)
        soma_2d = _project_2d(soma_xyz_clean, elev, azim)
        if mesh_segs3d is not None:
            m2d = _project_2d(mesh_segs3d.reshape(-1, 3),
                              elev, azim).reshape(-1, 2, 2)
        else:
            m2d = None
        all_pts = np.concatenate(
            [s2d.reshape(-1, 2)] +
            ([m2d.reshape(-1, 2)] if m2d is not None else []),
            axis=0,
        )
        pad = 0.04 * (all_pts.max(0) - all_pts.min(0))
        xlim = (all_pts[:, 0].min() - pad[0], all_pts[:, 0].max() + pad[0])
        ylim = (all_pts[:, 1].min() - pad[1], all_pts[:, 1].max() + pad[1])
        view_data.append({
            "segs2d": s2d, "mesh_segs2d": m2d, "soma_2d": soma_2d,
            "xlim": xlim, "ylim": ylim, "title": v["title"],
        })

    # ── trace data for the strip-chart ──────────────────────────────────
    n_total = rates_lit.shape[0]
    t_sec = np.arange(n_total) * dt
    # Virtual swim trajectory: the fish moves one unit per second in its
    # current heading. The 180-degree offset (negated cos/sin) keeps the
    # motion direction consistent with the head-on-the-left silhouette
    # convention: at theta=0 the fish points LEFT and the trajectory
    # advances LEFT.
    v_fish = 1.0
    swim_x = np.cumsum(-v_fish * dt * np.cos(theta[:n_total]))
    swim_y = np.cumsum(-v_fish * dt * np.sin(theta[:n_total]))
    trace_data = {
        "omega": omega_trace[:n_total],
        "theta": theta[:n_total],
        "decoded_hd": decoded_hd[:n_total],
        "dt": dt,
        "t_sec": t_sec,
        "turn_lr": turn_lr[:n_total] if turn_lr is not None else None,
        "swim_fb": swim_fb[:n_total] if swim_fb is not None else None,
        "scroll_window": args.scroll_window,
        "swim_x": swim_x,
        "swim_y": swim_y,
    }

    # ── 4. render frames ─────────────────────────────────────────────────
    print(f"[4/4] rendering frames into {args.out_dir}/")
    n_render = n_total
    frame_ids = list(range(0, n_render, args.stride))
    if args.max_frames is not None:
        frame_ids = frame_ids[:args.max_frames]

    fig, ax = None, None
    render_times = []
    fig_reset_every = 250
    # --show_roi_kinograph implies --show_roi_overlay (drosophila convention).
    if args.show_roi_kinograph:
        args.show_roi_overlay = True
    roi_overlay = None
    if args.show_roi_overlay:
        roi_overlay = {
            "first_only": True,
            "kinograph_enabled": bool(args.show_roi_kinograph),
            "intens_buf": [],
            "intens_t_buf": [],
            "dt": dt,
            "scroll_window": args.scroll_window,
        }
    pbar = tqdm(frame_ids, desc="rendering", unit="frame", ncols=150)
    for k, t in enumerate(pbar):
        if k > 0 and k % fig_reset_every == 0 and fig is not None:
            plt.close(fig)
            fig, ax = None, None
            import gc
            gc.collect()
        tic = time.time()
        out = os.path.join(args.out_dir, f"frame_{t:04d}.png")
        if args.montage:
            fig, ax = _render_montage_frame(
                out, view_data, seg_owner, types_str, rates_lit[t],
                soma_valid=soma_valid,
                frame_idx=t, hd_deg=float(np.rad2deg(theta[t])),
                alpha_max=args.alpha, bg=args.bg,
                fig_ref=fig, axes_ref=ax,
                trace_data=trace_data,
            )
        else:
            fig, ax = _render_frame(
                out, view_data, seg_owner, rates_lit[t],
                frame_idx=t,
                hd_deg=float(np.rad2deg(theta[t])),
                alpha_max=args.alpha, bg=args.bg,
                fig_ref=fig, axes_ref=ax,
                soma_valid=soma_valid,
                trace_data=trace_data,
                roi_overlay=roi_overlay,
            )
        render_times.append(time.time() - tic)
        pbar.set_postfix(s_per_frame=f"{np.mean(render_times):.2f}")

    plt.close(fig)
    print(f"done: {len(frame_ids)} frames, "
          f"mean {np.mean(render_times):.2f}s/frame, "
          f"total {sum(render_times):.1f}s")

    # Persist the per-frame ROI traces for downstream analysis.
    if roi_overlay is not None and roi_overlay.get("intens_buf"):
        import pandas as pd
        t_arr = np.asarray(roi_overlay["intens_t_buf"], dtype=np.float32)
        I_arr = np.asarray(roi_overlay["intens_buf"], dtype=np.float32)
        names = [n for n, *_ in ZHD_ROI_RECTS_FIJI]
        df = pd.DataFrame(I_arr, columns=names)
        df.insert(0, "t_s", t_arr)
        # image-based PVA (radians, in [0, 2π))
        n_w = I_arr.shape[1]
        ang = 2.0 * np.pi * np.arange(n_w) / n_w
        w = np.clip(I_arr, 0.0, None) + 1e-9
        df["pva_rad"] = np.arctan2(
            (w * np.sin(ang)).sum(1), (w * np.cos(ang)).sum(1)) % (2.0 * np.pi)
        csv_path = os.path.join(args.out_dir, "roi_kinograph.csv")
        df.to_csv(csv_path, index=False)
        print(f"wrote {csv_path}  ({len(t_arr)} samples)")


if __name__ == "__main__":
    main()

# ── example usage ────────────────────────────────────────────────────────
#
# Two-panel (frontal + dorsal) constant-omega:
#   python figures/zebrafish/fig_zebrafish_anatomy_3d_voltage_anim.py \
#     --z_lo 0 --z_hi 20 --alpha 1.0
#
# Two-panel swim-integration rollout:
#   python figures/zebrafish/fig_zebrafish_anatomy_3d_voltage_anim.py \
#     --swim --z_lo 0 --z_hi 20 --alpha 1.0
#
# Cell-type montage (frontal view only):
#   python figures/zebrafish/fig_zebrafish_anatomy_3d_voltage_anim.py \
#     --montage --z_lo 0 --z_hi 20 --alpha 1.0
#
# Quick smoke test (5 frames):
#   python figures/zebrafish/fig_zebrafish_anatomy_3d_voltage_anim.py \
#     --max_frames 5 --z_lo 0 --z_hi 4

# python /workspace/connectome-gnn-cx/figures/zebrafish/fig_zebrafish_anatomy_3d_voltage_anim.py --z_lo=0.0 --z_hi=6.0 --swim
# python figures/zebrafish/fig_zebrafish_anatomy_3d_voltage_anim.py --model zebrafish_hd_si_dipn --n_steps 10000 --stride 5 --z_lo 0.0 --z_hi 15.0 --swim_left --swim_interval 0.3 --out_dir figures/zebrafish/3D_voltage_const