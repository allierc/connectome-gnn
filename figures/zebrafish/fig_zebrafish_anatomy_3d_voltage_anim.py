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
    """Single-trial swim-integration stimulus (B=1)."""
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
    turn_lr[m_left] = +np.rad2deg(mag_LR[m_left])
    turn_lr[m_right] = -np.rad2deg(mag_LR[m_right])
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
                 base_color, lw_base, lw_top, soma_size):
    """Draw one view panel (base skeleton + green overlay + soma dots)."""
    ax.set_facecolor(bg)

    if mesh_segs2d is not None and len(mesh_segs2d):
        ax.add_collection(LineCollection(
            mesh_segs2d, colors=("0.85" if bg == "black" else "0.45",),
            linewidths=0.25, alpha=0.12,
        ))

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
        colors_lr = np.where(vals > 0, "#4488ff", "#ff4444")
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
                  trace_data=None):
    """Render three rows: frontal anatomy, trace strip, dorsal anatomy.

    view_data: list of dicts (frontal, dorsal).
    trace_data: dict with keys omega, theta, decoded_hd, dt, t_sec
                (None = no trace strip, just the two anatomy panels).
    """
    has_traces = trace_data is not None
    has_swim = (has_traces and trace_data.get("turn_lr") is not None)
    n_trace_rows = (4 if has_swim else 2) if has_traces else 0

    if fig_ref is None:
        from matplotlib.gridspec import GridSpec
        if has_traces:
            trace_h = [1] * n_trace_rows
            ratios = [3.5, 3.5] + trace_h
            total_h = 10.0 + (1.5 if has_swim else 0.0)
            fig = plt.figure(figsize=(10.0, total_h), facecolor=bg)
            gs = GridSpec(len(ratios), 1, figure=fig,
                         height_ratios=ratios, hspace=0.0,
                         top=0.995, bottom=0.04, left=0.10, right=0.995)
            axes = [fig.add_subplot(gs[i]) for i in range(len(ratios))]
        else:
            fig = plt.figure(figsize=(10.0, 11.0), facecolor=bg)
            gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1],
                         hspace=0.03,
                         top=0.995, bottom=0.005, left=0.005, right=0.995)
            axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    else:
        fig, axes = fig_ref, axes_ref
        for a in axes:
            a.clear()
        for txt in list(fig.texts):
            txt.remove()

    txt_color = "white" if bg == "black" else "black"

    # Anatomy panels: first two axes (frontal, dorsal)
    for i, vd in enumerate(view_data):
        ax = axes[i]
        _paint_panel(ax, vd["segs2d"], seg_owner, rates_t,
                     vd["mesh_segs2d"], vd["soma_2d"], soma_valid,
                     vd["xlim"], vd["ylim"], bg, green, alpha_max,
                     base_color, lw_base, lw_top, soma_size)
        if has_traces:
            ax.set_anchor("S" if i == 0 else "N")
        ax.text(0.02, 0.97, vd["title"], color=txt_color, fontsize=9,
                family="monospace", ha="left", va="top",
                transform=ax.transAxes)

    # Trace strip (axes after the two anatomy panels)
    if has_traces and frame_idx is not None:
        trace_axes = axes[2:]
        _paint_traces(trace_axes, trace_data["t_sec"], frame_idx,
                      trace_data, bg=bg,
                      frame_label_idx=frame_idx, hd_deg=hd_deg)

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

        for col, ct in enumerate(MONTAGE_TYPES):
            ax = panel_axes[col]
            ax.set_facecolor(bg)

            if mesh_segs2d is not None and len(mesh_segs2d):
                ax.add_collection(LineCollection(
                    mesh_segs2d, colors=(mesh_color,),
                    linewidths=0.2, alpha=0.10,
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
                   help="use swim-integration stimulus instead of "
                        "constant-omega")
    p.add_argument("--scroll_window", type=float, default=10.0,
                   help="trace window width in seconds; traces grow "
                        "left-to-right then scroll once this width is "
                        "reached")
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

    if args.out_dir is None:
        suffix = "swim" if args.swim else "const"
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

    if args.swim:
        print(f"[2/4] running swim-integration rollout, "
              f"n_steps={args.n_steps} seed={args.seed}")
        h_traj, theta, omega_trace, decoded_hd, turn_lr, swim_fb = \
            _run_swim(net, args.n_steps, dt, device, seed=args.seed)
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
    trace_data = {
        "omega": omega_trace[:n_total],
        "theta": theta[:n_total],
        "decoded_hd": decoded_hd[:n_total],
        "dt": dt,
        "t_sec": t_sec,
        "turn_lr": turn_lr[:n_total] if turn_lr is not None else None,
        "swim_fb": swim_fb[:n_total] if swim_fb is not None else None,
        "scroll_window": args.scroll_window,
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
            )
        render_times.append(time.time() - tic)
        pbar.set_postfix(s_per_frame=f"{np.mean(render_times):.2f}")

    plt.close(fig)
    print(f"done: {len(frame_ids)} frames, "
          f"mean {np.mean(render_times):.2f}s/frame, "
          f"total {sum(render_times):.1f}s")


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