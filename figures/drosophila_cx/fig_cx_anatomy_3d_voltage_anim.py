"""Voltage animation on the 3-D CX anatomy.

Runs the Known-ODE RNN under a constant-omega rollout, computes per-neuron
firing rate r_i(t) = sigmoid(h_i(t)), and renders one PNG every K frames
showing every CX skeleton in dark grey overlaid with a green tint whose
alpha is the current rate. Output: figures/drosophila/3D/frame_NNNN.png.

The geometry is the same hemibrain SWC pull used by fig_cx_anatomy_3d.py
(papers/janelia_cx/anatomy/cx_anatomy_test/). The model -> bodyId mapping
replays load_drosophila_cx_connectome's selection so model index i lines
up with the correct skeleton.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import navis
from tqdm import tqdm

from fig_cx_anatomy_3d import (
    TYPE_COLOR, TYPE_ORDER, _load_rois, _project_2d,
)
from fig_kinographs_const_omega import (
    _load, _run_const, _preferred_phase,
    _order_by_type_descending, _order_within_type_by_phase,
    _order_by_preferred_phase,
)

from connectome_gnn.generators.utils import generate_path_integration_batch


def _run_ou_rollout(net, n_steps, device, seed=0):
    """Natural OU velocity stream; returns (h_traj, theta_hd) like _run_const."""
    rng = np.random.default_rng(seed)
    batch = generate_path_integration_batch(
        batch_size=1, n_steps=n_steps, dt=float(net.dt),
        device=device, rng=rng,
    )
    with torch.no_grad():
        _, h = net(batch.stimulus)
    return h[0].cpu().numpy(), batch.theta_hd[0].cpu().numpy()


from fig_kinographs_const_omega import _build_const_omega_batch


def _run_const_with_traces(net, n_steps, dt, omega_deg, theta0, device):
    """Like _run_const but also returns omega and decoded HD."""
    batch = _build_const_omega_batch(n_steps, dt, omega_deg, theta0, device)
    theta = batch.theta_hd[0].cpu().numpy()
    omega = batch.omega[0].cpu().numpy()
    with torch.no_grad():
        y_hat, h = net(batch.stimulus)
    decoded_hd = np.arctan2(y_hat[0, :, 1].cpu().numpy(),
                            y_hat[0, :, 0].cpu().numpy())
    return h[0].cpu().numpy(), theta, omega, decoded_hd


def _run_ou_with_traces(net, n_steps, device, seed=0):
    """Like _run_ou_rollout but also returns omega and decoded HD."""
    rng = np.random.default_rng(seed)
    batch = generate_path_integration_batch(
        batch_size=1, n_steps=n_steps, dt=float(net.dt),
        device=device, rng=rng,
    )
    with torch.no_grad():
        y_hat, h = net(batch.stimulus)
    theta = batch.theta_hd[0].cpu().numpy()
    omega = batch.omega[0].cpu().numpy()
    decoded_hd = np.arctan2(y_hat[0, :, 1].cpu().numpy(),
                            y_hat[0, :, 0].cpu().numpy())
    return h[0].cpu().numpy(), theta, omega, decoded_hd


from connectome_gnn.utils import load_data_root_from_json, set_data_root


# Permutation reordering EPG indices in the connectome loader (glomerular
# ring order). Verbatim from connconstr_data.py.
EPG_PERM = np.array([
    23, 24, 0, 1, 42, 43, 44, 45, 2, 3, 39, 40, 41, 4, 5, 6,
    36, 37, 38, 7, 8, 9, 33, 34, 35, 10, 11, 12,
    30, 31, 32, 13, 14, 15, 27, 28, 29, 16, 17, 18,
    25, 26, 19, 20, 21, 22,
])


def _model_index_to_bodyid(datapath: str) -> np.ndarray:
    """Replay load_drosophila_cx_connectome's neuron selection so that
    model index i corresponds to a specific hemibrain bodyId."""
    neuronsall = pd.read_csv(os.path.join(datapath, "traced-neurons.csv"))
    neuronsall.sort_values(by=["instance"], ignore_index=True, inplace=True)
    types = np.array(neuronsall.type).astype(str)

    def sub(t):
        return np.nonzero([t in x for x in types])[0]

    epg, pen = sub("EPG"), sub("PEN")
    peg, delta7 = sub("PEG"), sub("Delta7")
    allcx = np.concatenate((epg, pen, delta7, peg))
    allcx[0:46] = allcx[EPG_PERM]
    er6 = np.array([i for i, t in enumerate(types) if t == "ER6"], dtype=int)
    if er6.size:
        allcx = np.concatenate((allcx, er6))
    return neuronsall.bodyId.values[allcx]


def _load_skeletons_in_model_order(anatomy_dir: str, body_ids: np.ndarray,
                                    downsample: int = 10):
    """Return a list of 156 navis TreeNeurons indexed by model order, and
    a parallel list of cell-type strings for colour lookup."""
    swcs = {}
    for fname in os.listdir(os.path.join(anatomy_dir, "skeletons")):
        if not fname.endswith(".swc"):
            continue
        stem = fname[:-4]
        safe_t, _, bid_str = stem.rpartition("__")
        swcs[int(bid_str)] = (
            os.path.join(anatomy_dir, "skeletons", fname),
            safe_t,
        )

    neurons = []
    types = []
    for bid in body_ids:
        if int(bid) not in swcs:
            raise SystemExit(f"missing skeleton for bodyId {bid}")
        path, safe_t = swcs[int(bid)]
        n = navis.read_swc(path)
        if downsample and downsample > 1:
            n = navis.downsample_neuron(n, downsampling_factor=downsample,
                                         preserve_nodes=None)
        neurons.append(n)
        for canon in TYPE_ORDER:
            safe = canon.replace("(", "_").replace(")", "")
            if safe_t == safe:
                types.append(canon); break
        else:
            types.append(safe_t)
    return neurons, types


def _extract_per_neuron_segments(neurons):
    """Return:
      seg_arrays: list of (E_i, 2, 3) arrays per neuron
      seg_owner:  flat (E_total,) int array, neuron index per segment
      all_segs:   stacked (E_total, 2, 3) array
    """
    seg_arrays = []
    for n in neurons:
        nodes = n.nodes
        child = nodes[nodes.parent_id != -1]
        if len(child) == 0:
            seg_arrays.append(np.zeros((0, 2, 3), dtype=np.float32)); continue
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
    """Hemibrain SWCs don't tag the soma (n.soma is None); we use the
    largest-radius node as a robust soma proxy. Returns:
      soma_xyz: (N, 3) float array
      soma_r:   (N,)  float array (radius in SWC units)
    """
    soma_xyz = np.zeros((len(neurons), 3), dtype=np.float32)
    soma_r = np.zeros(len(neurons), dtype=np.float32)
    for i, n in enumerate(neurons):
        nodes = n.nodes
        idx = int(nodes.radius.idxmax())
        row = nodes.loc[idx]
        soma_xyz[i] = [float(row.x), float(row.y), float(row.z)]
        soma_r[i] = float(row.radius)
    return soma_xyz, soma_r


def _style_trace_ax(ax, bg, ylabel, fontsize=11, bottom_labels=False):
    txt_color = "white" if bg == "black" else "black"
    ax.set_facecolor(bg)
    ax.set_ylabel(ylabel, color=txt_color, fontsize=fontsize, labelpad=2)
    ax.tick_params(axis="y", colors=txt_color, labelsize=10, length=3)
    ax.tick_params(axis="x", colors=txt_color, labelsize=10, length=3,
                   labelbottom=bottom_labels)
    ax.spines[:].set_visible(False)


def _paint_traces(trace_axes, t_sec, frame_t, trace_data, bg="black"):
    """Grow-then-scroll trace strip: ω(t) and HD (target vs decoded)."""
    txt_color = "white" if bg == "black" else "black"
    dim = "0.35" if bg == "black" else "0.70"

    omega_full = trace_data["omega"]
    theta_full = trace_data["theta"]
    decoded_hd_full = trace_data["decoded_hd"]
    win = trace_data.get("scroll_window", 10.0)

    n_now = frame_t + 1
    t_now_val = t_sec[frame_t]

    if t_now_val < win:
        x_lo, x_hi = 0.0, win
    else:
        x_lo, x_hi = t_now_val - win, t_now_val

    vis_mask = (t_sec[:n_now] >= x_lo) & (t_sec[:n_now] <= x_hi)

    # ── ω panel
    ax = trace_axes[0]
    ax.plot(t_sec[:n_now], omega_full[:n_now],
            color=(0.0, 0.85, 0.4), lw=1.4)
    ax.axhline(0, color=dim, lw=0.3, alpha=0.4)
    ax.set_xlim(x_lo, x_hi)
    o_abs = max(np.abs(omega_full).max(), 1.0)
    ax.set_ylim(-o_abs * 1.15, o_abs * 1.15)
    _style_trace_ax(ax, bg, "ω (°/s)")

    # ── HD panel
    ax = trace_axes[1]
    target_deg = np.rad2deg(theta_full)
    decoded_deg = np.rad2deg(decoded_hd_full)
    ax.plot(t_sec[:n_now], target_deg[:n_now], color=(0.0, 0.85, 0.4),
            lw=1.4, label="target")
    ax.plot(t_sec[:n_now], decoded_deg[:n_now],
            color="white" if bg == "black" else "black",
            lw=1.4, label="decoded")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-180, 180)
    _style_trace_ax(ax, bg, "HD (°)", bottom_labels=True)
    ax.set_xlabel("time (s)", color=txt_color, fontsize=11, labelpad=2)
    # Current HD value in the panel's top-right corner.
    hd_now_deg = float(np.rad2deg(theta_full[frame_t]))
    ax.text(0.99, 0.96, f"HD = {hd_now_deg:+.0f}°",
            color=txt_color, fontsize=11, family="monospace",
            ha="right", va="top", transform=ax.transAxes)


def _render_frame(out_path, segs2d, seg_owner, rates_t, mesh_segs2d,
                  bg="black", lw_base=0.18, lw_top=0.45,
                  base_color=(0.25, 0.25, 0.25), green=(0.0, 1.0, 0.3),
                  alpha_max=1.0,
                  xlim=None, ylim=None, frame_idx=None, total_frames=None,
                  hd_deg=None, ax_ref=None, fig_ref=None,
                  soma_2d=None, soma_size=18.0,
                  trace_data=None):
    """Render a single animation frame -- two LineCollections (dark base
    + green overlay with per-segment alpha) on top of the neuropil
    silhouette, with optional trace strip at the bottom."""
    has_traces = trace_data is not None
    if fig_ref is None:
        if has_traces:
            from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
            fig = plt.figure(figsize=(7.5, 12.0), facecolor=bg)
            gs = GridSpec(2, 1, figure=fig, height_ratios=[5, 2],
                          hspace=0.06)
            ax = fig.add_subplot(gs[0])
            gs_tr = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1],
                                             hspace=0.0)
            ax_omega = fig.add_subplot(gs_tr[0])
            ax_hd = fig.add_subplot(gs_tr[1])
            ax_ref = [ax, ax_omega, ax_hd]
        else:
            fig, ax = plt.subplots(figsize=(7.5, 8.5), facecolor=bg)
            ax_ref = ax
    else:
        fig = fig_ref
        if has_traces:
            ax = ax_ref[0]; ax.clear()
            ax_ref[1].clear(); ax_ref[2].clear()
        else:
            ax = ax_ref; ax.clear()
        for txt in list(fig.texts):
            txt.remove()
    if has_traces:
        ax = ax_ref[0]
    ax.set_facecolor(bg)

    # Neuropil silhouette
    if mesh_segs2d is not None and len(mesh_segs2d):
        ax.add_collection(LineCollection(
            mesh_segs2d, colors=("0.85" if bg == "black" else "0.45",),
            linewidths=0.25, alpha=0.12,
        ))

    # Base layer: every neuron in dark grey
    ax.add_collection(LineCollection(
        segs2d, colors=[base_color], linewidths=lw_base, alpha=0.5,
    ))

    # Green overlay: per-segment alpha driven by the owning neuron's rate.
    alpha = rates_t[seg_owner] * alpha_max
    # Drop segments whose rate is essentially zero to skip useless overdraw.
    keep = alpha > 0.02
    if keep.any():
        rgba = np.tile(np.array([*green, 1.0], dtype=np.float32),
                       (int(keep.sum()), 1))
        rgba[:, 3] = alpha[keep]
        ax.add_collection(LineCollection(
            segs2d[keep], colors=rgba, linewidths=lw_top,
        ))

    # Soma layer: dim grey dot for every neuron, green dot scaled by alpha
    # for active ones. Sits BELOW the skeleton lines so off-neuron grey
    # dots get hidden behind the skeleton trees.
    if soma_2d is not None and len(soma_2d):
        ax.scatter(soma_2d[:, 0], soma_2d[:, 1],
                   s=soma_size * 0.5, c=[base_color], edgecolors="none",
                   alpha=0.7, zorder=0)
        keep_n = rates_t > 0.02
        if keep_n.any():
            rgba_s = np.tile(np.array([*green, 1.0], dtype=np.float32),
                              (int(keep_n.sum()), 1))
            rgba_s[:, 3] = rates_t[keep_n]
            ax.scatter(soma_2d[keep_n, 0], soma_2d[keep_n, 1],
                       s=soma_size, c=rgba_s, edgecolors="none",
                       zorder=1)

    if xlim is not None:
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    else:
        ax.autoscale_view()
    # 180-deg rotation: dorsal up convention for fly CX.
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Frame index / HD label only in the no-trace layout; the trace-strip
    # layout prints HD in the HD panel's top-right corner instead.
    if not has_traces and frame_idx is not None:
        txt_color = "white" if bg == "black" else "black"
        label = f"t = {frame_idx:04d}"
        if hd_deg is not None:
            label += f"  HD = {hd_deg:+.0f}°"
        ax.text(0.02, 0.97, label, color=txt_color, fontsize=10,
                family="monospace", ha="left", va="top",
                transform=ax.transAxes)

    # Trace strip
    if has_traces and frame_idx is not None:
        _paint_traces(ax_ref[1:], trace_data["t_sec"], frame_idx,
                      trace_data, bg=bg)
        fig.subplots_adjust(top=0.995, bottom=0.06,
                            left=0.10, right=0.98)
        # The trace panels need left margin for the y-tick labels, but
        # the 3D view should span the full figure width so the brain is
        # centered. Re-place only the 3D axes after subplots_adjust.
        ax3d_pos = ax_ref[0].get_position()
        ax_ref[0].set_position([0.02, ax3d_pos.y0,
                                 0.96, ax3d_pos.height])

    fig.savefig(out_path, dpi=300, facecolor=bg)
    return fig, ax_ref


# Cell-type panel order for the montage (matches TYPE_ORDER from the
# anatomy script, plus an "all" last panel).
MONTAGE_TYPES = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)",
                 "Delta7", "PEG", "ER6", "all"]


def _render_montage_frame(out_path, segs2d, seg_owner, types_str, rates_t,
                           mesh_segs2d=None, soma_2d=None,
                           xlim=None, ylim=None,
                           frame_idx=None, hd_deg=None,
                           bg="black",
                           green=(0.0, 1.0, 0.3),
                           alpha_max=1.0,
                           base_color=(0.22, 0.22, 0.22),
                           lw_base=0.10, lw_top=0.40, soma_size=10.0,
                           fig_ref=None, axes_ref=None):
    """One PNG per frame: 2x4 grid of cell-type panels. Each panel draws
    the full skeleton bundle in dark grey and overlays only the named
    cell type's neurons in green (alpha = rates_t). Last panel = all
    types together for context."""
    types_arr = np.asarray(types_str)

    if fig_ref is None:
        fig, axes = plt.subplots(2, 4, figsize=(11.0, 6.0),
                                  facecolor=bg, squeeze=False)
        axes = list(axes.flat)
    else:
        fig, axes = fig_ref, axes_ref
        for a in axes:
            a.clear()
        # Clear any free-floating fig-level text from previous frame
        for txt in list(fig.texts):
            txt.remove()

    txt_color = "white" if bg == "black" else "black"
    mesh_color = "0.85" if bg == "black" else "0.45"

    for panel_idx, ct in enumerate(MONTAGE_TYPES):
        ax = axes[panel_idx]
        ax.set_facecolor(bg)

        # Neuropil silhouette (very dim) for anatomical context
        if mesh_segs2d is not None and len(mesh_segs2d):
            ax.add_collection(LineCollection(
                mesh_segs2d, colors=(mesh_color,),
                linewidths=0.2, alpha=0.10,
            ))

        # Per-panel mask
        if ct == "all":
            mask_n = np.ones(len(types_arr), dtype=bool)
        else:
            mask_n = (types_arr == ct)
        mask = mask_n[seg_owner]

        # Skeleton base layer -- only neurons of this type (dim grey)
        if mask.any():
            ax.add_collection(LineCollection(
                segs2d[mask], colors=[base_color],
                linewidths=lw_base, alpha=0.55,
            ))

        # Green overlay for this type only
        alpha = rates_t[seg_owner] * mask * alpha_max
        keep = alpha > 0.02
        if keep.any():
            rgba = np.tile(np.array([*green, 1.0], dtype=np.float32),
                           (int(keep.sum()), 1))
            rgba[:, 3] = alpha[keep]
            ax.add_collection(LineCollection(
                segs2d[keep], colors=rgba, linewidths=lw_top,
            ))

        if soma_2d is not None and len(soma_2d):
            ax.scatter(soma_2d[mask_n, 0], soma_2d[mask_n, 1],
                       s=soma_size * 0.5, c=[base_color],
                       edgecolors="none", alpha=0.7, zorder=0)
            lit_n = (rates_t > 0.02) & mask_n
            if lit_n.any():
                rgba_s = np.tile(np.array([*green, 1.0], dtype=np.float32),
                                  (int(lit_n.sum()), 1))
                rgba_s[:, 3] = rates_t[lit_n]
                ax.scatter(soma_2d[lit_n, 0], soma_2d[lit_n, 1],
                           s=soma_size, c=rgba_s,
                           edgecolors="none", zorder=1)

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.invert_xaxis(); ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_axis_off()
        n_count = int(mask_n.sum())
        title = ct if ct != "all" else "all (156)"
        if ct != "all":
            title = f"{ct}  (n={n_count})"
        ax.text(0.02, 0.97, title, color=txt_color, fontsize=9,
                family="monospace", ha="left", va="top",
                transform=ax.transAxes)

    # Global frame label in the figure suptitle area
    if frame_idx is not None:
        label = f"t = {frame_idx:04d}"
        if hd_deg is not None:
            label += f"   HD = {hd_deg:+.0f}°"
        fig.text(0.5, 0.985, label, color=txt_color, fontsize=11,
                 family="monospace", ha="center", va="top")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.965,
                        bottom=0.005, wspace=0.02, hspace=0.06)
    fig.savefig(out_path, dpi=360, facecolor=bg)
    return fig, axes


def _render_init_montage_frame(out_path, segs2d, seg_owner,
                                rates_per_init, theta_per_init,
                                init_thetas_deg,
                                mesh_segs2d=None, soma_2d=None,
                                xlim=None, ylim=None, frame_idx=None,
                                bg="black",
                                green=(0.0, 1.0, 0.3),
                                alpha_max=1.0,
                                base_color=(0.22, 0.22, 0.22),
                                lw_base=0.10, lw_top=0.40, soma_size=10.0,
                                fig_ref=None, axes_ref=None):
    """2x2 montage: same time index t shown for 4 different theta0 values.
    rates_per_init: list of (T, N) arrays, one per theta0.
    theta_per_init: list of (T,) arrays, one per theta0."""
    if fig_ref is None:
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 10.0),
                                  facecolor=bg, squeeze=False)
        axes = list(axes.flat)
    else:
        fig, axes = fig_ref, axes_ref
        for a in axes:
            a.clear()
        for txt in list(fig.texts):
            txt.remove()

    txt_color = "white" if bg == "black" else "black"
    mesh_color = "0.85" if bg == "black" else "0.45"

    for panel_idx, t0_deg in enumerate(init_thetas_deg):
        ax = axes[panel_idx]
        ax.set_facecolor(bg)

        if mesh_segs2d is not None and len(mesh_segs2d):
            ax.add_collection(LineCollection(
                mesh_segs2d, colors=(mesh_color,),
                linewidths=0.2, alpha=0.10,
            ))

        ax.add_collection(LineCollection(
            segs2d, colors=[base_color],
            linewidths=lw_base, alpha=0.45,
        ))

        rates_t = rates_per_init[panel_idx][frame_idx]
        alpha = rates_t[seg_owner] * alpha_max
        keep = alpha > 0.02
        if keep.any():
            rgba = np.tile(np.array([*green, 1.0], dtype=np.float32),
                           (int(keep.sum()), 1))
            rgba[:, 3] = alpha[keep]
            ax.add_collection(LineCollection(
                segs2d[keep], colors=rgba, linewidths=lw_top,
            ))

        if soma_2d is not None and len(soma_2d):
            ax.scatter(soma_2d[:, 0], soma_2d[:, 1],
                       s=soma_size * 0.5, c=[base_color],
                       edgecolors="none", alpha=0.7, zorder=0)
            lit_n = rates_t > 0.02
            if lit_n.any():
                rgba_s = np.tile(np.array([*green, 1.0], dtype=np.float32),
                                  (int(lit_n.sum()), 1))
                rgba_s[:, 3] = rates_t[lit_n]
                ax.scatter(soma_2d[lit_n, 0], soma_2d[lit_n, 1],
                           s=soma_size, c=rgba_s,
                           edgecolors="none", zorder=1)

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.invert_xaxis(); ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_axis_off()
        hd_now = float(np.rad2deg(theta_per_init[panel_idx][frame_idx]))
        ax.text(0.02, 0.97,
                f"theta_0 = {t0_deg:+.0f} deg\nHD = {hd_now:+.0f} deg",
                color=txt_color, fontsize=10, family="monospace",
                ha="left", va="top", transform=ax.transAxes)

    if frame_idx is not None:
        fig.text(0.5, 0.985, f"t = {frame_idx:04d}",
                 color=txt_color, fontsize=12, family="monospace",
                 ha="center", va="top")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.965,
                        bottom=0.005, wspace=0.02, hspace=0.04)
    fig.savefig(out_path, dpi=440, facecolor=bg)
    return fig, axes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anatomy_dir",
                   default="papers/janelia_cx/anatomy/cx_anatomy_test")
    p.add_argument("--datapath",
                   default="papers/Code_NN/Code_NN/Data/Figure5/"
                           "exported-traced-adjacencies-v1.2")
    p.add_argument("--model", default="drosophila_cx_pi_epg",
                   help="config name for the Known-ODE checkpoint")
    p.add_argument("--n_steps", type=int, default=2000,
                   help="number of rollout frames")
    p.add_argument("--stride", type=int, default=2,
                   help="render every Nth frame")
    p.add_argument("--omega_deg", type=float, default=60.0)
    p.add_argument("--theta0", type=float, default=0.0)
    p.add_argument("--elev", type=float, default=-7.6)
    p.add_argument("--azim", type=float, default=86.6)
    p.add_argument("--downsample", type=int, default=10)
    p.add_argument("--out_dir", default=None,
                   help="output directory for frame_NNNN.png files. "
                        "Default: figures/drosophila_cx/3D_<model-suffix> "
                        "(suffix = model name with drosophila_cx_pi_ stripped).")
    p.add_argument("--max_frames", type=int, default=None,
                   help="stop after N rendered frames (smoke-test)")
    p.add_argument("--z_lo", type=float, default=0.0,
                   help="z-score threshold: only z > z_lo lights up.")
    p.add_argument("--z_hi", type=float, default=4.0,
                   help="z-score saturation point: alpha=1 at z >= z_hi.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="global multiplier on the per-segment green alpha "
                        "(0 = off, 1 = current behavior).")
    p.add_argument("--reorder", default="none",
                   choices=["none", "type", "type_phase", "phase"],
                   help="permute the per-neuron voltage vector before "
                        "lighting the 3D skeletons. type_phase = Fig 9 "
                        "middle sort (cell-type primary, preferred-phase "
                        "secondary).")
    p.add_argument("--montage", action="store_true",
                   help="render a 2x4 cell-type panel montage per frame "
                        "instead of the single 3D view.")
    p.add_argument("--slow_motion", type=int, default=1,
                   help="time-interpolate the first --slow_init frames of "
                        "the rollout to N times as many output frames.")
    p.add_argument("--slow_init", type=int, default=40,
                   help="how many original model frames to stretch when "
                        "--slow_motion > 1 (default 40 = bump-formation).")
    p.add_argument("--init_montage", action="store_true",
                   help="render a 2x2 montage of the same time index across "
                        "4 initial headings (--init_thetas).")
    p.add_argument("--init_thetas", default="0,90,180,270",
                   help="comma-separated theta0 values in degrees for the "
                        "init montage (default 0,90,180,270).")
    p.add_argument("--ou", action="store_true",
                   help="use natural OU velocity rollout instead of "
                        "constant-omega.")
    p.add_argument("--seed", type=int, default=0,
                   help="rng seed for the OU rollout")
    p.add_argument("--scroll_window", type=float, default=10.0,
                   help="trace window width in seconds; traces grow "
                        "left-to-right then scroll")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=None)
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass

    if args.out_dir is None:
        suffix = args.model
        for prefix in ("drosophila_cx_pi_", "drosophila_cx_"):
            if suffix.startswith(prefix):
                suffix = suffix[len(prefix):]
                break
        args.out_dir = os.path.join("figures", "drosophila_cx",
                                     f"3D_{suffix}")
        print(f"[out_dir] auto-derived from --model {args.model}: "
              f"{args.out_dir}")

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    # Empty the output directory of any prior frame_*.png so we don't
    # mix runs.
    import glob as _glob
    stale = _glob.glob(os.path.join(args.out_dir, "frame_*.png"))
    for p in stale:
        os.remove(p)
    if stale:
        print(f"      cleared {len(stale)} prior frames from {args.out_dir}/")

    print(f"[1/4] loading model {args.model} ...")
    t0 = time.time()
    net = _load(args.model, device)
    print(f"      done ({time.time() - t0:.1f}s)")

    # --- preliminary 500-frame rollout: z-score diagnostics + LUT suggest
    # The first ~50 frames carry the bump-formation transient (huge |z|).
    # Steady-state stats (t>=50) drive the recommendation.
    print(f"[1.5/4] preliminary 500-frame rollout for LUT diagnostics ...")
    h_prev, theta_prev = _run_const(net, 500, float(net.dt),
                                     args.omega_deg, args.theta0, device)
    mu_p = h_prev.mean(axis=0, keepdims=True)
    sd_p = h_prev.std (axis=0, keepdims=True) + 1e-6
    z_p = (h_prev - mu_p) / sd_p
    peak = z_p.max(axis=1)
    trough = z_p.min(axis=1)
    z_ss = z_p[50:]
    pos_ss = z_ss[z_ss > 0]
    neg_ss = z_ss[z_ss < 0]
    p95 = float(np.percentile(pos_ss, 95)) if pos_ss.size else 1.0
    p99 = float(np.percentile(pos_ss, 99)) if pos_ss.size else 1.0
    n05 = float(np.percentile(neg_ss, 5)) if neg_ss.size else -1.0
    n01 = float(np.percentile(neg_ss, 1)) if neg_ss.size else -1.0
    z_lo_rec = 0.5
    z_hi_sat = max(round(p99 * 1.05, 2), z_lo_rec + 0.5)
    z_hi_sparse = max(round(peak.max() * 2.5, 0), 10.0)
    print(f"      z range: full [{z_p.min():.2f}, {z_p.max():.2f}]; "
          f"steady-state t>=50 [{z_ss.min():.2f}, {z_ss.max():.2f}]")
    print(f"      peak z:   frames 0..10 max={peak[:10].max():.2f}, "
          f"frames 10..50 max={peak[10:50].max():.2f}, "
          f"t>=50 mean={peak[50:].mean():.2f}")
    print(f"      trough z: frames 0..10 min={trough[:10].min():.2f}, "
          f"frames 10..50 min={trough[10:50].min():.2f}, "
          f"t>=50 mean={trough[50:].mean():.2f}")
    print(f"      steady-state z>0 percentiles: 95={p95:.2f}  99={p99:.2f}")
    print(f"      steady-state z<0 percentiles:  5={n05:.2f}   1={n01:.2f}")
    print(f"      current LUT:    --z_lo={args.z_lo} --z_hi={args.z_hi} "
          f"--alpha={args.alpha}")
    print(f"      saturated bump: --z_lo={z_lo_rec} --z_hi={z_hi_sat} "
          f"--alpha=1.0   (full alpha at steady-state peak)")
    print(f"      sparse peaks:   --z_lo=0    --z_hi={z_hi_sparse:.0f} "
          f"--alpha=1.0   (only strong moments visible, dark elsewhere)")

    if args.ou:
        print(f"[2/4] running OU velocity rollout, n_steps={args.n_steps} "
              f"seed={args.seed}")
    else:
        print(f"[2/4] running constant-omega rollout, "
              f"n_steps={args.n_steps} omega={args.omega_deg}")
    t0 = time.time()
    if args.ou:
        h_traj, theta, omega_trace, decoded_hd = _run_ou_with_traces(
            net, args.n_steps, device, seed=args.seed)
    else:
        h_traj, theta, omega_trace, decoded_hd = _run_const_with_traces(
            net, args.n_steps, float(net.dt),
            args.omega_deg, args.theta0, device)
    # Match Fig 9 (fig_kinographs_const_omega.py): per-neuron z-score of
    # the subthreshold state h over the rollout, displayed in [-3, 3].
    # Here we map z > 0 to green alpha (z=3 -> saturated).
    mu = h_traj.mean(axis=0, keepdims=True)
    sd = h_traj.std (axis=0, keepdims=True) + 1e-6
    z  = (h_traj - mu) / sd                       # (T, N)

    # Window z in [z_lo, z_hi] -> alpha in [0, 1].
    rng = max(args.z_hi - args.z_lo, 1e-6)
    rates_lit = np.clip((z - args.z_lo) / rng, 0.0, 1.0)

    # Optional reorder: permute the per-neuron voltage vector before it's
    # mapped onto the 3D skeletons. Uses the same sorts as Fig 9.
    perm = None
    if args.reorder != "none":
        neuron_types = np.asarray(net.neuron_types, dtype=np.int64)
        pref_phase = _preferred_phase(h_traj, theta)
        if args.reorder == "type":
            perm = _order_by_type_descending(neuron_types)
        elif args.reorder == "type_phase":
            perm = _order_within_type_by_phase(neuron_types, pref_phase)
        else:  # "phase"
            perm = _order_by_preferred_phase(pref_phase)
        rates_lit = rates_lit[:, perm]
        print(f"      reorder={args.reorder}: applied permutation to "
              f"rates_lit ({rates_lit.shape[1]} neurons)")

    # Slow-motion: linearly interpolate the first slow_init original frames
    # to slow_init * slow_motion output frames. The z-score baseline is
    # still computed over the full rollout (so the normalisation is the
    # same as the regular animation -- we're only slowing the playback).
    if args.slow_motion > 1:
        n_orig = min(args.slow_init, rates_lit.shape[0])
        n_target = n_orig * args.slow_motion
        x_old = np.arange(n_orig)
        x_new = np.linspace(0.0, n_orig - 1, n_target)
        rates_slow = np.empty((n_target, rates_lit.shape[1]),
                              dtype=np.float32)
        for i in range(rates_lit.shape[1]):
            rates_slow[:, i] = np.interp(x_new, x_old,
                                          rates_lit[:n_orig, i])
        theta_slow = np.interp(x_new, x_old, theta[:n_orig])
        rates_lit = rates_slow
        theta = theta_slow
        print(f"      slow-motion: first {n_orig} frames -> {n_target} "
              f"(x{args.slow_motion})")
    print(f"      done ({time.time() - t0:.1f}s); "
          f"z range = [{z.min():.2f}, {z.max():.2f}]; "
          f"lit median {np.median(rates_lit):.3f}, "
          f"frac > 0.5: {float((rates_lit > 0.5).mean()):.3f}")

    # --- init-montage extra rollouts -----------------------------------
    rates_per_init = None
    theta_per_init = None
    init_thetas_deg = None
    if args.init_montage:
        init_thetas_deg = [float(s) for s in args.init_thetas.split(",")]
        init_thetas_rad = [math.radians(t) for t in init_thetas_deg]
        rates_per_init = []
        theta_per_init = []
        for t0_deg, t0_rad in zip(init_thetas_deg, init_thetas_rad):
            print(f"      init rollout theta0={t0_deg:+.0f} deg ...")
            h_i, theta_i = _run_const(net, args.n_steps, float(net.dt),
                                       args.omega_deg, t0_rad, device)
            mu_i = h_i.mean(axis=0, keepdims=True)
            sd_i = h_i.std (axis=0, keepdims=True) + 1e-6
            z_i  = (h_i - mu_i) / sd_i
            r_i  = np.clip((z_i - args.z_lo) / rng, 0.0, 1.0)
            if perm is not None:
                r_i = r_i[:, perm]
            if args.slow_motion > 1:
                n_orig = min(args.slow_init, r_i.shape[0])
                n_target = n_orig * args.slow_motion
                x_old = np.arange(n_orig)
                x_new = np.linspace(0.0, n_orig - 1, n_target)
                r_slow = np.empty((n_target, r_i.shape[1]), dtype=np.float32)
                for j in range(r_i.shape[1]):
                    r_slow[:, j] = np.interp(x_new, x_old, r_i[:n_orig, j])
                t_slow = np.interp(x_new, x_old, theta_i[:n_orig])
                r_i, theta_i = r_slow, t_slow
            rates_per_init.append(r_i)
            theta_per_init.append(theta_i)
        # Render length comes from these arrays now.
        rates_lit = rates_per_init[0]   # placeholder; loop uses lists below
        theta = theta_per_init[0]

    print(f"[3/4] loading skeletons + meshes (downsample={args.downsample}) ...")
    t0 = time.time()
    body_ids = _model_index_to_bodyid(args.datapath)
    assert len(body_ids) == rates_lit.shape[1], (len(body_ids), rates_lit.shape)
    neurons, types_str = _load_skeletons_in_model_order(
        args.anatomy_dir, body_ids, downsample=args.downsample,
    )
    rois = _load_rois(args.anatomy_dir)
    seg_arrays, seg_owner, all_segs = _extract_per_neuron_segments(neurons)
    soma_xyz, soma_r = _extract_soma_positions(neurons)
    print(f"      done ({time.time() - t0:.1f}s); "
          f"{all_segs.shape[0]:,} skeleton segments; "
          f"soma radius median = {float(np.median(soma_r)):.1f}")

    # Project segments + soma centres once (camera doesn't move).
    segs2d = _project_2d(all_segs.reshape(-1, 3),
                          args.elev, args.azim).reshape(-1, 2, 2)
    soma_2d = _project_2d(soma_xyz, args.elev, args.azim)

    # Mesh outline silhouette
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
    if mesh_segs:
        mesh_segs3d = np.array(mesh_segs)
        mesh_segs2d = _project_2d(mesh_segs3d.reshape(-1, 3),
                                   args.elev, args.azim).reshape(-1, 2, 2)
    else:
        mesh_segs2d = None

    # Frame-invariant view limits: union of every projected point + margin
    pts = np.concatenate(
        [segs2d.reshape(-1, 2)] +
        ([mesh_segs2d.reshape(-1, 2)] if mesh_segs2d is not None else []),
        axis=0,
    )
    pad = 0.04 * (pts.max(0) - pts.min(0))
    xlim = (pts[:, 0].min() - pad[0], pts[:, 0].max() + pad[0])
    ylim = (pts[:, 1].min() - pad[1], pts[:, 1].max() + pad[1])

    # ── trace data for the strip-chart ──────────────────────────────────
    n_total = rates_lit.shape[0]
    dt_val = float(net.dt)
    t_sec = np.arange(n_total) * dt_val
    trace_data = {
        "omega": omega_trace[:n_total],
        "theta": theta[:n_total],
        "decoded_hd": decoded_hd[:n_total],
        "dt": dt_val,
        "t_sec": t_sec,
        "scroll_window": args.scroll_window,
    }

    # Render loop
    print(f"[4/4] rendering frames into {args.out_dir}/")
    n_render = n_total
    frame_ids = list(range(0, n_render, args.stride))
    if args.max_frames is not None:
        frame_ids = frame_ids[:args.max_frames]

    fig, ax = None, None
    render_times = []
    # Periodically close + recreate the figure to dodge any matplotlib
    # memory creep over a long render (some artist caches don't release
    # between ax.clear() calls).
    fig_reset_every = 250
    pbar = tqdm(frame_ids, desc="rendering", unit="frame", ncols=150)
    for k, t in enumerate(pbar):
        if k > 0 and k % fig_reset_every == 0 and fig is not None:
            plt.close(fig)
            fig, ax = None, None
            import gc; gc.collect()
        tic = time.time()
        out = os.path.join(args.out_dir, f"frame_{t:04d}.png")
        if args.init_montage:
            fig, ax = _render_init_montage_frame(
                out, segs2d, seg_owner,
                rates_per_init, theta_per_init, init_thetas_deg,
                mesh_segs2d=mesh_segs2d, soma_2d=soma_2d,
                xlim=xlim, ylim=ylim, frame_idx=t,
                alpha_max=args.alpha,
                fig_ref=fig, axes_ref=ax,
            )
        elif args.montage:
            fig, ax = _render_montage_frame(
                out, segs2d, seg_owner, types_str, rates_lit[t],
                mesh_segs2d=mesh_segs2d, soma_2d=soma_2d,
                xlim=xlim, ylim=ylim,
                frame_idx=t, hd_deg=float(np.rad2deg(theta[t])),
                alpha_max=args.alpha,
                fig_ref=fig, axes_ref=ax,
            )
        else:
            fig, ax = _render_frame(
                out, segs2d, seg_owner, rates_lit[t],
                mesh_segs2d=mesh_segs2d, soma_2d=soma_2d,
                xlim=xlim, ylim=ylim,
                frame_idx=t, total_frames=args.n_steps,
                hd_deg=float(np.rad2deg(theta[t])),
                alpha_max=args.alpha,
                fig_ref=fig, ax_ref=ax,
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
    
# python figures/drosophila_cx/fig_cx_anatomy_3d_voltage_anim.py --z_lo 0.5 --z_hi 1.6 --alpha 1.0 --reorder type_phase
# python figures/drosophila_cx/fig_cx_anatomy_3d_voltage_anim.py   --montage   --out_dir figures/drosophila_cx/3D_epg_montage   --z_lo 0 --z_hi 20 --alpha 1.0

# # Known-ODE RNN (default)
# python figures/drosophila_cx/fig_cx_anatomy_3d_voltage_anim.py \
#   --model drosophila_cx_pi_epg \
#   --z_lo 0 --z_hi 20 --alpha 1.0
# # → figures/drosophila_cx/3D_epg/

# # Fully-connected RNN
# python figures/drosophila_cx/fig_cx_anatomy_3d_voltage_anim.py \
#   --model drosophila_cx_pi_fc_epg \
#   --z_lo 0 --z_hi 20 --alpha 1.0
# # → figures/drosophila_cx/3D_fc_epg/

# # Frozen-Wrec control
# python figures/drosophila_cx/fig_cx_anatomy_3d_voltage_anim.py \
#   --model drosophila_cx_pi_frozen_Wrec_epg \
#   --z_lo 0 --z_hi 20 --alpha 1.0
# # → figures/drosophila_cx/3D_frozen_Wrec_epg/

# # GNN
# python figures/drosophila_cx/fig_cx_anatomy_3d_voltage_anim.py \
#   --model drosophila_cx_pi_gnn_epg \
#   --z_lo 0 --z_hi 20 --alpha 1.0
# # → figures/drosophila_cx/3D_gnn_epg/

# python figures/drosophila_cx/fig_cx_anatomy_3d_voltage_anim.py   --model drosophila_cx_pi_epg_tv --z_lo 0 --z_hi 20 --alpha 1.0 --n_steps 4000 --ou --out_dir figures/drosophila_cx/3D_epg_tv_ou