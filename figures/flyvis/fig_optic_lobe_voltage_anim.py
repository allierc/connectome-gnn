"""Voltage animation on the Janelia optic-lobe skeletons, driven by
the flyvis pretrained model's natural-video rollout.

V1 (per-cell-type mean): we don't yet have a flyvis-(u,v) -> Janelia-
bodyId column mapping, so every Janelia skeleton of cell-type T is
coloured by the *mean activity over flyvis neurons of type T* at each
timestep. Loses retinotopic structure but shows clean per-layer
dynamics (lamina lights up first, medulla cascades, lobula/T4-T5
respond last for motion).

Inputs:
    --activity_zarr  flyvis test-trial activity zarr,
                     shape (T, N_flyvis_neurons, 1).
    --nodes_parquet  flyvis e8_flywireRF nodes.parquet (gives type per
                     flyvis index).
    --anatomy_dir    papers/optic_lobe_anatomy/optic_lobe_full
                     (Janelia SWCs + ROI meshes).

Output:
    figures/flyvis/optic_lobe_3D/frame_NNNN.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "drosophila"))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from fig_cx_anatomy_3d import _project_2d, _load_rois  # type: ignore
from fig_optic_lobe_anatomy_3d import (  # type: ignore
    LAYER_COLORS, _classify, _load_segments_fast,
)

# Map Janelia type -> flyvis types that should provide activity.
# Inverse of FLYVIS_TO_JANELIA in fetch_optic_lobe_anatomy.py.
JANELIA_TO_FLYVIS = {
    "R1-R6": ["R1", "R2", "R3", "R4", "R5", "R6"],
    "R7d": ["R7"], "R7p": ["R7"], "R7y": ["R7"],
    "R7_unclear": ["R7"], "R7R8_unclear": ["R7"],
    "R8d": ["R8"], "R8p": ["R8"], "R8y": ["R8"],
    "R8_unclear": ["R8"],
    "CT1": ["CT1(Lo1)", "CT1(M10)"],
    "Am1": ["Am"],
    "TmY9a": ["TmY9"], "TmY9b": ["TmY9"],
}


def _build_segments_with_owner(anatomy_dir, downsample=30):
    """Like _load_segments_fast but keeps per-SWC ownership so we can
    colour individual skeletons (not just per-layer). Returns:
      all_segs:   (E_total, 2, 3)
      seg_owner:  (E_total,) per-segment index into types/janelia_types/somas
      janelia_types: (N_swc,) array of Janelia cell-type strings per skeleton
      soma_xyz:   (N_swc, 3)
    """
    swc_paths = sorted(glob.glob(os.path.join(anatomy_dir,
                                              "skeletons", "*.swc")))
    if not swc_paths:
        sys.exit(f"no SWCs under {anatomy_dir}/skeletons/")

    seg_list = []
    seg_counts = []
    types_list = []
    soma_list = []
    t0 = time.time()
    for k, path in enumerate(swc_paths):
        try:
            df = pd.read_csv(
                path, sep=r"\s+", comment="#", header=None,
                names=["nid", "label", "x", "y", "z", "r", "pid"],
            )
        except Exception:
            seg_counts.append(0)
            types_list.append("?")
            soma_list.append(np.zeros(3, dtype=np.float32))
            continue
        children = df[df.pid != -1]
        if len(children) > 0:
            try:
                parent_xyz = df.set_index("nid").loc[
                    children.pid.values, ["x", "y", "z"]
                ].values
            except KeyError:
                parent_xyz = np.zeros((0, 3), dtype=np.float32)
            child_xyz = children[["x", "y", "z"]].values
            segs = np.stack([parent_xyz, child_xyz], axis=1).astype(np.float32)
            if downsample > 1 and len(segs) > downsample:
                segs = segs[::downsample]
        else:
            segs = np.zeros((0, 2, 3), dtype=np.float32)
        seg_list.append(segs)
        seg_counts.append(len(segs))

        # Janelia type from filename: <type>__<bodyId>.swc
        stem = os.path.splitext(os.path.basename(path))[0]
        safe_type = stem.rpartition("__")[0]
        # de-sanitize R7_unclear -> R7_unclear (no parens to recover); just
        # keep the filename-safe form, lookup table handles it.
        types_list.append(safe_type)

        soma_row = df.loc[df.r.idxmax()] if len(df) else None
        soma_list.append(
            np.array([soma_row.x, soma_row.y, soma_row.z], dtype=np.float32)
            if soma_row is not None else np.zeros(3, dtype=np.float32)
        )

        if k % 5000 == 0 or k == len(swc_paths) - 1:
            print(f"  loaded {k+1}/{len(swc_paths)}  "
                  f"({(k+1)/(time.time()-t0):.0f}/s)", flush=True)

    all_segs = np.concatenate(seg_list, axis=0) if seg_list \
        else np.zeros((0, 2, 3), dtype=np.float32)
    seg_owner = np.repeat(np.arange(len(swc_paths)),
                          np.array(seg_counts, dtype=int))
    return (
        all_segs,
        seg_owner,
        np.asarray(types_list),
        np.stack(soma_list, axis=0),
    )


def _safe_to_janelia(safe: str) -> str:
    """Map filename-safe Janelia type (no parens) back to the canonical
    form used in JANELIA_TO_FLYVIS keys. e.g. PEN_a_PEN1 -> PEN_a(PEN1).
    Hemibrain optic-lobe names don't use parens (R7d, R1-R6, etc) so most
    types pass through unchanged."""
    return safe


def _build_uv_mapping(flyvis_df, jan_types, jan_soma_xyz, verbose=True):
    """Per cell-type, project Janelia somas onto their best-fit 2D plane
    (via PCA) and match each flyvis (type, u, v) neuron to the nearest
    Janelia soma in that plane.

    Sign ambiguity of the PCA axes is resolved by trying all 4 sign
    combinations and picking the one that minimises the sum of nearest-
    neighbour distances.

    Returns:
        flyvis_to_jan: (N_flyvis,) int array; -1 if no Janelia match.
        coverage: dict[type] -> match-rate
    """
    from scipy.spatial import cKDTree
    flyvis_to_jan = np.full(len(flyvis_df), -1, dtype=int)
    coverage = {}

    ftypes = flyvis_df.type.values
    uv = flyvis_df[["u", "v"]].values.astype(np.float32)

    for ftype in np.unique(ftypes):
        f_mask = ftypes == ftype
        f_idx = np.where(f_mask)[0]
        f_uv = uv[f_mask]

        # Janelia types that hold activity for this flyvis type
        jan_keys = JANELIA_TO_FLYVIS.get(ftype, None)
        # Inverse: which Janelia type-strings (filename-safe) should we look
        # at for this flyvis type? Build by inverting JANELIA_TO_FLYVIS plus
        # the direct-match case.
        jan_keys_for_ftype = []
        for jt, fts in JANELIA_TO_FLYVIS.items():
            if ftype in fts:
                jan_keys_for_ftype.append(jt)
        if not jan_keys_for_ftype:
            jan_keys_for_ftype = [ftype]
        # Filename-safe form: replace '(' / ')'
        safe = [k.replace("(", "_").replace(")", "") for k in jan_keys_for_ftype]
        j_mask = np.isin(jan_types, safe)
        j_idx = np.where(j_mask)[0]
        if len(j_idx) == 0:
            coverage[ftype] = 0.0
            continue
        j_pos = jan_soma_xyz[j_idx]

        if len(j_pos) < 3:
            # Too few -- everyone maps to the first
            flyvis_to_jan[f_idx] = j_idx[0]
            coverage[ftype] = 1.0
            continue

        # PCA via SVD
        j_centered = j_pos - j_pos.mean(axis=0)
        _, _, vh = np.linalg.svd(j_centered, full_matrices=False)
        j_2d = j_centered @ vh[:2].T          # (N_jan, 2)

        # Normalise both to [-1, 1]
        def _norm(p):
            p = p - p.mean(axis=0)
            scale = max(np.abs(p).max(), 1e-6)
            return p / scale

        f_n = _norm(f_uv)
        j_n_base = _norm(j_2d)

        # Try the four sign combinations of the two PCs; also try the
        # 90/180/270 rotations because PCA gives an unsigned frame.
        best_dist = np.inf
        best_nn = None
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                j_n = j_n_base * np.array([sx, sy])
                tree = cKDTree(j_n)
                d, nn = tree.query(f_n)
                if d.sum() < best_dist:
                    best_dist = d.sum()
                    best_nn = nn
        flyvis_to_jan[f_idx] = j_idx[best_nn]
        coverage[ftype] = 1.0 - (best_nn < 0).mean()

        if verbose:
            print(f"  {ftype:14s} flyvis={len(f_idx):4d}  "
                  f"jan_pool={len(j_idx):4d}  "
                  f"mean_nn_dist={best_dist/len(f_idx):.3f}", flush=True)

    return flyvis_to_jan, coverage


def _compute_per_type_activity(activity, flyvis_types, normalize="z_time"):
    """activity: (T, N_flyvis, 1) -> dict[type] -> (T,) z-scored over time.
    """
    a = activity[..., 0]  # (T, N)
    out = {}
    for t_name in np.unique(flyvis_types):
        idx = np.where(flyvis_types == t_name)[0]
        mean_act = a[:, idx].mean(axis=1)
        if normalize == "z_time":
            mu = mean_act.mean()
            sd = mean_act.std() + 1e-6
            out[t_name] = (mean_act - mu) / sd
        else:
            out[t_name] = mean_act
    return out


def _draw_panel(ax, segs2d, soma_2d, per_seg_alpha, per_neuron_alpha,
                 per_seg_color, per_neuron_color,
                 mesh_segs2d=None, xlim=None, ylim=None,
                 bg="black", linewidth=0.25, soma_size=0.5):
    ax.set_facecolor(bg)
    mesh_color = "0.85" if bg == "black" else "0.45"

    if mesh_segs2d is not None and len(mesh_segs2d):
        ax.add_collection(LineCollection(
            mesh_segs2d, colors=(mesh_color,),
            linewidths=0.25, alpha=0.10,
        ))
    ax.add_collection(LineCollection(
        segs2d, colors=[(0.22, 0.22, 0.22)],
        linewidths=linewidth, alpha=0.5,
    ))
    keep = per_seg_alpha > 0.02
    if keep.any():
        rgba = np.zeros((int(keep.sum()), 4), dtype=np.float32)
        rgba[:, :3] = per_seg_color[keep]
        rgba[:, 3] = per_seg_alpha[keep]
        ax.add_collection(LineCollection(
            segs2d[keep], colors=rgba, linewidths=linewidth * 3.0,
        ))
    ax.scatter(soma_2d[:, 0], soma_2d[:, 1],
               s=soma_size * 0.5, c=[(0.22, 0.22, 0.22)],
               edgecolors="none", alpha=0.7, zorder=4)
    keep_n = per_neuron_alpha > 0.02
    if keep_n.any():
        rgba_s = np.zeros((int(keep_n.sum()), 4), dtype=np.float32)
        rgba_s[:, :3] = per_neuron_color[keep_n]
        rgba_s[:, 3] = per_neuron_alpha[keep_n]
        # Bump lit-soma marker so it pops over the dim base layer
        ax.scatter(soma_2d[keep_n, 0], soma_2d[keep_n, 1],
                   s=soma_size * 3.0, c=rgba_s, edgecolors="none",
                   zorder=5)
    if xlim is not None:
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    else:
        ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()


def _render_frame(out_path, panels,
                   frame_idx=None, t_sec=None,
                   bg="black", linewidth=0.25, soma_size=0.5,
                   fig_ref=None, ax_refs=None,
                   inset_stim=None, inset_pos=None, inset_ax_ref=None,
                   screen_2d_per_panel=None, screen_stim=None,
                   screen_marker_size=40.0):
    """`panels` is a list of dicts, each with keys:
       segs2d, soma_2d, per_seg_alpha, per_neuron_alpha,
       per_seg_color, per_neuron_color, mesh_segs2d, xlim, ylim, title.
    Renders one subplot per panel."""
    n_panels = len(panels)
    if fig_ref is None:
        fig, axes = plt.subplots(1, n_panels,
                                  figsize=(6.5 * n_panels, 6.0),
                                  facecolor=bg, squeeze=False)
        axes = list(axes[0])
    else:
        fig, axes = fig_ref, ax_refs
        for ax in axes:
            ax.clear()
        for txt in list(fig.texts):
            txt.remove()
        # Remove any previous inset axes (anything that isn't one of the
        # main panel axes; the figure-coord inset is added below each frame)
        main_axes = set(id(a) for a in axes)
        for ax in list(fig.axes):
            if id(ax) not in main_axes:
                fig.delaxes(ax)
    txt_color = "white" if bg == "black" else "black"

    for v_idx, (ax, panel) in enumerate(zip(axes, panels)):
        _draw_panel(
            ax, panel["segs2d"], panel["soma_2d"],
            panel["per_seg_alpha"], panel["per_neuron_alpha"],
            panel["per_seg_color"], panel["per_neuron_color"],
            mesh_segs2d=panel.get("mesh_segs2d"),
            xlim=panel.get("xlim"), ylim=panel.get("ylim"),
            bg=bg, linewidth=linewidth, soma_size=soma_size,
        )
        # Screen draw: any panel whose proj_screen entry is non-None.
        if (screen_2d_per_panel is not None
                and screen_stim is not None
                and v_idx < len(screen_2d_per_panel)
                and screen_2d_per_panel[v_idx] is not None):
            s2d = screen_2d_per_panel[v_idx]
            ax.scatter(s2d[:, 0], s2d[:, 1], c=screen_stim,
                       cmap="viridis", marker="h", s=screen_marker_size,
                       vmin=0.0, vmax=1.05, linewidths=0,
                       zorder=6, alpha=0.95)
        # Retina-dot overlay (only the panel that opted-in via panel keys)
        r_all_pos = panel.get("r_all_soma_2d")
        if r_all_pos is not None:
            ax.scatter(r_all_pos[:, 0], r_all_pos[:, 1],
                       s=10.0, c=[(0.55, 0.18, 0.18)],
                       marker="o", linewidths=0, zorder=6, alpha=0.7)
        r_soma_pos = panel.get("r_soma_2d")
        r_soma_stim = panel.get("r_soma_stim")
        if r_soma_pos is not None and r_soma_stim is not None:
            ax.scatter(r_soma_pos[:, 0], r_soma_pos[:, 1],
                       c=r_soma_stim, cmap="viridis", marker="o",
                       s=14.0, vmin=0.0, vmax=1.05, linewidths=0,
                       zorder=7, alpha=0.98)
        if panel.get("title"):
            ax.text(0.02, 0.97, panel["title"], color=txt_color, fontsize=10,
                    family="monospace", ha="left", va="top",
                    transform=ax.transAxes)

    # --- stimulus hex inset placed in FIGURE coords (top-right) -------
    if (inset_stim is not None and inset_pos is not None
            and len(axes) >= 1):
        # DEBUG: large, bordered inset so we can confirm it renders
        inset = fig.add_axes([0.78, 0.65, 0.22, 0.30], zorder=20)
        inset.set_facecolor((0.05, 0.05, 0.05))   # near-black, slight contrast
        inset.scatter(inset_pos[:, 0], inset_pos[:, 1],
                      c=inset_stim, cmap="viridis",
                      marker="h", s=20, vmin=0.0, vmax=1.05,
                      linewidths=0)
        inset.set_aspect("equal")
        inset.set_xticks([]); inset.set_yticks([])
        # Bright red border so we can locate it
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color("red")
            spine.set_linewidth(1.5)
        margin = 0.5
        inset.set_xlim(inset_pos[:, 0].min() - margin,
                       inset_pos[:, 0].max() + margin)
        inset.set_ylim(inset_pos[:, 1].min() - margin,
                       inset_pos[:, 1].max() + margin)

    if frame_idx is not None:
        label = f"t = {frame_idx:04d}"
        if t_sec is not None:
            label += f"   ({t_sec:.2f}s)"
        fig.text(0.5, 0.98, label, color=txt_color, fontsize=12,
                 family="monospace", ha="center", va="top")

    # Subplots use full figure width unless we're keeping a side inset.
    right = 0.995
    fig.subplots_adjust(left=0.005, right=right, top=0.96, bottom=0.005,
                        wspace=0.02)
    fig.savefig(out_path, dpi=150, facecolor=bg)
    return fig, axes


def _render_html(out_path, all_segs, jan_types, soma_xyz,
                  screen_pos_3d=None, screen_stim=None,
                  linewidth=0.5, soma_size=1.0,
                  screen_marker_size=8.0):
    """One interactive plotly HTML with skeletons + somas + (optional)
    3-D screen, plus the live mpl-view-angle overlay."""
    import plotly.graph_objects as go

    traces = []
    # One trace per layer for the skeletons (single polyline with NaN breaks)
    for layer in LAYER_COLORS:
        mask = np.array([_classify(_safe_to_janelia(jt)) == layer
                          for jt in jan_types])
        if not mask.any():
            continue
        # Build per-layer segment list: gather from all_segs via seg owner
        # In the streaming loader segs are concatenated per-neuron in order.
        # We need to recover which segments belong to which Janelia. Use the
        # global seg_owner array passed in via closure — but to keep this
        # simple, re-build the index-to-mask.
        pass

    # Simpler approach: just render ALL skeletons in one trace per layer
    # by re-indexing through the seg_owner. We accept the loss of per-
    # neuron color granularity inside a layer for the HTML preview.
    return _render_html_simple(out_path, all_segs, jan_types, soma_xyz,
                                screen_pos_3d=screen_pos_3d,
                                screen_stim=screen_stim,
                                linewidth=linewidth, soma_size=soma_size,
                                screen_marker_size=screen_marker_size)


def _render_html_simple(out_path, all_segs, jan_types, soma_xyz,
                         screen_pos_3d=None, screen_stim=None,
                         linewidth=0.5, soma_size=1.0,
                         screen_marker_size=8.0):
    """Skeletons rendered as a single all-grey polyline (anatomy-only
    preview), somas coloured by layer, plus optional 3-D screen."""
    import plotly.graph_objects as go

    traces = []
    # Single grey polyline for all skeleton edges
    if len(all_segs):
        xs = np.empty(3 * len(all_segs), dtype=np.float32)
        ys = np.empty_like(xs); zs = np.empty_like(xs)
        xs[0::3] = all_segs[:, 0, 0]; xs[1::3] = all_segs[:, 1, 0]; xs[2::3] = np.nan
        ys[0::3] = all_segs[:, 0, 1]; ys[1::3] = all_segs[:, 1, 1]; ys[2::3] = np.nan
        zs[0::3] = all_segs[:, 0, 2]; zs[1::3] = all_segs[:, 1, 2]; zs[2::3] = np.nan
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="rgb(80,80,80)", width=linewidth),
            name="skeletons", opacity=0.5, hoverinfo="skip",
        ))

    # Somas per layer
    for layer, color in LAYER_COLORS.items():
        mask = np.array([_classify(_safe_to_janelia(jt)) == layer
                          for jt in jan_types])
        if not mask.any():
            continue
        rgb = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
        s = soma_xyz[mask]
        traces.append(go.Scatter3d(
            x=s[:, 0], y=s[:, 1], z=s[:, 2], mode="markers",
            marker=dict(size=soma_size, color=rgb, opacity=0.9),
            name=f"{layer} (n={int(mask.sum())})",
        ))

    # 3-D stimulus screen
    if screen_pos_3d is not None and screen_stim is not None:
        # Map stim to viridis RGB
        import matplotlib.cm as cm
        vir = cm.get_cmap("viridis")
        colors = vir(np.clip(screen_stim / 1.05, 0, 1))
        rgbs = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
                for c in colors]
        traces.append(go.Scatter3d(
            x=screen_pos_3d[:, 0], y=screen_pos_3d[:, 1],
            z=screen_pos_3d[:, 2], mode="markers",
            marker=dict(size=screen_marker_size, color=rgbs,
                        symbol="square", opacity=0.95),
            name="stimulus screen",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            dragmode="turntable",          # locks up-axis -> roll stays 0
            camera=dict(up=dict(x=0, y=0, z=1)),
        ),
        paper_bgcolor="black",
        legend=dict(font=dict(color="white")),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    from fig_cx_anatomy_3d import _inject_mpl_angle_readout
    _inject_mpl_angle_readout(out_path)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anatomy_dir",
                   default="papers/optic_lobe_anatomy/optic_lobe_full")
    p.add_argument("--activity_zarr",
                   default="/workspace/flyvis-gnn/graphs_data/fly/"
                           "flyvis_noise_005/x_list_test/voltage.zarr",
                   help="path to a (T, N) or (T, N, 1) voltage zarr. "
                        "x_list_test/voltage.zarr is the flyvis "
                        "membrane-potential output used for training; "
                        "y_list_test.zarr is the pre-nonlinearity drive.")
    p.add_argument("--stimulus_zarr",
                   default="/workspace/flyvis-gnn/graphs_data/fly/"
                           "flyvis_noise_005/x_list_test/stimulus.zarr",
                   help="visual luminance per neuron per frame for the "
                        "hex-grid stimulus inset.")
    p.add_argument("--pos_zarr",
                   default="/workspace/flyvis-gnn/graphs_data/fly/"
                           "flyvis_noise_005/x_list_test/pos.zarr",
                   help="cartesian (x, y) hex-grid coordinates per neuron "
                        "for the stimulus inset.")
    p.add_argument("--no_inset", action="store_true",
                   help="disable the top-right hex-stimulus inset.")
    p.add_argument("--screen", action="store_true",
                   help="render the visual stimulus as a 3-D screen "
                        "floating in front of the eye (in addition to "
                        "/instead of the inset).")
    p.add_argument("--screen_distance", type=float, default=2.5,
                   help="screen distance in eye-radii along the "
                        "photoreceptor-plane normal (default 2.5).")
    p.add_argument("--html", action="store_true",
                   help="render a single interactive plotly HTML "
                        "(skeletons + somas + screen) with live "
                        "matplotlib-angle overlay; skips frame loop.")
    p.add_argument("--html_frame", type=int, default=0,
                   help="frame index to use for the screen stimulus "
                        "in HTML mode (default 0).")
    p.add_argument("--retina_dots", action="store_true",
                   help="overlay R-cell somas (red) + viridis-coloured "
                        "stim-mapped subset on the left panel.")
    p.add_argument("--screen_scale", type=float, default=1.2,
                   help="screen size as a multiple of the eye's planar "
                        "extent (default 1.2).")
    p.add_argument("--screen_marker_size", type=float, default=40.0,
                   help="hex marker size for screen points.")
    p.add_argument("--nodes_parquet",
                   default="/workspace/connectome-gnn-ca/data/"
                           "hybrid_connectomes/e8_flywireRF/nodes.parquet")
    p.add_argument("--n_steps", type=int, default=2000,
                   help="number of frames to render (max = zarr length)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--start_step", type=int, default=0)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--z_lo", type=float, default=0.5)
    p.add_argument("--z_hi", type=float, default=3.0)
    p.add_argument("--elev", type=float, default=-38.0)
    p.add_argument("--azim", type=float, default=-134.3)
    p.add_argument("--two_panel", action="store_true",
                   help="render two-panel (left=full anatomy, "
                        "right=cross-section slice).")
    p.add_argument("--three_panel", action="store_true",
                   help="render three-panel: left=slice cross-section, "
                        "middle=eye-facing with screen, right=skeleton dump.")
    p.add_argument("--right_elev", type=float, default=18.4,
                   help="elev for the right (slice) panel.")
    p.add_argument("--right_azim", type=float, default=-17.6,
                   help="azim for the right (slice) panel.")
    p.add_argument("--third_elev", type=float, default=10.0,
                   help="elev for the third (skeleton-dump) panel.")
    p.add_argument("--third_azim", type=float, default=30.0,
                   help="azim for the third (skeleton-dump) panel.")
    p.add_argument("--slice_axis", default="v", choices=["u", "v"],
                   help="which (u,v) axis to slice along for the "
                        "cross-section panel (default v).")
    p.add_argument("--slice_range", type=float, nargs=2, default=[-1.0, 1.0],
                   help="(u,v) range to keep for the slice (default -1..1).")
    p.add_argument("--downsample", type=int, default=30)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--out_dir",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "optic_lobe_3D"))
    p.add_argument("--dt", type=float, default=1.0/200.0,
                   help="seconds per frame (flyvis default ~5 ms)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/4] loading flyvis activity + types ...")
    t0 = time.time()
    import zarr
    z = zarr.open(args.activity_zarr, mode="r")
    activity = np.asarray(z[args.start_step:
                            args.start_step + args.n_steps])
    if activity.ndim == 2:
        activity = activity[..., None]
    print(f"  activity shape: {activity.shape}  "
          f"(mean={activity.mean():.3f} std={activity.std():.3f})")

    # Load the flyvis parquet here so both the inset/screen setup AND the
    # later uv-mapping have access.
    flyvis_df = pd.read_parquet(args.nodes_parquet)
    flyvis_types = flyvis_df.type.values
    assert len(flyvis_types) == activity.shape[1], (
        len(flyvis_types), activity.shape)

    stim_inset = pos_inset = None
    # Load stimulus + hex positions whenever the inset OR the 3-D screen
    # is enabled (both consume the same per-frame luminance values).
    if (not args.no_inset) or args.screen:
        try:
            stim = np.asarray(zarr.open(args.stimulus_zarr, mode="r")
                              [args.start_step:
                               args.start_step + args.n_steps])
            # Use the parquet's integer (u, v) to compute Cartesian hex
            # coords. pos.zarr only has correct positions for the
            # photoreceptors; non-R types are clustered near origin.
            r_mask = flyvis_df.type.str.match(r"^R[1-8]$").values
            r_df = flyvis_df[r_mask].copy()
            r_df["_idx"] = np.where(r_mask)[0]
            # Take one R-cell per (u, v) column (same luminance across R1-R8)
            r_first = r_df.drop_duplicates(subset=["u", "v"]).sort_values(
                ["u", "v"]).reset_index(drop=True)
            uv = r_first[["u", "v"]].values.astype(np.float32)
            # Hex -> Cartesian
            cart_x = uv[:, 0] + 0.5 * uv[:, 1]
            cart_y = uv[:, 1] * (np.sqrt(3.0) / 2.0)
            pos_inset = np.stack([cart_x, cart_y], axis=1).astype(np.float32)
            stim_inset = stim[:, r_first["_idx"].values].astype(np.float32)
            print(f"  stim inset: {pos_inset.shape[0]} hex cells "
                  f"(one R-cell per column)")
        except Exception as e:
            print(f"  inset disabled: {e}")
            stim_inset = pos_inset = None
    print(f"  done ({time.time()-t0:.1f}s)")

    print("[2/4] loading skeletons (need soma positions before uv mapping)")
    t0 = time.time()
    all_segs, seg_owner, jan_types, soma_xyz = \
        _build_segments_with_owner(args.anatomy_dir, args.downsample)
    rois = _load_rois(args.anatomy_dir)
    print(f"  {len(jan_types)} skeletons, "
          f"{all_segs.shape[0]:,} segments  "
          f"({time.time()-t0:.0f}s)")

    print("[3/4] building flyvis (u, v) -> Janelia bodyId mapping via soma PCA ...")
    t0 = time.time()
    flyvis_to_jan, coverage = _build_uv_mapping(flyvis_df, jan_types, soma_xyz)
    matched = (flyvis_to_jan >= 0).sum()
    print(f"  mapped {matched}/{len(flyvis_to_jan)} flyvis neurons "
          f"({100*matched/len(flyvis_to_jan):.1f}%)  ({time.time()-t0:.1f}s)")

    # Build per-Janelia (u, v): take the (u, v) of the FIRST flyvis index
    # that maps to each Janelia bodyId.
    janelia_uv = np.full((len(jan_types), 2), np.nan, dtype=np.float32)
    flyvis_uv_arr = flyvis_df[["u", "v"]].values.astype(np.float32)
    for f_idx, j_idx in enumerate(flyvis_to_jan):
        if j_idx < 0 or not np.isnan(janelia_uv[j_idx, 0]):
            continue
        janelia_uv[j_idx] = flyvis_uv_arr[f_idx]

    # --- R-cell soma -> stimulus column lookup ---
    # For each Janelia R-cell, find which of the 217 stim columns it
    # corresponds to via its (u, v). Used to paint R-soma with viridis
    # luminance on the left panel.
    r_soma_stim_col = None
    r_soma_mask = None
    if pos_inset is not None:
        r_soma_mask = np.array([jt.startswith("R") for jt in jan_types])
        # Build (u, v) -> column-index dict from r_first
        uv_to_col = {(int(u), int(v)): i for i, (u, v) in enumerate(
            r_first[["u", "v"]].values)}
        r_soma_stim_col = np.full(len(jan_types), -1, dtype=np.int32)
        for j_idx in np.where(r_soma_mask)[0]:
            uv = janelia_uv[j_idx]
            if np.isnan(uv[0]):
                continue
            key = (int(round(float(uv[0]))), int(round(float(uv[1]))))
            if key in uv_to_col:
                r_soma_stim_col[j_idx] = uv_to_col[key]
        n_mapped = int((r_soma_stim_col >= 0).sum())
        print(f"  R-soma stim lookup: {n_mapped}/{int(r_soma_mask.sum())} "
              f"R-cells mapped to stim columns")

    # --- 3D stimulus screen geometry ---
    # PCA on R-cell somas to get the photoreceptor-plane basis. The
    # smallest singular vector is the eye normal; we lay the screen out
    # in 3D along that direction at `screen_distance * eye_radius`.
    screen_pos_3d = None
    if args.screen and pos_inset is not None:
        r_mask = np.array([jt.startswith("R") for jt in jan_types])
        if r_mask.any():
            r_soma = soma_xyz[r_mask]
            r_centroid = r_soma.mean(axis=0)
            _, S_eye, Vt_eye = np.linalg.svd(r_soma - r_centroid,
                                              full_matrices=False)
            basis1, basis2, normal = Vt_eye[0], Vt_eye[1], Vt_eye[2]
            eye_radius = float(S_eye[0]) / np.sqrt(len(r_soma))  # ~RMS
            # Try both signs of the normal and pick the one further from
            # the lobe centroid (so the screen ends up OUTSIDE the brain).
            lobe_centroid = soma_xyz.mean(axis=0)
            sign = (+1.0 if np.dot(normal, r_centroid - lobe_centroid) > 0
                    else -1.0)
            normal = sign * normal
            # Hex grid in screen-plane coords (pos_inset is (n_hex, 2))
            scale = args.screen_scale * eye_radius / 8.0   # hex range ±8
            screen_origin = r_centroid + args.screen_distance * eye_radius * normal
            screen_pos_3d = (screen_origin[None, :]
                             + scale * pos_inset[:, 0:1] * basis1[None, :]
                             + scale * pos_inset[:, 1:2] * basis2[None, :])
            print(f"  screen: {screen_pos_3d.shape[0]} hex pts at "
                  f"{args.screen_distance:.1f}x eye radius "
                  f"({eye_radius*args.screen_distance:.0f} nm)")

    # Slice mask for the cross-section panel
    ax_col = 0 if args.slice_axis == "u" else 1
    slice_lo, slice_hi = args.slice_range
    slice_mask_n = ((janelia_uv[:, ax_col] >= slice_lo) &
                    (janelia_uv[:, ax_col] <= slice_hi))
    print(f"  slice {args.slice_axis} in [{slice_lo}, {slice_hi}]: "
          f"{int(slice_mask_n.sum())}/{len(jan_types)} Janelia neurons")

    # Project views. Three-panel mode orders panels:
    #   [0] left = slice cross-section
    #   [1] middle = eye-facing view + screen
    #   [2] right = skeleton dump (different angle)
    if args.three_panel:
        views = [
            (args.right_elev, args.right_azim, "slice"),
            (args.elev,        args.azim,        "screen"),
            (args.third_elev,  args.third_azim,  "skeleton"),
        ]
    else:
        views = [(args.elev, args.azim, "front")]
        if args.two_panel:
            views.append((args.right_elev, args.right_azim, "side"))

    mesh_segs = []
    for mesh in rois.values():
        try:
            outline = mesh.outline().entities
            for ent in outline:
                pts = mesh.vertices[ent.points]
                mesh_segs.extend([(pts[i], pts[i+1])
                                  for i in range(len(pts)-1)])
        except Exception:
            pass
    mesh_segs_3d = np.array(mesh_segs) if mesh_segs else None

    # Per-segment slice mask (right panel only sees segments whose owning
    # neuron is in the v-slice).
    seg_slice_mask = slice_mask_n[seg_owner]

    proj_segs, proj_soma, proj_mesh, lims = [], [], [], []
    proj_seg_owner = []
    proj_screen = []                          # 2D screen pos per view
    for v_idx, (elev_v, azim_v, _name) in enumerate(views):
        # apply slice mask: two_panel right-most OR three_panel left-most
        is_slice_panel = (
            (args.two_panel and v_idx == 1)
            or (args.three_panel and v_idx == 0)
        )
        if is_slice_panel:
            sel_segs_3d = all_segs[seg_slice_mask]
            sel_owner   = seg_owner[seg_slice_mask]
            sel_soma_3d = soma_xyz[slice_mask_n]
        else:
            sel_segs_3d = all_segs
            sel_owner   = seg_owner
            sel_soma_3d = soma_xyz
        s2d = _project_2d(sel_segs_3d.reshape(-1, 3),
                           elev_v, azim_v).reshape(-1, 2, 2)
        proj_segs.append(s2d)
        proj_seg_owner.append(sel_owner)
        proj_soma.append(_project_2d(sel_soma_3d, elev_v, azim_v))
        proj_mesh.append(
            _project_2d(mesh_segs_3d.reshape(-1, 3),
                         elev_v, azim_v).reshape(-1, 2, 2)
            if mesh_segs_3d is not None else None
        )
        # Screen panel index: middle in three-panel, first otherwise.
        screen_panel_idx = 1 if args.three_panel else 0
        is_screen_panel = (v_idx == screen_panel_idx)
        # Project the 3D stimulus screen but only attach it to the
        # designated screen panel (others get None so it's not drawn).
        if screen_pos_3d is not None and is_screen_panel:
            sc2d = _project_2d(screen_pos_3d, elev_v, azim_v)
            proj_screen.append(sc2d)
        else:
            sc2d = None
            proj_screen.append(None)
        # Skeleton bbox via percentile, screen bbox via exact min/max
        sk_pts = s2d.reshape(-1, 2) if len(s2d) else np.zeros((1, 2))
        sk_lo = np.percentile(sk_pts, 0.1, axis=0)
        sk_hi = np.percentile(sk_pts, 99.9, axis=0)
        if sc2d is not None:
            sc_lo = sc2d.min(axis=0)
            sc_hi = sc2d.max(axis=0)
            lo = np.minimum(sk_lo, sc_lo)
            hi = np.maximum(sk_hi, sc_hi)
        else:
            lo, hi = sk_lo, sk_hi
        pad = 0.12 * (hi - lo)               # 12% pad on each side
        lims.append(((lo[0] - pad[0], hi[0] + pad[0]),
                      (lo[1] - pad[1], hi[1] + pad[1])))

    # Back-compat name (existing code below references segs2d/mesh_segs2d
    # for the single-view bbox computation).
    segs2d = proj_segs[0]
    soma_2d = proj_soma[0]
    mesh_segs2d = proj_mesh[0]

    # xlim/ylim for the legacy single-view path (overridden by `lims` below)
    xlim, ylim = lims[0]

    # Per-NEURON activity via the (u,v) mapping. Each Janelia bodyId gets
    # the z-scored activity of whichever flyvis neuron(s) maps to it.
    # If multiple flyvis neurons map to the same Janelia bodyId (because
    # the Janelia pool is smaller), we average their activities.
    T = activity.shape[0]
    rng = max(args.z_hi - args.z_lo, 1e-6)
    a_raw = activity[..., 0].astype(np.float32)   # (T, N_flyvis)

    # Per-flyvis-neuron z-score over time (Fig 9 normalisation)
    mu = a_raw.mean(axis=0, keepdims=True)
    sd = a_raw.std (axis=0, keepdims=True) + 1e-6
    z_per_flyvis = (a_raw - mu) / sd               # (T, N_flyvis)

    # Per-Janelia z: take the maximum-magnitude assignment per neuron, no
    # averaging (averaging anti-correlated flyvis signals washes out peaks).
    z_per_jan = np.zeros((T, len(jan_types)), dtype=np.float32)
    assigned = np.zeros(len(jan_types), dtype=bool)
    for f_idx, j_idx in enumerate(flyvis_to_jan):
        if j_idx < 0:
            continue
        if not assigned[j_idx]:
            z_per_jan[:, j_idx] = z_per_flyvis[:, f_idx]
            assigned[j_idx] = True

    alpha_per_neuron = np.clip((z_per_jan - args.z_lo) / rng,
                               0.0, 1.0) ** args.gamma   # (T, N_jan)

    # Layer color per Janelia neuron
    type_color = np.zeros((len(jan_types), 3), dtype=np.float32)
    for jt in np.unique(jan_types):
        layer = _classify(_safe_to_janelia(jt))
        type_color[jan_types == jt] = LAYER_COLORS.get(
            layer, LAYER_COLORS["other"]
        )
    per_seg_color = type_color[seg_owner]
    rates_per_neuron = alpha_per_neuron

    if args.html:
        out_html = os.path.join(args.out_dir,
                                 "fig_optic_lobe_voltage_anim.html")
        stim_for_screen = (stim_inset[args.html_frame]
                            if stim_inset is not None else None)
        _render_html_simple(
            out_html, all_segs, jan_types, soma_xyz,
            screen_pos_3d=screen_pos_3d,
            screen_stim=stim_for_screen,
            linewidth=args.linewidth if hasattr(args, "linewidth") else 0.5,
            soma_size=1.0, screen_marker_size=8.0,
        )
        return 0

    print(f"[4/4] rendering frames into {args.out_dir}/")
    frame_ids = list(range(0, T, args.stride))
    if args.max_frames is not None:
        frame_ids = frame_ids[:args.max_frames]

    fig, axes = None, None
    render_times = []
    for k, t in enumerate(frame_ids):
        if k > 0 and k % 250 == 0 and fig is not None:
            plt.close(fig); fig, axes = None, None
            import gc; gc.collect()
        tic = time.time()
        per_neuron_alpha = rates_per_neuron[t]
        per_seg_alpha = per_neuron_alpha[seg_owner]
        out = os.path.join(args.out_dir, f"frame_{t:04d}.png")

        panels = []
        for v_idx, (elev_v, azim_v, name) in enumerate(views):
            owner_v = proj_seg_owner[v_idx]
            alpha_v = per_neuron_alpha[owner_v] if len(owner_v) \
                else np.zeros(0)
            color_v = type_color[owner_v] if len(owner_v) \
                else np.zeros((0, 3))
            # For the slice panel, soma/per-neuron arrays must also be
            # restricted to slice_mask_n.
            is_slice_panel_now = (
                (args.two_panel and v_idx == 1)
                or (args.three_panel and v_idx == 0)
            )
            if is_slice_panel_now:
                nrn_alpha_v = per_neuron_alpha[slice_mask_n]
                nrn_color_v = type_color[slice_mask_n]
                title = f"slice {args.slice_axis}∈[{args.slice_range[0]:.0f},{args.slice_range[1]:.0f}]"
            else:
                nrn_alpha_v = per_neuron_alpha
                nrn_color_v = type_color
                title = (name if (args.two_panel or args.three_panel) else None)
            panel = {
                "segs2d": proj_segs[v_idx],
                "soma_2d": proj_soma[v_idx],
                "per_seg_alpha": alpha_v,
                "per_neuron_alpha": nrn_alpha_v,
                "per_seg_color": color_v,
                "per_neuron_color": nrn_color_v,
                "mesh_segs2d": proj_mesh[v_idx],
                "xlim": lims[v_idx][0],
                "ylim": lims[v_idx][1],
                "title": title,
            }
            # R-cell soma luminance overlay (screen-panel only, opt-in)
            screen_panel_idx = 1 if args.three_panel else 0
            if (args.retina_dots and v_idx == screen_panel_idx
                    and r_soma_mask is not None):
                panel["r_all_soma_2d"] = proj_soma[v_idx][r_soma_mask]
                if r_soma_stim_col is not None and stim_inset is not None:
                    valid = r_soma_stim_col >= 0
                    panel["r_soma_2d"] = proj_soma[v_idx][valid]
                    panel["r_soma_stim"] = stim_inset[t][
                        r_soma_stim_col[valid]]
            panels.append(panel)

        fig, axes = _render_frame(
            out, panels,
            frame_idx=t, t_sec=t * args.dt,
            fig_ref=fig, ax_refs=axes,
            inset_stim=(stim_inset[t]
                         if (not args.no_inset and stim_inset is not None)
                         else None),
            inset_pos=(pos_inset if not args.no_inset else None),
            screen_2d_per_panel=(proj_screen if args.screen else None),
            screen_stim=(stim_inset[t]
                          if (args.screen and stim_inset is not None)
                          else None),
            screen_marker_size=args.screen_marker_size,
        )
        render_times.append(time.time() - tic)
        if k < 3 or k % 50 == 0:
            print(f"  frame {t:04d} -> {out}  "
                  f"({render_times[-1]:.2f}s, mean "
                  f"{np.mean(render_times):.2f}s)", flush=True)

    plt.close(fig)
    print(f"done: {len(frame_ids)} frames, mean "
          f"{np.mean(render_times):.2f}s/frame")


if __name__ == "__main__":
    main()
