"""Static 3-panel figure: somas of every photoreceptor (R1-R8) in the
Janelia optic-lobe neuprint dataset, no skeletons, no model mapping.

Inputs:
    papers/optic_lobe_anatomy/optic_lobe_full/skeletons/R*.swc
    (any file whose name starts with 'R' is a photoreceptor)

Output:
    figures/flyvis/fig_retina_somas.png
    figures/flyvis/fig_retina_somas.html (interactive plotly, optional)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "drosophila_cx"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from fig_cx_anatomy_3d import _project_2d  # type: ignore


# One distinct colour per R-type. R1-R6 all get one colour (Janelia
# groups them); R7 and R8 subdivide into y/p/d/unclear sub-types.
R_COLORS = {
    "R1-R6":        (0.85, 0.18, 0.18),     # red
    "R7d":          (0.95, 0.55, 0.10),     # orange-bright
    "R7p":          (0.95, 0.75, 0.10),     # yellow
    "R7y":          (0.65, 0.30, 0.85),     # purple
    "R7_unclear":   (0.45, 0.45, 0.45),
    "R7R8_unclear": (0.55, 0.55, 0.55),
    "R8d":          (0.10, 0.55, 0.95),     # cyan-blue
    "R8p":          (0.10, 0.80, 0.45),     # teal
    "R8y":          (0.15, 0.45, 0.95),     # blue
    "R8_unclear":   (0.55, 0.55, 0.55),
}


LAMINA_TYPES = {"L1", "L2", "L3", "L4", "L5",
                 "Lawf1", "Lawf2", "C2", "C3", "Am1"}


def _load_somas_by_glob(anatomy_dir, patterns, with_segments=False,
                         neuron_stride=1):
    """Load somas for SWCs matching any of the given filename prefixes
    (e.g. ['R', 'L', 'C2', 'C3', 'Am1', 'Lawf']).

    Returns (xyz, type) by default. If `with_segments=True`, also returns
    (seg_xyz, seg_type) where seg_xyz has shape (M, 2, 3) — one row per
    (parent, child) skeleton segment. Soma loading is unaffected by
    `neuron_stride`; segment extraction keeps every `neuron_stride`-th
    neuron's *full* tree so each rendered skeleton stays connected."""
    swc_paths = []
    for pat in patterns:
        swc_paths.extend(glob.glob(
            os.path.join(anatomy_dir, "skeletons", f"{pat}*.swc")
        ))
    swc_paths = sorted(set(swc_paths))
    if not swc_paths:
        sys.exit(f"no matching SWCs under {anatomy_dir}/skeletons/")

    soma_xyz = []
    soma_type = []
    seg_chunks = []
    seg_chunk_type = []
    t0 = time.time()
    for k, path in enumerate(swc_paths):
        try:
            df = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                             names=["nid", "label", "x", "y", "z",
                                    "radius", "pid"])
        except Exception:
            continue
        if len(df) == 0:
            continue
        row = df.loc[df.radius.idxmax()]
        soma_xyz.append([float(row.x), float(row.y), float(row.z)])
        stem = os.path.splitext(os.path.basename(path))[0]
        ntype = stem.rpartition("__")[0]
        soma_type.append(ntype)
        keep_skeleton = (with_segments and len(df) > 1
                          and (neuron_stride <= 1 or k % neuron_stride == 0))
        if keep_skeleton:
            child = df[df.pid != -1]
            if len(child) > 0:
                pid_to_xyz = df.set_index("nid")[["x", "y", "z"]]
                mask = child.pid.isin(pid_to_xyz.index)
                child = child[mask]
                if len(child) > 0:
                    pxyz = pid_to_xyz.loc[child.pid.values].values
                    cxyz = child[["x", "y", "z"]].values
                    segs = np.stack([pxyz, cxyz], axis=1).astype(np.float32)
                    seg_chunks.append(segs)
                    seg_chunk_type.append(np.full(len(segs), ntype))
        if k % 2000 == 0:
            print(f"  loaded {k+1}/{len(swc_paths)}", flush=True)

    soma_xyz = np.asarray(soma_xyz, dtype=np.float32)
    soma_type = np.asarray(soma_type)
    if not with_segments:
        return soma_xyz, soma_type
    if seg_chunks:
        seg_xyz = np.concatenate(seg_chunks, axis=0)
        seg_type = np.concatenate(seg_chunk_type, axis=0)
    else:
        seg_xyz = np.zeros((0, 2, 3), dtype=np.float32)
        seg_type = np.zeros((0,), dtype=object)
    return soma_xyz, soma_type, seg_xyz, seg_type


def _load_r_somas(anatomy_dir):
    """Walk all R*.swc files, extract one soma (largest-radius node) per."""
    swc_paths = sorted(glob.glob(os.path.join(anatomy_dir,
                                               "skeletons", "R*.swc")))
    if not swc_paths:
        sys.exit(f"no R*.swc under {anatomy_dir}/skeletons/")

    soma_xyz = []
    soma_type = []
    t0 = time.time()
    for k, path in enumerate(swc_paths):
        try:
            df = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                             names=["nid", "label", "x", "y", "z",
                                    "radius", "pid"])
        except Exception:
            continue
        if len(df) == 0:
            continue
        row = df.loc[df.radius.idxmax()]
        soma_xyz.append([float(row.x), float(row.y), float(row.z)])
        stem = os.path.splitext(os.path.basename(path))[0]
        soma_type.append(stem.rpartition("__")[0])
        if k % 1000 == 0:
            print(f"  loaded {k+1}/{len(swc_paths)} "
                  f"({(k+1)/max(time.time()-t0, 1e-6):.0f}/s)", flush=True)
    return np.asarray(soma_xyz, dtype=np.float32), np.asarray(soma_type)


def _render_3panel(out_path, soma_xyz, soma_type,
                    views, marker_size=4.0,
                    bg="black"):
    """Three side-by-side panels at the requested (elev, azim) angles."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.5),
                              facecolor=bg, squeeze=False)
    axes = list(axes[0])
    txt_color = "white" if bg == "black" else "black"

    type_order = list(R_COLORS.keys())
    type_to_color = {t: R_COLORS.get(t, (0.6, 0.6, 0.6)) for t in type_order}

    for ax, (elev, azim, name) in zip(axes, views):
        ax.set_facecolor(bg)
        pts2d = _project_2d(soma_xyz, elev, azim)
        for t in type_order:
            mask = soma_type == t
            if not mask.any():
                continue
            ax.scatter(pts2d[mask, 0], pts2d[mask, 1],
                       s=marker_size, c=[type_to_color[t]],
                       edgecolors="none", alpha=0.85,
                       label=f"{t} (n={int(mask.sum())})")
        ax.set_aspect("equal")
        ax.set_axis_off()
        # tight bbox
        lo = np.percentile(pts2d, 0.5, axis=0)
        hi = np.percentile(pts2d, 99.5, axis=0)
        pad = 0.08 * (hi - lo)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.text(0.02, 0.97, name, color=txt_color, fontsize=11,
                family="monospace", ha="left", va="top",
                transform=ax.transAxes)
        ax.text(0.02, 0.03,
                f"elev={elev:+.1f}  azim={azim:+.1f}",
                color=txt_color, fontsize=8, family="monospace",
                ha="left", va="bottom", transform=ax.transAxes)

    # Single legend (from the last axes) at the bottom of the figure.
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        leg = fig.legend(handles, labels, loc="lower center",
                         ncol=min(len(handles), 5),
                         fontsize=8, frameon=False,
                         bbox_to_anchor=(0.5, 0.0))
        for txt in leg.get_texts():
            txt.set_color(txt_color)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.10,
                        wspace=0.02)
    fig.savefig(out_path, dpi=200, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _render_plotly(out_path, soma_xyz, soma_type, marker_size=2.5):
    """Interactive 3-D scatter, one trace per R-type, turntable rotation
    (no roll) and a live elev/azim overlay."""
    import plotly.graph_objects as go
    traces = []
    for t, color in R_COLORS.items():
        mask = soma_type == t
        if not mask.any():
            continue
        rgb = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
        p = soma_xyz[mask]
        traces.append(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
            marker=dict(size=marker_size, color=rgb, opacity=0.95),
            name=f"{t} (n={int(mask.sum())})",
        ))
    hidden_axis = dict(visible=False, showbackground=False,
                        showgrid=False, zeroline=False,
                        showticklabels=False, title="")
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            dragmode="turntable",
            camera=dict(up=dict(x=0, y=0, z=1)),
            xaxis=hidden_axis, yaxis=hidden_axis, zaxis=hidden_axis,
            bgcolor="black",
        ),
        paper_bgcolor="black",
        legend=dict(font=dict(color="white")),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    from fig_cx_anatomy_3d import _inject_mpl_angle_readout
    _inject_mpl_angle_readout(out_path)
    print(f"wrote {out_path}")


def _render_3panel_custom(out_path, soma_xyz, group, color_map,
                           views, marker_size=4.0, bg="black",
                           seg_xyz=None, seg_group=None,
                           seg_linewidth=0.18, seg_alpha=0.35):
    """Static 3-panel with an arbitrary group string per soma. If
    `seg_xyz` (shape (M, 2, 3)) and `seg_group` are provided, skeleton
    segments are drawn beneath the somas using the same per-group
    color_map."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.5),
                              facecolor=bg, squeeze=False)
    axes = list(axes[0])
    txt_color = "white" if bg == "black" else "black"

    for ax, (elev, azim, name) in zip(axes, views):
        ax.set_facecolor(bg)
        # skeleton lines first so somas land on top
        if seg_xyz is not None and len(seg_xyz) > 0 and seg_group is not None:
            flat2d = _project_2d(seg_xyz.reshape(-1, 3), elev, azim)
            segs2d = flat2d.reshape(-1, 2, 2)
            for g, c in color_map.items():
                mask = seg_group == g
                if not mask.any():
                    continue
                ax.add_collection(LineCollection(
                    segs2d[mask], colors=[c], linewidths=seg_linewidth,
                    alpha=seg_alpha))
        pts2d = _project_2d(soma_xyz, elev, azim)
        for g, c in color_map.items():
            mask = group == g
            if not mask.any():
                continue
            ax.scatter(pts2d[mask, 0], pts2d[mask, 1],
                       s=marker_size, c=[c], edgecolors="none",
                       alpha=0.85, label=f"{g} (n={int(mask.sum())})")
        ax.set_aspect("equal"); ax.set_axis_off()
        lo = np.percentile(pts2d, 0.5, axis=0)
        hi = np.percentile(pts2d, 99.5, axis=0)
        pad = 0.08 * (hi - lo)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.text(0.02, 0.97, name, color=txt_color, fontsize=11,
                family="monospace", ha="left", va="top",
                transform=ax.transAxes)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        leg = fig.legend(handles, labels, loc="lower center",
                         ncol=len(handles), fontsize=9, frameon=False,
                         bbox_to_anchor=(0.5, 0.0))
        for txt in leg.get_texts():
            txt.set_color(txt_color)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.10,
                        wspace=0.02)
    fig.savefig(out_path, dpi=200, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _render_plotly_custom(out_path, soma_xyz, group, color_map,
                           marker_size=2.5, seg_xyz=None, seg_group=None,
                           seg_width=1.0, seg_opacity=0.35):
    import plotly.graph_objects as go
    traces = []
    # skeleton lines as a single trace per group, joined via NaN separators
    if seg_xyz is not None and len(seg_xyz) > 0 and seg_group is not None:
        for g, c in color_map.items():
            mask = seg_group == g
            if not mask.any():
                continue
            rgb = f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
            segs = seg_xyz[mask]  # (Mg, 2, 3)
            nan_row = np.full((segs.shape[0], 1, 3), np.nan,
                               dtype=segs.dtype)
            flat = np.concatenate([segs, nan_row], axis=1).reshape(-1, 3)
            traces.append(go.Scatter3d(
                x=flat[:, 0], y=flat[:, 1], z=flat[:, 2],
                mode="lines",
                line=dict(color=rgb, width=seg_width),
                opacity=seg_opacity,
                name=f"{g} skeleton",
                hoverinfo="skip",
                showlegend=False,
            ))
    for g, c in color_map.items():
        mask = group == g
        if not mask.any():
            continue
        rgb = f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
        p = soma_xyz[mask]
        traces.append(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
            marker=dict(size=marker_size, color=rgb, opacity=0.95),
            name=f"{g} (n={int(mask.sum())})",
        ))
    hidden_axis = dict(visible=False, showbackground=False,
                        showgrid=False, zeroline=False,
                        showticklabels=False, title="")
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(aspectmode="data", dragmode="turntable",
                   camera=dict(up=dict(x=0, y=0, z=1)),
                   xaxis=hidden_axis, yaxis=hidden_axis, zaxis=hidden_axis,
                   bgcolor="black"),
        paper_bgcolor="black",
        legend=dict(font=dict(color="white")),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    from fig_cx_anatomy_3d import _inject_mpl_angle_readout
    _inject_mpl_angle_readout(out_path)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anatomy_dir",
                   default="papers/optic_lobe_anatomy/optic_lobe_full")
    p.add_argument("--out_dir",
                   default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--marker_size", type=float, default=2.0)
    p.add_argument("--html", action="store_true",
                   help="also write fig_retina_somas.html (interactive)")
    p.add_argument("--mode", default="full",
                   choices=["full", "r16_lamina"],
                   help="full=every R-type (10 colours), "
                        "r16_lamina=R1-R6 red + all lamina yellow.")
    p.add_argument("--neuron_stride", type=int, default=4,
                   help="keep every Nth neuron's full skeleton "
                        "(r16_lamina mode only). Per-neuron striding "
                        "keeps each rendered tree connected — unlike "
                        "per-segment striding, which produces orphan "
                        "fragments.")
    args = p.parse_args()

    if args.mode == "r16_lamina":
        # All R-cells (R1-R6 + all R7/R8 variants) red + lamina yellow
        soma_xyz, soma_type, seg_xyz, seg_type = _load_somas_by_glob(
            args.anatomy_dir,
            patterns=["R", "L1", "L2", "L3", "L4", "L5",
                      "Lawf1", "Lawf2", "C2", "C3", "Am1"],
            with_segments=True,
            neuron_stride=args.neuron_stride,
        )
        print(f"loaded {len(soma_xyz)} somas, "
              f"{len(seg_xyz)} skeleton segments")
        from collections import Counter
        for t, n in Counter(soma_type.tolist()).most_common():
            print(f"  {t:18s} {n}")
        # R-types: anything starting with R; lamina: everything else.
        is_r = np.array([str(t).startswith("R") for t in soma_type])
        group = np.where(is_r, "retina", "lamina")
        is_r_seg = np.array([str(t).startswith("R") for t in seg_type])
        seg_group = np.where(is_r_seg, "retina", "lamina")
        local_colors = {
            "retina": (0.90, 0.20, 0.20),     # red
            "lamina": (0.95, 0.85, 0.10),     # yellow
        }
        out_png = os.path.join(args.out_dir,
                                "fig_retina_r16_lamina.png")
        _render_3panel_custom(out_png, soma_xyz, group,
                               color_map=local_colors,
                               views=[(-0.8, -106.2, "side"),
                                       (78.4,  178.3, "front"),
                                       ( 2.5,  171.9, "top")],
                               marker_size=args.marker_size,
                               seg_xyz=seg_xyz, seg_group=seg_group,
                               seg_linewidth=0.18, seg_alpha=0.55)
        if args.html:
            out_html = os.path.join(args.out_dir,
                                     "fig_retina_r16_lamina.html")
            _render_plotly_custom(out_html, soma_xyz, group,
                                   color_map=local_colors,
                                   marker_size=max(0.6,
                                                    args.marker_size * 0.3),
                                   seg_xyz=seg_xyz, seg_group=seg_group,
                                   seg_width=1.0, seg_opacity=0.55)
        return

    soma_xyz, soma_type = _load_r_somas(args.anatomy_dir)
    print(f"loaded {len(soma_xyz)} photoreceptor somas")
    from collections import Counter
    counts = Counter(soma_type.tolist())
    for t, n in counts.most_common():
        print(f"  {t:18s} {n}")

    views = [
        (-0.8,  -106.2, "side"),
        ( 78.4,  178.3, "front"),
        (  2.5,  171.9, "top"),
    ]

    out = os.path.join(args.out_dir, "fig_retina_somas.png")
    _render_3panel(out, soma_xyz, soma_type, views,
                   marker_size=args.marker_size)
    if args.html:
        out_html = os.path.join(args.out_dir, "fig_retina_somas.html")
        _render_plotly(out_html, soma_xyz, soma_type,
                       marker_size=max(1.5, args.marker_size * 0.6))


if __name__ == "__main__":
    main()
