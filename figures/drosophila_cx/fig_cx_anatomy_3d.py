"""3-D view of the hemibrain CX neurons studied in drosophila.pdf,
rendered with navis from cached SWCs + neuropil ROI meshes.

Inputs (produced by fetch_cx_anatomy.py, run elsewhere with network):
    <anatomy_dir>/
        skeletons/<type>__<bodyId>.swc
        rois/<ROI>.obj
        index.csv

Default anatomy_dir: papers/janelia_cx/anatomy/cx_anatomy

Outputs:
    figure/fig_cx_anatomy_3d.png            # matplotlib static render
    figure/fig_cx_anatomy_3d.html (optional)# plotly interactive
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
import pandas as pd


# Colour assignment matches docs/figure/fig_hd_mi_summary.py: tab10 indexed
# by neuron_type id in load_drosophila_cx_connectome() order
# (EPG, EPGt, PEN_a(PEN1), PEN_b(PEN2), Delta7, PEG, ER6).
TYPE_ORDER = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "Delta7", "PEG", "ER6"]
TYPE_COLOR = {t: matplotlib.cm.tab10(i)[:3] for i, t in enumerate(TYPE_ORDER)}


def _load_skeletons(anatomy_dir: str, downsample: int = 10):
    """Load all SWCs as a navis.NeuronList, returning (neurons, types).

    Per-neuron type read from the filename prefix (type__bodyId.swc).
    `downsample` is passed to navis.downsample_neuron and preserves
    branch / end points while dropping intermediate nodes -- a factor of
    10 cuts ~1.4M segments down to ~140k with no visible loss in the
    static render.
    """
    import navis
    swc_paths = sorted(glob.glob(os.path.join(anatomy_dir, "skeletons", "*.swc")))
    if not swc_paths:
        sys.exit(f"no SWCs under {anatomy_dir}/skeletons/ -- "
                 "run fetch_cx_anatomy.py first")

    neurons = []
    types = []
    for path in swc_paths:
        n = navis.read_swc(path)
        if downsample and downsample > 1:
            n = navis.downsample_neuron(n, downsampling_factor=downsample,
                                         preserve_nodes=None)
        neurons.append(n)
        # filename: "<safe-type>__<bodyId>.swc"
        stem = os.path.splitext(os.path.basename(path))[0]
        safe_type, _, _ = stem.rpartition("__")
        # de-sanitise: "PEN_a_PEN1" -> "PEN_a(PEN1)"
        for canon in TYPE_ORDER:
            safe = canon.replace("(", "_").replace(")", "")
            if safe_type == safe:
                types.append(canon)
                break
        else:
            types.append(safe_type)

    nl = navis.NeuronList(neurons)
    return nl, np.asarray(types)


def _extract_segments_by_type(nl, types) -> dict:
    """Pre-extract (parent, child) skeleton segments per cell type as
    plain numpy arrays so subsequent re-renders skip the pandas loop."""
    out: dict[str, np.ndarray] = {}
    for t in np.unique(types):
        segs_t = []
        for n, nt in zip(nl, types):
            if nt != t:
                continue
            nodes = n.nodes
            child = nodes[nodes.parent_id != -1]
            if len(child) == 0:
                continue
            parent_xyz = nodes.set_index("node_id").loc[
                child.parent_id.values, ["x", "y", "z"]
            ].values
            child_xyz = child[["x", "y", "z"]].values
            segs_t.append(np.stack([parent_xyz, child_xyz], axis=1))
        if segs_t:
            out[str(t)] = np.concatenate(segs_t, axis=0)
    return out


def _load_rois(anatomy_dir: str):
    """Return dict roi_name -> trimesh.Trimesh."""
    import trimesh
    out = {}
    for path in sorted(glob.glob(os.path.join(anatomy_dir, "rois", "*.obj"))):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            mesh = trimesh.load(path, force="mesh")
        except Exception as e:
            print(f"  skip ROI {name}: {e}")
            continue
        out[name] = mesh
    return out


def _project_2d(xyz: np.ndarray, elev: float, azim: float) -> np.ndarray:
    """Project (N, 3) world coords to (N, 2) screen coords matching the
    matplotlib mplot3d convention for (elev, azim) in degrees. No depth
    sort, no MaskedArray plumbing -- one matmul."""
    e = np.deg2rad(elev)
    a = np.deg2rad(azim)
    ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    R = np.array([[-sa,           ca,         0.0],
                  [-ca * se,    -sa * se,     ce ]])
    return xyz @ R.T


def _render_fast(nl, types, rois, output_path,
                 elev=-7.6, azim=86.6, roll=0.0,
                 lw_large=0.2, lw_small=0.4,
                 alpha_mesh=0.10, crop_rois=None,
                 figsize=(7.5, 8.5), dpi=220,
                 background="black",
                 segs_by_type=None):
    """Fast PNG via direct 3D->2D projection. ~1 s for 140k segments
    vs ~40 s for the mplot3d path (no per-segment depth sort)."""
    from matplotlib.collections import LineCollection

    text_color = "white" if background == "black" else "black"
    mesh_color = (0.85, 0.85, 0.85) if background == "black" else (0.35, 0.35, 0.35)

    fig, ax = plt.subplots(figsize=figsize, facecolor=background)
    ax.set_facecolor(background)

    # --- meshes: silhouette via projected outline (very few lines) -----
    core = {"EB", "PB"}
    for name, mesh in rois.items():
        a_ = (alpha_mesh * 4.0) if name in core else (alpha_mesh * 2.0)
        try:
            outline = mesh.outline().entities
            segs = []
            for ent in outline:
                pts = mesh.vertices[ent.points]
                segs.extend([(pts[i], pts[i + 1])
                              for i in range(len(pts) - 1)])
            if not segs:
                continue
            segs3d = np.array(segs)                       # (E, 2, 3)
            segs2d = _project_2d(segs3d.reshape(-1, 3),
                                  elev, azim).reshape(-1, 2, 2)
            lc = LineCollection(segs2d, colors=[mesh_color],
                                linewidths=0.4, alpha=a_)
            ax.add_collection(lc)
        except Exception:
            pass

    # --- skeletons, one LineCollection per cell type -------------------
    type_counts = {t: int((types == t).sum()) for t in TYPE_ORDER}
    draw_order = sorted(TYPE_ORDER, key=lambda t: -type_counts.get(t, 0))
    if segs_by_type is None:
        segs_by_type = _extract_segments_by_type(nl, types)
    for t in draw_order:
        segs3d = segs_by_type.get(t)
        if segs3d is None or len(segs3d) == 0:
            continue
        segs2d = _project_2d(segs3d.reshape(-1, 3),
                              elev, azim).reshape(-1, 2, 2)
        big = type_counts[t] > 15
        lc = LineCollection(segs2d, colors=[TYPE_COLOR[t]],
                            linewidths=(lw_large if big else lw_small),
                            alpha=(0.7 if big else 0.95))
        ax.add_collection(lc)

    # --- axis cosmetics + legend ---------------------------------------
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=TYPE_COLOR[t], lw=2.5,
                      label=f"{t}  (n={type_counts[t]})")
               for t in TYPE_ORDER if type_counts[t] > 0]
    leg = ax.legend(handles=handles, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), fontsize=9,
                    frameon=False, handlelength=1.4)
    for txt in leg.get_texts():
        txt.set_color(text_color)

    fig.subplots_adjust(left=0.02, right=0.78, top=0.98, bottom=0.02)
    fig.savefig(output_path, dpi=dpi, facecolor=background,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}  (fast 2D, bg={background})")


def _render_matplotlib(nl, types, rois, output_path,
                       elev=-7.6, azim=86.6, roll=0.0,
                       alpha_mesh=0.05, crop_rois=None,
                       lw_large=0.2, lw_small=0.4,
                       background="black"):
    """Render the CX neurons + neuropil meshes.

    crop_rois: if given, bounding box for the view limits is the union
    of these ROI meshes; otherwise the bbox is the union of all neurons
    + all meshes (nothing gets clipped).
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    text_color = "white" if background == "black" else "black"
    mesh_color = "0.85" if background == "black" else "0.45"

    fig = plt.figure(figsize=(7.5, 8.5), facecolor=background)
    ax = fig.add_subplot(111, projection="3d", facecolor=background)

    # --- neuropil meshes as silhouette outlines (fast) ------------------
    # trimesh.outline() returns the projected boundary curves; drawn as a
    # 3-D Line3DCollection this gives a clean schematic envelope without
    # the ~30k-triangle face sort that Poly3DCollection does each render.
    core = {"EB", "PB"}
    for name, mesh in rois.items():
        a = (alpha_mesh * 4.0) if name in core else (alpha_mesh * 2.0)
        try:
            outline = mesh.outline().entities
            segs = []
            for ent in outline:
                pts = mesh.vertices[ent.points]
                segs.extend([(pts[i], pts[i + 1]) for i in range(len(pts) - 1)])
            if segs:
                segs = np.array(segs)
                lc = Line3DCollection(segs, colors=mesh_color,
                                       linewidths=0.25, alpha=a)
                ax.add_collection3d(lc)
        except Exception:
            edges = mesh.edges_unique
            segs = mesh.vertices[edges]
            lc = Line3DCollection(segs, colors=mesh_color,
                                   linewidths=0.15, alpha=a * 0.4)
            ax.add_collection3d(lc)

    # --- skeletons: draw large populations first so small ones land on top
    type_counts = {t: int((types == t).sum()) for t in TYPE_ORDER}
    draw_order = sorted(TYPE_ORDER, key=lambda t: -type_counts.get(t, 0))
    for t in draw_order:
        segs_t = []
        for n, nt in zip(nl, types):
            if nt != t:
                continue
            nodes = n.nodes
            child = nodes[nodes.parent_id != -1]
            if len(child) == 0:
                continue
            parent_xyz = nodes.set_index("node_id").loc[
                child.parent_id.values, ["x", "y", "z"]
            ].values
            child_xyz = child[["x", "y", "z"]].values
            segs_t.append(np.stack([parent_xyz, child_xyz], axis=1))
        if not segs_t:
            continue
        segs_all = np.concatenate(segs_t, axis=0)
        # Thinner lines for the big bundles, thicker for the small populations
        # (ER6=4, EPGt=4) so they're visible.
        big = type_counts.get(t, 0) > 15
        lw = lw_large if big else lw_small
        alpha = 0.65 if big else 0.95
        lc = Line3DCollection(
            segs_all, colors=[TYPE_COLOR[t]], linewidths=lw, alpha=alpha,
            rasterized=True,
        )
        ax.add_collection3d(lc)

    # --- view limits ----------------------------------------------------
    crop_pts = []
    if crop_rois:
        for name in crop_rois:
            if name in rois:
                crop_pts.append(rois[name].vertices)
    if not crop_pts:
        # No crop: union of every neuron and every mesh -> nothing clipped.
        for mesh in rois.values():
            crop_pts.append(mesh.vertices)
        for n in nl:
            crop_pts.append(n.nodes[["x", "y", "z"]].values)
    pts = np.concatenate(crop_pts, axis=0)
    pad = 0.04 * (pts.max(0) - pts.min(0))
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_axis_off()

    # --- legend (outside the axes, right side) --------------------------
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=TYPE_COLOR[t], lw=2.5,
                      label=f"{t}  (n={type_counts[t]})")
               for t in TYPE_ORDER if type_counts[t] > 0]
    leg = ax.legend(handles=handles, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), fontsize=9,
                    frameon=False, handlelength=1.4)
    for txt in leg.get_texts():
        txt.set_color(text_color)

    fig.subplots_adjust(left=0.0, right=0.78, top=1.0, bottom=0.0)
    fig.savefig(output_path, dpi=220, bbox_inches="tight",
                facecolor=background)
    plt.close(fig)
    print(f"wrote {output_path}  (bg={background})")


def _render_plotly(nl, types, rois, output_path, linewidth=0.25):
    import navis
    colours = {n.id: TYPE_COLOR.get(t, (0.4, 0.4, 0.4))
               for n, t in zip(nl, types)}
    # Skeletons only -- the volume meshes turn into heavy plotly Mesh3d
    # traces that bloat the HTML and obscure the morphology when spinning.
    fig = navis.plot3d(
        nl, backend="plotly", color=colours, soma=False, inline=False,
        linewidth=linewidth,
    )
    # Hide the legend (one entry per neuron is meaningless noise).
    fig.update_traces(showlegend=False)
    # Preserve true data proportions instead of stretching to fit the
    # browser window (plotly's aspectmode='auto' default makes the scene
    # look wider than tall).
    fig.update_layout(scene=dict(aspectmode="data"))
    # write_html with our overlay (see _inject_mpl_angle_readout).
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    _inject_mpl_angle_readout(output_path)
    print(f"wrote {output_path} (linewidth={linewidth})")


def _inject_mpl_angle_readout(html_path: str) -> None:
    """Append a small JS overlay that converts plotly's camera.eye to the
    matplotlib elev/azim/roll triple the static renderer uses, so you can
    spin the HTML, pick a view, and copy the angles into the CLI."""
    js = r"""
<style>
  #mpl_view {
    position: fixed; top: 10px; left: 10px;
    background: rgba(255,255,255,0.92); color: #111;
    font: 13px/1.35 monospace; padding: 8px 10px;
    border: 1px solid #888; border-radius: 4px; z-index: 9999;
    user-select: text;
  }
</style>
<div id="mpl_view">matplotlib view: (rotate the plot)</div>
<script>
(function() {
  function update() {
    var divs = document.querySelectorAll('.plotly-graph-div');
    if (!divs.length) return;
    var gd = divs[0];
    var cam = (((gd.layout || {}).scene || {}).camera || {}).eye;
    if (!cam) return;
    var r = Math.hypot(cam.x, cam.y, cam.z) || 1;
    var elev = Math.asin(cam.z / r) * 180 / Math.PI;
    var azim = Math.atan2(cam.y, cam.x) * 180 / Math.PI;
    document.getElementById('mpl_view').textContent =
      'matplotlib view:  --elev ' + elev.toFixed(1) +
      '  --azim ' + azim.toFixed(1) + '  --roll 0';
  }
  function wire() {
    var divs = document.querySelectorAll('.plotly-graph-div');
    if (!divs.length) { setTimeout(wire, 200); return; }
    divs[0].on('plotly_relayout', update);
    update();
  }
  wire();
})();
</script>
"""
    with open(html_path, "r") as f:
        html = f.read()
    html = html.replace("</body>", js + "</body>", 1)
    with open(html_path, "w") as f:
        f.write(html)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anatomy_dir",
                   default="papers/janelia_cx/anatomy/cx_anatomy")
    p.add_argument("--downsample", type=int, default=10,
                   help="navis.downsample_neuron factor (preserves "
                        "branch/end points). 1 = no downsample.")
    p.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--elev", type=float, default=-7.6)
    p.add_argument("--azim", type=float, default=86.6)
    p.add_argument("--roll", type=float, default=0.0)
    p.add_argument("--bg", default="black",
                   choices=["black", "white"],
                   help="figure background colour")
    p.add_argument("--slow", action="store_true",
                   help="use the matplotlib mplot3d renderer (~40 s) "
                        "instead of the fast 2D-projection path (~1 s)")
    p.add_argument("--alpha_mesh", type=float, default=0.05)
    p.add_argument("--png_lw_large", type=float, default=0.2,
                   help="PNG line width for large populations (>15 cells)")
    p.add_argument("--png_lw_small", type=float, default=0.4,
                   help="PNG line width for small populations (≤15 cells)")
    p.add_argument("--crop_rois", nargs="+", default=None,
                   help="ROI names whose bbox defines the view crop. "
                        "If unset, the bbox is the union of every neuron "
                        "and every mesh (nothing clipped).")
    p.add_argument("--plotly", action="store_true",
                   help="also write an interactive .html")
    p.add_argument("--html_linewidth", type=float, default=0.25,
                   help="line width for the plotly/HTML render "
                        "(navis default is 3)")
    args = p.parse_args()

    if not os.path.isdir(args.anatomy_dir):
        sys.exit(f"{args.anatomy_dir} does not exist -- "
                 "drop the cx_anatomy/ folder produced by fetch_cx_anatomy.py here")

    nl, types = _load_skeletons(args.anatomy_dir, downsample=args.downsample)
    rois = _load_rois(args.anatomy_dir)
    print(f"loaded {len(nl)} neurons, {len(rois)} ROI meshes "
          f"({sorted(rois.keys())})")
    for t in TYPE_ORDER:
        n_t = int((types == t).sum())
        if n_t:
            print(f"  {t:18s}  {n_t}")

    out_png = os.path.join(args.out_dir, "fig_cx_anatomy_3d.png")
    print(f"matplotlib view: elev={args.elev}  azim={args.azim}  "
          f"roll={args.roll}  bg={args.bg}  "
          f"{'slow-3d' if args.slow else 'fast-2d'}")
    renderer = _render_matplotlib if args.slow else _render_fast
    renderer(nl, types, rois, out_png,
             elev=args.elev, azim=args.azim, roll=args.roll,
             alpha_mesh=args.alpha_mesh,
             crop_rois=tuple(args.crop_rois) if args.crop_rois else None,
             lw_large=args.png_lw_large, lw_small=args.png_lw_small,
             background=args.bg)
    if args.plotly:
        out_html = os.path.join(args.out_dir, "fig_cx_anatomy_3d.html")
        _render_plotly(nl, types, rois, out_html,
                       linewidth=args.html_linewidth)


if __name__ == "__main__":
    main()
