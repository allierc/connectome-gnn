"""3-D view of the Janelia optic-lobe neurons (smoke fetch by
fetch_optic_lobe_anatomy.py). Direct sibling of fig_cx_anatomy_3d.py
but with layer-aware colouring tailored to the 65 flyvis cell-types.

Inputs:
    papers/optic_lobe_anatomy/<dir>/skeletons/<type>__<bodyId>.swc
    papers/optic_lobe_anatomy/<dir>/rois/<ROI>.obj

Outputs (under --out_dir, default = same as --anatomy_dir parent):
    fig_optic_lobe_anatomy_3d.png             # fast 2-D projection
    fig_optic_lobe_anatomy_3d.html (optional) # interactive plotly
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

# Reuse the CX renderer's machinery so we don't duplicate code.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "drosophila"))
from fig_cx_anatomy_3d import (  # noqa: E402
    _project_2d, _load_rois,
)


# ---- layer-based palette ------------------------------------------------
# Each cell-type group maps to a base hue. Within each group we vary
# brightness for the individual types.
LAYER_COLORS = {
    "photoreceptor": (0.85, 0.15, 0.15),    # red
    "lamina":         (0.95, 0.75, 0.10),   # yellow
    "medulla_shallow":(0.95, 0.45, 0.05),   # orange
    "medulla_deep":   (0.70, 0.35, 0.10),   # brown
    "TmY":            (0.55, 0.85, 0.10),   # lime
    "lobula":         (0.10, 0.75, 0.45),   # green
    "lobula_plate":   (0.15, 0.45, 0.95),   # blue
    "wide_field":     (0.65, 0.30, 0.85),   # purple
    "amacrine":       (0.85, 0.45, 0.85),   # magenta
    "other":          (0.55, 0.55, 0.55),   # grey fallback
}


def _classify(cell_type: str) -> str:
    """Heuristic cell-type -> layer mapping for the Janelia optic-lobe set."""
    t = cell_type
    if t.startswith("R"):
        return "photoreceptor"
    if t in {"L1", "L2", "L3", "L4", "L5", "Am1", "C2", "C3", "Lawf1", "Lawf2"}:
        return "lamina" if t.startswith(("L", "C")) else "amacrine"
    if t.startswith("Mi"):
        return "medulla_shallow"
    if t.startswith("Tm") and not t.startswith("TmY"):
        return "medulla_deep"
    if t.startswith("TmY"):
        return "TmY"
    if t.startswith("T1") or t in {"T2", "T2a", "T3"}:
        return "medulla_deep"
    if t.startswith(("T4", "T5")):
        return "lobula_plate"
    if t.startswith("CT1"):
        return "wide_field"
    if "Am" in t:
        return "amacrine"
    return "other"


def _load_segments_fast(anatomy_dir: str, downsample: int = 30,
                         verbose: bool = True):
    """Streaming SWC -> segment arrays. ~50x faster + ~10x less memory
    than the navis-based loader on tens of thousands of files. We never
    keep a Neuron object alive; segment arrays are accumulated per type
    and we discard the raw DataFrame after each SWC.

    Returns:
        segs_by_layer: dict layer_name -> (E, 2, 3) float32 array
        soma_xyz_by_layer: dict layer_name -> (N_neurons, 3)
        types_seen: list of unique type strings encountered (for the
                    layer-count print)
    """
    import pandas as pd
    swc_paths = sorted(glob.glob(os.path.join(anatomy_dir,
                                              "skeletons", "*.swc")))
    if not swc_paths:
        sys.exit(f"no SWCs under {anatomy_dir}/skeletons/")

    segs_by_layer: dict[str, list[np.ndarray]] = {}
    soma_by_layer: dict[str, list[np.ndarray]] = {}
    types_seen: list[str] = []

    import time as _t
    t0 = _t.time()
    last_print = t0
    for k, path in enumerate(swc_paths):
        try:
            df = pd.read_csv(
                path, sep=r"\s+", comment="#", header=None,
                names=["node_id", "label", "x", "y", "z", "radius", "parent_id"],
            )
        except Exception:
            continue
        if len(df) == 0:
            continue
        children = df[df.parent_id != -1]
        if len(children) > 0:
            pid_to_xyz = df.set_index("node_id")[["x", "y", "z"]]
            try:
                parent_xyz = pid_to_xyz.loc[children.parent_id.values].values
            except KeyError:
                # SWC with bad parent references; skip
                continue
            child_xyz = children[["x", "y", "z"]].values
            segs = np.stack([parent_xyz, child_xyz], axis=1).astype(np.float32)
            if downsample and downsample > 1 and len(segs) > downsample:
                segs = segs[::downsample]
        else:
            segs = np.zeros((0, 2, 3), dtype=np.float32)

        # Soma = largest-radius node (hemibrain SWC convention)
        soma_row = df.loc[df.radius.idxmax()]
        soma_xyz = np.array([soma_row.x, soma_row.y, soma_row.z],
                            dtype=np.float32)

        stem = os.path.splitext(os.path.basename(path))[0]
        safe_type = stem.rpartition("__")[0]
        layer = _classify(safe_type)
        segs_by_layer.setdefault(layer, []).append(segs)
        soma_by_layer.setdefault(layer, []).append(soma_xyz)
        types_seen.append(safe_type)

        if verbose and (k % 2000 == 0 or _t.time() - last_print > 30):
            elapsed = _t.time() - t0
            rate = (k + 1) / elapsed
            print(f"  loaded {k+1}/{len(swc_paths)} "
                  f"({rate:.0f}/s, ETA {(len(swc_paths)-k-1)/rate:.0f}s)",
                  flush=True)
            last_print = _t.time()

    # Concatenate
    out_segs = {k: np.concatenate(v, axis=0) for k, v in segs_by_layer.items()
                if v}
    out_soma = {k: np.stack(v, axis=0) for k, v in soma_by_layer.items() if v}
    elapsed = _t.time() - t0
    if verbose:
        print(f"  load done: {len(swc_paths)} files in {elapsed:.0f}s "
              f"({sum(s.shape[0] for s in out_segs.values()):,} segments)",
              flush=True)
    return out_segs, out_soma, types_seen


def _render(segs_by_layer, soma_by_layer, rois, output_path,
             elev=18.4, azim=-107.6, roll=0.0,
             bg="black", linewidth=0.25, alpha=0.7, soma_size=0.5):
    from matplotlib.collections import LineCollection

    text_color = "white" if bg == "black" else "black"
    mesh_color = "0.85" if bg == "black" else "0.45"

    fig, ax = plt.subplots(figsize=(8.5, 8.5), facecolor=bg)
    ax.set_facecolor(bg)

    # ROI outlines
    for name, mesh in rois.items():
        try:
            outline = mesh.outline().entities
            segs = []
            for ent in outline:
                pts = mesh.vertices[ent.points]
                segs.extend([(pts[i], pts[i + 1])
                              for i in range(len(pts) - 1)])
            if segs:
                segs = np.array(segs)
                segs2d = _project_2d(segs.reshape(-1, 3),
                                      elev, azim).reshape(-1, 2, 2)
                lc = LineCollection(segs2d, colors=[mesh_color],
                                     linewidths=0.4, alpha=0.20)
                ax.add_collection(lc)
        except Exception:
            pass

    # Per-layer skeleton segments
    for layer, segs3d in segs_by_layer.items():
        if len(segs3d) == 0:
            continue
        segs2d = _project_2d(segs3d.reshape(-1, 3),
                              elev, azim).reshape(-1, 2, 2)
        ax.add_collection(LineCollection(
            segs2d, colors=[LAYER_COLORS.get(layer, LAYER_COLORS["other"])],
            linewidths=linewidth, alpha=alpha,
        ))

    # Soma markers
    for layer, somas in soma_by_layer.items():
        if len(somas) == 0:
            continue
        s2d = _project_2d(somas, elev, azim)
        ax.scatter(s2d[:, 0], s2d[:, 1], s=soma_size,
                   c=[LAYER_COLORS.get(layer, LAYER_COLORS["other"])],
                   edgecolors="none", alpha=0.9, zorder=5)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=LAYER_COLORS[k], lw=2.5,
                       label=f"{k} (n={len(soma_by_layer.get(k, []))})")
               for k in LAYER_COLORS if k in segs_by_layer]
    leg = ax.legend(handles=handles, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), fontsize=9,
                    frameon=False, handlelength=1.4)
    for txt in leg.get_texts():
        txt.set_color(text_color)

    fig.subplots_adjust(left=0.02, right=0.78, top=0.98, bottom=0.02)
    fig.savefig(output_path, dpi=220, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def _render_plotly(segs_by_layer, soma_by_layer, rois, output_path,
                   linewidth=0.5):
    """Plotly HTML with one Scatter3d trace per layer (skeletons as
    multi-segment polylines with NaN breaks). Skips navis entirely."""
    import plotly.graph_objects as go
    traces = []
    for layer, segs3d in segs_by_layer.items():
        if len(segs3d) == 0:
            continue
        # Build a single polyline per layer with NaN breaks between segments
        n = segs3d.shape[0]
        xs = np.empty(3 * n, dtype=np.float32)
        ys = np.empty(3 * n, dtype=np.float32)
        zs = np.empty(3 * n, dtype=np.float32)
        xs[0::3] = segs3d[:, 0, 0]; xs[1::3] = segs3d[:, 1, 0]; xs[2::3] = np.nan
        ys[0::3] = segs3d[:, 0, 1]; ys[1::3] = segs3d[:, 1, 1]; ys[2::3] = np.nan
        zs[0::3] = segs3d[:, 0, 2]; zs[1::3] = segs3d[:, 1, 2]; zs[2::3] = np.nan
        c = LAYER_COLORS.get(layer, LAYER_COLORS["other"])
        rgb = f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=rgb, width=linewidth),
            name=layer, opacity=0.7, hoverinfo="skip",
        ))
    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(aspectmode="data"),
                      paper_bgcolor="black",
                      showlegend=False,
                      margin=dict(l=0, r=0, t=0, b=0))
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    from fig_cx_anatomy_3d import _inject_mpl_angle_readout
    _inject_mpl_angle_readout(output_path)
    print(f"wrote {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anatomy_dir", required=True,
                   help="dir containing skeletons/ and rois/")
    p.add_argument("--out_dir",
                   default=os.path.dirname(os.path.abspath(__file__)),
                   help="output dir (default: this script's dir, "
                        "figures/flyvis/)")
    p.add_argument("--elev", type=float, default=18.4)
    p.add_argument("--azim", type=float, default=-107.6)
    p.add_argument("--roll", type=float, default=0.0)
    p.add_argument("--downsample", type=int, default=20)
    p.add_argument("--plotly", action="store_true")
    p.add_argument("--linewidth", type=float, default=0.25)
    p.add_argument("--soma_size", type=float, default=0.5,
                   help="scatter marker size for soma dots (matplotlib s=)")
    args = p.parse_args()

    if not os.path.isdir(args.anatomy_dir):
        sys.exit(f"{args.anatomy_dir} does not exist")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    segs_by_layer, soma_by_layer, types_seen = _load_segments_fast(
        args.anatomy_dir, downsample=args.downsample,
    )
    rois = _load_rois(args.anatomy_dir)
    print(f"loaded {len(types_seen)} neurons across "
          f"{len(segs_by_layer)} layers, {len(rois)} ROIs")
    print("per-layer counts:")
    for layer in segs_by_layer:
        print(f"  {layer:18s} {len(soma_by_layer.get(layer, []))} neurons, "
              f"{segs_by_layer[layer].shape[0]:,} segments")

    print(f"view: elev={args.elev} azim={args.azim} roll={args.roll}")
    _render(segs_by_layer, soma_by_layer, rois,
             os.path.join(out_dir, "fig_optic_lobe_anatomy_3d.png"),
             elev=args.elev, azim=args.azim, roll=args.roll,
             linewidth=args.linewidth, soma_size=args.soma_size)
    if args.plotly:
        _render_plotly(
            segs_by_layer, soma_by_layer, rois,
            os.path.join(out_dir, "fig_optic_lobe_anatomy_3d.html"),
            linewidth=args.linewidth,
        )


if __name__ == "__main__":
    main()
