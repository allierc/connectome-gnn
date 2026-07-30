"""Per-instance coloured 3-D CX anatomy (companion to fig_cx_anatomy_3d.py).

Colours every CX skeleton by its hemibrain *instance* --- the per-cell
computational unit used in the Appendix-A degeneracy analysis --- instead of
by cell type. The 156 CX neurons fall into 81 instances; same-type neurons
that share an instance (clones, same heading tuning) get the same colour.

Output: figure/fig_cx_anatomy_instance.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_cx_anatomy_3d import _load_rois, _draw_mesh_outlines, _project_2d

CSV = ("papers/Code_NN/Code_NN/Data/Figure5/"
       "exported-traced-adjacencies-v1.2/traced-neurons.csv")


def _segments_by_instance(anatomy_dir, bid2inst, downsample=10):
    """{instance: (E,2,3) segments} from the cached SWCs, keyed by the
    presynaptic neuron's hemibrain instance (bodyId from the filename)."""
    import navis
    segs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(anatomy_dir, "skeletons", "*.swc"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        bid = int(stem.rpartition("__")[2])
        inst = bid2inst.get(bid, f"body{bid}")
        n = navis.read_swc(path)
        if downsample and downsample > 1:
            n = navis.downsample_neuron(n, downsampling_factor=downsample,
                                         preserve_nodes=None)
        nodes = n.nodes
        child = nodes[nodes.parent_id != -1]
        if len(child) == 0:
            continue
        p = nodes.set_index("node_id").loc[
            child.parent_id.values, ["x", "y", "z"]].values
        c = child[["x", "y", "z"]].values
        segs[inst].append(np.stack([p, c], axis=1))
    return {k: np.concatenate(v, axis=0) for k, v in segs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anatomy_dir",
                    default="papers/janelia_cx/anatomy/cx_anatomy_test")
    ap.add_argument("--downsample", type=int, default=10)
    ap.add_argument("--elev", type=float, default=-7.6)
    ap.add_argument("--azim", type=float, default=86.6)
    ap.add_argument("--bg", default="black", choices=["black", "white"])
    ap.add_argument("--flip_y", action="store_true")
    ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    bid2inst = dict(zip(pd.read_csv(CSV).bodyId, pd.read_csv(CSV).instance))
    segs = _segments_by_instance(args.anatomy_dir, bid2inst, args.downsample)
    rois = _load_rois(args.anatomy_dir)

    # Sort instances by name so same-type instances (shared prefix) cluster in
    # hue; sample a wide qualitative map for 81 distinguishable colours.
    instances = sorted(segs.keys())
    n_inst = len(instances)
    cmap = matplotlib.cm.get_cmap("gist_ncar")
    colours = {inst: cmap((i + 0.5) / n_inst)[:3]
               for i, inst in enumerate(instances)}

    bg = args.bg
    text_color = "white" if bg == "black" else "black"
    mesh_color = (0.85, 0.85, 0.85) if bg == "black" else (0.35, 0.35, 0.35)

    fig, ax = plt.subplots(figsize=(8.5, 9.0), facecolor=bg)
    ax.set_facecolor(bg)
    _draw_mesh_outlines(ax, rois, args.elev, args.azim, 0.10, mesh_color)
    # big bundles (>15 segments-heavy) thinner; draw all instances.
    for inst in instances:
        s2 = _project_2d(segs[inst].reshape(-1, 3),
                         args.elev, args.azim).reshape(-1, 2, 2)
        ax.add_collection(LineCollection(
            s2, colors=[colours[inst]], linewidths=0.5, alpha=0.9))

    ax.set_aspect("equal")
    ax.autoscale_view()
    if args.flip_y:
        ax.invert_yaxis()
    ax.set_axis_off()
    ax.text(0.5, 0.995,
            f"per-instance colouring — {n_inst} hemibrain instances "
            f"over 156 neurons",
            transform=ax.transAxes, ha="center", va="top",
            color=text_color, fontsize=13)

    out = os.path.join(args.out_dir, "fig_cx_anatomy_instance.png")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02)
    fig.savefig(out, dpi=220, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}   instances={n_inst}")
    # per-type breakdown of instance counts
    by_type = defaultdict(set)
    for inst in instances:
        t = inst.split("(")[0].split("_")[0] if "(" in inst else inst.split("_")[0]
        by_type[t].add(inst)
    for t, s in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        print(f"  {t:10s} {len(s)} instances")


if __name__ == "__main__":
    main()
