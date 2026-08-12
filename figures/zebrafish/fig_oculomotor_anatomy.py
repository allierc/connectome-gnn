"""3-D anatomy of the 285-cell oculomotor circuit, dorsal and lateral.

Renders the cache written by ``fetch_zebrafish_anatomy_OCULO.py`` through the
paper's own renderer (``fig_zebrafish_anatomy_3d_HD``), retargeting only its
two hooks — the type-to-category map and the palette — so the projection,
skeleton drawing, soma spheres and legend are the same code that drew Figure
1a, not a lookalike.

Two renderer defaults are tuned for the HD circuit and need overriding here:

  ROIs.   ``_draw_mesh_outlines`` draws ``mesh.outline()``, the boundary
          edges. For the compact HD compartments (dIPN, Habenula) that reads
          as a silhouette; for the whole ``Hindbrain`` and ``Diencephalon``
          it produces long straight chords across the entire field — the grey
          streaks. Only the small, informative ROIs are kept by default.

  Somas.  ``_draw_soma_icospheres`` draws a sphere of the SWC's own radius.
          fish2 radii are generous relative to this circuit's much tighter
          volume, so at true scale the cell bodies merge into blobs. They are
          scaled by ``--soma-scale`` for legibility; the value is printed and
          recorded on the figure so it is never mistaken for a measurement.

Usage::

    python fig_oculomotor_anatomy.py
    python fig_oculomotor_anatomy.py --soma-scale 0.25 --rois Abducens_L Abducens_R
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_zebrafish_anatomy_3d_HD as ref            # noqa: E402

# Role, not cell type: the same partition the config and the connectivity
# figure use, so the three read together.
ROLE = {
    "AF5_ipsi": "afferent AF5", "AF5_contra": "afferent AF5",
    "INTG_ipsi_m": "INTG ipsi (E)", "INTG_ipsi_i": "INTG ipsi (E)",
    "INTG_contra_m": "INTG contra (I)", "INTG_contra_i": "INTG contra (I)",
    "AMN": "AMN / lateral rectus", "AIN": "AIN / medial rectus",
}
COLOR = {
    "afferent AF5": (0.12, 0.44, 0.92),
    "INTG ipsi (E)": (0.18, 0.63, 0.26),
    "INTG contra (I)": (0.81, 0.13, 0.18),
    "AMN / lateral rectus": (0.54, 0.34, 0.90),
    "AIN / medial rectus": (0.85, 0.44, 0.72),
}
# Small enough that their outline reads as a silhouette rather than a chord.
KEEP_ROIS = ["Abducens_L", "Abducens_R"]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anatomy", default=os.path.join(HERE, "zebrafish_anatomy_OCULO"))
    p.add_argument("--out-prefix", default=os.path.join(HERE, "oculo_anatomy"))
    p.add_argument("--downsample", type=int, default=4)
    p.add_argument("--soma-scale", type=float, default=0.30,
                   help="multiply SWC soma radii for legibility (1.0 = true)")
    p.add_argument("--rois", nargs="+", default=KEEP_ROIS,
                   help="ROI outlines to draw; pass none to omit them")
    p.add_argument("--alpha-mesh", type=float, default=0.10)
    p.add_argument("--bg", default="white", choices=["white", "black"])
    a = p.parse_args()

    ref._type_to_category = lambda s: ROLE.get(s, "other")
    ref.TYPE_COLOR = dict(COLOR)
    ref.TYPE_ORDER = list(COLOR)

    nl, types = ref._load_skeletons(a.anatomy, downsample=a.downsample)
    rois = {k: v for k, v in ref._load_rois(a.anatomy).items()
            if k in set(a.rois)}
    print(f"{len(nl)} skeletons | ROIs kept: {sorted(rois)} "
          f"(of those cached)")
    print("  by role:", dict(collections.Counter(map(str, types))))

    segs = ref._extract_segments_by_type(nl, types)
    somas = ref._extract_somas_by_type(nl, types)
    if a.soma_scale != 1.0:                      # shrink the cell bodies
        somas = {t: (pos, np.asarray(rad) * a.soma_scale)
                 for t, (pos, rad) in somas.items()}
        rmed = np.median(np.concatenate([r for _, r in somas.values()]))
        print(f"  soma radii x{a.soma_scale} -> median {rmed / 1000:.2f} um")

    for elev, azim, tag in ((90, -90, "dorsal"), (0, -90, "lateral")):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2),
                                 facecolor=a.bg,
                                 gridspec_kw=dict(wspace=0.02))
        ref.draw_anatomy_panels(axes[0], axes[1], nl, types, rois,
                                elev=elev, azim=azim, background=a.bg,
                                alpha_mesh=a.alpha_mesh,
                                segs_by_type=segs, somas_by_type=somas,
                                legend=True)
        fg = "white" if a.bg == "black" else "black"
        axes[0].text(0.01, 0.99, "skeletons", transform=axes[0].transAxes,
                     va="top", ha="left", fontsize=10, color=fg)
        axes[1].text(0.01, 0.99,
                     f"cell bodies (radii x{a.soma_scale:g})",
                     transform=axes[1].transAxes, va="top", ha="left",
                     fontsize=10, color=fg)
        out = f"{a.out_prefix}_{tag}.png"
        fig.savefig(out, dpi=170, facecolor=a.bg, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
