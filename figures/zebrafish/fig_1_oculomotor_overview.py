"""Figure 1 of the oculomotor note — twin of the zebrafish paper's Figure 1.

Three panels in one matplotlib figure, so panel letters and fonts render at
the same physical scale on the page:

  (a) full skeletons against the ROI silhouettes
  (b) cell-body somas, same view
  (c) the velocity-tracking computation: eye velocity in, eye position out

Panel (d) of the paper's Figure 1 — the second circuit variant — has no
counterpart here: this circuit has one afferent taxonomy, not two.

Panels (a) and (b) come from ``fig_zebrafish_anatomy_3d_HD`` and panel (c)
from the box/arrow primitives of ``fig_zebrafish_circuit_variants``, so this
is the same renderer as the paper figure with the content swapped, not a
reimplementation of it.

Usage::

    python fig_1_oculomotor_overview.py [--view dorsal|lateral] [--bg white]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_zebrafish_anatomy_3d_HD as ref                    # noqa: E402
import fig_zebrafish_circuit_variants as cv                  # noqa: E402
from fig_oculomotor_anatomy import ROLE, COLOR, KEEP_ROIS    # noqa: E402

PANEL_LABEL_KW = dict(fontsize=13, fontweight="bold", va="top", ha="left")
# Fills echo the role colours of panels a/b, pale enough to take black text.
F_AF5 = "#c7d8e8"      # afferent, matches the blue skeletons
F_INTG_E = "#cfe8d2"   # excitatory ipsilateral integrator, green
F_INTG_I = "#f3cdd0"   # inhibitory contralateral integrator, red
F_AMN = "#ddd0f0"      # lateral rectus motor pool, purple
F_AIN = "#f4d6e6"      # medial rectus motor pool, pink
F_REC = "#f6f1e6"      # container behind the integrator


def _panel_c(ax):
    """Velocity in, position out — the computation, not the anatomy."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # --- input ------------------------------------------------------------
    cv._text(ax, (0.005, 0.52), [r"$v=\dot{x}_{\mathrm{eye}}$"],
             fontsize=cv.LABEL_FS + 2, ha="left")
    cv._arrow(ax, (0.085, 0.52), (0.13, 0.52), color=cv.COL_OMEGA, lw=2.2)

    cv._draw_afferent_block(ax, (0.13, 0.43), (0.17, 0.19),
                            "AF5 L / R", r"$n=29+12$",
                            sub_lines=["optokinetic drive"], fill=F_AF5)

    # --- recurrent integrator --------------------------------------------
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (0.365, 0.20), 0.27, 0.66, boxstyle="round,pad=0.012",
        linewidth=1.0, edgecolor="0.45", facecolor=F_REC, zorder=1))
    cv._text(ax, (0.50, 0.90), ["recurrent integrator"],
             fontsize=cv.LABEL_FS, ha="center")
    cv._draw_afferent_block(ax, (0.385, 0.575), (0.23, 0.17),
                            "INTG ipsi  (E)", r"$n=38+27$",
                            sub_lines=["INTGip1 / ip2"], fill=F_INTG_E)
    cv._draw_afferent_block(ax, (0.385, 0.315), (0.23, 0.17),
                            "INTG contra  (I)", r"$n=34+18$",
                            sub_lines=["INTGco1 / co2"], fill=F_INTG_I)
    # Mutual inhibition across the midline — the line-attractor motif. The
    # arrows flank the equation rather than crossing it.
    cv._arrow(ax, (0.408, 0.565), (0.408, 0.495), color="#b03a3a", lw=1.4)
    cv._arrow(ax, (0.592, 0.495), (0.592, 0.565), color="#3a8a45", lw=1.4)
    cv._text(ax, (0.50, 0.53), [r"$\tau\,\dot h=-h+Wr+W_{\rm in}v$"],
             fontsize=cv.LABEL_FS - 2, ha="center")

    # --- output ----------------------------------------------------------
    cv._draw_afferent_block(ax, (0.665, 0.575), (0.20, 0.17),
                            "AMN", r"$n=92$",
                            sub_lines=["lateral rectus"], fill=F_AMN)
    cv._draw_afferent_block(ax, (0.665, 0.315), (0.20, 0.17),
                            "AIN", r"$n=35$",
                            sub_lines=["medial rectus"], fill=F_AIN)

    cv._arrow(ax, (0.335, 0.52), (0.365, 0.52), color="0.4", lw=1.4)
    cv._arrow(ax, (0.635, 0.66), (0.665, 0.66), color="0.4", lw=1.4)
    cv._arrow(ax, (0.635, 0.40), (0.665, 0.40), color="0.4", lw=1.4)
    cv._arrow(ax, (0.875, 0.66), (0.925, 0.60), color="0.4", lw=1.4)
    cv._arrow(ax, (0.875, 0.40), (0.925, 0.46), color="0.4", lw=1.4)
    cv._text(ax, (0.995, 0.53),
             [r"$x_{\mathrm{eye}}=\int v\,dt$", "(LR $-$ MR)"],
             fontsize=cv.LABEL_FS, ha="right")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anatomy", default=os.path.join(HERE, "zebrafish_anatomy_OCULO"))
    p.add_argument("--out", default=os.path.join(HERE, "fig_1_oculomotor_overview.png"))
    p.add_argument("--view", default="dorsal", choices=["dorsal", "lateral"])
    p.add_argument("--downsample", type=int, default=4)
    p.add_argument("--soma-scale", type=float, default=0.30)
    p.add_argument("--bg", default="white", choices=["white", "black"])
    a = p.parse_args()

    ref._type_to_category = lambda s: ROLE.get(s, "other")
    ref.TYPE_COLOR = dict(COLOR)
    ref.TYPE_ORDER = list(COLOR)

    nl, types = ref._load_skeletons(a.anatomy, downsample=a.downsample)
    rois = {k: v for k, v in ref._load_rois(a.anatomy).items()
            if k in set(KEEP_ROIS)}
    segs = ref._extract_segments_by_type(nl, types)
    somas = {t: (pos, np.asarray(rad) * a.soma_scale)
             for t, (pos, rad) in ref._extract_somas_by_type(nl, types).items()}
    elev, azim = (90, -90) if a.view == "dorsal" else (0, -90)
    print(f"{len(nl)} skeletons, {len(rois)} ROIs, {a.view} view")

    fig = plt.figure(figsize=(13.5, 8.2), facecolor=a.bg)
    gs = GridSpec(2, 2, height_ratios=[1.0, 1.05], hspace=0.06, wspace=0.03,
                  figure=fig)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ref.draw_anatomy_panels(ax_a, ax_b, nl, types, rois, elev=elev, azim=azim,
                            background=a.bg, alpha_mesh=0.10,
                            segs_by_type=segs, somas_by_type=somas,
                            legend=True)
    _panel_c(ax_c)

    fg = "white" if a.bg == "black" else "black"
    for ax, L in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        ax.text(0.0, 1.0, L, transform=ax.transAxes, color=fg, **PANEL_LABEL_KW)
    fig.savefig(a.out, dpi=180, facecolor=a.bg, bbox_inches="tight")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
