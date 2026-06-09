"""fig_1_overview.py
============================

Unified Figure 1 of the zebrafish manuscript:
  (a) full skeletons rendered against ROI silhouettes
  (b) cell-body somas, same view
  (c) circuit variant 1: ω → ARTR, v_fwd → pt-IPN1
  (d) circuit variant 2: + motor_efferent proprioceptive branch

All four panels live in a single matplotlib figure so the panel
letters and surrounding fonts render at the same physical scale on
the page (rather than drifting between two separate PNGs that are
later stacked in LaTeX).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Reuse the anatomy + circuit-variant drawing helpers.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fig_zebrafish_anatomy_3d_HD import (  # noqa: E402
    _load_skeletons, _load_rois, _load_soma_meshes_by_type,
    draw_anatomy_panels,
)
from fig_zebrafish_circuit_variants import (  # noqa: E402
    _draw_artr_pt1_panel, _draw_propriocep_panel,
)


PANEL_LABEL_FS = 13
PANEL_LABEL_KW = dict(
    fontsize=PANEL_LABEL_FS, fontweight="bold",
    va="top", ha="left",
)


def _panel_letter(ax, letter):
    ax.text(0.01, 0.99, letter, transform=ax.transAxes, **PANEL_LABEL_KW)


def build_figure(out_path, anatomy_dir, extra_anatomy_dirs=(),
                 downsample=10, elev=90.0, azim=-90.0,
                 background="white", dpi=180, keep_bodyids=None):
    nl, types = _load_skeletons(anatomy_dir, downsample=downsample,
                                extra_dirs=extra_anatomy_dirs,
                                keep_bodyids=keep_bodyids)
    rois = _load_rois(anatomy_dir)
    soma_meshes_by_type = _load_soma_meshes_by_type(anatomy_dir)
    print(f"loaded {len(nl)} neurons, {len(rois)} ROI meshes, "
          f"{sum(len(v) for v in soma_meshes_by_type.values())} soma meshes")

    fig = plt.figure(figsize=(14.0, 8.5), facecolor=background)
    # 2×2 layout: (a, b) anatomy on top, (c, d) circuit diagrams below.
    # Use a nested GridSpec so the top row stays tight (a/b share the
    # full width with minimal gap, and the cell-type legend lives INSIDE
    # ax_b so it doesn't push b wider than a) while the bottom row keeps
    # extra horizontal breathing room between c and d.
    gs_root = GridSpec(2, 1, figure=fig,
                       height_ratios=[4.2, 4.0],
                       hspace=0.0,
                       left=0.01, right=0.99, top=0.99, bottom=0.02)
    gs_top = gs_root[0].subgridspec(1, 2, wspace=0.02)
    gs_bot = gs_root[1].subgridspec(1, 2, wspace=0.18)
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_bot[0, 0])
    ax_d = fig.add_subplot(gs_bot[0, 1])

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_facecolor(background)
    # Pin the anatomy axes to the bottom of their row so the wide,
    # short dorsal brain view sits flush against panels c, d below
    # (no big blank between rows when the data aspect leaves vertical
    # space).
    ax_a.set_anchor("S")
    ax_b.set_anchor("S")

    draw_anatomy_panels(
        ax_a, ax_b, nl, types, rois,
        elev=elev, azim=azim,
        background=background,
        soma_meshes_by_type=soma_meshes_by_type or None,
        legend=True, legend_inside=True,
    )
    _draw_artr_pt1_panel(ax_c)
    _draw_propriocep_panel(ax_d)

    _panel_letter(ax_a, "a")
    _panel_letter(ax_b, "b")
    _panel_letter(ax_c, "c")
    _panel_letter(ax_d, "d")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=background,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


def main():
    here = HERE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--anatomy_dir",
                    default=os.path.join(here, "zebrafish_anatomy_HD"))
    ap.add_argument("--extra_anatomy_dirs", nargs="+", default=None,
                    help="additional anatomy caches to merge in; "
                         "defaults to zebrafish_anatomy_IPN12")
    ap.add_argument("--downsample", type=int, default=10)
    ap.add_argument("--elev", type=float, default=90.0)
    ap.add_argument("--azim", type=float, default=-90.0)
    ap.add_argument("--bg", default="white", choices=["white", "black"])
    ap.add_argument("--out", default=os.path.join(
        here, "fig_1_overview.png"))
    args = ap.parse_args()

    if args.extra_anatomy_dirs is None:
        # Default: merge the IPN12 pool + the IPN-core families (943 circuit).
        cand = [os.path.join(here, "zebrafish_anatomy_IPN12"),
                os.path.join(here, "zebrafish_anatomy_IPN_core")]
        extra = tuple(d for d in cand if os.path.isdir(d))
    else:
        extra = tuple(args.extra_anatomy_dirs)

    # Restrict the render to the exact 943-cell circuit selection (the caches
    # hold all cells of each type, which can exceed the pickle's selection),
    # so the panel-b legend counts match the circuit / caption.
    keep = None
    conn = os.path.join(here, "zebrafish_connectome_HD_IPN_917", "neurons.csv")
    if os.path.isfile(conn):
        import csv as _csv
        with open(conn) as fh:
            keep = {int(r["bodyId"]) for r in _csv.DictReader(fh)}

    build_figure(args.out, args.anatomy_dir, extra_anatomy_dirs=extra,
                 downsample=args.downsample, elev=args.elev, azim=args.azim,
                 background=args.bg, keep_bodyids=keep)


if __name__ == "__main__":
    main()
