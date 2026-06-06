"""fig_zebrafish_circuit_variants.py
=====================================

Schematic of the two production afferent-partition variants of the
zebrafish HD / IPN circuit. Same 839-cell connectome in both panels.

Two panels (a–b), each laid out left-to-right:
    afferent populations ─→ recurrent network ─→ decoder readout

  (a) cell-type partition: ω routed to ARTR (RIPN01+02+03_a+03_b);
      v_fwd routed to pt-IPN1 (optic / water-flow afferents).
  (b) proprioception extension: same as (a) plus a third afferent
      route — v_fwd is now delivered through TWO parallel pathways,
      pt-IPN1 (exteroceptive) AND motor_efferent (proprioceptive copy
      of the swim command: RIPN11 + RIPN12_a + RIPN12_c).
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- Style ------------------------------------------------------------------
# Larger fonts throughout, no bold anywhere, no box outlines.
PANEL_LABEL_FS = 20
TITLE_FS = 14
LABEL_FS = 12
COUNT_FS = 11
SUB_FS = 10
LEGEND_FS = 12

# Drive-type palette — deep, saturated colours for the arrows.
COL_OMEGA    = "#1f6fb3"      # ω (angular velocity)
COL_VFWD_EXT = "#e07b1a"      # v_fwd exteroceptive
COL_VFWD_PRO = "#2a9d3d"      # v_fwd proprioceptive

# Box fills — no edge stroke; colour alone identifies the role.
COL_AFF_FILL = "#f4f1ec"      # afferent (warm grey)
COL_REC_FILL = "#f6e6ce"      # recurrent network (pale amber)
COL_DEC_FILL = "#e6efde"      # decoder readout (pale leaf)


def _panel_label(ax, letter: str):
    # Push the letter a bit further outside the axes so the larger
    # in-panel labels don't collide with it.
    ax.text(-0.04, 1.04, letter, transform=ax.transAxes,
            fontsize=PANEL_LABEL_FS, va="bottom", ha="right")


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _shadow(ax, xy, wh, dx=0.006, dy=-0.006):
    """Soft drop shadow under a box — adds depth without screaming."""
    x, y = xy; w, h = wh
    patch = FancyBboxPatch(
        (x + dx, y + dy), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.014",
        facecolor="0.65", edgecolor="none", alpha=0.25, zorder=1,
    )
    ax.add_patch(patch)


def _box(ax, xy, wh, *, fill, zorder=3):
    """Filled, edge-less rounded rectangle with a soft drop shadow."""
    x, y = xy; w, h = wh
    _shadow(ax, xy, wh)
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.014",
        facecolor=fill, edgecolor="none", linewidth=0, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, xy_from, xy_to, color="0.25", lw=1.5, ls="-",
           connectionstyle="arc3,rad=0.0", zorder=4):
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle="-|>,head_length=7,head_width=4",
        mutation_scale=1, color=color, lw=lw, linestyle=ls,
        connectionstyle=connectionstyle, shrinkA=2, shrinkB=2,
        zorder=zorder, capstyle="round",
    )
    ax.add_patch(arr)


def _drive_legend(ax, has_propriocep: bool):
    items = [(COL_OMEGA,    r"$\omega$"),
             (COL_VFWD_EXT, r"$v_{\mathrm{fwd}}$ (extero)")]
    if has_propriocep:
        items.append((COL_VFWD_PRO,
                       r"$v_{\mathrm{fwd}}$ (propriocep)"))
    x0, y0, dy = 0.60, 0.13, -0.045
    for i, (col, lab) in enumerate(items):
        ax.plot([x0, x0 + 0.05], [y0 + i * dy, y0 + i * dy],
                color=col, lw=2.6, solid_capstyle="round",
                transform=ax.transAxes)
        ax.text(x0 + 0.06, y0 + i * dy, lab, transform=ax.transAxes,
                fontsize=LEGEND_FS, va="center", ha="left")


def _text(ax, xy, lines, fontsize=LABEL_FS, va="center", ha="center"):
    """Multi-line text helper. No bold, no stroke effects."""
    x, y = xy
    ax.text(x, y, "\n".join(lines), ha=ha, va=va,
            fontsize=fontsize, zorder=5)


def _draw_recurrent_block(ax):
    """Centre column: bump network block (no ring schematic, per the rework
    brief). Two stacked panels — the IPNd ring pool and the IPN12 pool —
    both labelled as part of the same recurrent network."""
    cx = 0.55
    # Outer container.
    _box(ax, (0.40, 0.30), (0.30, 0.48), fill=COL_REC_FILL)
    _text(ax, (cx, 0.74), ["recurrent network"], fontsize=TITLE_FS)
    # Two stacked tiles inside (no graphical ring — text only).
    _box(ax, (0.43, 0.55), (0.24, 0.13), fill="#fff1d8", zorder=4)
    _text(ax, (cx, 0.615),
          [r"$d$IPN heading ring",
           r"($n=443$, $r1\pi$ cells)"],
          fontsize=LABEL_FS)
    _box(ax, (0.43, 0.36), (0.24, 0.13), fill="#fff1d8", zorder=4)
    _text(ax, (cx, 0.425),
          ["IPN12 pool",
           r"($n=108$)"],
          fontsize=LABEL_FS)


def _draw_decoder_block(ax, out_lines):
    _box(ax, (0.78, 0.40), (0.20, 0.30), fill=COL_DEC_FILL)
    _text(ax, (0.88, 0.62), ["decoder"], fontsize=TITLE_FS)
    _text(ax, (0.88, 0.50), out_lines, fontsize=LABEL_FS)


def _draw_afferent_block(ax, xy, wh, name, count, sub_lines=None):
    """Afferent population box — fill only, no edge stroke.

    Three lines stacked top-to-bottom: name, count, optional sub_lines.
    Positions are measured from the box's vertical centre so the layout
    stays balanced as the font sizes scale up.
    """
    _box(ax, xy, wh, fill=COL_AFF_FILL)
    cx = xy[0] + wh[0] / 2
    cy = xy[1] + wh[1] / 2
    if sub_lines:
        _text(ax, (cx, cy + 0.040), [name], fontsize=LABEL_FS)
        _text(ax, (cx, cy + 0.005), [count], fontsize=COUNT_FS)
        _text(ax, (cx, cy - 0.040), sub_lines, fontsize=SUB_FS)
    else:
        # Two lines only — stack them tightly around the centre.
        _text(ax, (cx, cy + 0.018), [name], fontsize=LABEL_FS)
        _text(ax, (cx, cy - 0.020), [count], fontsize=COUNT_FS)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _draw_artr_pt1_panel(ax):
    """ω → ARTR;  v_fwd → pt-IPN1."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _drive_legend(ax, has_propriocep=False)

    # Header — pushed a bit right and down so it doesn't collide with the
    # ω input label or the panel letter (a/b) in the top-left margin.
    _text(ax, (0.22, 0.93), ["afferent populations"],
          fontsize=TITLE_FS, ha="left")

    _draw_afferent_block(ax, (0.04, 0.60), (0.28, 0.17),
                         "ARTR L / R", r"$n=34+42$",
                         sub_lines=["RIPN01/02/03_a/03_b"])
    _draw_afferent_block(ax, (0.04, 0.32), (0.28, 0.17),
                         "pt-IPN1 L / R", r"$n=23+28$",
                         sub_lines=["optic / water flow"])

    _draw_recurrent_block(ax)
    _draw_decoder_block(ax, [r"$[\cos\theta,\sin\theta,\xi]$",
                             r"or $[\cos\theta,\sin\theta,x,y]$"])

    # Input labels + arrows on the LEFT of the afferent boxes.
    _text(ax, (0.03, 0.88), [r"$\omega$"], fontsize=LABEL_FS + 2)
    _arrow(ax, (0.04, 0.85), (0.10, 0.76),
           color=COL_OMEGA, lw=2.2)
    _text(ax, (0.03, 0.26), [r"$v_{\mathrm{fwd}}$"], fontsize=LABEL_FS + 2)
    _arrow(ax, (0.04, 0.28), (0.10, 0.37),
           color=COL_VFWD_EXT, lw=2.2)

    # Afferents → recurrent.
    _arrow(ax, (0.32, 0.69), (0.40, 0.62), color="0.4", lw=1.3)
    _arrow(ax, (0.32, 0.41), (0.40, 0.50), color="0.4", lw=1.3)
    # Recurrent → decoder.
    _arrow(ax, (0.70, 0.54), (0.78, 0.55), color="0.4", lw=1.5)


def _draw_propriocep_panel(ax):
    """ω → ARTR;  v_fwd → pt-IPN1 AND motor_efferent."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _drive_legend(ax, has_propriocep=True)

    _text(ax, (0.22, 0.93), ["afferent populations"],
          fontsize=TITLE_FS, ha="left")

    _draw_afferent_block(ax, (0.04, 0.72), (0.28, 0.13),
                         "ARTR L / R", r"$n=34+42$")
    _draw_afferent_block(ax, (0.04, 0.43), (0.28, 0.16),
                         "pt-IPN1 L / R", r"$n=23+28$",
                         sub_lines=["optic / water flow"])
    _draw_afferent_block(ax, (0.04, 0.15), (0.28, 0.16),
                         "motor_efferent L / R", r"$n=37+31$",
                         sub_lines=["RIPN11+12_a+12_c"])

    _draw_recurrent_block(ax)
    _draw_decoder_block(ax, [r"$[\cos\theta,\sin\theta,x,y]$"])

    # Input labels.
    _text(ax, (0.03, 0.89), [r"$\omega$"], fontsize=LABEL_FS + 2)
    _arrow(ax, (0.04, 0.86), (0.10, 0.79),
           color=COL_OMEGA, lw=2.2)
    _text(ax, (0.03, 0.06), [r"$v_{\mathrm{fwd}}$"], fontsize=LABEL_FS + 2)
    # v_fwd splits into two pathways (extero + propriocep).
    _arrow(ax, (0.04, 0.08), (0.10, 0.49),
           color=COL_VFWD_EXT, lw=2.2,
           connectionstyle="arc3,rad=-0.25")
    _arrow(ax, (0.04, 0.08), (0.10, 0.24),
           color=COL_VFWD_PRO, lw=2.2,
           connectionstyle="arc3,rad=-0.10")

    # Afferents → recurrent.
    _arrow(ax, (0.32, 0.77), (0.40, 0.66), color="0.4", lw=1.3)
    _arrow(ax, (0.32, 0.51), (0.40, 0.55), color="0.4", lw=1.3)
    _arrow(ax, (0.32, 0.25), (0.40, 0.44), color="0.4", lw=1.3)
    # Recurrent → decoder.
    _arrow(ax, (0.70, 0.54), (0.78, 0.55), color="0.4", lw=1.5)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure(out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.04,
                        wspace=0.06)
    _draw_artr_pt1_panel(axes[0])
    _panel_label(axes[0], "a")
    _draw_propriocep_panel(axes[1])
    _panel_label(axes[1], "b")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fig_zebrafish_circuit_variants.png",
    )
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()
    build_figure(args.out)


if __name__ == "__main__":
    main()
