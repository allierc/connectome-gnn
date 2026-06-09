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
# No box outlines; panel labels suppressed (the merged Figure 1 carries
# the a/b/c/d labels through the LaTeX caption only, since the two
# subfigures render at different physical scales and matching matplotlib
# fontsizes across them is unreliable).
TITLE_FS = 12
LABEL_FS = 11
COUNT_FS = 10
SUB_FS = 9
LEGEND_FS = 11

# Drive-type palette — deep, saturated colours for the arrows.
COL_OMEGA    = "#1f6fb3"      # ω (angular velocity)
COL_VFWD_EXT = "#e07b1a"      # v_fwd exteroceptive
COL_VFWD_PRO = "#2a9d3d"      # v_fwd proprioceptive

# Box fills — pastel versions of the partition palette used by
# Figure 1 (the 3-D anatomy render) so the same colour identifies
# the same cell population across the two figures. Each fill is the
# partition's saturated hex code lightened toward white so it doesn't
# fight the text on top.
COL_ARTR_FILL    = "#c7d8e8"   # ARTR (matches #1f6fb3)
COL_PT1_FILL     = "#f5d3b3"   # pt-IPN1 (matches #e07b1a)
COL_ME_FILL      = "#c0e2c5"   # motor_efferent (matches #2a9d3d)
COL_DIPN_FILL    = "#f0dfae"   # dIPN (matches #d49a3a)
COL_IPN12_FILL   = "#e6c4d4"   # IPN12 pool (matches #b15a8e)
COL_IPNCORE_FILL = "#bfe5e2"   # IPN-core (matches #17a39a)
COL_REC_FILL = "#f6e6ce"      # outer recurrent container (pale amber)
COL_DEC_FILL = "#e6efde"      # decoder readout (pale leaf)


def _panel_label(ax, letter: str):
    # Intentional no-op: panel labels are carried by the LaTeX caption.
    pass


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
             (COL_VFWD_EXT, r"$v_{\mathrm{ext}}$")]
    if has_propriocep:
        items.append((COL_VFWD_PRO, r"$\omega_{\mathrm{proprio}}$"))
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
    """Centre column: recurrent network block. Three stacked tiles — the
    dIPN ring, the IPN12 pool, and the IPN-core families — all labelled as
    part of the same recurrent network (917-cell nominal circuit)."""
    cx = 0.61
    # Outer container — neutral pale amber so the three coloured inner tiles
    # (dIPN + IPN12 pool + IPN-core, matching their partition swatches in
    # Figure 1) stand out.
    _box(ax, (0.46, 0.28), (0.30, 0.52), fill=COL_REC_FILL)
    _text(ax, (cx, 0.755), ["recurrent network"], fontsize=TITLE_FS)
    # dIPN tile — amber pastel matching the Figure 1 swatch.
    _box(ax, (0.49, 0.585), (0.24, 0.105), fill=COL_DIPN_FILL, zorder=4)
    _text(ax, (cx, 0.637),
          [r"$d$IPN ($n=408$, $r1\pi$)"],
          fontsize=LABEL_FS)
    # IPN12 tile — rose pastel matching the Figure 1 swatch.
    _box(ax, (0.49, 0.445), (0.24, 0.105), fill=COL_IPN12_FILL, zorder=4)
    _text(ax, (cx, 0.497),
          [r"IPN12 ($n=108$)"],
          fontsize=LABEL_FS)
    # IPN-core tile — teal pastel; the families that joined the ring.
    _box(ax, (0.49, 0.305), (0.24, 0.105), fill=COL_IPNCORE_FILL, zorder=4)
    _text(ax, (cx, 0.357),
          [r"IPN-core ($n=184$)"],
          fontsize=LABEL_FS)


def _draw_decoder_block(ax, out_lines):
    # Larger decoder box so the readout vector sits comfortably inside.
    _box(ax, (0.79, 0.34), (0.20, 0.42), fill=COL_DEC_FILL)
    _text(ax, (0.89, 0.66), ["decoder"], fontsize=TITLE_FS)
    _text(ax, (0.89, 0.52), out_lines, fontsize=LABEL_FS)


def _draw_afferent_block(ax, xy, wh, name, count, sub_lines=None,
                         fill=None):
    """Afferent population box — fill only, no edge stroke.

    `fill` selects the partition pastel that matches the Figure 1 colour
    code (e.g. COL_ARTR_FILL, COL_PT1_FILL, COL_ME_FILL). Defaults to a
    neutral warm grey for any afferent whose role isn't on the production
    gate.

    Three lines stacked top-to-bottom: name, count, optional sub_lines.
    Positions are measured from the box's vertical centre so the layout
    stays balanced as the font sizes scale up.
    """
    _box(ax, xy, wh, fill=fill if fill is not None else "#f4f1ec")
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
    """ω → ARTR;  v_extero → pt-IPN1."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _drive_legend(ax, has_propriocep=False)

    _draw_afferent_block(ax, (0.13, 0.60), (0.28, 0.17),
                         "ARTR L / R", r"$n=35+41$",
                         sub_lines=["RIPN01/02/03_a/03_b"],
                         fill=COL_ARTR_FILL)
    _draw_afferent_block(ax, (0.13, 0.32), (0.28, 0.17),
                         "pt-IPN1 L / R", r"$n=23+28$",
                         sub_lines=["optic / water flow"],
                         fill=COL_PT1_FILL)

    _draw_recurrent_block(ax)
    _draw_decoder_block(ax, [r"$[\cos\theta,\sin\theta,x,y]$"])

    # Input labels + horizontal arrows on the LEFT of the afferent boxes.
    _text(ax, (0.005, 0.685), [r"$\omega$"], fontsize=LABEL_FS + 2,
          ha="left")
    _arrow(ax, (0.055, 0.685), (0.13, 0.685),
           color=COL_OMEGA, lw=2.2)
    _text(ax, (0.005, 0.405), [r"$v_{\mathrm{ext}}$"],
          fontsize=LABEL_FS + 2, ha="left")
    _arrow(ax, (0.085, 0.405), (0.13, 0.405),
           color=COL_VFWD_EXT, lw=2.2)

    # Afferents → recurrent.
    _arrow(ax, (0.41, 0.69), (0.46, 0.62), color="0.4", lw=1.3)
    _arrow(ax, (0.41, 0.41), (0.46, 0.50), color="0.4", lw=1.3)
    # Recurrent → decoder.
    _arrow(ax, (0.76, 0.54), (0.79, 0.55), color="0.4", lw=1.5)


def _draw_propriocep_panel(ax):
    """ω → ARTR;  ω_proprio → motor_efferent (angular companion);
    v_ext → pt-IPN1. The green motor_efferent box sits between the blue
    ARTR and orange pt-IPN1 boxes — both angular drives are at the top."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _drive_legend(ax, has_propriocep=True)

    # ARTR (blue, ω) top — motor_efferent (green, ω_proprio) middle —
    # pt-IPN1 (orange, v_ext) bottom, so the two angular afferents are
    # adjacent and the translation afferent is separated below.
    _draw_afferent_block(ax, (0.13, 0.70), (0.28, 0.15),
                         "ARTR L / R", r"$n=35+41$",
                         sub_lines=["RIPN01/02/03_a/03_b"],
                         fill=COL_ARTR_FILL)
    _draw_afferent_block(ax, (0.13, 0.43), (0.28, 0.16),
                         "motor_efferent L / R", r"$n=36+29$",
                         sub_lines=["RIPN11+12_a+12_c"],
                         fill=COL_ME_FILL)
    _draw_afferent_block(ax, (0.13, 0.15), (0.28, 0.16),
                         "pt-IPN1 L / R", r"$n=23+28$",
                         sub_lines=["optic / water flow"],
                         fill=COL_PT1_FILL)

    _draw_recurrent_block(ax)
    _draw_decoder_block(ax, [r"$[\cos\theta,\sin\theta,x,y]$"])

    # Input labels + horizontal arrows on the LEFT of each afferent box.
    _text(ax, (0.005, 0.775), [r"$\omega$"], fontsize=LABEL_FS + 2,
          ha="left")
    _arrow(ax, (0.055, 0.775), (0.13, 0.775),
           color=COL_OMEGA, lw=2.2)
    _text(ax, (0.005, 0.510), [r"$\omega_{\mathrm{proprio}}$"],
          fontsize=LABEL_FS + 2, ha="left")
    _arrow(ax, (0.105, 0.510), (0.13, 0.510),
           color=COL_VFWD_PRO, lw=2.2)
    _text(ax, (0.005, 0.230), [r"$v_{\mathrm{ext}}$"],
          fontsize=LABEL_FS + 2, ha="left")
    _arrow(ax, (0.085, 0.230), (0.13, 0.230),
           color=COL_VFWD_EXT, lw=2.2)

    # Afferents → recurrent.
    _arrow(ax, (0.41, 0.77), (0.46, 0.66), color="0.4", lw=1.3)
    _arrow(ax, (0.41, 0.51), (0.46, 0.55), color="0.4", lw=1.3)
    _arrow(ax, (0.41, 0.25), (0.46, 0.44), color="0.4", lw=1.3)
    # Recurrent → decoder.
    _arrow(ax, (0.76, 0.54), (0.79, 0.55), color="0.4", lw=1.5)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure(out_path: str, labels=("c", "d")):
    """Render the two-panel circuit-variants diagram.

    `labels` controls the panel letters; the production embeds this
    diagram as the bottom row of a merged anatomy + circuit figure, so
    the panels default to (c) and (d) to follow the anatomy panels
    (a)/(b) above them.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.04,
                        wspace=0.06)
    _draw_artr_pt1_panel(axes[0])
    _panel_label(axes[0], labels[0])
    _draw_propriocep_panel(axes[1])
    _panel_label(axes[1], labels[1])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}  (panel labels: {labels[0]}, {labels[1]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fig_zebrafish_circuit_variants.png",
    )
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--labels", nargs=2, default=("c", "d"),
                    help="Panel letter labels (default: c d, matching the "
                         "merged anatomy + circuit figure)")
    args = ap.parse_args()
    build_figure(args.out, labels=tuple(args.labels))


if __name__ == "__main__":
    main()
