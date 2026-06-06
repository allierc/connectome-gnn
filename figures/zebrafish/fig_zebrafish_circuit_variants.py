"""fig_zebrafish_circuit_variants.py
=====================================

Schematic of the three afferent-partition variants of the zebrafish HD /
IPN circuit. Same connectome in every variant (839 cells: 443 dIPN + 108
IPN12 + 200 RIPN + 88 pt-IPN); what changes is **which afferent
subpopulations the velocity gate routes ω and v_fwd through**.

Three panels (a–c), each laid out left-to-right:
    afferent populations ─→ bump ring ─→ decoder readout

  (a) baseline: RIPN and pt-IPN aggregates both receive ω (4-scalar
      gate; translation conflated with rotation).
  (b) refined partition: ω routed to ARTR only (RIPN01+02+03_a+03_b);
      v_fwd routed to pt-IPN1 only (optic / water-flow afferents);
      rotation and translation drive paths cleanly separated.
  (c) proprioception extension: same as (b) plus a third afferent
      route — v_fwd is now delivered through TWO parallel pathways,
      pt-IPN1 (exteroceptive) AND motor_efferent (proprioceptive copy
      of the swim command: RIPN11 + RIPN12_a + RIPN12_c).

Decoder readout is whatever the active sub-task supervises: heading
(2-col), heading + scalar displacement (3-col), or heading + 2D
position (4-col).

Usage
-----
    python figures/zebrafish/fig_zebrafish_circuit_variants.py
    python figures/zebrafish/fig_zebrafish_circuit_variants.py --out my.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- Style ------------------------------------------------------------------
PANEL_LABEL_FS = 16
TITLE_FS = 12
LABEL_FS = 10
TICK_FS = 9
SUB_FS = 8

# Drive-type colours (used to identify ω vs v_fwd pathways).
COL_OMEGA = "#1f77b4"          # blue   — ω (angular velocity)
COL_VFWD_EXT = "#ff7f0e"       # orange — v_fwd exteroceptive
COL_VFWD_PRO = "#2ca02c"       # green  — v_fwd proprioceptive

# Box fills.
BOX_AFF = "#e8eef7"            # afferent populations (pale blue)
BOX_BUMP = "#fde9c8"           # bump ring (pale orange)
BOX_DEC = "#dcefdb"            # decoder (pale green)
BOX_EDGE = "#1a1a1a"


def _panel_label(ax, letter: str):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes,
            fontsize=PANEL_LABEL_FS, fontweight="bold",
            va="bottom", ha="right")


def _box(ax, xy, wh, text, fill=BOX_AFF, edge=BOX_EDGE, fontsize=LABEL_FS,
         lw=1.0):
    """Rounded rectangle with centred text. xy is bottom-left."""
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        facecolor=fill, edgecolor=edge, lw=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize)
    return patch


def _arrow(ax, xy_from, xy_to, color="0.2", lw=1.2, ls="-",
           connectionstyle="arc3,rad=0.0"):
    """Curved arrow from a → b."""
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle="-|>", mutation_scale=12,
        color=color, lw=lw, linestyle=ls,
        connectionstyle=connectionstyle, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)


def _drive_legend(ax, has_propriocep: bool):
    """Tiny inset legend for arrow colours, bottom-right of the panel.

    Positioned in the lower-right margin so it never collides with the
    top-left ω input label or the centre/right network boxes.
    """
    items = [(COL_OMEGA,    r"$\omega$"),
             (COL_VFWD_EXT, r"$v_{\mathrm{fwd}}$ (extero)")]
    if has_propriocep:
        items.append((COL_VFWD_PRO,
                       r"$v_{\mathrm{fwd}}$ (propriocep)"))
    x0, y0, dy = 0.62, 0.14, -0.045
    for i, (col, lab) in enumerate(items):
        ax.plot([x0, x0 + 0.05], [y0 + i * dy, y0 + i * dy],
                color=col, lw=2.5, solid_capstyle="round",
                transform=ax.transAxes)
        ax.text(x0 + 0.06, y0 + i * dy, lab, transform=ax.transAxes,
                fontsize=SUB_FS, va="center", ha="left")


def _draw_bump_and_decoder(ax, out_text: str):
    """Shared center + right columns for every variant.

    Center column: bump ring + IPN12 (the recurrent network — same in
    every variant).
    Right column: decoder readout, labelled by the active sub-task
    output columns.
    Returns (bump_center, decoder_center) coordinates for arrow targets.
    """
    bump_center = (0.55, 0.55)
    decoder_center = (0.86, 0.55)

    # Bump ring box (big rectangle wrapping the schematic ring).
    _box(ax, (0.42, 0.30), (0.26, 0.50),
         text="", fill=BOX_BUMP, fontsize=LABEL_FS)
    ax.text(bump_center[0], bump_center[1] + 0.20,
            "recurrent bump network",
            ha="center", va="center", fontsize=LABEL_FS, fontweight="bold")
    # Schematic ring — circle of dots inside the box.
    theta = np.linspace(0, 2 * np.pi, 22)
    rx = 0.07
    ry = 0.10
    ring_xs = bump_center[0] + rx * np.cos(theta)
    ring_ys = bump_center[1] + ry * np.sin(theta)
    ax.plot(ring_xs, ring_ys, "o-", color="#cc7a00", ms=3, lw=0.8,
            mec="#7a4d00", mfc="#f0a040")
    ax.text(bump_center[0], bump_center[1] - 0.18,
            r"$d$IPN ring ($n{=}443$) + IPN12 ($n{=}108$)",
            ha="center", va="center", fontsize=SUB_FS)

    # Decoder box (right).
    _box(ax, (0.78, 0.40), (0.20, 0.30),
         text=out_text, fill=BOX_DEC, fontsize=LABEL_FS)
    ax.text(decoder_center[0], decoder_center[1] + 0.18,
            "decoder readout",
            ha="center", va="center", fontsize=LABEL_FS, fontweight="bold")

    # Bump → decoder arrow.
    _arrow(ax, (bump_center[0] + 0.10, bump_center[1]),
           (decoder_center[0] - 0.08, decoder_center[1]),
           color="0.25", lw=1.4)
    return bump_center, decoder_center


def _draw_v1_panel(ax):
    """Baseline: ω goes through BOTH RIPN and pt-IPN aggregates."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("baseline circuit\n(aggregate RIPN + pt-IPN)",
                 fontsize=TITLE_FS, pad=8)
    _drive_legend(ax, has_propriocep=False)

    # Afferent boxes (left column).
    _box(ax, (0.04, 0.62), (0.30, 0.16),
         text="RIPN_L / RIPN_R\n($n{=}102+101$)",
         fill=BOX_AFF, fontsize=LABEL_FS)
    _box(ax, (0.04, 0.36), (0.30, 0.16),
         text="ptIPN_L / ptIPN_R\n($n{=}41+44$)",
         fill=BOX_AFF, fontsize=LABEL_FS)
    ax.text(0.19, 0.81, "afferent populations",
            ha="center", va="bottom", fontsize=LABEL_FS, fontweight="bold")

    bump_center, _ = _draw_bump_and_decoder(
        ax, out_text=r"$[\cos\theta,\sin\theta]$")

    # Input arrows.
    ax.text(0.04, 0.92, r"$\omega$", color=COL_OMEGA, fontsize=14,
            fontweight="bold", ha="center", va="center")
    _arrow(ax, (0.04, 0.90), (0.10, 0.79), color=COL_OMEGA, lw=2.0)
    _arrow(ax, (0.04, 0.90), (0.10, 0.53), color=COL_OMEGA, lw=2.0,
           connectionstyle="arc3,rad=-0.3")

    # Afferents → bump.
    _arrow(ax, (0.34, 0.70), (0.42, 0.62), color="0.4", lw=1.2)
    _arrow(ax, (0.34, 0.44), (0.42, 0.50), color="0.4", lw=1.2)


def _draw_artr_pt1_panel(ax):
    """Refined: ω → ARTR; v_fwd → pt-IPN1."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("refined afferent partition\n"
                 "($\\omega$ → ARTR; $v_{\\mathrm{fwd}}$ → pt-IPN1)",
                 fontsize=TITLE_FS, pad=8)
    _drive_legend(ax, has_propriocep=False)

    # Afferent boxes.
    _box(ax, (0.04, 0.62), (0.30, 0.16),
         text="ARTR_L / ARTR_R\n($n{=}34+42$)\nRIPN01/02/03_a/03_b",
         fill=BOX_AFF, fontsize=LABEL_FS - 1)
    _box(ax, (0.04, 0.36), (0.30, 0.16),
         text="pt-IPN1_L / pt-IPN1_R\n($n{=}23+28$)\noptic / water flow",
         fill=BOX_AFF, fontsize=LABEL_FS - 1)
    ax.text(0.19, 0.81, "afferent populations",
            ha="center", va="bottom", fontsize=LABEL_FS, fontweight="bold")

    bump_center, _ = _draw_bump_and_decoder(
        ax, out_text=r"$[\cos\theta,\sin\theta,\xi]$"
                     "\n" r"or $[\cos\theta,\sin\theta,x,y]$")

    # Input arrows (separated by drive type now).
    ax.text(0.04, 0.92, r"$\omega$", color=COL_OMEGA, fontsize=14,
            fontweight="bold", ha="center", va="center")
    _arrow(ax, (0.04, 0.90), (0.10, 0.79), color=COL_OMEGA, lw=2.0)
    ax.text(0.04, 0.30, r"$v_{\mathrm{fwd}}$", color=COL_VFWD_EXT,
            fontsize=14, fontweight="bold", ha="center", va="center")
    _arrow(ax, (0.04, 0.32), (0.10, 0.41), color=COL_VFWD_EXT, lw=2.0)

    # Afferents → bump.
    _arrow(ax, (0.34, 0.70), (0.42, 0.62), color="0.4", lw=1.2)
    _arrow(ax, (0.34, 0.44), (0.42, 0.50), color="0.4", lw=1.2)


def _draw_propriocep_panel(ax):
    """Proprioception: ω → ARTR; v_fwd → pt-IPN1 AND motor_efferent."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("proprioception-extended\n"
                 "(dual extero + propriocep $v_{\\mathrm{fwd}}$ path)",
                 fontsize=TITLE_FS, pad=8)
    _drive_legend(ax, has_propriocep=True)

    # Afferent boxes — three of them now.
    _box(ax, (0.04, 0.72), (0.30, 0.13),
         text="ARTR_L / ARTR_R\n($n{=}34+42$)",
         fill=BOX_AFF, fontsize=LABEL_FS - 1)
    _box(ax, (0.04, 0.47), (0.30, 0.13),
         text="pt-IPN1_L / pt-IPN1_R\n($n{=}23+28$)  optic/water flow",
         fill=BOX_AFF, fontsize=LABEL_FS - 1)
    _box(ax, (0.04, 0.22), (0.30, 0.13),
         text="motor_efferent_L/R ($n{=}37+31$)\nRIPN11+12_a+12_c (efference copy)",
         fill=BOX_AFF, fontsize=LABEL_FS - 1)
    ax.text(0.19, 0.88, "afferent populations",
            ha="center", va="bottom", fontsize=LABEL_FS, fontweight="bold")

    bump_center, _ = _draw_bump_and_decoder(
        ax, out_text=r"$[\cos\theta,\sin\theta,x,y]$")

    # Input arrows (split into 3 drive types).
    ax.text(0.04, 0.94, r"$\omega$", color=COL_OMEGA, fontsize=14,
            fontweight="bold", ha="center", va="center")
    _arrow(ax, (0.04, 0.92), (0.10, 0.85), color=COL_OMEGA, lw=2.0)
    # v_fwd splits into two pathways.
    ax.text(0.04, 0.06, r"$v_{\mathrm{fwd}}$", color="0.25",
            fontsize=14, fontweight="bold", ha="center", va="center")
    _arrow(ax, (0.04, 0.08), (0.10, 0.52),
           color=COL_VFWD_EXT, lw=2.0,
           connectionstyle="arc3,rad=-0.3")
    _arrow(ax, (0.04, 0.08), (0.10, 0.27),
           color=COL_VFWD_PRO, lw=2.0,
           connectionstyle="arc3,rad=-0.15")

    # Afferents → bump.
    _arrow(ax, (0.34, 0.78), (0.42, 0.65), color="0.4", lw=1.2)
    _arrow(ax, (0.34, 0.54), (0.42, 0.55), color="0.4", lw=1.2)
    _arrow(ax, (0.34, 0.28), (0.42, 0.45), color="0.4", lw=1.2)


def build_figure(out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.0))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.05,
                        wspace=0.06)
    _draw_v1_panel(axes[0])
    _panel_label(axes[0], "a")
    _draw_artr_pt1_panel(axes[1])
    _panel_label(axes[1], "b")
    _draw_propriocep_panel(axes[2])
    _panel_label(axes[2], "c")

    fig.suptitle(
        "zebrafish HD / IPN circuit variants — same 839-cell connectome, "
        "three afferent partitions",
        fontsize=13, fontweight="bold", y=0.99,
    )
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
