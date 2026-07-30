"""fig_cx_circuit_variants.py
=====================================

Circuit-partition schematic for the *Drosophila* central complex,
rendered in the SAME template / design / fonts / colours / per-function
panel labels as the zebrafish Figure 1 circuit panels
(figures/zebrafish/fig_zebrafish_circuit_variants.py). Self-contained:
all drawing primitives live in this single file.

Two panels (c, d), each laid out left-to-right:
    afferent populations ─→ recurrent network ─→ decoder readout

  (c) Heading integration (implemented). The angular-velocity drive
      ω = θ̇ enters through the PEN_a / PEN_b gate; the recurrent HD ring
      (EPG, EPGt, PEG, Δ7, ER6) maintains the bump; the heading readout
      θ = ∫ω is taken from EPG.

  (d) Path integration (proposed extension, dashed). Same heading
      pathway, plus a forward-velocity drive v_fwd = ẋ entering through a
      new PFN_d / PFN_v gate, multiplicatively gated by the EPG heading.
      Downstream vector cells (hΔB, PFR, PFN_a) accumulate the
      head-direction-rotated translation, giving the forward-distance
      readout d = ∫v_fwd (from PFN_a) and the allocentric position
      (x, y) = ∫ v_fwd e^{iθ} (from hΔB / PFR).

The full cell-type roster of the proposed circuit is
    {EPG, EPGt, PEG, ER6, PEN_a, PEN_b, Δ7, PFN_d, PFN_v, PFN_a, hΔB, PFR}.
Boxes drawn with a dashed outline are *not yet implemented*.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- Style ------------------------------------------------------------------
# Mirrors fig_zebrafish_circuit_variants.py exactly.
TITLE_FS = 12
LABEL_FS = 11
COUNT_FS = 10
SUB_FS = 9
LEGEND_FS = 11

PANEL_LABEL_FS = 13
PANEL_LABEL_KW = dict(fontsize=PANEL_LABEL_FS, fontweight="bold",
                      va="top", ha="left")

# --- Cell-type palette ------------------------------------------------------
# Saturated hexes match the tab10 ordering used by the 3-D anatomy render
# (figures/drosophila_cx/fig_cx_anatomy_3d.py) so the same colour identifies
# the same population across panels (a, b) and (c, d):
#   EPG, EPGt, PEN_a, PEN_b, Delta7, PEG, ER6.
COL_EPG  = "#1f77b4"
COL_EPGt = "#ff7f0e"
COL_PENa = "#2ca02c"
COL_PENb = "#d62728"
COL_D7   = "#9467bd"
COL_PEG  = "#8c564b"
COL_ER6  = "#e377c2"
# Proposed path-integration cell types (not in the 156-cell connectome).
COL_PFNd = "#17becf"
COL_PFNv = "#1f9e89"
COL_PFNa = "#7f7f7f"
COL_hDB  = "#393b79"
COL_PFR  = "#8c6d31"

# Drive-arrow colours: each drive is tinted like the afferent it feeds.
COL_OMEGA = COL_PENa      # ω  → PEN gate (green)
COL_VFWD  = COL_PFNd      # v_fwd → PFN gate (cyan)


def _pastel(hexcol: str, f: float = 0.75) -> tuple:
    """Lighten a colour toward white by fraction f (box fills sit under text).

    f = 0.75 reproduces the zebrafish fills, e.g. #1f6fb3 -> #c7d8e8.
    """
    r, g, b = to_rgb(hexcol)
    return (r + (1.0 - r) * f, g + (1.0 - g) * f, b + (1.0 - b) * f)


# ---------------------------------------------------------------------------
# Primitives (verbatim style port of the zebrafish script)
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


def _box(ax, xy, wh, *, fill, zorder=3, dashed=False):
    """Filled rounded rectangle with a soft drop shadow.

    `dashed=True` adds a grey dashed outline to flag a *proposed*
    (not-yet-implemented) population, keeping the same fill / fonts.
    """
    x, y = xy; w, h = wh
    _shadow(ax, xy, wh)
    edge = "0.35" if dashed else "none"
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.014",
        facecolor=fill, edgecolor=edge,
        linewidth=1.1 if dashed else 0,
        linestyle=(0, (4, 2)) if dashed else "-",
        zorder=zorder,
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


def _text(ax, xy, lines, fontsize=LABEL_FS, va="center", ha="center"):
    """Multi-line text helper. No bold, no stroke effects."""
    x, y = xy
    ax.text(x, y, "\n".join(lines), ha=ha, va=va,
            fontsize=fontsize, zorder=5)


def _panel_letter(ax, letter):
    ax.text(0.01, 0.99, letter, transform=ax.transAxes, **PANEL_LABEL_KW)


def _afferent_block(ax, xy, wh, name, count, sub_lines=None,
                    fill=None, dashed=False):
    """Afferent population box — fill only (or dashed if proposed).

    Three lines stacked around the box centre: name, count, optional sub.
    """
    _box(ax, xy, wh, fill=fill if fill is not None else "#f4f1ec",
         dashed=dashed)
    cx = xy[0] + wh[0] / 2
    cy = xy[1] + wh[1] / 2
    if sub_lines:
        _text(ax, (cx, cy + 0.040), [name], fontsize=LABEL_FS)
        _text(ax, (cx, cy + 0.005), [count], fontsize=COUNT_FS)
        _text(ax, (cx, cy - 0.040), sub_lines, fontsize=SUB_FS)
    else:
        _text(ax, (cx, cy + 0.018), [name], fontsize=LABEL_FS)
        _text(ax, (cx, cy - 0.020), [count], fontsize=COUNT_FS)


def _tile(ax, xy, wh, lines, fill, dashed=False):
    """Inner recurrent-network tile (one population group)."""
    _box(ax, xy, wh, fill=fill, zorder=4, dashed=dashed)
    cx = xy[0] + wh[0] / 2
    cy = xy[1] + wh[1] / 2
    _text(ax, (cx, cy), lines, fontsize=LABEL_FS)


# ---------------------------------------------------------------------------
# Panel (c): heading integration (implemented)
# ---------------------------------------------------------------------------

def _draw_heading_panel(ax):
    """ω → PEN_a/PEN_b;  recurrent HD ring;  heading readout θ = ∫ω."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Afferent gate: PEN_a / PEN_b (centred on the recurrent block).
    _afferent_block(ax, (0.13, 0.455), (0.28, 0.18),
                    "PEN$_a$ / PEN$_b$", r"$n=20+22$",
                    sub_lines=["angular-velocity gate"],
                    fill=_pastel(COL_PENa))

    # Recurrent network: HD ring substrate, three population tiles.
    _box(ax, (0.46, 0.28), (0.30, 0.52), fill="#f6e6ce")  # pale-amber container
    _text(ax, (0.61, 0.755), ["recurrent network"], fontsize=TITLE_FS)
    _tile(ax, (0.49, 0.585), (0.24, 0.105),
          [r"EPG / EPGt ($n=50$)"], _pastel(COL_EPG))
    _tile(ax, (0.49, 0.445), (0.24, 0.105),
          [r"PEG ($n=18$)"], _pastel(COL_PEG))
    _tile(ax, (0.49, 0.305), (0.24, 0.105),
          [r"$\Delta$7 / ER6 ($n=46$)"], _pastel(COL_D7))

    # Decoder: heading only.
    _box(ax, (0.79, 0.34), (0.20, 0.40), fill="#e6efde")  # pale-leaf
    _text(ax, (0.89, 0.685), ["decoder"], fontsize=TITLE_FS)
    _text(ax, (0.89, 0.575), [r"$\theta=\int\omega$"], fontsize=LABEL_FS)
    _text(ax, (0.89, 0.490), [r"$[\cos\theta,\sin\theta]$"], fontsize=SUB_FS)

    # Input drive on the LEFT (ω ≡ dθ/dt, integrated into heading).
    _text(ax, (0.005, 0.545), [r"$\omega\!=\!\dot\theta$"],
          fontsize=LABEL_FS + 2, ha="left")
    _arrow(ax, (0.075, 0.545), (0.13, 0.545), color=COL_OMEGA, lw=2.2)
    # Afferent → recurrent → decoder.
    _arrow(ax, (0.41, 0.545), (0.46, 0.55), color="0.4", lw=1.3)
    _arrow(ax, (0.76, 0.54), (0.79, 0.54), color="0.4", lw=1.5)


# ---------------------------------------------------------------------------
# Panel (d): path integration (proposed extension)
# ---------------------------------------------------------------------------

def _draw_pathint_panel(ax):
    """ω → PEN;  v_fwd → PFN (proposed);  vector cells;  (θ, d, x, y) readout."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Afferent gates: PEN (ω, implemented) top, PFN (v_fwd, proposed) bottom.
    _afferent_block(ax, (0.13, 0.62), (0.28, 0.16),
                    "PEN$_a$ / PEN$_b$", r"$n=20+22$",
                    sub_lines=["angular-velocity gate"],
                    fill=_pastel(COL_PENa))
    _afferent_block(ax, (0.13, 0.28), (0.28, 0.16),
                    "PFN$_d$ / PFN$_v$", "proposed",
                    sub_lines=["forward-velocity gate"],
                    fill=_pastel(COL_PFNd), dashed=True)

    # Recurrent network: HD ring (3 tiles) + proposed vector accumulator.
    _box(ax, (0.46, 0.205), (0.30, 0.625), fill="#f6e6ce")
    _text(ax, (0.61, 0.785), ["recurrent network"], fontsize=TITLE_FS)
    _tile(ax, (0.49, 0.645), (0.24, 0.090),
          [r"EPG / EPGt ($n=50$)"], _pastel(COL_EPG))
    _tile(ax, (0.49, 0.540), (0.24, 0.090),
          [r"PEG ($n=18$)"], _pastel(COL_PEG))
    _tile(ax, (0.49, 0.435), (0.24, 0.090),
          [r"$\Delta$7 / ER6 ($n=46$)"], _pastel(COL_D7))
    _tile(ax, (0.49, 0.245), (0.24, 0.150),
          [r"h$\Delta$B / PFR / PFN$_a$", "vector accumulator", "(proposed)"],
          _pastel(COL_hDB), dashed=True)

    # Decoder: heading + forward distance + 2-D position.
    _box(ax, (0.79, 0.235), (0.20, 0.56), fill="#e6efde")
    _text(ax, (0.89, 0.745), ["decoder"], fontsize=TITLE_FS)
    dec = [r"$\theta=\int\omega$",
           r"$d=\int v_{\mathrm{fwd}}$",
           r"$(x,y)=\int v_{\mathrm{fwd}}\,e^{i\theta}$"]
    y0, dy = 0.640, -0.082
    for i, ln in enumerate(dec):
        _text(ax, (0.89, y0 + i * dy), [ln], fontsize=LABEL_FS)
    _text(ax, (0.89, y0 + len(dec) * dy - 0.003),
          [r"$[\cos\theta,\sin\theta,x,y]$"], fontsize=SUB_FS)

    # Input drives on the LEFT.
    _text(ax, (0.005, 0.70), [r"$\omega\!=\!\dot\theta$"],
          fontsize=LABEL_FS + 2, ha="left")
    _arrow(ax, (0.075, 0.70), (0.13, 0.70), color=COL_OMEGA, lw=2.2)
    _text(ax, (0.005, 0.36), [r"$v_{\mathrm{fwd}}\!=\!\dot x$"],
          fontsize=LABEL_FS + 2, ha="left")
    _arrow(ax, (0.085, 0.36), (0.13, 0.36), color=COL_VFWD, lw=2.2,
           ls=(0, (4, 2)))

    # Afferent → recurrent.
    _arrow(ax, (0.41, 0.70), (0.46, 0.66), color="0.4", lw=1.3)
    _arrow(ax, (0.41, 0.36), (0.46, 0.32), color="0.4", lw=1.3,
           ls=(0, (4, 2)))
    # EPG heading multiplicatively gates the PFN translation channel.
    _arrow(ax, (0.49, 0.66), (0.41, 0.44), color=COL_EPG, lw=1.3,
           ls=(0, (4, 2)), connectionstyle="arc3,rad=0.25")
    _text(ax, (0.435, 0.555), ["heading"], fontsize=SUB_FS, ha="center")
    # Recurrent → decoder.
    _arrow(ax, (0.76, 0.52), (0.79, 0.52), color="0.4", lw=1.5)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure(out_path: str, labels=("c", "d")):
    """Render the two-panel CX circuit-variants diagram.

    Defaults to panel letters (c, d): the production embeds this diagram
    as the bottom row of Figure 1, below the anatomy panels (a, b).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.04,
                        wspace=0.06)
    _draw_heading_panel(axes[0])
    _panel_letter(axes[0], labels[0])
    _draw_pathint_panel(axes[1])
    _panel_letter(axes[1], labels[1])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}  (panel labels: {labels[0]}, {labels[1]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fig_cx_circuit_variants.png",
    )
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--labels", nargs=2, default=("c", "d"),
                    help="Panel letter labels (default: c d)")
    args = ap.parse_args()
    build_figure(args.out, labels=tuple(args.labels))


if __name__ == "__main__":
    main()
