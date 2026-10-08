"""Fig. 2: the hybrid Flyvis-FlyWire connectome figure, with panels e and f
redrawn from the general-form GNN (group lasso 25) of experiment 7.

    python scripts/fig_flywire_hybrid.py

Panels a-d are the published rendering (figures/fig_flywire_hybrid.png: the
3-D coregistration, the per-column FlyWire kernels and the FlyWire-eye vs
Flyvis simulation rollouts have no script in this repository) cropped from
that PNG. Panels e and f are drawn here: the GNN's 8,000-frame rollout
(black) against the simulated voltage (green) for the six example cell types
of the published figure over a 20 s window, on the FlyWire eye (e) and on the
FlyWire eye with proximal null edges (f), fold cv00 of experiment 7; the
Fisher-z pooled Pearson r over all neurons and frames is printed.
Data: <GNN_OUTPUT_ROOT>/log/fly/full_eye_flywireRF[_proximal_nulls]_noise_005_blank50_condl25_cv00/results/rollout_bundle.npz

Output: figures/fig_flywire_hybrid_new.{pdf,png}
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (CM, COLOR_GT, COLOR_PRED, FIG_DIR, FS_ANNOT, FS_LABEL, FS_PANEL, FS_TICK, LOG_ROOT,  # noqa: E402
                         fisher_r, save)
from fig_rollout_3col import T0, T1, DT_MS  # noqa: E402

PUBLISHED = os.path.join(FIG_DIR, "fig_flywire_hybrid.png")
PANELS = [
    ("e", "FlyWire eye model inferred", "full_eye_flywireRF_noise_005_blank50_condl25_cv00",
     "total: 50,412 neurons, 1,266,378 edges"),
    ("f", "FlyWire eye + null edges (n.e.)", "full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv00",
     "total: 50,412 neurons, 9,642,335 edges"),
]
TYPES = ["T5a", "Mi9", "Mi1", "Tm9", "T4c", "Tm3"]          # the published panels' examples, top to bottom
ROW_SPLIT = 0.675                                            # fraction of the PNG height where row d-f starts
D_SPLIT = 0.335                                              # fraction of the PNG width panel d spans
WIDTH = 18.0


# fonts of the published PNG once scaled to the page: its titles are about 4.5 pt,
# tick labels 4 pt, letters 7 pt bold -- the new panels use the same sizes
FS_T, FS_K = 4.9, 4.4   # panel d's title and labels at page scale (Arial there, DejaVu Sans here: ~12% larger glyphs)
# PANEL d's GRID, measured on the published PNG (rows): the title line, the six cell-type labels
# T5a .. Tm3 (60 px apart), and the trace x-extent within the d crop -- e and f are laid on the same lines
D_TITLE_ROW = 1208
D_LABEL_ROWS = [1263, 1323, 1384, 1444, 1504, 1564]
D_LABEL_COL_RIGHT = 205                                      # right edge of d's labels (px)
D_TRACE_COLS = (208, 1385)
# THE PANEL LETTERS ARE REDRAWN, all six at Fig. 1's size (FS_PANEL, 7.5 pt bold): the published PNG's own
# a-d are painted over at the pixel boxes measured on it (rows, cols), so the figure's letters are uniform.
# (top row, baseline row, left col, right col) of each published letter, measured as the largest dark
# component at an anti-aliasing threshold of 230 so the grey fringe is inside the box. "b" reaches row 0.
PNG_LETTERS = {"a": (19, 72, 8, 56), "b": (0, 72, 1496, 1545), "c": (19, 72, 2877, 2926), "d": (1160, 1232, 6, 56)}
# the "time" text and arrow of panel d, as fractions of the PNG height from its top (measured): the new
# panels' "time" and arrow are placed on the same lines
TIME_TEXT_Y, TIME_ARROW_Y = 0.9665, 0.9860


def traces(ax, b, idx, labels, step, title, footer, r, sd):
    t = np.arange(T0, T1 + 1) * DT_MS
    n = len(idx)
    for k, i in enumerate(idx):
        off = (n - 1 - k) * step
        tr, pr = b["activity_true"][i, T0:T1 + 1], b["activity_pred"][i, T0:T1 + 1]
        base = tr.mean()
        ax.plot(t, tr - base + off, color=COLOR_GT, lw=0.8)
        ax.plot(t, pr - base + off, color=COLOR_PRED, lw=0.35)
    ax.set_yticks([(n - 1 - k) * step for k in range(n)]); ax.set_yticklabels(labels, fontsize=FS_K)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(t[0], t[-1]); ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
    # the axes span exactly half a row beyond the first and last label row, so trace k's offset lands on
    # d's label row k
    ax.set_ylim(-0.5 * step, (n - 0.5) * step)


def main():
    img = np.asarray(Image.open(PUBLISHED).convert("RGB")).copy()
    H, W = img.shape[:2]
    for r0, r1, c0, c1 in PNG_LETTERS.values():
        # clamped at 0: a negative start would index from the END and erase nothing (the ghost "b")
        img[max(0, r0 - 3):r1 + 4, max(0, c0 - 3):c1 + 4] = 255
    top = img[: int(ROW_SPLIT * H)]
    d_panel = img[int(ROW_SPLIT * H):, : int(D_SPLIT * W)]
    top_h = WIDTH * top.shape[0] / top.shape[1]                 # cm, at full width
    d_w = WIDTH * D_SPLIT
    row_h = d_w * d_panel.shape[0] / d_panel.shape[1]
    gap, bottom = 0.4, 0.7
    height = bottom + row_h + gap + top_h + 0.2
    fig = plt.figure(figsize=(WIDTH * CM, height * CM))

    def row_cm(row):
        """A row of the published PNG's bottom strip, in figure cm from the bottom."""
        return bottom + row_h - (row - ROW_SPLIT * H) * WIDTH / W

    def rect(x, y, w, h):
        return [x / WIDTH, y / height, w / WIDTH, h / height]

    ax = fig.add_axes(rect(0, height - 0.2 - top_h, WIDTH, top_h)); ax.imshow(top); ax.set_axis_off()
    ax = fig.add_axes(rect(0, bottom, d_w, row_h)); ax.imshow(d_panel); ax.set_axis_off()
    # redrawn on the published letters' BASELINE (a, c have no ascender, b has one, so their tops differ)
    for letter, (r0, r1, c0, c1) in PNG_LETTERS.items():
        px = c0 * WIDTH / W
        py = (height - 0.2 - r1 * WIDTH / W) if r1 < ROW_SPLIT * H else (bottom + row_h - (r1 - ROW_SPLIT * H) * WIDTH / W)
        fig.text(px / WIDTH, py / height, letter, fontsize=FS_PANEL, fontweight="bold", ha="left", va="baseline")
    GAP_EF = 1.4                     # between e and f: room for the letter and "Examples"
    x = d_w + 1.2
    pw = (WIDTH - x - 0.3 - GAP_EF) / 2
    for k, (letter, title, run, footer) in enumerate(PANELS):
        b = np.load(os.path.join(LOG_ROOT, run, "results", "rollout_bundle.npz"), allow_pickle=True)
        names = list(b["type_names"]); types = b["type_ids"]
        idx = [int(np.where(types == names.index(nm))[0][0]) for nm in TYPES]
        step = 2.5 * float(np.std(b["activity_true"][idx, T0:T1]))
        r, sd = fisher_r(b["activity_true"], b["activity_pred"])
        sp = D_LABEL_ROWS[1] - D_LABEL_ROWS[0]
        y_top, y_bot = row_cm(D_LABEL_ROWS[0] - sp / 2), row_cm(D_LABEL_ROWS[-1] + sp / 2)
        axp = fig.add_axes(rect(x + k * (pw + GAP_EF), y_bot, pw, y_top - y_bot))
        traces(axp, b, idx, TYPES, step, title, footer, r, sd)
        # title, "Examples" and r on d's title line
        yt = row_cm(D_TITLE_ROW) / height
        # as in d: "Examples" over the labels, the title starting a little into the traces
        fig.text((x + k * (pw + GAP_EF) + 0.55) / WIDTH, yt, title, ha="left", va="center", fontsize=FS_T)
        fig.text((x + k * (pw + GAP_EF) + 0.02) / WIDTH, yt, "Examples", ha="right", va="center", fontsize=FS_K)
        fig.text((x + k * (pw + GAP_EF) + pw) / WIDTH, yt, f"$r$ = {r:.2f} $\\pm$ {sd:.2f}", ha="right", va="center",
                 fontsize=FS_K)
        xl = x + k * (pw + GAP_EF)
        y_base = bottom + row_h - (PNG_LETTERS["d"][1] - ROW_SPLIT * H) * WIDTH / W
        fig.text((xl - 0.85) / WIDTH, y_base / height, letter, fontsize=FS_PANEL, fontweight="bold", ha="right",
                 va="baseline")
        # "time ->" on panel d's lines
        y_txt = bottom + row_h - (TIME_TEXT_Y * H - ROW_SPLIT * H) * WIDTH / W
        y_arr = bottom + row_h - (TIME_ARROW_Y * H - ROW_SPLIT * H) * WIDTH / W
        fig.text(xl / WIDTH, y_txt / height, "time", ha="left", va="center", fontsize=FS_K)
        fig.add_artist(FancyArrowPatch((xl / WIDTH, y_arr / height), ((xl + 0.45) / WIDTH, y_arr / height),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=4,
                                       lw=0.6, color="k", shrinkA=0, shrinkB=0))
        # the totals on the "time" line, centred under the panel; the legend between e and f on that line
        fig.text((xl + 0.55) / WIDTH, y_txt / height, footer, ha="left", va="center", fontsize=FS_K)
        print(f"{letter} {run}: r {r:.3f} +- {sd:.3f}")
    # one "-- simulated  -- inferred" legend between e and f, on the footer line (the published strip),
    # as a legend of two proxy lines so the swatches sit at the text height
    xr = (x + pw + GAP_EF - 0.12) / WIDTH          # right-aligned just before panel f's "time" arrow
    yl = (bottom + row_h - (TIME_TEXT_Y * H - ROW_SPLIT * H) * WIDTH / W + 0.1) / height
    fig.legend(handles=[plt.Line2D([], [], color=COLOR_GT, lw=1.2, label="simulated"),
                        plt.Line2D([], [], color=COLOR_PRED, lw=1.2, label="inferred")],
               loc="upper right", bbox_to_anchor=(xr, yl), ncol=2, fontsize=FS_K, frameon=False,
               handlelength=1.2, handletextpad=0.3, columnspacing=0.7, borderaxespad=0)
    save(fig, os.path.join(FIG_DIR, "fig_flywire_hybrid_new"))


if __name__ == "__main__":
    main()
