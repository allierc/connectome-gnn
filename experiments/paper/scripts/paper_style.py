"""Shared style for the paper's figures, following the published figures:
full left/bottom spines (never trimmed), three ticks per axis spanning the
panel's fixed range, annotations top-left inside the panel, bold panel letters
outside top-left, 65-colour cell-type map.

Geometry: every full-width figure is drawn at 18 cm (included at the 13.97 cm
NeurIPS text width, scale 0.78), with labels 8 pt and ticks / annotations 6 pt.
"""
import os
import re
import string

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm

CM = 1.0 / 2.54
TEXT_W_CM = 18.0
FS_LABEL = 8
FS_TICK = 6
FS_ANNOT = 5.5
FS_PANEL = 7.5                # panel letters (bold), the same in every figure
LW_AXIS = 0.5

COLOR_GT = "#2ca02c"
COLOR_PRED = "black"
COLOR_STIM = "#cf222e"
COLOR_OUTLIER = "red"
COLOR_IDENT = "0.8"
DENSITY_CMAP = "inferno_r"

matplotlib.rcParams.update({
    "text.usetex": False, "mathtext.default": "it",
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS_LABEL, "axes.titlesize": FS_LABEL, "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_TICK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": LW_AXIS, "xtick.major.width": LW_AXIS, "ytick.major.width": LW_AXIS,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "axes.labelpad": 1.5, "lines.linewidth": 1.0, "legend.frameon": False,
    "savefig.dpi": 300, "pdf.fonttype": 42, "figure.dpi": 150,
})


def type_cmap(n_types=65):
    cols = []
    for name in ("tab20", "tab20b", "tab20c", "Set1"):
        cols.extend(plt.get_cmap(name).colors)
    return ListedColormap(cols[:n_types])


def _decimals(v, max_d=2):
    for d in range(max_d + 1):
        if abs(round(v, d) - v) < 1e-9:
            return d
    return max_d


def ticks3(ax, lo, hi, axis="both", mid_decimals=None):
    """Ticks at the range's two ends and its middle, labelled as the published
    panels ("0.0 0.5 1.0", "0.0 0.25 0.5"): each tick with the decimals it
    needs, at least one when any tick is fractional. `mid_decimals` rounds the
    middle tick itself (the embedding axes, whose ends are 1-decimal values)."""
    mid = (lo + hi) / 2 if mid_decimals is None else round((lo + hi) / 2, mid_decimals)
    t = [lo, mid, hi]
    d_min = 1 if any(_decimals(v) > 0 for v in t) else 0
    lab = [f"{v:.{max(d_min, _decimals(v))}f}" for v in t]
    if axis in ("x", "both"):
        ax.set_xlim(lo, hi); ax.set_xticks(t); ax.set_xticklabels(lab)
    if axis in ("y", "both"):
        ax.set_ylim(lo, hi); ax.set_yticks(t); ax.set_yticklabels(lab)


def annotate(ax, text):
    ax.text(0.04, 0.96, text, transform=ax.transAxes, ha="left", va="top", fontsize=FS_ANNOT,
            linespacing=1.15)


def identity(ax, lo, hi, thr=None):
    """The y = x line, and the +-thr outlier band of the published V_rest / tau panels."""
    ax.plot([lo, hi], [lo, hi], color=COLOR_IDENT, lw=0.5, zorder=1)
    if thr:
        for s in (-thr, thr):
            ax.plot([lo, hi], [lo + s, hi + s], color=COLOR_IDENT, lw=0.4, ls=":", zorder=1)


def scatter_density(ax, x, y, lo, hi, bins=300):
    """Pooled (neuron, frame) pairs as a log-density image, white where empty."""
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    ax.hist2d(x, y, bins=bins, range=[[lo, hi], [lo, hi]], cmin=1, cmap=DENSITY_CMAP, norm=LogNorm(),
              rasterized=True)
    ticks3(ax, lo, hi)
    ax.set_aspect("equal", adjustable="box")


def panel_labels(fig, axes, letters=None, dx_pt=1.0, dy_pt=2.0, align_rows=True, y_from="axes", ha="left"):
    """Bold letters outside the top-left corner of each panel, as in the
    published figures: the letter starts at the y-axis label's left edge (the
    tight bbox, so it sits over the label, next to its own panel and clear of
    the previous one), y at the top of the axes frame (`y_from="axes"`) or of
    the tight bbox; axes whose tops agree within 2 pt share one label height."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    letters = letters or string.ascii_lowercase
    bbs = [ax.get_tightbbox(rend) for ax in axes]
    xs = np.array([b.x0 for b in bbs])
    ys = np.array([(ax.get_window_extent(rend).y1 if y_from == "axes" else b.y1) for ax, b in zip(axes, bbs)])
    if align_rows:
        tol = 2.0 * fig.dpi / 72.0
        ys = np.array([ys[np.abs(ys - y) < tol].max() for y in ys])
    dx = dx_pt * fig.dpi / 72.0; dy = dy_pt * fig.dpi / 72.0
    for x, y, L in zip(xs, ys, letters):
        fx, fy = inv.transform((x + dx, y + dy))
        fig.text(fx, fy, L, fontsize=FS_PANEL, fontweight="bold", ha=ha, va="bottom")


def column_titles(fig, axes_per_block, titles, dy_pt=4.0, fontsize=FS_LABEL):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for axes, title in zip(axes_per_block, titles):
        bbs = [a.get_tightbbox(rend) for a in axes]
        x0 = min(b.x0 for b in bbs); x1 = max(b.x1 for b in bbs); y1 = max(b.y1 for b in bbs)
        fx0, _ = inv.transform((x0, y1)); fx1, fy = inv.transform((x1, y1 + dy_pt * fig.dpi / 72.0))
        fig.text((fx0 + fx1) / 2, fy, title, ha="center", va="bottom", fontsize=fontsize)


def save(fig, out_base, pad_pt=2.0):
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".pdf", bbox_inches="tight", pad_inches=pad_pt / 72.0)
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight", pad_inches=pad_pt / 72.0)
    print("wrote", out_base + ".{pdf,png}")


def read_metrics(run_dir):
    out = {}
    p = os.path.join(run_dir, "results", "metrics.txt")
    if not os.path.exists(p):
        return out
    for line in open(p):
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        try:
            out[k.strip()] = float(v)
        except ValueError:
            out[k.strip()] = v
    return out


def tester_rollout_r(log_path):
    """(r, sd) as -o test_plot wrote them: the "Pearson r: <mean> +/- <sd>" line of a
    results_rollout*.log (per-neuron Pearson r pooled by utils.fisher_pool). The
    paper never recomputes a rollout r; it reads this one."""
    m = re.search(r"Pearson r:\s*([-\d.eE+]+)\s*\+/-\s*([-\d.eE+]+)", open(log_path).read())
    if m is None:
        raise ValueError(f"no 'Pearson r: <r> +/- <sd>' line in {log_path}")
    return float(m.group(1)), float(m.group(2))


def tester_rollout_frames(log_path):
    """The number of rollout frames the tester scored ("Frames evaluated: 0 to N" -> N)."""
    m = re.search(r"Frames evaluated:\s*0 to (\d+)", open(log_path).read())
    if m is None:
        raise ValueError(f"no 'Frames evaluated' line in {log_path}")
    return int(m.group(1))


ROOT = os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
LOG_ROOT = os.path.join(ROOT, "log", "fly")
DATA_ROOT = os.path.join(ROOT, "graphs_data", "fly")
PAPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PAPER_DIR, "figures")
TAB_DIR = os.path.join(PAPER_DIR, "tables")
