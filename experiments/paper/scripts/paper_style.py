"""Shared style for the paper's figures: one place for every size, colour and
helper, so the panels of different figures read as one system.

Geometry: text width 18 cm (NeurIPS \textwidth is 5.5 in = 13.97 cm; figures
are included at \textwidth, so a figure drawn at 18 cm is scaled by 0.78 and
an 8 pt label prints at 6.2 pt). Font sizes below are set for that scaling:
labels 8 pt, ticks and annotations 6-7 pt in the drawn figure.

Panel labels: bold lower-case letter, top-left OUTSIDE the axes, at a fixed
offset from the axes' tight bounding box so the letters of one row share a
baseline and sit at the same distance from their panel.
"""
import os
import string

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

CM = 1.0 / 2.54
TEXT_W_CM = 18.0              # width every full-width figure is drawn at
FS_LABEL = 8                  # axis labels, column titles
FS_TICK = 6                   # tick labels
FS_ANNOT = 5.5                # R^2 / slope / r annotations inside panels
FS_PANEL = 9                  # panel letters
LW_AXIS = 0.5

COLOR_GT = "#2ca02c"          # ground truth: green
COLOR_PRED = "black"          # prediction: black
COLOR_STIM = "#cf222e"        # stimulus: red
COLOR_IDENT = "0.35"          # identity line

RC = {
    "text.usetex": False,
    "mathtext.default": "it",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS_LABEL,
    "axes.titlesize": FS_LABEL,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_TICK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": LW_AXIS,
    "xtick.major.width": LW_AXIS,
    "ytick.major.width": LW_AXIS,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "axes.labelpad": 1.5,
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "figure.dpi": 150,
}
matplotlib.rcParams.update(RC)


def type_cmap(n_types=65):
    """A categorical map with one distinct colour per cell type (65 classes):
    tab20, tab20b, tab20c give 60; the last 5 come from Set1."""
    cols = []
    for name in ("tab20", "tab20b", "tab20c", "Set1"):
        cols.extend(plt.get_cmap(name).colors)
    return ListedColormap(cols[:n_types])


def pretty_ticks(lo, hi, n_target=4):
    """Ticks at pretty values, the first and last inside [lo, hi], regular spacing."""
    span = hi - lo
    if span <= 0:
        return np.array([lo])
    raw = span / max(1, n_target - 1)
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    t0 = np.ceil(lo / step - 1e-9) * step
    ticks = np.arange(t0, hi + step * 1e-6, step)
    return ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]


def trim_axis(ax):
    """Spines span only the tick range (Janne's rule: axes rarely share an origin)."""
    xt = [t for t in ax.get_xticks() if ax.get_xlim()[0] - 1e-9 <= t <= ax.get_xlim()[1] + 1e-9]
    yt = [t for t in ax.get_yticks() if ax.get_ylim()[0] - 1e-9 <= t <= ax.get_ylim()[1] + 1e-9]
    if xt:
        ax.spines["bottom"].set_bounds(xt[0], xt[-1])
    if yt:
        ax.spines["left"].set_bounds(yt[0], yt[-1])


def r2_identity(true, pred):
    """Identity-line R^2 (Nash-Sutcliffe): 1 - MSE / Var(true)."""
    true = np.asarray(true, float); pred = np.asarray(pred, float)
    return 1.0 - np.mean((pred - true) ** 2) / np.var(true)


def slope_fit(true, pred):
    a, _ = np.polyfit(np.asarray(true, float), np.asarray(pred, float), 1)
    return a


def square_limits(x, y, pad=0.04, lo=None, hi=None):
    """One common range for a learned-vs-true scatter."""
    if lo is None:
        lo = min(np.nanmin(x), np.nanmin(y))
    if hi is None:
        hi = max(np.nanmax(x), np.nanmax(y))
    d = (hi - lo) * pad
    return lo - d, hi + d


def scatter_identity(ax, true, pred, colors=None, cmap=None, s=1.0, alpha=0.6,
                     n_max=100_000, seed=0, lims=None, annot=None, xlabel="", ylabel=""):
    """Learned-vs-true scatter with the identity line and a top-left annotation.

    `annot` is the text printed top-left inside the axes (e.g. the R^2 and slope
    read from the run's metrics.txt, so figure and table quote one number).
    """
    true = np.asarray(true); pred = np.asarray(pred)
    n = len(true)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, min(n, n_max), replace=False) if n > n_max else np.arange(n)
    c = None if colors is None else np.asarray(colors)[idx]
    lo, hi = square_limits(true[idx], pred[idx]) if lims is None else lims
    ax.plot([lo, hi], [lo, hi], color=COLOR_IDENT, lw=0.5, ls="--", zorder=1)
    ax.scatter(true[idx], pred[idx], c=c, cmap=cmap, s=s, alpha=alpha, lw=0,
               rasterized=True, zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    t = pretty_ticks(lo, hi, 4)
    ax.set_xticks(t); ax.set_yticks(t)
    ax.set_aspect("equal", adjustable="box")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if annot:   # bottom-right: empty on an identity scatter, and clear of the panel letter
        ax.text(0.97, 0.03, annot, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FS_ANNOT, linespacing=1.1)
    trim_axis(ax)


def panel_labels(fig, axes, letters=None, dx_pt=-2.0, dy_pt=1.0, align_rows=True):
    """Bold letters top-left of each axes, outside the tight bbox.

    With `align_rows`, axes whose tops are within 2 pt share one label height
    (the highest), so a row's letters sit on one line whatever each panel's
    tick labels do.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    letters = letters or string.ascii_lowercase
    pts = []
    for ax in axes:
        bb = ax.get_tightbbox(rend)
        pts.append((bb.x0, bb.y1))
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    if align_rows:
        ys_al = ys.copy()
        tol = 2.0 * fig.dpi / 72.0
        for i, y in enumerate(ys):
            ys_al[i] = ys[np.abs(ys - y) < tol].max()
        ys = ys_al
    dx = dx_pt * fig.dpi / 72.0; dy = dy_pt * fig.dpi / 72.0
    for (x, y), L in zip(zip(xs, ys), letters):
        fx, fy = inv.transform((x + dx, y + dy))
        fig.text(fx, fy, L, fontsize=FS_PANEL, fontweight="bold", ha="right", va="bottom")


def column_titles(fig, axes_per_block, titles, dy_pt=4.0):
    """One title centred above each block of axes (a list of axes per block)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for axes, title in zip(axes_per_block, titles):
        bbs = [a.get_tightbbox(rend) for a in axes]
        x0 = min(b.x0 for b in bbs); x1 = max(b.x1 for b in bbs); y1 = max(b.y1 for b in bbs)
        fx0, _ = inv.transform((x0, y1)); fx1, fy = inv.transform((x1, y1 + dy_pt * fig.dpi / 72.0))
        fig.text((fx0 + fx1) / 2, fy, title, ha="center", va="bottom", fontsize=FS_LABEL)


def save(fig, out_base, pad_pt=2.0):
    """PDF (vector, rasterised scatters) and a 300 dpi PNG, cropped to content."""
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".pdf", bbox_inches="tight", pad_inches=pad_pt / 72.0)
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight", pad_inches=pad_pt / 72.0)
    print("wrote", out_base + ".{pdf,png}")


def read_metrics(run_dir):
    """metrics.txt -> {key: float} (non-numeric values kept as strings)."""
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


LOG_ROOT = os.path.join(os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"), "log", "fly")
DATA_ROOT = os.path.join(os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"), "graphs_data", "fly")
PAPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PAPER_DIR, "figures")
TAB_DIR = os.path.join(PAPER_DIR, "tables")
