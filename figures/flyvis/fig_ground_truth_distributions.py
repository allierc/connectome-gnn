"""Appendix figure: ground-truth Flyvis circuit-parameter distributions.

Reworked layout of `GNN_PlotFigure.plot_ground_truth_distributions`:

  rows (top → bottom): incoming connections, W (true edge weights),
                       tau (membrane time constant), V_rest.
  cols: left  — per-neuron strip with red mean line per cell type
                (x = neuron index sorted by type, type names tick-labelled);
        right — histogram of the underlying value distribution.

Per-panel summary stats (top-left): mean, median, N, plus min/max for the
strip panels. Crowded type labels (consecutive types whose centre positions
are closer than ``LABEL_MIN_GAP_PX`` pixels apart) are staggered onto two
rows so that ``Lawf1``/``Lawf2``/``Am`` do not overlap.

Inputs (read directly from disk — no replot needed):
  <DATA_ROOT>/graphs_data/fly/flyvis_noise_005_010_blank50_cv00/ode_params.pt
      → tau_i, V_i_rest, edge_index, W

Output:
  figures/fig_ground_truth_distributions.{pdf,png}
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'janne.matplotlibrc'))

import matplotlib.pyplot as plt
import numpy as np
import torch


# ── inputs ──────────────────────────────────────────────────────────────────
DATA_ROOT  = '/groups/saalfeld/home/allierc/GraphData'
DATASET    = 'flyvis_noise_005_010_blank50_cv00'
ODE_PARAMS = f'{DATA_ROOT}/graphs_data/fly/{DATASET}/ode_params.pt'

# Flyvis 65-type index → name (matches GNN_PlotFigure.py around L1389).
INDEX_TO_NAME = {
    0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)', 5: 'L1', 6: 'L2',
    7: 'L3', 8: 'L4', 9: 'L5', 10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi10',
    14: 'Mi11', 15: 'Mi12', 16: 'Mi13', 17: 'Mi14', 18: 'Mi15', 19: 'Mi2',
    20: 'Mi3', 21: 'Mi4', 22: 'Mi9', 23: 'R1', 24: 'R2', 25: 'R3', 26: 'R4',
    27: 'R5', 28: 'R6', 29: 'R7', 30: 'R8', 31: 'T1', 32: 'T2', 33: 'T2a',
    34: 'T3', 35: 'T4a', 36: 'T4b', 37: 'T4c', 38: 'T4d', 39: 'T5a',
    40: 'T5b', 41: 'T5c', 42: 'T5d', 43: 'Tm1', 44: 'Tm16', 45: 'Tm2',
    46: 'Tm20', 47: 'Tm28', 48: 'Tm3', 49: 'Tm30', 50: 'Tm4', 51: 'Tm5Y',
    52: 'Tm5a', 53: 'Tm5b', 54: 'Tm5c', 55: 'Tm9', 56: 'TmY10', 57: 'TmY13',
    58: 'TmY14', 59: 'TmY15', 60: 'TmY18', 61: 'TmY3', 62: 'TmY4',
    63: 'TmY5a', 64: 'TmY9',
}

# Layout knobs.
FIGSIZE          = (10, 9)          # ~25 cm × 23 cm
LABEL_MIN_GAP_PX = 14               # crowd threshold for label staggering
HIST_BINS        = 60
COLOR_MEAN       = '#d62728'        # type-mean line
COLOR_HIST       = '#4c78a8'        # histogram bars
COLOR_DOT        = '#444444'

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── helpers ─────────────────────────────────────────────────────────────────

def _type_boundaries(type_list: np.ndarray) -> dict[int, tuple[int, int]]:
    """Run-length boundaries (start, end) per neuron-type id."""
    bounds: dict[int, tuple[int, int]] = {}
    cur = None
    for i, t in enumerate(type_list):
        t = int(t)
        if t != cur:
            if cur is not None:
                s, _ = bounds[cur]
                bounds[cur] = (s, i - 1)
            bounds[t] = (i, i)
            cur = t
    if cur is not None:
        s, _ = bounds[cur]
        bounds[cur] = (s, len(type_list) - 1)
    return bounds


def _stagger_xticks(ax, positions: list[float], names: list[str],
                    min_gap_px: float = LABEL_MIN_GAP_PX) -> None:
    """Render type labels under the axis, staggered onto two rows when
    consecutive labels would otherwise overlap (e.g. L5 / Lawf1 / Lawf2 / Am)."""
    ax.set_xticks([])  # we draw labels manually so we can stagger them.
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transData.inverted()
    # Convert min-gap from pixels to data units once at draw time.
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 0))
    px_per_data = abs(p1[0] - p0[0])
    min_gap = min_gap_px / max(px_per_data, 1e-9)

    # Two-row stagger: when consecutive labels are within `min_gap`, swap
    # rows so neighbouring crowded labels never share a row. After a
    # non-crowded gap, reset to the upper row so the layout stays
    # predictable.
    y_rows = (-0.02, -0.16)
    row    = 0
    last_x = -np.inf
    for x, name in zip(positions, names):
        if (x - last_x) < min_gap:
            row = 1 - row
        else:
            row = 0
        ax.text(x, y_rows[row], name,
                transform=ax.get_xaxis_transform(),
                rotation=90, ha='center', va='top', fontsize=4.8)
        last_x = x


def _fmt_stats(vals: np.ndarray, *, unit: str = '') -> str:
    mean = float(np.mean(vals))
    sd   = float(np.std(vals))
    med  = float(np.median(vals))
    return (f'mean ± SD = {mean:.3g} ± {sd:.3g}{unit}\n'
            f'median = {med:.3g}{unit}\n'
            f'N = {len(vals):,}')


def _strip_panel(ax, x_pts: np.ndarray, y_pts: np.ndarray,
                 type_bounds: dict[int, tuple[int, int]],
                 type_means: np.ndarray, ylabel: str, *,
                 ylim=None, n_neurons: int | None = None,
                 stats_unit: str = '', stats_vals: np.ndarray | None = None,
                 dot_alpha: float = 0.4, dot_size: float = 0.6) -> None:
    """Per-point scatter + red type-mean h-line per cell type.

    `x_pts`, `y_pts` may be per-neuron (n,) or per-edge (n_edges,) — the
    latter lets the W panel show every edge weight against its target
    neuron index. `type_means` is one value per type (in the iteration
    order of `type_bounds`) to draw the red mean line.
    """
    ax.scatter(x_pts, y_pts, c=COLOR_DOT, s=dot_size, alpha=dot_alpha,
               linewidths=0, rasterized=True)

    pos, names = [], []
    for (tid, (s, e)), mean in zip(type_bounds.items(), type_means):
        pos.append((s + e) / 2)
        names.append(INDEX_TO_NAME.get(tid, f'Type{tid}'))
        ax.hlines(float(mean), s, e,
                  colors=COLOR_MEAN, linewidth=1.4, zorder=5)

    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis='y', labelsize=8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    nn = n_neurons if n_neurons is not None else int(np.max(x_pts)) + 1
    ax.set_xlim(-50, nn + 50)
    _stagger_xticks(ax, pos, names)

    stats = stats_vals if stats_vals is not None else y_pts
    ax.text(0.025, 0.97, _fmt_stats(stats, unit=stats_unit),
            transform=ax.transAxes, va='top', ha='left', fontsize=7.5)


def _hist_panel(ax, vals: np.ndarray, xlabel: str, *,
                bins: int = HIST_BINS, log_y: bool = False, xlim=None) -> None:
    if xlim is None:
        xlim = (float(np.min(vals)), float(np.max(vals)))
    ax.hist(vals, bins=bins, range=xlim, color=COLOR_HIST,
            edgecolor='white', linewidth=0.3)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('count' + (' (log)' if log_y else ''), fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    if log_y:
        ax.set_yscale('log')
    ax.set_xlim(*xlim)


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f'loading {ODE_PARAMS}')
    d = torch.load(ODE_PARAMS, map_location='cpu', weights_only=False)
    edges    = d['edge_index'].cpu().numpy()        # (2, n_edges)
    w_true   = d['W'].cpu().numpy()                  # (n_edges,)
    tau_true = d['tau_i'].cpu().numpy() * 1000.0     # → ms
    vrest    = d['V_i_rest'].cpu().numpy()
    n_neurons = len(tau_true)

    # Neuron-type list — derive from the dataset bookkeeping if available;
    # otherwise the contiguous block layout produced by the simulator
    # already groups neurons by type, so a stable run-length grouping
    # over the natural index order is what the original plot relies on.
    # For Flyvis the neuron index IS sorted by type, so we synthesise the
    # type list from per-row mapping in the dataset's training_edges.
    # Quickest: read the per-fold panels npz which already stores type_ids.
    panels_npz = (f'{DATA_ROOT}/log/fly/flyvis_noise_005_010_blank50_unified_cv00/'
                  f'results/panels_noise_005_010_blank50_cv00.npz')
    type_list = np.load(panels_npz)['type_ids'].astype(int)
    assert len(type_list) == n_neurons, (
        f'type_ids length {len(type_list)} != n_neurons {n_neurons}')

    # Per-neuron incoming-edge count + mean-incoming-weight (target == row 1).
    targets = edges[1, :]
    n_incoming = np.bincount(targets, minlength=n_neurons).astype(float)
    sum_in_w   = np.bincount(targets, weights=w_true, minlength=n_neurons)
    mean_in_w  = np.divide(sum_in_w, n_incoming,
                           out=np.zeros_like(sum_in_w),
                           where=n_incoming > 0)

    bounds = _type_boundaries(type_list)
    x_idx  = np.arange(n_neurons)

    # Per-type means used to draw the red horizontal lines on each strip.
    def _type_means(per_neuron_vals: np.ndarray) -> np.ndarray:
        out = np.empty(len(bounds))
        for i, (_, (s, e)) in enumerate(bounds.items()):
            out[i] = float(np.mean(per_neuron_vals[s:e + 1]))
        return out

    # For the W strip we want every edge plotted against its target
    # neuron index, with the red mean line set to the per-type mean of
    # incoming-edge weights.
    type_mean_w = np.empty(len(bounds))
    for i, (_, (s, e)) in enumerate(bounds.items()):
        edge_mask = (targets >= s) & (targets <= e)
        type_mean_w[i] = float(np.mean(w_true[edge_mask])) if edge_mask.any() else 0.0

    # ── figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        4, 2, figsize=FIGSIZE, constrained_layout=True,
        gridspec_kw={'width_ratios': [3.0, 1.0]},
    )

    # row 0 — connections (moved to top per request)
    _strip_panel(axes[0, 0], x_idx, n_incoming, bounds,
                 _type_means(n_incoming),
                 ylabel='incoming connections',
                 n_neurons=n_neurons)
    _hist_panel(axes[0, 1], n_incoming, xlabel='incoming connections')

    # row 1 — true synaptic weights (every edge as a dot at its target)
    _strip_panel(axes[1, 0], targets, w_true, bounds, type_mean_w,
                 ylabel=r'true $W$',
                 n_neurons=n_neurons, ylim=(-2.0, 4.5),
                 stats_vals=w_true, dot_size=0.4, dot_alpha=0.15)
    _hist_panel(axes[1, 1], w_true, xlabel=r'true $W$',
                log_y=True, xlim=(-2.0, 4.5))

    # row 2 — tau (long-tailed; log y on the histogram)
    _strip_panel(axes[2, 0], x_idx, tau_true, bounds, _type_means(tau_true),
                 ylabel=r'true $\tau$ [ms]', stats_unit=' ms',
                 n_neurons=n_neurons)
    _hist_panel(axes[2, 1], tau_true, xlabel=r'true $\tau$ [ms]', log_y=True)

    # row 3 — V_rest
    _strip_panel(axes[3, 0], x_idx, vrest, bounds, _type_means(vrest),
                 ylabel=r'true $V_{\mathrm{rest}}$ [a.u.]',
                 n_neurons=n_neurons)
    _hist_panel(axes[3, 1], vrest, xlabel=r'true $V_{\mathrm{rest}}$ [a.u.]')

    # Panel labels a..h at the top-left of each outer panel box, all sharing
    # the same y so labels stay aligned. Pattern from figures/INSTRUCTIONS.md.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    flat_axes = [axes[r, c] for r in range(4) for c in range(2)]
    bboxes = [a.get_tightbbox(renderer) for a in flat_axes]
    # Stagger labels: each row of two panels shares its own y so the label
    # sits just above that row's tightbbox (one global y would push the
    # bottom rows' labels far above their panels).
    for row in range(4):
        ax_l, ax_r = axes[row, 0], axes[row, 1]
        bb_l = ax_l.get_tightbbox(renderer)
        bb_r = ax_r.get_tightbbox(renderer)
        y1 = max(inv.transform((bb_l.x0, bb_l.y1))[1],
                 inv.transform((bb_r.x0, bb_r.y1))[1])
        for ax, lbl in ((ax_l, 'abcdefgh'[2*row]),
                        (ax_r, 'abcdefgh'[2*row + 1])):
            bb = ax.get_tightbbox(renderer)
            x0 = inv.transform((bb.x0, bb.y1))[0]
            fig.text(x0, y1, lbl, fontsize=11, fontweight='bold',
                     va='bottom', ha='left', transform=fig.transFigure)

    out_pdf = os.path.join(OUT_DIR, 'fig_ground_truth_distributions.pdf')
    out_png = os.path.join(OUT_DIR, 'fig_ground_truth_distributions.png')
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f'saved {out_pdf}')
    print(f'saved {out_png}')


if __name__ == '__main__':
    main()
