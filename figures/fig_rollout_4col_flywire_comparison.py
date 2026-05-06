"""Figure: 4-column rollout comparison across the 4 flywireRF v2 conditions.

4-column variant of ``fig_rollout_3col_noise_comparison.py``. Sweeps the four
cv00 GNN models trained by ``run_GNN_flywire_blank50.py``:

    e8_flywireRF_noise_005_blank50_flywire_cv00
    e8_flywireRF_proximal_nulls_noise_005_blank50_flywire_cv00
    full_eye_flywireRF_noise_005_blank50_flywire_cv00
    full_eye_flywireRF_proximal_nulls_noise_005_blank50_flywire_cv00

All four are trained at noise σ=0.05 — there is no noise sweep, just a sweep
over connectome variants. The bottom row therefore has 4 scatter panels
(one per column), aligned with the 4 trace columns above.

Caching
-------
The full rollout_bundle.npz files are huge (full_eye ≈ 4.4 GB).  Default
mode loads each bundle, extracts the slim subset needed to plot, and saves
a per-condition cache to::

    figures/data/fig_rollout_4col_flywire_<condition_key>.npz

The cache contains the trace slice (12 selected neurons over 1000 frames),
a subsampled scatter pair (≤ 2 M points), and pre-computed Fisher-pooled
Pearson r ± SD.  Pass ``--from_data`` to skip bundle loading and read the
cache directly — useful when iterating on the figure layout.

Usage
-----
    # Full pipeline (loads bundles, writes cache, plots):
    conda run -n neural-graph-linux \\
        python figures/fig_rollout_4col_flywire_comparison.py

    # Plot from cache (cache must already exist on disk):
    conda run -n neural-graph-linux \\
        python figures/fig_rollout_4col_flywire_comparison.py --from_data

Output
------
    figures/fig_rollout_4col_flywire_comparison.{pdf,png}
    figures/data/fig_rollout_4col_flywire_<condition_key>.npz   (one per column)
"""

import argparse
import os
import string
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'unified_style.matplotlibrc'))

import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
import numpy as np


try:
    from flyvis.analysis.visualization.plt_utils import trim_axis as _trim_axis
except Exception:
    def _trim_axis(ax, xmargin=0.0, ymargin=0.0, yaxis=True, xaxis=True):
        if xaxis:
            xticks = ax.get_xticks()
            xlo, xhi = ax.get_xlim()
            xticks = [t for t in xticks if xlo <= t <= xhi]
            if xticks:
                ax.spines['bottom'].set_bounds(xticks[0], xticks[-1])
        if yaxis:
            yticks = ax.get_yticks()
            ylo, yhi = ax.get_ylim()
            yticks = [t for t in yticks if ylo <= t <= yhi]
            if yticks:
                ax.spines['left'].set_bounds(yticks[0], yticks[-1])


REPO_ROOT = '/workspace/connectome-gnn'
DATA_ROOT = os.environ.get('TRAINED_MODEL_OUTPUT_ROOT', '.')
CFG_DIR   = f'{DATA_ROOT}/config/fly'
CACHE_DIR = os.path.join(REPO_ROOT, 'figures', 'data')
CACHE_PREFIX = 'fig_rollout_4col_flywire_'

COLUMNS = [
    {
        'label': 'e8',
        'key': 'e8',
        'model': 'e8_flywireRF_noise_005_blank50_flywire_cv00',
    },
    {
        'label': 'e8 + proximal nulls',
        'key': 'e8_proximal_nulls',
        'model': 'e8_flywireRF_proximal_nulls_noise_005_blank50_flywire_cv00',
    },
    {
        'label': 'full eye',
        'key': 'full_eye',
        'model': 'full_eye_flywireRF_noise_005_blank50_flywire_cv00',
    },
    {
        'label': 'full eye + proximal nulls',
        'key': 'full_eye_proximal_nulls',
        'model': 'full_eye_flywireRF_proximal_nulls_noise_005_blank50_flywire_cv00',
    },
]


# ── trace / style constants ──────────────────────────────────────────────────
TRACE_START    = 500
TRACE_END      = 1500
SELECTED_TYPES = [23, 5, 6, 7, 12, 22, 43, 55, 35, 39, 31, 0]

COLOR_GT   = '#2ca02c'
COLOR_PRED = 'black'
COLOR_STIM = '#cf222e'
LW_GT, LW_PRED, LW_STIM = 1.2, 0.45, 0.6
DT_MS = 20.0

FS_LABEL  = 8
FS_TICK   = 6
FS_ANNOT  = 5
FS_TYPE   = 6
PANEL_LBL = 8

SCATTER_N_MAX = 2_000_000
SCATTER_RNG   = np.random.default_rng(0)
SCATTER_LO, SCATTER_HI = -10.0, 10.0

TRACE_SHRINK = 0.65

# 4 columns at the same per-column physical width as the 3-col version
# (~2.36 in/col). 4 × 2.36 ≈ 9.45 in (24 cm).
FIG_W_IN = 24.0 * 0.3937
FIG_H_IN = 7.5


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------
def cache_path(key):
    return os.path.join(CACHE_DIR, f'{CACHE_PREFIX}{key}.npz')


def _bundle_path(model):
    return f'{DATA_ROOT}/log/fly/{model}/results/rollout_bundle.npz'


def _subsample_pair(x_full, y_full, n_max=SCATTER_N_MAX):
    assert x_full.shape == y_full.shape
    x = x_full.reshape(-1).astype(np.float32)
    y = y_full.reshape(-1).astype(np.float32)
    n_tot = x.size
    if n_tot <= n_max:
        return x, y, n_tot
    stride = int(np.ceil(n_tot / n_max))
    offset = int(SCATTER_RNG.integers(0, stride))
    return x[offset::stride], y[offset::stride], n_tot


def _select_neurons(type_ids, type_names):
    index_to_name = {i: type_names[i] for i in range(len(type_names))}
    neuron_idx, labels = [], []
    for t in SELECTED_TYPES:
        ids = np.where(type_ids == t)[0]
        if len(ids) > 0:
            neuron_idx.append(int(ids[0]))
            labels.append(index_to_name.get(t, f'Type{t}'))
    return neuron_idx, labels


def extract_and_cache(col):
    """Load rollout_bundle.npz, extract the slim subset, save .npz cache."""
    sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
    sys.path.insert(0, REPO_ROOT)
    from connectome_gnn.utils import compute_trace_metrics, fisher_pool

    path = _bundle_path(col['model'])
    if not os.path.isfile(path):
        sys.exit(f"missing rollout bundle: {path}")
    print(f"[{col['key']}] loading bundle: {path}")
    b = np.load(path, allow_pickle=True)
    true_full = np.asarray(b['activity_true'])
    pred_full = np.asarray(b['activity_pred'])
    stim_full = np.asarray(b['stimulus']) if 'stimulus' in b.files else None
    type_ids   = np.asarray(b['type_ids']).astype(int)
    type_names = list(b['type_names'])

    neuron_idx, labels = _select_neurons(type_ids, type_names)

    sl = slice(TRACE_START, TRACE_END)
    trace_true = np.asarray(true_full[neuron_idx, sl], dtype=np.float32)
    trace_pred = np.asarray(pred_full[neuron_idx, sl], dtype=np.float32)
    trace_stim = (np.asarray(stim_full[neuron_idx, sl], dtype=np.float32)
                  if stim_full is not None else np.empty((0, 0), dtype=np.float32))

    print(f"[{col['key']}] computing per-neuron Pearson on full activity "
          f"(shape={true_full.shape})")
    _, pear, _, _ = compute_trace_metrics(np.asarray(true_full),
                                          np.asarray(pred_full))
    fp = fisher_pool(pear)
    r_mean   = float(fp['r_mean'])
    r_sd_sym = float(fp['r_sd_sym'])

    print(f"[{col['key']}] subsampling scatter "
          f"(target ≤ {SCATTER_N_MAX:,} of {true_full.size:,})")
    sx, sy, n_tot = _subsample_pair(true_full, pred_full)

    os.makedirs(CACHE_DIR, exist_ok=True)
    out = cache_path(col['key'])
    np.savez_compressed(
        out,
        trace_true=trace_true,
        trace_pred=trace_pred,
        trace_stim=trace_stim,
        trace_labels=np.asarray(labels, dtype=object),
        scatter_x_sub=sx,
        scatter_y_sub=sy,
        scatter_n_total=np.int64(n_tot),
        r_mean=np.float64(r_mean),
        r_sd_sym=np.float64(r_sd_sym),
        trace_start=np.int64(TRACE_START),
        trace_end=np.int64(TRACE_END),
        dt_ms=np.float64(DT_MS),
        type_ids_selected=np.asarray([int(SELECTED_TYPES[i])
                                      for i in range(len(neuron_idx))]),
    )
    print(f"[{col['key']}] wrote cache: {out}")


def load_cache(col):
    p = cache_path(col['key'])
    if not os.path.isfile(p):
        sys.exit(f"missing cache for --from_data: {p}\n"
                 f"  → run without --from_data first to produce it.")
    print(f"[{col['key']}] loading cache: {p}")
    z = np.load(p, allow_pickle=True)
    has_stim = z['trace_stim'].size > 0
    return {
        'trace_true':   np.asarray(z['trace_true']),
        'trace_pred':   np.asarray(z['trace_pred']),
        'trace_stim':   np.asarray(z['trace_stim']) if has_stim else None,
        'labels':       [str(s) for s in z['trace_labels']],
        'scatter_x':    np.asarray(z['scatter_x_sub']),
        'scatter_y':    np.asarray(z['scatter_y_sub']),
        'r_mean':       float(z['r_mean']),
        'r_sd_sym':     float(z['r_sd_sym']),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _pretty_ticks(lo, hi, n_target=4):
    span = hi - lo
    raw_step = span / max(1, n_target - 1)
    mag = 10 ** np.floor(np.log10(max(raw_step, 1e-12)))
    step = mag
    for m in (1, 2, 5, 10):
        if m * mag >= raw_step:
            step = m * mag
            break
    tick_lo = np.ceil(lo / step - 1e-9) * step
    ticks = np.arange(tick_lo, hi + step / 2, step)
    if len(ticks) == 0 or ticks[-1] < hi - step * 1e-6:
        ticks = np.append(ticks, hi)
    return ticks


def draw_traces(ax, true_w, pred_w, stim_w, labels, step_v, time_ms,
                column_title, show_type_labels, show_xlabel=True,
                header_text=None):
    n_traces, n_frames = true_w.shape
    baselines = true_w.mean(axis=1)
    s = TRACE_SHRINK
    for i in range(n_traces):
        bl = baselines[i]
        ax.plot(time_ms, s * (true_w[i] - bl) + i * step_v,
                lw=LW_GT, color=COLOR_GT, alpha=0.95, zorder=2)
        ax.plot(time_ms, s * (pred_w[i] - bl) + i * step_v,
                lw=LW_PRED, color=COLOR_PRED, alpha=0.95, zorder=3)
        if stim_w is not None and stim_w[i].std() > 1e-6:
            stim = stim_w[i]
            stim_y = i * step_v - 0.4 * step_v
            ax.plot(time_ms, s * (stim - stim.mean()) + stim_y,
                    lw=LW_STIM, color=COLOR_STIM, alpha=0.95, zorder=4)

    if show_type_labels:
        for i, lbl in enumerate(labels):
            ax.text(time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.02,
                    i * step_v, lbl, fontsize=FS_TYPE,
                    va='bottom', ha='right', color='black')
        ax.set_ylabel('neurons', fontsize=FS_LABEL, labelpad=18)

    if header_text is None:
        header_lines = []
    elif isinstance(header_text, (list, tuple)):
        header_lines = list(header_text)
    else:
        header_lines = [header_text]
    HEADER_DY = 0.05
    for k, line in enumerate(header_lines):
        y = 0.99 - k * HEADER_DY
        ax.text(0.015, y, line, transform=ax.transAxes,
                va='top', ha='left', fontsize=FS_TICK,
                fontweight='normal',
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=0.4))
    ax._column_title = column_title

    ax.set_ylim([-step_v, (n_traces - 1) * step_v + 2.2 * step_v])
    ax.set_yticks([])
    ax.set_xlim([time_ms[0], time_ms[-1]])
    ax.spines['left'].set_visible(False)
    if show_xlabel:
        ticks = _pretty_ticks(time_ms[0], time_ms[-1], n_target=3)
        ax.set_xticks(ticks)
        ax.set_xlabel('time (ms)', fontsize=FS_LABEL, labelpad=1)
        ax.tick_params(axis='x', labelsize=FS_TICK, pad=1)
        _trim_axis(ax, yaxis=False)
    else:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)


def draw_scatter(ax, x_sub, y_sub, r, r_sd, xlabel, ylabel):
    lo, hi = SCATTER_LO, SCATTER_HI
    ax.hexbin(x_sub, y_sub, gridsize=140, bins='log', cmap='magma_r',
              mincnt=1, extent=(lo, hi, lo, hi), linewidths=0.0)
    ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ticks = [lo, 0.0, hi]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])
    _trim_axis(ax)
    ax.text(0.05, 0.97, f"$r$ = {r:.2f} $\\pm$ {r_sd:.2f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=FS_TICK)


def add_panel_label(fig, ax, letter, dx=0.015, dy=0.01):
    pos = ax.get_position()
    fig.text(pos.x0 - dx, pos.y1 + dy, letter,
             fontsize=PANEL_LBL, fontweight='bold',
             va='bottom', ha='left', color='black',
             transform=fig.transFigure)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--from_data', action='store_true',
                   help='Skip rollout_bundle.npz loading and plot from the '
                        f'per-condition cache in {CACHE_DIR}/. The cache must '
                        'have been produced by an earlier run without this '
                        'flag.')
    args = p.parse_args()

    if not args.from_data:
        print('=== extracting per-condition data and writing cache ===')
        for col in COLUMNS:
            if os.path.isfile(cache_path(col['key'])):
                print(f"[{col['key']}] cache exists, skipping extraction: "
                      f"{cache_path(col['key'])}")
                continue
            extract_and_cache(col)

    print('\n=== loading cache and assembling figure ===')
    data = [load_cache(col) for col in COLUMNS]

    labels = data[0]['labels']
    step_vs = [3.0 * TRACE_SHRINK * float(np.std(d['trace_true'])) for d in data]
    step_v = max(0.5 * TRACE_SHRINK, max(step_vs))
    n_frames = TRACE_END - TRACE_START
    time_ms = np.arange(n_frames) * DT_MS + TRACE_START * DT_MS

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), constrained_layout=False)
    outer = mgs.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.4 * TRACE_SHRINK, 2.0],
        left=0.05, right=0.99, top=0.97, bottom=0.04,
        hspace=0.0,
    )
    TOP_WSPACE = 0.25
    top_gs    = mgs.GridSpecFromSubplotSpec(1, len(COLUMNS), outer[0, 0],
                                            wspace=TOP_WSPACE)
    bottom_gs = mgs.GridSpecFromSubplotSpec(1, len(COLUMNS), outer[1, 0],
                                            wspace=TOP_WSPACE)

    trace_axes = []
    for c, (col, d) in enumerate(zip(COLUMNS, data)):
        ax = fig.add_subplot(top_gs[0, c])
        true_w = d['trace_true']
        pred_w = d['trace_pred']
        stim_w = d['trace_stim']
        header = f"$r$ = {d['r_mean']:.2f} $\\pm$ {d['r_sd_sym']:.2f}"
        draw_traces(
            ax, true_w, pred_w, stim_w, labels, step_v, time_ms,
            column_title=col['label'],
            show_type_labels=(c == 0),
            show_xlabel=(c == 0),
            header_text=header,
        )
        trace_axes.append(ax)

    fig.canvas.draw()
    TRACE_TITLE_DY = 0.02
    for ax_t in trace_axes:
        pos = ax_t.get_position()
        x_center = pos.x0 + pos.width / 2
        fig.text(x_center, pos.y1 + TRACE_TITLE_DY, ax_t._column_title,
                 va='bottom', ha='center', fontsize=FS_LABEL,
                 fontweight='normal', transform=fig.transFigure)

    scatter_axes = []
    for c, (col, d) in enumerate(zip(COLUMNS, data)):
        ax = fig.add_subplot(bottom_gs[0, c])
        xlbl = 'ground truth voltage' if c == 0 else ''
        ylbl = 'rollout voltage'      if c == 0 else ''
        draw_scatter(ax, d['scatter_x'], d['scatter_y'],
                     d['r_mean'], d['r_sd_sym'],
                     xlabel=xlbl, ylabel=ylbl)
        scatter_axes.append(ax)

    SCATTER_PULL_UP = 0.13
    for ax in scatter_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + SCATTER_PULL_UP,
                         pos.width, pos.height])

    all_axes = trace_axes + scatter_axes
    letters = list(string.ascii_lowercase[:len(all_axes)])
    fig.canvas.draw()
    for ax, letter in zip(all_axes, letters):
        if ax in trace_axes:
            add_panel_label(fig, ax, letter, dy=TRACE_TITLE_DY)
        else:
            add_panel_label(fig, ax, letter)

    out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'fig_rollout_4col_flywire_comparison')
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_base}.pdf')
    print(f'Saved: {out_base}.png')


if __name__ == '__main__':
    main()
