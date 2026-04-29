"""Figure: 3-column rollout comparison with **50% data-driven edge ablation**.

Same layout as ``fig_rollout_3col_noise_comparison_ablation20.py``, but the
ablation is *matched* between simulator and GNN: the simulator regenerates the
test data with 50% of edges zeroed (saved as ``ablation_mask.pt`` next to the
test split), and graph_tester applies the same mask to the trained model's
``W`` before rollout. This is the apples-to-apples test from Notebook_03 of
flyvis-gnn (the network learned the message-passing rules iff a 50%-reduced
circuit still produces the right dynamics, no retraining).

Mechanism: passes the appropriate ``flyvis_noise_*_mask_50.yaml`` as the 4th
positional argument to ``GNN_Main.py`` (test_config). graph_tester picks up
``ablation_mask.pt`` from that test dataset and zeroes the corresponding
entries of ``model.W``.

Row 1 (traces): learned (black, ablated W) vs ground truth (green, ablated
simulator). For the noisy columns, the GT comes from the noise-matched
ablated dataset.

Row 2 (scatters): 5 panels in one row.
    • noise-free     : 1 panel — ablated rollout vs noise-free ablated GT
    • low model noise: 2 panels — vs noise-matched ablated GT / vs noise-free ablated GT
    • high model noise: 2 panels — same pair as low model noise

Every panel gets a letter label (a–h).

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/fig_rollout_3col_noise_comparison_ablation50.py

Output
------
    figures/fig_rollout_3col_noise_comparison_ablation50.{pdf,png}
"""

import os
import string
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'janne.matplotlibrc'))

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


# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = '/workspace/connectome-gnn'
DATA_ROOT = '/groups/saalfeld/home/allierc/GraphData'
CFG_DIR   = f'{DATA_ROOT}/config/fly'      # model yamls live here
MASK_CFG_DIR = f'{REPO_ROOT}/config/fly'   # flyvis_noise_*_mask_50.yaml live here

# Shared noise-free ablated test config — used as the cross-test for the noisy
# columns and as the only test for the noise-free column.
NF_MASK_YAML = f'{MASK_CFG_DIR}/flyvis_noise_free_mask_50.yaml'

COLUMNS = [
    {
        'label': 'noise-free',
        'sigma': r'$\sigma = 0$',
        'model': 'flyvis_noise_free_blank50_unified_cv00',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_free_blank50_unified_cv00.yaml',
        'matched_mask_yaml': NF_MASK_YAML,
        'matched_suffix': 'noise_free_mask_50',
        'noise_level': 0.0,
    },
    {
        'label': 'low model noise',
        'sigma': r'$\sigma = 0.05$',
        'model': 'flyvis_noise_005_blank50_unified_cv00',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_005_blank50_unified_cv00.yaml',
        'matched_mask_yaml': f'{MASK_CFG_DIR}/flyvis_noise_005_mask_50.yaml',
        'matched_suffix': 'noise_005_mask_50',
        'noise_level': 0.05,
    },
    {
        'label': 'high model noise',
        'sigma': r'$\sigma = 0.5$',
        'model': 'flyvis_noise_05_blank50_unified_cv00',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_05_blank50_unified_cv00.yaml',
        'matched_mask_yaml': f'{MASK_CFG_DIR}/flyvis_noise_05_mask_50.yaml',
        'matched_suffix': 'noise_05_mask_50',
        'noise_level': 0.5,
    },
]


# ── trace / style constants ──────────────────────────────────────────────────
TRACE_START    = 500
TRACE_END      = 1500
SELECTED_TYPES = [23, 5, 6, 7, 12, 22, 43, 55, 35, 39, 31, 0]

COLOR_GT   = '#2ca02c'   # brighter green, thicker line (see LW_GT)
COLOR_PRED = 'black'
COLOR_STIM = '#cf222e'
LW_GT, LW_PRED, LW_STIM = 1.2, 0.45, 0.6
DT_MS = 20.0

FS_LABEL  = 8
FS_TICK   = 6
FS_ANNOT  = 5
FS_TYPE   = 6
PANEL_LBL = 8

# Scatter: subsample to at most this many (neuron, frame) pairs.
SCATTER_N_MAX = 2_000_000
SCATTER_RNG   = np.random.default_rng(0)

# Trace shrink factor — applied to both per-trace amplitude AND inter-trace
# step. <1.0 shrinks both proportionally, with row 1 height reduced to match,
# so the visual outcome is smaller traces packed more tightly.
TRACE_SHRINK = 0.65

FIG_W_IN = 18.0 * 0.3937   # ≈ 7.09 in
FIG_H_IN = 5.2             # shorter overall — row 1 height reduced


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------
def _run(*args, tag):
    print(f'{tag} python GNN_Main.py {" ".join(args)}')
    subprocess.check_call(
        ['python', f'{REPO_ROOT}/GNN_Main.py', *args,
         '--output_root', DATA_ROOT],
        cwd=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# Bundle paths and on-demand generation
# ---------------------------------------------------------------------------
def _bundle_path(col, suffix):
    return (f"{DATA_ROOT}/log/fly/{col['model']}/results/"
            f"rollout_bundle_on_{suffix}.npz")


def matched_bundle_path(col):
    """Rollout on the noise-matched ablated test data."""
    return _bundle_path(col, col['matched_suffix'])


def nf_bundle_path(col):
    """Rollout on the noise-free ablated test data (cross-test)."""
    return _bundle_path(col, 'noise_free_mask_50')


def ensure_mask_test(col, mask_yaml, bundle_path):
    """Run `GNN_Main.py -o test <model_yaml> best <mask_yaml>` if bundle missing.

    graph_tester loads ablation_mask.pt from <mask_yaml>'s dataset and zeroes
    the corresponding entries of model.W before rollout, so the simulator and
    the GNN operate on identical 50%-reduced circuits.
    """
    if os.path.isfile(bundle_path):
        print(f"[{col['model']}] bundle exists: {bundle_path}")
        return
    if not os.path.isfile(col['model_yaml']):
        sys.exit(f"missing model yaml: {col['model_yaml']}")
    if not os.path.isfile(mask_yaml):
        sys.exit(f"missing mask yaml: {mask_yaml}")
    print(f"[{col['model']}] running rollout: test_config={mask_yaml}")
    _run('-o', 'test', col['model_yaml'], 'best', mask_yaml,
         tag=f"[{col['model']}]")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_bundle(path):
    if not os.path.isfile(path):
        sys.exit(f"missing rollout bundle: {path}")
    b = np.load(path, allow_pickle=True)
    return {
        'true' : np.asarray(b['activity_true']),
        'pred' : np.asarray(b['activity_pred']),
        'stim' : np.asarray(b['stimulus']) if 'stimulus' in b.files else None,
        'type_ids'   : np.asarray(b['type_ids']).astype(int),
        'type_names' : list(b['type_names']),
    }


def load_matched_bundle(col):
    return _load_bundle(matched_bundle_path(col))


def load_nf_bundle(col):
    return _load_bundle(nf_bundle_path(col))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _slice(arr, neuron_idx):
    return np.asarray(arr[neuron_idx, TRACE_START:TRACE_END], dtype=np.float32)


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


def _affine_align(pred, true):
    """Return (a*pred + b) where (a, b) minimize ||true - (a*pred + b)||² (OLS)."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    var = p.var()
    if var < 1e-12:
        return p - p.mean() + t.mean()
    a, b = np.polyfit(p, t, 1)
    return a * p + b


def draw_traces(ax, true_w, pred_w, stim_w, labels, step_v, time_ms,
                column_title, show_type_labels, show_xlabel=True,
                header_text=None):
    n_traces, n_frames = true_w.shape
    baselines = true_w.mean(axis=1)
    s = TRACE_SHRINK
    for i in range(n_traces):
        bl = baselines[i]
        pred_aligned = _affine_align(pred_w[i], true_w[i]).astype(np.float32)
        ax.plot(time_ms, s * (true_w[i] - bl) + i * step_v,
                lw=LW_GT, color=COLOR_GT, alpha=0.95, zorder=2)
        ax.plot(time_ms, s * (pred_aligned - bl) + i * step_v,
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


SCATTER_LO, SCATTER_HI = -10.0, 10.0


def draw_scatter(ax, x_all, y_all, xlabel, ylabel, show_fit=True, show_r=True):
    """Hexbin density of y vs x with per-neuron Fisher-pooled Pearson r.

    The hexbin uses subsampled flattened (x, y) for the visual; the
    correlation reported is computed on the full 2-D traces using
    compute_trace_metrics → per-neuron pearsonr → fisher_pool['r_mean'],
    matching the values reported by graph_tester and the cv tables.

    `show_r=False` suppresses the printed `r = X.XX` annotation
    (used for the "vs noisy ablated" panels where comparing the
    learned trace against the perturbed gt is misleading).
    """
    from connectome_gnn.utils import compute_trace_metrics, fisher_pool
    x_sub, y_sub, n_full = _subsample_pair(x_all, y_all)

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

    if show_r:
        _, _pear, _, _ = compute_trace_metrics(
            np.asarray(x_all), np.asarray(y_all))
        r = float(fisher_pool(_pear)['r_mean'])
        ax.text(0.05, 0.97, f"$r$ = {r:.2f}",
                transform=ax.transAxes, va='top', ha='left', fontsize=FS_TICK)


def affine_fit_stats(true_arr, pred_arr):
    """Per-neuron OLS: pred_aligned[i] = a[i] * pred[i] + b[i] ≈ true[i]."""
    true_arr = np.asarray(true_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    t_mean = true_arr.mean(axis=1, keepdims=True)
    p_mean = pred_arr.mean(axis=1, keepdims=True)
    dt = true_arr - t_mean
    dp = pred_arr - p_mean
    p_var = (dp ** 2).mean(axis=1)
    t_var = (dt ** 2).mean(axis=1)
    cov   = (dp * dt).mean(axis=1)
    a = np.where(p_var > 1e-12, cov / np.maximum(p_var, 1e-12), 0.0)
    b = t_mean.squeeze() - a * p_mean.squeeze()
    r_per = cov / np.sqrt(np.maximum(p_var * t_var, 1e-24))

    pred_aligned = a[:, None] * pred_arr + b[:, None]
    rmse_raw     = float(np.sqrt(((true_arr - pred_arr) ** 2).mean()))
    rmse_aligned = float(np.sqrt(((true_arr - pred_aligned) ** 2).mean()))

    return {
        'a': a, 'b': b, 'r_per': r_per,
        'a_mean': float(a.mean()),  'a_std': float(a.std()),
        'b_mean': float(b.mean()),  'b_std': float(b.std()),
        'r_mean': float(r_per.mean()), 'r_std': float(r_per.std()),
        'rmse_raw': rmse_raw, 'rmse_aligned': rmse_aligned,
    }


def print_affine_fit_table(rows):
    header = (f"{'bundle':<54} {'a (mean±std)':>17}  {'b (mean±std)':>17}  "
              f"{'r (mean±std)':>17}  {'RMSE raw→aligned':>22}")
    print()
    print('=== per-neuron affine fit (pred_aligned = a*pred + b ≈ true) ===')
    print(header)
    print('-' * len(header))
    for label, s in rows:
        a_str = f"{s['a_mean']:+.3f} ± {s['a_std']:.3f}"
        b_str = f"{s['b_mean']:+.3f} ± {s['b_std']:.3f}"
        r_str = f"{s['r_mean']:+.3f} ± {s['r_std']:.3f}"
        rm_str = f"{s['rmse_raw']:.3f} → {s['rmse_aligned']:.3f}"
        print(f"{label:<54} {a_str:>17}  {b_str:>17}  {r_str:>17}  {rm_str:>22}")
    print()


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
    # Step 1: ensure both ablated rollout bundles per column.
    # For the noise-free column, the matched and nf-cross bundles are the same
    # file (model and test data both at noise=0), so ensure_mask_test is a
    # no-op for the second call.
    for col in COLUMNS:
        ensure_mask_test(col, col['matched_mask_yaml'], matched_bundle_path(col))
        ensure_mask_test(col, NF_MASK_YAML,            nf_bundle_path(col))

    # Step 2: load bundles.
    matched = [load_matched_bundle(col) for col in COLUMNS]
    nf      = [load_nf_bundle(col)      for col in COLUMNS]

    # Diagnostic: per-neuron affine fit.
    rows = []
    for col, m, n in zip(COLUMNS, matched, nf):
        rows.append((f"{col['model']}  (matched-noise ablated)",
                     affine_fit_stats(m['true'], m['pred'])))
        rows.append((f"{col['model']}  (noise-free ablated)",
                     affine_fit_stats(n['true'], n['pred'])))
    print_affine_fit_table(rows)

    # Use the noise-matched ablated bundle for the trace plot (so the GT noise
    # level matches the column label).
    trace_src = matched

    # Neuron selection from the first column's type_ids.
    type_ids   = trace_src[0]['type_ids']
    type_names = trace_src[0]['type_names']
    index_to_name = {i: type_names[i] for i in range(len(type_names))}
    neuron_idx, labels = [], []
    for t in SELECTED_TYPES:
        ids = np.where(type_ids == t)[0]
        if len(ids) > 0:
            neuron_idx.append(int(ids[0]))
            labels.append(index_to_name.get(t, f'Type{t}'))

    step_vs = [3.0 * TRACE_SHRINK * float(np.std(_slice(ts['true'], neuron_idx)))
               for ts in trace_src]
    step_v = max(0.5 * TRACE_SHRINK, max(step_vs))
    n_frames = TRACE_END - TRACE_START
    time_ms = np.arange(n_frames) * DT_MS + TRACE_START * DT_MS

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), constrained_layout=False)
    outer = mgs.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.4 * TRACE_SHRINK, 1.0],
        left=0.06, right=0.98, top=0.97, bottom=0.04,
        hspace=0.0,
    )
    TOP_WSPACE    = 0.25
    GROUP_WSPACE  = 0.35
    top_gs   = mgs.GridSpecFromSubplotSpec(1, 3, outer[0, 0], wspace=TOP_WSPACE)
    group_gs = mgs.GridSpecFromSubplotSpec(1, 3, outer[1, 0], wspace=TOP_WSPACE)
    nf_gs = mgs.GridSpecFromSubplotSpec(1, 2, group_gs[0, 0], wspace=GROUP_WSPACE)
    lo_gs = mgs.GridSpecFromSubplotSpec(1, 2, group_gs[0, 1], wspace=GROUP_WSPACE)
    hi_gs = mgs.GridSpecFromSubplotSpec(1, 2, group_gs[0, 2], wspace=GROUP_WSPACE)

    # --- Row 1: traces ---
    trace_axes = []
    for c, (col, ts, n) in enumerate(zip(COLUMNS, trace_src, nf)):
        ax = fig.add_subplot(top_gs[0, c])
        true_w = _slice(ts['true'], neuron_idx)
        pred_w = _slice(ts['pred'], neuron_idx)
        stim_w = (_slice(ts['stim'], neuron_idx)
                  if ts.get('stim') is not None else None)
        # Per-neuron Pearson r, Fisher-z pooled — matches graph_tester /
        # cv-table recipe (compute_trace_metrics → fisher_pool['r_mean']).
        # Single line: only the "vs noise-free ablated" comparison is shown
        # (the noisy-vs-learned r is suppressed).
        from connectome_gnn.utils import compute_trace_metrics, fisher_pool
        _src = n if col['noise_level'] > 0 else ts
        _, _pear, _, _ = compute_trace_metrics(
            np.asarray(_src['true']), np.asarray(_src['pred']))
        r_nf = float(fisher_pool(_pear)['r_mean'])
        header = f"vs noise-free ablated, $r$ = {r_nf:.2f}"
        draw_traces(
            ax, true_w, pred_w, stim_w, labels, step_v, time_ms,
            column_title=f"{col['label']} ({col['sigma']})",
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

    # --- Row 2: scatters [nf] [005 matched, 005 nf] [05 matched, 05 nf].
    # Panels e (low-noise "vs noisy ablated") and g (high-noise "vs noisy
    # ablated") are disabled — comparing the learned trace against the
    # perturbed gt is misleading. Uncomment to restore.
    scatter_panels = [
        (nf_gs[0, 0], matched[0]['true'], matched[0]['pred'],
         'voltage', 'rollout voltage', True,  'vs noise-free ablated'),
        # (lo_gs[0, 0], matched[1]['true'], matched[1]['pred'],
        #  '',                     '',                True,  'vs noisy ablated'),
        # e moves into the LEFT half of the low-noise column so it left-
        # aligns with the trace panel b above it (was lo_gs[0, 1]).
        (lo_gs[0, 0], nf[1]['true'],      nf[1]['pred'],
         '',                     '',                True,  'vs noise-free ablated'),
        # (hi_gs[0, 0], matched[2]['true'], matched[2]['pred'],
        #  '',                     '',                False, 'vs noisy ablated'),
        # f moves into the LEFT half of the high-noise column so it left-
        # aligns with the trace panel c above it (was hi_gs[0, 1]).
        (hi_gs[0, 0], nf[2]['true'],      nf[2]['pred'],
         '',                     '',                True,  'vs noise-free ablated'),
    ]

    scatter_axes = []
    for cell, x, y, xlbl, ylbl, show_fit, subtitle in scatter_panels:
        ax = fig.add_subplot(cell)
        # Suppress the printed Pearson r on "vs noisy ablated" panels —
        # comparing the learned trace against perturbed gt would be
        # misleading there.
        show_r = subtitle != 'vs noisy ablated'
        draw_scatter(ax, x, y, xlabel=xlbl, ylabel=ylbl,
                     show_fit=show_fit, show_r=show_r)
        if subtitle is not None:
            ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                    va='bottom', ha='center', fontsize=FS_TICK,
                    fontweight='normal')
        scatter_axes.append(ax)

    SCATTER_PULL_UP = 0.07
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
            add_panel_label(fig, ax, letter, dy=0.030)

    out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'fig_rollout_3col_noise_comparison_ablation50')
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_base}.pdf')
    print(f'Saved: {out_base}.png')


if __name__ == '__main__':
    main()
