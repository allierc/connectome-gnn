"""Figure: 3-column rollout comparison (Known-ODE) — noise_free vs noise_005 vs noise_05.

Known-ODE counterpart of fig_rollout_3col_noise_comparison_known_ode.py: same
3-column noise-regime layout, same trace + scatter rows, but the GNN
models are replaced by the Known-ODE baselines
(flyvis_noise_{free,005,05}_blank50_known_ode_cv00).

Row 1 (traces): learned (black) vs training-style ground truth (green, thicker),
with a red stimulus trace overlaid on neurons that receive non-zero visual
input. The ground truth for the low / high model-noise columns comes from a
separately generated noisy test split (``noisy_test_data: true``) so that the
traces match the data the model was trained on.

Row 2 (scatters): 5 panels in one row.
    • noise-free     : 1 panel — learned vs ground truth
    • low model noise: 2 panels — learned vs noisy gt / learned vs noise-free gt
    • high model noise: 2 panels — same pair as low model noise

Every panel gets a letter label (a–h).

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/fig_rollout_3col_noise_comparison_known_ode.py

Output
------
    figures/fig_rollout_3col_noise_comparison_known_ode.{pdf,png}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Data root      : /groups/saalfeld/home/allierc/GraphData
# Configs        : <DATA_ROOT>/config/fly/flyvis_noise_free_blank50_known_ode_cv00.yaml
#                  <DATA_ROOT>/config/fly/flyvis_noise_005_blank50_known_ode_cv00.yaml
#                  <DATA_ROOT>/config/fly/flyvis_noise_05_blank50_known_ode_cv00.yaml
#                  (noisy-test twins generated on the fly:
#                   <DATA_ROOT>/config/fly/flyvis_noise_{free,005,05}_blank50_cv00_test.yaml)
# Stimulus root  : /groups/saalfeld/home/kumarv4/web_datasets/DAVIS2017-partial-test/
#                  /groups/saalfeld/home/allierc/signaling/DATAVIS/  (fallback)
# Training data  : <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_blank50_cv00/x_list_train/
#                  <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_blank50_cv00/{edge_index.pt, ode_params.pt}
# Test data      : <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_blank50_cv00_test/x_list_test/
# Trained models : <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_known_ode_cv00/models/best_model_with_0_graphs_0.pt
# Eval logs      : <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_known_ode_cv00/results/rollout_bundle.npz
#                  <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_known_ode_cv00/results/rollout_bundle_on_noise_{free,005,05}_blank50_cv00_test.npz
# Output         : figures/fig_rollout_3col_noise_comparison_known_ode.{pdf,png}
# ─────────────────────────────────────────────────────────────────────────────

import os
import shutil
import string
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'janne.matplotlibrc'))

import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
import numpy as np
import yaml


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
CFG_DIR   = f'{DATA_ROOT}/config/fly'

# DAVIS stimulus root — the original DAVIS2017-partial-test path referenced by
# the base yamls is no longer mounted, so fall back to the first candidate that
# actually contains JPEGImages/480p. The noisy-test dataset we generate for the
# 005 / 05 columns will use whichever DAVIS root is reachable.
_DAVIS_CANDIDATES = [
    '/groups/saalfeld/home/kumarv4/web_datasets/DAVIS2017-partial-test/',
    '/groups/saalfeld/home/allierc/signaling/DATAVIS/',
    os.environ.get('DATAVIS_ROOT', ''),
]
DAVIS_ROOT = next(
    (p for p in _DAVIS_CANDIDATES
     if p and os.path.isdir(os.path.join(p, 'JPEGImages/480p'))),
    None,
)

COLUMNS = [
    {
        'label': 'noise-free',
        'sigma': r'$\sigma = 0$',
        'model': 'flyvis_noise_free_blank50_known_ode_cv00',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_free_blank50_known_ode_cv00.yaml',
        'base_yaml': f'{REPO_ROOT}/config/fly/flyvis_noise_free_blank50.yaml',
        'cv00_dataset': 'flyvis_noise_free_blank50_cv00',
        'noise_level': 0.0,
    },
    {
        'label': 'low model noise',
        'sigma': r'$\sigma = 0.05$',
        'model': 'flyvis_noise_005_blank50_known_ode_cv00',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_005_blank50_known_ode_cv00.yaml',
        'base_yaml': f'{REPO_ROOT}/config/fly/flyvis_noise_005_blank50.yaml',
        'cv00_dataset': 'flyvis_noise_005_blank50_cv00',
        'noise_level': 0.05,
    },
    {
        'label': 'high model noise',
        'sigma': r'$\sigma = 0.5$',
        'model': 'flyvis_noise_05_blank50_known_ode_cv00',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_05_blank50_known_ode_cv00.yaml',
        'base_yaml': f'{REPO_ROOT}/config/fly/flyvis_noise_05_blank50.yaml',
        'cv00_dataset': 'flyvis_noise_05_blank50_cv00',
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
# Bottom row height ratio doubled (was [1.4*TS, 1.0]) so the scatter
# panels are roughly 2× the linear size of the previous left-half-only
# layout. FIG_H_IN bumped to keep the trace row physical size unchanged
# while the scatter row grows.
FIG_H_IN = 7.5


# ---------------------------------------------------------------------------
# Subprocess + yaml helpers
# ---------------------------------------------------------------------------
def _run(*args, tag):
    print(f'{tag} python GNN_Main.py {" ".join(args)}')
    subprocess.check_call(
        ['python', f'{REPO_ROOT}/GNN_Main.py', *args,
         '--output_root', DATA_ROOT],
        cwd=REPO_ROOT,
    )


def _clone_base(base_yaml, out_yaml, dataset_name, description, overrides):
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg['description'] = description
    cfg['dataset']     = dataset_name
    sim = cfg['simulation']
    sim['seed'] = 42
    sim.update(overrides)
    with open(out_yaml, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Noisy-test-variant pipeline (for noise_005 / noise_05 columns only)
# ---------------------------------------------------------------------------
def test_variant_for(col):
    ds_test = f"{col['cv00_dataset']}_test"
    return {
        'dataset'  : ds_test,
        'yaml'     : f'{CFG_DIR}/{ds_test}.yaml',
        'data_dir' : f'{DATA_ROOT}/graphs_data/fly/{ds_test}',
        'bundle'   : (f"{DATA_ROOT}/log/fly/{col['model']}/results/"
                      f"rollout_bundle_on_{ds_test.replace('flyvis_', '')}.npz"),
    }


def ensure_test_variant(col):
    nv = test_variant_for(col)

    if not os.path.isfile(nv['yaml']):
        print(f"[{col['model']}] cloning noisy-test variant -> {nv['yaml']}")
        sim_overrides = {
            'noise_model_level': col['noise_level'],
            'noisy_test_data'  : True,
        }
        if DAVIS_ROOT is not None:
            sim_overrides['datavis_roots'] = [DAVIS_ROOT]
        _clone_base(
            col['base_yaml'], nv['yaml'], nv['dataset'],
            description=(
                f"Noisy-test twin of {col['cv00_dataset']} for figure "
                "fig_rollout_3col_noise_comparison_known_ode.py: seed=42, "
                f"noise_model_level={col['noise_level']}, noisy_test_data=true."
            ),
            overrides=sim_overrides,
        )
    else:
        print(f"[{col['model']}] noisy config exists: {nv['yaml']}")

    marker = f"{nv['data_dir']}/noisy_test_data.ok"
    if not os.path.isfile(marker):
        if os.path.isdir(nv['data_dir']):
            print(f"[{col['model']}] removing stale {nv['data_dir']}")
            shutil.rmtree(nv['data_dir'])
        print(f"[{col['model']}] generating noisy test dataset {nv['dataset']} "
              "— tens of minutes")
        _run('-o', 'generate', nv['yaml'], tag=f"[{col['model']}]")
        if not os.path.isfile(marker):
            sys.exit(f'expected marker missing after generation: {marker}')
    else:
        print(f"[{col['model']}] noisy test dataset exists: {nv['data_dir']}")

    if not os.path.isfile(nv['bundle']):
        if not os.path.isfile(col['model_yaml']):
            sys.exit(f"missing model yaml for {col['model']}: {col['model_yaml']}")
        print(f"[{col['model']}] running rollout on {nv['dataset']}")
        _run('-o', 'test', col['model_yaml'], 'best', nv['yaml'],
             tag=f"[{col['model']}]")
    else:
        print(f"[{col['model']}] rollout bundle exists: {nv['bundle']}")


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


def load_primary_bundle(col):
    """Bundle from training-time test (deterministic / noise-free test split)."""
    path = f"{DATA_ROOT}/log/fly/{col['model']}/results/rollout_bundle.npz"
    return _load_bundle(path)


def load_test_bundle(col):
    return _load_bundle(test_variant_for(col)['bundle'])


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
        # Red stimulus trace — only for neurons with non-trivial visual input
        # (in our 12 selected types only R1 has stim). Plotted slightly BELOW
        # the neuron's voltage trace and ON TOP of the other lines, so it's
        # visually distinct from R1.
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
        # "neurons" group title, placed to the left of the stacked type names.
        ax.set_ylabel('neurons', fontsize=FS_LABEL, labelpad=18)

    # Header lines placed INSIDE the axes top-left, left-aligned with a
    # translucent white bbox. Moving them inside frees the space above the
    # axes, so the column title can sit close to the Am (top) trace.
    if header_text is None:
        header_lines = []
    elif isinstance(header_text, (list, tuple)):
        header_lines = list(header_text)
    else:
        header_lines = [header_text]
    HEADER_DY = 0.05   # axes-fraction step between stacked header lines
    for k, line in enumerate(header_lines):
        y = 0.99 - k * HEADER_DY
        ax.text(0.015, y, line, transform=ax.transAxes,
                va='top', ha='left', fontsize=FS_TICK,
                fontweight='normal',
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=0.4))
    ax._column_title = column_title

    # Extra headroom above the topmost trace (Am) so the in-axes header
    # labels at axes-frac 0.99 sit clearly above Am's wiggle peaks.
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
        # Drop the entire time scale (no ticks, no labels, no spine).
        # Skip _trim_axis here — it errors when there are zero ticks.
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


def draw_scatter(ax, x_all, y_all, xlabel, ylabel, show_fit=True):
    """Hexbin density of y vs x with per-neuron Fisher-pooled Pearson r.

    The hexbin uses subsampled flattened (x, y) for the visual; the
    correlation reported is computed on the full 2-D traces using the
    same recipe graph_tester / cv-tables use:
    `compute_trace_metrics` → per-neuron `pearsonr` → `fisher_pool`
    `r_mean`. Values printed here therefore agree with the
    {onestep,rollout}_pearson rows of the cv summaries.
    """
    from connectome_gnn.utils import compute_trace_metrics, fisher_pool
    x_sub, y_sub, n_full = _subsample_pair(x_all, y_all)

    _, _pear, _, _ = compute_trace_metrics(
        np.asarray(x_all), np.asarray(y_all))
    _fp = fisher_pool(_pear)
    r, r_sd = float(_fp['r_mean']), float(_fp['r_sd_sym'])

    lo, hi = SCATTER_LO, SCATTER_HI

    ax.hexbin(x_sub, y_sub, gridsize=140, bins='log', cmap='magma_r',
              mincnt=1, extent=(lo, hi, lo, hi), linewidths=0.0)

    ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(axis='both', labelsize=FS_TICK)

    # Explicit ticks at the axis endpoints, plus 0.
    ticks = [lo, 0.0, hi]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])
    _trim_axis(ax)

    # Pearson r ± SD (Fisher-pooled per-neuron) inside the axes, top-left.
    ax.text(0.05, 0.97, f"$r$ = {r:.2f} $\\pm$ {r_sd:.2f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=FS_TICK)


def add_panel_label(fig, ax, letter, dx=0.015, dy=0.01):
    """Place a bold lowercase letter above and slightly left of the axes box.

    Uses the axes' geometric position (not its tight bbox) so panels with and
    without axis labels all place their letter at the same relative offset.
    """
    pos = ax.get_position()
    fig.text(pos.x0 - dx, pos.y1 + dy, letter,
             fontsize=PANEL_LBL, fontweight='bold',
             va='bottom', ha='left', color='black',
             transform=fig.transFigure)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Step 1: ensure the "noisy variant" bundle for EVERY column (including
    # noise-free — where `noisy_test_data=true` is a no-op for the noise
    # generation, but it makes all three columns share the same freshly
    # generated test dataset with matching DAVIS videos + seed=42, so panel a
    # and panels b/c use identical stimuli and can be compared directly.
    for col in COLUMNS:
        ensure_test_variant(col)

    # Step 2: load bundles. Use the noisy-variant bundle everywhere so
    # traces and the "rollout vs gt" scatter all derive from the same test
    # dataset per column (primary bundles are kept only as the "noise-free
    # gt" comparator for the low/high columns where the two differ).
    primary = [load_primary_bundle(col) for col in COLUMNS]
    test_data = [load_test_bundle(col)   for col in COLUMNS]

    # For traces and the first-per-group scatter, always use the noisy
    # variant. For the secondary scatters (vs noise-free) in the low/high
    # columns, use the primary (deterministic) bundle.
    trace_src = test_data

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

    # Shared step_v across columns. Both the per-trace amplitude (in
    # draw_traces) and step_v scale with TRACE_SHRINK so the trace-to-gap
    # ratio is preserved while the absolute size shrinks.
    step_vs = [3.0 * TRACE_SHRINK * float(np.std(_slice(ts['true'], neuron_idx)))
               for ts in trace_src]
    step_v = max(0.5 * TRACE_SHRINK, max(step_vs))
    n_frames = TRACE_END - TRACE_START
    time_ms = np.arange(n_frames) * DT_MS + TRACE_START * DT_MS

    # Figure layout: outer 2-row split; top row = 3 trace axes, bottom row = 5
    # scatter axes in one row (1 for nf + 2 for 005 + 2 for 05).
    _nf_green = False
    if True:
        fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), constrained_layout=False)
        outer = mgs.GridSpec(
            2, 1, figure=fig,
            height_ratios=[1.4 * TRACE_SHRINK, 2.0],
            left=0.06, right=0.98, top=0.97, bottom=0.04,
            hspace=0.0,
        )
        # Top and bottom rows share identical column boundaries — three equal-width
        # groups with the same wspace — so noise-group boundaries in the bottom row
        # align with the trace-column boundaries in the top row.
        TOP_WSPACE    = 0.25
        GROUP_WSPACE  = 0.35   # separation between the two panels within a group
        top_gs   = mgs.GridSpecFromSubplotSpec(1, 3, outer[0, 0], wspace=TOP_WSPACE)
        group_gs = mgs.GridSpecFromSubplotSpec(1, 3, outer[1, 0], wspace=TOP_WSPACE)
        # Each noise group used to hold up to 2 panels (left / right); after
        # dropping the "vs noisy" panels (e, g) we collapse each group to a
        # single full-width cell so panels d / e / f are ~2× wider than the
        # half-cell layout. The remaining panel sits at [0, 0] of each
        # 1×1 sub-grid.
        nf_gs = mgs.GridSpecFromSubplotSpec(1, 1, group_gs[0, 0])
        lo_gs = mgs.GridSpecFromSubplotSpec(1, 1, group_gs[0, 1])
        hi_gs = mgs.GridSpecFromSubplotSpec(1, 1, group_gs[0, 2])

        # --- Row 1: traces ---
        trace_axes = []
        for c, (col, ts, prim) in enumerate(zip(COLUMNS, trace_src, primary)):
            ax = fig.add_subplot(top_gs[0, c])
            true_w = _slice((trace_src[0]['true'] if _nf_green else ts['true']), neuron_idx)
            pred_w = _slice(ts['pred'], neuron_idx)
            stim_w = (_slice(ts['stim'], neuron_idx)
                      if ts.get('stim') is not None else None)
            # Per-neuron Pearson r, Fisher-z pooled — same recipe graph_tester /
            # cv-table use. Single-line header: only the "vs noise-free" comparison
            # is printed on the noisy columns (b, c); the noisy-vs-learned r is
            # suppressed because the gt itself is perturbed.
            from connectome_gnn.utils import compute_trace_metrics, fisher_pool
            _src = prim if col['noise_level'] > 0 else ts
            _label = "vs noise-free, " if col['noise_level'] > 0 else ""
            _, _pear, _, _ = compute_trace_metrics(
                np.asarray(_src['true']), np.asarray(_src['pred']))
            _fp = fisher_pool(_pear)
            r, r_sd = float(_fp['r_mean']), float(_fp['r_sd_sym'])
            header = f"{_label}$r$ = {r:.2f} $\\pm$ {r_sd:.2f}"
            draw_traces(
                ax, true_w, pred_w, stim_w, labels, step_v, time_ms,
                column_title=f"{col['label']} ({col['sigma']})",
                show_type_labels=(c == 0),
                show_xlabel=(c == 0),
                header_text=header,
            )
            trace_axes.append(ax)

        # Draw column titles via fig.text at a uniform fig-y across all three
        # trace columns so the title baseline lines up with the panel letters
        # a / b / c (drawn at the same fig-y by add_panel_label below).
        fig.canvas.draw()
        TRACE_TITLE_DY = 0.02   # fig-fraction above each trace axes' top edge
                                # (small — headers now live inside the axes)
        for ax_t in trace_axes:
            pos = ax_t.get_position()
            x_center = pos.x0 + pos.width / 2
            fig.text(x_center, pos.y1 + TRACE_TITLE_DY, ax_t._column_title,
                     va='bottom', ha='center', fontsize=FS_LABEL,
                     fontweight='normal', transform=fig.transFigure)

        # --- Row 2: five scatter panels, grouped [nf] [005 noisy, 005 nf] [05 noisy, 05 nf].
        # Axis labels are shown only the FIRST time they occur, so g and h inherit
        # their axis meaning from e and f (same kind of plot, different noise level).
        # Panel tuple: (cell, x, y, xlabel, ylabel, show_fit, subtitle)
        # Sub-titles distinguish "noisy" vs "noise-free" within the low/high groups;
        # panel d has no subtitle (its noise-free status is implicit).
        scatter_panels = [
            # Panel d uses the noisy-variant bundle so its stimulus matches panels
            # b and c (same DAVIS videos, same seed=42).
            (nf_gs[0, 0], test_data[0]['true'], test_data[0]['pred'],
             'ground truth voltage', 'rollout voltage', True,  None),
            # Panels e (low-noise "vs noisy") and g (high-noise "vs noisy") are
            # disabled — comparing the learned trace against the perturbed gt is
            # misleading. Uncomment to restore.
            # (lo_gs[0, 0], test_data[1]['true'],   test_data[1]['pred'],
            #  '',                     '',                True,  'vs noisy'),
            # e moves into the LEFT half of the low-noise column so it left-
            # aligns with the trace panel b above it (was lo_gs[0, 1]).
            (lo_gs[0, 0], primary[1]['true'], primary[1]['pred'],
             '',                     '',                True,  'vs noise-free'),
            # (hi_gs[0, 0], test_data[2]['true'],   test_data[2]['pred'],
            #  '',                     '',                False, 'vs noisy'),
            # f moves into the LEFT half of the high-noise column so it left-
            # aligns with the trace panel c above it (was hi_gs[0, 1]).
            (hi_gs[0, 0], primary[2]['true'], primary[2]['pred'],
             '',                     '',                True,  'vs noise-free'),
        ]

        scatter_axes = []
        for cell, x, y, xlbl, ylbl, show_fit, subtitle in scatter_panels:
            ax = fig.add_subplot(cell)
            draw_scatter(ax, x, y, xlabel=xlbl, ylabel=ylbl, show_fit=show_fit)
            # Subtitle ("vs noisy" / "vs noise-free") above the axes.
            # R² and slope are drawn INSIDE the axes (top-left) by draw_scatter.
            if subtitle is not None:
                ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                        va='bottom', ha='center', fontsize=FS_TICK,
                        fontweight='normal')
            scatter_axes.append(ax)

        # Pull the entire scatter row upward by a fixed fig-fraction so the visible
        # gap between the trace row and the scatter row shrinks. Axis headers
        # (positioned via transAxes coords) shift up with their axes.
        SCATTER_PULL_UP = 0.16
        for ax in scatter_axes:
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 + SCATTER_PULL_UP,
                             pos.width, pos.height])

        # --- Panel labels (a–h) on every panel ---
        all_axes = trace_axes + scatter_axes
        letters = list(string.ascii_lowercase[:len(all_axes)])
        fig.canvas.draw()
        # Trace letters (a, b, c) sit at the same fig-y as the column titles
        # (TRACE_TITLE_DY above the trace axes), so the title baseline aligns
        # with the letter. Scatter letters keep the default tighter offset.
        for ax, letter in zip(all_axes, letters):
            if ax in trace_axes:
                add_panel_label(fig, ax, letter, dy=TRACE_TITLE_DY)
            else:
                add_panel_label(fig, ax, letter)

        out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'fig_rollout_3col_noise_comparison_known_ode')
        fig.savefig(out_base + '.pdf', bbox_inches='tight')
        fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_base}.pdf')
        print(f'Saved: {out_base}.png')


if __name__ == '__main__':
    main()
