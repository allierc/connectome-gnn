"""Figure: 3-column rollout comparison — noise_free vs noise_005 vs noise_05 (cv03).

Twin of ``fig_rollout_3col_noise_comparison.py`` (which uses cv00) pointing at
the cv03 fold instead. Motivation: ``flyvis_noise_free_blank50_unified_cv03``
is the outlier of the 5-fold CV — its rollout explodes (predictions clamp at
±100 starting at frame ~160). This twin makes the cv03 trace + scatter row
directly comparable to the cv00 figure so the explosion is visible side-by-
side.

Row 1 (traces): learned (black) vs training-style ground truth (green, thicker),
with a red stimulus trace overlaid on neurons that receive non-zero visual
input. The ground truth for the low / high model-noise columns comes from a
separately generated noisy test split (``noisy_test_data: true``) so that the
traces match the data the model was trained on.

Row 2 (scatters): one panel per noise level (vs noise-free gt for the noisy
columns).

Every panel gets a letter label (a–f).

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/fig_rollout_3col_noise_comparison_cv03.py

Output
------
    figures/fig_rollout_3col_noise_comparison_cv03.{pdf,png}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Configs        : <DATA_ROOT>/config/fly/flyvis_noise_free_blank50_unified_cv03.yaml
#                  <DATA_ROOT>/config/fly/flyvis_noise_005_blank50_unified_cv03.yaml
#                  <DATA_ROOT>/config/fly/flyvis_noise_05_blank50_unified_cv03.yaml
#                  (noisy-test twins generated on the fly:
#                   <DATA_ROOT>/config/fly/flyvis_noise_{free,005,05}_blank50_cv03_test.yaml)
# Training data  : <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_blank50_cv03/x_list_train/
#                  <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_blank50_cv03/{edge_index.pt, ode_params.pt}
# Test data      : <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_blank50_cv03_test/x_list_test/
# Trained models : <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_unified_cv03/models/best_model_with_0_graphs_0.pt
# Eval logs      : <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_unified_cv03/results/rollout_bundle.npz
#                  <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_unified_cv03/results/rollout_bundle_on_noise_{free,005,05}_blank50_cv03_test.npz
# Output         : figures/fig_rollout_3col_noise_comparison_cv03{,_nf_green}.{pdf,png}
# ─────────────────────────────────────────────────────────────────────────────

import os
import shutil
import string
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'unified_style.matplotlibrc'))

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
DATA_ROOT = os.environ.get('TRAINED_MODEL_OUTPUT_ROOT', '.')
CFG_DIR   = f'{DATA_ROOT}/config/fly'

# DAVIS stimulus root — same fallback chain as the cv00 figure.
_DAVIS_CANDIDATES = [
    os.environ.get('DATAVIS_TEST_ROOT', ''),
]
DAVIS_ROOT = next(
    (p for p in _DAVIS_CANDIDATES
     if p and os.path.isdir(os.path.join(p, 'JPEGImages/480p'))),
    None,
)

# Three noise levels at cv03. The 'cv_dataset' value is the per-fold base
# dataset name (used to derive the noisy-test variant). The base_yaml stays
# noise-level-specific (no _cvNN suffix on those — they are the fold-agnostic
# templates from /workspace/connectome-gnn/config/fly).
COLUMNS = [
    {
        'label': 'noise-free',
        'sigma': r'$\sigma = 0$',
        'model': 'flyvis_noise_free_blank50_unified_cv03',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_free_blank50_unified_cv03.yaml',
        'base_yaml': f'{REPO_ROOT}/config/fly/flyvis_noise_free_blank50.yaml',
        'cv_dataset': 'flyvis_noise_free_blank50_cv03',
        'noise_level': 0.0,
    },
    {
        'label': 'low model noise',
        'sigma': r'$\sigma = 0.05$',
        'model': 'flyvis_noise_005_blank50_unified_cv03',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_005_blank50_unified_cv03.yaml',
        'base_yaml': f'{REPO_ROOT}/config/fly/flyvis_noise_005_blank50.yaml',
        'cv_dataset': 'flyvis_noise_005_blank50_cv03',
        'noise_level': 0.05,
    },
    {
        'label': 'high model noise',
        'sigma': r'$\sigma = 0.5$',
        'model': 'flyvis_noise_05_blank50_unified_cv03',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_05_blank50_unified_cv03.yaml',
        'base_yaml': f'{REPO_ROOT}/config/fly/flyvis_noise_05_blank50.yaml',
        'cv_dataset': 'flyvis_noise_05_blank50_cv03',
        'noise_level': 0.5,
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

TRACE_SHRINK = 0.65

# cv03-specific: noise-free rollout explodes (clamps at ±100). Plotting it
# raw turns the trace panel into a solid black wall after frame ~160. Clip
# predicted voltages to ±PRED_CLAMP around each neuron's baseline so the
# explosion is visible as a saturated line at the panel edge instead of
# obliterating the panel. Ground-truth traces are not clipped.
PRED_CLAMP = 1.0

FIG_W_IN = 18.0 * 0.3937
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
    ds_test = f"{col['cv_dataset']}_test"
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
                f"Noisy-test twin of {col['cv_dataset']} for figure "
                "fig_rollout_3col_noise_comparison_cv03.py: seed=42, "
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
        pred_clipped = np.clip(pred_w[i], bl - PRED_CLAMP, bl + PRED_CLAMP)
        ax.plot(time_ms, s * (pred_clipped - bl) + i * step_v,
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


def draw_scatter(ax, x_all, y_all, xlabel, ylabel, show_fit=True):
    """Hexbin density of y vs x with per-neuron Fisher-pooled Pearson r."""
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
    for col in COLUMNS:
        ensure_test_variant(col)

    primary = [load_primary_bundle(col) for col in COLUMNS]
    test_data = [load_test_bundle(col)   for col in COLUMNS]

    trace_src = test_data

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

    for _nf_green in (False, True):
        fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), constrained_layout=False)
        outer = mgs.GridSpec(
            2, 1, figure=fig,
            height_ratios=[1.4 * TRACE_SHRINK, 2.0],
            left=0.06, right=0.98, top=0.97, bottom=0.04,
            hspace=0.0,
        )
        TOP_WSPACE    = 0.25
        GROUP_WSPACE  = 0.35
        top_gs   = mgs.GridSpecFromSubplotSpec(1, 3, outer[0, 0], wspace=TOP_WSPACE)
        group_gs = mgs.GridSpecFromSubplotSpec(1, 3, outer[1, 0], wspace=TOP_WSPACE)
        nf_gs = mgs.GridSpecFromSubplotSpec(1, 1, group_gs[0, 0])
        lo_gs = mgs.GridSpecFromSubplotSpec(1, 1, group_gs[0, 1])
        hi_gs = mgs.GridSpecFromSubplotSpec(1, 1, group_gs[0, 2])

        trace_axes = []
        for c, (col, ts, prim) in enumerate(zip(COLUMNS, trace_src, primary)):
            ax = fig.add_subplot(top_gs[0, c])
            true_w = _slice((trace_src[0]['true'] if _nf_green else ts['true']), neuron_idx)
            pred_w = _slice(ts['pred'], neuron_idx)
            stim_w = (_slice(ts['stim'], neuron_idx)
                      if ts.get('stim') is not None else None)
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

        fig.canvas.draw()
        TRACE_TITLE_DY = 0.02
        for ax_t in trace_axes:
            pos = ax_t.get_position()
            x_center = pos.x0 + pos.width / 2
            fig.text(x_center, pos.y1 + TRACE_TITLE_DY, ax_t._column_title,
                     va='bottom', ha='center', fontsize=FS_LABEL,
                     fontweight='normal', transform=fig.transFigure)

        scatter_panels = [
            (nf_gs[0, 0], test_data[0]['true'], test_data[0]['pred'],
             'ground truth voltage', 'rollout voltage', True,  None),
            (lo_gs[0, 0], primary[1]['true'], primary[1]['pred'],
             '',                     '',                True,  'vs noise-free'),
            (hi_gs[0, 0], primary[2]['true'], primary[2]['pred'],
             '',                     '',                True,  'vs noise-free'),
        ]

        scatter_axes = []
        for cell, x, y, xlbl, ylbl, show_fit, subtitle in scatter_panels:
            ax = fig.add_subplot(cell)
            draw_scatter(ax, x, y, xlabel=xlbl, ylabel=ylbl, show_fit=show_fit)
            if subtitle is not None:
                ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                        va='bottom', ha='center', fontsize=FS_TICK,
                        fontweight='normal')
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
                                'fig_rollout_3col_noise_comparison_cv03'
                                + ('_nf_green' if _nf_green else ''))
        fig.savefig(out_base + '.pdf', bbox_inches='tight')
        fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_base}.pdf')
        print(f'Saved: {out_base}.png')


if __name__ == '__main__':
    main()
