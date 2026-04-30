"""Figure: 3-column rollout comparison for MLP and EED baselines.

Mirrors ``fig_rollout_3col_noise_comparison.py`` (the GNN/unified figure) but
points at the MLP and EED ``_unified2_cv00`` runs that live under

    /groups/saalfeld/home/kumarv4/repos/connectome-gnn/log/fly/

For each architecture (mlp, eed) it builds the same 3-column trace + 5-panel
scatter layout (noise-free / σ=0.05 / σ=0.5) and writes

    figures/fig_rollout_3col_noise_comparison_mlp.{pdf,png}
    figures/fig_rollout_3col_noise_comparison_eed.{pdf,png}

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/fig_rollout_3col_noise_comparison_baselines.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Data root      : /groups/saalfeld/home/kumarv4/repos/connectome-gnn   (BASELINE_REPO; baseline runs live here)
# Configs        : <DATA_ROOT>/config/fly/flyvis_noise_{free,005,05}_mlp_unified2_cv00.yaml
#                  <DATA_ROOT>/config/fly/flyvis_noise_{free,005,05}_eed_unified2_cv00.yaml
#                  (noisy-test twins generated on the fly with suffix _noisy)
# Stimulus root  : /groups/saalfeld/home/kumarv4/web_datasets/DAVIS2017-partial-test/
#                  /groups/saalfeld/home/allierc/signaling/DATAVIS/  (fallback)
# Training data  : <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_{mlp,eed}_unified2_cv00/x_list_train/
# Test data      : <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_{mlp,eed}_unified2_cv00/x_list_test/
#                  <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_{mlp,eed}_unified2_cv00_noisy/x_list_test/
# Trained models : <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_{mlp,eed}_unified2_cv00/models/best_model_with_0_graphs_0.pt
# Eval logs      : <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_{mlp,eed}_unified2_cv00/results/rollout_bundle.npz
#                  <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_{mlp,eed}_unified2_cv00/results/rollout_bundle_on_noise_{free,005,05}_{mlp,eed}_unified2_cv00_noisy.npz
# Output         : figures/fig_rollout_3col_noise_comparison_{mlp,eed}.{pdf,png}
# ─────────────────────────────────────────────────────────────────────────────

import os
import shutil
import string
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
_RC = os.path.join(os.path.dirname(__file__), 'janne.matplotlibrc')
if os.path.isfile(_RC):
    matplotlib.rc_file(_RC)

# Make the sibling figure module importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Reuse plotting + selection helpers + style constants from the GNN figure so
# the two figures are visually identical apart from the model identity.
from fig_rollout_3col_noise_comparison import (  # type: ignore
    SELECTED_TYPES, DT_MS, FS_LABEL, FS_TICK, TRACE_SHRINK,
    FIG_W_IN, FIG_H_IN, TRACE_START, TRACE_END,
    SCATTER_LO, SCATTER_HI,
    _slice, _trim_axis, draw_traces, add_panel_label, _load_bundle,
)
from connectome_gnn.utils import compute_trace_metrics, fisher_pool


def _pooled_r(true_arr, pred_arr):
    """Fisher-z pooled per-neuron Pearson r and its symmetric SD."""
    _, pear, _, _ = compute_trace_metrics(np.asarray(true_arr),
                                          np.asarray(pred_arr))
    fz = fisher_pool(pear)
    return float(fz['r_mean']), float(fz['r_sd_sym'])


def draw_scatter(ax, true_arr, pred_arr, xlabel, ylabel, show_fit=True):
    """Hexbin density of pred vs true, annotated with Fisher-z pooled r ± SD.

    Replaces the upstream draw_scatter (which prints R² + slope) — same hexbin
    rendering, different annotation.
    """
    r_mean, r_sd = _pooled_r(true_arr, pred_arr)
    # x-axis: ground truth; y-axis: prediction (same convention as upstream).
    x = np.asarray(true_arr).reshape(-1).astype(np.float32)
    y = np.asarray(pred_arr).reshape(-1).astype(np.float32)
    lo, hi = SCATTER_LO, SCATTER_HI
    ax.hexbin(x, y, gridsize=140, bins='log', cmap='magma_r',
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
    ax.text(0.05, 0.97, f"$r$ = {r_mean:.2f} $\\pm$ {r_sd:.2f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=FS_TICK)


# ── paths ────────────────────────────────────────────────────────────────────
GNN_REPO_ROOT  = '/workspace/connectome-gnn'  # this repo (figure script lives here)
BASELINE_REPO  = '/groups/saalfeld/home/kumarv4/repos/connectome-gnn'
BASELINE_ROOT  = BASELINE_REPO  # output_root for baseline runs (logs + data)
CFG_DIR        = f'{BASELINE_REPO}/config/fly'

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


def build_columns(arch):
    """Three-column spec for one architecture (``arch`` ∈ {'mlp','eed'})."""
    base = {
        'noise_free': ('flyvis_noise_free_' + arch + '_unified2_cv00', 0.0,    r'$\sigma = 0$'),
        'noise_005' : ('flyvis_noise_005_'  + arch + '_unified2_cv00', 0.05,   r'$\sigma = 0.05$'),
        'noise_05'  : ('flyvis_noise_05_'   + arch + '_unified2_cv00', 0.5,    r'$\sigma = 0.5$'),
    }
    cols = []
    for label, (model, sigma, sigma_tex) in zip(
        ['noise-free', 'low model noise', 'high model noise'],
        base.values(),
    ):
        # GNN_Main.py extracts pre_folder from the yaml's parent dir, so the
        # model yaml must live under config/fly/ (copied from log/fly/<run>/).
        model_yaml = f'{CFG_DIR}/{model}.yaml'
        run_cfg    = f'{BASELINE_ROOT}/log/fly/{model}/config.yaml'
        if not os.path.isfile(model_yaml):
            sys.exit(f'missing model yaml at config/fly/: {model_yaml}\n'
                     f'  cp {run_cfg} {model_yaml}')
        cols.append({
            'label'         : label,
            'sigma'         : sigma_tex,
            'model'         : model,
            'model_yaml'    : model_yaml,
            'base_yaml'     : model_yaml,      # the run config doubles as base
            'cv00_dataset'  : model,           # dataset name == model name here
            'noise_level'   : sigma,
        })
    return cols


# ---------------------------------------------------------------------------
# Subprocess + yaml helpers (use the baseline repo's GNN_Main.py + paths)
# ---------------------------------------------------------------------------
def _run(*args, tag):
    print(f'{tag} python GNN_Main.py {" ".join(args)}')
    subprocess.check_call(
        ['python', f'{BASELINE_REPO}/GNN_Main.py', *args,
         '--output_root', BASELINE_ROOT],
        cwd=BASELINE_REPO,
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


def noisy_variant_for(col):
    ds_noisy = f"{col['cv00_dataset']}_noisy"
    return {
        'dataset'  : ds_noisy,
        'yaml'     : f'{CFG_DIR}/{ds_noisy}.yaml',
        'data_dir' : f'{BASELINE_ROOT}/graphs_data/fly/{ds_noisy}',
        'bundle'   : (f"{BASELINE_ROOT}/log/fly/{col['model']}/results/"
                      f"rollout_bundle_on_{ds_noisy.replace('flyvis_', '')}.npz"),
    }


def ensure_noisy_variant(col):
    nv = noisy_variant_for(col)

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
                "fig_rollout_3col_noise_comparison_baselines.py: seed=42, "
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
        print(f"[{col['model']}] running rollout on {nv['dataset']}")
        _run('-o', 'test', col['model_yaml'], 'best', nv['yaml'],
             tag=f"[{col['model']}]")
    else:
        print(f"[{col['model']}] rollout bundle exists: {nv['bundle']}")


def load_primary_bundle(col):
    return _load_bundle(
        f"{BASELINE_ROOT}/log/fly/{col['model']}/results/rollout_bundle.npz"
    )


def load_noisy_bundle(col):
    return _load_bundle(noisy_variant_for(col)['bundle'])


# ---------------------------------------------------------------------------
# Figure builder (mirrors main() in the GNN figure)
# ---------------------------------------------------------------------------
def build_figure(columns, out_base):
    # Show only "vs noise-free" comparison (drop the noisy-test panels): trace
    # row 1 displays the deterministic-test rollout, scatter row 2 has one
    # panel per noise level — 6 panels total.
    primary = [load_primary_bundle(col) for col in columns]
    trace_src = primary

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
        hspace=0.1,
    )
    TOP_WSPACE = 0.25
    top_gs    = mgs.GridSpecFromSubplotSpec(1, 3, outer[0, 0], wspace=TOP_WSPACE)
    bottom_gs = mgs.GridSpecFromSubplotSpec(1, 3, outer[1, 0], wspace=TOP_WSPACE)

    trace_axes = []
    for c, (col, ts) in enumerate(zip(columns, trace_src)):
        ax = fig.add_subplot(top_gs[0, c])
        true_w = _slice(ts['true'], neuron_idx)
        pred_w = _slice(ts['pred'], neuron_idx)
        stim_w = (_slice(ts['stim'], neuron_idx)
                  if ts.get('stim') is not None else None)
        # Per-neuron Pearson r, Fisher-z pooled — same recipe the
        # graph_tester / cv table use, so the values shown here agree
        # with the per-condition Pearson rows in the TeX/MD summaries.
        r_mean, r_sd = _pooled_r(ts['true'], ts['pred'])
        header = f"$r$ = {r_mean:.2f} $\\pm$ {r_sd:.2f}"
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

    # One scatter per column, all "vs noise-free" (deterministic test gt).
    scatter_panels = [
        (bottom_gs[0, 0], primary[0]['true'], primary[0]['pred'],
         'ground truth voltage', 'rollout voltage'),
        (bottom_gs[0, 1], primary[1]['true'], primary[1]['pred'],
         '', ''),
        (bottom_gs[0, 2], primary[2]['true'], primary[2]['pred'],
         '', ''),
    ]

    scatter_axes = []
    for cell, x, y, xlbl, ylbl in scatter_panels:
        ax = fig.add_subplot(cell)
        draw_scatter(ax, x, y, xlabel=xlbl, ylabel=ylbl)
        scatter_axes.append(ax)

    # Wider panels (3 vs 5) at aspect='equal' are taller, so no upward pull.
    SCATTER_PULL_UP = 0.0
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

    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_base}.pdf')
    print(f'Saved: {out_base}.png')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for arch in ('mlp', 'eed'):
        cols = build_columns(arch)
        out_base = os.path.join(
            here, f'fig_rollout_3col_noise_comparison_{arch}'
        )
        print(f'\n===== {arch.upper()} =====')
        build_figure(cols, out_base)


if __name__ == '__main__':
    main()
