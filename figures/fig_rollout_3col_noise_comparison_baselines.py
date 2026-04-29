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
    _slice, draw_traces, draw_scatter, add_panel_label, _load_bundle,
)


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
        run_cfg = f'{BASELINE_ROOT}/log/fly/{model}/config.yaml'
        if not os.path.isfile(run_cfg):
            sys.exit(f'missing run config: {run_cfg}')
        cols.append({
            'label'         : label,
            'sigma'         : sigma_tex,
            'model'         : model,
            'model_yaml'    : run_cfg,
            'base_yaml'     : run_cfg,         # the run config doubles as base
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
    # NOTE: the "vs noisy" panels are dropped for now — the kumarv4 run dirs
    # are read-only here, so we cannot generate the noisy-test variant or its
    # rollout bundle. Each column shows only the deterministic-test rollout
    # (primary bundle), giving a 3-trace + 3-scatter layout.
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
        r_nf = float(np.corrcoef(np.asarray(ts['true']).ravel(),
                                 np.asarray(ts['pred']).ravel())[0, 1])
        header = f"$r$ = {r_nf:.2f}"
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
         'ground truth voltage', 'rollout voltage', True),
        (bottom_gs[0, 1], primary[1]['true'], primary[1]['pred'],
         '', '', True),
        (bottom_gs[0, 2], primary[2]['true'], primary[2]['pred'],
         '', '', True),
    ]

    scatter_axes = []
    for cell, x, y, xlbl, ylbl, show_fit in scatter_panels:
        ax = fig.add_subplot(cell)
        draw_scatter(ax, x, y, xlabel=xlbl, ylabel=ylbl, show_fit=show_fit)
        scatter_axes.append(ax)

    # Scatter panels are wider than in the GNN figure (3 panels vs 5), so
    # aspect='equal' makes them taller — pull-up must be ~0 to avoid the row 1
    # x-ticks colliding with the row 2 panel tops.
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
