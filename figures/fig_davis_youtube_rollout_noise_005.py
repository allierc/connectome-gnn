"""
Figure: YT-trained GNN rollout on DAVIS (cv00) — noisy vs noise-free.

Two stacked panels (12 representative cell types, 1 000 frames each):
  a) model rollout vs DAVIS test data with σ = 0.05 process noise
     (noisy cv00: flyvis_noise_005_cv00, emitted by run_GNN_conditions.py)
  b) model rollout vs noise-free twin of the same DAVIS stimulus
     (same DAVIS videos + seed 42, noise_model_level=0)

End-to-end pipeline (each step cached by file existence, no manual commands):
  1. Clone <base>.yaml -> {cv00,cv00_nf}.yaml (DAVIS noisy + noise-free)
  2. python GNN_Main.py -o generate <cfg>            — regenerate ODE
  3. python GNN_Main.py -o test <model> best <cfg>   — cross-dataset rollout
  4. Plot both panels.

Usage
-----
    conda run -n neural-graph-linux python figures/fig_davis_youtube_rollout_noise_005.py

Output
------
    figures/fig_davis_youtube_rollout_noise_005.{png,pdf,jpg}
"""

import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D


# ── identifiers ──────────────────────────────────────────────────────────────
REPO_ROOT   = '/workspace/connectome-gnn'
DATA_ROOT   = '/groups/saalfeld/home/allierc/GraphData'

BASE        = 'flyvis_noise_005'                     # condition short-name
MODEL_NAME  = f'{BASE}_yt_per_cond_cv00'             # YT-trained model (fold 0)
CV00_DS     = f'{BASE}_cv00'                         # DAVIS noisy test (σ=0.05)
NF_TWIN     = f'{BASE}_cv00_nf'                      # DAVIS noise-free twin

# YT CV YAMLs live on shared FS; DAVIS base + nf twin live under it too.
CFG_DIR     = f'{DATA_ROOT}/config/fly'
BASE_YAML   = f'{REPO_ROOT}/config/fly/{BASE}.yaml'  # static DAVIS base (source of clone)
CV00_YAML   = f'{CFG_DIR}/{CV00_DS}.yaml'
NF_YAML     = f'{CFG_DIR}/{NF_TWIN}.yaml'

LOG_DIR     = f'{DATA_ROOT}/log/fly/{MODEL_NAME}'
RESULTS     = f'{LOG_DIR}/results'
CV00_DATA   = f'{DATA_ROOT}/graphs_data/fly/{CV00_DS}'
TWIN_DATA   = f'{DATA_ROOT}/graphs_data/fly/{NF_TWIN}'

TOP_BUNDLE  = f'{RESULTS}/rollout_bundle_on_{CV00_DS.replace("flyvis_", "")}.npz'
TOP_LOG     = f'{LOG_DIR}/results_rollout_on_{CV00_DS.replace("flyvis_", "")}.log'
BOT_BUNDLE  = f'{RESULTS}/rollout_bundle_on_{NF_TWIN.replace("flyvis_", "")}.npz'
BOT_LOG     = f'{LOG_DIR}/results_rollout_on_{NF_TWIN.replace("flyvis_", "")}.log'


# ── trace window / selection / style (matches fig_davis_youtube_rollout_noise_free.py) ──
TRACE_START    = 500
TRACE_END      = 1500
SELECTED_TYPES = [23, 5, 6, 7, 12, 22, 43, 55, 35, 39, 31, 0]

COLOR_GT   = '#66cc66'
COLOR_PRED = 'black'
COLOR_RES  = '#cf222e'
LW_GT, LW_PRED, RES_LW = 1.8, 0.7, 1.2
DT_MS = 20.0

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'text.usetex':     False,
    'mathtext.fontset': 'dejavusans',
})
_S = 0.52
FS_LABEL  = int(48 * _S)
FS_TICK   = int(24 * _S)
FS_ANNOT  = int(32 * _S)
FS_LEGEND = int(28 * _S)
FS_TYPE   = int(26 * _S)
PANEL_LBL = 20


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


def _clone_base(out_yaml, dataset_name, description, overrides):
    """Clone the DAVIS base condition YAML, keep DAVIS stimuli, override
    noise / seeds / dataset name. Used for the cv00 noisy + cv00_nf test
    twins (both DAVIS-side)."""
    with open(BASE_YAML) as f:
        cfg = yaml.safe_load(f)
    cfg['description'] = description
    cfg['dataset']     = dataset_name
    sim = cfg['simulation']
    sim['seed']        = 42           # matches run_GNN_conditions.py's cv00 seed
    sim.update(overrides)
    with open(out_yaml, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Top panel — noisy cv00 (σ=0.05 retained on test via noisy_test_data=True)
# ---------------------------------------------------------------------------
def ensure_cv00_config():
    if os.path.isfile(CV00_YAML):
        print(f'[top 1/3] cv00 config exists: {CV00_YAML}')
        return
    print(f'[top 1/3] cloning {BASE_YAML} -> {CV00_YAML}')
    _clone_base(
        CV00_YAML, CV00_DS,
        description=(
            'DAVIS noisy cv00 test variant of flyvis_noise_005 — DAVIS stimulus + '
            'seed 42, noise_model_level=0.05, noisy_test_data=true (so x_list_test '
            'keeps σ=0.05 process noise). Used as test data for the YT-trained '
            f'{MODEL_NAME} model (figures/fig_davis_youtube_rollout_noise_005.py).'
        ),
        overrides={'noise_model_level': 0.05, 'noisy_test_data': True},
    )


def ensure_cv00_data():
    marker = f'{CV00_DATA}/noisy_test_data.ok'
    if os.path.isfile(marker):
        print(f'[top 2/3] noisy cv00 dataset exists: {CV00_DATA}')
        return
    # Wipe any stale or noise-free partial dataset to avoid the generator's
    # x_list_train existence check short-circuiting the regeneration.
    if os.path.isdir(CV00_DATA):
        print(f'[top 2/3] removing stale cv00 dataset at {CV00_DATA}')
        shutil.rmtree(CV00_DATA)
    print(f'[top 2/3] generating noisy cv00 DAVIS dataset — tens of minutes')
    _run('-o', 'generate', CV00_DS, tag='[top 2/3]')
    if not os.path.isfile(marker):
        sys.exit(f'expected marker missing after generation: {marker}')


def ensure_cv00_rollout():
    dataset_params = f'{CV00_DATA}/ode_params.pt'
    if (os.path.isfile(TOP_BUNDLE) and os.path.isfile(dataset_params)
            and os.path.getmtime(TOP_BUNDLE) > os.path.getmtime(dataset_params)):
        print(f'[top 3/3] cv00 rollout up to date: {TOP_BUNDLE}')
        return
    if os.path.isfile(TOP_BUNDLE):
        print(f'[top 3/3] stale cv00 rollout — removing {TOP_BUNDLE}')
        os.remove(TOP_BUNDLE)
    print(f'[top 3/3] running {MODEL_NAME} rollout on {CV00_DS}')
    _run('-o', 'test', MODEL_NAME, 'best', CV00_DS, tag='[top 3/3]')


# ---------------------------------------------------------------------------
# Bottom panel — noise-free twin (noise_model_level=0)
# ---------------------------------------------------------------------------
def ensure_twin_config():
    if os.path.isfile(NF_YAML):
        print(f'[bot 1/3] twin config exists: {NF_YAML}')
        return
    print(f'[bot 1/3] cloning {BASE_YAML} -> {NF_YAML}')
    _clone_base(
        NF_YAML, NF_TWIN,
        description=(
            f'DAVIS noise-free twin of flyvis_noise_005 cv00 — same DAVIS videos + '
            f'seed 42, noise_model_level=0. Used as test data for the YT-trained '
            f'{MODEL_NAME} model (see figures/fig_davis_youtube_rollout_noise_005.py).'
        ),
        overrides={'noise_model_level': 0.0},
    )


def ensure_twin_data():
    if os.path.isfile(f'{TWIN_DATA}/ode_params.pt'):
        print(f'[bot 2/3] twin dataset exists: {TWIN_DATA}')
        return
    print(f'[bot 2/3] generating noise-free DAVIS twin dataset — tens of minutes')
    _run('-o', 'generate', NF_TWIN, tag='[bot 2/3]')


def ensure_twin_rollout():
    dataset_params = f'{TWIN_DATA}/ode_params.pt'
    if (os.path.isfile(BOT_BUNDLE) and os.path.isfile(dataset_params)
            and os.path.getmtime(BOT_BUNDLE) > os.path.getmtime(dataset_params)):
        print(f'[bot 3/3] twin rollout up to date: {BOT_BUNDLE}')
        return
    if os.path.isfile(BOT_BUNDLE):
        print(f'[bot 3/3] stale twin rollout — removing {BOT_BUNDLE}')
        os.remove(BOT_BUNDLE)
    print(f'[bot 3/3] running {MODEL_NAME} rollout on {NF_TWIN}')
    _run('-o', 'test', MODEL_NAME, 'best', NF_TWIN, tag='[bot 3/3]')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_pearson(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            if line.strip().startswith('Pearson r'):
                return float(line.split(':')[1].split('+/-')[0].strip())
    return None


def _slice(arr, neuron_idx):
    return np.asarray(arr[neuron_idx, TRACE_START:TRACE_END], dtype=np.float32)


# ---------------------------------------------------------------------------
# Plot one (traces, residual) row
# ---------------------------------------------------------------------------
def draw_row(fig, gs, row, true_w, pred_w, labels, step_v, time_ms,
             pearson_r, show_xlabel):
    ax     = fig.add_subplot(gs[row, 0])
    ax_res = fig.add_subplot(gs[row, 1], sharey=ax)
    n_traces, n_frames = true_w.shape
    baselines = true_w.mean(axis=1)

    for i in range(n_traces):
        bl = baselines[i]
        ax.plot(time_ms, true_w[i] - bl + i * step_v,
                lw=LW_GT, color=COLOR_GT, alpha=0.95)
        ax.plot(time_ms, pred_w[i] - bl + i * step_v,
                lw=LW_PRED, color=COLOR_PRED, alpha=0.95)

    for i, lbl in enumerate(labels):
        ax.text(time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.025, i * step_v,
                lbl, fontsize=FS_TYPE, va='bottom', ha='right', color='black')

    r_txt = f'{pearson_r:.2f}' if pearson_r is not None else 'n/a'
    ax.text(0.05, 1.00,
            f'rollout Pearson $r$ = {r_txt} (8 000 test frames)',
            transform=ax.transAxes, va='top', ha='left', fontsize=FS_ANNOT)

    ax.set_ylim([-step_v, (n_traces - 1) * step_v + 1.3 * step_v])
    ax.set_yticks([])
    ax.set_xlim([time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.07, time_ms[-1]])
    if show_xlabel:
        ax.set_xlabel('time (ms)', fontsize=FS_LABEL)
        ax_res.set_xlabel('time (ms)', fontsize=FS_LABEL)
    ax.tick_params(axis='x', labelsize=FS_TICK)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)

    res_n = min(int(5000.0 / DT_MS), n_frames)
    res_time = time_ms[:res_n]
    residual = pred_w[:, :res_n] - true_w[:, :res_n]
    for i in range(n_traces):
        ax_res.plot(res_time, residual[i] + i * step_v,
                    lw=RES_LW, color=COLOR_RES, alpha=0.95)
        ax_res.axhline(i * step_v, lw=0.3, color='black', alpha=0.3)
    ax_res.set_xlim([res_time[0], res_time[-1]])
    ax_res.set_yticks([])
    ax_res.tick_params(axis='x', labelsize=FS_TICK)
    for sp in ('top', 'right', 'left'):
        ax_res.spines[sp].set_visible(False)

    return ax, ax_res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Top panel — noisy cv00 (regenerates if flag was off previously)
    ensure_cv00_config()
    ensure_cv00_data()
    ensure_cv00_rollout()
    # Bottom panel — noise-free twin
    ensure_twin_config()
    ensure_twin_data()
    ensure_twin_rollout()

    # Load — top panel comes from the existing rollout bundle (noisy σ=0.05)
    if not os.path.isfile(TOP_BUNDLE):
        sys.exit(f'missing top-panel bundle: {TOP_BUNDLE}')
    bundle   = np.load(TOP_BUNDLE, allow_pickle=True)
    true_top = bundle['activity_true']
    pred_top = bundle['activity_pred']
    type_ids   = bundle['type_ids'].astype(int)
    type_names = list(bundle['type_names'])
    index_to_name = {i: type_names[i] for i in range(len(type_names))}
    print(f'loaded top panel: true={true_top.shape} pred={pred_top.shape}')

    # Load — bottom panel from the fresh noise-free rollout
    if not os.path.isfile(BOT_BUNDLE):
        sys.exit(f'missing bot-panel bundle: {BOT_BUNDLE}')
    bundle_bot = np.load(BOT_BUNDLE, allow_pickle=True)
    true_bot = bundle_bot['activity_true']
    pred_bot = bundle_bot['activity_pred']
    print(f'loaded bot panel: true={true_bot.shape} pred={pred_bot.shape}')

    # Pick neurons (one per selected type, using top-panel type_ids)
    neuron_idx, labels = [], []
    for t in SELECTED_TYPES:
        ids = np.where(type_ids == t)[0]
        if len(ids) > 0:
            neuron_idx.append(int(ids[0]))
            labels.append(index_to_name.get(t, f'Type{t}'))

    true_top_w = _slice(true_top, neuron_idx)
    pred_top_w = _slice(pred_top, neuron_idx)
    true_bot_w = _slice(true_bot, neuron_idx)
    pred_bot_w = _slice(pred_bot, neuron_idx)
    n_traces, n_frames = true_top_w.shape
    time_ms = np.arange(n_frames) * DT_MS + TRACE_START * DT_MS

    # Shared vertical step — anchored on the noisier panel so both align
    step_v = max(0.5, 3.0 * float(np.std(true_top_w)))

    r_top = _parse_pearson(TOP_LOG)
    r_bot = _parse_pearson(BOT_LOG)
    print(f'  top panel r = {r_top}')
    print(f'  bot panel r = {r_bot}')

    fig_h = max(6.0, n_traces * 0.5 + 2.0)
    fig = plt.figure(figsize=(18, 2 * fig_h), dpi=300, constrained_layout=True)
    gs = mgs.GridSpec(2, 2, figure=fig, width_ratios=[4, 1],
                      wspace=0.02, hspace=0.12)

    ax_a, ax_a_r = draw_row(fig, gs, 0, true_top_w, pred_top_w, labels, step_v, time_ms,
                            pearson_r=r_top, show_xlabel=False)
    ax_b, ax_b_r = draw_row(fig, gs, 1, true_bot_w, pred_bot_w, labels, step_v, time_ms,
                            pearson_r=r_bot, show_xlabel=True)

    handles = [
        Line2D([0], [0], color=COLOR_GT,   lw=LW_GT,   label='ground truth'),
        Line2D([0], [0], color=COLOR_PRED, lw=LW_PRED, label='GNN rollout prediction'),
        Line2D([0], [0], color=COLOR_RES,  lw=RES_LW,  label='residual (pred $-$ true)'),
    ]
    # Per-row legend anchored to the right of the residual column.
    for _ax_res in (ax_a_r, ax_b_r):
        _ax_res.legend(handles=handles, loc='upper left',
                       bbox_to_anchor=(1.04, 1.0), ncol=1, handlelength=2,
                       fontsize=int(1.3 * FS_LEGEND), frameon=False,
                       borderaxespad=0.0)

    # ── panel labels a) / b) — top-left of each outer panel box ──────────────
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax_panel, lbl in zip([ax_a, ax_b], ['a)', 'b)']):
        bb = ax_panel.get_tightbbox(renderer)
        x0, y1 = inv.transform((bb.x0, bb.y1))
        fig.text(x0, y1, lbl, fontsize=PANEL_LBL, fontweight='bold',
                 va='bottom', ha='left', color='black', transform=fig.transFigure)

    out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'fig_davis_youtube_rollout_noise_005')
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    fig.savefig(out_base + '.jpg', dpi=300, bbox_inches='tight',
                pil_kwargs={'quality': 95})
    plt.close(fig)
    print(f'Saved: {out_base}.png')
    print(f'Saved: {out_base}.pdf')
    print(f'Saved: {out_base}.jpg')


if __name__ == '__main__':
    main()
