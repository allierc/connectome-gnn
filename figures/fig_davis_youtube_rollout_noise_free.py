"""Figure: YT-trained GNN rollout on held-out DAVIS sequences (noise-free).

Janne-styled per figures/INSTRUCTIONS.md (the previous, larger-font version
is preserved at fig_davis_youtube_rollout_noise_free_original.py).

  • ~18 cm document-width figure (7.09 in) at 300 dpi
  • 6–8 pt fonts, 0.5 pt spines / ticks
  • top + right spines hidden globally (via janne.matplotlibrc)
  • trim_axis breaks each axis at the data range
  • PDF primary output (pdf.fonttype=42, svg.fonttype='none')

Model trained on YouTube-VOS stimuli
(flyvis_noise_free_yt_per_cond_cv00, the 434k-edge noise-free simulation),
evaluated zero-shot on DAVIS held-out fold cv00.

End-to-end pipeline (each step cached by file existence):
  1. Clone <base>.yaml -> cv00.yaml (DAVIS cv00, seed 42, noise-free)
  2. python GNN_Main.py -o generate <abs-yaml>            — regenerate ODE
  3. python GNN_Main.py -o test <model> best <abs-yaml>   — cross-dataset rollout
  4. Render the single panel.

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/fig_davis_youtube_rollout_noise_free.py

Output
------
    figures/fig_davis_youtube_rollout_noise_free.{pdf,png}
"""

import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'janne.matplotlibrc'))

import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D


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


# ── identifiers ──────────────────────────────────────────────────────────────
REPO_ROOT   = '/workspace/connectome-gnn'
DATA_ROOT   = '/groups/saalfeld/home/allierc/GraphData'

BASE        = 'flyvis_noise_free'
MODEL_NAME  = f'{BASE}_yt_per_cond_cv00'
CV00_DS     = f'{BASE}_cv00'

CFG_DIR     = f'{DATA_ROOT}/config/fly'
BASE_YAML   = f'{REPO_ROOT}/config/fly/{BASE}.yaml'
CV00_YAML   = f'{CFG_DIR}/{CV00_DS}.yaml'

LOG_DIR     = f'{DATA_ROOT}/log/fly/{MODEL_NAME}'
RESULTS     = f'{LOG_DIR}/results'
CV00_DATA   = f'{DATA_ROOT}/graphs_data/fly/{CV00_DS}'

TEST_SHORT  = CV00_DS.replace('flyvis_', '')
BUNDLE      = f'{RESULTS}/rollout_bundle_on_{TEST_SHORT}.npz'
LOG_FILE    = f'{LOG_DIR}/results_rollout_on_{TEST_SHORT}.log'


# ── trace window + selection (same curated list as the original) ─────────────
TRACE_START    = 500
TRACE_END      = 1500
SELECTED_TYPES = [
    23, 24, 26, 29, 30,       # photoreceptors: R1, R2, R4, R7, R8
    5, 6, 7, 8, 9,            # lamina: L1, L2, L3, L4, L5
    12, 21, 22,               # medulla intrinsic: Mi1, Mi4, Mi9
    43, 45, 49, 55,           # medulla transmedullary: Tm1, Tm2, Tm30, Tm9
    31, 32, 35, 39,           # T-cells: T1, T2, T4a, T5a
    0,                        # Am
]

# ── style (janne.matplotlibrc sets global defaults; locals for tweaks) ───────
COLOR_GT   = '#66cc66'
COLOR_PRED = 'black'
COLOR_RES  = '#cf222e'
LW_GT, LW_PRED, RES_LW = 0.9, 0.45, 0.6
DT_MS      = 20.0

FS_LABEL  = 8
FS_TICK   = 6
FS_ANNOT  = 6
FS_LEGEND = 6
FS_TYPE   = 6
FS_GROUP  = 7

FIG_W_IN  = 18.0 * 0.3937      # ≈ 7.09 in (18 cm)
FIG_H_IN  = 5.5                 # single panel, 22 traces


# ---------------------------------------------------------------------------
# Subprocess helper + data pipeline
# ---------------------------------------------------------------------------
def _run(*args, tag):
    print(f'{tag} python GNN_Main.py {" ".join(args)}')
    subprocess.check_call(
        ['python', f'{REPO_ROOT}/GNN_Main.py', *args,
         '--output_root', DATA_ROOT],
        cwd=REPO_ROOT,
    )


def _clone_base(out_yaml, dataset_name, description, overrides):
    with open(BASE_YAML) as f:
        cfg = yaml.safe_load(f)
    cfg['description'] = description
    cfg['dataset']     = dataset_name
    sim = cfg['simulation']
    sim['seed']        = 42
    sim.update(overrides)
    with open(out_yaml, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def ensure_cv00_config():
    if os.path.isfile(CV00_YAML):
        print(f'[1/3] cv00 config exists: {CV00_YAML}')
        return
    print(f'[1/3] cloning {BASE_YAML} -> {CV00_YAML}')
    _clone_base(
        CV00_YAML, CV00_DS,
        description=(
            'DAVIS noise-free cv00 test variant of flyvis_noise_free — DAVIS '
            'stimulus + seed 42, noise_model_level=0. Used as test data for '
            f'YT-trained {MODEL_NAME} '
            '(figures/fig_davis_youtube_rollout_noise_free.py).'
        ),
        overrides={'noise_model_level': 0.0},
    )


def ensure_cv00_data():
    if os.path.isfile(f'{CV00_DATA}/ode_params.pt'):
        print(f'[2/3] cv00 dataset exists: {CV00_DATA}')
        return
    print(f'[2/3] generating noise-free cv00 DAVIS dataset — tens of minutes')
    # Absolute yaml path — avoids GNN_Main's _cvNN fallback in config_path()
    # which would otherwise read the base flyvis_noise_free.yaml and miss
    # our cloned overrides.
    _run('-o', 'generate', CV00_YAML, tag='[2/3]')


def ensure_cv00_rollout():
    dataset_params = f'{CV00_DATA}/ode_params.pt'
    if (os.path.isfile(BUNDLE) and os.path.isfile(dataset_params)
            and os.path.getmtime(BUNDLE) > os.path.getmtime(dataset_params)):
        print(f'[3/3] cv00 rollout up to date: {BUNDLE}')
        return
    if os.path.isfile(BUNDLE):
        print(f'[3/3] stale cv00 rollout — removing {BUNDLE}')
        os.remove(BUNDLE)
    print(f'[3/3] running {MODEL_NAME} rollout on {CV00_DS}')
    _run('-o', 'test', MODEL_NAME, 'best', CV00_YAML, tag='[3/3]')


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _parse_pearson(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            if line.strip().startswith('Pearson r'):
                return float(line.split(':')[1].split('+/-')[0].strip())
    return None


def _pretty_xticks(ax, lo, hi, n_target=5):
    """Place x-ticks at lo and hi (pretty-snapped) with regular steps between."""
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
    ax.set_xticks(ticks)
    ax.set_xlim([lo, hi])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_cv00_config()
    ensure_cv00_data()
    ensure_cv00_rollout()

    if not os.path.isfile(BUNDLE):
        sys.exit(f'missing rollout bundle: {BUNDLE}')
    bundle   = np.load(BUNDLE, allow_pickle=True)
    true_arr = bundle['activity_true']
    pred_arr = bundle['activity_pred']
    type_ids   = bundle['type_ids'].astype(int)
    type_names = list(bundle['type_names'])
    index_to_name = {i: type_names[i] for i in range(len(type_names))}
    print(f'loaded bundle: true={true_arr.shape} pred={pred_arr.shape} types={len(type_names)}')

    neuron_idx, labels = [], []
    for t in SELECTED_TYPES:
        ids = np.where(type_ids == t)[0]
        if len(ids) > 0:
            neuron_idx.append(int(ids[0]))
            labels.append(index_to_name.get(t, f'Type{t}'))

    true_win = np.asarray(true_arr[neuron_idx, TRACE_START:TRACE_END], dtype=np.float32)
    pred_win = np.asarray(pred_arr[neuron_idx, TRACE_START:TRACE_END], dtype=np.float32)
    n_traces, n_frames = true_win.shape

    pearson_r = _parse_pearson(LOG_FILE)
    print(f'  rollout r = {pearson_r}')

    step_v = max(0.3, 1.6 * float(np.std(true_win)))
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), constrained_layout=True)
    gs = mgs.GridSpec(1, 2, figure=fig, width_ratios=[4, 1], wspace=0.04)
    ax     = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[0, 1], sharey=ax)

    baselines = true_win.mean(axis=1)
    time_ms   = np.arange(n_frames) * DT_MS + TRACE_START * DT_MS

    for i in range(n_traces):
        bl = baselines[i]
        ax.plot(time_ms, true_win[i] - bl + i * step_v,
                lw=LW_GT, color=COLOR_GT, alpha=0.95)
        ax.plot(time_ms, pred_win[i] - bl + i * step_v,
                lw=LW_PRED, color=COLOR_PRED, alpha=0.95)

    for i, lbl in enumerate(labels):
        ax.text(time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.02, i * step_v,
                lbl, fontsize=FS_TYPE, va='bottom', ha='right', color='black')

    # Vertical group labels on the far left
    _GROUPS = [
        (0,  4,  'photoreceptors'),
        (5,  9,  'lamina'),
        (10, 12, 'medulla\nintrinsic'),
        (13, 16, 'transmedullary'),
        (17, 20, 'T cells'),
        (21, 21, 'Am'),
    ]
    _group_x = time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.085
    for i0, i1, gname in _GROUPS:
        if i0 >= n_traces:
            continue
        i1 = min(i1, n_traces - 1)
        y_mid = ((i0 + i1) / 2.0) * step_v + step_v * 0.5
        ax.text(_group_x, y_mid, gname, fontsize=FS_GROUP,
                rotation=90, va='center', ha='center', color='black')

    # σ-regime label above the rollout Pearson annotation.
    ax.text(0.01, 1.10, r'noise-free ($\sigma = 0$)',
            transform=ax.transAxes, va='bottom', ha='left',
            fontsize=FS_LABEL, fontweight='bold')
    if pearson_r is not None:
        ax.text(0.01, 1.02,
                f'rollout Pearson $r$ = {pearson_r:.2f} (8 000 test frames)',
                transform=ax.transAxes, va='bottom', ha='left', fontsize=FS_ANNOT)

    ax.set_ylim([-step_v, (n_traces - 1) * step_v + 1.3 * step_v])
    ax.set_yticks([])
    _pretty_xticks(ax, time_ms[0], time_ms[-1])
    ax.set_xlabel('time (ms)', fontsize=FS_LABEL)
    ax.tick_params(axis='x', labelsize=FS_TICK)
    ax.spines['left'].set_visible(False)
    _trim_axis(ax, yaxis=False)

    # Residual — first 5 s
    res_n_frames = min(int(5000.0 / DT_MS), n_frames)
    res_time_ms  = time_ms[:res_n_frames]
    residual     = pred_win[:, :res_n_frames] - true_win[:, :res_n_frames]
    for i in range(n_traces):
        ax_res.plot(res_time_ms, residual[i] + i * step_v,
                    lw=RES_LW, color=COLOR_RES, alpha=0.95)
        ax_res.axhline(i * step_v, lw=0.25, color='black', alpha=0.3)
    _pretty_xticks(ax_res, res_time_ms[0], res_time_ms[-1])
    ax_res.set_yticks([])
    ax_res.set_xlabel('time (ms)', fontsize=FS_LABEL)
    ax_res.tick_params(axis='x', labelsize=FS_TICK)
    ax_res.spines['left'].set_visible(False)
    _trim_axis(ax_res, yaxis=False)

    handles = [
        Line2D([0], [0], color=COLOR_GT,   lw=LW_GT,   label='ground truth'),
        Line2D([0], [0], color=COLOR_PRED, lw=LW_PRED, label='GNN rollout prediction'),
        Line2D([0], [0], color=COLOR_RES,  lw=RES_LW,  label='residual (pred $-$ true)'),
    ]
    # Legend inside the main trace axes, top-right.
    ax.legend(handles=handles, loc='upper right', ncol=1, handlelength=1.5,
              fontsize=FS_LEGEND, frameon=False, borderaxespad=0.3)

    out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'fig_davis_youtube_rollout_noise_free')
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_base}.pdf')
    print(f'Saved: {out_base}.png')


if __name__ == '__main__':
    main()
