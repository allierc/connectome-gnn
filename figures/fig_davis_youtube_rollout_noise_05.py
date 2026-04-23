"""Figure: YT-trained GNN rollout on DAVIS (cv00) — noisy vs noise-free (σ = 0.5).

Janne-styled per figures/INSTRUCTIONS.md (the previous, larger-font version
is preserved at fig_davis_youtube_rollout_noise_05_original.py):

  • ~18 cm document-width figure (7.09 in) at 300 dpi
  • 6–8 pt fonts, 0.5 pt spines / ticks
  • top + right spines hidden globally (via janne.matplotlibrc)
  • trim_axis breaks each axis at the data range (upper & right gap)
  • PDF primary output (pdf.fonttype=42, svg.fonttype='none')

Two stacked panels, 12 representative cell types, 1 000 frames each:
  a) YT-trained GNN rollout vs DAVIS test with σ=0.5 process noise
  b) same model vs noise-free twin of the same DAVIS stimulus

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/fig_davis_youtube_rollout_noise_05.py

Output
------
    figures/fig_davis_youtube_rollout_noise_05.{pdf,png}
"""

import os
import shutil
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


# Try the flyvis trim_axis; fall back to a local equivalent if unavailable.
try:
    from flyvis.analysis.visualization.plt_utils import trim_axis as _trim_axis
except Exception:
    def _trim_axis(ax, xmargin=0.0, ymargin=0.0, yaxis=True, xaxis=True):
        """Local fallback: clip left/bottom spines to the data range so the
        axes break at the first/last data point (no spine beyond the data)."""
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


# ── identifiers (same as the original figure) ────────────────────────────────
REPO_ROOT   = '/workspace/connectome-gnn'
DATA_ROOT   = '/groups/saalfeld/home/allierc/GraphData'

BASE        = 'flyvis_noise_05'
MODEL_NAME  = f'{BASE}_yt_per_cond_cv00'
CV00_DS     = f'{BASE}_cv00'
NF_TWIN     = f'{BASE}_cv00_nf'

CFG_DIR     = f'{DATA_ROOT}/config/fly'
BASE_YAML   = f'{REPO_ROOT}/config/fly/{BASE}.yaml'
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


# ── trace window + selection (unchanged) ─────────────────────────────────────
TRACE_START    = 500
TRACE_END      = 1500
SELECTED_TYPES = [23, 5, 6, 7, 12, 22, 43, 55, 35, 39, 31, 0]

# Janne-style colours tuned for 6 pt legibility; top/right spines off globally.
COLOR_GT   = '#66cc66'
COLOR_PRED = 'black'
COLOR_RES  = '#cf222e'
LW_GT, LW_PRED, RES_LW = 0.9, 0.45, 0.6   # thin traces match 0.5 pt axes
DT_MS = 20.0

# Fonts (janne.matplotlibrc sets defaults to 8/6 pt; keep these as explicit
# override points so panel-specific tweaks are one-line edits).
FS_LABEL  = 8
FS_TICK   = 6
FS_ANNOT  = 6
FS_LEGEND = 6
FS_TYPE   = 6
PANEL_LBL = 8

# ~18 cm × double-panel height, both in inches (1 cm = 0.3937 in).
FIG_W_IN  = 18.0 * 0.3937       # ≈ 7.09 in
FIG_H_IN  = 7.0                  # two stacked panels of 12 traces each


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
    with open(BASE_YAML) as f:
        cfg = yaml.safe_load(f)
    cfg['description'] = description
    cfg['dataset']     = dataset_name
    sim = cfg['simulation']
    sim['seed']        = 42
    sim.update(overrides)
    with open(out_yaml, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Data-ensure helpers (identical to fig_davis_youtube_rollout_noise_05.py)
# ---------------------------------------------------------------------------
def ensure_cv00_config():
    if os.path.isfile(CV00_YAML):
        print(f'[top 1/3] cv00 config exists: {CV00_YAML}')
        return
    print(f'[top 1/3] cloning {BASE_YAML} -> {CV00_YAML}')
    _clone_base(
        CV00_YAML, CV00_DS,
        description=(
            'DAVIS noisy cv00 test variant of flyvis_noise_05 — DAVIS stimulus + '
            'seed 42, noise_model_level=0.5, noisy_test_data=true. Used as test '
            f'data for YT-trained {MODEL_NAME} '
            '(figures/fig_davis_youtube_rollout_noise_05.py).'
        ),
        overrides={'noise_model_level': 0.5, 'noisy_test_data': True},
    )


def ensure_cv00_data():
    marker = f'{CV00_DATA}/noisy_test_data.ok'
    if os.path.isfile(marker):
        print(f'[top 2/3] noisy cv00 dataset exists: {CV00_DATA}')
        return
    if os.path.isdir(CV00_DATA):
        print(f'[top 2/3] removing stale cv00 dataset at {CV00_DATA}')
        shutil.rmtree(CV00_DATA)
    print(f'[top 2/3] generating noisy cv00 DAVIS dataset — tens of minutes')
    # Pass the absolute YAML path so GNN_Main reads our cloned config
    # (with noisy_test_data: true) rather than the base flyvis_noise_05.yaml
    # via its _cvNN fallback in config_path() which only looks under <repo>/config/.
    _run('-o', 'generate', CV00_YAML, tag='[top 2/3]')
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
    _run('-o', 'test', MODEL_NAME, 'best', CV00_YAML, tag='[top 3/3]')


def ensure_twin_config():
    if os.path.isfile(NF_YAML):
        print(f'[bot 1/3] twin config exists: {NF_YAML}')
        return
    print(f'[bot 1/3] cloning {BASE_YAML} -> {NF_YAML}')
    _clone_base(
        NF_YAML, NF_TWIN,
        description=(
            f'DAVIS noise-free twin of flyvis_noise_05 cv00 — same DAVIS videos + '
            f'seed 42, noise_model_level=0. Used as test data for YT-trained '
            f'{MODEL_NAME} (figures/fig_davis_youtube_rollout_noise_05.py).'
        ),
        overrides={'noise_model_level': 0.0},
    )


def ensure_twin_data():
    if os.path.isfile(f'{TWIN_DATA}/ode_params.pt'):
        print(f'[bot 2/3] twin dataset exists: {TWIN_DATA}')
        return
    print(f'[bot 2/3] generating noise-free DAVIS twin dataset — tens of minutes')
    _run('-o', 'generate', NF_YAML, tag='[bot 2/3]')


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
    _run('-o', 'test', MODEL_NAME, 'best', NF_YAML, tag='[bot 3/3]')


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


def _slice(arr, neuron_idx):
    return np.asarray(arr[neuron_idx, TRACE_START:TRACE_END], dtype=np.float32)


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


def draw_row(fig, gs, row, true_w, pred_w, labels, step_v, time_ms,
             pearson_r, show_xlabel, sigma_title=None):
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
        ax.text(time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.02, i * step_v,
                lbl, fontsize=FS_TYPE, va='bottom', ha='right', color='black')

    r_txt = f'{pearson_r:.2f}' if pearson_r is not None else 'n/a'
    # Two lines in the top-left: σ-regime label (if provided) above the rollout r.
    if sigma_title is not None:
        ax.text(0.01, 1.10, sigma_title, transform=ax.transAxes,
                va='bottom', ha='left', fontsize=FS_LABEL, fontweight='bold')
    ax.text(0.01, 1.02,
            f'rollout Pearson $r$ = {r_txt} (8 000 test frames)',
            transform=ax.transAxes, va='bottom', ha='left', fontsize=FS_ANNOT)

    ax.set_ylim([-step_v, (n_traces - 1) * step_v + 1.3 * step_v])
    ax.set_yticks([])
    _pretty_xticks(ax, time_ms[0], time_ms[-1])
    if show_xlabel:
        ax.set_xlabel('time (ms)', fontsize=FS_LABEL)
        ax_res.set_xlabel('time (ms)', fontsize=FS_LABEL)
    ax.tick_params(axis='x', labelsize=FS_TICK)
    # Left spine hidden because traces carry no quantitative y-axis here.
    ax.spines['left'].set_visible(False)
    _trim_axis(ax, yaxis=False)

    res_n = min(int(5000.0 / DT_MS), n_frames)
    res_time = time_ms[:res_n]
    residual = pred_w[:, :res_n] - true_w[:, :res_n]
    for i in range(n_traces):
        ax_res.plot(res_time, residual[i] + i * step_v,
                    lw=RES_LW, color=COLOR_RES, alpha=0.95)
        ax_res.axhline(i * step_v, lw=0.25, color='black', alpha=0.3)
    _pretty_xticks(ax_res, res_time[0], res_time[-1])
    ax_res.set_yticks([])
    ax_res.tick_params(axis='x', labelsize=FS_TICK)
    ax_res.spines['left'].set_visible(False)
    _trim_axis(ax_res, yaxis=False)

    return ax, ax_res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_cv00_config()
    ensure_cv00_data()
    ensure_cv00_rollout()
    ensure_twin_config()
    ensure_twin_data()
    ensure_twin_rollout()

    if not os.path.isfile(TOP_BUNDLE):
        sys.exit(f'missing top-panel bundle: {TOP_BUNDLE}')
    bundle   = np.load(TOP_BUNDLE, allow_pickle=True)
    true_top = bundle['activity_true']
    pred_top = bundle['activity_pred']
    type_ids   = bundle['type_ids'].astype(int)
    type_names = list(bundle['type_names'])
    index_to_name = {i: type_names[i] for i in range(len(type_names))}
    print(f'loaded top panel: true={true_top.shape} pred={pred_top.shape}')

    if not os.path.isfile(BOT_BUNDLE):
        sys.exit(f'missing bot-panel bundle: {BOT_BUNDLE}')
    bundle_bot = np.load(BOT_BUNDLE, allow_pickle=True)
    true_bot = bundle_bot['activity_true']
    pred_bot = bundle_bot['activity_pred']
    print(f'loaded bot panel: true={true_bot.shape} pred={pred_bot.shape}')

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

    step_v = max(0.5, 3.0 * float(np.std(true_top_w)))

    r_top = _parse_pearson(TOP_LOG)
    r_bot = _parse_pearson(BOT_LOG)
    print(f'  top panel r = {r_top}')
    print(f'  bot panel r = {r_bot}')

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), constrained_layout=True)
    gs = mgs.GridSpec(2, 2, figure=fig, width_ratios=[4, 1],
                      wspace=0.04, hspace=0.20)

    ax_a, ax_a_r = draw_row(fig, gs, 0, true_top_w, pred_top_w, labels,
                            step_v, time_ms, pearson_r=r_top, show_xlabel=False,
                            sigma_title=r'high intrinsic noise ($\sigma = 0.5$)')
    ax_b, ax_b_r = draw_row(fig, gs, 1, true_bot_w, pred_bot_w, labels,
                            step_v, time_ms, pearson_r=r_bot, show_xlabel=True)

    handles = [
        Line2D([0], [0], color=COLOR_GT,   lw=LW_GT,   label='ground truth'),
        Line2D([0], [0], color=COLOR_PRED, lw=LW_PRED, label='GNN rollout prediction'),
        Line2D([0], [0], color=COLOR_RES,  lw=RES_LW,  label='residual (pred $-$ true)'),
    ]
    # Legend inside the top panel's trace ax, top-right (doesn't steal width).
    ax_a.legend(handles=handles, loc='upper right', ncol=1, handlelength=1.5,
                fontsize=FS_LEGEND, frameon=False, borderaxespad=0.3)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax_panel, lbl in zip([ax_a, ax_b], ['a', 'b']):
        bb = ax_panel.get_tightbbox(renderer)
        x0, y1 = inv.transform((bb.x0, bb.y1))
        fig.text(x0, y1, lbl, fontsize=PANEL_LBL, fontweight='bold',
                 va='bottom', ha='left', color='black', transform=fig.transFigure)

    out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'fig_davis_youtube_rollout_noise_05')
    # PDF first per janne.matplotlibrc default; PNG for quick preview.
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_base}.pdf')
    print(f'Saved: {out_base}.png')


if __name__ == '__main__':
    main()
