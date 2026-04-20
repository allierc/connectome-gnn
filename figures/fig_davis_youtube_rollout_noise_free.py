"""
Figure: YT-trained GNN rollout on held-out DAVIS sequences.

Model trained on YouTube-VOS stimuli (flyvis_noise_free_yt_per_cond_cv00,
the 434k-edge noise-free simulation with 50%-blank training frames),
evaluated zero-shot on DAVIS held-out fold cv00.

Layout
------
  Single panel — stacked voltage traces for 12 representative cell types:
    green (thick) : ground truth
    black (thin)  : GNN prediction

Data source
-----------
  /log/fly/flyvis_noise_free_yt_per_cond_cv00/results/
      rollout_bundle_on_noise_free_cv00.npz
        activity_true : (13741, 7999)
        activity_pred : (13741, 7999)
        type_ids, type_names

Usage
-----
    conda run -n neural-graph-linux python figures/fig_davis_youtube_rollout_noise_free.py

Output
------
    figures/fig_davis_youtube_rollout_noise_free.{png,pdf,jpg}
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── font style (INSTRUCTIONS.md §style) ──────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'text.usetex': False,
    'mathtext.fontset': 'dejavusans',
})

# ── font sizes (single wide panel — full GNN_PlotFigure scale) ───────────────
_S        = 0.52
FS_LABEL  = int(48 * _S)
FS_TICK   = int(24 * _S)
FS_ANNOT  = int(32 * _S)
FS_LEGEND = int(28 * _S)
FS_TYPE   = int(26 * _S)
FS_TITLE  = 22

# ── paths ────────────────────────────────────────────────────────────────────
# YT-trained model, DAVIS test (matches run_GNN_conditions.py convention).
MODEL_DIR   = '/groups/saalfeld/home/allierc/GraphData/log/fly/flyvis_noise_free_yt_per_cond_cv00'
RESULTS_DIR = f'{MODEL_DIR}/results'
TEST_SHORT  = 'noise_free_cv00'   # DAVIS CV fold 0 short-name

# ── trace window + selection (same curated list as fig_traces.py) ────────────
TRACE_START    = 500
TRACE_END      = 1500   # 1,000 frames = 20 s at dt=20 ms
SELECTED_TYPES = [
    23, 24, 26, 29, 30,       # photoreceptors: R1, R2, R4, R7, R8
    5, 6, 7, 8, 9,            # lamina: L1, L2, L3, L4, L5
    12, 21, 22,               # medulla intrinsic: Mi1, Mi4, Mi9
    43, 45, 49, 55,           # medulla transmedullary: Tm1, Tm2, Tm30, Tm9
    31, 32, 35, 39,           # T-cells: T1, T2, T4a, T5a
    0,                        # Am
]

# ── colors / line widths ─────────────────────────────────────────────────────
COLOR_GT   = '#66cc66'   # green (same as graph_tester.py)
COLOR_PRED = 'black'
LW_GT      = 1.8
LW_PRED    = 0.7
DT_MS      = 20.0        # flyvis delta_t


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(f'loading YT-trained model → DAVIS fold {TEST_SHORT} ...')
bundle_path = os.path.join(RESULTS_DIR, f'rollout_bundle_on_{TEST_SHORT}.npz')
if not os.path.isfile(bundle_path):
    sys.exit(f'missing rollout bundle: {bundle_path}\n'
             'Run run_GNN_conditions.py first (cross-test wave writes it).')
bundle   = np.load(bundle_path, allow_pickle=True)
true_arr = bundle['activity_true']
pred_arr = bundle['activity_pred']
type_ids   = bundle['type_ids'].astype(int)
type_names = list(bundle['type_names'])
index_to_name = {i: type_names[i] for i in range(len(type_names))}
print(f'  true: {true_arr.shape}  pred: {pred_arr.shape}  types={len(type_names)}')

# ── pick one neuron per selected type ────────────────────────────────────────
neuron_idx = []
labels     = []
for t in SELECTED_TYPES:
    ids = np.where(type_ids == t)[0]
    if len(ids) > 0:
        neuron_idx.append(int(ids[0]))
        labels.append(index_to_name.get(t, f'Type{t}'))

true_win = np.asarray(true_arr[neuron_idx, TRACE_START:TRACE_END], dtype=np.float32)
pred_win = np.asarray(pred_arr[neuron_idx, TRACE_START:TRACE_END], dtype=np.float32)
n_traces, n_frames = true_win.shape

# Global rollout r — take from the log so figure reports the exact number
rollout_log = f'{MODEL_DIR}/results_rollout_on_{TEST_SHORT}.log'
pearson_r = None
if os.path.isfile(rollout_log):
    with open(rollout_log) as f:
        for line in f:
            if line.strip().startswith('Pearson r'):
                pearson_r = float(line.split(':')[1].split('+/-')[0].strip())
                break


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
step_v = max(0.3, 1.6 * float(np.std(true_win)))
fig_h  = max(6.0, n_traces * 0.38 + 2.0)
fig = plt.figure(figsize=(18, fig_h), dpi=300, constrained_layout=True)
# main traces (left) + residual zoom (right) — 4:1 width with tight gap
import matplotlib.gridspec as _mgs
_gs = _mgs.GridSpec(1, 2, figure=fig, width_ratios=[4, 1], wspace=0.02)
ax     = fig.add_subplot(_gs[0, 0])
ax_res = fig.add_subplot(_gs[0, 1], sharey=ax)

baselines = true_win.mean(axis=1)
time_ms   = np.arange(n_frames) * DT_MS + TRACE_START * DT_MS

for i in range(n_traces):
    bl = baselines[i]
    ax.plot(time_ms, true_win[i] - bl + i * step_v,
            lw=LW_GT, color=COLOR_GT, alpha=0.95,
            label='ground truth' if i == 0 else None)

for i in range(n_traces):
    bl = baselines[i]
    ax.plot(time_ms, pred_win[i] - bl + i * step_v,
            lw=LW_PRED, color=COLOR_PRED, alpha=0.95,
            label='GNN rollout prediction' if i == 0 else None)

for i, lbl in enumerate(labels):
    ax.text(time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.025, i * step_v,
            lbl, fontsize=FS_TYPE, va='bottom', ha='right', color='black')

# ── vertical group labels on the far left ──────────────────────────────────
# Traces are rendered bottom-to-top in SELECTED_TYPES order. Groups follow
# the anatomical grouping in SELECTED_TYPES (5+5+3+4+4+1 = 22 traces).
_GROUPS = [
    (0,  4,  'photoreceptors'),
    (5,  9,  'lamina'),
    (10, 12, 'medulla\nintrinsic'),
    (13, 16, 'transmedullary'),
    (17, 20, 'T cells'),
    (21, 21, 'Am'),
]
_group_x = time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.10
for i0, i1, gname in _GROUPS:
    if i0 >= n_traces:
        continue
    i1 = min(i1, n_traces - 1)
    y_mid = ((i0 + i1) / 2.0) * step_v + step_v * 0.5
    ax.text(_group_x, y_mid, gname,
            fontsize=int(1.2 * FS_TYPE),
            rotation=90, va='center', ha='center', color='black')

# annotation — rollout r over the full 8,000-frame test
if pearson_r is not None:
    ax.text(0.05, 1.02,
            f'rollout Pearson $r$ = {pearson_r:.2f} (8 000 test frames)',
            transform=ax.transAxes, va='bottom', ha='left',
            fontsize=FS_ANNOT)

ax.set_ylim([-step_v, (n_traces - 1) * step_v + 1.3 * step_v])
ax.set_yticks([])
ax.set_xlim([time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.07,
             time_ms[-1]])
ax.set_xlabel('time (ms)', fontsize=FS_LABEL)
ax.tick_params(axis='x', labelsize=FS_TICK)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
from matplotlib.lines import Line2D
_legend_handles = [
    Line2D([0], [0], color=COLOR_GT,   lw=LW_GT,   label='ground truth'),
    Line2D([0], [0], color=COLOR_PRED, lw=LW_PRED, label='GNN rollout prediction'),
    Line2D([0], [0], color='#cf222e',  lw=1.2,     label='residual (pred $-$ true)'),
]
# Legend on the right of the residual column — attached after ax_res is
# fully populated below (see end of file).

# ── residual panel — first 5000 ms of the shown comparison window ────────────
COLOR_RES = '#cf222e'
RES_LW    = 1.2
res_n_frames = min(int(5000.0 / DT_MS), n_frames)   # 250 frames at 20 ms
res_time_ms  = time_ms[:res_n_frames]
residual     = pred_win[:, :res_n_frames] - true_win[:, :res_n_frames]

for i in range(n_traces):
    ax_res.plot(res_time_ms, residual[i] + i * step_v,
                lw=RES_LW, color=COLOR_RES, alpha=0.95)
    ax_res.axhline(i * step_v, lw=0.3, color='black', alpha=0.3)

ax_res.set_xlim([res_time_ms[0], res_time_ms[-1]])
ax_res.set_yticks([])
ax_res.set_xlabel('time (ms)', fontsize=FS_LABEL)
ax_res.tick_params(axis='x', labelsize=FS_TICK)
ax_res.spines['top'].set_visible(False)
ax_res.spines['right'].set_visible(False)
ax_res.spines['left'].set_visible(False)

# Legend on the right of the residual column (vertical stack — doesn't
# steal plot width).
ax_res.legend(handles=_legend_handles, loc='upper left',
              bbox_to_anchor=(1.04, 1.0), ncol=1, handlelength=2,
              fontsize=int(1.3 * FS_LEGEND), frameon=False,
              borderaxespad=0.0)
# ── save ─────────────────────────────────────────────────────────────────────
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
out_base = os.path.join(OUT_DIR, 'fig_davis_youtube_rollout_noise_free')
fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
fig.savefig(out_base + '.pdf', bbox_inches='tight')
fig.savefig(out_base + '.jpg', dpi=300, bbox_inches='tight',
            pil_kwargs={'quality': 95})
plt.close(fig)
print(f'Saved: {out_base}.png')
print(f'Saved: {out_base}.pdf')
print(f'Saved: {out_base}.jpg')
