"""
Figure: simulated neural activity at three intrinsic-noise levels.

Layout (2 rows × 3 columns)
---------------------------
  Row 1 (a–c): type-mean voltage heatmap — 65 cell types × 2 000 frames,
               z-scored per type (removes type-specific baseline/amplitude
               so all types are equally visible), anatomically sorted.
               Conveys the full dataset scale and type-specific dynamics.
  Row 2 (d–f): stacked voltage traces for 6 representative cell types
               (R1 · L1 · L2 · Mi1 · T4a · T5a) over a 500-frame window.
               Red dashed line = visual-input stimulus I_i(t).
               Conveys the noise effect on individual neuronal dynamics.

Columns left → right: σ = 0 (noise-free), σ = 0.05, σ = 0.5.
step_v for traces is fixed from σ = 0 data so noise is visually comparable.

Data source
-----------
x_list_train zarr (fields: voltage, stimulus, neuron_type) — the actual
training data, not the test rollout bundle.

Usage
-----
    conda run -n neural-graph-linux python figures/fig_simulations.py

Output
------
    figures/fig_simulations.{png,pdf,jpg}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Configs        : <DATA_ROOT>/config/fly/flyvis_noise_free.yaml
#                  <DATA_ROOT>/config/fly/flyvis_noise_005.yaml
#                  <DATA_ROOT>/config/fly/flyvis_noise_05.yaml
# Training data  : <DATA_ROOT>/graphs_data/fly/<dataset>/x_list_train/
#                  <DATA_ROOT>/graphs_data/fly/<dataset>/{edge_index.pt, ode_params.pt}
# Output         : figures/fig_simulations.{png,pdf,jpg}
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── project imports ───────────────────────────────────────────────────────────
REPO = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(REPO, 'src'))

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.metrics import ANATOMICAL_ORDER, INDEX_TO_NAME
from connectome_gnn.utils import set_data_root, graphs_data_path, add_pre_folder
from connectome_gnn.zarr_io import load_simulation_data

# ── font style (INSTRUCTIONS.md §style) ──────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'text.usetex': False,
    'mathtext.fontset': 'dejavusans',
})

# ── font sizes (col ≈ 7 in, _S = 0.52, INSTRUCTIONS.md §font) ────────────────
_S        = 0.52
FS_LABEL  = int(48 * _S)   # axis labels
FS_TICK   = int(24 * _S)   # tick labels
FS_ANNOT  = int(28 * _S)   # type-name annotations in trace panel
FS_TITLE  = 22             # panel subtitle
PANEL_LBL = 20             # a)–f)           (fixed, never scaled)
FS_CBAR   = int(48 * _S)   # colorbar label
FS_LEGEND = int(40 * _S)   # legend

# ── data ──────────────────────────────────────────────────────────────────────
DATA_ROOT = os.environ.get('TRAINED_MODEL_OUTPUT_ROOT', '.')
CONFIGS = [
    ('flyvis_noise_free', DATA_ROOT, 'noise-free'),
    ('flyvis_noise_005',  DATA_ROOT, 'low intrinsic noise'),
    ('flyvis_noise_05',   DATA_ROOT, 'high intrinsic noise'),
]
N_HEATMAP_FRAMES = 2000    # frames loaded for the heatmap (zarr subsample)

# ── selected types for trace panels ──────────────────────────────────────────
# Covers the full visual pathway: photoreceptors → lamina → medulla → T-cells
# R1=23, L1=5, L2=6, L3=7, Mi1=12, Mi9=22, Tm1=43, Tm9=55, T4a=35, T5a=39, T1=31, Am=0
SELECTED_TYPES = [23, 5, 6, 7, 12, 22, 43, 55, 35, 39, 31, 0]
TRACE_START    = 100
TRACE_END      = 600   # 500-frame window

# ── heatmap style ─────────────────────────────────────────────────────────────
CMAP  = 'RdBu_r'
VLIM  = 2.0          # ±2 σ clipping for z-scored heatmap

# ── trace colors — match graph_tester.py ─────────────────────────────────────
COLOR_GT   = 'black'     # black — voltage trace
COLOR_STIM = 'red'       # red    — stimulus
LW_GT      = 1.5
LW_STIM    = 0.8


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(config_name, data_root):
    """Load voltage, stimulus, and per-neuron type_ids from x_list_train zarr.

    Returns
    -------
    voltage  : (n_neurons, N_HEATMAP_FRAMES)  float32
    stimulus : (n_neurons, N_HEATMAP_FRAMES)  float32
    type_ids : (n_neurons,)                   int
    """
    set_data_root(data_root)
    cfg_path = os.path.join(REPO, 'config', 'fly', f'{config_name}.yaml')
    config   = NeuralGraphConfig.from_yaml(cfg_path)
    _, pre   = add_pre_folder(config_name)
    if not config.dataset.startswith(pre):
        config.dataset = pre + config.dataset

    gdata_dir = graphs_data_path(config.dataset)
    x_ts = load_simulation_data(
        os.path.join(gdata_dir, 'x_list_train'),
        fields=['voltage', 'stimulus', 'neuron_type'],
    )
    # subsample frames for speed (zarr lazy-loads only what is requested)
    voltage  = x_ts.voltage[:N_HEATMAP_FRAMES].numpy().T.astype(np.float32)
    stimulus = x_ts.stimulus[:N_HEATMAP_FRAMES].numpy().T.astype(np.float32)
    type_ids = x_ts.neuron_type.numpy().astype(int)   # (n_neurons,)
    return voltage, stimulus, type_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _type_heatmap(voltage, type_ids, n_types, anat_order):
    """Type-mean voltage, z-scored per type, anatomically sorted.

    z-scoring (subtract each type's mean, divide by its std) removes the
    type-specific voltage baseline and amplitude, making all 65 types
    equally visible in the heatmap regardless of absolute scale.

    Returns (n_sorted_types, n_frames) array and corresponding name list.
    """
    n_frames  = voltage.shape[1]
    type_mean = np.zeros((n_types, n_frames), dtype=np.float32)
    for t in range(n_types):
        mask = type_ids == t
        if mask.sum() > 0:
            type_mean[t] = voltage[mask].mean(axis=0)
    mu  = type_mean.mean(axis=1, keepdims=True)
    std = type_mean.std(axis=1,  keepdims=True)
    z   = (type_mean - mu) / (std + 1e-6)
    valid = [i for i in anat_order if i is not None and i < n_types]
    names = [INDEX_TO_NAME.get(i, f'Type{i}') for i in valid]
    return z[valid], names


# ---------------------------------------------------------------------------
# Load all data (one config at a time to keep peak memory low)
# ---------------------------------------------------------------------------
anat_order = ANATOMICAL_ORDER   # may contain None at index 0
n_types    = len(INDEX_TO_NAME) # 65

heatmaps   = []
trace_data = []   # list of (voltage, stimulus, type_ids) for trace window only
sorted_names_ref = None

for config_name, data_root, sigma_lbl in CONFIGS:
    print(f'loading {config_name} ...')
    voltage, stimulus, type_ids = load_training_data(config_name, data_root)

    # heatmap over all N_HEATMAP_FRAMES
    hz, snames = _type_heatmap(voltage, type_ids, n_types, anat_order)
    heatmaps.append(hz)
    if sorted_names_ref is None:
        sorted_names_ref = snames

    # keep only the trace window to save memory
    v_win   = voltage[:,  TRACE_START:TRACE_END]   # (n_neurons, 500)
    s_win   = stimulus[:, TRACE_START:TRACE_END]   # (n_neurons, 500)
    trace_data.append((v_win, s_win, type_ids))

    del voltage, stimulus   # free ~400 MB

# fixed step_v from σ=0 (noise-free) data so noise effect is visually comparable
v_free, _, tids_free = trace_data[0]
free_traces = np.stack([v_free[np.where(tids_free == t)[0][0]]
                        for t in SELECTED_TYPES
                        if len(np.where(tids_free == t)[0]) > 0])
step_v = max(0.5, 3.0 * float(np.std(free_traces)))


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    2, 3, figsize=(21, 14), dpi=300,
    gridspec_kw={'height_ratios': [3, 4]},
)
plt.subplots_adjust(left=0.08, right=0.88, top=0.96, bottom=0.06,
                    hspace=0.32, wspace=0.06)

n_sorted  = len(sorted_names_ref)
n_frames  = heatmaps[0].shape[1]
n_tframes = TRACE_END - TRACE_START

last_im = None
for col, (hz, (v_win, s_win, type_ids), (_, _, sigma_lbl)) in enumerate(
        zip(heatmaps, trace_data, CONFIGS)):

    # ── row 0: heatmap ────────────────────────────────────────────────────────
    ax_h = axes[0, col]
    im   = ax_h.imshow(hz, aspect='auto', interpolation='nearest',
                       cmap=CMAP, vmin=-VLIM, vmax=VLIM, origin='upper')
    last_im = im
    ax_h.set_title(sigma_lbl, fontsize=FS_TITLE, pad=4)
    ax_h.set_xticks([0, n_frames // 2, n_frames - 1])
    ax_h.set_xticklabels(['0', str(n_frames // 2), str(n_frames - 1)],
                          fontsize=FS_TICK)
    ax_h.set_xlabel('frame', fontsize=FS_LABEL)
    if col == 0:
        ax_h.set_yticks(range(n_sorted))
        ax_h.set_yticklabels(sorted_names_ref, fontsize=7)
        ax_h.set_ylabel('cell type', fontsize=FS_LABEL)
    else:
        ax_h.set_yticks([])

    # ── row 1: traces ─────────────────────────────────────────────────────────
    ax_t = axes[1, col]
    row  = 0
    first_gt   = True
    first_stim = True
    for t_idx in SELECTED_TYPES:
        neuron_rows = np.where(type_ids == t_idx)[0]
        if len(neuron_rows) == 0:
            continue
        nid   = neuron_rows[0]
        trace = v_win[nid]
        stim  = s_win[nid]
        bl    = trace.mean()

        # voltage trace (green)
        ax_t.plot(trace - bl + row * step_v,
                  lw=LW_GT, color=COLOR_GT, alpha=0.9)
        first_gt = False

        # stimulus trace (red dashed) — only when non-trivial
        if stim.mean() > 0:
            ax_t.plot(stim - stim.mean() + row * step_v,
                      lw=LW_STIM, color=COLOR_STIM, alpha=0.9,
                      linestyle='--')
            first_stim = False

        # type-name label on the left
        ax_t.text(-n_tframes * 0.025, row * step_v,
                  INDEX_TO_NAME.get(t_idx, f'Type{t_idx}'),
                  fontsize=FS_ANNOT, va='bottom', ha='right', color='black')
        row += 1

    n_rows = row
    ax_t.set_ylim([-step_v, (n_rows - 1) * step_v + step_v])
    ax_t.set_yticks([])
    ax_t.set_xticks([0, n_tframes // 2, n_tframes])
    ax_t.set_xticklabels([TRACE_START, (TRACE_START + TRACE_END) // 2, TRACE_END],
                          fontsize=FS_TICK)
    ax_t.set_xlabel('frame', fontsize=FS_LABEL)
    ax_t.set_xlim([-n_tframes * 0.08, n_tframes * 1.05])
    ax_t.set_title(sigma_lbl, fontsize=FS_TITLE, pad=4)
    ax_t.spines['top'].set_visible(False)
    ax_t.spines['right'].set_visible(False)
    ax_t.spines['left'].set_visible(False)
    if col == 0:
        ax_t.set_ylabel('voltage (a.u.)', fontsize=FS_LABEL, labelpad=28)


# ── shared colorbar — placed manually right next to panel c ──────────────────
# Draw once so subplots_adjust positions are committed.
fig.canvas.draw()
_pos = axes[0, 2].get_position()   # [x0, y0, w, h] in figure fraction
_cax = fig.add_axes([
    _pos.x1 + 0.008,               # just to the right of panel c
    _pos.y0 + _pos.height * 0.10,  # 10 % padding bottom
    0.012,                          # slim width
    _pos.height * 0.80,            # 80 % of panel height
])
cbar = fig.colorbar(last_im, cax=_cax)
cbar.set_label('z-scored voltage', fontsize=FS_CBAR)
cbar.ax.tick_params(labelsize=FS_CBAR)

# ── panel labels — row 0 aligned together, row 1 aligned together ─────────────
# (INSTRUCTIONS.md: use y1_max per row so labels within each row share the
#  same y; rows at different heights naturally get their own y level)
fig.canvas.draw()
renderer  = fig.canvas.get_renderer()
inv       = fig.transFigure.inverted()

bboxes_row0 = [ax.get_tightbbox(renderer) for ax in axes[0]]
bboxes_row1 = [ax.get_tightbbox(renderer) for ax in axes[1]]
y1_max_0 = max(inv.transform((bb.x0, bb.y1))[1] for bb in bboxes_row0)
y1_max_1 = max(inv.transform((bb.x0, bb.y1))[1] for bb in bboxes_row1)

for col, (lbl0, lbl1) in enumerate(zip(['a)', 'b)', 'c)'], ['d)', 'e)', 'f)'])):
    # x0 = leftmost edge of this column across both rows — aligns e.g. a) and d)
    x0_top = inv.transform((bboxes_row0[col].x0, bboxes_row0[col].y1))[0]
    x0_bot = inv.transform((bboxes_row1[col].x0, bboxes_row1[col].y1))[0]
    x0_col = min(x0_top, x0_bot)
    fig.text(x0_col, y1_max_0, lbl0, fontsize=PANEL_LBL, fontweight='bold',
             va='bottom', ha='left', color='black', transform=fig.transFigure)
    fig.text(x0_col, y1_max_1, lbl1, fontsize=PANEL_LBL, fontweight='bold',
             va='bottom', ha='left', color='black', transform=fig.transFigure)

# ── save ─────────────────────────────────────────────────────────────────────
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
out_base = os.path.join(OUT_DIR, 'fig_simulations')
plt.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
plt.savefig(out_base + '.pdf', bbox_inches='tight')
plt.savefig(out_base + '.jpg', dpi=300, bbox_inches='tight',
            pil_kwargs={'quality': 95})
plt.close()
print(f'Saved: {out_base}.png')
print(f'Saved: {out_base}.pdf')
print(f'Saved: {out_base}.jpg')
