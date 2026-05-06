"""
Figure: simulated neural activity at three intrinsic-noise levels (rows 1-2)
plus a zoomed R1 trace at two measurement-noise levels (row 3).

Unified-style-styled per figures/INSTRUCTIONS.md:

  * ~18 cm document-width figure (7.09 in) at 300 dpi
  * 6-8 pt fonts, 0.5 pt spines / ticks
  * top + right spines hidden globally (via unified_style.matplotlibrc)
  * trim_axis breaks each axis at the data range
  * PDF primary output (pdf.fonttype=42, svg.fonttype='none')

Layout (3 rows)
---------------
  Row 1 (a-c): type-mean voltage heatmap, 65 cell types x 2 000 frames,
               z-scored per type, anatomically sorted. 3 intrinsic-noise
               levels (sigma_model = 0, 0.05, 0.5).
  Row 2 (d-f): stacked voltage traces for 12 representative cell types
               over a 500-frame window. Same 3 intrinsic-noise columns.
               step_v fixed from sigma_model = 0 so noise is visually
               comparable.
  Row 3 (g-h): single zoomed R1 photoreceptor trace at two measurement-
               noise levels (sigma_meas = 0, 0.10) -- both with 50% blank
               stimulus, so blank periods are highlighted as semi-
               transparent steelblue spans (matches fig_vrest_blank.py
               panel d). Shows how measurement noise corrupts a single
               trace while the underlying stimulus structure remains.

Data source
-----------
x_list_train zarr (fields: voltage, stimulus, neuron_type) - the actual
training data, not the test rollout bundle.

Usage
-----
    conda run -n neural-graph-linux python figures/fig_simulations.py

Output
------
    figures/fig_simulations.{pdf,png}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Configs        : <DATA_ROOT>/config/fly/flyvis_noise_free_blank50_davispt.yaml
#                  <DATA_ROOT>/config/fly/flyvis_noise_005_blank50_davispt.yaml
#                  <DATA_ROOT>/config/fly/flyvis_noise_05_blank50_davispt.yaml
# Training data  : <DATA_ROOT>/graphs_data/fly/<dataset>/x_list_train/
#                  <DATA_ROOT>/graphs_data/fly/<dataset>/{edge_index.pt, ode_params.pt}
# Output         : figures/fig_simulations.{pdf,png}
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'unified_style.matplotlibrc'))

import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
import numpy as np

# -- project imports ---------------------------------------------------------
REPO = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(REPO, 'src'))

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.metrics import ANATOMICAL_ORDER, INDEX_TO_NAME
from connectome_gnn.utils import set_data_root, graphs_data_path, add_pre_folder
from connectome_gnn.zarr_io import load_simulation_data


# Try the flyvis trim_axis; fall back to a local equivalent if unavailable.
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


# -- font sizes (unified_style.matplotlibrc sets 8/6 pt defaults) --------------------
FS_LABEL  = 8
FS_TICK   = 6
FS_ANNOT  = 5    # type-name annotations next to trace stacks
FS_TITLE  = 8
PANEL_LBL = 8
FS_CBAR   = 6
FS_TYPE   = 4    # heatmap y-axis cell-type labels (very dense; 65 entries)

# Minimum vertical spacing (in heatmap rows) between displayed cell-type
# names. With 65 anatomical types crammed into ~3 in of vertical space,
# every other label is suppressed so adjacent names stop overlapping.
HEATMAP_LABEL_MIN_STEP = 2

# -- figure size: ~24 cm wide (full landscape page width) --------------------
FIG_W_IN  = 24.0 * 0.3937   # ~9.45 in
FIG_H_IN  = 7.0             # 3 rows; row 3 is shorter

# -- data --------------------------------------------------------------------
DATA_ROOT = os.environ.get('TRAINED_MODEL_OUTPUT_ROOT', '.')

# Rows 1-2: 3 model-noise columns (heatmaps + trace stacks). Using the
# blank50 variants so 50% blank-stimulus periods are visible — the blue
# steelblue shading on d/e/f then encodes the same on/off structure as
# panels g/h/i.
# Use the *_davispt variants — three blank50 datasets that pin
# datavis_roots to the same DAVIS2017-partial-test path and share
# seed=42, so the DAVIS clip order (and therefore the blank-prefix
# positions) are identical across columns. See
# `python GNN_Main.py -o generate flyvis_blank50_davispt` to regenerate.
INTRINSIC_CONFIGS = [
    ('flyvis_noise_free_blank50_davispt', r'noise-free ($\sigma=0$)'),
    ('flyvis_noise_005_blank50_davispt',  r'low model noise ($\sigma=0.05$)'),
    ('flyvis_noise_05_blank50_davispt',   r'high model noise ($\sigma=0.5$)'),
]

# Row 3: noise-free blank50 source. Panel g is the raw signal (gamma=0,
# noise-free reference); panels h & i add Gaussian measurement noise so
# all three R1 panels share the exact same underlying trace. We pull
# from the same _davispt dataset as the heatmaps for stimulus
# consistency across the figure.
MEAS_SOURCE_CFG = 'flyvis_noise_free_blank50_davispt'
MEAS_GAMMAS = [
    (0.00, r'noise-free ($\sigma=0,\,\gamma=0$)'),
    (0.10, r'low measurement noise ($\gamma=0.1$)'),
    (0.20, r'high measurement noise ($\gamma=0.2$)'),
]
MEAS_NOISE_SEED = 42

N_HEATMAP_FRAMES = 2000

# -- selected types for trace stack panels -----------------------------------
# Covers the visual pathway: photoreceptors -> lamina -> medulla -> T-cells
SELECTED_TYPES = [23, 5, 6, 7, 12, 22, 43, 55, 35, 39, 31, 0]
TRACE_START    = 100
TRACE_END      = 600   # 500-frame window for trace stacks

# -- R1 zoom (row 3) ---------------------------------------------------------
R1_TYPE_ID     = 23    # R1 photoreceptor index
R1_TRACE_START = 100
R1_TRACE_END   = 600   # 500-frame window — same as row 2 trace stacks

# -- heatmap style -----------------------------------------------------------
CMAP  = 'RdBu_r'
VLIM  = 2.0

# -- trace + blank-shading style ---------------------------------------------
COLOR_GT        = 'black'
COLOR_STIM      = 'red'
LW_GT           = 0.6
LW_STIM         = 0.5
LW_R1           = 0.6
BLANK_COLOR     = 'steelblue'
BLANK_ALPHA     = 0.18
BLANK_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(config_name, data_root, n_frames):
    """Load voltage / stimulus / type_ids slice from x_list_train zarr."""
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
    voltage  = x_ts.voltage[:n_frames].numpy().T.astype(np.float32)
    stimulus = x_ts.stimulus[:n_frames].numpy().T.astype(np.float32)
    type_ids = x_ts.neuron_type.numpy().astype(int)
    return voltage, stimulus, type_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _type_heatmap(voltage, type_ids, n_types, anat_order):
    """Type-mean voltage, z-scored per type, anatomically sorted."""
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


def _thin_labels(names, min_step):
    """Replace each name within `min_step` rows of the previously kept name
    with an empty string. Used to declutter dense y-axis label stacks."""
    out  = []
    last = -10**9
    for i, n in enumerate(names):
        if i - last >= min_step:
            out.append(n)
            last = i
        else:
            out.append('')
    return out


def _draw_blank_shading(ax, s_win):
    """Steelblue spans where NO neuron sees stimulus (true blank periods).
    Uses max-across-neurons rather than mean: only ~8 of 13.7k neurons
    (photoreceptors) get direct visual input, so the population mean is
    always close to zero and would mark every frame as 'blank'."""
    s_max = np.abs(s_win).max(axis=0)
    blank = s_max < BLANK_THRESHOLD
    if not blank.any():
        return
    edges  = np.diff(blank.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0]
    for bs, be in zip(starts, ends):
        ax.axvspan(bs, be, alpha=BLANK_ALPHA, color=BLANK_COLOR,
                   linewidth=0, zorder=0)


# ---------------------------------------------------------------------------
# Load data — rows 1-2 (intrinsic-noise heatmaps + trace stacks)
# ---------------------------------------------------------------------------
anat_order = ANATOMICAL_ORDER
n_types    = len(INDEX_TO_NAME)   # 65

heatmaps          = []
trace_data        = []   # list of (v_window, s_window, type_ids)
sorted_names_ref  = None

for config_name, sigma_lbl in INTRINSIC_CONFIGS:
    print(f'loading {config_name} ...')
    voltage, stimulus, type_ids = load_training_data(
        config_name, DATA_ROOT, N_HEATMAP_FRAMES)

    hz, snames = _type_heatmap(voltage, type_ids, n_types, anat_order)
    heatmaps.append(hz)
    if sorted_names_ref is None:
        sorted_names_ref = snames

    v_win = voltage[:,  TRACE_START:TRACE_END]
    s_win = stimulus[:, TRACE_START:TRACE_END]
    trace_data.append((v_win, s_win, type_ids))

    del voltage, stimulus

# Fixed step_v from sigma_model = 0 so vertical spacing is identical across columns.
v_free, _, tids_free = trace_data[0]
free_traces = np.stack([
    v_free[np.where(tids_free == t)[0][0]]
    for t in SELECTED_TYPES if len(np.where(tids_free == t)[0]) > 0
])
step_v = max(0.5, 3.0 * float(np.std(free_traces)))


# ---------------------------------------------------------------------------
# Load data — row 3: one noise-free blank50 source, then add measurement
# noise per panel so both R1 traces share the same underlying signal.
# ---------------------------------------------------------------------------
print(f'loading {MEAS_SOURCE_CFG} (R1 zoom source) ...')
_v_src, _s_src, _tids_src = load_training_data(
    MEAS_SOURCE_CFG, DATA_ROOT, R1_TRACE_END)
_v_src_win = _v_src[:,  R1_TRACE_START:R1_TRACE_END]
_s_src_win = _s_src[:, R1_TRACE_START:R1_TRACE_END]
del _v_src, _s_src

# Synthesise per-gamma R1 windows by adding independent Gaussian noise.
_rng = np.random.default_rng(MEAS_NOISE_SEED)
r1_data = []
for gamma, _title in MEAS_GAMMAS:
    v_noisy = _v_src_win + _rng.normal(0.0, gamma, size=_v_src_win.shape).astype(np.float32)
    r1_data.append((v_noisy, _s_src_win, _tids_src))


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
# 3 rows x 6 cols — rows 1-2 use 3 panels of width 2; row 3 uses 2 panels of width 3.
# Heights: heatmap (3) > traces (4) > R1 zoom (2).
gs_outer = mgs.GridSpec(
    3, 6, figure=fig,
    height_ratios=[3, 4, 2],
    left=0.08, right=0.88, top=0.95, bottom=0.06,
    hspace=0.36, wspace=0.30,
)

axes_row0 = [fig.add_subplot(gs_outer[0, c*2:(c+1)*2]) for c in range(3)]   # a, b, c
axes_row1 = [fig.add_subplot(gs_outer[1, c*2:(c+1)*2]) for c in range(3)]   # d, e, f
axes_row2 = [fig.add_subplot(gs_outer[2, c*2:(c+1)*2]) for c in range(3)]   # g, h, i

n_sorted  = len(sorted_names_ref)
n_frames  = heatmaps[0].shape[1]
n_tframes = TRACE_END - TRACE_START
n_r1f     = R1_TRACE_END - R1_TRACE_START


# -- Row 1: heatmaps (a, b, c) -----------------------------------------------
last_im = None
for col, (hz, (_cfg, sigma_lbl)) in enumerate(zip(heatmaps, INTRINSIC_CONFIGS)):
    ax = axes_row0[col]
    im = ax.imshow(hz, aspect='auto', interpolation='nearest',
                   cmap=CMAP, vmin=-VLIM, vmax=VLIM, origin='upper')
    last_im = im
    ax.set_title(sigma_lbl, fontsize=FS_TITLE, pad=4)
    ax.set_xticks([0, n_frames // 2, n_frames - 1])
    ax.set_xticklabels(['0', str(n_frames // 2), str(n_frames - 1)],
                       fontsize=FS_TICK)
    ax.set_xlabel('frame', fontsize=FS_LABEL)
    if col == 0:
        ax.set_yticks(range(n_sorted))
        ax.set_yticklabels(
            _thin_labels(sorted_names_ref, HEATMAP_LABEL_MIN_STEP),
            fontsize=FS_TYPE)
        ax.set_ylabel('cell type', fontsize=FS_LABEL)
    else:
        ax.set_yticks([])
    _trim_axis(ax, yaxis=False)


# -- Row 2: trace stacks (d, e, f) -------------------------------------------
for col, ((v_win, s_win, type_ids), (_cfg, sigma_lbl)) in enumerate(
        zip(trace_data, INTRINSIC_CONFIGS)):
    ax = axes_row1[col]
    # Steelblue blank-stimulus spans go down first so traces sit on top.
    _draw_blank_shading(ax, s_win)
    row = 0
    for t_idx in SELECTED_TYPES:
        neuron_rows = np.where(type_ids == t_idx)[0]
        if len(neuron_rows) == 0:
            continue
        nid   = neuron_rows[0]
        trace = v_win[nid]
        stim  = s_win[nid]
        bl    = trace.mean()

        ax.plot(trace - bl + row * step_v,
                lw=LW_GT, color=COLOR_GT, alpha=0.9)
        if stim.mean() > 0:
            ax.plot(stim - stim.mean() + row * step_v,
                    lw=LW_STIM, color=COLOR_STIM, alpha=0.9, linestyle='--')

        ax.text(-n_tframes * 0.025, row * step_v,
                INDEX_TO_NAME.get(t_idx, f'Type{t_idx}'),
                fontsize=FS_ANNOT, va='bottom', ha='right', color='black')
        row += 1

    n_rows = row
    ax.set_ylim([-step_v, (n_rows - 1) * step_v + step_v])
    ax.set_yticks([])
    ax.set_xticks([0, n_tframes // 2, n_tframes])
    ax.set_xticklabels([TRACE_START, (TRACE_START + TRACE_END) // 2, TRACE_END],
                       fontsize=FS_TICK)
    ax.set_xlabel('frame', fontsize=FS_LABEL)
    ax.set_xlim([-n_tframes * 0.08, n_tframes * 1.05])
    ax.set_title(sigma_lbl, fontsize=FS_TITLE, pad=4)
    ax.spines['left'].set_visible(False)
    if col == 0:
        ax.set_ylabel('voltage (a.u.)', fontsize=FS_LABEL, labelpad=12)
    _trim_axis(ax, yaxis=False)


# -- Row 3: single-R1 zoom (noise-free reference + 2 measurement-noise levels)
# Pre-compute the baseline-subtracted R1 trace for each panel so we can
# share a single y-axis range across g/h/i (fair noise comparison).
_r1_traces = []
for v_win, s_win, type_ids in r1_data:
    neuron_rows = np.where(type_ids == R1_TYPE_ID)[0]
    if len(neuron_rows) == 0:
        _r1_traces.append((None, s_win))
        continue
    nid   = neuron_rows[0]
    trace = v_win[nid]
    _r1_traces.append((trace - trace.mean(), s_win))

# Symmetric shared y-range covering every panel's most extreme value,
# with a 5% headroom so the trace edges aren't flush to the spine.
_r1_pad = 0.05
_r1_max = max(np.abs(t).max() for t, _ in _r1_traces if t is not None)
R1_YMIN = -(1 + _r1_pad) * _r1_max
R1_YMAX =  (1 + _r1_pad) * _r1_max

for col, ((trace_bl, s_win), (_gamma, title)) in enumerate(
        zip(_r1_traces, MEAS_GAMMAS)):
    ax = axes_row2[col]
    if trace_bl is None:
        ax.text(0.5, 0.5, f'no neurons of type R1 (id={R1_TYPE_ID})',
                ha='center', va='center', fontsize=FS_ANNOT,
                color='red', transform=ax.transAxes)
        continue

    _draw_blank_shading(ax, s_win)

    # Voltage trace only — stimulus overlay (red dashed) intentionally
    # omitted in the bottom row; the steelblue blank shading already
    # encodes when stimulus is on/off.
    ax.plot(trace_bl, lw=LW_R1, color=COLOR_GT, alpha=0.95, label='R1 voltage')

    ax.set_xticks([0, n_r1f // 2, n_r1f])
    ax.set_xticklabels([R1_TRACE_START,
                        (R1_TRACE_START + R1_TRACE_END) // 2,
                        R1_TRACE_END],
                       fontsize=FS_TICK)
    ax.set_xlabel('frame', fontsize=FS_LABEL)
    ax.set_xlim([0, n_r1f])
    ax.set_ylim([R1_YMIN, R1_YMAX])
    if col == 0:
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.set_ylabel('R1 voltage (a.u.)', fontsize=FS_LABEL)
    else:
        # Same scale as panel g; suppress repeated tick labels so h/i
        # have the same effective box width as a/b/c above.
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    _trim_axis(ax)


# Align row 3 (g/h/i) with the *trace content* of row 2 (d/e/f), not the
# full axes box. The trace stacks have an extended xlim that reserves a
# strip on the left for the type-name labels (text at negative data-x);
# without this remap, frame 0 of d sits ~7% inside the panel while frame
# 0 of g sits flush at the left edge — visually misaligning the columns.
fig.canvas.draw()
for col, ax_g in enumerate(axes_row2):
    ax_d = axes_row1[col]
    pos_d = ax_d.get_position()
    xlo_d, xhi_d = ax_d.get_xlim()
    # Figure-x of data frame 0 and data frame n_tframes inside row-2 panel.
    fig_x0 = pos_d.x0 + (0          - xlo_d) / (xhi_d - xlo_d) * pos_d.width
    fig_x1 = pos_d.x0 + (n_tframes  - xlo_d) / (xhi_d - xlo_d) * pos_d.width
    pos_g = ax_g.get_position()
    ax_g.set_position([fig_x0, pos_g.y0, fig_x1 - fig_x0, pos_g.height])


# -- shared colorbar — placed manually right next to panel c -----------------
fig.canvas.draw()
_pos = axes_row0[2].get_position()
_cax = fig.add_axes([
    _pos.x1 + 0.008,
    _pos.y0 + _pos.height * 0.10,
    0.012,
    _pos.height * 0.80,
])
cbar = fig.colorbar(last_im, cax=_cax)
cbar.set_label('z-scored voltage', fontsize=FS_CBAR)
cbar.ax.tick_params(labelsize=FS_CBAR)


# -- panel labels (a..h), one shared y per row -------------------------------
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv      = fig.transFigure.inverted()

bb_r0 = [ax.get_tightbbox(renderer) for ax in axes_row0]
bb_r1 = [ax.get_tightbbox(renderer) for ax in axes_row1]
bb_r2 = [ax.get_tightbbox(renderer) for ax in axes_row2]
y_r0 = max(inv.transform((bb.x0, bb.y1))[1] for bb in bb_r0)
y_r1 = max(inv.transform((bb.x0, bb.y1))[1] for bb in bb_r1)
y_r2 = max(inv.transform((bb.x0, bb.y1))[1] for bb in bb_r2)

row_letters = [
    (axes_row0, ['a', 'b', 'c'], y_r0, bb_r0),
    (axes_row1, ['d', 'e', 'f'], y_r1, bb_r1),
    (axes_row2, ['g', 'h', 'i'], y_r2, bb_r2),
]
for axes_row, lbls, y, bbs in row_letters:
    for bb, lbl in zip(bbs, lbls):
        x0 = inv.transform((bb.x0, bb.y1))[0]
        fig.text(x0, y, lbl, fontsize=PANEL_LBL, fontweight='bold',
                 va='bottom', ha='left', color='black',
                 transform=fig.transFigure)


# -- save --------------------------------------------------------------------
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
out_base = os.path.join(OUT_DIR, 'fig_simulations')
plt.savefig(out_base + '.pdf', bbox_inches='tight')
plt.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out_base}.pdf')
print(f'Saved: {out_base}.png')
