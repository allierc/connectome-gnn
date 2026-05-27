"""
Figure: input stimulus on the retinotopic hexagon lattice + per-cell-type voltage maps.

Layout (2 rows × 1 column block)
---------------------------------
  a) Stimulus series — 10 hexagon snapshots of the visual input across time.
  b) 8 × 9 spatial grid — top-left panel = stimulus, other 65 panels = voltage
     of each cell type on the hexagon lattice at a single frame. Identical
     layout to ``plot_spatial_activity_grid`` (the figure saved as
     ``Fig_0_000000.png`` during data generation), produced by calling
     ``_draw_hex_panel`` directly so each cell type renders onto the same
     retinotopic hex lattice (avoids the disc-of-random-points artefact).

Data source
-----------
x_list_train zarr (fields: voltage, stimulus, neuron_type, pos) for
config flyvis_noise_free — the training simulation.

Usage
-----
    conda run -n neural-graph-linux python figures/fig_stimulus_hexagons.py

Output
------
    figures/fig_stimulus_hexagons.{png,pdf,jpg}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Data root      : /groups/saalfeld/home/allierc/GraphData
# Configs        : <DATA_ROOT>/config/fly/flyvis_noise_free.yaml
# Training data  : <DATA_ROOT>/graphs_data/fly/flyvis_noise_free/x_list_train/
#                  <DATA_ROOT>/graphs_data/fly/flyvis_noise_free/{edge_index.pt, ode_params.pt}
# Output         : figures/fig_stimulus_hexagons.{png,pdf,jpg}
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
REPO = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(REPO, 'src'))

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.metrics import ANATOMICAL_ORDER, INDEX_TO_NAME
from connectome_gnn.utils import set_data_root, graphs_data_path, add_pre_folder
from connectome_gnn.zarr_io import load_simulation_data
from connectome_gnn.figure_style import default_style

# ── font style (INSTRUCTIONS.md §style) ──────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'text.usetex': False,
    'mathtext.fontset': 'dejavusans',
})

PANEL_LBL = 20

# ── data config ──────────────────────────────────────────────────────────────
CONFIG_NAME = 'flyvis_noise_free'
DATA_ROOT   = '/groups/saalfeld/home/allierc/GraphData'

SERIES_START   = 120
SERIES_STEP    = 1            # step of 1 frame between stimulus snapshots
SERIES_ROWS    = 2            # two rows of stimuli in panel a
SERIES_COLS    = 11
SERIES_FRAMES  = [SERIES_START + i * SERIES_STEP for i in range(SERIES_ROWS * SERIES_COLS)]
SNAPSHOT_FRAME = SERIES_FRAMES[0]   # first stimulus in a == stimulus in b
GRID_ROWS, GRID_COLS = 6, 11  # 6 × 11 = 66 slots = 1 stimulus + 65 cell types

# ── rendering style — same as data-generation default_style ──────────────────
style = default_style


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(config_name, data_root):
    set_data_root(data_root)
    cfg_path = os.path.join(REPO, 'config', 'fly', f'{config_name}.yaml')
    config   = NeuralGraphConfig.from_yaml(cfg_path)
    _, pre   = add_pre_folder(config_name)
    if not config.dataset.startswith(pre):
        config.dataset = pre + config.dataset
    gdata_dir = graphs_data_path(config.dataset)

    max_frame = max(max(SERIES_FRAMES), SNAPSHOT_FRAME) + 1
    x_ts = load_simulation_data(
        os.path.join(gdata_dir, 'x_list_train'),
        fields=['voltage', 'stimulus', 'neuron_type', 'pos'],
    )
    voltage   = x_ts.voltage[:max_frame].numpy().T.astype(np.float32)
    stimulus  = x_ts.stimulus[:max_frame].numpy().T.astype(np.float32)
    type_ids  = x_ts.neuron_type.numpy().astype(int)
    positions = x_ts.pos.numpy().astype(np.float32)
    n_inp     = int(config.simulation.n_input_neurons)
    dt_ms     = float(config.simulation.delta_t) * 1000.0   # s → ms
    return positions, voltage, stimulus, type_ids, n_inp, dt_ms


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
print(f'loading {CONFIG_NAME} ...')
positions, voltage, stimulus, type_ids, n_inp, dt_ms = load_data(CONFIG_NAME, DATA_ROOT)
print(f'  N={positions.shape[0]}  n_input={n_inp}  types={len(np.unique(type_ids))}  dt={dt_ms} ms')

vmin_v, vmax_v = -3.0, 3.0
vmin_s, vmax_s = -3.0, 3.0   # stimulus also displayed as z-score
CMAP        = 'RdBu_r'        # match the z-scored heatmap in fig_simulations.py
HEX_EDGE_C  = 'black'
HEX_EDGE_W  = 0.25
HEX_MARKER_S = 36             # same marker area in panel a and panel b

# Hex-lattice extent — pre-compute so every panel sets the same xlim/ylim and
# hence renders hexagons at identical size.
_px = positions[:n_inp, 0]
_py = positions[:n_inp, 1]
_pad_x = (_px.max() - _px.min()) * 0.03
_pad_y = (_py.max() - _py.min()) * 0.03
HEX_XLIM = (_px.min() - _pad_x, _px.max() + _pad_x)
HEX_YLIM = (_py.min() - _pad_y, _py.max() + _pad_y)


def _draw_hex(ax, xy, values, cmap, vmin, vmax):
    """Hex scatter with thin black outline + shared xlim/ylim so every panel
    draws the hexagons at identical size."""
    ax.scatter(
        xy[:, 0], xy[:, 1], c=values,
        s=HEX_MARKER_S, marker='h',
        cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors=HEX_EDGE_C, linewidths=HEX_EDGE_W,
        alpha=1.0,
    )
    ax.set_xlim(*HEX_XLIM)
    ax.set_ylim(*HEX_YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_facecolor(style.background)
    for spine in ax.spines.values():
        spine.set_visible(False)

# 3:2 figure aspect (w:h), wider layout
fig_w = 24.0
fig_h = 16.0

fig = plt.figure(figsize=(fig_w, fig_h), dpi=300, facecolor=style.background)
outer = mgs.GridSpec(
    2, 1, figure=fig,
    height_ratios=[SERIES_ROWS, GRID_ROWS],
    hspace=0.15,
)

def _zscore(v):
    return (v - v.mean()) / (v.std() + 1e-6)

# ── panel a) stimulus time-series — full hex on the 1736 photoreceptors ──────
gs_a = mgs.GridSpecFromSubplotSpec(SERIES_ROWS, SERIES_COLS,
                                    subplot_spec=outer[0],
                                    wspace=0.05, hspace=0.3)
axes_a = []
for k, t in enumerate(SERIES_FRAMES):
    r, c = divmod(k, SERIES_COLS)
    ax = fig.add_subplot(gs_a[r, c])
    stim_t = _zscore(stimulus[:n_inp, t])
    _draw_hex(ax, positions[:n_inp], stim_t, CMAP, vmin_s, vmax_s)
    ax.set_title(f't = {int(round(t * dt_ms))} ms', fontsize=style.font_size, pad=2)
    axes_a.append(ax)

# ── panel b) 8×9 grid — stimulus + 65 types, same as Fig_0_000000.png ────────
gs_b = mgs.GridSpecFromSubplotSpec(GRID_ROWS, GRID_COLS, subplot_spec=outer[1],
                                    wspace=0.05, hspace=0.3)
axes_b = []
stim_snap = _zscore(stimulus[:n_inp, SNAPSHOT_FRAME])

# Per-cell-type z-score at the snapshot frame: each hexagon is normalised
# by its own type's mean/std so all 65 panels share one comparable scale.
volt_snap_z = voltage[:, SNAPSHOT_FRAME].copy()
for _tid in np.unique(type_ids):
    _m = type_ids == _tid
    volt_snap_z[_m] = _zscore(volt_snap_z[_m])

n_panels = min(len(ANATOMICAL_ORDER), GRID_ROWS * GRID_COLS)
for idx in range(n_panels):
    type_idx = ANATOMICAL_ORDER[idx]
    r, c = divmod(idx, GRID_COLS)
    ax = fig.add_subplot(gs_b[r, c])
    if type_idx is None:
        _draw_hex(ax, positions[:n_inp], stim_snap, CMAP, vmin_s, vmax_s)
        name = 'stimulus'
    else:
        count = int((type_ids == type_idx).sum())
        vals = volt_snap_z[type_ids == type_idx]
        # Original plot reuses the first `count` retinotopic positions so
        # every cell type renders onto the same hexagonal lattice.
        _draw_hex(ax, positions[:count], vals, CMAP, vmin_v, vmax_v)
        name = INDEX_TO_NAME.get(type_idx, f'type_{type_idx}')
    ax.set_title(name, fontsize=style.font_size, pad=2)
    axes_b.append(ax)

# hide trailing cells
for idx in range(n_panels, GRID_ROWS * GRID_COLS):
    r, c = divmod(idx, GRID_COLS)
    ax = fig.add_subplot(gs_b[r, c])
    ax.set_visible(False)

# ── voltage colorbar — placed on the right of panel b ────────────────────────
fig.canvas.draw()
import matplotlib.cm as _mcm
import matplotlib.colors as _mcolors
# Use the last used right-most panel in panel b as anchor (column = GRID_COLS-1).
_last_col_axes = [axes_b[idx] for idx in range(n_panels)
                   if (idx % GRID_COLS) == (GRID_COLS - 1)]
if _last_col_axes:
    _anchor = _last_col_axes[len(_last_col_axes) // 2]   # middle of the right column
    _pos = _anchor.get_position()
    _cax_v = fig.add_axes([
        _pos.x1 + 0.010,
        _pos.y0 - _pos.height * 1.5,
        0.010,
        _pos.height * 4.0,
    ])
    _norm_v = _mcolors.Normalize(vmin=vmin_v, vmax=vmax_v)
    _sm_v = _mcm.ScalarMappable(norm=_norm_v, cmap=CMAP)
    _cbar_v = fig.colorbar(_sm_v, cax=_cax_v)
    _cbar_v.set_label('voltage (z-score)', fontsize=22)
    _cbar_v.ax.tick_params(labelsize=16)

# ── panel labels a) / b) — top-left of each outer row ────────────────────────
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
bb_a = axes_a[0].get_tightbbox(renderer)
bb_b = axes_b[0].get_tightbbox(renderer)
for bb, lbl in zip([bb_a, bb_b], ['a)', 'b)']):
    x0 = inv.transform((bb.x0, bb.y1))[0]
    y1 = inv.transform((bb.x0, bb.y1))[1]
    fig.text(x0, y1, lbl, fontsize=PANEL_LBL, fontweight='bold',
             va='bottom', ha='left', color='black', transform=fig.transFigure)

# ── save ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
out_base = os.path.join(OUT_DIR, 'fig_stimulus_hexagons')
fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
fig.savefig(out_base + '.pdf', bbox_inches='tight')
fig.savefig(out_base + '.jpg', dpi=300, bbox_inches='tight',
            pil_kwargs={'quality': 95})
plt.close(fig)
print(f'Saved: {out_base}.png')
print(f'Saved: {out_base}.pdf')
print(f'Saved: {out_base}.jpg')
