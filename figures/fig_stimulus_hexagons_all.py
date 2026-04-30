"""
Figure: stimulus + per-cell-type voltage maps on the full 721-column retinotopic
lattice (flyvis_noise_free_all, 45,669 neurons, 5,768 photoreceptors — roughly
3.3x more voxels than the 217-column flyvis_noise_free variant).

Janne-styled per figures/INSTRUCTIONS.md (the previous, larger-font version
is preserved at fig_stimulus_hexagons_all_original.py):

  * ~18 cm document-width figure (7.09 in) at 300 dpi
  * 6-8 pt fonts, 0.5 pt spines / ticks
  * top + right spines hidden globally (via janne.matplotlibrc)
  * trim_axis used on axes that have visible spines (the colorbar)
  * hexagon panels have no spines/ticks so trim_axis is skipped there
  * PDF primary output (pdf.fonttype=42, svg.fonttype='none')

Layout — same two-panel structure as the original:
  a) Stimulus series over consecutive 20-ms frames.
  b) 6 x 11 grid — stimulus + 65 cell types, per-hexagon z-score.

Usage
-----
    conda run -n neural-graph-linux python figures/fig_stimulus_hexagons_all.py

Output
------
    figures/fig_stimulus_hexagons_all.{pdf,png}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Data root      : /groups/saalfeld/home/allierc/GraphData
# Configs        : <DATA_ROOT>/config/fly/flyvis_noise_free_all.yaml
# Training data  : <DATA_ROOT>/graphs_data/fly/flyvis_noise_free_all/x_list_train/
#                  <DATA_ROOT>/graphs_data/fly/flyvis_noise_free_all/{edge_index.pt, ode_params.pt}
# Output         : figures/fig_stimulus_hexagons_all.{pdf,png}
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'janne.matplotlibrc'))

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


# ── font sizes (Janne 6-8 pt) ────────────────────────────────────────────────
FS_TITLE  = 6     # per-hexagon panel titles (cell-type names, time stamps)
FS_LABEL  = 7     # colorbar label
FS_TICK   = 6     # colorbar ticks
PANEL_LBL = 8     # a) / b) panel labels

# ── data config ──────────────────────────────────────────────────────────────
CONFIG_NAME = 'flyvis_noise_free_all'
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
HEX_EDGE_W  = 0.05            # thinner outline matches Janne 0.5 pt aesthetics
HEX_MARKER_S = 3              # scaled down for the smaller ~18 cm figure

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

# ~18 cm wide; height keeps the original 3:2 aspect (≈ 4.73 in).
FIG_W_IN = 18.0 * 0.3937          # ≈ 7.09 in
FIG_H_IN = FIG_W_IN * (20.0 / 30.0)  # preserve original 30:20 aspect

fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor=style.background)
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
    ax.set_title(f't = {int(round(t * dt_ms))} ms', fontsize=FS_TITLE, pad=2)
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
    ax.set_title(name, fontsize=FS_TITLE, pad=2)
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
    _cbar_v.set_label('voltage (z-score)', fontsize=FS_LABEL)
    _cbar_v.ax.tick_params(labelsize=FS_TICK, width=0.5)
    # Trim the colorbar's data axis (vertical) so the spine stops at the
    # first/last tick — Janne convention. Leave the (empty) horizontal axis.
    _trim_axis(_cax_v, xaxis=False, yaxis=True)

# ── panel labels a) / b) — top-left of each outer row ────────────────────────
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
bb_a = axes_a[0].get_tightbbox(renderer)
bb_b = axes_b[0].get_tightbbox(renderer)
for bb, lbl in zip([bb_a, bb_b], ['a', 'b']):
    x0 = inv.transform((bb.x0, bb.y1))[0]
    y1 = inv.transform((bb.x0, bb.y1))[1]
    fig.text(x0, y1, lbl, fontsize=PANEL_LBL, fontweight='bold',
             va='bottom', ha='left', color='black', transform=fig.transFigure)

# ── save ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
out_base = os.path.join(OUT_DIR, 'fig_stimulus_hexagons_all')
# PDF first per janne.matplotlibrc default; PNG for quick preview.
fig.savefig(out_base + '.pdf', bbox_inches='tight')
fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out_base}.pdf')
print(f'Saved: {out_base}.png')
