"""
Figure: effect of blank stimulus fraction on V_rest recovery.

Janne-styled per figures/INSTRUCTIONS.md (the previous, larger-font version
is preserved at fig_vrest_blank_original.py):

  • ~18 cm document-width figure (7.09 in) at 300 dpi
  • 6–8 pt fonts, 0.5 pt spines / ticks
  • top + right spines hidden globally (via janne.matplotlibrc)
  • trim_axis breaks each axis at the data range (upper & right gap)
  • PDF primary output (pdf.fonttype=42, svg.fonttype='none')

2×4 layout
----------
  row 0 (a–d): simulation panels — heatmap (a, b) + traces (c, d)
  row 1 (e–h): parameter recovery — V_rest scatters (e, f) +
               R² vs blank fraction (g) + rollout Pearson r vs blank fraction (h)

Best checkpoint is selected via sort_key (same logic as GNN_PlotFigure 'best').
Panel labels sit at the top-left of the outer panel box via get_tightbbox.

Usage
-----
    conda run -n neural-graph-linux python figures/fig_vrest_blank.py

Output
------
    figures/fig_vrest_blank.{pdf,png}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Data root      : /groups/saalfeld/home/allierc/GraphData
# Configs        : <REPO>/config/fly/flyvis_noise_005.yaml
#                  <REPO>/config/fly/flyvis_noise_005_blank50.yaml
# Training data  : <DATA_ROOT>/graphs_data/fly/flyvis_noise_005/x_list_train/
#                  <DATA_ROOT>/graphs_data/fly/flyvis_noise_005_blank50/x_list_train/
#                  <DATA_ROOT>/graphs_data/fly/flyvis_noise_005{,_blank50}/ode_params.pt
# Trained models : <DATA_ROOT>/log/fly/flyvis_noise_005{,_blank50}/models/best_model_with_*.pt (latest sort_key)
# Inline metrics : DATA dict in this script (no external load)
# Output         : figures/fig_vrest_blank.{pdf,png}
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import matplotlib
matplotlib.use("Agg")
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'janne.matplotlibrc'))

import matplotlib.pyplot as plt
import numpy as np
import torch

# ── project imports ──────────────────────────────────────────────────────────
REPO = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(REPO, 'src'))

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.metrics import compute_r_squared_NSE, get_model_W
from connectome_gnn.models.registry import create_model
from connectome_gnn.generators.ode_params import get_ode_params_class, FlyVisODEParams
from connectome_gnn.utils import (
    to_numpy, migrate_state_dict, set_data_root, graphs_data_path, log_path, add_pre_folder,
    sort_key,
)


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


# ---------------------------------------------------------------------------
# Data roots
# ---------------------------------------------------------------------------
DATA_ROOT_REMOTE = '/groups/saalfeld/home/allierc/GraphData'

# ---------------------------------------------------------------------------
# Scatter panel configs: (config_name, output_root, title)
# ---------------------------------------------------------------------------
CONFIGS = [
    ('flyvis_noise_005',         DATA_ROOT_REMOTE, 'no blank stimulus'),
    ('flyvis_noise_005_blank50', DATA_ROOT_REMOTE, r'50% blank stimulus ($I_i(t)=0$)'),
]

# ---------------------------------------------------------------------------
# Metrics: blank_pct → (W_R2, tau_R2, Vrest_R2, cluster_acc, rollout_pearson)
# ---------------------------------------------------------------------------
DATA = {
     0: (0.9193, 0.9735, 0.1899, 0.8242, 0.975),
     1: (0.9569, 0.9904, 0.3721, 0.8541, 0.971),
     5: (0.9820, 0.9863, 0.6833, 0.8363, 0.997),
    10: (0.9288, 0.9835, 0.5088, 0.8568, 0.997),
    25: (0.9588, 0.9907, 0.7019, 0.8621, 0.998),
    50: (0.9567, 0.9959, 0.7900, 0.8131, 1.000),
    75: (0.9671, 0.9764, 0.6883, 0.8410, 0.989),
}

# ---------------------------------------------------------------------------
# Janne-style font sizes — 6–8 pt
# ---------------------------------------------------------------------------
FS_LABEL  = 8     # axis labels
FS_TICK   = 6     # tick labels
FS_ANNOT  = 6     # in-plot annotation (R², slope, N)
FS_LEGEND = 6     # legend
FS_TITLE  = 7     # panel subtitle
PANEL_LBL = 8     # a) b) c) d)
MARKER_S  = 3
LW        = 0.7
FS_CBAR   = 6     # colorbar label / ticks
FS_TYPE   = 6     # cell-type name annotations in trace panels

# ~18 cm document-width figure; height matches the 2-row layout of the original.
FIG_W_IN = 18.0 * 0.3937   # ≈ 7.09 in
FIG_H_IN = 7.09            # two rows of 4 panels each, square-ish subpanels


# ── extra imports needed for V_rest extraction ───────────────────────────────
from connectome_gnn.zarr_io import load_simulation_data
from connectome_gnn.metrics import (
    compute_activity_stats,
    _vectorized_linspace, _batched_mlp_eval,
    _vectorized_linear_fit, _build_f_theta_features,
    ANATOMICAL_ORDER, INDEX_TO_NAME,
)


# ---------------------------------------------------------------------------
# Simulation panel constants
# ---------------------------------------------------------------------------
SIM_CONFIGS = [
    ('flyvis_noise_005',         DATA_ROOT_REMOTE, 'no blank stimulus'),
    ('flyvis_noise_005_blank50', DATA_ROOT_REMOTE, r'50% blank ($I_i(t)=0$)'),
]
N_SIM_FRAMES    = 2000
SIM_TRACE_START = 100
SIM_TRACE_END   = 600
SIM_TYPES       = [23, 5, 6, 7, 12, 22, 43, 55, 35, 39, 31, 0]  # R1…Am
SIM_CMAP        = 'RdBu_r'
SIM_VLIM        = 2.0
SIM_LW          = 0.5
SIM_LW_STIM     = 0.4
SIM_COLOR       = 'black'
SIM_COLOR_STIM  = 'red'


def _type_heatmap(voltage, type_ids, n_types, anat_order):
    """Type-mean z-scored heatmap (anatomically sorted)."""
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
    return z[valid], [INDEX_TO_NAME.get(i, f'Type{i}') for i in valid]


def load_sim_data(config_name, output_root):
    """Load voltage / stimulus / type_ids from x_list_train zarr."""
    set_data_root(output_root)
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
    voltage  = x_ts.voltage[:N_SIM_FRAMES].numpy().T.astype(np.float32)
    stimulus = x_ts.stimulus[:N_SIM_FRAMES].numpy().T.astype(np.float32)
    type_ids = x_ts.neuron_type.numpy().astype(int)
    return voltage, stimulus, type_ids


# ---------------------------------------------------------------------------
# Helper: load V_rest arrays for one config
# Replicates GNN_PlotFigure.py non-linear model branch (lines ~1029–1118)
# ---------------------------------------------------------------------------
def load_vrest(config_name: str, output_root: str):
    set_data_root(output_root)
    cfg_path = os.path.join(REPO, 'config', 'fly', f'{config_name}.yaml')
    config = NeuralGraphConfig.from_yaml(cfg_path)
    # add sub-folder prefix (e.g. 'fly/') the same way GNN_Main does
    _, pre = add_pre_folder(config_name)
    if not config.dataset.startswith(pre):
        config.dataset = pre + config.dataset
    config.config_file = pre + config_name

    n_neurons = config.simulation.n_neurons
    device = 'cpu'

    # ── ground truth ──────────────────────────────────────────────────────────
    try:
        OdeCls = get_ode_params_class(config.graph_model.signal_model_name)
    except (KeyError, AttributeError):
        OdeCls = FlyVisODEParams
    gdata_dir = graphs_data_path(config.dataset)
    ode_params = OdeCls.load(gdata_dir, device=device)
    gt_vrest = ode_params.gt_vrest(n_neurons)

    # ── model checkpoint — same selection logic as GNN_PlotFigure 'best' ─────
    run_log_dir = log_path(config.config_file)
    import glob as _glob
    ckpt_files = _glob.glob(os.path.join(run_log_dir, 'models', 'best_model_with_*.pt'))
    ckpt_files.sort(key=sort_key)
    ckpt = ckpt_files[-1]   # highest sort_key = best epoch
    print(f'  loading checkpoint: {os.path.basename(ckpt)}')
    state = torch.load(ckpt, map_location=device, weights_only=False)
    migrate_state_dict(state)
    if 'W' in state.get('model_state_dict', {}):
        config.simulation.n_edges = state['model_state_dict']['W'].shape[0]
    model = create_model(config.graph_model.signal_model_name,
                         aggr_type=config.graph_model.aggr_type,
                         config=config, device=device)
    model.load_state_dict(state['model_state_dict'], strict=False)
    model.eval()

    # ── per-neuron activity mu/sigma (load first 2000 frames for speed) ───────
    x_path = os.path.join(gdata_dir, 'x_list_train')
    x_ts = load_simulation_data(x_path, fields=['voltage'])
    # subsample to first 2000 frames — enough for stable mu/sigma
    x_ts.voltage = x_ts.voltage[:2000]
    mu_t, sigma_t = compute_activity_stats(x_ts, device)
    mu    = to_numpy(mu_t).astype(np.float32)
    sigma = to_numpy(sigma_t).astype(np.float32)

    # ── f_theta domain + slope extraction (same as GNN_PlotFigure L1029–1036) ─
    n_pts = 1000
    starts_phi = mu - 2 * sigma
    ends_phi   = mu + 2 * sigma
    with torch.no_grad():
        rr_domain_phi   = _vectorized_linspace(starts_phi, ends_phi, n_pts, device)
        func_domain_phi = _batched_mlp_eval(
            model.f_theta, model.a[:n_neurons], rr_domain_phi,
            lambda rr_f, emb_f: _build_f_theta_features(rr_f, emb_f), device,
        )
    slopes, offsets = _vectorized_linear_fit(rr_domain_phi, func_domain_phi)

    # ── derive V_rest: V_rest = -offset / slope ────────────────────────────────
    learned_vrest = ode_params.derive_vrest(slopes, offsets, n_neurons)

    return gt_vrest, learned_vrest


# ---------------------------------------------------------------------------
# Build figure — 2 rows: row 0 = simulation data (a–d),
#                         row 1 = parameter recovery (e–h)
# ---------------------------------------------------------------------------
import matplotlib.gridspec as _mgs

fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), constrained_layout=True)
_gs = _mgs.GridSpec(2, 4, figure=fig, height_ratios=[1, 1], hspace=0.06)
# row 0: simulation — heatmaps (a, b) then traces (c, d)
ax_e, ax_f = fig.add_subplot(_gs[0, 0]), fig.add_subplot(_gs[0, 1])
ax_g, ax_h = fig.add_subplot(_gs[0, 2]), fig.add_subplot(_gs[0, 3])
sim_hm = [ax_e, ax_f]   # heatmap panels
sim_tr = [ax_g, ax_h]   # trace panels
# row 1: parameter recovery (e, f, g, h)
axes = [fig.add_subplot(_gs[1, i]) for i in range(4)]

# ── panels a & b: V_rest scatter ─────────────────────────────────────────────
scatter_axes = []
for ax, (config_name, output_root, title) in zip(axes[:2], CONFIGS):
    gt, learned = load_vrest(config_name, output_root)
    r2, slope   = compute_r_squared_NSE(gt, learned)
    n           = len(gt)

    ax.scatter(gt, learned, c='k', s=0.5, alpha=0.3)
    ax.set_ylim(-1, 1.75)
    ax.text(0.05, 0.95,
            f'R²: {r2:.2f}\nslope: {slope:.2f}\nN: {n:,}',
            transform=ax.transAxes, va='top', fontsize=FS_ANNOT)
    ax.set_xlabel(r'true $V_{rest}$',    fontsize=FS_LABEL)
    ax.set_ylabel(r'learned $V_{rest}$', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    _trim_axis(ax)
    scatter_axes.append(ax)

# ── panel c: parameter R² vs blank % ─────────────────────────────────────────
ax_c  = axes[2]
pcts  = sorted(DATA.keys())
metric_series = [
    ([DATA[p][0] for p in pcts], "$R^2_W$",                    '#1f77b4'),
    ([DATA[p][1] for p in pcts], r'$R^2_\tau$',                '#ff7f0e'),
    ([DATA[p][2] for p in pcts], r'$R^2_{V_\mathrm{rest}}$',   '#d62728'),
    ([DATA[p][3] for p in pcts], 'cluster acc',                 '#2ca02c'),
]
for vals, key, col in metric_series:
    ax_c.plot(pcts, vals, marker='o', markersize=MARKER_S, linestyle='none', color=col, label=key)
ax_c.set_xlim(-2, 78);  ax_c.set_ylim(0, 1.05)
ax_c.set_xticks(pcts);  ax_c.set_xticklabels([str(p) for p in pcts], fontsize=FS_TICK)
ax_c.tick_params(axis='y', labelsize=FS_TICK)
ax_c.set_xlabel('blank fraction (%)', fontsize=FS_LABEL)
ax_c.set_ylabel('$R^2$ / accuracy',   fontsize=FS_LABEL)
ax_c.legend(fontsize=FS_LEGEND, frameon=False)
_trim_axis(ax_c)

# ── panel d: rollout Pearson r vs blank % ─────────────────────────────────────
ax_d    = axes[3]
pcts_r  = [p for p in pcts if DATA[p][4] is not None]
rollout = [DATA[p][4] for p in pcts_r]
ax_d.plot(pcts_r, rollout, marker='o', markersize=MARKER_S, linestyle='none', color='#9467bd')
ax_d.set_xlim(-2, 78);  ax_d.set_ylim(0, 1.05)
ax_d.set_xticks(pcts);  ax_d.set_xticklabels([str(p) for p in pcts], fontsize=FS_TICK)
ax_d.tick_params(axis='y', labelsize=FS_TICK)
ax_d.set_xlabel('blank fraction (%)',  fontsize=FS_LABEL)
ax_d.set_ylabel('rollout Pearson $r$', fontsize=FS_LABEL)
_trim_axis(ax_d)

# ---------------------------------------------------------------------------
# Row 1: simulation data panels (e–h)
# ---------------------------------------------------------------------------
_anat_order = ANATOMICAL_ORDER
_n_types    = len(INDEX_TO_NAME)   # 65
_sorted_names = None
_sim_last_im  = None
_sim_step_v   = None

for _col, (_cfg, _root, _title) in enumerate(SIM_CONFIGS):
    print(f'loading sim data {_cfg} ...')
    _vol, _stim, _tids = load_sim_data(_cfg, _root)

    # ── heatmap ──────────────────────────────────────────────────────────────
    _hz, _snames = _type_heatmap(_vol, _tids, _n_types, _anat_order)
    if _sorted_names is None:
        _sorted_names = _snames
    _ax_h = sim_hm[_col]
    _im   = _ax_h.imshow(_hz, aspect='auto', interpolation='nearest',
                         cmap=SIM_CMAP, vmin=-SIM_VLIM, vmax=SIM_VLIM, origin='upper')
    _sim_last_im = _im
    _ax_h.set_title(_title, fontsize=FS_TITLE, pad=4)
    _nf = _hz.shape[1]
    _ax_h.set_xticks([0, _nf // 2, _nf - 1])
    _ax_h.set_xticklabels(['0', str(_nf // 2), str(_nf - 1)], fontsize=FS_TICK)
    _ax_h.set_xlabel('frame', fontsize=FS_LABEL)
    if _col == 0:
        _ax_h.set_yticks(range(len(_sorted_names)))
        _ax_h.set_yticklabels(_sorted_names, fontsize=FS_TYPE)
        _ax_h.set_ylabel('cell type', fontsize=FS_LABEL)
    else:
        _ax_h.set_yticks([])

    # ── traces ────────────────────────────────────────────────────────────────
    _v_win = _vol[:,  SIM_TRACE_START:SIM_TRACE_END]
    _s_win = _stim[:, SIM_TRACE_START:SIM_TRACE_END]
    if _col == 0:
        _free = np.stack([_v_win[np.where(_tids == t)[0][0]]
                          for t in SIM_TYPES if len(np.where(_tids == t)[0]) > 0])
        _sim_step_v = max(0.5, 3.0 * float(np.std(_free)))

    _ntf = SIM_TRACE_END - SIM_TRACE_START
    _ax_t = sim_tr[_col]
    _row  = 0
    for _t in SIM_TYPES:
        _nrows = np.where(_tids == _t)[0]
        if len(_nrows) == 0:
            continue
        _tr  = _v_win[_nrows[0]]
        _st  = _s_win[_nrows[0]]
        _bl  = _tr.mean()
        _ax_t.plot(_tr - _bl + _row * _sim_step_v, lw=SIM_LW, color=SIM_COLOR, alpha=0.9)
        if _st.mean() > 0:
            _ax_t.plot(_st - _st.mean() + _row * _sim_step_v,
                       lw=SIM_LW_STIM, color=SIM_COLOR_STIM, alpha=0.9, linestyle='--')
        _ax_t.text(-_ntf * 0.025, _row * _sim_step_v,
                   INDEX_TO_NAME.get(_t, f'Type{_t}'),
                   fontsize=FS_TYPE, va='bottom', ha='right', color='black')
        _row += 1

    # ── blank-period shading (semi-transparent blue spans) ───────────────────
    _smean = _s_win.mean(axis=0)          # mean stimulus over neurons, per frame
    _bmask = _smean < 0.01                # True where stimulus is absent
    if _bmask.any():
        _bd = np.diff(_bmask.astype(int), prepend=0, append=0)
        for _bs, _be in zip(np.where(_bd == 1)[0], np.where(_bd == -1)[0]):
            _ax_t.axvspan(_bs, _be, alpha=0.18, color='steelblue',
                          linewidth=0, zorder=0)

    _ax_t.set_ylim([-_sim_step_v, (_row - 1) * _sim_step_v + _sim_step_v])
    _ax_t.set_yticks([])
    _ax_t.set_xticks([0, _ntf // 2, _ntf])
    _ax_t.set_xticklabels([SIM_TRACE_START, (SIM_TRACE_START + SIM_TRACE_END) // 2,
                            SIM_TRACE_END], fontsize=FS_TICK)
    _ax_t.set_xlabel('frame', fontsize=FS_LABEL)
    _ax_t.set_xlim([-_ntf * 0.08, _ntf * 1.05])
    _ax_t.set_title(_title, fontsize=FS_TITLE, pad=4)
    # Left spine hidden because traces carry no quantitative y-axis here.
    _ax_t.spines['left'].set_visible(False)
    if _col == 0:
        _ax_t.set_ylabel('voltage (a.u.)', fontsize=FS_LABEL, labelpad=12)
    _trim_axis(_ax_t, yaxis=False)

    del _vol, _stim

# colorbar right of panel f (heatmap 50% blank)
_cbar = fig.colorbar(_sim_last_im, ax=ax_f, fraction=0.046, shrink=0.9, pad=0.02)
_cbar.set_label('z-scored voltage', fontsize=FS_CBAR)
_cbar.ax.tick_params(labelsize=FS_CBAR)

# ── panel labels — all at the same y (max top across all panels) ──────────────
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()

_row0_axes = [ax_e, ax_f, ax_g, ax_h]          # simulation row (now first)
_row1_axes = [*scatter_axes, ax_c, ax_d]       # parameter row (now second)
_bb0 = [ax.get_tightbbox(renderer) for ax in _row0_axes]
_bb1 = [ax.get_tightbbox(renderer) for ax in _row1_axes]
_y0  = max(inv.transform((bb.x0, bb.y1))[1] for bb in _bb0)
_y1  = max(inv.transform((bb.x0, bb.y1))[1] for bb in _bb1)
for bb, lbl, y in zip(_bb0 + _bb1,
                      ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                      [_y0]*4 + [_y1]*4):
    x0 = inv.transform((bb.x0, bb.y1))[0]
    fig.text(x0, y, lbl, fontsize=PANEL_LBL, fontweight='bold',
             va='bottom', ha='left', color='black', transform=fig.transFigure)

# ---------------------------------------------------------------------------
# Save — PDF first per janne.matplotlibrc default; PNG for quick preview.
# ---------------------------------------------------------------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
out_pdf = os.path.join(OUT_DIR, 'fig_vrest_blank.pdf')
out_png = os.path.join(OUT_DIR, 'fig_vrest_blank.png')
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out_pdf}')
print(f'Saved: {out_png}')
