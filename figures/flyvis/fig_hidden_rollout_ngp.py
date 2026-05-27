"""GNN+InstantNGP hidden-neuron recovery and rollout - fig_hidden_rollout_ngp.py.

Modelled on figures/fig_stim_rollout_inr.py but for the hidden-neuron INR
(Instant-NGP, optionally spatial+temporal) instead of the visual SIREN.
The figure shows whether the joint NGP-T scheme can reconstruct the
voltages of the hidden neurons that the GNN never observes.

Layout (3 rows x 2 cols):

  row a) hexagonal map of per-neuron NGP R^2 across the eye, on the same
         (x, y) lattice as the rest of the repo's hex plots.
         Left  panel: hidden neurons only.
         Right panel: visible-anchor neurons (sanity check that anchor
         supervision is being learned everywhere).
  row b) NGP trace (b)   |  NGP scatter (c)
         12 evenly-sampled hidden neurons, ground truth (green) vs
         linearly-corrected NGP prediction (black). Hexbin density of
         (true, pred) pooled across all hidden neurons and frames; r is
         the Fisher-pooled per-neuron Pearson.
  row c) GNN rollout trace (d)  |  GNN rollout scatter (e)
         Standard rollout panels (same recipe as fig_stim_rollout_inr) —
         visible-neuron voltage rollout vs ground truth.

Data sources (rollout_bundle{,_on_*}.npz at
  <output_root>/log/fly/<config>/results/):
    - activity_true / activity_pred       (n_neurons, n_frames)
    - inr_true / inr_pred_raw / inr_pred_corr (n_hidden, n_frames)
    - inr_global_ids                      (n_hidden,)
    - inr_global_pos                      (n_hidden, 2)  -- requires the
                                          (u, v) pos fix in
                                          generators/graph_data_generator.py
                                          (2026-05-02). With pre-fix
                                          datasets, the hex map degrades
                                          to a random scatter.
    - inr_r2_per                          (n_hidden,) per-neuron R^2

Usage
-----
    /workspace/.conda_envs/neural-graph-linux/bin/python \\
        figures/fig_hidden_rollout_ngp.py [CONFIG_NAME]

    Default CONFIG_NAME = flyvis_noise_005_hidden_010_ngp_blank50_unified_spatial_cv00
    (the new spatial run); pass any other config to reuse the same layout
    on the time-only baseline (e.g.
    flyvis_noise_005_hidden_010_ngp_blank50_unified_cv00).

Output
------
    figures/fig_hidden_rollout_ngp__<config>.{pdf,png}
"""

import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'janne.matplotlibrc'))

import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.cm as _mcm
import matplotlib.colors as _mcolors
import numpy as np


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


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (os.path.join(REPO_ROOT, 'src'), REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from connectome_gnn.utils import (  # noqa: E402
    compute_trace_metrics, fisher_pool, graphs_data_path, load_data_root_from_json,
)
import connectome_gnn.utils as _cg_utils  # noqa: E402
from connectome_gnn.zarr_io import load_simulation_data  # noqa: E402


# ── config ──────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = 'flyvis_noise_005_hidden_010_ngp_blank50_unified_spatial_cv00'

DATA_ROOT = (os.environ.get('GNN_OUTPUT_ROOT') or load_data_root_from_json())


def _bundle_path(config_name):
    return os.path.join(
        DATA_ROOT, 'log', 'fly', config_name, 'results', 'rollout_bundle.npz'
    )


def _rollout_log(config_name):
    return os.path.join(DATA_ROOT, 'log', 'fly', config_name, 'results_rollout.log')


# Trace window (frame indices into the bundle arrays).
TRACE_START = 500
TRACE_END   = 1500
DT_MS       = 20.0
N_HIDDEN_TRACES = 12
N_VOLT_TRACES   = 12

COLOR_GT   = '#2ca02c'
COLOR_PRED = 'black'
LW_GT, LW_PRED = 1.2, 0.45

TRACE_SHRINK = 0.65

SCATTER_N_MAX = 2_000_000
SCATTER_RNG   = np.random.default_rng(0)
SCATTER_LO, SCATTER_HI = -7.5, 7.5  # voltage z-domain — hidden + visible share it

FS_LABEL  = 8
FS_TICK   = 6
FS_TYPE   = 6
PANEL_LBL = 8

FIG_W_IN  = 18.0 * 0.3937       # ~ 7.09 in
FIG_H_IN  = 11.0

# Hex R^2 colour ramp — clipped to [0, 1] so saturated R^2 reads as deep blue
# without distorting the median.
R2_CMAP = 'viridis'
R2_VMIN, R2_VMAX = 0.0, 1.0


# ── data loading ────────────────────────────────────────────────────────────
def _set_data_root(path):
    _cg_utils._data_root = path


def load_bundle(path):
    if not os.path.isfile(path):
        sys.exit(
            f'ERROR: bundle missing at {path}\n'
            '  re-run the test wave (e.g. python run_GNN_hidden_ngp_spatial.py '
            '--retest --replot) so graph_tester writes the inr_* arrays.'
        )
    b = np.load(path, allow_pickle=True)
    needed = ['inr_true', 'inr_pred_corr', 'inr_global_ids', 'inr_r2_per']
    missing = [k for k in needed if k not in b.files]
    if missing:
        sys.exit(
            'ERROR: rollout_bundle.npz is missing INR fields: '
            f'{missing}\n'
            '  re-run the test wave with the patched graph_tester '
            '(commit adds inr_r2_per + inr_global_pos and writes ALL hidden '
            'neurons rather than the first 20).'
        )
    return b


def load_positions_from_dataset(config_name):
    """Load per-neuron positions from x_list_train. Falls back to a list of
    plausible datasets so the figure keeps working if the cv-specific train
    zarr was deleted."""
    _set_data_root(DATA_ROOT)
    base = config_name.replace('_blank50_unified_spatial_cv00',
                               '_blank50_cv00')
    base = base.replace('_blank50_unified_cv00', '_blank50_cv00')
    candidates = [base, 'flyvis_noise_005_hidden_010_ngp_blank50_cv00',
                  'flyvis_noise_005']
    last_err = None
    for ds in candidates:
        path = os.path.join(graphs_data_path('fly', ds), 'x_list_train')
        try:
            x_ts = load_simulation_data(path, fields=['pos'])
            print(f'positions: loaded from {path}')
            return x_ts.pos.numpy().astype(np.float32)
        except FileNotFoundError as e:
            last_err = e
    raise FileNotFoundError(
        f'no x_list_train/pos found for {candidates} under '
        f'{DATA_ROOT}/graphs_data/fly/'
    ) from last_err


# ── helpers (same recipe as fig_stim_rollout_inr.py) ────────────────────────
def _draw_hex_r2(ax, xy, values, xlim, ylim,
                 vmin=R2_VMIN, vmax=R2_VMAX, cmap=R2_CMAP, marker_s=4):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values,
                    s=marker_s, marker='h',
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors='black', linewidths=0.05, alpha=1.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_visible(False)
    return sc


def draw_trace_panel(ax, true_w, pred_w, labels, step_v, time_ms,
                     pearson_r, header_label, show_xlabel,
                     show_type_labels=True, pearson_r_sd=None):
    n_traces, n_frames = true_w.shape
    baselines = true_w.mean(axis=1)
    s = TRACE_SHRINK
    for i in range(n_traces):
        bl = baselines[i]
        ax.plot(time_ms, s * (true_w[i] - bl) + i * step_v,
                lw=LW_GT, color=COLOR_GT, alpha=0.95, zorder=2)
        ax.plot(time_ms, s * (pred_w[i] - bl) + i * step_v,
                lw=LW_PRED, color=COLOR_PRED, alpha=0.95, zorder=3)
    if show_type_labels:
        for i, lbl in enumerate(labels):
            ax.text(time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.025,
                    i * step_v, lbl, fontsize=FS_TYPE,
                    va='bottom', ha='right', color='black')

    if pearson_r is None:
        r_txt = 'n/a'
    elif pearson_r_sd is not None:
        r_txt = f'{pearson_r:.2f} $\\pm$ {pearson_r_sd:.2f}'
    else:
        r_txt = f'{pearson_r:.2f}'
    ax.text(0.015, 0.99,
            f'{header_label}, $r$ = {r_txt}',
            transform=ax.transAxes, va='top', ha='left',
            fontsize=FS_TICK, fontweight='normal',
            bbox=dict(facecolor='white', edgecolor='none',
                      alpha=0.85, pad=0.4))

    ax.set_ylim([-step_v, (n_traces - 1) * step_v + 2.2 * step_v])
    ax.set_yticks([])
    _x_lo = float(time_ms[0])
    _x_hi = float(time_ms[-1] + DT_MS)
    ax.set_xlim([_x_lo, _x_hi])
    ax.spines['left'].set_visible(False)
    if show_xlabel:
        ticks = np.linspace(_x_lo, _x_hi, 3)
        ax.set_xticks(ticks)
        ax.set_xlabel('time (ms)', fontsize=FS_LABEL, labelpad=1)
        ax.tick_params(axis='x', labelsize=FS_TICK, pad=1)
        _trim_axis(ax, yaxis=False)
    else:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)


def _subsample_pair(x_full, y_full, n_max=SCATTER_N_MAX):
    assert x_full.shape == y_full.shape
    x = x_full.reshape(-1).astype(np.float32)
    y = y_full.reshape(-1).astype(np.float32)
    n_tot = x.size
    if n_tot <= n_max:
        return x, y, n_tot
    stride = int(np.ceil(n_tot / n_max))
    offset = int(SCATTER_RNG.integers(0, stride))
    return x[offset::stride], y[offset::stride], n_tot


def draw_scatter(ax, x_all, y_all, lo, hi, xlabel, ylabel, title=None):
    x_arr = np.asarray(x_all)
    y_arr = np.asarray(y_all)
    if x_arr.ndim >= 2 and y_arr.ndim >= 2:
        _, _pear, _, _ = compute_trace_metrics(x_arr, y_arr)
        _fp = fisher_pool(_pear)
        r, r_sd = float(_fp['r_mean']), float(_fp['r_sd_sym'])
    else:
        r = float(np.corrcoef(x_arr.ravel(), y_arr.ravel())[0, 1])
        r_sd = None
    x, y, _ = _subsample_pair(x_all, y_all)
    ax.hexbin(x, y, gridsize=140, bins='log', cmap='magma_r',
              mincnt=1, extent=(lo, hi, lo, hi), linewidths=0.0)
    ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    _mid = (lo + hi) / 2.0
    ax.set_xticks([lo, _mid, hi])
    ax.set_yticks([lo, _mid, hi])
    _trim_axis(ax)
    if title is not None:
        ax.text(0.5, 1.02, title, transform=ax.transAxes,
                va='bottom', ha='center', fontsize=FS_TICK,
                fontweight='normal')
    _r_txt = (f"$r$ = {r:.2f} $\\pm$ {r_sd:.2f}" if r_sd is not None
              else f"$r$ = {r:.2f}")
    ax.text(0.05, 0.97, _r_txt,
            transform=ax.transAxes, va='top', ha='left',
            fontsize=FS_TICK)


def _parse_rollout_log(path):
    out = {'voltage': None, 'hidden_R2': None}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        txt = f.read()
    m = re.search(r'Pearson r:\s*([-\d.]+)', txt)
    if m:
        out['voltage'] = float(m.group(1))
    m = re.search(r'hidden_nnr_R2:\s*([-\d.]+)', txt)
    if m:
        out['hidden_R2'] = float(m.group(1))
    return out


# ── main ────────────────────────────────────────────────────────────────────
def main():
    config_name = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG)
    bundle_path = _bundle_path(config_name)
    print(f'config: {config_name}')
    print(f'bundle: {bundle_path}')

    bundle = load_bundle(bundle_path)
    activity_true = bundle['activity_true']                  # (N, T)
    activity_pred = bundle['activity_pred']
    inr_true      = bundle['inr_true']                       # (n_hidden, T)
    inr_pred_corr = bundle['inr_pred_corr']
    inr_pred_raw  = bundle['inr_pred_raw'] if 'inr_pred_raw' in bundle.files else None
    inr_ids       = bundle['inr_global_ids'].astype(int)
    inr_r2_per    = bundle['inr_r2_per'].astype(np.float32)
    inr_pos       = (bundle['inr_global_pos'].astype(np.float32)
                     if 'inr_global_pos' in bundle.files else None)
    type_ids      = bundle['type_ids'].astype(int)
    type_names    = list(bundle['type_names'])
    inr_type      = (str(bundle['inr_type']) if 'inr_type' in bundle.files
                     else 'ngp_t')
    print(f'  inr_type           = {inr_type}')
    print(f'  hidden neurons     = {len(inr_ids)}')
    print(f'  inr R^2 (mean)     = {float(inr_r2_per.mean()):.3f}')
    print(f'  activity arrays    = {activity_true.shape}, {activity_pred.shape}')

    # Sanity check: pos must be retinotopically meaningful — if inr_pos is
    # absent or its (x, y) range is implausibly small, fall back to loading
    # positions from the dataset zarr.
    pos_full = None
    if inr_pos is None or inr_pos.shape[0] != len(inr_ids):
        try:
            pos_full = load_positions_from_dataset(config_name)
        except FileNotFoundError as exc:
            print(f'  WARNING: no positions found ({exc}); hex map will skip.')
    visible_mask = np.ones(activity_true.shape[0], dtype=bool)
    visible_mask[inr_ids] = False
    visible_ids = np.nonzero(visible_mask)[0]

    # ── visible-neuron rollout R^2 (per neuron) — used for the right hex map
    _act_t_vis = activity_true[visible_ids]
    _act_p_vis = activity_pred[visible_ids]
    _, _vis_pear, _, _ = compute_trace_metrics(_act_t_vis, _act_p_vis)
    visible_r2 = (_vis_pear ** 2).astype(np.float32)
    print(f'  visible R^2 (mean) = {float(visible_r2.mean()):.3f}')

    # ── per-neuron position arrays for the hex maps
    if inr_pos is not None and pos_full is None:
        # Bundle's hidden positions only — back-fill visible positions from
        # the bundle's (full) activity if available; otherwise skip the
        # visible map.
        pos_full = None
    if pos_full is None and inr_pos is not None:
        # Build a (N, 2) array with NaN for non-hidden neurons; the visible
        # hex map then plots only the hidden positions and we skip the
        # visible panel.
        full = np.full((activity_true.shape[0], 2), np.nan, dtype=np.float32)
        full[inr_ids] = inr_pos
        pos_full = full

    if pos_full is not None:
        pos_hidden = pos_full[inr_ids]
        pos_visible = pos_full[visible_ids]
        # Mask out NaNs that may exist in the visible set if pos_full came
        # from inr_global_pos only.
        vis_finite = np.all(np.isfinite(pos_visible), axis=1)
        pos_visible_plot = pos_visible[vis_finite]
        visible_r2_plot  = visible_r2[vis_finite]
    else:
        pos_hidden = None
        pos_visible_plot = None
        visible_r2_plot  = None

    # ── pick hidden traces (12 evenly-spaced)
    sel_h = np.linspace(0, len(inr_ids) - 1, N_HIDDEN_TRACES, dtype=int)
    inr_true_w = inr_true[sel_h, TRACE_START:TRACE_END].astype(np.float32)
    inr_pred_w = inr_pred_corr[sel_h, TRACE_START:TRACE_END].astype(np.float32)
    labels_h = [f'#{int(inr_ids[i])}' for i in sel_h]
    n_frames = inr_true_w.shape[1]
    time_ms = np.arange(n_frames) * DT_MS + TRACE_START * DT_MS

    # ── pick voltage traces (12 visible-neuron cell types — same selection
    # as fig_stim_rollout_inr.py to keep figures comparable)
    SELECTED_TYPES = [23, 5, 6, 7, 12, 22, 43, 55, 35, 39, 31, 0]
    index_to_name = {i: type_names[i] for i in range(len(type_names))}
    visible_type_ids = type_ids.copy()
    visible_type_ids[inr_ids] = -1
    neuron_idx, labels_v = [], []
    for t in SELECTED_TYPES:
        ids_t = np.where(visible_type_ids == t)[0]
        if len(ids_t) > 0:
            neuron_idx.append(int(ids_t[0]))
            labels_v.append(index_to_name.get(t, f'Type{t}'))
    true_v = activity_true[neuron_idx, TRACE_START:TRACE_END].astype(np.float32)
    pred_v = activity_pred[neuron_idx, TRACE_START:TRACE_END].astype(np.float32)
    step_v_inr  = max(0.5 * TRACE_SHRINK,
                      3.0 * TRACE_SHRINK * float(np.std(inr_true_w)))
    step_v_volt = max(0.5 * TRACE_SHRINK,
                      3.0 * TRACE_SHRINK * float(np.std(true_v)))

    # ── pearson r for headers
    _, _pear_v, _, _ = compute_trace_metrics(activity_true, activity_pred)
    _fp_v = fisher_pool(_pear_v)
    r_volt    = float(_fp_v['r_mean'])
    r_volt_sd = float(_fp_v['r_sd_sym'])
    _, _pear_h, _, _ = compute_trace_metrics(inr_true, inr_pred_corr)
    _fp_h = fisher_pool(_pear_h)
    r_inr    = float(_fp_h['r_mean'])
    r_inr_sd = float(_fp_h['r_sd_sym'])
    print(f'  voltage Pearson r (Fisher-pooled) = {r_volt:.3f}')
    print(f'  hidden  Pearson r (Fisher-pooled) = {r_inr:.3f}')

    # ── figure
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)
    outer = mgs.GridSpec(5, 1, figure=fig,
                         height_ratios=[1.0, 0.05, 1.5, 0.30, 1.5],
                         left=0.06, right=0.92, top=0.97, bottom=0.05,
                         hspace=0.0)

    # (a) Two hex panels side-by-side: hidden-neuron R^2, visible-neuron R^2.
    gs_a = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                        wspace=0.20)
    ax_a_hidden  = fig.add_subplot(gs_a[0, 0])
    ax_a_visible = fig.add_subplot(gs_a[0, 1])

    if pos_hidden is not None:
        valid = np.all(np.isfinite(pos_hidden), axis=1)
        if valid.any():
            xs = pos_hidden[valid, 0]; ys = pos_hidden[valid, 1]
            _pad_x = (xs.max() - xs.min()) * 0.05
            _pad_y = (ys.max() - ys.min()) * 0.05
            HEX_XLIM = (xs.min() - _pad_x, xs.max() + _pad_x)
            HEX_YLIM = (ys.min() - _pad_y, ys.max() + _pad_y)
            sc_h = _draw_hex_r2(ax_a_hidden, pos_hidden[valid],
                                 inr_r2_per[valid], HEX_XLIM, HEX_YLIM)
            ax_a_hidden.text(0.0, 1.05, 'hidden NGP $R^2$',
                              transform=ax_a_hidden.transAxes,
                              va='bottom', ha='left', fontsize=FS_LABEL)
            if pos_visible_plot is not None and len(pos_visible_plot) > 0:
                sc_v = _draw_hex_r2(ax_a_visible, pos_visible_plot,
                                    visible_r2_plot, HEX_XLIM, HEX_YLIM)
                ax_a_visible.text(0.0, 1.05, 'visible rollout $R^2$',
                                   transform=ax_a_visible.transAxes,
                                   va='bottom', ha='left', fontsize=FS_LABEL)
            else:
                ax_a_visible.set_axis_off()
                ax_a_visible.text(0.5, 0.5,
                                   'visible positions unavailable\n'
                                   '(bundle predates inr_global_pos)',
                                   transform=ax_a_visible.transAxes,
                                   va='center', ha='center', fontsize=FS_TICK,
                                   color='red')
        else:
            ax_a_hidden.set_axis_off()
            ax_a_visible.set_axis_off()

        # Single shared colorbar for both hex panels.
        fig.canvas.draw()
        _norm = _mcolors.Normalize(vmin=R2_VMIN, vmax=R2_VMAX)
        _sm = _mcm.ScalarMappable(norm=_norm, cmap=R2_CMAP)
        _pos_v = ax_a_visible.get_position()
        _cax = fig.add_axes([_pos_v.x1 + 0.012,
                              _pos_v.y0 + (_pos_v.height * 0.20),
                              0.008,
                              _pos_v.height * 0.60])
        _cbar = fig.colorbar(_sm, cax=_cax)
        _cbar.set_label('$R^2$', fontsize=FS_LABEL)
        _cbar.ax.tick_params(labelsize=FS_TICK)
        _cbar.outline.set_linewidth(0.5)
    else:
        ax_a_hidden.set_axis_off()
        ax_a_visible.set_axis_off()
        ax_a_hidden.text(0.5, 0.5,
                          'no per-neuron positions in bundle and no '
                          'x_list_train fallback;\nrun the data regen '
                          '(run_generate_hidden_010_ngp_blank50.py) and '
                          'retest.',
                          transform=ax_a_hidden.transAxes,
                          va='center', ha='center', fontsize=FS_TICK,
                          color='red')

    # (b + c) Hidden-neuron NGP row.
    gs_bc = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2],
                                         wspace=0.30, width_ratios=[1.0, 1.0])
    ax_b = fig.add_subplot(gs_bc[0, 0])
    ax_c = fig.add_subplot(gs_bc[0, 1])
    draw_trace_panel(ax_b, inr_true_w, inr_pred_w, labels_h,
                     step_v_inr, time_ms,
                     pearson_r=r_inr, pearson_r_sd=r_inr_sd,
                     header_label=f'hidden NGP ({inr_type})',
                     show_xlabel=True, show_type_labels=True)
    draw_scatter(ax_c, inr_true, inr_pred_corr,
                 lo=SCATTER_LO, hi=SCATTER_HI,
                 xlabel='true hidden voltage',
                 ylabel='NGP prediction (corrected)',
                 title=None)

    # (d + e) Visible-neuron rollout row.
    gs_de = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[4],
                                         wspace=0.30, width_ratios=[1.0, 1.0])
    ax_d = fig.add_subplot(gs_de[0, 0])
    ax_e = fig.add_subplot(gs_de[0, 1])
    draw_trace_panel(ax_d, true_v, pred_v, labels_v,
                     step_v_volt, time_ms,
                     pearson_r=r_volt, pearson_r_sd=r_volt_sd,
                     header_label='visible rollout, GNN vs ground truth',
                     show_xlabel=True, show_type_labels=True)
    draw_scatter(ax_e, activity_true, activity_pred,
                 lo=SCATTER_LO, hi=SCATTER_HI,
                 xlabel='ground truth voltage', ylabel='rollout voltage',
                 title=None)

    # Panel labels a..e
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    anchors = [(ax_a_hidden, 'a'),
               (ax_b, 'b'), (ax_d, 'd'),
               (ax_c, 'c'), (ax_e, 'e')]
    for ax_anchor, lbl in anchors:
        bb = ax_anchor.get_tightbbox(renderer)
        x0, y1 = inv.transform((bb.x0, bb.y1))
        fig.text(x0, y1, lbl, fontsize=PANEL_LBL, fontweight='bold',
                 va='bottom', ha='left', color='black',
                 transform=fig.transFigure)

    out_base = os.path.join(_SCRIPT_DIR,
                             f'fig_hidden_rollout_ngp__{config_name}')
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_base}.pdf')
    print(f'Saved: {out_base}.png')


if __name__ == '__main__':
    main()
