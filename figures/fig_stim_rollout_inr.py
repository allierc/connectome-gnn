"""
Figure: GNN+INR stimulus recovery and rollout on flyvis_noise_005_INR_davis_cv00.

Layout (4 rows):
  a) 2 x 11 hex grid of GT photoreceptor stimuli across 11 time points
     (row 1 = GT, row 2 = INR-predicted stimulus at the same time points).
  b) voltage rollout — 12 representative cell types, GT vs GNN prediction.
  c) stimulus rollout — 12 representative photoreceptors, GT vs INR prediction.

Data sources (rollout_bundle.npz at
  <output_root>/log/fly/flyvis_noise_005_INR_davis_cv00/results/):
  - activity_true / activity_pred            (n_neurons, n_frames)
  - stimulus                                 (n_neurons, n_frames)   GT, 13 741-wide
  - stimulus_input_true / stimulus_input_pred (n_frames, n_input)    produced by the
                                             augmented graph_tester.py; re-run
                                             `-o test` once if missing.
  - type_ids, type_names

Hex positions come from the simulation data (x_list_train/pos field).

Usage:
    /workspace/.conda_envs/neural-graph-linux/bin/python \\
        figures/fig_stim_rollout_inr.py

Output:
    figures/fig_stim_rollout_inr.{png,pdf,jpg}
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.cm as _mcm
import matplotlib.colors as _mcolors
import numpy as np


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (os.path.join(REPO_ROOT, 'src'), REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import connectome_gnn.utils as _cg_utils  # noqa: E402
from connectome_gnn.utils import graphs_data_path  # noqa: E402
from connectome_gnn.zarr_io import load_simulation_data  # noqa: E402


# ── config ───────────────────────────────────────────────────────────────────
CONFIG_NAME = 'flyvis_noise_005_INR_davis_cv00'
DATA_ROOT = '/groups/saalfeld/home/allierc/GraphData'
BASE_DIR = os.path.join(DATA_ROOT, 'log', 'fly', CONFIG_NAME)
BUNDLE_PATH = os.path.join(BASE_DIR, 'results', 'rollout_bundle.npz')

# hexagon panel — 2 rows x 11 cols at evenly spaced frames within trace window
N_INPUT = 1736                # photoreceptor count for 217-column flyvis
SERIES_COLS = 11

# trace window (frame indices into rollout_bundle arrays)
TRACE_START = 500
TRACE_END   = 1500
DT_MS = 20.0

# one neuron per type for voltage traces
SELECTED_TYPES = [23, 5, 6, 7, 12, 22, 43, 55, 35, 39, 31, 0]

# photoreceptor picks for stimulus traces — 12 indices spanning input_neurons.
# Default: evenly spaced across 0..N_INPUT.
N_STIM_TRACES = 12

COLOR_GT   = '#66cc66'
COLOR_PRED = 'black'
COLOR_RES  = '#cf222e'
LW_GT, LW_PRED, RES_LW = 1.8, 0.7, 1.2

_S = 0.52
FS_LABEL  = int(48 * _S)
FS_TICK   = int(24 * _S)
FS_ANNOT  = int(32 * _S)
FS_LEGEND = int(28 * _S)
FS_TYPE   = int(26 * _S)
PANEL_LBL = 20

CMAP = 'RdBu_r'
HEX_VMIN, HEX_VMAX = -3.0, 3.0
HEX_MARKER_S = 36
HEX_EDGE_C = 'black'
HEX_EDGE_W = 0.25

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'text.usetex': False,
    'mathtext.fontset': 'dejavusans',
})


# ── data loading ─────────────────────────────────────────────────────────────
def _set_data_root(path):
    _cg_utils._data_root = path


def load_bundle(path):
    if not os.path.isfile(path):
        sys.exit(
            f'ERROR: bundle missing at {path}\n'
            '  re-run `-o test` to regenerate with the new stimulus fields:\n'
            f'    python GNN_Main.py -o test {CONFIG_NAME} best {CONFIG_NAME} '
            f'--output_root {DATA_ROOT}'
        )
    b = np.load(path, allow_pickle=True)
    keys = list(b.keys())
    if 'stimulus_input_true' not in keys or 'stimulus_input_pred' not in keys:
        sys.exit(
            'ERROR: rollout_bundle.npz does not contain stimulus_input_true /\n'
            '       stimulus_input_pred — re-run `-o test` with the patched\n'
            '       graph_tester.py to regenerate:\n'
            f'    python GNN_Main.py -o test {CONFIG_NAME} best {CONFIG_NAME} '
            f'--output_root {DATA_ROOT}'
        )
    return b


def load_positions():
    _set_data_root(DATA_ROOT)
    gdata = graphs_data_path('fly', CONFIG_NAME)
    x_ts = load_simulation_data(
        os.path.join(gdata, 'x_list_train'),
        fields=['pos'],
    )
    return x_ts.pos.numpy().astype(np.float32)


# ── hex panel helpers ────────────────────────────────────────────────────────
def _zscore(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def _draw_hex(ax, xy, values, xlim, ylim, vmin=HEX_VMIN, vmax=HEX_VMAX):
    ax.scatter(xy[:, 0], xy[:, 1], c=values,
               s=HEX_MARKER_S, marker='h',
               cmap=CMAP, vmin=vmin, vmax=vmax,
               edgecolors=HEX_EDGE_C, linewidths=HEX_EDGE_W, alpha=1.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── trace panel helper (voltage or stimulus) ─────────────────────────────────
def draw_trace_panel(ax, ax_res, true_w, pred_w, labels, step_v, time_ms,
                     pearson_r, title, show_xlabel):
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
            f'{title} — Pearson $r$ = {r_txt} ($8\\,000$ test frames)',
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


# ── pearson helpers ──────────────────────────────────────────────────────────
def _parse_rollout_log(path):
    """Return dict with keys {voltage, stimulus} of global Pearson r values
    written by graph_tester — averaged across all neurons over all frames.
    Same convention as fig_davis_youtube_rollout_noise_05.py."""
    import re
    out = {'voltage': None, 'stimulus': None}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        txt = f.read()
    m = re.search(r'Pearson r:\s*([-\d.]+)', txt)
    if m:
        out['voltage'] = float(m.group(1))
    m = re.search(r'stimuli_r:\s*([-\d.]+)', txt)
    if m:
        out['stimulus'] = float(m.group(1))
    return out


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    bundle = load_bundle(BUNDLE_PATH)
    activity_true = bundle['activity_true']                         # (N, T)
    activity_pred = bundle['activity_pred']
    stim_in_true  = bundle['stimulus_input_true']                   # (T, n_input)
    # graph_tester fits a global linear correction ax+b so the INR output
    # shares the scale/offset of the true stimulus — that is the array used
    # to compute the reported stimuli_r. Prefer the corrected array; fall
    # back to the raw prediction if the corrected one isn't in the bundle.
    if 'stimulus_input_pred_corrected' in bundle.files:
        stim_in_pred = bundle['stimulus_input_pred_corrected']
    else:
        stim_in_pred = bundle['stimulus_input_pred']
    type_ids      = bundle['type_ids'].astype(int)
    type_names    = list(bundle['type_names'])
    index_to_name = {i: type_names[i] for i in range(len(type_names))}

    print(f'bundle: activity_true={activity_true.shape}  stim_in={stim_in_true.shape}')

    pos = load_positions()
    n_input = stim_in_true.shape[1]
    pos_input = pos[:n_input]
    # shared xlim/ylim for hex panels
    _pad_x = (pos_input[:, 0].max() - pos_input[:, 0].min()) * 0.03
    _pad_y = (pos_input[:, 1].max() - pos_input[:, 1].min()) * 0.03
    HEX_XLIM = (pos_input[:, 0].min() - _pad_x, pos_input[:, 0].max() + _pad_x)
    HEX_YLIM = (pos_input[:, 1].min() - _pad_y, pos_input[:, 1].max() + _pad_y)

    # 11 consecutive hex snapshots, 80 ms apart.
    T = stim_in_true.shape[0]
    hex_step_frames = int(round(80.0 / DT_MS))
    t0 = min(TRACE_START, T - 1 - hex_step_frames * (SERIES_COLS - 1))
    series_frames = np.array(
        [t0 + k * hex_step_frames for k in range(SERIES_COLS)], dtype=int
    )

    # pick voltage traces
    neuron_idx, labels_v = [], []
    for t in SELECTED_TYPES:
        ids = np.where(type_ids == t)[0]
        if len(ids) > 0:
            neuron_idx.append(int(ids[0]))
            labels_v.append(index_to_name.get(t, f'Type{t}'))

    true_v = activity_true[neuron_idx, TRACE_START:TRACE_END].astype(np.float32)
    pred_v = activity_pred[neuron_idx, TRACE_START:TRACE_END].astype(np.float32)

    # pick stimulus traces — evenly spaced photoreceptors
    stim_idx = np.linspace(0, n_input - 1, N_STIM_TRACES, dtype=int)
    labels_s = [f'R{(i % 8) + 1}' for i in range(N_STIM_TRACES)]
    true_s = stim_in_true[TRACE_START:TRACE_END, stim_idx].T.astype(np.float32)
    pred_s = stim_in_pred[TRACE_START:TRACE_END, stim_idx].T.astype(np.float32)

    n_frames = true_v.shape[1]
    time_ms = np.arange(n_frames) * DT_MS + TRACE_START * DT_MS

    step_v_volt = max(0.5, 3.0 * float(np.std(true_v)))
    step_v_stim = max(0.5, 3.0 * float(np.std(true_s)))

    # Pearson r reported in the panel titles is parsed from the same
    # results_rollout.log that graph_tester writes — values over *all*
    # neurons / *all* 8 000 frames (matches fig_davis_youtube_rollout_*).
    _rlog = os.path.join(BASE_DIR, 'results_rollout.log')
    _rs = _parse_rollout_log(_rlog)
    r_volt = _rs.get('voltage')
    r_stim = _rs.get('stimulus')
    print(f'  voltage Pearson r = {r_volt:.3f}')
    print(f'  stimulus Pearson r = {r_stim:.3f}')

    # ── figure layout ────────────────────────────────────────────────────────
    # constrained_layout ignores GridSpec hspace — use manual layout so the
    # inter-row blanks actually stick.
    fig = plt.figure(figsize=(18, 18), dpi=300)
    outer = mgs.GridSpec(3, 1, figure=fig,
                         height_ratios=[2.4, 3.0, 3.0],
                         left=0.06, right=0.92, top=0.97, bottom=0.05,
                         hspace=0.18)

    # (a) 3 x 11 hexagons: GT / learned / residual (learned - GT, in z-score
    # space). Modest hspace so the row labels don't overlap the hex above.
    gs_a = mgs.GridSpecFromSubplotSpec(3, SERIES_COLS, subplot_spec=outer[0],
                                        wspace=0.05, hspace=0.35)
    axes_hex_top = []
    axes_hex_mid = []
    axes_hex_res = []
    for col, t in enumerate(series_frames):
        # z-score the two frames independently (per-panel normalisation to
        # spread the colour scale) before taking the residual.
        vals_gt = _zscore(stim_in_true[t, :])
        vals_pd = _zscore(stim_in_pred[t, :])
        vals_rs = vals_pd - vals_gt

        ax_gt = fig.add_subplot(gs_a[0, col])
        _draw_hex(ax_gt, pos_input, vals_gt, HEX_XLIM, HEX_YLIM)
        ax_gt.set_title(f't = {int(t * DT_MS)} ms',
                        fontsize=int(20 * _S), pad=2)
        axes_hex_top.append(ax_gt)

        ax_pd = fig.add_subplot(gs_a[1, col])
        _draw_hex(ax_pd, pos_input, vals_pd, HEX_XLIM, HEX_YLIM)
        axes_hex_mid.append(ax_pd)

        ax_rs = fig.add_subplot(gs_a[2, col])
        _draw_hex(ax_rs, pos_input, vals_rs, HEX_XLIM, HEX_YLIM)
        axes_hex_res.append(ax_rs)
    # Row labels placed above the first hexagon of each row (left-aligned
    # with that first panel), so the column of hexagons starts flush at the
    # left edge of the figure.
    axes_hex_top[0].text(0.35, 1.28, 'ground truth visual stimulus',
                         transform=axes_hex_top[0].transAxes,
                         va='bottom', ha='left', fontsize=int(0.75 * FS_LABEL))
    axes_hex_mid[0].text(0.35, 1.10, 'learned visual stimulus',
                         transform=axes_hex_mid[0].transAxes,
                         va='bottom', ha='left', fontsize=int(0.75 * FS_LABEL))
    axes_hex_res[0].text(0.35, 1.10, 'residual (learned $-$ ground truth)',
                         transform=axes_hex_res[0].transAxes,
                         va='bottom', ha='left', fontsize=int(0.75 * FS_LABEL))

    # Single colorbar on the right of the hex block, anchored to span all
    # three rows (same z-score scale).
    fig.canvas.draw()
    _norm = _mcolors.Normalize(vmin=HEX_VMIN, vmax=HEX_VMAX)
    _sm = _mcm.ScalarMappable(norm=_norm, cmap=CMAP)
    _top_pos = axes_hex_top[-1].get_position()
    _res_pos = axes_hex_res[-1].get_position()
    # Center a short colorbar over the middle row of the hex block.
    _cbar_h = (_top_pos.y1 - _res_pos.y0) * 0.45
    _cbar_y0 = (_top_pos.y1 + _res_pos.y0) / 2.0 - _cbar_h / 2.0
    _cax = fig.add_axes([
        _top_pos.x1 + 0.010,
        _cbar_y0,
        0.010,
        _cbar_h,
    ])
    _cbar = fig.colorbar(_sm, cax=_cax)
    _cbar.set_label('voltage (z-score)', fontsize=int(0.75 * FS_LABEL))
    _cbar.ax.tick_params(labelsize=FS_TICK)

    # (b) stimulus rollout — placed second (input before output).
    gs_b = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                        width_ratios=[4, 1], wspace=0.02)
    ax_b   = fig.add_subplot(gs_b[0, 0])
    ax_b_r = fig.add_subplot(gs_b[0, 1], sharey=ax_b)
    draw_trace_panel(ax_b, ax_b_r, true_s, pred_s, labels_s,
                     step_v_stim, time_ms,
                     pearson_r=r_stim,
                     title='rollout stimulus (12 photoreceptors)',
                     show_xlabel=False)

    # (c) voltage rollout — placed third (cell-type voltage derived from stim).
    gs_c = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2],
                                        width_ratios=[4, 1], wspace=0.02)
    ax_c   = fig.add_subplot(gs_c[0, 0])
    ax_c_r = fig.add_subplot(gs_c[0, 1], sharey=ax_c)
    draw_trace_panel(ax_c, ax_c_r, true_v, pred_v, labels_v,
                     step_v_volt, time_ms,
                     pearson_r=r_volt,
                     title='rollout voltage (12 types)',
                     show_xlabel=True)

    # Per-panel legend — (b) uses INR (stimulus) prediction label, (c) uses
    # GNN (voltage) prediction label. Anchored to the right side of each
    # residual column so the legend sits outside the plot area.
    from matplotlib.lines import Line2D
    _handles_b = [
        Line2D([0], [0], color=COLOR_GT,   lw=LW_GT,   label='ground truth'),
        Line2D([0], [0], color=COLOR_PRED, lw=LW_PRED, label='INR rollout prediction'),
        Line2D([0], [0], color=COLOR_RES,  lw=RES_LW,  label='residual (pred $-$ true)'),
    ]
    _handles_c = [
        Line2D([0], [0], color=COLOR_GT,   lw=LW_GT,   label='ground truth'),
        Line2D([0], [0], color=COLOR_PRED, lw=LW_PRED, label='GNN rollout prediction'),
        Line2D([0], [0], color=COLOR_RES,  lw=RES_LW,  label='residual (pred $-$ true)'),
    ]
    for _ax_res, _h in ((ax_b_r, _handles_b), (ax_c_r, _handles_c)):
        _ax_res.legend(handles=_h, loc='upper left',
                       bbox_to_anchor=(1.04, 1.0), ncol=1, handlelength=2,
                       fontsize=int(1.3 * FS_LEGEND), frameon=False,
                       borderaxespad=0.0)

    # panel labels a / b / c
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    anchors = [axes_hex_top[0], ax_b, ax_c]
    for ax_anchor, lbl in zip(anchors, ['a)', 'b)', 'c)']):
        bb = ax_anchor.get_tightbbox(renderer)
        x0, y1 = inv.transform((bb.x0, bb.y1))
        fig.text(x0, y1, lbl, fontsize=PANEL_LBL, fontweight='bold',
                 va='bottom', ha='left', color='black', transform=fig.transFigure)

    out_base = os.path.join(_SCRIPT_DIR, 'fig_stim_rollout_inr')
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
