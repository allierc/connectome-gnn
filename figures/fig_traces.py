"""
Figure: rollout traces — GNN voltage dynamics and (optionally) INR visual stimulus.

Layout
------
  If bundle contains INR traces (GNN+INR model):
    2-row figure
      top row    — GNN rollout: selected cell-type voltage traces
      bottom row — INR: learned visual-stimulus traces vs ground truth

  Otherwise:
    1-row figure with GNN traces only.

Input
-----
  rollout_bundle.npz  produced by GNN_Main -o test (graph_tester.py)

  Required keys:
    activity_true  (n_neurons, n_frames)
    activity_pred  (n_neurons, n_frames)
    stimulus       (n_neurons, n_frames)
    type_ids       (n_neurons,)
    type_names     (n_types,)  dtype=object

  Optional keys (GNN+INR only):
    inr_true       (n_inr, n_frames)
    inr_pred_corr  (n_inr, n_frames)  linearly-corrected INR predictions
    inr_global_ids (n_inr,)           global neuron indices
    inr_type       ()                 string e.g. 'siren_t'

Usage
-----
    conda run -n neural-graph-linux python figures/fig_traces.py \\
        --bundle log/remote/fly/flyvis_noise_005/results/rollout_bundle.npz

    # GNN+INR model:
    conda run -n neural-graph-linux python figures/fig_traces.py \\
        --bundle log/remote/fly/flyvis_noise_005_hidden_010/results/rollout_bundle.npz

Output
------
    figures/fig_traces.png
    figures/fig_traces.pdf
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── font style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'text.usetex': False,
    'mathtext.fontset': 'dejavusans',
})

# ── font sizes ────────────────────────────────────────────────────────────────
FS_LABEL  = 14
FS_TICK   = 11
FS_ANNOT  = 10   # type-name labels on the left
FS_LEGEND = 11
PANEL_LBL = 14
LW_GT     = 2.0
LW_PRED   = 0.8

# ── selected cell types for flyvis (same as graph_tester curated list) ────────
SELECTED_TYPES = [55, 15, 43, 39, 35, 31, 23, 19, 12, 5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_trace_panel(ax, true_arr, pred_arr, labels,
                       start_frame=0, end_frame=None,
                       gt_color='#66cc66', pred_color='black',
                       stim_arr=None, gt_label='ground truth', pred_label='prediction'):
    """Draw stacked voltage/stimulus traces on ax.

    true_arr, pred_arr : (n_traces, n_frames)
    labels             : list[str], one per trace (empty string → no label)
    Returns step_v used for spacing.
    """
    n_traces, n_frames_total = true_arr.shape
    if end_frame is None:
        end_frame = n_frames_total
    true_s = true_arr[:, start_frame:end_frame]
    pred_s = pred_arr[:, start_frame:end_frame]
    n_frames = true_s.shape[1]

    activity_std = np.std(true_s)
    step_v = max(0.5, 3.0 * activity_std) if activity_std > 0 else 2.5

    baselines = np.mean(true_s, axis=1)

    for i in range(n_traces):
        bl = baselines[i]
        ax.plot(true_s[i] - bl + i * step_v,
                lw=LW_GT, color=gt_color, alpha=0.9,
                label=gt_label if i == 0 else None)

    for i in range(n_traces):
        bl = baselines[i]
        ax.plot(pred_s[i] - bl + i * step_v,
                lw=LW_PRED, color=pred_color, alpha=0.9,
                label=pred_label if i == 0 else None)

    for i in range(n_traces):
        if labels[i]:
            ax.text(-n_frames * 0.025, i * step_v, labels[i],
                    fontsize=FS_ANNOT, va='bottom', ha='right', color='black')

    ax.set_ylim([-step_v, (n_traces - 1) * step_v + step_v])
    ax.set_yticks([])
    ax.set_xticks([0, n_frames // 2, n_frames])
    ax.set_xticklabels([start_frame, (start_frame + end_frame) // 2, end_frame],
                       fontsize=FS_TICK)
    ax.set_xlabel('frame', fontsize=FS_LABEL)
    ax.set_xlim([-n_frames * 0.03, n_frames * 1.05])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0),
              bbox_transform=ax.transAxes, fontsize=FS_LEGEND, frameon=False)
    return step_v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True,
                        help='Path to rollout_bundle.npz')
    parser.add_argument('--out', default=None,
                        help='Output path prefix (default: figures/fig_traces)')
    args = parser.parse_args()

    bundle_path = args.bundle
    if not os.path.exists(bundle_path):
        sys.exit(f'Bundle not found: {bundle_path}')

    bundle = np.load(bundle_path, allow_pickle=True)

    activity_true = bundle['activity_true']   # (n_neurons, n_frames)
    activity_pred = bundle['activity_pred']
    type_ids      = bundle['type_ids'].astype(int)
    type_names    = list(bundle['type_names'])
    n_types       = len(type_names)
    index_to_name = {i: type_names[i] for i in range(n_types)}

    has_inr = 'inr_true' in bundle.files

    # ── select GNN traces ─────────────────────────────────────────────────────
    if n_types > 10:
        sel_types = [t for t in SELECTED_TYPES if t < n_types]
    else:
        sel_types = list(range(n_types))

    neuron_indices = []
    neuron_labels  = []
    for stype in sel_types:
        idxs = np.where(type_ids == stype)[0]
        if len(idxs) > 0:
            neuron_indices.append(idxs[0])
            neuron_labels.append(index_to_name.get(stype, f'Type{stype}'))

    gnn_true = activity_true[neuron_indices]
    gnn_pred = activity_pred[neuron_indices]

    # ── figure layout ─────────────────────────────────────────────────────────
    n_gnn = len(neuron_indices)
    if has_inr:
        inr_true      = bundle['inr_true']
        inr_pred_corr = bundle['inr_pred_corr']
        inr_global    = bundle['inr_global_ids'].astype(int)
        inr_type_str  = str(bundle['inr_type'])
        n_inr = len(inr_global)

        fig_h = max(6, n_gnn * 0.45 + 2) + max(4, n_inr * 0.45 + 2)
        fig, (ax_gnn, ax_inr) = plt.subplots(
            2, 1, figsize=(15, fig_h), dpi=300,
            constrained_layout=True,
            gridspec_kw={'height_ratios': [n_gnn, n_inr]},
        )
    else:
        fig_h = max(6, n_gnn * 0.45 + 2)
        fig, ax_gnn = plt.subplots(1, 1, figsize=(15, fig_h), dpi=300,
                                    constrained_layout=True)

    # ── GNN panel ─────────────────────────────────────────────────────────────
    _build_trace_panel(ax_gnn, gnn_true, gnn_pred, neuron_labels)
    ax_gnn.set_title('GNN rollout — selected cell types', fontsize=FS_LABEL, pad=4)

    # ── INR panel ─────────────────────────────────────────────────────────────
    if has_inr:
        inr_labels = [f'n{gid}' for gid in inr_global]
        _build_trace_panel(
            ax_inr, inr_true, inr_pred_corr, inr_labels,
            gt_color='#66cc66', pred_color='black',
            gt_label='ground truth',
            pred_label=f'{inr_type_str.upper().replace("_", "-")} (corrected)',
        )
        ax_inr.set_title('INR — learned visual stimulus (hidden neurons)',
                         fontsize=FS_LABEL, pad=4)

    # ── panel labels ──────────────────────────────────────────────────────────
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    panel_axes  = [ax_gnn, ax_inr] if has_inr else [ax_gnn]
    panel_lbls  = ['a)', 'b)'] if has_inr else ['a)']
    bboxes = [ax.get_tightbbox(renderer) for ax in panel_axes]
    for bb, lbl in zip(bboxes, panel_lbls):
        x0 = inv.transform((bb.x0, bb.y1))[0]
        y1 = inv.transform((bb.x0, bb.y1))[1]
        fig.text(x0, y1, lbl, fontsize=PANEL_LBL, fontweight='bold',
                 va='bottom', ha='left', color='black', transform=fig.transFigure)

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = args.out if args.out else os.path.join(os.path.dirname(__file__))
    out_base = os.path.join(out_dir, 'fig_traces')
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_base}.png')
    print(f'Saved: {out_base}.pdf')


if __name__ == '__main__':
    main()
