"""
Figure: simulated neural activity at three intrinsic-noise levels.

Layout (2 rows × 3 columns)
---------------------------
  Row 1 (a–c): type-mean voltage heatmap — 65 cell types × 7 999 frames,
               z-scored per type, sorted by anatomical order.
               Conveys the full scale of the training dataset.
  Row 2 (d–f): stacked voltage traces for 6 representative cell types
               (R1 · L1 · L2 · Mi1 · T4a · T5a) over a 500-frame window.
               Conveys the noise effect on individual neuronal dynamics.

Columns left → right: σ = 0 (noise-free), σ = 0.05, σ = 0.5.
step_v for traces is fixed from σ = 0 so noise is visually comparable.

Input
-----
Three rollout_bundle.npz produced by GNN_Main -o test (graph_tester.py).

Usage
-----
    conda run -n neural-graph-linux python figures/fig_simulations.py

Output
------
    figures/fig_simulations.{png,pdf,jpg}
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── project imports ───────────────────────────────────────────────────────────
REPO = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(REPO, 'src'))
from connectome_gnn.metrics import ANATOMICAL_ORDER

# ── font style (INSTRUCTIONS.md §style) ──────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'text.usetex': False,
    'mathtext.fontset': 'dejavusans',
})

# ── font sizes (col_width ≈ 7 in / 10 → _S = 0.52, INSTRUCTIONS.md §font) ───
_S        = 0.52
FS_LABEL  = int(48 * _S)    # axis labels
FS_TICK   = int(24 * _S)    # tick labels
FS_ANNOT  = int(28 * _S)    # type-name annotations in trace panel
FS_TITLE  = 17              # panel subtitle (fixed, INSTRUCTIONS.md)
PANEL_LBL = 20              # a)–f) (fixed, never scaled)
FS_CBAR   = int(22 * _S)    # colorbar label / ticks

# ── data ──────────────────────────────────────────────────────────────────────
DATA_ROOT = '/groups/saalfeld/home/allierc/GraphData'
CONFIGS = [
    ('flyvis_noise_free', DATA_ROOT, r'$\sigma = 0$'),
    ('flyvis_noise_005',  DATA_ROOT, r'$\sigma = 0.05$'),
    ('flyvis_noise_05',   DATA_ROOT, r'$\sigma = 0.5$'),
]

# ── selected types for trace panels ──────────────────────────────────────────
# Covers the canonical visual pathway: photoreceptors → lamina → medulla
# → direction-selective. Type indices from type_names list (alphabetical order).
#   R1=23, L1=5, L2=6, Mi1=12, T4a=35, T5a=39
SELECTED_TYPES = [23, 5, 6, 12, 35, 39]
TRACE_START    = 0
TRACE_END      = 500

# ── heatmap style ────────────────────────────────────────────────────────────
CMAP   = 'RdBu_r'
VLIM   = 2.0      # ±2 σ clipping
COLOR  = '#1a5276' # single trace color — consistent across noise levels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bundle_path(config_name, data_root):
    return os.path.join(data_root, 'log', 'fly', config_name,
                        'results', 'rollout_bundle.npz')


def _type_heatmap(activity_true, type_ids, n_types):
    """Mean voltage per type, z-scored across time. Returns (n_types, n_frames)."""
    n_frames = activity_true.shape[1]
    heat = np.zeros((n_types, n_frames), dtype=np.float32)
    for t in range(n_types):
        mask = type_ids == t
        if mask.sum() > 0:
            heat[t] = activity_true[mask].mean(axis=0)
    mu  = heat.mean(axis=1, keepdims=True)
    std = heat.std(axis=1,  keepdims=True)
    return (heat - mu) / (std + 1e-6)


def _selected_traces(activity_true, type_ids):
    """Return dict type_idx → (n_frames,) trace for one neuron per selected type."""
    out = {}
    for t in SELECTED_TYPES:
        rows = np.where(type_ids == t)[0]
        if len(rows):
            out[t] = activity_true[rows[0], TRACE_START:TRACE_END]
    return out


# ---------------------------------------------------------------------------
# Load all data (one bundle at a time to limit peak memory)
# ---------------------------------------------------------------------------
# anatomical sort order (filter None and out-of-range indices)
n_types_ref = 65
anat_order  = [i for i in ANATOMICAL_ORDER if i is not None and i < n_types_ref]

heatmaps     = []
trace_sets   = []
type_names_ref = None

for config_name, data_root, _ in CONFIGS:
    path = _bundle_path(config_name, data_root)
    print(f'loading {config_name} ...')
    b = np.load(path, allow_pickle=True)
    act   = b['activity_true']                # (n_neurons, n_frames)
    tids  = b['type_ids'].astype(int)
    tname = list(b['type_names'])
    n_types = len(tname)
    if type_names_ref is None:
        type_names_ref = tname

    hz = _type_heatmap(act, tids, n_types)    # (n_types, n_frames)
    heatmaps.append(hz[anat_order])           # anatomically sorted

    trace_sets.append(_selected_traces(act, tids))
    del act                                   # free 400 MB

sorted_names = [type_names_ref[i] for i in anat_order]

# step_v fixed from σ=0 data so noise is visually comparable across columns
free_traces = np.array(list(trace_sets[0].values()))   # (6, 500)
activity_std = np.std(free_traces)
step_v = max(0.5, 3.0 * activity_std)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    2, 3, figsize=(21, 9.5), dpi=300,
    constrained_layout=True,
    gridspec_kw={'height_ratios': [2, 3]},
)

n_types_plot = len(anat_order)
n_frames     = heatmaps[0].shape[1]

last_im = None
for col, (heat_z, traces, (_, _, sigma_lbl)) in enumerate(
        zip(heatmaps, trace_sets, CONFIGS)):

    # ── row 0: heatmap ────────────────────────────────────────────────────────
    ax_h = axes[0, col]
    im   = ax_h.imshow(heat_z, aspect='auto', interpolation='nearest',
                       cmap=CMAP, vmin=-VLIM, vmax=VLIM, origin='upper')
    last_im = im
    ax_h.set_title(sigma_lbl, fontsize=FS_TITLE, pad=4)
    ax_h.set_xticks([0, n_frames // 2, n_frames - 1])
    ax_h.set_xticklabels(['0', str(n_frames // 2), str(n_frames - 1)],
                          fontsize=FS_TICK)
    ax_h.set_xlabel('frame', fontsize=FS_LABEL)
    if col == 0:
        ax_h.set_yticks(range(n_types_plot))
        ax_h.set_yticklabels(sorted_names, fontsize=6)
        ax_h.set_ylabel('cell type', fontsize=FS_LABEL)
    else:
        ax_h.set_yticks([])

    # ── row 1: traces ─────────────────────────────────────────────────────────
    ax_t  = axes[1, col]
    row   = 0
    for t_idx in SELECTED_TYPES:
        if t_idx not in traces:
            continue
        trace = traces[t_idx]
        bl    = trace.mean()
        ax_t.plot(trace - bl + row * step_v,
                  lw=0.9, color=COLOR, alpha=0.9)
        ax_t.text(-(TRACE_END - TRACE_START) * 0.025, row * step_v,
                  type_names_ref[t_idx],
                  fontsize=FS_ANNOT, va='bottom', ha='right', color='black')
        row += 1

    n_rows   = row
    n_tframes = TRACE_END - TRACE_START
    ax_t.set_ylim([-step_v, (n_rows - 1) * step_v + step_v])
    ax_t.set_yticks([])
    ax_t.set_xticks([0, n_tframes // 2, n_tframes])
    ax_t.set_xticklabels([TRACE_START,
                           (TRACE_START + TRACE_END) // 2,
                           TRACE_END], fontsize=FS_TICK)
    ax_t.set_xlabel('frame', fontsize=FS_LABEL)
    ax_t.set_xlim([-(TRACE_END - TRACE_START) * 0.08,
                    (TRACE_END - TRACE_START) * 1.05])
    ax_t.set_title(sigma_lbl, fontsize=FS_TITLE, pad=4)
    ax_t.spines['top'].set_visible(False)
    ax_t.spines['right'].set_visible(False)
    ax_t.spines['left'].set_visible(False)
    if col == 0:
        ax_t.set_ylabel('voltage (a.u.)', fontsize=FS_LABEL)

# ── shared colorbar (right of top-right heatmap) ─────────────────────────────
cbar = fig.colorbar(last_im, ax=axes[0, 2], shrink=0.95, pad=0.02)
cbar.set_label('z-scored voltage', fontsize=FS_CBAR)
cbar.ax.tick_params(labelsize=FS_TICK)

# ── panel labels at outer box top-left, all at same y (INSTRUCTIONS.md) ──────
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv      = fig.transFigure.inverted()
all_axes = list(axes[0]) + list(axes[1])
bboxes   = [ax.get_tightbbox(renderer) for ax in all_axes]
y1_max   = max(inv.transform((bb.x0, bb.y1))[1] for bb in bboxes)
for bb, lbl in zip(bboxes, ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']):
    x0 = inv.transform((bb.x0, bb.y1))[0]
    fig.text(x0, y1_max, lbl, fontsize=PANEL_LBL, fontweight='bold',
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
