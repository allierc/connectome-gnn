"""Appendix figure: GMM clustering accuracy on different feature concatenations.

Single 4-row × 2-col composite for one model
(`flyvis_noise_005_blank50_unified_cv00`):

    row 0 (a, b)  — learned 2-D embedding `model.a`, scattered directly:
                    a coloured by GT cell type, b coloured by GMM cluster
                    label (n_components = 100, on `a` alone).
    row 1 (c, d)  — UMAP of (τ, V_rest)  2-D feature vector,
                    c on the TRUE (gt_τ, gt_V_rest), d on the LEARNED.
    row 2 (e, f)  — UMAP of (τ, V_rest, 8 W-stats) 10-D vector,
                    e TRUE, f LEARNED. (Same set of features known_ode uses.)
    row 3 (g, h)  — LEARNED-only: g = 10-D (τ, V_rest, W-stats),
                    h = 12-D (a, τ, V_rest, W-stats). Shows what the
                    2-D learned embedding `a` adds to the ODE+W vector.

Each panel reports the GMM accuracy in the top-left, fontsize matching
the R²/slope text on the parameter scatters. Same Janne style as
`fig_gnn_params_3col_noise_comparison.py` (FS_LABEL=8, FS_TICK=6, etc.).

Inputs (loaded directly — no fresh `--replot` required):
    <log_dir>/results/panels_noise_005_blank50_cv00.npz
        → a, tau_learned, V_rest_learned, W_learned, type_ids
    <data_root>/graphs_data/fly/flyvis_noise_005_blank50_cv00/ode_params.pt
        → edge_index (for the per-neuron W stats), tau_i, V_i_rest, W

Per-neuron W-stats (8 cols, mirrors GNN_PlotFigure.plot_synaptic):
    in  → mean, std, min, max
    out → mean, std, min, max

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/fig_clustering_appendix.py

Output
------
    figures/fig_clustering_appendix.{pdf,png}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Data root      : /groups/saalfeld/home/allierc/GraphData
# Configs        : <DATA_ROOT>/config/fly/flyvis_noise_005_blank50_unified_cv00.yaml
# Training data  : <DATA_ROOT>/graphs_data/fly/flyvis_noise_005_blank50_cv00/x_list_train/
#                  <DATA_ROOT>/graphs_data/fly/flyvis_noise_005_blank50_cv00/ode_params.pt
# Trained models : <DATA_ROOT>/log/fly/flyvis_noise_005_blank50_unified_cv00/models/best_model_with_0_graphs_0.pt
# Cached panels  : <DATA_ROOT>/log/fly/flyvis_noise_005_blank50_unified_cv00/results/panels_noise_005_blank50_cv00.npz
# Output         : figures/fig_clustering_appendix.{pdf,png}
# ─────────────────────────────────────────────────────────────────────────────

import os
import string
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'janne.matplotlibrc'))

import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture


# ── repo + imports ──────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_ROOT = '/groups/saalfeld/home/allierc/GraphData'
for _p in (os.path.join(REPO_ROOT, 'src'), REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from connectome_gnn.sparsify import clustering_gmm  # noqa: E402


# ── inputs ──────────────────────────────────────────────────────────────────
MODEL_BASE = 'flyvis_noise_005_blank50_unified_cv00'
DATASET    = 'flyvis_noise_005_blank50_cv00'
CONFIG_INDICES = 'noise_005_blank50_cv00'

LOG_DIR     = f'{DATA_ROOT}/log/fly/{MODEL_BASE}'
RESULTS_DIR = f'{LOG_DIR}/results'
PANELS_NPZ  = f'{RESULTS_DIR}/panels_{CONFIG_INDICES}.npz'
ODE_PARAMS  = f'{DATA_ROOT}/graphs_data/fly/{DATASET}/ode_params.pt'

# GMM/UMAP knobs — match plot_synaptic so accuracies in this figure agree
# with the values printed during data_plot.
N_GMM_COMPONENTS = 100
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
UMAP_RANDOM      = 42


# ── style ────────────────────────────────────────────────────────────────────
# This appendix figure uses larger fonts than the multi-condition figs
# (which sit at FS_LABEL=8 per Janne convention) because each panel is ~4 in
# wide here vs ~1.18 in there — at the same point size the labels would
# look ~3× smaller relative to the panel content. Doubling the FS_*
# values keeps the visual ratio comparable.
FS_LABEL    = 16   # axis labels
FS_TITLE    = 11   # per-panel title (smaller; sits next to the panel letter)
FS_TICK     = 11
FS_ANNOT    = 11   # GMM accuracy + cluster count line (highlighted)
FS_ANNOT_LO = 11   # ARI / NMI line — same size as GMM line
PANEL_LBL   = 16   # bold panel letters

# 2 rows × 2 cols of square panels. Source panels render directly
# (scatters, no PNG round-trip) so 300 dpi PNG is sharp enough.
# 2 panel cols × ~3.7 in + margins → ~8 in wide; 2 rows × ~3.7 in +
# title strip → ~8 in tall. Panel aspect forced to 1:1 via
# ax.set_box_aspect(1.0) inside _scatter().
FIG_W_IN = 8.0
FIG_H_IN = 8.0


# ── helpers ─────────────────────────────────────────────────────────────────

def _connectivity_stats(w, src, dst, n):
    """Per-neuron in/out edge-weight summary (mean, std, min, max).

    Vectorised copy of GNN_PlotFigure.plot_synaptic's helper so feature
    vectors here are bit-for-bit comparable.
    """
    in_count  = np.bincount(dst, minlength=n).astype(np.float64)
    out_count = np.bincount(src, minlength=n).astype(np.float64)
    in_sum    = np.bincount(dst, weights=w,       minlength=n)
    out_sum   = np.bincount(src, weights=w,       minlength=n)
    in_sq     = np.bincount(dst, weights=w ** 2,  minlength=n)
    out_sq    = np.bincount(src, weights=w ** 2,  minlength=n)
    safe_in   = np.where(in_count  > 0, in_count,  1)
    safe_out  = np.where(out_count > 0, out_count, 1)
    in_mean   = in_sum  / safe_in
    out_mean  = out_sum / safe_out
    in_std    = np.sqrt(np.maximum(in_sq  / safe_in  - in_mean  ** 2, 0))
    out_std   = np.sqrt(np.maximum(out_sq / safe_out - out_mean ** 2, 0))
    in_max  = np.full(n, -np.inf); np.maximum.at(in_max,  dst, w)
    in_min  = np.full(n,  np.inf); np.minimum.at(in_min,  dst, w)
    out_max = np.full(n, -np.inf); np.maximum.at(out_max, src, w)
    out_min = np.full(n,  np.inf); np.minimum.at(out_min, src, w)
    for arr, c in [(in_mean, in_count), (in_std, in_count),
                   (in_min, in_count),  (in_max, in_count),
                   (out_mean, out_count), (out_std, out_count),
                   (out_min, out_count),  (out_max, out_count)]:
        arr[c == 0] = 0
    return (in_mean, in_std, out_mean, out_std,
            in_min, in_max, out_min, out_max)


def _gmm_run(features, type_list, n_components=N_GMM_COMPONENTS):
    """Return (accuracy, ARI, NMI, n_found, labels).

    `n_found` is the count of distinct GMM components that actually got
    assigned at least one neuron — can be <= `n_components` when GMM
    collapses some Gaussians to empty clusters. Both metrics (via
    clustering_gmm) and labels (via fit_predict) come from the same
    n_components / random seed so the two are consistent.
    """
    res = clustering_gmm(features, type_list, n_components=n_components)
    labels = None
    for cov_type in ('full', 'diag', 'spherical'):
        try:
            labels = GaussianMixture(
                n_components=n_components, random_state=UMAP_RANDOM,
                reg_covar=1e-3, covariance_type=cov_type,
            ).fit_predict(features)
            break
        except Exception:
            continue
    if labels is None:
        labels = np.zeros(len(features), dtype=int)
    n_found = int(np.unique(labels).size)
    return (float(res['accuracy']), float(res['ari']), float(res['nmi']),
            n_found, labels)


def _umap2(features):
    reducer = umap.UMAP(n_components=2, random_state=UMAP_RANDOM,
                        n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST)
    return reducer.fit_transform(features)


# ── data load ───────────────────────────────────────────────────────────────

def load_inputs():
    print(f'loading {PANELS_NPZ}')
    p = np.load(PANELS_NPZ, allow_pickle=True)
    print(f'loading {ODE_PARAMS}')
    op = torch.load(ODE_PARAMS, weights_only=False, map_location='cpu')

    a_learned    = np.asarray(p['a']).astype(np.float32)         # (N, 2)
    tau_learned  = np.asarray(p['tau_learned']).astype(np.float32)
    V_learned    = np.asarray(p['V_rest_learned']).astype(np.float32)
    W_learned    = np.asarray(p['W_learned']).astype(np.float32)  # per-edge
    type_list    = np.asarray(p['type_ids']).astype(np.int64)

    edge_index = op['edge_index'].cpu().numpy()                  # (2, E)
    src, dst   = edge_index[0], edge_index[1]
    tau_true   = op['tau_i'].cpu().numpy().astype(np.float32)
    V_true     = op['V_i_rest'].cpu().numpy().astype(np.float32)
    W_true     = op['W'].cpu().numpy().astype(np.float32)        # per-edge
    n_neurons  = type_list.shape[0]

    print(f'  N = {n_neurons} neurons, |E| = {W_learned.shape[0]} edges')

    print('  computing per-neuron W-stats (true)...')
    w_t = _connectivity_stats(W_true, src, dst, n_neurons)
    print('  computing per-neuron W-stats (learned)...')
    w_l = _connectivity_stats(W_learned, src, dst, n_neurons)

    return {
        'a_learned': a_learned,
        'tau_true': tau_true,         'tau_learned': tau_learned,
        'V_true':   V_true,           'V_learned':   V_learned,
        'W_stats_true':    np.column_stack(w_t),    # (N, 8)
        'W_stats_learned': np.column_stack(w_l),    # (N, 8)
        'type_list': type_list,
        'n_neurons': n_neurons,
    }


# ── feature subsets ────────────────────────────────────────────────────────

def build_subsets(d):
    """Return ordered list of (label, feature_matrix, n_dim) tuples."""
    tau_t,  tau_l  = d['tau_true'][:, None],  d['tau_learned'][:, None]
    V_t,    V_l    = d['V_true'][:, None],    d['V_learned'][:, None]
    Ws_t,   Ws_l   = d['W_stats_true'],       d['W_stats_learned']
    a              = d['a_learned']           # (N, 2)

    return {
        'a':                     a,                                                 # 2-D learned
        'ode_true':              np.column_stack([tau_t, V_t]),                    # 2-D true
        'ode_learned':           np.column_stack([tau_l, V_l]),                    # 2-D learned
        'ode_W_true':            np.column_stack([tau_t, V_t, Ws_t]),              # 10-D true
        'ode_W_learned':         np.column_stack([tau_l, V_l, Ws_l]),              # 10-D learned
        'a_ode_W_learned':       np.column_stack([a, tau_l, V_l, Ws_l]),           # 12-D learned
    }


# ── figure ──────────────────────────────────────────────────────────────────

def _scatter(ax, xy, colors, cmap, *, s=4, alpha=0.65):
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, cmap=cmap,
               s=s, alpha=alpha, edgecolors='none')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_box_aspect(1.0)
    for spine in ax.spines.values():
        spine.set_alpha(0.5); spine.set_linewidth(0.5)


def _annot(ax, acc, ari, nmi, n_found):
    """Two-line top-left annotation:
       line 1 (FS_ANNOT): GMM acc + cluster count
       line 2 (FS_ANNOT_LO, smaller): ARI + NMI
       Extra y-gap between the two lines acts as a blank separator."""
    ax.text(0.04, 0.96, f'GMM accuracy = {acc:.2f}  (k={n_found}/{N_GMM_COMPONENTS})',
            transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=FS_ANNOT, color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.7))
    ax.text(0.04, 0.88, f'ARI = {ari:.2f}, NMI = {nmi:.2f}',
            transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=FS_ANNOT_LO, color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.7))


def assemble(d, subsets, out_base):
    """Build the 4 × 2 composite figure."""
    # 65 cell-type palette: 65 distinct hues at uniform saturation/lightness.
    # Replaces the previous Set3*6 (light pastels including a near-white
    # yellow that vanished on a white background, plus colour repeats every
    # 12 types). `husl` gives 65 unique, well-saturated colours.
    colors_65 = sns.color_palette("husl", 65)
    cmap_65   = ListedColormap(colors_65)

    type_list = d['type_list']

    # ── pre-compute every UMAP + accuracy + labels ──────────────────────
    print('GMM + UMAP per feature subset (this can take ~1 min)...')
    pre = {}
    for key in ('a',
                'ode_true', 'ode_learned',
                'ode_W_true', 'ode_W_learned',
                'a_ode_W_learned'):
        feats = subsets[key]
        n_comp = min(N_GMM_COMPONENTS, feats.shape[0] - 1)
        acc, ari, nmi, n_found, _labels = _gmm_run(
            feats, type_list, n_components=n_comp)
        # `a`, `ode_true`, `ode_learned` are visualised directly in their
        # raw 2-D feature space — for (τ, V_rest) this matches the space
        # GMM is actually fit on, so the eye and the metric agree.
        # The higher-D subsets (W-stats, augmented) get a UMAP projection.
        proj = feats if key in ('a', 'ode_true', 'ode_learned') else _umap2(feats)
        pre[key] = {'proj': proj, 'acc': acc, 'ari': ari, 'nmi': nmi,
                    'n_found': n_found}
        print(f"  {key:25s}  dim={feats.shape[1]:2d}  "
              f"acc={acc:.3f}  ARI={ari:.3f}  NMI={nmi:.3f}  "
              f"k={n_found}/{n_comp}")

    # ── render ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)
    gs  = mgs.GridSpec(2, 2, figure=fig,
                       left=0.06, right=0.98, top=0.95, bottom=0.05,
                       wspace=0.18, hspace=0.40)

    panel_axes = []
    letters = string.ascii_lowercase

    def add(row, col, *, key, colors, xlab, ylab, title, label):
        ax = fig.add_subplot(gs[row, col])
        m = pre[key]
        _scatter(ax, m['proj'], colors, cmap_65)
        _annot(ax, m['acc'], m['ari'], m['nmi'], m['n_found'])
        ax.set_xlabel(xlab, fontsize=FS_LABEL)
        ax.set_ylabel(ylab, fontsize=FS_LABEL)
        # Title rendered later in the panel-letter loop so it sits on the
        # same horizontal line as the bold letter (left-aligned, smaller).
        panel_axes.append((ax, label, title))

    # Row 0 (high-dim true vs learned, the headline comparison).
    add(0, 0, key='ode_W_true',       colors=type_list,
        xlab=r'UMAP$_1$', ylab=r'UMAP$_2$',
        title=r'true $(\tau,\,V_{rest},\,W\text{-stats})$',    label='a')
    add(0, 1, key='ode_W_learned',    colors=type_list,
        xlab=r'UMAP$_1$', ylab=r'UMAP$_2$',
        title=r'learned $(\tau,\,V_{rest},\,W\text{-stats})$', label='b')

    # Row 1 (learned `a`, then augmented combination).
    add(1, 0, key='a',                colors=type_list,
        xlab=r'$a_{i0}$', ylab=r'$a_{i1}$',
        title=r'learned $\mathbf{a}_i$',                       label='c')
    add(1, 1, key='a_ode_W_learned',  colors=type_list,
        xlab=r'UMAP$_1$', ylab=r'UMAP$_2$',
        title=r'learned $(\mathbf{a}_i,\,\tau,\,V_{rest},\,W\text{-stats})$',
        label='d')

    # Panel letters (bold) and titles (lighter) on the same horizontal line,
    # both anchored to the top-left corner of each axes' tight bbox.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, lbl, title in panel_axes:
        bb = ax.get_tightbbox(renderer)
        x0, y1 = inv.transform((bb.x0, bb.y1))
        fig.text(x0 - 0.005, y1 + 0.003, lbl, fontsize=PANEL_LBL,
                 fontweight='bold', va='bottom', ha='left',
                 transform=fig.transFigure)
        # Title sits just right of the bold letter, same baseline.
        fig.text(x0 + 0.020, y1 + 0.003, title, fontsize=FS_TITLE,
                 va='bottom', ha='left',
                 transform=fig.transFigure)

    fig.savefig(out_base + '.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'wrote {out_base}.{{pdf,png}}')


def main():
    if not os.path.isfile(PANELS_NPZ):
        sys.exit(f'missing panels npz: {PANELS_NPZ}\n  → run --replot first')
    if not os.path.isfile(ODE_PARAMS):
        sys.exit(f'missing ode_params.pt: {ODE_PARAMS}')

    d = load_inputs()
    subsets = build_subsets(d)
    out_base = os.path.join(REPO_ROOT, 'figures', 'fig_clustering_appendix')
    assemble(d, subsets, out_base)


if __name__ == '__main__':
    main()
