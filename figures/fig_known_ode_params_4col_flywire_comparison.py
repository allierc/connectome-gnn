"""Figure: Known_ODE parameter-extraction panels across the 4 flywireRF v2 conditions.

4-column flywireRF variant of ``fig_known_ode_params_3col_noise_comparison.py``.
Sweeps the four cv00 known_ode models trained by
``run_KnownODE_flywire_blank50.py``:

    e8_flywireRF_noise_005_blank50_flywire_known_ode_cv00
    e8_flywireRF_proximal_nulls_noise_005_blank50_flywire_known_ode_cv00
    full_eye_flywireRF_noise_005_blank50_flywire_known_ode_cv00
    full_eye_flywireRF_proximal_nulls_noise_005_blank50_flywire_known_ode_cv00

Per column (2×2 mini-grid):

    row 0:  W (col 0)       UMAP / embedding_augmented (col 1)
    row 1:  V_rest (col 0)  τ                          (col 1)

The W panel uses the raw (un-flux-corrected) PNG since known_ode learns no
edge MLP. V_rest / τ use the outlier-flagged plots emitted by
``GNN_PlotFigure.py`` (red = points with |learned − gt| above the threshold,
R² reported on inliers only). The UMAP panel is rendered fresh with the
husl-65 LUT coloured by GT cell type, identical to
``fig_known_ode_params_3col_noise_comparison.py``.

Modes
-----
    --mode load (default): assemble whatever is on disk (missing panels
        render as red placeholders).
    --mode regenerate (alias --redo / -r): re-runs GNN_PlotFigure.data_plot()
        before assembling.

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/fig_known_ode_params_4col_flywire_comparison.py [--redo]

Output
------
    figures/fig_known_ode_params_4col_flywire_comparison.{pdf,png}
"""

import argparse
import os
import string
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'unified_style.matplotlibrc'))

import matplotlib.gridspec as mgs
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_ROOT = os.environ.get('TRAINED_MODEL_OUTPUT_ROOT', '.')
for _p in (os.path.join(REPO_ROOT, 'src'), REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
import connectome_gnn.utils as _cg_utils  # noqa: E402
try:
    from connectome_gnn.utils import set_data_root  # noqa: E402
except ImportError:
    def set_data_root(path):
        _cg_utils._data_root = path

CFG_DIR = f'{DATA_ROOT}/config/fly'

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
UMAP_RANDOM      = 42
HUSL_65_CMAP     = ListedColormap(sns.color_palette("husl", 65))


def _connectivity_stats(w, src, dst, n):
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
    return np.column_stack([in_mean, in_std, out_mean, out_std,
                            in_min, in_max, out_min, out_max])


def compute_umap_panel(model_dir, dataset, log_dir):
    """Return (xy, type_ids) for the augmented-feature UMAP of a known_ode run."""
    import umap
    model_pt   = f'{log_dir}/models/best_model_with_0_graphs_0.pt'
    ode_pt     = f'{DATA_ROOT}/graphs_data/fly/{dataset}/ode_params.pt'
    bundle_npz = f'{log_dir}/results/rollout_bundle.npz'
    sd = torch.load(model_pt, weights_only=False, map_location='cpu')['model_state_dict']
    raw_tau = sd.get('_orig_mod.raw_tau', sd.get('raw_tau'))
    V_rest  = sd.get('_orig_mod.V_rest',  sd.get('V_rest'))
    W       = sd.get('_orig_mod.W',       sd.get('W'))
    tau_learned    = torch.nn.functional.softplus(raw_tau).cpu().numpy()
    V_rest_learned = V_rest.cpu().numpy()
    W_learned      = W.squeeze().cpu().numpy()
    op = torch.load(ode_pt, weights_only=False, map_location='cpu')
    edge_index = op['edge_index'].cpu().numpy()
    src, dst = edge_index[0], edge_index[1]
    n = tau_learned.shape[0]
    w_stats = _connectivity_stats(W_learned, src, dst, n)
    feats = np.column_stack([tau_learned[:, None], V_rest_learned[:, None], w_stats])
    type_ids = np.asarray(np.load(bundle_npz, allow_pickle=True)['type_ids']).astype(int)
    reducer = umap.UMAP(n_components=2, random_state=UMAP_RANDOM,
                        n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST)
    xy = reducer.fit_transform(feats)
    return xy, type_ids


def render_umap_png(xy, type_ids, out_path):
    fig = plt.figure(figsize=(10, 9))
    ax = plt.gca()
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_alpha(0.75)
        sp.set_linewidth(1.5)
    ax.scatter(xy[:, 0], xy[:, 1], c=type_ids, cmap=HUSL_65_CMAP,
               s=24, alpha=0.8, edgecolors='none')
    ax.set_xlabel(r'UMAP$_1$', fontsize=48)
    ax.set_ylabel(r'UMAP$_2$', fontsize=48)
    x_lo, x_hi = float(np.floor(xy[:, 0].min())), float(np.ceil(xy[:, 0].max()))
    y_lo, y_hi = float(np.floor(xy[:, 1].min())), float(np.ceil(xy[:, 1].max()))
    ax.set_xticks([x_lo, 0.0, x_hi])
    ax.set_yticks([y_lo, 0.0, y_hi])
    ax.tick_params(axis='both', labelsize=36,
                   top=True, right=True, direction='out')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# Four flywireRF cv00 known_ode models trained by run_KnownODE_flywire_blank50.py.
# config_indices = <base>_blank50_cv00 (no `_flywire_known_ode_` because the
# dataset folder uses dataset_tag='blank50').
COLUMNS = [
    {
        'label': 'e8',
        'model': 'e8_flywireRF_noise_005_blank50_flywire_known_ode_cv00',
        'model_yaml': f'{CFG_DIR}/e8_flywireRF_noise_005_blank50_flywire_known_ode_cv00.yaml',
        'config_indices': 'e8_flywireRF_noise_005_blank50_cv00',
    },
    {
        'label': 'e8 + proximal nulls',
        'model': 'e8_flywireRF_proximal_nulls_noise_005_blank50_flywire_known_ode_cv00',
        'model_yaml': f'{CFG_DIR}/e8_flywireRF_proximal_nulls_noise_005_blank50_flywire_known_ode_cv00.yaml',
        'config_indices': 'e8_flywireRF_proximal_nulls_noise_005_blank50_cv00',
    },
    {
        'label': 'full eye',
        'model': 'full_eye_flywireRF_noise_005_blank50_flywire_known_ode_cv00',
        'model_yaml': f'{CFG_DIR}/full_eye_flywireRF_noise_005_blank50_flywire_known_ode_cv00.yaml',
        'config_indices': 'full_eye_flywireRF_noise_005_blank50_cv00',
    },
    {
        'label': 'full eye + proximal nulls',
        'model': 'full_eye_flywireRF_proximal_nulls_noise_005_blank50_flywire_known_ode_cv00',
        'model_yaml': f'{CFG_DIR}/full_eye_flywireRF_proximal_nulls_noise_005_blank50_flywire_known_ode_cv00.yaml',
        'config_indices': 'full_eye_flywireRF_proximal_nulls_noise_005_blank50_cv00',
    },
]

PANELS = [
    (0, 0, 'weights_comparison_raw.png'),
    (0, 1, 'embedding_augmented_{ci}.png'),
    (1, 0, 'V_rest_comparison_wo_outliers_{ci}.png'),
    (1, 1, 'tau_comparison_wo_outliers_{ci}.png'),
]
N_PANEL_ROWS = 2
N_PANEL_COLS_PER_BLOCK = 2


FS_LABEL  = 6
FS_TICK   = 5
FS_ANNOT  = 5
PANEL_LBL = 6

# 4 conditions × 2 panels = 8 panel columns. Match the GNN-flywire 4col
# (~24 cm). Same height profile as the 3-col known_ode version.
FIG_W_IN = 24.0 * 0.3937
FIG_H_IN = 4.0


def load_config_from_yaml(yaml_path):
    if not os.path.isfile(yaml_path):
        sys.exit(f'missing model yaml: {yaml_path}')
    cfg = NeuralGraphConfig.from_yaml(yaml_path)
    parent = os.path.basename(os.path.dirname(os.path.abspath(yaml_path)))
    pre_folder = parent + '/' if parent else ''
    if not cfg.dataset.startswith(pre_folder):
        cfg.dataset = pre_folder + cfg.dataset
    if cfg.config_file == 'none':
        stem = os.path.splitext(os.path.basename(yaml_path))[0]
        cfg.config_file = pre_folder + stem
    return cfg


def regenerate_panels(cfg, device):
    from GNN_PlotFigure import data_plot
    data_plot(
        config=cfg,
        epoch_list=['best'],
        style='color',
        extended='plots',
        device=device,
        apply_weight_correction=False,
        skip_svd=False,
    )


def panel_path(results_dir, fname_template, ci):
    return os.path.join(results_dir, fname_template.format(ci=ci))


def panels_present(results_dir, ci):
    out = []
    for r, c, fname in PANELS:
        p = panel_path(results_dir, fname, ci)
        out.append((r, c, p, os.path.isfile(p)))
    return out


def assemble(blocks, out_base):
    """Build the 4-condition × 2-row × 2-col composite figure.

    Letters are assigned in row-major order across the entire figure:
    top row left→right = a..h, bottom row = i..p.
    """
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)
    outer = mgs.GridSpec(
        1, len(blocks), figure=fig,
        left=0.03, right=0.99, top=0.84, bottom=0.04,
        wspace=0.10,
    )

    inner_grids = []
    for k in range(len(blocks)):
        inner = mgs.GridSpecFromSubplotSpec(
            N_PANEL_ROWS, N_PANEL_COLS_PER_BLOCK,
            subplot_spec=outer[0, k],
            wspace=0.04, hspace=0.02,
            height_ratios=[1, 2],
        )
        inner_grids.append(inner)

    panel_by_rc = {(r, c): fname for r, c, fname in PANELS}

    panel_axes = []
    letters = string.ascii_lowercase
    letter_idx = 0
    for r in range(N_PANEL_ROWS):
        for k, blk in enumerate(blocks):
            inner = inner_grids[k]
            for c in range(N_PANEL_COLS_PER_BLOCK):
                ax = fig.add_subplot(inner[r, c])
                ax.set_axis_off()
                if (r, c) == (0, 1):
                    p = blk['umap_png']
                else:
                    fname = panel_by_rc[(r, c)]
                    p = panel_path(blk['results_dir'], fname, blk['ci'])
                if not os.path.isfile(p):
                    ax.text(0.5, 0.5, f'missing:\n{os.path.basename(p)}',
                            ha='center', va='center', fontsize=FS_ANNOT,
                            color='red', transform=ax.transAxes)
                else:
                    img = mpimg.imread(p)
                    h, w = img.shape[:2]
                    ax.imshow(img, aspect='auto', interpolation='lanczos')
                    ax.set_box_aspect(h / w)
                    ax.set_anchor('S' if r == 0 else 'N')
                panel_axes.append((ax, letters[letter_idx], k))
                letter_idx += 1

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    for k, blk in enumerate(blocks):
        block_axes = [a for a, _, bi in panel_axes if bi == k]
        bboxes = [a.get_tightbbox(renderer) for a in block_axes[:N_PANEL_COLS_PER_BLOCK]]
        if not bboxes:
            continue
        x_left  = min(inv.transform((bb.x0, bb.y1))[0] for bb in bboxes)
        x_right = max(inv.transform((bb.x1, bb.y1))[0] for bb in bboxes)
        y_top   = max(inv.transform((bb.x0, bb.y1))[1] for bb in bboxes)
        x_center = (x_left + x_right) / 2
        title = blk['label']
        fig.text(x_center, y_top + 0.06, title, fontsize=FS_LABEL,
                 fontweight='normal', va='bottom', ha='center',
                 transform=fig.transFigure)

    for ax, lbl, _ in panel_axes:
        bb = ax.get_tightbbox(renderer)
        x0, y1 = inv.transform((bb.x0, bb.y1))
        fig.text(x0 - 0.005, y1 + 0.005, lbl, fontsize=PANEL_LBL,
                 fontweight='bold', va='bottom', ha='left',
                 transform=fig.transFigure)

    fig.savefig(out_base + '.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_base + '.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'wrote {out_base}.{{pdf,png}}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['regenerate', 'load'], default='load',
                   help='regenerate: re-run GNN_PlotFigure.data_plot() to '
                        'refresh every column\'s panel PNGs (slow). '
                        'load (default): assemble from PNGs already on disk.')
    p.add_argument('--redo', '-r', action='store_true',
                   help='Alias for --mode regenerate.')
    p.add_argument('--device', default=None,
                   help='torch device for regenerate (default: cuda if avail, else cpu)')
    args = p.parse_args()
    if args.redo:
        args.mode = 'regenerate'

    if args.device is None:
        try:
            args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        except Exception:
            args.device = 'cpu'

    set_data_root(DATA_ROOT)

    blocks = []
    for col in COLUMNS:
        cfg = load_config_from_yaml(col['model_yaml'])
        log_dir = f"{DATA_ROOT}/log/{cfg.config_file}"
        results_dir = os.path.join(log_dir, 'results')
        print(f"\n=== {col['model']} ===")
        print(f"log_dir: {log_dir}")
        print(f"config_indices: {col['config_indices']}")

        if args.mode == 'regenerate':
            print('regenerating panels via GNN_PlotFigure.data_plot()')
            regenerate_panels(cfg, args.device)
        else:
            missing = [p for _r, _c, p, ok in panels_present(results_dir,
                                                              col['config_indices']) if not ok]
            if missing:
                print(f"WARNING: {len(missing)} panel(s) missing for {col['model']}:")
                for m in missing:
                    print(f"  - {os.path.basename(m)}")
                print('  → run with --mode regenerate to produce them')

        print(f'computing fresh UMAP (husl-65 LUT) for {col["model"]}...')
        umap_xy, umap_types = compute_umap_panel(
            model_dir=col['model'], dataset=cfg.dataset.split('/')[-1],
            log_dir=log_dir,
        )
        umap_png = os.path.join(
            results_dir,
            f"embedding_augmented_husl_{col['config_indices']}.png")
        render_umap_png(umap_xy, umap_types, umap_png)
        blocks.append({
            'umap_png': umap_png,
            'label':  col['label'],
            'results_dir': results_dir,
            'ci': col['config_indices'],
        })

    out_base = os.path.join(REPO_ROOT, 'figures',
                            'fig_known_ode_params_4col_flywire_comparison')
    assemble(blocks, out_base)


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# Example invocations
# ---------------------------------------------------------------------------
#
# # Default — composite the four-column figure from PNGs already on disk
# # (still recomputes UMAP per column; a few minutes).
# conda run -n neural-graph-linux \
#     python figures/fig_known_ode_params_4col_flywire_comparison.py
#
# # Force re-running GNN_PlotFigure.data_plot() for every condition.
# conda run -n neural-graph-linux \
#     python figures/fig_known_ode_params_4col_flywire_comparison.py --redo
