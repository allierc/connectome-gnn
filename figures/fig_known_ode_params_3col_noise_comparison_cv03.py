"""Figure: Known_ODE parameter-extraction panels across three noise levels — cv03.

Twin of ``fig_known_ode_params_3col_noise_comparison.py`` (which uses cv00)
pointing at the cv03 fold instead. Motivation: the unified GNN run
``flyvis_noise_free_blank50_unified_cv03`` is the outlier of the 5-fold CV
(rollout explodes; predictions clamp at ±100 starting at frame ~160). To
diagnose whether the parameter-recovery quality at cv03 looks any different
from cv00 — in particular whether τ is more strongly underestimated for the
high-true-τ neurons, which would shrink the effective leak time-constant
relative to the synaptic drive and predispose forward-Euler integration to
runaway feedback — we render the same 4-panel mini-grid (W, UMAP, V_rest,
τ) for the cv03 known_ode runs at all three noise levels.

3-column noise-level layout: noise-free / σ=0.05 / σ=0.5.
Per column: 2×2 mini-grid — W, augmented UMAP, V_rest, τ.

Models used:

    flyvis_noise_free_blank50_known_ode_cv03
    flyvis_noise_005_blank50_known_ode_cv03
    flyvis_noise_05_blank50_known_ode_cv03

Modes
-----
    --mode regenerate (default arg --redo): re-runs GNN_PlotFigure.data_plot()
        to ensure all PNGs are present and up-to-date for every column.
    --mode load (default): skips data_plot and assembles whatever is on
        disk (missing panels render as red placeholders).

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/fig_known_ode_params_3col_noise_comparison_cv03.py [--redo]

Output
------
    figures/fig_known_ode_params_3col_noise_comparison_cv03.{pdf,png}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Data root      : /groups/saalfeld/home/allierc/GraphData
# Configs        : <DATA_ROOT>/config/fly/flyvis_noise_free_blank50_known_ode_cv03.yaml
#                  <DATA_ROOT>/config/fly/flyvis_noise_005_blank50_known_ode_cv03.yaml
#                  <DATA_ROOT>/config/fly/flyvis_noise_05_blank50_known_ode_cv03.yaml
# Training data  : <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_blank50_cv03/x_list_train/
#                  <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_blank50_cv03/{edge_index.pt, ode_params.pt}
# Test data      : <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_blank50_cv03/x_list_test/
# Trained models : <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_known_ode_cv03/models/best_model_with_0_graphs_0.pt
# Eval logs      : <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_known_ode_cv03/results_{test,rollout}.log
#                  <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_known_ode_cv03/results/metrics.txt
# Cached panels  : <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_known_ode_cv03/results/{weights_comparison_raw,embedding_augmented_*,V_rest_comparison_wo_outliers_*,tau_comparison_wo_outliers_*}.png
# Output         : figures/fig_known_ode_params_3col_noise_comparison_cv03{,_nf_green}.{pdf,png}
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import string
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'janne.matplotlibrc'))

import matplotlib.gridspec as mgs
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# Make `connectome_gnn` importable + locate the repo root.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_ROOT = '/groups/saalfeld/home/allierc/GraphData'
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

# Three noise levels at cv03 — same blank50 datasets as the cv00 counterpart,
# but for the outlier fold of the 5-CV split.
COLUMNS = [
    {
        'label': 'noise-free',
        'sigma': r'$\sigma = 0$',
        'model': 'flyvis_noise_free_blank50_known_ode_cv03',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_free_blank50_known_ode_cv03.yaml',
        'config_indices': 'noise_free_blank50_cv03',
    },
    {
        'label': 'low model noise',
        'sigma': r'$\sigma = 0.05$',
        'model': 'flyvis_noise_005_blank50_known_ode_cv03',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_005_blank50_known_ode_cv03.yaml',
        'config_indices': 'noise_005_blank50_cv03',
    },
    {
        'label': 'high model noise',
        'sigma': r'$\sigma = 0.5$',
        'model': 'flyvis_noise_05_blank50_known_ode_cv03',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_05_blank50_known_ode_cv03.yaml',
        'config_indices': 'noise_05_blank50_cv03',
    },
]

# 4 panels per column in a 2×2 mini-grid, ordered (row, col_within_column).
# fname_template uses {ci} = config_indices. Known_ODE plots V_rest / τ via
# the outlier-aware PNGs (red dots = outliers, R² on inliers).
PANELS = [
    # (row, col, fname_template)
    (0, 0, 'weights_comparison_raw.png'),              # R²W (raw — no flux correction for known_ode)
    (0, 1, 'embedding_augmented_{ci}.png'),            # augmented UMAP embedding
    (1, 0, 'V_rest_comparison_wo_outliers_{ci}.png'),  # V_rest, outliers flagged red
    (1, 1, 'tau_comparison_wo_outliers_{ci}.png'),     # τ,      outliers flagged red
]
N_PANEL_ROWS = 2
N_PANEL_COLS_PER_BLOCK = 2


# ── style ────────────────────────────────────────────────────────────────────
FS_LABEL  = 8
FS_TICK   = 6
FS_ANNOT  = 6
PANEL_LBL = 8

FIG_W_IN = 18.0 * 0.3937          # ≈ 7.09 in
FIG_H_IN = 4.0


def load_config_from_yaml(yaml_path):
    """Load a NeuralGraphConfig directly from an absolute YAML path."""
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
    """Run GNN_PlotFigure.data_plot() to refresh every panel PNG."""
    from GNN_PlotFigure import data_plot
    data_plot(
        config=cfg,
        epoch_list=['best'],
        style='color',
        extended='plots',
        device=device,
        apply_weight_correction=False,   # known_ode: no flux correction
        skip_svd=False,
    )


def panel_path(results_dir, fname_template, ci):
    return os.path.join(results_dir, fname_template.format(ci=ci))


def panels_present(results_dir, ci):
    """Return list of (row, col, expected_path, status) for each PANEL entry."""
    out = []
    for r, c, fname in PANELS:
        p = panel_path(results_dir, fname, ci)
        out.append((r, c, p, os.path.isfile(p)))
    return out


def assemble(blocks, out_base):
    """Build the 3-noise-level × 2-row × 2-col composite figure."""
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)
    outer = mgs.GridSpec(
        1, 3, figure=fig,
        left=0.04, right=0.99, top=0.84, bottom=0.04,
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
                fname = panel_by_rc[(r, c)]
                ax = fig.add_subplot(inner[r, c])
                ax.set_axis_off()
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
        title = f"{blk['label']} ({blk['sigma']}) — cv03"
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
                   help='Force re-running data_plot() for every condition '
                        '(alias for --mode regenerate).')
    p.add_argument('--device', default=None,
                   help='torch device for regenerate (default: cuda if avail, else cpu)')
    args = p.parse_args()
    if args.redo:
        args.mode = 'regenerate'

    if args.device is None:
        try:
            import torch
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

        blocks.append({
            'label':  col['label'],
            'sigma':  col['sigma'],
            'results_dir': results_dir,
            'ci': col['config_indices'],
        })

    out_base = os.path.join(REPO_ROOT, 'figures',
                            'fig_known_ode_params_3col_noise_comparison_cv03')
    assemble(blocks, out_base)
    assemble(blocks, out_base + '_nf_green')


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# Example invocations
# ---------------------------------------------------------------------------
#
# # Default — composite from PNGs already on disk (fast; ~10 s).
# conda run -n neural-graph-linux \
#     python figures/fig_known_ode_params_3col_noise_comparison_cv03.py
#
# # Force re-running GNN_PlotFigure.data_plot() before assembling.
# conda run -n neural-graph-linux \
#     python figures/fig_known_ode_params_3col_noise_comparison_cv03.py --redo
