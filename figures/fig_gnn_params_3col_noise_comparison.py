"""Figure: GNN parameter-extraction panels across three noise levels.

Replaces the single-condition ``fig_gnn_params_noise_005_blank50.png``. Same
3-column noise-level layout as
``fig_rollout_3col_noise_comparison.py`` (noise-free / σ=0.05 / σ=0.5), but
each column hosts a 3-row × 2-col mini-grid of six parameter-recovery
panels. Models used:

    flyvis_noise_free_blank50_unified_cv00
    flyvis_noise_005_blank50_unified_cv00
    flyvis_noise_05_blank50_unified_cv00

Per column (reading order):

    row 1: weights_corrected (R²W)        embedding (a_i, 2D)
    row 2: f_theta scatter (NEW)          tau (R²)
    row 3: V_rest (R²)                    g_phi scatter (NEW)

The four classical panels (W / embedding / τ / V_rest) are loaded as PNGs
already produced by ``GNN_PlotFigure.data_plot()`` under the model's
``results/`` directory. The two NEW scatter panels — learned vs true
output of f_theta(a_i, v_i) and g_phi(a_j, v_j) — are emitted by the same
``data_plot`` call after the recent change in ``GNN_PlotFigure.py``.

Modes
-----
    --mode regenerate (default): re-runs GNN_PlotFigure.data_plot() to
        ensure all six PNGs are present and up-to-date for every column.
    --mode load: skips data_plot and assembles whatever is on disk
        (missing panels render as red placeholders).

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/fig_gnn_params_3col_noise_comparison.py [--mode load]

Output
------
    figures/fig_gnn_params_3col_noise_comparison.{pdf,png}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Configs        : <DATA_ROOT>/config/fly/flyvis_noise_free_blank50_unified_cv00.yaml
#                  <DATA_ROOT>/config/fly/flyvis_noise_005_blank50_unified_cv00.yaml
#                  <DATA_ROOT>/config/fly/flyvis_noise_05_blank50_unified_cv00.yaml
# Training data  : <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_blank50_cv00/x_list_train/
#                  <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_blank50_cv00/{edge_index.pt, ode_params.pt}
# Test data      : <DATA_ROOT>/graphs_data/fly/flyvis_noise_{free,005,05}_blank50_cv00/x_list_test/
# Trained models : <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_unified_cv00/models/best_model_with_0_graphs_0.pt
# Eval logs      : <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_unified_cv00/results_{test,rollout}.log
#                  <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_unified_cv00/results/metrics.txt
# Cached panels  : <DATA_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_unified_cv00/results/{weights_comparison_corrected,embedding_*,f_theta_scatter_*,g_phi_scatter_*,V_rest_comparison_wo_outliers_*,tau_comparison_wo_outliers_*}.png
# Output         : figures/fig_gnn_params_3col_noise_comparison.{pdf,png}
# ─────────────────────────────────────────────────────────────────────────────

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


# Make `connectome_gnn` importable + locate the repo root.
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

# Three noise levels — same models as fig_rollout_3col_noise_comparison.py.
COLUMNS = [
    {
        'label': 'noise-free',
        'sigma': r'$\sigma = 0$',
        'model': 'flyvis_noise_free_blank50_unified_cv00',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_free_blank50_unified_cv00.yaml',
        'config_indices': 'noise_free_blank50_cv00',
    },
    {
        'label': 'low model noise',
        'sigma': r'$\sigma = 0.05$',
        'model': 'flyvis_noise_005_blank50_unified_cv00',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_005_blank50_unified_cv00.yaml',
        'config_indices': 'noise_005_blank50_cv00',
    },
    {
        'label': 'high model noise',
        'sigma': r'$\sigma = 0.5$',
        'model': 'flyvis_noise_05_blank50_unified_cv00',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_05_blank50_unified_cv00.yaml',
        'config_indices': 'noise_05_blank50_cv00',
    },
]

# 6 panels per column, ordered (row, col_within_column).
# fname_template uses {ci} = config_indices.
# τ and g_phi positions swapped (was: tau in row 1 col 1, g_phi in row 2 col 1).
PANELS = [
    # (row, col, fname_template)
    (0, 0, 'weights_comparison_corrected.png'),  # R²W
    (0, 1, 'embedding_{ci}.png'),                # 2D embedding
    (1, 0, 'f_theta_scatter_{ci}.png'),          # learned vs true f_theta
    (1, 1, 'g_phi_scatter_{ci}.png'),            # learned vs true g_phi  (was τ)
    (2, 0, 'V_rest_comparison_wo_outliers_{ci}.png'),  # V_rest (no outliers)
    (2, 1, 'tau_comparison_wo_outliers_{ci}.png'),     # τ      (no outliers)
]
N_PANEL_ROWS = 3
N_PANEL_COLS_PER_BLOCK = 2


# ── style ────────────────────────────────────────────────────────────────────
FS_LABEL  = 8
FS_TICK   = 6
FS_ANNOT  = 6
PANEL_LBL = 8

# ~18 cm wide. Each source PNG has aspect h/w ≈ 0.9, so for 6 panel columns
# (3 noise blocks × 2 panels each) at FIG_W_IN ≈ 7.09 in the per-panel width
# is ~1.18 in and the matching panel height is ~1.06 in. 3 rows ≈ 3.2 in
# plus margins/title strip — keep the figure short to eliminate the dead
# whitespace above/below each panel from previous (too-tall) layouts.
FIG_W_IN = 18.0 * 0.3937          # ≈ 7.09 in
FIG_H_IN = 4.2


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
        apply_weight_correction=True,
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
    """Build the 3-noise-level × 3-row × 2-col composite figure.

    blocks is a list of 3 dicts: {label, sigma, results_dir, ci}.
    Panel letters are assigned in ROW-MAJOR order across the entire
    figure (top row left→right = a..f, second row = g..l, third row =
    m..r), not block-major.
    """
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)
    # Outer GridSpec: 1 row, 3 columns (one per noise level). Tight wspace.
    outer = mgs.GridSpec(
        1, 3, figure=fig,
        left=0.04, right=0.99, top=0.88, bottom=0.04,
        wspace=0.10,
    )

    # Build the 3-row x 2-col mini-grids per block.
    inner_grids = []
    for k in range(len(blocks)):
        inner = mgs.GridSpecFromSubplotSpec(
            N_PANEL_ROWS, N_PANEL_COLS_PER_BLOCK,
            subplot_spec=outer[0, k],
            wspace=0.04, hspace=0.10,
        )
        inner_grids.append(inner)

    # Index PANELS by (row, col) so we can iterate in row-major order across
    # all three blocks for letter assignment a..r.
    panel_by_rc = {(r, c): fname for r, c, fname in PANELS}

    panel_axes = []                    # (ax, letter, block_idx)
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

    # Section titles per block (centered above each column block).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    for k, blk in enumerate(blocks):
        block_axes = [a for a, _, bi in panel_axes if bi == k]
        # Find the bbox of the top row of this block.
        bboxes = [a.get_tightbbox(renderer) for a in block_axes[:N_PANEL_COLS_PER_BLOCK]]
        if not bboxes:
            continue
        x_left  = min(inv.transform((bb.x0, bb.y1))[0] for bb in bboxes)
        x_right = max(inv.transform((bb.x1, bb.y1))[0] for bb in bboxes)
        y_top   = max(inv.transform((bb.x0, bb.y1))[1] for bb in bboxes)
        x_center = (x_left + x_right) / 2
        title = f"{blk['label']} ({blk['sigma']})"
        fig.text(x_center, y_top + 0.04, title, fontsize=FS_LABEL,
                 fontweight='normal', va='bottom', ha='center',
                 transform=fig.transFigure)

    # Panel letters at top-left of each axes (a..r over 18 panels).
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
                        '(alias for --mode regenerate). Convenience flag '
                        'when iterating on the per-panel rendering code.')
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
            missing = [p for *_, p, ok in panels_present(results_dir,
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
                            'fig_gnn_params_3col_noise_comparison')
    assemble(blocks, out_base)


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# Example invocations
# ---------------------------------------------------------------------------
#
# # Default — composite the three-column figure from PNGs already on disk
# # (fast; ~10 s). Use after a `--redo` run, or after re-running data_plot
# # via the GNN_PlotFigure pipeline by other means.
# conda run -n neural-graph-linux \
#     python figures/fig_gnn_params_3col_noise_comparison.py
#
# # Force re-running GNN_PlotFigure.data_plot() for every condition before
# # assembling the composite. Slow (~5–10 min per noise level on a GPU).
# # `--redo` and `-r` are aliases for `--mode regenerate`.
# conda run -n neural-graph-linux \
#     python figures/fig_gnn_params_3col_noise_comparison.py --redo
# conda run -n neural-graph-linux \
#     python figures/fig_gnn_params_3col_noise_comparison.py -r
# conda run -n neural-graph-linux \
#     python figures/fig_gnn_params_3col_noise_comparison.py --mode regenerate
#
# # Run on CPU (default picks cuda if available — only relevant in --redo
# # mode; the load-only path doesn't touch torch).
# conda run -n neural-graph-linux \
#     python figures/fig_gnn_params_3col_noise_comparison.py --redo --device cpu
