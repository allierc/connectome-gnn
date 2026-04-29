"""Figure: Known_ODE parameter-extraction panels across three noise levels.

Known_ODE counterpart of ``fig_gnn_params_3col_noise_comparison.py``. The
3-column noise-level layout is preserved (noise-free / σ=0.05 / σ=0.5),
but the inner panel set is reduced from 6 → 3 because Known_ODE has no
learned f_theta MLP, no g_phi MLP, and no per-neuron embedding `model.a`
to plot. The remaining recovery panels are the three direct-parameter
scatters (W, V_rest, τ), and V_rest / τ use the outlier-flagged plots
emitted by ``GNN_PlotFigure.py`` (red = points with |learned − gt| above
the threshold, R² reported on inliers only).

Models used:

    flyvis_noise_free_blank50_known_ode_cv00
    flyvis_noise_005_blank50_known_ode_cv00
    flyvis_noise_05_blank50_known_ode_cv00

Per column (reading order, top → bottom):

    row 1: weights_corrected (R²W)
    row 2: V_rest comparison with outliers in red
    row 3: τ comparison with outliers in red

Modes
-----
    --mode regenerate (default arg --redo): re-runs GNN_PlotFigure.data_plot()
        to ensure all PNGs are present and up-to-date for every column.
    --mode load (default): skips data_plot and assembles whatever is on
        disk (missing panels render as red placeholders).

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/fig_known_ode_params_3col_noise_comparison.py [--redo]

Output
------
    figures/fig_known_ode_params_3col_noise_comparison.{pdf,png}
"""

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

# Three noise levels — same blank50 datasets as the GNN counterpart, but the
# Known_ODE training run (different HP yaml).
COLUMNS = [
    {
        'label': 'noise-free',
        'sigma': r'$\sigma = 0$',
        'model': 'flyvis_noise_free_blank50_known_ode_cv00',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_free_blank50_known_ode_cv00.yaml',
        'config_indices': 'noise_free_blank50_cv00',
    },
    {
        'label': 'low model noise',
        'sigma': r'$\sigma = 0.05$',
        'model': 'flyvis_noise_005_blank50_known_ode_cv00',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_005_blank50_known_ode_cv00.yaml',
        'config_indices': 'noise_005_blank50_cv00',
    },
    {
        'label': 'high model noise',
        'sigma': r'$\sigma = 0.5$',
        'model': 'flyvis_noise_05_blank50_known_ode_cv00',
        'model_yaml': f'{CFG_DIR}/flyvis_noise_05_blank50_known_ode_cv00.yaml',
        'config_indices': 'noise_05_blank50_cv00',
    },
]

# 3 panels per column, ordered (row, col=0). fname_template uses {ci} =
# config_indices. Known_ODE plots V_rest / τ via the outlier-aware PNGs
# (red dots = outliers, R² on inliers).
PANELS = [
    # (row, fname_template)
    (0, 'weights_comparison_raw.png'),                 # R²W (raw — no flux correction for known_ode)
    (1, 'V_rest_comparison_wo_outliers_{ci}.png'),     # V_rest, outliers flagged red
    (2, 'tau_comparison_wo_outliers_{ci}.png'),        # τ,      outliers flagged red
]
N_PANEL_ROWS = 3
N_PANEL_COLS_PER_BLOCK = 1


# ── style ────────────────────────────────────────────────────────────────────
FS_LABEL  = 8
FS_TICK   = 6
FS_ANNOT  = 6
PANEL_LBL = 8

# ~18 cm wide. With one panel per column-row and 3 noise blocks × 1 panel
# wide each, we have 3 panels per row at FIG_W_IN ≈ 7.09 in → per-panel
# width ≈ 2.36 in. Source PNGs have aspect h/w ≈ 0.9, so 3 rows ≈ 6.4 in.
# Slightly shorter than that to avoid trailing whitespace from the title strip.
FIG_W_IN = 18.0 * 0.3937          # ≈ 7.09 in
FIG_H_IN = 6.4


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
    """Return list of (row, expected_path, status) for each PANEL entry."""
    out = []
    for r, fname in PANELS:
        p = panel_path(results_dir, fname, ci)
        out.append((r, p, os.path.isfile(p)))
    return out


def assemble(blocks, out_base):
    """Build the 3-noise-level × 3-row × 1-col composite figure.

    blocks is a list of 3 dicts: {label, sigma, results_dir, ci}.
    Panel letters are assigned in ROW-MAJOR order across the entire
    figure: top row left→right = a..c, second row = d..f, third row = g..i.
    """
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)
    # Outer GridSpec: 1 row, 3 columns (one per noise level). Tight wspace.
    outer = mgs.GridSpec(
        1, 3, figure=fig,
        left=0.04, right=0.99, top=0.92, bottom=0.04,
        wspace=0.10,
    )

    # Build the 3-row x 1-col mini-grids per block.
    inner_grids = []
    for k in range(len(blocks)):
        inner = mgs.GridSpecFromSubplotSpec(
            N_PANEL_ROWS, N_PANEL_COLS_PER_BLOCK,
            subplot_spec=outer[0, k],
            wspace=0.04, hspace=0.10,
        )
        inner_grids.append(inner)

    # Index PANELS by row so we can iterate in row-major order across all three
    # blocks for letter assignment a..i.
    panel_by_row = {r: fname for r, fname in PANELS}

    panel_axes = []                    # (ax, letter, block_idx)
    letters = string.ascii_lowercase
    letter_idx = 0
    for r in range(N_PANEL_ROWS):
        for k, blk in enumerate(blocks):
            inner = inner_grids[k]
            fname = panel_by_row[r]
            ax = fig.add_subplot(inner[r, 0])
            ax.set_axis_off()
            p = panel_path(blk['results_dir'], fname, blk['ci'])
            if not os.path.isfile(p):
                ax.text(0.5, 0.5, f'missing:\n{os.path.basename(p)}',
                        ha='center', va='center', fontsize=FS_ANNOT,
                        color='red', transform=ax.transAxes)
            else:
                img = mpimg.imread(p)
                h, w = img.shape[:2]
                ax.imshow(img, aspect='auto')
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
        fig.text(x_center, y_top + 0.03, title, fontsize=FS_LABEL,
                 fontweight='normal', va='bottom', ha='center',
                 transform=fig.transFigure)

    # Panel letters at top-left of each axes (a..i over 9 panels).
    for ax, lbl, _ in panel_axes:
        bb = ax.get_tightbbox(renderer)
        x0, y1 = inv.transform((bb.x0, bb.y1))
        fig.text(x0 - 0.005, y1 + 0.005, lbl, fontsize=PANEL_LBL,
                 fontweight='bold', va='bottom', ha='left',
                 transform=fig.transFigure)

    fig.savefig(out_base + '.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
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
                            'fig_known_ode_params_3col_noise_comparison')
    assemble(blocks, out_base)


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# Example invocations
# ---------------------------------------------------------------------------
#
# # Default — composite the three-column figure from PNGs already on disk
# # (fast; ~10 s).
# conda run -n neural-graph-linux \
#     python figures/fig_known_ode_params_3col_noise_comparison.py
#
# # Force re-running GNN_PlotFigure.data_plot() for every condition before
# # assembling the composite. Slow (~5–10 min per noise level on a GPU).
# conda run -n neural-graph-linux \
#     python figures/fig_known_ode_params_3col_noise_comparison.py --redo
