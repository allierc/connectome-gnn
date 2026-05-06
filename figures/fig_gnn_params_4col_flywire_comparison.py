"""Figure: GNN parameter-extraction panels across the 4 flywireRF v2 conditions.

4-column variant of ``fig_gnn_params_3col_noise_comparison.py``. Sweeps the
four cv00 GNN models trained by ``run_GNN_flywire_blank50.py``:

    e8_flywireRF_noise_005_blank50_flywire_cv00
    e8_flywireRF_proximal_nulls_noise_005_blank50_flywire_cv00
    full_eye_flywireRF_noise_005_blank50_flywire_cv00
    full_eye_flywireRF_proximal_nulls_noise_005_blank50_flywire_cv00

Per column (reading order):

    row 1: weights_corrected (R²W)        embedding (a_i, 2D)
    row 2: f_theta scatter                g_phi scatter
    row 3: V_rest (R²)                    tau (R²)

The six panels per column are loaded as PNGs already produced by
``GNN_PlotFigure.data_plot()`` under each model's ``results/`` directory.

Modes
-----
    --mode load (default): assemble whatever is on disk (missing panels
        render as red placeholders).
    --mode regenerate: re-runs GNN_PlotFigure.data_plot() for every column
        before assembling.

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/fig_gnn_params_4col_flywire_comparison.py [--mode regenerate]

Output
------
    figures/fig_gnn_params_4col_flywire_comparison.{pdf,png}
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

# Four flywireRF cv00 models trained by run_GNN_flywire_blank50.py.
# config_indices = <base>_blank50_cv00 (no `_flywire_` because the dataset
# folder uses dataset_tag='blank50').
COLUMNS = [
    {
        'label': 'e8',
        'model': 'e8_flywireRF_noise_005_blank50_flywire_cv00',
        'model_yaml': f'{CFG_DIR}/e8_flywireRF_noise_005_blank50_flywire_cv00.yaml',
        'config_indices': 'e8_flywireRF_noise_005_blank50_cv00',
    },
    {
        'label': 'e8 + proximal nulls',
        'model': 'e8_flywireRF_proximal_nulls_noise_005_blank50_flywire_cv00',
        'model_yaml': f'{CFG_DIR}/e8_flywireRF_proximal_nulls_noise_005_blank50_flywire_cv00.yaml',
        'config_indices': 'e8_flywireRF_proximal_nulls_noise_005_blank50_cv00',
    },
    {
        'label': 'full eye',
        'model': 'full_eye_flywireRF_noise_005_blank50_flywire_cv00',
        'model_yaml': f'{CFG_DIR}/full_eye_flywireRF_noise_005_blank50_flywire_cv00.yaml',
        'config_indices': 'full_eye_flywireRF_noise_005_blank50_cv00',
    },
    {
        'label': 'full eye + proximal nulls',
        'model': 'full_eye_flywireRF_proximal_nulls_noise_005_blank50_flywire_cv00',
        'model_yaml': f'{CFG_DIR}/full_eye_flywireRF_proximal_nulls_noise_005_blank50_flywire_cv00.yaml',
        'config_indices': 'full_eye_flywireRF_proximal_nulls_noise_005_blank50_cv00',
    },
]

PANELS = [
    (0, 0, 'weights_comparison_corrected.png'),
    (0, 1, 'embedding_{ci}.png'),
    (1, 0, 'f_theta_scatter_{ci}.png'),
    (1, 1, 'g_phi_scatter_{ci}.png'),
    (2, 0, 'V_rest_comparison_wo_outliers_{ci}.png'),
    (2, 1, 'tau_comparison_wo_outliers_{ci}.png'),
]
N_PANEL_ROWS = 3
N_PANEL_COLS_PER_BLOCK = 2


FS_LABEL  = 8
FS_TICK   = 6
FS_ANNOT  = 6
PANEL_LBL = 8

# 4 conditions × 2 panels = 8 panel columns. Keep per-panel width comparable
# to the 3-col version (~1.18 in) so 8 × 1.18 ≈ 9.45 in (24 cm).
FIG_W_IN = 24.0 * 0.3937
FIG_H_IN = 4.2


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
        apply_weight_correction=True,
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
    """Build the 4-condition × 3-row × 2-col composite figure.

    Letters are assigned in row-major order across the entire figure:
    top row left→right = a..h, second row = i..p, third row = q..x.
    """
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)
    outer = mgs.GridSpec(
        1, len(blocks), figure=fig,
        left=0.03, right=0.99, top=0.88, bottom=0.04,
        wspace=0.10,
    )

    inner_grids = []
    for k in range(len(blocks)):
        inner = mgs.GridSpecFromSubplotSpec(
            N_PANEL_ROWS, N_PANEL_COLS_PER_BLOCK,
            subplot_spec=outer[0, k],
            wspace=0.04, hspace=0.10,
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
        title = blk['label']
        fig.text(x_center, y_top + 0.04, title, fontsize=FS_LABEL,
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
            'results_dir': results_dir,
            'ci': col['config_indices'],
        })

    out_base = os.path.join(REPO_ROOT, 'figures',
                            'fig_gnn_params_4col_flywire_comparison')
    assemble(blocks, out_base)


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# Example invocations
# ---------------------------------------------------------------------------
#
# # Default — composite the four-column figure from PNGs already on disk
# # (fast; ~10 s).
# conda run -n neural-graph-linux \
#     python figures/fig_gnn_params_4col_flywire_comparison.py
#
# # Force re-running GNN_PlotFigure.data_plot() for every condition.
# conda run -n neural-graph-linux \
#     python figures/fig_gnn_params_4col_flywire_comparison.py --redo
