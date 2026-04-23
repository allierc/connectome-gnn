"""
Figure: GNN parameter-extraction panels (one composite per config).

Janne-styled per figures/INSTRUCTIONS.md (the previous, larger-font version
is preserved at fig_gnn_params_original.py):

  * ~18 cm document-width figure (7.09 in) at 300 dpi
  * 6-8 pt fonts, 0.5 pt spines / ticks
  * top + right spines hidden globally (via janne.matplotlibrc)
  * trim_axis breaks each axis at the data range (upper & right gap)
  * PDF primary output (pdf.fonttype=42, svg.fonttype='none')

2 rows x 5 cols (labels a-h) assembling the individual PNGs produced by
GNN_PlotFigure.data_plot().  Panels c and g span 2 columns each because
they are internally 2x1 (landscape) - see PANEL_LAYOUT.

  row 1:  a) weights_corrected  b) embedding  c) f_theta_domain (x2)  d) emb_aug
  row 2:  e) tau                f) V_rest     g) g_phi_domain (x2)    [blank]

where {ci} is the config_indices string (e.g. noise_005).

Modes (--mode)
--------------
  regenerate (default)
      Call GNN_PlotFigure.data_plot() to produce fresh panel PNGs from
      the trained model, then assemble the composite.
  load
      Skip data_plot(); just load the PNGs already under
      <log_dir>/results/ and assemble the composite.

Usage
-----
    # one or more configs in a single call; one composite per config.
    # --output_root works like GNN_Main.py --output_root - points at the
    # data root where log/<config>/ and graphs_data/ live.
    python figures/fig_gnn_params.py \
        --output_root /groups/saalfeld/home/allierc/GraphData \
        --configs flyvis_noise_free_winner flyvis_noise_005_winner flyvis_noise_05_winner

Output
------
    figures/fig_gnn_params_<config_indices>.{pdf,png}   # one per config
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'janne.matplotlibrc'))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# Try the flyvis trim_axis; fall back to a local equivalent if unavailable.
try:
    from flyvis.analysis.visualization.plt_utils import trim_axis as _trim_axis
except Exception:
    def _trim_axis(ax, xmargin=0.0, ymargin=0.0, yaxis=True, xaxis=True):
        """Local fallback: clip left/bottom spines to the data range so the
        axes break at the first/last data point (no spine beyond the data)."""
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


# Resolve repo root from this script's location (works local + cluster).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (os.path.join(REPO_ROOT, 'src'), REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
import connectome_gnn.utils as _cg_utils  # noqa: E402
from connectome_gnn.utils import add_pre_folder, config_path, log_path  # noqa: E402
try:
    from connectome_gnn.utils import set_data_root  # newer version
except ImportError:
    def set_data_root(path):  # fallback for older connectome_gnn
        _cg_utils._data_root = path


# Panels, in reading order, placed on a 2 row x 5 col GridSpec.
# (row, col_start, col_end_exclusive) - d and g span 2 cols (internally 2x1).
PANEL_LAYOUT = [
    ('a', 'weights_comparison_corrected.png',  0, 0, 1),
    ('b', 'embedding_{ci}.png',                0, 1, 2),
    ('c', 'f_theta_{ci}_domain.png',           0, 2, 4),   # double width
    ('d', 'embedding_augmented_{ci}.png',      0, 4, 5),
    ('e', 'tau_comparison_{ci}.png',           1, 0, 1),
    ('f', 'V_rest_comparison_{ci}.png',        1, 1, 2),
    ('g', 'g_phi_{ci}_domain.png',             1, 2, 4),   # double width
]


# Fonts (janne.matplotlibrc sets defaults to 8/6 pt; keep these as explicit
# override points so panel-specific tweaks are one-line edits).
FS_LABEL  = 8
FS_TICK   = 6
FS_ANNOT  = 6
FS_LEGEND = 6
FS_TYPE   = 6
PANEL_LBL = 8  # panel labels a) b) ... (Janne style: 8 pt, was 20 pt)

# ~18 cm document-width figure; height matches the original 22:9 aspect
# (~2.45) of the composite panel grid.
FIG_W_IN  = 18.0 * 0.3937       # ~7.09 in
FIG_H_IN  = FIG_W_IN * (9.0 / 22.0)  # ~2.90 in - keeps panel aspect ratios


def load_config(config_name):
    config_file, pre_folder = add_pre_folder(config_name)
    cfg = NeuralGraphConfig.from_yaml(config_path(f'{config_file}.yaml'))
    cfg.dataset = pre_folder + cfg.dataset
    cfg.config_file = pre_folder + config_name
    return cfg, pre_folder


def resolve_config_indices(cfg):
    """Replicates the config_indices convention used in plot_synaptic()."""
    base = os.path.basename(cfg.dataset)
    if 'flyvis_' in base:
        return base.split('flyvis_')[1]
    import re
    return re.sub(r'_\d{2}$', '', base)


def regenerate_panels(cfg, device):
    """Invoke GNN_PlotFigure.data_plot() to refresh every individual PNG."""
    # Import late so matplotlib 'Agg' is locked in first.
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


def panels_ready(results_dir, config_indices):
    missing = []
    for _, tmpl, *_ in PANEL_LAYOUT:
        fname = tmpl.format(ci=config_indices)
        if not os.path.exists(os.path.join(results_dir, fname)):
            missing.append(fname)
    return missing


def assemble(results_dir, config_indices, out_path):
    """Assemble the 2 row x 5 col composite preserving each panel's aspect.

    - Each axes' box is pinned to its image's native aspect via
      `ax.set_box_aspect`, so no internal whitespace remains inside the panel.
    - `constrained_layout=True` tightens inter-row/column gaps.
    - Panel labels (a), b), ...) are placed at the top-left of the outer
      panel box via `get_tightbbox`, all at the same y per row - matches the
      convention in `figures/INSTRUCTIONS.md` and the fig_davis_* scripts.
    """
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=5, wspace=0.04, hspace=0.04)

    panel_axes = []  # list of (ax, label, row)
    for label, tmpl, row, c0, c1 in PANEL_LAYOUT:
        ax = fig.add_subplot(gs[row, c0:c1])
        fname = tmpl.format(ci=config_indices)
        img_path = os.path.join(results_dir, fname)
        ax.set_axis_off()
        if not os.path.exists(img_path):
            ax.text(0.5, 0.5, f'missing:\n{fname}',
                    ha='center', va='center', fontsize=FS_ANNOT, color='red',
                    transform=ax.transAxes)
        else:
            img = mpimg.imread(img_path)
            h, w = img.shape[:2]
            ax.imshow(img, aspect='auto')
            ax.set_box_aspect(h / w)
        # Hide left spine to match the original (axis_off does this implicitly,
        # but be explicit so any reactivation also drops the left spine).
        ax.spines['left'].set_visible(False)
        _trim_axis(ax, yaxis=False)
        panel_axes.append((ax, f'{label}', row))

    # Place panel labels at top-left of each outer panel box; align per row.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bboxes = [ax.get_tightbbox(renderer) for ax, _, _ in panel_axes]
    # max top-y within each row so labels within a row share the same baseline.
    y_by_row = {}
    for (_, _, row), bb in zip(panel_axes, bboxes):
        y = inv.transform((bb.x0, bb.y1))[1]
        y_by_row[row] = max(y_by_row.get(row, -1), y)
    for (ax, lbl, row), bb in zip(panel_axes, bboxes):
        x0 = inv.transform((bb.x0, bb.y1))[0]
        fig.text(x0, y_by_row[row], lbl,
                 fontsize=PANEL_LBL, fontweight='bold',
                 va='bottom', ha='left', color='black',
                 transform=fig.transFigure)

    # PDF first per janne.matplotlibrc default; PNG for quick preview.
    fig.savefig(out_path + '.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_path + '.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'wrote {out_path}.{{pdf,png}}')


def assemble_combined(blocks, out_path):
    """Stack multiple condition blocks vertically in a single figure.

    blocks: list of dicts with keys {results_dir, config_indices, title}.

    Each block is a 2 x 5 panel sub-grid following PANEL_LAYOUT. A
    (non-bold) section title is placed above each block. Panel labels
    (a), b), ...) are re-generated per block so every block reads
    (a)-(g) on its own.
    """
    import matplotlib.gridspec as _mgs
    n_blocks = len(blocks)
    # Height per block (in inches) - keep the same 22:9 aspect as the
    # single-condition figure, scaled down to ~18 cm width.
    block_h   = FIG_W_IN * (9.0 / 22.0)
    title_pad = 0.35  # ~9 mm padding for the section title strip
    fig_h = n_blocks * (block_h + title_pad)
    fig_w = FIG_W_IN
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

    # Outer grid: one row per block, tall hspace for breathing room between
    # conditions (so the section title has a place to sit).
    outer = _mgs.GridSpec(n_blocks, 1, figure=fig,
                           left=0.04, right=0.98, top=0.98, bottom=0.02,
                           hspace=0.08)

    renderer = None
    section_info = []  # (gs_block, panel_axes, title, left_x)
    for k, blk in enumerate(blocks):
        inner = _mgs.GridSpecFromSubplotSpec(2, 5,
                                             subplot_spec=outer[k],
                                             wspace=0.04, hspace=0.04)
        panel_axes = []
        for label, tmpl, row, c0, c1 in PANEL_LAYOUT:
            ax = fig.add_subplot(inner[row, c0:c1])
            fname = tmpl.format(ci=blk['config_indices'])
            img_path = os.path.join(blk['results_dir'], fname)
            ax.set_axis_off()
            if not os.path.exists(img_path):
                ax.text(0.5, 0.5, f'missing:\n{fname}',
                        ha='center', va='center', fontsize=FS_ANNOT, color='red',
                        transform=ax.transAxes)
            else:
                img = mpimg.imread(img_path)
                h, w = img.shape[:2]
                ax.imshow(img, aspect='auto')
                ax.set_box_aspect(h / w)
            ax.spines['left'].set_visible(False)
            _trim_axis(ax, yaxis=False)
            panel_axes.append((ax, f'{label}', row))
        section_info.append((panel_axes, blk['title']))

    # Compute label and title positions after layout is finalised.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    for panel_axes, title in section_info:
        bboxes = [ax.get_tightbbox(renderer) for ax, _, _ in panel_axes]
        # per-row label y + min left x for this block
        y_by_row = {}
        for (_, _, row), bb in zip(panel_axes, bboxes):
            y = inv.transform((bb.x0, bb.y1))[1]
            y_by_row[row] = max(y_by_row.get(row, -1), y)
        # place panel labels per row
        for (ax, lbl, row), bb in zip(panel_axes, bboxes):
            x0 = inv.transform((bb.x0, bb.y1))[0]
            fig.text(x0, y_by_row[row], lbl,
                     fontsize=PANEL_LBL, fontweight='bold',
                     va='bottom', ha='left', color='black',
                     transform=fig.transFigure)
        # place section title just above the top-row panels of this block,
        # left-aligned with the leftmost panel
        top_row = 0
        top_axes = [(ax, bb) for (ax, _, r), bb in zip(panel_axes, bboxes) if r == top_row]
        if top_axes:
            bb_top = top_axes[0][1]
            x_left = inv.transform((bb_top.x0, bb_top.y1))[0]
            y_top  = max(inv.transform((bb.x0, bb.y1))[1] for _, bb in top_axes)
            fig.text(x_left, y_top + 0.018, title,
                     fontsize=FS_LABEL, fontweight='normal',
                     va='bottom', ha='left', color='black',
                     transform=fig.transFigure)

    # PDF first per janne.matplotlibrc default; PNG for quick preview.
    fig.savefig(out_path + '.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_path + '.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'wrote {out_path}.{{pdf,png}}')


def process_one(config_name, device, mode):
    cfg, _ = load_config(config_name)
    log_dir = log_path(cfg.config_file)
    results_dir = os.path.join(log_dir, 'results')
    ci = resolve_config_indices(cfg)
    print(f'\n=== {config_name} ===')
    print(f'log_dir: {log_dir}')
    print(f'config_indices: {ci}')
    print(f'mode: {mode}')

    if mode == 'regenerate':
        print('regenerating all panels via data_plot()')
        regenerate_panels(cfg, device)
    elif mode == 'load':
        missing = panels_ready(results_dir, ci)
        if missing:
            print(f'WARNING: {len(missing)} panel(s) missing under {results_dir}:')
            for m in missing:
                print(f'  - {m}')
            print('  (missing panels will render as red placeholders; '
                  'use --mode regenerate to produce them)')
        else:
            print('all panels present, loading from results folder')
    else:
        raise ValueError(f'unknown mode: {mode}')

    out_base = os.path.join(REPO_ROOT, 'figures', f'fig_gnn_params_{ci}')
    assemble(results_dir, ci, out_base)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', nargs='+',
                   default=['flyvis_noise_free', 'flyvis_noise_005', 'flyvis_noise_05'],
                   help='One or more config file names (without .yaml / pre-folder)')
    p.add_argument('--mode', choices=['regenerate', 'load'],
                   default='regenerate',
                   help='regenerate: run GNN_PlotFigure.data_plot() to '
                        'produce fresh panel PNGs. '
                        'load: skip data_plot() and assemble composite from '
                        'the PNGs already under <log_dir>/results/. '
                        '(default: regenerate)')
    p.add_argument('--device', default=None,
                   help='torch device (default: cuda:0 if available, else cpu)')
    p.add_argument('--output_root', default=None,
                   help='Data root (same as GNN_Main.py --output_root). '
                        'Sets log/<config> under this path. '
                        'e.g. /groups/saalfeld/home/allierc/GraphData')
    p.add_argument('--combined', action='store_true',
                   help='Stack all --configs into a single figure with a '
                        '(non-bold) section title per block instead of '
                        'writing one composite per config.')
    p.add_argument('--combined-titles', nargs='+', default=None,
                   help='Optional section titles (one per --configs entry) '
                        'for the --combined output. Defaults are generated '
                        'from the config name.')
    p.add_argument('--combined-out', default='fig_gnn_params_combined',
                   help='Basename (no extension) of the combined output '
                        'file under figures/. Default: fig_gnn_params_combined')
    args = p.parse_args()

    if args.output_root:
        assert os.path.isdir(args.output_root), \
            f'--output_root does not exist: {args.output_root}'
        set_data_root(args.output_root)
        print(f'output_root: {args.output_root}')

    if args.device is None:
        import torch
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device: {args.device}')
    print(f'configs: {args.configs}')

    if args.combined:
        # Regenerate or verify each config's panels first, then assemble one
        # stacked composite. Re-uses process_one's load/regenerate logic but
        # skips the per-config assemble() step by handing blocks straight to
        # assemble_combined().
        blocks = []
        for k, cfg_name in enumerate(args.configs):
            cfg, _ = load_config(cfg_name)
            log_dir = log_path(cfg.config_file)
            results_dir = os.path.join(log_dir, 'results')
            ci = resolve_config_indices(cfg)
            print(f'\n=== {cfg_name} ({ci}) ===')
            if args.mode == 'regenerate':
                print('regenerating all panels via data_plot()')
                regenerate_panels(cfg, args.device)
            elif args.mode == 'load':
                missing = panels_ready(results_dir, ci)
                if missing:
                    print(f'WARNING: {len(missing)} panel(s) missing for {cfg_name}')
            title = (args.combined_titles[k]
                     if args.combined_titles and k < len(args.combined_titles)
                     else cfg_name)
            blocks.append({
                'results_dir':     results_dir,
                'config_indices':  ci,
                'title':           title,
            })
        out_base = os.path.join(REPO_ROOT, 'figures', args.combined_out)
        assemble_combined(blocks, out_base)
    else:
        for cfg_name in args.configs:
            process_one(cfg_name, args.device, args.mode)


if __name__ == '__main__':
    main()


# # regenerate everything (default)
# python figures/fig_gnn_params.py \
#     --output_root /groups/saalfeld/home/allierc/GraphData \
#     --configs flyvis_noise_free_winner flyvis_noise_005_winner flyvis_noise_05_winner

# # just assemble the composite from existing PNGs in <log_dir>/results/
# python figures/fig_gnn_params.py --mode load \
#     --output_root /groups/saalfeld/home/allierc/GraphData \
#     --configs flyvis_noise_free_winner flyvis_noise_005_winner flyvis_noise_05_winner
