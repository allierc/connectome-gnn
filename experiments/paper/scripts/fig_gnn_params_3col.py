"""Fig. 1: circuit-parameter extraction of the general-form GNN (group lasso
25) on Flyvis-217 at the three model-noise levels, fold cv00 (experiment 2).

    python scripts/fig_gnn_params_3col.py

Three blocks (noise-free, sigma 0.05, sigma 0.5), each 2 x 2: W | embedding
on the top row, V_rest | tau on the bottom row; letters row-major (a-f, g-l).
Data: <GNN_OUTPUT_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_condl25_cv00/
(results/panels_*.npz, models/template_fit_alt.pt, results/metrics.txt).

Output: figures/fig_gnn_params_3col_noise_comparison.{pdf,png}
"""
import os
import sys

import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import CM, FIG_DIR, column_titles, panel_labels, save, type_cmap  # noqa: E402
from params_panels import draw_block, load_run  # noqa: E402

BLOCKS = [
    ("noise-free ($\\sigma = 0$)", "flyvis_noise_free_blank50_condl25_cv00"),
    ("low model noise ($\\sigma = 0.05$)", "flyvis_noise_005_blank50_condl25_cv00"),
    ("high model noise ($\\sigma = 0.5$)", "flyvis_noise_05_blank50_condl25_cv00"),
]
PANEL_CM, GAP_IN_CM, GAP_BLOCK_CM, GAP_ROW_CM = 1.95, 1.35, 1.35, 1.4   # the published figure's even spacing
MARGIN_L_CM, MARGIN_B_CM = 0.9, 0.75


def main():
    n_blocks = len(BLOCKS)
    width = MARGIN_L_CM + n_blocks * (2 * PANEL_CM + GAP_IN_CM) + (n_blocks - 1) * GAP_BLOCK_CM + 0.3
    height = MARGIN_B_CM + 2 * PANEL_CM + GAP_ROW_CM + 1.0
    fig = plt.figure(figsize=(width * CM, height * CM))
    cmap = type_cmap()
    rows = [[], []]
    blocks_axes = []
    for k, (title, run) in enumerate(BLOCKS):
        x0 = (MARGIN_L_CM + k * (2 * PANEL_CM + GAP_IN_CM + GAP_BLOCK_CM)) / width
        y0 = MARGIN_B_CM / height
        gs = mgs.GridSpec(2, 2, figure=fig, left=x0, right=x0 + (2 * PANEL_CM + GAP_IN_CM) / width,
                          bottom=y0, top=y0 + (2 * PANEL_CM + GAP_ROW_CM) / height,
                          wspace=GAP_IN_CM / PANEL_CM, hspace=GAP_ROW_CM / PANEL_CM)
        axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
        draw_block((axes[0], axes[1], axes[2], axes[3]), load_run(run), cmap=cmap)
        rows[0] += axes[:2]; rows[1] += axes[2:]
        blocks_axes.append(axes)
    column_titles(fig, [b[:2] for b in blocks_axes], [t for t, _ in BLOCKS], dy_pt=16, fontsize=9)
    panel_labels(fig, rows[0] + rows[1], dy_pt=4)
    save(fig, os.path.join(FIG_DIR, "fig_gnn_params_3col_noise_comparison"))


if __name__ == "__main__":
    main()
