"""Noise sweep (Fig. 1 until 2026-10-12): circuit-parameter extraction of the general-form GNN
(group lasso 25) on Flyvis-217 at the three model-noise levels, fold cv00 (experiment 2).
Fig. 1 is now the sigma = 0.05 block alone (fig_gnn_params_noise_005.py).

    python scripts/fig_gnn_params_noise_sweep.py

Three blocks (noise-free, sigma 0.05, sigma 0.5), each 3 x 2 as the published
figure: W | embedding, then the aggregated message m_i | the per-edge message
(the general-form counterparts of f_theta and g_phi), then V_rest | tau;
letters row-major (a-f, g-l, m-r).
Data: <GNN_OUTPUT_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_condl25_cv00/
(results/panels_*.npz, models/template_fit_alt.pt, results/metrics.txt).

Output: figures/fig_gnn_params_noise_sweep.{pdf,png}
"""
import os
import sys

import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import CM, FIG_DIR, column_titles, panel_labels, save, type_cmap  # noqa: E402
from params_panels import draw_block, draw_message_row, load_run  # noqa: E402

BLOCKS = [
    ("noise-free ($\\sigma = 0$)", "flyvis_noise_free_blank50_condl25_cv00"),
    ("low model noise ($\\sigma = 0.05$)", "flyvis_noise_005_blank50_condl25_cv00"),
    ("high model noise ($\\sigma = 0.5$)", "flyvis_noise_05_blank50_condl25_cv00"),
]
PANEL_CM, GAP_IN_CM, GAP_BLOCK_CM, GAP_ROW_CM = 1.95, 1.35, 1.35, 1.4   # the published figure's even spacing
MARGIN_L_CM, MARGIN_B_CM = 0.9, 0.75


def main():
    # one row per noise level: W, tau, V_rest, the embedding
    n = len(BLOCKS)
    width = MARGIN_L_CM + 0.6 + 4 * PANEL_CM + 3 * GAP_IN_CM + 0.3
    height = MARGIN_B_CM + n * PANEL_CM + (n - 1) * GAP_ROW_CM + 0.6
    fig = plt.figure(figsize=(width * CM, height * CM))
    gs = mgs.GridSpec(n, 4, figure=fig, left=(MARGIN_L_CM + 0.6) / width, right=1 - 0.3 / width,
                      bottom=MARGIN_B_CM / height, top=1 - 0.6 / height,
                      wspace=GAP_IN_CM / PANEL_CM, hspace=GAP_ROW_CM / PANEL_CM)
    cmap = type_cmap()
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(n)]
    for r, (title, run) in enumerate(BLOCKS):
        draw_block((axes[r][0], axes[r][3], axes[r][2], axes[r][1]), load_run(run), cmap=cmap)
        axes[r][0].text(-0.75, 0.5, title, transform=axes[r][0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=8)
    panel_labels(fig, [a for row in axes for a in row], dy_pt=4)
    save(fig, os.path.join(FIG_DIR, "fig_gnn_params_noise_sweep"))


if __name__ == "__main__":
    main()
