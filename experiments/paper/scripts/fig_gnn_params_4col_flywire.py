"""Supp. Fig.: circuit-parameter extraction of the general-form GNN (group
lasso 25) on the four FlyWire connectome variants, fold cv00 (experiment 7).

    python scripts/fig_gnn_params_4col_flywire.py

Four columns (e8 hybrid, e8 hybrid + n.e., FlyWire eye, FlyWire eye + n.e.),
four rows (W, embedding, V_rest, tau); letters run row-major.
Data: <GNN_OUTPUT_ROOT>/log/fly/<variant>_blank50_condl25_cv00/.

Output: figures/fig_gnn_params_4col_flywire_comparison.{pdf,png}
"""
import os
import sys

import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import CM, FIG_DIR, column_titles, panel_labels, save, type_cmap  # noqa: E402
from params_panels import draw_block, load_run  # noqa: E402

COLS = [
    ("e8 hybrid", "e8_flywireRF_noise_005_blank50_condl25_cv00"),
    ("e8 hybrid + n.e.", "e8_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv00"),
    ("FlyWire eye", "full_eye_flywireRF_noise_005_blank50_condl25_cv00"),
    ("FlyWire eye + n.e.", "full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv00"),
]
PANEL_CM, GAP_CM = 2.9, 1.5
MARGIN_L_CM, MARGIN_B_CM = 1.1, 0.8


def main():
    # one row per variant: W, tau, V_rest, the embedding
    n = len(COLS)
    width = MARGIN_L_CM + 4 * PANEL_CM + 3 * GAP_CM + 0.3
    height = MARGIN_B_CM + n * PANEL_CM + (n - 1) * GAP_CM + 0.3
    fig = plt.figure(figsize=(width * CM, height * CM))
    gs = mgs.GridSpec(n, 4, figure=fig, left=MARGIN_L_CM / width, right=1 - 0.3 / width,
                      bottom=MARGIN_B_CM / height, top=1 - 0.3 / height,
                      wspace=GAP_CM / PANEL_CM, hspace=GAP_CM / PANEL_CM)
    cmap = type_cmap()
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(n)]
    for r, (title, run) in enumerate(COLS):
        draw_block((axes[r][0], axes[r][3], axes[r][2], axes[r][1]), load_run(run), cmap=cmap)
        axes[r][0].text(-0.62, 0.5, title, transform=axes[r][0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=9)
    panel_labels(fig, [a for row in axes for a in row], dy_pt=4)
    save(fig, os.path.join(FIG_DIR, "fig_gnn_params_4col_flywire_comparison"))


if __name__ == "__main__":
    main()
