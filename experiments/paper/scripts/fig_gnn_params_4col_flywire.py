"""Supp. Fig.: circuit-parameter extraction of the general-form GNN (group
lasso 25) on the four FlyWire connectome variants, fold cv00 (experiment 7).

    python scripts/fig_gnn_params_4col_flywire.py

One row per variant (e8 hybrid, e8 hybrid + n.e., FlyWire eye, FlyWire eye + n.e.), six panels each:
W, tau, V_rest, the embedding, and the embedding again coloured by its true tau and by its true V_rest
(params_panels.draw_colored_embedding, as Fig. 1 e, f); letters run row-major.
Data: <GNN_OUTPUT_ROOT>/log/fly/<variant>_blank50_condl25_cv00/.

Output: figures/fig_gnn_params_4col_flywire_comparison.{pdf,png}
"""
import os
import sys

import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import CM, FIG_DIR, column_titles, panel_labels, save, type_cmap  # noqa: E402
from params_panels import draw_block, draw_colored_embedding, load_run  # noqa: E402

COLS = [
    ("e8 hybrid", "e8_flywireRF_noise_005_blank50_condl25_cv00"),
    ("e8 hybrid + n.e.", "e8_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv00"),
    ("FlyWire eye", "full_eye_flywireRF_noise_005_blank50_condl25_cv00"),
    ("FlyWire eye + n.e.", "full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv00"),
]
PANEL_CM, GAP_CM = 1.9, 1.1
MARGIN_L_CM, MARGIN_B_CM = 1.1, 0.8
N_COL = 6


def main():
    # one row per variant: W, tau, V_rest, the embedding, the embedding by true tau, by true V_rest
    n = len(COLS)
    width = MARGIN_L_CM + N_COL * PANEL_CM + (N_COL - 1) * GAP_CM + 0.3
    height = MARGIN_B_CM + n * PANEL_CM + (n - 1) * GAP_CM + 0.3
    fig = plt.figure(figsize=(width * CM, height * CM))
    gs = mgs.GridSpec(n, N_COL, figure=fig, left=MARGIN_L_CM / width, right=1 - 0.3 / width,
                      bottom=MARGIN_B_CM / height, top=1 - 0.3 / height,
                      wspace=GAP_CM / PANEL_CM, hspace=GAP_CM / PANEL_CM)
    cmap = type_cmap()
    axes = [[fig.add_subplot(gs[r, c]) for c in range(N_COL)] for r in range(n)]
    for r, (title, run) in enumerate(COLS):
        data = load_run(run)
        draw_block((axes[r][0], axes[r][3], axes[r][2], axes[r][1]), data, cmap=cmap)
        draw_colored_embedding(fig, (axes[r][4], axes[r][5]), data, axes[r][3])
        axes[r][0].text(-0.66, 0.5, title.replace(" + n.e.", "\n+ n.e."), transform=axes[r][0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=7.5, linespacing=1.0)
    panel_labels(fig, [a for row in axes for a in row], dy_pt=4)
    save(fig, os.path.join(FIG_DIR, "fig_gnn_params_4col_flywire_comparison"))


if __name__ == "__main__":
    main()
