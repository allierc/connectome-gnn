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
PANEL_CM, GAP_CM = 3.0, 1.1
MARGIN_L_CM, MARGIN_B_CM = 1.1, 0.8


def main():
    n = len(COLS)
    width = MARGIN_L_CM + n * PANEL_CM + (n - 1) * GAP_CM + 0.3
    height = MARGIN_B_CM + 4 * PANEL_CM + 3 * GAP_CM + 0.6
    fig = plt.figure(figsize=(width * CM, height * CM))
    gs = mgs.GridSpec(4, n, figure=fig, left=MARGIN_L_CM / width, right=1 - 0.3 / width,
                      bottom=MARGIN_B_CM / height, top=1 - 0.6 / height,
                      wspace=GAP_CM / PANEL_CM, hspace=GAP_CM / PANEL_CM)
    cmap = type_cmap()
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n)] for r in range(4)]
    for c, (title, run) in enumerate(COLS):
        draw_block((axes[0][c], axes[1][c], axes[2][c], axes[3][c]), load_run(run), cmap=cmap)
    column_titles(fig, [[axes[0][c]] for c in range(n)], [t for t, _ in COLS], dy_pt=14)
    panel_labels(fig, [a for row in axes for a in row], dy_pt=3)
    save(fig, os.path.join(FIG_DIR, "fig_gnn_params_4col_flywire_comparison"))


if __name__ == "__main__":
    main()
