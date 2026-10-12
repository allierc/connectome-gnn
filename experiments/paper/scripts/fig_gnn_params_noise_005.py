"""Fig. 1: circuit-parameter extraction of the general-form GNN (group lasso 25) on
Flyvis-217 at model noise sigma = 0.05, fold cv00 (experiment 2).

    python scripts/fig_gnn_params_noise_005.py [--masked]

--masked: the same panels from the -o test_plot pass with recovery.score_mask set to the silent
edges and neurons of flyvis 000 at EPE 5.3 (GraphData/flyvis/silent_mask_000_epe5.3.npz,
tools/task_pruning.py): W and the per-edge message over kept edges, embedding, tau and V_rest
over kept neurons -> figures/fig_gnn_params_noise_005_masked.{pdf,png}

One 3 x 2 block, the published figure's template: W | embedding, then the aggregated
message m_i | the per-edge message, then V_rest | tau; letters row-major (a-f).
The three noise levels side by side: fig_gnn_params_noise_sweep.py.
Data: <GNN_OUTPUT_ROOT>/log/fly/flyvis_noise_005_blank50_condl25_cv00/
(results/panels_*.npz, models/template_fit_alt.pt, results/metrics.txt).

Output: figures/fig_gnn_params_noise_005.{pdf,png}
"""
import os
import sys

import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import CM, FIG_DIR, panel_labels, save, type_cmap  # noqa: E402
from params_panels import draw_block, draw_message_row, load_run  # noqa: E402

RUN = "flyvis_noise_005_blank50_condl25_cv00"
PANEL_CM, GAP_IN_CM, GAP_ROW_CM = 1.95, 1.35, 1.4   # the published figure's spacing
MARGIN_L_CM, MARGIN_B_CM = 0.9, 0.75


def main(masked=False):
    width = MARGIN_L_CM + 2 * PANEL_CM + GAP_IN_CM + 0.3
    height = MARGIN_B_CM + 3 * PANEL_CM + 2 * GAP_ROW_CM + 0.6
    fig = plt.figure(figsize=(width * CM, height * CM))
    gs = mgs.GridSpec(3, 2, figure=fig, left=MARGIN_L_CM / width, right=(MARGIN_L_CM + 2 * PANEL_CM + GAP_IN_CM) / width,
                      bottom=MARGIN_B_CM / height, top=(MARGIN_B_CM + 3 * PANEL_CM + 2 * GAP_ROW_CM) / height,
                      wspace=GAP_IN_CM / PANEL_CM, hspace=GAP_ROW_CM / PANEL_CM)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
    data = load_run(RUN, masked=masked)
    draw_block((axes[0], axes[1], axes[4], axes[5]), data, cmap=type_cmap())
    draw_message_row((axes[2], axes[3]), data)
    panel_labels(fig, axes, dy_pt=4)
    save(fig, os.path.join(FIG_DIR, "fig_gnn_params_noise_005" + ("_masked" if masked else "")))


if __name__ == "__main__":
    main(masked="--masked" in sys.argv)
