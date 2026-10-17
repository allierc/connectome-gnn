"""Fig. 1: circuit-parameter extraction of the general-form GNN (group lasso 25) on
Flyvis-217 at model noise sigma = 0.05, fold cv00 (experiment 2).

    python scripts/fig_gnn_params_noise_005.py [run] [--masked] [--counts] [--proposal]

run: another trained run (default the cv00 one below); its fold tag is appended to the output name

--counts: the number of edges / neurons each panel uses, above its top-right corner (the
slides; the paper's figure has none) -> <name>_counts.{pdf,png}

--masked: the same panels from the -o test_plot pass with recovery.score_mask set to the silent
edges and neurons of flyvis 000 at EPE 5.3 (GraphData/flyvis/silent_mask_000_epe5.3.npz,
tools/task_pruning.py): W and the per-edge message over kept edges, embedding, tau and V_rest
over kept neurons -> figures/fig_gnn_params_noise_005_masked.{pdf,png}

Two rows (the paper's figure; --masked keeps the first row only):
a W, b tau, c V_rest, d the embedding; e, f every neuron of d on d's axes, coloured by its true tau (log
scale) and by its true V_rest (params_panels.draw_colored_embedding), under a and b.

--proposal: the meeting deck's version of the second row, with g, h next to e, f: the sensitivity
mean |d out/d x| x SD(x) of f_theta and of g_phi to each input, from their Jacobians on sampled frames
(tools/embedding_jacobian.py, results/extras/embedding_jacobian.npz) -> <name>_proposal.{pdf,png}
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
import numpy as np  # noqa: E402

from paper_style import CM, FIG_DIR, FS_ANNOT, FS_TICK, LOG_ROOT, SHOW_COUNTS, panel_labels, save, type_cmap  # noqa: E402
from params_panels import draw_block, draw_colored_embedding, load_run  # noqa: E402

RUN = "flyvis_noise_005_blank50_condl25_cv00"
PANEL_CM, GAP_IN_CM, GAP_ROW_CM = 1.95, 1.35, 1.4   # the published figure's spacing
MARGIN_L_CM, MARGIN_B_CM = 0.9, 0.75


def jacobian_row(fig, axes, data, ax_d, run=RUN):
    """e, f: the coloured embeddings; g, h the sensitivity mean |d out/d x| x SD(x) of f_theta and of
    g_phi to each input (the deck's --proposal figure)."""
    import matplotlib.ticker as mtick
    p = data["panels"]
    jz = np.load(os.path.join(LOG_ROOT, run, "results", "extras", "embedding_jacobian.npz"))
    assert np.array_equal(np.asarray(jz["type_ids"]).astype(int), np.asarray(p["type_ids"]).astype(int)), \
        "Jacobian file from another run"
    e, f, g, h = axes
    draw_colored_embedding(fig, (e, f), data, ax_d)
    lab = {"v_i": "$v_i$", "a_i0": "$a_{i0}$", "a_i1": "$a_{i1}$", "msg_i": "$m_i$", "I_i": "$I_i$",
           "v_j": "$v_j$", "a_j0": "$a_{j0}$", "a_j1": "$a_{j1}$"}
    for ax, key, ylab in ((g, "f", "$f_\\theta$ sensitivity"), (h, "g", "$g_\\phi$ sensitivity")):
        names = [str(x) for x in jz[f"{key}_names"]]
        ax.bar(range(len(names)), jz[f"{key}_sensitivity"], color="0.3", width=0.7)
        ax.set_yscale("log"); ax.set_xticks(range(len(names)))
        ax.set_xticklabels([lab[n] for n in names], fontsize=FS_TICK)
        ax.set_ylabel(ylab)
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)


def main(masked=False, counts=False, proposal=False, run=RUN):
    SHOW_COUNTS["on"] = counts
    rows = 1 if masked else 2
    width = MARGIN_L_CM + 4 * PANEL_CM + 3 * GAP_IN_CM + 0.3
    height = MARGIN_B_CM + rows * PANEL_CM + (rows - 1) * GAP_ROW_CM + 0.6
    fig = plt.figure(figsize=(width * CM, height * CM))
    left, right = MARGIN_L_CM / width, (MARGIN_L_CM + 4 * PANEL_CM + 3 * GAP_IN_CM) / width
    y_top = MARGIN_B_CM + (rows - 1) * (PANEL_CM + GAP_ROW_CM)          # bottom edge of the first row, cm
    gs = mgs.GridSpec(1, 4, figure=fig, left=left, right=right, bottom=y_top / height,
                      top=(y_top + PANEL_CM) / height, wspace=GAP_IN_CM / PANEL_CM)
    axes = [fig.add_subplot(gs[0, c]) for c in range(4)]
    data = load_run(run, masked=masked)
    # a W, b tau, c V_rest, d the embedding (draw_block takes W, embedding, V_rest, tau)
    draw_block((axes[0], axes[3], axes[2], axes[1]), data, cmap=type_cmap())
    if rows == 2:
        # e, f (and, with --proposal, g, h) under a, b, c, d: the first row's columns
        gs2 = mgs.GridSpec(1, 4, figure=fig, left=left, right=right, bottom=MARGIN_B_CM / height,
                           top=(MARGIN_B_CM + PANEL_CM) / height, wspace=GAP_IN_CM / PANEL_CM)
        row2 = [fig.add_subplot(gs2[0, c]) for c in range(4 if proposal else 2)]
        if proposal:
            jacobian_row(fig, row2, data, axes[3], run)
        else:
            draw_colored_embedding(fig, row2, data, axes[3])
        axes = axes + row2
    panel_labels(fig, axes, dy_pt=4)
    save(fig, os.path.join(FIG_DIR, "fig_gnn_params_noise_005" + ("_masked" if masked else "")
                           + ("_counts" if counts else "") + ("_proposal" if proposal else "")
                           + ("" if run == RUN else "_" + run.split("_")[-1])))


if __name__ == "__main__":
    _run = next((a for a in sys.argv[1:] if not a.startswith("--")), RUN)   # another run: its fold tag is appended
    main(masked="--masked" in sys.argv, counts="--counts" in sys.argv, proposal="--proposal" in sys.argv, run=_run)
