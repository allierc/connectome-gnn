"""Supp. Fig. (Known-ODE parameter recovery): the Known-ODE models on Flyvis-217
at the three model-noise levels, fold cv00, re-analysed by experiment 15 (the
published May 2026 checkpoints scored with the current -o test_plot).

    python scripts/fig_known_ode_params_3col.py

Three blocks (noise-free, sigma 0.05, sigma 0.5), each 2 x 2 as the published
figure: W | cell-type map, then V_rest | tau; letters row-major (a-f, g-l).

    W         learned vs true synaptic weight, all 434,112 edges (the Known-ODE
              learns W directly, so every edge is scored: metrics Wij_n)
    map       a 2-D UMAP (uniform manifold approximation and projection) of the
              per-neuron feature stack the clustering metric is computed on --
              learned tau, V_rest and eight statistics of the learned weights
              around the neuron (mean/std/min/max of incoming and outgoing),
              standardised -- with the UMAP settings of GNN_PlotFigure
              (n_neighbors 15, min_dist 0.1, random_state 42); coloured by TRUE
              cell type (the published panel coloured by fitted GMM cluster);
              the stack is rebuilt by metrics.cluster_recovery and its accuracy
              is checked against metrics.txt clustering_accuracy
    V_rest    learned vs true, outliers |dV| > 0.2 in red, as Fig. 1
    tau       learned vs true, outliers |dtau| > 0.1 s in red, as Fig. 1

Data: <GNN_OUTPUT_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_kode217_cv00/
(results/extras/recovered_pairs.npz, results/rollout_bundle.npz for the cell
types, training_edges.pt, results/metrics.txt).

Output: figures/fig_known_ode_params_3col_noise_comparison_new.{pdf,png}
"""
import json
import os
import sys
import warnings

import matplotlib.gridspec as mgs
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# this repository's connectome_gnn, not whichever worktree the env has installed (cluster_recovery lives here)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
from paper_style import CM, FIG_DIR, LOG_ROOT, column_titles, panel_labels, read_metrics, save, ticks3, type_cmap  # noqa: E402
from params_panels import THR, draw_block  # noqa: E402

BLOCKS = [
    ("noise-free ($\\sigma = 0$)", "flyvis_noise_free_blank50_kode217_cv00"),
    ("low model noise ($\\sigma = 0.05$)", "flyvis_noise_005_blank50_kode217_cv00"),
    ("high model noise ($\\sigma = 0.5$)", "flyvis_noise_05_blank50_kode217_cv00"),
]
PANEL_CM, GAP_IN_CM, GAP_BLOCK_CM, GAP_ROW_CM = 1.95, 1.35, 1.35, 1.4   # Fig. 1's spacing
MARGIN_L_CM, MARGIN_B_CM = 0.9, 0.75


def _umap(X):
    import umap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(X)


def load_kode(run):
    from connectome_gnn.metrics import cluster_recovery
    d = os.path.join(LOG_ROOT, run)
    m = read_metrics(d)
    rp = np.load(os.path.join(d, "results", "extras", "recovered_pairs.npz"))
    types = np.asarray(np.load(os.path.join(d, "results", "rollout_bundle.npz"), allow_pickle=True)["type_ids"])
    edges = np.asarray(torch.load(os.path.join(d, "training_edges.pt"), map_location="cpu", weights_only=False))
    W_t, W_l = rp["W_true"].astype(np.float64), rp["W_learned"].astype(np.float64)
    tau_t, tau_l = rp["tau_true"].astype(np.float64), rp["tau_learned"].astype(np.float64)
    V_t, V_l = rp["V_rest_true"].astype(np.float64), rp["V_rest_learned"].astype(np.float64)
    n = len(tau_t)
    # every annotated number is the metrics file's; the arrays drawn must reproduce it
    r2w = 1 - np.mean((W_l - W_t) ** 2) / np.var(W_t)
    assert len(W_t) == m["Wij_n"] and abs(r2w - m["Wij_R2"]) < 5e-3, (run, len(W_t), r2w, m["Wij_R2"])
    for name, t, l in (("tau", tau_t, tau_l), ("V_rest", V_t, V_l)):
        ok = np.abs(l - t) <= THR[name]
        r2 = 1 - np.mean((l[ok] - t[ok]) ** 2) / np.var(t[ok])
        assert abs(r2 - m[f"{name}_R2"]) < 5e-3 and abs(100 * (~ok).mean() - m[f"{name}_pct_outliers"]) < 0.2, \
            (run, name, r2, m[f"{name}_R2"])
    cl = cluster_recovery(types, edges, W_l, n, learned_tau=tau_l, learned_vrest=V_l,
                          n_components=min(100, n - 1), return_features=True)
    assert abs(cl["clustering_accuracy"] - m["clustering_accuracy"]) < 0.01, \
        (run, cl["clustering_accuracy"], m["clustering_accuracy"])
    print(f"{run}: W R2 {r2w:.3f} over {len(W_t)} edges, clustering accuracy {cl['clustering_accuracy']:.3f} "
          f"(metrics {m['clustering_accuracy']:.3f}), "
          f"{int(((W_t < -1) | (W_t > 2) | (W_l < -1) | (W_l > 2)).sum())} edges outside the [-1, 2] axes")
    n_out = int(((W_t < -1) | (W_t > 2) | (W_l < -1) | (W_l > 2)).sum())
    return dict(panels={"a": _umap(cl["_X"]), "type_ids": types}, metrics=m, W_true=W_t, W_learned=W_l,
                tau_true=tau_t, tau=tau_l, V_true=V_t, V=V_l, n_outside=n_out)


def main():
    n_blocks = len(BLOCKS)
    width = MARGIN_L_CM + n_blocks * (2 * PANEL_CM + GAP_IN_CM) + (n_blocks - 1) * GAP_BLOCK_CM + 0.3
    height = MARGIN_B_CM + 2 * PANEL_CM + GAP_ROW_CM + 1.0
    fig = plt.figure(figsize=(width * CM, height * CM))
    cmap = type_cmap()
    rows = [[], []]
    blocks_axes = []
    n_outside = []
    for k, (title, run) in enumerate(BLOCKS):
        x0 = (MARGIN_L_CM + k * (2 * PANEL_CM + GAP_IN_CM + GAP_BLOCK_CM)) / width
        y0 = MARGIN_B_CM / height
        gs = mgs.GridSpec(2, 2, figure=fig, left=x0, right=x0 + (2 * PANEL_CM + GAP_IN_CM) / width,
                          bottom=y0, top=y0 + (2 * PANEL_CM + GAP_ROW_CM) / height,
                          wspace=GAP_IN_CM / PANEL_CM, hspace=GAP_ROW_CM / PANEL_CM)
        axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
        data = load_kode(run)
        n_outside.append(data["n_outside"]); n_edges = len(data["W_true"])
        draw_block(axes, data, cmap=cmap)
        # the map's axes: symmetric about 0 in whole UMAP units, ticks -L, 0, L
        a = data["panels"]["a"]
        L = float(np.ceil(np.abs(a).max() + 1))
        ticks3(axes[1], -L, L)
        axes[1].set_xlabel("UMAP$_1$"); axes[1].set_ylabel("UMAP$_2$")
        # V_rest / tau: the R2 box stays top-left as in Fig. 1, drawn over the points with a thin white halo --
        # at sigma 0 and 0.05 the cluster at true V_rest 0.5-0.8 reaches under it (the lower-right corner,
        # tried, crosses the identity and the dotted outlier line instead)
        for ax in axes[2:]:
            for t in ax.texts:
                t.set_zorder(5); t.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])
        rows[0] += axes[:2]; rows[1] += axes[2:]
        blocks_axes.append(axes)
    column_titles(fig, [b[:2] for b in blocks_axes], [t for t, _ in BLOCKS], dy_pt=16, fontsize=9)
    panel_labels(fig, rows[0] + rows[1], dy_pt=4)
    save(fig, os.path.join(FIG_DIR, "fig_known_ode_params_3col_noise_comparison_new"))
    # the caption's clipping count (edit_tex.py reads it): the most edges any column puts off the W axes
    json.dump({"n_outside": max(n_outside), "n_edges": int(n_edges)},
              open(os.path.join(FIG_DIR, "fig_known_ode_params_3col_noise_comparison_new.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
