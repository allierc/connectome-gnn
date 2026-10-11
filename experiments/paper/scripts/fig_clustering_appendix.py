"""Supp. Fig.: cell-type clusterability of the general-form GNN's learned
quantities (group lasso 25, sigma = 0.05, fold cv00; experiment 2), in the
published 2 x 2 layout.

    python scripts/fig_clustering_appendix.py

    a  true (tau, V_rest, W-stats): ground-truth tau, V_rest and 8 per-neuron
       statistics of the true W (mean, SD, min, max of incoming and outgoing)
    b  learned (tau, V_rest, W-stats): the same from the learned quantities
    c  learned a_i: the 2-D embedding, shown directly
    d  learned (a_i, tau, V_rest, W-stats): the stack clustering_accuracy is on

Everything printed is -o test_plot's: the plot pass clusters each stack
(metrics.cluster_recovery / cluster_recovery_variants: a 100-component Gaussian
mixture, Hungarian-matched accuracy, ARI, NMI) and writes the scores to
metrics.txt (clustering_gt_*, clustering_params_*, clustering_emb_*,
clustering_*) and the stacks to results/extras/clustering_features.npz. This
script only projects those stacks to 2-D with UMAP for display.

Output: figures/fig_clustering_appendix.{pdf,png}
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import CM, FIG_DIR, FS_LABEL, FS_TICK, LOG_ROOT, panel_labels, read_metrics, save, type_cmap  # noqa: E402

RUN = "flyvis_noise_005_blank50_condl25_cv00"
SEED = 42
PANELS = [   # (title, stack in clustering_features.npz, metrics prefix, UMAP or shown directly)
    ("true ($\\tau$, $V_{rest}$, $W$-stats)", "X_gt", "clustering_gt_", True),
    ("learned ($\\tau$, $V_{rest}$, $W$-stats)", "X_params", "clustering_params_", True),
    ("learned $\\mathbf{a}_i$", "X_emb", "clustering_emb_", False),
    ("learned ($\\mathbf{a}_i$, $\\tau$, $V_{rest}$, $W$-stats)", "X_full", "clustering_", True),
]


def umap2(features):
    import umap
    z = (features - features.mean(0)) / (features.std(0) + 1e-9)
    return umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1).fit_transform(z)


def main():
    d = os.path.join(LOG_ROOT, RUN)
    f = np.load(os.path.join(d, "results", "extras", "clustering_features.npz"))
    m = read_metrics(d)
    types = f["type_ids"].astype(int)
    cmap = type_cmap()
    side, gap, margin = 5.5, 1.6, 0.9
    w = h = 2 * side + gap + 2 * margin
    fig = plt.figure(figsize=(w * CM, h * CM))
    axes = []
    for k, (title, key, pre, do_umap) in enumerate(PANELS):
        r, c = divmod(k, 2)
        ax = fig.add_axes([(margin + c * (side + gap)) / w, (margin + (1 - r) * (side + gap)) / h, side / w, side / h])
        axes.append(ax)
        X = f[key]
        acc, ari, nmi = m[pre + "accuracy"], m[pre + "ari"], m[pre + "nmi"]
        assert int(m[pre + "n_features"]) == X.shape[1], (key, m[pre + "n_features"], X.shape)
        xy = umap2(X) if do_umap else X
        ax.scatter(xy[:, 0], xy[:, 1], c=types, cmap=cmap, s=2.0, alpha=0.8, lw=0, rasterized=True)
        if not do_umap:
            for lim, v in ((ax.set_xlim, xy[:, 0]), (ax.set_ylim, xy[:, 1])):
                lo, hi = np.percentile(v, [0.5, 99.5]); dd = 0.05 * (hi - lo); lim(lo - dd, hi + dd)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=FS_LABEL, pad=4)
        ax.text(0.03, 0.97, f"GMM accuracy = {acc:.2f}\nARI = {ari:.2f}, NMI = {nmi:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=FS_TICK, linespacing=1.4)
        ax.set_xlabel("UMAP$_1$" if do_umap else "$a_{i0}$"); ax.set_ylabel("UMAP$_2$" if do_umap else "$a_{i1}$")
        print(f"{title}: acc {acc:.3f} ari {ari:.3f} nmi {nmi:.3f} ({X.shape[1]} features)")
    panel_labels(fig, axes, dy_pt=14, y_from="tight")
    save(fig, os.path.join(FIG_DIR, "fig_clustering_appendix"))


if __name__ == "__main__":
    main()
