"""Supp. Fig.: cell-type clusterability of the general-form GNN's learned
quantities (group lasso 25, sigma = 0.05, fold cv00; experiment 2).

    python scripts/fig_clustering_appendix.py

Four panels, each the 13,741 neurons coloured by true cell type (65 classes):
    a  ground truth (tau, V_rest) + 8 W-statistics of the true W (10-D, UMAP)
    b  the same 10 features from the learned tau, V_rest and template-readout W
    c  the learned 2-D embedding a_i, shown directly
    d  a_i with the learned (tau, V_rest, W-stats) (12-D, UMAP)
A Gaussian mixture (100 components) is fitted on the raw z-scored features;
accuracy (Hungarian-matched), ARI, NMI and the number of populated components
are quoted in each panel. UMAP is for display only.

Per-neuron W-statistics: mean, SD, min, max of the incoming and of the
outgoing edge weights; for the learned W over the fitted edges of the template
readout (models/template_fit_alt.pt, 76% of edges at this noise level).

Output: figures/fig_clustering_appendix.{pdf,png}
"""
import os
import sys

import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import CM, DATA_ROOT, FIG_DIR, FS_ANNOT, LOG_ROOT, panel_labels, save, type_cmap  # noqa: E402
sys.path.insert(0, "/workspace/connectome-gnn/src")
from connectome_gnn.sparsify import clustering_gmm  # noqa: E402

RUN = "flyvis_noise_005_blank50_condl25_cv00"
DATASET = "flyvis_noise_005_blank50_cv00"
N_GMM, SEED = 100, 42


def w_stats(w, src, dst, n, use=None):
    """(N, 8): in-mean, in-sd, in-min, in-max, out-mean, out-sd, out-min, out-max."""
    use = np.ones(len(w), bool) if use is None else use
    out = []
    for node in (dst, src):
        cols = []
        order = np.argsort(node[use]); nd = node[use][order]; wv = w[use][order]
        starts = np.searchsorted(nd, np.arange(n)); ends = np.searchsorted(nd, np.arange(n), side="right")
        for f in (np.mean, np.std, np.min, np.max):
            v = np.zeros(n, np.float32)
            for i in range(n):
                if ends[i] > starts[i]:
                    v[i] = f(wv[starts[i]:ends[i]])
            cols.append(v)
        out += cols
    return np.column_stack(out)


def gmm(features, types):
    from sklearn.mixture import GaussianMixture
    z = (features - features.mean(0)) / (features.std(0) + 1e-9)
    res = clustering_gmm(z, types, n_components=N_GMM)
    labels = GaussianMixture(n_components=N_GMM, random_state=SEED, reg_covar=1e-3).fit_predict(z)
    return float(res["accuracy"]), float(res["ari"]), float(res["nmi"]), int(np.unique(labels).size)


def umap2(features):
    import umap
    z = (features - features.mean(0)) / (features.std(0) + 1e-9)
    return umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1).fit_transform(z)


def main():
    d = os.path.join(LOG_ROOT, RUN)
    p = np.load(os.path.join(d, "results", f"panels_noise_005_blank50_cv00.npz"), allow_pickle=True)
    op = torch.load(os.path.join(DATA_ROOT, DATASET, "ode_params.pt"), map_location="cpu", weights_only=False)
    types = np.asarray(p["type_ids"]).astype(int)
    n = len(types)
    src, dst = np.asarray(op["edge_index"])
    W_true = np.asarray(op["W"], np.float32)
    edges = np.asarray(torch.load(os.path.join(d, "training_edges.pt"), map_location="cpu", weights_only=False))
    assert np.array_equal(edges, np.asarray(op["edge_index"]))
    W_fit = torch.load(os.path.join(d, "models", "template_fit_alt.pt"), map_location="cpu",
                       weights_only=False)["model_state_dict"]["W"].detach().numpy().ravel()
    fitted = np.isfinite(W_fit) & (W_fit != 0)
    tau_t, V_t = np.asarray(op["tau_i"], np.float32), np.asarray(op["V_i_rest"], np.float32)
    tau_l, V_l, a = np.asarray(p["tau_learned"]), np.asarray(p["V_rest_learned"]), np.asarray(p["a"])
    Ws_t = w_stats(W_true, src, dst, n)
    Ws_l = w_stats(W_fit, src, dst, n, use=fitted)
    feats = [
        ("ground truth $(\\tau, V^{\\mathrm{rest}})$ + $\\mathbf{W}$-stats (10-D, UMAP)",
         np.column_stack([tau_t, V_t, Ws_t]), True),
        ("learned $(\\hat\\tau, \\hat V^{\\mathrm{rest}})$ + $\\hat{\\mathbf{W}}$-stats (10-D, UMAP)",
         np.column_stack([tau_l, V_l, Ws_l]), True),
        ("learned embedding $\\mathbf{a}_i$ (2-D)", a, False),
        ("$\\mathbf{a}_i$ + learned $(\\hat\\tau, \\hat V^{\\mathrm{rest}}, \\hat{\\mathbf{W}}$-stats$)$ (12-D, UMAP)",
         np.column_stack([a, tau_l, V_l, Ws_l]), True),
    ]
    cmap = type_cmap()
    side, gap, margin = 5.0, 1.2, 0.7
    w = h = 2 * side + gap + 2 * margin
    fig = plt.figure(figsize=(w * CM, h * CM))
    gs = mgs.GridSpec(2, 2, figure=fig, left=margin / w, right=1 - margin / w, bottom=margin / h,
                      top=1 - margin / h, wspace=gap / side, hspace=gap / side)
    axes = []
    for k, (title, X, do_umap) in enumerate(feats):
        ax = fig.add_subplot(gs[k // 2, k % 2]); axes.append(ax)
        acc, ari, nmi, kk = gmm(X, types)
        xy = umap2(X) if do_umap else X
        ax.scatter(xy[:, 0], xy[:, 1], c=types, cmap=cmap, s=1.5, alpha=0.7, lw=0, rasterized=True)
        if not do_umap:   # the raw embedding has a few far outliers: show the 0.5-99.5 percentile box
            for lim, v in ((ax.set_xlim, xy[:, 0]), (ax.set_ylim, xy[:, 1])):
                lo, hi = np.percentile(v, [0.5, 99.5]); d = 0.05 * (hi - lo); lim(lo - d, hi + d)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, fontsize=FS_ANNOT + 1, pad=3)
        ax.text(0.02, 0.98, f"GMM acc. {acc:.2f}, k = {kk}\nARI {ari:.2f}, NMI {nmi:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=FS_ANNOT + 1, linespacing=1.15)
        ax.set_xlabel("UMAP 1" if do_umap else "$a_{i,1}$"); ax.set_ylabel("UMAP 2" if do_umap else "$a_{i,2}$")
        print(f"{title}: acc {acc:.3f} ari {ari:.3f} nmi {nmi:.3f} k {kk}")
    panel_labels(fig, axes)
    save(fig, os.path.join(FIG_DIR, "fig_clustering_appendix"))


if __name__ == "__main__":
    main()
