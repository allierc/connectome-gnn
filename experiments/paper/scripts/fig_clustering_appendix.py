"""Supp. Fig.: cell-type clusterability of the general-form GNN's learned
quantities (group lasso 25, sigma = 0.05, fold cv00; experiment 2), in the
published 2 x 2 layout.

    python scripts/fig_clustering_appendix.py

    a  true (tau, V_rest, W-stats): ground-truth tau, V_rest and 8 per-neuron
       statistics of the true W (10-D), UMAP
    b  learned (tau, V_rest, W-stats): the template readout's tau, V_rest and W
       over its fitted edges (10-D), UMAP
    c  learned a_i: the 2-D embedding, shown directly
    d  learned (a_i, tau, V_rest, W-stats): 12-D, UMAP
A Gaussian mixture (100 components, z-scored features) gives the accuracy
(Hungarian-matched), ARI, NMI and the number of populated components k.

Output: figures/fig_clustering_appendix.{pdf,png}
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import CM, DATA_ROOT, FIG_DIR, FS_LABEL, FS_TICK, LOG_ROOT, panel_labels, save, type_cmap  # noqa: E402
sys.path.insert(0, "/workspace/connectome-gnn/src")
from connectome_gnn.sparsify import clustering_gmm  # noqa: E402

RUN = "flyvis_noise_005_blank50_condl25_cv00"
DATASET = "flyvis_noise_005_blank50_cv00"
N_GMM, SEED = 100, 42


def w_stats(w, src, dst, n, use=None):
    """(N, 8): mean, SD, min, max of the incoming, then of the outgoing edge weights."""
    use = np.ones(len(w), bool) if use is None else use
    out = []
    for node in (dst, src):
        order = np.argsort(node[use]); nd = node[use][order]; wv = w[use][order]
        starts = np.searchsorted(nd, np.arange(n)); ends = np.searchsorted(nd, np.arange(n), side="right")
        for f in (np.mean, np.std, np.min, np.max):
            v = np.zeros(n, np.float32)
            for i in range(n):
                if ends[i] > starts[i]:
                    v[i] = f(wv[starts[i]:ends[i]])
            out.append(v)
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
    p = np.load(os.path.join(d, "results", "panels_noise_005_blank50_cv00.npz"), allow_pickle=True)
    op = torch.load(os.path.join(DATA_ROOT, DATASET, "ode_params.pt"), map_location="cpu", weights_only=False)
    types = np.asarray(p["type_ids"]).astype(int)
    n = len(types)
    src, dst = np.asarray(op["edge_index"])
    assert np.array_equal(np.asarray(torch.load(os.path.join(d, "training_edges.pt"), map_location="cpu",
                                                weights_only=False)), np.asarray(op["edge_index"]))
    sd = torch.load(os.path.join(d, "models", "template_fit_alt.pt"), map_location="cpu",
                    weights_only=False)["model_state_dict"]
    W_fit = sd["W"].detach().numpy().ravel().astype(np.float64)
    fitted = np.isfinite(W_fit) & (W_fit != 0)
    tau_l = np.log1p(np.exp(sd["raw_tau"].numpy().astype(np.float64)))
    V_l = sd["V_rest"].numpy().astype(np.float64)
    tau_t, V_t = np.asarray(op["tau_i"], np.float64), np.asarray(op["V_i_rest"], np.float64)
    a = np.asarray(p["a"])
    Ws_t = w_stats(np.asarray(op["W"], np.float64), src, dst, n)
    Ws_l = w_stats(W_fit, src, dst, n, use=fitted)
    panels = [
        ("true ($\\tau$, $V_{rest}$, $W$-stats)", np.column_stack([tau_t, V_t, Ws_t]), True),
        ("learned ($\\tau$, $V_{rest}$, $W$-stats)", np.column_stack([tau_l, V_l, Ws_l]), True),
        ("learned $\\mathbf{a}_i$", a, False),
        ("learned ($\\mathbf{a}_i$, $\\tau$, $V_{rest}$, $W$-stats)", np.column_stack([a, tau_l, V_l, Ws_l]), True),
    ]
    cmap = type_cmap()
    side, gap, margin = 5.5, 1.6, 0.9
    w = h = 2 * side + gap + 2 * margin
    fig = plt.figure(figsize=(w * CM, h * CM))
    axes = []
    for k, (title, X, do_umap) in enumerate(panels):
        r, c = divmod(k, 2)
        ax = fig.add_axes([(margin + c * (side + gap)) / w, (margin + (1 - r) * (side + gap)) / h, side / w, side / h])
        axes.append(ax)
        acc, ari, nmi, kk = gmm(X, types)
        xy = umap2(X) if do_umap else X
        ax.scatter(xy[:, 0], xy[:, 1], c=types, cmap=cmap, s=2.0, alpha=0.8, lw=0, rasterized=True)
        if not do_umap:
            for lim, v in ((ax.set_xlim, xy[:, 0]), (ax.set_ylim, xy[:, 1])):
                lo, hi = np.percentile(v, [0.5, 99.5]); dd = 0.05 * (hi - lo); lim(lo - dd, hi + dd)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=FS_LABEL, pad=4)
        ax.text(0.03, 0.97, f"GMM accuracy = {acc:.2f}  (k={kk}/{N_GMM})\nARI = {ari:.2f}, NMI = {nmi:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=FS_TICK, linespacing=1.4)
        ax.set_xlabel("UMAP$_1$" if do_umap else "$a_{i0}$"); ax.set_ylabel("UMAP$_2$" if do_umap else "$a_{i1}$")
        print(f"{title}: acc {acc:.3f} ari {ari:.3f} nmi {nmi:.3f} k {kk}")
    panel_labels(fig, axes, dy_pt=14)
    save(fig, os.path.join(FIG_DIR, "fig_clustering_appendix"))


if __name__ == "__main__":
    main()
