"""The four circuit-parameter panels of one trained run, annotated with the
numbers of its results/metrics.txt so that figure and table quote one value:

    W      learned vs true synaptic weight, one point per fitted edge, coloured
           by the postsynaptic cell type. The learned W is the template
           readout's per-edge fit (models/template_fit_alt.pt, the current form
           W * ReLU(v_j) + C fitted on 1,024 frames); an edge whose presynaptic
           neuron is never above the activity floor is unfitted (stored as 0)
           and left out, as in the table's R^2 (metrics Wij_n / Wij_R2).
    a      the learned 2-D embedding a_i, one point per neuron, by cell type
    V_rest learned vs true resting potential (inlier R^2, outlier % in metrics)
    tau    learned vs true membrane time constant (same)

Used by fig_gnn_params_3col.py (three noise levels, Fig. 1) and
fig_gnn_params_4col_flywire.py (four connectome variants, Supp. Fig.).
"""
import glob
import os

import numpy as np
import torch

from paper_style import (LOG_ROOT, read_metrics, scatter_identity, type_cmap, trim_axis,
                         pretty_ticks)


def _arr(x):
    return x.detach().cpu().numpy().ravel() if torch.is_tensor(x) else np.asarray(x).ravel()


def load_run(run):
    """Arrays of one run: W pairs over the fitted edges, embedding, tau, V_rest."""
    d = os.path.join(LOG_ROOT, run)
    npz = glob.glob(os.path.join(d, "results", "panels_*.npz"))
    if not npz:
        raise FileNotFoundError(f"no panels_*.npz under {d}/results")
    p = dict(np.load(npz[0], allow_pickle=True))
    m = read_metrics(d)
    W_true = _arr(torch.load(os.path.join(d, "gt_weights.pt"), map_location="cpu", weights_only=False))
    W_fit = _arr(torch.load(os.path.join(d, "models", "template_fit_alt.pt"), map_location="cpu",
                            weights_only=False)["model_state_dict"]["W"])
    edges = np.asarray(torch.load(os.path.join(d, "training_edges.pt"), map_location="cpu",
                                  weights_only=False))
    fitted = np.isfinite(W_fit) & (W_fit != 0)
    # The table's Wij_R2 is outlier-filtered (Wij_n_outliers) and a few fitted
    # edges land exactly on 0; both are tolerated, the figure quotes the table.
    r2 = 1 - np.mean((W_fit[fitted] - W_true[fitted]) ** 2) / np.var(W_true[fitted])
    if abs(fitted.sum() - m["Wij_n"]) > 0.01 * m["Wij_n"] or abs(r2 - m["Wij_R2"]) > 0.02:
        print(f"WARNING {run}: fitted {fitted.sum()} vs Wij_n {m['Wij_n']:.0f}, "
              f"R2 on shown edges {r2:.3f} vs metrics {m['Wij_R2']:.3f}")
    return dict(panels=p, metrics=m, W_true=W_true[fitted], W_learned=W_fit[fitted],
                W_post=edges[1][fitted], pct_fitted=100.0 * fitted.mean())


def _lims(true, learned=None, pad=0.08):
    """Axis range: the TRUE values, widened to the 2nd-98th percentile of the
    learned ones so the bulk is inside; the remaining learned outliers are
    clipped at the border (they are what the outlier percentage counts)."""
    lo, hi = float(np.min(true)), float(np.max(true))
    if learned is not None:
        lo = min(lo, float(np.percentile(learned, 2))); hi = max(hi, float(np.percentile(learned, 98)))
    d = (hi - lo) * pad
    return lo - d, hi + d


def draw_block(axes, run_data, cmap=None, show_ylabels=True, n_edges_max=150_000):
    """axes = (ax_W, ax_a, ax_V, ax_tau)."""
    p, m = run_data["panels"], run_data["metrics"]
    cmap = cmap or type_cmap()
    types = np.asarray(p["type_ids"])
    ax_W, ax_a, ax_V, ax_tau = axes

    scatter_identity(ax_W, run_data["W_true"], run_data["W_learned"], colors=types[run_data["W_post"]],
                     cmap=cmap, s=0.6, alpha=0.5, n_max=n_edges_max, lims=_lims(run_data["W_true"], run_data["W_learned"]),
                     annot=f"$R^2$ {m['Wij_R2']:.2f}, slope {m['Wij_slope']:.2f}\n"
                           f"{run_data['pct_fitted']:.0f}% of edges fitted",
                     xlabel="true $W_{ij}$", ylabel="learned $\\hat W_{ij}$" if show_ylabels else "")

    a = np.asarray(p["a"])
    ax_a.scatter(a[:, 0], a[:, 1], c=types, cmap=cmap, s=1.2, alpha=0.7, lw=0, rasterized=True)
    for setter, v in ((ax_a.set_xticks, a[:, 0]), (ax_a.set_yticks, a[:, 1])):
        t = pretty_ticks(v.min(), v.max(), 3)
        setter(t if len(t) >= 2 else pretty_ticks(v.min(), v.max(), 5))
    ax_a.set_xlabel("$a_{i,1}$")
    if show_ylabels:
        ax_a.set_ylabel("$a_{i,2}$")
    trim_axis(ax_a)

    scatter_identity(ax_V, p["V_rest_true"], p["V_rest_learned"], colors=types, cmap=cmap, s=1.5,
                     alpha=0.7, n_max=len(types), lims=_lims(p["V_rest_true"], p["V_rest_learned"]),
                     annot=f"$R^2$ {m['V_rest_R2']:.2f}, slope {m['V_rest_slope']:.2f}\n"
                           f"outliers {m['V_rest_pct_outliers']:.1f}%",
                     xlabel="true $V^{\\mathrm{rest}}_i$",
                     ylabel="learned $\\hat V^{\\mathrm{rest}}_i$" if show_ylabels else "")
    scatter_identity(ax_tau, p["tau_true"], p["tau_learned"], colors=types, cmap=cmap, s=1.5,
                     alpha=0.7, n_max=len(types), lims=_lims(p["tau_true"], p["tau_learned"]),
                     annot=f"$R^2$ {m['tau_R2']:.2f}, slope {m['tau_slope']:.2f}\n"
                           f"outliers {m['tau_pct_outliers']:.1f}%",
                     xlabel="true $\\tau_i$ (s)", ylabel="learned $\\hat\\tau_i$ (s)" if show_ylabels else "")
