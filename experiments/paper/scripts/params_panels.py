"""The four circuit-parameter panels of one trained run, in the published
figure's conventions, from the template readout that the tables score:

    W      learned vs true synaptic weight, one black point per fitted edge (the
           template readout's per-edge fit, models/template_fit_alt.pt, i.e.
           W * ReLU(v_j) + C fitted on 1,024 frames; an edge whose presynaptic
           neuron never rises above the activity floor is unfitted, stored as 0,
           and left out as in metrics Wij_n / Wij_R2); fixed range [-1, 2]
    a      the learned 2-D embedding a_i, one point per neuron, by cell type
    V_rest learned vs true resting potential: the template readout's per-neuron
           value (template_fit_alt.pt V_rest); range [0, 1]; neurons beyond the
           paper's outlier threshold |dV| > 0.2 in red and excluded from the
           quoted R^2, dotted lines at +-0.2 (metrics V_rest_R2 / _pct_outliers)
    tau    the same for the membrane time constant, softplus(raw_tau); range
           [0, 0.5] s; threshold |dtau| > 0.1 s

Annotations: R^2 (inlier) with the all-neuron R^2 in parentheses, slope,
outlier %, every value read from metrics.txt and checked against the arrays.
"""
import glob
import os

import numpy as np
import torch

from paper_style import (COLOR_OUTLIER, LOG_ROOT, annotate, identity, read_metrics, ticks3,
                         type_cmap)

W_RANGE, V_RANGE, TAU_RANGE = (-1.0, 2.0), (0.0, 1.0), (0.0, 0.5)
THR = {"tau": 0.1, "V_rest": 0.2}


def _arr(x):
    return x.detach().cpu().numpy().ravel() if torch.is_tensor(x) else np.asarray(x).ravel()


def load_run(run):
    d = os.path.join(LOG_ROOT, run)
    npz = glob.glob(os.path.join(d, "results", "panels_*.npz"))
    if not npz:
        raise FileNotFoundError(f"no panels_*.npz under {d}/results")
    p = dict(np.load(npz[0], allow_pickle=True))
    m = read_metrics(d)
    W_true = _arr(torch.load(os.path.join(d, "gt_weights.pt"), map_location="cpu", weights_only=False))
    sd = torch.load(os.path.join(d, "models", "template_fit_alt.pt"), map_location="cpu",
                    weights_only=False)["model_state_dict"]
    W_fit = _arr(sd["W"])
    tau = np.log1p(np.exp(_arr(sd["raw_tau"]).astype(np.float64)))
    V = _arr(sd["V_rest"]).astype(np.float64)
    edges = np.asarray(torch.load(os.path.join(d, "training_edges.pt"), map_location="cpu", weights_only=False))
    fitted = np.isfinite(W_fit) & (W_fit != 0)
    r2w = 1 - np.mean((W_fit[fitted] - W_true[fitted]) ** 2) / np.var(W_true[fitted])
    if abs(fitted.sum() - m["Wij_n"]) > 0.01 * m["Wij_n"] or abs(r2w - m["Wij_R2"]) > 0.02:
        print(f"WARNING {run}: fitted {fitted.sum()} vs Wij_n {m['Wij_n']:.0f}, R2 {r2w:.3f} vs {m['Wij_R2']:.3f}")
    tau_true, V_true = np.asarray(p["tau_true"], np.float64), np.asarray(p["V_rest_true"], np.float64)
    for name, t, l in (("tau", tau_true, tau), ("V_rest", V_true, V)):
        ok = np.abs(l - t) <= THR[name]
        r2n = 1 - np.mean((l[ok] - t[ok]) ** 2) / np.var(t[ok])
        assert abs(r2n - m[f"{name}_R2"]) < 5e-3 and abs(100 * (~ok).mean() - m[f"{name}_pct_outliers"]) < 0.2, \
            (run, name, r2n, m[f"{name}_R2"], 100 * (~ok).mean(), m[f"{name}_pct_outliers"])
    return dict(panels=p, metrics=m, W_true=W_true[fitted], W_learned=W_fit[fitted],
                W_post=edges[1][fitted], pct_fitted=100.0 * fitted.mean(),
                tau_true=tau_true, tau=tau, V_true=V_true, V=V)


def _r2_all(v):
    return f" ({v:.2f})" if v >= -1 else " (<$-$1)"


def draw_block(axes, run_data, cmap=None, show_ylabels=True, n_edges_max=150_000, seed=0):
    """axes = (ax_W, ax_a, ax_V, ax_tau)."""
    p, m = run_data["panels"], run_data["metrics"]
    cmap = cmap or type_cmap()
    types = np.asarray(p["type_ids"])
    ax_W, ax_a, ax_V, ax_tau = axes
    rng = np.random.default_rng(seed)

    # W
    n = len(run_data["W_true"])
    idx = rng.choice(n, min(n, n_edges_max), replace=False)
    lo, hi = W_RANGE
    identity(ax_W, lo, hi)
    # the published panels: black points, alpha 0.3 (s = 1 on 10-inch panels, s = 0.1 for W)
    ax_W.scatter(run_data["W_true"][idx], run_data["W_learned"][idx], c="k", s=0.1, alpha=0.1, lw=0,
                 rasterized=True, zorder=2)
    ticks3(ax_W, lo, hi)
    ax_W.set_aspect("equal", adjustable="box")
    annotate(ax_W, f"R²: {m['Wij_R2']:.2f}\nslope: {m['Wij_slope']:.2f}")   # the fitted fraction is in the caption
    ax_W.set_xlabel("true $W_{ij}$")
    if show_ylabels:
        ax_W.set_ylabel("learned $\\hat W_{ij}$")

    # embedding
    a = np.asarray(p["a"])
    ax_a.scatter(a[:, 0], a[:, 1], c=types, cmap=cmap, s=0.5, alpha=0.5, lw=0, rasterized=True)
    for axis, v in (("x", a[:, 0]), ("y", a[:, 1])):
        lo_, hi_ = np.percentile(v, [0.2, 99.8]); d = 0.05 * (hi_ - lo_)
        ticks3(ax_a, round(lo_ - d, 1), round(hi_ + d, 1), axis, mid_decimals=1)
    ax_a.set_xlabel("$a_{i0}$")
    if show_ylabels:
        ax_a.set_ylabel("$a_{i1}$")

    # V_rest, tau
    for ax, t, l, key, rng_, xl, yl in (
            (ax_V, run_data["V_true"], run_data["V"], "V_rest", V_RANGE, "true $V_{rest}$", "learned $V_{rest}$"),
            (ax_tau, run_data["tau_true"], run_data["tau"], "tau", TAU_RANGE, "true $\\tau$", "learned $\\tau$")):
        lo, hi = rng_
        out = np.abs(l - t) > THR[key]
        identity(ax, lo, hi, thr=THR[key])
        ax.scatter(t[out], np.clip(l[out], lo, hi), color=COLOR_OUTLIER, s=0.4, alpha=0.5, lw=0,
                   rasterized=True, zorder=2)
        ax.scatter(t[~out], l[~out], c="k", s=0.3, alpha=0.3, lw=0, rasterized=True, zorder=3)
        ticks3(ax, lo, hi)
        ax.set_aspect("equal", adjustable="box")
        annotate(ax, f"R²: {m[key + '_R2']:.2f}{_r2_all(m[key + '_R2_all'])}\nslope: {m[key + '_slope']:.2f}\n"
                     f"Outliers: {m[key + '_pct_outliers']:.1f}%")
        ax.set_xlabel(xl)
        if show_ylabels:
            ax.set_ylabel(yl)
