"""Structure of the learned embedding a_i of one run, by cell type (meeting slide).

    python tools/embedding_structure.py [run]      (default flyvis_noise_005_blank50_condl25_cv00)

Per cell type: the centre of its neurons' embeddings (median a_i), its true tau and V_rest (the
type's median). The band = types whose centre has a_i0 > -1. Prints the rank correlations of the
centres' coordinates with tau and V_rest, the left types and the types split across a_i0 = -1, and
writes presentation/Fig/meeting_embedding.png, 2 x 2: (a) every neuron, coloured by cell type, the
central 90 % on each axis as in Fig. 1d; (b) one dot per cell-type centre, on panel a's axes, red for types whose true
tau is the minimum 0.020 s, blue for the others; (c) centre a_i1 against true V_rest; (d) centre a_i0
against true tau.
"""
import glob
import os
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from connectome_gnn.metrics import INDEX_TO_NAME

LOG = "/groups/saalfeld/home/allierc/GraphData/log/fly"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "presentation", "Fig", "meeting_embedding.png")
SPLIT = -1.0


def _cbar(fig, A, sc, label, fs):
    """A thin colour bar just right of panel A, outside it, so it takes no space from the panel."""
    cax = A.inset_axes([1.04, 0.0, 0.06, 1.0])
    cb = fig.colorbar(sc, cax=cax)
    cax.set_title(label, fontsize=fs, pad=2)
    cb.ax.tick_params(labelsize=fs - 1, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)


def main(run="flyvis_noise_005_blank50_condl25_cv00"):
    d = np.load(glob.glob(f"{LOG}/{run}/results/panels_*.npz")[0])
    a, t, tau, V = d["a"], d["type_ids"].astype(int), d["tau_true"], d["V_rest_true"]
    T = np.unique(t)
    names = np.array([INDEX_TO_NAME[k] for k in T])
    C = np.array([np.median(a[t == k], 0) for k in T])
    tt = np.array([np.median(tau[t == k]) for k in T])
    vv = np.array([np.median(V[t == k]) for k in T])
    band = C[:, 0] > SPLIT
    print(f"{run}: {len(T)} types, band (centre a_i0 > {SPLIT}) {band.sum()}, left {(~band).sum()}")
    for x, lx in ((0, "a_i0"), (1, "a_i1")):
        for y, ly in ((tt, "tau"), (vv, "V_rest")):
            print(f"  band types, Spearman {lx} vs {ly}: {spearmanr(C[band, x], y[band]).statistic:+.2f}")
    for y, ly in ((tt, "tau"), (vv, "V_rest")):
        print(f"  {ly}: left median {np.median(y[~band]):.3f}, band median {np.median(y[band]):.3f}, "
              f"Mann-Whitney p {mannwhitneyu(y[~band], y[band]).pvalue:.1e}")
    print("  left types:", ", ".join(f"{n} (tau {x:.3f})" for n, x in zip(names[~band], tt[~band])))
    for k, n in zip(T, names):
        fr = (a[t == k, 0] < SPLIT).mean()
        if 0.05 < fr < 0.95:
            print(f"  split type {n}: {fr:.0%} of its neurons left of a_i0 = {SPLIT}")

    lo = np.percentile(a, 5, axis=0); hi = np.percentile(a, 95, axis=0); pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    # the centre panels show every centre except Lawf1's far tail (a_i1 ~ -26), which is reported below
    keep = C[:, 1] > -10
    clo = C[keep].min(0); chi = C[keep].max(0); cpad = 0.06 * (chi - clo)
    short = np.abs(tt - 0.020) < 1e-3                  # the minimum time constant, 0.020 s
    # DRAWN AT THE SIZE THE SLIDE SHOWS IT (0.8 of the text height, ~2.8 in), with Fig. 1's font sizes
    # as they appear on slide 6 (paper_style FS_* at the ~0.93 scale of that crop): no rescaling
    FS_L, FS_T, FS_P, FS_A = 7.4, 5.6, 7.0, 5.1
    plt.rcParams.update({"font.size": FS_T, "axes.labelsize": FS_L, "xtick.labelsize": FS_T,
                         "ytick.labelsize": FS_T, "legend.fontsize": FS_A})
    # e, f: centres / neurons, x = a_i1, y = the learned mean outgoing weight of the neuron, the learned
    # clustering feature that best separates the cell types beside a_i0 (5-nearest-neighbour type
    # accuracy over the 22 clustering features, results/extras/clustering_features.npz)
    cf = np.load(f"{LOG}/{run}/results/extras/clustering_features.npz")
    wout = cf["X_full"][:, 8]                          # [a_i0, a_i1, tau, V_rest, W in mean/SD/min/max, out mean, ...]
    Cw = np.array([np.median(wout[t == k]) for k in T])
    # g, h: the Jacobians of the two MLPs (tools/embedding_jacobian.py, results/extras/embedding_jacobian.npz):
    # per neuron f = alpha0 + alpha1 v + alpha2 msg + alpha3 I over sampled frames; the leak -alpha1 and the
    # model's resting level -alpha0/alpha1, per cell type; g_phi's sensitivity to each of its six inputs
    jz = np.load(f"{LOG}/{run}/results/extras/embedding_jacobian.npz")
    al = jz["alpha"]; tj = jz["type_ids"]
    with np.errstate(divide="ignore", invalid="ignore"):
        lvl = -al[:, 0] / al[:, 1]
    Cleak = np.array([np.median(-al[tj == k, 1]) for k in T])
    Clvl = np.array([np.median(lvl[tj == k]) for k in T])
    import matplotlib.gridspec as mgs
    fig = plt.figure(figsize=(5.9, 2.8), facecolor="white")
    top = mgs.GridSpec(1, 4, figure=fig, left=0.075, right=0.96, top=0.95, bottom=0.62, wspace=0.55)
    bot = mgs.GridSpec(1, 5, figure=fig, left=0.075, right=0.94, top=0.43, bottom=0.11, wspace=1.05)
    A_a, A_b, A_c, A_d = (fig.add_subplot(top[0, k]) for k in range(4))
    A_g1, A_g2, A_h, A_t, A_v = (fig.add_subplot(bot[0, k]) for k in range(5))
    # g: the twin of a, every neuron coloured by its true tau (log scale)
    from matplotlib.colors import LogNorm
    sc = A_t.scatter(a[:, 0], a[:, 1], c=tau, cmap="viridis", norm=LogNorm(tau.min(), tau.max()),
                     s=0.15, alpha=0.6, lw=0, rasterized=True)
    A_t.set_xlim(lo[0], hi[0]); A_t.set_ylim(lo[1], hi[1])
    A_t.set_xlabel("$a_{i0}$"); A_t.set_ylabel("$a_{i1}$")
    _cbar(fig, A_t, sc, "$\\tau$ (s)", FS_A)
    _cb = A_t.child_axes[-1]
    _cb.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.02, 0.05, 0.1, 0.3]))
    _cb.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.02", "0.05", "0.1", "0.3"]))
    _cb.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # h: the same, coloured by true V_rest
    sc = A_v.scatter(a[:, 0], a[:, 1], c=V, cmap="viridis", s=0.15, alpha=0.6, lw=0, rasterized=True)
    A_v.set_xlim(lo[0], hi[0]); A_v.set_ylim(lo[1], hi[1])
    A_v.set_xlabel("$a_{i0}$"); A_v.set_ylabel("$a_{i1}$")
    _cbar(fig, A_v, sc, "$V^{rest}$", FS_A)
    A_a.scatter(a[:, 0], a[:, 1], c=t, cmap="tab20", s=0.15, alpha=0.5, lw=0, rasterized=True)
    A_a.set_xlim(lo[0], hi[0]); A_a.set_ylim(lo[1], hi[1])
    A_a.set_xlabel("$a_{i0}$"); A_a.set_ylabel("$a_{i1}$")
    # red / blue: two classes of cell type, by their true tau
    groups = ((short, "#c0522a", "true $\\tau$ = 0.020 s"), (~short, "#1f6fb8", "true $\\tau$ > 0.020 s"))
    wl = np.percentile(wout, [5, 95]); wl = (wl[0] - 0.05 * np.ptp(wl), wl[1] + 0.05 * np.ptp(wl))
    for m, col, lab in groups:
        A_b.scatter(C[m & keep, 0], C[m & keep, 1], s=2.5, color=col, lw=0)
        A_c.scatter(tt[m & keep], C[m & keep, 0], s=2.5, color=col, lw=0, label=lab)
        A_d.scatter(vv[m & keep], C[m & keep, 1], s=2.5, color=col, lw=0)
    A_b.set_xlim(lo[0], hi[0]); A_b.set_ylim(lo[1], hi[1])          # panel a's limits, for comparison
    A_b.set_xlabel("centre $a_{i0}$"); A_b.set_ylabel("centre $a_{i1}$")
    A_c.legend(frameon=False, loc="upper right", handletextpad=0.1, markerscale=2.0, borderaxespad=0.1)
    A_c.set_xlabel("true $\\tau$ (s)"); A_c.set_ylabel("centre $a_{i0}$")
    A_d.set_xlabel("true $V^{rest}$"); A_d.set_ylabel("centre $a_{i1}$")
    # g: f_theta's leak against true tau, coloured by the centre a_i0; its resting level against true V_rest,
    # coloured by the centre a_i1
    sc = A_g1.scatter(tt[keep], Cleak[keep], c=C[keep, 0], cmap="viridis", s=3, lw=0)
    A_g1.set_xscale("log"); A_g1.set_yscale("log")
    A_g1.set_xlabel("true $\\tau$ (s)"); A_g1.set_ylabel("leak $-\\partial f_\\theta/\\partial v$")
    _cbar(fig, A_g1, sc, "$a_{i0}$", FS_A)
    sc = A_g2.scatter(vv[keep], Clvl[keep], c=C[keep, 1], cmap="viridis", s=3, lw=0)
    A_g2.set_xlabel("true $V^{rest}$"); A_g2.set_ylabel("rest level")
    _cbar(fig, A_g2, sc, "$a_{i1}$", FS_A)
    # h: g_phi's sensitivity, mean |dg/dx| x SD(x), to each input
    gn = [str(x) for x in jz["g_names"]]
    lab = {"v_j": "$v_j$", "a_j0": "$a_{j0}$", "a_j1": "$a_{j1}$", "v_i": "$v_i$", "a_i0": "$a_{i0}$", "a_i1": "$a_{i1}$"}
    A_h.bar(range(len(gn)), jz["g_sensitivity"], color=["0.15"] + ["0.6"] * (len(gn) - 1), width=0.7)
    A_h.set_yscale("log"); A_h.set_xticks(range(len(gn))); A_h.set_xticklabels([lab[g] for g in gn], rotation=90)
    A_h.set_ylabel("$g_\\phi$ sens.")
    for A, L in ((A_a, "a"), (A_b, "b"), (A_c, "c"), (A_d, "d"), (A_g1, "e"), (A_g2, None), (A_h, "f"), (A_t, "g"), (A_v, "h")):
        for sp in ("top", "right"):
            A.spines[sp].set_visible(False)
        if L:
            A.text(-0.42, 1.03, L, transform=A.transAxes, fontsize=FS_P, fontweight="bold", va="bottom")
        A.tick_params(length=2, width=0.5, pad=1.5)
        for sp in ("left", "bottom"):
            A.spines[sp].set_linewidth(0.5)
    fig.savefig(OUT, dpi=400)
    for i in np.flatnonzero(~keep):
        print(f"  not drawn in b, c: {names[i]} centre ({C[i, 0]:.2f}, {C[i, 1]:.2f})")
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main(*sys.argv[1:])
