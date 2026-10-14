"""The learned embedding a_i of a 3-D-embedding general-form GNN, as panels d, e, f of the Figure 1
proposal: by cell type, by true tau, by true V_rest.

    python tools/embedding_3d.py [run]      (default flyvis_noise_005_blank50_condl25_emb3_cv00)

Row 1, a-c: the three coordinates (a_i0, a_i1, a_i2) in 3-D. Row 2, d-f: the same neurons on the
first two principal components of a, each axis labelled with the share of the embedding's
variance it carries. The embedding is defined only up to a rotation (the MLPs reading it can
absorb any), so the principal axes are the one orientation the data itself fixes. Every panel
shows the central 90 % of the neurons on each axis (5th to 95th percentile), as panel d of
Figure 1 does. Reads results/panels_*.npz (a, type_ids, tau_true, V_rest_true); writes
presentation/Fig/meeting_embedding3d_<fold>.{pdf,png}.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402
import matplotlib.ticker as mtick                       # noqa: E402
import numpy as np                                      # noqa: E402
from matplotlib.colors import LogNorm                   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments", "paper", "scripts"))
from paper_style import CM, FS_ANNOT, FS_LABEL, FS_TICK, LOG_ROOT, type_cmap  # noqa: E402

RUN = "flyvis_noise_005_blank50_condl25_emb3_cv00"


def central(x):
    lo, hi = np.percentile(x, [5, 95])
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def colour_bar(fig, ax, sc, label, log, three_d):
    cax = ax.inset_axes([0.45, 1.02 if three_d else 1.05, 0.55, 0.04])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=FS_ANNOT, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    cax.xaxis.set_ticks_position("top")
    cax.text(-0.04, 0.5, label, transform=cax.transAxes, ha="right", va="center", fontsize=FS_ANNOT)
    ticks = ([0.02, 0.1, 0.3], ["0.02", "0.1", "0.3"]) if log else ([0.2, 0.5, 0.8], ["0.2", "0.5", "0.8"])
    cax.xaxis.set_major_locator(mtick.FixedLocator(ticks[0]))
    cax.xaxis.set_major_formatter(mtick.FixedFormatter(ticks[1]))
    cax.xaxis.set_minor_locator(mtick.NullLocator())


def main(run=RUN):
    p = np.load(glob.glob(os.path.join(LOG_ROOT, run, "results", "panels_*.npz"))[0])
    a = np.asarray(p["a"], np.float64)
    assert a.shape[1] == 3, a.shape
    types = np.asarray(p["type_ids"]).astype(int)
    tau, V = np.asarray(p["tau_true"]), np.asarray(p["V_rest_true"])
    # principal axes of the centred embedding
    ac = a - a.mean(0)
    _, s, vt = np.linalg.svd(ac, full_matrices=False)
    pcs = ac @ vt.T
    share = 100 * s ** 2 / np.sum(s ** 2)

    colourings = (("type", types, dict(cmap=type_cmap(), vmin=0, vmax=64)),
                  ("tau", tau, dict(cmap="viridis", norm=LogNorm(tau.min(), tau.max()))),
                  ("V", V, dict(cmap="viridis")))
    fig = plt.figure(figsize=(18 * CM, 11.5 * CM))
    lims = [central(a[:, k]) for k in range(3)]
    keep3 = np.all([(a[:, k] >= lims[k][0]) & (a[:, k] <= lims[k][1]) for k in range(3)], axis=0)
    plims = [central(pcs[:, k]) for k in range(2)]
    letters = iter("abcdef")
    for col, (name, c, kw) in enumerate(colourings):
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        sc = ax.scatter(a[keep3, 0], a[keep3, 1], a[keep3, 2], c=c[keep3], s=0.4, alpha=0.5, lw=0,
                        rasterized=True, **kw)
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        for lab, setter in (("$a_{i0}$", ax.set_xlabel), ("$a_{i1}$", ax.set_ylabel), ("$a_{i2}$", ax.set_zlabel)):
            setter(lab, fontsize=FS_LABEL, labelpad=-6)
        ax.tick_params(labelsize=FS_TICK - 1, pad=-2)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(mtick.MaxNLocator(3))
        ax.view_init(elev=20, azim=-60)
        ax.text2D(-0.02, 1.0, next(letters), transform=ax.transAxes, fontsize=FS_LABEL, fontweight="bold")
        if name != "type":
            colour_bar(fig, ax, sc, "true $\\tau$" if name == "tau" else "true $V_{rest}$", name == "tau", True)
    for col, (name, c, kw) in enumerate(colourings):
        ax = fig.add_subplot(2, 3, col + 4)
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=c, s=0.5, alpha=0.5, lw=0, rasterized=True, **kw)
        ax.set_xlim(plims[0]); ax.set_ylim(plims[1])
        ax.set_xlabel(f"PC1 ({share[0]:.0f}% of variance)", fontsize=FS_LABEL)
        ax.set_ylabel(f"PC2 ({share[1]:.0f}%)", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.xaxis.set_major_locator(mtick.MaxNLocator(3)); ax.yaxis.set_major_locator(mtick.MaxNLocator(3))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(-0.18, 1.04, next(letters), transform=ax.transAxes, fontsize=FS_LABEL, fontweight="bold")
        if name != "type":
            colour_bar(fig, ax, sc, "true $\\tau$" if name == "tau" else "true $V_{rest}$", name == "tau", False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.09, wspace=0.35, hspace=0.25)
    out = os.path.join(REPO, "presentation", "Fig", f"meeting_embedding3d_{run.split('_')[-1]}")
    fig.savefig(out + ".pdf", dpi=300); fig.savefig(out + ".png", dpi=300)
    print("wrote", out + ".{pdf,png}", " PC variance shares %:", np.round(share, 1))


if __name__ == "__main__":
    main(next((a for a in sys.argv[1:] if not a.startswith("--")), RUN))
