#!/usr/bin/env python
"""fig_static_map -- the fitted static map of eye G, against the holds it came from.

    python fig_static_map.py

Section 4.2 states the static map as one quadratic per angle over the six drives,

    g^k(m) = sum_i a^k_i m_i + sum_{i<=j} b^k_ij m_i m_j ,   k in {theta, phi, psi}

and quotes its residual. This draws it, so the residual can be seen rather than
taken on trust, and so the two things the fit is for are visible:

  (a) how well 27 coefficients per angle reproduce all 109 settled holds -- the
      single-muscle sweep, the pair screen and the 64-point Sobol sweep at once;
  (b) the six marginals, the model driven one muscle at a time, against the holds
      that measured them. This is where eye G's strong convexity lives: several
      muscles do almost nothing below a quarter drive and most of their work in
      the top half, which is the shape a linearisation about rest would miss;
  (c) the cross terms. Fifteen pairs, three angles. The pair screen found 15 of 15
      non-additive, and this is what that looks like as coefficients.

Everything comes from `<eye>/charac/stage*.json` -- the same rows `train_eyeG.fit_deep`
fits -- so the figure and the model cannot drift apart.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_eyeG as TG                                     # noqa: E402

ANG = [r"$\theta$  horizontal", r"$\varphi$  vertical", r"$\psi$  torsion"]
ACOL = ["#cf222e", "#1f6feb", "#2ea043"]
LBL = dict(fontsize=17, fontweight="bold", va="top", ha="left")


def rows(eye_dir):
    out = []
    for st in ("stage0.json", "stage1.json", "stage2a.json", "stage2b.json",
               "stage6d.json"):
        out += [r for r in TG._charac(eye_dir, st) if r.get("settled")]
    idx = {m: k for k, m in enumerate(TG.MUSCLES)}
    U = np.zeros((len(out), 6))
    for k, r in enumerate(out):
        for nm, lv in zip(r["muscles"], r["level"]):
            U[k, idx[str(nm)]] = float(lv)
    P = np.array([r["pose_deg"] for r in out], float)
    n = np.array([int((u > 1e-6).sum()) for u in U])
    return U, P, n


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eye-dir", default=TG.EYE_DIR)
    p.add_argument("--out", default=os.path.join(TG.MODELS, "fig_static_map.png"))
    a = p.parse_args()

    U, P, ndr = rows(a.eye_dir)
    A = TG.quad_design(U)
    beta, *_ = np.linalg.lstsq(A, P, rcond=None)
    pred = A @ beta
    rms = np.sqrt(((P - pred) ** 2).mean(0))
    lin, *_ = np.linalg.lstsq(U, P, rcond=None)
    rms_lin = np.sqrt(((P - U @ lin) ** 2).mean(0))
    print(f"{len(U)} settled holds; quad rms {np.round(rms,3)}  "
          f"linear rms {np.round(rms_lin,3)}  ranges {np.round(P.max(0)-P.min(0),1)}")

    fig = plt.figure(figsize=(15.0, 13.0), facecolor="white")
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.30,
                          left=0.07, right=0.975, top=0.955, bottom=0.055)
    MPOS = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]   # the six marginals

    def dress(ax, letter=None):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("black")
        ax.tick_params(colors="black", labelsize=12)
        ax.xaxis.label.set_size(13.5); ax.yaxis.label.set_size(13.5)
        if letter:
            ax.text(-0.16, 1.10, letter, transform=ax.transAxes, color="black", **LBL)

    # --- (a) predicted against measured, all 109 holds --------------------
    ax = fig.add_subplot(gs[0, 0]); dress(ax, "a")
    mk = {1: "o", 2: "s", 6: "^"}
    for nd, m in mk.items():
        sel = ndr == nd
        if not sel.any():
            continue
        for k in range(3):
            ax.plot(P[sel, k], pred[sel, k], m, ms=4.2, mfc="none", mew=1.0,
                    color=ACOL[k], alpha=0.85)
    lim = np.abs(np.concatenate([P.ravel(), pred.ravel()])).max() * 1.05
    ax.plot([-lim, lim], [-lim, lim], "-", color="0.4", lw=1.1, zorder=0)
    ax.set_xlabel("measured pose (deg)"); ax.set_ylabel("fitted $g^k(m)$ (deg)")
    ax.text(0.03, 0.97, f"{len(U)} settled holds\nrms  "
            + " / ".join(f"{r:.2f}" for r in rms) + "$^\\circ$\n"
            + "linear  " + " / ".join(f"{r:.2f}" for r in rms_lin) + "$^\\circ$",
            transform=ax.transAxes, va="top", fontsize=11.5)
    h = [plt.Line2D([], [], color=ACOL[k], marker="o", ls="none", mfc="none",
                    label=ANG[k]) for k in range(3)]
    h += [plt.Line2D([], [], color="0.35", marker=m, ls="none", mfc="none",
                     label=f"{nd} muscle" + ("s" if nd > 1 else "") + " driven")
          for nd, m in mk.items()]
    ax.legend(handles=h, frameon=False, fontsize=9.5, loc="lower right", ncol=1)

    # --- (b) the six marginals, model against the holds -------------------
    for j, mus in enumerate(TG.MUSCLES):
        ax = fig.add_subplot(gs[MPOS[j]])
        dress(ax, "b" if j == 0 else None)
        u = np.linspace(0, 1, 100)
        M = np.zeros((100, 6)); M[:, j] = u
        y = TG.quad_design(M) @ beta
        sel = (ndr == 1) & (U[:, j] > 1e-6)
        for k in range(3):
            ax.plot(u, y[:, k], "-", color=ACOL[k], lw=2.0)
            ax.plot(U[sel, j], P[sel, k], "o", color=ACOL[k], ms=5.5, mfc="white",
                    mew=1.4)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_title(mus, fontsize=13, color="black", pad=6)
        ax.set_xlabel("drive $m_i$")
        if MPOS[j][1] == 0 or j == 0:
            ax.set_ylabel("pose (deg)")

    # --- (c) the cross terms ---------------------------------------------
    ax = fig.add_subplot(gs[2, 1]); dress(ax, "c")
    B = np.zeros((3, 6, 6))
    for k in range(3):
        c = beta[6:, k]
        B[k][np.diag_indices(6)] = c[:6]
        for t, (i, j) in enumerate(TG.PAIRS):
            B[k][i, j] = B[k][j, i] = c[6 + t]
    Bm = np.abs(B).max(0)
    im = ax.imshow(Bm, cmap="viridis")
    ax.set_xticks(range(6)); ax.set_xticklabels(TG.MUSCLES, fontsize=11)
    ax.set_yticks(range(6)); ax.set_yticklabels(TG.MUSCLES, fontsize=11)
    ax.set_xlabel("")
    ax.text(0.5, 1.13, r"$\max_k |b^k_{ij}|$  (deg)", transform=ax.transAxes,
            ha="center", fontsize=12.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax.spines[["top", "right"]].set_visible(True)

    fig.savefig(a.out, dpi=170, facecolor="white", bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
