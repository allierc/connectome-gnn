"""Figure 1 of the oculomotor note — twin of the zebrafish paper's Figure 1.

Three panels in one matplotlib figure, so panel letters and fonts render at
the same physical scale on the page:

  (a) full skeletons against the ROI silhouettes
  (b) cell-body somas, same view
  (c) skeletons and (d) somas recoloured by Dale sign, red = inhibitory
  (e) the velocity-tracking computation: target velocity in, eye angles out

Panel (d) of the paper's Figure 1 — the second circuit variant — has no
counterpart here: this circuit has one afferent taxonomy, not two.

Panels (a) and (b) come from ``fig_zebrafish_anatomy_3d_HD`` and panel (c)
from the box/arrow primitives of ``fig_zebrafish_circuit_variants``, so this
is the same renderer as the paper figure with the content swapped, not a
reimplementation of it.

Usage::

    python fig_1_oculomotor_overview.py [--view dorsal|lateral] [--bg white]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_zebrafish_anatomy_3d_HD as ref                    # noqa: E402
import fig_zebrafish_circuit_variants as cv                  # noqa: E402
from fig_oculomotor_anatomy import ROLE, COLOR, KEEP_ROIS    # noqa: E402

PANEL_LABEL_KW = dict(fontsize=13, fontweight="bold", va="top", ha="left")
# Dale sign per role, read off the `sign:` field of every cell type in
# config/zebrafish/zebrafish_om_intg_285_v1.yaml. Only the contralateral
# integrator is inhibitory; the afferents, the ipsilateral integrator and
# both motor pools are excitatory.
EI_OF_ROLE = {
    "afferent AF5": "excitatory (E)",
    "INTG ipsi (E)": "excitatory (E)",
    "INTG contra (I)": "inhibitory (I)",
    "AMN / lateral rectus": "excitatory (E)",
    "AIN / medial rectus": "excitatory (E)",
}
EI_COLOR = {"excitatory (E)": (0.12, 0.35, 0.85),     # blue
            "inhibitory (I)": (0.85, 0.13, 0.16)}     # red
# Fills echo the role colours of panels a/b, pale enough to take black text.
F_AF5 = "#c7d8e8"      # afferent, matches the blue skeletons
F_INTG_E = "#cfe8d2"   # excitatory ipsilateral integrator, green
F_INTG_I = "#f3cdd0"   # inhibitory contralateral integrator, red
F_AMN = "#ddd0f0"      # lateral rectus motor pool, purple
F_AIN = "#f4d6e6"      # medial rectus motor pool, pink
F_REC = "#f6f1e6"      # container behind the integrator


def _by_sign(types, segs, somas):
    """Collapse the five role categories onto the two Dale signs, merging the
    pre-extracted segment and soma arrays so the E/I row costs one dict merge
    rather than a second pass over the skeletons."""
    ei_types = np.array([EI_OF_ROLE.get(str(t), "other") for t in types])
    segs_ei: dict[str, list] = {}
    somas_ei: dict[str, tuple[list, list]] = {}
    for role, sign in EI_OF_ROLE.items():
        if role in segs:
            segs_ei.setdefault(sign, []).append(segs[role])
        if role in somas:
            pos, rad = somas[role]
            somas_ei.setdefault(sign, ([], []))
            somas_ei[sign][0].append(pos)
            somas_ei[sign][1].append(rad)
    segs_ei = {k: np.concatenate(v, axis=0) for k, v in segs_ei.items()}
    somas_ei = {k: (np.concatenate(p, axis=0), np.concatenate(r, axis=0))
                for k, (p, r) in somas_ei.items()}
    return ei_types, segs_ei, somas_ei


def _stack(ax, cx, cy, lines, gap=0.085):
    """Centre a short stack of lines on (cx, cy).

    ``cv._draw_afferent_block`` spaces its three lines by offsets tuned for a
    tall panel; this panel is wide and short, so the same offsets in axes
    fraction collide. Spacing is passed in instead.
    """
    n = len(lines)
    for k, (txt, fs) in enumerate(lines):
        y = cy + (n - 1) / 2 * gap - k * gap
        cv._text(ax, (cx, y), [txt], fontsize=fs, ha="center")


def _panel_e(ax):
    """Velocity in, eye angles out — the computation, not the anatomy.

    The chain of Section 4.6, left to right: the target velocity that reaches
    the afferents, the sign-locked recurrent integrator, the non-negative
    motor pools, the push-pull commands, the eye plant, the gaze angles.
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    LF, CF, SF = cv.LABEL_FS, cv.COUNT_FS, cv.SUB_FS

    # --- input ------------------------------------------------------------
    cv._text(ax, (0.004, 0.56), [r"$(\dot x,\dot y)$"],
             fontsize=LF + 4, ha="left")
    cv._text(ax, (0.004, 0.44), ["target velocity"], fontsize=SF, ha="left")
    cv._arrow(ax, (0.075, 0.52), (0.105, 0.52), color=cv.COL_OMEGA, lw=2.2)

    cv._box(ax, (0.105, 0.40), (0.155, 0.24), fill=F_AF5)
    _stack(ax, 0.1825, 0.52, [("AF5 L / R", LF), (r"$n=29+12$", CF),
                              (r"$\mathbf{I}=\hat W^{\rm in}\,(\dot x,\dot y)^{\!\top}$", CF)])

    # --- recurrent integrator --------------------------------------------
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (0.295, 0.16), 0.265, 0.74, boxstyle="round,pad=0.012",
        linewidth=1.0, edgecolor="0.45", facecolor=F_REC, zorder=1))
    cv._text(ax, (0.4275, 0.94), ["recurrent integrator  ($n=182$)"],
             fontsize=LF, ha="center")
    cv._box(ax, (0.315, 0.65), (0.225, 0.19), fill=F_INTG_E)
    _stack(ax, 0.4275, 0.745, [("INTG ipsi  (E)", LF), (r"$n=38+27$", CF)],
           gap=0.075)
    cv._box(ax, (0.315, 0.215), (0.225, 0.19), fill=F_INTG_I)
    _stack(ax, 0.4275, 0.31, [("INTG contra  (I)", LF), (r"$n=34+18$", CF)],
           gap=0.075)
    # Mutual inhibition across the midline — the line-attractor motif. The
    # arrows flank the equation rather than crossing it.
    cv._arrow(ax, (0.335, 0.635), (0.335, 0.425), color="#b03a3a", lw=1.4)
    cv._arrow(ax, (0.520, 0.425), (0.520, 0.635), color="#3a8a45", lw=1.4)
    _stack(ax, 0.4275, 0.525,
           [(r"$\tau_i\dot v_i=-v_i+\sum_j\hat W_{ij}\,r_j+I_i$", CF),
            (r"$\hat W_{ij}=|\hat S_{ij}|\,{\rm sign}(W^{\rm con}_{ij})$", SF)],
           gap=0.085)

    # --- motor pools and push-pull commands -------------------------------
    # LR and MR have an anatomical pool in the selected 285 cells; SR and IR
    # would come from OMN, which section 1.5 leaves out of the pool.
    cv._box(ax, (0.595, 0.40), (0.175, 0.30), fill=F_AMN)
    _stack(ax, 0.6825, 0.55, [("AMN / AIN", LF), (r"$n=92+35$", CF),
                              (r"$\mathbf{m}=[\hat W^{\rm out}\mathbf{r}]_+$", CF),
                              ("LR, MR;  SR, IR from OMN", SF)], gap=0.075)

    cv._box(ax, (0.805, 0.40), (0.135, 0.30), fill=F_AIN)
    _stack(ax, 0.8725, 0.55, [("eye plant", LF),
                              (r"$u_\theta=m_{\rm LR}-m_{\rm MR}$", CF),
                              (r"$u_\varphi=m_{\rm SR}-m_{\rm IR}$", CF),
                              (r"$\Phi,\ \omega_n,\ \zeta$  frozen", SF)],
           gap=0.075)

    cv._arrow(ax, (0.260, 0.52), (0.295, 0.52), color="0.4", lw=1.4)
    cv._arrow(ax, (0.560, 0.52), (0.595, 0.52), color="0.4", lw=1.4)
    cv._arrow(ax, (0.770, 0.52), (0.805, 0.52), color="0.4", lw=1.4)
    cv._arrow(ax, (0.940, 0.52), (0.968, 0.52), color="0.4", lw=1.4)
    cv._text(ax, (0.996, 0.58), [r"$(\theta,\varphi)$"],
             fontsize=LF + 4, ha="right")
    cv._text(ax, (0.996, 0.45), ["eye angles"], fontsize=SF, ha="right")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anatomy", default=os.path.join(HERE, "zebrafish_anatomy_OCULO"))
    p.add_argument("--out", default=os.path.join(HERE, "fig_1_oculomotor_overview.png"))
    p.add_argument("--view", default="dorsal", choices=["dorsal", "lateral"])
    p.add_argument("--downsample", type=int, default=4)
    p.add_argument("--soma-scale", type=float, default=0.30)
    p.add_argument("--bg", default="white", choices=["white", "black"])
    a = p.parse_args()

    ref._type_to_category = lambda s: ROLE.get(s, "other")
    ref.TYPE_COLOR = dict(COLOR)
    ref.TYPE_ORDER = list(COLOR)

    nl, types = ref._load_skeletons(a.anatomy, downsample=a.downsample)
    rois = {k: v for k, v in ref._load_rois(a.anatomy).items()
            if k in set(KEEP_ROIS)}
    segs = ref._extract_segments_by_type(nl, types)
    somas = {t: (pos, np.asarray(rad) * a.soma_scale)
             for t, (pos, rad) in ref._extract_somas_by_type(nl, types).items()}
    elev, azim = (90, -90) if a.view == "dorsal" else (0, -90)
    print(f"{len(nl)} skeletons, {len(rois)} ROIs, {a.view} view")

    fig = plt.figure(figsize=(13.5, 10.6), facecolor=a.bg)
    gs = GridSpec(3, 2, height_ratios=[0.86, 0.86, 1.0], hspace=0.02,
                  wspace=0.03, figure=fig)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    # Row 1 — the five functional roles.
    ref.draw_anatomy_panels(ax_a, ax_b, nl, types, rois, elev=elev, azim=azim,
                            background=a.bg, alpha_mesh=0.10,
                            segs_by_type=segs, somas_by_type=somas,
                            legend=True)

    # Row 2 — the same cells, recoloured by the Dale sign the model locks.
    ei_types, segs_ei, somas_ei = _by_sign(types, segs, somas)
    n_e = int((ei_types == "excitatory (E)").sum())
    n_i = int((ei_types == "inhibitory (I)").sum())
    print(f"Dale sign: {n_e} excitatory, {n_i} inhibitory of {len(ei_types)}")
    ref.TYPE_COLOR = dict(EI_COLOR)
    ref.TYPE_ORDER = list(EI_COLOR)
    ref.draw_anatomy_panels(ax_c, ax_d, nl, ei_types, rois, elev=elev,
                            azim=azim, background=a.bg, alpha_mesh=0.10,
                            segs_by_type=segs_ei, somas_by_type=somas_ei,
                            legend=True)

    _panel_e(ax_e)

    fg = "white" if a.bg == "black" else "black"
    for ax, L in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d"),
                  (ax_e, "e")):
        ax.text(0.0, 1.0, L, transform=ax.transAxes, color=fg, **PANEL_LABEL_KW)
    fig.savefig(a.out, dpi=180, facecolor=a.bg, bbox_inches="tight")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
