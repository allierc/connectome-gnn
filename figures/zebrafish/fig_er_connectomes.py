"""Appendix figure: the three null-connectome families vs the measured circuit.

Companion of Figure 2 (``fig_2_connectome.py``): the same z-scored,
partition-sorted signed-matrix rendering, applied to the measured 917-cell
connectome and one representative of each of the three randomization families
used by the 'is the computation specific to the wiring?' controls.

Three null families (all matched on size, density and the empirical |W|
magnitude distribution, all inheriting the node identities so the afferent gate
and dIPN decoder are unchanged):

  1. full-network ER, recurrent inhibitory-fraction sweep   (er_1..er_4)
     -- a brand-new uniform random support over the whole matrix; the block
        structure is destroyed; p_inh is swept 0.4 -> 1.0.
  2. full-network ER, overall-E/I matched                   (er_ei_1..er_ei_10)
     -- the same uniform random support, but the overall inhibitory edge
        fraction is matched to the measured 66.3%; 10 seeds = a null band.
  3. block-shuffle (block-preserving)                        (bs_1..bs_10)
     -- the measured block-to-block connectivity is kept exactly (each
        functional block-pair retains its edge count and magnitudes); only the
        fine wiring WITHIN each block-pair is randomized; 10 seeds.

Panels:
  (a) measured W_con (the functional-partition block structure is visible).
  (b) full-network ER (er_4): no block structure survives.
  (c) E/I-matched ER (er_ei_1): no block structure, colour balance matched.
  (d) block-shuffle (bs_1): block structure PRESERVED, fine wiring randomized.
  (e) block-density correlation to the measured circuit, per family -- the
      quantitative form of the block-structure test (~1 for block-shuffle, ~0
      for both ER families).
  (f) inhibitory edge fraction per family, with the measured 66.3% marked.

Usage:
    python figures/zebrafish/fig_er_connectomes.py --out_dir figures/zebrafish/
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, _HERE)

BASE_CIRCUIT = "zebrafish_HD_IPN_917_artr_pt1"
LABEL_FS, TICK_FS, PANEL_LABEL_FS = 13, 11, 18

FULL_ER = [f"er_{i}.npz" for i in (1, 2, 3, 4)]
EI_ER = [f"er_ei_{i}.npz" for i in range(1, 11)]
BLOCK = [f"bs_{i}.npz" for i in range(1, 11)]
FAMILIES = (("full-network ER", FULL_ER, "#1f6fb3"),
            ("E/I-matched ER", EI_ER, "#d1495b"),
            ("block-shuffle", BLOCK, "#2a9d8f"))


def _panel_label(ax, letter, y=1.02, x=-0.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL_LABEL_FS,
            fontweight="bold", va="bottom", ha="right")


def _Dmap(J, part, blocks):
    """Per block-pair edge density (fraction of possible off-diagonal cells)."""
    idx = {b: np.where(part == b)[0] for b in blocks}
    D = np.zeros((len(blocks), len(blocks)))
    for a, pb in enumerate(blocks):
        for b, qb in enumerate(blocks):
            sub = J[np.ix_(idx[pb], idx[qb])]
            poss = sub.size - (pb == qb) * min(sub.shape)
            D[a, b] = (sub != 0).sum() / max(poss, 1)
    return D


def _inh_frac(J):
    nz = J != 0
    return float((J < 0).sum() / nz.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=_HERE)
    args = p.parse_args()

    from fig_2_connectome import _panel_W, _partition_ids
    from connectome_gnn.generators.circuits import get_circuit
    from connectome_gnn.utils import (
        graphs_data_path, load_data_fallback_roots, set_data_root,
    )
    roots = load_data_fallback_roots()
    if roots:
        set_data_root(roots[0])
    er_dir = graphs_data_path("zebrafish", "er_connectomes")

    def _npz(name):
        d = np.load(os.path.join(er_dir, name), allow_pickle=True)
        return np.asarray(d["J_effective"], dtype=np.float32)

    base = get_circuit(BASE_CIRCUIT).as_loader_dict()
    nt = np.asarray(base["neuron_types"]).astype(int)
    names = list(base["type_names"])
    part = _partition_ids(nt, names)
    blocks = list(dict.fromkeys(part.tolist()))
    W_con = np.asarray(base["J_effective"], dtype=np.float32)
    D_base = _Dmap(W_con, part, blocks)
    measured_inh = _inh_frac(W_con)
    ref_nz = W_con[W_con != 0]

    # representative matrices for the visual panels
    W_full = _npz("er_4.npz")
    W_ei = _npz("er_ei_1.npz")
    W_bs = _npz("bs_1.npz")

    # quantitative summaries per family
    fam_corr, fam_inh = {}, {}
    for fam, members, _c in FAMILIES:
        cors, inhs = [], []
        for f in members:
            J = _npz(f)
            cors.append(float(np.corrcoef(D_base.ravel(),
                                          _Dmap(J, part, blocks).ravel())[0, 1]))
            inhs.append(100 * _inh_frac(J))
        fam_corr[fam] = np.array(cors)
        fam_inh[fam] = np.array(inhs)

    # ---------------------------------- figure ------------------------------
    fig = plt.figure(figsize=(19, 5.4))
    gs = fig.add_gridspec(1, 4, wspace=0.30,
                          left=0.04, right=0.99, top=0.88, bottom=0.07)

    mats = [("a", W_con, "measured"),
            ("b", W_full, "full-network ER"),
            ("c", W_ei, "E/I-matched ER"),
            ("d", W_bs, "block-shuffle")]
    for j, (lab, W, title) in enumerate(mats):
        ax = fig.add_subplot(gs[0, j])
        _panel_W(ax, W, nt.copy(), names,
                 ref_nz=(None if lab == "a" else ref_nz),
                 partition=part, show_cbar=False)
        ax.text(0.5, 1.06, title, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=TICK_FS + 1, fontweight="bold")
        _panel_label(ax, lab)

    out_path = os.path.join(args.out_dir, "fig_er_connectomes.png")
    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[fig_er_connectomes] block-density corr to measured:")
    for fam, _m, _c in FAMILIES:
        print(f"   {fam:18s} {fam_corr[fam].mean():.3f} "
              f"(inh {fam_inh[fam].mean():.1f}%)")
    print(f"[fig_er_connectomes] wrote {out_path}")


if __name__ == "__main__":
    main()
