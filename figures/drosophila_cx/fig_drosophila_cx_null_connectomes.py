"""Appendix figure: the three null-connectome families vs the measured
Drosophila CX (drosophila_cx_338_v1). Fly companion of the zebrafish
``fig_er_connectomes.py``.

Renders the measured 338-cell connectome and one representative of each null
family (built by ``build_null_connectomes.py``) as z-scored, cell-type-sorted
signed matrices (red excitatory / blue inhibitory), and prints the per-family
block-density correlation to the measured circuit + inhibitory edge fraction.

  (a) measured W_con           — cell-type block structure visible
  (b) full-network ER (er_5)   — block structure destroyed
  (c) E/I-matched ER (er_ei_1) — structureless, ~31% inhibitory matched
  (d) block-shuffle (bs_1)     — block-to-block kept, fine wiring randomised

Usage:
    python figures/drosophila_cx/fig_drosophila_cx_null_connectomes.py
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

BASE_CIRCUIT = "drosophila_cx_338_v1"
FAMILIES = (("full-network ER", [f"er_{i}.npz" for i in range(1, 6)], "#1f6fb3"),
            ("E/I-matched ER", [f"er_ei_{i}.npz" for i in range(1, 6)], "#d1495b"),
            ("block-shuffle", [f"bs_{i}.npz" for i in range(1, 6)], "#2a9d8f"))


def _Dmap(J, part, blocks):
    idx = {b: np.where(part == b)[0] for b in blocks}
    D = np.zeros((len(blocks), len(blocks)))
    for a, pb in enumerate(blocks):
        for b, qb in enumerate(blocks):
            sub = J[np.ix_(idx[pb], idx[qb])]
            poss = sub.size - (pb == qb) * min(sub.shape)
            D[a, b] = (sub != 0).sum() / max(poss, 1)
    return D


def _inh_frac(J):
    return float((J < 0).sum() / (J != 0).sum())


def _panel(ax, J, nt, names, ref_nz, title, lab):
    """z-scored signed-matrix render with cell-type block lines."""
    scale = float(np.percentile(np.abs(ref_nz), 90)) if ref_nz.size else 1.0
    Z = np.clip(np.sign(J) * np.sqrt(np.abs(J) / (scale + 1e-12)), -1, 1)
    ax.imshow(Z, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal",
              interpolation="nearest", origin="upper")
    bounds, cur = [], int(nt[0])
    for i, t in enumerate(nt):
        if int(t) != cur:
            bounds.append(i); cur = int(t)
    for b in bounds:
        ax.axhline(b - 0.5, color="k", lw=0.3, alpha=0.4)
        ax.axvline(b - 0.5, color="k", lw=0.3, alpha=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(-0.04, 1.02, lab, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="bottom", ha="right")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=_HERE)
    args = p.parse_args()
    from connectome_gnn.generators.circuits import get_circuit
    from connectome_gnn.utils import (
        graphs_data_path, load_data_fallback_roots, set_data_root,
    )
    roots = load_data_fallback_roots()
    if roots:
        set_data_root(roots[0])
    nd = graphs_data_path("drosophila_cx", "er_connectomes")

    def _npz(name):
        return np.asarray(np.load(os.path.join(nd, name), allow_pickle=True)
                          ["J_effective"], dtype=np.float32)

    base = get_circuit(BASE_CIRCUIT).as_loader_dict()
    nt = np.asarray(base["neuron_types"]).astype(int)
    names = list(base["type_names"])
    part = nt
    blocks = list(dict.fromkeys(part.tolist()))
    W_con = np.asarray(base["J_effective"], dtype=np.float32)
    ref_nz = W_con[W_con != 0]
    D_base = _Dmap(W_con, part, blocks)
    measured_inh = 100 * _inh_frac(W_con)

    mats = [("a", W_con, "measured"), ("b", _npz("er_5.npz"), "full-network ER"),
            ("c", _npz("er_ei_1.npz"), "E/I-matched ER"),
            ("d", _npz("bs_1.npz"), "block-shuffle")]
    fig = plt.figure(figsize=(16, 4.6))
    gs = fig.add_gridspec(1, 4, wspace=0.12, left=0.03, right=0.99,
                          top=0.88, bottom=0.04)
    for j, (lab, W, title) in enumerate(mats):
        _panel(fig.add_subplot(gs[0, j]), W, nt, names,
               ref_nz if lab != "a" else ref_nz, title, lab)
    out = os.path.join(args.out_dir, "fig_drosophila_cx_null_connectomes.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"[null] measured inh={measured_inh:.1f}%  block-density corr / inh per family:")
    for fam, members, _c in FAMILIES:
        cors = [float(np.corrcoef(D_base.ravel(),
                _Dmap(_npz(f), part, blocks).ravel())[0, 1]) for f in members]
        inhs = [100 * _inh_frac(_npz(f)) for f in members]
        print(f"   {fam:18s} corr={np.mean(cors):.3f}  inh={np.mean(inhs):.1f}%")
    print(f"[null] wrote {out}")


if __name__ == "__main__":
    main()
