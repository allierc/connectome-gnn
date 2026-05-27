"""Zebrafish HD-circuit connectome summary figure (companion to
figures/drosophila_cx/fig_connectome_summary.py).

Two panels:
  (a) Signed W_con, z-scored over non-zero entries and clipped to +/-3.
  (b) Binary support mask (black = edge present, white = absent).

Data source: load_zebrafish_hd_connectome reads
  figures/zebrafish/zebrafish_connectome_HD/{neurons.csv, connections.csv}
(produced by fetch_zebrafish_connectivity_HD.py), applies the same
Dale-flip + spectral-radius normalisation Hulse 2025 uses for the
drosophila CX (Delta7/ER6 -> here: IPNd*/IPNds*, which are GABAergic in
the larval-zebrafish HD ring per Petrucco et al. 2023).

Output: figures/zebrafish/fig_connectome_summary_HD.png
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from connectome_gnn.generators.connconstr_data import load_zebrafish_hd_connectome


def _draw_block_grid(ax, type_ids, names, color="k", alpha=0.5, lw=0.3):
    """Reorder rows/cols by type id, draw block boundary lines, and put
    type-name tick labels at block centres."""
    order = np.argsort(type_ids, kind="stable")
    b = np.where(np.diff(type_ids[order]) != 0)[0] + 0.5
    for x in b:
        ax.axvline(x, color=color, lw=lw, alpha=alpha)
        ax.axhline(x, color=color, lw=lw, alpha=alpha)
    boundaries = np.concatenate([[0], b + 0.5, [type_ids.size]])
    centres = (boundaries[:-1] + boundaries[1:]) / 2 - 0.5
    lab = [names[int(type_ids[order[int(c)]])] for c in centres]
    ax.set_xticks(centres); ax.set_xticklabels(lab, fontsize=6,
                                                rotation=60, ha="right")
    ax.set_yticks(centres); ax.set_yticklabels(lab, fontsize=6)
    ax.set_xlabel("presynaptic", fontsize=9)
    ax.set_ylabel("postsynaptic", fontsize=9)


def _panel_W(ax, W, type_ids, names):
    nz = W[W != 0]
    mu, sd = float(nz.mean()), float(nz.std())
    Z = np.where(W != 0, (W - mu) / max(sd, 1e-8), 0.0).clip(-3.0, 3.0)
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-3.0, vmax=3.0,
                   interpolation="nearest", aspect="equal")
    _draw_block_grid(ax, type_ids, names, color="k", alpha=0.5)
    ax.set_title(r"$W^{\mathrm{con}}$ (signed, $z$-scored, $\pm 3$ clip)",
                 fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=7)


def _panel_mask(ax, W, type_ids, names):
    M = (W != 0).astype(np.float32)
    ax.imshow(M, cmap="binary", vmin=0, vmax=1,
              interpolation="nearest", aspect="equal")
    _draw_block_grid(ax, type_ids, names, color="r", alpha=0.6)
    density = float(M.sum()) / float(M.size)
    ax.set_title(
        rf"$\mathrm{{supp}}(W^{{\mathrm{{con}}}})$  "
        rf"(density $={density:.3f}$)",
        fontsize=10,
    )


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datapath",
        default=os.path.join(here, "zebrafish_connectome_HD"),
        help="directory with neurons.csv and connections.csv",
    )
    p.add_argument("--out", default=os.path.join(here, "fig_connectome_summary_HD.png"))
    args = p.parse_args()

    cx = load_zebrafish_hd_connectome(args.datapath)
    W = cx["J_effective"]
    type_ids = np.asarray(cx["neuron_types"]).astype(int)
    names = list(cx["type_names"])
    N = cx["N"]
    nnz = int((W != 0).sum())
    print(f"N = {N}   non-zero edges = {nnz}   density = {nnz / W.size:.4f}")
    print(f"n_epg (IPNd* + IPNds* = r1pi HD ring) = {cx['n_epg']}")
    print(f"pen_subpop_ix sizes: "
          f"{ {k: len(v) for k, v in cx['pen_subpop_ix'].items()} }")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _panel_W(axes[0], W, type_ids, names)
    _panel_mask(axes[1], W, type_ids, names)

    for k, ax in enumerate(axes.flat):
        ax.text(-0.10, 1.04, "ab"[k], transform=ax.transAxes,
                ha="left", va="top", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
