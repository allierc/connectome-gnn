"""Connectivity matrix of the colleagues' oculomotor reconstruction, rendered
in the layout of Figure 2(a) of ``docs/zebrafish_paper.tex``.

This is a LOOK-AT-THE-DATA script, not a circuit definition. It reads the raw
pickle, groups cells by cell type for display, and draws the support mask with
coloured type strips — the same renderer the HD figure uses
(``figures/zebrafish/fig_2_connectome._panel_partition_matrix``), so the two
matrices are directly comparable by eye. Nothing here decides which cells are
recurrent, which are afferent, or which are inhibitory: those are the circuit
owner's calls (see ``docs/HOWTO_add_oculomotor_circuit.md`` §2).

Expected pickle schema (Oculomotor_sortedData_081126.pkl, 2949 cells):

    Type                     (N,)    str     42 cell types
    Hemi                     (N,)    str     'left' / 'right'
    NeuronIDs                (N,)    int64   EM body ids (labels only)
    adjacency_matrix_counts  (N, N)  float64 synapse counts
    adjacency_matrix_size    (N, N)  float64 synapse contact area

Compared with the HD pickle (IPN_sortedData_060826.pkl) this file has no
``IPN_angles`` (no per-cell ordering variable) and no ``Location``.

ORIENTATION. ``A[i, j]`` is assumed to be pre i -> post j, the convention the
HD reconstruction uses. The figure plots ``W = A.T`` so rows are postsynaptic,
matching Figure 2. ``--check-orientation`` runs the same empirical test used
for the HD file — a purely sensory population (retinal ganglion cells) must be
overwhelmingly presynaptic — and prints which reading the data supports. Use
``--transpose`` if it says the opposite.

FAMILY GROUPING is display-only and mechanical: the substring before the first
underscore, case-folded so ``V2_ipsi_m`` and ``v2_contra_l`` land together.
Default block order is descending family size; pass ``--family-order`` to
impose an anatomical order (e.g. sensory -> integrator -> motor), which makes
the block structure far easier to read.

Usage::

    python scripts/plot_oculomotor_connectome.py --check-orientation
    python scripts/plot_oculomotor_connectome.py \\
        --family-order RGC Tectum AF7 AF5 PVPN INTG INTGx Burst OMN AMN AIN
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "figures", "zebrafish"))

_DEFAULT_PKL = os.path.join(
    _REPO, "config", "zebrafish", "Oculomotor_sortedData_081126.pkl")
_DEFAULT_OUT = os.path.join(
    _REPO, "config", "oculomotor_connectome_matrix.png")

# Okabe-Ito extended — colour-blind safe, and the first five match the
# palette Figure 1 / Figure 2 of the HD paper use for their partitions.
_PALETTE = ["#cf222e", "#d29922", "#2ea043", "#1f6feb", "#8957e5",
            "#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9",
            "#d55e00", "#666666", "#8c564b", "#17becf", "#bcbd22",
            "#7f7f7f", "#aec7e8", "#ffbb78"]


def load_pickle(path):
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    types = np.asarray(d["Type"]).astype(str)
    hemi = np.asarray(d["Hemi"]).astype(str)
    body = np.asarray(d["NeuronIDs"]).astype(np.int64)
    return d, types, hemi, body


def family_of(type_name):
    """Display-only coarse family: the token before the first underscore,
    case-folded (``V2_ipsi_m`` and ``v2_contra_l`` -> ``V2``)."""
    head = str(type_name).split("_")[0]
    return head.upper() if head.lower().startswith("v2") else head


def check_orientation(A, types, sensory_prefix="RGC"):
    """Is ``A[i, j]`` pre->post or post->pre?

    A retinal-ganglion-cell population is sensory input: it must send far more
    synaptic weight than it receives. Under the correct reading its ROW sum
    (outgoing) dominates its COLUMN sum (incoming). Returns True if the data
    supports A[i, j] = pre i -> post j.
    """
    sel = np.array([str(t).startswith(sensory_prefix) for t in types])
    if not sel.any():
        print(f"[orient] no {sensory_prefix}* cells found — cannot check")
        return None
    out_w = A[sel, :].sum()
    in_w = A[:, sel].sum()
    ratio = out_w / max(in_w, 1e-9)
    verdict = ratio > 1.0
    print(f"[orient] {sensory_prefix}* cells (n={int(sel.sum())}): "
          f"row-sum(outgoing)={out_w:.4g}  col-sum(incoming)={in_w:.4g}  "
          f"ratio={ratio:.1f}x")
    print(f"[orient] -> A[i,j] is {'pre i -> post j' if verdict else 'POST i <- PRE j'}"
          f"  ({'matches' if verdict else 'CONTRADICTS'} the HD convention)")
    if not verdict:
        print("[orient] pass --transpose to correct the figure.")
    return verdict


def build_figure(W, families, family_order, out_path, *, title_note=""):
    """Panel (a) of the HD Figure 2, with cell-type families in place of the
    HD functional partition. Reuses the paper's own renderer so the support
    mask, dilation, strip geometry, boundary lines and tick placement are
    identical rather than re-implemented."""
    import fig_2_connectome as ref

    colours = {f: _PALETTE[i % len(_PALETTE)]
               for i, f in enumerate(family_order)}
    # The renderer reads the block order and palette off module globals.
    ref.PARTITION_ORDER = list(family_order)
    ref.PARTITION_COLOR = colours

    fig, ax = plt.subplots(figsize=(11, 10))
    ref._panel_partition_matrix(ax, W, families)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    # The renderer labels every block; with 16 families the narrow ones
    # overlap into an unreadable smear along the x axis. Blank the labels of
    # blocks under `min_frac` of the population — the colour strip and the
    # legend still carry their identity — and keep every y label, which has
    # room to stack vertically.
    min_frac = 0.02
    keep = {f for f in family_order
            if (families == f).sum() >= min_frac * families.size}
    ax.set_xticklabels([t.get_text() if t.get_text() in keep else ""
                        for t in ax.get_xticklabels()],
                       rotation=30, ha="right")
    dropped = [f for f in family_order if f not in keep]
    if dropped:
        print(f"[fig] x-labels suppressed for {len(dropped)} narrow blocks "
              f"(<{min_frac:.0%} of cells): {', '.join(dropped)}")

    counts = {f: int((families == f).sum()) for f in family_order}
    ax.legend(
        handles=[Patch(facecolor=colours[f], label=f"{f} ({counts[f]})")
                 for f in family_order],
        loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=6,
        frameon=False, fontsize=8,
    )
    ax.text(0.0, 1.02, "a", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom", ha="left")
    if title_note:
        ax.text(1.0, 1.02, title_note, transform=ax.transAxes, fontsize=8,
                va="bottom", ha="right", color="0.35")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"[fig] wrote {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl", default=_DEFAULT_PKL)
    p.add_argument("--out", default=_DEFAULT_OUT)
    p.add_argument("--weights", default="size", choices=["size", "counts"],
                   help="which adjacency matrix to take the support from")
    p.add_argument("--transpose", action="store_true",
                   help="treat A[i,j] as post<-pre instead of pre->post")
    p.add_argument("--check-orientation", action="store_true",
                   help="run the sensory-population test and report")
    p.add_argument("--family-order", nargs="+", default=None,
                   help="explicit block order (families omitted are appended "
                        "in descending size)")
    args = p.parse_args()

    d, types, hemi, body = load_pickle(args.pkl)
    A = np.asarray(d[f"adjacency_matrix_{args.weights}"], dtype=np.float64)
    N = types.size
    print(f"[data] {os.path.basename(args.pkl)}: N={N}  types={len(set(types))}  "
          f"edges={int((A > 0).sum())}  density={(A > 0).sum() / A.size:.4f}")
    print(f"[data] keys: {sorted(d)}")
    if A.shape != (N, N):
        sys.exit(f"adjacency is {A.shape}, expected ({N}, {N}) — the matrix "
                 f"must be index-aligned with Type/Hemi/NeuronIDs")

    if args.check_orientation:
        check_orientation(A, types)

    # Figure 2 plots W[post, pre]; the pickle is A[pre, post].
    W = A if args.transpose else A.T

    families = np.array([family_of(t) for t in types], dtype=object)
    uniq, cnt = np.unique(families.astype(str), return_counts=True)
    by_size = [str(f) for f in uniq[np.argsort(-cnt)]]
    if args.family_order:
        order = [f for f in args.family_order if f in by_size]
        order += [f for f in by_size if f not in order]
        missing = [f for f in args.family_order if f not in by_size]
        if missing:
            print(f"[warn] --family-order names not in the data, ignored: "
                  f"{missing}")
    else:
        order = by_size
    print(f"[data] {len(order)} display families: "
          + ", ".join(f"{f}({int((families == f).sum())})" for f in order))

    note = (f"N={N}, {len(set(types))} types, "
            f"{int((A > 0).sum())} edges, weight={args.weights}")
    build_figure(W, families, order, args.out, title_note=note)


if __name__ == "__main__":
    main()
