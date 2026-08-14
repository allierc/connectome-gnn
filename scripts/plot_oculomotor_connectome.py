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


_ROLE_PALETTE = {
    ("afferent", "E"):  ["#1f6feb", "#56b4e9"],   # blues
    ("recurrent", "E"): ["#2ea043", "#009e73"],   # greens
    ("recurrent", "I"): ["#cf222e", "#d55e00"],   # reds
    ("output", "E"):    ["#8957e5", "#cc79a7"],   # purples
    ("output", "I"):    ["#6a3d9a", "#a6761d"],
}
_ROLE_ORDER = ("afferent", "recurrent", "output")


def _spec_colours(specs):
    """One colour per declared type, shaded by (role, sign) so the palette
    itself carries the biology: blues in, greens excitatory recurrent, reds
    inhibitory recurrent, purples out."""
    used, out = {}, {}
    for s in specs:
        key = (s.role, s.sign)
        pool = _ROLE_PALETTE.get(key, ["#666666", "#999999"])
        k = used.get(key, 0)
        out[s.name] = pool[k % len(pool)]
        used[key] = k + 1
    return out


_EI_COLOR = {"E": "#1f4fd8", "I": "#d81b26"}      # blue excitatory, red inhibitory


def _panel_signed_matrix(ax, W, partition, sign_of_type, ref):
    """The same matrix as the support panel, with each synapse coloured by the
    Dale sign of its PRESYNAPTIC cell: blue excitatory, red inhibitory.

    This is what the model actually multiplies -- the sign lock of section 4.4
    of the note, ``W_hat_ij = |S_hat_ij| sign(W_con_ij)`` -- rather than the
    support mask, which cannot show it. The sign is a property of the column,
    so the panel reads as vertical bands: an inhibitory type is a red column
    block, whatever its targets.

    Magnitudes are log-compressed because synapse contact area spans four
    decades; without it the panel is a handful of dark pixels on white.
    """
    from scipy.ndimage import grey_dilation

    order_key = {k: i for i, k in enumerate(ref.PARTITION_ORDER)}
    perm = np.argsort([order_key.get(p, len(ref.PARTITION_ORDER))
                       for p in partition], kind="stable")
    W_sorted = W[np.ix_(perm, perm)]
    part_sorted = partition[perm]

    mag = np.log1p(np.abs(W_sorted))
    mag = grey_dilation(mag, size=(2, 2))            # visibility at panel dpi
    sign = np.array([1.0 if sign_of_type.get(str(p), "E") == "E" else -1.0
                     for p in part_sorted])
    signed = mag * sign[None, :]                     # sign is the column's
    vmax = float(np.percentile(mag[mag > 0], 99)) if (mag > 0).any() else 1.0
    im = ax.imshow(signed, cmap="RdBu", vmin=-vmax, vmax=vmax,
                   interpolation="nearest", aspect="equal")

    ref._add_partition_strips(ax, part_sorted, alpha_overlay=0.0,
                              boundary_color="0.35", boundary_lw=0.5)
    ref._set_partition_tick_labels(ax, part_sorted)
    ax.set_xlabel("presynaptic", fontsize=13)
    ax.set_ylabel("postsynaptic", fontsize=13)
    return im


def build_pair_figure(A, types, hemi, specs, families, family_order, out_path):
    """(a) the whole reconstruction, (b) the sub-circuit this config selects,
    (c) the same sub-circuit signed by the Dale assignment.

    Panel (b) orders cells afferent -> recurrent -> output, and within a type
    left hemisphere before right, so the L/R block structure of the gate is
    visible directly in the matrix. Panel (c) repeats that ordering with the
    signs applied, which is the only view in which the E/I claim of section
    1.4 is falsifiable by eye.
    """
    import fig_2_connectome as ref

    names = [s.name for s in specs]
    role_rank = {s.name: _ROLE_ORDER.index(s.role or "recurrent")
                 for s in specs}
    type_rank = {n: i for i, n in enumerate(names)}
    sel = np.where(np.isin(types, names))[0]
    # Pre-sort by (role, declaration order, hemisphere); the renderer's
    # stable argsort over the block key then preserves this within-block.
    key = [(role_rank[types[i]], type_rank[types[i]],
            0 if str(hemi[i]).lower().startswith("l") else 1) for i in sel]
    sel = sel[np.lexsort((
        [k[2] for k in key], [k[1] for k in key], [k[0] for k in key]))]
    sub = A[np.ix_(sel, sel)].T                      # W[post, pre]
    sub_types = types[sel]
    block_order = sorted(names, key=lambda n: (role_rank[n], type_rank[n]))
    colours = _spec_colours(specs)

    fig, axes = plt.subplots(1, 3, figsize=(27.5, 9.2))

    ref.PARTITION_ORDER = list(family_order)
    ref.PARTITION_COLOR = {f: _PALETTE[i % len(_PALETTE)]
                           for i, f in enumerate(family_order)}
    ref._panel_partition_matrix(axes[0], A.T, families)
    keep = {f for f in family_order
            if (families == f).sum() >= 0.02 * families.size}
    axes[0].set_xticklabels([t.get_text() if t.get_text() in keep else ""
                             for t in axes[0].get_xticklabels()],
                            rotation=30, ha="right")
    axes[0].tick_params(labelsize=7)

    ref.PARTITION_ORDER = block_order
    ref.PARTITION_COLOR = colours
    ref._panel_partition_matrix(axes[1], sub, sub_types)
    axes[1].tick_params(labelsize=8)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")

    sign_of_type = {s.name: s.sign for s in specs}
    _panel_signed_matrix(axes[2], sub, sub_types, sign_of_type, ref)
    axes[2].tick_params(labelsize=8)
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=30, ha="right")
    n_e = sum(int((sub_types == n).sum()) for n in names
              if sign_of_type.get(n) == "E")
    axes[2].text(1.0, 1.02,
                 f"{n_e} excitatory, {sel.size - n_e} inhibitory cells",
                 transform=axes[2].transAxes, fontsize=9, va="bottom",
                 ha="right", color="0.35")
    axes[2].legend(
        handles=[Patch(facecolor=_EI_COLOR["E"], label="excitatory presynaptic"),
                 Patch(facecolor=_EI_COLOR["I"], label="inhibitory presynaptic")],
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1,
        frameon=False, fontsize=9)

    for ax, letter in zip(axes, "abc"):
        ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="bottom", ha="right")
    axes[0].text(1.0, 1.02, f"all {A.shape[0]} cells, {len(set(types))} types",
                 transform=axes[0].transAxes, fontsize=9, va="bottom",
                 ha="right", color="0.35")
    axes[1].text(1.0, 1.02,
                 f"{sel.size} cells, {len(names)} types, "
                 f"{int((sub > 0).sum())} edges",
                 transform=axes[1].transAxes, fontsize=9, va="bottom",
                 ha="right", color="0.35")

    lab = {s.name: f"{s.name} — {s.role}, {s.sign}"
           + (f" → {s.effector}" if s.effector else "")
           for s in specs}
    axes[1].legend(
        handles=[Patch(facecolor=colours[n], label=lab[n]) for n in block_order],
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
        frameon=False, fontsize=9)
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
    p.add_argument("--config", default=None,
                   help="a circuit yaml with circuit.cell_types; switches to "
                        "the two-panel figure (whole reconstruction + the "
                        "sub-circuit that config selects)")
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

    if args.config:
        sys.path.insert(0, os.path.join(_REPO, "src"))
        from connectome_gnn.config import NeuralGraphConfig
        cfg = NeuralGraphConfig.from_yaml(args.config)
        specs = cfg.circuit.cell_types or []
        if not specs:
            sys.exit(f"{args.config} declares no circuit.cell_types")
        missing = [s.name for s in specs if s.name not in set(types)]
        if missing:
            sys.exit(f"cell_types absent from the reconstruction: {missing}")
        n_sel = int(np.isin(types, [s.name for s in specs]).sum())
        print(f"[cfg] {cfg.circuit.name}: {len(specs)} types, {n_sel} cells "
              f"| afferent={cfg.circuit.types_by_role('afferent')} "
              f"recurrent={cfg.circuit.types_by_role('recurrent')} "
              f"output={cfg.circuit.types_by_role('output')} "
              f"inhibitory={cfg.circuit.inhibitory_types()}")
        unsigned = [s.name for s in specs if s.sign is None]
        if unsigned:
            print(f"[warn] no Dale sign declared for: {unsigned}")
        build_pair_figure(A, types, hemi, specs, families, order, args.out)
        return

    note = (f"N={N}, {len(set(types))} types, "
            f"{int((A > 0).sum())} edges, weight={args.weights}")
    build_figure(W, families, order, args.out, title_note=note)


if __name__ == "__main__":
    main()
