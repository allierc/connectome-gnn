"""Two-panel connectome summary for the path-integration-extended
Drosophila central complex (drosophila_cx_338_v1), in the style of
Figure 2 of docs/zebrafish.tex (fig_2_connectome.py).

The 338-cell circuit is the strict superset of the 156-cell heading core
with the FB columnar / vector families appended. Both panels share the same
row/column ordering: neurons are sorted by a four-way *functional* partition
that mirrors the heading-integration circuit of Figure 1c/d --

  HD ring   (recurrent substrate)        EPG, EPGt, PEG, Delta7, ER6
  PEN       (angular-velocity gate)       PEN_a(PEN1), PEN_b(PEN2)
  PFN       (forward-velocity gate)        PFNd, PFNv
  vector    (displacement readout)         PFNa, hDeltaB, PFR_a, PFR_b

so the same block appears at the same matrix coordinates across (a) and (b),
and the partition colour code matches Figure 1. The cross-species mapping is
deliberate: angular afferent = blue, translation afferent = orange,
recurrent ring = amber, readout = green, as in zebrafish.tex Figure 1/2.

  (a) Support mask sorted by partition, with colour-coded row/column strips
      and translucent column bands carrying the partition identity.
  (b) Signed W_con, z-scored over non-zero entries and clipped to +/-3
      (red excitatory, blue inhibitory), same partition ordering.

Data source: the W_con matrix is the effective connectome the trained
Known-ODE / GNN models see, loaded via the circuit registry
(connectome_gnn.generators.circuits.get_circuit) from the hemibrain export
fetched into figures/drosophila_cx/drosophila_cx_connectome_338/ by
fetch_cx_connectivity_pfn.py, with the Hulse-2025 sign convention
(Delta7, ER6 inhibitory) and spectral normalisation to radius 0.9.

Output: figures/drosophila_cx/fig_connectome_summary.png
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

from connectome_gnn.utils import load_data_root_from_json, set_data_root


# Four-way functional partition of the CX path-integration circuit (Fig 1c/d).
# Ordered recurrent-ring -> afferent gates -> readout, mirroring the
# recurrent-first layout of zebrafish.tex Figure 2.
PARTITION_ORDER = ["HD ring", "PEN", "PFN", "vector"]
PARTITION_COLOR = {
    "HD ring": "#d49a3a",   # amber  -- recurrent HD substrate (cf. dIPN ring)
    "PEN":     "#1f6fb3",   # blue   -- angular-velocity gate  (cf. ARTR)
    "PFN":     "#e07b1a",   # orange -- forward-velocity gate  (cf. pt-IPN1)
    "vector":  "#2a9d3d",   # green  -- displacement readout   (cf. decoder)
}

_HD_RING_TYPES = {"EPG", "EPGt", "PEG", "Delta7", "ER6"}
_PEN_TYPES     = {"PEN_a(PEN1)", "PEN_b(PEN2)"}
_PFN_GATE_TYPES = {"PFNd", "PFNv"}
_VECTOR_TYPES  = {"PFNa", "hDeltaB", "PFR_a", "PFR_b"}


def _partition_of(name):
    if name in _PEN_TYPES:
        return "PEN"
    if name in _PFN_GATE_TYPES:
        return "PFN"
    if name in _VECTOR_TYPES:
        return "vector"
    if name in _HD_RING_TYPES:
        return "HD ring"
    return "HD ring"  # any unlisted heading-core type stays in the ring block


def _partition_ids(nt, names):
    return np.array([_partition_of(names[int(t)]) for t in nt], dtype=object)


def _load_W(circuit_name):
    from connectome_gnn.generators.circuits import get_circuit
    cxo = get_circuit(circuit_name)
    W = np.asarray(cxo.J_effective, dtype=np.float32)
    nt = np.asarray(cxo.neuron_types).astype(int)
    names = list(cxo.type_names)
    return W, nt, names


def _set_partition_tick_labels(ax, part_sorted, *, fontsize=10, rotation=30):
    """Tick labels = the partition group names at each block centre."""
    order_key = {k: i for i, k in enumerate(PARTITION_ORDER)}
    part_ids = np.array([order_key.get(p, len(PARTITION_ORDER))
                         for p in part_sorted])
    part_changes = np.where(np.diff(part_ids) != 0)[0] + 0.5
    bnds = np.concatenate([[0], part_changes + 0.5, [part_ids.size]])
    centres = (bnds[:-1] + bnds[1:]) / 2 - 0.5
    lab = [part_sorted[int(round(c))] for c in centres]
    ax.set_xticks(centres)
    ax.set_xticklabels(lab, fontsize=fontsize, rotation=rotation, ha="right")
    ax.set_yticks(centres)
    ax.set_yticklabels(lab, fontsize=fontsize)


def _add_partition_strips(ax, part_sorted, *, alpha_overlay=0.0,
                          boundary_color="r", boundary_lw=0.5, strips=True):
    """Partition colour strips along the top/left of a matrix axes plus
    block-boundary lines and optional translucent column bands. Returns the
    strip width in data units (0 if ``strips=False``)."""
    N = part_sorted.size
    order_key = {k: i for i, k in enumerate(PARTITION_ORDER)}
    bnd = np.where(np.diff(
        np.array([order_key.get(p, len(PARTITION_ORDER)) for p in part_sorted])
    ) != 0)[0] + 0.5
    for x in bnd:
        ax.axvline(x, color=boundary_color, lw=boundary_lw, alpha=0.7, zorder=3)
        ax.axhline(x, color=boundary_color, lw=boundary_lw, alpha=0.7, zorder=3)
    boundaries = np.concatenate([[0], bnd + 0.5, [N]])
    strip = max(6, int(N * 0.02)) if strips else 0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        lo = int(round(lo)); hi = int(round(hi))
        cat = part_sorted[lo]
        col = PARTITION_COLOR.get(cat, "0.5")
        if strips:
            ax.add_patch(Rectangle(
                (lo - 0.5, -strip - 0.5), hi - lo, strip,
                facecolor=col, edgecolor="none", clip_on=False, zorder=4))
            ax.add_patch(Rectangle(
                (-strip - 0.5, lo - 0.5), strip, hi - lo,
                facecolor=col, edgecolor="none", clip_on=False, zorder=4))
        if alpha_overlay > 0:
            ax.add_patch(Rectangle(
                (lo - 0.5, -0.5), hi - lo, N,
                facecolor=col, edgecolor="none", alpha=alpha_overlay, zorder=2))
    if strips:
        ax.set_xlim(-strip - 0.5, N - 0.5)
        ax.set_ylim(N - 0.5, -strip - 0.5)
    return strip


def _partition_perm(partition):
    order_key = {k: i for i, k in enumerate(PARTITION_ORDER)}
    return np.argsort([order_key.get(p, len(PARTITION_ORDER)) for p in partition],
                      kind="stable")


def _panel_mask(ax, W, partition, dilate_iter=1):
    """Support mask sorted by the functional partition, with colour strips."""
    from scipy.ndimage import binary_dilation
    perm = _partition_perm(partition)
    W_sorted = W[np.ix_(perm, perm)]
    part_sorted = partition[perm]
    M = (W_sorted != 0)
    Mvis = binary_dilation(M, iterations=dilate_iter).astype(np.float32)
    ax.imshow(Mvis, cmap="binary", vmin=0, vmax=1,
              interpolation="nearest", aspect="equal")
    _add_partition_strips(ax, part_sorted, alpha_overlay=0.16,
                          boundary_color="r", boundary_lw=0.5)
    _set_partition_tick_labels(ax, part_sorted)
    ax.set_xlabel("presynaptic", fontsize=13)
    ax.set_ylabel("postsynaptic", fontsize=13)


def _panel_W(ax, W, partition, dilate_iter=1, show_cbar=True):
    """Signed W_con z-scored +/-3, sorted by the functional partition."""
    from scipy.ndimage import maximum_filter, minimum_filter
    perm = _partition_perm(partition)
    W_sorted = W[np.ix_(perm, perm)]
    part_sorted = partition[perm]
    nz = W_sorted[W_sorted != 0]
    mu, sd = float(nz.mean()), float(nz.std())
    Z = np.where(W_sorted != 0, (W_sorted - mu) / max(sd, 1e-8), 0.0).clip(-3.0, 3.0)
    size = 2 * int(dilate_iter) + 1
    Zpos = maximum_filter(np.where(Z > 0, Z, 0.0), size=size)
    Zneg = minimum_filter(np.where(Z < 0, Z, 0.0), size=size)
    Zvis = np.where(np.abs(Zpos) >= np.abs(Zneg), Zpos, Zneg)
    im = ax.imshow(Zvis, cmap="RdBu_r", vmin=-3.0, vmax=3.0,
                   interpolation="nearest", aspect="equal")
    _add_partition_strips(ax, part_sorted, alpha_overlay=0.0,
                          boundary_color="k", boundary_lw=0.3, strips=False)
    _set_partition_tick_labels(ax, part_sorted)
    ax.set_xlabel("presynaptic", fontsize=13)
    ax.set_ylabel("postsynaptic", fontsize=13)
    if show_cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
        cb.ax.tick_params(labelsize=11)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--circuit", default="drosophila_cx_338_v1",
                   help="registry circuit name")
    p.add_argument("--output_root", default=None)
    p.add_argument("--out", default=os.path.join(here, "fig_connectome_summary.png"))
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass

    W, nt, names = _load_W(args.circuit)
    partition = _partition_ids(nt, names)
    nnz = int((W != 0).sum())
    print(f"{args.circuit}: N={W.shape[0]} nnz={nnz} dens={nnz / W.size:.4f} "
          f"E={int((W > 0).sum())} I={int((W < 0).sum())}")
    for k in PARTITION_ORDER:
        print(f"  {k:10s}  {int((partition == k).sum())}")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6),
                             gridspec_kw={"wspace": 0.34})
    _panel_mask(axes[0], W, partition)
    _panel_W(axes[1], W, partition, show_cbar=True)

    # Partition legend below the support-mask panel.
    handles = [Patch(facecolor=PARTITION_COLOR[k], edgecolor="none", label=k)
               for k in PARTITION_ORDER]
    axes[0].legend(handles=handles, ncol=4, fontsize=10, frameon=False,
                   loc="upper center", bbox_to_anchor=(0.5, -0.16))

    for k, ax in enumerate(axes.flat):
        ax.text(-0.02, 1.06, "ab"[k], transform=ax.transAxes,
                ha="right", va="bottom", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
