"""Zebrafish HD-circuit connectome summary for the nominal 839-cell
IPN12 circuit, in its two Dale-sign variants.

The nominal circuit includes the IPN12_a / IPN12_b sub-types in the
recurrent pool. Two circuits are built from the same topology:
  v1  zebrafish_HD_IPN12_839_v1 : IPN12 outgoing weights Dale-flipped
                                  to inhibitory (GABAergic hypothesis)
  v2  zebrafish_HD_IPN12_839_v2 : IPN12 outgoing weights left positive
                                  (glutamatergic / excitatory hypothesis)
Only the IPN12 outgoing sign differs; the support is identical.

2 x 3 panels:
  (a) Binary support mask supp(W_con) (shared by v1 and v2).
  (b) Signed W_con, v1 (IPN12 inhibitory), z-scored over non-zero
      entries and clipped to +/-3.
  (c) Signed W_con, v2 (IPN12 excitatory), same scale.
  (d) Support mask with neurons sorted by the six-way functional
      partition introduced in the proprioception circuit (ARTR,
      pt-IPN1, motor_efferent, dIPN, IPN12 pool, other), with
      colour-coded axis strips marking each block. Colour code is
      shared with Figure 1 of zebrafish.tex so the two figures read
      together.
  (e) Per-cell-type INCOMING edge-weight distributions (violins,
      v1 vs v2) on the coarse type categories.
  (f) Per-cell-type OUTGOING edge-weight distributions (violins,
      v1 vs v2).

Output: figures/zebrafish/fig_connectome_summary_HD_IPN12.png
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

from connectome_gnn.utils import load_data_root_from_json, set_data_root
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.models.registry import create_model


# Coarse cell-type categories for the violin panels (the 33 fine fish2
# subtypes are too many to read as violins). IPN12 is the category that
# flips sign between v1 and v2.
COARSE_ORDER = ["IPNd", "IPNds", "IPN12", "RIPN", "pt-IPN"]
COARSE_COLOR = {
    "IPNd": "#cf222e",    # red   (dIPN HD ring)
    "IPNds": "#d29922",   # orange (dorsal-subset)
    "IPN12": "#2ea043",   # green  (the unknown-sign sub-types)
    "RIPN": "#1f6feb",    # blue   (habenular afferents)
    "pt-IPN": "#8957e5",  # purple (pretectal afferents)
}
V1_COLOR = "#1f6feb"  # blue  -> v1 (IPN12 inhibitory)
V2_COLOR = "#cf222e"  # red   -> v2 (IPN12 excitatory)

# Six-way functional partition introduced in the proprioception circuit
# (Figure 3 of zebrafish.tex). Same colour code as Figure 1 of the same
# document so the partition reads consistently across figures.
PARTITION_ORDER = [
    "dIPN", "IPN12 pool", "ARTR", "pt-IPN1", "motor_efferent", "other",
]
PARTITION_COLOR = {
    "dIPN":      "#d49a3a",
    "IPN12 pool":     "#b15a8e",
    "ARTR":           "#1f6fb3",
    "pt-IPN1":        "#e07b1a",
    "motor_efferent": "#2a9d3d",
    "other":          "#888888",
}
_ARTR_TYPES = {"RIPN01", "RIPN02", "RIPN03_a", "RIPN03_b"}
_MOTOR_EFFERENT_TYPES = {"RIPN11", "RIPN12_a", "RIPN12_c"}


def _coarse_of(name):
    if name.startswith("IPN12"):
        return "IPN12"
    if name.startswith("IPNds"):
        return "IPNds"
    if name.startswith("IPNd"):
        return "IPNd"
    if name.startswith("RIPN"):
        return "RIPN"
    if name.startswith("pt-IPN") or name.startswith("ptIPN"):
        return "pt-IPN"
    return "other"


def _partition_of(name):
    """Six-way functional partition matching the proprioception circuit."""
    if name in _ARTR_TYPES:
        return "ARTR"
    if name in _MOTOR_EFFERENT_TYPES:
        return "motor_efferent"
    if name == "pt-IPN1":
        return "pt-IPN1"
    if name.startswith("IPN12"):
        return "IPN12 pool"
    if name.startswith("IPNds") or name.startswith("IPNd"):
        return "dIPN"
    return "other"


def _partition_ids(nt, names):
    """Per-neuron partition label for each neuron index."""
    return np.array([_partition_of(names[int(t)]) for t in nt], dtype=object)


def _load_W(config_name, device):
    config, _ = load_run_config(config_name, explicit_output_root=False, task="train")
    net = create_model(
        config.graph_model.signal_model_name,
        aggr_type=config.graph_model.aggr_type,
        config=config, device=device,
    )
    W = net.W_con.detach().cpu().numpy()
    nt = np.asarray(net.neuron_types).astype(int)
    names = list(net.type_names)
    return W, nt, names


def _coarse_ids(nt, names):
    """Per-neuron coarse-category label (string) for each neuron index."""
    return np.array([_coarse_of(names[int(t)]) for t in nt], dtype=object)


def _draw_block_grid(ax, type_ids, names, color="k", alpha=0.5, lw=0.3):
    order = np.argsort(type_ids, kind="stable")
    b = np.where(np.diff(type_ids[order]) != 0)[0] + 0.5
    for x in b:
        ax.axvline(x, color=color, lw=lw, alpha=alpha)
        ax.axhline(x, color=color, lw=lw, alpha=alpha)
    boundaries = np.concatenate([[0], b + 0.5, [type_ids.size]])
    centres = (boundaries[:-1] + boundaries[1:]) / 2 - 0.5
    lab = [names[int(type_ids[order[int(c)]])] for c in centres]
    ax.set_xticks(centres); ax.set_xticklabels(lab, fontsize=5,
                                                rotation=60, ha="right")
    ax.set_yticks(centres); ax.set_yticklabels(lab, fontsize=5)
    ax.set_xlabel("presynaptic", fontsize=9)
    ax.set_ylabel("postsynaptic", fontsize=9)


def _panel_W(ax, W, type_ids, names, ref_nz=None, title="", dilate_iter=1):
    from scipy.ndimage import maximum_filter, minimum_filter
    nz = W[W != 0] if ref_nz is None else ref_nz
    mu, sd = float(nz.mean()), float(nz.std())
    Z = np.where(W != 0, (W - mu) / max(sd, 1e-8), 0.0).clip(-3.0, 3.0)
    size = 2 * int(dilate_iter) + 1
    Zpos = maximum_filter(np.where(Z > 0, Z, 0.0), size=size)
    Zneg = minimum_filter(np.where(Z < 0, Z, 0.0), size=size)
    Zvis = np.where(np.abs(Zpos) >= np.abs(Zneg), Zpos, Zneg)
    im = ax.imshow(Zvis, cmap="RdBu_r", vmin=-3.0, vmax=3.0,
                   interpolation="nearest", aspect="equal")
    _draw_block_grid(ax, type_ids, names, color="k", alpha=0.5)
    if title:
        ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=7)


def _panel_mask(ax, W, type_ids, names, dilate_iter=1):
    from scipy.ndimage import binary_dilation
    M = (W != 0)
    Mvis = binary_dilation(M, iterations=dilate_iter).astype(np.float32)
    ax.imshow(Mvis, cmap="binary", vmin=0, vmax=1,
              interpolation="nearest", aspect="equal")
    _draw_block_grid(ax, type_ids, names, color="r", alpha=0.6)
    density = float(M.sum()) / float(M.size)
    ax.set_title(rf"$\mathrm{{supp}}(W^{{\mathrm{{con}}}})$ "
                 rf"(shared; density $={density:.3f}$)", fontsize=10)


def _collect(W, coarse, mode):
    """Return {category: 1D nonzero-weight array} for incoming (post=cat)
    or outgoing (pre=cat) edges."""
    out = {}
    for c in COARSE_ORDER:
        idx = np.where(coarse == c)[0]
        if idx.size == 0:
            out[c] = np.array([])
            continue
        block = W[idx, :] if mode == "incoming" else W[:, idx]
        v = block.ravel()
        out[c] = v[v != 0]
    return out


def _paired_violin(ax, d1, d2, ylabel, title):
    xs = np.arange(len(COARSE_ORDER))
    for sign, dd, col, dx in ((-1, d1, V1_COLOR, -0.2), (1, d2, V2_COLOR, 0.2)):
        data, pos = [], []
        for k, c in enumerate(COARSE_ORDER):
            arr = dd[c]
            if arr.size >= 2:
                data.append(arr); pos.append(xs[k] + dx)
        if not data:
            continue
        parts = ax.violinplot(data, positions=pos, widths=0.34,
                              showmeans=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(col); body.set_edgecolor("0.3")
            body.set_linewidth(0.5); body.set_alpha(0.78)
        if "cmeans" in parts:
            parts["cmeans"].set_color("0.15"); parts["cmeans"].set_linewidth(1.0)
    ax.axhline(0.0, color="0.6", lw=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(COARSE_ORDER, rotation=20, ha="right",
                                          fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(handles=[Patch(facecolor=V1_COLOR, alpha=0.78,
                             label="v1 (IPN12 inhib.)"),
                       Patch(facecolor=V2_COLOR, alpha=0.78,
                             label="v2 (IPN12 excit.)")],
              fontsize=7, loc="upper right", frameon=False)


def _panel_partition_matrix(ax, W, partition):
    """Support mask with neurons sorted by the functional partition.

    Neurons are reordered so members of the same partition are
    contiguous; the matrix becomes a block-structured view of the
    connectome where the ARTR / pt-IPN1 / motor_efferent /
    dIPN / IPN12 / other blocks are visible. Coloured strips on
    the row and column axes carry the partition identity in the same
    palette used by Figure 1 of zebrafish.tex.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    from scipy.ndimage import binary_dilation
    # Stable order: PARTITION_ORDER groups, then original index within.
    order_key = {k: i for i, k in enumerate(PARTITION_ORDER)}
    perm = np.argsort(
        [order_key.get(p, len(PARTITION_ORDER)) for p in partition],
        kind="stable",
    )
    W_sorted = W[np.ix_(perm, perm)]
    part_sorted = partition[perm]

    # Support mask, lightly dilated for visibility at panel resolution.
    M = (W_sorted != 0)
    Mvis = binary_dilation(M, iterations=1).astype(np.float32)
    ax.imshow(Mvis, cmap="binary", vmin=0, vmax=1,
              interpolation="nearest", aspect="equal")

    # Block boundaries between partitions.
    b = np.where(np.diff(
        np.array([order_key.get(p, len(PARTITION_ORDER)) for p in part_sorted])
    ) != 0)[0] + 0.5
    for x in b:
        ax.axvline(x, color="r", lw=0.5, alpha=0.7)
        ax.axhline(x, color="r", lw=0.5, alpha=0.7)
    boundaries = np.concatenate([[0], b + 0.5, [part_sorted.size]])

    # Strip width as a fraction of the axes — drawn as rectangles
    # outside the matrix area so the partition is visible without
    # cluttering the connectivity.
    N = part_sorted.size
    strip = max(8, int(N * 0.015))
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        lo = int(round(lo)); hi = int(round(hi))
        cat = part_sorted[lo]
        col = PARTITION_COLOR.get(cat, "0.5")
        # Top strip (along columns).
        ax.add_patch(Rectangle(
            (lo - 0.5, -strip - 0.5), hi - lo, strip,
            facecolor=col, edgecolor="none", clip_on=False, zorder=4,
        ))
        # Left strip (along rows).
        ax.add_patch(Rectangle(
            (-strip - 0.5, lo - 0.5), strip, hi - lo,
            facecolor=col, edgecolor="none", clip_on=False, zorder=4,
        ))
        # Transparent column overlay across the full matrix body so the
        # partition identity is readable inside the matrix, not just on
        # the axis strips. Low alpha so the connectivity stays visible.
        ax.add_patch(Rectangle(
            (lo - 0.5, -0.5), hi - lo, N,
            facecolor=col, edgecolor="none", alpha=0.18, zorder=2,
        ))

    ax.set_xlim(-strip - 0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -strip - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("presynaptic (sorted by partition)", fontsize=9)
    ax.set_ylabel("postsynaptic (sorted by partition)", fontsize=9)
    ax.set_title("functional partition (matches Figure 1 colour code)",
                 fontsize=10)

    # Compact legend below the panel — Figure 1's same colour swatches.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=PARTITION_COLOR[k], edgecolor="none",
                     label=f"{k} (n={int((part_sorted==k).sum())})")
               for k in PARTITION_ORDER if (part_sorted == k).any()]
    ax.legend(handles=handles, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=7,
              frameon=False, handlelength=1.4)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config_v1", default="zebrafish_hd_si_ipn12_v1")
    p.add_argument("--config_v2", default="zebrafish_hd_si_ipn12_v2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=None)
    p.add_argument("--out", default=os.path.join(
        here, "fig_connectome_summary_HD_IPN12.png"))
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass

    device = torch.device(args.device)
    W1, nt, names = _load_W(args.config_v1, device)
    W2, nt2, names2 = _load_W(args.config_v2, device)
    assert names == names2 and np.array_equal(nt, nt2), \
        "v1 and v2 must share neuron ordering / type vocabulary"
    coarse = _coarse_ids(nt, names)
    partition = _partition_ids(nt, names)

    for tag, W in (("v1", W1), ("v2", W2)):
        nnz = int((W != 0).sum())
        print(f"{tag}: N={W.shape[0]} nnz={nnz} dens={nnz / W.size:.4f} "
              f"E={int((W > 0).sum())} I={int((W < 0).sum())}")
    print("partition counts:")
    for k in PARTITION_ORDER:
        n_k = int((partition == k).sum())
        print(f"  {k:18s}  {n_k}")

    # Shared z-score reference so v1 and v2 matrices are directly comparable.
    ref_nz = np.concatenate([W1[W1 != 0], W2[W2 != 0]])

    # Layout: a = support, b = W_con v1, c = W_con v2,
    #         d = partition-sorted matrix (NEW), e = incoming violins
    #         (was d), f = outgoing violins (was e). The old IPN12
    #         sign-flip histogram is removed.
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    _panel_mask(axes[0, 0], W1, nt, names)
    _panel_W(axes[0, 1], W1, nt, names, ref_nz=ref_nz,
             title=r"$W^{\mathrm{con}}$ v1 — IPN12 inhibitory")
    _panel_W(axes[0, 2], W2, nt, names, ref_nz=ref_nz,
             title=r"$W^{\mathrm{con}}$ v2 — IPN12 excitatory")
    _panel_partition_matrix(axes[1, 0], W1, partition)
    _paired_violin(axes[1, 1],
                   _collect(W1, coarse, "incoming"),
                   _collect(W2, coarse, "incoming"),
                   ylabel=r"signed $W^{\mathrm{con}}_{ij}$ (incoming)",
                   title="incoming edge weights (post = category)")
    _paired_violin(axes[1, 2],
                   _collect(W1, coarse, "outgoing"),
                   _collect(W2, coarse, "outgoing"),
                   ylabel=r"signed $W^{\mathrm{con}}_{ij}$ (outgoing)",
                   title="outgoing edge weights (pre = category)")

    for k, ax in enumerate(axes.flat):
        ax.text(-0.12, 1.05, "abcdef"[k], transform=ax.transAxes,
                ha="left", va="top", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
