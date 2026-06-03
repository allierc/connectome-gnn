"""Zebrafish HD-circuit connectome summary for the nominal 839-cell
IPN12 circuit, in its two Dale-sign variants (companion to
figures/drosophila_cx/fig_connectome_summary.py).

The nominal circuit now includes the IPN12_a / IPN12_b sub-types in the
bump pool. We do not know whether IPN12 cells are inhibitory or
excitatory, so two circuits are built from the same topology:
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
  (d) Per-cell-type INCOMING edge-weight distributions (violins,
      v1 vs v2), drosophila-2c style on coarse categories.
  (e) Per-cell-type OUTGOING edge-weight distributions (violins,
      v1 vs v2), drosophila-2d style.
  (f) IPN12_a + IPN12_b outgoing weights only: the load-bearing sign
      flip between v1 (inhibitory) and v2 (excitatory).

Data source: each circuit's effective connectome is read straight off
the trained-model template via create_model(...).W_con, so the figure
uses the same Dale-flip + spectral-rescaling the registry applies for
training (no re-derivation here).

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


def _panel_ipn12(ax, W1, W2, coarse):
    idx = np.where(coarse == "IPN12")[0]
    o1 = W1[:, idx].ravel(); o1 = o1[o1 != 0]
    o2 = W2[:, idx].ravel(); o2 = o2[o2 != 0]
    lo = float(min(o1.min(), o2.min())); hi = float(max(o1.max(), o2.max()))
    bins = np.linspace(lo, hi, 41)
    ax.hist(o1, bins=bins, color=V1_COLOR, alpha=0.6,
            label=f"v1 inhib. (mean {o1.mean():+.3f})")
    ax.hist(o2, bins=bins, color=V2_COLOR, alpha=0.6,
            label=f"v2 excit. (mean {o2.mean():+.3f})")
    ax.axvline(0.0, color="0.4", lw=0.6)
    ax.set_xlabel(r"signed $W^{\mathrm{con}}_{ij}$ (IPN12 outgoing)", fontsize=9)
    ax.set_ylabel("edge count", fontsize=9)
    ax.set_title(f"IPN12 outgoing sign flip ({o1.size} edges)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)


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

    for tag, W in (("v1", W1), ("v2", W2)):
        nnz = int((W != 0).sum())
        print(f"{tag}: N={W.shape[0]} nnz={nnz} dens={nnz / W.size:.4f} "
              f"E={int((W > 0).sum())} I={int((W < 0).sum())}")

    # Shared z-score reference so v1 and v2 matrices are directly comparable.
    ref_nz = np.concatenate([W1[W1 != 0], W2[W2 != 0]])

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    _panel_mask(axes[0, 0], W1, nt, names)
    _panel_W(axes[0, 1], W1, nt, names, ref_nz=ref_nz,
             title=r"$W^{\mathrm{con}}$ v1 — IPN12 inhibitory")
    _panel_W(axes[0, 2], W2, nt, names, ref_nz=ref_nz,
             title=r"$W^{\mathrm{con}}$ v2 — IPN12 excitatory")
    _paired_violin(axes[1, 0],
                   _collect(W1, coarse, "incoming"),
                   _collect(W2, coarse, "incoming"),
                   ylabel=r"signed $W^{\mathrm{con}}_{ij}$ (incoming)",
                   title="incoming edge weights (post = category)")
    _paired_violin(axes[1, 1],
                   _collect(W1, coarse, "outgoing"),
                   _collect(W2, coarse, "outgoing"),
                   ylabel=r"signed $W^{\mathrm{con}}_{ij}$ (outgoing)",
                   title="outgoing edge weights (pre = category)")
    _panel_ipn12(axes[1, 2], W1, W2, coarse)

    for k, ax in enumerate(axes.flat):
        ax.text(-0.12, 1.05, "abcdef"[k], transform=ax.transAxes,
                ha="left", va="top", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
