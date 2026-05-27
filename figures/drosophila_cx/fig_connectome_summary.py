"""2x2 connectome summary figure for the Drosophila CX.

(a) W_con: signed, z-scored over non-zero entries and clipped to +/-3
    (same convention as Fig 1a / fig_evolution).
(b) Binary mask M = (W_con != 0): black = edge present, white = absent.
(c) Per-cell-type INCOMING edge weight distributions (violins). For each
    cell type t, the data are the non-zero entries W_con[i, j] with the
    postsynaptic neuron i of type t.
(d) Per-cell-type OUTGOING edge weight distributions (violins). For each
    cell type t, the data are the non-zero entries W_con[i, j] with the
    presynaptic neuron j of type t.

Violins use the same cell-type ordering and the same per-type-id colour
palette as fig_hd_mi_summary so the two figures read as a matched pair.

Data source: the W_con matrix is the same effective connectome the trained
Known-ODE / GNN models see, loaded via
connectome_gnn.generators.connconstr_data.load_drosophila_cx_connectome
from the hemibrain export
(papers/Code_NN/Code_NN/Data/Figure5/exported-traced-adjacencies-v1.2/),
combining traced-neurons.csv and traced-total-connections.csv, with the
Hulse-2025 sign convention (Delta7, ER6 inhibitory).

Output: figures/drosophila/fig_connectome_summary.png
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from connectome_gnn.utils import load_data_root_from_json, set_data_root
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.models.registry import create_model


# Same cell-type ordering as fig_hd_mi_summary.py.
HD_TYPE_ORDER = ["EPGt", "EPG", "PEG", "Delta7", "PEN_b(PEN2)",
                 "PEN_a(PEN1)", "ER6"]


def _load_W_con(config_name, device):
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


def _panel_W(ax, W, nt, names):
    nz = W[W != 0]
    mu, sd = float(nz.mean()), float(nz.std())
    Z = np.where(W != 0, (W - mu) / max(sd, 1e-8), 0.0).clip(-3.0, 3.0)
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-3.0, vmax=3.0,
                   interpolation="nearest", aspect="equal")
    order = np.argsort(nt, kind="stable")
    b = np.where(np.diff(nt[order]) != 0)[0] + 0.5
    for x in b:
        ax.axvline(x, color="k", lw=0.3, alpha=0.5)
        ax.axhline(x, color="k", lw=0.3, alpha=0.5)
    boundaries = np.concatenate([[0], b + 0.5, [nt.size]])
    centres = (boundaries[:-1] + boundaries[1:]) / 2 - 0.5
    lab = [names[int(nt[order[int(c)]])] for c in centres]
    ax.set_xticks(centres); ax.set_xticklabels(lab, fontsize=7,
                                                rotation=45, ha="right")
    ax.set_yticks(centres); ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("presynaptic", fontsize=9)
    ax.set_ylabel("postsynaptic", fontsize=9)
    ax.set_title(r"$W^{\mathrm{con}}$ (signed, $z$-scored, $\pm 3$ clip)",
                 fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=7)


def _panel_mask(ax, W, nt, names):
    M = (W != 0).astype(np.float32)
    ax.imshow(M, cmap="binary", vmin=0, vmax=1,
              interpolation="nearest", aspect="equal")
    order = np.argsort(nt, kind="stable")
    b = np.where(np.diff(nt[order]) != 0)[0] + 0.5
    for x in b:
        ax.axvline(x, color="r", lw=0.3, alpha=0.6)
        ax.axhline(x, color="r", lw=0.3, alpha=0.6)
    boundaries = np.concatenate([[0], b + 0.5, [nt.size]])
    centres = (boundaries[:-1] + boundaries[1:]) / 2 - 0.5
    lab = [names[int(nt[order[int(c)]])] for c in centres]
    ax.set_xticks(centres); ax.set_xticklabels(lab, fontsize=7,
                                                rotation=45, ha="right")
    ax.set_yticks(centres); ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("presynaptic", fontsize=9)
    ax.set_ylabel("postsynaptic", fontsize=9)
    density = float(M.sum()) / float(M.size)
    ax.set_title(
        rf"$\mathrm{{supp}}(W^{{\mathrm{{con}}}})$  "
        rf"(density $={density:.3f}$)",
        fontsize=10,
    )


def _panel_violin(ax, distrs, type_ids, names, ylabel, title):
    """Violins, one per cell type, in HD_TYPE_ORDER. distrs[t] is a 1D array."""
    palette = plt.get_cmap("tab10").colors
    cols = [palette[t % len(palette)] for t in type_ids]
    xs = np.arange(len(type_ids))
    data = [distrs[t] if distrs[t].size else np.array([0.0])
            for t in type_ids]
    parts = ax.violinplot(data, positions=xs, showmeans=True, showmedians=False,
                          showextrema=False, widths=0.85)
    for k, body in enumerate(parts["bodies"]):
        body.set_facecolor(cols[k])
        body.set_edgecolor("0.3")
        body.set_linewidth(0.5)
        body.set_alpha(0.75)
    if "cmeans" in parts:
        parts["cmeans"].set_color("0.15")
        parts["cmeans"].set_linewidth(1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([names[t] for t in type_ids],
                       rotation=30, ha="right", fontsize=8)
    ax.axhline(0.0, color="0.6", lw=0.4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="drosophila_cx_pi")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=None)
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass

    device = torch.device(args.device)
    W, nt, names = _load_W_con(args.config, device)
    print(f"W_con shape: {W.shape}   non-zero edges: {int((W != 0).sum())} / "
          f"{W.size}   density: {float((W != 0).sum()) / W.size:.3f}")

    # Per-cell-type incoming and outgoing weight distributions (non-zero only).
    name_to_id = {n: i for i, n in enumerate(names)}
    type_ids = [name_to_id[n] for n in HD_TYPE_ORDER if n in name_to_id]

    incoming, outgoing = {}, {}
    for t in type_ids:
        rows = np.where(nt == t)[0]
        cols = np.where(nt == t)[0]
        W_in  = W[rows, :].ravel()
        W_out = W[:, cols].ravel()
        incoming[t] = W_in[W_in != 0]
        outgoing[t] = W_out[W_out != 0]
        print(f"  {names[t]:18s} incoming n={incoming[t].size:5d} "
              f"|mean|={np.abs(incoming[t]).mean():.3f}  "
              f"outgoing n={outgoing[t].size:5d} "
              f"|mean|={np.abs(outgoing[t]).mean():.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    _panel_W(axes[0, 0], W, nt, names)
    _panel_mask(axes[0, 1], W, nt, names)
    _panel_violin(axes[1, 0], incoming, type_ids, names,
                  ylabel=r"signed $W^{\mathrm{con}}_{ij}$ (incoming)",
                  title="incoming edge weights per cell type (post = type)")
    _panel_violin(axes[1, 1], outgoing, type_ids, names,
                  ylabel=r"signed $W^{\mathrm{con}}_{ij}$ (outgoing)",
                  title="outgoing edge weights per cell type (pre = type)")

    for k, ax in enumerate(axes.flat):
        ax.text(-0.13, 1.04, "abcd"[k], transform=ax.transAxes,
                ha="left", va="top", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fig_connectome_summary.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
