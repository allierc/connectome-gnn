"""W_rec comparison across 10 Known-ODE CV models + W_con.

Three-panel figure for the "Known-ODE RNN: many circuits, same activity"
section of drosophila.tex:

  (a) 11x11 pairwise cosine similarity between {W_con, W_1, ..., W_10}
      over the connectome support, ordered W_con first.

  (b) Per-edge coefficient of variation across the 10 learned models,
      |W| std / |W| mean, grouped by (post cell type, pre cell type).
      Drawn as a 7x7 grid of violin distributions on the same cell-type
      ordering as Fig 2c/d.

  (c) Per-block magnitude comparison: each (post_t, pre_t) cell shows
      the mean of |W_con| for that block and the mean of |W^| across
      the 10 models, with the SD across models as an error bar.

CV checkpoints expected at
  $GNN_OUTPUT_ROOT/log/drosophila_cx/drosophila_cx_pi_epg_tv_cv{0..9}/models/
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from connectome_gnn.utils import (
    log_path, load_data_root_from_json, set_data_root,
)
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.models.registry import create_model


HD_TYPE_ORDER = ["EPGt", "EPG", "PEG", "Delta7",
                  "PEN_b(PEN2)", "PEN_a(PEN1)", "ER6"]
HD_TYPE_SHORT = {"EPGt": "EPGt", "EPG": "EPG", "PEG": "PEG",
                  "Delta7": r"$\Delta 7$", "PEN_b(PEN2)": r"PEN$_b$",
                  "PEN_a(PEN1)": r"PEN$_a$", "ER6": "ER6"}


def _load_net(config_name, device):
    config, _ = load_run_config(config_name, explicit_output_root=False,
                                 task="train")
    net = create_model(
        config.graph_model.signal_model_name,
        aggr_type=config.graph_model.aggr_type,
        config=config, device=device,
    )
    return net, config


def _load_cv_W_rec(base, n_folds, device):
    """Return a stacked (n_folds, N, N) tensor of trained W_rec matrices,
    plus type metadata from the first fold."""
    net, _ = _load_net(f"{base}_cv0", device)
    nt = np.asarray(net.neuron_types).astype(int)
    names = list(net.type_names)
    W_con = net.W_con.detach().cpu().numpy().astype(np.float32)
    ws = np.empty((n_folds, W_con.shape[0], W_con.shape[1]),
                  dtype=np.float32)
    for k in range(n_folds):
        cfg = f"{base}_cv{k}"
        net_k, config_k = _load_net(cfg, device)
        ckpt_dir = os.path.join(log_path(config_k.config_file), "models")
        cands = sorted(
            glob.glob(os.path.join(ckpt_dir, "best_model_with_0_graphs_*.pt")),
            key=lambda p_: int(p_.rsplit("_", 1)[1].rstrip(".pt")),
        )
        if not cands:
            raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
        sd = torch.load(cands[-1], map_location=device,
                        weights_only=False)["model_state_dict"]
        net_k.load_state_dict(sd)
        ws[k] = net_k.W_rec.detach().cpu().numpy().astype(np.float32)
        print(f"  loaded {cfg}: {os.path.basename(cands[-1])}")
    return W_con, ws, nt, names


def _cosine_similarity(A, B):
    a = A.ravel(); b = B.ravel()
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _panel_affinity(ax, W_con, Ws):
    """(a) 11x11 cosine similarity over the connectome support."""
    mask = (W_con != 0)
    mats = [W_con * mask] + [Ws[k] * mask for k in range(Ws.shape[0])]
    n = len(mats)
    S = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            s = _cosine_similarity(mats[i], mats[j])
            S[i, j] = s; S[j, i] = s
    labels = [r"$W^{\rm con}$"] + [rf"$\hat W_{{{k+1}}}$" for k in range(Ws.shape[0])]
    im = ax.imshow(S, vmin=0.0, vmax=1.0, cmap="viridis",
                   interpolation="nearest", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = S[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6,
                    color="white" if v < 0.6 else "black")
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(
        "(a) pairwise cosine similarity over the connectome support",
        fontsize=10,
    )

    # Also report the bottom-right 10x10 block stats for caption.
    sub = S[1:, 1:]
    iu = np.triu_indices_from(sub, k=1)
    inter_model = sub[iu]
    vs_con = S[0, 1:]
    return {
        "inter_model_mean": float(inter_model.mean()),
        "inter_model_std": float(inter_model.std()),
        "vs_con_mean": float(vs_con.mean()),
        "vs_con_std": float(vs_con.std()),
    }


def _block_indices(nt, names):
    """Return list of (type_id, row_indices)] in HD_TYPE_ORDER."""
    name_to_id = {n: i for i, n in enumerate(names)}
    out = []
    for nm in HD_TYPE_ORDER:
        tid = name_to_id.get(nm)
        if tid is None:
            continue
        idx = np.where(nt == tid)[0]
        out.append((nm, tid, idx))
    return out


def _panel_cv_grid(ax, Ws, W_con, nt, names):
    """(b) Per-block median CV across 10 learned models. One scalar per
    (post type, pre type) cell, drawn as a 7x7 heatmap. Empty blocks
    (no connectome edges) shown in grey."""
    blocks = _block_indices(nt, names)
    n_types = len(blocks)

    abs_ws = np.abs(Ws)
    mean_ws = abs_ws.mean(axis=0)
    std_ws = abs_ws.std(axis=0)
    cv = np.zeros_like(mean_ws)
    nz = mean_ws > 1e-8
    cv[nz] = std_ws[nz] / mean_ws[nz]

    grid = np.full((n_types, n_types), np.nan, dtype=np.float32)
    counts = np.zeros((n_types, n_types), dtype=np.int32)
    for r, (_, _, idx_post) in enumerate(blocks):
        for c, (_, _, idx_pre) in enumerate(blocks):
            sub_mask = (W_con[np.ix_(idx_post, idx_pre)] != 0)
            n_e = int(sub_mask.sum())
            counts[r, c] = n_e
            if n_e == 0:
                continue
            grid[r, c] = float(np.median(cv[np.ix_(idx_post, idx_pre)][sub_mask]))

    # Mask empty cells visually.
    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.92")
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.5,
                   interpolation="nearest", aspect="equal")
    for r in range(n_types):
        for c in range(n_types):
            if np.isnan(grid[r, c]):
                continue
            val = grid[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if val < 0.8 else "black")
    labels = [HD_TYPE_SHORT[nm] for nm, _, _ in blocks]
    ax.set_xticks(range(n_types))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("presynaptic", fontsize=9, labelpad=2)
    ax.set_ylabel("postsynaptic", fontsize=9, labelpad=2)
    ax.set_title(
        "(b) per-block median CV across 10 learned models",
        fontsize=10,
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=7)
    cb.set_label(r"median $\sigma_k|\hat W|/\langle|\hat W|\rangle_k$",
                 fontsize=8)


def _panel_block_means(ax, W_con, Ws, nt, names):
    """(c) Per-block log-ratio of learned-mean magnitude vs W_con. Red =
    learned solution amplifies the block, blue = attenuates. Annotated
    with the mean log-ratio and the SD across models."""
    blocks = _block_indices(nt, names)
    n_types = len(blocks)

    K = Ws.shape[0]
    log_ratio = np.full((n_types, n_types), np.nan, dtype=np.float32)
    log_ratio_sd = np.full((n_types, n_types), np.nan, dtype=np.float32)
    for r, (_, _, idx_post) in enumerate(blocks):
        for c, (_, _, idx_pre) in enumerate(blocks):
            mask = (W_con[np.ix_(idx_post, idx_pre)] != 0)
            if not mask.any():
                continue
            con_vals = np.abs(W_con[np.ix_(idx_post, idx_pre)])[mask]
            con_mean = float(con_vals.mean())
            if con_mean <= 0:
                continue
            per_k = np.empty(K)
            for k in range(K):
                v = np.abs(Ws[k][np.ix_(idx_post, idx_pre)])[mask]
                per_k[k] = float(v.mean())
            ratios = per_k / con_mean
            log_ratio[r, c] = float(np.log2(ratios).mean())
            log_ratio_sd[r, c] = float(np.log2(ratios).std())

    masked = np.ma.masked_invalid(log_ratio)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.92")
    vmax = float(np.nanmax(np.abs(log_ratio)))
    vmax = max(vmax, 0.5)
    im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation="nearest", aspect="equal")
    for r in range(n_types):
        for c in range(n_types):
            if np.isnan(log_ratio[r, c]):
                continue
            v = log_ratio[r, c]
            sd = log_ratio_sd[r, c]
            ax.text(c, r, f"{v:+.2f}\n±{sd:.2f}",
                    ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 * vmax else "black")
    labels = [HD_TYPE_SHORT[nm] for nm, _, _ in blocks]
    ax.set_xticks(range(n_types))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("presynaptic", fontsize=9, labelpad=2)
    ax.set_ylabel("postsynaptic", fontsize=9, labelpad=2)
    ax.set_title(
        r"(c) per-block $\log_2 \langle|\hat W|\rangle / \langle|W^{\rm con}|\rangle$",
        fontsize=10,
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=7)
    cb.set_label(r"$\log_2$ ratio", fontsize=8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="drosophila_cx_pi_epg_tv",
                   help="config base name; the 10 folds are "
                        "<base>_cv0..<base>_cv9.")
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=None)
    p.add_argument("--out_dir",
                   default=os.path.dirname(os.path.abspath(__file__)))
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass

    device = torch.device(args.device)
    print(f"loading {args.n_folds} CV checkpoints ...")
    W_con, Ws, nt, names = _load_cv_W_rec(args.base, args.n_folds, device)
    print(f"W_con shape {W_con.shape}, Ws shape {Ws.shape}")

    fig = plt.figure(figsize=(16.5, 6.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.20,
                          left=0.05, right=0.98, top=0.92, bottom=0.10)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    stats = _panel_affinity(ax_a, W_con, Ws)
    _panel_cv_grid(ax_b, Ws, W_con, nt, names)
    _panel_block_means(ax_c, W_con, Ws, nt, names)

    out_png = os.path.join(args.out_dir, "fig_w_rec_comparison.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(
        "panel (a) stats: "
        f"inter-model cos sim = {stats['inter_model_mean']:.3f} "
        f"± {stats['inter_model_std']:.3f}; "
        f"vs W_con = {stats['vs_con_mean']:.3f} "
        f"± {stats['vs_con_std']:.3f}"
    )


if __name__ == "__main__":
    main()
