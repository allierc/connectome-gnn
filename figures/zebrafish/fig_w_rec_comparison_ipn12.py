"""W_rec comparison across 10 CV folds for the two IPN12 Dale-sign
variants (zebrafish companion to figures/drosophila_cx/fig_w_rec_comparison.py,
drosophila.tex Figure 9).

The stats runner (scripts/run_GNN_zebrafish_hd_si_ipn12_stats.py) re-trains
zebrafish_hd_si_ipn12_v1 and _v2 ten times each with fresh training seeds
(shared task data). This figure asks, over those 2 x 10 folds:

  (a) Combined cosine-similarity affinity over the connectome support of
      {W_con, v1 fold 1..10, v2 fold 1..10}. Reveals at once
        * within-v1 / within-v2 reproducibility (identifiability across seeds),
        * v1<->v2 cross-similarity (robustness to the unknown IPN12 sign),
        * each fold vs the raw connectome template.
  (b) Per-(coarse cell-type) block median coefficient of variation of
      |W_rec| across the 10 folds, one 5x5 grid per variant.
  (c) Per-block log2 ratio of the mean learned |W_rec| to the mean
      |W_con| (amplification), annotated mu +/- SD across folds, per variant.

Coarse cell-type categories (the 33 fine fish2 subtypes are too many to
read as a block grid): IPNd, IPNds, IPN12, RIPN, pt-IPN.

CV checkpoints expected at
  $GNN_OUTPUT_ROOT/log/zebrafish/zebrafish_hd_si_ipn12_v{1,2}_cv{0..9}/models/

Output: figures/zebrafish/fig_w_rec_comparison_ipn12.png
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

from connectome_gnn.utils import log_path, load_data_root_from_json, set_data_root
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.models.registry import create_model


COARSE_ORDER = ["IPNd", "IPNds", "IPN12", "RIPN", "pt-IPN"]
_LEGACY_GATE = {"v_pena_l": "v_ripn_l", "v_pena_r": "v_ripn_r",
                "v_penb_l": "v_ptipn_l", "v_penb_r": "v_ptipn_r"}


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


def _load_net(config_name, device):
    config, _ = load_run_config(config_name, explicit_output_root=False,
                                 task="train")
    net = create_model(config.graph_model.signal_model_name,
                       aggr_type=config.graph_model.aggr_type,
                       config=config, device=device)
    return net, config


def _load_cv_W_rec(base, n_folds, device):
    """Stacked (n_folds, N, N) trained W_rec + the variant's W_con and types."""
    net0, _ = _load_net(f"{base}_cv0", device)
    nt = np.asarray(net0.neuron_types).astype(int)
    names = list(net0.type_names)
    W_con = net0.W_con.detach().cpu().numpy().astype(np.float32)
    ws = np.empty((n_folds, W_con.shape[0], W_con.shape[1]), dtype=np.float32)
    for k in range(n_folds):
        net_k, cfg_k = _load_net(f"{base}_cv{k}", device)
        ckpt_dir = os.path.join(log_path(cfg_k.config_file), "models")
        cands = sorted(
            glob.glob(os.path.join(ckpt_dir, "best_model_with_0_graphs_*.pt")),
            key=lambda p_: int(p_.rsplit("_", 1)[1].rstrip(".pt")))
        if not cands:
            raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
        sd = torch.load(cands[-1], map_location=device,
                        weights_only=False)["model_state_dict"]
        mk = set(net_k.state_dict().keys())
        for o, nw in _LEGACY_GATE.items():
            if o in sd and nw in mk and nw not in sd:
                sd[nw] = sd.pop(o)
        net_k.load_state_dict(sd, strict=False)
        ws[k] = net_k.W_rec.detach().cpu().numpy().astype(np.float32)
        print(f"  loaded {base}_cv{k}: {os.path.basename(cands[-1])}")
    return W_con, ws, nt, names


def _cos(A, B):
    a, b = A.ravel(), B.ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def _coarse_blocks(nt, names):
    """[(coarse_label, neuron_index_array)] in COARSE_ORDER (non-empty only)."""
    coarse = np.array([_coarse_of(names[int(t)]) for t in nt], dtype=object)
    out = []
    for c in COARSE_ORDER:
        idx = np.where(coarse == c)[0]
        if idx.size:
            out.append((c, idx))
    return out


def _panel_affinity(ax, W_con, Ws_v1, Ws_v2):
    """Combined cosine affinity over the connectome support."""
    mask = (W_con != 0)
    mats = ([W_con * mask]
            + [Ws_v1[k] * mask for k in range(Ws_v1.shape[0])]
            + [Ws_v2[k] * mask for k in range(Ws_v2.shape[0])])
    n = len(mats)
    S = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            s = _cos(mats[i], mats[j]); S[i, j] = s; S[j, i] = s
    im = ax.imshow(S, vmin=0.0, vmax=1.0, cmap="viridis",
                   interpolation="nearest", aspect="equal")
    n1 = Ws_v1.shape[0]
    # block separators: after W_con (0.5) and after v1 (0.5 + n1)
    for b in (0.5, 0.5 + n1):
        ax.axhline(b, color="w", lw=1.2); ax.axvline(b, color="w", lw=1.2)
    centres = [0, 0.5 + n1 / 2.0, 0.5 + n1 + Ws_v2.shape[0] / 2.0]
    ax.set_xticks(centres)
    ax.set_xticklabels([r"$W^{\rm con}$", "v1 (×10)", "v2 (×10)"], fontsize=9)
    ax.set_yticks(centres)
    ax.set_yticklabels([r"$W^{\rm con}$", "v1 (×10)", "v2 (×10)"], fontsize=9,
                       rotation=90, va="center")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.9)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("cosine similarity", fontsize=9)
    # block-mean stats
    v1 = S[1:1 + n1, 1:1 + n1]; v2 = S[1 + n1:, 1 + n1:]
    cross = S[1:1 + n1, 1 + n1:]
    iu1 = np.triu_indices_from(v1, k=1); iu2 = np.triu_indices_from(v2, k=1)
    return {
        "within_v1": (float(v1[iu1].mean()), float(v1[iu1].std())),
        "within_v2": (float(v2[iu2].mean()), float(v2[iu2].std())),
        "cross": (float(cross.mean()), float(cross.std())),
        "vs_con_v1": (float(S[0, 1:1 + n1].mean()), float(S[0, 1:1 + n1].std())),
        "vs_con_v2": (float(S[0, 1 + n1:].mean()), float(S[0, 1 + n1:].std())),
    }


def _panel_cv(ax, Ws, W_con, blocks, title):
    aw = np.abs(Ws)
    mean_w, std_w = aw.mean(axis=0), aw.std(axis=0)
    cv = np.zeros_like(mean_w); nz = mean_w > 1e-8
    cv[nz] = std_w[nz] / mean_w[nz]
    n = len(blocks)
    grid = np.full((n, n), np.nan, dtype=np.float32)
    for r, (_, ip) in enumerate(blocks):
        for c, (_, jp) in enumerate(blocks):
            m = (W_con[np.ix_(ip, jp)] != 0)
            if m.any():
                grid[r, c] = float(np.median(cv[np.ix_(ip, jp)][m]))
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("0.92")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0.0, vmax=1.5,
                   interpolation="nearest", aspect="equal")
    for r in range(n):
        for c in range(n):
            if not np.isnan(grid[r, c]):
                ax.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if grid[r, c] < 0.8 else "black")
    labs = [b[0] for b in blocks]
    ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=45, ha="right",
                                                fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlabel("presynaptic", fontsize=9); ax.set_ylabel("postsynaptic",
                                                            fontsize=9)
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.9)
    cb.ax.tick_params(labelsize=7)
    cb.set_label(r"median $\sigma_k|\hat W|/\langle|\hat W|\rangle_k$", fontsize=8)


def _panel_logratio(ax, Ws, W_con, blocks, title):
    K = Ws.shape[0]; n = len(blocks)
    lr = np.full((n, n), np.nan, dtype=np.float32)
    sd = np.full((n, n), np.nan, dtype=np.float32)
    for r, (_, ip) in enumerate(blocks):
        for c, (_, jp) in enumerate(blocks):
            m = (W_con[np.ix_(ip, jp)] != 0)
            if not m.any():
                continue
            cmean = float(np.abs(W_con[np.ix_(ip, jp)])[m].mean())
            if cmean <= 0:
                continue
            per_k = np.array([np.abs(Ws[k][np.ix_(ip, jp)])[m].mean()
                              for k in range(K)])
            l2 = np.log2(per_k / cmean)
            lr[r, c] = float(l2.mean()); sd[r, c] = float(l2.std())
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("0.92")
    vmax = max(float(np.nanmax(np.abs(lr))) if np.isfinite(lr).any() else 0.5, 0.5)
    im = ax.imshow(np.ma.masked_invalid(lr), cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation="nearest", aspect="equal")
    for r in range(n):
        for c in range(n):
            if not np.isnan(lr[r, c]):
                ax.text(c, r, f"{lr[r, c]:+.1f}\n±{sd[r, c]:.1f}", ha="center",
                        va="center", fontsize=6.5,
                        color="white" if abs(lr[r, c]) > 0.6 * vmax else "black")
    labs = [b[0] for b in blocks]
    ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=45, ha="right",
                                                fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlabel("presynaptic", fontsize=9); ax.set_ylabel("postsynaptic",
                                                            fontsize=9)
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.9)
    cb.ax.tick_params(labelsize=7); cb.set_label(r"$\log_2$ ratio", fontsize=8)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_v1", default="zebrafish_hd_si_ipn12_v1")
    p.add_argument("--base_v2", default="zebrafish_hd_si_ipn12_v2")
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=None)
    p.add_argument("--out", default=os.path.join(
        here, "fig_w_rec_comparison_ipn12.png"))
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass
    device = torch.device(args.device)

    print(f"[{args.base_v1}] loading {args.n_folds} folds ...")
    Wc1, Ws1, nt, names = _load_cv_W_rec(args.base_v1, args.n_folds, device)
    print(f"[{args.base_v2}] loading {args.n_folds} folds ...")
    Wc2, Ws2, _, _ = _load_cv_W_rec(args.base_v2, args.n_folds, device)
    assert (Wc1 != 0).sum() == (Wc2 != 0).sum(), "support must match"
    blocks = _coarse_blocks(nt, names)

    fig = plt.figure(figsize=(13, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1],
                          wspace=0.28, hspace=0.32,
                          left=0.07, right=0.96, top=0.95, bottom=0.05)
    # panel a: same size as the others (one column) and left-aligned; top-right
    # cell left blank. Titles removed from every panel (info moves to caption).
    ax_aff = fig.add_subplot(gs[0, 0])
    st = _panel_affinity(ax_aff, Wc1, Ws1, Ws2)
    fig.add_subplot(gs[0, 1]).axis("off")

    _panel_cv(fig.add_subplot(gs[1, 0]), Ws1, Wc1, blocks, "")
    _panel_cv(fig.add_subplot(gs[1, 1]), Ws2, Wc2, blocks, "")
    _panel_logratio(fig.add_subplot(gs[2, 0]), Ws1, Wc1, blocks, "")
    _panel_logratio(fig.add_subplot(gs[2, 1]), Ws2, Wc2, blocks, "")

    for ax, lab in ((ax_aff, "a"),):
        ax.text(-0.02, 1.02, lab, transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="bottom", ha="right")
    fig.text(0.045, 0.635, "b", fontsize=16, fontweight="bold")
    fig.text(0.045, 0.315, "c", fontsize=16, fontweight="bold")

    # Cross-variant cosine excluding the flipped IPN12-outgoing edges, to show
    # v1 and v2 agree everywhere else.
    coarse = np.array([_coarse_of(names[int(t)]) for t in nt], dtype=object)
    ipn12_cols = np.where(coarse == "IPN12")[0]
    non_ipn12 = (Wc1 != 0).copy()
    non_ipn12[:, ipn12_cols] = False
    cross_non = np.mean([
        _cos(Ws1[i] * non_ipn12, Ws2[j] * non_ipn12)
        for i in range(Ws1.shape[0]) for j in range(Ws2.shape[0])])

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {args.out}\n")
    print(f"  within-v1 cosine : {st['within_v1'][0]:.3f} ± {st['within_v1'][1]:.3f}")
    print(f"  within-v2 cosine : {st['within_v2'][0]:.3f} ± {st['within_v2'][1]:.3f}")
    print(f"  v1<->v2 cosine   : {st['cross'][0]:.3f} ± {st['cross'][1]:.3f}  "
          f"(excl. IPN12-outgoing edges: {cross_non:.3f})")
    print(f"  v1 vs W_con      : {st['vs_con_v1'][0]:.3f} ± {st['vs_con_v1'][1]:.3f}")
    print(f"  v2 vs W_con      : {st['vs_con_v2'][0]:.3f} ± {st['vs_con_v2'][1]:.3f}")


if __name__ == "__main__":
    main()
