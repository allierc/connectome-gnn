"""Compile / analyse the ARTR / pt-IPN1 head-direction model family.

Reads each run dir under
    log/zebrafish/zebrafish_hd_si_ipn12_artr_pt1_<suffix>/
and overlays cross-model summary panels:

    (a) Final test loss per variant (heading and/or distance / position
        head as available).
    (b) Per-frame test R² per variant, split by output channel.
    (c) Connectome-vs-learned per-edge correlation: Pearson r between
        |W_ij^con| and |W_ij^rec|, per variant.
    (d) Per-cell-type recurrent magnitude amplification |W^rec| / |W^con|
        averaged within each cell-type block, stacked across variants.

The figure is a single, self-describing summary of how the connectome
template re-scales under increasingly demanding sub-tasks (rotation
→ +distance → +2D path), and serves as a launchpad for the
MI-partition and edge-analysis figures that come next.

Usage:
    python figures/zebrafish/fig_zebrafish_artr_pt1_summary.py
    python figures/zebrafish/fig_zebrafish_artr_pt1_summary.py \\
        --data_root /groups/saalfeld/home/allierc/GraphData \\
        --out figures/zebrafish/fig_zebrafish_artr_pt1_summary.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from connectome_gnn.utils import (  # noqa: E402
    load_data_root_from_json, set_data_root,
)
from connectome_gnn.models.utils import load_run_config  # noqa: E402
from connectome_gnn.models.registry import create_model  # noqa: E402


PREFIX = "zebrafish_hd_si_ipn12_artr_pt1_"
VARIANTS = [
    # (suffix, short label)
    ("selfmotion_rotation",          "rot"),
    ("selfmotion_translation",       "trans"),
    ("selfmotion_translation_leaky", "trans\nleaky"),
    ("selfmotion_both",              "rot+d"),
    ("selfmotion_both_leaky",        "rot+d\nleaky"),
    ("position_2d",                  "rot+(x,y)"),
    ("position_2d_leaky",            "rot+(x,y)\nleaky"),
]

# Coarse cell-type bins for the per-block amplification panel.
COARSE_ORDER = ["IPNd", "IPNds", "IPN12", "RIPN", "pt-IPN"]


def _coarse_of(name: str) -> str:
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


def _read_final_metrics(run_dir: str) -> dict:
    """Last row of tmp_training/metrics.log → dict of trailing test-R² cols."""
    path = os.path.join(run_dir, "tmp_training", "metrics.log")
    if not os.path.isfile(path):
        return {}
    last = None
    with open(path) as f:
        for row in csv.reader(f):
            last = row
    if not last:
        return {}
    # column 7 = final train MSE; trailing columns are heading/d R²+RMSE pairs.
    out = {"mse": _to_float(last[7])}
    # By convention the last 5 numeric columns are R²/RMSE pairs (heading
    # gain, heading R², heading RMSE, secondary R², secondary RMSE) but
    # some columns may be nan if the head wasn't supervised. Pick out the
    # plausible R² values (between 0 and 1.0001).
    r2_vals = []
    for v in last[8:]:
        try:
            x = float(v)
        except ValueError:
            continue
        if 0.0 <= x <= 1.0001:
            r2_vals.append(x)
    out["r2_vals"] = r2_vals
    return out


def _to_float(s) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _load_W(run_dir: str, device):
    """Reconstruct the trained model from the run dir, return
    (W^rec, W^con, neuron_types, type_names)."""
    # The run-dir basename matches the config name under config/zebrafish/.
    config_name = os.path.basename(run_dir)
    config, _ = load_run_config(config_name, explicit_output_root=False,
                                 task="train")
    net = create_model(
        config.graph_model.signal_model_name,
        aggr_type=config.graph_model.aggr_type,
        config=config, device=device,
    )
    # load latest checkpoint
    import glob
    ckpts = sorted(glob.glob(os.path.join(run_dir, "models",
                                          "best_model_with_*.pt")))
    if not ckpts:
        return None
    sd = torch.load(ckpts[-1], map_location=device, weights_only=False)
    sd = sd if isinstance(sd, dict) else sd.state_dict()
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    net.load_state_dict(sd, strict=False)
    W_rec = net.W_rec.detach().cpu().numpy()
    W_con = net.W_con.detach().cpu().numpy()
    nt = np.asarray(net.neuron_types).astype(int)
    names = list(net.type_names)
    return W_rec, W_con, nt, names


def _per_block_amplification(W_rec, W_con, coarse) -> dict:
    """Mean |W_rec| / mean |W_con| per coarse-type block."""
    out = {}
    for c in COARSE_ORDER:
        rows = np.where(coarse == c)[0]
        if rows.size == 0:
            out[c] = np.nan
            continue
        # outgoing edges from this block
        w_rec = np.abs(W_rec[rows, :])
        w_con = np.abs(W_con[rows, :])
        mask = w_con > 0
        if not mask.any():
            out[c] = np.nan
            continue
        out[c] = float(w_rec[mask].mean() / max(w_con[mask].mean(), 1e-12))
    return out


def _per_edge_correlation(W_rec, W_con) -> float:
    a = np.abs(W_rec.ravel())
    b = np.abs(W_con.ravel())
    mask = b > 0
    if not mask.any():
        return float("nan")
    a = a[mask]; b = b[mask]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root",
                   default="/groups/saalfeld/home/allierc/GraphData")
    p.add_argument("--out", default=os.path.join(
        HERE, "fig_zebrafish_artr_pt1_summary.png"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    try:
        set_data_root(load_data_root_from_json())
    except FileNotFoundError:
        pass

    device = torch.device(args.device)

    rows = []
    coarse = None
    for suffix, label in VARIANTS:
        run = PREFIX + suffix
        run_dir = os.path.join(args.data_root, "log", "zebrafish", run)
        if not os.path.isdir(run_dir):
            print(f"  SKIP {label}: missing {run_dir}")
            continue
        m = _read_final_metrics(run_dir)
        wrec_wcon = _load_W(run_dir, device)
        if wrec_wcon is None:
            print(f"  SKIP {label}: no checkpoint in {run_dir}")
            continue
        W_rec, W_con, nt, names = wrec_wcon
        coarse_now = np.array([_coarse_of(names[int(t)]) for t in nt],
                              dtype=object)
        coarse = coarse_now if coarse is None else coarse
        amp = _per_block_amplification(W_rec, W_con, coarse_now)
        rho = _per_edge_correlation(W_rec, W_con)
        rows.append({
            "label": label,
            "mse": m.get("mse", float("nan")),
            "r2_vals": m.get("r2_vals", []),
            "rho_wrec_wcon": rho,
            "amp": amp,
        })
        print(f"  {label:14s}  mse={m.get('mse', float('nan')):.4f}  "
              f"ρ(|W_rec|,|W_con|)={rho:.3f}  amp(IPNd)={amp['IPNd']:.2f}")

    if not rows:
        sys.exit("no run dirs found")

    # ---- plot ---------------------------------------------------------
    # Two-panel layout: (a) connectome-magnitude correlation across
    # variants, (b) per-cell-type recurrent-magnitude amplification.
    # Final training loss and per-head test R² panels were dropped
    # (saturated / uninformative at convergence).
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    xs = np.arange(len(rows))
    labels = [r["label"] for r in rows]

    LABEL_FS = 14
    TICK_FS = 12
    LETTER_FS = 20

    # (a) Pearson r(|W_rec|, |W_con|)
    ax = axes[0]
    ax.bar(xs, [r["rho_wrec_wcon"] for r in rows], color="#d49a3a")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_ylabel(r"Pearson $r(|\hat W^{rec}|, |W^{con}|)$",
                   fontsize=LABEL_FS)
    ax.set_ylim(0, 1)

    # (b) per-cell-type amplification, grouped bars
    ax = axes[1]
    n_var = len(rows); n_cls = len(COARSE_ORDER)
    width = 0.8 / max(n_var, 1)
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(rows):
        vals = [r["amp"].get(c, np.nan) for c in COARSE_ORDER]
        x = np.arange(n_cls) + (i - (n_var - 1) / 2) * width
        ax.bar(x, vals, width=width, color=cmap(i % 10),
               label=r["label"].replace("\n", " "))
    ax.set_xticks(np.arange(n_cls))
    ax.set_xticklabels(COARSE_ORDER, fontsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_ylabel(r"$\langle|\hat W^{rec}|\rangle/\langle|W^{con}|\rangle$",
                   fontsize=LABEL_FS)
    ax.legend(fontsize=TICK_FS - 2, ncol=2, loc="upper right",
              frameon=False)

    for letter, ax in zip("ab", axes):
        ax.text(-0.10, 1.04, letter, transform=ax.transAxes,
                fontsize=LETTER_FS, fontweight="bold",
                ha="left", va="bottom")

    plt.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_artr_pt1_summary] wrote {args.out}")


if __name__ == "__main__":
    main()
