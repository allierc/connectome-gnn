"""Readout-weight and HD mutual-information analysis on the zebrafish HD model.

Two complementary questions on the trained zebrafish_hd_si_dipn model:

1. **Where does the model read HD from?**  W_out is (2, 443) — the
   readout to (cos theta, sin theta). Rows of W_out give the
   contribution of each dIPN neuron to the two HD outputs. We render
   the full heatmap and the per-cell-type mean |W_out| so the
   "informative" types are obvious.

2. **Which neuron types encode HD?**  Mirroring the drosophila
   `fig_hd_mi_summary.py` analysis, we compute per-neuron MI
   I(activity; theta) by plug-in histogram, then aggregate per cell
   type. This is a model-free measure (doesn't depend on W_out being
   linear); a type with high MI but low W_out weight is a candidate
   read-out gap.

Layout (2x2):
  a. W_out heatmap (cos / sin rows, 443 columns sorted by type)
  b. mean |W_out| per cell type, separate cos / sin bars
  c. per-neuron MI per cell type (bar = mean, dots = individual neurons)
  d. joint MI per cell type (CV-logreg lower bound, all neurons of that
     type pooled)

Usage:
  python fig_zebrafish_readout_mi.py --n_steps 30000 --seed 0
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from fig_zebrafish_anatomy_3d_voltage_anim import _load, _run_swim
from connectome_gnn.utils import load_data_root_from_json, set_data_root


# ── MI estimators (verbatim from fig_hd_mi_summary.py) ────────────────────

def _mi_neuron(r, theta, n_t=32, n_r=20):
    """Plug-in MI I(r; theta) in bits for one neuron."""
    if r.std() < 1e-8:
        return 0.0
    tw = np.angle(np.exp(1j * theta))
    ti = np.clip(np.digitize(tw, np.linspace(-np.pi, np.pi, n_t + 1)) - 1,
                 0, n_t - 1)
    re = np.linspace(r.min() - 1e-8, r.max() + 1e-8, n_r + 1)
    ri = np.clip(np.digitize(r, re) - 1, 0, n_r - 1)
    j, _, _ = np.histogram2d(ti, ri, bins=[n_t, n_r])
    j /= j.sum()
    pt = j.sum(axis=1, keepdims=True)
    pr = j.sum(axis=0, keepdims=True)
    nz = j > 0
    return float((j[nz] * np.log2(j[nz] / (pt @ pr)[nz])).sum())


def _mi_joint_logreg(R_group, theta_bin, n_bins, n_splits=5, C=1.0, seed=0):
    """CV-logreg lower bound on I(R_group; theta_bin) in bits."""
    T, K = R_group.shape
    if K == 0 or T < 4 * n_splits:
        return 0.0
    counts = np.bincount(theta_bin, minlength=n_bins).astype(np.float64)
    p_theta = counts / counts.sum()
    nz = p_theta > 0
    h_theta_emp = float(-(p_theta[nz] * np.log2(p_theta[nz])).sum())

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    total_nats = 0.0
    n_total = 0
    for tr, te in kf.split(R_group, theta_bin):
        clf = LogisticRegression(max_iter=500, C=C, solver="lbfgs")
        clf.fit(R_group[tr], theta_bin[tr])
        log_p = clf.predict_log_proba(R_group[te])
        cls_to_col = {int(c): j for j, c in enumerate(clf.classes_)}
        col = np.array([cls_to_col.get(int(y), -1) for y in theta_bin[te]])
        valid = col >= 0
        if not valid.any():
            continue
        lp = log_p[np.arange(len(te))[valid], col[valid]]
        lp = np.clip(lp, math.log(1e-12), 0.0)
        total_nats += -lp.sum()
        n_total += int(valid.sum())
    if n_total == 0:
        return 0.0
    h_cond_bits = (total_nats / n_total) / math.log(2)
    return float(max(0.0, h_theta_emp - h_cond_bits))


# ── helpers ───────────────────────────────────────────────────────────────

def _category_of(type_name: str) -> str:
    """Map fine type like 'IPNd13B' / 'pt-IPN1' -> coarse category."""
    if type_name.startswith("IPNds"):
        return "IPNds"
    if type_name.startswith("IPNd"):
        return "IPNd"
    if type_name.startswith("RIPN"):
        return "RIPN"
    if type_name.startswith("pt-IPN"):
        return "pt-IPN"
    return "other"


_CATEGORY_COLOR = {
    "IPNd":   "#1f77b4",
    "IPNds":  "#2ca02c",
    "RIPN":   "#d62728",
    "pt-IPN": "#9467bd",
    "other":  "#7f7f7f",
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="zebrafish_hd_si_dipn")
    p.add_argument("--n_steps", type=int, default=30000)
    p.add_argument("--burn_in_s", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_theta_bins", type=int, default=32)
    p.add_argument("--n_r_bins", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=None)
    p.add_argument("--out_path", default=None)
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass

    device = torch.device(args.device)

    print(f"[1/4] loading model: {args.model}")
    net, _ = _load(args.model, device)
    dt = float(net.dt)
    type_names = list(net.type_names)
    neuron_types = np.asarray(net.neuron_types).astype(int)

    # W_out: shape (n_output=2, n_readout=443). The readout is gated to the
    # first 443 neurons (output_from_dipn_only=True).
    W_out = net.W_out.detach().cpu().numpy()  # (2, 443)
    n_readout = W_out.shape[1]
    print(f"      W_out shape = {W_out.shape}  (cos / sin readout)")
    print(f"      n_total = {len(neuron_types)}  n_types = {len(type_names)}")

    print(f"[2/4] swim rollout n_steps={args.n_steps} "
          f"({args.n_steps * dt:.0f} s, seed={args.seed})")
    h, theta, _omega, _decoded, *_ = _run_swim(
        net, args.n_steps, dt, device, seed=args.seed)

    burn = int(args.burn_in_s / dt)
    h = h[burn:]
    theta = theta[burn:]

    # ── 3. analysis: MI per neuron, MI per type ─────────────────────────
    print(f"[3/4] MI (n_theta_bins={args.n_theta_bins})")
    theta_w = np.angle(np.exp(1j * theta))
    edges = np.linspace(-np.pi, np.pi, args.n_theta_bins + 1)
    theta_bin = np.clip(np.digitize(theta_w, edges) - 1,
                        0, args.n_theta_bins - 1).astype(np.int64)

    per_type_mi = {}     # type_id -> list of per-neuron MI
    joint_per_type = {}
    for t in sorted(set(neuron_types.tolist())):
        idx = np.where(neuron_types == t)[0]
        per_type_mi[t] = [
            _mi_neuron(h[:, i], theta,
                       n_t=args.n_theta_bins, n_r=args.n_r_bins)
            for i in idx
        ]
        joint_per_type[t] = _mi_joint_logreg(
            h[:, idx], theta_bin, n_bins=args.n_theta_bins, seed=args.seed,
        )
        print(f"      {type_names[t]:14s} n={len(idx):3d}  "
              f"mean_MI={np.mean(per_type_mi[t]):.3f}  "
              f"joint={joint_per_type[t]:.3f}")

    h_theta_ceil = float(math.log2(args.n_theta_bins))

    # Type ordering: by mean per-neuron MI, descending
    all_types = sorted(per_type_mi.keys())
    means = {t: float(np.mean(per_type_mi[t])) for t in all_types}
    type_order = sorted(all_types, key=lambda t: -means[t])
    labels = [type_names[t] for t in type_order]
    cats = [_category_of(lab) for lab in labels]
    cols = [_CATEGORY_COLOR[c] for c in cats]

    # ── 4. render ───────────────────────────────────────────────────────
    print("[4/4] render")
    fig = plt.figure(figsize=(15.0, 8.5), facecolor="white")
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 1.4], width_ratios=[1.5, 1.0],
        left=0.06, right=0.985, top=0.93, bottom=0.18,
        hspace=0.55, wspace=0.22,
    )

    # panel a — W_out heatmap, columns sorted by type (group by category)
    # Build a permutation that sorts the 443 readout neurons by category,
    # then by fine type within category, with category banding.
    readout_types = neuron_types[:n_readout]  # types of first 443 neurons
    readout_cats = np.array(
        [_category_of(type_names[t]) for t in readout_types]
    )
    cat_to_rank = {"IPNd": 0, "IPNds": 1, "RIPN": 2, "pt-IPN": 3, "other": 9}
    sort_key = np.array(
        [(cat_to_rank[c], readout_types[i]) for i, c in enumerate(readout_cats)],
        dtype=[("c", int), ("t", int)],
    )
    perm = np.argsort(sort_key, order=("c", "t"))
    W_perm = W_out[:, perm]
    perm_cats = readout_cats[perm]

    ax = fig.add_subplot(gs[0, 0])
    vlim = float(np.abs(W_out).max())
    im = ax.imshow(W_perm, aspect="auto", cmap="RdBu_r",
                   vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["cos", "sin"])
    ax.set_xlabel("readout neuron (sorted by category)")
    ax.set_title(
        f"W_out heatmap (2 x {n_readout}), |w|_max = {vlim:.2f}",
        fontsize=11,
    )
    ax.text(-0.10, 1.06, "a", transform=ax.transAxes,
            ha="left", va="top", fontsize=13, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("W_out weight", fontsize=9)
    # Category band markers
    boundaries = np.where(np.diff([cat_to_rank[c] for c in perm_cats]))[0] + 1
    centers = np.concatenate(([0], boundaries, [n_readout]))
    for b in boundaries:
        ax.axvline(b - 0.5, color="black", lw=0.8, alpha=0.6)
    for lo, hi in zip(centers[:-1], centers[1:]):
        cat = perm_cats[lo]
        ax.text((lo + hi) / 2, -0.7, f"{cat} (n={hi - lo})",
                ha="center", va="bottom", fontsize=8,
                color=_CATEGORY_COLOR[cat])

    # panel b — mean |W_out| per fine type (cos / sin separated)
    # Only IPNd / IPNds types contribute to W_out.
    readout_unique_types = sorted(set(readout_types.tolist()),
                                  key=lambda t: -np.mean(
                                      np.abs(W_out[:, readout_types == t])))
    labels_b = [type_names[t] for t in readout_unique_types]
    cats_b = [_category_of(lab) for lab in labels_b]
    cols_b = [_CATEGORY_COLOR[c] for c in cats_b]
    mean_cos = np.array([
        np.abs(W_out[0, readout_types == t]).mean()
        for t in readout_unique_types
    ])
    mean_sin = np.array([
        np.abs(W_out[1, readout_types == t]).mean()
        for t in readout_unique_types
    ])
    ax = fig.add_subplot(gs[0, 1])
    xs = np.arange(len(readout_unique_types))
    ax.bar(xs - 0.18, mean_cos, width=0.35, color=cols_b,
           alpha=0.85, edgecolor="black", lw=0.4, label="cos")
    ax.bar(xs + 0.18, mean_sin, width=0.35, color=cols_b,
           alpha=0.45, edgecolor="black", lw=0.4, label="sin",
           hatch="//")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels_b, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("mean |W_out|")
    ax.set_title("per-type readout magnitude (cos / sin)", fontsize=11)
    ax.text(-0.10, 1.06, "b", transform=ax.transAxes,
            ha="left", va="top", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # panel c — per-neuron MI per type
    ax = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(0)
    means_arr = np.array([np.mean(per_type_mi[t]) for t in type_order])
    xs = np.arange(len(type_order))
    ax.bar(xs, means_arr, color=cols, alpha=0.85, edgecolor="black", lw=0.4)
    for i, t in enumerate(type_order):
        vals = per_type_mi[t]
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(i + jitter, vals, s=6, color="black", alpha=0.45,
                   edgecolors="none")
    ax.axhline(h_theta_ceil, color="0.5", lw=0.8, ls="--",
               label=f"H(theta) = log2({args.n_theta_bins})")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("per-neuron MI (bits)")
    ax.set_title("per-neuron HD mutual information (sorted by mean)",
                 fontsize=11)
    ax.text(-0.10, 1.06, "c", transform=ax.transAxes,
            ha="left", va="top", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # panel d — joint MI per type
    ax = fig.add_subplot(gs[1, 1])
    joint_arr = np.array([joint_per_type[t] for t in type_order])
    ax.bar(xs, joint_arr, color=cols, alpha=0.85, edgecolor="black", lw=0.4)
    ax.axhline(h_theta_ceil, color="0.5", lw=0.8, ls="--",
               label=f"H(theta) = log2({args.n_theta_bins})")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("joint MI per type (bits)")
    ax.set_title("joint MI per cell type (CV-logreg lower bound)",
                 fontsize=11)
    ax.text(-0.10, 1.06, "d", transform=ax.transAxes,
            ha="left", va="top", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    if args.out_path is None:
        args.out_path = os.path.join(here, "fig_zebrafish_readout_mi.png")
    fig.savefig(args.out_path, dpi=180)
    print(f"saved {args.out_path}")


if __name__ == "__main__":
    main()
