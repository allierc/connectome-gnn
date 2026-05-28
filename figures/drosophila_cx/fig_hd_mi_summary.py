"""Per-cell-type HD mutual-information summary across the three converged
CX models. 3 rows (Known-ODE RNN, fully connected RNN, GNN) x 2 cols:

  col 1: per-neuron mean MI per cell type (bar = mean, dots = individual
         neurons). Captures the "average HD informativeness of a cell of
         this type".
  col 2: joint MI per cell type (CV-logreg lower bound). Captures the
         "what does the population of this type collectively encode".

Both columns share the same x-axis ordering: cell types are sorted by the
first model's per-neuron mean MI (descending), so vertical comparison
across rows is a per-cell-type fingerprint test.

Dashed grey line: H(theta) = log2(n_theta_bins) bits (entropy ceiling).

Output: figures/drosophila/fig_hd_mi_summary.png
"""
from __future__ import annotations

import argparse
import glob
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

from connectome_gnn.utils import log_path, load_data_root_from_json, set_data_root
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.models.registry import create_model
from connectome_gnn.generators.utils import generate_path_integration_batch


MODELS = [
    ("drosophila_cx_pi_epg",                    "Known-ODE RNN"),
    ("drosophila_cx_pi_fc_epg",                 "fully connected RNN"),
    ("drosophila_cx_pi_gnn_epg",                "GNN"),
]


def _load(config_name, device, prefer_epoch=None):
    config, _ = load_run_config(config_name, explicit_output_root=False, task="train")
    ckpt_dir = os.path.join(log_path(config.config_file), "models")
    cands = sorted(
        glob.glob(os.path.join(ckpt_dir, "best_model_with_0_graphs_*.pt")),
        key=lambda p_: int(p_.rsplit("_", 1)[1].rstrip(".pt")),
    )
    if not cands:
        raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
    if prefer_epoch is None and "gnn_tailloss" in config_name:
        prefer_epoch = 5
    elif prefer_epoch is None and "gnn_epg" in config_name:
        prefer_epoch = 3
    ckpt_path = cands[-1]
    if prefer_epoch is not None:
        match = [p_ for p_ in cands
                 if int(p_.rsplit("_", 1)[1].rstrip(".pt")) == prefer_epoch]
        if match:
            ckpt_path = match[0]
    net = create_model(
        config.graph_model.signal_model_name,
        aggr_type=config.graph_model.aggr_type,
        config=config, device=device,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(state["model_state_dict"])
    net.eval()
    print(f"loaded {config_name}: {ckpt_path}")
    return net


def _run_ou(net, n_steps, device, seed=0):
    rng = np.random.default_rng(seed)
    batch = generate_path_integration_batch(
        batch_size=1, n_steps=n_steps, dt=float(net.dt), device=device, rng=rng,
    )
    with torch.no_grad():
        _, h = net(batch.stimulus)
    rates = torch.sigmoid(h[0]).cpu().numpy()
    theta = batch.theta_hd[0].cpu().numpy()
    return rates, theta


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
        clf = LogisticRegression(max_iter=500, C=C, solver="lbfgs",
                                 multi_class="multinomial")
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


def _analyse(rates, theta, neuron_types, n_theta_bins, n_r_bins, seed):
    nt = np.asarray(neuron_types).astype(int)
    theta_w = np.angle(np.exp(1j * theta))
    edges = np.linspace(-np.pi, np.pi, n_theta_bins + 1)
    theta_bin = np.clip(np.digitize(theta_w, edges) - 1,
                        0, n_theta_bins - 1).astype(np.int64)

    per = {}    # type_id -> list of per-neuron MI
    joint = {}  # type_id -> joint MI
    for t in sorted(set(nt.tolist())):
        idx = np.where(nt == t)[0]
        per[t] = [_mi_neuron(rates[:, i], theta,
                             n_t=n_theta_bins, n_r=n_r_bins) for i in idx]
        joint[t] = _mi_joint_logreg(rates[:, idx], theta_bin,
                                    n_bins=n_theta_bins, seed=seed)
    # joint over ALL neurons in the model.
    joint_all = _mi_joint_logreg(rates, theta_bin,
                                  n_bins=n_theta_bins, seed=seed)
    return per, joint, joint_all


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_steps_ou", type=int, default=10000)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--n_theta_bins", type=int, nargs="+", default=[32, 64],
                   help="one or more theta-bin counts. One figure per value.")
    p.add_argument("--n_r_bins",     type=int, default=20)
    p.add_argument("--device",       default="cpu")
    p.add_argument("--output_root",  default=None)
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass

    device = torch.device(args.device)

    # Run one OU rollout per model, reuse the rates/theta for every bin count.
    rollouts = []   # list of (title, rates, theta, neuron_types, type_names)
    for cfg, title in MODELS:
        net = _load(cfg, device)
        rates, theta = _run_ou(net, args.n_steps_ou, device, seed=args.seed)
        rollouts.append((title, rates, theta,
                         np.asarray(net.neuron_types).astype(int),
                         list(net.type_names)))

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for n_bins in args.n_theta_bins:
        results = []   # list of (title, per, joint, joint_all, type_names)
        print(f"\n=== n_theta_bins = {n_bins} ===")
        for title, rates, theta, nt, names in rollouts:
            per, joint, joint_all = _analyse(
                rates, theta, nt, n_bins, args.n_r_bins, args.seed,
            )
            results.append((title, per, joint, joint_all, names))
            print(f"\n{title}:")
            for t in sorted(per.keys()):
                mu = float(np.mean(per[t])) if per[t] else 0.0
                print(f"  {names[t]:18s} n={len(per[t]):3d}  "
                      f"per-neuron mean={mu:.3f}  joint={joint[t]:.3f}")
            print(f"  joint (ALL {sum(len(v) for v in per.values())} neurons) "
                  f"= {joint_all:.3f} bits   "
                  f"(H(theta)=log2({n_bins})={math.log2(n_bins):.2f})")

        # Type ordering = first model's per-neuron mean, descending.
        all_types = sorted({t for _, per, *_ in results for t in per.keys()})
        means_ref = {t: float(np.mean(results[0][1].get(t, [0.0]) or [0.0]))
                     for t in all_types}
        type_order = sorted(all_types, key=lambda t: -means_ref[t])
        name_ref = results[0][4]
        labels = [name_ref[t] for t in type_order]

        ymax_per = max(
            max(per[t] + [0.0]) for _, per, *_ in results for t in per
        ) * 1.10
        ymax_joint = max(
            joint[t] for _, _, joint, *_ in results for t in joint
        ) * 1.10

        palette = plt.get_cmap("tab10").colors
        cols = [palette[t % len(palette)] for t in type_order]

        fig, axes = plt.subplots(len(MODELS), 2,
                                  figsize=(11, 3.4 * len(MODELS)),
                                  sharex=False)
        rng = np.random.default_rng(0)

        panel_letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
        for row, (title, per, joint, joint_all, _) in enumerate(results):
            means = np.array([np.mean(per.get(t, [0.0]) or [0.0])
                              for t in type_order])
            ax = axes[row, 0]
            xs = np.arange(len(type_order))
            ax.bar(xs, means, color=cols, edgecolor="0.3", linewidth=0.5,
                   alpha=0.75)
            for k, t in enumerate(type_order):
                vals = np.array(per.get(t, []))
                if vals.size == 0:
                    continue
                jit = rng.uniform(-0.18, 0.18, size=vals.size)
                ax.scatter(np.full(vals.size, k) + jit, vals,
                           s=10, color="0.15", alpha=0.7, linewidths=0)
            ax.set_ylim(0, ymax_per)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("per-neuron MI (bits)", fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.text(-0.13, 1.04, f"{panel_letters[2*row]}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=13, fontweight="bold")

            ax = axes[row, 1]
            joints = np.array([joint.get(t, 0.0) for t in type_order])
            ax.bar(xs, joints, color=cols, edgecolor="0.3", linewidth=0.5)
            ax.set_ylim(0, ymax_joint)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("joint MI (bits)", fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.text(0.02, 0.95,
                    f"all-neurons joint = {joint_all:.2f} bits",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=9)
            ax.text(-0.13, 1.04, f"{panel_letters[2*row+1]}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=13, fontweight="bold")

        plt.tight_layout()
        out = os.path.join(out_dir, f"fig_hd_mi_summary_nbins{n_bins}.png")
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
