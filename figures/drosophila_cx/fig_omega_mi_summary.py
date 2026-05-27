"""Per-cell-type angular-velocity mutual-information summary across the
three converged CX models. Same 3 x 2 layout as the HD MI figure but with
omega (instantaneous angular velocity, rad/s) as the target variable
instead of theta. Uniform binning over the empirical omega range.

Output: figures/drosophila/fig_omega_mi_summary.png
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
    ("drosophila_cx_pi",                        "Known-ODE RNN"),
    ("drosophila_cx_pi_fc",                     "fully connected RNN"),
    ("drosophila_cx_pi_gnn_tailloss_unsquared", "GNN"),
]

# Same x-axis ordering and colour assignment as fig_hd_mi_summary.png, so
# the two figures read as a matched pair. Order: descending Known-ODE per-
# neuron HD MI (EPGt strongest, ER6 silent).
HD_TYPE_ORDER = ["EPGt", "EPG", "PEG", "Delta7", "PEN_b(PEN2)",
                 "PEN_a(PEN1)", "ER6"]


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
    # batch.stimulus[..., 0] = omega in deg/s. Convert to rad/s for the target.
    omega = np.deg2rad(batch.stimulus[0, :, 0].cpu().numpy())
    return rates, omega


def _omega_edges(omega, n_w):
    """Symmetric edges around 0 covering ~99% of the OU distribution.

    Using a fixed [-q, q] range with q = 99th percentile of |omega| avoids
    long-tail bins with zero counts.
    """
    q = float(np.quantile(np.abs(omega), 0.99))
    q = max(q, 1e-6)
    return np.linspace(-q, q, n_w + 1)


def _mi_neuron(r, omega, n_w=32, n_r=20, edges=None):
    """Plug-in MI I(r; omega) in bits for one neuron."""
    if r.std() < 1e-8:
        return 0.0
    if edges is None:
        edges = _omega_edges(omega, n_w)
    wi = np.clip(np.digitize(omega, edges) - 1, 0, n_w - 1)
    re = np.linspace(r.min() - 1e-8, r.max() + 1e-8, n_r + 1)
    ri = np.clip(np.digitize(r, re) - 1, 0, n_r - 1)
    j, _, _ = np.histogram2d(wi, ri, bins=[n_w, n_r])
    if j.sum() == 0:
        return 0.0
    j /= j.sum()
    pw = j.sum(axis=1, keepdims=True)
    pr = j.sum(axis=0, keepdims=True)
    nz = j > 0
    return float((j[nz] * np.log2(j[nz] / (pw @ pr)[nz])).sum())


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


def _analyse(rates, omega, neuron_types, n_omega_bins, n_r_bins, seed):
    nt = np.asarray(neuron_types).astype(int)
    edges = _omega_edges(omega, n_omega_bins)
    omega_bin = np.clip(np.digitize(omega, edges) - 1,
                        0, n_omega_bins - 1).astype(np.int64)

    per = {}    # type_id -> list of per-neuron MI
    joint = {}  # type_id -> joint MI
    for t in sorted(set(nt.tolist())):
        idx = np.where(nt == t)[0]
        per[t] = [_mi_neuron(rates[:, i], omega,
                             n_w=n_omega_bins, n_r=n_r_bins, edges=edges)
                  for i in idx]
        joint[t] = _mi_joint_logreg(rates[:, idx], omega_bin,
                                    n_bins=n_omega_bins, seed=seed)
    # joint over ALL neurons in the model.
    joint_all = _mi_joint_logreg(rates, omega_bin,
                                  n_bins=n_omega_bins, seed=seed)
    return per, joint, joint_all


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_steps_ou", type=int, default=10000)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--n_omega_bins", type=int, nargs="+", default=[32, 64],
                   help="one or more omega-bin counts. One figure per value.")
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

    # Run one OU rollout per model, reuse the rates/omega for every bin count.
    rollouts = []   # list of (title, rates, omega, neuron_types, type_names)
    for cfg, title in MODELS:
        net = _load(cfg, device)
        rates, omega = _run_ou(net, args.n_steps_ou, device, seed=args.seed)
        rollouts.append((title, rates, omega,
                         np.asarray(net.neuron_types).astype(int),
                         list(net.type_names)))

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for n_bins in args.n_omega_bins:
        results = []   # list of (title, per, joint, joint_all, type_names)
        print(f"\n=== n_omega_bins = {n_bins} ===")
        for title, rates, omega, nt, names in rollouts:
            per, joint, joint_all = _analyse(
                rates, omega, nt, n_bins, args.n_r_bins, args.seed,
            )
            results.append((title, per, joint, joint_all, names))
            print(f"\n{title}:")
            for t in sorted(per.keys()):
                mu = float(np.mean(per[t])) if per[t] else 0.0
                print(f"  {names[t]:18s} n={len(per[t]):3d}  "
                      f"per-neuron mean={mu:.3f}  joint={joint[t]:.3f}")
            print(f"  joint (ALL {sum(len(v) for v in per.values())} neurons) "
                  f"= {joint_all:.3f} bits   "
                  f"(H(omega)<=log2({n_bins})={math.log2(n_bins):.2f})")
            # Velocity-precision via Gaussian channel: sigma_post = sigma_prior * 2^(-I).
            omega_use = next(r[2] for r in rollouts if r[0] == title)
            sigma_omega_deg = float(np.std(np.rad2deg(omega_use)))
            sigma_post_deg = sigma_omega_deg * (2.0 ** (-joint_all))
            print(f"  empirical sigma(omega)={sigma_omega_deg:.2f} deg/s "
                  f"-> Gaussian-channel sigma_post={sigma_post_deg:.2f} deg/s")

        # Type ordering: use the HD-figure order so the two figures read as
        # a matched pair. Resolve names -> type_ids per (first) model.
        name_ref = results[0][4]
        name_to_id = {n: i for i, n in enumerate(name_ref)}
        type_order = [name_to_id[n] for n in HD_TYPE_ORDER if n in name_to_id]
        labels = [name_ref[t] for t in type_order]
        # Color palette: tab10 indexed by the model's own type_id, matching
        # the HD MI figure so the same cell type lights up in the same color
        # in both figures.
        palette = plt.get_cmap("tab10").colors
        cols = [palette[t % len(palette)] for t in type_order]

        ymax_per = max(
            max(per[t] + [0.0]) for _, per, *_ in results for t in per
        ) * 1.10
        ymax_joint = max(
            joint[t] for _, _, joint, *_ in results for t in joint
        ) * 1.10

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
        out = os.path.join(out_dir, f"fig_omega_mi_summary_nbins{n_bins}.png")
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
