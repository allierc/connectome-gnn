"""Figure 11: does the MI sum up? Group (joint) MI vs summed per-neuron MI.

The per-neuron partition (\\cref{fig:zhd_mi_partition}) adds up single-cell
mutual informations. Here we instead estimate the *joint* MI carried by a
whole group of recurrent neurons about the heading θ and about the forward
displacement d, and compare it to the sum of the group's per-neuron MIs.

For each anatomical group (the four families dIPN / dsIPN / IPN12 / IPN-core,
and each of the fish2 recurrent cell types) we compute:

  * summed per-neuron MI   Σ_i I(h_i; ·)        (plug-in, same as Fig. 10)
  * group joint MI         I(h_group; ·)        (cross-validated decoder
                                                 lower bound, eq. mi_joint_lb)

The decoder LB is  H(·) − CE_test  in bits, with an L2 multinomial logistic
regression predicting the binned target from the group's activity vector on a
held-out split. If neurons coded independently the group MI would track the
sum; in a redundant code it saturates near the target entropy H(·) while the
sum keeps growing with group size.

Four columns (one model each) × two rows (heading, translation).

Usage:
    python figures/zebrafish/fig_zebrafish_mi_grouped.py --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."), )
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from connectome_gnn.utils import (  # noqa: E402
    load_data_root_from_json, set_data_root, graphs_data_path,
)
from connectome_gnn.zarr_io import load_raw_array  # noqa: E402

# Reuse the model-loading / MI machinery of the per-neuron partition figure.
from fig_zebrafish_mi_partition import (  # noqa: E402
    _load_run, _accumulate_hidden, _circular_mi, _mi_plugin, _is_recurrent,
    _rec_family, REC_FAMILY_ORDER, REC_FAMILY_COLOR, DEFAULT_RUNS,
    _PROFILE_BY_TARGET, _RECOGNISED,
)

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

N_THETA_BINS = 16
N_D_BINS = 16
MAX_DECODER_SAMPLES = 9000   # subsample time points to keep the LR fits fast


def _bin_theta(theta):
    th = ((np.asarray(theta) + np.pi) % (2 * np.pi)) - np.pi
    edges = np.linspace(-np.pi, np.pi + 1e-9, N_THETA_BINS + 1)
    return np.clip(np.digitize(th, edges) - 1, 0, N_THETA_BINS - 1)


def _bin_quantile(d, n_bins):
    d = np.asarray(d, dtype=float)
    qs = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    if qs.size < 3:                       # near-constant target
        return np.zeros(d.size, dtype=int), 1
    lab = np.clip(np.digitize(d, qs[1:-1]), 0, qs.size - 2)
    return lab, qs.size - 1


def _decoder_mi_bits(X, ybin, rng):
    """Cross-validated decoder lower bound  H(y) − CE_test  in bits."""
    vals, counts = np.unique(ybin, return_counts=True)
    if vals.size < 2:
        return 0.0
    # Drop classes with too few members for a stratified split.
    keep_cls = vals[counts >= 4]
    keep = np.isin(ybin, keep_cls)
    X, ybin = X[keep], ybin[keep]
    if X.shape[0] < 50 or np.unique(ybin).size < 2:
        return 0.0
    Xtr, Xte, ytr, yte = train_test_split(
        X, ybin, test_size=0.33, random_state=0, stratify=ybin)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xte))
    cls = list(clf.classes_)
    idx = {c: i for i, c in enumerate(cls)}
    p = np.array([proba[i, idx[y]] if y in idx else 1e-12
                  for i, y in enumerate(yte)])
    p = np.clip(p, 1e-12, 1.0)
    ce = float(-np.mean(np.log2(p)))
    v, c = np.unique(yte, return_counts=True)
    py = c / c.sum()
    Hy = float(-(py * np.log2(py)).sum())
    return max(Hy - ce, 0.0)


def _compute_grouped(run_basename, data_root, n_trials, device, rng):
    run_dir = os.path.join(data_root, "log", "zebrafish", run_basename)
    print(f"  loading {run_dir}")
    net, config = _load_run(run_dir, device)
    type_names = list(net.type_names)
    nt = np.asarray(net.neuron_types).astype(int)
    is_rec = np.array([_is_recurrent(type_names[int(t)]) for t in nt])
    rec_ix = np.where(is_rec)[0]
    ctype = np.array([type_names[int(nt[i])] for i in rec_ix], dtype=object)
    fam = np.array([_rec_family(type_names[int(nt[i])]) for i in rec_ix],
                   dtype=object)

    root = graphs_data_path(config.dataset)
    u_test = load_raw_array(f"{root}/test/stimulus.zarr")
    y_test = load_raw_array(f"{root}/test/target.zarr")
    _task_raw = list(getattr(config.training, "task_targets", None) or [])
    _task_key = tuple(t for t in _RECOGNISED if t in _task_raw)
    if u_test.shape[-1] >= 4 and _task_key:
        key = (int(y_test.shape[-1]), _task_key)
        if key in _PROFILE_BY_TARGET:
            in_cols, out_cols = _PROFILE_BY_TARGET[key]
            u_test = u_test[..., in_cols]
            y_test = y_test[..., out_cols]

    H, theta, d = _accumulate_hidden(net, u_test, y_test, device, n_trials)
    Hrec = H[:, rec_ix]                       # (T, n_rec)

    # per-neuron MI (for the summed reference)
    I_theta = np.array([_circular_mi(Hrec[:, k], theta)
                        for k in range(Hrec.shape[1])])
    I_d = np.array([_mi_plugin(Hrec[:, k], d) for k in range(Hrec.shape[1])])

    # subsample time points for the decoder fits
    T = Hrec.shape[0]
    if T > MAX_DECODER_SAMPLES:
        sel = rng.choice(T, MAX_DECODER_SAMPLES, replace=False)
    else:
        sel = np.arange(T)
    Xs = Hrec[sel]
    yb_theta = _bin_theta(theta)[sel]
    yb_d, _ = _bin_quantile(d, N_D_BINS)
    yb_d = yb_d[sel]

    groups = list(REC_FAMILY_ORDER) + sorted(set(ctype))
    rows = []
    for g in groups:
        gm = (fam == g) if g in REC_FAMILY_ORDER else (ctype == g)
        k = int(gm.sum())
        if k == 0:
            continue
        Xg = Xs[:, gm]
        rows.append(dict(
            group=g, is_family=g in REC_FAMILY_ORDER, n=k,
            sum_theta=float(I_theta[gm].sum()),
            sum_d=float(I_d[gm].sum()),
            grp_theta=_decoder_mi_bits(Xg, yb_theta, rng),
            grp_d=_decoder_mi_bits(Xg, yb_d, rng),
            fam=(g if g in REC_FAMILY_ORDER else _rec_family(g)),
        ))
        print(f"    {g:12s} n={k:3d}  Σθ={rows[-1]['sum_theta']:6.1f} "
              f"grpθ={rows[-1]['grp_theta']:.2f}  "
              f"Σd={rows[-1]['sum_d']:6.1f} grpd={rows[-1]['grp_d']:.2f}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    p.add_argument("--data_root",
                   default="/groups/saalfeld/home/allierc/GraphData")
    p.add_argument("--n_trials", type=int, default=48)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=os.path.join(
        HERE, "fig_zebrafish_mi_grouped.png"))
    args = p.parse_args()
    try:
        set_data_root(load_data_root_from_json())
    except FileNotFoundError:
        pass

    rng = np.random.default_rng(0)
    device = torch.device(args.device)
    results = []
    for j, run in enumerate(args.runs):
        print(f"[{j + 1}/{len(args.runs)}] {run}")
        results.append(_compute_grouped(run, args.data_root, args.n_trials,
                                         device, rng))

    # ---- plot: 2 rows (heading, translation) x N cols (models) -----------
    from matplotlib.lines import Line2D
    ncol = len(results)
    LF, TF, LET = 13, 11, 14
    fig, axes = plt.subplots(2, ncol, figsize=(4.6 * ncol, 9.0),
                             sharex=True, sharey=True)
    letters = "abcdefghijklmnop"
    targets = [("sum_theta", "grp_theta", "heading"),
               ("sum_d", "grp_d", "translation")]

    xmax = max(row[s] for res in results for row in res
               for s in ("sum_theta", "sum_d")) * 1.3
    ymax = max(row[g] for res in results for row in res
               for g in ("grp_theta", "grp_d")) * 1.15 + 0.2

    for i, (skey, gkey, tlabel) in enumerate(targets):
        for j, res in enumerate(results):
            ax = axes[i, j]
            # cell types: small dots
            for row in res:
                if row["is_family"]:
                    continue
                ax.scatter(row[skey], row[gkey], s=22, alpha=0.7,
                           color=REC_FAMILY_COLOR[row["fam"]],
                           edgecolor="none")
            # families: large labelled markers
            for row in res:
                if not row["is_family"]:
                    continue
                ax.scatter(row[skey], row[gkey], s=130, alpha=0.95,
                           color=REC_FAMILY_COLOR[row["fam"]],
                           edgecolor="k", linewidth=0.8, marker="D", zorder=5)
            ax.set_xscale("log")
            ax.set_xlim(0.5, xmax)
            ax.set_ylim(0, ymax)
            if i == 0 and j == 0:
                ax.set_ylabel(r"group joint MI  (bits)", fontsize=LF)
                ax.set_xlabel(r"$\sum_i I(\hat h_i;\,\cdot)$  (bits, summed)",
                              fontsize=LF)
            if j == 0:
                ax.text(0.03, 0.93, tlabel, transform=ax.transAxes,
                        fontsize=LF, fontweight="bold", va="top")
            ax.tick_params(labelsize=TF)
            ax.text(-0.10, 1.04, letters[i * ncol + j], transform=ax.transAxes,
                    fontsize=LET, fontweight="bold")

    handles = [Line2D([0], [0], marker="D", ls="none", ms=9, mec="k",
                      color=REC_FAMILY_COLOR[f], label=f)
               for f in REC_FAMILY_ORDER]
    handles.append(Line2D([0], [0], marker="o", ls="none", ms=6, color="0.5",
                          label="cell type"))
    axes[0, 0].legend(handles=handles, fontsize=8, loc="lower right",
                      frameon=False)
    plt.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_mi_grouped] wrote {args.out}")


if __name__ == "__main__":
    main()
