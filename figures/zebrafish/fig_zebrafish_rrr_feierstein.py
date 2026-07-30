"""Apples-to-apples reduced-rank regression (Feierstein et al., Curr. Biol.
2023) applied to the trained connectome model.

Feierstein et al. imaged the larval-zebrafish hindbrain during decoupled
oculomotor / swimming behaviour and used reduced-rank regression (RRR) to
show that behaviour-related population activity collapses onto an
effectively TWO-dimensional subspace -- one "vergence / translation +
swim" feature and one "rotation" feature -- and that neurons separate by
their loadings into three clusters (translation/swim, leftward rotation,
rightward rotation), with the two rotation clusters segregating to opposite
hemispheres.

Here we run the SAME analysis on the trained connectome model so the two can
be compared directly. Population firing rates r = sigma(h) are regressed on
time-shifted behavioural regressors (kernels), grouped into a translation
family (forward velocity, swim vigour, accumulated displacement) and a
rotation family (angular velocity, heading cos/sin). We report:

  (a) cross-validated population explained variance vs RRR rank -> the
      behaviour subspace dimensionality (their Fig 4C);
  (b) the rank-2 feature regressor-weights, grouped by family -> do the two
      dominant features separate into translation and rotation? (their 4D/6);
  (c) per-neuron contributions on the translation vs rotation feature,
      coloured by ground-truth population (their Fig 5);
  (d) the same, by hemisphere for the rotation afferents (their Fig 7);
  (e) a translation-vs-rotation selectivity index per population -- the
      model's extra handle: WHERE the separation lives (anatomically split
      afferents vs the mixed recurrent integrator).

One script = one figure (fig_zebrafish_rrr_feierstein.png) + a printed
consistency block.

Usage:
    python figures/zebrafish/fig_zebrafish_rrr_feierstein.py --device cuda
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from connectome_gnn.utils import (  # noqa: E402
    load_data_root_from_json, set_data_root, graphs_data_path,
)
from connectome_gnn.models.utils import load_run_config  # noqa: E402
from connectome_gnn.models.registry import create_model  # noqa: E402
from connectome_gnn.zarr_io import load_raw_array  # noqa: E402

# Models for the dimensionality ladder (a). heading-only has rotation drive
# only (no forward velocity) -> behaviour should be ~1-D; both / 2-D paths
# add translation -> ~2-D, the apples-to-apples match to their "two features".
LADDER = (
    ("zebrafish_hd_si_ipn_917_v1_selfmotion_rotation", "heading only", False),
    ("zebrafish_hd_si_ipn_917_v1_selfmotion_both",     "+ distance",   True),
    ("zebrafish_hd_si_ipn_917_v1_position_2d",         "2-D path",     True),
)
FEATURE_RUN = "zebrafish_hd_si_ipn_917_v1_position_2d"   # panels b-e

# time-shift kernels (seconds, past) -- mirror their ~4 s Ca kernels
LAGS_S = (0.0, 0.1, 0.2, 0.4, 0.8, 1.6)

ROT_BASE = ("omega", "cos_th", "sin_th")
TRANS_BASE = ("v_fwd", "swim", "trans_pos")

POP_COLORS = {
    "ARTR_L":   "#cf222e",   # rotation afferent, left  (red)
    "ARTR_R":   "#0072b2",   # rotation afferent, right (blue)
    "ptIPN1":   "#e69f00",   # translation afferent     (orange)
    "recurrent": "#444444",  # dIPN/dsIPN/IPN12/IPN-core (grey)
    "other":    "#bbbbbb",
}
POP_ORDER = ("recurrent", "ptIPN1", "ARTR_L", "ARTR_R", "other")


def _load_run(run_dir, device):
    config_name = os.path.basename(run_dir)
    config, _ = load_run_config(config_name, explicit_output_root=False,
                                task="train")
    net = create_model(config.graph_model.signal_model_name,
                       aggr_type=config.graph_model.aggr_type,
                       config=config, device=device)
    ckpts = sorted(glob.glob(os.path.join(run_dir, "models",
                                          "best_model_with_*.pt")))
    if not ckpts:
        sys.exit(f"no checkpoint in {run_dir}")
    sd = torch.load(ckpts[-1], map_location=device, weights_only=False)
    sd = sd if isinstance(sd, dict) else sd.state_dict()
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net, config


def _profile(u_test, y_test, config):
    """Slice padded stimulus/target columns to the active channels."""
    recognised = ("rotation", "translation", "position_2d")
    task = tuple(t for t in recognised
                 if t in (getattr(config.training, "task_targets", None) or []))
    prof = {
        (3, ("rotation",)):               ([0, 2, 3],    [0, 1]),
        (3, ("rotation", "translation")): ([0, 1, 2, 3], [0, 1, 2]),
        (4, ("position_2d",)):            ([0, 1, 2, 3], [0, 1, 2, 3]),
    }
    key = (int(y_test.shape[-1]), task)
    if u_test.shape[-1] >= 4 and key in prof:
        ic, oc = prof[key]
        return u_test[..., ic], y_test[..., oc], task
    return u_test, y_test, task


def _pop_labels(net):
    """Per-neuron population label over all N neurons."""
    N = len(net.neuron_types)
    type_names = list(net.type_names)
    nt = np.asarray(net.neuron_types).astype(int)
    lab = np.empty(N, dtype=object)
    for i in range(N):
        nm = type_names[int(nt[i])]
        lab[i] = "recurrent" if nm.startswith("IPN") else "other"
    def _mask(buf):
        b = getattr(net, buf, None)
        if b is None:
            return np.zeros(N, dtype=bool)
        return np.asarray(b.detach().cpu().numpy()) > 0.5
    lab[_mask("_afferent_ind_pt_ipn1_l")] = "ptIPN1"
    lab[_mask("_afferent_ind_pt_ipn1_r")] = "ptIPN1"
    lab[_mask("_afferent_ind_artr_l")] = "ARTR_L"
    lab[_mask("_afferent_ind_artr_r")] = "ARTR_R"
    return lab


def _rollout(net, u_test, y_test, device, n_trials, has_trans):
    """Return per-trial firing-rate tensor and behavioural base regressors.

    R: (n_trials, T, N) firing rates; B: dict base_name -> (n_trials, T)."""
    n_trials = min(n_trials, u_test.shape[0])
    Rs = []
    base = {k: [] for k in (ROT_BASE + (TRANS_BASE if has_trans else ()))}
    for k in range(n_trials):
        with torch.no_grad():
            u_t = torch.from_numpy(u_test[k][None]).to(device)
            _, h_buf = net(u_t)
            r = net._sigma(h_buf)[0].cpu().numpy().astype(np.float64)
        Rs.append(r)
        y = y_test[k]
        omega = u_test[k][:, 0].astype(np.float64)
        cos_th, sin_th = y[:, 0].astype(np.float64), y[:, 1].astype(np.float64)
        base["omega"].append(omega)
        base["cos_th"].append(cos_th)
        base["sin_th"].append(sin_th)
        if has_trans:
            vf = u_test[k][:, 1].astype(np.float64)
            base["v_fwd"].append(vf)
            base["swim"].append(np.abs(vf))
            if y.shape[-1] == 4:
                tp = np.sqrt(y[:, 2] ** 2 + y[:, 3] ** 2)
            else:
                tp = y[:, 2].astype(np.float64)
            base["trans_pos"].append(tp.astype(np.float64))
    R = np.stack(Rs, 0)
    B = {k: np.stack(v, 0) for k, v in base.items()}
    return R, B


def _design(B, T, dt):
    """Time-shifted regressor design tensor X: (n_trials, T, P) and the
    (base, lag) provenance of each of the P columns."""
    lags = [int(round(s / dt)) for s in LAGS_S]
    names = list(B.keys())
    n_tr = next(iter(B.values())).shape[0]
    cols, prov = [], []
    for nm in names:
        sig = B[nm]                                   # (n_tr, T)
        for L in lags:
            shifted = np.zeros_like(sig)
            if L == 0:
                shifted = sig
            else:
                shifted[:, L:] = sig[:, :-L]          # past shift, per trial
            cols.append(shifted)
            prov.append((nm, L))
    X = np.stack(cols, axis=-1)                       # (n_tr, T, P)
    return X, prov


def _zfit(A):
    mu = A.mean(0, keepdims=True)
    sd = A.std(0, keepdims=True) + 1e-8
    return mu, sd


def _rrr_popev(Xtr, Ytr, Xte, Yte, ranks, lam):
    """Cross-validated population explained variance at each rank."""
    xmu, xsd = _zfit(Xtr); ymu, ysd = _zfit(Ytr)
    Xtr_, Xte_ = (Xtr - xmu) / xsd, (Xte - xmu) / xsd
    Ytr_, Yte_ = (Ytr - ymu) / ysd, (Yte - ymu) / ysd
    P = Xtr_.shape[1]
    B = np.linalg.solve(Xtr_.T @ Xtr_ + lam * np.eye(P), Xtr_.T @ Ytr_)
    Yhat_tr = Xtr_ @ B
    Uu, Ss, Vt = np.linalg.svd(Yhat_tr - Yhat_tr.mean(0, keepdims=True),
                               full_matrices=False)
    sstot = float((Yte_ ** 2).sum())
    out = []
    for r in ranks:
        Vr = Vt[:r].T
        Br = B @ Vr @ Vr.T
        Yhat_te = Xte_ @ Br
        out.append(1.0 - float(((Yte_ - Yhat_te) ** 2).sum()) / sstot)
    return np.asarray(out)


def _rrr_fit(X, Y, rank, lam):
    """Full-data RRR. Returns per-neuron contributions Vr (N x rank), the
    feature regressor-weights B@Vr (P x rank), and the standardised ridge map
    B (P x N) used for the per-family per-neuron encoding strengths."""
    xmu, xsd = _zfit(X); ymu, ysd = _zfit(Y)
    X_, Y_ = (X - xmu) / xsd, (Y - ymu) / ysd
    P = X_.shape[1]
    B = np.linalg.solve(X_.T @ X_ + lam * np.eye(P), X_.T @ Y_)
    Yhat = X_ @ B
    _, _, Vt = np.linalg.svd(Yhat - Yhat.mean(0, keepdims=True),
                             full_matrices=False)
    Vr = Vt[:rank].T                       # (N, rank)  contributions
    feat = B @ Vr                          # (P, rank)  regressor weights
    return Vr, feat, B


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root",
                    default="/groups/saalfeld/home/allierc/GraphData")
    ap.add_argument("--n_trials", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--out", default=os.path.join(
        HERE, "fig_zebrafish_rrr_feierstein.png"))
    args = ap.parse_args()
    try:
        set_data_root(load_data_root_from_json())
    except FileNotFoundError:
        pass
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    ranks = list(range(1, 9))

    # ---- (a) dimensionality ladder -----------------------------------------
    ladder = []
    for run, lab, has_trans in LADDER:
        net, config = _load_run(
            os.path.join(args.data_root, "log", "zebrafish", run), device)
        ut = load_raw_array(f"{graphs_data_path(config.dataset)}/test/stimulus.zarr")
        yt = load_raw_array(f"{graphs_data_path(config.dataset)}/test/target.zarr")
        ut, yt, _ = _profile(ut, yt, config)
        R, B = _rollout(net, ut, yt, device, args.n_trials, has_trans)
        X, prov = _design(B, R.shape[1], net.dt)
        ntr = R.shape[0]
        sp = max(1, ntr // 3)
        te = np.arange(sp); tr = np.arange(sp, ntr)
        Xtr = X[tr].reshape(-1, X.shape[-1]); Ytr = R[tr].reshape(-1, R.shape[-1])
        Xte = X[te].reshape(-1, X.shape[-1]); Yte = R[te].reshape(-1, R.shape[-1])
        ev = _rrr_popev(Xtr, Ytr, Xte, Yte, ranks, args.lam)
        ladder.append((lab, ev))
        print(f"[ladder] {lab:12s} popEV(rank) = "
              f"{np.array2string(ev, precision=3)}")

    # ---- (b-e) feature/contribution analysis on the 2-D path model ---------
    net, config = _load_run(
        os.path.join(args.data_root, "log", "zebrafish", FEATURE_RUN), device)
    ut = load_raw_array(f"{graphs_data_path(config.dataset)}/test/stimulus.zarr")
    yt = load_raw_array(f"{graphs_data_path(config.dataset)}/test/target.zarr")
    ut, yt, _ = _profile(ut, yt, config)
    R, B = _rollout(net, ut, yt, device, args.n_trials, True)
    X, prov = _design(B, R.shape[1], net.dt)
    Xf = X.reshape(-1, X.shape[-1]); Yf = R.reshape(-1, R.shape[-1])
    # Fit RRR at their reported best rank (~6), then IDENTIFY the translation
    # and rotation features among them by regressor-family loading -- exactly
    # their "feature 1 = vergence, feature 2 = rotation" post-hoc step.
    RANK_FEAT = 6
    Vr, feat = _rrr_fit(Xf, Yf, RANK_FEAT, args.lam)
    pops = _pop_labels(net)

    prov_base = np.array([p[0] for p in prov])
    def _fam_load(col, fam):
        m = np.isin(prov_base, fam)
        denom = np.abs(feat[:, col]).sum() + 1e-12
        return float(np.abs(feat[m, col]).sum() / denom)   # family fraction
    trans_frac = np.array([_fam_load(c, TRANS_BASE) for c in range(RANK_FEAT)])
    rot_frac = np.array([_fam_load(c, ROT_BASE) for c in range(RANK_FEAT)])
    trans_feat = int(np.argmax(trans_frac))
    rot_feat = int(np.argmax(rot_frac))
    if rot_feat == trans_feat:                       # disjoint fallback
        rot_feat = int(np.argmax(np.where(
            np.arange(RANK_FEAT) == trans_feat, -1, rot_frac)))
    print(f"[features] translation feature = #{trans_feat} "
          f"(transl. frac {trans_frac[trans_feat]:.2f}); "
          f"rotation feature = #{rot_feat} "
          f"(rot. frac {rot_frac[rot_feat]:.2f})")
    # orient contribution signs so ARTR_L is negative on the rotation feature
    artr_l = pops == "ARTR_L"
    if artr_l.any() and Vr[artr_l, rot_feat].mean() > 0:
        Vr[:, rot_feat] *= -1.0; feat[:, rot_feat] *= -1.0
    ct = Vr[:, trans_feat]; cr = Vr[:, rot_feat]    # per-neuron contributions

    # selectivity index per population: (|rot|-|trans|)/(|rot|+|trans|)
    print("\n[consistency] feature separation & population selectivity")
    sel = {}
    for pop in POP_ORDER:
        m = pops == pop
        if not m.any():
            continue
        si = (np.abs(cr[m]) - np.abs(ct[m])) / (np.abs(cr[m]) + np.abs(ct[m]) + 1e-9)
        sel[pop] = (float(np.median(si)), int(m.sum()))
        print(f"    {pop:10s} n={m.sum():4d}  rot-vs-trans selectivity "
              f"median={np.median(si):+.2f}  "
              f"<rot>={cr[m].mean():+.4f} <trans>={ct[m].mean():+.4f}")

    # ---- figure ------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9, "lines.linewidth": 1.8,
        "axes.linewidth": 0.9, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    })
    fig = plt.figure(figsize=(13.2, 7.8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34)

    def _pl(ax, s):
        ax.text(-0.17, 1.04, s, transform=ax.transAxes, fontweight="bold",
                fontsize=14)

    # (a) popEV vs rank
    axA = fig.add_subplot(gs[0, 0])
    cols_l = ("#2aa198", "#c9468a", "#6a51a3")
    for (lab, ev), c in zip(ladder, cols_l):
        axA.plot(ranks, ev, marker="o", ms=4, color=c, label=lab)
    axA.axvline(2, color="0.6", ls=":", lw=1.2)
    axA.set_xlabel("RRR rank (number of features)")
    axA.set_ylabel("population explained\nvariance (cross-val.)")
    axA.legend(loc="lower right", frameon=False)
    _pl(axA, "a")

    # (b) feature regressor-weights grouped by family
    axB = fig.add_subplot(gs[0, 1])
    bases = list(ROT_BASE + TRANS_BASE)
    nice = {"omega": r"$\omega$", "cos_th": r"$\cos\theta$",
            "sin_th": r"$\sin\theta$", "v_fwd": r"$v_{fwd}$",
            "swim": "swim", "trans_pos": "displ."}
    w_rot = [np.abs(feat[prov_base == b, rot_feat]).sum() for b in bases]
    w_trn = [np.abs(feat[prov_base == b, trans_feat]).sum() for b in bases]
    xb = np.arange(len(bases))
    axB.bar(xb - 0.2, w_rot, width=0.38, color="#0072b2",
            label="rotation feature")
    axB.bar(xb + 0.2, w_trn, width=0.38, color="#e69f00",
            label="translation feature")
    axB.axvline(2.5, color="0.7", lw=0.8, ls="--")
    axB.set_xticks(xb)
    axB.set_xticklabels([nice[b] for b in bases], rotation=0)
    axB.set_ylabel("|regressor weight| (sum/lags)")
    axB.text(1.0, axB.get_ylim()[1] * 0.92, "rotation", ha="center",
             color="#0072b2", fontsize=9)
    axB.text(4.0, axB.get_ylim()[1] * 0.92, "translation", ha="center",
             color="#e69f00", fontsize=9)
    axB.legend(loc="upper center", frameon=False, fontsize=8)
    _pl(axB, "b")

    # (c) per-neuron contribution scatter coloured by population
    axC = fig.add_subplot(gs[0, 2])
    for pop in POP_ORDER:
        m = pops == pop
        if not m.any():
            continue
        z = 5 if pop.startswith("ARTR") or pop == "ptIPN1" else 2
        axC.scatter(ct[m], cr[m], s=14 if z == 5 else 6,
                    color=POP_COLORS[pop], alpha=0.7 if z == 5 else 0.35,
                    edgecolor="none", zorder=z,
                    label={"ARTR_L": "ARTR L (rot)", "ARTR_R": "ARTR R (rot)",
                           "ptIPN1": "pt-IPN1 (transl.)",
                           "recurrent": "recurrent pool",
                           "other": "other"}[pop])
    axC.axhline(0, color="0.8", lw=0.7); axC.axvline(0, color="0.8", lw=0.7)
    axC.set_xlabel("translation-feature contribution")
    axC.set_ylabel("rotation-feature contribution")
    axC.legend(loc="upper left", frameon=False, fontsize=7.5, handletextpad=0.2)
    _pl(axC, "c")

    # (d) recurrent pool only: contribution density (their mixed-vs-separated)
    axD = fig.add_subplot(gs[1, 0])
    m = pops == "recurrent"
    axD.scatter(ct[m], cr[m], s=7, color="#444444", alpha=0.4, edgecolor="none")
    axD.axhline(0, color="0.8", lw=0.7); axD.axvline(0, color="0.8", lw=0.7)
    axD.set_xlabel("translation-feature contribution")
    axD.set_ylabel("rotation-feature contribution")
    axD.set_title("")  # no title (caption only)
    axD.text(0.04, 0.92, "recurrent pool only", transform=axD.transAxes,
             fontsize=9, color="#444444")
    _pl(axD, "d")

    # (e) selectivity index per population
    axE = fig.add_subplot(gs[1, 1])
    order_e = [p for p in ("ARTR_L", "ARTR_R", "ptIPN1", "recurrent")
               if p in sel]
    vals = [sel[p][0] for p in order_e]
    ns = [sel[p][1] for p in order_e]
    cols_e = [POP_COLORS[p] for p in order_e]
    axE.bar(np.arange(len(order_e)), vals, color=cols_e, width=0.62)
    axE.axhline(0, color="0.5", lw=0.8)
    axE.set_xticks(np.arange(len(order_e)))
    axE.set_xticklabels([{"ARTR_L": "ARTR L", "ARTR_R": "ARTR R",
                          "ptIPN1": "pt-IPN1", "recurrent": "recurrent"}[p]
                         for p in order_e], rotation=20, ha="right")
    axE.set_ylabel("rotation $-$ translation\nselectivity (median)")
    axE.set_ylim(-1.05, 1.05)
    for i, n in enumerate(ns):
        axE.text(i, (vals[i] + 0.08 * np.sign(vals[i] or 1)), f"n={n}",
                 ha="center", va="bottom" if vals[i] >= 0 else "top",
                 fontsize=8)
    _pl(axE, "e")

    # (f) consistency summary text panel
    axF = fig.add_subplot(gs[1, 2]); axF.axis("off")
    ev2d = ladder[-1][1]
    drop = ev2d[1] - ev2d[0]
    lines = [
        r"$\bf{Apples\!-\!to\!-\!apples\ with\ Feierstein\ 2023}$",
        "",
        f"• behaviour subspace ≈ 2-D: popEV jumps",
        f"   to rank 2 then flattens (2-D path:",
        f"   {ev2d[0]:.2f}→{ev2d[1]:.2f}→{ev2d[2]:.2f} at ranks 1,2,3)",
        "• the two features split cleanly into",
        "   a rotation and a translation feature (b)",
        "• afferents form separate translation",
        "   (pt-IPN1) and L/R rotation (ARTR) groups,",
        "   ARTR L vs R on opposite rotation signs (c)",
        "— matches their 3 clusters + hemispheric",
        "   rotation segregation",
        "• BUT the recurrent integrator MIXES both",
        "   (d, e): unlike the afferents it is not",
        "   translation- or rotation-selective",
    ]
    axF.text(0.0, 1.0, "\n".join(lines), transform=axF.transAxes,
             va="top", ha="left", fontsize=9.5, family="DejaVu Sans")
    _pl(axF, "f")

    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig_rrr_feierstein] wrote {args.out}")


if __name__ == "__main__":
    main()
