"""Geometry of the distributed translation code.

Section 3.3 established that the recurrent pool reads out 2-D position near
the information ceiling while no single neuron carries it -- a distributed
code. This script turns "distributed" into an explicit geometric account by
analysing the population firing-rate activity r = sigma(h) of the readout
pool over held-out swim rollouts, for the four joint-target models:

    both          rotation + 1-D forward distance (cumulative)
    both_leaky    rotation + 1-D forward distance (leaky tau)
    position_2d   2-D path integration (cumulative)
    position_2d_leaky   2-D path integration (leaky tau)

It measures, in the SAME neuron basis (the circuit is identical across the
four trained models):

  1. Intrinsic dimensionality of the population activity (PCA participation
     ratio; PCs for 90 % variance).
  2. Targeted dimensionality reduction (TDR): per-neuron linear regression of
     firing rate on the readout targets (cos th, sin th, and x,y or xi),
     giving a heading subspace and a translation subspace in neuron space.
     Geometry = principal angles heading <-> translation, and the x<->y axis
     angle; variance carried along each.
  3. Low-rank sufficiency: linear decode of position from the top-k PCs vs k,
     and the k* at which it saturates -- the true dimensionality of the
     readable place estimate, compared with the model's own readout rank.
  4. Subspace reuse vs rotation as the task moves 1-D distance -> 2-D path:
     principal angles between the translation subspaces across models, and
     whether the 1-D distance axis lies inside the 2-D position plane.

One script = one figure (fig_zebrafish_position_subspace.png) plus a printed
metrics block and an .npz of the numbers for the Results text.

Usage:
    python figures/zebrafish/fig_zebrafish_position_subspace.py --device cuda
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
from scipy.linalg import subspace_angles
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from connectome_gnn.utils import (  # noqa: E402
    load_data_root_from_json, set_data_root, graphs_data_path,
)
from connectome_gnn.models.utils import load_run_config  # noqa: E402
from connectome_gnn.models.registry import create_model  # noqa: E402
from connectome_gnn.zarr_io import load_raw_array  # noqa: E402

# Four joint-target RNN models (same circuit -> same neuron basis).
DEFAULT_RUNS = (
    "zebrafish_hd_si_ipn_917_v1_selfmotion_both",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_both_leaky",
    "zebrafish_hd_si_ipn_917_v1_position_2d",
    "zebrafish_hd_si_ipn_917_v1_position_2d_leaky",
)
LABELS = ("distance", "distance (leaky)", "2-D path", "2-D path (leaky)")
IS_2D = (False, False, True, True)

# input-column / output-column profiles, mirroring fig_zebrafish_mi_partition
_PROFILE_BY_TARGET = {
    (3, ("rotation", "translation")):  ([0, 1, 2, 3], [0, 1, 2]),
    (4, ("position_2d",)):             ([0, 1, 2, 3], [0, 1, 2, 3]),
}
_RECOGNISED = ("rotation", "translation", "position_2d")

COL = {"heading": "#0072b2", "translation": "#e69f00", "full": "0.35"}
MODEL_COL = ("#6a51a3", "#c9468a", "#d29922", "#2aa198")


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


def _rollout_rates(net, u_test, y_test, device, n_trials, readout_dim):
    """Return firing rates r=sigma(h) of the readout pool stacked over trials,
    plus the readout targets. R: (n_trials*T, readout_dim); Z columns are the
    model's readout targets [cos, sin, (x,y) or xi]."""
    Rs, Zs = [], []
    n_trials = min(n_trials, u_test.shape[0])
    for k in range(n_trials):
        with torch.no_grad():
            u_t = torch.from_numpy(u_test[k][None]).to(device)
            _, h_buf = net(u_t)
            r = net._sigma(h_buf)[0, :, :readout_dim].cpu().numpy()
        Rs.append(r.astype(np.float64))
        Zs.append(y_test[k].astype(np.float64))
    return np.concatenate(Rs, 0), np.concatenate(Zs, 0)


def _participation_ratio(R):
    """PR = (sum lam)^2 / sum lam^2 of the centered covariance spectrum."""
    Rc = R - R.mean(0, keepdims=True)
    # covariance eigenvalues via SVD of centered data
    s = np.linalg.svd(Rc, full_matrices=False, compute_uv=False)
    lam = (s ** 2)
    pr = (lam.sum() ** 2) / (lam ** 2).sum()
    cum = np.cumsum(lam) / lam.sum()
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    return float(pr), n90, lam / lam.sum()


def _orthonormal(M):
    """Column-orthonormal basis for the column space of M (P x d)."""
    q, _ = np.linalg.qr(M)
    return q


def _tdr_axes(R, Z, is_2d):
    """Targeted dimensionality reduction. Regress centered firing rates on
    centered, unit-variance readout targets; return the neuron-space axes.

    Returns dict with heading basis (P x 2), translation basis (P x dt),
    raw x/y (or xi) axes, and variance carried along each subspace."""
    Rc = R - R.mean(0, keepdims=True)
    Zc = Z - Z.mean(0, keepdims=True)
    Zc = Zc / (Zc.std(0, keepdims=True) + 1e-9)
    # least-squares per-neuron encoding weights: W (n_var x P)
    W = np.linalg.lstsq(Zc, Rc, rcond=None)[0]
    head = _orthonormal(W[:2].T)                  # cos, sin  -> P x 2
    if is_2d:
        trans = _orthonormal(W[2:4].T)            # x, y      -> P x 2
        ax_a, ax_b = W[2], W[3]
    else:
        trans = _orthonormal(W[2:3].T)            # xi        -> P x 1
        ax_a, ax_b = W[2], None
    total_var = float((Rc ** 2).sum())
    def _vf(basis):
        proj = Rc @ basis
        return float((proj ** 2).sum() / total_var)
    # principal angles heading <-> translation
    pa = np.rad2deg(subspace_angles(head, trans))
    # x <-> y axis angle (raw encoding axes)
    if ax_b is not None:
        cosxy = (ax_a @ ax_b) / (np.linalg.norm(ax_a) * np.linalg.norm(ax_b) + 1e-12)
        xy_angle = float(np.rad2deg(np.arccos(np.clip(abs(cosxy), 0, 1))))
    else:
        xy_angle = float("nan")
    return dict(head=head, trans=trans,
                vf_head=_vf(head), vf_trans=_vf(trans),
                pa_head_trans=pa, xy_angle=xy_angle)


def _decode_vs_rank(R, Z, is_2d, k_list, seed=0):
    """Linear decode of position (or distance) from the top-k variance PCs vs k.

    The optimal full-pool linear readout is itself rank-q (q = #displacement
    targets: 1 for distance, 2 for the 2-D path), so a low-rank readout
    trivially *exists*; ``pos_full`` is that optimal rank-q readout. The point
    of the curve is the MISALIGNMENT with variance: how many top-variance PCs
    the variance-ordered decode needs before it catches the tuned rank-q
    readout. ``r2_pos_topq`` is the variance-matched rank-q decode (top-q PCs);
    contrast with ``pos_full`` (task-aligned rank-q). ``kstar`` is the smallest
    k reaching 95 % of ``pos_full``.
    """
    tr, te = train_test_split(np.arange(R.shape[0]), test_size=0.33,
                              random_state=seed)
    pca = PCA(n_components=int(max(k_list))).fit(R[tr])
    Str, Ste = pca.transform(R[tr]), pca.transform(R[te])
    pos_cols = [2, 3] if is_2d else [2]
    head_cols = [0, 1]
    q = len(pos_cols)
    def _r2(cols, k):
        m = Ridge(alpha=1.0).fit(Str[:, :k], Z[tr][:, cols])
        return float(r2_score(Z[te][:, cols], m.predict(Ste[:, :k]),
                              multioutput="uniform_average"))
    def _r2_full(cols):
        m = Ridge(alpha=1.0).fit(R[tr], Z[tr][:, cols])
        return float(r2_score(Z[te][:, cols], m.predict(R[te]),
                              multioutput="uniform_average"))
    pos = np.array([_r2(pos_cols, k) for k in k_list])
    head = np.array([_r2(head_cols, k) for k in k_list])
    pos_full = _r2_full(pos_cols)
    head_full = _r2_full(head_cols)
    r2_pos_topq = float(pos[list(k_list).index(q)])
    r2_head_topq = float(head[list(k_list).index(q)])
    thr = 0.95 * pos_full
    kstar = int(k_list[int(np.argmax(pos >= thr))]) if np.any(pos >= thr) else None
    return dict(k_list=np.asarray(k_list), pos=pos, head=head,
                pos_full=pos_full, head_full=head_full, kstar=kstar,
                q=q, r2_pos_topq=r2_pos_topq, r2_head_topq=r2_head_topq)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    p.add_argument("--data_root",
                   default="/groups/saalfeld/home/allierc/GraphData")
    p.add_argument("--n_trials", type=int, default=24)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=os.path.join(
        HERE, "fig_zebrafish_position_subspace.png"))
    args = p.parse_args()

    try:
        set_data_root(load_data_root_from_json())
    except FileNotFoundError:
        pass
    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    print(f"device = {device}")

    runs, res = list(args.runs), []
    for j, run in enumerate(runs):
        print(f"[{j+1}/{len(runs)}] {run}")
        run_dir = os.path.join(args.data_root, "log", "zebrafish", run)
        net, config = _load_run(run_dir, device)
        readout_dim = int(net._readout_dim)
        root = graphs_data_path(config.dataset)
        u_test = load_raw_array(f"{root}/test/stimulus.zarr")
        y_test = load_raw_array(f"{root}/test/target.zarr")
        task_key = tuple(t for t in _RECOGNISED
                         if t in (getattr(config.training, "task_targets", None) or []))
        key = (int(y_test.shape[-1]), task_key)
        if key in _PROFILE_BY_TARGET:
            ic, oc = _PROFILE_BY_TARGET[key]
            u_test, y_test = u_test[..., ic], y_test[..., oc]
        is_2d = IS_2D[j]
        R, Z = _rollout_rates(net, u_test, y_test, device, args.n_trials,
                              readout_dim)
        pr, n90, varfrac = _participation_ratio(R)
        tdr = _tdr_axes(R, Z, is_2d)
        kmax = min(200, readout_dim)
        k_list = (list(range(1, 13)) + list(range(14, 41, 3))
                  + list(range(45, kmax + 1, 10)))
        dec = _decode_vs_rank(R, Z, is_2d, k_list)
        res.append(dict(run=run, label=LABELS[j], is_2d=is_2d,
                        readout_dim=readout_dim, pr=pr, n90=n90,
                        varfrac=varfrac, tdr=tdr, dec=dec, R=R, Z=Z))
        pa = tdr["pa_head_trans"]
        print(f"    readout_pool={readout_dim}  PR={pr:.2f}  n90={n90}  "
              f"vf_head={tdr['vf_head']:.3f} vf_trans={tdr['vf_trans']:.3f}  "
              f"pa(head,trans)={np.array2string(pa, precision=1)} deg  "
              f"xy_angle={tdr['xy_angle']:.1f}  "
              f"pos R2 full(rank{dec['q']})={dec['pos_full']:.3f} k*={dec['kstar']}  "
              f"R2 pos top-{dec['q']}PC={dec['r2_pos_topq']:.3f} "
              f"R2 head top-{dec['q']}PC={dec['r2_head_topq']:.3f}")

    # ---- cross-model subspace reuse (same neuron basis) --------------------
    # heading subspace alignment (all 4, each 2-D): mean principal angle
    nM = len(res)
    head_ang = np.full((nM, nM), np.nan)
    for a in range(nM):
        for b in range(nM):
            head_ang[a, b] = np.rad2deg(
                subspace_angles(res[a]["tdr"]["head"],
                                res[b]["tdr"]["head"])).mean()
    # translation reuse: 2-D models -> 2x2 plane angles; 1-D axis vs 2-D plane
    idx2d = [i for i in range(nM) if res[i]["is_2d"]]
    idx1d = [i for i in range(nM) if not res[i]["is_2d"]]
    trans_ang = np.full((nM, nM), np.nan)
    for a in range(nM):
        for b in range(nM):
            pa = subspace_angles(res[a]["tdr"]["trans"], res[b]["tdr"]["trans"])
            trans_ang[a, b] = np.rad2deg(pa).mean()
    print("\n[cross-model] heading-subspace mean principal angle (deg):")
    print(np.array2string(head_ang, precision=1))
    print("[cross-model] translation-subspace mean principal angle (deg):")
    print(np.array2string(trans_ang, precision=1))
    # 1-D distance axis inside 2-D position plane?
    reuse = {}
    for a in idx1d:
        for b in idx2d:
            ang = float(np.rad2deg(subspace_angles(
                res[b]["tdr"]["trans"], res[a]["tdr"]["trans"]))[0])
            reuse[(res[a]["label"], res[b]["label"])] = ang
            print(f"    distance axis '{res[a]['label']}' vs plane "
                  f"'{res[b]['label']}': {ang:.1f} deg")

    # ---- figure ------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9, "lines.linewidth": 1.8,
        "axes.linewidth": 0.9, "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
    })
    fig = plt.figure(figsize=(13.0, 7.6))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.32)

    def _panel_letter(ax, s):
        ax.text(-0.16, 1.03, s, transform=ax.transAxes, fontweight="bold",
                fontsize=14)

    # (a) PCA scree (cumulative variance) for the 4 models
    axA = fig.add_subplot(gs[0, 0])
    for j, r in enumerate(res):
        cv = np.cumsum(r["varfrac"])[:30]
        axA.plot(np.arange(1, cv.size + 1), cv, color=MODEL_COL[j],
                 label=f"{r['label']} (PR={r['pr']:.0f})")
    axA.axhline(0.90, color="0.6", ls=":", lw=1.2)
    axA.set_xlabel("principal components")
    axA.set_ylabel("cumulative variance")
    axA.set_ylim(0, 1.02)
    axA.legend(loc="lower right", frameon=False)
    _panel_letter(axA, "a")

    # (b) heading ring + position sheet for one 2-D model (position_2d)
    ref = res[2]
    Rc = ref["R"] - ref["R"].mean(0, keepdims=True)
    ph = Rc @ ref["tdr"]["head"]      # heading plane coords
    pt = Rc @ ref["tdr"]["trans"]     # translation plane coords
    th = np.arctan2(ref["Z"][:, 1], ref["Z"][:, 0])
    sub = np.random.default_rng(0).choice(ph.shape[0],
                                          min(4000, ph.shape[0]), replace=False)
    axB = fig.add_subplot(gs[0, 1])
    sc = axB.scatter(ph[sub, 0], ph[sub, 1], c=th[sub], cmap="twilight",
                     s=4, alpha=0.6)
    axB.set_xlabel("heading axis 1")
    axB.set_ylabel("heading axis 2")
    axB.set_aspect("equal", "box")
    cb = fig.colorbar(sc, ax=axB, fraction=0.046, pad=0.04)
    cb.set_label("true heading (rad)")
    _panel_letter(axB, "b")

    # (c) position sheet coloured by true x
    axC = fig.add_subplot(gs[0, 2])
    sc2 = axC.scatter(pt[sub, 0], pt[sub, 1], c=ref["Z"][sub, 2],
                      cmap="coolwarm", s=4, alpha=0.6)
    axC.set_xlabel("translation axis 1")
    axC.set_ylabel("translation axis 2")
    axC.set_aspect("equal", "box")
    cb2 = fig.colorbar(sc2, ax=axC, fraction=0.046, pad=0.04)
    cb2.set_label("true x")
    _panel_letter(axC, "c")

    # (d) low-rank sufficiency: position R^2 vs #PCs (2-D models)
    axD = fig.add_subplot(gs[1, 0])
    for j, r in enumerate(res):
        if not r["is_2d"]:
            continue
        d = r["dec"]
        axD.plot(d["k_list"], d["pos"], color=MODEL_COL[j], label=r["label"])
        # optimal task-aligned rank-2 readout (= full-pool linear)
        axD.axhline(d["pos_full"], color=MODEL_COL[j], ls=":", lw=1.0)
        # variance-matched rank-2 readout (top-2 PCs)
        axD.plot(2, d["r2_pos_topq"], marker="*", ms=14, color=MODEL_COL[j],
                 mec="k", mew=0.6, ls="none", zorder=6)
    axD.set_xlabel("number of top-variance principal components")
    axD.set_ylabel(r"position decode $R^2$")
    axD.set_xlim(0, 120)
    axD.set_ylim(-0.05, 1.05)
    handD = [Line2D([0], [0], ls=":", color="0.3",
                    label="tuned rank-2 readout (optimal)"),
             Line2D([0], [0], marker="*", color="0.3", mec="k", ls="none",
                    ms=12, label="variance rank-2 (top-2 PCs)")]
    leg1 = axD.legend(loc="center right", frameon=False)
    axD.add_artist(axD.legend(handles=handD, loc="lower right", frameon=False))
    axD.add_artist(leg1)
    _panel_letter(axD, "d")

    # (e) within-model geometry: head<->trans principal angles + x<->y angle
    axE = fig.add_subplot(gs[1, 1])
    xs = np.arange(nM)
    pa_min = [res[j]["tdr"]["pa_head_trans"].min() for j in range(nM)]
    pa_max = [res[j]["tdr"]["pa_head_trans"].max() for j in range(nM)]
    axE.bar(xs - 0.18, pa_min, width=0.34, color="#7ab7e0",
            label="min angle (head,trans)")
    axE.bar(xs + 0.18, pa_max, width=0.34, color="#0072b2",
            label="max angle (head,trans)")
    for j in range(nM):
        if res[j]["is_2d"]:
            axE.plot(j, res[j]["tdr"]["xy_angle"], marker="D", color="#e69f00",
                     ms=8, ls="none")
    axE.axhline(90, color="0.6", ls=":", lw=1.0)
    axE.set_xticks(xs)
    axE.set_xticklabels([r["label"] for r in res], rotation=30, ha="right")
    axE.set_ylabel("principal angle (deg)")
    axE.set_ylim(0, 100)
    axE.legend(loc="lower left", frameon=False)
    handles = [Line2D([0], [0], marker="D", color="#e69f00", ls="none",
                      label="x vs y axis (2-D)")]
    axE.add_artist(axE.legend(handles=handles, loc="lower right",
                              frameon=False))
    _panel_letter(axE, "e")

    # (f) cross-model translation-subspace alignment matrix
    axF = fig.add_subplot(gs[1, 2])
    im = axF.imshow(trans_ang, cmap="viridis_r", vmin=0, vmax=90)
    axF.set_xticks(xs); axF.set_yticks(xs)
    axF.set_xticklabels([r["label"] for r in res], rotation=30, ha="right")
    axF.set_yticklabels([r["label"] for r in res])
    for a in range(nM):
        for b in range(nM):
            axF.text(b, a, f"{trans_ang[a, b]:.0f}", ha="center", va="center",
                     color="w" if trans_ang[a, b] > 45 else "k", fontsize=9)
    cb3 = fig.colorbar(im, ax=axF, fraction=0.046, pad=0.04)
    cb3.set_label("translation subspace\nmean principal angle (deg)")
    _panel_letter(axF, "f")

    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig_position_subspace] wrote {args.out}")

    # save numbers for the Results text
    np.savez(os.path.splitext(args.out)[0] + "_metrics.npz",
             labels=np.array([r["label"] for r in res]),
             readout_dim=np.array([r["readout_dim"] for r in res]),
             pr=np.array([r["pr"] for r in res]),
             n90=np.array([r["n90"] for r in res]),
             vf_head=np.array([r["tdr"]["vf_head"] for r in res]),
             vf_trans=np.array([r["tdr"]["vf_trans"] for r in res]),
             xy_angle=np.array([r["tdr"]["xy_angle"] for r in res]),
             pos_full=np.array([r["dec"]["pos_full"] for r in res]),
             head_full=np.array([r["dec"]["head_full"] for r in res]),
             kstar=np.array([(r["dec"]["kstar"] or -1) for r in res]),
             r2_pos_topq=np.array([r["dec"]["r2_pos_topq"] for r in res]),
             r2_head_topq=np.array([r["dec"]["r2_head_topq"] for r in res]),
             head_ang=head_ang, trans_ang=trans_ang)
    print("[fig_position_subspace] wrote metrics npz")


if __name__ == "__main__":
    main()
