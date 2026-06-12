"""Heading forms a dominant ring manifold, whereas translation emerges as a
distributed population code.

Single figure for the merged Results section on the recurrent code. Six panels
arranged 2 rows x 3 columns, each carrying visual evidence rather than a
summary barplot:

ROW 1 -- the geometry of the code
  (a) the recurrent population projected onto its two heading axes, coloured by
      true heading -> a clean ring attractor; participation ratio + heading
      decode R^2 annotated.
  (b) the same population in a 3-D frame (two heading axes + the translation
      axis orthogonal to the heading plane), coloured by true position: the
      translation axis lifts out of the heading ring -> a distinct population
      direction.
  (c) translation information vs group size (single neurons -> cell types ->
      anatomical families -> whole recurrent pool), cross-validated decoder
      lower bound: single neurons carry ~no 2-D position, the population carries
      it near the ceiling.

ROW 2 -- why the population code is hidden, and how it maps to the imaging
  (d) translation decode R^2 vs number of top-variance principal components
      (curve) against the task-tuned readout the network learns (line):
      important != high variance.
  (e) apples-to-apples with Feierstein 2023: every recurrent/afferent neuron in
      the reduced-rank (heading-feature, translation-feature) loading plane,
      coloured by population -> translation afferents, L/R heading afferents,
      and a mixed recurrent cloud.
  (f) cross-task geometry schematic: the heading ring is reused across tasks
      while the translation subspace rotates from the 1-D distance axis to the
      2-D path plane (measured principal angles annotated).

One script = one figure (fig_zebrafish_distributed_code.png) + a printed
metrics/consistency block.

Usage:
    python figures/zebrafish/fig_zebrafish_distributed_code.py --device cuda
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
from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np
import torch
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from connectome_gnn.utils import (  # noqa: E402
    load_data_root_from_json, set_data_root, graphs_data_path,
)
from connectome_gnn.models.utils import load_run_config  # noqa: E402
from connectome_gnn.models.registry import create_model  # noqa: E402
from connectome_gnn.zarr_io import load_raw_array  # noqa: E402

# ---- models -----------------------------------------------------------------
GEOM_RUNS = (
    ("zebrafish_hd_si_ipn_917_v1_selfmotion_both",       "distance",        False),
    ("zebrafish_hd_si_ipn_917_v1_selfmotion_both_leaky", "distance (leaky)", False),
    ("zebrafish_hd_si_ipn_917_v1_position_2d",           "2-D path",        True),
    ("zebrafish_hd_si_ipn_917_v1_position_2d_leaky",     "2-D path (leaky)", True),
)
GEOM_COL = ("#6a51a3", "#c9468a", "#d29922", "#2aa198")
REF_IX = 2                 # the 2-D path model is the reference for a/b/c/d/e
FEATURE_RUN = "zebrafish_hd_si_ipn_917_v1_position_2d"

LAGS_S = (0.0, 0.1, 0.2, 0.4, 0.8, 1.6)
ROT_BASE = ("omega", "cos_th", "sin_th")
TRANS_BASE = ("v_fwd", "swim", "trans_pos")
POP_COLORS = {"ARTR_L": "#cf222e", "ARTR_R": "#0072b2", "ptIPN1": "#e69f00",
              "recurrent": "#444444", "other": "#bbbbbb"}
POP_ORDER = ("recurrent", "ptIPN1", "ARTR_L", "ARTR_R", "other")


# ---- shared loaders ---------------------------------------------------------
def _load_run(run, data_root, device):
    run_dir = os.path.join(data_root, "log", "zebrafish", run)
    config, _ = load_run_config(run, explicit_output_root=False, task="train")
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


def _profile(u, y, config):
    recognised = ("rotation", "translation", "position_2d")
    task = tuple(t for t in recognised
                 if t in (getattr(config.training, "task_targets", None) or []))
    prof = {(3, ("rotation",)): ([0, 2, 3], [0, 1]),
            (3, ("rotation", "translation")): ([0, 1, 2, 3], [0, 1, 2]),
            (4, ("position_2d",)): ([0, 1, 2, 3], [0, 1, 2, 3])}
    key = (int(y.shape[-1]), task)
    if u.shape[-1] >= 4 and key in prof:
        ic, oc = prof[key]
        return u[..., ic], y[..., oc]
    return u, y


def _is_rec(name):
    return name.startswith("IPN")


def _load_test(config):
    root = graphs_data_path(config.dataset)
    u = load_raw_array(f"{root}/test/stimulus.zarr")
    y = load_raw_array(f"{root}/test/target.zarr")
    return _profile(u, y, config)


# ---- geometry ---------------------------------------------------------------
def _rates_recpool(net, u, y, device, n_trials):
    """Firing rates of the recurrent GABAergic pool + readout targets Z."""
    type_names = list(net.type_names)
    nt = np.asarray(net.neuron_types).astype(int)
    rec_ix = np.where([_is_rec(type_names[int(t)]) for t in nt])[0]
    Rs, Zs = [], []
    n_trials = min(n_trials, u.shape[0])
    for k in range(n_trials):
        with torch.no_grad():
            r = net._sigma(net(torch.from_numpy(u[k][None]).to(device))[1])
            r = r[0].cpu().numpy().astype(np.float64)
        Rs.append(r[:, rec_ix]); Zs.append(y[k].astype(np.float64))
    return np.concatenate(Rs, 0), np.concatenate(Zs, 0)


def _participation_ratio(R):
    Rc = R - R.mean(0, keepdims=True)
    s = np.linalg.svd(Rc, full_matrices=False, compute_uv=False)
    lam = s ** 2
    pr = (lam.sum() ** 2) / (lam ** 2).sum()
    return float(pr), lam / lam.sum()


def _tdr(R, Z, is_2d):
    Rc = R - R.mean(0, keepdims=True)
    Zc = Z - Z.mean(0, keepdims=True)
    Zc = Zc / (Zc.std(0, keepdims=True) + 1e-9)
    W = np.linalg.lstsq(Zc, Rc, rcond=None)[0]
    head, _ = np.linalg.qr(W[:2].T)
    trans, _ = np.linalg.qr(W[2:4].T if is_2d else W[2:3].T)
    pa = np.rad2deg(subspace_angles(head, trans))
    return head, trans, pa


def _heading_r2(R, Z, seed=0):
    """Cross-validated full-rank ridge decode of heading (cosθ, sinθ)."""
    tr, te = train_test_split(np.arange(R.shape[0]), test_size=0.33,
                              random_state=seed)
    m = Ridge(alpha=1.0).fit(R[tr], Z[tr][:, :2])
    return float(r2_score(Z[te][:, :2], m.predict(R[te]),
                          multioutput="uniform_average"))


def _decode_vs_rank(R, Z, is_2d, k_list, seed=0):
    tr, te = train_test_split(np.arange(R.shape[0]), test_size=0.33,
                              random_state=seed)
    pca = PCA(n_components=int(max(k_list))).fit(R[tr])
    Str, Ste = pca.transform(R[tr]), pca.transform(R[te])
    pc = [2, 3] if is_2d else [2]
    q = len(pc)
    def _r2(cols, k):
        m = Ridge(alpha=1.0).fit(Str[:, :k], Z[tr][:, cols])
        return float(r2_score(Z[te][:, cols], m.predict(Ste[:, :k]),
                              multioutput="uniform_average"))
    pos = np.array([_r2(pc, k) for k in k_list])
    m = Ridge(alpha=1.0).fit(R[tr], Z[tr][:, pc])
    pos_full = float(r2_score(Z[te][:, pc], m.predict(R[te]),
                              multioutput="uniform_average"))
    topq = float(pos[list(k_list).index(q)])
    thr = 0.95 * pos_full
    kstar = int(k_list[int(np.argmax(pos >= thr))]) if np.any(pos >= thr) else None
    return dict(k_list=np.asarray(k_list), pos=pos, pos_full=pos_full,
                topq=topq, kstar=kstar, q=q)


# ---- RRR (Feierstein) -------------------------------------------------------
def _pop_labels(net):
    N = len(net.neuron_types)
    type_names = list(net.type_names)
    nt = np.asarray(net.neuron_types).astype(int)
    lab = np.array(["recurrent" if type_names[int(nt[i])].startswith("IPN")
                    else "other" for i in range(N)], dtype=object)
    def _mask(buf):
        b = getattr(net, buf, None)
        return (np.asarray(b.detach().cpu().numpy()) > 0.5 if b is not None
                else np.zeros(N, dtype=bool))
    lab[_mask("_afferent_ind_pt_ipn1_l") | _mask("_afferent_ind_pt_ipn1_r")] = "ptIPN1"
    lab[_mask("_afferent_ind_artr_l")] = "ARTR_L"
    lab[_mask("_afferent_ind_artr_r")] = "ARTR_R"
    return lab


def _rates_all(net, u, y, device, n_trials, has_trans):
    n_trials = min(n_trials, u.shape[0])
    Rs = []
    base = {k: [] for k in (ROT_BASE + (TRANS_BASE if has_trans else ()))}
    for k in range(n_trials):
        with torch.no_grad():
            r = net._sigma(net(torch.from_numpy(u[k][None]).to(device))[1])
            Rs.append(r[0].cpu().numpy().astype(np.float64))
        yy = y[k]
        base["omega"].append(u[k][:, 0].astype(np.float64))
        base["cos_th"].append(yy[:, 0].astype(np.float64))
        base["sin_th"].append(yy[:, 1].astype(np.float64))
        if has_trans:
            vf = u[k][:, 1].astype(np.float64)
            base["v_fwd"].append(vf); base["swim"].append(np.abs(vf))
            tp = (np.sqrt(yy[:, 2] ** 2 + yy[:, 3] ** 2) if yy.shape[-1] == 4
                  else yy[:, 2].astype(np.float64))
            base["trans_pos"].append(tp.astype(np.float64))
    return np.stack(Rs, 0), {k: np.stack(v, 0) for k, v in base.items()}


def _design(B, dt):
    lags = [int(round(s / dt)) for s in LAGS_S]
    cols, prov = [], []
    for nm, sig in B.items():
        for L in lags:
            sh = np.zeros_like(sig)
            if L == 0:
                sh = sig
            else:
                sh[:, L:] = sig[:, :-L]
            cols.append(sh); prov.append((nm, L))
    return np.stack(cols, -1), prov


def _z(A):
    return A.mean(0, keepdims=True), A.std(0, keepdims=True) + 1e-8


def _rrr_fit(X, Y, rank, lam):
    xm, xs = _z(X); ym, ys = _z(Y)
    X_, Y_ = (X - xm) / xs, (Y - ym) / ys
    P = X_.shape[1]
    B = np.linalg.solve(X_.T @ X_ + lam * np.eye(P), X_.T @ Y_)
    Yh = X_ @ B
    _, _, Vt = np.linalg.svd(Yh - Yh.mean(0, keepdims=True), full_matrices=False)
    Vr = Vt[:rank].T
    return Vr, B @ Vr, B


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root",
                    default="/groups/saalfeld/home/allierc/GraphData")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--out", default=os.path.join(
        HERE, "fig_zebrafish_distributed_code.png"))
    args = ap.parse_args()
    try:
        set_data_root(load_data_root_from_json())
    except FileNotFoundError:
        pass
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # ========================= geometry =====================================
    geom = []
    for run, lab, is_2d in GEOM_RUNS:
        net, cfg = _load_run(run, args.data_root, device)
        u, y = _load_test(cfg)
        R, Z = _rates_recpool(net, u, y, device, args.n_trials)
        pr, varfrac = _participation_ratio(R)
        head, trans, pa = _tdr(R, Z, is_2d)
        kmax = min(140, R.shape[1])
        k_list = list(range(1, 13)) + list(range(14, 41, 3)) + list(range(45, kmax + 1, 10))
        dec = _decode_vs_rank(R, Z, is_2d, k_list)
        hr2 = _heading_r2(R, Z)
        geom.append(dict(run=run, lab=lab, is_2d=is_2d, pr=pr, varfrac=varfrac,
                         head=head, trans=trans, pa=pa, dec=dec, hr2=hr2,
                         R=R, Z=Z))
        print(f"[geom] {lab:16s} PR={pr:.2f} pa(head,trans)="
              f"{np.array2string(pa, precision=0)} deg  head R2={hr2:.3f}"
              f"  pos R2_full={dec['pos_full']:.3f} topq={dec['topq']:.3f} k*={dec['kstar']}")
    # cross-task subspace alignment (same neuron basis)
    nG = len(geom)
    head_ang = np.array([[np.rad2deg(subspace_angles(geom[a]["head"], geom[b]["head"])).mean()
                          for b in range(nG)] for a in range(nG)])
    trans_ang = np.array([[np.rad2deg(subspace_angles(geom[a]["trans"], geom[b]["trans"])).mean()
                           for b in range(nG)] for a in range(nG)])
    iu = np.triu_indices(nG, k=1)
    head_reuse = float(head_ang[iu].mean())
    # 1-D distance models {0,1} vs 2-D path models {2,3}
    trans_rot = float(np.mean([trans_ang[a, b] for a in (0, 1) for b in (2, 3)]))
    print("[geom] heading cross-task mean angle (deg):\n", np.array2string(head_ang, precision=0))
    print("[geom] translation cross-task mean angle (deg):\n", np.array2string(trans_ang, precision=0))
    print(f"[geom] heading reused {head_reuse:.0f} deg, translation rotated {trans_rot:.0f} deg")

    # ================ panel C: translation MI vs group size =================
    # cross-validated decoder lower bound, one estimator across every scale
    # (single neurons -> cell types -> families -> whole pool), 2-D model.
    from fig_zebrafish_mi_partition import (  # noqa: E402
        _accumulate_hidden, _is_recurrent, _rec_family,
        REC_FAMILY_ORDER, REC_FAMILY_COLOR,
    )
    from fig_zebrafish_mi_grouped import _decoder_mi_bits, _bin_quantile  # noqa: E402

    net_f, cfg_f = _load_run(FEATURE_RUN, args.data_root, device)
    u_f, y_f = _load_test(cfg_f)
    H, theta, d, pos2d = _accumulate_hidden(net_f, u_f, y_f, device, args.n_trials)
    tn = list(net_f.type_names)
    nt_f = np.asarray(net_f.neuron_types).astype(int)
    rec_mask = np.array([_is_recurrent(tn[int(t)]) for t in nt_f])
    rec_ix = np.where(rec_mask)[0]
    ctype = np.array([tn[int(nt_f[i])] for i in rec_ix], dtype=object)
    fam = np.array([_rec_family(tn[int(nt_f[i])]) for i in rec_ix], dtype=object)
    Hrec = H[:, rec_ix]
    rngC = np.random.default_rng(0)
    T = Hrec.shape[0]
    sel = (rngC.choice(T, 9000, replace=False) if T > 9000 else np.arange(T))
    Xs = Hrec[sel]
    lx, _ = _bin_quantile(pos2d[sel, 0], 8)
    ly, _ = _bin_quantile(pos2d[sel, 1], 8)
    yb_pos = lx * 8 + ly                       # joint (x,y) on an 8x8 grid

    micurve = {"single": [], "type": [], "family": [], "all": None}
    single_ix = rngC.choice(Xs.shape[1], min(40, Xs.shape[1]), replace=False)
    for j in single_ix:
        micurve["single"].append((1, _decoder_mi_bits(Xs[:, [j]], yb_pos, rngC),
                                  fam[j]))
    for ct in sorted(set(ctype)):
        m = ctype == ct
        micurve["type"].append((int(m.sum()),
                                _decoder_mi_bits(Xs[:, m], yb_pos, rngC),
                                _rec_family(ct)))
    for g in REC_FAMILY_ORDER:
        m = fam == g
        if m.any():
            micurve["family"].append((int(m.sum()),
                                      _decoder_mi_bits(Xs[:, m], yb_pos, rngC), g))
    micurve["all"] = (Xs.shape[1], _decoder_mi_bits(Xs, yb_pos, rngC))
    print(f"[mi] single-neuron median translation MI = "
          f"{np.median([v for _, v, _ in micurve['single']]):.3f} bits; "
          f"whole pool = {micurve['all'][1]:.3f} bits")

    # ===================== panel E: RRR loading plane ========================
    R_all, B_all = _rates_all(net_f, u_f, y_f, device, args.n_trials, True)
    X, prov = _design(B_all, net_f.dt)
    Vr, feat, Bmap = _rrr_fit(X.reshape(-1, X.shape[-1]),
                              R_all.reshape(-1, R_all.shape[-1]), 6, args.lam)
    pops = _pop_labels(net_f)
    prov_base = np.array([p[0] for p in prov])
    def _frac(col, family):
        m = np.isin(prov_base, family)
        return float(np.abs(feat[m, col]).sum() / (np.abs(feat[:, col]).sum() + 1e-12))
    tfrac = np.array([_frac(c, TRANS_BASE) for c in range(6)])
    rfrac = np.array([_frac(c, ROT_BASE) for c in range(6)])
    trans_feat, rot_feat = int(np.argmax(tfrac)), int(np.argmax(rfrac))
    if rot_feat == trans_feat:
        rot_feat = int(np.argmax(np.where(np.arange(6) == trans_feat, -1, rfrac)))
    # per-neuron loadings on the heading and translation features
    head_load = Vr[:, rot_feat].copy()
    trans_load = Vr[:, trans_feat].copy()
    # sign-fix so the plane reads like Feierstein: translation afferent on the
    # +translation axis, ARTR L/R split along +/- heading.
    if np.median(trans_load[pops == "ptIPN1"] if (pops == "ptIPN1").any()
                 else trans_load) < 0:
        trans_load = -trans_load
    dL = (np.mean(head_load[pops == "ARTR_L"]) if (pops == "ARTR_L").any() else 0.0)
    dR = (np.mean(head_load[pops == "ARTR_R"]) if (pops == "ARTR_R").any() else 0.0)
    if dL - dR < 0:
        head_load = -head_load
    print(f"[rrr] heading feature #{rot_feat} (rot frac {rfrac[rot_feat]:.2f}), "
          f"translation feature #{trans_feat} (trans frac {tfrac[trans_feat]:.2f})")

    # ============================ figure =====================================
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 11.5, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5, "legend.fontsize": 8.5, "lines.linewidth": 1.7,
        "axes.linewidth": 0.9, "xtick.major.size": 3.2, "ytick.major.size": 3.2,
    })
    fig = plt.figure(figsize=(14.5, 9.2))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.34)

    def _pl(ax, s, dx=-0.18):
        ax.text(dx, 1.06, s, transform=ax.transAxes, fontweight="bold",
                fontsize=15)

    ref = geom[REF_IX]

    # ---- (a) heading ring -------------------------------------------------
    Rc = ref["R"] - ref["R"].mean(0, keepdims=True)
    ph = Rc @ ref["head"]
    th = np.arctan2(ref["Z"][:, 1], ref["Z"][:, 0])
    rng = np.random.default_rng(0)
    sub = rng.choice(ph.shape[0], min(4000, ph.shape[0]), replace=False)
    axA = fig.add_subplot(gs[0, 0])
    scb = axA.scatter(ph[sub, 0], ph[sub, 1], c=th[sub], cmap="twilight",
                      s=5, alpha=0.6, edgecolor="none")
    axA.set_xlabel("heading axis 1"); axA.set_ylabel("heading axis 2")
    axA.set_aspect("equal", "box")
    cb = fig.colorbar(scb, ax=axA, fraction=0.046, pad=0.04)
    cb.set_label("true heading", fontsize=9)
    axA.text(0.5, 0.5, f"ring attractor\nPR = {ref['pr']:.0f}\n"
             f"heading decode\n$R^2$ = {ref['hr2']:.2f}",
             transform=axA.transAxes, fontsize=8.5, va="center", ha="center")
    _pl(axA, "a")

    # ---- (b) 3-D manifold: translation axis lifts out of the heading ring -
    axB = fig.add_subplot(gs[0, 1], projection="3d")
    e1, e2 = ref["head"][:, 0], ref["head"][:, 1]
    t = ref["trans"][:, 0]
    t_perp = t - (t @ e1) * e1 - (t @ e2) * e2
    e3 = t_perp / (np.linalg.norm(t_perp) + 1e-12)
    frame = np.stack([e1, e2, e3], axis=1)
    P3 = Rc @ frame
    radial = np.sqrt(ref["Z"][:, 2] ** 2 + ref["Z"][:, 3] ** 2)
    sub3 = rng.choice(P3.shape[0], min(3500, P3.shape[0]), replace=False)
    scB = axB.scatter(P3[sub3, 0], P3[sub3, 1], P3[sub3, 2], c=radial[sub3],
                      cmap="viridis", s=5, alpha=0.55, edgecolor="none")
    s12 = float(np.percentile(np.abs(P3[:, :2]), 95))
    s3 = float(np.percentile(np.abs(P3[:, 2]), 95)) + 1e-6
    ang = np.linspace(0, 2 * np.pi, 60)
    axB.plot(s12 * np.cos(ang), s12 * np.sin(ang), np.zeros_like(ang),
             color="0.4", lw=1.0, alpha=0.6)
    tvec = np.array([t @ e1, t @ e2, t @ e3]); tvec = tvec / np.linalg.norm(tvec)
    L = 1.15 * max(s12, s3)
    axB.quiver(0, 0, 0, tvec[0] * L, tvec[1] * L, tvec[2] * L,
               color="#e69f00", lw=2.6, arrow_length_ratio=0.12)
    axB.quiver(0, 0, 0, -tvec[0] * L, -tvec[1] * L, -tvec[2] * L,
               color="#e69f00", lw=2.6, arrow_length_ratio=0.0, alpha=0.5)
    axB.text2D(0.02, 0.90, f"heading $\\perp$ translation: "
               f"{ref['pa'].min():.0f}–{ref['pa'].max():.0f}$^\\circ$",
               transform=axB.transAxes, fontsize=9, color="#b07400")
    axB.set_xlabel("heading axis 1", labelpad=-6)
    axB.set_ylabel("heading axis 2", labelpad=-6)
    axB.set_zlabel("translation axis", labelpad=-6)
    axB.set_xticklabels([]); axB.set_yticklabels([]); axB.set_zticklabels([])
    axB.view_init(elev=18, azim=-58)
    axB.set_box_aspect((1, 1, 0.75))
    cbB = fig.colorbar(scB, ax=axB, fraction=0.030, pad=0.02)
    cbB.set_label("true distance", fontsize=9)
    axB.text2D(-0.06, 1.02, "b", transform=axB.transAxes, fontweight="bold",
               fontsize=15)

    # ---- (c) translation MI vs group size ---------------------------------
    axC = fig.add_subplot(gs[0, 2])
    sx = np.array([n for n, _, _ in micurve["single"]], dtype=float)
    sv = np.array([v for _, v, _ in micurve["single"]])
    axC.scatter(sx * rng.uniform(0.85, 1.15, sx.size), sv, s=14, color="0.55",
                alpha=0.7, edgecolor="none", label="single neurons", zorder=2)
    for n, v, f in micurve["type"]:
        axC.scatter(n, v, s=26, color=REC_FAMILY_COLOR[f], alpha=0.85,
                    edgecolor="none", zorder=3)
    for n, v, f in micurve["family"]:
        axC.scatter(n, v, s=130, color=REC_FAMILY_COLOR[f], marker="D",
                    edgecolor="k", linewidth=0.8, zorder=5)
    axC.plot(micurve["all"][0], micurve["all"][1], marker="+", color="red",
             ms=15, mew=2.4, ls="none", zorder=8)
    # log trend through every group (excluding the whole-pool point)
    allpts = ([(n, v) for n, v, _ in micurve["single"]]
              + [(n, v) for n, v, _ in micurve["type"]]
              + [(n, v) for n, v, _ in micurve["family"]])
    apx = np.array([p[0] for p in allpts]); apy = np.array([p[1] for p in allpts])
    good = apx > 0
    bb, aa = np.polyfit(np.log10(apx[good]), apy[good], 1)
    xf = np.logspace(0, np.log10(micurve["all"][0]), 60)
    axC.plot(xf, aa + bb * np.log10(xf), color="0.35", ls="--", lw=1.1, zorder=1)
    axC.set_xscale("log")
    axC.set_xlabel("group size (neurons)")
    axC.set_ylabel("translation information (bits)")
    axC.set_ylim(-0.1, max(3.6, micurve["all"][1] * 1.1))
    hC = [Line2D([0], [0], marker="o", ls="none", ms=6, color="0.55",
                 label="single neurons"),
          Line2D([0], [0], marker="o", ls="none", ms=6, color="0.5",
                 label="cell types"),
          Line2D([0], [0], marker="D", ls="none", ms=8, mec="k", color="0.5",
                 label="families"),
          Line2D([0], [0], marker="+", ls="none", ms=11, mew=2, color="red",
                 label="whole pool")]
    axC.legend(handles=hC, loc="upper left", frameon=False, fontsize=8)
    _pl(axC, "c")

    # ---- (d) decode R^2 vs #PCs  vs  task readout -------------------------
    axD = fig.add_subplot(gs[1, 0])
    for j, g in enumerate(geom):
        if not g["is_2d"]:
            continue
        dd = g["dec"]
        prim = (j == REF_IX)
        axD.plot(dd["k_list"], dd["pos"], color=GEOM_COL[j],
                 lw=2.0 if prim else 1.3, alpha=1.0 if prim else 0.55,
                 label="top-variance PCs" if prim else None, zorder=3 if prim else 2)
        axD.plot(2, dd["topq"], marker="*", ms=14, color=GEOM_COL[j], mec="k",
                 mew=0.5, zorder=6)
    dref = ref["dec"]
    axD.axhline(dref["pos_full"], color="0.25", ls=":", lw=1.4)
    axD.annotate(f"task-tuned readout: $R^2$ = {dref['pos_full']:.2f}",
                 xy=(axD.get_xlim()[1] * 0.55, dref["pos_full"]),
                 xytext=(0, -14), textcoords="offset points", fontsize=8.5,
                 ha="center", va="top", color="0.25")
    axD.annotate(f"top 2 PCs: $R^2$ = {dref['topq']:.2f}", xy=(2, dref["topq"]),
                 xytext=(16, 26), textcoords="offset points", fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0))
    axD.set_xlabel("number of top-variance PCs")
    axD.set_ylabel(r"translation decode $R^2$")
    axD.set_xlim(0, 90); axD.set_ylim(-0.05, 1.05)
    _pl(axD, "d")

    # ---- (e) Feierstein loading plane -------------------------------------
    axE = fig.add_subplot(gs[1, 1])
    for pop in POP_ORDER:
        m = pops == pop
        if not m.any():
            continue
        big = pop.startswith("ARTR") or pop == "ptIPN1"
        axE.scatter(head_load[m], trans_load[m], s=20 if big else 7,
                    color=POP_COLORS[pop], alpha=0.85 if big else 0.3,
                    edgecolor="none", zorder=5 if big else 2,
                    label={"ARTR_L": "ARTR L (heading)", "ARTR_R": "ARTR R (heading)",
                           "ptIPN1": "pt-IPN1 (translation)", "recurrent": "recurrent",
                           "other": "other"}[pop])
    axE.axhline(0, color="0.85", lw=0.7); axE.axvline(0, color="0.85", lw=0.7)
    axE.set_xlabel("heading feature loading")
    axE.set_ylabel("translation feature loading")
    axE.legend(loc="upper right", frameon=False, fontsize=7.3, handletextpad=0.2)
    _pl(axE, "e")

    # ---- (f) cross-task geometry schematic --------------------------------
    axF = fig.add_subplot(gs[1, 2])
    axF.set_xlim(0, 1); axF.set_ylim(0, 1); axF.axis("off")
    blue, orange = "#0072b2", "#e69f00"
    for cx, title in ((0.27, "1-D distance"), (0.73, "2-D path")):
        # heading ring (same in both -> reused)
        axF.add_patch(Ellipse((cx, 0.45), 0.30, 0.13, fill=False,
                              edgecolor=blue, lw=2.2))
        axF.text(cx, 0.27, title, ha="center", fontsize=9.5)
    axF.text(0.27, 0.36, "heading ring", ha="center", color=blue, fontsize=8.5)
    # translation: a single axis on the left, a rotated plane on the right
    axF.add_patch(FancyArrowPatch((0.27, 0.45), (0.27, 0.80), color=orange,
                                  lw=2.4, arrowstyle="-|>", mutation_scale=14))
    axF.add_patch(FancyArrowPatch((0.73, 0.45), (0.66, 0.80), color=orange,
                                  lw=2.4, arrowstyle="-|>", mutation_scale=14))
    axF.add_patch(FancyArrowPatch((0.73, 0.45), (0.82, 0.78), color=orange,
                                  lw=2.4, arrowstyle="-|>", mutation_scale=14,
                                  alpha=0.7))
    axF.text(0.27, 0.85, "translation\naxis", ha="center", color="#b07400",
             fontsize=8.5)
    axF.text(0.74, 0.86, "translation\nplane", ha="center", color="#b07400",
             fontsize=8.5)
    # measured angles
    axF.annotate("", xy=(0.42, 0.45), xytext=(0.58, 0.45),
                 arrowprops=dict(arrowstyle="<->", color=blue, lw=1.3))
    axF.text(0.5, 0.49, f"heading reused\n{head_reuse:.0f}$^\\circ$", ha="center",
             color=blue, fontsize=8.5)
    axF.text(0.5, 0.74, f"translation\nrotates {trans_rot:.0f}$^\\circ$",
             ha="center", color="#b07400", fontsize=8.5)
    _pl(axF, "f", dx=-0.05)

    from _despine import open_axes
    open_axes(fig)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig_distributed_code] wrote {args.out}")


if __name__ == "__main__":
    main()
