"""MI-based partition of the recurrent pool, four joint-task models.

Run across four trained joint-task models (rotation+distance and 2-D
path, each cumulative and leaky). For each recurrent neuron i — the
whole GABAergic substrate dIPN + dsIPN + IPN12 + IPN-core
(IPN28/29/31-36); afferents RIPN*/pt-IPN* are excluded by construction —
we compute the per-neuron MI on the integrated heading θ and on the
forward displacement d, then split into four pools:

    angular      :  I(h_i;θ) ≥ τθ, I(h_i;d) < τd
    displacement :  I(h_i;θ) < τθ, I(h_i;d) ≥ τd
    shared       :  I(h_i;θ) ≥ τθ, I(h_i;d) ≥ τd
    neither      :  low on both

The thresholds (τθ, τd) default to the per-axis medians, so each pool
is a quadrant of the MI scatter.

Four rows × four columns (one column per model):
    row 1  MI scatter I(h;θ) vs I(h;d), coloured by functional pool.
    row 2  the SAME scatter, coloured by anatomical family
           (dIPN, dsIPN, IPN12, IPN-core).
    row 3  per-family functional-pool composition (stacked bars).
    row 4  per-cell-type functional-pool composition (stacked bars):
           does the functional split line up with anatomy?

Usage:
    python figures/zebrafish/fig_zebrafish_mi_partition.py
    python figures/zebrafish/fig_zebrafish_mi_partition.py \\
        --runs RUN1 RUN2 RUN3 RUN4 --device cuda
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from connectome_gnn.utils import (  # noqa: E402
    load_data_root_from_json, set_data_root,
)
from connectome_gnn.models.utils import load_run_config  # noqa: E402
from connectome_gnn.models.registry import create_model  # noqa: E402
from connectome_gnn.zarr_io import load_raw_array  # noqa: E402
from connectome_gnn.utils import graphs_data_path  # noqa: E402

# Recurrent-pool cell-type prefixes (everything that's *not* an
# afferent: dIPN, IPNds, IPN12).
RECURRENT_PREFIXES = ("IPNd", "IPNds", "IPN12", "IPN-core")
POOL_ORDER = ("angular", "shared", "displacement", "neither")
# Okabe-Ito-derived qualitative palette (colour-blind safe, prints well).
POOL_COLOR = {
    "angular":      "#0072b2",   # blue          — heading-tuned (bottom-right)
    "displacement": "#e69f00",   # orange        — distance-tuned (top-left)
    "shared":       "#009e73",   # bluish green  — both (top-right)
    "neither":      "#cccccc",   # light grey    — low on both (bottom-left)
}


# Coarse recurrent-substrate families (for the cell-type colouring of
# the bottom row). dsIPN (IPNds*) is split out from dIPN (IPNd*) so the
# three anatomical families of the ring substrate are distinguishable.
REC_FAMILY_ORDER = ("dIPN", "dsIPN", "IPN12", "IPN-core")
REC_FAMILY_COLOR = {
    "dIPN":     "#d29922",   # amber
    "dsIPN":    "#2aa198",   # teal
    "IPN12":    "#c9468a",   # rose
    "IPN-core": "#6a51a3",   # purple
}


def _rec_family(name: str) -> str:
    """Map a recurrent cell-type name to its coarse anatomical family."""
    if name.startswith("IPNds"):
        return "dsIPN"
    if name.startswith("IPN12"):
        return "IPN12"
    if name.startswith("IPNd"):
        return "dIPN"
    return "IPN-core"      # IPN28/29/31-36


def _is_recurrent(name: str) -> bool:
    # The recurrent GABAergic substrate is every IPN* cell type: dIPN
    # (IPNd*), dsIPN (IPNds*), the IPN12 pool, and the IPN-core families
    # (IPN28/29/31-36). The afferents are RIPN* and pt-IPN*, neither of
    # which starts with "IPN", so a single prefix test suffices.
    return name.startswith("IPN")


def _mi_plugin(x: np.ndarray, y: np.ndarray, n_bins_x=32, n_bins_y=20) -> float:
    """Plug-in histogram MI(x; y) in bits. x: continuous scalar over T
    timepoints, y: continuous scalar over T timepoints (same length)."""
    x = np.asarray(x); y = np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]; y = y[finite]
    if x.size < 100:
        return 0.0
    # Bin both axes.
    xb = np.linspace(x.min(), x.max() + 1e-9, n_bins_x + 1)
    yb = np.linspace(y.min(), y.max() + 1e-9, n_bins_y + 1)
    H, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    Pxy = H / H.sum()
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = Pxy * (np.log2(Pxy + 1e-12) - np.log2(Px + 1e-12) - np.log2(Py + 1e-12))
    return float(np.nansum(term))


def _circular_mi(x: np.ndarray, theta: np.ndarray, n_bins_x=32, n_bins_theta=24) -> float:
    """MI(x; θ) where θ is a circular variable in (-π, π]."""
    x = np.asarray(x); theta = np.asarray(theta)
    finite = np.isfinite(x) & np.isfinite(theta)
    x = x[finite]; theta = ((theta[finite] + np.pi) % (2 * np.pi)) - np.pi
    if x.size < 100:
        return 0.0
    xb = np.linspace(x.min(), x.max() + 1e-9, n_bins_x + 1)
    tb = np.linspace(-np.pi, np.pi + 1e-9, n_bins_theta + 1)
    H, _, _ = np.histogram2d(x, theta, bins=[xb, tb])
    Pxy = H / H.sum()
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = Pxy * (np.log2(Pxy + 1e-12) - np.log2(Px + 1e-12) - np.log2(Py + 1e-12))
    return float(np.nansum(term))


# --- whole-pool group MI (decoder lower bound) for the red-cross reference ---
# What ALL recurrent neurons jointly encode about θ and d, as a population
# decoder lower bound H(y)−CE_test in bits. Shown as a red "+" on each scatter
# so the single-cell cloud can be read against the population ceiling. This is
# a DIFFERENT estimator from the per-neuron plug-in MIs (comparable across the
# four columns, but not numerically to the individual dots).
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

N_THETA_BINS = 16
N_D_BINS = 16
MAX_DECODER_SAMPLES = 9000


def _bin_theta_dec(theta):
    th = ((np.asarray(theta) + np.pi) % (2 * np.pi)) - np.pi
    edges = np.linspace(-np.pi, np.pi + 1e-9, N_THETA_BINS + 1)
    return np.clip(np.digitize(th, edges) - 1, 0, N_THETA_BINS - 1)


def _bin_quantile_dec(d, n_bins):
    d = np.asarray(d, dtype=float)
    qs = np.unique(np.quantile(d, np.linspace(0, 1, n_bins + 1)))
    if qs.size < 3:
        return np.zeros(d.size, dtype=int)
    return np.clip(np.digitize(d, qs[1:-1]), 0, qs.size - 2)


def _group_mi_bits(X, ybin):
    """Cross-validated decoder lower bound H(y)−CE_test (bits) for a whole
    population block X (T, n_neurons) predicting the binned target ybin."""
    vals, counts = np.unique(ybin, return_counts=True)
    keep = np.isin(ybin, vals[counts >= 4])
    X, ybin = X[keep], ybin[keep]
    if X.shape[0] < 50 or np.unique(ybin).size < 2:
        return 0.0
    Xtr, Xte, ytr, yte = train_test_split(
        X, ybin, test_size=0.33, random_state=0, stratify=ybin)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=300, C=1.0).fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xte))
    idx = {c: i for i, c in enumerate(clf.classes_)}
    p = np.clip([proba[i, idx[y]] if y in idx else 1e-12
                 for i, y in enumerate(yte)], 1e-12, 1.0)
    ce = float(-np.mean(np.log2(p)))
    v, c = np.unique(yte, return_counts=True)
    py = c / c.sum()
    return max(float(-(py * np.log2(py)).sum()) - ce, 0.0)


def _disp_decode_label(d, pos2d):
    """Binned target for the population displacement decode. For the 2-D
    models, the JOINT (x,y) position on an 8×8 grid — the quantity the rollout
    metric scores — so the population reference matches the trajectory error.
    For the 1-D distance tasks, a 16-bin quantile of the scalar displacement."""
    if pos2d is not None:
        return (_bin_quantile_dec(pos2d[:, 0], 8) * 8
                + _bin_quantile_dec(pos2d[:, 1], 8))
    return _bin_quantile_dec(d, N_D_BINS)


def _load_run(run_dir: str, device):
    config_name = os.path.basename(run_dir)
    config, _ = load_run_config(config_name, explicit_output_root=False,
                                 task="train")
    net = create_model(
        config.graph_model.signal_model_name,
        aggr_type=config.graph_model.aggr_type,
        config=config, device=device,
    )
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


def _run_trial(net, u, device):
    """Forward pass on one (T, n_in) trial. Returns h: (T, N) numpy."""
    with torch.no_grad():
        u_t = torch.from_numpy(u[None]).to(device)
        _, h_buf = net(u_t)
    return h_buf[0].cpu().numpy()


def _accumulate_hidden(net, u_test, y_test, device, n_trials):
    """Concatenate hidden state, heading θ, a scalar displacement d, and (for
    the 2-D models) the joint position ``pos2d`` across n_trials.

    For the 1-D distance tasks (3-col target [cosθ,sinθ,ξ]) d is the supervised
    scalar ξ. For the 2-D path tasks (4-col target [cosθ,sinθ,x,y]) the model
    outputs the bounded leaky position (x,y); the per-neuron displacement
    scalar is the RADIAL distance r=|(x,y)| (the proper 2-D analogue of ξ, and
    a quantity the bounded state actually represents) — NOT the trajectory arc
    length, which is ~monotone in elapsed time and which a leaky integrator
    forgets by design (so MI on it is low even when the model integrates the
    path well). ``pos2d`` returns (x,y) so callers can score the JOINT 2-D
    position — the quantity the rollout metric reports — at the population
    level."""
    hs, thetas, ds, xs, ys = [], [], [], [], []
    n_trials = min(n_trials, u_test.shape[0])
    has_d_target = y_test.shape[-1] >= 3
    is_2d = y_test.shape[-1] == 4
    for k in range(n_trials):
        h = _run_trial(net, u_test[k], device)  # (T, N)
        y = y_test[k]                            # (T, n_out)
        # heading: atan2(sinθ, cosθ) from y[:, :2]
        theta = np.arctan2(y[:, 1], y[:, 0])
        hs.append(h); thetas.append(theta)
        if has_d_target and y.shape[-1] == 3:
            ds.append(y[:, 2])
            xs.append(y[:, 2]); ys.append(np.zeros_like(theta))
        elif is_2d:
            x = y[:, 2]; yp = y[:, 3]
            ds.append(np.sqrt(x ** 2 + yp ** 2))   # radial distance r = |pos|
            xs.append(x); ys.append(yp)
        else:
            ds.append(np.zeros_like(theta))
            xs.append(np.zeros_like(theta)); ys.append(np.zeros_like(theta))
    H = np.concatenate(hs, axis=0)            # (n_trials*T, N)
    theta = np.concatenate(thetas)
    d = np.concatenate(ds)
    pos2d = (np.stack([np.concatenate(xs), np.concatenate(ys)], axis=1)
             if is_2d else None)
    return H, theta, d, pos2d


# The four joint-target models compared across the columns. Each carries
# both a heading target θ and a forward-displacement target (scalar d for
# the "both" task, the 2-D path for "position_2d"), so the I(h;θ) vs I(h;d)
# partition is well defined for all four.
DEFAULT_RUNS = (
    "zebrafish_hd_si_ipn_917_v1_selfmotion_both",
    "zebrafish_hd_si_ipn_917_v1_selfmotion_both_leaky",
    "zebrafish_hd_si_ipn_917_v1_position_2d",
    "zebrafish_hd_si_ipn_917_v1_position_2d_leaky",
)

_PROFILE_BY_TARGET = {
    (3, ("rotation",)):                ([0, 2, 3],    [0, 1]),
    (3, ("translation",)):             ([1],          [2]),
    (3, ("rotation", "translation")):  ([0, 1, 2, 3], [0, 1, 2]),
    (4, ("rotation",)):                ([0, 2, 3],    [0, 1]),
    (4, ("position_2d",)):             ([0, 1, 2, 3], [0, 1, 2, 3]),
}
_RECOGNISED = ("rotation", "translation", "position_2d")


def _compute_run(run_basename: str, data_root: str, n_trials: int, device,
                 tau_theta_fixed: float = 1.5, tau_d_fixed: float = 1.5):
    """Load one trained run, roll out, and return the per-recurrent-neuron
    MI partition plus the cell-type family of each neuron."""
    run_dir = os.path.join(data_root, "log", "zebrafish", run_basename)
    print(f"  loading {run_dir}")
    net, config = _load_run(run_dir, device)
    type_names = list(net.type_names)
    nt = np.asarray(net.neuron_types).astype(int)
    is_rec = np.array([_is_recurrent(type_names[int(t)]) for t in nt])
    rec_ix = np.where(is_rec)[0]
    fam = np.array([_rec_family(type_names[int(nt[i])]) for i in rec_ix],
                   dtype=object)
    ctype = np.array([type_names[int(nt[i])] for i in rec_ix], dtype=object)

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

    H, theta, d, pos2d = _accumulate_hidden(net, u_test, y_test, device, n_trials)
    I_theta = np.zeros(rec_ix.size)
    I_d = np.zeros(rec_ix.size)
    for k, i in enumerate(rec_ix):
        x = H[:, i]
        I_theta[k] = _circular_mi(x, theta)
        I_d[k] = _mi_plugin(x, d)

    # Fixed, SYMMETRIC thresholds (shared across all four models so the
    # columns are directly comparable) rather than per-model medians.
    # τθ=τd=1.5: 1 bit is barely a left/right distinction, so we require a
    # genuine ≥1.5-bit single-cell code on each axis. Only cells above the
    # cutoff are called angular/displacement/shared; the low-MI core falls in
    # "neither" — honest for the harder (2-D, leaky) tasks where the code
    # compresses into a blob with little per-neuron MI.
    tau_theta = float(tau_theta_fixed)
    tau_d = float(tau_d_fixed)
    pool = np.full(rec_ix.size, "neither", dtype=object)
    pool[(I_theta >= tau_theta) & (I_d <  tau_d)] = "angular"
    pool[(I_theta <  tau_theta) & (I_d >= tau_d)] = "displacement"
    pool[(I_theta >= tau_theta) & (I_d >= tau_d)] = "shared"
    counts = {p: int(np.sum(pool == p)) for p in POOL_ORDER}
    fam_counts = {f: int(np.sum(fam == f)) for f in REC_FAMILY_ORDER}

    # Per-group joint MI in the SAME (θ, d) plane as the per-neuron scatter,
    # for the merged group-MI row: one dot per cell type, one per family, and
    # one for the whole pool. Cross-validated decoder lower bound; for the 2-D
    # models the displacement target is the joint (x,y) position. Dot size will
    # be ∝ log(count), so single-cell-type dots (small) sit low and the whole
    # pool (large) sits near the ceiling — the redundancy / pooling story.
    Hrec = H[:, rec_ix]
    Tn = Hrec.shape[0]
    _rng = np.random.default_rng(0)
    sel = (_rng.choice(Tn, MAX_DECODER_SAMPLES, replace=False)
           if Tn > MAX_DECODER_SAMPLES else np.arange(Tn))
    Xs = Hrec[sel]
    yb_theta = _bin_theta_dec(theta)[sel]
    yb_disp = _disp_decode_label(d, pos2d)[sel]

    def _grp(mask):
        Xg = Xs[:, mask]
        return _group_mi_bits(Xg, yb_theta), _group_mi_bits(Xg, yb_disp)

    groups = []
    for ct in sorted(set(ctype)):                        # cell types
        m = (ctype == ct); gt, gd = _grp(m)
        groups.append(dict(name=ct, kind="ctype", fam=_rec_family(ct),
                           n=int(m.sum()), grp_theta=gt, grp_d=gd))
    for f in REC_FAMILY_ORDER:                           # families
        m = (fam == f)
        if not m.any():
            continue
        gt, gd = _grp(m)
        groups.append(dict(name=f, kind="family", fam=f,
                           n=int(m.sum()), grp_theta=gt, grp_d=gd))
    gt_all, gd_all = _grp(np.ones(rec_ix.size, dtype=bool))   # whole pool
    groups.append(dict(name="ALL", kind="total", fam=None,
                       n=int(rec_ix.size), grp_theta=gt_all, grp_d=gd_all))
    grp_theta_all, grp_d_all = gt_all, gd_all
    print(f"    n_rec={rec_ix.size}  τθ={tau_theta:.3f} τd={tau_d:.3f}  "
          f"pools={counts}  group_all θ={gt_all:.2f} d={gd_all:.2f} bits "
          f"({len(groups)} group dots)")
    return dict(I_theta=I_theta, I_d=I_d, pool=pool, counts=counts,
                fam=fam, ctype=ctype, tau_theta=tau_theta, tau_d=tau_d,
                grp_theta_all=grp_theta_all, grp_d_all=grp_d_all,
                groups=groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS),
                   help="four trained run directory basenames (one per column)")
    p.add_argument("--data_root",
                   default="/groups/saalfeld/home/allierc/GraphData")
    p.add_argument("--n_trials", type=int, default=64,
                   help="how many test trials to concatenate for the MI estimate")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=os.path.join(
        HERE, "fig_zebrafish_mi_partition.png"))
    p.add_argument("--tau_theta", type=float, default=1.5,
                   help="fixed I(h;theta) threshold (vertical line), shared across models")
    p.add_argument("--tau_d", type=float, default=1.5,
                   help="fixed I(h;d) threshold (horizontal line), shared across models")
    args = p.parse_args()

    try:
        set_data_root(load_data_root_from_json())
    except FileNotFoundError:
        pass

    device = torch.device(args.device)
    results = []
    for j, run in enumerate(args.runs):
        print(f"[{j + 1}/{len(args.runs)}] {run}")
        results.append(_compute_run(run, args.data_root, args.n_trials, device,
                                     tau_theta_fixed=args.tau_theta,
                                     tau_d_fixed=args.tau_d))

    # ---- plot: 2x2, MI in the (heading, displacement) plane ----------------
    # One panel per task model. Dots: single neurons (per-neuron MI, faint
    # cloud), per-cell-type and whole-pool joint MI, dot area growing with
    # group size — the cloud climbs from single cells (low MI) to the whole
    # pool (near the ceiling): more neurons pooled → more information, and more
    # for the richer (2-D + heading) tasks. Light grey diagonals mark constant
    # total information (heading MI + displacement MI = each integer bit).
    from matplotlib.lines import Line2D
    LF, TF, LET = 13, 11, 14
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.0), sharex=True, sharey=True)
    axes = axes.ravel()
    letters = "abcdefghijklmnopqrstuvwxyz"
    xmax = ymax = 4.0

    # Dot area ∝ count**S_P (S_P<1 keeps single cells visible while still
    # spreading cell type / whole pool clearly apart). Tunable via S_A / S_P.
    S_A, S_P = 14.0, 0.55

    def _dsize(n):
        return S_A * (max(float(n), 1.0) ** S_P)

    for j, r in enumerate(results):
        ax = axes[j]
        row, col = divmod(j, 2)
        # iso-total-information diagonals at every integer bit:
        # heading MI + displacement MI = c
        for c in range(1, 9):
            x0, x1 = max(0.0, c - 4.0), min(4.0, c)
            ax.plot([x0, x1], [c - x0, c - x1], color="0.85", lw=0.7, zorder=0)
            if j == 1 and c < 8:                  # label the diagonals once
                ax.text(x0 + 0.05, (c - x0) - 0.05, f"{c}", color="0.72",
                        fontsize=6.5, rotation=-45, ha="left", va="top",
                        zorder=1)
        # single neurons: faint family-coloured cloud (per-neuron plug-in MI)
        for famk in REC_FAMILY_ORDER:
            m = r["fam"] == famk
            if not m.any():
                continue
            ax.scatter(r["I_theta"][m], r["I_d"][m], s=_dsize(1),
                       color=REC_FAMILY_COLOR[famk], edgecolor="none",
                       alpha=0.38, zorder=2)
        # pooled groups: cell type and whole pool (family dots omitted)
        for g in sorted(r["groups"], key=lambda z: -z["n"]):
            if g["kind"] == "family":
                continue
            dcol = "k" if g["kind"] == "total" else REC_FAMILY_COLOR[g["fam"]]
            ax.scatter(g["grp_theta"], g["grp_d"], s=_dsize(g["n"]),
                       color=dcol, edgecolor="none", alpha=0.80, zorder=5)
        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
        ax.text(-0.04, 1.03, letters[j], transform=ax.transAxes,
                fontsize=LET, fontweight="bold")
        if col == 0:
            ax.set_ylabel("displacement (bits)", fontsize=LF)
        if row == 1:
            ax.set_xlabel("heading (bits)", fontsize=LF)
        ax.tick_params(labelsize=TF, labelleft=(col == 0),
                       labelbottom=(row == 1))
        if j == 0:                                # family-colour key
            fhandles = [Line2D([0], [0], marker="o", ls="none", ms=7,
                               color=REC_FAMILY_COLOR[f], label=f)
                        for f in REC_FAMILY_ORDER]
            ax.legend(handles=fhandles, fontsize=8, loc="upper left",
                      frameon=False)
        if j == len(results) - 1:                 # dot-size key on panel d
            size_ref = [(1, "single cell"), (20, "cell type"), (700, "all rec")]
            shandles = [Line2D([0], [0], marker="o", ls="none", color="0.5",
                               markersize=float(np.sqrt(_dsize(n))), label=lab)
                        for n, lab in size_ref]
            ax.legend(handles=shandles, fontsize=8, loc="upper left",
                      frameon=False, labelspacing=1.5, borderpad=1.0,
                      handletextpad=1.5)

    plt.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_mi_partition] wrote {args.out}")


if __name__ == "__main__":
    main()
