"""Latent path-integration probe: does a HEADING-ONLY model build a translation
representation it was never asked for?

Three models share the same 917-cell circuit and the same 4-channel input
([omega, v_fwd, cos0, sin0]) and differ ONLY in what they are supervised on:

    1. rotation_vfwd  -- heading only      (v_fwd is in the input but never a target)
    2. both           -- heading + 1-D distance
    3. position_2d    -- heading + 2-D position

All three are evaluated on the SAME held-out position_2d test rollouts, so the
ground-truth v_fwd / distance / (x, y) are identical across models and the only
difference is supervision. From each model's recurrent hidden state we decode
(cross-validated ridge, split by trial):

    heading (cos, sin)         -- positive control, every model should nail it
    instantaneous v_fwd        -- trivially present (it is an input)
    integrated distance d=int v_fwd dt
    2-D position (x, y)        -- the discriminating test

Interpretation of the (x, y) decode from model 1:
    inherited from the input  -> v_fwd high, distance/position low
    induced by the task       -> position only high for models 2/3
    spontaneously generated   -> position high even for model 1 (surprising)

Panel (b) further asks WHERE position lives: position decode R^2 vs the number of
top-variance PCs (solid) against the tuned full-rank readout (dotted) and the
variance-matched top-2 PCs (star) -- a low-variance, task-tuned slice if the
solid curve needs many PCs while the full readout is high.

One script = one figure (fig_zebrafish_latent_translation.png) + a printed
metrics table.

Usage:
    /home/allierc@hhmi.org/miniforge3/envs/neural-graph-linux/bin/python \\
        figures/zebrafish/fig_zebrafish_latent_translation.py --device cuda
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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

# Models compared. (run_basename, label, colour). The measured heading-only
# probe goes first; the ER (randomized-connectome) heading-only control second
# (Test C); then the supervised references. The measured-vs-ER heading-only
# contrast is the key one: if latent position survives only on the measured
# connectome, it is a property of the wiring, not generic recurrence.
MODELS = (
    ("zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd",    "measured heading only",          "#6a51a3"),
    ("zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd_er", "ER heading only",                "#999999"),
    ("zebrafish_hd_si_ipn_917_v1_selfmotion_both",             "measured heading + distance",    "#c9468a"),
    ("zebrafish_hd_si_ipn_917_v1_position_2d",                 "measured heading + 2-D position", "#2aa198"),
)
# Common evaluation set: the position_2d test rollouts carry the full GT
# (omega, v_fwd, cos, sin, x, y), and all three models take the same 4-ch input.
EVAL_DATASET = "zebrafish_hd_si_task_917_position_2d"
# Decode targets, in plotting order.
DECODE_KEYS = ("heading", "v_fwd", "distance", "position")
DECODE_LABEL = {"heading": "heading\n(cos,sin)", "v_fwd": "v_fwd\n(input)",
                "distance": r"distance $\int v\,dt$", "position": "position\n(x,y)"}


def _is_rec(name):
    return name.startswith("IPN")


def _load_run(run, device):
    """Load a trained model + its config. Returns (net, config) or (None, None)
    if the config or checkpoint does not exist yet (graceful degradation)."""
    try:
        config, _ = load_run_config(run, explicit_output_root=False, task="train")
    except Exception as e:
        print(f"  [skip] no config for {run} ({type(e).__name__})")
        return None, None
    data_root = "/groups/saalfeld/home/allierc/GraphData"
    run_dir = os.path.join(data_root, "log", "zebrafish", run)
    ckpts = sorted(glob.glob(os.path.join(run_dir, "models",
                                          "best_model_with_*.pt")))
    if not ckpts:
        print(f"  [skip] no checkpoint in {run_dir} (train it first)")
        return None, None
    net = create_model(config.graph_model.signal_model_name,
                       aggr_type=config.graph_model.aggr_type,
                       config=config, device=device)
    sd = torch.load(ckpts[-1], map_location=device, weights_only=False)
    sd = sd if isinstance(sd, dict) else sd.state_dict()
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net, config


def _eval_root():
    """Resolve the eval dataset dir, with or without the biomodel subdir."""
    for parts in (("zebrafish", EVAL_DATASET), (EVAL_DATASET,)):
        root = graphs_data_path(*parts)
        if os.path.exists(os.path.join(root, "test", "stimulus.zarr")):
            return root
    raise FileNotFoundError(
        f"could not find {EVAL_DATASET}/test/stimulus.zarr under "
        f"{graphs_data_path('zebrafish', EVAL_DATASET)} or "
        f"{graphs_data_path(EVAL_DATASET)}")


def _load_eval_set(n_trials):
    """Raw position_2d test rollouts: u (N,T,4)=[omega,v_fwd,cos0,sin0],
    y (N,T,4)=[cos,sin,x,y]."""
    root = _eval_root()
    u = load_raw_array(f"{root}/test/stimulus.zarr").astype(np.float64)
    y = load_raw_array(f"{root}/test/target.zarr").astype(np.float64)
    n = min(n_trials, u.shape[0])
    return u[:n], y[:n]


def _gt_targets(u, y, dt):
    """Per-(trial,time) ground-truth decode targets from the common eval set."""
    cos, sin = y[..., 0], y[..., 1]
    v_fwd = u[..., 1]
    x, yy = y[..., 2], y[..., 3]
    distance = np.cumsum(v_fwd, axis=1) * dt          # integral of v_fwd dt
    return {
        "heading":  np.stack([cos, sin], axis=-1),    # (N,T,2)
        "v_fwd":    v_fwd[..., None],                  # (N,T,1)
        "distance": distance[..., None],              # (N,T,1)
        "position": np.stack([x, yy], axis=-1),       # (N,T,2)
    }


def _rec_ix(net):
    """Indices of the recurrent GABAergic pool (every IPN* cell type)."""
    type_names = list(net.type_names)
    nt = np.asarray(net.neuron_types).astype(int)
    return np.where([_is_rec(type_names[int(t)]) for t in nt])[0]


def _rates(net, u, device):
    """Recurrent-pool firing rates over all eval trials.
    Returns R (N*T, n_rec), trial_id (N*T,)."""
    rec_ix = _rec_ix(net)
    n_in = int(net.n_input)
    Rs, tid = [], []
    for k in range(u.shape[0]):
        uk = u[k, :, :n_in][None].astype(np.float32)  # all 3 models use cols 0..3
        with torch.no_grad():
            r = net._sigma(net(torch.from_numpy(uk).to(device))[1])
        r = r[0].cpu().numpy().astype(np.float64)[:, rec_ix]
        Rs.append(r)
        tid.append(np.full(r.shape[0], k))
    return np.concatenate(Rs, 0), np.concatenate(tid, 0)


def _leaky_filter(v, dt, tau):
    """Leaky integral of v (shape (N,T)) with time constant tau, per trial:
    d[t] = (1 - dt/tau) d[t-1] + v[t] dt. tau=inf -> cumulative integral."""
    if not np.isfinite(tau):
        return np.cumsum(v, axis=1) * dt
    a = 1.0 - dt / tau
    out = np.zeros_like(v)
    acc = np.zeros(v.shape[0])
    for t in range(v.shape[1]):
        acc = a * acc + v[:, t] * dt
        out[:, t] = acc
    return out


def _follow_time(decoded, true, dt, rel=0.20, floor=0.5, warmup=10):
    """Time (s) the decoded vector stays within a relative tolerance of the
    true vector, evaluated once the true magnitude clears ``floor`` (so the
    near-origin start does not trip the relative test). Capped at the rollout
    length if it never falls behind."""
    mag = np.linalg.norm(true, axis=-1)
    err = np.linalg.norm(decoded - true, axis=-1)
    bad = (mag > floor) & (err > rel * mag)
    bad[:warmup] = False
    if bad.any():
        return float(np.argmax(bad) * dt)
    return float(len(mag) * dt)


def _ou_rollout(net, rec, device, n_steps, dt, seed):
    """Long NATURALISTIC Ornstein--Uhlenbeck rollout, matching the statistics of
    the precision-horizon probe (in-distribution, unlike a constant drive).
    Returns recurrent firing rates (T, n_rec) and the true 2-D position (T, 2)."""
    import math
    T = int(n_steps)
    rng = np.random.default_rng(int(seed))

    def _ou(tau, sd, mu, lo, hi):
        a = math.exp(-dt / tau)
        x = np.empty(T, np.float32); x[0] = mu
        for t in range(1, T):
            x[t] = mu + a * (x[t - 1] - mu) + math.sqrt(1 - a * a) * rng.normal(0, sd)
        return np.clip(x, lo, hi)

    om = _ou(1.0, 55.0, 0.0, -150, 150)        # angular velocity (deg/s)
    vf = _ou(1.5, 0.4, 1.0, 0, 3)              # forward velocity
    n_in = int(net.n_input)
    u = np.zeros((1, T, n_in), np.float32)
    u[0, :, 0] = om
    if n_in >= 4:
        u[0, :, 1] = vf
        u[0, 0, 2] = 1.0                       # cos(theta0) cue at t=0
    elif n_in == 3:
        u[0, 0, 1] = 1.0
    with torch.no_grad():
        r = net._sigma(net(torch.from_numpy(u).to(device))[1])
    rr = r[0].cpu().numpy().astype(np.float64)[:, rec]
    th = np.cumsum(np.deg2rad(om)) * dt
    xt = np.cumsum(vf * np.cos(th)) * dt
    yt = np.cumsum(vf * np.sin(th)) * dt
    return rr, np.stack([xt, yt], axis=-1)


def _split(n_samples, test_frac=0.33, seed=0):
    """Random timepoint split (instantaneous-readability question; matches the
    distributed-code analysis). Returns train/test index arrays."""
    return train_test_split(np.arange(n_samples), test_size=test_frac,
                            random_state=seed)


def _decode_r2(X, Y, tr, te, alpha=1.0):
    """Cross-validated ridge decode R^2 (uniform-averaged over target columns)."""
    sc = StandardScaler().fit(X[tr])
    m = Ridge(alpha=alpha).fit(sc.transform(X[tr]), Y[tr])
    return float(r2_score(Y[te], m.predict(sc.transform(X[te])),
                          multioutput="uniform_average"))


def _position_vs_pcs(X, pos, tr, te, k_list, alpha=1.0):
    """Position decode R^2 from the top-k variance PCs (solid), the full tuned
    readout (dotted), and the variance-matched top-2 PCs (star)."""
    pca = PCA(n_components=int(max(k_list))).fit(X[tr])
    Str, Ste = pca.transform(X[tr]), pca.transform(X[te])

    def _r2(k):
        m = Ridge(alpha=alpha).fit(Str[:, :k], pos[tr])
        return float(r2_score(pos[te], m.predict(Ste[:, :k]),
                              multioutput="uniform_average"))
    curve = np.array([_r2(k) for k in k_list])
    full = _decode_r2(X, pos, tr, te, alpha)
    top2 = float(curve[list(k_list).index(2)]) if 2 in k_list else _r2(2)
    return curve, full, top2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_trials", type=int, default=40,
                    help="how many position_2d test rollouts to evaluate on")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge penalty")
    ap.add_argument("--horizon_steps", type=int, default=6000,
                    help="Test A: length of the naturalistic-OU extrapolation rollout")
    ap.add_argument("--n_ou_seed", type=int, default=4,
                    help="Test A: number of OU rollout realisations to average")
    ap.add_argument("--rel_d", type=float, default=0.20,
                    help="Test A: relative tolerance for the follow-time")
    ap.add_argument("--out", default=os.path.join(
        HERE, "fig_zebrafish_latent_translation.png"))
    args = ap.parse_args()
    # Test B leak-time-constant grid (seconds); inf = the true cumulative integral.
    TAUS = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, np.inf]
    try:
        set_data_root(load_data_root_from_json())
    except FileNotFoundError:
        pass
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    u, y = _load_eval_set(args.n_trials)
    print(f"eval set: {EVAL_DATASET}  u={u.shape}  y={y.shape}")

    results = []
    for run, lab, col in MODELS:
        print(f"[model] {run}")
        net, cfg = _load_run(run, device)
        if net is None:
            results.append(dict(run=run, lab=lab, col=col, trained=False))
            continue
        dt = float(getattr(net, "dt", None)
                   or cfg.task.swim_integration.dt)
        gt = _gt_targets(u, y, dt)
        R, tid = _rates(net, u, device)
        tr, te = _split(R.shape[0], seed=0)
        r2 = {k: _decode_r2(R, gt[k].reshape(-1, gt[k].shape[-1]), tr, te, args.alpha)
              for k in DECODE_KEYS}
        kmax = min(120, R.shape[1])
        k_list = list(range(1, 13)) + list(range(14, 41, 3)) + list(range(45, kmax + 1, 15))
        pos = gt["position"].reshape(-1, 2)
        curve, full, top2 = _position_vs_pcs(R, pos, tr, te, k_list, args.alpha)

        # ---- Test B: decode the true integral vs a leaky-filtered v_fwd -----
        # If the state predicts a finite-tau leaky filter much better than the
        # true integral (tau=inf), it is holding a slow trace, not integrating.
        v_trials = u[..., 1]                                   # (N, T)
        r2_tau = [_decode_r2(R, _leaky_filter(v_trials, dt, t).reshape(-1, 1),
                             tr, te, args.alpha) for t in TAUS]
        fin = [r2_tau[i] for i, t in enumerate(TAUS) if np.isfinite(t)]
        best_fin_i = int(np.argmax(fin))
        best_tau = [t for t in TAUS if np.isfinite(t)][best_fin_i]
        r2_inf = r2_tau[-1]                                    # tau = inf (true integral)

        # ---- Test A: extrapolation horizon of the position decode ----------
        # Fit the position decoder on the swim rollouts, then apply it to a long
        # NATURALISTIC Ornstein--Uhlenbeck rollout (the same in-distribution
        # drive as the precision-horizon probe) and measure how long it tracks
        # the true (growing) integral. A true integrator tracks for seconds; a
        # leaky echo saturates and falls behind.
        sc_pos = StandardScaler().fit(R[tr])
        dec_pos = Ridge(alpha=args.alpha).fit(sc_pos.transform(R[tr]), pos[tr])
        rec = _rec_ix(net)
        horizons, trace = [], None
        for sd in range(args.n_ou_seed):
            rr, true_xy = _ou_rollout(net, rec, device, args.horizon_steps, dt, sd)
            dec_xy = dec_pos.predict(sc_pos.transform(rr))    # (T, 2)
            tf = _follow_time(dec_xy, true_xy, dt, args.rel_d)
            horizons.append(tf)
            if sd == 0:
                t_arr = np.arange(true_xy.shape[0]) * dt
                trace = dict(t=t_arr, dec=np.linalg.norm(dec_xy, axis=-1),
                             true=np.linalg.norm(true_xy, axis=-1), tfollow=tf)
        t_follow = float(np.mean(horizons))

        results.append(dict(run=run, lab=lab, col=col, trained=True, r2=r2,
                            k_list=np.asarray(k_list), curve=curve,
                            pos_full=full, pos_top2=top2,
                            r2_tau=np.asarray(r2_tau), best_tau=best_tau,
                            r2_inf=r2_inf, t_follow=t_follow, trace=trace))
        print("   decode R^2: " + "  ".join(f"{k}={r2[k]:.3f}" for k in DECODE_KEYS)
              + f"   | pos top2PC={top2:.3f}")
        print(f"   Test A (extrapolation): T_follow={t_follow:.2f}s "
              f"(rollout {args.horizon_steps * dt:.0f}s)")
        print(f"   Test B (integral vs leaky): R2(true integral)={r2_inf:.3f}  "
              f"best leaky R2={max(fin):.3f} @ tau={best_tau:.2f}s")

    # ---- combined metrics table --------------------------------------------
    # Decode columns are cross-validated linear-decode R^2 from the recurrent
    # hidden state. pos(all)=position from the full population; pos(top2)=from
    # only the top-2 variance PCs (pos(all) >> pos(top2) = low-variance slice).
    # T_follow (Test A): seconds the position decode tracks the true integral on
    #   a constant-drive rollout (long = integrator, short ~tau = leaky echo).
    # R2_int / R2_leaky (Test B): true-integral vs best leaky-filter decode
    #   (int >= leaky = accumulating; leaky >> int = holding a slow trace).
    fin_mask = np.isfinite(np.asarray(TAUS))
    print("\n=== latent translation: decode + integrator-vs-echo (Tests A,B) "
          "+ connectome control (C) ===")
    cols = ("heading", "v_fwd", "distance", "pos(all)", "pos(top2)",
            "T_follow", "R2_int", "R2_leaky")
    print(f"{'model':30s}" + "".join(f"{c:>10s}" for c in cols))
    for r in results:
        tag = r["lab"].split("\n")[0]
        if not r["trained"]:
            print(f"{tag:30s}   (not trained yet)")
            continue
        r2_leaky = float(np.max(r["r2_tau"][fin_mask]))
        vals = [r["r2"]["heading"], r["r2"]["v_fwd"], r["r2"]["distance"],
                r["r2"]["position"], r["pos_top2"], r["t_follow"],
                r["r2_inf"], r2_leaky]
        print(f"{tag:30s}" + "".join(f"{v:10.3f}" for v in vals))

    # ---- figure ------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 11.5, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5, "legend.fontsize": 8.5, "lines.linewidth": 1.8,
        "axes.linewidth": 0.9, "xtick.major.size": 3.2, "ytick.major.size": 3.2,
    })
    fig = plt.figure(figsize=(11.5, 9.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.34,
                           width_ratios=[1.25, 1.0])

    def _pl(ax, s):
        ax.text(-0.12, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=14)

    trained = [r for r in results if r["trained"]]

    # (a) grouped bars: decode R^2 per quantity per model
    axA = fig.add_subplot(gs[0, 0])
    nk = len(DECODE_KEYS)
    nm = max(1, len(trained))
    width = 0.8 / nm
    xk = np.arange(nk)
    for j, r in enumerate(trained):
        vals = [max(r["r2"][k], 0.0) for k in DECODE_KEYS]
        axA.bar(xk + (j - (nm - 1) / 2) * width, vals, width=width,
                color=r["col"], label=r["lab"])
    axA.axhline(0, color="0.6", lw=0.8)
    axA.set_xticks(xk)
    axA.set_xticklabels([DECODE_LABEL[k] for k in DECODE_KEYS])
    axA.set_ylabel(r"decode $R^2$ (cross-val.)")
    axA.set_ylim(0, 1.05)
    axA.legend(loc="upper right", frameon=False)
    _pl(axA, "a")

    # (b) position decode vs #top-variance PCs
    axB = fig.add_subplot(gs[0, 1])
    for r in trained:
        axB.plot(r["k_list"], r["curve"], color=r["col"], label=r["lab"].replace("\n", " "))
        axB.axhline(r["pos_full"], color=r["col"], ls=":", lw=1.0)
        axB.plot(2, r["pos_top2"], marker="*", ms=13, color=r["col"], mec="k", mew=0.5, zorder=6)
    axB.set_xlabel("number of top-variance PCs")
    axB.set_ylabel(r"position $(x,y)$ decode $R^2$")
    axB.set_ylim(-0.05, 1.05)
    hB = [Line2D([0], [0], ls=":", color="0.3", label="full tuned readout"),
          Line2D([0], [0], marker="*", color="0.3", mec="k", ls="none", ms=11,
                 label="top-2 PCs")]
    axB.legend(handles=hB, loc="lower right", frameon=False)
    _pl(axB, "b")

    # (c) Test A: extrapolation horizon on a naturalistic OU rollout
    axC = fig.add_subplot(gs[1, 0])
    if trained:
        t0 = trained[0]["trace"]
        axC.plot(t0["t"], t0["true"], color="k", lw=2.2, label="true integral", zorder=1)
        for r in trained:
            tc = r["trace"]
            axC.plot(tc["t"], tc["dec"], color=r["col"], lw=1.6,
                     label=r["lab"].split("\n")[0])
            axC.axvline(tc["tfollow"], color=r["col"], ls="--", lw=0.9, alpha=0.7)
    axC.set_xlabel("time (s)")
    axC.set_ylabel(r"displacement $|x,y|$"+"\n(decoded vs true)")
    axC.legend(loc="upper left", frameon=False, fontsize=8)
    _pl(axC, "c")

    # (d) Test B: decode the true integral vs a leaky-filtered v_fwd
    axD = fig.add_subplot(gs[1, 1])
    finite_taus = [t for t in TAUS if np.isfinite(t)]
    for r in trained:
        axD.plot(finite_taus, r["r2_tau"][:len(finite_taus)], marker="o", ms=4,
                 color=r["col"], label=r["lab"].split("\n")[0])
        axD.axhline(r["r2_inf"], color=r["col"], ls=":", lw=1.0)
    axD.set_xscale("log")
    axD.set_xlabel(r"leak time constant $\tau$ (s)")
    axD.set_ylabel(r"distance decode $R^2$")
    hD = [Line2D([0], [0], marker="o", color="0.3", label=r"leaky filter at $\tau$"),
          Line2D([0], [0], ls=":", color="0.3", label=r"true integral ($\tau=\infty$)")]
    axD.legend(handles=hD, loc="lower left", frameon=False)
    _pl(axD, "d")

    if not trained:
        fig.text(0.5, 0.5, "no trained models found yet\n(train the configs, then re-run)",
                 ha="center", va="center", fontsize=13, color="0.4")

    try:
        from _despine import open_axes
        open_axes(fig)
    except Exception:
        pass
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig_latent_translation] wrote {args.out}")


if __name__ == "__main__":
    main()
