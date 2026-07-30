"""Per-trial distance-decode dashboard for the two heading-only (+v_fwd) models:
the MEASURED connectome vs the ER (randomized-connectome) control.

For each model we fit a linear decoder from the recurrent hidden state to the
forward distance on held-in trials, then plot the decode on held-out trials
against two ground-truth references:

  * cumulative  d = int v_fwd dt          (a true integrator should track this)
  * leaky       d = leaky-filtered v_fwd  (a slow echo saturates at ~tau here)

Layout: rows = held-out trials; columns = {measured, ER} x {cumulative, leaky}.
Each panel overlays the true target (black) and the decode (coloured) with the
per-trial R^2. The measured-vs-ER contrast is the connectome-necessity test; the
cumulative-vs-leaky contrast is the integrator-vs-echo test, made per-trial.

One script = one figure (fig_zebrafish_vfwd_distance_dashboard.png).

Usage:
    /home/allierc@hhmi.org/miniforge3/envs/neural-graph-linux/bin/python \\
        figures/zebrafish/fig_zebrafish_vfwd_distance_dashboard.py \\
        --device cuda --n_show 6 --tau 0.5
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from fig_zebrafish_latent_translation import (  # noqa: E402
    _load_run, _load_eval_set, _rec_ix, _leaky_filter,
)
from connectome_gnn.utils import (  # noqa: E402
    load_data_root_from_json, set_data_root,
)

# (run_basename, label, colour) — measured first, ER control second.
MODELS = (
    ("zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd",    "measured", "#6a51a3"),
    ("zebrafish_hd_si_ipn_917_v1_selfmotion_rotation_vfwd_er", "ER",       "#d1495b"),
)
TARGETS = ("cumulative", "leaky")


def _rates_per_trial(net, u, device):
    """List of (T, n_rec) recurrent firing-rate arrays, one per trial."""
    rec = _rec_ix(net)
    n_in = int(net.n_input)
    out = []
    for k in range(u.shape[0]):
        uk = u[k, :, :n_in][None].astype(np.float32)
        with torch.no_grad():
            r = net._sigma(net(torch.from_numpy(uk).to(device))[1])
        out.append(r[0].cpu().numpy().astype(np.float64)[:, rec])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_trials", type=int, default=120,
                    help="test rollouts loaded (held-out shown + held-in for the fit; "
                         "the cumulative target is non-stationary, so the decoder "
                         "needs many held-in trials to generalise across trajectories)")
    ap.add_argument("--n_show", type=int, default=6, help="held-out trials to plot (rows)")
    ap.add_argument("--tau", type=float, default=0.5, help="leaky-target time constant (s)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(
        HERE, "fig_zebrafish_vfwd_distance_dashboard.png"))
    args = ap.parse_args()
    try:
        set_data_root(load_data_root_from_json())
    except FileNotFoundError:
        pass
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    u, y = _load_eval_set(args.n_trials)
    n_show = min(args.n_show, u.shape[0] - 1)
    show_ix = np.arange(n_show)              # held-out (plotted)
    fit_ix = np.arange(n_show, u.shape[0])   # held-in (decoder fit)
    print(f"eval u={u.shape}  show {n_show} trials, fit on {len(fit_ix)} trials")

    per_model = []
    for run, lab, col in MODELS:
        print(f"[model] {run}")
        net, cfg = _load_run(run, device)
        if net is None:
            per_model.append(dict(lab=lab, col=col, trained=False))
            continue
        dt = float(getattr(net, "dt", None) or cfg.task.swim_integration.dt)
        rates = _rates_per_trial(net, u, device)
        v = u[..., 1]                                    # (N, T) forward velocity
        tgt = {
            "cumulative": np.cumsum(v, axis=1) * dt,     # (N, T)
            "leaky":      _leaky_filter(v, dt, args.tau),
        }
        t_arr = np.arange(u.shape[1]) * dt
        panels = {}
        for name in TARGETS:
            Xtr = np.concatenate([rates[k] for k in fit_ix], 0)
            Ytr = np.concatenate([tgt[name][k] for k in fit_ix], 0)
            sc = StandardScaler().fit(Xtr)
            dec = Ridge(alpha=args.alpha).fit(sc.transform(Xtr), Ytr)
            rows = []
            for k in show_ix:
                pred = dec.predict(sc.transform(rates[k]))
                r2 = float(r2_score(tgt[name][k], pred))
                rows.append(dict(t=t_arr, true=tgt[name][k], pred=pred, r2=r2))
            panels[name] = rows
        per_model.append(dict(lab=lab, col=col, trained=True, panels=panels))

    # ---------------------------- figure ------------------------------------
    plt.rcParams.update({
        "font.size": 9.5, "axes.labelsize": 10, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "lines.linewidth": 1.4,
        "axes.linewidth": 0.8, "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    })
    n_cols = len(MODELS) * len(TARGETS)
    fig = plt.figure(figsize=(3.4 * n_cols, 1.7 * n_show + 0.6))
    gs = gridspec.GridSpec(n_show, n_cols, figure=fig, hspace=0.32, wspace=0.30)

    col_specs = [(mi, m, tn) for mi, m in enumerate(per_model) for tn in TARGETS]
    for ci, (mi, m, tname) in enumerate(col_specs):
        for ri in range(n_show):
            ax = fig.add_subplot(gs[ri, ci])
            if ri == 0:
                ax.text(-0.05, 1.06, "abcdefgh"[ci], transform=ax.transAxes,
                        fontweight="bold", fontsize=13, va="bottom", ha="right")
            if not m["trained"]:
                ax.text(0.5, 0.5, "not trained yet", transform=ax.transAxes,
                        ha="center", va="center", color="0.5", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            p = m["panels"][tname][ri]
            ax.plot(p["t"], p["true"], color="green", lw=1.6, label="ground truth")
            ax.plot(p["t"], p["pred"], color="k", lw=1.4, label="prediction")
            ax.text(0.03, 0.93, f"$R^2$={p['r2']:.2f}", transform=ax.transAxes,
                    va="top", fontsize=8, color="0.25")
            # clip the y-axis to the true-target range so a poor decode cannot
            # blow up the panel (keeps the dashboard readable mid-training).
            lo, hi = float(np.min(p["true"])), float(np.max(p["true"]))
            pad = 0.25 * (hi - lo + 1e-6)
            ax.set_ylim(min(lo, 0.0) - pad, hi + pad)
            if ci == 0:
                ax.set_ylabel(f"trial {ri}\ndistance", fontsize=8)
            if ri == n_show - 1:
                ax.set_xlabel("time (s)")
            if ri == 0 and ci == 0:
                ax.legend(loc="lower right", frameon=False, fontsize=7.5)

    try:
        from _despine import open_axes
        open_axes(fig)
    except Exception:
        pass
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_vfwd_distance_dashboard] wrote {args.out}  (leaky tau={args.tau}s)")


if __name__ == "__main__":
    main()
