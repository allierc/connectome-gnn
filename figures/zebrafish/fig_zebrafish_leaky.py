"""Downstream leaky-integrator calibration of the zebrafish HD decoder.

The trained network gives an accurate angular velocity (R = 0.88 in
30-s chunks vs ground truth, see fig_zebrafish_pca_ring panel f) but a
slowly drifting absolute heading (bias ~ -1.3 deg/s + diffusion ~ 250
deg^2/s in a 5-min swim rollout). This is the classic failure mode of
a pure path integrator: angular velocity gets accumulated faithfully,
but unbiased errors compound and a small systematic gain error
produces a linear bias.

A biological fix is downstream: anchor the integrated heading
periodically toward an external reference (visual landmark,
gravitational vector, ...). In the larval zebrafish, the habenular
input to the IPN via the fasciculus retroflexus is in the right
anatomical position to carry such a signal.

We implement the simplest version of this here as a post-hoc filter
on the trained network's output, NOT a retraining:

  theta_leaky(t+1) = theta_leaky(t)
                     + d_theta_net(t)             # network angular vel.
                     - (dt / tau) * sin(theta_leaky(t) - theta_anchor(t))

The `sin(...)` term is a circular-leak pull toward the anchor, with
time constant `tau`. With tau = infinity we recover the pure
integrator (matches network output exactly). With small `tau` the
anchor dominates and drift is bounded.

Three panels:
  a. HD traces: ground truth, network-decoded, leaky-calibrated
  b. drift |decoded - true| over time, for several values of tau
  c. residual error (RMS) vs anchor time constant tau (sweep)

Usage:
  python fig_zebrafish_leaky.py --n_steps 30000 --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig_zebrafish_anatomy_3d_voltage_anim import _load, _run_swim
from connectome_gnn.utils import load_data_root_from_json, set_data_root
from connectome_gnn.generators.connconstr_data import (
    load_zebrafish_hd_connectome,
)


def _wrap(x_rad):
    return np.angle(np.exp(1j * x_rad))


def leaky_integrate(decoded_rad, theta_anchor_rad, dt, tau,
                    anchor_every_s=None):
    """Run the leaky integrator on a single trajectory.

    decoded_rad: network's decoded HD (radians), length T.
    theta_anchor_rad: external anchor (radians), length T.
    dt: timestep.
    tau: leak time constant in s. inf = no leak.
    anchor_every_s: if set, only use the anchor sample once every this
        many seconds (other steps drop the leak term). None = use anchor
        at every step.
    """
    T = len(decoded_rad)
    d_unwrap = np.unwrap(decoded_rad)
    dtheta = np.diff(d_unwrap, prepend=d_unwrap[0])
    hat = np.zeros(T)
    hat[0] = decoded_rad[0]
    if anchor_every_s is not None:
        anchor_stride = max(1, int(anchor_every_s / dt))
    else:
        anchor_stride = 1
    leak = 0.0 if not np.isfinite(tau) else (dt / float(tau))
    for k in range(1, T):
        hat[k] = hat[k - 1] + dtheta[k]
        if leak > 0 and (k % anchor_stride == 0):
            err = _wrap(hat[k] - theta_anchor_rad[k])
            hat[k] = hat[k] - leak * np.sin(err) * float(anchor_stride)
    return hat


def _rms_circular(a_rad, b_rad):
    """Root-mean-square circular distance in degrees."""
    d = _wrap(a_rad - b_rad)
    return float(np.sqrt(np.mean(np.rad2deg(d) ** 2)))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="zebrafish_hd_si_dipn")
    p.add_argument("--n_steps", type=int, default=30000)
    p.add_argument("--burn_in_s", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--taus", type=float, nargs="+",
                   default=[float("inf"), 60.0, 10.0, 2.0],
                   help="leak time constants (s) to compare in panel b")
    p.add_argument("--tau_sweep", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0,
                            120.0, 300.0],
                   help="time constants for the sweep in panel c")
    p.add_argument("--anchor_every_s", type=float, default=None,
                   help="if set, the anchor is sampled once every this "
                        "many seconds; otherwise every dt.")
    p.add_argument("--connconstr_datapath",
                   default=os.path.join(here, "zebrafish_connectome_HD"))
    p.add_argument("--cell_types", nargs="+",
                   default=["IPNd", "IPNds"])
    p.add_argument("--top_anticorr", type=int, default=50,
                   help="top-K most-anticorrelated dIPN neurons used for "
                        "the PCA-ring panel d")
    p.add_argument("--leaky_tau_pca", type=float, default=2.0,
                   help="leak time constant used to colour panel d")
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

    print(f"[1/3] loading {args.model}")
    net, _ = _load(args.model, device)
    dt = float(net.dt)

    print(f"[2/3] swim rollout n_steps={args.n_steps} "
          f"({args.n_steps * dt:.0f} s, seed={args.seed})")
    h, theta, _omega, decoded, *_ = _run_swim(
        net, args.n_steps, dt, device, seed=args.seed)
    burn = int(args.burn_in_s / dt)
    theta = theta[burn:]
    decoded = decoded[burn:]
    h = h[burn:]
    t_sec = np.arange(len(theta)) * dt

    # PCA for panel d: same selection as fig_zebrafish_pca_ring (50 most
    # anticorrelated IPNd+IPNds neurons), so the trajectory is comparable.
    cx = load_zebrafish_hd_connectome(args.connconstr_datapath)
    cat = np.asarray(cx["category"])
    keep = np.isin(cat, args.cell_types)
    h_sub = h[:, keep]
    h_z = (h_sub - h_sub.mean(0)) / (h_sub.std(0) + 1e-6)
    C = np.corrcoef(h_z.T)
    np.fill_diagonal(C, 0.0)
    K = int(args.top_anticorr)
    ring_idx = np.argsort(C.min(axis=1))[:K]
    h_ring = h_z[:, ring_idx]
    X = h_ring - h_ring.mean(0)
    U, S, _Vt = np.linalg.svd(X, full_matrices=False)
    scores = U * S

    # Network drift (no leak) for reference
    rms_net = _rms_circular(decoded, theta)
    print(f"      no leak (network)        : RMS error = {rms_net:.1f} deg")

    # Run leaky integration for each tau
    print("[3/3] leaky integration sweep")
    traces = {}  # tau -> hat trajectory
    for tau in args.taus:
        hat = leaky_integrate(decoded, theta, dt, tau,
                              anchor_every_s=args.anchor_every_s)
        rms = _rms_circular(hat, theta)
        traces[tau] = hat
        print(f"      tau = {tau:>7.1f} s: RMS = {rms:.1f} deg")

    sweep_rms = []
    for tau in args.tau_sweep:
        hat = leaky_integrate(decoded, theta, dt, tau,
                              anchor_every_s=args.anchor_every_s)
        sweep_rms.append(_rms_circular(hat, theta))
    sweep_rms = np.array(sweep_rms)

    # --- figure ---------------------------------------------------------
    # 2 x 2: top row = traces + drift; bottom row = RMS sweep + PCA-ring
    # under leaky colouring.
    fig = plt.figure(figsize=(13.5, 9.0), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0],
                          left=0.07, right=0.985, top=0.95,
                          bottom=0.08, wspace=0.28, hspace=0.32)
    PANEL_FS = 13
    TITLE_FS = 11
    LABEL_FS = 11
    TICK_FS = 10

    def _style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=TICK_FS)

    def _panel_letter(ax, letter):
        ax.text(-0.10, 1.06, letter, transform=ax.transAxes,
                ha="left", va="top", fontsize=PANEL_FS, fontweight="bold")

    # panel a — HD traces (true / network / one leaky)
    ax = fig.add_subplot(gs[0, 0])
    theta_unwrap = np.unwrap(theta)
    decoded_unwrap = np.unwrap(decoded)
    ax.plot(t_sec, np.rad2deg(theta_unwrap),
            color="black", lw=1.4, label="true HD")
    ax.plot(t_sec, np.rad2deg(decoded_unwrap),
            color="C3", lw=1.2, alpha=0.85,
            label=f"network ({rms_net:.0f}° RMS)")
    finite_taus = [t for t in args.taus if np.isfinite(t)]
    if finite_taus:
        tau_demo = min(finite_taus)
        rms_demo = _rms_circular(traces[tau_demo], theta)
        ax.plot(t_sec, np.rad2deg(traces[tau_demo]),
                color="C2", lw=1.2, alpha=0.85,
                label=fr"leaky, $\tau$={tau_demo:g} s "
                       fr"({rms_demo:.0f}° RMS)")
    ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax.set_ylabel("unwrapped HD (°)", fontsize=LABEL_FS)
    ax.set_title("HD traces", fontsize=TITLE_FS, pad=8)
    ax.legend(loc="upper left", fontsize=TICK_FS, frameon=False)
    _panel_letter(ax, "a")
    _style(ax)

    # panel b — wrapped error |hat - true| over time, for several tau
    ax = fig.add_subplot(gs[0, 1])
    for tau, hat in traces.items():
        err = np.abs(np.rad2deg(_wrap(hat - theta)))
        label = (r"no leak ($\tau\to\infty$)" if not np.isfinite(tau)
                 else fr"$\tau$ = {tau:g} s")
        ax.plot(t_sec, err, lw=1.2, alpha=0.8, label=label)
    ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax.set_ylabel("|decoded − true| HD (°, wrapped)",
                  fontsize=LABEL_FS)
    ax.set_title("drift suppression by leak", fontsize=TITLE_FS, pad=8)
    ax.set_ylim(0, 190)
    ax.legend(loc="upper right", fontsize=TICK_FS, frameon=False)
    _panel_letter(ax, "b")
    _style(ax)

    # panel c — RMS error vs tau
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogx(args.tau_sweep, sweep_rms,
                marker="o", lw=1.4, ms=6, color="C0")
    ax.axhline(rms_net, color="C3", ls="--", lw=1.0,
               label=f"no leak: {rms_net:.0f}°")
    ax.set_xlabel(r"leak time constant $\tau$ (s)",
                  fontsize=LABEL_FS)
    ax.set_ylabel("RMS HD error (°)", fontsize=LABEL_FS)
    ax.set_title(r"RMS error vs $\tau$", fontsize=TITLE_FS, pad=8)
    ax.legend(loc="upper left", fontsize=TICK_FS, frameon=False)
    _panel_letter(ax, "c")
    _style(ax)

    # panel d — PCA-ring trajectory coloured by leaky-calibrated HD.
    # KEY POINT: the PCA is over neural activity, so the trajectory is
    # identical to fig:pca_ring; only the colour values are different.
    # If the leak corrects drift, this panel should look more like
    # fig:pca_ring c (true HD smearing) than d (decoded HD clean ring),
    # demonstrating that the drift lives in the readout phase, not in
    # the ring structure itself.
    ax = fig.add_subplot(gs[1, 1])
    hat_pca = leaky_integrate(decoded, theta, dt, args.leaky_tau_pca,
                              anchor_every_s=args.anchor_every_s)
    leaky_deg = (((np.rad2deg(hat_pca) + 180.0) % 360.0) - 180.0)
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=leaky_deg,
                    cmap="hsv", s=4.0, alpha=0.6, linewidths=0,
                    vmin=-180, vmax=180)
    ax.set_xlabel("PC1", fontsize=LABEL_FS)
    ax.set_ylabel("PC2", fontsize=LABEL_FS)
    ax.set_aspect("equal")
    ax.set_title(
        fr"PCA ring coloured by leaky HD ($\tau={args.leaky_tau_pca:g}\,$s)",
        fontsize=TITLE_FS, pad=8)
    cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=20,
                      ticks=[-180, -90, 0, 90, 180])
    cb.set_label("leaky HD (°)", fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)
    _panel_letter(ax, "d")
    _style(ax)

    if args.out_path is None:
        args.out_path = os.path.join(here, "fig_zebrafish_leaky.png")
    fig.savefig(args.out_path, dpi=180)
    print(f"saved {args.out_path}")


if __name__ == "__main__":
    main()
