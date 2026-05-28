"""PCA ring-attractor analysis on dIPN activity.

Replays the Petrucco et al. 2023 (Fig 1e-f) ring-in-PC-space analysis on
our trained zebrafish HD model. Runs a long swim-integration rollout,
restricts to IPNd + IPNds neurons (the dIPN populations that form the
ring attractor in their data), performs PCA over time, and renders:

  A) cumulative variance explained by the first PCs
  B) PC1-PC2 trajectory coloured by elapsed time
  C) PC1-PC2 trajectory coloured by ground-truth HD angle

A clean ring in panel B + colour-matched angle in panel C demonstrates
that the model has learned a ring-attractor representation of HD in the
dIPN, recapitulating the Petrucco finding.

Usage:
  python fig_zebrafish_pca_ring.py --n_steps 30000 --seed 0
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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="zebrafish_hd_si_dipn")
    p.add_argument("--n_steps", type=int, default=30000,
                   help="rollout length in simulation steps (default 5 min "
                        "at dt=0.01)")
    p.add_argument("--burn_in_s", type=float, default=5.0,
                   help="seconds discarded at the start (transient)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cell_types", nargs="+",
                   default=["IPNd", "IPNds"],
                   help="categories included in the PCA")
    p.add_argument("--r_chunk_s", type=float, default=30.0,
                   help="chunk length (s) for the Petrucco-style "
                        "correlation distribution")
    p.add_argument("--anticorr_thresh", type=float, default=-0.5,
                   help="Petrucco-style r1pi selection: keep neurons whose "
                        "minimum correlation with any other neuron in the "
                        "pool is below this threshold (<0 = anticorrelated "
                        "partner exists). Set to 0 to disable selection.")
    p.add_argument("--connconstr_datapath",
                   default=os.path.join(here, "zebrafish_connectome_HD"))
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

    print(f"[1/3] loading model: {args.model}")
    net, _ = _load(args.model, device)
    dt = float(net.dt)

    print(f"[2/3] swim rollout n_steps={args.n_steps} "
          f"({args.n_steps * dt:.0f} s, seed={args.seed})")
    h, theta, _omega, decoded_hd, *_ = _run_swim(
        net, args.n_steps, dt, device, seed=args.seed)

    cx = load_zebrafish_hd_connectome(args.connconstr_datapath)
    types_arr = np.asarray(cx["category"])
    if len(types_arr) != h.shape[1]:
        raise SystemExit(f"category length {len(types_arr)} != model N "
                         f"{h.shape[1]}")
    keep_mask = np.isin(types_arr, args.cell_types)
    n_keep = int(keep_mask.sum())
    if n_keep == 0:
        raise SystemExit(f"no neurons match cell_types={args.cell_types}; "
                         f"available categories: {sorted(set(types_arr))}")
    h_sub = h[:, keep_mask]
    print(f"      {n_keep} neurons matching {args.cell_types}")

    burn_in = int(args.burn_in_s / dt)
    h_sub = h_sub[burn_in:]
    theta_used = theta[burn_in:]
    decoded_used = decoded_hd[burn_in:]

    # Z-score each neuron over time so units don't dominate
    h_z = (h_sub - h_sub.mean(0)) / (h_sub.std(0) + 1e-6)

    # Petrucco r1pi selection: a ring neuron has at least one strongly
    # anticorrelated partner. Without this step, non-ring neurons dilute
    # the PCA and the ring smears into a cloud.
    if args.anticorr_thresh < 0:
        C = np.corrcoef(h_z.T)
        np.fill_diagonal(C, 0.0)
        ring_mask = C.min(axis=1) < args.anticorr_thresh
        n_ring = int(ring_mask.sum())
        if n_ring < 10:
            print(f"      WARNING: only {n_ring} neurons pass anticorr "
                  f"threshold {args.anticorr_thresh}; relaxing to "
                  f"top-50 most-anticorrelated.")
            order = np.argsort(C.min(axis=1))
            ring_mask = np.zeros_like(ring_mask)
            ring_mask[order[:50]] = True
            n_ring = int(ring_mask.sum())
        h_z = h_z[:, ring_mask]
        print(f"      ring-neuron selection: kept {n_ring} of {n_keep} "
              f"(anticorr threshold {args.anticorr_thresh:+.2f})")
        n_keep = n_ring

    # PCA via SVD on the time-centred matrix (samples=time, features=neuron)
    print("[3/3] PCA + render")
    X = h_z - h_z.mean(0)
    U, S, _Vt = np.linalg.svd(X, full_matrices=False)
    var_total = float((S ** 2).sum())
    var_ratio = (S ** 2) / var_total
    cum_var = np.cumsum(var_ratio)
    scores = U * S  # (T, K)

    print(f"      var explained: PC1={var_ratio[0]:.3f}  "
          f"PC2={var_ratio[1]:.3f}  "
          f"cum@2={cum_var[1]:.3f}  cum@3={cum_var[2]:.3f}")

    # --- figure -----------------------------------------------------------
    # 2 x 3 layout: top row = PCA, bottom row = drift / R analysis
    fig = plt.figure(figsize=(15.5, 9.5), facecolor="white")
    gs = fig.add_gridspec(
        2, 3,
        left=0.06, right=0.985, top=0.95, bottom=0.07,
        hspace=0.32, wspace=0.30,
    )

    PANEL_FS = 13   # bold corner letter (matches Figure 2 style)
    TITLE_FS = 11
    LABEL_FS = 11
    TICK_FS = 10

    def _style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=TICK_FS)

    def _panel_letter(ax, letter):
        ax.text(-0.10, 1.06, letter, transform=ax.transAxes,
                ha="left", va="top", fontsize=PANEL_FS, fontweight="bold")

    t_sec = np.arange(scores.shape[0]) * dt
    theta_deg_wrapped = (((np.rad2deg(theta_used) + 180.0) % 360.0) - 180.0)
    decoded_deg_wrapped = (((np.rad2deg(decoded_used) + 180.0) % 360.0)
                            - 180.0)

    # panel a — cumulative variance
    ax = fig.add_subplot(gs[0, 0])
    n_show = min(15, len(cum_var))
    ax.plot(np.arange(1, n_show + 1), cum_var[:n_show],
            marker="o", color="black", lw=1.4, ms=5)
    ax.axhline(0.80, color="0.6", lw=0.8, ls="--")
    ax.axvline(2, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("PC index", fontsize=LABEL_FS)
    ax.set_ylabel("cumulative variance", fontsize=LABEL_FS)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(1, n_show + 1, 2))
    ax.set_title(
        f"PCA over time, n={n_keep}, first 2 PCs = {cum_var[1]:.2f}",
        fontsize=TITLE_FS, pad=8,
    )
    _panel_letter(ax, "a")
    _style(ax)

    # panel b — PC1 vs PC2 coloured by elapsed time
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=t_sec,
                    cmap="viridis", s=4.0, alpha=0.55, linewidths=0)
    ax.set_xlabel("PC1", fontsize=LABEL_FS)
    ax.set_ylabel("PC2", fontsize=LABEL_FS)
    ax.set_aspect("equal")
    ax.set_title("trajectory (by time)", fontsize=TITLE_FS, pad=8)
    cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=20)
    cb.set_label("time (s)", fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)
    _panel_letter(ax, "b")
    _style(ax)

    # panel c — PC1 vs PC2 coloured by ground-truth HD
    ax = fig.add_subplot(gs[0, 2])
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=theta_deg_wrapped,
                    cmap="hsv", s=4.0, alpha=0.6, linewidths=0,
                    vmin=-180, vmax=180)
    ax.set_xlabel("PC1", fontsize=LABEL_FS)
    ax.set_ylabel("PC2", fontsize=LABEL_FS)
    ax.set_aspect("equal")
    ax.set_title("trajectory (by ground-truth HD)",
                 fontsize=TITLE_FS, pad=8)
    cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=20,
                      ticks=[-180, -90, 0, 90, 180])
    cb.set_label("true HD (°)", fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)
    _panel_letter(ax, "c")
    _style(ax)

    # panel d — PC1 vs PC2 coloured by decoded HD
    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=decoded_deg_wrapped,
                    cmap="hsv", s=4.0, alpha=0.6, linewidths=0,
                    vmin=-180, vmax=180)
    ax.set_xlabel("PC1", fontsize=LABEL_FS)
    ax.set_ylabel("PC2", fontsize=LABEL_FS)
    ax.set_aspect("equal")
    ax.set_title("trajectory (by network-decoded HD)",
                 fontsize=TITLE_FS, pad=8)
    cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=20,
                      ticks=[-180, -90, 0, 90, 180])
    cb.set_label("decoded HD (°)", fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)
    _panel_letter(ax, "d")
    _style(ax)

    # panel e — drift over time: ψ(t) = unwrap(decoded - true)
    # unwrap on the raw decoded/theta (radians), then convert to deg
    err_rad = np.unwrap(decoded_used) - np.unwrap(theta_used)
    err_rad = err_rad - err_rad[0]  # zero at start
    err_deg = np.rad2deg(err_rad)
    # bias = linear slope over time; D = residual variance growth rate
    slope, intercept = np.polyfit(t_sec, err_deg, 1)  # deg per second
    resid = err_deg - (slope * t_sec + intercept)
    var_t = np.cumsum(resid ** 2) / np.arange(1, len(resid) + 1)
    # diffusion D from msd: <resid^2> ~ D * t  (deg^2 / s)
    D_fit = np.polyfit(t_sec[10:], resid[10:] ** 2, 1)[0]

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_sec, err_deg, color="0.25", lw=1.2, label="decoded − true")
    ax.plot(t_sec, slope * t_sec + intercept, color="C3", lw=1.6,
            label=f"bias = {slope:+.2f}°/s")
    ax.axhline(0, color="0.6", lw=0.6, ls="--")
    ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax.set_ylabel("drift  decoded − true HD (°)", fontsize=LABEL_FS)
    ax.set_title(
        f"integration drift  (bias {slope:+.2f}°/s, "
        f"$D$ = {D_fit:.1f} deg²/s)",
        fontsize=TITLE_FS, pad=8)
    ax.legend(loc="upper left", fontsize=TICK_FS, frameon=False)
    _panel_letter(ax, "e")
    _style(ax)

    # panel f — Petrucco R distribution over chunks
    chunk_s = float(args.r_chunk_s)
    chunk_n = int(chunk_s / dt)
    n_chunks = len(t_sec) // chunk_n
    R_vals = []
    # circular correlation R between decoded HD and true HD within each
    # window. Use complex unit vectors to handle wrap; report Pearson R of
    # the unwrapped values for direct comparison with Petrucco Fig 3f.
    for k in range(n_chunks):
        sl = slice(k * chunk_n, (k + 1) * chunk_n)
        d = np.unwrap(decoded_used[sl])
        t = np.unwrap(theta_used[sl])
        if d.std() < 1e-8 or t.std() < 1e-8:
            continue
        R_vals.append(float(np.corrcoef(d, t)[0, 1]))
    R_vals = np.array(R_vals)
    # shuffle baseline: pair decoded chunk k with true chunk k+1
    R_shuf = []
    for k in range(n_chunks - 1):
        sl1 = slice(k * chunk_n, (k + 1) * chunk_n)
        sl2 = slice((k + 1) * chunk_n, (k + 2) * chunk_n)
        d = np.unwrap(decoded_used[sl1])
        t = np.unwrap(theta_used[sl2])
        if d.std() < 1e-8 or t.std() < 1e-8 or len(d) != len(t):
            continue
        R_shuf.append(float(np.corrcoef(d, t)[0, 1]))
    R_shuf = np.array(R_shuf)

    ax = fig.add_subplot(gs[1, 2])
    bins = np.linspace(-1.0, 1.0, 21)
    if len(R_shuf):
        ax.hist(R_shuf, bins=bins, color="0.7", alpha=0.65,
                label=f"shuffle (n={len(R_shuf)})",
                edgecolor="0.4", lw=0.4)
    ax.hist(R_vals, bins=bins, color="C2", alpha=0.85,
            label=f"data (n={len(R_vals)})",
            edgecolor="black", lw=0.5)
    ax.axvline(np.median(R_vals), color="C2", lw=1.8, ls="--")
    ax.set_xlabel(f"corr(decoded, true) in {chunk_s:.0f}-s chunks",
                  fontsize=LABEL_FS)
    ax.set_ylabel("count", fontsize=LABEL_FS)
    ax.set_xlim(-1, 1)
    ax.set_title(
        f"Petrucco-style R in {chunk_s:.0f}-s chunks  "
        f"(median = {np.median(R_vals):+.2f})",
        fontsize=TITLE_FS, pad=8)
    ax.legend(loc="upper left", fontsize=TICK_FS, frameon=False)
    _panel_letter(ax, "f")
    _style(ax)

    delta = (((decoded_deg_wrapped - theta_deg_wrapped) + 180.0)
             % 360.0) - 180.0
    print(f"      decoded-vs-true HD wrapped: "
          f"bias={np.median(delta):+.1f} deg, std={delta.std():.1f} deg")
    print(f"      drift fit: bias = {slope:+.3f} deg/s, "
          f"diffusion D = {D_fit:.2f} deg^2/s")
    print(f"      R in {chunk_s:.0f}s chunks: "
          f"median={np.median(R_vals):+.3f} "
          f"(shuffle median={np.median(R_shuf):+.3f})")

    if args.out_path is None:
        args.out_path = os.path.join(here, "fig_zebrafish_pca_ring.png")
    fig.savefig(args.out_path, dpi=180)
    print(f"saved {args.out_path}")


if __name__ == "__main__":
    main()
