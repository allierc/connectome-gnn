"""Preferred-heading angle map of HD-coding cells in the trained
zebrafish_hd_si_dipn RNN.

For every model neuron classified in $\\mathsf{R}$ or $\\mathsf{L}$ by the
four-class partition of ``fig_zebrafish_four_classes.py`` (i.e. every
unit whose mutual information with $\\theta$ exceeds the median across
the network), this script:

  1. Runs a constant-omega rollout so $\\theta(t)$ sweeps the full
     $[-\\pi, +\\pi)$ at a known rate. Cells are silenced at the start
     (burn-in) so the tuning curve isn't dominated by the initial
     condition.
  2. Bins $\\hat h_i(t)$ by $\\theta(t)$ to obtain a tuning curve
     $\\bar r_i(\\theta_k)$ over $n_{\\theta}$ heading bins.
  3. Computes the centred resultant
        $\\rho_i e^{i\\phi_i} = \\sum_k (\\bar r_{ik} - \\langle \\bar r_i\\rangle)
                                  \\, e^{i\\theta_k}$
     and the specificity
        $\\sigma_i = |\\rho_i| / \\sum_k |\\bar r_{ik} - \\langle \\bar r_i\\rangle|$
     (mean resultant length of the centred tuning curve, in $[0, 1]$).
  4. Keeps only cells with $\\sigma_i \\ge \\sigma^\\star$ (default 0.30):
     these are the cells with a sharp single preferred direction.

The figure has two anatomy panels with the same view:
  (a) somata of the surviving cells, colour-coded by preferred angle
      $\\phi_i$ (cyclic HSV, $-180^\\circ \\dots +180^\\circ$).
  (b) skeletons of the same cells, also coloured by $\\phi_i$, so the
      anatomical organisation of the inferred heading code is visible
      across the dIPN neuropil.

Both panels share a dark-grey skeleton backdrop of every available
SWC for spatial context.

Usage:
  python fig_zebrafish_pref_angle.py \\
      --model zebrafish_hd_si_dipn --classes_csv fig_zebrafish_four_classes.csv \\
      --omega_deg_per_s 90 --n_steps 6000 --sigma_thr 0.30
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig_zebrafish_anatomy_3d_voltage_anim import (
    _run_const, _run_single_impulse,
    _model_index_to_bodyid, _load_skeletons_in_model_order,
)
from fig_zebrafish_four_classes import (
    _load_with_override, _project_2d, _extract_per_neuron_segments,
    _per_neuron_soma,
)
from connectome_gnn.utils import load_data_root_from_json, set_data_root


def _tuning_curves(h, theta, n_theta):
    """Returns r_bar (N_neurons, n_theta) — per-bin mean of h.
    theta is wrapped to [-pi, pi) before binning."""
    theta_w = np.angle(np.exp(1j * theta))
    edges = np.linspace(-np.pi, np.pi, n_theta + 1)
    bin_ix = np.clip(np.digitize(theta_w, edges) - 1, 0, n_theta - 1)
    T, N = h.shape
    r_bar = np.zeros((N, n_theta), dtype=np.float32)
    counts = np.zeros(n_theta, dtype=np.int64)
    for k in range(n_theta):
        mask = bin_ix == k
        if mask.any():
            r_bar[:, k] = h[mask].mean(axis=0)
            counts[k] = int(mask.sum())
    return r_bar, counts


def _preferred_angle_and_specificity(r_bar):
    """For each neuron return (phi, sigma) — preferred angle in (-pi, pi]
    and specificity in [0, 1]. Both vectors length N."""
    N, K = r_bar.shape
    theta_k = np.linspace(-np.pi, np.pi, K, endpoint=False) + (np.pi / K)
    centered = r_bar - r_bar.mean(axis=1, keepdims=True)
    # resultant vector of the centred tuning curve
    res = (centered * np.exp(1j * theta_k)).sum(axis=1)
    phi = np.angle(res)
    denom = np.abs(centered).sum(axis=1)
    sigma = np.where(denom > 1e-9, np.abs(res) / denom, 0.0)
    return phi.astype(np.float32), sigma.astype(np.float32)


def _render(seg_per_neuron, somas, keep_mask, phi, classes_per_neuron,
            output_path, elev=90.0, azim=-90.0,
            background="black", cmap_name="hsv",
            grey_color=(0.30, 0.30, 0.30),
            grey_lw=0.2, grey_alpha=0.30,
            keep_lw=0.5, keep_alpha=0.95,
            n_bins=9):
    """3x3 montage: bin preferred angle into n_bins equal wedges spanning
    [-pi, +pi); each panel shows only the skeleton arbors of cells whose
    preferred angle falls in that wedge, drawn in the wedge's central hue
    on top of a dark-grey backbone of the full circuit."""
    from matplotlib.collections import LineCollection

    text_color = "white" if background == "black" else "black"
    cmap = plt.get_cmap(cmap_name)

    # Pre-project every segment to 2D once (shared across all panels).
    seg2d = []
    for segs3d in seg_per_neuron:
        if len(segs3d) == 0:
            seg2d.append(np.zeros((0, 2, 2), dtype=np.float32))
        else:
            seg2d.append(
                _project_2d(segs3d.reshape(-1, 3), elev, azim).reshape(-1, 2, 2)
                .astype(np.float32)
            )
    grey_backdrop = (np.concatenate(seg2d, axis=0)
                      if seg2d else np.zeros((0, 2, 2)))
    soma_xy = _project_2d(somas, elev, azim)
    valid = np.isfinite(soma_xy[:, 0])

    # Bin edges over (-pi, +pi]. Centre angles serve as the panel colour.
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # bin assignment, only for kept cells; -1 elsewhere
    bin_ix = np.full_like(phi, -1, dtype=np.int64)
    keep_show = keep_mask & valid
    keep_ix = np.where(keep_show)[0]
    bin_ix[keep_ix] = np.clip(np.digitize(phi[keep_ix], edges) - 1,
                               0, n_bins - 1)

    # 3x3 layout (or as close as we can get if n_bins != 9)
    n_rows = int(np.ceil(np.sqrt(n_bins)))
    n_cols = int(np.ceil(n_bins / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.0 * n_cols, 3.6 * n_rows),
                              facecolor=background)
    axes = np.atleast_2d(axes)

    # shared view limits from the backdrop
    if len(grey_backdrop):
        all_pts = grey_backdrop.reshape(-1, 2)
        x_lo, y_lo = np.percentile(all_pts, 1, axis=0)
        x_hi, y_hi = np.percentile(all_pts, 99, axis=0)
        pad_x = 0.04 * (x_hi - x_lo)
        pad_y = 0.04 * (y_hi - y_lo)
    else:
        x_lo, x_hi, y_lo, y_hi, pad_x, pad_y = -1, 1, -1, 1, 0, 0

    panel_letters = list("abcdefghijklmnopqrstuvwxyz")
    for k, ax in enumerate(axes.flat):
        ax.set_facecolor(background)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if k >= n_bins:
            continue
        # dark-grey backbone
        if len(grey_backdrop):
            ax.add_collection(LineCollection(
                grey_backdrop, colors=[grey_color], linewidths=grey_lw,
                alpha=grey_alpha, zorder=1,
            ))
        in_bin = np.where(bin_ix == k)[0]
        c = cmap((centres[k] + np.pi) / (2 * np.pi))
        # Skeleton arbors of this bin's cells
        segs_in = [seg2d[i] for i in in_bin if len(seg2d[i])]
        if segs_in:
            ax.add_collection(LineCollection(
                np.concatenate(segs_in, axis=0),
                colors=[c], linewidths=keep_lw, alpha=0.5,
                zorder=2,
            ))
        # Somata on top — these are what would show anatomical topography
        # if it existed. The skeleton arbors all converge to the same dIPN
        # neuropil so they cannot resolve it.
        if in_bin.size:
            ax.scatter(soma_xy[in_bin, 0], soma_xy[in_bin, 1],
                       c=[c], s=14, edgecolors="none", zorder=3)
        ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
        ax.set_ylim(y_lo - pad_y, y_hi + pad_y)

        lo_deg = int(round(np.degrees(edges[k])))
        hi_deg = int(round(np.degrees(edges[k + 1])))
        ax.text(0.02, 0.97, panel_letters[k], transform=ax.transAxes,
                ha="left", va="top", fontsize=14, fontweight="bold",
                color=text_color)
        ax.text(0.10, 0.965,
                f"$\\phi \\in [{lo_deg:+d}^\\circ, {hi_deg:+d}^\\circ)$"
                f"   (n={len(in_bin)})",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, color=text_color)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.06,
                        hspace=0.04, wspace=0.02)

    # Single horizontal phase colour reference at the bottom.
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=-180, vmax=180))
    sm.set_array([])
    cax = fig.add_axes([0.30, 0.025, 0.40, 0.018])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(r"preferred heading angle $\phi_i$ (deg)",
                  color=text_color, fontsize=10)
    cb.set_ticks(np.round(np.degrees(edges)).astype(int).tolist())
    cb.ax.tick_params(colors=text_color, labelsize=8)
    cb.outline.set_edgecolor(text_color)

    fig.savefig(output_path, dpi=210, facecolor=background,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="zebrafish_hd_si_dipn")
    p.add_argument("--classes_csv",
                   default=os.path.join(here, "fig_zebrafish_four_classes.csv"),
                   help="per-neuron classification produced by "
                        "fig_zebrafish_four_classes.py; cells whose class is "
                        "in --hd_classes are scored for tuning")
    p.add_argument("--hd_classes", nargs="+", default=["R", "L"])
    p.add_argument("--omega_deg_per_s", type=float, default=90.0,
                   help="constant turn rate for the rollout (deg/s). With "
                        "the default the heading sweeps 360 deg every 4 s.")
    p.add_argument("--n_steps", type=int, default=10000)
    p.add_argument("--burn_in_s", type=float, default=4.0)
    p.add_argument("--rollout", default="periodic",
                   choices=["const", "periodic"],
                   help="probe stimulus. 'periodic' (default) matches "
                        "the kinograph rollout: discrete L swim impulses "
                        "every --swim_interval_s seconds at "
                        "--swim_magnitude_rad rad. 'const' uses the "
                        "smooth omega sweep from the previous version.")
    p.add_argument("--swim_interval_s", type=float, default=0.3)
    p.add_argument("--swim_magnitude_rad", type=float, default=0.393)
    p.add_argument("--swim_direction", default="L", choices=["L", "R"])
    p.add_argument("--n_theta", type=int, default=36,
                   help="number of heading bins (default 36 = 10 deg)")
    p.add_argument("--theta0", type=float, default=0.0)
    p.add_argument("--sigma_thr", type=float, default=0.70,
                   help="specificity threshold: cells with sigma < thr are "
                        "discarded as not sharply tuned. Empirically the "
                        "R/L cells form a tight cluster at sigma ~ 0.75; a "
                        "thr of 0.70 drops the ~10%% below-mode outliers.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--anatomy_dir",
                   default=os.path.join(here, "zebrafish_anatomy_HD"))
    p.add_argument("--connectome_dir",
                   default=os.path.join(here, "zebrafish_connectome_HD"))
    p.add_argument("--downsample", type=int, default=10)
    p.add_argument("--elev", type=float, default=90.0)
    p.add_argument("--azim", type=float, default=-90.0)
    p.add_argument("--bg", default="white", choices=["black", "white"])
    p.add_argument("--cmap", default="hsv")
    p.add_argument("--run_name", default=None,
                   help="optional override of the checkpoint run directory")
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--out", default=os.path.join(here, "fig_zebrafish_pref_angle.png"))
    p.add_argument("--csv_out", default=os.path.join(here, "fig_zebrafish_pref_angle.csv"))
    p.add_argument("--output_root", default=None)
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass

    device = torch.device(args.device)

    # 1) model
    from connectome_gnn.utils import log_path
    ckpt_dir = args.ckpt_dir or (
        log_path("zebrafish", args.run_name, "models") if args.run_name else None
    )
    print(f"[1/6] loading model config={args.model}  ckpt_dir={ckpt_dir}")
    net, _ = _load_with_override(args.model, device, ckpt_dir=ckpt_dir)
    dt = float(net.dt)
    type_names = list(net.type_names)
    neuron_types = np.asarray(net.neuron_types).astype(int)
    type_per_neuron = np.array([type_names[t] for t in neuron_types])
    N = len(neuron_types)

    # 2) rollout — periodic single-direction swim impulses by default
    # (matches the kinograph rollout: discrete heading steps of
    # `magnitude_rad` every `interval_s`); pass --rollout const for the
    # smooth omega sweep used in the previous version of this figure.
    if args.rollout == "const":
        print(f"[2/6] const rollout omega={args.omega_deg_per_s} deg/s, "
              f"n_steps={args.n_steps} ({args.n_steps * dt:.1f} s)")
        h, theta, _omega, _decoded, *_ = _run_const(
            net, args.n_steps, dt, args.omega_deg_per_s, args.theta0, device)
    else:
        print(f"[2/6] periodic-{args.swim_direction} rollout, "
              f"Δt={args.swim_interval_s}s, "
              f"mag={args.swim_magnitude_rad:.3f} rad, "
              f"n_steps={args.n_steps} ({args.n_steps * dt:.1f} s)")
        h, theta, _omega, _decoded, *_ = _run_single_impulse(
            net, args.n_steps, dt, device,
            direction=args.swim_direction,
            magnitude_rad=args.swim_magnitude_rad,
            t_event_s=0.0,
            interval_s=args.swim_interval_s,
            theta0=args.theta0,
        )
    burn = int(args.burn_in_s / dt)
    h = h[burn:]
    theta = theta[burn:]
    span_rev = (theta[-1] - theta[0]) / (2 * np.pi)
    print(f"      swept {span_rev:+.1f} revolutions during the analysis window")

    # 3) tuning curves
    print(f"[3/6] tuning curves at {args.n_theta} bins")
    r_bar, counts = _tuning_curves(h, theta, args.n_theta)
    if (counts == 0).any():
        print(f"      warning: {int((counts == 0).sum())} empty heading bins")
    phi, sigma = _preferred_angle_and_specificity(r_bar)

    # 4) select HD-coding cells & specificity threshold
    print(f"[4/6] loading classes from {args.classes_csv}")
    if not os.path.exists(args.classes_csv):
        sys.exit(f"missing {args.classes_csv} -- run "
                 "fig_zebrafish_four_classes.py first")
    cls_df = pd.read_csv(args.classes_csv)
    cls_per_neuron = cls_df["klass"].to_numpy()
    if len(cls_per_neuron) != N:
        print(f"warning: classes_csv has {len(cls_per_neuron)} rows, model N={N}")

    in_hd = np.isin(cls_per_neuron, list(args.hd_classes))
    above_thr = sigma >= args.sigma_thr
    keep = in_hd & above_thr
    n_hd = int(in_hd.sum())
    n_keep = int(keep.sum())
    print(f"      HD-class (R/L) cells: {n_hd}/{N}")
    print(f"      with sigma >= {args.sigma_thr}: {n_keep}/{n_hd} kept")

    df_out = pd.DataFrame({
        "model_ix": np.arange(N),
        "type": type_per_neuron,
        "klass": cls_per_neuron,
        "pref_angle_deg": np.degrees(phi),
        "specificity": sigma,
        "kept": keep,
    })
    df_out.to_csv(args.csv_out, index=False)
    print(f"      wrote {args.csv_out}")

    # 5) skeletons in model order
    print(f"[5/6] loading skeletons (downsample={args.downsample})")
    model_bodyids, model_categories = _model_index_to_bodyid(args.connectome_dir)
    neurons, _cats, has_skel = _load_skeletons_in_model_order(
        args.anatomy_dir, model_bodyids, model_categories,
        downsample=args.downsample,
    )
    seg_per_neuron = _extract_per_neuron_segments(neurons)
    somas = _per_neuron_soma(neurons)
    if len(seg_per_neuron) != N:
        n_min = min(len(seg_per_neuron), N)
        seg_per_neuron = seg_per_neuron[:n_min]
        somas = somas[:n_min]
        keep = keep[:n_min]
        phi = phi[:n_min]
        cls_per_neuron = cls_per_neuron[:n_min]

    # 6) render
    print(f"[6/6] rendering")
    _render(seg_per_neuron, somas, keep, phi, cls_per_neuron,
            args.out, elev=args.elev, azim=args.azim,
            background=args.bg, cmap_name=args.cmap)


if __name__ == "__main__":
    main()
