"""Tuning-curve sharpness of the HD-coding cells in the trained
zebrafish_hd_si_dipn RNN.

Illustrates the claim that the preferred-heading code is sharp in
firing-rate space, even though it is anatomically scrambled
(fig_zebrafish_pref_angle.py). For every cell in $\\mathsf{R} \\cup
\\mathsf{L}$ above the specificity threshold $\\specf^\\star = 0.70$ we
shift its tuning curve so the peak sits at $0^\\circ$, normalise to
unit peak, and overlay all curves. The mean centred profile is a
clean von-Mises-like bump roughly $90^\\circ$ wide at half maximum.

Two panels:
  (a) histogram of $\\specf_i$ over $\\cls{R}\\cup\\cls{L}$, showing the
      tight cluster at $0.75$ that motivates the $\\specf^\\star=0.70$
      cut.
  (b) overlay of every cell's centred tuning curve (thin grey) plus
      the population mean and 25--75 percentile band in red, with a
      reference von-Mises fit shown in dashed black.
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
)
from fig_zebrafish_four_classes import _load_with_override
from fig_zebrafish_pref_angle import (
    _tuning_curves, _preferred_angle_and_specificity,
)
from connectome_gnn.utils import load_data_root_from_json, set_data_root


def _von_mises_unit(theta_grid, kappa):
    """Unit-peak von Mises centred at 0 with concentration kappa."""
    c = np.cos(theta_grid)
    return np.exp(kappa * (c - 1.0))


def _fit_kappa_to_mean(mean_curve, theta_grid):
    """Pick the kappa that minimises L2 error of unit von Mises against
    the supplied centred mean curve (lazy grid search; the answer feeds
    only the dashed-reference overlay, not the data)."""
    norm = mean_curve / mean_curve.max()
    candidates = np.linspace(0.2, 10.0, 200)
    err = [(k, np.mean((_von_mises_unit(theta_grid, k) - norm) ** 2))
           for k in candidates]
    return min(err, key=lambda x: x[1])[0]


def _half_max_width_deg(profile, theta_grid_deg):
    """FWHM of a unimodal profile by linear interpolation on its
    centred, max-normalised curve."""
    y = profile / profile.max()
    above = y >= 0.5
    if not above.any():
        return float("nan")
    ix = np.where(above)[0]
    return float(theta_grid_deg[ix[-1]] - theta_grid_deg[ix[0]])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="zebrafish_hd_si_dipn")
    p.add_argument("--classes_csv",
                   default=os.path.join(here, "fig_zebrafish_four_classes.csv"))
    p.add_argument("--hd_classes", nargs="+", default=["R", "L"])
    p.add_argument("--rollout", default="periodic",
                   choices=["const", "periodic"])
    p.add_argument("--n_steps", type=int, default=10000)
    p.add_argument("--burn_in_s", type=float, default=4.0)
    p.add_argument("--n_theta", type=int, default=36)
    p.add_argument("--theta0", type=float, default=0.0)
    p.add_argument("--swim_interval_s", type=float, default=0.3)
    p.add_argument("--swim_magnitude_rad", type=float, default=0.393)
    p.add_argument("--swim_direction", default="L", choices=["L", "R"])
    p.add_argument("--omega_deg_per_s", type=float, default=90.0,
                   help="ignored unless --rollout const")
    p.add_argument("--sigma_thr", type=float, default=0.70)
    p.add_argument("--example_cell", type=int, default=None,
                   help="model-index of the cell shown in panel d "
                        "(default: cell with median specificity among "
                        "the kept HD-coding cells)")
    p.add_argument("--bg", default="white", choices=["black", "white"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=os.path.join(
        here, "fig_zebrafish_tuning_sharpness.png"))
    args = p.parse_args()

    try:
        set_data_root(load_data_root_from_json())
    except FileNotFoundError:
        pass

    device = torch.device(args.device)
    print(f"[1/4] loading model {args.model}")
    net, _ = _load_with_override(args.model, device)
    dt = float(net.dt)

    print(f"[2/4] {args.rollout} rollout, n_steps={args.n_steps}")
    if args.rollout == "const":
        h, theta, *_ = _run_const(net, args.n_steps, dt,
                                   args.omega_deg_per_s,
                                   args.theta0, device)
    else:
        h, theta, *_ = _run_single_impulse(
            net, args.n_steps, dt, device,
            direction=args.swim_direction,
            magnitude_rad=args.swim_magnitude_rad,
            t_event_s=0.0,
            interval_s=args.swim_interval_s,
            theta0=args.theta0,
        )
    burn = int(args.burn_in_s / dt)
    h = h[burn:]; theta = theta[burn:]

    print(f"[3/4] tuning curves at {args.n_theta} bins")
    r_bar, _counts = _tuning_curves(h, theta, args.n_theta)
    phi, sigma = _preferred_angle_and_specificity(r_bar)

    cls_df = pd.read_csv(args.classes_csv)
    cls = cls_df["klass"].to_numpy()
    in_hd = np.isin(cls, list(args.hd_classes))
    keep = in_hd & (sigma >= args.sigma_thr)
    n_hd = int(in_hd.sum())
    n_keep = int(keep.sum())
    print(f"      {n_keep}/{n_hd} cells in R∪L pass σ ≥ {args.sigma_thr}")

    # Centre each cell's curve on its preferred angle: bin-shift so the
    # max sits at index n_theta//2 (=0 deg), then normalise to unit
    # peak. Operate on the centred curves (mean-subtracted) so cells
    # with strong DC offset don't dominate the population mean.
    K = args.n_theta
    theta_grid_deg = (np.arange(K) - K // 2) * (360.0 / K)
    centred = r_bar - r_bar.mean(axis=1, keepdims=True)
    centred_kept = centred[keep]
    if not len(centred_kept):
        sys.exit("no cells pass the specificity threshold; nothing to plot")
    peak_ix = centred_kept.argmax(axis=1)
    shifted = np.zeros_like(centred_kept)
    for i, p_i in enumerate(peak_ix):
        roll = (K // 2) - int(p_i)
        shifted[i] = np.roll(centred_kept[i], roll)
    norm = shifted / shifted.max(axis=1, keepdims=True)

    pop_mean = norm.mean(axis=0)
    pop_lo = np.percentile(norm, 25, axis=0)
    pop_hi = np.percentile(norm, 75, axis=0)
    fwhm = _half_max_width_deg(pop_mean, theta_grid_deg)
    print(f"      population-mean FWHM ≈ {fwhm:.0f} deg")

    # von Mises reference fit for visual comparison.
    theta_grid_rad = np.deg2rad(theta_grid_deg)
    kappa = _fit_kappa_to_mean(pop_mean, theta_grid_rad)
    vm_ref = _von_mises_unit(theta_grid_rad, kappa)

    print("[4/4] render")
    bg = args.bg
    text_color = "white" if bg == "black" else "black"
    fig = plt.figure(figsize=(12.0, 8.6), facecolor=bg)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig,
                  left=0.07, right=0.985, top=0.96, bottom=0.08,
                  wspace=0.22, hspace=0.34)
    # Panel order: a = example single-neuron tuning curve,
    # b = specificity histogram, c = centred-curve overlay,
    # d = preferred-angle distribution.
    ax_ex = fig.add_subplot(gs[0, 0]); ax_ex.set_facecolor(bg)
    ax_hist = fig.add_subplot(gs[0, 1]); ax_hist.set_facecolor(bg)
    ax_curves = fig.add_subplot(gs[1, 0]); ax_curves.set_facecolor(bg)
    ax_phi = fig.add_subplot(gs[1, 1]); ax_phi.set_facecolor(bg)

    # (a) specificity histogram across R∪L
    ax_hist.hist(sigma[in_hd], bins=40, range=(0.0, 1.0),
                  color=(0.55, 0.55, 0.55), edgecolor=text_color, lw=0.4)
    ax_hist.axvline(args.sigma_thr, color="#e41a1c", lw=1.4,
                     linestyle="--", label=fr"$\sigma^\star = {args.sigma_thr}$")
    ax_hist.set_xlabel("specificity  $\\mathcal{T}_i$", color=text_color)
    ax_hist.set_ylabel("# cells", color=text_color)
    ax_hist.set_title(f"specificity across "
                       fr"$\mathsf{{R}}\cup\mathsf{{L}}$ (n={n_hd})",
                       color=text_color, fontsize=10)
    ax_hist.tick_params(colors=text_color, labelsize=9)
    for s_ in ax_hist.spines.values():
        s_.set_color(text_color)
    leg = ax_hist.legend(loc="upper left", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(text_color)
    ax_hist.text(-0.12, 1.06, "b", transform=ax_hist.transAxes,
                  ha="left", va="bottom", fontsize=15, fontweight="bold",
                  color=text_color)

    # (b) overlay of centred normalised tuning curves
    for i in range(len(norm)):
        ax_curves.plot(theta_grid_deg, norm[i],
                        color=(0.45, 0.45, 0.45), lw=0.3, alpha=0.25)
    ax_curves.fill_between(theta_grid_deg, pop_lo, pop_hi,
                            color=(1.0, 0.3, 0.3), alpha=0.30,
                            label="25–75 %ile")
    ax_curves.plot(theta_grid_deg, pop_mean,
                    color=(0.85, 0.10, 0.10), lw=2.0,
                    label="population mean")
    ax_curves.plot(theta_grid_deg, vm_ref,
                    color=text_color, lw=1.0, linestyle="--",
                    label=fr"von Mises fit  $\kappa={kappa:.1f}$")
    ax_curves.axhline(0.5, color=text_color, lw=0.4, alpha=0.4)
    ax_curves.set_xlim(-180, 180)
    ax_curves.set_ylim(-0.4, 1.05)
    ax_curves.set_xlabel(r"$\theta - \phi_i$  (deg)", color=text_color)
    ax_curves.set_ylabel(r"normalised firing rate", color=text_color)
    ax_curves.set_xticks([-180, -90, 0, 90, 180])
    ax_curves.set_title(
        f"centred tuning curves   (n={n_keep},  FWHM ≈ {fwhm:.0f}°)",
        color=text_color, fontsize=10)
    ax_curves.tick_params(colors=text_color, labelsize=9)
    for s_ in ax_curves.spines.values():
        s_.set_color(text_color)
    leg = ax_curves.legend(loc="upper right", frameon=False, fontsize=8)
    for txt in leg.get_texts():
        txt.set_color(text_color)
    ax_curves.text(-0.10, 1.06, "c", transform=ax_curves.transAxes,
                    ha="left", va="bottom", fontsize=15, fontweight="bold",
                    color=text_color)

    # (c) histogram of preferred angles over kept HD-coding cells. Uses
    # the same K=n_theta bins as the tuning-curve grid so the bin centres
    # match the theta_grid_deg axis of panel b.
    phi_deg_kept = np.rad2deg(phi[keep])
    edges_deg = np.linspace(-180.0, 180.0, K + 1)
    counts_phi, _ = np.histogram(phi_deg_kept, bins=edges_deg)
    centres_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    bar_w = 360.0 / K * 0.9
    mean_count = counts_phi.mean() if K else 0.0
    ax_phi.bar(centres_deg, counts_phi, width=bar_w,
               color=(0.55, 0.55, 0.55), edgecolor=text_color, lw=0.4)
    ax_phi.axhline(mean_count, color="#e41a1c", lw=1.4, linestyle="--",
                   label=fr"uniform: {mean_count:.1f}/bin")
    ax_phi.set_xlim(-180, 180)
    ax_phi.set_xticks([-180, -90, 0, 90, 180])
    ax_phi.set_xlabel(r"preferred angle  $\phi_i$  (deg)", color=text_color)
    ax_phi.set_ylabel("# HD-coding cells", color=text_color)
    ax_phi.set_title(
        fr"preferred-angle distribution   "
        fr"($n={n_keep}$, {K} bins of "
        fr"{360.0 / K:.0f}$^\circ$)",
        color=text_color, fontsize=10)
    ax_phi.tick_params(colors=text_color, labelsize=9)
    for s_ in ax_phi.spines.values():
        s_.set_color(text_color)
    leg = ax_phi.legend(loc="upper right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(text_color)
    ax_phi.text(-0.12, 1.06, "d", transform=ax_phi.transAxes,
                ha="left", va="bottom", fontsize=15, fontweight="bold",
                color=text_color)

    # Chi-squared deviation from uniform (printed in console only).
    if K > 0 and counts_phi.sum() > 0:
        exp = counts_phi.sum() / K
        chi2 = float(((counts_phi - exp) ** 2 / max(exp, 1e-9)).sum())
        print(f"      preferred-angle chi^2 vs uniform "
              f"(K={K} bins, df={K-1}): {chi2:.1f}  "
              f"min={int(counts_phi.min())}  "
              f"max={int(counts_phi.max())}  "
              f"mean={mean_count:.1f}")

    # (a) three example neurons spanning the range:
    #   - two specific cells (sigma >= sigma*) with different preferred
    #     angles, picked near the median sigma of the kept population;
    #   - one non-specific cell (sigma << sigma*) from R u L, picked
    #     near the median sigma of the discarded cells.
    kept_ix = np.where(keep)[0]
    sigma_kept = sigma[kept_ix]
    med_kept = float(np.median(sigma_kept))
    # specific cell 1: closest to median sigma among kept
    ex1_ix = int(kept_ix[np.argmin(np.abs(sigma_kept - med_kept))])
    ex1_phi = float(phi[ex1_ix])
    # specific cell 2: among kept cells whose sigma is also near median,
    # pick the one whose preferred angle is farthest from cell 1's phi.
    near_med = kept_ix[np.abs(sigma_kept - med_kept) < 0.05]
    if len(near_med) > 1:
        dphi = np.abs(np.angle(
            np.exp(1j * (phi[near_med] - ex1_phi))))
        ex2_ix = int(near_med[np.argmax(dphi)])
    else:
        ex2_ix = int(kept_ix[np.argsort(-sigma_kept)[0]])
    # non-specific cell: clearly below threshold (sigma ~ 0.2), picked
    # from anywhere in the population so the flat shape is unambiguous.
    target_low_sigma = 0.20
    ex3_ix = int(np.argmin(np.abs(sigma - target_low_sigma)))

    edges_theta = np.linspace(-180.0, 180.0, K + 1)
    bin_centres = 0.5 * (edges_theta[:-1] + edges_theta[1:])
    cells = [
        (ex1_ix, "#0072b2", "specific A"),
        (ex2_ix, "#009e73", "specific B"),
        (ex3_ix, "#bbbbbb", "non-specific"),
    ]
    # Plot each cell with its mean subtracted so the three curves
    # share a zero baseline; without this the per-cell DC offsets
    # dominate the y-axis and the bumps are invisible. Relative
    # amplitudes are preserved (no per-cell normalisation).
    ax_ex.axhline(0.0, color=text_color, lw=0.5, alpha=0.4,
                  linestyle=":")
    for cell_ix, color, tag in cells:
        phi_deg = float(np.rad2deg(phi[cell_ix]))
        sig = float(sigma[cell_ix])
        curve = r_bar[cell_ix] - r_bar[cell_ix].mean()
        ax_ex.plot(bin_centres, curve,
                   color=color, lw=1.5, marker="o", markersize=3,
                   label=f"{tag}: cell {cell_ix}, "
                         fr"$\mathcal{{T}} = {sig:.2f}$, "
                         fr"$\phi = {phi_deg:+.0f}^\circ$")
        ax_ex.axvline(phi_deg, color=color, lw=1.0, linestyle="--",
                      alpha=0.6)
        print(f"      panel-a {tag}: idx={cell_ix} "
              f"phi={phi_deg:+.1f} deg  sigma={sig:.2f}  "
              f"amp(p2p)={float(curve.max() - curve.min()):.3f}")
    ax_ex.set_xlim(-180, 180)
    ax_ex.set_xticks([-180, -90, 0, 90, 180])
    ax_ex.set_xlabel(r"heading  $\theta$  (deg)", color=text_color)
    ax_ex.set_ylabel(r"$\bar r_i(\theta) - \langle\bar r_i\rangle$",
                     color=text_color)
    ax_ex.set_title("example tuning curves: two specific cells "
                    "and one non-specific",
                    color=text_color, fontsize=10)
    ax_ex.tick_params(colors=text_color, labelsize=9)
    for s_ in ax_ex.spines.values():
        s_.set_color(text_color)
    leg = ax_ex.legend(loc="best", frameon=False, fontsize=7)
    for txt in leg.get_texts():
        txt.set_color(text_color)
    ax_ex.text(-0.12, 1.06, "a", transform=ax_ex.transAxes,
               ha="left", va="bottom", fontsize=15, fontweight="bold",
               color=text_color)

    fig.savefig(args.out, dpi=200, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}  (FWHM ≈ {fwhm:.0f}°)")


if __name__ == "__main__":
    main()
