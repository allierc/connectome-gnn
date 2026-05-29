"""Tuning-curve sharpness of the HD-coding cells in the trained
CX models. Drosophila-CX analogue of
figures/zebrafish/fig_zebrafish_tuning_sharpness.py.

For every cell in the R \cup L classes of fig_cx_four_classes (HD-MI
above the median) we compute its tuning curve r_i(theta_k) on an OU
rollout, the specificity

    spec_i = |Sum_k (rbar_i(k) - <rbar>) exp(i theta_k)|
             / Sum_k |rbar_i(k) - <rbar>|

and the preferred angle

    phi_i = arg(Sum_k (rbar_i(k) - <rbar>) exp(i theta_k)) .

Four panels:
    (a)  three example tuning curves -- two specific, one flat
    (b)  histogram of spec_i over R \cup L  (red dashed cut spec >= 0.7)
    (c)  centred + unit-peak-normalised curves overlay with population
         mean and 25--75 percentile band, von-Mises reference fit
    (d)  histogram of preferred angles phi_i across the surviving
         R \cup L cells (uniform-coverage line at n/K)

Requires:
    fig_cx_four_classes__<config>.csv  (produced by
    fig_cx_four_classes.py) for the R \cup L cell list.

CLI:
    python figures/drosophila_cx/fig_cx_tuning_sharpness.py \
        --model drosophila_cx_pi_epg_tv_cv0
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from connectome_gnn.utils import log_path, load_data_root_from_json, set_data_root
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.models.registry import create_model
from connectome_gnn.generators.utils import generate_path_integration_batch


def _load(config_name, device, prefer_epoch=None):
    config, _ = load_run_config(config_name, explicit_output_root=False, task="train")
    ckpt_dir = os.path.join(log_path(config.config_file), "models")
    cands = sorted(
        glob.glob(os.path.join(ckpt_dir, "best_model_with_0_graphs_*.pt")),
        key=lambda p_: int(p_.rsplit("_", 1)[1].rstrip(".pt")),
    )
    if not cands:
        raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
    if prefer_epoch is None and "gnn_epg" in config_name:
        prefer_epoch = 3
    ckpt_path = cands[-1]
    if prefer_epoch is not None:
        match = [p_ for p_ in cands
                 if int(p_.rsplit("_", 1)[1].rstrip(".pt")) == prefer_epoch]
        if match:
            ckpt_path = match[0]
    model = create_model(
        config.graph_model.signal_model_name,
        aggr_type=config.graph_model.aggr_type,
        config=config, device=device,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"loaded {config_name}: {ckpt_path}")
    return model


def _run_ou(net, n_steps, device, seed):
    rng = np.random.default_rng(seed)
    batch = generate_path_integration_batch(
        batch_size=1, n_steps=n_steps,
        dt=float(net.dt), device=device, rng=rng,
    )
    theta = batch.theta_hd[0].cpu().numpy()
    with torch.no_grad():
        _, h = net(batch.stimulus)
    return h[0].cpu().numpy(), theta


def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def _tuning_curves(h_traj, theta, n_bins=36):
    """Per-neuron mean firing rate per heading bin. Returns
    (N, n_bins) array of bin means in firing-rate units, and the
    bin-centre angles (radians).

    ``theta`` is the unwrapped heading trajectory (radians) emitted by
    the OU batch generator; we wrap it to (-pi, pi] before binning so
    bins cover the whole circle uniformly."""
    theta_wrap = ((theta + math.pi) % (2 * math.pi)) - math.pi
    edges = np.linspace(-math.pi, math.pi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_ix = np.digitize(theta_wrap, edges) - 1
    bin_ix = np.clip(bin_ix, 0, n_bins - 1)
    r = _sigmoid(h_traj)            # (T, N)
    N = r.shape[1]
    tc = np.zeros((N, n_bins), dtype=np.float32)
    counts = np.bincount(bin_ix, minlength=n_bins)
    for k in range(n_bins):
        m = bin_ix == k
        if m.any():
            tc[:, k] = r[m].mean(axis=0)
    return tc, centres, counts


def _specificity_and_angle(tc, centres):
    """Return (spec, phi) per neuron from the centred tuning curve."""
    centred = tc - tc.mean(axis=1, keepdims=True)
    e = np.exp(1j * centres)[None, :]   # (1, K)
    R = (centred * e).sum(axis=1)        # (N,) complex
    L1 = np.abs(centred).sum(axis=1) + 1e-12
    spec = np.abs(R) / L1
    phi = np.angle(R)
    return spec, phi


def _centre_and_normalise(tc, centres, peak_to_zero=True):
    """Shift each tuning curve so its peak sits at theta=0 and divide
    by the peak. Returns (N, K) array on the same theta-grid."""
    centred = tc - tc.mean(axis=1, keepdims=True)
    K = tc.shape[1]
    peaks = np.argmax(centred, axis=1)
    out = np.zeros_like(centred)
    for i in range(tc.shape[0]):
        shift = (K // 2) - peaks[i] if peak_to_zero else 0
        out[i] = np.roll(centred[i], shift)
        m = out[i].max()
        if m > 1e-9:
            out[i] /= m
    return out


def _von_mises_unit(theta_grid, kappa):
    return np.exp(kappa * (np.cos(theta_grid) - 1.0))


def _fit_kappa(mean_curve, theta_grid):
    norm = mean_curve / max(1e-9, mean_curve.max())
    candidates = np.linspace(0.2, 10.0, 200)
    err = [(k, np.mean((_von_mises_unit(theta_grid, k) - norm) ** 2))
           for k in candidates]
    return min(err, key=lambda x: x[1])[0]


def _fwhm(curve_unit_peak, theta_grid):
    """Full width at half maximum of a unit-peak curve on a circular
    grid, by linear interpolation around the peak."""
    K = len(theta_grid)
    i_peak = int(np.argmax(curve_unit_peak))
    if curve_unit_peak[i_peak] < 0.5:
        return float("nan")
    half = 0.5
    def _cross(lo, hi, direction):
        for j in range(K):
            i0 = (i_peak + direction * j) % K
            i1 = (i_peak + direction * (j + 1)) % K
            v0 = curve_unit_peak[i0]; v1 = curve_unit_peak[i1]
            if (v0 - half) * (v1 - half) <= 0:
                t = (half - v0) / max(1e-9, (v1 - v0))
                return theta_grid[i0] + t * (theta_grid[i1] - theta_grid[i0])
        return float("nan")
    left = _cross(None, None, -1)
    right = _cross(None, None, +1)
    if not (np.isfinite(left) and np.isfinite(right)):
        return float("nan")
    delta = right - left
    if delta < 0:
        delta += 2 * math.pi
    return float(np.degrees(delta))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="drosophila_cx_pi_epg_tv_cv0")
    p.add_argument("--four_classes_csv",
                   default=os.path.join(here,
                       "fig_cx_four_classes__drosophila_cx_pi_epg_tv_cv0.csv"))
    p.add_argument("--n_steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_bins", type=int, default=36)
    p.add_argument("--spec_cut", type=float, default=0.70)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=os.path.join(here,
                                                  "fig_cx_tuning_sharpness.png"))
    p.add_argument("--csv_out", default=os.path.join(here,
                                                      "fig_cx_tuning_sharpness.csv"))
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

    print(f"[1/5] loading model {args.model}")
    net = _load(args.model, device)
    type_names = list(net.type_names)
    nt = np.asarray(net.neuron_types).astype(int)
    tps = np.array([type_names[t] for t in nt])
    N = len(nt)

    print(f"[2/5] OU rollout n_steps={args.n_steps}")
    h, theta = _run_ou(net, args.n_steps, device, args.seed)

    print(f"[3/5] tuning curves K={args.n_bins}")
    tc, centres, _ = _tuning_curves(h, theta, n_bins=args.n_bins)
    spec_all, phi_all = _specificity_and_angle(tc, centres)

    print(f"[4/5] selecting R \cup L cells from {args.four_classes_csv}")
    if os.path.exists(args.four_classes_csv):
        df_cls = pd.read_csv(args.four_classes_csv)
        RLmask = df_cls["klass"].isin(["R", "L"]).to_numpy()
        if len(RLmask) != N:
            print(f"  warning: csv N={len(RLmask)} != model N={N}; "
                  f"falling back to MI-median split")
            RLmask = spec_all >= np.median(spec_all)
    else:
        print(f"  no csv found; using spec-median as the R \cup L proxy")
        RLmask = spec_all >= np.median(spec_all)

    n_RL = int(RLmask.sum())
    print(f"  |R \\cup L| = {n_RL} of {N}")

    spec_RL = spec_all[RLmask]
    phi_RL = phi_all[RLmask]
    tc_RL = tc[RLmask]

    keep = spec_RL >= args.spec_cut
    n_keep = int(keep.sum())
    # Fall back to the R \cup L median if too few cells pass the
    # requested cut, so the figure always renders.
    if n_keep < max(5, n_RL // 5):
        fallback = float(np.quantile(spec_RL, 0.50)) if n_RL else 0.0
        print(f"  only {n_keep} cells pass spec >= {args.spec_cut}; "
              f"falling back to spec >= {fallback:.3f} (R\\cupL median)")
        args.spec_cut = fallback
        keep = spec_RL >= args.spec_cut
        n_keep = int(keep.sum())
    print(f"  kept {n_keep}/{n_RL} cells at spec >= {args.spec_cut:.3f}")

    norm_curves = _centre_and_normalise(tc_RL[keep], centres,
                                         peak_to_zero=True)
    if n_keep:
        mean_curve = norm_curves.mean(axis=0)
        q25 = np.percentile(norm_curves, 25, axis=0)
        q75 = np.percentile(norm_curves, 75, axis=0)
    else:
        mean_curve = np.zeros_like(centres)
        q25 = np.zeros_like(centres); q75 = np.zeros_like(centres)

    theta_grid_centred = centres - centres[len(centres) // 2]
    kappa = _fit_kappa(mean_curve, theta_grid_centred)
    fwhm = _fwhm(mean_curve, theta_grid_centred)
    print(f"  kappa (vM fit) = {kappa:.2f}   FWHM = {fwhm:.1f} deg")

    print(f"[5/5] rendering {args.out}")
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.0))

    # Panel a: three example tuning curves
    ax = axes[0]
    centred_RL = tc_RL - tc_RL.mean(axis=1, keepdims=True)
    if keep.sum() >= 2:
        order = np.argsort(np.abs(spec_RL - np.median(spec_RL[keep])))
        ex1 = order[0]
        # pick another with a different preferred angle
        ex2_cands = order[1:]
        ex2 = ex2_cands[np.argmax(np.abs(
            ((phi_RL[ex2_cands] - phi_RL[ex1] + math.pi) % (2 * math.pi))
            - math.pi))]
    else:
        ex1 = ex2 = 0
    nonspec_cands = np.where(spec_RL < 0.3)[0]
    ex3 = nonspec_cands[0] if len(nonspec_cands) else np.argmin(spec_RL)
    for ex, color, label in [
        (ex1, "tab:blue", f"spec={spec_RL[ex1]:.2f}, $\\phi$={np.degrees(phi_RL[ex1]):+.0f}$^\\circ$"),
        (ex2, "tab:green", f"spec={spec_RL[ex2]:.2f}, $\\phi$={np.degrees(phi_RL[ex2]):+.0f}$^\\circ$"),
        (ex3, "0.5", f"spec={spec_RL[ex3]:.2f}, flat"),
    ]:
        ax.plot(np.degrees(centres), centred_RL[ex], lw=1.4, color=color,
                label=label)
        ax.axvline(np.degrees(phi_RL[ex]), ls="--", color=color, lw=0.6, alpha=0.7)
    ax.axhline(0, ls=":", color="0.5", lw=0.5)
    ax.set_xlabel("heading (deg)", fontsize=9)
    ax.set_ylabel(r"$\bar r_i(\theta) - \langle\bar r_i\rangle$", fontsize=9)
    ax.set_title("a  example tuning curves", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False)

    # Panel b: specificity histogram
    ax = axes[1]
    ax.hist(spec_RL, bins=np.linspace(0, 1, 31), color="0.4",
            edgecolor="white", linewidth=0.4)
    ax.axvline(args.spec_cut, ls="--", color="red", lw=1.2,
               label=f"$s^*={args.spec_cut}$")
    ax.set_xlabel(r"specificity $s_i$", fontsize=9)
    ax.set_ylabel("# cells", fontsize=9)
    ax.set_title(rf"b  $|\mathsf{{R}}\cup\mathsf{{L}}|={n_RL}$, "
                 rf"kept {n_keep}", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # Panel c: centred normalised curves overlay
    ax = axes[2]
    if n_keep > 0:
        for c in norm_curves:
            ax.plot(np.degrees(theta_grid_centred), c, lw=0.4, color="0.7",
                    alpha=0.5)
        ax.fill_between(np.degrees(theta_grid_centred), q25, q75,
                        color="red", alpha=0.18)
        ax.plot(np.degrees(theta_grid_centred), mean_curve,
                color="red", lw=2.0, label="mean")
        ax.plot(np.degrees(theta_grid_centred),
                _von_mises_unit(theta_grid_centred, kappa),
                color="black", lw=1.2, ls="--",
                label=fr"von Mises $\kappa={kappa:.1f}$")
    ax.set_xlabel(r"$\theta - \phi_i$ (deg)", fontsize=9)
    ax.set_ylabel("normalised rate", fontsize=9)
    ax.set_title(f"c  FWHM $\\approx {fwhm:.0f}^\\circ$",
                  fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # Panel d: preferred-angle histogram
    ax = axes[3]
    phi_keep = phi_RL[keep]
    if len(phi_keep):
        edges = np.linspace(-math.pi, math.pi, args.n_bins + 1)
        ax.hist(np.degrees(phi_keep), bins=np.degrees(edges),
                color="0.4", edgecolor="white", linewidth=0.4)
        ax.axhline(len(phi_keep) / args.n_bins, ls="--", color="red", lw=1.2,
                   label=f"$n/K \\approx {len(phi_keep)/args.n_bins:.1f}$")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_xlabel(r"$\phi_i$ (deg)", fontsize=9)
    ax.set_ylabel("# cells", fontsize=9)
    ax.set_title("d  preferred-angle coverage", fontsize=10, loc="left")

    fig.tight_layout()
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")

    df = pd.DataFrame({
        "model_ix": np.arange(N),
        "type":     tps,
        "spec":     spec_all,
        "phi_rad":  phi_all,
        "in_RL":    RLmask.astype(int),
        "kept":     (RLmask & (spec_all >= args.spec_cut)).astype(int),
    })
    df.to_csv(args.csv_out, index=False)
    print(f"wrote {args.csv_out}")

    # Per-cell-type summary for the kept population
    print("\n=== kept (R\\cupL & spec>=cut) per type ===")
    by_type = (df[df.kept.astype(bool)]
                .groupby("type").size().sort_values(ascending=False))
    print(by_type.to_string())


if __name__ == "__main__":
    main()
