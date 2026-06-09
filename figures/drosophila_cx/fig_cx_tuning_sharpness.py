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
        --model drosophila_cx_pi_epg_no_tv_cv0
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
    """Full width at half maximum of a unit-peak curve. Assumes the
    peak is near the centre of the grid (after _centre_and_normalise)
    and the curve decreases monotonically away from the peak on
    either side, so we can walk left/right without modular wrap."""
    K = len(theta_grid)
    i_peak = int(np.argmax(curve_unit_peak))
    if curve_unit_peak[i_peak] < 0.5:
        return float("nan")
    half = 0.5

    def _interp_cross(i_lo, i_hi):
        """Linear-interpolate the theta at which the curve crosses 0.5
        between grid indices i_lo (further from peak) and i_hi (closer)."""
        v_lo, v_hi = curve_unit_peak[i_lo], curve_unit_peak[i_hi]
        denom = v_hi - v_lo
        if abs(denom) < 1e-9:
            return theta_grid[i_lo]
        t = (half - v_lo) / denom
        return theta_grid[i_lo] + t * (theta_grid[i_hi] - theta_grid[i_lo])

    left = None
    for j in range(1, K):
        i = i_peak - j
        if i < 0:
            break
        if curve_unit_peak[i] <= half:
            left = _interp_cross(i, i + 1)
            break

    right = None
    for j in range(1, K):
        i = i_peak + j
        if i >= K:
            break
        if curve_unit_peak[i] <= half:
            right = _interp_cross(i, i - 1)
            break

    if left is None or right is None:
        return float("nan")
    return float(np.degrees(right - left))


MODELS = [
    ("drosophila_cx_pi_epg_no_tv_cv0",        "Known-ODE"),
    ("drosophila_cx_pi_gnn_epg_no_tv_cv0",    "GNN"),
    ("drosophila_cx_pi_fc_epg_cv0",           "fully connected"),
    ("drosophila_cx_pi_frozen_Wrec_epg_cv0",  "frozen $W^{\\mathrm{rec}}$"),
]


def _compute(net, cfg_name, four_classes_dir, n_steps, seed, n_bins, spec_cut):
    """Return all per-row quantities needed to render one condition."""
    type_names = list(net.type_names)
    nt = np.asarray(net.neuron_types).astype(int)
    N = len(nt)

    h, theta = _run_ou(net, n_steps, torch.device("cpu"), seed)
    tc, centres, _ = _tuning_curves(h, theta, n_bins=n_bins)
    spec_all, phi_all = _specificity_and_angle(tc, centres)

    fc_csv = os.path.join(four_classes_dir,
                          f"fig_cx_four_classes__{cfg_name}.csv")
    if os.path.exists(fc_csv):
        df_cls = pd.read_csv(fc_csv)
        RLmask = df_cls["klass"].isin(["R", "L"]).to_numpy()
        if len(RLmask) != N:
            RLmask = spec_all >= np.median(spec_all)
    else:
        # FC / frozen do not have a four_classes csv; use spec-median.
        RLmask = spec_all >= np.median(spec_all)

    n_RL = int(RLmask.sum())
    spec_RL = spec_all[RLmask]
    phi_RL = phi_all[RLmask]
    tc_RL = tc[RLmask]

    keep = spec_RL >= spec_cut
    n_keep = int(keep.sum())
    cut_used = float(spec_cut)
    if n_keep < max(5, n_RL // 5):
        cut_used = float(np.quantile(spec_RL, 0.50)) if n_RL else 0.0
        keep = spec_RL >= cut_used
        n_keep = int(keep.sum())

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
    kappa = _fit_kappa(mean_curve, theta_grid_centred) if n_keep else 0.0
    fwhm = _fwhm(mean_curve, theta_grid_centred) if n_keep else float("nan")

    centred_RL = tc_RL - tc_RL.mean(axis=1, keepdims=True)

    return dict(
        N=N, n_RL=n_RL, n_keep=n_keep, cut=cut_used,
        spec_all=spec_all, phi_all=phi_all,
        spec_RL=spec_RL, phi_RL=phi_RL,
        centres=centres, theta_grid_centred=theta_grid_centred,
        norm_curves=norm_curves, mean_curve=mean_curve,
        q25=q25, q75=q75, kappa=kappa, fwhm=fwhm,
        centred_RL=centred_RL, keep=keep,
        type_names=type_names, neuron_types=nt,
    )


def _draw_row(axes_row, data, spec_cut, n_bins, letter_offset):
    """Render the four panels for one condition into the given (ax_a..ax_d)."""
    spec_RL = data["spec_RL"]; phi_RL = data["phi_rad"] if "phi_rad" in data else data["phi_RL"]
    keep = data["keep"]; centres = data["centres"]
    centred_RL = data["centred_RL"]; n_RL = data["n_RL"]; n_keep = data["n_keep"]
    norm_curves = data["norm_curves"]; mean_curve = data["mean_curve"]
    q25 = data["q25"]; q75 = data["q75"]
    theta_grid_centred = data["theta_grid_centred"]
    kappa = data["kappa"]; fwhm = data["fwhm"]; cut_used = data["cut"]
    ax_a, ax_b, ax_c, ax_d = axes_row

    # (a) example tuning curves
    if keep.sum() >= 2:
        order = np.argsort(np.abs(spec_RL - np.median(spec_RL[keep])))
        ex1 = order[0]
        ex2_cands = order[1:]
        ex2 = ex2_cands[np.argmax(np.abs(
            ((phi_RL[ex2_cands] - phi_RL[ex1] + math.pi) % (2 * math.pi))
            - math.pi))]
    elif len(spec_RL):
        ex1 = ex2 = 0
    else:
        ex1 = ex2 = -1
    nonspec_cands = np.where(spec_RL < 0.3)[0]
    ex3 = (nonspec_cands[0] if len(nonspec_cands)
           else (int(np.argmin(spec_RL)) if len(spec_RL) else -1))
    for ex, color in [(ex1, "tab:blue"), (ex2, "tab:green"), (ex3, "0.5")]:
        if ex < 0:
            continue
        ax_a.plot(np.degrees(centres), centred_RL[ex], lw=1.2, color=color)
        ax_a.axvline(np.degrees(phi_RL[ex]),
                      ls="--", color=color, lw=0.5, alpha=0.6)
    ax_a.axhline(0, ls=":", color="0.5", lw=0.4)
    ax_a.set_xlabel("heading (deg)", fontsize=8)
    ax_a.set_ylabel(r"$\bar r_i - \langle\bar r_i\rangle$", fontsize=8)
    ax_a.tick_params(labelsize=7)

    # (b) specificity histogram
    if len(spec_RL):
        ax_b.hist(spec_RL, bins=np.linspace(0, 1, 31),
                   color="0.4", edgecolor="white", linewidth=0.3)
    ax_b.axvline(cut_used, ls="--", color="red", lw=1.0)
    ax_b.set_xlabel(r"specificity $s_i$", fontsize=8)
    ax_b.set_ylabel("# cells", fontsize=8)
    ax_b.tick_params(labelsize=7)

    # (c) centred normalised curves
    if n_keep > 0:
        for c in norm_curves:
            ax_c.plot(np.degrees(theta_grid_centred), c, lw=0.3,
                       color="0.7", alpha=0.5)
        ax_c.fill_between(np.degrees(theta_grid_centred), q25, q75,
                           color="red", alpha=0.18)
        ax_c.plot(np.degrees(theta_grid_centred), mean_curve,
                   color="red", lw=1.6)
        ax_c.plot(np.degrees(theta_grid_centred),
                   _von_mises_unit(theta_grid_centred, kappa),
                   color="black", lw=1.0, ls="--")
    ax_c.set_xlabel(r"$\theta - \phi_i$ (deg)", fontsize=8)
    ax_c.set_ylabel("normalised rate", fontsize=8)
    ax_c.tick_params(labelsize=7)

    # (d) preferred-angle histogram
    phi_keep = phi_RL[keep] if len(spec_RL) else np.array([])
    if len(phi_keep):
        edges = np.linspace(-math.pi, math.pi, n_bins + 1)
        ax_d.hist(np.degrees(phi_keep), bins=np.degrees(edges),
                   color="0.4", edgecolor="white", linewidth=0.3)
        ax_d.axhline(len(phi_keep) / n_bins, ls="--", color="red", lw=1.0)
    ax_d.set_xlabel(r"$\phi_i$ (deg)", fontsize=8)
    ax_d.set_ylabel("# cells", fontsize=8)
    ax_d.tick_params(labelsize=7)

    # Bold single-letter panel labels (a, b, c, ... in row-major order)
    letters = "abcdefghijklmnopqrstuvwxyz"
    for k, ax in enumerate(axes_row):
        ax.text(-0.18, 1.06, letters[letter_offset + k],
                 transform=ax.transAxes,
                 fontsize=12, fontweight="bold", va="bottom", ha="right")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--four_classes_dir", default=here,
                   help="directory containing fig_cx_four_classes__<config>.csv files")
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

    rows = []
    for cfg, label in MODELS:
        print(f"[{label}] loading + scoring")
        net = _load(cfg, device)
        data = _compute(net, cfg, args.four_classes_dir,
                         args.n_steps, args.seed, args.n_bins, args.spec_cut)
        print(f"  |R cup L| = {data['n_RL']} of {data['N']};  "
              f"kept {data['n_keep']} at spec >= {data['cut']:.3f};  "
              f"FWHM = {data['fwhm']:.0f} deg;  kappa = {data['kappa']:.2f}")
        rows.append((cfg, label, data))

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16.0, 2.6 * n_rows),
                              squeeze=False)
    for r, (cfg, label, data) in enumerate(rows):
        _draw_row(axes[r], data, args.spec_cut, args.n_bins,
                   letter_offset=4 * r)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.04,
                         hspace=0.55, wspace=0.45)

    # Bold condition label on the left edge of each row.
    for r, (cfg, label, data) in enumerate(rows):
        bbox = axes[r, 0].get_position()
        y = (bbox.y0 + bbox.y1) / 2
        fig.text(0.025, y, label,
                  rotation=90, va="center", ha="center",
                  fontsize=11, fontweight="bold")
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")

    # CSV: stack per-cell rows across conditions with a condition column.
    csv_rows = []
    for cfg, label, data in rows:
        df = pd.DataFrame({
            "condition": cfg,
            "model_ix":  np.arange(data["N"]),
            "spec":      data["spec_all"],
            "phi_rad":   data["phi_all"],
        })
        csv_rows.append(df)
    df_all = pd.concat(csv_rows, ignore_index=True)
    df_all.to_csv(args.csv_out, index=False)
    print(f"wrote {args.csv_out}")


if __name__ == "__main__":
    main()
