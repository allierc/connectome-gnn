"""For every modelled bump-pool neuron, find the recorded neuron whose
power spectrum best matches it, over the full ZAPBench Rotations block.

Uses :func:`connectome_gnn.metrics.best_observed_match` to score every
recorded neuron against every modelled trace in one chunked vectorised
pass. Reads the ``functional_panel_*.npz`` files dumped by
``fig_functional_panel.py`` (see commit 5793b50 onward) so the rollout
does not have to run inside this script.

Figure layout (1 PNG, no PNG montage):
    (a) Distribution of best-match spectral distances across the 300
        modelled neurons — histogram + cumulative.
    (b) Example pairs: for the 5 best, 5 median, 5 worst modelled
        neurons, overlay the modelled trace (black) and the matched
        recorded trace (green) on the first ~100 s.
    (c) Per-modelled-neuron power spectrum vs. its matched recorded
        neighbour: median + IQR overlay on log–log axes (same style as
        Figure 12 panel e).

The script is designed to be a drop-in for the *next* analysis layer:
once the new ARTR/pt-IPN1 gcamp checkpoints are trained, point
``--config`` at the trained variant and the matched-neuron search
runs against ZAPBench's full 70k+ ROI pool (whatever ``--obs-npz``
supplies).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

from connectome_gnn.metrics import (  # noqa: E402
    best_observed_match, fft_power_spectrum,
)

GT_COLOR = "#4daf4a"
PRED_COLOR = "black"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--obs-npz", required=True,
        help="path to functional_panel_real_rotation.npz dumped by "
             "fig_functional_panel.py (kino + t_sec + dt_sec)",
    )
    p.add_argument(
        "--model-npz", required=True,
        help="path to functional_panel_<config>_<gcamp>.npz dumped by "
             "fig_functional_panel.py",
    )
    p.add_argument("--metric", default="cosine",
                   choices=("cosine", "l2", "l1", "jsd"),
                   help="spectral-distance metric")
    p.add_argument(
        "--band-hz", nargs=2, type=float, default=None, metavar=("LO", "HI"),
        help="restrict spectrum comparison to a frequency band (Hz). "
             "Default: full single-sided spectrum.",
    )
    p.add_argument("--out",
                   default=os.path.join(
                       _HERE, "fig_zebrafish_best_observed_match.png"))
    args = p.parse_args()

    obs = np.load(args.obs_npz)
    mod = np.load(args.model_npz)
    obs_k = np.asarray(obs["kino"])
    mod_k = np.asarray(mod["kino"])
    dt = float(obs["dt_sec"])
    t_sec = np.asarray(obs["t_sec"])

    # Trim to shared T and drop NaN rows.
    T = min(obs_k.shape[-1], mod_k.shape[-1])
    obs_k = obs_k[:, :T]
    mod_k = mod_k[:, :T]
    obs_k = obs_k[np.isfinite(obs_k).all(axis=1)]
    mod_k = mod_k[np.isfinite(mod_k).all(axis=1)]
    print(f"[match] obs={obs_k.shape}  mod={mod_k.shape}  dt={dt:.3f}s")

    best_idx, best_score, score_mtx = best_observed_match(
        obs_k, mod_k, dt=dt, metric=args.metric,
        band_hz=tuple(args.band_hz) if args.band_hz else None,
        return_scores=True,
    )
    print(f"[match] best score: min={best_score.min():.3g}  "
          f"median={np.median(best_score):.3g}  max={best_score.max():.3g}")

    # ---- 1×3 layout ---------------------------------------------------
    fig = plt.figure(figsize=(20, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.6, 1.4],
                          left=0.04, right=0.99, top=0.96, bottom=0.10,
                          wspace=0.22)

    # (a) distance distribution + cumulative
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.hist(best_score, bins=40, color=PRED_COLOR, alpha=0.7,
              edgecolor="white", linewidth=0.4)
    ax_a.set_xlabel(f"best-match {args.metric} distance", fontsize=12)
    ax_a.set_ylabel("modelled neurons", fontsize=12)
    ax_a.tick_params(labelsize=10)
    ax_a.text(-0.10, 1.02, "a", transform=ax_a.transAxes,
              ha="right", va="bottom", fontsize=16, fontweight="bold")

    # (b) example pairs: 5 best, 5 median, 5 worst
    ax_b = fig.add_subplot(gs[0, 1])
    order = np.argsort(best_score)
    n_each = min(5, mod_k.shape[0] // 3)
    picks = np.concatenate([
        order[:n_each],
        order[len(order) // 2 - n_each // 2:
              len(order) // 2 + n_each - n_each // 2],
        order[-n_each:],
    ])
    spacing = 4.0
    t_show = t_sec[:T]
    for k, mi in enumerate(picks):
        oi = int(best_idx[mi])
        offset = -k * spacing
        m = mod_k[mi]; o = obs_k[oi]
        ax_b.plot(t_show, m + offset, color=PRED_COLOR, lw=0.7)
        ax_b.plot(t_show, o + offset, color=GT_COLOR,   lw=0.7)
        ax_b.text(
            t_show[0] - 0.02 * (t_show[-1] - t_show[0]),
            offset, f"#{mi}\n→#{oi}",
            ha="right", va="center", fontsize=8, color="0.3",
        )
    ax_b.set_yticks([])
    ax_b.set_xlabel("time (s)", fontsize=12)
    ax_b.set_title(
        "best (top) → median → worst (bottom) modelled–recorded pairs",
        fontsize=11)
    ax_b.tick_params(labelsize=10)
    ax_b.text(-0.05, 1.02, "b", transform=ax_b.transAxes,
              ha="right", va="bottom", fontsize=16, fontweight="bold")

    # (c) power spectrum overlay — modelled vs. matched recorded
    ax_c = fig.add_subplot(gs[0, 2])
    matched_obs = obs_k[best_idx]
    freqs, p_mod = fft_power_spectrum(mod_k, dt=dt)
    _,    p_obs = fft_power_spectrum(matched_obs, dt=dt)
    freqs = freqs[1:]; p_mod = p_mod[:, 1:]; p_obs = p_obs[:, 1:]
    for population, color, lbl in [
        (p_mod, PRED_COLOR, "modelled"),
        (p_obs, GT_COLOR,   "matched recorded"),
    ]:
        med = np.median(population, axis=0)
        q25 = np.percentile(population, 25, axis=0)
        q75 = np.percentile(population, 75, axis=0)
        ax_c.fill_between(freqs, q25, q75, color=color, alpha=0.18, lw=0)
        ax_c.plot(freqs, med, color=color, lw=1.6, label=lbl)
    ax_c.set_xscale("log"); ax_c.set_yscale("log")
    ax_c.set_xlabel("frequency (Hz)", fontsize=12)
    ax_c.set_ylabel(r"$|X(f)|^{2}$", fontsize=12)
    ax_c.legend(fontsize=11, frameon=False, loc="upper right")
    ax_c.tick_params(labelsize=10)
    ax_c.text(-0.08, 1.02, "c", transform=ax_c.transAxes,
              ha="right", va="bottom", fontsize=16, fontweight="bold")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
