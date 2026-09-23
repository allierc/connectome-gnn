#!/usr/bin/env python
"""Experiment 1's recovery-error distributions, both derivative targets at once.

The twin of the deck's slides 4 and 5, with one change that is the point: those
were two slides, one per derivative-target vintage, and the two vintages were
two COMMITS. Here both arms are rows of a single figure and differ by one config
key at one commit, so before and after are on the same page and comparable.

Rows are the arm, columns the quantity, and the three model-noise levels overlay
within each panel exactly as before. Folds are pooled.

Error is learned minus true, from the run's own results/learned_ode_params.pt
against the generator's ode_params.pt.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from connectome_gnn.utils import graphs_data_path, set_data_root  # noqa: E402

set_data_root("/groups/saalfeld/home/allierc/GraphData")
LOG = "/groups/saalfeld/home/allierc/GraphData/log/fly"

LEVELS = (("noise_free", r"$\sigma = 0$",    "tab:blue"),
          ("noise_005",  r"$\sigma = 0.05$", "tab:green"),
          ("noise_05",   r"$\sigma = 0.5$",  "tab:red"))
# The arms, in before-then-after order so the eye reads down the page.
ARMS = (("dtbug", "target = generator's y_list  (the published bug)"),
        ("dtfd",  "target = finite difference of the observed voltage"))
# (key in learned, key in truth, axis label, symmetric range)
QTY = (("W", "W", r"$\hat W_{ij} - W_{ij}$", 0.5),
       ("tau_i", "tau_i", r"$\hat\tau_i - \tau_i$", 0.05),
       ("V_i_rest", "V_i_rest",
        r"$\hat V^{\mathrm{rest}}_i - V^{\mathrm{rest}}_i$", 2.0))


def collect(tag, level):
    """Pooled (learned - true) per quantity over the five folds of one cell."""
    out = {k: [] for k, _, _, _ in QTY}
    for d in sorted(glob.glob(f"{LOG}/flyvis_{level}_blank50_{tag}_cv0[0-4]")):
        res = f"{d}/results/learned_ode_params.pt"
        if not os.path.exists(res):
            continue
        learned = torch.load(res, map_location="cpu", weights_only=False)
        truth_p = graphs_data_path(
            f"fly/flyvis_{level}_blank50_cv{os.path.basename(d)[-2:]}",
            "ode_params.pt")
        if not os.path.exists(truth_p):
            continue
        truth = torch.load(truth_p, map_location="cpu", weights_only=False)
        for k, tk, _, _ in QTY:
            if k not in learned or tk not in truth:
                continue
            a = np.asarray(learned[k], dtype=np.float64).ravel()
            b = np.asarray(truth[tk], dtype=np.float64).ravel()
            n = min(a.size, b.size)
            out[k].append(a[:n] - b[:n])
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}


def main():
    fig, axes = plt.subplots(len(ARMS), len(QTY),
                             figsize=(10.5, 2.9 * len(ARMS)))
    for r, (tag, arm_label) in enumerate(ARMS):
        for c, (k, _tk, lab, lim) in enumerate(QTY):
            ax = axes[r, c]
            for level, llab, colour in LEVELS:
                e = collect(tag, level)[k]
                if e.size == 0:
                    continue
                e = e[np.isfinite(e)]
                ax.hist(np.clip(e, -lim, lim), bins=160, range=(-lim, lim),
                        log=True, histtype="step", lw=1.2, color=colour,
                        label=llab)
            ax.axvline(0.0, color="0.5", lw=0.8)
            ax.set_xlabel(lab)
            if c == 0:
                ax.set_ylabel("count")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        # The arm named ABOVE its row, left-aligned, not bold, no panel titles.
        axes[r, 0].text(0.0, 1.06, arm_label, transform=axes[r, 0].transAxes,
                        ha="left", va="bottom", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "Fig", "exp01_errors.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
