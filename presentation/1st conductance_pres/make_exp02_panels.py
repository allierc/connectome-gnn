#!/usr/bin/env python
"""Experiment 2's recovery errors: the current form against two lasso strengths.

The twin of experiment 1's figure, with the arm as the row again -- here three
arms rather than two, because the question is not "did the target change" but
"does the general form recover what the current form does, and at which lasso".

Rows are the arm, columns the quantity, and the three model-noise levels overlay
within each panel. A row that shows only one colour is an arm whose other noise
levels have not landed yet, which is the honest picture while they train.

Reuses experiment 1's `collect`: the run names differ only in the arm tag, so
there is one reader of `learned_ode_params.pt` against `ode_params.pt` and not
two that can drift apart.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_exp01_panels import LEVELS, QTY, collect  # noqa: E402

# In the table's order: the reference first, then the lasso that works, then the
# one that collapses at sigma = 0.
ARMS = (("dtfd",     "current"),
        ("condl25",  "conductance, lasso 25"),
        ("condl100", "conductance, lasso 100"))


def draw(out_name="exp02_errors.png"):
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
    out = os.path.join(HERE, "Fig", out_name)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    draw()
