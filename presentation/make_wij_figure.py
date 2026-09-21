#!/usr/bin/env python
"""Wij distributions of the two flyvis generators, 2x2.

Left column linear, right column |W| on a log axis: the left says where the
SIGN lives, the right says the two families span the same magnitudes. Red and
blue because these are two distinct sources, not ground truth against a
prediction.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

G = "/groups/saalfeld/home/allierc/GraphData/graphs_data/fly"
FAM = (("current",     "flyvis_noise_005_blank50_cv00",          "tab:blue"),
       ("conductance", "flyvis_flowcond_noise_005_blank50_cv00", "tab:red"))
LAB = "abcd"

fig, ax = plt.subplots(2, 2, figsize=(9.0, 5.0))
for r, (name, ds, colour) in enumerate(FAM):
    W = np.asarray(torch.load(f"{G}/{ds}/ode_params.pt", map_location="cpu",
                              weights_only=False)["W"]).ravel().astype(np.float64)
    neg = 100.0 * (W < 0).mean()

    a = ax[r, 0]
    a.hist(W, bins=400, color=colour, log=True)
    a.axvline(0.0, color="0.4", lw=0.8)
    a.set_xlabel(r"$W_{ij}$")
    a.set_ylabel("edges")
    a.set_title(f"{LAB[2*r]}   flyvis {name}: $W_{{ij}}$, "
                f"{neg:.1f}% negative", loc="left", fontsize=10)

    b = ax[r, 1]
    pos = np.abs(W); pos = pos[pos > 0]
    b.hist(np.log10(pos), bins=400, color=colour, log=True)
    b.set_xlabel(r"$\log_{10} |W_{ij}|$")
    b.set_ylabel("edges")
    b.set_title(f"{LAB[2*r+1]}   median $|W_{{ij}}|$ = {np.median(np.abs(W)):.4f}, "
                f"max {W.max():.3f}", loc="left", fontsize=10)

for a in ax.ravel():
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig", "Wij_distributions.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
