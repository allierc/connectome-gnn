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

G = f"{os.environ['GNN_OUTPUT_ROOT']}/graphs_data/fly"
FAM = (("current",     "flyvis_noise_005_blank50_cv00",          "tab:blue"),
       ("conductance", "flyvis_flowcond_noise_005_blank50_cv00", "tab:red"))
LAB = "abcd"

Ws = [np.asarray(torch.load(f"{G}/{ds}/ode_params.pt", map_location="cpu",
                            weights_only=False)["W"]).ravel().astype(np.float64)
      for _, ds, _ in FAM]
# SHARED AXES, both columns and both rows: the point of the figure is the
# comparison, and a panel rescaled to its own data hides exactly the range
# difference being compared.
allW = np.concatenate(Ws)
xlin = (allW.min() * 1.05, allW.max() * 1.05)
allL = np.log10(np.abs(allW)[np.abs(allW) > 0])
xlog = (np.floor(allL.min()), np.ceil(allL.max()))
bins_lin = np.linspace(*xlin, 401)
bins_log = np.linspace(*xlog, 401)

fig, ax = plt.subplots(2, 2, figsize=(9.0, 5.0), sharex="col", sharey="col")
for r, ((name, ds, colour), W) in enumerate(zip(FAM, Ws)):
    neg = 100.0 * (W < 0).mean()

    a = ax[r, 0]
    a.hist(W, bins=bins_lin, color=colour, log=True)
    a.set_xlim(*xlin)
    a.axvline(0.0, color="0.4", lw=0.8)
    a.set_xlabel(r"$W_{ij}$")
    a.set_ylabel("edges")
    a.set_title(f"{LAB[2*r]}   flyvis {name}: $W_{{ij}}$, "
                f"{neg:.1f}% negative", loc="left", fontsize=10)

    b = ax[r, 1]
    pos = np.abs(W); pos = pos[pos > 0]
    b.hist(np.log10(pos), bins=bins_log, color=colour, log=True)
    b.set_xlim(*xlog)
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
