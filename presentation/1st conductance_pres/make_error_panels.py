#!/usr/bin/env python
"""Recovery-error distributions for W, tau and V_rest at three model-noise levels.

One figure per derivative-target vintage: the pre-fix runs (unified_cv*) and the
post-fix ones (unified_fd_cv*), so the two can be read against each other. Each
panel overlays the three noise levels; folds are pooled.

Error is learned minus true, from the run's own results/corrected_W.pt and
results/learned_ode_params.pt against the generator's ode_params.pt.
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

set_data_root(os.environ["GNN_OUTPUT_ROOT"])
LOG = f"{os.environ['GNN_OUTPUT_ROOT']}/log/fly"
LEVELS = (("noise_free", r"$\sigma = 0$",    "tab:blue"),
          ("noise_005",  r"$\sigma = 0.05$", "tab:green"),
          ("noise_05",   r"$\sigma = 0.5$",  "tab:red"))
# (learned key, truth key, label, symmetric plotting range)
QTY = (("W",      "W",      r"$\hat W_{ij} - W_{ij}$",                      0.5),
       ("tau_i",  "tau_i",  r"$\hat\tau_i - \tau_i$",                       0.05),
       ("V_i_rest", "V_i_rest", r"$\hat V^{\mathrm{rest}}_i - V^{\mathrm{rest}}_i$", 2.0))


def collect(stem, level):
    """Pooled (learned - true) per quantity over the folds of one noise level."""
    out = {k: [] for k, _, _, _ in QTY}
    for d in sorted(glob.glob(f"{LOG}/archive_*/flyvis_{level}_blank50_{stem}_cv0[0-4]")):
        res = f"{d}/results/learned_ode_params.pt"
        if not os.path.exists(res):
            continue
        learned = torch.load(res, map_location="cpu", weights_only=False)
        ds = f"fly/flyvis_{level}_blank50_cv{os.path.basename(d)[-2:]}"
        truth_p = graphs_data_path(ds, "ode_params.pt")
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


def draw(stem, _title, out_png):
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for a, (k, _, lab, lim) in zip(ax, QTY):
        for level, llab, colour in LEVELS:
            e = collect(stem, level)[k]
            if e.size == 0:
                continue
            e = e[np.isfinite(e)]
            a.hist(np.clip(e, -lim, lim), bins=160, range=(-lim, lim), log=True,
                   histtype="step", lw=1.2, color=colour, label=llab)
        a.axvline(0.0, color="0.5", lw=0.8)
        a.set_xlabel(lab)
        a.set_ylabel("count")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out_png)


draw("unified",    "chain target",           os.path.join(HERE, "Fig", "errors_before.png"))
draw("unified_fd", "finite-difference target", os.path.join(HERE, "Fig", "errors_after.png"))
