#!/usr/bin/env python
"""Observed voltage at three measurement-noise levels, same cells and frames.

The three datasets share the generator seed, so the underlying trajectory is the
same trace in all three panels: only the measurement noise eta the model is fed
differs. Plots v + eta, which is what training sees, not the clean v.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from connectome_gnn.utils import graphs_data_path, set_data_root, to_numpy  # noqa: E402
from connectome_gnn.zarr_io import load_simulation_data  # noqa: E402

set_data_root(os.environ["GNN_OUTPUT_ROOT"])
SETS = (("no meas. noise", "flyvis_noise_005_blank50_cv00", 0.0),
        (r"$\gamma = 0.1$", "flyvis_noise_005_010_blank50_cv00", 0.1),
        (r"$\gamma = 0.2$", "flyvis_noise_005_020_blank50_cv00", 0.2))
T0, T1 = 500, 1000          # 500 frames = 10 s at dt = 20 ms
CELLS = [2895, 1089, 221, 655, 872, 3329]

fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=True)
for a, (label, ds, gamma) in zip(ax, SETS):
    x = load_simulation_data(graphs_data_path("fly/" + ds, "x_list_train"),
                             fields=["voltage", "noise"] if gamma > 0 else ["voltage"])
    v = to_numpy(x.voltage)[T0:T1, CELLS]
    if gamma > 0 and getattr(x, "noise", None) is not None:
        v = v + to_numpy(x.noise)[T0:T1, CELLS]
    t = np.arange(T0, T1) * 0.02
    for k in range(v.shape[1]):
        a.plot(t, v[:, k] + 2.5 * k, lw=0.6, color="tab:green")
    a.set_title(label, fontsize=10)
    a.set_xlabel("time (s)")
    a.set_yticks([])
    for sp in ("top", "right", "left"):
        a.spines[sp].set_visible(False)
ax[0].set_ylabel("neurons")
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig", "meas_noise_panels.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
