"""Per-neuron voltage mean +- SD over the training trajectory, by cell type, with the parameter
fit's activation floor drawn as a horizontal dashed line.

    PYTHONPATH=src python tools/activity_floor_plot.py [run]

The same panel as results/extras/activity_mu_sigma.png (GNN_PlotFigure.plot_neuron_activity_analysis):
one point per neuron, mu_i +- sigma_i of v_i over every training frame, neurons in index order
(grouped by type), type names on the x axis. The dashed line is the floor on relu(v_j) the
per-edge fit applies (`tmpl_vj_floor` of the run's results/metrics.txt): on each edge a frame is
kept only when the sender's relu(v_j) is above it. Writes presentation/Fig/meeting_activity_floor_<fold>.{pdf,png}.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from connectome_gnn.metrics import INDEX_TO_NAME               # noqa: E402
from connectome_gnn.config import NeuralGraphConfig            # noqa: E402
from connectome_gnn.zarr_io import load_simulation_data        # noqa: E402

ROOT = "/groups/saalfeld/home/allierc/GraphData"
RUN = "flyvis_noise_005_blank50_condl25_cv00"


def main(run=RUN):
    log_dir = f"{ROOT}/log/fly/{run}"
    cfg = NeuralGraphConfig.from_yaml(f"{log_dir}/config.yaml")
    ds = cfg.dataset if cfg.dataset.startswith("fly/") else "fly/" + cfg.dataset
    x = load_simulation_data(f"{ROOT}/graphs_data/{ds}/x_list_train", fields=["voltage", "neuron_type"])
    floor = next(float(line.split(":")[1]) for line in open(f"{log_dir}/results/metrics.txt")
                 if line.startswith("tmpl_vj_floor:"))
    v = x.voltage                                              # (T, N)
    T, N = int(v.shape[0]), int(v.shape[1])
    s1 = np.zeros(N); s2 = np.zeros(N)
    for lo in range(0, T, 4000):                               # streamed: (T, N) is several GB
        blk = np.asarray(v[lo:lo + 4000], dtype=np.float64)
        s1 += blk.sum(0); s2 += (blk ** 2).sum(0)
    mu = s1 / T
    sd = np.sqrt(np.maximum(s2 / T - mu ** 2, 0))
    types = np.asarray(x.neuron_type).astype(int).ravel()[:N]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.errorbar(np.arange(N), mu, yerr=sd, fmt="o", ecolor="lightgray", alpha=0.6, elinewidth=1,
                capsize=0, markersize=2, color="red")
    ax.axhline(floor, color="k", linestyle="--", linewidth=1.2)
    ax.text(N * 1.005, floor, f"filter {floor:.2f}", va="center", fontsize=12)
    starts = np.flatnonzero(np.r_[True, types[1:] != types[:-1]])
    ends = np.r_[starts[1:], N] - 1
    for s0 in starts[1:]:
        ax.axvline(s0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xticks((starts + ends) / 2)
    ax.set_xticklabels([INDEX_TO_NAME.get(int(types[s0]), f"Type{types[s0]}") for s0 in starts],
                       rotation=90, fontsize=9)
    ax.set_xlim(-50, N + 50)
    ax.set_ylabel(r"voltage $v_i(t)$:  $\mu_i \pm \sigma_i$", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    base = os.path.join(REPO, "presentation", "Fig", f"meeting_activity_floor_{run.split('_')[-1]}")
    fig.savefig(base + ".pdf", dpi=200); fig.savefig(base + ".png", dpi=200)
    print("wrote", base + ".{pdf,png}", f" floor {floor:.4f}, {T} frames,",
          f"neurons with mu > floor: {int((mu > floor).sum())} of {N}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=RUN)
    main(ap.parse_args().run)
