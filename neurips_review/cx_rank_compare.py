"""Activity / W spectra for the CX drive comparison (Vzfg Q1).

Recomputes the ranks quoted in the Q1 paragraph with a full SVD; the generator's
rank_info.txt caps the component count at 50, so its 99% figures saturate at 51.

    GNN_OUTPUT_ROOT=... PYTHONPATH=src python neurips_review/cx_rank_compare.py
"""

import os

import numpy as np

from connectome_gnn.zarr_io import load_simulation_data

ROOT = os.path.join(
    os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"),
    "graphs_data",
    "drosophila_cx",
)

RUNS = [
    ("full drive", "nr2_cx_ring_s00", 0.0),
    ("full drive", "nr2_cx_ring_s005", 0.05),
    ("full drive", "nr2_cx_ring_s05", 0.5),
    ("velocity only", "nr3_cx_vel_s00", 0.0),
    ("velocity only", "nr3_cx_vel_s005", 0.05),
    ("velocity only", "nr3_cx_vel_s05", 0.5),
]


def rank_at(S, frac):
    """Smallest k with cumulative spectral energy >= frac."""
    cum = np.cumsum(S**2) / np.sum(S**2)
    return int(np.searchsorted(cum, frac) + 1)


def spectra(matrix):
    S = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    return rank_at(S, 0.90), rank_at(S, 0.99)


def drive_share(dataset):
    """How the training loss splits between driven and purely recurrent neurons.

    Driven = receives a nonzero injected stimulus at some frame. For those, R^2
    of dv/dt on that neuron's own drive, energy-weighted -- the fraction of the
    loss a model can serve without using W at all.
    """
    x = load_simulation_data(os.path.join(ROOT, dataset, "x_list_train"))
    v = x.voltage.numpy().astype(np.float64)
    stim = x.stimulus.numpy().astype(np.float64)
    dv = np.diff(v, axis=0)
    stim = stim[:-1]

    driven = np.abs(stim).max(axis=0) > 1e-12
    energy = np.sum(dv**2, axis=0)
    total = energy.sum()

    ss_res = 0.0
    for j in np.flatnonzero(driven):
        s, d = stim[:, j], dv[:, j]
        sc, dc = s - s.mean(), d - d.mean()
        beta = np.dot(sc, dc) / np.dot(sc, sc)
        ss_res += float(np.sum((dc - beta * sc) ** 2))
    ss_tot = float(np.sum((dv[:, driven] - dv[:, driven].mean(axis=0)) ** 2))
    r2_driven = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "n_driven": int(driven.sum()),
        "share_driven": energy[driven].sum() / total,
        "share_recurrent": energy[~driven].sum() / total,
        "r2_driven": r2_driven,
    }


print(f"{'drive':<14} {'sigma':>5} {'W90':>4} {'W99':>4} "
      f"{'act90':>6} {'act99':>6} {'mc90':>5} {'mc99':>5} {'stim90':>7}")
for drive, dataset, sigma in RUNS:
    path = os.path.join(ROOT, dataset, "x_list_train")
    x = load_simulation_data(path)
    v = x.voltage.numpy()
    stim = x.stimulus.numpy()

    w = np.load(os.path.join(ROOT, dataset, "weights.pt"), allow_pickle=True) \
        if os.path.exists(os.path.join(ROOT, dataset, "weights.pt.npy")) else None
    import torch
    ei = torch.load(os.path.join(ROOT, dataset, "edge_index.pt"), weights_only=False)
    wt = torch.load(os.path.join(ROOT, dataset, "weights.pt"), weights_only=False)
    ei = ei.cpu().numpy() if hasattr(ei, "cpu") else np.asarray(ei)
    wt = wt.detach().cpu().numpy().ravel() if hasattr(wt, "detach") else np.asarray(wt).ravel()
    n = v.shape[1]
    W = np.zeros((n, n), dtype=np.float64)
    W[ei[0], ei[1]] = wt

    w90, w99 = spectra(W)
    a90, a99 = spectra(v)
    mc90, mc99 = spectra(v - v.mean(axis=0, keepdims=True))
    s90 = spectra(stim)[0] if np.abs(stim).max() > 1e-12 else 0
    print(f"{drive:<14} {sigma:>5} {w90:>4} {w99:>4} {a90:>6} {a99:>6} "
          f"{mc90:>5} {mc99:>5} {s90:>7}")

print()
print("split of the dv/dt energy an unweighted L2 loss sees")
for drive, dataset, sigma in RUNS:
    sh = drive_share(dataset)
    print(f"{drive:<14} sigma={sigma:<5} n_driven={sh['n_driven']:>3}  "
          f"driven {sh['share_driven']:6.1%} of loss (R2 on own drive {sh['r2_driven']:6.1%})  "
          f"recurrent {sh['share_recurrent']:6.1%}")
