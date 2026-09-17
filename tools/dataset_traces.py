#!/usr/bin/env python
"""Draw `activity_all.png` / `activity_selected.png` for an ALREADY GENERATED dataset.

The generator writes these itself now, but every dataset made before that does not
have them, and regenerating a 64,000-frame dataset to obtain two figures would be
absurd -- the data on disk is sufficient. This reads it and calls the same
`save_trace_figure` the generator calls, so a dataset drawn here and one drawn at
generation time are the same figure.

Usage:
    python tools/dataset_traces.py flyvis_noise_free_blank50_cv00 [more datasets...]
"""

import argparse
import os
import sys

import numpy as np
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from connectome_gnn.models.teacher_eval import save_trace_figure  # noqa: E402
from connectome_gnn.plot import INDEX_TO_NAME  # noqa: E402

# The ten types the tester's `selected` figure uses, so a dataset figure and a
# rollout figure show the same rows.
CURATED = [55, 15, 43, 39, 35, 31, 23, 19, 12, 5]
# 1,000 frames = 20 s of simulated time at delta_t = 20 ms, matching activity.png.
N_FRAMES = 1000


def draw(root: str, dataset: str, delta_t: float = 0.02) -> None:
    d = os.path.join(root, dataset)
    v = np.asarray(zarr.open(f"{d}/x_list_train/voltage.zarr", mode="r")[:N_FRAMES])
    types = np.asarray(zarr.open(f"{d}/x_list_train/neuron_type.zarr", mode="r")[:]).astype(int)
    if types.ndim > 1:
        types = types[0]
    stim_path = f"{d}/x_list_train/stimulus.zarr"
    stim = (np.asarray(zarr.open(stim_path, mode="r")[:N_FRAMES, 0])
            if os.path.exists(stim_path) else None)

    first = {}
    for i, t in enumerate(types):
        first.setdefault(int(t), i)

    for name, ids in (
        ("activity_all.png", [first[t] for t in sorted(first)]),
        ("activity_selected.png", [first[t] for t in CURATED if t in first]),
    ):
        if not ids:
            continue
        save_trace_figure(
            os.path.join(d, name), v[:, ids], None, stim, delta_t, None,
            n_traces=len(ids), type_names=INDEX_TO_NAME, type_list=types[ids],
            n_neurons=len(ids), true_color="black", stim_linestyle="--",
            figsize=(9.0, 0.28 * len(ids) + 1.6),
        )
        print(f"  {dataset}/{name}  ({len(ids)} cell types)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("datasets", nargs="+")
    p.add_argument("--root", default="/groups/saalfeld/home/allierc/GraphData/graphs_data/fly")
    p.add_argument("--delta-t", type=float, default=0.02)
    a = p.parse_args()
    for ds in a.datasets:
        draw(a.root, ds, a.delta_t)
