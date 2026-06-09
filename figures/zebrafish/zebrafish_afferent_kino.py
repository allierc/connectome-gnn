"""Shared loader for the three afferent-class ΔF/F kinographs.

The afferent partition is the Fig. 1 / Fig. 2 six-way colour code, restricted
to the three velocity-gate afferent classes that carry the model's external
drives:

    ARTR            (blue,   angular afferent       — ω drive)
    pt-IPN1         (orange, exteroceptive forward  — v_ext drive)
    motor_efferent  (green,  proprioceptive forward — v_prop drive)

Type membership and colours mirror ``connectome_gnn.plot_cx`` (``_HD_ARTR_TYPES``
/ ``_HD_MOTOR_EFFERENT_TYPES`` and ``_HD_PARTITION_COLORS``) so the recorded-
activity kinographs added to Figs. 14/22 use exactly the same taxonomy and
colour code as the anatomy/connectome figures.

Both consumer figures (``fig_zebrafish_all_blocks.py`` and
``fig_zebrafish_rotation_covariates.py``) read the recorded ΔF/F from the same
``circuit_functional_traces.npz`` and the cell types from the same
``bodyid_zapbench_map.csv``; this module returns the matched-cell traces per
class, leaving the time-slice + z-score to the caller so each afferent
kinograph matches the bump-pool kinograph rendered alongside it.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# (name, member cell-types, colour) — same as plot_cx._HD_* / _HD_PARTITION_COLORS.
AFFERENT_GROUPS = (
    ("ARTR",           ("RIPN01", "RIPN02", "RIPN03_a", "RIPN03_b"), "#1f6fb3"),
    ("pt-IPN1",        ("pt-IPN1",),                                  "#e07b1a"),
    ("motor_efferent", ("RIPN11", "RIPN12_a", "RIPN12_c"),           "#2a9d3d"),
)


def load_afferent_traces(connectome_dir):
    """Return ``[(name, colour, traces_TN), ...]`` for the three afferent classes.

    ``traces_TN`` is ``(T_full, n_class)`` recorded ΔF/F over the whole
    recording for the matched cells of that class, ordered side (left, right)
    then type then bodyId. The time axis aligns frame-for-frame with the
    bump-pool traces from the same ``circuit_functional_traces.npz``; the caller
    slices the block it wants and z-scores it the same way it z-scores the
    bump-pool kinograph.
    """
    func = os.path.join(connectome_dir, "functional")
    m = pd.read_csv(os.path.join(func, "bodyid_zapbench_map.csv"))
    npz = np.load(os.path.join(func, "circuit_functional_traces.npz"),
                  allow_pickle=True)
    traces = np.asarray(npz["traces"], np.float32)                 # (T, 481)
    pos = {int(b): i for i, b in enumerate(np.asarray(npz["bodyId"], np.int64))}

    side_rank = {"left": 0, "right": 1}
    out = []
    for name, types, colour in AFFERENT_GROUPS:
        sub = m[m["type"].astype(str).isin(types) & m["matched"]].copy()
        sub["_sr"] = sub["side"].map(lambda s: side_rank.get(str(s), 2))
        sub = sub.sort_values(["_sr", "type", "bodyId"])
        ix = [pos[int(b)] for b in sub["bodyId"].to_numpy() if int(b) in pos]
        out.append((name, colour, traces[:, ix]))                  # (T, n_class)
    return out


def load_afferent_bodyids(connectome_dir):
    """Per-class ordered ``[(name, colour, bodyId_array), ...]`` — the SAME
    side/type/bodyId ordering as ``load_afferent_traces`` (so a model
    kinograph gathered by these bodyIds lines up row-for-row with the
    recorded one)."""
    func = os.path.join(connectome_dir, "functional")
    m = pd.read_csv(os.path.join(func, "bodyid_zapbench_map.csv"))
    npz = np.load(os.path.join(func, "circuit_functional_traces.npz"),
                  allow_pickle=True)
    have = set(int(b) for b in np.asarray(npz["bodyId"], np.int64))
    side_rank = {"left": 0, "right": 1}
    out = []
    for name, types, colour in AFFERENT_GROUPS:
        sub = m[m["type"].astype(str).isin(types) & m["matched"]].copy()
        sub["_sr"] = sub["side"].map(lambda s: side_rank.get(str(s), 2))
        sub = sub.sort_values(["_sr", "type", "bodyId"])
        bids = np.asarray([int(b) for b in sub["bodyId"].to_numpy()
                           if int(b) in have], dtype=np.int64)
        out.append((name, colour, bids))
    return out
