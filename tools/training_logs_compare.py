"""Compare two runs' training trajectories at common iterations.

Either side may be a legacy run (tmp_training/metrics.log + Eij.log + msgi_r2.log +
rollout_r.log, pre-2026-09-11 names) or a catalogue run (tmp_training/<key>.log).
Prints, per metric, the number of common checkpoints and the max |difference|.

Usage: python tools/training_logs_compare.py <run_dir_A> <run_dir_B>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from connectome_gnn.metrics import training_log_read  # noqa: E402

# catalogue name -> (legacy file, legacy column)
LEGACY = {
    "Wij_R2": ("metrics.log", "connectivity_r2"),
    "tau_R2": ("metrics.log", "tau_r2_clean"), "tau_R2_all": ("metrics.log", "tau_r2_raw"),
    "V_rest_R2": ("metrics.log", "vrest_r2_clean"), "V_rest_R2_all": ("metrics.log", "vrest_r2_raw"),
    "Eij_R2": ("Eij.log", "r2"), "Eij_rmse": ("Eij.log", "rmse"), "Eij_slope": ("Eij.log", "slope"),
    "msg_i_R2": ("msgi_r2.log", "r2"), "msg_i_R2_scaled": ("msgi_r2.log", "r2_scaled"),
    "msg_i_gain": ("msgi_r2.log", "scale"),
    "rollout_r": ("rollout_r.log", 1), "rollout_rmse": ("rollout_r.log", 2),
}
KEY_OF = {"Wij": "Wij", "tau": "tau", "V_rest": "V_rest", "Eij": "Eij", "msg_i": "msg_i", "rollout": "rollout"}


def series(run, name):
    """(iterations, values) for catalogue metric `name` from either layout."""
    key = name.split("_")[0] if not name.startswith(("V_rest", "msg_i")) else ("V_rest" if name.startswith("V_rest") else "msg_i")
    d = training_log_read(run, key)
    if d is not None and name in d:
        return d["iteration"], d[name]
    f, col = LEGACY[name]
    p = os.path.join(run, "tmp_training", f)
    if not os.path.isfile(p):
        return None, None
    if isinstance(col, int):
        a = np.genfromtxt(p, delimiter=",")
        a = a[np.isfinite(a[:, 0])]
        return a[:, 0].astype(int), a[:, col]
    a = np.genfromtxt(p, delimiter=",", names=True)
    if col not in a.dtype.names:
        alt = {"tau_r2_raw": "tau_r2", "vrest_r2_raw": "vrest_r2"}.get(col)
        if alt not in (a.dtype.names or ()):
            return None, None
        col = alt
    return a["iteration"].astype(int), a[col]


def main(A, B):
    print(f"{'metric':18s} {'common':>6s} {'max|diff|':>12s}   last A / last B")
    for name in LEGACY:
        ia, va = series(A, name); ib, vb = series(B, name)
        if ia is None or ib is None:
            print(f"{name:18s} {'-':>6s} {'(absent on one side)':>12s}")
            continue
        common, ka, kb = np.intersect1d(ia, ib, return_indices=True)
        if common.size == 0:
            print(f"{name:18s} {0:6d}")
            continue
        x, y = np.asarray(va, float)[ka], np.asarray(vb, float)[kb]
        ok = np.isfinite(x) & np.isfinite(y)
        print(f"{name:18s} {common.size:6d} {np.abs(x[ok]-y[ok]).max() if ok.any() else float('nan'):12.3e}   {x[-1]:.4f} / {y[-1]:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
