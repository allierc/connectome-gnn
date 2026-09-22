"""Per-neuron least-squares recovery vs process noise (Supp. Fig. 12, completed).

Reads the SVD/recovery cache written by
figures/flyvis/fig_lstsq_identifiability_noise.py and reports identity-line R^2
for W, tau and V_rest at sigma = 0, 0.05, 0.5, mean +/- SD over cv00..cv04.

Splits each by the degeneracy label from the SAME cache, so the conditioning
analysis and the recovery analysis are read off one object:
  identifiable = onset > SLOPPY_EIG_TOL, degenerate = onset <= SLOPPY_EIG_TOL.

    PYTHONPATH=src python neurips_review/lstsq_recovery_noise.py
"""

import glob
import os

import numpy as np
import torch

from connectome_gnn.metrics import compute_r_squared_NSE

CACHE = os.path.join(os.path.dirname(__file__), "..", "figures", "flyvis",
                     "_identif_svd_cache")
NULL_EIG_TOL = 1e-22
SLOPPY_EIG_TOL = 1e-12

RUNS = [("0", "flyvis_noise_free_blank50_cv*"),
        ("0.05", "flyvis_noise_005_blank50_cv*"),
        ("0.5", "flyvis_noise_05_blank50_cv*")]


def r2(true, pred, mask=None):
    t, p = np.asarray(true).ravel(), np.asarray(pred).ravel()
    if mask is not None:
        t, p = t[mask], p[mask]
    if t.size < 2:
        return float("nan")
    ok = np.isfinite(t) & np.isfinite(p)
    if ok.sum() < 2:
        return float("nan")
    return float(compute_r_squared_NSE(t[ok], p[ok])[0])


def ms(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return (float(v.mean()), float(v.std())) if v.size else (float("nan"),) * 2


print(f"{'sigma':>6} {'param':>6} {'R2 all':>16} {'R2 identifiable':>18} "
      f"{'R2 degenerate':>16} {'% degenerate':>13}")
for sig, pat in RUNS:
    acc = {k: {"all": [], "id": [], "deg": [], "pct": []}
           for k in ("W", "tau", "vrest")}
    for f in sorted(glob.glob(os.path.join(CACHE, pat + ".pt"))):
        d = torch.load(f, weights_only=False)
        s = d["scatter"]
        for k in ("W", "tau", "vrest"):
            true, pred = np.asarray(s[f"{k}_true"]), np.asarray(s[f"{k}_pred"])
            onset = np.asarray(s[f"{k}_onset"])
            n = min(true.size, pred.size, onset.size)
            true, pred, onset = true[:n], pred[:n], onset[:n]
            deg = onset <= SLOPPY_EIG_TOL
            acc[k]["all"].append(r2(true, pred))
            acc[k]["id"].append(r2(true, pred, ~deg))
            acc[k]["deg"].append(r2(true, pred, deg))
            acc[k]["pct"].append(100.0 * deg.mean())
    for k in ("W", "tau", "vrest"):
        a, i, g, p = (ms(acc[k][x]) for x in ("all", "id", "deg", "pct"))
        print(f"{sig:>6} {k:>6} {a[0]:9.4f}+/-{a[1]:.4f} "
              f"{i[0]:11.4f}+/-{i[1]:.4f} {g[0]:9.4f}+/-{g[1]:.4f} "
              f"{p[0]:8.2f}+/-{p[1]:.2f}")
