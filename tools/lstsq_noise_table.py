"""Least-squares parameter recovery at the three model-noise levels, scored as the paper tables are.

    PYTHONPATH=src python tools/lstsq_noise_table.py [--dir /groups/saalfeld/home/allierc/GraphData/lstsq_noise]

Reads the `<dataset>_estimates.pt` that figures/fig_lstsq_param_recovery.py writes (per-neuron
least squares of dv_i/dt on [v_i, 1, relu(v_j), I_i], all neurons, all training frames) for
flyvis_{noise_free,noise_005,noise_05}_blank50_cv00..cv04, and scores each with
metrics.recovery_param_metrics at the outlier bands of RecoveryConfig (W 1.0, tau 0.1, V_rest 0.2),
the headline R2 being outlier-corrected as in Table 1. "fitted %" is the share of edges whose
least-squares W is finite. Prints mean +- SD over the five folds (ddof 0, as the tables) and
writes <dir>/lstsq_noise_table.json.
"""
import argparse
import json
import os

import numpy as np
import torch

from connectome_gnn.metrics import recovery_param_metrics

NOISE = ("noise_free", "noise_005", "noise_05")
THRESH = {"W": 1.0, "tau": 0.1, "V_rest": 0.2}


def score(path):
    e = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for q, (lk, tk) in {"W": ("W_lstsq", "W_true"), "tau": ("tau_lstsq", "tau_true"),
                        "V_rest": ("V_rest_lstsq", "V_rest_true")}.items():
        l = np.asarray(e[lk], dtype=np.float64).ravel(); t = np.asarray(e[tk], dtype=np.float64).ravel()
        ok = np.isfinite(l) & np.isfinite(t)
        m = recovery_param_metrics(t[ok], l[ok], THRESH[q])
        out[f"{q}_R2"] = float(m["r2_clean"]); out[f"{q}_pct_outliers"] = float(m["pct_outliers"])
        if q == "W":
            out["W_pct_fitted"] = float(100.0 * ok.mean())
    return out


def main(d):
    res = {}
    for nz in NOISE:
        folds = [score(os.path.join(d, f"flyvis_{nz}_blank50_cv0{k}_estimates.pt")) for k in range(5)]
        res[nz] = {k: (float(np.mean([f[k] for f in folds])), float(np.std([f[k] for f in folds])))
                   for k in folds[0]}
        r = res[nz]
        print(f"{nz:11s} W {r['W_R2'][0]:.3f}+-{r['W_R2'][1]:.3f} ({r['W_pct_fitted'][0]:.0f}% fitted)  "
              f"tau {r['tau_R2'][0]:.3f}+-{r['tau_R2'][1]:.3f} ({r['tau_pct_outliers'][0]:.1f}% out)  "
              f"V_rest {r['V_rest_R2'][0]:.3f}+-{r['V_rest_R2'][1]:.3f} ({r['V_rest_pct_outliers'][0]:.1f}% out)")
    json.dump(res, open(os.path.join(d, "lstsq_noise_table.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/groups/saalfeld/home/allierc/GraphData/lstsq_noise")
    main(ap.parse_args().dir)
