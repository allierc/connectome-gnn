"""Systematic 6-condition aggregator for drosophila_cx PI CV runs.

Pulls per-fold final metrics from
  <log_root>/drosophila_cx/<condition>_cv<k>/tmp_training/metrics.log
for every (condition, fold) pair and produces:

  - figures/drosophila_cx/tab_cv_summary.tex   (LaTeX table for drosophila.tex)
  - figures/drosophila_cx/cv_summary.json      (per-fold + summary numbers)

The six conditions:

  | code               | description                              |
  |--------------------|------------------------------------------|
  | epg_tv             | Known-ODE RNN, EPG readout, with TV      |
  | epg_no_tv          | Known-ODE RNN, EPG readout, no TV        |
  | gnn_epg_tv         | Message-passing GNN, EPG readout, TV     |
  | gnn_epg_no_tv      | Message-passing GNN, EPG readout, no TV  |
  | fc_epg             | Fully connected RNN, EPG readout         |
  | frozen_Wrec_epg    | Frozen W_rec, EPG readout                |

Per condition (n=10 folds) we report mean +/- std and converged-count for
the five metrics that drive every per-model paragraph in the Results:

  r_roll_1k   final-checkpoint Pearson r on a 1000-frame constant-omega rollout
  pi_acc      path-integration accuracy (1 - mean(|d_theta|)/pi)
  fwhm_deg    full-width half-max of the EPG bump, degrees
  rmse_roll   circular RMSE on the rollout, degrees
  loss        final training loss

Convergence flag = (|r_roll_1k| >= 0.9). The sign of r_roll_1k carries no
biological information (cos/sin readout is sign-symmetric); we use abs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np


CONDITIONS = [
    ("epg_no_tv",       "Known-ODE RNN, no TV"),
    ("epg_tv",          "Known-ODE RNN +TV"),
    ("gnn_epg_no_tv",   "GNN, no TV"),
    ("gnn_epg_tv",      "GNN +TV"),
    ("fc_epg",          "Fully connected RNN"),
    ("frozen_Wrec_epg", "Frozen $W^{\\mathrm{rec}}$"),
]

METRIC_COLS = {
    "loss":       "loss",
    "pi_acc":     "pi_acc",
    "fwhm_deg":   "fwhm_deg",
    "rmse_roll":  "rmse_roll_deg",
    "r_roll_1k":  "r_roll_1k",
}

R_CONVERGED = 0.9


def _read_last_row(path: str) -> dict | None:
    """Return the last row of a metrics.log as a {column: float} dict, or None."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rdr = csv.reader(f)
        try:
            header = next(rdr)
        except StopIteration:
            return None
        last = None
        for row in rdr:
            if row:
                last = row
        if last is None or len(last) != len(header):
            return None
    out = {}
    for name, val in zip(header, last):
        try:
            out[name] = float(val)
        except ValueError:
            out[name] = float("nan")
    return out


def _fold_metrics(log_root: str, condition: str, fold: int) -> dict | None:
    """Pull final metrics for one (condition, fold). Returns None on miss."""
    path = os.path.join(
        log_root, "drosophila_cx",
        f"drosophila_cx_pi_{condition}_cv{fold}",
        "tmp_training", "metrics.log",
    )
    return _read_last_row(path)


def _aggregate(log_root: str, n_folds: int = 10) -> dict:
    out = {}
    for code, label in CONDITIONS:
        per_fold = []
        for k in range(n_folds):
            row = _fold_metrics(log_root, code, k)
            per_fold.append(row)
        # arrays for each metric
        agg = {"label": label, "n_folds": n_folds, "per_fold": []}
        cols = {m: [] for m in METRIC_COLS}
        n_loaded = 0
        for k, row in enumerate(per_fold):
            if row is None:
                agg["per_fold"].append({"fold": k, **{m: None for m in METRIC_COLS}})
                continue
            n_loaded += 1
            this = {"fold": k}
            for short, col in METRIC_COLS.items():
                v = row.get(col, float("nan"))
                this[short] = v
                cols[short].append(v)
            agg["per_fold"].append(this)
        agg["n_loaded"] = n_loaded
        # summary stats
        for short in METRIC_COLS:
            arr = np.asarray(cols[short], dtype=float)
            arr = arr[~np.isnan(arr)]
            agg[f"{short}_mean"] = float(arr.mean()) if arr.size else float("nan")
            agg[f"{short}_std"]  = float(arr.std())  if arr.size else float("nan")
            agg[f"{short}_max"]  = float(np.nanmax(arr)) if arr.size else float("nan")
            agg[f"{short}_min"]  = float(np.nanmin(arr)) if arr.size else float("nan")
        # |r| stats + convergence rate
        r_arr = np.asarray(cols["r_roll_1k"], dtype=float)
        r_arr = r_arr[~np.isnan(r_arr)]
        abs_r = np.abs(r_arr)
        agg["abs_r_mean"] = float(abs_r.mean()) if abs_r.size else float("nan")
        agg["abs_r_std"]  = float(abs_r.std())  if abs_r.size else float("nan")
        agg["n_converged"] = int((abs_r >= R_CONVERGED).sum())
        out[code] = agg
    return out


def _emit_table(agg: dict, out_path: str) -> None:
    """Per-condition mean+/-std summary table (LaTeX)."""
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Condition & $n_{\mathrm{conv}}$/10 & $|r_{\mathrm{roll},1k}|$ & "
        r"$p_{\mathrm{acc}}$ & FWHM ($^\circ$) & RMSE ($^\circ$) \\",
        r"\midrule",
    ]
    for code, _label in CONDITIONS:
        a = agg[code]
        nconv = a["n_converged"]
        r = a["abs_r_mean"]; r_sd = a["abs_r_std"]
        p = a["pi_acc_mean"]; p_sd = a["pi_acc_std"]
        f = a["fwhm_deg_mean"]; f_sd = a["fwhm_deg_std"]
        rm = a["rmse_roll_mean"]; rm_sd = a["rmse_roll_std"]
        lines.append(
            f"{a['label']} & "
            f"{nconv}/10 & "
            f"${r:.3f}\\pm{r_sd:.3f}$ & "
            f"${p:.3f}\\pm{p_sd:.3f}$ & "
            f"${f:.1f}\\pm{f_sd:.1f}$ & "
            f"${rm:.1f}\\pm{rm_sd:.1f}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log_root",
                   default="/groups/saalfeld/home/allierc/GraphData/log")
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--out_dir",
                   default=os.path.dirname(os.path.abspath(__file__)))
    args = p.parse_args()

    agg = _aggregate(args.log_root, n_folds=args.n_folds)

    print(f"\n=== 6-condition CV summary (|r|>={R_CONVERGED} = converged) ===")
    print(f"{'condition':<32} {'n/10':>5}  {'|r|':>14}  {'pi_acc':>14}  "
          f"{'rmse':>14}")
    for code, _ in CONDITIONS:
        a = agg[code]
        print(
            f"{a['label']:<32} "
            f"{a['n_converged']:>2}/10  "
            f"{a['abs_r_mean']:>6.3f}±{a['abs_r_std']:>5.3f}  "
            f"{a['pi_acc_mean']:>6.3f}±{a['pi_acc_std']:>5.3f}  "
            f"{a['rmse_roll_mean']:>6.1f}±{a['rmse_roll_std']:>5.1f}"
        )

    _emit_table(agg, os.path.join(args.out_dir, "tab_cv_summary.tex"))
    out_json = os.path.join(args.out_dir, "cv_summary.json")
    with open(out_json, "w") as fh:
        json.dump(agg, fh, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
