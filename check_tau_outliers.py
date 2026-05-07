"""Standalone check of tau outlier % across CV folds.

Outlier definition matches GNN_PlotFigure.py: |tau_learned - tau_true| > 0.1.

Usage:
    /workspace/.conda_envs/neural-graph-linux/bin/python check_tau_outliers.py
"""
import os
import glob
import numpy as np

LOG_ROOT = "/groups/saalfeld/home/allierc/GraphData/log/fly"
THRESH = 0.1

CONDITIONS = [
    "flyvis_noise_005",
    "flyvis_noise_005_010",
    "flyvis_noise_005_020",
]


def fold_outlier_stats(panels_path: str):
    d = np.load(panels_path)
    t_true = d["tau_true"].ravel()
    t_lrn = d["tau_learned"].ravel()
    n = t_true.size
    diff = np.abs(t_lrn - t_true)
    mask = diff > THRESH
    n_out = int(mask.sum())

    # R^2 helpers (no-outlier R^2 vs raw R^2)
    def _r2(x, y):
        if x.size < 2:
            return float("nan")
        ss_res = float(np.sum((y - x) ** 2))
        ss_tot = float(np.sum((x - x.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return dict(
        n=n,
        n_out=n_out,
        pct=100.0 * n_out / n,
        r2_full=_r2(t_true, t_lrn),
        r2_clean=_r2(t_true[~mask], t_lrn[~mask]),
        true_min=float(t_true.min()),
        true_max=float(t_true.max()),
        lrn_min=float(t_lrn.min()),
        lrn_max=float(t_lrn.max()),
    )


def report(condition: str):
    folds = sorted(glob.glob(os.path.join(
        LOG_ROOT, f"{condition}_blank50_unified_cv*")))
    print(f"\n=== {condition}  ({len(folds)} folds) ===")
    print(f"{'fold':>4}  {'N':>6}  {'n_out':>6}  {'pct':>6}  "
          f"{'R²_full':>8}  {'R²_clean':>9}  "
          f"{'true_range':>20}  {'learned_range':>20}")
    rows = []
    for f in folds:
        panels = glob.glob(os.path.join(f, "results", "panels_*.npz"))
        if not panels:
            print(f"  {os.path.basename(f)}: no panels file")
            continue
        s = fold_outlier_stats(panels[0])
        rows.append(s)
        cv = os.path.basename(f).split("_cv")[-1]
        print(f"  cv{cv:>2}  {s['n']:>6}  {s['n_out']:>6}  {s['pct']:>5.2f}%  "
              f"{s['r2_full']:>8.3f}  {s['r2_clean']:>9.3f}  "
              f"[{s['true_min']:>6.4f},{s['true_max']:>6.4f}]  "
              f"[{s['lrn_min']:>6.4f},{s['lrn_max']:>6.4f}]")
    if rows:
        pcts = np.array([r["pct"] for r in rows])
        nouts = np.array([r["n_out"] for r in rows])
        r2f = np.array([r["r2_full"] for r in rows])
        r2c = np.array([r["r2_clean"] for r in rows])
        print(f"  mean: pct={pcts.mean():.2f}±{pcts.std():.2f}%  "
              f"n_out={nouts.mean():.0f}±{nouts.std():.0f}  "
              f"R²_full={r2f.mean():.3f}±{r2f.std():.3f}  "
              f"R²_clean={r2c.mean():.3f}±{r2c.std():.3f}")


if __name__ == "__main__":
    for c in CONDITIONS:
        report(c)
