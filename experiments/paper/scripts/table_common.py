"""Shared by the paper's table scripts (one script per table):

    table_1_gnn_vs_baselines.py   Tab. 1         tables/cv_table_gnn_vs_baselines.tex
    table_2_flybrid.py            Tab. 2         tables/flybrid_inliers.tex
    table_s4_cross_noise.py       Supp. Tab. 4   tables/cv_table_gnn_cross_noise.tex
    table_s7_known_ode.py         Supp. Tab. 7   tables/cv_table_known_ode_conditions.tex

Every campaign row is read from <GNN_OUTPUT_ROOT>/log/fly/<run>/results/metrics.txt,
as -o test_plot wrote it: one_step_r, rollout_r, Wij_R2 (template readout over the
fitted edges), tau_R2 and V_rest_R2 with their outlier percentages,
clustering_accuracy. Across the five folds cv00..cv04: mean and SD with ddof = 0 of
the per-fold values -- the aggregation tools/exp.py uses for the experiment files
and report.pdf, so the tables and the report agree number for number. The
held-out frame count printed in a caption is the tester's ("Frames evaluated" of
results_rollout.log), the same for every run of a table or the script stops.
Captions are green (colour `revised`, defined by edit_tex.py).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import LOG_ROOT, TAB_DIR, read_metrics, tester_rollout_frames  # noqa: E402,F401

FOLDS = [f"cv{i:02d}" for i in range(5)]
GOOD, BAD = 0.9, 0.3


# ----------------------------------------------------------------------------- aggregation
def aggregate(pattern, folds=FOLDS):
    """pattern has {fold}; returns {metric: (mean, sd)} and n folds found."""
    acc = {}
    frames = set()
    n = 0
    for f in folds:
        m = read_metrics(os.path.join(LOG_ROOT, pattern.format(fold=f)))
        if "Wij_R2" not in m:
            continue
        n += 1
        log = os.path.join(LOG_ROOT, pattern.format(fold=f), "results_rollout.log")
        if os.path.exists(log):
            frames.add(tester_rollout_frames(log))
        for k in ("one_step_r", "rollout_r", "Wij_R2", "tau_R2", "tau_pct_outliers",
                  "V_rest_R2", "V_rest_pct_outliers", "clustering_accuracy"):
            acc.setdefault(k, []).append(float(m.get(k, np.nan)))
    out = {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in acc.items()}
    out["n"] = n
    out["frames"] = sorted(frames)
    # every table row is five folds: a fold whose metrics.txt is missing or still being written
    # (no Wij_R2 yet) must stop the table, not silently average the other four
    assert n == len(folds), f"{pattern}: {n} of {len(folds)} folds have a complete metrics.txt"
    return out


def cell(ms, red=False):
    """$m{\\pm}s$, green above 0.9, orange below 0.3; red overrides."""
    m, s = ms
    if np.isnan(m):
        return "$\\cdot$"
    body = f"${m:.2f}{{\\pm}}{s:.2f}$"
    if red:
        return f"\\textcolor{{red}}{{{body}}}"
    if m > GOOD:
        return f"\\good{{{body}}}"
    if m < BAD:
        return f"\\bad{{{body}}}"
    return body


def pct(ms, red=False, like=None):
    """$\\,(x.x)$ outlier percentage, coloured like its R^2 cell."""
    m, _ = ms
    body = f"$\\,({m:.1f})$"
    if red:
        return f"\\textcolor{{red}}{{{body}}}"
    if like is not None and like > GOOD:
        return f"\\good{{{body}}}"
    if like is not None and like < BAD:
        return f"\\bad{{{body}}}"
    return body


def metric_cells(a, red=False):
    """The seven metric cells of a GNN / Known-ODE row (split tau / V_rest)."""
    return (f"  & {cell(a['one_step_r'], red)} & {cell(a['rollout_r'], red)}\n"
            f"  & {cell(a['Wij_R2'], red)}\n"
            f"  & {cell(a['tau_R2'], red)} & {pct(a['tau_pct_outliers'], red, a['tau_R2'][0])}\n"
            f"  & {cell(a['V_rest_R2'], red)} & {pct(a['V_rest_pct_outliers'], red, a['V_rest_R2'][0])}\n")


def frames_tex(*aggs):
    """The held-out rollout length(s) of these aggregates as LaTeX: one value
    ($7{,}207$) or the range over folds ($7{,}207$--$7{,}999$). Each fold holds
    out a different stretch of the recording, so the length is a property of
    the fold (cv00 7,207 ... cv04 7,999 frames), read from the tester's logs."""
    fr = sorted({f for a in aggs for f in a.get("frames", [])})
    assert fr, "no tester rollout log found for these rows"
    tex = lambda v: f"${v:,}$".replace(",", "{,}")
    return tex(fr[0]) if len(fr) == 1 else f"{tex(fr[0])}--{tex(fr[-1])}"


def save_numbers(key, value):
    """Merge this table's aggregates into tables/table_numbers.json under `key`."""
    p = os.path.join(TAB_DIR, "table_numbers.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    d[key] = value
    json.dump(d, open(p, "w"), indent=1)


def print_rows(name, d):
    print(f"== {name}")
    for k, a in d.items():
        print(f"  {k:<48} n={a['n']}  one {a['one_step_r'][0]:.3f}  roll {a['rollout_r'][0]:.3f}  "
              f"W {a['Wij_R2'][0]:.3f}  tau {a['tau_R2'][0]:.3f} ({a['tau_pct_outliers'][0]:.1f})  "
              f"V {a['V_rest_R2'][0]:.3f} ({a['V_rest_pct_outliers'][0]:.1f})  "
              f"cl {a.get('clustering_accuracy', (np.nan,))[0]:.3f}  frames {a.get('frames')}")
