#!/usr/bin/env python3
"""Collect flyvis CV results from log directories and output text tables.

Scans log/fly/ for all CV folds across model variants (GNN LLM-optimized,
GNN default, Known-ODE) and noise conditions. Parses results.log,
results_test.log, results_rollout.log. Outputs tables to docs/flyvis_tables.txt.

Usage:
    python scripts/collect_flyvis_results.py
"""

import math
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(REPO_ROOT, "log", "fly")
OUT_PATH = os.path.join(REPO_ROOT, "docs", "flyvis_tables.txt")

# Model variants and their directory patterns
VARIANTS = [
    ("GNN (LLM-optimized)", "flyvis_{noise}_cv{cv:02d}"),
    ("GNN (default)", "flyvis_{noise}_default_cv{cv:02d}"),
    ("Known-ODE", "flyvis_{noise}_known_ode_cv{cv:02d}"),
]

NOISE_CONDITIONS = [
    ("noise_free", "Noise-free"),
    ("noise_005", "Noise=0.05"),
    ("noise_05", "Noise=0.5"),
]

METRICS = ["W_R2", "tau_R2", "V_rest_R2", "cluster_acc", "onestep_r", "rollout_r"]
METRIC_LABELS = ["W R2", "tau R2", "V_rest R2", "Cluster acc", "One-step r", "Rollout r"]
MAX_CV = 10  # check cv00..cv09


def parse_results_log(path):
    """Parse results.log for W R2, tau R2, V_rest R2, cluster accuracy."""
    metrics = {}
    if not os.path.isfile(path):
        return metrics
    with open(path) as f:
        text = f.read()

    # W R2: try "effective W R²", then "second weights fit R²", then "weights R²"
    m = re.search(r'effective W R²:\s*([\d.]+)', text)
    if not m:
        m = re.search(r'second weights fit R²:\s*([\d.]+)', text)
    if not m:
        m = re.search(r'weights R²:\s*([\d.]+)', text)
    if m:
        metrics["W_R2"] = float(m.group(1))

    # tau R2: "tau reconstruction R²" or "tau R²"
    m = re.search(r'tau reconstruction R²:\s*([\d.]+)', text)
    if not m:
        m = re.search(r'tau R²:\s*([\d.]+)', text)
    if m:
        metrics["tau_R2"] = float(m.group(1))

    # V_rest R2: "V_rest reconstruction R²" or "V_rest R²"
    m = re.search(r'V_rest reconstruction R²:\s*([\d.]+)', text)
    if not m:
        m = re.search(r'V_rest R²:\s*([\d.]+)', text)
    if m:
        metrics["V_rest_R2"] = float(m.group(1))

    m = re.search(r'GMM.*?accuracy=([\d.]+)', text)
    if m:
        metrics["cluster_acc"] = float(m.group(1))

    return metrics


def parse_test_log(path):
    """Parse results_test.log for one-step Pearson r."""
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        text = f.read()
    m = re.search(r'Pearson r:\s*([\d.]+)', text)
    if m:
        return {"onestep_r": float(m.group(1))}
    return {}


def parse_rollout_log(path):
    """Parse results_rollout.log for rollout Pearson r."""
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        text = f.read()
    m = re.search(r'Pearson r:\s*([\d.]+)', text)
    if m:
        return {"rollout_r": float(m.group(1))}
    return {}


def collect_cv_metrics(dir_pattern, noise_key):
    """Collect metrics for all CV folds matching the pattern."""
    rows = []
    for cv in range(MAX_CV):
        dirname = dir_pattern.format(noise=noise_key, cv=cv)
        log_dir = os.path.join(LOG_ROOT, dirname)
        if not os.path.isdir(log_dir):
            continue

        m = {}
        m.update(parse_results_log(os.path.join(log_dir, "results.log")))
        m.update(parse_test_log(os.path.join(log_dir, "results_test.log")))
        m.update(parse_rollout_log(os.path.join(log_dir, "results_rollout.log")))

        if m:
            rows.append((cv, m))
    return rows


def stats(values):
    """Compute mean, std (population), min, max."""
    n = len(values)
    if n == 0:
        return None, None, None, None
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(var)
    return mean, std, min(values), max(values)


def color_code(val):
    """Return color marker for a value: G(reen) > 0.9, O(range) > 0.5, R(ed) <= 0.5."""
    if val is None:
        return " "
    if val > 0.9:
        return "G"
    elif val > 0.5:
        return "O"
    else:
        return "R"


def format_val(val, width=7):
    if val is None:
        return "—".center(width)
    return f"{val:.3f}".rjust(width)


def write_tables(f):
    """Write all tables to file handle."""

    # ── Summary tables ──
    f.write("=" * 100 + "\n")
    f.write("FLYVIS RESULTS — SUMMARY TABLES (mean ± std)\n")
    f.write("Color: G = green (>0.9), O = orange (>0.5), R = red (≤0.5)\n")
    f.write("=" * 100 + "\n\n")

    for variant_label, dir_pattern in VARIANTS:
        f.write(f"### {variant_label}\n\n")
        header = f"{'Condition':<14} {'Seeds':>5}"
        for label in METRIC_LABELS:
            header += f"  {label:>17}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for noise_key, noise_label in NOISE_CONDITIONS:
            rows = collect_cv_metrics(dir_pattern, noise_key)
            n_seeds = len(rows)
            if n_seeds == 0:
                f.write(f"{noise_label:<14} {0:>5}  (no data)\n")
                continue

            line = f"{noise_label:<14} {n_seeds:>5}"
            for metric in METRICS:
                vals = [r[1].get(metric) for r in rows if metric in r[1]]
                mean, std, mn, mx = stats(vals)
                if mean is not None:
                    c = color_code(mean)
                    line += f"  {c} {mean:.3f}±{std:.3f}"
                else:
                    line += f"  {'—':>17}"
            f.write(line + "\n")
        f.write("\n")

    # ── Per-seed detail tables ──
    f.write("\n" + "=" * 100 + "\n")
    f.write("PER-SEED DETAIL TABLES\n")
    f.write("=" * 100 + "\n\n")

    for variant_label, dir_pattern in VARIANTS:
        for noise_key, noise_label in NOISE_CONDITIONS:
            rows = collect_cv_metrics(dir_pattern, noise_key)
            if not rows:
                continue

            f.write(f"### {variant_label} — {noise_label}\n\n")
            header = f"{'Seed':<12}"
            for label in METRIC_LABELS:
                header += f"  {label:>11}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            all_vals = {m: [] for m in METRICS}

            for cv, m in rows:
                seed = 42 + cv
                line = f"cv{cv:02d} ({seed})" .ljust(12)
                for metric in METRICS:
                    val = m.get(metric)
                    if val is not None:
                        c = color_code(val)
                        line += f"  {c}{val:>10.3f}"
                        all_vals[metric].append(val)
                    else:
                        line += f"  {'—':>11}"
                f.write(line + "\n")

            # Summary rows
            for stat_label, stat_fn in [("Mean", lambda v: stats(v)[0]),
                                         ("Std", lambda v: stats(v)[1]),
                                         ("Min", lambda v: stats(v)[2]),
                                         ("Max", lambda v: stats(v)[3])]:
                line = f"{stat_label:<12}"
                for metric in METRICS:
                    vals = all_vals[metric]
                    val = stat_fn(vals) if vals else None
                    if val is not None:
                        if stat_label == "Mean":
                            c = color_code(val)
                            line += f"  {c}{val:>10.3f}"
                        else:
                            line += f"  {val:>11.3f}"
                    else:
                        line += f"  {'—':>11}"
                f.write(line + "\n")
            f.write("\n")


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(OUT_PATH, "w") as f:
        write_tables(f)

    print(f"Tables written to {OUT_PATH}")

    # Also print to stdout
    with open(OUT_PATH) as f:
        print(f.read())


if __name__ == "__main__":
    main()
