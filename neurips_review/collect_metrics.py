#!/usr/bin/env python
"""Parse recovery metrics for every completed run into a single CSV + printed table.

Reads, per config, from the shared GraphData log tree:
  log/fly/<config>/results/metrics.txt   (key: value)
  log/fly/<config>/results_test.log      (one-step 'Pearson r:' line)

Emits neurips_review/results_table.csv and prints a grouped summary for the
rebuttal (weight R2 + slope, tau/Vrest full-sample + inlier + outlier%, one-step r,
rollout r, cluster acc). n_neurons=13741 for Flyvis-217.
"""
import csv
import json
import os
import re

ROOT = "/groups/saalfeld/home/allierc/GraphData"
HERE = os.path.dirname(os.path.abspath(__file__))
N_NEURONS = 13741

# metrics.txt key -> output column
KEYS = {
    "W_corrected_no_outliers_R2": "R2_W",
    "W_corrected_R2": "R2_W_full",
    "W_corrected_no_outliers_slope": "slope_W",
    "W_corrected_n_outliers": "W_n_out",
    "W_structure_r": "W_struct_r",
    "W_zscored_R2": "W_zscored_R2",
    "tau_R2": "R2_tau_full",
    "tau_no_outliers_R2": "R2_tau_inlier",
    "tau_no_outliers_slope": "slope_tau",
    "tau_n_outliers": "tau_n_out",
    "V_rest_R2": "R2_Vrest_full",
    "V_rest_no_outliers_R2": "R2_Vrest_inlier",
    "V_rest_n_outliers": "Vrest_n_out",
    "rollout_pearson": "rollout_r",
    "clustering_accuracy": "cluster_acc",
}


def parse_metrics_txt(path):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path):
        if ":" in line:
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k in KEYS:
                try:
                    d[KEYS[k]] = float(v)
                except ValueError:
                    pass
    return d


def parse_onestep(path):
    if not os.path.exists(path):
        return None
    m = re.search(r"Pearson r:\s*([0-9.]+)", open(path).read())
    return float(m.group(1)) if m else None


def collect(config):
    ld = os.path.join(ROOT, "log", "fly", config)
    row = {"config": config, "complete": os.path.exists(os.path.join(ld, "_complete"))}
    row.update(parse_metrics_txt(os.path.join(ld, "results", "metrics.txt")))
    row["onestep_r"] = parse_onestep(os.path.join(ld, "results_test.log"))
    if "tau_n_out" in row:
        row["tau_out_pct"] = round(100.0 * row["tau_n_out"] / N_NEURONS, 1)
    if "Vrest_n_out" in row:
        row["Vrest_out_pct"] = round(100.0 * row["Vrest_n_out"] / N_NEURONS, 1)
    return row


def main():
    man = json.load(open(os.path.join(HERE, "manifest.json")))
    cols = ["test", "config", "model", "dataset", "complete", "R2_W", "slope_W",
            "W_struct_r", "R2_tau_full", "R2_tau_inlier", "tau_out_pct",
            "R2_Vrest_full", "R2_Vrest_inlier", "Vrest_out_pct",
            "onestep_r", "rollout_r", "cluster_acc"]
    rows = []
    for t in man["train"]:
        r = collect(t["config"])
        r["test"], r["model"], r["dataset"] = t["test"], t["model"], t["dataset"]
        rows.append(r)

    csv_path = os.path.join(HERE, "results_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    order = {"dt": 0, "mono": 1, "adapt": 2}
    rows.sort(key=lambda r: (order.get(r["test"], 9), r["config"]))
    print(f"\n{'config':40s} {'done':4s} {'R2_W':>6s} {'slope':>6s} {'R2_tau':>7s}(full/in) {'R2_Vr':>6s} {'1step':>6s} {'roll':>6s} {'clust':>6s}")
    print("-" * 118)
    cur = None
    for r in rows:
        if r["test"] != cur:
            cur = r["test"]
            print(f"== {cur} ==")

        def g(k, w=6, p=3):
            v = r.get(k)
            return (f"{v:.{p}f}".rjust(w)) if isinstance(v, float) else "  -  ".rjust(w)

        tau = f"{g('R2_tau_full',5)}/{g('R2_tau_inlier',5)}({r.get('tau_out_pct','-')}%)"
        print(f"{r['config']:40s} {'Y' if r['complete'] else 'n':>4s} "
              f"{g('R2_W')} {g('slope_W')} {tau:>16s} {g('R2_Vrest_full')} "
              f"{g('onestep_r')} {g('rollout_r')} {g('cluster_acc')}")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
