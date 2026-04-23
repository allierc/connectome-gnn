#!/usr/bin/env python
"""Test: evaluate tau_R2 threshold classifier on flyvis_noise_005_from_zero data.

Reads trajectory logs from the exploration directory, builds baseline_results
from the last row of each trajectory file, and evaluates the hypothesis that
tau_R2 at step 22401 is a better convergence predictor than conn_R2.

Prints PASS/FAIL on the last line and exits accordingly.
"""

from __future__ import annotations

import csv
import os
import sys

# --- Configuration ---
TRAJECTORY_DIR = (
    "/workspace/connectome-gnn/log/remote/Claude_exploration/"
    "LLM_flyvis_noise_005_from_zero/r2_trajectory"
)
THRESHOLD_STEP = 22401
CONVERGE_THRESHOLD = 0.90


def load_baseline_results(trajectory_dir: str) -> dict[int, float]:
    """Build baseline_results from trajectory files.

    For each iter_NNN.log, the final conn_R2 is the last row's
    connectivity_r2 column. Only includes iterations that contain
    step THRESHOLD_STEP.
    """
    results = {}
    if not os.path.isdir(trajectory_dir):
        return results

    for fname in sorted(os.listdir(trajectory_dir)):
        if not fname.startswith("iter_") or not fname.endswith(".log"):
            continue
        iter_num = int(fname.replace("iter_", "").replace(".log", ""))
        fpath = os.path.join(trajectory_dir, fname)

        has_threshold = False
        last_conn_r2 = None
        with open(fpath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row["iteration"])
                if step == THRESHOLD_STEP:
                    has_threshold = True
                last_conn_r2 = float(row["connectivity_r2"])

        if has_threshold and last_conn_r2 is not None:
            results[iter_num] = last_conn_r2

    return results


def main() -> int:
    # Import the staged function
    from connectome_gnn.LLM_code.staging.block_09.tau_threshold_classifier import (
        evaluate_tau_threshold_classifier,
    )

    # Check trajectory dir exists
    if not os.path.isdir(TRAJECTORY_DIR):
        print(f"FAIL: trajectory directory not found: {TRAJECTORY_DIR}")
        return 1

    # Build baseline results from trajectory files
    baseline_results = load_baseline_results(TRAJECTORY_DIR)
    print(f"Loaded {len(baseline_results)} seeds with step {THRESHOLD_STEP}")

    if len(baseline_results) < 10:
        print(f"FAIL: only {len(baseline_results)} seeds available (need >= 10 for meaningful test)")
        return 1

    # Count class balance
    n_conv = sum(1 for v in baseline_results.values() if v >= CONVERGE_THRESHOLD)
    n_fail = sum(1 for v in baseline_results.values() if v < CONVERGE_THRESHOLD)
    print(f"Class balance: {n_conv} converging (>={CONVERGE_THRESHOLD}), {n_fail} failing")

    # Run the classifier evaluation
    result = evaluate_tau_threshold_classifier(
        trajectory_dir=TRAJECTORY_DIR,
        baseline_results=baseline_results,
        threshold_step=THRESHOLD_STEP,
        converge_threshold=CONVERGE_THRESHOLD,
    )

    # Print detailed results
    print(f"\n--- Results at step {THRESHOLD_STEP} ---")
    print(f"Seeds analyzed: {result.get('n_seeds', 'N/A')}")
    print(f"Converging: {result.get('n_converging', 'N/A')}, Failing: {result.get('n_failing', 'N/A')}")
    print(f"tau_R2 AUROC: {result.get('tau_auroc', 'N/A'):.4f}" if isinstance(result.get('tau_auroc'), float) else f"tau_R2 AUROC: {result.get('tau_auroc', 'N/A')}")
    print(f"tau_R2 balanced accuracy: {result.get('tau_balanced_accuracy', 'N/A'):.4f}" if isinstance(result.get('tau_balanced_accuracy'), float) else f"tau_R2 balanced accuracy: {result.get('tau_balanced_accuracy', 'N/A')}")
    if isinstance(result.get('tau_optimal_threshold'), float):
        print(f"tau_R2 optimal threshold: {result['tau_optimal_threshold']:.4f}")
    print(f"conn_R2 max AUROC: {result.get('conn_auroc_max', 'N/A'):.4f} at step {result.get('conn_auroc_best_step', 'N/A')}" if isinstance(result.get('conn_auroc_max'), float) else "")

    # Print AUROC comparison across checkpoints
    auroc_by_step = result.get("auroc_by_step", {})
    if auroc_by_step:
        print("\n--- AUROC by checkpoint ---")
        print(f"{'Step':>8}  {'tau_AUROC':>10}  {'conn_AUROC':>10}  {'tau > conn':>10}")
        for step in sorted(auroc_by_step.keys()):
            info = auroc_by_step[step]
            tau_a = info["tau_auroc"]
            conn_a = info["conn_auroc"]
            better = "YES" if tau_a > conn_a else "no"
            print(f"{step:>8}  {tau_a:>10.4f}  {conn_a:>10.4f}  {better:>10}")

    # Final verdict
    print()
    if result.get("passed", False):
        print(f"PASS: tau_R2 AUROC={result['tau_auroc']:.3f} at step {THRESHOLD_STEP}, bal_acc={result['tau_balanced_accuracy']:.3f}, beats conn_R2 max AUROC={result['conn_auroc_max']:.3f} — tau pathway primacy confirmed ({result['n_seeds']} seeds)")
        return 0
    else:
        print(f"FAIL: {result.get('reason', 'unknown')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
