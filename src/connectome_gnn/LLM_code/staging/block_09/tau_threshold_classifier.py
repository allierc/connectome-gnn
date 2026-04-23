"""Evaluate tau_R2 as an early convergence predictor for FlyVis GNN training.

Hypothesis: tau_R2 at a specific training checkpoint discriminates converging
from failing seeds better than conn_R2 at any checkpoint (measured by AUROC).
"""

from __future__ import annotations

import csv
import os
from typing import Optional


def evaluate_tau_threshold_classifier(
    trajectory_dir: str,
    baseline_results: dict[int, float],
    threshold_step: int = 22401,
    converge_threshold: float = 0.90,
) -> dict:
    """Evaluate tau_R2 threshold classifier for early convergence prediction.

    PASS CONDITION: tau_R2 AUROC >= 0.85 at step 22401, AND tau_R2 balanced
    accuracy >= 75% with optimal threshold, AND tau_R2 AUROC > max conn_R2
    AUROC across all checkpoints (proving tau pathway primacy).

    Parameters
    ----------
    trajectory_dir : str
        Path to directory containing iter_NNN.log CSV files with columns:
        iteration, connectivity_r2, vrest_r2, tau_r2
    baseline_results : dict[int, float]
        Mapping of iteration number -> final conn_R2. Used to label each
        seed as converging (>= converge_threshold) or failing.
    threshold_step : int
        Training step at which to evaluate tau_R2 (default 22401).
    converge_threshold : float
        conn_R2 threshold for labeling a seed as converging (default 0.90).

    Returns
    -------
    dict with keys:
        n_seeds : int — number of seeds analyzed
        n_converging : int — seeds with final conn_R2 >= converge_threshold
        n_failing : int — seeds with final conn_R2 < converge_threshold
        tau_auroc : float — AUROC for tau_R2 at threshold_step
        tau_balanced_accuracy : float — balanced accuracy at optimal threshold
        tau_optimal_threshold : float — threshold maximizing balanced accuracy
        conn_auroc_max : float — max AUROC for conn_R2 across all checkpoints
        conn_auroc_best_step : int — step at which conn_R2 AUROC is highest
        auroc_by_step : dict[int, dict] — per-step {tau_auroc, conn_auroc}
        passed : bool — whether all PASS conditions are met
        reason : str — human-readable PASS/FAIL reason
    """
    # --- Load trajectory data for each iter in baseline_results ---
    # Each trajectory file: iter_NNN.log with CSV columns:
    #   iteration, connectivity_r2, vrest_r2, tau_r2
    trajectories: dict[int, dict[int, dict[str, float]]] = {}
    for iter_num in baseline_results:
        fname = os.path.join(trajectory_dir, f"iter_{iter_num:03d}.log")
        if not os.path.isfile(fname):
            continue
        steps = {}
        with open(fname, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row["iteration"])
                steps[step] = {
                    "conn_r2": float(row["connectivity_r2"]),
                    "tau_r2": float(row["tau_r2"]),
                }
        trajectories[iter_num] = steps

    # Filter to seeds that have the threshold_step
    valid_iters = [
        it for it in trajectories
        if threshold_step in trajectories[it] and it in baseline_results
    ]
    if len(valid_iters) < 4:
        return {
            "n_seeds": len(valid_iters),
            "passed": False,
            "reason": f"Only {len(valid_iters)} seeds have step {threshold_step} (need >= 4)",
        }

    # --- Label seeds ---
    labels = {}  # iter -> bool (True = converging)
    for it in valid_iters:
        labels[it] = baseline_results[it] >= converge_threshold

    n_converging = sum(1 for v in labels.values() if v)
    n_failing = sum(1 for v in labels.values() if not v)

    if n_converging == 0 or n_failing == 0:
        return {
            "n_seeds": len(valid_iters),
            "n_converging": n_converging,
            "n_failing": n_failing,
            "passed": False,
            "reason": f"Need both classes: {n_converging} converging, {n_failing} failing",
        }

    # --- Collect all checkpoint steps present in ALL valid iters ---
    all_steps_sets = [set(trajectories[it].keys()) for it in valid_iters]
    common_steps = sorted(set.intersection(*all_steps_sets))
    # Remove step 1 (initialization, always ~0)
    common_steps = [s for s in common_steps if s > 1]

    # --- Compute AUROC and balanced accuracy ---
    def compute_auroc(scores: list[float], labels_bin: list[bool]) -> float:
        """Compute AUROC using the Wilcoxon-Mann-Whitney statistic.
        Higher score -> predicted converging (positive class).
        """
        pos_scores = [s for s, l in zip(scores, labels_bin) if l]
        neg_scores = [s for s, l in zip(scores, labels_bin) if not l]
        if not pos_scores or not neg_scores:
            return 0.5
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        count = 0
        for p in pos_scores:
            for n in neg_scores:
                if p > n:
                    count += 1
                elif p == n:
                    count += 0.5
        return count / (n_pos * n_neg)

    def compute_balanced_accuracy_optimal(
        scores: list[float], labels_bin: list[bool]
    ) -> tuple[float, float]:
        """Find threshold maximizing balanced accuracy. Returns (bal_acc, threshold)."""
        # Try all unique midpoints between sorted scores as thresholds
        unique_scores = sorted(set(scores))
        if len(unique_scores) <= 1:
            return 0.5, unique_scores[0] if unique_scores else 0.0

        # Also try below-min and above-max
        thresholds = []
        for i in range(len(unique_scores) - 1):
            thresholds.append((unique_scores[i] + unique_scores[i + 1]) / 2)
        thresholds.insert(0, unique_scores[0] - 0.01)
        thresholds.append(unique_scores[-1] + 0.01)

        best_ba = 0.0
        best_thr = 0.0
        for thr in thresholds:
            tp = sum(1 for s, l in zip(scores, labels_bin) if s >= thr and l)
            fn = sum(1 for s, l in zip(scores, labels_bin) if s < thr and l)
            tn = sum(1 for s, l in zip(scores, labels_bin) if s < thr and not l)
            fp = sum(1 for s, l in zip(scores, labels_bin) if s >= thr and not l)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ba = (tpr + tnr) / 2
            if ba > best_ba:
                best_ba = ba
                best_thr = thr
        return best_ba, best_thr

    # --- Per-step AUROC for both tau and conn ---
    labels_list = [labels[it] for it in valid_iters]

    auroc_by_step: dict[int, dict[str, float]] = {}
    conn_auroc_max = 0.0
    conn_auroc_best_step = 0

    for step in common_steps:
        tau_scores = [trajectories[it][step]["tau_r2"] for it in valid_iters]
        conn_scores = [trajectories[it][step]["conn_r2"] for it in valid_iters]

        tau_auc = compute_auroc(tau_scores, labels_list)
        conn_auc = compute_auroc(conn_scores, labels_list)

        auroc_by_step[step] = {"tau_auroc": tau_auc, "conn_auroc": conn_auc}

        if conn_auc > conn_auroc_max:
            conn_auroc_max = conn_auc
            conn_auroc_best_step = step

    # --- Evaluate at threshold_step ---
    tau_at_threshold = [
        trajectories[it][threshold_step]["tau_r2"] for it in valid_iters
    ]
    tau_auroc = compute_auroc(tau_at_threshold, labels_list)
    tau_ba, tau_opt_thr = compute_balanced_accuracy_optimal(
        tau_at_threshold, labels_list
    )

    # --- Check PASS conditions ---
    cond_auroc = tau_auroc >= 0.85
    cond_ba = tau_ba >= 0.75
    cond_primacy = tau_auroc > conn_auroc_max

    passed = cond_auroc and cond_ba and cond_primacy

    reasons = []
    if cond_auroc:
        reasons.append(f"tau AUROC={tau_auroc:.3f}>=0.85")
    else:
        reasons.append(f"tau AUROC={tau_auroc:.3f}<0.85")
    if cond_ba:
        reasons.append(f"bal_acc={tau_ba:.3f}>=0.75")
    else:
        reasons.append(f"bal_acc={tau_ba:.3f}<0.75")
    if cond_primacy:
        reasons.append(f"tau AUROC>{conn_auroc_max:.3f} (best conn AUROC at step {conn_auroc_best_step})")
    else:
        reasons.append(f"tau AUROC={tau_auroc:.3f}<=conn max {conn_auroc_max:.3f} at step {conn_auroc_best_step}")

    reason = "; ".join(reasons)

    return {
        "n_seeds": len(valid_iters),
        "n_converging": n_converging,
        "n_failing": n_failing,
        "tau_auroc": tau_auroc,
        "tau_balanced_accuracy": tau_ba,
        "tau_optimal_threshold": tau_opt_thr,
        "conn_auroc_max": conn_auroc_max,
        "conn_auroc_best_step": conn_auroc_best_step,
        "auroc_by_step": auroc_by_step,
        "passed": passed,
        "reason": reason,
    }
