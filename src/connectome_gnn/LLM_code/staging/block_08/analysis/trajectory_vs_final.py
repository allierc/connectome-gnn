"""Investigate discrepancy between training trajectory conn_R2 and final reported conn_R2.

Block 1 seeds match perfectly (trajectory endpoint ≈ final). Block 8 seeds have
wild inversions (e.g., iter 74 trajectory=0.952 but final=0.744). This could be:
1. Trajectory logs slot 0 only, but final metrics are from different slots
2. Training-time vs test-time metric difference (noise→noisy R², test→clean R²)
3. The final eval uses different W post-processing

Also: analyze the RATE of early R² growth to see if failing seeds diverge in slope.
"""
import os
import csv
import numpy as np

LOG_DIR = "/groups/saalfeld/home/allierc/GraphData/log/Claude_exploration/LLM_flyvis_noise_005_from_zero/r2_trajectory"
RESULTS_DIR = "/groups/saalfeld/home/allierc/GraphData/log/Claude_exploration/LLM_flyvis_noise_005_from_zero"

def load_trajectory(iter_num):
    path = os.path.join(LOG_DIR, f"iter_{iter_num:03d}.log")
    if not os.path.exists(path):
        return None
    steps, conn_r2 = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['iteration']))
            conn_r2.append(float(row['connectivity_r2']))
    return np.array(steps), np.array(conn_r2)


def check_results_log():
    """Check if there are per-slot results files to compare."""
    log_file = os.path.join(RESULTS_DIR, "flyvis_noise_005_from_zero_Claude_00_analysis.log")
    if os.path.exists(log_file):
        print(f"\n--- Analysis log found: {log_file} ---")
        with open(log_file) as f:
            lines = f.readlines()
        # Look for Block 8 results (iters 73-84)
        for i, line in enumerate(lines):
            if any(f"iter_{n:03d}" in line or f"Iter {n}" in line or f"iter {n}" in line
                   for n in range(73, 85)):
                print(f"  Line {i+1}: {line.rstrip()[:120]}")

    # Check for individual results.log files in iteration directories
    for batch_num in range(19, 23):  # Batches 19-22 correspond to iters 73-84 (4 per batch)
        batch_dir = os.path.join(RESULTS_DIR, f"config")
        for slot in range(4):
            for possible_name in [
                f"iter_{batch_num:03d}_slot_{slot:02d}",
                f"batch_{batch_num:02d}_slot_{slot:02d}",
            ]:
                d = os.path.join(RESULTS_DIR, possible_name)
                if os.path.exists(d):
                    print(f"  Found dir: {d}")


def check_slot_structure():
    """Determine how iterations map to slots in the pipeline."""
    print("\n--- Checking directory structure for slot organization ---")
    # List any directories that look like iteration results
    base = RESULTS_DIR
    for entry in sorted(os.listdir(base)):
        full = os.path.join(base, entry)
        if os.path.isdir(full) and ("iter" in entry or "slot" in entry or "batch" in entry):
            print(f"  DIR: {entry}")


def analyze_early_slope():
    """Check if the SLOPE of R2 growth in the first 2-3 checkpoints predicts failure."""
    print("\n" + "=" * 80)
    print("EARLY SLOPE ANALYSIS: Does step 1→2241→4481 growth rate predict convergence?")
    print("=" * 80)

    # Use Block 1 (iters 1-4, DAL=35, all converged) vs known-failing DAL=35 seeds
    # Known DAL=35 from Block 6/7/8 controls
    all_iters = list(range(1, 85))

    slopes = []
    for i in all_iters:
        result = load_trajectory(i)
        if result is None:
            continue
        steps, conn_r2 = result
        if len(conn_r2) < 3:
            continue
        # Early slope: (R2@step4481 - R2@step2241) / (4481 - 2241)
        slope = (conn_r2[2] - conn_r2[1]) / (steps[2] - steps[1])
        # Initial value at step 2241
        init_val = conn_r2[1]
        # Final value
        final_val = conn_r2[-1]
        slopes.append((i, init_val, slope, final_val))

    slopes.sort(key=lambda x: x[3], reverse=True)
    print(f"\n{'Iter':>4} {'R2@2241':>8} {'Slope(2→4k)':>12} {'Final traj':>10}")
    print("-" * 40)
    for i, init_val, slope, final_val in slopes:
        marker = " *" if final_val < 0.90 else ""
        print(f"{i:>4} {init_val:>8.4f} {slope*1e4:>12.4f} {final_val:>10.4f}{marker}")

    print("\n* = trajectory endpoint < 0.90")

    # Correlation between early metrics and final
    inits = np.array([s[1] for s in slopes])
    slps = np.array([s[2] for s in slopes])
    finals = np.array([s[3] for s in slopes])

    corr_init_final = np.corrcoef(inits, finals)[0, 1]
    corr_slope_final = np.corrcoef(slps, finals)[0, 1]
    print(f"\nCorrelation (R2@2241, final traj): {corr_init_final:.3f}")
    print(f"Correlation (slope@2-4k, final traj): {corr_slope_final:.3f}")


if __name__ == "__main__":
    check_slot_structure()
    check_results_log()
    analyze_early_slope()
