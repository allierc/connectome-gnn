"""Characterize the TWO failure modes: low-start vs regression.

Some fail seeds start poorly (@2241 < 0.85) — "early commitment" failures.
Others start WELL (@2241 > 0.85) but REGRESS later — "overtraining" failures.

This analysis determines: are there two distinct mechanisms, and which dominates?
"""
import os
import csv
import numpy as np

LOG_DIR = "/groups/saalfeld/home/allierc/GraphData/log/Claude_exploration/LLM_flyvis_noise_005_from_zero/r2_trajectory"

BASELINE_DAL35 = {
    1: 0.972, 2: 0.978, 3: 0.970, 4: 0.969,
    47: 0.9756, 49: 0.8614, 53: 0.9591,
    57: 0.9722, 61: 0.9139, 65: 0.8550,
    73: 0.9735, 74: 0.7437, 75: 0.9553, 76: 0.8327,
    77: 0.9782, 78: 0.7949, 79: 0.9667, 80: 0.8604,
    81: 0.9688, 82: 0.9651, 83: 0.8050, 84: 0.9252,
}


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


def main():
    print("=" * 80)
    print("FAILURE MODE ANALYSIS — Low-start vs Regression")
    print("=" * 80)

    for iter_num, final_r2 in sorted(BASELINE_DAL35.items()):
        if final_r2 >= 0.90:
            continue  # only look at failures
        result = load_trajectory(iter_num)
        if result is None:
            continue
        steps, conn_r2 = result
        peak_idx = np.argmax(conn_r2)
        peak_r2 = conn_r2[peak_idx]
        peak_step = steps[peak_idx]
        regression = peak_r2 - final_r2
        start_2241 = conn_r2[1] if len(conn_r2) > 1 else np.nan

        # Classify
        if start_2241 < 0.83:
            mode = "LOW-START"
        elif peak_r2 - final_r2 > 0.05:
            mode = "REGRESSION"
        else:
            mode = "PLATEAU"

        print(f"\nIter {iter_num} [{mode}]: final={final_r2:.4f}")
        print(f"  @2241={start_2241:.4f}, peak={peak_r2:.4f} (step {peak_step}), "
              f"regression={regression:.4f}")
        # Print trajectory milestones
        milestones = [0, 1, 2, 5, 10, 15, 20, len(conn_r2)-1]
        for mi in milestones:
            if mi < len(conn_r2):
                print(f"  step {steps[mi]:>8}: conn_R2={conn_r2[mi]:.4f}")

    # Now check converged seeds for comparison
    print("\n" + "=" * 80)
    print("CONVERGED SEEDS — peak vs final (do they also have regression?)")
    print("=" * 80)

    conv_regressions = []
    for iter_num, final_r2 in sorted(BASELINE_DAL35.items()):
        if final_r2 < 0.90:
            continue
        result = load_trajectory(iter_num)
        if result is None:
            continue
        steps, conn_r2 = result
        peak_idx = np.argmax(conn_r2)
        peak_r2 = conn_r2[peak_idx]
        peak_step = steps[peak_idx]
        regression = peak_r2 - final_r2
        conv_regressions.append(regression)

        if regression > 0.01:  # show any with notable regression
            print(f"\nIter {iter_num}: final={final_r2:.4f}, peak={peak_r2:.4f} "
                  f"(step {peak_step}), regression={regression:.4f}")

    print(f"\nConverged seeds regression: mean={np.mean(conv_regressions):.4f}, "
          f"max={np.max(conv_regressions):.4f}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    fail_iters = {k: v for k, v in BASELINE_DAL35.items() if v < 0.90}
    low_start = 0
    regression = 0
    for it, final in fail_iters.items():
        result = load_trajectory(it)
        if result is None:
            continue
        steps, conn_r2 = result
        if conn_r2[1] < 0.83:
            low_start += 1
        elif np.max(conn_r2) - final > 0.05:
            regression += 1
        else:
            low_start += 1  # bucket plateaus with low-start

    print(f"Total failures: {len(fail_iters)}")
    print(f"  Low-start / plateau: {low_start}")
    print(f"  Regression (peak > final + 0.05): {regression}")
    print(f"\nImplication for lr_W warmup:")
    print(f"  Warmup helps LOW-START failures (slows early bad updates)")
    print(f"  Warmup does NOT help REGRESSION failures (these start well)")
    print(f"  If regression dominates, lr_W DECAY is the correct intervention")


if __name__ == "__main__":
    main()
