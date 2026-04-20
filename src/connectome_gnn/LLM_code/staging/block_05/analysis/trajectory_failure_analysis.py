"""Analyze r2 trajectory files to characterize the overtraining failure mechanism.

Key question: Can early-training conn_R2 dynamics predict whether a seed will
converge or fail? If so, early stopping at a fixed step count (equivalent to
DAL reduction) is the correct robustness mechanism.
"""
import os
import csv
import numpy as np

LOG_DIR = "/groups/saalfeld/home/allierc/GraphData/log/Claude_exploration/LLM_flyvis_noise_005_from_zero/r2_trajectory"

# Known DAL values for each iteration (from memory)
ITER_DAL = {
    # Block 1: DAL=35
    1: 35, 2: 35, 3: 35, 4: 35,
    # Block 2: Iter 5-8 DAL=95 (controls + mutations), Iter 9-12 DAL=95
    5: 95, 6: 95, 7: 95, 8: 95, 9: 95, 10: 95, 11: 95, 12: 95,
    # Block 3: all DAL=95
    13: 95, 14: 95, 15: 95, 16: 95, 17: 95, 18: 95, 19: 95, 20: 95,
    21: 95, 22: 95, 23: 95, 24: 95,
    # Block 4: all DAL=95
    25: 95, 26: 95, 27: 95, 28: 95, 29: 95, 30: 95, 31: 95, 32: 95,
    33: 95, 34: 95, 35: 95, 36: 95,
    # Block 5: all DAL=95
    37: 95, 38: 95, 39: 95, 40: 95, 41: 95, 42: 95, 43: 95, 44: 95,
    # Block 6: DAL varies
    45: 95, 46: 50, 47: 35, 48: 70,
}

def load_trajectory(iter_num):
    """Load r2 trajectory for an iteration."""
    path = os.path.join(LOG_DIR, f"iter_{iter_num:03d}.log")
    if not os.path.exists(path):
        return None
    steps = []
    conn_r2 = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['iteration']))
            conn_r2.append(float(row['connectivity_r2']))
    return np.array(steps), np.array(conn_r2)


def analyze_early_dynamics():
    """For DAL=95 seeds, check if the trajectory at step ~100k
    (equivalent to DAL=35 endpoint) predicts final outcome."""

    print("=" * 70)
    print("TRAJECTORY FAILURE ANALYSIS")
    print("=" * 70)

    # Classify all DAL=95 iterations
    dal95_converged = []  # final conn_R2 >= 0.95
    dal95_failed = []     # final conn_R2 < 0.90
    dal95_partial = []    # 0.90 <= final < 0.95

    for iter_num, dal in ITER_DAL.items():
        if dal != 95:
            continue
        result = load_trajectory(iter_num)
        if result is None:
            continue
        steps, conn_r2 = result
        final_r2 = conn_r2[-1]

        if final_r2 >= 0.95:
            dal95_converged.append((iter_num, steps, conn_r2, final_r2))
        elif final_r2 < 0.90:
            dal95_failed.append((iter_num, steps, conn_r2, final_r2))
        else:
            dal95_partial.append((iter_num, steps, conn_r2, final_r2))

    print(f"\nDAL=95 seeds: {len(dal95_converged)} converged, "
          f"{len(dal95_partial)} partial, {len(dal95_failed)} failed")

    # For each trajectory, find conn_R2 at approximately step 100k (DAL=35 equivalent)
    print("\n" + "-" * 70)
    print("CONN_R2 AT STEP ~100k (DAL=35 EQUIVALENT STOPPING POINT)")
    print("-" * 70)

    def get_r2_at_step(steps, conn_r2, target_step=100000):
        """Get conn_R2 at the closest step to target."""
        idx = np.argmin(np.abs(steps - target_step))
        return conn_r2[idx], steps[idx]

    def get_peak_and_final(steps, conn_r2):
        """Get peak conn_R2 and where it occurs."""
        peak_idx = np.argmax(conn_r2)
        return conn_r2[peak_idx], steps[peak_idx], conn_r2[-1]

    def has_early_decline(steps, conn_r2, threshold=0.02, before_step=60000):
        """Check if conn_R2 decreases by > threshold between consecutive checkpoints before step."""
        for i in range(1, len(steps)):
            if steps[i] > before_step:
                break
            if conn_r2[i] < conn_r2[i-1] - threshold:
                return True, steps[i], conn_r2[i-1] - conn_r2[i]
        return False, None, None

    print("\n--- CONVERGED SEEDS (final >= 0.95) ---")
    r2_at_100k_converged = []
    for iter_num, steps, conn_r2, final_r2 in dal95_converged:
        r2_100k, actual_step = get_r2_at_step(steps, conn_r2)
        peak, peak_step, _ = get_peak_and_final(steps, conn_r2)
        decline, dec_step, dec_mag = has_early_decline(steps, conn_r2)
        r2_at_100k_converged.append(r2_100k)
        dec_str = f"DECLINE at step {dec_step} (Δ={dec_mag:.3f})" if decline else "no early decline"
        print(f"  Iter {iter_num:2d}: r2@100k={r2_100k:.4f} (step {actual_step}), "
              f"peak={peak:.4f}@{peak_step}, final={final_r2:.4f}, {dec_str}")

    print("\n--- PARTIAL SEEDS (0.90 <= final < 0.95) ---")
    r2_at_100k_partial = []
    for iter_num, steps, conn_r2, final_r2 in dal95_partial:
        r2_100k, actual_step = get_r2_at_step(steps, conn_r2)
        peak, peak_step, _ = get_peak_and_final(steps, conn_r2)
        decline, dec_step, dec_mag = has_early_decline(steps, conn_r2)
        r2_at_100k_partial.append(r2_100k)
        dec_str = f"DECLINE at step {dec_step} (Δ={dec_mag:.3f})" if decline else "no early decline"
        print(f"  Iter {iter_num:2d}: r2@100k={r2_100k:.4f} (step {actual_step}), "
              f"peak={peak:.4f}@{peak_step}, final={final_r2:.4f}, {dec_str}")

    print("\n--- FAILED SEEDS (final < 0.90) ---")
    r2_at_100k_failed = []
    for iter_num, steps, conn_r2, final_r2 in dal95_failed:
        r2_100k, actual_step = get_r2_at_step(steps, conn_r2)
        peak, peak_step, _ = get_peak_and_final(steps, conn_r2)
        decline, dec_step, dec_mag = has_early_decline(steps, conn_r2)
        r2_at_100k_failed.append(r2_100k)
        dec_str = f"DECLINE at step {dec_step} (Δ={dec_mag:.3f})" if decline else "no early decline"
        print(f"  Iter {iter_num:2d}: r2@100k={r2_100k:.4f} (step {actual_step}), "
              f"peak={peak:.4f}@{peak_step}, final={final_r2:.4f}, {dec_str}")

    # Statistical comparison
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    if r2_at_100k_converged:
        print(f"\nConverged seeds r2@100k: mean={np.mean(r2_at_100k_converged):.4f}, "
              f"std={np.std(r2_at_100k_converged):.4f}, "
              f"min={np.min(r2_at_100k_converged):.4f}, n={len(r2_at_100k_converged)}")
    if r2_at_100k_partial:
        print(f"Partial seeds r2@100k:   mean={np.mean(r2_at_100k_partial):.4f}, "
              f"std={np.std(r2_at_100k_partial):.4f}, "
              f"min={np.min(r2_at_100k_partial):.4f}, n={len(r2_at_100k_partial)}")
    if r2_at_100k_failed:
        print(f"Failed seeds r2@100k:    mean={np.mean(r2_at_100k_failed):.4f}, "
              f"std={np.std(r2_at_100k_failed):.4f}, "
              f"min={np.min(r2_at_100k_failed):.4f}, n={len(r2_at_100k_failed)}")

    # Key test: would stopping at step 100k (DAL=35) have given good results for ALL seeds?
    all_r2_100k = r2_at_100k_converged + r2_at_100k_partial + r2_at_100k_failed
    print(f"\nALL DAL=95 seeds if stopped at 100k steps:")
    print(f"  mean={np.mean(all_r2_100k):.4f}, std={np.std(all_r2_100k):.4f}, "
          f"min={np.min(all_r2_100k):.4f}, max={np.max(all_r2_100k):.4f}")
    print(f"  Would achieve Stable-Robust (all >= 0.90, CV < 3%): "
          f"{'YES' if np.min(all_r2_100k) >= 0.90 and np.std(all_r2_100k)/np.mean(all_r2_100k)*100 < 3 else 'NO'}")
    print(f"  Seeds with r2@100k >= 0.95: {sum(1 for x in all_r2_100k if x >= 0.95)}/{len(all_r2_100k)}")
    print(f"  Seeds with r2@100k >= 0.90: {sum(1 for x in all_r2_100k if x >= 0.90)}/{len(all_r2_100k)}")

    # Check peak timing
    print("\n" + "=" * 70)
    print("PEAK TIMING ANALYSIS")
    print("=" * 70)
    print("\nDoes training past 100k steps EVER improve conn_R2 for failed seeds?")
    for iter_num, steps, conn_r2, final_r2 in dal95_failed:
        r2_100k, _ = get_r2_at_step(steps, conn_r2)
        peak, peak_step, _ = get_peak_and_final(steps, conn_r2)
        print(f"  Iter {iter_num:2d}: r2@100k={r2_100k:.4f}, peak={peak:.4f}@step{peak_step}, "
              f"final={final_r2:.4f} → {'PEAK AFTER 100k' if peak_step > 100000 else 'PEAK BEFORE 100k'}")

    # Check if blank_prefix_fraction would help by analyzing early transient
    print("\n" + "=" * 70)
    print("EARLY TRANSIENT ANALYSIS (steps 0-20k)")
    print("=" * 70)
    print("\nFailed seeds often show early instability. Checking r2 at step ~20k:")

    all_trajectories = [(iter_num, steps, conn_r2, final_r2, 'converged')
                        for iter_num, steps, conn_r2, final_r2 in dal95_converged]
    all_trajectories += [(iter_num, steps, conn_r2, final_r2, 'partial')
                         for iter_num, steps, conn_r2, final_r2 in dal95_partial]
    all_trajectories += [(iter_num, steps, conn_r2, final_r2, 'failed')
                         for iter_num, steps, conn_r2, final_r2 in dal95_failed]

    r2_20k_by_class = {'converged': [], 'partial': [], 'failed': []}
    for iter_num, steps, conn_r2, final_r2, cls in all_trajectories:
        r2_20k, _ = get_r2_at_step(steps, conn_r2, target_step=20000)
        r2_20k_by_class[cls].append(r2_20k)

    for cls in ['converged', 'partial', 'failed']:
        if r2_20k_by_class[cls]:
            vals = r2_20k_by_class[cls]
            print(f"  {cls:10s}: r2@20k mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")


if __name__ == "__main__":
    analyze_early_dynamics()
