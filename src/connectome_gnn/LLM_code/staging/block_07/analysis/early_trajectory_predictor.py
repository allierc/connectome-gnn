"""Analyze whether early-training conn_R2 predicts final convergence.

Hypothesis: seeds that enter the degenerate basin show lower conn_R2 at
an early checkpoint (e.g. step ~2000-6000) than seeds that converge.
If confirmed, an early-stopping + restart mechanism could eliminate the
~20% failure rate.
"""
import os
import csv
import numpy as np

LOG_DIR = "/groups/saalfeld/home/allierc/GraphData/log/Claude_exploration/LLM_flyvis_noise_005_from_zero/r2_trajectory"

# Known final conn_R2 from memory (all iterations with trajectory data)
FINAL_CONN_R2 = {
    # Block 1 (DAL=35, baseline)
    1: 0.972, 2: 0.978, 3: 0.970, 4: 0.969,
    # Block 2 (DAL=95)
    5: 0.9718, 6: 0.8193, 7: 0.9753, 8: 0.9322,
    9: 0.7354, 10: 0.9632, 11: 0.9766, 12: 0.9790,
    # Block 3 (DAL=95)
    13: 0.8604, 14: 0.9348, 15: 0.9615, 16: 0.9667,
    17: 0.9783, 18: 0.9752, 19: 0.9411, 20: 0.9724,
    21: 0.9776, 22: 0.9538, 23: 0.9755, 24: 0.7660,
    # Block 4 (DAL=95)
    25: 0.4926, 26: 0.8687, 27: 0.8946, 28: 0.9719,
    29: 0.9147, 30: 0.8756, 31: 0.9809, 32: 0.9670,
    33: 0.9551, 34: 0.9436, 35: 0.9593, 36: 0.9746,
    # Block 5 (DAL=95)
    37: 0.9651, 38: 0.9675, 39: 0.6404, 40: 0.9723,
    41: 0.9240, 42: 0.8616, 43: 0.9674, 44: 0.9756,
    # Block 6 (various DAL)
    45: 0.9094, 46: 0.8900, 47: 0.9756, 48: 0.5816,
    49: 0.8614, 50: 0.9649, 51: 0.9297, 52: 0.9651,
    53: 0.9591, 54: 0.8491, 55: 0.9660, 56: 0.9708,
    # Block 7
    57: 0.9722, 58: 0.9575, 59: 0.9701, 60: 0.9487,
    61: 0.9139, 62: 0.1873, 63: 0.9011, 64: 0.9784,
    65: 0.8550, 66: 0.9165, 67: 0.8492, 68: 0.9041,
    69: 0.805, 70: 0.805, 71: 0.805, 72: 0.805,  # Block 7 robustness
}

# DAL values per iteration
ITER_DAL = {
    1: 35, 2: 35, 3: 35, 4: 35,
    5: 95, 6: 95, 7: 95, 8: 95, 9: 95, 10: 95, 11: 95, 12: 95,
    13: 95, 14: 95, 15: 95, 16: 95, 17: 95, 18: 95, 19: 95, 20: 95,
    21: 95, 22: 95, 23: 95, 24: 95,
    25: 95, 26: 95, 27: 95, 28: 95, 29: 95, 30: 95, 31: 95, 32: 95,
    33: 95, 34: 95, 35: 95, 36: 95,
    37: 95, 38: 95, 39: 95, 40: 95, 41: 95, 42: 95, 43: 95, 44: 95,
    45: 95, 46: 50, 47: 35, 48: 70,
    49: 35, 50: 40, 51: 45, 52: 55,
    53: 35, 54: 30, 55: 25, 56: 40,
    57: 35, 58: 35, 59: 35, 60: 35,
    61: 35, 62: 35, 63: 35, 64: 35,
    65: 35, 66: 35, 67: 35, 68: 35,
    69: 35, 70: 35, 71: 35, 72: 35,
}


def load_trajectory(iter_num):
    path = os.path.join(LOG_DIR, f"iter_{iter_num:03d}.log")
    if not os.path.exists(path):
        return None
    steps, conn_r2, vrest_r2, tau_r2 = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['iteration']))
            conn_r2.append(float(row['connectivity_r2']))
            vrest_r2.append(float(row['vrest_r2']))
            tau_r2.append(float(row['tau_r2']))
    return np.array(steps), np.array(conn_r2), np.array(vrest_r2), np.array(tau_r2)


def find_r2_at_step(steps, values, target_step):
    """Find value closest to target_step."""
    idx = np.argmin(np.abs(steps - target_step))
    return values[idx], steps[idx]


def main():
    print("=" * 80)
    print("EARLY TRAJECTORY ANALYSIS: Does step-2241 conn_R2 predict convergence?")
    print("=" * 80)

    # Focus on DAL=35 seeds (our target operating point)
    dal35_iters = [i for i, d in ITER_DAL.items() if d == 35]

    print(f"\n--- DAL=35 seeds ({len(dal35_iters)} iterations) ---")
    print(f"{'Iter':>4} {'Final':>8} {'@2241':>8} {'@4481':>8} {'@8961':>8} {'Converged':>10}")

    early_converged = []
    early_failed = []

    for it in sorted(dal35_iters):
        result = load_trajectory(it)
        if result is None:
            continue
        steps, conn_r2, _, _ = result
        final = FINAL_CONN_R2.get(it, conn_r2[-1])
        converged = final >= 0.95

        r2_2241, _ = find_r2_at_step(steps, conn_r2, 2241)
        r2_4481, _ = find_r2_at_step(steps, conn_r2, 4481)
        r2_8961, _ = find_r2_at_step(steps, conn_r2, 8961)

        print(f"{it:4d} {final:8.4f} {r2_2241:8.4f} {r2_4481:8.4f} {r2_8961:8.4f} {'YES' if converged else 'NO':>10}")

        if converged:
            early_converged.append((r2_2241, r2_4481, r2_8961))
        else:
            early_failed.append((r2_2241, r2_4481, r2_8961))

    # Statistical separation
    if early_converged and early_failed:
        conv = np.array(early_converged)
        fail = np.array(early_failed)
        for col, name in [(0, "step~2241"), (1, "step~4481"), (2, "step~8961")]:
            conv_mean = np.mean(conv[:, col])
            conv_std = np.std(conv[:, col])
            fail_mean = np.mean(fail[:, col])
            fail_std = np.std(fail[:, col])
            gap = conv_mean - fail_mean
            # Cohen's d
            pooled_std = np.sqrt((conv_std**2 + fail_std**2) / 2) if (conv_std + fail_std) > 0 else 1e-9
            d = gap / pooled_std

            # Find best threshold
            all_vals = np.concatenate([conv[:, col], fail[:, col]])
            labels = np.concatenate([np.ones(len(conv)), np.zeros(len(fail))])
            best_acc, best_thresh = 0, 0
            for thresh in np.linspace(np.min(all_vals), np.max(all_vals), 100):
                pred = (all_vals >= thresh).astype(int)
                acc = np.mean(pred == labels)
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = thresh

            print(f"\n{name}: conv={conv_mean:.4f}±{conv_std:.4f}, fail={fail_mean:.4f}±{fail_std:.4f}, "
                  f"gap={gap:.4f}, Cohen_d={d:.2f}, best_thresh={best_thresh:.4f}, acc={best_acc:.1%}")

    # Now do the same for ALL seeds (all DALs)
    print("\n" + "=" * 80)
    print("ALL SEEDS (any DAL)")
    print("=" * 80)

    all_converged_early = []
    all_failed_early = []

    for it in sorted(FINAL_CONN_R2.keys()):
        result = load_trajectory(it)
        if result is None:
            continue
        steps, conn_r2, _, tau_r2 = result
        final = FINAL_CONN_R2[it]

        # Find value at approximate early checkpoint
        for target in [2241, 6081]:
            r2_val, actual_step = find_r2_at_step(steps, conn_r2, target)
            if abs(actual_step - target) < 3000:
                if final >= 0.95:
                    all_converged_early.append(r2_val)
                elif final < 0.90:
                    all_failed_early.append(r2_val)
                break

    if all_converged_early and all_failed_early:
        conv_arr = np.array(all_converged_early)
        fail_arr = np.array(all_failed_early)
        print(f"\nConverged (N={len(conv_arr)}): early R2 = {np.mean(conv_arr):.4f} ± {np.std(conv_arr):.4f}")
        print(f"Failed    (N={len(fail_arr)}): early R2 = {np.mean(fail_arr):.4f} ± {np.std(fail_arr):.4f}")
        gap = np.mean(conv_arr) - np.mean(fail_arr)
        pooled = np.sqrt((np.std(conv_arr)**2 + np.std(fail_arr)**2) / 2)
        print(f"Gap = {gap:.4f}, Cohen's d = {gap/pooled:.2f}")

    # Check DAL=95 overtraining pattern: peak R2 vs final R2
    print("\n" + "=" * 80)
    print("DAL=95 OVERTRAINING: Peak conn_R2 vs Final conn_R2")
    print("=" * 80)
    print(f"{'Iter':>4} {'Peak':>8} {'PeakStep':>10} {'Final':>8} {'Δ':>8} {'Type':>15}")

    dal95_iters = [i for i, d in ITER_DAL.items() if d == 95]
    overtrain_count = 0
    early_fail_count = 0
    converge_count = 0

    for it in sorted(dal95_iters):
        result = load_trajectory(it)
        if result is None:
            continue
        steps, conn_r2, _, _ = result
        peak_idx = np.argmax(conn_r2)
        peak_r2 = conn_r2[peak_idx]
        peak_step = steps[peak_idx]
        final_r2 = conn_r2[-1]
        delta = final_r2 - peak_r2

        if peak_r2 >= 0.95 and final_r2 < 0.90:
            ftype = "OVERTRAIN"
            overtrain_count += 1
        elif peak_r2 < 0.85:
            ftype = "EARLY-FAIL"
            early_fail_count += 1
        elif final_r2 >= 0.95:
            ftype = "CONVERGED"
            converge_count += 1
        else:
            ftype = "PARTIAL"

        if final_r2 < 0.95 or it <= 5:  # Print interesting cases
            print(f"{it:4d} {peak_r2:8.4f} {peak_step:10d} {final_r2:8.4f} {delta:8.4f} {ftype:>15}")

    total_95 = overtrain_count + early_fail_count + converge_count
    print(f"\nDAL=95 failure modes: {overtrain_count} overtrain, {early_fail_count} early-fail, "
          f"{converge_count} converged (of {len(dal95_iters)} total)")

    # DAL=35 trajectory shape: check if failing seeds show non-monotonic R2
    print("\n" + "=" * 80)
    print("DAL=35 TRAJECTORY SHAPE: Monotonicity analysis")
    print("=" * 80)

    for it in sorted(dal35_iters):
        result = load_trajectory(it)
        if result is None:
            continue
        steps, conn_r2, _, tau_r2 = result
        final = FINAL_CONN_R2.get(it, conn_r2[-1])

        # Count non-monotonic transitions
        diffs = np.diff(conn_r2[1:])  # skip initial 0
        n_decreases = np.sum(diffs < -0.01)
        max_drop = np.min(diffs) if len(diffs) > 0 else 0

        # tau_R2 correlation with convergence
        final_tau = tau_r2[-1] if len(tau_r2) > 0 else 0

        converged = "YES" if final >= 0.95 else "NO"
        if final < 0.95 or it <= 4:
            print(f"Iter {it:3d}: final={final:.4f}, tau={final_tau:.3f}, "
                  f"decreases={n_decreases}, max_drop={max_drop:.4f}, converged={converged}")


if __name__ == "__main__":
    main()
