"""Analyze r2 trajectory files to determine WHEN the bifurcation happens.

Key question: At what training step do failed seeds diverge from converged seeds?
If divergence happens in the first 10-20% of steps, a warmup on MLP lr could help
by preventing f_theta from absorbing W's role early.
"""
import os
import csv
import numpy as np

LOG_DIR = "/groups/saalfeld/home/allierc/GraphData/log/Claude_exploration/LLM_flyvis_noise_005_from_zero/r2_trajectory"

# From memory: known final conn_R2 for each iteration
FINAL_R2 = {
    1: 0.972, 2: 0.972, 3: 0.972, 4: 0.972,  # Block 1, DAL=35
    5: 0.9718, 6: 0.8193, 7: 0.9753, 8: 0.9322,  # Block 2
    9: 0.7354, 10: 0.9632, 11: 0.9766, 12: 0.979,
    13: 0.8604, 14: 0.9348, 15: 0.9615, 16: 0.9667,
    17: 0.9783, 18: 0.9752, 19: 0.9411, 20: 0.9724,
    21: 0.9776, 22: 0.9538, 23: 0.9755, 24: 0.766,
    25: 0.4926, 26: 0.8687, 27: 0.8946, 28: 0.9719,
    29: 0.9147, 30: 0.8756, 31: 0.9809, 32: 0.967,
    33: 0.9551, 34: 0.9436, 35: 0.9593, 36: 0.9746,
    37: 0.9651, 38: 0.9675, 39: 0.6404, 40: 0.9723,
    41: 0.924, 42: 0.8616, 43: 0.9674, 44: 0.9756,
    45: 0.9094, 46: 0.89, 47: 0.9756, 48: 0.5816,
    49: 0.8614, 50: 0.9649, 51: 0.9297, 52: 0.9651,
    53: 0.9591, 54: 0.8491, 55: 0.966, 56: 0.9708,
    57: 0.9722, 58: 0.9575, 59: 0.9701, 60: 0.9487,
}

# DAL values (determines total steps = DAL * batch_steps)
ITER_DAL = {
    **{i: 35 for i in range(1, 5)},
    **{i: 95 for i in range(5, 45)},
    45: 95, 46: 50, 47: 35, 48: 70,
    49: 35, 50: 40, 51: 45, 52: 55,
    53: 35, 54: 30, 55: 25, 56: 40,
    57: 35, 58: 35, 59: 35, 60: 35,
}


def load_trajectory(iter_num):
    """Load r2 trajectory for an iteration."""
    path = os.path.join(LOG_DIR, f"iter_{iter_num:03d}.log")
    if not os.path.exists(path):
        return None, None
    steps = []
    conn_r2 = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['iteration']))
            conn_r2.append(float(row['connectivity_r2']))
    if not steps:
        return None, None
    return np.array(steps), np.array(conn_r2)


def analyze():
    """Compare trajectories of converged vs failed seeds at DAL=95."""
    print("=" * 70)
    print("BIFURCATION TIMING ANALYSIS")
    print("=" * 70)

    # Focus on DAL=95 seeds (most data, longest trajectories)
    converged = []  # final >= 0.95
    failed = []     # final < 0.90

    for it in range(5, 45):
        if it not in FINAL_R2:
            continue
        steps, r2 = load_trajectory(it)
        if steps is None:
            continue
        fr2 = FINAL_R2[it]
        if fr2 >= 0.95:
            converged.append((it, steps, r2))
        elif fr2 < 0.90:
            failed.append((it, steps, r2))

    print(f"\nDAL=95 seeds: {len(converged)} converged (R2>=0.95), {len(failed)} failed (R2<0.90)")

    # Find the earliest step where failed seeds are distinguishable
    # Check at 10%, 20%, 30%, 50% of typical trajectory length
    if not converged or not failed:
        print("Not enough data for comparison")
        return

    # Get common step count (shortest trajectory)
    min_steps_converged = min(len(r2) for _, _, r2 in converged)
    min_steps_failed = min(len(r2) for _, _, r2 in failed)
    print(f"Trajectory lengths: converged min={min_steps_converged}, failed min={min_steps_failed}")

    # Sample at specific fractions of the shortest converged trajectory
    ref_len = min(min_steps_converged, min_steps_failed)
    checkpoints = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]

    print(f"\n{'Fraction':<10} {'Step':<8} {'Conv mean':<12} {'Conv std':<10} {'Fail mean':<12} {'Fail std':<10} {'Separable?':<12}")
    print("-" * 74)

    for frac in checkpoints:
        idx = min(int(frac * ref_len) - 1, ref_len - 1)
        if idx < 0:
            idx = 0

        conv_vals = [r2[idx] for _, _, r2 in converged if len(r2) > idx]
        fail_vals = [r2[idx] for _, _, r2 in failed if len(r2) > idx]

        if not conv_vals or not fail_vals:
            continue

        conv_mean = np.mean(conv_vals)
        conv_std = np.std(conv_vals)
        fail_mean = np.mean(fail_vals)
        fail_std = np.std(fail_vals)

        # Get the actual step number
        step_num = converged[0][1][idx] if len(converged[0][1]) > idx else "?"

        # Separable if means are > 2 std apart
        gap = conv_mean - fail_mean
        pooled_std = np.sqrt((conv_std**2 + fail_std**2) / 2) if (conv_std + fail_std) > 0 else 1
        separable = "YES" if gap > 2 * pooled_std else ("MAYBE" if gap > pooled_std else "NO")

        print(f"{frac:<10.2f} {step_num:<8} {conv_mean:<12.4f} {conv_std:<10.4f} {fail_mean:<12.4f} {fail_std:<10.4f} {separable:<12}")

    # Detailed early-trajectory analysis
    print("\n\n" + "=" * 70)
    print("EARLY TRAJECTORY DETAIL (first 20 checkpoints)")
    print("=" * 70)

    n_early = min(20, ref_len)
    print(f"\n{'Step':<8} {'Conv mean':<12} {'Fail mean':<12} {'Delta':<10}")
    print("-" * 42)
    for idx in range(n_early):
        conv_vals = [r2[idx] for _, _, r2 in converged if len(r2) > idx]
        fail_vals = [r2[idx] for _, _, r2 in failed if len(r2) > idx]
        if not conv_vals or not fail_vals:
            continue
        step_num = converged[0][1][idx]
        print(f"{step_num:<8} {np.mean(conv_vals):<12.4f} {np.mean(fail_vals):<12.4f} {np.mean(conv_vals)-np.mean(fail_vals):<10.4f}")

    # Check individual failed trajectories for early peak + decline
    print("\n\n" + "=" * 70)
    print("FAILED SEED TRAJECTORIES (early dynamics)")
    print("=" * 70)
    for it, steps, r2 in failed:
        peak_idx = np.argmax(r2[:min(len(r2), ref_len//2)])
        peak_val = r2[peak_idx]
        peak_step = steps[peak_idx]
        final_val = r2[-1]
        # Find first time R2 drops below 0.5 (if ever)
        below_half = np.where(r2 < 0.5)[0]
        drop_step = steps[below_half[0]] if len(below_half) > 0 else "never"
        print(f"  Iter {it}: peak={peak_val:.4f}@step{peak_step}, final={final_val:.4f}, drops<0.5@step={drop_step}")


if __name__ == "__main__":
    analyze()
