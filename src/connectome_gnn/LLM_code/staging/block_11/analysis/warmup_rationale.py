"""Quantify the early-training bifurcation to justify lr_W warmup.

Key question: Is the conn_R2 gap at step 2241 (first checkpoint) large enough
that a simple threshold separates converged vs failed seeds? If yes, the
bifurcation is committed within ~2241 optimizer steps — lr_W warmup over those
steps could prevent it.
"""
import os
import csv
import numpy as np

LOG_DIR = "/groups/saalfeld/home/allierc/GraphData/log/Claude_exploration/LLM_flyvis_noise_005_from_zero/r2_trajectory"

# DAL=35 baseline iterations (from memory) — these are pure baseline (no param changes)
BASELINE_DAL35 = {
    # Block 1 (DAL=35 robustness)
    1: 0.972, 2: 0.978, 3: 0.970, 4: 0.969,
    # Block 6 DAL=35 controls
    47: 0.9756, 49: 0.8614, 53: 0.9591,
    # Block 7 DAL=35 window=1 controls
    57: 0.9722, 61: 0.9139, 65: 0.8550,
    # Block 8 robustness (all DAL=35 baseline)
    73: 0.9735, 74: 0.7437, 75: 0.9553, 76: 0.8327,
    77: 0.9782, 78: 0.7949, 79: 0.9667, 80: 0.8604,
    81: 0.9688, 82: 0.9651, 83: 0.8050, 84: 0.9252,
}

CONVERGE_THRESHOLD = 0.90


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
    print("EARLY BIFURCATION ANALYSIS — DAL=35 baseline seeds")
    print("Hypothesis: lr_W warmup over first 2241 steps prevents degenerate basin entry")
    print("=" * 80)

    converged = []  # (iter, step2241_r2, final_r2)
    failed = []

    for iter_num, final_r2 in sorted(BASELINE_DAL35.items()):
        result = load_trajectory(iter_num)
        if result is None:
            print(f"  Iter {iter_num}: NO TRAJECTORY FILE")
            continue
        steps, conn_r2 = result
        # Get conn_R2 at step 2241 (first real checkpoint after step 1)
        if len(steps) < 2:
            continue
        r2_at_2241 = conn_r2[1]  # index 1 = step 2241
        r2_at_4481 = conn_r2[2] if len(conn_r2) > 2 else np.nan
        entry = (iter_num, r2_at_2241, r2_at_4481, final_r2)
        if final_r2 >= CONVERGE_THRESHOLD:
            converged.append(entry)
        else:
            failed.append(entry)

    print(f"\nConverged: {len(converged)} seeds, Failed: {len(failed)} seeds")
    print(f"\n{'Iter':>4} {'@2241':>8} {'@4481':>8} {'Final':>8} {'Status':>10}")
    print("-" * 45)
    for it, r2241, r4481, final in sorted(converged + failed, key=lambda x: x[1]):
        status = "CONV" if final >= CONVERGE_THRESHOLD else "FAIL"
        print(f"{it:>4} {r2241:>8.4f} {r4481:>8.4f} {final:>8.4f} {status:>10}")

    if converged and failed:
        conv_2241 = np.array([x[1] for x in converged])
        fail_2241 = np.array([x[1] for x in failed])
        conv_4481 = np.array([x[2] for x in converged])
        fail_4481 = np.array([x[2] for x in failed])

        print(f"\n--- Step 2241 statistics ---")
        print(f"Converged: mean={conv_2241.mean():.4f}, std={conv_2241.std():.4f}, "
              f"min={conv_2241.min():.4f}, max={conv_2241.max():.4f}")
        print(f"Failed:    mean={fail_2241.mean():.4f}, std={fail_2241.std():.4f}, "
              f"min={fail_2241.min():.4f}, max={fail_2241.max():.4f}")
        gap = conv_2241.min() - fail_2241.max()
        print(f"Gap (min_conv - max_fail): {gap:.4f}")
        print(f"Separable at step 2241? {'YES' if gap > 0 else 'NO (overlapping)'}")

        print(f"\n--- Step 4481 statistics ---")
        print(f"Converged: mean={conv_4481.mean():.4f}, std={conv_4481.std():.4f}, "
              f"min={conv_4481.min():.4f}")
        print(f"Failed:    mean={fail_4481.mean():.4f}, std={fail_4481.std():.4f}, "
              f"max={fail_4481.max():.4f}")
        gap4 = conv_4481.min() - fail_4481.max()
        print(f"Gap (min_conv - max_fail): {gap4:.4f}")
        print(f"Separable at step 4481? {'YES' if gap4 > 0 else 'NO (overlapping)'}")

        # Growth rate analysis: step 2241 → 4481
        conv_growth = conv_4481 - conv_2241
        fail_growth = fail_4481 - fail_2241
        print(f"\n--- Growth rate (2241→4481) ---")
        print(f"Converged: mean={conv_growth.mean():.4f}, std={conv_growth.std():.4f}")
        print(f"Failed:    mean={fail_growth.mean():.4f}, std={fail_growth.std():.4f}")
        print(f"Growth difference: {conv_growth.mean() - fail_growth.mean():.4f}")

        # Warmup analysis: if we had clipped early W updates,
        # would the failing seeds have had smaller initial divergence?
        print(f"\n--- Warmup rationale ---")
        print(f"All seeds start at conn_R2 ≈ 0 (random W)")
        print(f"After 2241 steps: CONV seeds reach {conv_2241.mean():.3f}, "
              f"FAIL seeds only reach {fail_2241.mean():.3f}")
        print(f"Deficit at step 2241: {conv_2241.mean() - fail_2241.mean():.3f}")
        print(f"This {conv_2241.mean() - fail_2241.mean():.3f} deficit is {100*(conv_2241.mean() - fail_2241.mean())/conv_2241.mean():.1f}% "
              f"of converged step-2241 value")
        print(f"\nIf lr_W warmup slows initial W updates, it gives g_phi/f_theta time")
        print(f"to co-adapt before W commits to a basin. The first 2241 steps are")
        print(f"the critical window where warmup would have maximum effect.")

        # Total training steps for DAL=35
        # batch_size=4, n_frames=64000 → samples_per_epoch = 64000
        # DAL=35 → effective epochs ≈ 35
        # Steps per DAL loop = n_frames / batch_size = 64000/4 = 16000? No...
        # Actually steps = DAL * (n_train_samples / batch_size)
        # From trajectories: max step for DAL=35 is the last entry
        result = load_trajectory(1)
        if result is not None:
            total_steps = result[0][-1]
            print(f"\nTotal training steps (DAL=35): {total_steps}")
            print(f"Critical window: first 2241 steps = {100*2241/total_steps:.1f}% of training")
            print(f"Proposed warmup: linear lr_W from 0→target over 2241 steps")


if __name__ == "__main__":
    main()
