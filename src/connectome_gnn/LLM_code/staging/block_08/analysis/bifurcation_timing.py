"""Analyze WHEN the convergence/failure bifurcation happens in training.

For all DAL=35 baseline seeds (the target config), compare the r2 trajectory
of converging seeds (final conn_R2 >= 0.90) vs failing seeds (< 0.90).

Key questions:
1. At which training step does the bifurcation become detectable?
2. Is there a critical early step where conn_R2 separates?
3. Do failing seeds show a PLATEAU (stuck) or REVERSAL (overtraining)?
"""
import os
import csv
import numpy as np

LOG_DIR = "/groups/saalfeld/home/allierc/GraphData/log/Claude_exploration/LLM_flyvis_noise_005_from_zero/r2_trajectory"

# All DAL=35 baseline iterations and their final conn_R2 (from memory)
# Block 1 (DAL=35 baseline robustness)
# Block 6-8 (DAL=35 runs)
BASELINE_DAL35 = {
    # Block 1
    1: 0.972, 2: 0.978, 3: 0.970, 4: 0.969,
    # Block 6 DAL=35 controls
    47: 0.9756, 49: 0.8614, 53: 0.9591,
    # Block 7 DAL=35 controls (window=1)
    57: 0.9722, 61: 0.9139, 65: 0.8550,
    # Block 8 robustness (all DAL=35 baseline)
    73: 0.9735, 74: 0.7437, 75: 0.9553, 76: 0.8327,
    77: 0.9782, 78: 0.7949, 79: 0.9667, 80: 0.8604,
    81: 0.9688, 82: 0.9651, 83: 0.8050, 84: 0.9252,
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
            vrest_r2.append(float(row.get('vrest_r2', '0')))
            tau_r2.append(float(row.get('tau_r2', '0')))
    return np.array(steps), np.array(conn_r2), np.array(vrest_r2), np.array(tau_r2)


def main():
    print("=" * 80)
    print("BIFURCATION TIMING ANALYSIS — DAL=35 baseline seeds")
    print("=" * 80)

    converged_trajs = []  # final >= 0.90
    failed_trajs = []     # final < 0.90

    for iter_num, final_r2 in sorted(BASELINE_DAL35.items()):
        result = load_trajectory(iter_num)
        if result is None:
            print(f"  Iter {iter_num}: NO TRAJECTORY FILE")
            continue
        steps, conn_r2, vrest_r2, tau_r2 = result
        if final_r2 >= 0.90:
            converged_trajs.append((iter_num, steps, conn_r2, vrest_r2, tau_r2))
        else:
            failed_trajs.append((iter_num, steps, conn_r2, vrest_r2, tau_r2))

    print(f"\nConverged seeds: {len(converged_trajs)}, Failed seeds: {len(failed_trajs)}")

    # Find common step grid (DAL=35 → ~35 * 2241 ≈ steps up to ~78435)
    # Use the minimum trajectory length
    if not converged_trajs or not failed_trajs:
        print("ERROR: need both converged and failed seeds")
        return

    # Get step values from first trajectory
    ref_steps = converged_trajs[0][1]
    n_steps = min(len(t[1]) for t in converged_trajs + failed_trajs)
    common_steps = ref_steps[:n_steps]

    print(f"\nCommon trajectory length: {n_steps} checkpoints, "
          f"steps {common_steps[0]}–{common_steps[-1]}")

    # Build matrices
    conv_matrix = np.array([t[2][:n_steps] for t in converged_trajs])
    fail_matrix = np.array([t[2][:n_steps] for t in failed_trajs])

    print(f"\n{'Step':>8} | {'Conv mean':>10} {'Conv std':>9} | "
          f"{'Fail mean':>10} {'Fail std':>9} | {'Gap':>8} {'Separable?':>10}")
    print("-" * 85)

    bifurcation_step = None
    for i, step in enumerate(common_steps):
        if i == 0:
            continue  # skip step 0
        c_mean = conv_matrix[:, i].mean()
        c_std = conv_matrix[:, i].std()
        f_mean = fail_matrix[:, i].mean()
        f_std = fail_matrix[:, i].std()
        gap = c_mean - f_mean
        # Check if distributions are separated (gap > sum of stds)
        separable = gap > (c_std + f_std)
        sep_str = "YES" if separable else "no"
        if separable and bifurcation_step is None:
            bifurcation_step = step
            sep_str = "** FIRST **"
        print(f"{step:>8} | {c_mean:>10.4f} {c_std:>9.4f} | "
              f"{f_mean:>10.4f} {f_std:>9.4f} | {gap:>8.4f} {sep_str:>10}")

    if bifurcation_step:
        print(f"\n** Bifurcation first detectable at step {bifurcation_step} **")
    else:
        print("\n** Bifurcation never cleanly separable within trajectory **")

    # Detailed per-seed view at key checkpoints
    print("\n" + "=" * 80)
    print("PER-SEED conn_R2 AT KEY CHECKPOINTS")
    print("=" * 80)

    checkpoints = [1, 2, 3, 4, 5]  # indices into common_steps (skip 0)
    if n_steps > 10:
        checkpoints.extend([n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1])
    checkpoints = sorted(set(c for c in checkpoints if c < n_steps))

    header = f"{'Iter':>4} {'Final':>7} {'Class':>6}"
    for ci in checkpoints:
        header += f" | step{common_steps[ci]:>6}"
    print(header)
    print("-" * len(header))

    for iter_num, steps, conn_r2, _, _ in converged_trajs:
        row = f"{iter_num:>4} {BASELINE_DAL35[iter_num]:>7.4f} {'CONV':>6}"
        for ci in checkpoints:
            row += f" | {conn_r2[ci]:>10.4f}"
        print(row)

    print("-" * len(header))
    for iter_num, steps, conn_r2, _, _ in failed_trajs:
        row = f"{iter_num:>4} {BASELINE_DAL35[iter_num]:>7.4f} {'FAIL':>6}"
        for ci in checkpoints:
            row += f" | {conn_r2[ci]:>10.4f}"
        print(row)

    # Analyze: is there a conn_R2 threshold at an early step that predicts failure?
    print("\n" + "=" * 80)
    print("EARLY PREDICTION: Can step-2241 conn_R2 predict failure?")
    print("=" * 80)

    if n_steps >= 2:
        step_idx = 1  # step 2241
        conv_early = conv_matrix[:, step_idx]
        fail_early = fail_matrix[:, step_idx]
        print(f"\nAt step {common_steps[step_idx]}:")
        print(f"  Converged: mean={conv_early.mean():.4f}, "
              f"min={conv_early.min():.4f}, max={conv_early.max():.4f}")
        print(f"  Failed:    mean={fail_early.mean():.4f}, "
              f"min={fail_early.min():.4f}, max={fail_early.max():.4f}")
        # Find optimal threshold
        all_early = np.concatenate([conv_early, fail_early])
        labels = np.concatenate([np.ones(len(conv_early)), np.zeros(len(fail_early))])
        best_acc, best_thresh = 0, 0
        for thresh in np.linspace(all_early.min(), all_early.max(), 100):
            pred = (all_early >= thresh).astype(float)
            acc = (pred == labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        print(f"  Best threshold: {best_thresh:.4f} → accuracy {best_acc:.1%}")
        print(f"  (Predicts convergence if conn_R2 >= {best_thresh:.4f} at step {common_steps[step_idx]})")

    # Analyze trajectory SHAPE: do failing seeds plateau or reverse?
    print("\n" + "=" * 80)
    print("TRAJECTORY SHAPE: Plateau vs Reversal in failing seeds")
    print("=" * 80)

    for iter_num, steps, conn_r2, _, _ in failed_trajs:
        if len(conn_r2) < 3:
            continue
        peak_idx = np.argmax(conn_r2[1:]) + 1  # skip step 0
        peak_val = conn_r2[peak_idx]
        final_val = conn_r2[-1]
        decline = peak_val - final_val
        print(f"  Iter {iter_num}: peak={peak_val:.4f} at step {steps[peak_idx]}, "
              f"final={final_val:.4f}, decline={decline:+.4f} "
              f"({'REVERSAL' if decline > 0.05 else 'PLATEAU' if decline > 0.01 else 'STUCK'})")


if __name__ == "__main__":
    main()
