"""Test whether tau_R2 at step 2241 is a stronger early predictor of final
convergence than conn_R2 at the same step.

Uses all DAL=35 baseline seeds with trajectory data.
PASS CONDITION: tau_R2@2241 AUROC >= 0.85 for classifying converging
(final conn_R2 >= 0.90) vs failing seeds, AND tau_R2 AUROC > conn_R2 AUROC.
"""
import os
import csv
import numpy as np

LOG_DIR = "/groups/saalfeld/home/allierc/GraphData/log/Claude_exploration/LLM_flyvis_noise_005_from_zero/r2_trajectory"

# All DAL=35 baseline iterations and their FINAL conn_R2 from the exploration memory
BASELINE_DAL35 = {
    # Block 1 (DAL=35 baseline robustness)
    1: 0.972, 2: 0.978, 3: 0.970, 4: 0.969,
    # Block 6 DAL=35 controls
    47: 0.9756, 49: 0.8614, 53: 0.9591,
    # Block 7 DAL=35 controls (window=1 only)
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
    steps, conn_r2, vrest_r2, tau_r2 = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['iteration']))
            conn_r2.append(float(row['connectivity_r2']))
            vrest_r2.append(float(row.get('vrest_r2', '0')))
            tau_r2.append(float(row.get('tau_r2', '0')))
    return np.array(steps), np.array(conn_r2), np.array(vrest_r2), np.array(tau_r2)


def auroc_manual(labels, scores):
    """Compute AUROC without sklearn. labels=1 for positive (converging)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    # Mann-Whitney U statistic
    count = 0
    for p in pos:
        for n in neg:
            if p > n:
                count += 1
            elif p == n:
                count += 0.5
    return count / (len(pos) * len(neg))


def main():
    print("=" * 80)
    print("TAU_R2 vs CONN_R2 AS EARLY CONVERGENCE PREDICTOR (step 2241)")
    print("=" * 80)

    # Collect early-step metrics for all DAL=35 baseline seeds
    results = []
    for iter_num, final_r2 in sorted(BASELINE_DAL35.items()):
        data = load_trajectory(iter_num)
        if data is None:
            print(f"  Iter {iter_num}: NO TRAJECTORY FILE — skipping")
            continue
        steps, conn_r2, vrest_r2, tau_r2 = data
        # Find step closest to 2241
        idx = np.argmin(np.abs(steps - 2241))
        actual_step = steps[idx]
        converged = 1 if final_r2 >= CONVERGE_THRESHOLD else 0
        results.append({
            'iter': iter_num,
            'final_r2': final_r2,
            'converged': converged,
            'conn_r2_2241': conn_r2[idx],
            'tau_r2_2241': tau_r2[idx],
            'vrest_r2_2241': vrest_r2[idx],
            'step': actual_step,
        })
        status = "CONV" if converged else "FAIL"
        print(f"  Iter {iter_num:3d}: final={final_r2:.4f} [{status}] "
              f"| @{actual_step}: conn={conn_r2[idx]:.4f} tau={tau_r2[idx]:.4f} vrest={vrest_r2[idx]:.4f}")

    if len(results) < 4:
        print(f"\nERROR: only {len(results)} seeds with trajectory data, need >= 4")
        return

    # Compute AUROC for each early metric
    labels = np.array([r['converged'] for r in results])
    conn_scores = np.array([r['conn_r2_2241'] for r in results])
    tau_scores = np.array([r['tau_r2_2241'] for r in results])
    vrest_scores = np.array([r['vrest_r2_2241'] for r in results])

    n_conv = labels.sum()
    n_fail = len(labels) - n_conv
    print(f"\n--- Summary: {len(results)} seeds, {n_conv} converging, {n_fail} failing ---")

    auroc_conn = auroc_manual(labels, conn_scores)
    auroc_tau = auroc_manual(labels, tau_scores)
    auroc_vrest = auroc_manual(labels, vrest_scores)

    print(f"\nAUROC @ step 2241:")
    print(f"  conn_R2:  {auroc_conn:.3f}")
    print(f"  tau_R2:   {auroc_tau:.3f}")
    print(f"  vrest_R2: {auroc_vrest:.3f}")

    # Group stats
    conv_conn = conn_scores[labels == 1]
    fail_conn = conn_scores[labels == 0]
    conv_tau = tau_scores[labels == 1]
    fail_tau = tau_scores[labels == 0]

    print(f"\nConn_R2 @2241: conv={conv_conn.mean():.4f}+/-{conv_conn.std():.4f} "
          f"fail={fail_conn.mean():.4f}+/-{fail_conn.std():.4f} gap={conv_conn.mean()-fail_conn.mean():.4f}")
    print(f"Tau_R2  @2241: conv={conv_tau.mean():.4f}+/-{conv_tau.std():.4f} "
          f"fail={fail_tau.mean():.4f}+/-{fail_tau.std():.4f} gap={conv_tau.mean()-fail_tau.mean():.4f}")

    # Also check later checkpoints (4481, 8961)
    for target_step in [4481, 8961]:
        print(f"\n--- Checkpoint @ step {target_step} ---")
        conn_at = []
        tau_at = []
        for r in results:
            data = load_trajectory(r['iter'])
            steps = data[0]
            idx = np.argmin(np.abs(steps - target_step))
            conn_at.append(data[1][idx])
            tau_at.append(data[3][idx])
        conn_at = np.array(conn_at)
        tau_at = np.array(tau_at)
        auc_c = auroc_manual(labels, conn_at)
        auc_t = auroc_manual(labels, tau_at)
        print(f"  AUROC conn_R2={auc_c:.3f}, tau_R2={auc_t:.3f}")
        print(f"  Conn gap: {conn_at[labels==1].mean()-conn_at[labels==0].mean():.4f}")
        print(f"  Tau gap:  {tau_at[labels==1].mean()-tau_at[labels==0].mean():.4f}")

    # PASS CONDITION check
    print("\n" + "=" * 80)
    print("PASS CONDITION CHECK:")
    print(f"  1. tau_R2 AUROC >= 0.85: {auroc_tau:.3f} {'PASS' if auroc_tau >= 0.85 else 'FAIL'}")
    print(f"  2. tau_R2 AUROC > conn_R2 AUROC: {auroc_tau:.3f} > {auroc_conn:.3f} "
          f"{'PASS' if auroc_tau > auroc_conn else 'FAIL'}")
    overall = auroc_tau >= 0.85 and auroc_tau > auroc_conn
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
