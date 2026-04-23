"""Analyze WHEN the bifurcation becomes detectable and what trajectory feature
best predicts convergence.

Key questions:
1. At which step does AUROC first exceed 0.85?
2. Does the slope (derivative) of conn_R2 predict better than the level?
3. Does the conn_R2 PEAK position (step at max) predict failure?
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

CONVERGE_THRESHOLD = 0.90


def load_trajectory(iter_num):
    path = os.path.join(LOG_DIR, f"iter_{iter_num:03d}.log")
    if not os.path.exists(path):
        return None
    steps, conn_r2, tau_r2 = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['iteration']))
            conn_r2.append(float(row['connectivity_r2']))
            tau_r2.append(float(row.get('tau_r2', '0')))
    return np.array(steps), np.array(conn_r2), np.array(tau_r2)


def auroc_manual(labels, scores):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
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
    print("TRAJECTORY SHAPE ANALYSIS — when does bifurcation become detectable?")
    print("=" * 80)

    # Load all trajectories
    all_data = {}
    for iter_num, final_r2 in sorted(BASELINE_DAL35.items()):
        data = load_trajectory(iter_num)
        if data is None:
            continue
        all_data[iter_num] = {
            'steps': data[0], 'conn': data[1], 'tau': data[2],
            'final': final_r2, 'converged': 1 if final_r2 >= CONVERGE_THRESHOLD else 0
        }

    labels = np.array([d['converged'] for d in all_data.values()])
    n = len(labels)
    print(f"\n{n} seeds loaded: {labels.sum()} converging, {n - labels.sum()} failing")

    # Get common steps
    ref = list(all_data.values())[0]['steps']
    min_len = min(len(d['steps']) for d in all_data.values())
    common_steps = ref[:min_len]

    # 1. AUROC at each checkpoint
    print(f"\n--- AUROC by checkpoint ---")
    print(f"{'Step':>8} {'conn_AUROC':>12} {'tau_AUROC':>12} {'slope_AUROC':>12}")
    for i, step in enumerate(common_steps):
        if step == 1:
            continue
        conn_at = np.array([d['conn'][i] for d in all_data.values()])
        tau_at = np.array([d['tau'][i] for d in all_data.values()])
        auc_c = auroc_manual(labels, conn_at)
        auc_t = auroc_manual(labels, tau_at)

        # Slope: delta from previous checkpoint
        if i > 1:
            prev_conn = np.array([d['conn'][i-1] for d in all_data.values()])
            slope = conn_at - prev_conn
            auc_s = auroc_manual(labels, slope)
        else:
            auc_s = float('nan')

        marker = " <-- FIRST >= 0.85" if auc_c >= 0.85 or auc_t >= 0.85 else ""
        print(f"{step:>8} {auc_c:>12.3f} {auc_t:>12.3f} {auc_s:>12.3f}{marker}")

    # 2. Trajectory features
    print(f"\n--- Trajectory features per seed ---")
    print(f"{'Iter':>4} {'Final':>7} {'Conv':>5} {'Peak':>8} {'PeakStep':>10} "
          f"{'Slope2-4k':>10} {'Slope44-67k':>12} {'Late drop':>10}")

    features = {k: {} for k in all_data}
    for it, d in sorted(all_data.items()):
        conn = d['conn'][:min_len]
        peak_val = conn.max()
        peak_idx = conn.argmax()
        peak_step = common_steps[peak_idx]

        # Early slope: step 2241 → 4481
        if min_len >= 3:
            early_slope = conn[2] - conn[1]  # idx 1=2241, idx 2=4481
        else:
            early_slope = 0

        # Late slope: step 44801 → 67201 (roughly idx 8 → 10)
        idx_44k = np.argmin(np.abs(common_steps - 44801))
        idx_67k = np.argmin(np.abs(common_steps - 67201))
        if idx_67k > idx_44k:
            late_slope = conn[idx_67k] - conn[idx_44k]
        else:
            late_slope = 0

        # Late drop from peak
        late_drop = peak_val - conn[-1]

        features[it] = {
            'peak': peak_val, 'peak_step': peak_step,
            'early_slope': early_slope, 'late_slope': late_slope,
            'late_drop': late_drop
        }

        status = "CONV" if d['converged'] else "FAIL"
        print(f"{it:>4} {d['final']:>7.4f} [{status}] {peak_val:>8.4f} {peak_step:>10} "
              f"{early_slope:>10.4f} {late_slope:>12.4f} {late_drop:>10.4f}")

    # 3. AUROC for trajectory features
    print(f"\n--- AUROC for trajectory features ---")
    peaks = np.array([features[k]['peak'] for k in all_data])
    early_slopes = np.array([features[k]['early_slope'] for k in all_data])
    late_slopes = np.array([features[k]['late_slope'] for k in all_data])
    late_drops = np.array([features[k]['late_drop'] for k in all_data])

    # For late_drop, LOWER is better (less regression), so flip sign
    print(f"  peak conn_R2:     {auroc_manual(labels, peaks):.3f}")
    print(f"  early slope:      {auroc_manual(labels, early_slopes):.3f}")
    print(f"  late slope:       {auroc_manual(labels, late_slopes):.3f}")
    print(f"  late drop (neg):  {auroc_manual(labels, -late_drops):.3f}")

    # 4. Combined: conn_R2 at step 44801 (near midpoint of DAL=95 but endpoint of DAL=35)
    print(f"\n--- conn_R2 at step 44801 (approximate endpoint for DAL=35) ---")
    idx_44k = np.argmin(np.abs(common_steps - 44801))
    conn_44k = np.array([d['conn'][idx_44k] for d in all_data.values()])
    print(f"  AUROC: {auroc_manual(labels, conn_44k):.3f}")
    conv_44k = conn_44k[labels == 1]
    fail_44k = conn_44k[labels == 0]
    print(f"  conv: {conv_44k.mean():.4f}+/-{conv_44k.std():.4f}")
    print(f"  fail: {fail_44k.mean():.4f}+/-{fail_44k.std():.4f}")
    print(f"  gap: {conv_44k.mean() - fail_44k.mean():.4f}")

    # 5. Check: is the best single-step AUROC high enough for practical early stopping?
    best_auroc = 0
    best_step = 0
    best_metric = ""
    for i, step in enumerate(common_steps[1:], 1):
        conn_at = np.array([d['conn'][i] for d in all_data.values()])
        tau_at = np.array([d['tau'][i] for d in all_data.values()])
        ac = auroc_manual(labels, conn_at)
        at = auroc_manual(labels, tau_at)
        if ac > best_auroc:
            best_auroc, best_step, best_metric = ac, step, "conn_R2"
        if at > best_auroc:
            best_auroc, best_step, best_metric = at, step, "tau_R2"

    print(f"\n--- Best single-step predictor ---")
    print(f"  {best_metric} @ step {best_step}: AUROC = {best_auroc:.3f}")
    print(f"  {'Practically useful (>=0.85)' if best_auroc >= 0.85 else 'NOT practically useful (<0.85)'}")


if __name__ == "__main__":
    main()
