"""
Emit the 8 TeX rows of tab:cv_per_condition.

The table splits each row in two halves (matching the caption):

  PREDICTION columns (one-step r, rollout r)
      The *DAVIS-trained base model* tested on each held-out
      YouTube-VOS fold — Phase 2 of cv_runner. Read per-fold from:
          <base_log_dir>/results_test_on_<config>_cv<i>.log
          <base_log_dir>/results_rollout_on_<config>_cv<i>.log
      Mean/SD taken across the 5 folds.

  PARAMETER-RECOVERY columns (W, tau, V_rest, cluster)
      The *YouTube-VOS-retrained fold models* — Phase 3 of cv_runner.
      Read as the 5-fold mean/SD from the "Metric" table at the
      bottom of:
          <base_log_dir>/results/cv_summary.txt

Output:
    <DATA_ROOT>/log/cv_per_condition_rows.tex
    (also printed to stdout)

Usage:
    python scripts/emit_conditions_table_rows.py \\
        --output_root /groups/saalfeld/home/allierc/GraphData
"""

import argparse
import os
import re
import sys

import numpy as np


# Resolve repo root from this script's location (robust local + cluster).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (os.path.join(REPO_ROOT, 'src'), REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


GOOD_THRESHOLD = 0.9


# Condition -> (TeX label, config name, noise_sigma, noise_gamma, edges)
CONDITIONS = [
    ('noise-free',            'flyvis_noise_free_winner',                 '0',    '0',   '434\\,112'),
    ('low intrinsic noise',   'flyvis_noise_005_winner',                  '0.05', '0',   '434\\,112'),
    ('high intrinsic noise',  'flyvis_noise_05_winner',                   '0.5',  '0',   '434\\,112'),
    ('low meas. noise',       'flyvis_noise_005_010_winner',              '0.05', '0.1', '434\\,112'),
    ('$+400\\%$ null edges',  'flyvis_noise_005_null_edges_pc_400',       '0.05', '0',   '2\\,170\\,560'),
    ('$-20\\%$ edges removed','flyvis_noise_005_removed_pc_20_winner',    '0.05', '0',   '347\\,000'),
    ('$1/5$ frames',          'flyvis_noise_005_stride_5_winner',         '0.05', '0',   '434\\,112'),
    ('$10\\%$ hidden',        'flyvis_noise_005_hidden_010_ngp_winner',   '0.05', '0',   '434\\,112'),
]


def fmt(mean, sd):
    if np.isnan(mean):
        return '$\\cdot$'
    body = f"${mean:.2f}{{\\pm}}{sd:.2f}$"
    return f"\\good{{{body}}}" if mean > GOOD_THRESHOLD else body


def parse_last_cv_block(summary_path):
    """Return dict metric_name -> (mean, sd) from the LAST block of cv_summary.txt."""
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path) as f:
        txt = f.read()
    # Each block ends with a stats table; grab the last one's lines.
    blocks = txt.split('\nCV log:')
    block = 'CV log:' + blocks[-1]
    stats = {}
    # Pattern: "<name>  mean   sd   cv%  min  max" after the Metric header.
    in_stats = False
    for ln in block.splitlines():
        if ln.strip().startswith('Metric'):
            in_stats = True
            continue
        if not in_stats:
            continue
        m = re.match(r'\s*(\w+)\s+([-\d.]+|\S)\s+([-\d.]+|\S)\s+', ln)
        if m:
            name = m.group(1)
            mean_s, sd_s = m.group(2), m.group(3)
            try:
                stats[name] = (float(mean_s), float(sd_s))
            except ValueError:
                stats[name] = (float('nan'), float('nan'))
    return stats


def _parse_pearson_log(path):
    """Return Pearson r mean from a results_{test,rollout}_on_*.log file
    written by graph_tester.data_test_gnn. Format:
        Pearson r: 0.996 +/- 0.032
    Returns the mean only (we aggregate across folds ourselves)."""
    if not os.path.exists(path):
        return float('nan')
    with open(path) as f:
        for line in f:
            if line.strip().startswith('Pearson r'):
                try:
                    return float(line.split(':')[1].split('+/-')[0].strip())
                except (IndexError, ValueError):
                    return float('nan')
    return float('nan')


def mean_sd(values):
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan')
    return float(arr.mean()), float(arr.std(ddof=0))


def phase2_prediction_stats(base_log_dir, config, n_seeds=5):
    """Phase 2 DAVIS-zero-shot Pearson r, mean/SD across 5 folds.

    The per-fold log naming in cv_runner.py is
        <base_log_dir>/results_{test,rollout}_on_<config>_cv<i>.log
    (i.e. fold name with 'flyvis_' prefix stripped, then '_cv<NN>').

    Returns ((one_m,one_s), (roll_m,roll_s), source) where source is
    'phase2' if any per-fold log parsed, else 'missing'.
    """
    one_step, rollout = [], []
    any_found = False
    for i in range(n_seeds):
        fold_short = f'{config}_cv{i:02d}'.replace('flyvis_', '')
        one_p  = os.path.join(base_log_dir, f'results_test_on_{fold_short}.log')
        roll_p = os.path.join(base_log_dir, f'results_rollout_on_{fold_short}.log')
        if os.path.exists(one_p) or os.path.exists(roll_p):
            any_found = True
        one_step.append(_parse_pearson_log(one_p))
        rollout.append(_parse_pearson_log(roll_p))
    return mean_sd(one_step), mean_sd(rollout), ('phase2' if any_found else 'missing')


def emit_row(label, config, noise_sig, noise_gam, edges_str, stats, pred):
    """stats: param-recovery stats dict (mean,sd) from cv_summary.txt
    pred : ((one_m, one_s), (roll_m, roll_s)) from phase-2 per-fold logs."""
    def get(key):
        return stats.get(key, (float('nan'), float('nan')))
    (one_m, one_s), (roll_m, roll_s) = pred
    W_m, W_s   = get('W_corrected_R2')
    tau_m, tau_s = get('tau_R2')
    V_m, V_s   = get('V_rest_R2')
    cl_m, cl_s = get('clustering_accuracy')
    return (
        f'{label:<22} & ${noise_sig}$ & ${noise_gam}$ & ${edges_str}$\n'
        f'  & {fmt(one_m, one_s)} & {fmt(roll_m, roll_s)}\n'
        f'  & {fmt(W_m, W_s)} & {fmt(tau_m, tau_s)} & {fmt(V_m, V_s)} & {fmt(cl_m, cl_s)} \\\\'
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_root',
                   default='/groups/saalfeld/home/allierc/GraphData')
    p.add_argument('--pre_folder', default='fly')
    p.add_argument('--n_seeds', type=int, default=5)
    args = p.parse_args()

    rows = []
    for label, cfg, nsig, ngam, edges in CONDITIONS:
        base_log_dir = os.path.join(args.output_root, 'log', args.pre_folder, cfg)
        summary = os.path.join(base_log_dir, 'results', 'cv_summary.txt')
        stats = parse_last_cv_block(summary)
        if not stats:
            print(f'WARN: no cv_summary.txt for {cfg} ({summary})',
                  file=sys.stderr)
        # Phase-2 prediction metrics from per-fold results_*_on_*.log files.
        # If no logs exist, leave prediction cells blank (rendered as "$\cdot$")
        # rather than silently falling back to Phase-3 YT-retrain metrics.
        one, roll, src = phase2_prediction_stats(base_log_dir, cfg,
                                                 n_seeds=args.n_seeds)
        if src == 'missing':
            print(f'WARN: no Phase-2 logs for {cfg} — leaving prediction '
                  f'cells blank (re-run cv with Phase 2 enabled to fill)',
                  file=sys.stderr)
        pred = (one, roll)
        rows.append(emit_row(label, cfg, nsig, ngam, edges, stats, pred))

    out_dir = os.path.join(args.output_root, 'log')
    os.makedirs(out_dir, exist_ok=True)
    out_tex = os.path.join(out_dir, 'cv_per_condition_rows.tex')
    with open(out_tex, 'w') as f:
        f.write('% --- tab:cv_per_condition — generated rows ---\n')
        for r in rows:
            f.write(r + '\n')
        f.write('% --------------------------------------------\n')

    print('% --- tab:cv_per_condition — generated rows ---')
    for r in rows:
        print(r)
    print('% --------------------------------------------')
    print(f'\nwrote {out_tex}')


if __name__ == '__main__':
    main()
