"""
Emit the two TeX rows of the GNN+INR table (tab:cv_inr) from the fold
directories produced by scripts/run_inr_cv.py.

For each condition C in {davis, yt} and fold i in 0..N-1, read:
    log/<pre>/<base>_C_cv<i>/results_test.log         -> one-step r
    log/<pre>/<base>_C_cv<i>/results_rollout.log      -> rollout r + stimuli r
    log/<pre>/<base>_C_cv<i>/results/metrics.txt      -> W/tau/V_rest/cluster

Report mean±SD over N folds for each metric.

  Row 1 "DAVIS"       from *_davis_cv{0..N-1}
  Row 2 "YouTube-VOS" from *_yt_cv{0..N-1}

Output:
  <base_log_dir>/results/cv_inr_table_rows.tex
  (also printed to stdout)

Usage:
    python scripts/emit_inr_table_rows.py \\
        --config flyvis_noise_005_INR \\
        --output_root /groups/saalfeld/home/allierc/GraphData \\
        [--n_seeds 5]
"""

import argparse
import os
import re
import sys

import numpy as np


# Resolve repo root from this script's location (works local + cluster).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (os.path.join(REPO_ROOT, 'src'), REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


GOOD_THRESHOLD = 0.9
LOW_THRESHOLD  = 0.3
N_NEURONS      = 13741   # flyvis_A — total neuron count for outlier %


def fmt(mean, sd):
    """Format as $x.xx{\\pm}y.yy$ with \\good{} (>0.9) / \\bad{} (<0.3)."""
    if np.isnan(mean):
        return '$\\cdot$'
    body = f"${mean:.2f}{{\\pm}}{sd:.2f}$"
    if mean > GOOD_THRESHOLD:
        return f"\\good{{{body}}}"
    if mean < LOW_THRESHOLD:
        return f"\\bad{{{body}}}"
    return body


def fmt_R2_out(mean, sd, count_mean, n_neurons=N_NEURONS):
    """${R2}{\\pm}{SD}\\,(out%)$ for tau / V_rest no-outliers R²,
    appending mean(n_outliers / n_neurons) across folds (1 decimal).
    \\good{} (>0.9) / \\bad{} (<0.3) on the R² mean wraps the whole cell."""
    if np.isnan(mean) or np.isnan(sd) or np.isnan(count_mean) or not n_neurons:
        return '$\\cdot$'
    body = (f"${mean:.2f}{{\\pm}}{sd:.2f}"
            f"\\,({100.0*count_mean/n_neurons:.1f})$")
    if mean > GOOD_THRESHOLD:
        return f"\\good{{{body}}}"
    if mean < LOW_THRESHOLD:
        return f"\\bad{{{body}}}"
    return body


def parse_pearson(log_file):
    """Read the first 'Pearson r: X +/- Y' line and return X."""
    if not os.path.exists(log_file):
        return float('nan')
    m = re.search(r'Pearson r:\s*([-\d.]+)', open(log_file).read())
    return float(m.group(1)) if m else float('nan')


def parse_stimuli_r(log_file):
    if not os.path.exists(log_file):
        return float('nan')
    m = re.search(r'stimuli_r:\s*([-\d.]+)', open(log_file).read())
    return float(m.group(1)) if m else float('nan')


def parse_metrics_txt(path):
    """Return dict from metrics.txt (key: float) or empty dict if missing."""
    if not os.path.exists(path):
        return {}
    out = {}
    for ln in open(path):
        m = re.match(r'(\w+):\s*([-\d.]+)', ln.strip())
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def mean_sd(values):
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan')
    return float(arr.mean()), float(arr.std(ddof=0))


def _load_pearson_npy(log_path_):
    """Return per-neuron pearson array from the sibling *_pearson.npy, or None."""
    stem = os.path.splitext(log_path_)[0]
    npy  = f'{stem}_pearson.npy'
    if not os.path.isfile(npy):
        return None
    try:
        return np.load(npy)
    except OSError:
        return None


def fisher_pool_fold_arrays(arrays):
    """Pool a list of per-fold per-neuron r arrays in Fisher-z space.

    Returns (r_mean, r_sd_sym). If any fold's array is missing, returns
    (nan, nan) so the caller can fall back to scalar log parsing.
    """
    if not arrays or any(a is None for a in arrays):
        return (float('nan'), float('nan'))
    from connectome_gnn.utils import fisher_pool
    fz = fisher_pool(np.concatenate([a.ravel() for a in arrays]))
    if fz['n'] == 0:
        return (float('nan'), float('nan'))
    return (fz['r_mean'], fz['r_sd_sym'])


def collect_condition(base_name, condition, pre_folder, output_root, n_seeds):
    """Return dict of metric_name -> list (len n_seeds) of per-fold values,
    plus a parallel dict of per-fold per-neuron arrays for the r-metrics."""
    cols = {k: [] for k in (
        'one_step_r', 'rollout_r', 'stimuli_r',
        'W_corrected_R2',
        'tau_no_outliers_R2',  'tau_n_outliers',
        'V_rest_no_outliers_R2', 'V_rest_n_outliers',
        'clustering_accuracy',
    )}
    r_arrays = {'one_step_r': [], 'rollout_r': []}
    for i in range(n_seeds):
        fold = f'{base_name}_{condition}_cv{i:02d}'
        fold_log = os.path.join(output_root, 'log', pre_folder, fold)
        test_log    = os.path.join(fold_log, 'results_test.log')
        rollout_log = os.path.join(fold_log, 'results_rollout.log')
        cols['one_step_r'].append(parse_pearson(test_log))
        cols['rollout_r'].append(parse_pearson(rollout_log))
        cols['stimuli_r'].append(parse_stimuli_r(rollout_log))
        r_arrays['one_step_r'].append(_load_pearson_npy(test_log))
        r_arrays['rollout_r'].append(_load_pearson_npy(rollout_log))
        m = parse_metrics_txt(os.path.join(fold_log, 'results', 'metrics.txt'))
        for k in ('W_corrected_R2',
                  'tau_no_outliers_R2', 'tau_n_outliers',
                  'V_rest_no_outliers_R2', 'V_rest_n_outliers',
                  'clustering_accuracy'):
            cols[k].append(m.get(k, float('nan')))
    cols['_r_arrays'] = r_arrays
    return cols


def emit_row(label, cols, noise_tex, edges_tex):
    def ms(k):
        # For r-metrics with per-neuron arrays on disk, Fisher-z-pool across
        # (neurons × folds) so the SD includes neuron-level variance. Fall
        # back to scalar mean-of-fold-means when any fold's .npy is missing.
        r_arrays = cols.get('_r_arrays', {})
        if k in r_arrays:
            m, s = fisher_pool_fold_arrays(r_arrays[k])
            if not np.isnan(m):
                return m, s
        return mean_sd(cols[k])
    parts = {k: ms(k) for k in cols if not k.startswith('_')}
    one  = fmt(*parts['one_step_r'])
    roll = fmt(*parts['rollout_r'])
    stim = fmt(*parts['stimuli_r'])
    W    = fmt(*parts['W_corrected_R2'])
    # tau / V_rest cells: R² on the no-outliers subset + mean outlier %
    # (count averaged across folds, then divided by N_NEURONS, 1 decimal).
    tau_R2_m, tau_R2_s = parts['tau_no_outliers_R2']
    tau_out_m, _       = parts['tau_n_outliers']
    V_R2_m,   V_R2_s   = parts['V_rest_no_outliers_R2']
    V_out_m,  _        = parts['V_rest_n_outliers']
    tau = fmt_R2_out(tau_R2_m, tau_R2_s, tau_out_m)
    V   = fmt_R2_out(V_R2_m,   V_R2_s,   V_out_m)
    cl  = fmt(*parts['clustering_accuracy'])
    return (
        f'{label:<12} & {noise_tex} & {edges_tex}\n'
        f'  & {one} & {roll} & {stim}\n'
        f'  & {W} & {tau} & {V} & {cl} \\\\'
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='flyvis_noise_005_INR',
                   help='CV config basename (no .yaml, no pre-folder)')
    p.add_argument('--output_root',
                   default='/groups/saalfeld/home/allierc/GraphData')
    p.add_argument('--pre_folder', default='fly',
                   help='Pre-folder under log/ (default: fly)')
    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--noise', default='0.05')
    p.add_argument('--edges', default='434\\,112')
    p.add_argument('--condition_tags', nargs='+',
                   default=['davis', 'yt'],
                   help='Condition tags used in the fold dirname '
                        '(<config>_<tag>_cv<i>). Default: davis yt. '
                        'For blank50 runs pass: davis_blank50 yt_blank50.')
    p.add_argument('--row_labels', nargs='+',
                   default=['DAVIS', 'YouTube-VOS'],
                   help='Display labels for each row, paired with '
                        '--condition_tags (same length).')
    args = p.parse_args()

    assert len(args.condition_tags) == len(args.row_labels), \
        '--condition_tags and --row_labels must have the same length'

    noise_tex = f'${args.noise}$'
    edges_tex = f'${args.edges}$'

    rows = []
    for tag, label in zip(args.condition_tags, args.row_labels):
        cols = collect_condition(args.config, tag, args.pre_folder,
                                 args.output_root, args.n_seeds)
        rows.append(emit_row(label, cols, noise_tex, edges_tex))
    # Backwards-compat aliases — the existing two-row write block below
    # references row_davis / row_yt explicitly. Map by position.
    row_davis = rows[0] if len(rows) >= 1 else None
    row_yt    = rows[1] if len(rows) >= 2 else None

    # Paper-bound .tex lives alongside the other cv_*.tex tables in figures/.
    figures_dir = os.path.join(REPO_ROOT, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    out_tex = os.path.join(figures_dir, 'cv_table_inr.tex')
    with open(out_tex, 'w') as f:
        f.write('% tab:cv_inr — joint GNN+INR rows.\n')
        f.write(f'% Conditions: {", ".join(args.row_labels)}.\n')
        f.write('% Cols: condition & sigma & edges | one-step r | rollout r | '
                'stimuli r | W R^2 | tau R^2 (out%%) | Vrest R^2 (out%%) | '
                'cluster acc.  (R^2 for tau/Vrest computed on no-outlier '
                'subset; out%% = mean(n_outliers / 13741) across folds, '
                '1 decimal.)\n')
        for r in rows:
            f.write(r + '\n')

    print('% --- tab:cv_inr — generated rows ---')
    for r in rows:
        print(r)
    print('% ---------------------------------')
    print(f'\nwrote {out_tex}')


if __name__ == '__main__':
    main()
