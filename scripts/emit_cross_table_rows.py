"""
Emit 8 TeX rows of a 6-column table (same schema as tab:cv_per_condition)
driven by YT-trained models cross-tested on DAVIS.

For each condition:
  prediction columns (one-step r, rollout r):
      from <DATA_ROOT>/log/fly/<base>_<suffix>/results_{test,rollout}_on_<base>.log
      = YT-trained model rolled out on DAVIS held-out test data.
  parameter-recovery columns (W, tau, V_rest, cluster):
      from <DATA_ROOT>/log/fly/<base>_<suffix>/results/metrics.txt
      = YT-trained model's own learned parameters.

Missing cells render as "$\\cdot$".

Usage:
    python scripts/emit_cross_table_rows.py \\
        --suffix yt_per_cond \\
        --output_tex cv_per_condition_rows.tex \\
        --output_root /groups/saalfeld/home/allierc/GraphData
"""

import argparse
import os
import re
import sys

import numpy as np


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)


GOOD_THRESHOLD = 0.9


# Condition base name -> (label, noise sigma, noise gamma, edges text)
CONDITIONS = [
    ('flyvis_noise_free',                'noise-free',              '0',    '0',   '434\\,112'),
    ('flyvis_noise_005',                 'low intrinsic noise',     '0.05', '0',   '434\\,112'),
    ('flyvis_noise_05',                  'high intrinsic noise',    '0.5',  '0',   '434\\,112'),
    ('flyvis_noise_005_010',             'low meas. noise',         '0.05', '0.1', '434\\,112'),
    ('flyvis_noise_005_null_edges_pc_400', '$+400\\%$ null edges',  '0.05', '0',   '2\\,170\\,560'),
    ('flyvis_noise_005_removed_pc_20',   '$-20\\%$ edges removed',  '0.05', '0',   '347\\,000'),
    ('flyvis_noise_005_stride_5',        '$1/5$ frames',            '0.05', '0',   '434\\,112'),
    ('flyvis_noise_005_hidden_010_ngp',  '$10\\%$ hidden',          '0.05', '0',   '434\\,112'),
]


def fmt(mean, sd):
    if np.isnan(mean):
        return '$\\cdot$'
    body = f"${mean:.2f}{{\\pm}}{sd:.2f}$"
    return f"\\good{{{body}}}" if mean > GOOD_THRESHOLD else body


def _parse_pearson(path):
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


def _parse_metrics_txt(path):
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


def _mean_sd(vals):
    arr = np.array([v for v in vals if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan')
    return float(arr.mean()), float(arr.std(ddof=0))


def _first_existing(paths):
    """Return the first path that exists, else paths[0] for nan-parse."""
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


def emit_row(base, label, nsig, ngam, edges, output_root, pre_folder,
             suffix, n_folds):
    """Aggregate mean ± SD for base <suffix>_cv00..cv<N-1>.

    Paired N-fold CV: YT fold i is rolled out against DAVIS fold i, so
    prediction columns (one-step r, rollout r) aggregate N values from:
        <fold_i_dir>/results_{test,rollout}_on_<short>_cv{i:02d}.log
    Falls back to the legacy single-DAVIS log name when the per-fold log
    is absent:
        <fold_i_dir>/results_{test,rollout}_on_<short>.log

    Parameter-recovery columns (W, τ, V_rest, cluster) aggregate one
    value per YT fold (data_plot runs once per YT model).
    """
    short = base.replace('flyvis_', '').replace('fly/', '')
    one_vals, roll_vals = [], []
    W_vals, tau_vals, V_vals, cl_vals = [], [], [], []
    found = 0
    for i in range(n_folds):
        fold_dir = os.path.join(output_root, 'log', pre_folder,
                                f'{base}_{suffix}_cv{i:02d}')
        if not os.path.isdir(fold_dir):
            continue
        found += 1
        # Paired: YT fold i × DAVIS fold i.
        test_paths = [
            os.path.join(fold_dir, f'results_test_on_{short}_cv{i:02d}.log'),
            os.path.join(fold_dir, f'results_test_on_{short}.log'),  # legacy
        ]
        roll_paths = [
            os.path.join(fold_dir, f'results_rollout_on_{short}_cv{i:02d}.log'),
            os.path.join(fold_dir, f'results_rollout_on_{short}.log'),  # legacy
        ]
        one_vals.append(_parse_pearson(_first_existing(test_paths)))
        roll_vals.append(_parse_pearson(_first_existing(roll_paths)))
        m = _parse_metrics_txt(os.path.join(fold_dir, 'results', 'metrics.txt'))
        W_vals.append(m.get('W_corrected_R2',  float('nan')))
        tau_vals.append(m.get('tau_R2',        float('nan')))
        V_vals.append(m.get('V_rest_R2',       float('nan')))
        cl_vals.append(m.get('clustering_accuracy', float('nan')))
    if found == 0:
        print(f'WARN: no fold dirs found for {base}_{suffix}_cv*',
              file=sys.stderr)
    one_m, one_s   = _mean_sd(one_vals)
    roll_m, roll_s = _mean_sd(roll_vals)
    W_m, W_s       = _mean_sd(W_vals)
    tau_m, tau_s   = _mean_sd(tau_vals)
    V_m, V_s       = _mean_sd(V_vals)
    cl_m, cl_s     = _mean_sd(cl_vals)
    return (
        f'{label:<22} & ${nsig}$ & ${ngam}$ & ${edges}$\n'
        f'  & {fmt(one_m, one_s)} & {fmt(roll_m, roll_s)}\n'
        f'  & {fmt(W_m, W_s)} & {fmt(tau_m, tau_s)} & {fmt(V_m, V_s)} & {fmt(cl_m, cl_s)} \\\\'
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suffix', required=True,
                   help='YT training yaml suffix used by run_cross_yt.py.')
    p.add_argument('--output_tex', default=None,
                   help='Basename of the output .tex file under <DATA_ROOT>/log/. '
                        'Default: cv_<suffix>_rows.tex')
    p.add_argument('--output_root',
                   default='/groups/saalfeld/home/allierc/GraphData')
    p.add_argument('--pre_folder', default='fly')
    p.add_argument('--n_folds', type=int, default=5)
    args = p.parse_args()

    out_name = args.output_tex or f'cv_{args.suffix}_rows.tex'

    rows = []
    for base, label, nsig, ngam, edges in CONDITIONS:
        rows.append(emit_row(base, label, nsig, ngam, edges,
                             args.output_root, args.pre_folder,
                             args.suffix, args.n_folds))

    out_dir = os.path.join(args.output_root, 'log')
    os.makedirs(out_dir, exist_ok=True)
    out_tex = os.path.join(out_dir, out_name)
    with open(out_tex, 'w') as f:
        f.write(f'% --- rows for YT-trained, DAVIS-cross-tested; suffix={args.suffix} ---\n')
        for r in rows:
            f.write(r + '\n')
        f.write('% ------------------------------------------------------------\n')

    print(f'% --- rows for YT-trained, DAVIS-cross-tested; suffix={args.suffix} ---')
    for r in rows:
        print(r)
    print('% ------------------------------------------------------------')
    print(f'\nwrote {out_tex}')


if __name__ == '__main__':
    main()
