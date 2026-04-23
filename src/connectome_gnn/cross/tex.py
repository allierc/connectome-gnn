"""
TeX emission for the hold-out-only cross-check table.

`emit_tex_file(suffix, output_root, n_folds)` aggregates n_folds mean±SD
for each of the 8 conditions and writes the 8-row TeX table to
<output_root>/log/cv_<suffix>_rows.tex. Reads per-fold results from
<log_dir>/results_test.log, <log_dir>/results_rollout.log, and
<log_dir>/results/metrics.txt.
"""

import os
import re

import numpy as np


GOOD_THRESHOLD = 0.9


# (base, label, sigma, gamma, edges) — table-row metadata.
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


def _fmt(mean, sd):
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


def emit_row(base, label, nsig, ngam, edges, output_root, pre_folder,
             suffix, n_folds):
    """N-fold CV: hold-out fold i rolled out on its own held-out 20%.
    Prediction columns aggregate N values; parameter columns also N.

    Silently emits NaN cells for conditions whose log dirs don't exist yet
    (expected mid-run when this function is called per completed condition).
    """
    one_vals, roll_vals = [], []
    W_vals, tau_vals, V_vals, cl_vals = [], [], [], []
    for i in range(n_folds):
        fold_dir = os.path.join(output_root, 'log', pre_folder,
                                f'{base}_{suffix}_cv{i:02d}')
        if not os.path.isdir(fold_dir):
            continue
        test_path = os.path.join(fold_dir, 'results_test.log')
        roll_path = os.path.join(fold_dir, 'results_rollout.log')
        one_vals.append(_parse_pearson(test_path))
        roll_vals.append(_parse_pearson(roll_path))
        m = _parse_metrics_txt(os.path.join(fold_dir, 'results', 'metrics.txt'))
        W_vals.append(m.get('W_corrected_R2',  float('nan')))
        tau_vals.append(m.get('tau_R2',        float('nan')))
        V_vals.append(m.get('V_rest_R2',       float('nan')))
        cl_vals.append(m.get('clustering_accuracy', float('nan')))
    one_m, one_s   = _mean_sd(one_vals)
    roll_m, roll_s = _mean_sd(roll_vals)
    W_m, W_s       = _mean_sd(W_vals)
    tau_m, tau_s   = _mean_sd(tau_vals)
    V_m, V_s       = _mean_sd(V_vals)
    cl_m, cl_s     = _mean_sd(cl_vals)
    return (
        f'{label:<22} & ${nsig}$ & ${ngam}$ & ${edges}$\n'
        f'  & {_fmt(one_m, one_s)} & {_fmt(roll_m, roll_s)}\n'
        f'  & {_fmt(W_m, W_s)} & {_fmt(tau_m, tau_s)} & '
        f'{_fmt(V_m, V_s)} & {_fmt(cl_m, cl_s)} \\\\'
    )


def emit_tex_file(suffix, output_root, n_folds=5, pre_folder='fly',
                   output_tex=None):
    """Emit the 8-row TeX table to <output_root>/log/cv_<suffix>_rows.tex.
    Safe to call repeatedly — overwrites in place."""
    out_name = output_tex or f'cv_{suffix}_rows.tex'
    rows = [emit_row(base, label, nsig, ngam, edges,
                     output_root, pre_folder, suffix, n_folds)
            for base, label, nsig, ngam, edges in CONDITIONS]
    out_dir = os.path.join(output_root, 'log')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, 'w') as f:
        f.write(f'% --- rows for hold-out-trained, hold-out-held-out-tested; suffix={suffix} ---\n')
        for r in rows:
            f.write(r + '\n')
        f.write('% ' + '-' * 60 + '\n')
    print(f'  [tex ] {out_path}')
