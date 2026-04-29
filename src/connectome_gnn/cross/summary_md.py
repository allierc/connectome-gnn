"""Per-condition CV summary table (markdown) for the blank50 pipelines.

`emit_summary_md(suffix, output_root, n_folds=5)` scans
<output_root>/log/fly/<base>_<suffix>_cv{00..n_folds-1}/ for every
condition declared in CONDITION_BASES, parses results/metrics.txt,
results_test.log and results_rollout.log, and writes one markdown file:

    <output_root>/log/cv_<suffix>_summary.md

Each condition gets its own table — one row per CV fold plus a final
**mean ± SD** row. Columns cover both prediction (test/rollout Pearson r,
RMSE) and parameter recovery (W R² + slope, tau R², V_rest R²,
clustering accuracy). The full per-fold log directory path is shown
above each table for traceability.

Called from run_all_conditions (and from the top-level run scripts) so
the markdown stays in sync with the TeX rows after every wave.
"""

import os
import re

import numpy as np

from connectome_gnn.cross.pipeline import CONDITION_BASES


def _parse_metrics_txt(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for ln in f:
            m = re.match(r'(\w+):\s*([-\d.eE+]+)', ln.strip())
            if m:
                try:
                    out[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    return out


def _parse_pearson_log(path):
    out = {'pearson': float('nan'), 'pearson_sd': float('nan'),
           'rmse': float('nan')}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if s.startswith('Pearson r:'):
                try:
                    rest = s.split(':', 1)[1].strip()
                    mean_s, sd_s = rest.split('+/-')
                    out['pearson']    = float(mean_s.strip())
                    out['pearson_sd'] = float(sd_s.strip())
                except Exception:
                    pass
            elif s.startswith('RMSE:'):
                try:
                    rest = s.split(':', 1)[1].strip()
                    mean_s = rest.split('+/-')[0]
                    out['rmse'] = float(mean_s.strip())
                except Exception:
                    pass
    return out


COLUMNS = [
    ('test_r',           'Test r',     '{:.3f}'),
    ('test_r_sd',        'Test r SD',  '{:.3f}'),
    ('test_rmse',        'Test RMSE',  '{:.3f}'),
    ('roll_r',           'Roll r',     '{:.3f}'),
    ('roll_r_sd',        'Roll r SD',  '{:.3f}'),
    ('roll_rmse',        'Roll RMSE',  '{:.4f}'),
    ('W_R2',             'W R²',       '{:.3f}'),
    ('W_slope',          'W slope',    '{:.3f}'),
    ('W_rel_err_med_pct',  'W rel.err median %', '{:.1f}'),
    ('W_rel_err_iqr_pct',  'W rel.err IQR %',    '{:.1f}'),
    ('tau_R2',           'τ R²',       '{:.3f}'),
    ('tau_R2_clean',     'τ R² (no outl.)', '{:.3f}'),
    ('tau_n_outliers',   'τ N outl.',  '{:.0f}'),
    ('tau_rel_err_med_pct', 'τ rel.err median %', '{:.1f}'),
    ('tau_rel_err_iqr_pct', 'τ rel.err IQR %',    '{:.1f}'),
    ('V_rest_R2',        'V_rest R²',  '{:.3f}'),
    ('V_rest_R2_clean',  'V_rest R² (no outl.)', '{:.3f}'),
    ('V_rest_n_outliers','V_rest N outl.', '{:.0f}'),
    ('V_rest_rel_err_med_pct', 'V_rest rel.err median %', '{:.1f}'),
    ('V_rest_rel_err_iqr_pct', 'V_rest rel.err IQR %',    '{:.1f}'),
    ('clustering',       'Clust acc',  '{:.3f}'),
]


def _collect_fold(fold_dir):
    m = _parse_metrics_txt(os.path.join(fold_dir, 'results', 'metrics.txt'))
    test_p = _parse_pearson_log(os.path.join(fold_dir, 'results_test.log'))
    roll_p = _parse_pearson_log(os.path.join(fold_dir, 'results_rollout.log'))
    return {
        'test_r':     test_p['pearson'],
        'test_r_sd':  test_p['pearson_sd'],
        'test_rmse':  test_p['rmse'],
        'roll_r':     roll_p['pearson'],
        'roll_r_sd':  roll_p['pearson_sd'],
        'roll_rmse':  roll_p['rmse'],
        'W_R2':              m.get('W_corrected_R2',         float('nan')),
        'W_slope':            m.get('W_corrected_slope',     float('nan')),
        # Stored as raw fractions; multiply by 100 for the % display columns.
        'W_rel_err_med_pct':  100.0 * m['W_rel_err_median'] if 'W_rel_err_median' in m else float('nan'),
        'W_rel_err_iqr_pct':  100.0 * m['W_rel_err_iqr']    if 'W_rel_err_iqr'    in m else float('nan'),
        'tau_R2':             m.get('tau_R2',                float('nan')),
        'tau_R2_clean':       m.get('tau_no_outliers_R2',    float('nan')),
        'tau_n_outliers':     m.get('tau_n_outliers',        float('nan')),
        'tau_rel_err_med_pct': 100.0 * m['tau_rel_err_median'] if 'tau_rel_err_median' in m else float('nan'),
        'tau_rel_err_iqr_pct': 100.0 * m['tau_rel_err_iqr']    if 'tau_rel_err_iqr'    in m else float('nan'),
        'V_rest_R2':          m.get('V_rest_R2',             float('nan')),
        'V_rest_R2_clean':    m.get('V_rest_no_outliers_R2', float('nan')),
        'V_rest_n_outliers':  m.get('V_rest_n_outliers',     float('nan')),
        'V_rest_rel_err_med_pct': 100.0 * m['V_rest_rel_err_median'] if 'V_rest_rel_err_median' in m else float('nan'),
        'V_rest_rel_err_iqr_pct': 100.0 * m['V_rest_rel_err_iqr']    if 'V_rest_rel_err_iqr'    in m else float('nan'),
        'clustering':         m.get('clustering_accuracy',   float('nan')),
    }


def _fmt_cell(val, fmt):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '–'
    return fmt.format(val)


def _fmt_meansd(vals, fmt):
    arr = np.array([v for v in vals if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return '–'
    return f'{fmt.format(arr.mean())} ± {fmt.format(arr.std(ddof=0))}'


def _emit_condition_table(base, suffix, output_root, n_folds, log_subdir):
    fold_dirs, fold_data = [], []
    for i in range(n_folds):
        fd = os.path.join(output_root, 'log', log_subdir,
                          f'{base}_{suffix}_cv{i:02d}')
        if os.path.isdir(fd):
            fold_dirs.append(fd)
            fold_data.append(_collect_fold(fd))
    if not fold_dirs:
        return None

    parent_dir = os.path.join(output_root, 'log', log_subdir)
    lines = [f'## {base}', '',
             f'**Log dir:** `{parent_dir}/{base}_{suffix}_cv*`', '']

    headers = ['Fold'] + [label for _, label, _ in COLUMNS]
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')

    for fd, vals in zip(fold_dirs, fold_data):
        cv_tag = os.path.basename(fd).rsplit('_', 1)[-1]
        cells = [cv_tag] + [_fmt_cell(vals[k], fmt) for k, _, fmt in COLUMNS]
        lines.append('| ' + ' | '.join(cells) + ' |')

    summary_cells = ['**mean ± SD**']
    for k, _, fmt in COLUMNS:
        summary_cells.append('**' + _fmt_meansd([v[k] for v in fold_data], fmt) + '**')
    lines.append('| ' + ' | '.join(summary_cells) + ' |')

    lines.extend(['', '<details><summary>Per-fold log directories</summary>', ''])
    for fd in fold_dirs:
        lines.append(f'- `{fd}`')
    lines.extend(['', '</details>'])

    return '\n'.join(lines)


def emit_summary_md(suffix, output_root, n_folds=5, log_subdir='fly'):
    """Write <output_root>/log/cv_<suffix>_summary.md.

    Skips conditions with no fold dirs on disk. Safe to call repeatedly —
    the file is overwritten in place.
    """
    sections = []
    for base in CONDITION_BASES:
        sec = _emit_condition_table(base, suffix, output_root, n_folds, log_subdir)
        if sec is not None:
            sections.append(sec)

    out_dir = os.path.join(output_root, 'log')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'cv_{suffix}_summary.md')

    with open(out_path, 'w') as f:
        f.write(f'# CV summary — `{suffix}`\n\n')
        f.write(f'**Output root:** `{output_root}`\n\n')
        f.write(f'**Conditions found:** {len(sections)} '
                f'(of {len(CONDITION_BASES)} declared)\n\n')
        if not sections:
            f.write('_No matching log directories found._\n')
        else:
            f.write('\n\n'.join(sections))
            f.write('\n')

    print(f'  [md  ] {out_path}  ({len(sections)} condition(s))')
    return out_path
