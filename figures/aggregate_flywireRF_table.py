"""Aggregate flywireRF (zero-edge augmentation) metrics into a LaTeX row block.

Single-run pipeline (no CV folds): one row per config. Reads
    log/fly/<base>/results_test.log         (one-step r)
    log/fly/<base>/results_rollout.log      (rollout r)
    log/fly/<base>/results/metrics.txt      (W_corrected_R2, tau_R2, V_rest_R2)

Layout: Known-ODE block first, then GNN block, separated by a single \\midrule.
The two extent-15 rows in each block carry \\textit{larger} (first row)
and \\textit{visual field} (second row) in the model column, forming a
two-line vertical "Larger visual field" sub-label.

Cell formatting (same scheme as figures/aggregate_blank50_tables.py):
    val > 0.9  -> \\good{$val$}                 (define \\good in preamble)
    val < 0.3  -> \\bad{$val$}                  (define \\bad  in preamble)
    NaN        -> $\\cdot$
Preamble:
    \\usepackage{xcolor}
    \\newcommand{\\good}[1]{\\textcolor{green!50!black}{#1}}
    \\newcommand{\\bad}[1]{\\textcolor{orange}{#1}}

Output: <output_root>/log/cv_table_flywireRF_zeroedge.tex

Console preview: orange when val < 0.3 or missing, green when > 0.9.

Example:
    python figures/aggregate_flywireRF_table.py
"""

import argparse
import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from connectome_gnn.cross.tex import _parse_pearson, _parse_metrics_txt
from connectome_gnn.utils import load_data_root_from_json, set_data_root


# Sentinel for the double-midrule between the Known-ODE and GNN blocks.
DOUBLE_RULE = ('__rule__',)

# (model_label, condition_label, edges, extent, base). 10 rows total
# (5 Known-ODE + 5 GNN). Edge counts read from each config YAML's `n_edges`.
# Empty model_label = continuation of previous model block.
# Extent-15 rows use \textit{larger} / \textit{visual field} in the model
# column to form a two-row "Larger visual field" vertical sub-label.
# zeroedge variants: `_sl_` = same-type spatially-local, `_cross_sl_` =
# cross-type spatially-local (different sampling strategies for the
# zero-weight edge augmentation).
TABLE_ROWS = [
    # ---- Known-ODE block ----
    ('Known ODE',                'het. RF (oracle)',                       '328\\,092',     '8',
        'flyvis_hybrid_flywireRF_known_ode_noise_005'),
    ('',                         'het.\\ RF + uncert.\\ edges (same-type)', '401\\,175',    '8',
        'flyvis_hybrid_flywireRF_zeroedge_sl_known_ode_noise_005'),
    ('',                         'het.\\ RF + uncert.\\ edges (cross-type)','1\\,959\\,994','8',
        'flyvis_hybrid_flywireRF_zeroedge_cross_sl_known_ode_noise_005'),
    ('\\textit{larger}',         'het. RF (oracle)',                       '1\\,256\\,695', '15',
        'flyvis_hybrid_flywireRF_e15_known_ode_noise_005'),
    ('\\textit{visual field}',   'het.\\ RF + uncert.\\ edges (cross-type)','5\\,411\\,743','15',
        'flyvis_hybrid_flywireRF_zeroedge_cross_sl_e15_known_ode_noise_005'),

    DOUBLE_RULE,

    # ---- GNN block ----
    ('GNN',                      'het. RF (oracle)',                       '328\\,092',     '8',
        'flyvis_hybrid_flywireRF_noise_005'),
    ('',                         'het.\\ RF + uncert.\\ edges (same-type)', '401\\,175',    '8',
        'flyvis_hybrid_flywireRF_zeroedge_sl_noise_005'),
    ('',                         'het.\\ RF + uncert.\\ edges (cross-type)','1\\,959\\,994','8',
        'flyvis_hybrid_flywireRF_zeroedge_cross_sl_noise_005'),
    ('\\textit{larger}',         'het. RF (oracle)',                       '1\\,256\\,695', '15',
        'flyvis_hybrid_flywireRF_e15_noise_005'),
    ('\\textit{visual field}',   'het.\\ RF + uncert.\\ edges (cross-type)','5\\,411\\,743','15',
        'flyvis_hybrid_flywireRF_zeroedge_cross_sl_e15_noise_005'),
]


_ANSI_ORANGE = '\033[38;5;208m'
_ANSI_GREEN  = '\033[92m'
_ANSI_RESET  = '\033[0m'
_LOW_THRESH  = 0.3
_GOOD_THRESH = 0.9


def _fmt(val):
    """LaTeX cell for a single (non-CV) value. Wraps \\good{...} when >0.9,
    \\bad{...} when <0.3, $\\cdot$ for NaN."""
    if isinstance(val, float) and math.isnan(val):
        return '$\\cdot$'
    body = f'${val:.2f}$'
    if val > _GOOD_THRESH:
        return f'\\good{{{body}}}'
    if val < _LOW_THRESH:
        return f'\\bad{{{body}}}'
    return body


def _ansi_cell(val, width=7):
    if isinstance(val, float) and math.isnan(val):
        body = '  nan  '
        return f'{_ANSI_ORANGE}{body:<{width}}{_ANSI_RESET}'
    body = f'{val:.2f}'
    if val < _LOW_THRESH:
        return f'{_ANSI_ORANGE}{body:<{width}}{_ANSI_RESET}'
    if val > _GOOD_THRESH:
        return f'{_ANSI_GREEN}{body:<{width}}{_ANSI_RESET}'
    return f'{body:<{width}}'


def _resolve_output_root(output_root):
    output_root = (output_root or os.environ.get('GNN_OUTPUT_ROOT')
                   or load_data_root_from_json())
    assert output_root and os.path.isdir(output_root), (
        f'output_root not set or missing: {output_root!r}.'
    )
    set_data_root(output_root)
    return output_root


def _read_metrics(output_root, base):
    """Read the 5 metrics for one config. Returns dict; missing -> NaN."""
    log_dir = os.path.join(output_root, 'log', 'fly', base)
    one_r  = _parse_pearson(os.path.join(log_dir, 'results_test.log'))
    roll_r = _parse_pearson(os.path.join(log_dir, 'results_rollout.log'))
    m = _parse_metrics_txt(os.path.join(log_dir, 'results', 'metrics.txt'))
    return {
        'one_r':  one_r,
        'roll_r': roll_r,
        'W_R2':   m.get('W_corrected_R2', float('nan')),
        'tau_R2': m.get('tau_R2',         float('nan')),
        'V_R2':   m.get('V_rest_R2',      float('nan')),
    }


def _strip_tex(s):
    return s.replace('\\,', ',').replace('\\ ', ' ').replace('\\', '')


def _print_console_row(model, cond, edges, extent, m):
    label = f'{(model or "..").strip():>10}  {_strip_tex(cond)}'
    edges_disp = _strip_tex(edges)
    print(f'    {label:<55} '
          f'edges={edges_disp:>10} ext={extent:>2} | '
          f'one={_ansi_cell(m["one_r"])} '
          f'roll={_ansi_cell(m["roll_r"])} '
          f'W={_ansi_cell(m["W_R2"])} '
          f'tau={_ansi_cell(m["tau_R2"])} '
          f'V={_ansi_cell(m["V_R2"])}')


def _emit_table(output_root):
    lines = []
    print('\n  [tab ] flywireRF zero-edge  (orange: <0.3 or missing, green: >0.9)')
    for entry in TABLE_ROWS:
        if entry == DOUBLE_RULE:
            lines.append('\\midrule')
            print('    ' + '-' * 80)
            continue
        model, cond, edges, extent, base = entry
        m = _read_metrics(output_root, base)
        lines.append(
            f'{model:<24} & {cond:<40} & ${edges}$ & {extent}\n'
            f'  & {_fmt(m["one_r"])} & {_fmt(m["roll_r"])}\n'
            f'  & {_fmt(m["W_R2"])} & {_fmt(m["tau_R2"])} & {_fmt(m["V_R2"])} \\\\'
        )
        _print_console_row(model, cond, edges, extent, m)

    out_dir = os.path.join(output_root, 'log')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'cv_table_flywireRF_zeroedge.tex')
    with open(path, 'w') as f:
        f.write('% flywireRF zero-edge augmentation; rows only; '
                'Known-ODE block, then GNN block separated by midrule midrule.\n')
        for ln in lines:
            f.write(ln + '\n')
    print(f'\n  [tex ] {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output_root', default=None,
                   help='data root with log/ subdir '
                        '(default: $GNN_OUTPUT_ROOT or data_paths.json)')
    args = p.parse_args()

    output_root = _resolve_output_root(args.output_root)

    print('=' * 60)
    print('aggregate flywireRF -> tex table')
    print(f'  data root: {output_root}')
    print('=' * 60)

    _emit_table(output_root)
    print('\ndone.')


if __name__ == '__main__':
    main()
