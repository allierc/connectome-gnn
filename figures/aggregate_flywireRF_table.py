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

Output: figures/cv_table_flywireRF_zeroedge.tex

Console preview: orange when val < 0.3 or missing, green when > 0.9.

Example:
    python figures/aggregate_flywireRF_table.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Data root      : /groups/saalfeld/home/allierc/GraphData
# Configs        : <DATA_ROOT>/config/fly/<base>.yaml
#                    <base> in:
#                      flyvis_hybrid_flywireRF_known_ode_noise_005
#                      flyvis_hybrid_flywireRF_zeroedge_sl_known_ode_noise_005
#                      flyvis_hybrid_flywireRF_zeroedge_cross_sl_known_ode_noise_005
#                      flyvis_hybrid_flywireRF_e15_known_ode_noise_005
#                      flyvis_hybrid_flywireRF_zeroedge_cross_sl_e15_known_ode_noise_005
#                      flyvis_hybrid_flywireRF_noise_005
#                      flyvis_hybrid_flywireRF_zeroedge_sl_noise_005
#                      flyvis_hybrid_flywireRF_zeroedge_cross_sl_noise_005
#                      flyvis_hybrid_flywireRF_e15_noise_005
#                      flyvis_hybrid_flywireRF_zeroedge_cross_sl_e15_noise_005
# Trained models : <DATA_ROOT>/log/fly/<base>/models/best_model_with_0_graphs_0.pt
# Eval logs      : <DATA_ROOT>/log/fly/<base>/results_test.log
#                  <DATA_ROOT>/log/fly/<base>/results_rollout.log
#                  <DATA_ROOT>/log/fly/<base>/results/metrics.txt
# Output         : figures/cv_table_flywireRF_zeroedge.tex
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import math
import os
import sys

_FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_FIGURES_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from connectome_gnn.cross.tex import _parse_pearson, _parse_metrics_txt
from connectome_gnn.utils import load_data_root_from_json, set_data_root


# Sentinel for the double-midrule between the Known-ODE and GNN blocks.
DOUBLE_RULE = ('__rule__',)

# (model_label, condition_label, neurons, edges, extent, base). 10 rows
# total (5 Known-ODE + 5 GNN). Edge / neuron counts read from each config
# YAML's `n_edges` and `n_neurons`. Extent-8 has 13,741 neurons; extent-15
# has 45,669 neurons (full hex hex visual field). Empty model_label =
# continuation of previous model block. Extent-15 rows use
# \textit{larger} / \textit{visual field} in the model column to form a
# two-row "Larger visual field" vertical sub-label.
# zeroedge variants: `_sl_` = same-type spatially-local, `_cross_sl_` =
# cross-type spatially-local (different sampling strategies for the
# zero-weight edge augmentation).
TABLE_ROWS = [
    # ---- Known-ODE block ----
    ('Known ODE',                'het. RF (oracle)',                       '13\\,741', '328\\,092',     '8',
        'e8_flywireRF_known_ode_noise_005'),
    ('',                         'het.\\ RF + uncert.\\ edges (same-type)', '13\\,741', '401\\,175',    '8',
        'e8_flywireRF_zeroedge_sl_known_ode_noise_005'),
    ('',                         'het.\\ RF + uncert.\\ edges (cross-type)','13\\,741', '1\\,959\\,994','8',
        'e8_flywireRF_proximal_nulls_known_ode_noise_005'),
    ('\\textit{larger}',         'het. RF (oracle)',                       '45\\,669', '1\\,256\\,695', '—',
        'full_eye_flywireRF_known_ode_noise_005'),
    ('\\textit{visual field}',   'het.\\ RF + uncert.\\ edges (cross-type)','45\\,669', '5\\,411\\,743','—',
        'full_eye_flywireRF_proximal_nulls_known_ode_noise_005'),

    DOUBLE_RULE,

    # ---- GNN block ----
    ('GNN',                      'het. RF (oracle)',                       '13\\,741', '328\\,092',     '8',
        'e8_flywireRF_noise_005'),
    ('',                         'het.\\ RF + uncert.\\ edges (same-type)', '13\\,741', '401\\,175',    '8',
        'e8_flywireRF_zeroedge_sl_noise_005'),
    ('',                         'het.\\ RF + uncert.\\ edges (cross-type)','13\\,741', '1\\,959\\,994','8',
        'e8_flywireRF_proximal_nulls_noise_005'),
    ('\\textit{larger}',         'het. RF (oracle)',                       '45\\,669', '1\\,256\\,695', '—',
        'full_eye_flywireRF_noise_005'),
    ('\\textit{visual field}',   'het.\\ RF + uncert.\\ edges (cross-type)','45\\,669', '5\\,411\\,743','—',
        'full_eye_flywireRF_proximal_nulls_noise_005'),
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


def _fmt_rel(median, iqr):
    """LaTeX cell for relative error: ${median%}{\\pm}{IQR%}$.
    Used for $\\widehat{W}$ (no outlier metric).
    median and iqr are fractions in [0,1]; emit as percent. NaN -> $\\cdot$."""
    if (isinstance(median, float) and math.isnan(median)) or \
       (isinstance(iqr, float) and math.isnan(iqr)):
        return '$\\cdot$'
    return f'${100*median:.1f}{{\\pm}}{100*iqr:.1f}$'


def _fmt_rel_out(median, iqr, count, n_neurons):
    """Combined LaTeX cell: ${med%}{\\pm}{IQR%}\\,(out%)$.
    Used for $\\widehat{\\tau}$ and $\\widehat{V}^{\\mathrm{rest}}$.
    NaN in any field -> $\\cdot$."""
    if (isinstance(median, float) and math.isnan(median)) or \
       (isinstance(iqr, float) and math.isnan(iqr)) or \
       (isinstance(count, float) and math.isnan(count)) or \
       not n_neurons:
        return '$\\cdot$'
    return (f'${100*median:.1f}{{\\pm}}{100*iqr:.1f}'
            f'\\,({100.0*count/n_neurons:.2f})$')


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


def _ansi_rel(median, iqr, width=11):
    if isinstance(median, float) and math.isnan(median):
        return f'{_ANSI_ORANGE}{"  nan  ":<{width}}{_ANSI_RESET}'
    return f'{f"{100*median:.1f}±{100*iqr:.1f}%":<{width}}'


def _ansi_pct(count, n_neurons, width=8):
    if isinstance(count, float) and math.isnan(count) or not n_neurons:
        return f'{_ANSI_ORANGE}{"  nan ":<{width}}{_ANSI_RESET}'
    return f'{f"{100*count/n_neurons:.2f}%":<{width}}'


def _resolve_output_root(output_root):
    output_root = (output_root or os.environ.get('GNN_OUTPUT_ROOT')
                   or load_data_root_from_json())
    assert output_root and os.path.isdir(output_root), (
        f'output_root not set or missing: {output_root!r}.'
    )
    set_data_root(output_root)
    return output_root


def _read_metrics(output_root, base):
    """Read parameter-recovery metrics for one config. Returns dict; missing -> NaN."""
    log_dir = os.path.join(output_root, 'log', 'fly', base)
    one_r  = _parse_pearson(os.path.join(log_dir, 'results_test.log'))
    roll_r = _parse_pearson(os.path.join(log_dir, 'results_rollout.log'))
    m = _parse_metrics_txt(os.path.join(log_dir, 'results', 'metrics.txt'))
    return {
        'one_r':   one_r,
        'roll_r':  roll_r,
        'W_med':   m.get('W_rel_err_median',      float('nan')),
        'W_iqr':   m.get('W_rel_err_iqr',         float('nan')),
        'tau_med': m.get('tau_rel_err_median',    float('nan')),
        'tau_iqr': m.get('tau_rel_err_iqr',       float('nan')),
        'tau_out': m.get('tau_n_outliers',        float('nan')),
        'V_med':   m.get('V_rest_rel_err_median', float('nan')),
        'V_iqr':   m.get('V_rest_rel_err_iqr',    float('nan')),
        'V_out':   m.get('V_rest_n_outliers',     float('nan')),
    }


def _strip_tex(s):
    return s.replace('\\,', ',').replace('\\ ', ' ').replace('\\', '')


def _parse_n_neurons(neurons_str):
    """'13\\,741' -> 13741. Returns 0 on failure."""
    try:
        return int(_strip_tex(neurons_str).replace(',', ''))
    except (ValueError, TypeError):
        return 0


def _print_console_row(model, cond, neurons, edges, extent, m):
    label = f'{(model or "..").strip():>10}  {_strip_tex(cond)}'
    neurons_disp = _strip_tex(neurons)
    edges_disp   = _strip_tex(edges)
    n_neurons    = _parse_n_neurons(neurons)
    print(f'    {label:<55} '
          f'N={neurons_disp:>7} edges={edges_disp:>10} ext={extent:>2} | '
          f'one={_ansi_cell(m["one_r"])} '
          f'roll={_ansi_cell(m["roll_r"])} '
          f'W%={_ansi_rel(m["W_med"], m["W_iqr"])} '
          f'tau%={_ansi_rel(m["tau_med"], m["tau_iqr"])} '
          f'tau_out={_ansi_pct(m["tau_out"], n_neurons)} '
          f'V%={_ansi_rel(m["V_med"], m["V_iqr"])} '
          f'V_out={_ansi_pct(m["V_out"], n_neurons)}')


def _emit_table(output_root):
    lines = []
    print('\n  [tab ] flywireRF zero-edge  (orange: <0.3 or missing, green: >0.9)')
    for entry in TABLE_ROWS:
        if entry == DOUBLE_RULE:
            lines.append('\\midrule')
            print('    ' + '-' * 80)
            continue
        model, cond, neurons, edges, extent, base = entry
        m = _read_metrics(output_root, base)
        n_neurons = _parse_n_neurons(neurons)
        lines.append(
            f'{model:<24} & {cond:<40} & ${neurons}$ & ${edges}$ & {extent}\n'
            f'  & {_fmt(m["one_r"])} & {_fmt(m["roll_r"])}\n'
            f'  & {_fmt_rel(m["W_med"], m["W_iqr"])}\n'
            f'  & {_fmt_rel_out(m["tau_med"], m["tau_iqr"], m["tau_out"], n_neurons)}\n'
            f'  & {_fmt_rel_out(m["V_med"], m["V_iqr"], m["V_out"], n_neurons)} \\\\'
        )
        _print_console_row(model, cond, neurons, edges, extent, m)

    os.makedirs(_FIGURES_DIR, exist_ok=True)
    path = os.path.join(_FIGURES_DIR, 'cv_table_flywireRF_zeroedge.tex')
    with open(path, 'w') as f:
        f.write('% flywireRF zero-edge augmentation; rows only; '
                'Known-ODE block, then GNN block separated by single midrule. '
                'Cols: model & condition & neurons & edges & extent | one-step r | rollout r | '
                'W rel.err%% | tau rel.err%% (out%%) | Vrest rel.err%% (out%%).\n')
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
