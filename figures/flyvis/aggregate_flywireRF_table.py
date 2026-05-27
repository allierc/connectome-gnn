"""Aggregate the 4 flywireRF blank50 connectomes (KODE + GNN, 5-fold CV)
into a single LaTeX table for neurips.tex.

Reads per-fold metrics under
    <output_root>/log/fly/<base>_<suffix>_cv{i:02d}/results/metrics.txt
    <output_root>/log/fly/<base>_<suffix>_cv{i:02d}/results_test.log
    <output_root>/log/fly/<base>_<suffix>_cv{i:02d}/results_rollout.log

Suffixes:
    blank50_flywire             — run_GNN_flywire_blank50.py        (GNN)
    blank50_flywire_known_ode   — run_KnownODE_flywire_blank50.py   (KODE)

Output: figures/cv_table_flywireRF.tex  — full \\begin{table}...\\end{table}
block, drop-in replacement for the hand-written zero-edge table.
Pass --rows-only to emit just the row block (no caption / wrapper).

Single mean across folds (no ±SD shown); matches the format of the table
this script replaces.

Coloring (mirrors caption macros \\good / \\bad):
    val > 0.9   -> \\good{$val$}      (green!50!black)
    val < 0.3   -> \\bad{$val$}       (orange)
For the R²(corrected, out%) triplet cells:
    both R² > 0.9                  -> wrap whole cell in \\good{}
    R² < 0.3                       -> \\bad{R²}, corrected/out% plain
    R² in [0.3, 0.9], corrected>0.9 -> R² plain, \\good{corrected}
    else                           -> all plain

Run from devcontainer:
    /workspace/.conda_envs/neural-graph-linux/bin/python \\
      figures/aggregate_flywireRF_table.py
"""

import argparse
import math
import os
import sys

_FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_FIGURES_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from connectome_gnn.cross.tex import _mean_sd, _parse_pearson, _parse_metrics_txt
from connectome_gnn.utils import load_data_root_from_json, set_data_root


GNN_SUFFIX = 'blank50_flywire'
KO_SUFFIX  = 'blank50_flywire_known_ode'

_LOW_THRESH  = 0.3
_GOOD_THRESH = 0.9


# (base, condition_label, n_neurons, n_edges, eye_map_label).
# Order: per model block, e8 (13,741) rows first then full_eye (50,412),
# with proximal_nulls placed second within each n_neurons group.
ROW_META = [
    ('e8_flywireRF_noise_005',
        'het.\\ RF',                   13_741,   327_358, 'flyvis hex lattice'),
    ('e8_flywireRF_proximal_nulls_noise_005',
        'het.\\ RF + uncert.\\ edges', 13_741, 2_418_403, 'flyvis hex lattice'),
    ('full_eye_flywireRF_noise_005',
        'het.\\ RF',                   50_412, 1_266_378, 'flywire eye map'),
    ('full_eye_flywireRF_proximal_nulls_noise_005',
        'het.\\ RF + uncert.\\ edges', 50_412, 9_642_335, 'flywire eye map'),
]


def _resolve_output_root(output_root):
    output_root = (output_root or os.environ.get('GNN_OUTPUT_ROOT')
                   or load_data_root_from_json())
    assert output_root and os.path.isdir(output_root), (
        f'output_root not set or missing: {output_root!r}.'
    )
    set_data_root(output_root)
    return output_root


def _fold_dir(output_root, base, suffix, fold_i):
    return os.path.join(output_root, 'log', 'fly',
                        f'{base}_{suffix}_cv{fold_i:02d}')


def _aggregate(output_root, base, suffix, n_folds):
    """Per-condition mean across folds for the metrics this table needs."""
    one, roll = [], []
    W_R2 = []
    tau_R2_full, tau_R2_corr, tau_n_out = [], [], []
    V_R2_full,   V_R2_corr,   V_n_out   = [], [], []
    n_present = 0
    for i in range(n_folds):
        fd = _fold_dir(output_root, base, suffix, i)
        if not os.path.isdir(fd):
            continue
        n_present += 1
        one.append(_parse_pearson(os.path.join(fd, 'results_test.log')))
        roll.append(_parse_pearson(os.path.join(fd, 'results_rollout.log')))
        m = _parse_metrics_txt(os.path.join(fd, 'results', 'metrics.txt'))
        W_R2.append(m.get('W_corrected_R2',         float('nan')))
        tau_R2_full.append(m.get('tau_R2',                  float('nan')))
        tau_R2_corr.append(m.get('tau_no_outliers_R2',      float('nan')))
        tau_n_out.append(   m.get('tau_n_outliers',         float('nan')))
        V_R2_full.append(  m.get('V_rest_R2',                float('nan')))
        V_R2_corr.append(  m.get('V_rest_no_outliers_R2',    float('nan')))
        V_n_out.append(    m.get('V_rest_n_outliers',        float('nan')))

    def _mean(xs): return _mean_sd(xs)[0]
    return {
        'n_present':    n_present,
        'one':          _mean(one),
        'roll':         _mean(roll),
        'W_R2':         _mean(W_R2),
        'tau_R2_full':  _mean(tau_R2_full),
        'tau_R2_corr':  _mean(tau_R2_corr),
        'tau_n_out':    _mean(tau_n_out),
        'V_R2_full':    _mean(V_R2_full),
        'V_R2_corr':    _mean(V_R2_corr),
        'V_n_out':      _mean(V_n_out),
    }


def _is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def _fmt_simple(x):
    """one-step r / rollout r / R²_W: $X.XX$ with optional \\good{}/\\bad{}."""
    if _is_nan(x):
        return '$\\cdot$'
    body = f'${x:.2f}$'
    if x > _GOOD_THRESH:
        return f'\\good{{{body}}}'
    if x < _LOW_THRESH:
        return f'\\bad{{{body}}}'
    return body


def _fmt_R2_pair(R2_corr, n_out, n_neurons):
    """R² pair cell: outlier-corrected R² with parenthetical out%.
    The full-set R² is no longer reported (a few high-leverage outliers can
    drag it negative even when the corrected value is in the 0.85–0.95 range,
    which was misleading)."""
    if any(_is_nan(v) for v in (R2_corr, n_out)) or not n_neurons:
        return '$\\cdot$'
    out_pct = 100.0 * n_out / n_neurons
    corr_s  = f'{R2_corr:.2f}'
    out_s   = f'{out_pct:.1f}'

    if R2_corr > _GOOD_THRESH:
        return f'\\good{{${corr_s}\\,({out_s})$}}'
    if R2_corr < _LOW_THRESH:
        return f'\\bad{{${corr_s}$}}\\,$({out_s})$'
    return f'${corr_s}\\,({out_s})$'


def _fmt_int_thousands(n):
    """13741 -> '13\\,741'  (LaTeX thin-space thousand separator)."""
    return f'{n:,}'.replace(',', '\\,')


def _row(model_label, condition, n_neurons, n_edges, eye_map, s):
    neurons_s = f'${_fmt_int_thousands(n_neurons)}$'
    edges_s   = f'${_fmt_int_thousands(n_edges)}$'
    return (
        f'{model_label:<24} & {condition:<28} & {neurons_s:<10} & {edges_s:<14} & {eye_map}\n'
        f'  & {_fmt_simple(s["one"])} & {_fmt_simple(s["roll"])}\n'
        f'  & {_fmt_simple(s["W_R2"])}\n'
        f'  & {_fmt_R2_pair(s["tau_R2_corr"], s["tau_n_out"], n_neurons)}\n'
        f'  & {_fmt_R2_pair(s["V_R2_corr"], s["V_n_out"], n_neurons)} \\\\'
    )


def _build_block(output_root, n_folds, suffix, model_label):
    """4-row block for one model. \\cmidrule between 13,741 and 50,412 sub-blocks.
    The 50,412 sub-block carries a two-line model label
    (\\textit{larger} on row 1, \\textit{visual field} on row 2) — the same
    visual hierarchy used by the table this script replaces."""
    rows = []
    prev_n = None
    is_first = True
    second_eye_block_row = 0
    for base, cond, n_neurons, n_edges, eye_map in ROW_META:
        crossed = prev_n is not None and n_neurons != prev_n
        if crossed:
            rows.append('  \\cmidrule[0.2pt](lr){2-10}')
            second_eye_block_row = 0
        s = _aggregate(output_root, base, suffix, n_folds)
        if is_first:
            label = model_label
            is_first = False
        elif crossed:
            label = '\\textit{larger}'
        elif prev_n is not None and n_neurons == prev_n and not is_first:
            # 13,741 sub-block: only the first row gets the model_label.
            # 50,412 sub-block: the row right after \textit{larger}
            # gets \textit{visual field} as the second-line label.
            second_eye_block_row += 1
            label = ('\\textit{visual field}'
                     if n_neurons == 50_412 and second_eye_block_row == 1
                     else '')
        else:
            label = ''
        rows.append(_row(label, cond, n_neurons, n_edges, eye_map, s))
        prev_n = n_neurons
    return rows


def _print_console(output_root, n_folds):
    print('\n  flywireRF table preview  (mean across folds; · = NaN)')
    print('  ' + '-' * 120)
    print(f'  {"model":<6} {"variant":<7} {"neurons":>7} {"edges":>10}  '
          f'{"folds":>5}  {"one":>5} {"roll":>5}  {"W":>5}  '
          f'{"tau_R2":>6} {"tau_no":>6} {"tau%":>5}  '
          f'{"V_R2":>6} {"V_no":>6} {"V%":>5}')
    for model_label, suffix in (('KODE', KO_SUFFIX), ('GNN', GNN_SUFFIX)):
        for base, _, n_neurons, n_edges, _ in ROW_META:
            s = _aggregate(output_root, base, suffix, n_folds)
            variant = '+null' if 'proximal_nulls' in base else 'plain'
            tau_pct = (100*s['tau_n_out']/n_neurons
                       if n_neurons and not _is_nan(s['tau_n_out'])
                       else float('nan'))
            V_pct   = (100*s['V_n_out']/n_neurons
                       if n_neurons and not _is_nan(s['V_n_out'])
                       else float('nan'))
            def _f(v, w=5):
                return ('  · '.rjust(w) if _is_nan(v)
                        else f'{v:{w}.2f}')
            print(f'  {model_label:<6} {variant:<7} {n_neurons:>7} {n_edges:>10}  '
                  f'{s["n_present"]:>5}  {_f(s["one"])} {_f(s["roll"])}  {_f(s["W_R2"])}  '
                  f'{_f(s["tau_R2_full"], 6)} {_f(s["tau_R2_corr"], 6)} {_f(tau_pct, 5)}  '
                  f'{_f(s["V_R2_full"], 6)} {_f(s["V_R2_corr"], 6)} {_f(V_pct, 5)}')


_TABLE_PREAMBLE = r"""\begin{table}[h]
\centering
\caption{GNN recovery on hybrid connectome variants under connectivity uncertainty (zero-edge augmentation, no coregistration perturbation).
All runs use low model noise $\sigma = 0.05$ ($65$ cell types). Eye map is either the flyvis hex lattice ($13{,}741$ neurons, extent $= 8$) or the larger flywire eye map ($50{,}412$ neurons, extent $= 15$).
$R^2_{\hat{\tau}}$ and $R^2_{\hat{V}^{\mathrm{rest}}}$ are reported on the outlier-corrected neuron set; the parenthetical value is the percentage of neurons dropped (residual-based outlier filter).
\textcolor{green!50!black}{Green}: value $> 0.9$. \bad{Orange}: value $< 0.3$.}
\label{tab:zero_edge}
\tiny
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccrrrrrr}
\toprule
& & & & & \multicolumn{2}{c}{prediction} & \multicolumn{3}{c}{parameter recovery} \\
model & condition & neurons & edges & eye map
  & one-step $r$ & rollout $r$
  & $R^2_{\widehat{W}}$
  & $R^2_{\widehat{\tau}}$ (out.\ \%)
  & $R^2_{\widehat{V}^{\mathrm{rest}}}$ (out.\ \%) \\
\midrule
"""

_TABLE_POSTAMBLE = r"""\bottomrule
\end{tabular}
\end{table}
"""


def _emit(output_root, n_folds, out_path, full_table):
    ko_rows  = _build_block(output_root, n_folds, KO_SUFFIX,  'Known ODE')
    gnn_rows = _build_block(output_root, n_folds, GNN_SUFFIX, 'GNN')
    body = '\n'.join(ko_rows) + '\n\\midrule\n' + '\n'.join(gnn_rows) + '\n'

    with open(out_path, 'w') as f:
        if full_table:
            f.write(_TABLE_PREAMBLE)
            f.write(body)
            f.write(_TABLE_POSTAMBLE)
        else:
            f.write('% flywireRF zero-edge table — KODE block, then GNN block.\n')
            f.write('% Cols: model & condition & neurons & edges & eye map | one-step r | rollout r | '
                    'W R^2 | tau R^2 (corr., out%%) | Vrest R^2 (corr., out%%).\n')
            f.write(body)
    print(f'  [tex ] {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output_root', default=None,
                        help='GraphData root (default: GNN_OUTPUT_ROOT or '
                             'data_paths.json fallback).')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--rows-only', action='store_true',
                        help='Emit only the row block (no \\begin{table} wrapper).')
    parser.add_argument('--out', default=None,
                        help='Output path. Default: figures/cv_table_flywireRF.tex')
    args = parser.parse_args()

    output_root = _resolve_output_root(args.output_root)
    out_path = args.out or os.path.join(_FIGURES_DIR, 'cv_table_flywireRF.tex')

    print('=' * 60)
    print('aggregate flywireRF -> tex table')
    print(f'  data root:  {output_root}')
    print(f'  n_folds:    {args.n_folds}')
    print(f'  out:        {out_path}')
    print(f'  full table: {not args.rows_only}')
    print('=' * 60)

    _print_console(output_root, args.n_folds)
    _emit(output_root, args.n_folds, out_path, full_table=not args.rows_only)
    print('done.')
