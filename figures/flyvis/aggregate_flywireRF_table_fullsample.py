"""FULL-SAMPLE twin of aggregate_flywireRF_table.py (rebuttal / NeurIPS review).

Same purpose as aggregate_blank50_tables_fullsample.py, for the hybrid
FlyWire/Flyvis table (Tab. 2: 4 Known-ODE + 4 GNN rows). Reports the parameter
recovery R^2 for tau_hat and V_rest_hat over *all* neurons alongside the inlier
value the submitted table shows.

The submitted aggregate_flywireRF_table.py already reads the full-sample values
(tau_R2 / V_rest_R2) but its _fmt_R2_pair() deliberately dropped them from the
cell ("a few high-leverage outliers can drag it negative ... which was
misleading"). That dropped number is exactly what the reviewer now asks to see,
so this twin puts it back:

    inlier R^2  (full-sample R^2)  [excluded %]

No retraining / re-running: it reads the same per-fold results/metrics.txt.
Full-sample = Supp. Eq. 23 over ALL neurons (variant A: tau_hat = -1/s_i clipped
to [0, 1] by metrics.derive_tau). Does NOT touch the submitted script.

The submitted flywireRF table shows a single mean across folds (no +-SD); this
twin keeps that convention for every column.

Output:
    figures/cv_table_flywireRF_fullsample.tex  — full \\begin{table}...\\end{table}
Pass --rows-only for just the row block.

Run from the devcontainer:
    GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData \
      /workspace/.conda_envs/neural-graph-linux/bin/python \
      figures/flyvis/aggregate_flywireRF_table_fullsample.py
"""

import argparse
import math
import os
import sys

_FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_FIGURES_DIR))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from connectome_gnn.cross.tex import _mean_sd, _parse_pearson, _parse_metrics_txt
from connectome_gnn.utils import load_data_root_from_json, set_data_root


GNN_SUFFIX = 'blank50_flywire'
KO_SUFFIX  = 'blank50_flywire_known_ode'

_LOW_THRESH  = 0.3
_GOOD_THRESH = 0.9
_TAU_CLIP_RANGE = '[0, 1]'
COLOR_ON = 'full'   # 'full' | 'inlier'

# (base, condition_label, n_neurons, n_edges, eye_map_label) — copied verbatim
# from aggregate_flywireRF_table.py so rows line up with the submitted table.
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
    """Per-condition mean across folds (no +-SD, matching the submitted table)."""
    one, roll, W_R2 = [], [], []
    tau_in, tau_full, tau_out = [], [], []
    V_in,   V_full,   V_out   = [], [], []
    n_present = 0
    for i in range(n_folds):
        fd = _fold_dir(output_root, base, suffix, i)
        if not os.path.isdir(fd):
            continue
        n_present += 1
        one.append(_parse_pearson(os.path.join(fd, 'results_test.log')))
        roll.append(_parse_pearson(os.path.join(fd, 'results_rollout.log')))
        m = _parse_metrics_txt(os.path.join(fd, 'results', 'metrics.txt'))
        W_R2.append(m.get('W_corrected_R2',        float('nan')))
        tau_in.append(  m.get('tau_no_outliers_R2',    float('nan')))
        tau_full.append(m.get('tau_R2',                float('nan')))
        tau_out.append( m.get('tau_n_outliers',        float('nan')))
        V_in.append(    m.get('V_rest_no_outliers_R2', float('nan')))
        V_full.append(  m.get('V_rest_R2',             float('nan')))
        V_out.append(   m.get('V_rest_n_outliers',     float('nan')))

    def _m(xs):
        return _mean_sd(xs)[0]
    return {
        'n_present': n_present,
        'one':       _m(one),
        'roll':      _m(roll),
        'W_R2':      _m(W_R2),
        'tau_in':    _m(tau_in),
        'tau_full':  _m(tau_full),
        'tau_out':   _m(tau_out),
        'V_in':      _m(V_in),
        'V_full':    _m(V_full),
        'V_out':     _m(V_out),
    }


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


def _fmt_simple(x):
    if _isnan(x):
        return '$\\cdot$'
    body = f'${x:.2f}$'
    if x > _GOOD_THRESH:
        return f'\\good{{{body}}}'
    if x < _LOW_THRESH:
        return f'\\bad{{{body}}}'
    return body


def _fmt_R2_full(inl, full, n_out, n_neurons):
    """Cell: inlier R^2 (full-sample R^2) [excluded %].  Single means (no SD)."""
    if _isnan(inl) or _isnan(full) or _isnan(n_out) or not n_neurons:
        return '$\\cdot$'
    excl = 100.0 * n_out / n_neurons
    body = f'${inl:.2f}\\,({full:.2f})\\,[{excl:.1f}]$'
    key = full if COLOR_ON == 'full' else inl
    if key > _GOOD_THRESH:
        return f'\\good{{{body}}}'
    if key < _LOW_THRESH:
        return f'\\bad{{{body}}}'
    return body


def _fmt_int_thousands(n):
    return f'{n:,}'.replace(',', '\\,')


def _row(model_label, condition, n_neurons, n_edges, eye_map, s):
    neurons_s = f'${_fmt_int_thousands(n_neurons)}$'
    edges_s   = f'${_fmt_int_thousands(n_edges)}$'
    return (
        f'{model_label:<24} & {condition:<28} & {neurons_s:<10} & {edges_s:<14} & {eye_map}\n'
        f'  & {_fmt_simple(s["one"])} & {_fmt_simple(s["roll"])}\n'
        f'  & {_fmt_simple(s["W_R2"])}\n'
        f'  & {_fmt_R2_full(s["tau_in"], s["tau_full"], s["tau_out"], n_neurons)}\n'
        f'  & {_fmt_R2_full(s["V_in"],   s["V_full"],   s["V_out"],   n_neurons)} \\\\'
    )


def _build_block(output_root, n_folds, suffix, model_label):
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
    print('\n  flywireRF full-sample preview  (mean across folds; · = NaN)')
    print(f'  {"model":<6} {"variant":<7} {"neurons":>7}  {"folds":>5}  '
          f'{"tau_in":>7} {"tau_full":>8} {"tau%":>5}   '
          f'{"V_in":>7} {"V_full":>8} {"V%":>5}')
    for model_label, suffix in (('KODE', KO_SUFFIX), ('GNN', GNN_SUFFIX)):
        for base, _, n_neurons, _, _ in ROW_META:
            s = _aggregate(output_root, base, suffix, n_folds)
            variant = '+null' if 'proximal_nulls' in base else 'plain'
            tp = (100 * s['tau_out'] / n_neurons
                  if n_neurons and not _isnan(s['tau_out']) else float('nan'))
            vp = (100 * s['V_out'] / n_neurons
                  if n_neurons and not _isnan(s['V_out']) else float('nan'))

            def _f(v, w=7):
                return ('·'.rjust(w) if _isnan(v) else f'{v:{w}.2f}')
            print(f'  {model_label:<6} {variant:<7} {n_neurons:>7}  {s["n_present"]:>5}  '
                  f'{_f(s["tau_in"])} {_f(s["tau_full"],8)} {_f(tp,5)}   '
                  f'{_f(s["V_in"])} {_f(s["V_full"],8)} {_f(vp,5)}')


_TABLE_PREAMBLE = r"""\begin{table}[h]
\centering
\caption{GNN recovery on hybrid connectome variants (full-sample twin of the
submitted table). All runs use low model noise $\sigma = 0.05$ ($65$ cell types).
Eye map is either the flyvis hex lattice ($13{,}741$ neurons, extent $= 8$) or
the larger flywire eye map ($50{,}412$ neurons, extent $= 15$).
$R^2_{\hat{\tau}}$ and $R^2_{\hat{V}^{\mathrm{rest}}}$ cells report the
inlier $R^2$ (full-sample $R^2$) [excluded \%]: the inlier value is
\cref{eq:r2_identity} over the residual-filtered set (\cref{eq:outlier_threshold}),
the parenthetical value is the same statistic over \emph{all} neurons (may be
negative), and the bracketed value is the percentage of neurons excluded.
$\hat{\tau}$ is variant~(A), $-1/s_i$ clipped to $[0,1]$.
\textcolor{green!50!black}{Green}: full-sample $> 0.9$. \bad{Orange}: full-sample $< 0.3$.}
\label{tab:zero_edge_fullsample}
\tiny
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccrrrrrr}
\toprule
& & & & & \multicolumn{2}{c}{prediction} & \multicolumn{3}{c}{parameter recovery} \\
model & condition & neurons & edges & eye map
  & one-step $r$ & rollout $r$
  & $R^2_{\widehat{W}}$
  & $R^2_{\widehat{\tau}}$ in (full) [out.\ \%]
  & $R^2_{\widehat{V}^{\mathrm{rest}}}$ in (full) [out.\ \%] \\
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
            f.write('% FULL-SAMPLE twin of cv_table_flywireRF.tex — rows only.\n')
            f.write('% Cols: model & condition & neurons & edges & eye map | one-step r | '
                    'rollout r | W R^2 | tau: inlier (FULL) [out%%] | Vrest: inlier (FULL) [out%%].\n'
                    f'% Full-sample = Supp. Eq. 23 over ALL neurons (variant A, clip {_TAU_CLIP_RANGE}). '
                    f'Colour on {COLOR_ON}-sample.\n')
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
                        help='Output path. Default: figures/cv_table_flywireRF_fullsample.tex')
    args = parser.parse_args()

    output_root = _resolve_output_root(args.output_root)
    out_path = args.out or os.path.join(_FIGURES_DIR, 'cv_table_flywireRF_fullsample.tex')

    print('=' * 66)
    print('aggregate flywireRF -> FULL-SAMPLE tex table (review rebuttal)')
    print(f'  data root:  {output_root}')
    print(f'  n_folds:    {args.n_folds}   colour on: {COLOR_ON}-sample')
    print(f'  out:        {out_path}')
    print('=' * 66)

    _print_console(output_root, args.n_folds)
    _emit(output_root, args.n_folds, out_path, full_table=not args.rows_only)
    print('done.')
