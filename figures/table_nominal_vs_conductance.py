"""Nominal (flyvis_A) vs conductance (flyvis_conductance) — 5-fold CV comparison,
same row/column structure as tab:cv_known_ode_vs_gnn (Table 1) in neurips.tex.

Reads per-fold results from <output_root>/log/fly/flyvis_noise_005_<series>_cv{i:02d}/:
    results_test.log         (one-step Pearson r)
    results_rollout.log      (rollout Pearson r)
    results/metrics.txt      (W_corrected_no_outliers_R2, tau_no_outliers_R2,
                              V_rest_no_outliers_R2, clustering_accuracy)
Produced by `python GNN_Main.py -o test_plot flyvis_noise_005_<series>_cvNN`
(or `-o test_plot_cv flyvis_noise_005_<series>` to submit all 5 folds).

Output: figures/table_nominal_vs_conductance.tex (rows only — paste into an
existing tabular, same 7-col layout as cv_table_known_ode_vs_gnn.tex: model &
one-step r & rollout r & W R^2 & tau R^2 (out%) & Vrest R^2 (out%) & cluster).

Usage:
    python table_nominal_vs_conductance.py [--output_root PATH] [--n_folds 5]
"""
import argparse
import os
import sys

_FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_FIGURES_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from connectome_gnn.cross.tex import _mean_sd, _parse_pearson, _parse_metrics_txt
from connectome_gnn.utils import load_data_root_from_json, set_data_root

SERIES = [
    ('nominal',      'Nominal (flyvis\\_A)'),
    ('conductance',  'Conductance (flyvis\\_conductance)'),
]

_LOW_THRESH = 0.3
_GOOD_THRESH = 0.9
_N_NEURONS = 13741


def _resolve_output_root(output_root):
    output_root = (output_root or os.environ.get('GNN_OUTPUT_ROOT')
                   or load_data_root_from_json())
    assert output_root and os.path.isdir(output_root), f'output_root not set or missing: {output_root!r}.'
    set_data_root(output_root)
    return output_root


def _fold_dir(output_root, series, fold_i):
    return os.path.join(output_root, 'log', 'fly', f'flyvis_noise_005_{series}_cv{fold_i:02d}')


def _fmt(mean, sd):
    import math
    if isinstance(mean, float) and math.isnan(mean):
        return '$\\cdot$'
    body = f"${mean:.3f}{{\\pm}}{sd:.3f}$"
    if mean > _GOOD_THRESH:
        return f"\\good{{{body}}}"
    if mean < _LOW_THRESH:
        return f"\\bad{{{body}}}"
    return body


def _fmt_R2_out(mean, sd, count_mean, n_neurons):
    """Two cells (R^2, then its (out%) companion), colored together — matches
    tab:cv_gnn_vs_baselines' 2-column R^2/out%% span (cv_table_known_ode_vs_gnn.tex
    uses one merged cell instead; this table follows the split-cell convention)."""
    import math
    if (isinstance(mean, float) and math.isnan(mean)) or \
       (isinstance(sd, float) and math.isnan(sd)) or \
       (isinstance(count_mean, float) and math.isnan(count_mean)) or not n_neurons:
        return '$\\cdot$', '$\\cdot$'
    r2_body = f"${mean:.3f}{{\\pm}}{sd:.3f}$"
    out_body = f"$\\,({100.0*count_mean/n_neurons:.1f})$"
    if mean > _GOOD_THRESH:
        return f"\\good{{{r2_body}}}", f"\\good{{{out_body}}}"
    if mean < _LOW_THRESH:
        return f"\\bad{{{r2_body}}}", f"\\bad{{{out_body}}}"
    return r2_body, out_body


def _ansi_cell(mean, sd, width=15):
    import math
    if isinstance(mean, float) and math.isnan(mean):
        return f'{"nan":<{width}}'
    return f'{mean:.3f}±{sd:.3f}'.ljust(width)


def _aggregate(output_root, series, n_folds):
    one, roll = [], []
    W_R2 = []
    tau_R2, tau_out = [], []
    V_R2, V_out = [], []
    cl = []
    missing = []
    for i in range(n_folds):
        fd = _fold_dir(output_root, series, i)
        if not os.path.isdir(fd):
            missing.append(i)
            continue
        one.append(_parse_pearson(os.path.join(fd, 'results_test.log')))
        roll.append(_parse_pearson(os.path.join(fd, 'results_rollout.log')))
        m = _parse_metrics_txt(os.path.join(fd, 'results', 'metrics.txt'))
        W_R2.append(m.get('W_corrected_no_outliers_R2', float('nan')))
        tau_R2.append(m.get('tau_no_outliers_R2', float('nan')))
        tau_out.append(m.get('tau_n_outliers', float('nan')))
        V_R2.append(m.get('V_rest_no_outliers_R2', float('nan')))
        V_out.append(m.get('V_rest_n_outliers', float('nan')))
        cl.append(m.get('clustering_accuracy', float('nan')))
    return {
        'one_r': _mean_sd(one), 'roll_r': _mean_sd(roll),
        'W_R2': _mean_sd(W_R2),
        'tau_R2': _mean_sd(tau_R2), 'tau_out': _mean_sd(tau_out)[0],
        'V_R2': _mean_sd(V_R2), 'V_out': _mean_sd(V_out)[0],
        'cluster': _mean_sd(cl),
        'n_folds_found': n_folds - len(missing), 'missing': missing,
    }, {'W_R2_raw': W_R2, 'one_r_raw': one, 'roll_r_raw': roll}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output_root', default=None)
    p.add_argument('--n_folds', type=int, default=5)
    args = p.parse_args()

    output_root = _resolve_output_root(args.output_root)

    print('=' * 78)
    print('nominal vs conductance — 5-fold CV comparison')
    print(f'  data root: {output_root}')
    print('=' * 78)

    lines = []
    for series, label in SERIES:
        s, raw = _aggregate(output_root, series, args.n_folds)
        if s['missing']:
            print(f'  [warn] {series}: missing folds {s["missing"]}')
        print(f'\n  {label}  ({s["n_folds_found"]}/{args.n_folds} folds found)')
        print(f'    one-step r : {_ansi_cell(*s["one_r"])}   raw={["%.3f"%v if v==v else float("nan") for v in raw["one_r_raw"]]}')
        print(f'    rollout r  : {_ansi_cell(*s["roll_r"])}   raw={["%.3f"%v if v==v else float("nan") for v in raw["roll_r_raw"]]}')
        print(f'    W R2       : {_ansi_cell(*s["W_R2"])}   raw={["%.3f"%v if v==v else float("nan") for v in raw["W_R2_raw"]]}')
        print(f'    tau R2     : {_ansi_cell(*s["tau_R2"])}  (out%={100.0*s["tau_out"]/_N_NEURONS:.1f})')
        print(f'    Vrest R2   : {_ansi_cell(*s["V_R2"])}  (out%={100.0*s["V_out"]/_N_NEURONS:.1f})')
        print(f'    cluster    : {_ansi_cell(*s["cluster"])}')

        tau_r2_cell, tau_out_cell = _fmt_R2_out(*s["tau_R2"], s["tau_out"], _N_NEURONS)
        V_r2_cell, V_out_cell = _fmt_R2_out(*s["V_R2"], s["V_out"], _N_NEURONS)
        lines.append(
            f'{label}\n'
            f'  & {_fmt(*s["one_r"])} & {_fmt(*s["roll_r"])}\n'
            f'  & {_fmt(*s["W_R2"])}\n'
            f'  & {tau_r2_cell} & {tau_out_cell}\n'
            f'  & {V_r2_cell} & {V_out_cell}\n'
            f'  & {_fmt(*s["cluster"])} \\\\'
        )

    rows_tex = ('\n\\midrule\n'.join(lines))

    path = os.path.join(_FIGURES_DIR, 'table_nominal_vs_conductance.tex')
    with open(path, 'w') as f:
        f.write('% Nominal (flyvis_A) vs conductance (flyvis_conductance), blank50, 5-fold CV (rows only).\n')
        f.write('% Cols: model | one-step r | rollout r | W R^2 | tau R^2 | (out%%) | Vrest R^2 | (out%%) | cluster acc.\n')
        f.write('% (R^2 for tau/Vrest/W on the no-outlier subset; out%% = mean(n_outliers / n_neurons) across folds, 1 dec.)\n')
        f.write('% Split R^2/(out%%) into two cells each -- matches tab:cv_gnn_vs_baselines header spans,\n')
        f.write('% not cv_table_known_ode_vs_gnn.tex\'s single merged-cell format.\n')
        f.write(rows_tex + '\n')
    print(f'\n  [tex] {path}')

    wrapper_path = os.path.join(_FIGURES_DIR, 'table_nominal_vs_conductance_standalone.tex')
    with open(wrapper_path, 'w') as f:
        f.write(_STANDALONE_TEMPLATE.replace('%%ROWS%%', rows_tex))
    print(f'  [tex] {wrapper_path}  (standalone, compilable: pdflatex table_nominal_vs_conductance_standalone.tex)')


_STANDALONE_TEMPLATE = r"""\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{xcolor}
\newcommand{\good}[1]{\textcolor{green!50!black}{#1}}
\newcommand{\bad}[1]{\textcolor{orange}{#1}}
\pagestyle{empty}
\begin{document}
\begin{table}[t]
\centering
\caption{\textbf{Nominal vs.\ conductance GNN on Flyvis-217}, blank50, noise $\sigma=0.05$; 5-fold CV (mean~$\pm$~SD). Prediction metrics on noise-free held-out stimuli. Parameter recovery: $R^2_{\hat{W}}$ over all $434{,}112$ non-zero edges; $R^2_{\hat{\tau}}$ and $R^2_{\hat{V}^{\mathrm{rest}}}$ over all $13{,}741$ neurons, outlier-corrected; GMM clustering accuracy over $65$ cell types. \good{Green}: $>0.9$, \bad{orange}: $<0.3$.}
\label{tab:nominal_vs_conductance}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} lrrrr@{}rr@{}rr@{}}
\toprule
& \multicolumn{2}{c}{prediction} & \multicolumn{6}{c}{parameter recovery} \\
model
  & \multicolumn{1}{c}{one-step $r$} & \multicolumn{1}{c}{rollout $r$}
  & \multicolumn{1}{c}{$R^2_{\hat{W}}$}
  & \multicolumn{2}{c}{$R^2_{\hat{\tau}}\ (\text{out.}\,\%)$}
  & \multicolumn{2}{c}{$R^2_{\hat{V}^{\mathrm{rest}}}\ (\text{out.}\,\%)$}
  & \multicolumn{1}{c}{cluster\ acc.} \\
\midrule
%%ROWS%%
\bottomrule
\end{tabular*}
\end{table}
\end{document}
"""


if __name__ == '__main__':
    main()
