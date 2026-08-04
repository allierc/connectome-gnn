"""FULL-SAMPLE twin of aggregate_blank50_tables.py (rebuttal / NeurIPS review).

Reviewer request (W3 / meta-review point 2, Question 2): report the parameter
recovery R^2 for tau_hat, V_rest_hat and W_hat over *all* neurons/edges, not
only the inlier set left after the residual filter (neurips.tex
eq:outlier_threshold, delta_tau=0.1 / delta_Vrest=0.2 / delta_W=5.0).
Negative values are expected and are meant to be reported.

This script does NOT retrain or re-run anything and does NOT touch the submitted
scripts. The full-sample statistic is *already computed and stored* per fold,
via metrics.recovery_param_metrics (the single function computing both the
full-sample and inlier R^2 together for all three quantities):

    results/metrics.txt:
        tau_R2 / V_rest_R2 / W_corrected_R2
            <- compute_r_squared_NSE(gt, learned)   [ALL neurons/edges]
        tau_no_outliers_R2 / V_rest_no_outliers_R2 / W_corrected_no_outliers_R2
            <- same NSE restricted to the inlier set (headline number)
        tau_n_outliers / V_rest_n_outliers / W_corrected_n_outliers
            (excluded counts)

tau_R2 / V_rest_R2 are exactly Supp. Eq. 23 (identity-line Nash-Sutcliffe,
metrics.compute_r_squared_NSE = 1 - mean((gt-hat)^2)/var(gt)) with N = all
neurons. This is *variant (A) clipped*: learned_tau comes from
ode_params.derive_tau(), which reads tau_hat = -1/s_i and clips it to [0, 1]
(metrics.derive_tau). The unclipped *variant (B)* (raw -1/s_i, which can be
catastrophically negative for degenerate slopes s_i -> 0) is NOT stored on disk
and would require re-running the extraction with clipping disabled; report (B)
in one sentence in the text, stating the clip range [0, 1] explicitly.

Cell format (matches the reviewer's spec and Supp. Fig. 15):
    inlier R^2  (full-sample R^2)  [excluded %]
e.g.  0.76+-0.09 (-0.39+-0.20) [13.7]
Green/orange colouring is applied on the FULL-SAMPLE value (COLOR_ON below),
because the honesty of the full-sample number is the whole point of this table;
switch COLOR_ON = 'inlier' to colour on the inlier value instead (matches the
submitted tables).

Outputs (rows-only .tex, drop-in next to the submitted rows):
    figures/cv_table_known_ode_vs_gnn_fullsample.tex    (Tab. 1: 3 Known-ODE + 3 GNN)
    figures/cv_table_gnn_cross_noise_fullsample.tex     (Supp. Tab. 4: 10 GNN degradations)
    figures/cv_table_known_ode_conditions_fullsample.tex(Supp. Tab. 7: 8 Known-ODE degradations)
    figures/fullsample_caption_snippet.tex              (caption sentence to paste)

Run from the devcontainer:
    GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData \
      /workspace/.conda_envs/neural-graph-linux/bin/python \
      figures/flyvis/aggregate_blank50_tables_fullsample.py
"""

import argparse
import math
import os
import sys

_FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_FIGURES_DIR))  # .../connectome-gnn-cx
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from connectome_gnn.cross.tex import _mean_sd, _parse_pearson, _parse_metrics_txt
from connectome_gnn.utils import load_data_root_from_json, set_data_root


# ─────────────────────────────────────────────────────────────────────────────
# Row specs — copied verbatim from aggregate_blank50_tables.py so the two tables
# line up row-for-row. Keep in sync if the submitted script's rows change.
# ─────────────────────────────────────────────────────────────────────────────
GNN_SUFFIX = 'blank50_unified'
KO_SUFFIX  = 'blank50_known_ode'
INR_SUFFIX = 'davis_blank50'

_N_NEURONS_BLANK50 = 13741

# Clip range applied to variant (A) tau_hat by metrics.derive_tau (see module
# docstring). Reported in the caption so the dependence of (A) on the bound is
# explicit, as the reviewer asked.
_TAU_CLIP_RANGE = '[0, 1]'

# Colour the cell on 'full' (full-sample R^2) or 'inlier' (matches submitted).
COLOR_ON = 'full'

ROW_META = {
    'flyvis_noise_free':                   ('noise-free',             '0',    '0',   '434\\,112'),
    'flyvis_noise_005':                    ('low model noise',        '0.05', '0',   '434\\,112'),
    'flyvis_noise_05':                     ('high model noise',       '0.5',  '0',   '434\\,112'),
    'flyvis_noise_005_010':                ('low meas. noise',        '0.05', '0.1', '434\\,112'),
    'flyvis_noise_005_020':                ('mid meas. noise',        '0.05', '0.2', '434\\,112'),
    'flyvis_noise_005_INR':                ('unknown stimulus',       '0.05', '0',   '434\\,112'),
    'flyvis_noise_005_null_edges_pc_400':  ('$+400\\%$ null edges',   '0.05', '0',   '2\\,170\\,560'),
    'flyvis_noise_005_removed_pc_20':      ('$-20\\%$ edges removed', '0.05', '0',   '347\\,000'),
    'flyvis_noise_005_removed_pc_50':      ('$-50\\%$ edges removed', '0.05', '0',   '217\\,056'),
    'flyvis_noise_005_stride_5':           ('$1/5$ frames',           '0.05', '0',   '434\\,112'),
    'flyvis_noise_005_hidden_010_no_ngp':  ('$10\\%$ hidden (no NGP)', '0.05', '0',   '434\\,112'),
    'flyvis_noise_005_hidden_020_no_ngp':  ('$20\\%$ hidden (no NGP)', '0.05', '0',   '434\\,112'),
}

# Tab. 1: (model_label, condition_label, sigma, suffix, base).
TABLE1_SPEC = [
    ('Known ODE',  'noise-free', '0',    KO_SUFFIX,  'flyvis_noise_free'),
    ('',           'low noise',  '0.05', KO_SUFFIX,  'flyvis_noise_005'),
    ('',           'high noise', '0.5',  KO_SUFFIX,  'flyvis_noise_05'),
    ('GNN (ours)', 'noise-free', '0',    GNN_SUFFIX, 'flyvis_noise_free'),
    ('',           'low noise',  '0.05', GNN_SUFFIX, 'flyvis_noise_005'),
    ('',           'high noise', '0.5',  GNN_SUFFIX, 'flyvis_noise_05'),
]

# Supp. Tab. 4: 10 GNN degradation rows (entries are (base, suffix) so the
# GNN+INR row can override the suffix).
GNN_TABLE_BASES = [
    ('flyvis_noise_005',                   GNN_SUFFIX),
    ('flyvis_noise_005_010',               GNN_SUFFIX),
    ('flyvis_noise_005_020',               GNN_SUFFIX),
    ('flyvis_noise_005_INR',               INR_SUFFIX),
    ('flyvis_noise_005_null_edges_pc_400', GNN_SUFFIX),
    ('flyvis_noise_005_removed_pc_20',     GNN_SUFFIX),
    ('flyvis_noise_005_removed_pc_50',     GNN_SUFFIX),
    ('flyvis_noise_005_stride_5',          GNN_SUFFIX),
    ('flyvis_noise_005_hidden_010_no_ngp', GNN_SUFFIX),
    ('flyvis_noise_005_hidden_020_no_ngp', GNN_SUFFIX),
]

# Supp. Tab. 7: 8 Known-ODE degradation rows.
KO_BASES = [
    'flyvis_noise_free',
    'flyvis_noise_005',
    'flyvis_noise_05',
    'flyvis_noise_005_010',
    'flyvis_noise_005_020',
    'flyvis_noise_005_null_edges_pc_400',
    'flyvis_noise_005_removed_pc_20',
    'flyvis_noise_005_removed_pc_50',
]


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────
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
    """Per-(base,suffix) mean+-SD across folds. Adds full-sample R^2 for W,
    tau and V_rest (the bare _R2 keys) next to the inlier values used by the
    submitted table (the _no_outliers_R2 keys) — same convention for all three
    now that W_corrected_R2 was made full-sample to match tau/V_rest."""
    one, roll, cl = [], [], []
    W_in,   W_full,   W_out   = [], [], []
    tau_in, tau_full, tau_out = [], [], []
    V_in,   V_full,   V_out   = [], [], []
    for i in range(n_folds):
        fd = _fold_dir(output_root, base, suffix, i)
        if not os.path.isdir(fd):
            continue
        one.append(_parse_pearson(os.path.join(fd, 'results_test.log')))
        roll.append(_parse_pearson(os.path.join(fd, 'results_rollout.log')))
        m = _parse_metrics_txt(os.path.join(fd, 'results', 'metrics.txt'))
        cl.append(  m.get('clustering_accuracy',   float('nan')))
        W_in.append(    m.get('W_corrected_no_outliers_R2', float('nan')))
        W_full.append(  m.get('W_corrected_R2',             float('nan')))
        W_out.append(   m.get('W_corrected_n_outliers',     float('nan')))
        tau_in.append(  m.get('tau_no_outliers_R2',    float('nan')))
        tau_full.append(m.get('tau_R2',                float('nan')))
        tau_out.append( m.get('tau_n_outliers',        float('nan')))
        V_in.append(    m.get('V_rest_no_outliers_R2', float('nan')))
        V_full.append(  m.get('V_rest_R2',             float('nan')))
        V_out.append(   m.get('V_rest_n_outliers',     float('nan')))
    return {
        'one_r':    _mean_sd(one),
        'roll_r':   _mean_sd(roll),
        'cluster':  _mean_sd(cl),
        'W_in':     _mean_sd(W_in),
        'W_full':   _mean_sd(W_full),
        'W_out':    _mean_sd(W_out)[0],
        'tau_in':   _mean_sd(tau_in),
        'tau_full': _mean_sd(tau_full),
        'tau_out':  _mean_sd(tau_out)[0],
        'V_in':     _mean_sd(V_in),
        'V_full':   _mean_sd(V_full),
        'V_out':    _mean_sd(V_out)[0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX cell formatting
# ─────────────────────────────────────────────────────────────────────────────
_LOW_THRESH  = 0.3
_GOOD_THRESH = 0.9


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


def _fmt(mean, sd):
    """one-step r / rollout r / W R^2 / cluster: $mean+-sd$ with good/bad colour.
    Unchanged from the submitted table (these columns are unaffected)."""
    if _isnan(mean):
        return '$\\cdot$'
    body = f"${mean:.2f}{{\\pm}}{sd:.2f}$"
    if mean > _GOOD_THRESH:
        return f"\\good{{{body}}}"
    if mean < _LOW_THRESH:
        return f"\\bad{{{body}}}"
    return body


def _fmt_R2_full(inl, full, out_count, n_neurons):
    """Combined cell: inlier R^2 (full-sample R^2) [excluded %].

    inl, full are (mean, sd) tuples across folds; out_count is the mean number
    of excluded neurons across folds. Colouring follows COLOR_ON on the
    full-sample (or inlier) mean."""
    inl_m, inl_s = inl
    full_m, full_s = full
    if _isnan(inl_m) or _isnan(full_m) or _isnan(out_count) or not n_neurons:
        return '$\\cdot$'
    excl_pct = 100.0 * out_count / n_neurons
    body = (f"${inl_m:.2f}{{\\pm}}{inl_s:.2f}"
            f"\\,({full_m:.2f}{{\\pm}}{full_s:.2f})"
            f"\\,[{excl_pct:.1f}]$")
    key = full_m if COLOR_ON == 'full' else inl_m
    if key > _GOOD_THRESH:
        return f"\\good{{{body}}}"
    if key < _LOW_THRESH:
        return f"\\bad{{{body}}}"
    return body


# ─────────────────────────────────────────────────────────────────────────────
# Console preview
# ─────────────────────────────────────────────────────────────────────────────
_ANSI = {'orange': '\033[38;5;208m', 'green': '\033[92m', 'reset': '\033[0m'}


def _cell(mean, sd, w=12):
    if _isnan(mean):
        return f"{_ANSI['orange']}{'nan':<{w}}{_ANSI['reset']}"
    body = f'{mean:.2f}±{sd:.2f}'
    col = (_ANSI['orange'] if mean < _LOW_THRESH else
           _ANSI['green'] if mean > _GOOD_THRESH else '')
    return f"{col}{body:<{w}}{_ANSI['reset'] if col else ''}" if col else f'{body:<{w}}'


def _print_row(label, s):
    tp = (100 * s['tau_out'] / _N_NEURONS_BLANK50
          if not _isnan(s['tau_out']) else float('nan'))
    vp = (100 * s['V_out'] / _N_NEURONS_BLANK50
          if not _isnan(s['V_out']) else float('nan'))
    print(f"    {label:<24} "
          f"tau[in={_cell(*s['tau_in'])} full={_cell(*s['tau_full'])} excl={tp:5.1f}%]  "
          f"V[in={_cell(*s['V_in'])} full={_cell(*s['V_full'])} excl={vp:5.1f}%]")


# ─────────────────────────────────────────────────────────────────────────────
# Emitters
# ─────────────────────────────────────────────────────────────────────────────
_HEADER_NOTE = (
    "% Cols: ... | tau R^2 | Vrest R^2 | cluster.  "
    "tau/Vrest cell = inlier R^2 (FULL-SAMPLE R^2) [excluded %], "
    "mean{\\pm}SD over 5 folds; excluded %% = mean(n_outliers/n_neurons). "
    "Full-sample = Supp. Eq. 23 over ALL neurons (variant A: tau_hat clipped to "
    f"{_TAU_CLIP_RANGE}). W R^2 and cluster are unfiltered (unchanged). "
    f"Colour on {COLOR_ON}-sample value.\n"
)


def _emit_table1(output_root, n_folds):
    lines = []
    prev_suffix = None
    print('\n  [Tab. 1] Known-ODE vs GNN  (full-sample twin)')
    for model, label, sigma, suffix, base in TABLE1_SPEC:
        if prev_suffix is not None and suffix != prev_suffix:
            lines.append('\\midrule')
        s = _aggregate(output_root, base, suffix, n_folds)
        lines.append(
            f'{model:<10} & {label:<11} & ${sigma}$\n'
            f'  & {_fmt(*s["one_r"])} & {_fmt(*s["roll_r"])}\n'
            f'  & {_fmt_R2_full(s["W_in"], s["W_full"], s["W_out"], _N_NEURONS_BLANK50)}\n'
            f'  & {_fmt_R2_full(s["tau_in"], s["tau_full"], s["tau_out"], _N_NEURONS_BLANK50)}\n'
            f'  & {_fmt_R2_full(s["V_in"],   s["V_full"],   s["V_out"],   _N_NEURONS_BLANK50)}\n'
            f'  & {_fmt(*s["cluster"])} \\\\'
        )
        _print_row(f'{(model or "..").strip()} {label}'.strip(), s)
        prev_suffix = suffix
    path = os.path.join(_FIGURES_DIR, 'cv_table_known_ode_vs_gnn_fullsample.tex')
    with open(path, 'w') as f:
        f.write('% FULL-SAMPLE twin of cv_table_known_ode_vs_gnn.tex (rows only).\n')
        f.write(_HEADER_NOTE)
        f.write('\n'.join(lines) + '\n')
    print(f'  [tex ] {path}')


def _emit_condition_table(output_root, n_folds, suffix, bases, out_name, header):
    lines = []
    print(f'\n  [{out_name}]  (full-sample twin)')
    for entry in bases:
        base, row_suffix = entry if isinstance(entry, tuple) else (entry, suffix)
        meta = ROW_META.get(base)
        if meta is None:
            print(f'  [warn] no ROW_META for {base!r} — skipping')
            continue
        label, sigma, gamma, edges = meta
        s = _aggregate(output_root, base, row_suffix, n_folds)
        lines.append(
            f'{label:<24} & ${sigma}$ & ${gamma}$ & ${edges}$\n'
            f'  & {_fmt(*s["one_r"])} & {_fmt(*s["roll_r"])}\n'
            f'  & {_fmt_R2_full(s["W_in"], s["W_full"], s["W_out"], _N_NEURONS_BLANK50)}\n'
            f'  & {_fmt_R2_full(s["tau_in"], s["tau_full"], s["tau_out"], _N_NEURONS_BLANK50)}\n'
            f'  & {_fmt_R2_full(s["V_in"],   s["V_full"],   s["V_out"],   _N_NEURONS_BLANK50)}\n'
            f'  & {_fmt(*s["cluster"])} \\\\'
        )
        _print_row(label, s)
    path = os.path.join(_FIGURES_DIR, out_name)
    with open(path, 'w') as f:
        f.write(f'% {header}\n')
        f.write(_HEADER_NOTE)
        f.write('\n'.join(lines) + '\n')
    print(f'  [tex ] {path}')


def _emit_caption_snippet():
    """One-liner caption + the variant-(B) sentence the reviewer asked for."""
    path = os.path.join(_FIGURES_DIR, 'fullsample_caption_snippet.tex')
    txt = (
        "% Paste into the caption of every table that now carries full-sample R^2.\n"
        "For the per-neuron parameters, each cell reports the inlier $R^2$ "
        "(full-sample $R^2$) [excluded \\%]: the inlier value is the identity-line "
        "$R^2$ (\\cref{eq:r2_identity}) over the residual-filtered set "
        "(\\cref{eq:outlier_threshold}), the parenthetical value is the same "
        "statistic over \\emph{all} neurons, and the bracketed value is the "
        "percentage of neurons excluded by the filter. Full-sample $R^2$ can be "
        "negative and is reported as-is. Here $\\hat{\\tau}$ uses variant~(A), "
        f"i.e.\\ $\\hat{{\\tau}}_i=-1/s_i$ clipped to {_TAU_CLIP_RANGE}; the "
        "unclipped variant~(B) ($\\hat{\\tau}_i=-1/s_i$ with no bound) can be "
        "arbitrarily negative for degenerate slopes $s_i\\to 0$ and is not "
        "tabulated. $R^2_{\\hat{W}}$ and cluster accuracy are already unfiltered "
        "and are unchanged.\n"
    )
    with open(path, 'w') as f:
        f.write(txt)
    print(f'  [tex ] {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output_root', default=None,
                   help='data root with log/ subdir '
                        '(default: $GNN_OUTPUT_ROOT or data_paths.json)')
    p.add_argument('--n_folds', type=int, default=5)
    args = p.parse_args()

    output_root = _resolve_output_root(args.output_root)

    print('=' * 66)
    print('aggregate blank50 -> FULL-SAMPLE tex tables (review rebuttal)')
    print(f'  data root: {output_root}')
    print(f'  n folds:   {args.n_folds}   colour on: {COLOR_ON}-sample')
    print('=' * 66)

    _emit_table1(output_root, args.n_folds)
    _emit_condition_table(
        output_root, args.n_folds, GNN_SUFFIX, GNN_TABLE_BASES,
        out_name='cv_table_gnn_cross_noise_fullsample.tex',
        header='FULL-SAMPLE twin of cv_table_gnn_cross_noise.tex '
               '(Supp. Tab. 4, 10 GNN degradations; rows only).')
    _emit_condition_table(
        output_root, args.n_folds, KO_SUFFIX, KO_BASES,
        out_name='cv_table_known_ode_conditions_fullsample.tex',
        header='FULL-SAMPLE twin of cv_table_known_ode_conditions.tex '
               '(Supp. Tab. 7, 8 Known-ODE degradations; rows only).')
    _emit_caption_snippet()

    print('\ndone.')


if __name__ == '__main__':
    main()
