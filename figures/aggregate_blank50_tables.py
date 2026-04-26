"""Aggregate blank50 GNN + Known-ODE metrics into 3 LaTeX row blocks.

Reads per-fold results under <output_root>/log/fly/<base>_<suffix>_cv{i:02d}/
for each (base, fold) in the two blank50 pipelines and writes 3 tex files
to <output_root>/log/. Each file contains only the row block (no \\begin{table},
no caption, no \\bottomrule wrapper) — paste into the paper between the
existing column headers and \\bottomrule.

Inputs (per fold):
    log/fly/<base>_<suffix>_cv{i:02d}/results_test.log         (one-step r)
    log/fly/<base>_<suffix>_cv{i:02d}/results_rollout.log      (rollout r)
    log/fly/<base>_<suffix>_cv{i:02d}/results/metrics.txt      (W, tau, V_rest, cluster)

Suffixes:
    blank50_unified    — 11 conditions x 5 folds (run_GNN_unified_blank50.py)
    blank50_known_ode  —  8 conditions x 5 folds (run_KnownODE_blank50.py)

Outputs (rows-only; no captions):
    log/cv_table_known_ode_vs_gnn.tex      Known-ODE x{2 noise levels} + GNN x{3 noise levels}
    log/cv_table_gnn_cross_noise.tex       all 11 GNN blank50 conditions
    log/cv_table_known_ode_conditions.tex  all 8  Known-ODE blank50 conditions

Optional --data_plot: re-submit data_plot-only cluster jobs for every GNN fold
(55 jobs) before reading metrics. Forces overwrite of existing metrics.txt.
Use after editing parameter extraction in GNN_PlotFigure.py.

Optional --test_plot: re-submit the full cross test+plot wave (data_test
rollout + data_plot) for every GNN fold (55 jobs). Force-removes existing
rollout/test logs and metrics.txt so stale artifacts can't shadow the rerun.
Implies --data_plot. Use when you want to redo the whole post-training
evaluation pipeline.

Example:
    python aggregate_blank50_tables.py --test_plot --node_name l4 --test_plot_runtime_min 360
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from connectome_gnn.cross.tex import _mean_sd, _parse_pearson, _parse_metrics_txt
from connectome_gnn.cross.yaml_io import shared_cv_yaml_path, _load_yaml_either
from connectome_gnn.LLM.cluster import (
    submit_cluster_cross_test_plot_job, submit_cluster_data_plot_job,
    wait_for_cluster_jobs,
)
from connectome_gnn.utils import (
    load_data_root_from_json, log_path, set_data_root,
)


GNN_SUFFIX = 'blank50_unified'
KO_SUFFIX  = 'blank50_known_ode'

# All 11 GNN blank50 conditions (matches CONDITION_BASES in cross/pipeline.py).
# Used by --data_plot to iterate every trained fold.
GNN_BASES = [
    'flyvis_noise_free',
    'flyvis_noise_005',
    'flyvis_noise_05',
    'flyvis_noise_005_010',
    'flyvis_noise_005_020',
    'flyvis_noise_005_null_edges_pc_400',
    'flyvis_noise_005_removed_pc_20',
    'flyvis_noise_005_removed_pc_50',
    'flyvis_noise_005_stride_5',
    'flyvis_noise_005_hidden_010_ngp',
    'flyvis_noise_005_hidden_020_ngp',
]

# Subset shown in tab:cv_cross_noise (drops noise_free + noise_05; both are
# already covered by tab:cv_known_ode-vs-gnn, so this table focuses on the
# variants away from the canonical low-noise reference).
GNN_TABLE_BASES = [b for b in GNN_BASES
                   if b not in ('flyvis_noise_free', 'flyvis_noise_05')]

# 8 Known-ODE blank50 conditions (run_KnownODE_blank50.py CONDITION_NODES).
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

# (base, label, sigma, gamma, edges) — used by tab:cv_cross_noise (GNN) and
# tab:cv_known_ode-conditions (Known-ODE). Edges count for removed_pc_50 is
# approximate (50% of 434,112). hidden_*_ngp and stride_5 use full edges.
ROW_META = {
    'flyvis_noise_free':                   ('noise-free',             '0',    '0',   '434\\,112'),
    'flyvis_noise_005':                    ('low model noise',        '0.05', '0',   '434\\,112'),
    'flyvis_noise_05':                     ('high model noise',       '0.5',  '0',   '434\\,112'),
    'flyvis_noise_005_010':                ('low meas. noise',        '0.05', '0.1', '434\\,112'),
    'flyvis_noise_005_020':                ('mid meas. noise',        '0.05', '0.2', '434\\,112'),
    'flyvis_noise_005_null_edges_pc_400':  ('$+400\\%$ null edges',   '0.05', '0',   '2\\,170\\,560'),
    'flyvis_noise_005_removed_pc_20':      ('$-20\\%$ edges removed', '0.05', '0',   '347\\,000'),
    'flyvis_noise_005_removed_pc_50':      ('$-50\\%$ edges removed', '0.05', '0',   '217\\,056'),
    'flyvis_noise_005_stride_5':           ('$1/5$ frames',           '0.05', '0',   '434\\,112'),
    'flyvis_noise_005_hidden_010_ngp':     ('$10\\%$ hidden',         '0.05', '0',   '434\\,112'),
    'flyvis_noise_005_hidden_020_ngp':     ('$20\\%$ hidden',         '0.05', '0',   '434\\,112'),
}

# Table 1 (Known-ODE vs GNN) row spec: (model_label, condition_label, sigma, suffix, base).
# Empty model_label = continuation of previous model block (suppress the column).
TABLE1_SPEC = [
    ('Known ODE',  'noise-free', '0',    KO_SUFFIX,  'flyvis_noise_free'),
    ('',           'low noise',  '0.05', KO_SUFFIX,  'flyvis_noise_005'),
    ('',           'high noise', '0.5',  KO_SUFFIX,  'flyvis_noise_05'),
    ('GNN (ours)', 'noise-free', '0',    GNN_SUFFIX, 'flyvis_noise_free'),
    ('',           'low noise',  '0.05', GNN_SUFFIX, 'flyvis_noise_005'),
    ('',           'high noise', '0.5',  GNN_SUFFIX, 'flyvis_noise_05'),
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


_ANSI_ORANGE = '\033[38;5;208m'
_ANSI_GREEN  = '\033[92m'
_ANSI_RESET  = '\033[0m'
_LOW_THRESH  = 0.3   # mean strictly below this -> orange / \bad{}
_GOOD_THRESH = 0.9   # mean strictly above this -> green  / \good{}


def _fmt(mean, sd):
    """LaTeX cell. Wraps \\good{...} when mean > 0.9, \\bad{...} when mean < 0.3,
    and emits $\\cdot$ for NaN. Define both macros once in the preamble:
        \\usepackage{xcolor}
        \\newcommand{\\good}[1]{\\textcolor{green!50!black}{#1}}
        \\newcommand{\\bad}[1]{\\textcolor{orange}{#1}}"""
    import math
    if isinstance(mean, float) and math.isnan(mean):
        return '$\\cdot$'
    body = f"${mean:.2f}{{\\pm}}{sd:.2f}$"
    if mean > _GOOD_THRESH:
        return f"\\good{{{body}}}"
    if mean < _LOW_THRESH:
        return f"\\bad{{{body}}}"
    return body


def _ansi_cell(mean, sd, width=11):
    """ANSI-coloured cell for console preview. Orange when mean < 0.3 or NaN,
    green when mean > 0.9, plain otherwise."""
    import math
    if isinstance(mean, float) and math.isnan(mean):
        body = '   nan   '
        return f'{_ANSI_ORANGE}{body:<{width}}{_ANSI_RESET}'
    body = f'{mean:.2f}±{sd:.2f}'
    if mean < _LOW_THRESH:
        return f'{_ANSI_ORANGE}{body:<{width}}{_ANSI_RESET}'
    if mean > _GOOD_THRESH:
        return f'{_ANSI_GREEN}{body:<{width}}{_ANSI_RESET}'
    return f'{body:<{width}}'


def _print_console_row(label, s):
    """Print one aggregated row to stdout with red highlighting on weak means."""
    print(f'    {label:<24} '
          f'one={_ansi_cell(*s["one_r"])} '
          f'roll={_ansi_cell(*s["roll_r"])} '
          f'W={_ansi_cell(*s["W_R2"])} '
          f'tau={_ansi_cell(*s["tau_R2"])} '
          f'V={_ansi_cell(*s["V_R2"])} '
          f'cl={_ansi_cell(*s["cluster"])}')


def _aggregate(output_root, base, suffix, n_folds):
    """Aggregate per-fold metrics for one (base, suffix). Returns dict of
    (mean, sd) tuples — NaN means no folds had that metric."""
    one, roll = [], []
    W, tau, V, cl = [], [], [], []
    for i in range(n_folds):
        fd = _fold_dir(output_root, base, suffix, i)
        if not os.path.isdir(fd):
            continue
        one.append(_parse_pearson(os.path.join(fd, 'results_test.log')))
        roll.append(_parse_pearson(os.path.join(fd, 'results_rollout.log')))
        m = _parse_metrics_txt(os.path.join(fd, 'results', 'metrics.txt'))
        W.append(m.get('W_corrected_R2',     float('nan')))
        tau.append(m.get('tau_R2',           float('nan')))
        V.append(m.get('V_rest_R2',          float('nan')))
        cl.append(m.get('clustering_accuracy', float('nan')))
    return {
        'one_r':    _mean_sd(one),
        'roll_r':   _mean_sd(roll),
        'W_R2':     _mean_sd(W),
        'tau_R2':   _mean_sd(tau),
        'V_R2':     _mean_sd(V),
        'cluster':  _mean_sd(cl),
    }


def _emit_table1(output_root, n_folds):
    """tab:cv_known_ode (model x noise comparison). 5 rows, 9-col layout
    (model, condition, sigma, one-step r, rollout r, W, tau, V, cluster).
    Inserts \\midrule\\midrule between the Known-ODE block and the GNN block."""
    lines = []
    prev_suffix = None
    print('\n  [tab1] Known-ODE vs GNN  (orange: <0.3 or missing, green: >0.9)')
    for model, label, sigma, suffix, base in TABLE1_SPEC:
        if prev_suffix is not None and suffix != prev_suffix:
            lines.append('\\midrule\n\\midrule')
        s = _aggregate(output_root, base, suffix, n_folds)
        lines.append(
            f'{model:<10} & {label:<11} & ${sigma}$\n'
            f'  & {_fmt(*s["one_r"])} & {_fmt(*s["roll_r"])}\n'
            f'  & {_fmt(*s["W_R2"])} & {_fmt(*s["tau_R2"])} & '
            f'{_fmt(*s["V_R2"])} & {_fmt(*s["cluster"])} \\\\'
        )
        _print_console_row(f'{(model or "..").strip()} {label}'.strip(), s)
        prev_suffix = suffix
    path = os.path.join(output_root, 'log', 'cv_table_known_ode_vs_gnn.tex')
    with open(path, 'w') as f:
        f.write('% Known-ODE vs GNN, blank50, 5-fold CV (rows only).\n')
        for ln in lines:
            f.write(ln + '\n')
    print(f'  [tex ] {path}')


def _emit_condition_table(output_root, n_folds, suffix, bases, out_name, header):
    """tab:cv_cross_noise / tab:cv_known_ode-conditions: condition rows,
    10-col layout (condition, sigma, gamma, edges, one-step r, rollout r,
    W, tau, V, cluster)."""
    lines = []
    print(f'\n  [tab ] {out_name}  (orange: <0.3 or missing, green: >0.9)')
    for base in bases:
        meta = ROW_META.get(base)
        if meta is None:
            print(f'  [warn] no ROW_META for {base!r} — skipping')
            continue
        label, sigma, gamma, edges = meta
        s = _aggregate(output_root, base, suffix, n_folds)
        lines.append(
            f'{label:<24} & ${sigma}$ & ${gamma}$ & ${edges}$\n'
            f'  & {_fmt(*s["one_r"])} & {_fmt(*s["roll_r"])}\n'
            f'  & {_fmt(*s["W_R2"])} & {_fmt(*s["tau_R2"])} & '
            f'{_fmt(*s["V_R2"])} & {_fmt(*s["cluster"])} \\\\'
        )
        _print_console_row(label, s)
    path = os.path.join(output_root, 'log', out_name)
    with open(path, 'w') as f:
        f.write(f'% {header}\n')
        for ln in lines:
            f.write(ln + '\n')
    print(f'  [tex ] {path}')


def _submit_data_plot_jobs(output_root, n_folds, node_name, runtime_min):
    """Force-rerun data_plot on every GNN blank50 fold (55 jobs). Removes
    existing metrics.txt first so missing/stale extraction is regenerated."""
    job_ids, log_dirs = {}, {}
    slot = 0
    for base in GNN_BASES:
        for i in range(n_folds):
            fd = _fold_dir(output_root, base, GNN_SUFFIX, i)
            if not os.path.isdir(fd):
                print(f'  [skip] {base} fold {i}: no log dir')
                continue
            cfg_file_field = f'fly/{base}_{GNN_SUFFIX}_cv{i:02d}'
            cfg_path = shared_cv_yaml_path(cfg_file_field, output_root)
            if not os.path.isfile(cfg_path):
                cfg_path = _load_yaml_either(f'{base}_{GNN_SUFFIX}_cv{i:02d}',
                                              output_root)
            if not os.path.isfile(cfg_path):
                print(f'  [skip] {base} fold {i}: yaml not found')
                continue
            metrics_path = os.path.join(fd, 'results', 'metrics.txt')
            if os.path.isfile(metrics_path):
                os.remove(metrics_path)
                print(f'  [force] removed {metrics_path}')
            jid = submit_cluster_data_plot_job(
                slot=slot, config_path=cfg_path,
                analysis_log_path=os.path.join(fd, 'cluster_data_plot.log'),
                config_file_field=cfg_file_field,
                log_dir=fd, node_name=node_name,
                output_root=output_root,
                hard_runtime_limit_min=runtime_min,
            )
            if jid is not None:
                job_ids[slot] = jid
                log_dirs[slot] = fd
            slot += 1
    if not job_ids:
        print('  [data_plot] no jobs submitted')
        return
    print(f'\n  [wait] {len(job_ids)} data_plot job(s)')
    # NOTE: simple bjobs poll (no training-metrics readout). The
    # training-time tau in tmp_training/metrics.log is known wrong; only
    # the post-data_plot results/metrics.txt matters here.
    wait_for_cluster_jobs(
        job_ids, log_dir=None, poll_interval=60,
        job_prefix='cluster_data_plot',
    )


def _submit_test_plot_jobs(output_root, n_folds, node_name, runtime_min):
    """Force-rerun the cross test+plot wave for every GNN blank50 fold
    (55 jobs). Each job re-runs data_test (rollout on the held-out 20% of
    its own fold) AND data_plot. Removes existing rollout logs and
    metrics.txt first so stale artifacts can't shadow the rerun.

    Mirrors submit_test_plot_wave in cross/pipeline.py but ignores the
    fold-level skip — every fold is forced."""
    job_ids, log_dirs = {}, {}
    slot = 0
    for base in GNN_BASES:
        for i in range(n_folds):
            fd = _fold_dir(output_root, base, GNN_SUFFIX, i)
            if not os.path.isdir(fd):
                print(f'  [skip] {base} fold {i}: no log dir')
                continue
            cfg_file_field = f'fly/{base}_{GNN_SUFFIX}_cv{i:02d}'
            cfg_path = shared_cv_yaml_path(cfg_file_field, output_root)
            if not os.path.isfile(cfg_path):
                cfg_path = _load_yaml_either(f'{base}_{GNN_SUFFIX}_cv{i:02d}',
                                              output_root)
            if not os.path.isfile(cfg_path):
                print(f'  [skip] {base} fold {i}: yaml not found')
                continue
            for stale in (
                os.path.join(fd, 'results_rollout.log'),
                os.path.join(fd, 'results_test.log'),
                os.path.join(fd, 'results', 'metrics.txt'),
                os.path.join(fd, '_cross_test_plot_complete'),
            ):
                if os.path.isfile(stale):
                    os.remove(stale)
                    print(f'  [force] removed {stale}')
            jid = submit_cluster_cross_test_plot_job(
                slot=slot, config_path=cfg_path,
                test_config_paths=[cfg_path],
                analysis_log_path=os.path.join(fd, 'cluster_cross_test_plot.log'),
                config_file_field=cfg_file_field,
                test_config_file_fields=[cfg_file_field],
                log_dir=fd, node_name=node_name,
                output_root=output_root,
                hard_runtime_limit_min=runtime_min,
                n_rollout_frames=250,
            )
            if jid is not None:
                job_ids[slot] = jid
                log_dirs[slot] = fd
            slot += 1
    if not job_ids:
        print('  [test_plot] no jobs submitted')
        return
    print(f'\n  [wait] {len(job_ids)} test+plot job(s)')
    wait_for_cluster_jobs(
        job_ids, log_dir=None, poll_interval=60,
        job_prefix='cluster_cross_test_plot',
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output_root', default=None,
                   help='data root with log/ and graphs_data/ subdirs '
                        '(default: $GNN_OUTPUT_ROOT or data_paths.json)')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--data_plot', action='store_true',
                   help='re-run data_plot on all 55 GNN blank50 folds before '
                        'aggregating (cluster jobs, force-overwrite metrics.txt)')
    p.add_argument('--test_plot', action='store_true',
                   help='re-run the cross test+plot wave (data_test rollout '
                        '+ data_plot) on all 55 GNN blank50 folds before '
                        'aggregating. Force-removes existing results_rollout.log, '
                        'results_test.log, metrics.txt, and the '
                        '_cross_test_plot_complete marker. Implies --data_plot.')
    p.add_argument('--node_name', default='a100',
                   help='LSF queue suffix for --data_plot/--test_plot jobs '
                        '(default a100)')
    p.add_argument('--runtime_min', type=int, default=60,
                   help='--data_plot job runtime limit in minutes (default 60)')
    p.add_argument('--test_plot_runtime_min', type=int, default=240,
                   help='--test_plot job runtime limit in minutes (default 240; '
                        'rollout dominates wall time)')
    args = p.parse_args()

    output_root = _resolve_output_root(args.output_root)
    os.makedirs(os.path.join(output_root, 'log'), exist_ok=True)

    print('=' * 60)
    print('aggregate blank50 -> tex tables')
    print(f'  data root:  {output_root}')
    print(f'  n folds:    {args.n_folds}')
    print(f'  test_plot:  {args.test_plot}')
    print(f'  data_plot:  {args.data_plot}')
    print('=' * 60)

    if args.test_plot:
        # test_plot supersedes data_plot — the same subprocess runs data_plot
        # at the end of each cross test+plot job.
        print('\n[1] re-running cross test+plot on all GNN blank50 folds')
        _submit_test_plot_jobs(output_root, args.n_folds,
                               args.node_name, args.test_plot_runtime_min)
    elif args.data_plot:
        print('\n[1] re-running data_plot on all GNN blank50 folds')
        _submit_data_plot_jobs(output_root, args.n_folds,
                               args.node_name, args.runtime_min)

    print('\n[2] emit tex')
    _emit_table1(output_root, args.n_folds)
    _emit_condition_table(
        output_root, args.n_folds, GNN_SUFFIX, GNN_TABLE_BASES,
        out_name='cv_table_gnn_cross_noise.tex',
        header='GNN blank50 cross-noise (rows only; '
               'noise-free and high-noise rows omitted, see vs-Known-ODE table).')
    _emit_condition_table(
        output_root, args.n_folds, KO_SUFFIX, KO_BASES,
        out_name='cv_table_known_ode_conditions.tex',
        header='Known-ODE blank50 (rows only).')

    print('\ndone.')


if __name__ == '__main__':
    main()
