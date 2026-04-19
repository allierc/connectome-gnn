"""
Parallel cross-YT runner using cluster bsub jobs for training.

For each condition in CONDITION_BASES, and for each fold i in 0..N-1:

  1. data_generate  (LOCAL, devcontainer GPU — or skipped if data exists)
  2. data_train     (CLUSTER bsub via submit_cluster_job — 5 folds in parallel)
  3. wait for the wave of 5 cluster jobs to finish
  4. data_test      (LOCAL; YT-trained model cross-rolled on DAVIS test data)
  5. data_plot      (LOCAL; extract params of the YT-trained model)

After each condition finishes, the TeX table is re-emitted so the user can
watch rows fill in one by one.

Prerequisite YAMLs on disk:
    config/fly/<base>_<suffix>_cv<i>.yaml   — YT training YAMLs (5 per base)
    config/fly/<base>.yaml                  — DAVIS base YAML (serves as
                                              test_config for cross-rollout)

The 5 YT fold YAMLs are produced by scripts/write_cross_yt_configs.py
(pass `--n_folds 5`).

Usage:
    python scripts/run_cross_yt_parallel.py \\
        --suffix yt_per_cond \\
        --output_root /groups/saalfeld/home/allierc/GraphData \\
        --emit_tex cv_yt_per_cond_rows.tex \\
        [--n_folds 5] [--node_name a100] [--force_test]
"""

import argparse
import glob
import os
import subprocess
import sys
import time

import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (os.path.join(REPO_ROOT, 'src'), REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
import connectome_gnn.utils as _cg_utils  # noqa: E402
from connectome_gnn.utils import (  # noqa: E402
    add_pre_folder, config_path, log_path, set_device, graphs_data_path,
)
try:
    from connectome_gnn.utils import set_data_root
except ImportError:
    def set_data_root(path):
        _cg_utils._data_root = path
from connectome_gnn.generators.graph_data_generator import data_generate  # noqa: E402
from connectome_gnn.models.graph_trainer import data_test  # noqa: E402
from connectome_gnn.LLM.cluster import (  # noqa: E402
    submit_cluster_job, wait_for_cluster_jobs, CLUSTER_ROOT_DIR,
    submit_cluster_cross_test_plot_job, wait_for_cluster_jobs_with_metrics,
)
from GNN_PlotFigure import data_plot  # noqa: E402


CONDITION_BASES = [
    'flyvis_noise_free',
    'flyvis_noise_005',
    'flyvis_noise_05',
    'flyvis_noise_005_010',
    'flyvis_noise_005_null_edges_pc_400',
    'flyvis_noise_005_removed_pc_20',
    'flyvis_noise_005_stride_5',
    'flyvis_noise_005_hidden_010_ngp',
]


def _load(cfg_name):
    cfg_file, pre = add_pre_folder(cfg_name)
    cfg = NeuralGraphConfig.from_yaml(config_path(f'{cfg_file}.yaml'))
    cfg.dataset = pre + cfg.dataset
    cfg.config_file = pre + cfg_name
    return cfg


def emit_davis_cv_yaml(base_name, fold_i, force=False):
    """Emit config/fly/<base>_cv<i:02d>.yaml — copy of <base>.yaml with
    simulation.seed = 42 + fold_i and dataset = <base>_cv<i:02d>. Returns
    True if the file was written."""
    src = os.path.join(REPO_ROOT, 'config', 'fly', f'{base_name}.yaml')
    dst = os.path.join(REPO_ROOT, 'config', 'fly',
                       f'{base_name}_cv{fold_i:02d}.yaml')
    if os.path.exists(dst) and not force:
        return False
    if not os.path.isfile(src):
        print(f'  [warn] missing DAVIS base yaml {src}')
        return False
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg['simulation'] = dict(cfg.get('simulation', {}))
    cfg['simulation']['seed'] = 42 + fold_i
    cfg['dataset'] = f'{base_name}_cv{fold_i:02d}'
    cfg['description'] = (
        f'DAVIS CV fold {fold_i} of {base_name} '
        f'(sim_seed={42 + fold_i}).'
    )
    with open(dst, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return True


def _have_data(graphs_dir):
    return os.path.isdir(os.path.join(graphs_dir, 'x_list_train'))


def _have_model(log_dir):
    return bool(glob.glob(os.path.join(log_dir, 'models', 'best_model_with_*.pt')))


def _cross_log(log_dir, base_name):
    short = base_name.replace('flyvis_', '').replace('fly/', '')
    return os.path.join(log_dir, f'results_rollout_on_{short}.log')


def _have_plot(log_dir):
    return os.path.exists(os.path.join(log_dir, 'results', 'metrics.txt'))


def ensure_davis_base_data(base_cfg, device):
    """Ensure DAVIS base simulation data exists (test_config for cross-rollout)."""
    base_gdir = graphs_data_path(base_cfg.dataset)
    if _have_data(base_gdir):
        print(f'  [skip] DAVIS base data exists: {base_gdir}')
    else:
        print(f'  [run ] data_generate DAVIS base -> {base_gdir}')
        data_generate(base_cfg, device=device, visualize=False, run_vizualized=0,
                      style='color', alpha=1, erase=True, save=True, step=100)


def ensure_yt_data(yt_cfg, device):
    yt_gdir = graphs_data_path(yt_cfg.dataset)
    if _have_data(yt_gdir):
        print(f'  [skip] YT fold data exists: {yt_gdir}')
    else:
        print(f'  [run ] data_generate YT -> {yt_gdir}')
        data_generate(yt_cfg, device=device, visualize=False, run_vizualized=0,
                      style='color', alpha=1, erase=True, save=True, step=100)


def submit_training_wave(yt_cfgs, output_root, node_name, hard_runtime_limit_min,
                          metrics_interval=300):
    """Submit 5 cluster training bsub jobs (one per fold), wait, return.
    Skips folds whose model is already on disk.

    Uses wait_for_cluster_jobs_with_metrics: every `metrics_interval` seconds
    the training conn/Vr/τ R² for each active slot is read from the slot's
    tmp_training/metrics.log and printed with ANSI color coding.
    """
    job_ids = {}
    log_dirs = {}
    for slot, yt_cfg in enumerate(yt_cfgs):
        yt_log_dir = log_path(yt_cfg.config_file)
        os.makedirs(yt_log_dir, exist_ok=True)
        if _have_model(yt_log_dir):
            print(f'  [skip] fold {slot}: model already trained at {yt_log_dir}/models')
            continue
        # submit_cluster_job expects the absolute CLUSTER-side config path; our
        # config files live at <REPO>/config/fly/*.yaml which is shared FS.
        cfg_basename = os.path.basename(yt_cfg.config_file) + '.yaml'
        cfg_path = f'{CLUSTER_ROOT_DIR}/config/fly/{cfg_basename}'
        analysis_log = f'{yt_log_dir}/cluster_train.log'
        jid = submit_cluster_job(
            slot=slot,
            config_path=cfg_path,
            analysis_log_path=analysis_log,
            config_file_field=yt_cfg.config_file,
            log_dir=yt_log_dir,
            erase=True,
            node_name=node_name,
            output_root=output_root,
            hard_runtime_limit_min=hard_runtime_limit_min,
        )
        if jid is not None:
            job_ids[slot]  = jid
            log_dirs[slot] = yt_log_dir
    if job_ids:
        print(f'  [wait] {len(job_ids)} cluster job(s): {job_ids}  '
              f'(metrics every {metrics_interval}s)')
        wait_for_cluster_jobs_with_metrics(
            job_ids, log_dirs, poll_interval=60,
            metrics_interval=metrics_interval,
            job_prefix='cluster_train',
        )
        # After training, warn on any fold where tau or Vrest collapsed to 0.
        _warn_zero_training_metrics(log_dirs)


def _warn_zero_training_metrics(log_dirs):
    """Read the final line of each slot's tmp_training/metrics.log; print a
    warning if tau_r2 or vrest_r2 rounds to 0.00 (training didn't learn the
    dynamics parameter)."""
    from connectome_gnn.LLM.cluster import _read_latest_training_metrics
    for slot, ld in sorted(log_dirs.items()):
        tm = _read_latest_training_metrics(ld)
        if tm is None:
            continue
        _, conn, vr, tau = tm
        if abs(vr) < 5e-3 or abs(tau) < 5e-3:
            print(f'\033[91m  [WARN] slot {slot}: post-training V_rest_R²={vr:.3f} '
                  f'τ_R²={tau:.3f} — dynamics parameter may have collapsed\033[0m')


def _warn_zero_plot_metrics(yt_log_dir, slot_tag=''):
    """After data_plot, read <yt_log_dir>/results/metrics.txt and warn if
    tau_R2 or V_rest_R2 rounds to 0.00."""
    path = os.path.join(yt_log_dir, 'results', 'metrics.txt')
    if not os.path.isfile(path):
        return
    vals = {}
    try:
        with open(path) as f:
            for line in f:
                if ':' in line:
                    k, v = line.split(':', 1)
                    try:
                        vals[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
    except OSError:
        return
    tau = vals.get('tau_R2')
    vr  = vals.get('V_rest_R2')
    if tau is not None and abs(tau) < 5e-3:
        print(f'\033[91m  [WARN]{slot_tag} post-plot tau_R2={tau:.3f} '
              f'— parameter extraction failed / collapsed\033[0m')
    if vr is not None and abs(vr) < 5e-3:
        print(f'\033[91m  [WARN]{slot_tag} post-plot V_rest_R2={vr:.3f} '
              f'— parameter extraction failed / collapsed\033[0m')


def submit_test_plot_wave(yt_cfgs, base_cfgs, output_root, node_name,
                           hard_runtime_limit_min, force_test,
                           metrics_interval=300):
    """Submit cross-test+plot jobs to the cluster — ONE job per YT fold, each
    runs rollouts against ALL base_cfgs (full N_yt × N_davis Cartesian).

    Cache: submit the job unless every <yt_log_dir>/results_rollout_on_<base_j>.log
    already exists AND metrics.txt is present; --force_test wipes them.
    """
    test_cfg_paths = [
        f'{CLUSTER_ROOT_DIR}/config/fly/' +
        os.path.basename(bc.config_file) + '.yaml'
        for bc in base_cfgs
    ]
    test_cfg_fields = [bc.config_file for bc in base_cfgs]

    job_ids = {}
    log_dirs = {}
    for slot, yt_cfg in enumerate(yt_cfgs):
        yt_log_dir = log_path(yt_cfg.config_file)
        os.makedirs(yt_log_dir, exist_ok=True)

        cross_logs = [
            _cross_log(yt_log_dir, bc.config_file.replace('fly/', ''))
            for bc in base_cfgs
        ]
        metrics_path = os.path.join(yt_log_dir, 'results', 'metrics.txt')
        if force_test:
            for p_ in cross_logs + [metrics_path]:
                if os.path.exists(p_):
                    os.remove(p_)
                    print(f'  [force] removed {p_}')
        all_logs_exist = all(os.path.exists(p_) for p_ in cross_logs)
        if all_logs_exist and os.path.exists(metrics_path):
            print(f'  [skip] fold {slot}: all {len(cross_logs)} rollout logs + metrics.txt already exist')
            continue
        if not _have_model(yt_log_dir):
            print(f'\033[91m  [skip] fold {slot}: no trained model, cannot test\033[0m')
            continue

        cfg_basename = os.path.basename(yt_cfg.config_file) + '.yaml'
        cfg_path = f'{CLUSTER_ROOT_DIR}/config/fly/{cfg_basename}'
        analysis_log = f'{yt_log_dir}/cluster_cross_test_plot.log'

        jid = submit_cluster_cross_test_plot_job(
            slot=slot,
            config_path=cfg_path,
            test_config_paths=test_cfg_paths,
            analysis_log_path=analysis_log,
            config_file_field=yt_cfg.config_file,
            test_config_file_fields=test_cfg_fields,
            log_dir=yt_log_dir,
            node_name=node_name,
            output_root=output_root,
            hard_runtime_limit_min=hard_runtime_limit_min,
            n_rollout_frames=250,
        )
        if jid is not None:
            job_ids[slot]  = jid
            log_dirs[slot] = yt_log_dir
    if job_ids:
        print(f'  [wait] {len(job_ids)} cross test+plot job(s): {job_ids}')
        wait_for_cluster_jobs_with_metrics(
            job_ids, log_dirs, poll_interval=60,
            metrics_interval=metrics_interval,
            job_prefix='cluster_cross_test_plot',
        )
        # Post-plot: warn on zero tau/Vrest R².
        for slot, ld in sorted(log_dirs.items()):
            _warn_zero_plot_metrics(ld, slot_tag=f' slot {slot}:')


def run_test_and_plot(yt_cfg, base_cfgs, device, force_test):
    """Local path: for ONE YT fold, rollout against every DAVIS CV fold
    (Cartesian), then a single data_plot."""
    if not isinstance(base_cfgs, (list, tuple)):
        base_cfgs = [base_cfgs]  # legacy single-DAVIS compatibility.
    yt_log_dir = log_path(yt_cfg.config_file)

    for base_cfg in base_cfgs:
        cross_log = _cross_log(yt_log_dir,
                               base_cfg.config_file.replace('fly/', ''))
        if force_test and os.path.exists(cross_log):
            os.remove(cross_log)
        if os.path.exists(cross_log):
            print(f'  [skip] cross-test log exists: {cross_log}')
            continue
        print(f'  [run ] data_test YT->DAVIS  ({yt_cfg.dataset} -> {base_cfg.dataset})')
        data_test(config=yt_cfg, visualize=False, best_model='best', run=0,
                  step=10, n_rollout_frames=250, device=device,
                  test_config=base_cfg)

    # Param extraction from YT-trained model (once per YT fold).
    metrics_path = os.path.join(yt_log_dir, 'results', 'metrics.txt')
    if force_test and os.path.exists(metrics_path):
        os.remove(metrics_path)
    if _have_plot(yt_log_dir):
        print(f'  [skip] metrics.txt exists')
    else:
        print(f'  [run ] data_plot -> {metrics_path}')
        data_plot(config=yt_cfg, epoch_list=['best'], style='color',
                  extended='plots', device=device,
                  apply_weight_correction=True, skip_svd=True)
    _warn_zero_plot_metrics(yt_log_dir)


def run_condition(base_name, suffix, n_folds, device, output_root,
                  node_name, hard_runtime_limit_min, force_test, emit_tex,
                  cluster_test_plot=False, metrics_interval=300):
    print(f'\n=== {base_name}  (5-fold YT-train / DAVIS-test, suffix={suffix}) ===')

    # 0. Emit per-fold DAVIS CV YAMLs (<base>_cv<i:02d>.yaml) if missing.
    #    Each uses simulation.seed = 42 + i so the 5 test sets are
    #    independent — this is what makes the rollout mean±SD a proper
    #    5-fold CV rather than 5 models on a single test set.
    for i in range(n_folds):
        if emit_davis_cv_yaml(base_name, i, force=False):
            print(f'  [emit] config/fly/{base_name}_cv{i:02d}.yaml')

    # Load 5 DAVIS CV configs (paired one-to-one with YT folds).
    base_cfgs = []
    for i in range(n_folds):
        base_cfg_name = f'{base_name}_cv{i:02d}'
        if not os.path.isfile(config_path('fly', f'{base_cfg_name}.yaml')):
            print(f'  [warn] missing DAVIS CV yaml {base_cfg_name}.yaml')
            continue
        base_cfgs.append(_load(base_cfg_name))

    # 1a. Generate 5 DAVIS test datasets (LOCAL, skipped if data exists).
    for base_cfg in base_cfgs:
        ensure_davis_base_data(base_cfg, device)

    # 1b. Load + generate YT fold data (LOCAL, one per fold).
    yt_cfgs = []
    for i in range(n_folds):
        yt_cfg_name = f'{base_name}_{suffix}_cv{i:02d}'
        yt_yaml = config_path('fly', f'{yt_cfg_name}.yaml')
        if not os.path.isfile(yt_yaml):
            print(f'  [skip] fold {i}: missing YT YAML {yt_yaml}')
            continue
        yt_cfgs.append(_load(yt_cfg_name))
    for yt_cfg in yt_cfgs:
        ensure_yt_data(yt_cfg, device)

    if len(base_cfgs) != len(yt_cfgs):
        print(f'\033[91m  [warn] YT folds ({len(yt_cfgs)}) != '
              f'DAVIS folds ({len(base_cfgs)}); truncating to match\033[0m')
        n = min(len(base_cfgs), len(yt_cfgs))
        base_cfgs = base_cfgs[:n]
        yt_cfgs   = yt_cfgs[:n]

    # 2/3. submit 5 cluster training jobs + wait for the wave
    submit_training_wave(yt_cfgs, output_root, node_name, hard_runtime_limit_min,
                         metrics_interval=metrics_interval)

    # 4/5. cross-test + plot each fold — cluster or local depending on flag.
    if cluster_test_plot:
        submit_test_plot_wave(yt_cfgs, base_cfgs, output_root, node_name,
                              hard_runtime_limit_min, force_test,
                              metrics_interval=metrics_interval)
    else:
        for yt_cfg in yt_cfgs:
            run_test_and_plot(yt_cfg, base_cfgs, device, force_test)

    # 6. emit TeX after this condition is done
    if emit_tex:
        subprocess.run(
            ['python', os.path.join(REPO_ROOT, 'scripts', 'emit_cross_table_rows.py'),
             '--suffix', suffix,
             '--output_tex', emit_tex,
             '--output_root', output_root,
             '--n_folds', str(n_folds)],
            check=False,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suffix', required=True,
                   help='Suffix that the YT yamls were written with.')
    p.add_argument('--output_root',
                   default='/groups/saalfeld/home/allierc/GraphData')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--conditions', nargs='+', default=CONDITION_BASES)
    p.add_argument('--force_test', action='store_true',
                   help='Delete + re-run test/plot (does NOT force re-train).')
    p.add_argument('--emit_tex', default=None,
                   help='Basename of the TeX file emitted after each condition.')
    p.add_argument('--node_name', default='a100')
    p.add_argument('--hard_runtime_limit_min', type=int, default=120)
    args = p.parse_args()

    assert os.path.isdir(args.output_root), f'missing {args.output_root}'
    set_data_root(args.output_root)

    base_config = NeuralGraphConfig.from_yaml(
        config_path('fly', f'{args.conditions[0]}.yaml'))
    device = set_device(base_config.training.device)

    print(f'Cross YT parallel runner — suffix={args.suffix}  '
          f'n_folds={args.n_folds}  node={args.node_name}  '
          f'force_test={args.force_test}  emit_tex={args.emit_tex or "<off>"}')

    for base_name in args.conditions:
        run_condition(
            base_name, args.suffix, args.n_folds, device, args.output_root,
            args.node_name, args.hard_runtime_limit_min, args.force_test,
            args.emit_tex,
        )

    print('\nCross YT parallel runner complete.')


if __name__ == '__main__':
    main()
