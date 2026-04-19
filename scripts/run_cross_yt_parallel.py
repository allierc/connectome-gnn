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


def submit_training_wave(yt_cfgs, output_root, node_name, hard_runtime_limit_min):
    """Submit 5 cluster training bsub jobs (one per fold), wait, return.
    Skips folds whose model is already on disk."""
    job_ids = {}
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
            job_ids[slot] = jid
    if job_ids:
        print(f'  [wait] {len(job_ids)} cluster job(s): {job_ids}')
        wait_for_cluster_jobs(job_ids, log_dir=None, poll_interval=60,
                              job_prefix='cluster_train')


def run_test_and_plot(yt_cfg, base_cfg, device, force_test):
    yt_log_dir = log_path(yt_cfg.config_file)
    # Cross-test (YT-trained -> DAVIS held-out).
    cross_log = _cross_log(yt_log_dir, base_cfg.config_file.replace('fly/', ''))
    if force_test and os.path.exists(cross_log):
        os.remove(cross_log)
    if os.path.exists(cross_log):
        print(f'  [skip] cross-test log exists: {cross_log}')
    else:
        print(f'  [run ] data_test YT->DAVIS  ({yt_cfg.dataset} -> {base_cfg.dataset})')
        data_test(config=yt_cfg, visualize=False, best_model='best', run=0,
                  step=10, n_rollout_frames=250, device=device,
                  test_config=base_cfg)
    # Param extraction from YT-trained model.
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


def run_condition(base_name, suffix, n_folds, device, output_root,
                  node_name, hard_runtime_limit_min, force_test, emit_tex):
    print(f'\n=== {base_name}  (5-fold YT-train / DAVIS-test, suffix={suffix}) ===')

    base_cfg = _load(base_name)
    ensure_davis_base_data(base_cfg, device)

    yt_cfgs = []
    for i in range(n_folds):
        yt_cfg_name = f'{base_name}_{suffix}_cv{i:02d}'
        yt_yaml = config_path('fly', f'{yt_cfg_name}.yaml')
        if not os.path.isfile(yt_yaml):
            print(f'  [skip] fold {i}: missing YT YAML {yt_yaml}')
            continue
        yt_cfg = _load(yt_cfg_name)
        yt_cfgs.append(yt_cfg)

    # 1. generate YT fold data (LOCAL, one by one — typically cheap)
    for yt_cfg in yt_cfgs:
        ensure_yt_data(yt_cfg, device)

    # 2/3. submit 5 cluster training jobs + wait for the wave
    submit_training_wave(yt_cfgs, output_root, node_name, hard_runtime_limit_min)

    # 4/5. cross-test + plot each fold LOCALLY
    for yt_cfg in yt_cfgs:
        run_test_and_plot(yt_cfg, base_cfg, device, force_test)

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
