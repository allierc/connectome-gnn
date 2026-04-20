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


def cv_config_dir(output_root):
    """CV configs (both YT fold YAMLs and DAVIS fold YAMLs) are emitted to
    <output_root>/config/fly/ — a shared-FS location visible from both the
    devcontainer AND the cluster. The repo's own config/fly/ is reserved
    for static, hand-written base configs that are version-controlled."""
    return os.path.join(output_root, 'config', 'fly')


def _load_yaml_either(cfg_name, output_root):
    """Load a config YAML: prefer <output_root>/config/fly/, fall back to
    the repo's config/fly/ for static base YAMLs. Returns absolute path."""
    shared = os.path.join(cv_config_dir(output_root), f'{cfg_name}.yaml')
    if os.path.isfile(shared):
        return shared
    repo_path = config_path('fly', f'{cfg_name}.yaml')
    return repo_path


def _load(cfg_name, output_root=None):
    if output_root is not None:
        yaml_path = _load_yaml_either(cfg_name, output_root)
        cfg = NeuralGraphConfig.from_yaml(yaml_path)
        _, pre = add_pre_folder(cfg_name)
    else:
        cfg_file, pre = add_pre_folder(cfg_name)
        cfg = NeuralGraphConfig.from_yaml(config_path(f'{cfg_file}.yaml'))
    # Guard against double-prefixing: CV YAMLs (both YT and DAVIS) now
    # bake `fly/` into cfg.dataset so the cluster's train_subprocess.py
    # reads it directly; static base YAMLs still have a bare dataset name
    # and need the prefix added here.
    if not cfg.dataset.startswith(pre):
        cfg.dataset = pre + cfg.dataset
    if not cfg.config_file.startswith(pre):
        cfg.config_file = pre + cfg_name
    return cfg


def emit_davis_cv_yaml(base_name, fold_i, output_root, force=False):
    """Emit <output_root>/config/fly/<base>_cv<i:02d>.yaml — copy of
    <repo>/config/fly/<base>.yaml with simulation.seed = 42 + fold_i
    and dataset = <base>_cv<i:02d>. Placed on shared FS so the cluster
    can read it directly. Returns True if the file was written."""
    src = os.path.join(REPO_ROOT, 'config', 'fly', f'{base_name}.yaml')
    out_dir = cv_config_dir(output_root)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f'{base_name}_cv{fold_i:02d}.yaml')
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
    """Data is considered complete iff:
    - <graphs_dir>/x_list_train/ exists (zarr v3 layout written by
      ZarrSimulationWriterV3).
    - <graphs_dir>/generation_log.txt exists (written AFTER finalize;
      its absence means the previous run was interrupted mid-generation).
    - If generation_log.txt records `n_frames_train: N`, the voltage.zarr
      shape[0] matches N (verifies the zarr wasn't truncated).
    """
    if not os.path.isdir(os.path.join(graphs_dir, 'x_list_train')):
        return False
    gen_log = os.path.join(graphs_dir, 'generation_log.txt')
    if not os.path.isfile(gen_log):
        print(f'\033[93m  [incomplete] {graphs_dir}: missing generation_log.txt '
              f'-- will regenerate\033[0m')
        return False
    expected = None
    try:
        with open(gen_log) as f:
            for line in f:
                if line.startswith('n_frames_train:'):
                    expected = int(line.split(':', 1)[1].strip())
                    break
    except (OSError, ValueError):
        return False
    if expected is None:
        return True  # old-format log without count; trust x_list_train
    try:
        import zarr
        v = zarr.open(os.path.join(graphs_dir, 'x_list_train', 'voltage.zarr'),
                      mode='r')
        actual = int(v.shape[0])
    except Exception as _e:
        print(f'\033[93m  [incomplete] {graphs_dir}: cannot read voltage.zarr '
              f'({_e.__class__.__name__}) -- will regenerate\033[0m')
        return False
    if actual != expected:
        print(f'\033[93m  [incomplete] {graphs_dir}: voltage.zarr has '
              f'{actual} frames, expected {expected} -- will regenerate\033[0m')
        return False
    return True


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


def _shared_cv_yaml_path(config_file, output_root):
    """Build the shared-FS absolute path for a CV YAML:
       <output_root>/config/fly/<basename>.yaml"""
    basename = os.path.basename(config_file) + '.yaml'
    return os.path.join(cv_config_dir(output_root), basename)



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
        # CV YAMLs live on the shared FS under <output_root>/config/fly/
        # so both devcontainer and cluster read the same file.
        cfg_path = _shared_cv_yaml_path(yt_cfg.config_file, output_root)
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
            job_ids, log_dirs, poll_interval=metrics_interval,
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
    """Submit cross-test+plot jobs to the cluster — ONE job per YT fold,
    PAIRED with DAVIS fold of the same index. 5 rollouts total (not 25).

    Cache: submit the job unless <yt_log_dir>/results_rollout_on_<base_i>.log
    AND metrics.txt both exist; --force_test wipes them.
    """
    assert len(yt_cfgs) == len(base_cfgs), (
        f'YT folds ({len(yt_cfgs)}) and DAVIS folds ({len(base_cfgs)}) differ'
    )

    job_ids = {}
    log_dirs = {}
    for slot, (yt_cfg, base_cfg) in enumerate(zip(yt_cfgs, base_cfgs)):
        yt_log_dir = log_path(yt_cfg.config_file)
        os.makedirs(yt_log_dir, exist_ok=True)

        cross_log = _cross_log(yt_log_dir,
                               base_cfg.config_file.replace('fly/', ''))
        metrics_path = os.path.join(yt_log_dir, 'results', 'metrics.txt')
        if force_test:
            for p_ in (cross_log, metrics_path):
                if os.path.exists(p_):
                    os.remove(p_)
                    print(f'  [force] removed {p_}')
        if os.path.exists(cross_log) and os.path.exists(metrics_path):
            print(f'  [skip] fold {slot}: rollout log + metrics.txt already exist')
            continue
        if not _have_model(yt_log_dir):
            print(f'\033[91m  [skip] fold {slot}: no trained model, cannot test\033[0m')
            continue

        cfg_path = _shared_cv_yaml_path(yt_cfg.config_file, output_root)
        test_cfg_path = _shared_cv_yaml_path(base_cfg.config_file, output_root)
        analysis_log = f'{yt_log_dir}/cluster_cross_test_plot.log'

        jid = submit_cluster_cross_test_plot_job(
            slot=slot,
            config_path=cfg_path,
            test_config_paths=[test_cfg_path],
            analysis_log_path=analysis_log,
            config_file_field=yt_cfg.config_file,
            test_config_file_fields=[base_cfg.config_file],
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
            job_ids, log_dirs, poll_interval=metrics_interval,
            metrics_interval=metrics_interval,
            job_prefix='cluster_cross_test_plot',
        )
        # Post-plot: warn on zero tau/Vrest R².
        for slot, ld in sorted(log_dirs.items()):
            _warn_zero_plot_metrics(ld, slot_tag=f' slot {slot}:')


def run_test_and_plot(yt_cfg, base_cfg, device, force_test):
    """Local path: for ONE YT fold, rollout against its PAIRED DAVIS fold,
    then a single data_plot."""
    yt_log_dir = log_path(yt_cfg.config_file)

    cross_log = _cross_log(yt_log_dir,
                           base_cfg.config_file.replace('fly/', ''))
    if force_test and os.path.exists(cross_log):
        os.remove(cross_log)
    if os.path.exists(cross_log):
        print(f'  [skip] cross-test log exists: {cross_log}')
    else:
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

    # 0. Emit per-fold DAVIS CV YAMLs (<base>_cv<i:02d>.yaml) into the
    #    shared-FS CV config dir (<output_root>/config/fly/). Each uses
    #    simulation.seed = 42 + i so the 5 test sets are independent.
    for i in range(n_folds):
        if emit_davis_cv_yaml(base_name, i, output_root, force=False):
            print(f'  [emit] {cv_config_dir(output_root)}/{base_name}_cv{i:02d}.yaml')

    # Load 5 DAVIS CV configs (paired one-to-one with YT folds).
    base_cfgs = []
    for i in range(n_folds):
        base_cfg_name = f'{base_name}_cv{i:02d}'
        shared = os.path.join(cv_config_dir(output_root),
                              f'{base_cfg_name}.yaml')
        if not os.path.isfile(shared):
            print(f'  [warn] missing DAVIS CV yaml {shared}')
            continue
        base_cfgs.append(_load(base_cfg_name, output_root=output_root))

    # 1a. Generate 5 DAVIS test datasets (LOCAL, skipped if data exists).
    for base_cfg in base_cfgs:
        ensure_davis_base_data(base_cfg, device)

    # 1b. Load + generate YT fold data (LOCAL, one per fold). YT YAMLs are
    #     expected on the shared FS (<output_root>/config/fly/) — emitted by
    #     run_GNN_conditions.py / run_GNN_cross.py before this runs.
    yt_cfgs = []
    for i in range(n_folds):
        yt_cfg_name = f'{base_name}_{suffix}_cv{i:02d}'
        yt_yaml = _load_yaml_either(yt_cfg_name, output_root)
        if not os.path.isfile(yt_yaml):
            print(f'  [skip] fold {i}: missing YT YAML {yt_yaml}')
            continue
        yt_cfgs.append(_load(yt_cfg_name, output_root=output_root))
    for yt_cfg in yt_cfgs:
        ensure_yt_data(yt_cfg, device)

    if len(base_cfgs) != len(yt_cfgs):
        print(f'\033[91m  [warn] YT folds ({len(yt_cfgs)}) != '
              f'DAVIS folds ({len(base_cfgs)}); truncating to match\033[0m')
        n = min(len(base_cfgs), len(yt_cfgs))
        base_cfgs = base_cfgs[:n]
        yt_cfgs   = yt_cfgs[:n]

    # 2/3. submit 5 cluster training jobs + wait for the wave.
    submit_training_wave(yt_cfgs, output_root, node_name, hard_runtime_limit_min,
                         metrics_interval=metrics_interval)

    # 4/5. cross-test + plot each fold — cluster or local depending on flag.
    if cluster_test_plot:
        submit_test_plot_wave(yt_cfgs, base_cfgs, output_root, node_name,
                              hard_runtime_limit_min, force_test,
                              metrics_interval=metrics_interval)
    else:
        for yt_cfg, base_cfg in zip(yt_cfgs, base_cfgs):
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
