"""
Per-condition orchestration for the cross-check pipeline.

Exposes `run_condition(...)` which for one base condition:
    1. emits DAVIS CV YAMLs (if missing)
    2. generates DAVIS CV data (local)
    3. generates YT CV data (local)
    4. submits 5 cluster training jobs, waits for the wave
    5. submits 5 cluster cross-test+plot jobs (or runs locally)
    6. warns if V_rest / tau R² collapsed

Iterate over `CONDITION_BASES` to cover all 8 conditions.
"""

import glob
import os

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.utils import (
    add_pre_folder, log_path, graphs_data_path,
)
from connectome_gnn.generators.graph_data_generator import data_generate
from connectome_gnn.models.graph_trainer import data_test
from connectome_gnn.LLM.cluster import (
    submit_cluster_job, submit_cluster_cross_test_plot_job,
    wait_for_cluster_jobs_with_metrics,
)
from connectome_gnn.cross.yaml_io import (
    cv_config_dir, shared_cv_yaml_path, emit_davis_cv_yaml, _load_yaml_either,
)


CONDITION_BASES = [
    # 'flyvis_noise_free',
    # 'flyvis_noise_005',
    # 'flyvis_noise_05',
    # 'flyvis_noise_005_010',
    'flyvis_noise_005_null_edges_pc_400',
    'flyvis_noise_005_removed_pc_20',
    'flyvis_noise_005_stride_5',
    'flyvis_noise_005_hidden_010_ngp',
]


def _load(cfg_name, output_root):
    yaml_path = _load_yaml_either(cfg_name, output_root)
    cfg = NeuralGraphConfig.from_yaml(yaml_path)
    _, pre = add_pre_folder(cfg_name)
    if not cfg.dataset.startswith(pre):
        cfg.dataset = pre + cfg.dataset
    if not cfg.config_file.startswith(pre):
        cfg.config_file = pre + cfg_name
    return cfg


def _have_data(graphs_dir):
    """Require x_list_train/ + generation_log.txt + matching zarr frame count."""
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
        return True
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


def _have_plot(log_dir):
    return os.path.exists(os.path.join(log_dir, 'results', 'metrics.txt'))


def _cross_log(log_dir, base_name):
    short = base_name.replace('flyvis_', '').replace('fly/', '')
    return os.path.join(log_dir, f'results_rollout_on_{short}.log')


def ensure_davis_base_data(base_cfg, device):
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


def _warn_zero_training_metrics(log_dirs):
    from connectome_gnn.LLM.cluster import _read_latest_training_metrics
    for slot, ld in sorted(log_dirs.items()):
        tm = _read_latest_training_metrics(ld)
        if tm is None:
            continue
        _, _, vr, tau = tm
        if abs(vr) < 5e-3 or abs(tau) < 5e-3:
            print(f'\033[91m  [WARN] slot {slot}: post-training V_rest_R²={vr:.3f} '
                  f'τ_R²={tau:.3f} — dynamics parameter may have collapsed\033[0m')


def _warn_zero_plot_metrics(yt_log_dir, slot_tag=''):
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
        print(f'\033[91m  [WARN]{slot_tag} post-plot tau_R2={tau:.3f}\033[0m')
    if vr is not None and abs(vr) < 5e-3:
        print(f'\033[91m  [WARN]{slot_tag} post-plot V_rest_R2={vr:.3f}\033[0m')


def submit_training_wave(yt_cfgs, output_root, node_name, hard_runtime_limit_min,
                          metrics_interval=300):
    job_ids = {}
    log_dirs = {}
    for slot, yt_cfg in enumerate(yt_cfgs):
        yt_log_dir = log_path(yt_cfg.config_file)
        os.makedirs(yt_log_dir, exist_ok=True)
        if _have_model(yt_log_dir):
            print(f'  [skip] fold {slot}: model already trained at {yt_log_dir}/models')
            continue
        cfg_path = shared_cv_yaml_path(yt_cfg.config_file, output_root)
        analysis_log = f'{yt_log_dir}/cluster_train.log'
        jid = submit_cluster_job(
            slot=slot, config_path=cfg_path,
            analysis_log_path=analysis_log,
            config_file_field=yt_cfg.config_file,
            log_dir=yt_log_dir, erase=True, node_name=node_name,
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
        _warn_zero_training_metrics(log_dirs)


def submit_test_plot_wave(yt_cfgs, base_cfgs, output_root, node_name,
                           hard_runtime_limit_min, force_test,
                           metrics_interval=300):
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
        cfg_path = shared_cv_yaml_path(yt_cfg.config_file, output_root)
        test_cfg_path = shared_cv_yaml_path(base_cfg.config_file, output_root)
        jid = submit_cluster_cross_test_plot_job(
            slot=slot, config_path=cfg_path,
            test_config_paths=[test_cfg_path],
            analysis_log_path=f'{yt_log_dir}/cluster_cross_test_plot.log',
            config_file_field=yt_cfg.config_file,
            test_config_file_fields=[base_cfg.config_file],
            log_dir=yt_log_dir, node_name=node_name, output_root=output_root,
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
        for slot, ld in sorted(log_dirs.items()):
            _warn_zero_plot_metrics(ld, slot_tag=f' slot {slot}:')


def run_test_and_plot_local(yt_cfg, base_cfg, device, force_test):
    """Local fallback: rollout YT fold against its paired DAVIS fold,
    then one data_plot."""
    from GNN_PlotFigure import data_plot
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
                  node_name, hard_runtime_limit_min, force_test,
                  cluster_test_plot=True, metrics_interval=300):
    print(f'\n=== {base_name}  ({n_folds}-fold YT-train / DAVIS-test, suffix={suffix}) ===')

    # 0. Emit per-fold DAVIS CV YAMLs.
    for i in range(n_folds):
        if emit_davis_cv_yaml(base_name, i, output_root, force=False):
            print(f'  [emit] {cv_config_dir(output_root)}/{base_name}_cv{i:02d}.yaml')

    # Load DAVIS CV configs.
    base_cfgs = []
    for i in range(n_folds):
        shared = os.path.join(cv_config_dir(output_root),
                              f'{base_name}_cv{i:02d}.yaml')
        if not os.path.isfile(shared):
            print(f'  [warn] missing DAVIS CV yaml {shared}')
            continue
        base_cfgs.append(_load(f'{base_name}_cv{i:02d}', output_root))

    # 1a. Generate DAVIS CV data.
    for base_cfg in base_cfgs:
        ensure_davis_base_data(base_cfg, device)

    # 1b. Load + generate YT CV data.
    yt_cfgs = []
    for i in range(n_folds):
        yt_cfg_name = f'{base_name}_{suffix}_cv{i:02d}'
        yt_yaml = _load_yaml_either(yt_cfg_name, output_root)
        if not os.path.isfile(yt_yaml):
            print(f'  [skip] fold {i}: missing YT YAML {yt_yaml}')
            continue
        yt_cfgs.append(_load(yt_cfg_name, output_root))
    for yt_cfg in yt_cfgs:
        ensure_yt_data(yt_cfg, device)

    if len(base_cfgs) != len(yt_cfgs):
        print(f'\033[91m  [warn] YT folds ({len(yt_cfgs)}) != '
              f'DAVIS folds ({len(base_cfgs)}); truncating\033[0m')
        n = min(len(base_cfgs), len(yt_cfgs))
        base_cfgs = base_cfgs[:n]
        yt_cfgs   = yt_cfgs[:n]

    # 2/3. Cluster training wave.
    submit_training_wave(yt_cfgs, output_root, node_name,
                         hard_runtime_limit_min,
                         metrics_interval=metrics_interval)

    # 4/5. Cluster test+plot wave (or local).
    if cluster_test_plot:
        submit_test_plot_wave(yt_cfgs, base_cfgs, output_root, node_name,
                              hard_runtime_limit_min, force_test,
                              metrics_interval=metrics_interval)
    else:
        for yt_cfg, base_cfg in zip(yt_cfgs, base_cfgs):
            run_test_and_plot_local(yt_cfg, base_cfg, device, force_test)
