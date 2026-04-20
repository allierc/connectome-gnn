"""
Per-condition orchestration for the YT-only cross-check pipeline.

Exposes `run_condition(...)` which for one base condition:
    1. emits YT CV YAMLs (done upstream in runner.emit_yt_yamls)
    2. generates YT CV data (local) — noop if already present
    3. submits n_folds cluster training jobs, waits for the wave
    4. submits n_folds cluster test+plot jobs (rollout on held-out 20% of
       the same YT fold) or runs locally
    5. warns if V_rest / tau R² collapsed

Also exposes `generate_yt_data_for_condition(...)`, a generate-only entry
point used by `run_generate_YT_data.py` so the three training-runner
scripts (run_GNN_conditions / run_GNN_cross / run_KnownODE_conditions)
can be launched in parallel on the pre-built shared datasets.
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
    shared_cv_yaml_path, _load_yaml_either,
)


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


def ensure_yt_data(yt_cfg, device):
    yt_gdir = graphs_data_path(yt_cfg.dataset)
    if _have_data(yt_gdir):
        print(f'  [skip] YT fold data exists: {yt_gdir}')
    else:
        print(f'  [run ] data_generate YT -> {yt_gdir}')
        # visualize=False: no activity/trace figures or mp4 videos.
        # compute_ranks=False: skip SVD + the kinograph.png that it drives.
        data_generate(yt_cfg, device=device, visualize=False, run_vizualized=0,
                      style='color', alpha=1, erase=True, save=True, step=100,
                      compute_ranks=False)


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


def submit_test_plot_wave(yt_cfgs, output_root, node_name,
                           hard_runtime_limit_min, force_test,
                           metrics_interval=300):
    """Rollout each YT-trained model on the held-out 20% of its own YT fold.

    Since test_config == training config, graph_tester sets test_suffix=''
    so the rollout log lands at <log_dir>/results_rollout.log (no _on_X suffix).
    """
    job_ids = {}
    log_dirs = {}
    for slot, yt_cfg in enumerate(yt_cfgs):
        yt_log_dir = log_path(yt_cfg.config_file)
        os.makedirs(yt_log_dir, exist_ok=True)
        rollout_log = os.path.join(yt_log_dir, 'results_rollout.log')
        metrics_path = os.path.join(yt_log_dir, 'results', 'metrics.txt')
        if force_test:
            for p_ in (rollout_log, metrics_path):
                if os.path.exists(p_):
                    os.remove(p_)
                    print(f'  [force] removed {p_}')
        if os.path.exists(rollout_log) and os.path.exists(metrics_path):
            print(f'  [skip] fold {slot}: rollout log + metrics.txt already exist')
            continue
        if not _have_model(yt_log_dir):
            print(f'\033[91m  [skip] fold {slot}: no trained model, cannot test\033[0m')
            continue
        cfg_path = shared_cv_yaml_path(yt_cfg.config_file, output_root)
        jid = submit_cluster_cross_test_plot_job(
            slot=slot, config_path=cfg_path,
            test_config_paths=[cfg_path],
            analysis_log_path=f'{yt_log_dir}/cluster_cross_test_plot.log',
            config_file_field=yt_cfg.config_file,
            test_config_file_fields=[yt_cfg.config_file],
            log_dir=yt_log_dir, node_name=node_name, output_root=output_root,
            hard_runtime_limit_min=hard_runtime_limit_min,
            n_rollout_frames=250,
        )
        if jid is not None:
            job_ids[slot]  = jid
            log_dirs[slot] = yt_log_dir
    if job_ids:
        print(f'  [wait] {len(job_ids)} test+plot job(s): {job_ids}')
        wait_for_cluster_jobs_with_metrics(
            job_ids, log_dirs, poll_interval=metrics_interval,
            metrics_interval=metrics_interval,
            job_prefix='cluster_cross_test_plot',
        )
        for slot, ld in sorted(log_dirs.items()):
            _warn_zero_plot_metrics(ld, slot_tag=f' slot {slot}:')


def run_test_and_plot_local(yt_cfg, device, force_test):
    """Local fallback: rollout YT fold on its own held-out 20%, then data_plot."""
    from GNN_PlotFigure import data_plot
    yt_log_dir = log_path(yt_cfg.config_file)
    rollout_log = os.path.join(yt_log_dir, 'results_rollout.log')
    if force_test and os.path.exists(rollout_log):
        os.remove(rollout_log)
    if os.path.exists(rollout_log):
        print(f'  [skip] rollout log exists: {rollout_log}')
    else:
        print(f'  [run ] data_test YT held-out  (dataset: {yt_cfg.dataset})')
        data_test(config=yt_cfg, visualize=False, best_model='best', run=0,
                  step=10, n_rollout_frames=250, device=device)
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


def _load_yt_cfgs(base_name, suffix, n_folds, output_root):
    yt_cfgs = []
    for i in range(n_folds):
        yt_cfg_name = f'{base_name}_{suffix}_cv{i:02d}'
        yt_yaml = _load_yaml_either(yt_cfg_name, output_root)
        if not os.path.isfile(yt_yaml):
            print(f'  [skip] fold {i}: missing YT YAML {yt_yaml}')
            continue
        yt_cfgs.append(_load(yt_cfg_name, output_root))
    return yt_cfgs


def run_condition(base_name, suffix, n_folds, device, output_root,
                  node_name, hard_runtime_limit_min, force_test,
                  cluster_test_plot=True, metrics_interval=300):
    """Train + test + plot one condition on YouTube-VOS (no DAVIS)."""
    print(f'\n=== {base_name}  ({n_folds}-fold YT-train / YT-test, suffix={suffix}) ===')

    yt_cfgs = _load_yt_cfgs(base_name, suffix, n_folds, output_root)
    for yt_cfg in yt_cfgs:
        ensure_yt_data(yt_cfg, device)

    submit_training_wave(yt_cfgs, output_root, node_name,
                         hard_runtime_limit_min,
                         metrics_interval=metrics_interval)

    if cluster_test_plot:
        submit_test_plot_wave(yt_cfgs, output_root, node_name,
                              hard_runtime_limit_min, force_test,
                              metrics_interval=metrics_interval)
    else:
        for yt_cfg in yt_cfgs:
            run_test_and_plot_local(yt_cfg, device, force_test)


def generate_yt_data_for_condition(base_name, suffix, n_folds, device, output_root):
    """Generate-only entry point: produces YT datasets for n_folds of one base
    condition and returns. Used by run_generate_YT_data.py to pre-build the
    shared datasets before the 3 parallel training runners."""
    print(f'\n=== {base_name}  (generate-only, n_folds={n_folds}, suffix={suffix}) ===')
    yt_cfgs = _load_yt_cfgs(base_name, suffix, n_folds, output_root)
    for yt_cfg in yt_cfgs:
        ensure_yt_data(yt_cfg, device)
