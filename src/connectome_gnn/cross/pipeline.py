"""
Per-condition orchestration for the hold-out-only cross-check pipeline.

Exposes `run_condition(...)` which for one base condition:
    1. emits hold-out CV YAMLs (done upstream in runner.emit_yt_yamls)
    2. generates hold-out CV data (local) — noop if already present
    3. submits n_folds cluster training jobs, waits for the wave
    4. submits n_folds cluster test+plot jobs (rollout on held-out 20% of
       the same hold-out fold) or runs locally
    5. warns if V_rest / tau R² collapsed

Also exposes `generate_yt_data_for_condition(...)`, a generate-only entry
point used by `run_generate_holdout_data.py` so the three training-runner
scripts (run_GNN_conditions / run_GNN_unique / run_KnownODE_conditions)
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
    wait_for_cluster_jobs, wait_for_cluster_jobs_with_metrics,
    _r2_color, _ANSI_RESET,
)
from connectome_gnn.cross.yaml_io import (
    shared_cv_yaml_path, _load_yaml_either,
)


CONDITION_BASES = [
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
    # AR(1) measurement-noise sweep (must match CONDITIONS in cross/yaml_io.py).
    'flyvis_noise_005_010_blank50_ar1_rho25',
    'flyvis_noise_005_010_blank50_ar1_rho50',
    'flyvis_noise_005_010_blank50_ar1_rho75',
    'flyvis_noise_005_010_blank50_ar1_rho90',
    'flyvis_noise_005_010_blank50_ar1_rho95',
    'flyvis_noise_005_010_blank50_ar1_rho99',
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
        print(f'  [skip] hold-out fold data exists: {yt_gdir}')
    else:
        print(f'  [run ] data_generate hold-out -> {yt_gdir}')
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
        vr, tau = tm['vr'], tm['tau']
        if abs(vr) < 5e-3 or abs(tau) < 5e-3:
            print(f'\033[91m  [WARN] slot {slot}: post-training V_rest_R²={vr:.3f} '
                  f'τ_R²={tau:.3f} — dynamics parameter may have collapsed\033[0m')


def _read_plot_metrics(yt_log_dir):
    """Parse <yt_log_dir>/results/metrics.txt into a {key: float} dict.
    Returns an empty dict if the file is missing or unreadable."""
    path = os.path.join(yt_log_dir, 'results', 'metrics.txt')
    vals = {}
    if not os.path.isfile(path):
        return vals
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
        pass
    return vals


def _warn_zero_plot_metrics(yt_log_dir, slot_tag=''):
    vals = _read_plot_metrics(yt_log_dir)
    tau = vals.get('tau_R2')
    vr  = vals.get('V_rest_R2')
    if tau is not None and abs(tau) < 5e-3:
        print(f'\033[91m  [WARN]{slot_tag} post-plot tau_R2={tau:.3f}\033[0m')
    if vr is not None and abs(vr) < 5e-3:
        print(f'\033[91m  [WARN]{slot_tag} post-plot V_rest_R2={vr:.3f}\033[0m')


def print_plot_metrics_summary(yt_log_dir, slot=None, prefix='  [plot   ]',
                                n_neurons=13741):
    """Print colored one-line summary parsed from results/metrics.txt — uses
    the same column layout as the [metrics]/[final ] training prints so the
    terminal stream is consistent across all runners:

        [plot   ] slot S  R²W=0.99  R²Vr=0.94(3.3%)  R²τ=0.99(0.0%)  cluster=0.92

    R²W   = W_corrected_R2 (nominal)
    R²Vr  = V_rest_no_outliers_R2  (out% = V_rest_n_outliers / n_neurons)
    R²τ   = tau_no_outliers_R2     (out% = tau_n_outliers     / n_neurons)
    cluster = clustering_accuracy
    """
    vals = _read_plot_metrics(yt_log_dir)
    if not vals:
        slot_str = f' slot {slot}' if slot is not None else ''
        print(f'{prefix}{slot_str}: (no metrics.txt yet)')
        return

    def _c(val, thresholds=(0.9, 0.7, 0.3)):
        if val is None:
            return f'{_ANSI_RESET}n/a{_ANSI_RESET}'
        return f'{_r2_color(val, thresholds)}{val:.2f}{_ANSI_RESET}'

    def _fmt_R2_out(r2_no, n_out):
        if r2_no is None:
            return f'{_ANSI_RESET}n/a{_ANSI_RESET}'
        col = _r2_color(r2_no)
        if n_out is None or not n_neurons:
            return f'{col}{r2_no:.2f}{_ANSI_RESET}'
        pct = 100.0 * n_out / n_neurons
        return f'{col}{r2_no:.2f}({pct:.1f}%){_ANSI_RESET}'

    w_r2  = vals.get('W_corrected_R2')
    tau_n   = vals.get('tau_n_outliers')
    tau_r2  = vals.get('tau_no_outliers_R2')
    vr_n    = vals.get('V_rest_n_outliers')
    vr_r2   = vals.get('V_rest_no_outliers_R2')
    cl      = vals.get('clustering_accuracy')

    slot_str = f' slot {slot}' if slot is not None else ''
    print(
        f'{prefix}{slot_str}  '
        f'R²W={_c(w_r2)}  '
        f'R²Vr={_fmt_R2_out(vr_r2, vr_n)}  '
        f'R²τ={_fmt_R2_out(tau_r2, tau_n)}  '
        f'cluster={_c(cl)}'
    )


def submit_training_wave(yt_cfgs, output_root, node_name, hard_runtime_limit_min,
                          metrics_interval=300, force_train=False):
    job_ids = {}
    log_dirs = {}
    for slot, yt_cfg in enumerate(yt_cfgs):
        yt_log_dir = log_path(yt_cfg.config_file)
        os.makedirs(yt_log_dir, exist_ok=True)
        if force_train:
            # Mirror force_test pattern (line ~211): remove existing
            # checkpoints + downstream artefacts so the cluster job re-trains
            # from scratch rather than hitting the [skip] guard.
            for sub in ('models', 'results', 'tmp_training'):
                p_ = os.path.join(yt_log_dir, sub)
                if os.path.isdir(p_):
                    import shutil
                    shutil.rmtree(p_)
                    print(f'  [force] removed {p_}')
            for fname in ('results_rollout.log', 'cluster_train.log',
                          'cluster_cross_test_plot.log'):
                p_ = os.path.join(yt_log_dir, fname)
                if os.path.exists(p_):
                    os.remove(p_)
                    print(f'  [force] removed {p_}')
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
                           force_plot=False,
                           metrics_interval=300):
    """Rollout each hold-out-trained model on the held-out 20% of its own fold.

    Since test_config == training config, graph_tester sets test_suffix=''
    so the rollout log lands at <log_dir>/results_rollout.log (no _on_X suffix).

    Per-phase forcing:
      force_test=True  → remove results_rollout.log and re-run data_test
      force_plot=True  → remove results/metrics.txt and re-run data_plot
      both             → re-run everything (legacy --force-test behaviour)
    The cluster job is told to --skip-test / --skip-plot for whichever phase
    the artefacts already cover, so a plot-only re-run does not redo the
    multi-minute rollout (and vice versa).
    """
    job_ids = {}
    log_dirs = {}
    for slot, yt_cfg in enumerate(yt_cfgs):
        yt_log_dir = log_path(yt_cfg.config_file)
        os.makedirs(yt_log_dir, exist_ok=True)
        rollout_log = os.path.join(yt_log_dir, 'results_rollout.log')
        metrics_path = os.path.join(yt_log_dir, 'results', 'metrics.txt')
        if force_test and os.path.exists(rollout_log):
            os.remove(rollout_log)
            print(f'  [force-test] removed {rollout_log}')
        if force_plot and os.path.exists(metrics_path):
            os.remove(metrics_path)
        need_test = force_test or not os.path.exists(rollout_log)
        need_plot = force_plot or not os.path.exists(metrics_path)
        if not need_test and not need_plot:
            print(f'  [skip] fold {slot}: rollout log + metrics.txt already exist')
            continue
        if need_test and not _have_model(yt_log_dir):
            print(f'\033[91m  [skip] fold {slot}: no trained model, cannot test\033[0m')
            continue
        skip_test = not need_test
        skip_plot = not need_plot
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
            skip_test=skip_test, skip_plot=skip_plot,
        )
        if jid is not None:
            job_ids[slot]  = jid
            log_dirs[slot] = yt_log_dir
    if job_ids:
        print(f'  [wait] {len(job_ids)} test+plot job(s): {job_ids}')
        # Use the plain (no-metrics) wait so we don't keep re-printing the
        # stale tmp_training/metrics.log line during a force-plot wait
        # (training is already done; only the plot job runs here). The
        # post-plot summary below is the only post-job line we want.
        wait_for_cluster_jobs(
            job_ids, poll_interval=metrics_interval,
            job_prefix='cluster_cross_test_plot',
        )
        for slot, ld in sorted(log_dirs.items()):
            _warn_zero_plot_metrics(ld, slot_tag=f' slot {slot}:')
            print_plot_metrics_summary(ld, slot=slot)


def run_test_and_plot_local(yt_cfg, device, force_test, force_plot=False):
    """Local fallback: rollout hold-out fold on its own held-out 20%, then data_plot."""
    from GNN_PlotFigure import data_plot
    yt_log_dir = log_path(yt_cfg.config_file)
    rollout_log = os.path.join(yt_log_dir, 'results_rollout.log')
    if force_test and os.path.exists(rollout_log):
        os.remove(rollout_log)
    if os.path.exists(rollout_log):
        print(f'  [skip] rollout log exists: {rollout_log}')
    else:
        print(f'  [run ] data_test hold-out  (dataset: {yt_cfg.dataset})')
        data_test(config=yt_cfg, visualize=False, best_model='best', run=0,
                  step=10, n_rollout_frames=250, device=device)
    metrics_path = os.path.join(yt_log_dir, 'results', 'metrics.txt')
    if force_plot and os.path.exists(metrics_path):
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
            print(f'  [skip] fold {i}: missing hold-out YAML {yt_yaml}')
            continue
        yt_cfgs.append(_load(yt_cfg_name, output_root))
    return yt_cfgs


def _assert_yt_data_present(yt_cfg):
    """Fail loud if the expected hold-out dataset is missing. Data generation is
    the sole responsibility of run_generate_holdout_data.py so the three training
    runners can safely start in parallel without racing on generation."""
    yt_gdir = graphs_data_path(yt_cfg.dataset)
    if not _have_data(yt_gdir):
        raise RuntimeError(
            f'Hold-out dataset missing or incomplete: {yt_gdir}\n'
            f'  Run `python run_generate_holdout_data.py` first to pre-build the '
            f'shared {{base}}_yt_cv{{i:02d}} datasets.')
    print(f'  [ok ] hold-out fold data present: {yt_gdir}')


def run_condition(base_name, suffix, n_folds, device, output_root,
                  node_name, hard_runtime_limit_min, force_test,
                  cluster_test_plot=True, metrics_interval=300,
                  force_train=False, force_plot=False,
                  skip_test_plot=False):
    """Train + test + plot one condition on the hold-out dataset (no DAVIS).

    Requires hold-out datasets to already exist (built by run_generate_holdout_data.py).
    Does NOT generate data — keeps the three training runners cheap to launch
    in parallel and avoids redundant 40× rebuilds of the hold-out augmentation cache.
    """
    run_condition_wave(
        [base_name], suffix=suffix, n_folds=n_folds, device=device,
        output_root=output_root, node_name=node_name,
        hard_runtime_limit_min=hard_runtime_limit_min, force_test=force_test,
        cluster_test_plot=cluster_test_plot,
        metrics_interval=metrics_interval,
        force_train=force_train, force_plot=force_plot,
        skip_test_plot=skip_test_plot,
    )


def run_condition_wave(base_names, suffix, n_folds, device, output_root,
                        node_name, hard_runtime_limit_min, force_test,
                        cluster_test_plot=True, metrics_interval=300,
                        force_train=False, force_plot=False,
                        skip_test_plot=False):
    """Train + test + plot MULTIPLE conditions as a single wave.

    All (base, fold) pairs are submitted together in one training wave
    (up to len(base_names) * n_folds concurrent cluster jobs), then the
    whole wave is awaited, then test+plot is submitted as a single wave.

    `skip_test_plot=True` runs ONLY the training wave and returns — no
    rollout, no parameter plot. Used by training-only re-runs where the
    user wants to inspect the trained model before committing to test/plot.
    """
    tag = '+'.join(base_names)
    print(f'\n=== wave[{len(base_names)}]: {tag}  ({n_folds}-fold hold-out train / hold-out test, suffix={suffix}) ===')

    yt_cfgs = []
    for base_name in base_names:
        yt_cfgs.extend(_load_yt_cfgs(base_name, suffix, n_folds, output_root))
    for yt_cfg in yt_cfgs:
        _assert_yt_data_present(yt_cfg)

    submit_training_wave(yt_cfgs, output_root, node_name,
                         hard_runtime_limit_min,
                         metrics_interval=metrics_interval,
                         force_train=force_train)

    if skip_test_plot:
        print(f'  [skip] test+plot wave suppressed (skip_test_plot=True)')
        return

    if cluster_test_plot:
        submit_test_plot_wave(yt_cfgs, output_root, node_name,
                              hard_runtime_limit_min, force_test,
                              force_plot=force_plot,
                              metrics_interval=metrics_interval)
    else:
        for yt_cfg in yt_cfgs:
            run_test_and_plot_local(yt_cfg, device, force_test, force_plot=force_plot)


def generate_yt_data_for_condition(base_name, suffix, n_folds, device, output_root):
    """Generate-only entry point: produces hold-out datasets for n_folds of one
    base condition and returns. Used by run_generate_holdout_data.py to pre-build the
    shared datasets before the 3 parallel training runners."""
    print(f'\n=== {base_name}  (generate-only, n_folds={n_folds}, suffix={suffix}) ===')
    yt_cfgs = _load_yt_cfgs(base_name, suffix, n_folds, output_root)
    for yt_cfg in yt_cfgs:
        ensure_yt_data(yt_cfg, device)
