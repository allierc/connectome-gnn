import json
import os
import re
import subprocess
import time

# ---------------------------------------------------------------------------
# Cluster constants (loaded from data_paths.json)
# ---------------------------------------------------------------------------

def _load_cluster_config() -> dict:
    candidates = [
        os.path.join(os.getcwd(), 'data_paths.json'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data_paths.json'),
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    return {}

_cluster_cfg = _load_cluster_config()
CLUSTER_USER     = _cluster_cfg.get('cluster_user', 'allierc')
CLUSTER_LOGIN    = _cluster_cfg.get('cluster_login', 'login1')
CLUSTER_ROOT_DIR = _cluster_cfg.get('cluster_root_dir', '/groups/saalfeld/home/allierc/GraphCluster/connectome-gnn')
CLUSTER_SSH      = f"{CLUSTER_USER}@{CLUSTER_LOGIN}"


# ---------------------------------------------------------------------------
# Cluster helpers
# ---------------------------------------------------------------------------

# Per-GPU-node CPU sizing. l4 GPUs are slower per-frame, so the data-loader
# starves the GPU at the default n_cpus=2; bumping to 4 keeps the input
# pipeline ahead of the trainer. Other nodes (a100, h100) are fine at 2.
_CPUS_PER_NODE = {'l4': 4}


def _resolve_n_cpus(node_name, n_cpus_default=2):
    """Return the bsub -n count for a given GPU node. Override per-node via
    _CPUS_PER_NODE; otherwise use the caller's default."""
    return _CPUS_PER_NODE.get(node_name, n_cpus_default)


def check_cluster_repo():
    """Check that the cluster repo has no uncommitted source changes.

    Runs `git diff HEAD` on the cluster via SSH, excluding config/ (which is
    expected to be modified by the LLM).  Returns True if clean, False if dirty.
    """
    ssh_cmd = (
        f"ssh {CLUSTER_SSH} "
        f"\"bash -l -c 'cd {CLUSTER_ROOT_DIR} && git diff HEAD --stat -- . \\\":!config/\\\"'\""
    )
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
    diff_output = result.stdout.strip()
    if diff_output:
        return False
    print(f"\033[92mCluster repo at {CLUSTER_ROOT_DIR}: git diff clean (no uncommitted source changes)\033[0m")
    return True


def submit_cluster_job(slot, config_path, analysis_log_path, config_file_field,
                       log_dir, erase=True, node_name='a100',
                       conda_env='connectome-gnn', n_cpus=2, device='cuda',
                       exploration_dir=None, iteration=None, output_root=None,
                       hard_runtime_limit_min=120):
    """Submit a single flyvis training job to the cluster WITHOUT -K (non-blocking).

    All paths are on a shared filesystem accessible from both local and cluster.
    Data generation and test/plot are handled locally in GNN_LLM.py.
    The cluster job runs training only.
    """
    cluster_script_path = f"{log_dir}/cluster_train_{slot:02d}.sh"
    error_details_path = f"{log_dir}/training_error_{slot:02d}.log"

    # Resolve 'auto' → 'cuda' for cluster (PyTorch doesn't accept 'auto' as device string)
    if device == 'auto':
        device = 'cuda'

    assert os.path.isfile(config_path), f"Config file not found: {config_path}"

    cluster_train_cmd = f"python train_subprocess.py --config '{config_path}' --device {device}"
    if output_root:
        cluster_train_cmd += f" --output_root '{output_root}'"
    cluster_train_cmd += f" --log_file '{analysis_log_path}'"
    cluster_train_cmd += f" --config_file '{config_file_field}'"
    cluster_train_cmd += f" --error_log '{error_details_path}'"
    if erase:
        cluster_train_cmd += " --erase"
    if exploration_dir is not None and iteration is not None:
        cluster_train_cmd += f" --exploration_dir '{exploration_dir}'"
        cluster_train_cmd += f" --iteration {iteration}"
        cluster_train_cmd += f" --slot {slot}"

    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash -l\n")
        f.write(f"cd {CLUSTER_ROOT_DIR}\n")
        f.write(f"conda run -n {conda_env} {cluster_train_cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_stdout = f"{log_dir}/cluster_train_{slot:02d}.out"
    cluster_stderr = f"{log_dir}/cluster_train_{slot:02d}.err"

    # Per-GPU-node CPU sizing (l4 needs more CPUs to keep up; see _resolve_n_cpus).
    n_cpus_eff = _resolve_n_cpus(node_name, n_cpus_default=n_cpus)
    if device == 'cpu':
        bsub_resources = f"bsub -n {n_cpus_eff} -W {hard_runtime_limit_min}"
        queue_label = "cpu"
    else:
        bsub_resources = f"bsub -n {n_cpus_eff} -gpu 'num=1' -q gpu_{node_name} -W {hard_runtime_limit_min}"
        queue_label = f"gpu_{node_name}"
    ssh_cmd = (
        f"ssh {CLUSTER_SSH} \"bash -l -c 'cd {CLUSTER_ROOT_DIR} && "
        f"{bsub_resources} "
        f"-o {cluster_stdout!r} -e {cluster_stderr!r} "
        f"bash -l {cluster_script_path}'\""
    )
    print(f"\033[96m  slot {slot}: submitting to {queue_label} via SSH\033[0m", flush=True)
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    match = re.search(r'Job <(\d+)>', result.stdout)
    if match:
        job_id = match.group(1)
        print(f"\033[92m  slot {slot}: job {job_id} submitted to gpu_{node_name}\033[0m")
        return job_id
    else:
        print(f"\033[91m  slot {slot}: submission FAILED\033[0m")
        print(f"    stdout: {result.stdout.strip()}")
        print(f"    stderr: {result.stderr.strip()}")
        return None


def wait_for_cluster_jobs(job_ids, log_dir=None, poll_interval=60, job_prefix='cluster_train'):
    """Poll bjobs via SSH until all jobs finish."""
    pending = dict(job_ids)
    results = {}

    while pending:
        ids_str = ' '.join(pending.values())
        ssh_cmd = f"ssh {CLUSTER_SSH} \"source /etc/profile.d/profile.lsf.sh && bjobs {ids_str}\""
        out = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
        if out.returncode != 0 and not out.stdout.strip():
            raise RuntimeError(
                f"bjobs failed (rc={out.returncode}): {out.stderr.strip() or '(no output)'}"
            )

        for slot, jid in list(pending.items()):
            for line in out.stdout.splitlines():
                if jid in line:
                    if 'DONE' in line:
                        results[slot] = True
                        del pending[slot]
                        print(f"\033[92m  slot {slot} (job {jid}): DONE\033[0m")
                    elif 'EXIT' in line:
                        results[slot] = False
                        del pending[slot]
                        err_hint = ''
                        if log_dir:
                            err_file = f"{log_dir}/{job_prefix}_{slot:02d}.err"
                            if os.path.exists(err_file):
                                err_hint = f"  (see {err_file})"
                        print(
                            f"\033[91m  slot {slot} (job {jid}): FAILED (EXIT)"
                            f"{err_hint}\033[0m"
                        )

            if slot in pending and jid not in out.stdout:
                results[slot] = True
                del pending[slot]
                print(f"\033[93m  slot {slot} (job {jid}): no longer in queue (assuming DONE)\033[0m")

        if pending:
            statuses = [f"slot {s}" for s in pending]
            print(f"\033[90m  ... waiting for {', '.join(statuses)} ({poll_interval}s)\033[0m")
            time.sleep(poll_interval)

    return results


def submit_cluster_cross_test_plot_job(slot, config_path, test_config_paths,
                                        analysis_log_path, config_file_field,
                                        test_config_file_fields,
                                        log_dir, node_name='a100',
                                        conda_env='connectome-gnn', n_cpus=2,
                                        device='cuda', iteration=None,
                                        output_root=None,
                                        hard_runtime_limit_min=60,
                                        n_rollout_frames=250):
    """Submit cross_test_plot_subprocess.py to the cluster.

    One cluster job per YT fold. Inside the job, the subprocess loops over
    every test_config in `test_config_paths` (so 1 YT model × N DAVIS CV
    folds ⇒ N rollout logs) and then runs a single data_plot.

    Args:
        test_config_paths:      list of DAVIS CV YAML paths (shared FS).
        test_config_file_fields: matching config_file fields (same length).
    """
    assert isinstance(test_config_paths, (list, tuple)) and test_config_paths
    assert isinstance(test_config_file_fields, (list, tuple))
    assert len(test_config_file_fields) == len(test_config_paths)

    cluster_script_path = f"{log_dir}/cluster_cross_test_plot_{slot:02d}.sh"
    error_details_path = f"{log_dir}/cross_test_plot_error_{slot:02d}.log"

    if device == 'auto':
        device = 'cuda'

    assert os.path.isfile(config_path), f"Config file not found: {config_path}"
    for p_ in test_config_paths:
        assert os.path.isfile(p_), f"Test config file not found: {p_}"

    quoted_test_yamls = ' '.join(f"'{p}'" for p in test_config_paths)
    quoted_test_fields = ' '.join(f"'{f}'" for f in test_config_file_fields)

    cluster_cmd = (
        f"python cross_test_plot_subprocess.py "
        f"--config '{config_path}' "
        f"--test_configs {quoted_test_yamls} "
        f"--test_config_files {quoted_test_fields} "
        f"--device {device}"
    )
    if output_root:
        cluster_cmd += f" --output_root '{output_root}'"
    cluster_cmd += f" --log_file '{analysis_log_path}'"
    cluster_cmd += f" --config_file '{config_file_field}'"
    cluster_cmd += f" --error_log '{error_details_path}'"
    cluster_cmd += f" --n_rollout_frames {n_rollout_frames}"
    if iteration is not None:
        cluster_cmd += f" --iteration {iteration}"
        cluster_cmd += f" --slot {slot}"

    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash -l\n")
        f.write(f"cd {CLUSTER_ROOT_DIR}\n")
        f.write(f"conda run -n {conda_env} {cluster_cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_stdout = f"{log_dir}/cluster_cross_test_plot_{slot:02d}.out"
    cluster_stderr = f"{log_dir}/cluster_cross_test_plot_{slot:02d}.err"

    # Per-GPU-node CPU sizing (l4 needs more CPUs to keep up; see _resolve_n_cpus).
    n_cpus_eff = _resolve_n_cpus(node_name, n_cpus_default=n_cpus)
    if device == 'cpu':
        bsub_resources = f"bsub -n {n_cpus_eff} -W {hard_runtime_limit_min}"
        queue_label = "cpu"
    else:
        bsub_resources = f"bsub -n {n_cpus_eff} -gpu 'num=1' -q gpu_{node_name} -W {hard_runtime_limit_min}"
        queue_label = f"gpu_{node_name}"

    ssh_cmd = (
        f"ssh {CLUSTER_SSH} \"bash -l -c 'cd {CLUSTER_ROOT_DIR} && "
        f"{bsub_resources} "
        f"-o {cluster_stdout!r} -e {cluster_stderr!r} "
        f"bash -l {cluster_script_path}'\""
    )
    print(f"\033[96m  slot {slot}: submitting cross test+plot to {queue_label} via SSH\033[0m", flush=True)
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    match = re.search(r'Job <(\d+)>', result.stdout)
    if match:
        job_id = match.group(1)
        print(f"\033[92m  slot {slot}: cross test+plot job {job_id} submitted to {queue_label}\033[0m")
        return job_id
    else:
        print(f"\033[91m  slot {slot}: cross test+plot submission FAILED\033[0m")
        print(f"    stdout: {result.stdout.strip()}")
        print(f"    stderr: {result.stderr.strip()}")
        return None


# ---------------------------------------------------------------------------
# ANSI color helper for training metrics (mirrors graph_trainer.r2_color).
# ---------------------------------------------------------------------------
_ANSI_RESET  = '\033[0m'
_ANSI_GREEN  = '\033[92m'
_ANSI_YELLOW = '\033[93m'
_ANSI_ORANGE = '\033[38;5;208m'
_ANSI_RED    = '\033[91m'


def _r2_color(val, thresholds=(0.9, 0.7, 0.3)):
    t0, t1, t2 = thresholds
    return (_ANSI_GREEN if val > t0 else
            _ANSI_YELLOW if val > t1 else
            _ANSI_ORANGE if val > t2 else _ANSI_RED)


def _read_latest_training_metrics(log_dir):
    """Return (iter, conn_r2, vr_r2, tau_r2, hid_nnr, anc_nnr) from the last
    line of the training metrics log. The last two are None when the run is
    not a hidden-INR model (fields absent or 'nan'). Returns None if the file
    is missing / empty."""
    path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            # Skip header; return last non-empty data line.
            last = None
            for line in f:
                line = line.strip()
                if not line or line.startswith('iteration'):
                    continue
                last = line
        if last is None:
            return None
        parts = last.split(',')
        if len(parts) < 4:
            return None
        it = int(parts[0])
        conn = float(parts[1])
        vr   = float(parts[2])
        tau  = float(parts[3])

        def _opt(idx):
            if len(parts) <= idx:
                return None
            v = parts[idx].strip()
            if not v or v.lower() == 'nan':
                return None
            try:
                return float(v)
            except ValueError:
                return None

        hid = _opt(4)
        anc = _opt(5)
        return (it, conn, vr, tau, hid, anc)
    except (OSError, ValueError):
        return None


def _read_clustering_accuracy(log_dir):
    """Return clustering_accuracy float from <log_dir>/results/metrics.txt
    (written by data_plot), or None if not available yet."""
    path = os.path.join(log_dir, 'results', 'metrics.txt')
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            for line in f:
                if ':' not in line:
                    continue
                key, val = line.split(':', 1)
                if key.strip() == 'clustering_accuracy':
                    return float(val.strip())
    except (OSError, ValueError):
        return None
    return None


def _print_training_metrics(log_dirs, slots_active, prefix='  [metrics]'):
    """Read latest metrics.log line for each active slot and print colored."""
    for slot in sorted(slots_active):
        log_dir = log_dirs.get(slot)
        if log_dir is None:
            continue
        tm = _read_latest_training_metrics(log_dir)
        if tm is None:
            print(f"{prefix} slot {slot}: (no metrics.log yet)")
            continue
        it, conn, vr, tau, hid, anc = tm
        parts = [
            f"{_r2_color(conn)}conn={conn:.3f}{_ANSI_RESET}",
            f"{_r2_color(vr)}Vr={vr:.3f}{_ANSI_RESET}",
            f"{_r2_color(tau)}τ={tau:.3f}{_ANSI_RESET}",
        ]
        # Hidden-INR diagnostics (only present for hidden-NGP / hidden-SIREN runs).
        # Pearson thresholds: green > 0.5, yellow > 0.3, orange > 0.1, red <.
        if hid is not None:
            nnr_str = f"nnr={hid:.3f}"
            if anc is not None:
                nnr_str += f"({anc:.3f})"
            parts.append(f"{_r2_color(hid, thresholds=(0.5, 0.3, 0.1))}{nnr_str}{_ANSI_RESET}")
        print(f"{prefix} slot {slot}  iter={it:>6}  " + '  '.join(parts))


def wait_for_cluster_jobs_with_metrics(job_ids, log_dirs, poll_interval=60,
                                       metrics_interval=300,
                                       job_prefix='cluster_train'):
    """Like wait_for_cluster_jobs, but also reads and prints training metrics
    from each slot's tmp_training/metrics.log every `metrics_interval` seconds
    (plus once per slot the moment it reports DONE).

    Args:
        job_ids: {slot: job_id}
        log_dirs: {slot: log_dir}   (per-slot log dir; metrics.log is
                                      <log_dir>/tmp_training/metrics.log)
        poll_interval:   bjobs poll cadence (sec).  Default 60.
        metrics_interval: metrics-print cadence (sec). Default 300.
    """
    pending = dict(job_ids)
    results = {}
    # Wait one full metrics_interval before the first print — otherwise we
    # might show stale metrics from a previous run that the new cluster job
    # hasn't truncated yet (graph_trainer opens metrics.log in 'w' mode
    # only when it hits the first plot_training_flyvis checkpoint).
    last_metric_print = time.time()

    while pending:
        now = time.time()
        # Run bjobs.
        ids_str = ' '.join(pending.values())
        ssh_cmd = f"ssh {CLUSTER_SSH} \"source /etc/profile.d/profile.lsf.sh && bjobs {ids_str}\""
        out = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
        if out.returncode != 0 and not out.stdout.strip():
            raise RuntimeError(
                f"bjobs failed (rc={out.returncode}): {out.stderr.strip() or '(no output)'}"
            )

        just_finished = []
        for slot, jid in list(pending.items()):
            for line in out.stdout.splitlines():
                if jid in line:
                    if 'DONE' in line:
                        results[slot] = True
                        del pending[slot]
                        just_finished.append(slot)
                        # The [final ] metrics line below reports DONE
                        # implicitly; skip the redundant status print.
                    elif 'EXIT' in line:
                        results[slot] = False
                        del pending[slot]
                        err_hint = ''
                        lg = log_dirs.get(slot)
                        if lg:
                            err_file = f"{lg}/{job_prefix}_{slot:02d}.err"
                            if os.path.exists(err_file):
                                err_hint = f"  (see {err_file})"
                        print(
                            f"\033[91m  slot {slot} (job {jid}): FAILED (EXIT)"
                            f"{err_hint}\033[0m"
                        )
            if slot in pending and jid not in out.stdout:
                results[slot] = True
                del pending[slot]
                just_finished.append(slot)
                # [final ] line below reports this too.

        # Print metrics for any slot that just finished (final snapshot).
        if just_finished:
            _print_training_metrics(log_dirs, just_finished,
                                    prefix='  [final ]')

        # Periodic metrics print for still-pending slots.
        if pending and (now - last_metric_print) >= metrics_interval:
            _print_training_metrics(log_dirs, pending.keys(),
                                    prefix='  [metrics]')
            last_metric_print = now

        if pending:
            statuses = [f"slot {s}" for s in pending]
            print(f"\033[90m  ... waiting for {', '.join(statuses)} ({poll_interval}s)\033[0m")
            time.sleep(poll_interval)

    return results


def submit_cluster_test_plot_job(slot, config_path, analysis_log_path, config_file_field,
                                  log_dir, node_name='a100',
                                  conda_env='connectome-gnn', n_cpus=2, device='cuda',
                                  iteration=None, output_root=None,
                                  hard_runtime_limit_min=120):
    """Submit a single test+plot job to the cluster WITHOUT -K (non-blocking).

    Runs test_plot_subprocess.py on the cluster after training completes.
    """
    cluster_script_path = f"{log_dir}/cluster_test_plot_{slot:02d}.sh"
    error_details_path = f"{log_dir}/test_plot_error_{slot:02d}.log"

    if device == 'auto':
        device = 'cuda'

    assert os.path.isfile(config_path), f"Config file not found: {config_path}"

    cluster_cmd = f"python test_plot_subprocess.py --config '{config_path}' --device {device}"
    if output_root:
        cluster_cmd += f" --output_root '{output_root}'"
    cluster_cmd += f" --log_file '{analysis_log_path}'"
    cluster_cmd += f" --config_file '{config_file_field}'"
    cluster_cmd += f" --error_log '{error_details_path}'"
    if iteration is not None:
        cluster_cmd += f" --iteration {iteration}"
        cluster_cmd += f" --slot {slot}"

    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash -l\n")
        f.write(f"cd {CLUSTER_ROOT_DIR}\n")
        f.write(f"conda run -n {conda_env} {cluster_cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_stdout = f"{log_dir}/cluster_test_plot_{slot:02d}.out"
    cluster_stderr = f"{log_dir}/cluster_test_plot_{slot:02d}.err"

    # Per-GPU-node CPU sizing (l4 needs more CPUs to keep up; see _resolve_n_cpus).
    n_cpus_eff = _resolve_n_cpus(node_name, n_cpus_default=n_cpus)
    if device == 'cpu':
        bsub_resources = f"bsub -n {n_cpus_eff} -W {hard_runtime_limit_min}"
        queue_label = "cpu"
    else:
        bsub_resources = f"bsub -n {n_cpus_eff} -gpu 'num=1' -q gpu_{node_name} -W {hard_runtime_limit_min}"
        queue_label = f"gpu_{node_name}"

    ssh_cmd = (
        f"ssh {CLUSTER_SSH} \"bash -l -c 'cd {CLUSTER_ROOT_DIR} && "
        f"{bsub_resources} "
        f"-o {cluster_stdout!r} -e {cluster_stderr!r} "
        f"bash -l {cluster_script_path}'\""
    )
    print(f"\033[96m  slot {slot}: submitting test+plot to {queue_label} via SSH\033[0m", flush=True)
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    match = re.search(r'Job <(\d+)>', result.stdout)
    if match:
        job_id = match.group(1)
        print(f"\033[92m  slot {slot}: test+plot job {job_id} submitted to {queue_label}\033[0m")
        return job_id
    else:
        print(f"\033[91m  slot {slot}: test+plot submission FAILED\033[0m")
        print(f"    stdout: {result.stdout.strip()}")
        print(f"    stderr: {result.stderr.strip()}")
        return None


def submit_cluster_data_plot_job(slot, config_path, analysis_log_path, config_file_field,
                                  log_dir, node_name='a100',
                                  conda_env='connectome-gnn', n_cpus=2, device='cuda',
                                  iteration=None, output_root=None,
                                  hard_runtime_limit_min=120):
    """Submit a single data_plot-only job to the cluster (no rollout).

    Runs data_plot_subprocess.py — only re-extracts parameters from the
    already-trained model and overwrites <log_dir>/results/metrics.txt.
    Used by aggregate_blank50_tables.py --data_plot.
    """
    cluster_script_path = f"{log_dir}/cluster_data_plot_{slot:02d}.sh"
    error_details_path  = f"{log_dir}/data_plot_error_{slot:02d}.log"

    if device == 'auto':
        device = 'cuda'

    assert os.path.isfile(config_path), f"Config file not found: {config_path}"

    cluster_cmd = f"python data_plot_subprocess.py --config '{config_path}' --device {device}"
    if output_root:
        cluster_cmd += f" --output_root '{output_root}'"
    cluster_cmd += f" --log_file '{analysis_log_path}'"
    cluster_cmd += f" --config_file '{config_file_field}'"
    cluster_cmd += f" --error_log '{error_details_path}'"
    if iteration is not None:
        cluster_cmd += f" --iteration {iteration}"
        cluster_cmd += f" --slot {slot}"

    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash -l\n")
        f.write(f"cd {CLUSTER_ROOT_DIR}\n")
        f.write(f"conda run -n {conda_env} {cluster_cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_stdout = f"{log_dir}/cluster_data_plot_{slot:02d}.out"
    cluster_stderr = f"{log_dir}/cluster_data_plot_{slot:02d}.err"

    # Per-GPU-node CPU sizing (l4 needs more CPUs to keep up; see _resolve_n_cpus).
    n_cpus_eff = _resolve_n_cpus(node_name, n_cpus_default=n_cpus)
    if device == 'cpu':
        bsub_resources = f"bsub -n {n_cpus_eff} -W {hard_runtime_limit_min}"
        queue_label = "cpu"
    else:
        bsub_resources = f"bsub -n {n_cpus_eff} -gpu 'num=1' -q gpu_{node_name} -W {hard_runtime_limit_min}"
        queue_label = f"gpu_{node_name}"

    ssh_cmd = (
        f"ssh {CLUSTER_SSH} \"bash -l -c 'cd {CLUSTER_ROOT_DIR} && "
        f"{bsub_resources} "
        f"-o {cluster_stdout!r} -e {cluster_stderr!r} "
        f"bash -l {cluster_script_path}'\""
    )
    print(f"\033[96m  slot {slot}: submitting data_plot to {queue_label} via SSH\033[0m", flush=True)
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    match = re.search(r'Job <(\d+)>', result.stdout)
    if match:
        job_id = match.group(1)
        print(f"\033[92m  slot {slot}: data_plot job {job_id} submitted to {queue_label}\033[0m")
        return job_id
    else:
        print(f"\033[91m  slot {slot}: data_plot submission FAILED\033[0m")
        print(f"    stdout: {result.stdout.strip()}")
        print(f"    stderr: {result.stderr.strip()}")
        return None
