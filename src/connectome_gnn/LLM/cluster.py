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

def check_cluster_repo():
    """Check that GraphCluster/flyvis-gnn has no uncommitted source changes.

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
        print("\033[91mERROR: GraphCluster repo has uncommitted changes — commit and push before running\033[0m")
        for line in diff_output.splitlines():
            print(f"  \033[91m{line}\033[0m")
        return False
    print("\033[92mCluster repo: git diff clean (no uncommitted source changes)\033[0m")
    return True


def submit_cluster_job(slot, config_path, analysis_log_path, config_file_field,
                       log_dir, erase=True, node_name='a100',
                       conda_env='connectome-gnn', n_cpus=2, device='cuda',
                       exploration_dir=None, iteration=None, output_root=None):
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

    if device == 'cpu':
        bsub_resources = f"bsub -n {n_cpus} -W 1440"
        queue_label = "cpu"
    else:
        bsub_resources = f"bsub -n {n_cpus} -gpu 'num=1' -q gpu_{node_name} -W 6000"
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
        ssh_cmd = f"ssh {CLUSTER_SSH} \"bash -l -c 'bjobs {ids_str}'\""
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
                        print(f"\033[91m  slot {slot} (job {jid}): FAILED (EXIT)\033[0m")
                        if log_dir:
                            err_file = f"{log_dir}/{job_prefix}_{slot:02d}.err"
                            if os.path.exists(err_file):
                                try:
                                    with open(err_file, 'r') as ef:
                                        err_content = ef.read().strip()
                                    if err_content:
                                        print(f"\033[91m  --- slot {slot} error log ---\033[0m")
                                        for eline in err_content.splitlines()[-30:]:
                                            print(f"\033[91m    {eline}\033[0m")
                                        print("\033[91m  --- end error log ---\033[0m")
                                except Exception:
                                    pass

            if slot in pending and jid not in out.stdout:
                results[slot] = True
                del pending[slot]
                print(f"\033[93m  slot {slot} (job {jid}): no longer in queue (assuming DONE)\033[0m")

        if pending:
            statuses = [f"slot {s}" for s in pending]
            print(f"\033[90m  ... waiting for {', '.join(statuses)} ({poll_interval}s)\033[0m")
            time.sleep(poll_interval)

    return results


def submit_cluster_test_plot_job(slot, config_path, analysis_log_path, config_file_field,
                                  log_dir, node_name='a100',
                                  conda_env='connectome-gnn', n_cpus=2, device='cuda',
                                  iteration=None, output_root=None):
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

    if device == 'cpu':
        bsub_resources = f"bsub -n {n_cpus} -W 1440"
        queue_label = "cpu"
    else:
        bsub_resources = f"bsub -n {n_cpus} -gpu 'num=1' -q gpu_{node_name} -W 6000"
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
