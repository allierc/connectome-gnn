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
# starves the GPU at the default n_cpus=2; bumping to 8 keeps the input
# pipeline ahead of the trainer. a100 / h100 jobs run on the larger
# flywire connectomes (full_eye, 50k neurons; proximal_nulls, ≤9.6M edges)
# — the default n_cpus=2 caps LSF memory at ~80 GB which is below the
# voltage.zarr + stimulus.zarr + DAL footprint, so bump both to n_cpus=8
# (~320 GB) to avoid TERM_MEMLIMIT during data load.
_CPUS_PER_NODE = {'l4': 8, 'a100': 8, 'h100': 8}


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
                                        n_rollout_frames=250,
                                        skip_test=False, skip_plot=False):
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
    if skip_test:
        cluster_cmd += " --skip-test"
    if skip_plot:
        cluster_cmd += " --skip-plot"
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
    """Return a dict from the last line of the training metrics log.

    Always-present keys: iter, conn, vr, tau.
    Optional (None when absent / 'nan'):
      hid, anc — hidden-NNR Pearson (hidden-INR runs only).
      vr_clean, n_out_vr, n_total_vr — V_rest cleaned R² + outlier counts.
      tau_clean, n_out_tau, n_total_tau — τ cleaned R² + outlier counts.
    Returns None if the file is missing / empty.

    Backward-compatible: legacy 6-column logs still parse (the new fields
    come back as None / 0)."""
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

        def _opt_float(idx):
            if len(parts) <= idx:
                return None
            v = parts[idx].strip()
            if not v or v.lower() == 'nan':
                return None
            try:
                return float(v)
            except ValueError:
                return None

        def _opt_int(idx, default=0):
            if len(parts) <= idx:
                return default
            v = parts[idx].strip()
            if not v or v.lower() == 'nan':
                return default
            try:
                return int(float(v))
            except ValueError:
                return default

        return {
            'iter':         int(parts[0]),
            'conn':         float(parts[1]),
            'vr':           float(parts[2]),
            'tau':          float(parts[3]),
            'hid':          _opt_float(4),
            'anc':          _opt_float(5),
            'vr_clean':     _opt_float(6),
            'n_out_vr':     _opt_int(7),
            'n_total_vr':   _opt_int(8),
            'tau_clean':    _opt_float(9),
            'n_out_tau':    _opt_int(10),
            'n_total_tau':  _opt_int(11),
            'loss':         _opt_float(12),
        }
    except (OSError, ValueError):
        return None


def _read_total_iter(log_dir):
    """Return total expected training iterations, or None if not written yet.

    data_train writes <log_dir>/tmp_training/total_iter.txt at startup so the
    poller can display iter=I/total in the periodic [metrics] line.
    """
    path = os.path.join(log_dir, 'tmp_training', 'total_iter.txt')
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return int(f.read().strip())
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


def _fmt_count(n):
    """Compact integer → '32K' / '1.6M' / raw for n<1000.

    Strips trailing '.0' so 1_000_000 → '1M' (not '1.0M') and 32_000 → '32K'
    (not '32.0K'); preserves one decimal otherwise (32_500 → '32.5K').
    """
    if n is None or n < 0:
        return '?'
    if n < 1000:
        return str(n)
    # Cut over to M before n hits 1_000_000 so 999_500 doesn't render '1000K'.
    div, suffix = (1_000_000, 'M') if n >= 950_000 else (1000, 'K')
    s = f'{round(n / div, 1):.1f}'.rstrip('0').rstrip('.')
    return f'{s}{suffix}'


def _print_training_metrics(log_dirs, slots_active, prefix='  [metrics]'):
    """Read latest metrics.log line for each active slot and print colored.

    Same column layout as cross.pipeline.print_plot_metrics_summary so the
    terminal stream stays consistent across [metrics]/[final ]/[plot   ]:

        [metrics] slot N  iter=I  R²W=conn  R²Vr=clean(out%)  R²τ=clean(out%)
                                  cluster=…   [nnr=hid(anc)]

    R²W      = `conn` field from tmp_training/metrics.log (W R²)
    R²Vr     = vr_clean (no-outliers R²); out% = n_out_vr / n_total_vr
    R²τ      = tau_clean (no-outliers R²); out% = n_out_tau / n_total_tau
    cluster  = clustering_accuracy from results/metrics.txt (n/a until plot
               phase has run; never available during training-only waves).
    nnr      = hidden-INR Pearson (hidden-NGP / hidden-SIREN runs only).
    Falls back to the all-neurons R² when no-outliers fields are absent
    (legacy 6-column metrics.log).

    Block ordering: slots are grouped by condition (cfg_tag with the
    trailing `_cv\\d+` stripped) and the resulting blocks are printed in
    descending order of mean R²W across that condition's folds. Within a
    block the per-fold rows are sorted by slot id (= cv00, cv01, ... given
    that the runners enumerate yt_cfgs in CV-fold order). Each block is
    preceded by a `[summary]` line showing the mean ± SD of W R² over the
    folds present in this dump."""
    # First pass: pull per-slot metrics + cfg_tag + condition + iter_str.
    rows = []
    for slot in slots_active:
        log_dir = log_dirs.get(slot)
        if log_dir is None:
            continue
        cfg_tag = os.path.basename(log_dir.rstrip('/'))
        cond = re.sub(r'_cv\d+$', '', cfg_tag)
        tm = _read_latest_training_metrics(log_dir)
        iter_str = ''
        if tm is not None:
            total = _read_total_iter(log_dir)
            if total is not None:
                iter_str = f"iter={_fmt_count(tm['iter'])}/{_fmt_count(total)}"
            else:
                iter_str = f"iter={_fmt_count(tm['iter'])}"
        rows.append({'slot': slot, 'log_dir': log_dir, 'cfg_tag': cfg_tag,
                     'cond': cond, 'tm': tm, 'iter_str': iter_str})
    if not rows:
        return

    # Pad cfg_tag, slot text, and iter_str so the R²W column lines up across
    # every row — including the [summary] header, which uses an empty
    # placeholder of width (slot_w + 2 + tag_w + 2 + iter_w) so its R²W lands
    # at the same offset as the per-fold metrics rows below it.
    tag_w  = max(len(r['cfg_tag']) for r in rows)
    slot_w = max(len(f"slot {r['slot']}") for r in rows)
    iter_w = max((len(r['iter_str']) for r in rows if r['iter_str']),
                 default=0)
    summary_lhs_w = slot_w + 2 + tag_w + 2 + iter_w

    groups = {}
    for r in rows:
        groups.setdefault(r['cond'], []).append(r)

    def _mean_sd_n(vals):
        n = len(vals)
        if n == 0:
            return float('-inf'), 0.0, 0
        m = sum(vals) / n
        sd = (sum((v - m) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
        return m, sd, n

    group_stats = {
        c: _mean_sd_n([r['tm']['conn'] for r in rs
                       if r['tm'] is not None and r['tm'].get('conn') is not None])
        for c, rs in groups.items()
    }
    ordered_conds = sorted(groups.keys(), key=lambda c: -group_stats[c][0])

    def _fmt_R2_out(name, r2_clean, r2_all, n_out, n_total):
        # Prefer the no-outliers (cleaned) R² for display when available; the
        # legacy 6-column metrics.log only carries the all-neurons R².
        r2 = r2_clean if r2_clean is not None else r2_all
        base = f"{_r2_color(r2)}{name}={r2:.3f}"
        if r2_clean is None or not n_total or n_total <= 0:
            return base + _ANSI_RESET
        pct = 100.0 * n_out / n_total
        return base + f"({pct:.1f}%)" + _ANSI_RESET

    for cond in ordered_conds:
        rs = sorted(groups[cond], key=lambda x: x['slot'])
        m, sd, n = group_stats[cond]
        # Per-condition summary header (skip when only a single run is present
        # — the per-slot row already shows that value, so a header would just
        # repeat it). The cond placeholder is padded to summary_lhs_w so the
        # `R²W =` token lands at the same column as `R²W=` on the rows below.
        n_folds = len(rs)
        if n_folds > 1 and n > 0:
            lhs = cond.ljust(summary_lhs_w)
            print(f"  [summary] {lhs}  "
                  f"{_r2_color(m)}R²W = {m:.3f} ± {sd:.3f}{_ANSI_RESET}  "
                  f"(n={n}/{n_folds})")

        for r in rs:
            slot, log_dir, tm, cfg_tag = (r['slot'], r['log_dir'],
                                          r['tm'], r['cfg_tag'])
            if tm is None:
                print(f"{prefix} slot {slot}: (no metrics.log yet)")
                continue

            cl = _read_clustering_accuracy(log_dir)
            cl_str = (f"{_r2_color(cl)}{cl:.2f}{_ANSI_RESET}"
                      if cl is not None else f"{_ANSI_RESET}n/a{_ANSI_RESET}")

            parts = [
                f"{_r2_color(tm['conn'])}R²W={tm['conn']:.3f}{_ANSI_RESET}",
                _fmt_R2_out('R²Vr', tm['vr_clean'], tm['vr'],
                            tm['n_out_vr'], tm['n_total_vr']),
                _fmt_R2_out('R²τ', tm['tau_clean'], tm['tau'],
                            tm['n_out_tau'], tm['n_total_tau']),
                f"cluster={cl_str}",
            ]
            if tm.get('loss') is not None:
                parts.append(f"loss={tm['loss']:.2e}")
            # Hidden-INR diagnostics (only present for hidden-NGP / hidden-SIREN).
            # Pearson thresholds: green > 0.5, yellow > 0.3, orange > 0.1, red <.
            if tm['hid'] is not None:
                nnr_str = f"nnr={tm['hid']:.3f}"
                if tm['anc'] is not None:
                    nnr_str += f"({tm['anc']:.3f})"
                parts.append(f"{_r2_color(tm['hid'], thresholds=(0.5, 0.3, 0.1))}{nnr_str}{_ANSI_RESET}")
            slot_text = f"slot {slot}"
            print(f"{prefix} {slot_text:<{slot_w}}  {cfg_tag:<{tag_w}}  "
                  f"{r['iter_str']:<{iter_w}}  " + '  '.join(parts))


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
