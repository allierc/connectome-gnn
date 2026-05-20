"""Submit the drosophila_cx_pi_column_dale L1 sweep to the cluster.

Submits the three `coeff_W_L1` variants (1e-3, 1e-4, 1e-5) as independent
cluster training jobs on A100 (override with --cluster), then polls each
slot's tmp_training/metrics.log every 5 minutes and prints the current
pi_acc / fwhm / loss / mse plus per-epoch trajectory until every job
finishes.

Examples
--------
# Default: all three L1 variants on a100, 5-minute metric polling.
python run_cx_column_dale.py

# Include the no-L1 baseline as a fourth slot.
python run_cx_column_dale.py --with-baseline

# Cheaper queue, tighter wall-clock.
python run_cx_column_dale.py --cluster l4 --hard-runtime-min 90
"""

import argparse
import os
import sys
import threading

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from connectome_gnn.LLM.cluster import (             # noqa: E402
    check_cluster_repo,
    submit_cluster_job,
    wait_for_cluster_jobs,
)
from connectome_gnn.LLM.task_pipeline import _print_task_metrics  # noqa: E402
from connectome_gnn.utils import (                    # noqa: E402
    load_data_root_from_json,
    log_path,
    set_data_root,
)


_VARIANTS = [
    ("drosophila_cx/drosophila_cx_pi_column_dale_1", 1.0e-3),
    ("drosophila_cx/drosophila_cx_pi_column_dale_2", 1.0e-4),
    ("drosophila_cx/drosophila_cx_pi_column_dale_3", 1.0e-5),
]
_BASELINE = ("drosophila_cx/drosophila_cx_pi_column_dale", 0.0)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cluster", default="a100",
                   choices=["a100", "l4", "h100"])
    p.add_argument("--hard-runtime-min", type=int, default=180,
                   help="bsub -W ceiling per job (minutes).")
    p.add_argument("--conda-env", default="connectome-gnn")
    p.add_argument("--n-cpus", type=int, default=8)
    p.add_argument("--with-baseline", action="store_true",
                   help="Also submit the no-L1 column_dale baseline.")
    p.add_argument("--poll-min", type=int, default=5,
                   help="Metric print + bjobs poll interval (minutes).")
    return p.parse_args()


def main():
    args = _parse_args()
    output_root = (os.environ.get("GNN_OUTPUT_ROOT")
                   or load_data_root_from_json())

    # Put log/ under the shared filesystem so the cluster_train_XX.sh files
    # this runner writes locally are reachable from the cluster nodes.
    # Without this, log_path() returns "./log/..." (local-cwd only) and the
    # cluster can't bash the .sh, so every job EXITs immediately.
    set_data_root(output_root)

    # Pass yaml paths as RELATIVE to the repo root. The cluster .sh does
    # `cd CLUSTER_ROOT_DIR` before running, so a relative path resolves
    # against the cluster repo while the local assert below resolves
    # against the local repo (cwd).
    os.chdir(_ROOT)

    variants = list(_VARIANTS)
    if args.with_baseline:
        variants.append(_BASELINE)

    if not check_cluster_repo():
        print("WARNING: cluster repo has uncommitted source changes — "
              "proceeding anyway. Pull on the cluster and re-run to be safe.")

    job_ids: dict = {}
    slot_log_dirs: dict = {}
    for slot, (config_file_field, l1) in enumerate(variants):
        cfg_rel = os.path.join("config", config_file_field + ".yaml")
        assert os.path.isfile(cfg_rel), f"config not found locally: {cfg_rel}"
        slot_log_dir = log_path(config_file_field)
        os.makedirs(slot_log_dir, exist_ok=True)
        analysis_log = os.path.join(slot_log_dir, "cluster_train.log")
        slot_log_dirs[slot] = slot_log_dir

        print(f"slot {slot}: {config_file_field}   coeff_W_L1={l1:.0e}")
        jid = submit_cluster_job(
            slot=slot,
            config_path=cfg_rel,
            analysis_log_path=analysis_log,
            config_file_field=config_file_field,
            log_dir=slot_log_dir,
            erase=True,
            node_name=args.cluster,
            conda_env=args.conda_env,
            n_cpus=args.n_cpus,
            device="cuda",
            output_root=output_root,
            hard_runtime_limit_min=args.hard_runtime_min,
        )
        if jid:
            job_ids[slot] = jid

    if not job_ids:
        print("No jobs were submitted — aborting.")
        return

    poll_seconds = max(60, args.poll_min * 60)
    print(f"\nPolling {len(job_ids)} job(s) every {args.poll_min} min "
          f"(metrics + bjobs)\n")

    stop_print = threading.Event()

    def _periodic_print():
        while not stop_print.wait(timeout=poll_seconds):
            _print_task_metrics(slot_log_dirs, list(job_ids.keys()),
                                prefix="  [metrics]")

    printer = threading.Thread(target=_periodic_print, daemon=True)
    printer.start()
    try:
        results = wait_for_cluster_jobs(
            job_ids, log_dir=None, poll_interval=poll_seconds,
            job_prefix="cluster_train",
        )
    finally:
        stop_print.set()
        printer.join(timeout=1.0)

    print("\n=== Final status ===")
    for slot, (cff, l1) in enumerate(variants):
        ok = results.get(slot, False)
        tag = "DONE" if ok else "FAILED/UNKNOWN"
        print(f"  slot {slot}  L1={l1:.0e}  {cff}  → {tag}")
    _print_task_metrics(slot_log_dirs, list(job_ids.keys()),
                        prefix="  [final ]")


if __name__ == "__main__":
    main()
