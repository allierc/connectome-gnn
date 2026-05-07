"""Generate all optogenetics sweep datasets in parallel.

For each YAML produced by `scripts/generate_opto_configs.py`, runs
add_optogenetics_stimulus(config) once. The baseline source dataset
(flyvis_noise_free_blank50_cv00) must already exist on disk.

Two execution modes:

  --serial    Run conditions one after the other in the current process.
              Simplest; no cluster needed. ~30 min per condition on GPU.
  default     Spawn each condition as a background subprocess (parallel
              local execution). Limited by --n-parallel slots.

Usage:
    python run_generate_optogenetics.py
    python run_generate_optogenetics.py --serial
    python run_generate_optogenetics.py --conditions TmY15_white_noise Mi1_heaviside
    python run_generate_optogenetics.py --dry-run
"""
import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from connectome_gnn.utils import (  # noqa: E402
    config_path, get_data_root, graphs_data_path,
    load_data_root_from_json, set_data_root,
)


BASELINE = "flyvis_noise_free_blank50_cv00"
SOURCE_NAME_PREFIX = f"{BASELINE}_opto_"

CONDITIONS = [
    "TmY15_white_noise", "TmY15_heaviside",
    "Mi1_white_noise",   "Mi1_heaviside",
    "T4c_white_noise",   "T4c_heaviside",
    "R1_white_noise",    "R1_heaviside",
    "T1_white_noise",    "T1_heaviside",
]


def _config_yaml_for(cond: str) -> str:
    """Find the config YAML for a condition, trying repo and data-root config dirs."""
    name = f"{BASELINE}_opto_{cond}.yaml"
    candidates = [config_path("fly", name)]
    try:
        load_data_root_from_json()
        candidates.append(os.path.join(get_data_root(), "config", "fly", name))
    except Exception:
        pass
    candidates.append(f"/groups/saalfeld/home/allierc/GraphData/config/fly/{name}")
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"opto config for {cond!r} not found; tried {candidates}")


def _baseline_exists() -> bool:
    base = graphs_data_path("fly", BASELINE)
    if not os.path.isdir(base):
        return False
    return os.path.isdir(os.path.join(base, "x_list_train", "voltage.zarr"))


def _run_serial(yaml_paths: list[str]):
    runner = os.path.join(REPO_ROOT, "scripts", "run_add_optogenetics.py")
    py = sys.executable
    for cfg in yaml_paths:
        print(f"\n=== {os.path.basename(cfg)} ===", flush=True)
        proc = subprocess.run([py, runner, cfg])
        if proc.returncode != 0:
            sys.exit(f"FAILED: {cfg} (returncode={proc.returncode})")


def _run_parallel(yaml_paths: list[str], n_parallel: int, log_dir: str):
    runner = os.path.join(REPO_ROOT, "scripts", "run_add_optogenetics.py")
    py = sys.executable
    os.makedirs(log_dir, exist_ok=True)
    queue = list(yaml_paths)
    procs: dict[str, subprocess.Popen] = {}
    log_files: dict[str, object] = {}

    def _spawn(cfg: str):
        log = os.path.join(log_dir, f"{os.path.basename(cfg)}.log")
        f = open(log, "w")
        log_files[cfg] = f
        p = subprocess.Popen([py, runner, cfg], stdout=f, stderr=subprocess.STDOUT)
        procs[cfg] = p
        print(f"[spawn] {os.path.basename(cfg)}  pid={p.pid}  log={log}", flush=True)

    while queue and len(procs) < n_parallel:
        _spawn(queue.pop(0))

    failed: list[str] = []
    while procs:
        time.sleep(2)
        for cfg, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            log_files[cfg].close()
            print(f"[done ] {os.path.basename(cfg)}  rc={rc}", flush=True)
            del procs[cfg]
            if rc != 0:
                failed.append(cfg)
            if queue and len(procs) < n_parallel:
                _spawn(queue.pop(0))
    if failed:
        sys.exit(f"FAILED ({len(failed)}): {[os.path.basename(c) for c in failed]}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial", action="store_true",
                   help="run conditions sequentially in this process")
    p.add_argument("--n-parallel", type=int, default=4,
                   help="parallel local subprocesses (default 4)")
    p.add_argument("--conditions", nargs="*", default=None,
                   help=f"subset of conditions to run (default: all 10). "
                        f"Available: {CONDITIONS}")
    p.add_argument("--dry-run", action="store_true",
                   help="print intended commands without executing")
    p.add_argument("--log-dir", default=None,
                   help="parallel-mode per-condition log dir "
                        "(default: <data_root>/log/opto_generate)")
    args = p.parse_args()

    # Load the user's environment-specific data root (data_paths.json).
    try:
        set_data_root(load_data_root_from_json())
    except Exception:
        pass

    if not _baseline_exists():
        sys.exit(
            f"baseline {BASELINE!r} not found at {graphs_data_path('fly', BASELINE)}. "
            f"Generate it first via the unified blank50 pipeline."
        )

    selected = args.conditions or CONDITIONS
    yaml_paths = [_config_yaml_for(c) for c in selected]
    print(f"baseline: {graphs_data_path('fly', BASELINE)}")
    print(f"conditions ({len(yaml_paths)}):")
    for y in yaml_paths:
        print(f"  {os.path.basename(y)}")
    if args.dry_run:
        return

    if args.serial:
        _run_serial(yaml_paths)
    else:
        log_dir = args.log_dir or os.path.join(
            get_data_root(), "log", "opto_generate"
        )
        _run_parallel(yaml_paths, args.n_parallel, log_dir)


if __name__ == "__main__":
    main()
