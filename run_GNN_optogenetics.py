"""Train one GNN per optogenetics sweep condition.

For each opto YAML produced by `scripts/generate_opto_configs.py`:
    1. Verify the dataset exists at graphs_data/fly/<config.dataset>/
       (generate it via run_generate_optogenetics.py first).
    2. Call data_train(config) — runs the full training pipeline
       (writes models/, results/, tmp_training/ under
        log/fly/<config.dataset>_<run_id>/).

This script is intentionally simpler than run_GNN_unified_blank50.py: no
cross-validation, no shared HP yaml, no LSF orchestration. Each opto
condition is a single training job derived directly from its standalone
config; the user fans out via --serial / --background / their own scheduler.

Usage:
    python run_GNN_optogenetics.py --serial
    python run_GNN_optogenetics.py --conditions TmY15_white_noise Mi1_white_noise
    python run_GNN_optogenetics.py --print-bsub > submit.sh   # emit bsub commands
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


OUTPUT_PREFIX = "flyvis_noise_free_blank50"

CONDITIONS = [
    "TmY15_white_noise", "TmY15_heaviside",  # 43,299
    "Mi1_white_noise",   "Mi1_heaviside",    # 25,834
    "Tm3_white_noise",   "Tm3_heaviside",    # 20,471
    "Tm4_white_noise",   "Tm4_heaviside",    # 15,971
    "Tm1_white_noise",   "Tm1_heaviside",    # 15,525
    "Mi4_white_noise",   "Mi4_heaviside",    # 14,439
    "T4c_white_noise",   "T4c_heaviside",    # 12,564
    "Mi9_white_noise",   "Mi9_heaviside",    # 11,889
    "Tm2_white_noise",   "Tm2_heaviside",    # 11,068
]


def _config_yaml_for(cond: str) -> str:
    name = f"{OUTPUT_PREFIX}_opto_{cond}.yaml"
    candidates = [config_path("fly", name)]
    try:
        load_data_root_from_json()
        candidates.append(os.path.join(get_data_root(), "config", "fly", name))
    except Exception:
        pass
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"opto config for {cond!r} not found; tried {candidates}")


def _dataset_exists(cond: str) -> bool:
    ds = f"{OUTPUT_PREFIX}_opto_{cond}"
    return os.path.isdir(os.path.join(graphs_data_path("fly", ds), "x_list_train", "voltage.zarr"))


def _train_one(cfg_path: str):
    """Run data_train on one config in the current process."""
    from connectome_gnn.config import NeuralGraphConfig
    from connectome_gnn.models.graph_trainer import data_train
    cfg = NeuralGraphConfig.from_yaml(cfg_path)
    print(f"\n=== training: {cfg.dataset} ===", flush=True)
    data_train(config=cfg)


def _run_serial(yaml_paths: list[str]):
    for cfg in yaml_paths:
        _train_one(cfg)


def _run_background(yaml_paths: list[str], n_parallel: int, log_dir: str):
    """Spawn each condition as a python subprocess. Limited by --n-parallel."""
    py = sys.executable
    runner = os.path.join(REPO_ROOT, "run_GNN_optogenetics.py")
    os.makedirs(log_dir, exist_ok=True)
    queue = list(yaml_paths)
    procs: dict[str, subprocess.Popen] = {}
    log_files: dict[str, object] = {}

    def _spawn(cfg: str):
        name = os.path.basename(cfg).replace(".yaml", "")
        log = os.path.join(log_dir, f"{name}.log")
        f = open(log, "w")
        log_files[cfg] = f
        # Re-invoke this script with --serial --conditions <one> so the
        # subprocess does the actual training.
        cond = name.split("_opto_", 1)[-1]
        p = subprocess.Popen(
            [py, runner, "--serial", "--conditions", cond],
            stdout=f, stderr=subprocess.STDOUT,
        )
        procs[cfg] = p
        print(f"[spawn] {name}  pid={p.pid}  log={log}", flush=True)

    while queue and len(procs) < n_parallel:
        _spawn(queue.pop(0))

    failed: list[str] = []
    while procs:
        time.sleep(5)
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


def _emit_bsub(yaml_paths: list[str], queue: str, runtime_min: int):
    """Print one bsub command per condition for cluster submission."""
    py = sys.executable
    runner = os.path.join(REPO_ROOT, "run_GNN_optogenetics.py")
    log_root = os.path.join(get_data_root() or ".", "log", "opto_train")
    print(f"# emitted by run_GNN_optogenetics.py --print-bsub")
    print(f"# log root: {log_root}")
    print(f"mkdir -p {log_root}")
    for cfg in yaml_paths:
        name = os.path.basename(cfg).replace(".yaml", "")
        cond = name.split("_opto_", 1)[-1]
        log = f"{log_root}/{name}.cluster.log"
        cmd = (
            f'bsub -J opto_{cond} -q gpu_{queue} -gpu "num=1" '
            f'-W {runtime_min} -o {log} '
            f'"{py} {runner} --serial --conditions {cond}"'
        )
        print(cmd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--serial", action="store_true",
                   help="train one condition at a time in this process")
    g.add_argument("--background", action="store_true",
                   help="spawn parallel local subprocesses (no cluster)")
    g.add_argument("--print-bsub", action="store_true",
                   help="print bsub commands for cluster submission, then exit")
    p.add_argument("--n-parallel", type=int, default=2,
                   help="background-mode concurrent slots (default 2)")
    p.add_argument("--conditions", nargs="*", default=None,
                   help=f"subset of conditions (default: all). "
                        f"Available: {CONDITIONS}")
    p.add_argument("--queue", default="a100",
                   help="cluster GPU queue suffix for --print-bsub (default a100)")
    p.add_argument("--runtime-min", type=int, default=2880,
                   help="cluster runtime cap minutes for --print-bsub (default 2880)")
    p.add_argument("--log-dir", default=None,
                   help="background-mode log dir (default: <data_root>/log/opto_train)")
    p.add_argument("--dry-run", action="store_true",
                   help="print intended actions without running")
    args = p.parse_args()

    try:
        set_data_root(load_data_root_from_json())
    except Exception:
        pass

    selected = args.conditions or CONDITIONS
    yaml_paths = [_config_yaml_for(c) for c in selected]

    missing = [c for c in selected if not _dataset_exists(c)]
    if missing:
        print(f"WARNING: {len(missing)} dataset(s) missing on disk: {missing}")
        print("         run run_generate_optogenetics.py first.")

    print(f"conditions ({len(yaml_paths)}):")
    for y in yaml_paths:
        print(f"  {os.path.basename(y)}")

    if args.dry_run:
        return

    if args.print_bsub:
        _emit_bsub(yaml_paths, args.queue, args.runtime_min)
        return

    if args.background:
        log_dir = args.log_dir or os.path.join(get_data_root(), "log", "opto_train")
        _run_background(yaml_paths, args.n_parallel, log_dir)
        return

    if args.serial:
        _run_serial(yaml_paths)
        return

    # default: print bsub commands so the user opts in to a runtime mode
    print("\nNo execution mode selected. Pick one:")
    print("  --serial            run sequentially in this process")
    print("  --background        local parallel subprocesses (--n-parallel slots)")
    print("  --print-bsub        emit cluster submit commands")


if __name__ == "__main__":
    main()
