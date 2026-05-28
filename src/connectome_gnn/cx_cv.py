"""10-fold CV helper for drosophila_cx_pi_* configs.

Per fold, we permute three seeds (simulation, training, path_integration)
and re-run the dataset generation + training pipeline. Each fold writes
to its own dataset / log directories so all 10 folds can train in
parallel on the cluster.

Output layout (rooted at GNN_OUTPUT_ROOT or load_data_root_from_json()):

  config/drosophila_cx/<base>_cv{0..9}.yaml
  graphs_data/drosophila_cx/<base>_cv{0..9}/...
  log/drosophila_cx/<base>_cv{0..9}/...
  log/drosophila_cx/<base>_cv_summary.md

Mimics the high-level shape of run_GNN_flywire_blank50.py (yaml emit ->
data gen -> bsub fan-out -> wait -> summary) but skips the cross-task
machinery — drosophila_cx_pi_task is single-condition + synthetic so
the YT-VOS hold-out plumbing in connectome_gnn.cross does not apply.
"""
from __future__ import annotations

import copy
import glob
import os
import re
import subprocess
import time

import yaml

from connectome_gnn.LLM.cluster import (
    CLUSTER_SSH,
    CLUSTER_ROOT_DIR,
    submit_cluster_job,
    wait_for_cluster_jobs,
    wait_for_cluster_jobs_with_metrics,
)
from connectome_gnn.utils import load_data_root_from_json


# Base seed offsets per fold (i in 0..9). Offsets are large enough that
# the (sim, train, task) triples never collide across folds for any of
# the four base configs.
_SIM_SEED_OFFSET   = 100_000
_TRAIN_SEED_OFFSET = 200_000
_TASK_SEED_OFFSET  = 300_000


def _cv_seeds(fold: int) -> dict:
    """Return the per-fold seed triple."""
    return {
        "sim":   _SIM_SEED_OFFSET   + fold,
        "train": _TRAIN_SEED_OFFSET + fold,
        "task":  _TASK_SEED_OFFSET  + fold,
    }


def _output_root(explicit: str | None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    env = os.environ.get("GNN_OUTPUT_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.abspath(load_data_root_from_json())


def _repo_root() -> str:
    """Path to the local checkout of the repo (we live in src/connectome_gnn/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _base_yaml_path(base: str) -> str:
    return os.path.join(_repo_root(), "config", "drosophila_cx", f"{base}.yaml")


def emit_cv_yamls(base: str, output_root: str, n_folds: int = 10,
                  data_base: str | None = None) -> list[str]:
    """Write `<output_root>/config/drosophila_cx/<base>_cv{i}.yaml` for each
    fold. Returns the list of absolute yaml paths.

    If `data_base` is set, the emitted yamls point their `dataset:` field at
    `<data_base>_cv{i}` instead of `<base>_cv{i}` — i.e. the trainer reads
    data from another base's graphs_data/ tree. Logs still go under
    `<base>_cv{i}/` because the trainer's log path is controlled by
    --config_file (see `_config_file_field`), not the dataset field.
    """
    src_path = _base_yaml_path(base)
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"base yaml not found: {src_path}")
    with open(src_path) as f:
        base_cfg = yaml.safe_load(f)

    out_dir = os.path.join(output_root, "config", "drosophila_cx")
    os.makedirs(out_dir, exist_ok=True)

    dataset_base = data_base if data_base else base

    written = []
    for i in range(n_folds):
        seeds = _cv_seeds(i)
        cfg = copy.deepcopy(base_cfg)

        # Per-fold seeds (every seed key we know about).
        cfg.setdefault("simulation", {})["seed"] = seeds["sim"]
        cfg.setdefault("training", {})["seed"]   = seeds["train"]
        cfg.setdefault("task", {}).setdefault(
            "path_integration", {})["seed"] = seeds["task"]

        # Dataset name controls where the trainer reads graphs_data/ from.
        # When data_base is set we point at the shared dataset; otherwise
        # each fold owns its own graphs_data/ tree.
        cfg["dataset"] = f"{dataset_base}_cv{i}"

        out_path = os.path.join(out_dir, f"{base}_cv{i}.yaml")
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        written.append(out_path)
    if data_base:
        print(f"[emit] wrote {len(written)} CV yamls under {out_dir} "
              f"(dataset -> {data_base}_cv{{0..{n_folds-1}}})")
    else:
        print(f"[emit] wrote {len(written)} CV yamls under {out_dir}")
    return written


def _dataset_exists(output_root: str, base: str, fold: int) -> bool:
    """Check whether the swim/PI zarr dataset for this fold is already on
    disk. We probe for `train/theta_hd.zarr` which is the last file
    `_generate_path_integration_task` writes."""
    ds_dir = os.path.join(
        output_root, "graphs_data", "drosophila_cx",
        f"{base}_cv{fold}", "train",
    )
    return os.path.isdir(ds_dir) and bool(
        glob.glob(os.path.join(ds_dir, "*.zarr*"))
    )


def generate_cv_datasets(base: str, output_root: str, n_folds: int = 10,
                          force: bool = False) -> None:
    """Generate the per-fold datasets locally (sequential).

    Per fold we shell out to GNN_Main.py so the data generator picks up
    the correct task config + seed from the freshly-written CV yaml.
    """
    repo = _repo_root()
    for i in range(n_folds):
        if not force and _dataset_exists(output_root, base, i):
            print(f"  fold {i:02d}: dataset already present, skipping")
            continue
        cmd = (
            f"python {repo}/GNN_Main.py "
            f"-o generate drosophila_cx/{base}_cv{i} "
            f"--output_root {output_root}"
        )
        print(f"  fold {i:02d}: generating dataset ...")
        rc = subprocess.run(cmd, shell=True).returncode
        if rc != 0:
            raise RuntimeError(f"data generation failed for fold {i}")


def _config_file_field(base: str, fold: int) -> str:
    """The dotted relative name we pass via --config_file (e.g.
    'drosophila_cx/drosophila_cx_pi_fc_epg_cv3'). It controls where the
    trainer writes log/<this>/ and reads graphs_data/<this>/."""
    return f"drosophila_cx/{base}_cv{fold}"


def submit_cv_training(base: str, output_root: str, *,
                       n_folds: int = 10,
                       node_name: str = "l4",
                       conda_env: str = "connectome-gnn",
                       device: str = "cuda",
                       hard_runtime_limit_min: int = 240,
                       erase: bool = True) -> dict[int, str]:
    """Submit one bsub training job per fold via SSH. Returns slot ->
    job_id mapping. Caller can wait on this via wait_for_cluster_jobs."""
    yaml_dir = os.path.join(output_root, "config", "drosophila_cx")

    log_dir = os.path.join(output_root, "log", "drosophila_cx",
                           f"{base}_cv_runner")
    os.makedirs(log_dir, exist_ok=True)

    job_ids: dict[int, str] = {}
    for i in range(n_folds):
        config_path = os.path.join(yaml_dir, f"{base}_cv{i}.yaml")
        config_file_field = _config_file_field(base, i)
        analysis_log = os.path.join(log_dir, f"fold_{i:02d}_analysis.log")

        jid = submit_cluster_job(
            slot=i,
            config_path=config_path,
            analysis_log_path=analysis_log,
            config_file_field=config_file_field,
            log_dir=log_dir,
            erase=erase,
            node_name=node_name,
            conda_env=conda_env,
            n_cpus=2,
            device=device,
            output_root=output_root,
            hard_runtime_limit_min=hard_runtime_limit_min,
        )
        if jid is None:
            raise RuntimeError(f"failed to submit fold {i} to {node_name}")
        job_ids[i] = jid
    return job_ids


def _parse_final_r_roll_1k(analysis_log: str) -> float | None:
    """Best-effort parse of the last r_roll_1k value from the analysis log."""
    if not os.path.isfile(analysis_log):
        return None
    try:
        with open(analysis_log) as f:
            txt = f.read()
    except OSError:
        return None
    matches = re.findall(r"r_roll_1k\s*[:=]\s*([+-]?\d+\.\d+)", txt)
    return float(matches[-1]) if matches else None


def emit_summary_md(base: str, output_root: str, n_folds: int = 10) -> str:
    """Read each fold's analysis log, pull the final r_roll_1k, and emit a
    summary markdown table next to the runner logs."""
    log_root = os.path.join(output_root, "log", "drosophila_cx",
                            f"{base}_cv_runner")
    summary_path = os.path.join(log_root, f"{base}_cv_summary.md")
    rows = []
    for i in range(n_folds):
        analysis_log = os.path.join(log_root, f"fold_{i:02d}_analysis.log")
        r = _parse_final_r_roll_1k(analysis_log)
        rows.append((i, r))

    vals = [r for _, r in rows if r is not None]
    n_have = len(vals)
    n_pass = sum(1 for r in vals if r >= 0.95)
    mean = sum(vals) / n_have if n_have else float("nan")
    sd = (
        (sum((r - mean) ** 2 for r in vals) / n_have) ** 0.5
        if n_have else float("nan")
    )

    lines = [
        f"# {base} — 10-fold CV summary",
        "",
        f"- folds with data: {n_have}/{n_folds}",
        f"- mean r_roll_1k:  {mean:.4f}",
        f"- std  r_roll_1k:  {sd:.4f}",
        f"- pass rate (>= 0.95): {n_pass}/{n_have if n_have else n_folds}",
        "",
        "| fold | r_roll_1k |",
        "|------|-----------|",
    ]
    for i, r in rows:
        lines.append(f"| {i} | {'-' if r is None else f'{r:.4f}'} |")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[summary] wrote {summary_path}")
    return summary_path


def run_cv10(base: str, *,
             output_root: str | None = None,
             n_folds: int = 10,
             node_name: str = "l4",
             conda_env: str = "connectome-gnn",
             skip_emit: bool = False,
             skip_generate: bool = False,
             skip_train: bool = False,
             force_generate: bool = False,
             wait: bool = True,
             hard_runtime_limit_min: int = 240,
             data_base: str | None = None) -> None:
    """End-to-end 10-fold CV driver for a single drosophila_cx base config.

    If `data_base` is set, the emitted yamls point their `dataset:` field at
    `<data_base>_cv{i}` (shared with another runner) and dataset generation
    is skipped automatically.
    """
    out = _output_root(output_root)
    print(f"\n=== {base}: 10-fold CV on {node_name} (output_root={out}) ===")

    if not skip_emit:
        emit_cv_yamls(base, out, n_folds=n_folds, data_base=data_base)
    if data_base:
        skip_generate = True
    if not skip_generate:
        generate_cv_datasets(base, out, n_folds=n_folds, force=force_generate)
    if skip_train:
        print("[train] skipped (--skip-train)")
        return

    job_ids = submit_cv_training(
        base, out,
        n_folds=n_folds,
        node_name=node_name,
        conda_env=conda_env,
        hard_runtime_limit_min=hard_runtime_limit_min,
    )
    if wait:
        log_dir = os.path.join(out, "log", "drosophila_cx",
                               f"{base}_cv_runner")
        per_slot_log_dirs = {
            i: os.path.join(out, "log", "drosophila_cx", f"{base}_cv{i}")
            for i in job_ids
        }
        wait_for_cluster_jobs_with_metrics(
            job_ids,
            log_dirs=per_slot_log_dirs,
            poll_interval=300,
            metrics_interval=300,
        )
        emit_summary_md(base, out, n_folds=n_folds)
    else:
        print(f"[submitted] {len(job_ids)} jobs; not waiting "
              f"(use --no-wait was set)")
