#!/usr/bin/env python
"""Overnight collector for the NeurIPS rebuttal runs (Vzfg Q1 + Q2).

Two jobs, polled every POLL_S seconds until everything is terminal:

  1. Harvest. A run counts as finished only when its `_complete` marker exists
     AND results/metrics.txt is parseable. This is deliberate: finish_joint.py
     treated a timed-out `bjobs` as "job gone -> done", which is why the joint
     rows never reached results_table.csv. Here an ssh failure is UNKNOWN and
     never terminal, so the worst case is polling longer, never a false row.

  2. Unblock. The sat/rate5/full oracle arms could not be submitted with their
     GNN twins: both would have generated the same dataset concurrently under
     --force. This submits each oracle arm with -o train_test_plot exactly once,
     as soon as its dataset's `_completed_generate` marker appears.

Outputs, rewritten every pass so a killed session loses nothing:
  neurips_review/overnight_results.csv   one row per finished run
  neurips_review/overnight_status.md     human-readable snapshot
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path("/groups/saalfeld/home/allierc/GraphData")
CX = Path("/workspace/connectome-gnn-cx")
OUT_CSV = CX / "neurips_review" / "overnight_results.csv"
OUT_MD = CX / "neurips_review" / "overnight_status.md"
LOG = CX / "neurips_review" / "_overnight.log"

POLL_S = 300
MAX_HOURS = 16
CA_REMOTE = "/groups/saalfeld/home/allierc/Graph/connectome-gnn-ca"

# config name -> (biomodel dir, arm label, model)
RUNS: dict[str, tuple[str, str, str]] = {
    # Vzfg Q1 -- drosophila CX ring attractor, local
    "nr2_cx_ring_s00":            ("drosophila_cx", "CX sigma=0",    "gnn"),
    "nr2_cx_ring_s005":           ("drosophila_cx", "CX sigma=0.05", "gnn"),
    "nr2_cx_ring_s05":            ("drosophila_cx", "CX sigma=0.5",  "gnn"),
    "nr2_cx_ring_s00_known_ode":  ("drosophila_cx", "CX sigma=0",    "oracle"),
    "nr2_cx_ring_s005_known_ode": ("drosophila_cx", "CX sigma=0.05", "oracle"),
    "nr2_cx_ring_s05_known_ode":  ("drosophila_cx", "CX sigma=0.5",  "oracle"),
    # Vzfg Q2 -- calcium observation model, cluster
    "nr2_ca_voltage_unified":     ("fly", "voltage control",   "gnn"),
    "nr2_ca_calcium_unified":     ("fly", "kernel only",       "gnn"),
    "nr2_ca_calcium_known_ode":   ("fly", "kernel only",       "oracle"),
    "nr2_ca_calcium_shot01_unified":   ("fly", "kernel+shot g=0.1", "gnn"),
    "nr2_ca_calcium_shot01_known_ode": ("fly", "kernel+shot g=0.1", "oracle"),
    "nr2_ca_calcium_shot02_unified":   ("fly", "kernel+shot g=0.2", "gnn"),
    "nr2_ca_calcium_shot02_known_ode": ("fly", "kernel+shot g=0.2", "oracle"),
    "nr2_ca_calcium_sat_unified":      ("fly", "kernel+saturation", "gnn"),
    "nr2_ca_calcium_rate25_unified":   ("fly", "kernel+rate 1/25",  "gnn"),
    "nr2_ca_calcium_full25_unified":   ("fly", "full obs model",    "gnn"),
    "nr2_ca_calcium_sat_known_ode":    ("fly", "kernel+saturation", "oracle"),
    "nr2_ca_calcium_rate25_known_ode": ("fly", "kernel+rate 1/25",  "oracle"),
    "nr2_ca_calcium_full25_known_ode": ("fly", "full obs model",    "oracle"),
    # Deconvolved observable: the calcium trace pushed back through the known
    # kernel, i.e. what an experimenter would hand a fitting pipeline.
    "nr2_ca_deconv_kernel_unified":    ("fly", "deconv kernel-only", "gnn"),
    "nr2_ca_deconv_kernel_known_ode":  ("fly", "deconv kernel-only", "oracle"),
    "nr2_ca_deconv_sat_unified":       ("fly", "deconv saturation",  "gnn"),
    "nr2_ca_deconv_sat_known_ode":     ("fly", "deconv saturation",  "oracle"),
}

# oracle arm -> GNN twin whose generate pass creates the shared dataset
DEFERRED = {
    "nr2_ca_calcium_sat_known_ode":    "nr2_ca_calcium_sat_unified",
    "nr2_ca_calcium_rate25_known_ode": "nr2_ca_calcium_rate25_unified",
    "nr2_ca_calcium_full25_known_ode": "nr2_ca_calcium_full25_unified",
}

WANT = [
    "W_corrected_no_outliers_R2", "W_corrected_R2", "W_corrected_no_outliers_slope",
    "rollout_pearson", "rollout_RMSE",
    "tau_R2", "tau_no_outliers_R2", "tau_n_outliers",
    "V_rest_R2", "V_rest_no_outliers_R2", "V_rest_n_outliers",
    "clustering_accuracy",
]


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def run_dir(cfg: str) -> Path:
    return ROOT / "log" / RUNS[cfg][0] / cfg


def is_complete(cfg: str) -> bool:
    """Terminal only on the marker AND a parseable metrics file."""
    d = run_dir(cfg)
    return (d / "_complete").exists() and (d / "results" / "metrics.txt").exists()


def read_metrics(cfg: str) -> dict[str, str]:
    out: dict[str, str] = {}
    p = run_dir(cfg) / "results" / "metrics.txt"
    try:
        for line in p.read_text().splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            if k.strip() in WANT:
                out[k.strip()] = v.strip()
    except OSError as exc:
        log(f"  metrics unreadable for {cfg}: {exc}")
    return out


def ssh(cmd: str, timeout: int = 90) -> str | None:
    """Return stdout, or None on any failure. None means UNKNOWN, never done."""
    try:
        r = subprocess.run(
            ["ssh", "login1", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError) as exc:
        log(f"  ssh failed ({exc.__class__.__name__}) -- treating as UNKNOWN")
        return None


def local_running() -> set[str]:
    try:
        r = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=30)
        return {c for c in RUNS if f"{c}.yaml" in r.stdout}
    except (subprocess.TimeoutExpired, OSError):
        return set()


def submit_deferred(submitted: set[str]) -> None:
    """Submit each oracle arm once its twin's dataset generation has landed.

    The guard is a marker file, not just the in-memory set: restarting the
    collector would otherwise re-submit an arm that is already queued, and two
    jobs writing one log dir corrupt each other. This happened once already.
    """
    for oracle, twin in DEFERRED.items():
        marker = CX / "neurips_review" / f".submitted_{oracle}"
        if oracle in submitted or marker.exists() or is_complete(oracle):
            continue
        if not (run_dir(twin) / "_completed_generate").exists():
            continue
        cmd = (
            f"cd {CA_REMOTE} && source /etc/profile.d/profile.lsf.sh && "
            f'bsub -n 4 -gpu "num=1" -q gpu_a100 -W 6000 '
            f'"conda run -n connectome-gnn python GNN_Main.py -o train_test_plot '
            f'config/fly/{oracle}.yaml --output_root {ROOT} --force"'
        )
        out = ssh(cmd, timeout=180)
        if out and "is submitted" in out:
            submitted.add(oracle)
            marker.write_text(out.strip().splitlines()[-1] + "\n")
            log(f"  SUBMITTED deferred oracle arm {oracle}: {out.strip().splitlines()[-1]}")
        else:
            log(f"  deferred submit for {oracle} did not confirm -- will retry next pass")


def write_outputs(done: dict[str, dict[str, str]], submitted: set[str]) -> None:
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "biomodel", "arm", "model"] + WANT)
        for cfg, m in sorted(done.items()):
            bio, arm, model = RUNS[cfg]
            w.writerow([cfg, bio, arm, model] + [m.get(k, "") for k in WANT])

    lines = [
        "# Overnight run status",
        "",
        f"Updated {datetime.now():%Y-%m-%d %H:%M:%S}. "
        f"{len(done)}/{len(RUNS)} finished.",
        "",
        "| arm | model | R2_W | slope | rollout r | tau R2 (inlier) | Vrest R2 (inlier) | clust |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cfg, m in sorted(done.items(), key=lambda kv: (RUNS[kv[0]][0], kv[0])):
        _, arm, model = RUNS[cfg]
        lines.append(
            f"| {arm} | {model} | {m.get('W_corrected_no_outliers_R2','')} | "
            f"{m.get('W_corrected_no_outliers_slope','')} | {m.get('rollout_pearson','')} | "
            f"{m.get('tau_no_outliers_R2','')} | {m.get('V_rest_no_outliers_R2','')} | "
            f"{m.get('clustering_accuracy','')} |"
        )
    pending = [c for c in RUNS if c not in done]
    if pending:
        lines += ["", "## Not yet finished", ""]
        lines += [
            f"- {c}"
            + ("  (deferred oracle, submitted)" if c in submitted else "")
            + ("  (deferred oracle, awaiting dataset)" if c in DEFERRED and c not in submitted else "")
            for c in sorted(pending)
        ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    done: dict[str, dict[str, str]] = {}
    submitted: set[str] = set()
    deadline = datetime.now() + timedelta(hours=MAX_HOURS)
    log(f"collector start: {len(RUNS)} runs tracked, deadline {deadline:%H:%M}")

    while datetime.now() < deadline:
        for cfg in RUNS:
            if cfg in done or not is_complete(cfg):
                continue
            m = read_metrics(cfg)
            if not m:
                continue  # marker present but metrics not flushed yet
            done[cfg] = m
            log(f"  DONE {cfg}: R2_W={m.get('W_corrected_no_outliers_R2','?')} "
                f"rollout_r={m.get('rollout_pearson','?')}")

        submit_deferred(submitted)
        write_outputs(done, submitted)

        if len(done) == len(RUNS):
            log("all tracked runs finished")
            return 0

        log(f"  {len(done)}/{len(RUNS)} done; local still running: "
            f"{sorted(local_running()) or 'none'}")
        time.sleep(POLL_S)

    log(f"deadline reached with {len(done)}/{len(RUNS)} finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
