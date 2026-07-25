#!/usr/bin/env python
"""End-to-end autonomous finisher (run in background).

1. Wait until the 7 dt/adapt datasets are generated locally.
2. Submit the 14 dt/adapt train+test+plot jobs to gpu_a100 (Test 2 mono jobs are
   submitted separately and merged in via submitted_jobs.json).
3. Poll _complete markers for all 17 runs until done (or timeout).
4. Collect metrics into results_table.csv.

Idempotent: skips datasets already present, skips configs already complete.
Run:
  nohup /workspace/.conda_envs/neural-graph-linux/bin/python \
    neurips_review/finish_pipeline.py > neurips_review/_finish.log 2>&1 &
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import launch_cluster as L  # noqa: E402

ROOT = L.ROOT
DATA_MAX_H = 14.0      # max wait for local generation
TRAIN_MAX_H = 16.0     # max wait for cluster training
POLL = 180


def dataset_ready(ds):
    # Gate on ode_params.pt (written LAST, at end of generation) not voltage.zarr
    # (written early) — else training can race ahead of ground-truth params.
    d = f"{ROOT}/graphs_data/fly/{ds}"
    return (os.path.exists(f"{d}/ode_params.pt")
            and os.path.exists(f"{d}/x_list_train/voltage.zarr/.zarray"))


def log(msg):
    print(f"[finish {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    man = json.load(open(os.path.join(HERE, "manifest.json")))
    dt_adapt = [t for t in man["train"] if t["test"] in ("dt", "adapt")]
    all_cfgs = [t["config"] for t in man["train"]]

    sj = os.path.join(HERE, "submitted_jobs.json")
    submitted = {k: v for k, v in json.load(open(sj)).items() if v} if os.path.exists(sj) else {}

    # 1+2. Incrementally submit each dt/adapt job as soon as its dataset is ready
    #      (overlaps local generation with cluster training). Loop until every
    #      dt/adapt config is submitted or its data timed out.
    t0 = time.time()
    while (time.time() - t0) < DATA_MAX_H * 3600:
        pending = [t for t in dt_adapt
                   if t["config"] not in submitted and not L.complete_marker(t["config"])]
        if not pending:
            break
        for t in pending:
            c, ds = t["config"], t["dataset"]
            if dataset_ready(ds):
                jid, raw = L.submit(c)
                submitted[c] = jid
                json.dump(submitted, open(sj, "w"), indent=2)
                log(f"submitted {c}: job {jid or '??'} {'' if jid else raw[:160]}")
                time.sleep(1)
        still = [t["dataset"] for t in dt_adapt
                 if t["config"] not in submitted and not L.complete_marker(t["config"])]
        if still:
            log(f"submitted {len(submitted)} so far; waiting on datasets: {sorted(set(still))}")
            time.sleep(POLL)
    else:
        log("TIMEOUT waiting for datasets; proceeding to poll whatever was submitted")

    # 3. poll all 17 _complete markers
    t0 = time.time()
    while (time.time() - t0) < TRAIN_MAX_H * 3600:
        done = [c for c in all_cfgs if L.complete_marker(c)]
        if len(done) == len(all_cfgs):
            log("ALL 17 runs complete")
            break
        log(f"{len(done)}/{len(all_cfgs)} complete; waiting")
        time.sleep(POLL)
    else:
        log(f"TIMEOUT: only {len(done)}/{len(all_cfgs)} complete after {TRAIN_MAX_H}h")

    # 4. collect
    log("collecting metrics")
    subprocess.run([sys.executable, os.path.join(HERE, "collect_metrics.py")])
    log("DONE")


if __name__ == "__main__":
    main()
