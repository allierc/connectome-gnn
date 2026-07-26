#!/usr/bin/env python
"""Submit the 17 train+test+plot jobs to the LSF cluster (one bsub per config),
poll to completion, then collect metrics.

Mirrors the repo's working submission pattern (src/connectome_gnn/LLM/cluster.py):
ssh to the login node, source the LSF profile, bsub a `GNN_Main.py -o
train_test_plot` payload in the shared cluster checkout with --output_root on the
shared GraphData filesystem (where our locally-generated data + configs live).

  /workspace/.conda_envs/neural-graph-linux/bin/python neurips_review/launch_cluster.py
  # options: --dry-run (print bsub cmds, submit nothing) ; --no-wait (submit only)
"""
import argparse
import json
import os
import re
import subprocess
import time

ROOT = "/groups/saalfeld/home/allierc/GraphData"
CONFIG_DIR = os.path.join(ROOT, "config", "fly")
CLUSTER_SSH = "allierc@login1"
CLUSTER_ROOT = "/groups/saalfeld/home/allierc/Graph/connectome-gnn-cx"
CONDA_ENV = "connectome-gnn"           # cluster training env (matches LLM pipeline default)
QUEUE = "gpu_a100"
NCPU = 4
WALL_MIN = 6000
JOB_LOG_DIR = os.path.join(ROOT, "log", "neurips_review_jobs")
HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd, timeout=90):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def bsub_cmd(config):
    cfg_abs = os.path.join(CONFIG_DIR, config + ".yaml")
    out = os.path.join(JOB_LOG_DIR, f"{config}.out")
    err = os.path.join(JOB_LOG_DIR, f"{config}.err")
    payload = (
        f"conda run -n {CONDA_ENV} python GNN_Main.py -o train_test_plot "
        f"{cfg_abs} --output_root {ROOT} --force"
    )
    inner = (
        f"cd {CLUSTER_ROOT} && source /etc/profile.d/profile.lsf.sh && "
        f"mkdir -p {JOB_LOG_DIR} && "
        f"bsub -n {NCPU} -gpu \\\"num=1\\\" -q {QUEUE} -W {WALL_MIN} "
        f"-o {out} -e {err} \\\"{payload}\\\""
    )
    return f"ssh {CLUSTER_SSH} \"bash -l -c '{inner}'\""


def log_dir(config):
    return os.path.join(ROOT, "log", "fly", config)


def submit(config):
    rc, out, err = sh(bsub_cmd(config))
    m = re.search(r"Job <(\d+)>", out + err)
    jid = m.group(1) if m else None
    return jid, (out + err).strip()


def poll_jobs(job_ids, timeout=90):
    ids = " ".join(job_ids)
    rc, out, err = sh(
        f"ssh {CLUSTER_SSH} \"source /etc/profile.d/profile.lsf.sh && bjobs {ids}\"",
        timeout=timeout,
    )
    status = {}
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            status[parts[0]] = parts[2]  # STAT column
    return status  # missing id => no longer in queue (treat as finished)


def complete_marker(config):
    return os.path.exists(os.path.join(log_dir(config), "_complete"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--test", nargs="+", default=None,
                    help="only submit these test groups (subset of: dt mono adapt)")
    ap.add_argument("--poll-secs", type=int, default=120)
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--manifest", default="manifest.json")
    args = ap.parse_args()

    man = json.load(open(args.manifest if os.path.isabs(args.manifest)
                         else os.path.join(HERE, args.manifest)))
    entries = man["train"]
    if args.test:
        entries = [t for t in entries if t["test"] in set(args.test)]
    configs = [t["config"] for t in entries]
    print(f"selected {len(configs)} configs" + (f" (test={args.test})" if args.test else ""))

    if args.dry_run:
        for c in configs:
            print(bsub_cmd(c))
            print()
        print(f"{len(configs)} jobs (dry-run, nothing submitted)")
        return 0

    sh(f"ssh {CLUSTER_SSH} \"mkdir -p {JOB_LOG_DIR}\"")

    submitted = {}  # config -> job_id
    for c in configs:
        jid, raw = submit(c)
        submitted[c] = jid
        print(f"[submit] {c}: job {jid or '??'}  {'' if jid else raw[:200]}")
        time.sleep(1)

    ok = [c for c, j in submitted.items() if j]
    print(f"\nsubmitted {len(ok)}/{len(configs)} jobs")
    json.dump(submitted, open(os.path.join(HERE, "submitted_jobs.json"), "w"), indent=2)

    if args.no_wait:
        return 0

    # poll
    t0 = time.time()
    id2cfg = {j: c for c, j in submitted.items() if j}
    remaining = set(id2cfg)
    while remaining and (time.time() - t0) < args.max_hours * 3600:
        time.sleep(args.poll_secs)
        try:
            st = poll_jobs(list(remaining))
        except subprocess.TimeoutExpired:
            print("[poll] bjobs timed out; retrying next cycle")
            continue
        done = []
        for jid in list(remaining):
            in_queue = jid in st and st[jid] not in ("DONE", "EXIT")
            if not in_queue:
                cfg = id2cfg[jid]
                fin = complete_marker(cfg)
                print(f"[done] {cfg} (job {jid}) stat={st.get(jid,'gone')} complete_marker={fin}")
                done.append(jid)
        remaining -= set(done)
        print(f"[poll] {len(remaining)} still running ({int((time.time()-t0)/60)} min elapsed)")

    if remaining:
        print(f"TIMEOUT: {len(remaining)} jobs still running after {args.max_hours}h")
    else:
        print("ALL JOBS FINISHED")

    # collect
    print("\n--- collecting metrics ---")
    subprocess.run([os.sys.executable, os.path.join(HERE, "collect_metrics.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
