#!/usr/bin/env python
"""Throughput benchmark with a bit-identity gate, for the message-passing rewrite.

An optimisation is only interesting if it changes the wall clock and NOTHING else.
This runs `GNN_Main.py -o train <spec>` on a chosen queue, stops it once the
metrics row at iteration 16001 exists, and records two things:

  SPEED     it/s, taken from tqdm's own smoothed rate rather than
            (iterations / total elapsed), which would be dragged down by the
            torch.compile warm-up and the 8 GB dataset load.
  IDENTITY  the metrics.log rows at iteration 1 and 16001. Those are the five
            numbers the run is judged on:
                connectivity_r2, vrest_r2_clean, n_out_vrest,
                tau_r2_clean, n_out_tau
            The reference on flyvis_noise_005_nominal is
                it 1      conn=-0.016496  Vr=0.755660 (13701/13741)  tau=-7.511984 (12894/13741)
                it 16001  conn= 0.950012  Vr=0.590809 ( 2535/13741)  tau= 0.901986 (  222/13741)

16001 is not arbitrary: `early_r2_frequency = (n_iter // 20) // 5` = 16000 at the
nominal 1.6M iterations, so it is the first checkpoint after the start row and it
arrives in ~5 min on an a100 and ~19 min on an l4.

Each invocation gets its own `--tag`, hence its own spec and its own log dir, so
an a100 and an l4 run of the same commit do not overwrite each other.

    python tools/bench_throughput.py --tag main_a100 --queue gpu_a100
    python tools/bench_throughput.py --tag main_l4   --queue gpu_l4 --compile false
    python tools/bench_throughput.py --compare main_a100 fused_a100
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO, "config", "fly")
LOG_ROOT = os.environ.get(
    "GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"
)
BENCH_DIR = os.path.join(REPO, "tools", "bench_results")
CLUSTER_CWD = "~/Graph/connectome-gnn"
# The interpreter by absolute path, not `python`. A bsub launched over ssh gets a
# non-interactive shell with no conda activation, so bare `python` resolves to
# miniforge3/bin/python and the job dies in 2 s on `import matplotlib` -- which
# LSF reports as exit code 1 with no other clue.
CLUSTER_PYTHON = "/groups/saalfeld/home/allierc/miniforge3/envs/connectome-gnn/bin/python"

GATE_COLUMNS = ("connectivity_r2", "vrest_r2_clean", "n_out_vrest",
                "tau_r2_clean", "n_out_tau")
GATE_ITERATIONS = (1, 16001)


# ----------------------------------------------------------------- spec + submit

def write_spec(base, tag, compile_flag):
    """A bench spec is the base spec under a new name, so it gets its own log dir."""
    src = os.path.join(CONFIG_DIR, base + ".yaml")
    name = f"flyvis_noise_005_bench_{tag}"
    dst = os.path.join(CONFIG_DIR, name + ".yaml")
    lines = open(src).read().split("\n")
    out = []
    for ln in lines:
        if compile_flag is not None and re.match(r"^  torch_compile:", ln):
            ln = f"  torch_compile: {str(compile_flag).lower()}"
        if ln.startswith("config_file:"):
            ln = f"config_file: fly/{name}"
        out.append(ln)
    open(dst, "w").write("\n".join(out))
    return name


def submit(spec, queue, walltime, stdout_rel):
    # -o must be a path the CLUSTER can write: there is no /workspace there, the
    # repo is ~/Graph/connectome-gnn on the same shared filesystem. LSF resolves a
    # relative -o against the submission cwd, so keep it relative and let the
    # `cd` do the work -- an absolute /workspace/... path makes bsub exit 1 with
    # no output file at all, which reads as "the training crashed".
    cmd = (f"cd {CLUSTER_CWD} && bsub -n 8 -gpu \"num=1\" -q {queue} -W {walltime} "
           f"-o {stdout_rel} \"{CLUSTER_PYTHON} GNN_Main.py -o train {spec}\"")
    r = subprocess.run(["ssh", "allierc@login1", cmd],
                       capture_output=True, text=True, timeout=180)
    m = re.search(r"Job <(\d+)>", r.stdout)
    if not m:
        raise RuntimeError(f"bsub gave no job id:\n{r.stdout}\n{r.stderr}")
    return m.group(1)


def bkill(job_id):
    subprocess.run(["ssh", "allierc@login1", f"bkill {job_id}"],
                   capture_output=True, text=True, timeout=120)


def job_is_alive(job_id):
    r = subprocess.run(["ssh", "allierc@login1", f"bjobs -o stat -noheader {job_id}"],
                       capture_output=True, text=True, timeout=120)
    return r.stdout.strip() in ("RUN", "PEND")


# ----------------------------------------------------------------- parsing

def read_gate(metrics_path):
    """The five judged numbers at each gate iteration, or None if not there yet."""
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path) as f:
        header = f.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in f if ln.strip()]
    idx = {c: header.index(c) for c in GATE_COLUMNS if c in header}
    if len(idx) != len(GATE_COLUMNS):
        raise RuntimeError(f"metrics.log is missing columns: "
                           f"{set(GATE_COLUMNS) - set(idx)}")
    it_col = header.index("iteration")
    out = {}
    for r in rows:
        it = int(r[it_col])
        if it in GATE_ITERATIONS:
            out[it] = {c: r[idx[c]] for c in GATE_COLUMNS}
    return out


def read_rate(stdout_path):
    """tqdm's smoothed it/s. Returns (last, median_of_last_20, n_samples).

    tqdm rewrites its bar with \\r, so every update is still in the file; that
    gives many samples of the CURRENT rate rather than one cumulative average.
    """
    if not os.path.exists(stdout_path):
        return None, None, 0
    txt = open(stdout_path, errors="replace").read()
    rates = [float(m) for m in re.findall(r",\s*([\d.]+)it/s", txt)]
    if not rates:
        # tqdm flips to s/it when slower than 1 it/s
        rates = [1.0 / float(m) for m in re.findall(r",\s*([\d.]+)s/it", txt)
                 if float(m) > 0]
    if not rates:
        return None, None, 0
    tail = sorted(rates[-20:])
    return rates[-1], tail[len(tail) // 2], len(rates)


# ----------------------------------------------------------------- run

def run(args):
    os.makedirs(BENCH_DIR, exist_ok=True)
    spec = write_spec(args.base_config, args.tag, args.compile)
    log_dir = os.path.join(LOG_ROOT, "log", "fly", spec)
    metrics = os.path.join(log_dir, "tmp_training", "metrics.log")
    stdout_rel = os.path.join("tools", "bench_results", f"{args.tag}.out")
    stdout_path = os.path.join(REPO, stdout_rel)

    # A stale log dir would let read_gate() see the PREVIOUS run's 16001 row and
    # declare victory before this run has taken a single step.
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir, ignore_errors=True)
    for p in (stdout_path,):
        if os.path.exists(p):
            os.remove(p)

    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    job_id = submit(spec, args.queue, args.walltime, stdout_rel)
    print(f"[{args.tag}] job {job_id} on {args.queue}, spec {spec}, "
          f"{branch}@{commit}", flush=True)

    t0 = time.time()
    gate = {}
    try:
        while True:
            time.sleep(args.poll)
            gate = read_gate(metrics)
            last, med, n = read_rate(stdout_path)
            elapsed = time.time() - t0
            have = sorted(gate)
            print(f"[{args.tag}] {elapsed/60:6.1f} min  rows={have}  "
                  f"rate={med if med else '-'} it/s (n={n})", flush=True)
            if all(it in gate for it in GATE_ITERATIONS):
                break
            if not job_is_alive(job_id):
                print(f"[{args.tag}] job ended before iteration "
                      f"{GATE_ITERATIONS[-1]}", flush=True)
                break
            if elapsed > args.timeout * 60:
                print(f"[{args.tag}] timeout after {args.timeout} min", flush=True)
                break
    finally:
        bkill(job_id)

    last, med, n = read_rate(stdout_path)
    result = {
        "tag": args.tag, "queue": args.queue, "spec": spec,
        "branch": branch, "commit": commit, "job_id": job_id,
        "torch_compile": args.compile,
        "rate_last": last, "rate_median_tail": med, "rate_samples": n,
        "minutes_to_gate": round((time.time() - t0) / 60, 1),
        "gate": {str(k): v for k, v in sorted(gate.items())},
    }
    path = os.path.join(BENCH_DIR, f"{args.tag}.json")
    json.dump(result, open(path, "w"), indent=2)
    print(f"[{args.tag}] wrote {path}", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return result


# ----------------------------------------------------------------- compare

def compare(tags):
    res = []
    for t in tags:
        p = os.path.join(BENCH_DIR, f"{t}.json")
        if not os.path.exists(p):
            sys.exit(f"no result for tag {t!r} ({p})")
        res.append(json.load(open(p)))

    print(f"\n{'tag':<18}{'queue':<10}{'commit':<10}{'compile':<9}"
          f"{'it/s':>8}{'min':>7}")
    for r in res:
        print(f"{r['tag']:<18}{r['queue']:<10}{r['commit']:<10}"
              f"{str(r['torch_compile']):<9}"
              f"{(r['rate_median_tail'] or 0):>8.2f}{r['minutes_to_gate']:>7.1f}")

    ref = res[0]
    print(f"\nbit-identity vs {ref['tag']}:")
    ok = True
    for r in res[1:]:
        for it in map(str, GATE_ITERATIONS):
            a, b = ref["gate"].get(it), r["gate"].get(it)
            if a is None or b is None:
                print(f"  {r['tag']:<18} it {it:<6} MISSING"); ok = False; continue
            bad = [c for c in GATE_COLUMNS if a[c] != b[c]]
            if bad:
                ok = False
                print(f"  {r['tag']:<18} it {it:<6} DIFFERS on {bad}")
                for c in bad:
                    print(f"      {c}: {a[c]}  ->  {b[c]}")
            else:
                print(f"  {r['tag']:<18} it {it:<6} identical "
                      f"(conn={a['connectivity_r2']}, Vr={a['vrest_r2_clean']}, "
                      f"tau={a['tau_r2_clean']})")
        speed = (r["rate_median_tail"] or 0) / (ref["rate_median_tail"] or 1)
        print(f"  {r['tag']:<18} speedup {speed:.2f}x")
    print("\nGATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag")
    p.add_argument("--queue", default="gpu_a100", choices=["gpu_a100", "gpu_l4"])
    p.add_argument("--base-config", default="flyvis_noise_005_nominal")
    p.add_argument("--compile", type=lambda s: s.lower() == "true", default=None,
                   help="override torch_compile in the bench spec")
    p.add_argument("--walltime", default="4:00")
    p.add_argument("--poll", type=int, default=60, help="seconds between polls")
    p.add_argument("--timeout", type=int, default=90, help="minutes")
    p.add_argument("--compare", nargs="+", metavar="TAG",
                   help="compare saved results; the first tag is the reference")
    a = p.parse_args()

    if a.compare:
        sys.exit(compare(a.compare))
    if not a.tag:
        p.error("--tag is required unless --compare is given")
    run(a)


if __name__ == "__main__":
    main()
