"""Submit + live-monitor the Drosophila CX companion training runs.

Companion to ``run_GNN_drosophila_cx_pi_epg_no_tv_cv10.py`` but for the fresh
single-run task suite (drosophila_paper.tex): it ``bsub``-submits one GPU job
per config and then prints, **every 300 s**, the latest training-progress line
of each job — e.g.

    [drosophila_cx_gnn_rotation  job 1513xxxx RUN ] epoch 3/10 done  |  loss=0.0370 rmse_roll=31.8° r_roll=1.000 (0.861) best=0.0266

so you can watch convergence from one terminal without tailing logs by hand.

Each job runs ``python GNN_Main.py -o train_task <config>`` on the chosen GPU
queue (default l4). The configs ship n_epochs=10; with data_augmentation_loop=3
an RNN epoch is ~15 min on a100 / ~30 min on l4 (≈5 h for 10 epochs), so the
default wall-clock limit is generous.

Usage::

    python scripts/run_GNN_drosophila_cx_companion.py                 # whole suite on l4
    python scripts/run_GNN_drosophila_cx_companion.py --config drosophila_cx_gnn_rotation
    python scripts/run_GNN_drosophila_cx_companion.py --cluster a100 --config drosophila_cx_rotation drosophila_cx_both
    python scripts/run_GNN_drosophila_cx_companion.py --no-monitor    # submit only, don't watch
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The fresh companion suite (RNN + GNN), ordered heading-first (the GNN heading
# run is the fastest / most precise — a good convergence canary).
SUITE = [
    "drosophila_cx_gnn_rotation",      "drosophila_cx_rotation",
    "drosophila_cx_gnn_rotation_vfwd", "drosophila_cx_rotation_vfwd",
    "drosophila_cx_gnn_both",          "drosophila_cx_both",
    "drosophila_cx_gnn_both_leaky",    "drosophila_cx_both_leaky",
    "drosophila_cx_gnn_position_2d",   "drosophila_cx_position_2d",
    "drosophila_cx_gnn_position_2d_leaky", "drosophila_cx_position_2d_leaky",
    "drosophila_cx_rotation_mlpdec",   "drosophila_cx_rotation_kbins",
]

# tqdm postfix written by the trainer, e.g.
#   loss=0.0370 rmse_roll=31.8° r_roll=1.000 (0.861) best=0.0266
_PROG_RE = re.compile(r"loss=[-\d.]+.*?best=[-\d.]+")
_EPOCH_RE = re.compile(r"epoch \d+/\d+ done[^\n]*")


# LSF only exists on the cluster login nodes, reached over SSH (the devcontainer
# has no bsub at all). We run every LSF call through a login shell so the LSF
# profile + conda init are sourced; by default that login shell is on the
# cluster via `ssh <SSH_TARGET>`, mirroring connectome_gnn.LLM.cluster. Pass
# --local to skip the ssh hop (use when you are already ON a login node).
# Logs are written by the job to the shared /groups filesystem, which the
# devcontainer also mounts, so monitoring reads them directly.
def _cluster_cfg() -> dict:
    """Load cluster_user/login/root_dir/data_dir from data_paths.json (same
    source connectome_gnn.LLM.cluster reads), so the ssh target + cluster
    checkout path aren't hardcoded."""
    import json
    for c in (os.path.join(REPO, "data_paths.json"),
              os.path.join(os.getcwd(), "data_paths.json")):
        if os.path.isfile(c):
            return json.load(open(c))
    return {}


_CFG = _cluster_cfg()
SSH_TARGET = f"{_CFG.get('cluster_user', 'allierc')}@{_CFG.get('cluster_login', 'login1')}"
CLUSTER_REPO = _CFG.get("cluster_root_dir",
                        "/groups/saalfeld/home/allierc/Graph/connectome-gnn-cx")
CLUSTER_DATA = _CFG.get("cluster_data_dir",
                        "/groups/saalfeld/home/allierc/GraphData")


def _lsf(command: str, *, ssh: str | None) -> subprocess.CompletedProcess:
    """Run `command` in a cluster login shell (`bash -lc`), over ssh unless local."""
    if ssh:
        # ssh runs the single string argument through the remote login shell.
        return subprocess.run(
            ["ssh", ssh, f"bash -lc {_q(command)}"],
            capture_output=True, text=True)
    return subprocess.run(["bash", "-lc", command], capture_output=True, text=True)


def _q(s: str) -> str:
    """Single-quote for the remote shell (command contains no single quotes)."""
    return "'" + s.replace("'", "'\\''") + "'"


def _submit(cfg: str, *, cluster: str, n_cpus: int, w_min: int, log_dir: str,
            ssh: str | None, repo: str) -> tuple[str | None, str]:
    log = os.path.join(log_dir, f"{cfg}.lsf.log")
    # No `export`/`;` in the bsub command (keeps it a single program for LSF);
    # GNN_OUTPUT_ROOT is inherited from the login shell that bsub copies.
    cmd = (f"cd {repo} && bsub -n {n_cpus} -gpu num=1 -q gpu_{cluster} "
           f"-W {w_min} -o {log} -J {cfg} "
           f"python GNN_Main.py -o train_task {cfg}")
    out = _lsf(cmd, ssh=ssh)
    m = re.search(r"Job <(\d+)>", out.stdout)
    jid = m.group(1) if m else None
    err = (out.stderr or out.stdout).strip().splitlines()
    state = f"job {jid}" if jid else f"SUBMIT FAILED ({err[-1][:90] if err else '?'})"
    print(f"  {cfg:38s} -> {state}   log: {log}")
    return jid, log


def _job_state(jid: str, *, ssh: str | None) -> str:
    """RUN / PEND / DONE / EXIT — DONE once the job leaves the queue."""
    r = _lsf(f"bjobs {jid}", ssh=ssh)
    for tok in ("RUN", "PEND", "EXIT", "DONE"):
        if tok in r.stdout:
            return tok
    return "DONE"  # no longer in bjobs -> finished


def _latest(log: str) -> tuple[str, str]:
    if not os.path.isfile(log):
        return "(pending)", ""
    txt = open(log, errors="ignore").read().replace("\r", "\n")
    prog = _PROG_RE.findall(txt)
    epoch = _EPOCH_RE.findall(txt)
    return (epoch[-1] if epoch else "(starting)"), (prog[-1] if prog else "")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", nargs="+", default=None,
                   help="config name(s) to run (default: the whole companion suite)")
    p.add_argument("--cluster", choices=["l4", "a100", "h100"], default="l4")
    p.add_argument("--n-cpus", type=int, default=8)
    p.add_argument("--hard-runtime-min", type=int, default=600,
                   help="bsub -W wall-clock limit (default 600 = 10 h; 10 epochs "
                        "x ~30 min/epoch on l4 + margin)")
    p.add_argument("--interval", type=int, default=300,
                   help="seconds between progress prints (default 300)")
    p.add_argument("--ssh", default=SSH_TARGET,
                   help=f"cluster login for bsub/bjobs (default {SSH_TARGET} "
                        f"from data_paths.json)")
    p.add_argument("--local", action="store_true",
                   help="run bsub/bjobs locally (use when already ON a login node)")
    p.add_argument("--cluster-repo", default=CLUSTER_REPO,
                   help=f"cluster checkout to bsub from (default {CLUSTER_REPO})")
    p.add_argument("--output_root", default=CLUSTER_DATA,
                   help="GraphData root where job logs are written (shared /groups)")
    p.add_argument("--no-monitor", dest="monitor", action="store_false",
                   help="submit the jobs and exit without the 300 s progress loop")
    args = p.parse_args()

    ssh = None if args.local else args.ssh
    configs = args.config or SUITE
    log_dir = os.path.join(args.output_root, "log", "drosophila_cx",
                           "_companion_runner")
    os.makedirs(log_dir, exist_ok=True)

    where = "locally" if ssh is None else f"via ssh {ssh}"
    print(f"submitting {len(configs)} job(s) to gpu_{args.cluster} {where} "
          f"(cd {args.cluster_repo}; -W {args.hard_runtime_min} min):")
    jobs = {}  # cfg -> (jid, log)
    for cfg in configs:
        jid, log = _submit(cfg, cluster=args.cluster, n_cpus=args.n_cpus,
                           w_min=args.hard_runtime_min, log_dir=log_dir,
                           ssh=ssh, repo=args.cluster_repo)
        if jid:
            jobs[cfg] = (jid, log)

    if not args.monitor or not jobs:
        print("submitted; not monitoring." if not args.monitor
              else "no jobs submitted.")
        return 0

    print(f"\nmonitoring every {args.interval}s (Ctrl-C to stop watching; "
          f"jobs keep running):\n")
    try:
        while jobs:
            time.sleep(args.interval)
            stamp = time.strftime("%H:%M:%S")
            done = []
            for cfg, (jid, log) in jobs.items():
                st = _job_state(jid, ssh=ssh)
                epoch, prog = _latest(log)
                line = f"[{stamp}] {cfg:38s} job {jid} {st:4s} | {epoch}"
                if prog:
                    line += f"  |  {prog}"
                print(line, flush=True)
                if st in ("DONE", "EXIT"):
                    done.append(cfg)
            for cfg in done:
                jobs.pop(cfg)
            if done:
                print(f"  ({len(done)} finished; {len(jobs)} still running)\n")
    except KeyboardInterrupt:
        print("\nstopped watching; jobs continue on the cluster "
              "(bjobs / bkill to manage).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
