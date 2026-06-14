"""Submit + live-monitor the zebrafish HD/IPN companion training runs.

Zebrafish twin of ``run_GNN_drosophila_cx_companion.py``: it ``bsub``-submits
one GPU job per ``config/zebrafish/*.yaml`` and then prints, **every 300 s**,
the latest training-progress line of each job — e.g.

    [zebrafish_hd_si_gnn_ipn_917_v1_propriocep_distance  job 1513xxxx RUN ] epoch 3/5 done  |  loss=0.0370 rmse_roll=31.8° r_roll=1.000 (0.861) best=0.0266

so you can watch convergence from one terminal without tailing logs by hand.

By default the auto-discovered set SKIPS configs that already have a checkpoint
(only untrained ones run); pass --rerun-trained to include them, or name configs
explicitly with --config.

Each job runs ``python GNN_Main.py -o train_task <config>`` on the chosen GPU
queue (default l4).

Usage::

    python scripts/run_GNN_zebrafish_companion.py                 # untrained config/zebrafish/*.yaml on l4
    python scripts/run_GNN_zebrafish_companion.py --config zebrafish_hd_si_ipn_917_v1_propriocep_distance
    python scripts/run_GNN_zebrafish_companion.py --cluster a100 --rerun-trained
    python scripts/run_GNN_zebrafish_companion.py --no-monitor    # submit only, don't watch
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUP = "zebrafish"   # config/<GROUP>/, log/<GROUP>/, graphs_data/<GROUP>/

# ANSI colours for the terminal metrics print.
_R = "\033[0m"; _B = "\033[1m"; _DIM = "\033[2m"
_GRN = "\033[92m"; _YEL = "\033[93m"; _RED = "\033[91m"; _CYN = "\033[96m"; _BLU = "\033[94m"
_STATE_COL = {"RUN": _GRN, "PEND": _YEL, "EXIT": _RED, "DONE": _BLU}

# Heading-first ordering hint (the GNN heading run is the fastest / most precise
# convergence canary). Configs not listed here are appended alphabetically, so
# the default (no --config) runs EVERY config/zebrafish/*.yaml.
_ORDER_HINT = [
    # NEW — proprioception + translation (run these first; most else is trained
    # and skipped by default).
    "zebrafish_hd_si_ipn_917_v1_propriocep_distance",
    "zebrafish_hd_si_gnn_ipn_917_v1_propriocep_distance",
    "zebrafish_hd_si_ipn_917_v1_propriocep_distance_leaky",
    "zebrafish_hd_si_gnn_ipn_917_v1_propriocep_distance_leaky",
    "zebrafish_hd_si_ipn_917_v1_propriocep_position_2d",
    "zebrafish_hd_si_gnn_ipn_917_v1_propriocep_position_2d",
    "zebrafish_hd_si_ipn_917_v1_propriocep_position_2d_leaky",
    "zebrafish_hd_si_gnn_ipn_917_v1_propriocep_position_2d_leaky",
    # core self-motion suite (RNN + GNN).
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation", "zebrafish_hd_si_ipn_917_v1_selfmotion_rotation",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_both",     "zebrafish_hd_si_ipn_917_v1_selfmotion_both",
    "zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_both_leaky", "zebrafish_hd_si_ipn_917_v1_selfmotion_both_leaky",
    "zebrafish_hd_si_ipn_917_v1_position_2d",             "zebrafish_hd_si_ipn_917_v1_position_2d_leaky",
    "zebrafish_hd_si_ipn_917_v1_propriocep_mismatch",
]

# ---------------------------------------------------------------------------
# Config inventory (status note as of 2026-06-13). The runner auto-discovers
# EVERY config/zebrafish/*.yaml and SKIPS already-trained ones by default; this
# is a record of what is trained vs. what still needs a run.
#
# ALREADY TRAINED (checkpoints in log/zebrafish/<cfg>/models/):
#   the 917-cell artr_pt1 self-motion suite (RNN + GNN): selfmotion_rotation,
#   selfmotion_both(_leaky), position_2d(_leaky), rotation_kbins / _mlpdec,
#   the clustermove GNN variants, the bs / er / er_ei null controls, and the
#   proprioceptive-gain MISMATCH pair
#   (zebrafish_hd_si_ipn_917_v1_propriocep_mismatch + its GNN twin).
#
# NEW — PROPRIOCEPTION + TRANSLATION (pen_artr_ptipn1_propriocep gate; NOT yet
# trained). The proprioception circuit (zebrafish_HD_IPN_917_artr_pt1_
# proprioception) with omega -> ARTR, exteroceptive v_fwd -> pt-IPN1, efference
# omega_proprio=omega -> motor_efferent; 5-channel stimulus
# [omega, v_fwd, omega_proprio, cos0, sin0]. RNN = zebrafish_hd_si,
# GNN = zebrafish_hd_si_gnn (gate ported in this change).
#   forward distance (target_kind=scalar_xi):
#     zebrafish_hd_si_ipn_917_v1_propriocep_distance        / ..._gnn_..._propriocep_distance         (cumulative)
#     zebrafish_hd_si_ipn_917_v1_propriocep_distance_leaky  / ..._gnn_..._propriocep_distance_leaky   (leaky, tau_d=0.5 s)
#   2-D position (target_kind=position_2d):
#     zebrafish_hd_si_ipn_917_v1_propriocep_position_2d     / ..._gnn_..._propriocep_position_2d      (cumulative)
#     zebrafish_hd_si_ipn_917_v1_propriocep_position_2d_leaky/..._gnn_..._propriocep_position_2d_leaky(leaky, tau_p=0.5 s)
#   Each RNN/GNN pair shares one dataset (zebrafish_hd_si_task_917_propriocep_*),
#   generated on first run.
# ---------------------------------------------------------------------------


def _dataset_of(cfg: str) -> str | None:
    """The on-disk dataset name from config/<GROUP>/<cfg>.yaml (or None)."""
    import yaml
    y = os.path.join(REPO, "config", GROUP, cfg + ".yaml")
    try:
        return (yaml.safe_load(open(y)) or {}).get("dataset")
    except Exception:
        return None


def _dataset_dir(cfg: str, output_root: str) -> str | None:
    ds = _dataset_of(cfg)
    if not ds:
        return None
    return os.path.join(output_root, "graphs_data", GROUP, ds)


def _generate_missing(configs, output_root: str) -> None:
    """`-o train_task` does NOT generate data, so a config whose dataset is not
    on disk would EXIT immediately at the data-load. Generate each unique
    missing dataset locally (shared /groups, so the cluster jobs see it) BEFORE
    submitting. Deduped by dataset name (RNN+GNN of a variant share one)."""
    need = {}  # dataset -> first cfg that builds it
    for cfg in configs:
        d = _dataset_dir(cfg, output_root)
        if d and not os.path.isdir(os.path.join(d, "train")):
            need.setdefault(os.path.basename(d), cfg)
    if not need:
        return
    print(f"{_YEL}generating {len(need)} missing dataset(s) locally before "
          f"submit (--no-generate to skip):{_R}")
    env = {**os.environ, "PYTHONPATH": os.path.join(REPO, "src"),
           "GNN_OUTPUT_ROOT": output_root}
    for ds, cfg in need.items():
        print(f"  {_CYN}{ds}{_R} (via {cfg}) ...", end="", flush=True)
        subprocess.run([sys.executable, "GNN_Main.py", "-o", "generate", cfg],
                       cwd=REPO, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ok = os.path.isdir(os.path.join(output_root, "graphs_data", GROUP, ds,
                                        "train"))
        print(f" {_GRN}OK{_R}" if ok else f" {_RED}FAILED{_R}")


def _is_trained(cfg: str, output_root: str) -> bool:
    """True if a finished checkpoint already exists for this config. The job
    writes to the shared /groups log dir, which the devcontainer also mounts,
    so the glob resolves locally."""
    import glob
    patt = os.path.join(output_root, "log", "zebrafish", cfg, "models",
                        "best_model_with_*.pt")
    return bool(glob.glob(patt))


def _all_configs() -> list[str]:
    """Every config/zebrafish/*.yaml (top level, not archive/), ordered with
    the core suite first then the rest (nulls, rep folds, ...) alphabetically."""
    import glob
    d = os.path.join(REPO, "config", "zebrafish")
    names = sorted(os.path.splitext(os.path.basename(f))[0]
                   for f in glob.glob(os.path.join(d, "*.yaml")))
    if not names:
        return list(_ORDER_HINT)
    head = [c for c in _ORDER_HINT if c in names]
    rest = [c for c in names if c not in head]
    return head + rest

# Parse the trainer's log lines, e.g.
#   start training: 10 epochs × 31250 iters/epoch ...
#   epoch 3 (T=200):  74%|...| 1157/1562 [06:32<02:14, ..., loss=0.0098 ... best=0.0043]
#   epoch 3/10 done — ...
_PROG_RE = re.compile(r"loss=[-\d.]+.*?best=[-\d.]+")
_TOTEP_RE = re.compile(r"start training:\s*(\d+)\s+epochs")
_EPTQDM_RE = re.compile(r"epoch (\d+)\s*\(T=")          # current epoch (tqdm header)
_DONE_RE = re.compile(r"epoch (\d+)/(\d+) done")        # last completed epoch / total
_ITER_RE = re.compile(r"(\d+)/(\d+) \[")                # tqdm iter / total


def _k(n) -> str:
    n = int(n)
    return f"{n / 1000:.1f}K" if n >= 1000 else str(n)


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
            ssh: str | None, repo: str, conda_env: str) -> tuple[str | None, str]:
    log = os.path.join(log_dir, f"{cfg}.lsf.log")
    # The bsub JOB runs `bash -lc 'conda run -n <env> python ...'` so the
    # compute node sources conda itself and runs in the right env — robust to
    # LSF not propagating the submission environment (a plain `python` job
    # lands in base and dies on ModuleNotFoundError: matplotlib). Mirrors
    # connectome_gnn.LLM.cluster (bash -l + conda run). --no-capture-output
    # keeps tqdm streaming to the log so the 300 s monitor can read progress.
    job = (f"cd {repo} && conda run --no-capture-output -n {conda_env} "
           f"python GNN_Main.py -o train_task {cfg}")
    cmd = (f"cd {repo} && bsub -n {n_cpus} -gpu num=1 -q gpu_{cluster} "
           f"-W {w_min} -oo {log} -J {cfg} bash -lc {shlex.quote(job)}")
    out = _lsf(cmd, ssh=ssh)
    m = re.search(r"Job <(\d+)>", out.stdout)
    jid = m.group(1) if m else None
    err = (out.stderr or out.stdout).strip().splitlines()
    state = (f"{_GRN}job {jid}{_R}" if jid
             else f"{_RED}SUBMIT FAILED ({err[-1][:90] if err else '?'}){_R}")
    print(f"  {_CYN}{cfg:38s}{_R} -> {state}")
    return jid, log


# LSF/ssh transient failures that must NOT be mistaken for "job finished".
_LSF_TRANSIENT = ("cannot connect", "not responding", "batch system",
                  "please wait", "try again", "connection closed",
                  "connection timed out", "timed out", "broken pipe",
                  "connection refused", "no route to host")
_LSF_STATES = ("RUN", "PEND", "DONE", "EXIT", "SSUSP", "PSUSP", "USUSP")


def _job_states(jids, *, ssh: str | None) -> dict:
    """Batched jid -> RUN/PEND/DONE/EXIT/SUSP/UNKNOWN via ONE bjobs call.

    One query per poll (not one ssh per job): with dozens of jobs, a separate
    `ssh ... bjobs` each would routinely hit a transient hiccup whose empty
    output the old code read as DONE -> jobs falsely reported 'finished' while
    still RUN. Here a job ABSENT from a healthy response has genuinely left the
    queue (-> DONE), but if the response looks like an ssh/LSF failure every
    job is marked UNKNOWN so the monitor keeps polling instead of false-finishing.
    """
    jids = list(jids)
    if not jids:
        return {}
    r = _lsf('bjobs -a -o "jobid stat" ' + " ".join(jids), ssh=ssh)
    blob = ((r.stdout or "") + "\n" + (r.stderr or "")).lower()
    transient = any(k in blob for k in _LSF_TRANSIENT)
    seen = {}
    for ln in (r.stdout or "").splitlines():
        p = ln.split()
        if len(p) >= 2 and p[0] in jids and p[1] in _LSF_STATES:
            seen[p[0]] = p[1]
    out = {}
    for j in jids:
        if j in seen:
            out[j] = seen[j]
        elif transient:
            out[j] = "UNKNOWN"          # query failed: keep polling
        else:
            out[j] = "DONE"             # absent from a healthy bjobs -a -> gone
    return out


def _latest(log: str) -> tuple[str, str]:
    """Return (status, progress) where status = 'epoch C/T  iterK/totK'."""
    if not os.path.isfile(log):
        return "(pending)", ""
    txt = open(log, errors="ignore").read().replace("\r", "\n")
    prog = _PROG_RE.findall(txt)
    prog = prog[-1] if prog else ""
    tot_ep = _TOTEP_RE.findall(txt)
    T = tot_ep[-1] if tot_ep else "?"
    cur = None
    if (tq := _EPTQDM_RE.findall(txt)):
        cur = int(tq[-1])
    if (dn := _DONE_RE.findall(txt)):
        cur = max(cur or 0, int(dn[-1][0]))
    if cur is None:
        return "(starting)", prog
    it = _ITER_RE.findall(txt)
    status = f"epoch {cur}/{T}"
    if it:
        status += f"  {_k(it[-1][0])}/{_k(it[-1][1])}"
    return status, prog


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", nargs="+", default=None,
                   help="config name(s) to run (default: EVERY config/zebrafish/*.yaml)")
    p.add_argument("--cluster", choices=["l4", "a100", "h100"], default="l4")
    p.add_argument("--n-cpus", type=int, default=8)
    p.add_argument("--hard-runtime-min", type=int, default=1440,
                   help="bsub -W wall-clock limit in minutes (default 1440 = 24 h)")
    p.add_argument("--interval", type=int, default=300,
                   help="seconds between progress prints (default 300)")
    p.add_argument("--ssh", default=SSH_TARGET,
                   help=f"cluster login for bsub/bjobs (default {SSH_TARGET} "
                        f"from data_paths.json)")
    p.add_argument("--local", action="store_true",
                   help="run bsub/bjobs locally (use when already ON a login node)")
    p.add_argument("--cluster-repo", default=CLUSTER_REPO,
                   help=f"cluster checkout to bsub from (default {CLUSTER_REPO})")
    p.add_argument("--conda-env", default="connectome-gnn",
                   help="conda env activated before bsub so the job inherits its "
                        "python (default connectome-gnn)")
    p.add_argument("--output_root", default=CLUSTER_DATA,
                   help="GraphData root where job logs are written (shared /groups)")
    p.add_argument("--no-monitor", dest="monitor", action="store_false",
                   help="submit the jobs and exit without the 300 s progress loop")
    p.add_argument("--rerun-trained", action="store_true",
                   help="also (re)submit configs that already have a checkpoint. "
                        "By default the auto-discovered set (no --config) SKIPS "
                        "configs with an existing model so only untrained ones "
                        "run. Ignored when --config names configs explicitly.")
    p.add_argument("--no-generate", dest="generate", action="store_false",
                   help="do NOT pre-generate missing datasets locally before "
                        "submitting. By default any config whose dataset is not "
                        "on disk has it built first (so the cluster train_task "
                        "job doesn't EXIT at the data-load).")
    args = p.parse_args()

    ssh = None if args.local else args.ssh
    explicit = args.config is not None
    configs = args.config or _all_configs()
    # Auto-discovered set: skip configs that are already trained (unless
    # --rerun-trained). Explicit --config always runs what the user named.
    if not explicit and not args.rerun_trained:
        skipped = [c for c in configs if _is_trained(c, args.output_root)]
        configs = [c for c in configs if not _is_trained(c, args.output_root)]
        if skipped:
            print(f"{_DIM}skipping {len(skipped)} already-trained config(s) "
                  f"(--rerun-trained to include):{_R} "
                  f"{', '.join(skipped[:6])}"
                  f"{' …+%d more' % (len(skipped) - 6) if len(skipped) > 6 else ''}")
        if not configs:
            print("nothing to run: every discovered config is already trained "
                  "(use --rerun-trained or --config to force).")
            return 0
    if args.generate:
        _generate_missing(configs, args.output_root)

    log_dir = os.path.join(args.output_root, "log", "zebrafish",
                           "_companion_runner")
    os.makedirs(log_dir, exist_ok=True)

    where = "locally" if ssh is None else f"via ssh {ssh}"
    print(f"submitting {len(configs)} job(s) to gpu_{args.cluster} {where} "
          f"(cd {args.cluster_repo}; -W {args.hard_runtime_min} min):")
    jobs = {}  # cfg -> (jid, log)
    for cfg in configs:
        jid, log = _submit(cfg, cluster=args.cluster, n_cpus=args.n_cpus,
                           w_min=args.hard_runtime_min, log_dir=log_dir,
                           ssh=ssh, repo=args.cluster_repo,
                           conda_env=args.conda_env)
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
            # ONE batched bjobs for all live jobs (not one ssh per job).
            states = _job_states([jid for jid, _ in jobs.values()], ssh=ssh)
            done = []
            for cfg, (jid, log) in jobs.items():
                st = states.get(jid, "UNKNOWN")
                status, prog = _latest(log)
                col = _STATE_COL.get(st, _DIM)
                line = (f"[{_DIM}{stamp}{_R}] {_CYN}{cfg:38s}{_R} "
                        f"{col}{st[:4]:4s}{_R} | {status}")
                if prog:
                    line += f"  |  {_B}{_GRN}{prog}{_R}"
                print(line, flush=True)
                # Only retire on an EXPLICIT terminal state; UNKNOWN (failed
                # poll) keeps the job so a transient hiccup never false-finishes.
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
