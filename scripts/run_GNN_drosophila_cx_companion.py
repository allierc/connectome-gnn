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

    python scripts/run_GNN_drosophila_cx_companion.py                 # ALL configs on l4
    python scripts/run_GNN_drosophila_cx_companion.py --config drosophila_cx_gnn_rotation
    python scripts/run_GNN_drosophila_cx_companion.py --cluster a100 --config drosophila_cx_rotation drosophila_cx_both
    python scripts/run_GNN_drosophila_cx_companion.py --no-monitor    # submit only, don't watch
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
GROUP = "drosophila_cx"   # config/<GROUP>/, log/<GROUP>/, graphs_data/<GROUP>/

# ANSI colours for the terminal metrics print.
_R = "\033[0m"; _B = "\033[1m"; _DIM = "\033[2m"
_GRN = "\033[92m"; _YEL = "\033[93m"; _RED = "\033[91m"; _CYN = "\033[96m"; _BLU = "\033[94m"
_STATE_COL = {"RUN": _GRN, "PEND": _YEL, "EXIT": _RED, "DONE": _BLU}

# Heading-first ordering hint (the GNN heading run is the fastest / most precise
# convergence canary). Configs not listed here are appended alphabetically, so
# the default (no --config) runs EVERY config/drosophila_cx/*.yaml.
_ORDER_HINT = [
    "drosophila_cx_gnn_rotation",      "drosophila_cx_rotation",
    "drosophila_cx_gnn_rotation_vfwd", "drosophila_cx_rotation_vfwd",
    "drosophila_cx_gnn_both",          "drosophila_cx_both",
    "drosophila_cx_gnn_both_leaky",    "drosophila_cx_both_leaky",
    "drosophila_cx_gnn_position_2d",   "drosophila_cx_position_2d",
    "drosophila_cx_gnn_position_2d_leaky", "drosophila_cx_position_2d_leaky",
    "drosophila_cx_rotation_mlpdec",   "drosophila_cx_rotation_kbins",
]

# Default re-run set: the 18 trainings that did NOT reach n_epochs in the
# 2026-06-12..14 batch. All 16 GNN runs OOM'd at epoch 2 (the T=100 rollout
# step) on the L4 before the gradient-checkpointing fix (commit 702557b); the
# two RNN runs were killed by the owner. Bare `python run_GNN_..._companion.py`
# now relaunches exactly these on a100. Delete their log/drosophila_cx/<cfg>/
# folders first (or pass --all) so the runner does not skip them as "trained".
_RERUN_GNN_FIXED = [
    "drosophila_cx_gnn_rotation",
    "drosophila_cx_gnn_rotation_vfwd",
    "drosophila_cx_gnn_rotation_rep_1", "drosophila_cx_gnn_rotation_rep_2",
    "drosophila_cx_gnn_rotation_rep_3", "drosophila_cx_gnn_rotation_rep_4",
    "drosophila_cx_gnn_rotation_rep_5",
    "drosophila_cx_gnn_both", "drosophila_cx_gnn_both_leaky",
    "drosophila_cx_gnn_position_2d", "drosophila_cx_gnn_position_2d_leaky",
    "drosophila_cx_gnn_propriocep_distance",
    "drosophila_cx_gnn_propriocep_distance_leaky",
    "drosophila_cx_gnn_propriocep_mismatch",
    "drosophila_cx_gnn_propriocep_position_2d",
    "drosophila_cx_gnn_propriocep_position_2d_leaky",
    "drosophila_cx_rotation",
    "drosophila_cx_propriocep_position_2d",
]

# ---------------------------------------------------------------------------
# Config inventory (status note as of 2026-06-13). The runner auto-discovers
# EVERY config/drosophila_cx/*.yaml; this is just a record of what is already
# trained vs. what still needs a run, not a filter.
#
# ALREADY TRAINED (checkpoints in log/drosophila_cx/<cfg>/models/):
#   core task suite (RNN + GNN):
#     drosophila_cx_rotation              / drosophila_cx_gnn_rotation
#     drosophila_cx_rotation_vfwd         / drosophila_cx_gnn_rotation_vfwd
#     drosophila_cx_both                  / drosophila_cx_gnn_both
#     drosophila_cx_both_leaky            / drosophila_cx_gnn_both_leaky
#     drosophila_cx_position_2d           / drosophila_cx_gnn_position_2d
#     drosophila_cx_position_2d_leaky     / drosophila_cx_gnn_position_2d_leaky
#     drosophila_cx_rotation_kbins, drosophila_cx_rotation_mlpdec
#   seed replicates:  {drosophila_cx_,drosophila_cx_gnn_}rotation_rep_1..5
#   null / batch-shuffle controls (RNN):
#     drosophila_cx_rotation_bs_1..5, _er_1..5, _er_ei_1..5
#   proprioceptive-gain MISMATCH (sensory omega vs efference omega_proprio=g*omega):
#     drosophila_cx_propriocep_mismatch   / drosophila_cx_gnn_propriocep_mismatch
#
# NEW — PROPRIOCEPTION + TRANSLATION (pen_propriocep gate; NOT yet trained):
#   sensory omega -> PEN_a, efference omega_proprio=omega -> PEN_b, v_fwd -> PFN;
#   5-channel stimulus [omega, v_fwd, omega_proprio, cos0, sin0].
#   forward distance (target_kind=scalar_xi):
#     drosophila_cx_propriocep_distance         / drosophila_cx_gnn_propriocep_distance         (cumulative)
#     drosophila_cx_propriocep_distance_leaky   / drosophila_cx_gnn_propriocep_distance_leaky   (leaky, tau_d=0.5 s)
#   2-D position (target_kind=position_2d):
#     drosophila_cx_propriocep_position_2d      / drosophila_cx_gnn_propriocep_position_2d      (cumulative)
#     drosophila_cx_propriocep_position_2d_leaky/ drosophila_cx_gnn_propriocep_position_2d_leaky(leaky, tau_p=0.5 s)
#   sensory/efference control for any of the above: set
#     graph_model.velocity_gate: pen_propriocep_swap   (omega -> PEN_b, omega_proprio -> PEN_a).
#   Each RNN/GNN pair shares one dataset (drosophila_cx_si_task_338_propriocep_*);
#   only propriocep_position_2d is generated so far — the rest build on first run.
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
    patt = os.path.join(output_root, "log", GROUP, cfg, "models",
                        "best_model_with_*.pt")
    return bool(glob.glob(patt))


def _train_complete(cfg: str, output_root: str) -> bool:
    """True if training ran to the last epoch. GNN_Main writes a
    log/<GROUP>/<cfg>/_completed_train marker only after train_task finishes all
    epochs, so a run that stopped early (wall-clock / OOM) — which still left a
    best_model checkpoint — is correctly seen as INCOMPLETE and routed to a
    resume-train job rather than a test-only job."""
    return os.path.isfile(os.path.join(output_root, "log", GROUP, cfg,
                                       "_completed_train"))


def _latest_mtime(paths) -> float | None:
    """Newest mtime among existing paths, or None if none exist."""
    ts = [os.path.getmtime(p) for p in paths if os.path.exists(p)]
    return max(ts) if ts else None


def _test_is_fresh(cfg: str, output_root: str) -> bool:
    """True iff a test result newer than the newest model checkpoint exists.

    The decision rule the user asked for: a results file written AFTER the last
    model write means the current model has already been tested, so we skip it.
    No results at all, or results older than the model (stale — produced by an
    earlier checkpoint that has since been retrained), counts as NOT fresh, so a
    `-o test_plot` job is (re)launched. Test artefacts are
    log/<GROUP>/<cfg>/results_*.log and everything under log/<GROUP>/<cfg>/results/;
    models are log/<GROUP>/<cfg>/models/*.pt."""
    import glob
    base = os.path.join(output_root, "log", GROUP, cfg)
    model_t = _latest_mtime(glob.glob(os.path.join(base, "models", "*.pt")))
    if model_t is None:
        return False                      # no model -> nothing has been tested
    result_t = _latest_mtime(
        glob.glob(os.path.join(base, "results_*.log"))
        + glob.glob(os.path.join(base, "results", "*")))
    return result_t is not None and result_t >= model_t


def _all_configs() -> list[str]:
    """Every config/drosophila_cx/*.yaml (top level, not archive/), ordered with
    the core suite first then the rest (nulls, rep folds, ...) alphabetically."""
    import glob
    d = os.path.join(REPO, "config", GROUP)
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


def _job_group_name(ssh: str | None) -> str:
    """LSF job-group path for the concurrency cap, namespaced by cluster user."""
    user = _CFG.get("cluster_user", "allierc")
    return f"/{user}/{GROUP}_companion"


def _setup_job_group(limit: int, *, ssh: str | None) -> str | None:
    """Create (or re-limit) an LSF job group capping CONCURRENTLY RUNNING jobs at
    `limit`. All jobs submitted with `-g <group>` then run at most `limit` at a
    time — LSF queues the rest and starts them as slots free, so the cap holds
    even if the monitor is Ctrl-C'd. Returns the group name, or None on failure
    (caller falls back to no cap)."""
    g = _job_group_name(ssh)
    # bgadd is idempotent-ish: it errors if the group exists, so always follow
    # with bgmod to set the limit to the requested value either way.
    _lsf(f"bgadd -L {limit} {g}", ssh=ssh)
    r = _lsf(f"bgmod -L {limit} {g}", ssh=ssh)
    blob = ((r.stdout or "") + (r.stderr or "")).lower()
    if "no matching" in blob or "not found" in blob or "cannot" in blob:
        print(f"{_RED}could not set up job group {g} (limit {limit}); "
              f"submitting without a concurrency cap.{_R}")
        return None
    print(f"{_CYN}job group {g}: max {limit} concurrent running jobs "
          f"(rest queue, cap survives Ctrl-C).{_R}")
    return g


def _submit(cfg: str, *, cluster: str, n_cpus: int, w_min: int, log_dir: str,
            ssh: str | None, repo: str, conda_env: str,
            resume: bool = False, op: str = "train_task",
            job_group: str | None = None) -> tuple[str | None, str]:
    log = os.path.join(log_dir, f"{cfg}.lsf.log")
    # The bsub JOB runs `bash -lc 'conda run -n <env> python ...'` so the
    # compute node sources conda itself and runs in the right env — robust to
    # LSF not propagating the submission environment (a plain `python` job
    # lands in base and dies on ModuleNotFoundError: matplotlib). Mirrors
    # connectome_gnn.LLM.cluster (bash -l + conda run). --no-capture-output
    # keeps tqdm streaming to the log so the 300 s monitor can read progress.
    # --resume continues from the last per-epoch checkpoint and, crucially,
    # runs GNN_Main with erase=False so the existing models/ are NOT wiped.
    # `op` is the GNN_Main -o operation: "train_task" (train only),
    # "train_task_test_plot" (train then test+plot in one job, with a CUDA
    # cleanup between the two stages), or "test_plot" (test+plot only, loads the
    # existing best_model — never trains, never erases).
    resume_flag = " --resume" if resume else ""
    job = (f"cd {repo} && conda run --no-capture-output -n {conda_env} "
           f"python GNN_Main.py -o {op} {cfg}{resume_flag}")
    grp = f"-g {job_group} " if job_group else ""
    cmd = (f"cd {repo} && bsub -n {n_cpus} -gpu num=1 -q gpu_{cluster} "
           f"-W {w_min} {grp}-oo {log} -J {cfg} bash -lc {shlex.quote(job)}")
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
                   help="config name(s) to run (default: the 18-run GNN/RNN "
                        "re-run set _RERUN_GNN_FIXED; pass --all for EVERY "
                        "config/drosophila_cx/*.yaml)")
    p.add_argument("--all", action="store_true",
                   help="run EVERY config/drosophila_cx/*.yaml instead of the "
                        "default 18-run re-run set")
    p.add_argument("--cluster", choices=["l4", "a100", "h100"], default="a100")
    p.add_argument("--n-cpus", type=int, default=8)
    p.add_argument("--hard-runtime-min", type=int, default=5760,
                   help="bsub -W wall-clock limit in minutes (default 5760 = 96 h). "
                        "NB the chosen queue may cap this lower; verify with "
                        "`bqueues -l gpu_<cluster>`.")
    p.add_argument("--resume", action="store_true",
                   help="continue each run from its latest per-epoch checkpoint "
                        "(passes --resume to GNN_Main, so erase=False and the "
                        "existing models/ are PRESERVED, not wiped). Use this to "
                        "relaunch jobs killed by the wall-clock limit.")
    p.add_argument("--rnn-only", action="store_true",
                   help="run ONLY the RNN task configs — drop every GNN run "
                        "(any config whose name contains '_gnn_'). IMPLIES "
                        "--resume so each RNN checkpoint is CONTINUED and the "
                        "bsub job never erases models/ (train_task erases iff "
                        "NOT --resume). Use to finish the RNN trainings on their "
                        "own without touching the GNN runs.")
    p.add_argument("--gnn-only", action="store_true",
                   help="run ONLY the GNN task configs — keep every config whose "
                        "name contains '_gnn_', drop the rest. Inverse of "
                        "--rnn-only. Does NOT force --resume (pass --resume "
                        "yourself to continue OOM-killed GNN runs; the safety "
                        "guard refuses a non-resume submit that would erase an "
                        "existing checkpoint). Use to finish the GNN trainings "
                        "without touching the RNN runs.")
    p.add_argument("--test", action="store_true",
                   help="also run the test+plot stage. Per config: an UNTRAINED "
                        "(or --resume) run trains then tests in ONE job "
                        "(-o train_task_test_plot); an ALREADY-TRAINED run is "
                        "NOT retrained — it gets a test-only job (-o test_plot) "
                        "ONLY when its test results are missing or older than the "
                        "latest model checkpoint, and is skipped when results are "
                        "already newer than the model. test_plot never erases.")
    p.add_argument("--max-concurrent", type=int, default=0,
                   help="cap CONCURRENTLY RUNNING jobs at this many via an LSF "
                        "job group (e.g. --max-concurrent 16). All jobs are "
                        "submitted at once; LSF queues the surplus and starts "
                        "them as slots free, so the cap holds even if you Ctrl-C "
                        "the monitor. 0 = no cap (default).")
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
    # Config set: explicit --config wins; else --all = every yaml; else the
    # curated 18-run re-run set (default).
    if args.config is not None:
        configs = args.config
    elif args.all:
        configs = _all_configs()
    else:
        configs = list(_RERUN_GNN_FIXED)
    # --rnn-only: keep only the RNN task configs and force resume so the bsub
    # NEVER erases. GNN runs are named "*_gnn_*"; the RNN runs are the same
    # names without that infix. Forcing --resume is the safety the user asked
    # for: train_task passes erase=not --resume, so with resume on no models/
    # dir is ever wiped — these RNN trainings only ever continue. A config with
    # no checkpoint yet simply starts at epoch 0 (nothing to erase).
    if args.rnn_only and args.gnn_only:
        print(f"{_RED}--rnn-only and --gnn-only are mutually exclusive.{_R}")
        return 1
    if args.rnn_only:
        before = len(configs)
        configs = [c for c in configs if "_gnn_" not in c]
        if not args.resume:
            args.resume = True
            print(f"{_YEL}--rnn-only implies --resume: forcing resume=True so no "
                  f"RNN checkpoint is erased.{_R}")
        print(f"{_CYN}--rnn-only: {len(configs)} RNN config(s) kept "
              f"(dropped {before - len(configs)} GNN '_gnn_' run(s)).{_R}")
        if not configs:
            print("nothing to run: the selected set has no RNN (non-_gnn_) "
                  "configs.")
            return 0
    # --gnn-only: inverse filter. Unlike --rnn-only it does NOT force resume —
    # the erase safety guard below refuses a non-resume submit that would wipe an
    # existing GNN checkpoint, so the user passes --resume explicitly to continue
    # the OOM-killed runs (or --rerun-trained to intentionally restart fresh).
    if args.gnn_only:
        before = len(configs)
        configs = [c for c in configs if "_gnn_" in c]
        print(f"{_CYN}--gnn-only: {len(configs)} GNN config(s) kept "
              f"(dropped {before - len(configs)} non-GNN run(s)).{_R}")
        if not configs:
            print("nothing to run: the selected set has no GNN (_gnn_) configs.")
            return 0
    # Skip already-trained only for the broad --all sweep; the explicit and the
    # default re-run sets always run what is named (the re-run set is exactly
    # the failed jobs we want to relaunch). NOT under --test: there the trained
    # configs are precisely the ones we want to (re)test, and the --test plan
    # below routes each one (test-only when results are stale, skip when fresh).
    if args.all and args.config is None and not args.rerun_trained and not args.test:
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
    # Per-config submission plan: (cfg, op, resume). Without --test every config
    # runs the plain train_task op (legacy). With --test we route each config:
    #   * untrained (or --resume): train THEN test in one job
    #     (-o train_task_test_plot) — the resume continues the run, the test+plot
    #     stage follows automatically when training finishes.
    #   * already trained, results stale/missing: test-only (-o test_plot) — no
    #     retraining, never erases. "stale" = no results newer than the latest
    #     model checkpoint (see _test_is_fresh).
    #   * already trained, results already newer than the model: skipped.
    if args.test:
        plan, tested_skip = [], []
        for cfg in configs:
            if _train_complete(cfg, args.output_root):
                if _test_is_fresh(cfg, args.output_root):
                    tested_skip.append(cfg)
                else:
                    plan.append((cfg, "test_plot", False))
            else:
                # not finished (early-stopped or never ran) -> resume training,
                # then test in the same job.
                plan.append((cfg, "train_task_test_plot", args.resume))
        n_train = sum("train" in op for _, op, _ in plan)
        print(f"{_CYN}--test: {n_train} train+test job(s), {len(plan) - n_train} "
              f"test-only job(s); {len(tested_skip)} already-tested config(s) "
              f"skipped (results newer than model).{_R}")
        if tested_skip:
            print(f"{_DIM}  skipped: {', '.join(tested_skip[:6])}"
                  f"{' …+%d more' % (len(tested_skip) - 6) if len(tested_skip) > 6 else ''}{_R}")
    else:
        plan = [(cfg, "train_task", args.resume) for cfg in configs]
    if not plan:
        print("nothing to run.")
        return 0

    # SAFETY GUARD against the destructive default: a TRAIN job WITHOUT --resume
    # runs GNN_Main with erase=True, which DELETES every checkpoint in
    # log/<GROUP>/<cfg>/models/ before training (create_log_dir). Refuse to
    # submit when that would silently wipe an existing trained run; force the
    # user to choose --resume (continue) or --rerun-trained (intentional fresh).
    # test_plot ops never train, so they are never an erase risk.
    would_erase = [cfg for cfg, op, res in plan
                   if "train" in op and not res
                   and _is_trained(cfg, args.output_root)]
    resuming = [cfg for cfg, op, res in plan
                if "train" in op and res and _is_trained(cfg, args.output_root)]
    if resuming:
        print(f"{_CYN}--resume: {len(resuming)} run(s) continue from a checkpoint "
              f"(models/ preserved).{_R}")
    if would_erase and not args.rerun_trained:
        print(f"{_RED}{_B}REFUSING TO SUBMIT:{_R}{_RED} {len(would_erase)} "
              f"target config(s) already have checkpoints that a non-resume train "
              f"run would ERASE before training (erase = not --resume):{_R}")
        for c in would_erase[:12]:
            print(f"    {_RED}{c}{_R}")
        if len(would_erase) > 12:
            print(f"    {_DIM}…+{len(would_erase) - 12} more{_R}")
        print(f"{_YEL}Pass --resume to CONTINUE them from the last per-epoch "
              f"checkpoint (safe), or --rerun-trained to intentionally restart "
              f"from scratch (this wipes the checkpoints).{_R}")
        return 1

    if args.generate:
        _generate_missing([cfg for cfg, _, _ in plan], args.output_root)

    log_dir = os.path.join(args.output_root, "log", GROUP,
                           "_companion_runner")
    os.makedirs(log_dir, exist_ok=True)

    # Optional concurrency cap: an LSF job group limits how many of these jobs
    # RUN at once; the rest queue and start as slots free (cap survives Ctrl-C).
    job_group = None
    if args.max_concurrent and args.max_concurrent > 0:
        job_group = _setup_job_group(args.max_concurrent, ssh=ssh)

    where = "locally" if ssh is None else f"via ssh {ssh}"
    print(f"submitting {len(plan)} job(s) to gpu_{args.cluster} {where} "
          f"(cd {args.cluster_repo}; -W {args.hard_runtime_min} min):")
    jobs = {}  # cfg -> (jid, log)
    for cfg, op, res in plan:
        jid, log = _submit(cfg, cluster=args.cluster, n_cpus=args.n_cpus,
                           w_min=args.hard_runtime_min, log_dir=log_dir,
                           ssh=ssh, repo=args.cluster_repo,
                           conda_env=args.conda_env, resume=res, op=op,
                           job_group=job_group)
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
