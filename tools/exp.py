#!/usr/bin/env python
"""exp -- launch an experiment by NUMBER, poll it, and print the plan.

One markdown file per experiment, `experiments/exp<NN>_<name>.md`, is the whole
record: YAML front matter the tool reads, prose describing what differs, a table
of every spec, and a STATUS block the tool rewrites in place. Nothing else knows
a spec name -- not this script's output, not the PDF.

    exp launch <n>    stage the specs, bsub one job per run, record the ids
    exp poll   <n>    rewrite the STATUS block from the log tree
    exp report        one PDF page per experiment, read from the md files

The number is the only handle. `exp poll 1` and then opening
`experiments/exp01_*.md` answers "how far along is it and what do the numbers
look like" without anyone having to remember a config name.

STATUS HAS TWO BLOCKS AND THEY DO NOT SHARE A COLUMN. The landed block is
held-out, from results/metrics.txt, and is what a result is read off. The
running block is the TRAIN split, from tmp_training/<key>.log, tagged with the
iteration it was read at; the quantities not written per checkpoint (one-step r,
k_i, C_i) print blank there rather than borrowing the landed column's meaning.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP_DIR = os.path.join(ROOT, "experiments")
sys.path.insert(0, os.path.join(ROOT, "src"))

PAPER = "arxiv-2026"
LOG_ROOT = "/groups/saalfeld/home/allierc/GraphData/log/fly"
# Where a spec has to sit for the CLUSTER to read it. /groups is mounted both in
# the devcontainer and on the login node, so staging is a local copy here and no
# process of ours ever runs on a login node.
STAGE_DIR = "/groups/saalfeld/home/allierc/GraphData/config/fly"

# SLIDE 22's COLUMN SET, plus the two the offset question needs: C_i, the
# neuron's own incoming message offset in volts, and V_rest scored without it.
# (key, header, live) -- `live` says whether tmp_training/<stem>.log carries the
# quantity, which is what decides the blanks in the running block.
COLUMNS = [
    ("one_step_r",                       "one-step r",   False),
    ("rollout_r",                        "rollout r",    True),
    # THESE TWO ARE RELATIVE TO THE MODEL, NOT TO A FIXED FAMILY. The tester
    # writes `template_rollout_r` for the template of the model's OWN message
    # family and `template_alt_rollout_r` for the other one, and which family is
    # which is in the run's `alt_form_family`. Hard-coding "cond"/"curr" here
    # labelled every current-model table backwards; `_form_heads` resolves it
    # from the runs instead.
    ("template_rollout_r",               "fit roll own form",   False),
    ("template_alt_rollout_r",           "fit roll other form", False),
    ("Wij_R2",                           "R2_W",         True),
    ("tau_R2",                           "R2_tau",       True),
    ("V_rest_R2",                        "R2_Vrest",     True),
    ("V_rest_R2_uncorrected",            "R2_Vrest noC", True),
    ("msg_i_R2",                         "R2_msg",       True),
    ("tmpl_offset_per_neuron_absmedian", "C_i",          False),
    ("k_i",                              "k_i",          False),
    ("clustering_accuracy",              "cluster",      True),
]
_LIVE_FILE = {"rollout_r": "rollout", "Wij_R2": "Wij", "tau_R2": "tau",
              "V_rest_R2": "V_rest", "V_rest_R2_uncorrected": "V_rest",
              "msg_i_R2": "msg_i", "clustering_accuracy": "cluster"}

_BEGIN, _END = "<!-- STATUS:BEGIN -->", "<!-- STATUS:END -->"


# --------------------------------------------------------------------------- #
#  the experiment file                                                         #
# --------------------------------------------------------------------------- #
def exp_path(number):
    hits = sorted(glob.glob(os.path.join(EXP_DIR, f"exp{int(number):02d}_*.md")))
    if not hits:
        raise SystemExit(f"no experiment {number} in {EXP_DIR}")
    if len(hits) > 1:
        raise SystemExit(f"experiment {number} is ambiguous: {hits}")
    return hits[0]


def all_experiments():
    return sorted(glob.glob(os.path.join(EXP_DIR, "exp[0-9][0-9]_*.md")))


def load(path):
    """(front matter, whole text). The front matter is the machine-readable half."""
    text = open(path).read()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise SystemExit(f"{path}: no YAML front matter")
    return yaml.safe_load(m.group(1)), text


def runs(fm):
    """Every (arm, point, run name) of the grid, in the file's own order.

    The run name is derived here from the arm's spec pattern and the grid point
    rather than stored, so the md's spec table and the thing on disk cannot
    drift apart.
    """
    import itertools
    axes = fm["axes"]
    names = list(axes)
    out = []
    for arm in fm["arms"]:
        # AN ARM MAY COVER PART OF THE GRID. A probe added to settle one cell --
        # lambda 25 at noise_free, to separate "the lasso is too strong" from
        # "sigma 0 cannot hold the message" -- exists only there, and expanding
        # it over the whole noise axis would invent ten runs that were never
        # submitted and print them as pending for ever.
        only = arm.get("axes_only") or {}
        vals = [only.get(k, axes[k]) for k in names]
        for combo in itertools.product(*vals):
            pt = dict(zip(names, combo))
            out.append((arm, pt, arm["spec_pattern"].format(**pt)))
    return out


def _cell(pt):
    """A grid point with `fold` dropped: what a mean +- SD is taken over."""
    return tuple((k, v) for k, v in pt.items() if k != "fold")


# --------------------------------------------------------------------------- #
#  reading a run                                                               #
# --------------------------------------------------------------------------- #
def gauge_k(run):
    """k_i = tau_i * df/dmsg_i, median over neurons, from docs/gauge_k.json.

    Not in metrics.txt and not written per checkpoint: it needs a checkpoint and
    a data pass, so tools/measure_gauge_k.py computes it and caches it. Absent
    cache -> the column is blank, which is honest rather than zero.
    """
    import json
    p = os.path.join(ROOT, "docs", "gauge_k.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p)).get(run)
    return d["k_median"] if d else None


def metrics_of(run):
    p = os.path.join(LOG_ROOT, run, "results", "metrics.txt")
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    # Only the rollout lines means the plot pass never reached the readout: no
    # recovered parameters, which for every column but one is the same state as
    # no file at all.
    return out if "Wij_R2" in out else None


def _last_row(run, stem):
    """(iteration, {column: value}) from the last line of tmp_training/<stem>.log."""
    p = os.path.join(LOG_ROOT, run, "tmp_training", f"{stem}.log")
    if not os.path.exists(p):
        return None, {}
    head, last = None, None
    for line in open(p):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("iteration"):
            head = line.split(",")
            continue
        last = line
    if head is None or last is None:
        return None, {}
    d = dict(zip(head, last.split(",")))
    try:
        return int(float(d.get("iteration", "nan"))), d
    except ValueError:
        return None, d


def _rows_by_iter(run, stem):
    """{iteration: row} for every snapshot tmp_training/<stem>.log holds."""
    p = os.path.join(LOG_ROOT, run, "tmp_training", f"{stem}.log")
    if not os.path.exists(p):
        return {}
    head, out = None, {}
    for line in open(p):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("iteration"):
            head = line.split(",")
            continue
        if head is None:
            continue
        d = dict(zip(head, line.split(",")))
        try:
            out[int(float(d["iteration"]))] = d
        except (KeyError, ValueError):
            pass
    return out


def live_of(run, at=None):
    """The train-split numbers this run has written, and the iteration they are at.

    `at` pins the snapshot. WITHOUT IT A GROUP MEAN MIXES ITERATIONS: five folds
    launched together do not reach the same snapshot at the same moment, and
    averaging a fold at iteration 1 with four at 16,001 reported R2_W as
    0.450 +- 0.472 for a group whose five folds were 0.913, 0.932, 0.923, 0.919
    and 0.910. The iteration printed beside it -- the max -- made that look like
    a measurement at 16,001 rather than an average over two different training
    states.
    """
    out, it = {}, None
    for key, stem in _LIVE_FILE.items():
        rows = _rows_by_iter(run, stem)
        if not rows:
            continue
        i = max(rows) if at is None else (at if at in rows else None)
        if i is None:
            continue
        it = i if it is None else max(it, i)
        if key in rows[i]:
            out[key] = rows[i][key]
    return it, out


def common_iter(group_runs):
    """The latest snapshot every run in the group has reached, or None."""
    sets = []
    for run in group_runs:
        rows = _rows_by_iter(run, "Wij")
        if not rows:
            return None
        sets.append(set(rows))
    common = set.intersection(*sets) if sets else set()
    return max(common) if common else None


def commit_of(run):
    for marker in ("_completed_train", "_complete"):
        p = os.path.join(LOG_ROOT, run, marker)
        if os.path.exists(p):
            for line in open(p):
                if line.startswith("commit="):
                    return line.split("=", 1)[1].strip()
    return None


def trained(run):
    """Has `-o train` finished? The marker it writes on the way out."""
    return os.path.exists(os.path.join(LOG_ROOT, run, "_completed_train"))


_RUNTIME_RE = re.compile(r"^\s*Run time :\s+(\d+) sec\.", re.M)


def wall_seconds(run):
    """LSF's own wall clock for the TRAIN job, from cluster.out.

    THE ONLY HONEST SOURCE for how long a run took. The training logs carry no
    timestamps, file mtimes move when a later pass rewrites a figure, and a rate
    typed into a markdown file is a number nobody can re-derive -- which is
    exactly how experiment 0's first speed table came to disagree with the
    cluster by a factor of two to three. `-o train` is submitted as
    <exp>_<spec> and writes cluster.out; `-o test_plot` writes analyse.out, so
    this is training wall only.

    None when the job has not reported yet, so a running job shows blank rather
    than a rate computed against a clock that is still going.
    """
    p = os.path.join(LOG_ROOT, run, "cluster.out")
    if not os.path.isfile(p):
        return None
    hits = _RUNTIME_RE.findall(open(p, errors="replace").read())
    return int(hits[-1]) if hits else None


def status_of(run):
    """landed > trained > running > pending.

    `trained` IS ITS OWN STATE and the tool was wrong to lack it. `-o train`
    does not write results/metrics.txt -- that is `-o test_plot`'s job -- so a
    run whose training had finished still read as "running", and 69 of them sat
    that way while the held-out table stayed empty and nothing said why.
    """
    if metrics_of(run) is not None:
        return "landed"
    if trained(run):
        return "trained"
    return "running" if live_of(run)[0] is not None else "pending"


# --------------------------------------------------------------------------- #
#  launch                                                                      #
# --------------------------------------------------------------------------- #
def stage(fm, names):
    """Copy the specs where the cluster can read them.

    The repo is the record; the cluster reads /groups. This copy happens in the
    devcontainer, so it is a local filesystem operation and not a process on a
    login node.
    """
    src_dir = os.path.join(ROOT, fm["specs_dir"])
    for n in names:
        src = os.path.join(src_dir, f"{n}.yaml")
        if not os.path.exists(src):
            raise SystemExit(f"missing spec: {src}")
        shutil.copy2(src, os.path.join(STAGE_DIR, f"{n}.yaml"))
    return names


def launch(number, dry_run=False, only_arm=None, where=None):
    """Submit an experiment, or just one arm of it, or one slice of one arm.

    `only_arm` exists because an arm failing on its own is ordinary -- a code
    path the other arm does not take, a dataset the other arm does not read --
    and relaunching all thirty would kill the fifteen that are running fine.

    `where` is the finer cut, `[("noise", {"noise_005", "noise_05"})]`. WIDENING
    AN ARM'S AXIS IS THE CASE IT EXISTS FOR: exp02's lasso-25 arm was a
    noise-free probe and then wanted the other two levels, and `--arm cond_l25`
    alone would have cleared and resubmitted the five folds that had already
    landed. Anything not named in `where` is unconstrained.
    """
    path = exp_path(number)
    fm, _ = load(path)
    where = dict(where or [])
    # AN ARM WITH `submit: false` IS READ, NEVER RUN. A comparison arm is often
    # another experiment's runs -- exp02's reference is exp01's nominal folds --
    # and launching it would resubmit them AND, because launch clears a run
    # directory before reusing it, delete a running experiment's logs.
    names = [r for arm, pt, r in runs(fm)
             if arm.get("submit", True)
             and (only_arm is None or arm["id"] == only_arm)
             and all(pt.get(k) in v for k, v in where.items())]
    if not names:
        raise SystemExit(f"no runs match arm={only_arm!r} where={where} "
                         f"in experiment {number}")
    stage(fm, names)
    print(f"staged {len(names)} specs -> {STAGE_DIR}")
    if dry_run:
        for n in names:
            print(f"  would submit: -o {fm['task']} {n}")
        return 0

    from connectome_gnn.LLM.cluster import _bsub_over_ssh
    hh, mm = str(fm["wall"]).split(":")
    wall_min = int(hh) * 60 + int(mm)
    # AN ARM MAY PIN ITS OWN QUEUE. For a hardware benchmark the arm IS the GPU,
    # so the queue cannot be a property of the experiment; everywhere else the
    # arms share the experiment's queue and this falls through.
    queue_of = {r: (arm.get("queue") or fm["queue"]).replace("gpu_", "")
                for arm, _pt, r in runs(fm)}
    ids = {}
    for n in names:
        node = queue_of[n]
        log_dir = os.path.join(LOG_ROOT, n)
        # CLEAR IT FIRST. A run directory is keyed by name alone, so anything
        # that ever used this name left its tmp_training behind -- and `poll`
        # would read those numbers and print them as this experiment's. Seen:
        # exp00's markdown filled with rates from an ad-hoc benchmark that had
        # used the same four names an hour earlier.
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        jid, queue, res = _bsub_over_ssh(
            # THE STAGED ABSOLUTE PATH, not the bare stem. add_pre_folder
            # classifies a bare name by looking for a domain keyword in it, so
            # `bench_rtx6000_bf16` is unrecognised and every job dies in 30
            # seconds with "does not exist or is not recognized". An absolute
            # path skips that: load_run_config takes the domain from the parent
            # directory, which staging guarantees is `fly`.
            cluster_cmd=(f"python GNN_Main.py -o {fm['task']} "
                         f"{os.path.join(STAGE_DIR, n + '.yaml')}"),
            conda_env="connectome-gnn", node_name=node, n_cpus=8, device="gpu",
            hard_runtime_limit_min=wall_min,
            stdout_path=os.path.join(log_dir, "cluster.out"),
            stderr_path=os.path.join(log_dir, "cluster.err"),
            job_name=f"exp{int(number):02d}_{n}")
        if jid is None:
            print(f"  FAILED {n}: {res.stderr.strip()[:200]}")
        else:
            ids[n] = jid
            print(f"  {jid}  {queue}  {n}")
    if ids:
        # MERGE, never replace. `launch --arm` submits a subset, and assigning
        # the subset dropped the other arm's ids from the record -- the one
        # place the mapping from run to job exists.
        merged = dict(fm.get("job_ids") or {})
        merged.update(ids)
        fm["job_ids"] = merged
        _rewrite_front_matter(path, fm)
    print(f"{len(ids)}/{len(names)} submitted")
    return 0 if len(ids) == len(names) else 1


def analyse(number, dry_run=False):
    """bsub `-o test_plot` for every run whose training has finished.

    The held-out numbers the landed table reads come from results/metrics.txt,
    which only the plot pass writes. Training and analysis are two jobs, and
    this is the second one -- skipping runs that already have metrics so it is
    safe to re-run as more of the grid finishes.
    """
    path = exp_path(number)
    fm, _ = load(path)
    todo = [r for _a, _pt, r in runs(fm) if trained(r) and metrics_of(r) is None]
    if not todo:
        print(f"experiment {number}: nothing to analyse")
        return 0
    print(f"experiment {number}: {len(todo)} runs to analyse")
    if dry_run:
        for n in todo:
            print(f"  would submit: -o test_plot {n}")
        return 0
    from connectome_gnn.LLM.cluster import _bsub_over_ssh
    queue_of = {r: (arm.get("queue") or fm["queue"]).replace("gpu_", "")
                for arm, _pt, r in runs(fm)}
    ids = {}
    for n in todo:
        d = os.path.join(LOG_ROOT, n)
        jid, queue, res = _bsub_over_ssh(
            cluster_cmd=(f"python GNN_Main.py -o test_plot "
                         f"{os.path.join(STAGE_DIR, n + '.yaml')}"),
            conda_env="connectome-gnn", node_name=queue_of[n], n_cpus=8,
            device="gpu", hard_runtime_limit_min=240,
            stdout_path=os.path.join(d, "analyse.out"),
            stderr_path=os.path.join(d, "analyse.err"),
            job_name=f"an{int(number):02d}_{n}")
        if jid is None:
            print(f"  FAILED {n}: {res.stderr.strip()[:160]}")
        else:
            ids[n] = jid
    print(f"{len(ids)}/{len(todo)} submitted")
    if ids:
        fm["analyse_job_ids"] = {**(fm.get("analyse_job_ids") or {}), **ids}
        _rewrite_front_matter(path, fm)
    return 0 if len(ids) == len(todo) else 1


def _rewrite_front_matter(path, fm):
    text = open(path).read()
    body = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S)
    open(path, "w").write(
        "---\n" + yaml.safe_dump(fm, sort_keys=False,
                                 default_flow_style=False).rstrip()
        + "\n---\n" + body)


# --------------------------------------------------------------------------- #
#  poll -> the STATUS block                                                    #
# --------------------------------------------------------------------------- #
def _mean(vals):
    xs = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:
            xs.append(f)
    return sum(xs) / len(xs) if xs else None


def _mean_sd(vals, nd=3):
    xs = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:
            xs.append(f)
    if not xs:
        return ""
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if len(xs) > 1 else 0.0
    return f"{m:.{nd}f} ± {sd:.{nd}f}"


# The R2 columns that have an outlier band, and the metrics key that counts it.
# Printed in parentheses beside the value, as the first deck does: an R2 read
# without the fraction it dropped is not comparable with one that dropped none.
_OUTLIER_OF = {"template_rollout_r": "template_rollout_pct_clamped",
               "template_alt_rollout_r": "template_alt_rollout_pct_clamped",
               "Wij_R2": "Wij_pct_outliers", "tau_R2": "tau_pct_outliers",
               "V_rest_R2": "V_rest_pct_outliers", "msg_i_R2": "msg_i_pct_outliers"}


def _summary_rows(fm, rs, source, nd=3):
    """One row per (arm, grid cell) summarised over folds.

    `source` is either metrics_of for the landed block or live_of for the
    running one; thirty individual rows is not a comparison, and the comparison
    this file exists for is between arms at one noise level.
    """
    groups = []
    for arm, pt, _run in rs:
        g = (arm["id"], _cell(pt))
        if g not in groups:
            groups.append(g)
    out = []
    for arm_id, cell in groups:
        acc = {k: [] for k, _, _ in COLUMNS}
        out_acc = {}
        its, n_here = [], 0
        # EVERY FOLD READ AT THE SAME SNAPSHOT, so the mean describes one
        # training state rather than a mixture of however far each job happens
        # to have got.
        at_iter = (None if source == "landed" else common_iter(
            [r for a, p, r in rs
             if a["id"] == arm_id and _cell(p) == cell
             and status_of(r) == "running"]))
        for arm, pt, run in rs:
            if arm["id"] != arm_id or _cell(pt) != cell:
                continue
            if source == "landed":
                m = metrics_of(run)
                if not m:
                    continue
                n_here += 1
                for k, _, _ in COLUMNS:
                    acc[k].append(gauge_k(run) if k == "k_i" else m.get(k))
                for k, pk in _OUTLIER_OF.items():
                    out_acc.setdefault(k, []).append(m.get(pk))
            else:
                if status_of(run) != "running":
                    continue
                it, d = live_of(run, at=at_iter)
                if it is None:
                    continue
                its.append(it)
                n_here += 1
                for k, _, live in COLUMNS:
                    if live:
                        acc[k].append(d.get(k))
        if not n_here:
            continue
        cells = []
        for k, _h, live in COLUMNS:
            if source != "landed" and not live:
                cells.append("")
                continue
            txt = _mean_sd(acc[k], nd)
            pct = _mean(out_acc.get(k, [])) if source == "landed" else None
            if txt and pct is not None:
                txt += f" ({pct:.1f})"
            cells.append(txt)
        out.append((arm_id, cell, (f"{max(its):,}" if its else str(n_here)), cells))
    return out


def status_block(fm):
    rs = runs(fm)
    counts = {"landed": 0, "trained": 0, "running": 0, "pending": 0}
    for _arm, _pt, run in rs:
        counts[status_of(run)] += 1
    axes_no_fold = [k for k in fm["axes"] if k != "fold"]
    heads = [h for _, h, _ in COLUMNS]
    sep = "|" + "---|" * (2 + len(axes_no_fold) + len(heads))

    L = [_BEGIN, "", "## Status", "",
         f"**{counts['landed']}/{len(rs)} landed**, "
         f"{counts['trained']} trained (awaiting `-o test_plot`), "
         f"{counts['running']} running, {counts['pending']} pending", ""]

    for src, title, last in (
            ("landed", "Landed --- held-out, `results/metrics.txt`", "n"),
            ("running", "Running --- train split, `tmp_training/`, "
                        "blank where not written per checkpoint", "iter")):
        L += [f"### {title}", "",
              "| arm | " + " | ".join(axes_no_fold) + f" | {last} | "
              + " | ".join(heads) + " |", sep]
        rows = _summary_rows(fm, rs, src)
        if not rows:
            L.append("| — |" + " |" * (len(axes_no_fold) + 1 + len(heads)))
        for arm_id, cell, n, cells in rows:
            L.append(f"| {arm_id} | " + " | ".join(v for _, v in cell)
                     + f" | {n} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["### Per run", "", "| run | status | iter | commit |", "|---|---|---|---|"]
    for _arm, _pt, run in rs:
        it = live_of(run)[0]
        L.append(f"| `{run}` | {status_of(run)} | "
                 f"{f'{it:,}' if it else ''} | `{(commit_of(run) or '')[:12]}` |")
    L += ["", _END]
    return "\n".join(L)


def write_index():
    """experiments/INDEX.md -- number, name, one line, progress.

    The one file to open when the number itself is what you have forgotten.
    Regenerated on every poll so it cannot go stale.
    """
    L = ["# Experiments", "",
         "| # | name | purpose | landed | file |", "|---|---|---|---|---|"]
    for path in all_experiments():
        fm, _ = load(path)
        rs = runs(fm)
        n = sum(1 for _a, _p, r in rs if status_of(r) == "landed")
        L.append(f"| {fm['number']} | {fm['name']} | {fm['purpose']} | "
                 f"{n}/{len(rs)} | [`{os.path.basename(path)}`]"
                 f"({os.path.basename(path)}) |")
    open(os.path.join(EXP_DIR, "INDEX.md"), "w").write("\n".join(L) + "\n")


def poll(number):
    path = exp_path(number)
    fm, text = load(path)
    block = status_block(fm)
    if _BEGIN in text and _END in text:
        i, j = text.index(_BEGIN), text.index(_END) + len(_END)
        text = text[:i] + block + text[j:]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    open(path, "w").write(text)
    rs = runs(fm)
    counts = {}
    for _a, _p, run in rs:
        counts[status_of(run)] = counts.get(status_of(run), 0) + 1
    write_index()
    print(f"{os.path.basename(path)}: "
          + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    return 0


# --------------------------------------------------------------------------- #
#  report                                                                      #
# --------------------------------------------------------------------------- #
def _tex(s):
    out = str(s)
    for ch, rep in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                    ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                    ("}", r"\}"), ("~", r"\textasciitilde{}"),
                    ("^", r"\textasciicircum{}")):
        out = out.replace(ch, rep)
    return out


# The column headers as maths, applied to the plain names in COLUMNS so the
# table and the markdown cannot drift apart.
_PRETTY = {
    "one-step r": r"one-step $r$", "rollout r": r"rollout $r$",
    "fit roll own form": r"\shortstack{fit roll $r$\\own form}",
    "fit roll other form": r"\shortstack{fit roll $r$\\other form}",
    "R2_W": r"$R^2_{\hat W}$", "R2_tau": r"$R^2_{\hat\tau}$",
    "R2_Vrest": r"$R^2_{V^{\mathrm{rest}}}$",
    "R2_Vrest noC": r"$R^2_{V^{\mathrm{rest}}}$ \tiny no $C_i$",
    "R2_msg": r"$R^2_{\mathrm{msg}}$", "C_i": r"$C_i$", "k_i": r"$k_i$",
    "cluster": "cluster",
}
# A figure a slide should carry after its table, if it exists. Keyed by
# experiment name so a new experiment adds one line, not a branch.
_FIGURES = {"derivative_target":
            [("Fig/exp01_errors.png", "recovery errors, five folds pooled")]
            + [(f"Fig/exp01_errors_cv{i:02d}.png",
                f"recovery errors, fold cv{i:02d}") for i in range(5)]
            # The same neuron, the same fold, the two derivative targets, so
            # panel f's per-synapse fits sit side by side across the arms.
            # The third entry is the height the graphic may occupy, as a
            # fraction of the slide's text height. These panels are nearly
            # square and carry small print, so they get the whole slide where
            # the error histograms are wide and do not need it.
            + [("Fig/exp01_panels_bug_noise005.png",
                "neuron 2895 Am, bug", 0.72),
               ("Fig/exp01_panels_nominal_noise005.png",
                "neuron 2895 Am, nominal", 0.72)],
            "conductance_lasso":
            [("Fig/exp02_panels_current_noise005.png",
              "neuron 2895 Am, current form", 0.72),
             ("Fig/exp02_panels_conductance_noise005.png",
              "neuron 2895 Am, conductance lasso 100", 0.72)]}
_GREEN = 0.9


def _cellf(txt):
    """A table cell, green above 0.9 on the value it leads with."""
    if not txt:
        return "---"
    out = txt.replace("±", r"$\pm$")
    try:
        v = float(txt.split()[0])
    except (ValueError, IndexError):
        return out
    return r"\good{" + out + "}" if v > _GREEN else out


def _form_heads(rs):
    """Name the two `fit roll` columns after the families they actually are.

    FROM THE ROLLOUT'S OWN RECORD, NOT FROM `alt_form_family`. The two are about
    different things and only agree by accident. `alt_form_family` is the family
    of the ALTERNATIVE FIT ON THE MESSAGE, and `metrics.py` decides it from
    `_is_conductance_data(ode_params)` -- the family of the GENERATOR. The
    rollout's own/other, in contrast, follows the MODEL: on current data a
    conductance model reports `alt_form_family: conductance` while its
    `template_rollout_model` is `flyvis_conductance_known_ode`, so reading the
    header off `alt_form_family` labels that table backwards.

    `template_rollout_model` and `template_alt_rollout_model` name the known-ODE
    each column was actually rolled out with, per run, which is the thing the
    header claims. Returns None when the landed runs disagree -- experiment 2
    mixes current and conductance models in one table, and there "own form" and
    "other form" are the only true names.
    """
    def fam(name):
        return "conductance" if "conductance" in (name or "") else "current"

    pairs = {(fam(m["template_rollout_model"]), fam(m["template_alt_rollout_model"]))
             for _a, _p, r in rs
             if (m := metrics_of(r)) and "template_rollout_model" in m
             and "template_alt_rollout_model" in m}
    if len(pairs) != 1:
        return None
    own, alt = pairs.pop()
    if own == alt:                      # cannot happen, but do not claim it did
        return None
    short = {"current": r"curr.\ form", "conductance": r"cond.\ form"}
    return short[own], short[alt]


def _arm_columns(fm):
    """Extra leading columns read off each arm's `differs_by`.

    Declared as `report.arm_columns: {<header>: <config key>}`. The arm id alone
    does not say what an arm IS -- `conductance` and `cond_l25` differ in one
    number and the reader has to go and find it -- and that number is already in
    the front matter, so the column is derived, not typed. An arm that does not
    set the key gets "---", which is the true answer for the current arm: it has
    no lasso at all, which is not the same as a lasso of zero.
    """
    return list(fm.get("report", {}).get("arm_columns", {}).items())


def _arm_label(fm, arm_id):
    """What the arm is CALLED in the table, which need not be its id.

    `report.arm_labels: {cond_l25: conductance}`. Once a column carries the one
    number the two conductance arms differ by, the ids stop earning their keep:
    `conductance` and `cond_l25` beside a `lasso` column of 100 and 25 says the
    same thing twice and invites the reader to look for a third difference. The
    id stays the handle everywhere else -- `--arm`, the job record, the per-run
    table -- because it has to be unique and this does not.
    """
    return fm.get("report", {}).get("arm_labels", {}).get(arm_id, arm_id)


def _arm_value(arms, arm_id, key):
    for a in arms:
        if a["id"] == arm_id:
            v = (a.get("differs_by") or {}).get(key)
            if v is None:
                return "---"
            return f"{v:g}" if isinstance(v, (int, float)) else str(v)
    return "---"


def _table(fm, source="landed"):
    """One of the markdown's two tables, every column it has.

    `source` is "landed" (held-out, results/metrics.txt) or "running" (the TRAIN
    split at a stated iteration, from tmp_training/). They do not share a
    column's meaning and the slide labels which one it is showing.
    """
    rs = runs(fm)
    heads = [_PRETTY.get(h, _tex(h)) for _, h, _ in COLUMNS]
    fh = _form_heads(rs)
    if fh:
        for h, name in zip(("fit roll own form", "fit roll other form"), fh):
            heads[[c[1] for c in COLUMNS].index(h)] = \
                r"\shortstack{fit roll $r$\\" + name + "}"
    axes_no_fold = [k for k in fm["axes"] if k != "fold"]
    extra = _arm_columns(fm)
    # NUMBERS RIGHT, NAMES CENTRED. The values line up on their decimal point,
    # which `r` gives; the two-line headers are wider than the values they sit
    # over, so left as `r` they hang off to one side. `\multicolumn{1}{c}` centres
    # each name over its own column without moving the values under it.
    spec = "l" + " r" * len(extra) + " l" * len(axes_no_fold) + " r" + " r" * len(heads)
    heads = [r"\multicolumn{1}{c}{" + h + "}" for h in heads]
    L = [r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{spec}}}",
         r"\toprule",
         " & ".join(["arm"] + [r"\multicolumn{1}{c}{" + _tex(h) + "}" for h, _ in extra]
                    + [_tex(k) for k in axes_no_fold]
                    + [r"\multicolumn{1}{c}{"
                       + ("n" if source == "landed" else "iter") + "}"] + heads)
         + r" \\", r"\midrule"]
    rows = _summary_rows(fm, rs, source, nd=2)
    _first = fm.get("report", {}).get("arm_order")
    if _first:
        rows.sort(key=lambda r: _first.index(r[0]) if r[0] in _first else len(_first))
    if not rows:
        L.append(" & ".join(["---"] * (2 + len(extra) + len(axes_no_fold)
                                       + len(heads))) + r" \\")
    for arm_id, cell, n, cells in rows:
        L.append(" & ".join(
            [_tex(_arm_label(fm, arm_id))]
            + [_tex(_arm_value(fm["arms"], arm_id, k)) for _h, k in extra]
            + [_tex(v) for _, v in cell] + [n]
            + [_cellf(c) for c in cells]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(L)


def _hms(sec):
    return f"{int(sec) // 3600}:{(int(sec) % 3600) // 60:02d}"


def _timing_table(fm):
    """Wall clock per arm, from LSF, with the rate it implies.

    Every number here is derived: `Run time` out of each run's cluster.out and
    the last iteration in its tmp_training/Wij.log. The projection is that rate
    held for `report.timing_iters` iterations -- experiment 0 runs a tenth of a
    campaign run, so the useful question is not "how long did the benchmark
    take" but "how long would the real thing take at this rate".
    """
    rs = runs(fm)
    iters = int(fm.get("report", {}).get("timing_iters", 0)) or None
    axes_no_fold = [k for k in fm["axes"] if k != "fold"]
    extra = _arm_columns(fm)
    heads = ["wall (h:mm)", "iterations", "it/s"]
    if iters:
        heads.append(f"{iters / 1e6:g} M it (h)")
    groups, out = [], []
    for arm, pt, run in rs:
        g = (arm["id"], _cell(pt))
        if g not in groups:
            groups.append(g)
    for arm_id, cell in groups:
        secs, its = [], []
        for arm, pt, run in rs:
            if arm["id"] != arm_id or _cell(pt) != cell:
                continue
            w, it = wall_seconds(run), live_of(run)[0]
            if w and it:
                secs.append(w)
                its.append(it)
        if not secs:
            continue
        w, it = sum(secs) / len(secs), sum(its) / len(its)
        row = [_hms(w), f"{it:,.0f}", f"{it / w:.1f}"]
        if iters:
            row.append(f"{iters / (it / w) / 3600:.1f}")
        out.append((arm_id, cell, row))
    _first = fm.get("report", {}).get("arm_order")
    if _first:
        out.sort(key=lambda r: _first.index(r[0]) if r[0] in _first else len(_first))
    if not out:
        return ""
    spec = "l" + " r" * len(extra) + " l" * len(axes_no_fold) + " r" * len(heads)
    L = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
         " & ".join(["arm"] + [_tex(h) for h, _ in extra]
                    + [_tex(k) for k in axes_no_fold]
                    + [r"\multicolumn{1}{c}{" + _tex(h) + "}" for h in heads])
         + r" \\", r"\midrule"]
    for arm_id, cell, row in out:
        L.append(" & ".join([_tex(_arm_label(fm, arm_id))]
                            + [_tex(_arm_value(fm["arms"], arm_id, k)) for _h, k in extra]
                            + [_tex(v) for _, v in cell] + row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def _slides(fm):
    """One table slide per experiment, then one slide per figure it declares."""
    rs = runs(fm)
    n_land = sum(1 for _a, _p, r in rs if status_of(r) == "landed")
    L = [rf"\begin{{frame}}{{\ft{{Experiment {fm['number']} --- {_tex(fm['name'])}}}}}",
         rf"\srcpath{{experiments/{_tex(os.path.basename(exp_path(fm['number'])))}}}",
         r"\vspace*{0.3cm}", r"\centering\tiny",
         r"\setlength{\tabcolsep}{2pt}", _table(fm),
         r"\\[6pt]",
         r"{\tiny\raggedright",
         rf"{_tex(fm['purpose'])} \\[2pt]",
         rf"{n_land} of {len(rs)} runs landed; mean $\pm$ SD over the folds that "
         rf"have. In parentheses: the percentage of outliers dropped on the "
         rf"$R^2$ columns, and on the two fit-roll columns the percentage of "
         rf"neuron-frames the $\pm$100\,V divergence clamp was holding --- a "
         rf"fit roll $r$ beside a large one is a diverged rollout, not a score. "
         rf"Green above ${_GREEN}$.\par}}"]
    # THE RUNNING BLOCK GOES ON THE SLIDE TOO, WHEN THERE IS ONE. An experiment
    # mid-flight otherwise shows a row of dashes and says nothing, while its jobs
    # have been writing per-checkpoint numbers for hours. This is the TRAIN split
    # at a stated iteration, not held-out, so it is a second table with its own
    # caption rather than blanks filled in from a different meaning.
    if _summary_rows(fm, rs, "running", nd=2):
        L += [r"\\[10pt]", r"\centering\tiny", _table(fm, "running"), r"\\[4pt]",
              r"{\tiny\raggedright Still training --- the TRAIN split from "
              r"\texttt{tmp\_training/}, every fold read at the same iteration. "
              r"Not held-out and not comparable to the table above; the "
              r"quantities not written per checkpoint are blank rather than "
              r"borrowed from it.\par}"]
    tt = _timing_table(fm) if fm.get("report", {}).get("timing") else ""
    if tt:
        L += [r"\\[10pt]", r"\centering\tiny", tt, r"\\[4pt]",
              r"{\tiny\raggedright LSF \texttt{Run time} from each run's "
              r"\texttt{cluster.out}, over the last iteration its "
              r"\texttt{tmp\_training/Wij.log} reached. Training wall only; "
              r"\texttt{-o test\_plot} is a separate job.\par}"]
    L += [r"\end{frame}"]
    for fig, cap, *rest in _FIGURES.get(fm["name"], []):
        box = rest[0] if rest else 0.72
        if os.path.exists(os.path.join(ROOT, "presentation", fig)):
            L += [rf"\begin{{frame}}{{\ft{{Experiment {fm['number']} --- {_tex(cap)}}}}}",
                  r"\vspace*{0.2cm}", r"\begin{center}",
                  rf"\setlength{{\panelbox}}{{{box}\textheight}}",
                  rf"\fitgfx{{{fig}}}", r"\end{center}", r"\end{frame}"]
    return "\n".join(L)


def report(paths, make_pdf=True):
    """experiments/report.pdf -- the same deck style as presentation/.

    Built in presentation/ because the Janelia theme, people.sty and Fig/ live
    there, then copied to experiments/report.pdf so the number is still the only
    handle. Every value comes from the experiment markdown, which comes from the
    runs: nothing in this document is typed.
    """
    L = [r"""%!TEX program = pdflatex
\documentclass[aspectratio=169]{beamer}
\usetheme{Janelia}
\usepackage{people}
\usepackage[english]{babel}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{tabularx,booktabs,array,graphicx}
\definecolor{goodgreen}{rgb}{0.0,0.45,0.0}
\newcommand{\good}[1]{\textcolor{goodgreen}{#1}}
\usepackage{helvet}
% SMALLER THAN THE THEME'S FRAME TITLE. The Janelia theme sizes it for a short
% section name; these titles carry an experiment number, its name and what the
% slide shows, and at the theme's size they wrap onto a second line that the
% green title bar then clips.
%
% \setbeamerfont{frametitle} DOES NOTHING HERE. beamerthemeJanelia.sty:215 draws
% the title as a literal \LARGE\textbf{\insertframetitle} inside a tikz node, so
% the beamer font never reaches it. Every frame title in this deck is therefore
% wrapped in \ft, whose size command is inside that group and wins. Patching the
% theme would be the other fix, and it is shared with conductance.tex.
\newcommand{\ft}[1]{{\large #1}}
\newlength{\panelbox}
\setlength{\panelbox}{0.68\textheight}
\newcommand{\fitgfx}[2][\linewidth]{%
  \includegraphics[width=#1,height=\panelbox,keepaspectratio]{#2}}
""",
         rf"\title{{{_tex(PAPER)} --- experimental plan}}",
         r"\subtitle{Results as they land}",
         r"\author{Cedric Allier}", r"\institute{Saalfeld lab, Janelia}",
         r"\date{\today}", r"\begin{document}", r"\frame{\titlepage}"]
    for p in paths:
        fm, _ = load(p)
        L.append(_slides(fm))
    L.append(r"\end{document}")

    build = os.path.join(ROOT, "presentation")
    tex_path = os.path.join(build, "_exp_report.tex")
    open(tex_path, "w").write("\n".join(L) + "\n")
    if not make_pdf:
        print(f"wrote {tex_path}")
        return 0
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "_exp_report.tex"], cwd=build,
                           capture_output=True, text=True)
    pdf = os.path.join(build, "_exp_report.pdf")
    if not os.path.exists(pdf):
        print(r.stdout[-2500:])
        return 1
    out = os.path.join(EXP_DIR, "report.pdf")
    shutil.copy2(pdf, out)
    for ext in (".aux", ".log", ".nav", ".out", ".snm", ".toc", ".tex", ".pdf"):
        try:
            os.remove(os.path.join(build, "_exp_report" + ext))
        except OSError:
            pass
    print(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("verb", choices=["launch", "analyse", "poll", "report"])
    ap.add_argument("numbers", nargs="*", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--arm", default=None, help="submit only this arm")
    ap.add_argument("--where", action="append", default=[], metavar="AXIS=V1,V2",
                    help="submit only these values of an axis, e.g. "
                         "--where noise=noise_005,noise_05")
    a = ap.parse_args(argv)

    if a.verb == "launch":
        if not a.numbers:
            raise SystemExit("launch needs an experiment number")
        where = [(w.split("=", 1)[0], set(w.split("=", 1)[1].split(",")))
                 for w in a.where]
        return max(launch(n, dry_run=a.dry_run, only_arm=a.arm, where=where)
                   for n in a.numbers)
    if a.verb == "analyse":
        ns = a.numbers or [int(re.match(r"exp(\d+)_", os.path.basename(p)).group(1))
                           for p in all_experiments()]
        return max(analyse(n, dry_run=a.dry_run) for n in ns)
    if a.verb == "poll":
        ns = a.numbers or [int(re.match(r"exp(\d+)_", os.path.basename(p)).group(1))
                           for p in all_experiments()]
        return max(poll(n) for n in ns)
    paths = [exp_path(n) for n in a.numbers] if a.numbers else all_experiments()
    return report(paths, make_pdf=not a.no_pdf)


if __name__ == "__main__":
    raise SystemExit(main())
