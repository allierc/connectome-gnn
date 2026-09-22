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
        for combo in itertools.product(*axes.values()):
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


def live_of(run):
    """The train-split numbers a running job has written, and its iteration."""
    out, it = {}, None
    for key, stem in _LIVE_FILE.items():
        i, d = _last_row(run, stem)
        if i is not None:
            it = i if it is None else max(it, i)
            if key in d:
                out[key] = d[key]
    return it, out


def commit_of(run):
    for marker in ("_completed_train", "_complete"):
        p = os.path.join(LOG_ROOT, run, marker)
        if os.path.exists(p):
            for line in open(p):
                if line.startswith("commit="):
                    return line.split("=", 1)[1].strip()
    return None


def status_of(run):
    if metrics_of(run) is not None:
        return "landed"
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
    import shutil
    src_dir = os.path.join(ROOT, fm["specs_dir"])
    for n in names:
        src = os.path.join(src_dir, f"{n}.yaml")
        if not os.path.exists(src):
            raise SystemExit(f"missing spec: {src}")
        shutil.copy2(src, os.path.join(STAGE_DIR, f"{n}.yaml"))
    return names


def launch(number, dry_run=False):
    path = exp_path(number)
    fm, _ = load(path)
    names = [r for _, _, r in runs(fm)]
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
        os.makedirs(log_dir, exist_ok=True)
        jid, queue, res = _bsub_over_ssh(
            cluster_cmd=f"python GNN_Main.py -o {fm['task']} {n}",
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
        fm["job_ids"] = ids
        _rewrite_front_matter(path, fm)
    print(f"{len(ids)}/{len(names)} submitted")
    return 0 if len(ids) == len(names) else 1


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


def _summary_rows(fm, rs, source):
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
        its, n_here = [], 0
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
            else:
                if status_of(run) != "running":
                    continue
                it, d = live_of(run)
                if it is None:
                    continue
                its.append(it)
                n_here += 1
                for k, _, live in COLUMNS:
                    if live:
                        acc[k].append(d.get(k))
        if not n_here:
            continue
        cells = [_mean_sd(acc[k]) if (source == "landed" or live) else ""
                 for k, _, live in COLUMNS]
        out.append((arm_id, cell, (f"{max(its):,}" if its else str(n_here)), cells))
    return out


def status_block(fm):
    rs = runs(fm)
    counts = {"landed": 0, "running": 0, "pending": 0}
    for _arm, _pt, run in rs:
        counts[status_of(run)] += 1
    axes_no_fold = [k for k in fm["axes"] if k != "fold"]
    heads = [h for _, h, _ in COLUMNS]
    sep = "|" + "---|" * (2 + len(axes_no_fold) + len(heads))

    L = [_BEGIN, "", "## Status", "",
         f"**{counts['landed']}/{len(rs)} landed**, "
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


def _page(fm):
    """One page: title, purpose, and the slide-22 table over the landed runs."""
    rs = runs(fm)
    heads = [h for _, h, _ in COLUMNS]
    axes_no_fold = [k for k in fm["axes"] if k != "fold"]
    spec = ("p{2.1cm}" + "l" * len(axes_no_fold) + "r"
            + r">{\raggedleft\arraybackslash}p{1.5cm}" * len(heads))
    L = [rf"\section*{{Experiment {fm['number']} --- {_tex(fm['name'])}}}",
         rf"{{\small \textbf{{{_tex(fm['title'])}}}}}\\[2pt]",
         rf"{{\small \textbf{{Purpose.}} {_tex(fm['purpose'])}}}\\[2pt]",
         rf"{{\small \textbf{{Baseline.}} \texttt{{{_tex(fm['baseline'])}}}}}",
         r"\vspace{8pt}", r"\begin{table}[H]", r"\scriptsize", r"\raggedright",
         r"\setlength{\tabcolsep}{2.5pt}",
         rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
         " & ".join(["arm"] + [_tex(k) for k in axes_no_fold] + ["n"]
                    + [_tex(h) for h in heads]) + r" \\", r"\midrule"]
    rows = _summary_rows(fm, rs, "landed")
    if not rows:
        L.append(" & ".join(["---"] * (2 + len(axes_no_fold) + len(heads))) + r" \\")
    for arm_id, cell, n, cells in rows:
        L.append(" & ".join([_tex(arm_id)] + [_tex(v) for _, v in cell] + [n]
                            + [c.replace("±", r"$\pm$") for c in cells]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def report(paths, make_pdf=True):
    L = [r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=0.8cm,landscape]{geometry}
\usepackage{booktabs,float,amsmath,xcolor,array}
\usepackage[T1]{fontenc}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\begin{document}"""]
    L.append(rf"""\begin{{center}}{{\Large {_tex(PAPER)} --- experiments}}\\[2pt]
{{\small Mean $\pm$ SD over the folds that have landed, held-out, from
\texttt{{results/metrics.txt}}; \texttt{{n}} is how many. Spec names are not
repeated here --- they live in the experiment's markdown file.}}\end{{center}}
\vspace{{6pt}}""")
    for i, p in enumerate(paths):
        fm, _ = load(p)
        if i:
            L.append(r"\newpage")
        L.append(_page(fm))
    L.append(r"\end{document}")
    tex_path = os.path.join(EXP_DIR, "report.tex")
    open(tex_path, "w").write("\n".join(L) + "\n")
    if not make_pdf:
        print(f"wrote {tex_path}")
        return 0
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "report.tex"], cwd=EXP_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2500:])
        return 1
    print(f"wrote {os.path.join(EXP_DIR, 'report.pdf')}")
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("verb", choices=["launch", "poll", "report"])
    ap.add_argument("numbers", nargs="*", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)

    if a.verb == "launch":
        if not a.numbers:
            raise SystemExit("launch needs an experiment number")
        return max(launch(n, dry_run=a.dry_run) for n in a.numbers)
    if a.verb == "poll":
        ns = a.numbers or [int(re.match(r"exp(\d+)_", os.path.basename(p)).group(1))
                           for p in all_experiments()]
        return max(poll(n) for n in ns)
    paths = [exp_path(n) for n in a.numbers] if a.numbers else all_experiments()
    return report(paths, make_pdf=not a.no_pdf)


if __name__ == "__main__":
    raise SystemExit(main())
