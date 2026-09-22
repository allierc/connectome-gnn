#!/usr/bin/env python
"""exp -- watch an experimental plan and print it.

The plan lives in experiments/ and is specified by experiments/SCHEMA.md. This
script is the three verbs over it, and it OWNS NOTHING: it does not generate
specs, does not submit jobs and does not compute a metric. Every number it
prints was written by a run, every spec it reads was written by you, and the
only thing it adds is the checking and the arrangement.

    exp status [<id> ...]   one row per run: status, iteration, commit
    exp report [<id> ...]   regenerate experiments/report.pdf, blanks included

The plan yaml says what the grid is; `status` says which corners of it have
landed; `report` arranges the ones that have into the PDF and leaves the rest
blank. Nothing here validates anything -- a spec that is wrong will simply
produce a run whose numbers are wrong, and the yaml is the record of what was
intended.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLAN_DIR = os.path.join(ROOT, "experiments")

# WHAT plan.yaml USED TO HOLD. A separate file for four constants was a file to
# keep in step for no gain; the experiment yamls are the plan.
PAPER = "arxiv-2026"
LOG_ROOT = "/groups/saalfeld/home/allierc/GraphData/log/fly"

# THE ONE METRIC VOCABULARY, named once so a cell means the same thing in every
# table. `recovery` stems print as `clean [all] (pct dropped)`; the other two
# print as plain numbers, straight from results/metrics.txt.
COLUMNS = {
    "scalars": ["one_step_r", "rollout_r"],
    "recovery": ["Wij", "tau", "V_rest", "msg_i"],
    "extra": ["V_rest_R2_uncorrected", "tmpl_offset_per_neuron_absmedian",
              "clustering_accuracy", "Wij_gain"],
}
sys.path.insert(0, os.path.join(ROOT, "src"))

# The LaTeX formatting is the first document's, to the letter, so a cell means
# the same thing in this PDF as in docs/experiment_tables2.pdf. Loaded by path
# because docs/ is not a package; nothing but the formatters is used.
_spec = importlib.util.spec_from_file_location(
    "_tables2", os.path.join(ROOT, "docs", "make_experiment_tables2.py"))
_T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_T)
esc, num, r2, green = _T.esc, _T.num, _T.r2, _T.green

def tex(s):
    """LaTeX-escape a label the plan author wrote in prose.

    `esc` from the tables module only handles _ and %, which is right for a run
    name; a label is free text and `#` (as in "pre-#58") starts a macro
    parameter, which is a fatal error rather than a bad glyph.
    """
    out = str(s)
    for ch, rep in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                    ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                    ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                    ("^", r"\textasciicircum{}")):
        out = out.replace(ch, rep)
    return out


# Keys that MUST differ between two specs and say nothing about the experiment:
# every arm has its own file, so its own config_file, and a description written
# for a human.
def _pt(point):
    """A grid point as one short string, for messages and keys."""
    return "/".join(str(v) for v in point.values())


# --------------------------------------------------------------------------- #
#  loading                                                                     #
# --------------------------------------------------------------------------- #
def experiment_ids():
    """Every experiment yaml in experiments/, alphabetically.

    The section order of the PDF. Alphabetical rather than declared, because a
    declared order is a second list to keep in step with the directory.
    """
    import glob
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(PLAN_DIR, "*.yaml")))


def load_experiment(eid):
    path = os.path.join(PLAN_DIR, f"{eid}.yaml")
    if not os.path.exists(path):
        raise SystemExit(f"no such experiment: {path}")
    with open(path) as fh:
        exp = yaml.safe_load(fh)
    if exp.get("id") != eid:
        raise SystemExit(f"{path}: id is {exp.get('id')!r}, filename says {eid!r}")
    return exp


def points(exp):
    """Every cell of the experiment's grid, as a dict of axis -> value.

    An experiment is rarely one dimension. The derivative-target question is
    asked at three model-noise levels, so its grid is noise x arm x fold and
    `folds` alone cannot describe it. Axes are ordered as written; `fold` is
    conventional only in that the report averages over it.
    """
    import itertools
    axes = exp["axes"]
    names = list(axes)
    return [dict(zip(names, combo)) for combo in itertools.product(*axes.values())]


def group_of(exp, point):
    """A point's grid cell with `fold` removed: what a mean +- SD is taken over."""
    return tuple((k, point[k]) for k in exp["axes"] if k != "fold")


def spec_path(exp, arm, point):
    """The yaml this run is launched from, under the experiment's specs folder.

    `specs` in the experiment yaml is what makes the file self-contained: the
    baseline says what the configs derive FROM, `specs` says where they landed,
    and `spec` is their stem. Without it the only record of where the thirty
    files live is the PDF.
    """
    root = exp["specs"]
    root = root if os.path.isabs(root) else os.path.join(ROOT, root)
    return os.path.join(root, run_name(arm, point) + ".yaml")


def run_name(arm, point):
    """Binding B3: the run a given arm and grid point produce.

    `spec` is a format string over the axis names, so one arm covers the whole
    grid and the run name stays the thing on disk rather than a second naming
    scheme to keep in step.
    """
    return arm["spec"].format(**point)


def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def commit_of(run):
    """The sha recorded in <run>/_completed_train, or None if it never trained."""
    p = os.path.join(LOG_ROOT, run, "_completed_train")
    if not os.path.exists(p):
        return None
    for line in open(p):
        if line.startswith("commit="):
            return line.split("=", 1)[1].strip()
    return None


def metrics_of(run):
    p = os.path.join(LOG_ROOT, run, "results", "metrics.txt")
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    # A metrics.txt with only the rollout lines means the plot pass never
    # reached the readout: no recovered parameters, which for every column but
    # the rollout ones is the same state as no file at all.
    return out if "Wij_R2" in out else None


def iteration_of(run):
    """The last iteration tmp_training/Wij.log reached, or None."""
    p = os.path.join(LOG_ROOT, run, "tmp_training", "Wij.log")
    if not os.path.exists(p):
        return None
    last = None
    for line in open(p):
        if line and not line.startswith("#") and not line.startswith("iteration"):
            last = line
    if last is None:
        return None
    try:
        return int(last.split(",")[0])
    except (ValueError, IndexError):
        return None


def status_of(run):
    if metrics_of(run) is not None:
        return "landed"
    return "running" if iteration_of(run) is not None else "pending"


# --------------------------------------------------------------------------- #
#  report                                                                      #
# --------------------------------------------------------------------------- #
def _cells(m):
    """One row's cells, in the plan's column order. None -> blanks."""
    cols = COLUMNS
    if m is None:
        return ["" for _ in (cols["scalars"] + cols["recovery"] + cols["extra"])]
    out = [num(m.get(k)) for k in cols["scalars"]]
    out += [r2(m.get(f"{k}_R2"), m.get(f"{k}_R2_all"), m.get(f"{k}_pct_outliers"))
            for k in cols["recovery"]]
    out += [num(m.get(k)) for k in cols["extra"]]
    return out


def _header():
    cols = COLUMNS
    pretty = {"one_step_r": r"one-step $r$", "rollout_r": r"rollout $r$",
              "Wij": r"$W_{ij}$ $R^2$", "tau": r"$\tau$ $R^2$",
              "V_rest": r"$V_{rest}$ $R^2$", "msg_i": r"$\mathrm{msg}_i$ $R^2$",
              "Eij": r"$E_{ij}$ $R^2$",
              "V_rest_R2_uncorrected": r"$V_{rest}$ $R^2$ \tiny no $C_{ij}$",
              "tmpl_offset_per_neuron_absmedian": r"$|C_i|$ \tiny V",
              "clustering_accuracy": "cluster", "Wij_gain": r"$W_{ij}$ gain"}
    return [pretty.get(k, esc(k))
            for k in cols["scalars"] + cols["recovery"] + cols["extra"]]


def _mean_sd(vals):
    xs = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:
            xs.append(f)
    if not xs:
        return "--"
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    return rf"{m:.2f} $\pm$ {sd:.2f}"


def table(exp):
    """One table: a row per arm x grid point, a mean +- SD per group.

    A group is the grid cell with `fold` removed, so a 3 x 2 x 5 experiment
    prints thirty rows and six summaries -- the summaries being the comparison
    and the rows being what it rests on.
    """
    cols = COLUMNS
    keys = cols["scalars"] + [f"{k}_R2" for k in cols["recovery"]] + cols["extra"]
    head = _header()
    ncol = len(head) + 1
    spec = ("p{3.6cm}" + r">{\raggedleft\arraybackslash}p{1.5cm}" * len(head))
    out = [r"\begin{table}[H]", r"\scriptsize", r"\raggedright",
           r"\setlength{\tabcolsep}{2pt}",
           rf"\caption{{{exp['report']['caption']}}}",
           rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
           " & ".join(["arm / " + " / ".join(exp["axes"])] + head) + r" \\",
           r"\midrule"]
    pts = points(exp)
    for a in exp["arms"]:
        groups = []
        for pt in pts:
            g = group_of(exp, pt)
            if g not in groups:
                groups.append(g)
        for g in groups:
            acc = {k: [] for k in keys}
            for pt in [p_ for p_ in pts if group_of(exp, p_) == g]:
                run = run_name(a, pt)
                m = metrics_of(run)
                label = f"{tex(a['id'])} / {tex(_pt(pt))}" + ("" if m else r"$^{*}$")
                out.append(" & ".join([label] + _cells(m)) + r" \\")
                out.append(rf"\multicolumn{{{ncol}}}{{@{{}}l@{{}}}}"
                           rf"{{\tiny\texttt{{{esc(run)}}}}} \\[1pt]")
                if m:
                    for k in keys:
                        acc[k].append(m.get(k))
            gname = ", ".join(f"{k}={v}" for k, v in g) or "all"
            out.append(" & ".join(
                [rf"\textbf{{{tex(a['label'])}}} \tiny({tex(gname)})"]
                + [_mean_sd(acc[k]) for k in keys]) + r" \\")
            out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def report(exps, make_pdf=True):
    L = [r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=0.8cm,landscape]{geometry}
\usepackage{booktabs,float,amsmath,xcolor,array}
\usepackage[T1]{fontenc}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\begin{document}"""]
    L.append(rf"""\begin{{center}}{{\Large {esc(PAPER)} --- experimental plan}}\\[2pt]
{{\small Every $R^2$ is written \emph{{outlier-filtered}} [full sample] (\% dropped),
final, from \texttt{{results/metrics.txt}} on the held-out test split.
Rows marked $^{{*}}$ are blank: those runs have not landed.}}\end{{center}}
\vspace{{4pt}}""")
    for exp in exps:
        L.append(rf"\section*{{{tex(exp['title'])}}}")
        L.append(rf"{{\small \textbf{{Purpose.}} {tex(exp['purpose'])}}}")
        if exp.get("baseline"):
            L.append(r"\\[2pt]")
            L.append(rf"{{\small \textbf{{Baseline.}} "
                     rf"\texttt{{{esc(exp['baseline'])}}} \quad "
                     rf"\textbf{{Specs.}} \texttt{{{esc(exp.get('specs', '--'))}}}}}")
        L.append(table(exp))
    L.append(r"\end{document}")
    tex_path = os.path.join(PLAN_DIR, "report.tex")
    open(tex_path, "w").write("\n".join(L) + "\n")
    print(f"wrote {tex_path}")
    if not make_pdf:
        return 0
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "report.tex"], cwd=PLAN_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2500:])
        return 1
    print(f"wrote {os.path.join(PLAN_DIR, 'report.pdf')}")
    return 0


# --------------------------------------------------------------------------- #
#  cli                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("verb", choices=["status", "report"])
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)

    ids = a.ids or experiment_ids()
    exps = [load_experiment(i) for i in ids]

    if a.verb == "status":
        print(f"{'run':56s} {'spec':8s} {'status':8s} {'iter':>9s}  commit")
        for e in exps:
            for arm in e["arms"]:
                for pt in points(e):
                    run = run_name(arm, pt)
                    it = iteration_of(run)
                    sha = commit_of(run) or "--"
                    spec = "ok" if os.path.exists(spec_path(e, arm, pt)) else "MISSING"
                    print(f"{run[:56]:56s} {spec:8s} {status_of(run):8s} "
                          f"{(f'{it:,}' if it else '--'):>9s}  {sha[:12]}")
        return 0

    return report(exps, make_pdf=not a.no_pdf)


if __name__ == "__main__":
    raise SystemExit(main())
