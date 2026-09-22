#!/usr/bin/env python
"""exp -- conduct one experimental plan: check it, watch it, print it.

The plan lives in experiments/ and is specified by experiments/SCHEMA.md. This
script is the three verbs over it, and it OWNS NOTHING: it does not generate
specs, does not submit jobs and does not compute a metric. Every number it
prints was written by a run, every spec it reads was written by you, and the
only thing it adds is the checking and the arrangement.

    exp check  [<id> ...]   assert the invariants; exit 1 on the first failure
    exp status [<id> ...]   one row per arm x fold: run, status, iteration, commit
    exp report [<id> ...]   regenerate experiments/report.pdf, blanks included

`check` is the reason the tool exists. An experiment is a claim that two sets of
runs differ in exactly one way, and that claim is checkable from the resolved
configs and the markers the runs leave behind. Until it is checked it is a
recollection.
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
_BOOKKEEPING = {"config_file", "description"}


# --------------------------------------------------------------------------- #
#  loading                                                                     #
# --------------------------------------------------------------------------- #
def load_plan():
    with open(os.path.join(PLAN_DIR, "plan.yaml")) as fh:
        return yaml.safe_load(fh)


def load_experiment(eid):
    path = os.path.join(PLAN_DIR, f"{eid}.yaml")
    if not os.path.exists(path):
        raise SystemExit(f"no such experiment: {path}")
    with open(path) as fh:
        exp = yaml.safe_load(fh)
    if exp.get("id") != eid:
        raise SystemExit(f"{path}: id is {exp.get('id')!r}, filename says {eid!r}")
    return exp


def run_name(arm, fold):
    """Binding B3: the run a given arm and fold produce."""
    return f"{arm['spec']}_{fold}"


def spec_paths(plan, arm, fold):
    """Binding B1: every config root that has this arm's spec for this fold.

    Returns the full list rather than the first hit, because a stem present in
    two roots is an error, not a precedence rule -- it means two different
    experiments are about to print in one table.
    """
    stem = f"{arm['spec']}_{fold}.yaml"
    out = []
    for root in plan["config_roots"]:
        root = root if os.path.isabs(root) else os.path.join(ROOT, root)
        p = os.path.join(root, stem)
        if os.path.exists(p):
            out.append(p)
    return out


def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def resolved(path):
    """The spec as the trainer sees it: defaults filled in, types coerced.

    Comparing the raw yamls would call a key that one file omits and the other
    sets to its default a DIFFERENCE, and would miss a key whose two spellings
    coerce to the same value.
    """
    from connectome_gnn.config import NeuralGraphConfig
    return _flatten(NeuralGraphConfig.from_yaml(path).model_dump())


def commit_of(plan, run):
    """The sha recorded in <run>/_completed_train, or None if it never trained."""
    p = os.path.join(plan["log_root"], run, "_completed_train")
    if not os.path.exists(p):
        return None
    for line in open(p):
        if line.startswith("commit="):
            return line.split("=", 1)[1].strip()
    return None


def metrics_of(plan, run):
    p = os.path.join(plan["log_root"], run, "results", "metrics.txt")
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


def iteration_of(plan, run):
    """The last iteration tmp_training/Wij.log reached, or None."""
    p = os.path.join(plan["log_root"], run, "tmp_training", "Wij.log")
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


def status_of(plan, run):
    if metrics_of(plan, run) is not None:
        return "landed"
    return "running" if iteration_of(plan, run) is not None else "pending"


# --------------------------------------------------------------------------- #
#  check                                                                       #
# --------------------------------------------------------------------------- #
def check(plan, exp):
    """Invariants I1-I8 of SCHEMA.md. Returns a list of failure strings."""
    bad = []
    eid, folds = exp["id"], exp["folds"]
    arms = {a["id"]: a for a in exp["arms"]}

    # I6 / I7: the plan closes over itself.
    for q in exp.get("feeds", []):
        if q not in plan.get("questions", {}):
            bad.append(f"I6 {eid}: feeds unknown question {q!r}")
    if exp["baseline"] not in arms:
        bad.append(f"I6 {eid}: baseline {exp['baseline']!r} is not an arm")
        return bad

    # I4: arm x fold -> run is injective.
    seen = {}
    for a in exp["arms"]:
        for f in folds:
            r = run_name(a, f)
            if r in seen:
                bad.append(f"I4 {eid}: {a['id']}/{f} and {seen[r]} both resolve to {r}")
            seen[r] = f"{a['id']}/{f}"

    # I1: every spec resolves, in exactly one root.
    paths = {}
    for a in exp["arms"]:
        for f in folds:
            ps = spec_paths(plan, a, f)
            if not ps:
                bad.append(f"I1 {eid}: no spec for {a['id']}/{f} "
                           f"({a['spec']}_{f}.yaml in any config root)")
            elif len(ps) > 1:
                bad.append(f"I1 {eid}: {a['spec']}_{f}.yaml is in {len(ps)} roots: "
                           + ", ".join(ps))
            else:
                paths[(a["id"], f)] = ps[0]

    base = exp["baseline"]
    for f in folds:
        if (base, f) not in paths:
            continue
        try:
            bcfg = resolved(paths[(base, f)])
        except Exception as e:
            bad.append(f"I1 {eid}: baseline {base}/{f} will not load: "
                       f"{type(e).__name__}: {e}")
            continue

        # I8: the preconditions the experiment declares about its own data.
        # The derivative-target flag, for instance, is a bit-exact no-op on a
        # dataset with no measurement noise, so an experiment about it run on
        # one would produce two identical arms and a table saying the bug does
        # not matter.
        for cond in exp.get("preconditions", []):
            key, op, want = cond["key"], cond["op"], cond["value"]
            got = bcfg.get(key)
            ok = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
                  "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
                  "==": lambda a, b: a == b, "!=": lambda a, b: a != b}[op](got, want)
            if not ok:
                bad.append(f"I8 {eid}/{f}: precondition {key} {op} {want} "
                           f"violated, it is {got!r}")

        for a in exp["arms"]:
            if a["id"] == base or (a["id"], f) not in paths:
                continue
            try:
                acfg = resolved(paths[(a["id"], f)])
            except Exception as e:
                bad.append(f"I1 {eid}: {a['id']}/{f} will not load: "
                           f"{type(e).__name__}: {e}")
                continue
            # I2: the controlled-variable audit, EQUALITY in both directions.
            # A key that changed but was not declared is an uncontrolled
            # variable; a key declared but unchanged is a claim the experiment
            # does not actually make.
            changed = {k for k in set(acfg) | set(bcfg)
                       if acfg.get(k) != bcfg.get(k)} - _BOOKKEEPING
            declared = set(a.get("overrides") or {})
            for k in sorted(changed - declared):
                bad.append(f"I2 {eid}/{f}: {a['id']} changes undeclared {k}: "
                           f"{bcfg.get(k)!r} -> {acfg.get(k)!r}")
            for k in sorted(declared - changed):
                bad.append(f"I2 {eid}/{f}: {a['id']} declares {k} but it is "
                           f"unchanged at {bcfg.get(k)!r}")

    # I5: one commit across every run that has trained, and none dirty.
    shas = {}
    for a in exp["arms"]:
        for f in folds:
            sha = commit_of(plan, run_name(a, f))
            if sha is not None:
                shas[f"{a['id']}/{f}"] = sha
    want = exp.get("commit")
    for who, sha in sorted(shas.items()):
        if sha.endswith("-dirty"):
            bad.append(f"I5 {eid}: {who} trained at a DIRTY tree ({sha})")
        if want and sha.split("-")[0] != want.split("-")[0]:
            bad.append(f"I5 {eid}: {who} trained at {sha}, plan says {want}")
    if not want and len(set(s.split('-')[0] for s in shas.values())) > 1:
        bad.append(f"I5 {eid}: arms trained at {len(set(shas.values()))} different "
                   f"commits and the plan pins none")
    return bad


# --------------------------------------------------------------------------- #
#  report                                                                      #
# --------------------------------------------------------------------------- #
def _cells(plan, m):
    """One row's cells, in the plan's column order. None -> blanks."""
    cols = plan["columns"]
    if m is None:
        return ["" for _ in (cols["scalars"] + cols["recovery"] + cols["extra"])]
    out = [num(m.get(k)) for k in cols["scalars"]]
    out += [r2(m.get(f"{k}_R2"), m.get(f"{k}_R2_all"), m.get(f"{k}_pct_outliers"))
            for k in cols["recovery"]]
    out += [num(m.get(k)) for k in cols["extra"]]
    return out


def _header(plan):
    cols = plan["columns"]
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


def table(plan, exp):
    cols = plan["columns"]
    keys = cols["scalars"] + [f"{k}_R2" for k in cols["recovery"]] + cols["extra"]
    head = _header(plan)
    ncol = len(head) + 1
    spec = ("p{3.4cm}" + r">{\raggedleft\arraybackslash}p{1.55cm}" * len(head))
    out = [r"\begin{table}[H]", r"\scriptsize", r"\raggedright",
           r"\setlength{\tabcolsep}{2pt}",
           rf"\caption{{{exp['report']['caption']}}}",
           rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
           " & ".join(["arm / fold"] + head) + r" \\", r"\midrule"]
    for a in exp["arms"]:
        acc = {k: [] for k in keys}
        for f in exp["folds"]:
            run = run_name(a, f)
            m = metrics_of(plan, run)
            label = f"{tex(a['id'])} / {tex(f)}" + ("" if m else r"$^{*}$")
            out.append(" & ".join([label] + _cells(plan, m)) + r" \\")
            out.append(rf"\multicolumn{{{ncol}}}{{@{{}}l@{{}}}}"
                       rf"{{\tiny\texttt{{{esc(run)}}}}} \\[1pt]")
            if m:
                for k in keys:
                    acc[k].append(m.get(k))
        out.append(r"\midrule")
        out.append(" & ".join([rf"\textbf{{{tex(a['label'])}}}"]
                              + [_mean_sd(acc[k]) for k in keys]) + r" \\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def report(plan, exps, make_pdf=True):
    L = [r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=0.8cm,landscape]{geometry}
\usepackage{booktabs,float,amsmath,xcolor,array}
\usepackage[T1]{fontenc}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\begin{document}"""]
    L.append(rf"""\begin{{center}}{{\Large {esc(plan['paper'])} --- experimental plan}}\\[2pt]
{{\small Every $R^2$ is written \emph{{outlier-filtered}} [full sample] (\% dropped),
final, from \texttt{{results/metrics.txt}} on the held-out test split.
Rows marked $^{{*}}$ are blank: those runs have not landed.}}\end{{center}}
\vspace{{4pt}}""")
    qs = plan.get("questions", {})
    if qs:
        L.append(r"\section*{Questions}")
        L.append(r"\begin{tabular}{lp{20cm}}\toprule")
        for qid, q in qs.items():
            L.append(rf"\texttt{{{esc(qid)}}} & {tex(q["text"])} \\")
        L.append(r"\bottomrule\end{tabular}")
    for exp in exps:
        feeds = ", ".join(exp.get("feeds", [])) or "--"
        L.append(rf"\section*{{{tex(exp['title'])}}}")
        L.append(rf"{{\small \textbf{{Question.}} {tex(exp['question'])} \quad "
                 rf"\textbf{{Feeds.}} \texttt{{{esc(feeds)}}} \quad "
                 rf"\textbf{{Commit.}} \texttt{{{esc(exp.get('commit') or 'not pinned')}}}}}")
        L.append(table(plan, exp))
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
    ap.add_argument("verb", choices=["check", "status", "report"])
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)

    plan = load_plan()
    ids = a.ids or plan["experiments"]
    exps = [load_experiment(i) for i in ids]

    if a.verb == "check":
        bad = [b for e in exps for b in check(plan, e)]
        for b in bad:
            print(b)
        print(f"{len(bad)} failure(s) over {len(exps)} experiment(s)")
        return 1 if bad else 0

    if a.verb == "status":
        print(f"{'run':62s} {'status':8s} {'iter':>9s}  commit")
        for e in exps:
            for arm in e["arms"]:
                for f in e["folds"]:
                    run = run_name(arm, f)
                    it = iteration_of(plan, run)
                    sha = commit_of(plan, run) or "--"
                    print(f"{run[:62]:62s} {status_of(plan, run):8s} "
                          f"{(f'{it:,}' if it else '--'):>9s}  {sha[:12]}")
        return 0

    return report(plan, exps, make_pdf=not a.no_pdf)


if __name__ == "__main__":
    raise SystemExit(main())
