#!/usr/bin/env python
"""Build docs/experiment_tables2.pdf -- the conductance-on-conductance campaign.

THE FIRST DOCUMENT'S FORMAT, TO THE LETTER: landscape, the same preamble, the same
fixed column widths, every $R^2$ written `clean [all] (pct dropped)`, two decimals,
green above 0.9 and no other colour, a one-sentence caption above each table, the
config name on its own `\\multicolumn` line under its row, and a `mean $\\pm$ SD`
row closing every table. NO PROSE: the first document is tables under dated
sections and nothing else, and any paragraph here would be the one thing a reader
has to decide whether to trust separately from the numbers.

A run that has not finished gets BLANK CELLS AND A STAR, also from the first
document. Its train-split numbers exist in tmp_training/ but printing them in the
same column as a held-out number invites the comparison the star forbids; they are
QUARANTINED in a table of their own, with the iteration each was read at.

A SECOND DOCUMENT rather than a section in the first, because the first is driven
by experiment_manifest.tsv and lists runs that have finished; this one reads the
log tree directly so it can be regenerated at any moment during the campaign.

Usage:
    python docs/make_experiment_tables2.py [--no-pdf]
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from connectome_gnn.metrics import recovery_log_columns  # noqa: E402

LOG = "/groups/saalfeld/home/allierc/GraphData/log/fly"
HERE = os.path.dirname(os.path.abspath(__file__))
DATE = "2026-09-17"

# The first document's threshold, so a green cell means the same thing in both.
_GREEN = 0.9


# --------------------------------------------------------------------------- #
#  formatting, copied from make_experiment_tables.py                           #
# --------------------------------------------------------------------------- #
def green(text, value):
    """Above 0.9 is green, and nothing is red. Applied to the value the cell leads
    with, never to the bracketed full-sample figure."""
    if value is None or value != value or value <= _GREEN:
        return text
    return r"\textcolor{green!45!black}{" + text + "}"


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def num(v):
    """Two decimals, or a bare integer once the value is off the scale R2 lives on."""
    if v in ("", None):
        return "--"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "--"
    if f != f:
        return "nan"
    if abs(f) >= 1000:
        return f"{f:.0f}"
    if abs(f) >= 100:
        return f"{f:.1f}"
    return f"{f:.2f}"


def r2(clean, allv=None, pct=None):
    """clean [all] (pct outliers) -- the three numbers metrics.txt keeps apart."""
    if clean in (None, "", "nan"):
        return "--"
    try:
        x = float(clean)
    except (TypeError, ValueError):
        return "--"
    s = num(clean)
    if allv not in (None, "", "nan"):
        s += f" [{num(allv)}]"
    if pct not in (None, "", "nan"):
        try:
            s += f" ({float(pct):.1f})"
        except ValueError:
            pass
    return green(s, x)


def mean_sd(values):
    """`0.93 $\\pm$ 0.03` over the rows that have a number, `--` when none do."""
    xs = [float(v) for v in values
          if v not in (None, "", "nan") and float(v) == float(v)]
    if not xs:
        return "--"
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    return rf"{m:.2f} $\pm$ {sd:.2f}"


# --------------------------------------------------------------------------- #
#  reading runs                                                                #
# --------------------------------------------------------------------------- #
def metrics(run):
    p = f"{LOG}/{run}/results/metrics.txt"
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    # A metrics.txt with only the rollout lines means `data_plot` never reached
    # the template readout -- the run has no recovered parameters, which is the
    # same state as no file at all for every column but the two rollout ones.
    return out if "Wij_R2" in out else None


def live(run, key, col):
    p = f"{LOG}/{run}/tmp_training/{key}.log"
    if not os.path.exists(p):
        return None, None
    rows = [r for r in open(p).read().strip().split("\n") if r]
    if not rows:
        return None, None
    v = rows[-1].split(",")
    return int(float(v[0])), dict(zip(recovery_log_columns(key), v[1:])).get(col)


ARMS = []
for _noise, _sig in (("noise_free", "0"), ("noise_005", "0.05")):
    for _lam, _tag in ((0, "lasso0"), (0.1, "lasso0p1"), (0.25, "lasso0p25")):
        for _cv in (0, 1):
            ARMS.append(dict(block="lasso", sigma=_sig,
                             label=rf"$\lambda={_lam}$, fold {_cv:02d}",
                             run=f"flyvis_flowcond_{_noise}_gnn_{_tag}_cv{_cv:02d}"))
for _lam, _tag in ((0, "lasso0"), (0.1, "lasso0p1"), (0.25, "lasso0p25")):
    for _cv in (0, 1):
        ARMS.append(dict(block="rc10", sigma="0",
                         label=rf"$\lambda={_lam}$, fold {_cv:02d}",
                         run=f"flyvis_flowcond_noise_free_gnn_{_tag}_rc10_cv{_cv:02d}"))
for _noise, _sig in (("noise_free", "0"), ("noise_005", "0.05")):
    for _cv in (0, 1):
        ARMS.append(dict(block="gsil", sigma=_sig, label=rf"silent anchor, fold {_cv:02d}",
                         run=f"flyvis_flowcond_{_noise}_gnn_gsil_lasso0_cv{_cv:02d}"))


def pick(block, sigma=None):
    return [a for a in ARMS
            if a["block"] == block and (sigma is None or a["sigma"] == sigma)
            and os.path.isdir(f"{LOG}/{a['run']}")]


# --------------------------------------------------------------------------- #
#  tables                                                                      #
# --------------------------------------------------------------------------- #
# key in metrics.txt -> column header, in the first document's order.
_COLS = [("Wij", r"$W_{ij}$ $R^2$"), ("tau", r"$\tau$ $R^2$"),
         ("V_rest", r"$V_{rest}$ $R^2$"), ("msg_i", r"$\mathrm{msg}_i$ $R^2$"),
         ("Eij", r"$E_{ij}$ $R^2$")]


def table(entries, caption):
    """The held-out table. Same column set, widths and closing mean row as the
    first document's tables."""
    _R = r">{\raggedleft\arraybackslash}p{2.6cm}"
    _P = r">{\raggedleft\arraybackslash}p{1.15cm}"
    ncols = r"p{3.2cm}" + _P * 2 + _R * len(_COLS)
    head = ["arm", "one-step $r$", "rollout $r$"] + [h for _, h in _COLS]
    n = len(head)
    out = [r"\begin{table}[H]", r"\scriptsize", r"\raggedright",
           r"\setlength{\tabcolsep}{1.5pt}", rf"\caption{{{caption}}}",
           r"\setlength{\tabcolsep}{3pt}",
           rf"\begin{{tabular}}{{{ncols}}}", r"\toprule",
           " & ".join(head) + r" \\", r"\midrule"]
    cols = {k: [] for k in ("one_step_r", "rollout_r", *(k for k, _ in _COLS))}
    for a in entries:
        m = metrics(a["run"])
        if m is None:
            out.append(a["label"] + r"$^{*}$" + " & " * (n - 1) + r" \\")
        else:
            cells = [a["label"], num(m.get("one_step_r")), num(m.get("rollout_r"))]
            cols["one_step_r"].append(m.get("one_step_r"))
            cols["rollout_r"].append(m.get("rollout_r"))
            for k, _ in _COLS:
                cells.append(r2(m.get(f"{k}_R2"), m.get(f"{k}_R2_all"),
                                m.get(f"{k}_pct_outliers")))
                cols[k].append(m.get(f"{k}_R2"))
            out.append(" & ".join(cells) + r" \\")
        out.append(rf"\multicolumn{{{n}}}{{@{{}}l@{{}}}}"
                   rf"{{\tiny\texttt{{{esc(a['run'])}}}}} \\[1pt]")
    out += [r"\midrule",
            " & ".join([r"\textbf{mean} $\pm$ SD",
                        mean_sd(cols["one_step_r"]), mean_sd(cols["rollout_r"])]
                       + [mean_sd(cols[k]) for k, _ in _COLS]) + r" \\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def live_table(entries, caption):
    """QUARANTINED: the training split, with the iteration each number was read at.
    Separate columns and its own caption so no row can be read as held-out."""
    _P = r">{\raggedleft\arraybackslash}p{1.6cm}"
    out = [r"\begin{table}[H]", r"\scriptsize", r"\raggedright",
           r"\setlength{\tabcolsep}{1.5pt}", rf"\caption{{{caption}}}",
           r"\setlength{\tabcolsep}{3pt}",
           rf"\begin{{tabular}}{{p{{5.4cm}}{_P * 6}}}", r"\toprule",
           " & ".join(["run (training split)", "iteration", r"$W_{ij}$ $R^2$",
                       r"$W_{ij}$ gain", r"$E_{ij}$ $R^2$",
                       r"$\mathrm{msg}_i$ $R^2$", r"$\tau$ $R^2$"]) + r" \\",
           r"\midrule"]
    for a in entries:
        if metrics(a["run"]):
            continue
        it, w = live(a["run"], "Wij", "Wij_R2")
        if it is None:
            continue
        _, g = live(a["run"], "Wij", "Wij_gain")
        _, e = live(a["run"], "Eij", "Eij_R2")
        _, ms = live(a["run"], "msg_i", "msg_i_R2")
        _, t = live(a["run"], "tau", "tau_R2")
        out.append(" & ".join([rf"\texttt{{{esc(a['run'])}}}", f"{it:,}",
                               r2(w), num(g), r2(e), r2(ms), r2(t)]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)

    # The first document's preamble verbatim, including plain xcolor: green!45!black
    # is the only colour either document uses.
    L = [r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=0.8cm,landscape]{geometry}
\usepackage{booktabs,float,amsmath,xcolor,array}
\usepackage[T1]{fontenc}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\begin{document}
\begin{center}{\Large Conductance GNN on task-trained conductance data}\\[2pt]
{\small Every $R^2$ is written \emph{outlier-filtered} [full sample] (\% dropped),
final, from \texttt{results/metrics.txt} on the held-out test split.
Rows marked $^{*}$ are left blank: those runs are still training and have no
held-out numbers yet.}\end{center}
\vspace{4pt}""",
         rf"\section*{{{DATE}}}"]

    # THE FIRST DOCUMENT'S CAPTION TEMPLATE, unchanged:
    # "<Model> model on <data> data with <knob>, $\sigma = <noise>$."
    for sig in ("0", "0.05"):
        e = pick("lasso", sig)
        if e:
            L.append(table(e, "Conductance model on conductance data with a group "
                              rf"lasso, $\sigma = {sig}$."))
    e = pick("rc10")
    if e:
        L.append(table(e, "Conductance model on conductance data with a group lasso "
                          r"and a $K = 10$ rollout, $\sigma = 0$."))
    e = pick("gsil")
    if e:
        L.append(table(e, "Conductance model on conductance data with a silent-input "
                          r"anchor, $\sigma = 0$ and $0.05$."))

    L.append(live_table(
        ARMS, "Conductance model on conductance data, training split at the "
              "iteration shown."))
    L.append(r"\end{document}")

    tex = os.path.join(HERE, "experiment_tables2.tex")
    open(tex, "w").write("\n".join(L) + "\n")
    print(f"wrote {tex}")
    if a.no_pdf:
        return 0
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "experiment_tables2.tex"], cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2500:])
        return 1
    print(f"wrote {os.path.join(HERE, 'experiment_tables2.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
