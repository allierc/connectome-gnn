#!/usr/bin/env python
"""Build docs/experiment_tables2.pdf -- the conductance-on-conductance campaign.

SAME FORMAT AND SAME DISCIPLINE AS experiment_tables.tex, deliberately: landscape,
fixed column widths so tables line up down the page, every $R^2$ written
`clean [all] (pct dropped)`, captions above the table, and -- the rule that matters
-- A RUN THAT HAS NOT FINISHED GETS BLANK CELLS AND A STAR. Its train-split numbers
exist, in tmp_training/, but printing them in the same column as a held-out number
invites the comparison the star is there to forbid.

The live numbers are not suppressed, they are QUARANTINED: one table at the end,
its own columns, labelled as the training split with the iteration each number was
read at. That way the campaign can be read while it runs without any row implying
a held-out result it does not have.

A SECOND DOCUMENT rather than a section in the first, because the first is driven
by experiment_manifest.tsv and lists runs that have finished; this one reads the
log tree directly so it can be regenerated at any moment during the campaign.

Usage:
    python docs/make_experiment_tables2.py [--no-pdf]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from connectome_gnn.metrics import recovery_log_columns  # noqa: E402

LOG = "/groups/saalfeld/home/allierc/GraphData/log/fly"
HERE = os.path.dirname(os.path.abspath(__file__))
DATE = "2026-09-17"


# --------------------------------------------------------------------------- #
#  formatting, matching make_experiment_tables.py                              #
# --------------------------------------------------------------------------- #
def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def num(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "--"


def green(text, value):
    """Green above 0.5, red below 0. Same thresholds as the first document, so a
    colour means the same thing in both."""
    if value is None:
        return text
    if value >= 0.5:
        return rf"\textcolor{{ForestGreen}}{{{text}}}"
    if value < 0.0:
        return rf"\textcolor{{BrickRed}}{{{text}}}"
    return text


def r2(clean, allv=None, pct=None):
    """clean [all] (pct outliers) -- the three numbers metrics.txt keeps apart."""
    if clean in (None, "", "nan"):
        return "--"
    try:
        x = float(clean)
    except (TypeError, ValueError):
        return "--"
    s = num(x)
    if allv not in (None, "", "nan"):
        s += f" [{num(allv)}]"
    if pct not in (None, "", "nan"):
        try:
            s += f" ({float(pct):.1f})"
        except ValueError:
            pass
    return green(s, x)


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
    return out


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
def table(entries, caption):
    """The held-out table. Same column set and widths as the first document."""
    _R = r">{\raggedleft\arraybackslash}p{2.6cm}"
    _P = r">{\raggedleft\arraybackslash}p{1.15cm}"
    ncols = r"p{3.2cm}" + _P * 2 + _R * 5
    head = ["arm", "one-step $r$", "rollout $r$",
            r"$W_{ij}$ $R^2$", r"$E_{ij}$ $R^2$", r"$\mathrm{msg}_i$ $R^2$",
            r"$\tau$ $R^2$", r"$V_{rest}$ $R^2$"]
    out = [r"\begin{table}[H]", r"\scriptsize", r"\raggedright",
           r"\setlength{\tabcolsep}{1.5pt}", rf"\caption{{{caption}}}",
           r"\setlength{\tabcolsep}{3pt}",
           rf"\begin{{tabular}}{{{ncols}}}", r"\toprule",
           " & ".join(head) + r" \\", r"\midrule"]
    for a in entries:
        m = metrics(a["run"])
        if m is None:
            # PENDING: blank cells and a star. Its train-split numbers are in the
            # quarantined table at the end, never in this column.
            out.append(esc(a["label"]) + r"$^{*}$" + " & " * 7 + r" \\")
            out.append(rf"\multicolumn{{8}}{{l}}{{\tiny\texttt{{{esc(a['run'])}}}}} \\")
            continue
        cells = [esc(a["label"]),
                 num(m.get("one_step_r")), num(m.get("rollout_r")),
                 r2(m.get("Wij_R2"), m.get("Wij_R2_all"), m.get("Wij_pct_outliers")),
                 r2(m.get("Eij_R2"), m.get("Eij_R2_all"), m.get("Eij_pct_outliers")),
                 r2(m.get("msg_i_R2"), m.get("msg_i_R2_all"), m.get("msg_i_pct_outliers")),
                 r2(m.get("tau_R2"), m.get("tau_R2_all"), m.get("tau_pct_outliers")),
                 r2(m.get("V_rest_R2"), m.get("V_rest_R2_all"), m.get("V_rest_pct_outliers"))]
        out.append(" & ".join(cells) + r" \\")
        out.append(rf"\multicolumn{{8}}{{l}}{{\tiny\texttt{{{esc(a['run'])}}}}} \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def live_table(entries, caption):
    """QUARANTINED: the training split, with the iteration each number was read
    at. Separate columns and a separate caption so no row can be mistaken for a
    held-out result."""
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

    L = [r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=0.8cm,landscape]{geometry}
\usepackage{booktabs,float,amsmath,array}
\usepackage[dvipsnames]{xcolor}
\usepackage[T1]{fontenc}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\begin{document}
\begin{center}{\Large Conductance GNN on task-trained conductance data}\\[2pt]""",
         rf"""{{\small {DATE}. Every $R^2$ is written \emph{{outlier-filtered}} [full sample]
(\% dropped), final, from \texttt{{results/metrics.txt}} on the held-out test split.
Rows marked $^{{*}}$ are left blank: those runs are still training and have no
held-out numbers yet -- their training-split numbers are quarantined in the last
table.}}\end{{center}}
\vspace{{4pt}}""",
         r"""
The generator is a flyvis network trained on the OPTIC-FLOW TASK
(\texttt{flow/2000/001}, checkpoint 28), not a teacher--student twin fitted to
reproduce a current model. Every earlier conductance result came from such a twin,
which is what made the two synapse models hard to separate; here the conductance in
the data was shaped by a task instead.

Ten datasets, five folds at $\sigma=0$ and five at $\sigma=0.05$, 13{,}741 neurons
and 434{,}112 edges. They integrate with an exponential-Euler step: forward Euler
contracts only while $(\Delta t/\tau_i)(1+G_i)<2$ and the generating network reaches
4.4 at $\Delta t = 20$\,ms, so a matched forward-Euler control diverges at frame 8
and is 95.8\% NaN. For the CURRENT model the two integrators agree to
$7.2\times10^{-7}$ -- the step is decisive for one model and irrelevant for the other.
"""]

    L.append(rf"\section*{{{DATE} --- the lasso sweep}}")
    L.append(r"""
The group lasso over $g_\phi$'s input columns zeroes the $v_i$ and $a_i$ columns,
which is correct on CURRENT data where the message does not depend on the
postsynaptic voltage. On conductance data $(E_{ij}-v_i)$ carries the sign of the
message, so the same penalty removes the term the model needs: $R^2_W$ should fall
with $\lambda$, and if it does not, the GNN was not using the driving force.
""")
    for sig in ("0", "0.05"):
        e = pick("lasso", sig)
        if e:
            L.append(table(e, rf"Conductance GNN on conductance data, $\sigma = {sig}$."))
    e = pick("rc10")
    if e:
        L.append(table(e, r"""As above, $\sigma=0$, under a rollout:
\texttt{rollout\_horizon\_schedule} $[1,2,4,7,10]$, one entry per epoch. $K=1$ is
term-for-term the one-step objective, so this is a strict extension of the table
above rather than a different experiment."""))
    e = pick("gsil")
    if e:
        L.append(table(e, r"""$\lambda=0$ with \texttt{coeff\_g\_phi\_silent} $=5$
over $[-2,0]$, pinning the message's level where the presynaptic cell is quiet.
One parameter away from the $\lambda=0$ rows above."""))

    L.append(r"\section*{Training split --- not held-out, not comparable to the above}")
    L.append(r"""
Read at the iteration shown, from \texttt{tmp\_training/<key>.log}, on the data the
model is being fitted to. Two rows are comparable to each other only if their
iterations match.
""")
    L.append(live_table(ARMS, r"""Progress of the runs still training.
$W_{ij}$ gain is learned/true: a gain far from 1 means the message scale is free
and $W$ has absorbed it, which is a GAUGE failure rather than a recovery failure --
$E_{ij}$ can be recovered while $W$ is off by that one factor."""))

    L.append(r"""
\section*{What is already visible}

$R^2_{\mathrm{msg}}$ sits at 0.78--0.92 and $E_{ij}$ at 0.55--0.73 while $R^2_W$ is
between $-0.30$ and $+0.15$ -- and the $W_{ij}$ gain is 0.21--0.32. The conductances
are recovered up to a single scale factor of three to five, with a median relative
error of 0.96 that is almost entirely that factor. This is NOT the $W$--$E$
degeneracy of the raw product: the template readout fits the generator's own closed
form per edge, and it recovers $E$. What is free is the message's overall scale,
which the ODE fixes by requiring $\tau_i\,\partial f_\theta/\partial\mathrm{msg}_i = 1$
and which nothing in these runs penalises --
\texttt{coeff\_f\_theta\_msg\_gain} is 0 in all of them.

No verdict on the lasso. At comparable iterations the differences between $\lambda$
values are smaller than the differences between FOLDS, and they do not move
consistently in one direction. Two folds cannot separate them.
\end{document}""")

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
