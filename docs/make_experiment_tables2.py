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
same column as a held-out number invites the comparison the star forbids, so they
are QUARANTINED in a table of their own, with the iteration each was read at and
no shared column with the tables above.

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

import torch  # noqa: E402  -- only for reading loss_components.pt

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


# THE GAUGE GRID, launched 2026-09-17: the two terms that fix the two halves of
# the affine message redundancy, crossed, at one fold and two lasso strengths.
# `coeff_g_phi_silent` pins the message's LEVEL (beta_i -> 0) at relu(v_j) = 0,
# where the true message is exactly zero; `coeff_f_theta_msg_gain` pins its
# SCALE (k_i -> 1) by charging df/dmsg + df/dv, which the generator makes zero
# for every neuron whatever its tau. The nominal config carries neither.
_SIL = [("s1", 1), ("s5", 5), ("s25", 25)]
_GAIN = [("g1e5", "10^{-5}"), ("g1e4", "10^{-4}"), ("g1e3", "10^{-3}")]

def live(run, key, col):
    """The last row of tmp_training/<key>.log, by column NAME. `#` lines are the
    readout descriptor, not data."""
    p = f"{LOG}/{run}/tmp_training/{key}.log"
    if not os.path.exists(p):
        return None, None
    rows = [r for r in open(p).read().strip().split("\n")
            if r and not r.startswith("#")]
    if len(rows) < 2:
        return None, None
    v = rows[-1].split(",")
    return int(float(v[0])), dict(zip(recovery_log_columns(key), v[1:])).get(col)


def gauge_k(run):
    """Measured k_i = tau_i * df/dmsg_i, median over neurons, from gauge_k.json.

    THE QUANTITY THE GRID EXISTS TO MOVE, and the only one that says outright
    whether the gauge is fixed: k = 1 is fixed, and W is then in the generator's
    units. Measured by autograd on real frames with the MODEL's own tau (blind --
    no ground truth in it) by tools/measure_gauge_k.py, cached here because it
    needs a checkpoint and a data pass and this script must stay instant.
    """
    import json
    p = os.path.join(HERE, "gauge_k.json")
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p)).get(run)
    return (d["k_median"], d["iteration"]) if d else (None, None)


def loss_term(run, name):
    """The last value of one regularizer component, from loss_components.pt.

    THE COLUMN THAT SAYS WHETHER AN ARM IS AN ARM. A coefficient in the yaml is
    not a penalty in the loss: the term it multiplies can be so small that the
    arm is indistinguishable from the control, and reading only the outcome
    columns would record that as "the term did not help" rather than "the term
    was never applied".
    """
    p = f"{LOG}/{run}/loss_components.pt"
    if not os.path.exists(p):
        return None
    try:
        v = torch.load(p, map_location="cpu", weights_only=False).get(name)
        return float(v[-1]) if v is not None and len(v) else None
    except Exception:
        return None


ARMS = []
for _ltag, _lam in (("lasso0", "0"), ("lasso0p1", "0.1")):
    for _stag, _sil in _SIL:
        for _gtag, _g in _GAIN:
            ARMS.append(dict(
                block=f"grid_{_ltag}", sigma="0.05",
                label=rf"silent {_sil}, gain ${_g}$",
                run=f"flyvis_flowcond_noise_005_gnn_{_stag}{_gtag}_{_ltag}_cv00"))
# The conductance model on CURRENT data at lasso 0.1, two folds: the one cell
# missing from the group-lasso sweep of the first document.
for _cv in (0, 1):
    ARMS.append(dict(block="cur0p1", sigma="0.05", label=rf"fold {_cv:02d}",
                     run=f"flyvis_current_noise_005_conductance_lasso_0p1_cv{_cv:02d}"))


def pick(block, sigma=None):
    return [a for a in ARMS
            if a["block"] == block and (sigma is None or a["sigma"] == sigma)]


# --------------------------------------------------------------------------- #
#  tables                                                                      #
# --------------------------------------------------------------------------- #
# THE FIRST DOCUMENT'S TWELVE COLUMNS, in its order and at its widths, so a
# table from either document can be read against the other without re-learning
# the layout. Four scalar columns at 1.15 cm, five R2 columns at 2.6 cm, the
# three-part fit R2 at 2.4 cm, cluster accuracy at 1.15 cm.
#
# The four scalars, with the metrics.txt key each reads:
#   fit roll r, same form   template_rollout_r       the template's own constants
#                                                    put back into the generator's
#                                                    equation and integrated
#   fit roll r, other form  template_alt_rollout_r   the same, with the OTHER
#                                                    family's constants
#   one-step r              one_step_r
#   rollout r               rollout_r
_SCALARS = [("template_rollout_r", r"fit roll $r$ \tiny same form"),
            ("template_alt_rollout_r", r"fit roll $r$ \tiny other form"),
            ("one_step_r", "one-step $r$"),
            ("rollout_r", "rollout $r$")]
# The five recovered quantities, `clean [all] (pct dropped)` each.
_COLS = [("Wij", r"$W_{ij}$ $R^2$"), ("tau", r"$\tau$ $R^2$"),
         ("V_rest", r"$V_{rest}$ $R^2$"), ("msg_i", r"$\mathrm{msg}_i$ $R^2$"),
         ("Eij", r"$E_{ij}$ $R^2$")]
# Does the model obey the generator's equation at all: the median per-neuron R2
# of the update template, then the median per-edge R2 under each synapse family.
_FIT3 = ("update_form_r2_median", "conductance_form_r2_median",
         "current_form_r2_median")


def table(entries, caption):
    """The held-out table: the first document's twelve columns, its widths, and
    its closing mean row."""
    _R = r">{\raggedleft\arraybackslash}p{2.6cm}"
    _P = r">{\raggedleft\arraybackslash}p{1.15cm}"
    ncols = (r"p{2.8cm}" + _P * len(_SCALARS) + _R * len(_COLS)
             + r">{\raggedleft\arraybackslash}p{2.4cm}" + _P)
    head = (["arm"] + [h for _, h in _SCALARS] + [h for _, h in _COLS]
            + [r"fit $R^2$ \tiny upd/cond/cur", "cluster acc."])
    n = len(head)
    out = [r"\begin{table}[H]", r"\scriptsize", r"\raggedright",
           r"\setlength{\tabcolsep}{1.5pt}", rf"\caption{{{caption}}}",
           r"\setlength{\tabcolsep}{3pt}",
           rf"\begin{{tabular}}{{{ncols}}}", r"\toprule",
           " & ".join(head) + r" \\", r"\midrule"]
    cols = {k: [] for k, _ in _SCALARS}
    cols.update({k: [] for k, _ in _COLS})
    cols["cluster"] = []
    for a in entries:
        m = metrics(a["run"])
        if m is None:
            out.append(a["label"] + r"$^{*}$" + " & " * (n - 1) + r" \\")
        else:
            cells = [a["label"]]
            for k, _ in _SCALARS:
                cells.append(num(m.get(k)))
                cols[k].append(m.get(k))
            for k, _ in _COLS:
                cells.append(r2(m.get(f"{k}_R2"), m.get(f"{k}_R2_all"),
                                m.get(f"{k}_pct_outliers")))
                cols[k].append(m.get(f"{k}_R2"))
            # Three numbers in one cell, slashed, each coloured on its own value
            # -- the first document's `1.00 / 1.00 / 1.00`.
            _f = [m.get(k) for k in _FIT3]
            cells.append(" / ".join(
                green(num(v), float(v)) if v not in (None, "", "nan") else "--"
                for v in _f))
            cells.append(num(m.get("clustering_accuracy")))
            cols["cluster"].append(m.get("clustering_accuracy"))
            out.append(" & ".join(cells) + r" \\")
        out.append(rf"\multicolumn{{{n}}}{{@{{}}l@{{}}}}"
                   rf"{{\tiny\texttt{{{esc(a['run'])}}}}} \\[1pt]")
    out += [r"\midrule",
            " & ".join([r"\textbf{mean} $\pm$ SD"]
                       + [mean_sd(cols[k]) for k, _ in _SCALARS]
                       + [mean_sd(cols[k]) for k, _ in _COLS]
                       + ["--", mean_sd(cols["cluster"])]) + r" \\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)




def live_table(entries, caption):
    """QUARANTINED: the training split, with the iteration each number was read
    at. Its own columns and its own caption, so no row can be read as held-out.
    `Wij gain` leads, because that is the quantity the grid is trying to move."""
    _P = r">{\raggedleft\arraybackslash}p{1.5cm}"
    out = [r"\begin{table}[H]", r"\scriptsize", r"\raggedright",
           r"\setlength{\tabcolsep}{1.5pt}", rf"\caption{{{caption}}}",
           r"\setlength{\tabcolsep}{3pt}",
           rf"\begin{{tabular}}{{p{{4.0cm}}{_P * 8}}}", r"\toprule",
           " & ".join(["arm", "iteration", "$k$", r"$W_{ij}$ gain", "gain term",
                       r"$W_{ij}$ $R^2$", r"$E_{ij}$ $R^2$",
                       r"$\mathrm{msg}_i$ $R^2$", r"$\tau$ $R^2$"]) + r" \\",
           r"\midrule"]
    n = 0
    for a in entries:
        it, w = live(a["run"], "Wij", "Wij_R2")
        if it is None:
            continue
        n += 1
        _, g = live(a["run"], "Wij", "Wij_gain")
        _, e = live(a["run"], "Eij", "Eij_R2")
        _, ms = live(a["run"], "msg_i", "msg_i_R2")
        _, t = live(a["run"], "tau", "tau_R2")
        gt = loss_term(a["run"], "f_theta_msg_gain")
        kk, _ = gauge_k(a["run"])
        out.append(" & ".join([a["label"], f"{it:,}", num(kk), num(g),
                               "--" if gt is None else f"{gt:.1e}",
                               num(w), num(e), num(ms), num(t)]) + r" \\")
    if not n:
        return ""
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
    for _ltag, _lam in (("lasso0", "0"), ("lasso0p1", "0.1")):
        L.append(table(pick(f"grid_{_ltag}"),
                       "Conductance model on conductance data with a silent-input "
                       "anchor and a message-gain term, group lasso "
                       rf"$\lambda = {_lam}$, $\sigma = 0.05$."))
    L.append(table(pick("cur0p1"),
                   "Conductance model on current data with a group lasso "
                   r"$\lambda = 0.1$, $\sigma = 0.05$."))

    for _ltag, _lam in (("lasso0", "0"), ("lasso0p1", "0.1")):
        _t = live_table(
            pick(f"grid_{_ltag}"),
            "Training split at the iteration shown, group lasso "
            rf"$\lambda = {_lam}$. `gain term\' is the value "
            r"\texttt{coeff\_f\_theta\_msg\_gain} contributes to the loss "
            r"(the per-neuron figure the log stores, $\times\,13{,}741$), against a "
            r"trajectory loss of order $0.05$. $k = \tau_i\,\partial f_\theta/"
            r"\partial\mathrm{msg}_i$, median over neurons, by autograd on real "
            r"frames with the model\'s own $\tau$: the gauge is fixed at $k = 1$.")
        if _t:
            L.append(_t)
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
