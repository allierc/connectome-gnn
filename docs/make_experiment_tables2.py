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

LOG = f"{os.environ['GNN_OUTPUT_ROOT']}/log/fly"
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
def loss_term(run, name):
    """The last value of one regularizer component, from loss_components.pt.

    THE COLUMN THAT SAYS WHETHER AN ARM IS AN ARM. A coefficient in the yaml is
    not a penalty in the loss: the term it multiplies can be so small that the
    arm is indistinguishable from the control, and reading only the outcome
    columns would record that as "the term did not help" rather than "the term
    was never applied". Note the log stores the PER-NEURON figure, so the value
    the loss actually saw is this times n_neurons.
    """
    p = f"{LOG}/{run}/loss_components.pt"
    if not os.path.exists(p):
        return None
    try:
        v = torch.load(p, map_location="cpu", weights_only=False).get(name)
        return float(v[-1]) if v is not None and len(v) else None
    except Exception:
        return None


def gauge_k(run):
    """Measured k_i = tau_i * df/dmsg_i, median over neurons, from gauge_k.json.

    THE QUANTITY THE GAUGE GRID EXISTS TO MOVE, and the only one that says
    outright whether the gauge is fixed: k = 1 is fixed, and W is then in the
    generator's units. Measured by autograd on real frames with the MODEL's own
    tau (blind -- no ground truth in it) by tools/measure_gauge_k.py, cached
    because it needs a checkpoint and a data pass and this script must stay
    instant. Absent cache -> the column prints `--`.
    """
    import json
    p = os.path.join(HERE, "gauge_k.json")
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p)).get(run)
    return (d["k_median"], d["iteration"]) if d else (None, None)


_SIL = [("s1", 1), ("s5", 5), ("s25", 25)]
_GAIN = [("g1e5", "10^{-5}"), ("g1e4", "10^{-4}"), ("g1e3", "10^{-3}")]
_L1 = [("l10", "0"), ("l17p5e5", r"7.5\times10^{-5}"), ("l17p5e4", r"7.5\times10^{-4}")]
_L2 = [("l20", "0"), ("l27p5e7", r"7.5\times10^{-7}"), ("l27p5e6", r"7.5\times10^{-6}")]
# M, and z = (delta_t/tau)(1+G) at the stiffest neuron. Forward Euler contracts
# only while z < 2, so M=2 is the arm that should still fail.
_SUB = [(2, 2.21), (3, 1.47), (5, 0.88), (10, 0.44)]

_B = "flyvis_flowcond_noise_005_gnn_nosq"
ARMS = [dict(block="base", label="baseline", run=f"{_B}_cv00")]
for _stag, _sil in _SIL:
    for _gtag, _g in _GAIN:
        ARMS.append(dict(block="gauge", label=rf"silent {_sil}, gain ${_g}$",
                         run=f"{_B}_{_stag}{_gtag}_cv00"))
for _l1t, _l1 in _L1:
    for _l2t, _l2 in _L2:
        run = f"{_B}_{_l1t}_{_l2t}_cv00" if not (_l1t == "l17p5e5" and _l2t == "l27p5e7") \
            else f"{_B}_cv00"
        ARMS.append(dict(block="wl", label=rf"$\lambda_1 = {_l1}$, $\lambda_2 = {_l2}$",
                         run=run))
for _m, _z in _SUB:
    ARMS.append(dict(block="sub", label=rf"$M = {_m}$ \tiny($z = {_z}$)",
                     run=f"{_B}_sub{_m}_cv00"))
# THE SAME M=5 ARM AT THREE SEEDS, because the substep arms showed the widest
# spread of the campaign: at 20k iterations the three sat at Wij_R2 +0.150,
# -0.861 and -0.230. A single substep number cannot be read without them.
for _tag, _lab in (("sub5", "seed 42"), ("sub5_s2", "seed 43"), ("sub5_s3", "seed 44")):
    ARMS.append(dict(block="subseed", label=rf"$M = 5$, \tiny {_lab}",
                     run=f"{_B}_{_tag}_cv00"))


def pick(block):
    return [a for a in ARMS if a["block"] == block]


# --------------------------------------------------------------------------- #
#  tables                                                                      #
# --------------------------------------------------------------------------- #
# ONE COLUMN SET FOR EVERY TABLE IN THIS DOCUMENT, held-out and training split
# alike. The first document's twelve, in its order, plus the three this campaign
# is about. A table missing a column its neighbour has cannot be read against it,
# so a quantity a given source does not carry prints `--` rather than dropping
# the column.
#
#   k          tau_i * df/dmsg_i, median over neurons, by autograd on real frames
#              with the MODEL's own tau. k = 1 IS the fixed gauge -- neither
#              Wij_R2 nor Wij gain says whether the gauge is fixed, this does.
#   Wij gain   learned/true: the single factor W is wrong by when k != 1.
#   gain term  what coeff_f_theta_msg_gain contributes to the loss, so an arm
#              that was never actually applied cannot be recorded as "the term
#              did not help".
_SCALARS = [("template_rollout_r", r"fit roll $r$ \tiny same form"),
            ("template_alt_rollout_r", r"fit roll $r$ \tiny other form"),
            ("one_step_r", "one-step $r$"),
            ("rollout_r", "rollout $r$")]
_COLS = [("Wij", r"$W_{ij}$ $R^2$"), ("tau", r"$\tau$ $R^2$"),
         ("V_rest", r"$V_{rest}$ $R^2$"), ("msg_i", r"$\mathrm{msg}_i$ $R^2$"),
         ("Eij", r"$E_{ij}$ $R^2$")]
_FIT3 = ("update_form_r2_median", "conductance_form_r2_median",
         "current_form_r2_median")

_R = r">{\raggedleft\arraybackslash}p{2.35cm}"
_P = r">{\raggedleft\arraybackslash}p{1.05cm}"
NCOLS = (r"p{2.8cm}" + _P * len(_SCALARS) + _R * len(_COLS)
         + r">{\raggedleft\arraybackslash}p{2.2cm}" + _P * 4)
HEAD = (["arm"] + [h for _, h in _SCALARS] + [h for _, h in _COLS]
        + [r"fit $R^2$ \tiny upd/cond/cur", "cluster acc.",
           "$k$", r"$W_{ij}$ gain", "gain term"])
NC = len(HEAD)


def _fmt_term(v):
    return "--" if v is None else f"{v:.0e}"


def _held_out_row(a):
    """Cells from results/metrics.txt; None when the run has not finished."""
    m = metrics(a["run"])
    if m is None:
        return None, {}
    cells = [num(m.get(k)) for k, _ in _SCALARS]
    seen = {k: m.get(k) for k, _ in _SCALARS}
    for k, _ in _COLS:
        cells.append(r2(m.get(f"{k}_R2"), m.get(f"{k}_R2_all"),
                        m.get(f"{k}_pct_outliers")))
        seen[k] = m.get(f"{k}_R2")
    cells.append(" / ".join(
        green(num(v), float(v)) if v not in (None, "", "nan") else "--"
        for v in (m.get(k) for k in _FIT3)))
    cells.append(num(m.get("clustering_accuracy")))
    seen["cluster"] = m.get("clustering_accuracy")
    kk, _ = gauge_k(a["run"])
    cells += [num(kk), num(m.get("Wij_gain")),
              _fmt_term(loss_term(a["run"], "f_theta_msg_gain"))]
    seen["k"], seen["gain"] = kk, m.get("Wij_gain")
    return cells, seen


def _training_row(a):
    """The same columns from tmp_training/<key>.log. The four rollout scalars,
    the three-part fit R2 and the clustering are not written per checkpoint, so
    they print `--`: the column stays, the value is honestly absent."""
    it, _w = live(a["run"], "Wij", "Wij_R2")
    if it is None:
        return None, None, {}
    cells = ["--"] * len(_SCALARS)
    seen = {}
    for k, _ in _COLS:
        v = live(a["run"], k, f"{k}_R2")[1]
        cells.append(r2(v, live(a["run"], k, f"{k}_R2_all")[1],
                        live(a["run"], k, f"{k}_pct_outliers")[1]))
        seen[k] = v
    cells += ["--", "--"]
    kk, _ = gauge_k(a["run"])
    g = live(a["run"], "Wij", "Wij_gain")[1]
    cells += [num(kk), num(g), _fmt_term(loss_term(a["run"], "f_theta_msg_gain"))]
    seen["k"], seen["gain"] = kk, g
    return it, cells, seen


def table(entries, caption, training=False):
    """One table, the shared column set.

    `training` reads tmp_training/<key>.log and tags each row with the iteration
    it was read at; otherwise cells come from results/metrics.txt and a run with
    none gets blanks and a star. CURRENTLY OFF EVERYWHERE: a training-split
    number read at whatever iteration a run happened to reach is not comparable
    to the row beneath it, let alone to a held-out column, and the campaign was
    relaunched on corrected configs so nothing on disk describes a live run yet.
    Flip the flag on the call to bring the numbers back."""
    out = [r"\begin{table}[H]", r"\scriptsize", r"\raggedright",
           r"\setlength{\tabcolsep}{1.5pt}", rf"\caption{{{caption}}}",
           r"\setlength{\tabcolsep}{2pt}",
           rf"\begin{{tabular}}{{{NCOLS}}}", r"\toprule",
           " & ".join(HEAD) + r" \\", r"\midrule"]
    acc, n_rows = {}, 0
    for a in entries:
        if training:
            it, cells, seen = _training_row(a)
            if cells is None:
                continue
            label = a["label"] + rf" \tiny({it:,})"
        else:
            cells, seen = _held_out_row(a)
            label = a["label"] + ("" if cells else r"$^{*}$")
        if cells is None:
            out.append(label + " & " * (NC - 1) + r" \\")
        else:
            n_rows += 1
            out.append(" & ".join([label] + cells) + r" \\")
            for k, v in seen.items():
                acc.setdefault(k, []).append(v)
        out.append(rf"\multicolumn{{{NC}}}{{@{{}}l@{{}}}}"
                   rf"{{\tiny\texttt{{{esc(a['run'])}}}}} \\[1pt]")
    if training and not n_rows:
        return ""
    keys = [k for k, _ in _SCALARS] + [k for k, _ in _COLS]
    means = [mean_sd(acc.get(k, [])) for k in keys]
    tail = ["--", mean_sd(acc.get("cluster", [])),
            mean_sd(acc.get("k", [])), mean_sd(acc.get("gain", [])), "--"]
    out += [r"\midrule",
            " & ".join([r"\textbf{mean} $\pm$ SD"] + means + tail) + r" \\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}"]
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
Rows marked $^{*}$ are left blank: those runs have no held-out numbers yet.
All arms share one fold (cv00), $\sigma = 0.05$, and the unsquared
parameterisation $\mathrm{msg} = W g_\phi$: neither factor is squared, so the sign of a
synapse may sit in either, and $|W|$ is what the readout reports.}\end{center}
\vspace{4pt}""",
         rf"\section*{{{DATE}}}"]

    # THE FIRST DOCUMENT'S CAPTION TEMPLATE, unchanged:
    # "<Model> model on <data> data with <knob>, $\sigma = <noise>$."
    L.append(table(pick("base"),
                   "Conductance model on conductance data with neither factor "
                   r"squared, $\sigma = 0.05$."))
    L.append(table(pick("gauge"),
                   "Conductance model on conductance data with a silent-input "
                   r"anchor and a message-gain term, $\sigma = 0.05$."))
    L.append(table(pick("wl"),
                   r"Conductance model on conductance data with $W$ penalties "
                   r"$\lambda_1$ (L1) and $\lambda_2$ (L2) on $W$, "
                   r"$\sigma = 0.05$."))
    L.append(table(pick("sub"),
                   r"Conductance model on conductance data integrated as $M$ "
                   r"sub-steps of $20/M$ ms, $\sigma = 0.05$. Forward Euler "
                   r"contracts only while $z < 2$."))
    L.append(table(pick("subseed"),
                   r"The $M = 5$ sub-step arm at three seeds, $\sigma = 0.05$."))
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
