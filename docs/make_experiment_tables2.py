#!/usr/bin/env python
"""Build docs/experiment_tables2.pdf -- the conductance-on-conductance campaign.

WHY A SECOND DOCUMENT rather than a section in the first. experiment_tables.tex is
driven by docs/experiment_manifest.tsv, which lists runs that have FINISHED and
carries a hand-curated block structure. This campaign is read while it runs: most
rows have no results/metrics.txt yet, only the per-quantity logs the trainer
appends during training. Mixing the two would make the first document's "final,
from metrics.txt" promise false.

Every row therefore says WHERE its numbers came from -- a finished run's
metrics.txt, or the last line of tmp_training/<key>.log with the iteration it was
written at. A number without that provenance is not comparable to one with it.

Usage:
    python docs/make_experiment_tables2.py          # writes .tex and compiles
    python docs/make_experiment_tables2.py --no-pdf # .tex only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from connectome_gnn.metrics import recovery_log_columns  # noqa: E402

LOG = "/groups/saalfeld/home/allierc/GraphData/log/fly"
DATA = "/groups/saalfeld/home/allierc/GraphData/graphs_data/fly"
HERE = os.path.dirname(os.path.abspath(__file__))
DATE = "2026-09-17"


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def fmt(v, nd=3):
    if v is None or v == "" or str(v).lower() == "nan":
        return "--"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return esc(v)


def colour(v, good=0.5, bad=0.0):
    """Green above `good`, red below `bad`, plain between. Only ever applied to
    R2, where the scale is shared and the reader knows what 1.0 would mean."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "--"
    c = "ForestGreen" if x >= good else ("BrickRed" if x < bad else "black")
    return rf"\textcolor{{{c}}}{{{x:.3f}}}"


def read_metrics(run: str) -> dict | None:
    """results/metrics.txt of a FINISHED run, as a dict."""
    p = f"{LOG}/{run}/results/metrics.txt"
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def read_live(run: str, key: str) -> tuple[int, dict] | None:
    """Last row of tmp_training/<key>.log, for a run still training."""
    p = f"{LOG}/{run}/tmp_training/{key}.log"
    if not os.path.exists(p):
        return None
    rows = [r for r in open(p).read().strip().split("\n") if r]
    if not rows:
        return None
    v = rows[-1].split(",")
    return int(float(v[0])), dict(zip(recovery_log_columns(key), v[1:]))


def row_for(run: str) -> dict:
    """One table row: finished numbers if there are any, else live ones."""
    m = read_metrics(run)
    if m:
        return dict(source="final", iteration="",
                    Wij=m.get("Wij_R2"), Eij=m.get("Eij_R2"),
                    msg=m.get("msg_i_R2"), tau=m.get("tau_R2"),
                    vrest=m.get("V_rest_R2"), roll=m.get("rollout_r"))
    got, out = {}, dict(source="live", iteration="")
    for key, col, name in (("Wij", "Wij_R2", "Wij"), ("Eij", "Eij_R2", "Eij"),
                           ("msg_i", "msg_i_R2", "msg"), ("tau", "tau_R2", "tau"),
                           ("V_rest", "V_rest_R2", "vrest")):
        r = read_live(run, key)
        if r:
            got[name] = r[1].get(col)
            out["iteration"] = f"{r[0]:,}"
    out.update(got)
    out.setdefault("roll", None)
    return out


def table(rows, caption, label_head="run"):
    L = [r"\begin{table}[H]\centering\small",
         r"\begin{tabular}{l r r r r r r}", r"\toprule",
         rf"{label_head} & iteration & $R^2_W$ & $R^2_E$ & $R^2_{{msg}}$ & "
         rf"$R^2_\tau$ & $R^2_{{V_{{rest}}}}$ \\", r"\midrule"]
    for r in rows:
        L.append(" & ".join([
            esc(r["label"]), r.get("iteration", "") or "--",
            colour(r.get("Wij")), colour(r.get("Eij")), colour(r.get("msg")),
            colour(r.get("tau")), colour(r.get("vrest")),
        ]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption*{{{caption}}}", r"\end{table}"]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)

    L = [r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.4cm]{geometry}
\usepackage{booktabs,float,amsmath,array}
\usepackage[dvipsnames]{xcolor}
\usepackage[T1]{fontenc}
\usepackage{caption}
\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}
\begin{document}
\begin{center}{\Large Conductance GNN on task-trained conductance data}\\[3pt]""",
         rf"{{\small {DATE}}}\end{{center}}",
         r"""
This campaign asks whether a conductance GNN can recover the parameters of a
conductance generator whose data was NOT built to match a current model. Every
earlier conductance result came from a teacher--student twin fitted to reproduce a
current-based teacher, which is what made the two synapse models hard to separate.
Here the generator is a flyvis network trained on the optic-flow task, so the
conductance in the data was shaped by a task rather than by another model.

Numbers marked \emph{live} are the last line of
\texttt{tmp\_training/<key>.log} at the iteration shown, written while the run is
still training. They are NOT comparable to a finished run's
\texttt{results/metrics.txt}, and they are not comparable to each other unless the
iteration matches.
"""]

    # ---------------------------------------------------------------- data
    L.append(r"\section*{The data}")
    L.append(r"""
Ten datasets from \texttt{flow/2000/001} checkpoint 28, the argmin-EPE checkpoint
of a conductance flyvis model trained on optic flow: five folds at $\sigma = 0$ and
five at $\sigma = 0.05$, 13{,}741 neurons and 434{,}112 edges.

\medskip
\begin{table}[H]\centering\small
\begin{tabular}{l r l l}
\toprule
generator & $n$ & finite & voltage range \\
\midrule
conductance, exponential Euler & 10 & \textcolor{ForestGreen}{10/10} & $-1.598$ \ldots $3.939$ \\
conductance, forward Euler (control) & 10 & \textcolor{BrickRed}{0/10} & $-\infty$ \ldots $\infty$, 95.84\% NaN \\
current, exponential Euler & 1 & 1/1 & $-21.5$ \ldots $378.0$ \\
current, forward Euler (control) & 1 & 1/1 & $-40.9$ \ldots $392.5$ \\
\bottomrule
\end{tabular}
\caption*{Forward Euler contracts only while $(\Delta t/\tau_i)(1+G_i) < 2$, with
$G_i$ the total synaptic conductance onto neuron $i$ in units of its leak. The
generating network reaches $G_i = 3.2$ at $\tau_i = 19$\,ms, giving 4.4 at
$\Delta t = 20$\,ms. Measured over 300 frames on the real parameters: forward Euler
diverges at frame 8, exponential Euler stays finite. For the CURRENT model the two
integrators agree to $7.2\times10^{-7}$, so the step is decisive for one model and
irrelevant for the other. The current-model range is anomalous and unresolved.}
\end{table}
""")

    # ---------------------------------------------------------- the GNN runs
    L.append(r"\section*{The lasso sweep}")
    L.append(r"""
The group lasso over $g_\phi$'s input columns exists to zero the $v_i$ and $a_i$
columns, which is correct on CURRENT data where the message does not depend on the
postsynaptic voltage. On conductance data $(E_{ij} - v_i)$ carries the sign of the
message, so the same penalty removes the term the model needs. The prediction is
therefore that $R^2_W$ falls monotonically with $\lambda$; if it does not, the GNN
was not using the driving force.
""")
    for noise, tag in (("noise_free", r"$\sigma = 0$"), ("noise_005", r"$\sigma = 0.05$")):
        rows = []
        for lam, lam_tag in ((0, "lasso0"), (0.1, "lasso0p1"), (0.25, "lasso0p25")):
            for cv in (0, 1):
                run = f"flyvis_flowcond_{noise}_gnn_{lam_tag}_cv{cv:02d}"
                if not os.path.isdir(f"{LOG}/{run}"):
                    continue
                r = row_for(run)
                r["label"] = rf"$\lambda = {lam}$, fold {cv:02d}"
                rows.append(r)
        if rows:
            L.append(table(rows, rf"Conductance GNN on conductance data, {tag}."))

    # recurrent arm
    rows = []
    for lam, lam_tag in ((0, "lasso0"), (0.1, "lasso0p1"), (0.25, "lasso0p25")):
        for cv in (0, 1):
            run = f"flyvis_flowcond_noise_free_gnn_{lam_tag}_rc10_cv{cv:02d}"
            if not os.path.isdir(f"{LOG}/{run}"):
                continue
            r = row_for(run)
            r["label"] = rf"$\lambda = {lam}$, fold {cv:02d}"
            rows.append(r)
    if rows:
        L.append(r"\section*{The same sweep under a rollout}")
        L.append(table(rows, r"""As above, $\sigma = 0$, with
\texttt{rollout\_horizon\_schedule} $[1,2,4,7,10]$ -- one entry per epoch, so the
loss is scored over an unrolled trajectory reaching ten steps. $K=1$ is
term-for-term the one-step objective, which makes this a strict extension of the
table above rather than a different experiment."""))

    # silent anchor
    rows = []
    for noise in ("noise_free", "noise_005"):
        for cv in (0, 1):
            run = f"flyvis_flowcond_{noise}_gnn_gsil_lasso0_cv{cv:02d}"
            if not os.path.isdir(f"{LOG}/{run}"):
                continue
            r = row_for(run)
            r["label"] = rf"{esc(noise)}, fold {cv:02d}"
            rows.append(r)
    if rows:
        L.append(r"\section*{The silent anchor}")
        L.append(table(rows, r"""$\lambda = 0$ with
\texttt{coeff\_g\_phi\_silent} $= 5$ over $[-2, 0]$, which pins the message's level
where the presynaptic cell is quiet. One parameter away from the $\lambda = 0$ rows
above."""))

    L.append(r"""
\section*{What is already visible}

$R^2_{msg}$ sits at 0.78--0.92 while $R^2_W$ is between $-0.30$ and $+0.15$: the
GNN reproduces the per-neuron message but not its factorisation into a conductance
and a reversal. That is the degeneracy the product $W_{ij}\,\mathrm{relu}(v_j)\,
(E_{ij}-v_i)$ allows, and $R^2_{msg}$ is precisely the quantity that does not care
how the product is split.

No verdict on the lasso yet. At comparable iterations the differences between
$\lambda$ values are smaller than the differences between FOLDS -- at $\lambda=0$,
$\sigma=0$, fold 00 gives $-0.166$ and fold 01 gives $-0.252$ -- and they do not
move consistently in one direction. Two folds cannot separate them.
\end{document}""")

    tex = os.path.join(HERE, "experiment_tables2.tex")
    open(tex, "w").write("\n".join(L) + "\n")
    print(f"wrote {tex}")
    if a.no_pdf:
        return 0
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "experiment_tables2.tex"], cwd=HERE,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2500:])
        return 1
    print(f"wrote {os.path.join(HERE, 'experiment_tables2.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
