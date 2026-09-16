"""Build docs/experiment_tables.pdf from a TSV of run metrics.

ONE ROW PER RUN, and every R2 written the same way: the outlier-filtered value
first, the full-sample value in brackets, and the share of points dropped in
parentheses -- `0.995 [0.995] (0.0)`. A single R2 hides which of the two it is,
and on these runs the two differ by a lot (V_rest at fold 02: 0.653 clean
against 0.494 over everything, 8% dropped), so a table printing one number
invites the reader to compare a filtered figure in one row with an unfiltered
one in the next.

Input: the TSV written by the collection step, columns
    section  block  label  config  status  Wij  tau  V_rest  Eij  rollout_r
with each metric field itself "clean|all|pct".
"""

import os
import subprocess
import sys

TSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/exp_rows.tsv"
OUT = os.path.dirname(os.path.abspath(__file__))
DATE_14, DATE_15 = "2026-09-14", "2026-09-15"
# The group-lasso sweep predates the two nights above and is read against them:
# it is the same conductance GNN on CURRENT data, so its reversal has no
# ground truth and its W is the only recovery it can be judged on.
DATE_30 = "2026-08-30"
DATE_16 = "2026-09-16"


_GREEN = 0.9


def green(text, value):
    """The paper's convention: a recovery or prediction number above 0.9 is
    green. Applied to the value the cell leads with, never to the bracketed
    full-sample figure, so the colour always means the same thing."""
    if value is None or value != value or value <= _GREEN:
        return text
    return r"\textcolor{green!45!black}{" + text + "}"


def esc(s):
    return s.replace("_", r"\_")


def num(v):
    """Three decimals, or a bare integer once the value is off the scale R2 lives on."""
    if v in ("", None):
        return "--"
    try:
        f = float(v)
    except ValueError:
        return "--"
    if f != f:
        return "nan"
    if abs(f) >= 1000:
        return f"{f:.0f}"
    if abs(f) >= 100:
        return f"{f:.1f}"
    return f"{f:.2f}"


def r2(field, colour=True):
    """clean [all] (pct outliers) -- the three numbers metrics.txt keeps apart."""
    if not field or field == "||":
        return "--"
    parts = (field.split("|") + ["", "", ""])[:3]
    clean, allv, pct = parts
    if clean in ("", None):
        return "--"
    s = num(clean)
    if allv not in ("", None):
        s += f" [{num(allv)}]"
    if pct not in ("", None) and pct != "":
        try:
            s += f" ({float(pct):.1f})"
        except ValueError:
            pass
    return green(s, _clean(field)) if colour else s


rows = []
for line in open(TSV):
    f = line.rstrip("\n").split("\t")
    if len(f) < 10:
        continue
    rows.append(dict(sec=f[0], block=f[1], label=f[2], config=f[3], status=f[4],
                     Wij=f[5], tau=f[6], V_rest=f[7], Eij=f[8], roll=f[9],
                     msg=f[10] if len(f) > 10 else "",
                     onestep=f[11] if len(f) > 11 else "",
                     cluster=f[12] if len(f) > 12 else "",
                     fit=f[13] if len(f) > 13 else "",
                     ufit=f[14] if len(f) > 14 else "",
                     cfit=f[15] if len(f) > 15 else "",
                     curfit=f[16] if len(f) > 16 else "",
                     gain=f[17] if len(f) > 17 else "",
                     gainsd=f[18] if len(f) > 18 else "",
                     evi=f[19] if len(f) > 19 else "",
                     roll_own=f[20] if len(f) > 20 else "",
                     roll_alt=f[21] if len(f) > 21 else ""))


def _clean(field):
    """The outlier-filtered value out of a `clean|all|pct` field, or None."""
    if not field or field == "||":
        return None
    v = field.split("|")[0]
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def mean_sd(values):
    """`mean +- SD` over the finite entries, or `--`.

    Over the ARMS of one table, which is what the row is for: how much the
    quantity moves when the only thing that changes is the arm. It is a summary
    of a handful of numbers, not a statistic -- with n in single digits the SD is
    itself noisy, and one arm at $R^2$ -28938 drags a mean nobody should read.
    """
    xs = [v for v in values if v is not None]
    if not xs:
        return "--"
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return num(m)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return f"{num(m)} $\\pm$ {num(sd)}"


def pick(sec, block):
    return [r for r in rows if r["sec"] == sec and r["block"] == block]


def _med(entries, key):
    """Median over the arms of one table, or None when nothing was measured."""
    xs = []
    for r_ in entries:
        try:
            v = float(r_[key])
        except (TypeError, ValueError, KeyError):
            continue
        if v == v:
            xs.append(v)
    if not xs:
        return None
    xs.sort()
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# The reversal has to lie where the voltage goes. A conductance form fitted to a
# message with no driving force puts E far outside the range the cells ever
# reach -- 201x on the current GNN against 1.4x on the conductance one -- so the
# rule reads the ratio, not the R2, and 10x is the round number between those
# two populations rather than a level anything was tuned to.
_E_OUTSIDE = 10.0


# WHICH SILENCE PRIOR EACH ROW TRAINED UNDER. There are two, and they do the
# same job by different means: coeff_g_phi_zero_below pushes g_phi to zero below
# an input threshold, coeff_g_phi_silent anchors what it emits at silent input.
# The champion uses the anchor ALONE, with zero_below at 0; several probes were
# built on a base that sets zero_below and then added the anchor, so they
# trained under both and their comparison with the champion is not clean. A
# table whose rows disagree about this says so under itself rather than leaving
# it in the yaml.
def _anchor_tag(cfg):
    import os
    import yaml
    path = os.path.join(os.path.dirname(OUT), "config", "fly", cfg + ".yaml")
    if not os.path.exists(path):
        return None
    t = (yaml.safe_load(open(path)) or {}).get("training", {}) or {}
    zb = t.get("coeff_g_phi_zero_below", 0) or 0
    si = t.get("coeff_g_phi_silent", 0) or 0
    return ("both" if (zb and si) else
            "anchor" if si else "zero-below" if zb else "neither")


def anchor_note(entries):
    tags = {}
    for r_ in entries:
        tags.setdefault(_anchor_tag(r_["config"]), []).append(r_["label"])
    tags.pop(None, None)
    if "both" in tags:
        rows_ = ", ".join(esc(l) for l in tags["both"])
        return (r"\textbf{Silence prior:} " + rows_ + " train under \emph{both} "
                r"\texttt{coeff\_g\_phi\_zero\_below} and "
                r"\texttt{coeff\_g\_phi\_silent}; the champion uses the anchor alone, "
                r"so those rows are not a clean comparison with it.")
    if len(tags) > 1:
        return (r"\textbf{Silence prior:} rows differ --- "
                + "; ".join(f"{k}: " + ", ".join(esc(l) for l in v)
                            for k, v in sorted(tags.items())) + ".")
    return None


def form_verdict(entries):
    """One sentence under a table: was the generator's own form the one learned?

    WRITTEN FROM THE ROWS, not by hand. Three measurements decide it and each is
    quoted with its reference, because every one of them is a ratio or a
    difference that means nothing on its own:

      the gain    how much R2 per edge the driving-force column buys over the
                  current form, mean over edges; the two forms are NESTED, so
                  this is >= 0 by algebra and only its size is informative
      the reversal the fitted |E| as a multiple of the 99th percentile of |v_i|
                  in the data; a reversal the voltage never approaches is a
                  column being used to rescale u, not a driving force
      the rollout  each reconstruction loaded back into its family's known-ODE
                  and run on noise-free data; the one test the nested pair
                  cannot dodge, and readable only when the own-family
                  reconstruction is itself faithful
    """
    g, gsd = _med(entries, "gain"), _med(entries, "gainsd")
    evi = _med(entries, "evi")
    ro, ra = _med(entries, "roll_own"), _med(entries, "roll_alt")
    if g is None and evi is None:
        # A known-ODE table is not missing the test, it cannot be given it: the
        # model IS the generator's equation with its parameters learned, so
        # there is no free message to fit two forms to. Said once, under the
        # table, rather than leaving a reader to wonder which runs failed.
        if entries and all("knownode" in r_.get("config", "") for r_ in entries):
            return (r"\textbf{Is the right form learned?} not asked here: this model "
                    r"IS the generator's equation with its constants learned, so there "
                    r"is no free message for the two forms to be fitted to.")
        return None
    bits = []
    if g is not None:
        bits.append(rf"the driving-force column buys ${g:+.3f}$"
                    + (rf" $\pm$ {gsd:.3f}" if gsd is not None else "")
                    + " $R^2$ per edge over the current form")
    if evi is not None:
        if evi > _E_OUTSIDE:
            bits.append(rf"and the reversal it needs sits {evi:.0f}$\times$ outside the "
                        rf"voltage range the data reaches, so the conductance reading is "
                        rf"an artefact of the extra column")
        else:
            bits.append(rf"and the reversal it needs, {evi:.1f}$\times$ the data's own "
                        rf"voltage range, is inside the range where it can be identified")
    if ro is not None and ra is not None:
        if ro >= ra + 0.05:
            bits.append(rf"rolled out, the own-family reconstruction reaches $r={ro:.2f}$ "
                        rf"against {ra:.2f} for the other family")
        elif ra >= ro + 0.05:
            bits.append(rf"rolled out, the OTHER family's reconstruction reaches "
                        rf"$r={ra:.2f}$ against {ro:.2f}, so the trajectory does not "
                        rf"prefer the form the data was made with")
        else:
            bits.append(rf"the two reconstructions roll out alike ($r={ro:.2f}$ and "
                        rf"{ra:.2f}), so the trajectory does not separate the families")
    return ("\textbf{Is the right form learned?} " + ", ".join(bits) + ".") if bits else None


def table(entries, caption, extra_col=None, eij=True, note=None):
    """One block. `extra_col` is (header, fn) for a leading column such as sigma."""
    # FIXED WIDTHS, so the tables line up with each other down the page rather
    # than each sizing itself to its own longest entry. The sigma column, where
    # there is one, is taken out of the arm column's width, so the first metric
    # column starts at the same place in every table.
    # Wide enough that "0.805 [-1.527] (18.6)" does not wrap: a wrapped cell
    # breaks the row alignment the fixed widths are for.
    _R = r">{\raggedleft\arraybackslash}p{2.2cm}"        # an R2 cell
    _P = r">{\raggedleft\arraybackslash}p{1.0cm}"        # a prediction r
    ncols = ((r"p{0.75cm}p{1.45cm}" if extra_col else r"p{2.2cm}")
             + _P * 2 + _R * (5 if eij else 4) + r">{\raggedleft\arraybackslash}p{2.3cm}" + _P
             + r">{\raggedright\arraybackslash}p{6.4cm}")
    head = ["arm"]
    if extra_col:
        head.insert(0, extra_col[0])
    head += ["one-step $r$", "rollout $r$",
             r"$W_{ij}$ $R^2$", r"$\tau$ $R^2$", r"$V_{rest}$ $R^2$",
             r"$\mathrm{msg}_i$ $R^2$"]
    if eij:
        head += [r"$E_{ij}$ $R^2$"]
    head += [r"fit $R^2$ \tiny upd/cond/cur", "cluster acc.", r"\tiny config"]
    # Flush left, not centred: the tables differ in width by several columns and
    # centring made each one start at a different indent down the page.
    # Caption ABOVE the table: read top to bottom, the name of the thing comes
    # before the thing. \caption placed before \begin{tabular} is what puts it
    # there; the float would otherwise number it after the rules.
    out = [r"\begin{table}[H]", r"\scriptsize", r"\raggedright",
           r"\setlength{\tabcolsep}{2pt}",
           rf"\caption{{{caption}}}",
           r"\setlength{\tabcolsep}{3pt}",
           rf"\begin{{tabular}}{{{ncols}}}", r"\toprule",
           " & ".join(head) + r" \\", r"\midrule"]
    # THE BEST ROW IN BOLD, and best means the headline recovery number: the
    # highest outlier-filtered W_ij R2 among the finished arms, with rollout r as
    # the tie-break for a table where no arm recovered anything. Marking the best
    # FIT instead would bold a row that reproduces the trajectory with the wrong
    # circuit, which is the confusion this whole campaign is about.
    def _rank(r_):
        w = _clean(r_["Wij"])
        if w is not None:
            return w
        try:
            return float(r_["roll"]) - 1e6      # far below any real W_ij R2
        except (TypeError, ValueError):
            return float("-inf")
    _finished = [r_ for r_ in entries if r_["status"] != "RUN"]
    _best = max(_finished, key=_rank) if _finished else None

    for r_ in entries:
        # A RUN THAT HAS NOT FINISHED GETS BLANK CELLS, not its train-split
        # numbers. Those come from tmp_training on the data the model is being
        # fitted to, and printing them in the same column as a held-out test
        # number -- even starred -- invites exactly the comparison the star is
        # there to forbid. The row stays so the arm is visible as pending.
        pending = r_["status"] == "RUN"
        star = r"$^{*}$" if pending else ""
        cells = [esc(r_["label"]) + star]
        if extra_col:
            cells.insert(0, extra_col[1](r_))
        if pending:
            cells += [""] * (9 if eij else 8)
        else:
            def _pred(v):
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    return "--"
                return green(num(x), x)
            cells += [_pred(r_["onestep"]), _pred(r_["roll"]),
                      r2(r_["Wij"]), r2(r_["tau"]), r2(r_["V_rest"]), r2(r_["msg"])]
            if eij:
                cells += [r2(r_["Eij"])]
            # THE FIT QUALITY BESIDE THE RECOVERY. msg_form_r2_median is how well
            # the generator's form describes the model's own per-edge message;
            # the R2 columns to its left are whether the constants that form
            # implies are the generator's. A row with a high fit and a negative
            # W or E is the interesting case, and it is the common one.
            # ONE CELL, THREE FITS: the per-neuron update against the model's
            # own dv/dt, then the per-edge message against BOTH families' forms
            # -- conductance b1*u + b2*u*v_i + b3, then current W*u + C, which is
            # that form with the driving-force column deleted. The third number
            # is what stops the second from being read as evidence: a
            # conductance model whose message the current form fits just as well
            # has not shown it learned a driving force, only that this data does
            # not require one.
            def _triple_fit(*vals):
                got = [_pred(v) for v in vals]
                got = [g for g in got if g != "--"]
                return " / ".join(got) if got else "--"
            # A run plotted before the two-form fit landed carries only its own
            # family's median; it goes in the middle slot rather than being
            # dropped, so old and new rows still show the same kind of number
            # where the column says update / conductance / current.
            _c, _cur = r_["cfit"], r_["curfit"]
            if not _c and not _cur:
                _c = r_["fit"]
            cells += [_triple_fit(r_["ufit"], _c, _cur), _pred(r_["cluster"])]
        cells += [r"\tiny\texttt{" + esc(r_["config"]) + "}"]
        if _best is not None and r_ is _best:
            cells = [(r"\textbf{" + c + "}") if c else c for c in cells]
        out.append(" & ".join(cells) + r" \\")
    # The summary row, over the finished arms only -- a mean that quietly
    # included a pending run's train-split number would be the one number in the
    # table nobody could trace back to a row.
    done = [r_ for r_ in entries if r_["status"] != "RUN"]
    if len(done) > 1:
        cells = [r"\textbf{mean} $\pm$ SD"]
        if extra_col:
            cells.insert(0, "")
        def _floats(key):
            out_ = []
            for r_ in done:
                try:
                    out_.append(float(r_[key]))
                except (TypeError, ValueError):
                    pass
            return out_
        cells += [mean_sd(_floats("onestep")), mean_sd(_floats("roll"))]
        for q in ["Wij", "tau", "V_rest", "msg"]:
            cells.append(mean_sd([_clean(r_[q]) for r_ in done]))
        if eij:
            cells.append(mean_sd([_clean(r_["Eij"]) for r_ in done]))
        cells += [mean_sd(_floats("cfit") or _floats("fit")),
                  mean_sd(_floats("cluster")), ""]
        out += [r"\midrule", " & ".join(cells) + r" \\"]
    out += [r"\bottomrule", r"\end{tabular}"]
    _verdict = form_verdict(entries)
    if _verdict:
        out.append(rf"\\[3pt]{{\scriptsize {_verdict}}}")
    _anchor = anchor_note(entries)
    if _anchor:
        out.append(rf"\\[2pt]{{\scriptsize {_anchor}}}")
    if note:
        out.append(rf"\\[2pt]{{\scriptsize {note}}}")
    out += [r"\end{table}"]
    return "\n".join(out)


L = []
L.append(r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.2cm,landscape]{geometry}
\usepackage{booktabs,float,amsmath,xcolor,array}
\usepackage[T1]{fontenc}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\begin{document}
\begin{center}{\Large Connectome-GNN experiment tables}\\[2pt]
{\small Every $R^2$ is written \emph{outlier-filtered} [full sample] (\% dropped),
final, from \texttt{results/metrics.txt} on the held-out test split.
Rows marked $^{*}$ are left blank: those runs are still training and have no
held-out numbers yet.}\end{center}
\vspace{4pt}""")

# ----------------------------------------------------------------- 2026-08-30
_lasso = pick("D30", "lasso current data")
if _lasso:
    L.append(rf"\section*{{{DATE_30}}}")
    L.append(table(_lasso,
                   rf"{DATE_30}: group-lasso sweep, conductance model on current data, "
                   rf"$\sigma=0.05$; $\lambda$ is \texttt{{coeff\_g\_phi\_input\_group\_L1}}.",
                   eij=False))

# ----------------------------------------------------------------- 2026-09-14
L.append(rf"\section*{{{DATE_14}}}")

L.append(table(pick("D14", "CV current sigma 0"),
               rf"{DATE_14}: current model on current data, $\sigma=0$."))

# Fold 1 keeps its ROW and loses its numbers: it is retraining, because its spec
# pointed at fold 0's data. A blank line in the fold sequence says that where a
# footnote about a missing fold would have to be read to be believed.
_cv005 = pick("D14", "CV current sigma 0.05")
if not any(r_["label"].endswith("01") for r_ in _cv005):
    _cv005 = _cv005[:1] + [dict(sec="D14", block="CV current sigma 0.05",
                                label="fold 01", status="RUN",
                                config="flyvis_current_noise_005_current_cv01",
                                Wij="", tau="", V_rest="", Eij_rmse="", roll="",
                                msg="", onestep="", cluster="")] + _cv005[1:]
L.append(table(_cv005,
               rf"{DATE_14}: current model on current data, $\sigma=0.05$."))

# The known-ODE gets its own table. It is a different model class -- the
# generator's structure with only its parameters learned -- so a row of it inside
# the GNN tables invites reading it as one more arm, when it is the control the
# arms are measured against.
_c005 = pick("D14", "conductance sigma 0.05")
_c000 = pick("D14", "conductance sigma 0")
_ko = pick("D14", "known-ODE sigma 0") + pick("D14", "known-ODE sigma 0.05")
L.append(table(_c000,
               rf"{DATE_14}: conductance model on conductance data, $\sigma=0$."))

L.append(table(_c005,
               rf"{DATE_14}: conductance model on conductance data, $\sigma=0.05$."))

def _ko_sigma(r_):
    return "0" if r_["block"].endswith("sigma 0") else "0.05"


if _ko:
    L.append(table(_ko,
                   rf"{DATE_14}: known-ODE on conductance data --- the generator's structure, "
                   rf"parameters learned.",
                   extra_col=(r"$\sigma$", _ko_sigma)))

# ------------------------------------------------------------- between blocks

# ----------------------------------------------------------------- 2026-09-15
L.append(rf"\section*{{{DATE_15}}}")

def _iteration_now(entries):
    """How far the blank rows have got, read from their own training logs.

    The footnote used to carry a hand-typed iteration, which went stale the
    hour after it was written and then said a run was less far along than it
    was. LOG_ROOT matches tools/collect_exp_rows.py.
    """
    best = 0
    for r_ in entries:
        p_ = os.path.join(os.environ.get(
            "GNN_LOG_ROOT", "/groups/saalfeld/home/allierc/GraphData/log/fly"),
            r_["config"], "tmp_training", "rollout.log")
        if not os.path.exists(p_):
            continue
        lines = [l for l in open(p_).read().splitlines()[1:] if l]
        if lines:
            try:
                best = max(best, int(float(lines[-1].split(",")[0])))
            except ValueError:
                pass
    return best


_A = pick("D15", "A current/current")
_pending = [r_ for r_ in _A if not _clean(r_["Wij"])]
_iter = _iteration_now(_pending)
_note = (rf"$^{{*}}$ still training, at iteration {_iter:,} of 1,600,000 when this "
         rf"table was built." if _iter else None)
L.append(table(_A,
               rf"{DATE_15} Block A: current model on current data, $\sigma=0.05$.",
               eij=False, note=_note))

# Noise-free first everywhere, the run without the complication before the one
# with it; the reference twin closes the table.
_B = pick("D15", "B cond/cond sigma free") + pick("D15", "B cond/cond sigma 005") + \
     pick("D15", "TWIN cond sigma 0")


def _sigma(r_):
    if r_["block"].endswith("005"):
        return "0.05"
    if "TWIN" in r_["block"]:
        return r"0\,$\dagger$"
    return "0"


L.append(table(_B, rf"{DATE_15} Block B: conductance model on conductance data, $\sigma=0$ and $0.05$.",
               extra_col=(r"$\sigma$", _sigma),
               note=r"$\dagger$ the reproduction twin, with both weight penalties \emph{on} and no anchor, for reference."))

L.append(table(pick("D15", "C cond model / current data"),
               rf"{DATE_15} Block C: conductance model on current data, $\sigma=0.05$.",
               eij=False))


def _mn(r_):
    return "0.1" if r_["block"].endswith("010") else "0.2"


L.append(table(pick("D15", "D meas noise 010") + pick("D15", "D meas noise 020"),
               rf"{DATE_15} Block D: current model on current data, $\sigma=0.05$, measurement noise $0.1$ and $0.2$.",
               extra_col=(r"$\sigma_{meas}$", _mn), eij=False))

# ----------------------------------------------------------------- 2026-09-16
# THE CROSS-MODEL GRID. Each cell is a known-ODE -- the generator's structure
# with only its constants learned -- fitted to data made by one family or the
# other. The diagonal says how well the constants are identifiable at all; the
# off-diagonal says what a wrong structural assumption costs, and the answer the
# whole grid gives is that it costs nothing in PREDICTION and everything in
# recovery.
def _ko_sig(r_):
    return "0" if r_["block"].endswith("s0") else "0.05"


_grid = [(d, m) for d in ("current", "conductance") for m in ("current", "conductance")]
_any_grid = any(pick("D16", f"KO {d} data {m} ODE s0") for d, m in _grid)
if _any_grid:
    L.append(rf"\section*{{{DATE_16}}}")
    for d, m in _grid:
        rows_ = (pick("D16", f"KO {d} data {m} ODE s0")
                 + pick("D16", f"KO {d} data {m} ODE s005"))
        if not rows_:
            continue
        L.append(table(rows_,
                       rf"{DATE_16}: known-ODE cross-model grid --- {m} known-ODE on "
                       rf"{d}-generated data, five folds at each $\sigma$.",
                       extra_col=(r"$\sigma$", _ko_sig),
                       eij=(d == "conductance" and m == "conductance")))

_lassoC = pick("D16", "lasso conductance data")
if _lassoC:
    L.append(table(_lassoC,
                   rf"{DATE_16}: the group lasso carried over to conductance data, "
                   rf"$\sigma=0.05$; $\lambda$ is \texttt{{coeff\_g\_phi\_input\_group\_L1}}.",
                   note=r"on \emph{current} data the same penalty took $W_{ij}$ $R^2$ "
                        r"from 0.90 at $\lambda=0.25$ to 0.98 at 25 with rollout "
                        r"$r=1.00$ throughout; the best arm on \emph{this} data is 0.42."))

_probes = pick("D16", "probe goal W") + pick("D16", "probe goal roll")
if _probes:
    L.append(table(_probes,
                   rf"{DATE_16}: one-hour probes on conductance data, $\sigma=0.05$, "
                   rf"at about 100{{,}}000 iterations; \texttt{{p2w}} arms aim at "
                   rf"$W_{{ij}}$ and \texttt{{p2r}} arms at the rollout.",
                   note=r"reference at the same iteration: \texttt{arm\_base} "
                        r"rollout $r\approx0.60$, $W_{ij}$ $R^2\approx-0.23$; the "
                        r"known-ODE on this data reaches $W_{ij}$ 0.88."))

L.append(r"\end{document}")

tex = os.path.join(OUT, "experiment_tables.tex")
open(tex, "w").write("\n".join(L))
r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                    "-output-directory", OUT, tex],
                   capture_output=True, text=True)
if r.returncode:
    print(r.stdout[-3000:])
    sys.exit("pdflatex failed")
print("wrote", os.path.join(OUT, "experiment_tables.pdf"))
