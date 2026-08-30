#!/usr/bin/env python
"""Analysis for the weekend experiment — see papers/weekend_experiment_2026_08_28.md.

Produces one table per task, applying the reading rules from Part 3 of that note:

  * R^2_W is a MEDIAN over a trailing window, never a single final checkpoint. The
    reference runs contain one-checkpoint collapses of 0.2-0.5 in R^2_W (0.8985 ->
    0.3637 -> 0.9104 in three consecutive rows), so a final-value readout is a
    coin flip on whether it landed on one.
  * Differences below a resolution threshold are reported as UNRESOLVED, not
    ranked. The threshold is estimated FROM THE GRID (see --sigma below), not
    assumed.
  * ratio_noise, not the cosine, is the cross-family statistic: the cosine's
    denominator has 3 non-zero groups on flyvis_A and 5 on flyvis_conductance, so
    part of any cross-family gap in it is arithmetic rather than behavioural.

Safe to run while jobs are still training: every run is marked prelim/partial with
the fraction of its schedule completed, and rows with too few checkpoints to form a
window are reported as such rather than silently averaged over nothing.

Usage
-----
    PYTHONPATH=src python tools/analyze_weekend_experiment.py
    PYTHONPATH=src python tools/analyze_weekend_experiment.py --task 5
    PYTHONPATH=src python tools/analyze_weekend_experiment.py --window 320000 --csv out.csv
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from dataclasses import dataclass, field

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.utils import log_path, config_path, set_data_root

PREFIX = "flyvis_noise_005_"

# ------------------------------------------------------------------ #
#  The grid. Mirrors Part 1 of the note; keep the two in sync.        #
# ------------------------------------------------------------------ #

TASKS = {
    0: dict(
        # Pre-existing baseline, not part of the weekend launch: flyvis_A trained on
        # flyvis_A data with NO noise probe and NO group lasso -- the original setting
        # the reviewer called "too close to the simulation". Same 1.6M-iteration budget.
        # Valid despite predating the g_phi layout reorder and the g_phi_norm fix:
        # both are exact no-ops on flyvis_A (its layout [v_j, a_j] is unchanged, and
        # its norm anchor was always column 0 = v_j, i.e. already correct).
        # NB the folds are NOT seed/dataset-matched to the weekend grid: these use
        # seeds 1042-1046 and nominal_cv01 points at the cv00 DATASET, so compare the
        # fold-level distribution, not fold-by-fold.
        title="BASELINE: nominal flyvis_A five-fold (no probe, no lasso)",
        runs=[f"nominal_cv0{c}" for c in range(5)],
    ),
    1: dict(
        title="Does rollout training beat plain t+1?",
        runs=[f"rs_{a}_cv00" for a in
              ("onestep", "onestep_step_matched", "uniform", "pushforward",
               "shoot2", "shoot1", "discount", "last")],
    ),
    2: dict(
        title="Conductance noise probe, 5 folds x group lasso on/off",
        runs=[f"noiseprobe_{l}_cv0{c}" for l in ("lasso2", "nolasso") for c in range(5)],
    ),
    3: dict(
        title="The same noise probe on the correct model family (flyvis_A)",
        runs=[f"noiseprobe_flyvisA_cv0{c}" for c in range(5)],
    ),
    4: dict(
        title="Group-lasso dose response, width 8 (with noise) vs width 6 (without)",
        runs=(["noiseprobe_nolasso_cv00", "noiseprobe_lasso2_cv00"]
              + [f"noiseprobe_lasso{l}_cv00" for l in (5, 10, 20, 50)]
              + ["noiseseed_off_s1041"]
              + [f"lassoW6_{l}_cv00" for l in (2, 5, 10, 20, 50)]),
    ),
    5: dict(
        title="Is the noise-probe benefit repeatable, or a seed lottery?",
        runs=[f"noiseseed_{o}_s{s}" for s in range(1041, 1046) for o in ("on", "off")],
    ),
    6: dict(
        title="Control for the family-dependent g_phi_norm prior",
        runs=["normctl_cond_cv00", "normctl_flyvisA_cv00"],
    ),
}

# noiseprobe_nolasso_cv00 and noiseseed_on_s1041 are config-identical; with
# deterministic:false they are the grid's only exact replicate and therefore its
# only direct measurement of the run-to-run floor.
REPLICATE_PAIR = ("noiseprobe_nolasso_cv00", "noiseseed_on_s1041")


# ------------------------------------------------------------------ #
#  Loading                                                            #
# ------------------------------------------------------------------ #

@dataclass
class Run:
    name: str
    ok: bool = False
    why: str = ""
    # config
    model: str = ""
    dataset: str = ""
    seed: int = 0
    input_size: int = 0
    n_noise: int = 0
    lasso: float = 0.0
    g_phi_norm: float = 0.0
    reduction: str = ""
    schedule: list = field(default_factory=list)
    weighting: str = ""
    gamma: float = 0.0
    bptt_window: int = 0
    shooting_stride: int = 0
    planned_steps: int = 0
    planned_evals: int = 0
    # series
    it: list = field(default_factory=list)
    r2: list = field(default_factory=list)
    d_it: list = field(default_factory=list)
    cos: list = field(default_factory=list)
    r_vi: list = field(default_factory=list)
    r_ai: list = field(default_factory=list)
    r_noise: list = field(default_factory=list)

    @property
    def last_it(self):
        return self.it[-1] if self.it else 0

    @property
    def progress(self):
        return self.last_it / self.planned_steps if self.planned_steps else 0.0


def _floats(tok):
    try:
        v = float(tok)
    except (TypeError, ValueError):
        return None
    return None if v != v else v  # drop NaN


def _read_csv(path, n_cols):
    """Read a headerless-or-headered numeric log into columns; skip junk rows."""
    cols = [[] for _ in range(n_cols)]
    if not os.path.exists(path):
        return cols
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if not parts or not parts[0].strip().isdigit():
                continue
            if len(parts) < n_cols:
                continue
            cols[0].append(int(parts[0]))
            for k in range(1, n_cols):
                cols[k].append(_floats(parts[k]))
    return cols


def load(name) -> Run:
    r = Run(name=name)
    cfg_file = config_path("fly", f"{PREFIX}{name}.yaml")
    if not os.path.exists(cfg_file):
        r.why = "no config"
        return r
    try:
        c = NeuralGraphConfig.from_yaml(cfg_file)
    except Exception as e:                                    # noqa: BLE001
        r.why = f"config error: {type(e).__name__}"
        return r
    t, g, s = c.training, c.graph_model, c.simulation
    r.model, r.dataset, r.seed = g.signal_model_name, c.dataset.split("_")[-1], t.seed
    r.input_size, r.n_noise = g.input_size, getattr(g, "n_g_phi_noise_inputs", 0)
    r.lasso = t.coeff_g_phi_input_group_L1
    r.g_phi_norm = t.coeff_g_phi_norm
    r.reduction = t.fit_reduction
    r.schedule = list(t.rollout_horizon_schedule or [])
    r.weighting, r.gamma = t.rollout_step_weighting, t.rollout_discount
    r.bptt_window, r.shooting_stride = t.rollout_bptt_window, t.rollout_shooting_stride

    niter = int(s.n_frames * t.data_augmentation_loop // t.batch_size * 0.2)
    K = r.schedule or [1] * t.n_epochs
    K = (K + [K[-1]] * t.n_epochs)[: t.n_epochs]
    r.planned_steps = sum(max(1, niter // k) for k in K)
    r.planned_evals = sum(max(1, niter // k) * k for k in K)

    d = log_path("fly", f"{PREFIX}{name}", "tmp_training")
    it, r2 = _read_csv(os.path.join(d, "metrics.log"), 2)
    r.it, r.r2 = it, r2
    dc = _read_csv(os.path.join(d, "g_phi_discard.log"), 5)
    r.d_it, r.cos, r.r_vi, r.r_ai, r.r_noise = dc

    r.ok = bool(r.it)
    if not r.ok:
        r.why = "no metrics rows yet"
    return r


# ------------------------------------------------------------------ #
#  Reading rules                                                      #
# ------------------------------------------------------------------ #

def window_median(iters, vals, window, min_pts=3):
    """Median of `vals` over the trailing `window` iterations.

    Returns (value, n_points, is_full_window). Never a single final checkpoint:
    that is the rule this function exists to enforce. If fewer than `min_pts`
    checkpoints fall inside the window (early in a run), widen to the last
    `min_pts` available and report is_full_window=False so the caller can mark
    the number preliminary.
    """
    # v != v filters NaN. _read_csv already drops it, but ratio_vi/ratio_ai are
    # legitimately nan on the flyvis_A layout, so a caller passing raw values must
    # not silently get a nan median back (statistics.median on a list containing
    # nan returns nan, without raising).
    pairs = [(i, v) for i, v in zip(iters, vals) if v is not None and v == v]
    if not pairs:
        return None, 0, False
    last = pairs[-1][0]
    inside = [v for i, v in pairs if i >= last - window]
    full = len(inside) >= min_pts and last >= window
    if len(inside) < min_pts:
        inside = [v for _, v in pairs[-min_pts:]]
    return statistics.median(inside), len(inside), full


def fmt(v, nd=4):
    return "     -" if v is None else f"{v:.{nd}f}"


def mark(r: Run, full: bool):
    """Progress marker: '' complete, '~' partial window, '!' barely started."""
    if r.progress >= 0.99:
        return "" if full else "~"
    return "~" if full else "!"


# ------------------------------------------------------------------ #
#  Resolution threshold, estimated from the grid itself               #
# ------------------------------------------------------------------ #

def estimate_sigma(runs, window):
    """Two variance estimates, both from inside the grid.

    sigma_run  : |difference| between the one exact replicate pair. This is the
                 floor from GPU nondeterminism alone (deterministic:false), at
                 identical seed and config.
    sigma_seed : sd across Task 5's five noise-OFF seeds on a fixed dataset. This
                 is the spread a seed change alone produces, which is what any
                 n=1 conductance contrast is actually competing against.
    """
    out = {}
    a, b = (runs.get(REPLICATE_PAIR[0]), runs.get(REPLICATE_PAIR[1]))
    if a and b and a.ok and b.ok:
        va, _, fa = window_median(a.it, a.r2, window)
        vb, _, fb = window_median(b.it, b.r2, window)
        if va is not None and vb is not None:
            out["sigma_run"] = (abs(va - vb), fa and fb)
    seeds = [runs.get(f"noiseseed_off_s{s}") for s in range(1041, 1046)]
    vals, full = [], True
    for r in seeds:
        if r and r.ok:
            v, _, f = window_median(r.it, r.r2, window)
            if v is not None:
                vals.append(v)
                full &= f
    if len(vals) >= 2:
        out["sigma_seed"] = (statistics.stdev(vals), full)
    return out


# ------------------------------------------------------------------ #
#  Per-task reports                                                   #
# ------------------------------------------------------------------ #

def hdr(text, ch="="):
    print()
    print(ch * 100)
    print(text)
    print(ch * 100)


def _r2row(r, window):
    v, n, full = window_median(r.it, r.r2, window)
    return v, n, full


def task1(runs, window, thresh):
    hdr(f"TASK 1 — {TASKS[1]['title']}")
    print("R^2_W is the median over the trailing window. Rollout arms are compared at equal")
    print("NETWORK EVALUATIONS, not equal optimizer steps: the Niter/K rule gives them ~2.2x")
    print("fewer updates than rs_onestep for the same compute.\n")
    print(f"{'arm':24}{'knob':22}{'steps':>10}{'evals':>11}{'prog':>7}{'R2_W':>9}{'n':>4}")
    print("-" * 100)
    ref = None
    rows = []
    for n in TASKS[1]["runs"]:
        r = runs[n]
        if not r.ok:
            print(f"{n[3:-5]:24}{'':22}{'':>10}{'':>11}{'':>7}{'  ' + r.why}")
            continue
        knob = ("no rollout" if not r.schedule else
                f"bptt_window {r.bptt_window}" if r.bptt_window else
                f"shoot stride {r.shooting_stride}" if r.shooting_stride else
                f"weight {r.weighting}" + (f" g={r.gamma}" if r.weighting == "discount" else ""))
        v, npts, full = _r2row(r, window)
        rows.append((n, v))
        if n == "rs_onestep_cv00":
            ref = v
        print(f"{n[3:-5]:24}{knob:22}{r.planned_steps:>10,}{r.planned_evals:>11,}"
              f"{r.progress*100:>6.0f}%{fmt(v):>9}{npts:>4}{mark(r, full)}")
    if ref is not None:
        print(f"\n  vs rs_onestep (R2_W {ref:.4f}), resolution threshold {thresh:.4f}:")
        for n, v in rows:
            if v is None or n == "rs_onestep_cv00":
                continue
            d = v - ref
            verdict = "UNRESOLVED" if abs(d) < thresh else ("better" if d > 0 else "worse")
            print(f"    {n[3:-5]:26}{d:+.4f}   {verdict}")
    _shared_prefix_check(runs)


def _shared_prefix_check(runs):
    """All six rollout arms must be identical while K == 1 (epoch 0).

    Every rollout knob is a no-op at K=1 by construction, so any divergence inside
    epoch 0 means a knob is leaking into the wrong regime.
    """
    names = [f"rs_{a}_cv00" for a in ("uniform", "pushforward", "shoot2", "shoot1", "discount", "last")]
    got = [runs[n] for n in names if runs.get(n) and runs[n].ok]
    if len(got) < 2:
        return
    branch = None
    r0 = got[0]
    if r0.schedule:
        niter = r0.planned_steps  # recompute epoch-0 length from the schedule
        K = r0.schedule
        niter0 = None
        # epoch 0 length = Niter // K[0]; recover Niter from planned_steps
        denom = sum(1 / k for k in K)
        niter0 = int(round(r0.planned_steps / denom)) // K[0]
        branch = niter0
    common = sorted(set.intersection(*[{i for i in r.it} for r in got]))
    inside = [i for i in common if branch is None or i <= branch]
    if not inside:
        print("\n  [K=1 identity check] no shared checkpoint inside epoch 0 yet")
        return
    i = inside[-1]
    vals = []
    for r in got:
        v = r.r2[r.it.index(i)]
        if v is not None:
            vals.append(v)
    spread = max(vals) - min(vals) if len(vals) > 1 else 0.0
    ok = spread < 1e-9
    print(f"\n  [K=1 identity check] iter {i:,}: {len(vals)} rollout arms, spread {spread:.2e} "
          f"-> {'IDENTICAL as expected' if ok else '*** DIVERGED — a knob is active at K=1 ***'}")


def _discard_row(r, window):
    c, _, fc = window_median(r.d_it, r.cos, window)
    vi, _, _ = window_median(r.d_it, r.r_vi, window)
    ai, _, _ = window_median(r.d_it, r.r_ai, window)
    no, _, _ = window_median(r.d_it, r.r_noise, window)
    return c, vi, ai, no, fc


def task2(runs, window, thresh):
    hdr(f"TASK 2 — {TASKS[2]['title']}")
    print("Paired within fold: the only difference across a row is the group lasso.\n")
    print(f"{'fold':6}{'seed':6}"
          f"{'R2_W lasso0':>13}{'R2_W lasso2':>13}{'delta':>9}   "
          f"{'noise0':>9}{'noise2':>9}{'vi 0':>8}{'vi 2':>8}")
    print("-" * 100)
    for c in range(5):
        a, b = runs[f"noiseprobe_nolasso_cv0{c}"], runs[f"noiseprobe_lasso2_cv0{c}"]
        if not (a.ok and b.ok):
            print(f"cv0{c:<3}{'':6}  {a.why or b.why}")
            continue
        va, _, fa = _r2row(a, window)
        vb, _, fb = _r2row(b, window)
        _, via, _, noa, _ = _discard_row(a, window)
        _, vib, _, nob, _ = _discard_row(b, window)
        d = (vb - va) if (va is not None and vb is not None) else None
        dtxt = "     -" if d is None else f"{d:+.4f}"
        flag = "" if d is None or abs(d) >= thresh else " (unres)"
        print(f"cv0{c:<3}{a.seed:<6}{fmt(va):>13}{fmt(vb):>13}{dtxt:>9}{flag:9}"
              f"{fmt(noa):>9}{fmt(nob):>9}{fmt(via):>8}{fmt(vib):>8}"
              f"{mark(a, fa)}{mark(b, fb)}")


def task3(runs, window, thresh):
    hdr(f"TASK 3 — {TASKS[3]['title']}")
    print("Strictly paired with Task 2's nolasso arms: only signal_model_name and input_size")
    print("differ. ratio_noise is the cross-family statistic (the cosine is not comparable:")
    print("its denominator has 3 non-zero groups on flyvis_A vs 5 on conductance).")
    print("On flyvis_A there are no v_i/a_i inputs, so ratio_vi/ratio_ai are nan BY DESIGN.\n")
    print(f"{'fold':6}{'seed':6}{'R2_W A':>9}{'R2_W cond':>11}   "
          f"{'noise A':>10}{'noise cond':>12}{'vi cond':>9}{'ai cond':>9}")
    print("-" * 100)
    for c in range(5):
        a, b = runs[f"noiseprobe_flyvisA_cv0{c}"], runs[f"noiseprobe_nolasso_cv0{c}"]
        if not (a.ok and b.ok):
            print(f"cv0{c:<3}  {a.why or b.why}")
            continue
        va, _, fa = _r2row(a, window)
        vb, _, fb = _r2row(b, window)
        _, _, _, noa, _ = _discard_row(a, window)
        _, vib, aib, nob, _ = _discard_row(b, window)
        print(f"cv0{c:<3}{a.seed:<6}{fmt(va):>9}{fmt(vb):>11}   "
              f"{fmt(noa, 5):>10}{fmt(nob, 5):>12}{fmt(vib):>9}{fmt(aib):>9}"
              f"{mark(a, fa)}{mark(b, fb)}")
    print("\n  Reading: noise -> 0 on flyvis_A says discarding WORKS when the family is correct.")
    print("  If conductance then keeps v_i/a_i up while ITS noise also -> 0, v_i is retained for")
    print("  a reason (redundancy), not through failed credit assignment. If neither family")
    print("  drives noise down, the whole framing is about credit assignment instead.")


def task4(runs, window, thresh):
    hdr(f"TASK 4 / 4b — {TASKS[4]['title']}")
    print("Two dose curves, cv00 seed 1041, differing only in whether the noise probe is")
    print("attached. n=1 per dose: read as a SCREEN, not a ranking.\n")
    w8 = {0: "noiseprobe_nolasso_cv00", 2: "noiseprobe_lasso2_cv00", 5: "noiseprobe_lasso5_cv00",
          10: "noiseprobe_lasso10_cv00", 20: "noiseprobe_lasso20_cv00", 50: "noiseprobe_lasso50_cv00"}
    w6 = {0: "noiseseed_off_s1041", 2: "lassoW6_2_cv00", 5: "lassoW6_5_cv00",
          10: "lassoW6_10_cv00", 20: "lassoW6_20_cv00", 50: "lassoW6_50_cv00"}
    print(f"{'lasso':7}{'W8 R2_W':>10}{'W8 noise':>11}{'W8 vi':>9}{'W8 ai':>9}   "
          f"{'W6 R2_W':>10}{'W6 vi':>9}{'W6 ai':>9}")
    print("-" * 100)
    for dose in (0, 2, 5, 10, 20, 50):
        a, b = runs.get(w8[dose]), runs.get(w6[dose])
        va = na = va_vi = va_ai = None
        vb = vb_vi = vb_ai = None
        ma = mb = ""
        if a and a.ok:
            va, _, fa = _r2row(a, window)
            _, va_vi, va_ai, na, _ = _discard_row(a, window)
            ma = mark(a, fa)
        if b and b.ok:
            vb, _, fb = _r2row(b, window)
            _, vb_vi, vb_ai, _, _ = _discard_row(b, window)
            mb = mark(b, fb)
        print(f"{dose:<7}{fmt(va):>10}{fmt(na, 5):>11}{fmt(va_vi):>9}{fmt(va_ai):>9}{ma:>2} "
              f"{fmt(vb):>10}{fmt(vb_vi):>9}{fmt(vb_ai):>9}{mb:>2}")
    print("\n  Expect vi/ai to fall as the dose rises, and R2_W to fall past some point.")
    print("  W8 vs W6 at matched dose tests whether the probe itself perturbs the response.")


def task5(runs, window, thresh, sigma):
    hdr(f"TASK 5 — {TASKS[5]['title']}   <-- the headline statistic")
    print("Dataset cv00 fixed; only the seed and the noise columns vary. This is the only cell")
    print("in the grid with enough replication to support a claim.\n")
    print(f"{'seed':7}{'ON R2_W':>10}{'OFF R2_W':>11}{'ON-OFF':>10}   {'ON noise':>10}{'ON vi':>9}{'OFF vi':>9}")
    print("-" * 100)
    deltas = []
    for s in range(1041, 1046):
        a, b = runs[f"noiseseed_on_s{s}"], runs[f"noiseseed_off_s{s}"]
        if not (a.ok and b.ok):
            print(f"{s:<7}  {a.why or b.why}")
            continue
        va, _, fa = _r2row(a, window)
        vb, _, fb = _r2row(b, window)
        _, avi, _, ano, _ = _discard_row(a, window)
        _, bvi, _, _, _ = _discard_row(b, window)
        d = (va - vb) if (va is not None and vb is not None) else None
        if d is not None:
            deltas.append(d)
        print(f"{s:<7}{fmt(va):>10}{fmt(vb):>11}{('     -' if d is None else f'{d:+.4f}'):>10}   "
              f"{fmt(ano, 5):>10}{fmt(avi):>9}{fmt(bvi):>9}{mark(a, fa)}{mark(b, fb)}")
    print()
    if len(deltas) >= 2:
        m, sd = statistics.mean(deltas), statistics.stdev(deltas)
        se = sd / len(deltas) ** 0.5
        lo, hi = m - 2 * se, m + 2 * se
        excl = (lo > 0) or (hi < 0)
        print(f"  paired (ON - OFF) over n={len(deltas)} seeds: mean {m:+.4f}  sd {sd:.4f}  "
              f"2*se interval [{lo:+.4f}, {hi:+.4f}]")
        print(f"  -> {'REAL: interval excludes zero' if excl else 'NOT ESTABLISHED: interval contains zero'}")
        print("  (This replaces a 4-way-confounded n=1 comparison whose claimed effect, 0.27,")
        print("   was smaller than the 0.375 spread a seed change alone produces.)")
    elif deltas:
        print(f"  only n={len(deltas)} paired seed(s) so far — need >=2 for a spread")
    else:
        print("  no paired seeds complete yet")


def task6(runs, window, thresh):
    hdr(f"TASK 6 — {TASKS[6]['title']}")
    print("coeff_g_phi_norm 0 in both families. The prior anchors g_phi's FIRST input column,")
    print("which is v_i on conductance but v_j on flyvis_A, so at 0.9 the two families do not")
    print("receive the same prior. If these reproduce the Task 2/3 pattern, it did not matter.\n")
    print(f"{'run':24}{'model':20}{'g_phi_norm':>11}{'R2_W':>9}{'noise':>10}{'vi':>9}{'ai':>9}")
    print("-" * 100)
    for n in TASKS[6]["runs"] + ["noiseprobe_nolasso_cv00", "noiseprobe_flyvisA_cv00"]:
        r = runs.get(n)
        if not (r and r.ok):
            continue
        v, _, f = _r2row(r, window)
        _, vi, ai, no, _ = _discard_row(r, window)
        tag = "  <- prior ON (reference)" if n.startswith("noiseprobe") else ""
        print(f"{n[:23]:24}{r.model:20}{r.g_phi_norm:>11}{fmt(v):>9}{fmt(no, 5):>10}"
              f"{fmt(vi):>9}{fmt(ai):>9}{mark(r, f)}{tag}")


# ------------------------------------------------------------------ #

TASK_OF = {}
for _t, _spec in TASKS.items():
    for _n in _spec["runs"]:
        TASK_OF.setdefault(_n, []).append(_t)


def _baseline(runs, window):
    hdr(f"TASK 0 — {TASKS[0]['title']}")
    vals = []
    print(f"{'fold':6}{'dataset':8}{'seed':6}{'prog':>6}{'R2_W':>9}")
    print("-" * 100)
    for n in TASKS[0]["runs"]:
        r = runs.get(n)
        if not (r and r.ok):
            print(f"{n[-4:]:6}  {(r.why if r else 'missing')}")
            continue
        v, _, full = window_median(r.it, r.r2, window)
        if v is not None:
            vals.append(v)
        print(f"{n[-4:]:6}{r.dataset:8}{r.seed:<6}{r.progress*100:>5.0f}%{fmt(v):>9}{mark(r, full)}")
    if len(vals) >= 2:
        print(f"\n  mean {statistics.mean(vals):.4f}  sd {statistics.stdev(vals):.4f}  (n={len(vals)})")


def flat_table(runs, window, thresh):
    """One row per run: full spec + every metric, all tasks in one table.

    Same trailing-median rule as the per-task tables, so numbers here and there
    agree. `!`/`~` mark preliminary rows exactly as elsewhere.
    """
    hdr(f"ALL RUNS — one row per configuration ({len(runs)} configs)")
    print(f"{'task':6}{'config':30}{'model':6}{'ds':6}{'seed':6}{'w':3}{'nz':3}"
          f"{'lasso':7}{'gnorm':8}{'knob':16}{'prog':>6}"
          f"{'R2_W':>9}{'cosine':>9}{'d/dvi':>9}{'d/dai':>9}{'d/dnoise':>10}")
    print("-" * 148)
    order = sorted(runs, key=lambda n: (min(TASK_OF.get(n, [9])), n))
    prev = None
    for n in order:
        r = runs[n]
        t = ",".join(str(x) for x in TASK_OF.get(n, []))
        if prev is not None and t != prev:
            print("-" * 148)
        prev = t
        if not r.ok:
            print(f"{t:6}{n[:29]:30}  {r.why}")
            continue
        v, _, full = window_median(r.it, r.r2, window)
        c, vi, ai, no, _ = _discard_row(r, window)
        knob = ("t+1" if not r.schedule else
                f"bptt_win {r.bptt_window}" if r.bptt_window else
                f"shoot {r.shooting_stride}" if r.shooting_stride else
                (f"{r.weighting} g={r.gamma}" if r.weighting == "discount" else r.weighting))
        model = "cond" if r.model == "flyvis_conductance" else "flyA"
        print(f"{t:6}{n[:29]:30}{model:6}{r.dataset:6}{r.seed:<6}{r.input_size:<3}{r.n_noise:<3}"
              f"{r.lasso:<7}{r.g_phi_norm:<8}{knob:16}{r.progress*100:>5.0f}%"
              f"{fmt(v):>9}{fmt(c):>9}{fmt(vi):>9}{fmt(ai):>9}{fmt(no, 5):>10}{mark(r, full)}")
    print("-" * 148)
    print("model: flyA = flyvis_A (g_phi sees v_j,a_j) | cond = flyvis_conductance (also v_i,a_i)")
    print("w = g_phi first-layer width, nz = pure-noise columns, gnorm = coeff_g_phi_norm")
    print("d/dvi, d/dai, d/dnoise are gradient magnitudes RELATIVE to |d g_phi/d v_j|;")
    print("nan for vi/ai on flyA because those inputs do not exist there.")
    print(f"prog = fraction of the planned schedule completed.  ~/! = preliminary window.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=320_000,
                    help="trailing iterations to median over (default 320000 = one epoch)")
    ap.add_argument("--task", type=int, action="append",
                    help="only this task (repeatable); default all")
    ap.add_argument("--threshold", type=float, default=None,
                    help="resolution threshold; default = max(sigma_seed, sigma_run, 0.010)")
    ap.add_argument("--csv", help="also dump one row per run")
    ap.add_argument("--flat", action="store_true",
                    help="one table with every run and every metric, instead of per-task tables")
    ap.add_argument("--output_root", default=None,
                    help="data root holding log/; same resolution as GNN_Main "
                         "(--output_root, else $GNN_OUTPUT_ROOT)")
    a = ap.parse_args(argv)

    output_root = a.output_root or os.environ.get("GNN_OUTPUT_ROOT")
    if output_root:
        if not os.path.isdir(output_root):
            ap.error(f"--output_root does not exist: {output_root}")
        set_data_root(output_root)

    names = sorted({n for t in TASKS.values() for n in t["runs"]})
    runs = {n: load(n) for n in names}

    live = sum(1 for r in runs.values() if r.ok)
    done = sum(1 for r in runs.values() if r.progress >= 0.99)
    hdr("WEEKEND EXPERIMENT — see papers/weekend_experiment_2026_08_28.md", "#")
    print(f"{len(runs)} configurations, {live} with metrics, {done} complete.")
    print(f"Trailing-median window: {a.window:,} iterations.")
    print("Markers:  (blank) full window on a finished run   ~ partial   ! barely started")

    sigma = estimate_sigma(runs, a.window)
    print("\nResolution threshold, estimated from inside the grid:")
    for k, label in (("sigma_run", "run-to-run floor (exact replicate pair)"),
                     ("sigma_seed", "seed spread (5 noise-OFF seeds, fixed data)")):
        if k in sigma:
            v, full = sigma[k]
            print(f"  {label:48} {v:.4f}{'' if full else '  (preliminary)'}")
        else:
            print(f"  {label:48} not yet estimable")
    thresh = a.threshold
    if thresh is None:
        cand = [v for v, _ in sigma.values()] + [0.010]
        thresh = max(cand)
    print(f"  -> differences below {thresh:.4f} reported as UNRESOLVED")

    if 0 in (a.task or [0]):
        _baseline(runs, a.window)
    if a.flat:
        flat_table(runs, a.window, thresh)
        want = []
    else:
        want = a.task or sorted(TASKS)
    fns = {1: task1, 2: task2, 3: task3, 4: task4, 6: task6}
    for t in want:
        if t == 5:
            task5(runs, a.window, thresh, sigma)
        elif t in fns:
            fns[t](runs, a.window, thresh)

    if a.csv:
        import csv as _csv
        with open(a.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["run", "model", "dataset", "seed", "input_size", "n_noise", "lasso",
                        "g_phi_norm", "reduction", "schedule", "weighting", "gamma",
                        "bptt_window", "shooting_stride", "planned_steps", "planned_evals",
                        "last_iter", "progress", "r2_w", "cosine", "ratio_vi", "ratio_ai",
                        "ratio_noise"])
            for n in names:
                r = runs[n]
                v, _, _ = window_median(r.it, r.r2, a.window) if r.ok else (None, 0, False)
                c, vi, ai, no, _ = _discard_row(r, a.window) if r.ok else (None,) * 5
                w.writerow([n, r.model, r.dataset, r.seed, r.input_size, r.n_noise, r.lasso,
                            r.g_phi_norm, r.reduction, r.schedule, r.weighting, r.gamma,
                            r.bptt_window, r.shooting_stride, r.planned_steps, r.planned_evals,
                            r.last_it, f"{r.progress:.3f}", v, c, vi, ai, no])
        print(f"\nwrote {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
