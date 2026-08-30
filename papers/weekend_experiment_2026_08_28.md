# Weekend experiment — 44 jobs, launched 2026-08-28

All on `gpu_l4`, ~33 h each, one batch. 44 jobs, **41 distinct configurations**
(`noiseprobe_nolasso_cv00` and `noiseseed_on_s1041` are the same config — kept as a replicate).

> **Results: [`weekend_benchmark_results_2026_08_29.md`](weekend_benchmark_results_2026_08_29.md).**
> This document is the plan — what was launched and what each contrast was meant to show.
> It deliberately contains no numbers.

Everything below trains a GNN to recover a connectome `W` from simulated neural dynamics.
Headline number is **`R²_W`** — how well the learned weights match the true ones.

---

# PART 1 — THE JOBS

## Task 1 — Does rollout training beat plain t+1?

Model `flyvis_A` · dataset cv00 · seed 1042 · `fit_reduction: mean` · 5 epochs · K = [1,2,3,4,5]

| job | arm | what is changed | optimizer steps | network evals |
|---|---|---|---|---|
| 153765360 | `onestep` | no rollout (plain t+1) | 1,600,000 | 1,600,000 |
| 153765361 | `onestep_step_matched` | no rollout, `dal 46` | 736,000 | 736,000 |
| 153765362 | `uniform` | rollout, current scheme | 730,666 | 1,599,998 |
| 153765363 | `pushforward` | `rollout_bptt_window: 1` | 730,666 | 1,599,998 |
| 153765364 | `shoot2` | `rollout_shooting_stride: 2` | 730,666 | 1,599,998 |
| 153765365 | `shoot1` | `rollout_shooting_stride: 1` | 730,666 | 1,599,998 |
| 153765548 | `discount` | `rollout_step_weighting: discount`, γ = 0.5 | 730,666 | 1,599,998 |
| 153765367 | `last` | `rollout_step_weighting: last` | 730,666 | 1,599,998 |

## Task 2 — Conductance noise probe, 5 folds × group lasso on/off

Model `flyvis_conductance` · `input_size 8` (`v_i,v_j,a_i,a_j` + 2 noise) · `norm2` · 1.6 M steps

| job | config | dataset | seed | group lasso |
|---|---|---|---|---|
| 153765368 | `noiseprobe_lasso2_cv00` | cv00 | 1041 | 2.0 |
| 153765369 | `noiseprobe_lasso2_cv01` | cv01 | 1042 | 2.0 |
| 153765370 | `noiseprobe_lasso2_cv02` | cv02 | 1043 | 2.0 |
| 153765371 | `noiseprobe_lasso2_cv03` | cv03 | 1044 | 2.0 |
| 153765372 | `noiseprobe_lasso2_cv04` | cv04 | 1045 | 2.0 |
| 153765373 | `noiseprobe_nolasso_cv00` | cv00 | 1041 | 0 |
| 153765374 | `noiseprobe_nolasso_cv01` | cv01 | 1042 | 0 |
| 153765375 | `noiseprobe_nolasso_cv02` | cv02 | 1043 | 0 |
| 153765376 | `noiseprobe_nolasso_cv03` | cv03 | 1044 | 0 |
| 153765377 | `noiseprobe_nolasso_cv04` | cv04 | 1045 | 0 |

## Task 3 — The same noise probe on the correct model family

Model `flyvis_A` · `input_size 5` (`v_j,a_j` + 2 noise) · group lasso 0 · 1.6 M steps.
Each job differs from its Task-2 `nolasso` partner in **only** `signal_model_name` and `input_size`.

| job | config | dataset | seed | pairs with |
|---|---|---|---|---|
| 153765537 | `noiseprobe_flyvisA_cv00` | cv00 | 1041 | 153765373 |
| 153765538 | `noiseprobe_flyvisA_cv01` | cv01 | 1042 | 153765374 |
| 153765539 | `noiseprobe_flyvisA_cv02` | cv02 | 1043 | 153765375 |
| 153765540 | `noiseprobe_flyvisA_cv03` | cv03 | 1044 | 153765376 |
| 153765541 | `noiseprobe_flyvisA_cv04` | cv04 | 1045 | 153765377 |

## Task 4 / 4b — How hard must the group lasso push?

All cv00 · seed 1041 · 1.6 M steps. Task 4 keeps the noise columns (width 8); Task 4b removes
them (width 6). Two dose curves that differ only in whether the calibration probe is attached.

| lasso | Task 4 config (width 8, **with** noise) | job | Task 4b config (width 6, **no** noise) | job |
|---|---|---|---|---|
| 0 | `noiseprobe_nolasso_cv00` *(from Task 2)* | 153765373 | `noiseseed_off_s1041` *(from Task 5)* | 153765457 |
| 2 | `noiseprobe_lasso2_cv00` *(from Task 2)* | 153765368 | `lassoW6_2_cv00` | 153765451 |
| 5 | `noiseprobe_lasso5_cv00` | 153765398 | `lassoW6_5_cv00` | 153765452 |
| 10 | `noiseprobe_lasso10_cv00` | 153765399 | `lassoW6_10_cv00` | 153765453 |
| 20 | `noiseprobe_lasso20_cv00` | 153765400 | `lassoW6_20_cv00` | 153765454 |
| 50 | `noiseprobe_lasso50_cv00` | 153765401 | `lassoW6_50_cv00` | 153765455 |

## Task 5 — Is the noise-probe benefit repeatable, or a seed lottery?

Dataset **cv00 held fixed** · group lasso 0 · every coefficient identical.
Only the noise columns and the seed vary.

| seed | noise ON config (width 8) | job | noise OFF config (width 6) | job |
|---|---|---|---|---|
| 1041 | `noiseseed_on_s1041` | 153765456 | `noiseseed_off_s1041` | 153765457 |
| 1042 | `noiseseed_on_s1042` | 153765458 | `noiseseed_off_s1042` | 153765459 |
| 1043 | `noiseseed_on_s1043` | 153765460 | `noiseseed_off_s1043` | 153765461 |
| 1044 | `noiseseed_on_s1044` | 153765462 | `noiseseed_off_s1044` | 153765463 |
| 1045 | `noiseseed_on_s1045` | 153765464 | `noiseseed_off_s1045` | 153765465 |

`noiseseed_on_s1041` is config-identical to Task 2's `noiseprobe_nolasso_cv00` — the grid's only
exact replicate, and the cheapest estimate of the run-to-run floor.

## Task 6 — Control for a prior that is not family-neutral

cv00 · seed 1041 · `coeff_g_phi_norm: 0` (vs 0.9 everywhere else).

| job | config | model | input_size |
|---|---|---|---|
| 153765549 | `normctl_cond_cv00` | `flyvis_conductance` | 8 |
| 153765550 | `normctl_flyvisA_cv00` | `flyvis_A` | 5 |

---

# PART 2 — WHAT WE EXPECT TO LEARN

## The two big questions

**A. Should we train on trajectories instead of single steps?** (Task 1)

**B. Can the network learn to ignore inputs it doesn't need?** (Tasks 2–6)

---

## Question A — trajectories vs single steps

### The coarse answer

We already know rollout training *starts* better and *ends* worse. It reached a higher `R²_W`
than one-step training ever did (0.983 vs 0.975) and got there ten times faster — then declined
steadily as the trajectory got longer, ending at 0.961. Task 1 asks **why it declines**, because
if we can stop the decline we keep the fast, accurate start.

### The mechanism

Three things could cause it, and each arm switches one off:

- **The gradient chain gets too long.** Unrolling K steps means differentiating through K
  applications of the network. `pushforward` cuts the chain to one step while still letting the
  model see its own drifted predictions. *If this arm stays flat as K grows, the chain is the cause.*
- **The model gets less and less real data to anchor on.** Only the first step of a rollout starts
  from a true observation; the rest start from the model's own output. So the fraction of clean,
  data-anchored supervision falls as 1/K. `shoot2` re-anchors on real data every 2 steps.
  *If this arm stays flat, dilution is the cause.*
- **The far, drifted steps dominate.** Late steps have the largest errors and, under MSE, the
  largest gradients. `discount` down-weights them. *If this arm stays flat, that's the cause.*

### The controls that stop us fooling ourselves

- `onestep_step_matched` — the rollout arms take 2.2× **fewer** gradient updates than `onestep` at
  the same compute. This arm gives plain t+1 the same reduced number of updates. If it matches the
  rollout arms, the whole difference was update count, not the objective.
- `shoot1` — a rollout where every step is re-anchored is just K separate one-step fits. If it
  matches the real rollout arms, unrolling the trajectory adds nothing and only the extra frames did.
- `last` — scores only the final step, the old objective. Separates "supervise every step" from
  "use a long horizon".

---

## Question B — can the network ignore what it doesn't need?

### The setup in one paragraph

The simulated data is generated by a rule that uses only the **sending** neuron's state
(`v_j, a_j`). We train two models against it: `flyvis_A`, which sees only those, and
`flyvis_conductance`, which also sees the **receiving** neuron's state (`v_i, a_i`) — two inputs
the truth does not use. The question is whether training discards them. To make "discarded"
measurable we bolt on **two columns of pure random noise**, redrawn every step, so they carry no
information whatsoever. Whatever the network does with those columns is what full discarding
looks like. Then we can ask: does it treat `v_i, a_i` like the noise, or like real signal?

### Coarse → specific

**1. Does the network discard useless inputs at all?** (Task 3)

Run the noise probe on `flyvis_A`, where nothing is redundant and the noise columns are the *only*
useless inputs. If `∂g_phi/∂noise` falls toward zero, discarding works when the model is correct.
If it doesn't fall even here, then nothing we see on the conductance model is about redundancy —
it's about the optimiser failing to assign credit. **This is the single most informative job in
the grid**, because it decides how every other result gets interpreted.

**2. If it discards noise but keeps `v_i`/`a_i`, why?** (Task 3 vs Task 2)

Same probe, same folds, same everything except the model family. Two readings:
- noise → 0 but `v_i` stays up ⟹ `v_i` is being **kept for a reason** (it is redundant with
  information the update function already has), not through failed credit assignment.
- both stay up ⟹ the problem is optimisation, and a better optimiser — not a better prior — is
  the fix.

**3. Can we force the discard with a penalty, and what does it cost?** (Task 4 / 4b)

Six lasso strengths, 0 → 50. Expect the useless-input weights to fall as the penalty rises and
`R²_W` to fall too past some point — a trade-off curve. An older sweep put the sweet spot near 5,
but it never measured *what got discarded*, only what it cost. Task 4b repeats it without the
noise columns to check the probe isn't itself changing the answer.

**4. Is the reported "noise helps" effect real?** (Task 5 — the one with real statistics)

The claim that adding noise columns rescued the difficult seed comes from comparing two runs that
differ in **four** ways at once, and the spread from changing the seed alone (0.375 in `R²_W`) is
**larger than the claimed benefit** (0.27). So the existing evidence cannot tell a real effect from
a lucky draw. Task 5 fixes the data, varies only the seed and the noise columns, and gives five
paired measurements. **Report the mean and spread of (ON − OFF) across the five seeds.** If the
interval excludes zero, the effect is real; otherwise it was the seed.

Early signal at step 3,201 on the hard seed is ON 0.371 vs OFF −0.043 — consistent with the claim,
but that is one seed at one early checkpoint, which is exactly the evidence standard Task 5 replaces.

**5. Is the family comparison distorted by an unequal prior?** (Task 6)

One regulariser anchors the *first column* of `g_phi`'s input — which is `v_i` for conductance but
`v_j` for `flyvis_A`. So the two families are not receiving the same prior, and the Task 2 ↔ Task 3
pairing is not perfect. Task 6 switches that prior off in both. If the comparison survives, the
asymmetry didn't matter.

---

# PART 3 — RULES FOR READING THE RESULTS

These are implemented in `tools/analyze_weekend_experiment.py`, which prints one table per
task and is safe to run mid-training (every partial run is marked):

```bash
PYTHONPATH=src GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData \
    python tools/analyze_weekend_experiment.py            # all tasks
    python tools/analyze_weekend_experiment.py --task 5   # just the headline
    python tools/analyze_weekend_experiment.py --csv out.csv
```

It estimates the resolution threshold **from inside the grid** rather than assuming one —
`sigma_run` from the exact replicate pair, `sigma_seed` from Task 5's five noise-OFF seeds —
and labels anything smaller `UNRESOLVED`. It also asserts the K=1 identity: all six rollout
arms must agree exactly at any epoch-0 checkpoint, and it prints the spread so a knob leaking
into the wrong regime shows up immediately.


- **Never quote a single final checkpoint.** The reference runs contain one-checkpoint collapses of
  0.2–0.5 in `R²_W` (0.8985 → **0.3637** → 0.9104 in three consecutive rows). Use the median over
  the last ~320 k steps.
- **Differences below ~0.010 are unresolved, not ranked.** Tasks 1, 4 and 4b are n = 1 per cell and
  the expected effects (0.007–0.022) sit at or below the known run-to-run spread. Treat them as
  screens. Estimate the error bar from Task 5's five seeds and from the replicate pair
  153765373 / 153765456.
- **1.6 M steps is enough.** The reference noise-probe run is flat from 800 k on: 0.9159 / 0.9192 /
  0.9004 / 0.9097 at 800 k / 960 k / 1.6 M / 2.16 M.
- **Use `ratio_noise`, not the cosine, to compare across families.** The cosine's denominator has 3
  non-zero groups on `flyvis_A` and 5 on conductance, so part of any gap is arithmetic.
  `ratio_noise` is defined identically in both.
- **`shoot2` may be indistinguishable from `uniform`.** With stride 2 over K = [1,2,3,4,5] the
  re-anchor fires only 4 times in 730,666 steps. A null result there is not evidence that shooting
  fails — `shoot1` carries that axis.
- **Task 1 is not comparable to the historical 0.9752 baseline.** All eight arms use `mean` with
  coefficients divided by 70,467. Internally consistent; don't mix with the old `norm2` numbers.
- **All six rollout arms are bit-identical for their first 320,000 steps** (verified: 0.950013 at
  step 16,001), because every rollout knob is inert at K = 1. That is deliberate paired design —
  they branch from one shared checkpoint — but it means they only differentiate over the last 410 k.

---

# APPENDIX — prior results this builds on

| run | `R²_W` | @step |
|---|---|---|
| `nominal_grok_cv00` (flyvis_A, t+1) | 0.9752 | 1.59 M |
| `nominal_rollout_mean_grok_cv00` (K = 1…10) | 0.9605 | 468 k |
| `conductance_cv00` (seed 1041) | **0.5582** | 1.52 M |
| `conductance_cv01` (seed 1042, **same data**) | **0.9334** | 1.52 M |
| `conductance_grok_cv00` (seed 1041) | 0.6425 | 5.76 M |
| `conductance_noiseprobe_grok_cv00` (seed 1041) | 0.9097 | 2.16 M |
| `grouplasso_{01,02,05,10}` | 0.549 / 0.594 / 0.711 / 0.616 | 6.32 M |

Rows 3–4 are the reason Task 5 exists: **same dataset, different seed, 0.375 apart.** Seed 1041 is
a difficult *seed*, not difficult *data*.

## Code changes behind this run

`4d34b70` rollout objective knobs · `65789e0` g_phi randperm hoisted out of the compiled
regulariser (2.04× on A100) · `4187d87` discard metrics generalised to the `flyvis_A` layout ·
`c80eed7` test device pin · `8495205` discard-log gate keyed on layout, not model name.

Three defects an adversarial design audit caught before the weekend: Task 3 was writing **no
discard log at all** (fixed and relaunched); the `g_phi_norm` prior anchors a column that means
different things in the two families (deliberately *not* patched — the patch would force `g_phi`
flat in `v_i`, which is the hypothesis under test — Task 6 added instead); and
`rollout_discount: 0.9` shifted step weights by only 4.4 percentage points, a null dose, re-dosed
to 0.5.

`torch.compile` verified sound on `flyvis_A`: compiled-vs-eager `max|ΔR²_W|` = 0.022 against 0.252
for a seed change, step 1 bit-identical. Conductance runs uncompiled.
