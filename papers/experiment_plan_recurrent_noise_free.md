# Experiment plan — recurrent training on noise-free

Two experiments. **Experiment 1** is the main one: recurrent-training options on noise-free, five
folds. **Experiment 2** is GraphCast-specific and depends on Experiment 1's checkpoints.

Prior: [`benchmark_results.md`](benchmark_results.md) (Task 1),
[`graphcast_for_recurrent_training.md`](graphcast_for_recurrent_training.md).

---

# PART 0 — WHY, AND ONE THING WE MUST MEASURE FIRST

## Task 1 established three things on noise-005

Eight arms, `flyvis_A`, cv00, n = 1, resolution floor 0.015:

| arm | `R²_W` | vs `onestep` |
|---|---|---|
| `onestep_step_matched` | 0.9831 | +0.0017 |
| `onestep` / `shoot1` | 0.9815 | — |
| `shoot2` | 0.9799 | −0.0016 |
| `discount` | 0.9797 | −0.0018 |
| `uniform` | 0.9757 | −0.0057 |
| `last` | 0.9538 | **−0.0277** |
| `pushforward` | 0.8991 | **−0.0823** |

Rollout did not beat one-step; truncating BPTT is expensive; dense per-step supervision earns its
keep. **GraphCast independently agrees with all three** — the two arms that cleared the floor are
exactly the two things it refuses to do.

## But six of eight arms were unresolvable

Everything except `pushforward` and `last` landed within ±0.006, inside the noise floor. A short
pilot on noise-free showed why that is worth fixing:

| | control `R²_W` | across-fold sd |
|---|---|---|
| noise-005 | 0.959 | **0.011** |
| noise-free | 0.741 | **0.116** |

Noise-free is a much harder inverse and has roughly **10× the discriminating power**. That is the
reason to re-run the comparison there.

## ⚠ The baseline nobody has measured

`docs/flyvis_results.md` records **noise-free `R²_W` = 0.899 ± 0.030** (10 seeds). That file was
written **2026-04-12**. The finite-difference fix — `f88290e`, **2026-07-30** — changed the training
target from the generator's analytic drift `f(v[t])` to the observed finite difference
`(v[t+1] − v[t])/dt`.

**No noise-free run has been done since.** I searched every branch, remote and archived log
directory: `0.899` is the only noise-free number that exists anywhere, and it is pre-fix.

**RESOLVED 2026-08-31 — the fix is a no-op at σ = 0, and the pilot's 0.74 was a config error.**
The generator integrates by forward Euler at the same `Δt`
(`x.voltage = x.voltage + sim.delta_t * dv_step`), so at σ = 0 the finite difference **is** the
analytic drift exactly; there is no truncation error to introduce. The published before/after table
agrees: 0.889 → 0.896, Δ +0.006.

An earlier draft of this section argued the opposite — that the fix introduces an O(Δt) truncation
error that bites hardest at σ = 0. That was wrong; it assumed a finer internal integrator than the
code uses.

The pilot's 0.741 came from the **config lineage**, not the fix: those arms inherited
`fit_reduction: mean` with every `coeff_*` divided by 70,467 from the Task 1 `rs_onestep` base — a
rescale calibrated at the noise-005 operating point, which does not transfer to σ = 0. Rerunning the
published `norm2` spec on current `main` reproduces the documented row: at 60 % trained,
R²_W 0.884 (target 0.896 ± 0.038), R²_τ 0.915 (target 0.912 ± 0.024), R²_Vrest 0.850
(target 0.779 ± 0.080). **No code regression; no bisect needed.**

**Consequence for this plan: every arm must be built on the published `norm2` spec**, not on the
Task 1 `mean` lineage. Experiment 1's control arm still doubles as the post-fd-fix noise-free
reference — report it against 0.896.

---

# EXPERIMENT 1 (MAIN) — recurrent-training options on noise-free

`flyvis_A`, `flyvis_noise_free_blank50_cv00..04`, seeds 1041–1045, **`norm2` with the published
coefficients** (base `flyvis_noise_free_blank50_unified_fd`), 5 epochs × dal 100,
`frame_target_offset: 5` on every arm so K=1 recurrent samples identically to one-step. **7 arms × 5 folds = 35 jobs.**

| arm | knob | question |
|---|---|---|
| `onestep` | none (t+1) | **control + the post-fd-fix baseline** |
| `onestep_step_matched` | dal 46 | is any rollout gap just fewer optimizer steps? |
| `uniform` | K = 1..5 | the rollout reference |
| `pushforward` | `rollout_bptt_window: 1` | does truncating BPTT still cost 0.08 here? |
| `shoot1` | `rollout_shooting_stride: 1` | does unrolling add anything over more frames? |
| `discount` | γ = 0.5 | do the drifted late steps need down-weighting? |
| `last` | endpoint-only | does dense supervision still earn its keep? |

**`shoot2` is dropped.** With stride 2 over K = [1,2,3,4,5] the re-anchor fires `floor((K−1)/2)` =
0,0,1,1,2 — **four anchor events in 730,666 steps**, and it is bit-identical to `uniform` for 65.7 %
of them. It cannot resolve its own manipulation. `shoot1` carries the shooting axis.

**What each outcome means.** If `pushforward` and `last` stay clearly negative, Task 1's two findings
replicate on the harder problem and the GraphCast agreement is solid. If `uniform` now *beats*
`onestep` where it could not on noise-005, then rollout does help and noise-005 was simply too easy
to show it — which would overturn the headline. If everything again lands inside the floor, the
conclusion is that this benchmark cannot separate recurrent-training objectives at all, which is
itself worth stating.

---

# EXPERIMENT 2 — GraphCast specifics

The first attempt at this was wrong in a way worth recording. It ran 4 arms × 5 folds × 2 noise
levels, of which three arms were `recurrent_training: false` and the fourth did **1,600,000
one-step updates before any rollout** — its 58,000 rollout updates were the last 3.5 %. After 30 h of
compute **no recurrent training had happened in any of the 40 jobs.** Faithful to GraphCast, but it
spends a full run to test one hour of it.

**Fix: split it into two stages and start the tail from a checkpoint**, which is what GraphCast
itself does (its phase 3 continues from phase 2 rather than retraining).

## Stage 2a — is it the annealing or the clipping? (10 jobs)

K = 1 throughout, five folds, noise-free. `onestep` from Experiment 1 is the control, so only two
new arms are needed:

| arm | LR | global clip |
|---|---|---|
| *(control = Exp 1 `onestep`)* | constant | — |
| `anneal` | `graphcast` schedule | — |
| `annealclip` | `graphcast` schedule | 32 |

Today `grad_clip_W` clips only `model.W` and defaults to 0.0, so **nothing bounds `f_theta`,
`g_phi` or `a` gradients at all**. The pilot hinted clipping is the active ingredient — `anneal`
alone was 3/5 folds positive and not significant, `annealclip` was 5/5 and significant — but that
was at 15 % trained and must be re-established.

Secondary readout: **report the final checkpoint alongside the trailing median.** The collapses that
forced the median rule (0.8985 → 0.3637 → 0.9104 in three consecutive checkpoints) are hypothesised
to be a symptom of never annealing. If the two agree, the median rule retires and that agreement is
itself a result.

## Stage 2b — does a rollout tail help? (5 short jobs, after 2a)

Take the best Stage-2a checkpoint per fold and continue with K ramping 2→5 for ~58,000 updates at
the tail LR. **This is the arm Task 1 never tested** — rollout as a fine-tune rather than as the
objective — and at ~1–2 h per fold instead of 30 h, because the one-step phase is not repeated.

Task 1's rollout arms ramped K across the whole run at constant `lr: 0.0018`; GraphCast runs its
rollout phase at ~6,000× lower LR. That is a precise candidate explanation for the Task 1 shape
*"reached 0.983 ten times faster, then declined to 0.961"*, and 2b is the test.

---

# READING RULES

- **Report both** the trailing-window median and the final checkpoint, for every arm.
- The arms are fold-matched, so use a **paired** comparison (per-fold difference vs control, then
  mean ± 2·se). Comparing arm means against a global threshold wastes most of the power — in the
  pilot it turned two significant results into "unresolved".
- The noise-005 `flyvis_A` run-to-run floor was 0.015. Noise-free's is unknown; estimate it from the
  across-fold spread of the control arm.
- This grid uses the published `norm2` coefficients, so it IS comparable to the `nominal_cv`
  series. (The superseded draft used `fit_reduction: mean` with coefficients divided by 70,467,
  which is what produced the void 0.741 pilot.)
- Watch for a single dominant fold. In the pilot, noise-free cv00 was an outlier in every arm
  (control 0.555 vs 0.75–0.85 elsewhere) and contributed most of the between-arm spread. Report
  per-fold values, not just means.

---

# CODE

Already in `main` (`73eb4e6`), unused until Experiment 2:

| knob | default | meaning |
|---|---|---|
| `lr_scheduler: 'graphcast'` | — | warmup → one half-cosine → flat tail |
| `lr_scheduler_decay_frac` | 0.965 | where the cosine ends |
| `lr_scheduler_tail_ratio` | 3e-4 | tail LR as a fraction of peak |
| `grad_clip_norm` | 0.0 | global clip over **all** parameters |
| `rollout_tail_iters_per_epoch` | 0 | cap on K > 1 epochs |

⚠ `linear_warmup_cosine` does **not** decay to zero — it chains warmup with
`CosineAnnealingWarmRestarts` (`T_mult = 2`), so the LR sawtooths back up and floors at `0.01 × lr`.
`'graphcast'` was written because of this.

Still not done: **increment-variance target weighting**. `ynorm` is hardcoded to `1.0`
(`training_utils.py:1290`) — there is no target normalisation at all today. GraphCast's per-variable
`Var[x_{t+1} − x_t]^{-1}` is the principled replacement for dividing every coefficient by 70,467.
Deferred because `ynorm` feeds the R² path, so changing it breaks comparability with every existing
run; it needs its own controlled arm.
