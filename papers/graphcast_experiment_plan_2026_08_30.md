# Experiment plan — recurrent training, GraphCast options, noise-free and noise-005

40 jobs, `gpu_l4`, `-W 2880`, launched 2026-08-30. ~30 h each.
Reads on: [`weekend_benchmark_results_2026_08_29.md`](weekend_benchmark_results_2026_08_29.md)
(Task 1) and [`graphcast_for_recurrent_training.md`](graphcast_for_recurrent_training.md).

---

# PART 0 — WHAT TASK 1 ESTABLISHED, AND WHAT IT DID NOT

Task 1 compared eight recurrent-training strategies on `flyvis_A` / noise-005 / cv00, at matched
compute. Resolution floor 0.015 (measured run-to-run).

| arm | `R²_W` | vs `onestep` | verdict |
|---|---|---|---|
| `onestep_step_matched` | 0.9831 | +0.0017 | — |
| `onestep` / `shoot1` | 0.9815 | — | — |
| `shoot2` | 0.9799 | −0.0016 | unresolved |
| `discount` γ=0.5 | 0.9797 | −0.0018 | unresolved |
| `uniform` | 0.9757 | −0.0057 | unresolved |
| `last` | 0.9538 | **−0.0277** | **worse** |
| `pushforward` | 0.8991 | **−0.0823** | **worse** |

**Three findings.**

1. **Rollout did not beat one-step.** The best arm is plain t+1 at 46 % of the updates.
2. **Truncating BPTT is expensive** (`pushforward` −0.082). The long gradient chain is load-bearing —
   which rules out "the chain is too long" as the cause of the horizon decay we set out to explain.
3. **Dense per-step supervision earns its keep** (`last` −0.028).

**GraphCast independently agrees with all three.** It never truncates BPTT, it scores every step,
and it weights lead times uniformly — the two arms that cleared our floor are exactly the two things
it refuses to do. That is worth reporting as confirmation rather than coincidence.

**What Task 1 could not distinguish, and why.** Our rollout arms ramped K across the *whole* run at
a constant `lr: 0.0018`. GraphCast spends **96.5 % of its 311,000 updates at K=1** under a monotone
decay, and runs rollout only as a **3.5 % tail at 3e-7** — some 6,000× below our rollout LR. So Task 1
tested *rollout as the objective*; it never tested *rollout as a tail fine-tune*, which is the only
form GraphCast actually uses. That is the gap this grid closes.

A second unexplained observation points the same way: the reference runs contain **one-checkpoint
collapses of 0.2–0.5 in `R²_W`** (0.8985 → 0.3637 → 0.9104 in three consecutive rows), which forced
the trailing-median reading rule. Constant LR with no global gradient clipping is a natural cause.

---

# PART 1 — THE JOBS

Four arms × five folds × two noise levels. `flyvis_A`, `fit_reduction: mean`, base
`flyvis_noise_005_rs_onestep_cv00`, folds cv00–cv04 with seeds 1041–1045.

| arm | K schedule | LR | global clip | updates |
|---|---|---|---|---|
| **A `control`** | 1 (5 epochs) | constant 0.0018 | — | 1,600,000 |
| **B `anneal`** | 1 (5 epochs) | graphcast | — | 1,600,000 |
| **C `annealclip`** | 1 (5 epochs) | graphcast | 32 | 1,600,000 |
| **D `graphcast`** | 1×5, then 2,3,4,5 | graphcast | 32 | 1,658,000 |

Config names: `flyvis_{noise_005,noise_free}_gc_{control,anneal,annealclip,graphcast}_cv0{0..4}`.

**The `graphcast` LR schedule** is linear warmup (1,000) → **one** half-cosine decay → a constant
floor at `3e-4 ×` peak, with the decay ending at 96.5 % of planned updates. Both numbers are
GraphCast's own.

**Arm D's tail is genuinely short.** `rollout_tail_iters_per_epoch: 14500` caps the K>1 epochs, so
the rollout phase is 58,000 of 1,658,000 updates = **3.5 %**, matching GraphCast exactly. The LR
decay reaches its floor at 96.5 %, i.e. precisely where K starts ramping — so the rollout phase runs
at the tail LR, which is the whole point.

---

# PART 2 — WHY NOISE-FREE

Noise-005 may be too easy to separate the arms: six of eight Task 1 arms landed within ±0.006, i.e.
inside the noise floor. **Noise-free is the harder inverse problem** — with no observation noise to
regularise it, the fit is sharper and small differences in optimisation should show up rather than
being masked.

The grid runs **both** levels so the comparison is itself measurable: if the arms separate on
noise-free but not on noise-005, that tells us the benchmark's discriminating power is
noise-limited, which is worth knowing before designing anything further.

---

# PART 3 — WHAT EACH ARM DECIDES

**A → B: does annealing fix the checkpoint collapses?** *(ranked first — nearly free)*
If B's final checkpoint agrees with its trailing median, the trailing-median workaround retires and
the final checkpoint becomes usable. **Report both numbers for every arm** — the agreement is itself
the result.

**B → C: does global clipping add anything?** Today `grad_clip_W` clips only `model.W` and defaults
off, so nothing bounds `f_theta` / `g_phi` / `a` gradients. If B already removes the collapses, C
should be a no-op; if collapses persist in B and vanish in C, clipping was the cause.

**C → D: does rollout help when it is a tail fine-tune?** The arm Task 1 never ran, and the one
GraphCast uses. Predicted: keeps the fast accurate start *and* gains horizon robustness. If D ≈ C,
then rollout adds nothing at any schedule and Task 1's conclusion is final rather than an artefact
of running it at full LR.

**A vs D is the headline.** It is the honest version of "should we train on trajectories" — current
practice against the best-known recipe.

---

# PART 4 — READING RULES

- Report the **final checkpoint and the trailing median** for every arm (see Part 3).
- Arm D takes 3.5 % more updates than A/B/C. That is smaller than any effect the grid can resolve,
  but state it rather than call the arms matched.
- This grid uses `fit_reduction: mean` with coefficients divided by 70,467. It is internally
  consistent but **not** comparable to the `norm2` `nominal_cv` series.
- n = 5 per cell (folds), n = 1 per fold. Use the across-fold spread as the error bar; the
  noise-005 `flyvis_A` run-to-run floor was 0.015.

---

# PART 5 — NOT DONE, AND WHY

**Arm E, increment-variance target weighting.** GraphCast normalises each target by the inverse
variance of the *time difference*, `Var[x_{t+1} − x_t]^{-1}`, so every target is unit-variance as an
increment. Our `ynorm` is **hardcoded to 1.0** (`training_utils.py:1290`) — there is no target
normalisation at all today. Per-neuron `1/Var_t(v_{t+1} − v_t)` is the principled replacement for
dividing every coefficient by 70,467, and it makes one loss weight transfer across neurons with very
different activity levels. Deferred because `ynorm` feeds the R² path and changing it would break
comparability with every existing run; it deserves its own controlled arm.

**AdamW β2 = 0.95.** Low cost, low confidence. Only worth trying if A–D leave residual instability.

**Deliberately not taken from GraphCast:** the 16 message-passing steps (depth there is a physical
requirement of a 6-hour jump; ours should be synaptic hops per Δt ≈ 1 — use substeps if more reach
is needed), and the learned 512-d edge latent, which would dissolve the `W_ij · g_φ` separability
that makes `W` recoverable at all.

---

# PART 6 — CODE

`73eb4e6`. Three new `training` knobs:

| knob | default | meaning |
|---|---|---|
| `lr_scheduler: 'graphcast'` | — | warmup → one half-cosine → flat tail |
| `lr_scheduler_decay_frac` | 0.965 | where the cosine ends |
| `lr_scheduler_tail_ratio` | 3e-4 | tail LR as a fraction of peak |
| `grad_clip_norm` | 0.0 | global clip over **all** parameters |
| `rollout_tail_iters_per_epoch` | 0 | cap on K>1 epochs |

⚠️ **`linear_warmup_cosine` does not decay to zero** — it chains warmup with
`CosineAnnealingWarmRestarts` (`T_mult=2`), so the LR sawtooths back up and floors at
`0.01 × lr`. The handoff note assumed it was usable as-is; it is not. `'graphcast'` was written for
this.

`tests/test_graphcast_schedule.py` (9 tests) pins the schedule shape, that the decay floor coincides
with the start of the K ramp, that the tail is <5 % of updates, and that the cap is a no-op when
disabled. Verified end-to-end on a smoke run: K=1 epochs at full length, then the K=2 epoch capped
from 3,200 to 120 iterations.
