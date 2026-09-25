---
number: 5
name: stride5_frames
title: 'One frame in five observed: horizon 5 against horizon 20, current against
  the general form'
purpose: 'redo the 1/5-frames row of the NeurIPS supplementary table with the sampling
  stated once instead of three times: does scoring the rollout only on the observed
  frames recover the circuit, does a deeper horizon over the same sparse supervision
  help, and does the general form under a group lasso of 100 hold up where the current
  form does'
baseline: experiments/baseline/gnn_current_baseline.yaml
specs_dir: experiments/specs/exp05/fly
task: train
queue: gpu_rtx6000
wall: '96:00'
axes:
  horizon:
  - h06
  - h21
  fold:
  - cv00
  - cv01
  - cv02
  - cv03
  - cv04
arms:
- id: current
  label: current form, no lasso
  spec_pattern: flyvis_noise_005_s5{horizon}_cur_{fold}
  differs_by:
    training.rollout_loss_stride: 5
- id: conductance
  label: conductance, group lasso 100
  spec_pattern: flyvis_noise_005_s5{horizon}_condl100_{fold}
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 100.0
    training.rollout_loss_stride: 5
- id: cond_l25
  label: conductance, group lasso 25
  spec_pattern: flyvis_noise_005_s5{horizon}_condl25_{fold}
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
    training.rollout_loss_stride: 5
job_ids:
  flyvis_noise_005_s5h06_cur_cv00: '154448167'
  flyvis_noise_005_s5h06_cur_cv01: '154448168'
  flyvis_noise_005_s5h06_cur_cv02: '154448169'
  flyvis_noise_005_s5h06_cur_cv03: '154448170'
  flyvis_noise_005_s5h06_cur_cv04: '154448171'
  flyvis_noise_005_s5h21_cur_cv00: '154448172'
  flyvis_noise_005_s5h21_cur_cv01: '154448173'
  flyvis_noise_005_s5h21_cur_cv02: '154448174'
  flyvis_noise_005_s5h21_cur_cv03: '154448175'
  flyvis_noise_005_s5h21_cur_cv04: '154448176'
  flyvis_noise_005_s5h06_condl100_cv00: '154448177'
  flyvis_noise_005_s5h06_condl100_cv01: '154448178'
  flyvis_noise_005_s5h06_condl100_cv02: '154448179'
  flyvis_noise_005_s5h06_condl100_cv03: '154448180'
  flyvis_noise_005_s5h06_condl100_cv04: '154448181'
  flyvis_noise_005_s5h21_condl100_cv00: '154448182'
  flyvis_noise_005_s5h21_condl100_cv01: '154448183'
  flyvis_noise_005_s5h21_condl100_cv02: '154448184'
  flyvis_noise_005_s5h21_condl100_cv03: '154448185'
  flyvis_noise_005_s5h21_condl100_cv04: '154448186'
  flyvis_noise_005_s5h06_condl25_cv00: '154448187'
  flyvis_noise_005_s5h06_condl25_cv01: '154448188'
  flyvis_noise_005_s5h06_condl25_cv02: '154448189'
  flyvis_noise_005_s5h06_condl25_cv03: '154448190'
  flyvis_noise_005_s5h06_condl25_cv04: '154448191'
  flyvis_noise_005_s5h21_condl25_cv00: '154448192'
  flyvis_noise_005_s5h21_condl25_cv01: '154448193'
  flyvis_noise_005_s5h21_condl25_cv02: '154448194'
  flyvis_noise_005_s5h21_condl25_cv03: '154448195'
  flyvis_noise_005_s5h21_condl25_cv04: '154448196'
report:
  arm_order:
  - current
  - cond_l25
  - conductance
  arm_labels:
    cond_l25: conductance
  axis_labels:
    horizon:
      h06: '6'
      h21: '21'
  arm_columns:
    lasso: training.coeff_g_phi_input_group_L1
analyse_job_ids: {}
---
# Experiment 5 — stride5_frames

**One frame in five observed: horizon 5 against horizon 20, current against the general form**

**Purpose.** redo the 1/5-frames row of the NeurIPS supplementary table with the sampling stated once instead of three times: does scoring the rollout only on the observed frames recover the circuit, does a deeper horizon over the same sparse supervision help, and does the general form under a group lasso of 100 hold up where the current form does

## What 1/5 frames means here, and what it used to mean

The old `time_step: 5` expressed partial temporal sampling by **decimating
the dataset**, which also deepened the rollout and moved the target — three
effects from one number, and a result could not be attributed to any of
them. That knob is gone (`a080aa6a`).

`rollout_loss_stride: 5` says it once. The data is untouched and the model
still integrates every intermediate frame; only the **supervision** is
sparse, which is what observing one frame in five actually is.

**This paragraph described the intent, and the implementation does not match
it** — see the audit below. The loss lands on frames `k+4, k+9, k+14, k+19`,
none of which a 1-in-5 recording anchored at `k` observes.

## The grid

Model noise 0.05, no measurement noise. Two horizons and two edge
functions, crossed, five folds each.

| | `h05` | `h20` |
|---|---|---|
| schedule | `[5] x 20` | `[5]x5 [10]x5 [15]x5 [20]x5` |
| steps scored | 5 | 5, 10, 15, 20 |

| | `current` | `conductance` |
|---|---|---|
| `signal_model_name` | `flyvis_current` | `flyvis_conductance` |
| `input_size` | 3 | 6 |
| `coeff_g_phi_input_group_L1` | 0 | 100 |

**The schedule ramps in multiples of the stride.** An epoch whose horizon
is below 5 scores no step at all — a loss of exactly zero, training on
nothing while looking busy — so it starts at 5 and climbs in fives, with
equal epochs per rung so the compute per epoch is comparable between the
two horizons. `graph_trainer` raises if any scheduled horizon is shorter
than the stride.

## Specs

20 = 2 arms x 2 horizons x 5 folds.

| arm | horizon | fold | spec |
|---|---|---|---|
| current | 05 | cv00 | `flyvis_noise_005_s5h05_cur_cv00` |
| current | 05 | cv01 | `flyvis_noise_005_s5h05_cur_cv01` |
| current | 05 | cv02 | `flyvis_noise_005_s5h05_cur_cv02` |
| current | 05 | cv03 | `flyvis_noise_005_s5h05_cur_cv03` |
| current | 05 | cv04 | `flyvis_noise_005_s5h05_cur_cv04` |
| current | 20 | cv00 | `flyvis_noise_005_s5h20_cur_cv00` |
| current | 20 | cv01 | `flyvis_noise_005_s5h20_cur_cv01` |
| current | 20 | cv02 | `flyvis_noise_005_s5h20_cur_cv02` |
| current | 20 | cv03 | `flyvis_noise_005_s5h20_cur_cv03` |
| current | 20 | cv04 | `flyvis_noise_005_s5h20_cur_cv04` |
| conductance | 05 | cv00 | `flyvis_noise_005_s5h05_condl100_cv00` |
| conductance | 05 | cv01 | `flyvis_noise_005_s5h05_condl100_cv01` |
| conductance | 05 | cv02 | `flyvis_noise_005_s5h05_condl100_cv02` |
| conductance | 05 | cv03 | `flyvis_noise_005_s5h05_condl100_cv03` |
| conductance | 05 | cv04 | `flyvis_noise_005_s5h05_condl100_cv04` |
| conductance | 20 | cv00 | `flyvis_noise_005_s5h20_condl100_cv00` |
| conductance | 20 | cv01 | `flyvis_noise_005_s5h20_condl100_cv01` |
| conductance | 20 | cv02 | `flyvis_noise_005_s5h20_condl100_cv02` |
| conductance | 20 | cv03 | `flyvis_noise_005_s5h20_condl100_cv03` |
| conductance | 20 | cv04 | `flyvis_noise_005_s5h20_condl100_cv04` |

## Audit of the stride implementation, 2026-09-24

Six independent reviewers over the stride path, each finding then adversarially
refuted by two more. 95 verdicts; the numbers below are the ones that survived.

### The bug: the stride scores the frames it is supposed to skip

`_rollout_step_weights` masks on `(s + 1) % m == 0`
([recurrent_step.py:149](../src/connectome_gnn/models/recurrent_step.py#L149)).
Step `s` is 0-indexed and the state at step `s` is frame `k+s` — verified to
2e-6 by running `_dense_rollout_loss` on synthetic data with the model set equal
to the true right-hand side. So with stride 5:

```
horizon  5 : scored steps [4]            -> frames k+4
horizon 20 : scored steps [4, 9, 14, 19] -> frames k+4, k+9, k+14, k+19
a 1-in-5 recording anchored at k observes  k, k+5, k+10, k+15, k+20
overlap                                    NONE
```

**Not one scored frame is an observed frame.** The knob exists to score only the
observed steps and scores exactly the unobserved ones. Two consequences, both
measured:

- **The anchor is deleted.** Step 0 is the un-integrated observed frame `k`,
  the one term with zero drift. `(s+1) % m` can never select `s = 0`, so the
  strided objective has no anchored term at all.
- **The target is an oracle the regime cannot supply.** `y_ts[k+4]` is built as
  `(v[k+5] - v[k+4]) / delta_t`, an adjacent-frame difference across a frame a
  1-in-5 recording never sees. Worse, `pred` at step 4 is evaluated on the
  model's *drifted* state while `y_ts[k+4]` is the derivative of the *true*
  trajectory; `delta_t/tau` has median 1.009 on this dataset, so the two
  decorrelate within one Euler step — even the exact generator explains only
  **35%** of the target at `s = 4`.

The documentation says the opposite. Every comment, the commit message and
`tests/test_rollout_loss_stride.py` claim "steps 5, 10, 15, 20", which is true
only on a 1-indexed reading that nothing in the loop uses. **All nine stride
tests call `_rollout_step_weights` in isolation and assert on a list of floats;
not one calls `_dense_rollout_loss`**, so nothing links the mask to the frame it
consumes. That is why this was invisible.

### Why the conductance arm dies and the current arm does not

The arms differ in one knob: the conductance arm carries
`coeff_g_phi_input_group_L1`, the current arm carries none. The lasso applies a
**constant** pull of 200 per input column-group of `g_phi.layers[0].weight`.
Measured gradient on the `v_j` column — the one input the conductance message
needs:

| supervision | fit gradient on `v_j` | lasso pull | ratio |
|---|---|---|---|
| one-step (exp02) | 186–239 | 200 | **1.1 : 1** |
| strided (exp05) | 0.69–6.1 | 200 | **33 : 1** |
| strided, lasso 25 | 0.69–6.1 | 50 | 8–70 : 1 |

Striding flattens the objective's sensitivity to connectome amplitude by 28x:
scaling a recovered connectome by `alpha`, the step-0 objective rises 1030% at
`alpha = 0` while the step-4 objective — the only one scored — rises 37%. The
flattened direction is exactly "shrink the message", which is the direction the
lasso pushes. The lasso then annihilates the `v_j` column, `g_phi` becomes a
constant, the message goes to zero, and `R2_W` lands on -0.012.

This predicts, correctly, that **lasso 25 does not rescue it** — 50 is still
8–70x the strided fit gradient — and all ten lasso-25 runs did collapse. It also
explains exp03's `0.616 +- 0.314`: fold cv03 suffered the identical
`g_phi`-first-layer annihilation, so that is four working folds plus one
failure, not a distribution around 0.6.

### Regression against NeurIPS: no on W, yes on tau

Like-for-like against the old `time_step: 5` path that produced the published
1/5-frames row, on the **current** form:

| | old `time_step: 5` | new `rollout_loss_stride: 5` |
|---|---|---|
| `W R2` | 0.303 | **0.710** |
| one-step `r` | 0.863 | **0.973** |
| rollout `r` | 0.943 | **0.990** |
| `tau R2` | **0.829 +- 0.057** | 0.655 +- 0.176 |

So the new path is a clear improvement on W and a **regression on tau**, with
three times the fold spread. The old path scored
`||(v_endpoint - v_obs[k+time_step]) / (delta_t * time_step)||` — model state
against observation at the same clock — where the new one scores a derivative
against an unobserved frame. There is **no NeurIPS-era conductance run under any
stride**: every run that ever set `time_step > 1` is current-form with no lasso,
so `R2_W = -0.012` has no predecessor to regress from.

### What the experiment cannot answer as designed

Every conductance arm in exp02, exp03 and exp05 carries a non-zero group lasso
and every current arm carries none, so `signal_model_name` and
`coeff_g_phi_input_group_L1` are **perfectly confounded**. exp05 cannot
separate "the stride breaks the conductance form" from "the stride breaks the
group lasso". The audit's gradient measurement says it is the lasso, but that is
an inference from one measurement, not a controlled comparison.

### Two more, smaller

- **`horizon == stride` degenerates to endpoint-only training.** At K=5, m=5
  exactly one step is scored and the objective is bit-identical to
  `rollout_step_weighting: "last"`. Ten of exp05's twenty runs are that.
- **`Niter // K` holds compute constant but not supervision.** Dividing
  iterations by the horizon leaves the scored-term count invariant in the dense
  path; with a stride only `K/m` steps are scored, so exp05 gets **1/5** the
  scored terms of exp03.

### Test plan, before any fix

1. `_dense_rollout_loss` on synthetic data with the model set to the exact
   generator, stride 5, horizon 20: assert the scored residual is ~0 only when
   the state's frame and the target's frame agree. Expected to FAIL today.
2. A test asserting which **frame** each scored step consumes, not which weight
   it carries — the missing test class that hid this.
3. Re-run one exp05 conductance fold with the mask corrected to `s % m == 0`
   and the horizon raised so `s = m` exists (K >= m+1). If `R2_W` recovers, the
   off-by-one was sufficient.
4. **The decisive control**: conductance form, stride 5, `lasso = 0`. This is
   the only run that breaks the confound. If it recovers, the lasso is the
   killer; if it still collapses, the stride is.

### Proposed fixes, in order

1. **Correct the mask to `s % m == 0`** so the scored steps are `0, m, 2m, ...`
   — the observed grid, anchor included. This changes horizon semantics: scoring
   `s = m` needs `K >= m + 1`, so `h05` becomes horizon 6 and `h20` becomes 21,
   and the `graph_trainer` guard must change with it. Caveat from the audit: the
   `mean` reduction then halves each term, so the `v_j` gradient reaches only
   **0.53x** the lasso rather than one-step's 1.1x — **necessary but possibly
   not sufficient**.
2. **Anneal the lasso.** The baseline sets `regul_annealing_rate: 0.0` against a
   code default of 0.5, so the full 200-per-group pull is live at iteration 1
   where the fit gradient is ~3e-5. Annealing costs nothing and is defensible
   whatever else is true.
3. **Add the frame-level tests** in the plan above, so the next off-by-one in
   this path is caught by the suite rather than by a 30-job experiment.
4. Only then re-run exp05.

<!-- STATUS:BEGIN -->

## Status

**0/30 landed**, 0 trained (awaiting `-o test_plot`), 30 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | horizon | n | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | horizon | iter | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | h06 | 369,320 |  | 0.998 ± 0.000 |  |  | 0.943 ± 0.014 | 0.944 ± 0.029 | 0.838 ± 0.049 | 0.798 ± 0.070 | 0.975 ± 0.014 |  |  | 0.879 ± 0.017 |
| current | h21 | 212,056 |  | 0.997 ± 0.001 |  |  | 0.925 ± 0.014 | 0.915 ± 0.024 | 0.797 ± 0.041 | 0.747 ± 0.065 | 0.966 ± 0.013 |  |  | 0.876 ± 0.010 |
| conductance | h06 | 377,324 |  | 0.702 ± 0.241 |  |  | 0.369 ± 0.467 | 0.637 ± 0.263 | 0.667 ± 0.116 | 0.757 ± 0.044 | 0.349 ± 0.524 |  |  | 0.568 ± 0.258 |
| conductance | h21 | 243,556 |  | 0.603 ± 0.197 |  |  | 0.177 ± 0.379 | 0.639 ± 0.150 | 0.621 ± 0.081 | 0.739 ± 0.050 | -0.040 ± 0.544 |  |  | 0.484 ± 0.201 |
| cond_l25 | h06 | 393,320 |  | 0.998 ± 0.000 |  |  | 0.962 ± 0.005 | 0.948 ± 0.017 | 0.846 ± 0.024 | 0.839 ± 0.021 | 0.979 ± 0.005 |  |  | 0.882 ± 0.009 |
| cond_l25 | h21 | 236,356 |  | 0.997 ± 0.000 |  |  | 0.947 ± 0.006 | 0.930 ± 0.013 | 0.828 ± 0.022 | 0.822 ± 0.023 | 0.974 ± 0.007 |  |  |  |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_noise_005_s5h06_cur_cv00` | running | 426,923 | `` |  |
| `flyvis_noise_005_s5h06_cur_cv01` | running | 405,323 | `` |  |
| `flyvis_noise_005_s5h06_cur_cv02` | running | 370,653 | `` |  |
| `flyvis_noise_005_s5h06_cur_cv03` | running | 429,323 | `` |  |
| `flyvis_noise_005_s5h06_cur_cv04` | running | 429,323 | `` |  |
| `flyvis_noise_005_s5h21_cur_cv00` | running | 212,556 | `` |  |
| `flyvis_noise_005_s5h21_cur_cv01` | running | 212,556 | `` |  |
| `flyvis_noise_005_s5h21_cur_cv02` | running | 261,756 | `` |  |
| `flyvis_noise_005_s5h21_cur_cv03` | running | 263,276 | `` |  |
| `flyvis_noise_005_s5h21_cur_cv04` | running | 262,896 | `` |  |
| `flyvis_noise_005_s5h06_condl100_cv00` | running | 379,990 | `` |  |
| `flyvis_noise_005_s5h06_condl100_cv01` | running | 382,656 | `` |  |
| `flyvis_noise_005_s5h06_condl100_cv02` | running | 401,055 | `` |  |
| `flyvis_noise_005_s5h06_condl100_cv03` | running | 378,657 | `` |  |
| `flyvis_noise_005_s5h06_condl100_cv04` | running | 401,055 | `` |  |
| `flyvis_noise_005_s5h21_condl100_cv00` | running | 250,056 | `` |  |
| `flyvis_noise_005_s5h21_condl100_cv01` | running | 250,556 | `` |  |
| `flyvis_noise_005_s5h21_condl100_cv02` | running | 256,284 | `` |  |
| `flyvis_noise_005_s5h21_condl100_cv03` | running | 244,056 | `` |  |
| `flyvis_noise_005_s5h21_condl100_cv04` | running | 248,056 | `` |  |
| `flyvis_noise_005_s5h06_condl25_cv00` | running | 394,653 | `` |  |
| `flyvis_noise_005_s5h06_condl25_cv01` | running | 403,990 | `` |  |
| `flyvis_noise_005_s5h06_condl25_cv02` | running | 406,656 | `` |  |
| `flyvis_noise_005_s5h06_condl25_cv03` | running | 407,989 | `` |  |
| `flyvis_noise_005_s5h06_condl25_cv04` | running | 407,989 | `` |  |
| `flyvis_noise_005_s5h21_condl25_cv00` | running | 260,236 | `` |  |
| `flyvis_noise_005_s5h21_condl25_cv01` | running | 238,556 | `` |  |
| `flyvis_noise_005_s5h21_condl25_cv02` | running | 236,456 | `` |  |
| `flyvis_noise_005_s5h21_condl25_cv03` | running | 237,056 | `` |  |
| `flyvis_noise_005_s5h21_condl25_cv04` | running | 261,756 | `` |  |

<!-- STATUS:END -->

