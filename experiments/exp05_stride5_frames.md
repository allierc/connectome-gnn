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
  flyvis_noise_005_s5h21_cur_cv00: '154453765'
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
analyse_job_ids:
  flyvis_noise_005_s5h06_cur_cv00: '154453735'
  flyvis_noise_005_s5h06_cur_cv01: '154453736'
  flyvis_noise_005_s5h06_cur_cv02: '154453737'
  flyvis_noise_005_s5h06_cur_cv03: '154453738'
  flyvis_noise_005_s5h06_cur_cv04: '154453739'
  flyvis_noise_005_s5h21_cur_cv01: '154453740'
  flyvis_noise_005_s5h21_cur_cv02: '154453741'
  flyvis_noise_005_s5h21_cur_cv03: '154453742'
  flyvis_noise_005_s5h21_cur_cv04: '154453743'
  flyvis_noise_005_s5h06_condl100_cv00: '154453744'
  flyvis_noise_005_s5h06_condl100_cv01: '154453745'
  flyvis_noise_005_s5h06_condl100_cv02: '154453746'
  flyvis_noise_005_s5h06_condl100_cv03: '154453747'
  flyvis_noise_005_s5h06_condl100_cv04: '154453749'
  flyvis_noise_005_s5h21_condl100_cv00: '154453750'
  flyvis_noise_005_s5h21_condl100_cv01: '154453751'
  flyvis_noise_005_s5h21_condl100_cv02: '154453752'
  flyvis_noise_005_s5h21_condl100_cv03: '154453753'
  flyvis_noise_005_s5h21_condl100_cv04: '154453754'
  flyvis_noise_005_s5h06_condl25_cv00: '154453755'
  flyvis_noise_005_s5h06_condl25_cv01: '154453756'
  flyvis_noise_005_s5h06_condl25_cv02: '154453757'
  flyvis_noise_005_s5h06_condl25_cv03: '154453758'
  flyvis_noise_005_s5h06_condl25_cv04: '154453759'
  flyvis_noise_005_s5h21_condl25_cv00: '154453760'
  flyvis_noise_005_s5h21_condl25_cv01: '154453761'
  flyvis_noise_005_s5h21_condl25_cv02: '154453762'
  flyvis_noise_005_s5h21_condl25_cv03: '154453763'
  flyvis_noise_005_s5h21_condl25_cv04: '154453764'
  flyvis_noise_005_s5h21_cur_cv00: '154459603'
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

## Results after the fix, 2026-09-26 (held-out)

The 30 runs relaunched at horizons 6 and 21 with the corrected mask. 5 folds
each, except current h21 at 4 (its cv00 crashed in the readout at a training
checkpoint -- a `None` pair unpacked in metrics.py, now fixed -- and is
retraining).

| arm | h06: R2_W / R2_tau / R2_Vrest | h21: R2_W / R2_tau / R2_Vrest |
|---|---|---|
| current | 0.943 / 0.941 / 0.861 | 0.926 / 0.916 / 0.830 |
| conductance l25 | **0.961** / **0.953** / 0.854 | **0.940** / **0.933** / 0.835 |
| conductance l100 | 0.368 +- 0.466 / 0.588 / 0.657 | 0.177 +- 0.379 / 0.577 / 0.615 |
| *withdrawn buggy current* | *0.753 / 0.655 / 0.458* | *0.722 / 0.689 / 0.434* |
| *unstrided one-step, exp01* | *0.956 / 0.979 / --* | |

**One frame in five now costs about 0.01 of `R2_W`**, and the tau regression is
gone (0.941, against the NeurIPS `time_step: 5` row's 0.829). **Conductance at
lasso 25 is the best arm** at both horizons, with fold spread 0.004-0.005; lasso
100 still collapses on most folds, the same over-penalisation exp02 found at
noise_free. Horizon 21 is slightly worse than 6 on every column.

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

**29/30 landed**, 1 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | horizon | n | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | h06 | 5 | 0.998 ± 0.000 | 0.998 ± 0.000 | 0.994 ± 0.002 (0.0) | 0.577 ± 0.065 (4.7) | 0.943 ± 0.018 (0.0) | 0.941 ± 0.028 (0.4) | 0.861 ± 0.040 (4.6) | 0.829 ± 0.065 | 0.977 ± 0.016 (0.0) | 0.027 ± 0.020 | 0.416 ± 0.011 | 0.878 ± 0.020 |
| current | h21 | 4 | 0.998 ± 0.000 | 0.998 ± 0.001 | 0.994 ± 0.002 (0.0) | 0.603 ± 0.046 (7.5) | 0.926 ± 0.014 (0.0) | 0.916 ± 0.014 (1.6) | 0.830 ± 0.031 (8.6) | 0.779 ± 0.058 | 0.968 ± 0.014 (0.1) | 0.027 ± 0.015 | 0.524 ± 0.009 | 0.864 ± 0.021 |
| conductance | h06 | 5 | 0.758 ± 0.196 | 0.690 ± 0.252 | 0.348 ± 0.185 (34.5) | 0.487 ± 0.018 (3.3) | 0.368 ± 0.466 (0.3) | 0.588 ± 0.304 (13.8) | 0.657 ± 0.131 (46.1) | 0.706 ± 0.080 | 0.385 ± 0.488 (38.7) | 49.810 ± 43.042 | 0.182 ± 0.223 | 0.690 ± 0.144 |
| conductance | h21 | 5 | 0.678 ± 0.161 | 0.598 ± 0.200 | 0.430 ± 0.159 (20.6) | 0.510 ± 0.001 (5.1) | 0.177 ± 0.379 (0.4) | 0.577 ± 0.234 (15.7) | 0.615 ± 0.092 (59.6) | 0.743 ± 0.040 | 0.071 ± 0.460 (62.4) | 158.376 ± 98.876 | 0.156 ± 0.238 | 0.494 ± 0.189 |
| cond_l25 | h06 | 5 | 0.998 ± 0.000 | 0.998 ± 0.000 | 0.118 ± 0.002 (77.9) | 0.492 ± 0.006 (0.0) | 0.961 ± 0.005 (0.0) | 0.953 ± 0.010 (0.5) | 0.854 ± 0.010 (6.6) | 0.854 ± 0.010 | 0.979 ± 0.005 (0.0) | 0.007 ± 0.001 | 0.451 ± 0.011 | 0.870 ± 0.013 |
| cond_l25 | h21 | 5 | 0.997 ± 0.000 | 0.997 ± 0.000 | 0.117 ± 0.002 (77.8) | 0.481 ± 0.015 (0.0) | 0.940 ± 0.004 (0.0) | 0.933 ± 0.017 (0.2) | 0.835 ± 0.020 (8.4) | 0.824 ± 0.041 | 0.974 ± 0.006 (0.0) | 0.010 ± 0.004 | 0.577 ± 0.013 | 0.872 ± 0.015 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | horizon | iter | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_noise_005_s5h06_cur_cv00` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h06_cur_cv01` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h06_cur_cv02` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h06_cur_cv03` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h06_cur_cv04` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h21_cur_cv00` | trained | 294,132 | `23551c2bfb47` |  |
| `flyvis_noise_005_s5h21_cur_cv01` | landed | 294,132 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h21_cur_cv02` | landed | 294,132 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h21_cur_cv03` | landed | 294,132 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h21_cur_cv04` | landed | 294,132 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h06_condl100_cv00` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h06_condl100_cv01` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h06_condl100_cv02` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h06_condl100_cv03` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h06_condl100_cv04` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h21_condl100_cv00` | landed | 294,132 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h21_condl100_cv01` | landed | 294,132 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h21_condl100_cv02` | landed | 294,132 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h21_condl100_cv03` | landed | 294,132 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h21_condl100_cv04` | landed | 294,132 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h06_condl25_cv00` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h06_condl25_cv01` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h06_condl25_cv02` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h06_condl25_cv03` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h06_condl25_cv04` | landed | 533,315 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h21_condl25_cv00` | landed | 294,132 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h21_condl25_cv01` | landed | 294,132 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h21_condl25_cv02` | landed | 294,132 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h21_condl25_cv03` | landed | 294,132 | `f9fe96acc7f1` |  |
| `flyvis_noise_005_s5h21_condl25_cv04` | landed | 294,132 | `f9fe96acc7f1` |  |

<!-- STATUS:END -->

