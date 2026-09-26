---
number: 3
name: meas_noise_recurrent
title: 'Measurement noise with 20-step recurrent training: current against the general
  form'
purpose: does 20-step recurrent training recover the circuit at measurement noise
  0.1 and 0.2, where one-step training fell to R2_W 0.63 and 0.38 in the published
  rows; and does the general form g_phi = MLP(a_i, a_j, v_i, v_j) under a group lasso
  of 100 recover as well as the current form while killing the per-edge offset C_ij,
  read as R2_Vrest against R2_Vrest without the C_i correction
baseline: experiments/baseline/gnn_current_baseline.yaml
specs_dir: experiments/specs/exp03/fly
task: train
queue: gpu_rtx6000
wall: '96:00'
axes:
  meas:
  - '010'
  - '020'
  fold:
  - cv00
  - cv01
  - cv02
  - cv03
  - cv04
arms:
- id: current_1s
  label: current form, one-step (control)
  spec_pattern: flyvis_noise_005_{meas}_cur1s_{fold}
  differs_by:
    training.recurrent_training: false
- id: current
  label: current form, no lasso, recurrent 20
  spec_pattern: flyvis_noise_005_{meas}_currc20_{fold}
  differs_by:
    training.recurrent_training: true
    training.rollout_horizon_schedule: '[1..20]'
- id: conductance
  label: conductance, lasso 100, recurrent 20
  spec_pattern: flyvis_noise_005_{meas}_condl100rc20_{fold}
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 100.0
    training.recurrent_training: true
    training.rollout_horizon_schedule: '[1..20]'
- id: cond_l25
  label: conductance, lasso 25, recurrent 20
  spec_pattern: flyvis_noise_005_{meas}_condl25rc20_{fold}
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
    training.recurrent_training: true
    training.rollout_horizon_schedule: '[1..20]'
- id: current_bi8
  label: current form, recurrent, burn-in 8, warm start 1 epoch
  spec_pattern: flyvis_noise_005_{meas}_curbi8w1_{fold}
  differs_by:
    training.recurrent_training: true
    training.rollout_horizon_schedule: '[9..20]'
    training.rollout_burn_in: 8
    training.rollout_burn_in_start_epoch: 1
- id: cond_l25_bi8
  label: conductance, lasso 25, recurrent, burn-in 8, warm start 1 epoch
  spec_pattern: flyvis_noise_005_{meas}_condl25bi8w1_{fold}
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
    training.recurrent_training: true
    training.rollout_horizon_schedule: '[9..20]'
    training.rollout_burn_in: 8
    training.rollout_burn_in_start_epoch: 1
job_ids:
  flyvis_noise_005_010_currc20_cv00: '154396214'
  flyvis_noise_005_010_currc20_cv01: '154396215'
  flyvis_noise_005_010_currc20_cv02: '154396216'
  flyvis_noise_005_010_currc20_cv03: '154396217'
  flyvis_noise_005_010_currc20_cv04: '154396218'
  flyvis_noise_005_020_currc20_cv00: '154396219'
  flyvis_noise_005_020_currc20_cv01: '154396220'
  flyvis_noise_005_020_currc20_cv02: '154396221'
  flyvis_noise_005_020_currc20_cv03: '154396222'
  flyvis_noise_005_020_currc20_cv04: '154396223'
  flyvis_noise_005_010_condl100rc20_cv00: '154396224'
  flyvis_noise_005_010_condl100rc20_cv01: '154396225'
  flyvis_noise_005_010_condl100rc20_cv02: '154396226'
  flyvis_noise_005_010_condl100rc20_cv03: '154396227'
  flyvis_noise_005_010_condl100rc20_cv04: '154396228'
  flyvis_noise_005_020_condl100rc20_cv00: '154396229'
  flyvis_noise_005_020_condl100rc20_cv01: '154396230'
  flyvis_noise_005_020_condl100rc20_cv02: '154396231'
  flyvis_noise_005_020_condl100rc20_cv03: '154396232'
  flyvis_noise_005_020_condl100rc20_cv04: '154396233'
  flyvis_noise_005_010_condl25rc20_cv00: '154400326'
  flyvis_noise_005_010_condl25rc20_cv01: '154400327'
  flyvis_noise_005_010_condl25rc20_cv02: '154400328'
  flyvis_noise_005_010_condl25rc20_cv03: '154400329'
  flyvis_noise_005_010_condl25rc20_cv04: '154400330'
  flyvis_noise_005_020_condl25rc20_cv00: '154400331'
  flyvis_noise_005_020_condl25rc20_cv01: '154400332'
  flyvis_noise_005_020_condl25rc20_cv02: '154400333'
  flyvis_noise_005_020_condl25rc20_cv03: '154400334'
  flyvis_noise_005_020_condl25rc20_cv04: '154400335'
  flyvis_noise_005_010_cur1s_cv00: '154452587'
  flyvis_noise_005_010_cur1s_cv01: '154452588'
  flyvis_noise_005_010_cur1s_cv02: '154452589'
  flyvis_noise_005_010_cur1s_cv03: '154452590'
  flyvis_noise_005_010_cur1s_cv04: '154452591'
  flyvis_noise_005_020_cur1s_cv00: '154452592'
  flyvis_noise_005_020_cur1s_cv01: '154452593'
  flyvis_noise_005_020_cur1s_cv02: '154452594'
  flyvis_noise_005_020_cur1s_cv03: '154452595'
  flyvis_noise_005_020_cur1s_cv04: '154452596'
  flyvis_noise_005_010_curbi8w1_cv00: '154452605'
  flyvis_noise_005_010_curbi8w1_cv01: '154452606'
  flyvis_noise_005_010_curbi8w1_cv02: '154452607'
  flyvis_noise_005_010_curbi8w1_cv03: '154452608'
  flyvis_noise_005_010_curbi8w1_cv04: '154452609'
  flyvis_noise_005_020_curbi8w1_cv00: '154452610'
  flyvis_noise_005_020_curbi8w1_cv01: '154452611'
  flyvis_noise_005_020_curbi8w1_cv02: '154452612'
  flyvis_noise_005_020_curbi8w1_cv03: '154452613'
  flyvis_noise_005_020_curbi8w1_cv04: '154452614'
  flyvis_noise_005_010_condl25bi8w1_cv00: '154452617'
  flyvis_noise_005_010_condl25bi8w1_cv01: '154452618'
  flyvis_noise_005_010_condl25bi8w1_cv02: '154452620'
  flyvis_noise_005_010_condl25bi8w1_cv03: '154452621'
  flyvis_noise_005_010_condl25bi8w1_cv04: '154452622'
  flyvis_noise_005_020_condl25bi8w1_cv00: '154452623'
  flyvis_noise_005_020_condl25bi8w1_cv01: '154452624'
  flyvis_noise_005_020_condl25bi8w1_cv02: '154452625'
  flyvis_noise_005_020_condl25bi8w1_cv03: '154452626'
  flyvis_noise_005_020_condl25bi8w1_cv04: '154452627'
report:
  arm_order:
  - current_1s
  - current
  - current_bi8
  - cond_l25
  - cond_l25_bi8
  - conductance
  arm_labels:
    cond_l25: conductance
    current_1s: current, one-step
    current_bi8: current
    cond_l25_bi8: conductance
  arm_columns:
    lasso: training.coeff_g_phi_input_group_L1
    horizon: training.rollout_horizon_schedule
    burn-in: training.rollout_burn_in
analyse_job_ids:
  flyvis_noise_005_010_currc20_cv00: '154400355'
  flyvis_noise_005_010_currc20_cv01: '154400356'
  flyvis_noise_005_010_currc20_cv02: '154400357'
  flyvis_noise_005_010_currc20_cv03: '154400358'
  flyvis_noise_005_010_currc20_cv04: '154400359'
  flyvis_noise_005_020_currc20_cv00: '154400360'
  flyvis_noise_005_020_currc20_cv01: '154400361'
  flyvis_noise_005_020_currc20_cv02: '154400362'
  flyvis_noise_005_020_currc20_cv03: '154400363'
  flyvis_noise_005_020_currc20_cv04: '154400364'
  flyvis_noise_005_010_condl100rc20_cv00: '154400365'
  flyvis_noise_005_010_condl100rc20_cv01: '154400366'
  flyvis_noise_005_010_condl100rc20_cv02: '154400367'
  flyvis_noise_005_010_condl100rc20_cv04: '154400368'
  flyvis_noise_005_020_condl100rc20_cv01: '154400369'
  flyvis_noise_005_010_condl100rc20_cv03: '154431185'
  flyvis_noise_005_020_condl100rc20_cv00: '154431186'
  flyvis_noise_005_020_condl100rc20_cv02: '154431187'
  flyvis_noise_005_020_condl100rc20_cv03: '154431188'
  flyvis_noise_005_020_condl100rc20_cv04: '154431189'
  flyvis_noise_005_010_condl25rc20_cv00: '154431275'
  flyvis_noise_005_010_condl25rc20_cv01: '154431276'
  flyvis_noise_005_010_condl25rc20_cv02: '154431277'
  flyvis_noise_005_010_condl25rc20_cv03: '154431278'
  flyvis_noise_005_010_condl25rc20_cv04: '154431279'
  flyvis_noise_005_020_condl25rc20_cv00: '154431280'
  flyvis_noise_005_020_condl25rc20_cv01: '154431281'
  flyvis_noise_005_020_condl25rc20_cv02: '154431282'
  flyvis_noise_005_020_condl25rc20_cv03: '154431283'
  flyvis_noise_005_020_condl25rc20_cv04: '154431284'
---
# Experiment 3 — meas_noise_recurrent

**Measurement noise with 20-step recurrent training: current against the general form**

**Purpose.** does 20-step recurrent training recover the circuit at measurement noise 0.1 and 0.2, where one-step training fell to R2_W 0.63 and 0.38 in the published rows; and does the general form g_phi = MLP(a_i, a_j, v_i, v_j) under a group lasso of 100 recover as well as the current form while killing the per-edge offset C_ij, read as R2_Vrest against R2_Vrest without the C_i correction

## What differs

Model noise is fixed at 0.05; the axis is **measurement** noise, 0.1 and
0.2, on the `flyvis_noise_005_{010,020}_blank50` datasets, which already
exist with the noise baked into `noise.zarr` at exactly sd 0.1 and 0.2.

**Both arms are recurrent.** The control is the current form with no
lasso; without it, "the general form works under measurement noise"
could not be told apart from "recurrent training works under measurement
noise", which is slide 6's own finding and not this experiment's.

| | `current` | `conductance` |
|---|---|---|
| `signal_model_name` | `flyvis_current` | `flyvis_conductance` |
| `input_size` | 3 | 6 |
| `coeff_g_phi_input_group_L1` | 0 | 100 |
| recurrent | 20-step, schedule [1..20] | same |

**The recurrent ladder is self-contained.** Slide 6's `rc10` ran
`rollout_horizon_schedule: [1..10]` over ten epochs from scratch; `rc20`
and `rc40` were *continuations* from the previous rung's checkpoint. One
run to twenty needs the same ramp, twenty long. `data_augmentation_loop`
is 50, as in the ladder, because Niter is divided by the horizon and the
compute per epoch has to stay fixed.

**Watch the horizon against tau.** At K = 20 and `delta_t` 0.02 the
unrolled window is 0.4 s against a tau near 0.02 s — twenty time
constants, so the target may have forgotten its initial condition. If
`rollout r` rises while `R2_W` falls, that is what happened. The warning
is the rc20 spec's own.

## Reading it against slide 6

Slide 6 stacks two different measurements: its published 5-fold rows
(0.63, 0.38) are the **chain** readout under the old vocabulary
(`W_corrected_R2`, from `log/fly/archive_4/`), and its recurrent rows
(0.78, 0.61) are the **template** readout on one fold. Both arms here use
the template readout on five folds, so this experiment is internally
comparable and only loosely comparable with the published pair.

## Specs

20 = 2 arms x 2 measurement-noise levels x 5 folds.

| arm | meas noise | fold | spec | dataset |
|---|---|---|---|---|
| current | 0.1 | cv00 | `flyvis_noise_005_010_currc20_cv00` | `flyvis_noise_005_010_blank50_cv00` |
| current | 0.1 | cv01 | `flyvis_noise_005_010_currc20_cv01` | `flyvis_noise_005_010_blank50_cv01` |
| current | 0.1 | cv02 | `flyvis_noise_005_010_currc20_cv02` | `flyvis_noise_005_010_blank50_cv02` |
| current | 0.1 | cv03 | `flyvis_noise_005_010_currc20_cv03` | `flyvis_noise_005_010_blank50_cv03` |
| current | 0.1 | cv04 | `flyvis_noise_005_010_currc20_cv04` | `flyvis_noise_005_010_blank50_cv04` |
| current | 0.2 | cv00 | `flyvis_noise_005_020_currc20_cv00` | `flyvis_noise_005_020_blank50_cv00` |
| current | 0.2 | cv01 | `flyvis_noise_005_020_currc20_cv01` | `flyvis_noise_005_020_blank50_cv01` |
| current | 0.2 | cv02 | `flyvis_noise_005_020_currc20_cv02` | `flyvis_noise_005_020_blank50_cv02` |
| current | 0.2 | cv03 | `flyvis_noise_005_020_currc20_cv03` | `flyvis_noise_005_020_blank50_cv03` |
| current | 0.2 | cv04 | `flyvis_noise_005_020_currc20_cv04` | `flyvis_noise_005_020_blank50_cv04` |
| conductance | 0.1 | cv00 | `flyvis_noise_005_010_condl100rc20_cv00` | `flyvis_noise_005_010_blank50_cv00` |
| conductance | 0.1 | cv01 | `flyvis_noise_005_010_condl100rc20_cv01` | `flyvis_noise_005_010_blank50_cv01` |
| conductance | 0.1 | cv02 | `flyvis_noise_005_010_condl100rc20_cv02` | `flyvis_noise_005_010_blank50_cv02` |
| conductance | 0.1 | cv03 | `flyvis_noise_005_010_condl100rc20_cv03` | `flyvis_noise_005_010_blank50_cv03` |
| conductance | 0.1 | cv04 | `flyvis_noise_005_010_condl100rc20_cv04` | `flyvis_noise_005_010_blank50_cv04` |
| conductance | 0.2 | cv00 | `flyvis_noise_005_020_condl100rc20_cv00` | `flyvis_noise_005_020_blank50_cv00` |
| conductance | 0.2 | cv01 | `flyvis_noise_005_020_condl100rc20_cv01` | `flyvis_noise_005_020_blank50_cv01` |
| conductance | 0.2 | cv02 | `flyvis_noise_005_020_condl100rc20_cv02` | `flyvis_noise_005_020_blank50_cv02` |
| conductance | 0.2 | cv03 | `flyvis_noise_005_020_condl100rc20_cv03` | `flyvis_noise_005_020_blank50_cv03` |
| conductance | 0.2 | cv04 | `flyvis_noise_005_020_condl100rc20_cv04` | `flyvis_noise_005_020_blank50_cv04` |

## Two controls added 2026-09-25: one-step, and burn-in

**`current_1s`, 10 jobs -- the one-step control.** The `currc20` spec with
`recurrent_training: false` and no horizon schedule, nothing else changed. exp03
had no one-step arm, so its recurrent gain was read against the published rows,
which use a different readout; this arm puts the one-step number in the same
table, on the same readout, folds, seeds and GPU model.

**`current_bi8` and `cond_l25_bi8`, 20 jobs -- the burn-in arms.** The `currc20`
and `condl25rc20` specs with `training.rollout_burn_in: 8` and
`rollout_horizon_schedule: [9..20]` held at 20 for the remaining epochs, nothing
else changed -- verified by parsing each against its parent.

The burn-in leaves rollout steps 0-7 unscored while still integrating them and
back-propagating through them; it is a weight mask, not a detach. It is the
audit's top proposal (below): under measurement noise the dense loss prefers a
shrunken connectome over the true one, and all of that preference sits in steps
0-8. In a known-ODE test at meas 0.20, a burn-in of 8 lifted `R2_W` 0.921 ->
0.988 and R2 on log tau 0.61 -> 0.91. How much of that reaches the GNN is what
these arms measure.

**The first launch started cold and was stopped.** Launched on 2026-09-25 with
the burn-in active from epoch 0, the arms began training at horizon 9 with only
step 8 scored and nothing anchored at step 0. After 20-29k iterations, 8 of the
10 current-form folds were still at W ~ 0 (learned/true slope 0.002-0.006),
while every run that scores step 0 had reached `R2_W` 0.72 by iteration 4,800.
The known-ODE test behind the burn-in started from the TRUE parameters and never
faced that. All 30 new jobs were stopped, including the one-step controls, and
relaunched with `training.rollout_burn_in_start_epoch: 1`: epoch 0 (horizon 9)
scores every step, step 0 included, and the burn-in applies from epoch 1. The
cold-start runs keep their directories (`_curbi8_`, `_condl25bi8_`) as the
evidence; the relaunch is `_curbi8w1_` and `_condl25bi8w1_`, differing only in
that one key.

**These arms change two things, not one.** A burn-in of 8 makes horizons 1-8
score nothing, and the trainer refuses a schedule containing them, so the
schedule starts at 9. The burn-in arms therefore also drop exp03's short-horizon
epochs, including the 160,000-iteration one-step epoch that the audit found sets
the slow cells' leak. If they beat `current` and `cond_l25`, the gain belongs to
the pair; separating the two would take a third arm on `[9..20]` with no burn-in.

The conductance form is at lasso 25, the setting exp02, exp04 and the fixed
exp05 all favour; lasso 100 is not repeated.

| | `current` | `current_bi8` | `cond_l25_bi8` |
|---|---|---|---|
| schedule | `[1..20]` | `[9..20]`, then 20 | `[9..20]`, then 20 |
| scored steps at horizon 20 | 0-19 | 8-19 | 8-19 |
| `coeff_g_phi_input_group_L1` | 0 | 0 | 25 |

## Audit of the dense recurrent path, 2026-09-25

Six reviewers were launched with adversarial verification. Four finished; the
state/target-alignment and curriculum reviewers were cut off, and their ground
is covered by the other four (alignment verified to 2e-6, the curriculum's
iteration budget computed). Verdicts below say which findings were upheld by two
independent refuters and which were only reported.

### No code bug in the dense path

The production `_dense_rollout_loss` gradient matches fp64 central finite
differences to 1e-5 through 20 Euler steps, is bitwise unchanged in loss under
`torch.compile`, and is stable in fp32 (relative error <= 1.4e-3, not growing
with the horizon). No graph cut, no off-by-one: at step s the state is frame
k+s and the target y_ts[k+s], and k+19 stays inside the data. The two datasets
differ only in eta (eta_020 = 2 x eta_010 exactly), so the two columns are a clean
dose-response in measurement noise.

Two small real defects, both upheld 2/2, neither touching the table:
- `observed_derivative_target`'s docstring gives std(d eta/dt) = 14.1 at meas
  0.10 assuming delta_t 0.01; these datasets use 0.02, so it is **7.07** at 0.10
  and 14.14 at 0.20 -- 33% and 66% of the target's variance.
- `tools/pysr_recovery.load_run` sorts checkpoints lexicographically and picks
  `..._graphs_9_5333.pt`, an epoch-9 checkpoint, on exp03 runs.

### What limits exp03 is the objective, not the code or the readout

**The readout is exact.** The true generator wrapped as a GNN reads out at
`Wij_R2` 0.999999 on the same meas-0.20 frames, and none of the readout's knobs
moves the exp03 number.

**Under measurement noise the loss is minimised away from the truth.** At meas
0.20 the trained GNN scores **5.9% below the true generator on its own loss**
(2.9% at 0.10), and all of that preference sits in steps 0-8; from step 9 on,
the truth wins. The mechanism is errors-in-variables:
- at step 0 the input is `v + eta` and the target contains `-eta/dt`, so every
  weight out of a noisy sender is diluted, and slow cells' leak is pulled toward
  `1/dt` (learned -33.5/s, true -6.0/s, one-step theory -35.9/s);
- that eta stays in the state for steps 1-4 (slow cells keep 0.76 of it at step
  1, 0.33 at step 5) while those targets carry only fresh noise, so the loss
  rewards shrinking messages. At steps 1-2 a ZERO connectome scores below the
  true one.

**Where it lands:** 47% of sum(W_true^2) sits on edges from senders whose signal
SD is below the noise (L3, Tm4, L5, Tm1, C2, L4); at meas 0.20 those weights come
back at a median **4% of their true size**. tau_R2 0.22 is the slow 11% of
neurons made 5.5x too fast.

### The recurrent gain is real but about ten times smaller than reported at 0.10

Upheld 2/2. The published 0.634 / 0.384 use the chain readout on all edges;
this file's `R2_W` is the template readout with |error| > 1 edges dropped. The
chain readout at HEAD reproduces all ten published values to 5e-5. Decomposing
0.634 -> 0.769 at meas 0.10:

| source | share |
|---|---|
| metric switch, same checkpoints | +0.092 |
| better one-step recipe | +0.032 |
| **recurrence** | **+0.012** (p = 0.16, not significant) |

At meas 0.20 recurrence adds +0.058 (p = 0.014) -- and it is **W scale, not
wiring**: Pearson over all edges stays at 0.72 while the learned/true gain rises
0.34 -> 0.51. The headline at 0.20 also rests on the outlier band: 0.579 filtered
against 0.499 over all edges, and in 2 of 5 folds recurrence gave no gain on the
paper's metric. The one-step control arm (`current_1s`, launched 2026-09-25)
settles this inside one table.

### Proposals, ranked

1. **Burn-in: leave the first ~8 rollout steps unscored, keep full BPTT.** In a
   known-ODE test trained from the true parameters at meas 0.20, `R2_W` rises
   0.921 -> **0.988** and R2 on log tau 0.61 -> 0.91; at 0.10, 0.966 -> 0.992. A
   reduced model agrees (0.764 -> 0.973 with a 10-step burn-in). How much of this
   reaches the GNN is unmeasured.
   **Zero-code probe:** `rollout_step_weighting: last` with
   `rollout_horizon_schedule: [1, 20]` -- 0.993 in the known-ODE test. 10 jobs.
2. **Report what the headline hides**, no retraining: add `Wij_R2_all` and
   Pearson columns. The 1.0 outlier band always drops L3->Mi9 (26.6% of var W)
   and hid a collapse: `flyvis_noise_005_020_condl25rc20_cv00` reports 0.621 with
   `R2_all` -0.269.
3. **Keep these off under measurement noise**, all measured worse in the reduced
   model at meas 0.20: `noise_recurrent_level` (R2_W -2.16), a resampled-eta
   input (-0.41), pushforward (0.79), a detached burn-in (0.75).
4. Contested or small: dropping the K=1 epoch (0/2 upheld), relaxing
   `coeff_W_L1`, denoising the initial state.

<!-- STATUS:BEGIN -->

## Status

**30/60 landed**, 0 trained (awaiting `-o test_plot`), 20 running, 10 pending

### Landed --- held-out, `results/metrics.txt`

| arm | meas | n | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | 010 | 5 | 0.985 ± 0.001 | 0.994 ± 0.000 | 0.989 ± 0.005 (0.0) | 0.660 ± 0.032 (2.3) | 0.769 ± 0.012 (0.1) | 0.576 ± 0.054 (5.2) | 0.717 ± 0.029 (23.6) | 0.702 ± 0.059 | 0.884 ± 0.035 (0.3) | 0.034 ± 0.011 |  | 0.843 ± 0.014 |
| current | 020 | 5 | 0.956 ± 0.003 | 0.741 ± 0.306 | 0.938 ± 0.030 (0.0) | 0.584 ± 0.042 (4.9) | 0.579 ± 0.039 (0.1) | 0.145 ± 0.054 (9.7) | 0.619 ± 0.027 (40.0) | 0.606 ± 0.018 | 0.801 ± 0.034 (2.6) | 0.043 ± 0.016 |  | 0.810 ± 0.023 |
| conductance | 010 | 5 | 0.905 ± 0.156 | 0.894 ± 0.199 | 0.217 ± 0.148 (60.3) | 0.467 ± 0.023 (0.0) | 0.616 ± 0.314 (0.1) | 0.461 ± 0.407 (6.8) | 0.642 ± 0.043 (35.0) | 0.675 ± 0.053 | 0.724 ± 0.339 (3.7) | 0.035 ± 0.016 |  | 0.752 ± 0.159 |
| conductance | 020 | 5 | 0.954 ± 0.003 | 0.984 ± 0.001 | 0.182 ± 0.026 (72.8) | 0.475 ± 0.025 (0.0) | 0.601 ± 0.027 (0.1) | 0.169 ± 0.034 (9.7) | 0.569 ± 0.051 (39.4) | 0.629 ± 0.021 | 0.819 ± 0.028 (1.3) | 0.041 ± 0.034 |  | 0.798 ± 0.033 |
| cond_l25 | 010 | 5 | 0.983 ± 0.001 | 0.992 ± 0.001 | 0.152 ± 0.030 (73.6) | 0.474 ± 0.011 (0.0) | 0.772 ± 0.008 (0.1) | 0.504 ± 0.046 (5.4) | 0.684 ± 0.008 (24.7) | 0.727 ± 0.035 | 0.905 ± 0.018 (0.3) | 0.019 ± 0.011 |  | 0.852 ± 0.019 |
| cond_l25 | 020 | 5 | 0.953 ± 0.003 | 0.984 ± 0.001 | 0.200 ± 0.037 (70.9) | 0.490 ± 0.015 (0.0) | 0.590 ± 0.042 (0.1) | 0.195 ± 0.031 (9.7) | 0.597 ± 0.031 (39.8) | 0.654 ± 0.040 | 0.825 ± 0.026 (1.6) | 0.023 ± 0.013 |  | 0.807 ± 0.009 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | meas | iter | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current_1s | 010 | 1 |  | 0.003 ± 0.003 |  |  | -0.023 ± 0.006 | -4.934 ± 0.398 | 0.005 ± 0.373 | 0.005 ± 0.373 | -0.030 ± 0.013 |  |  |  |
| current_1s | 020 | 1 |  | 0.004 ± 0.003 |  |  | -0.029 ± 0.000 |  |  |  | -0.018 ± 0.000 |  |  |  |
| current_bi8 | 010 | 1 |  | 0.004 ± 0.003 |  |  |  |  |  |  |  |  |  |  |
| current_bi8 | 020 | 1 |  | 0.003 ± 0.003 |  |  |  |  |  |  |  |  |  |  |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_noise_005_010_cur1s_cv00` | running | 1 | `` |  |
| `flyvis_noise_005_010_cur1s_cv01` | running | 1 | `` |  |
| `flyvis_noise_005_010_cur1s_cv02` | running | 1 | `` |  |
| `flyvis_noise_005_010_cur1s_cv03` | running | 1 | `` |  |
| `flyvis_noise_005_010_cur1s_cv04` | running | 1 | `` |  |
| `flyvis_noise_005_020_cur1s_cv00` | running | 1 | `` |  |
| `flyvis_noise_005_020_cur1s_cv01` | running | 1 | `` |  |
| `flyvis_noise_005_020_cur1s_cv02` | running | 1 | `` |  |
| `flyvis_noise_005_020_cur1s_cv03` | running | 1 | `` |  |
| `flyvis_noise_005_020_cur1s_cv04` | running | 1 | `` |  |
| `flyvis_noise_005_010_currc20_cv00` | landed | 575,233 | `6ff97411577c` |  |
| `flyvis_noise_005_010_currc20_cv01` | landed | 575,233 | `6ff97411577c` |  |
| `flyvis_noise_005_010_currc20_cv02` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_010_currc20_cv03` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_010_currc20_cv04` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_020_currc20_cv00` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_020_currc20_cv01` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_020_currc20_cv02` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_020_currc20_cv03` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_020_currc20_cv04` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_010_condl100rc20_cv00` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_010_condl100rc20_cv01` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_010_condl100rc20_cv02` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_010_condl100rc20_cv03` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_010_condl100rc20_cv04` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_020_condl100rc20_cv00` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_020_condl100rc20_cv01` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_020_condl100rc20_cv02` | landed | 575,233 | `d543c97a1fd0` |  |
| `flyvis_noise_005_020_condl100rc20_cv03` | landed | 575,233 | `900703a509a8` |  |
| `flyvis_noise_005_020_condl100rc20_cv04` | landed | 575,233 | `71e4d78710c4` |  |
| `flyvis_noise_005_010_condl25rc20_cv00` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_010_condl25rc20_cv01` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_010_condl25rc20_cv02` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_010_condl25rc20_cv03` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_010_condl25rc20_cv04` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_020_condl25rc20_cv00` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_020_condl25rc20_cv01` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_020_condl25rc20_cv02` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_020_condl25rc20_cv03` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_020_condl25rc20_cv04` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_010_curbi8w1_cv00` | running | 1 | `` |  |
| `flyvis_noise_005_010_curbi8w1_cv01` | running | 1 | `` |  |
| `flyvis_noise_005_010_curbi8w1_cv02` | running | 1 | `` |  |
| `flyvis_noise_005_010_curbi8w1_cv03` | running | 1 | `` |  |
| `flyvis_noise_005_010_curbi8w1_cv04` | running | 1 | `` |  |
| `flyvis_noise_005_020_curbi8w1_cv00` | running | 1 | `` |  |
| `flyvis_noise_005_020_curbi8w1_cv01` | running | 1 | `` |  |
| `flyvis_noise_005_020_curbi8w1_cv02` | running | 1 | `` |  |
| `flyvis_noise_005_020_curbi8w1_cv03` | running | 1 | `` |  |
| `flyvis_noise_005_020_curbi8w1_cv04` | running | 1 | `` |  |
| `flyvis_noise_005_010_condl25bi8w1_cv00` | pending |  | `` |  |
| `flyvis_noise_005_010_condl25bi8w1_cv01` | pending |  | `` |  |
| `flyvis_noise_005_010_condl25bi8w1_cv02` | pending |  | `` |  |
| `flyvis_noise_005_010_condl25bi8w1_cv03` | pending |  | `` |  |
| `flyvis_noise_005_010_condl25bi8w1_cv04` | pending |  | `` |  |
| `flyvis_noise_005_020_condl25bi8w1_cv00` | pending |  | `` |  |
| `flyvis_noise_005_020_condl25bi8w1_cv01` | pending |  | `` |  |
| `flyvis_noise_005_020_condl25bi8w1_cv02` | pending |  | `` |  |
| `flyvis_noise_005_020_condl25bi8w1_cv03` | pending |  | `` |  |
| `flyvis_noise_005_020_condl25bi8w1_cv04` | pending |  | `` |  |

<!-- STATUS:END -->

