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
  - h05
  - h20
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
job_ids:
  flyvis_noise_005_s5h05_cur_cv00: '154397959'
  flyvis_noise_005_s5h05_cur_cv01: '154397960'
  flyvis_noise_005_s5h05_cur_cv02: '154397961'
  flyvis_noise_005_s5h05_cur_cv03: '154397962'
  flyvis_noise_005_s5h05_cur_cv04: '154397963'
  flyvis_noise_005_s5h20_cur_cv00: '154397964'
  flyvis_noise_005_s5h20_cur_cv01: '154397965'
  flyvis_noise_005_s5h20_cur_cv02: '154397966'
  flyvis_noise_005_s5h20_cur_cv03: '154397967'
  flyvis_noise_005_s5h20_cur_cv04: '154397968'
  flyvis_noise_005_s5h05_condl100_cv00: '154397969'
  flyvis_noise_005_s5h05_condl100_cv01: '154397970'
  flyvis_noise_005_s5h05_condl100_cv02: '154397971'
  flyvis_noise_005_s5h05_condl100_cv03: '154397974'
  flyvis_noise_005_s5h05_condl100_cv04: '154397975'
  flyvis_noise_005_s5h20_condl100_cv00: '154397976'
  flyvis_noise_005_s5h20_condl100_cv01: '154397977'
  flyvis_noise_005_s5h20_condl100_cv02: '154397978'
  flyvis_noise_005_s5h20_condl100_cv03: '154397979'
  flyvis_noise_005_s5h20_condl100_cv04: '154397980'
report:
  arm_order:
  - current
  - conductance
  arm_columns:
    lasso: training.coeff_g_phi_input_group_L1
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
sparse, which is what observing one frame in five actually is. At horizon
20 the loss lands on steps 5, 10, 15 and 20 and nowhere else.

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

<!-- STATUS:BEGIN -->

## Status

**0/20 landed**, 0 trained (awaiting `-o test_plot`), 20 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | horizon | n | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | horizon | iter | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | h05 | 441,601 |  | 0.988 ± 0.003 |  |  | 0.725 ± 0.089 | 0.611 ± 0.180 | 0.452 ± 0.113 | 0.435 ± 0.129 | 0.865 ± 0.040 |  |  | 0.634 ± 0.046 |
| current | h20 | 281,593 |  | 0.989 ± 0.003 |  |  | 0.702 ± 0.084 | 0.679 ± 0.138 | 0.433 ± 0.116 | 0.432 ± 0.111 | 0.850 ± 0.049 |  |  | 0.639 ± 0.034 |
| conductance | h05 | 427,201 |  | 0.292 ± 0.082 |  |  | -0.012 ± 0.000 | -1.036 ± 1.800 | 0.451 ± 0.095 | 0.537 ± 0.061 | 0.097 ± 0.077 |  |  | 0.218 ± 0.031 |
| conductance | h20 | 275,197 |  | 0.389 ± 0.073 |  |  | -0.016 ± 0.006 | 0.016 ± 0.533 | 0.499 ± 0.104 | 0.559 ± 0.146 | -0.062 ± 0.062 |  |  | 0.238 ± 0.034 |

### Per run

| run | status | iter | commit |
|---|---|---|---|
| `flyvis_noise_005_s5h05_cur_cv00` | running | 443,201 | `` |
| `flyvis_noise_005_s5h05_cur_cv01` | running | 449,281 | `` |
| `flyvis_noise_005_s5h05_cur_cv02` | running | 459,201 | `` |
| `flyvis_noise_005_s5h05_cur_cv03` | running | 456,001 | `` |
| `flyvis_noise_005_s5h05_cur_cv04` | running | 449,281 | `` |
| `flyvis_noise_005_s5h20_cur_cv00` | running | 281,593 | `` |
| `flyvis_noise_005_s5h20_cur_cv01` | running | 283,731 | `` |
| `flyvis_noise_005_s5h20_cur_cv02` | running | 287,995 | `` |
| `flyvis_noise_005_s5h20_cur_cv03` | running | 282,771 | `` |
| `flyvis_noise_005_s5h20_cur_cv04` | running | 286,929 | `` |
| `flyvis_noise_005_s5h05_condl100_cv00` | running | 443,201 | `` |
| `flyvis_noise_005_s5h05_condl100_cv01` | running | 433,601 | `` |
| `flyvis_noise_005_s5h05_condl100_cv02` | running | 427,201 | `` |
| `flyvis_noise_005_s5h05_condl100_cv03` | running | 430,401 | `` |
| `flyvis_noise_005_s5h05_condl100_cv04` | running | 436,801 | `` |
| `flyvis_noise_005_s5h20_condl100_cv00` | running | 281,593 | `` |
| `flyvis_noise_005_s5h20_condl100_cv01` | running | 281,593 | `` |
| `flyvis_noise_005_s5h20_condl100_cv02` | running | 275,730 | `` |
| `flyvis_noise_005_s5h20_condl100_cv03` | running | 278,928 | `` |
| `flyvis_noise_005_s5h20_condl100_cv04` | running | 277,862 | `` |

<!-- STATUS:END -->

