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
- id: cond_l25
  label: conductance, group lasso 25
  spec_pattern: flyvis_noise_005_s5{horizon}_condl25_{fold}
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
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
  flyvis_noise_005_s5h05_condl25_cv00: '154400341'
  flyvis_noise_005_s5h05_condl25_cv01: '154400342'
  flyvis_noise_005_s5h05_condl25_cv02: '154400343'
  flyvis_noise_005_s5h05_condl25_cv03: '154400344'
  flyvis_noise_005_s5h05_condl25_cv04: '154400345'
  flyvis_noise_005_s5h20_condl25_cv00: '154400346'
  flyvis_noise_005_s5h20_condl25_cv01: '154400347'
  flyvis_noise_005_s5h20_condl25_cv02: '154400348'
  flyvis_noise_005_s5h20_condl25_cv03: '154400349'
  flyvis_noise_005_s5h20_condl25_cv04: '154400350'
report:
  arm_order:
  - current
  - cond_l25
  - conductance
  arm_labels:
    cond_l25: conductance
  axis_labels:
    horizon:
      h05: '5'
      h20: '20'
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

**0/30 landed**, 0 trained (awaiting `-o test_plot`), 25 running, 5 pending

### Landed --- held-out, `results/metrics.txt`

| arm | horizon | n | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | horizon | iter | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | h05 | 492,801 |  | 0.989 ± 0.003 |  |  | 0.735 ± 0.085 | 0.635 ± 0.174 | 0.457 ± 0.113 | 0.442 ± 0.131 | 0.871 ± 0.039 |  |  | 0.645 ± 0.028 |
| current | h20 | 297,331 |  | 0.989 ± 0.003 |  |  | 0.709 ± 0.080 | 0.682 ± 0.140 | 0.426 ± 0.120 | 0.432 ± 0.123 | 0.853 ± 0.048 |  |  | 0.641 ± 0.040 |
| conductance | h05 | 478,401 |  | 0.285 ± 0.093 |  |  | -0.012 ± 0.000 | -0.876 ± 1.533 | 0.445 ± 0.122 | 0.551 ± 0.057 | 0.117 ± 0.068 |  |  | 0.219 ± 0.022 |
| conductance | h20 | 292,259 |  | 0.382 ± 0.079 |  |  | -0.012 ± 0.000 | -0.046 ± 0.585 | 0.411 ± 0.184 | 0.449 ± 0.216 | -0.062 ± 0.048 |  |  | 0.221 ± 0.027 |
| cond_l25 | h05 | 321 |  | 0.005 ± 0.007 |  |  | -0.200 ± 0.167 | -4625.057 ± 9239.700 | -3.506 ± 4.363 | -5.854 ± 11.049 | 0.025 ± 0.056 |  |  |  |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_noise_005_s5h05_cur_cv00` | running | 492,801 | `` |  |
| `flyvis_noise_005_s5h05_cur_cv01` | running | 502,401 | `` |  |
| `flyvis_noise_005_s5h05_cur_cv02` | running | 512,961 | `` |  |
| `flyvis_noise_005_s5h05_cur_cv03` | running | 512,001 | `` |  |
| `flyvis_noise_005_s5h05_cur_cv04` | running | 502,401 | `` |  |
| `flyvis_noise_005_s5h20_cur_cv00` | running | 297,331 | `` |  |
| `flyvis_noise_005_s5h20_cur_cv01` | running | 299,731 | `` |  |
| `flyvis_noise_005_s5h20_cur_cv02` | running | 302,531 | `` |  |
| `flyvis_noise_005_s5h20_cur_cv03` | running | 298,531 | `` |  |
| `flyvis_noise_005_s5h20_cur_cv04` | running | 302,131 | `` |  |
| `flyvis_noise_005_s5h05_condl100_cv00` | running | 492,801 | `` |  |
| `flyvis_noise_005_s5h05_condl100_cv01` | running | 483,201 | `` |  |
| `flyvis_noise_005_s5h05_condl100_cv02` | running | 478,401 | `` |  |
| `flyvis_noise_005_s5h05_condl100_cv03` | running | 481,281 | `` |  |
| `flyvis_noise_005_s5h05_condl100_cv04` | running | 486,401 | `` |  |
| `flyvis_noise_005_s5h20_condl100_cv00` | running | 296,531 | `` |  |
| `flyvis_noise_005_s5h20_condl100_cv01` | running | 296,931 | `` |  |
| `flyvis_noise_005_s5h20_condl100_cv02` | running | 292,792 | `` |  |
| `flyvis_noise_005_s5h20_condl100_cv03` | running | 295,331 | `` |  |
| `flyvis_noise_005_s5h20_condl100_cv04` | running | 294,131 | `` |  |
| `flyvis_noise_005_s5h05_condl25_cv00` | running | 1,281 | `` |  |
| `flyvis_noise_005_s5h05_condl25_cv01` | running | 1,281 | `` |  |
| `flyvis_noise_005_s5h05_condl25_cv02` | running | 1,281 | `` |  |
| `flyvis_noise_005_s5h05_condl25_cv03` | running | 1,281 | `` |  |
| `flyvis_noise_005_s5h05_condl25_cv04` | running | 321 | `` |  |
| `flyvis_noise_005_s5h20_condl25_cv00` | pending |  | `` |  |
| `flyvis_noise_005_s5h20_condl25_cv01` | pending |  | `` |  |
| `flyvis_noise_005_s5h20_condl25_cv02` | pending |  | `` |  |
| `flyvis_noise_005_s5h20_condl25_cv03` | pending |  | `` |  |
| `flyvis_noise_005_s5h20_condl25_cv04` | pending |  | `` |  |

<!-- STATUS:END -->

