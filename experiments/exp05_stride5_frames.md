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
  flyvis_noise_005_s5h20_condl25_cv01: '154428043'
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
analyse_job_ids:
  flyvis_noise_005_s5h05_cur_cv00: '154431206'
  flyvis_noise_005_s5h05_cur_cv01: '154431207'
  flyvis_noise_005_s5h05_cur_cv02: '154431208'
  flyvis_noise_005_s5h05_cur_cv03: '154431209'
  flyvis_noise_005_s5h05_cur_cv04: '154431210'
  flyvis_noise_005_s5h20_cur_cv00: '154431211'
  flyvis_noise_005_s5h20_cur_cv01: '154431212'
  flyvis_noise_005_s5h20_cur_cv02: '154431213'
  flyvis_noise_005_s5h20_cur_cv03: '154431214'
  flyvis_noise_005_s5h20_cur_cv04: '154431215'
  flyvis_noise_005_s5h05_condl100_cv00: '154431216'
  flyvis_noise_005_s5h05_condl100_cv01: '154431217'
  flyvis_noise_005_s5h05_condl100_cv02: '154431218'
  flyvis_noise_005_s5h05_condl100_cv03: '154431219'
  flyvis_noise_005_s5h05_condl100_cv04: '154431220'
  flyvis_noise_005_s5h20_condl100_cv00: '154431221'
  flyvis_noise_005_s5h20_condl100_cv01: '154431222'
  flyvis_noise_005_s5h20_condl100_cv02: '154431223'
  flyvis_noise_005_s5h20_condl100_cv03: '154431224'
  flyvis_noise_005_s5h20_condl100_cv04: '154431225'
  flyvis_noise_005_s5h05_condl25_cv00: '154431265'
  flyvis_noise_005_s5h05_condl25_cv01: '154431266'
  flyvis_noise_005_s5h05_condl25_cv02: '154431267'
  flyvis_noise_005_s5h05_condl25_cv03: '154431268'
  flyvis_noise_005_s5h05_condl25_cv04: '154431269'
  flyvis_noise_005_s5h20_condl25_cv00: '154431270'
  flyvis_noise_005_s5h20_condl25_cv01: '154431271'
  flyvis_noise_005_s5h20_condl25_cv02: '154431272'
  flyvis_noise_005_s5h20_condl25_cv03: '154431273'
  flyvis_noise_005_s5h20_condl25_cv04: '154431274'
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

**30/30 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | horizon | n | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | h05 | 5 | 0.973 ± 0.008 | 0.990 ± 0.003 | 0.977 ± 0.015 (0.0) | 0.526 ± 0.057 (4.9) | 0.753 ± 0.076 (0.1) | 0.655 ± 0.176 (6.2) | 0.458 ± 0.120 (37.4) | 0.433 ± 0.144 | 0.878 ± 0.038 (2.7) | 0.059 ± 0.016 |  | 0.642 ± 0.039 |
| current | h20 | 5 | 0.975 ± 0.009 | 0.991 ± 0.002 | 0.976 ± 0.017 (0.0) | 0.539 ± 0.052 (4.8) | 0.722 ± 0.071 (0.1) | 0.689 ± 0.136 (6.6) | 0.434 ± 0.125 (38.7) | 0.446 ± 0.125 | 0.859 ± 0.045 (3.5) | 0.054 ± 0.014 |  | 0.544 ± 0.116 |
| conductance | h05 | 5 | 0.296 ± 0.098 | 0.360 ± 0.079 | 0.270 ± 0.128 (19.3) | 0.270 ± 0.128 (19.3) | -0.012 ± 0.000 (0.4) | -0.751 ± 1.486 (19.9) | 0.362 ± 0.161 (77.7) | 0.434 ± 0.162 | 0.029 ± 0.036 (36.4) | 0.182 ± 0.351 |  | 0.292 ± 0.045 |
| conductance | h20 | 5 | 0.379 ± 0.093 | 0.451 ± 0.049 | 0.378 ± 0.106 (17.0) | 0.380 ± 0.103 (16.7) | -0.012 ± 0.001 (0.4) | -0.135 ± 0.663 (21.7) | 0.444 ± 0.098 (75.3) | 0.433 ± 0.145 | -0.048 ± 0.043 (38.7) | 2.672 ± 3.986 |  | 0.322 ± 0.067 |
| cond_l25 | h05 | 5 | 0.288 ± 0.147 | 0.377 ± 0.078 | 0.288 ± 0.128 (20.7) | 0.288 ± 0.128 (20.7) | -0.012 ± 0.000 (0.4) | -0.161 ± 0.805 (27.8) | 0.338 ± 0.296 (77.8) | 0.364 ± 0.352 | 0.013 ± 0.038 (34.6) | 0.026 ± 0.034 |  | 0.319 ± 0.034 |
| cond_l25 | h20 | 5 | 0.293 ± 0.077 | 0.383 ± 0.054 | 0.288 ± 0.083 (20.9) | 0.291 ± 0.084 (19.1) | -0.037 ± 0.024 (0.5) | 0.147 ± 0.864 (29.0) | 0.354 ± 0.183 (79.3) | 0.466 ± 0.088 | -0.073 ± 0.028 (54.1) | 115.067 ± 217.546 |  | 0.327 ± 0.050 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | horizon | iter | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_noise_005_s5h05_cur_cv00` | landed | 638,401 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h05_cur_cv01` | landed | 638,401 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h05_cur_cv02` | landed | 638,401 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h05_cur_cv03` | landed | 638,401 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h05_cur_cv04` | landed | 638,401 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h20_cur_cv00` | landed | 332,931 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h20_cur_cv01` | landed | 332,931 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h20_cur_cv02` | landed | 332,931 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h20_cur_cv03` | landed | 332,931 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h20_cur_cv04` | landed | 332,931 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h05_condl100_cv00` | landed | 638,401 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h05_condl100_cv01` | landed | 638,401 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h05_condl100_cv02` | landed | 638,401 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h05_condl100_cv03` | landed | 638,401 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h05_condl100_cv04` | landed | 638,401 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h20_condl100_cv00` | landed | 332,931 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h20_condl100_cv01` | landed | 332,931 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h20_condl100_cv02` | landed | 332,931 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h20_condl100_cv03` | landed | 332,931 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h20_condl100_cv04` | landed | 332,931 | `71e4d78710c4` |  |
| `flyvis_noise_005_s5h05_condl25_cv00` | landed | 603,201 | `948eb7b17ab2` |  |
| `flyvis_noise_005_s5h05_condl25_cv01` | landed | 628,801 | `948eb7b17ab2` |  |
| `flyvis_noise_005_s5h05_condl25_cv02` | landed | 628,801 | `948eb7b17ab2` |  |
| `flyvis_noise_005_s5h05_condl25_cv03` | landed | 630,401 | `948eb7b17ab2` |  |
| `flyvis_noise_005_s5h05_condl25_cv04` | landed | 628,801 | `948eb7b17ab2` |  |
| `flyvis_noise_005_s5h20_condl25_cv00` | landed | 292,792 | `948eb7b17ab2` |  |
| `flyvis_noise_005_s5h20_condl25_cv01` | landed | 287,462 | `948eb7b17ab2` |  |
| `flyvis_noise_005_s5h20_condl25_cv02` | landed | 304,931 | `948eb7b17ab2` |  |
| `flyvis_noise_005_s5h20_condl25_cv03` | landed | 306,131 | `948eb7b17ab2` |  |
| `flyvis_noise_005_s5h20_condl25_cv04` | landed | 307,731 | `948eb7b17ab2` |  |

<!-- STATUS:END -->

