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
report:
  arm_order:
  - current
  - cond_l25
  - conductance
  arm_labels:
    cond_l25: conductance
  arm_columns:
    lasso: training.coeff_g_phi_input_group_L1
    horizon: training.rollout_horizon_schedule
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

<!-- STATUS:BEGIN -->

## Status

**30/30 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 0 pending

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
| — | | | | | | | | | | | | | | |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
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
| `flyvis_noise_005_010_condl25rc20_cv00` | landed | 500,817 | `948eb7b17ab2` |  |
| `flyvis_noise_005_010_condl25rc20_cv01` | landed | 516,813 | `948eb7b17ab2` |  |
| `flyvis_noise_005_010_condl25rc20_cv02` | landed | 575,233 | `948eb7b17ab2` |  |
| `flyvis_noise_005_010_condl25rc20_cv03` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_010_condl25rc20_cv04` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_020_condl25rc20_cv00` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_020_condl25rc20_cv01` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_020_condl25rc20_cv02` | landed | 575,233 | `948eb7b17ab2` |  |
| `flyvis_noise_005_020_condl25rc20_cv03` | landed | 575,233 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_020_condl25rc20_cv04` | landed | 568,833 | `948eb7b17ab2` |  |

<!-- STATUS:END -->

