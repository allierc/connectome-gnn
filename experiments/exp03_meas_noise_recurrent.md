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

**0/20 landed**, 20 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | meas | n | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | meas | iter | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | 010 | 557,872 |  | 0.994 ± 0.000 | 0.767 ± 0.011 | 0.552 ± 0.039 | 0.721 ± 0.031 | 0.693 ± 0.051 | 0.882 ± 0.034 |  |  | 0.849 ± 0.012 |
| current | 020 | 557,428 |  | 0.983 ± 0.002 | 0.577 ± 0.038 | 0.139 ± 0.057 | 0.615 ± 0.021 | 0.606 ± 0.033 | 0.797 ± 0.032 |  |  | 0.809 ± 0.028 |
| conductance | 010 | 546,553 |  | 0.896 ± 0.194 | 0.616 ± 0.314 | 0.411 ± 0.387 | 0.661 ± 0.048 | 0.664 ± 0.047 | 0.714 ± 0.366 |  |  | 0.747 ± 0.212 |
| conductance | 020 | 498,972 |  | 0.981 ± 0.001 | 0.579 ± 0.031 | 0.155 ± 0.041 | 0.530 ± 0.043 | 0.579 ± 0.046 | 0.821 ± 0.030 |  |  | 0.826 ± 0.025 |

### Per run

| run | status | iter | commit |
|---|---|---|---|
| `flyvis_noise_005_010_currc20_cv00` | running | 563,422 | `` |
| `flyvis_noise_005_010_currc20_cv01` | running | 560,896 | `` |
| `flyvis_noise_005_010_currc20_cv02` | running | 558,316 | `` |
| `flyvis_noise_005_010_currc20_cv03` | running | 559,380 | `` |
| `flyvis_noise_005_010_currc20_cv04` | running | 559,296 | `` |
| `flyvis_noise_005_020_currc20_cv00` | running | 557,428 | `` |
| `flyvis_noise_005_020_currc20_cv01` | running | 558,760 | `` |
| `flyvis_noise_005_020_currc20_cv02` | running | 557,428 | `` |
| `flyvis_noise_005_020_currc20_cv03` | running | 559,633 | `` |
| `flyvis_noise_005_020_currc20_cv04` | running | 559,296 | `` |
| `flyvis_noise_005_010_condl100rc20_cv00` | running | 551,212 | `` |
| `flyvis_noise_005_010_condl100rc20_cv01` | running | 553,432 | `` |
| `flyvis_noise_005_010_condl100rc20_cv02` | running | 552,100 | `` |
| `flyvis_noise_005_010_condl100rc20_cv03` | running | 547,023 | `` |
| `flyvis_noise_005_010_condl100rc20_cv04` | running | 552,100 | `` |
| `flyvis_noise_005_020_condl100rc20_cv00` | running | 550,764 | `` |
| `flyvis_noise_005_020_condl100rc20_cv01` | running | 552,988 | `` |
| `flyvis_noise_005_020_condl100rc20_cv02` | running | 515,100 | `` |
| `flyvis_noise_005_020_condl100rc20_cv03` | running | 520,780 | `` |
| `flyvis_noise_005_020_condl100rc20_cv04` | running | 499,587 | `` |

<!-- STATUS:END -->

