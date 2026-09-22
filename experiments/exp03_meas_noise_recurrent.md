---
number: 3
name: meas_noise_recurrent
title: The general form under measurement noise, with 20-step recurrent training
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
- id: cond_rc20
  label: conductance, lasso 100, recurrent 20
  spec_pattern: flyvis_noise_005_{meas}_condl100_rc20_{fold}
  differs_by:
    simulation.measurement_noise_level: 0.1 / 0.2
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 100.0
    training.recurrent_training: true
    training.rollout_horizon_schedule: '[1..20]'
---

# Experiment 3 — meas_noise_recurrent

**The general form under measurement noise, with 20-step recurrent training**

**Purpose.** does 20-step recurrent training recover the circuit at measurement noise 0.1 and 0.2, where one-step training fell to R2_W 0.63 and 0.38 in the published rows; and does the general form g_phi = MLP(a_i, a_j, v_i, v_j) under a group lasso of 100 recover as well as the current form while killing the per-edge offset C_ij, read as R2_Vrest against R2_Vrest without the C_i correction

## What differs

Model noise is fixed at 0.05; the axis is **measurement** noise, 0.1 and
0.2, on the `flyvis_noise_005_{010,020}_blank50` datasets. The edge
function is experiment 2's — `flyvis_conductance`, `input_size: 6`,
`coeff_g_phi_input_group_L1: 100` — so the only new thing here is the
training scheme.

**The recurrent ladder is self-contained.** Slide 6's `rc10` ran
`rollout_horizon_schedule: [1..10]` over ten epochs from scratch; `rc20`
and `rc40` were *continuations* from the previous rung's checkpoint. One
run to twenty needs the same ramp, twenty long, so the horizon grows with
the model rather than starting at its deepest — which is what the ladder
was for. `data_augmentation_loop` is 50, as in the ladder, because Niter
is divided by the horizon and the compute per epoch has to stay fixed.

**Watch the horizon against tau.** At K = 20 and `delta_t` 0.02 the
unrolled window is 0.4 s against a tau near 0.02 s — twenty time
constants, so the target may have forgotten its initial condition. If
`rollout r` rises while `R2_W` falls, that is what happened. The warning
is the rc20 spec's own.

**96 h wall**, against 48 for experiments 1 and 2: twenty epochs of
recurrent training unroll up to twenty steps per iteration.

## Specs

10 = 2 measurement-noise levels x 5 folds.

| meas noise | fold | spec | dataset |
|---|---|---|---|
| 0.1 | cv00 | `flyvis_noise_005_010_condl100_rc20_cv00` | `flyvis_noise_005_010_blank50_cv00` |
| 0.1 | cv01 | `flyvis_noise_005_010_condl100_rc20_cv01` | `flyvis_noise_005_010_blank50_cv01` |
| 0.1 | cv02 | `flyvis_noise_005_010_condl100_rc20_cv02` | `flyvis_noise_005_010_blank50_cv02` |
| 0.1 | cv03 | `flyvis_noise_005_010_condl100_rc20_cv03` | `flyvis_noise_005_010_blank50_cv03` |
| 0.1 | cv04 | `flyvis_noise_005_010_condl100_rc20_cv04` | `flyvis_noise_005_010_blank50_cv04` |
| 0.2 | cv00 | `flyvis_noise_005_020_condl100_rc20_cv00` | `flyvis_noise_005_020_blank50_cv00` |
| 0.2 | cv01 | `flyvis_noise_005_020_condl100_rc20_cv01` | `flyvis_noise_005_020_blank50_cv01` |
| 0.2 | cv02 | `flyvis_noise_005_020_condl100_rc20_cv02` | `flyvis_noise_005_020_blank50_cv02` |
| 0.2 | cv03 | `flyvis_noise_005_020_condl100_rc20_cv03` | `flyvis_noise_005_020_blank50_cv03` |
| 0.2 | cv04 | `flyvis_noise_005_020_condl100_rc20_cv04` | `flyvis_noise_005_020_blank50_cv04` |

<!-- STATUS:BEGIN -->

## Status

**0/10 landed**, 0 running, 10 pending

### Landed --- held-out, `results/metrics.txt`

| arm | meas | n | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | meas | iter | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | |

### Per run

| run | status | iter | commit |
|---|---|---|---|
| `flyvis_noise_005_010_condl100_rc20_cv00` | pending |  | `` |
| `flyvis_noise_005_010_condl100_rc20_cv01` | pending |  | `` |
| `flyvis_noise_005_010_condl100_rc20_cv02` | pending |  | `` |
| `flyvis_noise_005_010_condl100_rc20_cv03` | pending |  | `` |
| `flyvis_noise_005_010_condl100_rc20_cv04` | pending |  | `` |
| `flyvis_noise_005_020_condl100_rc20_cv00` | pending |  | `` |
| `flyvis_noise_005_020_condl100_rc20_cv01` | pending |  | `` |
| `flyvis_noise_005_020_condl100_rc20_cv02` | pending |  | `` |
| `flyvis_noise_005_020_condl100_rc20_cv03` | pending |  | `` |
| `flyvis_noise_005_020_condl100_rc20_cv04` | pending |  | `` |

<!-- STATUS:END -->

