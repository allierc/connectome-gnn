---
number: 9
name: hidden_neurons
title: 20% hidden neurons, current against conductance lasso 25, one-step against
  20-step recurrent training, and five ways to fill the hidden traces
purpose: with 20% of the non-retina neurons unobserved, does 20-step recurrent training
  or a learned hidden-trace network (SIREN(x, y, t), NGP-T, with anchors or a warm-up)
  recover the circuit better than zero-silencing, for the current-form and the conductance
  lasso-25 model; one fold per approach for a landscape, five for the zero-silenced
  baseline
baseline: GraphData/config/fly/flyvis_noise_005_hidden_020_no_ngp_blank50_unified_cv0N.yaml
  (the paper's "20% hidden" row, Supp. Tab. 4)
specs_dir: experiments/specs/exp09/fly
task: train
queue: gpu_rtx6000
wall: '48:00'
n_cpus: 12
axes:
  hidden:
  - none
  - sirentxy
  - sirentxywu
  - ngpt
  - ngptanc
  - ngptwu
  - ngptancwu
  fold:
  - cv00
  - cv01
  - cv02
  - cv03
  - cv04
arms:
- id: cur1s
  label: current, one-step
  spec_pattern: flyvis_noise_005_hid20_{hidden}_cur1s_{fold}
  axes_only:
    hidden:
    - none
  differs_by: {}
- id: cur1s_inr
  label: current, one-step
  spec_pattern: flyvis_noise_005_hid20_{hidden}_cur1s_{fold}
  axes_only:
    hidden:
    - sirentxy
    - sirentxywu
    - ngpt
    - ngptanc
    - ngptwu
    - ngptancwu
    fold:
    - cv00
  differs_by: {}
- id: currc20
  label: current, recurrent 20
  spec_pattern: flyvis_noise_005_hid20_{hidden}_currc20_{fold}
  axes_only:
    hidden:
    - none
  differs_by:
    training.recurrent_training: true
- id: currc20_inr
  label: current, recurrent 20
  spec_pattern: flyvis_noise_005_hid20_{hidden}_currc20_{fold}
  axes_only:
    hidden:
    - sirentxy
    - sirentxywu
    - ngpt
    - ngptanc
    - ngptwu
    - ngptancwu
    fold:
    - cv00
  differs_by:
    training.recurrent_training: true
- id: condl251s
  label: conductance lasso 25, one-step
  spec_pattern: flyvis_noise_005_hid20_{hidden}_condl251s_{fold}
  axes_only:
    hidden:
    - none
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
- id: condl251s_inr
  label: conductance lasso 25, one-step
  spec_pattern: flyvis_noise_005_hid20_{hidden}_condl251s_{fold}
  axes_only:
    hidden:
    - sirentxy
    - sirentxywu
    - ngpt
    - ngptanc
    - ngptwu
    - ngptancwu
    fold:
    - cv00
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
- id: condl25rc20
  label: conductance lasso 25, recurrent 20
  spec_pattern: flyvis_noise_005_hid20_{hidden}_condl25rc20_{fold}
  axes_only:
    hidden:
    - none
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
    training.recurrent_training: true
- id: condl25rc20_inr
  label: conductance lasso 25, recurrent 20
  spec_pattern: flyvis_noise_005_hid20_{hidden}_condl25rc20_{fold}
  axes_only:
    hidden:
    - sirentxy
    - sirentxywu
    - ngpt
    - ngptanc
    - ngptwu
    - ngptancwu
    fold:
    - cv00
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
    training.recurrent_training: true
report:
  axis_labels:
    hidden:
      none: zero-silenced
      sirentxy: SIREN(x,y,t)
      sirentxywu: SIREN(x,y,t) + warm-up
      ngpt: NGP-T
      ngptanc: NGP-T + anchors
      ngptwu: NGP-T + warm-up
      ngptancwu: NGP-T + anchors + warm-up
  metric_columns:
    hidden_rollout_pearson: hidden trace r
job_ids:
  flyvis_noise_005_hid20_none_cur1s_cv00: '154469613'
  flyvis_noise_005_hid20_none_cur1s_cv01: '154469614'
  flyvis_noise_005_hid20_none_cur1s_cv02: '154469615'
  flyvis_noise_005_hid20_none_cur1s_cv03: '154469616'
  flyvis_noise_005_hid20_none_cur1s_cv04: '154469617'
  flyvis_noise_005_hid20_sirentxy_cur1s_cv00: '154469631'
  flyvis_noise_005_hid20_sirentxywu_cur1s_cv00: '154469632'
  flyvis_noise_005_hid20_ngpt_cur1s_cv00: '154469633'
  flyvis_noise_005_hid20_ngptanc_cur1s_cv00: '154469634'
  flyvis_noise_005_hid20_ngptwu_cur1s_cv00: '154469635'
  flyvis_noise_005_hid20_ngptancwu_cur1s_cv00: '154469636'
  flyvis_noise_005_hid20_none_condl251s_cv00: '154469645'
  flyvis_noise_005_hid20_none_condl251s_cv01: '154469646'
  flyvis_noise_005_hid20_none_condl251s_cv02: '154469647'
  flyvis_noise_005_hid20_none_condl251s_cv03: '154469648'
  flyvis_noise_005_hid20_none_condl251s_cv04: '154469649'
  flyvis_noise_005_hid20_sirentxy_condl251s_cv00: '154469650'
  flyvis_noise_005_hid20_sirentxywu_condl251s_cv00: '154469651'
  flyvis_noise_005_hid20_ngpt_condl251s_cv00: '154469652'
  flyvis_noise_005_hid20_ngptanc_condl251s_cv00: '154469653'
  flyvis_noise_005_hid20_ngptwu_condl251s_cv00: '154469654'
  flyvis_noise_005_hid20_ngptancwu_condl251s_cv00: '154469655'
  flyvis_noise_005_hid20_none_currc20_cv00: '154469722'
  flyvis_noise_005_hid20_none_currc20_cv01: '154469723'
  flyvis_noise_005_hid20_none_currc20_cv02: '154469724'
  flyvis_noise_005_hid20_none_currc20_cv03: '154469726'
  flyvis_noise_005_hid20_none_currc20_cv04: '154469727'
  flyvis_noise_005_hid20_sirentxy_currc20_cv00: '154469731'
  flyvis_noise_005_hid20_sirentxywu_currc20_cv00: '154469732'
  flyvis_noise_005_hid20_ngpt_currc20_cv00: '154469733'
  flyvis_noise_005_hid20_ngptanc_currc20_cv00: '154469734'
  flyvis_noise_005_hid20_ngptwu_currc20_cv00: '154469735'
  flyvis_noise_005_hid20_ngptancwu_currc20_cv00: '154469736'
  flyvis_noise_005_hid20_none_condl25rc20_cv00: '154469737'
  flyvis_noise_005_hid20_none_condl25rc20_cv01: '154469738'
  flyvis_noise_005_hid20_none_condl25rc20_cv02: '154469739'
  flyvis_noise_005_hid20_none_condl25rc20_cv03: '154469740'
  flyvis_noise_005_hid20_none_condl25rc20_cv04: '154469741'
  flyvis_noise_005_hid20_sirentxy_condl25rc20_cv00: '154469742'
  flyvis_noise_005_hid20_sirentxywu_condl25rc20_cv00: '154469743'
  flyvis_noise_005_hid20_ngpt_condl25rc20_cv00: '154469744'
  flyvis_noise_005_hid20_ngptanc_condl25rc20_cv00: '154469745'
  flyvis_noise_005_hid20_ngptwu_condl25rc20_cv00: '154469746'
  flyvis_noise_005_hid20_ngptancwu_condl25rc20_cv00: '154469747'
---

# Experiment 9 — hidden_neurons

**20% hidden neurons, current against conductance lasso 25, one-step against 20-step recurrent training, and five ways to fill the hidden traces**

**Purpose.** with 20% of the non-retina neurons unobserved, does 20-step recurrent training or a learned hidden-trace network (SIREN(x, y, t), NGP-T, with anchors or a warm-up) recover the circuit better than zero-silencing, for the current-form and the conductance lasso-25 model; one fold per approach for a landscape, five for the zero-silenced baseline

## Why

The paper's "20% hidden" row (Supp. Tab. 4) hides 20% of the non-retina
neurons and zero-silences them: their voltage enters the GNN as 0 at every step
and the loss uses the visible neurons only. It reads `R2_W` 0.52, rollout r
0.88, `R2_Vrest` 0.49 on the readout of the time. The paper also tried an NGP
hash grid to fill the hidden traces; "preliminary results recover part of the
lost W signal but do not help in parameter recovery", and its archived run
(`archive_4/flyvis_noise_005_hidden_020_ngp_blank50_unified_cv00`) has a hidden
trace r of -0.005, i.e. the grid did not recover the hidden voltages at all.

Two things are new here. Recurrent training at horizon 20 judges a hidden
trace by its effect on the visible neurons over 20 steps rather than one, which
is the most plausible way for the visible data to constrain it. And the
conductance lasso-25 model, which experiments 2-5 put level with the current
form on fully observed data (fixed readout, 2026-09-28).

## What the hidden approaches are

All are existing config options (`graph_model.inr_type_hidden` and training
keys); nothing is new code except the warm-up fix below.

| axis value | hidden voltages | keys |
|---|---|---|
| zero-silenced | 0 at every step | none |
| SIREN(x,y,t) | a SIREN of (x, y, t) sampled at each hidden neuron's position: one spatially smooth field | `inr_type_hidden: siren_txy` (defaults: 2,048 wide, 4 layers, omega 4,096) |
| NGP-T | a multi-resolution temporal hash grid, one output per hidden neuron | `inr_type_hidden: ngp_t`, grid of the paper's NGP spec (24 levels, 4 features, base 16, scale 1.4, decoder 512 x 3) |
| + anchors | 3,600 extra grid outputs trained directly on observed neurons' ground truth, weight 3,000 (the paper's NGP recipe) | `train_with_anchor_neurons`, `coeff_anchor_voltage: 3000`, `n_anchor: 3600` |
| + warm-up | zeros for the first 20% of the iterations, then the learned traces, with the GNN's learning rate damped 100x over a V of 5% either side of the switch | `warmup_inject_nnr_iter_frac: 0.2`, `warmup_inject_nnr_ramp_iter_frac: 0.05` |

The learned traces are trained only through the visible neurons' loss (and
the anchors'), at `lr_NNR_f` 0.0005 (the paper's NGP value). Anchors are not
available for SIREN(x,y,t): the code supports them only for the per-neuron
outputs of `siren_t` and `ngp_t`.

## What else differs

Every spec is the paper's zero-silenced spec for its fold with only these
changes, verified by parsing both:

- **conductance lasso 25**: `signal_model_name: flyvis_conductance`,
  `input_size: 6`, `coeff_g_phi_input_group_L1: 25` (as experiments 2-7);
- **recurrent 20**: `recurrent_training: true`, `n_epochs: 20`,
  `data_augmentation_loop: 50`, `rollout_horizon_schedule: 1..20`, experiment
  3's recipe. The one-step arms keep the paper's `n_epochs: 1`,
  `data_augmentation_loop: 500`, so the two trainings do not see the same
  number of iterations (the recurrent trainer also divides each epoch's
  iterations by its horizon);
- the hidden-approach keys above.

Hyperparameters of the GNN itself are the paper's zero-silenced ones in every
arm; the paper's NGP spec had also changed learning rates, batch size and
regularisation, which would confound the approach with the tuning.

## The warm-up was once per epoch

The trainer rebuilds the injection schedule at every epoch with a per-epoch
iteration count, so with `n_epochs: 20` the learned traces would have been
switched back to zeros at the start of every epoch. Fixed 2026-09-29
(`init_hidden_injection_schedule(..., first_epoch=...)`,
`tests/test_hidden_injection_warmup.py`): the warm-up happens in the first
epoch only. No earlier spec set a warm-up, so nothing earlier is affected.

## Also not usable

`coeff_hidden_voltage` would supervise the hidden neurons with their ground
truth (an oracle) and its code sits in a dead loop, so it has no effect; it is
not set here.

## Specs

44 runs in `experiments/specs/exp09/fly/`, named
`flyvis_noise_005_hid20_<hidden>_<model><training>_cv0N`: the zero-silenced
baseline on 5 folds x 4 (model, training) = 20, and the 6 learned-trace
variants on fold cv00 x 4 = 24. Queue `gpu_rtx6000` for all, wall 48 h
(experiment 3's recurrent runs took 22-34 h there).

<!-- STATUS:BEGIN -->

## Status

**0/44 landed**, 0 trained (awaiting `-o test_plot`), 29 running, 15 pending

### Landed --- held-out, `results/metrics.txt`

| arm | hidden | n | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster | hidden trace r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | | |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | hidden | iter | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster | hidden trace r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cur1s | none | 48,001 |  | 0.808 ± 0.007 |  |  | 0.607 ± 0.006 | 0.919 ± 0.035 | 0.638 ± 0.022 | 0.436 ± 0.055 | 0.836 ± 0.031 |  |  |  |  |
| cur1s_inr | sirentxy | 16,001 |  | 0.810 ± 0.000 |  |  | 0.624 ± 0.000 | 0.914 ± 0.000 | 0.558 ± 0.000 | 0.466 ± 0.000 | 0.839 ± 0.000 |  |  |  |  |
| cur1s_inr | sirentxywu | 48,001 |  | 0.814 ± 0.000 |  |  | 0.601 ± 0.000 | 0.877 ± 0.000 | 0.616 ± 0.000 | 0.383 ± 0.000 | 0.825 ± 0.000 |  |  |  |  |
| cur1s_inr | ngpt | 16,001 |  | 0.812 ± 0.000 |  |  | 0.621 ± 0.000 | 0.840 ± 0.000 | 0.488 ± 0.000 | 0.290 ± 0.000 | 0.848 ± 0.000 |  |  |  |  |
| cur1s_inr | ngptanc | 16,001 |  | 0.812 ± 0.000 |  |  | 0.627 ± 0.000 | 0.841 ± 0.000 | 0.523 ± 0.000 | 0.343 ± 0.000 | 0.838 ± 0.000 |  |  |  |  |
| cur1s_inr | ngptwu | 32,001 |  | 0.812 ± 0.000 |  |  | 0.597 ± 0.000 | 0.918 ± 0.000 | 0.647 ± 0.000 | 0.420 ± 0.000 | 0.821 ± 0.000 |  |  |  |  |
| cur1s_inr | ngptancwu | 32,001 |  | 0.808 ± 0.000 |  |  | 0.623 ± 0.000 | 0.936 ± 0.000 | 0.543 ± 0.000 | 0.350 ± 0.000 | 0.815 ± 0.000 |  |  |  |  |
| currc20 | none | 1 |  | 0.003 ± 0.002 |  |  | -0.020 ± 0.006 | -9.552 ± 1.670 | 0.233 ± 0.166 | 0.180 ± 0.163 | -0.017 ± 0.033 |  |  |  |  |
| currc20_inr | sirentxy | 1 |  | -0.000 ± 0.000 |  |  |  |  |  |  |  |  |  |  |  |
| currc20_inr | sirentxywu | 1 |  | -0.000 ± 0.000 |  |  |  |  |  |  |  |  |  |  |  |
| condl251s | none | 32,001 |  | 0.800 ± 0.010 |  |  | 0.408 ± 0.096 | 0.895 ± 0.060 | 0.671 ± 0.031 | 0.619 ± 0.043 | 0.741 ± 0.074 |  |  |  |  |
| condl251s_inr | sirentxy | 16,001 |  | 0.798 ± 0.000 |  |  | 0.340 ± 0.000 | 0.705 ± 0.000 | 0.651 ± 0.000 | 0.648 ± 0.000 | 0.682 ± 0.000 |  |  |  |  |
| condl251s_inr | sirentxywu | 32,001 |  | 0.803 ± 0.000 |  |  | 0.307 ± 0.000 | 0.779 ± 0.000 | 0.651 ± 0.000 | 0.600 ± 0.000 | 0.699 ± 0.000 |  |  |  |  |
| condl251s_inr | ngpt | 16,001 |  | 0.809 ± 0.000 |  |  | 0.338 ± 0.000 | 0.654 ± 0.000 | 0.542 ± 0.000 | 0.571 ± 0.000 | 0.750 ± 0.000 |  |  |  |  |
| condl251s_inr | ngptanc | 16,001 |  | 0.813 ± 0.000 |  |  | 0.452 ± 0.000 | 0.562 ± 0.000 | 0.588 ± 0.000 | 0.529 ± 0.000 | 0.774 ± 0.000 |  |  |  |  |
| condl251s_inr | ngptwu | 32,001 |  | 0.804 ± 0.000 |  |  | 0.348 ± 0.000 | 0.846 ± 0.000 | 0.686 ± 0.000 | 0.640 ± 0.000 | 0.697 ± 0.000 |  |  |  |  |
| condl251s_inr | ngptancwu | 16,001 |  | 0.799 ± 0.000 |  |  | 0.323 ± 0.000 | 0.786 ± 0.000 | 0.677 ± 0.000 | 0.586 ± 0.000 | 0.659 ± 0.000 |  |  |  |  |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_noise_005_hid20_none_cur1s_cv00` | running | 48,001 | `` |  |
| `flyvis_noise_005_hid20_none_cur1s_cv01` | running | 48,001 | `` |  |
| `flyvis_noise_005_hid20_none_cur1s_cv02` | running | 48,001 | `` |  |
| `flyvis_noise_005_hid20_none_cur1s_cv03` | running | 48,001 | `` |  |
| `flyvis_noise_005_hid20_none_cur1s_cv04` | running | 48,001 | `` |  |
| `flyvis_noise_005_hid20_sirentxy_cur1s_cv00` | running | 16,001 | `` |  |
| `flyvis_noise_005_hid20_sirentxywu_cur1s_cv00` | running | 48,001 | `` |  |
| `flyvis_noise_005_hid20_ngpt_cur1s_cv00` | running | 16,001 | `` |  |
| `flyvis_noise_005_hid20_ngptanc_cur1s_cv00` | running | 16,001 | `` |  |
| `flyvis_noise_005_hid20_ngptwu_cur1s_cv00` | running | 32,001 | `` |  |
| `flyvis_noise_005_hid20_ngptancwu_cur1s_cv00` | running | 32,001 | `` |  |
| `flyvis_noise_005_hid20_none_currc20_cv00` | running | 1 | `` |  |
| `flyvis_noise_005_hid20_none_currc20_cv01` | running | 1 | `` |  |
| `flyvis_noise_005_hid20_none_currc20_cv02` | running | 1 | `` |  |
| `flyvis_noise_005_hid20_none_currc20_cv03` | running | 1 | `` |  |
| `flyvis_noise_005_hid20_none_currc20_cv04` | running | 1 | `` |  |
| `flyvis_noise_005_hid20_sirentxy_currc20_cv00` | running | 1 | `` |  |
| `flyvis_noise_005_hid20_sirentxywu_currc20_cv00` | running | 1 | `` |  |
| `flyvis_noise_005_hid20_ngpt_currc20_cv00` | pending |  | `` |  |
| `flyvis_noise_005_hid20_ngptanc_currc20_cv00` | pending |  | `` |  |
| `flyvis_noise_005_hid20_ngptwu_currc20_cv00` | pending |  | `` |  |
| `flyvis_noise_005_hid20_ngptancwu_currc20_cv00` | pending |  | `` |  |
| `flyvis_noise_005_hid20_none_condl251s_cv00` | running | 48,001 | `` |  |
| `flyvis_noise_005_hid20_none_condl251s_cv01` | running | 48,001 | `` |  |
| `flyvis_noise_005_hid20_none_condl251s_cv02` | running | 48,001 | `` |  |
| `flyvis_noise_005_hid20_none_condl251s_cv03` | running | 48,001 | `` |  |
| `flyvis_noise_005_hid20_none_condl251s_cv04` | running | 48,001 | `` |  |
| `flyvis_noise_005_hid20_sirentxy_condl251s_cv00` | running | 16,001 | `` |  |
| `flyvis_noise_005_hid20_sirentxywu_condl251s_cv00` | running | 48,001 | `` |  |
| `flyvis_noise_005_hid20_ngpt_condl251s_cv00` | running | 16,001 | `` |  |
| `flyvis_noise_005_hid20_ngptanc_condl251s_cv00` | running | 16,001 | `` |  |
| `flyvis_noise_005_hid20_ngptwu_condl251s_cv00` | running | 32,001 | `` |  |
| `flyvis_noise_005_hid20_ngptancwu_condl251s_cv00` | running | 16,001 | `` |  |
| `flyvis_noise_005_hid20_none_condl25rc20_cv00` | pending |  | `` |  |
| `flyvis_noise_005_hid20_none_condl25rc20_cv01` | pending |  | `` |  |
| `flyvis_noise_005_hid20_none_condl25rc20_cv02` | pending |  | `` |  |
| `flyvis_noise_005_hid20_none_condl25rc20_cv03` | pending |  | `` |  |
| `flyvis_noise_005_hid20_none_condl25rc20_cv04` | pending |  | `` |  |
| `flyvis_noise_005_hid20_sirentxy_condl25rc20_cv00` | pending |  | `` |  |
| `flyvis_noise_005_hid20_sirentxywu_condl25rc20_cv00` | pending |  | `` |  |
| `flyvis_noise_005_hid20_ngpt_condl25rc20_cv00` | pending |  | `` |  |
| `flyvis_noise_005_hid20_ngptanc_condl25rc20_cv00` | pending |  | `` |  |
| `flyvis_noise_005_hid20_ngptwu_condl25rc20_cv00` | pending |  | `` |  |
| `flyvis_noise_005_hid20_ngptancwu_condl25rc20_cv00` | pending |  | `` |  |

<!-- STATUS:END -->
