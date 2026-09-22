---
number: 2
name: conductance_lasso
title: The general g_phi under a group lasso, against the current form
purpose: does the general form g_phi = MLP(a_i, a_j, v_i, v_j) recover the circuit
  as well as the current form once the group lasso is strong enough to prune its extra
  inputs, across the model-noise sweep; and does that lasso kill the per-edge offset
  C_ij, read as R2_Vrest against R2_Vrest without the C_i correction
baseline: experiments/baseline/gnn_current_baseline.yaml
specs_dir: experiments/specs/exp02/fly
task: train
queue: gpu_rtx6000
wall: '48:00'
axes:
  noise:
  - noise_free
  - noise_005
  - noise_05
  fold:
  - cv00
  - cv01
  - cv02
  - cv03
  - cv04
arms:
- id: current
  label: current form, no lasso (experiment 1)
  spec_pattern: flyvis_{noise}_blank50_dtfd_{fold}
  submit: false
  differs_by: {}
- id: conductance
  label: conductance form, group lasso 100
  spec_pattern: flyvis_{noise}_blank50_condl100_{fold}
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 100.0
job_ids:
  flyvis_noise_free_blank50_condl100_cv00: '154396165'
  flyvis_noise_free_blank50_condl100_cv01: '154396166'
  flyvis_noise_free_blank50_condl100_cv02: '154396167'
  flyvis_noise_free_blank50_condl100_cv03: '154396168'
  flyvis_noise_free_blank50_condl100_cv04: '154396169'
  flyvis_noise_005_blank50_condl100_cv00: '154396170'
  flyvis_noise_005_blank50_condl100_cv01: '154396171'
  flyvis_noise_005_blank50_condl100_cv02: '154396172'
  flyvis_noise_005_blank50_condl100_cv03: '154396173'
  flyvis_noise_005_blank50_condl100_cv04: '154396174'
  flyvis_noise_05_blank50_condl100_cv00: '154396175'
  flyvis_noise_05_blank50_condl100_cv01: '154396176'
  flyvis_noise_05_blank50_condl100_cv02: '154396177'
  flyvis_noise_05_blank50_condl100_cv03: '154396178'
  flyvis_noise_05_blank50_condl100_cv04: '154396179'
---

# Experiment 2 — conductance_lasso

**The general g_phi under a group lasso, against the current form**

**Purpose.** does the general form g_phi = MLP(a_i, a_j, v_i, v_j) recover the circuit as well as the current form once the group lasso is strong enough to prune its extra inputs, across the model-noise sweep; and does that lasso kill the per-edge offset C_ij, read as R2_Vrest against R2_Vrest without the C_i correction

## What differs

Three keys, and they go together:

- `graph_model.signal_model_name: flyvis_conductance` — the edge function
  reads `[v_j, a_j, v_i, a_i]` instead of `[v_j, a_j]`, so the message can
  carry a driving force and, with it, a per-edge offset.
- `graph_model.input_size: 6` — `2 + 2*embedding_dim`, not the current
  form's `1 + embedding_dim = 3`. Leaving it at 3 is a shape error, not a
  weaker model.
- `training.coeff_g_phi_input_group_L1: 100` — an L2 per input block of
  `g_phi`'s first layer, summed as an L1, so a whole input can go to zero.
  The deck swept this 0.25 → 25 on the current form; 100 is past that.

Everything else is the baseline's, including `deterministic`,
`torch_compile` and every other coefficient.

## The two arms

**`current` is not run here.** It is experiment 1's `nominal` arm — the
same fifteen runs, same seeds, same datasets — read from the log tree for
the comparison. It carries `submit: false`, so `exp launch 2` neither
resubmits it nor clears its directories.

**15 new jobs**: 3 model-noise levels x 5 folds, conductance arm only.

## Specs

| arm | noise | fold | spec | submitted |
|---|---|---|---|---|
| current | noise_free | cv00 | `flyvis_noise_free_blank50_dtfd_cv00` | no (experiment 1) |
| current | noise_free | cv01 | `flyvis_noise_free_blank50_dtfd_cv01` | no (experiment 1) |
| current | noise_free | cv02 | `flyvis_noise_free_blank50_dtfd_cv02` | no (experiment 1) |
| current | noise_free | cv03 | `flyvis_noise_free_blank50_dtfd_cv03` | no (experiment 1) |
| current | noise_free | cv04 | `flyvis_noise_free_blank50_dtfd_cv04` | no (experiment 1) |
| current | noise_005 | cv00 | `flyvis_noise_005_blank50_dtfd_cv00` | no (experiment 1) |
| current | noise_005 | cv01 | `flyvis_noise_005_blank50_dtfd_cv01` | no (experiment 1) |
| current | noise_005 | cv02 | `flyvis_noise_005_blank50_dtfd_cv02` | no (experiment 1) |
| current | noise_005 | cv03 | `flyvis_noise_005_blank50_dtfd_cv03` | no (experiment 1) |
| current | noise_005 | cv04 | `flyvis_noise_005_blank50_dtfd_cv04` | no (experiment 1) |
| current | noise_05 | cv00 | `flyvis_noise_05_blank50_dtfd_cv00` | no (experiment 1) |
| current | noise_05 | cv01 | `flyvis_noise_05_blank50_dtfd_cv01` | no (experiment 1) |
| current | noise_05 | cv02 | `flyvis_noise_05_blank50_dtfd_cv02` | no (experiment 1) |
| current | noise_05 | cv03 | `flyvis_noise_05_blank50_dtfd_cv03` | no (experiment 1) |
| current | noise_05 | cv04 | `flyvis_noise_05_blank50_dtfd_cv04` | no (experiment 1) |
| conductance | noise_free | cv00 | `flyvis_noise_free_blank50_condl100_cv00` | yes |
| conductance | noise_free | cv01 | `flyvis_noise_free_blank50_condl100_cv01` | yes |
| conductance | noise_free | cv02 | `flyvis_noise_free_blank50_condl100_cv02` | yes |
| conductance | noise_free | cv03 | `flyvis_noise_free_blank50_condl100_cv03` | yes |
| conductance | noise_free | cv04 | `flyvis_noise_free_blank50_condl100_cv04` | yes |
| conductance | noise_005 | cv00 | `flyvis_noise_005_blank50_condl100_cv00` | yes |
| conductance | noise_005 | cv01 | `flyvis_noise_005_blank50_condl100_cv01` | yes |
| conductance | noise_005 | cv02 | `flyvis_noise_005_blank50_condl100_cv02` | yes |
| conductance | noise_005 | cv03 | `flyvis_noise_005_blank50_condl100_cv03` | yes |
| conductance | noise_005 | cv04 | `flyvis_noise_005_blank50_condl100_cv04` | yes |
| conductance | noise_05 | cv00 | `flyvis_noise_05_blank50_condl100_cv00` | yes |
| conductance | noise_05 | cv01 | `flyvis_noise_05_blank50_condl100_cv01` | yes |
| conductance | noise_05 | cv02 | `flyvis_noise_05_blank50_condl100_cv02` | yes |
| conductance | noise_05 | cv03 | `flyvis_noise_05_blank50_condl100_cv03` | yes |
| conductance | noise_05 | cv04 | `flyvis_noise_05_blank50_condl100_cv04` | yes |

<!-- STATUS:BEGIN -->

## Status

**0/30 landed**, 15 running, 15 pending

### Landed --- held-out, `results/metrics.txt`

| arm | noise | n | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | noise | iter | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | noise_free | 32,001 |  | 0.657 ± 0.415 | 0.796 ± 0.027 | 0.813 ± 0.061 | 0.671 ± 0.056 | 0.585 ± 0.029 | 0.909 ± 0.032 |  |  |  |
| current | noise_005 | 32,001 |  | 0.997 ± 0.001 | 0.930 ± 0.007 | 0.956 ± 0.018 | 0.737 ± 0.047 | 0.598 ± 0.043 | 0.958 ± 0.010 |  |  |  |
| current | noise_05 | 32,001 |  | 0.989 ± 0.003 | 0.977 ± 0.001 | 0.986 ± 0.001 | 0.758 ± 0.034 | 0.819 ± 0.045 | 0.984 ± 0.001 |  |  |  |

### Per run

| run | status | iter | commit |
|---|---|---|---|
| `flyvis_noise_free_blank50_dtfd_cv00` | running | 32,001 | `` |
| `flyvis_noise_free_blank50_dtfd_cv01` | running | 32,001 | `` |
| `flyvis_noise_free_blank50_dtfd_cv02` | running | 32,001 | `` |
| `flyvis_noise_free_blank50_dtfd_cv03` | running | 32,001 | `` |
| `flyvis_noise_free_blank50_dtfd_cv04` | running | 32,001 | `` |
| `flyvis_noise_005_blank50_dtfd_cv00` | running | 32,001 | `` |
| `flyvis_noise_005_blank50_dtfd_cv01` | running | 32,001 | `` |
| `flyvis_noise_005_blank50_dtfd_cv02` | running | 32,001 | `` |
| `flyvis_noise_005_blank50_dtfd_cv03` | running | 32,001 | `` |
| `flyvis_noise_005_blank50_dtfd_cv04` | running | 32,001 | `` |
| `flyvis_noise_05_blank50_dtfd_cv00` | running | 32,001 | `` |
| `flyvis_noise_05_blank50_dtfd_cv01` | running | 32,001 | `` |
| `flyvis_noise_05_blank50_dtfd_cv02` | running | 32,001 | `` |
| `flyvis_noise_05_blank50_dtfd_cv03` | running | 32,001 | `` |
| `flyvis_noise_05_blank50_dtfd_cv04` | running | 32,001 | `` |
| `flyvis_noise_free_blank50_condl100_cv00` | pending |  | `` |
| `flyvis_noise_free_blank50_condl100_cv01` | pending |  | `` |
| `flyvis_noise_free_blank50_condl100_cv02` | pending |  | `` |
| `flyvis_noise_free_blank50_condl100_cv03` | pending |  | `` |
| `flyvis_noise_free_blank50_condl100_cv04` | pending |  | `` |
| `flyvis_noise_005_blank50_condl100_cv00` | pending |  | `` |
| `flyvis_noise_005_blank50_condl100_cv01` | pending |  | `` |
| `flyvis_noise_005_blank50_condl100_cv02` | pending |  | `` |
| `flyvis_noise_005_blank50_condl100_cv03` | pending |  | `` |
| `flyvis_noise_005_blank50_condl100_cv04` | pending |  | `` |
| `flyvis_noise_05_blank50_condl100_cv00` | pending |  | `` |
| `flyvis_noise_05_blank50_condl100_cv01` | pending |  | `` |
| `flyvis_noise_05_blank50_condl100_cv02` | pending |  | `` |
| `flyvis_noise_05_blank50_condl100_cv03` | pending |  | `` |
| `flyvis_noise_05_blank50_condl100_cv04` | pending |  | `` |

<!-- STATUS:END -->

