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
- id: cond_l25
  label: conductance, group lasso 25 (noise_free probe)
  spec_pattern: flyvis_{noise}_blank50_condl25_{fold}
  axes_only:
    noise:
    - noise_free
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
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
  flyvis_noise_free_blank50_condl25_cv00: '154396309'
  flyvis_noise_free_blank50_condl25_cv01: '154396310'
  flyvis_noise_free_blank50_condl25_cv02: '154396311'
  flyvis_noise_free_blank50_condl25_cv03: '154396312'
  flyvis_noise_free_blank50_condl25_cv04: '154396313'
analyse_job_ids:
  flyvis_noise_free_blank50_dtfd_cv00: '154399734'
  flyvis_noise_free_blank50_dtfd_cv01: '154399735'
  flyvis_noise_free_blank50_dtfd_cv02: '154399738'
  flyvis_noise_free_blank50_dtfd_cv03: '154399740'
  flyvis_noise_free_blank50_dtfd_cv04: '154399741'
  flyvis_noise_005_blank50_dtfd_cv00: '154399742'
  flyvis_noise_005_blank50_dtfd_cv01: '154399743'
  flyvis_noise_005_blank50_dtfd_cv02: '154399744'
  flyvis_noise_005_blank50_dtfd_cv03: '154399745'
  flyvis_noise_005_blank50_dtfd_cv04: '154399746'
  flyvis_noise_05_blank50_dtfd_cv00: '154399747'
  flyvis_noise_05_blank50_dtfd_cv01: '154399748'
  flyvis_noise_05_blank50_dtfd_cv02: '154399749'
  flyvis_noise_05_blank50_dtfd_cv03: '154399750'
  flyvis_noise_05_blank50_dtfd_cv04: '154399751'
  flyvis_noise_free_blank50_condl100_cv00: '154399752'
  flyvis_noise_free_blank50_condl100_cv01: '154399753'
  flyvis_noise_free_blank50_condl100_cv02: '154399754'
  flyvis_noise_free_blank50_condl100_cv03: '154399755'
  flyvis_noise_free_blank50_condl100_cv04: '154399756'
  flyvis_noise_005_blank50_condl100_cv00: '154399757'
  flyvis_noise_005_blank50_condl100_cv01: '154399758'
  flyvis_noise_005_blank50_condl100_cv02: '154399759'
  flyvis_noise_005_blank50_condl100_cv03: '154399760'
  flyvis_noise_005_blank50_condl100_cv04: '154399761'
  flyvis_noise_05_blank50_condl100_cv00: '154399762'
  flyvis_noise_05_blank50_condl100_cv01: '154399763'
  flyvis_noise_05_blank50_condl100_cv02: '154399764'
  flyvis_noise_05_blank50_condl100_cv03: '154399765'
  flyvis_noise_05_blank50_condl100_cv04: '154399766'
  flyvis_noise_free_blank50_condl25_cv00: '154399767'
  flyvis_noise_free_blank50_condl25_cv01: '154399768'
  flyvis_noise_free_blank50_condl25_cv02: '154399769'
  flyvis_noise_free_blank50_condl25_cv03: '154399770'
  flyvis_noise_free_blank50_condl25_cv04: '154399771'
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

## The lambda-25 probe at noise_free

At `noise_free` the lasso-100 arm split two ways at iteration 80,001: cv01,
cv02 and cv04 recovered normally while **cv00 and cv03 collapsed** —
`R2_W` −0.019 and −0.018, which is the value `R2` takes when the weights have
gone to zero, with `R2_msg` negative, so the message is anti-correlated with
the truth.

| fold | R2_W | R2_tau | R2_msg |
|---|---|---|---|
| cv00 | **-0.019** | **-5451** | **-0.862** |
| cv01 | 0.865 | 0.927 | 0.912 |
| cv02 | 0.776 | 0.913 | 0.837 |
| cv03 | **-0.018** | **-18.4** | **-0.455** |
| cv04 | 0.779 | 0.898 | 0.877 |

Two candidates, and they are separable: lambda 100 is four times past the end of
slide 21's ladder, which stopped at 25 and was still improving; and at sigma 0
there is no process noise to keep the message alive against the penalty. The
`cond_l25` arm is five folds at lambda 25 on the same `noise_free` data, and
nothing else changed. If it recovers on all five, the lasso is too strong; if it
splits the same way, sigma 0 is.

<!-- STATUS:BEGIN -->

## Status

**0/35 landed**, 35 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | noise | n | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | noise | iter | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | |

### Per run

| run | status | iter | commit |
|---|---|---|---|
| `flyvis_noise_free_blank50_dtfd_cv00` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_free_blank50_dtfd_cv01` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_free_blank50_dtfd_cv02` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_free_blank50_dtfd_cv03` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_free_blank50_dtfd_cv04` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtfd_cv00` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtfd_cv01` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtfd_cv02` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtfd_cv03` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtfd_cv04` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtfd_cv00` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtfd_cv01` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtfd_cv02` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtfd_cv03` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtfd_cv04` | trained | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_free_blank50_condl100_cv00` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_free_blank50_condl100_cv01` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_free_blank50_condl100_cv02` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_free_blank50_condl100_cv03` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_free_blank50_condl100_cv04` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_005_blank50_condl100_cv00` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_005_blank50_condl100_cv01` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_005_blank50_condl100_cv02` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_005_blank50_condl100_cv03` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_005_blank50_condl100_cv04` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_05_blank50_condl100_cv00` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_05_blank50_condl100_cv01` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_05_blank50_condl100_cv02` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_05_blank50_condl100_cv03` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_05_blank50_condl100_cv04` | trained | 1,520,001 | `950fc40dfda9` |
| `flyvis_noise_free_blank50_condl25_cv00` | trained | 1,520,001 | `71e4d78710c4` |
| `flyvis_noise_free_blank50_condl25_cv01` | trained | 1,520,001 | `71e4d78710c4` |
| `flyvis_noise_free_blank50_condl25_cv02` | trained | 1,520,001 | `71e4d78710c4` |
| `flyvis_noise_free_blank50_condl25_cv03` | trained | 1,520,001 | `71e4d78710c4` |
| `flyvis_noise_free_blank50_condl25_cv04` | trained | 1,520,001 | `71e4d78710c4` |

<!-- STATUS:END -->

