---
number: 1
name: derivative_target
title: Recovery errors, before and after the derivative-target fix
purpose: regenerate slide 5 and 6 of presentation/conductance.pdf with sound training
  (one parameter changes only) -- it was not -- and look at these plots again
baseline: experiments/baseline/gnn_current_baseline.yaml
specs_dir: experiments/specs/fly
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
- id: nominal
  label: finite difference of the observed voltage
  differs_by:
    training.derivative_target: observed_fd
  spec_pattern: flyvis_{noise}_blank50_dtfd_{fold}
- id: bug
  label: the generator's stored y_list
  differs_by:
    training.derivative_target: y_list
  spec_pattern: flyvis_{noise}_blank50_dtbug_{fold}
job_ids:
  flyvis_noise_free_blank50_dtbug_cv00: '154396059'
  flyvis_noise_free_blank50_dtbug_cv01: '154396060'
  flyvis_noise_free_blank50_dtbug_cv02: '154396061'
  flyvis_noise_free_blank50_dtbug_cv03: '154396062'
  flyvis_noise_free_blank50_dtbug_cv04: '154396063'
  flyvis_noise_005_blank50_dtbug_cv00: '154396064'
  flyvis_noise_005_blank50_dtbug_cv01: '154396065'
  flyvis_noise_005_blank50_dtbug_cv02: '154396066'
  flyvis_noise_005_blank50_dtbug_cv03: '154396067'
  flyvis_noise_005_blank50_dtbug_cv04: '154396068'
  flyvis_noise_05_blank50_dtbug_cv00: '154396069'
  flyvis_noise_05_blank50_dtbug_cv01: '154396070'
  flyvis_noise_05_blank50_dtbug_cv02: '154396071'
  flyvis_noise_05_blank50_dtbug_cv03: '154396072'
  flyvis_noise_05_blank50_dtbug_cv04: '154396073'
---

# Experiment 1 — derivative_target

**Recovery errors, before and after the derivative-target fix**

**Purpose.** regenerate slide 5 and 6 of presentation/conductance.pdf with sound training (one parameter changes only) -- it was not -- and look at these plots again

## What differs

One key, `training.derivative_target`. `observed_fd` is the finite difference
of the observed voltage, `f(v) + xi/dt`; `y_list` is the generator's own stored
derivative, `f(v)`. The integrator adds the process noise `xi` to the state
*after* that right-hand side was evaluated, so the two targets differ by
`xi/dt` — exactly zero at `noise_free` and growing with `noise_model_level`.

`training.deterministic` is **on** in all thirty, so a rerun at the same seed
reproduces bitwise; `scatter_add` on CUDA accumulates via atomics otherwise
and eight identical calls gave eight distinct results.

Everything else comes from the baseline unchanged. Per run the generator axis
sets `dataset`, `simulation.noise_model_level` and the two seeds
(`42 + fold`, `1042 + fold`); nothing else varies.

## Specs

30 = 2 arms x 3 model-noise levels x 5 folds, all under `experiments/specs/fly/`.

| arm | noise | fold | spec | dataset | noise_model_level | derivative_target |
|---|---|---|---|---|---|---|
| nominal | noise_free | cv00 | `flyvis_noise_free_blank50_dtfd_cv00` | `flyvis_noise_free_blank50_cv00` | 0.0 | `observed_fd` |
| nominal | noise_free | cv01 | `flyvis_noise_free_blank50_dtfd_cv01` | `flyvis_noise_free_blank50_cv01` | 0.0 | `observed_fd` |
| nominal | noise_free | cv02 | `flyvis_noise_free_blank50_dtfd_cv02` | `flyvis_noise_free_blank50_cv02` | 0.0 | `observed_fd` |
| nominal | noise_free | cv03 | `flyvis_noise_free_blank50_dtfd_cv03` | `flyvis_noise_free_blank50_cv03` | 0.0 | `observed_fd` |
| nominal | noise_free | cv04 | `flyvis_noise_free_blank50_dtfd_cv04` | `flyvis_noise_free_blank50_cv04` | 0.0 | `observed_fd` |
| nominal | noise_005 | cv00 | `flyvis_noise_005_blank50_dtfd_cv00` | `flyvis_noise_005_blank50_cv00` | 0.05 | `observed_fd` |
| nominal | noise_005 | cv01 | `flyvis_noise_005_blank50_dtfd_cv01` | `flyvis_noise_005_blank50_cv01` | 0.05 | `observed_fd` |
| nominal | noise_005 | cv02 | `flyvis_noise_005_blank50_dtfd_cv02` | `flyvis_noise_005_blank50_cv02` | 0.05 | `observed_fd` |
| nominal | noise_005 | cv03 | `flyvis_noise_005_blank50_dtfd_cv03` | `flyvis_noise_005_blank50_cv03` | 0.05 | `observed_fd` |
| nominal | noise_005 | cv04 | `flyvis_noise_005_blank50_dtfd_cv04` | `flyvis_noise_005_blank50_cv04` | 0.05 | `observed_fd` |
| nominal | noise_05 | cv00 | `flyvis_noise_05_blank50_dtfd_cv00` | `flyvis_noise_05_blank50_cv00` | 0.5 | `observed_fd` |
| nominal | noise_05 | cv01 | `flyvis_noise_05_blank50_dtfd_cv01` | `flyvis_noise_05_blank50_cv01` | 0.5 | `observed_fd` |
| nominal | noise_05 | cv02 | `flyvis_noise_05_blank50_dtfd_cv02` | `flyvis_noise_05_blank50_cv02` | 0.5 | `observed_fd` |
| nominal | noise_05 | cv03 | `flyvis_noise_05_blank50_dtfd_cv03` | `flyvis_noise_05_blank50_cv03` | 0.5 | `observed_fd` |
| nominal | noise_05 | cv04 | `flyvis_noise_05_blank50_dtfd_cv04` | `flyvis_noise_05_blank50_cv04` | 0.5 | `observed_fd` |
| bug | noise_free | cv00 | `flyvis_noise_free_blank50_dtbug_cv00` | `flyvis_noise_free_blank50_cv00` | 0.0 | `y_list` |
| bug | noise_free | cv01 | `flyvis_noise_free_blank50_dtbug_cv01` | `flyvis_noise_free_blank50_cv01` | 0.0 | `y_list` |
| bug | noise_free | cv02 | `flyvis_noise_free_blank50_dtbug_cv02` | `flyvis_noise_free_blank50_cv02` | 0.0 | `y_list` |
| bug | noise_free | cv03 | `flyvis_noise_free_blank50_dtbug_cv03` | `flyvis_noise_free_blank50_cv03` | 0.0 | `y_list` |
| bug | noise_free | cv04 | `flyvis_noise_free_blank50_dtbug_cv04` | `flyvis_noise_free_blank50_cv04` | 0.0 | `y_list` |
| bug | noise_005 | cv00 | `flyvis_noise_005_blank50_dtbug_cv00` | `flyvis_noise_005_blank50_cv00` | 0.05 | `y_list` |
| bug | noise_005 | cv01 | `flyvis_noise_005_blank50_dtbug_cv01` | `flyvis_noise_005_blank50_cv01` | 0.05 | `y_list` |
| bug | noise_005 | cv02 | `flyvis_noise_005_blank50_dtbug_cv02` | `flyvis_noise_005_blank50_cv02` | 0.05 | `y_list` |
| bug | noise_005 | cv03 | `flyvis_noise_005_blank50_dtbug_cv03` | `flyvis_noise_005_blank50_cv03` | 0.05 | `y_list` |
| bug | noise_005 | cv04 | `flyvis_noise_005_blank50_dtbug_cv04` | `flyvis_noise_005_blank50_cv04` | 0.05 | `y_list` |
| bug | noise_05 | cv00 | `flyvis_noise_05_blank50_dtbug_cv00` | `flyvis_noise_05_blank50_cv00` | 0.5 | `y_list` |
| bug | noise_05 | cv01 | `flyvis_noise_05_blank50_dtbug_cv01` | `flyvis_noise_05_blank50_cv01` | 0.5 | `y_list` |
| bug | noise_05 | cv02 | `flyvis_noise_05_blank50_dtbug_cv02` | `flyvis_noise_05_blank50_cv02` | 0.5 | `y_list` |
| bug | noise_05 | cv03 | `flyvis_noise_05_blank50_dtbug_cv03` | `flyvis_noise_05_blank50_cv03` | 0.5 | `y_list` |
| bug | noise_05 | cv04 | `flyvis_noise_05_blank50_dtbug_cv04` | `flyvis_noise_05_blank50_cv04` | 0.5 | `y_list` |

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
| nominal | noise_free | 1 |  | 0.003 ± 0.003 |  |  |  |  |  |  |  |  |
| nominal | noise_005 | 1 |  | 0.003 ± 0.003 |  |  |  |  |  |  |  |  |
| nominal | noise_05 | 1 |  | 0.003 ± 0.005 |  |  |  |  |  |  |  |  |

### Per run

| run | status | iter | commit |
|---|---|---|---|
| `flyvis_noise_free_blank50_dtfd_cv00` | running | 1 | `` |
| `flyvis_noise_free_blank50_dtfd_cv01` | running | 1 | `` |
| `flyvis_noise_free_blank50_dtfd_cv02` | running | 1 | `` |
| `flyvis_noise_free_blank50_dtfd_cv03` | running | 1 | `` |
| `flyvis_noise_free_blank50_dtfd_cv04` | running | 1 | `` |
| `flyvis_noise_005_blank50_dtfd_cv00` | running | 1 | `` |
| `flyvis_noise_005_blank50_dtfd_cv01` | running | 1 | `` |
| `flyvis_noise_005_blank50_dtfd_cv02` | running | 1 | `` |
| `flyvis_noise_005_blank50_dtfd_cv03` | running | 1 | `` |
| `flyvis_noise_005_blank50_dtfd_cv04` | running | 1 | `` |
| `flyvis_noise_05_blank50_dtfd_cv00` | running | 1 | `` |
| `flyvis_noise_05_blank50_dtfd_cv01` | running | 1 | `` |
| `flyvis_noise_05_blank50_dtfd_cv02` | running | 1 | `` |
| `flyvis_noise_05_blank50_dtfd_cv03` | running | 1 | `` |
| `flyvis_noise_05_blank50_dtfd_cv04` | running | 1 | `` |
| `flyvis_noise_free_blank50_dtbug_cv00` | pending |  | `` |
| `flyvis_noise_free_blank50_dtbug_cv01` | pending |  | `` |
| `flyvis_noise_free_blank50_dtbug_cv02` | pending |  | `` |
| `flyvis_noise_free_blank50_dtbug_cv03` | pending |  | `` |
| `flyvis_noise_free_blank50_dtbug_cv04` | pending |  | `` |
| `flyvis_noise_005_blank50_dtbug_cv00` | pending |  | `` |
| `flyvis_noise_005_blank50_dtbug_cv01` | pending |  | `` |
| `flyvis_noise_005_blank50_dtbug_cv02` | pending |  | `` |
| `flyvis_noise_005_blank50_dtbug_cv03` | pending |  | `` |
| `flyvis_noise_005_blank50_dtbug_cv04` | pending |  | `` |
| `flyvis_noise_05_blank50_dtbug_cv00` | pending |  | `` |
| `flyvis_noise_05_blank50_dtbug_cv01` | pending |  | `` |
| `flyvis_noise_05_blank50_dtbug_cv02` | pending |  | `` |
| `flyvis_noise_05_blank50_dtbug_cv03` | pending |  | `` |
| `flyvis_noise_05_blank50_dtbug_cv04` | pending |  | `` |

<!-- STATUS:END -->

