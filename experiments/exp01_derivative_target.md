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
  flyvis_noise_free_blank50_dtfd_cv00: '154396091'
  flyvis_noise_free_blank50_dtfd_cv01: '154396092'
  flyvis_noise_free_blank50_dtfd_cv02: '154396093'
  flyvis_noise_free_blank50_dtfd_cv03: '154396094'
  flyvis_noise_free_blank50_dtfd_cv04: '154396095'
  flyvis_noise_005_blank50_dtfd_cv00: '154396096'
  flyvis_noise_005_blank50_dtfd_cv01: '154396097'
  flyvis_noise_005_blank50_dtfd_cv02: '154396098'
  flyvis_noise_005_blank50_dtfd_cv03: '154396099'
  flyvis_noise_005_blank50_dtfd_cv04: '154396100'
  flyvis_noise_05_blank50_dtfd_cv00: '154396101'
  flyvis_noise_05_blank50_dtfd_cv01: '154396102'
  flyvis_noise_05_blank50_dtfd_cv02: '154396103'
  flyvis_noise_05_blank50_dtfd_cv03: '154396104'
  flyvis_noise_05_blank50_dtfd_cv04: '154396105'
  flyvis_noise_free_blank50_dtbug_cv00: '154396106'
  flyvis_noise_free_blank50_dtbug_cv01: '154396107'
  flyvis_noise_free_blank50_dtbug_cv02: '154396108'
  flyvis_noise_free_blank50_dtbug_cv03: '154396109'
  flyvis_noise_free_blank50_dtbug_cv04: '154396110'
  flyvis_noise_005_blank50_dtbug_cv00: '154396111'
  flyvis_noise_005_blank50_dtbug_cv01: '154396112'
  flyvis_noise_005_blank50_dtbug_cv02: '154396113'
  flyvis_noise_005_blank50_dtbug_cv03: '154396114'
  flyvis_noise_005_blank50_dtbug_cv04: '154396115'
  flyvis_noise_05_blank50_dtbug_cv00: '154396116'
  flyvis_noise_05_blank50_dtbug_cv01: '154396117'
  flyvis_noise_05_blank50_dtbug_cv02: '154396118'
  flyvis_noise_05_blank50_dtbug_cv03: '154396119'
  flyvis_noise_05_blank50_dtbug_cv04: '154396120'
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

**0/30 landed**, 30 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | noise | n | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | noise | iter | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | noise_free | 80,001 |  | 0.830 ± 0.335 | 0.845 ± 0.023 | 0.839 ± 0.032 | 0.708 ± 0.047 | 0.575 ± 0.042 | 0.925 ± 0.023 |  |  | 0.865 ± 0.018 |
| nominal | noise_005 | 80,001 |  | 0.997 ± 0.001 | 0.944 ± 0.006 | 0.972 ± 0.008 | 0.785 ± 0.043 | 0.644 ± 0.051 | 0.968 ± 0.006 |  |  | 0.886 ± 0.023 |
| nominal | noise_05 | 80,001 |  | 0.823 ± 0.336 | 0.981 ± 0.001 | 0.992 ± 0.001 | 0.837 ± 0.021 | 0.862 ± 0.035 | 0.989 ± 0.001 |  |  | 0.891 ± 0.015 |
| bug | noise_free | 80,001 |  | 0.830 ± 0.335 | 0.850 ± 0.027 | 0.844 ± 0.033 | 0.725 ± 0.044 | 0.645 ± 0.056 | 0.921 ± 0.036 |  |  | 0.847 ± 0.013 |
| bug | noise_005 | 80,001 |  | 0.999 ± 0.000 | 0.970 ± 0.012 | 0.981 ± 0.007 | 0.918 ± 0.021 | 0.859 ± 0.050 | 0.985 ± 0.008 |  |  | 0.908 ± 0.012 |
| bug | noise_05 | 80,001 |  | 0.999 ± 0.000 | 0.998 ± 0.000 | 1.000 ± 0.000 | 0.984 ± 0.008 | 0.989 ± 0.007 | 1.000 ± 0.000 |  |  | 0.921 ± 0.006 |

### Per run

| run | status | iter | commit |
|---|---|---|---|
| `flyvis_noise_free_blank50_dtfd_cv00` | running | 80,001 | `` |
| `flyvis_noise_free_blank50_dtfd_cv01` | running | 80,001 | `` |
| `flyvis_noise_free_blank50_dtfd_cv02` | running | 80,001 | `` |
| `flyvis_noise_free_blank50_dtfd_cv03` | running | 80,001 | `` |
| `flyvis_noise_free_blank50_dtfd_cv04` | running | 80,001 | `` |
| `flyvis_noise_005_blank50_dtfd_cv00` | running | 80,001 | `` |
| `flyvis_noise_005_blank50_dtfd_cv01` | running | 80,001 | `` |
| `flyvis_noise_005_blank50_dtfd_cv02` | running | 80,001 | `` |
| `flyvis_noise_005_blank50_dtfd_cv03` | running | 80,001 | `` |
| `flyvis_noise_005_blank50_dtfd_cv04` | running | 80,001 | `` |
| `flyvis_noise_05_blank50_dtfd_cv00` | running | 80,001 | `` |
| `flyvis_noise_05_blank50_dtfd_cv01` | running | 80,001 | `` |
| `flyvis_noise_05_blank50_dtfd_cv02` | running | 80,001 | `` |
| `flyvis_noise_05_blank50_dtfd_cv03` | running | 80,001 | `` |
| `flyvis_noise_05_blank50_dtfd_cv04` | running | 80,001 | `` |
| `flyvis_noise_free_blank50_dtbug_cv00` | running | 80,001 | `` |
| `flyvis_noise_free_blank50_dtbug_cv01` | running | 80,001 | `` |
| `flyvis_noise_free_blank50_dtbug_cv02` | running | 80,001 | `` |
| `flyvis_noise_free_blank50_dtbug_cv03` | running | 80,001 | `` |
| `flyvis_noise_free_blank50_dtbug_cv04` | running | 80,001 | `` |
| `flyvis_noise_005_blank50_dtbug_cv00` | running | 80,001 | `` |
| `flyvis_noise_005_blank50_dtbug_cv01` | running | 80,001 | `` |
| `flyvis_noise_005_blank50_dtbug_cv02` | running | 80,001 | `` |
| `flyvis_noise_005_blank50_dtbug_cv03` | running | 80,001 | `` |
| `flyvis_noise_005_blank50_dtbug_cv04` | running | 80,001 | `` |
| `flyvis_noise_05_blank50_dtbug_cv00` | running | 80,001 | `` |
| `flyvis_noise_05_blank50_dtbug_cv01` | running | 80,001 | `` |
| `flyvis_noise_05_blank50_dtbug_cv02` | running | 80,001 | `` |
| `flyvis_noise_05_blank50_dtbug_cv03` | running | 80,001 | `` |
| `flyvis_noise_05_blank50_dtbug_cv04` | running | 80,001 | `` |

<!-- STATUS:END -->

