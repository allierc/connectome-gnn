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
analyse_job_ids:
  flyvis_noise_free_blank50_dtfd_cv00: '154399703'
  flyvis_noise_free_blank50_dtfd_cv01: '154399704'
  flyvis_noise_free_blank50_dtfd_cv02: '154399705'
  flyvis_noise_free_blank50_dtfd_cv03: '154399706'
  flyvis_noise_free_blank50_dtfd_cv04: '154399707'
  flyvis_noise_005_blank50_dtfd_cv00: '154399708'
  flyvis_noise_005_blank50_dtfd_cv01: '154399709'
  flyvis_noise_005_blank50_dtfd_cv02: '154399710'
  flyvis_noise_005_blank50_dtfd_cv03: '154399711'
  flyvis_noise_005_blank50_dtfd_cv04: '154399712'
  flyvis_noise_05_blank50_dtfd_cv00: '154399713'
  flyvis_noise_05_blank50_dtfd_cv01: '154399714'
  flyvis_noise_05_blank50_dtfd_cv02: '154399715'
  flyvis_noise_05_blank50_dtfd_cv03: '154399716'
  flyvis_noise_05_blank50_dtfd_cv04: '154399717'
  flyvis_noise_free_blank50_dtbug_cv00: '154399718'
  flyvis_noise_free_blank50_dtbug_cv01: '154399719'
  flyvis_noise_free_blank50_dtbug_cv02: '154399720'
  flyvis_noise_free_blank50_dtbug_cv03: '154399721'
  flyvis_noise_free_blank50_dtbug_cv04: '154399722'
  flyvis_noise_005_blank50_dtbug_cv00: '154399723'
  flyvis_noise_005_blank50_dtbug_cv01: '154399724'
  flyvis_noise_005_blank50_dtbug_cv02: '154399725'
  flyvis_noise_005_blank50_dtbug_cv03: '154399726'
  flyvis_noise_005_blank50_dtbug_cv04: '154399728'
  flyvis_noise_05_blank50_dtbug_cv00: '154399729'
  flyvis_noise_05_blank50_dtbug_cv01: '154399730'
  flyvis_noise_05_blank50_dtbug_cv02: '154399731'
  flyvis_noise_05_blank50_dtbug_cv03: '154399732'
  flyvis_noise_05_blank50_dtbug_cv04: '154399733'
report:
  arm_order:
  - bug
  - nominal
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

## Reading the two `fit roll r` columns

Both arms here are CURRENT models trained on current data, so the model's own
message family is the current one. `template_rollout_r` is that own form and
`template_alt_rollout_r` is the conductance form — `alt_form_family` in each
run's `results/metrics.txt` says which, and it reads `conductance` for all 30.

The gap is large and it is not a bug: at `noise_005` cv00 the current form
rolls out at `r = 0.999` (RMSE 0.023 V) and the conductance form at `r = 0.617`
(RMSE 13.66 V), against a voltage that spans about 3.4 V.

Both forms fit the *message* at `R2 = 1.000` per edge — `current_form_r2_median`
and `conductance_form_r2_median` are both 0.999997. The conductance form gets
there by degenerating: on the neuron-2895 panel its per-synapse fit reports
`W = -0.0001` with `E = +10523 V` where the generator has `W = -0.5401`, and
`conductance_form_E_over_vi` is 774, i.e. the reversal it wants sits 774x the
99th-percentile |v_i| away from the voltages the data visits. With `E` that far
out, `W * relu(v_j) * (E - v_i)` is `W*E * relu(v_j)` plus a term in `v_i` that
is negligible at the true voltages, so the product `W*E` is identified and the
two factors separately are not.

That is exactly why the R2 and the rollout disagree. The R2 is measured at the
TRUE `v_i`, where the driving force is a constant the fit absorbs; the rollout
feeds back its OWN `v_i`, and there the neglected `-W * relu(v_j) * v_i` term
has gain `|W*E| / |v_i|` ~ 774 on every deviation, so the trajectory leaves the
data. The conductance readout of a current model is therefore not a competing
description that happens to win — it is an unidentified reparametrisation whose
rollout is unstable, and the rollout is what exposes it.

<!-- STATUS:BEGIN -->

## Status

**30/30 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | noise | n | one-step r | rollout r | fit roll cond | fit roll curr | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | noise_free | 5 | 1.000 ± 0.000 | 0.999 ± 0.000 |  |  | 0.904 ± 0.022 | 0.929 ± 0.018 | 0.828 ± 0.058 | 0.818 ± 0.063 | 0.963 ± 0.013 | 0.020 ± 0.008 |  | 0.856 ± 0.026 |
| nominal | noise_005 | 5 | 0.999 ± 0.000 | 0.999 ± 0.000 | 0.995 ± 0.001 | 0.604 ± 0.047 | 0.956 ± 0.010 | 0.979 ± 0.013 | 0.874 ± 0.037 | 0.756 ± 0.094 | 0.980 ± 0.009 | 0.042 ± 0.020 |  | 0.886 ± 0.009 |
| nominal | noise_05 | 5 | 0.996 ± 0.000 | 0.993 ± 0.002 | 0.994 ± 0.001 | 0.695 ± 0.015 | 0.984 ± 0.001 | 0.996 ± 0.001 | 0.875 ± 0.016 | 0.893 ± 0.012 | 0.994 ± 0.001 | 0.011 ± 0.005 |  | 0.879 ± 0.018 |
| bug | noise_free | 5 | 1.000 ± 0.000 | 0.909 ± 0.180 |  |  | 0.894 ± 0.024 | 0.911 ± 0.042 | 0.830 ± 0.035 | 0.814 ± 0.040 | 0.963 ± 0.018 | 0.024 ± 0.007 |  | 0.831 ± 0.043 |
| bug | noise_005 | 5 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.997 ± 0.005 | 0.621 ± 0.066 | 0.981 ± 0.009 | 0.995 ± 0.003 | 0.967 ± 0.024 | 0.939 ± 0.029 | 0.993 ± 0.005 | 0.011 ± 0.005 |  | 0.923 ± 0.012 |
| bug | noise_05 | 5 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.641 ± 0.068 | 0.999 ± 0.001 | 1.000 ± 0.000 | 0.997 ± 0.001 | 0.998 ± 0.001 | 1.000 ± 0.000 | 0.001 ± 0.000 |  | 0.922 ± 0.004 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | noise | iter | one-step r | rollout r | fit roll cond | fit roll curr | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | |

### Per run

| run | status | iter | commit |
|---|---|---|---|
| `flyvis_noise_free_blank50_dtfd_cv00` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_free_blank50_dtfd_cv01` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_free_blank50_dtfd_cv02` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_free_blank50_dtfd_cv03` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_free_blank50_dtfd_cv04` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtfd_cv00` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtfd_cv01` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtfd_cv02` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtfd_cv03` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtfd_cv04` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtfd_cv00` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtfd_cv01` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtfd_cv02` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtfd_cv03` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtfd_cv04` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_free_blank50_dtbug_cv00` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_free_blank50_dtbug_cv01` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_free_blank50_dtbug_cv02` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_free_blank50_dtbug_cv03` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_free_blank50_dtbug_cv04` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtbug_cv00` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtbug_cv01` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtbug_cv02` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtbug_cv03` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_005_blank50_dtbug_cv04` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtbug_cv00` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtbug_cv01` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtbug_cv02` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtbug_cv03` | landed | 1,520,001 | `bce608981ef9` |
| `flyvis_noise_05_blank50_dtbug_cv04` | landed | 1,520,001 | `bce608981ef9` |

<!-- STATUS:END -->

