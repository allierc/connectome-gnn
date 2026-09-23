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
  label: conductance, group lasso 25
  spec_pattern: flyvis_{noise}_blank50_condl25_{fold}
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
  flyvis_noise_005_blank50_condl25_cv00: '154399862'
  flyvis_noise_005_blank50_condl25_cv01: '154399863'
  flyvis_noise_005_blank50_condl25_cv02: '154399864'
  flyvis_noise_005_blank50_condl25_cv03: '154399865'
  flyvis_noise_005_blank50_condl25_cv04: '154399866'
  flyvis_noise_05_blank50_condl25_cv00: '154399867'
  flyvis_noise_05_blank50_condl25_cv01: '154399868'
  flyvis_noise_05_blank50_condl25_cv02: '154399869'
  flyvis_noise_05_blank50_condl25_cv03: '154399870'
  flyvis_noise_05_blank50_condl25_cv04: '154399871'
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
report:
  arm_order:
  - current
  - cond_l25
  - conductance
  arm_labels:
    cond_l25: conductance
  arm_columns:
    lasso: training.coeff_g_phi_input_group_L1
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

## The three arms

**`current` is not run here.** It is experiment 1's `nominal` arm — the
same fifteen runs, same seeds, same datasets — read from the log tree for
the comparison. It carries `submit: false`, so `exp launch 2` neither
resubmits it nor clears its directories.

**`conductance`**, 15 jobs, lasso 100: 3 model-noise levels x 5 folds.

**`cond_l25`**, 15 jobs, lasso 25. It began as a 5-fold probe at `noise_free`
only, to separate "lambda 100 is too strong" from "sigma 0 cannot hold the
message"; it answered the first (see below), so on 2026-09-23 it was widened to
the full noise axis and the missing **10 jobs** — `noise_005` and `noise_05`,
five folds each — were launched as `154399862`–`154399871`. The five
`noise_free` runs that had already landed were not resubmitted:

```
python tools/exp.py launch 2 --arm cond_l25 --where noise=noise_005,noise_05
```

`--where` exists for exactly this: `--arm cond_l25` alone would have cleared and
relaunched all fifteen.

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

**Answered: the lasso is too strong.** All five lambda-25 folds recovered —
`R2_W` 0.898 +- 0.007 against lambda 100's 0.487 +- 0.409, and no fold collapsed.
Sigma 0 is not the problem. That is what widened this arm to the whole noise
axis.

## The conductance fit-roll column is a clamp, not a score

The lasso-100 arm reports `template_rollout_r` between **0.112 and 0.120** on
every one of its ten `noise_005` and `noise_05` runs, with RMSE 83–85 V. Ten
independent runs at two noise levels agreeing to three digits is a rail, not a
measurement, and it is: **67–71% of the 13,741 neurons sit on the +-100 V
divergence clamp** in
[graph_tester.py:725](../src/connectome_gnn/models/graph_tester.py#L725), which
exists to stop a NaN and instead turns a divergence into a finite number. Flyvis
voltages span about 3.4 V (99th percentile of |v|), so a neuron held at 100 V is
not a bad prediction, it is no prediction. The per-window CSV shows it locking in
the first 500 frames and never moving: RMSE 89.05 in every window thereafter,
pearson 0.03.

| readout | conductance model (lasso 100) | current model (exp01 nominal) |
|---|---|---|
| conductance form | r 0.112–0.120, **67–71% clamped** | r 0.54–0.71, 0.3–19.6% clamped |
| current form | r 0.47–0.51, 0% clamped | r 0.99, 0% clamped |

Read down the column, not across: the current-form readout never clamps and the
conductance-form readout clamps on both models. `template_rollout_roundtrip_rel_dev`
is 0.000000, so the fitted constants were written into the known-ODE faithfully —
the instability is in the constants, not in the write.

**Why the conductance constants are unstable.** All three form-R2 medians are
0.999989 on this run, so the conductance template describes the message as well
as the current one does. It does so degenerately: `conductance_form_E_over_vi`
is 76–593, i.e. the reversal it wants sits one to six hundred times the voltage
scale away from any voltage the data visits. There `W*relu(v_j)*(E - v_i)` is
`W*E*relu(v_j)` plus a term in `v_i` that is negligible **at the true voltages**,
so only the product `W*E` is identified. The R2 is measured at the true `v_i`;
the rollout feeds back its own, and the neglected `-W*relu(v_j)*v_i` term is a
positive feedback with gain `|W*E|/|v_i|` of that same order. It runs away and
the clamp catches it.

So there is no bug in the readout arithmetic or in the rollout. There *was* a
reporting defect: a diverged rollout scored as if it were a weak model.
`Clamped at +/-100 V` now goes into `results_rollout*.log`, through
`template_rollout_pct_clamped` into `metrics.txt`, and into the report table's
parentheses, so the rail is visible beside the number it produced. Runs that
landed before 2026-09-23 have no such field and their fit-roll numbers must be
read against the table above.

<!-- STATUS:BEGIN -->

## Status

**35/45 landed**, 0 trained (awaiting `-o test_plot`), 10 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | noise | n | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | noise_free | 5 | 1.000 ± 0.000 | 0.999 ± 0.000 |  |  | 0.904 ± 0.022 (0.0) | 0.929 ± 0.018 (2.5) | 0.828 ± 0.058 (11.9) | 0.818 ± 0.063 | 0.963 ± 0.013 (0.2) | 0.020 ± 0.008 |  | 0.856 ± 0.026 |
| current | noise_005 | 5 | 0.999 ± 0.000 | 0.999 ± 0.000 | 0.995 ± 0.001 (0.0) | 0.604 ± 0.047 (10.9) | 0.956 ± 0.010 (0.0) | 0.979 ± 0.013 (0.1) | 0.874 ± 0.037 (4.0) | 0.756 ± 0.094 | 0.980 ± 0.009 (0.0) | 0.042 ± 0.020 |  | 0.886 ± 0.009 |
| current | noise_05 | 5 | 0.996 ± 0.000 | 0.993 ± 0.002 | 0.994 ± 0.001 (0.0) | 0.695 ± 0.015 (2.0) | 0.984 ± 0.001 (0.0) | 0.996 ± 0.001 (0.0) | 0.875 ± 0.016 (2.2) | 0.893 ± 0.012 | 0.994 ± 0.001 (0.0) | 0.011 ± 0.005 |  | 0.879 ± 0.018 |
| conductance | noise_free | 5 | 0.847 ± 0.186 | 0.794 ± 0.250 |  |  | 0.487 ± 0.409 (0.3) | -5.304 ± 12.026 (23.6) | 0.668 ± 0.127 (42.9) | 0.625 ± 0.082 | 0.456 ± 0.376 (24.1) | 6.632 ± 12.208 |  | 0.531 ± 0.110 |
| conductance | noise_005 | 5 | 0.998 ± 0.000 | 0.929 ± 0.139 | 0.118 ± 0.002 (68.4) | 0.487 ± 0.018 (0.0) | 0.963 ± 0.013 (0.0) | 0.964 ± 0.015 (0.4) | 0.856 ± 0.026 (4.8) | 0.843 ± 0.066 | 0.984 ± 0.009 (0.0) | 0.018 ± 0.011 |  | 0.888 ± 0.015 |
| conductance | noise_05 | 5 | 0.994 ± 0.001 | 0.902 ± 0.179 | 0.112 ± 0.000 (70.2) | 0.480 ± 0.004 (0.0) | 0.984 ± 0.001 (0.0) | 0.995 ± 0.001 (0.0) | 0.876 ± 0.035 (2.2) | 0.909 ± 0.013 | 0.994 ± 0.002 (0.0) | 0.011 ± 0.006 |  | 0.870 ± 0.021 |
| cond_l25 | noise_free | 5 | 1.000 ± 0.000 | 0.999 ± 0.000 |  |  | 0.898 ± 0.007 (0.1) | 0.902 ± 0.044 (4.1) | 0.873 ± 0.022 (19.6) | 0.774 ± 0.027 | 0.940 ± 0.012 (0.2) | 0.039 ± 0.008 |  | 0.771 ± 0.039 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | noise | iter | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_l25 | noise_005 | 240,001 |  | 0.998 ± 0.000 |  |  | 0.973 ± 0.002 | 0.977 ± 0.008 | 0.878 ± 0.014 | 0.875 ± 0.015 | 0.987 ± 0.003 |  |  | 0.893 ± 0.010 |
| cond_l25 | noise_05 | 240,001 |  | 0.993 ± 0.001 |  |  | 0.985 ± 0.001 | 0.996 ± 0.001 | 0.867 ± 0.034 | 0.901 ± 0.017 | 0.993 ± 0.001 |  |  | 0.887 ± 0.016 |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_noise_free_blank50_dtfd_cv00` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_free_blank50_dtfd_cv01` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_free_blank50_dtfd_cv02` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_free_blank50_dtfd_cv03` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_free_blank50_dtfd_cv04` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_005_blank50_dtfd_cv00` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_005_blank50_dtfd_cv01` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_005_blank50_dtfd_cv02` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_005_blank50_dtfd_cv03` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_005_blank50_dtfd_cv04` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_05_blank50_dtfd_cv00` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_05_blank50_dtfd_cv01` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_05_blank50_dtfd_cv02` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_05_blank50_dtfd_cv03` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_05_blank50_dtfd_cv04` | landed | 1,520,001 | `bce608981ef9` |  |
| `flyvis_noise_free_blank50_condl100_cv00` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_free_blank50_condl100_cv01` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_free_blank50_condl100_cv02` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_free_blank50_condl100_cv03` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_free_blank50_condl100_cv04` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_005_blank50_condl100_cv00` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_005_blank50_condl100_cv01` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_005_blank50_condl100_cv02` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_005_blank50_condl100_cv03` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_005_blank50_condl100_cv04` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_05_blank50_condl100_cv00` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_05_blank50_condl100_cv01` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_05_blank50_condl100_cv02` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_05_blank50_condl100_cv03` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_05_blank50_condl100_cv04` | landed | 1,520,001 | `950fc40dfda9` |  |
| `flyvis_noise_free_blank50_condl25_cv00` | landed | 1,520,001 | `71e4d78710c4` |  |
| `flyvis_noise_free_blank50_condl25_cv01` | landed | 1,520,001 | `71e4d78710c4` |  |
| `flyvis_noise_free_blank50_condl25_cv02` | landed | 1,520,001 | `71e4d78710c4` |  |
| `flyvis_noise_free_blank50_condl25_cv03` | landed | 1,520,001 | `71e4d78710c4` |  |
| `flyvis_noise_free_blank50_condl25_cv04` | landed | 1,520,001 | `71e4d78710c4` |  |
| `flyvis_noise_005_blank50_condl25_cv00` | running | 240,001 | `` |  |
| `flyvis_noise_005_blank50_condl25_cv01` | running | 240,001 | `` |  |
| `flyvis_noise_005_blank50_condl25_cv02` | running | 240,001 | `` |  |
| `flyvis_noise_005_blank50_condl25_cv03` | running | 240,001 | `` |  |
| `flyvis_noise_005_blank50_condl25_cv04` | running | 240,001 | `` |  |
| `flyvis_noise_05_blank50_condl25_cv00` | running | 240,001 | `` |  |
| `flyvis_noise_05_blank50_condl25_cv01` | running | 240,001 | `` |  |
| `flyvis_noise_05_blank50_condl25_cv02` | running | 240,001 | `` |  |
| `flyvis_noise_05_blank50_condl25_cv03` | running | 240,001 | `` |  |
| `flyvis_noise_05_blank50_condl25_cv04` | running | 240,001 | `` |  |

<!-- STATUS:END -->

