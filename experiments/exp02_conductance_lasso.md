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
  flyvis_noise_005_blank50_condl25_cv03: '154428041'
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
  flyvis_noise_free_blank50_condl100_cv00: '154465700'
  flyvis_noise_free_blank50_condl100_cv01: '154465701'
  flyvis_noise_free_blank50_condl100_cv02: '154465702'
  flyvis_noise_free_blank50_condl100_cv03: '154465703'
  flyvis_noise_free_blank50_condl100_cv04: '154465704'
  flyvis_noise_005_blank50_condl100_cv00: '154465705'
  flyvis_noise_005_blank50_condl100_cv01: '154465706'
  flyvis_noise_005_blank50_condl100_cv02: '154465707'
  flyvis_noise_005_blank50_condl100_cv03: '154465708'
  flyvis_noise_005_blank50_condl100_cv04: '154465709'
  flyvis_noise_05_blank50_condl100_cv00: '154465710'
  flyvis_noise_05_blank50_condl100_cv01: '154465711'
  flyvis_noise_05_blank50_condl100_cv02: '154465712'
  flyvis_noise_05_blank50_condl100_cv03: '154465713'
  flyvis_noise_05_blank50_condl100_cv04: '154465714'
  flyvis_noise_free_blank50_condl25_cv00: '154465679'
  flyvis_noise_free_blank50_condl25_cv01: '154465715'
  flyvis_noise_free_blank50_condl25_cv02: '154465716'
  flyvis_noise_free_blank50_condl25_cv03: '154465717'
  flyvis_noise_free_blank50_condl25_cv04: '154465718'
  flyvis_noise_005_blank50_condl25_cv00: '154465719'
  flyvis_noise_005_blank50_condl25_cv01: '154465720'
  flyvis_noise_005_blank50_condl25_cv02: '154465721'
  flyvis_noise_005_blank50_condl25_cv03: '154465722'
  flyvis_noise_005_blank50_condl25_cv04: '154465723'
  flyvis_noise_05_blank50_condl25_cv00: '154465724'
  flyvis_noise_05_blank50_condl25_cv01: '154465725'
  flyvis_noise_05_blank50_condl25_cv02: '154465726'
  flyvis_noise_05_blank50_condl25_cv03: '154465727'
  flyvis_noise_05_blank50_condl25_cv04: '154465728'
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

### Per synapse: the readout deletes the inhibition

Reading `models/template_fit.pt` — the conductance known-ODE that is actually
rolled out — against the generator's `ode_params.pt`, on
`flyvis_noise_005_blank50_condl100_cv00`:

| | all | truly inhibitory | truly excitatory |
|---|---|---|---|
| edges | 434,112 | 151,155 | 282,957 |
| conductance fit set to **zero** | 44.3% | **90.5%** | 19.6% |
| current fit keeps the sign | | 62.9% | 49.0% |

| written into the known-ODE | |
|---|---|
| edges marked inhibitory | **0.0%** (truth: 34.8%) |
| `E_exc`, every neuron | **+17.45 V** = 5.1x the 99th percentile of \|v\| (3.44 V) |
| driving force `E_exc - v_i` | +14.0 to +20.9 V, **never changes sign** |
| conductance `g >= 0` | 100.0% |

Three steps, each following from the last:

1. **The reversal is unidentified, so it runs away.** `conductance_form_E_absmedian`
   is 1,130 V against a 3.44 V signal. Only the product `g*E` is determined, so
   the fit is free to put `E` anywhere far out and scale `g` down to match.
2. **The sign rule then sees no inhibition.** An edge is called inhibitory when
   its recovered reversal sits below the postsynaptic cell's resting potential,
   `E_e < v_rest` ([template_rollout.py:280](../src/connectome_gnn/template_rollout.py#L280)).
   A runaway `E` is overwhelmingly positive, so **zero** of 434,112 edges
   qualify against a truth of 34.8%.
3. **What inhibition survived as a negative conductance is clipped away.** The
   conductance class squares `W`, so a negative recovered value has no square
   root and enters as zero
   ([template_rollout.py:164](../src/connectome_gnn/template_rollout.py#L164)).
   That deletes **90.5% of the 151,155 inhibitory synapses** against 19.6% of
   the excitatory ones.

What is rolled out is therefore an **all-excitatory recurrent network** with
`g >= 0` on every edge and a driving force that never changes sign. That is
unconditionally unstable — higher voltage, larger `relu(v_j)`, more positive
current — and it saturates the clamp in under 500 frames. The current-form
readout of the *same message* keeps the sign on 62.9% of inhibitory edges and
rolls out without clamping at all.

**So `fit roll r` on the conductance form of a current model measures the
readout, not the model.** The message is fit at R2 0.999989 by both forms; it is
the conversion of that message into conductance known-ODE parameters that
destroys the circuit. Two ways out, neither taken yet: let the known-ODE carry
signed per-edge reversals instead of two per-neuron rows chosen by a sign rule,
or stop reporting this column when `conductance_form_E_over_vi` says the
reversal was never identified.

So there is no bug in the readout ARITHMETIC or in the rollout itself. There *was* a
reporting defect: a diverged rollout scored as if it were a weak model.
`Clamped at +/-100 V` now goes into `results_rollout*.log`, through
`template_rollout_pct_clamped` into `metrics.txt`, and into the report table's
parentheses, so the rail is visible beside the number it produced. Runs that
landed before 2026-09-23 have no such field and their fit-roll numbers must be
read against the table above.

<!-- READOUT_FIX:BEGIN -->

## Re-analysed on the fixed readout (2026-09-28)

Every landed run was re-analysed after the second-pass fix and the switch to one uniform draw of 1,024 frames (commit 9ef188e6, experiment 8); the earlier `metrics.txt` is kept as `superseded/pre_readout_fix/`. Fold means, old -> new, from `tools/readout_fix_compare.py`. The status table below is the new readout.

| arm | lasso | noise | n | R2_W | R2_Vrest | fit roll r current form | fit roll r conductance form | cluster | edges fitted % |
|---|---|---|---|---|---|---|---|---|---|
| current | --- | noise_free | 5 | 0.904 -> 0.936 | 0.828 -> 0.831 | 0.997 -> 0.977 | 0.632 -> 0.693 | 0.856 -> 0.861 | 99.3 -> 76.8 |
| current | --- | noise_005 | 5 | 0.956 -> 0.989 | 0.874 -> 0.911 | 0.995 -> 0.979 | 0.604 -> 0.609 | 0.886 -> 0.884 | 99.3 -> 76.9 |
| current | --- | noise_05 | 5 | 0.984 -> 0.990 | 0.875 -> 0.880 | 0.994 -> 0.994 | 0.695 -> 0.712 | 0.879 -> 0.879 | 100.0 -> 96.2 |
| conductance | 100 | noise_free | 5 | 0.487 -> 0.535 | 0.668 -> 0.634 |  |  | 0.531 -> 0.530 | 99.3 -> 76.8 |
| conductance | 100 | noise_005 | 5 | 0.963 -> 0.976 | 0.856 -> 0.849 |  |  | 0.888 -> 0.887 | 99.3 -> 76.9 |
| conductance | 100 | noise_05 | 5 | 0.984 -> 0.990 | 0.876 -> 0.875 |  |  | 0.870 -> 0.870 | 100.0 -> 96.2 |
| conductance | 25 | noise_free | 5 | 0.898 -> 0.934 | 0.873 -> 0.825 |  |  | 0.771 -> 0.772 | 99.3 -> 76.8 |
| conductance | 25 | noise_005 | 5 | 0.974 -> 0.989 | 0.897 -> 0.903 |  |  | 0.895 -> 0.893 | 99.3 -> 76.9 |
| conductance | 25 | noise_05 | 5 | 0.985 -> 0.991 | 0.887 -> 0.887 |  |  | 0.886 -> 0.888 | 100.0 -> 96.2 |

<!-- READOUT_FIX:END -->

**What changes:** the current form and the conductance form under a lasso of 25 are now tied at every noise level (`R2_W` 0.936 / 0.934, 0.989 / 0.989, 0.990 / 0.991). Without the lasso the conductance form still fails at noise_free (0.535). The fit roll of the conductance form stays diverged in every arm.


<!-- STATUS:BEGIN -->

## Status

**45/45 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | noise | n | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | noise_free | 5 | 1.000 ± 0.000 | 0.999 ± 0.000 | 0.977 ± 0.002 (0.2) | 0.693 ± 0.033 (3.7) | 0.936 ± 0.028 (0.0) | 0.929 ± 0.018 (2.5) | 0.831 ± 0.057 (11.3) | 0.818 ± 0.063 | 0.963 ± 0.013 (0.2) | 0.015 ± 0.007 | 0.225 ± 0.294 | 0.861 ± 0.017 |
| current | noise_005 | 5 | 0.999 ± 0.000 | 0.999 ± 0.000 | 0.979 ± 0.001 (0.0) | 0.609 ± 0.035 (9.1) | 0.989 ± 0.009 (0.0) | 0.979 ± 0.013 (0.1) | 0.911 ± 0.024 (2.7) | 0.756 ± 0.094 | 0.980 ± 0.009 (0.0) | 0.036 ± 0.018 | 0.324 ± 0.005 | 0.884 ± 0.013 |
| current | noise_05 | 5 | 0.996 ± 0.000 | 0.993 ± 0.002 | 0.994 ± 0.001 (0.0) | 0.712 ± 0.009 (2.0) | 0.990 ± 0.001 (0.0) | 0.996 ± 0.001 (0.0) | 0.880 ± 0.015 (2.2) | 0.893 ± 0.012 | 0.994 ± 0.001 (0.0) | 0.010 ± 0.005 | 0.062 ± 0.324 | 0.879 ± 0.018 |
| conductance | noise_free | 5 | 0.847 ± 0.186 | 0.794 ± 0.250 | 0.760 ± 0.213 (15.2) | 0.602 ± 0.085 (16.6) | 0.535 ± 0.437 (0.2) | -5.304 ± 12.026 (23.6) | 0.634 ± 0.071 (53.6) | 0.623 ± 0.083 | 0.459 ± 0.374 (25.4) | 11.717 ± 21.403 | 0.250 ± 0.206 | 0.530 ± 0.102 |
| conductance | noise_005 | 5 | 0.998 ± 0.000 | 0.935 ± 0.127 | 0.976 ± 0.003 (0.0) | 0.698 ± 0.030 (4.0) | 0.976 ± 0.024 (0.0) | 0.964 ± 0.015 (0.4) | 0.849 ± 0.041 (4.2) | 0.843 ± 0.066 | 0.984 ± 0.009 (0.0) | 0.014 ± 0.009 | 0.346 ± 0.008 | 0.887 ± 0.016 |
| conductance | noise_05 | 5 | 0.994 ± 0.001 | 0.902 ± 0.179 | 0.993 ± 0.001 (0.0) | 0.745 ± 0.014 (1.1) | 0.990 ± 0.001 (0.0) | 0.995 ± 0.001 (0.0) | 0.875 ± 0.034 (2.2) | 0.909 ± 0.013 | 0.994 ± 0.002 (0.0) | 0.010 ± 0.006 | 0.167 ± 0.251 | 0.870 ± 0.021 |
| cond_l25 | noise_free | 5 | 1.000 ± 0.000 | 0.999 ± 0.000 | 0.971 ± 0.002 (0.2) | 0.707 ± 0.052 (2.1) | 0.934 ± 0.010 (0.0) | 0.902 ± 0.044 (4.1) | 0.825 ± 0.018 (19.3) | 0.774 ± 0.027 | 0.940 ± 0.012 (0.2) | 0.029 ± 0.006 | 0.387 ± 0.010 | 0.772 ± 0.047 |
| cond_l25 | noise_005 | 5 | 0.999 ± 0.000 | 0.998 ± 0.000 | 0.979 ± 0.001 (0.0) | 0.654 ± 0.055 (2.0) | 0.989 ± 0.004 (0.0) | 0.980 ± 0.004 (0.0) | 0.903 ± 0.017 (3.1) | 0.903 ± 0.017 | 0.990 ± 0.002 (0.0) | 0.006 ± 0.001 | 0.340 ± 0.003 | 0.893 ± 0.014 |
| cond_l25 | noise_05 | 5 | 0.995 ± 0.000 | 0.878 ± 0.229 | 0.995 ± 0.001 (0.0) | 0.748 ± 0.014 (1.1) | 0.991 ± 0.001 (0.0) | 0.997 ± 0.001 (0.0) | 0.887 ± 0.022 (2.1) | 0.924 ± 0.016 | 0.995 ± 0.001 (0.0) | 0.011 ± 0.006 | 0.173 ± 0.226 | 0.888 ± 0.006 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | noise | iter | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | |

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
| `flyvis_noise_005_blank50_condl25_cv00` | landed | 1,520,001 | `7278aa10679f` |  |
| `flyvis_noise_005_blank50_condl25_cv01` | landed | 1,520,001 | `7278aa10679f` |  |
| `flyvis_noise_005_blank50_condl25_cv02` | landed | 1,520,001 | `7278aa10679f` |  |
| `flyvis_noise_005_blank50_condl25_cv03` | landed | 1,520,001 | `69d59d4e6a93` |  |
| `flyvis_noise_005_blank50_condl25_cv04` | landed | 1,520,001 | `7278aa10679f` |  |
| `flyvis_noise_05_blank50_condl25_cv00` | landed | 1,520,001 | `7278aa10679f` |  |
| `flyvis_noise_05_blank50_condl25_cv01` | landed | 1,520,001 | `7278aa10679f` |  |
| `flyvis_noise_05_blank50_condl25_cv02` | landed | 1,520,001 | `7278aa10679f` |  |
| `flyvis_noise_05_blank50_condl25_cv03` | landed | 1,520,001 | `7278aa10679f` |  |
| `flyvis_noise_05_blank50_condl25_cv04` | landed | 1,520,001 | `7278aa10679f` |  |

<!-- STATUS:END -->

