---
number: 11
name: siren_visual_stimulus
title: 'Supplementary Figure 9 again: joint GNN + visual SIREN (unknown stimulus),
  current against conductance lasso 25, one-step against 20-step recurrent'
purpose: redo Supplementary Figure 9 (the visual stimulus recovered by a SIREN jointly
  with the GNN, rollout on the first 8,000 training frames) with the conductance lasso-25
  model beside the current form, each with one-step and with 20-step recurrent training
baseline: GraphData/config/fly/flyvis_noise_005_INR_davis_blank50_cv00.yaml (Supp.
  Tab. 4, "unknown stim. + SIREN recovery")
specs_dir: experiments/specs/exp11/fly
task: train
queue: gpu_h100
wall: '24:00'
n_cpus: 12
axes:
  fold:
  - cv00
  - cv01
  - cv02
  - cv03
  - cv04
arms:
- id: cur1s
  label: current, one-step
  spec_pattern: flyvis_noise_005_INR_davis_blank50_cur1s_{fold}
  axes_only:
    fold:
    - cv00
  differs_by: {}
- id: currc20
  label: current, recurrent 20
  spec_pattern: flyvis_noise_005_INR_davis_blank50_currc20_{fold}
  axes_only:
    fold:
    - cv00
  differs_by:
    training.recurrent_training: true
- id: condl251s
  label: cond. lasso 25, one-step
  spec_pattern: flyvis_noise_005_INR_davis_blank50_condl251s_{fold}
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
- id: condl25rc20
  label: cond. lasso 25, recurrent 20
  spec_pattern: flyvis_noise_005_INR_davis_blank50_condl25rc20_{fold}
  axes_only:
    fold:
    - cv00
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
    training.recurrent_training: true
- id: cur1sylist
  label: current, one-step, noise-free target
  spec_pattern: flyvis_noise_005_INR_davis_blank50_cur1sylist_{fold}
  axes_only:
    fold:
    - cv00
  differs_by:
    training.derivative_target: y_list
- id: cur1sseed2
  label: current, one-step, seed 2
  spec_pattern: flyvis_noise_005_INR_davis_blank50_cur1sseed2_{fold}
  axes_only:
    fold:
    - cv00
  differs_by:
    training.seed: 2042
report:
  metric_columns:
    stimuli_r: stimulus r
    stimuli_R2: stimulus R2
  arm_labels:
    cur1s: current, one-step
    currc20: current, recurrent 20
    condl251s: conductance lasso 25, one-step
    condl25rc20: conductance lasso 25, recurrent 20
    cur1sylist: current, one-step, noise-free target
    cur1sseed2: current, one-step, seed 2
job_ids:
  flyvis_noise_005_INR_davis_blank50_cur1s_cv00: '154479611'
  flyvis_noise_005_INR_davis_blank50_condl251s_cv00: '154479615'
  flyvis_noise_005_INR_davis_blank50_currc20_cv00: '154479794'
  flyvis_noise_005_INR_davis_blank50_condl25rc20_cv00: '154479797'
  flyvis_noise_005_INR_davis_blank50_condl251s_cv01: '154487999'
  flyvis_noise_005_INR_davis_blank50_condl251s_cv02: '154488000'
  flyvis_noise_005_INR_davis_blank50_condl251s_cv03: '154488001'
  flyvis_noise_005_INR_davis_blank50_condl251s_cv04: '154488002'
  flyvis_noise_005_INR_davis_blank50_cur1sylist_cv00: '154489406'
  flyvis_noise_005_INR_davis_blank50_cur1sseed2_cv00: '154489407'
analyse_job_ids:
  flyvis_noise_005_INR_davis_blank50_cur1s_cv00: '154482384'
  flyvis_noise_005_INR_davis_blank50_condl251s_cv00: '154594758'
  flyvis_noise_005_INR_davis_blank50_condl25rc20_cv00: '154485332'
  flyvis_noise_005_INR_davis_blank50_currc20_cv00: '154485349'
  flyvis_noise_005_INR_davis_blank50_condl251s_cv01: '154489228'
  flyvis_noise_005_INR_davis_blank50_condl251s_cv02: '154489229'
  flyvis_noise_005_INR_davis_blank50_condl251s_cv03: '154489230'
  flyvis_noise_005_INR_davis_blank50_condl251s_cv04: '154490806'
  flyvis_noise_005_INR_davis_blank50_cur1sylist_cv00: '154492646'
  flyvis_noise_005_INR_davis_blank50_cur1sseed2_cv00: '154493744'
---

# Experiment 11 — siren_visual_stimulus

**Supplementary Figure 9 again: joint GNN + visual SIREN (unknown stimulus), current against conductance lasso 25, one-step against 20-step recurrent**

**Purpose.** redo Supplementary Figure 9 (the visual stimulus recovered by a SIREN jointly with the GNN, rollout on the first 8,000 training frames) with the conductance lasso-25 model beside the current form, each with one-step and with 20-step recurrent training

## The figure

Supplementary Figure 9: joint GNN + INR (visual SIREN) on the central
217-column flyvis data, model noise 0.05, 64,000 training frames; the rollout is
evaluated on the first 8,000 TRAINING frames, because a SIREN of absolute time
cannot extrapolate to test-time indices. Panels: (a) the SIREN's stimulus on
the photoreceptor lattice every 80 ms (ground truth, SIREN, residual); (b) the
SIREN's stimulus for 12 photoreceptors over 20 s (1,000 frames at 20 ms);
(c) the GNN's voltage rollout against the noise-free ground truth for 12 cell
types over the same window; (d, e) learned against true stimulus, and rollout
voltage against noise-free ground truth, pooled over (neuron, frame), Pearson r
on the full 8,000-frame rollout. Drawn by `figures/fig_stim_rollout_inr.py`
from each run's `results/rollout_bundle.npz`.

The published run of this spec (archive_4, cv00, current form, A100, 2.0 h):
rollout r 0.745 (mean per neuron), stimulus r 0.924 (stimulus R2 0.854),
R2_tau 0.960, cluster 0.886.

## What differs

Each spec is `flyvis_noise_005_INR_davis_blank50_cv00` (current form,
`inr_type: siren_txy` for the stimulus: 2,048 wide, 4 layers, omega 4,096,
`alternate_training` with the GNN's learning rates at 0.05x after epoch 0) with:

- **conductance lasso 25**: `signal_model_name: flyvis_conductance`,
  `input_size: 6`, `coeff_g_phi_input_group_L1: 25`;
- **recurrent 20**: `recurrent_training: true`, `rollout_horizon_schedule`
  1..20, `n_epochs` 3 -> 20 (one horizon per epoch) and
  `data_augmentation_loop` 25 -> 4, so the total frames visited stay close to
  the one-step run's (20 x 4 = 80 against 3 x 25 = 75 augmentation loops).
  With `alternate_training` kept, only the first of the 20 epochs runs at the
  GNN's full learning rate (one of three in the one-step run).

## Not yet available

Panel (e) compares with the NOISE-FREE ground truth. This dataset has no
noise-free twin (`flyvis_noise_free_INR_davis_blank50_cv00` does not exist), so
the tester scores against the noisy recording; the published figure read a
separately made `rollout_bundle_nf_synthetic.npz`. Generating the twin (same
seed and stimulus, noise 0) is a data-generation job.

## Five folds for the conductance lasso-25 one-step arm (2026-10-01)

The arm that reproduces the paper's run (stimulus r 0.90, rollout r 0.745 on cv00) is extended to folds cv01-cv04 for the new paper; the other three arms stay at cv00.

## Audit of the two failures (2026-10-01)

Read-only audit with local checks; no code bug found, both are training
dynamics and settings.

**Current form, one-step (stimulus r 0.50 against the paper's 0.92 on the same
spec).**
- The one effective change since the paper run (2026-04-29) is
  `training.derivative_target`, now `observed_fd`: the target is the finite
  difference of the recorded voltage, which carries the process noise
  (variance 6.25 = (0.05/0.02)^2 against 42 for the photoreceptor targets); the
  paper trained on the generator's noise-free derivative. The pred loss per
  epoch doubled (4,018 -> 8,630).
- Mechanism: the update MLP's response to the stimulus input became
  NON-MONOTONE (+42 below an input of 0.15, -27 above), while the paper's was
  monotone (+46 to +75) and the conductance run's monotone with the other sign
  (-44 to -71). With the SIREN giving one value per column (the 8 photoreceptor
  types share 217 positions), a folded response maps the stimulus onto two
  branches and the best affine fit reaches 0.50. Signed r per epoch: paper
  +0.79/+0.90/+0.92, current -0.26/-0.43/-0.50, conductance -0.68/-0.86/-0.89 (the
  reported r is |r| after the affine fit).
- One run per arm cannot separate the noisy target from seed luck.

**Recurrent runs (stimulus r ~0).** No gradient break: the SIREN receives
gradients through the rollout and is in the optimizer. The recurrent epochs are
short (51,200 iterations in epoch 0 against 320,000 one-step; 184,198 SIREN
steps in total against 960,000), and the weight-L1 annealing is indexed by
EPOCH, so it switches on after 51k iterations and, with the 0.05x learning-rate
damping of alternate training, drives the update MLP's stimulus column to zero
(norm 4.36 -> 0.52 -> 5.8e-5 over epochs 0-2) before the SIREN carries signal --
an absorbing state: no stimulus pathway, no gradient to the SIREN.

Also found: the optimizer is rebuilt every epoch (Adam state reset 19 times in
a recurrent run), and the embedding-unfreeze rebuild omits `lr_NNR_f` (inactive
here).

**Fixes proposed**: current one-step rerun with `derivative_target: y_list` (the
paper's objective) plus a second seed; for recurrent, warm-start from a one-step
checkpoint (or a full 320k-iteration one-step epoch 0), exclude the stimulus
column from the weight L1 or anneal by iteration, delay the alternate-training
damping, and keep the optimizer across epochs.

## Test of the current-form shortfall (launched 2026-10-01)

Two current-form one-step runs on cv00: `cur1sylist` trains on the generator's
noise-free derivative (`derivative_target: y_list`, the paper run's objective);
`cur1sseed2` keeps today's target with a second training seed (2042). Stimulus
r back near 0.9 in the first says the noisy target caused the fold; in the
second, that the 0.50 was an unlucky seed.

## Results of the test and of the five folds (2026-10-01)

| run | stimulus r | rollout r | R2_W | R2_tau | cluster |
|---|---|---|---|---|---|
| paper run (current, cv00, 2026-04) | 0.924 | 0.745 | -- | 0.960 | 0.886 |
| current, one-step, cv00 | 0.501 | 0.445 | 0.920 | 0.776 | 0.922 |
| current, one-step, noise-free target (y_list) | 0.820 | 0.603 | 0.939 | 0.912 | 0.922 |
| current, one-step, seed 2 | 0.724 | 0.541 | 0.952 | 0.919 | 0.902 |
| cond. lasso 25, one-step, cv00 | 0.898 | 0.745 | 0.904 | 0.949 | 0.897 |
| cond. lasso 25, one-step, cv01 | 0.913 | 0.740 | 0.860 | 0.921 | 0.813 |
| cond. lasso 25, one-step, cv02 | 0.904 | 0.742 | 0.858 | 0.739 | 0.802 |
| cond. lasso 25, one-step, cv03 | 0.923 | 0.044 | -0.262 | 0.518 | 0.705 |
| cond. lasso 25, one-step, cv04 | 0.433 | 0.336 | 0.906 | 0.798 | 0.900 |

**Both causes contribute to the current form's shortfall.** The noise-free
target lifts stimulus r 0.50 -> 0.82 and a second seed alone gives 0.72, so the
noisy target costs about 0.1-0.3 and seed-to-seed spread is large on its own;
neither reaches the paper's 0.92.

**The joint GNN + SIREN fit is unstable across folds as well.** The conductance
one-step arm recovers the stimulus at 0.90-0.92 on four folds out of five (cv04:
0.43), and the circuit on three (cv03: the stimulus at 0.92 but R2_W -0.26 and
rollout r 0.04). Mean over the five folds: stimulus r 0.81 +- 0.19, R2_W
0.65 +- 0.46; over the three folds where both are recovered: stimulus r
0.90-0.91, rollout r 0.74, R2_W 0.86-0.90.

## Specs

4 runs in `experiments/specs/exp11/fly/`, fold cv00. Queue `gpu_h100`, 12
slots, wall 24 h.

<!-- STATUS:BEGIN -->

## Status

**10/10 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | n | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster | stimulus r | stimulus R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cur1s | 1 | 0.997 ± 0.000 | 0.445 ± 0.000 |  |  | 0.920 ± 0.000 (0.1) | 0.776 ± 0.000 (6.9) | 0.751 ± 0.000 (18.2) | 0.751 ± 0.000 | 0.900 ± 0.000 (0.8) | 0.044 ± 0.000 | 2.182 ± 0.000 | 0.922 ± 0.000 | 0.501 ± 0.000 | 0.251 ± 0.000 |
| currc20 | 1 | 0.980 ± 0.000 | -0.016 ± 0.000 |  |  | 0.892 ± 0.000 (0.0) | 0.890 ± 0.000 (12.3) | 0.726 ± 0.000 (31.5) | 0.599 ± 0.000 | 0.879 ± 0.000 (0.4) | 0.053 ± 0.000 | 3.942 ± 0.000 | 0.889 ± 0.000 | 0.003 ± 0.000 | 0.000 ± 0.000 |
| condl251s | 5 | 0.994 ± 0.002 | 0.521 ± 0.286 |  |  | 0.653 ± 0.458 (0.3) | 0.785 ± 0.154 (3.6) | 0.720 ± 0.060 (41.0) | 0.721 ± 0.084 | 0.756 ± 0.246 (18.3) | 0.126 ± 0.182 | 1.155 ± 1.918 | 0.823 ± 0.072 | 0.815 ± 0.191 | 0.700 ± 0.257 |
| condl25rc20 | 1 | 0.966 ± 0.000 | -0.009 ± 0.000 |  |  | 0.695 ± 0.000 (0.0) | 0.165 ± 0.000 (6.4) | 0.496 ± 0.000 (39.7) | 0.430 ± 0.000 | 0.847 ± 0.000 (0.9) | 0.106 ± 0.000 | 2.787 ± 0.000 | 0.755 ± 0.000 | 0.005 ± 0.000 | 0.000 ± 0.000 |
| cur1sylist | 1 | 0.999 ± 0.000 | 0.603 ± 0.000 |  |  | 0.939 ± 0.000 (0.0) | 0.912 ± 0.000 (0.9) | 0.866 ± 0.000 (11.4) | 0.821 ± 0.000 | 0.911 ± 0.000 (0.2) | 0.018 ± 0.000 | 2.568 ± 0.000 | 0.922 ± 0.000 | 0.820 ± 0.000 | 0.672 ± 0.000 |
| cur1sseed2 | 1 | 0.997 ± 0.000 | 0.541 ± 0.000 |  |  | 0.952 ± 0.000 (0.0) | 0.919 ± 0.000 (1.4) | 0.770 ± 0.000 (22.2) | 0.691 ± 0.000 | 0.906 ± 0.000 (0.2) | 0.035 ± 0.000 | 2.185 ± 0.000 | 0.902 ± 0.000 | 0.724 ± 0.000 | 0.523 ± 0.000 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | iter | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster | stimulus r | stimulus R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | | |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_noise_005_INR_davis_blank50_cur1s_cv00` | landed | 944,001 | `b089e21e6998` |  |
| `flyvis_noise_005_INR_davis_blank50_currc20_cv00` | landed | 184,071 | `b089e21e6998` |  |
| `flyvis_noise_005_INR_davis_blank50_condl251s_cv00` | landed | 944,001 | `b089e21e6998` |  |
| `flyvis_noise_005_INR_davis_blank50_condl251s_cv01` | landed | 944,001 | `b089e21e6998` |  |
| `flyvis_noise_005_INR_davis_blank50_condl251s_cv02` | landed | 944,001 | `b089e21e6998` |  |
| `flyvis_noise_005_INR_davis_blank50_condl251s_cv03` | landed | 944,001 | `b089e21e6998` |  |
| `flyvis_noise_005_INR_davis_blank50_condl251s_cv04` | landed | 944,001 | `b089e21e6998` |  |
| `flyvis_noise_005_INR_davis_blank50_condl25rc20_cv00` | landed | 184,071 | `b089e21e6998` |  |
| `flyvis_noise_005_INR_davis_blank50_cur1sylist_cv00` | landed | 944,001 | `b089e21e6998` |  |
| `flyvis_noise_005_INR_davis_blank50_cur1sseed2_cv00` | landed | 944,001 | `b089e21e6998` |  |

<!-- STATUS:END -->
