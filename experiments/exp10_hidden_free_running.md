---
number: 10
name: hidden_free_running
title: Hidden neurons, free-running against re-injected, zero against a learned mixture
  of visible neurons, at 10% and 20% hidden (conductance lasso 25, recurrent 20)
purpose: does integrating the hidden neurons in the recurrent rollout (free-running)
  and/or generating them as a learned mixture of blindly chosen visible neurons recover
  the hidden voltages and R2_W over all edges, at 10% and 20% hidden
baseline: GraphData/config/fly/flyvis_noise_005_hidden_0NN_no_ngp_blank50_unified_cv00.yaml
  with experiment 9's conductance lasso-25 and 20-step recurrent keys
specs_dir: experiments/specs/exp10/fly
task: train
queue: gpu_rtx6000
wall: '48:00'
n_cpus: 12
axes:
  hidden:
  - '10'
  - '20'
  generator:
  - none
  - mix
  rollout:
  - rs
  - fr
  fold:
  - cv00
arms:
- id: condl25rc20
  label: cond. lasso 25, recurrent 20
  spec_pattern: flyvis_noise_005_hid{hidden}_{generator}_{rollout}_condl25rc20_{fold}
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
    training.recurrent_training: true
report:
  axis_labels:
    hidden:
      '10': 10% hidden
      '20': 20% hidden
    generator:
      none: zero
      mix: mixture of 1024 visible
    rollout:
      rs: re-injected
      fr: free-running
  metric_columns:
    hidden_pearson_mean:
      header: hidden trace r (train)
      live: nnr_pearson
    hidden_rollout_pearson: hidden rollout r
    visible_rollout_pearson: visible rollout r
job_ids:
  flyvis_noise_005_hid10_none_rs_condl25rc20_cv00: '154479522'
  flyvis_noise_005_hid10_none_fr_condl25rc20_cv00: '154479523'
  flyvis_noise_005_hid10_mix_rs_condl25rc20_cv00: '154479525'
  flyvis_noise_005_hid10_mix_fr_condl25rc20_cv00: '154479526'
  flyvis_noise_005_hid20_none_rs_condl25rc20_cv00: '154479527'
  flyvis_noise_005_hid20_none_fr_condl25rc20_cv00: '154479528'
  flyvis_noise_005_hid20_mix_rs_condl25rc20_cv00: '154479529'
  flyvis_noise_005_hid20_mix_fr_condl25rc20_cv00: '154479530'
---

# Experiment 10 — hidden_free_running

**20% hidden neurons with the hidden neurons free-running in 20-step recurrent training, current against conductance lasso 25**

**Purpose.** when the hidden neurons are integrated by the model during the recurrent rollout instead of being re-injected at every step, does the visible-neuron loss constrain their voltages and the edges that touch them, and recover R2_W over all edges; zero start on 5 folds, NGP-T start with and without anchors on cv00

## Why

Experiment 9's audit (2026-09-30) found that a hidden neuron was never
simulated: at every step of the recurrent rollout (and of the test rollout) its
voltage was overwritten by the trace network or by 0, and the model's own
prediction for it was discarded. A wrong hidden trace therefore never
propagated; it acted as a free per-frame input to the visible neurons, the
visible -> hidden weights got no gradient, and the 35% of edges touching a
hidden neuron stayed unlearned.

`training.hidden_free_running: true` injects the hidden voltages at the
rollout's first frame only (0, or the trace network's value there) and then
integrates them with the model like every other neuron, in training and in the
test rollout. A wrong hidden voltage now corrupts the visible neurons for the
rest of the 20-step horizon, so the loss depends on the hidden neurons'
dynamics, their incoming edges and their update function.

Also fixed for this experiment: the anchor loss now reaches recurrent training
(it existed only in one-step training, so experiment 9's recurrent anchor runs
trained none of their 3,600 anchors).

## The mixture generator (blind)

`graph_model.inr_type_hidden: basis_mix` (`models/hidden_basis.py`) writes each
hidden voltage as h_i(t) = sum_m W_im v_{b_m}(t) + c_i over a basis of 1,024
visible neurons. The basis is chosen from the VISIBLE training traces only
(pivoted QR on the standardised traces: the columns that best span the other
visible neurons), and W and c start at 0, so training starts at the zero-silenced
baseline and nothing about the hidden neurons is used to build it. It is a
function of the current state, so it runs in a free-running rollout and on
held-out videos alike; W and c learn at `lr_NNR_f` 0.0005.

**Basis size, chosen blind** (visible traces of the 20% dataset, 4,000 training
frames, least squares of all visible neurons on the basis): 128 neurons explain
37% of the visible variance, 256 42%, 512 54%, 1,024 71%, 2,048 90% (the last
over-fitted on 4,000 frames). 1,024 is used: 2,401 x 1,024 weights.

## What differs

A 2 x 2 x 2 grid, fold cv00, conductance lasso 25 with 20-step recurrent
training (experiment 9's keys) on each hidden fraction's own paper spec:
hidden fraction 10% / 20%, generator zero / mixture, rollout re-injected /
free-running (`training.hidden_free_running`). The zero, re-injected cell is
experiment 9's recurrent baseline, re-run here beside the others.

## Specs

8 runs in `experiments/specs/exp10/fly/`, named
`flyvis_noise_005_hid<10|20>_<none|mix>_<rs|fr>_condl25rc20_cv00`. Queue
`gpu_rtx6000`, wall 48 h.
