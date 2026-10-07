---
number: 14
name: manuscript_gaps
title: 'The three rows of the degraded-data table the general form never ran: -20%
  and -50% edges removed, 10% hidden neurons (conductance lasso 25, one-step, 5 folds)'
purpose: fill the three red rows of the manuscript's Supplementary Table 4 (experiments/paper)
  with the general-form GNN (g_phi = MLP(a_i, a_j, v_i, v_j), group lasso 25, one-step
  training) at low model noise, five folds each, so every GNN row of the paper comes
  from the same model and readout
baseline: experiments/specs/exp02/fly/flyvis_noise_005_blank50_condl25_cv0N.yaml for
  the edge-removal rows (dataset and edge-removal keys of the published flyvis_noise_005_removed_pc_{20,50}_blank50_unified
  specs); experiments/specs/exp09/fly/flyvis_noise_005_hid20_none_condl251s_cv0N.yaml
  for the 10%-hidden row (hidden_neuron_fraction 0.1 on the hidden_010 datasets)
specs_dir: experiments/specs/exp14/fly
task: train
queue: gpu_rtx6000
wall: '48:00'
n_cpus: 12
axes:
  condition:
  - rm20
  - rm50
  - hid10
  fold:
  - cv00
  - cv01
  - cv02
  - cv03
  - cv04
arms:
- id: condl251s
  label: conductance lasso 25, one-step
  spec_pattern: flyvis_noise_005_{condition}_condl251s_{fold}
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
report:
  axis_labels:
    condition:
      rm20: -20% edges removed (347,000 edges)
      rm50: -50% edges removed (217,056 edges)
      hid10: 10% hidden neurons
  arm_labels:
    condl251s: conductance lasso 25, one-step
job_ids:
  flyvis_noise_005_rm20_condl251s_cv00: '154565155'
  flyvis_noise_005_rm20_condl251s_cv01: '154565156'
  flyvis_noise_005_rm20_condl251s_cv02: '154565157'
  flyvis_noise_005_rm20_condl251s_cv03: '154565158'
  flyvis_noise_005_rm20_condl251s_cv04: '154565159'
  flyvis_noise_005_rm50_condl251s_cv00: '154565160'
  flyvis_noise_005_rm50_condl251s_cv01: '154565161'
  flyvis_noise_005_rm50_condl251s_cv02: '154565162'
  flyvis_noise_005_rm50_condl251s_cv03: '154565164'
  flyvis_noise_005_rm50_condl251s_cv04: '154565165'
  flyvis_noise_005_hid10_condl251s_cv00: '154565166'
  flyvis_noise_005_hid10_condl251s_cv01: '154565167'
  flyvis_noise_005_hid10_condl251s_cv02: '154565168'
  flyvis_noise_005_hid10_condl251s_cv03: '154565169'
  flyvis_noise_005_hid10_condl251s_cv04: '154565170'
---

# Experiment 14 — manuscript_gaps

**The three rows of the degraded-data table the general form never ran: -20% and -50%
edges removed, 10% hidden neurons (conductance lasso 25, one-step, 5 folds)**

**Purpose.** fill the three red rows of the manuscript's Supplementary Table 4
(experiments/paper) with the general-form GNN at low model noise, five folds each, so
every GNN row of the paper comes from the same model and readout.

## Why

The manuscript rebuild of 2026-10-07 (experiments/paper, `scripts/make_tables.py`)
replaced every GNN row of the paper by the campaign's lasso-25 runs. Three conditions of
the published degraded-data table had no such run: the two edge-removal rows (false
negatives in the connectome prior) and the 10%-hidden row (experiment 9 ran only 20%
hidden with the general form). Those rows are printed in red with the published
current-form numbers until this experiment lands.

## What differs

One arm, the conductance lasso-25 model with one-step training, on three conditions at
model noise 0.05, five folds each:

- `rm20`, `rm50`: experiment 2's low-noise spec with the dataset and the edge-removal
  keys (`n_edges`, `edge_removal_ratio`, `edge_removal_mode: per_column`,
  `edge_removal_seed: 42`, `edge_mask_path`) of the published
  `flyvis_noise_005_removed_pc_{20,50}_blank50_unified_cv0N` specs. The removed edges
  are absent from the prior graph the GNN trains on, while the data were simulated with
  the full connectome.
- `hid10`: experiment 9's 20%-hidden one-step spec with `hidden_neuron_fraction: 0.1`
  on the `flyvis_noise_005_hidden_010_no_ngp_blank50_cv0N` datasets.

The published rows these replace: -20% R2_W 0.78 (rollout r 0.99), -50% R2_W 0.19
(rollout r 0.56), 10% hidden R2_W 0.69 (rollout r 0.88).

## Specs

15 runs in `experiments/specs/exp14/fly/`, named
`flyvis_noise_005_<rm20|rm50|hid10>_condl251s_cv0N`. Queue `gpu_rtx6000`, wall 48 h.
After landing: `python scripts/make_tables.py` in experiments/paper reads them once the
three `pub=` rows of `table_cross` are switched to their run patterns.
