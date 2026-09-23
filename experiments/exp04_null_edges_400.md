---
number: 4
name: null_edges_400
title: '+400% null edges: the general form under a group lasso, against nominal current
  training'
purpose: does the general form g_phi = MLP(a_i, a_j, v_i, v_j) under a group lasso
  of 10 regress against nominal current training when the connectome is inflated five-fold
  with null edges; and does the lasso kill the per-edge offset C_ij, read as R2_Vrest
  against R2_Vrest without the C_i correction
baseline: experiments/baseline/gnn_current_baseline.yaml
specs_dir: experiments/specs/exp04/fly
task: train
queue: gpu_rtx6000
wall: '48:00'
axes:
  fold:
  - cv00
  - cv01
  - cv02
  - cv03
  - cv04
arms:
- id: current
  label: current form, no lasso
  spec_pattern: flyvis_noise_005_null400_cur_{fold}
  differs_by: {}
- id: conductance
  label: conductance, group lasso 10
  spec_pattern: flyvis_noise_005_null400_condl10_{fold}
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 10.0
job_ids:
  flyvis_noise_005_null400_cur_cv00: '154396266'
  flyvis_noise_005_null400_cur_cv01: '154396267'
  flyvis_noise_005_null400_cur_cv02: '154396268'
  flyvis_noise_005_null400_cur_cv03: '154396269'
  flyvis_noise_005_null400_cur_cv04: '154396270'
  flyvis_noise_005_null400_condl10_cv00: '154396271'
  flyvis_noise_005_null400_condl10_cv01: '154396272'
  flyvis_noise_005_null400_condl10_cv02: '154396273'
  flyvis_noise_005_null400_condl10_cv03: '154396274'
  flyvis_noise_005_null400_condl10_cv04: '154396275'
---

# Experiment 4 — null_edges_400

**+400% null edges: the general form under a group lasso, against nominal current training**

**Purpose.** does the general form g_phi = MLP(a_i, a_j, v_i, v_j) under a group lasso of 10 regress against nominal current training when the connectome is inflated five-fold with null edges; and does the lasso kill the per-edge offset C_ij, read as R2_Vrest against R2_Vrest without the C_i correction

## The data

`flyvis_noise_005_null_edges_pc_400_blank50_cv0{0..4}`, model noise 0.05,
no measurement noise. The true 434,112 edges plus 1,736,448 random null
edges drawn per presynaptic column — **2,170,560 edges, five times the
true count**, of which 80% should learn a weight of zero. This is the
`+400% null edges` row of the NeurIPS supplementary table
(`figures/cv_table_gnn_cross_noise.tex`), whose published 5-fold numbers
are `R2_W 0.96 ± 0.02` and `R2_Vrest 0.46 ± 0.15`: W recovered well, the
resting level badly.

`n_extra_null_edges` and `null_edges_mode` are **simulation** fields and
must be in the training spec, not only the generator's: W is sized
`n_edges + n_extra_null_edges`, so a spec that names the dataset without
them would build 434,112 weights for a 2,170,560-edge graph.

## What differs

| | `current` | `conductance` |
|---|---|---|
| `signal_model_name` | `flyvis_current` | `flyvis_conductance` |
| `input_size` | 3 | 6 |
| `coeff_g_phi_input_group_L1` | 0 | 10 |

The lasso is the group one — an L2 per input block of `g_phi`'s first
layer, summed as an L1 — so it selects which inputs the edge function may
read. It is **not** a penalty on W, so it does not itself prune a null
edge's weight; what it tests is whether the general form's extra inputs
survive contact with a graph that is 80% spurious.

## Specs

10 = 2 arms x 5 folds.

| arm | fold | spec | dataset |
|---|---|---|---|
| current | cv00 | `flyvis_noise_005_null400_cur_cv00` | `flyvis_noise_005_null_edges_pc_400_blank50_cv00` |
| current | cv01 | `flyvis_noise_005_null400_cur_cv01` | `flyvis_noise_005_null_edges_pc_400_blank50_cv01` |
| current | cv02 | `flyvis_noise_005_null400_cur_cv02` | `flyvis_noise_005_null_edges_pc_400_blank50_cv02` |
| current | cv03 | `flyvis_noise_005_null400_cur_cv03` | `flyvis_noise_005_null_edges_pc_400_blank50_cv03` |
| current | cv04 | `flyvis_noise_005_null400_cur_cv04` | `flyvis_noise_005_null_edges_pc_400_blank50_cv04` |
| conductance | cv00 | `flyvis_noise_005_null400_condl10_cv00` | `flyvis_noise_005_null_edges_pc_400_blank50_cv00` |
| conductance | cv01 | `flyvis_noise_005_null400_condl10_cv01` | `flyvis_noise_005_null_edges_pc_400_blank50_cv01` |
| conductance | cv02 | `flyvis_noise_005_null400_condl10_cv02` | `flyvis_noise_005_null_edges_pc_400_blank50_cv02` |
| conductance | cv03 | `flyvis_noise_005_null400_condl10_cv03` | `flyvis_noise_005_null_edges_pc_400_blank50_cv03` |
| conductance | cv04 | `flyvis_noise_005_null400_condl10_cv04` | `flyvis_noise_005_null_edges_pc_400_blank50_cv04` |

<!-- STATUS:BEGIN -->

## Status

**0/10 landed**, 10 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm |  | n | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm |  | iter | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|
| current |  | 1 |  | 0.002 ± 0.002 |  |  |  |  |  |  |  |  |
| conductance |  | 1 |  | 0.002 ± 0.002 |  |  |  |  |  |  |  |  |

### Per run

| run | status | iter | commit |
|---|---|---|---|
| `flyvis_noise_005_null400_cur_cv00` | running | 1 | `` |
| `flyvis_noise_005_null400_cur_cv01` | running | 1 | `` |
| `flyvis_noise_005_null400_cur_cv02` | running | 1 | `` |
| `flyvis_noise_005_null400_cur_cv03` | running | 1 | `` |
| `flyvis_noise_005_null400_cur_cv04` | running | 1 | `` |
| `flyvis_noise_005_null400_condl10_cv00` | running | 1 | `` |
| `flyvis_noise_005_null400_condl10_cv01` | running | 1 | `` |
| `flyvis_noise_005_null400_condl10_cv02` | running | 1 | `` |
| `flyvis_noise_005_null400_condl10_cv03` | running | 1 | `` |
| `flyvis_noise_005_null400_condl10_cv04` | running | 1 | `` |

<!-- STATUS:END -->

