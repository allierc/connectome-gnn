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
n_cpus: 12
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
- id: cond_l25
  label: conductance, group lasso 25
  spec_pattern: flyvis_noise_005_null400_condl25_{fold}
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
job_ids:
  flyvis_noise_005_null400_cur_cv00: '154400229'
  flyvis_noise_005_null400_cur_cv01: '154400230'
  flyvis_noise_005_null400_cur_cv02: '154400231'
  flyvis_noise_005_null400_cur_cv03: '154400232'
  flyvis_noise_005_null400_cur_cv04: '154400233'
  flyvis_noise_005_null400_condl10_cv00: '154400234'
  flyvis_noise_005_null400_condl10_cv01: '154400235'
  flyvis_noise_005_null400_condl10_cv02: '154400236'
  flyvis_noise_005_null400_condl10_cv03: '154400237'
  flyvis_noise_005_null400_condl10_cv04: '154400238'
  flyvis_noise_005_null400_condl25_cv00: '154400336'
  flyvis_noise_005_null400_condl25_cv01: '154400337'
  flyvis_noise_005_null400_condl25_cv02: '154400338'
  flyvis_noise_005_null400_condl25_cv03: '154400339'
  flyvis_noise_005_null400_condl25_cv04: '154400340'
report:
  arm_order:
  - current
  - cond_l25
  - conductance
  arm_labels:
    cond_l25: conductance
  arm_columns:
    lasso: training.coeff_g_phi_input_group_L1
analyse_job_ids:
  flyvis_noise_005_null400_cur_cv00: '154431190'
  flyvis_noise_005_null400_cur_cv01: '154431191'
  flyvis_noise_005_null400_cur_cv02: '154431192'
  flyvis_noise_005_null400_cur_cv03: '154431193'
  flyvis_noise_005_null400_cur_cv04: '154431194'
  flyvis_noise_005_null400_condl10_cv00: '154431195'
  flyvis_noise_005_null400_condl10_cv01: '154431196'
  flyvis_noise_005_null400_condl10_cv02: '154431197'
  flyvis_noise_005_null400_condl25_cv00: '154431198'
  flyvis_noise_005_null400_condl25_cv01: '154431199'
  flyvis_noise_005_null400_condl25_cv02: '154431200'
  flyvis_noise_005_null400_condl25_cv03: '154431201'
  flyvis_noise_005_null400_condl25_cv04: '154431202'
  flyvis_noise_005_null400_condl10_cv03: '154451107'
  flyvis_noise_005_null400_condl10_cv04: '154451108'
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

## The first launch died on HOST memory, and the queue was not the cause

All ten jobs of the 2026-09-22 launch were killed with `TERM_MEMLIMIT` after
writing one snapshot each, and went unnoticed for fifteen hours because the tool
had no `died` state and read their one snapshot as "running".

`Max Memory 169,230 MB` against `Total Requested 163,840 MB` — over by 5,390 MB,
about 3%. **That is host RAM, not GPU memory**: a card running out would raise a
CUDA out-of-memory error from Python, not have LSF take the job. LSF allocates
**20 GB of host RAM per slot**, so the slot count is how a job asks for memory
and `bsub -n 8` caps it at 160 GB whatever the GPU is. Moving to `gpu_a100`
would not have changed anything; an A100 job at 8 slots would have died the same
way.

Why this grid and not the others: `null_edges_pc_400` inflates the connectome
five-fold, 434,112 → 2.17 M edges, and the per-edge tensors scale with it.

Relaunched 2026-09-23 with `n_cpus: 12` (240 GB) as `154400100`–`154400109`, all
ten `RUN` on `gpu_rtx6000` within a minute. Measured live: **144–201 GB, peak
84% of the limit**, and the peak is the data-loading transient — one fold had
already fallen to 7.5 GB ten minutes in. `gpu_a100` had 2,392 jobs pending
against `gpu_rtx6000`'s zero at the time, so the RTX 6000 was also the faster
place to be.

## Precision: bf16, and it is the only half-precision option

The ten runs are `training.mlp_precision: bf16` as of 2026-09-23,
`154400229`–`154400238`. **fp16 is not accepted** — `mlp_precision` is
`Literal["fp32", "tf32", "bf16"]` and a spec naming fp16 fails validation at
load. bf16 is the half-precision that exists, and experiment 0 measured it as
free on recovery (`R2_W` 0.944 against fp32's 0.948, within the 0.939–0.948
spread of all four arms) and +6% in throughput on this card, 48.6 against
39.7 it/s.

Both arms carry it, so the current-against-conductance comparison this
experiment exists for is unaffected. It does make exp04 the only experiment not
at fp32; exp00 is what licenses that.

One A100 argument does survive and is not settled here: the **analysis** pass
runs the template readout over all 2.17 M edges on the GPU, and that is VRAM
rather than host RAM. If `-o test_plot` hits a CUDA out-of-memory error, the
answer is `queue: gpu_a100` on the arm, not more slots.

<!-- STATUS:BEGIN -->

## Status

**15/15 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm |  | n | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current |  | 5 | 0.997 ± 0.000 | 0.996 ± 0.000 |  |  | 0.889 ± 0.013 (0.0) | 0.963 ± 0.023 (0.3) | 0.537 ± 0.068 (18.8) | 0.242 ± 0.064 | 0.943 ± 0.009 (0.0) | 0.062 ± 0.007 |  | 0.821 ± 0.007 |
| conductance |  | 5 | 0.996 ± 0.001 | 0.995 ± 0.000 |  |  | 0.922 ± 0.036 (0.0) | 0.963 ± 0.005 (1.3) | 0.519 ± 0.169 (20.8) | 0.305 ± 0.131 | 0.911 ± 0.082 (2.4) | 0.068 ± 0.051 |  | 0.840 ± 0.007 |
| cond_l25 |  | 5 | 0.996 ± 0.000 | 0.797 ± 0.395 |  |  | 0.945 ± 0.019 (0.0) | 0.965 ± 0.010 (0.9) | 0.420 ± 0.062 (16.2) | 0.337 ± 0.075 | 0.949 ± 0.009 (0.2) | 0.021 ± 0.010 |  | 0.831 ± 0.026 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm |  | iter | one-step r | rollout r | fit roll own form | fit roll other form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_noise_005_null400_cur_cv00` | landed | 1,520,001 | `aeb17d0a3c57` |  |
| `flyvis_noise_005_null400_cur_cv01` | landed | 1,520,001 | `aeb17d0a3c57` |  |
| `flyvis_noise_005_null400_cur_cv02` | landed | 1,520,001 | `aeb17d0a3c57` |  |
| `flyvis_noise_005_null400_cur_cv03` | landed | 1,520,001 | `aeb17d0a3c57` |  |
| `flyvis_noise_005_null400_cur_cv04` | landed | 1,520,001 | `aeb17d0a3c57` |  |
| `flyvis_noise_005_null400_condl10_cv00` | landed | 1,520,001 | `aeb17d0a3c57` |  |
| `flyvis_noise_005_null400_condl10_cv01` | landed | 1,520,001 | `aeb17d0a3c57` |  |
| `flyvis_noise_005_null400_condl10_cv02` | landed | 1,520,001 | `aeb17d0a3c57` |  |
| `flyvis_noise_005_null400_condl10_cv03` | landed | 1,520,001 | `aeb17d0a3c57` |  |
| `flyvis_noise_005_null400_condl10_cv04` | landed | 1,520,001 | `aeb17d0a3c57` |  |
| `flyvis_noise_005_null400_condl25_cv00` | landed | 1,520,001 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_null400_condl25_cv01` | landed | 1,520,001 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_null400_condl25_cv02` | landed | 1,520,001 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_null400_condl25_cv03` | landed | 1,520,001 | `64c8a3b9e6ca` |  |
| `flyvis_noise_005_null400_condl25_cv04` | landed | 1,520,001 | `64c8a3b9e6ca` |  |

<!-- STATUS:END -->

