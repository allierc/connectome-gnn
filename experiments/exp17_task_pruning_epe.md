---
number: 17
name: task_pruning_epe
title: Silent edges, silent neurons and both pruned from flyvis network 0000/000 and from
  the circuit the general-form GNN recovered from it, read on the optic-flow task's EPE
purpose: validate three pruning analyses -- remove the silent edges, remove the silent
  neurons, remove both -- increasingly, each cut validated by keeping the task-training
  metric constant (EPE, the optic-flow end-point error of the flyvis paper), on the
  ground-truth flyvis model 000 (best EPE, 5.1) and on the learned model (experiment 2's
  general-form GNN MLP(ai,aj,vi,vj), group lasso 25, model noise 0.05, Table 1); prune
  up to EPE 5.4
baseline: none -- no training; flyvis flow/0000/000 and the five experiment-2 runs below
specs_dir: experiments/specs/exp17/fly
task: none
queue: local
wall: '00:00'
axes:
  noise:
  - noise_005
  fold:
  - cv00
  - cv01
  - cv02
  - cv03
  - cv04
arms:
- id: cond_l25
  label: general-form GNN, group lasso 25 (experiment 2), read
  spec_pattern: flyvis_{noise}_blank50_condl25_{fold}
  submit: false
  differs_by: {}
report:
  slide_title: Pruning silent edges and neurons at constant EPE
  arm_labels:
    cond_l25: GNN MLP(ai,aj,vi,vj), lasso 25
---

# Experiment 17 --- pruning at constant task error

## Question (Cedric, 2026-10-12)

"Validate three pruning analyses: remove the silent edges, remove the silent neurons,
remove both, increasingly, to be validated with keeping the simulation training metric
constant." The metric is the task-training one: the end-point error (EPE) of the decoded
optic flow (Lappalainen et al. 2024). The ground truth is flyvis model 000, the one with
the best EPE; the learned model is experiment 2's general-form GNN at model noise 0.05
(Table 1); pruning goes up to EPE 5.4.

Precedent: Plexus experiment 17, batch 23 (slide 55 of its site, "the lattice grid,
pruned"): edges removed level by level, each threshold kept only while the free
rollout's score moved less than the seed spread.

## Models

| model | what | parameters |
|---|---|---|
| ground truth | flyvis `flow/0000/000`, published | the network experiment 2's data was generated from (`SimulationConfig.ensemble_id` 0000, `model_id` 000, central 217 columns) |
| learned, cv00--cv04 | `results/learned_ode_params.pt` of `flyvis_noise_005_blank50_condl25_cv0N` | W per edge (434,112), tau and V_rest per neuron (13,741): the template readout scored in Table 1 |

Stored validation EPE of the 50 published members (`validation/epe.h5`, last entry):
000 = 5.1348, rank 1 of 50; median 5.4031, mean 5.4097, SD 0.1024. The 5.4 bound is
the median member's error, 0.27 above 000's (2.6 ensemble SD).

## Design

**EPE.** `flyvis.task.objectives.epe` on the held-out Sintel clips of 000's own task
config with `original_split` forced (the published config predates the key; without it
flyvis draws a random split sharing 6 of the 16 validation sequences, and 000 scores
2.0896 on it), through the loop of `MultiTaskSolver.test`: steady state (0.25 s at 0.5),
the network, `decoder["flow"]`; mean over the validation batches. This is
`evaluate_checkpoints` in `src/flyvis_conductance_optical_flow/analyze.py`. Recomputed for
every model, the ground truth included, never read from `epe.h5`.

**Learned model into flyvis.** flyvis shares every parameter: an edge's weight is
`sign_e * syn_count_e * syn_strength[source type, target type]`; `time_const` and `bias`
(V_rest) are per cell type. The learned strength of a pair is the least-squares fit of
`W_e = sign_e * syn_count_e * strength` over the pair's 217-column edges (count-weighted);
`time_const` and `bias` are the medians of `tau_i`, `V_rest_i` over the type's neurons.
These go into the 721-column network (extent 15) that 000's
decoder was trained with; the decoder is 000's, unchanged. A strength that comes out
negative (the learned W flipped the connectome sign on most of a pair's edges) is clamped
to 0, as flyvis's `non_negative` clamp would; the count is reported.

**Silent**, measured on each model's own activity over the TRAINING clips (the EPE is
read on the held-out ones):

- neuron output `s_i` = mean of `relu(v_i)`, what neuron i sends, over every step the
  network simulates for a clip: the initial state (activity = bias), the 12 steps of the
  0.25-s grey-screen warm-up, then the clip's frames;
- edge traffic `m_e` = `|weight_e| * s_source(e)`, the mean magnitude of the message.

**The three analyses**, each a ladder of rungs q = exact zeros, then 5, 10, 20, ..., 90 %:

| analysis | removed at rung q |
|---|---|
| edges | edges with `m_e` below the q-quantile of `m` |
| neurons | neurons with `s_i` below the q-quantile of `s`: every edge into and out of them |
| both | the union of the two at the same q |

Removed = weight 0. The exact zeros are removed at every rung, the quantile set on top: a
learned model's strengths clamped to 0 make ~12 % of its edges exactly silent, more than
the 5 % rung. The photoreceptors R1--R8 (the stimulus input) are never removed.

**Acceptance.** The task gain of a model is the EPE it saves over predicting zero flow
(Z = 5.7742 on the held-out clips, the score of having learned nothing). The 5.4 bound
keeps (Z - 5.4) / (Z - 5.1348) = 58.5 % of 000's gain; every model must keep that
fraction of its own: a rung is kept while EPE <= Z - 0.585 * (Z - unpruned EPE). For 000
the bound is 5.4. The ladder
stops at the first rejected rung and bisects three times between it and the last kept
one. Reported per model and analysis: the largest kept rung, the fraction of edges and of
neurons removed, its EPE. Also: the union of the edge and neuron sets each at its own
kept rung, and the cell types of the neurons between the neuron ladder's kept rung and
its first rejected one (the cliff).

**Masking without patching flyvis.** A `NetworkDynamics` subclass carrying an edge mask
(weight times mask), selected by class name and registered by import, as
`flyvis_conductance_optical_flow` does. `forward_subclass` falls back silently to the
base class, whose methods are `pass`, on an unknown name, so the run asserts the
constructed dynamics is the subclass.

## Gates (before any number is read)

1. The recomputed EPE of 000 equals its stored 5.1348.
2. The aggregation applied to the TRUE per-edge weights (`gt_weights.pt`) gives back
   000's `syn_strength` exactly: proves the edge order and the aggregation.
3. Mask all ones: EPE identical to gate 1. Mask removing only edges whose source output
   is exactly zero on every step of the held-out clips: EPE identical to gate 1.

Results (2026-10-12): gate 1 5.134818 against 5.1348 stored; gate 2 max |diff| 1.5e-8
over the 604 strengths, the edge order of our datasets is flyvis's extent-8 order (max
|weight diff| 0.0); gate 3 5.134818 with the mask of ones and with the 3.9 % exact-zero
edges removed. Gate 3 failed twice first, and both failures changed the design: silence
counted on the clip frames only removed edges from 14,565 sources that send during the
warm-up (10,562 of them in the grey-screen steps, all 14,565 at the initial state, where
activity = bias > 0); EPE 5.1437, then 5.1440.

## Decisions

1. Silent = low mean rectified output (what is sent), not low variance. A tonically
   active neuron sends a constant drive; removing it shifts its targets' baseline, so it
   is not silent here.
2. Per neuron and per edge of the 721-column network, not per cell type: border columns
   are free to be silent where the centre is not.
3. Learned strengths by count-weighted least squares, not the median of
   `W_e / (sign_e * count_e)`: low-count edges dominate the median. Per-edge R² of the
   weights rebuilt from the shared strengths against the true ones, folds cv00-cv04:
   least squares 0.970-0.974, median 0.965-0.969 (the raw learned W: 0.963-0.969, scored
   as 1 - SSE/SST; Table 1's 0.982-0.989 is the squared correlation). 81-90 of the 604
   strengths come out negative and are clamped to 0.
4. The bound is a fraction of each model's task gain (58.5 %, the fraction 5.4 keeps of
   000's). An absolute 5.4 prunes nothing on experiment 2's circuits, which score
   5.31-5.58 unpruned. A fixed EPE tolerance (+0.265, tried first) put cv04's bound at
   5.847, above Z = 5.7742: its EPE saturates at 5.77-5.78 however much is removed, and
   removing 90 % of its neurons passed. 000 has no seed spread of its own; the 50-member
   SD (0.10) is reported beside it.
5. Local GPU: inference only, no training, no cluster job.
6. Silence on the training clips, EPE on the held-out ones: the pruning sets are not
   chosen on the clips that score them.

## Specs

None. The `cond_l25` arm reads experiment 2's five `noise_005` folds; nothing is
launched.

## Code

`tools/task_pruning.py` (`--gates`; `--model gt|cv00..cv04`): builds the ground-truth and
learned networks, runs the gates, the three ladders per model, and writes
`GraphData/log/fly/exp17_task_pruning/<model>/pruning.json` (one row per rung) and
`<model>.log`. Silence is one pass over the 51 training clips (fixed order, none dropped);
each rung one pass over the 16 held-out clips, under a second on an RTX A6000.

## Log

- 2026-10-12: file written; gates passed after three design changes (original split,
  silence including the warm-up and the initial state, least-squares strengths); first
  ladders run, then the bound made relative (cv00 unpruned 5.4999), then a fraction of the
  task gain (a fixed tolerance let cv04 pass at 90 % of its neurons removed); the exact
  zeros removed at every rung (skipping the rungs below them emptied the "both" ladder of
  the learned models).
- 2026-10-12: all six ladders rerun on the final design; results table and
  presentation/Fig/exp17_pruning.png written by `--report`; report.pdf slide 41.

<!-- PRUNING:BEGIN -->

## Results (written by `tools/task_pruning.py --report`)

Largest kept rung per analysis: % of the 1,513,231 edges removed / % of the non-photoreceptor neurons removed / held-out EPE. Task gain = the EPE saved over predicting zero flow (5.7742); bound = keep 58.5% of the model's own gain, 5.4 for 000.

| model | EPE unpruned | task gain vs 000 | bound | edges: % edges / % neurons / EPE | neurons | both, same rung | union of the kept sets | strengths < 0 |
|---|---|---|---|---|---|---|---|---|
| flyvis 000 | 5.1348 | 100% | 5.4000 | 50.0% / 0.0% / 5.378 | 5.1% / 5.6% / 5.145 | 9.6% / 5.6% / 5.145 | 51.2% / 5.6% / 5.378 | -- |
| cv00 | 5.4999 | 43% | 5.6137 | 48.7% / 0.0% / 5.610 | 7.8% / 5.0% / 5.580 | 18.0% / 5.0% / 5.580 | 50.2% / 5.0% / 5.638 (over) | 89 of 604 |
| cv01 | 5.4593 | 49% | 5.5899 | 46.2% / 0.0% / 5.571 | 5.4% / 5.6% / 5.487 | 17.8% / 5.6% / 5.487 | 47.6% / 5.6% / 5.596 (over) | 90 of 604 |
| cv02 | 5.3078 | 73% | 5.5013 | 47.5% / 0.0% / 5.479 | 5.8% / 4.4% / 5.439 | 15.9% / 4.4% / 5.439 | 48.6% / 4.4% / 5.547 (over) | 83 of 604 |
| cv03 | 5.3248 | 70% | 5.5112 | 55.0% / 0.0% / 5.499 | 5.0% / 5.6% / 5.319 | 16.9% / 5.6% / 5.319 | 56.3% / 5.6% / 5.492 | 90 of 604 |
| cv04 | 5.5821 | 30% | 5.6618 | 47.5% / 0.0% / 5.615 | 3.8% / 3.7% / 5.582 | 14.5% / 3.7% / 5.582 | 48.2% / 3.7% / 5.615 | 81 of 604 |

Cell types of the neurons between the neuron ladder's kept rung and its first rejected rung (count of neurons):

- flyvis 000: Tm4 188, Lawf1 42, T4d 15, L5 5
- cv00: Mi15 125, Tm4 120, L5 4
- cv01: Tm4 153, Am 83, T4d 14
- cv02: Tm4 249
- cv03: Tm4 194, L2 56
- cv04: Tm4 181, L2 69

<!-- PRUNING:END -->

<!-- STATUS:BEGIN -->

## Status

**5/5 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | noise | n | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_l25 | noise_005 | 5 | 0.999 ± 0.000 | 0.998 ± 0.000 | 0.979 ± 0.001 (0.0) | 0.654 ± 0.055 (2.0) | 0.989 ± 0.004 (0.0) | 0.980 ± 0.004 (0.0) | 0.903 ± 0.017 (3.1) | 0.903 ± 0.017 | 0.990 ± 0.002 (0.0) | 0.006 ± 0.001 | 0.340 ± 0.003 | 0.895 ± 0.013 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | noise | iter | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_noise_005_blank50_condl25_cv00` | landed | 1,520,001 | `7278aa10679f` |  |
| `flyvis_noise_005_blank50_condl25_cv01` | landed | 1,520,001 | `7278aa10679f` |  |
| `flyvis_noise_005_blank50_condl25_cv02` | landed | 1,520,001 | `7278aa10679f` |  |
| `flyvis_noise_005_blank50_condl25_cv03` | landed | 1,520,001 | `69d59d4e6a93` |  |
| `flyvis_noise_005_blank50_condl25_cv04` | landed | 1,520,001 | `7278aa10679f` |  |

<!-- STATUS:END -->
