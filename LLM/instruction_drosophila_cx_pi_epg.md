# Drosophila CX — Path Integration (Known-ODE RNN, EPG-only readout)

## Goal

Find the best training recipe for `DrosophilaCxTaskRNN` with the
**EPG-only readout** flag (`graph_model.output_from_epg_only: true`)
on the path-integration task. The decoder W_out has shape `(2, 46)`
and reads only from the first 46 EPG neurons, matching the Hulse 2021
`wout[0:46, :]` convention. This is the connectome-constrained model
(`W_rec = |S| ⊙ sign(W_con)`, Dale's law).

**Primary metric**: `r_roll_1k` — Pearson correlation between the
unwrapped decoded heading and the ground-truth heading on a
deterministic-sweep rollout (ω = 60°/s constant, **T = 1000**,
warmup 10 frames). Target **`r_roll_1k` ≥ 0.95**.

No secondary metrics.

**Dataset (fixed)**: `drosophila_cx_pi_task` (100k train + 10k test,
T=1000 frames, dt=0.01s, written by ZarrTaskTrialsWriter).

**Why this exploration is needed**: the standard 10-epoch DAL=1
recipe (parent: [config/drosophila_cx/drosophila_cx_pi_epg.yaml](../config/drosophila_cx/drosophila_cx_pi_epg.yaml))
plateaus around `r_roll_1k` ≈ 0.85 instead of saturating at 0.999.
Restricting the decoder to 46 EPG neurons changes the gradient flow:
the optimiser now must push the heading code into the EPG block, which
exposes a different loss landscape than the all-156-neuron readout.

## Budget

**60 iterations**, 5 slots × 4 batches = 20 iter/block, **3 blocks total**.

**Per-iter training**: `n_epochs: 5`, `data_augmentation_loop: 1`,
`n_steps_schedule: [100, 200, 300, 400, 500]` — slow ramp from T=100
up to T=500 (jumping straight to T=500 in one epoch starves the early
bump-formation phase). Each iter ~2 h per slot on a100.

## Parent recipe (iter 0 / B1 slot 0)

```yaml
dataset: drosophila_cx_pi_task
graph_model:
  signal_model_name: drosophila_cx_pi
  input_proj: matrix
  output_proj: matrix
  velocity_gate: pen_4scalar
  lock_edge_signs: true
  wrec_param: edge_magnitude        # sign-locked connectome
  output_from_epg_only: true        # NEW — decoder reads only EPG
training:
  n_epochs: 5
  data_augmentation_loop: 1
  batch_size: 64
  lr: 2.0e-3
  lr_W_rec_schedule: [2.0e-3, 2.0e-3, 1.0e-3, 5.0e-4, 5.0e-4]
  lr_W_ED: 5.0e-4
  grad_clip_W: 2.5
  noise_recurrent_level: 0.05
  n_steps_schedule: [100, 200, 300, 400, 500]   # slow ramp, NOT one-shot T=500
  coeff_tail_loss: 0.05             # ON from the start — essential late-T regulariser
```

## Block plan

### B1 — Stabilisation + tail-loss sweep (iter 1-20)

Tail-loss is essential for late-T tracking; start with it ON and
sweep it jointly with the stabilisation axes. Establish a parent
recipe that consistently reaches `r_roll_1k` ≥ 0.7 on T=500 training.

| axis | mutations to try |
|---|---|
| `coeff_tail_loss` | **{0.0, 0.02, 0.05, 0.1}** — primary axis for B1 |
| `lr_W_rec` (single value, n_epochs=1) | {5e-4, 1e-3, 2e-3, 4e-3} |
| `noise_recurrent_level` | {0.0, 0.01, 0.05, 0.1} |
| `grad_clip_W` | {0.0, 1.0, 2.5, 5.0} |
| `lr_W_ED` | {1e-4, 5e-4, 1e-3} |

Best slot from each batch becomes the parent of the next. After B1
the `coeff_tail_loss` value is fixed (winner-take-all) for B2 / B3.

### B2 — Curriculum + initialisation (iter 21-40)

Once a stable parent is found, probe:

| axis | mutations |
|---|---|
| `n_steps_schedule[0]` (single T_epoch) | {200, 300, 500, 700, 1000} — does training at full T=1000 help, or is shorter+extrapolate better? |
| `w_init_mode` | {const, randn, w_con} |
| `w_init_scale` | {0.01, 0.05, 0.1, 0.5} |
| `coeff_W_L1` | {0, 1e-5, 1e-4} |

### B3 — Robustness / fine-tune (iter 41-60)

Re-probe the lr × grad_clip × noise grid around the B2 winner; verify
stability across seeds; revisit `coeff_tail_loss` ±50% in case the
B2 changes shifted the optimum.

| slot | mutation around B2 parent |
|---|---|
| 0 | reference (no change) |
| 1 | `lr_W_rec` ×0.7 |
| 2 | `lr_W_rec` ×1.4 |
| 3 | `coeff_tail_loss` ±50% |
| 4 | `noise_recurrent_level` shifted ±1 step in the B1 grid |

Best by **r_roll_1k** averaged over the last 3 iter snapshots.

## Mutation guardrails

- Don't touch `lock_edge_signs`, `velocity_gate`, `wrec_param`,
  `signal_model_name`, or `output_from_epg_only`. These are fixed
  for this variant.
- The dataset is fixed (`drosophila_cx_pi_task`). Never call
  `data_generate`.
- Keep `n_steps_schedule` as a multi-epoch slow ramp (length == `n_epochs`).
  Don't collapse to single-T training.
- `lr_W_rec_schedule` length must match `n_epochs`.

### Anti-repetition rule (important — saves budget)

- **At most ONE slot per batch is the parent-reference**: in B1 batch 1
  that's slot 0. After each batch, the parent for the *next* batch is
  whichever slot won; **do not keep a frozen-baseline reference slot
  across batches**. The full per-epoch trajectory of the parent is
  already in the prior batch's logs, so a fresh re-anchor measurement
  per batch is wasted budget.
- In B2/B3 every slot should be a distinct mutation off the (possibly
  re-chosen) parent. If you want to verify stability of the current
  best, use a seed-varied slot (different `simulation_seed` /
  `training_seed`) rather than a literal duplicate of the parent.

## Stop conditions

- **Success**: best slot reaches `r_roll_1k ≥ 0.95` for two consecutive
  iterations.
- **Halt**: best slot stays below `r_roll_1k = 0.5` after B2 — implies
  the EPG-only readout is fundamentally incompatible with this model;
  report and stop.
