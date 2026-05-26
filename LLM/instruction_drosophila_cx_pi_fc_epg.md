# Drosophila CX — Path Integration (FC RNN, EPG-only readout)

## Goal

Find the best training recipe for the **fully-connected** path-integration
RNN with the EPG-only readout. The recurrent matrix is dense
(`wrec_param: column_dale` — Dale's law per pre-column, no connectome
topology mask) and W_out reads only from the 46 EPG neurons
(`graph_model.output_from_epg_only: true`).

**Primary metric**: `r_roll_1k` ≥ 0.95 at T=1000.

**Dataset (fixed)**: `drosophila_cx_pi_task`.

**Why this exploration is needed**: the standard 10-epoch DAL=1 recipe
(parent: [config/drosophila_cx/drosophila_cx_pi_fc_epg.yaml](../config/drosophila_cx/drosophila_cx_pi_fc_epg.yaml))
fails outright — current run at iter ~12k shows
`r_roll_1k = -0.15` at epoch 8 (T=900). The dense W_rec gives the
optimiser much more freedom than the connectome-locked Known-ODE, and
with the EPG-only readout the previous "EPGt-silent" solution
(Fig 10c-d) is structurally impossible, so the FC variant needs a
different recipe than what the legacy `pi_fc.yaml` provides.

## Budget

**60 iterations**, 5 slots × 4 batches = 20 iter/block, **3 blocks**.

**Per-iter training**: `n_epochs: 5`, `data_augmentation_loop: 1`,
`n_steps_schedule: [100, 200, 300, 400, 500]` — slow ramp from T=100
up to T=500 (jumping to T=500 in one epoch starves bump formation).

## Parent recipe (iter 0 / B1 slot 0)

```yaml
dataset: drosophila_cx_pi_task
graph_model:
  signal_model_name: drosophila_cx_pi
  input_proj: matrix
  output_proj: matrix
  velocity_gate: pen_4scalar
  lock_edge_signs: true
  wrec_param: column_dale           # dense W_rec, per-col Dale signs
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
  coeff_norm_floor: 0.5             # FC needs anatomy regulariser
  kappa_norm_floor: 0.05
  n_steps_schedule: [100, 200, 300, 400, 500]   # slow ramp, NOT one-shot T=500
  coeff_tail_loss: 0.05             # ON from the start — essential late-T regulariser
```

## Block plan

### B1 — Stabilisation + tail-loss sweep (iter 1-20)

Dense W_rec at T=500 is unstable without strong regularisation, and
tail-loss is essential for late-T tracking — probe both axes jointly.

| axis | mutations |
|---|---|
| `coeff_tail_loss` | **{0.0, 0.02, 0.05, 0.1}** — primary axis for B1 |
| `lr_W_rec` | {5e-4, 1e-3, 2e-3, 4e-3} |
| `grad_clip_W` | {0.5, 1.0, 2.5, 5.0} — tighter than Known-ODE |
| `noise_recurrent_level` | {0.0, 0.01, 0.05, 0.1} |
| `coeff_norm_floor`, `kappa_norm_floor` | {(0, 0), (0.5, 0.05), (1.0, 0.1)} |

Best slot from each batch becomes the parent. After B1, freeze
`coeff_tail_loss` at the winner for B2 / B3.

### B2 — wrec_param + L1 sparsity (iter 21-40)

Once stable, probe the recurrent parameterisation:

| axis | mutations |
|---|---|
| `wrec_param` | {column_dale, edge_free} — does Dale's law help or hurt? |
| `coeff_W_L1` | {0, 1e-5, 1e-4, 1e-3} — pull W_rec toward sparse |
| `w_init_scale` | {0.01, 0.05, 0.1} |
| `lr_W_rec_schedule` (single value) | refine around B1 winner |

### B3 — Robustness / fine-tune (iter 41-60)

Re-probe around the B2 winner. Verify stability across seeds; revisit
`coeff_tail_loss` ±50% in case B2 shifted the optimum.

| slot | mutation around B2 parent |
|---|---|
| 0 | reference |
| 1 | `lr_W_rec` ×0.7 |
| 2 | `lr_W_rec` ×1.4 |
| 3 | `coeff_tail_loss` ±50% |
| 4 | `coeff_W_L1` ±1 step in the B2 grid |

## Mutation guardrails

- Keep `output_from_epg_only: true`. The decoder shape is what we're
  characterising.
- Dataset fixed (`drosophila_cx_pi_task`). Never regenerate.
- `wrec_param: edge_magnitude` is invalid for FC (no connectome
  template); only `column_dale` and `edge_free` are valid.
- Keep `n_steps_schedule` as a multi-epoch slow ramp
  (length == `n_epochs`). `lr_W_rec_schedule` length must match.

### Anti-repetition rule (important — saves budget)

- **At most ONE slot per batch is the parent-reference**: in B1 batch 1
  that's slot 0. After each batch, the parent for the *next* batch is
  whichever slot won; **do not keep a frozen-baseline reference slot
  across batches**.
- In B2/B3 every slot should be a distinct mutation off the (re-chosen)
  parent. For stability checks, use a seed-varied slot rather than a
  literal duplicate.

## Stop conditions

- **Success**: `r_roll_1k ≥ 0.95` for two consecutive iterations.
- **Halt**: best slot stays `< 0.3` after B2 — FC + EPG readout may
  be fundamentally underdetermined; report and stop.
