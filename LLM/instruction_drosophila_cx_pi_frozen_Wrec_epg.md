# Drosophila CX — Path Integration (Frozen-W_rec control, EPG-only readout)

## Goal

Push the **frozen-W_rec control** as far as possible with the EPG-only
readout. The recurrent matrix is held at
`Ŵ^rec ≡ W^con` (raw connectome, no learning); only the encoder
`W_in`, decoder `W_out`, biases, and the 4 PEN velocity scalars are
trainable. The decoder W_out has shape `(2, 46)` and reads only from
the 46 EPG neurons (`output_from_epg_only: true`).

**Why this exploration is needed**: with the standard recipe and the
all-156-neuron readout, the frozen-W_rec control fails outright
(`r_roll_1k = -0.37` in the paper). Preliminary results with the
EPG-only readout show this control unexpectedly reaches
`r_roll_1k ≈ 0.4` and `r_roll = +0.96` at T=100 — i.e., the EPG
readout partially rescues it. This exploration asks: **how far can it
go?** If r_roll_1k climbs above 0.9, the paper's "raw connectome alone
doesn't suffice" claim is readout-dependent; if it plateaus around
0.4-0.5, the rescue is partial and the original claim survives.

**Primary metric**: `r_roll_1k` at T=1000. No fixed target — characterise
the achievable plateau.

**Dataset (fixed)**: `drosophila_cx_pi_task`.

## Budget

**60 iterations**, 5 slots × 4 batches = 20 iter/block, **3 blocks**.

**Per-iter training**: `n_epochs: 5`, `data_augmentation_loop: 1`,
`n_steps_schedule: [100, 200, 300, 400, 500]` — slow ramp.

## Parent recipe (iter 0 / B1 slot 0)

```yaml
dataset: drosophila_cx_pi_task
graph_model:
  signal_model_name: drosophila_cx_pi
  input_proj: matrix
  output_proj: matrix
  velocity_gate: pen_4scalar
  lock_edge_signs: true
  wrec_param: edge_magnitude
  output_from_epg_only: true
training:
  n_epochs: 5
  data_augmentation_loop: 1
  batch_size: 64
  lr: 2.0e-3                        # biases only
  lr_W_ED: 5.0e-4                   # the main learning channel
  grad_clip_W: 2.5
  noise_recurrent_level: 0.05
  n_steps_schedule: [100, 200, 300, 400, 500]
  coeff_tail_loss: 0.05
  # NB: lr_W_rec_schedule unused — W_rec is frozen.
```

The frozen flag is set in the `data_train_task` dispatch (see
[src/connectome_gnn/models/drosophila_cx_task_rnn.py](../src/connectome_gnn/models/drosophila_cx_task_rnn.py));
the parent yaml inherits this from
[config/drosophila_cx/drosophila_cx_pi_frozen_Wrec_epg.yaml](../config/drosophila_cx/drosophila_cx_pi_frozen_Wrec_epg.yaml).

## Block plan

The search space is much smaller — only encoder/decoder/biases are
trainable. Don't over-explore.

### B1 — Stabilisation + tail-loss sweep (iter 1-20)

Same shape as the other variants but with a narrower axis list. The
only meaningful LR is `lr_W_ED` since W_rec is frozen.

| axis | mutations |
|---|---|
| `coeff_tail_loss` | **{0.0, 0.02, 0.05, 0.1, 0.2}** — primary axis |
| `lr_W_ED` | {1e-4, 5e-4, 1e-3, 2e-3, 5e-3} |
| `noise_recurrent_level` | {0.0, 0.01, 0.05, 0.1} |
| `grad_clip_W` | {0.0, 1.0, 2.5, 5.0} |
| `lr` (biases) | {1e-3, 2e-3, 5e-3} |

### B2 — Encoder / decoder init + PEN scalars (iter 21-40)

| axis | mutations |
|---|---|
| `velocity_gate` | {pen_4scalar, pen_only, none} — does freeing the velocity routing help? |
| `lr_W_ED_schedule` (single value) | annealing toward end-of-training |
| `batch_size` | {32, 64, 128} — does smoother gradient help? |
| PEN scalar init (in the trainer) | {default, ±0.05 wider, ±0.005 narrower} — if exposed |

If `pen_4scalar` is no longer optimal, that itself is a paper-relevant
finding for the frozen control.

### B3 — Curriculum + late-T regulariser (iter 41-60)

| axis | mutations |
|---|---|
| `n_steps_schedule[0]` | {200, 300, 500, 800, 1000} — does training at T=1000 directly help, or is short-T enough? |
| `coeff_tail_loss` | revisit ±50% around B1 winner |
| `coeff_W_L1` on encoder | {0, 1e-5, 1e-4} |
| `noise_recurrent_level` | refine around B1 winner |

## Mutation guardrails

- Don't touch `lock_edge_signs`, `wrec_param`, or
  `output_from_epg_only`. These are fixed.
- `lr_W_rec`/`lr_W_rec_schedule` are unused (W_rec frozen) — don't
  mutate them.
- Dataset fixed (`drosophila_cx_pi_task`).
- Keep `n_steps_schedule` as a multi-epoch slow ramp
  (length == `n_epochs`).

### Anti-repetition rule (important — saves budget)

- **At most ONE slot per batch is the parent-reference**: in B1 batch 1
  that's slot 0. After each batch, the parent for the *next* batch is
  whichever slot won; **do not keep a frozen-baseline reference slot
  across batches**.
- In B2/B3 every slot should be a distinct mutation off the (re-chosen)
  parent. For stability checks, use a seed-varied slot rather than a
  literal duplicate.

## Stop conditions

- **Conclusive success**: `r_roll_1k ≥ 0.9` for two consecutive iters.
  Reported as a paper-relevant finding: "with the EPG-only readout,
  the raw connectome alone suffices for path integration."
- **Conclusive plateau**: best slot stays in `[0.3, 0.7]` after B2.
  Reported as: "EPG-only readout partially rescues the frozen control
  but cannot reach behavioural saturation."
- **Conclusive failure**: best slot stays `< 0.2` after B2. Reported
  as: "Original 'raw connectome alone doesn't suffice' claim survives
  the readout change."
