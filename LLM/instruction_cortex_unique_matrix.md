# Cortex Multi-Task Capacity Sweep — Matrix Encoder / Decoder

## Goal

Find the **smallest W_rec size** at which a SINGLE matrix-mode
`CortexTaskRNN` (free-W mode) can saturate **R²** on the full 20-task
Yang battery. All 20 slots train the same multi-task model on
`task_cortex_all`; the only per-slot variation is `graph_model.n_units`
∈ {256, 512, 1024}. Mutations apply uniformly across all slots so we
can see whether the saturation curve shifts under different
hyperparameter regimes.

Sister loop to `instruction_cortex_matrix.md` (per-task, 20 separate
RNNs). This loop asks: **does one shared RNN match 20 specialists, and
at what capacity?**

Anchor numbers:
- **Single-task ceiling** (cortex_delaygo_sigma at 256 units, batch_size 64):
  R² = 0.989, dir_acc = 0.93 at 6,000 iters. That's the per-task winner
  config taken solo on one rule.
- **Per-task loop** (cortex_matrix, sister exploration): each rule gets
  its own RNN. Expected per-task R² ≥ 0.95 (easy) to ≥ 0.80 (hard).
- **This loop's question**: can one matrix-mode RNN at n_units ∈
  {256, 512, 1024} reach mean R² across 20 tasks comparable to the
  20-specialist baseline, or does multi-task interference cap it?

## Loop semantics (READ THIS)

20-slot capacity sweep, NOT per-task:

- All 20 slots train an **identical multi-task RNN** on the same
  `task_cortex_all` dataset (20k train / 4k test trials sampled
  uniformly across the 20 Yang rules).
- The **only difference between slots** is `graph_model.n_units`:

| Slot range | `n_units` | Seeds |
|----:|----:|---|
| 0–6   | **256**  | 7 different sim/train seeds |
| 7–13  | **512**  | 7 different sim/train seeds |
| 14–19 | **1024** | 6 different sim/train seeds |

- Block 1 is a clean **baseline robustness measurement** of the
  3-size sweep (no mutations). Subsequent blocks adjust **one family of
  knobs** uniformly across all slots and re-run the sweep to test
  whether that knob changes the saturation curve.
- Hyperparameter mutations are applied **uniformly to all 20 slots**.
  Never edit `n_units` (it's part of the experimental design, not a
  free knob), and never edit `dataset` / `task.cortex.*` (locked to
  `task_cortex_all`).

128 was dropped from the original 4-size sweep: matrix proj is likely
capacity-bound at 128 units on the multi-task battery regardless of
hyperparameters, so seeds there are mostly wasted. The 20-slot batch
is bottlenecked by the 1024 slots' wall time in every layout, so
concentrating budget at 256/512/1024 gives more informative seeds at
no batch-time cost.

## Slot ↔ n_units mapping (fixed, pre-seeded)

| Slot | n_units | Slot | n_units | Slot | n_units |
|---:|---:|---:|---:|---:|---:|
| 0 | 256 | 7  | 512 | 14 | 1024 |
| 1 | 256 | 8  | 512 | 15 | 1024 |
| 2 | 256 | 9  | 512 | 16 | 1024 |
| 3 | 256 | 10 | 512 | 17 | 1024 |
| 4 | 256 | 11 | 512 | 18 | 1024 |
| 5 | 256 | 12 | 512 | 19 | 1024 |
| 6 | 256 | 13 | 512 |    |      |

## Baseline (Block 1, canonical-winner config)

Carries cortex_delaygo_sigma's hyperparameters, scaled for multi-task:

| Field                  | Value                                              | Source |
|------------------------|----------------------------------------------------|--------|
| `recurrent_activation` | tanh                                               | cortex_delaygo loop |
| `readout_uses_sigma`   | **true**                                           | cortex_delaygo A/B finding |
| `input_proj`           | **matrix** (pinned)                                | this loop's premise |
| `output_proj`          | **matrix** (pinned)                                | this loop's premise |
| `n_units`              | **256 / 512 / 1024** (per-slot)                    | experimental axis |
| `w_init_mode`          | randn_scaled                                       | delaygo loop |
| `w_init_scale`         | 0.5                                                | delaygo loop |
| `lr`                   | 2e-3 (biases)                                      | cortex_delaygo canonical winner |
| `lr_W_rec_schedule`    | `[2e-3, 2e-3, 1e-3, 1e-3, 4e-4, 4e-4, 2e-4×4]` (slower-decay × 2.0, 10 entries) | extended to 10 epochs for multi-task |
| `lr_W_ED_schedule`     | same as `lr_W_rec_schedule` (tied)                 | start tied; decouple via Block 2 |
| `grad_clip_W`          | 2.0                                                | delaygo loop |
| `coeff_rate_L2`        | 1e-2                                               | delaygo loop |
| `coeff_W_L2`           | 0.0                                                | silent killer in delaygo |
| `noise_recurrent_level`| 0.0                                                | neutral in delaygo |
| `n_epochs`             | **10**                                             | matches schedule length |
| `batch_size`           | **128**                                            | multi-task default (vs 64 in single-task) — larger dataset, want stable per-task gradient |
| `data_augmentation_loop` | 80                                               | gives 20000/128 × 80 ≈ 12500 iter/epoch × 10 ep ≈ 125k iter/slot |

Expected per-slot baseline:
- **n_units=256**: mean R² ≈ 0.75-0.85 (multi-task interference shows)
- **n_units=512**: mean R² ≈ 0.80-0.90
- **n_units=1024**: mean R² ≈ 0.85-0.92

If 256 floors at R² < 0.70 the matrix decoder is capacity-bound there
and the saturation point lives at ≥ 512. If 1024 is no better than 512,
the matrix decoder is the bottleneck and recurrent capacity isn't the
limiting factor.

## Sweepable hyperparameter families

Apply each mutation **uniformly to all 20 slots** (n_units stays fixed
per slot). The exploration tests *families* of knobs, not single
parameters in isolation.

### LR family

| Field                    | Default                                                                     | Sweep values                                                                                  |
|--------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `lr_W_rec_schedule[0]` (peak) | 2e-3                                                                   | {1e-3 (half-peak), 2e-3 *parent*, 5e-3 (high-peak)}                                           |
| `lr_W_rec_schedule` shape | slower-decay × 2.0 (parent)                                                | {steeper (3-ep tail), cosine, longer-plateau (3 ep at peak)}                                  |
| `lr_W_ED_schedule`       | tied to w_rec (parent)                                                      | {tied *parent*, decoupled_low `[5e-4, 2e-4, 1e-4 × 8]`, decoupled_high `[3e-3 × 2, 1e-3 × 2, 4e-4 × 6]`} |

### Regularisation family

| Field           | Default | Sweep values            |
|-----------------|---------|-------------------------|
| `coeff_rate_L2` | 1e-2    | {0, 1e-3, 1e-2 *parent*, 1e-1} |
| `coeff_W_L2`    | 0.0     | {0 *parent*, 1e-6, 1e-5} |
| `noise_recurrent_level` | 0.0  | {0 *parent*, 1e-3, 1e-2} |
| `grad_clip_W`   | 2.0     | {1.0, 2.0 *parent*, 5.0} |

### Scheduler shape family

| Field    | Default                              | Sweep values                                                          |
|----------|--------------------------------------|-----------------------------------------------------------------------|
| schedule | slower-decay × 2.0 (2-ep plateau)    | {cosine annealing, 3-ep plateau then geometric ×0.5, linear-decay}    |

### Architecture (mostly pinned)

| Field                  | Default | Sweep values             | Notes |
|------------------------|---------|--------------------------|-------|
| `recurrent_activation` | tanh    | {tanh *parent*, softplus} | sigmoid was worst in delaygo |
| `readout_uses_sigma`   | true    | {true *parent*, false}    | verify the cortex_delaygo finding generalises across n_units |
| `input_proj`           | matrix  | **pinned** | the point of the loop |
| `output_proj`          | matrix  | **pinned** | the point of the loop |
| `n_units`              | per-slot| **pinned** (256/512/1024) | the experimental axis |

## Things you must NOT change

- `dataset` (locked to `task_cortex_all`).
- `task.cortex.*` (locked to all 20 rules).
- `graph_model.signal_model_name` (`cortex_all`).
- `graph_model.W_param` (must stay `free`).
- `graph_model.n_units` (**part of the experimental design**, pre-seeded per slot).
- `graph_model.input_proj` / `output_proj` (must stay `matrix`).
- `n_input` / `n_output` (fixed by Yang data shape: 85 / 33).

## Metrics

Each slot's `tmp_training/metrics.log` writes:
```
iteration,epoch,loss,mse,motor_max,motor_peak_mean,r2,direction_acc
```

Per-slot R² = the slot's multi-task aggregate (mean across all 20 rules
in that slot's eval batch — NOT per-task R², since one model trains all
20).

### Analysis dimensions (per batch)

For each **(n_units, mutation)** cell, you have 6-7 seeds. Report (on
the primary R²):

| Stat | Formula | Use |
|---|---|---|
| **mean** | mean(r2 across seeds at that n_units) | aggregate at that size |
| **std**  | std(r2 across seeds) | seed sensitivity |
| **ceiling** | max(r2 across seeds) | best-case capability |

Look for the **smallest n_units where mean R² approaches the ceiling
and std drops**. That's the saturation point.

### Comparison anchors

| Anchor                                                     | R² (mean) | Source |
|------------------------------------------------------------|-----------|--------|
| Matrix proj, old constant-LR + n_epochs=1                   | ≈ 0.69    | LLM_cortex_all_matrix block 1 (n=80, pre-rebuild) |
| cortex_delaygo_sigma single-task at 256 units               | **0.989**     | one-rule single-seed run |
| cortex_matrix per-task loop average (target)                | ~0.85-0.95 | 20 separate RNNs, one per task |
| MLP unique loop (instruction_cortex_all_unique)             | open       | sister loop with MLP encoder/decoder |

A Block-1 mean R² ≥ 0.85 at 1024 units would mean matrix proj + sigma=true +
slower-decay schedule has caught up enough that multi-task interference is
the main remaining bottleneck.

## Block plan

20 slots/batch. **Iterations: 280 total / 14 batches** (~5.8 h wall on
a100 — bottlenecked by 1024-unit slots at ~25 min/slot vs ~12 min at
256). Each block is an exploration step covering a *family* of knobs in
one pass.

| Block | Focus                                          | Batches × profiles                                                                                                  | Iters |
| ----- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----: |
| 1     | **Baseline** (cortex_delaygo_sigma starter)    | 2 batches of the baseline config, different forced seeds.                                                           | 40    |
| 2     | **Sweep all LR** (peak × shape × decoupling)   | 3 batches, each a different multi-knob LR profile.                                                                  | 60    |
| 3     | **Sweep all regularisation**                   | 3 batches: rate_L2 + W_L2 + noise + grad_clip combos.                                                               | 60    |
| 4     | **Fine-tune scheduler shape**                  | 2 batches: cosine, longer-plateau (around the Block-2 winner peak).                                                  | 40    |
| 5     | **CV final on winner**                         | 4 batches of the combined winner — different forced seeds.                                                          | 80    |
| **Total** |                                            | **14**                                                                                                              | **280** |

### Block 2 profiles (LR sweep — multi-knob)

| Batch | `lr_W_rec[0]` | `lr_W_rec_schedule` shape | `lr_W_ED_schedule` | Hypothesis |
| --- | --- | --- | --- | --- |
| 2.1 | **1e-3** (half-peak) | proportional shrink of parent | match w_rec | "Larger N may favour smaller updates per step." |
| 2.2 | **5e-3** (high-peak) | steeper decay (3-epoch tail) | match w_rec | "Bigger updates early, anneal faster — finds basin quickly." |
| 2.3 | 2e-3 (parent)        | parent shape | **decoupled_low** `[5e-4, 2e-4, 1e-4 × 8]` | "The (n_input, n_output) ED matrices are tiny vs N² — they want lower lr, especially at 1024." |

### Block 3 profiles (regularisation sweep — multi-knob)

| Batch | `coeff_rate_L2` | `coeff_W_L2` | `noise_recurrent_level` | `grad_clip_W` | Hypothesis |
| --- | --- | --- | --- | --- | --- |
| 3.1 | **0** | 0 | 0 | 5.0 | "No-reg baseline — does sigma=true alone keep activity bounded at 1024?" |
| 3.2 | 1e-2 (parent) | **1e-6** | 0 | 2.0 | "Mild W_L2 might lift the 1024 ceiling at converged training." |
| 3.3 | **1e-1** | 0 | **1e-3** | 2.0 | "Stronger rate L2 + noise — does it help the regression on dm1, dmsgo (the canaries)?" |

### Block 4 profiles (scheduler fine-tune)

Use the Block-2 winner's peak LR.

| Batch | Schedule shape | Hypothesis |
| --- | --- | --- |
| 4.1 | **Cosine annealing** from peak to 1e-5 over 10 epochs | "Smooth annealing avoids step-discontinuity perturbations." |
| 4.2 | **3-epoch plateau** at peak, then geometric ×0.5/epoch | "Longer high-LR exploration may help 1024 fill its capacity." |

### Block 5 (CV final)

4 batches × 20 slots = 80 runs at the winner config, different pipeline-
forced seeds per batch. Cumulative seed counts per size after the full
exploration:

| Size | Block 1 | Blocks 2-4 | Block 5 | Total seeds |
| --- | ---: | ---: | ---: | ---: |
| 256 (slots 0-6)   | 14 | 56 | 28 | **98** |
| 512 (slots 7-13)  | 14 | 56 | 28 | **98** |
| 1024 (slots 14-19)| 12 | 48 | 24 | **84** |

→ Per-size std/√n ≈ ±0.002 on mean R² — tight enough to declare the
smallest saturating n_units.

## Exploration rule

Each block tests a *family* of knobs. Within a block, batches test
different **multi-knob coordinated profiles** — one profile per batch,
applied uniformly to all 20 slots. Across blocks, the winner from
block B becomes the parent for block B+1.

The per-slot n_units assignment is fixed across all blocks — slot 0 is
always n_units=256, slot 14 is always n_units=1024, etc. Mutations
only touch the shared hyperparameters in `training:` / `graph_model:`
(within the "sweepable" tables above).

In **CV mode** (Block 5), every batch has identical config except the
pipeline-forced seed.

## Mutation log format (per batch)

```
## Iter N (block B, batch K): [exploration | CV]
Mutation: <multi-knob profile description>   (applied to all 20 slots)
Hypothesis: "[testable claim — does the saturation curve shift]"

Per-size summary (mean ± std across seeds at each n_units):
  n_units=256:  mean=X.XXX  std=Y.YYY  ceiling=Z.ZZZ
  n_units=512:  mean=X.XXX  std=Y.YYY  ceiling=Z.ZZZ
  n_units=1024: mean=X.XXX  std=Y.YYY  ceiling=Z.ZZZ

Saturation diagnosis: [256 < 512 ≈ 1024 → 512 saturates]
                  OR  [256 ≈ 512 ≈ 1024 → 256 saturates]
                  OR  [256 < 512 < 1024 → 1024 isn't enough]

Late-stage check: dm1_peak=X.XXX@iter_J  dm1_final=Y.YYY
                  dmsgo_peak=X.XXX@iter_J  dmsgo_final=Y.YYY
Verdict: [supported | falsified | inconclusive]
Next mutation: <next block's profile>
```

## Winner config

At every block boundary, copy the best-mean config (smallest-saturating
n_units) to `config/cortex/cortex_unique_matrix_winner.yaml`:

```yaml
# Winner: cortex_unique_matrix_winner.yaml
# Source: iter_NNN (best n_units=K saturates at mean r2=X.XXX)
# Block: B  (focus: <focus>)
# Date: YYYY-MM-DD
#
# Saturation curve (mean r2):
#   n_units=256:  X.XXX
#   n_units=512:  X.XXX
#   n_units=1024: X.XXX
#
# Hyperparameters: <list mutations from baseline that helped>
# Anchors:
#   matrix old baseline: 0.691  (n=80, single LR, n_epochs=1)
#   cortex_delaygo_sigma single-task: 0.989 (1 rule, 1 seed)
```

## Notes / hints

- **`readout_uses_sigma: true` is load-bearing.** Single-seed A/B on
  cortex_delaygo showed sigma=true reaches R² = 0.93 in **one epoch**
  vs 0.49 for raw h. Verify in Block 6's sigma=false batch that this
  generalises across the multi-task setting at every n_units.
- **`claude.n_epochs: 10` is load-bearing.** The 10-entry slower-decay
  schedule only activates if `claude.n_epochs` ≥ schedule length.
- **batch_size 128** is the multi-task default (vs 64 for single-task).
  Bigger batches average gradients across more task identities, which
  stabilises per-task signal. Block 2's `lr_W_rec[0] = 1e-3` half-peak
  is partly a test of whether bigger batches want smaller LR.
- **1024-unit slots are the wall-time bottleneck** (~25 min/slot vs
  ~12 min at 256). The 1024-heavy seeds make 1024 the slowest cell to
  fill statistically.
- **Per-task R² is NOT available in this loop.** Each slot trains one
  multi-task RNN — its R² is the 20-task aggregate. If you want
  per-task breakdown, that's the sister loop `cortex_matrix`.
- **Saturation curve interpretation**: a *flat* curve (256 ≈ 512 ≈
  1024) means the matrix decoder is the bottleneck; an *increasing*
  curve means recurrent capacity matters; a *peaked* curve means
  something else (likely encoder/decoder gain) is the limiting factor.
