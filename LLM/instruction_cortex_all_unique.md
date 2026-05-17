# Cortex 20-Task — W_rec Capacity Sweep (128 → 1024)

## Goal

Determine the **smallest W_rec size** at which a `TaskRNN` (free-W mode)
can saturate **R²** (coefficient of determination on the masked motor
readout — same definition as `models/graph_tester.py::stimuli_R2`) on
the full 20-task Yang battery. `direction_acc` is kept as a secondary
diagnostic. We expect
that the cortex_delaygo single-task winners (n_units ≈ 192–256) are
**under-capacity** for 20 tasks and that performance keeps improving as
n_units grows — but probably saturates somewhere between 512 and 1024.
Find the saturation point.

Secondary: characterise **per-task** saturation. Hard tasks
(`contextdelaydm*`, `dmc*`) likely need more units than easy ones
(`fdgo`, `reactgo`). The "smallest size where the *floor* task hits its
ceiling" is the operational answer for downstream papers.

## Loop semantics (READ THIS — different from earlier loops)

This is a **20-slot capacity sweep**, not per-task:

- All 20 slots train an **identical multi-task TaskRNN** on the same
  `task_cortex_all` dataset (20k train / 4k test trials sampled uniformly
  across the 20 Yang rules).
- The **only difference between slots** is `graph_model.n_units`:

| Slot range | `n_units` | Seeds (pipeline-forced)                                |
|----:|----:|-----------------------------------|
| 0–4   | **128**  | 5 different sim/train seeds       |
| 5–9   | **256**  | 5 different sim/train seeds       |
| 10–14 | **512**  | 5 different sim/train seeds       |
| 15–19 | **1024** | 5 different sim/train seeds       |

- Block 1 is a clean **baseline robustness measurement** of the 4-size
  sweep (no mutations). Subsequent blocks adjust **one** hyperparameter
  uniformly across all slots and re-run the sweep to test whether that
  knob changes the saturation curve.
- Hyperparameter mutations are applied **uniformly to all 20 slots**.
  Never edit `n_units` (it's part of the experimental design, not a
  free knob), and never edit `dataset` / `task.cortex.*` (locked to the
  cortex_all multi-task split).

## Slot ↔ size mapping (fixed)

| Slot | n_units | Slot | n_units | Slot | n_units | Slot | n_units |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 5 | 256 | 10 | 512 | 15 | 1024 |
| 1 | 128 | 6 | 256 | 11 | 512 | 16 | 1024 |
| 2 | 128 | 7 | 256 | 12 | 512 | 17 | 1024 |
| 3 | 128 | 8 | 256 | 13 | 512 | 18 | 1024 |
| 4 | 128 | 9 | 256 | 14 | 512 | 19 | 1024 |

Each row of 5 = 5 seeds at one size.

## Hyperparameter defaults (cortex_delaygo winners)

All slots start with:

| Field                     | Value           | Source |
|---------------------------|-----------------|--------|
| `recurrent_activation`    | tanh            | delaygo loop (vs relu/softplus/sigmoid) |
| `w_init_mode`             | randn_scaled    | delaygo loop (vs uniform_scaled) |
| `w_init_scale`            | 0.5             | delaygo loop |
| `lr`                      | 1e-3            | delaygo loop |
| `lr_schedule`             | 1e-3→1e-4 over 5 epochs, flat after | delaygo loop |
| `grad_clip_W`             | 2.0             | delaygo loop |
| `coeff_rate_L2`           | 1e-2            | delaygo loop |
| `coeff_W_L2`              | 0.0             | delaygo loop (silent killer when on) |
| `noise_recurrent_level`   | 0.0             | delaygo loop (neutral) |
| `n_epochs`                | 10              | matches budget |
| `batch_size`              | 64              | Yang default |
| `data_augmentation_loop`  | 80              | doubled from delaygo |
| `input_proj`              | mlp             | load-bearing per delaygo |
| `output_proj`             | mlp             | neutral but kept consistent |
| `hidden_dim`              | 128             | encoder/decoder MLP width |
| `n_layers`                | 2               | encoder/decoder MLP depth |

## Available hyperparameter mutations (subsequent blocks)

The **only** knobs that should change across blocks. Apply each mutation
**uniformly to all 20 slots** (the per-slot variation must remain the
fixed n_units sweep — never edit `graph_model.n_units`). Sub-tables
mirror the structure of `instruction_cortex_all.md` so both loops cover
the same axes.

### Recurrent training scheme (PRIORITY)

| Field                    | Default                   | Sweep values                                 | What it tests |
|--------------------------|---------------------------|----------------------------------------------|---------------|
| `lr`                     | `1e-3`                    | {3e-4, 1e-3, 3e-3}                           | Does optimal lr shift with capacity? |
| `lr_schedule`            | per-epoch decay 1e-3→1e-4 | shallower / steeper variants                  | Larger models may benefit from a longer fine-tune phase. |
| `n_epochs`               | `10`                      | {10, 20, 30}                                 | Bigger models may need more passes. |
| `batch_size`             | `64`                      | {1, 8, 32, 64, 128, 256}                     | Larger batches average gradients across more tasks; smaller batches give more updates per epoch. |
| `grad_clip_W`            | `2.0`                     | {1.0, 2.0, 5.0}                              | Bigger models more prone to gradient explosion. |
| `noise_recurrent_level`  | `0.0`                     | {1e-5, 1e-4, 1e-3, 1e-2}                     | May help bigger networks generalise. |
| `data_augmentation_loop` | `80`                      | {40, 80, 160}                                | More iters/epoch with bigger models. |

### Regularisers

| Field           | Default | Sweep values            | Notes |
|-----------------|---------|-------------------------|-------|
| `coeff_rate_L2` | `1e-2`  | {0, 1e-3, 1e-2, 1e-1}   | May matter more with bigger W (more saturation). |
| `coeff_W_L2`    | `0.0`   | {0, 1e-6, 1e-5}         | Silent killer in delaygo at non-zero magnitudes — re-test if 1024-unit nets need decay. |

### W init (under **`training:`**)

| Field          | Default        | Sweep values                                      | Notes |
|----------------|----------------|---------------------------------------------------|-------|
| `w_init_mode`  | `randn_scaled` | {randn_scaled, uniform_scaled}                    | Delaygo found randn_scaled > uniform_scaled. Re-check at 1024. |
| `w_init_scale` | `0.5`          | {0.3, 0.5, 1.0, 1.5}                              | Delaygo plateau 0.3–3.0; may matter more for hard tasks / large N. |

### Architecture (free-W mode)

| Field                  | Default | Sweep values             | Notes |
|------------------------|---------|--------------------------|-------|
| `recurrent_activation` | `tanh`  | {tanh, softplus}         | Re-check at 1024 (delaygo found tanh ≈ relu, sigmoid worst). |
| `hidden_dim`           | `128`   | {128, 256, 512}          | Encoder capacity scaling vs recurrent capacity. |
| `n_layers`             | `2`     | {2, 3}                   | Encoder depth scaling. |
| `MLP_activation`       | `relu`  | {relu, gelu}             | Activation inside encoder/decoder MLP hidden layers (untested in delaygo loop). |
| `input_proj`           | `mlp`   | **Must stay `mlp`**       | Delaygo confirmed linear input caps perf at 0.969 — rule × stim gating is non-linear. |
| `output_proj`          | `mlp`   | `mlp` or `matrix`         | Neutral in delaygo; keep default for consistency. |

## Things you must NOT change

- `dataset` (locked to `task_cortex_all`).
- `task.cortex.*` (locked).
- `graph_model.signal_model_name` (slot-specific or `cortex_all`).
- `graph_model.W_param` (must stay `free`).
- `graph_model.n_units` (**part of the experimental design**).
- `n_input` / `n_output` (fixed by data shape: 85 / 33).

## Metrics

Each slot's `tmp_training/metrics.log` writes:
```
iteration,epoch,loss,mse,motor_max,motor_peak_mean,r2,direction_acc
```

**Primary**: `r2` — coefficient of determination on the masked motor
readout, mean across 20 tasks in that slot's eval batch (since each slot
trains on the multi-task dataset). `direction_acc` is a secondary
diagnostic kept for compatibility with earlier loops.

### Analysis dimensions

For each **(size, hyperparameter)** cell, you have 5 seeds. Report (on
the primary R²):

| Stat | Formula | Use |
|---|---|---|
| mean | mean(r2, 5 seeds) | aggregate perf |
| std  | std(r2, 5 seeds)  | seed sensitivity |
| ceiling | max(r2, 5 seeds) | best-case capability |

Look for the **smallest n_units where mean R² approaches the ceiling
and std drops**. That's the saturation point.

## Block plan

20 slots/batch. Iterations: 640 total = 8 blocks × 80 iter/block = 4 batches/block (matches an 18 h budget at ~30 min/slot).

| Block | Focus                          | What to vary across the 4 batches                                          | Why                                                                              |
| ----- | ------------------------------ | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 1     | **Baseline robustness sweep**  | Nothing — 4 batches of the same 4-size × 5-seed design (16 seeds × 4 sizes).| Establish per-size mean/std/ceiling at default hyperparameters.                  |
| 2     | **lr × size interaction**      | Per-batch: lr ∈ {3e-4, 1e-3, 3e-3} (3 batches) + 1 retest of best          | Bigger models often want lower lr.                                               |
| 3     | **Training budget × size**     | Per-batch: n_epochs ∈ {10, 15, 20}; DAL ∈ {80, 160}                         | Capacity utilisation needs longer training.                                      |
| 4     | **Encoder capacity × size**    | hidden_dim ∈ {128, 256, 512} per batch                                     | Does encoder scaling matter equally at all recurrent sizes?                      |
| 5     | **Activation × size**          | recurrent_activation ∈ {tanh, softplus} per batch                          | Re-check that tanh remains best at 1024 (it was best at 256).                    |
| 6     | **Regularisation × size**      | coeff_rate_L2 ∈ {0, 1e-3, 1e-2, 1e-1}; coeff_W_L2 ∈ {0, 1e-6, 1e-5}         | Bigger networks have more units to keep from saturating.                         |
| 7     | **Per-task floor analysis**    | Use best config from blocks 1–6; analyse per-task `r2` floor              | Identify which task lags at each size; targeted intervention if needed.          |
| 8     | **Final robustness**           | Best config across all knobs, full 4-size × 5-seed grid                    | Confirm winning sweep is seed-robust across all 4 sizes.                         |

## Mutation log format (per batch)

All 20 slots share the same hyperparameter mutation; the per-slot
variation is **n_units (4 sizes) × seed (5 each)**. Report a **2D
table** per batch:

```
## Iter N (block B): [exploration | robustness]
Mutation: [knob -> value]   (applied to all 20 slots, n_units fixed)
Hypothesis: "[testable claim about whether knob × n_units helps]"

Per-size summary (5 seeds each):
  n_units=128:  mean=X.XXX  std=Y.YYY  ceiling=Z.ZZZ
  n_units=256:  mean=X.XXX  std=Y.YYY  ceiling=Z.ZZZ
  n_units=512:  mean=X.XXX  std=Y.YYY  ceiling=Z.ZZZ
  n_units=1024: mean=X.XXX  std=Y.YYY  ceiling=Z.ZZZ

Saturation diagnosis: [128 < 256 ≈ 512 ≈ 1024 → 256 saturates]
                  OR [128 < 256 < 512 ≈ 1024 → 512 saturates]
                  OR [128 < 256 < 512 < 1024 → 1024 isn't enough]
Verdict: [supported | falsified | inconclusive]
Next mutation: [knob -> value]
```

## Winner config

At every block boundary, copy the best-mean config (with the
smallest-saturating n_units) to `config/cortex/cortex_all_unique_winner.yaml`:

```yaml
# Winner: cortex_all_unique_winner.yaml
# Source: iter_NNN (5 seeds × 4 sizes; best size saturates at n_units=K)
# Block: B
# Date: YYYY-MM-DD
#
# Saturation curve (mean r2):
#   n_units=128:  X.XXX
#   n_units=256:  X.XXX
#   n_units=512:  X.XXX
#   n_units=1024: X.XXX
#
# Hyperparameters: <list mutations from baseline that helped>
```

## Notes / hints

- **1024-unit slots will be the slowest** (~4× longer than 128 in
  matmul). The `training_time_target_min: 60` is set to accommodate them.
  If 1024 slots run > hard_runtime_limit_min=180 they'll be killed; bump
  the limit if needed.
- **Memory ceiling on l4**: 1024×1024 W_rec ≈ 4 MB float32; trivial. The
  bottleneck is gradient memory through 400-frame BPTT × batch_size=64
  × hidden state — bigger n_units means quadratic recurrent + linear
  hidden memory. l4 (24 GB) handles this comfortably.
- **Per-task floor at small n_units**: expect `dmc*` and `contextdelaydm*`
  to lag hardest at 128 units. If a single task drops below 0.3 while
  others are at 0.9, that's the floor task and the capacity bottleneck.
- **If 1024 still doesn't saturate**, the bottleneck is elsewhere (likely
  encoder capacity or training budget) — Block 3 / 4 will catch this.
