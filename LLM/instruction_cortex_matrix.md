# Cortex Per-Task Exploration — Matrix Encoder / Decoder

## Goal

Find the **best training scheme** for an INDIVIDUAL `CortexTaskRNN` with
matrix encoder/decoder (`input_proj: matrix`, `output_proj: matrix`),
trained on one Yang 2019 rule at a time. Each of the 20 slots trains an
**independent** RNN on its own single-task dataset; mutations apply
uniformly across the 20 slots so we can see which knobs help which task
families.

The single-task baseline `cortex_delaygo_sigma` reached **R² = 0.989,
direction_acc = 0.93** in 6,000 iterations (5 epochs, batch_size 64, lr
peak 2e-3 slower-decay × 2.0, readout_uses_sigma true, rate_L2 1e-2). The
per-task slots are pre-seeded with this exact config — the only thing
this loop mutates is hyperparameters (uniformly across all 20 rules).

## Loop semantics (READ THIS)

20-slot per-task design. Each slot trains its own rule independently:

- Each of the 20 slots trains an **independent CortexTaskRNN** on its
  own single-task dataset (slot 0 → `task_cortex_fdgo`, slot 1 →
  `task_cortex_reactgo`, …, slot 19 → `task_cortex_dmcnogo`). The slot
  ↔ task assignment is **fixed**, pre-seeded by us — never edit
  `dataset` / `signal_model_name` / `task.cortex.rules` in any slot
  YAML.
- **All 20 slots share the same training hyperparameters per batch.**
  The per-slot variation is the **task identity**, not the seeds or
  LRs. When you mutate, propagate the same edit to **all 20** slot
  YAMLs in a batch.
- **Read each batch as a 20-row table**: each row = one task. Look for
  which mutations help the hardest tasks (`contextdelaydm*`, `dmc*`)
  without hurting the easy ones (`fdgo`, `reactgo`).
- Seeds within a slot are forced by the pipeline → variance across
  slots = variance across tasks, NOT seed variance.

## Slot ↔ task mapping (fixed, pre-seeded)

| Slot | Task         | Family              | Slot | Task              | Family                |
| ---: | ------------ | ------------------- | ---: | ----------------- | --------------------- |
|    0 | `fdgo`       | Memory-Pro          |   10 | `multidm`         | Multisensory DM       |
|    1 | `reactgo`    | Reaction-Pro        |   11 | `delaydm1`        | Delayed DM (ring 1)   |
|    2 | `delaygo`    | Memory-Pro (delay)  |   12 | `delaydm2`        | Delayed DM (ring 2)   |
|    3 | `fdanti`     | Memory-Anti         |   13 | `contextdelaydm1` | Context+delay DM (1)  |
|    4 | `reactanti`  | Reaction-Anti       |   14 | `contextdelaydm2` | Context+delay DM (2)  |
|    5 | `delayanti`  | Memory-Anti (delay) |   15 | `multidelaydm`    | Multisensory delay DM |
|    6 | `dm1`        | DM (ring 1)         |   16 | `dmsgo`           | DMS                   |
|    7 | `dm2`        | DM (ring 2)         |   17 | `dmsnogo`         | DNMS                  |
|    8 | `contextdm1` | Context DM (1)      |   18 | `dmcgo`           | DMC                   |
|    9 | `contextdm2` | Context DM (2)      |   19 | `dmcnogo`         | DNMC                  |

**Primary metric**: per-slot **R²** (masked motor readout, computed by
`compute_cortex_task_metrics`). Each slot's R² is the **per-task** score
for that rule.

## Baseline (Block 1, canonical-winner config)

Carries cortex_delaygo_sigma's settings verbatim:

| Field                  | Value                                              | Source |
|------------------------|----------------------------------------------------|--------|
| `recurrent_activation` | tanh                                               | delaygo loop |
| `readout_uses_sigma`   | **true**                                           | cortex_delaygo A/B (R²=0.927 in 1 ep vs 0.493 for raw h) |
| `input_proj`           | **matrix** (pinned)                                | this loop's premise |
| `output_proj`          | **matrix** (pinned)                                | this loop's premise |
| `w_init_mode`          | randn_scaled                                       | delaygo loop |
| `w_init_scale`         | 0.5                                                | delaygo loop |
| `lr`                   | 2e-3 (biases)                                      | cortex_delaygo canonical winner |
| `lr_W_rec_schedule`    | `[2e-3, 2e-3, 1e-3, 4e-4, 2e-4]` (slower-decay×2)  | canonical winner |
| `lr_W_ED_schedule`     | same as `lr_W_rec_schedule` (tied)                 | start tied, decouple via Block 2 |
| `grad_clip_W`          | 2.0                                                | delaygo loop |
| `coeff_rate_L2`        | 1e-2                                               | delaygo loop |
| `coeff_W_L2`           | 0.0                                                | silent killer when on |
| `noise_recurrent_level`| 0.0                                                | neutral |
| `n_epochs`             | **5**                                              | matches schedule length |
| `batch_size`           | 64                                                 | delaygo canonical |
| `data_augmentation_loop` | 80                                               | gives ~1250 iter/epoch × 5 = 6250 iter/slot |

Expected per-slot baseline: **R² ≈ 0.95-0.99 on easy tasks** (fdgo,
reactgo, delaygo, fdanti, reactanti), **R² ≈ 0.80-0.95 on medium tasks**
(dm*), **R² ≈ 0.60-0.85 on hard tasks** (contextdelaydm*, dmc*). The
single-seed delaygo number (R²=0.989) is the high anchor.

## Sweepable hyperparameter families

Apply each mutation **uniformly to all 20 slots** (slot ↔ task mapping
stays fixed). The exploration tests *families* of knobs, not single
parameters in isolation.

### LR family

| Field                    | Default                                                         | Sweep values                                                                            |
|--------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `lr_W_rec_schedule[0]` (peak) | 2e-3                                                       | {1e-3 (half-peak), 2e-3 *parent*, 5e-3 (high-peak)}                                     |
| `lr_W_rec_schedule` shape | slower-decay × 2.0 (parent)                                    | {steeper (3-ep tail), shallower (4 epochs at peak), cosine}                             |
| `lr_W_ED_schedule`       | tied to w_rec (parent)                                          | {tied *parent*, decoupled_low `[5e-4, 2e-4, 1e-4, 1e-4, 1e-4]`, decoupled_high `[5e-3, 2e-3, 1e-3, 4e-4, 2e-4]`} |

### Regularisation family

| Field           | Default | Sweep values            |
|-----------------|---------|-------------------------|
| `coeff_rate_L2` | 1e-2    | {0, 1e-3, 1e-2 *parent*, 1e-1} |
| `coeff_W_L2`    | 0.0     | {0 *parent*, 1e-6, 1e-5} |
| `noise_recurrent_level` | 0.0  | {0 *parent*, 1e-3, 1e-2} |
| `grad_clip_W`   | 2.0     | {1.0, 2.0 *parent*, 5.0} |

### Scheduler shape family

| Field          | Default        | Sweep values                                      |
|----------------|----------------|---------------------------------------------------|
| schedule       | slower-decay × 2.0 (2-ep plateau, parent) | {cosine annealing, 3-ep plateau then geometric ×0.5, linear-decay} |

### Architecture (mostly pinned)

| Field                  | Default | Sweep values             | Notes |
|------------------------|---------|--------------------------|-------|
| `recurrent_activation` | tanh    | {tanh *parent*, softplus} | sigmoid was worst in delaygo; relu ≈ tanh |
| `readout_uses_sigma`   | true    | {true *parent*, false}    | re-verify the cortex_delaygo finding generalises across all 20 tasks |
| `input_proj`           | matrix  | **pinned** | the point of the loop |
| `output_proj`          | matrix  | **pinned** | the point of the loop |
| `n_units`              | 256     | **pinned** | capacity sweep is in `cortex_all_unique_matrix` |

## Things you must NOT change

- `dataset` (each slot pre-seeded to its rule's single-task dataset).
- `signal_model_name` (slot-specific cortex_<rule> registry entry).
- `task.cortex.rules` (single rule per slot, pre-seeded).
- `graph_model.W_param` (must stay `free`).
- `graph_model.n_units` (256 — pinned for this loop; capacity sweep is the unique_matrix loop).
- `graph_model.input_proj` / `output_proj` (must stay `matrix`).
- `n_input` / `n_output` (fixed by Yang data shape: 85 / 33).

## Metrics

Each slot's `tmp_training/metrics.log` writes:
```
iteration,epoch,loss,mse,motor_max,motor_peak_mean,r2,direction_acc
```

Per-slot R² = per-task R² for that rule.

### Analysis dimensions (per batch)

| Stat | Formula | Use |
|---|---|---|
| **mean R²** | mean(r2, 20 tasks) | aggregate per-batch perf |
| **floor R²** | min(r2, 20 tasks) | hardest task's score |
| **spread R²** | max − min | cross-task uniformity |
| **per-task table** | r2 for each of 20 rules | which tasks the mutation helps/hurts |

A useful mutation **raises floor without inflating spread**. Watch the
hard-task floor (typically `contextdelaydm1`, `dmcgo`, or `dmcnogo`).

## Block plan

20 slots/batch. **Iterations: 280 total / 14 batches** (~1.5 h wall on
a100 at 5 epochs × DAL 80 ≈ ~6 min/slot). Each block is an exploration
step covering a *family* of knobs in one pass.

| Block | Focus                                          | Batches × profiles                                                                                                  | Iters |
| ----- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----: |
| 1     | **Baseline** (cortex_delaygo_sigma starter)    | 2 batches of the baseline config, different pipeline-forced seeds.                                                  | 40    |
| 2     | **Sweep all LR** (peak × shape × decoupling)   | 3 batches, each a different multi-knob LR profile.                                                                  | 60    |
| 3     | **Sweep all regularisation**                   | 3 batches: rate_L2 + W_L2 + noise + grad_clip combos.                                                               | 60    |
| 4     | **Fine-tune scheduler shape**                  | 2 batches: cosine, 3-ep plateau (around the Block-2 winner peak).                                                   | 40    |
| 5     | **CV final on winner**                         | 4 batches of the combined winner — different forced seeds.                                                          | 80    |
| **Total** |                                            | **14**                                                                                                              | **280** |

### Block 2 profiles (LR sweep — multi-knob)

| Batch | `lr_W_rec[0]` | `lr_W_rec_schedule` shape | `lr_W_ED_schedule` | Hypothesis |
| --- | --- | --- | --- | --- |
| 2.1 | **1e-3** (half-peak) | proportional shrink of parent | match w_rec | "Some tasks (esp. context+delay) may want gentler updates." |
| 2.2 | **5e-3** (high-peak) | steeper decay (3-epoch tail) | match w_rec | "Easier tasks may reach ceiling faster with bigger LR." |
| 2.3 | 2e-3 (parent)        | parent shape | **decoupled_low** `[5e-4, 2e-4, 1e-4, 1e-4, 1e-4]` | "The (n_input, n_output) ED matrices are tiny — they want lower LR than the (256, 256) W_rec." |

### Block 3 profiles (regularisation sweep — multi-knob)

| Batch | `coeff_rate_L2` | `coeff_W_L2` | `noise_recurrent_level` | `grad_clip_W` | Hypothesis |
| --- | --- | --- | --- | --- | --- |
| 3.1 | **0** | 0 | 0 | 5.0 | "No-reg baseline — does sigma=true alone keep activity bounded?" |
| 3.2 | 1e-2 (parent) | **1e-6** | 0 | 2.0 | "Mild W_L2 might raise the hard-task floor." |
| 3.3 | **1e-1** | 0 | **1e-3** | 2.0 | "Stronger rate L2 + noise — does it help the regression on dm1/dmsgo?" |

### Block 4 profiles (scheduler fine-tune)

Use the Block-2 winner's peak LR.

| Batch | Schedule shape | Hypothesis |
| --- | --- | --- |
| 4.1 | **Cosine annealing** from peak to 1e-5 over 5 epochs | "Smooth annealing avoids step-discontinuity perturbations." |
| 4.2 | **3-epoch plateau** at peak, then geometric ×0.5/epoch | "Easy tasks benefit from a longer high-LR exploration phase." |

### Block 5 (CV final)

4 batches × 20 slots = 80 runs at the winner config, different
pipeline-forced seeds per batch. Per-task R² is now estimated from
**4 seeds per task** — std/√4 ≈ ±0.025 precision per task is enough to
declare which mutations beat baseline on individual tasks.

## Exploration rule

Each block tests a *family* of knobs (LR, regularisation, schedule
shape, etc.). Within a block, batches test different **multi-knob
coordinated profiles** — one profile per batch, applied uniformly to
all 20 slots. Across blocks, the winner from block B becomes the parent
for block B+1.

The per-slot task assignment is fixed across all blocks — slot 0 is
always fdgo, slot 19 is always dmcnogo, etc. Mutations only touch the
shared hyperparameters in `training:` / `graph_model:` (within the
"sweepable" tables above).

In **CV mode** (Block 5), every batch has identical config except the
pipeline-forced seed.

## Mutation log format (per batch)

```
## Iter N (block B, batch K): [exploration | CV]
Mutation: <multi-knob profile description>   (applied uniformly to all 20 slots)
Hypothesis: "[testable claim — which task families benefit]"

Per-task R² table:
  Slot 0  (fdgo):            r2=X.XXX   dir_acc=Y.YY
  Slot 1  (reactgo):         r2=X.XXX   ...
  ...
  Slot 19 (dmcnogo):         r2=X.XXX   ...

Aggregate stats:  mean=X.XXX   floor=Y.YYY (slot K, task T)   spread=Z.ZZZ
Late-stage check: dm1_peak=X.XXX@iter_J  dm1_final=Y.YYY
                  dmsgo_peak=X.XXX@iter_J  dmsgo_final=Y.YYY
Verdict: [supported | falsified | inconclusive]
Next mutation: <next block's profile>
```

The **late-stage check** is specific to this matrix-mode loop: the
matrix decoder showed a peak→final regression on `dm1` and `dmsgo`
under the old constant-LR regime. Track those two specifically — any
mutation that eliminates the regression without inflating spread is a
strong positive signal even if mean R² barely moves.

## Winner config

At every block boundary, copy the best-mean config to
`config/cortex/cortex_matrix_winner.yaml`:

```yaml
# Winner: cortex_matrix_winner.yaml
# Source: iter_NNN  (mean r2 = X.XXX, floor r2 = Y.YYY, spread = Z.ZZZ)
# Block: B  (focus: <focus>)
# Date: YYYY-MM-DD
#
# Per-task R² breakdown (sorted by R²):
#   fdgo:           0.XXX   reactgo:        0.XXX   delaygo:        0.XXX
#   ...
#   contextdelaydm1: 0.XXX  dmcnogo:        0.XXX
#
# Hyperparameters: <list mutations from baseline that helped>
```

## Notes / hints

- **Per-task R² floors at convergence**: from the cortex_delaygo single-
  task winner (R²=0.989), expect easy tasks to land at R² ≥ 0.95 and
  hard tasks (dmc*, contextdelaydm*) to land at R² ≥ 0.70 with matrix
  proj. If multiple slots are below 0.70 at Block-1 convergence,
  matrix proj is hitting a per-task expressiveness ceiling.
- **`dm1` and `dmsgo` are the canaries** for late-stage drift. If a
  mutation makes them stop regressing, it's working even if the floor
  is set by a different (truly hard) task.
- **batch_size 64** matches the cortex_delaygo canonical winner; do
  not raise it without re-testing — the LR peak 2e-3 was tuned at this
  batch size and effective LR drops as batch_size grows.
- **Don't compare directly to the multi-task `cortex_all_unique_matrix`
  loop**: that loop trains ONE RNN over all 20 rules and tests the
  capacity tradeoff. This loop trains 20 separate RNNs; per-slot R²
  here is the per-task ceiling, not the multi-task aggregate.
