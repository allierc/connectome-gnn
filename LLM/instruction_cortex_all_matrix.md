# Cortex 20-Task LR Sweep — Matrix Encoder / Decoder

## Goal

Find the **best 3-group learning-rate scheme** (`lr_W_rec_schedule` for
the recurrent core, `lr_W_ED_schedule` for the matrix encoder /
decoder, and `lr` for biases) that produces high **R²** across all 20
Yang 2019 cognitive tasks, **fixing the encoder/decoder to a linear
matrix projection** (`input_proj: matrix`, `output_proj: matrix`).

This is the matrix-mode sibling of `instruction_cortex_all.md`. Same
20-slot per-task design, same dataset assignments, same baseline
hyperparameters — but the **only knobs that get mutated** here are the
three LR groups. Everything architectural is frozen.

## Loop semantics (READ THIS)

20-slot per-task exploration (same as `instruction_cortex_all.md`):

- Each of the 20 slots trains an **independent CortexTaskRNN** on its
  own single-task dataset (slot 0 → `task_cortex_fdgo`, slot 1 →
  `task_cortex_reactgo`, …, slot 19 → `task_cortex_dmcnogo`). The slot
  ↔ task assignment is **fixed** — never edit `dataset` or
  `task.cortex.rules` in a slot YAML.
- **All 20 slots run the same hyperparameters per batch.** The
  per-slot variation is the **task identity**, not the LRs. When you
  mutate, propagate the same edit to **all 20** slot YAMLs in a batch.
- **Read each batch as a 20-row table**: each row = one task. Look for
  which LR mutations help the **hardest tasks** (`contextdelaydm*`,
  `dmc*`) without hurting the easy tasks (`fdgo`, `reactgo`).
- Seed within a slot is forced by the pipeline → variance across
  slots = variance across tasks, not seed-variance.

## Why a dedicated matrix-mode LR sweep

Two facts motivate this loop:

1. The matrix encoder/decoder previously underperformed MLP on a 5600-iter
   per-task run (per-task R² floor ≈ 0.36 for matrix vs ≈ 0.74 for MLP).
   But that was under a **single-LR global schedule** that the cortex
   trainer no longer uses.
2. The trainer was just refactored to a **three-group Adam optimiser**
   (`w_rec` / `w_ED` / `other`) keyed by parameter name (see
   `graph_trainer._data_train_cortex_task`, name partitioning at
   `_name_to_group`). With matrix proj, the encoder/decoder is exactly
   two matrices (`W_in`, `W_out`) and the **`w_ED` group is trivially
   isolated** — perfect for sweeping its LR independently of the
   recurrent core.

So the open question this loop answers is: **does matrix proj catch up
to MLP when its LR is tuned in isolation from the recurrent core?** And
if so, what schedule shapes (steeper / shallower decay, lower / higher
peak) recover or beat the MLP baseline.

## Slot ↔ task mapping (fixed)

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

**Primary metric**: per-slot **R²** (masked motor readout). Headline =
**mean R²** across 20 slots; critical secondary = per-task **floor R²**.

## The 3-group split LR (this is the search space)

`_data_train_cortex_task` builds an Adam optimiser with **three named
param groups**, partitioned by parameter name:

| Group | Members (matrix mode)                       | Field that sets it      | Per-epoch schedule field |
| ----- | ------------------------------------------- | ----------------------- | ------------------------ |
| `w_rec` | `_W_rec_free` (the N×N recurrent matrix)   | `training.lr` fallback  | `training.lr_W_rec_schedule` |
| `w_ED`  | `W_in`, `W_out` (the two encoder/decoder matrices) | `training.lr_W_ED` fallback | `training.lr_W_ED_schedule`  |
| `other` | biases (`b`, `b_out`)                      | `training.lr`           | — (always constant)      |

Each schedule is optional. When present, the trainer overwrites that
group's LR at the start of every epoch (graph_trainer.py loop
`for _gname, _gsched in (("w_rec", lr_W_rec_schedule), ("w_ED",
lr_W_ED_schedule)): ...`). When absent, the group's LR stays at its
init value (constant).

### Baseline (block 1)

| Group   | Value                                                    |
| ------- | -------------------------------------------------------- |
| `lr_W_rec_schedule` | `[1e-3, 5e-4, 2e-4, 1e-4 × 7]` (matches per-task winner) |
| `lr_W_ED_schedule`  | `[1e-3, 5e-4, 2e-4, 1e-4 × 7]` (same as w_rec)           |
| `lr` (biases)       | `1e-3`                                                   |
| `n_epochs`          | `10`                                                     |

This is the *recovered-old-behavior* baseline: every group decays on
the same trajectory the old global `lr_schedule` used, just now plumbed
through three independent fields.

### Search axes

Three axes, two of which (`lr_W_rec_schedule`, `lr_W_ED_schedule`) are
schedule-shaped (10 values per epoch), one (`lr`) is a scalar.

#### Axis A — `lr_W_rec_schedule` (recurrent core)

Knobs to vary:

| Variant       | Schedule                                                    | Hypothesis                                        |
| ------------- | ----------------------------------------------------------- | ------------------------------------------------- |
| `baseline`    | `[1e-3, 5e-4, 2e-4, 1e-4 × 7]`                             | reference (block 1)                               |
| `steeper`     | `[2e-3, 1e-3, 5e-4, 2e-4, 1e-4 × 6]`                       | bigger init then faster decay; faster bifurcation crossing |
| `shallower`   | `[1e-3 × 4, 5e-4 × 3, 2e-4 × 3]`                           | longer high-LR phase to let recurrent core lock in |
| `low_peak`    | `[5e-4, 2e-4, 1e-4 × 8]`                                   | smaller init; safer for unstable W_rec            |
| `high_peak`   | `[3e-3, 1e-3, 5e-4, 2e-4, 1e-4 × 6]`                       | aggressive — only viable if grad_clip_W=2 holds   |
| `flat_tail`   | `[1e-3, 5e-4, 2e-4, 5e-5 × 7]`                             | colder tail; tests whether 1e-4 was over-stepping |

#### Axis B — `lr_W_ED_schedule` (matrix encoder + decoder)

| Variant       | Schedule                                                    | Hypothesis                                        |
| ------------- | ----------------------------------------------------------- | ------------------------------------------------- |
| `baseline`    | `[1e-3, 5e-4, 2e-4, 1e-4 × 7]`                             | reference (block 1)                               |
| `match_rec`   | identical to `lr_W_rec_schedule` of the block               | tied — verifies independence is actually useful   |
| `decoupled_low`  | `[5e-4, 2e-4, 1e-4 × 8]`                                | encoder/decoder anneal *faster* than W_rec        |
| `decoupled_high` | `[2e-3, 1e-3, 5e-4, 2e-4, 1e-4 × 6]`                   | encoder/decoder need *bigger* updates than W_rec to disambiguate stim/category |
| `constant_5e-4` | `[5e-4] × 10`                                             | matches PI's drosophila_cx convention (constant W_ED) — does the schedule help at all? |
| `constant_1e-4` | `[1e-4] × 10`                                             | encoder/decoder might just want to be small and stable |

#### Axis C — `lr` (biases, "other" group, constant)

| Value      | Hypothesis                                              |
| ---------- | ------------------------------------------------------- |
| `1e-3`     | baseline (the "other" group's only knob)                |
| `1e-4`     | match the schedule tail — biases stop drifting after the W groups anneal |
| `0`        | freeze biases — they're tiny, do they matter?           |

### Things you must NOT change

- `dataset` / `task.cortex.rules` / `task.cortex.ruleset` (slot ↔ task locked).
- `graph_model.signal_model_name` (slot-specific or `cortex_all`).
- `graph_model.W_param` (must stay `free`).
- `graph_model.input_proj` (**must stay `matrix`** — this is the point of the loop).
- `graph_model.output_proj` (**must stay `matrix`** — same).
- `graph_model.n_input` / `n_output` (fixed by data shape: 85 / 33).
- `graph_model.recurrent_activation` (`tanh` — delaygo winner, frozen here).
- `graph_model.n_units` (256 — frozen; see `instruction_cortex_all_unique.md` for the capacity loop).
- `training.w_init_*`, `coeff_*`, `grad_clip_W`, `noise_recurrent_level`, `n_epochs`,
  `batch_size`, `data_augmentation_loop` — frozen at the per-task winner values.

**The only knobs in scope are the three LR fields above.** If a result
suggests an architectural change would help (e.g. add an MLP encoder
back), record it but **do not mutate** — that's `instruction_cortex_all.md`
territory.

## Metrics (per slot = per task)

Same `tmp_training/metrics.log` schema as the other cortex loops:

```
iteration,epoch,loss,mse,motor_max,motor_peak_mean,r2,direction_acc
```

Read the batch as a 20-row table, indexed primarily by R²:

| Metric                                | What it measures                                | Target                          |
| ------------------------------------- | ----------------------------------------------- | ------------------------------- |
| **per-slot `r2`** (PRIMARY)           | masked-motor R² for that task                    | ≥ 0.85; hardest may floor lower |
| **mean** R² across 20 slots           | aggregate matrix-mode capability                 | mean ≥ 0.80 (matrix is weaker than mlp; lower the bar slightly) |
| **floor** (min) R² across 20 slots    | weakest task                                     | min ≥ 0.40                      |
| **spread** (max − min) R² across 20 slots | cross-task uniformity                        | < 0.50                          |

For per-task degradation specifically (the regression we're trying to
fix): track **dm1** and **dmsgo**. In the recent matrix run those two
peaked mid-training (R² ~ 0.89, 0.90 at iter 3700) then regressed to
0.74, 0.75 by iter 5600. If a candidate LR scheme keeps those peaks,
that's a strong positive signal even if the mean barely moves.

## Block plan

20 slots/batch. Iterations: 640 total = 8 blocks × 80 iter/block = 4 batches/block.

| Block | Focus                                       | What to vary across the 4 batches                                                                | Why                                                                                  |
| ----- | ------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| 1     | **Baseline robustness**                     | 4 batches of the baseline config (no mutations) — 4 different forced seeds                       | Establish per-task spread floor with matrix proj + recovered-old-behavior LRs.       |
| 2     | **w_rec schedule shape**                    | Per batch: `lr_W_rec_schedule` ∈ {baseline, steeper, shallower, low_peak}                        | Does the recurrent core want a different decay profile under matrix proj?            |
| 3     | **w_rec schedule magnitude**                | Per batch: `lr_W_rec_schedule` ∈ {high_peak, flat_tail, baseline, baseline}                      | Headroom on the recurrent core when w_ED LR is decoupled.                            |
| 4     | **w_ED schedule shape**                     | Per batch: `lr_W_ED_schedule` ∈ {baseline, decoupled_low, decoupled_high, constant_5e-4}         | Does encoder/decoder want a different shape than the recurrent core? Does PI's constant choice port over? |
| 5     | **w_ED schedule magnitude**                 | Per batch: `lr_W_ED_schedule` ∈ {constant_1e-4, decoupled_low × 2 different tails, baseline}      | Pin down the sweet spot for the encoder/decoder magnitude.                           |
| 6     | **w_rec × w_ED coupling**                   | Per batch: best `lr_W_rec_schedule` from blocks 2-3 × {match_rec, decoupled_low, decoupled_high} | Test whether the two groups interact (e.g. does optimal w_ED shift when w_rec changes)? |
| 7     | **Bias LR (`lr` for "other")**              | Per batch: `lr` ∈ {1e-3, 5e-4, 1e-4, 0}                                                          | Does freezing/lowering biases help the late-stage regression?                        |
| 8     | **Final robustness**                        | 4 batches of the combined winner — different forced seeds                                        | Confirm winner is seed-robust on the 20-task battery with matrix proj.               |

Each batch's mutation is applied **uniformly to all 20 slots** before
training. After the batch, read the 20-row R² table, compute mean /
floor / spread, and record the verdict (supported / falsified /
inconclusive). Carry forward only the schedules that improve **floor
without inflating spread**.

## Causality rule

One LR group changes per batch. Combinations are tested in **block 6
only**, after both groups' individual sweet spots are known.

In **robustness mode** (blocks 1, 8), every batch has identical config
except the pipeline-forced seed.

## Mutation log format (per batch)

```
## Iter N (block B): [exploration | robustness]
Mutation: [LR-group -> schedule]   (applied uniformly across all 20 slots)
Hypothesis: "[claim about which group benefits / which task floor moves]"
Slot 0 (fdgo):           r2=X.XXX  dir_acc=Y.YY  motor_max=Z.ZZ  …
Slot 1 (reactgo):        r2=X.XXX  …
…
Slot 19 (dmcnogo):       r2=X.XXX  …
Mean / floor / spread:   mean=X.XXX  floor=Y.YYY (slot K, task T)  spread=Z.ZZZ
Late-stage check:        dm1_peak=X.XXX@iter_J  dm1_final=Y.YYY
                         dmsgo_peak=X.XXX@iter_J  dmsgo_final=Y.YYY
Verdict: [supported | falsified | inconclusive]
Next mutation: [LR-group -> schedule]
```

The **late-stage check** is specific to this matrix-mode loop: any LR
mutation that *removes* the dm1/dmsgo late-training regression (peak →
final R² drop) is a strong positive even if it doesn't move the mean
much.

## Winner config

At every block boundary, copy the best batch's config to
`config/cortex/cortex_all_matrix_winner.yaml` with header:

```yaml
# Winner: cortex_all_matrix_winner.yaml
# Source: iter_NNN_block_B  (mean r2 = X.XXX, floor r2 = Y.YYY, spread = Z.ZZZ)
# Block: B  (focus: <focus>)
# Date: YYYY-MM-DD
#
# Winning LR scheme (3-group):
#   lr (biases):         X.XXe-Y
#   lr_W_rec_schedule:   [a, b, c, d, e, f, g, h, i, j]
#   lr_W_ED_schedule:    [a, b, c, d, e, f, g, h, i, j]
#
# Why this is the winner:
#   - <one-sentence reason>
#   - <which late-stage regression it eliminated, if any>
#
# Per-task breakdown (hardest 5):
#   contextdelaydm1: 0.XX   contextdelaydm2: 0.XX   dmcnogo: 0.XX
#   multidelaydm:    0.XX   dmcgo:           0.XX
```

## Notes / hints

- **Matrix proj is genuinely weaker than MLP** for the harder cortex
  tasks (per the 5600-iter shootout — matrix floor 0.36 vs MLP floor
  0.74 with the old single-LR schedule). The goal here is *not* to beat
  MLP — it's to recover the matrix baseline's old behaviour now that
  the trainer has a 3-group optimiser, and then push as far as the
  decoupled LRs allow.
- **`dm1` and `dmsgo` are the canaries** for late-stage drift. If a
  mutation makes them stop regressing, it's working even if the floor
  is set by a different (truly hard) task.
- **`lr` (biases) is a small lever** — biases are ~N parameters vs ~N²
  for W_rec and ~N×n_input for W_in/W_out. Don't expect huge swings
  from block 7; it's a closeout pass.
- The PI trainer's choice (`lr_W_ED` *constant* at 5e-4) is one of the
  variants in axis B (`constant_5e-4`). If it wins, the cortex pipeline
  converges on the PI convention. If it loses to a schedule, the two
  pipelines diverge — but on principle, not on machinery.
