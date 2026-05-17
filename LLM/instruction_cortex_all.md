# Cortex 20-Task Exploration — One Slot Per Task

## Goal

Find a **single training scheme** (one set of hyperparameters for the
`TaskRNN` free-W model) that produces high **R²** (coefficient of
determination on the masked motor readout) across **all 20 Yang 2019
cognitive tasks**, using the per-task-slot exploration paradigm
described below.

## Loop semantics (READ THIS — different from earlier cortex_delaygo runs)

This is a **20-slot per-task** exploration, not multi-task training:

- Each of the 20 slots trains an **independent TaskRNN model** on its own
  **single-task dataset** (slot 0 → `task_cortex_fdgo`, slot 1 →
  `task_cortex_reactgo`, …, slot 19 → `task_cortex_dmcnogo`). The slot ↔
  task assignment is **fixed** — never edit `dataset` or
  `task.cortex.rules` in a slot YAML.
- **All 20 slots run the same hyperparameters per batch.** The
  per-slot variation is the **task identity**, not the hyperparameters.
  When you mutate, propagate the same edit to **all 20** slot YAMLs in a
  batch (use the Edit tool with the same old_string / new_string across
  files).
- **Read each batch as a 20-row table**: each row = one task, columns =
  the metrics. Look for which mutations help the **hardest tasks**
  (`contextdelaydm*`, `dmc*`, `dms*`) without hurting the easy tasks
  (`fdgo`, `reactgo`).
- The seed forced by the pipeline per slot becomes the **only** source
  of within-batch noise (since hyperparameters are identical). Variance
  across slots = variance across tasks, not seed-variance.

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

**Primary metric**: per-slot **R²** (coefficient of determination of
the masked motor readout — same definition as
`models/graph_tester.py::stimuli_R2`). The exploration's **headline
number** is the slot-mean **R²**; the **critical secondary** is per-task
**R² floor** (`min_r2`) — never lose sight of which tasks lag.
`direction_acc` (Yang `get_perf`) is kept as a secondary diagnostic
(binary-ish accuracy of decoded ring location); R² is the strictly
better quantitative signal because it tracks per-frame fit quality, not
just argmax correctness.

## What's known from the single-task cortex_delaygo loop

A 72-iter agentic loop on **delaygo alone** converged on:

| Knob                    | Value              | Effect                                                                  |
| ----------------------- | ------------------ | ----------------------------------------------------------------------- |
| `recurrent_activation`  | **tanh**           | beats relu / softplus / sigmoid (sigmoid plateaus at e1≈0.2)            |
| `w_init_scale`          | 0.5 (randn_scaled) | flat plateau 0.3–3.0; uniform_scaled clearly worse                      |
| `grad_clip_W`           | 2.0                | tightens best-in-ep distribution by +0.01 vs 1.0                        |
| `coeff_rate_L2`         | 1e-2               | tightens loss/final distributions without ceiling cost                  |
| `coeff_W_L2`            | 0                  | silent killer at every magnitude ≥ 1e-5                                 |
| `n_units`               | 192–256            | 64 caps perf at 0.969; 512 wastes budget                                |
| `input_proj`            | mlp (default)      | linear input caps perf at 0.969 — encoder non-linearity is load-bearing |
| `output_proj`           | matrix or mlp      | neutral                                                                 |
| `lr`                    | 1e-3 ≈ 2e-3        | 3e-3 looser, 5e-4 underuses budget                                      |
| `noise_recurrent_level` | 0                  | tanh absorbs noise at every magnitude tested                            |

**Open question for multi-task**: do these single-task winners transfer
when the same model must solve 20 different tasks? Specifically:

1. Is `n_units=256` enough capacity for 20 tasks, or do we need 384–512?
2. Does the **encoder MLP capacity** (hidden_dim, n_layers) become more
   binding when rule×stim gating must distinguish 20 rules instead of 1?
3. Does the **lr schedule** need to be longer / shallower because each task
   gets only ~1/20 of the gradient signal per batch?
4. Are some tasks systematically harder (e.g. `dmcnogo`, `multidelaydm`)
   and do they need targeted regularisation (`rule_weights`)?

## The 20 tasks

All 20 share the same I/O shape (1+20+64 input channels, 1+32 output
channels), trial length up to 200 frames (dt=20 ms). They differ in
the trial structure and what the network must compute.

| #   | Rule              | Family                          | What it tests                                                                                                              |
| --- | ----------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 1   | `fdgo`            | Memory-Pro / Go                 | Fixate while stim on; saccade to stim location when fixation drops. _Stimulus-then-go._                                    |
| 2   | `reactgo`         | Reaction-Pro                    | Saccade to stim **as soon as** fixation drops. _Pure reaction time._                                                       |
| 3   | `delaygo`         | Memory-Pro (delay)              | Brief stim, **delay period** (200–1600 ms), then saccade to remembered location. _Working-memory of a single angle._       |
| 4   | `fdanti`          | Memory-Anti                     | Same as fdgo but saccade to **opposite** of stim. _Stimulus-then-anti._                                                    |
| 5   | `reactanti`       | Reaction-Anti                   | Same as reactgo but anti.                                                                                                  |
| 6   | `delayanti`       | Memory-Anti (delay)             | Same as delaygo but anti.                                                                                                  |
| 7   | `dm1`             | Decision-Making (ring 1)        | Two stims simultaneous on ring 1; saccade to **stronger** stim.                                                            |
| 8   | `dm2`             | DM (ring 2)                     | Same as dm1 but on the other stim modality / ring.                                                                         |
| 9   | `contextdm1`      | Context DM (attend ring 1)      | Two stims per ring; rule says **attend ring 1** → saccade to stronger of ring 1. _Context-gated attention._                |
| 10  | `contextdm2`      | Context DM (attend ring 2)      | Same as contextdm1, attend ring 2.                                                                                         |
| 11  | `multidm`         | Multisensory DM                 | Two stims per ring; **integrate both rings** to find stronger combined.                                                    |
| 12  | `delaydm1`        | Delayed DM (ring 1)             | Two stims **separated in time** on ring 1; integrate evidence across the delay. _Working-memory of evidence accumulation._ |
| 13  | `delaydm2`        | Delayed DM (ring 2)             | Same as delaydm1, ring 2.                                                                                                  |
| 14  | `contextdelaydm1` | Context delayed DM (attend 1)   | Time-separated stims, attend ring 1. _Hardest of the DM family._                                                           |
| 15  | `contextdelaydm2` | Context delayed DM (attend 2)   | Same, attend ring 2.                                                                                                       |
| 16  | `multidelaydm`    | Multisensory delayed DM         | Time-separated stims, integrate both rings.                                                                                |
| 17  | `dmsgo`           | DMS (delayed-match-to-sample)   | Two stims separated in time. If **same location**, saccade; else fixate.                                                   |
| 18  | `dmsnogo`         | DNMS (no-match)                 | If stims **different**, saccade; else fixate. _Inverted dmsgo._                                                            |
| 19  | `dmcgo`           | DMC (delayed-match-to-category) | Two stims; if locations belong to **same category** (semicircle), saccade. _Categorical generalisation._                   |
| 20  | `dmcnogo`         | DNMC (no-category)              | If categories **differ**, saccade. _Inverted dmcgo._                                                                       |

Conceptual difficulty (Yang's empirical ordering, hardest last):

```
fdgo, reactgo                                 ← simplest reflex
fdanti, reactanti                             ← + remap
delaygo, delayanti                            ← + delay
dm1, dm2                                      ← + comparison
contextdm1, contextdm2                        ← + context gating
multidm                                       ← + multisensory integration
delaydm1, delaydm2                            ← + delay
contextdelaydm1, contextdelaydm2              ← + context + delay
multidelaydm                                  ← + multisensory + delay
dmsgo, dmsnogo                                ← + category-free match
dmcgo, dmcnogo                                ← + category abstraction
```

The hardest tasks (contextdelaydm\*, dmcgo, dmcnogo) typically need the
most representational capacity. They may set the floor for `n_units` and
encoder MLP size.

## Available hyperparameters (the search space)

Same axes as cortex_delaygo, with adjusted priorities given multi-task
challenges.

### Recurrent training scheme (PRIORITY)

| Field                    | Default (multi-task)      | What it controls                                                                            |
| ------------------------ | ------------------------- | ------------------------------------------------------------------------------------------- |
| `lr`                     | `1e-3`                    | Yang papers use 1e-3 → 1e-4. Try {5e-4, 1e-3, 2e-3}. Slower decay may help multi-task.      |
| `lr_schedule`            | per-epoch decay 1e-3→1e-4 | Try shallower decay {1e-3, 1e-3, 5e-4, …} since each task gets only ~5% of gradient signal. |
| `n_epochs`               | `10`                      | 20 tasks may need more passes than single-task; try {10, 20, 30}.                           |
| `batch_size`             | `64`                      | Try {1, 8, 32, 64, 128, 256}. Larger may help cross-task gradient averaging.                |
| `grad_clip_W`            | `2.0` (delaygo winner)    | Try {1.0, 2.0, 5.0}.                                                                        |
| `noise_recurrent_level`  | `0.0`                     | Try {1e-5, 1e-4, 1e-3, 1e-2}. Delaygo found it neutral but multi-task may benefit.          |
| `data_augmentation_loop` | `40`                      | Multiplies iters/epoch. 20× more data per epoch ⇒ may want DAL ≥ 40 to converge.            |

### Regularisers

| Field           | Default | Notes                                                                            |
| --------------- | ------- | -------------------------------------------------------------------------------- |
| `coeff_rate_L2` | `1e-2`  | delaygo winner; keep on by default.                                              |
| `coeff_W_L2`    | `0.0`   | Silent killer in delaygo. Test if multi-task changes this (try {0, 1e-6, 1e-5}). |

### W init

| Field          | Default        | Notes                                                                            |
| -------------- | -------------- | -------------------------------------------------------------------------------- |
| `w_init_mode`  | `randn_scaled` | (under **`training:`**) Best for free W.                                         |
| `w_init_scale` | `0.5`          | (under **`training:`**) Delaygo plateau 0.3–3.0; may matter more for hard tasks. |

### Architecture (free-W mode)

| Field                  | Default | Notes                                                                               |
| ---------------------- | ------- | ----------------------------------------------------------------------------------- |
| `n_units`              | `256`   | **High priority for multi-task**: try {256, 384, 512}. Single-task tied at 192–384. |
| `recurrent_activation` | `tanh`  | Delaygo winner.                                                                     |
| `hidden_dim`           | `128`   | **High priority**: encoder MLP must gate 20 rules × stim. Try {128, 256, 512}.      |
| `n_layers`             | `2`     | Try {2, 3}. Single-task was capacity-neutral.                                       |
| `MLP_activation`       | `relu`  | Try {relu, gelu}.                                                                   |
| `input_proj`           | `mlp`   | **Must stay `mlp`** — single-task confirmed linear input caps perf at 0.969.        |
| `output_proj`          | `mlp`   | Neutral in single-task; keep as default.                                            |

### Things you must NOT change

- `dataset` (each slot is locked to its assigned task — slot 0 ↔
  `task_cortex_fdgo`, slot 1 ↔ `task_cortex_reactgo`, etc. Editing
  `dataset` in a slot breaks the slot→task mapping).
- `task.cortex.rules` (each slot has exactly one rule = its task).
- `task.cortex.ruleset` (must stay `"all"`).
- `task.task_type` (must stay `cortex`).
- `graph_model.signal_model_name` (slot-specific, e.g. `cortex_fdgo`).
- `graph_model.W_param` (must stay `free`).
- `n_input` / `n_output` (fixed by data shape: 85 / 33).

**Crucially**: hyperparameter mutations must be applied **uniformly to
all 20 slots**. Do not make slot K differ from slot J in any field other
than the locked task-related ones above. The whole point of the
exploration is comparing **the same configuration** across 20 tasks.

## Metrics (per slot = per task)

Each slot's `tmp_training/metrics.log` writes:

```
iteration,epoch,loss,mse,motor_max,motor_peak_mean,r2,direction_acc
```

Because each slot trains on a **single task**, each slot's metrics
describe that **specific task's** behaviour, not a multi-task mean.

**Read the batch as a 20-row table**, indexed primarily by R²:

| Metric                                    | What it measures                                                                                       | Target                                                              |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| **per-slot `r2`** (PRIMARY)               | Coefficient of determination on the masked motor readout — `1 − SS_res/SS_tot` over supervised frames. | **≥ 0.85**; hard tasks (dmc*, contextdelaydm*) may floor below 0.7. |
| per-slot `direction_acc` (secondary)      | Yang `get_perf` — fraction of trials with argmax(pred motor) == argmax(target motor).                  | Tracks R² qualitatively; less sensitive (binary).                   |
| **mean** R² across 20 slots               | Aggregate "how well does this config work on average across the 20-task battery".                      | **mean ≥ 0.85**                                                     |
| **floor** (min) R² across 20 slots        | Which task is the weakest link?                                                                        | **min ≥ 0.5** (no task left behind)                                 |
| **spread** (max − min) R² across 20 slots | If a config helps the easy tasks but tanks the hard ones, spread balloons.                             | **< 0.40**                                                          |

A config with **mean R² = 0.85 / floor R² = 0.40** is _worse_ than one
with **mean = 0.75 / floor = 0.65** — the latter generalises, the
former specialises. The agentic loop should prefer the latter unless
explicitly testing single-task ceiling.

## Block plan

4 slots/batch. Iterations: 96 total = 8 blocks × 12 iter/block = 3 batches/block.

| Block | Focus                               | Knobs to scan                                                                                | Why                                                                |
| ----- | ----------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 1     | **Baseline + delaygo-winner check** | 4 seeds of delaygo-winner config (no mutations)                                              | Robustness baseline on 20 tasks. Establish per-task spread floor.  |
| 2     | **Recurrent capacity**              | `n_units` ∈ {192, 256, 384, 512}                                                             | Hardest tasks (context delay DM, dmcnogo) may need more units.     |
| 3     | **Encoder MLP capacity**            | `hidden_dim` ∈ {128, 256, 512}; `n_layers` ∈ {2, 3}                                          | Encoder is load-bearing; 20-rule gating may be harder than 1-rule. |
| 4     | **lr + schedule**                   | `lr` ∈ {5e-4, 1e-3, 2e-3}; `lr_schedule` shallower vs steeper                                | Each task gets ~5% of gradient — may need slower decay.            |
| 5     | **Training budget**                 | `n_epochs` ∈ {10, 20, 30}; `data_augmentation_loop` ∈ {40, 80, 160}                          | 20× data per epoch — may need longer training to converge.         |
| 6     | **Stability + regularisers**        | `grad_clip_W` ∈ {1, 2, 5}; `noise_recurrent_level` ∈ {0, 1e-3, 1e-2}; `coeff_rate_L2` levels | May behave differently when 20 tasks compete for representations.  |
| 7     | **Per-task targeted help**          | `rule_weights` upweighting hardest tasks (`contextdelaydm*`, `dmc*`)                         | If specific tasks lag, try targeted sampling.                      |
| 8     | **Final robustness**                | 4 seeds of the combined winner config                                                        | Confirm winner is seed-robust on the 20-task battery.              |

## Causality rule

You can change one or two parameters per slot.

In **robustness mode** (every slot identical), the pipeline forces N
different seeds; this measures seed sensitivity of a candidate winner on
the 20-task battery.

## Mutation log format (per batch)

All 20 slots share the same hyperparameter mutation; the per-slot
variation is the **task**, not the knob. So the entry header records
**one mutation** and the slot table reports **per-task results**:

```
## Iter N (block B): [exploration | robustness]
Mutation: [knob -> value]   (applied uniformly across all 20 slots)
Hypothesis: "[testable claim about what the mutation should do]"
Slot 0 (fdgo):           r2=X.XXX  dir_acc=Y.YY  motor_max=Z.ZZ  traj=e1=A …
Slot 1 (reactgo):        r2=X.XXX  …
Slot 2 (delaygo):        r2=X.XXX  …
…
Slot 19 (dmcnogo):       r2=X.XXX  …
Mean / floor / spread:   mean=X.XXX  floor=Y.YYY (slot K, task T)  spread=Z.ZZZ
Verdict: [supported | falsified | inconclusive]
Next mutation: [knob -> value]
```

The **floor task** (lowest perf) is the most diagnostic. Track which
task becomes the floor across iterations — if it's always the same task
(say `dmcnogo`), targeted regularisation or `rule_weights`-style
upweighting becomes a candidate intervention.

## Winner config

At every block boundary, copy the best slot's config to
`config/cortex/cortex_all_winner.yaml` with header:

```yaml
# Winner: cortex_all_winner.yaml
# Source: iter_NNN_slot_KK  (mean r2 = X.XXX, floor r2 = Y.YYY, spread = Z.ZZZ)
# Block: B  (focus: <focus>)
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - <one-sentence reason>
#   - <key knob change vs delaygo winner>
#
# Per-task breakdown (hardest 5):
#   contextdelaydm1: 0.XX   contextdelaydm2: 0.XX   dmcnogo: 0.XX
#   multidelaydm:    0.XX   dmcgo:           0.XX
```

## Notes / hints

- **Yang 2019 takes ~20 000 iters** to reach mean perf > 0.85 on the
  20-task battery with a 256-unit RNN. With DAL=40 and batch_size=64 our
  budget per slot is ~3 000 gradient steps × 10 epochs = 30 000 — should
  be enough but right at the edge.
- **Per-task signal is the gold standard**, not the mean. A mean=0.85 with
  4 tasks stuck at 0.4 is worse than mean=0.75 uniform. The 8-panel
  snapshot mixes all 20 tasks in its 64-trial sample, so individual task
  curves may be drowned out.
- **`rule_weights` is the targeted lever for late-stage blocks**. Yang
  found that upweighting hard tasks (`contextdelaydm*`) by 2–4× helps
  them catch up without harming the easy ones.
- The 20 individual `cortex_<task>.yaml` configs are available for
  _per-task_ analysis. If the multi-task loop fails for a specific task,
  train just that task with the same hyperparameters as a sanity check.
