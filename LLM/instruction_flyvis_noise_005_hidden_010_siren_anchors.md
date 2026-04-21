# FlyVis GNN Hidden-Neuron Exploration — flyvis_noise_005_hidden_010_siren_anchors

## Goal

Recover **connectivity W** from partial observations: 10% of non-retinal neurons
(~1200 of 12005) are hidden during training. Their voltages are reconstructed
by a **SIREN-T temporal INR** with anchor-neuron supervision and a
SIREN↔GNN consistency loss at hidden slots.

The core scientific question:

> **Given 10% hidden neurons, can anchor-voltage supervision + consistency loss
> at hidden slots drive conn_R² → baseline (no-hidden ≈ 0.98) while recovering
> hidden voltages with mean per-neuron Pearson > 0.5, using a non-local INR
> (SIREN) where every time query touches every weight?**

### Objective hierarchy

**Primary — connectivity**
- `W_corrected_R2` (**all-edges** — reported as `conn_R2`; visible-only in parens)

**Secondary — hidden-neuron rollout** *(intricately coupled with conn)*
- `hidden_rollout_pearson` — mean per-neuron Pearson on rollout at hidden neurons
- `visible_rollout_pearson` — same on visible subset (the easy half)

A configuration that improves `conn_R2` but collapses `hidden_rollout_pearson`
should be flagged: the connectivity is being fit at the cost of the hidden
dynamics, which usually means the GNN is compensating for a failing INR. The
long-run goal is joint improvement on both axes. When forced to pick between
two configs with similar conn_R2, prefer the one with higher
`hidden_rollout_pearson`.

**Tertiary — INR reconstruction diagnostics**
- `hidden_nnr_pearson` — mean per-neuron Pearson between SIREN hidden output and GT
- `anchor_nnr_pearson` — mean per-neuron Pearson between SIREN anchor output and GT

**Other**
- `tau_R2`, `V_rest_R2`, `cluster_accuracy`

### SIREN vs NGP — key difference

- **SIREN**: global (every time query reads every weight) → **waterbed**:
  updating one frame corrupts others. Historically forced `batch_size=1` to
  limit the cross-frame interference. With the new anchor loss supplying
  dense supervision across many frames each step, larger batch sizes become
  worth retesting.
- **NGP**: local (each query reads only 2 grid cells per level) → no waterbed.
  See parallel exploration `flyvis_noise_005_hidden_010_ngp_anchors`.

## Recent Code Changes (relative to prior hidden exploration)

1. **Oracle leak REMOVED in `coeff_hidden_voltage` loss.**
   Old: `target = x_ts.voltage[k+1, hidden_ids]` (simulator GT for hidden).
   New: `target = model.forward_hidden_batched(k_starts+1)` (SIREN's own prediction).
   Now a **consistency loss** between GNN dynamics and SIREN trace at hidden
   neurons. Previous tuning of `coeff_hidden_voltage` was calibrated against
   the oracle magnitude and may need retuning.

2. **Anchor loss is now wired**.
   `coeff_anchor_voltage > 0` now actually updates the SIREN anchor slots
   (direct L2 to observed GT at sampled anchor neurons). Previously this
   field existed in config but was not consumed.

3. **R² unmasked**. `conn_R2` now uses all edges; visible-only shown in parens.

4. **Rollout eval on training frames** for hidden-INR models (graph_tester.py
   commit 861ba01). New split `hidden_rollout_pearson` / `visible_rollout_pearson`.

5. **Consistency-loss degeneracy guard.** The anchor loss pins the shared
   SIREN trunk; the main visible-rollout loss drives GNN derivatives.
   Both must be active to prevent the trivial constant solution at hidden.

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i(t)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors)
- DAVIS visual input, `noise_model_level=0.05`
- 64,000 frames, `delta_t=0.02`

## Hidden-Neuron Setup

- `hidden_neuron_fraction: 0.1` → ~1200 hidden neurons (saved to `log_dir/hidden_neuron_ids.pt`).
- Hidden neurons are excluded from the visible prediction loss.
- SIREN-T fills in hidden voltages at every forward pass:
  `x.voltage[hidden_ids] = model.forward_hidden(x, k, hidden_ids)`.
- Anchors are sampled from visible non-retina neurons (disjoint from hidden),
  saved to `log_dir/anchor_neuron_ids.pt` on first run.

## Explorable Parameters

### Loss Coefficients (main sweep axes)

| Parameter              | Default | Sweep values           | Note                                                      |
| ---------------------- | ------- | ---------------------- | --------------------------------------------------------- |
| `coeff_hidden_voltage` | 3000    | {0, 100, 1000, 3000}   | Now CONSISTENCY loss — smaller values likely sufficient   |
| `coeff_anchor_voltage` | 3000    | {0, 1000, 3000, 10000} | Direct anchor supervision of SIREN trunk                  |
| `n_anchor`             | 3600    | {300, 1000, 3000, 3600}| More anchors ⇒ stronger trunk signal; diminishing returns |

### Learning Rates

| Parameter   | Default | Sweep values              | Note                                                          |
| ----------- | ------- | ------------------------- | ------------------------------------------------------------- |
| `lr_W`      | 1e-4    | {5e-5, 1e-4, 2e-4, 5e-4}  | GNN weight LR                                                 |
| `lr`        | 1e-3    | {5e-4, 1e-3, 2e-3}        | GNN MLP LR                                                    |
| `lr_NNR_f`  | 1e-6    | {1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3} | **Same LR is applied to NNR_hidden (the SIREN)** — wide band |
| `lr_embedding` | 1e-3 | do not sweep              | fixed                                                         |

Note: `lr_NNR_f` is the config-parameter name but drives both `NNR_f` and
`NNR_hidden` param groups (see `models/utils.py:329`). SIREN gradient is
weaker than NGP (indirect, global), so the viable band historically sat
at 1e-7..1e-5; the anchor loss may permit higher rates.

### Batch Size (axis reopened)

| Parameter    | Default | Sweep values    | Note                                                       |
| ------------ | ------- | --------------- | ---------------------------------------------------------- |
| `batch_size` | 1       | {1, 8, 16, 32}  | Waterbed; with anchor loss providing broad supervision, bs>1 may now be stable |

Scale `data_augmentation_loop` inversely with `batch_size` to keep the same
per-iteration budget.

### SIREN Architecture (architecture block)

| Parameter                | Default | Sweep values           | Note                                     |
| ------------------------ | ------- | ---------------------- | ---------------------------------------- |
| `hidden_dim_nnr_hidden`  | 2048    | {512, 1024, 2048, 4096}| SIREN hidden layer width                 |
| `n_layers_nnr_hidden`    | 4       | {2, 3, 4, 5}           | Number of SIREN hidden layers            |
| `omega_hidden`           | 4096    | {1024, 2048, 4096, 8192}| SIREN frequency scale (first/hidden)    |
| `outermost_linear_nnr_hidden` | true | true/false          | Final layer: linear or sine              |

### Always-on GNN hyperparameters

| Parameter               | Default  | Description                                                    |
| ----------------------- | -------- | -------------------------------------------------------------- |
| `coeff_g_phi_diff`      | 750      | g_phi monotonicity regularizer                                 |
| `coeff_g_phi_norm`      | 1.0      | g_phi range constraint                                         |
| `coeff_W_L1`            | 5e-5     | L1 on W                                                        |
| `coeff_W_L2`            | 1.5e-6   | L2 on W                                                        |
| `alternate_training`    | true     | alternate-phase training (W vs MLP)                            |
| `alternate_lr_ratio`    | 0.05     | LR ratio during alternate phase                                |

## Parallel Mode — 4 Slots Per Batch

Data is **pre-generated at startup** (`claude.generate_data: false`) and
re-used across iterations. `simulation.seed` and `training.seed` are set
by the pipeline — DO NOT modify them.

Seed formula (automatic):
- `simulation.seed = 1000 + slot` (data generation — fixed at startup)
- `training.seed = iteration * 1000 + slot + 500`

**Robustness classification (primary = conn_R²; secondary = hidden_rollout_pearson):**

- **Stable-Robust**: all 4 seeds conn_R2 > 0.90 AND hidden_rollout_pearson > 0.5 AND CV < 5%
- **Robust**: all 4 seeds conn_R2 > 0.85 AND hidden_rollout_pearson > 0.3, CV 5-10%
- **Partially robust**: 2-3 seeds meet criteria above
- **Fragile**: 0-1 seeds meet criteria — reject
- **DISQUALIFIED**: any seed conn_R2 < 0.70, OR any seed hidden_rollout_pearson < 0

## Multi-parameter Exploration

You may change more than one parameter per iteration when a hypothesis
predicts a **joint effect** — e.g., "larger `omega_hidden` + lower `lr_NNR_f`
together because higher frequency increases gradient variance". Single-parameter
sweeps give cleaner signal; multi-parameter moves should cite a specific
interaction hypothesis. Log which axes moved in the `Mutation:` field.

## Budget

Each iteration targets **~120 min** wall-clock (`claude.training_time_target_min: 120`).
Calibrate `n_epochs` × `data_augmentation_loop` to stay within budget:
- bs=1, DAL=25, n_epochs=6  ≈ 120 min (starting point for bs=1)
- scale DAL inversely with bs (e.g., bs=8, DAL≈3; bs=16, DAL≈2)
- scale n_epochs up if bs×DAL drops below ~25

## Block Partition (suggested)

| Block | Focus                      | Key axes                                                                 |
| ----- | -------------------------- | ------------------------------------------------------------------------ |
| 1     | **Anchor-loss baseline**   | Confirm conn_R² and hidden_nnr_pearson with default config               |
| 2     | **Anchor strength**        | `coeff_anchor_voltage` ∈ {0, 1000, 3000, 10000}                          |
| 3     | **Anchor count**           | `n_anchor` ∈ {300, 1000, 3000, 3600}                                     |
| 4     | **Consistency strength**   | `coeff_hidden_voltage` ∈ {0, 100, 1000, 3000}                            |
| 5     | **Learning rate**          | `lr_W`, `lr`, `lr_NNR_f` scans — anchor loss changes gradient magnitude  |
| 6     | **Batch size (waterbed)**  | `batch_size` ∈ {1, 8, 16, 32} — critical axis for SIREN with anchor loss |
| 7     | **SIREN architecture**     | `hidden_dim_nnr_hidden`, `n_layers_nnr_hidden`, `omega_hidden`           |
| 8     | **Combined best**          | Best of blocks 2–7                                                       |
| 9     | **Validation**             | Best config × 4 seeds, cross-check at n_epochs×DAL budget multipliers    |

## YAML Rules

> Always wrap the `description` field value in double quotes — colons inside
> unquoted YAML strings cause parse errors.

## Variable Names

- **`{base_config_name}`**: `flyvis_noise_005_hidden_010_siren_anchors`
- **`{llm_task_name}`**: `flyvis_noise_005_hidden_010_siren_anchors_Claude`

**Config file paths:**

- `config/fly/flyvis_noise_005_hidden_010_siren_anchors_Claude_00.yaml` through `_03.yaml`
- `config/fly/flyvis_noise_005_hidden_010_siren_anchors_winner.yaml`

## File Structure

### 1. Full Log (append-only)
**File**: `flyvis_noise_005_hidden_010_siren_anchors_Claude_analysis.md`

### 2. Working Memory (read + update every batch)
**File**: `flyvis_noise_005_hidden_010_siren_anchors_Claude_memory.md`

### 3. User Input
**File**: `user_input.md`

## Iteration Workflow (every batch)

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

**Metrics from `analysis.log` / `metrics.txt`:**

- `W_corrected_R2` → `conn_R2` in memory
- `tau_R2`, `V_rest_R2`, `clustering_accuracy`
- `hidden_nnr_pearson`, `anchor_nnr_pearson`
- `hidden_rollout_pearson`, `visible_rollout_pearson`
- `training_time_min`

**Key observations to track:**

- Did conn_R² reach the ≥0.85 zone?
- Did hidden_nnr_pearson cross 0.3 / 0.5?
- Is anchor_nnr_pearson close to ceiling? If not, trunk is under-supervised.
- Gap `anchor_nnr_pearson − hidden_nnr_pearson`: large gap means trunk fits
  visible anchors but hidden slots still fail → increase consistency weight
  or SIREN capacity.
- `hidden_rollout_pearson` vs `visible_rollout_pearson`.
- Waterbed check: if bs > 1 causes hid_pear collapse while bs=1 doesn't at
  the same config, the waterbed is still dominant (axis stays capped).

### Step 3: Write Log Entry

```
## Iter N: [stability] [improving/same/regressing vs previous]

Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: coeff_hv=A, coeff_av=B, n_anc=C, lr_W=D, lr=E, lr_NNR_f=F, bs=G, [arch: h_dim=H, n_layers=L, omega=O]
Slot 0: conn_R2=A, hid_pear=B, anc_pear=C, tau_R2=D, V_rest_R2=E, cluster=F, hid_rollout=G, vis_rollout=H, sim_seed=S, train_seed=T
Slot 1: ...
Slot 2: ...
Slot 3: ...
Seed stats: mean_conn_R2=X, std=Y, CV=Z%, min=W
           mean_hid_pear=X, std=Y, mean_anc_pear=X, std=Y
Stability: [Stable-Robust / Robust / Partially robust / Fragile / DISQUALIFIED]
Mutation: [params]: [old -> new]
Verdict: [supported/falsified/inconclusive] — [explanation]
Waterbed note: [bs interaction effect, if any]
Next: parent=P
```

### Step 4: Acknowledge User Input

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Winner Config (COMPULSORY at every block boundary)

Save to `config/fly/flyvis_noise_005_hidden_010_siren_anchors_winner.yaml` with header:

```yaml
# Winner config: flyvis_noise_005_hidden_010_siren_anchors_winner.yaml
# Source: iter_XXX_slot_YY (conn_R2 = X.XXX, hidden_nnr_pearson = X.XXX, anchor_nnr_pearson = X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [narrative on what the anchor / consistency balance contributed]
#
# Metrics (4-seed):
#   conn_R2:              X.XXX +/- X.XXX (CV=X.X%)
#   hidden_nnr_pearson:   X.XXX +/- X.XXX
#   anchor_nnr_pearson:   X.XXX +/- X.XXX
#   tau_R2:               X.XXX
#   V_rest_R2:            X.XXX
#   cluster_accuracy:     X.XXX
#   hidden_rollout_pear:  X.XXX
#   visible_rollout_pear: X.XXX
#
# Key differences from baseline:
#   - [changed parameters and why]
```

## Block Boundaries

1. Update "Paper Summary" in memory.md
2. Summarize findings in "Previous Block Summaries"
3. Update "Established Principles" (3+ supporting iterations AND cross-seed consistency)
4. Move falsified hypotheses to "Falsified Hypotheses"
5. Clear "Current Block"
6. Compare hid_pear evolution; if stuck below 0.3 after Block 5, flag SIREN
   capacity as the likely bottleneck and prioritize Block 7 (architecture).

## Known Results (reference)

- **Parallel NGP exploration** (`flyvis_noise_005_hidden_010_ngp_anchors`): NGP
  hashtable gives a standalone ceiling of Pearson≈0.65. SIREN ceiling has
  not been measured — capacity may be the limiting factor.
- **No-hidden flyvis** (`flyvis_noise_005`): conn_R²≈0.980, CV<1%.
- **Prior hidden_010 SIREN exploration (OBSOLETE — used oracle loss, no anchor)**:
  SIREN-T never drove hidden_nnr_R² > 0. Ignore these numbers.

## Start Call

When prompt says `PARALLEL START`:

- Read base config `config/fly/flyvis_noise_005_hidden_010_siren_anchors_Claude_00.yaml`
- Set all 4 configs identically to the baseline
- **Initial hypothesis**: "Anchor loss + consistency push hidden_nnr_pearson above 0.2 at iter 1 with default coefficients (3000/3000, n_anchor=3600, bs=1)"
- **Null hypothesis**: "SIREN's waterbed prevents the trunk from benefiting from anchor supervision — hidden_nnr_pearson stays < 0.1 even with anchor loss active"
- Write both hypotheses to working memory
- Block 1 tests the null hypothesis — no mutation yet, just 4 seeds of baseline

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_hidden_010_siren_anchors

## Paper Summary (update at every block boundary)

- **Hidden-neuron inverse problem + SIREN consistency/anchor**: [pending]
- **Waterbed under anchor loss**: [pending]

## Knowledge Base

### Results Table

| Iter | Config summary | conn_R2 | hid_pear | anc_pear | tau_R2 | hid_rollout | time_min | Stability |
| ---- | -------------- | ------- | -------- | -------- | ------ | ----------- | -------- | --------- |
| 1    | baseline       | ?       | ?        | ?        | ?      | ?           | ?        | ?         |

### Established Principles

[Confirmed patterns — require 3+ supporting iterations AND cross-seed consistency]

### Falsified Hypotheses

### Open Questions

1. Does coeff_hidden_voltage still need to be ~3000 under consistency, or is 100-1000 sufficient?
2. Does anchor loss saturate at n_anchor > 3000, or does the gradient keep helping?
3. What is the effective lr_NNR_f sweet spot under the new anchor gradient?
4. Does batch_size > 1 converge now that anchor loss provides distributed supervision, or does the waterbed still dominate?
5. Is SIREN capacity (hidden_dim / n_layers / omega) the bottleneck relative to NGP's standalone ceiling of 0.65?

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest→newest.**

---

## Current Block (Block N)

### Block Info

Focus: [parameter subspace]
Iterations: M to M+n_iter_block

### Current Hypothesis

**Hypothesis**: [specific, testable prediction]
**Rationale**: [prior evidence]
**Test**: [config change — single param or joint]
**Expected outcome**: [what supports vs falsifies]
**Stability constraint**: CV < 5% on primary metrics
**Status**: untested / supported / falsified / revised

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
