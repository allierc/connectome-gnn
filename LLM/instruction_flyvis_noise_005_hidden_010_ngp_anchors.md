# FlyVis GNN Hidden-Neuron Exploration — flyvis_noise_005_hidden_010_ngp_anchors

## Goal

Recover **connectivity W** from partial observations: 10% of non-retinal neurons
(~1200 of 12005) are hidden during training. Their voltages are reconstructed
by a **MultiResTemporalGrid (NGP-T)** with anchor-neuron supervision and a
NGP↔GNN consistency loss at hidden slots.

The core scientific question:

> **Given 10% hidden neurons, can anchor-voltage supervision + consistency loss
> at hidden slots drive conn_R² → baseline (no-hidden ≈ 0.98) while recovering
> hidden voltages with mean per-neuron Pearson > 0.5?**

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
- `hidden_nnr_pearson` — mean per-neuron Pearson between NGP hidden output and GT
- `anchor_nnr_pearson` — mean per-neuron Pearson between NGP anchor output and GT

**Other**
- `tau_R2`, `V_rest_R2`, `cluster_accuracy`

### Reference ceilings

- **Standalone NGP on GT traces** (direct supervised fit, same hash config, 500k steps):
  final R² = 0.428 ≈ Pearson ≈ 0.65. This is the **architectural ceiling** for
  hidden/anchor Pearson at this NGP config.
  File: `/groups/saalfeld/home/allierc/GraphData/log/fly/flyvis_noise_005_hidden_010/tmp_training/ngp_voltage/results.log`
- **Stride-1 no-hidden** (`flyvis_noise_005`): `connectivity_R2 ≈ 0.980`.

## Recent Code Changes (relative to prior ngp exploration)

Two of the prior exploration's conclusions are now obsolete — don't import
them as Established Principles:

1. **Oracle leak REMOVED in `coeff_hidden_voltage` loss.**
   Old: `target = x_ts.voltage[k+1, hidden_ids]` (simulator GT for hidden).
   New: `target = model.forward_hidden_batched(k_starts+1)` (NGP's own prediction).
   It is now a **consistency loss** between GNN dynamics and NGP trace at
   hidden neurons. Previous tuning of `coeff_hidden_voltage` at 3000 was
   calibrated against the oracle magnitude and may need retuning.

2. **Anchor loss is now wired**.
   `coeff_anchor_voltage > 0` now actually updates the NGP anchor slots
   (direct L2 to observed GT at sampled anchor neurons). Previously this
   field was defined in config but not consumed — `n_anchor` and
   `coeff_anchor_voltage` in the old sweep changed only the NGP output
   layer width, not the objective.

3. **R² unmasked**. `conn_R2` now uses all edges (visible + hidden-touching).
   The old masked version is kept as `visible_R2` in parens. Don't chase
   the parenthetical — optimize for the main `conn_R2`.

4. **Rollout eval switched to training frames for hidden-NGP models**
   (graph_tester.py commit 861ba01). Prior `rollout_pearson` from
   hidden_010_ngp runs was out-of-distribution and systematically low;
   new split into `hidden_rollout_pearson` / `visible_rollout_pearson`.

5. **Consistency-loss degeneracy guard.** The new `coeff_hidden_voltage` can
   be trivially zeroed by `(NGP constant, GNN deriv = 0)`. The anchor loss
   pins the shared NGP trunk; the visible-neuron rollout loss drives the
   GNN derivative. Both must be active for the consistency loss to
   constrain the system — keep at least one of `coeff_anchor_voltage > 0`
   or the main loss (always active) on.

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

- `hidden_neuron_fraction: 0.1` → ~1200 hidden neurons (fixed sample, saved to
  `log_dir/hidden_neuron_ids.pt`; regenerated if absent).
- Hidden neurons are **excluded from the visible prediction loss** (`ids_batch`
  contains only visible neurons).
- NGP-T fills in hidden voltages at every forward pass:
  `x.voltage[hidden_ids] = model.forward_hidden(x, k, hidden_ids)`.
- Anchors are sampled from visible non-retina neurons (disjoint from hidden),
  saved to `log_dir/anchor_neuron_ids.pt` on first run.

## Explorable Parameters

### Loss Coefficients (main sweep axes)

| Parameter              | Default | Sweep values           | Note                                                      |
| ---------------------- | ------- | ---------------------- | --------------------------------------------------------- |
| `coeff_hidden_voltage` | 3000    | {0, 100, 1000, 3000}   | Now CONSISTENCY loss — smaller values likely sufficient   |
| `coeff_anchor_voltage` | 3000    | {0, 1000, 3000, 10000} | Direct anchor supervision of NGP trunk                    |
| `n_anchor`             | 3600    | {300, 1000, 3000, 3600}| More anchors ⇒ stronger trunk signal; diminishing returns |

### Learning Rates

| Parameter   | Default | Sweep values              | Note                                                         |
| ----------- | ------- | ------------------------- | ------------------------------------------------------------ |
| `lr_W`      | 1e-4    | {5e-5, 1e-4, 2e-4, 5e-4}  | GNN weight LR                                                |
| `lr`        | 1e-3    | {5e-4, 1e-3, 2e-3}        | GNN MLP LR                                                   |
| `lr_NNR_f`  | 1e-3    | {1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3} | **Same LR is applied to NNR_hidden (the NGP)** — wide band |
| `lr_embedding` | 1e-3 | do not sweep              | fixed                                                        |

Note: `lr_NNR_f` is the parameter name but drives both `NNR_f` and `NNR_hidden`
groups (see `models/utils.py:329`). For flyvis with no learned visual field, it
moves the NGP only.

### Batch Size (axis reopened)

| Parameter    | Default | Sweep values   | Note                                               |
| ------------ | ------- | -------------- | -------------------------------------------------- |
| `batch_size` | 16      | {8, 16, 32, 64}| Historical cap at 16; retest with anchor loss on  |

Scale `data_augmentation_loop` inversely with `batch_size` to keep the same
per-iteration budget.

### NGP Hashtable Encoding (architecture block — hashtable first)

The prior exploration tuned these with the oracle loss. Retest with the new
consistency+anchor setup.

| Parameter                         | Default | Sweep values          | Note                                |
| --------------------------------- | ------- | --------------------- | ----------------------------------- |
| `ngp_hidden_n_levels`             | 24      | {16, 20, 24, 28, 32}  | Number of multi-res grid levels     |
| `ngp_hidden_n_features_per_level` | 4       | {2, 4, 6, 8}          | Features per grid cell              |
| `ngp_hidden_base_resolution`      | 16      | {8, 16, 32}           | Coarsest grid resolution            |
| `ngp_hidden_per_level_scale`      | 1.4     | {1.2, 1.4, 1.7, 2.0}  | Resolution multiplier between levels|

MLP (`ngp_hidden_mlp_width`, `ngp_hidden_mlp_layers`) is **deferred** — focus
the architecture block on the hashtable first.

### Always-on GNN hyperparameters

| Parameter               | Default  | Description                                                    |
| ----------------------- | -------- | -------------------------------------------------------------- |
| `coeff_g_phi_diff`      | 750      | g_phi monotonicity regularizer                                 |
| `coeff_g_phi_norm`      | 5.0      | g_phi range constraint                                         |
| `coeff_W_L1`            | 5e-5     | L1 on W                                                        |
| `coeff_W_L2`            | 1.5e-6   | L2 on W                                                        |
| `alternate_training`    | true     | alternate-phase training (W vs MLP)                            |
| `alternate_lr_ratio`    | 0.4      | LR ratio during alternate phase                                |

## Parallel Mode — 4 Slots Per Batch

All 4 slots run with different random seeds (assigned automatically).
Use all 4 slots for seed robustness testing.

Data is **pre-generated at startup** (`claude.generate_data: false`) and
re-used across iterations. `simulation.seed` and `training.seed` are set by
the pipeline — DO NOT modify them.

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
predicts a **joint effect** — e.g., "lr_W and lr_NNR_f must move together
because they compete for the anchor-loss gradient". Single-parameter
sweeps are still the clearest signal; multi-parameter moves should cite
a specific interaction hypothesis. Either way, log which axes moved in
the `Mutation:` field.

## Budget

Each iteration targets **~120 min** wall-clock (`claude.training_time_target_min: 120`).
Calibrate `n_epochs` × `data_augmentation_loop` to stay within budget:
- bs=16, DAL=8, n_epochs=6  ≈ 120 min (starting point)
- scale DAL inversely with bs; scale n_epochs up if bs×DAL goes below ~256

## Block Partition (suggested)

| Block | Focus                      | Key axes                                                                 |
| ----- | -------------------------- | ------------------------------------------------------------------------ |
| 1     | **Anchor-loss baseline**   | Confirm conn_R² and hidden_nnr_pearson with default config               |
| 2     | **Anchor strength**        | `coeff_anchor_voltage` ∈ {0, 1000, 3000, 10000} — how much anchor helps  |
| 3     | **Anchor count**           | `n_anchor` ∈ {300, 1000, 3000, 3600} — diminishing returns of more trunk signal |
| 4     | **Consistency strength**   | `coeff_hidden_voltage` ∈ {0, 100, 1000, 3000} — consistency scale vs oracle-era |
| 5     | **Learning rate**          | `lr_W`, `lr`, `lr_NNR_f` scans — anchor loss changes gradient magnitude  |
| 6     | **Batch size**             | `batch_size` ∈ {8, 16, 32, 64} — retest with anchor loss                 |
| 7     | **Hashtable encoding**     | `n_levels`, `features_per_level`, `base_res`, `per_level_scale`          |
| 8     | **Combined best**          | Best of blocks 2–7                                                       |
| 9     | **Validation**             | Best config × 4 seeds, cross-check at n_epochs×DAL budget multipliers    |

## YAML Rules

> Always wrap the `description` field value in double quotes — colons inside
> unquoted YAML strings cause parse errors.

## Variable Names

- **`{base_config_name}`**: `flyvis_noise_005_hidden_010_ngp_anchors`
- **`{llm_task_name}`**: `flyvis_noise_005_hidden_010_ngp_anchors_Claude`

**Config file paths:**

- `config/fly/flyvis_noise_005_hidden_010_ngp_anchors_Claude_00.yaml` through `_03.yaml`
- `config/fly/flyvis_noise_005_hidden_010_ngp_anchors_winner.yaml`

## File Structure

### 1. Full Log (append-only)
**File**: `flyvis_noise_005_hidden_010_ngp_anchors_Claude_analysis.md`

### 2. Working Memory (read + update every batch)
**File**: `flyvis_noise_005_hidden_010_ngp_anchors_Claude_memory.md`

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
- Did hidden_nnr_pearson cross 0.3 / 0.5 / 0.65 (near ceiling)?
- Is anchor_nnr_pearson close to ceiling (~0.65)? If not, trunk is
  under-supervised.
- Is there a gap `anchor_nnr_pearson − hidden_nnr_pearson`? Large gap
  means trunk fits visible but hidden slots still fail → increase
  consistency weight or NGP capacity.
- `hidden_rollout_pearson` vs `visible_rollout_pearson`: hidden should
  trail visible; if `hidden < 0` check for diverging rollout at hidden
  neurons.

### Step 3: Write Log Entry

```
## Iter N: [stability] [improving/same/regressing vs previous]

Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: coeff_hv=A, coeff_av=B, n_anc=C, lr_W=D, lr=E, lr_NNR_f=F, bs=G, [arch: n_levels=L, feat=K, base=R, scale=S]
Slot 0: conn_R2=A, hid_pear=B, anc_pear=C, tau_R2=D, V_rest_R2=E, cluster=F, hid_rollout=G, vis_rollout=H, sim_seed=S, train_seed=T
Slot 1: ...
Slot 2: ...
Slot 3: ...
Seed stats: mean_conn_R2=X, std=Y, CV=Z%, min=W
           mean_hid_pear=X, std=Y, mean_anc_pear=X, std=Y
Stability: [Stable-Robust / Robust / Partially robust / Fragile / DISQUALIFIED]
Mutation: [params]: [old -> new]
Verdict: [supported/falsified/inconclusive] — [explanation]
Ceiling check: hid_pear ≈ ceiling (0.65)? anc_pear ≈ ceiling?
Next: parent=P
```

### Step 4: Acknowledge User Input

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Winner Config (COMPULSORY at every block boundary)

Save to `config/fly/flyvis_noise_005_hidden_010_ngp_anchors_winner.yaml` with header:

```yaml
# Winner config: flyvis_noise_005_hidden_010_ngp_anchors_winner.yaml
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
6. Compare hid_pear to standalone NGP ceiling (0.65); comment on headroom

## Known Results (reference)

- **Standalone NGP on GT traces** (hash: 24L×4f, base 16, scale 1.4, MLP 512×4):
  R²=0.428, Pearson≈0.65 — architectural ceiling.
- **No-hidden flyvis** (`flyvis_noise_005`): conn_R²≈0.980, CV<1%.
- **Prior hidden_010_ngp exploration (OBSOLETE — used oracle loss + broken anchor)**:
  best conn_R²≈0.65 at visible-only, hidden_nnr_R²<0 across 120 iters. Ignore
  these numbers; the loss shape is different now.

## Start Call

When prompt says `PARALLEL START`:

- Read base config `config/fly/flyvis_noise_005_hidden_010_ngp_anchors_Claude_00.yaml`
- Set all 4 configs identically to the baseline
- **Initial hypothesis**: "Consistency + anchor losses push hidden_nnr_pearson above 0.3 at iter 1 with default coefficients (3000/3000, n_anchor=3600)"
- **Null hypothesis**: "Default coefficients leave hidden_nnr_pearson < 0.2 — anchor trunk supervision alone doesn't rescue hidden slots"
- Write both hypotheses to working memory
- Block 1 tests the null hypothesis — no mutation yet, just 4 seeds of baseline

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_hidden_010_ngp_anchors

## Paper Summary (update at every block boundary)

- **Hidden-neuron inverse problem + NGP consistency/anchor**: [pending]
- **Architecture vs supervision trade-off**: [pending]

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
4. Does batch_size > 16 converge now that anchor loss provides dense supervision?

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
