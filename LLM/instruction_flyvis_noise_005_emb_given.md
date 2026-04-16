# FlyVis GNN Training Exploration — flyvis_noise_005_emb_given

## Goal

Optimize hyperparameters for the **Drosophila visual system GNN** with **cell-type-structured embedding initialization** and a **cluster regularizer** that pulls same-type embeddings toward their per-type centroid.

The primary challenge is to **recover rollout performance** (currently 0.235 — collapsed) while **preserving tau_R² ≥ 0.9 and V_rest_R² ≥ 0.3**, which represent genuinely novel results enabled by the structured embedding.

The known starting point (1 run, seed 42) is:
- `rollout_r = 0.235` — collapsed (hard failure)
- `tau_R² = 0.936` — excellent, must preserve
- `V_rest_R² = 0.353` — very good, must preserve
- `effective_W_R² = 0.700` — connectivity partially recovered

**PRIMARY OBJECTIVE: Fix rollout collapse.**
**SECONDARY OBJECTIVE: Maintain or improve tau_R² ≥ 0.9 and V_rest_R² ≥ 0.3.**
**TERTIARY OBJECTIVE: Maximize effective_W_R² (connectivity recovery).**

A config is **successful** when: `rollout_r > 0.85` AND `tau_R² ≥ 0.85` AND `V_rest_R² ≥ 0.25`.
A config is **excellent** when: `rollout_r > 0.95` AND `tau_R² ≥ 0.9` AND `V_rest_R² ≥ 0.3`.

The **root cause hypothesis** at the start of exploration is that `coeff_embedding_cluster = 1.0` is too strong — it pulls embeddings into tight clusters that deform the dynamics landscape and break rollout. This must be tested first.

## Scientific Context

This exploration adds a **structural prior** to GNN training: cell-type identity is used to initialize the embedding (`embedding_cell_type_init: true`) with equidistant Fibonacci-spiral points (one point per type), scaled by `embedding_cell_type_scale: 2.0`. A cluster regularizer (`coeff_embedding_cluster`) then penalizes same-type embeddings that drift too far from their per-type centroid, allowing centroids to move freely while maintaining intra-type cohesion.

The scientific question is: **Can this structural prior improve parameter recovery (tau, V_rest, W) compared to random embedding initialization, without sacrificing rollout dynamics quality?** The structured embedding may help the model discover cell-type-specific dynamics, but the cluster regularizer strength is a delicate balance — too strong collapses dynamics; too weak loses the structural benefit.

This is a 6-block exploration. Each block targets a specific axis of the problem. Results from each block directly inform the next.

## Noise Model

Same as `flyvis_noise_005`: additive Gaussian noise level 0.05 applied during data generation.

$$v_i(t+1) = v_i(t) + dt \cdot f_\theta(v_i, a_i, \sum_j W_{ij} g_\phi(v_j, a_j)^2, I_i) + \epsilon_{\text{dyn}}(t)$$

where $\epsilon_{\text{dyn}}(t) \sim \mathcal{N}(0, 0.05^2)$.

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

- 13,741 neurons, **65 cell types**, 434,112 edges
- 1,736 input neurons (photoreceptors)
- DAVIS visual input, **noise_model_level=0.05**
- 64,000 frames, delta_t=0.02

## GNN Architecture

- **g_phi** (MLP1): Edge message function. `g_phi_positive=true` squares output to enforce positivity.
- **f_theta** (MLP0): Node update function.
- **Embedding a_i**: learnable 2D embedding per neuron, **initialized with equidistant points per cell type**.

Current architecture (from `flyvis_noise_005_emb_given.yaml`):
- `n_layers: 3`, `hidden_dim: 80` (g_phi)
- `n_layers_update: 3`, `hidden_dim_update: 80` (f_theta)
- `embedding_dim: 2`

**CRITICAL — coupled parameters**: When changing `embedding_dim`, you MUST also update:
- `input_size = 1 + embedding_dim`
- `input_size_update = 3 + embedding_dim`

## Embedding Parameters (NEW — key to this config)

| Parameter                    | Current value | Description                                                                          |
| ---------------------------- | ------------- | ------------------------------------------------------------------------------------ |
| `embedding_cell_type_init`   | `true`        | Initialize `model.a` with Fibonacci-spiral equidistant points, one per cell type    |
| `embedding_cell_type_scale`  | `2.0`         | Radius multiplier for equidistant points (points land at radius ≤ 2.0 in emb space) |
| `fix_embedding`              | `false`       | If true: freeze `model.a` completely (no gradient). Currently off.                  |
| `coeff_embedding_cluster`    | `1.0`         | L2 penalty pulling same-type embeddings toward their per-type centroid               |

**Embedding cluster regularizer** formula (computed per neuron, summed):

$$\mathcal{L}_{\text{cluster}} = \text{coeff\_embedding\_cluster} \times \sum_i \| a_i - \bar{a}_{\text{type}(i)} \|_2$$

where $\bar{a}_{\text{type}(i)}$ is the centroid of all embeddings of the same cell type, computed on-the-fly. The centroid is free to drift — this regularizer enforces cohesion within a type but does not fix the centroid position.

**Expected behavior**:
- `coeff = 1.0`: very strong pull → tight clusters → may constrain dynamics representation → rollout collapse
- `coeff = 0.1`: moderate pull → clusters remain coherent but deformable → likely preserves rollout
- `coeff = 0.01`: weak pull → light bias → minimal effect on dynamics
- `coeff = 0.0`: disabled → pure random-walk from structured init (test whether init alone helps)

## Regularization Parameters

| Config parameter          | Role                                                                                | Default | Annealed? |
| ------------------------- | ----------------------------------------------------------------------------------- | ------- | --------- |
| `coeff_g_phi_diff`        | Monotonicity penalty on g_phi: ReLU(-dg_phi/dv)                                    | 750     | No        |
| `coeff_g_phi_norm`        | Normalization penalty on g_phi at saturation voltage                                | 0.9     | No        |
| `coeff_g_phi_weight_L1`   | L1 penalty on g_phi MLP weights                                                     | 0.28    | **Yes**   |
| `coeff_g_phi_weight_L2`   | L2 penalty on g_phi MLP weights                                                     | 0       | **Yes**   |
| `coeff_f_theta_weight_L1` | L1 penalty on f_theta MLP weights                                                   | 0.05    | **Yes**   |
| `coeff_f_theta_weight_L2` | L2 penalty on f_theta MLP weights                                                   | 0.001   | **Yes**   |
| `coeff_W_L1`              | L1 sparsity penalty on connectivity W                                               | 7.5e-05 | **Yes**   |
| `coeff_W_L2`              | L2 penalty on W                                                                     | 1.5e-06 | **Yes**   |
| `coeff_embedding_cluster` | L2 pull toward per-type centroid for `model.a`                                      | 1.0     | No        |

### Regularization Annealing

The 6 weight regularization coefficients (L1 and L2 for g_phi, f_theta, W) share a single exponential ramp-up:

| Config parameter       | Default | Description                                      |
| ---------------------- | ------- | ------------------------------------------------ |
| `regul_annealing_rate` | 0.5     | Shared annealing rate for all L1/L2 regularizers |

**Formula**: `effective_coeff = coeff * (1 - exp(-rate * epoch))`

**IMPORTANT**: `coeff_embedding_cluster` is **NOT annealed** — it applies at full strength from epoch 0. This is critical: the cluster penalty is active even during early training when the model is most sensitive to gradient landscape deformation.

## Explorable Parameters

| Parameter                       | Current value | Description                                                         |
| ------------------------------- | ------------- | ------------------------------------------------------------------- |
| `coeff_embedding_cluster`       | 1.0           | **PRIMARY**: cluster regularizer strength — reduce to fix rollout   |
| `embedding_cell_type_scale`     | 2.0           | Init radius — may affect gradient scale early in training           |
| `learning_rate_W_start`         | 6e-4          | Learning rate for W                                                  |
| `learning_rate_start`           | 1.2e-3        | Learning rate for g_phi and f_theta                                 |
| `learning_rate_embedding_start` | 2.325e-3      | Learning rate for neuron embeddings (lr_emb in `lr_embedding`)      |
| `coeff_g_phi_diff`              | 750           | Monotonicity — too low causes non-monotonic messages                 |
| `coeff_W_L1`                    | 7.5e-05       | W sparsity                                                           |
| `recurrent_training`            | false         | Enable multi-step rollout training (may directly improve rollout)   |
| `time_step`                     | 1             | Recurrent steps if recurrent_training=true                          |
| `n_epochs`                      | 2             | Epochs per iteration                                                 |
| `data_augmentation_loop`        | 20            | Data augmentation multiplier                                        |

> **YAML rule**: Always wrap the `description` field value in double quotes — colons inside unquoted YAML strings cause parse errors.

## CRITICAL: Data is PRE-GENERATED at startup (fixed across iterations)

At startup, data is generated **once** for all 4 slots with **different random seeds**. These datasets are **reused across all iterations**.

Both `simulation.seed` and `training.seed` are **forced by the pipeline** — DO NOT modify them in config files.

Seed formula (set automatically by GNN_LLM.py):
- `simulation.seed = 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

## Parallel Mode — 4 Slots Per Batch

Each batch runs 4 configs with different seeds, allowing within-batch seed robustness assessment.

### Robustness Classification

After each batch, classify using **rollout_r AND parameter R² metrics across all 4 seeds**:

- **Excellent**: all 4 slots: rollout_r > 0.95 AND tau_R² ≥ 0.9 AND V_rest_R² ≥ 0.3 — **TARGET**
- **Good**: all 4 slots: rollout_r > 0.85 AND tau_R² ≥ 0.85 — **SUCCESS**
- **Partial**: rollout_r > 0.7 on 2-3 slots — some recovery, investigate
- **Collapsed**: rollout_r < 0.5 on all/most slots — rollout still broken, reject
- **DISQUALIFIED**: rollout_r < 0.3 on any slot AND tau_R² < 0.7 — complete failure

**If rollout_r collapses but tau/V_rest remain high**: this is a dissociation (embedding helps parameter recovery but hurts dynamics) — investigate via coeff_embedding_cluster sweep.

**If tau_R² drops below 0.7** when reducing coeff_embedding_cluster: structured embedding has lost its advantage — consider re-enabling init without regularizer.

### Slot Strategy

All 4 slots run the **same config** (seeds differ automatically) to assess seed robustness.

### Config Files

- Edit: `config/fly/{base_config_name}_Claude_00.yaml` through `config/fly/{base_config_name}_Claude_03.yaml`
- Winner: `config/fly/{base_config_name}_winner.yaml`

where `{base_config_name}` = `flyvis_noise_005_emb_given`.

## Variable Names

- **`{base_config_name}`** = `flyvis_noise_005_emb_given`
- **`{llm_task_name}`** = `flyvis_noise_005_emb_given_Claude`

## Iteration Loop Structure

Each block = `n_iter_block` iterations (default 12).

## File Structure

### 1. Full Log (append-only)

**File**: `{llm_task_name}_analysis.md`

### 2. Working Memory (read + update every batch)

**File**: `{llm_task_name}_memory.md`

### 3. User Input (read every batch)

**File**: `user_input.md`

## Iteration Workflow (every batch)

### Step 1: Read Working Memory + User Input

- Read `{llm_task_name}_memory.md`
- Read `user_input.md` for pending instructions

### Step 2: Analyze Results (4 slots)

**Metrics from `analysis.log`:**

- `rollout_r`: Pearson r of rollout prediction (**PRIMARY** — must recover from 0.235)
- `connectivity_R2` (or `effective_W_R2`): R² of learned W vs true W
- `tau_R2`: R² of learned time constants (**must preserve ≥ 0.9**)
- `V_rest_R2`: R² of learned resting potentials (**must preserve ≥ 0.3**)
- `cluster_accuracy`: neuron type clustering accuracy from embeddings
- `test_R2`: one-step prediction R²
- `training_time_min`: training duration

**Robustness classification (across all 4 seeds):**

- **Excellent**: all 4 slots: rollout_r > 0.95 AND tau_R² ≥ 0.9
- **Good**: all 4 slots: rollout_r > 0.85 AND tau_R² ≥ 0.85
- **Partial**: rollout_r > 0.7 on 2-3 slots
- **Collapsed**: rollout_r < 0.5 on all/most slots
- **DISQUALIFIED**: any slot with rollout_r < 0.3 AND tau_R² < 0.7

**Seed variance analysis (compute every batch):**

- Compute mean, std, CV for rollout_r AND tau_R² across 4 slots
- Report min rollout_r (critical — if any seed collapses, note separately)

### Step 3: Write Log Entries + Update Memory

**3a. Append to Full Log** (`{llm_task_name}_analysis.md`) and **Current Block** in memory.md:

```
## Iter N: [Excellent/Good/Partial/Collapsed/DISQUALIFIED]

Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config (same for all slots): coeff_emb_cluster=A, emb_scale=B, lr_W=C, lr=D, lr_emb=E
Slot 0: rollout_r=A, tau_R2=B, V_rest_R2=C, W_R2=D, test_R2=E, sim_seed=S, train_seed=T
Slot 1: rollout_r=A, tau_R2=B, V_rest_R2=C, W_R2=D, test_R2=E, sim_seed=S, train_seed=T
Slot 2: rollout_r=A, tau_R2=B, V_rest_R2=C, W_R2=D, test_R2=E, sim_seed=S, train_seed=T
Slot 3: rollout_r=A, tau_R2=B, V_rest_R2=C, W_R2=D, test_R2=E, sim_seed=S, train_seed=T
Seed stats: mean_rollout=X, std=Y, CV=Z%, min_rollout=W; mean_tau=X, std=Y
Stability: [Excellent/Good/Partial/Collapsed/DISQUALIFIED]
Mutation: [param]: [old] -> [new]
Verdict: [recovered/partially/unchanged/worsened] — [one line]
Observation: [one line about tau/V_rest preservation or rollout pattern]
Next: parent=P
```

**3b. Update Hypotheses in memory.md.**

**3c. Update Comparison Table in memory.md** (see Working Memory Structure below).

### Step 4: Acknowledge User Input (if any)

Move pending items to "Acknowledged" with `[ACK batch_N]` marker.

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

1. Formulate next hypothesis
2. Design ONE parameter change to test it
3. All 4 configs identical — seeds assigned automatically
4. Write hypothesis to memory.md before editing configs

## Block Partition (6 blocks)

| Block | Focus                          | Parameters to test                                                                          |
| ----- | ------------------------------ | ------------------------------------------------------------------------------------------- |
| 1     | Coeff cluster sweep            | `coeff_embedding_cluster`: 1.0 → 0.1 → 0.01 → 0.0 — **root cause test**                  |
| 2     | Init vs regularizer dissection | `embedding_cell_type_init` on/off × `coeff_embedding_cluster` at best value from block 1   |
| 3     | Learning rate tuning           | `lr_embedding`, `lr_W`, `lr` — optimize for joint rollout + parameter recovery             |
| 4     | Regularization fine-tuning     | `coeff_g_phi_diff`, `coeff_W_L1`, `recurrent_training` — improve W and rollout jointly     |
| 5     | Embedding scale + architecture | `embedding_cell_type_scale`, `hidden_dim`, `n_layers` — squeeze out remaining gains        |
| 6     | Best combined + validation     | Best params from blocks 1–5 combined; run across 4+ seeds to confirm stability             |

### Block 1 Detail — Coeff Cluster Sweep (ROOT CAUSE)

This is the most critical block. The known result at `coeff=1.0` (rollout_r=0.235) is **already confirmed** — do NOT repeat it. Start immediately with `coeff=0.1`.

Test systematically, starting from the first iteration:
- `coeff=0.1` — primary fix candidate
- `coeff=0.01` — weaker regularization
- `coeff=0.0` (disable completely — does structured init alone help without regularizer?)

**Decision rule**:
- If rollout recovers at `coeff=0.1` → carry forward `coeff=0.1` to block 2
- If rollout recovers only at `coeff=0.01` → that's the sweet spot, investigate why in block 2
- If rollout only recovers at `coeff=0.0` → the regularizer is always harmful; structured init alone may suffice

Track **tau_R² and V_rest_R²** carefully at each step — if they degrade as coeff decreases, there is a fundamental trade-off between parameter recovery and rollout quality.

### Block 2 Detail — Init vs Regularizer Dissection

Using the best `coeff_embedding_cluster` from block 1, test:
- Init ON + coeff at best value (baseline)
- Init OFF + coeff at best value (does init matter or is regularizer doing all the work?)
- Init ON + coeff=0.0 (does structured init alone provide any tau/V_rest benefit?)
- Init OFF + coeff=0.0 (standard flyvis_noise_005 baseline for comparison)

**Decision rule**: If tau_R² advantage disappears when coeff→0, the regularizer is the mechanism. If it persists even without regularizer, structured init alone carries value.

### Block 3 Detail — Learning Rate Tuning

After fixing the rollout collapse, fine-tune the three learning rates:
- `learning_rate_embedding_start` (lr_emb): embedding LR was 2.325e-3 — try lower (1e-3) and higher (5e-3)
- `learning_rate_W_start` (lr_W): try 3e-4, 6e-4, 1e-3
- `learning_rate_start` (lr): MLPs — try 6e-4, 1.2e-3, 2e-3

### Block 4 Detail — Regularization Fine-Tuning

Focus on:
- `coeff_g_phi_diff`: critical for dynamics quality — test 500, 750, 1000
- `coeff_W_L1`: affects W sparsity and recovery — test 3e-5, 7.5e-5, 2e-4
- `recurrent_training: true` with `time_step=2`: multi-step rollout training may directly fix rollout

### Block 5 Detail — Embedding Scale + Architecture

- `embedding_cell_type_scale`: try 1.0, 1.5, 2.0 (current), 3.0 — does larger init radius help?
- `hidden_dim`: test 64, 80 (current), 96 — balanced against training time
- `n_layers`: test 2 (lighter) vs 3 (current)

### Block 6 Detail — Best Combined + Validation

Combine top insights from blocks 1–5. Run best config across all 4 seeds with the goal of achieving Excellent classification. If Good is achieved, accept as validated.

## Block Boundaries

At the end of each block:

1. Update "Paper Summary" at top of memory.md — rewrite both bullets to reflect current state
2. Summarize findings in "Previous Block Summary"
3. Update "Established Principles" with confirmed insights (require 3+ supporting iterations)
4. Move falsified hypotheses to "Falsified Hypotheses"
5. Clear "Current Block" for next block
6. **Save winner config** (see below)

## Winner Config (COMPULSORY at every block boundary)

1. Identify best iteration (highest rollout_r that also preserves tau_R² ≥ 0.85)
2. Copy its config from `log/Claude_exploration/LLM_{llm_task_name}/config/iter_XXX_slot_YY.yaml`
3. Write to `config/fly/{base_config_name}_winner.yaml` with header:

```yaml
# Winner config: flyvis_noise_005_emb_given_winner.yaml
# Source: iter_XXX_slot_YY (rollout_r = X.XXX, tau_R2 = X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [narrative: what fixed rollout collapse while preserving parameter recovery]
#   - [key coeff_embedding_cluster value and why it works]
#
# Metrics:
#   rollout_r:     X.XXX (best single seed)
#   tau_R2:        X.XXX (must stay ≥ 0.9)
#   V_rest_R2:     X.XXX (must stay ≥ 0.3)
#   effective_W_R2: X.XXX
#   cluster_accuracy: X.XXX
#
# Key config differences from baseline flyvis_noise_005_emb_given:
#   - coeff_embedding_cluster: [changed from 1.0 to X]
#   - [other changed params]
```

## Failed Slots

If a slot is `[FAILED]`:
- Note in log entry
- A single slot failure may indicate seed sensitivity — note separately
- Still propose next config
- Do not draw conclusions from single failure

## Known Results (prior experiments)

- `flyvis_noise_005` baseline: connectivity_R2=0.95, tau_R2=0.80, V_rest_R2=0.40
- `flyvis_noise_005_emb_given` with `coeff_embedding_cluster=1.0`: rollout_r=0.235 (collapsed), tau_R2=0.936, V_rest_R2=0.353, effective_W_R2=0.700
- The structured embedding init improved tau_R2 (+0.136) and V_rest_R2 (-0.047 → improvement from ~0.30) vs baseline — this benefit must be preserved
- Root cause hypothesis: coeff=1.0 forces tight clusters that constrain the learned dynamics representation
- W initialization: `randn_scaled` and `zeros` perform similarly; plain `randn` performs poorly
- Larger MLP (80-dim/3-layer) works better than smaller (32-dim/2-layer)
- `coeff_g_phi_diff` (monotonicity) is among the most important regularizers

## Start Call

When prompt says `PARALLEL START`:

- Read base config `config/fly/flyvis_noise_005_emb_given.yaml` to understand training regime
- Set all 4 configs to `coeff_embedding_cluster=0.1` — **do NOT use 1.0**, that result is already known (rollout_r=0.235, collapsed)
- Write **initial hypothesis** to working memory: "reducing coeff_embedding_cluster from 1.0 to 0.1 will recover rollout_r > 0.85 while preserving tau_R² ≥ 0.9"
- Data is generated with different seeds per slot automatically

---

# Working Memory Structure

The memory file (`flyvis_noise_005_emb_given_Claude_memory.md`) must follow this structure:

```markdown
# Working Memory: flyvis_noise_005_emb_given

## Paper Summary (update at every block boundary)

- **Structured embedding**: [pending — does cell-type init + cluster regularizer improve parameter recovery?]
- **Rollout recovery**: [pending — what coeff_embedding_cluster value restores rollout_r > 0.9?]

## Knowledge Base (accumulated across all blocks)

### Comparison Table

| Iter | coeff_emb_cluster | rollout_r (mean±std) | tau_R2 (mean) | V_rest_R2 (mean) | W_R2 (mean) | Stability | Hypothesis tested |
| ---- | ----------------- | -------------------- | ------------- | ---------------- | ----------- | --------- | ----------------- |
| 0    | 1.0 (known)       | 0.235 (1 seed)       | 0.936         | 0.353            | 0.700       | Collapsed | baseline          |

### Established Principles

[Confirmed patterns — require 3+ supporting iterations AND cross-seed consistency]

### Falsified Hypotheses

[Hypotheses contradicted by evidence]

### Open Questions

[Patterns needing more testing]

---

## Previous Block Summaries

**RULE: Keep summaries for last 4 completed blocks, sorted oldest→newest. This section MUST appear before ## Current Block.**

---

## Current Block (Block N)

### Block Info

Focus: [parameter subspace]
Iterations: M to M+n_iter_block

### Current Hypothesis

**Hypothesis**: [specific, testable prediction]
**Rationale**: [why, based on prior evidence]
**Test**: [what config change tests this]
**Expected outcome**: [what would support vs falsify]
**Success criterion**: rollout_r > 0.85 on all 4 seeds AND tau_R² ≥ 0.85 preserved
**Status**: untested / supported / falsified / revised

### Iterations This Block

[Current block iterations — cleared at block boundary]

### Emerging Observations

[Running notes on rollout/parameter recovery trade-off across seeds and iterations]
**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
