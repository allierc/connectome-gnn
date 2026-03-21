# Connconstr Drosophila CX — LLM Exploration

## Goal

Maximize **connectivity_R2** (R² between learned W and ground-truth W) for the **Drosophila central complex ring attractor** model (Beiran & Litwin-Kumar 2023, Figure 5).

Current ceiling: **connectivity_R2 ≈ 0.5**. The goal is to break past this ceiling by systematic hyperparameter exploration.

Data is **re-generated each iteration** with a different seed to verify seed independence.

Primary metric: **connectivity_R2** (R² between learned W and ground-truth W).
Secondary metrics: **test_R2** (one-step prediction), **cluster_accuracy** (neuron type clustering from embeddings).

**NOTE**: tau_R2 and V_rest_R2 will always be 0.0 for this model — the CX ODE does not have explicit V_rest or tau_i parameters in the same sense as the flyvis model. Ignore these metrics.

## Scientific Method

This exploration follows a strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Based on available data, form a hypothesis about what will improve connectivity_R2
2. **Design experiment**: Choose a mutation that specifically tests the hypothesis — change ONE parameter at a time
3. **Run training**: The experiment runs across 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use both metrics AND cross-seed variance to evaluate
5. **Update understanding**: Revise hypotheses based on evidence

**CRITICAL**: You can only hypothesize. Only training results can validate or falsify your hypotheses.

## CRITICAL: Data is RE-GENERATED per slot

Each slot re-generates its data with a **different random seed**.
Both `simulation.seed` and `training.seed` are **forced by the pipeline** — DO NOT modify them in config files.

Seed formula (set automatically by GNN_LLM.py):

- `simulation.seed = iteration * 1000 + slot` (controls data generation)
- `training.seed = iteration * 1000 + slot + 500` (controls weight init & training randomness)

The actual seed values are provided in the prompt for each slot — **log them in your iteration entries**.

Simulation parameters (n_neurons, n_frames, etc.) stay fixed — **DO NOT change them**.

## CX Ring Attractor Model

The teacher model is a trained RNN implementing a ring attractor circuit:

```
dh/dt = α * (-h + exp(g_i) * softplus(h_j + b_j, β=5) @ J^T + input) / τ_i
```

- **152 neurons**, 4 cell types (EPG, PEN, Delta7, PEG), **9,722 edges**
- τ bounded [0.2, 5.0] via tanh: τ = 2.6 + 2.4·tanh(τ_raw)
- α = 0.2, β = 5 (softplus sharpness)
- Pretrained teacher weights from hemibrain connectivity
- 3,000 frames total, delta_t=0.1, with bump + velocity stimuli

**Key differences from flyvis**:
- Much smaller network (152 vs 13,741 neurons)
- Softplus activation (not ReLU) → g_phi should learn softplus-like curves
- No explicit V_rest → f_theta should learn pure decay: slope ≈ -α/τ_i
- The "corrected W" metric may not be meaningful (correction assumes ReLU); focus on **raw W R²**
- Only 4 cell types → embedding should separate 4 clusters

## GNN Architecture

Two MLPs learn the neural dynamics:

- **g_phi** (MLP1): Edge message function. Maps (v_j, a_j) → message. `g_phi_positive=true` squares output to enforce positivity.
- **f_theta** (MLP0): Node update function. Maps (v_i, a_i, aggregated_messages, I_i) → dv_i/dt.
- **Embedding a_i**: learnable low-dimensional embedding per neuron type.

Architecture parameters (explorable):

- `hidden_dim` / `n_layers`: g_phi MLP width/depth (default: 80 / 3)
- `hidden_dim_update` / `n_layers_update`: f_theta MLP width/depth (default: 80 / 3)
- `embedding_dim`: embedding dimension (default: 4)

**CRITICAL — coupled parameters**: When changing `embedding_dim`, you MUST also update:

- `input_size = 1 + embedding_dim` (v_j + a_j for g_phi)
- `input_size_update = 3 + embedding_dim` (v_i + a_i + msg + I_i for f_theta)

Example: embedding_dim=4 → input_size=5, input_size_update=7. Shape mismatch crashes otherwise.

## W Initialization — KEY EXPLORATION AXIS (Block 1)

The W initialization strategy may be critical for this smaller network. Test all three modes:

| `w_init_mode` | Description | Hypothesis |
| ------------- | ----------- | ---------- |
| `zeros` | All W start at 0 | Safe but may converge slowly |
| `randn_scaled` | W ~ N(0, scale/√n_edges) | May help escape local minima |
| `randn` | W ~ N(0, 1) | Likely too noisy for 9722 edges |

Also explore `w_init_scale` (default 1.0) when using `randn_scaled`.

## Regularization Parameters

| Config parameter          | Role                                                                | Default |
| ------------------------- | ------------------------------------------------------------------- | ------- |
| `coeff_g_phi_diff`        | Monotonicity penalty on g_phi: enforces increasing edge messages    | 1500    |
| `coeff_g_phi_norm`        | Normalization penalty on g_phi at saturation voltage                | 0       |
| `coeff_g_phi_weight_L1`   | L1 penalty on g_phi MLP weights                                    | 0.0     |
| `coeff_g_phi_weight_L2`   | L2 penalty on g_phi MLP weights                                    | 0       |
| `coeff_f_theta_weight_L1` | L1 penalty on f_theta MLP weights                                  | 0       |
| `coeff_f_theta_weight_L2` | L2 penalty on f_theta MLP weights                                  | 0.001   |
| `coeff_f_theta_msg_diff`  | Monotonicity of f_theta w.r.t. message input                       | 0       |
| `coeff_W_L1`              | L1 sparsity penalty on connectivity W                              | 0       |
| `coeff_W_L2`              | L2 penalty on W                                                    | 1.5e-06 |

### Regularization Annealing (Block 5 — requires n_epochs ≥ 2)

`regul_annealing_rate` controls an exponential ramp-up schedule for **all 6 L1/L2 weight regularizers** (W_L1, W_L2, g_phi_weight_L1, g_phi_weight_L2, f_theta_weight_L1, f_theta_weight_L2).

**Implementation** (see `src/flyvis_gnn/models/regularizer.py`):

```python
def anneal(coeff):
    return coeff * (1 - exp(-rate * epoch)) if rate > 0 else coeff

# These 6 coefficients are annealed:
W_L1 = anneal(coeff_W_L1)
W_L2 = anneal(coeff_W_L2)
g_phi_weight_L1 = anneal(coeff_g_phi_weight_L1)
g_phi_weight_L2 = anneal(coeff_g_phi_weight_L2)
f_theta_weight_L1 = anneal(coeff_f_theta_weight_L1)
f_theta_weight_L2 = anneal(coeff_f_theta_weight_L2)

# These are NOT annealed (always at full strength):
g_phi_diff = coeff_g_phi_diff          # monotonicity
g_phi_norm = coeff_g_phi_norm          # normalization
f_theta_msg_diff = coeff_f_theta_msg_diff  # message monotonicity
```

**Schedule** (`epoch` is the epoch index, starting at 0):

| rate | Epoch 0 | Epoch 1 | Epoch 2 | Epoch 5 | Epoch 10 |
| ---- | ------- | ------- | ------- | ------- | -------- |
| 0    | 100%    | 100%    | 100%    | 100%    | 100%     |
| 0.5  | 0%      | 39%     | 63%     | 92%     | 99%      |
| 1.0  | 0%      | 63%     | 86%     | 99%     | 100%     |
| 2.0  | 0%      | 86%     | 98%     | 100%    | 100%     |

**Purpose**: Allows the model to learn dynamics freely in early epochs before regularization pushes weights toward sparsity/smoothness. The idea is:
- Epoch 0: no L1/L2 → model fits the data without constraint
- Epoch 1+: L1/L2 ramps up → regularization cleans up weights, encourages sparsity

**CRITICAL — 1-epoch training**: With `n_epochs=1`, only epoch 0 runs, where `1 - exp(0) = 0`. ALL six L1/L2 regularizers are completely inactive regardless of their configured coefficients. Only the **non-annealed** coefficients (`coeff_g_phi_diff`, `coeff_g_phi_norm`, `coeff_f_theta_msg_diff`) have any effect.

**For Block 5**: Set `n_epochs: 2` to test annealing. With rate=0.5:
- Epoch 0: L1/L2 = 0 (model learns dynamics freely)
- Epoch 1: L1/L2 at 39% of configured values (regularization cleans up)

To keep training time comparable, halve `data_augmentation_loop` when doubling epochs.

## Training Parameters (explorable)

| Parameter                       | Default      | Description                                  |
| ------------------------------- | ------------ | -------------------------------------------- |
| `learning_rate_W_start`         | 6e-4         | Learning rate for connectivity matrix W      |
| `learning_rate_start`           | 1.8e-3       | Learning rate for g_phi and f_theta MLPs     |
| `learning_rate_embedding_start` | 1.55e-3      | Learning rate for neuron embeddings          |
| `n_epochs`                      | 1            | Keep at 1 except Block 5 (annealing test)    |
| `batch_size`                    | 2            | Batch size for training                      |
| `data_augmentation_loop`        | 20           | Data augmentation multiplier                 |
| `w_init_mode`                   | randn_scaled | W initialization: "zeros", "randn", or "randn_scaled" |
| `w_init_scale`                  | 1.0          | Scale factor for randn_scaled init           |
| `lr_scheduler`                  | none         | LR schedule: "none", "cosine_warm_restarts", "linear_warmup_cosine" |

## Training Time Constraint

This is a small model (152 neurons, 9722 edges). Training should be fast (~5-15 min per epoch on A100).
Keep total training time ≤ 60 min/iteration.

## Parallel Mode — 4 Slots Per Batch

You receive **4 results per batch** and propose **4 mutations** for the next batch.
Each slot runs with a **different random seed** for data generation, so you can directly assess seed robustness within a single batch.

### Robustness Assessment

- **Robust**: all 4 slots have connectivity_R2 > 0.7
- **Partially robust**: 2-3 slots have connectivity_R2 > 0.7
- **Fragile**: 0-1 slots have connectivity_R2 > 0.7

### Slot Strategy

All 4 slots should run the **same config** (different seeds are applied automatically).

### Config Files

- Edit all 4 config files: `{name}_00.yaml` through `{name}_03.yaml`
- **All 4 configs should be identical** (only seeds differ, set automatically)
- Only modify `training:` and `graph_model:` parameters (and `claude:` where allowed)
- **DO NOT change `simulation:` parameters** (except that seed is managed automatically)

## Iteration Loop Structure

Each block = `n_iter_block` iterations (default 12).
The prompt provides: `Block info: block {block_number}, iterations {iter_in_block}/{n_iter_block} within block`

## File Structure

You maintain **THREE** files:

### 1. Full Log (append-only)

**File**: `{llm_task_name}_analysis.md`

- Append every iteration's log entry (4 entries per batch)
- Append block summaries at block boundaries
- **Never read** — human record only

### 2. Working Memory (read + update every batch)

**File**: `{llm_task_name}_memory.md`

- Read at start, update at end
- Contains: robustness comparison table, hypotheses, established principles, current block iterations
- Keep ≤ 500 lines

### 3. User Input (read every batch, acknowledge pending items)

**File**: `user_input.md`

- Read at every batch
- If "Pending Instructions" section has content: act on it, then move entries to "Acknowledged" section with timestamp

## Iteration Workflow (every batch)

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

**Metrics from `analysis.log`:**

- `connectivity_R2`: R² of learned vs true W (PRIMARY)
- `cluster_accuracy`: neuron type clustering accuracy from embeddings
- `test_R2`: one-step prediction R²
- `training_time_min`: training duration

**Note**: tau_R2 and V_rest_R2 are always 0.0 — ignore them.

**Robustness classification (across all 4 seeds):**

- **Robust**: all 4 slots connectivity_R2 > 0.7
- **Partially robust**: 2-3 slots connectivity_R2 > 0.7
- **Fragile**: 0-1 slots connectivity_R2 > 0.7

**Seed variance analysis (compute every batch):**

- Compute mean, std, and CV (coefficient of variation = std/mean) for connectivity_R2 across the 4 slots
- CV < 5% → highly stable; CV 5-15% → moderate variance; CV > 15% → seed-sensitive

### Step 3: Write Log Entries + Update Memory

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config (same for all slots): lr_W=X, lr=Y, lr_emb=Z, w_init_mode=M, coeff_g_phi_diff=A, hidden_dim=D, batch_size=B
Slot 0: connectivity_R2=A, cluster_accuracy=B, test_R2=C, sim_seed=S, train_seed=T
Slot 1: connectivity_R2=A, cluster_accuracy=B, test_R2=C, sim_seed=S, train_seed=T
Slot 2: connectivity_R2=A, cluster_accuracy=B, test_R2=C, sim_seed=S, train_seed=T
Slot 3: connectivity_R2=A, cluster_accuracy=B, test_R2=C, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive]
Next: parent=P
```

### Step 4: Acknowledge User Input

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Block Partition

| Block | Focus                | Parameters                                                               |
| ----- | -------------------- | ------------------------------------------------------------------------ |
| 1     | W initialization     | w_init_mode (zeros, randn_scaled, randn), w_init_scale                   |
| 2     | Learning rates       | lr_W, lr, lr_emb — especially lr_W which drives W recovery              |
| 3     | Batch size           | batch_size (1, 2, 4, 8) — gradient noise vs. stability tradeoff          |
| 4     | Regularization       | coeff_g_phi_diff, coeff_W_L1, coeff_W_L2, coeff_f_theta_weight_L2       |
| 5     | Multi-epoch + anneal | n_epochs=2, test regul_annealing_rate (0, 0.5, 1.0, 2.0). Halve data_augmentation_loop to keep time constant. |
| 6     | Combined best        | Best parameters from blocks 1–5                                         |
| 7     | MLP size             | hidden_dim (32, 64, 80, 128), n_layers (2, 3, 4), hidden_dim_update     |
| 8     | Free exploration     | Any parameter — test novel hypotheses, combinations, LR schedulers, etc. |

### Block 5 details — Annealing Rate Exploration

Block 5 is specifically designed to test whether **2-epoch training with regularization annealing** improves connectivity_R2 over 1-epoch training.

Set `n_epochs: 2` and halve `data_augmentation_loop` (e.g., 20 → 10) to keep total iteration time ≤ 60 min.

Test `regul_annealing_rate` values:
- **0** (no annealing): L1/L2 at full strength from epoch 0
- **0.5** (default): epoch 0 = 0%, epoch 1 = 39%
- **1.0** (faster): epoch 0 = 0%, epoch 1 = 63%
- **2.0** (aggressive): epoch 0 = 0%, epoch 1 = 86%

The hypothesis: annealing allows epoch 0 to fit data freely, then epoch 1 applies regularization to clean up — this 2-phase strategy may break the R² ceiling.

## Block Boundaries

1. Update "Paper Summary"
2. Summarize findings
3. Update "Established Principles"
4. Clear "Current Block"
5. Carry forward best config

## Known Results (prior experiments)

- Default config (flyvis_noise_free params): connectivity_R2 ≈ 0.5 after 1 epoch
- g_phi learns 4 distinct softplus-like curves (correct — matches 4 cell types)
- f_theta learns linear negative slope ≈ -0.2 (consistent with -α/τ decay)
- Training is fast (~15 min/epoch on A100)

## Start Call

When prompt says `PARALLEL START`:

- Read base config
- Set all 4 configs **identically** to baseline
- Data will be generated with different seeds per slot automatically
- First iteration establishes baseline — do not change hyperparameters yet
- State the baseline hypothesis: "The default config achieves connectivity_R2 > 0.7 robustly across seeds"

---

# Working Memory Structure

The memory file (`{llm_task_name}_memory.md`) must follow this structure:

```markdown
# Working Memory: drosophila_cx

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]
- **Future works**: [pending]

## Knowledge Base (accumulated across all blocks)

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean±std) | CV% | min | max | Robust? | Hypothesis tested |
| ---- | -------------- | ------------------ | --- | --- | --- | ------- | ----------------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

---

## Previous Block Summary

---

## Current Block

### Block Info

### Current Hypothesis

**Hypothesis**: [specific, testable prediction]
**Rationale**: [why you believe this]
**Test**: [what config change tests this]
**Expected outcome**: [what would support vs falsify]
**Status**: untested / supported / falsified / revised

### Iterations This Block

### Emerging Observations
**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
