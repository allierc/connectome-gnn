# Connconstr Zebrafish Oculomotor — LLM Exploration

## Goal

Maximize **connectivity_R2** (R² between learned W and ground-truth W) for the **zebrafish oculomotor integrator** model (Beiran & Litwin-Kumar 2023, Figure 5g-i).

Data is **re-generated each iteration** with a different seed to verify seed independence.

Primary metric: **connectivity_R2** (effective W R² — learned W×g_phi_gain vs true W×gain).
Secondary metrics: **rollout_pearson** (autoregressive rollout Pearson r), **f_theta_functional_R2** (functional curve match), **g_phi_functional_R2** (functional curve match), **cluster_accuracy** (neuron type clustering).
Informational metrics (not for optimization): **onestep_pearson** (one-step prediction — easy to fit, less discriminative).

**NOTE**: tau_R2 is not applicable — the zebrafish ODE has fixed τ=1. V_rest_R2 is also not applicable.

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

## Zebrafish Oculomotor Integrator Model

The teacher model is a **linear** RNN implementing an oculomotor integrator:

```
dr/dt = (-r + W @ r + I(t) * v_in) / tau
```

- **609 neurons**, 6 cell types (_Int_, _DOs_, _Axl_, ABD_m, ABD_i, vSPNs), from Goldman lab connectome
- **LINEAR**: no activation function (identity g_phi)
- τ = 1.0 (fixed, not learned)
- dt = 0.001
- W scaled to spectral radius = 0.9
- Stimulus: filtered pulse (exponential kernel), scalar I(t) distributed via per-neuron v_in vector
- 21,000 frames total (3 pulse repeats × 7,000 frames)

**Key differences from flyvis**:
- **Linear ODE** — no activation function. g_phi should learn identity (slope=1)
- Medium network (609 neurons) but with many zero'd populations
- No learnable tau or V_rest → f_theta should learn f(v) = -v (slope=-1, offset=0)
- 6 cell types but some have zeroed connections (ABD, axial, vSPNs)
- Spectral radius normalization is critical — the dynamics are determined by W eigenstructure

**Key differences from CX and larva**:
- Linear (no softplus/ReLU) → simplest activation but challenging because dynamics are purely determined by W eigenvalues
- Larger (609 neurons) but sparser connectivity after zeroing
- No per-neuron gain/bias to recover (no tau, no V_rest, no gain)
- The main challenge is recovering the W matrix structure

**What connectivity_R2 measures here**:
- Since g_phi is identity, the "effective W" correction simplifies to just the learned W vs true W
- The model must recover the spectral structure (eigenvalues/eigenvectors) of W to achieve high R²

## GNN Architecture

Two MLPs learn the neural dynamics:

- **g_phi**: Edge message function. Maps (v_j, a_j) → message. Should learn identity for v_j.
- **f_theta**: Node update function. Maps (v_i, a_i, aggregated_messages, I_i) → dv_i/dt. Should learn -v_i + msg + stim.
- **Embedding a_i**: learnable low-dimensional embedding per neuron type.

Architecture parameters (explorable):

- `hidden_dim` / `n_layers`: g_phi MLP width/depth (default: 64 / 3)
- `hidden_dim_update` / `n_layers_update`: f_theta MLP width/depth (default: 64 / 3)
- `embedding_dim`: embedding dimension (default: 2)

**CRITICAL — coupled parameters**: When changing `embedding_dim`, you MUST also update:

- `input_size = 1 + embedding_dim` (v_j + a_j for g_phi)
- `input_size_update = 3 + embedding_dim` (v_i + a_i + msg + I_i for f_theta)

Example: embedding_dim=2 → input_size=3, input_size_update=5. Shape mismatch crashes otherwise.

**g_phi_positive**: Set to `false` (default). The linear ODE has no positivity constraint on messages. The identity function g_phi(v) = v passes through negative values.

## W Initialization — KEY EXPLORATION AXIS (Block 1)

The W initialization strategy may be critical for this larger network. Test all three modes:

| `w_init_mode` | Description | Hypothesis |
| ------------- | ----------- | ---------- |
| `zeros` | All W start at 0 | Safe but may converge slowly |
| `randn_scaled` | W ~ N(0, scale/√n_edges) | May help escape local minima |
| `randn` | W ~ N(0, 1) | Likely too noisy |

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
| `coeff_W_sign`            | Dale's law penalty — penalizes mixed-sign outgoing weights per neuron | 0     |

### Regularization Annealing (Block 5 — requires n_epochs ≥ 2)

`regul_annealing_rate` controls an exponential ramp-up schedule for **all 6 L1/L2 weight regularizers**.

**CRITICAL — 1-epoch training**: With `n_epochs=1`, only epoch 0 runs, where `1 - exp(0) = 0`. ALL six L1/L2 regularizers are completely inactive regardless of their configured coefficients.

## Training Parameters (explorable)

| Parameter                       | Default      | Description                                  |
| ------------------------------- | ------------ | -------------------------------------------- |
| `learning_rate_W_start`         | 1e-3         | Learning rate for connectivity matrix W      |
| `learning_rate_start`           | 1e-3         | Learning rate for g_phi and f_theta MLPs     |
| `learning_rate_embedding_start` | 1e-3         | Learning rate for neuron embeddings          |
| `n_epochs`                      | 1            | Keep at 1 except Block 5 (annealing test)    |
| `batch_size`                    | 4            | Batch size for training                      |
| `data_augmentation_loop`        | 500          | Data augmentation multiplier                 |
| `w_init_mode`                   | randn_scaled | W initialization: "zeros", "randn", or "randn_scaled" |
| `w_init_scale`                  | 1.0          | Scale factor for randn_scaled init           |
| `lr_scheduler`                  | none         | LR schedule: "none", "cosine_warm_restarts", "linear_warmup_cosine" |

## Training Time Constraint

This is a medium-to-large model (609 neurons). Training may take longer due to larger graph.
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
- **DO NOT change `simulation:` parameters** except: `n_frames` (data volume — Block 6), and seed (managed automatically)

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

**Note**: tau_R2 and V_rest_R2 are not applicable — zebrafish ODE has fixed τ=1 and no resting potential.

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
| 4     | Regularization       | coeff_g_phi_diff, coeff_W_L1, coeff_W_L2, coeff_W_sign, coeff_f_theta_weight_L2 |
| 5     | Multi-epoch + anneal | n_epochs=2, test regul_annealing_rate (0, 0.5, 1.0, 2.0). Halve data_augmentation_loop to keep time constant. |
| 6     | Data volume          | n_frames (21000, 42000), data_augmentation_loop (250, 500, 1000)        |
| 7     | Combined best        | Best parameters from blocks 1–6                                         |
| 8     | MLP size             | hidden_dim (32, 64, 80, 128), n_layers (2, 3, 4), hidden_dim_update     |
| 9     | Free exploration     | Any parameter — test novel hypotheses, combinations, LR schedulers, etc. |

## Block Boundaries

1. Update "Paper Summary"
2. Summarize findings
3. Update "Established Principles"
4. Clear "Current Block"
5. Carry forward best config

## Known Results (prior experiments)

- No prior experiments — this is the first exploration of the zebrafish model.
- CX model (similar architecture, nonlinear) achieved connectivity_R2 ≈ 0.57 with default config.
- Flyvis model (much larger, nonlinear) achieved connectivity_R2 ≈ 0.93.
- The linear ODE may be easier or harder to recover — the lack of nonlinearity means W must be precisely recovered from linear dynamics alone.

## Start Call

When prompt says `PARALLEL START`:

- Read base config
- Set all 4 configs **identically** to baseline
- Data will be generated with different seeds per slot automatically
- First iteration establishes baseline — do not change hyperparameters yet
- State the baseline hypothesis: "The default config achieves connectivity_R2 > 0.5 robustly across seeds"

---

# Working Memory Structure

The memory file (`{llm_task_name}_memory.md`) must follow this structure:

```markdown
# Working Memory: zebrafish_oculomotor

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
