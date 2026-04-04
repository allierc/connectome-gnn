# FlyVis GNN + SIREN Joint Training Exploration — flyvis_noise_005_INR

## Scientific Context

The research question explores whether jointly learning neural connectivity and visual stimulus representation can improve circuit recovery. Standard GNN training takes ground-truth visual stimulus as input; here, we replace that with a learned SIREN model that reconstructs the stimulus field from spatiotemporal coordinates. This forces the GNN to learn connectivity that works with imperfect stimulus reconstruction, potentially improving robustness and revealing degeneracies. The dual objectives (connectivity R2 > 0.9 and field R2 > 0.7) test whether both components can converge simultaneously.

## Noise Model

The training data includes realistic measurement noise on the visual stimulus:

```
v_i(t+1) = v_i(t) + dt * f(v_i(t), W, a_i, I_observed(t)) + epsilon_i(t)
I_observed(x,y,t) = I_true(x,y,t) + delta(x,y,t)
epsilon_i ~ N(0, 0.05)  [dynamics noise on neurons]
delta ~ N(0, varies by spatial position)  [implicit in stimulus data]
```

The SIREN learns I_true(x,y,t) from observed (noisy) samples, while the GNN learns to invert the mapping from this reconstructed stimulus to neural dynamics.

## Goal

Jointly optimize a **GNN** and a **SIREN implicit neural representation (INR)** for the **Drosophila visual system** with noise level 0.05 (DAVIS input).

The GNN learns the neural connectivity (W, g_phi, f_theta) while the SIREN reconstructs the visual stimulus field from (t, x, y) coordinates. Both are trained end-to-end: the SIREN output replaces the ground-truth stimulus as input to the GNN, so the GNN loss backpropagates through the SIREN.

**Dual objectives:**

1. **Connectivity recovery**: connectivity_R2 > 0.9 on all 4 seeds, CV < 5%
2. **Visual field reconstruction**: field_R2 > 0.7 (R² between SIREN-reconstructed stimulus and ground truth)

**The main challenge is balancing GNN and SIREN learning rates across epochs.** The GNN converges quickly (peaks at end of epoch 0) while the SIREN needs more epochs to converge. If GNN LRs stay high in later epochs, the GNN overfits and connectivity degrades.

## Training Scheme — Alternate Training (Two-Phase LR)

The key mechanism is **`alternate_training`**: train GNN + SIREN jointly in epoch 0 with full LRs, then **reduce GNN LRs by `alternate_lr_ratio`** in epochs 1+ while keeping SIREN LR unchanged. This prevents the GNN from degrading while the SIREN continues to improve.

**Implementation** (already in `graph_trainer.py`): At the start of each epoch >= 1, if `alternate_training: true`, ALL GNN parameter group LRs (W, g_phi, f_theta, embedding) are multiplied by `alternate_lr_ratio`. The SIREN LR (`learning_rate_NNR_f`) is **not affected**.

**3 epochs total** with `alternate_lr_ratio: 0.05` (20x GNN LR reduction):

| Epoch | GNN LRs | SIREN LR | Purpose |
|-------|---------|----------|---------|
| 0 | Full: lr_W=5e-4, lr=1.2e-3, lr_emb=1.5e-3 | 1e-8 | Joint warmup — GNN learns connectivity, SIREN starts learning field |
| 1-2 | Reduced: lr_W=2.5e-5, lr=6e-5, lr_emb=7.5e-5 | 1e-8 | SIREN refinement — GNN stabilizes, SIREN catches up with improving field |

**Explorable alternate training parameters:**

| Parameter | Default | Description | Suggested range |
|-----------|---------|-------------|-----------------|
| `alternate_lr_ratio` | 0.05 | GNN LR multiplier for epochs 1+ | 0.01 – 0.2 |
| `n_epochs` | 3 | Total epochs | 3 (fixed baseline, explore 2-5) |
| `learning_rate_NNR_f` | 1e-8 | SIREN learning rate (unchanged across phases) | 1e-9 – 1e-6 |

**Key insight from flyvis_63_2_noise_005 experiments**: Connectivity peaks at end of epoch 0 (~0.957) and then _degrades_ with continued full-LR training (0.914 by mid epoch 1). The alternate training with ratio=0.05 preserves connectivity while letting the SIREN improve its field reconstruction over epochs 1-2.

## SIREN Architecture (flyvis_63_2 base)

The base config uses a **large SIREN** (4096 hidden, omega=4096) from the flyvis_63_1 archive:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim_nnr_f` | 4096 | Hidden dimension — ~33M params |
| `n_layers_nnr_f` | 3 | Hidden SineLayer blocks |
| `omega_f` | 4096 | Frequency parameter for sine activation |
| `learning_rate_NNR_f` | 1e-8 | SIREN learning rate |
| `batch_size` | 1 | One sample per iteration (required for large SIREN) |

**Training time**: With batch_size=1 and 4096-hidden SIREN, each epoch takes ~100-120 min. Total 3 epochs = ~360 min. This is at the budget limit.

**Exploration axes for SIREN:**
- `hidden_dim_nnr_f`: 4096 is large. Test 2048, 1024 for faster training if time exceeds budget.
- `omega_f`: Controls frequency resolution. Test lower values (2048, 1024) if field_R2 is already good.
- `learning_rate_NNR_f`: The SIREN LR is the most critical parameter for field quality. May need to increase in later epochs.

## Scientific Method

This exploration follows a strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Based on available data (metrics, seed variance, prior results), form a hypothesis about what controls joint GNN+SIREN training quality
2. **Design experiment**: Choose a mutation that specifically tests the hypothesis — change ONE parameter at a time
3. **Run training**: The experiment runs across 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use both GNN metrics (connectivity_R2, CV) AND SIREN metrics (field_R2) to evaluate
5. **Update understanding**: Revise hypotheses based on evidence

**CRITICAL**: You can only hypothesize. Only training results can validate or falsify your hypotheses. Never assume a hypothesis is correct without experimental evidence.

**Evidence hierarchy:**

| Level | Criterion | Action |
|-------|-----------|--------|
| **Established** | Consistent across 3+ iterations AND 4/4 seeds | Add to Principles |
| **Tentative** | Observed 1-2 times or inconsistent across seeds | Add to Open Questions |
| **Contradicted** | Conflicting evidence across iterations/seeds | Note in Open Questions |

## CRITICAL: Data is RE-GENERATED per slot

Each slot re-generates its data with a **different random seed**.
Both `simulation.seed` and `training.seed` are **forced by the pipeline** — DO NOT modify them in config files.

Seed formula (set automatically by GNN_LLM.py):

- `simulation.seed = iteration * 1000 + slot` (controls data generation)
- `training.seed = iteration * 1000 + slot + 500` (controls weight init & training randomness)

The actual seed values are provided in the prompt for each slot — **log them in your iteration entries**.

Simulation parameters (n_neurons, n_frames, etc.) stay fixed — **DO NOT change them**.

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i(t)
dv_i/dt = f_theta(v_i, a_i, sum_j W_ij * g_phi(v_j, a_j)^2, I_i)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors)
- DAVIS visual input, **noise_model_level=0.05**
- 64,000 frames, delta_t=0.02

## GNN Architecture

Two MLPs learn the neural dynamics:

- **g_phi** (MLP1): Edge message function. Maps (v_j, a_j) → message. `g_phi_positive=true` squares output to enforce positivity.
- **f_theta** (MLP0): Node update function. Maps (v_i, a_i, aggregated_messages, I_i) → dv_i/dt.
- **Embedding a_i**: learnable low-dimensional embedding per neuron type.

Architecture parameters (explorable):

- `hidden_dim` / `n_layers`: g_phi MLP width/depth (default: 64 / 3)
- `hidden_dim_update` / `n_layers_update`: f_theta MLP width/depth (default: 64 / 3)
- `embedding_dim`: embedding dimension (default: 2)

**CRITICAL — coupled parameters**: When changing `embedding_dim`, you MUST also update:

- `input_size = 1 + embedding_dim` (v_j + a_j for g_phi)
- `input_size_update = 3 + embedding_dim` (v_i + a_i + msg + I_i for f_theta)

### Optimizer Structure

The optimizer has **separate parameter groups** with independent learning rates:

| Group | Parameters | LR config key | Affected by alternate_training? |
|-------|-----------|---------------|-------------------------------|
| W | Connectivity matrix | `learning_rate_W_start` | **Yes** (x0.05 at epoch 1+) |
| g_phi | Edge message MLP | `learning_rate_start` | **Yes** (x0.05 at epoch 1+) |
| f_theta | Node update MLP | `learning_rate_start` (same as g_phi) | **Yes** (x0.05 at epoch 1+) |
| embedding | Neuron type embeddings | `learning_rate_embedding_start` | **Yes** (x0.05 at epoch 1+) |
| NNR_f | SIREN network | `learning_rate_NNR_f` | **No** (unchanged) |

This means in epochs 1+, only the SIREN continues at full learning rate while all GNN components are nearly frozen.

## Regularization Parameters

| Config parameter | Role | Default | Annealed? |
|-----------------|------|---------|-----------|
| `coeff_g_phi_diff` | Monotonicity penalty on g_phi | 750 | No |
| `coeff_g_phi_norm` | Normalization penalty at saturation | 1.0 | No |
| `coeff_g_phi_weight_L1` | L1 on g_phi MLP weights | 0.5 | **Yes** |
| `coeff_g_phi_weight_L2` | L2 on g_phi MLP weights | 0 | **Yes** |
| `coeff_f_theta_weight_L1` | L1 on f_theta MLP weights | 0.5 | **Yes** |
| `coeff_f_theta_weight_L2` | L2 on f_theta MLP weights | 0.001 | **Yes** |
| `coeff_W_L1` | L1 sparsity penalty on W | 5e-5 | **Yes** |

### Regularization Annealing

**Formula**: `effective_coeff = coeff * (1 - exp(-rate * epoch))`

With **3 epochs** and `regul_annealing_rate=0.5`:

| Epoch | Multiplier | Meaning |
|-------|-----------|---------|
| 0 | 0.00 | No regularization — free learning |
| 1 | 0.39 | ~39% strength |
| 2 | 0.63 | ~63% strength |

This is beneficial for alternate training: epoch 0 lets GNN+SIREN learn freely with no regularization, then regularization activates in epochs 1-2 when GNN LRs are already reduced.

## Training Parameters (explorable)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate_W_start` | 5e-4 | LR for connectivity W |
| `learning_rate_start` | 1.2e-3 | LR for g_phi and f_theta |
| `learning_rate_embedding_start` | 1.5e-3 | LR for embeddings |
| `learning_rate_NNR_f` | 1e-8 | LR for SIREN (key exploration axis) |
| `alternate_training` | true | Enable two-phase training |
| `alternate_lr_ratio` | 0.05 | GNN LR multiplier for epochs 1+ |
| `n_epochs` | 3 | Epochs |
| `batch_size` | 1 | Batch size (constrained by SIREN memory) |
| `data_augmentation_loop` | 25 | Data augmentation multiplier |
| `w_init_mode` | zeros | W initialization |

## Training Time Constraint

With 3 epochs, batch_size=1, and 4096-hidden SIREN: **target <= 360 min/iteration**.
Each epoch processes 64K x aug_loop frames. The large SIREN adds significant overhead per iteration.
Monitor `training_time_min` and adjust if over budget.

## Parallel Mode — 4 Slots Per Batch

You receive **4 results per batch** and propose **4 mutations** for the next batch.
Each slot runs with a **different random seed**, so you can assess seed robustness within a single batch.

### Robustness Assessment

After each batch, evaluate using **both** GNN and SIREN metrics:

- **Excellent**: all 4 slots connectivity_R2 > 0.9 AND field_R2 > 0.7 — **TARGET**
- **Good GNN**: all 4 slots connectivity_R2 > 0.9 but field_R2 < 0.7 — SIREN needs improvement
- **Good SIREN**: field_R2 > 0.7 but some connectivity_R2 < 0.9 — GNN needs improvement
- **Partial**: mixed results — analyze which component is failing
- **Failed**: connectivity_R2 < 0.8 or field_R2 < 0.3 — reject

**Stability**: compute CV of connectivity_R2 across 4 seeds. Target CV < 5%.

### Slot Strategy

All 4 slots should run the **same config** (different seeds are applied automatically).

### Config Files

- Edit all 4 config files: `{name}_00.yaml` through `{name}_03.yaml`
- **All 4 configs should be identical** (only seeds differ, set automatically)
- Only modify `training:` and `graph_model:` parameters
- **DO NOT change `simulation:` parameters**

## Iteration Loop Structure

Each block = `n_iter_block` iterations (default 12).

## Iteration Workflow (every batch)

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

**Metrics from `analysis.log`:**

- `connectivity_R2`: R² of learned vs true W (**PRIMARY GNN METRIC**)
- `field_R2`: R² of SIREN reconstruction vs true stimulus (**PRIMARY SIREN METRIC**)
- `field_slope`: linear fit slope of SIREN vs true (target ~1.0)
- `tau_R2`: R² of learned vs true time constants
- `V_rest_R2`: R² of learned vs true resting potentials
- `training_time_min`: training duration

**Dual assessment:**

- GNN quality: connectivity_R2 across seeds (mean, std, CV, min, max)
- SIREN quality: field_R2 across seeds (mean, std, min)
- Joint quality: both must be good simultaneously

### Step 3: Write Log Entries + Update Memory

```
## Iter N: [excellent/good_gnn/good_siren/partial/failed]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, lr_NNR_f=W, alternate_lr_ratio=R, coeff_g_phi_diff=A, batch_size=B, n_epochs=C
Slot 0: conn_R2=A, field_R2=B, tau_R2=C, V_rest_R2=D, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, field_R2=B, tau_R2=C, V_rest_R2=D, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, field_R2=B, tau_R2=C, V_rest_R2=D, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, field_R2=B, tau_R2=C, V_rest_R2=D, sim_seed=S, train_seed=T
GNN stats: mean_conn_R2=X, std=Y, CV=Z%, min=W
SIREN stats: mean_field_R2=X, std=Y, min=W
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive] — [one line]
Next: parent=P

## Winner Config (COMPULSORY)

**At every block boundary**, you MUST save the current best config as a winner file.
This is a COMPULSORY task — do not skip it.

1. Identify the **best iteration** (highest connectivity_R2, or primary metric)
2. Copy its saved config from `log/Claude_exploration/LLM_<task_name>/config/iter_XXX_slot_YY.yaml`
3. Save it to `config/fly/flyvis_noise_005_INR_winner.yaml` with a YAML comment header:

```yaml
# Winner config: flyvis_noise_005_INR_winner.yaml
# Source: iter_XXX_slot_YY (connectivity_R2 = X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [1-2 sentence narrative: what made this config the best]
#   - [key hyperparameter choices and why they matter]
#
# Metrics:
#   connectivity_R2: X.XXX (best single seed)
#   robust_mean:     X.XXX +/- X.XXX (N seeds, CV=X.X%)
#   rollout_pearson: X.XXX
#   cluster_accuracy: X.XXX
#   spectral_radius: X.XXX (true: X.XXX)
#
# Key config differences from baseline:
#   - [list the parameters that differ from the initial baseline]
```

Destination: `config/fly/flyvis_noise_005_INR_winner.yaml`

```

### Step 4: Acknowledge User Input (if any)

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Block Partition (suggested)

| Block | Focus | Parameters |
|-------|-------|-----------|
| 1 | Baseline | flyvis_63_2 config — establish baseline with alternate training |
| 2 | alternate_lr_ratio | Sweep: {0.01, 0.05, 0.1, 0.2} — how much GNN LR reduction is optimal? |
| 3 | SIREN LR | learning_rate_NNR_f: {1e-9, 1e-8, 1e-7} — can SIREN learn faster? |
| 4 | GNN epoch-0 LRs | Test noise_005 champion LRs (1.5x higher) for epoch 0 |
| 5 | SIREN architecture | Test smaller hidden_dim (2048, 1024) for faster training |
| 6 | Regularization | coeff_g_phi_diff, w_init_mode, aug tuning |
| 7 | Combined best | Integrate findings from blocks 1-6 |

## Known Results (prior experiments)

**From flyvis_63_2_noise_005 (the base config)**:
- Alternate training with ratio=0.05 preserves connectivity while SIREN improves
- Connectivity peaks at end of epoch 0 (~0.957), preserved through epochs 1-2
- Without alternate training (flyvis_63_1), connectivity degrades from 0.957 to 0.914 by mid epoch 1

**From noise_005 GNN-only champion (220 iterations)**:
- connectivity_R2=0.982±0.003 with aug=35, 1.5x LRs (lr_W=9e-4, lr=1.8e-3, lr_emb=2.325e-3), hidden=80, 1 epoch
- The GNN-only exploration reached a global optimum across 45 tested dimensions
- Joint training connectivity will likely be lower due to SIREN interaction, but alternate training should preserve epoch-0 gains

**From standalone SIREN exploration (108 iterations)**:
- Best: R²=0.824 with dim=512, 7L, omega=1024, lr=3e-7, batch=150, 60k steps
- Joint training SIREN LR is much lower (1e-8) because gradients come through GNN loss

## Sibling Explorations — Reference Memory Files

Two sibling explorations have accumulated knowledge available to you. **Read these files at each block boundary** to leverage their findings:

- **GNN-only exploration** (noise_005): `./log/Claude_exploration/LLM_flyvis_noise_005/flyvis_noise_005_Claude_memory.md`
- **SIREN-only exploration** (noise_005_INR_siren): `./log/Claude_exploration/LLM_flyvis_noise_005_INR_siren/flyvis_noise_005_INR_siren_Claude_memory.md`

Use their established principles and best configs as starting points:
- From noise_005: best GNN hyperparameters (LRs, regularization, architecture)
- From noise_005_INR_siren: best SIREN hyperparameters (hidden_dim, n_layers, omega, LR, batch_size)

## File Structure

You maintain **THREE** files:

### 1. Full Log (append-only)

**File**: `{llm_task_name}_analysis.md`

### 2. Working Memory (read + update every batch)

**File**: `{llm_task_name}_memory.md`

### 3. User Input (read every batch, acknowledge pending items)

**File**: `user_input.md`

## Knowledge Base Guidelines

### What to Add to Established Principles

A principle must satisfy ALL of:

1. Observed consistently across **3+ iterations**
2. Consistent across **all 4 seeds** (not just mean, but low variance)
3. States a **causal relationship** (not just a correlation)

### What to Add to Open Questions

- Patterns observed 1-2 times
- Seed-dependent effects (works for some seeds but not others)
- Contradictions between iterations
- Theoretical predictions not yet verified

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:

1. State the original hypothesis
2. State the contradicting evidence (iteration number, metrics)
3. State what was learned from the falsification
4. Propose a revised hypothesis if applicable

## Start Call

When prompt says `PARALLEL START`:

- Read base config to understand training regime (alternate training with ratio=0.05)
- Set all 4 configs **identically** to the baseline config
- Write planned config and **initial hypothesis** to working memory
- First iteration establishes baseline — do not change hyperparameters yet
- Baseline hypothesis: "The flyvis_63_2 alternate training config (3 epochs, lr_ratio=0.05) preserves connectivity_R2 > 0.9 from epoch 0 while achieving field_R2 > 0.3 by epoch 2"

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_INR

## Paper Summary (update at every block boundary)

- **Joint GNN+SIREN optimization**: [...]
- **LLM-driven exploration**: [...]
- **Future works**: [...]

## Knowledge Base

### Results Table

| Iter | Config summary | conn_R2 (mean±std) | CV% | min | field_R2 (mean) | field_min | time_min | Rating | Hypothesis |
| ---- | -------------- | ------------------ | --- | --- | --------------- | --------- | -------- | ------ | ---------- |
| 1 | baseline (63_2) | ? | ? | ? | ? | ? | ? | ? | baseline |

### Established Principles

[Confirmed patterns — require 3+ supporting iterations AND cross-seed consistency]

### Falsified Hypotheses

[Keep as record]

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
