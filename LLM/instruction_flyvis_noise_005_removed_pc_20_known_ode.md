# FlyVis known_ode Training Exploration — flyvis_noise_005_removed_pc_20

## Goal

Test **known_ode robustness to incomplete connectivity** for the **Drosophila visual system** with dynamics noise 0.005 and 20% missing edges (removed uniformly). The goal is to find an **incomplete-connectivity-robust config** that achieves **connectivity_R2 > 0.85 on ALL 4 seeds with CV < 4%**, demonstrating resilience to sparse/incomplete structural information.

**Primary objective**: Establish a **robust config** that achieves **connectivity_R2 > 0.85 across all 4 seeds with CV < 4%**, testing known_ode when the true connectivity is partially unknown due to incomplete connectome mapping or experimental dropout.

**Why this matters**: known_ode is an inverse problem — we have the ODE dynamics `f(v, W)` known, and we learn connectivity W and biophysical parameters (tau_i, V_rest_i) directly from noisy neural observations. When 20% of the true edges are missing from the ground-truth connectivity matrix, the model must learn to infer or compensate for missing connections while recovering the present ones. This tests the limits of parameter recovery under structural incompleteness.

Primary metric: **connectivity_R2** (R² between learned W and ground-truth W, computed only on present edges).

Stability metric: **CV (coefficient of variation)** of connectivity_R2 across 4 seeds — target CV < 4%.

Secondary metrics: **tau_R2**, **V_rest_R2**, **cluster_accuracy** (neuron type clustering).

## Scientific Method

This exploration follows a strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Based on available data (metrics, seed variance, prior results), form a specific, testable hypothesis about which parameter controls robustness
2. **Design experiment**: Choose a mutation that specifically tests the hypothesis — change **exactly ONE parameter at a time**
3. **Run training**: The experiment runs across 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use both metrics AND cross-seed variance to evaluate whether the hypothesis was supported or contradicted
5. **Update understanding**: Revise hypotheses based on evidence. A falsified hypothesis is valuable information.

**CRITICAL**: You can only hypothesize. Only training results can validate or falsify your hypotheses. Never assume a hypothesis is correct without experimental evidence.

**Evidence hierarchy:**

| Level | Criterion | Action |
| --- | --- | --- |
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

## Scientific Context

The **known_ode model** assumes the ODE is known:
```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * g_phi(v_j)^2 + I_i
```

Given noisy voltage observations with incomplete structural information, the inverse problem is to recover:
1. **Connectivity matrix W_ij** (synaptic strengths, 434K+ parameters, with 20% of true edges absent)
2. **Time constants tau_i** (13.7K parameters)
3. **Resting potentials V_rest_i** (13.7K parameters)

With 20% missing edges, the model must infer the missing synaptic weights or learn to work without them. This tests whether regularization and learning rates designed for complete connectivity transfer to incomplete connectivity scenarios.

## Noise Model

Two independent noise sources in the training data:

1. **Dynamics noise** (`noise_model_level=0.005`): `v(t+1) = v(t) + dt * f(v, W, I) + epsilon_dyn(t)`, epsilon_dyn ~ N(0, 0.005)
2. **Measurement noise** (`measurement_noise_level=0.0`): Clean observations

## Data

**Pre-generated, fixed across all iterations**:
- Dataset: `fly/flyvis_noise_005_removed_pc_20` (DAVIS visual input, 64,000 frames, 20% edges removed)
- Noise model: `noise_model_level=0.005, measurement_noise_level=0.0`
- Edge removal: **Uniform 20% removal** — connectivity matrix has 80% of true edges
- Re-generation: **YES** — each iteration generates new data with different `simulation.seed` to test robustness

**DO NOT change**: `simulation.n_neurons`, `simulation.n_edges`, `simulation.n_frames`, `simulation.delta_t`, dataset name, or visual input type.

Seeds are managed by pipeline:
- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

## FlyVis Neuronal Dynamics Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i(t)
```

Where:
- `tau_i`: membrane time constant (learned)
- `V_i^rest`: resting potential (learned)
- `W_ij`: synaptic weight (connectivity, learned, with 20% missing)
- `g_phi`: edge activation function (fixed, typically ReLU)
- `a_j`: learnable neuron type embedding

**Model specs**:
- 13,741 neurons, 65 cell types, 347,290 edges (434,112 × 0.8, with 20% removed)
- 1,736 input neurons (photoreceptors)
- DAVIS visual input stimulus
- 64,000 frames, delta_t=0.02 (time resolution)

## known_ode Learning Task

The known_ode model **directly learns parameters** from voltage dynamics because the ODE is known:

**Learned parameters**:
- `W_ij`: connectivity (synaptic weights, 20% missing) — PRIMARY TARGET
- `tau_i`: time constants
- `V_rest_i`: resting potentials
- Neuron type embeddings (if used)

**Not learned** (frozen):
- ODE structure (f and g_phi are given)
- Network architecture (graph structure)
- Visual input mapping
- Measurement model

**CRITICAL**: Do NOT modify `coeff_g_phi_diff, coeff_f_theta_msg_diff, coeff_g_phi_norm, coeff_g_phi_weight_L1, coeff_g_phi_weight_L2, coeff_f_theta_weight_L1, coeff_f_theta_weight_L2, embedding_dim, lr_embedding` — these are not used by known_ode.

## Explorable Parameters

### Learning Rates

| Parameter | Default | Range | Description |
| --- | --- | --- | --- |
| `lr_W` | 0.0009 | [1e-5, 1e-3] | Learning rate for connectivity matrix W |
| `lr` | 0.0018 | [1e-4, 1e-2] | Learning rate for tau_i, V_rest_i, embeddings |

### Weight Initialization

| Parameter | Default | Options | Description |
| --- | --- | --- | --- |
| `w_init_mode` | `randn` | `randn`, `randn_scaled`, `zeros` | Initialization distribution for W |
| `w_init_scale` | 1.0 | [0.5, 1.0, 2.0] | Scaling factor for `randn_scaled` mode |

### Batch Size & Regularization

| Parameter | Default | Description |
| --- | --- | --- |
| `batch_size` | 4 | Number of time windows per gradient step (INTEGER) |
| `coeff_W_L1` | 0 | L1 sparsity penalty on W (0 = no sparsity) |
| `coeff_W_L2` | 0.00015 | L2 penalty on W (weight decay) |
| `coeff_W_sign` | 1.5e-06 | Dale's law penalty (enforce sign consistency) |

**Trade-off**: Incomplete connectivity may require adjusted regularization; too much regularization may bias estimates toward zero for missing edges.

**CRITICAL CONSTRAINTS**:
- `batch_size` MUST be INTEGER (1, 2, 4, 8, etc.)
- `n_epochs` MUST be INTEGER (1, 2, 3, etc.) — NOT 0.5
- `w_init_mode` MUST be LOWERCASE: `randn`, `randn_scaled`, `zeros`

## Training Time Budget — FIXED for Fair Comparison

**LOCKED PARAMETERS** (DO NOT MODIFY):
- `n_epochs: 1` — Single epoch training only
- `data_augmentation_loop (DAL): 35` — Fixed data augmentation

**Why these are locked**:

To fairly compare different LLM explorations across noise levels and biomodels, all known_ode variants must use the same training budget. This ensures observed differences in connectivity_R2 reflect parameter choices, not training time variation.

**Target training time**: ~12 minutes per iteration (consistent, cluster-efficient).

**If you believe n_epochs or DAL should be varied** to test a specific hypothesis, first post in `user_input.md` for authorization. Do NOT change these without explicit user approval.

## Parallel Mode — 4 Slots Per Batch

Each batch runs **4 slots simultaneously**, each with a different config (forced seeds differ automatically):

### Exploration Mode (default)
- Slot 0: Baseline (no changes)
- Slots 1-3: Each changes **exactly ONE parameter** from the block focus

This gives **3 independent causal tests** per batch while maintaining slot-0 baseline for reference.

### Robustness Mode (when validating a promising config)
- All 4 slots: Same config, different seeds
- Measures stability across seed variation

**Robustness criteria** (incomplete connectivity data):
- **Robust**: all 4 slots connectivity_R2 > 0.85
- **Partially robust**: 2-3 slots > 0.80
- **Fragile**: ≤1 slots > 0.75

## Block Structure — 10 Blocks × 8 Iterations Each

With `n_iter_block=8` and `iterations=80`, the exploration spans 10 hypothesis-driven blocks:

### Block 1 (iter 1-8): Learning Rate Sweep
**Hypothesis**: "Incomplete connectivity requires careful learning rates. Optimal LR will achieve connectivity_R2 > 0.80 while maintaining stability across seeds"

**Test**: Sweep `lr_W` and `lr` systematically
- Slot 0: lr_W=0.0009, lr=0.0018 (baseline)
- Slot 1: lr_W=0.0005, lr=0.0009 (conservative)
- Slot 2: lr_W=0.002, lr=0.004 (aggressive)
- Slot 3: lr_W=0.0001, lr=0.0002 (very conservative)

**Expected outcome**: Identify LR range for incomplete connectivity.

### Block 2 (iter 9-16): W Regularization
**Hypothesis**: "Missing edges may require different regularization balance. Test whether reduced L2 helps infer missing connectivity."

**Test**: Sweep `coeff_W_L1, coeff_W_L2, coeff_W_sign`
- Slot 0: coeff_W_L1=0, coeff_W_L2=0.00015, coeff_W_sign=1.5e-6 (baseline)
- Slot 1: coeff_W_L1=1e-4, coeff_W_L2=5e-5, coeff_W_sign=1e-6 (reduced)
- Slot 2: coeff_W_L1=0, coeff_W_L2=0, coeff_W_sign=0 (minimal)
- Slot 3: coeff_W_L1=5e-4, coeff_W_L2=3e-4, coeff_W_sign=5e-6 (increased)

**Expected outcome**: Validate regularization trade-off with missing edges.

### Block 3 (iter 17-24): W Initialization + LR
**Hypothesis**: "Sparse initialization may be better for incomplete connectivity. Test whether zero-initialization helps."

**Test**: Sweep `w_init_mode, w_init_scale` with optimized `lr_W, lr` from Block 1
- Slot 0: w_init_mode=randn, w_init_scale=1.0, optimized LRs
- Slot 1: w_init_mode=randn_scaled, w_init_scale=0.5, optimized LRs
- Slot 2: w_init_mode=randn_scaled, w_init_scale=2.0, optimized LRs
- Slot 3: w_init_mode=zeros, optimized LRs

**Expected outcome**: Determine whether initialization strategy changes with incomplete connectivity.

### Block 4 (iter 25-32): Batch Size + LR
**Hypothesis**: "Incomplete connectivity may benefit from larger batch sizes for stable gradient estimates."

**Test**: Sweep `batch_size` with LRs from Block 1-3
- Slot 0: batch_size=4, optimized LRs
- Slot 1: batch_size=1, conservative LRs
- Slot 2: batch_size=8, optimized LRs
- Slot 3: batch_size=16, conservative LRs

**Expected outcome**: Quantify batch size effect with missing edges.

### Block 5 (iter 33-40): Free Exploration
**Hypothesis**: Form based on Blocks 1-4 results. Explore parameter combinations not yet tested.

Test combinations of best settings from previous blocks.

### Block 6 (iter 41-48): Refinement
**Hypothesis**: Polish the best config to maximize connectivity_R2.

Fine-tune learning rates and regularization around the best config found.

### Block 7 (iter 49-56): Robustness Validation I
**Strategy**: Switch to **robustness mode** (all 4 slots same config, different seeds).

**Test**: Run the best config from Blocks 1-6 on 4 independent seeds to validate connectivity_R2 > 0.85 with CV < 4%.

### Block 8 (iter 57-64): Parameter Interaction
**Hypothesis**: Test interactions between parameters that individually showed promise. Combine best findings from Blocks 1-7.

Test combinations of best parameters from previous blocks.

### Block 9 (iter 65-72): Fine-Tuning
**Hypothesis**: Polish and refine the best config to maximize connectivity_R2 and minimize CV.

Fine-tune learning rates and regularization around the current best.

### Block 10 (iter 73-80): Final Robustness Validation
**Strategy**: Final robustness check on best config found.

**Final test**: Run the best config from Blocks 1-9 on 4 independent seeds to validate connectivity_R2 > 0.85 with CV < 4%.

## File Structure

You maintain THREE files:

1. **Full Log (append-only)**: `flyvis_noise_005_removed_pc_20_known_ode_Claude_analysis.md`
   - Append every iteration's log entry (4 entries per batch)
   - Never read — human record only

2. **Working Memory (read + update every batch)**: `flyvis_noise_005_removed_pc_20_known_ode_Claude_memory.md`
   - Read at start, update at end
   - Contains: robustness comparison table, hypotheses, established principles, current block iterations

3. **User Input (read every batch, acknowledge pending items)**: `user_input.md`
   - Read at every batch
   - If "Pending Instructions" section has content: act on it, then move entries to "Acknowledged" section

## Knowledge Base Guidelines

### What to Add to Established Principles

A principle must satisfy ALL of:
- Observed consistently across 3+ iterations
- Consistent across all 4 seeds (not just mean, but low variance)
- States a causal relationship (not just a correlation)

Example: "w_init_mode=randn_scaled with scale=0.5 improves convergence with incomplete connectivity (3/3 iterations, CV < 3%)"

### What to Add to Open Questions

- Patterns observed 1-2 times
- Seed-dependent effects (works for some seeds but not others)
- Contradictions between iterations
- Theoretical predictions not yet verified

Example: "Does zero-initialization help infer missing edges? Only iter 1 tested, needs more validation."

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:
- State the original hypothesis
- State the contradicting evidence (iteration number, metrics)
- State what was learned from the falsification
- Propose a revised hypothesis if applicable

Example: "Hypothesis: 'Missing edges benefit from minimal regularization' — Falsified by iter 6 (coeff_W_L2=0 → mean R2=0.75, CV=6%). Revised: 'Incomplete connectivity still requires regularization to prevent overfitting to present edges.'"

## Iteration Workflow

### Step 1: Read Working Memory + User Input

Review the "Emerging Observations" section and "Results Comparison Table" to understand progress.

### Step 2: Analyze Current Batch Results (4 slots)

For each slot, extract from `analysis.log`:
- `connectivity_R2` (primary metric)
- `tau_R2, V_rest_R2` (secondary)
- `rollout_pearson_r` (dynamics quality)
- `training_time_min`

**Robustness classification** (across all 4 seeds):
- **Stable-Robust**: all 4 slots connectivity_R2 > 0.85 AND CV < 3% — TARGET
- **Robust**: all 4 slots connectivity_R2 > 0.85, CV 3-5%
- **Partially robust**: 2-3 slots connectivity_R2 > 0.85
- **Fragile**: 0-1 slots connectivity_R2 > 0.85
- **DISQUALIFIED**: any slot < 0.80 — reject config immediately

### Step 3: Write Log Entry + Update Memory

Append to full log and current block in memory:

```
## Iter N: [robust/partially robust/fragile]
Hypothesis tested: "[quoted hypothesis]"
Slot 0: config=[params] → connectivity_R2=X.XXX, tau_R2=Y.YYY, V_rest_R2=Z.ZZZ, time=T min
Slot 1: config=[params] → connectivity_R2=X.XXX, tau_R2=Y.YYY, V_rest_R2=Z.ZZZ, time=T min
Slot 2: config=[params] → connectivity_R2=X.XXX, tau_R2=Y.YYY, V_rest_R2=Z.ZZZ, time=T min
Slot 3: config=[params] → connectivity_R2=X.XXX, tau_R2=Y.YYY, V_rest_R2=Z.ZZZ, time=T min
Seed stats: mean_conn_R2=X, std=Y, CV=Z%, min=W, max=V
Stability: [Stable-Robust / Robust / Partially robust / Fragile / DISQUALIFIED]
Verdict: [supported/falsified/inconclusive]
Next: [what to test next]
```

### Step 4: Update Working Memory

Add row to "Results Comparison Table", update "Emerging Observations" section.

### Step 5: Design Next 4 Configs

For next batch, design 4 configs based on current results.

## Block Boundaries — Winner Config (COMPULSORY)

**At every block end** (iterations 8, 16, 24, 32, 40, 48, 56, 64, 72, 80), you MUST save the best config as a winner file:

1. Identify best iteration (highest connectivity_R2)
2. Copy config from `log/Claude_exploration/LLM_flyvis_noise_005_removed_pc_20_known_ode/config/iter_XXX_slot_YY.yaml`
3. Save to `config/fly/flyvis_noise_005_removed_pc_20_known_ode_winner.yaml` with header:

```yaml
# Winner config: flyvis_noise_005_removed_pc_20_known_ode_winner.yaml
# Source: iter_XXX_slot_YY (connectivity_R2 = X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [1-2 sentence narrative]
#   - [key hyperparameter choices]
#
# Metrics:
#   connectivity_R2: X.XXX
#   tau_R2: X.XXX
#   V_rest_R2: X.XXX
#   rollout_pearson: X.XXX
#
# Key config differences from baseline:
#   - [list parameter changes]
```

## Start Call

When prompt says `PARALLEL START`:

- Read base config — note 20% edges removed (incomplete connectivity)
- Initialize 4 slot configs for Block 1 (learning rate sweep)
- **Initial hypothesis**: "Incomplete connectivity requires careful learning rates. Optimal LR will achieve connectivity_R2 > 0.80 while maintaining CV < 4%"
- Set Slot 0 = baseline (lr_W=0.0009, lr=0.0018)
- Slots 1-3 = three different LR values to test the hypothesis

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_removed_pc_20_known_ode

## Paper Summary (update at every block boundary)

- **known_ode with incomplete connectivity**: [How known_ode handles 20% missing edges, which parameters matter most, best connectivity_R2 achieved, impact of structural incompleteness]
- **Robustness findings**: [How stable is the best config across seeds? Key insights about learning with incomplete ground truth.]
- **Optimization dynamics**: [Convergence behavior, learning rate sensitivity, regularization balance with missing edges]

## Knowledge Base

### Results Comparison Table

| Iter | Slot | lr_W | lr | w_init_mode | w_init_scale | batch_size | coeff_W_L1 | coeff_W_L2 | coeff_W_sign | conn_R2 | tau_R2 | Vrest_R2 | rollout_r | time_min |
| ---- | ---- | ---- | --- | ----------- | ------------ | ---------- | ---------- | ---------- | ------------ | ------- | ------ | -------- | --------- | -------- |

### Established Principles

[Rules that have been proven across 3+ iterations and 4/4 seeds]

### Falsified Hypotheses

[Hypotheses disproven by evidence]

### Open Questions

[Uncertainties still under investigation]

---

## Previous Block Summary

[Summary of findings from the last completed block]

---

## Current Block (Block N)

### Block Info
- Iterations: N-N+12
- Focus: [Block focus — learning rates, regularization, initialization, batch size, etc.]

### Current Hypothesis
**Hypothesis**: [specific, testable prediction]
**Rationale**: [why this matters]
**Test**: [what the 4 configs test]
**Expected outcome**: [what supports vs falsifies]
**Status**: untested / supported / falsified

### Iterations This Block
[List iterations completed in this block with summary of each]

### Emerging Observations

[Key findings emerging from this block. Update this as iterations complete.]

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
