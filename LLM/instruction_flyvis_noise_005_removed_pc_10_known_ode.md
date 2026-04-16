# FlyVis known_ode Training Exploration — flyvis_noise_005_removed_pc_10

## Goal

Test **known_ode robustness to incomplete connectivity** for the **Drosophila visual system** with dynamics noise 0.005 and 10% missing edges (removed uniformly per column). The goal is to find an **incomplete-connectivity-robust config** that achieves **connectivity_R2 > 0.87 on ALL 4 seeds with CV < 4%**, demonstrating resilience to sparse/incomplete structural information.

**Primary objective**: Establish a **robust config** that achieves **connectivity_R2 > 0.87 across all 4 seeds with CV < 4%**, testing known_ode when the true connectivity is partially unknown due to incomplete connectome mapping or experimental dropout.

**Why this matters**: known_ode is an inverse problem — we have the ODE dynamics `f(v, W)` known, and we learn connectivity W and biophysical parameters (tau_i, V_rest_i) directly from noisy neural observations. When 10% of the true edges are missing from the ground-truth connectivity matrix, the model must learn to infer or compensate for missing connections while recovering the present ones. By comparing to the 20% removal case, we can quantify how degradation scales with missing fraction.

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

## CRITICAL: Data is PRE-GENERATED at startup (fixed across iterations)

At startup, data is generated **once** for all 4 slots with **different random seeds** (one per slot). These datasets are **reused across all iterations** — data is NOT re-generated each iteration.
Both `simulation.seed` and `training.seed` are **forced by the pipeline** — DO NOT modify them in config files.

Seed formula (set automatically by GNN_LLM.py):
- `simulation.seed = 1000 + slot` (controls data generation — fixed at startup, slot 0–3)
- `training.seed = iteration * 1000 + slot + 500` (controls weight init & training randomness)

The actual seed values are provided in the prompt for each slot — **log them in your iteration entries**.

**Seed robustness testing**: To re-generate data with new seeds and test robustness, set `claude.test_robustness_seed: true` in all 4 slot configs. The pipeline will re-generate data for that batch only, then reset the flag automatically.

Simulation parameters (n_neurons, n_frames, etc.) stay fixed — **DO NOT change them**.

## Scientific Context

The **known_ode model** assumes the ODE is known:
```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * g_phi(v_j)^2 + I_i
```

Given noisy voltage observations with incomplete structural information, the inverse problem is to recover:
1. **Connectivity matrix W_ij** (synaptic strengths, 434K+ parameters, with 10% of true edges absent)
2. **Time constants tau_i** (13.7K parameters)
3. **Resting potentials V_rest_i** (13.7K parameters)

With 10% missing edges, the model must infer the missing synaptic weights or learn to work without them. The 10% removal case is expected to be easier than 20%, but the same qualitative questions apply: do hyperparameters transfer, and what regularization strategy handles the incompleteness best?

## Noise Model

Two independent noise sources in the training data:

1. **Dynamics noise** (`noise_model_level=0.005`): `v(t+1) = v(t) + dt * f(v, W, I) + epsilon_dyn(t)`, epsilon_dyn ~ N(0, 0.005)
2. **Measurement noise** (`measurement_noise_level=0.0`): Clean observations

## Data

**Pre-generated, fixed across all iterations**:
- Dataset: `fly/flyvis_noise_005_removed_pc_10_known_ode` (DAVIS visual input, 64,000 frames, 10% edges removed)
- Noise model: `noise_model_level=0.005, measurement_noise_level=0.0`
- Edge removal: **Per-column 10% removal** — connectivity matrix has 90% of true edges (390,701 edges)
- Re-generation: **NO** — data is fixed at startup. To test seed robustness, set `claude.test_robustness_seed: true`.

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
- `W_ij`: synaptic weight (connectivity, learned, with 10% missing)
- `g_phi`: edge activation function (fixed, typically ReLU)
- `a_j`: learnable neuron type embedding

**Model specs**:
- 13,741 neurons, 65 cell types, 390,701 edges (434,112 × 0.9, with 10% removed per column)
- 1,736 input neurons (photoreceptors)
- DAVIS visual input stimulus
- 64,000 frames, delta_t=0.02 (time resolution)

## known_ode Learning Task

The known_ode model **directly learns parameters** from voltage dynamics because the ODE is known:

**Learned parameters**:
- `W_ij`: connectivity (synaptic weights, 10% missing) — PRIMARY TARGET
- `tau_i`: time constants
- `V_rest_i`: resting potentials

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
| `lr_W` | 0.003 | [1e-4, 1e-2] | Learning rate for connectivity matrix W |
| `lr` | 0.006 | [1e-4, 1e-2] | Learning rate for tau_i, V_rest_i |

### Weight Initialization

| Parameter | Default | Options | Description |
| --- | --- | --- | --- |
| `w_init_mode` | `zeros` | `randn`, `randn_scaled`, `zeros` | Initialization distribution for W |
| `w_init_scale` | 1.0 | [0.1, 0.5, 1.0, 2.0] | Scaling factor for `randn_scaled` mode |

### Batch Size & Regularization

| Parameter | Default | Description |
| --- | --- | --- |
| `batch_size` | 16 | Number of time windows per gradient step (INTEGER) |
| `coeff_W_L1` | 0.00015 | L1 sparsity penalty on W |
| `coeff_W_L2` | 0.0 | L2 penalty on W (weight decay) |
| `dale_law` | true | Enforce Dale's law (sign consistency per pre-synaptic neuron) |
| `data_augmentation_loop` | 120 | Data augmentation multiplier (keep fixed unless testing) |

**CRITICAL CONSTRAINTS**:
- `batch_size` MUST be INTEGER (1, 2, 4, 8, 16, etc.)
- `n_epochs` MUST be INTEGER (1, 2, 3, etc.) — NOT 0.5
- `w_init_mode` MUST be LOWERCASE: `randn`, `randn_scaled`, `zeros`

## Training Time Budget — FIXED for Fair Comparison

**LOCKED PARAMETERS** (DO NOT MODIFY):
- `n_epochs: 1` — Single epoch training only
- `data_augmentation_loop (DAL): 120` — Fixed data augmentation

**Why these are locked**:

To fairly compare different LLM explorations across noise levels and biomodels, all known_ode variants must use the same training budget. This ensures observed differences in connectivity_R2 reflect parameter choices, not training time variation.

**Target training time**: ~60 minutes per iteration (consistent, cluster-efficient).

**If you believe n_epochs or DAL should be varied** to test a specific hypothesis, first post in `user_input.md` for authorization. Do NOT change these without explicit user approval.

> **YAML rule**: Always wrap the `description` field value in double quotes — colons inside unquoted YAML strings cause parse errors (e.g., `description: "Block 7 Slot 1: testing W_L2"`).

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
- **Stable-Robust**: all 4 slots connectivity_R2 > 0.87 AND CV < 3% — **TARGET**
- **Robust**: all 4 slots connectivity_R2 > 0.87, CV 3-5%
- **Partially robust**: 2-3 slots > 0.85
- **Fragile**: ≤1 slots > 0.80
- **DISQUALIFIED**: any slot < 0.75 — reject config immediately

## Block Structure — 10 Blocks × 8 Iterations Each

With `n_iter_block=8` and `iterations=80`, the exploration spans 10 hypothesis-driven blocks:

### Block 1 (iter 1-8): Robustness Validation — Starting Config
**Hypothesis**: "The starting config (bs=16, zeros, DAL=120, dale_law=True, lr_W=0.003) achieves connectivity_R2 > 0.87 across all 4 seeds for 10% edge removal, matching or exceeding the 20% removal baseline."

**Test**: Run all 4 slots with the base config unchanged.

**Expected outcome**: Validate the starting point before any optimization.

### Block 2 (iter 9-16): Learning Rate Sweep
**Hypothesis**: "10% removal may tolerate different LRs than 20% removal. Optimal LR will achieve connectivity_R2 > 0.87 while maintaining stability across seeds."

**Test**: Sweep `lr_W` and `lr` systematically
- Slot 0: lr_W=0.003, lr=0.006 (baseline)
- Slot 1: lr_W=0.001, lr=0.002 (conservative)
- Slot 2: lr_W=0.005, lr=0.010 (aggressive)
- Slot 3: lr_W=0.0005, lr=0.001 (very conservative)

### Block 3 (iter 17-24): W Regularization
**Hypothesis**: "L1 regularization may need tuning for 10% missing edges. Test whether coeff_W_L1 should stay at 0.00015 or be adjusted."

**Test**: Sweep `coeff_W_L1` and `coeff_W_L2`
- Slot 0: coeff_W_L1=0.00015, coeff_W_L2=0 (baseline)
- Slot 1: coeff_W_L1=0, coeff_W_L2=0 (no regularization)
- Slot 2: coeff_W_L1=0.0003, coeff_W_L2=0 (doubled)
- Slot 3: coeff_W_L1=0.00015, coeff_W_L2=1e-6 (add L2)

### Block 4 (iter 25-32): W Initialization + LR
**Hypothesis**: "Zero initialization may not be optimal for 10% removal. Test randn_scaled initialization."

**Test**: Sweep `w_init_mode, w_init_scale` with best LRs from Block 2
- Slot 0: w_init_mode=zeros, best LRs
- Slot 1: w_init_mode=randn_scaled, w_init_scale=0.1, best LRs
- Slot 2: w_init_mode=randn_scaled, w_init_scale=0.5, best LRs
- Slot 3: w_init_mode=randn_scaled, w_init_scale=1.0, best LRs

### Block 5 (iter 33-40): Batch Size
**Hypothesis**: "Incomplete connectivity may benefit from different batch sizes for stable gradient estimates."

**Test**: Sweep `batch_size`
- Slot 0: batch_size=16, best config
- Slot 1: batch_size=4, best config
- Slot 2: batch_size=8, best config
- Slot 3: batch_size=32, best config

### Block 6 (iter 41-48): Dale's Law
**Hypothesis**: "Dale's law constrains learning with incomplete edges. Test whether disabling it improves connectivity recovery."

**Test**: Sweep `dale_law`
- Slot 0: dale_law=True (baseline)
- Slot 1: dale_law=False
- Slot 2: dale_law=False, best init
- Slot 3: dale_law=True, best init + LR combination

### Block 7 (iter 49-56): Free Exploration
**Hypothesis**: Form based on Blocks 1-6 results. Explore parameter combinations not yet tested.

Test combinations of best settings from previous blocks.

### Block 8 (iter 57-64): Refinement
**Hypothesis**: Polish the best config to maximize connectivity_R2.

Fine-tune learning rates and regularization around the best config found.

### Block 9 (iter 65-72): Robustness Validation I
**Strategy**: Switch to **robustness mode** (all 4 slots same config, different seeds).

**Test**: Run the best config from Blocks 1-8 on 4 independent seeds.

### Block 10 (iter 73-80): Final Robustness Validation
**Strategy**: Final robustness check on best config found.

**Final test**: Run the best config from Blocks 1-9 on 4 independent seeds to validate connectivity_R2 > 0.87 with CV < 4%.

## File Structure

You maintain THREE files:

1. **Full Log (append-only)**: `flyvis_noise_005_removed_pc_10_known_ode_Claude_analysis.md`
   - Append every iteration's log entry (4 entries per batch)
   - Never read — human record only

2. **Working Memory (read + update every batch)**: `flyvis_noise_005_removed_pc_10_known_ode_Claude_memory.md`
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

### What to Add to Open Questions

- Patterns observed 1-2 times
- Seed-dependent effects (works for some seeds but not others)
- Contradictions between iterations
- Theoretical predictions not yet verified

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:
- State the original hypothesis
- State the contradicting evidence (iteration number, metrics)
- State what was learned from the falsification
- Propose a revised hypothesis if applicable

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
- **Stable-Robust**: all 4 slots connectivity_R2 > 0.87 AND CV < 3% — TARGET
- **Robust**: all 4 slots connectivity_R2 > 0.87, CV 3-5%
- **Partially robust**: 2-3 slots connectivity_R2 > 0.85
- **Fragile**: 0-1 slots connectivity_R2 > 0.80
- **DISQUALIFIED**: any slot < 0.75 — reject config immediately

### Step 3: Write Log Entry + Update Memory

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
2. Copy config from `log/Claude_exploration/LLM_flyvis_noise_005_removed_pc_10_known_ode/config/iter_XXX_slot_YY.yaml`
3. Save to `config/fly/flyvis_noise_005_removed_pc_10_known_ode_winner.yaml` with header:

```yaml
# Winner config: flyvis_noise_005_removed_pc_10_known_ode_winner.yaml
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

- Read base config — note 10% edges removed (incomplete connectivity)
- Initialize 4 slot configs for Block 1 (robustness validation)
- **Initial hypothesis**: "The starting config (bs=16, zeros init, DAL=120, dale_law=True, lr_W=0.003) achieves connectivity_R2 > 0.87 on all 4 seeds with CV < 4% for 10% edge removal"
- Set all 4 slots = baseline config (robustness test)

## Final Summary (write at exploration completion)

When the exploration is complete (all blocks done or budget exhausted), append to
`/home/node/.claude/projects/-workspace--devcontainer/memory/exploration_results.md`
a section with header `## flyvis_noise_005_removed_pc_10_known_ode — Key Discoveries (YYYY-MM-DD)` containing
exactly **8 bullet points**:

1. **Best metric**: conn_R2 = X.XXX ± std (N seeds, CV=X.X%), winner config = [key params]
2–8. **Key causal discoveries** — report findings of this kind:
   - Which HP had the largest single-parameter impact, and its optimal value
   - Which failure mode was confirmed across 3+ iterations (cite iteration numbers)
   - How 10% removal compares to 20% removal in terms of connectivity_R2 degradation
   - Which hypothesis was falsified and what was learned from it
   - Whether dale_law helped or hurt with incomplete connectivity
   - What W_init mode (zeros vs randn_scaled) proved optimal and why
   - Any fundamental limit encountered (e.g., conn_R2 ceiling due to 10% missing edges)

Each bullet must state the **finding**, the **evidence** (iteration count or specific iterations),
and whether it is **established** (3+ iterations, all 4 seeds) or **tentative** (1–2 iterations).

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_removed_pc_10_known_ode

## Paper Summary (update at every block boundary)

**GNN optimization** (2 sentences on HPO findings):
Sentence 1: Best hyperparameter configuration found and the connectivity_R2 it achieves (cite mean ± std, CV%, N seeds).
Sentence 2: Which hyperparameters were most critical to parameter recovery under 10% incomplete connectivity — what worked and what failed (cite values).

**LLM-driven exploration** (2 sentences on exploration findings):
Sentence 1: What the systematic exploration revealed about known_ode robustness to 10% missing edges (how much degradation vs. full connectivity, key failure modes).
Sentence 2: Main causal principle established — what this tells us about recovering W, tau, V_rest when 10% of edges are absent, and how this compares to 20% removal.

## Knowledge Base

### Results Comparison Table

| Iter | Slot | lr_W | lr | w_init_mode | w_init_scale | batch_size | coeff_W_L1 | coeff_W_L2 | dale_law | conn_R2 | tau_R2 | Vrest_R2 | rollout_r | time_min |
| ---- | ---- | ---- | --- | ----------- | ------------ | ---------- | ---------- | ---------- | -------- | ------- | ------ | -------- | --------- | -------- |

### Established Principles

[Rules that have been proven across 3+ iterations and 4/4 seeds]

### Falsified Hypotheses

[Hypotheses disproven by evidence]

### Open Questions

[Uncertainties still under investigation]

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest→newest. This section MUST appear before ## Current Block.**

### Block 1 Summary
[Summary of findings from block 1]

### Block 2 Summary
[Summary of findings from block 2]

### Block 3 Summary
[Summary of findings from block 3]

### Block 4 Summary
[Summary of findings from block 4]

---

## Current Block (Block N)

### Block Info
- Iterations: N-N+8
- Focus: [Block focus]

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
