# Drosophila CX (Noise 0.05) — Known ODE Exploration

## Goal

Maximize **connectivity_R2** (PRIMARY) for the **Drosophila central complex ring attractor** using the **known_ode model** under **intrinsic noise (sigma=0.05)**.

The known_ode model uses the **exact activation function**: `g_phi = exp(g) * softplus(v + b, beta=5)` and **exact dynamics**: `dv/dt = alpha * (-v + msg + I) / tau`. All parameters are **learned from data**: W (synaptic weights), tau (time constants), g (gain), bias (per-neuron bias in activation). This is an **upper bound** on what is achievable with perfect structural knowledge.

**Starting hypothesis**: "Known ODE + noise=0.05 achieves W R2 > 0.95 (GNN: 0.982)"

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned W and ground-truth W
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free data
3. **tau_R2** (TERTIARY) — R² between learned tau and ground-truth tau

Informational: onestep_pearson, spectral_radius_learned vs spectral_radius_true, training_time_min.

**NOTE**: V_rest_R2 is always 0.0. cluster_accuracy is not applicable (no learned embeddings).

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

### CAUSALITY RULE (MANDATORY — READ THIS)

**If you change more than one parameter per slot, you CANNOT attribute the effect. This is a fatal experimental design error.**

- In EXPLORATION mode: Slot 0 = parent/baseline (unchanged control). Slots 1-3 each change **exactly one** parameter from the parent.
- Do NOT change parameters outside the current block focus.
- Do NOT skip the baseline — always keep one slot as an unchanged control.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness).

## CRITICAL: Data is RE-GENERATED per slot

Each slot re-generates its data with a **different random seed**.
Both `simulation.seed` and `training.seed` are **forced by the pipeline** — DO NOT modify them in config files.

Seed formula (set automatically by GNN_LLM.py):
- `simulation.seed = iteration * 1000 + slot` (controls data generation)
- `training.seed = iteration * 1000 + slot + 500` (controls weight init & training randomness)

The actual seed values are provided in the prompt for each slot — **log them in your iteration entries**.

Simulation parameters (n_neurons, n_frames, etc.) stay fixed — **DO NOT change them**.

## Data Generation

Each slot re-generates data with a **different random seed**.
Seeds are **forced by the pipeline** — DO NOT modify them in config files.

- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

**DO NOT change `simulation:` parameters** except seed (managed automatically).

**IMPORTANT**: `noise_model_level` is set to **0.05** in the base config. Do NOT change it — this file is specifically for the noise=0.05 experiment.

## Noise Model

Two independent noise sources in the training data:

1. **Dynamics noise** (`noise_model_level=0.05`): `v(t+1) = v(t) + dt * f(v, W, I) + epsilon_dyn(t)`, epsilon_dyn ~ N(0, 0.05)
2. **Measurement noise** (`measurement_noise_level=0.0`): Clean observations

At this mild noise level, known_ode optimization should remain fairly tractable. The challenge is parameter estimation from noisy observations.

## CX Ring Attractor Model

```
dh/dt = alpha * (-h + exp(g_i) * softplus(h_j + b_j, beta=5) @ J^T + input) / tau_i
```

- **152 neurons**, 6 cell types, **9,722 GT edges**, **22,952 FC edges**
- tau bounded [0.2, 5.0], alpha=0.2, beta=5 (softplus sharpness)
- 10,000 frames, delta_t=0.1, bump + velocity stimuli, **noise_model_level=0.05**

## Known ODE Architecture

The model is registered as `drosophila_cx_known_ode`. Unlike the GNN:

- **No learned MLP curves**: Activation function `g_phi = exp(g) * softplus(v + b, beta=5)` is hardcoded. Parameters g and b are learned per neuron.
- **No embeddings**: No per-neuron type embedding vectors.
- **Direct W learning**: Synaptic weight matrix W is learned directly on graph edges.
- **Direct tau learning**: Time constants tau are learned per neuron.

**Parameters NOT used by known_ode** (do not modify): coeff_g_phi_diff, coeff_f_theta_msg_diff, coeff_g_phi_norm, coeff_g_phi_weight_L1/L2, coeff_f_theta_weight_L1/L2, embedding_dim, lr_embedding.

## Training Parameters

| Parameter                 | Default | Description                                            |
| ------------------------- | ------- | ------------------------------------------------------ |
| `lr_W`                    | 1e-3    | Learning rate for W (synaptic weights)                 |
| `lr`                      | 1e-3    | Learning rate for other params (tau, g, b)             |
| `n_epochs`                | 2       | Number of training epochs                              |
| `batch_size`              | 2       | Batch size                                             |
| `data_augmentation_loop`  | 100     | Data augmentation multiplier                           |
| `coeff_W_L1`              | 0       | L1 sparsity on W                                       |
| `coeff_W_L2`              | 0       | L2 penalty on W                                        |
| `coeff_W_sign`            | 0       | Dale's law penalty on W                                |
| `use_gt_edges`            | false   | Fully connected graph (22,952 edges)                   |
| `noise_model_level`       | 0.05    | **FIXED** — intrinsic noise level for this experiment  |


## Parallel Mode — 4 Slots Per Batch

Each batch runs 4 slots with different seeds (forced by pipeline). You choose the strategy:

- **Exploration** (default): Slot 0 = parent/control. Slots 1-3 each change **exactly one** parameter.
- **Robustness test**: ALL 4 slots use the SAME config (different seeds test robustness).

### Robustness Assessment

- **Robust**: all 4 slots connectivity_R2 > 0.7
- **Partially robust**: 2-3 slots > 0.7
- **Fragile**: 0-1 slots > 0.7

## Block Structure

**CRITICAL**: The exploration is organized into blocks for hypothesis testing. Each block contains `n_iter_block` iterations (from config file, typically 12).

With `iterations=84` and `n_iter_block=12`:
- **Total blocks**: 7 (iterations 1-12, 13-24, 25-36, 37-48, 49-60, 61-72, 73-84)
- **Per batch**: 4 parallel slots (determined by `n_parallel`, typically 4)
- **Per block**: 3 batches of 4 slots each = 12 iterations

**How to read the prompt**:
The prompt will say: `Block info: block {N}, iterations {X}-{Y}/{n_iter_block} within block`

Example: `block 1, iterations 1-4/12 within block` means:
- You are in block 1 (first hypothesis test)
- Current batch is iterations 1-4
- Block 1 will eventually run iterations 1-12

**Your role**: Plan a coherent hypothesis for the entire block (12 iterations = 3 batches of 4 slots). The 4 slots per batch let you test multiple parameters in parallel, while the 12 iterations per block give you 3 rounds to refine based on evidence.

## Block Partition

| Block | Focus                          | Parameters to scan                          | Ranges                                                                                                           |
| ----- | ------------------------------ | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1     | **lr_W + lr sweep**            | `lr_W`, `lr`                                | lr_W: {1e-4, 5e-4, 1e-3, 3e-3}, lr: {1e-4, 5e-4, 1e-3, 3e-3}. Noise may require different lr balance.          |
| 2     | **Training volume**            | `data_augmentation_loop`, `n_epochs`        | DAL: {50, 100, 200, 500}, n_epochs: {2, 4, 8}. Noise reduces signal-to-noise — more training may help average out. |
| 3     | **W regularization**           | `coeff_W_L1`, `coeff_W_L2`, `coeff_W_sign` | W_L1: {0, 1e-6, 1e-5, 1e-4}, W_L2: {0, 1e-6, 1e-5, 1e-4}, W_sign: {0, 0.01, 0.1}.                             |
| 4     | **Batch size**                 | `batch_size`                                | batch_size: {1, 2, 4, 8}. Larger batches may smooth noisy gradients.                                            |
| 5-8   | **Free exploration**           | Any parameter                               | Consolidate best from blocks 1-4, ceiling-breaking, final robustness test.                                       |

### Noise-specific considerations

- **Noise may help known_ode**: Noise enriches state-space exploration, potentially making W more identifiable from dynamics.
- **Noise=0.05 is mild**: From GNN experience, noise=0.05 improved CX connectivity R2 from 0.804 to 0.982. Known ODE should benefit similarly or more.
- **More training may help**: Noise adds variance to gradients — more DAL or epochs help average out.

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

From `analysis.log`: connectivity_R2, rollout_pearson, tau_R2, training_time_min.

### Step 3: Write Log Entries + Update Memory

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, DAL=D, n_epochs=E, W_L1=A, W_L2=B, W_sign=C, batch_size=B
Slot 0: conn_R2=A, rollout_pearson=B, tau_R2=C, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, rollout_pearson=B, tau_R2=C, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, rollout_pearson=B, tau_R2=C, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, rollout_pearson=B, tau_R2=C, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive]
Next: parent=P
```

## Winner Config (COMPULSORY)

**At every block boundary**, you MUST save the current best config as a winner file.

1. Identify the **best iteration** (highest connectivity_R2)
2. Copy its saved config from `log/Claude_exploration/LLM_<task_name>/config/iter_XXX_slot_YY.yaml`
3. Save it to `config/drosophila_cx/drosophila_cx_known_ode_noise_005_winner.yaml` with a YAML comment header:

```yaml
# Winner config: drosophila_cx_known_ode_noise_005_winner.yaml
# Source: iter_XXX_slot_YY (connectivity_R2 = X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [1-2 sentence narrative]
#
# Metrics:
#   connectivity_R2: X.XXX (best single seed)
#   robust_mean:     X.XXX +/- X.XXX (N seeds, CV=X.X%)
#   rollout_pearson: X.XXX
#   tau_R2:          X.XXX
#
# Key config differences from baseline:
#   - [list]
```

Destination: `config/drosophila_cx/drosophila_cx_known_ode_noise_005_winner.yaml`

### Step 4: Acknowledge User Input

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Block Boundaries

1. Update "Paper Summary"
2. Summarize block findings
3. Update "Established Principles"
4. Clear "Current Block"
5. Carry forward best config

## File Structure

You maintain THREE files:

1. **Full Log (append-only)**: `drosophila_cx_known_ode_noise_005_Claude_analysis.md`
   - Append every iteration's log entry (4 entries per batch)
   - Never read — human record only

2. **Working Memory (read + update every batch)**: `drosophila_cx_known_ode_noise_005_Claude_memory.md`
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

Example: "lr_W=1e-3 with lr=1e-3 on noise=0.05 achieves connectivity_R2 > 0.5 robustly (3/3 iterations, all seeds > 0.45, CV < 3%)"

### What to Add to Open Questions

- Patterns observed 1-2 times
- Seed-dependent effects (works for some seeds but not others)
- Contradictions between iterations
- Theoretical predictions not yet verified

Example: "Does higher W_L2 improve robustness at moderate noise? Only iter 2 tested."

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:
- State the original hypothesis
- State the contradicting evidence (iteration number, metrics)
- State what was learned from the falsification
- Propose a revised hypothesis if applicable

Example: "Hypothesis: 'Known ODE + noise=0.05 achieves R2 > 0.95' — Falsified by iter 1 (best only 0.60). Revised: 'Moderate noise reduces recovery efficiency; stronger regularization needed.'"

## Start Call

When prompt says `PARALLEL START`:

- Read base config — this IS the baseline. Do NOT change any default values.
- Slot 0 = baseline (no changes at all).
- Slots 1-3: each changes EXACTLY ONE parameter from the block focus.
- Hypothesis: "Known ODE + noise=0.05 achieves connectivity_R2 > 0.5 with default parameters"

---

# Working Memory Structure

```markdown
# Working Memory: drosophila_cx_known_ode_noise_005

## Paper Summary (update at every block boundary)

- **Known ODE optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean+-std) | CV% | onestep_pearson | rollout_pearson | tau_R2 | V_rest_R2 | Robust? | Hypothesis |
| ---- | -------------- | ------------------- | --- | --------------- | --------------- | ------ | --------- | ------- | ---------- |

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
**Rationale**: [why]
**Test**: [what config change]
**Expected outcome**: [support vs falsify]
**Status**: untested / supported / falsified

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
