# Drosophila CX (GT Edges, Noise 0.5) — Known ODE Exploration

## Goal

Maximize **connectivity_R2** (PRIMARY) for the **Drosophila central complex ring attractor** using the **known_ode model** with **GT edge topology** (9,722 edges) and **strong intrinsic noise (sigma=0.5)**.

The known_ode model uses the **exact activation function**: `g_phi = exp(g) * softplus(v + b, beta=5)` and **exact dynamics**: `dv/dt = alpha * (-v + msg + I) / tau`. All parameters are **learned from data**: W, tau, g, b. This is the most favorable setting: perfect ODE structure, known topology, and strong noise for state-space enrichment.

**Starting hypothesis**: "Known ODE + GT edges + noise=0.5 achieves W R2 approximately 1.0 (GNN: 0.999+)"

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned W and ground-truth W
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free data
3. **tau_R2** (TERTIARY) — R² between learned tau and ground-truth tau

Informational: onestep_pearson, spectral_radius_learned vs spectral_radius_true, training_time_min.

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

### CAUSALITY RULE (MANDATORY)

**If you change more than one parameter per slot, you CANNOT attribute the effect.**

- In EXPLORATION mode: Slot 0 = parent/baseline. Slots 1-3 each change **exactly one** parameter.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds).

## CRITICAL: Data is RE-GENERATED per slot

Each slot re-generates its data with a **different random seed**.
Both `simulation.seed` and `training.seed` are **forced by the pipeline** — DO NOT modify them in config files.

Seed formula (set automatically by GNN_LLM.py):
- `simulation.seed = iteration * 1000 + slot` (controls data generation)
- `training.seed = iteration * 1000 + slot + 500` (controls weight init & training randomness)

The actual seed values are provided in the prompt for each slot — **log them in your iteration entries**.

Simulation parameters (n_neurons, n_frames, etc.) stay fixed — **DO NOT change them**.

**IMPORTANT**: `use_gt_edges=true` and `noise_model_level=0.5` are FIXED. Do NOT change them.

## Noise Model

Two independent noise sources in the training data:

1. **Dynamics noise** (`noise_model_level=0.5`): `v(t+1) = v(t) + dt * f(v, W, I) + epsilon_dyn(t)`, epsilon_dyn ~ N(0, 0.5)
2. **Measurement noise** (`measurement_noise_level=0.0`): Clean observations

At high noise with GT edges and perfect structure, strong state-space enrichment should enable near-perfect parameter recovery.

## CX Ring Attractor Model

```
dh/dt = alpha * (-h + exp(g_i) * softplus(h_j + b_j, beta=5) @ J^T + input) / tau_i
```

- **152 neurons**, 6 cell types, **9,722 GT edges**, **noise_model_level=0.5**

## Known ODE Architecture

Registered as `drosophila_cx_known_ode`:

- **Hardcoded activation**: `g_phi = exp(g) * softplus(v + b, beta=5)` — g, b learned per neuron.
- **Direct W learning** on 9,722 GT edges only.
- **Direct tau learning** per neuron.
- **No embeddings, no MLP curves.**

**Parameters NOT used**: coeff_g_phi_diff, coeff_f_theta_msg_diff, coeff_g_phi_norm, coeff_g_phi_weight_L1/L2, coeff_f_theta_weight_L1/L2, embedding_dim, lr_embedding.

## Training Parameters

| Parameter                 | Default | Description                                            |
| ------------------------- | ------- | ------------------------------------------------------ |
| `lr_W`                    | 1e-3    | Learning rate for W                                    |
| `lr`                      | 1e-3    | Learning rate for tau, g, b                            |
| `n_epochs`                | 2       | Number of training epochs                              |
| `batch_size`              | 2       | Batch size                                             |
| `data_augmentation_loop`  | 100     | Data augmentation multiplier                           |
| `coeff_W_L1`              | 0       | L1 sparsity on W                                       |
| `coeff_W_L2`              | 0       | L2 penalty on W                                        |
| `coeff_W_sign`            | 0       | Dale's law penalty on W                                |
| `use_gt_edges`            | true    | **FIXED** — GT edge topology                           |
| `noise_model_level`       | 0.5     | **FIXED** — strong noise level                         |

## Training Time Constraint

**Target ~60 min per iteration.** GT edges are faster — increase DAL to fill the time budget.

## Parallel Mode — 4 Slots Per Batch

- **Exploration**: Slot 0 = control. Slots 1-3 each change one parameter.
- **Robustness test**: all 4 slots same config.

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

## Block Structure

| Block | Focus                          | Parameters to scan                          | Ranges                                                                     |
| ----- | ------------------------------ | ------------------------------------------- | -------------------------------------------------------------------------- |
| 1     | **lr_W + lr sweep**            | `lr_W`, `lr`                                | lr_W: {1e-4, 5e-4, 1e-3, 3e-3}, lr: {1e-4, 5e-4, 1e-3, 3e-3}            |
| 2     | **Training volume**            | `data_augmentation_loop`, `n_epochs`        | DAL: {100, 200, 500, 1000}, n_epochs: {2, 4, 8}                           |
| 3     | **W regularization**           | `coeff_W_L1`, `coeff_W_L2`, `coeff_W_sign` | W_L1: {0, 1e-6, 1e-5}, W_L2: {0, 1e-6, 1e-5}, W_sign: {0, 0.01, 0.1}    |
| 4     | **Batch size**                 | `batch_size`                                | batch_size: {1, 2, 4, 8}                                                  |
| 5-8   | **Free exploration**           | Any parameter                               | Consolidate, ceiling-breaking, final robustness                            |

### Context

- **This is the theoretical ceiling**: Known ODE + GT edges + strong noise = maximum information. If W R2 < 1.0 here, some fundamental identifiability issue exists.
- **GNN achieves 0.999+**: The GNN already nearly saturates under these conditions. Known ODE should match — if it does not, the GNN's flexible MLP may be compensating for some systematic mismatch.
- **Strong noise + GT edges**: High gradient variance but rich state coverage on only true connections.

## Iteration Workflow

From `analysis.log`: connectivity_R2, rollout_pearson, tau_R2, training_time_min.

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, DAL=D, n_epochs=E, W_L1=A, W_L2=B, W_sign=C, batch_size=B
Slot 0-3: conn_R2=A, rollout_pearson=B, tau_R2=C
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Verdict: [supported/falsified/inconclusive]
```

## Winner Config (COMPULSORY)

**At every block boundary**, save the best config.

Destination: `config/drosophila_cx/drosophila_cx_known_ode_gt_edges_noise05_winner.yaml`

```yaml
# Winner config: drosophila_cx_known_ode_gt_edges_noise05_winner.yaml
# Source: iter_XXX_slot_YY (connectivity_R2 = X.XXX)
# Metrics:
#   connectivity_R2: X.XXX
#   rollout_pearson: X.XXX
#   tau_R2:          X.XXX
```

## File Structure

You maintain THREE files:

1. **Full Log (append-only)**: `drosophila_cx_known_ode_gt_edges_noise05_Claude_analysis.md`
   - Append every iteration's log entry (4 entries per batch)
   - Never read — human record only

2. **Working Memory (read + update every batch)**: `drosophila_cx_known_ode_gt_edges_noise05_Claude_memory.md`
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

Example: "lr_W=1e-3 with GT edges + noise=0.5 achieves connectivity_R2 > 0.9 robustly (3/3 iterations, all seeds > 0.85, CV < 2%)"

### What to Add to Open Questions

- Patterns observed 1-2 times
- Seed-dependent effects (works for some seeds but not others)
- Contradictions between iterations
- Theoretical predictions not yet verified

Example: "Does known_ode match GNN performance at this optimal setting? Still exploring."

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:
- State the original hypothesis
- State the contradicting evidence (iteration number, metrics)
- State what was learned from the falsification
- Propose a revised hypothesis if applicable

Example: "Hypothesis: 'Known ODE + GT edges + noise=0.5 achieves R2 ≈ 1.0' — Falsified by iter 1 (best 0.91). Revised: 'Even with perfect structure and favorable noise, parameter estimation has fundamental limits.'"

## Start Call

When prompt says `PARALLEL START`:

- Read base config — this IS the baseline. Do NOT change any default values.
- Slot 0 = baseline (no changes at all).
- Slots 1-3: each changes EXACTLY ONE parameter from the block focus.
- Hypothesis: "Known ODE + GT edges + noise=0.5 achieves connectivity_R2 > 0.95 with default parameters"

---

# Working Memory Structure

```markdown
# Working Memory: drosophila_cx_known_ode_gt_edges_noise05

## Paper Summary (update at every block boundary)

- **Known ODE + GT edges + noise=0.5**: [pending]

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

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
