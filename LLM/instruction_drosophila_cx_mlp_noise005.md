# Drosophila CX — MLP Baseline + Noise 0.05 Exploration

## Goal

Maximize **rollout_pearson** (primary) and **connectivity_R2** (secondary) for the **MLP baseline** on the Drosophila central complex ring attractor with **intrinsic noise σ=0.05**.

The MLP baseline is a **flat, graph-free model**: `dv/dt = MLP([v_all; stimulus_all])`. No edges, no message passing — just a black-box MLP mapping all neuron states and stimuli to all derivatives. Connectivity is extracted post-hoc via the **Jacobian dF/dv**.

This serves as an **uninformed baseline** to compare against the GNN under noisy conditions. Prior result on clean data: W R2 ≈ 0 always (Jacobian fundamentally limited). Noise may or may not help MLP.

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Metrics (ranked by importance)

1. **rollout_pearson** (PRIMARY) — autoregressive rollout Pearson r on noise-free data
2. **connectivity_R2** (SECONDARY) — R² between Jacobian dF/dv and ground-truth W (dense matrix comparison)
3. **cluster_accuracy** (THIRD) — not applicable (no embeddings), always 0

Informational: onestep_pearson, spectral_radius_learned vs spectral_radius_true, training_time_min.

**NOTE**: tau_R2 and V_rest_R2 are always 0.0 (no explicit tau/V_rest parameters to compare).

## Scientific Method

Strict **hypothesize -> test -> validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction
2. **Design experiment**: Change **EXACTLY ONE** parameter at a time to understand causality
3. **Run training**: 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use metrics AND cross-seed variance
5. **Update understanding**: Revise hypotheses based on evidence

**CRITICAL**: You can only hypothesize. Only training results validate or falsify.

### CAUSALITY RULE (MANDATORY — READ THIS)

**If you change more than one parameter per slot, you CANNOT attribute the effect. This is a fatal experimental design error.**

- In EXPLORATION mode: Slot 0 = parent/baseline (unchanged control). Slots 1-3 each change **exactly one** parameter from the parent.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness).

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

At this mild noise level, the Jacobian may be disrupted. Test whether MLP with noise shows different Jacobian structure than clean data.

## CX Ring Attractor Model

```
dh/dt = alpha * (-h + exp(g_i) * softplus(h_j + b_j, beta=5) @ J^T + input) / tau_i
```

- **152 neurons**, 6 cell types, **9,722 edges**
- Ground truth connectivity is a structured ring attractor (block-diagonal + inhibitory bands)
- **Intrinsic noise σ=0.05** added during simulation

## MLP Architecture

```
input  = [v_1, ..., v_152, stim_1, ..., stim_152]    (304 dims)
output = [dv_1/dt, ..., dv_152/dt]                     (152 dims)
```

- Hidden layers with ReLU activation
- Last layer initialized to zeros for stable training start
- No graph structure, no per-edge weights
- Connectivity extracted via Jacobian: J[i,j] = d(dv_i/dt) / dv_j

## Training Parameters

| Parameter                 | Default | Description                                            |
| ------------------------- | ------- | ------------------------------------------------------ |
| `lr`                      | 1e-3    | Learning rate for MLP weights                          |
| `n_epochs`                | 20      | Number of training epochs                              |
| `batch_size`              | 2       | Batch size                                             |
| `data_augmentation_loop`  | 500     | Data augmentation multiplier                           |
| `hidden_dim`              | 256     | Hidden layer width                                     |
| `n_layers`                | 4       | Number of MLP layers (including input and output)      |

**Parameters NOT used by MLP** (set to 0, do not modify): lr_W, lr_embedding, coeff_W_L1, coeff_W_L2, coeff_W_sign, coeff_g_phi_diff, coeff_f_theta_diff, coeff_f_theta_msg_diff, embedding_dim.

## Training Time Constraint

**Target ~60 min per iteration.** Use `data_augmentation_loop` (DAL) to control training time.

- If training_time_min < 40 min: **increase** DAL
- If training_time_min > 70 min: **decrease** DAL

## Parallel Mode — 4 Slots Per Batch

Each batch runs 4 slots with different seeds (forced by pipeline). You choose the strategy:

- **Exploration** (default): Slot 0 = parent/control. Slots 1-3 each change one parameter.
- **Robustness test**: ALL 4 slots use the SAME config. Measures seed robustness.

## Block Structure

| Block | Focus                    | Parameters to scan                          | Ranges                                                          |
| ----- | ------------------------ | ------------------------------------------- | --------------------------------------------------------------- |
| 1     | **lr + architecture**    | `lr`, `hidden_dim`, `n_layers`              | lr: {1e-4, 5e-4, 1e-3, 3e-3}, hidden_dim: {128, 256, 512}, n_layers: {3, 4, 5} |
| 2     | **Training volume**      | `data_augmentation_loop`, `n_epochs`        | DAL: {200, 500, 1000}, n_epochs: {5, 10, 20}                   |
| 3     | **Batch size**           | `batch_size`                                | batch_size: {1, 2, 4, 8}                                       |
| 4     | **Capacity vs regularization** | `hidden_dim`, weight decay via lr     | Test if larger MLPs overfit or if more capacity helps Jacobian recovery |
| 5     | **Free exploration I**   | Any parameter                               | Consolidate best from blocks 1-4, test novel combinations       |
| 6     | **Free exploration II**  | Any parameter                               | Continue ceiling-breaking attempts                              |
| 7     | **Free exploration III** | Any parameter                               | Continue ceiling-breaking attempts                              |
| 8     | **Free exploration IV**  | Any parameter                               | Final refinement and robustness confirmation                    |

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

From `analysis.log`: connectivity_R2, rollout_pearson, training_time_min.

### Step 3: Write Log Entries + Update Memory

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr=X, DAL=D, n_epochs=E, hidden_dim=H, n_layers=L, batch_size=B
Slot 0: conn_R2=A, rollout_pearson=B, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, rollout_pearson=B, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, rollout_pearson=B, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, rollout_pearson=B, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive]
Next: parent=P

## Winner Config (COMPULSORY)

**At every block boundary**, you MUST save the current best config as a winner file.
This is a COMPULSORY task — do not skip it.

1. Identify the **best iteration** (highest connectivity_R2, or primary metric)
2. Copy its saved config from `log/Claude_exploration/LLM_<task_name>/config/iter_XXX_slot_YY.yaml`
3. Save it to `config/drosophila_cx/drosophila_cx_mlp_noise005_winner.yaml` with a YAML comment header:

```yaml
# Winner config: drosophila_cx_mlp_noise005_winner.yaml
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

Destination: `config/drosophila_cx/drosophila_cx_mlp_noise005_winner.yaml`

```

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

1. **Full Log (append-only)**: `drosophila_cx_mlp_noise005_Claude_analysis.md`
   - Append every iteration's log entry (4 entries per batch)
   - Never read — human record only

2. **Working Memory (read + update every batch)**: `drosophila_cx_mlp_noise005_Claude_memory.md`
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

Example: "lr=1e-3 with hidden_dim=256 on noise=0.05 achieves rollout_pearson > 0.55 robustly (3/3 iterations, all seeds > 0.50, CV < 5%)"

### What to Add to Open Questions

- Patterns observed 1-2 times
- Seed-dependent effects (works for some seeds but not others)
- Contradictions between iterations
- Theoretical predictions not yet verified

Example: "Does noise=0.05 improve Jacobian-based connectivity detection over clean? Only iter 1 tested."

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:
- State the original hypothesis
- State the contradicting evidence (iteration number, metrics)
- State what was learned from the falsification
- Propose a revised hypothesis if applicable

Example: "Hypothesis: 'Mild noise helps MLP extract Jacobian structure' — Falsified by iter 2 (noise=0.05 gave CV=9%, worse than clean). Revised: 'Noise disrupts Jacobian; clean data remains better for MLP.'"

## Start Call

When prompt says `PARALLEL START`:

- Read base config — this IS the baseline. Do NOT change any default values.
- Slot 0 = baseline (no changes at all).
- Slots 1-3: each changes EXACTLY ONE parameter from the block focus.
- Hypothesis: "The MLP baseline with noise=0.05 achieves rollout_pearson > 0.5 and connectivity_R2 > 0.1 via Jacobian extraction"

---

# Working Memory Structure

```markdown
# Working Memory: drosophila_cx_mlp_noise005

## Paper Summary (update at every block boundary)

- **MLP baseline + noise 0.05**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean+-std) | CV% | rollout_r | Robust? | Hypothesis |
| ---- | -------------- | ------------------- | --- | --------- | ------- | ---------- |

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
