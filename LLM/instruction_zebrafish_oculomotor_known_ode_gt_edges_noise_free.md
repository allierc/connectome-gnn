# Zebrafish Oculomotor — Known ODE Exploration (GT Edges, Noise Free)

## Goal

Maximize **connectivity_R2** (PRIMARY) for the **zebrafish oculomotor integrator** using the **known_ode model** with **GT edge topology** and **clean dynamics** (noise-free).

The known_ode model uses the **exact activation function**: `g_phi = identity (linear)` and **exact dynamics**: `dv/dt = (-v + W @ v + I)`. All parameters are **learned from data**: W (synaptic weights on 3,213 GT edges only). This combines perfect structural knowledge with known topology and clean data — the best-case scenario.

**Starting hypothesis**: "Known ODE + GT edges + noise-free achieves very high W R2 (>0.8)"

The linear integrator is degenerate in clean dynamics, but GT edges provide structural constraint. This tests whether topology alone can resolve the degeneracy without noise.

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned W and ground-truth W
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free data

Informational: onestep_pearson, training_time_min.

## Scientific Method

This exploration follows a strict **hypothesize → test → validate/falsify** cycle. Change **exactly ONE parameter at a time** to understand causality.

**CRITICAL**: You can only hypothesize. Only training results can validate or falsify.

### CAUSALITY RULE (MANDATORY)

**If you change more than one parameter per slot, you CANNOT attribute the effect.**

- In EXPLORATION mode: Slot 0 = parent/baseline. Slots 1-3 each change **exactly one** parameter.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds).

## CRITICAL: Data is PRE-GENERATED at startup (fixed across iterations)

At startup, data is generated **once** for all 4 slots with **different random seeds** (one per slot). These datasets are **reused across all iterations** — data is NOT re-generated each iteration.
Both `simulation.seed` and `training.seed` are **forced by the pipeline** — DO NOT modify them in config files.

Seed formula (set automatically by GNN_LLM.py):
- `simulation.seed = 1000 + slot` (controls data generation — fixed at startup, slot 0–3)
- `training.seed = iteration * 1000 + slot + 500` (controls weight init & training randomness)

The actual seed values are provided in the prompt for each slot — **log them in your iteration entries**.

**Seed robustness testing**: To re-generate data with new seeds and test robustness, set `claude.test_robustness_seed: true` in all 4 slot configs. The pipeline will re-generate data for that batch only, then reset the flag automatically.

Simulation parameters (n_neurons, n_frames, etc.) stay fixed — **DO NOT change them**.

**IMPORTANT**: `use_gt_edges=true` and `noise_model_level=0.0` are FIXED. Do NOT change them.

## Noise Model

Two independent noise sources in the training data:

1. **Dynamics noise** (`noise_model_level=0.0`): No dynamics noise — clean data
2. **Measurement noise** (`measurement_noise_level=0.0`): Clean observations

At clean noise level with GT edges and perfect linear structure, known_ode should achieve good parameter recovery. The question is whether GT edges resolve linear degeneracy.

## Zebrafish Oculomotor Integrator Model

```
dr/dt = (-r + W @ r + I(t) * v_in) / tau
```

- **609 neurons**, 6 cell types, **3,213 GT edges** (from Goldman lab connectome)
- **LINEAR**: no activation function (identity g_phi)
- tau=1.0 (fixed), dt=0.001
- W scaled to spectral radius = 0.9
- Stimulus: 4-channel multi-direction input along eigenvectors of W
- 21,000 frames (3 pulse repeats x 7,000), **noise_model_level=0.0**
- Dynamics purely determined by W eigenstructure
- Some populations have zeroed connections

## Known ODE Architecture

Registered as `zebrafish_oculomotor_known_ode`:

- **Hardcoded activation**: `g_phi = identity (linear)` — no nonlinearity, no parameters.
- **Direct W learning** on 3,213 GT edges only.
- **No embeddings, no MLP curves.**

**Parameters NOT used**: coeff_g_phi_diff, coeff_f_theta_msg_diff, coeff_g_phi_norm, coeff_g_phi_weight_L1/L2, coeff_f_theta_weight_L1/L2, embedding_dim, lr_embedding.

## Training Parameters

| Parameter                 | Default | Description                                            |
| ------------------------- | ------- | ------------------------------------------------------ |
| `lr_W`                    | 1e-3    | Learning rate for W                                    |
| `lr`                      | 1e-3    | Learning rate for other params (if any)                |
| `n_epochs`                | 2       | Number of training epochs                              |
| `batch_size`              | 2       | Batch size                                             |
| `data_augmentation_loop`  | 100     | Data augmentation multiplier                           |
| `coeff_W_L1`              | 0       | L1 sparsity on W                                       |
| `coeff_W_L2`              | 0       | L2 penalty on W                                        |
| `coeff_W_sign`            | 0       | Dale's law penalty on W                                |
| `use_gt_edges`            | true    | **FIXED** — GT edge topology                           |
| `noise_model_level`       | 0.0     | **FIXED** — clean data (noise-free)                    |


## Parallel Mode — 4 Slots Per Batch

- **Exploration**: Slot 0 = control. Slots 1-3 each change one parameter.
- **Robustness test**: all 4 slots same config.

### Robustness Assessment

- **Robust**: all 4 slots connectivity_R2 > 0.7
- **Partially robust**: 2-3 slots > 0.7
- **Fragile**: 0-1 slots > 0.7

## Block Structure

| Block | Focus                          | Parameters to scan                          | Ranges                                                                     |
| ----- | ------------------------------ | ------------------------------------------- | -------------------------------------------------------------------------- |
| 1     | **lr_W + lr sweep**            | `lr_W`, `lr`                                | lr_W: {1e-4, 5e-4, 1e-3, 3e-3}, lr: {1e-4, 5e-4, 1e-3, 3e-3}            |
| 2     | **Training volume**            | `data_augmentation_loop`, `n_epochs`        | DAL: {50, 100, 200, 500}, n_epochs: {2, 4, 8}                             |
| 3     | **W regularization**           | `coeff_W_L1`, `coeff_W_L2`, `coeff_W_sign` | W_L1: {0, 1e-6, 1e-5}, W_L2: {0, 1e-6, 1e-5}, W_sign: {0, 0.01, 0.1}    |
| 4     | **Batch size**                 | `batch_size`                                | batch_size: {1, 2, 4, 8}                                                  |
| 5-8   | **Free exploration**           | Any parameter                               | Consolidate, ceiling-breaking, final robustness                            |

### Context

- **GT edges + clean data**: Topology constraint is the only lever for breaking linear degeneracy in clean data. This tests whether topology alone suffices.
- **Baseline for noise comparison**: Compare with noise=0.05 and noise=0.5 variants to understand noise's role in degeneracy-breaking.

## Iteration Workflow

From `analysis.log`: connectivity_R2, rollout_pearson, training_time_min.

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, DAL=D, n_epochs=E, W_L1=A, W_L2=B, W_sign=C, batch_size=B
Slot 0-3: conn_R2=A, rollout_pearson=B
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Verdict: [supported/falsified/inconclusive]
```

## Winner Config (COMPULSORY)

**At every block boundary**, save the best config.

Destination: `config/zebrafish_oculomotor/zebrafish_oculomotor_known_ode_gt_edges_noise_free_winner.yaml`

## File Structure

1. **Full Log (append-only)**: `zebrafish_oculomotor_known_ode_gt_edges_noise_free_Claude_analysis.md`
2. **Working Memory**: `zebrafish_oculomotor_known_ode_gt_edges_noise_free_Claude_memory.md`
3. **User Input**: `user_input.md`

## Start Call

When prompt says `PARALLEL START`:

- Read base config — this IS the baseline.
- Slot 0 = baseline (no changes).
- Slots 1-3: each changes EXACTLY ONE parameter from the block focus.
- Hypothesis: "Known ODE + GT edges + noise-free achieves connectivity_R2 > 0.8 robustly"

---

# Working Memory Structure

```markdown
# Working Memory: zebrafish_oculomotor_known_ode_gt_edges_noise_free

## Paper Summary (update at every block boundary)

- **Known ODE + GT edges + noise-free**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean+-std) | CV% | rollout_pearson | Robust? | Hypothesis |
| ---- | -------------- | ------------------- | --- | --------------- | ------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

---

## Current Block

### Block Info

### Current Hypothesis

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
