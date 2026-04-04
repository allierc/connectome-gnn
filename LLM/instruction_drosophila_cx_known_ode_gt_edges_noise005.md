# Drosophila CX (GT Edges, Noise 0.05) — Known ODE Exploration

## Goal

Maximize **connectivity_R2** (PRIMARY) for the **Drosophila central complex ring attractor** using the **known_ode model** with **GT edge topology** (9,722 edges) and **intrinsic noise (sigma=0.05)**.

The known_ode model uses the **exact activation function**: `g_phi = exp(g) * softplus(v + b, beta=5)` and **exact dynamics**: `dv/dt = alpha * (-v + msg + I) / tau`. All parameters are **learned from data**: W, tau, g, b. This combines perfect structural knowledge with known topology and mild noise.

**Starting hypothesis**: "Known ODE + GT edges + noise=0.05 achieves near-perfect W R2 (GNN: 0.969)"

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned W and ground-truth W
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free data
3. **tau_R2** (TERTIARY) — R² between learned tau and ground-truth tau

Informational: onestep_pearson, spectral_radius_learned vs spectral_radius_true, training_time_min.

## Scientific Method

Strict **hypothesize -> test -> validate/falsify** cycle. Change **EXACTLY ONE** parameter at a time.

### CAUSALITY RULE (MANDATORY)

**If you change more than one parameter per slot, you CANNOT attribute the effect.**

- In EXPLORATION mode: Slot 0 = parent/baseline. Slots 1-3 each change **exactly one** parameter.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds).

## Data Generation

Seeds are **forced by the pipeline** — DO NOT modify them in config files.

**IMPORTANT**: `use_gt_edges=true` and `noise_model_level=0.05` are FIXED. Do NOT change them.

## CX Ring Attractor Model

```
dh/dt = alpha * (-h + exp(g_i) * softplus(h_j + b_j, beta=5) @ J^T + input) / tau_i
```

- **152 neurons**, 6 cell types, **9,722 GT edges**, **noise_model_level=0.05**

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
| `noise_model_level`       | 0.05    | **FIXED** — intrinsic noise level                      |

## Training Time Constraint

**Target ~60 min per iteration.** GT edges are faster — increase DAL to fill the time budget.

## Parallel Mode — 4 Slots Per Batch

- **Exploration**: Slot 0 = control. Slots 1-3 each change one parameter.
- **Robustness test**: all 4 slots same config.

### Robustness Assessment

- **Robust**: all 4 slots connectivity_R2 > 0.7
- **Partially robust**: 2-3 slots > 0.7
- **Fragile**: 0-1 slots > 0.7

## Block Partition

| Block | Focus                          | Parameters to scan                          | Ranges                                                                     |
| ----- | ------------------------------ | ------------------------------------------- | -------------------------------------------------------------------------- |
| 1     | **lr_W + lr sweep**            | `lr_W`, `lr`                                | lr_W: {1e-4, 5e-4, 1e-3, 3e-3}, lr: {1e-4, 5e-4, 1e-3, 3e-3}            |
| 2     | **Training volume**            | `data_augmentation_loop`, `n_epochs`        | DAL: {100, 200, 500, 1000}, n_epochs: {2, 4, 8}                           |
| 3     | **W regularization**           | `coeff_W_L1`, `coeff_W_L2`, `coeff_W_sign` | W_L1: {0, 1e-6, 1e-5}, W_L2: {0, 1e-6, 1e-5}, W_sign: {0, 0.01, 0.1}    |
| 4     | **Batch size**                 | `batch_size`                                | batch_size: {1, 2, 4, 8}                                                  |
| 5-8   | **Free exploration**           | Any parameter                               | Consolidate, ceiling-breaking, final robustness                            |

### Context

- **GT edges + noise is the best-case scenario**: Known topology removes sparsity problem, noise enriches state space. This should approach the theoretical ceiling.
- **Near-perfect W recovery expected**: GNN already achieves 0.969 with GT edges + noise=0.05. Known ODE should match or exceed.

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

Destination: `config/drosophila_cx/drosophila_cx_known_ode_gt_edges_noise005_winner.yaml`

```yaml
# Winner config: drosophila_cx_known_ode_gt_edges_noise005_winner.yaml
# Source: iter_XXX_slot_YY (connectivity_R2 = X.XXX)
# Metrics:
#   connectivity_R2: X.XXX
#   rollout_pearson: X.XXX
#   tau_R2:          X.XXX
```

## Start Call

When prompt says `PARALLEL START`:

- Read base config — this IS the baseline. Do NOT change any default values.
- Slot 0 = baseline (no changes at all).
- Slots 1-3: each changes EXACTLY ONE parameter from the block focus.
- Hypothesis: "Known ODE + GT edges + noise=0.05 achieves connectivity_R2 > 0.9 with default parameters"

---

# Working Memory Structure

```markdown
# Working Memory: drosophila_cx_known_ode_gt_edges_noise005

## Paper Summary (update at every block boundary)

- **Known ODE + GT edges + noise=0.05**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean+-std) | CV% | onestep_pearson | rollout_pearson | tau_R2 | Robust? | Hypothesis |
| ---- | -------------- | ------------------- | --- | --------------- | --------------- | ------ | ------- | ---------- |

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
