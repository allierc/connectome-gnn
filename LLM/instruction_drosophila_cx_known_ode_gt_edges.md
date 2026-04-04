# Drosophila CX (GT Edges) — Known ODE Exploration

## Goal

Maximize **connectivity_R2** (PRIMARY) for the **Drosophila central complex ring attractor** using the **known_ode model** with **ground-truth edge topology** (9,722 edges instead of 22,952 FC edges).

The known_ode model uses the **exact activation function**: `g_phi = exp(g) * softplus(v + b, beta=5)` and **exact dynamics**: `dv/dt = alpha * (-v + msg + I) / tau`. All parameters are **learned from data**: W (synaptic weights), tau (time constants), g (gain), bias (per-neuron bias in activation). This is an **upper bound** on what is achievable with perfect structural knowledge.

With GT edges, the model only needs to learn W values on the **correct edges** — no need to zero out non-existent connections. This removes the sparsity recovery problem entirely.

**Starting hypothesis**: "Known ODE + GT edges achieves W R2 > 0.9 (GNN: 0.893)"

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned W and ground-truth W
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free data
3. **tau_R2** (TERTIARY) — R² between learned tau and ground-truth tau

Informational: onestep_pearson, spectral_radius_learned vs spectral_radius_true, training_time_min.

**NOTE**: V_rest_R2 is always 0.0. cluster_accuracy is not applicable (no learned embeddings).

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

**IMPORTANT**: `use_gt_edges` is set to **true** in the base config. Do NOT change it — this file is specifically for the GT edges experiment.

## CX Ring Attractor Model

```
dh/dt = alpha * (-h + exp(g_i) * softplus(h_j + b_j, beta=5) @ J^T + input) / tau_i
```

- **152 neurons**, 6 cell types, **9,722 GT edges** (used in this experiment)
- tau bounded [0.2, 5.0], alpha=0.2, beta=5 (softplus sharpness)
- 10,000 frames, delta_t=0.1, bump + velocity stimuli

## Known ODE Architecture

The model is registered as `drosophila_cx_known_ode`. Unlike the GNN:

- **No learned MLP curves**: Activation function `g_phi = exp(g) * softplus(v + b, beta=5)` is hardcoded. Parameters g and b are learned per neuron.
- **No embeddings**: No per-neuron type embedding vectors.
- **Direct W learning**: Synaptic weight matrix W is learned directly on the **9,722 GT edges** only.
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
| `use_gt_edges`            | true    | **FIXED** — GT edge topology (9,722 edges)             |
| `noise_model_level`       | 0.0     | No observation noise                                   |

## Training Time Constraint

**Target ~60 min per iteration.** Use `data_augmentation_loop` (DAL) to control training time.

- If training_time_min < 40 min: **increase** DAL
- If training_time_min > 70 min: **decrease** DAL

Note: GT edges (9,722) are ~2.4x fewer than FC (22,952), so training is faster per epoch. Increase DAL accordingly.

## Parallel Mode — 4 Slots Per Batch

- **Exploration** (default): Slot 0 = parent/control. Slots 1-3 each change **exactly one** parameter.
- **Robustness test**: ALL 4 slots use the SAME config (different seeds test robustness).

### Robustness Assessment

- **Robust**: all 4 slots connectivity_R2 > 0.7
- **Partially robust**: 2-3 slots > 0.7
- **Fragile**: 0-1 slots > 0.7

## Block Partition

| Block | Focus                          | Parameters to scan                          | Ranges                                                                                                           |
| ----- | ------------------------------ | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1     | **lr_W + lr sweep**            | `lr_W`, `lr`                                | lr_W: {1e-4, 5e-4, 1e-3, 3e-3}, lr: {1e-4, 5e-4, 1e-3, 3e-3}. Fewer edges = different lr landscape.            |
| 2     | **Training volume**            | `data_augmentation_loop`, `n_epochs`        | DAL: {100, 200, 500, 1000}, n_epochs: {2, 4, 8}. GT edges are faster — can afford higher DAL.                   |
| 3     | **W regularization**           | `coeff_W_L1`, `coeff_W_L2`, `coeff_W_sign` | W_L1: {0, 1e-6, 1e-5}, W_L2: {0, 1e-6, 1e-5}, W_sign: {0, 0.01, 0.1}. L1 less needed (no spurious edges to zero out). |
| 4     | **Batch size**                 | `batch_size`                                | batch_size: {1, 2, 4, 8}.                                                                                       |
| 5-8   | **Free exploration**           | Any parameter                               | Consolidate best, ceiling-breaking, final robustness test.                                                       |

### GT edges context

- **No sparsity recovery needed**: With GT edges, W only exists on true connections. L1 regularization is less important — focus on W magnitude accuracy.
- **Faster training**: 9,722 vs 22,952 edges means ~2.4x fewer W parameters and faster message passing. Use higher DAL to fill the time budget.
- **W_sign / Dale's law may help**: GT edges + Dale's law directly constrains sign per column, which is known a priori for CX.

## Iteration Workflow

From `analysis.log`: connectivity_R2, rollout_pearson, tau_R2, training_time_min.

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, DAL=D, n_epochs=E, W_L1=A, W_L2=B, W_sign=C, batch_size=B
Slot 0-3: conn_R2=A, rollout_pearson=B, tau_R2=C, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive]
Next: parent=P
```

## Winner Config (COMPULSORY)

**At every block boundary**, save the best config.

Destination: `config/drosophila_cx/drosophila_cx_known_ode_gt_edges_winner.yaml`

```yaml
# Winner config: drosophila_cx_known_ode_gt_edges_winner.yaml
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

## Start Call

When prompt says `PARALLEL START`:

- Read base config — this IS the baseline. Do NOT change any default values.
- Slot 0 = baseline (no changes at all).
- Slots 1-3: each changes EXACTLY ONE parameter from the block focus.
- Hypothesis: "Known ODE + GT edges achieves connectivity_R2 > 0.7 with default parameters, since the sparsity problem is eliminated"

---

# Working Memory Structure

```markdown
# Working Memory: drosophila_cx_known_ode_gt_edges

## Paper Summary (update at every block boundary)

- **Known ODE + GT edges optimization**: [pending]
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

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
