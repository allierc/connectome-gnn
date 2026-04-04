# Drosophila CX (Noise 0.5) — Known ODE Exploration

## Goal

Maximize **connectivity_R2** (PRIMARY) for the **Drosophila central complex ring attractor** using the **known_ode model** under **strong intrinsic noise (sigma=0.5)**.

The known_ode model uses the **exact activation function**: `g_phi = exp(g) * softplus(v + b, beta=5)` and **exact dynamics**: `dv/dt = alpha * (-v + msg + I) / tau`. All parameters are **learned from data**: W, tau, g, b. This is an **upper bound** on what is achievable with perfect structural knowledge.

**Starting hypothesis**: "Known ODE + noise=0.5 matches or exceeds GNN (GNN: 0.999)"

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

**IMPORTANT**: `noise_model_level` is set to **0.5** in the base config. Do NOT change it — this file is specifically for the noise=0.5 experiment.

## CX Ring Attractor Model

```
dh/dt = alpha * (-h + exp(g_i) * softplus(h_j + b_j, beta=5) @ J^T + input) / tau_i
```

- **152 neurons**, 6 cell types, **9,722 GT edges**, **22,952 FC edges**
- tau bounded [0.2, 5.0], alpha=0.2, beta=5 (softplus sharpness)
- 10,000 frames, delta_t=0.1, bump + velocity stimuli, **noise_model_level=0.5**

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
| `noise_model_level`       | 0.5     | **FIXED** — strong noise level for this experiment     |

## Training Time Constraint

**Target ~60 min per iteration.** Use `data_augmentation_loop` (DAL) to control training time.

- If training_time_min < 40 min: **increase** DAL
- If training_time_min > 70 min: **decrease** DAL

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
| 1     | **lr_W + lr sweep**            | `lr_W`, `lr`                                | lr_W: {1e-4, 5e-4, 1e-3, 3e-3}, lr: {1e-4, 5e-4, 1e-3, 3e-3}. Strong noise may need lower lr for stability.    |
| 2     | **Training volume**            | `data_augmentation_loop`, `n_epochs`        | DAL: {50, 100, 200, 500}, n_epochs: {2, 4, 8}. Strong noise needs more data to average out variance.            |
| 3     | **W regularization**           | `coeff_W_L1`, `coeff_W_L2`, `coeff_W_sign` | W_L1: {0, 1e-6, 1e-5, 1e-4}, W_L2: {0, 1e-6, 1e-5, 1e-4}, W_sign: {0, 0.01, 0.1}.                             |
| 4     | **Batch size**                 | `batch_size`                                | batch_size: {1, 2, 4, 8}. Larger batches smooth noisy gradients under strong noise.                             |
| 5-8   | **Free exploration**           | Any parameter                               | Consolidate best, ceiling-breaking, final robustness test.                                                       |

### Noise-specific considerations

- **Strong noise (0.5) greatly enriches state space**: From GNN experience, noise=0.5 pushed CX connectivity R2 to 0.999. Known ODE should achieve similar or better.
- **Gradient noise is high**: sigma=0.5 is large relative to signal — may need larger batches or more training to converge.
- **W recovery may be near-perfect**: With known structure + strong noise, the inverse problem becomes well-conditioned.

## Iteration Workflow

### Step 1-5: Same as base known_ode instruction

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

Destination: `config/drosophila_cx/drosophila_cx_known_ode_noise05_winner.yaml`

```yaml
# Winner config: drosophila_cx_known_ode_noise05_winner.yaml
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
- Hypothesis: "Known ODE + noise=0.5 achieves connectivity_R2 > 0.8 with default parameters due to strong state-space enrichment"

---

# Working Memory Structure

```markdown
# Working Memory: drosophila_cx_known_ode_noise05

## Paper Summary (update at every block boundary)

- **Known ODE optimization**: [pending]
- **LLM-driven exploration**: [pending]

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
