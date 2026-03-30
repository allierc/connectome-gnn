# Drosophila CX — MLP Baseline + GT Edges + Noise 0.05 Exploration

## Goal

Maximize **rollout_pearson** (primary) and **connectivity_R2** (secondary) for the **MLP baseline** on the Drosophila central complex ring attractor with **GT edges topology + intrinsic noise σ=0.05**.

The MLP baseline is a **flat, graph-free model**: `dv/dt = MLP([v_all; stimulus_all])`. No edges, no message passing — just a black-box MLP mapping all neuron states and stimuli to all derivatives. Connectivity is extracted post-hoc via the **Jacobian dF/dv**.

**Important**: The MLP itself does not use graph structure — `use_gt_edges: true` is set for pipeline consistency. This experiment serves as a **control baseline** against the GNN GT-edges + noise=0.05 experiment (GNN best: 0.969). Prior MLP result on clean FC data: W R2 ≈ 0.

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Metrics (ranked by importance)

1. **rollout_pearson** (PRIMARY) — autoregressive rollout Pearson r on noise-free data
2. **connectivity_R2** (SECONDARY) — R² between Jacobian dF/dv and ground-truth W (dense matrix comparison)
3. **cluster_accuracy** (THIRD) — not applicable (no embeddings), always 0

Informational: onestep_pearson, spectral_radius_learned vs spectral_radius_true, training_time_min.

## Scientific Method

Strict **hypothesize -> test -> validate/falsify** cycle. Change **EXACTLY ONE** parameter at a time.

## Data Generation

Seeds are **forced by the pipeline** — DO NOT modify them in config files.

## CX Ring Attractor Model

```
dh/dt = alpha * (-h + exp(g_i) * softplus(h_j + b_j, beta=5) @ J^T + input) / tau_i
```

- **152 neurons**, 6 cell types, **9,722 edges** (GT topology)
- **Intrinsic noise σ=0.05** added during simulation

## MLP Architecture

```
input  = [v_1, ..., v_152, stim_1, ..., stim_152]    (304 dims)
output = [dv_1/dt, ..., dv_152/dt]                     (152 dims)
```

- Hidden layers with ReLU activation, last layer zero-initialized
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

## Parallel Mode — 4 Slots Per Batch

- **Exploration** (default): Slot 0 = parent/control. Slots 1-3 each change one parameter.
- **Robustness test**: ALL 4 slots use the SAME config (different seeds).

## Block Partition

| Block | Focus                    | Parameters to scan                          | Ranges                                                          |
| ----- | ------------------------ | ------------------------------------------- | --------------------------------------------------------------- |
| 1     | **lr + architecture**    | `lr`, `hidden_dim`, `n_layers`              | lr: {1e-4, 5e-4, 1e-3, 3e-3}, hidden_dim: {128, 256, 512}, n_layers: {3, 4, 5} |
| 2     | **Training volume**      | `data_augmentation_loop`, `n_epochs`        | DAL: {200, 500, 1000}, n_epochs: {5, 10, 20}                   |
| 3     | **Batch size**           | `batch_size`                                | batch_size: {1, 2, 4, 8}                                       |
| 4     | **Capacity vs regularization** | `hidden_dim`, weight decay via lr     | Test if larger MLPs overfit or if more capacity helps            |
| 5-8   | **Free exploration**     | Any parameter                               | Consolidate, ceiling-break, final refinement                    |

## Winner Config

Destination: `config/drosophila_cx/drosophila_cx_mlp_gt_edges_noise005_winner.yaml`

## Start Call

When prompt says `PARALLEL START`:

- Read base config — this IS the baseline. Do NOT change any default values.
- Slot 0 = baseline (no changes at all).
- Slots 1-3: each changes EXACTLY ONE parameter from the block focus.
- Hypothesis: "The MLP baseline with GT edges + noise=0.05 achieves similar W R2 ≈ 0 as clean MLP — noise does not help MLP Jacobian extraction"

---

# Working Memory Structure

```markdown
# Working Memory: drosophila_cx_mlp_gt_edges_noise005

## Paper Summary (update at every block boundary)

- **MLP baseline + GT edges + noise 0.05**: [pending]
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
