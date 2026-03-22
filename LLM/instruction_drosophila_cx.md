# Connconstr Drosophila CX — LLM Exploration

## Goal

Maximize **connectivity_R2** for the **Drosophila central complex ring attractor** (Beiran & Litwin-Kumar 2023, Figure 5).

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned effective W and ground-truth effective W
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free data
3. **cluster_accuracy** (THIRD) — neuron type clustering accuracy from learned embeddings

Informational (not for optimization): onestep_pearson, f_theta_R2, g_phi_R2, spectral_radius_learned vs spectral_radius_true.

**NOTE**: V_rest_R2 is always 0.0 (no resting potential). tau_R2 is unreliable (slope ~ -0.05, noise-amplified).

## Scientific Method

Strict **hypothesize -> test -> validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction
2. **Design experiment**: Change ONE parameter at a time
3. **Run training**: 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use metrics AND cross-seed variance
5. **Update understanding**: Revise hypotheses based on evidence

**CRITICAL**: You can only hypothesize. Only training results validate or falsify.

## Data Generation

Each slot re-generates data with a **different random seed**.
Seeds are **forced by the pipeline** — DO NOT modify them in config files.

- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

**DO NOT change `simulation:` parameters** except `noise_model_level` (Block 6) and seed (managed automatically).

## CX Ring Attractor Model

```
dh/dt = alpha * (-h + exp(g_i) * softplus(h_j + b_j, beta=5) @ J^T + input) / tau_i
```

- **152 neurons**, 6 cell types (EPG, EPGt, PEN_a, PEN_b, Delta7, PEG), **9,722 edges**
- tau bounded [0.2, 5.0], alpha=0.2, beta=5 (softplus sharpness)
- Pretrained teacher weights from hemibrain connectivity
- 10,000 frames, delta_t=0.1, bump + velocity stimuli
- Softplus activation -> g_phi should learn softplus-like curves
- No V_rest -> f_theta learns pure decay slope ~ -alpha/tau_i
- 6 cell types -> embedding should separate 6 clusters
- Delta7 = inhibitory (negative W), others = excitatory (positive W)

## GNN Architecture

- **g_phi**: Edge message MLP. Maps (v_j, a_j) -> message. `g_phi_positive=true`.
- **f_theta**: Node update MLP. Maps (v_i, a_i, aggregated_msg, I_i) -> dv_i/dt.
- **Embedding a_i**: learnable per-neuron type vector.

**CRITICAL — coupled parameters**: When changing `embedding_dim`, you MUST also update:
- `input_size = 1 + embedding_dim`
- `input_size_update = 3 + embedding_dim`

Example: embedding_dim=4 -> input_size=5, input_size_update=7.

## Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `lr_W` | 3e-4 | Learning rate for connectivity W |
| `lr` | 1e-3 | Learning rate for g_phi and f_theta MLPs |
| `lr_embedding` | 1e-3 | Learning rate for neuron embeddings |
| `n_epochs` | 2 | Number of training epochs |
| `batch_size` | 2 | Batch size |
| `data_augmentation_loop` | 100 | Data augmentation multiplier |
| `w_init_mode` | zeros | W initialization: "zeros", "randn_scaled" |
| `coeff_g_phi_diff` | 1500 | Monotonicity penalty on g_phi |
| `coeff_f_theta_weight_L2` | 0.001 | L2 penalty on f_theta MLP weights |
| `coeff_f_theta_msg_diff` | 0 | Monotonicity of f_theta w.r.t. message input |
| `coeff_W_L1` | 0 | L1 sparsity on W |
| `coeff_W_L2` | 1e-5 | L2 penalty on W |
| `coeff_W_sign` | 0 | Dale's law penalty |
| `use_gt_edges` | true | If false, train on fully connected graph (N^2-N edges) |
| `noise_model_level` | 0.0 | Observation noise std added to trajectories |

## Training Time Constraint

**Keep total training time <= 60 min per iteration.** This is a small model (152 neurons). Typical: ~15 min/epoch. When increasing n_epochs, halve data_augmentation_loop to stay within budget.

## Parallel Mode — 4 Slots Per Batch

All 4 slots run the **same config** (different seeds applied automatically).
Edit all 4 configs identically: `{name}_00.yaml` through `{name}_03.yaml`.

### Robustness Assessment

- **Robust**: all 4 slots connectivity_R2 > 0.7
- **Partially robust**: 2-3 slots > 0.7
- **Fragile**: 0-1 slots > 0.7

Compute mean, std, CV for connectivity_R2 across 4 slots every batch.

## Block Partition

The blocks below provide a **recommended exploration roadmap**. Follow the block focus as a guide but use your scientific judgment — if early results clearly suggest a detour or shortcut, adapt. The block boundaries are soft: you can revisit earlier axes or combine parameters across blocks when evidence supports it.

| Block | Focus | Parameters to scan | Ranges |
|---|---|---|---|
| 1 | **lr_W + W_L2** | `lr_W`, `coeff_W_L2` | lr_W: {1e-4, 3e-4, 6e-4, 1e-3}, W_L2: {5e-6, 1e-5, 2e-5} |
| 2 | **Training volume** | `data_augmentation_loop`, `n_epochs` | DAL: {50, 100, 200}, n_epochs: {2, 4} (halve DAL when doubling epochs) |
| 3 | **Regularization** | `coeff_W_sign`, `coeff_W_L1`, `coeff_g_phi_diff`, `coeff_f_theta_msg_diff` | W_sign: {0, 0.01, 0.1}, W_L1: {0, 5e-5, 1e-4}, g_phi_diff: {500, 1000, 1500}, f_theta_msg_diff: {0, 10, 100} |
| 4 | **Architecture** | `hidden_dim`, `embedding_dim` | hidden_dim: {48, 64, 80}, embedding_dim: {2, 4} (update input_size accordingly) |
| 5 | **Combined best + edge discovery** | Best from 1-4, `use_gt_edges`, `coeff_W_L1` | use_gt_edges: {true, false}, W_L1: {0, 5e-5, 1e-4} when fully connected |
| 6 | **Noise robustness** | Best from 1-4, `noise_model_level` | noise_model_level: {0, 0.05, 0.5} |
| 7 | **Free exploration I** | Any parameter | Consolidate best from blocks 1-6, test novel combinations, attempt to break R2 ceiling |
| 8 | **Free exploration II** | Any parameter | Continue ceiling-breaking attempts, confirm final robust config |

### What NOT to explore (established from prior 124-iteration CX exploration)

These axes were thoroughly tested and found harmful or suboptimal:
- `batch_size` other than 2 (2 is optimal; 1 too noisy, 4 too few steps)
- `w_init_mode=randn` or `randn_scaled` (zeros is best)
- `lr_scheduler` (all schedules hurt W recovery)
- `regul_annealing_rate` (unnecessary; rate=0 matches annealed)
- `n_epochs > 4` (3rd+ epoch degrades W)
- `coeff_g_phi_weight_L2` (catastrophic — causes loss divergence)
- `n_layers > 3` or `hidden_dim > 80` (fragile or catastrophic)
- `derivative_smoothing_window > 1` (destroys signal)

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

From `analysis.log`: connectivity_R2, rollout_pearson, cluster_accuracy, training_time_min.

### Step 3: Write Log Entries + Update Memory

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, DAL=D, n_epochs=E, W_L2=A, hidden_dim=H, batch_size=B
Slot 0: conn_R2=A, rollout_pearson=B, cluster_acc=C, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, rollout_pearson=B, cluster_acc=C, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, rollout_pearson=B, cluster_acc=C, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, rollout_pearson=B, cluster_acc=C, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive]
Next: parent=P
```

### Step 4: Acknowledge User Input

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Block Boundaries

1. Update "Paper Summary"
2. Summarize block findings
3. Update "Established Principles"
4. Clear "Current Block"
5. Carry forward best config

## Start Call

When prompt says `PARALLEL START`:
- Read base config
- Set all 4 configs identically to baseline
- First iteration = baseline (no changes)
- Hypothesis: "The baseline config achieves connectivity_R2 > 0.5 robustly across seeds"

---

# Working Memory Structure

```markdown
# Working Memory: drosophila_cx

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean+-std) | CV% | rollout_r | cluster_acc | Robust? | Hypothesis |
| ---- | -------------- | ------------------- | --- | --------- | ----------- | ------- | ---------- |

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
