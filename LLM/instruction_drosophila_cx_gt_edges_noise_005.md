# Drosophila CX GT Edges (Noise 0.05) — LLM Exploration

## Goal

Maximize **connectivity_R2** for the **Drosophila central complex ring attractor** (Beiran & Litwin-Kumar 2023, Figure 5) using **ground-truth edges** under **intrinsic noise (sigma=0.05)**.

This exploration combines two independently validated improvements: GT edges (which reduced the search space by ~58%, achieving 0.893 best / 0.710 mean noise-free) and noise=0.05 (which improved FC from 0.804→0.982). The question is whether GT edges + noise=0.05 exceeds the FC+noise=0.05 result (0.982), since GT edges remove 58% spurious edges that noise must disambiguate. The parent config (dale_law=true, g_phi_wL1=0.003) was optimal for noise-free GT — noise may change the optimal regularization balance.

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Parent config (best GT edges noise-free, from 104-iter exploration)

```
lr_W: 2e-5
lr: 7e-4
lr_embedding: 1e-3
n_epochs: 2
data_augmentation_loop: 105
w_init_mode: zeros
hidden_dim: 96
embedding_dim: 2
coeff_g_phi_diff: 1500
coeff_g_phi_weight_L1: 0.003
coeff_f_theta_weight_L2: 0.001
coeff_f_theta_msg_diff: 0
coeff_W_L1: 3e-6
coeff_W_L2: 1e-5
coeff_W_sign: 0.01
dale_law: true
use_gt_edges: true
batch_size: 2
noise_model_level: 0.05
```

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned effective W and ground-truth effective W
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free data
3. **cluster_accuracy** (THIRD) — neuron type clustering accuracy from learned embeddings

Informational (not for optimization): onestep_pearson, f_theta_R2, g_phi_R2, spectral_radius_learned vs spectral_radius_true.

**NOTE**: V_rest_R2 is always 0.0 (no resting potential). tau_R2 is unreliable (slope ~ -0.05, noise-amplified).

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
- Do NOT change parameters outside the current block focus.
- Do NOT skip the baseline — always keep one slot as an unchanged control.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness).

## Data Generation

Each slot re-generates data with a **different random seed**.
Seeds are **forced by the pipeline** — DO NOT modify them in config files.

- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

**DO NOT change `simulation:` parameters** except seed (managed automatically).

**IMPORTANT**: `noise_model_level` is set to **0.05** in the base config. Do NOT change it — this file is specifically for the noise=0.05 experiment.

**IMPORTANT**: `use_gt_edges` is set to **true** in the base config. Do NOT change it — this file is specifically for the GT edges experiment.

## Noise Model

Two independent noise sources in the training data:

1. **Dynamics noise** (`noise_model_level=0.05`): `v(t+1) = v(t) + dt * f(v, W, I) + epsilon_dyn(t)`, epsilon_dyn ~ N(0, 0.05)
2. **Measurement noise** (`measurement_noise_level=0.0`): Clean observations

At this mild noise level with GT edges (reduced search space), signal-to-noise ratio should be favorable for W recovery. This regime tests whether GT edges' advantage persists under noise.

## CX Ring Attractor Model

```
dh/dt = alpha * (-h + exp(g_i) * softplus(h_j + b_j, beta=5) @ J^T + input) / tau_i
```

- **152 neurons**, 6 cell types (EPG, EPGt, PEN_a, PEN_b, Delta7, PEG), **9,722 edges** (GT)
- tau bounded [0.2, 5.0], alpha=0.2, beta=5 (softplus sharpness)
- Pretrained teacher weights from hemibrain connectivity
- 10,000 frames, delta_t=0.1, bump + velocity stimuli, **noise_model_level=0.05**
- Softplus activation -> g_phi should learn softplus-like curves
- No V_rest -> f_theta learns pure decay slope ~ -alpha/tau_i
- 6 cell types -> embedding should separate 6 clusters
- Delta7 = inhibitory (negative W), others = excitatory (positive W)

## GNN Architecture

- **g_phi**: Edge message MLP. Maps (v_j, a_j) -> message. `g_phi_positive=true`.
- **f_theta**: Node update MLP. Maps (v_i, a_i, aggregated_msg, I_i) -> dv_i/dt.
- **Embedding a_i**: learnable per-neuron type vector.

**CRITICAL — coupled parameters**: `embedding_dim` must be >= 2 (embedding_dim=1 crashes plotting). When changing `embedding_dim`, you MUST also update:

- `input_size = 1 + embedding_dim`
- `input_size_update = 3 + embedding_dim`

Example: embedding_dim=4 -> input_size=5, input_size_update=7.

## Training Parameters

| Parameter                 | Default | Description                                            |
| ------------------------- | ------- | ------------------------------------------------------ |
| `lr_W`                    | 2e-5    | Learning rate for connectivity W                       |
| `lr`                      | 7e-4    | Learning rate for g_phi and f_theta MLPs               |
| `lr_embedding`            | 1e-3    | Learning rate for neuron embeddings                    |
| `n_epochs`                | 2       | Number of training epochs                              |
| `batch_size`              | 2       | Batch size                                             |
| `data_augmentation_loop`  | 105     | Data augmentation multiplier                           |
| `w_init_mode`             | zeros   | W initialization: "zeros", "randn_scaled"              |
| `hidden_dim`              | 96      | MLP hidden dimension (from GT edges winner)            |
| `embedding_dim`           | 2       | Embedding dimension (from GT edges winner)             |
| `coeff_g_phi_diff`        | 1500    | Monotonicity penalty on g_phi                          |
| `coeff_g_phi_weight_L1`   | 0.003   | L1 penalty on g_phi weights (from GT edges winner)     |
| `coeff_f_theta_weight_L2` | 0.001   | L2 penalty on f_theta MLP weights                      |
| `coeff_f_theta_diff`      | 0       | Negative monotonicity of f_theta w.r.t. state v_i      |
| `coeff_f_theta_msg_diff`  | 0       | Positive monotonicity of f_theta w.r.t. message input  |
| `coeff_W_L1`              | 3e-6    | L1 sparsity on W                                       |
| `coeff_W_L2`              | 1e-5    | L2 penalty on W                                        |
| `coeff_W_sign`            | 0.01    | Dale's law penalty                                     |
| `use_gt_edges`            | true    | **FIXED** — ground-truth edges for this experiment     |
| `dale_law`                | true    | **Enforce Dale's law** (from GT edges winner — critical stabilizer, +34% improvement) |
| `noise_model_level`       | 0.05    | **FIXED** — intrinsic noise level for this experiment  |


> **YAML rule**: Always wrap the `description` field value in double quotes — colons inside unquoted YAML strings cause parse errors (e.g., `description: "Block 7 Slot 1: testing W_L2"`).

## Parallel Mode — 4 Slots Per Batch

Each batch runs 4 slots with different seeds (forced by pipeline). You choose the strategy:

- **Exploration** (default): Slot 0 = parent/control (unchanged). Slots 1-3 each change **exactly one** parameter. This gives 3 causal tests per batch.
- **Robustness test**: ALL 4 slots use the SAME config. The pipeline forces different seeds, so this measures seed robustness. Use this when a config looks promising.

State your choice (exploration vs robustness test) in the log entry.

### Robustness Assessment (when running same config across 4 slots)

- **Robust**: all 4 slots connectivity_R2 > 0.8
- **Partially robust**: 2-3 slots > 0.8
- **Fragile**: 0-1 slots > 0.8

## Block Structure

These blocks start from the best GT edges noise-free config (dale_law=true, lr_W=2e-5, g_phi_wL1=0.003, hidden_dim=96, embedding_dim=2). The focus is on whether adding noise=0.05 changes the optimal configuration that was tuned for clean GT edges, and whether the GT+noise combination exceeds the FC+noise result (0.982).

| Block | Focus                          | Parameters to scan                                                         | Ranges                                                                                                           |
| ----- | ------------------------------ | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1     | **Baseline validation**        | None (robustness test)                                                     | Run best GT edges noise-free config with noise=0.05 across 4 seeds. Does noise improve above the noise-free baseline (0.893 best, 0.710 mean)? |
| 2     | **Regularization re-tune**     | `coeff_W_L1`, `coeff_W_L2`, `coeff_g_phi_weight_L1`                       | W_L1: {1e-6, 3e-6, 1e-5}, W_L2: {5e-6, 1e-5, 3e-5}, g_phi_wL1: {0.001, 0.003, 0.01}. Noise may shift the regularization balance — g_phi_wL1=0.003 was key for noise-free GT. |
| 3     | **Training volume re-tune**    | `data_augmentation_loop`, `n_epochs`, `batch_size`                         | DAL: {70, 105, 200}, n_epochs: {2, 4}, batch_size: {2, 4}. Noise adds gradient variance — more averaging may help. |
| 4     | **Architecture**               | `hidden_dim`, `embedding_dim`                                              | hidden_dim: {64, 96, 128}, embedding_dim: {2, 4}. GT edges winner used hidden_dim=96, embedding_dim=2 — noise may need more capacity. |
| 5     | **Monotonicity + f_theta**     | `coeff_g_phi_diff`, `coeff_f_theta_diff`, `coeff_f_theta_msg_diff`         | g_phi_diff: {500, 1000, 1500, 2000}, f_theta_diff: {0, 10, 50}, f_theta_msg_diff: {0, 25, 50, 100}. Noise may corrupt learned nonlinearities — stronger constraints may help. |
| 6     | **Free exploration I**         | Any parameter                                                              | Consolidate best from blocks 1-5, test novel combinations                                                        |
| 7     | **Free exploration II**        | Any parameter                                                              | Continue ceiling-breaking attempts                                                                               |
| 8     | **Final robustness**           | None (robustness test)                                                     | 4-seed robustness test of best config from blocks 1-7                                                            |

### Noise-specific considerations

- **Cross-model evidence strongly supports noise helping**: CX FC noise-free=0.804 -> noise_005=0.982 (+22%). Flyvis noise-free=0.926 -> noise_005=0.985. All models show monotonic improvement with noise.
- **GT edges + noise is a novel combination**: GT edges already achieved 0.893 best (vs FC 0.804), and noise already achieved 0.982 best (FC). The combination should be additive or synergistic — GT edges constrain the topology while noise enriches the state-space.
- **dale_law=true is established**: The GT edges exploration proved this is the single biggest lever (+34%). Keep it unless evidence clearly says otherwise.
- **g_phi_weight_L1=0.003 is established**: Key regularizer for GT edges. Noise may shift the optimal value.
- **~20% catastrophic failure rate in noise-free GT**: The noise-free GT exploration showed ~80% convergence with ~20% catastrophic failures. Noise may help by smoothing the loss landscape and reducing these failures.
- **lr_W=2e-5 is very low**: The GT edges winner used 10x lower lr_W than FC configs. Noise may allow slightly higher lr_W since noisy gradients provide implicit regularization.

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

From `analysis.log`: connectivity_R2, rollout_pearson, cluster_accuracy, training_time_min.

### Step 3: Write Log Entries + Update Memory

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, DAL=D, n_epochs=E, W_L1=A, W_L2=B, hidden_dim=H, batch_size=B
Slot 0: conn_R2=A, rollout_pearson=B, cluster_acc=C, dale_score=D, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, rollout_pearson=B, cluster_acc=C, dale_score=D, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, rollout_pearson=B, cluster_acc=C, dale_score=D, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, rollout_pearson=B, cluster_acc=C, dale_score=D, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
W matrix: [visual comment from connectivity heatmap]
Verdict: [supported/falsified/inconclusive]
Next: parent=P
```

## Winner Config (COMPULSORY)

**At every block boundary**, you MUST save the current best config as a winner file.
This is a COMPULSORY task — do not skip it.

1. Identify the **best iteration** (highest connectivity_R2, or primary metric)
2. Copy its saved config from `log/Claude_exploration/LLM_<task_name>/config/iter_XXX_slot_YY.yaml`
3. Save it to `config/drosophila_cx/drosophila_cx_gt_edges_noise_005_winner.yaml` with a YAML comment header:

```yaml
# Winner config: drosophila_cx_gt_edges_noise_005_winner.yaml
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

Destination: `config/drosophila_cx/drosophila_cx_gt_edges_noise_005_winner.yaml`

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

1. **Full Log (append-only)**: `drosophila_cx_gt_edges_noise_005_Claude_analysis.md`
   - Append every iteration's log entry (4 entries per batch)
   - Never read — human record only

2. **Working Memory (read + update every batch)**: `drosophila_cx_gt_edges_noise_005_Claude_memory.md`
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

Example: "lr_W=2e-5 with GT edges + noise=0.05 achieves connectivity_R2 > 0.75 robustly (3/3 iterations, all seeds > 0.70, CV < 5%)"

### What to Add to Open Questions

- Patterns observed 1-2 times
- Seed-dependent effects (works for some seeds but not others)
- Contradictions between iterations
- Theoretical predictions not yet verified

Example: "Does g_phi_wL1 require adjustment at noise=0.05? Only iter 1 tested."

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:
- State the original hypothesis
- State the contradicting evidence (iteration number, metrics)
- State what was learned from the falsification
- Propose a revised hypothesis if applicable

Example: "Hypothesis: 'Noise=0.05 improves GT edges as much as FC' — Falsified by iter 2 (GT+noise CV=8%, worse than FC+noise CV=4%). Revised: 'GT edges benefit from noise less than FC due to reduced search space.'"

## Start Call

When prompt says `PARALLEL START`:

- Read base config — the parent GT edges noise-free config + noise_model_level=0.05 IS the baseline.
- Block 1 is a **robustness test**: all 4 slots use the same config (different seeds).
- Hypothesis: "Adding noise=0.05 to the best GT edges config improves connectivity_R2 above the noise-free baseline (0.893 best, 0.710 mean), consistent with cross-model evidence"

---

# Working Memory Structure

```markdown
# Working Memory: drosophila_cx_gt_edges_noise_005

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean+-std) | CV% | rollout_r | cluster_acc | dale_score | Robust? | Hypothesis |
| ---- | -------------- | ------------------- | --- | --------- | ----------- | ------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

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
