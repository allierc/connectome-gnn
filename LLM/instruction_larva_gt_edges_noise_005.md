# Drosophila Larva GT Edges (Noise 0.05) — LLM Exploration

## Goal

Maximize **connectivity_R2** for the **Drosophila larva two-population motor model** (Beiran & Litwin-Kumar 2023, Figure 5a-c) using **ground-truth edges** under **intrinsic noise (sigma=0.05)**.

This exploration tests whether GT edges with mild noise improves upon clean GT baseline (best conn_R2=0.908). GT edges (4,222) provide strong structural constraints vs FC (52,670), and noise=0.05 has shown cross-model benefits. The question is whether this combination achieves higher robustness than clean GT.

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Parent config (best GT edges noise-free)

From prior GT edges exploration.

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned effective W and ground-truth effective W
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free data
3. **cluster_accuracy** (THIRD) — neuron type clustering accuracy from learned embeddings

Informational (not for optimization): onestep_pearson, f_theta_R2, g_phi_R2, tau_R2.

**NOTE**: V_rest_R2 is not applicable (no resting potential). tau_R2 is 0.0 (fixed tau=1.0).

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

At this mild noise level with GT edges, signal-to-noise ratio should be favorable for W recovery. This regime tests whether GT edges' advantage persists under noise.

## Larva Two-Population Motor Model

### Premotor neurons (N=178):

```
dup/dt = (-up + gp * softplus(up @ Jpp) + bp + wsp @ stim) / taup
```

### Motor neurons (M=52):

```
dum/dt = (-um + gm * softplus(up @ Jpm) + bm) / taum
```

- **230 neurons** total (178 premotor + 52 motor), **2 cell types**, **4,222 edges** (GT)
- Activation: **Softplus** (log(1 + exp(x)))
- Gains gp, gm clamped to [0.5, 5.0]
- taup, taum ~ 1.0, dt=0.05
- 2 stimulus conditions (forward/backward), 2 stimulus channels
- Inhibitory neurons get negative weights (Dale's law in connectome)
- 2,400 frames, delta_t=0.05, **noise_model_level=0.05**
- Feedforward: premotor->motor only, plus premotor recurrence
- Only 2 neuron types -> embedding should separate 2 clusters

## GNN Architecture

- **g_phi**: Edge message MLP. Maps (v_j, a_j) -> message. `g_phi_positive=true`.
- **f_theta**: Node update MLP. Maps (v_i, a_i, aggregated_msg, I_i) -> dv_i/dt.
- **Embedding a_i**: learnable per-neuron type vector.

**CRITICAL — coupled parameters**: `embedding_dim` must be >= 2 (embedding_dim=1 crashes plotting). When changing `embedding_dim`, you MUST also update:

- `input_size = 1 + embedding_dim`
- `input_size_update = 3 + embedding_dim`

Example: embedding_dim=2 -> input_size=3, input_size_update=5.

## Training Parameters

| Parameter                 | Default | Description                                            |
| ------------------------- | ------- | ------------------------------------------------------ |
| `lr_W`                    | 1e-3    | Learning rate for connectivity W                       |
| `lr`                      | 1e-3    | Learning rate for g_phi and f_theta MLPs               |
| `lr_embedding`            | 1e-3    | Learning rate for neuron embeddings                    |
| `n_epochs`                | 2       | Number of training epochs                              |
| `batch_size`              | 2       | Batch size                                             |
| `data_augmentation_loop`  | 100     | Data augmentation multiplier                           |
| `w_init_mode`             | zeros   | W initialization: "zeros", "randn_scaled"              |
| `coeff_g_phi_diff`        | 1500    | Monotonicity penalty on g_phi                          |
| `coeff_f_theta_weight_L2` | 0.001   | L2 penalty on f_theta MLP weights                      |
| `coeff_f_theta_diff`      | 0       | Negative monotonicity of f_theta w.r.t. state v_i      |
| `coeff_f_theta_msg_diff`  | 0       | Positive monotonicity of f_theta w.r.t. message input  |
| `coeff_W_L1`              | 0       | L1 sparsity on W                                       |
| `coeff_W_L2`              | 1e-5    | L2 penalty on W                                        |
| `coeff_W_sign`            | 0       | Dale's law penalty                                     |
| `use_gt_edges`            | true    | **FIXED** — ground-truth edges for this experiment     |
| `dale_law`                | false   | Enforce Dale's law                                     |
| `noise_model_level`       | 0.05    | **FIXED** — intrinsic noise level for this experiment  |


## Parallel Mode — 4 Slots Per Batch

Each batch runs 4 slots with different seeds (forced by pipeline). You choose the strategy:

- **Exploration** (default): Slot 0 = parent/control (unchanged). Slots 1-3 each change **exactly one** parameter. This gives 3 causal tests per batch.
- **Robustness test**: ALL 4 slots use the SAME config. The pipeline forces different seeds, so this measures seed robustness. Use this when a config looks promising.

State your choice (exploration vs robustness test) in the log entry.

### Robustness Assessment (when running same config across 4 slots)

- **Robust**: all 4 slots connectivity_R2 > 0.7
- **Partially robust**: 2-3 slots > 0.7
- **Fragile**: 0-1 slots > 0.7

## Block Structure

| Block | Focus                          | Parameters to scan                                                         | Ranges                                                                                                           |
| ----- | ------------------------------ | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1     | **Baseline validation**        | None (robustness test)                                                     | Run best GT edges noise-free config with noise=0.05 across 4 seeds. Does noise improve robustness? |
| 2     | **Regularization re-tune**     | `coeff_W_L1`, `coeff_W_L2`, `coeff_W_sign`                                | W_L1: {0, 1e-6, 5e-6}, W_L2: {5e-6, 1e-5, 2e-5}, W_sign: {0, 0.05, 0.2}. Noise may require different regularization. |
| 3     | **Training volume re-tune**    | `data_augmentation_loop`, `n_epochs`                                       | DAL: {50, 100, 200}, n_epochs: {2, 4}. Noise may require more training to average out. |
| 4     | **Architecture + batch_size**  | `hidden_dim`, `embedding_dim`, `batch_size`                                | hidden_dim: {48, 64, 80}, embedding_dim: {2, 4}, batch_size: {2, 4}. Noisy data may need different capacity. |
| 5     | **Monotonicity + Dale's law**  | `coeff_g_phi_diff`, `coeff_f_theta_diff`, `coeff_f_theta_msg_diff`, `dale_law` | g_phi_diff: {1000, 1500, 2000}, f_theta_diff: {0, 10}, dale_law: {false, true}. Noise may corrupt learned nonlinearities. |
| 6     | **Free exploration I**         | Any parameter                                                              | Consolidate best from blocks 1-5, test novel combinations                                                        |
| 7     | **Free exploration II**        | Any parameter                                                              | Continue ceiling-breaking attempts                                                                               |
| 8     | **Final robustness**           | None (robustness test)                                                     | 4-seed robustness test of best config from blocks 1-7                                                            |

### Noise-specific considerations

- **Cross-model evidence supports noise helping**: CX FC noise-free=0.804 -> noise_005=0.982. Larva GT edges should benefit similarly.
- **GT edges constraint is strong**: With only 4,222 edges (vs FC 52,670), the search space is already constrained. Noise may provide consistent marginal improvements without fundamental phase changes.

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
3. Save it to `config/larva/larva_gt_edges_noise_005_winner.yaml` with a YAML comment header:

```yaml
# Winner config: larva_gt_edges_noise_005_winner.yaml
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
#
# Key config differences from baseline:
#   - [list the parameters that differ from the initial baseline]
```

Destination: `config/larva/larva_gt_edges_noise_005_winner.yaml`

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

1. **Full Log (append-only)**: `larva_gt_edges_noise_005_Claude_analysis.md`
   - Append every iteration's log entry (4 entries per batch)
   - Never read — human record only

2. **Working Memory (read + update every batch)**: `larva_gt_edges_noise_005_Claude_memory.md`
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

### What to Add to Open Questions

- Patterns observed 1-2 times
- Seed-dependent effects (works for some seeds but not others)
- Contradictions between iterations
- Theoretical predictions not yet verified

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:
- State the original hypothesis
- State the contradicting evidence (iteration number, metrics)
- State what was learned from the falsification
- Propose a revised hypothesis if applicable

## Start Call

When prompt says `PARALLEL START`:

- Read base config — the parent GT edges noise-free config + noise_model_level=0.05 IS the baseline.
- Block 1 is a **robustness test**: all 4 slots use the same config (different seeds).
- Hypothesis: "Adding noise=0.05 to the best GT edges config improves or maintains connectivity_R2 robustness"

---

# Working Memory Structure

```markdown
# Working Memory: larva_gt_edges_noise_005

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean+-std) | CV% | rollout_r | cluster_acc | Robust? | Hypothesis |
| ---- | -------------- | ------------------- | --- | --------- | ----------- | ---------- |

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
