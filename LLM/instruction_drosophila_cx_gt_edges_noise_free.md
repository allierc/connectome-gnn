# Drosophila CX (GT Edges) — LLM Exploration

## Goal

Maximize **connectivity_R2** for the **Drosophila central complex ring attractor** (Beiran & Litwin-Kumar 2023, Figure 5) using **ground-truth edges** instead of fully connected.

This exploration tests whether known topology improves W recovery. From zebrafish experience, GT edges dramatically improved connectivity recovery (0.022 FC -> 0.777 GT). CX has 152 neurons with 9,722 true edges vs 22,952 FC edges. GT edges reduce the search space by ~58%, which should help W recovery.

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Parent config (best clean FC, from 128-iter exploration)

```
lr_W: 5e-5
lr: 1e-3
lr_embedding: 1e-3
n_epochs: 2
data_augmentation_loop: 200
w_init_mode: zeros
coeff_W_L1: 3e-6
coeff_W_L2: 1e-5
coeff_W_sign: 0.01
coeff_g_phi_diff: 1500
coeff_f_theta_weight_L2: 0.001
use_gt_edges: true
noise_model_level: 0.0
hidden_dim: 64
embedding_dim: 2
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

**IMPORTANT**: `use_gt_edges` is set to **true** in the base config. Do NOT change it — this file is specifically for the GT edges experiment.

## CX Ring Attractor Model

```
dh/dt = alpha * (-h + exp(g_i) * softplus(h_j + b_j, beta=5) @ J^T + input) / tau_i
```

- **152 neurons**, 6 cell types (EPG, EPGt, PEN_a, PEN_b, Delta7, PEG), **9,722 edges** (GT)
- tau bounded [0.2, 5.0], alpha=0.2, beta=5 (softplus sharpness)
- Pretrained teacher weights from hemibrain connectivity
- 10,000 frames, delta_t=0.1, bump + velocity stimuli, **noise_model_level=0.0** (clean)
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
| `lr_W`                    | 5e-5    | Learning rate for connectivity W                       |
| `lr`                      | 1e-3    | Learning rate for g_phi and f_theta MLPs               |
| `lr_embedding`            | 1e-3    | Learning rate for neuron embeddings                    |
| `n_epochs`                | 2       | Number of training epochs                              |
| `batch_size`              | 2       | Batch size                                             |
| `data_augmentation_loop`  | 200     | Data augmentation multiplier                           |
| `w_init_mode`             | zeros   | W initialization: "zeros", "randn_scaled"              |
| `coeff_g_phi_diff`        | 1500    | Monotonicity penalty on g_phi                          |
| `coeff_f_theta_weight_L2` | 0.001   | L2 penalty on f_theta MLP weights                      |
| `coeff_f_theta_diff`      | 0       | Negative monotonicity of f_theta w.r.t. state v_i      |
| `coeff_f_theta_msg_diff`  | 0       | Positive monotonicity of f_theta w.r.t. message input  |
| `coeff_W_L1`              | 3e-6    | L1 sparsity on W                                       |
| `coeff_W_L2`              | 1e-5    | L2 penalty on W                                        |
| `coeff_W_sign`            | 0.01    | Dale's law penalty                                     |
| `use_gt_edges`            | true    | **FIXED** — ground-truth edges for this experiment     |
| `dale_law`                | false   | Enforce Dale's law                                     |
| `hidden_dim`              | 64      | Hidden dimension of MLPs                               |
| `embedding_dim`           | 2       | Embedding dimension per neuron type                    |


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

These blocks focus on whether GT edges change the optimal hyperparameters compared to the FC baseline. The reduced search space (9,722 vs 22,952 edges) may shift optimal regularization and learning rates.

| Block | Focus                          | Parameters to scan                                                         | Ranges                                                                                                           |
| ----- | ------------------------------ | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1     | **lr_W re-tune for GT edges**  | `lr_W`                                                                     | lr_W: {1e-5, 5e-5, 1e-4, 3e-4}. GT edges change the W gradient landscape — optimal lr_W may shift.             |
| 2     | **Regularization re-tune**     | `coeff_W_L1`, `coeff_W_L2`                                                | W_L1: {1e-6, 3e-6, 1e-5}, W_L2: {5e-6, 1e-5, 3e-5}. With fewer edges, sparsity pressure may need adjustment.   |
| 3     | **W_sign + Dale's law**        | `coeff_W_sign`, `dale_law`                                                | W_sign: {0, 0.01, 0.05, 0.1}, dale_law: {false, true}. Test whether sign constraints still help with GT edges.  |
| 4     | **Architecture**               | `hidden_dim`, `embedding_dim`, `batch_size`                                | hidden_dim: {48, 64, 80}, embedding_dim: {2, 4}, batch_size: {2, 4}. GT edges may allow smaller or larger models.|
| 5     | **Free exploration I**         | Any parameter                                                              | Consolidate best from blocks 1-4, test novel combinations                                                        |
| 6     | **Free exploration II**        | Any parameter                                                              | Continue ceiling-breaking attempts                                                                               |
| 7     | **Free exploration III**       | Any parameter                                                              | Further refinement based on accumulated evidence                                                                 |
| 8     | **Final robustness**           | None (robustness test)                                                     | 4-seed robustness test of best config from blocks 1-7                                                            |

### GT-edges-specific considerations

- **GT edges dramatically helped zebrafish**: connectivity_R2 went from 0.022 (FC) to 0.777 (GT). CX may see a similar boost — the FC baseline is 0.804 best / 0.574 mean.
- **lr_W may need re-tuning**: With ~58% fewer edges, gradients on each W_ij are less diluted. The optimal lr_W may be lower or higher than the FC optimum.
- **L1 sparsity may be less important**: GT edges are already sparse — L1 pressure may be counterproductive or need reduction.
- **W_sign/Dale's law may interact differently**: With only true edges, sign constraints apply to meaningful connections only, which may strengthen their benefit.

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
3. Save it to `config/drosophila_cx/drosophila_cx_gt_edges_winner.yaml` with a YAML comment header:

```yaml
# Winner config: drosophila_cx_gt_edges_winner.yaml
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

Destination: `config/drosophila_cx/drosophila_cx_gt_edges_winner.yaml`

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

1. **Full Log (append-only)**: `drosophila_cx_gt_edges_Claude_analysis.md`
   - Append every iteration's log entry (4 entries per batch)
   - Never read — human record only

2. **Working Memory (read + update every batch)**: `drosophila_cx_gt_edges_Claude_memory.md`
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

Example: "lr_W=5e-5 with GT edges achieves connectivity_R2 > 0.7 robustly (3/3 iterations, all seeds > 0.65, CV < 5%)"

### What to Add to Open Questions

- Patterns observed 1-2 times
- Seed-dependent effects (works for some seeds but not others)
- Contradictions between iterations
- Theoretical predictions not yet verified

Example: "Does lower W_L2 help GT edges further? Only iter 1 tested with mixed results."

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:
- State the original hypothesis
- State the contradicting evidence (iteration number, metrics)
- State what was learned from the falsification
- Propose a revised hypothesis if applicable

Example: "Hypothesis: 'GT edges + aggressive lr_W (1e-4) improve over baseline' — Falsified by iter 2 (CV=9%, only 1/4 seeds > 0.7). Revised: 'GT edges require careful lr tuning; aggressive LR increases variance.'"

## Start Call

When prompt says `PARALLEL START`:

- Read base config — the parent clean FC config + use_gt_edges=true IS the baseline.
- Block 1 is a **robustness test**: all 4 slots use the same config (different seeds).
- Hypothesis: "GT edges improve connectivity_R2 above the FC baseline (0.804 best, 0.574 mean) by constraining the W search space"

---

# Working Memory Structure

```markdown
# Working Memory: drosophila_cx_gt_edges

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
