# Drosophila Larva — LLM Exploration

## Goal

Maximize **connectivity_R2** for the **Drosophila larva two-population motor model** (Beiran & Litwin-Kumar 2023, Figure 5a-c).

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned effective W and ground-truth effective W
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free data
3. **cluster_accuracy** (THIRD) — neuron type clustering accuracy from learned embeddings

Informational (not for optimization): onestep_pearson, f_theta_R2, g_phi_R2, tau_R2.

**NOTE**: V_rest_R2 is not applicable (no resting potential).

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
- Do NOT change parameters outside the current block focus (e.g. do not touch w_init_mode, W_L2, batch_size, hidden_dim unless the block explicitly includes them).
- Do NOT skip the baseline — always keep one slot as an unchanged control.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness).

## Data Generation

Each slot re-generates data with a **different random seed**.
Seeds are **forced by the pipeline** — DO NOT modify them in config files.

- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

**DO NOT change `simulation:` parameters** except `noise_model_level` (Block 6) and seed (managed automatically).

## Larva Two-Population Motor Model

### Premotor neurons (N=178):

```
dup/dt = (-up + gp * softplus(up @ Jpp) + bp + wsp @ stim) / taup
```

### Motor neurons (M=52):

```
dum/dt = (-um + gm * softplus(up @ Jpm) + bm) / taum
```

- **230 neurons** total (178 premotor + 52 motor), **2 cell types**, **4,222 edges** (Jpp=2,390 + Jpm=1,832)
- Activation: **Softplus** (log(1 + exp(x)))
- Gains gp, gm clamped to [0.5, 5.0]
- taup, taum ~ 1.0, dt=0.05
- 2 stimulus conditions (forward/backward), 2 stimulus channels
- Inhibitory neurons get negative weights (Dale's law in connectome)
- 2,400 frames, delta_t=0.05
- Feedforward: premotor->motor only, plus premotor recurrence
- Only 2 neuron types -> embedding should separate 2 clusters

## GNN Architecture

- **g_phi**: Edge message MLP. Maps (v_j, a_j) -> message. `g_phi_positive=true`.
- **f_theta**: Node update MLP. Maps (v_i, a_i, aggregated_msg, I_i) -> dv_i/dt.
- **Embedding a_i**: learnable per-neuron type vector.

**CRITICAL — coupled parameters**: When changing `embedding_dim`, you MUST also update:

- `input_size = 1 + embedding_dim`
- `input_size_update = 3 + embedding_dim`

Example: embedding_dim=2 -> input_size=3, input_size_update=5.

## Training Parameters

| Parameter                 | Default | Description                                  |
| ------------------------- | ------- | -------------------------------------------- |
| `lr_W`                    | 1e-3    | Learning rate for connectivity W             |
| `lr`                      | 1e-3    | Learning rate for g_phi and f_theta MLPs     |
| `lr_embedding`            | 1e-3    | Learning rate for neuron embeddings          |
| `n_epochs`                | 2       | Number of training epochs                    |
| `batch_size`              | 2       | Batch size                                   |
| `data_augmentation_loop`  | 100     | Data augmentation multiplier                 |
| `w_init_mode`             | zeros   | W initialization: "zeros", "randn_scaled"    |
| `coeff_g_phi_diff`        | 1500    | Monotonicity penalty on g_phi                |
| `coeff_f_theta_weight_L2` | 0.001   | L2 penalty on f_theta MLP weights            |
| `coeff_f_theta_diff`      | 0       | Negative monotonicity of f_theta w.r.t. state v_i (enforces leak: df/dv < 0) |
| `coeff_f_theta_msg_diff`  | 0       | Positive monotonicity of f_theta w.r.t. message input |
| `coeff_W_L1`              | 0       | L1 sparsity on W                             |
| `coeff_W_L2`              | 1e-5    | L2 penalty on W                              |
| `coeff_W_sign`            | 0       | Dale's law penalty                           |
| `use_gt_edges`            | true    | If false, train on fully connected graph     |
| `dale_law`                | false   | Enforce Dale's law: force consistent sign per W column 3× per epoch |
| `noise_model_level`       | 0.0     | Observation noise std added to trajectories  |

## Training Time Constraint

**Target ~60 min per iteration.** Use `data_augmentation_loop` (DAL) to control training time. After each batch, check `training_time_min` in the metrics and adjust DAL for the next batch:

- If training_time_min < 40 min: **increase** DAL (e.g. multiply by 1.5-2×)
- If training_time_min > 70 min: **decrease** DAL (e.g. divide by 1.5-2×)
- DAL scales training time linearly — doubling DAL ≈ doubles training time

Longer training gives W more time to converge. Always use the full time budget.

## Parallel Mode — 4 Slots Per Batch

Each batch runs 4 slots with different seeds (forced by pipeline). You choose the strategy:

- **Exploration** (default): Slot 0 = parent/control (unchanged). Slots 1-3 each change **exactly one** parameter. This gives 3 causal tests per batch.
- **Robustness test**: ALL 4 slots use the SAME config. The pipeline forces different seeds, so this measures seed robustness. Use this when a config looks promising.

State your choice (exploration vs robustness test) in the log entry.

### Robustness Assessment (when running same config across 4 slots)

- **Robust**: all 4 slots connectivity_R2 > 0.7
- **Partially robust**: 2-3 slots > 0.7
- **Fragile**: 0-1 slots > 0.7

## Block Partition

The blocks below provide a **recommended exploration roadmap**. Follow the block focus as a guide but use your scientific judgment — if early results clearly suggest a detour or shortcut, adapt. The block boundaries are soft: you can revisit earlier axes or combine parameters across blocks when evidence supports it.

| Block | Focus                    | Parameters to scan                                                         | Ranges                                                                                                           |
| ----- | ------------------------ | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1     | **lr_W + W_L1**          | `lr_W`, `coeff_W_L1`                                                       | lr_W: {1e-4, 3e-4, 6e-4, 1e-3}, W_L1: {0, 1e-6, 1e-5, 5e-5}                                                      |
| 2     | **W initialization**     | `w_init_mode`                                                              | {zeros, randn, randn_scaled} — low-rank dynamics may favor randn                                                 |
| 3     | **Training volume**      | `data_augmentation_loop`, `n_epochs`                                       | DAL: {50, 100, 200}, n_epochs: {2, 4} (halve DAL when doubling epochs)                                           |
| 4     | **GT edges comparison**  | `use_gt_edges`                                                             | use_gt_edges: {true, false} — default is fully connected. One block to test if providing GT edges helps or hurts |
| 5     | **Regularization + Dale's law** | `coeff_W_L2`, `coeff_W_sign`, `dale_law`, `coeff_g_phi_diff`, `coeff_f_theta_diff`, `coeff_f_theta_msg_diff` | W_L2: {5e-6, 1e-5, 2e-5}, W_sign: {0, 0.05, 0.2}, dale_law: {false, true}, g_phi_diff: {500, 1000, 1500}, f_theta_diff: {0, 10, 50} (leak), f_theta_msg_diff: {0, 10, 50}. Monitor dale_law_score in all iterations. |
| 6     | **Architecture + noise** | `hidden_dim`, `embedding_dim`, `noise_model_level`                         | hidden_dim: {48, 64, 80}, embedding_dim: {2, 4} (update input_size accordingly), noise: {0, 0.05, 0.5}           |
| 7     | **Free exploration I**   | Any parameter                                                              | Consolidate best from blocks 1-6, test novel combinations, attempt to break R2 ceiling                           |
| 8     | **Free exploration II**  | Any parameter                                                              | Continue ceiling-breaking attempts, confirm final robust config                                                  |

### Low-rank context

These biological connectomes produce **low-rank activity** (two-population feedforward structure). From prior low-rank exploration (NeuralGraph, 100-1000 neurons):

- **W_L1 calibration is critical**: L1=1E-6 unlocks near-perfect dynamics recovery; L1=1E-5 gives good W but partial rollout. Too much L1 destroys the low-rank structure.
- **W initialization matters**: `randn` outperforms `zeros` for low-rank regimes (opposite of chaotic regime). Must be tested — Block 2.
- **Fully connected training is the default** (`use_gt_edges=false`): the GNN trains on all-to-all edges and must learn which are zero via L1 sparsity. Block 4 compares GT edges vs fully connected.

### Model-specific notes for Block 5

- The larva connectome has clear excitatory/inhibitory split -> W_sign regularization may be more effective here than CX
- Feedforward premotor->motor structure means W_L1 sparsity is physically motivated (motor neurons don't project back)
- Only 2 types, so f_theta_msg_diff at moderate values (10-50) may help without over-constraining

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
Slot 0: conn_R2=A, rollout_pearson=B, cluster_acc=C, dale_score=D, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, rollout_pearson=B, cluster_acc=C, dale_score=D, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, rollout_pearson=B, cluster_acc=C, dale_score=D, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, rollout_pearson=B, cluster_acc=C, dale_score=D, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
W matrix: [visual comment from connectivity heatmap — sparsity, sign structure, convergence]
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

- Read base config — this IS the baseline. Do NOT change any default values.
- Slot 0 = baseline (no changes at all).
- Slots 1-3: each changes EXACTLY ONE parameter from the block focus. Keep everything else at baseline.
- Hypothesis: "The baseline config achieves connectivity_R2 > 0.5 robustly across seeds"

---

# Working Memory Structure

```markdown
# Working Memory: larva

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
