# Drosophila Larva FC (Noise 0.05) — LLM Exploration

## Goal

Maximize **connectivity_R2** for the **Drosophila larva two-population motor model** (Beiran & Litwin-Kumar 2023, Figure 5a-c) using a **fully connected graph** under **intrinsic noise (sigma=0.05)**.

This exploration starts from the **best FC noise-free config** (conn_R2=0.435 best, mean=0.268+-0.106, CV=40%). FC larva is structurally limited — 52,670 edges vs 4,222 true (12.5x search space) creates massive degeneracy. Cross-model evidence strongly suggests noise breaks FC degeneracy:
- **CX FC**: noise-free=0.804 -> noise005=0.982 (22% improvement)
- **Zebrafish FC**: noise-free=0.022 -> noise005=0.918 (42x improvement)
- **Larva GT**: noise-free=0.908 -> noise005=0.870 best (but better mean stability)

The parent config's strong W_L1=4e-3 (667x stronger than GT baseline) and g_phi_norm=0.01 were critical levers for noise-free FC. Under noise, implicit regularization from stochastic dynamics may reduce the need for explicit regularization — these parameters may need re-tuning.

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Parent config (best FC noise-free)

```
lr_W: 1e-3
lr: 1e-3
lr_embedding: 1e-3
n_epochs: 2
data_augmentation_loop: 630
w_init_mode: zeros
hidden_dim: 64
embedding_dim: 4
coeff_g_phi_diff: 1500
coeff_g_phi_norm: 0.01
coeff_f_theta_diff: 10
coeff_f_theta_msg_diff: 50
coeff_f_theta_weight_L2: 0.001
coeff_W_L1: 0.004
coeff_W_L2: 1.5e-6
coeff_W_sign: 0.05
dale_law: false
use_gt_edges: false
batch_size: 4
regul_annealing_rate: 0.7
noise_model_level: 0.05
```

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

**IMPORTANT**: `use_gt_edges` is set to **false** in the base config. Do NOT change it — this file is specifically for the FC experiment.

## Larva Two-Population Motor Model

### Premotor neurons (N=178):

```
dup/dt = (-up + gp * softplus(up @ Jpp) + bp + wsp @ stim) / taup
```

### Motor neurons (M=52):

```
dum/dt = (-um + gm * softplus(up @ Jpm) + bm) / taum
```

- **230 neurons** total (178 premotor + 52 motor), **2 cell types**, **4,222 true edges** (Jpp=2,390 + Jpm=1,832)
- **FC edge count**: 230x229 = 52,670 edges (12.5x more than GT)
- Activation: **Softplus** (log(1 + exp(x)))
- Gains gp, gm clamped to [0.5, 5.0]
- taup, taum ~ 1.0, dt=0.05
- 2 stimulus conditions (forward/backward), 2 stimulus channels
- Inhibitory neurons get negative weights (Dale's law in connectome)
- 2,400 frames, delta_t=0.05, **noise_model_level=0.05**
- Feedforward: premotor->motor only, plus premotor recurrence
- Only 2 neuron types -> embedding should separate 2 clusters

## GNN Architecture

- **g_phi**: Edge message MLP. Maps (v_j, a_j) -> message. `g_phi_positive=false` (needs to pass softplus shape).
- **f_theta**: Node update MLP. Maps (v_i, a_i, aggregated_msg, I_i) -> dv_i/dt.
- **Embedding a_i**: learnable per-neuron type vector.

**CRITICAL — coupled parameters**: `embedding_dim` must be >= 2 (embedding_dim=1 crashes plotting). When changing `embedding_dim`, you MUST also update:

- `input_size = 1 + embedding_dim`
- `input_size_update = 3 + embedding_dim`

Example: embedding_dim=4 -> input_size=5, input_size_update=7.

## Training Parameters

| Parameter                 | Default | Description                                  |
| ------------------------- | ------- | -------------------------------------------- |
| `lr_W`                    | 1e-3    | Learning rate for connectivity W             |
| `lr`                      | 1e-3    | Learning rate for g_phi and f_theta MLPs     |
| `lr_embedding`            | 1e-3    | Learning rate for neuron embeddings          |
| `n_epochs`                | 2       | Number of training epochs                    |
| `batch_size`              | 4       | Batch size                                   |
| `data_augmentation_loop`  | 630     | Data augmentation multiplier                 |
| `w_init_mode`             | zeros   | W initialization: "zeros", "randn_scaled"    |
| `hidden_dim`              | 64      | Hidden dimension for MLPs                    |
| `embedding_dim`           | 4       | Neuron embedding dimension                   |
| `coeff_g_phi_diff`        | 1500    | Monotonicity penalty on g_phi                |
| `coeff_g_phi_norm`        | 0.01    | Norm penalty on g_phi output                 |
| `coeff_f_theta_weight_L2` | 0.001   | L2 penalty on f_theta MLP weights            |
| `coeff_f_theta_diff`      | 10      | Negative monotonicity of f_theta w.r.t. state v_i |
| `coeff_f_theta_msg_diff`  | 50      | Positive monotonicity of f_theta w.r.t. message input |
| `coeff_W_L1`              | 0.004   | L1 sparsity on W (667x stronger than GT baseline) |
| `coeff_W_L2`              | 1.5e-6  | L2 penalty on W                              |
| `coeff_W_sign`            | 0.05    | Dale's law penalty                           |
| `regul_annealing_rate`    | 0.7     | Regularization annealing rate                |
| `use_gt_edges`            | false   | **FIXED** — fully connected graph            |
| `dale_law`                | false   | Enforce Dale's law                           |
| `noise_model_level`       | 0.05    | **FIXED** — intrinsic noise level for this experiment |

## Training Time Constraint

**Target ~60 min per iteration.** Use `data_augmentation_loop` (DAL) to control training time. After each batch, check `training_time_min` in the metrics and adjust DAL for the next batch:

- If training_time_min < 40 min: **increase** DAL (e.g. multiply by 1.5-2x)
- If training_time_min > 70 min: **decrease** DAL (e.g. divide by 1.5-2x)
- DAL scales training time linearly — doubling DAL ~ doubles training time
- **NOTE**: FC has 12.5x more edges than GT — expect slower per-step training. Initial DAL may need reduction.

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

These blocks start from the best FC noise-free config with noise_model_level=0.05. The focus is on whether noise=0.05 breaks FC degeneracy (as seen in CX and zebrafish) and whether the strong regularization from noise-free FC needs re-tuning under noise.

| Block | Focus                          | Parameters to scan                                                         | Ranges                                                                                                           |
| ----- | ------------------------------ | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1     | **Baseline validation**        | None (robustness test)                                                     | Run best FC noise-free config + noise=0.05 across 4 seeds. Establish baseline under mild noise.                  |
| 2     | **Regularization re-tune**     | `coeff_W_L1`, `coeff_W_L2`, `coeff_g_phi_norm`                            | W_L1: {1e-3, 4e-3, 1e-2} (noise provides implicit regularization — L1 may need reduction). W_L2: {1e-6, 1.5e-6, 5e-6, 1e-5}. g_phi_norm: {0, 0.01, 0.05}. |
| 3     | **Training volume re-tune**    | `data_augmentation_loop`, `n_epochs`                                       | DAL: {400, 630, 900}, n_epochs: {2, 3}. Noisy data may need more training to average out stochastic gradients.   |
| 4     | **Architecture + batch_size**  | `hidden_dim`, `embedding_dim`, `batch_size`                                | hidden_dim: {48, 64, 80, 96}, batch_size: {2, 4, 8}. Noisy data may need larger capacity.                        |
| 5     | **Monotonicity + Dale's law**  | `coeff_g_phi_diff`, `coeff_f_theta_diff`, `coeff_f_theta_msg_diff`, `dale_law` | g_phi_diff: {500, 1000, 1500, 2000}, f_theta_msg_diff: {0, 50, 100}, dale_law: {false, true}. CX found dale_law was key lever. |
| 6     | **Free exploration I**         | Any parameter                                                              | Consolidate best from blocks 1-5, test novel combinations                                                        |
| 7     | **Free exploration II**        | Any parameter                                                              | Continue ceiling-breaking attempts                                                                               |
| 8     | **Final robustness**           | None (robustness test)                                                     | Multi-seed robustness confirmation of best config                                                                |

### Noise=0.05 + FC specific considerations

- **Noise=0.05 dramatically helped other FC modes**: CX FC went from 0.804 to 0.982 (+22%). Zebrafish FC went from 0.022 to 0.918 (42x). The mechanism is noise-induced state-space exploration that breaks weight degeneracy in the FC search space.
- **W_L1=4e-3 may need reduction**: The parent FC config uses extremely strong L1 (667x GT baseline) to prune 48K null edges. Noise already provides implicit regularization by breaking degeneracy — L1 may be redundant or even harmful (over-sparsifying before noise signal accumulates).
- **g_phi_norm=0.01 was the biggest single lever for noise-free FC**: This prevents g_phi from absorbing W structure. Under noise, g_phi may need different normalization since noisy inputs change the message scale.
- **regul_annealing_rate=0.7 interacts with noise**: Regularization decays during training. If noise already regularizes, the annealing may be too aggressive or redundant.
- **Feedforward structure limits noise propagation**: Motor neurons receive noisy premotor output but don't feed back — noise effect is asymmetric across the two populations.
- **f_theta_msg_diff=50 was strongest regularizer for clean larva**: Keep it and test further.
- **Larva GT showed marginal noise=0.05 effect (0.908->0.870 best)**: But GT has no degeneracy to break. The FC mode is where noise should have maximal impact.

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

From `analysis.log`: connectivity_R2, rollout_pearson, cluster_accuracy, training_time_min.

### Step 3: Write Log Entries + Update Memory

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, DAL=D, n_epochs=E, W_L1=A, W_L2=B, W_sign=C, g_phi_norm=G, hidden_dim=H, batch_size=B
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

## Winner Config (COMPULSORY)

**At every block boundary**, you MUST save the current best config as a winner file.
This is a COMPULSORY task — do not skip it.

1. Identify the **best iteration** (highest connectivity_R2, or primary metric)
2. Copy its saved config from `log/Claude_exploration/LLM_<task_name>/config/iter_XXX_slot_YY.yaml`
3. Save it to `config/larva/larva_fc_noise005_winner.yaml` with a YAML comment header:

```yaml
# Winner config: larva_fc_noise005_winner.yaml
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

Destination: `config/larva/larva_fc_noise005_winner.yaml`

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

- Read base config — the parent FC noise-free best config + noise_model_level=0.05 IS the baseline.
- Block 1 is a **robustness test**: all 4 slots use the same config (different seeds).
- Hypothesis: "Adding noise=0.05 to the best FC config significantly improves connectivity_R2 above the noise-free FC ceiling (0.435 best, 0.268 mean), consistent with CX and zebrafish FC noise evidence"

---

# Working Memory Structure

```markdown
# Working Memory: larva_fc_noise005

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
