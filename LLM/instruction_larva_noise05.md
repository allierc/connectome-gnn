# Drosophila Larva (Noise 0.5) — LLM Exploration

## Goal

Maximize **connectivity_R2** for the **Drosophila larva two-population motor model** (Beiran & Litwin-Kumar 2023, Figure 5a-c) under **intrinsic noise (sigma=0.5)**.

This exploration starts from the **best noise=0.05 config** (W_L2=2e-5 gave best single-seed 0.801). Cross-model evidence strongly suggests noise=0.5 improves W recovery: flyvis 0.990, CX 0.9997, zebrafish 0.988. However, noise=0.5 also introduces bimodal convergence (50% failure in zebrafish) — the larva softplus model may show similar instability.

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Parent config (best noise=0.05)

```
lr_W: 1e-4
lr: 1e-3
lr_embedding: 1e-3
n_epochs: 2
data_augmentation_loop: 600
w_init_mode: zeros
coeff_W_L1: 1e-6
coeff_W_L2: 2e-5
coeff_W_sign: 0.05
coeff_g_phi_diff: 1500
coeff_f_theta_weight_L2: 0.001
coeff_f_theta_msg_diff: 50
use_gt_edges: true
noise_model_level: 0.5
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

**IMPORTANT**: `noise_model_level` is set to **0.5** in the base config. Do NOT change it — this file is specifically for the noise=0.5 experiment.

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
- 2,400 frames, delta_t=0.05, **noise_model_level=0.5**
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

| Parameter                 | Default | Description                                  |
| ------------------------- | ------- | -------------------------------------------- |
| `lr_W`                    | 1e-4    | Learning rate for connectivity W             |
| `lr`                      | 1e-3    | Learning rate for g_phi and f_theta MLPs     |
| `lr_embedding`            | 1e-3    | Learning rate for neuron embeddings          |
| `n_epochs`                | 2       | Number of training epochs                    |
| `batch_size`              | 2       | Batch size                                   |
| `data_augmentation_loop`  | 600     | Data augmentation multiplier                 |
| `w_init_mode`             | zeros   | W initialization: "zeros", "randn_scaled"    |
| `coeff_g_phi_diff`        | 1500    | Monotonicity penalty on g_phi                |
| `coeff_f_theta_weight_L2` | 0.001   | L2 penalty on f_theta MLP weights            |
| `coeff_f_theta_diff`      | 0       | Negative monotonicity of f_theta w.r.t. state v_i |
| `coeff_f_theta_msg_diff`  | 50      | Positive monotonicity of f_theta w.r.t. message input |
| `coeff_W_L1`              | 1e-6    | L1 sparsity on W                             |
| `coeff_W_L2`              | 2e-5    | L2 penalty on W                              |
| `coeff_W_sign`            | 0.05    | Dale's law penalty                           |
| `use_gt_edges`            | true    | If false, train on fully connected graph     |
| `dale_law`                | false   | Enforce Dale's law                           |
| `noise_model_level`       | 0.5     | **FIXED** — intrinsic noise level for this experiment |

## Training Time Constraint

**Target ~60 min per iteration.** Use `data_augmentation_loop` (DAL) to control training time. After each batch, check `training_time_min` in the metrics and adjust DAL for the next batch:

- If training_time_min < 40 min: **increase** DAL (e.g. multiply by 1.5-2x)
- If training_time_min > 70 min: **decrease** DAL (e.g. divide by 1.5-2x)
- DAL scales training time linearly — doubling DAL ~ doubles training time

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

These blocks incorporate learnings from larva noise=0.05 (W_L2=2e-5 best lever, seed sensitivity dominant) and cross-model noise=0.5 results (CX: W_L2=1e-4 critical; zebrafish: 50% failure from W collapse).

| Block | Focus                          | Parameters to scan                                                         | Ranges                                                                                                           |
| ----- | ------------------------------ | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1     | **Baseline validation**        | None (robustness test)                                                     | Run noise=0.05 best config + noise=0.5 across 4 seeds. Establish baseline under strong noise.                    |
| 2     | **Regularization re-tune**     | `coeff_W_L1`, `coeff_W_L2`, `coeff_W_sign`                                | W_L2: {1e-5, 2e-5, 5e-5, 1e-4} (CX needed 1e-4 at noise=0.5). W_L1: {1e-7, 1e-6, 5e-6}. W_sign: {0.01, 0.05, 0.1}. |
| 3     | **Training volume re-tune**    | `data_augmentation_loop`, `n_epochs`                                       | DAL: {400, 600, 900}, n_epochs: {2, 3}. Strong noise may need more training.                                     |
| 4     | **Architecture + batch_size**  | `hidden_dim`, `embedding_dim`, `batch_size`                                | hidden_dim: {48, 64, 80, 96}, batch_size: {2, 4}. Noisy data may need larger capacity.                           |
| 5     | **Monotonicity + Dale's law**  | `coeff_g_phi_diff`, `coeff_f_theta_diff`, `coeff_f_theta_msg_diff`, `dale_law` | g_phi_diff: {500, 1000, 1500, 2000}, f_theta_msg_diff: {0, 50, 100}, dale_law: {false, true}. CX found dale_law was key lever. |
| 6     | **Free exploration I**         | Any parameter                                                              | Consolidate best from blocks 1-5, test novel combinations                                                        |
| 7     | **Free exploration II**        | Any parameter                                                              | Continue ceiling-breaking attempts                                                                               |
| 8     | **Final robustness**           | None (robustness test)                                                     | Multi-seed robustness confirmation of best config                                                                |

### Noise=0.5 specific considerations (from cross-model evidence)

- **Strong noise dramatically helps W recovery in other models**: CX went from 0.804 (clean) to 0.9997 (noise=0.5). Zebrafish from 0.022 to 0.988. Flyvis from 0.926 to 0.990. Expect larva improvement.
- **W_L2 is the critical lever at noise=0.5**: CX required 10x increase (1e-5 -> 1e-4) to prevent epoch-2 collapse. Start scanning W_L2 early.
- **Bimodal convergence is common**: Zebrafish had 50% failure (W collapse to near-zero). Watch for spectral radius collapse as diagnostic. Failed seeds show spec_rad << true.
- **W_L1 may cause W collapse**: At noise=0.5, gradient variance is high. Strong L1 can drive W to zero before signal accumulates. Consider reducing L1.
- **dale_law=true was breakthrough for CX noise=0.05**: May also help larva stabilize signs under noise.
- **Feedforward structure limits noise propagation**: Motor neurons receive noisy premotor output but don't feed back — noise effect is asymmetric.
- **f_theta_msg_diff=50 was strongest regularizer for clean larva**: Keep it and test further.

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

From `analysis.log`: connectivity_R2, rollout_pearson, cluster_accuracy, training_time_min.

### Step 3: Write Log Entries + Update Memory

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, DAL=D, n_epochs=E, W_L1=A, W_L2=B, W_sign=C, hidden_dim=H, batch_size=B
Slot 0: conn_R2=A, rollout_pearson=B, cluster_acc=C, dale_score=D, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, rollout_pearson=B, cluster_acc=C, dale_score=D, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, rollout_pearson=B, cluster_acc=C, dale_score=D, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, rollout_pearson=B, cluster_acc=C, dale_score=D, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
W matrix: [visual comment from connectivity heatmap]
Verdict: [supported/falsified/inconclusive]
Next: parent=P

## Winner Config (COMPULSORY)

**At every block boundary**, you MUST save the current best config as a winner file.
This is a COMPULSORY task — do not skip it.

1. Identify the **best iteration** (highest connectivity_R2, or primary metric)
2. Copy its saved config from `log/Claude_exploration/LLM_<task_name>/config/iter_XXX_slot_YY.yaml`
3. Save it to `config/larva/larva_noise05_winner.yaml` with a YAML comment header:

```yaml
# Winner config: larva_noise05_winner.yaml
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

Destination: `config/larva/larva_noise05_winner.yaml`

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

- Read base config — the parent noise=0.05 best config + noise_model_level=0.5 IS the baseline.
- Block 1 is a **robustness test**: all 4 slots use the same config (different seeds).
- Hypothesis: "The best noise=0.05 config with noise=0.5 achieves connectivity_R2 >= noise=0.05 baseline (0.60) robustly across seeds"

---

# Working Memory Structure

```markdown
# Working Memory: larva_noise05

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
