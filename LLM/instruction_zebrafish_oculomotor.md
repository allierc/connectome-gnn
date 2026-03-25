# Zebrafish Oculomotor — LLM Exploration

## Goal

Maximize **connectivity_R2** for the **zebrafish oculomotor integrator** (Beiran & Litwin-Kumar 2023, Figure 5g-i).

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned effective W and ground-truth effective W
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free data
3. **cluster_accuracy** (THIRD) — neuron type clustering accuracy from learned embeddings

Informational (not for optimization): onestep_pearson, f_theta_R2, g_phi_R2, spectral_radius_learned vs spectral_radius_true.

**NOTE**: tau_R2 and V_rest_R2 are not applicable (fixed tau=1, no resting potential).

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
- Do NOT change parameters outside the current block focus (e.g. do not touch w_init_mode, W_L2, hidden_dim unless the block explicitly includes them).
- Do NOT skip the baseline — always keep one slot as an unchanged control.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness).

## Data Generation

Each slot re-generates data with a **different random seed**.
Seeds are **forced by the pipeline** — DO NOT modify them in config files.

- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

**DO NOT change `simulation:` parameters** except seed (managed automatically).

## Zebrafish Oculomotor Integrator Model

```
dr/dt = (-r + W @ r + I(t) * v_in) / tau
```

- **609 neurons**, 6 cell types (_Int_, _DOs_, _Axl_, ABD_m, ABD_i, vSPNs), from Goldman lab connectome
- **LINEAR**: no activation function (identity g_phi)
- tau=1.0 (fixed), dt=0.001
- W scaled to spectral radius = 0.9
- Stimulus: 4-channel multi-direction input along eigenvectors of W
- 21,000 frames (3 pulse repeats x 7,000)
- g_phi should learn identity (slope=1), f_theta should learn f(v)=-v (slope=-1)
- Dynamics purely determined by W eigenstructure
- Some populations have zeroed connections (ABD, axial, vSPNs)

**Key challenge**: Linear ODE means W must be precisely recovered from linear dynamics alone — no nonlinearity to disambiguate.

## GNN Architecture

- **g_phi**: Edge message MLP. Maps (v_j, a_j) -> message. `g_phi_positive=false` (linear model needs negative pass-through).
- **f_theta**: Node update MLP. Maps (v_i, a_i, aggregated_msg, I_i) -> dv_i/dt.
- **Embedding a_i**: learnable per-neuron type vector.

**CRITICAL — coupled parameters**: `embedding_dim` must be >= 2 (embedding_dim=1 crashes plotting). When changing `embedding_dim`, you MUST also update:

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
| `use_gt_edges`            | false   | If true, train on ground-truth edges only    |
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
| 4     | **Regularization + Dale's law** | `coeff_W_L2`, `coeff_W_sign`, `dale_law`, `coeff_g_phi_diff`, `coeff_f_theta_diff`, `coeff_f_theta_msg_diff` | W_L2: {5e-6, 1e-5, 2e-5}, W_sign: {0, 0.01, 0.05}, dale_law: {false, true}, g_phi_diff: {500, 1000, 1500}, f_theta_diff: {0, 10, 100} (leak), f_theta_msg_diff: {0, 10, 100}. Monitor dale_law_score in all iterations. |
| 5     | **Architecture + batch_size** | `hidden_dim`, `embedding_dim`, `batch_size`                               | hidden_dim: {48, 64, 80}, embedding_dim: {2, 4}, batch_size: {2, 4, 8}. From flyvis: bs=4 eliminated catastrophic failures. |
| 6     | **Free exploration I**   | Any parameter                                                              | Consolidate best from blocks 1-5, test novel combinations, attempt to break R2 ceiling                           |
| 7     | **Free exploration II**  | Any parameter                                                              | Continue ceiling-breaking attempts, confirm final robust config                                                  |
| 8     | **Final robustness**     | None (robustness test)                                                     | 4-seed robustness test of best config from blocks 1-7                                                            |

### Low-rank context

These biological connectomes produce **low-rank activity** (linear integrator, dynamics purely determined by W eigenstructure). From prior low-rank exploration (NeuralGraph, 100-1000 neurons):

- **W_L1 calibration is critical**: L1=1E-6 unlocks near-perfect dynamics recovery; L1=1E-5 gives good W but partial rollout. Too much L1 destroys the low-rank structure.
- **W initialization matters**: `randn` outperforms `zeros` for low-rank regimes (opposite of chaotic regime). Must be tested — Block 2.
- **Fully connected training is the default** (`use_gt_edges=false`): the GNN trains on all-to-all edges and must learn which are zero via L1 sparsity. See `instruction_zebrafish_oculomotor_gt_edges.md` for the GT edges variant.

### Model-specific notes for Block 4

- **Linear model**: f_theta_msg_diff is physically well-motivated here — f_theta should be monotonically increasing in message (it learns -v + msg + stim, which IS monotonic in msg). Values up to 100 may help.
- g_phi_diff may be less important since g_phi learns identity — lower values (500) may suffice.
- W_sign: zebrafish connectome has mixed excitatory/inhibitory types but some populations are zeroed — gentle W_sign ({0.01, 0.05}) only.
- W_L1 sparsity: many zero'd populations mean true W is relatively sparse — L1 may help here.
- **Spectral radius**: the true W has spectral_radius=0.9. If learned spectral_radius diverges far from 0.9, W recovery fails. Monitor this diagnostic.

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

## Winner Config (COMPULSORY)

**At every block boundary**, you MUST save the current best config as a winner file.
This is a COMPULSORY task — do not skip it.

1. Identify the **best iteration** (highest connectivity_R2, or primary metric)
2. Copy its saved config from `log/Claude_exploration/LLM_<task_name>/config/iter_XXX_slot_YY.yaml`
3. Save it to `config/zebrafish_oculomotor/zebrafish_oculomotor_winner.yaml` with a YAML comment header:

```yaml
# Winner config: zebrafish_oculomotor_winner.yaml
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

Destination: `config/zebrafish_oculomotor/zebrafish_oculomotor_winner.yaml`

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
# Working Memory: zebrafish_oculomotor

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
