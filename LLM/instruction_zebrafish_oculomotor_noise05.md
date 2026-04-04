# Zebrafish Oculomotor (Noise 0.5) — LLM Exploration

## Goal

Maximize **connectivity_R2** for the **zebrafish oculomotor integrator** (Beiran & Litwin-Kumar 2023, Figure 5g-i) under **strong intrinsic noise (sigma=0.5)**.

This exploration maps the **noise dose-response curve** for connectivity recovery. The noise=0.05 exploration (128 iterations) already proved the degeneracy-breaking hypothesis: process noise dramatically improves W identifiability (best R2=0.918, mean=0.371) compared to the clean baseline (R2~0.02). Now noise=0.5 tests whether **stronger noise further helps or starts to hurt** — does the dose-response curve plateau, continue rising, or reverse at high noise levels?

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Parent config (best from noise=0.05 exploration, 128 iterations)

```
lr_W: 1e-4
lr: 1e-3
lr_embedding: 1e-3
n_epochs: 2
data_augmentation_loop: 35
w_init_mode: randn_scaled
coeff_W_L1: 1e-4
coeff_W_L2: 1e-5
coeff_g_phi_diff: 1500
coeff_f_theta_weight_L2: 0.001
coeff_W_sign: 0
use_gt_edges: false
noise_model_level: 0.5
```

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

## Noise Model

Single noise source in the training data:

**Dynamics noise** (`noise_model_level=0.5`): `v(t+1) = v(t) + dt * f(v, W, I) + epsilon_dyn(t)`, epsilon_dyn ~ N(0, 0.5)

### CAUSALITY RULE (MANDATORY — READ THIS)

**If you change more than one parameter per slot, you CANNOT attribute the effect. This is a fatal experimental design error.**

- In EXPLORATION mode: Slot 0 = parent/baseline (unchanged control). Slots 1-3 each change **exactly one** parameter from the parent.
- Do NOT change parameters outside the current block focus.
- Do NOT skip the baseline — always keep one slot as an unchanged control.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness).

## Scientific Context

The zebrafish **oculomotor integrator** under **strong noise (sigma=0.5)** maps the high end of the noise dose-response curve. The noise=0.05 exploration (128 iterations) proved process noise breaks linear degeneracy (R2=0.918 best, mean=0.371, 46× improvement). At sigma=0.5, the question is whether stronger noise further improves identifiability or creates a signal-to-noise tradeoff that degrades performance. The bimodal convergence observed at this noise level (50% failure) suggests the SNR limit may be reached — too much noise adds measurement error that contradicts the structural constraints of the linear integrator.

## Data Generation

Each slot re-generates data with a **different random seed**.
Seeds are **forced by the pipeline** — DO NOT modify them in config files.

- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

**DO NOT change `simulation:` parameters** except seed (managed automatically).

**IMPORTANT**: `noise_model_level` is set to **0.5** in the base config. Do NOT change it — this file is specifically for the noise=0.5 experiment.

## Zebrafish Oculomotor Integrator Model

```
dr/dt = (-r + W @ r + I(t) * v_in) / tau
```

- **609 neurons**, 6 cell types (_Int_, _DOs_, _Axl_, ABD_m, ABD_i, vSPNs), from Goldman lab connectome
- **LINEAR**: no activation function (identity g_phi)
- tau=1.0 (fixed), dt=0.001
- W scaled to spectral radius = 0.9
- Stimulus: 4-channel multi-direction input along eigenvectors of W
- 21,000 frames (3 pulse repeats x 7,000), **noise_model_level=0.5**
- g_phi should learn identity (slope=1), f_theta should learn f(v)=-v (slope=-1)
- Dynamics purely determined by W eigenstructure
- Some populations have zeroed connections (ABD, axial, vSPNs)

**Key challenge**: Linear ODE means W must be precisely recovered from linear dynamics alone — no nonlinearity to disambiguate. Noise=0.05 proved that process noise breaks the degeneracy (0.918 best R2). **Now we test whether 10x stronger noise (0.5) continues to help or introduces signal-to-noise tradeoffs.**

## GNN Architecture

- **g_phi**: Edge message MLP. Maps (v_j, a_j) -> message. `g_phi_positive=false` (linear model needs negative pass-through).
- **f_theta**: Node update MLP. Maps (v_i, a_i, aggregated_msg, I_i) -> dv_i/dt.
- **Embedding a_i**: learnable per-neuron type vector.

**CRITICAL — coupled parameters**: `embedding_dim` must be >= 2 (embedding_dim=1 crashes plotting). When changing `embedding_dim`, you MUST also update:

- `input_size = 1 + embedding_dim`
- `input_size_update = 3 + embedding_dim`

Example: embedding_dim=2 -> input_size=3, input_size_update=5.

## Explorable Parameters

| Parameter                 | Default      | Description                                  |
| ------------------------- | ------------ | -------------------------------------------- |
| `lr_W`                    | 1e-4         | Learning rate for connectivity W             |
| `lr`                      | 1e-3         | Learning rate for g_phi and f_theta MLPs     |
| `lr_embedding`            | 1e-3         | Learning rate for neuron embeddings          |
| `n_epochs`                | 2            | Number of training epochs                    |
| `batch_size`              | 2            | Batch size                                   |
| `data_augmentation_loop`  | 35           | Data augmentation multiplier (budget, can go higher for quality) |
| `w_init_mode`             | randn_scaled | W initialization: "zeros", "randn_scaled"    |
| `coeff_g_phi_diff`        | 1500         | Monotonicity penalty on g_phi                |
| `coeff_f_theta_weight_L2` | 0.001        | L2 penalty on f_theta MLP weights            |
| `coeff_f_theta_diff`      | 0            | Negative monotonicity of f_theta w.r.t. state v_i |
| `coeff_f_theta_msg_diff`  | 0            | Positive monotonicity of f_theta w.r.t. message input |
| `coeff_W_L1`              | 1e-4         | L1 sparsity on W                             |
| `coeff_W_L2`              | 1e-5         | L2 penalty on W                              |
| `coeff_W_sign`            | 0            | Dale's law penalty                           |
| `use_gt_edges`            | false        | If true, train on ground-truth edges only    |
| `dale_law`                | false        | Enforce Dale's law                           |
| `noise_model_level`       | 0.5          | **FIXED** — intrinsic noise level for this experiment |

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

## Block Structure

These blocks build on the noise=0.05 exploration results. The parent config already incorporates the best hyperparameters found there (lr_W=1e-4, W_L1=1e-4, W_L2=1e-5, W_sign=0). The focus is on whether 10x stronger noise changes the optimal operating point.

| Block | Focus                          | Parameters to scan                                                         | Ranges                                                                                                           |
| ----- | ------------------------------ | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1     | **Baseline validation**        | None (robustness test)                                                     | Run best noise=0.05 config + noise_model_level=0.5 across 4 seeds. Does 10x stronger noise maintain or improve the 0.918 best R2? |
| 2     | **Regularization re-tune**     | `coeff_W_L1`, `coeff_W_L2`, `coeff_W_sign`                                | W_L1: {5e-5, 1e-4, 5e-4}, W_L2: {5e-6, 1e-5, 5e-5}, W_sign: {0, 0.01, 0.05}. Stronger noise may need different regularization to suppress noise-induced W errors. |
| 3     | **Training volume re-tune**    | `data_augmentation_loop`, `n_epochs`                                       | DAL: {35, 70, 140}, n_epochs: {2, 4}. Noisier data may need more training to average out noise, or less if noise already provides implicit regularization. |
| 4     | **Architecture + batch_size**  | `hidden_dim`, `embedding_dim`, `batch_size`                                | hidden_dim: {48, 64, 80}, embedding_dim: {2, 4}, batch_size: {2, 4, 8}. Noisier signals may benefit from larger hidden_dim or batch averaging. |
| 5     | **Monotonicity + f_theta**     | `coeff_g_phi_diff`, `coeff_f_theta_diff`, `coeff_f_theta_msg_diff`         | g_phi_diff: {500, 1000, 1500}, f_theta_diff: {0, 10, 100}, f_theta_msg_diff: {0, 10, 100}. Linear model: f_theta_msg_diff is well-motivated. |
| 6     | **Free exploration I**         | Any parameter                                                              | Consolidate best from blocks 1-5, test novel combinations                                                        |
| 7     | **Free exploration II**        | Any parameter                                                              | Continue ceiling-breaking attempts                                                                               |
| 8     | **Final robustness**           | None (robustness test)                                                     | 4-seed robustness test of best config from blocks 1-7                                                            |

### Noise-specific considerations

- **Dose-response is the central question**: Noise=0.05 proved degeneracy-breaking works (0.918 best, 0.371 mean vs 0.02 clean). Now we test whether 10x more noise (0.5) continues to improve, plateaus, or reverses. This maps the noise dose-response curve.
- **Signal-to-noise tradeoff**: At sigma=0.5, noise power may rival or exceed signal power. The GNN must separate W-driven dynamics from noise — regularization and training volume may need to compensate.
- **Spectral radius monitoring is critical**: The true W has spectral_radius=0.9. Under strong noise, if the learned spectral radius diverges, W recovery fails. Monitor this diagnostic closely.
- **g_phi_diff may be less important**: g_phi learns identity for this linear model — lower values (500) may suffice, freeing capacity for W learning.
- **f_theta_msg_diff is physically motivated**: f_theta should be monotonically increasing in message (-v + msg + stim). Values up to 100 may help.
- **W_L1 sparsity may help more under strong noise**: Many populations are zeroed in the true W — L1 can enforce this structure even when strong noise obscures it.
- **Batch averaging**: Larger batch_size may help average out noise-induced gradient variance.

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

## Winner Config (COMPULSORY)

**At every block boundary**, you MUST save the current best config as a winner file.
This is a COMPULSORY task — do not skip it.

1. Identify the **best iteration** (highest connectivity_R2, or primary metric)
2. Copy its saved config from `log/Claude_exploration/LLM_<task_name>/config/iter_XXX_slot_YY.yaml`
3. Save it to `config/zebrafish_oculomotor/zebrafish_oculomotor_noise05_winner.yaml` with a YAML comment header:

```yaml
# Winner config: zebrafish_oculomotor_noise05_winner.yaml
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

Destination: `config/zebrafish_oculomotor/zebrafish_oculomotor_noise05_winner.yaml`

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

1. **Full Log (append-only)**: `zebrafish_oculomotor_noise05_Claude_analysis.md`
   - Append every iteration's log entry (4 entries per batch)
   - Never read — human record only

2. **Working Memory (read + update every batch)**: `zebrafish_oculomotor_noise05_Claude_memory.md`
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

Example: "Strong noise (sigma=0.5) maintains high connectivity_R2 > 0.95 robustly (3/3 iterations, all seeds > 0.93, CV < 2%)" OR "Signal-to-noise tradeoff emerges at sigma=0.5; R2 plateaus or declines despite stronger noise (3/3 iterations, all seeds < 0.92)"

### What to Add to Open Questions

- Patterns observed 1-2 times
- Seed-dependent effects (works for some seeds but not others)
- Contradictions between iterations
- Theoretical predictions not yet verified

Example: "Is there an optimal noise level between 0.05 and 0.5? Dose-response curve unclear — may peak and decline."

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:
- State the original hypothesis
- State the contradicting evidence (iteration number, metrics)
- State what was learned from the falsification
- Propose a revised hypothesis if applicable

Example: "Hypothesis: 'Stronger noise always improves identifiability' — Falsified by iter 3 (noise=0.5 showed 50% failure rate, mean CV=12%). Revised: 'Noise helps but has SNR limits; strong noise can create new failure modes.'"

## Start Call

When prompt says `PARALLEL START`:

- Read base config — the parent noise=0.05 best config + noise_model_level=0.5 IS the baseline.
- Block 1 is a **robustness test**: all 4 slots use the same config (different seeds).
- Hypothesis: "Stronger process noise (0.5) maintains or improves connectivity_R2 above the noise=0.05 baseline (0.918 best, 0.371 mean)"

---

# Working Memory Structure

```markdown
# Working Memory: zebrafish_oculomotor_noise05

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
