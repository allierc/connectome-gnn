# FlyVis GNN Training Exploration — flyvis_noise_005_stride_5

## Goal

Test whether **recurrent (BPTT) training with temporal stride=5** improves connectivity recovery
compared to 1-step training on the same flyvis noise_005 dataset.

The core scientific question is:
> **Can BPTT through k=5 Euler steps reduce the effective null space of the inverse problem,
> yielding higher connectivity_R2 than standard 1-step training?**

From the pseudoinverse analysis, standard 1-step training yields global connectivity_R2 ≈ 0.71 (noise_005).
The hypothesis is that BPTT gradients create nonlinear constraints across 5 timesteps that break the
per-neuron null-space structure, especially for high-degree neurons.

**Primary metric**: `connectivity_R2` — must exceed the stride-1 baseline (0.95–0.98).
**Comparison target**: stride-1 winner config achieves connectivity_R2 ≈ 0.980 (CV < 1%).
**Hard floor**: connectivity_R2 > 0.90 on all 4 seeds.
**Secondary metrics**: `tau_R2`, `V_rest_R2`, `cluster_accuracy`, `rollout_pearson`.

## Scientific Context

**Why recurrent training might help:**
The inverse problem — recovering W from voltage trajectories — suffers from a per-neuron null space:
if neurons j and k are always co-active (same cell type), their incoming weights W_ij and W_ik
are unidentifiable from 1-step observations. With stride=5, each training sample requires the model
to predict v(t+5dt) from v(t), unrolling 5 Euler steps. This creates gradient signal that is
sensitive to the *interaction* of weights over time — two weight configurations that look identical
at t+1dt may diverge at t+5dt, providing discriminating signal.

**Risk: gradient instability.** BPTT through k steps multiplies the gradient by the Jacobian k
times. If the Jacobian has eigenvalues > 1 (which happens near the spectral radius ≈ 1.72 for
flyvis), gradients can explode. Learning rates must be scaled down and gradient clipping may be
required.

**Risk: fewer effective training pairs.** With stride=5 and data_augmentation_loop=35, each epoch
sees T/5 × 35 = 448,000 training pairs (vs T × 35 = 2,240,000 for stride=1). This is a 5×
reduction in the number of weight update steps. Larger data_augmentation_loop or more epochs may
be needed to compensate.

## CRITICAL: Data is PRE-GENERATED at startup (fixed across iterations)

At startup, data is generated **once** for all 4 slots with **different random seeds** (one per slot).
These datasets are **reused across all iterations** — data is NOT re-generated each iteration.
Both `simulation.seed` and `training.seed` are **forced by the pipeline** — DO NOT modify them.

Seed formula (set automatically by GNN_LLM.py):
- `simulation.seed = 1000 + slot` (controls data generation — fixed at startup, slot 0–3)
- `training.seed = iteration * 1000 + slot + 500` (controls weight init & training randomness)

**Seed robustness testing**: To re-generate data with new seeds, set `claude.test_robustness_seed: true`
in all 4 slot configs. The pipeline resets the flag automatically after one batch.

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i(t)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors)
- DAVIS visual input, **noise_model_level=0.05**
- 64,000 frames, delta_t=0.02
- **Spectral radius ≈ 1.72** — relevant for gradient explosion risk in BPTT

## Recurrent Training Mechanism

With `recurrent_training: true` and `time_step: k`:

1. Sample a random frame index `t` (aligned to nearest multiple of k)
2. Starting from the **true** `v(t)`, unroll k Euler steps using the **model's own predictions**:
   ```
   v̂(t+1) = v(t)   + dt * GNN(v(t),   W, g_phi, f_theta)
   v̂(t+2) = v̂(t+1) + dt * GNN(v̂(t+1), W, g_phi, f_theta)
   ...
   v̂(t+k) = v̂(t+k-1) + dt * GNN(v̂(t+k-1), W, g_phi, f_theta)
   ```
3. Loss = `||v̂(t+k) - v(t+k)||² / (dt * k)`
4. Backpropagate through all k steps (BPTT)

**Note**: `multi_start_recurrent` cannot be used with `time_step=5` because all launch frames must be
multiples of `time_step`. This leaves no room for "multiple starts targeting the same frame T" at
different offsets — not applicable here.

**Gradient scaling**: BPTT through k steps multiplies the gradient by the Jacobian k times.
→ **Rule of thumb**: scale lr_W down by factor ~k compared to stride-1 baseline.

## Explorable Parameters

### Core Recurrent Parameters

| Parameter | Default | Description |
|---|---|---|
| `time_step` | 5 | Stride: unroll this many Euler steps per training sample |
| `recurrent_training` | true | Must stay true (defines this experiment) |
| `recurrent_training_start_epoch` | 0 | Epoch fraction at which to switch to recurrent; 0 = immediate (Block 5) |
| `noise_recurrent_level` | 0.0 | Add Gaussian noise at each recurrent step (prevents error compounding) |

### Learning Rates (scale down from stride-1 by ~k)

| Parameter | Stride-1 baseline | Stride-5 starting point |
|---|---|---|
| `lr_W` | 0.0009 | 0.0002 (÷5 rule of thumb) |
| `lr` | 0.0018 | 0.0004 |
| `lr_embedding` | 0.002325 | 0.0005 |

### Gradient Stability

| Parameter | Default | Description |
|---|---|---|
| `grad_clip_W` | 0.0 | Gradient clipping for W (0 = disabled); try 0.1–1.0 |
| `lr_scheduler` | none | `cosine_warm_restarts` or `linear_warmup_cosine` — stabilize BPTT convergence (Block 8) |
| `lr_scheduler_T0` | 1000 | Period for cosine warm restarts (if scheduler enabled) |
| `lr_scheduler_warmup_iters` | 100 | Linear warmup before cosine (if scheduler enabled) |

### Training Volume (compensate for stride-5 reduction in steps)

| Parameter | Default | Description |
|---|---|---|
| `data_augmentation_loop` | 35 | Increase to compensate for 5× fewer step-pairs |
| `batch_size` | 4 | Reduce if OOM due to BPTT memory; increase for stability |
| `n_epochs` | 1 | Stride-5 may benefit from n_epochs=2 |

### Regularization (may need rebalancing with recurrent signal)

| Parameter | Default | Description |
|---|---|---|
| `coeff_g_phi_diff` | 750 | Monotonicity of g_phi — critical for rollout stability |
| `coeff_W_L1` | 0.00015 | L1 on W — recurrent training may prefer lighter regularization |
| `coeff_W_L2` | 1.5e-6 | L2 on W |
| `coeff_g_phi_weight_L1` | 0.28 | L1 on g_phi weights |

### Architecture

| Parameter | Default | Note |
|---|---|---|
| `embedding_dim` | 2 | Keep fixed — changing requires updating input_size, input_size_update |
| `hidden_dim` | 80 | g_phi width |
| `hidden_dim_update` | 80 | f_theta width |

**CRITICAL — coupled parameters**: if you change `embedding_dim`:
- `input_size = 1 + embedding_dim`
- `input_size_update = 3 + embedding_dim`

## Regularization Annealing

All L1/L2 coefficients are multiplied by `(1 - exp(-rate * epoch))`.
With `n_epochs=1`, **all L1/L2 regularizers are inactive** (annealing multiplier = 0).
→ Use `n_epochs: 2` + halve `data_augmentation_loop` OR set `regul_annealing_rate: 0`.

Non-annealed: `coeff_g_phi_diff`, `coeff_g_phi_norm`, `coeff_f_theta_msg_diff` — active from epoch 0.

## Parallel Mode — 4 Slots Per Batch

All 4 slots run the **same config** with different random seeds (assigned automatically).
Use all 4 slots for seed robustness testing.

**Robustness classification:**

- **Stable-Robust**: all 4 seeds connectivity_R2 > 0.9 AND CV < 3% — **TARGET**
- **Robust**: all 4 seeds > 0.9, CV 3-5%
- **Partially robust**: 2-3 seeds > 0.9
- **Fragile**: 0-1 seeds > 0.9 — reject
- **DISQUALIFIED**: any seed < 0.87

## Block Partition

| Block | Focus | Key parameters |
|---|---|---|
| 1 | **Baseline comparison** | stride-5 default vs stride-1 winner — does BPTT help at all? |
| 2 | **LR rescaling** | lr_W, lr, lr_emb — find stable range for BPTT gradient magnitude |
| 3 | **Noise injection** | noise_recurrent_level = 0.0, 0.01, 0.03, 0.1 — prevent error compounding in BPTT rollout |
| 4 | **Gradient clipping** | grad_clip_W = 0, 0.1, 0.5, 1.0 — stabilize BPTT |
| 5 | **Curriculum** | recurrent_training_start_epoch = 0, 0.1, 0.3, 0.5 fraction — warm up with stride=1 first |
| 6 | **Training volume** | data_augmentation_loop, n_epochs — compensate for 5× fewer step-pairs |
| 7 | **Regularization** | coeff_g_phi_diff, coeff_W_L1 — rebalance with recurrent gradient signal |
| 8 | **LR scheduler** | lr_scheduler = cosine_warm_restarts vs none, warmup tuning — stabilize BPTT convergence |
| 9 | **Combined best** | Best parameters from blocks 1–8 |
| 10 | **Validation** | Best stride-5 config vs stride-1 winner — final comparison |

### Block Focus Notes

**Block 1** is the most important: if stride-5 does NOT improve over stride-1 baseline (0.980),
the null-space hypothesis is falsified and the exploration should pivot to understanding why.

**Block 3** tests `noise_recurrent_level`: injecting small Gaussian noise at each BPTT step prevents
the model from overfitting to the exact noiseless rollout trajectory and may reduce error compounding
across the 5 unrolled steps. Try 0.0 (baseline), 0.01, 0.03, 0.1.

**Block 5** tests curriculum learning: start training with stride=1 to learn the basic dynamics,
then switch to stride=5 to refine with BPTT. `recurrent_training_start_epoch` controls the switch.
Note: with `n_epochs=1`, this parameter has no effect — needs `n_epochs ≥ 2`.

**Block 8** tests LR scheduler: with BPTT gradients being noisier than 1-step gradients, a cosine
warm restart schedule (`lr_scheduler: cosine_warm_restarts`) or linear warmup + cosine decay
(`linear_warmup_cosine`) may stabilize convergence better than a constant LR.

## Variable Names

- **`{base_config_name}`**: `flyvis_noise_005_stride_5`
- **`{llm_task_name}`**: `flyvis_noise_005_stride_5_Claude`

**Config file paths:**
- `config/fly/flyvis_noise_005_stride_5_Claude_00.yaml` through `_03.yaml`
- `config/fly/flyvis_noise_005_stride_5_winner.yaml`

## File Structure

### 1. Full Log (append-only)
**File**: `flyvis_noise_005_stride_5_Claude_analysis.md`

### 2. Working Memory (read + update every batch)
**File**: `flyvis_noise_005_stride_5_Claude_memory.md`

### 3. User Input
**File**: `user_input.md`

## Iteration Workflow (every batch)

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

**Metrics from `analysis.log`:**
- `connectivity_R2`: R² of learned vs true W (PRIMARY — compare to stride-1 baseline 0.980)
- `tau_R2`, `V_rest_R2`, `cluster_accuracy`, `rollout_pearson`
- `training_time_min`: BPTT is ~k× slower per iteration than stride-1

**BPTT-specific observations to track:**
- Did training diverge (NaN loss, connectivity_R2 ≈ 0)?
- Is connectivity_R2 above or below the stride-1 baseline (0.980)?
- Is training time scaling as expected (~5× longer per epoch vs stride-1)?
- Are tau_R2 and V_rest_R2 improved, degraded, or unchanged vs stride-1?

### Step 3: Write Log Entry

```
## Iter N: [robust/partially robust/fragile] [BETTER/WORSE/SAME vs stride-1]

Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: time_step=K, lr_W=X, lr=Y, lr_emb=Z, grad_clip_W=G, data_aug=D, epochs=E
Slot 0: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, sim_seed=S, train_seed=T
Slot 1: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, sim_seed=S, train_seed=T
Slot 2: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, sim_seed=S, train_seed=T
Slot 3: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%, min=W, max=V
vs stride-1 baseline: [+X.XXX / -X.XXX] (absolute difference in mean connectivity_R2)
Stability: [Stable-Robust / Robust / Partially robust / Fragile / DISQUALIFIED]
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive] — [explanation]
BPTT note: [any gradient explosion, NaN, or unusual training dynamics]
Next: parent=P
```

### Step 4: Acknowledge User Input

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Winner Config (COMPULSORY at every block boundary)

Save to `config/fly/flyvis_noise_005_stride_5_winner.yaml` with header:

```yaml
# Winner config: flyvis_noise_005_stride_5_winner.yaml
# Source: iter_XXX_slot_YY (connectivity_R2 = X.XXX)
# vs stride-1 baseline: +/-X.XXX (stride-5 [better/worse/same])
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [narrative on what recurrent training contributed]
#   - [key hyperparameter choices and why they matter for BPTT stability]
#
# Metrics:
#   connectivity_R2: X.XXX (best single seed)
#   robust_mean:     X.XXX +/- X.XXX (N seeds, CV=X.X%)
#   tau_R2:          X.XXX
#   V_rest_R2:       X.XXX
#   cluster_accuracy: X.XXX
#
# Key differences from stride-1 config:
#   - time_step: 1 -> K
#   - lr_W: 0.0009 -> X (scaled for BPTT)
#   - [other changed parameters]
```

## Block Boundaries

1. Update "Paper Summary" in memory.md (both bullet points)
2. Summarize findings in "Previous Block Summaries"
3. Update "Established Principles" (3+ supporting iterations AND cross-seed consistency)
4. Move falsified hypotheses to "Falsified Hypotheses"
5. Clear "Current Block"
6. **Compare best stride-5 result to stride-1 baseline (0.980)** — is BPTT helping?

## Known Results (prior experiments)

- **Stride-1 winner** (`flyvis_noise_005`): connectivity_R2 ≈ 0.980, CV < 1%, tau_R2 ≈ 0.984, V_rest_R2 ≈ 0.427
- Pseudoinverse analysis (noise_005): global connectivity_R2 = 0.71 (linear, no dynamics)
- Pseudoinverse breakdown: high-degree neurons (61-208 inputs) reach R²=0.83; low-degree (1-5) only 0.51
- BPTT rule of thumb: scale lr down by ~k to maintain similar gradient magnitude
- Spectral radius of true W ≈ 1.72 — Jacobian eigenvalues > 1 are possible → exploding gradients
- `coeff_g_phi_diff` is critical for rollout stability; too low causes diverging rollouts

## Start Call

When prompt says `PARALLEL START`:

- Read base config `config/fly/flyvis_noise_005_stride_5_Claude_00.yaml`
- Set all 4 configs identically to the baseline stride-5 config
- **Initial hypothesis**: "BPTT with time_step=5 achieves higher connectivity_R2 than stride-1 (0.980) by breaking per-neuron null-space degeneracy"
- **Null hypothesis**: "Stride-5 achieves similar or lower connectivity_R2 than stride-1 due to fewer training pairs and gradient instability"
- Write both hypotheses to working memory
- Block 1 tests the null hypothesis — do not change hyperparameters yet

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_stride_5

## Paper Summary (update at every block boundary)

- **GNN + recurrent training**: [pending]
- **Null-space reduction via BPTT**: [pending]

## Knowledge Base

### vs Stride-1 Comparison Table

| Iter | time_step | LR scale | conn_R2 (mean±std) | CV% | min | vs baseline | Stability | Note |
|------|-----------|----------|--------------------|-----|-----|-------------|-----------|------|
| 1    | 5         | default  | ?                  | ?   | ?   | ?           | ?         | baseline |

### Established Principles

[Confirmed patterns — require 3+ supporting iterations AND cross-seed consistency]

### Falsified Hypotheses

### Open Questions

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest→newest.**

---

## Current Block (Block N)

### Block Info

Focus: [parameter subspace]
Iterations: M to M+n_iter_block

### Current Hypothesis

**Hypothesis**: [specific, testable prediction]
**Rationale**: [prior evidence]
**Test**: [config change]
**Expected outcome**: [what supports vs falsifies]
**Stability constraint**: CV < 3%, all seeds > 0.90
**vs stride-1**: expected [better/worse/same] by [magnitude]
**Status**: untested / supported / falsified / revised

### Iterations This Block

### Emerging Observations
**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
