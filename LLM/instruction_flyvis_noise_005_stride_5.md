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

## Scientific Method

This exploration follows a strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Based on available data (metrics, seed variance, prior results), form a specific, testable hypothesis about which parameter controls robustness
2. **Design experiment**: Choose a mutation that specifically tests the hypothesis — change **exactly ONE parameter at a time**
3. **Run training**: The experiment runs across 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use both metrics AND cross-seed variance to evaluate whether the hypothesis was supported or contradicted
5. **Update understanding**: Revise hypotheses based on evidence. A falsified hypothesis is valuable information. 

**CRITICAL**: You can only hypothesize. Only training results can validate or falsify your hypotheses. Never assume a hypothesis is correct without experimental evidence.

**Evidence hierarchy:**

| Level            | Criterion                                       | Action                 |
| ---------------- | ----------------------------------------------- | ---------------------- |
| **Established**  | Consistent across 3+ iterations AND 4/4 seeds   | Add to Principles      |
| **Tentative**    | Observed 1-2 times or inconsistent across seeds | Add to Open Questions  |
| **Contradicted** | Conflicting evidence across iterations/seeds    | Note in Open Questions |

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
- **Spectral radius ≈ 1.72** — property of the TRUE GT connectivity W. Do NOT add spectral radius regularization — it would push the learned W away from the ground truth answer. The effective Jacobian is naturally dampened by the g_phi^2 nonlinearity, g_phi_norm=0.1, and g_phi_diff=9000 (already the dominant levers). The real bottleneck is gradient NOISE from 5× fewer training pairs per epoch, not gradient explosion per se.

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

| Parameter                        | Default | Description                                                            |
| -------------------------------- | ------- | ---------------------------------------------------------------------- |
| `time_step`                      | 5       | Stride: unroll this many Euler steps per training sample               |
| `recurrent_training`             | true    | Must stay true (defines this experiment)                               |
| `recurrent_training_start_epoch` | 0       | **Must stay 0** — immediate recurrent training from epoch 0            |
| `noise_recurrent_level`          | 0.0     | Add Gaussian noise at each recurrent step (prevents error compounding) |

### Learning Rates (scale down from stride-1 by ~k)

| Parameter      | Stride-1 baseline | Stride-5 starting point   |
| -------------- | ----------------- | ------------------------- |
| `lr_W`         | 0.0009            | 0.0002 (÷5 rule of thumb) |
| `lr`           | 0.0018            | 0.0004                    |
| `lr_embedding` | 0.002325          | 0.0005                    |

### Gradient Stability

| Parameter                   | Default | Description                                                                             |
| --------------------------- | ------- | --------------------------------------------------------------------------------------- |
| `grad_clip_W`               | 0.0     | Gradient clipping for W (0 = disabled); try 0.1–1.0                                     |
| `lr_scheduler`              | none    | `cosine_warm_restarts` or `linear_warmup_cosine` — stabilize BPTT convergence (Block 6) |
| `lr_scheduler_T0`           | 1000    | Period for cosine warm restarts (if scheduler enabled)                                  |
| `lr_scheduler_warmup_iters` | 100     | Linear warmup before cosine (if scheduler enabled)                                      |

### Regularization (may need rebalancing with recurrent signal)

| Parameter               | Default | Description                                                    |
| ----------------------- | ------- | -------------------------------------------------------------- |
| `coeff_g_phi_diff`      | 750     | Monotonicity of g_phi — critical for rollout stability         |
| `coeff_W_L1`            | 0.00015 | L1 on W — recurrent training may prefer lighter regularization |
| `coeff_W_L2`            | 1.5e-6  | L2 on W                                                        |
| `coeff_g_phi_weight_L1` | 0.28    | L1 on g_phi weights                                            |

### Architecture

| Parameter           | Default | Note                                                                  |
| ------------------- | ------- | --------------------------------------------------------------------- |
| `embedding_dim`     | 2       | Keep fixed — changing requires updating input_size, input_size_update |
| `hidden_dim`        | 80      | g_phi width                                                           |
| `hidden_dim_update` | 80      | f_theta width                                                         |

**CRITICAL — coupled parameters**: if you change `embedding_dim`:

- `input_size = 1 + embedding_dim`
- `input_size_update = 3 + embedding_dim`

## Regularization Annealing

All L1/L2 coefficients are multiplied by `(1 - exp(-rate * epoch))`.
With `n_epochs=1` (fixed in this experiment), **all L1/L2 regularizers are inactive** (annealing multiplier = 0) unless `regul_annealing_rate: 0` is set.

Non-annealed: `coeff_g_phi_diff`, `coeff_g_phi_norm`, `coeff_f_theta_msg_diff` — active from epoch 0.

> **YAML rule**: Always wrap the `description` field value in double quotes — colons inside unquoted YAML strings cause parse errors (e.g., `description: "Block 7 Slot 1: testing W_L2"`).

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

| Block | Focus                   | Key parameters                                                                           |
| ----- | ----------------------- | ---------------------------------------------------------------------------------------- |
| 1     | **Baseline comparison** | stride-5 default vs stride-1 winner — does BPTT help at all?                             |
| 2     | **LR rescaling**        | lr_W, lr, lr_emb — find stable range for BPTT gradient magnitude                         |
| 3     | **Noise injection**     | noise_recurrent_level = 0.0, 0.01, 0.03, 0.1 — prevent error compounding in BPTT rollout |
| 4     | **Gradient clipping**   | grad_clip_W = 0, 0.1, 0.5, 1.0 — stabilize BPTT                                          |
| 5     | **Regularization**      | coeff_g_phi_diff, coeff_W_L1 — rebalance with recurrent gradient signal                  |
| 6     | **LR scheduler**        | lr_scheduler = cosine_warm_restarts vs none, warmup tuning — stabilize BPTT convergence  |
| 7     | **Combined best**       | Best parameters from blocks 1–6                                                          |
| 8     | **Validation**          | Best stride-5 config vs stride-1 winner — final comparison                               |
| 9     | **Remaining axes**      | dale_law, coeff_f_theta_msg_diff, coeff_W_L2 — last untested single-parameter knobs     |
| 10    | **Batch size**          | batch_size ∈ {1, 2, 4} at half/quarter DAL — BPTT has no waterbed problem; smoother gradients may help |
| 11    | **Best-of combination** | Combine Block 9 winners with batch_size winner; if no winner from 9–10, escalate to warm initialization |

**Block 9 expectation note**: These are the last 3 untested single-parameter axes. No individual parameter is expected to close the −0.609 gap (vs stride-1). Criterion for moving on: if no axis gives > +0.05 conn_R2 over the Block 8 champion (0.371), single-parameter tuning has reached its limit. Move to Block 10 (batch size) regardless of Block 9 outcome.

**Block 10 rationale**: Unlike SIREN (which requires bs=1 due to the waterbed problem), BPTT training does not share model parameters across time — each batch item is an independent trajectory window. Larger batch sizes give smoother gradient estimates for the same wall time (bs=2 / DAL=40 ≈ same as bs=1 / DAL=80 in compute). Reference: bs=1, DAL=80, n_epochs=2 → ~103 min. Scale DAL inversely with batch_size to maintain the same budget.

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
Config: time_step=5, lr_W=X, lr=Y, lr_emb=Z, grad_clip_W=G, noise_rec=N, lr_sched=S
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
- Spectral radius of true W ≈ 1.72 — this is a fixed property of the GT system, NOT a training artifact. Do NOT regularize spectral radius: it would push W away from the correct answer. Effective Jacobian is dampened by g_phi^2 + g_phi_norm=0.1 + g_phi_diff=9000 (already exploited). The real bottleneck is gradient NOISE from 5× fewer training pairs/epoch vs stride-1.
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

| Iter | time_step | LR scale | conn_R2 (mean±std) | CV% | min | vs baseline | Stability | Note     |
| ---- | --------- | -------- | ------------------ | --- | --- | ----------- | --------- | -------- |
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
