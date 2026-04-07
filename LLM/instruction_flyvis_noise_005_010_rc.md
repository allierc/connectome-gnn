# FlyVis GNN — Recurrent Training as Implicit Denoising (flyvis_noise_005_010_rc)

## Goal

Explore **recurrent training** as a method for **implicit denoising through the gradient** when training on noisy FlyVis data (dynamics noise 0.05 + measurement noise 0.10).

The GNN is **fine-tuned from a pre-trained baseline checkpoint** (`flyvis_noise_005_010`) using multi-step rollout loss. By integrating multiple steps autoregressively during training, the model must produce self-consistent trajectories — noise-fitting artifacts that accumulate across steps get penalized, acting as an implicit low-pass filter.

Primary metric: **connectivity_R2** (R² between learned W and ground-truth W), evaluated on noise-free cross-test data.

Target: **connectivity_R2 > 0.80** (the baseline without recurrent training achieves 0.739).

## Scientific Context: Denoising Through Gradient

Under high measurement noise (0.10), the derivative targets used for one-step training have noise std ~ 7.07 (from `measurement_noise * sqrt(2) / dt`). This overwhelms the clean signal. Rather than preprocessing the targets (Wiener filter, wavelet — which hurt connectivity), we denoise **through the training procedure itself**:

1. **Recurrent rollout** (`recurrent_training + time_step`): The model integrates `time_step` steps forward and compares to ground truth at frame `k + time_step`. Multi-step consistency rejects per-frame noise.
2. **Batch averaging** (`batch_size`): Larger batches average out noise across samples in each gradient step.
3. **Consecutive batching** (`consecutive_batch`): Batches of temporally consecutive frames exploit temporal correlation for smoother gradients.
4. **Multi-start rollout** (`multi_start_recurrent`): Randomize rollout start points within each batch window, preventing memorization of noise patterns tied to specific initial frames.

Prior experiments (Wiener denoising, wavelet) showed that explicit target denoising improves rollout but **degrades connectivity**. Recurrent training may achieve both — better dynamics AND better connectivity — by operating on the loss landscape rather than the data.

## Training Mode: Fine-Tuning from Baseline

All training in this exploration **continues from the pre-trained baseline checkpoint**:

```
pretrained_model: "./log/fly/flyvis_noise_005_010/models/best_model_with_0_graphs_1.pt"
```

This checkpoint was trained for 2 epochs on noisy data with standard one-step loss. Recurrent fine-tuning refines this model by adding multi-step consistency.

**DO NOT change** `pretrained_model`. All slots fine-tune from the same checkpoint.
**DO NOT change** `n_epochs` — keep at 2.

## Data

Training data is **pre-generated and fixed** (`generate_data: false`). All slots use the same `flyvis_noise_005_010` dataset. The only source of variation across slots is `training.seed` (weight initialization randomness from checkpoint + training stochasticity).

**DO NOT change** simulation parameters, dataset, or test_dataset.

## Noise Model

Two independent noise sources in the training data:

1. **Dynamics noise** (`noise_model_level=0.05`): `v(t+1) = v(t) + dt * f(v, W, I) + epsilon_dyn(t)`, epsilon_dyn ~ N(0, 0.05)
2. **Measurement noise** (`measurement_noise_level=0.10`): `v_obs(t) = v_clean(t) + epsilon_meas(t)`, epsilon_meas ~ N(0, 0.10)

Derivative noise std ~ `0.10 * sqrt(2) / 0.02 ~ 7.07` — very large relative to clean signals.

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i(t)
dv_i/dt = f_theta(v_i, a_i, sum_j W_ij * g_phi(v_j, a_j)^2, I_i)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors)
- DAVIS visual input, 64,000 frames, delta_t=0.02

## GNN Architecture

- **g_phi** (MLP1): Edge message function. Maps (v_j, a_j) -> message. `g_phi_positive=true`.
- **f_theta** (MLP0): Node update function. Maps (v_i, a_i, aggregated_messages, I_i) -> dv_i/dt.
- **Embedding a_i**: learnable low-dimensional embedding per neuron type.

**Do NOT change** architecture parameters (hidden_dim, n_layers, embedding_dim, etc.).

## Explorable Parameters

### Recurrent Training Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `recurrent_training` | true | Enable multi-step rollout loss |
| `time_step` | 5 | Number of autoregressive integration steps per training sample |
| `consecutive_batch` | false | Use temporally consecutive frames in each batch |
| `multi_start_recurrent` | false | Randomize rollout start points within batch window |
| `batch_size` | 6 | Batch size — larger batches average out noise across samples |

### Learning Rates

| Parameter | Default | Description |
| --- | --- | --- |
| `learning_rate_W_start` | 9e-4 | Learning rate for connectivity matrix W |
| `learning_rate_start` | 1.8e-3 | Learning rate for g_phi and f_theta MLPs |
| `learning_rate_embedding_start` | 2.325e-3 | Learning rate for neuron embeddings |

### Regularization (active from epoch 0 since regul_annealing_rate=0)

| Parameter | Default | Description | Annealed? |
| --- | --- | --- | --- |
| `coeff_g_phi_diff` | 1200 | Monotonicity penalty on g_phi | No |
| `coeff_g_phi_norm` | 0.9 | Normalization penalty on g_phi | No |
| `coeff_g_phi_weight_L1` | 0.28 | L1 on g_phi weights | Yes (but rate=0 → active) |
| `coeff_f_theta_weight_L1` | 0.05 | L1 on f_theta weights | Yes (but rate=0 → active) |
| `coeff_f_theta_weight_L2` | 0.001 | L2 on f_theta weights | Yes (but rate=0 → active) |
| `coeff_W_L1` | 5e-4 | L1 sparsity on W | Yes (but rate=0 → active) |
| `coeff_W_L2` | 1.5e-6 | L2 on W | Yes (but rate=0 → active) |
| `data_augmentation_loop` | 30 | Data augmentation multiplier |

**Note**: `regul_annealing_rate=0` disables annealing, so ALL regularizers are active at full strength from epoch 0. Do not change `regul_annealing_rate`.


## Slot Strategy — 4 Different Configs Per Batch

Each batch runs **4 different configurations** simultaneously. Unlike other explorations where all slots test the same config with different seeds, here each slot tests a **different parameter setting** to maximize exploration speed.

The LLM should design 4 related but distinct configs for each batch — for example, sweeping one parameter across 4 values, or testing 4 different combinations.

### Config Files

- Edit all 4 config files: `{name}_00.yaml` through `{name}_03.yaml`
- Each config can be **different** — the LLM decides what each slot tests
- **DO NOT change**: simulation parameters, dataset, test_dataset, pretrained_model, n_epochs, architecture
- **Seeds are set automatically** by the pipeline — DO NOT modify them

## Evaluation

After each training run, the pipeline calls `data_test_flyvis()` which:
1. Loads the best model checkpoint
2. Evaluates connectivity R2, tau R2, V_rest R2 against ground truth
3. Runs rollout on noise-free test data (`test_dataset: fly/flyvis_noise_free`)
4. Reports rollout Pearson r and RMSE

Metrics from `analysis.log`:
- `connectivity_R2`: R2 of learned vs true W (**PRIMARY**)
- `tau_R2`: R2 of learned vs true time constants
- `V_rest_R2`: R2 of learned vs true resting potentials
- `test_R2`: one-step prediction R2
- `rollout_pearson_r`: multi-step rollout correlation on noise-free data
- `training_time_min`: training duration

## Known Results (from manual sweeps)

These results were obtained from prior manual experiments. Use them to inform your exploration — do NOT simply replicate these exact configs, but use them as starting points and baselines.

### Baseline (no recurrent training)

| Model | R2_conn | R2_tau | R2_Vrest |
| --- | --- | --- | --- |
| noise_005_010 (baseline, 1-step) | 0.739 | 0.782 | 0.113 |
| noise_free (gold standard) | 0.924 | 0.844 | 0.079 |

### Recurrent Training — time_step sweep (batch_size=6)

| time_step | R2_conn | R2_tau | R2_Vrest |
| --- | --- | --- | --- |
| 2 | 0.717 | 0.897 | 0.165 |
| 3 | 0.722 | 0.912 | 0.089 |
| 4 | 0.718 | 0.953 | 0.073 |
| 5 | 0.727 | **0.965** | 0.058 |
| 10 | 0.620 | 0.928 | 0.031 |

**Key finding**: tau_R2 peaks at time_step=5 (+23% over baseline). Connectivity stays flat. time_step=10 overshoots — connectivity drops to 0.620.

### Recurrent Training — batch_size sweep (time_step=5)

| batch_size | R2_conn | R2_tau | R2_Vrest |
| --- | --- | --- | --- |
| 2 | 0.641 | 0.942 | 0.011 |
| 4 | 0.682 | 0.964 | 0.034 |
| 10 | 0.757 | 0.961 | 0.110 |
| 16 | **0.768** | 0.962 | 0.145 |

**Key finding**: Batch size strongly improves connectivity (0.641→0.768). tau_R2 is robust across batch sizes. Trend suggests larger batches may help further.

### Consecutive Batching (no recurrent, batch_size sweep)

| batch_size | R2_conn | R2_tau | R2_Vrest |
| --- | --- | --- | --- |
| 2 | 0.732 | 0.758 | 0.100 |
| 4 | 0.750 | 0.788 | 0.108 |
| 8 | 0.742 | 0.787 | 0.113 |
| 16 | 0.741 | 0.787 | 0.116 |

**Key finding**: Consecutive batching alone provides minimal improvement. The temporal correlation does not help without recurrent rollout.

### Self-Distillation (GS — train on rollout data)

| recurrent | time_step | R2_conn | R2_tau | R2_Vrest |
| --- | --- | --- | --- | --- |
| no | — | 0.516 | 0.092 | 0.102 |
| yes | 5 | 0.682 | 0.078 | 0.103 |

**Key finding**: Self-distillation catastrophically destroys tau_R2. Not a viable approach.

### Summary of Established Facts

1. **Recurrent training is the dominant denoising lever** — time_step=5 improves tau_R2 by +23%
2. **Batch size strongly helps connectivity** under recurrent training (0.641→0.768 for bs 2→16)
3. **Consecutive batching alone is ineffective** without recurrent rollout
4. **time_step=10 overshoots** — connectivity drops, suggesting instability at long rollouts
5. **tau_R2 is robust** across batch sizes once recurrent training is enabled
6. **Current champion**: time_step=5, batch_size=16 → R2_conn=0.768, R2_tau=0.962

## Block Structure

### Block 1 (iterations 1-16): Recurrent + Batch Size Sweep

Systematically explore the `time_step × batch_size` space. Each batch tests 4 different (time_step, batch_size) combinations.

Configs to cover:
- time_step=10 with batch_size=2, 4, 10, 16 (can larger batches rescue rc10's connectivity drop?)
- time_step=5 with batch_size=24, 32, 48 (does the bs trend continue?)
- Revalidate time_step=5, batch_size=16 (best known) on H100

Goal: Find the optimal (time_step, batch_size) pair. Establish whether the batch_size trend saturates and whether larger batches rescue time_step=10.

### Block 2 (iterations 17-32): Multi-Start Recurrent

Test `multi_start_recurrent: true` — rollout from random starting frames rather than fixed initial frame.

Configs to cover:
- multi_start with time_step=2, 3, 5, 10 at batch_size=6 (compare to standard recurrent)
- multi_start with best (time_step, batch_size) from Block 1

Goal: Determine whether randomizing rollout start points improves connectivity by preventing noise memorization. Compare directly to standard recurrent results.

### Block 3 (iterations 33-48): Consecutive Batch + Recurrent

Test `consecutive_batch: true` combined with `recurrent_training: true`.

Configs to cover:
- consecutive + recurrent with time_step=5, batch_size=2, 4, 6, 8, 10
- Compare to standard (non-consecutive) recurrent at same settings

Goal: Test whether consecutive batching amplifies the recurrent denoising effect. The hypothesis: consecutive frames have correlated noise, so recurrent rollout over consecutive frames produces smoother gradient updates.

### Block 4 (iterations 49-64): Combined Best + Consecutive-Recurrent Sweep

Combine consecutive_batch + recurrent at optimal (time_step, batch_size) from Block 1.

Additional configs:
- consecutive + multi_start + recurrent (triple combination)
- Sweep batch_size for cb_rc at the best time_step

Goal: Find the best combination of all recurrent approaches.

### Block 5+ (iterations 65+): Open — Learning Rates and Regularization

Explore learning rates and regularization for the best recurrent config found in Blocks 1-4.

Parameters to investigate:
- **Learning rates**: Lower LR may improve stability for long rollouts. Try lr_W in [3e-4, 5e-4, 9e-4], lr in [5e-4, 1e-3, 1.8e-3]
- **coeff_g_phi_diff**: May need different value for recurrent training (try 600, 1200, 2400)
- **coeff_W_L1**: Sparsity regularization interaction with recurrent loss (try 0, 2.5e-4, 5e-4, 1e-3)
- **data_augmentation_loop**: Trade off with batch_size — is it better to have large batches with fewer augmentation passes or small batches with more passes?

Goal: Polish the best recurrent config to maximize connectivity_R2.

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots — each with different config)

For each slot, report:
- Config: time_step, batch_size, consecutive_batch, multi_start_recurrent, and any other changed params
- connectivity_R2, tau_R2, V_rest_R2, rollout_pearson_r, training_time_min

Compare results across the 4 configs within the batch. Identify which parameter settings help and which hurt.

### Step 3: Write Log Entries

```
## Iter N
Hypothesis tested: "[quoted hypothesis]"
Slot 0: config=[params] → conn_R2=X, tau_R2=Y, Vrest_R2=Z, rollout_r=W, time=T min
Slot 1: config=[params] → conn_R2=X, tau_R2=Y, Vrest_R2=Z, rollout_r=W, time=T min
Slot 2: config=[params] → conn_R2=X, tau_R2=Y, Vrest_R2=Z, rollout_r=W, time=T min
Slot 3: config=[params] → conn_R2=X, tau_R2=Y, Vrest_R2=Z, rollout_r=W, time=T min
Best slot: [which] with conn_R2=X
Verdict: [supported/falsified/inconclusive]
Next: [what to test in next batch]
```

## Winner Config (COMPULSORY)

**At every block boundary**, you MUST save the current best config as a winner file.
This is a COMPULSORY task — do not skip it.

1. Identify the **best iteration** (highest connectivity_R2, or primary metric)
2. Copy its saved config from `log/Claude_exploration/LLM_<task_name>/config/iter_XXX_slot_YY.yaml`
3. Save it to `config/fly/flyvis_noise_005_010_rc_winner.yaml` with a YAML comment header:

```yaml
# Winner config: flyvis_noise_005_010_rc_winner.yaml
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

Destination: `config/fly/flyvis_noise_005_010_rc_winner.yaml`

### Step 4: Acknowledge User Input

### Step 5: Design Next 4 Configs

Based on results, design 4 configs for the next batch. Each config should test a specific hypothesis. Document what each slot tests and why.

## File Structure

### 1. Full Log (append-only): `{llm_task_name}_analysis.md`
### 2. Working Memory (read + update): `{llm_task_name}_memory.md`
### 3. User Input: `user_input.md`

## Block Boundaries

1. Update "Paper Summary" — focus on recurrent training as denoising
2. Summarize findings in "Previous Block Summary"
3. Update "Established Principles"
4. Move falsified hypotheses to "Falsified Hypotheses"
5. Clear "Current Block"
6. Note best config found so far

## Knowledge Base Guidelines

### What to Add to Established Principles

A principle must satisfy ALL of:

1. Observed consistently across **3+ iterations**
2. Consistent across **all 4 slots** (not just mean, but low variance)
3. States a **causal relationship** (not just a correlation)

### What to Add to Open Questions

- Patterns observed 1-2 times
- Slot-dependent effects (works for some slots but not others)
- Contradictions between iterations
- Theoretical predictions not yet verified

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:

1. State the original hypothesis
2. State the contradicting evidence (iteration number, metrics)
3. State what was learned from the falsification
4. Propose a revised hypothesis if applicable

## Start Call

When prompt says `PARALLEL START`:

- Read base config — note recurrent_training=true, pretrained_model set, n_epochs=2
- Review the Known Results section above
- Design 4 initial configs for Block 1 (recurrent + batch size sweep)
- **Initial hypothesis**: "Increasing batch_size beyond 16 with time_step=5 will continue to improve connectivity_R2 (extrapolating the 0.641→0.768 trend), while time_step=10 with larger batches will recover from its connectivity drop (0.620) to match or exceed time_step=5"
- Set all 4 slot configs with different (time_step, batch_size) combinations

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_010_rc

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Results Comparison Table

| Iter | Slot | time_step | batch_size | consec | multi_start | aug_loop | LR_W | coeff_g_phi_diff | R2_conn | R2_tau | R2_Vrest | rollout_r | time_min |
| ---- | ---- | --------- | ---------- | ------ | ----------- | -------- | ---- | ---------------- | ------- | ------ | -------- | --------- | -------- |

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

## Current Block (Block N)

### Block Info

### Current Hypothesis

**Hypothesis**: [specific, testable prediction]
**Rationale**: [why]
**Test**: [what 4 configs test this]
**Expected outcome**: [what supports vs falsifies]
**Status**: untested / supported / falsified / revised

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```

## UCB and Exploration Tree

UCB scores track exploration quality based on **connectivity_R2**.

## Failed Slots

A slot failure indicates the config is unstable — note which parameter caused it and avoid that region of the parameter space.
