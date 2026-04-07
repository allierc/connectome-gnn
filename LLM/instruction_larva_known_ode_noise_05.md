# Drosophila Larva — Known ODE Exploration (Noise 0.5)

## Goal

Maximize **connectivity_R2** (PRIMARY) for the **Drosophila larva two-population motor model** using the **known_ode model** under **strong intrinsic noise (sigma=0.5)**.

The known_ode model uses the **exact activation function**: `g_phi = softplus(v + b)` and **exact dynamics**: `dv/dt = (-v + g * msg + I)`. All parameters are **learned from data**: W (synaptic weights), g (gain per neuron), bias per neuron. This is an **upper bound** on what is achievable with perfect structural knowledge.

**Starting hypothesis**: "Known ODE + noise=0.5 achieves connectivity_R2 > 0.3 robustly (significant degradation from noise=0.05 expected)"

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned W and ground-truth W
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free data

Informational: onestep_pearson, training_time_min.

**NOTE**: V_rest_R2 is not applicable. cluster_accuracy is not applicable (no learned embeddings). tau_R2 is not applicable (tau is fixed).

## Scientific Method

This exploration follows a strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Based on available data (metrics, seed variance, prior results), form a specific, testable hypothesis about which parameter controls robustness
2. **Design experiment**: Choose a mutation that specifically tests the hypothesis — change **exactly ONE parameter at a time**
3. **Run training**: The experiment runs across 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use both metrics AND cross-seed variance to evaluate whether the hypothesis was supported or contradicted
5. **Update understanding**: Revise hypotheses based on evidence. A falsified hypothesis is valuable information.

**CRITICAL**: You can only hypothesize. Only training results can validate or falsify your hypotheses. Never assume a hypothesis is correct without experimental evidence.

**Evidence hierarchy:**

| Level | Criterion | Action |
| --- | --- | --- |
| **Established** | Consistent across 3+ iterations AND 4/4 seeds | Add to Principles |
| **Tentative** | Observed 1-2 times or inconsistent across seeds | Add to Open Questions |
| **Contradicted** | Conflicting evidence across iterations/seeds | Note in Open Questions |

### CAUSALITY RULE (MANDATORY — READ THIS)

**If you change more than one parameter per slot, you CANNOT attribute the effect. This is a fatal experimental design error.**

- In EXPLORATION mode: Slot 0 = parent/baseline (unchanged control). Slots 1-3 each change **exactly one** parameter from the parent.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness).

## CRITICAL: Data is PRE-GENERATED at startup (fixed across iterations)

At startup, data is generated **once** for all 4 slots with **different random seeds** (one per slot). These datasets are **reused across all iterations** — data is NOT re-generated each iteration.
Both `simulation.seed` and `training.seed` are **forced by the pipeline** — DO NOT modify them in config files.

Seed formula (set automatically by GNN_LLM.py):
- `simulation.seed = 1000 + slot` (controls data generation — fixed at startup, slot 0–3)
- `training.seed = iteration * 1000 + slot + 500` (controls weight init & training randomness)

The actual seed values are provided in the prompt for each slot — **log them in your iteration entries**.

**Seed robustness testing**: To re-generate data with new seeds and test robustness, set `claude.test_robustness_seed: true` in all 4 slot configs. The pipeline will re-generate data for that batch only, then reset the flag automatically.

Simulation parameters (n_neurons, n_frames, etc.) stay fixed — **DO NOT change them**.

## Data Generation

Each slot re-generates data with a **different random seed**.
Seeds are **forced by the pipeline** — DO NOT modify them in config files.

- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

**DO NOT change `simulation:` parameters** except seed (managed automatically).

**IMPORTANT**: `noise_model_level` is set to **0.5** in the base config. Do NOT change it — this file is specifically for the noise=0.5 experiment.

## Noise Model

Two independent noise sources in the training data:

1. **Dynamics noise** (`noise_model_level=0.5`): `v(t+1) = v(t) + dt * f(v, W, I) + epsilon_dyn(t)`, epsilon_dyn ~ N(0, 0.5)
2. **Measurement noise** (`measurement_noise_level=0.0`): Clean observations

At high noise level, even with perfect ODE structure, parameter estimation becomes very difficult. Testing whether known_ode can maintain reasonable performance under strong noise.

## Larva Two-Population Motor Model

### Premotor neurons (N=178):

```
dup/dt = (-up + gp * softplus(up @ Jpp) + bp + wsp @ stim) / taup
```

### Motor neurons (M=52):

```
dum/dt = (-um + gm * softplus(up @ Jpm) + bm) / taum
```

- **230 neurons** total (178 premotor + 52 motor), **2 cell types**, **4,222 GT edges**, **52,670 FC edges**
- tau=1.0 (fixed), dt=0.05
- 2 stimulus conditions (forward/backward), 2 stimulus channels
- Inhibitory neurons get negative weights (Dale's law in connectome)
- 2,400 frames, delta_t=0.05, **noise_model_level=0.5**
- Feedforward: premotor->motor only, plus premotor recurrence

## Known ODE Architecture

The model is registered as `larva_known_ode`. Unlike the GNN:

- **Hardcoded activation**: `g_phi = softplus(v + b)` — b (bias) learned per neuron, g (gain) learned per neuron.
- **Direct W learning** on either FC or GT edges.
- **No embeddings, no MLP curves.**

**Parameters NOT used by known_ode** (do not modify): coeff_g_phi_diff, coeff_f_theta_msg_diff, coeff_g_phi_norm, coeff_g_phi_weight_L1/L2, coeff_f_theta_weight_L1/L2, embedding_dim, lr_embedding.

## Training Parameters

| Parameter                 | Default | Description                                            |
| ------------------------- | ------- | ------------------------------------------------------ |
| `lr_W`                    | 1e-3    | Learning rate for W (synaptic weights)                 |
| `lr`                      | 1e-3    | Learning rate for other params (g, b)                  |
| `n_epochs`                | 2       | Number of training epochs                              |
| `batch_size`              | 2       | Batch size                                             |
| `data_augmentation_loop`  | 100     | Data augmentation multiplier                           |
| `coeff_W_L1`              | 0       | L1 sparsity on W                                       |
| `coeff_W_L2`              | 0       | L2 penalty on W                                        |
| `coeff_W_sign`            | 0       | Dale's law penalty on W                                |
| `use_gt_edges`            | false   | Fully connected graph (52,670 edges)                   |
| `noise_model_level`       | 0.5     | **FIXED** — intrinsic noise level for this experiment  |


## Parallel Mode — 4 Slots Per Batch

Each batch runs 4 slots with different seeds (forced by pipeline). You choose the strategy:

- **Exploration** (default): Slot 0 = parent/control. Slots 1-3 each change **exactly one** parameter.
- **Robustness test**: ALL 4 slots use the SAME config (different seeds test robustness).

### Robustness Assessment

- **Robust**: all 4 slots connectivity_R2 > 0.3
- **Partially robust**: 2-3 slots > 0.3
- **Fragile**: 0-1 slots > 0.3

## Block Structure

| Block | Focus                          | Parameters to scan                          | Ranges                                                                                                           |
| ----- | ------------------------------ | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1     | **lr_W + lr sweep**            | `lr_W`, `lr`                                | lr_W: {5e-5, 1e-4, 5e-4, 1e-3}, lr: {1e-4, 5e-4, 1e-3, 5e-3}. Very high noise may require different lr. |
| 2     | **Training volume**            | `data_augmentation_loop`, `n_epochs`        | DAL: {100, 200, 500, 1000}, n_epochs: {2, 4, 8}. High noise requires very high training volume. |
| 3     | **W regularization**           | `coeff_W_L1`, `coeff_W_L2`, `coeff_W_sign` | W_L1: {0, 1e-6, 1e-5, 1e-4}, W_L2: {1e-6, 1e-5, 1e-4, 1e-3}, W_sign: {0, 0.1, 0.5}. |
| 4     | **Batch size**                 | `batch_size`                                | batch_size: {2, 4, 8, 16}. Larger batches may help smooth very noisy gradients.                                 |
| 5-8   | **Free exploration**           | Any parameter                               | Consolidate best from blocks 1-4, ceiling-breaking, final robustness test.                                       |

### Noise-specific considerations

- **10× stronger noise than noise=0.05**: Expected significant performance degradation.
- **Very strong regularization may be needed**: Even known ODE structure may not rescue identifiability under overwhelming noise.

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

From `analysis.log`: connectivity_R2, rollout_pearson, training_time_min.

### Step 3: Write Log Entries + Update Memory

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, DAL=D, n_epochs=E, W_L1=A, W_L2=B, W_sign=C, batch_size=B
Slot 0: conn_R2=A, rollout_pearson=B, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, rollout_pearson=B, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, rollout_pearson=B, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, rollout_pearson=B, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive]
Next: parent=P
```

## Winner Config (COMPULSORY)

**At every block boundary**, you MUST save the current best config as a winner file.

1. Identify the **best iteration** (highest connectivity_R2)
2. Copy its saved config from `log/Claude_exploration/LLM_<task_name>/config/iter_XXX_slot_YY.yaml`
3. Save it to `config/larva/larva_known_ode_noise_05_winner.yaml` with a YAML comment header:

```yaml
# Winner config: larva_known_ode_noise_05_winner.yaml
# Source: iter_XXX_slot_YY (connectivity_R2 = X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [1-2 sentence narrative]
#
# Metrics:
#   connectivity_R2: X.XXX (best single seed)
#   robust_mean:     X.XXX +/- X.XXX (N seeds, CV=X.X%)
#   rollout_pearson: X.XXX
#
# Key config differences from baseline:
#   - [list]
```

Destination: `config/larva/larva_known_ode_noise_05_winner.yaml`

## File Structure

You maintain THREE files:

1. **Full Log (append-only)**: `larva_known_ode_noise_05_Claude_analysis.md`
2. **Working Memory (read + update every batch)**: `larva_known_ode_noise_05_Claude_memory.md`
3. **User Input (read every batch, acknowledge pending items)**: `user_input.md`

## Start Call

When prompt says `PARALLEL START`:

- Read base config — the parent known_ode noise=0.05 config + noise_model_level=0.5 IS the baseline.
- Block 1 is a **robustness test**: all 4 slots use the same config (different seeds).
- Hypothesis: "Known ODE with noise=0.5 maintains reasonable connectivity_R2 robustness, though significantly degraded from noise=0.05"

---

# Working Memory Structure

```markdown
# Working Memory: larva_known_ode_noise_05

## Paper Summary (update at every block boundary)

- **Known ODE optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean+-std) | CV% | rollout_pearson | Robust? | Hypothesis |
| ---- | -------------- | ------------------- | --- | --------------- | ------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

---

## Previous Block Summary

---

## Current Block

### Block Info

### Current Hypothesis

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
