# FlyVis known_ode Training Exploration — flyvis_noise_05

## Goal

Test **known_ode performance in high-noise regime** for the **Drosophila visual system** with noise level 0.5 (DAVIS input). The goal is to find a **noise-robust config** that achieves **connectivity_R2 > 0.7 on ALL 4 seeds with CV < 5%**, demonstrating resilience to severe noise corruption. Data is thus **re-generated each iteration** with a different seed. This variant tests the limits of the known_ode approach when noise severely corrupts the training signal. A config with mean connectivity_R2=0.75 and CV=3% establishes the expected performance ceiling in this challenging regime.

Primary metric: **connectivity_R2** (R² between learned W and ground-truth W).
**Stability metric: CV (coefficient of variation) of connectivity_R2 across 4 seeds — target CV < 5%.**
Secondary metrics: **tau_R2** (time constant recovery), **V_rest_R2** (resting potential recovery), **cluster_accuracy** (neuron type clustering from embeddings).

## Scientific Method

You can only hypothesize. Only training results validate or falsify.
**If you change more than one parameter per slot, you CANNOT attribute the effect. This is a fatal experimental design error.**

1. **Hypothesize**: Form a specific, testable prediction
2. **Design experiment**: Change **EXACTLY ONE** parameter at a time to understand causality
3. **Run training**: 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use metrics AND cross-seed variance
5. **Update understanding**: Revise hypotheses based on evidence

**Evidence hierarchy:**
| Level | Criterion | Action |
| ---------------- | ----------------------------------------------- | ---------------------- |
| **Established** | Consistent across 3+ iterations AND 4/4 seeds | Add to Principles |
| **Tentative** | Observed 1-2 times or inconsistent across seeds | Add to Open Questions |
| **Contradicted** | Conflicting evidence across iterations/seeds | Note in Open Questions |

## Data Generation

Each slot re-generates data with a **different random seed**.
Seeds are **forced by the pipeline**

- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`
  **DO NOT change `simulation:` parameters** except seed (managed automatically).

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i(t)
dv_i/dt = f_theta(v_i, a_i, sum_j W_ij * g_phi(v_j, a_j)^2, I_i)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors)
- DAVIS visual input, **noise_model_level=0.5** (high noise, severe corruption)
- 64,000 frames, delta_t=0.02

## known_ode ML model

- the ODE of the neural dynamics is given,
- hence known_ode learnes directly the parameters tau_i, Vrest_i and W_ij

**Parameters NOT used by known_ode** (do not modify): coeff_g_phi_diff, coeff_f_theta_msg_diff, coeff_g_phi_norm, coeff_g_phi_weight_L1, coeff_g_phi_weight_L2, coeff_f_theta_weight_L1, coeff_f_theta_weight_L2, embedding_dim, lr_embedding.

## Training Parameters TO BE SWEEPED

| Parameter      | Default | Description                                                                      |
| -------------- | ------- | -------------------------------------------------------------------------------- |
| `lr_W`         | 0.0009  | Learning rate for W (synaptic weights)                                           |
| `lr`           | 0.0018  | Learning rate for other params (tau, Vrest)                                      |
| `w_init_mode`  | RANDN   | W initialization mode: RANDN (std=1), RANDN_SCALED (std=scale/sqrt(N)), or ZEROS |
| `w_init_scale` | 1.0     | Scaling factor for RANDN_SCALED mode                                             |
| `batch_size`   | 4       | Batch size                                                                       |
| `coeff_W_L1`   | 0       | L1 sparsity on W                                                                 |
| `coeff_W_L2`   | 0.00015 | L2 penalty on W                                                                  |
| `coeff_W_sign` | 1.5e-06 | Dale's law penalty on W                                                          |

## Parallel Mode — 4 Slots Per Batch

Each batch runs 4 slots with different seeds (forced by pipeline). You choose the strategy:

- **Exploration** (default): Slots 0-3 each change **exactly one** parameter. This gives 3 causal tests per batch.
- **Robustness test**: ALL 4 slots use the SAME config. The pipeline forces different seeds, so this measures seed robustness. Use this when a config looks promising.
  State your choice (exploration vs robustness test) in the log entry.

### Robustness Assessment (when running same config across 4 slots)

- **Robust**: all 4 slots connectivity_R2 > 0.7
- **Partially robust**: 2-3 slots > 0.6
- **Fragile**: 0-1 slots < 0.5

## Block Partition

| Block | Focus | Parameters | range  
| 1 | **learning rate sweep** | `lr_W`, `lr` | lr_W: {1e-5 to 1e-3}, lr: {1e-4 to 1e-2}  
| 2 | **W regularization** | `coeff_W_L1`, `coeff_W_L2`, `coeff_W_sign` | {1e-6 to 1e-4}.  
| 3 | **W initialization + lr** | `w_init_mode`, `w_init_scale`, `lr_W`, `lr` | w_init_mode: {RANDN, RANDN_SCALED, ZEROS}, w_init_scale: {0.5, 1.0, 2.0}  
| 4 | **Batch size + lr** | `batch_size`, `lr_W`, `lr` | batch_size: {1, 2, 4, 8}.
| 5 | **free exploration** | Any parameter |  
| 6 | **Final robustness** | None (robustness test) | 4-seed robustness test of best config from blocks

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

From `analysis.log`: connectivity_R2, rollout_pearson, tau_R2, training_time_min.

### Step 3: Write Log Entries + Update Memory

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, DAL=D, n_epochs=E, W_L1=A, W_L2=B, W_sign=C, batch_size=B
Slot 0: conn_R2=A, rollout_pearson=B, tau_R2=C, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, rollout_pearson=B, tau_R2=C, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, rollout_pearson=B, tau_R2=C, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, rollout_pearson=B, tau_R2=C, sim_seed=S, train_seed=T
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
3. Save it to `config/drosophila_cx/drosophila_cx_known_ode_winner.yaml` with a YAML comment header:

```yaml
# Winner config: drosophila_cx_known_ode_winner.yaml
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
#   tau_R2:          X.XXX
#   spectral_radius: X.XXX (true: X.XXX)
#
# Key config differences from baseline:
#   - [list the parameters that differ from the initial baseline]
```

Destination: `config/fly/fly_noise_05_known_ode_winner.yaml`

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
- Slots 1-3: each changes EXACTLY ONE parameter from the block focus.
- Hypothesis: "Known ODE can achieve connectivity_R2 > 0.7 even with high noise corruption (0.5)"

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_05_known_ode

## Paper Summary (update at every block boundary)

- **Known ODE optimization in high-noise regime**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean+-std) | CV% | onestep_pearson | rollout_pearson | tau_R2 | Robust? | Hypothesis |
| ---- | -------------- | ------------------- | --- | --------------- | --------------- | ------ | ------- | ---------- |

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
