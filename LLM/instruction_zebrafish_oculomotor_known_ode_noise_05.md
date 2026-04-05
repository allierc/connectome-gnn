# Zebrafish Oculomotor — Known ODE Exploration (Noise 0.5)

## Goal

Maximize **connectivity_R2** (PRIMARY) for the **zebrafish oculomotor integrator** using the **known_ode model** under **strong intrinsic noise (sigma=0.5)**.

The known_ode model uses the **exact activation function**: `g_phi = identity (linear)` and **exact dynamics**: `dv/dt = (-v + W @ v + I)`. All parameters are **learned from data**: W (synaptic weights). This is an **upper bound** on what is achievable with perfect structural knowledge.

**Starting hypothesis**: "Known ODE + noise=0.5 still achieves connectivity_R2 > 0.3 robustly (10x stronger noise may limit recovery)"

The linear integrator degeneracy is broken by noise's enriched activity covariance. At noise=0.5, can we still recover W?

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned W and ground-truth W
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free data

Informational: onestep_pearson, training_time_min.

## Scientific Method

This exploration follows a strict **hypothesize → test → validate/falsify** cycle. Change **exactly ONE parameter at a time** to understand causality.

### CAUSALITY RULE (MANDATORY — READ THIS)

**If you change more than one parameter per slot, you CANNOT attribute the effect. This is a fatal experimental design error.**

- In EXPLORATION mode: Slot 0 = parent/baseline. Slots 1-3 each change **exactly one** parameter.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness).

## CRITICAL: Data is RE-GENERATED per slot

Each slot re-generates its data with a **different random seed**.
Both `simulation.seed` and `training.seed` are **forced by the pipeline** — DO NOT modify them in config files.

Seed formula (set automatically by GNN_LLM.py):
- `simulation.seed = iteration * 1000 + slot` (controls data generation)
- `training.seed = iteration * 1000 + slot + 500` (controls weight init & training randomness)

Simulation parameters stay fixed — **DO NOT change them**.

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

At high noise level, even with perfect linear structure, parameter estimation becomes very difficult. Testing whether noise-induced degeneracy-breaking persists under overwhelming noise.

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
- g_phi should learn identity, f_theta should learn f(v)=-v
- Dynamics purely determined by W eigenstructure
- Some populations have zeroed connections

**Key challenge**: High noise may overwhelm noise-induced identifiability, limiting W recovery.

## Known ODE Architecture

The model is registered as `zebrafish_oculomotor_known_ode`. Unlike the GNN:

- **Hardcoded activation**: `g_phi = identity (linear)` — no nonlinearity, no parameters.
- **Direct W learning**: Synaptic weight matrix W is learned directly on graph edges.
- **No embeddings, no MLP curves.**

**Parameters NOT used by known_ode** (do not modify): coeff_g_phi_diff, coeff_f_theta_msg_diff, coeff_g_phi_norm, coeff_g_phi_weight_L1/L2, coeff_f_theta_weight_L1/L2, embedding_dim, lr_embedding.

## Training Parameters

| Parameter                 | Default | Description                                            |
| ------------------------- | ------- | ------------------------------------------------------ |
| `lr_W`                    | 1e-3    | Learning rate for W (synaptic weights)                 |
| `lr`                      | 1e-3    | Learning rate for other params (if any)                |
| `n_epochs`                | 2       | Number of training epochs                              |
| `batch_size`              | 2       | Batch size                                             |
| `data_augmentation_loop`  | 100     | Data augmentation multiplier                           |
| `coeff_W_L1`              | 0       | L1 sparsity on W                                       |
| `coeff_W_L2`              | 0       | L2 penalty on W                                        |
| `coeff_W_sign`            | 0       | Dale's law penalty on W                                |
| `use_gt_edges`            | false   | Fully connected graph (609x609 = 370,881 edges)        |
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
| 4     | **Batch size**                 | `batch_size`                                | batch_size: {2, 4, 8, 16}. Larger batches may help smooth very noisy gradients. |
| 5-8   | **Free exploration**           | Any parameter                               | Consolidate best from blocks 1-4, ceiling-breaking, final robustness test. |

### Noise-specific considerations

- **10× stronger noise than noise=0.05**: Expected significant performance degradation.
- **May hit identifiability ceiling**: Very strong noise could overwhelm noise-induced identifiability benefit.
- **Very strong regularization may be needed**: W_L2 and W_sign become critical.

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

Destination: `config/zebrafish_oculomotor/zebrafish_oculomotor_known_ode_noise_05_winner.yaml`

## File Structure

1. **Full Log (append-only)**: `zebrafish_oculomotor_known_ode_noise_05_Claude_analysis.md`
2. **Working Memory**: `zebrafish_oculomotor_known_ode_noise_05_Claude_memory.md`
3. **User Input**: `user_input.md`

## Start Call

When prompt says `PARALLEL START`:

- Read base config — the parent noise=0.05 config + noise_model_level=0.5 IS the baseline.
- Block 1 is a **robustness test**: all 4 slots use the same config (different seeds).
- Hypothesis: "Known ODE with noise=0.5 maintains some identifiability (conn_R2 > 0.3), though degraded from noise=0.05"

---

# Working Memory Structure

```markdown
# Working Memory: zebrafish_oculomotor_known_ode_noise_05

## Paper Summary (update at every block boundary)

- **Known ODE + strong noise**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean+-std) | CV% | rollout_pearson | Robust? | Hypothesis |
| ---- | -------------- | ------------------- | --- | --------------- | ------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

---

## Current Block

### Block Info

### Current Hypothesis

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
