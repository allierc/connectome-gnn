# FlyVis MLP Baseline — flyvis_noise_005 Rollout Optimization

## Goal

# TODO: add metric in the top-line itself

# TODO: explain stable, use a metric always

Optimize the **MLP baseline** hyperparameters for stable autoregressive rollout on the flyvis model with noise level σ=0.05.
Our goal is to generate the longest possible rollout with the lowest possible MSE.

The MLP is a flat, graph-free model: `dv/dt = MLP([v_all; stimulus_all])`. No edges, no message passing.

The activity traces are (T, N). Training involves evolving a batch of initial conditions
(t_1, ..., t_B) that are randomly sampled. We predict t_i -> (t_i+1, t_i+2, ..., t_i+R)
in an autoregressive manner from the previously generated time point. This is "rollout training"
and R = `rollout_train_steps` in the config. We apply an MSE loss between expected and predicted.

**Always use metrics defined to guide decision making**

## Metrics

- Training. We compute the following two rollout metrics during training that are
  printed to the stdout in this format:

  ```
  epoch 7/20 | train: 4.4128e-02 | div_time=1262 rollout_mse=3.9962e-01 (3.0s) | duration: 41.1s (total: 327.8s)
  ```

  - `train`: this is the rollout loss mentioned above
  - `div_time`: the time step at which the rollout MSE reaches 1.0
  - `rollout_mse`: the MSE over neurons and time steps up until `div_time`

- Constant baseling: as a baseline we compute the voltage(t) - voltage(t+1) MSE. We ideally want
  this MSE over the entire 8k step rollout in validation/test.
- During test/validation:
  - **PRIMARY METRIC: `rollout_RMSE`** (lower is better).
  - **TARGET: rollout_RMSE < 0.1, want rollout_RMSE to remain bounded for different seeds**

Data is **re-generated each iteration** with a different seed to verify seed independence.

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction about what affects rollout stability
2. **Design experiment**: Change **EXACTLY ONE** parameter at a time (causality rule)
3. **Run training**: 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use rollout_RMSE AND the per-step CSV to understand divergence timing
5. **Update understanding**: Revise hypotheses based on evidence

**CRITICAL**: You can only hypothesize. Only training results validate or falsify.

### CAUSALITY RULE (MANDATORY)

**If you change more than one parameter per slot, you CANNOT attribute the effect. Fatal experimental design error.**

- In EXPLORATION mode: Slot 0 = parent/baseline (unchanged control). Slots 1-3 each change **exactly one** parameter from the parent.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness).

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * g(v_j) + I_i(t)
```

- **13,741 neurons**, 65 cell types, **434,112 edges**
- **1,736 input neurons** (photoreceptors, DAVIS visual input)
- **noise_model_level = 0.05** (Gaussian noise added during simulation)
- 64,000 frames, delta_t = 0.02

## MLP Architecture

```
input  = [v_1, ..., v_13741, stim_1, ..., stim_1736]   (15,477 dims)
output = [dv_1/dt, ..., dv_13741/dt]                    (13,741 dims)
```

- This is a simple way to think about the MLP - Encoder → hidden layers (ReLU) → Decoder
- `use_residual_connection`: adds a residual connection across the hidden layers of the MLP for better gradient flow. It also zero-initializes the final hidden layer to
  keep the rollout stable.
- No graph structure, no per-edge weights

**Architecture parameters:**

| Parameter                 | Default | Description                         |
| ------------------------- | ------- | ----------------------------------- |
| `hidden_dim`              | 256     | Hidden layer width                  |
| `n_layers`                | 5       | Number of layers (including in/out) |
| `use_residual_connection` | true    | Zero-init final layer for stability |

**Parameters NOT used by MLP** (always 0, do not modify):
`lr_W`, `lr_embedding`, `coeff_W_L1`, `coeff_W_L2`, `coeff_g_phi_diff`, `coeff_f_theta_msg_diff`, `coeff_g_phi_norm`, `coeff_g_phi_weight_L1`, `coeff_g_phi_weight_L2`, `coeff_f_theta_weight_L1`, `coeff_f_theta_weight_L2`, `embedding_dim`.

## Training Parameters

| Parameter                | Default | Description                                         |
| ------------------------ | ------- | --------------------------------------------------- |
| `lr`                     | 0.00001 | Learning rate for MLP weights                       |
| `n_epochs`               | 20      | Training epochs                                     |
| `batch_size`             | 256     | Frames per batch                                    |
| `data_augmentation_loop` | 100     | Passes over data per epoch                          |
| `rollout_train_steps`    | 5       | Multi-step rollout unroll during training (K steps) |
| `train_start`            | 4000    | First training frame (skip burn-in)                 |
| `train_end`              | 54000   | Last training frame                                 |
| `val_start`              | 54000   | Validation start frame                              |
| `val_end`                | 64000   | Validation end frame                                |

### Key insight: rollout_train_steps

`rollout_train_steps` controls how many Euler steps are unrolled during training and backpropagated through. This directly penalizes error compounding:

- `rollout_train_steps=1`: one-step MSE only — fast but errors compound at test time
- `rollout_train_steps=5`: unroll 5 steps — teaches the model to be stable over short horizons
- `rollout_train_steps=20`: longer horizon — more expensive, may help or hurt stability
- Higher values increase training time roughly linearly
- Manual experimentation has shown that we can get stable rollouts of ~ 1000+ steps
- But it seems difficult to reach ~ 8500 steps which would be needed

## Training Time Constraint

**Target ~60 min per iteration.** Use `data_augmentation_loop` (DAL) and `n_epochs` to control training time.

- If `training_time_min` < 40 min: **increase** DAL or n_epochs
- If `training_time_min` > 70 min: **decrease** DAL or n_epochs

## Metrics

### Primary: `rollout_RMSE`

From `results_rollout.log` (and written to `analysis.log`):

```
rollout_RMSE: X.XXXX
rollout_RMSE_std: X.XXXX
```

**Target: rollout_RMSE < 0.1**

Classification:

- **Excellent**: rollout_RMSE < 0.1
- **Good**: 0.1 – 0.5
- **Poor**: 0.5 – 5.0
- **Diverged**: > 5.0 (model explodes)

### Divergence profile: `results_rollout_by_step.csv`

**This is the most informative diagnostic.** Read this CSV to understand _when_ the model diverges:

```
frame_start,frame_end,RMSE,pearson
0,500,0.05,0.92
500,1000,0.12,0.75
1000,1500,2.3,0.10
...
```

- If RMSE is low early but spikes later: model diverges gradually — try more `rollout_train_steps`
- If RMSE spikes immediately (first window): model is unstable from the start — try lower `lr` or `use_residual_connection`
- If RMSE stays flat and low: rollout is stable

**Always read and report the per-step CSV for each slot.** Note the first window where RMSE exceeds 1.0 (divergence point).

### Secondary: `onestep_pearson`, `onestep_RMSE`

One-step prediction quality. A model can have good one-step metrics but diverge in rollout. Use these as sanity checks — if one-step quality is bad, the model hasn't learned the dynamics at all.

### Informational: `training_time_min`

Monitor training time and adjust DAL to stay within budget.

## Data Generation

Each slot re-generates data with a **different random seed** (forced by pipeline):

- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

**DO NOT modify simulation parameters** (n_neurons, n_frames, n_edges, delta_t, noise_model_level).

## Block Partition

| Block | Focus                   | Parameters to scan                   | Ranges                                           |
| ----- | ----------------------- | ------------------------------------ | ------------------------------------------------ |
| 1     | **rollout_train_steps** | `rollout_train_steps`                | {1, 5, 10, 20}                                   |
| 2     | **learning rate**       | `lr`                                 | {1e-5, 5e-5, 1e-4, 5e-4}                         |
| 3     | **architecture**        | `hidden_dim`, `n_layers`             | hidden_dim: {128, 256, 512}, n_layers: {3, 5, 7} |
| 4     | **training volume**     | `data_augmentation_loop`, `n_epochs` | DAL: {50, 100, 200}, n_epochs: {10, 20, 40}      |
| 5     | **batch size**          | `batch_size`                         | {64, 128, 256, 512}                              |
| 6     | **free exploration I**  | Any parameter                        | Consolidate best from blocks 1-5                 |
| 7     | **free exploration II** | Any parameter                        | Push toward target, robustness confirmation      |
| 8     | **robustness**          | Best config, all 4 slots same        | Confirm CV < 10% across seeds                    |

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

For each slot:

1. Read `rollout_RMSE` and `rollout_RMSE_std` from metrics log
2. Read `results_rollout_by_step.csv` — note divergence point (first window with RMSE > 1.0)
3. Read `onestep_pearson` as sanity check
4. Note `training_time_min`

### Step 3: Write Log Entry + Update Memory

```
## Iter N: [excellent/good/poor/diverged]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr=X, rollout_train_steps=K, DAL=D, n_epochs=E, hidden_dim=H, n_layers=L, batch_size=B
Slot 0: rollout_RMSE=X, divergence_at=frame_Y, onestep_pearson=Z, sim_seed=S, train_seed=T
Slot 1: rollout_RMSE=X, divergence_at=frame_Y, onestep_pearson=Z, sim_seed=S, train_seed=T
Slot 2: rollout_RMSE=X, divergence_at=frame_Y, onestep_pearson=Z, sim_seed=S, train_seed=T
Slot 3: rollout_RMSE=X, divergence_at=frame_Y, onestep_pearson=Z, sim_seed=S, train_seed=T
Seed stats: mean_RMSE=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive]
Next: parent=P
```

### Step 4: Acknowledge User Input

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Block Boundaries

At every block boundary:

1. Update "Paper Summary" in memory
2. Summarize block findings
3. Update "Established Principles" and "Falsified Hypotheses"
4. Clear "Current Block"
5. Carry forward best config as parent for next block

## Start Call

When prompt says `PARALLEL START`:

- Slot 0 = baseline (no changes from base config)
- Slots 1-3: each changes EXACTLY ONE parameter per block focus
- Hypothesis: "The MLP baseline can achieve stable rollout (RMSE < 0.1) on flyvis noise_005 with appropriate rollout_train_steps"

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_mlp

## Paper Summary (update at every block boundary)

- **MLP rollout optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | rollout_RMSE (mean±std) | CV% | divergence_at | Verdict | Hypothesis |
| ---- | -------------- | ----------------------- | --- | ------------- | ------- | ---------- |

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
