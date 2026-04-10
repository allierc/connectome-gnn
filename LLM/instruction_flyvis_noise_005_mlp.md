# FlyVis MLP Baseline — flyvis_noise_005 Rollout Optimization

## Goal

Optimize the **MLP baseline** hyperparameters for lowest possible autoregressive rollout error ~
O(1e-2) or lower on the flyvis model with noise level σ=0.05. Two sub-goals:

- minimize the overall RMSE
- maintain a roughly constant RMSE during rollout. An increase in the RMSE during rollout is an
  indicator of the discrepancy between true and inferred dynamics.

The MLP is a flat, graph-free model: `dv/dt = MLP([v_all; stimulus_all])`. No edges, no message
passing.

We are given the simulated neural activity traces as matrix (T, N). T=time steps, N=neurons. Given
the activity at time t plus the input stimulus, the MLP can be used to compute the activity at the
next time point (via Euler integration). We continue this process to generate an autoregressive
rollout from the initial activity.

Our initial starting point MLP is trained with a loss on the 1-step update, i.e., t -> t+1. And this
MLP is able to generate a rollout over 8000 steps with RMSE < 0.2 on the test dataset that is not
used during training at all.

The dynamic range of voltages in the simulation is bounded, therefore, if a model were to predict
v(t) = v(t=0), i.e., a constant, the RMSE is bounded and is between 0.2 and 0.3. To be a convincing
demonstration the MLP should produce an RMSE that is ~ 2e-2, or about 10x better. This may or may
not be feasible, but that is the goal.

## Scientific Context

The core research question is: **Can a flat, graph-free MLP function approximator achieve stable
autoregressive rollout on the simulated FlyVis model of the Drosophila visual system model under
realistic noise (σ=0.05)?**

Here we are exploring:

- can black-box models for neural activity predict long rollout beyond the training horizon? The
  initial config already demonstrates that we can train on t->t+1 and achieve long rollout.
- The key challenge is error compounding in autoregressive mode — small per-step errors accumulate
  rapidly unless the model learns a regime where errors remain bounded.
- How low can we drive the RMSE without the inductive biases of the underylying connectivity
  structure and activation functions & weights?

## Noise Model

The FlyVis simulation includes realistic noise:

```
v_i(t+1) = v_i(t) + dt * f(v_i(t), W, a_i, I_i(t)) + epsilon_i(t)
epsilon_i ~ N(0, 0.05)  [dynamics noise]
```

The MLP must learn to predict `dv_i/dt` from the observed activity traces, which are corrupted by
this noise.

## Metrics

**Always use metrics defined to guide decision making**

- Training. We compute the following two rollout metrics during training that are printed to the
  stdout in this format:

  ```
  epoch 7/20 | train: 4.4128e-02 | div_time=1262 rollout_mse=3.9962e-01 (3.0s) | duration: 41.1s (total: 327.8s)
  ```

  - `train`: this is the rollout loss mentioned above
  - `div_time`: the time step at which the rollout MSE reaches 1.0 - a proxy for divergence.
  - `rollout_mse`: the MSE over neurons and time steps up until `div_time` The extra rollout
    metrics - though computed on training data - will nevertheless be important factors in guiding
    parameter loss. If a model fails to produce a stable rollout on training data, it is very
    unlikely to do so on new data. **IMPORTANT** these metrics are computed over the training data.

- Constant baseline: as a baseline we compute the voltage(t) - voltage(t+1) MSE. We ideally want
  this MSE over the entire 8k step rollout in validation/test. Keep this baseline in mind. Fast
  learning rates can drive us to the trivial local minimum of predicting v(t) = constant.

- During test/validation:
  - **PRIMARY METRIC: `rollout_RMSE`** (lower is better).
  - **TARGET: rollout_RMSE < 2e-2**
  - Use the `results_rollout_by_step.csv` with a snapshot pasted below from a real run. Notice that
    the RMSE degrades from ~ 0.13 to 0.27. One of your goals is to reduce this gap.

```
frame_start,frame_end,RMSE,pearson
0,500,0.1238,0.7808
500,1000,0.1474,0.7689
...
8000,8500,0.2710,0.7778
8500,8527,0.2681,0.6372
```

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

**If you change more than one parameter per slot, you CANNOT attribute the effect. Fatal
experimental design error.**

- In EXPLORATION mode: Slot 0 = parent/baseline (unchanged control). Slots 1-3 each change **exactly
  one** parameter from the parent.
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

- No graph structure, no per-edge weights
- We want to find the smallest architecture that achieves our goal

**YOU ARE ONLY ALLOWED TO MODIFY THE PARAMETERS BELOW TO ACHIEVE THE GOAL**

## Architecture parameters

| Parameter        | Default | Description                                                      |
| ---------------- | ------- | ---------------------------------------------------------------- |
| `hidden_dim`     | 256     | Hidden layer width                                               |
| `n_layers`       | 5       | Number of layers (including in/out)                              |
| `MLP_activation` | "relu"  | Activation: "relu", "tanh", "sigmoid", "leaky_relu", "soft_relu" |
| `add_residual`   | true    | Add residual/skip connection                                     |

## Training Parameters

| Parameter                | Default | Description                                         |
| ------------------------ | ------- | --------------------------------------------------- |
| `lr`                     | 0.00001 | Learning rate for MLP weights                       |
| `n_epochs`               | 20      | Training epochs                                     |
| `batch_size`             | 256     | Frames per batch                                    |
| `rollout_train_steps`    | 20      | Multi-step rollout unroll during training (K steps) |
| `seed`                   | 42      | Random seed - test for robustness to init.          |
| `zero_init_output`       | false   | Zero-initialize final layer weights                 |
| ------------------------ | ------- | --------------------------------------------------- |

## Comments

`rollout_train_steps` controls how many Euler steps are unrolled during training and backpropagated
through. This directly penalizes error compounding:

- `rollout_train_steps=1`: one-step MSE only — fast but errors may compound
- `rollout_train_steps=5`: unroll 5 steps — teaches the model to be stable over short horizons
- `rollout_train_steps=20`: longer horizon — more expensive computationally, ONLY USE IF ABSOLUTELY
  NECESSARY.
- Higher values increase training time roughly linearly

Note that with T starts per epoch, if you apply a rollout, each data point is actually visited
`rollout_train_steps` times. So it is possible that when increasing `rollout_time_steps` you can
reduce n_epochs to maintain a reasonable training time.

> **YAML rule**: Always wrap the `description` field value in double quotes — colons inside unquoted
> YAML strings cause parse errors (e.g., `description: "Block 7 Slot 1: testing W_L2"`).

## Data Generation

Each slot re-generates data with a **different random seed** (forced by pipeline):

- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

**DO NOT modify simulation parameters** (n_neurons, n_frames, n_edges, delta_t, noise_model_level).

## Block Structure

| Block | Focus                   | Parameters to scan            | Ranges                                                    |
| ----- | ----------------------- | ----------------------------- | --------------------------------------------------------- |
| 1     | **batch size**          | `batch_size`                  | {32, 64, 128, 256}                                        |
| 2     | **learning rate**       | `lr`                          | {1e-5, 5e-5, 1e-4, 5e-4}                                  |
| 3     | **rollout_train_steps** | `rollout_train_steps`         | {10, 20, 50}                                              |
| 4     | **model capacity**      | `hidden_dim`, `n_layers`      | hidden_dim: {64, 128, 256, 512}; n_layers: {3, 4, 5, 7}   |
| 5     | **free exploration I**  | Any parameter                 | Consolidate best from blocks 1-4, test novel combinations |
| 6     | **free exploration II** | Any parameter                 | Continue pushing toward target with simplest viable model |
| 7     | **robustness**          | Best config, all 4 slots same | Confirm CV < 10% across seeds                             |

## File Structure

You maintain **THREE** files:

### 1. Full Log (append-only)

**File**: `{llm_task_name}_analysis.md`

### 2. Working Memory (read + update every batch)

**File**: `{llm_task_name}_memory.md`

### 3. User Input (read every batch, acknowledge pending items)

**File**: `user_input.md`

## Knowledge Base Guidelines

### What to Add to Established Principles

A principle must satisfy ALL of:

1. Observed consistently across **3+ iterations**
2. Consistent across **all 4 seeds** (not just mean, but low variance)
3. States a **causal relationship** (not just a correlation)

### What to Add to Open Questions

- Patterns observed 1-2 times
- Seed-dependent effects (works for some seeds but not others)
- Contradictions between iterations
- Theoretical predictions not yet verified

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:

1. State the original hypothesis
2. State the contradicting evidence (iteration number, metrics)
3. State what was learned from the falsification
4. Propose a revised hypothesis if applicable

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

For each slot:

1. Read `train_div_time`, `train_rollout_mse`, and `train_best_epoch` from the analysis log (these
   are computed on training data — a hard fail if `train_div_time` is < 1000)
2. Read `rollout_RMSE` and `rollout_RMSE_std` from metrics log
3. Read `results_rollout_by_step.csv` — note first window where RMSE exceeds 1.0 (divergence point)
4. Read `onestep_pearson` as sanity check — if poor, model hasn't learned dynamics at all
5. Note `training_time_min` and adjust n_epochs for next batch if needed

### Step 3: Write Log Entry + Update Memory

```
## Iter N: [excellent/good/poor/diverged]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr=X, rollout_train_steps=K, n_epochs=E, hidden_dim=H, n_layers=L, batch_size=B
Slot 0: rollout_RMSE=X, divergence_at=frame_Y, train_div_time=Z, train_best_epoch=W/E, onestep_pearson=P, sim_seed=S, train_seed=T
Slot 1: rollout_RMSE=X, divergence_at=frame_Y, train_div_time=Z, train_best_epoch=W/E, onestep_pearson=P, sim_seed=S, train_seed=T
Slot 2: rollout_RMSE=X, divergence_at=frame_Y, train_div_time=Z, train_best_epoch=W/E, onestep_pearson=P, sim_seed=S, train_seed=T
Slot 3: rollout_RMSE=X, divergence_at=frame_Y, train_div_time=Z, train_best_epoch=W/E, onestep_pearson=P, sim_seed=S, train_seed=T
Seed stats: mean_RMSE=X, std=Y, CV=Z%, mean_train_div_time=Z, mean_best_epoch=W
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
- Hypothesis: "The MLP baseline can achieve stable rollout (RMSE < 0.1) on flyvis noise_005 with
  appropriate rollout_train_steps"

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_mlp

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | rollout_RMSE (mean±std) | CV% | div_time (mean) | divergence_at | Verdict | Hypothesis |
| ---- | -------------- | ----------------------- | --- | --------------- | ------------- | ------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest→newest. This section MUST
appear before ## Current Block.**

### Block 1 Summary

[Summary of findings from block 1]

### Block 2 Summary

[Summary of findings from block 2]

### Block 3 Summary

[Summary of findings from block 3]

### Block 4 Summary

[Summary of findings from block 4]

---

## Current Block

### Block Info

### Current Hypothesis

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
