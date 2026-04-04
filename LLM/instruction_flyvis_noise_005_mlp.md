# FlyVis MLP Baseline — flyvis_noise_005 Rollout Optimization

## Goal

Optimize the **MLP baseline** hyperparameters for stable autoregressive rollout
on the flyvis model with noise level σ=0.05.

The MLP is a flat, graph-free model: `dv/dt = MLP([v_all; stimulus_all])`. No
edges, no message passing.

We are given the simulated neural activity traces as matrix (T, N). Given the activity
at time t plus the input stimulus, the MLP can be used to compute the activity
at the next time point (via Euler integration). We
continue this process to generate an autoregressive rollout from the initial activity.
We call this rollout training and the number of steps is `rollout_train_steps` in
the config. We apply an MSE loss to train the MLP to produce the exact correct
1-step update so that the rollout agrees with the ground truth data.

The GNN models in this repo can learn the underlying mechanistic update and
consequently can generate an accurate rollout for over 8000 steps. The MLPs here
are not meant to be mechanistic, but instead serve as function approximators to
the true 1-step update. Since the system is input driven we want to find a regime
of parameters where we can predict ~ 8000 steps ahead accurately.

## Scientific Context

The core research question is: **Can a flat, graph-free MLP function approximator achieve stable autoregressive rollout on the FlyVis Drosophila visual system model under realistic noise (σ=0.05)?** 

This explores whether the mechanistic, GNN-based approaches used elsewhere in the codebase are necessary, or whether a simpler statistical model can approximate the true 1-step dynamics well enough to sustain multi-thousand-step predictions. The MLP is not expected to learn causal structure, but rather to memorize the input-output manifold of the true system. The key challenge is error compounding in autoregressive mode — small per-step errors accumulate rapidly unless the model learns a regime where errors remain bounded.

## Noise Model

The FlyVis simulation includes realistic noise:

```
v_i(t+1) = v_i(t) + dt * f(v_i(t), W, a_i, I_i(t)) + epsilon_i(t)
epsilon_i ~ N(0, 0.05)  [dynamics noise]
```

The MLP must learn to predict `dv_i/dt` from the observed activity traces, which are corrupted by this noise. Larger rollout windows expose the model to accumulated noise, making training with longer `rollout_train_steps` (unrolling 20+ steps) essential for stability.

## Metrics

**Always use metrics defined to guide decision making**

- Training. We compute the following two rollout metrics during training that are
  printed to the stdout in this format:

  ```
  epoch 7/20 | train: 4.4128e-02 | div_time=1262 rollout_mse=3.9962e-01 (3.0s) | duration: 41.1s (total: 327.8s)
  ```

  - `train`: this is the rollout loss mentioned above
  - `div_time`: the time step at which the rollout MSE reaches 1.0 - a proxy
    for divergence.
  - `rollout_mse`: the MSE over neurons and time steps up until `div_time`
    The extra rollout metrics - though computed on training data - will nevertheless
    be important factors in guiding parameter loss. If a model fails to produce a
    stable rollout on training data, it is very unlikely to do so on new data.

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

- `use_residual_connection`: adds a residual connection across the hidden layers
  of the MLP for better gradient flow. It also zero-initializes the final hidden
  layer to keep the rollout stable. We do not have evidence for this assertion.
- No graph structure, no per-edge weights
- We want to find the smallest architecture that achieves our goal

**YOU ARE ONLY ALLOWED TO MODIFY THE PARAMETERS BELOW TO ACHIEVE THE GOAL**

## Architecture parameters

| Parameter                 | Default | Description                         |
| ------------------------- | ------- | ----------------------------------- |
| `hidden_dim`              | 256     | Hidden layer width                  |
| `n_layers`                | 5       | Number of layers (including in/out) |
| `use_residual_connection` | true    | Zero-init final layer for stability |

## Training Parameters

| Parameter                    | Default | Description                                         |
| ---------------------------- | ------- | --------------------------------------------------- |
| `lr`                         | 0.00001 | Learning rate for MLP weights                       |
| `n_epochs`                   | 20      | Training epochs                                     |
| `batch_size`                 | 256     | Frames per batch                                    |
| `data_augmentation_loop`     | 10      | Passes over data per epoch                          |
| `rollout_train_steps`        | 20      | Multi-step rollout unroll during training (K steps) |
| `early_stop_patience_epochs` | 5       | Stop if rollout doesn't improve for N epochs        |
| `seed`                       | 42      | Random seed - test for robustness to init.          |
| ------------------------     | ------- | --------------------------------------------------- |

## Comments

`rollout_train_steps` controls how many Euler steps are unrolled during training and
backpropagated through. This directly penalizes error compounding:

- `rollout_train_steps=1`: one-step MSE only — fast but errors compound at test time
- `rollout_train_steps=5`: unroll 5 steps — teaches the model to be stable over short horizons
- `rollout_train_steps=20`: longer horizon — more expensive, may help or hurt stability
- Higher values increase training time roughly linearly
- Manual experimentation has shown that we can get stable rollouts of ~ 1000+ steps
- But it seems difficult to reach ~ 8500 steps which would be needed

Note that with T starts per epoch, if you apply a rollout, each data point is actually
visited `rollout_train_steps` times. So with epochs=10, DAL=1, rollout_train_steps=2 we are training
on T*10*2 datapoints.

The `data_augmentation_loop` just controls how long each epoch is. We pay a fixed time
cost (~ few secs) to run the rollout after each epoch, and we want to keep 1 epoch training time
greater epoch validation cost.

We have observed that as training proceeds the rollout divergence time gets longer and in some
cases starts to get worse. That's why we have the early stop. But if this is detrimental you should
remove early stopping. For example, if you find that epochs=20, but at 10 epochs you hit the
lowest rollout mse, you can consider to reduce the epoch count from 20 -> 15 say.

## Training Time Constraint

**Target <=60 min per iteration.**
Use `data_augmentation_loop` (DAL) and `n_epochs` to control training time.
If the training time is much less than 60min you can consider running for more epochs to
explore any further reduction in rollout performance.

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

**Block 4 notes**: Goal is the _smallest_ model that achieves rollout_RMSE < 0.1. Start from best config of blocks 1-3. First find the minimum `hidden_dim` that achieves the target, then verify with the minimum `n_layers`. Prefer smaller models: a model with half the parameters that hits the target beats a larger model.

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

1. Read `train_div_time`, `train_rollout_mse`, and `train_best_epoch` from the analysis log (these are computed on training data — a hard fail if `train_div_time` is < 1000)
2. Read `rollout_RMSE` and `rollout_RMSE_std` from metrics log
3. Read `results_rollout_by_step.csv` — note first window where RMSE exceeds 1.0 (divergence point)
4. Read `onestep_pearson` as sanity check — if poor, model hasn't learned dynamics at all
5. Note `training_time_min` and adjust DAL for next batch if needed

### Step 3: Write Log Entry + Update Memory

```
## Iter N: [excellent/good/poor/diverged]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr=X, rollout_train_steps=K, DAL=D, n_epochs=E, hidden_dim=H, n_layers=L, batch_size=B
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

| Iter | Config summary | rollout_RMSE (mean±std) | CV% | div_time (mean) | divergence_at | Verdict | Hypothesis |
| ---- | -------------- | ----------------------- | --- | --------------- | ------------- | ------- | ---------- |

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
