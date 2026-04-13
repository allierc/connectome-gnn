# FlyVis Stimulus Baseline — One-Step Reconstruction Optimization

## Goal

Optimize the **stimulus-only baseline** hyperparameters for the lowest possible one-step
reconstruction RMSE on the flyvis noise-free model.

The stimulus baseline is a feedforward model that predicts neuron voltage from a window of stimulus
frames ending at the current time step — it has **no dependence on past voltage/activity**:

```
v(t) = Predictor(flatten(StimEncoder(stim[t-tw+1 : t+1])))
```

Two sub-networks:
- **Stimulus encoder**: per-frame MLP that maps each stimulus frame independently to a latent vector
  (`n_input_neurons -> stim_latent_dims`)
- **Predictor**: MLP that takes the concatenation of all encoded frames and predicts voltage
  (`tw * stim_latent_dims -> n_neurons`)

This model serves as a lower-bound baseline: it measures how much of the neural activity can be
predicted from the stimulus trajectory alone, without any knowledge of the network's internal
dynamics.

## Scientific Context

The core research question is: **How much of the neural activity in the simulated FlyVis model is
directly predictable from the visual stimulus history?**

Here we are exploring:

- The lower bound on reconstruction error achievable without modelling internal dynamics
- How many frames of stimulus history are needed to capture the stimulus-driven component
- Whether a compact encoder-predictor architecture can efficiently learn the stimulus-to-activity
  mapping

## Noise Model

This config uses `noise_model_level: 0` (noise-free simulation). Both training and test data are
deterministic given the seed — there is no irreducible noise floor.

Data is **re-generated each iteration** with a different seed to verify seed independence.

## Metrics

**Always use metrics defined to guide decision making**

- Training. The following is printed to stdout each epoch:

  ```
  epoch 7/100 | train: 4.4128e-02 | duration: 41.1s (total: 327.8s)
  ```

  - `train`: MSE loss on the training data (one-step prediction).

- During test/validation (from `results_test.log`):
  - **PRIMARY METRIC: `onestep_RMSE`** (lower is better). Per-neuron RMSE averaged across all
    neurons and test frames.
  - **SECONDARY METRIC: `onestep_pearson`** (higher is better). Per-neuron Pearson correlation
    averaged across neurons.

- **There is no rollout evaluation** — since each prediction is independent (no recurrence), errors
  do not compound. No `results_rollout.log` or `results_rollout_by_step.csv` is generated.

- The analysis log also contains `train_constant_baseline_rmse`: the RMSE of a trivial model that
  predicts `v(t) = v(t-1)`. Your model should beat this comfortably.

## Scientific Method

Strict **hypothesize -> test -> validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction about what affects reconstruction quality
2. **Design experiment**: Change **EXACTLY ONE** parameter at a time (causality rule)
3. **Run training**: 4 slots (1 control + 3 experiments in EXPLORATION; 4 identical configs with
   different seeds in ROBUSTNESS) — you cannot predict the outcome
4. **Analyze results**: Use onestep_RMSE and onestep_pearson to evaluate
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
- Noise level: 0 (noise-free)
- 64,000 frames, delta_t = 0.02

## Stimulus Baseline Architecture

```
input  = stim[t-tw+1 : t+1]                               (tw, 1736)
         |
         v
StimEncoder(per-frame): 1736 -> stim_latent_dims           applied to each of tw frames
         |
         v
flatten: tw * stim_latent_dims                             concatenated
         |
         v
Predictor: tw * stim_latent_dims -> n_neurons (13,741)     single output
```

Both MLPs use ReLU activations (hardcoded in `_build_mlp`).

**YOU ARE ONLY ALLOWED TO MODIFY THE PARAMETERS BELOW TO ACHIEVE THE GOAL**

## Architecture Parameters

| Parameter                 | Default | Description                                                              |
| ------------------------- | ------- | ------------------------------------------------------------------------ |
| `stim_latent_dims`        | 64      | Output dimensionality of the per-frame stimulus encoder                  |
| `hidden_dim_stim_encoder` | 64      | Hidden layer width for the stimulus encoder MLP                          |
| `n_layers_stim_encoder`   | 3       | Number of layers in the stimulus encoder MLP (input + hidden + output)   |
| `hidden_dim`              | 512     | Hidden layer width for the predictor MLP                                 |
| `n_layers`                | 3       | Number of layers in the predictor MLP (input + hidden + output)          |

Note: `stim_latent_dims` affects the predictor's input dimension (`tw * stim_latent_dims`), so
changing it affects both sub-networks.

## Training Parameters

| Parameter                | Default | Description                                                          |
| ------------------------ | ------- | -------------------------------------------------------------------- |
| `lr`                     | 1e-4    | Learning rate for all weights (Adam optimizer)                       |
| `batch_size`             | 256     | Frames per batch                                                     |
| `data_augmentation_loop` | 10      | Number of data augmentation loops per epoch (controls training time) |
| `time_window`            | 5       | Number of stimulus frames used as input context (ending at t)        |

### Parameter Interactions

- **`time_window`** changes the predictor's input dimension (`tw * stim_latent_dims`), so it
  effectively changes the model architecture. Increasing `time_window` gives the model more
  temporal context but increases the predictor's input dimensionality linearly.
- **`stim_latent_dims`** also changes the predictor's input dimension. Increasing it gives a richer
  per-frame representation but increases the predictor's burden.
- When changing `time_window` or `stim_latent_dims`, consider whether `hidden_dim` needs to scale.

**Training time budget**: Each training run should take ~60 minutes. Use `data_augmentation_loop`
(DAL) to stay within this budget. When increasing parameters that scale training time, reduce DAL
pre-emptively to compensate. Check `training_time_min` in results and adjust for the next iteration.

**Do NOT modify `n_epochs`** — keep it at the configured value. Multiple epochs give us visibility
into how losses change during training, which is essential for diagnosing convergence.

**Hard runtime limit (120 minutes)**: The cluster enforces a hard 120-minute wall-clock limit per
job. If training approaches this limit, the job receives SIGUSR2 and writes an `_interrupted` file
in the run log directory. When analyzing results, check for `_interrupted` in each slot's log
directory — if present, training was cut short and the results are from a partial run. Reduce DAL
for that config in the next batch to fit within the 120-minute limit. If the job is killed after
the grace period without exiting cleanly, you will see
`TERM_RUNLIMIT: job killed after reaching LSF run time limit.` in `cluster_train_XX.out` (where XX
is the slot number). Otherwise, you will just see the `_interrupted` flag. Do NOT modify
`hard_runtime_limit_min` in the config — adjust DAL instead.

**Note**: Seeds are pipeline-controlled and overwritten before each run
(`simulation.seed = iteration * 1000 + slot`, `training.seed = iteration * 1000 + slot + 500`). Do
not set seeds in config files.

> **YAML rule**: Always wrap the `description` field value in double quotes — colons inside unquoted
> YAML strings cause parse errors (e.g., `description: "Block 7 Slot 1: testing W_L2"`).

## Data Generation

Data is re-generated each iteration with pipeline-controlled seeds (see Note above).

**DO NOT modify simulation parameters** (n_neurons, n_frames, n_edges, delta_t, noise_model_level).

## Block Structure

| Block | Focus                    | Parameters to scan                                                              | Ranges                                                                                                                                   |
| ----- | ------------------------ | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | **Training I**           | `lr`, `batch_size`                                                              | lr: {1e-5, 5e-5, 1e-4, 5e-4}; batch_size: {64, 128, 256, 512}                                                                          |
| 2     | **Architecture I**       | `hidden_dim`, `n_layers`, `stim_latent_dims`, `hidden_dim_stim_encoder`, `n_layers_stim_encoder` | hidden_dim: {128, 256, 512, 1024}; n_layers: {2, 3, 5}; stim_latent_dims: {32, 64, 128}; encoder dims/layers similarly                   |
| 3     | **Context window**       | `time_window`                                                                   | time_window: {1, 3, 5, 10, 20, 40}; goal is to find the smallest window that achieves near-best RMSE                                    |
| 4     | **Architecture II**      | Any architecture parameter                                                      | Re-explore architecture with best time_window from Block 3; refine capacity, encoder/predictor balance                                    |
| 5     | **Training II**          | `lr`, `batch_size`, `data_augmentation_loop`                                    | Narrow ranges around Block 1 winner, re-tune for best architecture from Blocks 2-4                                                       |
| 6     | **Free exploration**     | Any parameter                                                                   | Consolidate best from Blocks 1-5, test novel combinations, push RMSE as low as possible                                                  |

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

1. Read `onestep_RMSE` and `onestep_pearson` from the analysis log
2. Read `RMSE` and `Pearson r` from `results_test.log`
3. Note `training_time_min` and adjust DAL for next batch if needed
4. **Convergence check**: Compare `train` loss across the last 3 epochs. If it is still consistently
   decreasing at the final epoch, the model has not converged — note `convergence: not_reached` in
   the log entry. Do NOT extend training beyond the ~60 min budget; instead, flag the config as a
   candidate for longer training in the analysis.

### Step 3: Write Log Entry + Update Memory

```
## Iter N: [excellent/good/poor]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr=X, DAL=D, batch_size=B, time_window=TW, stim_latent_dims=SLD, hidden_dim_stim_encoder=HSE, n_layers_stim_encoder=LSE, hidden_dim=H, n_layers=L
Slot 0: onestep_RMSE=X, onestep_pearson=P, sim_seed=S, train_seed=T
Slot 1: onestep_RMSE=X, onestep_pearson=P, sim_seed=S, train_seed=T
Slot 2: onestep_RMSE=X, onestep_pearson=P, sim_seed=S, train_seed=T
Slot 3: onestep_RMSE=X, onestep_pearson=P, sim_seed=S, train_seed=T
Seed stats: mean_RMSE=X, std=Y, CV=Z%, mean_pearson=P
Mutation: [param]: [old] -> [new]
Convergence: [converged/not_reached] — if not_reached, note "may benefit from more epochs"
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
- Hypothesis: "The stimulus baseline can achieve low one-step reconstruction RMSE on flyvis
  noise-free data with appropriate architecture and training parameters"

---

# Working Memory Structure

```markdown
# Working Memory: {llm_task_name}

## Paper Summary (update at every block boundary)

- **Stimulus baseline optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | onestep_RMSE (mean+/-std) | CV% | onestep_pearson (mean) | Verdict | Hypothesis |
| ---- | -------------- | ------------------------- | --- | ---------------------- | ------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest->newest. This section MUST
appear before ## Current Block.**

### Block N Summary

[Summary of findings from block N]

---

## Current Block

### Block Info

### Current Hypothesis

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
