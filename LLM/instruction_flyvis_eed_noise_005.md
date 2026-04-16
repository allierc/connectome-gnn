# FlyVis EED Baseline — Rollout Optimization (noise=0.05)

## Goal

Optimize the **EED (Encode-Evolve-Decode) baseline** hyperparameters for lowest possible
autoregressive rollout error ~ O(1e-2) or lower on the flyvis model with **noise** (sigma=0.05).
Two sub-goals:

- minimize the overall RMSE
- maintain a roughly constant RMSE during rollout. An increase in the RMSE during rollout is an
  indicator of the discrepancy between true and inferred dynamics.

The EED is a flat, graph-free model that operates through a latent bottleneck:
`v(t+1) = decode(encode(v(t)) + evolve(encode(v(t)), stim_encode(stimulus(t))))`. No edges, no
message passing.

We are given the simulated neural activity traces as matrix (T, N). T=time steps, N=neurons. Given
the activity at time t plus the input stimulus, the EED predicts the activity at the next time point
(via Euler integration). We continue this process to generate an autoregressive rollout from the
initial activity.

Our initial starting point EED is trained with a loss on the 1-step update, i.e., t -> t+1. And this
EED is able to generate a rollout over 8000 steps with RMSE < 0.2 on the test dataset that is not
used during training at all.

The dynamic range of voltages in the simulation is bounded, therefore, if a model were to predict
v(t) = v(t=0), i.e., a constant, the RMSE is bounded and is between 0.2 and 0.3. To be a convincing
demonstration the EED should produce an RMSE that is ~ 1e-2, or about 10x better. This may or may
not be feasible, but that is the goal.

## Scientific Context

The core research question is: **Can a latent-space EED model achieve stable autoregressive rollout
on the simulated FlyVis model of the Drosophila visual system when trained on noisy data?**

Here we are exploring:

- can a dimensionality-reducing architecture (encode to latent space, evolve, decode) predict long
  rollout beyond the training horizon?
- The latent bottleneck forces the model to learn a compressed representation of the 13,741-neuron
  state. This is a form of inductive bias — the model must discover a low-dimensional manifold.
- The key challenge is error compounding in autoregressive mode — small per-step errors accumulate
  rapidly unless the model learns a regime where errors remain bounded.
- How does the latent bottleneck compare to a flat MLP in terms of rollout stability?
- The latent bottleneck may act as an implicit denoiser — compressing 13,741 dimensions into a
  low-dimensional manifold could filter out noise that doesn't lie on the manifold.

## Noise Model

The FlyVis forward simulation is an SDE — noise is injected at each Euler integration step,
making the training trajectories stochastic realizations:

```
v_i(t+1) = v_i(t) + dt * f(v_i(t), W, a_i, I_i(t)) + sigma * z_i(t)
z_i ~ N(0, 1),  sigma = 0.05 (noise_model_level)
```

The EED sees trajectories sampled from this SDE and must learn the deterministic drift f(v, W, a, I)
despite the additive noise. The per-step noise on voltages has std=0.05, while the clean dynamics
step dt * f(...) is typically O(dt) ~ O(0.02).

**Important**: Noise is only added to the **training data**. Test/validation rollouts are computed
using the deterministic ODE (sigma=0). The test metric (`rollout_RMSE`) reflects pure model quality.
Do not compare training and test RMSE directly — the training RMSE has an irreducible noise floor.

## Metrics

**Always use metrics defined to guide decision making**

- Training. We compute the following metrics during training that are printed to stdout in this
  format:

  ```
  epoch 7/20 | total: 4.41e-02 recon: 2.10e-02 evolve: 2.31e-02 | div_time=1262 rollout_rmse=3.99e-01 (3.0s) | duration: 41.1s (total: 327.8s)
  ```

  - `total`: sum of reconstruction and evolution losses
  - `recon`: reconstruction loss — MSE(decoder(encoder(v_t)), v_t). Measures autoencoder quality. If
    poor, the latent space does not faithfully represent the full state.
  - `evolve`: evolution loss — MSE(decoder(encoder(v*t) + evolver(encoder(v_t),
    stim_encode(stim_t))), v*{t+1}). Measures one-step prediction quality.
  - `div_time`: the time step at which the rollout MSE reaches 1.0 - a proxy for divergence.
  - `rollout_rmse`: the RMSE over neurons and time steps up until `div_time`. The extra rollout
    metrics - though computed on training data - will nevertheless be important factors in guiding
    parameter choices. If a model fails to produce a stable rollout on training data, it is very
    unlikely to do so on new data. **IMPORTANT** these metrics are computed over the training data.

- Constant baseline RMSE ~ **0.25**. If rollout_RMSE is near 0.25, the model has collapsed to
  predicting v(t) ~ constant. Fast learning rates can drive to this trivial local minimum.

- During test/validation:
  - **PRIMARY METRIC: `rollout_RMSE`** (lower is better).
  - **TARGET: rollout_RMSE < 1e-2**
  - Use the `results_rollout_by_step.csv` to track how RMSE evolves across the rollout. One of your
    goals is to minimize the gap between early and late rollout RMSE. Example output:

```
frame_start,frame_end,RMSE,pearson
0,500,0.1238,0.7808
500,1000,0.1474,0.7689

...middle rows omitted for brevity...

8000,8500,0.2710,0.7778
8500,8527,0.2681,0.6372
```

Data is **re-generated each iteration** with a different seed to verify seed independence.

## Scientific Method

Strict **hypothesize -> test -> validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction about what affects rollout stability
2. **Design experiment**: Change **EXACTLY ONE** parameter at a time (causality rule)
3. **Run training**: 4 slots (1 control + 3 experiments in EXPLORATION; 4 identical configs with
   different seeds in ROBUSTNESS) — you cannot predict the outcome
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
- Noise level: sigma=0.05 per time step
- 64,000 frames, delta_t = 0.02

## EED Architecture

The EED has four sub-networks, each implemented as an MLPWithSkips (MLP where each hidden layer
receives a direct linear skip-projection of the input, concatenated with the previous layer's
output):

```
encoder:          v(t)      -> z(t)         [n_neurons -> latent_dim]
stimulus_encoder: stim(t)   -> s(t)         [n_input_neurons -> stim_latent_dims]
evolver:          [z(t);s(t)] -> dz          [latent_dim+stim_latent_dims -> latent_dim]
decoder:          z(t+1)    -> v_pred(t+1)  [latent_dim -> n_neurons]
```

Forward pass: `z(t+1) = encoder(v(t)) + evolver([encoder(v(t)), stim_encode(stim(t))])` then
`v_pred(t+1) = decoder(z(t+1))`, and `dv/dt = (v_pred(t+1) - v(t)) / dt`.

The evolver output layer is **zero-initialized**, so it starts as an identity mapping
(`z(t+1) = z(t) + 0`). The encoder and decoder form an autoencoder that must first learn a faithful
latent representation before the evolver can learn to evolve it.

**YOU ARE ONLY ALLOWED TO MODIFY THE PARAMETERS BELOW TO ACHIEVE THE GOAL**

## Architecture Parameters

| Parameter                 | Default | Description                                                                           |
| ------------------------- | ------- | ------------------------------------------------------------------------------------- |
| `latent_dim`              | 256     | Latent space dimensionality; also hidden width for encoder, decoder, and evolver      |
| `n_layers_encoder`        | 1       | Hidden layers in encoder and decoder (symmetric)                                      |
| `n_layers_evolver`        | 1       | Hidden layers in evolver                                                              |
| `stim_latent_dims`        | 64      | Stimulus encoder output dimensionality                                                |
| `hidden_dim_stim_encoder` | 64      | Stimulus encoder hidden width                                                         |
| `n_layers_stim_encoder`   | 3       | Hidden layers in stimulus encoder                                                     |
| `MLP_activation`          | "relu"  | Activation for all sub-networks: "relu", "tanh", "sigmoid", "leaky_relu", "soft_relu" |

## Training Parameters

| Parameter             | Default | Description                                                     |
| --------------------- | ------- | --------------------------------------------------------------- |
| `lr`                  | 0.00001 | Learning rate for all EED weights                               |
| `data_augmentation_loop` | 100  | Number of data augmentation loops per epoch (controls training time) |
| `batch_size`          | 256     | Frames per batch                                                |
| `rollout_train_steps` | 1       | Multi-step rollout unroll during training (K steps)             |

**Caution on `rollout_train_steps`**: Increasing this value may degrade one-step prediction performance (`onestep_pearson`), which is also an important metric. Do not increase `rollout_train_steps` unless you can show experimentally that one-step performance is unaffected.

**Note**: The EED loss has two components: (1) **reconstruction loss** at t=0 —
MSE(decoder(encoder(v_t)), v_t), ensuring the autoencoder faithfully represents the state; and (2)
**rollout loss** over K steps using `predict_dvdt` + Euler integration, identical to the MLP rollout
training. The reconstruction loss is always computed regardless of `rollout_train_steps`.

**Training time budget**: Each training run should take ~60 minutes. Use `data_augmentation_loop` (DAL) to stay within
this budget. When increasing parameters that scale training time, reduce DAL pre-emptively to
compensate. Check `training_time_min` in results and adjust for the next iteration.

**Do NOT modify `n_epochs`** — keep it at 20. Multiple epochs give us visibility into how losses change during training, which is essential for diagnosing convergence.

**Hard runtime limit (120 minutes)**: The cluster enforces a hard 120-minute wall-clock limit per job. If training approaches this limit, the job receives SIGUSR2 and writes an `_interrupted` file in the run log directory. When analyzing results, check for `_interrupted` in each slot's log directory — if present, training was cut short and the results are from a partial run. Reduce DAL for that config in the next batch to fit within the 120-minute limit. If the job is killed after the grace period without exiting cleanly, you will see `TERM_RUNLIMIT: job killed after reaching LSF run time limit.` in `cluster_train_XX.out` (where XX is the slot number). Otherwise, you will just see the `_interrupted` flag. Do NOT modify `hard_runtime_limit_min` in the config — adjust DAL instead.

**Note**: Seeds are pipeline-controlled and overwritten before each run
(`simulation.seed = iteration * 1000 + slot`, `training.seed = iteration * 1000 + slot + 500`). Do
not set seeds in config files.

> **YAML rule**: Always wrap the `description` field value in double quotes — colons inside unquoted
> YAML strings cause parse errors (e.g., `description: "Block 7 Slot 1: testing latent_dim"`).

## Data Generation

Data is re-generated each iteration with pipeline-controlled seeds (see Note above).

**DO NOT modify simulation parameters** (n_neurons, n_frames, n_edges, delta_t, noise_model_level).

## Block Structure

| Block | Focus                | Parameters to scan                                                     | Ranges                                                                                                                                           |
| ----- | -------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | **Training I**       | `lr`, `batch_size`, `rollout_train_steps`                              | lr: {5e-6, 1e-5, 5e-5, 1e-4}; batch_size: {64, 128, 256, 512}; rollout_train_steps: {1, 5, 10, 20}                                               |
| 2     | **Architecture I**   | `latent_dim`, `n_layers_encoder`, `n_layers_evolver`, `MLP_activation` | latent_dim: {64, 128, 256, 512}; n_layers_encoder: {1, 2, 3, 5}; n_layers_evolver: {1, 2, 3, 5}; activation: {relu, tanh, leaky_relu, soft_relu} |
| 3     | **Training II**      | Refine best training params for Block 2 architecture                   | Narrow ranges around Block 1 winner, re-tune for best architecture from Block 2                                                                  |
| 4     | **Architecture II**  | Any architecture parameter                                             | Re-explore architecture with optimized training from Block 3; refine stimulus encoder, latent capacity                                           |
| 5     | **Free exploration** | Any parameter                                                          | Consolidate best from Blocks 1-4, test novel combinations                                                                                        |
| 6     | **Robustness**       | Best config, all 4 slots same                                          | Confirm CV < 10% across seeds                                                                                                                    |

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

1. Read `train_div_time`, `train_rollout_rmse`, and `train_best_epoch` from the analysis log (these
   are computed on training data — a hard fail if `train_div_time` is < 1000)
2. Read `rollout_RMSE` and `rollout_RMSE_std` from metrics log
3. Read `results_rollout_by_step.csv` — note first window where RMSE exceeds 1.0 (divergence point)
4. Read `onestep_pearson` as sanity check — if poor, model hasn't learned dynamics at all
5. Note `training_time_min` and adjust DAL for next batch if needed
6. **Convergence check**: Compare `train_rollout_rmse` across the last 3 epochs. If it is still
   consistently decreasing at the final epoch, the model has not converged — note
   `convergence: not_reached` in the log entry. Do NOT extend training beyond the ~60 min budget;
   instead, flag the config as a candidate for longer training in the analysis (e.g., "Config may
   benefit from more epochs — train_rollout_rmse still decreasing at epoch 20/20")
7. **Reconstruction check**: If `recon` loss is much larger than `evolve` loss, the autoencoder is
   the bottleneck — consider increasing `latent_dim` or `n_layers_encoder`. If `recon` is very small
   but rollout diverges, the evolver is the bottleneck.

### Step 3: Write Log Entry + Update Memory

```
## Iter N: [excellent/good/poor/diverged]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr=X, DAL=D, batch_size=B, rollout_train_steps=K, latent_dim=L, n_layers_encoder=NE, n_layers_evolver=NV, stim_latent_dims=SL, hidden_dim_stim_encoder=HS, n_layers_stim_encoder=NS, MLP_activation=A
Slot 0: rollout_RMSE=X, divergence_at=frame_Y, train_div_time=Z, train_best_epoch=W/E, onestep_pearson=P, recon=R, evolve=V, sim_seed=S, train_seed=T
Slot 1: rollout_RMSE=X, divergence_at=frame_Y, train_div_time=Z, train_best_epoch=W/E, onestep_pearson=P, recon=R, evolve=V, sim_seed=S, train_seed=T
Slot 2: rollout_RMSE=X, divergence_at=frame_Y, train_div_time=Z, train_best_epoch=W/E, onestep_pearson=P, recon=R, evolve=V, sim_seed=S, train_seed=T
Slot 3: rollout_RMSE=X, divergence_at=frame_Y, train_div_time=Z, train_best_epoch=W/E, onestep_pearson=P, recon=R, evolve=V, sim_seed=S, train_seed=T
Seed stats: mean_RMSE=X, std=Y, CV=Z%, mean_train_div_time=Z, mean_best_epoch=W
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
- Hypothesis: "The EED baseline can achieve stable rollout (RMSE < 1e-2) on flyvis with appropriate
  training parameters (lr, batch_size, rollout_train_steps)"

---

# Working Memory Structure

```markdown
# Working Memory: {llm_task_name}

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | rollout_RMSE (mean+/-std) | CV% | div_time (mean) | divergence_at | Verdict | Hypothesis |
| ---- | -------------- | ----------------------- | --- | --------------- | ------------- | ------- | ---------- |

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
