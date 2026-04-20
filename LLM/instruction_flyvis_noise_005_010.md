# FlyVis GNN Training Exploration — flyvis_noise_005_010

## Goal

Test **robustness of GNN training** for the **Drosophila visual system** under **dual noise**: dynamics noise (0.05) + **high measurement noise (0.10)** with DAVIS input.
The goal is to find a config that achieves **connectivity_R2 > 0.80 across different random seeds**, demonstrating that the result is not data-dependent.

Data is **re-generated each iteration** with a different seed to verify seed independence.

Primary metric: **connectivity_R2** (R² between learned W and ground-truth W).
Secondary metrics: **tau_R2** (time constant recovery), **V_rest_R2** (resting potential recovery), **cluster_accuracy** (neuron type clustering from embeddings).

## Noise Model

This exploration trains under **two independent noise sources**:

1. **Dynamics noise** (`noise_model_level=0.05`): Additive Gaussian noise injected into the Euler integration at each timestep. This simulates biological variability in membrane dynamics. The ODE integration becomes:
   ```
   v(t+1) = v(t) + dt * f(v, W, I) + epsilon_dyn(t),   epsilon_dyn ~ N(0, 0.05)
   ```

2. **Measurement noise** (`measurement_noise_level=0.10`): Additive Gaussian noise applied to observed voltages. This simulates imperfect recording instruments. The GNN sees:
   ```
   v_obs(t) = v_clean(t) + epsilon_meas(t),   epsilon_meas ~ N(0, 0.10)
   ```

### Impact on training data

- **Input voltages**: The GNN receives `v_obs = v_clean + epsilon_meas` — noisy observations
- **Derivative targets**: Computed from noisy observations, so they contain a noise difference term:
  ```
  y_noisy(t) = y_clean(t) + (epsilon_meas(t+1) - epsilon_meas(t)) / dt
  ```
  This derivative noise has std ~ `measurement_noise_level * sqrt(2) / dt = 0.10 * 1.414 / 0.02 ~ 7.07`, which is **very large** relative to clean derivative magnitudes.
- **Test evaluation**:
  - **connectivity_R2** (primary metric): Compares the learned W matrix directly against the ground-truth W matrix. This is independent of test data — it's a pure parameter comparison, so noise does not affect the evaluation itself, only the GNN's ability to learn W correctly.
  - **test_R2**: Evaluated on pre-generated test data that has dynamics noise (`noise_model_level=0.05`) but **no measurement noise** added. The test voltages are the same as the `flyvis_noise_005` case.

### Key challenges compared to lower measurement noise (0.04)

- The derivative noise (~7.07 std) is **2.5x larger** than the 0.04 case (~2.83 std) — this severely corrupts the training signal
- The GNN must average out much more noise — **data_augmentation_loop** and **batch_size** become critical
- Stronger regularization is likely needed to prevent overfitting to noisy derivatives
- The performance ceiling will be **lower** than the 0.04 case — expect significantly reduced connectivity_R2
- `coeff_g_phi_diff` (monotonicity) is particularly important: it constrains the function space, which helps when targets are very noisy

### Comparison with flyvis_noise_005_004

| Metric | noise_005_004 (meas=0.04) | noise_005_010 (meas=0.10) |
| --- | --- | --- |
| Derivative noise std | ~2.83 | ~7.07 |
| Expected connectivity_R2 ceiling | ~0.85-0.90 | ~0.70-0.85 (TBD) |
| Expected difficulty | Moderate | High |

## Scientific Method

This exploration follows a strict **hypothesize -> test -> validate/falsify** cycle:

1. **Hypothesize**: Based on available data (metrics, seed variance, prior results), form a hypothesis about what controls robustness (e.g., "Stronger coeff_g_phi_diff will reduce seed variance because monotonicity constraints limit the space of solutions the GNN can overfit to")
2. **Design experiment**: Choose a mutation that specifically tests the hypothesis — change ONE parameter at a time
3. **Run training**: The experiment runs across 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use both metrics AND cross-seed variance to evaluate whether the hypothesis was supported or contradicted
5. **Update understanding**: Revise hypotheses based on evidence. A falsified hypothesis is valuable information.

**CRITICAL**: You can only hypothesize. Only training results can validate or falsify your hypotheses. Never assume a hypothesis is correct without experimental evidence. When results contradict your hypothesis, update it — do not rationalize away the evidence.

**Evidence hierarchy:**

| Level            | Criterion                                       | Action                 |
| ---------------- | ----------------------------------------------- | ---------------------- |
| **Established**  | Consistent across 3+ iterations AND 4/4 seeds   | Add to Principles      |
| **Tentative**    | Observed 1-2 times or inconsistent across seeds | Add to Open Questions  |
| **Contradicted** | Conflicting evidence across iterations/seeds    | Note in Open Questions |

## CRITICAL: Data is RE-GENERATED per slot

Each slot re-generates its data with a **different random seed**.
Both `simulation.seed` and `training.seed` are **forced by the pipeline** — DO NOT modify them in config files.

Seed formula (set automatically by GNN_LLM.py):

- `simulation.seed = iteration * 1000 + slot` (controls data generation AND measurement noise realization)
- `training.seed = iteration * 1000 + slot + 500` (controls weight init & training randomness)

The actual seed values are provided in the prompt for each slot — **log them in your iteration entries**.

Simulation parameters (n_neurons, n_frames, noise levels, etc.) stay fixed — **DO NOT change them**.

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i(t)
dv_i/dt = f_theta(v_i, a_i, sum_j W_ij * g_phi(v_j, a_j)^2, I_i)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors)
- DAVIS visual input, **noise_model_level=0.05**, **measurement_noise_level=0.10**
- 64,000 frames, delta_t=0.02

## GNN Architecture

Two MLPs learn the neural dynamics:

- **g_phi** (MLP1): Edge message function. Maps (v_j, a_j) -> message. `g_phi_positive=true` squares output to enforce positivity.
- **f_theta** (MLP0): Node update function. Maps (v_i, a_i, aggregated_messages, I_i) -> dv_i/dt.
- **Embedding a_i**: learnable low-dimensional embedding per neuron type.

Architecture parameters (explorable):

- `hidden_dim` / `n_layers`: g_phi MLP width/depth (default: 80 / 3)
- `hidden_dim_update` / `n_layers_update`: f_theta MLP width/depth (default: 80 / 3)
- `embedding_dim`: embedding dimension (default: 2)

**CRITICAL — coupled parameters**: When changing `embedding_dim`, you MUST also update:

- `input_size = 1 + embedding_dim` (v_j + a_j for g_phi)
- `input_size_update = 3 + embedding_dim` (v_i + a_i + msg + I_i for f_theta)

Example: embedding_dim=4 -> input_size=5, input_size_update=7. Shape mismatch crashes otherwise.

## Regularization Parameters

The training loss includes:

| Config parameter          | Role                                                                                | Default | Annealed? |
| ------------------------- | ----------------------------------------------------------------------------------- | ------- | --------- |
| `coeff_g_phi_diff`        | Monotonicity penalty on g_phi: ReLU(-dg_phi/dv) -> enforces increasing edge messages | 1200    | No        |
| `coeff_g_phi_norm`        | Normalization penalty on g_phi at saturation voltage                                | 0.9     | No        |
| `coeff_g_phi_weight_L1`   | L1 penalty on g_phi MLP weights                                                     | 0.28    | **Yes**   |
| `coeff_g_phi_weight_L2`   | L2 penalty on g_phi MLP weights                                                     | 0       | **Yes**   |
| `coeff_f_theta_weight_L1` | L1 penalty on f_theta MLP weights                                                   | 0.05    | **Yes**   |
| `coeff_f_theta_weight_L2` | L2 penalty on f_theta MLP weights                                                   | 0.001   | **Yes**   |
| `coeff_f_theta_msg_diff`  | Monotonicity of f_theta w.r.t. message input                                        | 0       | No        |
| `coeff_W_L1`              | L1 sparsity penalty on connectivity W                                               | 0.00015 | **Yes**   |
| `coeff_W_L2`              | L2 penalty on W                                                                     | 1.5e-06 | **Yes**   |

### Regularization Annealing

All 6 weight regularization coefficients (L1 and L2 for g_phi, f_theta, and W) share a **single exponential ramp-up annealing** controlled by one parameter:

| Config parameter        | Default | Description                                    |
| ----------------------- | ------- | ---------------------------------------------- |
| `regul_annealing_rate`  | 0.5     | Shared annealing rate for all L1/L2 regularizers |

**Formula**: `effective_coeff = coeff * (1 - exp(-rate * epoch))`

**Ramp-up schedule** (rate=0.5):

| Epoch | Multiplier | Meaning                        |
| ----- | ---------- | ------------------------------ |
| 0     | 0.00       | No regularization at start     |
| 1     | 0.39       | ~39% of configured coefficient |
| 2     | 0.63       | ~63%                           |
| 5     | 0.92       | ~92% (near full strength)      |
| 10    | 0.99       | ~full strength                 |

**Purpose**: Allows the model to learn dynamics first before regularization pressure is applied. At epoch 0, all L1/L2 penalties are zero regardless of their configured coefficients.

**CRITICAL — 1-epoch training**: With `n_epochs=1` (the default for this exploration), training runs only epoch 0, where the annealing multiplier is exactly **0.00**. This means:

**INACTIVE at epoch 0** (configured values have NO effect):
- `coeff_g_phi_weight_L1` (0.28) — zero effective
- `coeff_g_phi_weight_L2` (0) — zero effective
- `coeff_f_theta_weight_L1` (0.05) — zero effective
- `coeff_f_theta_weight_L2` (0.001) — zero effective
- `coeff_W_L1` (0.00015) — zero effective
- `coeff_W_L2` (1.5e-06) — zero effective

**ACTIVE at epoch 0** (full strength regardless of annealing):
- `coeff_g_phi_diff` (1200) — monotonicity constraint
- `coeff_g_phi_norm` (0.9) — normalization constraint
- `coeff_f_theta_msg_diff` (0) — f_theta monotonicity

**To enable L1/L2 regularizers**, either:
1. Set `n_epochs: 2` and halve `data_augmentation_loop` to keep training time constant (epoch 0: no L1/L2, epoch 1: 39% strength)
2. Set `regul_annealing_rate: 0` to disable annealing entirely (full L1/L2 from epoch 0)

**Bottom line**: With the default 1-epoch setup, only `coeff_g_phi_diff`, `coeff_g_phi_norm`, and `coeff_f_theta_msg_diff` are the effective regularization knobs. Changing L1/L2 coefficients without also changing `n_epochs` or `regul_annealing_rate` has zero effect.

### Noise-Specific Regularization Considerations

Under high measurement noise (0.10), the noisy derivative targets have very high variance (~7.07 std from noise alone). This affects regularization in several ways:

- **coeff_g_phi_diff** (monotonicity): Likely needs to be **significantly higher** than the 0.04 case to constrain the solution space and prevent overfitting to very noisy targets
- **coeff_W_L1** (connectivity sparsity): May need tuning — too aggressive L1 could prune real connections that are hard to detect under high noise, too weak allows spurious connections from noise correlations
- **coeff_g_phi_weight_L1 / coeff_f_theta_weight_L1**: Weight sparsity helps prevent overfitting; may benefit from being higher than the 0.04 case
- **data_augmentation_loop**: More passes over different temporal windows helps average out noise — consider increasing substantially
- **batch_size**: Larger batches provide more averaging per gradient step — may be particularly beneficial under high noise

## Training Parameters (explorable)

| Parameter                       | Default      | Description                                  |
| ------------------------------- | ------------ | -------------------------------------------- |
| `learning_rate_W_start`         | 0.0009       | Learning rate for connectivity matrix W      |
| `learning_rate_start`           | 0.0018       | Learning rate for g_phi and f_theta MLPs     |
| `learning_rate_embedding_start` | 0.002325     | Learning rate for neuron embeddings          |
| `n_epochs`                      | 1 (claude)   | Epochs per iteration (keep <= 2 for time)     |
| `batch_size`                    | 6            | Batch size for training                      |
| `data_augmentation_loop`        | 30           | Data augmentation multiplier                 |
| `recurrent_training`            | false        | Multi-step rollout training (see below)      |
| `time_step`                     | 1            | Number of autoregressive integration steps per training sample |
| `noise_recurrent_level`         | 0.0          | Noise added during recurrent rollout steps   |
| `w_init_mode`                   | randn_scaled | W initialization: "zeros" or "randn_scaled"  |
| `lr_scheduler`                  | none         | LR schedule: "none", "cosine_warm_restarts", "linear_warmup_cosine" |
| `lr_scheduler_T0`               | 1000         | First restart period in iterations (cosine schedulers only) |
| `lr_scheduler_T_mult`           | 2            | Period multiplier after each restart         |
| `lr_scheduler_eta_min_ratio`    | 0.01         | Min LR as fraction of base LR               |
| `lr_scheduler_warmup_iters`     | 100          | Linear warmup iterations (linear_warmup_cosine only) |

### Recurrent Training & time_step

When `recurrent_training: true`, the model integrates `time_step` steps forward autoregressively during training and compares to the ground-truth voltage at frame `k + time_step`. This teaches the model to produce stable multi-step dynamics rather than just accurate single-frame derivatives.

- `time_step: 1` — equivalent to non-recurrent (predict one step ahead)
- `time_step: 5-10` — the model must maintain accuracy over multiple integration steps
- Higher `time_step` increases training time roughly proportionally
- `noise_recurrent_level` (default 0.0) can add noise during rollout steps for robustness

**Under high measurement noise**: recurrent training may help because it optimizes multi-step consistency rather than fitting noisy single-frame derivatives. The noisy derivative signal (std ~7.07) is hard to fit directly, but multi-step rollout targets are less affected by per-frame noise.

### LR Scheduler Notes

When `lr_scheduler="none"` (default), per-iteration LR is constant and the legacy epoch-level halving (every 10 epochs) remains active. When a scheduler is enabled, it steps **per iteration** (not per epoch), so the LR oscillates within each epoch.

**Recommended exploration**: `cosine_warm_restarts` with `T0=500-2000` provides periodic LR restarts that can help escape local minima. `linear_warmup_cosine` adds a warmup ramp for stability with large initial LR. The `T0` parameter controls how frequently the LR resets — smaller T0 means more frequent restarts.

## Training Time Constraint

Baseline (batch_size=6, 64K frames, hidden_dim=80): **~60 min/epoch on A100**.
Data generation adds **~10-15 min** per slot.
Keep total training time (generation + training) <= 100 min/iteration. Monitor `training_time_min`.

Factors that increase training time:

- Larger `hidden_dim` / `n_layers`
- Larger `data_augmentation_loop`
- Smaller `batch_size`
- `recurrent_training=true` with large `time_step`

## Parallel Mode — 4 Slots Per Batch

You receive **4 results per batch** and propose **4 mutations** for the next batch.
Each slot runs with a **different random seed** for data generation, so you can directly assess seed robustness within a single batch.

### Robustness Assessment

After each batch, evaluate:

- **Robust**: all 4 slots have connectivity_R2 > 0.80
- **Partially robust**: 2-3 slots have connectivity_R2 > 0.80
- **Fragile**: 0-1 slots have connectivity_R2 > 0.80

A config is considered **validated** only when it achieves connectivity_R2 > 0.80 on all 4 seeds.

**Note**: The robustness threshold is set at 0.80 (vs 0.85 for 0.04 noise, vs 0.9 for noise-free) because higher measurement noise creates a larger irreducible error floor. This may be revised based on actual results.

### Slot Strategy

Since the goal is robustness testing, all 4 slots should run the **same config** (different seeds are applied automatically).

| Slot | Role          | Description                            |
| ---- | ------------- | -------------------------------------- |
| 0    | **seed test** | Same config, seed varies automatically |
| 1    | **seed test** | Same config, seed varies automatically |
| 2    | **seed test** | Same config, seed varies automatically |
| 3    | **seed test** | Same config, seed varies automatically |

When a config is validated as robust (all 4 seeds > 0.80), you may switch to exploring a variation:

| Slot | Role        | Description                                             |
| ---- | ----------- | ------------------------------------------------------- |
| 0-3  | **exploit** | All 4 slots test the next candidate config across seeds |

### Config Files

- Edit all 4 config files: `{name}_00.yaml` through `{name}_03.yaml`
- **All 4 configs should be identical** (only seeds differ, set automatically)
- Only modify `training:` and `graph_model:` parameters (and `claude:` where allowed)
- **DO NOT change `simulation:` parameters** (except that seed is managed automatically)

## Iteration Loop Structure

Each block = `n_iter_block` iterations (default 12).
The prompt provides: `Block info: block {block_number}, iterations {iter_in_block}/{n_iter_block} within block`

### Interactive Code Phase (interaction_code=true)

This exploration has `interaction_code: true`. At every block boundary, the script enters an **interactive code modification session** before running the next block:

1. The LLM generates a structured **code brief** based on accumulated knowledge
2. The brief is presented to the human for review
3. The human can provide feedback, approve code changes, or skip
4. If code changes are applied, the human must git push locally and git pull on the cluster before continuing

The purpose is to allow **structural code changes** (not just hyperparameter tuning) between blocks — for example, modifying the training loss, adding noise-aware processing, or changing the GNN architecture.

## File Structure

You maintain **THREE** files:

### 1. Full Log (append-only)

**File**: `{llm_task_name}_analysis.md`

- Append every iteration's log entry (4 entries per batch)
- Append block summaries at block boundaries
- **Never read** — human record only

### 2. Working Memory (read + update every batch)

**File**: `{llm_task_name}_memory.md`

- Read at start, update at end
- Contains: robustness comparison table, hypotheses, established principles, current block iterations
- Keep <= 500 lines

### 3. User Input (read every batch, acknowledge pending items)

**File**: `user_input.md`

- Read at every batch
- If "Pending Instructions" section has content: act on it, then move entries to "Acknowledged" section with timestamp
- Do not remove acknowledged entries — append them with `[ACK {batch}]` marker

## Iteration Workflow (every batch)

### Step 1: Read Working Memory + User Input

- Read `{llm_task_name}_memory.md` for context — especially hypotheses and robustness table
- Read `user_input.md` for any pending user instructions

### Step 2: Analyze Results (4 slots)

**Metrics from `analysis.log`:**

- `connectivity_R2`: R2 of learned vs true W (PRIMARY)
- `tau_R2`: R2 of learned vs true time constants
- `V_rest_R2`: R2 of learned vs true resting potentials
- `cluster_accuracy`: neuron type clustering accuracy from embeddings
- `test_R2`: one-step prediction R2
- `training_time_min`: training duration

**Robustness classification (across all 4 seeds):**

- **Robust**: all 4 slots connectivity_R2 > 0.80
- **Partially robust**: 2-3 slots connectivity_R2 > 0.80
- **Fragile**: 0-1 slots connectivity_R2 > 0.80

**Per-slot classification:**

- **Converged**: connectivity_R2 > 0.80
- **Partial**: connectivity_R2 0.3-0.80
- **Failed**: connectivity_R2 < 0.3

**Seed variance analysis (compute every batch):**

- Compute mean, std, and CV (coefficient of variation = std/mean) for connectivity_R2 across the 4 slots
- CV < 5% -> highly stable; CV 5-15% -> moderate variance; CV > 15% -> seed-sensitive

**UCB scores from `ucb_scores.txt`:**

- UCB(k) = R2_k + c * sqrt(ln(N) / n_k) where c = `ucb_c` (default 1.414)
- At block boundaries the UCB file is empty — use `parent=root`

### Step 3: Write Log Entries + Update Hypotheses + Update Memory

**3a. Append to Full Log** (`{llm_task_name}_analysis.md`) and **Current Block** in memory.md:

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis being tested]"
Config (same for all slots): lr_W=X, lr=Y, lr_emb=Z, coeff_g_phi_diff=A, coeff_W_L1=B, batch_size=C, hidden_dim=D
Slot 0: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, sim_seed=S, train_seed=T
Slot 1: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, sim_seed=S, train_seed=T
Slot 2: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, sim_seed=S, train_seed=T
Slot 3: connectivity_R2=A, tau_R2=B, V_rest_R2=C, cluster_accuracy=D, test_R2=E, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive] — [one line explanation]
Observation: [one line about seed sensitivity or robustness pattern]
Next: parent=P
```

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder — always include exact parameter change.
**CRITICAL**: `Next: parent=P` — P must be from a previous batch or current batch, NEVER `id+1`.

**3b. Update Hypotheses in memory.md:**

After analyzing results, update the `## Hypotheses` section:

- If results **support** the hypothesis -> increase confidence, note supporting evidence
- If results **falsify** the hypothesis -> mark as falsified, formulate a new hypothesis informed by the contradicting evidence
- If results are **inconclusive** -> note what additional experiment would clarify

**3c. Update Robustness Comparison Table in memory.md** (see Working Memory Structure below).

### Step 4: Acknowledge User Input (if any)

If `user_input.md` has content in "Pending Instructions":

- Edit `user_input.md`: move the pending items to "Acknowledged" with `[ACK batch_{batch_first}-{batch_last}]` prefix
- Incorporate the instructions into your next config mutations

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

1. Based on current understanding, formulate the **next hypothesis** to test
2. Design a config mutation that specifically tests this hypothesis (ONE parameter change)
3. All 4 configs should be **identical** — the pipeline assigns different seeds automatically
4. Write the hypothesis to memory.md before editing configs

## Block Partition (suggested)

| Block | Focus                           | Parameters                                                               |
| ----- | ------------------------------- | ------------------------------------------------------------------------ |
| 1     | Baseline under high noise       | Default config across 4 seeds — establish high measurement noise baseline |
| 2     | Regularization for very noisy data | coeff_g_phi_diff (try much higher), coeff_g_phi_weight_L1, coeff_W_L1   |
| 3     | Data augmentation & batch size  | data_augmentation_loop (increase), batch_size (increase for noise averaging) |
| 4     | Learning rates under high noise | lr_W, lr, lr_emb (may need lower LR due to very noisy gradients)        |
| 5     | f_theta regularization          | coeff_f_theta_weight_L1, coeff_f_theta_weight_L2, coeff_f_theta_msg_diff |
| 6     | Architecture & multi-epoch      | hidden_dim, n_layers, n_epochs (more capacity or training may help)      |
| 7     | Combined best                   | Best parameters from blocks 1-6                                          |
| 8     | Validation                      | Re-run best config with more seeds / longer training                     |

## Block Boundaries

At the end of each block:

1. Update "Paper Summary" at the top of memory.md — rewrite both bullet points to reflect the current state of knowledge
2. Summarize findings in memory.md "Previous Block Summary"
3. Update "Established Principles" with confirmed insights (require 3+ supporting iterations AND cross-seed consistency)
4. Move falsified hypotheses to "Falsified Hypotheses" with evidence summary
5. Clear "Current Block" for next block
6. Carry forward best **robust** config as starting point

## Failed Slots

If a slot is `[FAILED]`:

- Write a brief note in the log entry
- A single slot failure may indicate seed sensitivity — note this
- Still propose the next config for the next batch
- Do not draw conclusions from a single failure

## Known Results (prior experiments)

- `flyvis_62_1` (DAVIS + dynamics noise 0.05, no measurement noise): connectivity_R2=0.95, tau_R2=0.80, V_rest_R2=0.40 (10 epochs, full regularization)
- `flyvis_noise_005` (dynamics noise 0.05, no measurement noise, 1 epoch): connectivity_R2 ~ 0.95-0.98 (champion config)
- `flyvis_noise_005_004` (dynamics noise 0.05 + measurement noise 0.04): exploration ongoing — expect connectivity_R2 ceiling ~0.85-0.90
- W initialization: `randn_scaled` and `zeros` perform similarly; plain `randn` performs poorly
- Larger MLP (80-dim/3-layer) works better than smaller (32-dim/2-layer)
- `coeff_g_phi_diff` (monotonicity) is among the most important regularizers — too low causes non-monotonic messages
- Higher noise requires stronger regularization to constrain the solution space
- **Measurement noise 0.10 is 2.5x higher than 0.04** — expect significantly lower connectivity_R2 ceiling. Derivative noise std ~7.07 vs ~2.83.

## Start Call

When prompt says `PARALLEL START`:

- Read base config to understand training regime
- Note the dual noise setup: dynamics noise 0.05 + measurement noise 0.10
- Set all 4 configs **identically** to the baseline config
- Data will be generated with different seeds per slot automatically (each seed produces different noise realizations)
- Write planned config and **initial hypothesis** to working memory
- First iteration establishes baseline robustness — do not change hyperparameters yet
- State the baseline hypothesis: "The default config (from 0.04 exploration) achieves connectivity_R2 > 0.80 robustly across seeds under high measurement noise (0.10)"

---

# Working Memory Structure

The memory file (`{llm_task_name}_memory.md`) must follow this structure:

```markdown
# Working Memory: flyvis_noise_005_010

## Paper Summary (update at every block boundary)

Brief paragraph for a scientific paper summarizing the current state of this exploration:

- **GNN optimization under high measurement noise**: [What has been learned about training GNN to recover Drosophila visual system connectivity from doubly-noisy DAVIS data (dynamics noise 0.05 + measurement noise 0.10). Best connectivity_R2 achieved, which regularization/learning rate regimes work, how high measurement noise impacts recoverability compared to 0.04 and noise-free cases, key challenges.]
- **LLM-driven exploration**: [How the LLM-in-the-loop approach performed as an automated hyperparameter search strategy. Number of iterations run, how hypothesis-driven exploration compared to random search, whether the LLM discovered non-obvious parameter interactions.]
- **Future works**: [what would a google deepmind senior ML suggest as structural change in the code to break ceiling with good rationale, this can include modified GNN class, innovative training scheme, use recent innovations. Cite references of scientific publications, blog, youtube channel. Limit to 10 suggestions.]

## Knowledge Base (accumulated across all blocks)

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean+/-std) | CV% | min | max | tau_R2 (mean) | V_rest_R2 (mean) | Robust? | Hypothesis tested |
| ---- | -------------- | -------------------- | --- | --- | --- | ------------- | ---------------- | ------- | ----------------- |
| 1    | defaults       | ?                    | ?   | ?   | ?   | ?             | ?                | ?       | baseline          |

### Established Principles

[Confirmed patterns — require 3+ supporting iterations AND cross-seed consistency]

Examples of good principles:

- "coeff_g_phi_diff >= 1500 is necessary for robust convergence under high measurement noise (3/3 iterations, all seeds > 0.80)"
- "lr_W > 1e-3 causes seed-dependent failures under high noise (CV > 20% in 2 iterations)"

### Falsified Hypotheses

[Hypotheses that were contradicted by experimental evidence — keep as record]

### Open Questions

[Patterns needing more testing, contradictions, seed-dependent observations]
[Comparisons with 0.04 measurement noise results (from flyvis_noise_005_004 exploration)]

---

## Previous Block Summary (Block N-1)

[Short summary: 2-3 lines. NOT individual iterations.]

---

## Current Block (Block N)

### Block Info

Focus: [which parameter subspace]
Iterations: M to M+n_iter_block

### Current Hypothesis

**Hypothesis**: [specific, testable prediction]
**Rationale**: [why you believe this, based on prior evidence]
**Test**: [what config change tests this]
**Expected outcome**: [what would support vs falsify]
**Status**: untested / supported / falsified / revised

### Iterations This Block

[Current block iterations — cleared at block boundary]

### Emerging Observations

[Running notes on what patterns are emerging across seeds and iterations]
**CRITICAL: This section must ALWAYS be at the END of memory file.**
```

---

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
- Comparisons with 0.04 measurement noise results (from flyvis_noise_005_004 exploration)
- Comparisons with dynamics-only noise results (from flyvis_noise_005 exploration)

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:

1. State the original hypothesis
2. State the contradicting evidence (iteration number, metrics)
3. State what was learned from the falsification
4. Propose a revised hypothesis if applicable
