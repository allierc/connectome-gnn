# FlyVis GNN — Connectome Recovery (noise=0.05), bare start

## File access scope (READ FIRST — saves >5 min per Claude call)

This exploration is **fully self-contained**. The only files you should ever
Read are the ones whose absolute paths are explicitly named in the prompt
(instruction file, working memory, analysis log, user input, the 4 slot
config files). Do not Glob, list, or Read anything else.

**DO NOT** look at:

- Other YAML files in `config/fly/` — there are ~120 of them from unrelated
  experiments. They are not "base configs" or "winners" for this run. The 4
  slot files named in the prompt are the only configs that matter.
- Any directory under `log/remote/Claude_exploration/` other than this
  exploration's own dir (`LLM_flyvis_noise_005_from_zero/`). Other
  explorations use different noise regimes, different baselines, and
  different objectives — their results have **zero transfer value** here
  and looking at them will (a) bias your hypotheses against the
  "discover-from-scratch" mandate stated in the Goal section and
  (b) burn 5–10 minutes of wall clock per call.
- Other instruction files in `LLM/instruction_*.md`.
- Source code under `src/connectome_gnn/` unless a specific failure
  diagnosis explicitly requires it.

**DO** look at:

- This instruction file.
- The working memory file named in the prompt (this exploration's only).
- The analysis log named in the prompt (this exploration's only).
- The 4 slot YAML files named in the prompt.
- The current iteration's `tmp_training/matrix/connectivity_*.png` and
  metrics output, when analyzing results.

If you need a "base config" reference and one is not explicitly named:
**use slot 0's current YAML**. Do not search the directory.

## ⏱ Time budget — SIGTERM at 10 minutes

This Claude call has a **hard wall-clock limit of 10 minutes**. The wrapper
sends SIGTERM at the deadline — *any unfinished Write / Edit is lost, and
the next iteration silently runs with this batch's configs **unchanged***.
A SIGTERMed analysis = a wasted iteration (no HPO progress, no signal).

**Hard rule: finish every YAML Write by the 8-minute mark.** Keep the last
2 minutes as safety slack for Read/Edit latency on the shared filesystem.

Practical pacing (adapt to the phase of the iteration):
- 0–6 min: Read results + analyse the last batch.
- 6–8 min: Decide the next parameter change and Write the 4 slot YAMLs.
- 8–10 min: Only used if a write failed; otherwise exit cleanly.

If you are still analysing at minute 6, stop: **cut analysis short and
write the proposal now**. A narrow, committed parameter change is
infinitely more valuable than a richer analysis that gets killed at 10:00.

## Goal

Optimize the hyperparameters listed below for maximum **connectivity matrix recovery (conn_R2)**
on FlyVis with noise level σ=0.05.

The starting config sets all tunable regularization coefficients to 0 and all learning rates to
a neutral level. No prior optimization results, winner values, block ordering, or parameter
ranges are provided. You must discover what matters from experiment alone.

## Noise Model

```
v_i(t+1) = v_i(t) + dt * f(v_i(t), W, a_i, I_i(t)) + epsilon_i(t)
epsilon_i ~ N(0, sigma)  where sigma = 0.05 (noise_model_level)
```

Noise is added to training data only; test rollouts are noise-free. Do not compare training and
test metrics directly.

## Metrics

During training (stdout):

```
epoch 0/1 | train: ... | conn_R2=0.XXX tau_R2=0.XXX Vr_R2=0.XXX | duration: XXs
```

During test/validation:

- **PRIMARY METRIC: `conn_R2`** (higher is better; R² of learned W vs ground-truth W)
- `tau_R2`: R² of τ (time constant) recovery
- `V_rest_R2`: R² of V_rest (resting potential) recovery
- `cluster_accuracy`: cell-type clustering accuracy from neuron embeddings
- `rollout_pearson_r`: Pearson r of autoregressive rollout vs ground truth

Robustness classification (4 seeds per iteration) — use this, don't invent your own:

- **Stable-Robust**: all 4 seeds conn_R2 ≥ 0.90, CV < 3%
- **Stable**: mean conn_R2 ≥ 0.85, CV < 10%
- **Unstable**: mean < 0.85 OR CV ≥ 10%
- **Catastrophic**: any seed conn_R2 < 0.50

Data is **NOT re-generated** each iteration (`generate_data: false`).

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: form a specific, testable prediction.
2. **Design experiment**: change **EXACTLY ONE** parameter at a time (causality rule).
3. **Run training**: 4 slots (1 control + 3 experiments in EXPLORATION mode; 4 same config in
   ROBUSTNESS mode).
4. **Analyze results**: use conn_R2 and rollout_pearson_r together.
5. **Update understanding**: revise hypotheses based on evidence.

**You can only hypothesize. Only training results validate or falsify.** You do not have access
to prior results, prior winners, or expected magnitudes — every relationship must be evidenced
by your own iterations.

### CAUSALITY RULE (MANDATORY)

**If you change more than one parameter per slot, you cannot attribute the effect. Fatal
experimental design error.**

- EXPLORATION mode: Slot 0 = parent/baseline (unchanged control). Slots 1–3 each change
  **exactly one** parameter from the parent.
- ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness).

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * g(v_j) + I_i(t)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors, DAVIS visual input)
- Noise level: σ=0.05 per time step
- 64,000 frames, delta_t = 0.02
- Model `flyvis_A`: f_theta absorbs τ and V_rest implicitly (τ_R2=0, Vr_R2=0 is expected)

## GNN Architecture

```
g_phi(v_j, embed_j) → message_ij          (edge MLP, per-edge messages)
sum_j W_ij * g_phi(v_j) → agg_i           (weighted aggregation)
f_theta(v_i, agg_i, embed_i) → dv_i/dt   (node update MLP)
```

Per-neuron embedding: learnable `embedding_dim`-dimensional vector concatenated to inputs.
`g_phi_positive=true`: g_phi output clipped to [0, ∞).

**YOU ARE ONLY ALLOWED TO MODIFY THE PARAMETERS BELOW.**

## GNN Architecture Parameters

| Parameter       | Starting value | Description                                             |
| --------------- | -------------- | ------------------------------------------------------- |
| `hidden_dim`    | 80             | Width of hidden layers in g_phi and f_theta             |
| `n_layers`      | 3              | Depth of g_phi and f_theta networks                     |
| `embedding_dim` | 2              | Per-neuron learnable embedding dimension                |

## Training Parameters

| Parameter                 | Starting value | Description                                                                      |
| ------------------------- | -------------- | -------------------------------------------------------------------------------- |
| `lr_W`                    | 0.0006         | Learning rate for W matrix (synaptic weights)                                    |
| `lr`                      | 0.0012         | Learning rate for g_phi and f_theta MLP weights                                  |
| `lr_embedding`            | 0.00155        | Learning rate for per-neuron embeddings                                          |
| `data_augmentation_loop`  | 35             | Augmentation loops per epoch — controls training time (DAL)                      |
| `batch_size`              | 4              | Samples per batch                                                                |
| `coeff_g_phi_diff`        | 0              | L2 penalty pulling g_phi away from the trivial constant solution                 |
| `coeff_g_phi_norm`        | 0              | L2 norm regularization on g_phi output values                                    |
| `coeff_g_phi_weight_L1`   | 0              | L1 weight regularization on g_phi network                                        |
| `coeff_f_theta_weight_L1` | 0              | L1 weight regularization on f_theta network                                      |
| `coeff_f_theta_weight_L2` | 0              | L2 weight regularization on f_theta network                                      |
| `coeff_W_L1`              | 0              | L1 regularization on W                                                           |
| `coeff_W_L2`              | 0              | L2 regularization on W                                                           |
| `regul_annealing_rate`    | 0.0            | Regularization annealing: **MUST be 0.0 with n_epochs=1** (otherwise all L1/L2=0) |
| `w_init_mode`             | `randn_scaled` | W initialization: `randn_scaled`, `zeros`, `uniform_scaled`                      |
| `w_init_scale`            | 1.0            | Scale for randn_scaled/uniform_scaled init (bound = scale/sqrt(n_edges))         |

## Simulation Parameters (sweep-only with regenerate)

Changing any parameter in this group **requires re-simulating the ground-truth
voltage traces** for that slot. The pipeline does this automatically when the
slot's YAML has `generate_data: true` for one iteration. See the "Data
Generation" section below for the procedure.

| Parameter                 | Starting value | Description                                                                |
| ------------------------- | -------------- | -------------------------------------------------------------------------- |
| `blank_prefix_fraction`   | 0.0            | Fraction of each input sequence blanked at the start (0.0 = no blanks; 0.1 = first 10 % zero stimulus). Sweeping this changes the stimulus statistics the network sees during both training and test, so it must be regenerated. |

Hard facts about the code (not hints — infrastructure):

- `n_epochs=1` is fixed. With `n_epochs=1`, `regul_annealing_rate` must be 0.0 (annealing
  formula: `effective_coeff = coeff × (1 − exp(−rate × epoch))` = 0 at epoch 0).
- Seeds are pipeline-controlled (`sim_seed = iter × 1000 + slot`,
  `train_seed = iter × 1000 + slot + 500`). Do not set seeds in config files.
- **Training time budget**: target ~30–45 min per run; check `training_time_min` after each
  iteration and adjust DAL if systematically off.
- **Hard runtime limit (60 min)**: cluster kills jobs at wall-clock 60 min. If `_interrupted`
  appears in a slot log directory, reduce DAL.

> **YAML rule**: always wrap the `description` field value in double quotes — colons inside
> unquoted YAML strings cause parse errors.

## Data Generation

By default `generate_data: false` in every slot — the ground-truth voltage
traces are simulated once, cached, and reused across iterations. **Never
touch the frozen simulation parameters**: `n_neurons`, `n_frames`,
`n_edges`, `delta_t`, `noise_model_level`.

### When to regenerate (`generate_data: true`)

To sweep a **Simulation Parameter** (see the table above — currently only
`blank_prefix_fraction`), the ground-truth traces must be re-simulated
with the new value. Procedure for one iteration:

1. In that slot's YAML, set both the new simulation-parameter value **and**
   `generate_data: true`. Leave the other 3 slots unchanged (control + two
   other experiments, or 3 controls if you want a clean A/B/C comparison).
2. Run the iteration. The pipeline regenerates that slot's cached data
   and then trains; runtime ≈ DAL + ~10–15 min regen.
3. **Flip `generate_data: false` back in that slot's YAML for the next
   iteration.** Leaving it on burns ~10 min/iter forever and hides
   genuine HPO signal behind regen-noise.

### Causality note for simulation sweeps

A `blank_prefix_fraction` change is a **simulation change**, not an HPO
change — so the comparison is *different data vs. different data*, not
*same data vs. same data*. To keep causality clean:

- **Control slot (slot 0)**: keep `blank_prefix_fraction = parent value`,
  `generate_data: false`. This is the baseline the comparison is measured
  against.
- **Experiment slot(s)**: new `blank_prefix_fraction`,
  `generate_data: true`, everything else identical to control.
- **Seeds** are pipeline-controlled, so slot i's regenerated data uses the
  iteration's sim_seed — reproducible, but **different** from slot 0's
  cached data. Report the effect size with care (it is the *combined*
  effect of stimulus change + sim_seed change).

If you see a large effect from a `blank_prefix_fraction` sweep, run a
ROBUSTNESS block at the new value (4 slots same config, `generate_data:
true` for the first iteration only, then back to `false`) before declaring
a winner.

## Block Structure

There is no pre-defined block agenda. You design the exploration blocks yourself based on
evidence gathered iteration by iteration. Use your own judgement about which parameter to sweep
first, what ranges to try, and when to move on.

Two modes are available:

- **EXPLORATION**: Slot 0 = current parent, Slots 1–3 change one parameter each.
- **ROBUSTNESS**: all 4 slots identical, different seeds.

Guidance on when to switch modes:

- Start ROBUSTNESS if you need to measure variance of the current parent.
- Stay in EXPLORATION while conn_R2 is still improving.
- Before declaring a final winner, run a ROBUSTNESS block (`generate_data: false`) and a
  CV-robustness block (`generate_data: true`, 8 seeds over 2 iterations). The `generate_data`
  flag must be reset to `false` after the CV block.

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

- Patterns observed 1–2 times
- Seed-dependent effects
- Contradictions between iterations

### What to Add to Falsified Hypotheses

1. State the original hypothesis
2. State the contradicting evidence (iteration number, metrics)
3. State what was learned from the falsification
4. Propose a revised hypothesis if applicable

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

For each slot:

1. Read `conn_R2`, `tau_R2`, `V_rest_R2`, `cluster_accuracy`, `rollout_pearson_r` from metrics log
2. Check `training_time_min` — adjust DAL if > 70 min or < 50 min
3. Check for `_interrupted` in slot log directory
4. Classify: Stable-Robust / Stable / Unstable / Catastrophic

### Step 3: Write Log Entry + Update Memory

```
## Iter N: [stable_robust/stable/unstable/catastrophic]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: [full list of non-default parameters]
Slot 0: conn_R2=X, tau_R2=Y, Vr_R2=Z, cluster_acc=W, rollout_r=P, sim_seed=S, train_seed=T
Slot 1: conn_R2=X, tau_R2=Y, Vr_R2=Z, cluster_acc=W, rollout_r=P, sim_seed=S, train_seed=T
Slot 2: conn_R2=X, tau_R2=Y, Vr_R2=Z, cluster_acc=W, rollout_r=P, sim_seed=S, train_seed=T
Slot 3: conn_R2=X, tau_R2=Y, Vr_R2=Z, cluster_acc=W, rollout_r=P, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%, catastrophic=N/4
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive]
Next: parent=P
```

### Step 4: Acknowledge User Input

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Start Call

When prompt says `PARALLEL START`:

- Slot 0 = the starting config (all values in the tables above).
- Mode for the first iteration: ROBUSTNESS — slots 1–3 also use the same starting config
  (different seeds) so you have a variance estimate before you start changing anything.
- Hypothesis: write your own based solely on the code description; you have no benchmarks.
- Launch:
  `python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_from_zero iterations=80 --cluster --resume`

---

## Final Summary

At exploration completion append to
`/home/node/.claude/projects/-workspace--devcontainer/memory/exploration_results.md`:

### flyvis_noise_005_from_zero — Key Discoveries (YYYY-MM-DD)

1. Starting conn_R2 of the bare baseline (mean ± std, 4 seeds)
2. Final best conn_R2 (mean ± std, CV%, N seeds) and the full winning config
3. Iteration number at which the first parameter change produced a statistically clear
   improvement, and which parameter it was
4. Ranking of parameters by magnitude of their largest single-parameter improvement
5. Any failure mode confirmed across 3+ iterations
6. Any hypothesis that was falsified and what was learned from it
7. Any surprising or counter-intuitive HP interaction

---

# Working Memory Structure

```markdown
# Working Memory: {llm_task_name}

## Paper Summary (update at every block boundary)

Two sentences on HPO findings:

Sentence 1: best config found + the conn_R2 it achieves (mean ± std, CV%, N seeds).
Sentence 2: which parameter had the largest single-parameter impact and in what direction.

Two sentences on exploration findings:

Sentence 1: what the systematic exploration revealed about the optimization landscape (basin
width, failure modes, interactions).
Sentence 2: main causal principle established from hypothesis testing.

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean±std) | CV% | catastrophic | Verdict | Hypothesis |
| ---- | -------------- | ------------------- | --- | ------------ | ------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

---

## Previous Block Summaries

**RULE: keep summaries for the last 4 completed blocks, sorted oldest → newest. This section
MUST appear before `## Current Block`.**

### Block N Summary

[Summary of findings from block N]

---

## Current Block

### Block Info

### Current Hypothesis

### Iterations This Block

### Emerging Observations

**CRITICAL: this section must ALWAYS be at the END of memory file.**
```
