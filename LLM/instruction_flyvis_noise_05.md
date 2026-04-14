# FlyVis GNN — Connectome Recovery (noise=0.5)

## Goal

Optimize GNN hyperparameters for maximum **connectivity matrix recovery (conn_R2)** on FlyVis with
high noise level σ=0.5. Two sub-goals:

- Maximize conn_R2 (primary): recover synaptic weight matrix W under 10× higher noise than noise_005
- Minimize catastrophic failures: achieve CV < 5%, ≤1 catastrophic/8 seeds, mean conn_R2 > 0.95

The exploration starts from the **default config** below to enable before/after comparison with
the prior winner (conn_R2=0.987, CV=1.3%, 0/4 catastrophic).

Active case study: **CV_robustness** — target CV<5%, mean conn_R2>0.95, ≤1 catastrophic/8 seeds.

## Scientific Context

**Core research question: How does 10× higher noise (σ=0.5 vs σ=0.05) change the optimization
landscape for connectome recovery from Drosophila optic lobe activity?**

At σ=0.5, noise masks the neural signal, making W recovery fundamentally harder. Key challenges:
- Catastrophic failures (conn_R2 < 0.5) appear randomly across seeds — root cause unknown
- w_init_scale=0.25 (smaller W initialization) was found to reduce catastrophic rate — explore why
- Higher f_theta regularization (L1=0.5 vs 0.05 at noise_005) may prevent over-fitting to noise
- The optimization landscape is wider but has deep failure basins — understand which HPs prevent falls
- Compare landscape width vs noise_005: does high noise regularize or destabilize training?

## Noise Model

```
v_i(t+1) = v_i(t) + dt * f(v_i(t), W, a_i, I_i(t)) + epsilon_i(t)
epsilon_i ~ N(0, sigma)  where sigma = 0.5 (noise_model_level)
```

**Important**: Noise is added to **training data only**. Test rollouts use noise-free data. Do not
compare training and test metrics directly — training has an irreducible noise floor.

## Metrics

**Always use metrics defined to guide decision making**

During training (stdout):
```
epoch 0/1 | train: ... | conn_R2=0.XXX tau_R2=0.XXX Vr_R2=0.XXX | duration: XXs
```

During test/validation:
- **PRIMARY METRIC: `conn_R2`** (higher is better; R² of learned W vs ground-truth W)
- `tau_R2`: R² of τ (time constant) recovery
- `V_rest_R2`: R² of V_rest (resting potential) recovery — at σ=0.5 this may be partially recoverable
- `cluster_accuracy`: cell-type clustering accuracy from neuron embeddings
- `rollout_pearson_r`: Pearson r of autoregressive rollout vs ground truth

**Robustness classification** (4 seeds per iteration):
- **Stable-Robust**: all 4 seeds conn_R2 ≥ 0.90, CV < 5% (relaxed threshold vs noise_005 due to high noise)
- **Stable**: mean conn_R2 ≥ 0.85, CV < 10%
- **Unstable**: mean < 0.85 OR CV ≥ 10%
- **Catastrophic**: any seed conn_R2 < 0.50

**Extended robustness (Block 8)**: 8-seed CV_robustness validation. Target: CV < 5%, mean > 0.95,
≤1 catastrophic/8 seeds.

**Note on τ_R2 and V_rest_R2**: Model `flyvis_A` absorbs τ and V_rest implicitly into f_theta.
These metrics show 0.00 or N/A — this is expected behavior, not a failure.

Data is **NOT re-generated** each iteration (`generate_data: false`).

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction
2. **Design experiment**: Change **EXACTLY ONE** parameter at a time (causality rule)
3. **Run training**: 4 slots (1 control + 3 experiments in EXPLORATION; 4 same config in ROBUSTNESS)
4. **Analyze results**: Use conn_R2 AND catastrophic rate to understand stability
5. **Update understanding**: Revise hypotheses based on evidence

**CRITICAL**: You can only hypothesize. Only training results validate or falsify.

### CAUSALITY RULE (MANDATORY)

**If you change more than one parameter per slot, you CANNOT attribute the effect. Fatal
experimental design error.**

- In EXPLORATION mode: Slot 0 = parent/baseline (unchanged control). Slots 1-3 each change
  **exactly one** parameter from the parent.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness).

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * g(v_j) + I_i(t)
```

- **13,741 neurons**, 65 cell types, **434,112 edges**
- **1,736 input neurons** (photoreceptors, DAVIS visual input)
- Noise level: σ=0.5 per time step (10× higher than noise_005)
- 64,000 frames, delta_t = 0.02
- Model `flyvis_A`: f_theta absorbs τ and V_rest implicitly (τ_R2=0, Vr_R2=0 is expected)

## GNN Architecture

```
g_phi(v_j, embed_j) → message_ij          (edge MLP, per-edge messages)
sum_j W_ij * g_phi(v_j) → agg_i           (weighted aggregation)
f_theta(v_i, agg_i, embed_i) → dv_i/dt   (node update MLP)
```

- Per-neuron embedding: learnable `embedding_dim`-dimensional vector (concatenated to inputs)
- `g_phi_positive=true`: g_phi output clipped to [0, ∞) (Dale's law approximation)

**YOU ARE ONLY ALLOWED TO MODIFY THE PARAMETERS BELOW TO ACHIEVE THE GOAL**

## GNN Architecture Parameters

| Parameter       | Default | Description                                             |
| --------------- | ------- | ------------------------------------------------------- |
| `hidden_dim`    | 80      | Width of hidden layers in g_phi and f_theta             |
| `n_layers`      | 3       | Depth of g_phi and f_theta networks                     |
| `embedding_dim` | 2       | Per-neuron learnable embedding dimension                |

## Training Parameters

| Parameter                 | Default  | Description                                                                      |
| ------------------------- | -------- | -------------------------------------------------------------------------------- |
| `lr_W`                    | 0.0006   | Learning rate for W matrix (synaptic weights)                                    |
| `lr`                      | 0.0012   | Learning rate for g_phi and f_theta MLP weights                                  |
| `lr_embedding`            | 0.00155  | Learning rate for per-neuron embeddings                                          |
| `data_augmentation_loop`  | 20       | Augmentation loops per epoch — controls training time (DAL)                      |
| `batch_size`              | 2        | Samples per batch (smaller than noise_005 due to higher per-sample cost)         |
| `coeff_g_phi_diff`        | 750      | L2 penalty driving g_phi toward non-trivial activations (**critical!**)          |
| `coeff_g_phi_norm`        | 0.9      | L2 norm regularization on g_phi output values                                    |
| `coeff_g_phi_weight_L1`   | 0.28     | L1 weight regularization on g_phi network                                        |
| `coeff_f_theta_weight_L1` | 0.5      | L1 weight regularization on f_theta network (higher than noise_005 to resist noise) |
| `coeff_f_theta_weight_L2` | 0.001    | L2 weight regularization on f_theta network                                      |
| `coeff_W_L1`              | 7.5e-5   | L1 regularization on W (lower than noise_005 — noise itself acts as regularizer) |
| `coeff_W_L2`              | 1.5e-6   | L2 regularization on W                                                           |
| `regul_annealing_rate`    | 0.0      | Regularization annealing: **MUST be 0.0 with n_epochs=1** (otherwise all L1/L2 = 0) |
| `w_init_mode`             | randn_scaled | W initialization: `randn_scaled`, `zeros`, `uniform_scaled`                |
| `w_init_scale`            | 1.0      | Scale for randn_scaled W initialization (default; prior winner used 0.25)        |

**Training time budget**: Target ~60 min per run. Adjust DAL to stay within budget. Check
`training_time_min` in results after each iteration.

**Hard runtime limit (120 min)**: Cluster enforces 120-min wall-clock limit. Check for
`_interrupted` in slot log directories. If interrupted, reduce DAL for next iteration.

**Fixed: n_epochs=1** — do not change. With n_epochs=1, `regul_annealing_rate` MUST be 0.0
(annealing formula: effective_coeff = coeff × (1 − exp(−rate × epoch)) = 0 at epoch 0).

**Note**: Seeds are pipeline-controlled (`sim_seed = iter × 1000 + slot`,
`train_seed = iter × 1000 + slot + 500`). Do not set seeds in config files.

> **YAML rule**: Always wrap the `description` field value in double quotes — colons inside
> unquoted YAML strings cause parse errors.

## Data Generation

`generate_data: false` — data is pre-generated and NOT regenerated each iteration.

**DO NOT modify simulation parameters** (n_neurons, n_frames, n_edges, delta_t, noise_model_level).

## Block Structure

| Block | Focus                      | Parameters to scan                                               | Ranges                                                                           |
| ----- | -------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 1     | **Baseline robustness**    | All 4 slots = default config (ROBUSTNESS)                       | Establish pre-optimization baseline; measure catastrophic failure rate           |
| 2     | **W initialization**       | `w_init_scale`, `w_init_mode`                                   | scale: {0.1, 0.25, 0.5, 1.0}; mode: {randn_scaled, zeros, uniform_scaled}      |
| 3     | **Learning rates**         | `lr_W`, `lr`, `lr_embedding`                                    | lr_W: {3e-4, 6e-4, 9e-4, 1.2e-3}; lr: {6e-4, 1.2e-3, 2.4e-3}; lr_W/lr ratio  |
| 4     | **g_phi terms**            | `coeff_g_phi_diff`, `coeff_g_phi_norm`, `coeff_g_phi_weight_L1` | diff: {375, 750, 1500}; norm: {0, 0.5, 0.9, 1.5}; g_L1: {0, 0.1, 0.28, 0.5}  |
| 5     | **f_theta reg**            | `coeff_f_theta_weight_L1`, `coeff_f_theta_weight_L2`            | f_L1: {0.1, 0.3, 0.5, 1.0}; f_L2: {0, 0.001, 0.01}                            |
| 6     | **W reg + training**       | `coeff_W_L1`, `batch_size`, `data_augmentation_loop`            | W_L1: {0, 3e-5, 7.5e-5, 2e-4}; bs: {1, 2, 4}; DAL: {15, 20, 30}               |
| 7     | **Free exploration**       | Any parameter — combine best, target catastrophic failure rate  | Focus: reduce catastrophic rate; compare rescued vs persistent failure modes     |
| 8     | **Final robustness**       | Best config, all 4 slots same (ROBUSTNESS, `generate_data: false`) | Confirm CV < 5%, zero catastrophic; same data across seeds                 |
| 9     | **CV robustness (8 seeds)**| Best config, 8 seeds over 2 iterations (ROBUSTNESS, `generate_data: true`) | True seed independence; target CV < 5%, mean > 0.95, ≤1 catastrophic/8 |

**Extra blocks** (optional, use if Block 7 did not converge to a clear winner):
append additional EXPLORATION iterations on any block focus before proceeding to Blocks 8-9.

> **generate_data flag for CV robustness**: Before running Block 9, set `generate_data: true` in
> the config. This causes the pipeline to regenerate data with a fresh simulation seed for each
> slot, testing true independence across both training data and model initialization. After Block 9,
> reset `generate_data: false` to avoid unnecessary data regeneration.

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
- Seed-dependent effects (catastrophic in some seeds but not others)
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
2. Count catastrophic failures (conn_R2 < 0.5) — this is the key safety metric at σ=0.5
3. Check `training_time_min` — adjust DAL for next batch if > 70 min or < 50 min
4. Check for `_interrupted` in slot log directory (indicates job was killed by wall-clock limit)
5. Classify: Stable-Robust / Stable / Unstable / Catastrophic

### Step 3: Write Log Entry + Update Memory

```
## Iter N: [stable_robust/stable/unstable/catastrophic]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, DAL=D, bs=B,
        g_diff=A, g_norm=B, g_L1=C, f_L1=D, W_L1=E, W_L2=F,
        w_init=G, w_scale=H, emb_dim=I
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

## Block Boundaries

At every block boundary:

1. Update "Paper Summary" in memory
2. Summarize block findings (including catastrophic failure rate trend)
3. Update "Established Principles" and "Falsified Hypotheses"
4. Clear "Current Block"
5. Carry forward best config as parent for next block

## Start Call

When prompt says `PARALLEL START`:

- Slot 0 = **default config** (before-exploration baseline):
  `lr_W=0.0006, lr=0.0012, lr_embedding=0.00155, batch_size=2, DAL=20`
  `coeff_g_phi_diff=750, coeff_g_phi_norm=0.9, coeff_g_phi_weight_L1=0.28`
  `coeff_f_theta_weight_L1=0.5, coeff_W_L1=7.5e-5, w_init_mode=randn_scaled, w_init_scale=1.0, embedding_dim=2`
- Block 1 is ROBUSTNESS mode: Slots 1-3 also use the same default config (different seeds)
- Hypothesis: "The default GNN config at σ=0.5 achieves conn_R2 ≥ 0.80 but may show catastrophic
  failures. Prior winner: conn_R2=0.987, CV=1.3%, 0/4 catastrophic (used w_init_scale=0.25)."
- Launch: `python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_05 iterations=80 --cluster --resume`

---

## Final Summary

At exploration completion (after Block 8), append to
`/home/node/.claude/projects/-workspace--devcontainer/memory/exploration_results.md`:

### flyvis_noise_05 — Key Discoveries (YYYY-MM-DD)

1. **Best metric**: conn_R2 = X.XXX ± std (N seeds, CV=X.X%, catastrophic=N/8), winner config = [key params]
2. **Rescued**: Which noise_005 failure modes were rescued at σ=0.5 (cite iteration numbers)
3. **Not rescued**: Which failure modes survived even at high noise (a fundamental constraint)
4. **HP impact**: Which HP had the largest single-parameter impact on catastrophic failure rate
5. **Landscape**: How the optimization landscape width at σ=0.5 compares to σ=0.05
6. **W initialization**: Whether w_init_scale=0.25 consistently prevented catastrophic failure and why
7. **Minimal config**: What minimal configuration still achieved robust recovery (CV < 5%)
8. **Fundamental limit**: Any fundamental limit encountered (e.g., catastrophic failure rate floor)

---

# Working Memory Structure

```markdown
# Working Memory: {llm_task_name}

## Paper Summary (update at every block boundary)

**GNN optimization** (2 sentences on HPO findings):
Sentence 1: Best hyperparameter configuration found and the conn_R2 it achieves (cite mean ± std, CV%, catastrophic/8 seeds).
Sentence 2: Which hyperparameters were most critical for preventing catastrophic failures — what worked and what failed (cite values).

**LLM-driven exploration** (2 sentences on exploration findings):
Sentence 1: What the systematic exploration revealed about the high-noise optimization landscape (catastrophic basin structure, rescued vs persistent failure modes).
Sentence 2: Main causal principle established — what controls catastrophic failure rate at σ=0.5 and how this compares to the σ=0.05 landscape.

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean±std) | CV% | catastrophic | Verdict | Hypothesis |
| ---- | -------------- | ------------------- | --- | ------------ | ------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest→newest. This section MUST
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
