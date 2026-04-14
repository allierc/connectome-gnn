# FlyVis GNN — Connectome Recovery (noise-free)

## Goal

Optimize GNN hyperparameters for maximum **connectivity matrix recovery (conn_R2)** on FlyVis with
no noise (σ=0.0). Two sub-goals:

- Maximize conn_R2 (primary): recover synaptic weight matrix W from clean neural activity
- Understand the noise-free ceiling: why is conn_R2=0.923 (prior winner) lower than noise_005's 0.982?

The exploration starts from the **default config** below to enable before/after comparison with
the prior winner (conn_R2=0.923, CV=0.82%).

## Scientific Context

**Core research question: Why does noise-free training produce lower connectome recovery (0.923)
than σ=0.05 training (0.982)? Does noise act as regularization?**

The noise-free case is surprising — removing noise should make W recovery easier, yet the prior winner
is lower. Key hypotheses to test:
- Noise acts as implicit regularization (Langevin dynamics): σ=0.05 prevents over-fitting to W
- The embedding_dim=4 (unique to noise-free) may be under-exploited — extra dims could help or hurt
- All regularization is 0 in the default config — should some regularization be introduced?
- coeff_g_phi_diff=1500 is 2× higher than noisy variants — explore if this is optimal
- The absence of noise sharpens the optimization landscape, potentially creating narrow basins

Key differences from noisy variants:
- embedding_dim=4 (vs 2 for noise_005 and noise_05)
- All L1/L2 regularization = 0 (vs non-zero for noisy variants)
- coeff_g_phi_norm = 0 (vs 0.9 for noisy variants)
- coeff_g_phi_diff = 1500 (vs 750 for noisy variants)

## Noise Model

```
v_i(t+1) = v_i(t) + dt * f(v_i(t), W, a_i, I_i(t))
sigma = 0.0 (noise_model_level)
```

No noise is added to training data. Train and test conditions are identical.

## Metrics

**Always use metrics defined to guide decision making**

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

**Robustness classification** (4 seeds per iteration):
- **Stable-Robust**: all 4 seeds conn_R2 ≥ 0.90, CV < 3%
- **Stable**: mean conn_R2 ≥ 0.85, CV < 10%
- **Unstable**: mean < 0.85 OR CV ≥ 10%
- **Catastrophic**: any seed conn_R2 < 0.50

**Note on τ_R2 and V_rest_R2**: Model `flyvis_A` absorbs τ and V_rest implicitly into f_theta.
These metrics show 0.00 or N/A — this is expected behavior, not a failure.

Data is **NOT re-generated** each iteration (`generate_data: false`).

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction
2. **Design experiment**: Change **EXACTLY ONE** parameter at a time (causality rule)
3. **Run training**: 4 slots (1 control + 3 experiments in EXPLORATION; 4 same config in ROBUSTNESS)
4. **Analyze results**: Use conn_R2 AND cluster_accuracy to understand embedding quality
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
- Noise level: σ=0.0 (noise-free)
- 64,000 frames, delta_t = 0.02
- Model `flyvis_A`: f_theta absorbs τ and V_rest implicitly (τ_R2=0, Vr_R2=0 is expected)

## GNN Architecture

```
g_phi(v_j, embed_j) → message_ij          (edge MLP, per-edge messages)
sum_j W_ij * g_phi(v_j) → agg_i           (weighted aggregation)
f_theta(v_i, agg_i, embed_i) → dv_i/dt   (node update MLP)
```

- Per-neuron embedding: learnable `embedding_dim`-dimensional vector (concatenated to inputs)
- **embedding_dim=4** (unique to noise-free; 2× larger than noisy variants — 2 extra input channels)
- `g_phi_positive=true`: g_phi output clipped to [0, ∞) (Dale's law approximation)

**YOU ARE ONLY ALLOWED TO MODIFY THE PARAMETERS BELOW TO ACHIEVE THE GOAL**

## GNN Architecture Parameters

| Parameter       | Default | Description                                                              |
| --------------- | ------- | ------------------------------------------------------------------------ |
| `hidden_dim`    | 80      | Width of hidden layers in g_phi and f_theta                              |
| `n_layers`      | 3       | Depth of g_phi and f_theta networks                                      |
| `embedding_dim` | 4       | Per-neuron learnable embedding dimension (2× vs noisy variants)          |

## Training Parameters

| Parameter                 | Default  | Description                                                                      |
| ------------------------- | -------- | -------------------------------------------------------------------------------- |
| `lr_W`                    | 0.0006   | Learning rate for W matrix (synaptic weights)                                    |
| `lr`                      | 0.0018   | Learning rate for g_phi and f_theta MLP weights                                  |
| `lr_embedding`            | 0.00155  | Learning rate for per-neuron embeddings                                          |
| `data_augmentation_loop`  | 30       | Augmentation loops per epoch — controls training time (DAL)                      |
| `batch_size`              | 4        | Samples per batch                                                                |
| `coeff_g_phi_diff`        | 1500     | L2 penalty driving g_phi toward non-trivial activations (2× vs noisy variants)  |
| `coeff_g_phi_norm`        | 0.0      | L2 norm regularization on g_phi (0 for noise-free; contrast with 0.9 at σ=0.05) |
| `coeff_g_phi_weight_L1`   | 0.0      | L1 weight regularization on g_phi (0 for noise-free)                            |
| `coeff_f_theta_weight_L1` | 0.0      | L1 weight regularization on f_theta (0 for noise-free)                          |
| `coeff_f_theta_weight_L2` | 0.0      | L2 weight regularization on f_theta (0 for noise-free)                          |
| `coeff_W_L1`              | 0.0      | L1 regularization on W (0 for noise-free — clean signal doesn't need sparsity)  |
| `coeff_W_L2`              | 0.0      | L2 regularization on W (0 for noise-free)                                       |
| `regul_annealing_rate`    | 0.0      | Regularization annealing: **MUST be 0.0 with n_epochs=1** (otherwise all L1/L2 = 0) |
| `w_init_mode`             | randn_scaled | W initialization: `randn_scaled`, `zeros`, `uniform_scaled`                |
| `w_init_scale`            | 1.0          | Scale for randn_scaled/uniform_scaled init (bound = scale/sqrt(n_edges))   |

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

| Block | Focus                       | Parameters to scan                                               | Ranges                                                                           |
| ----- | --------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 1     | **Baseline robustness**     | All 4 slots = default config (ROBUSTNESS)                       | Establish pre-optimization baseline; compare to noise_005 baseline               |
| 2     | **Learning rates**          | `lr_W`, `lr`, `lr_embedding`                                    | lr_W: {3e-4, 6e-4, 9e-4, 1.5e-3}; lr: {9e-4, 1.8e-3, 3.6e-3}; lr_W/lr ratio  |
| 3     | **g_phi diff + norm**       | `coeff_g_phi_diff`, `coeff_g_phi_norm`                          | diff: {750, 1000, 1500, 2500}; norm: {0, 0.5, 0.9, 1.5}                         |
| 4     | **Regularization intro**    | `coeff_g_phi_weight_L1`, `coeff_f_theta_weight_L1`, `coeff_W_L1`| g_L1: {0, 0.1, 0.28}; f_L1: {0, 0.05, 0.2}; W_L1: {0, 5e-5, 1.5e-4}          |
| 5     | **Architecture**            | `embedding_dim`, `hidden_dim`                                    | emb: {2, 4, 6, 8}; hidden: {40, 80, 128}; test if emb=4 is truly optimal        |
| 6     | **Training regime**         | `batch_size`, `data_augmentation_loop`                           | bs: {2, 4, 8}; DAL: {20, 30, 40, 60}                                            |
| 7     | **Free exploration**        | Any parameter — combine best from Blocks 2-6                    | Test noise-free-specific interactions; target conn_R2 > 0.95                     |
| 8     | **Final robustness**        | Best config, all 4 slots same (ROBUSTNESS, `generate_data: false`) | Confirm CV < 3%, conn_R2 > prior winner 0.923; same data across seeds        |
| 9     | **CV robustness**           | Best config, 8 seeds over 2 iterations (ROBUSTNESS, `generate_data: true`) | True seed independence: data regenerated per slot; target CV < 3%  |

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
- Seed-dependent effects (works for some seeds but not others)
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
2. Compare conn_R2 to the noise-free baseline (0.923) and to noise_005 (0.982) — note the gap
3. Check `training_time_min` — adjust DAL for next batch if > 70 min or < 50 min
4. Check for `_interrupted` in slot log directory (indicates job was killed by wall-clock limit)
5. Classify: Stable-Robust / Stable / Unstable / Catastrophic

### Step 3: Write Log Entry + Update Memory

```
## Iter N: [stable_robust/stable/unstable/catastrophic]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, DAL=D, bs=B,
        g_diff=A, g_norm=B, g_L1=C, f_L1=D, W_L1=E,
        w_init=F, emb_dim=G, hidden=H
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
2. Summarize block findings (note gap vs noise_005 and whether it narrowed)
3. Update "Established Principles" and "Falsified Hypotheses"
4. Clear "Current Block"
5. Carry forward best config as parent for next block

## Start Call

When prompt says `PARALLEL START`:

- Slot 0 = **default config** (before-exploration baseline):
  `lr_W=0.0006, lr=0.0018, lr_embedding=0.00155, batch_size=4, DAL=30`
  `coeff_g_phi_diff=1500, coeff_g_phi_norm=0.0, coeff_g_phi_weight_L1=0.0`
  `coeff_f_theta_weight_L1=0.0, coeff_W_L1=0.0, w_init_mode=randn_scaled, embedding_dim=4`
- Block 1 is ROBUSTNESS mode: Slots 1-3 also use the same default config (different seeds)
- Hypothesis: "The default noise-free GNN config achieves conn_R2 ≥ 0.90 with CV < 5% but remains
  below noise_005's winner (0.982). Prior winner: conn_R2=0.923, CV=0.82%."
- Launch: `python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_free iterations=80 --cluster --resume`

---

## Final Summary

At exploration completion (after Block 8), append to
`/home/node/.claude/projects/-workspace--devcontainer/memory/exploration_results.md`:

### flyvis_noise_free — Key Discoveries (YYYY-MM-DD)

1. **Best metric**: conn_R2 = X.XXX ± std (N seeds, CV=X.X%), winner config = [key params]
2. **Noise gap**: Did the exploration close the gap vs noise_005 (0.982)? What was the final delta?
3. **HP impact**: Which HP had the largest single-parameter impact on conn_R2 in the noise-free case
4. **Noise as regularizer**: Whether evidence supports or falsifies the hypothesis that noise acts as regularization (compare noise-free optimal reg to noise_005 optimal reg)
5. **Embedding**: Whether embedding_dim=4 was better/worse/neutral vs emb_dim=2
6. **g_phi_diff**: Whether coeff_g_phi_diff=1500 was validated or should be different from the 750 used in noisy variants
7. **Training regime**: What training regime (DAL, batch_size) proved optimal and why
8. **Fundamental limit**: Any ceiling effect observed — maximum achievable conn_R2 in noise-free case

---

# Working Memory Structure

```markdown
# Working Memory: {llm_task_name}

## Paper Summary (update at every block boundary)

**GNN optimization** (2 sentences on HPO findings):
Sentence 1: Best hyperparameter configuration found and the conn_R2 it achieves (cite mean ± std, CV%, N seeds), vs prior winner 0.923.
Sentence 2: Which hyperparameters were most critical in the noise-free case — what worked and what failed (cite values and CV impact).

**LLM-driven exploration** (2 sentences on exploration findings):
Sentence 1: What the systematic exploration revealed about the noise-free optimization landscape vs noisy variants (basin width, regularization needs, embedding role).
Sentence 2: Main causal principle — whether noise acts as implicit regularization, and what this tells us about GNN training for connectome recovery under clean signal conditions.

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
