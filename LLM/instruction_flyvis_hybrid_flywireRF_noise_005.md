# FlyVis Hybrid FlyWire-RF GNN — Connectome Recovery (noise=0.05)

## Goal

Optimize GNN hyperparameters for maximum **connectivity matrix recovery (conn_R2)** on the
**hybrid FlyVis / FlyWire per-column-RF** connectome (328,092 edges) with
noise level σ=0.05. Two sub-goals:

- Maximize conn_R2 (primary): recover the synaptic weight matrix W from noisy neural activity
- Maximize robustness: CV < 3% across seeds, zero catastrophic failures

The exploration starts from the **current default config** below (previously tuned:
`lr_W=9e-4, lr=1.8e-3, lr_embedding=2.325e-3, DAL=100, coeff_W_L1=1.5e-4, coeff_W_L2=2.5e-6`)
so baseline Block 1 establishes the state of the art on this edge set.

## Scientific Context

**Core research question: Can a GNN recover the full synaptic weight matrix when the FlyVis
connectome is pruned to the edges consistent with per-column FlyWire receptive fields
(328,092 edges — ~76% of the original 434,112), trained on noisy Drosophila optic-lobe
activity (σ=0.05)?**

The hybrid `flyvis_hybrid_flywireRF` connectome drops FlyVis edges that are inconsistent with
FlyWire per-column receptive-field evidence, yielding a biologically more plausible but sparser
graph. The GNN inverts the resulting dynamics: given noisy activity traces, learn W such that
GNN dynamics reproduce observed activity. With ~25% of edges removed, the per-edge gradient
magnitude increases (fewer sinks for the same aggregate message), so regularization and LR
settings that worked on the full `flyvis_A` connectome may need re-tuning.

Key challenges to investigate:
- `coeff_g_phi_diff` prevents trivial g_phi collapse — its tuning is critical for W recovery
- DAL may have a cliff: above DAL=50 the model may over-fit to noise patterns
- `lr_W` vs `lr` ratio — W learns the structural prior, g_phi/f_theta learn dynamics
- Regularization on W (L1/L2) — the yaml ships with `coeff_W_L1=1.5e-4` which may need
  adjustment once edge count drops from 434K to 328K

## Noise Model

```
v_i(t+1) = v_i(t) + dt * f(v_i(t), W, a_i, I_i(t)) + epsilon_i(t)
epsilon_i ~ N(0, sigma)  where sigma = 0.05 (noise_model_level)
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
- **PRIMARY METRIC: `conn_R2`** (higher is better; R² of learned W vs ground-truth W on the
  326,092 retained edges)
- `tau_R2`: R² of τ (time constant) recovery
- `V_rest_R2`: R² of V_rest (resting potential) recovery
- `cluster_accuracy`: cell-type clustering accuracy from neuron embeddings
- `rollout_pearson_r`: Pearson r of autoregressive rollout vs ground truth

**Robustness classification** (4 seeds per iteration):
- **Stable-Robust**: all 4 seeds conn_R2 ≥ 0.90, CV < 3%
- **Stable**: mean conn_R2 ≥ 0.85, CV < 10%
- **Unstable**: mean < 0.85 OR CV ≥ 10%
- **Catastrophic**: any seed conn_R2 < 0.50

**Note on τ_R2 and V_rest_R2**: Model `flyvis_hybrid_flywireRF` (like `flyvis_A`) absorbs τ and
V_rest implicitly into f_theta. These metrics show 0.00 or N/A — this is expected behavior,
not a failure.

Data is **NOT re-generated** each iteration (`generate_data: false`).

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction
2. **Design experiment**: Change **EXACTLY ONE** parameter at a time (causality rule)
3. **Run training**: 4 slots (1 control + 3 experiments in EXPLORATION; 4 same config in ROBUSTNESS)
4. **Analyze results**: Use conn_R2 AND rollout_pearson_r to understand convergence
5. **Update understanding**: Revise hypotheses based on evidence

**CRITICAL**: You can only hypothesize. Only training results validate or falsify.

### CAUSALITY RULE (MANDATORY)

**If you change more than one parameter per slot, you CANNOT attribute the effect. Fatal
experimental design error.**

- In EXPLORATION mode: Slot 0 = parent/baseline (unchanged control). Slots 1-3 each change
  **exactly one** parameter from the parent.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness).

## FlyVis Hybrid FlyWire-RF Model

Non-spiking compartment model of the Drosophila optic lobe on a FlyWire-RF-pruned connectome:

```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * g(v_j) + I_i(t)
```

- **13,741 neurons**, 65 cell types, **328,092 edges** (FlyWire per-column-RF pruned)
- **1,736 input neurons** (photoreceptors, DAVIS visual input)
- Noise level: σ=0.05 per time step
- 64,000 frames, delta_t = 0.02
- Model `flyvis_hybrid_flywireRF`: f_theta absorbs τ and V_rest implicitly
  (τ_R2=0, Vr_R2=0 is expected)

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

| Parameter                 | Default   | Description                                                                 |
| ------------------------- | --------- | --------------------------------------------------------------------------- |
| `lr_W`                    | 9e-4      | Learning rate for W matrix (synaptic weights)                               |
| `lr`                      | 1.8e-3    | Learning rate for g_phi and f_theta MLP weights                             |
| `lr_embedding`            | 2.325e-3  | Learning rate for per-neuron embeddings                                     |
| `data_augmentation_loop`  | 35        | Augmentation loops per epoch — controls training time (DAL)                 |
| `batch_size`              | 4         | Samples per batch                                                           |
| `coeff_g_phi_diff`        | 750       | L2 penalty driving g_phi toward non-trivial activations (**critical!**)     |
| `coeff_g_phi_norm`        | 0.9       | L2 norm regularization on g_phi output values                               |
| `coeff_f_theta_msg_diff`  | 0         | Monotonicity of f_theta w.r.t. aggregated message                           |
| `coeff_g_phi_weight_L1`   | 0.28      | L1 weight regularization on g_phi network                                   |
| `coeff_f_theta_weight_L1` | 0.05      | L1 weight regularization on f_theta network                                 |
| `coeff_g_phi_weight_L2`   | 0         | L2 weight regularization on g_phi network                                   |
| `coeff_f_theta_weight_L2` | 0.001     | L2 weight regularization on f_theta network                                 |
| `coeff_W_L1`              | 1.5e-4    | L1 regularization on W (promotes sparse connectome recovery)                |
| `coeff_W_L2`              | 2.5e-6    | L2 regularization on W                                                      |
| `regul_annealing_rate`    | 0.0       | Regularization annealing: **MUST be 0.0 with n_epochs=1** (otherwise all L1/L2 = 0) |
| `w_init_mode`             | randn_scaled | W initialization: `randn_scaled`, `zeros`, `uniform_scaled`              |
| `w_init_scale`            | 1.0          | Scale for randn_scaled/uniform_scaled init (bound = scale/sqrt(n_edges))  |

### Known considerations at 328K edges

At ~76% of the original FlyVis edge count, the per-edge gradient magnitude is ≈1.3× that of the
full `flyvis_A` connectome (same aggregated message distributed over fewer W entries). This means:

- L1 penalties scaled for 434K edges may now be mildly over-strong; try reducing `coeff_W_L1`
  toward 5e-5–1e-4 in Block 5 if conn_R2 plateaus
- `coeff_g_phi_weight_L1 ≥ 0.1` can still dominate the connectivity gradient — monitor for
  W→0 collapse and reduce if conn_R2 < 0.5

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

**DO NOT modify simulation parameters** (n_neurons, n_frames, n_edges, delta_t,
noise_model_level, connectivity_file, signal_model_name).

## Block Structure

| Block | Focus                   | Parameters to scan                                               | Ranges                                                                          |
| ----- | ----------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1     | **Baseline robustness** | All 4 slots = default config (ROBUSTNESS)                        | Establish baseline conn_R2 before optimization; measure seed variance on 328K edges |
| 2     | **Learning rates**      | `lr_W`, `lr`, `lr_embedding`                                     | lr_W: {4.5e-4, 9e-4, 1.4e-3, 1.8e-3}; lr: {9e-4, 1.8e-3, 3.6e-3}; lr_W/lr ratio |
| 3     | **g_phi terms**         | `coeff_g_phi_diff`, `coeff_g_phi_norm`, `coeff_g_phi_weight_L1`  | diff: {375, 750, 1500}; norm: {0, 0.5, 0.9, 1.5}; g_L1: {0, 0.1, 0.28, 0.5}     |
| 4     | **f_theta reg**         | `coeff_f_theta_weight_L1`, `coeff_f_theta_weight_L2`             | f_L1: {0, 0.01, 0.05, 0.2}; f_L2: {0, 0.001, 0.01}                              |
| 5     | **W reg + init**        | `coeff_W_L1`, `coeff_W_L2`, `w_init_mode`                        | W_L1: {0, 5e-5, 1e-4, 1.5e-4, 5e-4}; W_L2: {0, 2.5e-6, 1e-5}; init: {randn_scaled, zeros, uniform_scaled} |
| 6     | **Training regime**     | `batch_size`, `data_augmentation_loop`                           | bs: {2, 4, 8}; DAL: {20, 35, 50, 75}; test whether DAL cliff shifts at 328K edges |
| 7     | **Free exploration**    | Any parameter — combine best from Blocks 2-6                     | Novel combinations; test surprising interactions                                |
| 8     | **Final robustness**    | Best config, all 4 slots same (ROBUSTNESS, `generate_data: false`) | Confirm CV < 3%, zero catastrophic failures; same data across seeds           |
| 9     | **CV robustness**       | Best config, 8 seeds over 2 iterations (ROBUSTNESS, `generate_data: true`) | True seed independence: data regenerated per slot; target CV < 3%          |

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
2. Check `training_time_min` — adjust DAL for next batch if > 70 min or < 50 min
3. Check for `_interrupted` in slot log directory (indicates job was killed by wall-clock limit)
4. Classify: Stable-Robust / Stable / Unstable / Catastrophic

### Step 3: Write Log Entry + Update Memory

```
## Iter N: [stable_robust/stable/unstable/catastrophic]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, DAL=D, bs=B,
        g_diff=A, g_norm=B, g_L1=C, f_L1=D, W_L1=E, W_L2=F,
        w_init=G, emb_dim=H
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
2. Summarize block findings
3. Update "Established Principles" and "Falsified Hypotheses"
4. Clear "Current Block"
5. Carry forward best config as parent for next block

## Start Call

When prompt says `PARALLEL START`:

- Slot 0 = **default config** (current yaml, before-exploration baseline):
  `lr_W=9e-4, lr=1.8e-3, lr_embedding=2.325e-3, batch_size=4, DAL=35`
  `coeff_g_phi_diff=750, coeff_g_phi_norm=0.9, coeff_g_phi_weight_L1=0.28`
  `coeff_f_theta_weight_L1=0.05, coeff_f_theta_weight_L2=0.001`
  `coeff_W_L1=1.5e-4, coeff_W_L2=2.5e-6, w_init_mode=randn_scaled, embedding_dim=2`
- Block 1 is ROBUSTNESS mode: Slots 1-3 also use the same default config (different seeds)
- Hypothesis: "The current yaml config achieves conn_R2 ≥ 0.90 with CV < 10% across 4 seeds
  on the 328K-edge FlyWire-RF hybrid connectome (pre-optimization baseline)."
- Launch:
  `python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_hybrid_flywireRF_noise_005 iterations=80 --cluster --resume`

---

## Final Summary

At exploration completion (after Block 8), append to
`/home/node/.claude/projects/-workspace--devcontainer/memory/exploration_results.md`:

### flyvis_hybrid_flywireRF_noise_005 — Key Discoveries (YYYY-MM-DD)

1. **Best metric**: conn_R2 = X.XXX ± std (N seeds, CV=X.X%), winner config = [key params]
2. **HP impact**: Which HP had the largest single-parameter impact, and its optimal value
3. **Failure mode**: Which failure mode was confirmed across 3+ iterations (cite iteration numbers)
4. **Surprise**: Which HP interaction produced an unexpected or surprising result
5. **Falsified hypothesis**: Which hypothesis was falsified and what was learned from it
6. **Regularization**: Whether regularization helped/hurt and under what conditions at 328K edges
7. **Training regime**: What training regime (DAL, batch_size) proved optimal and why
8. **Fundamental limit**: Any fundamental limit encountered (e.g., metric plateau despite HP variation)
9. **Comparison to flyvis_A**: How conn_R2 and CV compare vs the full 434K-edge `flyvis_A`
   baseline (`flyvis_noise_005`) — did edge reduction help or hurt recovery?

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_hybrid_flywireRF_noise_005

## Paper Summary (update at every block boundary)

**GNN optimization** (2 sentences on HPO findings):
Sentence 1: Best hyperparameter configuration found and the conn_R2 it achieves (cite mean ± std, CV%, N seeds).
Sentence 2: Which hyperparameters were most critical to stability at 328K edges — what worked and what failed (cite values and CV impact).

**LLM-driven exploration** (2 sentences on exploration findings):
Sentence 1: What the systematic exploration revealed about the optimization landscape on the FlyWire-RF-pruned connectome (basin width, failure modes, critical regularization interactions).
Sentence 2: Main causal principle established from hypothesis testing — what this tells us about GNN training for connectome recovery when FlyVis edges are pruned by FlyWire per-column receptive fields under low noise (σ=0.05).

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean±std) | CV% | catastrophic | Verdict | Hypothesis |
| ---- | -------------- | ------------------- | --- | ------------ | ------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

- Do the `flyvis_noise_005` best HPs (lr_W=9e-4, lr=1.8e-3, lr_emb=2.325e-3) still rank first
  on the 328K-edge FlyWire-RF connectome, or does lower edge count favor a different optimum?
- Does the W_L1 penalty (1.5e-4) need to shrink proportional to the edge count reduction?
- Is there a DAL cliff analogous to the flyvis_noise_005 DAL=35 boundary on this edge set?

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
