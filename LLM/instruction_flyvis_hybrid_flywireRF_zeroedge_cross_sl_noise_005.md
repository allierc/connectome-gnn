# FlyVis Hybrid FlyWire-RF + Zero-Edge (cross-type) GNN — Connectome Recovery (noise=0.05)

## Goal

Optimize GNN hyperparameters for maximum **connectivity matrix recovery (conn_R2)** on the
**hybrid FlyVis / FlyWire per-column-RF connectome augmented with cross-type zero-weight
edges** (1,959,994 edges) with noise level σ=0.05. Two sub-goals:

- Maximize conn_R2 (primary): recover the synaptic weight matrix W from noisy neural
  activity, including correctly inferring **near-zero weights** for the ~1.6M added
  cross-type "uncertain" edges
- Maximize robustness: CV < 3% across seeds, **no late-training W_R² collapse**

The exploration starts from the **current default config** (inherited from
`flyvis_hybrid_flywireRF_noise_005_winner.yaml`:
`lr_W=9e-4, lr=1.8e-3, lr_embedding=2.325e-3, DAL=35, coeff_W_L1=1.5e-4, coeff_W_L2=2.5e-6`).
Block 1 establishes the baseline on the 1.96M-edge augmented graph.

## Scientific Context

**Core research question: Can a GNN recover the true synaptic weight matrix when ~83% of
the input edges are spurious (zero-weight, cross-type, spatially-local connectivity
hypotheses), trained on noisy Drosophila optic-lobe activity (σ=0.05)?**

The cross-type zero-edge augmentation simulates **connectome uncertainty** at the
inference stage: 328,092 oracle edges from `flyvis_hybrid_flywireRF` are mixed with
1,631,902 spurious cross-type edges that share the same spatial locality but cross cell
types. The GNN must learn to drive the spurious edges' W toward 0 while preserving the
true edge weights.

**Known failure mode at this edge density**: the prior unoptimized run shows
**catastrophic late-training collapse** of conn_R2:

- iter 11,201: peak conn_R2 ≈ 0.864
- iter 100,801: conn_R2 = 0.789 (declining)

This is **not under-training** — it's overfitting / loss-tradeoff drift. The exploration
must find a config that maintains conn_R2 ≥ 0.85 without the late peak-and-drop pattern.

Key hypotheses to investigate:

- **Higher coeff_W_L1**: with 6× more edges and ~83% spurious, the L1 prior for sparsity
  should be substantially stronger than the 1.5e-4 inherited from the oracle case.
  Sweep range up to 1e-2.
- **Lower lr_W or LR-decay**: the early peak (iter 11k) suggests the W signal is being
  overwritten as the optimizer continues. Reducing lr_W or introducing decay may stabilize.
- **DAL ceiling**: at 1.96M edges, each iteration is ~6× more expensive per edge; DAL
  may need to drop further to fit wall-clock.
- **g_phi terms re-tune**: with ~6× the message-passing volume, the per-edge gradient
  through g_phi changes, so `coeff_g_phi_diff` and `coeff_g_phi_norm` may need re-tuning.

## Noise Model

```
v_i(t+1) = v_i(t) + dt * f(v_i(t), W, a_i, I_i(t)) + epsilon_i(t)
epsilon_i ~ N(0, sigma)  where sigma = 0.05 (noise_model_level)
```

**Important**: Noise is added to **training data only**. Test rollouts use noise-free data.

## Metrics

**Always use metrics defined to guide decision making**

During training (stdout):
```
epoch 0/1 | train: ... | conn_R2=0.XXX tau_R2=0.XXX Vr_R2=0.XXX | duration: XXs
```

During test/validation:
- **PRIMARY METRIC: `conn_R2`** (higher is better; R² of learned W vs ground-truth W
  computed over **all 1,959,994 edges** including the 1.6M zero-weight cross-type edges,
  so a high score requires both correct true-edge magnitude AND zero-shrinkage of the
  spurious ones)
- `tau_R2`: R² of τ recovery
- `V_rest_R2`: R² of V_rest recovery
- `cluster_accuracy`: cell-type clustering accuracy
- `rollout_pearson_r`: Pearson r of autoregressive rollout vs ground truth

**Robustness classification** (4 seeds per iteration):
- **Stable-Robust**: all 4 seeds conn_R2 ≥ 0.90, CV < 3%, **no late-training drop ≥ 0.05
  in the final 50% of iterations**
- **Stable**: mean conn_R2 ≥ 0.85, CV < 10%, no catastrophic seed
- **Unstable**: mean < 0.85 OR CV ≥ 10%
- **Catastrophic**: any seed conn_R2 < 0.50, OR late-training drop ≥ 0.10

**Note on τ_R2 and V_rest_R2**: Model `flyvis_hybrid_flywireRF_zeroedge_cross_sl` (like
`flyvis_A`) absorbs τ and V_rest implicitly into f_theta. These metrics show 0.00 — this
is expected behavior, not a failure.

Data is **NOT re-generated** each iteration (`generate_data: false`).

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction
2. **Design experiment**: Change **EXACTLY ONE** parameter at a time (causality rule)
3. **Run training**: 4 slots (1 control + 3 experiments in EXPLORATION; 4 same config
   in ROBUSTNESS)
4. **Analyze results**: Use conn_R2 trajectory (not just final) — flag any peak-drop
   pattern in metrics.log
5. **Update understanding**: Revise hypotheses based on evidence

**CRITICAL**: You can only hypothesize. Only training results validate or falsify.

### CAUSALITY RULE (MANDATORY)

**If you change more than one parameter per slot, you CANNOT attribute the effect.**

- EXPLORATION mode: Slot 0 = parent/baseline (unchanged control). Slots 1-3 each change
  **exactly one** parameter from the parent.
- ROBUSTNESS mode: all 4 slots use the same config (different seeds).

### TRAJECTORY-AWARE ANALYSIS (this exploration only)

Because of the documented late-collapse failure mode, after each iteration also extract
from `tmp_training/metrics.log` the **iter-of-peak-conn_R2** and the **delta from peak
to final**. Treat any slot with `final_conn_R2 / peak_conn_R2 < 0.95` as DISQUALIFIED
even if final conn_R2 looks acceptable.

## FlyVis Hybrid FlyWire-RF Cross-SL Model

Non-spiking compartment model on a 1.96M-edge augmented graph:

```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * g(v_j) + I_i(t)
```

- **13,741 neurons**, 65 cell types, **1,959,994 edges** (FlyWire-RF oracle 328K +
  ~1.63M cross-type zero-weight uncertain edges, spatially-local sampling)
- **1,736 input neurons** (photoreceptors, DAVIS visual input)
- Noise level: σ=0.05 per time step
- 64,000 frames, delta_t = 0.02
- Model `flyvis_hybrid_flywireRF_zeroedge_cross_sl`: f_theta absorbs τ and V_rest
  implicitly (τ_R2=0, Vr_R2=0 is expected)

## GNN Architecture

```
g_phi(v_j, embed_j) → message_ij          (edge MLP, per-edge messages)
sum_j W_ij * g_phi(v_j) → agg_i           (weighted aggregation)
f_theta(v_i, agg_i, embed_i) → dv_i/dt   (node update MLP)
```

- Per-neuron embedding: learnable `embedding_dim`-dimensional vector
- `g_phi_positive=true`: g_phi output clipped to [0, ∞) (Dale's law approximation)

**YOU ARE ONLY ALLOWED TO MODIFY THE PARAMETERS BELOW.**

## GNN Architecture Parameters

| Parameter       | Default | Description                                             |
| --------------- | ------- | ------------------------------------------------------- |
| `hidden_dim`    | 80      | Width of hidden layers in g_phi and f_theta             |
| `n_layers`      | 3       | Depth of g_phi and f_theta networks                     |
| `embedding_dim` | 2       | Per-neuron learnable embedding dimension                |

## Training Parameters

| Parameter                 | Default   | Description                                                                 |
| ------------------------- | --------- | --------------------------------------------------------------------------- |
| `lr_W`                    | 9e-4      | Learning rate for W matrix                                                  |
| `lr`                      | 1.8e-3    | Learning rate for g_phi and f_theta MLP weights                             |
| `lr_embedding`            | 2.325e-3  | Learning rate for per-neuron embeddings                                     |
| `data_augmentation_loop`  | 35        | Augmentation loops per epoch (DAL); may need lower at 1.96M edges           |
| `batch_size`              | 4         | Samples per batch                                                           |
| `coeff_g_phi_diff`        | 750       | L2 penalty driving g_phi toward non-trivial activations                     |
| `coeff_g_phi_norm`        | 0.9       | L2 norm regularization on g_phi output values                               |
| `coeff_f_theta_msg_diff`  | 0         | Monotonicity of f_theta w.r.t. aggregated message                           |
| `coeff_g_phi_weight_L1`   | 0.28      | L1 weight regularization on g_phi network                                   |
| `coeff_f_theta_weight_L1` | 0.05      | L1 weight regularization on f_theta network                                 |
| `coeff_g_phi_weight_L2`   | 0         | L2 weight regularization on g_phi network                                   |
| `coeff_f_theta_weight_L2` | 0.001     | L2 weight regularization on f_theta network                                 |
| `coeff_W_L1`              | 1.5e-4    | L1 regularization on W (**critical** at 1.96M edges with 83% zero-weight)   |
| `coeff_W_L2`              | 2.5e-6    | L2 regularization on W                                                      |
| `regul_annealing_rate`    | 0.0       | Regularization annealing: **MUST be 0.0 with n_epochs=1**                   |
| `w_init_mode`             | randn_scaled | W init: `randn_scaled`, `zeros`, `uniform_scaled`                        |
| `w_init_scale`            | 1.0          | Scale for randn_scaled/uniform_scaled init                               |

### Known considerations at 1.96M edges (cross-type augmentation)

At ~6× the oracle edge count and ~83% spurious zero-weight edges, the optimization
landscape changes substantially:

- **`coeff_W_L1` likely under-strength** at 1.5e-4. With ~1.63M edges that should
  converge to W=0, the L1 prior must dominate. Sweep up through 1e-3, 5e-3, 1e-2 in
  Block 5.
- **Per-edge gradient is ≈1/6 of the oracle case** (same aggregate message distributed
  over 6× more W entries), so a higher lr_W may help — try 1.4e-3, 1.8e-3 in Block 2.
- **g_phi L1 (0.28)**: with 6× messages the cumulative g_phi penalty is correspondingly
  larger. Consider reducing toward 0.1 if W_R² collapses.

**Training time budget**: Target ~60 min per run. With 6× edges, DAL may need to drop
to 15-20 to stay within budget. Check `training_time_min` after each iteration.

**Hard runtime limit (120 min)**: cluster enforces 120-min wall-clock limit.

**Fixed: n_epochs=1** — do not change. With n_epochs=1, `regul_annealing_rate` MUST be 0.0.

**Note**: Seeds are pipeline-controlled. Do not set seeds in config files.

> **YAML rule**: Always wrap `description` field in double quotes.

## Data Generation

`generate_data: false` — data is pre-generated and NOT regenerated each iteration.

**DO NOT modify simulation parameters** (n_neurons, n_frames, n_edges, delta_t,
noise_model_level, connectivity_file, signal_model_name).

## Block Structure

| Block | Focus                   | Parameters to scan                                               | Ranges                                                                          |
| ----- | ----------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1     | **Baseline robustness** | All 4 slots = default config (ROBUSTNESS)                        | Establish baseline conn_R2 + collapse pattern on 1.96M edges                    |
| 2     | **Learning rates**      | `lr_W`, `lr`, `lr_embedding`                                     | lr_W: {4.5e-4, 9e-4, 1.4e-3, 2.7e-3}; lr: {9e-4, 1.8e-3, 3.6e-3}                |
| 3     | **W L1 sweep** (HIGH PRIORITY) | `coeff_W_L1`                                              | {1.5e-4, 5e-4, 1e-3, 5e-3, 1e-2} — directly attacks zero-weight recovery         |
| 4     | **g_phi terms**         | `coeff_g_phi_diff`, `coeff_g_phi_norm`, `coeff_g_phi_weight_L1`  | diff: {375, 750, 1500}; norm: {0.5, 0.9, 1.5}; g_L1: {0, 0.1, 0.28, 0.5}        |
| 5     | **W reg + init**        | `coeff_W_L2`, `w_init_mode`                                      | W_L2: {0, 2.5e-6, 1e-5, 1e-4}; init: {randn_scaled, zeros, uniform_scaled}      |
| 6     | **Training regime**     | `batch_size`, `data_augmentation_loop`                           | bs: {2, 4, 8}; DAL: {15, 20, 35}; test late-collapse vs DAL                     |
| 7     | **Free exploration**    | Combine best from Blocks 2-6                                     | Address peak-drop pattern; novel combinations                                    |
| 8     | **Final robustness**    | Best config, all 4 slots same (ROBUSTNESS, `generate_data: false`) | Confirm CV < 3%, no late collapse                                              |
| 9     | **CV robustness**       | Best config, 8 seeds over 2 iterations (`generate_data: true`)   | True seed independence; target CV < 3%                                          |

**Extra blocks** (optional, if Block 7 did not converge): append additional EXPLORATION
iterations on any block focus.

> **generate_data flag for CV robustness**: Before Block 9, set `generate_data: true`
> in the config. Reset to `false` after.

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
2. Consistent across **all 4 seeds**
3. States a **causal relationship**

### What to Add to Open Questions

- Patterns observed 1-2 times
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

1. Read `conn_R2`, `tau_R2`, `V_rest_R2`, `cluster_accuracy`, `rollout_pearson_r`
2. **Read trajectory from `tmp_training/metrics.log`**: peak iter, peak conn_R2, final
   conn_R2, final/peak ratio
3. Check `training_time_min` — adjust DAL if > 70 min or < 50 min
4. Check for `_interrupted` (job killed by wall-clock limit)
5. Classify: Stable-Robust / Stable / Unstable / Catastrophic (DROP-AWARE)

### Step 3: Write Log Entry + Update Memory

```
## Iter N: [stable_robust/stable/unstable/catastrophic]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, DAL=D, bs=B,
        g_diff=A, g_norm=B, g_L1=C, f_L1=D, W_L1=E, W_L2=F,
        w_init=G, emb_dim=H
Slot 0: conn_R2_final=X, peak=Y@iterZ, drop=W%, sim_seed=S, train_seed=T
Slot 1: ...
Slot 2: ...
Slot 3: ...
Seed stats: mean_conn_R2=X, std=Y, CV=Z%, max_drop=W%
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive]
Next: parent=P
```

### Step 4: Acknowledge User Input

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Block Boundaries

At every block boundary:

1. Update "Paper Summary" in memory
2. Summarize block findings (incl. trajectory pattern)
3. Update "Established Principles" and "Falsified Hypotheses"
4. Clear "Current Block"
5. Carry forward best config as parent for next block

## Start Call

When prompt says `PARALLEL START`:

- Slot 0 = **default config** (current yaml, before-exploration baseline):
  `lr_W=9e-4, lr=1.8e-3, lr_embedding=2.325e-3, batch_size=4, DAL=35`
  `coeff_g_phi_diff=750, coeff_g_phi_norm=0.9, coeff_g_phi_weight_L1=0.28`
  `coeff_f_theta_weight_L1=0.05, coeff_f_theta_weight_L2=0.001`
  `coeff_W_L1=1.5e-4, coeff_W_L2=1.5e-6, w_init_mode=randn_scaled, embedding_dim=2`
- Block 1 is ROBUSTNESS mode: Slots 1-3 also use the default config (different seeds)
- Hypothesis: "The current yaml config achieves conn_R2 ≥ 0.85 with CV < 10% at peak,
  but exhibits a late-training drop ≥ 5% from peak on the 1.96M-edge cross-type-augmented
  connectome under σ=0.05 (pre-optimization baseline)."
- Launch:
  `python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_hybrid_flywireRF_zeroedge_cross_sl_noise_005 iterations=80 --cluster --resume`

---

## Final Summary

At exploration completion (after Block 8), append to
`/home/node/.claude/projects/-workspace--devcontainer/memory/exploration_results.md`:

### flyvis_hybrid_flywireRF_zeroedge_cross_sl_noise_005 — Key Discoveries (YYYY-MM-DD)

1. **Best metric**: conn_R2 = X.XXX ± std (N seeds, CV=X.X%, max-drop=Y%), winner config
2. **HP impact**: Which HP had largest single-parameter impact, and its optimal value
3. **Failure mode**: Was the peak-drop pattern eliminated? If yes, by which HP?
4. **Surprise**: Unexpected HP interaction
5. **Falsified hypothesis**: Which was falsified and what was learned
6. **Regularization**: Did stronger W_L1 close the zero-edge gap? Quantify
7. **Training regime**: Optimal DAL, batch_size, and why
8. **Fundamental limit**: Any plateau encountered
9. **Comparison to oracle**: How conn_R2 compares vs the 328K-edge oracle baseline
   (`flyvis_hybrid_flywireRF_noise_005`) — does the augmentation cost a recoverable
   fraction of W signal?

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_hybrid_flywireRF_zeroedge_cross_sl_noise_005

## Paper Summary (update at every block boundary)

**GNN optimization** (2 sentences on HPO findings):
Sentence 1: Best hyperparameter configuration found and the conn_R2 it achieves
(cite mean ± std, CV%, max-drop%, N seeds).
Sentence 2: Which hyperparameters were most critical to stability at 1.96M edges with
~83% zero-weight cross-type edges — what worked and what failed.

**LLM-driven exploration** (2 sentences on exploration findings):
Sentence 1: What the systematic exploration revealed about the optimization landscape
on the cross-type augmented connectome (basin width, late-collapse mechanism, critical
regularization interactions).
Sentence 2: Main causal principle established from hypothesis testing — what this tells
us about GNN training under heavy connectivity uncertainty (σ=0.05).

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean±std) | CV% | max_drop% | Verdict | Hypothesis |
| ---- | -------------- | ------------------- | --- | --------- | ------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

- Does coeff_W_L1 ≥ 1e-3 close the zero-weight recovery gap without regressing
  conn_R² on the true 328K oracle edges?
- Is the late-training drop a function of total iters (DAL × batch × n_epochs) or of
  a specific loss-tradeoff dynamic that early-stopping can fix?
- Does lower lr_W eliminate the peak-drop pattern, or just delay it?

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest→newest. This
section MUST appear before ## Current Block.**

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
