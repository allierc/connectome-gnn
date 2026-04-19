# FlyVis Null Edges (400%) — LLM Exploration

## Goal

Maximize **connectivity_R2** on real edges for the FlyVis connectome model with **400% extra null
edges** (2,170,560 total edges: 434,112 real + 1,736,448 null per-column, 4× the real count). The
GNN must learn W ≈ 0 for null edges and correct W on real edges despite a 5× increase in edge
count vs noise_005 baseline.

Two sub-goals:

- Maximize conn_R2 on real edges (primary): recover synaptic weight matrix W even with 80% of
  edges being null distractors
- Maximize robustness: CV < 5% across seeds, zero catastrophic failures

The pc_100 runs achieved **conn_R2 ≈ 0.98** with all MLP/W penalties = 0; pc_200 confirmed
robustness at 2× null contamination. At 4× null edges the per-edge gradient magnitude drops
further, so regularization settings that worked at 100%–200% may no longer be safe.

Data is **re-generated per slot** with a different null-edge placement each time (different seed).
This tests robustness to arbitrary null edge structures.

## Scientific Context

**Core research question: Can a GNN still recover the real synaptic weight matrix when the
connectome is contaminated with 4× as many spurious null edges as real ones?**

The NeurIPS 2026 paper reports conn_R2 = 0.97 ± 0.01 at +400% null edges (Table
`tab:cv_per_condition`, row `+400% null edges`). The goal of this exploration is to validate that
result on the current code path, probe whether the working config is stable across null-edge
placements, and determine whether any regularization can be added without breaking recovery.

Key challenges to investigate at 2.17M edges:
- L1 gradient per edge shrinks as N_edges grows — penalties that worked at 100K are dangerous at 2M
- coeff_g_phi_diff prevents trivial g_phi collapse — its tuning is critical
- Data regeneration per slot tests robustness to null-edge placement, not just training stochasticity
- Whether `use_gt_edges: true` mask is the right interface for null edges (real edges get correct W,
  null edges must be pushed to 0 by gradient signal alone)

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
- **PRIMARY METRIC: `conn_R2`** (R² of learned W vs ground-truth W, **real edges only**)
- `tau_R2`: R² of τ (time constant) recovery
- `V_rest_R2`: R² of V_rest (resting potential) recovery
- `cluster_accuracy`: cell-type clustering accuracy from neuron embeddings
- `rollout_pearson_r`: Pearson r of autoregressive rollout vs ground truth

**Robustness classification** (4 seeds per iteration, different null-edge placements):
- **Stable-Robust**: all 4 seeds conn_R2 ≥ 0.90, CV < 5%
- **Stable**: mean conn_R2 ≥ 0.85, CV < 10%
- **Unstable**: mean < 0.85 OR CV ≥ 10%
- **Catastrophic**: any seed conn_R2 < 0.50

**Note on τ_R2 and V_rest_R2**: Model `flyvis_A` absorbs τ and V_rest implicitly into f_theta.
These metrics show 0.00 or N/A — this is expected behavior, not a failure.

Data **IS re-generated** each slot (`generate_data: true`) — each slot gets a different null-edge
placement via the per-slot sim seed. Do not change this to `false`.

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

- In EXPLORATION mode: Slot 0 = parent/baseline (unchanged control). Slots 1–3 each change
  **exactly one** parameter from the parent.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness across
  null-edge placements).

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i(t)
```

- **13,741 neurons**, 65 cell types
- **2,170,560 total edges**: 434,112 real (W ≠ 0) + 1,736,448 null per-column (W = 0, 4× real)
- **1,736 input neurons** (photoreceptors, DAVIS visual input)
- Noise level: σ=0.05 per time step
- 64,000 frames, delta_t = 0.02
- Model `flyvis_A`: f_theta absorbs τ and V_rest implicitly (τ_R2=0, Vr_R2=0 is expected)

## GNN Architecture

```
g_phi(v_j, embed_j) → message_ij          (edge MLP, per-edge messages)
sum_j W_ij * g_phi(v_j) → agg_i           (weighted aggregation)
f_theta(v_i, agg_i, embed_i) → dv_i/dt   (node update MLP)
```

- Per-neuron embedding: learnable `embedding_dim`-dimensional vector (concatenated to inputs)
- `g_phi_positive=true`: g_phi output squared (non-negative; Dale's-law approximation)

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
| `lr_W`                    | 9e-4      | Learning rate for connectivity W                                            |
| `lr`                      | 1.8e-3    | Learning rate for g_phi and f_theta MLP weights                             |
| `lr_embedding`            | 2.5e-3    | Learning rate for per-neuron embeddings                                     |
| `data_augmentation_loop`  | 35        | Augmentation loops per epoch — controls training time (DAL)                 |
| `batch_size`              | 4         | Samples per batch                                                           |
| `coeff_g_phi_diff`        | 750       | Monotonicity penalty on g_phi (**critical — do not zero**)                  |
| `coeff_g_phi_norm`        | 0.9       | Normalization at saturation voltage                                         |
| `coeff_f_theta_msg_diff`  | 0         | Monotonicity of f_theta w.r.t. aggregated message                           |
| `coeff_g_phi_weight_L1`   | 0.28      | L1 on g_phi MLP weights (**dangerous at 2.17M edges — see failure mode**)  |
| `coeff_f_theta_weight_L1` | 0.05      | L1 on f_theta MLP weights                                                   |
| `coeff_g_phi_weight_L2`   | 0         | L2 on g_phi MLP weights                                                     |
| `coeff_f_theta_weight_L2` | 0.001     | L2 on f_theta MLP weights                                                   |
| `coeff_W_L1`              | 1.5e-4    | L1 sparsity on W (null→0)                                                   |
| `coeff_W_L2`              | 1.5e-6    | L2 on W                                                                     |
| `regul_annealing_rate`    | 0.0       | **MUST be 0.0 with n_epochs=1** (otherwise all L1/L2 = 0)                   |
| `w_init_mode`             | randn_scaled | W initialization: `randn_scaled`, `zeros`, `uniform_scaled`              |
| `w_init_scale`            | 1.0       | Scale for randn_scaled/uniform_scaled init (bound = scale/sqrt(n_edges))    |

### Known failure mode (scaled for 2.17M edges)

**coeff_g_phi_weight_L1 > 0.1** with n_epochs=1 and regul_annealing_rate=0: the L1 gradient
dominates the connectivity gradient (2.17M edges → per-edge gradient ≈ 1/5 of pc_100,
1/2.5 of pc_200). The per-column null structure means null edges outnumber real 4:1, so if the
L1 penalty washes out the data gradient, W collapses to 0 → conn_R2 ≈ 0.002. Be especially
conservative with g_phi_L1 at this edge count; pc_100/pc_200 found all MLP penalties = 0 safest.

**Training time budget**: Target ~60 min per run. Adjust DAL to stay within budget. Check
`training_time_min` in results after each iteration. At 2.17M edges each forward/backward is
slower — expect longer runtimes than noise_005 baseline.

**Hard runtime limit (120 min)**: Cluster enforces 120-min wall-clock limit. Check for
`_interrupted` in slot log directories. If interrupted, reduce DAL for next iteration.

**Fixed: n_epochs=1** — do not change unless entering Block 4 (annealing experiments). With
n_epochs=1, `regul_annealing_rate` MUST be 0.0 (annealing formula: effective_coeff =
coeff × (1 − exp(−rate × epoch)) = 0 at epoch 0).

**Note**: Seeds are pipeline-controlled (`sim_seed = iter × 1000 + slot`,
`train_seed = iter × 1000 + slot + 500`). Do not set seeds in config files.

> **YAML rule**: Always wrap the `description` field value in double quotes — colons inside
> unquoted YAML strings cause parse errors.

## Regularization Annealing

```
effective_coeff(epoch) = configured_coeff × (1 − exp(−rate × epoch))
```

| Epoch | rate=0.5 | rate=1.0 |
|-------|:--------:|:--------:|
| 0     | 0.00     | 0.00     |
| 1     | 0.39     | 0.63     |
| 2     | 0.63     | 0.86     |
| 5     | 0.92     | 0.99     |

**Critical**: With `n_epochs=1` (only epoch 0), annealed coefficients are **always zero**
regardless of configured values. To activate annealing, use `n_epochs: 2`.

**With n_epochs=2**: epoch 0 runs with all L1/L2 = 0 (model learns dynamics freely), epoch 1
applies penalties at `(1−exp(−rate))` strength (~39% for rate=0.5, ~63% for rate=1.0).

To keep total training time constant when switching from 1 to 2 epochs, **halve
`data_augmentation_loop`** (35 → 17). At 2.17M edges this may still overrun — monitor
`training_time_min` and adjust.

**Setting `regul_annealing_rate: 0`** with `n_epochs=1` makes coefficients apply at **full strength
from epoch 0**. Use this to test direct (non-annealed) penalties.

## Data Generation

`generate_data: true` — data is re-generated **per slot** with a different null-edge placement.
Per-column null edges are sampled from the pre-synaptic neuron set of each column, so null
placements vary across seeds while remaining structurally consistent.

**DO NOT modify simulation parameters** (n_neurons, n_frames, n_edges, n_extra_null_edges,
null_edges_mode, delta_t, noise_model_level, use_gt_edges).

## Block Structure

| Block | Mode        | Focus                                  | Parameters                                                        | Ranges                                          |
| ----- | ----------- | -------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------- |
| 1     | Robustness  | **Baseline validation**                | Default config (March 2026 penalties, n_epochs=1)                 | Verify conn_R2 ≈ 0.97 across 4 null-edge placements |
| 2     | Robustness  | **Penalty-free baseline**              | All MLP/W penalties = 0, n_epochs=1                               | Test pc_100/pc_200 "no-penalty" finding at 2.17M edges |
| 3     | Exploration | **Learning rate tuning**               | `lr_W`, `lr`, `lr_embedding`                                      | lr_W: {3e-4, 6e-4, 9e-4, 1.5e-3}; lr: {9e-4, 1.8e-3, 3.6e-3}; lr_W/lr ratio |
| 4     | Exploration | **W sparsity (direct, n_epochs=1)**    | `coeff_W_L1`, `regul_annealing_rate: 0`                           | W_L1: {0, 1e-6, 5e-6, 5e-5, 1.5e-4} at full strength from epoch 0 |
| 5     | Exploration | **Annealing (n_epochs=2)**             | `coeff_W_L1`, `coeff_g_phi_weight_L1`, `regul_annealing_rate`     | W_L1: {1e-5, 5e-5, 1e-4}, rate: {0.5, 1.0}, DAL=17 |
| 6     | Exploration | **MLP weight penalties with annealing**| `coeff_g_phi_weight_L1`, `coeff_f_theta_weight_L1`                | g_phi_L1: {0, 0.05, 0.1}; f_theta_L1: {0, 0.01, 0.05} — keep g_phi_L1 < 0.1 |
| 7     | Exploration | **Free exploration**                   | Any parameter — combine best from Blocks 3–6                      | Novel combinations; test surprising interactions |
| 8     | Robustness  | **Final robustness**                   | Best config, all 4 slots same (ROBUSTNESS, `generate_data: true`) | Confirm CV < 5%, zero catastrophic failures across 4 null-edge placements |
| 9     | Robustness  | **CV robustness**                      | Best config, 8 seeds over 2 iterations (ROBUSTNESS, `generate_data: true`) | True seed independence across 8 null-edge placements; target CV < 5% |

### Block-specific guidance

- **Block 1**: Confirm conn_R2 ≈ 0.97 across 4 null-edge placements with the March 2026 default
  config (L1 penalties active). If performance drops vs pc_200, investigate whether the penalties
  are too strong at 2.17M edges.
- **Block 2**: Compare to penalty-free baseline — pc_100/pc_200 found all penalties = 0 often
  worked better. This tests whether the same holds at 4× null contamination.
- **Block 3**: The baseline LRs may not scale to 2.17M edges — per-edge gradient is diluted.
  Best LR config becomes the new parent. Expect lower lr_W may be needed for stability.
- **Block 4**: `regul_annealing_rate: 0` bypasses annealing, so W_L1 is active from epoch 0 in
  n_epochs=1. Test small values (1e-6–5e-5) to nudge null edges to zero without breaking conn.
- **Block 5**: Switch to `n_epochs: 2`, `data_augmentation_loop: 17`. Epoch 0 is penalty-free →
  model learns dynamics; epoch 1 applies L1 at (1−exp(−rate)) strength.
- **Block 6**: With n_epochs=2 annealing established, try MLP weight penalties. Keep
  g_phi_weight_L1 < 0.1 — values ≥ 0.1 are known to collapse training at 2.17M edges.
- **Block 7**: Combine best LRs (Block 3) + best regularization (Blocks 4–6). Free to test novel
  combinations.
- **Block 8**: Run best config as robustness test (all 4 slots identical, fresh null-edge
  placements). Target: Stable-Robust (all > 0.9, CV < 5%).
- **Block 9**: Extend Block 8 to 8 seeds (2 iterations × 4 slots). Confirms true independence
  across null-edge placements and training stochasticity.

**Extra blocks** (optional, use if Block 7 did not converge to a clear winner):
append additional EXPLORATION iterations on any block focus before proceeding to Blocks 8–9.

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
- Keep ≤ 500 lines

### 3. User Input (read every batch, acknowledge pending items)

**File**: `user_input.md`

- Read at every batch
- If "Pending Instructions" section has content: act on it, then move entries to "Acknowledged"
  section with timestamp
- Do not remove acknowledged entries — append them with `[ACK {batch}]` marker

## Knowledge Base Guidelines

### What to Add to Established Principles

A principle must satisfy ALL of:

1. Observed consistently across **3+ iterations**
2. Consistent across **all 4 seeds** (not just mean, but low variance)
3. States a **causal relationship** (not just a correlation)

Examples:
- ✓ "W_L1 with annealing (n_epochs=2) safely enables null-edge sparsification at 2.17M edges
  (3/3 iterations, CV < 5%)"
- ✓ "Direct g_phi_L1 ≥ 0.1 (n_epochs=1, rate=0) drives conn_R2→0 at 2.17M edges — L1 gradient
  dominates connectivity gradient"
- ✗ "LRs matter for null edges" (too vague, needs specifics)

### What to Add to Open Questions

- Patterns observed 1-2 times
- Seed-dependent effects across different null-edge placements
- Contradictions between iterations or vs pc_100/pc_200
- Scaling behavior as null-edge count increases

### What to Add to Falsified Hypotheses

1. State the original hypothesis
2. State the contradicting evidence (iteration number, metrics)
3. State what was learned from the falsification
4. Propose a revised hypothesis if applicable

## Winner Config (COMPULSORY)

**At every block boundary**, save the current best config as a winner file.

1. Identify the **best iteration** (highest mean conn_R2 across 4 seeds)
2. Copy its config from `log/Claude_exploration/LLM_<task_name>/config/iter_XXX_slot_YY.yaml`
3. Save to `config/fly/flyvis_noise_005_null_edges_pc_400_winner.yaml` with header:

```yaml
# Winner config: flyvis_noise_005_null_edges_pc_400_winner.yaml
# Source: iter_XXX_slot_YY (connectivity_R2 = X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [1-2 sentence narrative]
#   - [key hyperparameter choices]
#
# Metrics:
#   connectivity_R2: X.XXX (mean ± std, N seeds, CV=X.X%)
#   tau_R2:          X.XXX
#   V_rest_R2:       X.XXX
#   cluster_accuracy: X.XXX
#
# Key differences from baseline:
#   - [list changed parameters]
```

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
Mode: [Exploration / Robustness test]
Hypothesis tested: "[quoted hypothesis]"
Config: n_epochs=X, DAL=Y, lr_W=A, lr=B, lr_emb=C, rate=D,
        W_L1=E, W_L2=F, g_phi_diff=G, g_phi_norm=H, g_phi_L1=I, f_theta_L1=J,
        w_init=K, emb_dim=L
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

1. Save winner config file (COMPULSORY)
2. Update "Paper Summary" in memory
3. Summarize block findings
4. Update "Established Principles" and "Falsified Hypotheses"
5. Clear "Current Block"
6. Carry forward best config as parent for next block

## Start Call

When prompt says `PARALLEL START`:

- Block 1 is ROBUSTNESS mode: all 4 slots use the default config
  (`flyvis_noise_005_null_edges_pc_400.yaml`) with different null-edge placements:
  `lr_W=9e-4, lr=1.8e-3, lr_embedding=2.5e-3, batch_size=4, DAL=35`
  `coeff_g_phi_diff=750, coeff_g_phi_norm=0.9, coeff_g_phi_weight_L1=0.28`
  `coeff_f_theta_weight_L1=0.05, coeff_f_theta_weight_L2=0.001`
  `coeff_W_L1=1.5e-4, coeff_W_L2=1.5e-6, w_init_mode=randn_scaled, embedding_dim=2`
- Hypothesis: "The default config (noise_005-style penalties, n_epochs=1) achieves
  conn_R2 ≥ 0.95 with CV < 5% across 4 different null-edge placements at 2.17M edges.
  Paper target: conn_R2 = 0.97 ± 0.01."
- Launch:
  `python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_null_edges_pc_400 iterations=96 --cluster --resume`

The pipeline auto-creates `config/fly/flyvis_noise_005_null_edges_pc_400_Claude_00.yaml` through
`_03.yaml` on first run. **Do not create these files manually.**

---

## Final Summary

At exploration completion (after Block 9), append to
`/home/node/.claude/projects/-workspace--devcontainer/memory/exploration_results.md`:

### flyvis_noise_005_null_edges_pc_400 — Key Discoveries (YYYY-MM-DD)

1. **Best metric**: conn_R2 = X.XXX ± std (N seeds, CV=X.X%), winner config = [key params]
2. **HP impact**: Which HP had the largest single-parameter impact, and its optimal value
3. **Failure mode**: Which failure mode was confirmed across 3+ iterations (cite iteration numbers)
4. **Surprise**: Which HP interaction produced an unexpected or surprising result
5. **Falsified hypothesis**: Which hypothesis was falsified and what was learned from it
6. **Regularization at scale**: Whether the gradient-dilution principle (pc_100/pc_200) held,
   broke, or had a different threshold at 2.17M edges
7. **Training regime**: What training regime (DAL, batch_size, n_epochs) proved optimal and why
8. **Fundamental limit**: Any fundamental limit encountered (e.g., metric plateau despite HP variation)
9. **Scaling**: How conn_R2 and CV compare across pc_100 → pc_200 → pc_400

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_null_edges_pc_400

## Paper Summary (update at every block boundary)

**GNN optimization** (2 sentences on HPO findings):
Sentence 1: Best hyperparameter configuration found and the conn_R2 it achieves (cite mean ± std, CV%, N seeds).
Sentence 2: Which hyperparameters were most critical to stability at 2.17M edges — what worked and what failed (cite values and CV impact).

**LLM-driven exploration** (2 sentences on exploration findings):
Sentence 1: What the systematic exploration revealed about the optimization landscape at 4× null contamination (basin width, failure modes, gradient-dilution thresholds).
Sentence 2: Main causal principle established from hypothesis testing — what this tells us about GNN training for connectome recovery when null edges outnumber real edges 4:1.

## Knowledge Base

### Robustness Comparison Table

| Iter | n_ep | DAL | lr_W | lr | W_L1 | g_phi_L1 | f_theta_L1 | rate | conn_R2 mean±std | CV% | min | Stability |
|------|------|-----|------|----|----- |----------|------------|------|------------------|-----|-----|-----------|

### Established Principles

### Falsified Hypotheses

- **coeff_g_phi_weight_L1 > 0.1 (n_epochs=1, rate=0) at 2.17M edges**: expected failure mode —
  gradient dilution drives W→0. Test in Block 1/2 to confirm at this edge count.

### Open Questions

- Does conn_R2 degrade vs pc_200 at 4× null contamination?
- Can coeff_W_L1 > 0 improve null-edge sparsity without collapsing conn at 2.17M edges?
- Do the pc_100/pc_200 "penalty-free best" findings hold at pc_400?
- Does annealing (n_epochs=2) allow safe use of MLP weight penalties at this scale?
- Are the March 2026 LRs (lr_W=9e-4) still optimal with 2.17M edges, or is the per-edge
  gradient dilution enough to warrant lower LR?

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

**Hypothesis**: [specific, testable prediction]
**Rationale**: [why]
**Test**: [what config change]
**Expected outcome**: [support vs falsify]
**Status**: untested / supported / falsified

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
