# FlyVis Edge Removal (20%) — LLM Exploration

## Goal

Optimize GNN hyperparameters for maximum **connectivity matrix recovery (conn_R2)** on FlyVis
with **20% of edges removed per column** (347,290 edges out of the full 434,112). The GNN must
recover the dynamics of the **partial** connectome — edges that were removed are completely absent
from edge_index, and W is not allocated for them.

**Scientific question**: Can the GNN still recover connectivity with 20% of connections missing?
Does the model learn compensatory W values on remaining edges, or does performance collapse? How
much does regularization need to change relative to the full-connectivity case?

**Key difference from full connectivity**: with 20% edges removed, there is no "correct W=0"
signal for missing edges. The model must explain observed dynamics with fewer connections.
This is a harder inverse problem than the full connectivity case and likely requires different
regularization balance.

**Key difference from 10% removal**: double the structural incompleteness. Compensatory W
adjustments must span more missing pathways. Regularization that worked at 10% may be too
aggressive here.

## Launch Command

```bash
# Run from /workspace/connectome-gnn/
python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_removed_pc_20 iterations=108 --cluster --resume
```

The pipeline auto-creates `config/fly/flyvis_noise_005_removed_pc_20_Claude_00.yaml` through
`_03.yaml` on first run. **Do not create these files manually.**

## Metrics

**Always use metrics defined to guide decision making**

During training (stdout):
```
epoch 0/1 | train: ... | conn_R2=0.XXX tau_R2=0.XXX Vr_R2=0.XXX | duration: XXs
```

During test/validation:
- **PRIMARY METRIC: `conn_R2`** — R² on the **available** edges only (not the removed ones)
- `tau_R2` — time constant recovery
- `V_rest_R2` — resting potential recovery
- `cluster_accuracy` — neuron type clustering from embeddings
- `rollout_pearson_r` — Pearson r of autoregressive rollout vs ground truth

**Robustness classification** (4 seeds per iteration):
- **Stable-Robust**: all 4 seeds conn_R2 ≥ 0.87, CV < 5% — **TARGET**
- **Robust**: all 4 seeds conn_R2 ≥ 0.87, CV 5–10%
- **Partially robust**: 2–3 seeds ≥ 0.87
- **Fragile**: 0–1 seeds ≥ 0.87
- **DISQUALIFIED**: any seed conn_R2 < 0.80 — reject config immediately

> Note: targets are slightly relaxed vs full connectivity (0.87 vs 0.90) to reflect the
> increased difficulty of 20% edge removal.

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction
2. **Design experiment**: Change **EXACTLY ONE** parameter per slot to understand causality
3. **Run training**: 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use metrics AND cross-seed variance
5. **Update understanding**: Revise hypotheses based on evidence

**CRITICAL**: You can only hypothesize. Only training results validate or falsify.

### CAUSALITY RULE (MANDATORY — READ THIS)

**If you change more than one parameter per slot, you CANNOT attribute the effect. This is a
fatal experimental design error.**

- In **EXPLORATION** mode: Slot 0 = parent/baseline (unchanged control). Slots 1–3 each change
  **exactly one** parameter from the parent.
- In **ROBUSTNESS** mode: all 4 slots use the same config (different simulation seeds test
  training robustness).
- Do NOT change parameters outside the current block focus.

## Data Generation

Each slot re-generates simulation data with a **different random seed** (different voltage traces),
but the **same edge removal pattern** (`edge_removal_seed: 42` is fixed). Seeds are **forced by
the pipeline**.

- `simulation.seed = iteration × 1000 + slot`
- `training.seed = iteration × 1000 + slot + 500`

**Fixed — do not change**: `n_edges`, `edge_removal_ratio`, `edge_removal_mode`,
`edge_removal_seed`, `n_neurons`, `n_frames`, `delta_t`, `noise_model_level`, `use_gt_edges: true`.

## FlyVis Model (20% Edge Removal)

```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i
```

- **13,741 neurons**, 65 cell types
- **347,290 edges** (20% of 434,112 real edges removed per column)
- Simulation uses the **full** connectome; GNN trains on the **pruned** graph
- DAVIS visual input, `noise_model_level=0.05`, 64,000 frames, delta_t=0.02
- Removed edges are not present in edge_index — W is not allocated for them

**Key difference from null edges**: there is no "correct W = 0" signal. The model must simply
explain observed dynamics with fewer connections. It may learn compensatory W values on remaining
edges.

## GNN Architecture Parameters

| Parameter       | Default | Description                                             |
| --------------- | ------- | ------------------------------------------------------- |
| `hidden_dim`    | 80      | Width of hidden layers in g_phi and f_theta             |
| `n_layers`      | 3       | Depth of g_phi and f_theta networks                     |
| `embedding_dim` | 2       | Per-neuron learnable embedding dimension                |

## Training Parameters

| Parameter                 | Default  | Description                                                                      |
| ------------------------- | -------- | -------------------------------------------------------------------------------- |
| `lr_W`                    | 9e-4     | Learning rate for W matrix (synaptic weights)                                    |
| `lr`                      | 1.8e-3   | Learning rate for g_phi and f_theta MLP weights                                  |
| `lr_embedding`            | 2.5e-3   | Learning rate for per-neuron embeddings                                          |
| `data_augmentation_loop`  | 35       | Augmentation loops per epoch (DAL)                                               |
| `batch_size`              | 4        | Samples per batch                                                                |
| `coeff_g_phi_diff`        | 750      | Monotonicity penalty on g_phi (**critical — do not zero**)                       |
| `coeff_g_phi_norm`        | 0.9      | Normalization at saturation voltage                                              |
| `coeff_f_theta_msg_diff`  | 0        | Monotonicity of f_theta w.r.t. aggregated message                                |
| `coeff_g_phi_weight_L1`   | 0.28     | L1 weight regularization on g_phi network                                       |
| `coeff_f_theta_weight_L1` | 0.05     | L1 weight regularization on f_theta network                                     |
| `coeff_g_phi_weight_L2`   | 0        | L2 weight regularization on g_phi network                                       |
| `coeff_f_theta_weight_L2` | 0.001    | L2 weight regularization on f_theta network                                     |
| `coeff_W_L1`              | 1.5e-4   | L1 sparsity on W — **see failure risk below**                                    |
| `coeff_W_L2`              | 1.5e-6   | L2 regularization on W                                                          |
| `regul_annealing_rate`    | 0.0      | Annealing rate — **0.0 = no annealing = full strength from epoch 0**             |
| `w_init_mode`             | randn_scaled | W initialization: `randn_scaled`, `zeros`, `uniform_scaled`               |
| `w_init_scale`            | 1.0      | Scale for randn_scaled/uniform_scaled init (bound = scale/sqrt(n_edges))        |

**Training time budget**: Target ~60 min per run. Adjust DAL to stay within budget. Check
`training_time_min` in results after each iteration.

**Hard runtime limit (120 min)**: Cluster enforces 120-min wall-clock limit. Check for
`_interrupted` in slot log directories. If interrupted, reduce DAL for next iteration.

**Fixed: n_epochs=1** — do not change. With n_epochs=1, `regul_annealing_rate` MUST be 0.0
(annealing formula: effective_coeff = coeff × (1 − exp(−rate × epoch)) = 0 at epoch 0).

**Note**: Seeds are pipeline-controlled. Do not set seeds in config files.

> **YAML rule**: Always wrap the `description` field value in double quotes — colons inside
> unquoted YAML strings cause parse errors.

## Regularization Annealing

```
effective_coeff(epoch) = configured_coeff × (1 − exp(−rate × epoch))
```

| Epoch | rate=0.0 (no annealing) | rate=0.5 | rate=1.0 |
|-------|:-----------------------:|:--------:|:--------:|
| 0     | **full strength**       | 0.00     | 0.00     |
| 1     | full strength           | 0.39     | 0.63     |
| 2     | full strength           | 0.63     | 0.86     |

**Critical with `rate=0.0`**: All penalties apply at **full configured strength from epoch 0**.
With `n_epochs=1` (only epoch 0), this is the only regime available.

**Critical with `rate>0` and `n_epochs=1`**: Annealed coefficients are **always zero**
regardless of configured values (epoch 0 → formula gives 0 for any rate > 0).

To switch to annealed training: set `n_epochs: 2`, `data_augmentation_loop: 17` (halve to
preserve budget), and `regul_annealing_rate: 0.5`. Then epoch 0 is penalty-free and epoch 1
applies penalties at ~39% strength.

## Known Failure Risk

> **⚠ coeff_W_L1 = 1.5e-4 with regul_annealing_rate = 0.0**: The default config applies
> W_L1 at full strength from epoch 0. For the 10% removal case (same noise level), this
> configuration drove W→0 and conn_R2≈0.001. **Block 1 must verify whether this also
> collapses training for 20% removal.** If Block 1 shows catastrophic failure, immediately
> switch to coeff_W_L1=0 as the Block 2 baseline.

Prior working configuration for edge-removal variants: `coeff_W_L1=0, regul_annealing_rate=0.5`
(even with n_epochs=1 this is safe — annealing zeroes all penalties at epoch 0).

## Data Generation

`generate_data: false` — data is pre-generated and NOT regenerated each iteration.

**DO NOT modify simulation parameters** (n_neurons, n_frames, n_edges, delta_t, noise_model_level,
edge_removal_ratio, edge_removal_seed).

## Block Structure

| Block | Focus                      | Parameters to scan                                               | Ranges                                                                              |
| ----- | -------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 1     | **Baseline robustness**    | All 4 slots = default config (ROBUSTNESS)                       | Verify default config works with 20% removal; diagnose W_L1 failure risk           |
| 2     | **Learning rates**         | `lr_W`, `lr`, `lr_embedding`                                    | lr_W: {3e-4, 6e-4, 9e-4, 1.2e-3}; lr: {9e-4, 1.8e-3, 3.6e-3}; lr_W/lr ratio     |
| 3     | **g_phi regularization**   | `coeff_g_phi_diff`, `coeff_g_phi_norm`, `coeff_g_phi_weight_L1` | diff: {375, 750, 1500}; norm: {0, 0.5, 0.9, 1.5}; g_L1: {0, 0.1, 0.28, 0.5}     |
| 4     | **f_theta regularization** | `coeff_f_theta_weight_L1`, `coeff_f_theta_weight_L2`            | f_L1: {0, 0.01, 0.05, 0.2}; f_L2: {0, 0.001, 0.01}                               |
| 5     | **W reg + annealing**      | `coeff_W_L1`, `coeff_W_L2`, `regul_annealing_rate`, `n_epochs`  | W_L1: {0, 5e-5, 1.5e-4, 5e-4}; W_L2: {0, 1.5e-6}; rate: {0, 0.5, 1.0}; ep: {1, 2} |
| 6     | **W initialization**       | `w_init_mode`, `w_init_scale`                                   | mode: {randn_scaled, zeros, uniform_scaled}; scale: {0.1, 0.25, 0.5, 1.0}         |
| 7     | **Training regime**        | `batch_size`, `data_augmentation_loop`                           | bs: {2, 4, 8}; DAL: {15, 25, 35, 50}                                              |
| 8     | **Free exploration**       | Any parameter — combine best from Blocks 2-7                    | Focus: reduce fragility; compare with 10% removal findings                         |
| 9     | **Final robustness**       | Best config, all 4 slots same (ROBUSTNESS, `generate_data: false`) | Confirm CV < 5%, zero catastrophic; same data across seeds                    |
| 10    | **CV robustness (8 seeds)**| Best config, 8 seeds over 2 iterations (ROBUSTNESS, `generate_data: true`) | True seed independence; target CV < 5%, mean > 0.87, ≤1 catastrophic/8 |

**Extra blocks** (optional, use if Block 8 did not converge to a clear winner):
append additional EXPLORATION iterations on any block focus before proceeding to Blocks 9-10.

> **Block 1 emergency protocol**: If Block 1 shows catastrophic failure (conn_R2 < 0.5 for
> all seeds), skip Blocks 2-4 and immediately run a diagnostic: Slot 0 = default,
> Slot 1 = coeff_W_L1=0 (all else same), Slot 2 = coeff_W_L1=0 + rate=0.5, Slot 3 = 
> coeff_W_L1=0 + n_epochs=2 + DAL=17 + rate=0.5. This identifies the failure cause and
> establishes a safe baseline for Blocks 2+.

> **generate_data flag for CV robustness**: Before running Block 10, set `generate_data: true`
> in the config. After Block 10, reset to `generate_data: false`.

## Parallel Mode — 4 Slots Per Batch

Each batch runs 4 slots with different simulation seeds (forced by pipeline). Same edge removal
pattern across all slots (edge_removal_seed fixed). Choose the strategy:

- **Exploration** (default): Slot 0 = parent/control (unchanged). Slots 1–3 each change **exactly
  one** parameter. Gives 3 causal tests per batch.
- **Robustness test**: ALL 4 slots use the SAME config. Measures training robustness across
  different voltage trace seeds.

State your choice (exploration vs robustness test) in the log entry.

## Winner Config (COMPULSORY)

**At every block boundary**, save the current best config as a winner file.

1. Identify the **best iteration** (highest mean conn_R2 across 4 seeds)
2. Copy its config from `log/Claude_exploration/LLM_flyvis_noise_005_removed_pc_20/config/iter_XXX_slot_YY.yaml`
3. Save to `config/fly/flyvis_noise_005_removed_pc_20_winner.yaml` with header:

```yaml
# Winner config: flyvis_noise_005_removed_pc_20_winner.yaml
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
#   rollout_pearson_r: X.XXX
#
# Key differences from baseline (default config):
#   - [list changed parameters]
#
# Comparison to edge-removal variants:
#   - removed_pc_10 winner: conn_R2=X.XXX
#   - removed_pc_05 winner: conn_R2=X.XXX
```

## Iteration Log Format

```
## Iter N: [Stable-Robust / Robust / Partially robust / Fragile / DISQUALIFIED]
Node: id=N, parent=P
Mode: [Exploration / Robustness test]
Hypothesis tested: "[quoted hypothesis]"
Config: n_epochs=X, DAL=Y, lr_W=A, lr=B, lr_emb=C, rate=D, W_L1=E, W_L2=F,
        g_diff=G, g_norm=H, g_L1=I, f_L1=J, f_L2=K, w_init=L(scale=M), bs=N
Slot 0: conn_R2=, tau_R2=, V_rest_R2=, cluster_acc=, rollout_r=, sim_seed=, train_seed=
Slot 1: conn_R2=, tau_R2=, V_rest_R2=, cluster_acc=, rollout_r=, sim_seed=, train_seed=
Slot 2: conn_R2=, tau_R2=, V_rest_R2=, cluster_acc=, rollout_r=, sim_seed=, train_seed=
Slot 3: conn_R2=, tau_R2=, V_rest_R2=, cluster_acc=, rollout_r=, sim_seed=, train_seed=
Seed stats: mean_conn_R2=, std=, CV=%, min=, max=, catastrophic=N/4
Mutation: [param]: [old] -> [new]
Verdict: [supported / falsified / inconclusive] — [one line]
Next: parent=P
```

## Block Boundaries

At each block boundary:
1. Save winner config file (COMPULSORY)
2. Update "Established Principles" in memory
3. Summarize block findings
4. Clear "Current Block Iterations"
5. Carry forward best config as parent for next block

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
- If "Pending Instructions" section has content: act on it, then move entries to "Acknowledged" section with timestamp
- Do not remove acknowledged entries — append them with `[ACK {batch}]` marker

## Knowledge Base Guidelines

### What to Add to Established Principles

A principle must satisfy ALL of:

1. Observed consistently across **3+ iterations**
2. Consistent across **all 4 seeds** (not just mean, but low variance)
3. States a **causal relationship** (not just a correlation)

Examples:
- ✓ "W_L1 with annealing (n_epochs=2) safely enables sparsification without collapsing connectivity (3/3 iterations, CV < 5%)"
- ✓ "Direct W_L1=1.5e-4 (rate=0) drives conn_R2→0 on pruned graphs — incompatible with 20% edge removal"
- ✗ "LR tuning helps" (too vague, needs specifics)

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

## Start Call

When prompt says `PARALLEL START`:

- Block 1 is a **robustness test**: all 4 slots use the **default config** (as in
  `flyvis_noise_005_removed_pc_20.yaml`).
- Hypothesis: "The default config (coeff_W_L1=1.5e-4, rate=0.0) achieves conn_R2 ≥ 0.87 with
  CV < 10% across 4 seeds with 20% edges removed. Risk: direct W_L1 may collapse W→0 as seen in
  the 10% removal case."
- If Block 1 is catastrophic: immediately run the Block 1 emergency protocol (see Block Structure).

## Final Summary (write at exploration completion)

When the exploration is complete (all blocks done or budget exhausted), append to
`/home/node/.claude/projects/-workspace--devcontainer/memory/exploration_results.md`
a section with header `## flyvis_noise_005_removed_pc_20 — Key Discoveries (YYYY-MM-DD)` containing
exactly **8 bullet points**:

1. **Best metric**: conn_R2 = X.XXX ± std (N seeds, CV=X.X%), winner config = [key params]
2–8. **Key causal discoveries** — report findings of this kind:
   - Which HP had the largest single-parameter impact, and its optimal value
   - Which failure mode was confirmed across 3+ iterations (cite iteration numbers)
   - Which HP interaction produced an unexpected or surprising result
   - Which hypothesis was falsified and what was learned from it
   - Whether W_L1 with/without annealing helped or hurt with 20% missing edges
   - What training regime (n_epochs, DAL, batch_size) proved optimal and why
   - How conn_R2 compares to 10% removal winner — is the degradation linear or not?

Each bullet must state the **finding**, the **evidence** (iteration count or specific iterations),
and whether it is **established** (3+ iterations, all 4 seeds) or **tentative** (1–2 iterations).

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_removed_pc_20

## Paper Summary (update at every block boundary)

**GNN optimization** (2 sentences on HPO findings):
Sentence 1: Best hyperparameter configuration found and the connectivity_R2 it achieves (cite mean ± std, CV%, N seeds).
Sentence 2: Which hyperparameters were most critical to stability — what worked and what failed (cite values).

**LLM-driven exploration** (2 sentences on exploration findings):
Sentence 1: What the systematic exploration revealed about the optimization landscape (basin width, failure modes, critical interactions) with 20% missing edges.
Sentence 2: Main causal principle established from hypothesis testing — what this tells us about GNN training for connectome recovery with heavily pruned graphs.

## Knowledge Base

### Robustness Comparison Table

| Iter | n_ep | DAL | lr_W | W_L1 | rate | g_L1 | f_L1 | w_init | conn_R2 mean±std | CV% | min | Stability |
|------|------|-----|------|------|------|------|------|--------|------------------|-----|-----|-----------|

### Established Principles

### Falsified Hypotheses

- **coeff_W_L1 = 1.5e-4 (n_epochs=1, rate=0)**: RISK from 10% removal case — may drive W→0.
  Pending Block 1 validation.

### Open Questions

- Does the default config (W_L1=1.5e-4, rate=0) collapse training at 20% removal, as it did
  at 10% removal?
- How much does conn_R2 degrade vs 10% removal under the best config?
- Does the model learn compensatory W values on remaining edges, and does regularization interfere?
- Is uniform_scaled init more robust than randn_scaled for incomplete connectivity?

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest→newest. This section MUST
appear before ## Current Block.**

### Block 1 Summary
[Summary of findings from block 1]

### Block 2 Summary
[Summary of findings from block 2]

### Block 3 Summary
[Summary of findings from block 3]

### Block 4 Summary
[Summary of findings from block 4]

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
