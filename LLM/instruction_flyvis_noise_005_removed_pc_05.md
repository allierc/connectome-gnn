# FlyVis Edge Removal (5%) — LLM Exploration

## Goal

Maximize **connectivity_R2** for the FlyVis connectome model with **5% of edges removed per column** (412,406 edges out of the full 434,112). The GNN must recover the dynamics of the full connectome using only the pruned graph — edges that were removed are completely absent from the GNN's edge_index.

**Scientific question**: How well does the GNN recover connectivity on the available edges when 5% of connections are missing? Does the model compensate by adjusting W on remaining edges, or does performance degrade?

The March 2026 `_er_00` runs achieved good conn_R2 with all MLP/W penalties = 0. The goal is to reproduce this and find whether any regularization helps or hurts with a pruned graph.

## Launch Command

```bash
# Run from /workspace/connectome-gnn/
python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_removed_pc_05 iterations=48 --cluster --resume
```

The pipeline auto-creates `config/fly/flyvis_noise_005_removed_pc_05_Claude_00.yaml` through `_03.yaml` on first run. **Do not create these files manually.**

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² on the **available** edges only (not the removed ones)
2. **tau_R2** — time constant recovery
3. **V_rest_R2** — resting potential recovery
4. **cluster_accuracy** — neuron type clustering from embeddings

Robustness target: **CV < 5%**, all seeds > 0.87.

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction
2. **Design experiment**: Change **EXACTLY ONE** parameter per slot to understand causality
3. **Run training**: 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use metrics AND cross-seed variance
5. **Update understanding**: Revise hypotheses based on evidence

**CRITICAL**: You can only hypothesize. Only training results validate or falsify.

### CAUSALITY RULE (MANDATORY — READ THIS)

**If you change more than one parameter per slot, you CANNOT attribute the effect. This is a fatal experimental design error.**

- In **EXPLORATION** mode: Slot 0 = parent/baseline (unchanged control). Slots 1–3 each change **exactly one** parameter from the parent.
- In **ROBUSTNESS** mode: all 4 slots use the same config (different simulation seeds test training robustness).
- Do NOT change parameters outside the current block focus.

## Data Generation

Each slot re-generates simulation data with a **different random seed** (different voltage traces), but the **same edge removal pattern** (`edge_removal_seed: 42` is fixed). Seeds are **forced by the pipeline**.

- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

**Fixed — do not change**: `n_edges`, `edge_removal_ratio`, `edge_removal_mode`, `edge_removal_seed`, `n_neurons`, `n_frames`, `delta_t`, `noise_model_level`, `use_gt_edges: true`.

## FlyVis Model (Edge Removal Variant)

```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i
```

- **13,741 neurons**, 65 cell types
- **412,406 edges** (5% of 434,112 real edges removed per column)
- Simulation uses the **full** connectome; GNN trains on the **pruned** graph
- DAVIS visual input, `noise_model_level=0.05`, 64,000 frames
- Removed edges are not present in edge_index — W is not allocated for them

**Key difference from null edges**: there is no "correct W = 0" signal. The model must simply explain observed dynamics with fewer connections. It may learn compensatory W values on remaining edges.

## Training Parameters

| Parameter                 | Default   | Description                                            |
| ------------------------- | --------- | ------------------------------------------------------ |
| `lr_W`                    | 9e-4      | Learning rate for connectivity W                       |
| `lr`                      | 1.8e-3    | Learning rate for g_phi and f_theta MLPs               |
| `lr_embedding`            | 2.325e-3  | Learning rate for neuron embeddings                    |
| `n_epochs`                | 1         | Number of training epochs                              |
| `data_augmentation_loop`  | 35        | Data augmentation multiplier                           |
| `coeff_g_phi_diff`        | 750       | Monotonicity penalty on g_phi (critical — do not zero) |
| `coeff_g_phi_norm`        | 0.9       | Normalization at saturation voltage                    |
| `coeff_f_theta_msg_diff`  | 0         | Monotonicity of f_theta w.r.t. aggregated message      |
| `coeff_g_phi_weight_L1`   | 0         | L1 on g_phi MLP weights                               |
| `coeff_f_theta_weight_L1` | 0         | L1 on f_theta MLP weights                             |
| `coeff_g_phi_weight_L2`   | 0         | L2 on g_phi MLP weights                               |
| `coeff_f_theta_weight_L2` | 0         | L2 on f_theta MLP weights                             |
| `coeff_W_L1`              | 0         | L1 sparsity on W                                       |
| `coeff_W_L2`              | 0         | L2 on W                                               |
| `regul_annealing_rate`    | 0.5       | Annealing rate for L1/L2 penalties (see below)        |

### Known failure mode

**coeff_W_L1 = 0.00015 (fixed, not annealed)**: drove W→0, conn_R2≈0.001. The old `_er_00` working runs had coeff_W_L1=0. Do not use non-zero W_L1 without annealing.

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

**Critical**: With `n_epochs=1` (only epoch 0), annealed coefficients are **always zero** regardless of configured values. To activate annealing, use `n_epochs: 2`.

**With n_epochs=2**: epoch 0 runs with all L1/L2 = 0 (model learns dynamics freely), epoch 1 applies penalties at `(1−exp(−rate))` strength (~39% for rate=0.5, ~63% for rate=1.0).

To keep total training time constant when switching from 1 to 2 epochs, **halve `data_augmentation_loop`** (35 → 17). The `claude.n_epochs` and `claude.data_augmentation_loop` fields in the template config control what the pipeline sets.

**Setting `regul_annealing_rate: 0`** with `n_epochs=1` makes coefficients apply at **full strength from epoch 0**. Use this to test direct (non-annealed) penalties.

## Training Time Constraint

**~90 min per slot on A100** (412K edges, slightly fewer than base). Data generation: ~15 min/slot. Total budget: ~120 min/slot.

With `n_epochs=2`, halve `data_augmentation_loop` (35 → 17) to stay within budget.

Use `training_time_min` from metrics after each batch to verify you're on budget.

## Parallel Mode — 4 Slots Per Batch

Each batch runs 4 slots with different simulation seeds (forced by pipeline). Same edge removal pattern across all slots (edge_removal_seed fixed). Choose the strategy:

- **Exploration** (default): Slot 0 = parent/control (unchanged). Slots 1–3 each change **exactly one** parameter. Gives 3 causal tests per batch.
- **Robustness test**: ALL 4 slots use the SAME config. Measures training robustness across different voltage trace seeds.

State your choice (exploration vs robustness test) in the log entry.

### Robustness Classification

| Class               | Criterion                                     |
|---------------------|-----------------------------------------------|
| **Stable-Robust**   | all 4 conn_R2 > 0.9, CV < 5% — **TARGET**   |
| **Robust**          | all 4 > 0.9, CV 5–10%                        |
| **Partially robust**| 2–3 > 0.9                                    |
| **Fragile**         | 0–1 > 0.9                                    |
| **DISQUALIFIED**    | any seed < 0.87                               |

## Block Partition

| Block | Mode        | Focus                                  | Parameters                                                        | Ranges                                               |
|-------|-------------|----------------------------------------|-------------------------------------------------------------------|------------------------------------------------------|
| 1     | Robustness  | **Baseline validation**                | None (all penalties=0, n_epochs=1)                                | Verify old working condition (conn > 0.9 expected)   |
| 2     | Exploration | **Learning rate tuning**               | `lr_W`, `lr`, `lr_embedding`                                      | lr_W: 5e-4–2e-3, lr: 1e-3–4e-3, lr_emb: 1e-3–4e-3  |
| 3     | Exploration | **W sparsity (direct, n_epochs=1)**    | `coeff_W_L1`, `regul_annealing_rate: 0`                           | W_L1: 1e-6–1e-4 at full strength from epoch 0        |
| 4     | Exploration | **Annealing (n_epochs=2)**             | `coeff_W_L1`, `coeff_g_phi_weight_L1`, `regul_annealing_rate`    | W_L1: 1e-5–1e-4, rate: 0.5–1.0, DAL=17             |
| 5     | Exploration | **MLP weight penalties with annealing**| `coeff_g_phi_weight_L1`, `coeff_f_theta_weight_L1`                | g_phi_L1: 0.05–0.2, f_theta_L1: 0.01–0.05           |
| 6     | Exploration | **Free exploration**                   | Any parameter                                                     | Consolidate best from blocks 1–5                     |
| 7     | Robustness  | **Final robustness**                   | None (best config from blocks 1–6)                                | 4-seed robustness test of winner config              |

### Block-specific guidance

- **Block 1**: Confirm conn_R2 > 0.9 with all penalties=0. If fragile, investigate before proceeding.
- **Block 2**: Use best LR config as parent for subsequent blocks.
- **Block 3**: `regul_annealing_rate: 0` applies W_L1 at full strength in n_epochs=1. Unlike null edges, fewer edges here means gradient per edge is slightly larger — W_L1 may be less catastrophic. Test cautiously (start at 1e-6).
- **Block 4**: Switch to `n_epochs: 2`, `data_augmentation_loop: 17`. Epoch 0 penalty-free, epoch 1 applies L1 at (1−exp(−rate)) strength. More forgiving than Block 3.
- **Block 5**: With n_epochs=2 annealing established, try MLP weight penalties. Monitor for training collapse (conn_R2 < 0.1).
- **Block 6**: Combine best LRs + best regularization. Free to test novel combinations.
- **Block 7**: Run best config as robustness test (all 4 slots identical). Target: Stable-Robust.

## Winner Config (COMPULSORY)

**At every block boundary**, save the current best config as a winner file.

1. Identify the **best iteration** (highest mean conn_R2 across 4 seeds)
2. Copy its config from `log/Claude_exploration/LLM_<task_name>/config/iter_XXX_slot_YY.yaml`
3. Save to `config/fly/flyvis_noise_005_removed_pc_05_winner.yaml` with header:

```yaml
# Winner config: flyvis_noise_005_removed_pc_05_winner.yaml
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
# Key differences from baseline (all penalties=0):
#   - [list changed parameters]
```

## Iteration Log Format

```
## Iter N: [Stable-Robust / Robust / Partially robust / Fragile / DISQUALIFIED]
Node: id=N, parent=P
Mode: [Exploration / Robustness test]
Hypothesis tested: "[quoted hypothesis]"
Config: n_epochs=X, DAL=Y, lr_W=A, lr=B, lr_emb=C, rate=D, W_L1=E, g_phi_L1=F, g_phi_diff=G
Slot 0: conn_R2=, tau_R2=, V_rest_R2=, cluster_acc=, sim_seed=, train_seed=
Slot 1: conn_R2=, tau_R2=, V_rest_R2=, cluster_acc=, sim_seed=, train_seed=
Slot 2: conn_R2=, tau_R2=, V_rest_R2=, cluster_acc=, sim_seed=, train_seed=
Slot 3: conn_R2=, tau_R2=, V_rest_R2=, cluster_acc=, sim_seed=, train_seed=
Seed stats: mean_conn_R2=, std=, CV=%, min=, max=
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
- ✓ "Direct W_L1 (n_epochs=1, rate=0) drives conn_R2→0 on pruned graphs — incompatible with edge removal"
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

- Block 1 is a **robustness test**: all 4 slots use the baseline config (all MLP/W penalties = 0, n_epochs=1).
- Hypothesis: "The March 2026 working condition (conn_R2 > 0.9) is reproducible with 5% edges removed, all penalties=0, across 4 training seeds."

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_removed_pc_05

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | n_ep | DAL | lr_W | W_L1 | g_phi_L1 | rate | conn_R2 mean±std | CV% | min | Stability |
|------|------|-----|------|------|----------|------|------------------|-----|-----|-----------|

### Established Principles

### Falsified Hypotheses

- **coeff_W_L1 = 0.00015 (n_epochs=1, rate=0)**: conn_R2≈0.001 — direct L1 drives W→0.

### Open Questions

- Does removing 5% of edges measurably reduce conn_R2 vs baseline (0% removal)?
- Can W_L1 with annealing safely push null-ish edges to zero?
- Does the model learn compensatory W values on remaining edges?

---

## Previous Block Summary

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
