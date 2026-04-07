# FlyVis Null Edges (200%) — LLM Exploration

## Goal

Maximize **connectivity_R2** for the FlyVis connectome model with **200% extra null edges** (1,302,336 total edges: 434,112 real + 868,224 null per-column). The GNN must learn W ≈ 0 for null edges and correct W for real edges.

The March 2026 runs achieved **conn_R2 ≈ 0.98, tau_R2 ≈ 0.99** with MLP regularization at 100% null edges. The goal is to determine whether the GNN remains robust at 2x the null edge contamination.

Data is **re-generated per slot** with a different null-edge placement each time (different seed). This tests robustness to arbitrary null edge structures.

## Launch Command

```bash
# Run from /workspace/connectome-gnn/
python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_null_edges_pc_200 iterations=48 --cluster --resume
```

The pipeline auto-creates `config/fly/flyvis_noise_005_null_edges_pc_200_Claude_00.yaml` through `_03.yaml` on first run. **Do not create these files manually.**

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² on real edges only
2. **tau_R2** — time constant recovery
3. **V_rest_R2** — resting potential recovery
4. **cluster_accuracy** — neuron type clustering from embeddings

Stability target: **CV < 5%**, all seeds > 0.87.

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
- In **ROBUSTNESS** mode: all 4 slots use the same config (different seeds test robustness across null-edge placements).
- Do NOT change parameters outside the current block focus.

## Data Generation

Each slot re-generates data with a **different random seed** (different null edge placement).
Seeds are **forced by the pipeline** — DO NOT modify them in config files.

- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

**Fixed — do not change**: `n_edges`, `n_extra_null_edges`, `null_edges_mode`, `n_neurons`, `n_frames`, `delta_t`, `noise_model_level`, `use_gt_edges: true`.

## FlyVis Model

```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i
```

- **13,741 neurons**, 65 cell types
- **1,302,336 total edges**: 434,112 real (W ≠ 0) + 868,224 null per-column (W = 0)
- DAVIS visual input, `noise_model_level=0.05`, 64,000 frames

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
| `coeff_g_phi_weight_L1`   | 0         | L1 on g_phi MLP weights (known failure if > 0.1)       |
| `coeff_f_theta_weight_L1` | 0         | L1 on f_theta MLP weights                             |
| `coeff_g_phi_weight_L2`   | 0         | L2 on g_phi MLP weights                               |
| `coeff_f_theta_weight_L2` | 0         | L2 on f_theta MLP weights                             |
| `coeff_W_L1`              | 0         | L1 sparsity on W (null→0)                             |
| `coeff_W_L2`              | 0         | L2 on W                                               |
| `regul_annealing_rate`    | 0.5       | Annealing rate for L1/L2 penalties (see below)        |

### Known failure mode

**coeff_g_phi_weight_L1 > 0.1** with n_epochs=1 and regul_annealing_rate=0: the L1 gradient dominates the connectivity gradient (1.3M edges → small gradient per edge), driving W→0 → conn_R2≈0.002. Do NOT set g_phi_weight_L1 > 0 in n_epochs=1 mode.

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


## Parallel Mode — 4 Slots Per Batch

Each batch runs 4 slots with different seeds (forced by pipeline). Choose the strategy:

- **Exploration** (default): Slot 0 = parent/control (unchanged). Slots 1–3 each change **exactly one** parameter. Gives 3 causal tests per batch.
- **Robustness test**: ALL 4 slots use the SAME config. Different seeds → different null-edge placements. Use when a config looks promising.

State your choice (exploration vs robustness test) in the log entry.

### Robustness Classification

| Class            | Criterion                                     |
|------------------|-----------------------------------------------|
| **Stable-Robust** | all 4 conn_R2 > 0.9, CV < 5% — **TARGET**   |
| **Robust**        | all 4 > 0.9, CV 5–10%                        |
| **Partially robust** | 2–3 > 0.9                               |
| **Fragile**       | 0–1 > 0.9                                    |
| **DISQUALIFIED**  | any seed < 0.87                               |

## Block Partition

| Block | Mode        | Focus                                  | Parameters                                                        | Ranges                                          |
|-------|-------------|----------------------------------------|-------------------------------------------------------------------|-------------------------------------------------|
| 1     | Robustness  | **Baseline validation**                | None (March config with MLP reg, n_epochs=1)                      | Verify conn≈0.98 with 200% null edges           |
| 2     | Exploration | **Learning rate tuning**               | `lr_W`, `lr`, `lr_embedding`                                      | lr_W: 5e-4–2e-3, lr: 1e-3–4e-3, lr_emb: 1e-3–4e-3 |
| 3     | Exploration | **W sparsity (direct, n_epochs=1)**    | `coeff_W_L1`, `regul_annealing_rate: 0`                           | W_L1: 1e-6–5e-5 at full strength from epoch 0  |
| 4     | Exploration | **Annealing (n_epochs=2)**             | `coeff_W_L1`, `coeff_g_phi_weight_L1`, `regul_annealing_rate`    | W_L1: 1e-5–1e-4, rate: 0.5–1.0, DAL=17        |
| 5     | Exploration | **MLP weight penalties with annealing**| `coeff_g_phi_weight_L1`, `coeff_f_theta_weight_L1`                | g_phi_L1: 0.05–0.15, f_theta_L1: 0.01–0.05     |
| 6     | Exploration | **Free exploration**                   | Any parameter                                                     | Consolidate best from blocks 1–5               |
| 7     | Robustness  | **Final robustness**                   | None (best config from blocks 1–6)                                | 4-seed robustness test of winner config         |

### Block-specific guidance

- **Block 1**: Confirm conn_R2 ≈ 0.98 across all 4 null-edge placements with 200% null edges. If performance drops vs 100%, note the degradation. If fragile (variance > 10%), investigate before proceeding.
- **Block 2**: The baseline LRs may not be optimal for 1.3M edges — explore systematically. Best LR config becomes the new parent.
- **Block 3**: `regul_annealing_rate: 0` bypasses annealing, so W_L1 is active from epoch 0 in n_epochs=1. Test small values (1e-6–5e-5) to nudge null edges to zero without breaking conn.
- **Block 4**: Switch to `n_epochs: 2`, `data_augmentation_loop: 17`. Epoch 0 is penalty-free → model learns dynamics; epoch 1 applies L1 at (1−exp(−rate)) strength. This should safely allow higher L1 values than Block 3.
- **Block 5**: With n_epochs=2 annealing established, try MLP weight penalties. Keep g_phi_weight_L1 < 0.15 — values above 0.28 are known to collapse training.
- **Block 6**: Combine best LRs (Block 2) + best regularization (Blocks 3–5). Free to test novel combinations.
- **Block 7**: Run best config as robustness test (all 4 slots identical). Target: Stable-Robust (all > 0.9, CV < 5%).

## Winner Config (COMPULSORY)

**At every block boundary**, save the current best config as a winner file.

1. Identify the **best iteration** (highest mean conn_R2 across 4 seeds)
2. Copy its config from `log/Claude_exploration/LLM_<task_name>/config/iter_XXX_slot_YY.yaml`
3. Save to `config/fly/flyvis_noise_005_null_edges_pc_200_winner.yaml` with header:

```yaml
# Winner config: flyvis_noise_005_null_edges_pc_200_winner.yaml
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
- ✓ "W_L1 with annealing (n_epochs=2) safely enables null-edge sparsification at 1.3M edges (3/3 iterations, CV < 5%)"
- ✓ "Direct penalties (n_epochs=1, rate=0) catastrophically fail at 1.3M edges — gradient per edge too small"
- ✗ "Regularization helps" (too vague, needs specifics about mechanism)

### What to Add to Open Questions

- Patterns observed 1-2 times
- Seed-dependent effects across different null-edge placements
- Contradictions between 100% and 200% null edge regimes
- Scaling behavior as edge count increases

### What to Add to Falsified Hypotheses

When a hypothesis is falsified:

1. State the original hypothesis
2. State the contradicting evidence (iteration number, metrics, comparison to 100% null edges)
3. State what was learned from the falsification
4. Propose a revised hypothesis if applicable

## Start Call

When prompt says `PARALLEL START`:

- Block 1 is a **robustness test**: all 4 slots use the March baseline config (MLP regularization active, n_epochs=1).
- Hypothesis: "The March 2026 working condition (conn_R2 ≈ 0.98) is reproducible with 200% null edges across 4 different null-edge placements."

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_null_edges_pc_200

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | n_ep | DAL | lr_W | W_L1 | g_phi_L1 | rate | conn_R2 mean±std | CV% | min | Stability |
|------|------|-----|------|------|----------|------|------------------|-----|-----|-----------|

### Established Principles

### Falsified Hypotheses

- **coeff_g_phi_weight_L1 > 0.1 (n_epochs=1, rate=0)**: conn_R2≈0.002 — L1 gradient dominates connectivity gradient with 1.3M edges.

### Open Questions

- Does 200% null edges degrade conn_R2 vs 100%?
- Can coeff_W_L1 > 0 improve null-edge sparsity without collapsing conn?
- Does annealing (n_epochs=2) allow safe use of MLP weight penalties?
- Are the March 2026 LRs (lr_W=9e-4) still optimal with 1.3M edges?

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest→newest. This section MUST appear before ## Current Block.**

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
