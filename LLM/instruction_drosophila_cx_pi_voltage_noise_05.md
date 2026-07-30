# Drosophila CX PI Voltage (Noise 0.5) — LLM Exploration

## Goal

Maximize **connectivity_R2** for the **Drosophila CX path-integration voltage-recovery** task on the **noise=0.5 rollout dataset**: a NeuralGNN learns to recover the 156-neuron CX recurrent weight matrix `W_rec` from the teacher (`drosophila_cx_pi` DrosophilaCxTaskRNN) rollout activity over 64K frames, with **opto-style stimulus** restricted to PEN_a + PEN_b cells.

**Current baseline ceiling**: `connectivity_R2 ≈ 0.3`. Break this ceiling.

Data is **already generated** by `python GNN_Main.py -o generate drosophila_cx_pi_voltage_noise_05` and **reused across all iterations** — no regeneration per slot.

### Dataset summary

- `<data_root>/graphs_data/drosophila_cx/drosophila_cx_pi_voltage_noise_05/`
- 156 CX neurons, 7 cell types (EPG, EPGt, PEN_a, PEN_b, PEG, Delta7, ER6)
- ~10,263 edges (E auto-detected from `ode_params.pt`)
- 64,000 train frames + 16,000 test frames, `delta_t = 0.01s`
- Teacher: `drosophila_cx_pi` (DrosophilaCxTaskRNN, sign-locked, velocity_gate=pen_4scalar)
- Stimulus: `W_in @ u(t)` zeroed on all non-PEN rows (opto target)
- Train-rollout noise: `σ = 0.5` on `h_t` per Euler step; test split is deterministic (clean rollout for R² evaluation)

### Parent config (current best, baseline)

```
signal_model_name: drosophila_cx_voltage     # routes to NeuralGNN
use_gt_edges: true
batch_size: 8
data_augmentation_loop: 50
n_epochs: 1                                   # FORCED by claude.n_epochs
lr_W: 5e-5
lr: 1e-3
lr_embedding: 1e-3
hidden_dim: 64
embedding_dim: 2
g_phi_positive: true
coeff_g_phi_diff: 1500
coeff_g_phi_norm: 10
coeff_g_phi_weight_L1: 0.0
coeff_f_theta_weight_L2: 0.001
coeff_f_theta_diff: 5
coeff_f_theta_msg_diff: 0
coeff_W_L1: 0
coeff_W_L2: 0
coeff_W_sign: 0
dale_law: false
w_init_mode: uniform_scaled
w_init_scale: 0.01
```

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned per-edge W and ground-truth W from `ode_params.pt`
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free **test** data
3. **cluster_accuracy** (THIRD) — neuron-type clustering accuracy from learned embeddings (7 types expected)

Informational: onestep_pearson, f_theta_R2, g_phi_R2, spectral_radius_learned vs spectral_radius_true.

**NOTE**: V_rest_R2 may be unreliable (the DrosophilaCxTaskRNN bias `b` was trained, not the conventional resting potential). tau_R2 is constant (τ=0.1 for all neurons) — both informational only.

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: form a specific, testable prediction
2. **Design experiment**: change **EXACTLY ONE** parameter per slot to attribute causality
3. **Run training**: 10 slots per batch (different seeds forced by pipeline)
4. **Analyze results**: use metrics AND cross-seed variance
5. **Update understanding**: revise hypotheses based on evidence

**CRITICAL**: You can only hypothesize. Only training results validate or falsify.

### CAUSALITY RULE (MANDATORY)

**If you change more than one parameter per slot, you CANNOT attribute the effect. This is a fatal experimental design error.**

- In EXPLORATION mode: Slot 0 = parent/baseline (unchanged control). Slots 1–9 each change **exactly one** parameter from the parent.
- Do NOT change parameters outside the current block focus.
- Do NOT skip the baseline — always keep one slot as an unchanged control.
- In ROBUSTNESS mode: all 10 slots use the same config (different seeds test robustness).

## Data Generation

**`generate_data: false`** — data is already on disk. The pipeline does NOT regenerate per slot. All slots read the same `drosophila_cx_pi_voltage_noise_05/` dataset.

Per-slot variation comes from `training.seed` (forced by the pipeline: `iteration * 1000 + slot + 500`). Do NOT modify `simulation:` parameters.

**IMPORTANT**: `simulation.noise_model_level = 0.5` and `simulation.noisy_test_data = false` are part of the dataset on disk. Re-running `-o generate` is required if you want a different noise regime — use the `drosophila_cx_pi_voltage_noise_free` or `drosophila_cx_pi_voltage_noise_005` configs for that.

## Stimulus + Noise Model

Two distinct noise sources in this experiment:

1. **Rollout dynamics noise** (`σ = 0.5`, baked into the train zarrs): `h(t+1) = h(t) + dt · f(h, W_in·u, b) + ε`, ε ~ N(0, 0.5). Applied during teacher rollout, NOT during GNN training.
2. **Measurement noise**: 0 (clean voltage observations from the teacher rollout).

The **test split** is clean — rollout R² and onestep Pearson are evaluated against noise-free teacher dynamics.

The **stimulus** is opto-style: only ~42 PEN cells (PEN_a L+R, PEN_b L+R) have nonzero `stimulus[:, n]`. The other 114 neurons have `state.stimulus = 0` for all frames — their activity comes from recurrent dynamics alone.

## CX Voltage-Recovery Model

- **156 neurons**, **7 cell types** (EPG, EPGt, PEN_a, PEN_b, PEG, Delta7, ER6)
- ~10,263 edges (GT, sign-locked Dale-conformant connectome from hemibrain)
- `tau_i = 0.1` (constant per-neuron), `delta_t = 0.01s`
- Activation in teacher: sigmoid (DrosophilaCxTaskRNN); GNN learns `f_theta` directly
- PEN stim has visible CW/CCW antisymmetry in PENb_L vs PENb_R; PENa_L/R are roughly symmetric (see `task_sanity_stim.png`)

## GNN Architecture

- **g_phi**: edge message MLP. Maps `(v_src, a_src)` → message. `g_phi_positive=true` (output squared).
- **f_theta**: node update MLP. Maps `(v_i, a_i, agg_msg, I_i)` → `dv_i/dt`.
- **Embedding `a_i`**: learnable per-neuron type vector.

**CRITICAL — coupled parameters**: `embedding_dim` must be ≥ 2 (embedding_dim=1 crashes plotting). When changing `embedding_dim`, you MUST also update:

- `input_size = 1 + embedding_dim`
- `input_size_update = 3 + embedding_dim`

Example: embedding_dim=4 → input_size=5, input_size_update=7.

## Training Parameters

| Parameter                 | Default     | Description                                            |
| ------------------------- | ----------- | ------------------------------------------------------ |
| `lr_W`                    | 5e-5        | Learning rate for connectivity W                       |
| `lr`                      | 1e-3        | Learning rate for g_phi and f_theta MLPs               |
| `lr_embedding`            | 1e-3        | Learning rate for neuron embeddings                    |
| `n_epochs`                | **1**       | Forced by `claude.n_epochs` for fast iteration         |
| `batch_size`              | 8           | Batch size                                             |
| `data_augmentation_loop`  | 50          | DAL — iters/epoch multiplier                           |
| `w_init_mode`             | uniform_scaled | W init: "zeros", "uniform_scaled", "randn_scaled"  |
| `w_init_scale`            | 0.01        | Scale factor for scaled inits                          |
| `hidden_dim`              | 64          | MLP hidden dim                                         |
| `embedding_dim`           | 2           | Per-neuron embedding dim                               |
| `coeff_g_phi_diff`        | 1500        | Monotonicity penalty on g_phi                          |
| `coeff_g_phi_norm`        | 10          | g_phi output norm penalty                              |
| `coeff_g_phi_weight_L1`   | 0.0         | L1 on g_phi MLP weights                                |
| `coeff_f_theta_weight_L2` | 0.001       | L2 on f_theta MLP weights                              |
| `coeff_f_theta_diff`      | 5           | Negative-monotonicity prior on f_theta w.r.t. state    |
| `coeff_f_theta_msg_diff`  | 0           | Positive-monotonicity of f_theta w.r.t. message input  |
| `coeff_W_L1`              | 0           | L1 sparsity on W                                       |
| `coeff_W_L2`              | 0           | L2 penalty on W                                        |
| `coeff_W_sign`            | 0           | Dale's law penalty                                     |
| `dale_law`                | false       | Hard Dale's-law projection                             |
| `use_gt_edges`            | true        | **FIXED** — use ode_params edge_index for GT topology  |

> **YAML rule**: Always wrap the `description` field value in double quotes — colons inside unquoted YAML strings cause parse errors.

## Parallel Mode — 10 Slots Per Batch

Each batch runs **10 slots in parallel on A100 nodes**. Different `training.seed` per slot (forced by pipeline).

- **Exploration** (default): Slot 0 = parent/control (unchanged). Slots 1–9 each change **exactly one** parameter. Up to 9 causal tests per batch.
- **Robustness test**: ALL 10 slots use the SAME config. Measures seed robustness. Use when a config looks promising.

State your choice (exploration vs robustness test) in the log entry.

### Robustness Assessment (when running same config across 10 slots)

- **Robust**: mean conn_R2 high, CV < 5%, all slots within 0.05 of mean
- **Partially robust**: mean OK but CV 5–20% or 1–2 outliers
- **Fragile**: CV > 20% or multiple slots far below mean

## Block Structure

Total: **4 blocks × 50 iterations each = 200 iterations** (5 batches/block × 10 slots/batch).

| Block | Focus                          | Parameters to scan                                              | Suggested ranges                                                                                                                    |
| ----- | ------------------------------ | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 1     | **Learning rates**             | `lr_W`, `lr`, `lr_embedding`                                    | lr_W: {1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4}; lr: {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}; lr_emb: {1e-4, 5e-4, 1e-3, 5e-3}. 50 iters total.   |
| 2     | **Regularisation + batch size**| `coeff_W_L1`, `coeff_W_L2`, `coeff_W_sign`, `coeff_g_phi_diff`, `coeff_g_phi_norm`, `coeff_f_theta_diff`, `batch_size` | W_L1: {0, 1e-6, 3e-6, 1e-5}; W_L2: {0, 1e-6, 1e-5, 1e-4}; W_sign: {0, 0.001, 0.01, 0.1}; g_phi_diff: {500, 1000, 1500, 3000}; g_phi_norm: {0, 1, 10, 100}; f_theta_diff: {0, 1, 5, 20}; batch_size: {2, 4, 8, 16}. 50 iters. |
| 3     | **Training volume + architecture** | `data_augmentation_loop`, `hidden_dim`, `embedding_dim`, `w_init_mode`, `w_init_scale` | DAL: {25, 50, 100, 200}; hidden_dim: {32, 64, 128}; embedding_dim: {2, 4, 8}; w_init_mode: {zeros, uniform_scaled, randn_scaled}; w_init_scale: {0.001, 0.01, 0.1}. 50 iters. |
| 4     | **Free exploration / sweep everything** | Any parameter or combination of best from B1–B3              | Consolidate best of B1–B3 into a new parent; try novel combinations including parameters not touched in earlier blocks (e.g., `dale_law`, `coeff_f_theta_msg_diff`, `coeff_g_phi_weight_L1`). 50 iters. |

### Block boundary checkpoint

**At the end of every block**, save the best iteration's config as a winner file:

`config/drosophila_cx/drosophila_cx_pi_voltage_noise_05_winner.yaml`

with the same YAML-header convention as the gt_edges winners.

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (10 slots per batch)

From `analysis.log`: connectivity_R2, rollout_pearson, cluster_accuracy, dale_score, sim_seed, train_seed, training_time_min.

### Step 3: Write Log Entries + Update Memory

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, DAL=D, batch_size=B, W_L1=A, W_L2=L, W_sign=S, g_phi_diff=G, g_phi_norm=N, f_theta_diff=F, hidden_dim=H, embedding_dim=E
Slot 0: conn_R2=A, rollout_r=B, cluster_acc=C, train_seed=T
Slot 1: conn_R2=A, rollout_r=B, cluster_acc=C, train_seed=T
...
Slot 9: conn_R2=A, rollout_r=B, cluster_acc=C, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
W matrix: [visual comment from connectivity heatmap]
Verdict: [supported/falsified/inconclusive]
Next: parent=P
```

### Step 4: Acknowledge User Input

### Step 5: Formulate Next Hypothesis + Edit 10 Config Files

## Block Boundaries

1. Update "Paper Summary"
2. Summarise block findings
3. Update "Established Principles"
4. Clear "Current Block"
5. Carry forward best config
6. **Write winner YAML** (see above)

## File Structure

You maintain THREE files:

1. **Full Log (append-only)**: `drosophila_cx_pi_voltage_noise_05_Claude_analysis.md`
   - Append every iteration's log entry (10 entries per batch)
   - Never read — human record only

2. **Working Memory (read + update every batch)**: `drosophila_cx_pi_voltage_noise_05_Claude_memory.md`
   - Read at start, update at end
   - Contains: robustness comparison table, hypotheses, established principles, current block iterations

3. **User Input (read every batch, acknowledge pending items)**: `user_input.md`
   - Read at every batch
   - If "Pending Instructions" has content: act on it, then move to "Acknowledged" section

## Knowledge Base Guidelines

### Established Principles

Must satisfy ALL of:

- Observed consistently across 3+ iterations
- Consistent across all 10 seeds (low CV, not just mean)
- States a causal relationship

Example: "Lowering lr_W to 5e-6 lifts conn_R2 from 0.30 to 0.45 robustly (3/3 iterations, CV < 5%)."

### Open Questions

- Patterns observed 1–2 times
- Seed-dependent effects
- Contradictions between iterations
- Theoretical predictions not yet verified

### Falsified Hypotheses

- State the original hypothesis
- State the contradicting evidence (iteration number, metrics)
- State what was learned
- Propose a revised hypothesis if applicable

## Start Call

When prompt says `PARALLEL START`:

- Block 1 is an **exploration** of learning rates. Slot 0 = parent baseline (lr_W=5e-5, lr=1e-3, lr_emb=1e-3); slots 1–9 each change ONE LR.
- Hypothesis: "Lowering lr_W and/or lr_embedding from the current defaults lifts conn_R2 above the 0.3 ceiling."

---

# Working Memory Structure

```markdown
# Working Memory: drosophila_cx_pi_voltage_noise_05

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean±std) | CV% | rollout_r | cluster_acc | Robust? | Hypothesis |
| ---- | -------------- | ------------------ | --- | --------- | ----------- | ------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest→newest. This section MUST appear before ## Current Block.**

### Block 1 Summary
[Summary of findings from block 1 — LR sweep]

### Block 2 Summary
[Summary of findings from block 2 — regul + batch_size]

### Block 3 Summary
[Summary of findings from block 3 — training volume + architecture]

### Block 4 Summary
[Summary of findings from block 4 — free exploration]

---

## Current Block

### Block Info

- **Block number**:
- **Focus**:
- **Iterations completed**: N of 50
- **Best so far this block**: conn_R2 = X.XXX (iter Y, slot Z)

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
