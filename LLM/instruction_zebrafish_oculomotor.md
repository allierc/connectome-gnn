# Connconstr Zebrafish Oculomotor — LLM Exploration

## Goal

Maximize **connectivity_R2** for the **zebrafish oculomotor integrator** (Beiran & Litwin-Kumar 2023, Figure 5g-i).

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned effective W and ground-truth effective W
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free data
3. **cluster_accuracy** (THIRD) — neuron type clustering accuracy from learned embeddings

Informational (not for optimization): onestep_pearson, f_theta_R2, g_phi_R2, spectral_radius_learned vs spectral_radius_true.

**NOTE**: tau_R2 and V_rest_R2 are not applicable (fixed tau=1, no resting potential).

## Scientific Method

Strict **hypothesize -> test -> validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction
2. **Design experiment**: Change ONE parameter at a time (at most two) to understand causality — IF YOU CHANGE MORE, YOU CANNOT ATTRIBUTE THE EFFECT
3. **Run training**: 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use metrics AND cross-seed variance
5. **Update understanding**: Revise hypotheses based on evidence

**CRITICAL**: You can only hypothesize. Only training results validate or falsify.

## Data Generation

Each slot re-generates data with a **different random seed**.
Seeds are **forced by the pipeline** — DO NOT modify them in config files.

- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

**DO NOT change `simulation:` parameters** except `noise_model_level` (Block 6) and seed (managed automatically).

## Zebrafish Oculomotor Integrator Model

```
dr/dt = (-r + W @ r + I(t) * v_in) / tau
```

- **609 neurons**, 6 cell types (_Int_, _DOs_, _Axl_, ABD_m, ABD_i, vSPNs), from Goldman lab connectome
- **LINEAR**: no activation function (identity g_phi)
- tau=1.0 (fixed), dt=0.001
- W scaled to spectral radius = 0.9
- Stimulus: 4-channel multi-direction input along eigenvectors of W
- 21,000 frames (3 pulse repeats x 7,000)
- g_phi should learn identity (slope=1), f_theta should learn f(v)=-v (slope=-1)
- Dynamics purely determined by W eigenstructure
- Some populations have zeroed connections (ABD, axial, vSPNs)

**Key challenge**: Linear ODE means W must be precisely recovered from linear dynamics alone — no nonlinearity to disambiguate.

## GNN Architecture

- **g_phi**: Edge message MLP. Maps (v_j, a_j) -> message. `g_phi_positive=false` (linear model needs negative pass-through).
- **f_theta**: Node update MLP. Maps (v_i, a_i, aggregated_msg, I_i) -> dv_i/dt.
- **Embedding a_i**: learnable per-neuron type vector.

**CRITICAL — coupled parameters**: When changing `embedding_dim`, you MUST also update:
- `input_size = 1 + embedding_dim`
- `input_size_update = 3 + embedding_dim`

Example: embedding_dim=2 -> input_size=3, input_size_update=5.

## Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `lr_W` | 1e-3 | Learning rate for connectivity W |
| `lr` | 1e-3 | Learning rate for g_phi and f_theta MLPs |
| `lr_embedding` | 1e-3 | Learning rate for neuron embeddings |
| `n_epochs` | 2 | Number of training epochs |
| `batch_size` | 2 | Batch size |
| `data_augmentation_loop` | 100 | Data augmentation multiplier |
| `w_init_mode` | zeros | W initialization: "zeros", "randn_scaled" |
| `coeff_g_phi_diff` | 1500 | Monotonicity penalty on g_phi |
| `coeff_f_theta_weight_L2` | 0.001 | L2 penalty on f_theta MLP weights |
| `coeff_f_theta_msg_diff` | 0 | Monotonicity of f_theta w.r.t. message input |
| `coeff_W_L1` | 0 | L1 sparsity on W |
| `coeff_W_L2` | 1e-5 | L2 penalty on W |
| `coeff_W_sign` | 0 | Dale's law penalty |
| `use_gt_edges` | true | If false, train on fully connected graph |
| `noise_model_level` | 0.0 | Observation noise std added to trajectories |

## Training Time Constraint

**Keep total training time <= 60 min per iteration.** Larger model (609 neurons, many edges). Training may be slower. When increasing n_epochs, halve data_augmentation_loop to stay within budget. Monitor training_time_min closely.

## Parallel Mode — 4 Slots Per Batch

All 4 slots run the **same config** (different seeds applied automatically).
Edit all 4 configs identically: `{name}_00.yaml` through `{name}_03.yaml`.

### Robustness Assessment

- **Robust**: all 4 slots connectivity_R2 > 0.7
- **Partially robust**: 2-3 slots > 0.7
- **Fragile**: 0-1 slots > 0.7

Compute mean, std, CV for connectivity_R2 across 4 slots every batch.

## Block Partition

The blocks below provide a **recommended exploration roadmap**. Follow the block focus as a guide but use your scientific judgment — if early results clearly suggest a detour or shortcut, adapt. The block boundaries are soft: you can revisit earlier axes or combine parameters across blocks when evidence supports it.

| Block | Focus | Parameters to scan | Ranges |
|---|---|---|---|
| 1 | **lr_W + W_L1** | `lr_W`, `coeff_W_L1` | lr_W: {1e-4, 3e-4, 6e-4, 1e-3}, W_L1: {0, 1e-6, 1e-5, 5e-5} |
| 2 | **W initialization** | `w_init_mode` | {zeros, randn, randn_scaled} — low-rank dynamics may favor randn |
| 3 | **Training volume** | `data_augmentation_loop`, `n_epochs` | DAL: {50, 100, 200}, n_epochs: {2, 4} (halve DAL when doubling epochs) |
| 4 | **GT edges comparison** | `use_gt_edges` | use_gt_edges: {true, false} — default is fully connected. One block to test if providing GT edges helps or hurts |
| 5 | **Regularization** | `coeff_W_L2`, `coeff_W_sign`, `coeff_g_phi_diff`, `coeff_f_theta_msg_diff` | W_L2: {5e-6, 1e-5, 2e-5}, W_sign: {0, 0.01, 0.05}, g_phi_diff: {500, 1000, 1500}, f_theta_msg_diff: {0, 10, 100} |
| 6 | **Architecture + noise** | `hidden_dim`, `embedding_dim`, `noise_model_level` | hidden_dim: {48, 64, 80}, embedding_dim: {2, 4} (update input_size accordingly), noise: {0, 0.05, 0.5} |
| 7 | **Free exploration I** | Any parameter | Consolidate best from blocks 1-6, test novel combinations, attempt to break R2 ceiling |
| 8 | **Free exploration II** | Any parameter | Continue ceiling-breaking attempts, confirm final robust config |

### Low-rank context

These biological connectomes produce **low-rank activity** (linear integrator, dynamics purely determined by W eigenstructure). From prior low-rank exploration (NeuralGraph, 100-1000 neurons):
- **W_L1 calibration is critical**: L1=1E-6 unlocks near-perfect dynamics recovery; L1=1E-5 gives good W but partial rollout. Too much L1 destroys the low-rank structure.
- **W initialization matters**: `randn` outperforms `zeros` for low-rank regimes (opposite of chaotic regime). Must be tested — Block 2.
- **Fully connected training is the default**: NeuralGraph trains on fully connected graphs (no ground-truth edges). The GNN must learn which edges are zero via L1 sparsity. Block 4 compares GT edges vs fully connected.

### What NOT to explore

- `lr_scheduler` (all schedules hurt W recovery)
- `coeff_g_phi_weight_L2` (catastrophic)
- `n_layers > 3` (fragile)
- `derivative_smoothing_window > 1` (destroys signal)

### Model-specific notes for Block 5

- **Linear model**: f_theta_msg_diff is physically well-motivated here — f_theta should be monotonically increasing in message (it learns -v + msg + stim, which IS monotonic in msg). Values up to 100 may help.
- g_phi_diff may be less important since g_phi learns identity — lower values (500) may suffice.
- W_sign: zebrafish connectome has mixed excitatory/inhibitory types but some populations are zeroed — gentle W_sign ({0.01, 0.05}) only.
- W_L1 sparsity: many zero'd populations mean true W is relatively sparse — L1 may help here.
- **Spectral radius**: the true W has spectral_radius=0.9. If learned spectral_radius diverges far from 0.9, W recovery fails. Monitor this diagnostic.

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

From `analysis.log`: connectivity_R2, rollout_pearson, cluster_accuracy, training_time_min.

### Step 3: Write Log Entries + Update Memory

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, DAL=D, n_epochs=E, W_L2=A, hidden_dim=H, batch_size=B
Slot 0: conn_R2=A, rollout_pearson=B, cluster_acc=C, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, rollout_pearson=B, cluster_acc=C, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, rollout_pearson=B, cluster_acc=C, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, rollout_pearson=B, cluster_acc=C, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive]
Next: parent=P
```

### Step 4: Acknowledge User Input

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Block Boundaries

1. Update "Paper Summary"
2. Summarize block findings
3. Update "Established Principles"
4. Clear "Current Block"
5. Carry forward best config

## Start Call

When prompt says `PARALLEL START`:
- Read base config
- Set all 4 configs identically to baseline
- First iteration = baseline (no changes)
- Hypothesis: "The baseline config achieves connectivity_R2 > 0.5 robustly across seeds"

---

# Working Memory Structure

```markdown
# Working Memory: zebrafish_oculomotor

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean+-std) | CV% | rollout_r | cluster_acc | Robust? | Hypothesis |
| ---- | -------------- | ------------------- | --- | --------- | ----------- | ------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

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
