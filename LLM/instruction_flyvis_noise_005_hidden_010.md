# FlyVis GNN + Hidden-Neuron SIREN Exploration — flyvis_noise_005_hidden_010

## Scientific Context

The research question: can we recover the full neural connectome even when 10% of non-retinal neurons are unobserved? Standard GNN training uses all neuron voltages as input; here, we silence 10% of non-retinal neurons and jointly train a SIREN(t) that predicts their voltages from time alone. This tests whether the GNN can learn connectivity from incomplete observations and whether the SIREN can infer the missing dynamics from the network's implicit constraints.

Unlike the visual SIREN (which reconstructs an external stimulus field from spatial coordinates), the hidden SIREN has **no direct supervision** — it is only trained through the GNN loss. The gradient flows: GNN loss → visible neuron residuals → GNN weights → messages from hidden neurons → hidden voltages → NNR_hidden SIREN. This is a purely indirect signal.

## Goal

Jointly optimize a **GNN** and a **SIREN(t) implicit neural representation** for the **Drosophila visual system** with noise level 0.05, **10% of non-retinal neurons hidden**.

The GNN learns the neural connectivity (W, g_phi, f_theta) while the SIREN(t) reconstructs the voltage time series of hidden neurons. The GNN uses SIREN-predicted voltages as messages from hidden neurons, so the GNN loss backpropagates through the SIREN.

**Dual objectives:**

1. **Connectivity recovery**: connectivity_R2 computed on **all edges** (including those touching hidden neurons) > 0.9
2. **Hidden voltage reconstruction**: hidden_siren_R2 > 0.5 (R² of SIREN-predicted hidden voltages vs GT)

**Key difference from visual INR**: The connectivity R2 is now computed on ALL edges (no masking), because the SIREN provides a signal for hidden-neuron edges. With zero-silencing, those edges are masked; with SIREN active, they are meaningful.

## Model Architecture

**NNR_hidden (SIREN(t))**: `t → (n_hidden,)` — a single SIREN network that maps time step t to the full vector of hidden neuron voltages simultaneously.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `inr_type_hidden` | `siren_t` | SIREN(t) → (n_hidden,) |
| `hidden_neuron_fraction` | 0.10 | 10% of non-retinal neurons hidden (1200 neurons) |
| `hidden_dim_nnr_hidden` | 2048 | Hidden dimension |
| `n_layers_nnr_hidden` | 4 | Hidden SIREN layers |
| `omega_hidden` | 4096.0 | Frequency parameter |
| `nnr_hidden_T_period` | 64000.0 | Time normalisation (full sequence) |

The SIREN has output size = n_hidden = ~1200 neurons. This is much larger than the visual SIREN output (1 scalar). Each output neuron gets its own independent time series learned by the same shared network.

**SIREN LR**: `lr_NNR_f` controls NNR_hidden (same parameter group as visual NNR_f). The indirect gradient signal means optimal LR may differ significantly from visual INR.

## Training Scheme

No `alternate_training` — set to `false`. The hidden SIREN has no independent signal in epoch 0; it needs the GNN to learn the dynamics first before it can refine hidden voltages. Alternate training would freeze the GNN when the SIREN needs it most.

Instead: single-phase joint training, `n_epochs=3`, SIREN LR kept constant throughout.

| Epoch | GNN LRs | SIREN LR | Purpose |
|-------|---------|----------|---------|
| 0-2 | Full throughout | `lr_NNR_f` (constant) | GNN + SIREN learn together |

**Regularization annealing** (`regul_annealing_rate=0.5`) ramps up from epoch 0 to 2:
| Epoch | Multiplier |
|-------|-----------|
| 0 | 0.00 |
| 1 | 0.39 |
| 2 | 0.63 |

## Key Difference from Visual INR Exploration

| Aspect | Visual INR | Hidden SIREN |
|--------|-----------|--------------|
| SIREN input | (x, y, t) | (t) only |
| SIREN output | 1 scalar (stimulus) | ~1200 voltages simultaneously |
| SIREN supervision | Indirect (through GNN loss) | Indirect only (no direct GT loss) |
| Connectivity R2 mask | Edges NOT touching retina only | ALL edges (SIREN fills hidden gaps) |
| Alternate training | Critical (GNN converges fast) | Not used (GNN convergence needed) |
| SIREN LR cliff | Sharp: viable band 1e-8 to 2.5e-8 | Unknown — first key exploration axis |

**Critical SIREN LR warning from visual INR**: The viable LR band was only 3.5x wide (7e-9 to 2.5e-8). Total collapse occurred at 3e-8. The hidden SIREN has a different gradient pathway and different output dimensionality — the safe range must be determined from scratch. Do NOT assume the same cliff positions.

## FlyVis Model

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i(t)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors, never hidden)
- 10% of remaining ~12,005 non-retinal neurons = ~1,200 hidden
- DAVIS visual input, noise_model_level=0.05
- 64,000 frames, delta_t=0.02

## GNN Architecture

Same as visual INR champion:

| Parameter | Value |
|-----------|-------|
| `hidden_dim` / `n_layers` | 80 / 3 (g_phi) |
| `hidden_dim_update` / `n_layers_update` | 80 / 3 (f_theta) |
| `embedding_dim` | 2 |
| `input_size` | 3 (= 1 + embedding_dim) |
| `input_size_update` | 5 (= 3 + embedding_dim + 1 stimulus) |

**CRITICAL — coupled parameters**: When changing `embedding_dim`, you MUST also update `input_size` and `input_size_update`.

## Optimizer Structure

| Group | Parameters | LR config key |
|-------|-----------|---------------|
| W | Connectivity matrix | `lr_W` |
| g_phi | Edge message MLP | `lr` |
| f_theta | Node update MLP | `lr` |
| embedding | Neuron type embeddings | `lr_embedding` |
| NNR_hidden | Hidden SIREN | `lr_NNR_f` |

## Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `connectivity_R2` | R² of learned vs true W, all edges | > 0.9 |
| `hidden_siren_R2` | R² of SIREN-predicted hidden voltages vs GT | > 0.5 |
| `tau_R2` | R² of learned vs true time constants | — |
| `V_rest_R2` | R² of learned vs true resting potentials | — |
| `rollout_pearson` | Pearson r of rollout on visible neurons | — |

**Primary SIREN metric is `hidden_siren_R2`** — analogous to `stimuli_R2` in the visual INR case. It is computed during test/rollout and written to the analysis log automatically.

## Regularization Parameters

| Config parameter | Default | Annealed? |
|-----------------|---------|-----------|
| `coeff_g_phi_diff` | 750 | No |
| `coeff_g_phi_norm` | 1.0 | No |
| `coeff_g_phi_weight_L1` | 0.5 | Yes |
| `coeff_f_theta_weight_L1` | 0.5 | Yes |
| `coeff_f_theta_weight_L2` | 0.001 | Yes |
| `coeff_W_L1` | 5e-5 | Yes |
| `regul_annealing_rate` | 0.5 | — |

## CRITICAL: Data is PRE-GENERATED at startup

At startup, data is generated **once** for all 4 slots with **different random seeds**. The **hidden neuron IDs are fixed** at startup and reused across all iterations (saved to `hidden_neuron_ids.pt`). This means the same 1200 neurons are always hidden across all iterations within a run. Do NOT modify `simulation:` parameters.

Seed formula:
- `simulation.seed = 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle. Cannot predict outcomes — only training reveals the truth.

**CRITICAL**: With indirect SIREN supervision, the hidden_siren_R2 may improve slowly. Do not abandon a direction after 1 iteration if the GNN connectivity is improving — they may decouple.

**Evidence hierarchy:**

| Level | Criterion |
|-------|-----------|
| **Established** | Consistent across 3+ iterations AND 4/4 seeds |
| **Tentative** | Observed 1-2 times or inconsistent |
| **Contradicted** | Conflicting evidence |

## Parallel Mode — 4 Slots Per Batch

4 slots, different seeds. All 4 configs identical (seeds set automatically).

### Robustness Assessment

Evaluate using **both** metrics:

- **Excellent**: conn_R2 > 0.9 AND hidden_siren_R2 > 0.5 (all 4 seeds) — **TARGET**
- **Good GNN**: conn_R2 > 0.9 but hidden_siren_R2 < 0.5 — SIREN needs improvement
- **Good SIREN**: hidden_siren_R2 > 0.5 but conn_R2 < 0.9 — GNN degraded by SIREN
- **Partial**: mixed results
- **Failed**: conn_R2 < 0.8 OR hidden_siren_R2 < 0.0 — reject

## Iteration Log Format

```
## Iter N: [excellent/good_gnn/good_siren/partial/failed]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, lr_NNR_f=W, n_epochs=C, regul_annealing_rate=R
Slot 0: conn_R2=A, hidden_siren_R2=B, tau_R2=C, V_rest_R2=D, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, hidden_siren_R2=B, tau_R2=C, V_rest_R2=D, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, hidden_siren_R2=B, tau_R2=C, V_rest_R2=D, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, hidden_siren_R2=B, tau_R2=C, V_rest_R2=D, sim_seed=S, train_seed=T
GNN stats: mean_conn_R2=X, std=Y, CV=Z%, min=W
SIREN stats: mean_hidden_siren_R2=X, std=Y, min=W
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive] — [one line]
Next: parent=P
```

## Winner Config (COMPULSORY)

At every block boundary, save best config to `config/fly/flyvis_noise_005_hidden_010_winner.yaml` with header:

```yaml
# Winner config: flyvis_noise_005_hidden_010_winner.yaml
# Source: iter_XXX_slot_YY (connectivity_R2=X.XXX, hidden_siren_R2=X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner: [narrative]
#
# Metrics:
#   connectivity_R2: X.XXX (best single seed)
#   robust_mean:     X.XXX +/- X.XXX (N seeds, CV=X.X%)
#   hidden_siren_R2: X.XXX
#   rollout_pearson: X.XXX
```

## Block Partition (suggested)

| Block | Focus | Parameters |
|-------|-------|-----------|
| 1 | Baseline | Establish baseline — what conn_R2 and hidden_siren_R2 do we get with current config? |
| 2 | SIREN LR | `lr_NNR_f`: sweep {1e-9, 1e-8, 1e-7, 1e-6} — find viable range (no prior data) |
| 3 | SIREN LR fine | Narrow sweep around block 2 winner — find cliff positions |
| 4 | n_epochs | Test 1 vs 3 vs 5 — how many epochs does SIREN need to converge? |
| 5 | GNN LRs | Test lower LRs (like INR baseline) vs current — does GNN LR affect SIREN learning? |
| 6 | SIREN architecture | `hidden_dim_nnr_hidden` {512, 1024, 2048, 4096} — output is 1200 neurons, small enough? |
| 7 | Regularization | `regul_annealing_rate`, `coeff_W_L1` — does L1 on W hurt hidden-edge learning? |
| 8 | Combined best | Integrate findings |

**Block 1 priority**: Establish whether the SIREN learns at all with `lr_NNR_f=1e-8`. If `hidden_siren_R2 ≈ 0`, the LR is too small and the gradient is too indirect — block 2 should then test much higher LRs (1e-6, 1e-5).

## Known Prior Results

**From visual INR exploration (40 iterations)**:
- SIREN LR cliff on BOTH sides: viable band 7e-9 to 2.5e-8. Only 1e-8 confirmed safe.
- 4L SIREN eliminates catastrophic failures. 4L chosen for hidden SIREN as well.
- 2048 hidden dim is Pareto-optimal (same quality, 43% faster than 4096).
- alternate_training critical for visual SIREN; NOT used for hidden SIREN.

**From noise_005 GNN-only champion**:
- Best GNN: conn_R2=0.982±0.003 with lr_W=9e-4, lr=1.8e-3, lr_emb=2.325e-3, aug=35, 1 epoch.
- With 10% hidden + SIREN, connectivity will likely be lower — indirect SIREN gradients add noise.

**Key open question**: Will the hidden SIREN actually learn useful voltages from indirect gradients alone, or will hidden_siren_R2 remain near 0 regardless of LR?

## Sibling Exploration References

- **GNN-only**: `./log/Claude_exploration/LLM_flyvis_noise_005/flyvis_noise_005_Claude_memory.md`
- **Visual INR**: `./log/Claude_exploration/LLM_flyvis_noise_005_INR/flyvis_noise_005_INR_Claude_memory.md`

## File Structure

You maintain **THREE** files:

1. **Full Log** (append-only): `{llm_task_name}_analysis.md`
2. **Working Memory** (read + update every batch): `{llm_task_name}_memory.md`
3. **User Input** (read every batch): `user_input.md`

> **YAML rule**: Always wrap the `description` field value in double quotes — colons inside unquoted YAML strings cause parse errors.

## Start Call

When prompt says `PARALLEL START`:

- Read base config to understand training regime
- Set all 4 configs identically to baseline
- Write planned config and initial hypothesis to working memory
- First iteration establishes baseline — do NOT change hyperparameters yet
- Baseline hypothesis: "The current config (lr_NNR_f=1e-8, 3 epochs, no alternate training) achieves conn_R2 > 0.8 while hidden_siren_R2 > 0 (SIREN learns something from indirect gradients)"

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_hidden_010

## Paper Summary (update at every block boundary)

- **Hidden-neuron problem**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Results Table

| Iter | Config summary | conn_R2 (mean±std) | CV% | min | hidden_siren_R2 (mean) | siren_min | time_min | Rating | Hypothesis |
| ---- | -------------- | ------------------ | --- | --- | ---------------------- | --------- | -------- | ------ | ---------- |
| 1 | baseline | ? | ? | ? | ? | ? | ? | ? | baseline |

### Established Principles

[Confirmed patterns — require 3+ supporting iterations AND cross-seed consistency]

### Falsified Hypotheses

[Keep as record]

### Open Questions

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest→newest.**

---

## Current Block

### Block Info

### Current Hypothesis

**Hypothesis**: [specific, testable prediction]
**Rationale**: [why you believe this]
**Test**: [what config change tests this]
**Expected outcome**: [what would support vs falsify]
**Status**: untested / supported / falsified / revised

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
