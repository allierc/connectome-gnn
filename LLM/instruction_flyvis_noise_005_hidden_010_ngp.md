# FlyVis GNN + Hidden-Neuron NGP-T Exploration — flyvis_noise_005_hidden_010_ngp

## Scientific Context

The research question: can we recover the full neural connectome even when 10% of non-retinal neurons are unobserved? Standard GNN training uses all neuron voltages as input; here, we silence 10% of non-retinal neurons and jointly train a **MultiResTemporalGrid (NGP-T)** that predicts their voltages from time alone. This tests whether the GNN can learn connectivity from incomplete observations and whether the NGP-T can infer the missing dynamics from the network's implicit constraints.

Unlike the visual SIREN (which reconstructs an external stimulus field from spatial coordinates), the hidden NGP-T has **no direct supervision** — it is only trained through the GNN loss. The gradient flows: GNN loss → visible neuron residuals → GNN weights → messages from hidden neurons → hidden voltages → NNR_hidden NGP-T. This is a purely indirect signal.

**Why NGP-T instead of SIREN**: SIREN suffers from the "waterbed problem" — fitting a few frames destroys all others. The multi-resolution grid avoids this: each time step touches only 2 grid cells per level (local update), so gradient steps don't interfere across time. This allows `batch_size=16` (vs bs=1 required for SIREN) and `lr_NNR_f=1e-3` (vs ~1e-8 for SIREN).

## Goal

Jointly optimize a **GNN** and a **MultiResTemporalGrid (NGP-T) implicit neural representation** for the **Drosophila visual system** with noise level 0.05, **10% of non-retinal neurons hidden**.

The GNN learns the neural connectivity (W, g_phi, f_theta) while the NGP-T reconstructs the voltage time series of hidden neurons. The GNN uses NGP-predicted voltages as messages from hidden neurons, so the GNN loss backpropagates through the NGP-T.

**Dual objectives:**

1. **Connectivity recovery**: connectivity_R2 computed on **all edges** (including those touching hidden neurons) > 0.9
2. **Hidden voltage reconstruction**: hidden_nnr_R2 > 0.3 (per-neuron R² of NGP-predicted hidden voltages vs GT; metric key kept as `hidden_nnr_R2` for log compatibility)

**Key difference from visual INR**: The connectivity R2 is now computed on ALL edges (no masking), because the NGP-T provides a signal for hidden-neuron edges. With zero-silencing, those edges are masked; with NGP-T active, they are meaningful.

## Model Architecture

**NNR_hidden (NGP-T)**: `t ∈ [0,1] → (n_hidden,)` — a multi-resolution 1-D feature grid with a trailing MLP, mapping a normalized time coordinate to the full vector of hidden neuron voltages.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `inr_type_hidden` | `ngp_t` | MultiResTemporalGrid(t) → (n_hidden,) |
| `hidden_neuron_fraction` | 0.10 | 10% of non-retinal neurons hidden (~1200 neurons) |
| `ngp_hidden_n_levels` | 24 | Number of grid resolution levels |
| `ngp_hidden_n_features_per_level` | 4 | Feature vector size per grid cell |
| `ngp_hidden_base_resolution` | 16 | Coarsest grid resolution |
| `ngp_hidden_per_level_scale` | 1.4 | Resolution multiplier per level |
| `ngp_hidden_mlp_width` | 512 | MLP hidden width after grid encoding |
| `ngp_hidden_mlp_layers` | 4 | MLP hidden layers after grid encoding |

Grid encoding output dim = 24 × 4 = 96 features → MLP (512×4) → ~1200 hidden neuron voltages.  
Total NNR_hidden parameters: ~2.4M.

**Time normalization**: t is normalized to [0, 1] using `t / n_frames` (not radians like SIREN).

**No waterbed problem**: each forward pass for frame k reads only 2 cells per level (linear interpolation neighbors). Gradient from frame k only updates those 2 cells — adjacent frames are unaffected. Allows `batch_size=16`.

**NGP-T LR**: `lr_NNR_f=1e-3` — much higher than SIREN (which needed ~1e-8). The grid is local; large LR steps don't destabilize distant frames.

## Training Scheme

`alternate_training: true` with `alternate_lr_ratio: 0.05` — same as visual INR. The GNN converges fast (epoch 0 peak) and without LR reduction in epochs 1-2 it overfits and destroys the connectivity it learned. Reducing GNN LRs by 20x freezes the GNN while the hidden NGP-T continues refining.

The hidden NGP-T LR (`lr_NNR_f`) is **not affected** by alternate training — it stays constant across all epochs.

| Epoch | GNN LRs | NGP-T LR | Purpose |
|-------|---------|----------|---------|
| 0 | Full: lr_W=1e-4, lr=1e-3, lr_emb=1e-3 | `lr_NNR_f=1e-3` | Joint warmup — GNN learns connectivity, NGP-T starts learning hidden voltages |
| 1-2 | Reduced 20x: lr_W=5e-6, lr=5e-5, lr_emb=5e-5 | `lr_NNR_f=1e-3` (unchanged) | GNN stabilizes, hidden NGP-T refines on reduced gradient noise |

**Regularization annealing** (`regul_annealing_rate=0.5`) ramps up the weight penalty coefficients from epoch 0 to 2. Only the annealed coefficients are affected — the structural penalties (`coeff_g_phi_diff`, `coeff_g_phi_norm`) are always at full strength:

| Coefficient | Annealed? |
|-------------|-----------|
| `coeff_W_L1` | **Yes** — L1 sparsity on W |
| `coeff_g_phi_weight_L1` | **Yes** — L1 on g_phi MLP weights |
| `coeff_f_theta_weight_L1` | **Yes** — L1 on f_theta MLP weights |
| `coeff_f_theta_weight_L2` | **Yes** — L2 on f_theta MLP weights |
| `coeff_g_phi_diff` | No — monotonicity penalty, always active |
| `coeff_g_phi_norm` | No — norm penalty, always active |

| Epoch | Multiplier | Effect on annealed coefficients |
|-------|-----------|--------------------------------|
| 0 | 0.00 | No weight regularization — free learning |
| 1 | 0.39 | ~39% strength |
| 2 | 0.63 | ~63% strength |

## Key Difference from Visual INR Exploration

| Aspect | Visual SIREN | Hidden NGP-T |
|--------|-------------|--------------|
| INR type | SIREN (t, x, y) | MultiResTemporalGrid (t only) |
| INR output | 1 scalar (stimulus) | ~1200 voltages simultaneously |
| INR supervision | Indirect (through GNN loss) | Indirect only (no direct GT loss) |
| Waterbed problem | Yes — needs bs=1, lr~1e-8 | No — bs=16, lr=1e-3 safe |
| Connectivity R2 mask | Edges NOT touching retina only | ALL edges (NGP fills hidden gaps) |
| Alternate training | Critical (GNN converges fast) | Also used — same rationale applies |
| INR LR regime | Cliff: viable band ~7e-9 to 2.5e-8 | Grid-local: 1e-3 baseline, wider range |

**Effective rank of hidden voltages**: SVD of 1000 hidden neurons shows rank_90=1, rank_99=26 — the hidden voltage signal is low-rank. The NGP-T's 96-dim encoding far exceeds this. Capacity is NOT the bottleneck; gradient quality and LR are the primary axes.

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
| NNR_hidden | Hidden NGP-T grid + MLP | `lr_NNR_f` |

## Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `connectivity_R2` | R² of learned vs true W, all edges | > 0.9 |
| `hidden_nnr_R2` | Per-neuron R² of NGP-predicted hidden voltages vs GT | > 0.3 |
| `tau_R2` | R² of learned vs true time constants | — |
| `V_rest_R2` | R² of learned vs true resting potentials | — |
| `rollout_pearson` | Pearson r of rollout on visible neurons | — |

**Primary NGP metric is `hidden_nnr_R2`** (log key kept for backward compatibility). It is a per-neuron R² — global linear correction (a·pred+b) applied first, then per-neuron R² averaged. This purely measures temporal dynamics, not DC offsets.

**Progress bar**: during training the bar shows `nnr=X.XXX` (color-coded green/yellow/orange/red) alongside `conn=X.XXX`.

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

At startup, data is generated **once** for all 4 slots with **different random seeds**. The **hidden neuron IDs are fixed** at startup and reused across all iterations (saved to `hidden_neuron_ids.pt`). This means the same ~1200 neurons are always hidden across all iterations within a run. Do NOT modify `simulation:` parameters.

Seed formula:
- `simulation.seed = 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle. Cannot predict outcomes — only training reveals the truth.

**CRITICAL**: With indirect NGP-T supervision, the `hidden_nnr_R2` may improve slowly. Do not abandon a direction after 1 iteration if the GNN connectivity is improving — they may decouple.

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

- **Excellent**: conn_R2 > 0.9 AND hidden_nnr_R2 > 0.3 (all 4 seeds) — **TARGET**
- **Good GNN**: conn_R2 > 0.9 but hidden_nnr_R2 < 0.3 — NGP-T needs improvement
- **Good NGP**: hidden_nnr_R2 > 0.3 but conn_R2 < 0.9 — GNN degraded by NGP
- **Partial**: mixed results
- **Failed**: conn_R2 < 0.8 OR hidden_nnr_R2 < 0.0 — reject

## Iteration Log Format

```
## Iter N: [excellent/good_gnn/good_ngp/partial/failed]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, lr_NNR_f=W, n_epochs=C, regul_annealing_rate=R
Slot 0: conn_R2=A, hidden_nnr_R2=B, tau_R2=C, V_rest_R2=D, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, hidden_nnr_R2=B, tau_R2=C, V_rest_R2=D, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, hidden_nnr_R2=B, tau_R2=C, V_rest_R2=D, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, hidden_nnr_R2=B, tau_R2=C, V_rest_R2=D, sim_seed=S, train_seed=T
GNN stats: mean_conn_R2=X, std=Y, CV=Z%, min=W
NGP stats: mean_hidden_nnr_R2=X, std=Y, min=W
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive] — [one line]
Next: parent=P
```

## Winner Config (COMPULSORY)

At every block boundary, save best config to `config/fly/flyvis_noise_005_hidden_010_ngp_winner.yaml` with header:

```yaml
# Winner config: flyvis_noise_005_hidden_010_ngp_winner.yaml
# Source: iter_XXX_slot_YY (connectivity_R2=X.XXX, hidden_nnr_R2=X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner: [narrative]
#
# Metrics:
#   connectivity_R2: X.XXX (best single seed)
#   robust_mean:     X.XXX +/- X.XXX (N seeds, CV=X.X%)
#   hidden_nnr_R2: X.XXX (per-neuron R², linear-corrected)
#   rollout_pearson: X.XXX
```

## Block Partition (suggested)

| Block | Focus | Parameters |
|-------|-------|-----------|
| 1 | Baseline | Establish baseline — what conn_R2 and hidden_nnr_R2 do we get with current NGP-T config (lr_NNR_f=1e-3, bs=16)? |
| 2 | NGP-T LR | `lr_NNR_f`: sweep {1e-4, 1e-3, 5e-3, 1e-2} — find optimal range for indirect gradient pathway |
| 3 | Grid capacity | `ngp_hidden_n_features_per_level` {2, 4, 8} × `ngp_hidden_n_levels` {16, 24} — rank_99=26 is the ceiling, does a smaller grid suffice? |
| 4 | batch_size vs DAL | Slots: (bs=8,DAL=50), (bs=16,DAL=25), (bs=32,DAL=13), (bs=64,DAL=7) — same wall time; larger batches give smoother gradients |
| 5 | alternate_training on/off | 2 slots with `alternate_training=true, ratio=0.05`, 2 slots with `alternate_training=false` — does GNN freeze help or hurt NGP-T? |
| 6 | alternate_lr_ratio | If block 5 favours alternate_training: sweep ratio {0.01, 0.05, 0.1, 0.2} |
| 7 | Combined best | Integrate findings from blocks 1-6 |
| 11 | NGP architecture | `ngp_hidden_n_levels` {16, 24, 32} and `ngp_hidden_mlp_width` {256, 512, 1024} — does architecture affect gradient quality under indirect supervision? |
| 12 | Recurrent training | `recurrent_training=True`, `time_step` ∈ {1, 2, 3} — does multi-step unrolling provide richer gradient signal to NGP-T? |
| 13 | Best-of combination | Combine Block 11 winner × Block 12 winner × best GNN config from Block 10 |
| 14 | Multi-seed validation | Run 4+ fresh seeds on the Block 13 winner — quantify true mean±std and CV% for nnr and conn_R2 |
| 15 | Local search around champion | Fine-grained sweep of top-2 impactful parameters near the Block 14 winner (e.g. ±20% lr_NNR_f, coeff_g_phi_diff, or time_step) |

**Block 4 note**: `data_augmentation_loop` (DAL) × `batch_size` ≈ constant keeps wall time fixed. The NGP-T is local, so larger batches are NOT required for stability (no waterbed), but they may give smoother indirect gradients.

**Block 1 priority**: Establish whether the NGP-T learns at all with `lr_NNR_f=1e-3`. Standalone training (train_ngp_voltage.py) achieved R²~0.31 on 1000 hidden neurons with direct supervision — indirect supervision in the joint loop will likely yield lower R².

---

## Block 11: NGP Architecture Sweep

### Scientific Rationale

All iterations in Blocks 1–10 used the default NGP-T architecture (n_levels=24, n_features=4, mlp_width=512, mlp_layers=4). Standalone tests with direct supervision showed `n_features_per_level=4 vs 16` makes no difference (rank_99=26, capacity not bottleneck). However, under **indirect supervision** the gradient landscape is fundamentally different — architecture may affect how well gradient signal propagates from the GNN loss back into the grid cells.

Hypotheses tested:
1. **Fewer levels (n_levels=16)**: 16×4=64-dim encoding, still above rank_99=26. Simpler grid → fewer parameters → potentially cleaner gradient accumulation per cell.
2. **More levels (n_levels=32)**: 32×4=128-dim encoding. Finer resolution may capture fast voltage dynamics that the 24-level grid cannot resolve.
3. **Wider MLP (mlp_width=1024)**: With indirect gradients the final MLP projection (grid_dim → 1200 neurons) may be the capacity bottleneck. More width allows more expressive mapping.

### Implementation Notes

- `ngp_hidden_n_levels` changes grid encoding output dim: output = n_levels × 4. MLP input size adjusts automatically.
- `ngp_hidden_mlp_width` changes only MLP hidden width — input and output dims are fixed.
- Keep `ngp_hidden_n_features_per_level=4` and `ngp_hidden_mlp_layers=4` fixed.
- Use `recurrent_training=False` to isolate architecture from recurrent effects.

### Experimental Design

| Slot | n_levels | mlp_width | Hypothesis |
|------|----------|-----------|-----------|
| CTRL | 24 | 512 | Baseline architecture (Blocks 1–10) |
| Slot 1 | 16 | 512 | Simpler grid: fewer cells, cleaner gradient per cell |
| Slot 2 | 32 | 512 | Finer resolution: captures fast dynamics |
| Slot 3 | 24 | 1024 | Wider MLP: more expressive final projection |

### Parent Config

Best config from Block 10. Keep: n_epochs=6, DAL=4, bs=16, alt=true, ratio=0.4, coeff_hv=3000, lr_NNR_f=1e-3. Use `recurrent_training=False`.

---

## Block 12: Recurrent Training for NGP-T Gradient Enrichment

### Scientific Rationale

With `recurrent_training=False` (all Blocks 1–10), only 1 GNN step separates NGP-T output from the training loss. With `recurrent_training=True` and `time_step=T`, the gradient path becomes:

```
loss at frames k+1 ... k+T  →  T GNN steps  →  NGP-T at frames k ... k+T-1
```

This forces **temporal consistency**: NGP predictions at k must be coherent enough for the GNN to propagate T steps and match GT at k+T. Implemented in `recurrent_step.py` lines 186–191 — at each unroll step the NGP is queried independently for that frame's hidden voltages.

### Training Time Warning

Each recurrent step performs T full GNN forward passes per batch item. Wall time ≈ time_step × non-recurrent time.

**Reference**: bs=16, DAL=4, n_epochs=6, time_step=1 → ~68 min (established iters 86–100).

| time_step | DAL required for ≤ 120 min | Estimated time |
|-----------|---------------------------|----------------|
| 1 | 4 | ~68 min |
| 2 | 2 | ~68 min |
| 3 | 2 | ~102 min |

**NEVER use time_step=3 with DAL=4** (→ ~200 min, exceeds budget).

### Experimental Design

| Slot | recurrent | time_step | DAL | Notes |
|------|-----------|-----------|-----|-------|
| CTRL | False | 1 | 4 | Non-recurrent baseline |
| Slot 1 | True | 1 | 4 | Recurrent 1-step: does recurrent loss formulation alone help? |
| Slot 2 | True | 2 | 2 | 2-step unroll, same wall time as CTRL |
| Slot 3 | True | 3 | 2 | 3-step unroll, primary test (~102 min) |

If Slot 3 wins: validate with 2nd seed before proceeding to Block 13.

### Parent Config

Use Block 11 CTRL config (or Block 11 winner if architecture improved nnr). Test recurrent independently before combining with architecture changes.

---

## Block 13: Best-of Combination

### Scientific Rationale

Combine the winning changes from Blocks 11 (architecture) and 12 (recurrent). The combination may be additive, synergistic, or interfering.

### Experimental Design

| Slot | Config | Purpose |
|------|--------|---------|
| CTRL | Best single-axis winner from Blocks 11–12 | Anchor |
| Slot 1 | Block 11 winner architecture only | Isolate architecture gain |
| Slot 2 | Block 12 winner recurrent only | Isolate recurrent gain |
| Slot 3 | Block 11 + Block 12 winners combined | Full combination |

If neither Block 11 nor Block 12 improved nnr: Block 13 = 4-seed validation of the best historical config (nnr=-4.44, iter 85) to establish its true mean and CV.

---

## Block 14: Multi-Seed Robustness Validation

### Scientific Rationale

The exploration history shows extreme seed variance: single seeds can show nnr=-4 to -70 at the same config. Block 14 runs the Block 13 winner on 4+ fresh seeds to measure true mean±std and CV%.

This is not an exploration block — it is a **validation block**. Run the same config 4× with distinct (sim_seed, train_seed) pairs. Report:
- mean hidden_nnr_R2 ± std, CV%
- mean conn_R2 ± std, CV%
- count of seeds with rollout_pearson > 0.75

**Decision rule**: If CV% < 30% and mean nnr > -20, the config is reproducible. If CV% > 60%, the best seed was lucky — report the mean as the true performance.

### Experimental Design

Run 4 slots with the Block 13 winner config, different seeds each. No parameter changes. Seed formula: iteration × 1000 + slot offset (standard).

---

## Block 15: Local Search Around Champion

### Scientific Rationale

After confirming the champion in Block 14, do a fine-grained sweep of the top-2 most impactful parameters found across the entire exploration. The goal is to push nnr toward 0 from the champion's position.

Candidate axes (LLM should choose based on evidence at the time):
- `lr_NNR_f`: try ×0.5, ×2 around champion value
- `coeff_g_phi_diff`: try ×0.5, ×2 (if diff=375 effect validated)
- `time_step` neighborhood (if recurrent helps): try time_step ±1
- `coeff_hidden_voltage` interaction with best architecture

### Experimental Design

| Slot | Config | Purpose |
|------|--------|---------|
| CTRL | Block 14 champion (exact) | Anchor |
| Slot 1 | Top param −20% | Downward perturbation |
| Slot 2 | Top param +20% | Upward perturbation |
| Slot 3 | 2nd param best level | Cross-check second axis |

**Termination criterion**: If nnr remains ≤ -4 across Blocks 11–15 with best configs, indirect gradient is fundamentally insufficient. The next research direction is auxiliary supervision (anchor neurons, partial hidden set, or pre-training phase with direct voltage supervision).

## Training Time Budget

**Hard constraint: total training time ≤ 120 minutes per iteration** (3 epochs on A100).

**Training time scales linearly with DAL**: `wall_time ≈ DAL × (time_per_DAL_unit)`. Do NOT exceed a config whose estimated wall time exceeds 120 minutes. Always compute the expected wall time before proposing a config change:

```
estimated_time = n_epochs × DAL × (reference_time_per_epoch_at_DAL1)
```

`data_augmentation_loop` (DAL) is the primary knob — it controls how many random frames are sampled per epoch. With `batch_size=16`, DAL=25 → ~40 min/epoch → ~120 min total (3 epochs). **Never exceed DAL=25 at bs=16 for 3 epochs** without reducing `n_epochs` proportionally.

When changing `batch_size`, adjust DAL to keep wall time constant:
- bs=8, DAL=50 → baseline wall time
- bs=16, DAL=25 → baseline wall time
- bs=32, DAL=13 → baseline wall time
- bs=64, DAL=7 → baseline wall time

If testing `n_epochs=1`, DAL can go up to 75 for the same wall time budget.

## Known Prior Results

**From standalone NGP-T training (train_ngp_voltage.py)**:
- With direct supervision on 1000 neurons, 500K steps, lr=1e-3: R²~0.31–0.41 (plateau region)
- Effective rank of hidden voltages: rank_90=1, rank_99=26 (data is low-rank, not capacity-limited)
- n_features_per_level=4 vs 16: no R² improvement → 4 features is sufficient for the data rank
- Grid is local: bs=16, lr=1e-3 are stable. No waterbed.

**From visual INR exploration (40 iterations)**:
- alternate_training critical for GNN stability; used here for same reason.
- 3-epoch training standard; epoch 0 peaks, epochs 1-2 refine.

**From noise_005 GNN-only champion**:
- Best GNN: conn_R2=0.982±0.003 with lr_W=9e-4, lr=1.8e-3, lr_emb=2.325e-3, aug=35, 1 epoch.
- With 10% hidden + NGP-T, connectivity will likely be lower — indirect NGP-T gradients add noise.

**Key open question**: Will the NGP-T learn useful voltages from indirect gradients alone? The standalone R²=0.31 is an upper bound (direct supervision). The joint training may yield lower values, especially early in training.

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
- Baseline hypothesis: "The current config (lr_NNR_f=1e-3, bs=16, 3 epochs, alternate_training=true, ratio=0.05) achieves conn_R2 > 0.8 while hidden_nnr_R2 > 0 (NGP-T learns something from indirect gradients)"

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_hidden_010

## Paper Summary (update at every block boundary)

- **Hidden-neuron problem**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Results Table

| Iter | Config summary | conn_R2 (mean±std) | CV% | min | hidden_nnr_R2 (mean) | ngp_min | time_min | Rating | Hypothesis |
| ---- | -------------- | ------------------ | --- | --- | ---------------------- | ------- | -------- | ------ | ---------- |
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
