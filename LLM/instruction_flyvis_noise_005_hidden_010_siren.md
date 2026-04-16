# FlyVis GNN + Hidden-Neuron SIREN-T Exploration — flyvis_noise_005_hidden_010_siren

## Scientific Context

The research question: can we recover the full neural connectome even when 10% of non-retinal neurons are unobserved? Here we silence 10% of non-retinal neurons and jointly train a **SIREN(t)** that predicts their voltages from time alone.

Unlike the visual SIREN (which reconstructs an external stimulus), the hidden SIREN has **no direct supervision** — it is only trained through the GNN loss. The gradient flows: GNN loss → visible neuron residuals → GNN weights → messages from hidden neurons → hidden voltages → NNR_hidden SIREN. This is a purely indirect signal.

**SIREN vs NGP-T**: A parallel exploration (`flyvis_noise_005_hidden_010_ngp`) uses a MultiResTemporalGrid instead. SIREN is a sinusoidal network — globally entangled in time (waterbed problem). This forces `batch_size=1` and a very low `lr_NNR_f` (~1e-6). The scientific question: can SIREN learn hidden voltages from indirect gradients at all, and how does it compare to NGP-T?

**Waterbed problem**: Fitting SIREN on a few frames changes all other frames. This is why `batch_size=1` is required — larger batches create conflicting gradients across frames and destabilize SIREN. This is NOT a problem for NGP-T (local grid).

## Goal

Jointly optimize a **GNN** and a **SIREN(t) implicit neural representation** for the **Drosophila visual system** with noise level 0.05, **10% of non-retinal neurons hidden**.

**Dual objectives:**

1. **Connectivity recovery**: connectivity_R2 computed on **all edges** (including those touching hidden neurons) > 0.9
2. **Hidden voltage reconstruction**: hidden_nnr_R2 > 0.3 (per-neuron R² of SIREN-predicted hidden voltages vs GT)

## Model Architecture

**NNR_hidden (SIREN(t))**: `t → (n_hidden,)` — a sinusoidal network mapping time step t to the full vector of hidden neuron voltages simultaneously.

| Parameter                | Value     | Description                                               |
| ------------------------ | --------- | --------------------------------------------------------- |
| `inr_type_hidden`        | `siren_t` | SIREN(t) → (n_hidden,)                                    |
| `hidden_neuron_fraction` | 0.10      | 10% of non-retinal neurons hidden (~1200 neurons)         |
| `hidden_dim_nnr_hidden`  | 2048      | SIREN hidden dimension                                    |
| `n_layers_nnr_hidden`    | 4         | SIREN hidden layers (4L eliminates catastrophic failures) |
| `omega_hidden`           | 4096.0    | SIREN frequency parameter                                 |
| `nnr_hidden_T_period`    | 64000.0   | Time normalisation — full sequence length                 |

SIREN output size = n_hidden ≈ 1200 neurons. Total NNR_hidden parameters: ~4L×2048² + 2048×1200 ≈ 18M.

**Time normalization**: t is normalized to [0, 2π] as `t / T_period` (radians).

**Waterbed problem**: SIREN is globally entangled. **`batch_size=1` is mandatory.** Do NOT increase batch_size — larger batches create conflicting gradients and will destroy SIREN learning.

**SIREN LR**: `lr_NNR_f=1e-6` baseline. The indirect gradient pathway is very weak. The viable range is unknown for hidden SIREN but expected to be narrow (~1 order of magnitude), similar to visual SIREN (7e-9 to 2.5e-8 for direct supervision). The hidden case may need higher LR due to weaker gradients.

## Training Scheme

`alternate_training: true` with `alternate_lr_ratio: 0.05`. GNN converges fast (epoch 0 peak); reducing GNN LRs by 20x in epochs 1-2 lets the SIREN refine without GNN destabilizing it.

The hidden SIREN LR (`lr_NNR_f`) is **not affected** by alternate training.

| Epoch | GNN LRs                               | SIREN LR               | Purpose                       |
| ----- | ------------------------------------- | ---------------------- | ----------------------------- |
| 0     | Full: lr_W=1e-4, lr=1e-3, lr_emb=1e-3 | `lr_NNR_f`             | Joint warmup                  |
| 1-2   | Reduced 20x                           | `lr_NNR_f` (unchanged) | GNN stabilizes, SIREN refines |

**Regularization annealing** (`regul_annealing_rate=0`): disabled by default. All coefficients at full strength from epoch 0.

## Key Difference from NGP-T Exploration

| Aspect              | SIREN-T (this exploration)           | NGP-T (parallel exploration)       |
| ------------------- | ------------------------------------ | ---------------------------------- |
| `inr_type_hidden`   | `siren_t`                            | `ngp_t`                            |
| Architecture        | Sinusoidal MLP, globally entangled   | Multi-res 1-D grid, locally sparse |
| Waterbed problem    | **Yes** — bs=1 mandatory             | No — bs=16 safe                    |
| `lr_NNR_f` baseline | 1e-6 (indirect, narrow viable range) | 1e-3 (local grid, wide range)      |
| Parameters          | ~18M (SIREN)                         | ~2.4M (grid + MLP)                 |
| Time normalization  | [0, 2π] via T_period                 | [0, 1] via n_frames                |
| Primary LR sweep    | {1e-7, 1e-6, 1e-5, 1e-4}             | {1e-4, 1e-3, 5e-3, 1e-2}           |

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

| Parameter                               | Value                                |
| --------------------------------------- | ------------------------------------ |
| `hidden_dim` / `n_layers`               | 80 / 3 (g_phi)                       |
| `hidden_dim_update` / `n_layers_update` | 80 / 3 (f_theta)                     |
| `embedding_dim`                         | 2                                    |
| `input_size`                            | 3 (= 1 + embedding_dim)              |
| `input_size_update`                     | 5 (= 3 + embedding_dim + 1 stimulus) |

**CRITICAL**: When changing `embedding_dim`, you MUST also update `input_size` and `input_size_update`.

## Optimizer Structure

| Group      | Parameters             | LR config key  |
| ---------- | ---------------------- | -------------- |
| W          | Connectivity matrix    | `lr_W`         |
| g_phi      | Edge message MLP       | `lr`           |
| f_theta    | Node update MLP        | `lr`           |
| embedding  | Neuron type embeddings | `lr_embedding` |
| NNR_hidden | Hidden SIREN           | `lr_NNR_f`     |

## Metrics

| Metric            | Description                                            | Target |
| ----------------- | ------------------------------------------------------ | ------ |
| `connectivity_R2` | R² of learned vs true W, all edges                     | > 0.9  |
| `hidden_nnr_R2`   | Per-neuron R² of SIREN-predicted hidden voltages vs GT | > 0.3  |
| `tau_R2`          | R² of learned vs true time constants                   | —      |
| `rollout_pearson` | Pearson r of rollout on visible neurons                | —      |

**Per-neuron R²**: global linear correction (a·pred+b) applied first, then per-neuron R² averaged. Purely measures temporal dynamics, not DC offsets. Early in training, deeply negative values (-10 to -50) are normal — the SIREN is outputting near-random values before the gradient builds up.

**Progress bar**: shows `nnr=X.XXX` alongside `conn=X.XXX` (color-coded).

## Regularization Parameters

| Config parameter          | Default | Notes                                        |
| ------------------------- | ------- | -------------------------------------------- |
| `coeff_g_phi_diff`        | 750     | Monotonicity — always active                 |
| `coeff_g_phi_norm`        | 1.0     | Norm penalty — always active                 |
| `coeff_g_phi_weight_L1`   | 0.5     |                                              |
| `coeff_f_theta_weight_L1` | 0.5     |                                              |
| `coeff_f_theta_weight_L2` | 0.001   |                                              |
| `coeff_W_L1`              | 5e-5    |                                              |
| `regul_annealing_rate`    | 0       | Disabled — all coefficients at full strength |

## CRITICAL: Data is PRE-GENERATED at startup

At startup, data is generated **once** for all 4 slots with **different random seeds**. The **hidden neuron IDs are fixed** at startup and reused across all iterations. Do NOT modify `simulation:` parameters.

Seed formula:

- `simulation.seed = 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

## CRITICAL: SIREN-specific constraints

1. **`batch_size: 1` always** — waterbed problem makes larger batches destructive
2. **`lr_NNR_f` is the primary axis** — the viable range is narrow and unknown; sweep carefully
3. **Do NOT modify SIREN architecture** (`hidden_dim_nnr_hidden`, `n_layers_nnr_hidden`, `omega_hidden`) until LR is established — 4L/2048 is already validated for visual SIREN
4. **SIREN may need many epochs** before hidden_nnr_R2 goes positive — do not abandon early if conn_R2 is good

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle.

**Evidence hierarchy:**

| Level            | Criterion                                     |
| ---------------- | --------------------------------------------- |
| **Established**  | Consistent across 3+ iterations AND 4/4 seeds |
| **Tentative**    | Observed 1-2 times or inconsistent            |
| **Contradicted** | Conflicting evidence                          |

## Parallel Mode — 4 Slots Per Batch

### Robustness Assessment

- **Excellent**: conn_R2 > 0.9 AND hidden_nnr_R2 > 0.3 (all 4 seeds) — **TARGET**
- **Good GNN**: conn_R2 > 0.9 but hidden_nnr_R2 < 0.3 — SIREN needs improvement
- **Good SIREN**: hidden_nnr_R2 > 0.3 but conn_R2 < 0.9 — GNN degraded
- **Partial**: mixed results
- **Failed**: conn_R2 < 0.8 OR hidden_nnr_R2 < -5 at end of epoch 0 — reject

## Iteration Log Format

```
## Iter N: [excellent/good_gnn/good_siren/partial/failed]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, lr_NNR_f=W, batch_size=B, n_epochs=C
Slot 0: conn_R2=A, hidden_nnr_R2=B, tau_R2=C, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, hidden_nnr_R2=B, tau_R2=C, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, hidden_nnr_R2=B, tau_R2=C, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, hidden_nnr_R2=B, tau_R2=C, sim_seed=S, train_seed=T
GNN stats: mean_conn_R2=X, std=Y, CV=Z%, min=W
SIREN stats: mean_hidden_nnr_R2=X, std=Y, min=W
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive] — [one line]
Next: parent=P
```

## Winner Config (COMPULSORY)

At every block boundary, save best config to `config/fly/flyvis_noise_005_hidden_010_siren_winner.yaml`.

## Block Partition (suggested)

| Block | Focus                        | Parameters                                                                                              |
| ----- | ---------------------------- | ------------------------------------------------------------------------------------------------------- |
| 1     | Baseline                     | Establish baseline — what conn_R2 and hidden_nnr_R2 with lr_NNR_f=1e-6, bs=1?                           |
| 2     | SIREN LR                     | `lr_NNR_f`: sweep {1e-7, 1e-6, 1e-5, 1e-4} — find viable range for indirect gradients                   |
| 3     | n_epochs                     | Does SIREN need more epochs to accumulate gradient? Try {3, 5}                                          |
| 4     | alternate_lr_ratio           | {0.01, 0.05, 0.1, 0.2} — does deeper GNN freeze help SIREN gradient quality?                            |
| 5     | SIREN architecture           | `hidden_dim_nnr_hidden` {512, 1024, 2048} — once LR is known                                            |
| 6     | Combined best                | Integrate findings                                                                                      |
| 9     | SIREN architecture extras    | `learnable_omega`, `outermost_linear=False`, `n_layers_nnr_hidden=6` — sweep uncharted SIREN parameters |
| 10    | Recurrent training           | `recurrent_training=True`, `time_step` ∈ {1, 2, 3} — with waterbed-aware DAL and LR adjustment          |
| 11    | Best-of combination          | Combine Block 9 winner × Block 10 winner × best GNN config from Block 8                                 |
| 12    | Multi-seed validation        | Run 4+ seeds on Block 11 winner — establish true mean±std and CV%                                       |
| 13    | Local search around champion | Fine-grained sweep of top-2 impactful parameters near Block 12 winner                                   |

**Block 1 priority**: Establish whether hidden_nnr_R2 stays deeply negative (< -5) at end of all epochs. If so, lr_NNR_f=1e-6 is too small. If it reaches near 0, it's in the viable range.

---

## Block 9: SIREN Architecture Extras

### Scientific Rationale

Blocks 1–8 swept `lr_NNR_f`, `n_epochs`, `alternate_lr_ratio`, `hidden_dim_nnr_hidden` (512/1024/2048), `n_layers_nnr_hidden` (2 vs 4), and `omega_hidden` (256–4096). Three SIREN-specific parameters have never been tested:

1. **`learnable_omega`**: Make `omega_0` a learnable `nn.Parameter` initialized to the current `omega_hidden` value. Instead of a fixed sinusoidal frequency, the SIREN adapts its frequency to minimize the indirect loss. This could allow the model to discover the optimal frequency for the hidden voltage dynamics without manual sweep. Risk: gradient through omega may be noisy with indirect supervision.

2. **`outermost_linear=False`**: Currently the final layer is a linear projection (output ∈ ℝ, unbounded). With `outermost_linear=False` the final layer also uses sinusoidal activation (output ∈ [-1, 1] bounded). This enforces bounded voltage predictions and may stabilize training, at the cost of limited output range.

3. **`n_layers_nnr_hidden=6`**: Currently 4 layers are optimal (2 layers caused severe failures). 6 layers gives more representational depth — may improve indirect gradient backpropagation through deeper intermediate representations. Risk: deeper network is slower and may suffer from vanishing gradients through the sinusoidal chain.

### Implementation Notes

- `learnable_omega=True` requires checking `Siren_Network.py` to confirm the parameter is exposed. If the keyword is different, use the correct YAML key.
- `outermost_linear=False`: set `outermost_linear_nnr_hidden: false` in YAML (or equivalent key).
- Keep `hidden_dim_nnr_hidden=512` (Block 5/6 winner), `omega_hidden=1024` (unless Block 8 finds omega=2048 superior).
- Keep `lr_NNR_f` at the best value from Block 7/8 for the chosen omega.

### Experimental Design

| Slot   | Change                    | Expected effect                                     |
| ------ | ------------------------- | --------------------------------------------------- |
| CTRL   | Current best from Block 8 | Anchor                                              |
| Slot 1 | `learnable_omega=True`    | Adaptive frequency — may converge to a better omega |
| Slot 2 | `outermost_linear=False`  | Bounded output — may stabilize SIREN voltages       |
| Slot 3 | `n_layers_nnr_hidden=6`   | Deeper SIREN — richer intermediate representations  |

### Parent Config

Best config from Block 8: omega winner × lr_NNR_f winner × hidden_dim=512 × n_layers=4 × n_epochs=3 × DAL=15 × alt=true × ratio=0.05.

### Expected Outcomes

- **Supported**: Any slot shows nnr improvement > 5 points or conn_R2 improvement vs CTRL, reproduced on 2nd seed.
- **Falsified**: All slots match CTRL within noise — confirms SIREN internal architecture is not the limiting factor under indirect supervision, and gradient quality is the only lever.

---

## Block 10: Recurrent Training for SIREN Gradient Enrichment

### Scientific Rationale

All Blocks 1–8 used `recurrent_training=False` (teacher-forced training). The indirect gradient sends only 1 GNN step of backpropagation to the SIREN. Recurrent training with `time_step=T` expands this to T steps:

```
loss at frames k+1 ... k+T  →  T GNN steps  →  SIREN at frames k ... k+T-1
```

This forces temporal consistency in the SIREN predictions.

### SIREN-Specific Waterbed Warning

**The SIREN waterbed problem makes recurrent training fundamentally different from NGP:**

With SIREN, every gradient update modifies ALL grid points globally (sinusoidal basis functions are non-local). With `time_step=T`, there are T SIREN gradient passes per training step, each globally updating the SIREN weights. This is equivalent to running at T× the effective lr_NNR_f.

**CRITICAL**: When enabling recurrent training with SIREN, you MUST reduce lr_NNR_f proportionally:

```
lr_NNR_f_recurrent = lr_NNR_f_nonrecurrent / time_step
```

- time_step=2: lr_NNR_f = 5e-5 (if baseline is 1e-4)
- time_step=3: lr_NNR_f = 3.3e-5 (if baseline is 1e-4)

Failure to reduce lr_NNR_f will likely push SIREN past the viable LR cliff → catastrophic nnr collapse (as seen in Block 7: lr=3e-4 gave -42.37, vs -20.44 at 1e-4).

### Training Time Budget for Recurrent SIREN

Reference: bs=1, n_epochs=3, DAL=15 → ~57 min (established, Blocks 5–8).

Wall time ≈ time_step × time(same DAL, non-recurrent). Reduce DAL to compensate:

| time_step | DAL | lr_NNR_f | Estimated time |
| --------- | --- | -------- | -------------- |
| 1         | 15  | 1e-4     | ~57 min (CTRL) |
| 2         | 8   | 5e-5     | ~57 min        |
| 3         | 5   | 3e-5     | ~57 min        |

**This is budget-feasible** (unlike NGP where time_step=3 required DAL=2 and cost 102 min).

### Experimental Design

| Slot   | recurrent | time_step | DAL | lr_NNR_f | Notes                                                               |
| ------ | --------- | --------- | --- | -------- | ------------------------------------------------------------------- |
| CTRL   | False     | 1         | 15  | 1e-4     | Non-recurrent baseline                                              |
| Slot 1 | True      | 1         | 15  | 1e-4     | Recurrent 1-step: does recurrent loss formulation alone change nnr? |
| Slot 2 | True      | 2         | 8   | 5e-5     | 2-step unroll + LR halved                                           |
| Slot 3 | True      | 3         | 5   | 3e-5     | 3-step unroll + LR ×3 reduced                                       |

If Slot 2 or 3 shows improvement: validate on 2nd seed before Block 11.
If rollout degrades catastrophically on any recurrent slot: the waterbed problem is amplifying — try lr_NNR_f further reduced.

### Expected Outcomes

- **Recurrent helps despite waterbed**: Temporal consistency pressure outweighs the amplified waterbed risk. nnr improves toward 0.
- **Waterbed catastrophe**: nnr collapses to < -100 with recurrent. Even with reduced lr, the T-step update frequency exceeds SIREN's tolerance.
- **Neutral**: Recurrent slots match CTRL ±10 points. DAL reduction (fewer samples/epoch) exactly cancels the benefit of more gradient steps.

### Parent Config

Use Block 9 winner or CTRL. If Block 9 found `learnable_omega` helpful, combine with recurrent (but be cautious: double novel change).

---

## Block 11: Best-of Combination

### Scientific Rationale

Combine the winning changes from Block 9 (SIREN architecture) and Block 10 (recurrent training).

### Experimental Design

| Slot   | Config                                   | Purpose                   |
| ------ | ---------------------------------------- | ------------------------- |
| CTRL   | Best single-axis winner from Blocks 9–10 | Anchor                    |
| Slot 1 | Block 9 winner only                      | Isolate architecture gain |
| Slot 2 | Block 10 winner only                     | Isolate recurrent gain    |
| Slot 3 | Block 9 + Block 10 winners combined      | Full combination          |

If neither Block 9 nor Block 10 produced nnr improvement: Block 11 = 4-seed validation of the best historical config (nnr=-19.06, iter 21 — noting this was a "good seed") to establish the true mean.

---

## Block 12: Multi-Seed Robustness Validation

### Scientific Rationale

The SIREN exploration shows extreme bimodal seed behavior: ~10% of seeds are "good" (dale_score > 0.74, conn_R2 > 0.4) and ~90% are "bad" (dale_score < 0.60, conn_R2 ≈ 0). Single-seed results are not interpretable. Block 12 runs the Block 11 winner on 4+ seeds to measure true mean±std.

**Decision rule**:

- If good-seed rate < 20%: the config is fragile — report mean as the true performance, not single best.
- If all 4 seeds are bad: the mean nnr (even at -19 to -25 range) is the best achievable.
- If any seed shows nnr > -10: focus future work on understanding the good-seed vs bad-seed divergence.

Run 4 slots with the Block 11 winner config, different seeds. No parameter changes.

---

## Block 13: Local Search Around Champion

### Scientific Rationale

After confirming the champion in Block 12, do a fine-grained sweep of the 2 most impactful parameters found across the full exploration. Candidates (LLM should choose based on evidence at the time):

- `omega_hidden`: try ±1 octave around winner (e.g. 768, 1024, 1536 if winner is 1024)
- `lr_NNR_f`: try ×0.5 and ×2 around champion value
- `time_step` ±1 if recurrent helped in Block 10
- `hidden_dim_nnr_hidden`: try 384 or 640 around winner of 512

### Experimental Design

| Slot   | Config                    | Purpose                 |
| ------ | ------------------------- | ----------------------- |
| CTRL   | Block 12 champion (exact) | Anchor                  |
| Slot 1 | Top param −20%            | Downward perturbation   |
| Slot 2 | Top param +20%            | Upward perturbation     |
| Slot 3 | 2nd param best level      | Cross-check second axis |

**Termination criterion**: If hidden_nnr_R2 remains ≤ -10 across Blocks 9–13 with best configs, the indirect SIREN gradient is fundamentally insufficient for hidden neuron inference from dynamics alone. The waterbed problem and global entanglement may make SIREN inherently unsuitable for this task — NGP-T with its local grid structure is the preferred architecture going forward.

## Training Time Budget

**Hard constraint: total training time ≤ 120 minutes per iteration** (3 epochs on A100).

With `batch_size=1`, DAL=25 → ~40 min/epoch → ~120 min total (3 epochs). DAL is the primary time knob. **Training time scales linearly with DAL** — always verify estimated time before proposing a config.

```
estimated_time = n_epochs × DAL × (reference_time_per_epoch_at_DAL1)
```

If testing `n_epochs=5`, reduce DAL to ~15 to stay within budget.

## Known Prior Results

**From visual SIREN exploration (40 iterations, direct supervision)**:

- SIREN LR cliff on BOTH sides: viable band 7e-9 to 2.5e-8 for direct visual supervision
- 4L/2048 SIREN eliminates catastrophic failures vs 2L/2048
- LR=1e-8 was safe; LR=3e-8 caused catastrophic failure

**Hidden SIREN is different**: indirect gradient pathway (~100x weaker than visual SIREN direct gradient). Expected viable LR band is higher — around 1e-6 to 1e-4. The LLM exploration previously tried lr=1e-8 with 0 epochs, got hidden_nnr_R2 ≈ -42 (SIREN did not learn). Baseline starts at lr=1e-6.

**From noise_005 GNN-only**: conn_R2=0.982 without hidden neurons. With 10% hidden + SIREN, connectivity will be lower.

## Sibling Exploration References

- **GNN-only**: `./log/Claude_exploration/LLM_flyvis_noise_005/flyvis_noise_005_Claude_memory.md`
- **NGP-T (parallel)**: `./log/Claude_exploration/LLM_flyvis_noise_005_hidden_010_ngp/flyvis_noise_005_hidden_010_ngp_Claude_memory.md`

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
- Baseline hypothesis: "The current config (lr_NNR_f=1e-6, bs=1, 3 epochs, alternate_training=true) achieves conn_R2 > 0.8 while hidden_nnr_R2 > -5 (SIREN receives some gradient from indirect pathway)"

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_hidden_010_siren

## Paper Summary (update at every block boundary)

- **Hidden-neuron problem**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Results Table

| Iter | Config summary   | conn_R2 (mean±std) | CV% | min | hidden_nnr_R2 (mean) | siren_min | time_min | Rating | Hypothesis |
| ---- | ---------------- | ------------------ | --- | --- | -------------------- | --------- | -------- | ------ | ---------- |
| 1    | baseline lr=1e-6 | ?                  | ?   | ?   | ?                    | ?         | ?        | ?      | baseline   |

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
