# Drosophila CX — Branch 0 voltage-recovery GNN (STAGE 1: goal-driven)

## Goal

Find a **converging training recipe** for the `drosophila_cx_voltage` GNN
(`NeuralGNN`) that recovers the connectome of the no-TV cv0 path-integration
teacher (`drosophila_cx_pi_epg_no_tv_cv0`) from a **voltage rollout** —
GT edge topology + **hard Dale sign-lock** (Eq 10) + g_φ monotonicity (Eq 11).

This is **Stage 1 of a two-stage protocol**: it is an *existence proof* — push
the primary metric to its ceiling and emit a winner recipe. (Stage 2, the
knowledge-driven causal map across all HP families, is a separate instruction
run from this winner.)

**Starting hypothesis**: "the wiring is already recovered (`W_structure_r ≈ 0.87`)
but **under-scaled** (`connectivity_R2 ≈ 0.24`, slope ≈ 0.13) due to the W↔g_φ
scale degeneracy. `g_phi_norm_target: unit` + `coeff_g_phi_norm` should close the
scale gap and raise `connectivity_R2` toward `W_structure_r`, without lowering
`W_structure_r` or breaking `rollout_pearson`."

## Metrics (ranked) — WHERE to read them

All metrics are in **`<exploration_dir>/r2_trajectory/iter_NNN.log`** (one file).
The CSV rows are the training-loop trajectory (`connectivity_r2`, `vrest_r2`,
`tau_r2` per iteration). The **post-hoc recovery metrics are injected as the
trailing `# post_hoc …` line**: `W_structure_r`, `W_zscored_R2`, `W_corrected_R2`,
`W_corrected_slope`, `rollout_pearson`, `clustering_accuracy`. Parse that line.

1. **`W_structure_r`** (PRIMARY) — scale-free Pearson r between corrected learned W
   and GT W (non-zero edges). "Did we recover the **wiring**." Baseline ≈ **0.87**.
2. **`W_corrected_R2`** (= the NSE connectivity_R2) + **`W_corrected_slope`** —
   scale-sensitive. Low (≈0.24, slope 0.13) only because W is under-scaled; the
   g_φ-norm knobs target raising it toward `W_structure_r`. Also **`W_zscored_R2`**.
3. **`rollout_pearson`** (GUARD) — must stay **≥ 0.99**. If it drops, the recipe broke
   (a recipe can have high `W_structure_r` but a broken rollout — reject it).
4. g_φ shape (sigmoid, ≈0.99) — sanity that g_φ matches the teacher's **sigmoid**.

(`connectivity_r2` in the CSV rows is the *training-loop* NSE W R² — same axis as
`W_corrected_R2`, useful for the per-iteration trajectory shape / failure modes.)

**Ignore**: `tau_R2` / `V_rest_R2` — the teacher's τ is constant (degenerate R², shows N/A);
V_rest is informational. Do NOT optimise them.

## Scientific Method (hypothesize → test → validate/falsify)

Change **EXACTLY ONE** parameter per slot vs the parent (causality rule). With
**8 slots**: slot 0 = parent/control (unchanged); slots 1–7 each change one
parameter. A falsified hypothesis is valuable information — log it.

**Evidence hierarchy**: Established = consistent across batches AND seeds; Tentative
= 1–2 observations; Contradicted = conflicting across seeds.

### CAUSALITY RULE (MANDATORY)
More than one parameter changed per slot ⇒ you cannot attribute the effect (fatal
design error). EXPLORATION: slot 0 control, slots 1–7 single-axis. ROBUSTNESS: all
8 slots identical (different seeds).

## Data / seeds (forced by the pipeline — DO NOT set in configs)

Data is generated **once per slot** at startup (voltage rollout of the cv0
teacher) and reused across iterations. `simulation.seed = 1000 + slot`,
`training.seed = iteration*1000 + slot + 500`. Simulation params
(`n_neurons=156`, `n_frames`, teacher path) are **fixed — do not touch**.

## Training budget — ACTIVELY TUNE DAL to ≈ 60 min/training

`training_time_target_min = 60`. **`data_augmentation_loop` (DAL) is your time
knob.** Calibration (measured): ~34 min/epoch at DAL=50 on **l4**; at
`n_epochs=4`, **DAL≈20 ⇒ ~60 min** (time ∝ DAL × n_epochs, ~linear). Each batch,
read `training_time_min` from the analysis log and adjust DAL for the next batch:
- `training_time_min < 50` → raise DAL (more data = better recovery, up to a point).
- `training_time_min > 70` → lower DAL.
Aim every training at ~60 min. DAL is otherwise a *free* knob (more augmentation
generally helps recovery) — fill the budget. (`n_epochs` is a secondary time knob;
prefer DAL. Recovery plateaus early here, so don't over-spend epochs.)

## The model (drosophila_cx_voltage = NeuralGNN)

```
dĥ_i/dt·τ_i = f_θ(ĥ_i, a_i) + Σ_j Ŵ_ij · g_φ(ĥ_j, a_j)²        (g_phi_positive=true)
```
- **156 neurons, 7 CX types** (EPG, EPGt, PEN_a, PEN_b, Delta7, PEG, ER6), ~10,263 **GT edges** (≈39% inhibitory).
- **Teacher firing-rate nonlinearity = sigmoid** ϕ(h)=1/(1+e⁻ʰ) (recorded in `ode_params.pt` `activation`). The learned `g_φ²` should match it (sigmoid-shaped, saturating).
- **Hard sign-lock (Eq 10)**: `Ŵ = |W|·sign_GT` — `lock_edge_signs_from_connectome: true`. **FIXED — do not change.**
- **g_φ monotonicity (Eq 11)**: `coeff_g_phi_diff` (∂g_φ/∂ĥ ≥ 0).
- **g_φ-norm scale lock**: `coeff_g_phi_norm` + `g_phi_norm_target` anchor `g_φ(2·xnorm)²` (`unit`→1, `xnorm`→2·xnorm, `auto`→trainer default). This is the lever against the under-scaling.

**FIXED — never change**: `signal_model_name`, `g_phi_positive`, `lock_edge_signs_from_connectome`, `aggr_type`, simulation params.
**Coupled**: if you change `embedding_dim`, also set `input_size = 1+embedding_dim` and `input_size_update = 3+embedding_dim` (emb 2 → 3/5).

## Training parameters

| Parameter | Default | Notes |
|---|---|---|
| `g_phi_norm_target` | unit | {auto, unit, xnorm} — **scale lever** |
| `coeff_g_phi_norm` | 10 | {0, 10, 30, 100} — strength of the scale anchor |
| `coeff_g_phi_diff` | 1500 | {500, 1000, 1500, 3000} — g_φ monotonicity |
| `lr_W` | (config) | {1e-4, 3e-4, 6e-4, 1e-3} |
| `lr_embedding`, `lr` | (config) | {5e-4, 1e-3, 3e-3} |
| `data_augmentation_loop` | 20 | TIME KNOB → ~60 min on l4 |
| `n_epochs` | 4 | secondary time knob |
| `batch_size` | 8 | {4, 8} |
| `coeff_W_L1`, `coeff_W_L2`, `coeff_W_sign` | 0 | regularisation (GT edges: L1 less needed) |
| `hidden_dim`, `embedding_dim`, `n_layers` | 64 / 2 / 3 | architecture |

> **YAML rule**: always wrap `description` values in double quotes.

## Block plan (72 iter, 3 blocks × 24, 8 slots × 3 batches)

| Block | Focus | Scan (one axis per slot) |
|---|---|---|
| **B1** (1–24) | **Scale lock + learning rate** — the two biggest levers | batch1: `g_phi_norm_target`/`coeff_g_phi_norm`; batch2: `lr_W`; batch3: `lr_embedding`/`lr` + consolidate the best |
| **B2** (25–48) | **g_φ shape + training volume + regularisation** | `coeff_g_phi_diff`, `data_augmentation_loop`, `n_epochs`, `coeff_W_L1`/`L2`/`W_sign` |
| **B3** (49–72) | **Architecture, ceiling-push, ROBUSTNESS** | `hidden_dim`, `embedding_dim`, `batch_size`; free moves to push `W_structure_r`; final batch = **ROBUSTNESS** (8 seeds of the winner) |

Plan one coherent hypothesis per block; the 3 batches give 3 rounds to refine.
**Stop early** if `W_structure_r` ceilings (no slot beats the best by >0.01 for a
full block) AND `connectivity_R2` slope is near 1 — emit the winner.

## Iteration workflow

```
## Iter N (block B): [exploration | robustness]
Node: id=N, parent=P
Hypothesis: "[quoted]"
Config: g_phi_norm_target=…, coeff_g_phi_norm=…, coeff_g_phi_diff=…, lr_W=…, lr_emb=…, DAL=…, n_epochs=…, bs=…
Slot 0–7: W_structure_r=…, connectivity_R2=…(slope=…), W_zscored_R2=…, rollout_pearson=…, g_phi_r2=…, train_min=…, sim_seed=…, train_seed=…
Seed stats (robustness batches): mean W_structure_r=…, std=…, CV=…%
Mutation: [param]: [old]→[new]
Verdict: [supported/falsified/inconclusive]
Next parent: P
```

## Winner config (COMPULSORY — at every block boundary)

Save to `config/drosophila_cx/drosophila_cx_obs_branch0_v1_winner.yaml`:
```yaml
# Winner: drosophila_cx_obs_branch0_v1_winner.yaml
# Source: iter_XXX_slot_YY (W_structure_r = X.XXX)
# Why: [1–2 sentences]
# Metrics: W_structure_r X.XXX | connectivity_R2 X.XXX (slope X.XX) | rollout_pearson X.XXX | g_phi_r2 X.XXX | robust_mean X.XXX±X.XXX (CV X.X%)
# Key diffs from baseline: [list]
```

## File structure (you maintain THREE files)

1. **Full log (append-only)**: `drosophila_cx_obs_branch0_v1_Claude_analysis.md` — every iteration entry; human record.
2. **Working memory (read+update each batch)**: `drosophila_cx_obs_branch0_v1_Claude_memory.md` — robustness table, principles, current block.
3. **User input (read each batch)**: `user_input.md` — act on pending items, then move to Acknowledged.

## Start call (PARALLEL START)

- Read the base config — it IS the baseline (slot 0). Do NOT change defaults in slot 0.
- Slots 1–7: each changes EXACTLY ONE B1 parameter.
- Hypothesis: "`g_phi_norm_target: unit` raises `connectivity_R2`/slope toward
  `W_structure_r` (≈0.87) while keeping `rollout_pearson ≥ 0.99`."

---

# Working Memory template

```markdown
# Working Memory: drosophila_cx_obs_branch0_v1 (Stage 1, goal-driven)

## Knowledge Base
### Robustness Comparison Table
| Iter | Config summary | W_structure_r (mean±std) | CV% | connectivity_R2 (slope) | W_zscored_R2 | rollout_pearson | g_phi_r2 | train_min | Robust? | Hypothesis |
| ---- | -------------- | ------------------------ | --- | ----------------------- | ------------ | --------------- | -------- | --------- | ------- | ---------- |

### Established Principles
### Falsified Hypotheses
### Open Questions

## Previous Block Summaries
### Block 1 Summary
### Block 2 Summary

## Current Block
### Block Info
### Current Hypothesis
### Iterations This Block
### Emerging Observations
```
