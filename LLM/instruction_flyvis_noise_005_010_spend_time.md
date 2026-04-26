# FlyVis GNN — SPEND Add-on #1 (time-permutation N2N)

## Goal

Use SPEND's **even/odd time-permutation** Noise2Noise denoising — the direct
analog of `Img_Split_Conc` along the time axis — to break the
**conn_R2 = 0.745 ceiling** under dual noise (σ=0.05, γ=0.10).

Within each minibatch window of `spend_time_window` consecutive frames, the
even-frame trace is the smoother input and the (linearly-interpolated)
odd-frame trace is the N2N target. Cheaper than replay (no extra noise
tensor); works directly on the observed `v + noise` so no clean-voltage
loading required.

Primary metric: **connectivity_R2**. Target: **conn_R2 > 0.80**.

Cite: https://github.com/buchenglab/SPEND  (Ding et al. 2025, Newton 1, 100195;
`datasplit_with_aug_choose.py:75-97` — `Img_Split_Conc`).
Bottleneck analysis: `docs/measurement_noise_bottleneck.pdf` §7.1, §7.4.

## Scientific Context: Why time-permutation breaks the ceiling

Same Noise2Noise rationale as Add-on #3 — but the two views come from
**stride-2 splits along the time axis** of the *single* observed trace
(no second simulation, no synthesised noise). Linear half-frame
interpolation of the odd view is justified by `sim.delta_t = 0.02` (20 ms),
small relative to the dynamics correlation time. Lehtinen's N2N theorem
applies: independent per-frame Gaussian noise satisfies the independence
hypothesis; the smoother converges to E[v_clean | v_obs].

Trade-off vs replay: signal is **not** identical between the two views (a
half-frame shift exists), so this puts a small upper bound on the achievable
SNR gain. Empirically the gap to replay should be 5–15% — measure it.

## Training Mode

`data_train_spend(config, ...)`. Invoked via
`python GNN_Main.py -o train_SPEND <config>` or by the agentic pipeline.

**Training from scratch**, **1 epoch**, ~60 min/iteration on a100.

## Data

`generate_data: false`. All slots reuse the existing `fly/flyvis_noise_005_010`
dataset. The trainer loads it normally (with disk-saved noise) — no
clean-voltage override. **DO NOT** modify simulation parameters.

Robustness across seeds: 4 slots either run different configs (default) or
the same config with different `training.seed` for validation blocks.

## Noise Model

Identical to `flyvis_noise_005_010` (σ_dyn=0.05, γ_meas=0.10). The trainer
uses the disk-saved `noise` field to construct `v_obs = v + noise`.

## FlyVis Model

13,741 neurons, 65 cell types, 434,112 edges. **Do not change** architecture.

## Explorable Parameters

### SPEND time-permute knobs (PRIMARY)

| Parameter                | Default | Safe range       | Notes |
|--------------------------|---------|------------------|-------|
| `coeff_spend_time`       | 1.0     | 0.1–10.0 (log)   | Weight on the time-permute N2N loss |
| `spend_time_window`      | 16      | 8, 16, 32, 64    | Frames per N2N window |
| `spend_smoother_hidden`  | 32      | 16, 32, 64       | 1D-conv smoother width |
| `spend_smoother_lr`      | 1e-3    | 1e-4 – 1e-2      | Smoother param-group LR |

Locked: `spend_load_clean: false`; `coeff_spend_replay: 0`;
`coeff_spend_typed: 0`.

**Window-vs-Δt note.** Larger windows give more time-domain averaging
(better SNR) but the linear interp assumption degrades if the dynamics
varies non-linearly within the window. With `Δt = 20 ms`, windows up to
~40 frames (800 ms) should still satisfy linearity for slow neurons; fast
neurons (e.g. T4/T5) may need shorter windows. Sweep: window=8 (fast-cell
safe) vs window=32 (slow-cell SNR).

### Standard GNN levers (compatible)

Same table as the replay instruction file (coeff_g_phi_diff, coeff_W_L1,
lr_W, lr, data_augmentation_loop, batch_size).

**Locked / do not touch:** `recurrent_training` (false), `n_epochs` (1),
`pretrained_model` (empty), simulation params, architecture.

## Slot Strategy — 4 Different Configs Per Batch

Each batch tests **4 distinct configs**. Only at robustness-validation
blocks do all 4 share a config and vary `training.seed`.

### Config Files

- Edit all 4 config files: `flyvis_noise_005_010_spend_time_00.yaml` through
  `flyvis_noise_005_010_spend_time_03.yaml`.
- **DO NOT change**: `simulation:` parameters, dataset, architecture.

## Evaluation

Metrics from `analysis.log`:
- `connectivity_R2` (PRIMARY), `tau_R2`, `V_rest_R2`, `cluster_accuracy`,
  `rollout_pearson_r`, `training_time_min`.

Per-iteration metrics from `tmp_training/metrics.log` (standard 6-column
schema, same as baseline): `iteration, connectivity_r2, vrest_r2, tau_r2,
hidden_nnr_pearson, anchor_nnr_pearson`.

**SPEND-specific** per-iter from `tmp_training/spend_components.log`:
`iteration, loss_main, loss_replay, loss_time, loss_typed`. The HPO agent
MUST read this file.

Progress-bar abbreviations (20-iter EMAs of per-batch losses):

| Bar label | File column | Meaning |
|-----------|-------------|---------|
| `conn=`   | connectivity_r2 | R² of learned W |
| `Vr=`     | vrest_r2 | R² of V_rest |
| `tau=`    | tau_r2 | R² of τ |
| `tim=`    | loss_time | **Add-on #1** N2N MSE: ‖smoother(even) − odd_interp‖² |
| `typ=`    | loss_typed | (only if combined with #2) noise-cancelled estimator |

Diagnostics for the HPO agent:
- `loss_time` should drop steadily for ~3k iterations then plateau near the
  bias floor of the linear-interp approximation. If it plateaus high
  (> ~0.1 × initial), `spend_time_window` may be too large for the
  dynamics correlation time.
- If `loss_time` drops fast but `conn_R2` stagnates, the smoother is
  over-smoothing — the GNN gets a too-low-bandwidth trace and cannot fit
  fast cell types (T4/T5). Reduce `spend_time_window` or
  `spend_smoother_hidden`.

## Known Results (priors)

- **Baseline**: conn_R2 = 0.7457 ± 0.0043.
- **Oracle at γ=0.10**: ≈ 0.78.
- **Add-on #3 (replay)**: targets the same ceiling; expected ~0.80 from
  preliminary instruction. Time-permute should be 5–15% behind replay due
  to the half-frame shift.

## Block Structure

### Block 1 (iter 1–8): Coefficient + window joint sweep

4 slots:
- Slot 0: `coeff=0.3, window=16` (control)
- Slot 1: `coeff=1.0, window=16` (default)
- Slot 2: `coeff=1.0, window=32` (longer window)
- Slot 3: `coeff=3.0, window=8` (heavier weight, shorter window)

Goal: coarse joint sweep — identify the productive (coeff, window) region.

### Block 2 (iter 9–16): Window refine at best Block-1 coefficient

Sweep `spend_time_window` ∈ {8, 16, 24, 32} at the Block-1 coefficient.
Test the linearity assumption boundary; expect U-shaped curve (too short
= no SNR gain; too long = interp bias).

### Block 3 (iter 17–24): Smoother capacity + LR

Sweep `spend_smoother_hidden` ∈ {16, 32, 64} and `spend_smoother_lr` at
the Block-1/2 winner.

### Block 4 (iter 25–32): Combine with standard knobs

`coeff_g_phi_diff` reduce to 1200 (less standard regularization with N2N
on); `data_augmentation_loop` increase to 30 (more passes through windows).

### Block 5 (iter 33–40): Robustness validation

4-seed CV at champion config. CV% < 10% AND mean > 0.80 → declare champion.

### Block 6+ (iter 41+): Stretch

Larger window (40+) for slow-dynamics regime; cross-cell-type smoother
(needs Add-on #2 instruction).

## Iteration Workflow

(Same Steps 1–5 as the replay instruction.)

## Winner Config (COMPULSORY)

Save best as `config/fly/flyvis_noise_005_010_spend_time_winner.yaml`
with full YAML comment header (same template as replay).

## File Structure

1. `flyvis_noise_005_010_spend_time_analysis.md` (full log)
2. `flyvis_noise_005_010_spend_time_memory.md` (working memory)
3. `user_input.md`

## Block Boundaries

(Same as replay instruction.)

## Knowledge Base Guidelines

### Established Principles examples

- "spend_time_window > 32 introduces interp bias for fast cell types
  (T4/T5 conn_R2 drops > 0.05)"
- "coeff_spend_time = 1.0 ± 3× is the productive regime"

## Start Call

When prompt says `PARALLEL START`:

- Read base config: `config/fly/flyvis_noise_005_010_spend_time.yaml`
- Read `docs/measurement_noise_bottleneck.pdf` §7.1, §7.4 and the
  `_build_smoother` / time-permute branch in `graph_trainer_spend.py`
- Initial hypothesis: **"coeff_spend_time = 1.0 with window=16 achieves
  conn_R2 > 0.78 — Block 1 sweeps the (coeff, window) joint surface."**
- Set 4 slots per Block 1 spec above

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_010_spend_time

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]
- **SPEND time-permute specifics**: [coefficient regime, window-Δt
  trade-off, comparison vs Add-on #3 (replay)]

## Knowledge Base

### Results Comparison Table

| Iter | Slot | coeff_time | window | smoother_h | smoother_lr | conn_R2 | tau_R2 | time_loss | rollout_r | time_min |
| ---- | ---- | ---------- | ------ | ---------- | ----------- | ------- | ------ | --------- | --------- | -------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

---

## Previous Block Summaries

### Block 1 Summary

---

## Current Block (Block N)

### Block Info
### Current Hypothesis
### Iterations This Block
### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```

## Failed Slots

Common time-permute failure modes:
- `spend_time_window > sim.n_frames / 100` → window samples may overlap
  too much; gradient noise increases.
- `coeff_spend_time > 10` → smoother dominates; GNN underfits dynamics.
