# FlyVis GNN — SPEND Add-on #3 (stimulus-replay N2N)

## Goal

Use SPEND's stimulus-replay Noise2Noise denoising to break the
**conn_R2 = 0.745 ceiling** under dual noise (σ=0.05 model, γ=0.10 measurement).
Train an inline 1D-conv smoother co-supervised with the GNN: it learns
`E[v_clean | v + n_a]` from the pair `(v + n_a, v + n_b)` where `n_a, n_b` are
independent Gaussian noise realisations of the same clean trajectory. The
denoised trace replaces `v + n_a` as the GNN input.

Primary metric: **connectivity_R2** (R² between learned W and ground-truth W).
Target: **connectivity_R2 > 0.80** (oracle ceiling at γ=0.10 is ~0.78; the
GNN has more structure than the oracle so 0.80 is achievable).

Cite: https://github.com/buchenglab/SPEND  (Ding et al. 2025, Newton 1, 100195).
Bottleneck analysis: `docs/measurement_noise_bottleneck.pdf` §7.1, §7.4.

## Scientific Context: Why N2N breaks the ceiling

The pointwise MSE on `(v[t+1] − v[t])/Δt` has derivative noise std ≈ 7.07 at
γ=0.10, drowning the supervised signal and locking the optimizer onto the
slope-shrunk minimum of the W↔f_theta scale symmetry. SPEND's insight: train
a denoiser via N2N (two independent noisy views of the same clean signal);
the denoiser converges to the posterior mean E[v_clean | noisy] (Lehtinen
et al. 2018). The denoised trace has variance reduced ~8× — the downstream
MSE then operates on a near-clean target and the slope-shrunk basin loses
its energetic preference.

The *replay* variant generates two views by synthesising two noise tensors
with explicit RNG seeds on top of the **clean** voltage. This requires
loading the dataset with `measurement_noise_level=0` (handled by
`spend_load_clean: true` — see `graph_trainer_spend.py`).

## Training Mode

`data_train_spend(config, ...)` in `src/connectome_gnn/models/graph_trainer_spend.py`.
Invoked via `python GNN_Main.py -o train_SPEND <config>` or inside the
agentic loop by the pipeline.

**Training from scratch** every iteration; no `pretrained_model`.
**1 epoch** per iteration; baseline expects ~60 min/iteration on a100.

## Data

Training data is **pre-generated and fixed** (`generate_data: false`). All
slots reuse the existing `fly/flyvis_noise_005_010` dataset; the SPEND
trainer loads it with `measurement_noise_level=0` (clean voltage) and
synthesises two independent noise tensors inline at training start using the
seeds `spend_replay_noise_seed_a/b`. **DO NOT** modify simulation parameters
or the dataset name.

Seed strategy: 4 slots run different (a, b) pairs to test sensitivity, OR 4
slots run the same pair with different `training.seed` for robustness — see
slot strategy below.

## Noise Model

Identical to `flyvis_noise_005_010` (σ_dyn=0.05, γ_meas=0.10). The replay
trainer ignores the disk-saved noise tensor and uses the synth-noise pair
described above.

## FlyVis Model

13,741 neurons, 65 cell types, 434,112 edges. Non-spiking compartmental ODE
with `g_phi(v_j)^2` edge messages and `f_theta(v_i, msg, I_i)` updates. **Do
not change** `signal_model_name`, `embedding_dim`, `hidden_dim`, etc.

## Explorable Parameters

### SPEND replay knobs (PRIMARY)

| Parameter                       | Default | Safe range       | Notes |
|---------------------------------|---------|------------------|-------|
| `coeff_spend_replay`            | 1.0     | 0.1–10.0 (log)   | Weight on the N2N denoising loss |
| `spend_smoother_hidden`         | 32      | 16, 32, 64       | 1D-conv smoother width |
| `spend_smoother_lr`             | 1e-3    | 1e-4 – 1e-2      | Separate LR for smoother param group |
| `spend_replay_noise_seed_a`     | 0       | any int          | RNG seed for first noise realisation |
| `spend_replay_noise_seed_b`     | 1       | any int (≠ a)    | RNG seed for second noise realisation |

Locked: `spend_load_clean: true`; `coeff_spend_time: 0`; `coeff_spend_typed: 0`.

### Standard GNN levers (compatible; sweep cautiously)

| Parameter                  | Default  | Safe range   | Notes |
|----------------------------|----------|--------------|-------|
| `coeff_g_phi_diff`         | 2000     | 1200–3000    | Most effective lever in winner config |
| `coeff_W_L1`               | 1.5e-4   | 5e-5–5e-4    | Sparsity prior on W |
| `lr_W`                     | 9e-4     | 3e-4–1.2e-3  | LR for W |
| `lr`                       | 1.8e-3   | 1e-3–2.5e-3  | LR for MLPs |
| `data_augmentation_loop`   | 20       | 10–40        | Trades training time for SNR |
| `batch_size`               | 6        | 4–12         | Larger batches average noise across samples |

**Locked / do not touch:** `recurrent_training` (false), `n_epochs` (1),
`pretrained_model` (empty), simulation params, architecture.

> **YAML rule:** Always wrap `description:` value in double quotes (colons in
> values otherwise break YAML parsing).

## Slot Strategy — 4 Different Configs Per Batch

Each batch tests **4 distinct configs** simultaneously (rc-style). The LLM
designs 4 related but distinct settings to maximise exploration speed.

### Config Files

- Edit all 4 config files: `flyvis_noise_005_010_spend_replay_00.yaml` through
  `flyvis_noise_005_010_spend_replay_03.yaml`.
- Each config can be **different** — the LLM decides what each slot tests.
- **DO NOT change**: `simulation:` parameters, `dataset`, `pretrained_model`,
  `n_epochs`, architecture. Seeds are managed automatically.

## Evaluation

After each training run, the pipeline calls `data_test_flyvis()` which:

1. Loads the best model checkpoint (smoother is silently ignored via
   `strict=False`).
2. Evaluates connectivity R2, tau R2, V_rest R2 against ground truth.
3. Runs rollout on noise-free test data (`test_dataset: fly/flyvis_noise_free`).
4. Reports rollout Pearson r and RMSE.

Metrics from `analysis.log`:

- `connectivity_R2`: R2 of learned vs true W (**PRIMARY**)
- `tau_R2`, `V_rest_R2`, `cluster_accuracy`, `rollout_pearson_r`, `training_time_min`

Per-iteration metrics from `tmp_training/metrics.log` (standard 6-column
schema; identical to baseline trainer): `iteration, connectivity_r2,
vrest_r2, tau_r2, hidden_nnr_pearson, anchor_nnr_pearson` (last two are
`nan` for SPEND).

**SPEND-specific** per-iteration metrics from `tmp_training/spend_components.log`:
header `iteration, loss_main, loss_replay, loss_time, loss_typed`. **The
HPO agent MUST read this file** to diagnose the SPEND mechanism.

Progress-bar abbreviations (live tail; values are 20-iter EMAs of the
per-batch losses, which fluctuate ±100% raw):

| Bar label | File column   | Meaning |
|-----------|---------------|---------|
| `conn=`   | connectivity_r2 (metrics.log) | R² of learned W vs ground-truth W |
| `Vr=`     | vrest_r2      | R² of recovered V_rest vs ground-truth |
| `tau=`    | tau_r2        | R² of recovered τ vs ground-truth |
| `rep=`    | loss_replay   | Add-on #3 N2N MSE: ‖smoother(v+n_a) − (v+n_b)‖² |
| `tim=`    | loss_time     | Add-on #1 N2N MSE: ‖smoother(even) − odd_interp‖² |
| `typ=`    | loss_typed    | Add-on #2 noise-cancelled estimator: ‖v_i1 − v_i2‖² − 2γ² |

Diagnostic rules of thumb for the HPO agent:
- `loss_replay` should drop monotonically over the first ~3k iterations,
  then plateau near the irreducible-noise floor (~γ² ≈ 0.01 at γ=0.10).
  If it stays flat or rises, `coeff_spend_replay` is too small or the
  smoother LR is too high.
- `loss_main` (the GNN MSE on dv/dt) should drop only after `loss_replay`
  starts plateauing — the smoother needs to converge before its denoised
  trace is useful.
- If `conn_R2` plateaus while `loss_replay` keeps dropping, the smoother is
  over-fitting the noise pair; reduce `spend_smoother_hidden` or
  `spend_smoother_lr`.

## Known Results (priors)

- **Baseline (no SPEND)**: conn_R2 = 0.7457 ± 0.0043 (4-seed CV=0.58%). The
  hard ceiling under derivative-noise std ~7.07.
- **Oracle (Known-ODE) at γ=0.10**: conn_R2 ≈ 0.78. SPEND should approach or
  exceed this.
- **SPEND in imaging (Ding et al. 2025)**: 8× SNR gain on hyperspectral SRS.
  Translation to our setting: derivative-noise std should drop from 7.07 to
  ~0.9, putting the loss landscape well into the noise-free regime.

## Block Structure

### Block 1 (iter 1–8): Coefficient sweep

Sweep `coeff_spend_replay` ∈ {0.1, 0.3, 1.0, 3.0} across 4 slots. Identify
the regime where the N2N loss is meaningful but does not dominate the
main MSE. Use seed=baseline.

Goal: find the order of magnitude of `coeff_spend_replay`.

### Block 2 (iter 9–16): Smoother capacity sweep

Fix `coeff_spend_replay` at the Block-1 winner. Sweep
`spend_smoother_hidden` ∈ {16, 32, 64} + one slot at the smoother LR
boundary (1e-2). Goal: smaller is better unless the smoother under-fits.

### Block 3 (iter 17–24): Smoother LR + warm-up interaction

Sweep `spend_smoother_lr` ∈ {1e-4, 3e-4, 1e-3, 3e-3} with the Block-1/2
winner. Hypothesis: a slower smoother prevents it from absorbing the
clean-signal correction in the first 1k iterations before the GNN finds W.

### Block 4 (iter 25–32): Combine SPEND replay with standard knobs

Test interactions: lower `coeff_g_phi_diff` (e.g. 1200) with strong
SPEND on; lower `coeff_W_L1` (e.g. 5e-5) with strong SPEND on. SPEND
denoising may make some existing regularizers redundant.

### Block 5 (iter 33–40): Robustness validation

Run **identical** champion config with 4 different `training.seed` values.
Decision: if CV% < 10% AND mean > 0.80, declare champion. If
CV% > 20% → result was lucky; reopen Blocks 2/3.

### Block 6+ (iter 41+): Stretch — beat the oracle

Try `data_augmentation_loop=40`, `batch_size=10`, larger smoother window
(needs `spend_time_window` knob even though we're in replay mode — irrelevant
here, leave at 16). Goal: conn_R2 > 0.85.

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots — each with different config)

For each slot, report:
- Config: coeff_spend_replay, spend_smoother_hidden, spend_smoother_lr, +
  any standard-lever changes
- Metrics: connectivity_R2, tau_R2, V_rest_R2, rollout_pearson_r,
  training_time_min, **loss_replay** (final value)

### Step 3: Write Log Entries

```
## Iter N
Hypothesis: "[quoted hypothesis]"
Slot 0: config=[params] → conn_R2=X, tau_R2=Y, Vrest_R2=Z, replay_loss=L, time=T
Slot 1: ...
Slot 2: ...
Slot 3: ...
Best slot: [which] with conn_R2=X
Verdict: [supported/falsified/inconclusive]
Next: [what to test in next batch]
```

### Step 4: Acknowledge User Input

### Step 5: Design Next 4 Configs

## Winner Config (COMPULSORY)

At every block boundary, save the best config as
`config/fly/flyvis_noise_005_010_spend_replay_winner.yaml` with a YAML
comment header:

```yaml
# Winner config: flyvis_noise_005_010_spend_replay_winner.yaml
# Source: iter_XXX_slot_YY (connectivity_R2 = X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [1-2 sentence narrative]
#   - [key SPEND knob choices and why]
#
# Metrics:
#   connectivity_R2: X.XXX (best single seed)
#   robust_mean:     X.XXX +/- X.XXX (N seeds, CV=X.X%)
#   loss_replay:     X.XXX (final)
#   rollout_pearson: X.XXX
```

## File Structure

1. **Full Log** (append-only): `flyvis_noise_005_010_spend_replay_analysis.md`
2. **Working Memory** (read + update): `flyvis_noise_005_010_spend_replay_memory.md`
3. **User Input**: `user_input.md`

## Block Boundaries

1. Update "Paper Summary" — focus on SPEND replay as N2N denoising
2. Summarize findings in "Previous Block Summary"
3. Update "Established Principles" (3+ supporting iterations rule)
4. Move falsified hypotheses to "Falsified Hypotheses"
5. Clear "Current Block"
6. Save winner YAML

## Knowledge Base Guidelines

### What to Add to Established Principles

A principle must satisfy ALL of:
1. Observed consistently across **3+ iterations**
2. Consistent across **all 4 slots** (not just mean)
3. States a **causal relationship**

Examples:
- "coeff_spend_replay > 3.0 destabilizes W learning (loss_replay grows;
  conn_R2 drops monotonically)"
- "spend_smoother_hidden > 32 leads to over-smoothing (loss_replay drops
  fast but conn_R2 plateaus early)"

### What to Add to Falsified Hypotheses

Original hypothesis → contradicting evidence (iter, metrics) → revised hypothesis.

## Start Call

When prompt says `PARALLEL START`:

- Read base config: `config/fly/flyvis_noise_005_010_spend_replay.yaml`
- Note: `spend_load_clean: true`, `coeff_spend_replay: 1.0`, `node_name: a100`,
  `generate_data: false`
- Read `docs/measurement_noise_bottleneck.pdf` §7.1 and the SPEND header
  comment in `graph_trainer_spend.py` for context
- Initial hypothesis: **"coeff_spend_replay = 1.0 with default smoother
  (h=32, lr=1e-3) achieves conn_R2 > 0.78 (oracle ceiling) — Block 1
  validates the order of magnitude."**
- Set 4 slots to coeff_spend_replay ∈ {0.1, 0.3, 1.0, 3.0}, all else
  identical to the base config

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_010_spend_replay

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]
- **SPEND replay specifics**: [pending — what coefficient regime works,
  smoother size, interaction with standard regularizers]

## Knowledge Base

### Results Comparison Table

| Iter | Slot | coeff_replay | smoother_h | smoother_lr | seed_a | seed_b | conn_R2 | tau_R2 | replay_loss | rollout_r | time_min |
| ---- | ---- | ------------ | ---------- | ----------- | ------ | ------ | ------- | ------ | ----------- | --------- | -------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

---

## Previous Block Summaries

### Block 1 Summary
[Summary of Block 1 findings]

---

## Current Block (Block N)

### Block Info

### Current Hypothesis

**Hypothesis**: [specific, testable]
**Rationale**: [why]
**Test**: [what 4 configs test this]
**Expected outcome**: [supports vs falsifies]
**Status**: untested / supported / falsified / revised

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```

## Failed Slots

A slot failure indicates the config is unstable — note which parameter
caused it and avoid that region of parameter space. Common SPEND replay
failure modes:
- `coeff_spend_replay > 10` → smoother dominates loss, GNN gradient vanishes
- `spend_smoother_lr > 1e-2` → smoother oscillates, conn_R2 ~ 0
