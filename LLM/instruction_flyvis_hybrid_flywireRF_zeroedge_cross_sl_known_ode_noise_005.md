# FlyVis known_ode + FlyWire-RF + Zero-Edge (cross-type) — Connectome Recovery (noise=0.05)

## Goal

Optimize known_ode hyperparameters for maximum **connectivity matrix recovery (conn_R2)**
on the **hybrid FlyVis / FlyWire per-column-RF connectome augmented with cross-type
zero-weight edges** (1,959,994 edges) at noise level σ=0.05.

The prior unoptimized run (with the noise_free_known_ode_reg-tuned coefficients
`coeff_tau_L2=3e-3, coeff_V_rest_L2=1e-3, coeff_W_L1=1e-5`) **collapses on conn_R²**:

- iter 32,001: peak conn_R2 ≈ 0.885
- iter 288,001: conn_R2 = 0.665 (–25% from peak — DISQUALIFIED under any drop-aware
  classification)

τ_R² and V_rest_R² are reasonably preserved (last τ_R² = 0.901, Vr = 0.354), so the
**bottleneck is W recovery**, not the dynamics anchors. The exploration must find a
config that maintains conn_R² ≥ 0.90 at the final iteration (no late-training collapse)
without sacrificing the τ/V_rest recovery already in place.

Two sub-goals:

1. **Eliminate the late-training W collapse**: find HP settings where conn_R² is
   monotonically non-decreasing (or at least final/peak ≥ 0.95).
2. **Solve the zero-edge sparsity problem**: with ~1.63M added cross-type spurious
   edges, the L1 prior must dominate to drive their W toward 0 *without* harming the
   true 328K oracle edges.

Primary metric: **conn_R2** (promoted from secondary in noise_free explorations — this
is the bottleneck in σ=0.05 + zero-edge regime).
Secondary metrics: **tau_R2** (must not regress below 0.85), **V_rest_R2** (currently
~0.35; aim ≥ 0.5), **rollout_pearson_r**.

**Note**: σ=0.05 already breaks the τ identifiability degeneracy that motivated the
noise-free `flyvis_noise_free_known_ode_reg` exploration — τ priors here are
"defense-in-depth" rather than primary. Treat the τ/V_rest reg coefficients as already
near their winners and focus tuning effort on W.

## Existing Regularization Knobs (carry forward from noise_free exploration)

The four reg coefficients introduced in `regularizer.py` are computed alongside W_L1/W_L2:

```python
loss += coeff_tau_L1     * model.raw_tau.norm(1)
loss += coeff_tau_L2     * model.raw_tau.norm(2)
loss += coeff_V_rest_L1  * model.V_rest.norm(1)
loss += coeff_V_rest_L2  * model.V_rest.norm(2)
```

**Inherited values** (from `flyvis_noise_free_known_ode_reg_winner`-style yaml):

| Parameter         | Inherited value | Range to sweep here                     |
| ----------------- | --------------- | --------------------------------------- |
| `coeff_tau_L2`    | 3e-3            | {0, 1e-3, 3e-3, 1e-2} — light revisit  |
| `coeff_tau_L1`    | 0               | hold at 0 unless tau_R² regresses      |
| `coeff_V_rest_L2` | 1e-3            | {0, 3e-4, 1e-3, 3e-3} — light revisit  |
| `coeff_V_rest_L1` | 0               | hold at 0                              |

These are NOT the focus of this exploration. The focus is **W regularization** at
1.96M edges with 83% spurious.

## New Frozen Parameters (DO NOT TOUCH unless flagged in user_input.md)

These are inherited from the noise_free_known_ode_reg winner:

- `lr_W: 6e-4`, `lr: 6e-3`, `lr_embedding: 0.0`
- `batch_size: 4`, `n_epochs: 1`, `regul_annealing_rate: 0`
- `w_init_mode: zeros`, `w_init_scale: 1.0`
- `use_gt_edges: true`
- All `coeff_g_phi_*` and `coeff_f_theta_*` = 0 (irrelevant for known_ode)

`data_augmentation_loop: 100` is the current default; may be reduced if wall-clock
becomes binding.

If you believe one of the frozen parameters needs to change in light of the
zero-edge augmentation regime, post in `user_input.md` first.

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize** which value of (coeff_W_L1, coeff_W_L2, lr_W, DAL) will eliminate
   the late-training conn_R² collapse and improve final conn_R².
2. Change **exactly ONE coefficient** per slot vs slot 0 (the control)
3. Run training (4 seeds in parallel)
4. Analyze conn_R² **trajectory** (peak iter, peak value, final value, drop %),
   plus tau_R², V_rest_R², plus cross-seed CV
5. Update Established Principles / Open Questions / Falsified Hypotheses

### CAUSALITY RULE (MANDATORY)

**One parameter change per slot.** Slot 0 is always the parent control.

### TRAJECTORY-AWARE ANALYSIS

Because of the documented late-collapse failure mode, after each iteration also extract
from `tmp_training/metrics.log` the **iter-of-peak-conn_R2** and **(final − peak)/peak**.
Treat any slot with `final_conn_R2 / peak_conn_R2 < 0.95` as DISQUALIFIED even if final
conn_R2 looks acceptable.

Evidence hierarchy:

| Level | Criterion | Action |
| --- | --- | --- |
| **Established** | Consistent across 3+ iterations AND 4/4 seeds | Add to Principles |
| **Tentative** | Observed 1-2 times or inconsistent | Add to Open Questions |
| **Contradicted** | Conflicting evidence across iterations | Note in Open Questions |

## CRITICAL: Data is PRE-GENERATED at startup (fixed across iterations)

`generate_data: false` in the template. Both `simulation.seed` and `training.seed` are
**forced by the pipeline** — DO NOT modify them.

Seed formula:
- `simulation.seed = 1000 + slot` (fixed at startup, slot 0–3)
- `training.seed = iteration * 1000 + slot + 500`

To re-generate data with new seeds for robustness checks, set
`claude.test_robustness_seed: true` in all 4 slot configs.

Simulation parameters (n_neurons, n_frames, n_edges, delta_t, noise_model_level) stay
fixed — **DO NOT change**. In particular `noise_model_level: 0.05`.

## Scientific Context

The flyvis_known_ode model on the cross-type-augmented graph is a known-structure
linear ODE:

```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * ReLU(v_j) + I_i
```

learned simultaneously across:

| Parameter | Tensor | Shape | # Params | Reg priority here |
| --- | --- | --- | --- | --- |
| `W` (connectivity) | `model.W` | (n_edges, 1) | **1,959,994** | **PRIMARY** (sweep) |
| `tau` (time constants) | `softplus(model.raw_tau)` | (n_neurons,) | 13,741 | inherited (light revisit) |
| `V_rest` (resting potential) | `model.V_rest` | (n_neurons,) | 13,741 | inherited (light revisit) |

With ~1.63M edges that should converge to W=0, the L1 prior on W must be tuned to
drive them to zero without distorting the 328K oracle edges. This is the dual to the
oracle-edge case: there it was easy because every edge carried real signal; here, ~83%
carry zero signal.

## FlyVis Specs (cross-type augmented)

- 13,741 neurons, 65 cell types, **1,959,994 edges** (328K oracle + 1.63M cross-type
  spurious), 1,736 input neurons
- DAVIS visual input, 64,000 frames, delta_t = 0.02
- noise_model_level = 0.05 (σ=0.05 — distinct from noise_free)
- measurement_noise_level = 0.0

## Training Time Budget

- **target ≈ 25 min per iteration** (longer than the 15 min noise_free target because
  6× more edges)
- node: `a100`
- DAL=100 default (inherited); reduce to 50 or 35 if iteration > 30 min
- batch_size=4, n_epochs=1 unchanged

**Hard runtime limit (120 min)**: cluster enforces 120-min wall-clock. If
`_interrupted` markers appear, drop DAL.

> **YAML rule**: Always wrap `description` field in double quotes.

## Parallel Mode — 4 Slots Per Batch

### Exploration Mode (default)
- Slot 0: CONTROL (parent config from prior iteration)
- Slots 1-3: each changes **exactly ONE** parameter

### Robustness Mode (when validating a promising config)
- All 4 slots: same config, different seeds

**Robustness criteria** (cross-type known_ode at σ=0.05):
- **Stable-Robust**: 4/4 conn_R2 ≥ 0.90, max-drop ≤ 5%, tau_R² ≥ 0.85, CV < 1%
- **Robust**: 4/4 conn_R2 ≥ 0.85, max-drop ≤ 10%, tau_R² ≥ 0.80, CV < 3%
- **Partially robust**: 2-3 slots meet targets
- **Fragile / DISQUALIFIED**: any slot conn_R² < 0.70 OR max-drop > 20%

## Block Structure — 6 Blocks × 12 Iterations Each

`n_iter_block=12`, total budget ≈ 72 iterations.

### Block 1 (iter 1-12): Baseline + W-L1 dose-response (HIGH PRIORITY)
**Hypothesis**: The current `coeff_W_L1=1e-5` is far too weak for a 1.96M-edge graph
where ~83% of W entries should be 0. Increasing W_L1 by 1-3 orders of magnitude
should both lift final conn_R² and eliminate the late-training drop.

- Slot 0: inherited config (CONTROL: coeff_W_L1=1e-5)
- Slot 1: coeff_W_L1 = 1e-4
- Slot 2: coeff_W_L1 = 1e-3
- Slot 3: coeff_W_L1 = 1e-2

Expected: monotonic conn_R² climb with W_L1 until tau_R² or rollout_r begins to
regress, identifying the sweet spot.

### Block 2 (iter 13-24): W-L2 dose-response
Mirror Block 1 with L2.
- Slot 0: best Block 1 setting (CONTROL)
- Slots 1-3: coeff_W_L2 ∈ {1e-6, 1e-5, 1e-4}

### Block 3 (iter 25-36): lr_W decay / lower LR
**Hypothesis**: The peak-then-drop trajectory may be a stable-LR phenomenon. Lower
lr_W or LR-decay may push the peak to the final iteration.
- Slot 0: best Block 2 setting (CONTROL)
- Slots 1-3: lr_W ∈ {3e-4, 6e-4, 1.2e-3}

### Block 4 (iter 37-48): tau / V_rest reg revisit
Light revisit on the inherited tau/V_rest coefficients in case the σ=0.05 + zero-edge
regime shifts the optimum vs the noise_free regime.
- Slot 0: best Block 3 setting (CONTROL)
- Slot 1: coeff_tau_L2 = 0
- Slot 2: coeff_V_rest_L2 = 0
- Slot 3: both off

### Block 5 (iter 49-60): DAL / batch_size
Test whether the late-training drop is a function of total optimization steps. Lower
DAL or smaller batches may eliminate it.
- Slot 0: best Block 4 setting (CONTROL)
- Slots 1-3: DAL ∈ {35, 50, 100} or batch_size ∈ {2, 4, 8}

### Block 6 (iter 61-72): Robustness validation
**Switch to Robustness Mode**. Re-run the best config across 4 fresh seeds (use
`claude.test_robustness_seed: true` once if possible) and confirm conn_R² ≥ 0.90 mean
AND max-drop ≤ 5%.

## File Structure

You maintain THREE files:

1. **Full Log (append-only)**:
   `flyvis_hybrid_flywireRF_zeroedge_cross_sl_known_ode_noise_005_Claude_analysis.md`
2. **Working Memory (read + update every batch)**:
   `flyvis_hybrid_flywireRF_zeroedge_cross_sl_known_ode_noise_005_Claude_memory.md`
3. **User Input (read every batch, acknowledge pending items)**: `user_input.md`

## Knowledge Base Guidelines

### What to Add to Established Principles
- Observed across 3+ iterations
- Consistent across 4/4 seeds
- States a causal relationship

### What to Add to Open Questions
- 1-2 observations only
- Seed-dependent
- Conflicting evidence

### What to Add to Falsified Hypotheses
- Original hypothesis
- Contradicting iteration / metrics
- Lesson learned
- Revised hypothesis

A negative final result (no W_L1 / W_L2 / lr_W combination eliminates the late
collapse) is a **valid publishable finding** — document it with the same rigour as a
positive finding.

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Current Batch Results

For each slot, extract from `analysis.log` AND `tmp_training/metrics.log`:
- **`connectivity_r2` trajectory**: peak iter, peak value, final value, drop %
- `tau_R2`, `V_rest_R2`, `rollout_pearson_r` (final values)
- `training_time_min`

Robustness classification (cross-type known_ode at σ=0.05):
- **Stable-Robust**: 4/4 conn_R² ≥ 0.90, max-drop ≤ 5%, tau_R² ≥ 0.85, CV < 1%
- **Robust**: 4/4 conn_R² ≥ 0.85, max-drop ≤ 10%, tau_R² ≥ 0.80, CV < 3%
- **Partially robust**: 2-3 slots meet targets
- **Fragile**: 0-1 slots meet targets
- **DISQUALIFIED**: any slot conn_R² < 0.70, OR max-drop > 20%, OR tau_R² < 0.50

### Step 3: Append Log Entry

```
## Iter N: [Stable-Robust / Robust / Partially robust / Fragile / DISQUALIFIED]
Hypothesis tested: "[quoted hypothesis]"
Slot 0: lr_W=A, W_L1=B, W_L2=C, tau_L2=D, V_rest_L2=E, DAL=F
        → conn_R2_final=X, peak=Y@iterZ, drop=W%
        → tau_R2=T, V_rest_R2=V, time=R min
Slot 1: ...
Slot 2: ...
Slot 3: ...
Seed stats: mean_conn_R2=X, std=Y, CV=Z%, max_drop=W%, min_tau=T
Verdict: [supported / falsified / inconclusive]
Next: [planned next batch]
```

### Step 4: Update Working Memory

Append row to "Results Comparison Table", update "Emerging Observations".

### Step 5: Design Next 4 Configs

Single-parameter mutations versus the current control.

## Block Boundaries — Winner Config (COMPULSORY)

**At every block end** (iterations 12, 24, 36, 48, 60, 72):

1. Identify best iteration (highest final conn_R² with max-drop ≤ 5% AND tau_R² ≥ 0.85).
   If no iteration improves on the parent, record the parent as the block winner.
2. Copy that iteration's config from `log/Claude_exploration/.../config/iter_XXX_slot_YY.yaml`
3. Save to
   `config/fly/flyvis_hybrid_flywireRF_zeroedge_cross_sl_known_ode_noise_005_winner.yaml`
   with header:

```yaml
# Winner config: flyvis_hybrid_flywireRF_zeroedge_cross_sl_known_ode_noise_005_winner.yaml
# Source: iter_XXX_slot_YY (conn_R2 = X.XXX, max-drop = Y%, tau_R2 = X.XXX, V_rest_R2 = X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [1-2 sentence narrative]
#   - [the key HP changes vs prior winner]
#
# Metrics (4-seed mean ± std):
#   conn_R2 final  : X.XXX ± Y.YYY
#   conn_R2 peak   : X.XXX ± Y.YYY @ iter Z
#   max-drop       : X.X%
#   tau_R2         : X.XXX ± Y.YYY
#   V_rest_R2      : X.XXX ± Y.YYY
#   rollout_pearson: X.XXX ± Y.YYY
```

## Start Call

When prompt says `PARALLEL START`:

- Read `flyvis_hybrid_flywireRF_zeroedge_cross_sl_known_ode_noise_005.yaml` template
- Inherited reg coefficients: `coeff_tau_L2=3e-3, coeff_V_rest_L2=1e-3, coeff_W_L1=1e-5`
- Initialize 4 slot configs for **Block 1 (W_L1 dose-response)**
- **Initial hypothesis**: "On the 1.96M-edge cross-type-augmented connectome at σ=0.05,
  the inherited `coeff_W_L1=1e-5` is too weak to drive the ~1.63M spurious edges to 0,
  causing late-training conn_R² collapse from peak ≈ 0.885 (iter 32k) to ≈ 0.665
  (iter 288k). Increasing coeff_W_L1 to 1e-3 should both raise final conn_R² and
  eliminate the peak-drop pattern, without regressing tau_R² below 0.85."
- Slot 0 = baseline (inherited yaml, coeff_W_L1=1e-5)
- Slots 1-3 = coeff_W_L1 ∈ {1e-4, 1e-3, 1e-2}, all other coefficients held at inherited values

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_hybrid_flywireRF_zeroedge_cross_sl_known_ode_noise_005

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Results Comparison Table

| Iter | Slot | lr_W | W_L1 | W_L2 | tau_L2 | Vr_L2 | DAL | conn_R2_final | peak | drop% | tau_R2 | Vr_R2 | rollout_r | time |
| ---- | ---- | ---- | ---- | ---- | ------ | ----- | --- | ------------- | ---- | ----- | ------ | ----- | --------- | ---- |

### Established Principles

[Rules proven across 3+ iterations and 4/4 seeds]

### Falsified Hypotheses

[Hypotheses disproven by evidence]

### Open Questions

- Does coeff_W_L1 ≥ 1e-3 eliminate the late-training drop, or merely shift the peak?
- Is the τ_L2=3e-3 inherited from noise_free regime still optimal at σ=0.05, or
  does noise-broken identifiability allow τ_L2 → 0?
- Does the V_rest_R2 = 0.354 floor reflect a model-side limit or an under-tuned
  V_rest_L2 = 1e-3?

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest→newest.**

### Block 1 Summary
[W_L1 dose-response]

### Block 2 Summary
[W_L2 dose-response]

### Block 3 Summary
[lr_W tuning]

### Block 4 Summary
[tau / V_rest revisit]

---

## Current Block (Block N)

### Block Info
- Iterations: N-N+12
- Focus: [W_L1 / W_L2 / lr_W / tau-Vr revisit / DAL / robustness]

### Current Hypothesis
**Hypothesis**: [specific, testable]
**Rationale**: [why this matters]
**Test**: [the 4 configs]
**Expected outcome**: [supports vs falsifies]
**Status**: untested / supported / falsified

### Iterations This Block
[Per-iteration summary]

### Emerging Observations

[Key findings emerging from this block.]

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
