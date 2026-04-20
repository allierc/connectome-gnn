# FlyVis known_ode Regularization Exploration — flyvis_noise_005_known_ode_reg

## Goal

The previous noise_005 known_ode exploration (48 iterations, conn_R2=0.98841 ±
0.00010 on 12 seeds) optimized W learning rates, batch size, initialization,
and W_L1 / W_L2 regularization, but **never regularized the learnable
biophysical parameters tau and V_rest**. This exploration adds four new
regularization knobs and sweeps them to find whether explicit weight decay on
`raw_tau` and `V_rest` can:

1. **Improve tau_R2 / V_rest_R2 stability** (currently V_rest_R2 ≈ 0.986; small
   gains compound across noise regimes).
2. **Tighten the connectivity recovery further** — collinearity between W and
   per-neuron biases means under-regularized tau / V_rest can absorb signal
   that should belong to W.
3. **Reduce cross-seed variance** on the harder noise_005 dataset.

Primary metric: **connectivity_R2** (kept as the main success criterion;
target conn_R2 ≥ 0.985 mean across 4 seeds, CV < 1%).

Secondary metrics: **tau_R2**, **V_rest_R2** (these should NOT regress;
ideally V_rest_R2 climbs above 0.99).

## New Explorable Parameters (added in this branch)

The four new knobs are computed inside `LossRegularizer` (regularizer.py)
once per iteration, alongside W_L1/W_L2:

```python
loss += coeff_tau_L1     * model.raw_tau.norm(1)
loss += coeff_tau_L2     * model.raw_tau.norm(2)
loss += coeff_V_rest_L1  * model.V_rest.norm(1)
loss += coeff_V_rest_L2  * model.V_rest.norm(2)
```

| Parameter | Default | Suggested sweep range | Notes |
| --- | --- | --- | --- |
| `coeff_tau_L1`    | 0.0 | {0, 1e-6, 1e-5, 1e-4, 1e-3} | L1 on `raw_tau` (pre-softplus). Sparse-prior, will pull tau toward softplus(0) ≈ 0.69 |
| `coeff_tau_L2`    | 0.0 | {0, 1e-6, 1e-5, 1e-4, 1e-3} | L2 on `raw_tau`. Smooth shrinkage toward identity element of softplus |
| `coeff_V_rest_L1` | 0.0 | {0, 1e-6, 1e-5, 1e-4, 1e-3} | L1 on `V_rest`. Promotes per-neuron resting potentials near 0 |
| `coeff_V_rest_L2` | 0.0 | {0, 1e-6, 1e-5, 1e-4, 1e-3} | L2 on `V_rest`. Smooth shrinkage of the resting potential |

The penalty is summed across all 13,741 neurons, so coefficient magnitudes are
roughly comparable to `coeff_W_L2 = 1.5e-6` applied to 434K W entries (i.e.
N_W / N_neurons ≈ 32×, so equivalent per-parameter strength is ~30× larger
nominal coefficient on tau/V_rest than on W).

## Frozen Parameters (DO NOT TOUCH)

These were exhaustively optimized in the previous sweep and are **kept
fixed** at the winner values:

- `lr_W: 0.0003`, `lr: 0.0018`, `lr_embedding: 0.0`
- `batch_size: 4`, `data_augmentation_loop: 35`, `n_epochs: 1`
- `w_init_mode: randn_scaled`, `w_init_scale: 1.0`
- `coeff_W_L1: 0.00015`, `coeff_W_L2: 1.5e-06`
- `use_gt_edges: true`, `regul_annealing_rate: 0`
- All `coeff_g_phi_*` and `coeff_f_theta_*` = 0 (irrelevant for known_ode)

If you believe one of the frozen parameters needs to change in light of new
tau/V_rest reg, post in `user_input.md` first.

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize** which combination / range of (tau_L1, tau_L2, V_rest_L1, V_rest_L2) will improve metrics
2. Change **exactly ONE coefficient** per slot vs slot 0 (the control)
3. Run training (4 seeds in parallel)
4. Analyze conn_R2, tau_R2, V_rest_R2, plus cross-seed CV
5. Update Established Principles / Open Questions / Falsified Hypotheses

Evidence hierarchy:

| Level | Criterion | Action |
| --- | --- | --- |
| **Established** | Consistent across 3+ iterations AND 4/4 seeds | Add to Principles |
| **Tentative** | Observed 1-2 times or inconsistent | Add to Open Questions |
| **Contradicted** | Conflicting evidence across iterations | Note in Open Questions |

## CRITICAL: Data is PRE-GENERATED at startup (fixed across iterations)

`generate_data: false` in the template — the data is generated **once** for all
4 slots with different `simulation.seed` values, then **reused** across all
iterations. Both `simulation.seed` and `training.seed` are **forced by the
pipeline** — DO NOT modify them in config files.

Seed formula:
- `simulation.seed = 1000 + slot` (fixed at startup, slot 0–3)
- `training.seed = iteration * 1000 + slot + 500` (varies per iteration)

To re-generate data with new seeds for robustness checks, set
`claude.test_robustness_seed: true` in all 4 slot configs.

Simulation parameters (n_neurons, n_frames, n_edges, delta_t, noise_model_level)
stay fixed — **DO NOT change them**.

## Scientific Context

The flyvis_known_ode model (registered as `flyvis_known_ode`) is a
**known-structure** linear ODE:

```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * ReLU(v_j) + I_i
```

Three sets of parameters are learned simultaneously from voltage observations:

| Parameter | Tensor | Shape | # Params | Currently regularized |
| --- | --- | --- | --- | --- |
| `W` (connectivity) | `model.W` | (n_edges, 1) | 434,112 | yes (L1 + L2) |
| `tau` (time constants) | `softplus(model.raw_tau)` | (n_neurons,) | 13,741 | **NO (this exploration)** |
| `V_rest` (resting potential) | `model.V_rest` | (n_neurons,) | 13,741 | **NO (this exploration)** |

Without tau/V_rest regularization, those parameters have **complete freedom**
to absorb explanatory power that should land on W. At noise=0.05 this can
plausibly cost a few thousandths of conn_R2 — small but worth chasing because
the previous sweep already saturated the conventional knobs.

## FlyVis Specs

- 13,741 neurons, 65 cell types, 434,112 edges, 1,736 input neurons
- DAVIS visual input, 64,000 frames, delta_t = 0.02
- noise_model_level = 0.05 (moderate-low dynamics noise)
- measurement_noise_level = 0.0

## Training Time Budget

- **target ≈ 20 min per iteration** (matches the template
  `claude.training_time_target_min`)
- node: `a100`
- DAL=35, batch_size=4, n_epochs=1 — none of these may be changed

## Parallel Mode — 4 Slots Per Batch

Each batch runs **4 slots simultaneously**:

### Exploration Mode (default)
- Slot 0: CONTROL (all four new coeffs = 0, matches the previous winner exactly)
- Slots 1-3: each changes **exactly ONE** of the four new coeffs

### Robustness Mode (when validating a promising config)
- All 4 slots: same config, different seeds
- Measures stability across seed variation

**Robustness criteria** (noise_005 known_ode):
- **Stable-Robust**: 4/4 slots conn_R2 > 0.985 AND CV < 1%
- **Robust**: 4/4 slots conn_R2 > 0.985, CV 1-3%
- **Partially robust**: 2-3 slots > 0.98
- **Fragile / DISQUALIFIED**: any slot < 0.97

> **YAML rule**: Always wrap the `description` field value in double quotes —
> colons inside unquoted YAML strings cause parse errors.

## Block Structure — 6 Blocks × 12 Iterations Each

`n_iter_block=12`, total budget ≈ 72 iterations.

### Block 1 (iter 1-12): V_rest_L2 dose-response
**Hypothesis**: V_rest is the most under-constrained parameter (per-neuron
bias absorbs noise → conn_R2 ceiling). A small L2 on V_rest should improve
conn_R2 without hurting V_rest_R2.

- Slot 0: all reg = 0 (CONTROL)
- Slot 1: coeff_V_rest_L2 = 1e-6
- Slot 2: coeff_V_rest_L2 = 1e-5
- Slot 3: coeff_V_rest_L2 = 1e-4

Expected: monotonic curve in conn_R2 vs coeff_V_rest_L2; identify the
inflection.

### Block 2 (iter 13-24): V_rest_L1 dose-response
Mirror Block 1 with L1 in place of L2 (test sparsity prior on V_rest).

- Slot 0: best Block 1 setting (CONTROL)
- Slots 1-3: coeff_V_rest_L1 ∈ {1e-6, 1e-5, 1e-4}

### Block 3 (iter 25-36): tau_L2 dose-response
Same shape as Block 1 but on `coeff_tau_L2` (against the best V_rest reg
established so far).

### Block 4 (iter 37-48): tau_L1 dose-response
Same shape as Block 2 on `coeff_tau_L1`.

### Block 5 (iter 49-60): combinations and interactions
Test the best individual settings combined (e.g. best V_rest_L2 + best tau_L2),
and probe interactions (does L1 + L2 on the same parameter compete?).

### Block 6 (iter 61-72): Robustness validation
**Switch to Robustness Mode**. Re-run the best config across 4 fresh seeds
(use `claude.test_robustness_seed: true` once if possible) and confirm
conn_R2 ≥ 0.985 mean, CV < 1%.

## File Structure

You maintain THREE files:

1. **Full Log (append-only)**: `flyvis_noise_005_known_ode_reg_Claude_analysis.md`
2. **Working Memory (read + update every batch)**: `flyvis_noise_005_known_ode_reg_Claude_memory.md`
3. **User Input (read every batch, acknowledge pending items)**: `user_input.md`

## Knowledge Base Guidelines

### What to Add to Established Principles
- Observed across 3+ iterations
- Consistent across 4/4 seeds
- States a causal relationship

Example: "coeff_V_rest_L2 ∈ [1e-5, 5e-5] improves conn_R2 by ≥ 0.001 without
regression in V_rest_R2 (3/3 iters, 4/4 seeds, CV < 0.5%)."

### What to Add to Open Questions
- 1-2 observations only
- Seed-dependent
- Conflicting evidence

### What to Add to Falsified Hypotheses
- Original hypothesis
- Contradicting iteration / metrics
- Lesson learned
- Revised hypothesis

## Iteration Workflow

### Step 1: Read Working Memory + User Input
Review the comparison table and emerging observations.

### Step 2: Analyze Current Batch Results
For each slot, extract from `analysis.log`:
- `connectivity_R2` (primary)
- `tau_R2`, `V_rest_R2` (secondary — these MUST NOT regress)
- `rollout_pearson_r`
- `training_time_min`

Robustness classification:
- **Stable-Robust**: 4/4 conn_R2 > 0.985 AND CV < 1%
- **Robust**: 4/4 conn_R2 > 0.985, CV 1-3%
- **Partially robust**: 2-3 slots > 0.98
- **Fragile**: 0-1 slots > 0.98
- **DISQUALIFIED**: any slot < 0.97

### Step 3: Append Log Entry

```
## Iter N: [Stable-Robust / Robust / Partially robust / Fragile / DISQUALIFIED]
Hypothesis tested: "[quoted hypothesis]"
Slot 0: coeff_tau_L1=A, coeff_tau_L2=B, coeff_V_rest_L1=C, coeff_V_rest_L2=D
        → conn_R2=X, tau_R2=Y, V_rest_R2=Z, time=T min
Slot 1: ...
Slot 2: ...
Slot 3: ...
Seed stats: mean_conn_R2=X, std=Y, CV=Z%, min=W, max=V
Verdict: [supported / falsified / inconclusive]
Next: [planned next batch]
```

### Step 4: Update Working Memory

Append row to "Results Comparison Table", update "Emerging Observations".

### Step 5: Design Next 4 Configs

Single-parameter mutations versus the current control.

## Block Boundaries — Winner Config (COMPULSORY)

**At every block end** (iterations 12, 24, 36, 48, 60, 72):

1. Identify best iteration (highest conn_R2, with no tau_R2/V_rest_R2
   regression vs the prior winner).
2. Copy that iteration's config from
   `log/Claude_exploration/LLM_flyvis_noise_005_known_ode_reg/config/iter_XXX_slot_YY.yaml`
3. Save to `config/fly/flyvis_noise_005_known_ode_reg_winner.yaml` with header:

```yaml
# Winner config: flyvis_noise_005_known_ode_reg_winner.yaml
# Source: iter_XXX_slot_YY (conn_R2 = X.XXXX, tau_R2 = X.XXX, V_rest_R2 = X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [1-2 sentence narrative]
#   - [the four reg coeffs, vs prior winner all-zero]
#
# Metrics (4-seed mean ± std):
#   conn_R2 : X.XXX ± Y.YYY
#   tau_R2  : X.XXX ± Y.YYY
#   V_rest_R2: X.XXX ± Y.YYY
#   rollout_pearson: X.XXX ± Y.YYY
```

## Start Call

When prompt says `PARALLEL START`:

- Read `flyvis_noise_005_known_ode_reg.yaml` template
- All four new coeffs start at 0 (matches prior winner exactly = control)
- Initialize 4 slot configs for **Block 1 (V_rest_L2 dose-response)**
- **Initial hypothesis**: "V_rest is the most under-constrained parameter at
  noise=0.05; a moderate L2 on V_rest will improve conn_R2 by ≥ 0.001
  without regressing V_rest_R2."
- Slot 0 = baseline (all four = 0)
- Slots 1-3 = coeff_V_rest_L2 ∈ {1e-6, 1e-5, 1e-4}, all other coeffs = 0

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_known_ode_reg

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Results Comparison Table

| Iter | Slot | tau_L1 | tau_L2 | Vrest_L1 | Vrest_L2 | conn_R2 | tau_R2 | Vrest_R2 | rollout_r | time_min |
| ---- | ---- | ------ | ------ | -------- | -------- | ------- | ------ | -------- | --------- | -------- |

### Established Principles

[Rules proven across 3+ iterations and 4/4 seeds]

### Falsified Hypotheses

[Hypotheses disproven by evidence]

### Open Questions

[Uncertainties still under investigation]

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest→newest.**

### Block 1 Summary
[V_rest_L2 dose-response]

### Block 2 Summary
[V_rest_L1 dose-response]

### Block 3 Summary
[tau_L2 dose-response]

### Block 4 Summary
[tau_L1 dose-response]

---

## Current Block (Block N)

### Block Info
- Iterations: N-N+12
- Focus: [V_rest_L2 / V_rest_L1 / tau_L2 / tau_L1 / combinations / robustness]

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
