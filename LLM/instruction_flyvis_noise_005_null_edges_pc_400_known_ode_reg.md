# FlyVis known_ode Regularization Exploration — flyvis_noise_005_null_edges_pc_400_known_ode_reg

## Goal

The previous noise_005 + 400% null edges per column known_ode exploration
optimized W learning rates, batch size, initialization, and W_L1 / W_L2
regularization, but **never regularized the learnable biophysical parameters
tau and V_rest**. Under structured connectivity constraints (~80% sparse W,
1.74 M extra null edges), tau and V_rest can absorb explanatory power that
should belong to the (already heavily zeroed) W. This exploration adds four
new regularization knobs and sweeps them to find whether explicit weight
decay on `raw_tau` and `V_rest` can:

1. **Improve conn_R2** beyond the current null_edges plateau (~0.87-0.89
   single-seed, mean ≈ 0.87).
2. **Recover tau_R2 / V_rest_R2** (currently both very low: tau_R2 ≈ 0.01,
   V_rest_R2 ≈ 0.003) — the previous winner's bio-parameter recovery is
   essentially zero, suggesting the null-edge constraint creates a degenerate
   solution branch where tau / V_rest carry signal that should be in W.
3. **Tighten cross-seed CV** (currently 2.45% on conn_R2; rollout_pearson
   often near 0).

Primary metric: **connectivity_R2** (target conn_R2 ≥ 0.88 mean across 4
seeds, CV < 3%).

Secondary metrics: **tau_R2**, **V_rest_R2** (we expect these to **rise
substantially** with proper regularization — they're the diagnostic for
whether bio-params have been pulled away from W).

Tertiary: **rollout_pearson** (currently fragile; ideally improves once
tau / V_rest stop absorbing dynamics).

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
| `coeff_tau_L1`    | 0.0 | {0, 1e-6, 1e-5, 1e-4, 1e-3} | L1 on `raw_tau` (pre-softplus) |
| `coeff_tau_L2`    | 0.0 | {0, 1e-6, 1e-5, 1e-4, 1e-3} | L2 on `raw_tau` |
| `coeff_V_rest_L1` | 0.0 | {0, 1e-6, 1e-5, 1e-4, 1e-3} | L1 on `V_rest` |
| `coeff_V_rest_L2` | 0.0 | {0, 1e-6, 1e-5, 1e-4, 1e-3} | L2 on `V_rest` |

**Note vs. baseline noise_005 (no null edges)**: with 400% extra null edges
per column, the effective optimization problem is harder — tau / V_rest reg
may need to be **stronger** here than in the no-null-edges sweep because the
ill-posedness is worse.

## Frozen Parameters (DO NOT TOUCH)

These reuse the established null_edges_pc_400 known_ode parent settings and
are **kept fixed**:

- `lr_W: 0.0003`, `lr: 0.0018`, `lr_embedding: 0.0`
- `batch_size: 4`, `data_augmentation_loop: 35`, `n_epochs: 1`
- `w_init_mode: randn_scaled`, `w_init_scale: 1.0`
- `coeff_W_L1: 0.00015`, `coeff_W_L2: 1.5e-06`, `coeff_W_sign: 0`
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

## CRITICAL: Data is PRE-GENERATED at startup (fixed across iterations)

`generate_data: false` in the template — the data is generated **once** for all
4 slots with different `simulation.seed` values, then **reused** across all
iterations. Both `simulation.seed` and `training.seed` are **forced by the
pipeline** — DO NOT modify them in config files.

Seed formula:
- `simulation.seed = 1000 + slot` (fixed at startup, slot 0–3)
- `training.seed = iteration * 1000 + slot + 500`

To re-generate data with new seeds for robustness checks, set
`claude.test_robustness_seed: true` in all 4 slot configs.

Simulation parameters (n_neurons, n_frames, n_edges, n_extra_null_edges,
delta_t, noise_model_level) stay fixed — **DO NOT change them**.

## Scientific Context

The flyvis_known_ode model under structured null-edge constraints:

```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * ReLU(v_j) + I_i
```

where `W` is augmented with **400% extra null edges per column**, i.e.
~1.74 M of the W entries correspond to *anatomically forbidden* synapses.
The optimizer is free to push those entries to nonzero (W_L1/W_L2 punish it
mildly) and is free to absorb signal in tau / V_rest with **zero penalty**
in the previous parent config.

Three sets of parameters are learned simultaneously:

| Parameter | Tensor | Shape | # Params | Currently regularized |
| --- | --- | --- | --- | --- |
| `W` (connectivity + null) | `model.W` | (n_edges + n_null, 1) | 2,170,560 | yes (L1 + L2) |
| `tau` (time constants) | `softplus(model.raw_tau)` | (n_neurons,) | 13,741 | **NO (this exploration)** |
| `V_rest` (resting potential) | `model.V_rest` | (n_neurons,) | 13,741 | **NO (this exploration)** |

The current near-zero tau_R2 / V_rest_R2 in the parent config is a strong
hint that bio-params are doing real (but wrong) work. Pulling them toward
their priors should release that capacity back to W.

## FlyVis Specs

- 13,741 neurons, 65 cell types, 434,112 anatomical edges
- **n_extra_null_edges = 1,736,448** (`null_edges_mode: per_column`)
- 1,736 input neurons
- DAVIS visual input, 64,000 frames, delta_t = 0.02
- noise_model_level = 0.05, measurement_noise_level = 0.0

## Training Time Budget

- **target ≈ 20 min per iteration** (matches the template
  `claude.training_time_target_min`)
- node: `a100`
- DAL=35, batch_size=4, n_epochs=1 — none of these may be changed

## Parallel Mode — 4 Slots Per Batch

Each batch runs **4 slots simultaneously**:

### Exploration Mode (default)
- Slot 0: CONTROL (all four new coeffs = 0, matches the previous parent exactly)
- Slots 1-3: each changes **exactly ONE** of the four new coeffs

### Robustness Mode (when validating a promising config)
- All 4 slots: same config, different seeds

**Robustness criteria** (null_edges_pc_400, noise=0.05):
- **Stable-Robust**: 4/4 slots conn_R2 > 0.88 AND CV < 2%
- **Robust**: 4/4 slots conn_R2 > 0.87, CV 2-4%
- **Partially robust**: 2-3 slots > 0.85
- **Fragile / DISQUALIFIED**: any slot < 0.80

> **YAML rule**: Always wrap the `description` field value in double quotes —
> colons inside unquoted YAML strings cause parse errors.

## Block Structure — 6 Blocks × 12 Iterations Each

`n_iter_block=12`, total budget ≈ 72 iterations.

### Block 1 (iter 1-12): V_rest_L2 dose-response
**Hypothesis**: V_rest is currently absorbing ~all of the per-neuron signal
that the under-constrained W cannot explain (V_rest_R2 ≈ 0.003 = essentially
random); modest L2 on V_rest should pull it back, releasing capacity to W.

- Slot 0: all reg = 0 (CONTROL — matches previous parent)
- Slot 1: coeff_V_rest_L2 = 1e-6
- Slot 2: coeff_V_rest_L2 = 1e-5
- Slot 3: coeff_V_rest_L2 = 1e-4

**Expected outcome**: monotonic curve in conn_R2 vs coeff_V_rest_L2; sharp
rise in V_rest_R2 toward 0.5+; identify inflection.

### Block 2 (iter 13-24): V_rest_L1 dose-response
Mirror Block 1 with L1 in place of L2.

- Slot 0: best Block 1 setting (CONTROL)
- Slots 1-3: coeff_V_rest_L1 ∈ {1e-6, 1e-5, 1e-4}

### Block 3 (iter 25-36): tau_L2 dose-response
Same shape as Block 1 but on `coeff_tau_L2`, against the best V_rest reg
established so far.

### Block 4 (iter 37-48): tau_L1 dose-response
Same shape as Block 2 on `coeff_tau_L1`.

### Block 5 (iter 49-60): combinations and interactions
Test best individual settings combined; probe whether L1 and L2 on the same
parameter compete.

### Block 6 (iter 61-72): Robustness validation
**Switch to Robustness Mode**. Re-run the best config across 4 fresh seeds
and confirm conn_R2 ≥ 0.88 mean, CV < 2%.

## File Structure

You maintain THREE files:

1. **Full Log**: `flyvis_noise_005_null_edges_pc_400_known_ode_reg_Claude_analysis.md`
2. **Working Memory**: `flyvis_noise_005_null_edges_pc_400_known_ode_reg_Claude_memory.md`
3. **User Input**: `user_input.md`

## Knowledge Base Guidelines

### Established Principle
- 3+ iterations
- 4/4 seeds consistent
- States a causal relationship

Example: "coeff_V_rest_L2 = 1e-5 raises V_rest_R2 from 0.003 to >0.5 with
+0.005 conn_R2 improvement (3/3 iters, 4/4 seeds, CV < 2%)."

### Open Questions
- 1-2 observations
- Seed-dependent
- Conflicting evidence

### Falsified Hypotheses
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
- `tau_R2`, `V_rest_R2` (HEY — these should rise with reg, that's the goal)
- `rollout_pearson_r`
- `training_time_min`

Robustness classification:
- **Stable-Robust**: 4/4 conn_R2 > 0.88 AND CV < 2%
- **Robust**: 4/4 conn_R2 > 0.87, CV 2-4%
- **Partially robust**: 2-3 slots > 0.85
- **Fragile**: 0-1 slots > 0.85
- **DISQUALIFIED**: any slot < 0.80

### Step 3: Append Log Entry

```
## Iter N: [Stable-Robust / Robust / Partially robust / Fragile / DISQUALIFIED]
Hypothesis tested: "[quoted hypothesis]"
Slot 0: coeff_tau_L1=A, coeff_tau_L2=B, coeff_V_rest_L1=C, coeff_V_rest_L2=D
        → conn_R2=X, tau_R2=Y, V_rest_R2=Z, rollout=R, time=T min
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

1. Identify best iteration (highest conn_R2; tie-break on V_rest_R2 / tau_R2).
2. Copy that iteration's config from
   `log/Claude_exploration/LLM_flyvis_noise_005_null_edges_pc_400_known_ode_reg/config/iter_XXX_slot_YY.yaml`
3. Save to `config/fly/flyvis_noise_005_null_edges_pc_400_known_ode_reg_winner.yaml`
   with header:

```yaml
# Winner config: flyvis_noise_005_null_edges_pc_400_known_ode_reg_winner.yaml
# Source: iter_XXX_slot_YY (conn_R2 = X.XXXX, tau_R2 = X.XXX, V_rest_R2 = X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [1-2 sentence narrative]
#   - [the four reg coeffs, vs prior parent all-zero]
#
# Metrics (4-seed mean ± std):
#   conn_R2 : X.XXX ± Y.YYY
#   tau_R2  : X.XXX ± Y.YYY
#   V_rest_R2: X.XXX ± Y.YYY
#   rollout_pearson: X.XXX ± Y.YYY
```

## Start Call

When prompt says `PARALLEL START`:

- Read `flyvis_noise_005_null_edges_pc_400_known_ode_reg.yaml` template
- All four new coeffs start at 0 (matches prior parent exactly = control)
- Initialize 4 slot configs for **Block 1 (V_rest_L2 dose-response)**
- **Initial hypothesis**: "Under 400% null-edges constraint, V_rest is
  absorbing virtually all per-neuron explanatory signal (current V_rest_R2 ≈
  0.003); a moderate L2 on V_rest will release that capacity to W and lift
  conn_R2 by ≥ 0.005 while pushing V_rest_R2 above 0.3."
- Slot 0 = baseline (all four = 0)
- Slots 1-3 = coeff_V_rest_L2 ∈ {1e-6, 1e-5, 1e-4}, all other coeffs = 0

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_null_edges_pc_400_known_ode_reg

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
