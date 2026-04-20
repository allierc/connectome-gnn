# FlyVis known_ode Regularization Exploration — flyvis_noise_free_known_ode_reg

## Goal

The previous noise-free known_ode exploration (84 iterations, conn_R2 =
0.9776 ± 0.0001 across 8 seeds) achieved near-perfect W recovery but
**failed on τ recovery**: tau_R2 = 0.00025 ± 0.00004. V_rest_R2 = 0.9047
± 0.0010 is solid but below the noisy-regime ceiling of ≥ 0.99.

The prior failure on τ is not an optimization bug — it's an
**identifiability degeneracy**. With σ=0 the dynamics
`τ_i · dv_i/dt = -v_i + V_rest_i + Σ_j W_ij ReLU(v_j) + I_i(t)` are
invariant under the rescaling `(τ, W, V_rest, I) → (c·τ, c·W, c·V_rest, c·I)`
for any `c > 0`: any `c` gives the same voltage trajectory, so τ has no
signal to pin its magnitude. Noise σ > 0 breaks this degeneracy in the
σ=0.05 and σ=0.5 rows of `tab:cv_known_ode` (τ_R² jumps to ≥ 0.999).
**The central open question this exploration probes: can an L1/L2 prior on
`raw_tau` and/or `V_rest` restore τ_R² > 0.5 in the noise-free regime?**

Two sub-goals:

1. **Break the scaling degeneracy**. L1 / L2 on `raw_tau` both shrink the
   learnable `raw_tau` toward 0 (so τ = softplus(raw_tau) toward
   softplus(0) ≈ 0.69 ms). That supplies an *anchor* in absolute units.
   If the effective scale of the optimum matches the ground truth τ, this
   could move τ_R² materially above 0 — the first test of the hypothesis
   that reg can substitute for the missing noise-based timescale anchor.
2. **Improve V_rest_R2** from 0.905 toward 0.99, as a secondary metric.
   A modest L2 on `V_rest` could tighten per-neuron residuals.

Primary metric: **tau_R2** (promoted from secondary in the noisy-regime
explorations — this is the bottleneck in σ=0).
Secondary metrics: **connectivity_R2** (must not regress below 0.977),
**V_rest_R2** (ideally climbs above 0.95), **rollout_pearson_r**.

**Null-result possibility**: if no combination of the four new coefficients
moves tau_R² above ~0.05 without regressing conn_R², the final verdict is
that identifiability — not optimization — is the bottleneck in noise-free.
Document that outcome cleanly: it is a publishable negative finding.

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
| `coeff_tau_L1`    | 0.0 | {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2} | L1 on `raw_tau` (pre-softplus). Sparse-prior, pulls τ toward softplus(0) ≈ 0.69 ms |
| `coeff_tau_L2`    | 0.0 | {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2} | L2 on `raw_tau`. Smooth shrinkage toward identity element of softplus |
| `coeff_V_rest_L1` | 0.0 | {0, 1e-6, 1e-5, 1e-4, 1e-3} | L1 on `V_rest`. Promotes per-neuron resting potentials near 0 |
| `coeff_V_rest_L2` | 0.0 | {0, 1e-6, 1e-5, 1e-4, 1e-3} | L2 on `V_rest`. Smooth shrinkage of the resting potential |

Note the extended range for τ coefficients compared to noisy variants:
the σ=0 regime tolerates stronger τ priors because there is no competing
training-noise signal to be overpowered.

The penalty is summed across all 13,741 neurons, so coefficient magnitudes
are roughly comparable to `coeff_W_L1 = 1e-5` applied to 434K W entries
(i.e. `N_W / N_neurons ≈ 32×`, so per-parameter strength of nominal
tau/V_rest coefficients is ~30× larger than on W).

## Frozen Parameters (DO NOT TOUCH)

These were exhaustively optimized in the prior 84-iteration noise-free
sweep and are **kept fixed** at the winner values:

- `lr_W: 0.0006`, `lr: 0.006`, `lr_embedding: 0.0`
- `batch_size: 4`, `data_augmentation_loop: 35`, `n_epochs: 1`
- `w_init_mode: zeros`, `w_init_scale: 1.0`
- `coeff_W_L1: 1e-5`, `coeff_W_L2: 0`
- `use_gt_edges: true`, `regul_annealing_rate: 0`
- All `coeff_g_phi_*` and `coeff_f_theta_*` = 0 (irrelevant for known_ode)

If you believe one of the frozen parameters needs to change in light of
new tau/V_rest reg, post in `user_input.md` first.

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize** which combination / range of (tau_L1, tau_L2, V_rest_L1, V_rest_L2) will improve metrics
2. Change **exactly ONE coefficient** per slot vs slot 0 (the control)
3. Run training (4 seeds in parallel)
4. Analyze tau_R2, conn_R2, V_rest_R2, plus cross-seed CV
5. Update Established Principles / Open Questions / Falsified Hypotheses

Evidence hierarchy:

| Level | Criterion | Action |
| --- | --- | --- |
| **Established** | Consistent across 3+ iterations AND 4/4 seeds | Add to Principles |
| **Tentative** | Observed 1-2 times or inconsistent | Add to Open Questions |
| **Contradicted** | Conflicting evidence across iterations | Note in Open Questions |

## CRITICAL: Data is PRE-GENERATED at startup (fixed across iterations)

`generate_data: false` in the template — data is generated **once** for all 4
slots with different `simulation.seed` values, then **reused** across
iterations. Both `simulation.seed` and `training.seed` are **forced by the
pipeline** — DO NOT modify them in config files.

Seed formula:
- `simulation.seed = 1000 + slot` (fixed at startup, slot 0–3)
- `training.seed = iteration * 1000 + slot + 500` (varies per iteration)

To re-generate data with new seeds for robustness checks, set
`claude.test_robustness_seed: true` in all 4 slot configs.

Simulation parameters (n_neurons, n_frames, n_edges, delta_t,
noise_model_level) stay fixed — **DO NOT change them**. In particular,
`noise_model_level` must stay 0.0; this exploration exists to answer the
noise-free question.

## Scientific Context

The flyvis_known_ode model is a **known-structure** linear ODE:

```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * ReLU(v_j) + I_i
```

Three sets of parameters are learned simultaneously from voltage
observations:

| Parameter | Tensor | Shape | # Params | Currently regularized |
| --- | --- | --- | --- | --- |
| `W` (connectivity) | `model.W` | (n_edges, 1) | 434,112 | yes (L1 = 1e-5) |
| `tau` (time constants) | `softplus(model.raw_tau)` | (n_neurons,) | 13,741 | **NO (this exploration)** |
| `V_rest` (resting potential) | `model.V_rest` | (n_neurons,) | 13,741 | **NO (this exploration)** |

Without regularization, τ has no scale anchor in σ=0 (identifiability
degeneracy above). Any uniform rescaling of (τ, W, V_rest) gives the same
voltage trajectory. The prior exploration's tau_R² ≈ 0.0004 is consistent
with the optimizer finding *any* scale in the degenerate family. A
non-zero L1/L2 on `raw_tau` breaks that symmetry because the penalty's
minimum at `raw_tau = 0` provides a (biased but nonzero) absolute scale.

## FlyVis Specs

- 13,741 neurons, 65 cell types, 434,112 edges, 1,736 input neurons
- DAVIS visual input, 64,000 frames, delta_t = 0.02
- noise_model_level = 0.0 (deterministic dynamics — this run only)
- measurement_noise_level = 0.0

## Training Time Budget

- **target ≈ 15 min per iteration** (matches the template
  `claude.training_time_target_min`)
- node: `a100`
- DAL=35, batch_size=4, n_epochs=1 — none of these may be changed

## Parallel Mode — 4 Slots Per Batch

Each batch runs **4 slots simultaneously**:

### Exploration Mode (default)
- Slot 0: CONTROL (all four new coeffs = 0, matches the previous noise-free winner exactly)
- Slots 1-3: each changes **exactly ONE** of the four new coeffs

### Robustness Mode (when validating a promising config)
- All 4 slots: same config, different seeds
- Measures stability across seed variation

**Robustness criteria** (noise-free known_ode):
- **Stable-Robust**: 4/4 slots conn_R2 > 0.977 AND tau_R2 > 0.5 AND CV < 1%
- **Robust**: 4/4 slots conn_R2 > 0.975, tau_R2 ≥ its prior control value, CV 1-3%
- **Partially robust**: 2-3 slots meet targets
- **Fragile / DISQUALIFIED**: any slot conn_R2 < 0.95, or tau_R2 < prior control

> **YAML rule**: Always wrap the `description` field value in double
> quotes — colons inside unquoted YAML strings cause parse errors.

## Block Structure — 6 Blocks × 12 Iterations Each

`n_iter_block=12`, total budget ≈ 72 iterations.

### Block 1 (iter 1-12): τ-L2 dose-response
**Hypothesis**: The identifiability degeneracy in σ=0 keeps tau_R² ≈ 0
because the cost surface is flat along the rescaling direction. A small
L2 on `raw_tau` introduces a preferred scale that should lift tau_R²
without regressing conn_R².

- Slot 0: all reg = 0 (CONTROL)
- Slot 1: coeff_tau_L2 = 1e-5
- Slot 2: coeff_tau_L2 = 1e-4
- Slot 3: coeff_tau_L2 = 1e-3

Expected: either (a) monotonic climb of tau_R² with coefficient until
conn_R² starts to regress, identifying the sweet spot, or (b) flat
tau_R² ≈ 0 regardless of coefficient — which would be the null-result
outcome confirming identifiability as the bottleneck.

### Block 2 (iter 13-24): τ-L1 dose-response
Mirror Block 1 with L1 in place of L2 (sparse prior on `raw_tau`).

- Slot 0: best Block 1 setting (CONTROL)
- Slots 1-3: coeff_tau_L1 ∈ {1e-5, 1e-4, 1e-3}

### Block 3 (iter 25-36): V_rest-L2 dose-response
Same shape as Block 1 but on `coeff_V_rest_L2` (on top of the best τ reg
established so far). Targets the V_rest_R² = 0.905 ceiling.

- Slot 0: best Block 2 setting (CONTROL)
- Slots 1-3: coeff_V_rest_L2 ∈ {1e-6, 1e-5, 1e-4}

### Block 4 (iter 37-48): V_rest-L1 dose-response
Same shape as Block 2 on `coeff_V_rest_L1`.

### Block 5 (iter 49-60): Combinations and interactions
Test the best individual settings combined (e.g. best τ_L2 + best V_rest_L2),
and probe interactions (does L1 + L2 on the same parameter compete?).

### Block 6 (iter 61-72): Robustness validation
**Switch to Robustness Mode**. Re-run the best config across 4 fresh seeds
(use `claude.test_robustness_seed: true` once if possible) and confirm
tau_R² ≥ 0.5 mean AND conn_R² ≥ 0.975.

## File Structure

You maintain THREE files:

1. **Full Log (append-only)**: `flyvis_noise_free_known_ode_reg_Claude_analysis.md`
2. **Working Memory (read + update every batch)**: `flyvis_noise_free_known_ode_reg_Claude_memory.md`
3. **User Input (read every batch, acknowledge pending items)**: `user_input.md`

## Knowledge Base Guidelines

### What to Add to Established Principles
- Observed across 3+ iterations
- Consistent across 4/4 seeds
- States a causal relationship

Example: "coeff_tau_L2 ∈ [5e-5, 5e-4] improves tau_R² by ≥ 0.3 without
regressing conn_R² in noise-free regime (3/3 iters, 4/4 seeds, CV < 1%)."

### What to Add to Open Questions
- 1-2 observations only
- Seed-dependent
- Conflicting evidence

### What to Add to Falsified Hypotheses
- Original hypothesis
- Contradicting iteration / metrics
- Lesson learned
- Revised hypothesis

A negative final result (no reg value rescues tau_R²) is a **valid
publishable finding**, not a failure — document it with the same rigour as
a positive finding.

## Iteration Workflow

### Step 1: Read Working Memory + User Input
Review the comparison table and emerging observations.

### Step 2: Analyze Current Batch Results
For each slot, extract from `analysis.log`:
- `tau_R2` (primary — this is the bottleneck)
- `connectivity_R2` (must not regress)
- `V_rest_R2` (secondary)
- `rollout_pearson_r`
- `training_time_min`

Robustness classification (noise-free known_ode with reg):
- **Stable-Robust**: 4/4 conn_R2 > 0.977 AND tau_R2 > 0.5 AND CV < 1%
- **Robust**: 4/4 conn_R2 > 0.975, tau_R2 ≥ prior control, CV 1-3%
- **Partially robust**: 2-3 slots meet targets
- **Fragile**: 0-1 slots meet targets
- **DISQUALIFIED**: any slot conn_R2 < 0.95, or tau_R2 < prior control

### Step 3: Append Log Entry

```
## Iter N: [Stable-Robust / Robust / Partially robust / Fragile / DISQUALIFIED]
Hypothesis tested: "[quoted hypothesis]"
Slot 0: coeff_tau_L1=A, coeff_tau_L2=B, coeff_V_rest_L1=C, coeff_V_rest_L2=D
        → tau_R2=X, conn_R2=Y, V_rest_R2=Z, time=T min
Slot 1: ...
Slot 2: ...
Slot 3: ...
Seed stats: mean_tau_R2=X, std=Y, CV=Z%, min=W, max=V
Verdict: [supported / falsified / inconclusive]
Next: [planned next batch]
```

### Step 4: Update Working Memory

Append row to "Results Comparison Table", update "Emerging Observations".

### Step 5: Design Next 4 Configs

Single-parameter mutations versus the current control.

## Block Boundaries — Winner Config (COMPULSORY)

**At every block end** (iterations 12, 24, 36, 48, 60, 72):

1. Identify best iteration (highest tau_R² with no conn_R² regression vs the
   prior winner, i.e. conn_R² ≥ 0.977). If no iteration improves tau_R²
   above ~0.05, record a **null winner** with the same CONTROL config as
   slot 0 and state the null finding explicitly.
2. Copy that iteration's config from
   `log/Claude_exploration/LLM_flyvis_noise_free_known_ode_reg/config/iter_XXX_slot_YY.yaml`
3. Save to `config/fly/flyvis_noise_free_known_ode_reg_winner.yaml` with header:

```yaml
# Winner config: flyvis_noise_free_known_ode_reg_winner.yaml
# Source: iter_XXX_slot_YY (tau_R2 = X.XXX, conn_R2 = X.XXXX, V_rest_R2 = X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [1-2 sentence narrative; if null result, state it clearly]
#   - [the four reg coeffs, vs prior winner all-zero]
#
# Metrics (4-seed mean ± std):
#   tau_R2  : X.XXX ± Y.YYY
#   conn_R2 : X.XXX ± Y.YYY
#   V_rest_R2: X.XXX ± Y.YYY
#   rollout_pearson: X.XXX ± Y.YYY
```

## Start Call

When prompt says `PARALLEL START`:

- Read `flyvis_noise_free_known_ode_reg.yaml` template
- All four new coeffs start at 0 (matches prior noise-free winner exactly = control)
- Initialize 4 slot configs for **Block 1 (τ-L2 dose-response)**
- **Initial hypothesis**: "In σ=0 dynamics τ is under-identified by a
  scaling degeneracy, so τ_R² ≈ 0 regardless of optimizer. A non-zero
  coeff_tau_L2 ∈ [1e-5, 1e-3] should break the degeneracy by anchoring
  `raw_tau` near 0, lifting tau_R² above 0.1 without regressing conn_R²
  below 0.975."
- Slot 0 = baseline (all four reg = 0)
- Slots 1-3 = coeff_tau_L2 ∈ {1e-5, 1e-4, 1e-3}, all other new coeffs = 0

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_free_known_ode_reg

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Results Comparison Table

| Iter | Slot | tau_L1 | tau_L2 | Vrest_L1 | Vrest_L2 | tau_R2 | conn_R2 | Vrest_R2 | rollout_r | time_min |
| ---- | ---- | ------ | ------ | -------- | -------- | ------ | ------- | -------- | --------- | -------- |

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
[τ_L2 dose-response]

### Block 2 Summary
[τ_L1 dose-response]

### Block 3 Summary
[V_rest_L2 dose-response]

### Block 4 Summary
[V_rest_L1 dose-response]

---

## Current Block (Block N)

### Block Info
- Iterations: N-N+12
- Focus: [tau_L2 / tau_L1 / V_rest_L2 / V_rest_L1 / combinations / robustness]

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
