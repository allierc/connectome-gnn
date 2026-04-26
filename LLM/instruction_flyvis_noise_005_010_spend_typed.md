# FlyVis GNN — SPEND Add-on #2 (typed-equivariance N2N)

## Goal

Use SPEND-style **same-cell-type pairs** as Noise2Noise pairs to break the
**conn_R2 = 0.745 ceiling** under dual noise (σ=0.05, γ=0.10).

For each cell type with ≥2 neurons, pair each neuron with its nearest
same-type neighbour within `spend_typed_max_pos_dist` retinotopic distance.
Their clean voltages are approximately equal (typed equivariance under
columnar shift); their measurement-noise realisations are independent. The
loss `‖v_i1 − v_i2‖² − 2γ²` is a **noise-cancelled estimator** that goes to
zero when the GNN respects typed-equivariance and the noise is Gaussian.

This add-on **simultaneously** breaks:
- **(A)** the noise wall (N2N-style noise cancellation), and
- **(B-4)** the structural-prior anchor (typed equivariance is anchor class 4
  in the instruction-file taxonomy — never tested in prior agentic loops).

Primary metric: **connectivity_R2**. Target: **conn_R2 > 0.78**.

Cite: https://github.com/buchenglab/SPEND  (Ding et al. 2025, Newton 1, 100195).
Bottleneck analysis: `docs/measurement_noise_bottleneck.pdf` §7.2.

## Scientific Context: Typed-equiv as N2N + structural anchor

Two same-type neurons receiving stimulus-shifted copies of the same input
share the same clean signal up to a retinotopic time-shift. After
delay-correction (or for spatially co-located pairs where the shift is ~0),
their voltages differ only by independent Gaussian noise of variance 2γ².
Subtracting the bias `2γ²` from `‖v_i1 − v_i2‖²` gives an **unbiased
estimator of the residual cell-type heterogeneity** — which equals zero if
the GNN respects typed equivariance.

Unlike Add-ons #1/#3, this loss has a non-zero gradient component **on the
W↔f_theta orbit**: it injects information *orthogonal to the scale
symmetry*. That makes it the most direct attack on barrier (B), and the only
SPEND variant that does not require a smoother network.

## Training Mode

`data_train_spend(config, ...)` invoked normally. **Training from scratch**,
**1 epoch**, ~60 min/iteration on a100. **No smoother network** in this
add-on (`coeff_spend_replay` and `coeff_spend_time` both 0).

## Data

`generate_data: false`. All slots reuse `fly/flyvis_noise_005_010`.
Trainer loads with disk-saved noise; the typed-equiv loss reads
`x.voltage[typed_pairs]` directly. **DO NOT** modify simulation parameters.

The pair index `(P, 2)` is precomputed once at training start by
`_build_typed_pairs(type_list, x_ts.pos, max_dist)` from the connectome
metadata — no extra dataset preprocessing.

## Noise Model

Identical to `flyvis_noise_005_010` (σ_dyn=0.05, γ_meas=0.10). The
noise-cancelled estimator subtracts exactly `2 * 0.10² = 0.02` from the raw
pair-difference squared norm.

## FlyVis Model

13,741 neurons, 65 cell types, 434,112 edges. **Do not change** architecture.

## Explorable Parameters

### SPEND typed-equiv knobs (PRIMARY)

| Parameter                       | Default | Safe range       | Notes |
|---------------------------------|---------|------------------|-------|
| `coeff_spend_typed`             | 1.0     | 0.01–10.0 (log)  | Weight on typed-equiv loss |
| `spend_typed_max_pos_dist`      | 5.0     | 1.0–20.0         | Max retinotopic distance for pairing |

Locked: `coeff_spend_replay: 0`; `coeff_spend_time: 0`;
`spend_load_clean: false`.

**Pair-count diagnostic.** At `max_pos_dist = 5.0` we expect ~13,000 pairs
(close to 1 pair per neuron). At `max_pos_dist = 1.0`, only direct same-type
column-mates pair (~5,000 pairs). At `max_pos_dist = 20.0` pairs from
distant retinotopic positions enter — these have larger stimulus-shift,
so the noise-cancelled estimator becomes biased. Sweep cautiously.

### Standard GNN levers (compatible)

Same as the replay/time instructions. Note: typed-equiv is anchor class 4
and may make `coeff_g_phi_diff` redundant — test reducing it.

**Locked / do not touch:** `recurrent_training` (false), `n_epochs` (1),
`pretrained_model` (empty), simulation params, architecture.

## Slot Strategy — 4 Different Configs Per Batch

Each batch tests **4 distinct configs**.

**Causality rule (relaxed).** Each slot's config may differ from the current
baseline by **up to TWO parameters** (not strictly one). This enables 2×2
factorial sweeps in a single batch — Slot 0: control; Slot 1: change A;
Slot 2: change B; Slot 3: change A + B. Use only when A and B are
theoretically expected to interact; otherwise prefer one-parameter sweeps.

### Config Files

- Edit all 4: `flyvis_noise_005_010_spend_typed_00.yaml` through
  `flyvis_noise_005_010_spend_typed_03.yaml`.

## Evaluation

Metrics from `analysis.log` (PRIMARY: `connectivity_R2`).

Per-iteration metrics from `tmp_training/metrics.log` (standard 6-column
schema): `iteration, connectivity_r2, vrest_r2, tau_r2, hidden_nnr_pearson,
anchor_nnr_pearson`.

**SPEND-specific** per-iter from `tmp_training/spend_components.log`:
`iteration, loss_main, loss_replay, loss_time, loss_typed`. The HPO agent
MUST read this file.

Progress-bar abbreviations (20-iter EMAs of per-batch losses):

| Bar label | File column | Meaning |
|-----------|-------------|---------|
| `conn=`   | connectivity_r2 | R² of learned W |
| `Vr=`     | vrest_r2 | R² of V_rest |
| `tau=`    | tau_r2 | R² of τ |
| `typ=`    | loss_typed | **Add-on #2** noise-cancelled estimator: ‖v_i1 − v_i2‖² − 2γ² |

Diagnostics for the HPO agent:
- `loss_typed` is **clamped at zero** when the noise-cancelled estimator
  would go negative (`(diff² − 2γ²).clamp(min=0)`). A `typ=0.0000` flatline
  from the very first iteration usually means the noise-floor was already
  hit -- the loss provides no gradient. Increase
  `spend_typed_max_pos_dist` (more pairs → wider distribution of cleaner
  signal) or strengthen the loss with higher `coeff_spend_typed`.
- A monotonically *decreasing* `typ` value while `conn_R2` climbs is the
  desired pattern (the GNN is learning typed-equivariance).
- If `typ` stays high (> 0.1) and never drops, the typed pairs are biased
  (too distant); reduce `spend_typed_max_pos_dist`.

## Known Results (priors)

- **Baseline**: conn_R2 = 0.7457 ± 0.0043.
- **Oracle at γ=0.10**: ≈ 0.78.
- **Anchor class 4 prior tests**: never run in this codebase. The
  `instruction_flyvis_noise_005_010_code_change.md` document flags this as
  one of the highest-leverage untested anchor classes.

## Block Structure

### Block 1 (iter 1–8): SPEND coefficient + max-distance joint sweep

4 slots:
- Slot 0: `coeff=0.1, dist=5.0` (light, default-distance)
- Slot 1: `coeff=1.0, dist=5.0` (default)
- Slot 2: `coeff=1.0, dist=1.0` (column-only, strict)
- Slot 3: `coeff=3.0, dist=10.0` (heavier weight, looser pairing)

Goal: identify productive (coeff, distance) region.

### Block 2 (iter 9–16): Learning-rate sweep

Fix `coeff_spend_typed` and `spend_typed_max_pos_dist` at the Block-1
winner. Sweep `lr`, `lr_W`, and `lr_embedding` jointly. The typed-equiv
loss injects a gradient orthogonal to the W↔f_θ orbit, which can shift
LR sweet spots. Suggested:
- Slot 0: baseline (`lr_W=9e-4`, `lr=1.8e-3`, `lr_embedding=2.325e-3`)
- Slot 1: `lr_W=5e-4`
- Slot 2: `lr=1.0e-3` + `lr_embedding=1.5e-3` (two-param sweep)
- Slot 3: `lr_W=5e-4` + `lr=1.0e-3` (two-param sweep)

### Block 3 (iter 17–24): Distance refine

Fix coefficient at the Block-1/2 winner. Sweep
`spend_typed_max_pos_dist` ∈ {1.0, 2.5, 5.0, 10.0}. Hypothesis: optimum at
~5.0 — too small under-utilises pair count; too large brings biased pairs.

### Block 4 (iter 25–32): Regularization sweep (merged)

Typed-equiv is anchor class 4; it injects information orthogonal to the
W↔f_θ symmetry, so several existing regularizers may become redundant or
need re-tuning. Sweep regularization coefficients jointly using the
up-to-two-params rule. Pool to draw from (pick 2 to vary per slot, others
at baseline):
`coeff_g_phi_diff` (baseline 2000; try 600, 1200, 3000),
`coeff_W_L1` (baseline 1.5e-4; try 5e-5, 5e-4),
`coeff_g_phi_norm` (baseline 0.9; try 0.3, 1.5),
`coeff_g_phi_weight_L1` (baseline 0.28; try 0, 0.5),
`coeff_f_theta_weight_L1` (baseline 0.05; try 0, 0.1),
`coeff_W_L2` (baseline 1.5e-6; try 0, 3e-6).
Suggested 4-slot design:
- Slot 0: control
- Slot 1: vary `coeff_g_phi_diff`
- Slot 2: vary `coeff_W_L1`
- Slot 3: vary `coeff_g_phi_diff` + `coeff_W_L1` together

Hypothesis to test: typed-equiv makes W more identifiable, so a stronger
W L1 (5e-4) may help by pruning spurious connections — and a weaker
`coeff_g_phi_diff` may suffice since structural identifiability now flows
through the typed-equiv loss.

### Block 5 (iter 33–40): Robustness validation

4-seed CV at champion config. CV% < 10% AND mean > 0.78 → declare champion.

### Block 6+ (iter 41+): Stretch

Combine typed-equiv with Add-on #1 (time-permute) — see the combined
instruction file.

## Iteration Workflow

(Same Steps 1–5 as the replay instruction.)

## Winner Config (COMPULSORY)

Save best as `config/fly/flyvis_noise_005_010_spend_typed_winner.yaml`
with full YAML comment header.

## File Structure

1. `flyvis_noise_005_010_spend_typed_analysis.md` (full log)
2. `flyvis_noise_005_010_spend_typed_memory.md` (working memory)
3. `user_input.md`

## Block Boundaries

(Same as replay instruction.)

## Knowledge Base Guidelines

### Established Principles examples

- "spend_typed_max_pos_dist > 10 introduces stimulus-shift bias
  (loss_typed plateau at non-zero floor; conn_R2 stagnates)"
- "coeff_spend_typed = 1.0 redundant with coeff_g_phi_diff > 1500
  (combined effect saturates)"

## Start Call

When prompt says `PARALLEL START`:

- Read base config: `config/fly/flyvis_noise_005_010_spend_typed.yaml`
- Read `docs/measurement_noise_bottleneck.pdf` §7.2 (typed-equiv as
  noise-cancelled estimator) and `_build_typed_pairs` in
  `graph_trainer_spend.py`
- Read `instruction_flyvis_noise_005_010_code_change.md` lines 144–167 for
  anchor-class taxonomy context
- Initial hypothesis: **"coeff_spend_typed = 1.0 with max_dist = 5.0 breaks
  the 0.745 ceiling because typed-equiv injects information orthogonal to
  the W↔f_theta scale symmetry — Block 1 validates the regime."**
- Set 4 slots per Block 1 spec

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_010_spend_typed

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]
- **Typed-equiv specifics**: [productive coefficient range, distance
  threshold, interaction with standard regularizers, evidence for the
  anchor-class hypothesis]

## Knowledge Base

### Results Comparison Table

| Iter | Slot | coeff_typed | max_dist | conn_R2 | tau_R2 | typed_loss | rollout_r | time_min |
| ---- | ---- | ----------- | -------- | ------- | ------ | ---------- | --------- | -------- |

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

Common typed-equiv failure modes:
- `coeff_spend_typed > 10` → loss dominates; GNN ignores main MSE,
  conn_R2 drops.
- `spend_typed_max_pos_dist > 20` → many biased pairs (large
  stimulus-shift); noise-cancelled estimator becomes negative-biased and
  the `clamp(min=0)` zeros the gradient → loss looks fine but no learning.
