# FlyVis GNN — SPEND combined (replay + time-permute + typed-equiv)

## Goal

Combine all three SPEND add-ons (#1 time-permute, #2 typed-equiv, #3 replay)
to maximise the conn_R2 gain over the **baseline ceiling = 0.7457** under
dual noise (σ=0.05, γ=0.10). Target: **conn_R2 > 0.85** (oracle at γ=0.10
is ~0.78; the GNN has the architectural capacity to exceed it once the
loss-shape and identifiability barriers are both attacked).

Run **after** the three single-add-on explorations have each established a
champion config. This exploration tunes the **ratios** between the three
losses and the joint interaction with standard regularizers.

Cite: https://github.com/buchenglab/SPEND  (Ding et al. 2025, Newton 1, 100195).
Bottleneck analysis: `docs/measurement_noise_bottleneck.pdf` §7.4
(recommended order and integration with the agentic loop).

## Scientific Context: Why combine

Each add-on attacks a different facet of the bottleneck:
- **#3 replay** — strongest noise reduction (clean signal available; two
  fully-independent noise views) but only attacks barrier (A) noise wall,
  not (B) symmetry.
- **#1 time-permute** — same noise-reduction mechanism as #3 but cheaper;
  same orientation (attacks A only).
- **#2 typed-equiv** — only add-on whose gradient has a non-zero
  component on the W↔f_theta orbit; attacks (B-4) directly.

**Hypothesis:** combining (A)-attackers (#3 + #1) with (B)-attacker (#2)
should yield super-additive gains because the two barrier types are
orthogonal. The replay smoother feeds clean traces to the GNN; the typed-
equiv loss simultaneously breaks the slope-shrunk minimum's symmetry. Time-
permute is largely redundant with replay (same mechanism, weaker version)
and may be set to 0 or a small fraction (0.1–0.3 of replay weight).

## Training Mode

`data_train_spend(config, ...)`. **Training from scratch**, **1 epoch**,
~70–90 min/iteration on a100 (slightly slower than single-add-on due to
smoother forward + typed-pair gather).

## Data

`generate_data: false`. All slots reuse `fly/flyvis_noise_005_010`.
With `spend_load_clean: true`, the trainer loads with
`measurement_noise_level=0` and synthesises noise inline; this is required
for replay. **DO NOT** modify simulation parameters.

## Noise Model

Identical to `flyvis_noise_005_010` (σ_dyn=0.05, γ_meas=0.10).

## FlyVis Model

13,741 neurons, 65 cell types, 434,112 edges. **Do not change** architecture.

## Explorable Parameters

### SPEND combined knobs (PRIMARY)

| Parameter                       | Default | Safe range       | Notes |
|---------------------------------|---------|------------------|-------|
| `coeff_spend_replay`            | 1.0     | 0.0–3.0          | Anchor of the (A)-attack |
| `coeff_spend_time`              | 0.5     | 0.0–1.0          | Often redundant with replay |
| `coeff_spend_typed`             | 0.5     | 0.1–3.0          | The (B)-attacker; do not set to 0 |
| `spend_smoother_hidden`         | 32      | 16, 32, 64       | Shared by replay + time-permute |
| `spend_smoother_lr`             | 1e-3    | 1e-4 – 3e-3      | Smoother param-group LR |
| `spend_time_window`             | 16      | 8–32             | Time-permute window |
| `spend_typed_max_pos_dist`      | 5.0     | 1.0–10.0         | Pair-construction threshold |

Locked: `spend_load_clean: true` (required for replay);
`spend_replay_noise_seed_a: 0`; `spend_replay_noise_seed_b: 1`.

### Standard GNN levers (compatible)

Same as the single-add-on instructions. Note: under the combined regime
`coeff_g_phi_diff` may be reducible (typed-equiv covers some of its role).

**Locked / do not touch:** `recurrent_training` (false), `n_epochs` (1),
`pretrained_model` (empty), simulation params, architecture.

## Slot Strategy — 4 Different Configs Per Batch

Each batch tests **4 distinct configs**.

### Config Files

- Edit all 4: `flyvis_noise_005_010_spend_combined_00.yaml` through
  `flyvis_noise_005_010_spend_combined_03.yaml`.

## Evaluation

Metrics from `analysis.log` (PRIMARY: `connectivity_R2`).

Per-iteration metrics from `tmp_training/metrics.log` (standard 6-column
schema): `iteration, connectivity_r2, vrest_r2, tau_r2, hidden_nnr_pearson,
anchor_nnr_pearson`.

**SPEND-specific** per-iter from `tmp_training/spend_components.log`:
`iteration, loss_main, loss_replay, loss_time, loss_typed`. **All four
scalars are populated in combined mode** -- the HPO agent must read this
file to track per-add-on contributions.

Progress-bar abbreviations (20-iter EMAs of per-batch losses):

| Bar label | File column | Meaning |
|-----------|-------------|---------|
| `conn=`   | connectivity_r2 | R² of learned W |
| `Vr=`     | vrest_r2 | R² of V_rest |
| `tau=`    | tau_r2 | R² of τ |
| `rep=`    | loss_replay | Add-on #3 N2N MSE (smoother on synth-noise pair) |
| `tim=`    | loss_time   | Add-on #1 N2N MSE (even/odd time-permute) |
| `typ=`    | loss_typed  | Add-on #2 noise-cancelled estimator |

Diagnostics for the HPO agent (combined regime):
- All three loss columns should each drop monotonically at their own pace.
  If any one collapses to zero in the first 1k iters and stays there, that
  add-on is not contributing — consider increasing its coefficient or
  declaring it redundant in the combined regime.
- The hypothesis being tested is **super-additivity**: combined `conn_R2`
  should exceed `max(replay_solo, time_solo, typed_solo)` by ≥ 0.03. If
  not super-additive, the (A)-attackers (#1, #3) and (B)-attacker (#2) are
  not orthogonal in this dataset; reconsider weights.
- Watch for `loss_replay` and `loss_time` redundancy: if `tim=` is always
  ~ `rep=` × constant, time-permute is contributing nothing on top of
  replay; set `coeff_spend_time = 0`.

## Known Results (priors)

- **Baseline**: conn_R2 = 0.7457 ± 0.0043.
- **Oracle at γ=0.10**: ≈ 0.78.
- **Single-add-on champions** (to be filled in from prior explorations):
  - replay champion: conn_R2 = ?, coeff_replay = ?, smoother_h = ?, smoother_lr = ?
  - time-permute champion: conn_R2 = ?, coeff_time = ?, window = ?
  - typed champion: conn_R2 = ?, coeff_typed = ?, max_dist = ?

**REQUIRED before Block 1:** read the three single-add-on
`*_winner.yaml` files and copy their winners into the combined start
hypothesis.

## Block Structure

### Block 1 (iter 1–8): Replay-only re-validation + typed-only re-validation

Sanity-check that both single-add-on champions still work in the *combined
codepath* (some bookkeeping differs).
- Slot 0: replay-champion config, `coeff_typed=0`, `coeff_time=0`
- Slot 1: typed-champion config, `coeff_replay=0`, `coeff_time=0`
- Slot 2: time-permute-champion config, `coeff_replay=0`, `coeff_typed=0`
- Slot 3: all three on at default (1.0, 0.5, 0.5)

If any single-add-on slot regresses by > 0.02 vs its solo champion,
**stop** and debug the trainer integration before continuing.

### Block 2 (iter 9–16): Replay + typed pairwise sweep

Fix smoother config at replay-champion. Sweep:
- Slot 0: `coeff_replay=1.0, coeff_typed=0.0` (replay only, control)
- Slot 1: `coeff_replay=1.0, coeff_typed=0.5`
- Slot 2: `coeff_replay=1.0, coeff_typed=1.0`
- Slot 3: `coeff_replay=1.0, coeff_typed=2.0`

Goal: find typed weight where (A)+(B) attacks combine super-additively.

### Block 3 (iter 17–24): Add time-permute on top of best Block-2

- Slot 0: Block-2 best, `coeff_time=0` (control)
- Slot 1: Block-2 best, `coeff_time=0.1`
- Slot 2: Block-2 best, `coeff_time=0.3`
- Slot 3: Block-2 best, `coeff_time=1.0`

Hypothesis: time-permute is redundant with replay; expect Slot 0 to win or
near-win. If a non-zero time slot strictly beats Slot 0, time-permute is
contributing additional information (e.g. via the half-frame interpolation
acting as a regularizer).

### Block 4 (iter 25–32): Reduce standard regularization

With combined SPEND on, the `coeff_g_phi_diff` term may be redundant.
- Slot 0: combined-winner, `coeff_g_phi_diff=2000` (control)
- Slot 1: combined-winner, `coeff_g_phi_diff=1200`
- Slot 2: combined-winner, `coeff_g_phi_diff=600`
- Slot 3: combined-winner, `coeff_g_phi_diff=0`

Hypothesis: typed-equiv subsumes the structural role of `g_phi_diff`;
reducing the latter frees up some loss budget.

### Block 5 (iter 33–40): W_L1 + lr_W refine

With cleaner gradients, the W_L1 sweet spot may shift. Sweep
`coeff_W_L1` ∈ {5e-5, 1.5e-4, 3e-4, 5e-4} at the combined champion.

### Block 6 (iter 41–48): Robustness validation

4-seed CV at champion config. CV% < 10% AND mean > 0.85 → declare champion.
If mean is in [0.78, 0.85] but CV% > 15%, run a tighter validation block
before declaring.

### Block 7+ (iter 49+): Stretch

- Larger smoother (`spend_smoother_hidden=64`) at low LR (1e-4) — costs
  ~5% training time, may add 1–2% conn_R2.
- Push `coeff_spend_typed` higher with reduced max_dist (cleaner pairs).
- Try `data_augmentation_loop=40` with combined SPEND.

## Iteration Workflow

(Same Steps 1–5 as the replay instruction.)

## Winner Config (COMPULSORY)

Save best as `config/fly/flyvis_noise_005_010_spend_combined_winner.yaml`
with full YAML comment header.

## File Structure

1. `flyvis_noise_005_010_spend_combined_analysis.md` (full log)
2. `flyvis_noise_005_010_spend_combined_memory.md` (working memory)
3. `user_input.md`

## Block Boundaries

(Same as replay instruction.)

## Knowledge Base Guidelines

### Established Principles examples

- "Replay + typed-equiv combination is super-additive: combined conn_R2
  exceeds max(replay, typed) by > 0.03 (3+ iter, all seeds)"
- "coeff_spend_time = 0 in combined regime — redundant with replay"
- "coeff_g_phi_diff reducible to 600 under combined SPEND without conn_R2
  loss"

## Start Call

When prompt says `PARALLEL START`:

- Read base config: `config/fly/flyvis_noise_005_010_spend_combined.yaml`
- Read the three `*_winner.yaml` from prior explorations and the SPEND
  combined section in `docs/measurement_noise_bottleneck.pdf` §7.4
- Read the SPEND header comment in `graph_trainer_spend.py`
- Initial hypothesis: **"Combining replay + typed-equiv at their solo
  champion coefficients yields super-additive conn_R2 gains because the
  two add-ons attack orthogonal barriers (A: noise wall, B: symmetry).
  Block 1 re-validates each solo champion in the combined codepath
  before the joint sweep."**
- Set 4 slots per Block 1 spec — pull champion params from the three
  `*_winner.yaml` files

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_010_spend_combined

## Paper Summary (update at every block boundary)

- **GNN optimization**: [pending]
- **LLM-driven exploration**: [pending]
- **Combined SPEND specifics**: [pairwise super-additivity evidence,
  best (replay, typed, time) ratio, redundancy patterns with standard
  regularizers]

## Knowledge Base

### Results Comparison Table

| Iter | Slot | replay | time | typed | smoother_h | g_phi_diff | W_L1 | conn_R2 | tau_R2 | rollout_r | time_min |
| ---- | ---- | ------ | ---- | ----- | ---------- | ---------- | ---- | ------- | ------ | --------- | -------- |

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

Common combined-mode failure modes:
- All three coefficients large (≥1.0 each) → SPEND losses dominate; main
  MSE underweighted; conn_R2 drops.
- Block 1 sanity check fails (single-add-on regresses in combined trainer)
  → debug `data_train_spend` integration before more iterations.
- Memory pressure: replay's two `(T, N)` noise tensors + smoother forward
  + typed-pair index → ~10 GB on a100 80 GB. If OOM, reduce
  `data_augmentation_loop` or `batch_size`.
