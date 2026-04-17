# Exploration brief — flyvis_noise_005_010 (measurement-noise code-change loop)

## Objective

Raise connectivity W R² above **0.82** on flyvis_noise_005_010
(σ_intrinsic = 0.05, γ_measurement = 0.10) with ≥ 3-seed stability, **without
degrading** the other recovered unknowns: τ R², V_rest R², clustering
accuracy.

## Pre-block baseline (HPO-only RC winner, `flyvis_noise_005_010_rc_winner.yaml`)

| metric                | mean     | std     |
|-----------------------|----------|---------|
| W R² (primary)        | 0.8023   | 0.0014  |
| τ R²                  | 0.9525   | 0.0010  |
| V_rest R²             | 0.024    | 0.002   |
| clustering accuracy   | 0.84     | 0.01    |
| rollout Pearson r     | 0.939    | —       |
| one-step Pearson r    | 0.98+    | —       |

Five seeds, CV = 0.18 %. This is the target to beat.

## Inverse-problem frame — DO NOT LOSE SIGHT

**Observed**: voltage `v(t)`, stimulus `e(t)`.
**Unknown**: edge weights `W`, time constants `τ_i`, rest potentials
`V_rest_i`, cell types (65 discrete classes).

τ and V_rest are **NOT** learned by a head. They are extracted post-hoc by a
linear fit on the trained `f_theta` MLP near each neuron's rest state. Any
regularizer on `f_theta` that improves the conditioning of that fit is fair
game (see Block 3).

## Ground-truth ceilings for calibration

- **Noise-free (γ = 0) W R² ≈ 0.965** — physical ceiling; no model can do
  better than the information the clean data contains.
- **Known-ODE oracle under γ = 0.10 → W R² ≈ 0.78.** Your job is to *exceed*
  this bound. An oracle is not a ceiling: the GNN carries more structure
  (embedding, phi, recurrent loss) and can reject noise in ways a linear fit
  cannot.

## Already-falsified hypotheses — do NOT retry in Phase S

All found inefficient or catastrophic in the HPO-only exploration. Proposing
them in Phase R is a waste of the 10-minute cap.

- **Derivative smoothing in non-recurrent mode** — catastrophic (W R² → 0.03).
- **Multi-epoch training > 1** — harmful in both 1-step and recurrent.
- **Dale's law ON in non-recurrent mode** — harmful (fine in recurrent).
- **`f_theta_msg_diff > 100`** — catastrophic (W R² → 0.58).
- **`g_phi_norm < 0.9`** — strictly worse across 0.5–2.0 sweep.
- **`W_L1` above 5e-4 or below 5e-5** — sweet spot sharp; do not widen.
- **Batch size / LR / DAL single-variable scaling** — all bounded by the
  derivative-noise SNR; HPO has exhausted this surface.
- **Noise injection during recurrent rollout** (`noise_recurrent_level > 0`)
  — all tested levels hurt or are neutral at best.

## Phase discipline — block-level

| Phase | Cap     | What to do                                                                  |
|-------|---------|-----------------------------------------------------------------------------|
| R     | 10 min  | Read memory + allowlist lit. Optionally stage ONE analysis function. State ONE hypothesis + Phase-S function signature. |
| S     | 10 min  | Stage exactly ONE mechanism function + its pytest under `staging/block_NN/`. Test must print `PASS:` on the full-data cache. |
| C     | 5 min   | 3–10-line wire-up across at most four production files (allow-list below). |
| T     | ~block  | Cluster training across all slots/iterations in the block (automatic). |
| V     | auto    | KEEP or `git revert` the block based on multi-seed triple-check (mean improves, ≥ 75 % seeds better than pre-median, no catastrophe). |

## Block themes — fixed order

1. **Noise removal / denoising** — auxiliary denoising MLP on voltage, EMA
   estimate, smoothed-target consistency loss, learned Kalman-style prior.
2. **Recurrent-training scheme improvements** — truncated BPTT variants,
   teacher-forcing schedules, multi-start weighting, noise-aware unroll.
3. **Identifiability regularization** —
   - `f_theta` linearization prior near V_rest (tightens the post-hoc linear
     fit that extracts τ and V_rest),
   - contrastive embedding loss for cell-type clustering,
   - group-sparsity on W grouped by presynaptic cell type.
4. **Best-of combination** — union of the Phase-C wire-ups KEPT in blocks
   1–3; re-run Phase S tests jointly to check no interaction breaks them,
   then wire the union in a single commit.
5. **Robustness validation** — re-run the combined winner with N ≥ 8 seeds,
   CV < 1 %, leave-one-out ablation: drop any component whose LOO ablation
   is within noise of the full combination.

## Scientific method — non-negotiable

- **ONE hypothesis per block.** No "try A or B".
- **Falsifiable.** The Phase-S test's PASS/FAIL condition must be written
  BEFORE the test runs, in the staged function's docstring.
- **Causal verdict.** ΔW R² ≥ 0.005 AND ≥ 75 % seeds better than pre-median
  AND no seed catastrophically below pre-min − 0.02. Anything else → REVERT.
- **Revert adds to falsified list.** The failure goes into `falsified.md` so
  future blocks don't retry.

## Where to stage code

| What                      | Path                                                                 |
|---------------------------|----------------------------------------------------------------------|
| Mechanism + pytest        | `src/connectome_gnn/LLM_code/staging/block_NN/`                      |
| Optional analysis fns     | `src/connectome_gnn/LLM_code/staging/block_NN/analysis/`             |

Phase-C wire-up allow-list (edits allowed only to these production files):

- `src/connectome_gnn/models/regularizer.py` — add a new `COMPONENT` entry.
- `src/connectome_gnn/models/recurrent_step.py` — recurrent-only blocks.
- `src/connectome_gnn/models/graph_trainer.py` — call-site only, 1–3 lines.
- `src/connectome_gnn/models/neural_gnn.py` — `__init__` + `forward` if
  unavoidable (prefer regularizer.py first).

Any other file requires a human update to this instruction, not an autonomous
edit.

## What counts as progress

Blocks are measured on the primary metric (W R²) with the triple-check
verdict. Secondary metrics (τ R², V_rest R², clustering) must not catastrophe
below baseline min − 0.02; otherwise the block reverts even if W R² improves.
An elegant function that doesn't shift W R² by ≥ 0.005 after training is a
REVERT — the function stays in `staging/` as an artifact and its failure
joins `falsified.md` with its diff attached.

## Tips

- Prefer `regularizer.py::COMPONENTS` additions. They are the shortest
  production-surface edit (one entry in a list; one coefficient key; the
  agent already understands the pattern).
- The `coeff_*` YAML key for your new regularizer will also be tunable by
  later within-block HPO iterations, so mediocre coefficient choices will be
  improved by the existing loop.
- Do not delete existing regularizers. If you suspect one is harmful,
  propose setting its coefficient to 0 in Phase R as a secondary sanity-
  check analysis, but don't ship it in Phase C (that's a separate block).
