# Rebuttal data sheet — numbers and tables only

Prose is yours to write. This file holds only measured results, in each
reviewer's own question order. Paste-ready markdown pipe tables (≤6 cols).

Source: `neurips_review/results_table.csv`, `figures/flyvis/*_fullsample.tex`,
`neurips_review/excluded_neurons_summary.csv`.

Legend: ✅ measured · ⏳ running · ❌ not built. All new runs single-seed cv00,
Flyvis-217, σ=0.05, consensus HPs. Reference row: `R²_W = 0.99` (Tab. 1 low noise).

---

# REVIEWER 1

## Q1 — hypothesis class + Δt mismatch

τ / V_rest cells throughout Q1: **inlier (full-sample) [excluded %]**, same
convention as Q2. R²_W, slope, r and cluster are unfiltered.

### Q1a Finer generator step ✅
Generator at `h = Δt/M`; model observes **and** predicts at Δt = 20 ms.
Target = observed finite difference `(v(t+Δt)−v(t))/Δt`. Median τ = 19.8 ms.

| M | h | GNN R²_W | GNN slope | oracle R²_W | oracle slope |
|---|---|---|---|---|---|
| 1 | 20 ms | 0.970 | 0.965 | 0.955 | 1.025 |
| 2 | 10 ms | 0.864 | 0.796 | 0.873 | 0.854 |
| 5 | 4 ms | 0.822 | 0.722 | 0.844 | 0.826 |
| 10 | 2 ms | 0.812 | 0.712 | 0.835 | 0.819 |

| M | GNN R²_τ | GNN R²_Vrest | GNN 1-step r | GNN rollout r | GNN cluster |
|---|---|---|---|---|---|
| 1 | 0.95 (0.95) [0.0] | 0.84 (0.68) [7.6] | 0.999 | 0.999 | 0.887 |
| 2 | 0.94 (0.94) [0.1] | 0.48 (−0.39) [14.1] | 0.985 | 0.981 | 0.889 |
| 5 | 0.91 (0.91) [0.2] | 0.62 (−0.62) [20.1] | 0.976 | 0.978 | 0.825 |
| 10 | 0.89 (0.88) [0.3] | 0.68 (−0.59) [27.0] | 0.972 | 0.977 | 0.845 |

| M | oracle R²_τ | oracle R²_Vrest | oracle 1-step r | oracle cluster |
|---|---|---|---|---|
| 1 | 0.98 (0.97) [0.0] | 0.92 (0.67) [8.6] | 0.999 | 0.891 |
| 2 | 0.95 (0.95) [0.1] | 0.72 (−0.35) [18.8] | 0.988 | 0.895 |
| 5 | 0.92 (0.92) [0.1] | 0.65 (−0.69) [22.4] | 0.979 | 0.895 |
| 10 | 0.91 (0.91) [0.2] | 0.64 (−0.78) [23.0] | 0.976 | 0.892 |

Plain-text fallback: `M=1 (h=20ms) GNN 0.970/slope 0.965, oracle 0.955/1.025 | M=2 (10ms) 0.864/0.796, 0.873/0.854 | M=5 (4ms) 0.822/0.722, 0.844/0.826 | M=10 (2ms) 0.812/0.712, 0.835/0.819. GNN one-step r 0.999/0.985/0.976/0.972.`

Notes: R² −0.16, slope −0.25 → calibration loss (Euler truncation), not structure
loss. Oracle degrades equally. One-step r ≥ 0.97 throughout.

### Q1b Coarser observation cadence ✅ oracle / ⏳ GNN
Generator fixed at h = 2 ms. K-step BPTT; Δt_obs = 20K ms. Stimulus stays at 20 ms.

| Δt_obs | /τ | oracle R²_W | oracle slope | GNN R²_W | GNN slope |
|---|---|---|---|---|---|
| 20 ms (K=1) | 1 | 0.835 | 0.819 | 0.816 | 0.745 |
| 40 ms (K=2) | 2 | 0.711 | 0.723 | 0.720 | 0.717 |
| 100 ms (K=5) | 5 | 0.449 | 0.581 | ⏳ | ⏳ |
| 200 ms (K=10) | 10 | 0.287 | 0.490 | ⏳ | ⏳ |

| Δt_obs | oracle R²_τ | oracle R²_Vrest | oracle 1-step r | oracle rollout r | oracle cluster |
|---|---|---|---|---|---|
| 20 ms | 0.91 (0.91) [0.2] | 0.64 (−0.78) [23.0] | 0.976 | 0.974 | 0.890 |
| 40 ms | 0.92 (0.90) [0.4] | 0.58 (−0.91) [24.1] | 0.972 | 0.983 | 0.870 |
| 100 ms | 0.88 (0.86) [0.8] | 0.60 (−1.43) [32.5] | 0.961 | 0.986 | 0.839 |
| 200 ms | 0.80 (0.74) [2.2] | 0.63 (−2.25) [42.4] | 0.940 | 0.980 | 0.784 |

GNN full rows: K=1 R²_τ 0.88 (0.88) [0.3], R²_Vrest 0.58 (−0.49) [18.6], 1-step
0.971, rollout 0.975, cluster 0.789. K=2 R²_τ 0.94 (0.92) [0.3], R²_Vrest 0.52
(−0.52) [23.0], 1-step 0.963, rollout 0.981, cluster 0.819.

⏳ K=5 (job 153172777) and K=10 (153172779) still RUN at 22 h, still at epoch-0
checkpoints; K=10 was diverging at last inspection (R²_W peaked 0.138 at 32k iter,
0.044 at 240k; oracle reaches 0.287). Sharpest degradation in the whole battery.
Rollout r *rises* with K (0.974→0.986) while R²_W falls 0.835→0.287.

### Q1c Term outside the additive-pairwise form ✅
`τ_i v̇_i = −v_i + V_rest_i + Σ_j W_ij ReLU(v_j) − g_a·c_i + I_i`,
`ċ_i = (v_i − c_i)/τ_a`, τ_a = 200 ms, c_i never observed.

| g_a | GNN R²_W | GNN slope | oracle R²_W | oracle slope |
|---|---|---|---|---|
| 0 | 0.987 | 0.982 | 0.986 | 0.998 |
| 0.1 | 0.974 | 0.983 | 0.972 | 0.994 |
| 0.3 | 0.909 | 0.960 | 0.923 | 0.988 |

| g_a | GNN R²_τ | GNN R²_Vrest | GNN rollout r | GNN cluster |
|---|---|---|---|---|
| 0 | 0.99 (0.99) [0.0] | 0.97 (0.63) [5.0] | 1.000 | 0.917 |
| 0.1 | 0.99 (0.99) [0.0] | 0.80 (0.62) [5.7] | 0.998 | 0.910 |
| 0.3 | 0.98 (0.98) [0.0] | 0.54 (−0.35) [27.7] | 0.979 | 0.911 |

Oracle: R²_τ 1.00 (1.00) [0.0] / 0.99 (0.99) [0.0] / 0.97 (0.97) [0.0];
R²_Vrest 0.95 (0.95) [0.0] / 0.81 (0.80) [0.3] / 0.56 (−0.06) [19.6].

Ordering (g_a 0 → 0.3): V_rest breaks first (inlier 0.97 → 0.54, full-sample
0.63 → −0.35, excluded 5.0% → 27.7%), W next (0.987 → 0.909), τ last and nearly
untouched (0.99 → 0.98). Oracle degrades in the same order (−0.39 / −0.063 / −0.03).
g_a = 0 sanity gate reproduces base.

## Q2 — full-sample R² next to inlier

Cell = **inlier (full-sample) [excluded %]**, mean over 5 folds.

### Tab. 1 ✅
| model | σ | R²_τ | R²_Vrest |
|---|---|---|---|
| Known-ODE | 0 | 1.00 (0.09) [1.9] | 0.97 (0.86) [10.2] |
| Known-ODE | 0.05 | 1.00 (1.00) [0.0] | 0.99 (0.95) [4.7] |
| Known-ODE | 0.5 | 1.00 (1.00) [0.0] | 1.00 (1.00) [0.0] |
| GNN | 0 | 0.94 (−11.80) [3.1] | 0.76 (−0.39) [13.7] |
| GNN | 0.05 | 0.99 (0.99) [0.0] | 0.93 (0.76) [3.3] |
| GNN | 0.5 | 1.00 (1.00) [0.0] | 0.98 (0.97) [0.0] |

GNN noise-free R²_τ full-sample SD across folds = ±14.81.

### Supp. Tab. 4 — GNN degradations ✅
| condition | R²_τ | R²_Vrest |
|---|---|---|
| low model noise σ=0.05 | 0.99 (0.99) [0.0] | 0.93 (0.76) [3.3] |
| low meas. noise γ=0.1 | 0.28 (0.13) [8.9] | 0.63 (−1.76) [30.4] |
| mid meas. noise γ=0.2 | 0.08 (−0.04) [9.7] | 0.61 (−5.63) [49.1] |
| unknown stimulus (SIREN) | 0.93 (0.92) [0.4] | 0.72 (−0.72) [21.9] |
| +400% null edges | 0.97 (0.97) [0.0] | 0.46 (−0.52) [22.8] |
| −20% edges removed | 0.95 (0.95) [0.0] | 0.74 (−0.65) [18.3] |
| −50% edges removed | 0.93 (0.90) [0.7] | 0.62 (−3.58) [45.2] |
| 1/5 frames | 0.83 (0.35) [7.1] | 0.49 (−5.09) [56.3] |
| 10% hidden (no NGP) | 0.94 (−0.87) [1.8] | 0.62 (−149.37) [21.1] |
| 20% hidden (no NGP) | 0.93 (0.70) [2.4] | 0.49 (−2.58) [32.5] |

10% hidden SDs: R²_τ ±3.27, R²_Vrest ±288.90.

### Supp. Tab. 7 — Known-ODE conditions ✅
| condition | R²_τ | R²_Vrest |
|---|---|---|
| noise-free | 1.00 (0.09) [1.9] | 0.97 (0.86) [10.2] |
| low model noise | 1.00 (1.00) [0.0] | 0.99 (0.95) [4.7] |
| high model noise | 1.00 (1.00) [0.0] | 1.00 (1.00) [0.0] |
| low meas. noise γ=0.1 | 0.31 (0.17) [7.9] | 0.88 (−1.32) [49.1] |
| mid meas. noise γ=0.2 | 0.06 (−0.04) [9.7] | 0.89 (−2.95) [67.9] |
| +400% null edges | 1.00 (1.00) [0.0] | 1.00 (0.64) [11.8] |
| −20% edges removed | 0.99 (0.96) [0.3] | 0.94 (−1.14) [31.5] |
| −50% edges removed | 0.96 (0.88) [0.9] | 0.92 (−4.16) [57.8] |

### Tab. 2 ✅ built
`figures/flyvis/cv_table_flywireRF_fullsample.tex` (written 2026-07-25 02:26) via
`aggregate_flywireRF_table_fullsample.py`, which carries per-row `n_neurons`
(13,741 / 50,412) — the earlier hardcoded-`_N_NEURONS_BLANK50` blocker does not
apply to this aggregator, and 40.8% reproduces exactly. GNN rows:

| condition | eye map | R²_τ | R²_Vrest |
|---|---|---|---|
| het. RF | hex | 0.99 (0.97) [0.2] | 0.86 (−0.46) [7.3] |
| het. RF + uncert. edges | hex | 0.96 (0.86) [1.4] | 0.67 (−1.37) [24.7] |
| het. RF | FlyWire | 0.99 (0.99) [0.0] | 0.93 (0.77) [3.7] |
| het. RF + uncert. edges | FlyWire | 0.95 (0.38) [1.0] | 0.61 (−9.90) [40.8] |

Known-ODE rows all hold up (R²_Vrest full-sample 0.85–0.94).

### ⚠️ δ_Vrest inconsistency — decision needed ✅ verified
Recorded `V_rest_n_outliers` reproduces at:

| run family | δ_Vrest that reproduces | paper states |
|---|---|---|
| GNN (`_unified`, `full_eye`) | **0.2** | 0.2 |
| Known-ODE (`_known_ode`) | **0.1** | 0.2 |

Evidence (KO γ=0.2 cv00): recorded n_out 9296; @0.1 → 9296, inlier R² 0.8942,
slope 0.8840 (all three match to 4 dp); @0.2 → 5681, inlier R² 0.5654. Full-sample
R² (−2.9073) and rel-err median (0.3327) match at both, i.e. **the full-sample
tables above are unaffected**. Cause: KO runs predate commit `64fd6b9` "unify
outlier thresholds". Effect: KO V_rest inlier columns are filtered 2× tighter than
GNN → flatters the oracle in Tab. 1's V_rest comparison.
Options: (1) disclose + lean on full-sample; (2) recompute KO V_rest inlier at 0.2
(no retraining, ~1 h).

### Q2b Excluded-neuron characterisation ✅ all 5 conditions
δ detected per run by matching the recorded count. **All five reproduce the
paper's excluded-% exactly**, so these are the paper's own masks.

| condition | table | N | excl % (paper) | δ | @0.1 | @0.2 |
|---|---|---|---|---|---|---|
| full-eye prox. nulls (GNN) | Tab. 2 | 50,412 | 40.81 ±2.37 (40.8) | 0.2 | 61.8 | 40.8 |
| 1/5 frames (GNN) | Supp. 4 | 13,741 | 56.31 ±3.41 (56.3) | 0.2 | 78.6 | 56.3 |
| mid meas. γ=0.2 (GNN) | Supp. 4 | 13,741 | 49.05 ±2.44 (49.1) | 0.2 | 71.3 | 49.1 |
| low meas. γ=0.1 (KO) | Supp. 7 | 13,741 | 49.13 ±0.39 (49.1) | 0.1 | 49.1 | 28.8 |
| mid meas. γ=0.2 (KO) | Supp. 7 | 13,741 | 67.88 ±0.14 (67.9) | 0.1 | 67.9 | 41.8 |

τ excluded %: 1.0 / 7.1 / 9.7 / 7.9 / 9.7.

**Activity of excluded vs kept** (median over neurons, mean over folds):

| condition | mean v | voltage SD | frac v>0 |
|---|---|---|---|
| full-eye prox. nulls (GNN) | 0.555 / 0.414 | 0.240 / 0.183 | 0.991 / 0.998 |
| 1/5 frames (GNN) | 0.243 / 0.529 | 0.258 / 0.199 | 0.883 / 1.000 |
| mid meas. γ=0.2 (GNN) | 0.167 / 0.524 | 0.288 / 0.192 | 0.808 / 1.000 |
| low meas. γ=0.1 (KO) | 0.563 / 0.304 | 0.253 / 0.228 | 0.999 / 0.983 |
| mid meas. γ=0.2 (KO) | 0.509 / 0.344 | 0.256 / 0.192 | 0.973 / 0.993 |

Only invariant across all five: **excluded neurons have higher voltage SD**
(0.24–0.29 vs 0.18–0.23). The mean-v / active-fraction signature is *not*
consistent — the two Flyvis-217 GNN rows exclude weakly-driven neurons
(0.17–0.24 vs ~0.52), the Known-ODE and full-eye rows exclude the *more* active
ones. Caveat: KO rows use δ=0.1 and GNN rows δ=0.2, so the two families are not
directly comparable.

**Appx-C degenerate-set overlap** (set = in-degree > 45):

| condition | degen % | Jaccard | random | enrichment |
|---|---|---|---|---|
| full-eye prox. nulls (GNN) | 66.6 | 0.371 | 0.339 | 1.07× |
| 1/5 frames (GNN) | 24.8 | 0.271 | 0.208 | 1.24× |
| mid meas. γ=0.2 (GNN) | 24.8 | 0.271 | 0.197 | 1.29× |
| low meas. γ=0.1 (KO) | 24.8 | 0.180 | 0.197 | **0.93×** |
| mid meas. γ=0.2 (KO) | 24.8 | 0.231 | 0.222 | 1.03× |

→ enrichment 0.93–1.29×, i.e. **at chance**. The excluded set is *not* the
columnar degenerate set; filtering and Appx. C degeneracy are separate phenomena.

**Cross-fold consistency** (excluded in all 5 folds / excluded ever):
full-eye 11986/30247 = 40%; 1/5 frames 4955/10333 = 48%; γ=0.2 GNN 4809/8593 =
56%; γ=0.1 KO 5894/7770 = 76%; γ=0.2 KO 8451/10142 = 83%. Exclusion is largely
reproducible, not fold noise.

**Per-cell-type.** Types at 100% exclusion and their median active fraction:

| condition | types at 100% excl | median frac v>0 | enrich ceiling |
|---|---|---|---|
| 1/5 frames | Am, L1, L2, Lawf1 | 0.000 | 1.78× |
| mid meas. γ=0.2 (GNN) | L1, L2, Lawf1 | 0.000 | 2.04× |
| mid meas. γ=0.2 (GNN) | CT1(M10), Mi2, Mi9 | 0.933–1.000 | 2.04× |
| low meas. γ=0.1 (KO) | C3, CT1(M10), L4, Mi1 | 0.882–1.000 | 2.04× |
| low meas. γ=0.1 (KO) | L2, Lawf1 | 0.000 | 2.04× |
| mid meas. γ=0.2 (KO) | CT1(M10), L3, L4, Mi1 | 0.882–1.000 | 1.47× |

Silent types (frac v>0 = 0.000 — never cross ReLU, emit no message, so V_rest is
unconstrained) are excluded at 100% in every condition: **L2 and Lawf1 in all
five**. Full per-type table: `excluded_neurons_by_type.csv`. Enrichment never
exceeds ~2×, i.e. exclusion is broad rather than concentrated in a few types.

Diagnostic: effective rank of *measured* activity at 99% var = 637–677, not
Appx. C's 45, because measurement noise inflates rank — hence the fixed
in-degree > 45 definition of the degenerate set.

## Q3 — noise-free gap ✅ (numbers from Q2 Tab. 1)
GNN vs oracle, noise-free: R²_W 0.89 / 0.96; R²_Vrest inlier 0.76 / 0.97;
**full-sample −0.39 / 0.86**; cluster 0.86 / 0.92. Gap widens on full sample.

## Q4 — agentic HPO ❌ no baseline run
No random / Bayesian search at matched compute exists. Agent loop cost ≈ 99 GPU-h
(Supp. Tab. 9). Nothing to report; concede.

## Q5 — Allier et al. delineation ❌ needs author input
No experiment. Head-to-head not run.

## Q6 — Shiu et al. / BrainTrace ❌ no numbers needed

## W2 — monotonicity priors ✅
| variant | R²_W | slope | R²_τ | R²_Vrest | 1-step r | cluster |
|---|---|---|---|---|---|---|
| control | 0.9867 | 0.9835 | 0.990 | 0.499 | 1.000 | 0.917 |
| μ₀ = 0 | 0.9881 | 0.9901 | 0.990 | 0.879 | 1.000 | 0.928 |
| μ₁ = 0 | 0.9881 | 0.9914 | 0.997 | 0.667 | 1.000 | 0.903 |

Priors are **not load-bearing** at σ=0.05. μ₀ was already 0 in the consensus
config → the μ₀ row is a same-config replicate, not an ablation.
No `g_φ²` ablation run (sign factorisation, not a monotonicity prior — argue
analytically). No oracle row (μ₀, μ₁ are GNN-side; no oracle counterpart).

## W9 — N reconciliation ✅ exact
| quantity | value |
|---|---|
| neurons | 13,741 |
| neurons with in-degree ≥ 1 | **13,697** |
| neurons with in-degree = 0 | **44** |
| neurons with out-degree = 0 | 462 |
| self-loops | 3,724 |
| Σ max(0, d_i − 45) | **115,223** (matches Appx. C exactly) |

13,697 = **no incoming edge**, not "no output" as the reviewer guessed.

## W9 — ±0.00 variability ✅
Two runs of a byte-identical config (control vs μ₀=0): ΔR²_W = 0.0014,
ΔR²_τ = 0.0003, Δcluster = 0.011. Non-deterministic-GPU-reduction floor (lower
bound on seed spread, not the seed-to-seed SD).

---

# REVIEWER Vzfg

## Q1 — strongly recurrent benchmarks ❌ not run
Supporting number already in paper: R²_W 0.89 (σ=0) → 0.99 (σ=0.05), i.e. noise
enriching the activity distribution restores identifiability.

## Q2 — calcium-like observation model ❌ not run
Closest existing proxies: 1/5 frames R²_W 0.30 (temporal blur);
measurement noise γ=0.1 R²_W 0.63, γ=0.2 R²_W 0.38 (shot/read term).

## Q3 — OOD test ❌ not built
Design (no retraining; vary `I_i(t)` only, roll out fixed checkpoints):

| condition | GNN 1-step r | GNN rollout r | oracle rollout r |
|---|---|---|---|
| naturalistic held-out (ID ref) | | | |
| full-field white noise | | | |
| drifting gratings | | | |
| current inj., random 5% | | | |

**Blocker:** reviewer asks "all models" = incl. recurrent MLP + EED. Those
checkpoints **do not exist on this filesystem** — `fig_jacobian_l1_comparison.py`
reads `/groups/saalfeld/home/kumarv4/repos/connectome-gnn` (inaccessible),
`figures/_baseline_cache/` is empty, and the only MLP config held
(`flyvis_noise_005_mlp.yaml`) points at the non-blank50 dataset. Runnable for
GNN + oracle only until checkpoints are obtained or MLP/EED retrained.

## Q4 — GNN + Known-ODE Jacobian ❌ not built
Decision taken: compute on **noise-free blank50** to match Supp. Fig. 18 panels
and colour limits (`flyvis_noise_free_blank50_{unified,known_ode}_cv00`, both
present with `learned_ode_params.pt`).

GT: `J_ij = W_ij·1[v_j>0]/τ_i`, `J_ii = −1/τ_i`, `S_ik = δ_ik/τ_i`.
Oracle: same, analytic in Ŵ, τ̂.
GNN: `J_ij = (∂f_θ/∂m)·Ŵ_ij·2g_φ·(∂g_φ/∂v_j)`, `J_ii = ∂f_θ/∂v_i`,
`S_ik = (∂f_θ/∂I)·δ_ik` — autograd, not finite differences (ReLU kinks).
Evaluate all models on the same 200 held-out frames.

| model | R² vs GT J | Pearson r | off-support mass | sign agreement |
|---|---|---|---|---|
| Known-ODE | | | 0 by construction | |
| GNN | | | 0 by construction | |
| recurrent MLP | | | | |

Cost note: no dense 13,741² Jacobian needed — panels need a 1736×1736 stimulus
block and a 651×651 (3 types × 217) block; on-support entries are 434,112 values
via chain rule. Minutes, not hours.
**Blocker:** MLP row needs `kumarv4` checkpoints (published PDF gives panels, not
arrays).

---

# REVIEWER PVbi

## ¶1 — novelty vs AMAG / self-supervised GNN ❌ needs author input
## ¶2 — Das & Fiete precedent ❌ no numbers needed
Distinction to draw: their degeneracy is recurrence-driven; ours is columnar
redundancy — 217 near-identical motifs, 26.5% of edges near-null by rank argument,
independent of recurrence strength.
## ¶3 — Flyvis is a simulator ❌ no numbers needed
## ¶4 — senses of "mechanism recovery" ✅ table ready
| sense | asserts | our evidence |
|---|---|---|
| exact parameter recovery | θ̂ = θ | not claimed (26.5% edges near-null) |
| recovery up to equivalence | θ̂ agrees off null space | supported at σ>0 |
| counterfactually useful | correct intervention response | supported (edge ablation) |
| predictive equivalence | matches held-out activity | supported **and insufficient** |
## ¶5 — clarity: Flyvis relation, GNN vs oracle ❌ no numbers needed
## Q1 — SIREN = sinusoidal representation network ❌ no numbers needed
## Q2 — 0.9 green threshold: readability aid, no statistical meaning

---

# Cross-thread numeric consistency

- Appx. C: Eq. (11) is an **upper bound**, not exact kernel dim. Quantitative
  claim = rank-based: r = 45 at 99% var → 115,223 near-null directions (26.5% of
  edges). Verified reproducible. R1's Strength S2 endorses the exact-kernel
  reading — correct it in all three threads.
- `resp_model_match.tex` says the oracle ran "under all three conditions". It ran
  for Δt (4), cadence (4), adaptation (3) — **not** monotonicity. Fix to "under
  the temporal and adaptation conditions."
- 40.8% (Tab. 2) is scored on **50,412** neurons, not 13,741.

# Job status
Cluster: 25 jobs — 22 DONE, 3 RUN (cadence GNN K=2/5/10, ~7.4 h elapsed), 0 failed.
Local: excluded-neuron analysis complete (5/5 conditions).

# Not yet built
1. Vzfg Q3 OOD rollouts (GNN + oracle runnable now; MLP/EED blocked)
2. Vzfg Q4 Jacobian (GNN + oracle runnable now; MLP row blocked)
3. ~~Tab. 2 full-sample rows~~ — done, see Q2 above
4. δ_Vrest decision (disclose vs recompute KO inlier at 0.2)
5. R1 Q4 HPO baseline (none exists), Q5 Allier head-to-head
