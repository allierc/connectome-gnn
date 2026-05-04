# e8_flywireRF + proximal_nulls — V_rest Recovery (noise=0.05, blank50 cv00)

## Goal

Optimize GNN hyperparameters to **recover per-neuron V_rest** on the
`e8_flywireRF_proximal_nulls` connectome (13,741 neurons, 2,418,403 edges) at noise
σ=0.05, **without sacrificing connectivity recovery**. Two coupled targets:

- **PRIMARY: V_rest_no_outliers_R² ≥ 0.90** with **V_rest_outlier_rate ≤ 10%**
  (baseline cv00: V_rest_no_outliers_R² = 0.580, n_outliers = 3885 / 13741 = **28.3%**)
- **HARD FLOOR: W_corrected_R² ≥ 0.90** (baseline cv00: 0.943 — easy to hold)

The loop **must not** sacrifice W to gain V_rest. Any iteration whose best slot drops
W_corrected_R² below 0.90 is DISQUALIFIED for that slot, regardless of V_rest.

## Scientific Context

V_rest is **extracted post-hoc** from the learned `f_θ` MLP — given a neuron's voltage
distribution, V_rest is found by linearizing f_θ near the operating point and computing
where f_θ crosses zero (chord fit on `[μ − 2σ, μ + 2σ]` is the current default).
Because V_rest is downstream of f_θ shape rather than directly fit, the failure mode
is **MLP curvature at the operating point**: when f_θ is steep / curved / has multiple
near-zeros, the chord fit aliases and a few-percent of neurons land far from gt
(the "outliers" in `V_rest_comparison_wo_outliers_*.png`).

The proximal_nulls augmentation expands the edge set ~7.4× (327k → 2.4M), so per-edge
gradient is smaller and W converges before f_θ is fully shaped. This leaves f_θ in a
locally-curved regime → high V_rest outlier rate, even though W is fine.

**Hypotheses to test**:

- **Stronger / longer f_θ regularization smooths the operating point**: try increasing
  `coeff_f_theta_weight_L2` (currently 0.001) toward 5e-3 — 1e-2, and/or raising
  `coeff_g_phi_norm` to widen the f_θ linear region near the bias.
- **Longer DAL gives f_θ more time to converge after W stabilizes**: bump `data_augmentation_loop`
  from 20 to 30–60, watch wall-clock budget (20 ≈ 60 min on a100).
- **Higher lr_embedding sharpens cell-type clustering of a_i** which f_θ uses to
  disambiguate per-neuron V_rest; try 3e-3 — 5e-3.
- **f_θ centering / linearity terms (currently 0)**: `coeff_f_theta_centering`,
  `coeff_f_theta_linearity` exist in the trainer for exactly this; try sweeping
  small values (1e-4 — 1e-2) to enforce f_θ(0) = 0 / linear-near-bias.
- **V_rest direct supervision**: `coeff_V_rest` (currently 0) penalizes f_θ at the
  predicted V_rest; small values (1e-4 — 1e-3) may regularize the operating-point
  geometry without breaking dynamics.

These are **hypotheses to falsify**, not a recipe — the LLM must form one prediction per
slot and change exactly one parameter to test it.

## Noise Model

```
v_i(t+1) = v_i(t) + dt * f(v_i(t), W, a_i, I_i(t)) + ε_i(t)
ε_i ~ N(0, σ)  with  σ = 0.05  (noise_model_level)
```

Noise is added during training data generation only; test rollouts are noise-free.

## Data — DO NOT REGENERATE

`claude.generate_data: false`. The dataset
`fly/e8_flywireRF_proximal_nulls_noise_005_blank50_cv00/` is pre-built and shared with
the 5-fold CV runs from `run_GNN_flywire_blank50.py`. Do **not** modify simulation
parameters; do **not** delete or rebuild `graphs_data/fly/<dataset>/`.

## Metrics

Read from each slot's `results/metrics.txt` after data_plot completes:

- **PRIMARY (maximize)**: `V_rest_no_outliers_R2` (baseline cv00: 0.580)
- **PRIMARY (minimize)**: `V_rest_n_outliers / n_neurons` (baseline cv00: 28.3%)
- **FLOOR (≥ 0.90)**: `W_corrected_R2` (baseline cv00: 0.943)
- **DIAGNOSTIC**: `tau_no_outliers_R2`, `tau_n_outliers`, `rollout_pearson` (all should
  stay ≥ baseline ≈ 0.98 / ≤ 200 / ≥ 0.998 — flag regressions even if V_rest improves)

In training stdout (live):
```
epoch 0/1 | train: ... | conn_R2=0.XXX tau_R2=0.XXX Vr_R2=0.XXX | duration: XXs
```

`Vr_R2` here is the **post-hoc V_rest_R² with outliers** (matches `V_rest_R2` in
metrics.txt, baseline cv00: −1.15) — useful as a live signal but the **figure-of-merit
is V_rest_no_outliers_R² + outlier-rate** read from the final metrics.txt.

### Slot scoring

Define a per-slot composite **only for ranking** (do not fit to it, validate against the
two PRIMARY metrics independently):

```
score = V_rest_no_outliers_R2  −  λ * outlier_rate     with  λ = 0.5
   if W_corrected_R2 < 0.90: score := −∞   (DISQUALIFIED)
```

Goal trajectory: baseline score = 0.580 − 0.5×0.283 ≈ **0.44**; target ≥ 0.85.

### Robustness

Per-iteration causality (slot 0 = control, slots 1-3 = single-HP experiments) — same
rules as flywire conn_R2 explorations. Robustness sweeps of a candidate winner (4
seeds same config) only after EXPLORATION yields a clear winner — keep
robustness blocks rare; the priority is HP search.

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle.

1. **Hypothesize**: a specific predicted effect on V_rest geometry from one HP change.
2. **Design**: change **EXACTLY ONE** HP per slot (causality rule).
3. **Run training**: 5 slots — slot 0 = parent / unchanged control; slots 1-4 each test
   one hypothesis.
4. **Analyze**: rank by `score`, then verify W_corrected_R² floor and tau / rollout
   diagnostics. Disqualify any slot with W_R² < 0.90 even if V_rest improved.
5. **Update**: revise hypotheses only on training results.

**CRITICAL**: only training results validate or falsify hypotheses.

### CAUSALITY RULE (MANDATORY)

If slot k changes more than one HP from its parent, the iteration cannot attribute
cause and **the slot is DISQUALIFIED** (regardless of metrics).

## Model

`signal_model_name: e8_flywireRF` — per-edge MLP `g_phi` + per-node MLP `f_theta` with
ground-truth edge index from the proximal_nulls augmentation. f_θ does **not** absorb
τ / V_rest (unlike `flyvis_A`); both are recovered post-hoc from the learned MLPs, so
their R² panels are meaningful (not trivially zero).

```
g_phi(v_j, embed_j) → message_j           (edge MLP)
sum_j W_ij * g_phi(v_j) → agg_i           (weighted aggregation)
f_theta(v_i, agg_i, embed_i) → dv_i/dt    (node update MLP)
```

- 13,741 neurons, 65 cell types, **2,418,403 edges** (e8 base + proximal nulls)
- 1,736 input neurons (photoreceptors, DAVIS visual stimuli)
- delta_t = 0.02, 64,000 frames
- `g_phi_positive=true` (Dale's-law approximation)

## Allowed parameters (modify only these)

### Architecture

| Parameter       | Default | Notes                                              |
| --------------- | ------- | -------------------------------------------------- |
| `hidden_dim`    | 64      | Width of g_phi / f_theta hidden layers (consensus) |
| `n_layers`      | 4       | Depth of g_phi / f_theta (consensus)               |
| `embedding_dim` | 2       | Per-neuron `a_i` dimension                         |

### Training (V_rest-relevant levers)

| Parameter                 | Default  | Range to explore  | V_rest relevance                                |
| ------------------------- | -------- | ----------------- | ----------------------------------------------- |
| `lr_W`                    | 1.4e-3   | 5e-4 – 2e-3       | smaller may help W finish before f_θ            |
| `lr`                      | 1.8e-3   | 1e-3 – 3e-3       | f_θ / g_φ MLP rate                              |
| `lr_embedding`            | 2.325e-3 | 1e-3 – 5e-3       | sharpen a_i clustering → cleaner per-cell V_rest |
| `data_augmentation_loop`  | 20       | 20 – 60           | 20 ≈ 60 min on a100; longer = more f_θ refinement |
| `coeff_g_phi_diff`        | 750      | 100 – 1500        | g_φ activation regularizer                      |
| `coeff_g_phi_norm`        | 0.9      | 0.5 – 2.0         | output magnitude — widens f_θ linear region     |
| `coeff_f_theta_msg_diff`  | 0        | 0 – 1e-3          | enforces monotonicity of f_θ in agg             |
| `coeff_f_theta_weight_L2` | 0.001    | 1e-3 – 1e-2       | smoother f_θ → cleaner V_rest extraction        |
| `coeff_g_phi_weight_L1`   | 0.28     | 0.1 – 0.4         | g_φ sparsity                                    |
| `coeff_f_theta_weight_L1` | 0.05     | 0.01 – 0.1        | f_θ sparsity                                    |
| `coeff_W_L1`              | 1e-3     | 5e-4 – 2e-3       | hold W tight; do not relax below 5e-4           |
| `coeff_W_L2`              | 1.5e-6   | 1e-6 – 1e-5       | rarely matters                                  |
| `coeff_f_theta_linearity` | 0        | 0 – 1e-2          | enforces linear f_θ near bias                   |
| `f_theta_linearity_warmup_fraction` | 0.3 | 0.1 – 0.5    | when linearity term ramps in                    |
| `coeff_f_theta_centering` | 0        | 0 – 1e-2          | f_θ(0) = 0 prior                                |
| `coeff_V_rest`            | 0        | 0 – 1e-3          | direct V_rest supervision (use sparingly)       |
| `V_rest_warmup_fraction`  | 0.3      | 0.1 – 0.5         | when V_rest term turns on                       |
| `regul_annealing_rate`    | 0.0      | **MUST stay 0.0** | annealing × n_epochs=1 zeros all reg            |
| `w_init_mode`             | randn_scaled | unchanged    | initialization mode                             |

**Hard constraint**: `regul_annealing_rate` MUST be 0.0 with `n_epochs=1` (otherwise
all regularization terms = 0 at epoch 0).

### Forbidden modifications

- ANY `simulation.*` field (data is pre-built; changing these has no effect on the
  pre-generated zarr).
- ANY `graph_model.*` field except `hidden_dim`, `n_layers`, `embedding_dim`.
- `n_epochs` (fixed at 1).
- `seed` (pipeline-controlled: `sim_seed = iter × 1000 + slot`,
  `train_seed = iter × 1000 + slot + 500`).
- `use_gt_edges` (must remain `true`).

> **YAML rule**: always wrap `description:` value in double quotes — colons inside
> unquoted YAML strings are parse errors.

## Wall-clock budget

Target ~60 min per run. Cluster hard-limit 120 min. **Measured: DAL=26 → ~80 min on a100**;
loop default `data_augmentation_loop=20` → ~60 min. If `_interrupted` markers appear in
slot log dirs, reduce DAL next iteration. The 2.4M-edge graph trains ~3× slower than the
e8 base; budget DAL accordingly.

## Per-block sweep plan (10 blocks × 15 iterations × 5 slots)

Each block has a **theme** — slot 0 stays as the running parent (best config from the
preceding block, or consensus baseline in block 1), and slots 1-4 each change **exactly
one** HP from that block's theme. Across 15 iterations the LLM can probe ~60 single-HP
points per theme, narrowing on the best value before moving to the next block.

| Block | Theme                       | Levers (one per slot)                                                                                          |
| ----- | --------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 1     | Baseline + sanity           | reproduce consensus HPs across 5 seeds; flag intra-seed CV; no HP variation                                    |
| 2     | MLP size                    | `hidden_dim` (40 / 80 / 128 / 160), `n_layers` (2 / 3 / 4 / 5), `embedding_dim` (2 / 3 / 4 / 8)                |
| 3     | Learning rates              | `lr_W` (5e-4 — 2e-3), `lr` (1e-3 — 3e-3), `lr_embedding` (1e-3 — 5e-3); plus `lr_scheduler` if monotonic plateaus |
| 4     | W regularization            | `coeff_W_L1` (5e-4 — 2e-3), `coeff_W_L2` (1e-6 — 1e-5); guard W_R² ≥ 0.90 floor                                |
| 5     | Batch                       | `batch_size` (2 / 4 / 8 / 16), `batch_ratio` (0.5 / 1 / 2)                                                     |
| 6     | f_θ regularization          | `coeff_f_theta_weight_L2` (1e-3 — 1e-2), `coeff_f_theta_weight_L1` (0.01 — 0.1), `coeff_f_theta_msg_diff` (0 — 1e-3) |
| 7     | g_φ regularization          | `coeff_g_phi_norm` (0.5 — 2.0), `coeff_g_phi_diff` (250 — 1500), `coeff_g_phi_weight_L1` (0.1 — 0.4)           |
| 8     | f_θ shape priors for V_rest | `coeff_f_theta_centering` (0 — 1e-2), `coeff_f_theta_linearity` (0 — 1e-2), `f_theta_linearity_warmup_fraction`|
| 9     | Direct V_rest supervision   | `coeff_V_rest` (0 — 1e-3), `V_rest_warmup_fraction` (0.1 — 0.5), `V_rest_rampup_iters` (50 — 500)              |
| 10    | DAL + winner robustness     | `data_augmentation_loop` (20 / 30 / 45 / 60), then 4-seed reproducibility on the best block-1..9 winner         |

After each block, slot 0 of the next block is set to the **single best-scoring config
of the closing block** (composite score, W floor enforced). If no slot in the block beat
the entering parent, slot 0 stays unchanged and that block's lever is recorded as
*null lever for V_rest* in the analysis log.

## Stop conditions

- Per-block early-exit: if in any block the best-of-block `score` does not exceed the
  entering parent by ≥ 0.02 within 8 iterations, advance to the next block early.
- Loop-level success: stop the whole loop when target met
  (`V_rest_no_outliers_R² ≥ 0.90` AND `outlier_rate ≤ 10%` AND `W_corrected_R² ≥ 0.90`)
  for two consecutive iterations.

## Final validation

Once a candidate winner is identified, re-evaluate on **all 5 cv folds** by writing a
new winner yaml and running `run_GNN_flywire_blank50.py` with the appropriate
HP_YAML_OVERRIDES entry — that is **outside** the loop's responsibility, but the loop
should print a final summary block tagged `READY_FOR_5CV` once stop conditions are met.
