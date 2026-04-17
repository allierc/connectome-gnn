# FlyVis GNN — Code-Change Exploration (flyvis_noise_005_010_code_change)

This instruction is the **HPO-within-block** brief. It runs the classic
per-iteration analysis (config mutations on coefficients / learning rates /
etc.) operating on top of whatever the **block-level code-change agent**
(Phase R / S / C) has just wired in.

> The block-level brief — which is what drives the code changes — lives at
> `src/connectome_gnn/LLM_code/instructions/instruction_flyvis_noise_005_010.md`
> and is loaded by `GNN_LLM_code.py` at block start, separately from this file.

## Your scope (HPO-within-block only)

You see the iterations of a single block. At block start a new mechanism was
added to production source via an import from
`src/connectome_gnn/LLM_code/staging/block_NN/<name>.py` and a coefficient
key (typically `coeff_<name>`) was exposed in the YAML. **Your job is to tune
that coefficient and the existing levers — NOT to propose structural
changes, new regularizers, or new losses.**

Structural / architectural / new-regularizer ideas belong to the next
block's Phase R, not to this per-iteration analysis.

## Objective

Beat `connectivity_R2 = 0.8023` (HPO-only RC winner) under measurement noise
`γ = 0.10`. Secondary targets: `tau_R2 > 0.95`, `V_rest_R2 ≥ 0` (stretch),
`clustering_accuracy ≥ 0.84`, `rollout_pearson ≥ 0.93`.

## Baseline training setup (do not change outside allow-list)

- **Recurrent fine-tuning** from
  `./log/fly/flyvis_noise_005_010_rc/models/best_model_with_0_graphs_1.pt`
- `recurrent_training: true`, `time_step: 5`, `batch_size: 32`,
  `data_augmentation_loop: 130`, `n_epochs: 2`
- `dale_law: true`, `g_phi_positive: true`, `prediction: first_derivative`

## Noise model (context)

1. Dynamics noise `noise_model_level = 0.05`
2. Measurement noise `measurement_noise_level = 0.10`
   → derivative noise std ≈ 7.07 (measurement_noise · √2 / dt)

Recurrent training bypasses per-frame derivative noise via multi-step rollout
consistency. This is already enabled; do not disable it.

## Levers you MAY tune (per-iteration mutations)

| Parameter | Current | Safe range | Notes |
|-----------|---------|-----------|-------|
| `coeff_<new>` (from this block's wire-up) | see Phase C log | start ±3× | Tune first — this is why the block exists |
| `coeff_g_phi_diff` | 600 | 400–1000 | Known sensitive |
| `coeff_W_L1` | 1e-4 | 5e-5–5e-4 | Sweet spot sharp; widen cautiously |
| `coeff_W_L2` | 1.5e-6 | 0–3e-6 | Weak effect |
| `coeff_f_theta_weight_L1` | 0.05 | 0–0.1 | |
| `coeff_f_theta_weight_L2` | 1e-3 | 0–3e-3 | |
| `lr_W` | 5e-4 | 3e-4–9e-4 | Effective rate clamped by pipeline |
| `lr` | 1.8e-3 | 1e-3–2.5e-3 | |
| `data_augmentation_loop` | 130 | 80–160 | Trades training time for signal |
| `seed` | block-base ± slot | keep distinct per slot | |

## Do NOT propose

Already falsified under measurement noise `γ = 0.10` (do not retry):

- Derivative smoothing (`derivative_smoothing_window > 1`) in non-recurrent
  mode — catastrophic.
- `n_epochs > 2` — harmful.
- `dale_law: false` — worse in recurrent mode.
- `coeff_f_theta_msg_diff > 100` — catastrophic (conn_R² → 0.58 at 200).
- `coeff_g_phi_norm < 0.9` — strictly worse.
- `coeff_g_phi_weight_L1 > 0` — hurts W recovery.
- `noise_recurrent_level > 0` — harmful at all tested levels.
- Changes to architecture (`hidden_dim`, `n_layers`, `embedding_dim`,
  `aggr_type`, `update_type`, `g_phi_positive`, `prediction`).
- Changes to `batch_size` (keep at 32 under recurrent mode).
- Changes to `time_step` (keep at 5).
- Changes to `pretrained_model` — fine-tuning from the same checkpoint.

## Mutation discipline

- ONE parameter per slot per iteration (the CAUSALITY RULE).
- Keep one slot as the control (unchanged config) so the effect of the
  current block's code change + coefficient value is measurable.
- Write your analysis to the usual memory file; the block-level verdict
  agent reads it at block end.

## Output fields the verdict consumes

Please make sure the per-iteration analysis log includes (for at least the
last batch of the block):

- `connectivity_R2` → parsed as `W_R2`
- `tau_R2`, `V_rest_R2`
- `cluster_accuracy` → parsed as `clustering_accuracy`
- `rollout_pearson`, `onestep_pearson`
- `training_time_min`

These are already emitted by `data_plot`; just confirm the parser picked
them up.
