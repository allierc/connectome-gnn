# Zebrafish HD — Swim Integration (Known-ODE RNN, dIPN-only readout)

## Goal

Find the best training recipe for `ZebrafishHdTaskRNN` with the
**dIPN-only readout** flag (`graph_model.output_from_dipn_only: true`)
on the swim-integration task. The decoder W_out has shape `(2, 443)`
and reads only from the first 443 dIPN cells (IPNd* + IPNds*, the
r1π HD ring per Petrucco 2023). This is the connectome-constrained
model (`W_rec = |S| ⊙ sign(W_con)`, Dale's law) and is the larval-
zebrafish companion of [config/drosophila_cx/drosophila_cx_pi_epg.yaml](../config/drosophila_cx/drosophila_cx_pi_epg.yaml).

**Primary metric**: `r_roll_1k` — Pearson correlation between the
unwrapped decoded heading and the ground-truth heading on a
deterministic-sweep rollout (ω = 60°/s constant, **T = 1000**,
warmup 10 frames). Target **`r_roll_1k` ≥ 0.95**.

No secondary metrics.

**Dataset (fixed)**: `zebrafish_hd_si_task` (100k train + 10k test,
T=1000 frames, dt=0.01s, sparse Poisson swims; written by
`_generate_swim_integration_task` via ZarrTaskTrialsWriter).

**Why this exploration is needed**: the swim-integration task is
harder than fly path-integration in two ways:

1. **Train/eval distribution mismatch.** Training input ω(t) is a
   sparse stream of typed-swim boxcars (≈0.5 Hz onset rate, 0.3 s
   wide). The deterministic-sweep eval drives the network with a
   *constant* ω across all T=1000 frames — a regime the network
   never sees during training. The recipe must encourage the
   recurrent dynamics to generalise from boxcar-integration to
   constant-input integration.
2. **Larger ring, larger circuit.** 443 dIPN cells across 731 total
   neurons (vs fly 46 EPG / 156). The decoder must push the heading
   code across a much wider population block; aux losses that worked
   on the fly may not transfer one-for-one.

## Budget

**120 iterations**, 10 slots × 4 batches = 40 iter/block, **3 blocks total**.
Matches the yaml's `claude:` block (`n_iter_block: 40`, `n_parallel: 10`).

**Per-iter training**: `n_epochs: 5`, `data_augmentation_loop: 1`,
`n_steps_schedule: [100, 200, 300, 400, 500]` — slow ramp from T=100
up to T=500. Jumping straight to T=500 in one epoch starves the
early bump-formation phase, which the parent yaml's 10-epoch
schedule (`[100, 100, 200, 200, 300, 400, 500, 600, 700, 800]`)
deliberately avoids. Each iter ~30 min per slot on a100
(`training_time_target_min: 30`).

## Parent recipe (iter 0 / B1 slot 0)

```yaml
dataset: zebrafish_hd_si_task
graph_model:
  signal_model_name: zebrafish_hd_si
  aggr_type: add
  input_proj: matrix
  output_proj: matrix
  velocity_gate: pen_4scalar
  lock_edge_signs: true
  hidden_dim: 128
  n_layers: 3
  MLP_activation: relu
  output_from_dipn_only: true        # dIPN HD ring readout (n=443)
training:
  n_epochs: 5
  data_augmentation_loop: 1
  batch_size: 64
  lr: 2.0e-3
  lr_W_rec_schedule: [5.0e-4, 5.0e-4, 2.5e-4, 1.25e-4, 1.25e-4]
  lr_W_ED: 2.5e-3
  grad_clip_W: 2.5
  noise_recurrent_level: 0.03
  n_steps_schedule: [100, 200, 300, 400, 500]   # slow ramp
  coeff_tail_loss: 0.035             # ON from the start
  w_init_mode: w_con
```

## Block plan

### B1 — Stabilisation + tail-loss sweep (iter 1-40)

Tail-loss matters more here than on the fly because the eval-time
constant-ω input drives the bump for the full T=1000 long after any
training boxcar would have decayed. Start with it ON and sweep it
jointly with the stabilisation axes. Establish a parent recipe that
consistently reaches `r_roll_1k ≥ 0.7` on T=500 training.

| axis | mutations to try |
|---|---|
| `coeff_tail_loss` | **{0.0, 0.02, 0.035, 0.05, 0.1}** — primary axis for B1 |
| `lr_W_rec_schedule` (×scale) | {0.25, 0.5, 1, 2, 4} |
| `noise_recurrent_level` | {0.0, 0.01, 0.03, 0.05, 0.1} |
| `grad_clip_W` | {0.0, 1.0, 2.5, 5.0} |
| `lr_W_ED` | {5e-4, 1e-3, 2.5e-3, 5e-3} |

Best slot from each batch becomes the parent of the next. After B1
the `coeff_tail_loss` value is fixed (winner-take-all) for B2 / B3.

### B2 — Curriculum + initialisation (iter 41-80)

Once a stable parent is found, probe how the curriculum interacts
with the train/eval distribution mismatch. The hypothesis: longer
T_epoch in late epochs forces the network to internally generate a
sustained drive — closing the boxcar→constant-ω gap.

| axis | mutations |
|---|---|
| `n_steps_schedule` (length 5) | {[200×5], [300×5], [500×5], [100,200,300,400,500], [200,400,600,800,1000]} |
| `n_epochs` (with matching schedule length) | {3, 5, 8} |
| `w_init_mode` | {const, randn, w_con, zeros} |
| `w_init_scale` | {1e-3, 1e-2, 5e-2, 1e-1, 0.5} |
| `coeff_W_L1` | {0, 1e-5, 1e-4} |

### B3 — Robustness / fine-tune (iter 81-120)

Re-probe the lr × grad_clip × noise grid around the B2 winner;
verify stability across seeds; revisit `coeff_tail_loss` ±50% in
case the B2 changes shifted the optimum. With 10 slots/batch this
block has the budget to mix single-axis fine-tunes with seed-varied
duplicates of the current best.

| slot | mutation around B2 parent |
|---|---|
| 0 | reference (no change) — re-anchor only at the start of B3 |
| 1–3 | `lr_W_rec_schedule` ×{0.7, 1.0 (seed), 1.4} |
| 4–5 | `coeff_tail_loss` ±50% |
| 6–7 | `noise_recurrent_level` shifted ±1 step in the B1 grid |
| 8 | `grad_clip_W` ×0.5 |
| 9 | seed-varied duplicate of B2 parent (different `simulation_seed` / `training_seed`) |

Best by **r_roll_1k** averaged over the last 3 iter snapshots.

## Zebrafish-specific notes (do not silently drop)

- **Task input is fixed**: sparse Poisson swims at 0.5 Hz, 0.3 s
  boxcars, fractions L/R/F/B = 0.40 / 0.40 / 0.15 / 0.05, mean
  |Δθ|_LR ≈ 0.785 rad, mean |Δθ|_B ≈ π. Do **not** touch the
  `task.swim_integration:` block (changes the data distribution and
  would require regenerating the dataset).
- **Eval is OOD by design.** A slot that gets `loss` to ≈0 but
  `r_roll_1k < 0.5` is *not* a bug — it means the network learnt to
  integrate boxcars but cannot generalise to a sustained constant-ω
  drive. This is the central failure mode this exploration is trying
  to defeat; flag it as such in the log entry rather than treating
  it as a training instability.
- **Larger ring → longer per-iter wall-time.** If a slot consistently
  blows past `training_time_target_min: 30`, prefer reducing
  `data_augmentation_loop` or shortening the late-T schedule entries
  before reducing `n_epochs` (which collapses the curriculum).

## Mutation guardrails

- Don't touch `lock_edge_signs`, `velocity_gate`, `wrec_param`,
  `signal_model_name`, `aggr_type`, or `output_from_dipn_only`.
  These are fixed for this variant.
- The dataset is fixed (`zebrafish_hd_si_task`) — claude
  `generate_data: false`. Never call `data_generate`.
- Do **not** modify `task.swim_integration:` (would require a fresh
  data generation pass and break the 100k pre-generated trials).
- Keep `n_steps_schedule` as a multi-epoch slow ramp (length ==
  `n_epochs`). Don't collapse to single-T training.
- `lr_W_rec_schedule` length must match `n_epochs`.
- `connconstr_datapath: figures/zebrafish/zebrafish_connectome_HD`
  is fixed (loader path).

### Anti-repetition rule (important — saves budget)

- **At most ONE slot per batch is the parent-reference**: in B1
  batch 1 that's slot 0. After each batch, the parent for the *next*
  batch is whichever slot won; **do not keep a frozen-baseline
  reference slot across batches**. The full per-epoch trajectory of
  the parent is already in the prior batch's logs, so a fresh
  re-anchor measurement per batch is wasted budget.
- In B2/B3 every slot should be a distinct mutation off the
  (possibly re-chosen) parent. If you want to verify stability of
  the current best, use a seed-varied slot (different
  `simulation_seed` / `training_seed`) rather than a literal
  duplicate of the parent.

## Stop conditions

- **Success**: best slot reaches `r_roll_1k ≥ 0.95` for two
  consecutive iterations.
- **Halt**: best slot stays below `r_roll_1k = 0.3` after B2 —
  implies the constant-ω eval is fundamentally OOD from boxcar
  training under this architecture; report and stop. (Threshold is
  lower than the fly's 0.5 because of the known train/eval
  distribution mismatch; if 0.3 isn't even reached after 80 iter,
  the recipe family is dead.)
