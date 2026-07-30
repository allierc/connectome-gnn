# Drosophila CX — Path Integration Task Learning

## Goal

Find the **best recurrent training scheme** for the connectome-constrained
CX RNN (`DrosophilaCxTaskRNN` in
[src/connectome_gnn/models/drosophila_cx_task_rnn.py](../src/connectome_gnn/models/drosophila_cx_task_rnn.py))
on the path-integration task: given angular velocity ω(t) and a one-frame
initial-heading impulse at t=0, predict (cos θ_hd, sin θ_hd) at every frame.

**Primary metric**: `r_roll` — Pearson correlation between the unwrapped
decoded heading trajectory and the (monotone) ground-truth heading on a
deterministic-sweep rollout (ω = 60°/s constant, warmup 10 frames). It
captures both magnitude and shape of the integration trajectory; a
network that decodes a constant or slowly drifting bump scores ≈ 0
even if `pi_acc` (instantaneous cosine similarity) looks healthy. Target
**`r_roll` ≥ 0.95** at full T=1000.

No secondary metrics.

The dataset is **fixed**: 100k train + 10k test trials × T=1000 frames at
dt=0.01s, generated once and reused across iterations. Only the training
hyperparameters change.

## Budget

**160 iterations** (fresh run), 10 slots × 4 batches = 40 iter / block →
**4 blocks total**. There is no room to re-solve the T=500 collapse
problem (already solved by the existing winner curriculum). Every block
must be a *targeted measurement* of one or two axes, not a free sweep.

## Reference: how flyvis (official repo, Lappalainen 2024) does it

- **Truncated BPTT** — short rollout windows (~50–500 steps), never full
  clip. This is the strongest single difference vs our current
  full-T-per-batch design.
- **Sign-locked synaptic strength + Dale's law** (we have this via
  `W_rec = |S| ⊙ W_con`).
- **Learnable τ per cell type** (we currently hold τ=0.1s fixed — exploration
  axis).
- **L2 weight decay on synapses** (we have `coeff_W_L1`; `coeff_W_L2` is
  also available).
- **Adam with stepwise LR decay** (we have this).
- **Warmup period excluded from loss** (we already skip the first 10 frames
  in `path_integration_accuracy`; the loss itself does NOT skip — worth
  testing whether a warmup-aware MSE helps).

## Available hyperparameters (the search space)

These are the fields the agent may set per-slot in `training:` /
`graph_model:`. Anything else should NOT be touched unless explicitly noted.

### Recurrent training scheme (PRIORITY — this is what we're optimising)

**Three-group optimiser (2026-05-19).** The trainer splits parameters into
three named groups with separate learning rates. Each group has its own
optional schedule that drives only that group; `other` (biases) is always
constant at `lr`.

| Group   | Trainable params                                                    | LR field    | Schedule?            |
| ------- | ------------------------------------------------------------------- | ----------- | -------------------- |
| `w_rec` | `S` (DrosophilaCxTaskRNN); `W`+`a`+`g_phi.*`+`f_theta.*` (DrosophilaCxTaskGNN) | `lr_W_rec`  | optional — `lr_W_rec_schedule` drives this if set |
| `w_ED`  | `W_in`, `W_out`, `_W_in_mlp.*`, `_W_out_mlp.*`, `v_pen{a,b}_{l,r}`  | `lr_W_ED`   | optional — `lr_W_ED_schedule` drives this if set  |
| `other` | biases (`b`, `b_out`) and anything else                             | `lr`        | NO — constant         |

Each schedule is independent: a missing/empty schedule leaves that group
at its initial lr (constant across epochs). If `lr_W_rec` is unset it
falls back to `lr`; same for `lr_W_ED`. So at minimum you have
`lr` + `lr_W_rec_schedule` and the recurrent core is scheduled; everything
else stays at `lr`. The W_in/W_out matrices are small (~760 params) so a
constant `lr_W_ED` is usually sufficient, but `lr_W_ED_schedule` is
available when you want a different I/O annealing trajectory.

| Field                   | Default                          | What it controls                                                                                                                                                      |
| ----------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lr`                    | `2e-3`                           | Constant lr for biases / `other` group, and the fallback init for `w_rec` and `w_ED` when their dedicated fields are unset.                                           |
| `lr_W_rec`              | unset (→ `lr`)                   | **Initial** lr for the recurrent core. `lr_W_rec_schedule` then drives this group across epochs. Set this when you want the schedule to start from a different value than `lr`. |
| `lr_W_ED`               | `5.0e-4` in the new yamls        | **Initial** lr for W_in / W_out / velocity-gate scalars. Constant unless `lr_W_ED_schedule` is set. Try {1e-4, 5e-4, 1e-3, 2e-3}. Smaller = slower I/O drift, more recurrent specialisation. |
| `lr_W_rec_schedule`     | per-yaml (5 epochs, 2e-3→5e-5)   | Per-epoch trajectory of the `w_rec` group **only**. Try faster decay if `w_rec` over-fits at high T; gentler if it under-trains. |
| `lr_W_ED_schedule`      | unset (constant `lr_W_ED`)       | Per-epoch trajectory of the `w_ED` group **only**. Use to anneal the encoder/decoder separately from `w_rec` — e.g. let W_in find the velocity-bump mapping early, then decay it to lock it in. |
| `noise_recurrent_level` | `0.0` (off)                      | **flyvis stabiliser.** Stddev of Gaussian noise added to `h` at every Euler step during training. Try {0, 1e-3, 1e-2, 5e-2}. Eval/snapshot stays deterministic. |
| `grad_clip_W`           | `0.0` (off)                      | Max-norm gradient clip on all trainable params. Set 1.0–10.0 to prevent `S` blowups at long T.                                                                        |
| `n_steps_schedule`      | per-yaml (5 epochs, 300→1000)    | Per-epoch trial length (BPTT horizon). Try gentler ramps; longer warmup at small T helps the T=500 collapse.                                                          |
| `batch_size`            | `64`                             | Try {32, 64, 128}. Larger = smoother gradients, less BPTT-noise variance.                                                                                             |
| `coeff_W_L1`            | `0.0`                            | L1 on `S` (synaptic magnitude). Try {0, 1e-5, 1e-4, 1e-3}.                                                                                                            |
| `w_init_scale`          | `0.01`                           | Scalar multiplier on `S` at init. Try {1e-3, 1e-2, 5e-2, 1e-1, 0.5}.                                                                                                  |
| `w_init_mode`           | `const`                          | Init template for `S`: `const` (=scale × mask), `randn` (=scale × randn × mask), `zeros`, `w_con` (=`|W_con|`). Ignored in `wrec_param: column_dale` mode (always randn). |

**Mutation guidance for the three-group setup:**
- Use `lr_W_ED` < `lr_W_rec_schedule[0]` (e.g. ratio 1:4) when the model overfits
  the input projection early in training (decoded HD locks to a fixed
  direction). Use `lr_W_ED` > `lr_W_rec_schedule[0]` to give the encoder more
  freedom to find the velocity-bump mapping.
- If a slot collapses with the recurrent lr at `lr_W_rec_schedule[epoch]`,
  shrinking `lr_W_ED` does NOT help — the collapse is in `w_rec`. Touch
  `lr_W_rec_schedule`, `noise_recurrent_level`, or `grad_clip_W` instead.
- `lr_W_rec` controls **only the epoch-0 starting value** of the schedule.
  Don't sweep this independently of `lr_W_rec_schedule[0]` — set them
  consistently (or leave `lr_W_rec` unset so it follows `lr`).

### Connectome-prior aux losses (already wired)

| Field                | Default | Role                                                           |
| -------------------- | ------- | -------------------------------------------------------------- | --- | ---------------------- |
| `coeff_cos_distance` | `1.0`   | Per-block cosine alignment to W_con (directional anchor).      |
| `coeff_norm_floor`   | `1.0`   | Soft floor on mean `                                           | W   | ` per type-pair block. |
| `kappa_norm_floor`   | `0.05`  | Floor target for the norm-floor penalty.                       |
| `coeff_tv_circular`  | `0.0`   | Circular TV on EPG/PEN ring firing rates. Try {0, 1e-3, 1e-2}. |

### Architecture

| Field            | Default        | Role                                               |
| ---------------- | -------------- | -------------------------------------------------- |
| `input_proj`     | `"matrix"`     | `"matrix"` (default) or `"mlp"`.                   |
| `output_proj`    | `"matrix"`     | Same options.                                      |
| `velocity_gate`  | `"pen_only"` ★ | Anatomical gate on `W_in[:, 0]` (velocity column). `"none"` = free `(N,3)` matrix; `"pen_only"` = mask velocity to PEN rows only (42 cells, per-unit free); `"pen_4scalar"` = strict 4-scalar version (L/R × PENa/PENb broadcast, 4 learnable scalars total). **Primary axis for this run.** |
| `hidden_dim`     | `64`           | Used only when projection is `"mlp"`.              |
| `n_layers`       | `2`            | Used only when projection is `"mlp"`.              |
| `MLP_activation` | `relu`         | `relu` / `tanh` / `leaky_relu` / `soft_relu`.      |
| `include_er6`    | `true`         | 156-neuron CX spec vs 152-neuron core loader.      |

### Things you must NOT change

- The `simulation:` block (data is on disk; changing these doesn't regenerate).
- The `task.path_integration` block (same — this is dataset spec).
- `signal_model_name` (must stay `drosophila_cx_pi`).
- `aggr_type` (irrelevant for this model).
- The eval/snapshot fields (`snapshots_per_epoch`, `snapshot_n_steps`,
  `snapshot_omega_deg`) — they don't affect training, only diagnostics.

## Metrics (per slot, per iteration)

Read from `<exploration_dir>/<slot_name>_analysis.log` after training. The
trainer also writes a per-iteration `tmp_training/metrics.log` you can tail
during the run.

| Metric               | What it measures                                                                  | Target                |
| -------------------- | --------------------------------------------------------------------------------- | --------------------- |
| `r_roll` (final)     | Pearson correlation between unwrapped decoded and true HD on a deterministic-sweep rollout (ω = 60°/s, warmup 10). | **≥ 0.95** at full T. |
| `r_roll` (per epoch) | End-of-epoch `r_roll` at the curriculum's `T_epoch`.                              | Monotonically high.   |
| `loss`               | Total training loss = mse + cosd + norm + tv + l1S.                               | Smooth, decreasing.   |
| `mse`                | Per-frame MSE on (cos, sin) target.                                               | Decreasing.           |
| `collapse_detected`  | Set when end-of-epoch `r_roll` drops by ≥0.4 between consecutive epochs.          | Should be `no`.       |

**The per-epoch trajectory is the most diagnostic signal.** A run with
`e1=0.99 e2=0.99 e3=0.00 e4=0.00 e5=0.00` failed at the curriculum jump
(numerical instability); a run with `e1=0.20 e2=0.30 e3=0.40 e4=0.50 e5=0.55`
failed because the chosen lr/coeffs aren't strong enough.

## Causality rule

You can change one or two parameters per slot.

In **robustness mode** (every slot identical), the pipeline forces 8
different seeds; this measures seed sensitivity of a candidate winner.

## Block plan (160 iterations, 4 blocks × 40 iter, 10 slots/batch × 4 batches)

### Current best (2026-05-20)

Single-seed run of `drosophila_cx_pi_winner.yaml` hit **r_roll = 1.000
from epoch 2** and held through epoch 5 (T=500); final test pi_acc =
0.9913 at T=1000. The winning recipe:

```yaml
graph_model:
  velocity_gate: pen_4scalar
training:
  batch_size: 64
  lr: 2.0e-3
  lr_W_ED: 5.0e-4              # constant; encoder/decoder slow
  lr_W_rec_schedule: [2e-3, 2e-3, 1e-3, 1e-3, 5e-4, 4e-4, 3e-4, 2e-4, 5e-5, 5e-5]
  grad_clip_W: 2.5             # B1 plateau
  noise_recurrent_level: 0.05  # flyvis-style stabiliser
  coeff_cos_distance: 0.0      # KEY: connectome-prior disabled
  coeff_norm_floor: 0.5
  coeff_W_L1: 0.0
  n_epochs: 10
  n_steps_schedule: [100, 200, 300, 400, 500, 600, 800, 900, 1000, 1000]
  # ↑ gentle T-warmup starting at 100 was the missing piece
```

**This is a single-seed result.** Prior exploration documented severe seed
variance at neighbouring configs (`r_roll` flipping 0.92 → 0.44 → −0.24
across three seeds of the iter-28 config). The **next LLM iteration must
be a 10-seed robustness test on this exact config** before promoting it
to canonical winner. Decision rule for robustness:

- ≥ 8/10 above r_roll = 0.95 → winner confirmed; skip to B3 (gate +
  extras) — B1 (stabilisation) and B2 (curriculum) are effectively
  resolved.
- 4–7/10 above 0.95 → solid candidate; narrow B2 with this config as
  parent (try minor n_steps / lr_W_rec_schedule variants).
- < 4/10 above 0.95 → seed-fluke; investigate what made this seed
  special (interaction between curriculum jump timing, noise, and the
  random init of S).

### Fallback clean parent (only if winner.yaml robustness fails)

```
graph_model:
  velocity_gate: pen_4scalar
  wrec_param: edge_magnitude
training:
  lr: 2.0e-2                     # constant for biases / "other"
  lr_W_ED: 5.0e-4
  batch_size: 1                  # SGD-like noise
  coeff_cos_distance: 0.0
  coeff_norm_floor: 0.5
  noise_recurrent_level: 0.0     # re-evaluated in B1
  grad_clip_W: 0.0               # re-evaluated in B1
  n_epochs: 5
  n_steps_schedule: [300, 500, 700, 900, 1000]
  lr_W_rec_schedule:      [2.0e-2, 1.0e-3, 5.0e-4, 2.0e-4, 5.0e-5]
```

Reordered for the post-fix problem (**stabilisation first, curriculum
second, gate third, robustness fourth**). Each block's 40 iterations stay
on its own axis — do not drift.

| Block | Question | Slot layout (10 slots / batch) | Decision rule for the block boundary |
| ----- | -------- | ------------------------------- | ------------------------------------ |
| **1 — Recurrent-core stabilisation** | *What gets `r_roll` past the T=400 collapse via noise / clip / I-O timescale?* These are the "smoothness" knobs on the BPTT landscape. | s0 control (clean parent) · s1 `noise_recurrent_level: 1e-3` · s2 `noise_recurrent_level: 1e-2` · s3 `noise_recurrent_level: 5e-2` · s4 `noise_recurrent_level: 1e-1` · s5 `grad_clip_W: 1.0` · s6 `grad_clip_W: 5.0` · s7 `lr_W_ED: 1e-4` (slow I/O — does freezing the encoder save w_rec?) · s8 `lr_W_ED: 1e-3` (faster I/O) · s9 combo: noise=5e-2 + clip=1.0 | Promote the slot with the highest final `r_roll` and no collapse between consecutive epochs. This becomes the **stabilisation parent** for B2. If no slot exceeds the control by ≥0.05 in `r_roll`, run a second batch widening the most promising noise / clip / lr_W_ED range. |
| **2 — Curriculum (`n_steps_schedule` + matched `lr_W_rec_schedule` sweep)** | *Given B1's stabilisation, which (BPTT-horizon ramp, lr trajectory) pair maximises `r_roll`?* The two schedules must be co-tuned — a gentler `n_steps` ramp can carry a slower `lr` decay (the optimiser has more time at each horizon); an aggressive ramp needs faster `lr` decay to dampen the jump. Each slot varies BOTH lists together. | s0 control: `n_steps_schedule: [300,500,700,900,1000]` + `lr_W_rec_schedule: [2e-2,1e-3,5e-4,2e-4,5e-5]` + `batch_size: 1` (current defaults) · s1 gentle ramp + slow lr decay: `[200,350,500,750,1000]` + `[2e-3,1e-3,7e-4,3e-4,1e-4]` · s2 very-gentle + slow lr: `[100,200,400,700,1000]` + `[2e-3,1.5e-3,1e-3,5e-4,2e-4]` · s3 long-warmup + held lr: `[300,300,500,700,1000]` + `[2e-3,2e-3,1e-3,5e-4,2e-4]` · s4 linear ramp + default lr: `[300,400,500,700,1000]` + `[2e-3,1e-3,5e-4,2e-4,5e-5]` · s5 aggressive ramp + fast lr decay: `[500,700,900,1000,1000]` + `[2e-3,5e-4,1e-4,5e-5,1e-5]` · s6 full-T from start + very-fast lr decay (sanity probe): `[1000,1000,1000,1000,1000]` + `[1e-3,3e-4,1e-4,3e-5,1e-5]` · s7 extreme gentle + lowest lr: `[100,200,400,700,1000]` + `[1e-3,5e-4,2e-4,1e-4,5e-5]` · s8 `batch_size: 32` (defaults schedule) · s9 `batch_size: 128` (defaults schedule) | Promote the slot whose final `r_roll` exceeds B1's parent by ≥ 0.02 with no collapse. This becomes the **curriculum parent** for B3. If nothing wins by that margin, keep B1's parent. |
| **3 — Gate choice + remaining knobs** | *With B1+B2 frozen, does the velocity gate / extra regularisers still matter?* | s0 (B2 winner — bridge) · s1 `velocity_gate: pen_only` · s2 `velocity_gate: none` · s3 `coeff_norm_floor: 0.0` · s4 `kappa_norm_floor: 0.10` · s5 `coeff_W_L1: 1e-5` · s6 `coeff_W_L1: 1e-4` · s7 `coeff_tv_circular: 1e-3` · s8 `w_init_mode: w_con` · s9 `w_init_mode: randn` + scale=5e-2 | Promote any slot whose `r_roll` improves by ≥ 0.02 over the bridge. |
| **4 — 10-seed robustness** | *Is the B3 winner seed-robust?* | All 10 slots = identical config (B3 winner). Pipeline auto-forces 10 different seeds in robustness mode. | Report mean ± std for `r_roll` across the 10 seeds. Save as the new `drosophila_cx_pi_winner.yaml` if mean `r_roll` ≥ 0.95 **and** no seed collapses. |

### Budget guard-rails

- **B1 is the existential block.** If no slot in B1 produces a non-collapsing
  trajectory past T=500, do NOT proceed to B2 — widen B1 instead (more noise
  levels, combo knobs). Stabilisation is the gate.
- **One axis per block.** B1 is the noise/clip/I-O-lr axis; B2 is the
  schedule/curriculum/batch axis; B3 is gate + extra regularisers. Do not
  mix axes within a block.
- **`r_roll ≥ 0.95` is the bar.** Re-bar after B1+B2 once we know what the
  landscape supports.
- **`lr_W_rec_schedule` only affects `w_rec`** (the recurrent core). Mutations that
  touch `lr_W_ED` or `lr` change a *different* timescale — don't sweep them
  in the same slot as a schedule change, you won't know which axis moved
  `r_roll`.
- **`lr_W_rec` is redundant with `lr` + `lr_W_rec_schedule`** in most cases — the
  schedule overwrites it at every epoch. Setting `lr_W_rec` is only useful
  if you want a different epoch-0 value than `lr` (rare). Default: leave
  unset.

## Mutation log format (per iteration)

After each batch, append to working memory:

```
## Iter N (block B): [exploration | robustness]
Parent: iter_M_slot_K  (r_roll=X.XXX at full T)
Hypothesis: "[testable claim about what the mutation should do]"
Slot 0: [parent/control]   r_roll=X.XXX  collapse=no  traj=e1=A e2=B e3=C e4=D e5=E
Slot 1: [knob -> value]    r_roll=X.XXX  ...
...
Slot 9: [knob -> value]    r_roll=X.XXX  ...
Best slot: K  ->  r_roll=X.XXX
Verdict: [supported | falsified | inconclusive]
Next parent: iter_N_slot_K
```

When a slot collapses, note the epoch at which it dropped and the loss
value at that epoch — this is the most informative diagnostic.

## Winner config

At every block boundary, copy the best slot's config to
`config/drosophila_cx/drosophila_cx_pi_winner.yaml` with header. The
existing winner.yaml is from the bug era and is no longer authoritative —
overwriting it with the new B4 result is the goal. The bar is
**`r_roll` ≥ 0.95** until B1+B2 establish what the post-fix landscape
actually supports.

```yaml
# Winner: drosophila_cx_pi_winner.yaml
# Source: iter_NNN_slot_KK  (final r_roll = X.XXX)
# Block: B  (focus: <focus>)
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - <one-sentence reason>
#   - <key knob change>
#
# Per-epoch trajectory: e1=A e2=B e3=C e4=D e5=E  (no collapse)
# Robustness: tested across N seeds, mean=X.XXX ± Y.YYY
```

## Notes / hints

- **The trainer's `tmp_training/evolution/` directory** has 6-panel
  snapshots (GT W_con, learned W_rec, EPG kinograph, EPG raster, PEN raster,
  ω+HD overlay) at every snapshot interval. When diagnosing a collapse, look
  at the W_rec panel just before the collapse — it usually shows blown-up
  off-diagonal entries (no longer respecting the connectome block structure)
  before the bump destabilises.
- **Bump width going wide (~80° → 360° in snapshots)** signals delocalisation:
  the bump spread out and stopped tracking. Different failure mode from a
  W-explosion collapse (where `r_roll` drops to ~0 without the bump
  visibly spreading).
- **Noise injection is now a real knob** (`training.noise_recurrent_level`,
  default 0). It's the single most-promising stabiliser borrowed from
  flyvis. If Block 2 finds noise alone fixes the collapse, that's the
  simplest winner; if not, combine with `grad_clip_W` and/or `coeff_W_L1`.


