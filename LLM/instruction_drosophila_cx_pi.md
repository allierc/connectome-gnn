# Drosophila CX — Path Integration Task Learning

## Goal

Find the **best recurrent training scheme** for the Hulse-style
connectome-constrained CX RNN (`TaskRNN` in
[src/connectome_gnn/models/task_rnn.py](../src/connectome_gnn/models/task_rnn.py))
on the path-integration task: given angular velocity ω(t) and a one-frame
initial-heading impulse at t=0, predict (cos θ_hd, sin θ_hd) at every frame.

**Primary metric**: `pi_acc` (mean cosine similarity between decoded and true
heading on the test split, after a 10-frame warmup). Hulse-paper-level is
**pi_acc ≥ 0.95** at full T=1000.

The dataset is **fixed**: 100k train + 10k test trials × T=1000 frames at
dt=0.01s, generated once and reused across iterations. Only the training
hyperparameters change.

## What's known (baseline behaviour)

The current default (Hulse spec) hits pi_acc ≈ 0.999 at the T=100
curriculum stage and pi_acc ≈ 0.997 at T=250, then **collapses to pi_acc ≈
0.000 at T=500** (loss jumps from 0.01 to 0.50). This is the central failure
mode the agentic loop must understand and fix.

**Hypotheses for the T=500 collapse** (these are what the loop should test):

1. **Full-T BPTT through 500+ Euler steps without noise injection** is too
   sharp a landscape for Adam at lr=5e-4. (Flyvis injects
   `noise_recurrent_level · randn` at every step to smooth this; we don't.)
2. **Gradient clipping is off** (`grad_clip_W = 0`) — a single bad step can
   blow `|S|` out of the basin.
3. **lr schedule may decay too slowly** for the longer rollouts. Hulse drops
   5e-3 → 1e-4 over 5 epochs; perhaps 5e-3 → 1e-5 is needed.
4. **Sigmoid saturation** in the recurrent unit: long unrolls let `h` drift
   outside the linear regime of σ; once saturated, gradients vanish.
5. **Curriculum jump is too aggressive**: T=250 → T=500 is a 2× jump on a
   non-linear landscape. Smaller jumps (e.g. 100,200,300,500,1000) might
   help.
6. **Connectome regularizers** (cos-distance, norm-floor) may be holding W
   too close to the initial template at long T, preventing the small
   adjustments needed for stable integration.

The agentic loop should propose, run, and falsify these.

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

| Field                   | Default                          | What it controls                                                                                                                                                      |
| ----------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | -------------------------------------------------- |
| `noise_recurrent_level` | `0.0` (off)                      | **NEW — flyvis stabiliser.** Stddev of Gaussian noise added to `h` at every Euler step during training. Try {0, 1e-3, 1e-2, 5e-2}. Eval/snapshot stays deterministic. |
| `grad_clip_W`           | `0.0` (off)                      | **CRITICAL.** Max-norm gradient clip on all trainable params. Set 1.0–10.0 to prevent `                                                                               | S   | ` blowups at long T.                               |
| `n_steps_schedule`      | `[100, 250, 500, 1000, 1000]`    | Per-epoch trial length (BPTT horizon). Try gentler ramps: `[100,150,250,500,1000]` or longer warmup at small T.                                                       |
| `lr_schedule`           | `[5e-3, 1e-3, 5e-4, 2e-4, 1e-4]` | Per-epoch lr. Try faster decay: `[5e-3, 5e-4, 1e-4, 5e-5, 1e-5]`.                                                                                                     |
| `n_epochs`              | `5`                              | Number of curriculum stages. Could be 7–10 with finer T steps.                                                                                                        |
| `batch_size`            | `64`                             | Try {32, 64, 128}. Larger = smoother gradients, less BPTT-noise variance.                                                                                             |
| `coeff_W_L1`            | `0.0`                            | L1 on `                                                                                                                                                               | S   | ` (synaptic magnitude). Try {0, 1e-5, 1e-4, 1e-3}. |
| `w_init_scale`          | `0.01`                           | Scalar multiplier on the per-edge magnitude `S` at init (`S = w_init_scale * W_con_mask` in `const` mode). Try {1e-3, 1e-2, 5e-2, 1e-1, 0.5}.                         |
| `w_init_mode`           | `const`                          | Init template for `S`: `const` (=scale × mask, current default), `randn` (=scale × randn × mask, sign-symmetric noise on connectome support), `zeros`.                |

### Hulse aux losses (already wired)

| Field                | Default | Role                                                           |
| -------------------- | ------- | -------------------------------------------------------------- | --- | ---------------------- |
| `coeff_cos_distance` | `1.0`   | Hulse Eq. 10. Holds W_rec block-direction close to W_con.      |
| `coeff_norm_floor`   | `1.0`   | Hulse Eq. 11. Soft floor on mean `                             | W   | ` per type-pair block. |
| `kappa_norm_floor`   | `0.05`  | Floor target for the norm-floor penalty.                       |
| `coeff_tv_circular`  | `0.0`   | Circular TV on EPG/PEN ring firing rates. Try {0, 1e-3, 1e-2}. |

### Architecture

| Field            | Default    | Role                                               |
| ---------------- | ---------- | -------------------------------------------------- |
| `input_proj`     | `"matrix"` | `"matrix"` (Hulse default) or `"mlp"`.             |
| `output_proj`    | `"matrix"` | Same options.                                      |
| `hidden_dim`     | `64`       | Used only when projection is `"mlp"`.              |
| `n_layers`       | `2`        | Used only when projection is `"mlp"`.              |
| `MLP_activation` | `relu`     | `relu` / `tanh` / `leaky_relu` / `soft_relu`.      |
| `include_er6`    | `true`     | 156-neuron Hulse spec vs 152-neuron Beiran loader. |

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
| `pi_acc` (final)     | Mean cosine similarity decoded vs true HD on full test set, full T.               | **≥ 0.95** at full T. |
| `pi_acc` (per epoch) | End-of-epoch `pi_acc` at the curriculum's `T_epoch`.                              | Monotonically high.   |
| `fwhm_deg`           | Bump width in degrees (single bump → 60–180°; delocalised → 360°).                | < 180°.               |
| `loss`               | Total training loss = mse + cosd + norm + tv + l1S.                               | Smooth, decreasing.   |
| `mse`                | Per-frame MSE on (cos, sin) target.                                               | Tracks `1 - pi_acc`.  |
| `cosd`, `norm`, `tv` | Per-block reg values. Useful for detecting "loss is small but reg is dominating". | Should not climb.     |
| `collapse_detected`  | Set when end-of-epoch `pi_acc` drops by ≥0.4 between consecutive epochs.          | Should be `no`.       |

**The per-epoch trajectory is the most diagnostic signal.** A run with
`e1=0.99 e2=0.99 e3=0.00 e4=0.00 e5=0.00` failed at the curriculum jump
(numerical instability); a run with `e1=0.20 e2=0.30 e3=0.40 e4=0.50 e5=0.55`
failed because the chosen lr/coeffs aren't strong enough.

## Causality rule

You can change one or two parameters per slot.

In **robustness mode** (every slot identical), the pipeline forces 8
different seeds; this measures seed sensitivity of a candidate winner.

## Block plan

8 slots/batch. Iterations: 148 total ≈ 18 batches ≈ 5 batches/block.

| Block | Focus                                  | Knobs to scan                                                                                                  | Why                                                                              |
| ----- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | --- | --------------------------------------------------------------------------------------------------------- |
| 1     | **Baseline + collapse confirmation**   | None — robustness test of Hulse defaults across 8 seeds.                                                       | Confirm the T=500 collapse is real and seed-stable (vs an unlucky run).          |
| 2     | **Stabilising long-T BPTT (priority)** | `noise_recurrent_level` ∈ {0, 1e-3, 1e-2, 5e-2}; `grad_clip_W` ∈ {0, 1, 5, 10}; `coeff_W_L1` ∈ {0, 1e-5, 1e-4}; `w_init_scale` ∈ {1e-3, 1e-2, 5e-2, 1e-1}; `w_init_mode` ∈ {const, randn} | Direct attack on the collapse: noise injection + grad-clip + light L1 to bound ` | S   | ` growth. **Test the noise + clip combo last — if a single knob works alone, prefer the simpler config.** |
| 3     | **lr schedule tuning**                 | `lr_schedule` variants: faster decay, slower decay, all-low                                                    | Hypothesis: lr=5e-4 at T=500 is too high once h saturates.                       |
| 4     | **Curriculum smoothing**               | `n_steps_schedule` variants: gentler ramps, more epochs at small T                                             | Hypothesis: T=250→500 jump is the trigger.                                       |
| 5     | **Aux-loss strength**                  | `coeff_cos_distance`, `coeff_norm_floor`, `kappa_norm_floor`, `coeff_tv_circular`                              | Test whether tighter connectome priors stabilise W at long T.                    |
| 6     | **Architecture sweep**                 | `input_proj`/`output_proj` ∈ {"matrix","mlp"}; `hidden_dim`, `MLP_activation`                                  | Does an MLP I/O give the dynamics enough flexibility to integrate stably?        |
| 7     | **Free exploration**                   | Any combination of best knobs from blocks 2-6                                                                  | Combine winners; test interactions.                                              |
| 8     | **Final robustness**                   | None — 8-seed test of best config from blocks 1-7.                                                             | Confirm winner is seed-robust at full T=1000.                                    |

## Mutation log format (per iteration)

After each batch, append to working memory:

```
## Iter N (block B): [exploration | robustness]
Parent: iter_M_slot_K  (pi_acc=X.XXX at full T)
Hypothesis: "[testable claim about what the mutation should do]"
Slot 0: [parent/control]   pi_acc=X.XXX  fwhm=YY°  collapse=no   traj=e1=A e2=B e3=C e4=D e5=E
Slot 1: [knob -> value]    pi_acc=X.XXX  ...
...
Slot 7: [knob -> value]    pi_acc=X.XXX  ...
Best slot: K  ->  pi_acc=X.XXX
Verdict: [supported | falsified | inconclusive]
Next parent: iter_N_slot_K
```

When a slot collapses, note the epoch at which it dropped and the loss/cosd/norm
values at that epoch — this is the most informative diagnostic.

## Winner config

At every block boundary, copy the best slot's config to
`config/drosophila_cx/drosophila_cx_pi_winner.yaml` with header:

```yaml
# Winner: drosophila_cx_pi_winner.yaml
# Source: iter_NNN_slot_KK  (final pi_acc = X.XXX, fwhm = YY°)
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

- **The trainer's `tmp_training/kinograph_matrix/` directory** has 6-panel
  snapshots (GT W_con, learned W_rec, EPG kinograph, EPG raster, PEN raster,
  ω+HD overlay) at every snapshot interval. When diagnosing a collapse, look
  at the W_rec panel just before the collapse — it usually shows blown-up
  off-diagonal entries (no longer respecting the connectome block structure)
  before the bump destabilises.
- **`bump_fwhm` going from ~80° to 360°** signals delocalisation: the bump
  spread out and stopped tracking. Different failure mode from W explosion
  (which usually shows fwhm staying small but pi_acc → 0).
- **Noise injection is now a real knob** (`training.noise_recurrent_level`,
  default 0). It's the single most-promising stabiliser borrowed from
  flyvis. If Block 2 finds noise alone fixes the collapse, that's the
  simplest winner; if not, combine with `grad_clip_W` and/or `coeff_W_L1`.
