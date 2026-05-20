# Drosophila CX — Path Integration Task Learning

## Goal

Find the **best recurrent training scheme** for the connectome-constrained
CX RNN (`CxTaskRNN` in
[src/connectome_gnn/models/cx_task_rnn.py](../src/connectome_gnn/models/cx_task_rnn.py))
on the path-integration task: given angular velocity ω(t) and a one-frame
initial-heading impulse at t=0, predict (cos θ_hd, sin θ_hd) at every frame.

**Primary metric**: `pi_acc` (mean cosine similarity between decoded and true
heading on the test split, after a 10-frame warmup). Reference is
**pi_acc ≥ 0.95** at full T=1000.

**Secondary metric (now first-class — see preamble)**: `gt_R2` and
`gt_slope` — linear fit between learned `W_rec` and GT `W_con` on the
non-zero edges. Now that the W_rec forward bug is fixed (see preamble
below), these numbers measure real anatomical fidelity. Two circuits can
both hit pi_acc ≥ 0.99 but one might be anatomically close (`gt_R2 → 1`)
and the other functional-but-divergent (`gt_R2 ≈ 0.3`). **Always extract
both into the per-slot analysis-log line, not just the snapshot title.**

The dataset is **fixed**: 100k train + 10k test trials × T=1000 frames at
dt=0.01s, generated once and reused across iterations. Only the training
hyperparameters change.

## Convention fix (2026-05-19) + observed regression — READ FIRST

A bug in `CxTaskRNN` was discovered and fixed: the recurrent input was
computed as `r @ W_rec` instead of `r @ W_rec.T`, so message flow was
post→pre instead of pre→post. The fix changes which weight basin the
optimiser walks toward — **the prior winner config no longer converges
under correct dynamics.**

Measured under the fixed forward, using the previous winner
(`pen_4scalar` + `coeff_cos_distance=0.05` + `noise_recurrent_level=0.05`):

| epoch | T   | end-of-epoch pi_acc | status                |
|-------|-----|----------------------|------------------------|
| 1     | 300 | 0.574                | partial, not converged |
| 2     | 400 | 0.040                | **collapsed**          |
| 3     | 500 | ≈ 0.03               | still collapsed        |

So the T=500 collapse is a **real BPTT-landscape issue under correct
dynamics**, not the transpose artefact we suspected. Implications for
this exploration:

1. **`drosophila_cx_pi_winner.yaml` is no longer a valid parent.** All
   prior numerical optima (`cosd=0.05`, `noise=0.05`, etc.) were tuned
   against the buggy forward and do not transfer.
2. **Prior `gt_R2` numbers are not comparable** — they compared a
   transposed-effective `W_rec` to `W_con`. Treat any prior gt_R2 in
   memory files as void; rebuild the trajectory under correct dynamics.
3. **The block plan is reordered**: convergence first (find a config
   that survives T≥500), anatomy second (cos_distance trade-off), gate
   choice third. Old block plan put gate first — but the gate choice
   only matters once we have a converging baseline.
4. **Noise / cos_distance / clip move back to hypotheses**, not
   locked-in priors. The prior exploration baked them into the parent
   based on bug-era data; this exploration starts clean.

The qualitative search space, metrics, and infrastructure are unchanged;
only the parent and block ordering need to be reset.

## Budget

**160 iterations** (fresh run), 10 slots × 4 batches = 40 iter / block →
**4 blocks total**. There is no room to re-solve the T=500 collapse
problem (already solved by the existing winner curriculum). Every block
must be a *targeted measurement* of one or two axes, not a free sweep.

## What's known (baseline behaviour, under correct dynamics)

With the bug fixed, the prior winner config (`pen_4scalar` + `cosd=0.05`
+ `noise=0.05` + 10-epoch curriculum 300..1000) reaches pi_acc ≈ 0.57 at
T=300 and **collapses to pi_acc ≈ 0.04 at T=400**, loss going from
~0.35 to ~0.50. The full-T BPTT landscape under correct message flow
is the central problem to solve in Block 1.

Note: the OLD `drosophila_cx_pi_Claude_memory.md` exploration logs were
all under the buggy forward — treat their pi_acc numbers as a different
problem entirely. Do not parent off them.

**Hypotheses for the T≥400 collapse** (these are what the loop should test):

1. **Full-T BPTT through 500+ Euler steps without noise injection** is too
   sharp a landscape for Adam at lr=5e-4. (Flyvis injects
   `noise_recurrent_level · randn` at every step to smooth this; we don't.)
2. **Gradient clipping is off** (`grad_clip_W = 0`) — a single bad step can
   blow `|S|` out of the basin.
3. **lr schedule may decay too slowly** for the longer rollouts. The
   default drops 5e-3 → 1e-4 over 5 epochs; perhaps 5e-3 → 1e-5 is needed.
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

**Three-group optimiser (2026-05-19).** The trainer splits parameters into
three named groups with separate learning rates. **`lr_schedule` drives only
the `w_rec` group** — `w_ED` (encoder/decoder) and `other` (biases) stay
constant at their respective `lr_W_ED` / `lr` across epochs. Asymmetry is
deliberate: I/O matrices are small (W_in: N×3, W_out: 2×N, ~760 params total)
and don't need decay; the recurrent core (`S`, ~25k params) does.

| Group   | Trainable params                                                    | LR field    | Schedule?            |
| ------- | ------------------------------------------------------------------- | ----------- | -------------------- |
| `w_rec` | `S` (CxTaskRNN); `W`+`a`+`g_phi.*`+`f_theta.*` (CxTaskGNN)          | `lr_W_rec`  | **YES — `lr_schedule` drives this** |
| `w_ED`  | `W_in`, `W_out`, `_W_in_mlp.*`, `_W_out_mlp.*`, `v_pen{a,b}_{l,r}`  | `lr_W_ED`   | NO — constant         |
| `other` | biases (`b`, `b_out`) and anything else                             | `lr`        | NO — constant         |

If `lr_W_rec` is unset it falls back to `lr` (then the schedule still
drives it). If `lr_W_ED` is unset it also falls back to `lr` (but stays
constant — no schedule). So at minimum you have `lr` + `lr_schedule` and
the recurrent core is scheduled; everything else stays at `lr`.

| Field                   | Default                          | What it controls                                                                                                                                                      |
| ----------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lr`                    | `2e-3`                           | Constant lr for biases / `other` group, and the fallback init for `w_rec` and `w_ED` when their dedicated fields are unset.                                           |
| `lr_W_rec`              | unset (→ `lr`)                   | **Initial** lr for the recurrent core. `lr_schedule` then drives this group across epochs. Set this when you want the schedule to start from a different value than `lr`. |
| `lr_W_ED`               | `5.0e-4` in the new yamls        | **Constant** lr for W_in / W_out / velocity-gate scalars. Try {1e-4, 5e-4, 1e-3, 2e-3}. Smaller = slower I/O drift, more recurrent specialisation. |
| `lr_schedule`           | per-yaml (5 epochs, 2e-3→5e-5)   | Per-epoch trajectory of the `w_rec` group **only**. Try faster decay if `w_rec` over-fits at high T; gentler if it under-trains. |
| `noise_recurrent_level` | `0.0` (off)                      | **flyvis stabiliser.** Stddev of Gaussian noise added to `h` at every Euler step during training. Try {0, 1e-3, 1e-2, 5e-2}. Eval/snapshot stays deterministic. |
| `grad_clip_W`           | `0.0` (off)                      | Max-norm gradient clip on all trainable params. Set 1.0–10.0 to prevent `S` blowups at long T.                                                                        |
| `n_steps_schedule`      | per-yaml (5 epochs, 300→1000)    | Per-epoch trial length (BPTT horizon). Try gentler ramps; longer warmup at small T helps the T=500 collapse.                                                          |
| `n_epochs`              | `5`                              | Number of curriculum stages. Must match `len(lr_schedule)` and `len(n_steps_schedule)`. Reduced from 10 → 5 (2026-05-19) to fit the wall-clock budget; the 5-epoch ramp [300, 500, 700, 900, 1000] still reaches T=1000. |
| `batch_size`            | `64`                             | Try {32, 64, 128}. Larger = smoother gradients, less BPTT-noise variance.                                                                                             |
| `coeff_W_L1`            | `0.0`                            | L1 on `S` (synaptic magnitude). Try {0, 1e-5, 1e-4, 1e-3}.                                                                                                            |
| `w_init_scale`          | `0.01`                           | Scalar multiplier on `S` at init. Try {1e-3, 1e-2, 5e-2, 1e-1, 0.5}.                                                                                                  |
| `w_init_mode`           | `const`                          | Init template for `S`: `const` (=scale × mask), `randn` (=scale × randn × mask), `zeros`, `w_con` (=`|W_con|`). Ignored in `wrec_param: column_dale` mode (always randn). |

**Mutation guidance for the three-group setup:**
- Use `lr_W_ED` < `lr_schedule[0]` (e.g. ratio 1:4) when the model overfits
  the input projection early in training (decoded HD locks to a fixed
  direction). Use `lr_W_ED` > `lr_schedule[0]` to give the encoder more
  freedom to find the velocity-bump mapping.
- If a slot collapses with the recurrent lr at `lr_schedule[epoch]`,
  shrinking `lr_W_ED` does NOT help — the collapse is in `w_rec`. Touch
  `lr_schedule`, `noise_recurrent_level`, or `grad_clip_W` instead.
- `lr_W_rec` controls **only the epoch-0 starting value** of the schedule.
  Don't sweep this independently of `lr_schedule[0]` — set them
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

## Block plan (160 iterations, 4 blocks × 40 iter, 10 slots/batch × 4 batches)

Parent at iter 1 = a **clean, minimal starting config** (NOT the old
`drosophila_cx_pi_winner.yaml`, which is now known to collapse — see
preamble). Suggested clean parent:

```
graph_model:
  velocity_gate: pen_4scalar     # anatomically tightest; revisited in B3
  wrec_param: edge_magnitude     # Dale sign-locked (NOT column_dale)
training:
  lr: 2.0e-3                     # constant for biases / "other" group
  lr_W_ED: 5.0e-4                # constant for W_in / W_out / velocity-gate scalars
  # lr_W_rec unset → starts at `lr`, then driven by lr_schedule
  coeff_cos_distance: 0.1        # mid baseline, refined in B2
  coeff_norm_floor: 0.5
  noise_recurrent_level: 0.0     # off — re-evaluated in B1
  grad_clip_W: 0.0               # off — re-evaluated in B1
  coeff_W_L1: 0.0                # off — optional in B1
  n_epochs: 5
  n_steps_schedule: [300, 500, 700, 900, 1000]
  lr_schedule:      [2.0e-3, 1.0e-3, 5.0e-4, 2.0e-4, 5.0e-5]
  # ↑ drives the w_rec group only; w_ED / other stay constant.
```

Reordered for the post-fix problem (**convergence first, anatomy second,
gate third**). Each block's 40 iterations stay on its own axis — do not
drift.

| Block | Question | Slot layout (10 slots / batch) | Decision rule for the block boundary |
| ----- | -------- | ------------------------------- | ------------------------------------ |
| **1 — T≥500 stabilisation** | *What gets pi_acc ≥ 0.95 past the T=400 collapse under correct dynamics?* This is the new central failure mode. | s0 control (clean parent) · s1 `noise_recurrent_level: 1e-3` · s2 `noise_recurrent_level: 1e-2` · s3 `noise_recurrent_level: 5e-2` · s4 `grad_clip_W: 1.0` · s5 `grad_clip_W: 5.0` · s6 `lr_schedule` faster decay (e.g. [2e-3,5e-4,1e-4,5e-5,...]) · s7 `n_steps_schedule` gentler ramp (200..1000) · s8 `lr_W_ED: 1e-4` (slow I/O — does freezing the encoder save w_rec?) · s9 combo: noise=5e-2 + clip=1.0 | Promote any slot with end-of-epoch pi_acc ≥ 0.95 at T=1000 **and** no collapse between consecutive epochs. Among qualifying slots, pick the one with highest final pi_acc + lowest end-of-curriculum fwhm. This becomes the **convergence parent** for B2. If no slot qualifies, run a second batch with the most promising direction widened (consider also `lr_W_ED ∈ {1e-3, 2e-3}` if the encoder needs MORE freedom, not less). |
| **2 — `coeff_cos_distance` trade-off** | *Now that the run converges, where is the pi_acc / `gt_R2` Pareto front?* This is the precision vs anatomical-fidelity trade-off. **B2 cannot start until B1 has a converging parent.** | All slots inherit B1's convergence parent. Sweep `coeff_cos_distance` ∈ {0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0}: s0 0.0 · s1 0.0 · s2 0.05 · s3 0.05 · s4 0.1 · s5 0.25 · s6 0.5 · s7 1.0 · s8 1.0 · s9 2.0 (each level seed-doubled where possible). | Plot (pi_acc, `gt_R2`, `gt_slope`) vs coeff. Pick the **lowest coeff** at which mean pi_acc still ≥ 0.95 across its seeds. Call this `coeff_*` and freeze it. |
| **3 — Gate choice + remaining knobs** | *With B1+B2 frozen, does the velocity gate / extra regularisers still matter?* | s0 (B2 winner — bridge) · s1 `velocity_gate: pen_only` · s2 `velocity_gate: none` · s3 `coeff_norm_floor: 0.0` · s4 `kappa_norm_floor: 0.10` · s5 `coeff_W_L1: 1e-5` · s6 `coeff_W_L1: 1e-4` · s7 `coeff_tv_circular: 1e-3` · s8 `w_init_mode: w_con` · s9 `w_init_mode: randn` + scale=5e-2 | Promote any slot that keeps pi_acc ≥ 0.95 **and** improves either `gt_R2` by ≥ 0.05 over bridge or `fwhm` by ≥ 10° toward the biological 60–90° band. |
| **4 — 10-seed robustness** | *Is the B3 winner seed-robust?* | All 10 slots = identical config (B3 winner). Pipeline auto-forces 10 different seeds in robustness mode. | Report mean ± std for `pi_acc`, `fwhm_deg`, `gt_R2`, `gt_slope` across the 10 seeds. Save as the new `drosophila_cx_pi_winner.yaml` if mean pi_acc ≥ 0.95 **and** no seed collapses. |

### Budget guard-rails

- **B1 is the existential block.** If no slot in B1 converges past T=500,
  do NOT proceed to B2 — widen B1 instead (more noise levels, combo
  knobs, deeper schedule changes). Convergence is the gate.
- **One axis per block** is the rule, except B1 which is multi-axis by
  necessity (we don't yet know which stabiliser works).
- **Don't change `coeff_cos_distance` after B2** — it's frozen for B3/B4.
- **Always log `gt_R2` and `gt_slope` in the per-slot analysis-log line**,
  not just the snapshot title. Mutation-log lines without `gt_R2` are
  considered incomplete results — re-extract from the latest snapshot
  filename if necessary.
- **`pi_acc ≥ 0.95` is the post-fix bar**, not 0.99. We do not yet know
  whether 0.99 is achievable under correct dynamics. Re-bar after B1+B2.
- **`lr_schedule` only affects `w_rec`** (the recurrent core). Mutations that
  touch `lr_W_ED` or `lr` change a *different* timescale — don't sweep them
  in the same slot as a schedule change, you won't know which axis moved
  pi_acc.
- **`lr_W_rec` is redundant with `lr` + `lr_schedule`** in most cases — the
  schedule overwrites it at every epoch. Setting `lr_W_rec` is only useful
  if you want a different epoch-0 value than `lr` (rare). Default: leave
  unset.

## Mutation log format (per iteration)

After each batch, append to working memory:

```
## Iter N (block B): [exploration | robustness]
Parent: iter_M_slot_K  (pi_acc=X.XXX, gt_R2=Y.YY at full T)
Hypothesis: "[testable claim about what the mutation should do]"
Slot 0: [parent/control]   pi_acc=X.XXX  gt_R2=Y.YY  gt_slope=Z.ZZ  fwhm=YY°  collapse=no  traj=e1=A e2=B e3=C e4=D e5=E
Slot 1: [knob -> value]    pi_acc=X.XXX  gt_R2=Y.YY  ...
...
Slot 9: [knob -> value]    pi_acc=X.XXX  gt_R2=Y.YY  ...
Best slot: K  ->  pi_acc=X.XXX  gt_R2=Y.YY
Verdict: [supported | falsified | inconclusive]
Next parent: iter_N_slot_K
```

When a slot collapses, note the epoch at which it dropped and the loss/cosd/norm
values at that epoch — this is the most informative diagnostic. **Also record
`gt_R2` at the last pre-collapse snapshot** — it tells you whether the
collapse happened with `W_rec` close to GT (numerical instability) or far
from GT (the optimiser walked off the connectome manifold).

## Winner config

At every block boundary, copy the best slot's config to
`config/drosophila_cx/drosophila_cx_pi_winner.yaml` with header. The
existing winner.yaml is from the bug era and is no longer authoritative —
overwriting it with the new B4 result is the goal. Note the pi_acc bar
is **0.95**, not 0.99, until B1+B2 establish what the post-fix
landscape actually supports.

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

- **The trainer's `tmp_training/evolution/` directory** has 6-panel
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


