# Drosophila CX — Path Integration (TaskGNN variant)

## Goal

Find the **best training recipe** for the GNN-driven CX recurrent network
(`DrosophilaCxTaskGNN` in
[src/connectome_gnn/models/drosophila_cx_task_gnn.py](../src/connectome_gnn/models/drosophila_cx_task_gnn.py))
on the path-integration task. Given angular velocity ω(t) and a one-frame
initial-heading impulse at t=0, predict (cos θ_hd, sin θ_hd) at every frame.

This is the **hybrid model**: TaskRNN's encoder (`W_in` with PEN velocity
gate) and decoder (`W_out` cos/sin readout) are retained, but the linear
recurrent step `r @ W_rec.T` is replaced by a per-edge GNN update:

```
v        = h                                                  (B, N=156)   raw subthreshold state
msg_e    = W[e] · g_phi(v_src, a_src)^2                       (B, E≈10263, 1)
agg_j    = Σ_{dst(e)=j} msg_e                                 (B, N, 1)   scatter_add
rec_j    = f_theta(v_j, a_j, agg_j)                           (B, N)
τ·dh/dt  = rec + W_in·u                                       (no explicit -h leak; f_theta carries it)
y_hat    = W_out · σ(h) + b_out
```

**Learnables**:
- `W_in` (encoder, 156×3 with optional PEN gate)
- `W_out` (decoder, 2×156)
- `W` (per-edge weights, ≈10k — *sign-free, no Dale lock*)
- `a` (per-node embedding, 156×`embedding_dim`)
- `g_phi` MLP (edge message function), `f_theta` MLP (node update)

**Key conceptual difference vs the sign-locked RNN**: the recurrence is
**non-linear** — `f_theta` injects an MLP between aggregated messages
and `dh/dt`. The `W_rec` property exposed for the GT scatter and the
`loss_cos_distance` / `loss_norm_floor` regularisers only captures the
**linear** surface of the GNN, so anatomy regularisers act on per-edge
gains without touching the `f_theta` non-linearity.

**Primary metric**: `r_roll` — Pearson correlation between the unwrapped
decoded heading trajectory and the (monotone) ground-truth heading on a
deterministic-sweep rollout (ω = 60°/s constant, warmup 10 frames). Target
**`r_roll` ≥ 0.95** at full T=1000.

No secondary metrics. (`gt_R2` is still computed and printed in the
snapshot scatter for diagnostic context only — it is *not* a promotion
criterion. A GNN with `r_roll ≈ 1` and `gt_R2 ≈ 0.1` is a successful
result: the integration was carried by `f_theta`, not by per-edge `W`'s
match to the connectome. That finding is the *point* of the GNN
exploration, not a failure.)

The dataset is **fixed**: 100k train + 10k test trials × T=1000 frames at
dt=0.01s. Only training hyperparameters change.

## Budget

**60 iterations** (fresh run), 5 slots × 4 batches = 20 iter / block →
**3 blocks total**. Down from the 4-block / 10-slot default because (a)
the connectome-prior block was dropped (RNN exploration settled
`coeff_cos_distance: 0.0` as the `r_roll`-optimal value, inherited
here), and (b) the slot count is pruned to 5 — only the most informative
probe per axis is kept. Every block is a *targeted measurement* of one
axis, not a free sweep.

## Reference: what's transferred from the RNN winner

The sign-locked RNN exploration converged on
[`config/drosophila_cx/drosophila_cx_pi.yaml`](../config/drosophila_cx/drosophila_cx_pi.yaml),
which hit `r_roll = 1.000` on a single seed. **The GNN exploration starts
from those training-side knobs unchanged**; only the architecture and the
GNN-specific knobs differ.

Transferred from the RNN winner (do NOT re-sweep in B1/B2 unless a slot
collapses):
- `velocity_gate: pen_4scalar`
- `grad_clip_W: 2.5`
- `noise_recurrent_level: 0.05`
- `coeff_cos_distance: 0.0`, `coeff_norm_floor: 0.5`, `coeff_W_L1: 0.0`
- `n_steps_schedule: [100, 200, 300, 400, 500, 600, 800, 900, 1000, 1000]`
- `lr_W_rec_schedule: [2e-3, 2e-3, 1e-3, 1e-3, 5e-4, 4e-4, 3e-4, 2e-4, 5e-5, 5e-5]`
- `lr_W_ED: 5.0e-4`, `lr: 2.0e-3`

GNN-side defaults (these are the actual axes to explore):
- `embedding_dim: 2`
- `hidden_dim: 64`, `n_layers: 2` (g_phi)
- `hidden_dim_update: 64`, `n_layers_update: 2` (f_theta)
- `MLP_activation: tanh` (required for stability — see Notes)
- `g_phi_positive: true`
- `w_init_mode: zeros`, `w_init_scale: 0.01` (grokking regime)
- `coeff_f_theta_diff: 10.0` (∂f_θ/∂v < 0 prior — forces f_θ to learn the leak)
- `batch_size: 16` (GNN per-edge memory caps this lower than the RNN's 64)

## Available hyperparameters (search space)

These are the fields the agent may set per-slot in `training:` /
`graph_model:`. Anything else should NOT be touched unless explicitly noted.

### Three-group optimiser

The trainer splits parameters into three named groups with separate
learning rates. **`lr_W_rec_schedule` drives only the `w_rec` group** —
`w_ED` (encoder/decoder) and `other` (biases) stay constant at their
respective `lr_W_ED` / `lr` across epochs.

| Group   | Trainable params                                                                 | LR field    | Schedule?            |
| ------- | -------------------------------------------------------------------------------- | ----------- | -------------------- |
| `w_rec` | `W` (per-edge) + `a` (embedding) + `g_phi.*` + `f_theta.*`                       | `lr_W_rec`  | optional — `lr_W_rec_schedule` drives this if set |
| `w_ED`  | `W_in`, `W_out`, `_W_in_mlp.*`, `_W_out_mlp.*`, `v_pen{a,b}_{l,r}`               | `lr_W_ED`   | optional — `lr_W_ED_schedule` drives this if set  |
| `other` | biases (`b`, `b_out`) and anything else                                          | `lr`        | NO — constant         |

Each schedule is independent: a missing/empty schedule leaves that group
at its initial lr (constant across epochs). If `lr_W_rec` is unset it
falls back to `lr`; same for `lr_W_ED`.

### GNN core (PRIMARY axes — this is what the GNN exploration is about)

| Field                | Default | What it controls                                                                                                                                                       |
| -------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `embedding_dim`      | `2`     | Per-node embedding `a_i` dimension. Try {1, 2, 4, 8}.                                                                                                                  |
| `hidden_dim`         | `64`    | `g_phi` MLP width. Try {16, 32, 64, 128}.                                                                                                                              |
| `n_layers`           | `2`     | `g_phi` MLP depth. Try {2, 3, 4}.                                                                                                                                       |
| `hidden_dim_update`  | `64`    | `f_theta` MLP width. Try {16, 32, 64, 128}.                                                                                                                            |
| `n_layers_update`    | `2`     | `f_theta` MLP depth. Try {2, 3, 4}.                                                                                                                                     |
| `MLP_activation`     | `tanh`  | `relu` / `tanh` / `leaky_relu` / `soft_relu`. `tanh` bounds the MLP outputs (required since the GNN sees raw `v`, no σ-wrapping); `relu` is sparse but can saturate.   |
| `g_phi_positive`     | `true`  | Square `g_phi` output → positive messages. `false` lets messages take either sign on top of `W`. Breaks the "magnitude × sign" decomposition if `false`.               |
| `coeff_f_theta_diff` | `10.0`  | Negative-monotonicity prior on `∂f_θ/∂v` (forces f_θ to learn the leak). Try {0, 1, 5, 10, 25}. Higher = stiffer leak.                                                  |
| `w_init_mode`        | `zeros` | Per-edge `W` init. `zeros` → feedforward at t=0 (grokking regime); `randn_scaled` warm-starts at the edge of chaos.                                                    |
| `w_init_scale`       | `0.01`  | `randn_scaled` bound = scale/√n_edges. Ignored when `w_init_mode == zeros`.                                                                                            |

### Training infrastructure (transferred from the RNN winner — only revisit in B1 if a slot collapses)

| Field                   | Default                          | What it controls                                                                                                                                                      |
| ----------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lr`                    | `2e-3`                           | Constant lr for biases / `other` group.                                                                                                                               |
| `lr_W_rec`              | unset (→ `lr`)                   | Initial lr for w_rec; the schedule drives this group from epoch 1.                                                                                                    |
| `lr_W_ED`               | `5.0e-4`                         | Initial lr for W_in / W_out / velocity-gate scalars. Constant unless `lr_W_ED_schedule` is set. Try {1e-4, 5e-4, 1e-3, 2e-3}.                                          |
| `lr_W_rec_schedule`     | RNN-winner 10-epoch decay        | Per-epoch trajectory of the `w_rec` group only.                                                                                                                       |
| `lr_W_ED_schedule`      | unset (constant `lr_W_ED`)       | Per-epoch trajectory of the `w_ED` group only. Use this to anneal the encoder/decoder separately from `w_rec` — e.g. let W_in find the velocity-bump mapping early, then freeze. |
| `noise_recurrent_level` | `0.05`                           | Stddev of Gaussian noise on `h` per Euler step. Try {0, 1e-3, 1e-2, 5e-2, 1e-1}.                                                                                       |
| `grad_clip_W`           | `2.5`                            | Max-norm gradient clip. GNN can blow up more easily than the sign-locked RNN; keep the clip on.                                                                       |
| `n_steps_schedule`      | RNN-winner [100..1000]           | Per-epoch BPTT horizon.                                                                                                                                                |
| `n_epochs`              | `10`                             | Must match `len(lr_W_rec_schedule)` and `len(n_steps_schedule)`.                                                                                                       |
| `batch_size`            | `16`                             | GNN per-edge memory caps this lower than the RNN's 64. Try {8, 16, 32} only if memory allows.                                                                         |
| `coeff_W_L1`            | `0.0`                            | L1 on per-edge `|W|`. Try {0, 1e-6, 1e-5, 1e-4}.                                                                                                                       |

### Encoder gate + projection (inherited from RNN; usually leave alone)

| Field             | Default        | Role                                                                                                                |
| ----------------- | -------------- | -------------------------------------------------------------------------------------------------------------------- |
| `velocity_gate`   | `pen_4scalar`  | `"none"` / `"pen_only"` / `"pen_4scalar"`. RNN winner is `pen_4scalar`.                                              |
| `input_proj`      | `"matrix"`     | **Fixed at `"matrix"`** — MLP variant is not validated with the GNN core; do not change.                              |
| `output_proj`     | `"matrix"`     | **Fixed at `"matrix"`** — same reasoning.                                                                            |

### Connectome priors (operate on the *linear* W_rec surface only — `f_theta` is invisible to them)

| Field                | Default | Role                                                                                                                                       |
| -------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `coeff_cos_distance` | `0.0`   | Per-block cosine alignment of per-edge `W` to `W_con`. **Only constrains the linear surface — `f_theta` is unconstrained.**                |
| `coeff_norm_floor`   | `0.5`   | Soft floor on mean `|W|` per type-pair block. RNN winner is 0.5.                                                                            |
| `kappa_norm_floor`   | `0.05`  | Floor target for the norm-floor penalty.                                                                                                   |
| `coeff_tv_circular`  | `0.0`   | Circular TV on EPG/PEN ring firing rates.                                                                                                  |

### Things you must NOT change

- The `simulation:` block (data is on disk).
- The `task.path_integration` block.
- `signal_model_name` (must stay `drosophila_cx_pi_gnn`).
- `aggr_type` (irrelevant for this model).
- `input_proj` and `output_proj` (must stay `"matrix"`; MLP variants are
  unvalidated with the GNN core).
- Eval/snapshot fields (`snapshots_per_epoch`, `snapshot_omega_deg`) —
  diagnostics only.

## Metrics (per slot, per iteration)

Read from `<exploration_dir>/<slot_name>_analysis.log`. The trainer
writes `tmp_training/metrics.log` (CSV) per iteration with columns:
`iteration,epoch,loss,mse,cosd,norm,tv,l1S,pi_acc,fwhm_deg,r_roll,rmse_roll_deg,r_roll_1k`.

| Metric             | What it measures                                                                                          | Target               |
| ------------------ | --------------------------------------------------------------------------------------------------------- | -------------------- |
| `r_roll` (final)   | Pearson correlation on deterministic-sweep rollout (ω = 60°/s, warmup 10).                                | **≥ 0.95** at full T.|
| `r_roll` (per ep)  | End-of-epoch `r_roll` at the curriculum's `T_epoch`.                                                      | Monotonically high.  |
| `loss`             | Total training loss = mse + cosd + norm + tv + l1S + f_diff.                                              | Smooth, decreasing.  |
| `mse`              | Per-frame MSE on (cos, sin) target.                                                                       | Decreasing.          |
| `collapse_detected`| End-of-epoch `r_roll` drops by ≥0.4 between consecutive epochs.                                           | Should be `no`.      |

**Per-epoch trajectory is the most diagnostic signal.** For grokking
runs with `w_init_mode: zeros`, expect `r_roll ≈ 0` for several early
epochs, then a sudden jump — the metric `iter_first_above_0.5` (first
iter where `r_roll` crosses 0.5) is the cleanest single number to
compare grok speed across slots; log it where available.

## Causality rule

One or two parameters per slot. The pipeline forces 10 different seeds
when every slot is identical (robustness mode in B3).

## Block plan (60 iterations, 3 blocks × 20 iter, 5 slots/batch × 4 batches)

Parent at iter 1 = a clean GNN config built from the RNN winner's
training-side knobs + the GNN-side defaults listed in
[Reference: what's transferred from the RNN winner](#reference-whats-transferred-from-the-rnn-winner).
Suggested clean parent:

```yaml
graph_model:
  signal_model_name: drosophila_cx_pi_gnn
  velocity_gate: pen_4scalar
  embedding_dim: 2
  hidden_dim: 64
  n_layers: 2
  hidden_dim_update: 64
  n_layers_update: 2
  MLP_activation: tanh
  g_phi_positive: true
training:
  lr: 2.0e-3                # constant — biases / "other"
  lr_W_ED: 5.0e-4           # constant — W_in / W_out / velocity gates
  # lr_W_rec unset → starts at `lr`, then driven by lr_W_rec_schedule
  batch_size: 16
  noise_recurrent_level: 0.05
  grad_clip_W: 2.5
  coeff_cos_distance: 0.0
  coeff_norm_floor: 0.5
  coeff_W_L1: 0.0
  coeff_f_theta_diff: 10.0
  w_init_mode: zeros
  w_init_scale: 0.01
  n_epochs: 10
  n_steps_schedule: [100, 200, 300, 400, 500, 600, 800, 900, 1000, 1000]
  lr_W_rec_schedule:  [2e-3, 2e-3, 1e-3, 1e-3, 5e-4, 4e-4, 3e-4, 2e-4, 5e-5, 5e-5]
```

Three blocks total (the RNN exploration already settled the
connectome-prior trade-off — `coeff_cos_distance: 0.0` is optimal for
`r_roll`, so no separate priors block is needed here). Each block's 40
iterations stay on its own axis — do not drift.

| Block | Question | Slot layout (5 slots / batch) | Decision rule for the block boundary |
| ----- | -------- | ------------------------------ | ------------------------------------ |
| **1 — GNN convergence (stabilisation)** | *Does the GNN grok at all under the inherited curriculum? What noise / clip / I-O lr / f_θ-prior settings on top of the RNN winner are needed?* | s0 parent (control). s1 `noise_recurrent_level: 1e-2`. s2 `grad_clip_W: 1.0`. s3 `lr_W_ED: 1e-4` (slow encoder). s4 `coeff_f_theta_diff: 0.0` (drop the prior — does f_θ still learn a leak on its own?). | Promote the slot with the highest end-of-curriculum `r_roll` AND no collapse between consecutive epochs. If 0/5 cross `r_roll = 0.5` by the last epoch, escalate — the GNN may need a longer curriculum or different init. Otherwise the best slot becomes the **convergence parent** for B2. |
| **2 — GNN architecture, init, and sparsity** | *With the GNN converging, what minimum capacity for `a` / `g_phi` / `f_theta` keeps `r_roll`, and does the `W`-init choice or an L1 sparsity prior matter?* All GNN-specific axes the RNN exploration can't speak to are packed here, since priors are settled. | s0 B1 winner (bridge). s1 `embedding_dim: 4`. s2 `hidden_dim: 128` (both g_phi/f_θ). s3 `MLP_activation: relu`. s4 `w_init_mode: randn_scaled, w_init_scale: 1.0` (warm-start vs grokking). | Promote any slot that (i) keeps `r_roll ≥ 0.95` and (ii) is either *smaller* (parsimony — lower trainable param count) or *faster* to grok (lower `iter_first_above_0.5`). Tie-break: lower trainable params wins. |
| **3 — 5-seed robustness** | *Is the B2 winner seed-robust?* | All 5 slots = identical config (B2 winner). Pipeline auto-forces 5 different seeds. | Report mean ± std for `r_roll`. Save as the new `drosophila_cx_pi_gnn_winner.yaml` if mean `r_roll ≥ 0.95` AND no seed collapses. |

### Budget guard-rails

- **Total budget: 60 iterations** (3 blocks × 20 iter, 5 slots × 4 batches
  per block). Down from the standard 120/160 because (a) the
  connectome-prior block is settled by the RNN exploration
  (`coeff_cos_distance: 0.0` inherited), and (b) the slot count is pruned
  to 5 — one probe per axis, no duplicate-seed bridges. Trade-off: less
  statistical power per block; relies on the LLM's mutation loop to pick
  good variants across the 4 batches of each block.
- **B1 is the existential block.** If no slot in B1 reaches `r_roll ≥ 0.5`
  by the last epoch, do NOT proceed to B2 — widen B1 instead (more noise
  / clip variants, retry with `w_init_mode: randn_scaled`).
- **One axis per block.** B1 is stabilisation (noise / clip / lr_W_ED /
  f_θ_diff); B2 is GNN architecture + init + sparsity.
- **`r_roll ≥ 0.95` is the bar.** Don't promote slots that achieve high
  `pi_acc` but low `r_roll` — that pattern means the bump locked rather
  than integrated (see RNN exploration history).
- **`lr_W_rec_schedule` only affects `w_rec`** (recurrent core, including
  the GNN MLPs). Mutations that touch `lr_W_ED` or `lr` change a
  *different* timescale — don't sweep them in the same slot as a schedule
  change.
- **Don't change `velocity_gate`, curriculum, `lr_W_rec_schedule`,
  `coeff_cos_distance`, or `coeff_norm_floor` in any block** — these are
  resolved by the RNN exploration and re-touching them invalidates the
  inheritance.

## Mutation log format (per iteration)

After each batch, append to working memory:

```
## Iter N (block B): [exploration | robustness]
Parent: iter_M_slot_K  (r_roll=X.XXX at full T)
Hypothesis: "[testable claim about what the mutation should do]"
Slot 0: [parent/control]   r_roll=X.XXX  collapse=no  traj=e1=A e2=B e3=C e4=D e5=E ...
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
`config/drosophila_cx/drosophila_cx_pi_gnn_winner.yaml` with header:

```yaml
# Winner: drosophila_cx_pi_gnn_winner.yaml
# Source: iter_NNN_slot_KK  (final r_roll = X.XXX)
# Block: B  (focus: <focus>)
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - <one-sentence reason>
#   - <key knob change vs previous winner>
#
# Per-epoch trajectory: e1=A e2=B ... e10=J  (no collapse)
# Robustness: tested across N seeds, r_roll mean=X.XXX ± Y.YYY
```

## Notes / hints

- **Grokking is the default behaviour with `w_init_mode: zeros`.** Per-edge
  `W = 0` at t=0 → recurrence is pure feedforward until `|W|` grows past
  the bifurcation. Expect `r_roll ≈ 0` for the first several epochs, then
  a sudden jump. Don't kill a slot just because epochs 1–3 look flat;
  watch for the late-curriculum transition.
- **`f_theta` invisibility to anatomy regularisers.** The cos-distance /
  norm-floor regularisers and the `gt_R2` scatter only see the per-edge
  linear `W`. A circuit with `r_roll ≈ 1` and `gt_R2 ≈ 0.1` means the
  integration was carried by `f_theta`, not by `W`'s match to the
  connectome. **This is a finding, not a bug** — it characterises the
  GNN's degeneracy structure compared to the sign-locked RNN's.
- **`MLP_activation: tanh` is the default** because the GNN sees the raw
  state `v ≡ h` (no σ-wrapping like the RNN). Bounded outputs are
  required for stability; `relu` is worth trying in B2 but expect more
  saturation.
- **The snapshot panel scatter (`learned W` vs `GT W_con`)** is now
  informational only. With `coeff_cos_distance: 0.0` (inherited from
  the RNN winner), the GNN is free to walk off the connectome support;
  a low `gt_R2` in this exploration is expected, not a failure.

---

## Resumed exploration — 4 additional blocks (80 more iterations)

Triggered by the **soft-curriculum tail-loss finding (2026-05-22)**: adding
a tail-loss term `coeff_tail_loss = 0.05` (per-frame MSE = 1.0 for
`t < T_epoch` and 0.05 for `t ∈ [T_epoch, 2·T_epoch]`) eliminated the
late-time activity collapse and drove `r_roll_1k = 0.990` by epoch 2
(`T_epoch = 50`). This is a structural change to the trainer
(`_data_train_drosophila_cx_task`), not a hyperparameter sweep over an
existing knob, so the prior B1 / B2 conclusions need to be re-tested
with the tail anchor *on* before the winner is frozen for robustness.

### New parent (tail-loss enabled)

Use [`config/drosophila_cx/drosophila_cx_pi_gnn_tailloss.yaml`](../config/drosophila_cx/drosophila_cx_pi_gnn_tailloss.yaml)
as the iter-1 parent for the resumed run. Key fields vs the original
B1 winner + B2-batch-1 promotion:

```yaml
graph_model:
  embedding_dim: 4              # B2 batch-1 promotion (kept)
  # all other graph_model fields = B1 winner
training:
  coeff_tail_loss: 0.05         # NEW — drives the resumed exploration
  noise_recurrent_level: 0.01   # B1 winner (may be revisitable under tail)
  grad_clip_W: 1.0              # B1 winner (may be revisitable under tail)
  coeff_f_theta_diff: 15.0      # B1 winner (may be revisitable under tail)
  # 10-epoch budget + 10-epoch n_steps_schedule reaching 1000 (already in yaml)
```

The trainer rolls forward to `min(2·T_epoch, T_max)` and weights the
tail at `coeff_tail_loss`. Verify at iter 1 that `tail_loss=0.05` appears
in the trainer's startup log line; if it's `0.0`, the field didn't
propagate.

### Block plan (resumed — block numbers continue from B3)

| Block | Question | Slot layout (5 slots / batch) | Decision rule |
| ----- | -------- | ----------------------------- | ------------- |
| **B3 — Revisit B1+B2 axes with tail-loss on** | *Does the tail anchor change which knob settings are optimal? Specifically: is noise still needed? does the f_θ-diff prior still matter? does w_init_mode=zeros still beat warm-start?* Pack the most impactful axes from B1 and B2 into 4 batches. | Batch 1 — s0 parent. s1 `noise_recurrent_level: 0.0` (does tail replace noise?). s2 `coeff_f_theta_diff: 5.0` (looser leak prior). s3 `grad_clip_W: 2.5` (upper-end clip). s4 `w_init_mode: randn_scaled, w_init_scale: 0.5` (warm-start with tail). | Best slot by **r_roll_1k at epoch 10**, AND no epoch where r_roll_1k drops below the previous epoch by ≥ 0.2. Batches 2-4 mutate off the best-running slot following the same one-axis-per-slot rule (no joint mutations within a batch). |
| **B4 — Tune `coeff_tail_loss` × `lr_W_rec_schedule`** | *Is α = 0.05 optimal, or does a lighter (0.02) / heavier (0.1) tail help? Does the lr_W_rec_schedule need re-shaping now that the tail provides gradient on late-time dynamics?* | Batch 1 — s0 parent (`α = 0.05`, current schedule). s1 `coeff_tail_loss: 0.02`. s2 `coeff_tail_loss: 0.1`. s3 `lr_W_rec_schedule` slower decay (× 1.5 each entry except the flat tail). s4 `lr_W_rec_schedule` deeper decay (last 3 entries → 1e-5). | Best by **mean of last-3-epochs r_roll_1k** (so both peak and stability count). |
| **B5 — 5-seed CV robustness** | *Is the B4 winner seed-robust at T = 1000?* | All 5 slots = identical B4 winner. Pipeline auto-forces 5 different seeds. | Report mean ± std for r_roll_1k. **Save as the new `drosophila_cx_pi_gnn_winner.yaml` if mean r_roll_1k ≥ 0.95 AND no seed collapses (no seed r_roll_1k < 0.85).** Compare to the original B1 5-seed robustness (mean 0.931 ± 0.061, 3/5 ≥ 0.95) — the tail-loss winner should improve on this strictly. |
| **B6 — g_φ shape study (interpretability, not performance)** | *What does the firing-rate non-linearity g_φ actually look like, and how does it depend on (i) the squaring `g_phi(v)²` and (ii) the monotonicity prior `coeff_g_phi_diff`?* The current squared-and-free configuration gives a half-rectifier with ± degeneracy that obscures the learned shape (see [docs/drosophila.tex](../docs/drosophila.tex) Methods, paragraph on the GNN diagnostics). This block decouples the squaring from the prior. **Success criterion is qualitative**: the learned g_φ should be uniquely identified (no ± symmetry) and interpretable as a firing-rate non-linearity. r_roll_1k must stay ≥ 0.90 — this is a "characterise, don't break" block. | Batch 1 — s0 `g_phi_positive: true,  coeff_g_phi_diff: 0`   (current parent, ± degenerate). s1 `g_phi_positive: false, coeff_g_phi_diff: 0`   (un-squared, no prior; signed g_φ free to flip). s2 `g_phi_positive: false, coeff_g_phi_diff: 15`  (un-squared + Dale-conformant prior — the [tailloss_unsquared.yaml](../config/drosophila_cx/drosophila_cx_pi_gnn_tailloss_unsquared.yaml) config). s3 `g_phi_positive: true,  coeff_g_phi_diff: 15`  (squared + monotonicity — constrains the half-rectifier shape). s4 `g_phi_positive: false, coeff_g_phi_diff: 5`   (un-squared + lighter prior — probes the prior magnitude). | **No promotion to winner.** This is a side-study. Save the four `<log_dir>/tmp_training/function/g_phi/` snapshot directories side-by-side for visual inspection. Record verdicts per the H-g hypotheses below. If s2 (un-squared + prior, lam=15) gives a cleanly sigmoidal g_φ AND r_roll_1k ≥ 0.95, fold it into the methods section of [drosophila.tex](../docs/drosophila.tex) as the canonical configuration for diagnostic plots. |

### Hypotheses for B6 (g_φ shape)

- **H-g1**: with squaring on (`g_phi_positive: true`), the plotted g_φ
  shape is unidentified — the optimiser picks an arbitrary sign
  convention because g_φ and −g_φ give identical squared messages. Test:
  compare s0 vs s3. If both reach the same r_roll_1k but the learned
  g_φ shapes differ in sign / inflection point, the ± degeneracy is
  confirmed; the monotonicity prior partially constrains the shape but
  cannot break the ± symmetry.
- **H-g2**: without squaring and without the prior (`g_phi_positive:
  false`, `coeff_g_phi_diff: 0`), the unconstrained signed g_φ either
  (a) converges to a sigmoidal shape spontaneously, (b) finds a
  pathological solution that crosses zero in the operating range (Dale
  violation per edge), or (c) collapses to the trivial g_φ ≡ 0 fixed
  point (vanishing-gradient hypothesis). Test: read off s1's learned
  shape and the per-edge sign-flip rate.
- **H-g3**: without squaring and with the prior (`g_phi_positive:
  false`, `coeff_g_phi_diff: 15`), g_φ converges to a clean monotone
  sigmoid-like firing-rate non-linearity with the per-edge sign of
  synaptic drive aligned with the connectome edge sign. This is the
  "expected" outcome per the methods section. Test: read off s2's
  learned shape; check that g_φ has the same sign at the operating-range
  mean voltage as at +∞.
- **H-g4**: the squaring is the dominant identifiability problem; the
  monotonicity prior alone cannot fix it under squaring. Test: compare
  s3 (squared + lam=15) vs s2 (un-squared + lam=15). If s2 gives a
  cleaner shape, the un-squaring is the key intervention.
- **H-g5**: prior magnitude has a sweet spot. Test: compare s2 (lam=15)
  vs s4 (lam=5). If s4 is "looser" but still monotone, lam=5 is the
  more parsimonious choice; if s4 is non-monotone, lam=15 is necessary.

### Slot 0 reference (B6 parent)

Use the **post-B4 winner** as the structural parent for B6, varying
only `g_phi_positive` and `coeff_g_phi_diff` across slots. If B4 hasn't
landed yet, use the current best published config
([drosophila_cx_pi_gnn_tailloss.yaml](../config/drosophila_cx/drosophila_cx_pi_gnn_tailloss.yaml)).
Do NOT vary any other knob in B6 — this is a 2-knob × 5-slot study,
not a free sweep.

### Pre-registered hypotheses (write them in memory before each B3-B4 batch runs)

- **H-noise**: with the tail anchor providing late-time gradient,
  `noise_recurrent_level = 0` should no longer collapse. If H-noise holds,
  the noise knob can be dropped → simpler config.
- **H-f_diff**: the leak-prior at 15 may be over-regularising once the
  tail loss is on (the tail loss penalises any drift to zero, which is
  the failure mode `coeff_f_theta_diff` was added to fix). Test 5 → 0.
- **H-warmstart**: `w_init_mode: zeros` was preferred to avoid early
  divergence; with the tail anchor stabilising late-time dynamics, a
  warm-start `randn_scaled` may grok faster.
- **H-α**: α = 0.05 was set by a single observation. Lighter (0.02)
  may give an even cleaner separation between supervised and anchor
  gradients; heavier (0.1) may over-pull and degrade the supervised
  window. Brackets the 0.05 working point.
- **H-schedule**: with the tail loss driving the entire rollout,
  the late-epoch finetune (currently 5e-5 × 2) may need to be slower
  (longer at higher lr) to fully exploit the new gradient signal.

### Budget guard-rails (resumed)

- **Total resumed budget: 80 iterations** (B3 = 20, B4 = 20, B5 = 20,
  B6 = 20). B5 only needs 5 iters in principle (1 batch × 5 seeds);
  the extra 15 are held in reserve as a 5-seed bridge on a near-winner
  candidate (see B5 decision rule).
- **B3 is the existential block.** If 0/5 slots in B3 batch 1 reach
  r_roll_1k ≥ 0.95 by epoch 10, the tail-loss intervention is less
  general than the initial run suggested — escalate before B4.
- **`coeff_tail_loss` is the only NEW axis** for B3-B4. Don't sweep
  curriculum (`n_steps_schedule`) or encoder/decoder knobs unless a
  slot collapses; those are still inherited from the RNN winner.
- **B5 winner replaces the previous `drosophila_cx_pi_gnn_winner.yaml`**
  if it strictly improves on the prior B1-lineage 5-seed robustness
  (mean 0.931 ± 0.061). Otherwise the prior winner stands.
- **B6 is interpretability, not promotion.** Don't update the winner
  yaml from B6 outcomes — the role of B6 is to characterise the g_φ
  identifiability problem, not to find a better performer. The
  reading-off of g_φ shapes is the deliverable.
- **B6 can run in parallel with B5** if you want — it doesn't depend
  on the B4 winner being final (uses post-B4 winner as parent but
  varies only g_φ knobs that are orthogonal to everything B3-B4
  optimised). Run sequentially if cluster capacity is tight.
