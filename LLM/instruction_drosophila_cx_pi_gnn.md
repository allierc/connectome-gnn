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
