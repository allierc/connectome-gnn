# Drosophila CX — Path Integration (TaskGNN variant)

## Goal

Find the **best training recipe** for the GNN-driven CX RNN (`TaskGNN` in
[src/connectome_gnn/models/task_gnn.py](../src/connectome_gnn/models/task_gnn.py))
on the path-integration task. Given angular velocity ω(t) and a one-frame
initial-heading impulse at t=0, predict (cos θ_hd, sin θ_hd) at every frame.

This is the **hybrid model**: TaskRNN's encoder (`W_in` with PEN-only
velocity gate) and decoder (`W_out` cos/sin readout) are retained, but
the linear `r @ W_rec` recurrent step is replaced by a per-edge GNN
update:

```
r        = σ(h)                                          (B, N=156)
msg_e    = W_edge[e] · g_phi(r_src, a_src)^2             (B, E=10263, 1)
agg_j    = Σ_{dst(e)=j} msg_e                            (B, N, 1)   scatter_add
rec_j    = f_theta(r_j, a_j, agg_j)                      (B, N)
τ·dh/dt  = -h + rec + W_in·u + b
y_hat    = W_out·σ(h) + b_out
```

**Learnables**:
- `W_in` (encoder, 156×3 with optional PEN gate)
- `W_out` (decoder, 2×156)
- `W` (per-edge, 10 263, **sign-free** — no Dale lock)
- `a` (per-node embedding, 156×`embedding_dim`)
- `g_phi` MLP (edge message function), `f_theta` MLP (node update)

**Key conceptual difference vs TaskRNN sign_locked**: the recurrence is
**non-linear** — `f_theta` injects an MLP between aggregated messages
and `du/dt`. The `W_rec` property we expose (used by the GT scatter and
`loss_cos_distance` / `loss_norm_floor`) only captures the **linear**
surface of the GNN, so anatomy regularisers act on per-edge gains
without touching the `f_theta` non-linearity.

**Primary metric**: `pi_acc` (mean cosine similarity, ≥ 0.95 target at
full T=1000).

**Secondary metric**: `gt_R2`, `gt_slope` — linear fit between learned
per-edge `W` and GT `W_con` on the non-zero edges (rendered in the
bottom-right of every snapshot in `tmp_training/evolution/step_*.png`).
**Always report both.**

The dataset is **fixed**: 100k train + 10k test trials × T=1000 frames at
dt=0.01s. Only training hyperparameters change.

## Budget

**160 iterations** (fresh run), 10 slots × 4 batches = 40 iter / block →
**4 blocks total**. There is no room to re-solve curriculum or LR
issues; both are inherited from the TaskRNN winner and are known to
work. Every block targets one or two axes that are **specific to the
GNN variant**.

## Baseline behaviour (what's known)

The parent config
[`drosophila_cx_pi_gnn_winner.yaml`](../config/drosophila_cx/drosophila_cx_pi_gnn_winner.yaml)
starts with `w_init_mode: zeros, w_init_scale: 0.01` — per-edge `W=0`
at t=0 → recurrence is pure feedforward until `|W|` grows past the
bifurcation. This matches the **grokking regime** observed for TaskRNN
with `w_init_mode=zeros` (block-9 analysis): pi_acc sits near 0 for
~half the curriculum, then jumps to ≥0.9 within one epoch.

**Open questions this run must answer**:

1. **Does the GNN model grok at all** with the inherited TaskRNN
   curriculum + lr schedule? `f_theta` adds non-linearity that may
   require different optimisation.
2. **Does `gt_R2` track `pi_acc`** in the GNN? Or does the non-linear
   `f_theta` carry the integration, leaving per-edge `W` essentially
   untrained / arbitrary?
3. **How much capacity does `g_phi` / `f_theta` need**? Tiny MLPs
   (`hidden_dim=16`) vs medium (`64`) vs wider (`128`)?
4. **What does `g_phi_positive=False` give**? Allows negative messages
   on top of negative `W`; could break the "magnitude × sign" decomposition.

## Available hyperparameters (search space)

### GNN core (PRIMARY axes — what this exploration is about)

| Field             | Default       | What it controls                                                                                                                                                       |
| ----------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `embedding_dim`   | `2`           | Per-node embedding `a_i` dimension. Try {1, 2, 4, 8}.                                                                                                                  |
| `hidden_dim`      | `64`          | `g_phi` MLP width. Try {16, 32, 64, 128}.                                                                                                                              |
| `n_layers`        | `2`           | `g_phi` MLP depth. Try {2, 3, 4}.                                                                                                                                       |
| `hidden_dim_update` | `64`        | `f_theta` MLP width. Try {16, 32, 64, 128}.                                                                                                                            |
| `n_layers_update` | `2`           | `f_theta` MLP depth. Try {2, 3, 4}.                                                                                                                                     |
| `MLP_activation`  | `relu`        | `relu` / `tanh` / `leaky_relu` / `soft_relu`. `tanh` saturates → smoother grad; `relu` is sparse — both worth comparing.                                               |
| `g_phi_positive`  | `true`        | Square `g_phi` output → positive messages (NeuralGNN default). `false` lets messages take either sign on top of `W`.                                                   |
| `w_init_mode`     | `zeros`       | Per-edge `W` init. `zeros` (grokking regime), `randn_scaled` (warm-start at edge of chaos), `uniform_scaled`, `randn`.                                                  |
| `w_init_scale`    | `0.01`        | `randn_scaled`/`uniform_scaled`: bound = scale/√n_edges. Try {0.5, 1.0, 5.0}. For `zeros` mode this is ignored.                                                        |

### Encoder gate (inherited from TaskRNN)

| Field            | Default        | Role                                                                                                                |
| ---------------- | -------------- | -------------------------------------------------------------------------------------------------------------------- |
| `velocity_gate`  | `pen_only`     | `"none"` (W_in free) / `"pen_only"` (mask velocity to 42 PEN rows, per-unit free) / `"pen_4scalar"` (strict 4 scalars). |
| `input_proj`     | `"matrix"`     | `"matrix"` or `"mlp"`. MLP not yet validated with the GNN core.                                                     |
| `output_proj`    | `"matrix"`     | Same options.                                                                                                       |

### Connectome priors (act on the *linear* W_rec surface only)

| Field                | Default | Role                                                                                                                                       |
| -------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `coeff_cos_distance` | `0.0`   | Per-block cosine alignment of per-edge `W` to `W_con`. **Note: `f_theta` is invisible to this — it only constrains the linear surface.**   |
| `coeff_norm_floor`   | `0.0`   | Soft floor on mean `\|W\|` per type-pair block.                                                                                              |
| `kappa_norm_floor`   | `0.05`  | Floor target.                                                                                                                              |
| `coeff_tv_circular`  | `0.0`   | Circular TV on EPG/PEN ring firing rates.                                                                                                  |
| `coeff_W_L1`         | `0.0`   | L1 on `\|W_edge\|`. `model.S` is aliased to `model.W` so this hits per-edge magnitudes. Try {1e-7, 1e-6, 1e-5, 1e-4}.                       |

### Stability + curriculum (inherited from TaskRNN winner — only revisit if a slot collapses)

| Field                   | Default                          | Role                                                                                  |
| ----------------------- | -------------------------------- | -------------------------------------------------------------------------------------- |
| `grad_clip_W`           | `1.0`                            | GNN can blow up more easily than sign_locked TaskRNN; the default has a safety clip. |
| `noise_recurrent_level` | `0.0`                            | Stddev of Gaussian noise on `h` per Euler step.                                       |
| `n_steps_schedule`      | `[100, 200, 300, 500, …, 1000]`  | Per-epoch BPTT horizon. Don't touch unless a slot fails to grok.                      |
| `lr_schedule`           | `[5e-3, 1e-3, 1e-3, …, 5e-5]`    | Per-epoch lr. Don't touch unless a slot fails to grok.                                |
| `n_epochs`              | `10`                             |                                                                                       |
| `batch_size`            | `64`                             |                                                                                       |
| `data_augmentation_loop`| `2`                              | DAL=2 doubles iters/epoch via repeated independent shuffles.                          |

### Things you must NOT change

- The `simulation:` block (dataset is on disk).
- The `task.path_integration` block.
- `signal_model_name` (must stay `drosophila_cx_pi_gnn`).
- `aggr_type` (irrelevant for this model).
- Eval/snapshot fields (`snapshots_per_epoch`, `snapshot_n_steps`,
  `snapshot_omega_deg`) — diagnostics only.

## Metrics (per slot, per iteration)

Read from `<exploration_dir>/<slot_name>_analysis.log` after training.
The trainer writes `tmp_training/metrics.log` (CSV) per iteration; the
snapshot scatter title carries `gt_R2`/`gt_slope`.

| Metric              | What it measures                                                                  | Target                           |
| ------------------- | --------------------------------------------------------------------------------- | -------------------------------- |
| `pi_acc` (final)    | Mean cos similarity decoded vs true HD on full test set, full T.                  | **≥ 0.95** at full T.            |
| `pi_acc` (per epoch)| End-of-epoch `pi_acc` at the curriculum's `T_epoch`.                              | Monotonically high, no collapse. |
| `fwhm_deg`          | Bump width in degrees.                                                            | < 180°.                          |
| `gt_R2`             | Linear R² between per-edge `W` and `W_con` (non-zero edges only).                 | (informational, no target)       |
| `gt_slope`          | Linear-fit slope, same set.                                                       | (informational)                  |
| `loss`              | Total training loss.                                                              | Smooth, decreasing.              |
| `mse`               | Per-frame MSE on (cos, sin) target.                                               | Tracks `1 - pi_acc`.             |
| `cosd`, `norm`, `tv`| Per-block reg values.                                                             | Should not climb.                |
| `collapse_detected` | End-of-epoch `pi_acc` drops by ≥0.4.                                              | Should be `no`.                  |

**Per-epoch trajectory is still the most diagnostic signal.** For grokking
runs, expect `e1=0 e2=0 e3≈0 e4≈0.6 e5≈0.9` or similar — the transition
lives in one epoch.

## Causality rule

One or two parameters per slot. The pipeline forces 10 different seeds
when every slot is identical (robustness mode).

## Block plan (160 iterations, 4 blocks × 40 iter, 10 slots/batch × 4 batches)

Parent at iter 1 = current
[`drosophila_cx_pi_gnn_winner.yaml`](../config/drosophila_cx/drosophila_cx_pi_gnn_winner.yaml).
**B1 must first confirm the GNN trains at all** — if no slot in B1 reaches
pi_acc ≥ 0.8 by epoch 10, the curriculum needs adjustment (escalate to a
human before launching B2).

| Block | Question | Slot layout (10 slots) | Decision rule |
| ----- | -------- | ----------------------- | ------------- |
| **1 — Does the GNN grok? Capacity sanity.** | *With the inherited TaskRNN curriculum, can the GNN learn integration? And what minimum capacity is needed?* | s0–s2 parent (3 seeds — confirm baseline). s3 `embedding_dim: 1`. s4 `embedding_dim: 4`. s5 `hidden_dim: 32` (both g_phi/f_theta). s6 `hidden_dim: 128`. s7 `MLP_activation: tanh`. s8 `w_init_mode: randn_scaled, w_init_scale: 1.0` (warm-start vs grokking). s9 `g_phi_positive: false`. | At least one slot must hit pi_acc ≥ 0.8 by epoch 10. If parent fails (s0–s2 all < 0.5), escalate. Best slot (highest pi_acc, ties broken by lower `iter_first_above_0.5`) becomes B2's parent. |
| **2 — Encoder × capacity interaction.** | *Once the GNN is learning, does the `velocity_gate` choice or the f_theta/g_phi balance dominate?* | s0–s1 B1 winner (2 seeds — bridge). s2 `velocity_gate: none` (free W_in). s3 `velocity_gate: pen_4scalar` (strict). s4 `hidden_dim_update: 128, n_layers_update: 3` (bigger f_theta). s5 `hidden_dim_update: 16` (tiny f_theta). s6 `n_layers: 3` (bigger g_phi). s7 `embedding_dim: 8`. s8 `MLP_activation: leaky_relu`. s9 `MLP_activation: soft_relu`. | Promote any slot that (i) keeps pi_acc ≥ 0.95 across its seeds and (ii) is either a *smaller* model (parsimony) or *materially faster* (lower `iter_first_above_0.5`). |
| **3 — Connectome priors on the linear surface.** | *Does `coeff_cos_distance > 0` push `gt_R2` up without hurting pi_acc, given that `f_theta` is invisible to it?* This answers your earlier question: can the GNN be made to look biological while still solving the task? | s0–s1 B2 winner (2 seeds — bridge). s2 `coeff_cos_distance: 0.1`. s3 `coeff_cos_distance: 0.25`. s4 `coeff_cos_distance: 0.5`. s5 `coeff_cos_distance: 1.0`. s6 `coeff_cos_distance: 2.0`. s7 `coeff_norm_floor: 0.25, kappa_norm_floor: 0.05`. s8 `coeff_W_L1: 1e-5`. s9 `coeff_W_L1: 1e-4`. | Plot `gt_R2` vs `coeff_cos_distance` and report the Pareto front (pi_acc vs `gt_R2`). The **lowest** `coeff_cos_distance` that achieves pi_acc ≥ 0.99 AND `gt_R2` ≥ 0.3 becomes the candidate winner. |
| **4 — 10-seed robustness.** | *Is the B3 winner seed-robust?* | All 10 slots = identical (B3 winner config). Pipeline forces 10 seeds. | Report mean ± std for `pi_acc`, `fwhm_deg`, `gt_R2`, `gt_slope`. Promote to `drosophila_cx_pi_gnn_winner.yaml` if mean pi_acc ≥ 0.99 AND no seed has `collapse_detected: yes`. |

### Budget guard-rails

- **Don't re-tune curriculum or LR.** The TaskRNN winner's schedule is
  the parent; only revisit if multiple B1 slots fail to grok.
- **One axis per block** — Block 3 sweeps one coefficient at a time
  across slots, NOT in combinations.
- **Freeze `velocity_gate` and capacity** after B2; don't re-open them in
  B3 or B4.
- **Always log `gt_R2` and `gt_slope` per slot** — they're the entire
  point of B3. They make it into the mutation log.

## Mutation log format (per iteration)

After each batch, append to working memory:

```
## Iter N (block B): [exploration | robustness]
Parent: iter_M_slot_K  (pi_acc=X.XXX, gt_R2=Y.YY at full T)
Hypothesis: "[testable claim about what the mutation should do]"
Slot 0: [parent/control]   pi_acc=X.XXX  gt_R2=Y.YY  gt_slope=Z.ZZ  fwhm=YY°  collapse=no  traj=e1=A e2=B e3=C e4=D e5=E ...
Slot 1: [knob -> value]    pi_acc=X.XXX  gt_R2=Y.YY  ...
...
Slot 9: [knob -> value]    pi_acc=X.XXX  gt_R2=Y.YY  ...
Best slot: K  ->  pi_acc=X.XXX  gt_R2=Y.YY
Verdict: [supported | falsified | inconclusive]
Next parent: iter_N_slot_K
```

When a slot collapses, note the epoch at which it dropped and the
loss/cosd/norm values at that epoch. **Also record `gt_R2` at the last
pre-collapse snapshot** — it tells you whether the collapse happened
with `W` close to GT (numerical instability in `f_theta`) or far from
GT (optimiser walked off the connectome manifold).

## Winner config

At every block boundary, copy the best slot's config to
`config/drosophila_cx/drosophila_cx_pi_gnn_winner.yaml` with header:

```yaml
# Winner: drosophila_cx_pi_gnn_winner.yaml
# Source: iter_NNN_slot_KK  (final pi_acc = X.XXX, gt_R2 = Y.YY, fwhm = ZZ°)
# Block: B  (focus: <focus>)
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - <one-sentence reason>
#   - <key knob change vs previous winner>
#
# Per-epoch trajectory: e1=A e2=B ... e10=J  (no collapse)
# Robustness: tested across N seeds, pi_acc mean=X.XXX ± Y.YYY,
#             gt_R2 mean=A.AA ± B.BB
```

## Notes / hints

- **Grokking is the default**: `w_init_mode: zeros` makes the recurrence
  feedforward at t=0. Expect pi_acc near 0 for the first several epochs,
  then a sudden jump. The metric `iter_first_above_0.5` (first iter where
  pi_acc crosses 0.5) is the cleanest single number to compare grok speed
  across slots — log it.
- **`f_theta` invisibility**: the cos-distance / norm-floor regularisers
  and the `gt_R2` scatter only see the **linear** per-edge `W`. A circuit
  with high `pi_acc` and low `gt_R2` means the integration was carried by
  `f_theta`, not by `W`'s match to the connectome. This is a *finding*,
  not a bug — it's what distinguishes the GNN's degeneracy structure from
  TaskRNN's sign-locked one.
- **`tmp_training/evolution/step_*.png`** shows the 6 snapshot panels
  (GT W_con, learned `W` placed at edges, EPG/PEN rasters, ω+HD overlay)
  plus pi_acc trace, RMSE trace, and the GT-vs-learned scatter with
  slope and R². The bottom-right scatter is the **primary diagnostic**
  for this run.
- **Don't expect `gt_R2 → 1`** even with strong `coeff_cos_distance`. The
  GNN can satisfy the cos-distance prior with any *scaling* of `W` that
  preserves direction (the loss is scale-invariant); `gt_R2` measures
  linear fit including slope and so caps below 1 unless `gt_slope ≈ 1`
  too. Look at both.
