# FlyVis hidden-10% consensus-NGP HP sweep — push R²W in phase 1, hold it across the injection, recover joint stability in phase 2

## Goal

Optimize GNN + spatial-NGP hyperparameters on
`flyvis_noise_005_hidden_010_blank50_consensus_ngp` (σ=0.05, 10% non-retina
hidden, blank-prefix=50%) **for the full training trajectory, not just the
endpoint**. The scoring function explicitly rewards three behaviors that occur
at different phases of training:

1. **Phase 1 (warmup, NGP off, `[0, 0.20·Niter)`)** — push `conn_R²` (R²W) as
   high as possible. The GNN is training on visible-only voltages with
   hidden voltages silenced (v_h=0), the NGP backbone trains in parallel
   only via the anchor loss. This phase determines the "best W ever
   achievable" before any injection-induced shift.
2. **Phase 2 (warmup-with-NGP-shaping, alpha=0, `[0.20, 0.50·Niter)`)** —
   keep R²W as **stable** as possible. The hidden-voltage loss is on but
   alpha=0, so the NGP-hidden output is being shaped against
   GNN(v_h=0)-targets without yet contaminating the GNN forward. R²W
   should not regress here.
3. **Phase 3 (ramp + injected, `[0.50·Niter, end]`)** — survive the
   injection switch and reach a **stable joint** GNN+NGP regime where R²W
   stays high AND `hidden_pearson` / `anchor_pearson` keep climbing. This
   is where the consensus baseline currently sees a ~0.2 conn_R² drop at
   the injection switch followed by a slow partial recovery.

**End-of-training metrics alone are not the score.** A run that reaches
final conn_R²=0.55 by collapsing to 0.40 mid-training and clawing back is
worse than a run that holds 0.70 through phase 1, drops only to 0.65 at
the switch, and recovers to 0.68. The composite score below penalizes
mid-training regressions explicitly.

## Composite trajectory score

For each slot, parse the **full** `metrics.log` (the trajectory, not just
the last row) and compute:

```
P1_peak       = max  conn_R²  for iter in [0, 0.20·Niter)            # how high phase-1 W gets
P1_end        = mean conn_R²  for iter in [0.18·Niter, 0.20·Niter)   # W at end of phase 1
P2_min        = min  conn_R²  for iter in [0.20·Niter, 0.50·Niter)   # worst R²W during phase-2 shaping
P2_end        = mean conn_R²  for iter in [0.48·Niter, 0.50·Niter)   # W at end of phase 2 (just before ramp)
P3_drop       = P2_end - min conn_R²  for iter in [0.50·Niter, 0.62·Niter)   # injection-switch impact
P3_final      = mean conn_R²  for iter in [0.95·Niter, 1.00·Niter)   # final W with NGP fully on
hid_final     = mean hidden_pearson_mean from nnr_pearson.log,  iter in [0.90·Niter, 1.00·Niter)
anc_final     = mean anchor_pearson_mean from nnr_pearson.log,  iter in [0.90·Niter, 1.00·Niter)

score_phase1  = P1_end                                # higher = better
score_phase2  = P2_end - max(0, P2_end - P2_min)      # P2_end with a stability penalty
score_phase3  = P3_final - 0.5·max(0, P3_drop)        # P3_final with switch-drop penalty (drop by 0.10 ≡ -0.05)
score_joint   = min(P3_final, hid_final, anc_final)   # all three must be healthy at the end

composite = 0.30·score_phase1 + 0.20·score_phase2 + 0.30·score_phase3 + 0.20·score_joint
```

A slot that gets `P1_end=0.70, P2_end=0.68, P2_min=0.65, P3_final=0.66, P3_drop=0.04, hid=0.20, anc=0.55`
scores `0.30·0.70 + 0.20·(0.68 − 0.03) + 0.30·(0.66 − 0.5·0.04) + 0.20·min(0.66,0.20,0.55) = 0.21 + 0.13 + 0.197 + 0.04 = 0.577`.
A slot that gets the same `P1_end=0.70` but collapses to `P2_min=0.30,
P3_final=0.45, P3_drop=0.20, hid=0.05, anc=0.30` scores
`0.21 + 0.20·(0.65 − 0.40) + 0.30·(0.45 − 0.10) + 0.20·0.05 = 0.21 + 0.05 + 0.105 + 0.01 = 0.375` — worse despite the same phase-1 peak.

This is the only score that matters. Do NOT make decisions from a single
last-line read of metrics.log.

## Log files — REQUIRED reading every iteration

The training engine writes two trajectory logs per slot. **Both must be
read for every slot at every iteration.** A log entry that does not quote
trajectory numbers from these files is malformed.

### Per-slot paths

If the pipeline injects a `SLOT_LOG_DIRS` block at the top of your prompt,
use those paths. Otherwise, fall back to:

```
<DATA_ROOT>/log/fly/flyvis_noise_005_hidden_010_blank50_consensus_ngp_Claude_<NN>/tmp_training/metrics.log
<DATA_ROOT>/log/fly/flyvis_noise_005_hidden_010_blank50_consensus_ngp_Claude_<NN>/tmp_training/nnr_pearson.log
```

where `<DATA_ROOT>` is `/groups/saalfeld/home/allierc/GraphData` and
`<NN>` is `00`, `01`, `02`, `03` for the four slots.

### `metrics.log` schema

CSV header:
```
iteration, connectivity_r2, vrest_r2, tau_r2, hidden_nnr_pearson,
anchor_nnr_pearson, vrest_r2_clean, n_out_vrest, n_total_vrest,
tau_r2_clean, n_out_tau, n_total_tau
```
- 1 row written every ~100 iters by the in-training quick metric refresh,
  plus dense rows from `plot_training_flyvis` at every plot checkpoint.
- The `connectivity_r2` column is the trajectory of R²W. Use it for
  P1_peak, P1_end, P2_min, P2_end, P3_drop, P3_final.
- `hidden_nnr_pearson`/`anchor_nnr_pearson` here are the same quantities
  that `nnr_pearson.log` records at higher resolution; prefer the latter
  for hid_final / anc_final.

### `nnr_pearson.log` schema

CSV header:
```
iteration, hidden_pearson_mean, hidden_pearson_std,
anchor_pearson_mean, anchor_pearson_std
```
- Written every `_NGP_QUICK_FREQ=100` iters by `_quick_ngp_pearson`
  (subset of 64 neurons × 256 frames per refresh, no wall-time cost).
- This is the high-frequency NGP fit signal — use it to detect the
  injection-switch dip in `hidden_pearson_mean` (it should NOT drop at
  the phase 2→3 ramp; if it does, the NGP backbone got destabilized by
  the new gradient path) and to verify final convergence.
- `*_std` is the spread across the 64-neuron sample; high std at
  comparable mean usually signals memorization of a few easy neurons
  while the rest of the population is at 0.

### Required parse code (Python, run via Bash)

The agent must execute this for every slot at every iteration and quote
the resulting dict in the iteration log:

```python
import csv, statistics, sys, glob

def parse(slot_dir, Niter):
    m_path = f"{slot_dir}/tmp_training/metrics.log"
    n_path = f"{slot_dir}/tmp_training/nnr_pearson.log"
    metrics = list(csv.DictReader(open(m_path)))
    nnr     = list(csv.DictReader(open(n_path)))

    def w(rows, lo, hi, col):
        vals = [float(r[col]) for r in rows
                if lo <= int(r['iteration']) < hi
                and r[col] not in ('', 'nan')]
        return vals

    p1 = w(metrics, 0,             int(0.20*Niter), 'connectivity_r2')
    p1_end = w(metrics, int(0.18*Niter), int(0.20*Niter), 'connectivity_r2')
    p2 = w(metrics, int(0.20*Niter), int(0.50*Niter), 'connectivity_r2')
    p2_end = w(metrics, int(0.48*Niter), int(0.50*Niter), 'connectivity_r2')
    p3_dip = w(metrics, int(0.50*Niter), int(0.62*Niter), 'connectivity_r2')
    p3_final = w(metrics, int(0.95*Niter), Niter+1, 'connectivity_r2')

    hid = w(nnr, int(0.90*Niter), Niter+1, 'hidden_pearson_mean')
    anc = w(nnr, int(0.90*Niter), Niter+1, 'anchor_pearson_mean')

    P1_peak = max(p1) if p1 else float('nan')
    P1_end  = statistics.mean(p1_end) if p1_end else float('nan')
    P2_min  = min(p2) if p2 else float('nan')
    P2_end  = statistics.mean(p2_end) if p2_end else float('nan')
    P3_drop = P2_end - min(p3_dip) if (p3_dip and not isinstance(P2_end,float) is False) else float('nan')
    P3_final = statistics.mean(p3_final) if p3_final else float('nan')
    hid_final = statistics.mean(hid) if hid else float('nan')
    anc_final = statistics.mean(anc) if anc else float('nan')
    return dict(P1_peak=P1_peak, P1_end=P1_end, P2_min=P2_min, P2_end=P2_end,
                P3_drop=P3_drop, P3_final=P3_final,
                hid_final=hid_final, anc_final=anc_final)
```

Apply this per slot, then compute `composite` per slot, then
mean/min/CV across the 4 slots. Quote the per-phase numbers in every log
entry, not just the composite.

### What "trajectory" looks like in practice

A healthy run on the consensus baseline has roughly:
```
iter      conn_R²   hid_pearson   anc_pearson    phase
1                   nan           nan            phase 1 start (W untrained, NGP silent)
21400 (P1→P2)       nan/0.0       0.4 → 0.5      phase 1 end — anchor backbone trained, hidden=0
53500 (P2→ramp)     ~0.0          ~0.5           phase 2 end — NGP shaped, hidden voltage non-degenerate
~70000             0.55-0.65      0.55-0.65      ramp absorbed, joint signal climbing
107000             0.65-0.75      0.65-0.80      final
```
A *broken* run shows: P1 reaches 0.7, then `connectivity_r2` drops to
0.30-0.40 between iter 64200 and 80000, hid stays at 0.05.

## Scientific Context

**Architecture**: spatio-temporal Instant-NGP for hidden-neuron voltage
prediction. 1-D temporal grid (24 levels, 4 features, base=16, scale=1.4)
+ 2-D spatial hex grid (6 levels, base=4, scale=1.5) over per-neuron
retinotopic position; per-level features concatenated and decoded by a
shared MLP (256-wide × 3 layers) into a per-neuron scalar voltage. The
GNN's learned latent `a_i` is concatenated into the decoder so cell-type
identity is available throughout. `ngp_factorized_rank=0` (no factorised
correction head).

**Three-phase training schedule** (see `flyvis_noise_005_hidden_010_blank50_consensus_ngp.yaml`):
- phase 1 `[0, 0.20·Niter)` — `v_h=0`, hidden-voltage loss OFF, anchor loss ON
- phase 2 `[0.20, 0.50·Niter)` — `v_h=0`, hidden-voltage loss ON (shapes NGP-hidden against the GNN forward without injection)
- ramp    `[0.50, 0.60·Niter)` — alpha goes 0 → 1 over `warmup_inject_nnr_ramp_iter_frac·Niter`
- phase 3 `[0.60·Niter, end]` — `v_h ← NGP(t, x_i, y_i, a_i)`, full coupling, optional LR damping V on {W, f_θ, g_φ} to absorb the injection-distribution shift

`Niter ≈ 107000` for the consensus_ngp_light variant
(`max_iterations_per_epoch=107000`); ~320000 for the full consensus_ngp.
The phase boundaries scale with Niter automatically through the
`*_iter_frac` fractions, so the analysis code above works for both
variants.

**The HP coupling that drives this exploration**:
- Phase 1 R²W is set by `lr_W`, `coeff_W_L1`, and the GNN-only knobs
  (`lr`, `lr_embedding`, `coeff_g_phi_norm`).
- Phase 2 stability is set by `coeff_hidden_voltage` (too high → NGP
  collapses to zero attractor; too low → NGP backbone underfits).
- Phase 3 recovery is set by the LR-damping V (`lr_damping_factor`),
  the ramp length (`warmup_inject_nnr_ramp_iter_frac`), and how much
  `lr_NNR_f` runs while the GNN is recovering.

A successful HP point has to balance all three. The job of this sweep
is to find that point and verify it is robust across seeds.

## Noise Model

```
v_i(t+1) = v_i(t) + dt · f(v_i(t), W, a_i, I_i(t)) + ε_i(t)
ε_i ~ N(0, 0.05)
```

`generate_data: false` — the dataset is fixed across all iterations.

## Metrics during training

The tqdm bar shows:
```
conn=R²W(R²W_visible)  Vr=R²Vr(out%)  τ=R²τ(out%)  nnr=hidden_pearson(anchor_pearson)
```
plus `n/a(anchor)` while alpha=0 (phase 1 + phase 2), since the hidden
Pearson against GT-injection rollout is meaningless when the GNN
forward sees v_h=0. After the ramp it switches to
`hidden_pearson(anchor_pearson)`.

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: form a specific, testable prediction about which HP
   move improves the composite trajectory score (be explicit about
   *which phase* the move targets).
2. **Design experiment**: change **EXACTLY ONE** parameter per slot
   (causality rule).
3. **Run training**: 4 parallel slots (slot 0 = parent control + 3 single-axis mutations).
4. **Read both log files** for all 4 slots.
5. **Analyze**: compute the composite per slot; identify which phase
   moved and whether it moved in the predicted direction.
6. **Update understanding**: log the trajectory, revise hypotheses.

**CRITICAL**: never make a verdict from a single endpoint number.
Always reason about which phase the change affected.

### CAUSALITY RULE (MANDATORY)

If you change more than one parameter per slot, you cannot attribute
the effect. Slot 0 = parent (unchanged control). Slots 1–3 = single-parameter mutations.

## FlyVis Model

- 13,741 neurons (1,736 photoreceptors, 12,005 non-retinal), 65 cell types, 434,112 edges
- Hidden mask: 10% of non-retinal neurons (~1,200 hidden, ~10,800 visible)
- Anchor: 3,600 visible neurons supervised directly against GT voltages
- 64,000 training frames, delta_t=20 ms, blank-prefix=50%
- Stimulus: DAVIS-2017
- Hex coordinates `(u_i, v_i)` baked into `state.pos` for every neuron

## HP search space

The agent may freely mutate any parameter below at any iteration.
Starting point = consensus_ngp_light yaml values shown in "Default".

### Phase-1 R²W lift (target: P1_peak / P1_end)

| Parameter      | Default  | Range                                | Phase affected | Notes |
| -------------- | -------- | ------------------------------------ | -------------- | ----- |
| `lr_W`         | 9.0e-4   | {3e-4, 6e-4, 9e-4, 1.5e-3, 2.5e-3}   | 1, 3           | larger = faster phase-1 R²W but worse phase-3 recovery (LR damping cushions this) |
| `lr`           | 1.8e-3   | {5e-4, 1e-3, 1.8e-3, 3e-3}           | 1              | g_φ / f_θ rate; smaller = under-fit messages on blank50 |
| `lr_embedding` | 2.325e-3 | {1e-3, 2.325e-3, 4e-3}                | 1, 3           | controls how fast `a_i` clusters by cell type (load-bearing for hid_final) |
| `coeff_W_L1`   | 1.5e-4   | {0, 5e-5, 1.5e-4, 5e-4, 1.5e-3}       | 1, 3           | the only important regulariser for R²W; too high crushes phase 1 |
| `coeff_g_phi_norm` | 0.9  | {0.3, 0.9, 5.0}                       | 1              | unified-blank50-winner=0.9 |

### Phase-2 stability (target: P2_end − P2_min)

| Parameter             | Default | Range                | Phase affected | Notes |
| --------------------- | ------- | -------------------- | -------------- | ----- |
| `coeff_hidden_voltage`| 300     | {0, 100, 300, 1000}   | 2              | ZERO = no shaping; default 300; >=3000 caused the legacy zero-attractor collapse — investigate the 0..1000 band only |
| `coeff_anchor_voltage`| 3000    | {1000, 3000, 10000}   | 1, 2           | direct anchor supervision strength |
| `n_anchor`            | 3600    | {1800, 3600, 9000}    | 1, 2           | denser anchor supervision (n_anchor=9000 was no help over 3600 in the prior manual A/B but worth re-checking under the new schedule) |

### Phase-3 recovery (target: P3_drop, P3_final)

| Parameter                          | Default | Range                       | Phase affected | Notes |
| ---------------------------------- | ------- | --------------------------- | -------------- | ----- |
| `warmup_inject_nnr_ramp_iter_frac` | 0.10    | {0.033, 0.05, 0.10, 0.20}    | 3              | ramp length; longer = gentler injection-distribution shift but eats phase-3 wall time |
| `lr_damping_factor`                | 100.0   | {1.0, 10.0, 100.0, 1000.0}   | 3              | depth of the V-shaped LR damping on {W, f_θ, g_φ} at the injection switch; 1.0 = damping off |
| `lr_NNR_f`                         | 5.0e-5  | {1e-5, 5e-5, 1e-4, 5e-4}     | 3              | NGP backbone LR after the warmup; controls how fast NGP catches up to the new (v_h ≠ 0) regime |
| `alpha_inject_target`              | 1.0     | {0.0, 0.5, 1.0}              | 3              | post-ramp alpha cap; 0.0 = passive monitor (NGP never injected); 0.5 = half-strength injection — useful as a diagnostic if P3_drop is large |

### NGP capacity (search after lr / regularisation are settled)

| Parameter                       | Default | Range                            | Phase affected | Notes |
| ------------------------------- | ------- | -------------------------------- | -------------- | ----- |
| `ngp_hidden_n_levels`           | 24      | {16, 24, 32}                     | 2, 3           | finer temporal resolution |
| `ngp_hidden_n_features_per_level`| 4      | {2, 4, 8}                        | 2, 3           | feature width per level |
| `ngp_hidden_mlp_width`          | 256     | {128, 256, 384, 512}             | 2, 3           | decoder MLP width |
| `ngp_hidden_mlp_layers`         | 3       | {2, 3, 4}                        | 2, 3           | decoder MLP depth |
| `ngp_hidden_spatial_n_levels`   | 6       | {3, 6, 8}                        | 2, 3           | 2-D hex grid depth |

### Frozen / off-limits

Do NOT modify these (would change the experimental setup):
- `n_neurons`, `n_input_neurons`, `n_edges`, `n_frames`, `delta_t` — fixed dataset
- `noise_model_level` — fixed at 0.05
- `blank_prefix_fraction`, `skip_short_videos` — fixed
- `hidden_neuron_fraction` — fixed at 0.10
- `inr_type_hidden`, `ngp_hidden_spatial`, `ngp_factorized_rank`, `ngp_factorized_from_a` — fixed
- `embedding_dim` — fixed at 2
- `warmup_hidden_loss_iter_frac`, `warmup_inject_nnr_iter_frac` — fixed (the 3-phase boundaries are *the* experimental setup; the sweep is HP-within-this-schedule, not schedule-redesign)

## Training regime (fixed)

- `n_epochs: 1`
- `data_augmentation_loop: 100` (full-DAL config; LIGHT variant uses
  `max_iterations_per_epoch=107000` to cap total Niter at ~1/3)
- `batch_size: 4`
- `n_runs: 1`
- Cluster: `gpu_h100`
- Hard runtime limit: ~60 min per slot for the LIGHT variant; ~3 h for the full

Decisions should be based on the **trajectory** (composite score),
NOT terminal performance.

## Iteration Workflow

### Step 1: Read working memory + user input

### Step 2: Read trajectory logs for ALL FOUR slots

For each slot `s ∈ {00, 01, 02, 03}`:

1. Read `<slot_log_dir_s>/tmp_training/metrics.log` (full file, not just tail).
2. Read `<slot_log_dir_s>/tmp_training/nnr_pearson.log`.
3. Run the parse() function above to extract per-phase statistics.
4. Verify all four slots reached at least `iter ≥ 0.50·Niter` (i.e. past
   the injection switch). If a slot crashed earlier, mark its phase-3
   stats as `nan` and flag it as a partial result.

### Step 3: Score, classify, table-out

For each slot compute `composite`. Across slots compute
`composite_mean`, `composite_min`, `composite_CV%`.

Classify the iteration:
- **Stable-Robust**: `composite_min > 0.55` AND `composite_CV < 10%`
- **Stable**: `composite_min > 0.40`
- **Unstable**: `composite_min in [0, 0.40]`
- **Catastrophic**: any slot has `P3_drop > 0.30` OR `P3_final < 0` OR
  the parse failed (training crashed).

### Step 4: Write log entry — REQUIRED FORMAT

```
## Iter N: [stable_robust/stable/unstable/catastrophic]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis, naming the phase it targets]"
Mutation: [param]: [old] -> [new]   (slots 1..3, single axis each)

           P1_peak  P1_end  P2_min  P2_end  P3_drop  P3_final  hid   anc   composite
Slot 0:    0.XX     0.XX    0.XX    0.XX    0.XX     0.XX      0.XX  0.XX  0.XX
Slot 1:    0.XX     ...
Slot 2:    ...
Slot 3:    ...
Aggregate: composite_mean=X, composite_min=Y, CV=Z%

Quoted from logs:
- slot 0 metrics.log row at iter ≈ 0.50·Niter: "<paste raw csv row>"
- slot 0 nnr_pearson.log final row:               "<paste raw csv row>"
- (one quoted row per slot is enough as proof of read)

Verdict: [supported / falsified / inconclusive]
        Reasoning: which phase moved, by how much, in which direction.
Next: parent=P (best slot by composite_min)
```

A log entry without the per-phase table or the quoted rows is malformed
and must be rewritten before the next iteration.

### Step 5: Acknowledge user input

### Step 6: Formulate next hypothesis + edit 4 config files

Always name the *phase* the next mutation targets. Do not propose a
mutation without a phase.

## Block Boundaries

At every block boundary (every 8 iters):

1. Update "Paper Summary" with the current best composite and which HPs
   are "settled".
2. Summarize block findings — which mutations helped *which phase*.
3. Update Established Principles (require 3+ consistent iterations).
4. Update Falsified Hypotheses.
5. Carry forward best-composite-min slot as the parent for next block.

## Block structure (suggested focus)

| Block | Iters  | Phase being optimized                                              |
| ----- | ------ | ------------------------------------------------------------------ |
| 1     | 1–8    | Phase 1: lr_W, lr, coeff_W_L1 — push P1_peak / P1_end             |
| 2     | 9–16   | Phase 2: coeff_hidden_voltage, coeff_anchor_voltage, n_anchor      |
| 3     | 17–24  | Phase 3: lr_damping_factor, warmup_inject_nnr_ramp_iter_frac       |
| 4     | 25–32  | Phase 3 NGP recovery: lr_NNR_f, alpha_inject_target diagnostic     |
| 5     | 33–48  | Free combination: best phase-1 + best phase-2 + best phase-3 HPs   |
| 6     | 49–64  | NGP capacity sweep on the winning combo                            |
| 7     | 65–80  | Free exploration #2                                                |
| 8     | 81–96  | 4-seed robustness validation of best composite                     |

The "suggested focus" is a hint, not a rule. If a strong phase-1 signal
emerges in block 1, stay on it.

## Start Call

**Dataset is pre-generated. Do NOT regenerate.** The parent yaml must
contain a `claude:` block with `generate_data: false` so the LLM
pipeline reuses the dataset already on disk at
`<DATA_ROOT>/graphs_data/fly/flyvis_noise_005_hidden_010_ngp_blank50/`.
If the block is missing from
`<workspace>/config/fly/flyvis_noise_005_hidden_010_blank50_consensus_ngp.yaml`,
add the following before launching:

```yaml
claude:
  n_epochs: 1
  data_augmentation_loop: 100   # match the training block
  training_time_target_min: 180
  node_name: a100               # A100 queue (gpu_a100); override per-run with --node h100
  n_parallel: 4                 # 4-slot single-axis sweep per iteration
  n_iter_block: 8
  generate_data: false          # REUSE the existing dataset
```

When prompt says `PARALLEL START`:

**HARD RULE — config sources:** The ONLY authoritative parent yaml is
`<workspace>/config/fly/flyvis_noise_005_hidden_010_blank50_consensus_ngp.yaml`.
Do NOT read other yamls in the slot config directory for HP inspiration.

- **Slot 0 = baseline** — exact copy of the parent yaml. Parent HPs:
  - `lr_W=9.0e-4, lr=1.8e-3, lr_embedding=2.325e-3, lr_NNR_f=5.0e-5`
  - `coeff_W_L1=1.5e-4, coeff_W_L2=1.5e-6, coeff_g_phi_norm=0.9`
  - `coeff_g_phi_diff=750, coeff_g_phi_weight_L1=0.28, coeff_f_theta_weight_L1=0.05`
  - `ngp_hidden_n_levels=24, n_features_per_level=4, mlp_width=256, mlp_layers=3`
  - `ngp_hidden_spatial=true, ngp_factorized_rank=0, ngp_factorized_from_a=false`
  - `n_anchor=3600, coeff_anchor_voltage=3000, coeff_hidden_voltage=300`
  - 3-phase schedule fractions:
    `warmup_hidden_loss_iter_frac=0.20, warmup_inject_nnr_iter_frac=0.50, warmup_inject_nnr_ramp_iter_frac=0.10`
  - `lr_damping_factor=100.0`
  - `n_epochs=1, data_augmentation_loop=100, batch_size=4, embedding_dim=2`

- **Block 1 = phase-1 probes**: slot 0 = control (unchanged parent),
  slots 1–3 = single-axis mutations from the "Phase-1 R²W lift" table
  (e.g. lr_W=1.5e-3, coeff_W_L1=5e-4, lr_embedding=4e-3).

- **Launch (FRESH START — dataset is already on disk, do NOT regenerate)**:
  ```
  rm /groups/saalfeld/home/allierc/GraphData/config/fly/flyvis_noise_005_hidden_010_blank50_consensus_ngp_Claude_*.yaml
  python GNN_LLM.py -o generate_train_test_plot_Claude \
    flyvis_noise_005_hidden_010_blank50_consensus_ngp \
    iterations=96 --cluster
  ```
  (Drop `--resume` to clear `_analysis.md`, `_memory.md`, `_reasoning.log`.)
  The pipeline checks `claude.generate_data` in the parent yaml — when
  `false` (as required above), it skips `should_generate_data → generate_data_locally`
  for all batches and reuses the existing dataset under
  `<DATA_ROOT>/graphs_data/fly/flyvis_noise_005_hidden_010_ngp_blank50/`.

## Final Summary

At exploration completion (after Block 8), append to
`/home/node/.claude/projects/-workspace--devcontainer/memory/exploration_results.md`:

### flyvis_noise_005_hidden_010_blank50_consensus_ngp — Key Discoveries (YYYY-MM-DD)

1. **Best composite**: composite_min = X.XXX, config = [key params], seed-validated CV=X.X%
2. **Phase-1 ceiling**: max P1_end achieved across all probes; which HP set it.
3. **Phase-2 stability cost**: at the best P1_end, what was the worst P2_min observed?
4. **Phase-3 drop**: did any HP combination achieve `P3_drop < 0.05`? If yes, what?
5. **Joint NGP final**: best `min(P3_final, hid_final, anc_final)` and the config that produced it.
6. **HP impact ranking by phase**: which HP move had the largest single-parameter impact on each
   phase's score (P1_end, P2_end−P2_min, P3_final−0.5·P3_drop, joint).

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_hidden_010_blank50_consensus_ngp

## Paper Summary (update at every block boundary)

**Trajectory-based GNN+spatial-NGP optimization** (2 sentences):
Sentence 1: Best composite_min so far, config, seed CV — and *which phase
            limits it* (phase 1 ceiling, phase 2 collapse, phase 3 drop, or joint
            NGP underfit).
Sentence 2: Which HPs are "settled" (closed) and which still have open
            questions, organized by phase.

**LLM-driven exploration** (2 sentences):
Sentence 1: What the sweep revealed about *cross-phase coupling* (does
            pushing P1_end hurt P3_drop? Does softening
            coeff_hidden_voltage help phase 2 at the cost of phase 3?).
Sentence 2: Main causal principle — what HP move(s) reliably hold R²W
            across all three phases.

## Knowledge Base

### Per-phase HP impact table

| Iter | Mutation | ΔP1_end | ΔP2_min | ΔP3_drop | ΔP3_final | ΔComposite | Verdict |
| ---- | -------- | ------- | ------- | -------- | --------- | ---------- | ------- |

### Established Principles  (≥3 consistent iterations)
### Falsified Hypotheses
### Open Questions

---

## Previous Block Summaries

(keep last 4 blocks, organized by phase)

---

## Current Block

### Block Info
### Current Hypothesis (must name a phase)
### Iterations This Block
### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of the memory file.**
```
