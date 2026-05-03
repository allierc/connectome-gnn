# FlyVis spatial-NGP HP sweep on cv04 — break the (R²W, nnr_hidden, nnr_anchor) > 0.4 triple ceiling

## Goal

Optimize GNN + spatial-NGP hyperparameters on `flyvis_noise_005_hidden_010_ngp_blank50_unified_spatial`
(σ=0.05, 10% hidden, regenerated dataset with the (u, v) hex-coordinate fix) to **break the triple
ceiling**:

- `R²W > 0.4`     (connectivity recovery on all neurons)
- `nnr_hidden > 0.4`  (NGP hidden-neuron Pearson r)
- `nnr_anchor > 0.4`  (NGP anchor-neuron Pearson r)

ALL THREE must clear 0.4 to count as "broken ceiling". The current working point (cv04 baseline,
single-seed at iter ~5K) sits at:

- R²W ≈ -7 to -10 (deeply negative; W barely trained)
- nnr_anchor ≈ 0.518 (clears 0.4 ✓)
- nnr_hidden ≈ 0.05 (far below 0.4)

So the bottleneck is **R²W and nnr_hidden** simultaneously. They couple: hidden Pearson improves
when `a_hidden` clusters by cell type via `g_phi(v_NGP, a_hidden)`-back-propagation, which itself
depends on a stable W. Conversely W learns better when the NGP's hidden voltages are not
contaminating the message-passing graph with noise. The HP sweep needs to find a regime where
both can converge.

## Scientific Context

**Architecture**: spatial+temporal Instant-NGP for hidden-neuron voltage prediction, with the GNN's
learned latent `a_i` concatenated into the per-neuron query decoder (no factorised head;
`ngp_factorized_rank=0`).

**Per-neuron query**: forward(t, pos[i], a[i]) -> scalar v̂_i(t). For batch B and 4800 hidden+anchor
neurons, that's B × 4800 MLP forward calls per loss step (~80× the time-only NGP cost).

**Critical interplay observed in the manual sweep**:
- `lr_W=5e-5`, `lr_NNR_f=5e-4`: anchor reaches 0.5 in 5K iters but R²W stays at -7 to -10
- `lr_W=1e-4`, `lr_NNR_f=5e-5`: anchor stable at 0.5; R²W slowly climbs (still negative at iter 2K)
- `lr=1e-4`, `lr_W=5e-4`: untested (g_phi/f_theta frozen-ish, W aggressive)

The HP space the agent must search:

## Noise Model

```
v_i(t+1) = v_i(t) + dt * f(v_i(t), W, a_i, I_i(t)) + epsilon_i(t)
epsilon_i ~ N(0, 0.05)
```

`generate_data: false` — the cv04 dataset is fixed across all iterations; sim/train seeds vary
per slot via the standard pipeline-controlled formula.

## Metrics

During training, the tqdm bar shows:
```
conn=R²W(R²W_visible)  Vr=R²Vr(out%)  τ=R²τ(out%)  nnr=hidden_pearson(anchor_pearson)
```

The metrics.log columns (read every 200 iters by `_quick_ngp_pearson` and every Niter//10 by
`plot_training_flyvis`) are:
```
iter, conn_R2, vrest_R2, tau_R2, hidden_pearson, anchor_pearson,
vrest_R2_clean, n_out_vrest, n_total_vrest, tau_R2_clean, n_out_tau, n_total_tau
```

Primary objective for this exploration:

```
score = min(conn_R2, hidden_pearson, anchor_pearson)
```

This is a **min-of-three** because the user wants to break ALL three ceilings simultaneously. A
config that gets conn_R2=0.7 but hidden_pearson=0.0 has score=0 → no progress.

Secondary diagnostics:
- `cluster_accuracy` (cell-type clustering from a_i via UMAP) — only available after data_plot.
- `rollout_pearson_r` — only available from data_test (we are not running test in this loop).

**Robustness**: 4 slots per iteration → look at min/mean across slots. Catastrophic = any slot has
score < -1 (i.e. one of the three metrics is deeply negative).

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: form a specific, testable prediction about which HP move improves the
   `min(conn_R2, hidden_pearson, anchor_pearson)` score
2. **Design experiment**: change **EXACTLY ONE** parameter per slot (causality rule)
3. **Run training**: 4 parallel slots (1 control = current parent + 3 mutations)
4. **Analyze results**: compute the min-of-three score per slot; look for the mutation that
   improves it without regressing on the others
5. **Update understanding**: log the result, revise hypotheses

**CRITICAL**: you can only hypothesize. Only training results validate or falsify.

### CAUSALITY RULE (MANDATORY)

If you change more than one parameter per slot, you cannot attribute the effect. Slot 0 = parent
(unchanged control). Slots 1–3 = single-parameter mutations.

## FlyVis Model

- 13,741 neurons (1,736 photoreceptors, 12,005 non-retinal), 65 cell types, 434,112 edges
- Hidden mask: 10% of non-retinal neurons (~1,200 hidden, ~10,800 visible)
- Anchor: subset of visible neurons supervised directly against GT voltages
- 64,000 training frames, delta_t=20 ms, blank-prefix=50%
- Stimulus: DAVIS-2017
- Hex coordinates `(u_i, v_i)` baked into `state.pos` for every neuron (the spatial branch
  queries `pos[hidden_ids]` to share parameters across nearby columns)

## GNN Architecture (current cv04 baseline)

```
g_phi(v_j, a_j) -> message_ij           (edge MLP, hidden_dim=80, n_layers=3)
sum_j W_ij * g_phi(v_j) -> agg_i        (weighted aggregation; W is what we want to recover)
f_theta(v_i, a_i, agg_i, I_i) -> dv/dt  (node update MLP, hidden_dim=80, n_layers=3)
```

- `embedding_dim=2` (per-neuron `a_i` is 2-D)
- `g_phi_positive=true`

### Hidden-neuron InstantNGP (current cv04 baseline)

```
forward_hidden(t, pos[id], a[id]) -> v_id(t)
```

- `inr_type_hidden: ngp_t`
- `ngp_hidden_spatial: true` — 6-level 2-D hex grid (base=4, scale=1.5)
- `a_dim = embedding_dim = 2` — the GNN's `a[id]` is concatenated into the decoder
- `ngp_factorized_rank: 0` — no factorised head (a_i enters via concat, not as additive correction)
- temporal grid: 24 levels × 4 features × base 16 × scale 1.4
- decoder MLP: 256 wide × 3 layers, output dim 1
- per-neuron query: B × (n_hidden + n_anchor) MLP forwards per loss step

## HP search space

The agent may freely mutate any parameter below at any iteration (NO hard block boundaries — the
sweep is fluid). However the **starting point** for the exploration is the cv04 baseline values
shown in the "Default" column.

### Learning rates (the most-impactful axis based on the manual sweep)

| Parameter      | Default  | Range to explore                         | Notes |
| -------------- | -------- | ---------------------------------------- | ----- |
| `lr_W`         | 1.0e-4   | {1e-5, 5e-5, 1e-4, 5e-4, 9e-4}           | paper-baseline NGP-winner=1e-4; unified=9e-4. Larger → more W learning, more noise into NGP gradient |
| `lr` (g_φ,f_θ) | 1.0e-3   | {1e-4, 5e-4, 1e-3, 1.8e-3}               | paper time-only=1e-3; unified=1.8e-3. Smaller → freezes message function, gives W headroom |
| `lr_NNR_f`     | 5.0e-4   | {1e-5, 5e-5, 1e-4, 5e-4, 1e-3}           | smaller after anchor>0.5 stabilises (consolidate); larger speeds NGP catchup |
| `lr_embedding` | 1.0e-3   | {5e-4, 1e-3, 2e-3}                       | controls how fast `a_i` clusters by cell type (load-bearing for nnr_hidden) |

### Regularisation

| Parameter        | Default  | Range to explore                           | Notes |
| ---------------- | -------- | ------------------------------------------ | ----- |
| `coeff_W_L1`     | 5.0e-5   | {0, 5e-5, 1.5e-4, 5e-4}                    | paper appendix calls this "the only important regulariser" for R²W |
| `coeff_W_L2`     | 1.5e-6   | {0, 1.5e-6, 1.5e-5}                        | usually inert |
| `coeff_g_phi_norm` | 5.0    | {0.5, 0.9, 5.0}                            | unified-blank50-winner=0.9; current cv04=5.0 (test if lower helps W) |

### NGP capacity (search after lrs / regularisation are tuned)

| Parameter                       | Default | Range                            | Notes |
| ------------------------------- | ------- | -------------------------------- | ----- |
| `ngp_hidden_n_levels`           | 24      | {16, 24, 32}                     | finer temporal resolution; 16 was too low-pass in earlier runs |
| `ngp_hidden_n_features_per_level` | 4    | {2, 4, 8}                        | feature width per level |
| `ngp_hidden_mlp_width`          | 256     | {128, 256, 384, 512}             | decoder MLP width; 512 was tested working at iter ~800 |
| `ngp_hidden_mlp_layers`         | 3       | {2, 3, 4}                        | decoder MLP depth |
| `ngp_hidden_spatial_n_levels`   | 6       | {3, 6, 8}                        | 2-D hex grid depth |
| `n_anchor`                      | 3600    | {1800, 3600, 9000}               | denser anchor → more direct supervision but more compute (n_anchor=9000 didn't help over 3600 in the manual A/B) |

### NGP knobs that affect schedule (rare to touch)

| Parameter                   | Default | Range            | Notes |
| --------------------------- | ------- | ---------------- | ----- |
| `coeff_hidden_voltage`      | 3000    | {1000, 3000, 10000} | self-consistency loss strength |
| `coeff_anchor_voltage`      | 3000    | {1000, 3000, 10000} | direct anchor supervision strength |
| `alternate_lr_ratio`        | 0.4     | {0.05, 0.2, 0.4, 1.0} | ratio applied to GNN lrs at epoch 1; 1.0 = disabled |

### Frozen / off-limits

Do NOT modify these (they would change the experimental setup):
- `n_neurons`, `n_input_neurons`, `n_edges`, `n_frames`, `delta_t` — fixed dataset
- `noise_model_level` — fixed at 0.05
- `blank_prefix_fraction`, `skip_short_videos` — fixed
- `hidden_neuron_fraction` — fixed at 0.10
- `inr_type_hidden`, `ngp_hidden_spatial`, `ngp_factorized_rank`, `ngp_factorized_from_a` — fixed
  (this exploration is FOR the spatial+a_i-concat design, not a re-evaluation of it)
- `embedding_dim` — fixed at 2 (changing it would invalidate `a_dim` in the wrapper)

## Training regime (fixed)

- `n_epochs: 1`
- `data_augmentation_loop: 25` (DAL=25 → ~20,000 iterations per epoch on H100, ~35 min wall time)
- `batch_size: 16`
- `n_runs: 1`
- Cluster: `gpu_h100`
- Hard runtime limit: 60 min per slot (cluster default)

The shorter epoch (DAL=25 vs the canonical 50) means we burn through HP iterations faster — at the
cost of less convergence per iteration. Decisions should be based on **trajectory** (where the
score is heading at iter ~10K-16K) rather than terminal performance, since most experiments will
not converge in one epoch.

## Block structure (fluid, no hard topical boundaries)

Each "block" is just an administrative checkpoint of 8 iterations × 4 slots = **32 experiments per
block**. Within a block, the agent freely picks any HP from the search space above to mutate at
each iteration.

| Block | Iters  | Suggested focus (NOT a hard restriction)                 |
| ----- | ------ | -------------------------------------------------------- |
| 1     | 1–4    | Baseline robustness — 4 slots = control config (4 seeds) |
| 2     | 5–12   | Lr sweep (lr_W, lr, lr_NNR_f, lr_embedding) priority      |
| 3     | 13–20  | Regularisation sweep (coeff_W_L1 priority)               |
| 4     | 21–28  | NGP capacity (n_levels, mlp_width, mlp_layers)           |
| 5     | 29–36  | Free exploration combining best-of-block 2/3/4           |
| 6     | 37–44  | Free exploration #2                                      |
| 7     | 45–52  | Free exploration #3                                      |
| 8     | 53–56  | 4-seed robustness validation of best candidate           |

**The "suggested focus" is a hint, not a rule.** If the agent finds a strong signal in lr_W during
block 1, it can stay on lr_W in block 2. If a regularisation finding from block 3 is overturned in
block 5, that's expected. The blocks are just N=32-experiment checkpoints to update working memory.

## Iteration Workflow

### Step 1: Read working memory + user input

### Step 2: Analyze results (4 slots)

For each slot, read from metrics.log:
- last `iter`, `conn_R2`, `hidden_pearson`, `anchor_pearson`
- compute `score = min(conn_R2, hidden_pearson, anchor_pearson)`
- classify: **Stable-Robust** (score > 0.4 across slots, CV < 10%); **Stable** (score > 0.2);
  **Unstable** (score in [0, 0.2]); **Catastrophic** (any slot score < -1)

### Step 3: Write log entry + update memory

```
## Iter N: [stable_robust/stable/unstable/catastrophic]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, lr_emb=Z, lr_NNR_f=W, coeff_W_L1=A,
        ngp_hidden_n_levels=B, mlp_width=C, mlp_layers=D, n_anchor=E
Slot 0: iter=I, conn_R2=X, hid=Y, anc=Z, score=min(X,Y,Z)
Slot 1: iter=I, conn_R2=X, hid=Y, anc=Z, score=min(X,Y,Z)
Slot 2: iter=I, conn_R2=X, hid=Y, anc=Z, score=min(X,Y,Z)
Slot 3: iter=I, conn_R2=X, hid=Y, anc=Z, score=min(X,Y,Z)
Seed stats: mean_score=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive]
Next: parent=P (best slot)
```

### Step 4: Acknowledge user input

### Step 5: Formulate next hypothesis + edit 4 config files

## Block Boundaries

At every block boundary (every 8 iters):

1. Update "Paper Summary" in memory with the current best score and which HPs are "settled"
2. Summarise block findings — what mutations helped/hurt the min-of-three score
3. Update Established Principles (require 3+ iterations, all 4 seeds consistent)
4. Update Falsified Hypotheses
5. Carry forward best config as parent for next block

## Start Call

When prompt says `PARALLEL START`:

- **Slot 0 = baseline** (current cv04 config; must be present at
  `<workspace>/config/fly/flyvis_noise_005_hidden_010_ngp_blank50_unified_spatial.yaml`):
  - `lr_W=1.0e-4, lr=1.0e-3, lr_embedding=1.0e-3, lr_NNR_f=5.0e-4`
  - `coeff_W_L1=5.0e-5, coeff_W_L2=1.5e-6, coeff_g_phi_norm=5.0`
  - `ngp_hidden_n_levels=24, n_features_per_level=4, mlp_width=256, mlp_layers=3`
  - `ngp_hidden_spatial=true, ngp_factorized_rank=0, ngp_factorized_from_a=false`
  - `n_anchor=3600, coeff_anchor_voltage=3000, coeff_hidden_voltage=3000`
  - `alternate_training=true, alternate_lr_ratio=0.4`
  - `n_epochs=1, data_augmentation_loop=20, batch_size=16`
  - `embedding_dim=2`

- **Block 1 = ROBUSTNESS mode**: slots 1–3 also use the same baseline config (different seeds)
  → quantifies seed-dependent variance of the **starting** score

- **Hypothesis**: "The cv04 baseline (manual single-seed) reached
  `nnr_anchor≈0.5`, `nnr_hidden≈0.05`, `R²W≈-7` at iter ~5K (DAL=50). With DAL=25 (single epoch ≈
  16K iters), 4-seed robustness should give mean_score in [-2, 0] and CV high. The bottleneck is
  R²W and nnr_hidden — those are what the rest of the sweep must lift above 0.4."

- **Launch**:
  ```
  python GNN_LLM.py -o generate_train_test_plot_Claude \
    flyvis_noise_005_hidden_010_ngp_blank50_unified_spatial \
    iterations=128 --cluster --resume
  ```

## Final Summary

At exploration completion (after Block 8), append to
`/home/node/.claude/projects/-workspace--devcontainer/memory/exploration_results.md`:

### flyvis_noise_005_hidden_010_ngp_blank50_unified_spatial — Key Discoveries (YYYY-MM-DD)

1. **Best score**: min(R²W, hid, anc) = X.XXX, config = [key params], seed-validated CV=X.X%
2. **Did the triple ceiling break?**: did at least one slot reach R²W≥0.4 AND hid≥0.4 AND anc≥0.4?
3. **HP impact ranking**: which HP had the largest single-parameter impact on the score
4. **R²W vs nnr trade-offs**: did improving R²W (e.g. via larger lr_W) hurt nnr_hidden / nnr_anchor?
5. **NGP size sweet spot**: at DAL=25, what's the mlp_width / n_levels combination that maximises
   score per minute of wall time?
6. **anchor_pearson saturation**: did anchor settle at the time-only-NGP terminal of ~0.74,
   or higher / lower with the spatial+a_i design?

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_hidden_010_ngp_blank50_unified_spatial

## Paper Summary (update at every block boundary)

**GNN+spatial-NGP optimization** (2 sentences):
Sentence 1: Best (min-of-three) score achieved so far, config, seed CV.
Sentence 2: Which HPs are "settled" (closed) and which still have open questions.

**LLM-driven exploration** (2 sentences):
Sentence 1: What the sweep revealed about the HP coupling structure (does R²W trade off with
nnr_hidden? Is the spatial+a_i design hitting a fundamental ceiling?).
Sentence 2: Main causal principle — what HP move(s) reliably break the triple ceiling.

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 | hid | anc | score | Verdict | Hypothesis |
| ---- | -------------- | ------- | --- | --- | ----- | ------- | ---------- |

### Established Principles
### Falsified Hypotheses
### Open Questions

---

## Previous Block Summaries

(keep last 4 blocks)

---

## Current Block

### Block Info
### Current Hypothesis
### Iterations This Block
### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
