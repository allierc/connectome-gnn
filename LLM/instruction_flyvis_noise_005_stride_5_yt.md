# FlyVis GNN Training Exploration — flyvis_noise_005_stride_5_yt

## Motivation and Key Difference from stride_5

The previous exploration (`flyvis_noise_005_stride_5`, 36 iterations) was fundamentally **training-diversity limited**. With 64,000 frames and `time_step=5`, the `k = k - k % time_step` alignment in the trainer restricts valid starting frames to multiples of 5 — leaving only **12,800 unique starting positions** instead of 64,000. This means stride-5 BPTT had 5× less training-pair diversity than stride-1, even though both nominally had "64K frames".

**Fix applied here**: YouTube-VOS dataset (`datavis_roots` pointing to YouTube-VOS) with `n_frames=0` (single pass through all sequences) generates a much larger corpus. **`n_frames` in the config is fixed to 320,000** (= 64,000 × 5), giving **64,000 unique stride-5 starting positions** — exactly matching the training diversity of the stride-1 champion.

**DAL is reduced from 35 → 8** to keep wall time within 120 min:
- Old: `Niter = int(64000 × 35 // 4 × 0.2) = 112,000` iterations (~90 min)
- New: `Niter = int(320000 × 8 // 4 × 0.2) = 128,000` iterations (~105 min, within budget ✓)

**Wall time unchanged. Gradient diversity: 5× higher.**

## Scientific Goal

Test whether **recurrent (BPTT) training with temporal stride=5**, given sufficient training-pair diversity, improves connectivity recovery above the stride-1 champion.

> **Core question**: Was the stride-5 gap (best robust mean = 0.563 vs stride-1 = 0.980) due to training diversity starvation, or is spectral radius ≈ 1.72 a fundamental BPTT barrier?

**Primary metric**: `connectivity_R2` — must exceed stride-5 robust mean (0.563) and ideally exceed stride-1 baseline (0.980).
**Hard floor**: connectivity_R2 > 0.70 (all 4 seeds) — this already beats the best robust result from stride_5.
**Secondary metrics**: `tau_R2`, `V_rest_R2`, `rollout_pearson`.

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle.

1. **Hypothesize**: Based on prior evidence, form a specific, testable prediction
2. **Design experiment**: Mutate **exactly ONE parameter** at a time
3. **Run**: 4 seeds — cannot predict the outcome
4. **Analyze**: Use both metrics AND cross-seed variance
5. **Update**: Revise hypotheses. A falsified hypothesis is equally valuable.

**CRITICAL**: Only training results validate or falsify. Never assume.

**Evidence hierarchy:**

| Level | Criterion | Action |
|-------|-----------|--------|
| **Established** | Consistent across 3+ iterations AND 4/4 seeds | Add to Principles |
| **Tentative** | 1-2 times or inconsistent across seeds | Add to Open Questions |
| **Contradicted** | Conflicting evidence | Note in Open Questions |

## CRITICAL: Data is PRE-GENERATED at startup — NEVER RE-GENERATE

At startup, data is generated **once** for all 4 slots. Datasets are **reused across all iterations**. Both `simulation.seed` and `training.seed` are **forced by the pipeline** — DO NOT modify them.

**YouTube-VOS data generation takes ~1 hour per slot.** Re-generating would cost 4+ hours and is never justified. **Do NOT set `test_robustness_seed: true` under any circumstances in this exploration.** Seed robustness is instead achieved through the 4 fixed datasets (`_00` through `_03`) that were generated once at startup.

Seed formula (set automatically by GNN_LLM.py):
- `simulation.seed = 1000 + slot` (data generation — fixed at startup, **never change**)
- `training.seed = iteration * 1000 + slot + 500` (weight init & training randomness)

**Dataset**: `fly/flyvis_noise_005_stride_5_yt_{00..03}` — YouTube-VOS, 320,000 frames, noise_model_level=0.05.

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:

```
tau_i * dv_i(t)/dt = -v_i(t) + V_i^rest + sum_j W_ij * g_phi(v_j, a_j)^2 + I_i(t)
```

- 13,741 neurons, 65 cell types, 434,112 edges
- 1,736 input neurons (photoreceptors)
- YouTube-VOS visual input, **noise_model_level=0.05**
- **320,000 frames** (= 5 × 64,000), delta_t=0.02
- **Spectral radius ≈ 1.72** — relevant for gradient explosion risk in BPTT

## Recurrent Training Mechanism

With `recurrent_training: true` and `time_step: 5`:

1. Sample random frame `t` (aligned to nearest multiple of 5)
2. From **true** `v(t)`, unroll 5 Euler steps using model predictions:
   ```
   v̂(t+1) = v(t)   + dt · GNN(v(t))
   ...
   v̂(t+5) = v̂(t+4) + dt · GNN(v̂(t+4))
   ```
3. Loss = `||v̂(t+5) - v(t+5)||² / (dt × 5)`
4. Backpropagate through all 5 steps (BPTT)

## Established Knowledge from stride_5 Exploration (DO NOT RE-TEST)

These principles were confirmed across 3+ iterations and 4/4 seeds in the prior exploration. **Accept as constraints**, do not re-test:

1. **lr_W=0.0009 is locally optimal**: Monotonic decrease below this; above 0.002 causes gradient explosion (raw_W_R2 → 0.1).
2. **lr=0.0018 is required**: lr=0.0004 gives worst single result (0.511); lr=0.004 destabilizes W.
3. **noise_recurrent_level > 0 is HARMFUL**: All 4 tested levels (0.01, 0.03, 0.1) monotonically degrade conn_R2. Spectral radius > 1 amplifies noise in BPTT gradients. **Do not test further.**
4. **lr_emb=0.003 is the best known single-seed point**: ATB=0.6596 (stride_5 iter 31). 4-seed confirmation was pending at the end of stride_5 exploration — this is the first thing to confirm here.
5. **Single-seed comparisons are unreliable** (CV≈11% across seeds): Only 4-seed robustness tests are scientifically interpretable.
6. **Gradient instability thresholds**: lr_W > 0.002 and lr > 0.003 cause raw_W_R2 collapse. Stay below.

## Starting Configuration (Block 1 Baseline)

All parameters inherit from `flyvis_noise_005_stride_5`, except:

| Parameter | Old stride_5 | New _yt | Reason |
|-----------|-------------|---------|--------|
| `n_frames` | 64,000 | **320,000** | 5× diversity parity with stride-1 |
| `data_augmentation_loop` | 35 | **8** | Niter=128K, ~105 min (within 120 min budget) |
| `lr_embedding` | 0.002325 | **0.003** | Best known from stride_5 iter 31 |

Full baseline:
```yaml
training:
  recurrent_training: true
  time_step: 5
  batch_size: 4
  data_augmentation_loop: 8
  lr_W: 0.0009
  lr: 0.0018
  lr_embedding: 0.003
  noise_recurrent_level: 0.0
  grad_clip_W: 0.0
  n_epochs: 1
  regul_annealing_rate: 0.0
```

## Explorable Parameters

### Core (do not change)
- `time_step: 5` — defines this experiment
- `recurrent_training: true` — defines this experiment
- `noise_recurrent_level: 0.0` — CLOSED, harmful at all tested levels

### Primary exploration axes

| Parameter | Baseline | Exploration range | Notes |
|-----------|----------|-------------------|-------|
| `lr_embedding` | 0.003 | 0.002–0.005 | Confirm + fine-tune |
| `data_augmentation_loop` | 7 | 5–14 | More steps per unique position |
| `batch_size` | 4 | 4–8 | Larger batches = smoother BPTT gradients |
| `lr_W` | 0.0009 | 0.0005–0.0015 | May shift with more data diversity |
| `grad_clip_W` | 0.0 | 0.1–1.0 | Stabilize BPTT with spectral radius > 1 |
| `lr_scheduler` | none | `cosine_warm_restarts` | Stabilize BPTT convergence |

### Regularization (may need rebalancing)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `coeff_g_phi_diff` | 750 | Monotonicity of g_phi — critical for rollout stability |
| `coeff_W_L1` | 0.00015 | L1 on W |
| `coeff_g_phi_weight_L1` | 0.28 | L1 on g_phi weights |

## Regularization Annealing

With `n_epochs=1` and `regul_annealing_rate=0.0` (fixed), **all annealed L1/L2 are inactive**.
Non-annealed (always active): `coeff_g_phi_diff`, `coeff_g_phi_norm`, `coeff_f_theta_msg_diff`.

## Parallel Mode — 4 Slots Per Batch

All 4 slots run the **same config** with different random seeds (assigned automatically).

**Robustness classification:**

- **Stable-Robust**: all 4 seeds conn_R2 > 0.90 AND CV < 3% — **TARGET**
- **Robust**: all 4 seeds > 0.90, CV 3–5%
- **Partially robust**: 2–3 seeds > 0.90
- **Fragile**: 0–1 seeds > 0.90 — reject
- **DISQUALIFIED**: any seed < 0.87

Note: the hard floor for this exploration is > 0.70 (already beating stride_5 robust ATB). Reaching 0.90+ would be a major result.

## Block Partition

| Block | Focus | Key parameters | Goal |
|-------|-------|----------------|------|
| 1 | **Diversity-parity baseline** | 320K frames, DAL=7, lr_emb=0.003 | Does 5× data diversity close the stride_5 gap? |
| 2 | **DAL scaling** | DAL ∈ {5, 6, 8} or bs=8 + DAL=16 | More gradient steps per unique position vs larger batches |
| 3 | **Gradient clipping** | grad_clip_W ∈ {0.1, 0.3, 0.5, 1.0} | Stabilize BPTT with spectral radius > 1 |
| 4 | **lr_W fine-tuning** | lr_W ∈ {0.0005, 0.0009, 0.0012, 0.0015} | Does optimal lr_W shift with 5× more data? |
| 5 | **LR scheduler** | cosine_warm_restarts vs linear_warmup_cosine | Stabilize BPTT convergence |
| 6 | **Regularization rebalance** | coeff_g_phi_diff, coeff_W_L1 | Recurrent signal may need looser or tighter constraints |
| 7 | **Combined best** | Integrate findings from blocks 1–6 | Best stride-5-yt config |
| 8 | **Final validation** | Best config vs stride-1 (0.980) and stride-5 (0.563) | Does more data close the gap? |

**Block 1 priority**: The key question is whether the diversity fix alone is sufficient. If Block 1 robust mean > 0.70, the diversity hypothesis is confirmed and BPTT is viable. If mean ≈ 0.56, the ceiling is architectural/spectral, not data-related.

## Training Time Budget

**Hard constraint: ≤ 90 minutes per iteration** (1 epoch, batch_size=4).

`Niter = int(n_frames × DAL // batch_size × 0.2)`

With **n_frames=320,000**, **batch_size=4**:

| DAL | Niter | Est. time (min) | Status |
|-----|-------|-----------------|--------|
| 6 | 96,000 | ~80 | under-budget |
| 8 | 128,000 | ~105 | ← **baseline** ✓ |
| 10 | 160,000 | ~130 | **EXCEEDS BUDGET** |
| 12 | 192,000 | ~160 | **WAY OVER** |

**Hard cap: DAL ≤ 8 at batch_size=4 with 320K frames.** To test higher DAL, reduce n_frames or increase batch_size.

**Batch size scaling (DAL=8 baseline)**:
- bs=4, DAL=8 → Niter=128K, ~105 min ← baseline
- bs=8, DAL=8 → Niter=64K, ~55 min → allows DAL=16 for same time (smoother gradients)
- bs=8, DAL=16 → Niter=128K, ~105 min → same Niter, larger batches

## Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| `connectivity_R2` | R² of learned vs true W (PRIMARY) | > 0.70 (floor), > 0.90 (goal) |
| `raw_W_R2` | R² before Procrustes alignment | Close to conn_R2 = alignment stable |
| `tau_R2` | R² of learned vs true time constants | > 0.5 |
| `V_rest_R2` | R² of learned vs true resting potentials | > 0.1 (was near-zero in stride_5) |
| `rollout_pearson` | Autoregressive rollout Pearson r | > 0.93 (instability below this) |
| `onestep_pearson` | One-step prediction Pearson r | > 0.90 |
| `training_time_min` | Wall time | ≤ 90 |

**Diagnostic flags (from stride_5 experience)**:
- `raw_W_R2 < 0.3`: gradient explosion — reduce lr_W
- `rollout_pearson < 0.93`: dynamics instability — reduce lr or increase coeff_g_phi_diff
- Procrustes hurts (raw_W_R2 > conn_R2): sign distortion — indicates large noise or unstable training
- `tau_R2 > 0.5`: either truly recovered OR lucky data seed — verify across 4 seeds

## Variable Names

- **`{base_config_name}`**: `flyvis_noise_005_stride_5_yt`
- **`{llm_task_name}`**: `flyvis_noise_005_stride_5_yt_Claude`

**Config file paths:**
- `config/fly/flyvis_noise_005_stride_5_yt_Claude_00.yaml` through `_03.yaml`
- `config/fly/flyvis_noise_005_stride_5_yt_winner.yaml`

> **YAML rule**: Always wrap the `description` field in double quotes — colons inside unquoted YAML strings cause parse errors.

## File Structure

### 1. Full Log (append-only)

**File**: `flyvis_noise_005_stride_5_yt_Claude_analysis.md`

### 2. Working Memory (read + update every batch)

**File**: `flyvis_noise_005_stride_5_yt_Claude_memory.md`

### 3. User Input

**File**: `user_input.md`

## Iteration Workflow (every batch)

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

**Metrics from `analysis.log`:**
- `connectivity_R2`: PRIMARY — compare to stride_5 best (0.563 robust, 0.6596 single-seed ATB) and stride-1 (0.980)
- `raw_W_R2`: alignment diagnostic
- `tau_R2`, `V_rest_R2`, `rollout_pearson`, `training_time_min`

**Questions to answer each batch:**
- Did diversity fix close the gap vs stride_5? (Is mean > 0.70?)
- Any gradient explosion (raw_W_R2 < 0.3, NaN)?
- Wall time within budget (≤ 90 min)?

### Step 3: Write Log Entry

```
## Iter N: [robust/partially_robust/fragile/DISQUALIFIED] [BETTER/WORSE/SAME vs stride_5/stride-1]

Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: time_step=5, n_frames=320K, DAL=X, lr_W=X, lr=X, lr_emb=X, grad_clip=X, bs=X
Slot 0: conn_R2=A, raw_W_R2=B, tau_R2=C, V_rest_R2=D, rollout_p=E, time=Fmin
Slot 1: conn_R2=A, raw_W_R2=B, tau_R2=C, V_rest_R2=D, rollout_p=E, time=Fmin
Slot 2: conn_R2=A, raw_W_R2=B, tau_R2=C, V_rest_R2=D, rollout_p=E, time=Fmin
Slot 3: conn_R2=A, raw_W_R2=B, tau_R2=C, V_rest_R2=D, rollout_p=E, time=Fmin
Seed stats: mean_conn_R2=X, std=Y, CV=Z%, min=W
vs stride_5 robust baseline: [+X.XXX / -X.XXX]
vs stride-1 (0.980): [+X.XXX / -X.XXX]
Mutation: [param]: [old] → [new]
Verdict: [supported/falsified/inconclusive] — [explanation]
BPTT note: [explosion/NaN/unusual dynamics]
Next: parent=P
```

### Step 4: Acknowledge User Input

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Winner Config (COMPULSORY at every block boundary)

Save to `config/fly/flyvis_noise_005_stride_5_yt_winner.yaml`:

```yaml
# Winner config: flyvis_noise_005_stride_5_yt_winner.yaml
# Source: iter_XXX_slot_YY (connectivity_R2 = X.XXX)
# vs stride_5 robust baseline: +/-X.XXX
# vs stride-1 (0.980): +/-X.XXX
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [diversity fix contribution]
#   - [key hyperparameter choices for BPTT stability]
#
# Metrics:
#   connectivity_R2: X.XXX (best single seed)
#   robust_mean:     X.XXX +/- X.XXX (N seeds, CV=X.X%)
#   tau_R2:          X.XXX
#   V_rest_R2:       X.XXX
#   rollout_pearson: X.XXX
#
# Key differences from stride_5:
#   - n_frames: 64,000 → 320,000 (parity fix)
#   - data_augmentation_loop: 35 → 7 (same wall time)
#   - lr_embedding: 0.002325 → X
```

## Block Boundaries

1. Update "Paper Summary" in memory.md
2. Summarize findings in "Previous Block Summaries"
3. Update "Established Principles" (3+ iterations, cross-seed consistency)
4. Move falsified hypotheses to "Falsified Hypotheses"
5. Clear "Current Block"
6. **Compare best result to stride_5 (0.563 robust, 0.6596 single-seed ATB) AND stride-1 (0.980)**

## Known Prior Results

| Config | robust mean conn_R2 | CV% | Best single-seed | vs stride-1 |
|--------|---------------------|-----|------------------|-------------|
| stride-1 winner | **0.980** | <1% | 0.984 | baseline |
| stride_5 (64K frames, DAL=35) | 0.563 | 2.6% | 0.6596 | −0.417 |
| stride_5_yt (320K frames, DAL=7) | **?** | ? | ? | ? |

Pseudoinverse analysis (noise_005): global conn_R2 = 0.71 — linear upper bound without dynamics.

## Start Call

When prompt says `PARALLEL START`:

- Read base config `config/fly/flyvis_noise_005_stride_5_yt_Claude_00.yaml`
- Set all 4 configs identically to the diversity-parity baseline (320K frames, DAL=7, lr_emb=0.003)
- Write to working memory:
  - **Hypothesis**: "5× data diversity (320K frames, 64K unique stride-5 positions) closes the BPTT gap: robust mean conn_R2 > 0.70 (floor), ideally > 0.90"
  - **Null hypothesis**: "The stride_5 ceiling (0.563) is spectral-radius limited, not data-diversity limited — 320K frames yields similar robust mean ≈ 0.563"
- Block 1 tests the hypothesis directly — do NOT change hyperparameters yet

---

# Working Memory Structure

```markdown
# Working Memory: flyvis_noise_005_stride_5_yt

## Paper Summary (update at every block boundary)

- **GNN + recurrent BPTT (stride-5, YouTube-VOS)**: [pending]
- **Null-space reduction via BPTT with parity diversity**: [pending]

## Knowledge Base

### Comparison Table

| Iter | n_frames | DAL | lr_W | lr_emb | grad_clip | conn_R2 (mean±std) | CV% | min | vs stride_5 | vs stride-1 | Stability | Note |
| ---- | -------- | --- | ---- | ------- | --------- | ------------------ | --- | --- | ----------- | ----------- | --------- | ---- |
| 1    | 320K     | 8   | 0.0009 | 0.003 | 0.0     | ?                  | ?   | ?   | ?           | ?           | ?         | baseline |

### Established Principles (from stride_5 — do not re-test)

1. lr_W=0.0009 is locally optimal; > 0.002 causes explosion.
2. lr=0.0018 required; lower fatal, higher harmful.
3. noise_recurrent_level > 0: DEFINITIVELY HARMFUL. Do not test.
4. lr_emb=0.003: best single-seed in stride_5 (0.6596) — confirm here first.
5. Single-seed comparisons unreliable (CV≈11% in stride_5). Only 4-seed tests are valid.

### New Established Principles (from this exploration)

[3+ supporting iterations AND cross-seed consistency required]

### Falsified Hypotheses

### Open Questions

1. Does 5× data diversity (320K frames) lift robust mean above 0.70?
2. If yes, which remaining parameter limits performance toward 0.980?
3. If no, is the ceiling spectral-radius structural?

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest→newest.**

---

## Current Block (Block 1)

### Block Info

Focus: Diversity-parity baseline
Iterations: 1 to n_iter_block

### Current Hypothesis

**Hypothesis**: 5× data diversity (320K frames → 64K unique stride-5 positions) closes the training starvation gap. Robust mean conn_R2 > 0.70.
**Null**: Ceiling ≈ 0.563 regardless of data volume (spectral-radius limited).
**Test**: All 4 slots at baseline config — no hyperparameter changes.
**Expected outcome (diversity-limited)**: mean > 0.70, all seeds > 0.65.
**Expected outcome (spectral-limited)**: mean ≈ 0.56, same as stride_5.
**Status**: untested

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
