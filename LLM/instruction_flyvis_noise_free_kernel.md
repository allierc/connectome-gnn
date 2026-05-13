# FlyVis GNN — Connectome Recovery from Calcium Observable (noise-free)

## Goal

Optimize GNN hyperparameters for maximum **connectivity matrix recovery (conn_R2)** on FlyVis
with **no noise (σ=0.0)** and **calcium observable** (GCaMP6f-convolved): F(t) = (K * V)(t).
The training reads `state.calcium` and `state.stimulus_calcium` instead of voltage and raw
stimulus; the supervision target is dC/dt instead of dV/dt.

Two sub-goals:

- Maximize conn_R2 (primary): recover W from K-filtered neural activity.
- Quantify the **GCaMP recovery cost**: by how much does conn_R2 drop relative to the
  voltage-observable baseline (target ≥ 0.90 — voltage-mode prior winner = 0.923) when the
  same architecture and dataset are trained on the calcium-domain observable?

Exploration seeds from the **noise-free voltage defaults** (embedding_dim=4,
coeff_g_phi_diff=1500, all regul = 0), DAL=35.

## Scientific Context

**Core question: how much HP retuning does the calcium observable need?**

The voltage dynamics are (approximately) linear:
```
τ_i v̇_i = -v_i + V_i^rest + Σ_j W_ij v_j + I_i
```
K is linear and time-invariant, so it commutes with d/dt and with linear sums. Convolving
both sides with K:
```
τ_i ċ_i = -c_i + (τ_d - τ_r) V_i^rest + Σ_j W_ij c_j + (K * I)_i
```
Same algebraic form. So in principle the GNN trained on (calcium, K*I) → dC/dt should
recover the same (τ_i, W_ij) as voltage training, with V_rest scaled by the kernel area
(continuous-time ∫K = τ_d − τ_r ≈ 0.325 s for GCaMP6f; our discrete kernel is unit-sum so
the empirical V_rest scale is 1.0).

What this exploration tests:
- Whether the **same HP regime** that works for voltage transfers to calcium.
- Whether **slower observable dynamics** (calcium is K-smoothed) require lower lr / larger
  embedding / different g_phi regularization.
- Whether **conn_R2 reaches the voltage ceiling** under noise-free conditions, validating
  the linearity prediction.
- Whether the **dynamics-recovery metrics** (τ̂, V̂_rest) match the theoretical prediction
  (τ̂ ≈ τ; V̂_rest scaled by ∫K).

Key knobs that may differ from voltage:
- `lr_W`, `lr_embedding`: calcium signals have a smaller dynamic range (K is a low-pass
  filter) → may need different LR scaling.
- `coeff_g_phi_diff`: the calcium-domain g_phi sees smoother inputs → diff penalty may need
  retuning.
- `embedding_dim`: calcium loses high-frequency info → embedding may need to compensate.

## Noise Model

```
v_i(t+1) = v_i(t) + dt * f(v_i(t), W, a_i, I_i(t))
c_i(t)   = (K * v_i)(t)        ← saved to disk as `calcium.zarr`
sigma    = 0.0 (noise_model_level)
```

No process noise. Train and test conditions identical.

## Observable

**Calcium** throughout. The trainer reads `x.calcium` (column 7) and
`x.stimulus_calcium` (kernel-convolved visual input). The target is dC/dt
(`y_list_train_calcium.zarr`).

`training.observable: calcium` is set in the base config — do not change it during
exploration.

## Metrics

**Always use metrics defined to guide decision making.**

During training (stdout `[metrics]` line, every 300 s):
```
slot N  flyvis_noise_free_kernel_cvXX  iter=I/total  R²W=X.XXX  R²Vr=X.XXX(out%)  R²τ=X.XXX(out%)  cluster=n/a  loss=X.XXe-XX
```

During test/validation:
- **PRIMARY METRIC: `conn_R2`** (R² of learned W vs ground-truth W).
- `tau_R2`: R² of τ recovery. Calcium prediction: τ̂ ≈ τ (the kernel does not modify τ).
- `V_rest_R2`: R² of V_rest recovery. Note the linear-theory prediction is that V̂_rest is
  scaled by ∫K; the metric is invariant to a uniform scale, so the R² should still be
  high if the calcium derivation holds.
- `cluster_accuracy`: cell-type clustering accuracy from neuron embeddings.
- `rollout_pearson_r`: Pearson r of autoregressive calcium rollout vs ground-truth calcium.

**Robustness classification** (4 seeds per iteration):
- **Stable-Robust**: all 4 seeds conn_R2 ≥ 0.85, CV < 3%
- **Stable**: mean conn_R2 ≥ 0.80, CV < 10%
- **Unstable**: mean < 0.80 OR CV ≥ 10%
- **Catastrophic**: any seed conn_R2 < 0.50

**Note on τ_R2 and V_rest_R2**: Model `flyvis_A` absorbs τ and V_rest into f_theta.
These metrics show 0.00 or N/A for many runs — this is expected behavior, not a failure.

Data is **NOT re-generated** each iteration (`generate_data: false`).

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction.
2. **Design experiment**: Change **EXACTLY ONE** parameter at a time per slot
   (causality rule).
3. **Run training**: 4 slots — in EXPLORATION mode each slot is its OWN
   single-parameter mutation from the running-best parent; the previous
   iteration's best config is the implicit baseline. In ROBUSTNESS mode all 4
   slots share the same config (different seeds test robustness).
4. **Analyze**: Use conn_R2 AND cluster_accuracy to understand embedding quality.
5. **Update understanding**: Revise hypotheses based on evidence.

### CAUSALITY RULE (MANDATORY)

If a slot changes more than one parameter from the parent you CANNOT attribute the
effect. Fatal experimental design error.

- In EXPLORATION mode: **all 4 slots are mutations** — each changes exactly one
  parameter from the running-best parent (which is whichever config in the previous
  iteration scored highest mean conn_R2 across its seeds, or the previous parent if
  no slot improved). No dedicated control slot — 100% of compute is exploratory.
  Drift is detected by tracking the running-best score across iterations and
  re-running the best config in ROBUSTNESS mode at block boundaries.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness).

## FlyVis Model

Non-spiking compartment model of the Drosophila optic lobe:
```
tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * g(v_j) + I_i(t)
```
- **13,741 neurons**, 65 cell types, **434,112 edges**.
- **1,736 input neurons** (photoreceptors, DAVIS visual input).
- Process noise σ=0.0 (noise-free).
- 64,000 frames, delta_t = 0.02 s.
- Model `flyvis_A`: f_theta absorbs τ and V_rest implicitly.
- **Visual stimulus**: heaviside-var perturbation layered on DAVIS frames
  (frames_on=35, resample_amplitude_per_transition=True). Same waveform across folds; seed
  varies per CV fold so each fold has an independent realization.
- **Calcium**: F(t) = (K_GCaMP6f * V)(t), unit-sum kernel, τ_rise=75 ms, τ_decay=400 ms.
- **stimulus_calcium**: same kernel applied to the visual stimulus so the excitation
  channel lives in the same temporal regime as the calcium observable.

## GNN Architecture

```
g_phi(v_j, embed_j) → message_ij             (edge MLP)
sum_j W_ij * g_phi(v_j) → agg_i              (weighted aggregation)
f_theta(v_i, agg_i, embed_i) → dv_i/dt       (node update MLP)
```
In calcium mode `v_*` and `dv/dt` are replaced by `c_*` and `dc/dt` everywhere.

- Per-neuron embedding: learnable `embedding_dim`-dim vector.
- **embedding_dim=4** (noise-free default — same as voltage baseline).
- **`g_phi_positive=false`** — CALCIUM-SPECIFIC: voltage mode uses `true` so `g_phi^2 ≈ ReLU(v_j) ≥ 0`. In calcium mode `c_j = K * v_j` can be **negative** (V_rest is negative for many types, the kernel preserves sign), and per Eq. 4c of the calcium derivation the synaptic term is `Σ_j W_{ij} c_j` with NO rectification. Forcing `g_phi^2 ≥ 0` would require W to absorb the sign of c_j, but one W per edge cannot encode both positive-c and negative-c contributions. Set `false` so g_phi can output signed messages and W remains a clean per-edge scalar.

**YOU ARE ONLY ALLOWED TO MODIFY THE PARAMETERS BELOW TO ACHIEVE THE GOAL.**

## GNN Architecture Parameters

| Parameter         | Default | Description                                                              |
| ----------------- | ------- | ------------------------------------------------------------------------ |
| `hidden_dim`      | 80      | Width of hidden layers in g_phi and f_theta                              |
| `n_layers`        | 3       | Depth of g_phi and f_theta networks                                      |
| `embedding_dim`   | 4       | Per-neuron learnable embedding dimension                                 |
| `g_phi_positive`  | false   | **Keep `false` in calcium mode.** Setting `true` squares g_phi output (positive messages) — appropriate for voltage where g_phi ≈ ReLU(v_j), but in calcium mode `c_j` is signed and `Σ W_ij c_j` (Eq. 4c) has no rectification. Flipping to `true` forces W to absorb sign and creates an extra degeneracy. Treat as a diagnostic knob only — flip to `true` for **one** iteration to measure the cost, then revert. |

## Training Parameters

Defaults below match the production master
`flyvis_noise_free_blank50_heaviside_var_kernel.yaml` (the config that
generated the cv00 dataset the LLM seeds from). DAL is overridden to 35
for LLM speed.

| Parameter                 | Default      | Description                                                       |
| ------------------------- | ------------ | ----------------------------------------------------------------- |
| `lr_W`                    | 0.0009       | LR for W (synaptic weights)                                       |
| `lr`                      | 0.0018       | LR for g_phi and f_theta MLP weights                              |
| `lr_embedding`            | 0.002325     | LR for per-neuron embeddings                                      |
| `data_augmentation_loop`  | 35           | Augmentation loops per epoch (DAL). H100 ≈ 30-45 min/run at DAL=35 |
| `batch_size`              | 4            | Samples per batch                                                 |
| `coeff_g_phi_diff`        | 750          | L2 penalty driving g_phi toward non-trivial activations           |
| `coeff_g_phi_norm`        | 0.9          | L2 norm regularization on g_phi (tested in Block 2)              |
| `coeff_g_phi_weight_L1`   | 0.28         | L1 weight regularization on g_phi (tested in Block 4)            |
| `coeff_f_theta_weight_L1` | 0.05         | L1 weight regularization on f_theta (tested in Block 4)          |
| `coeff_f_theta_weight_L2` | 0.001        | L2 weight regularization on f_theta                              |
| `coeff_W_L1`              | 0.00015      | L1 regularization on W (tested in Block 1)                       |
| `coeff_W_L2`              | 1.5e-06      | L2 regularization on W                                            |
| `regul_annealing_rate`    | 0.0          | **MUST be 0.0 with n_epochs=1**                                  |
| `w_init_mode`             | randn_scaled | W init: `randn_scaled`, `zeros`, `uniform_scaled`                |
| `w_init_scale`            | 1.0          | Scale for randn_scaled/uniform_scaled                            |

**Training time budget**: Target ~45 min per run on H100 at DAL=35. Adjust DAL if drifting.

**Hard runtime limit (2880 min)**: Cluster enforces 48-h wall-clock cap. Check for
`_interrupted` in slot log directories. If interrupted, reduce DAL for next iteration.

**Fixed: n_epochs=1** — do not change. `regul_annealing_rate` MUST be 0.0 (annealing
formula yields effective_coeff = 0 at epoch 0).

**Note**: Seeds are pipeline-controlled (`sim_seed = iter × 1000 + slot`,
`train_seed = iter × 1000 + slot + 500`). Do not set seeds in config files.

> **YAML rule**: Always wrap the `description` field value in double quotes — colons
> inside unquoted YAML strings cause parse errors.

## Data Generation

`generate_data: false` — five CV folds are **pre-generated** on disk under
`graphs_data/fly/flyvis_noise_free_blank50_heaviside_var_kernel_cv0[0-4]/`. The base
config points at `cv00`. **Do not modify simulation parameters** (n_neurons, n_frames,
n_edges, delta_t, noise_model_level, calcium_kernel_*).

For Block 4 (CV robustness) the existing five CVs are reused — no regeneration.

## Block Structure

Total: **120 iterations** = 4 blocks × 30 iter/block, 4 slots/iter on **gpu_h100**.

| Block | Focus                                        | Parameters to scan                                                                                                                                        | Notes                                                                                                                                                                              |
| ----- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | **LR sweep** (30 iter)                       | `lr_W`, `lr`, `lr_embedding`, `w_init_mode`, `w_init_scale`                                                                                               | Pure LR landscape. All f_theta/g_phi regul = 0 in the entry-point YAML; do NOT touch regul coeffs here. Find LR triplet (W / MLP / embedding) that gives best conn_R2 unregularized. |
| 2     | **LR + regul + bs sweep** (30 iter)          | All Block-1 knobs PLUS `coeff_g_phi_diff`, `coeff_g_phi_norm`, `coeff_g_phi_weight_L1`, `coeff_g_phi_weight_L2`, `coeff_f_theta_weight_L1`, `coeff_f_theta_weight_L2`, `coeff_W_L1`, `coeff_W_L2`, `batch_size`, `data_augmentation_loop` | Re-open the LR knobs from Block 1 because regul and LR interact. f_theta_L1 ≥ 0.05 is a known W=0 attractor — explore sub-collapse range only. Track all 4 conn_R2 dimensions (W, τ, V_rest, cluster_acc). |
| 3     | **Recurrent training** (30 iter)             | `recurrent_training: true`, `time_step` ∈ {1..8}, `noise_recurrent_level`, and the best LR + regul from Blocks 1-2 (re-tunable)                          | Turn on multi-step rollout supervision. Expect LR to need to drop (gradient through `time_step` integrations). Goal: shrink the GCaMP recovery cost — the rollout loss matches the inference-time use case (Euler integration of dC/dt) so the data-mode mismatch from one-step training disappears. |
| 4     | **Fine-tuning + CV validation** (30 iter)    | First half (iter 1-15): narrow band around Block-3 winner — small perturbations to top 2-3 knobs. Second half (iter 16-30): ROBUSTNESS mode + CV folds.   | Fine-tune for ~15 iter at gpu_h100 with single-CV stability check at iter 10/15. Then ROBUSTNESS pass on cv00, then patch dataset to cv00..cv04 across 4 slots to measure true CV stability. Report mean ± SD across folds. |

> **Block 4 CV procedure** (second half): Patch each slot's emitted YAML to point at a
> different CV fold (`dataset: flyvis_noise_free_blank50_heaviside_var_kernel_cv0[0-4]`).
> Keep `generate_data: false` — the LLM pipeline will not regenerate.

> **LR knob re-opening in Block 2**: this is intentional. The Block-1 optimum was found
> with zero regul; introducing g_phi_diff/L1 will likely shift the optimal lr_W and lr
> because the effective loss landscape changes. Treat (LR triplet, regul triplet, bs/DAL)
> as one joint search space in Block 2.

## File Structure

You maintain **THREE** files:

### 1. Full Log (append-only)
**File**: `{llm_task_name}_analysis.md`

### 2. Working Memory (read + update every batch)
**File**: `{llm_task_name}_memory.md`

### 3. User Input (read every batch, acknowledge pending items)
**File**: `user_input.md`

## Knowledge Base Guidelines

### What to Add to Established Principles

A principle must satisfy ALL of:

1. Observed consistently across **3+ iterations**.
2. Consistent across **all 4 seeds** (not just mean, but low variance).
3. States a **causal relationship** (not just a correlation).

### What to Add to Open Questions

- Patterns observed 1-2 times.
- Seed-dependent effects.
- Contradictions between iterations.

### What to Add to Falsified Hypotheses

1. State the original hypothesis.
2. State the contradicting evidence (iteration number, metrics).
3. State what was learned.
4. Propose a revised hypothesis if applicable.

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

For each slot:

1. Read `conn_R2`, `tau_R2`, `V_rest_R2`, `cluster_accuracy`, `rollout_pearson_r`,
   `loss` from metrics log.
2. Compare conn_R2 to the running-best parent (the implicit calcium baseline) and to
   the voltage prior winner (0.923) — note the GCaMP recovery cost (Δ = voltage − calcium).
3. Check `training_time_min` — adjust DAL for next batch if outside 30-60 min window.
4. Check for `_interrupted` in slot log directory.
5. Classify: Stable-Robust / Stable / Unstable / Catastrophic.

### Step 3: Write Log Entry + Update Memory

```
## Iter N: [stable_robust/stable/unstable/catastrophic]
Parent (running-best from iter N-1): conn_R2=P_R2, params=[summary]
Hypotheses tested this iter: "[quoted slot-0]", "[quoted slot-1]", "[quoted slot-2]", "[quoted slot-3]"

Slot 0: mutation=[param]: [parent_val]->[new],  conn_R2=X, tau_R2=Y, Vr_R2=Z, cluster_acc=W, rollout_r=P, loss=L, sim_seed=S, train_seed=T
Slot 1: mutation=[param]: [parent_val]->[new],  conn_R2=X, tau_R2=Y, Vr_R2=Z, cluster_acc=W, rollout_r=P, loss=L, sim_seed=S, train_seed=T
Slot 2: mutation=[param]: [parent_val]->[new],  conn_R2=X, tau_R2=Y, Vr_R2=Z, cluster_acc=W, rollout_r=P, loss=L, sim_seed=S, train_seed=T
Slot 3: mutation=[param]: [parent_val]->[new],  conn_R2=X, tau_R2=Y, Vr_R2=Z, cluster_acc=W, rollout_r=P, loss=L, sim_seed=S, train_seed=T

Best slot this iter: slot=K, conn_R2=B_R2 (parent was P_R2; Δ=B_R2-P_R2).
New running-best (used as parent for iter N+1): [config summary if any slot beat parent, else parent]
Verdicts (per slot): [supported/falsified/inconclusive] x 4
```

EXPLORATION mode replaces the previous control-slot convention: each of the
4 slots is an independent single-parameter mutation from the **same**
running-best parent (no slot re-runs the parent). The parent is the config
with the highest mean conn_R2 across seeds observed so far; if none of this
iteration's 4 slots beats the parent on conn_R2, the parent is unchanged
and the next iteration tries 4 new mutations.

### Step 4: Acknowledge User Input

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Block Boundaries

At every block boundary:

1. Update "Paper Summary" in memory.
2. Summarize block findings (note GCaMP recovery cost vs voltage 0.923 and whether it
   narrowed).
3. Update "Established Principles" and "Falsified Hypotheses".
4. Clear "Current Block".
5. Carry forward best config as parent for next block.

## Start Call

When prompt says `PARALLEL START`:

- **Default config** (the entry-point
  `config/fly/flyvis_noise_free_kernel.yaml` HPs — calcium-specific defaults,
  all f_theta/g_phi regularizers zeroed; Block 1 seeds from here):
  `lr_W=0.0009, lr=0.0018, lr_embedding=0.002325, batch_size=4, DAL=35`
  `coeff_g_phi_diff=0, coeff_g_phi_norm=0, coeff_g_phi_weight_L1=0`
  `coeff_g_phi_weight_L2=0, coeff_f_theta_weight_L1=0, coeff_f_theta_weight_L2=0`
  `coeff_W_L1=0.00015, coeff_W_L2=1.5e-06, w_init_mode=randn_scaled`
  `embedding_dim=4, g_phi_positive=false, observable=calcium`
- **Block 1 is EXPLORATION mode** (no separate robustness pass): 4 slots
  mutate `lr_W`, `lr`, `lr_embedding`, `w_init_scale` (one parameter per slot)
  from the default. **Do NOT touch any f_theta/g_phi regularization coeff in
  Block 1** — that's reserved for Block 2 where LR and regul are jointly tuned.
  The baseline conn_R2 is implicitly captured by whichever slot in iter 1
  becomes the running-best.
- Hypothesis: "Noise-free voltage HPs transfer to the calcium observable with conn_R2 ≥
  0.85 and CV < 5%. The K-commutes argument predicts a recovery cost ≤ ~5%, so target is
  to come within Δ=0.05 of the voltage prior winner (0.923). The unregularized Block-1
  optimum will be a meaningful baseline because the calcium-domain closed-form linear
  regression already reaches conn_R2≈+0.36 without any regul."
- Launch: `python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_free_kernel iterations=120 --cluster --resume`

---

## Final Summary

At exploration completion (after Block 4), append to
`/home/node/.claude/projects/-workspace--devcontainer/memory/exploration_results.md`:

### flyvis_noise_free_kernel — Key Discoveries (YYYY-MM-DD)

1. **Best metric**: conn_R2 = X.XXX ± std (N seeds, CV=X.X%), winner config = [key params].
2. **GCaMP recovery cost**: Δ_conn_R2 vs voltage-mode 0.923 baseline; absolute and relative.
3. **Which HP changes were calcium-specific** (i.e. departed from voltage-mode optimum).
4. **lr scaling**: Did calcium training need lower / higher / same LRs than voltage?
5. **g_phi diff/norm**: Did the smoother input shift the optimal coeff_g_phi_diff away
   from the voltage value (1500)?
6. **Embedding**: Did the loss of high-frequency info push embedding_dim up?
7. **Regularization**: Did calcium training benefit from non-zero L1/L2 (vs voltage's all-0)?
8. **Theory check**: Did τ̂ ≈ τ as predicted? Did V̂_rest scale as ∫K predicts?
9. **CV robustness**: Block 7 mean ± SD across cv00..cv04 — is it consistent with
   single-cv variance from Block 6?

---

# Working Memory Structure

```markdown
# Working Memory: {llm_task_name}

## Paper Summary (update at every block boundary)

**GNN optimization** (2 sentences on HPO findings):
Sentence 1: Best HP configuration and the conn_R2 it achieves (mean ± std, CV%, N seeds),
vs noise-free voltage prior winner (0.923) — quote the GCaMP recovery cost.
Sentence 2: Which HPs were most critical in the calcium case — what worked and what
failed (cite values and CV impact); how did they compare to the voltage optimum.

**LLM-driven exploration** (2 sentences):
Sentence 1: What the systematic sweep revealed about the calcium-observable optimization
landscape vs voltage (basin width, regularization needs, embedding role, lr scaling).
Sentence 2: Main causal principle — does the linear-algebra prediction (same W, same τ,
V_rest scaled by ∫K) hold, and if not, what does the gap tell us about the GCaMP filter's
information cost?

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean±std) | CV% | catastrophic | Verdict | Hypothesis |
| ---- | -------------- | ------------------- | --- | ------------ | ------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

---

## Previous Block Summaries

**RULE: Keep summaries for the last 4 completed blocks, sorted oldest→newest. This section
MUST appear before ## Current Block.**

### Block N Summary

[Summary of findings from block N]

---

## Current Block

### Block Info

### Current Hypothesis

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
