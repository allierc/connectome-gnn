# Exploration brief — flyvis_noise_005_010 code-change loop

This is the **single** instruction document for the code-change exploration.
Both the per-block R/S/C code-session agent and the per-iteration HPO-analysis
agent read this exact file (it resolves to `base_state.instruction_path` in
both pipelines). Each agent focuses on the section for its phase and the
shared context; other phase sections are reference-only.

---

## Shared context
*Every consumer reads this section.*

### Objective

Raise connectivity **W R² above 0.76** on flyvis_noise_005_010
(σ_intrinsic = 0.05, γ_measurement = 0.10) with **≥ 3-seed stability**,
*without degrading* the other recovered unknowns: τ R², V_rest R²,
clustering accuracy.

### Training regime — read first

Baseline for this exploration is the **winner** (non-recurrent, 1 epoch,
batch_size 6, data_augmentation_loop 20, no pretrained checkpoint, a100,
~60 min/iteration). Every iteration trains from scratch; there is no
warm-start plateau. Config:
`config/fly/flyvis_noise_005_010_code_change.yaml` (training block mirrors
`config/fly/flyvis_noise_005_010_winner.yaml`).

### Pre-block baseline (winner)

| metric                | mean     | std     |
|-----------------------|----------|---------|
| W R² (primary)        | 0.7457   | 0.0043  |
| τ R²                  | 0.850    | —       |
| V_rest R²             | 0.042    | —       |
| one-step Pearson r    | ~0.93    | —       |

Source: `config/fly/flyvis_noise_005_010_winner.yaml`, 4 seeds, CV = 0.58 %.
Key lever in the baseline: `coeff_g_phi_diff = 2000` (the only effective
HPO lever found under this noise regime). W slope bias ~0.51 (model
under-estimates weight magnitudes ~2×). This is the target to beat.

### Inverse-problem frame — DO NOT LOSE SIGHT

- **Observed**: voltage `v(t)`, stimulus `e(t)`.
- **Unknown**: edge weights `W`, time constants `τ_i`, rest potentials
  `V_rest_i`, cell types (65 discrete classes).

τ and V_rest are **NOT learned by a head.** They are extracted post-hoc by a
**linear fit on the trained `f_theta` MLP** near each neuron's rest state.
Any regularizer on `f_theta` that improves the conditioning of that fit is
fair game (see block 3).

### Ground-truth ceilings for calibration

- **Noise-free (γ = 0) W R² ≈ 0.965** — physical ceiling.
- **Known-ODE oracle under γ = 0.10 → W R² ≈ 0.78.** This is the tightest
  bound available at this measurement-noise level; the GNN has more
  structure (embedding, phi) and can in principle approach or exceed it.
  Winner baseline 0.7457 sits just below this ceiling — breaking 0.78 is
  the aspirational target, 0.76 the concrete one.

### Already-falsified hypotheses — do NOT retry

All found inefficient or catastrophic in prior HPO-only exploration.
Proposing any of these is a waste of phase time. **Note: entries marked
(recurrent-only) were falsified under recurrent_training=true; the current
regime is non-recurrent, so those entries are advisory, not closed.**

- **Derivative smoothing in non-recurrent mode** — catastrophic (W R² → 0.03).
- **Multi-epoch training > 1** — harmful in both 1-step and recurrent.
- **`coeff_f_theta_msg_diff > 100`** — catastrophic (W R² → 0.58 at 200).
- **`coeff_g_phi_norm < 0.9`** — strictly worse across 0.5–2.0 sweep.
- **`coeff_W_L1` above 5e-4 or below 5e-5** — sweet spot sharp; do not widen.
- **Plain LR / batch / DAL single-variable scaling** — bounded by the
  derivative-noise SNR; HPO has exhausted this surface.
- **`noise_recurrent_level > 0`** (recurrent-only) — all tested levels hurt.
- **Dale's law OFF** (recurrent-only) — worse in recurrent mode; not
  tested in current non-recurrent regime.

### Scientific method — non-negotiable

- **ONE hypothesis per block.** No "try A or B".
- **Falsifiable.** Phase-S test's PASS/FAIL condition stated BEFORE the test
  runs, in the staged function's docstring.
- **Causal verdict** (Phase V triple-check): ΔW R² ≥ 0.005 AND ≥ 75 % seeds
  better than pre-median AND no seed catastrophically below pre-min − 0.02.
  Anything else → REVERT.
- **Revert adds the failure to the falsified list** so future blocks don't
  retry it.

### Metric-key map (the pipeline has two metric conventions — be aware)

| HPO analysis log (per iter) | metrics.txt / verdict   | used by                        |
|-----------------------------|-------------------------|--------------------------------|
| `connectivity_R2`           | `W_corrected_R2`        | primary → verdict's `W_R2`     |
| `cluster_accuracy`          | `clustering_accuracy`   | secondary                      |
| `tau_R2`, `V_rest_R2`       | same key in both        | secondary                      |
| `rollout_pearson`           | (not in metrics.txt)    | HPO per-iter only              |
| `onestep_pearson`           | (not in metrics.txt)    | HPO per-iter only              |
| `training_time_min`         | (not in metrics.txt)    | HPO per-iter only              |

HPO agent parses the per-iter log. Verdict parses `results/metrics.txt` via
`verdict.collect_metrics_from_run_dirs`. Do not assume keys are identical.

### Block themes — fixed order

1. **Noise removal / denoising** — auxiliary denoising MLP, EMA estimate,
   smoothed-target consistency loss, learned Kalman-style prior on voltage.
2. **Recurrent-training scheme improvements** — truncated BPTT variants,
   teacher-forcing schedules, multi-start weighting, noise-aware unroll.
3. **Identifiability regularization** — `f_theta` linearization prior near
   V_rest (tightens post-hoc τ/V_rest linear fit); contrastive embedding
   loss for cell-type clustering; group-sparsity on W by presynaptic type.
4. **Best-of combination** — union of Phase-C wire-ups KEPT in blocks 1–3.
5. **Robustness validation** — N ≥ 8 seeds, CV < 1 %, leave-one-out ablation.

### Staging conventions

| Artefact                  | Path                                                             |
|---------------------------|------------------------------------------------------------------|
| Mechanism + pytest        | `src/connectome_gnn/LLM_code/staging/block_NN/`                  |
| Analysis function(s)      | `src/connectome_gnn/LLM_code/staging/block_NN/analysis/`         |

Phase-C wire-up allow-list (production edits only here; anything else
requires a human update to this file):

- `src/connectome_gnn/models/regularizer.py` — add a `COMPONENT` entry
  (shortest surface; prefer this).
- `src/connectome_gnn/models/recurrent_step.py` — recurrent-only blocks.
- `src/connectome_gnn/models/graph_trainer.py` — call-site only (1–3 lines).
- `src/connectome_gnn/models/neural_gnn.py` — `__init__` + `forward` if
  unavoidable.

---

## [Phase R] Research + optional analysis staging — 10 min cap
*Read by the code-session agent at block start.*

Your job this phase:

1. Read memory (prior-block verdicts, falsified list) and, if useful,
   allowlisted literature (`src/connectome_gnn/LLM_code/literature/allowlist.json`).
2. **Optionally** stage ONE standalone analysis function at
   `src/connectome_gnn/LLM_code/staging/block_NN/analysis/<name>.py`
   if existing plots don't answer a question you need to answer. The harness
   runs it with Bash and captures its stdout + any saved plots into
   `research_block_NN.md`.
3. Write EXACTLY ONE concrete falsifiable hypothesis + Phase-S function
   signature to `research_block_NN.md`:

```markdown
# Block NN — Research
## Theme: <theme from block table>
## Hypothesis
<one sentence stating a testable mechanism>
## Phase S function signature
```python
def <name>(...) -> ...:
    """<one-line description>
    PASS CONDITION: <exact measurable criterion written NOW>
    """
```
## Rationale
- <≤ 5 bullets tying the hypothesis to memory or literature>
```

**Do not**:
- Edit any production source file.
- Propose anything on the falsified list.
- Propose "try A or B" — choose one.

Tools this phase: Read, Grep, Glob, WebFetch (allowlist only), Bash, Write /
Edit scoped to `staging/block_NN/analysis/`.

---

## [Phase S] Staging — mechanism function + pytest — 10 min cap
*Read by the code-session agent after Phase R completes.*

Create EXACTLY TWO files under `src/connectome_gnn/LLM_code/staging/block_NN/`:

1. **The staged function module** `<name>.py` with the exact signature and
   PASS CONDITION from Phase R's docstring.
2. **A standalone test script** `test_<name>.py` that exercises the
   function on the full flyvis dataset cache and prints on its last line:
   - `PASS: <one-line summary>` on success, exit code 0
   - `FAIL: <one-line reason>` on failure, exit code nonzero

Use the cached loader (loads once per block, ~2 min first call):

```python
from connectome_gnn.LLM_code.scratchpad import load_full_voltage
v_clean, v_noisy = load_full_voltage('fly/flyvis_noise_free', 0.10)   # (T, N)
```

**Do not**:
- Edit any production source file.
- Require a GPU in the test — the harness runs tests on CPU.
- Fake PASS. If the mechanism doesn't exhibit the hypothesised effect,
  print FAIL and exit nonzero. The Phase-C gate depends on an honest PASS.

Tools this phase: Read, Write / Edit under `staging/block_NN/`, Bash (to run
your test once before declaring done).

---

## [Phase C] Wire-up — 5 min cap
*Read by the code-session agent after Phase S emits PASS.*

Wire the staged function into production with a **minimal** edit — typically
3–10 lines across AT MOST the four allow-listed files listed in the shared
`Staging conventions` table.

**Mandatory: seed the new coefficient.** Your wire-up usually exposes a
`coeff_<name>` YAML key. You MUST do ONE of:

- **(a)** set a reasonable non-zero default in production YAML defaults
  (e.g. `config.py` default list), OR
- **(b)** print to stdout at the end of your session an explicit directive:

```
HPO-HANDOFF: new coefficient `coeff_<name>` added; default 0.
             Seed log-scale sweep across the 4 slots in batch 1
             (e.g. 0.001, 0.01, 0.1, 1.0).
```

(b) is the safe path when production YAML defaults are out of your scope.
The HPO within-block agent reads the `Phase-C hand-off → HPO` section and
will honour either (a) or (b). **Do not** leave the default at 0 silently
— the mechanism will never train and verdict will REVERT for the wrong
reason.

**Do not**:
- Edit any file outside the allow-list.
- Copy the function body into production — import it from the staged
  module. The staged module is the source of truth.
- Add more than one new COMPONENT or one new call site.
- Remove existing regularizers or edit unrelated logic.

**torch.compile / pydantic safety (HARD RULE)**:
The training step is wrapped by `torch.compile`. Dynamo cannot trace
attribute access on Pydantic models (`TrainingConfig`, `NeuralGraphConfig`,
etc.). Any `getattr(tc, ...)`, `tc.foo`, or `cfg.bar` reached from
`Regularizer.compute()`, `compute_update_regul()`, or any function called
inside the compiled forward/loss path will crash with
`torch._dynamo.exc.Unsupported: __getattribute__`.

Required pattern when wiring a new coefficient that needs config-derived
constants (warmup_fraction, rampup_iters, etc.):

1. Read every config value ONCE at `__init__` (or inside `_update_coeffs`,
   which runs outside the compiled region) and store as a plain Python
   float / int / tensor on `self` — e.g. `self._msg_lin_warmup = float(
   getattr(tc, 'f_theta_msg_linearity_warmup_fraction', 0.3))`.
2. Inside `compute()` use ONLY `self._msg_lin_warmup` — never touch `tc`.

If the new coefficient lives only in `self._coeffs[<name>]` (already
populated by `_update_coeffs`) you are fine. Block 3 of this exploration
crashed 9/9 runs by violating this rule — do not repeat it.

When done, print a one-paragraph summary covering:
1. Which production file(s) you edited and how many lines.
2. The YAML config key (if any) added for the coefficient.
3. The expected effect on training.

The harness will `git add` + `git commit` your changes on branch
`agentic_code_change` after this phase ends.

Tools this phase: Read, Edit (allow-list files only), Bash (read-only).

---

## [Phase V] Verdict — automatic, no LLM
*Documentation only. Stated here so every other phase knows the bar.*

After the block's training iterations finish, `verdict.decide` runs the
triple-check on per-seed `W_R2` (aliased from `W_corrected_R2` in
`metrics.txt`) and the no-catastrophe clause on all four guarded metrics
(`W_R2`, `tau_R2`, `V_rest_R2`, `clustering_accuracy`):

- (i)   ΔW R² ≥ 0.005  (mean improves over pre-block baseline)
- (ii)  ≥ 75 % of seeds strictly better than pre-block median
- (iii) no seed below pre-block min − 0.02 on any guarded metric

`KEEP`: block commit stays on branch; baseline updates to new metrics.
`REVERT`: `git revert` the block's commits on the branch; falsified list
updated with the diff.

---

## [Phase-C hand-off → HPO]
*Protocol every consumer reads so nothing falls between the cracks.*

### How HPO discovers this block's new mechanism

At the start of each block's iterations the HPO agent must do **one** of:

1. Read the `Block NN code-session` section in `memory.md` (appended by
   `pipeline._append_to_memory` after the code-session completes) — contains
   a summary with paths and whether Phase C committed anything.
2. Read `src/connectome_gnn/LLM_code/staging/block_NN/` — the function body
   and test tell you what the mechanism does.
3. Read the Phase-C commit on branch `agentic_code_change` — the commit
   message body contains the Phase-C session output tail.
4. Diff your slot YAML against `config/fly/flyvis_noise_005_010_winner.yaml`
   — any new `coeff_*` key is the lever this block introduced.

### First-batch-of-block directive

**Do NOT leave `coeff_<name>` at default 0 for the whole block.** That
trivialises the mechanism: the verdict will always REVERT, not because the
idea is bad, but because the mechanism never trained.

- **Batch 1 (first 4 slots):** sweep the new coefficient on a log scale —
  e.g. `[0.001, 0.01, 0.1, 1.0]` — one per slot. Keep other parameters
  identical to the baseline.
- **Batch 2+:** HPO-style refinement around the best slot from batch 1.
  Use the CAUSALITY RULE — one parameter per slot per iteration.

### If Phase S did not PASS

The code-session skipped Phase C and no new `coeff_<name>` exists. In that
case run the block as a normal HPO refinement against the pre-block
baseline — your task reduces to tuning existing levers.

---

## [HPO within-block]
*Per-iteration analysis agent reads this section.*

### Scope

**Tune coefficients and the existing lever set.** Do NOT touch architecture,
`batch_size`, `n_epochs`, `recurrent_training`, or anything under
`src/connectome_gnn/LLM_code/`. Structural / architectural / new-regularizer
ideas belong to the NEXT block's Phase R, not here.

### CAUSALITY RULE

- One parameter change per slot per iteration.
- Keep at least one slot as a control (unchanged config) so the effect of
  the current block's code change + coefficient value is measurable.

### Safe ranges for existing levers

(Winner baseline defaults shown in "Current".)

| Parameter                  | Current  | Safe range   | Notes                                  |
|----------------------------|----------|--------------|----------------------------------------|
| `coeff_<name>` (this block)| see C    | log-sweep then ±3× | Tune first — why the block exists |
| `coeff_g_phi_diff`         | 2000     | 1200–3000    | Most effective lever; keep ≥ 1200      |
| `coeff_g_phi_weight_L1`    | 0.28     | 0.1–0.5      | Non-zero here (non-recurrent regime)   |
| `coeff_W_L1`               | 1.5e-4   | 5e-5–5e-4    | Sweet spot sharp                       |
| `coeff_W_L2`               | 1.5e-6   | 0–3e-6       | Weak effect                            |
| `coeff_f_theta_weight_L1`  | 0.05     | 0–0.1        |                                        |
| `coeff_f_theta_weight_L2`  | 1e-3     | 0–3e-3       |                                        |
| `lr_W`                     | 9e-4     | 3e-4–1.2e-3  |                                        |
| `lr`                       | 1.8e-3   | 1e-3–2.5e-3  |                                        |
| `data_augmentation_loop`   | 20       | 10–40        | Trades training time for signal        |
| `regul_annealing_rate`     | 0.5      | 0.25–0.75    | Anneal regularizers over training      |

### What counts as progress this iteration

Primary: `connectivity_R2` moves upward (per-iter log) and
`W_corrected_R2` follows in `metrics.txt`. If they diverge substantially,
flag it — possible slope drift (L1 shrinking).

### Do not propose (reiterated from shared context)

See `Shared context → Already-falsified hypotheses`. The HPO agent must not
propose any of them either, even if the current block's code change could
hypothetically interact with them.
