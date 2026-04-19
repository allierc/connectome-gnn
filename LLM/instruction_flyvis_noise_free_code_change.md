# Exploration brief — flyvis_noise_free code-change loop

This is the **single** instruction document for the code-change exploration on
the *noise-free* regime. Both the per-block R/S/C code-session agent and the
per-iteration HPO-analysis agent read this exact file (it resolves to
`base_state.instruction_path` in both pipelines). Each agent focuses on the
section for its phase plus the shared context; other phase sections are
reference-only.

---

## Shared context
*Every consumer reads this section.*

### Objective

Raise connectivity **W R² above 0.96** on flyvis_noise_free (no dynamics
noise, no measurement noise) with **≥ 3-seed stability**, *without
degrading* the other recovered unknowns: τ R², V_rest R², clustering
accuracy.

Unlike the noisy regime, the ceiling here is **rank-limited, not
noise-limited**. HPO has found a narrow optimum at W R² ≈ 0.945 (24-seed
mean) with a single-seed peak of 0.9665; the remaining gap to the physical
ceiling (≈ 0.965) is driven by effective-rank constraints in the
activations, not by information loss.

### Pre-block baseline (HPO-only noise-free winner)

| metric                    | mean     | std     |
|---------------------------|----------|---------|
| W R² (primary, 24-seed)   | 0.9453   | 0.0159  |
| W R² single-seed (ATB)    | 0.9665   | —       |
| CV                        | 1.68 %   | —       |
| catastrophic seeds        | 0 / 24   | —       |

Source: `config/fly/flyvis_noise_free_winner.yaml` (Iter 77 ATB + Blocks 8/9
CV robustness, exploration date 2026-04-17). The key lever found by HPO was
`coeff_g_phi_weight_L1 = 0.14` (+0.054 over baseline) — without it, the
noise-free case *underperforms* the noisy regime, because noise-free
training lacks the implicit regularisation provided by stochastic
gradients under noise. HPO can tune this knob but cannot invent new
regularisers; that's what this code-change loop is for.

### Inverse-problem frame — DO NOT LOSE SIGHT

- **Observed**: voltage `v(t)`, stimulus `e(t)`.
- **Unknown**: edge weights `W`, time constants `τ_i`, rest potentials
  `V_rest_i`, cell types (65 discrete classes).

τ and V_rest are **NOT learned by a head.** They are extracted post-hoc by a
**linear fit on the trained `f_theta` MLP** near each neuron's rest state.

### Why noise-free is a different problem from noise-free-plus-measurement

In the noise-free regime there is NO derivative noise to reject, NO
measurement corruption to undo, and NO implicit regularisation from
stochastic gradients. The failure mode is opposite to the γ=0.10 case:
HPO overfits the narrow-basin optimum to a specific weight-init seed,
giving high single-seed peaks (0.9665) but high cross-seed variance
(CV = 1.68 %). The code-change interventions must target:

- **rank expansion** of neural activity (effective rank of `g_phi(v)` at
  the aggregator) so more edges become identifiable, and
- **basin-widening regularisation** that does what γ provides implicitly in
  the noisy regime: smooth the landscape so different inits converge to
  the same weights.

### Ground-truth ceilings for calibration

- **Noise-free physical ceiling W R² ≈ 0.965.** Imposed by the structural
  rank of `g_phi(v)` under the flyvis connectome, not by information loss.
  Lift the rank ceiling and this ceiling moves.
- **Rare stochastic seeds under extended data have reached rank 27** (vs
  the typical rank 10); these seeds also hit higher W R² but are unstable.
  Making rank-27 solutions *reliable* is a valid code-change target.
- **Known-ODE oracle under noise-free** — use as a sanity floor:
  your code change should not drag the GNN below the oracle.

### Already-falsified hypotheses — do NOT retry

From the noise-free HPO exploration (108 iterations, 9 blocks, exploration
date 2026-04-17). These are closed questions; proposing them wastes phase
time.

- **`coeff_g_phi_weight_L1` off the narrow peak** — 0.14 is a sharp
  optimum; 0.10 or 0.20 drops W R² by > 0.03. Do not widen the range.
- **`coeff_g_phi_norm > 0`** — harmful in the noise-free case (opposite
  of the noisy regime). Keep at 0.
- **`embedding_dim ≠ 4`** — emb_dim=2 (noisy-regime default) loses 0.02
  W R². Keep at 4.
- **All regularisation off** (the original default) — W R² ≈ 0.89, losing
  the noise-implicit-regularisation benefit without a replacement.
- **Recurrent training `time_step ≥ 2`** — noise-free rollout consistency
  doesn't help when there is no per-frame noise to reject; adds training
  time without benefit.
- **`n_epochs > 1`** — harmful as in the noisy regimes.
- **Trivial LR / batch scaling** — narrow basin; HPO has exhausted this
  surface.
- **Naïve data augmentation (`data_augmentation_loop > 200`)** — fits the
  narrow basin harder, worsens cross-seed variance.

### Scientific method — non-negotiable

- **ONE hypothesis per block.** No "try A or B".
- **Falsifiable.** Phase-S test's PASS/FAIL condition stated BEFORE the
  test runs, in the staged function's docstring.
- **Causal verdict** (Phase V triple-check): ΔW R² ≥ 0.005 AND ≥ 75 %
  seeds better than pre-block median AND no seed catastrophically below
  pre-block min − 0.02. Anything else → REVERT.
- **Revert adds the failure to the falsified list.**

### Metric-key map (the pipeline has two metric conventions — be aware)

| HPO analysis log (per iter) | metrics.txt / verdict   | used by                        |
|-----------------------------|-------------------------|--------------------------------|
| `connectivity_R2`           | `W_corrected_R2`        | primary → verdict's `W_R2`     |
| `cluster_accuracy`          | `clustering_accuracy`   | secondary                      |
| `tau_R2`, `V_rest_R2`       | same key in both        | secondary                      |
| `rollout_pearson`           | (not in metrics.txt)    | HPO per-iter only              |
| `onestep_pearson`           | (not in metrics.txt)    | HPO per-iter only              |
| `training_time_min`         | (not in metrics.txt)    | HPO per-iter only              |

### Block themes — fixed order (noise-free-specific)

1. **Basin-widening regularisation** — replacements / extensions for
   `coeff_g_phi_weight_L1 = 0.14` that broaden the optimum: e.g.
   spectral normalisation on `g_phi`, gradient-penalty on `f_theta`,
   weight-averaging across the last K training steps (SWA).
2. **Effective-rank promotion** — interventions on activations at the
   aggregator that lift the rank of `g_phi(v)` above ~10: e.g. orthogonality
   regulariser on the message matrix, contrastive spread on the edge-
   feature distribution, diversity term across neuron types.
3. **Identifiability regularization** — `f_theta` linearization near
   V_rest (tightens post-hoc τ/V_rest linear fit); contrastive embedding
   loss for cell-type clustering; group-sparsity on W by presynaptic type.
4. **Best-of combination** — union of Phase-C wire-ups KEPT in blocks 1–3.
5. **Robustness validation** — N ≥ 8 seeds, CV < 1 %, leave-one-out
   ablation.

### Staging conventions

| Artefact                  | Path                                                             |
|---------------------------|------------------------------------------------------------------|
| Mechanism + pytest        | `src/connectome_gnn/LLM_code/staging/block_NN/`                  |
| Analysis function(s)      | `src/connectome_gnn/LLM_code/staging/block_NN/analysis/`         |

Phase-C wire-up allow-list (production edits only here; anything else
requires a human update to this file):

- `src/connectome_gnn/models/regularizer.py` — add a `COMPONENT` entry
  (shortest surface; prefer this).
- `src/connectome_gnn/models/recurrent_step.py` — recurrent-only blocks
  (unlikely to be needed in the noise-free case).
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

**Useful analyses specific to noise-free**:
- Plot effective rank of `g_phi(v)` at the aggregator over training.
- Scatter W-recovery error vs. per-type activation rank; do hub types
  (many presynaptic partners) drive the error?
- Eigen-spectrum of the Hessian near the noise-free basin; compare to
  the noise-005 basin width to quantify the "narrow-basin" hypothesis.

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

Use the cached loader for a clean voltage tensor (the noise-free regime
does not add measurement noise, so `noise=0.0`):

```python
from connectome_gnn.LLM_code.scratchpad import load_full_voltage
v_clean, _ = load_full_voltage('fly/flyvis_noise_free', 0.0)   # (T, N)
```

**Do not**:
- Edit any production source file.
- Require a GPU in the test — the harness runs tests on CPU.
- Fake PASS. If the mechanism doesn't exhibit the hypothesised effect,
  print FAIL and exit nonzero.

Tools this phase: Read, Write / Edit under `staging/block_NN/`, Bash.

---

## [Phase C] Wire-up — 5 min cap
*Read by the code-session agent after Phase S emits PASS.*

Wire the staged function into production with a **minimal** edit — typically
3–10 lines across AT MOST the four allow-listed files.

**Mandatory: seed the new coefficient.** Your wire-up usually exposes a
`coeff_<name>` YAML key. You MUST do ONE of:

- **(a)** set a reasonable non-zero default in production YAML defaults
  (e.g. `config.py` default list), OR
- **(b)** print to stdout at the end of your session:

```
HPO-HANDOFF: new coefficient `coeff_<name>` added; default 0.
             Seed log-scale sweep across the 4 slots in batch 1
             (e.g. 0.001, 0.01, 0.1, 1.0).
```

Do NOT leave the default at 0 silently — the mechanism will never train
and verdict will REVERT for the wrong reason.

**Do not**:
- Edit any file outside the allow-list.
- Copy the function body into production — import it from the staged module.
- Add more than one new COMPONENT or one new call site.
- Remove existing regularizers or edit unrelated logic (especially
  `coeff_g_phi_weight_L1 = 0.14` which is a known knife-edge optimum).

When done, print a one-paragraph summary covering:
1. Which production file(s) you edited and how many lines.
2. The YAML config key (if any) added for the coefficient.
3. The expected effect on training.

Tools this phase: Read, Edit (allow-list files only), Bash (read-only).

---

## [Phase V] Verdict — automatic, no LLM
*Documentation only. Stated here so every other phase knows the bar.*

Triple-check on per-seed `W_R2` (aliased from `W_corrected_R2` in
`metrics.txt`) plus no-catastrophe on `tau_R2`, `V_rest_R2`,
`clustering_accuracy`:

- (i)   ΔW R² ≥ 0.005  (mean improves over pre-block baseline)
- (ii)  ≥ 75 % of seeds strictly better than pre-block median
- (iii) no seed below pre-block min − 0.02 on any guarded metric

Note for noise-free: the pre-block baseline has **high cross-seed variance**
(CV = 1.68 %, pre-block min ≈ 0.915). A winning mechanism here must also
*reduce* the variance — not just raise the mean while pushing some seeds
below 0.895. Catastrophes are realistic in this regime, not hypothetical.

`KEEP`: block commit stays on branch; baseline updates to new metrics.
`REVERT`: `git revert` the block's commits; falsified list updated.

---

## [Phase-C hand-off → HPO]
*Protocol every consumer reads.*

### How HPO discovers this block's new mechanism

1. Read the `Block NN code-session` section in `memory.md`.
2. Read `src/connectome_gnn/LLM_code/staging/block_NN/` — the function body
   and test tell you what the mechanism does.
3. Read the Phase-C commit on branch `agentic_code_change`.
4. Diff your slot YAML against `config/fly/flyvis_noise_free_winner.yaml`
   — any new `coeff_*` key is the lever this block introduced.

### First-batch-of-block directive

**Do NOT leave `coeff_<name>` at default 0 for the whole block.**

- **Batch 1 (first 4 slots):** sweep the new coefficient on a log scale —
  e.g. `[0.001, 0.01, 0.1, 1.0]` — one per slot. Keep other parameters
  identical to the baseline.
- **Batch 2+:** HPO-style refinement around the best slot. Use the
  CAUSALITY RULE — one parameter per slot per iteration.

### If Phase S did not PASS

The code-session skipped Phase C; no new `coeff_<name>` exists. Run the
block as a normal HPO refinement against the pre-block baseline.

---

## [HPO within-block]
*Per-iteration analysis agent reads this section.*

### Scope

**Tune coefficients and the existing lever set.** Do NOT touch architecture,
`batch_size`, `time_step`, or anything under `src/connectome_gnn/LLM_code/`.
Structural changes belong to the NEXT block's Phase R.

### CAUSALITY RULE

- One parameter change per slot per iteration.
- Keep at least one slot as a control (unchanged config) so the effect of
  the current block's code change + coefficient value is measurable.

### Safe ranges for existing levers (noise-free-specific)

| Parameter                  | Current | Safe range  | Notes                                   |
|----------------------------|---------|-------------|-----------------------------------------|
| `coeff_<name>` (this block)| see C   | log-sweep then ±3× | Tune first — why the block exists |
| `coeff_g_phi_weight_L1`    | 0.14    | 0.12–0.16   | **Knife-edge**; do not widen past ±0.02 |
| `coeff_g_phi_diff`         | 1500    | 1000–2000   | Noise-free-specific (vs 600 in noisy)   |
| `coeff_W_L1`               | 1.5e-4  | 5e-5–5e-4   |                                         |
| `lr_W`                     | 3e-4    | 1e-4–6e-4   | Lower than noisy regime                 |
| `lr`                       | 9e-4    | 5e-4–1.5e-3 |                                         |
| `lr_embedding`             | 1.55e-3 | 1e-3–2e-3   |                                         |
| `data_augmentation_loop`   | 150     | 100–200     | Trades training time for signal         |
| `batch_size`               | 4       | **do not change** | narrow-basin optimum               |

### What counts as progress this iteration

Primary: `connectivity_R2` rises AND cross-seed variance does not blow up.
If variance rises but mean also rises, flag it — likely the block's
mechanism widens the peak but the HPO search is still in the old narrow
basin. Lowering `coeff_g_phi_weight_L1` slightly may help the new
mechanism take over.

### Do not propose (reiterated from shared context)

See `Shared context → Already-falsified hypotheses`. Noise-free-specific
highlights: do NOT set `coeff_g_phi_norm > 0`, do NOT change
`embedding_dim`, do NOT enable recurrent training, do NOT widen the
`coeff_g_phi_weight_L1` range past ±0.02.
