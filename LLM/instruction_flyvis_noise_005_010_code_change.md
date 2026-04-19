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

The forward model is a neural-ODE GNN:

```
dv/dt  ≈  f_theta(v, emb, msg, exc)      with   msg = Σ_j W_ij · g_phi(v_j)
```

Near each neuron's rest state, `f_theta` must linearise to the biophysical
form `(V_rest − v)/τ  +  msg/τ`. τ and V_rest are **NOT learned by a head** —
they are extracted post-hoc by a **linear fit on the trained `f_theta` MLP**
around rest. Any regularizer on `f_theta` that improves the conditioning of
that fit is fair game.

### Measurement noise is the true bottleneck of the GNN framework

The regime γ = 0.10 is **not** a convenience parameter — it is a faithful
stand-in for the noise budgets actually seen in experimental voltage and
calcium imaging (γ ≈ 0.05–0.3 of signal amplitude). Any connectome-inverse
method that only works at γ → 0 is of theoretical interest only. Closing the
gap between γ = 0 (W R² ≈ 0.965) and γ = 0.10 (oracle ceiling 0.78; our
baseline 0.7457) is the research problem.

Why this is fundamental for *this* architecture:

- The standard training signal is **pointwise MSE** on predicted `dv/dt`
  vs the observed finite difference `(v[t+1] − v[t]) / Δt`. The observed
  derivative noise variance is `2 γ² / Δt²`; at our settings
  (γ=0.10, Δt=1) the **derivative noise std ≈ 0.14**, which sits on the
  same order as the signal std of `dv/dt` itself. The supervised signal
  is half-drowned before any regularizer fires.
- An MLP `f_theta` has **absorptive capacity**: faced with a noisy
  target, gradient descent finds solutions where `f_theta` soaks the
  noise variance by deforming along the W↔f_theta scale axis (the
  symmetry described below) — the loss drops but `W` drifts toward
  ~0.5× ground-truth slope. This is *exactly* the 0.51 fingerprint
  we observe.
- Voltage-level regularizers (smoothness, denoising, coherence, EMA
  — everything Block 1/2 tried) cannot fix this: they re-process the
  **same noise-corrupted observable** without adding information
  outside its null space.

The systems-ID literature has wrestled with this exact problem for decades
and has converged on a small set of noise-robust formulations that the
pointwise-MSE neural-ODE recipe silently leaves on the table. Any of them
is a legitimate candidate for a code-change block, and none has been tested
in this codebase:

| Family | Canonical reference | Why it helps at γ=0.10 |
|---|---|---|
| **Weak-form / integral loss** | Messenger & Bortz 2021 (*WSINDy*) | Integrates ODE against a smooth test function `φ(t)`: noise averages as `γ / √T`, an order-of-magnitude SNR gain with no derivative estimation. |
| **Multiple-shooting** | Bock 1983; Voss et al. 2004 | Fit over short windows with intermediate states as free variables; decouples long-horizon error accumulation from instantaneous noise. |
| **Physics-informed residual** | Raissi et al. 2019 (*PINNs*) | Penalise the ODE residual `f_theta − dv/dt` on **collocation** points (smooth interpolants of `v`) instead of raw finite differences. |
| **State-space EM / Kalman** | Särkkä 2013 ch. 12 | Treat `v` as a latent state with observation noise `γ`; E-step smooths, M-step fits `W`. Directly noise-aware. |
| **Spectral / Fourier matching** | Brunton et al. 2016 (*SINDy* variants) | Match dynamics in a frequency band where signal dominates noise; high-frequency noise is down-weighted without low-pass smoothing the signal. |
| **Total-variation regularised differentiation** | Chartrand 2011 | Estimate `dv/dt` via TV-regularised integration rather than finite differences; preserves edges while suppressing γ-band noise. |

The GNN framework **inherits** its noise sensitivity from the pointwise
neural-ODE loss. **Replacing or augmenting that loss with any one of the
six recipes above would be a novel contribution to the
connectome-constrained-modelling line** (Lappalainen et al. 2024 Nature,
FlyVis) — that work validated the forward model at low noise; none of its
losses are designed for γ=0.10. This is the research gap.

**Concrete implication for the next block:** if you are proposing a
mechanism that operates on `v`, `dv/dt`, or residual, you should justify
why it is **not** symmetry-blind (see "W↔f_theta scale degeneracy" below)
and how it behaves as `γ → 0.10` rather than as `γ → 0`. A weak-form or
multiple-shooting training objective — orthogonal to, but compatible with,
the anchor-class regularizers — would attack the bottleneck at its root.

### Why the baseline is stuck — the W↔f_theta scale degeneracy

The 0.51 slope bias is **not** an optimisation nuisance; it is a symptom of
a **structural identifiability symmetry** baked into the forward model:

```
(W, f_theta(·, ·, msg, ·))   ↔   (k · W, f_theta(·, ·, msg / k, ·))
```

Scaling `W` by any `k` and the msg-branch of `f_theta` by `1/k` produces
identical voltage trajectories. The MLP has enough capacity to absorb
`k ≈ 2` on the msg side, and it does so because the data alone cannot
distinguish the two solutions. Voltage-level losses (MSE on dv/dt, rollout
Pearson, etc.) are **blind to this symmetry** — which is why HPO has
plateaued at ~0.745 and why Block 1 (denoising) and Block 2 (coherence)
were null: neither injects information orthogonal to the symmetry.

The ground-truth ceilings corroborate this: the GNN at 0.7457 and the
known-ODE oracle at 0.78 sit in the same identifiability basin; the gap
between them is **architectural** (embedding, phi-shape freedom), while the
gap from 0.78 to the noise-free 0.965 is **genuinely noise-limited**.
Closing the first gap requires **anchors that pin the scale**, not more
smoothing or HPO.

Four classes of anchor break the symmetry (non-exhaustive, ranked by how
directly they touch the offending degree of freedom):

1. **`f_theta` msg-slope constraint** — force `∂ f_theta / ∂ msg` ≈ const
   (ideally 1 in normalised units). This is the **most direct** handle —
   it pins `k` exactly where the symmetry lives. Block 3 targeted this but
   crashed 9/9 on a torch.compile bug (now fixed in commit `6957aa7`);
   **the hypothesis itself was never tested.**
2. **Leak-anchor at V_rest** — `f_theta(v=V_rest, emb, msg=0, exc=0) = 0`
   pins the offset per cell type. Fixes V_rest identifiability and
   conditions the post-hoc τ fit.
3. **`g_phi` shape constraints** — `g_phi(V_rest_pre) = 0`, monotonicity,
   boundedness. Does **not** break the scale symmetry directly, but
   limits how much curvature `g_phi` can absorb (why `coeff_g_phi_diff`
   is the only effective HPO lever in the current baseline). Block 4's
   10-point identity loss was in this family, which is why it was
   NEUTRAL — it constrains `g_phi`, not the W↔f_theta coupling.
4. **Structural priors on W** — Dale's law, type-equivariance across the
   65 cell types, group-sparsity by presynaptic type. These inject
   information directly into `W` (orthogonal to the MLP's capacity to
   absorb scale).

**Proposals in future blocks should state which anchor class they
belong to and why the LEAST-tested classes (1, 2, 4) are better bets than
further refinements to class 3.**

### Ground-truth ceilings for calibration

- **Noise-free (γ = 0) W R² ≈ 0.965** — physical ceiling.
- **Known-ODE oracle under γ = 0.10 → W R² ≈ 0.78.** This is the tightest
  bound available at this measurement-noise level; the GNN has more
  structure (embedding, phi) and can in principle approach or exceed it.
  Winner baseline 0.7457 sits just below this ceiling — breaking 0.78 is
  the aspirational target, 0.76 the concrete one.

### What the prior blocks taught us — synthesise, don't re-propose

Read this *together* with the identifiability framing above. Every failed
block tells you something about the symmetry structure, not just "that idea
didn't work":

- **Block 1 — denoising (voltage EMA + derivative SNR reweight): REVERTED.**
  Lesson: losses that re-process existing observations cannot add
  information — at γ=0.10, the derivative SNR is already what it is. Any
  block that proposes "smooth / reweight / auto-encode observed voltage"
  is **re-fighting this battle**. Symmetry-blind → null.
- **Block 2 — coherence: REVERTED.** Lesson: unconstrained cross-neuron
  coherence losses penalise solutions orthogonally to the W↔f_theta
  symmetry, so the optimum simply absorbs the penalty into unused
  directions. Any coherence-style loss must be tied to a physical anchor
  (e.g. type-equivariance) to contribute.
- **Block 3 — f_theta msg linearity: REVERTED on a code bug.** Lesson:
  the *hypothesis was not actually tested* — 9/9 runs crashed from a
  torch.compile ndarray-vs-tensor violation (fixed in `6957aa7`). This
  is anchor class 1, the most-direct handle on the symmetry, and is
  still **open**. Re-proposing it with the fixed wire-up is allowed
  and encouraged.
- **Block 4 — best-of-combination (g_phi 10-pt identity + g_phi
  zero-intercept + msg_linearity retry): REVERTED.** Lessons:
  (i) `g_phi_zero_intercept` was **catastrophic** — it pins g_phi's
  offset and forces the MLP to compensate elsewhere, destroying
  conditioning;
  (ii) 10-point `g_phi` identity was **neutral** — consistent with the
  framing: constraining g_phi doesn't touch the scale symmetry;
  (iii) msg_linearity retry crashed again on a separate compile bug
  (the fix landed after this block was already running).

**Net implication for the next block:** the least-tested anchor classes are
the most promising. In order of expected leverage:

1. **`f_theta` msg-slope** (class 1) — re-try now that `6957aa7` is
   pushed; the mechanism is legitimate and the signal lives exactly on
   the degenerate axis.
2. **Leak-anchor at V_rest** (class 2) — currently V_rest R² = 0.042
   (essentially not recovered). This is both a symptom and an
   opportunity: an explicit `f_theta(V_rest, ·, 0, 0) = 0` constraint
   would likely move V_rest R² substantially AND propagate a scale
   reference through the post-hoc τ fit.
3. **Structural priors on W** (class 4) — Dale's law, type-equivariant
   W parametrisation, or group-sparsity on `W_ij` by presynaptic type
   label. These inject connectome structure the GNN currently ignores.

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

### Mindset — think like a chess player

You have access to **both a local and a global view** of this problem, and
strong proposals exploit both:

- **Global view (strategy).** The board is the identifiability landscape:
  two structural barriers — the γ=0.10 noise bottleneck (A) and the
  W↔f_theta scale degeneracy (B) — define the endgame. The ground-truth
  ceilings (0.78 oracle, 0.965 noise-free) are the target squares. Each
  block is one move in a multi-move plan; the plan only works if you
  anticipate which barrier(s) your current move *and the next one* will
  chip at, and which anchor classes remain uncontested.
- **Local view (tactics).** The current position is the memory + falsified
  list + the per-iter HPO surface. A tactical win is a single falsifiable
  mechanism that PASSES Phase S and survives the Phase-V triple-check.

Weak proposals are purely local ("let me try another g_phi loss") or
purely global ("denoise everything"). Strong proposals are **a local move
with explicit global justification** — name the barrier, name the anchor
class, argue why this specific mechanism advances a plan the next block
can build on. A reverted block is not a wasted move if it rules out a
line of play; a crashed block (Block 3) *is* a wasted move because it
resolves nothing.

### Confidence — the rising sea (Grothendieck)

**This problem is solvable.** It is a well-posed inverse problem with a
0.78 oracle already demonstrated by a strictly simpler estimator; the GNN
has more structure, not less. The question is not whether a solution
exists but how quickly the right framing makes it fall out.

Grothendieck's *rising sea* is the methodological stance to adopt here:
prefer the **reformulation that dissolves the problem** over the hammer
that tries to crack it. The nut is not meant to be shattered — it is meant
to be submerged until the shell softens and yields on its own.

- The W↔f_theta scale degeneracy is not broken by a bigger coefficient on
  an existing loss; it **dissolves** the moment a single well-chosen
  anchor makes the symmetry inadmissible — and recovering `W` then
  becomes almost incidental.
- The γ=0.10 noise bottleneck is not fought by averaging finite
  differences harder; it **dissolves** when the loss is rewritten in a
  form that never needs a finite difference (weak-form, multiple-shooting,
  state-space).

If your candidate mechanism feels like a hammer — "tune this harder", "add
a penalty to compensate" — the sea has not risen far enough yet; reframe.
A proposal where the answer feels obvious in retrospect ("of course that
works — how else could it not?") is the signal that the water has just
reached the nut.

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

### Block themes — fixed order (revised after blocks 1–4)

Blocks 1–4 are **all REVERTED**; see "What the prior blocks taught us"
above. Blocks 5+ are scheduled to tackle the identifiability symmetry
head-on — each block must name the anchor class (1–4) it belongs to in
its Phase-R research doc.

1. ~~Noise removal / denoising.~~ CLOSED. (Symmetry-blind.)
2. ~~Recurrent-training scheme improvements.~~ CLOSED. (Orthogonal to
   the current non-recurrent baseline; regime change out of scope.)
3. ~~Identifiability reg — f_theta msg linearity.~~ CRASHED, untested.
   **Re-opens as anchor class 1 candidate under block 5.**
4. ~~Best-of combination (g_phi anchors).~~ CLOSED. (Anchor class 3 —
   wrong axis; does not break the scale symmetry.)
5. **Noise-robust training objective** (attacks the γ=0.10 bottleneck
   directly, orthogonal to the anchors). Pick **one** concrete recipe
   from the table in "Measurement noise is the true bottleneck":
   weak-form / WSINDy integral loss, multiple-shooting, PINN collocation
   residual, state-space EM / Kalman-smoothed target, spectral matching,
   or TV-regularised derivative estimation. Phase-R must cite the
   reference, state the expected SNR gain at γ=0.10 analytically, and
   pick a falsifiable PASS criterion.
6. **Anchor class 1 — `f_theta` msg-slope.** Constrain
   `∂ f_theta / ∂ msg` via any of: differentiable linear-fit residual,
   Jacobian penalty at representative states, or architectural split
   (linear msg branch + residual MLP). **Pick one.**
7. **Anchor class 2 — leak at V_rest.** Enforce
   `f_theta(V_rest, emb, 0, 0) = 0` (per cell type) and/or explicit
   leak-current parametrisation. Expected: V_rest R² jumps from 0.042.
8. **Anchor class 4 — structural prior on W.** Dale's law (sign per
   presynaptic type), type-equivariant W, or group-sparsity by type.
9. **Best-of combination** — union of Phase-C wire-ups KEPT in 5–8.
10. **Robustness validation** — N ≥ 8 seeds, CV < 1 %, leave-one-out
    ablation. Run only if ≥ 1 block was KEPT.

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

### Before you propose anything — the 30-second framing

The baseline W R² = 0.7457 is stuck because **two distinct problems** are
both active at γ = 0.10. Your hypothesis must address at least one:

- **(A) Noise bottleneck.** Pointwise MSE on `(v[t+1]−v[t])/Δt` has
  derivative-noise std ≈ 0.14 ≈ signal std at γ=0.10. The training
  gradient is half-drowned before any regularizer fires. Attacks:
  weak-form / WSINDy, multiple-shooting, PINN collocation residual,
  state-space EM, spectral matching, TV-regularised derivative.
  (See main "Measurement noise is the true bottleneck" section for refs.)
- **(B) W↔f_theta scale degeneracy.** `(W, f_theta(·,msg,·))` and
  `(kW, f_theta(·,msg/k,·))` produce identical trajectories — any
  voltage-level loss is blind. Anchor classes that break it:
  1. `f_theta` **msg-slope** constraint (class 1) — Block 3 hypothesis,
     untested due to crash; now open again after `6957aa7`.
  2. **Leak-anchor** at V_rest — `f_theta(V_rest, emb, 0, 0) = 0`;
     V_rest R² currently 0.042, huge room.
  3. **Structural prior on W** — Dale's law, type-equivariance,
     group-sparsity by presynaptic type.

**Symmetry-blind ideas are the default failure mode.** Block 1 (denoise),
Block 2 (coherence), Block 4 (g_phi-side anchors) all failed precisely
because they touched neither (A) nor (B). If your proposal is **a new
loss on `v` or `dv/dt` with no term that breaks the symmetry AND no SNR
argument at γ=0.10**, you are re-running a failed block; pick again.

In `research_block_NN.md` you MUST state, in one line each:
- which problem your hypothesis attacks: `(A)`, `(B-1)`, `(B-2)`, `(B-4)`,
  or a specific (A+B) combination;
- for (A): the expected SNR gain at γ=0.10 (analytic, one sentence);
- for (B): why the proposed term is **not** absorbed by f_theta's capacity.

### Your job this phase

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
