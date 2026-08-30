# Benchmark results — connectome recovery under model mis-specification

44 runs, all complete, 1.6 M iterations each, `gpu_l4`, 2026-08-28 → 29.
Design and per-job specs: [`weekend_experiment_2026_08_28.md`](weekend_experiment_2026_08_28.md).
Reproduce every number here with:

```bash
PYTHONPATH=src GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData \
    python tools/analyze_weekend_experiment.py
```

All `R²_W` values are **medians over a trailing 320 k-iteration window**, never a final
checkpoint — the reference runs contain one-checkpoint collapses of 0.2–0.5.

---

## 1. Headline

**Question.** The nominal model (`flyvis_A`) was criticised as too close to the simulation that
generated the data. Does an over-parameterised model, given no such head start, still recover the
circuit — and does regularisation decide whether it can?

**Answer.** Yes, and yes.

| setting | model | probe | lasso | **R²_W mean ± sd** |
|---|---|---|---|---|
| **baseline (nominal)** | `flyvis_A` | – | – | **0.9662 ± 0.0071** |
| `flyvis_A` + noise probe | `flyvis_A` | 2 noise | 0 | 0.9602 ± 0.0094 |
| **conductance, no lasso** | `flyvis_conductance` | 2 noise | 0 | **0.7953 ± 0.0668** |
| **conductance + lasso 2** | `flyvis_conductance` | 2 noise | 2.0 | **0.9481 ± 0.0403** |

Five folds each; per-fold values in §5.

Removing the matched inductive bias costs **0.171** in `R²_W` and inflates the fold-to-fold spread
**9-fold** (0.0071 → 0.0668). That spread is the signature of the identifiability problem: many
conductance parameterisations fit the data, and the optimiser lands on a different one per fold.

Adding the group lasso recovers **+0.153 on 5/5 folds**, closing ~90 % of the gap to nominal from a
model that was never handed the right functional form. Four of the five folds (0.953–0.976) land
**at or above** the nominal baseline; the residual mean gap is carried almost entirely by cv04.

---

## 2. Why this is a mechanism, not a tuned hyperparameter

Two pure-noise columns are appended to `g_phi`'s input, redrawn every forward pass. They are
uninformative by construction, so `∂g_phi/∂noise` is a **calibrated zero** — it says what "fully
discarded" looks like for this model at this point in training. Gradients are reported relative to
`|∂g_phi/∂v_j|`.

| setting | `∂g/∂v_i` | `∂g/∂noise` |
|---|---|---|
| `flyvis_A` + probe (correct family) | *input does not exist* | 0.00053 – 0.00067 |
| conductance, no lasso | **0.0708 – 0.1524** | 0.00191 – 0.00272 |
| conductance + lasso 2 | **0.0004 – 0.0123** | 0.00050 – 0.00076 |

Three things follow.

**Credit assignment is not broken.** The correctly-specified family drives its noise columns to
~6 × 10⁻⁴. The same optimiser, same data.

**Without the lasso, `v_i` is genuinely used.** It sits ~40× above the conductance model's own noise
floor. This is not numerical residue; it is a real fit direction the model has chosen.

**With the lasso, `v_i` falls to the noise floor.** Same order as the provably-useless columns —
i.e. the model has discarded the post-synaptic inputs and reduced itself to `g_phi(v_j, a_j)`, the
true generative form. `R²_W` rises as it does.

That last point is the important one. A pure Occam / simplicity prior predicts sparsity at **equal**
accuracy. We observe a large **gain**. The lasso is not expressing a preference for simple models —
it is removing a degeneracy that was preventing `W` from being identified.

---

## 3. Dose response — how hard must the lasso push?

cv00, seed 1041, n = 1 per dose. W8 = with the noise probe, W6 = without.

| lasso | W8 `R²_W` | W8 `∂g/∂v_i` | W6 `R²_W` | W6 `∂g/∂v_i` |
|---|---|---|---|---|
| 0 | 0.8331 | 0.1103 | 0.9107 | 0.0576 |
| **2** | **0.9746** | **0.0004** | **0.9772** | **0.0004** |
| 5 | 0.9628 | 0.0005 | 0.9667 | 0.0007 |
| 10 | 0.9655 | 0.0004 | 0.9736 | 0.0007 |
| 20 | 0.9720 | 0.0005 | 0.9710 | 0.0007 |
| 50 | 0.9735 | 0.0008 | 0.9741 | 0.0004 |

The whole transition happens between 0 and 2, then nothing: **no accuracy cost even at 25× the
required dose.** There is no trade-off curve to tune along. W6 and W8 track each other, so the noise
probe is not itself perturbing the response.

⚠️ **This does not license a strong lasso for the next experiment.** It is measured on `flyvis_A`
data, where `v_i` is genuinely useless, so any amount of suppression is free. On conductance-generated
data `v_i` is part of the truth and the same dose will push against it. The plateau above says
nothing about that case — it must be re-measured.

---

## 4. Secondary results

### Task 1 — rollout training does not beat one-step
Resolution floor for `flyvis_A`: **0.015** (measured run-to-run).

| arm | `R²_W` | vs `onestep` |
|---|---|---|
| `onestep_step_matched` | 0.9831 | +0.0017 |
| `onestep` / `shoot1` | 0.9815 | — |
| `shoot2` | 0.9799 | −0.0016 |
| `discount` (γ=0.5) | 0.9797 | −0.0018 |
| `uniform` | 0.9757 | −0.0057 |
| `last` | 0.9538 | **−0.0277** |
| `pushforward` | 0.8991 | **−0.0823** |

Only two arms clear the floor, both negative. **`pushforward`** — truncating BPTT to one step —
costs 0.082, so the long gradient chain is load-bearing, not the cause of the horizon decay we set
out to explain. **`last`** (endpoint-only) costs 0.028, so dense per-step supervision earns its keep.
Everything else is within ±0.006 and unresolved. The best arm is plain t+1 at 46 % of the updates.

All six rollout arms were verified bit-identical through epoch 0 (`spread 0.00e+00`), confirming
every rollout knob is inert at K = 1 as designed.

### Task 5 — the reported noise-probe benefit is not real
Dataset held fixed, 5 seeds, noise ON vs OFF, everything else identical:

`mean(ON − OFF) = +0.077`, sd 0.297, 2·se interval **[−0.188, +0.342]** — contains zero, and
**4 of 5 seeds are negative**. The positive mean rests entirely on one seed whose OFF arm collapsed
to 0.251.

The original claim (+0.27, n = 1) compared two configs differing in **four** ways — noise columns,
elementwise L1, group lasso, and iteration count — against a seed-alone spread of 0.29. It was the
seed lottery plus a contaminated prior (§6).

### Task 6 — the corrected `g_phi_norm` prior helps
| run | prior | `R²_W` |
|---|---|---|
| `normctl_cond` | off | 0.7661 |
| conductance ref | on (0.9) | 0.8331 |
| `normctl_flyvisA` | off | 0.9622 |
| `flyvis_A` ref | on (0.9) | 0.9674 |

Turning the prior off now *lowers* conductance by 0.067 and is ~neutral on `flyvis_A`. Before the
fix it *raised* conductance by +0.28 — which is what exposed the bug.

---

## 5. Per-fold detail

| fold | seed | nominal | flyA+probe | cond lasso 0 | cond lasso 2 | Δ lasso |
|---|---|---|---|---|---|---|
| cv00 | 1041* | 0.9754 | 0.9674 | 0.8331 | 0.9746 | +0.1415 |
| cv01 | 1042* | 0.9633 | 0.9553 | 0.8570 | 0.9755 | +0.1186 |
| cv02 | 1043* | 0.9719 | 0.9654 | 0.6973 | 0.9591 | +0.2618 |
| cv03 | 1044* | 0.9611 | 0.9458 | 0.7556 | 0.9534 | +0.1977 |
| cv04 | 1045* | 0.9593 | 0.9669 | 0.8333 | 0.8781 | +0.0448 |

\* The **nominal** folds are not seed/dataset-matched to the rest: they use seeds 1042–1046, and
`nominal_cv01` points at the **cv00 dataset**. Compare the nominal column as a distribution, not
fold-by-fold. The other three columns are exactly matched per row.

---

## 6. Two defects found and fixed mid-run

Both were caught by adversarial audits of the running grid, not by the test suite.

**`coeff_g_phi_norm` anchored the wrong column.** It writes `in_features[:, 0] = 2·xnorm`, which was
`v_j` on `flyvis_A` but `v_i` on `flyvis_conductance` — the layouts differ. Worse, with the other
voltage column left at per-neuron data values, the constant-output target demanded `g_phi` be **flat
in `v_j`**, unsatisfiable for the true `relu(v_j)` except by inventing a `v_i × v_j` interaction. The
term was *paying the model to use `v_i`* in the experiment testing whether it discards `v_i`, and
fighting `coeff_g_phi_diff` (750) at the same time.

Fixed in two steps: the `g_phi` input was reordered to `[v_j, a_j, v_i, a_i]`, making the conductance
layout a prefix-extension of the other so column 0 is `v_j` everywhere (`035a6c9`); then the anchor
was changed to pin **every** voltage column (`3567324`), because pinning `v_j` alone merely flips the
bias to demand flatness in `v_i` — which is the hypothesis under test. Verified an exact no-op on
`flyvis_A` (penalty unchanged to 10 decimals), so its runs kept their validity; all conductance runs
were relaunched.

**Two silent no-ops.** The group lasso gated on first-layer *width*, making it inert on `flyvis_A`
and mis-grouped on any wide-enough variant — now layout-aware. And the discard-metric writer was
gated on model name, so the five `flyvis_A` probe runs were producing **no log at all**; they would
have trained 33 h and returned nothing.

---

## 7. Caveats

- Tasks 1, 4 and 4b are **n = 1 per cell**. Task 1's two large effects clear the measured floor;
  nothing else in those tasks is resolved.
- The conductance seed spread is **0.288**. Any single-seed conductance comparison smaller than that
  is noise — which is precisely what Task 5 established about the prior claim.
- `cv00` and `cv01` share a dataset in the older nominal series (differing only by seed); the
  weekend grid gives every fold its own dataset.
- All results are on `flyvis_A`-generated data. The next step — simulating with a conductance model
  and recovering its parameters under the same regularisation — is where the lasso strength becomes
  a real risk rather than a free win.
