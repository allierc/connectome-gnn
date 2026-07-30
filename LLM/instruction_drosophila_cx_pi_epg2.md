# Drosophila CX — Path Integration (Known-ODE RNN, EPG-only readout) — Knowledge Exploration

## Goal

**The behavioural metric is already met.** The `_epg` agentic loop converged
to `r_roll_1k ≈ 0.99996` on this variant with EPG-only readout; the winner
recipe is baked into the parent yaml
[config/drosophila_cx/drosophila_cx_pi_epg.yaml](../config/drosophila_cx/drosophila_cx_pi_epg.yaml).

**This exploration is about understanding, not goal-finding.** Deliverables
for the paper:

- **Basin geometry**: ordered list of HPs by "perturbation cost" — how much
  r_roll_1k drops per ±50% perturbation. (§Methods recipe justification.)
- **Causal lever**: the single HP that explains the most variance in the
  converged metric. (§Methods / §Discussion.)
- **Failure modes**: when the recipe breaks, *how* it breaks — catastrophic
  bump collapse vs polarity flip vs slow drift. (§Discussion.)
- **Seed robustness**: published numbers need CV across seeds.

## Starting Point

- Parent yaml: `config/drosophila_cx/drosophila_cx_pi_epg.yaml` (already at
  `_epg` winner — do not re-derive).
- Iter 0 known floor: r_roll_1k ≈ 0.99996, mse ≈ 0.0012, fwhm ≈ 35°.
- Dataset (fixed): `drosophila_cx_pi_task` (100k train / 10k test, T=1000).

## Why the agentic loop is primordial

The HP space here is high-dimensional; a single bad config lands in a
non-converging pocket routinely. **Any "X fails" / "Y is a fundamental
limit" claim in the final summary must be qualified by the breadth of
agentic search that produced it** — iter count + axes covered + whether
single-axis or joint sweeps were exhausted. Without the loop, a
non-converging single config is *unsearched*, not *impossible*. This
exploration's epistemic value comes from the systematic coverage, not
from any one slot's result.

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. Write the hypothesis BEFORE running the slot.
2. Change EXACTLY ONE parameter per slot (causality rule).
3. Run 10 slots — 1 control + 9 single-axis mutations in EXPLORATION mode;
   all 10 identical (different seeds) in ROBUSTNESS mode.
4. Read `r_roll_1k`, `mse`, `fwhm_deg`, per-epoch trajectory,
   `collapse_detected` from `tmp_training/metrics.log`. Classify each slot:
   stable-robust / stable / unstable / catastrophic.
5. Revise hypotheses; promote findings per the KB thresholds below.

Only training results validate or falsify — hypothesis alone proves nothing.

### Causality rule (MANDATORY)

If a slot mutates more than one parameter, the effect cannot be attributed —
fatal experimental design error. **EXPLORATION**: slot 0 = parent control,
slots 1–9 each change exactly one axis. **ROBUSTNESS**: all 10 slots same
config (pipeline forces different seeds).

Do not stack two new knobs in one slot just because both look promising —
confirm each in its own slot at the *current* parent first.

## Knowledge Base (in working memory)

### Established Principles

Promote a finding only if ALL of: (a) observed in ≥ 3 iterations, (b) low
CV across the 10 seeds, no catastrophic failures, (c) states a **causal**
relationship (not correlation). Example: *"coeff_tail_loss > 0.04
overshoots — mse climbs ≥ 50% across all 10 seeds; band [0.025, 0.035]
necessary for tight fwhm."*

### Open Questions

Patterns observed once or twice, seed-dependent effects, contradictions.
Becomes a hypothesis in the next batch.

### Falsified Hypotheses

State the original hypothesis verbatim, the contradicting evidence (iter
number + metric values), and what was learned. Propose a revised hypothesis
if applicable.

## File Structure

Under `<data_root>/log/Claude_exploration/LLM_drosophila_cx_pi_epg2/`:

1. **`*_Claude_analysis.md`** — append-only full log (every iter).
2. **`*_Claude_memory.md`** — working memory; read + update every batch.
3. **`user_input.md`** — read every batch; ack pending items by moving them
   to "Acknowledged" with timestamp.

### Working Memory template

```markdown
# Working Memory: drosophila_cx_pi_epg2

## Paper Summary (4 sentences, update at every block boundary)
- HP-causality: <which HP explains most variance, by how much>
- Failure mode: <how the recipe breaks>
- Robustness: <CV across seeds of the winner>
- Surprise: <unexpected interaction if any>

## Knowledge Base
### Robustness Comparison Table
| Iter | Block | Config | r_roll_1k (mean ± std, N) | CV% | catastrophic | Verdict | Hypothesis |

### Established Principles
### Falsified Hypotheses
### Open Questions

## Previous Block Summaries
### Block N: <paragraph summary + verdict>

## Current Block
### Hypothesis: <quoted>
### Iterations this block
### Emerging observations
```

## Iteration Workflow

**Step 1** — Read working memory + user input; identify the current parent
and the next-batch hypothesis.

**Step 2** — Analyse the previous batch's 10 slots; classify each as
stable-robust / stable / unstable / catastrophic (criteria above).

**Step 3** — Log entry, in `*_Claude_analysis.md`:

```
## Iter N (block B): [exploration | robustness]
Parent: iter_M_slot_K (r_roll_1k=X.XXX)
Hypothesis: "<verbatim>"
Mutation: <param>: <old> → <new>   (one axis per slot)
Per-slot: slot k → r_roll_1k=X, mse=Y, fwhm=Z, class=...
Seed stats (robustness only): mean ± std, CV%, catastrophic count
Verdict: supported | falsified | inconclusive
Next parent: <slot> | next hypothesis: <one line>
```

**Step 4** — Acknowledge pending user input (timestamp + move).

**Step 5** — Write the next-batch hypothesis into working memory, edit the
10 config files: slot 0 = (re-)chosen parent, slots 1–9 = single-axis
mutations.

## Block Boundaries

At every block boundary: (1) update Paper Summary, (2) write one-paragraph
block summary, (3) promote consistent findings to Established Principles,
(4) move falsified hypotheses (with evidence), (5) winning slot becomes the
parent for block B+1.

## Block plan (160 iter, 4 blocks × 40 iter, 10 slots × 4 batches)

The spine is **MIXED**: B1 chases the variant's central open question;
B2/B3 sweep HP families; B4 confirms robustness + free exploration.

### B1 — Basin geometry (central open question, iter 1-40)

**Question**: How wide is the basin around the `_epg` winner? Order the HPs
by perturbation cost.

- B1.1 (iter 1-10): ±50% on coeff_tail_loss, lr_W_ED, noise_recurrent_level,
  grad_clip_W, lr_W_rec_schedule (scale ×0.5 / ×2). One axis per slot.
- B1.2 (iter 11-20): ±25% narrower sweep on whichever 3 axes were most
  fragile in B1.1.
- B1.3 (iter 21-30): ±100% push on the flat axes — confirm flatness or
  uncover delayed failure.
- B1.4 (iter 31-40): **ROBUSTNESS** at the iter-0 parent (10 different
  seeds) — measure the winner's intrinsic CV.

Deliverable: an HP-perturbation-cost ranking for §Methods.

### B2 — Connectome-prior axes + initialisation (iter 41-80)

**Question**: Do the aux losses help under EPG-only readout?

Single-axis sweeps across: `coeff_cos_distance` {0, 0.5, 1.0, 2.0},
`(coeff_norm_floor, kappa_norm_floor)` pairs, `coeff_tv_circular`
{0, 1e-3, 1e-2}, `coeff_W_L1` {0, 1e-5, 1e-4, 1e-3}, `w_init_mode`
{const, randn, w_con, zeros}, `w_init_scale` {1e-3, 1e-2, 5e-2, 1e-1, 0.5}.

If any prior helps, §Methods cites the gain; if none helps, §Methods notes
that EPG-only readout makes them superfluous.

### B3 — Architecture and gating (iter 81-120)

**Question**: Does the velocity gate matter, given EPG-only readout?

Sweep `velocity_gate` {pen_4scalar, pen_only, none}, `input_proj` /
`output_proj` {matrix, mlp}, `hidden_dim`, `n_layers`, `MLP_activation`,
`batch_size`, `n_steps_schedule` shape. Hulse's PEN-velocity routing claim
is either supported or falsified here within our training regime.

### B4 — Robustness confirmation + free exploration (iter 121-160)

- B4.1 ROBUSTNESS: best config from B1-B3, 10 different seeds. Report
  mean ± std, CV%, catastrophic count.
- B4.2-B4.4 free exploration: combine the best findings from B1-B3.
  Single-axis vs the *new* parent each slot, but the axis can be any
  combination of two B1-B3 winners.

## Available hyperparameters

All fields the agent may set per-slot, organised by HP family. **One axis
per slot.**

### Learning rates (three-group optimiser)

| Field                   | Parent       | Sweep                              | What                                                                |
| ----------------------- | ------------ | ---------------------------------- | ------------------------------------------------------------------- |
| `lr`                    | 2e-3         | {1e-3, 2e-3, 4e-3}                 | Biases group; also fallback init for `lr_W_rec` / `lr_W_ED`.        |
| `lr_W_rec`              | unset        | {5e-4, 1e-3, 2e-3}                 | Initial lr for `S` recurrent core. Sets schedule[0].                |
| `lr_W_ED`               | **2.5e-3**   | {1e-4, 5e-4, 1e-3, 2e-3, 3e-3}     | Constant lr for `W_in`, `W_out`, velocity-gate scalars.             |
| `lr_W_rec_schedule`     | [5e-4 5e-4 2.5e-4 1.25e-4 1.25e-4] | scale by {0.25, 0.5, 1, 2, 4}      | Per-epoch trajectory of the `w_rec` group only.                     |
| `lr_W_ED_schedule`      | unset        | optional decay or ramp             | Per-epoch trajectory of the `w_ED` group.                           |

### Stabilisers

| Field                   | Parent       | Sweep                              | What                                                                |
| ----------------------- | ------------ | ---------------------------------- | ------------------------------------------------------------------- |
| `noise_recurrent_level` | **0.03**     | {0, 0.01, 0.025, 0.05, 0.1}        | Gaussian noise on `h` at every Euler step. Eval stays deterministic.|
| `grad_clip_W`           | 2.5          | {0, 1, 2.5, 3, 5}                  | Max-norm gradient clip on all trainable params.                     |
| `coeff_tail_loss`       | **0.035**    | {0, 0.02, 0.035, 0.05, 0.1}        | MSE weight on the rollout tail. Drives late-T tracking.             |

### Curriculum

| Field                    | Parent          | Sweep                                                      | What                                  |
| ------------------------ | --------------- | ---------------------------------------------------------- | ------------------------------------- |
| `n_steps_schedule`       | [200×5]         | {[100×5], [200×5], [500×5], [100,200,300,400,500], [300×5]}| Per-epoch BPTT horizon.               |
| `n_epochs`               | 5               | {3, 5, 8} (== len(schedule))                               | Curriculum stages.                    |
| `data_augmentation_loop` | 1               | {1, 2, 5}                                                  | Train-set passes per epoch.           |
| `batch_size`             | 64              | {32, 64, 128}                                              | Trials per gradient step.             |

### Connectome-prior aux losses

| Field                 | Parent | Sweep                  | What                                                          |
| --------------------- | ------ | ---------------------- | ------------------------------------------------------------- |
| `coeff_cos_distance`  | 0.0    | {0, 0.5, 1, 2}         | Per-block cosine alignment to W_con.                          |
| `coeff_norm_floor`    | 0.0    | {0, 0.5, 1}            | Soft floor on mean \|W\| per type-pair block.                 |
| `kappa_norm_floor`    | 0.0    | {0, 0.05, 0.1}         | Floor target for the norm-floor penalty.                      |
| `coeff_tv_circular`   | 0.0    | {0, 1e-3, 1e-2}        | Circular TV on EPG/PEN ring firing rates.                     |
| `coeff_W_L1`          | 0.0    | {0, 1e-5, 1e-4, 1e-3}  | L1 on `S` (synaptic magnitude).                               |

### Initialisation

| Field           | Parent | Sweep                                  | What                                                    |
| --------------- | ------ | -------------------------------------- | ------------------------------------------------------- |
| `w_init_mode`   | **w_con** | {const, randn, w_con, zeros}        | Init template for `S`.                                  |
| `w_init_scale`  | 0.01   | {1e-3, 1e-2, 5e-2, 1e-1, 0.5}          | Scalar multiplier on `S` at init.                       |

### Architecture

| Field            | Parent        | Sweep                                   | What                                                  |
| ---------------- | ------------- | --------------------------------------- | ----------------------------------------------------- |
| `input_proj`     | matrix        | {matrix, mlp}                           | Encoder shape.                                        |
| `output_proj`    | matrix        | {matrix, mlp}                           | Decoder shape.                                        |
| `velocity_gate`  | pen_4scalar   | {pen_4scalar, pen_only, none}           | Anatomical gate on the velocity column of `W_in`.     |
| `hidden_dim`     | 128 (mlp)     | {64, 128, 256}                          | MLP width (when `input_proj` or `output_proj` = mlp). |
| `n_layers`       | 3 (mlp)       | {2, 3, 4}                               | MLP depth (mlp only).                                 |
| `MLP_activation` | relu          | {relu, tanh, leaky_relu, soft_relu}     | MLP nonlinearity.                                     |

## Mutation guardrails — DO NOT change

- `signal_model_name`, `aggr_type`, `lock_edge_signs`, `wrec_param`,
  `output_from_epg_only` (variant identity).
- `simulation:` and `task.path_integration:` blocks (dataset spec).
- `dataset` (must stay `drosophila_cx_pi_task`).
- Seeds (pipeline-controlled).

## Final Summary

At exploration completion (after B4), write **two** outputs:

1. **Per-loop**: `<exploration_dir>/drosophila_cx_pi_epg2_summary.md`.
2. **Shared**: append to
   `/home/node/.claude/projects/-workspace--devcontainer/memory/exploration_results.md`
   under `## drosophila_cx_pi_epg2 — Key Discoveries (YYYY-MM-DD)`.

Eight bullets, **knowledge first, metric last**:

1. **HP causality** — the single HP explaining the most variance in
   r_roll_1k, with the explained fraction.
2. **Failure mode** — how the recipe breaks; HP regime; per-epoch
   trajectory shape; which metric flips first.
3. **Surprise** — unexpected HP interaction.
4. **Falsified hypothesis** — what we expected, what we got, what we
   learned.
5. **Basin geometry** — HP ranking by perturbation cost (the paper's
   robustness profile).
6. **Robustness numbers** — winner r_roll_1k mean ± std (10 seeds), CV%,
   catastrophic count.
7. **Regularisation story** — whether any aux loss helped, under what
   conditions.
8. **Best metric** (output, not goal) — best r_roll_1k + the recipe.
