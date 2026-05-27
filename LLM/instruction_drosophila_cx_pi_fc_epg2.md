# Drosophila CX — Path Integration (FC RNN, EPG-only readout) — Knowledge Exploration

## Goal

**Find a recipe that makes FC + EPG-only readout work *under Dale's law*
(`wrec_param: column_dale`).** Earlier work showed that relaxing the
recurrent matrix to `edge_free` (signed magnitude per edge, no Dale
constraint) trivially unlocks the metric (r_roll_1k ≈ 0.9999), but that
solution is scientifically uninteresting — it discards the connectome
constraint the paper rests on. The `_epg` sister exploration is responsible
for finding the column_dale recipe; this exploration is the **understanding
companion**: given the (eventual) column_dale winner, characterise the
basin, the HP causality, and the failure modes.

Deliverables for the paper:

- **What makes column_dale workable on FC + EPG-only readout**: which HP
  family (lr schedule, tail-loss, aux losses, noise, init) is load-bearing
  vs decorative.
- **Basin around the column_dale winner**: ±50% on each HP — which
  perturbations push r_roll_1k below 0.95? Which leave it untouched?
- **Failure mode when it breaks**: per-epoch trajectory shape — collapse,
  drift, polarity flip, or plateau?
- **Aux-loss role**: `coeff_norm_floor` / `kappa_norm_floor` /
  `coeff_cos_distance` — necessary under column_dale, or superseded by
  the EPG-only readout's implicit constraint?
- **Seed robustness**: CV across 10 seeds at the winner config.

## Starting Point

- Parent yaml: `config/drosophila_cx/drosophila_cx_pi_fc_epg.yaml`
  (`wrec_param: column_dale` + `coeff_tail_loss: 0.05` + EPG-only readout).
- Iter 0 status: r_roll_1k ≈ 0.31 with the current `_epg` recipe — the
  parent recipe is *unsolved* under Dale. Block B1 must find the recipe
  that crosses the r_roll_1k ≥ 0.95 threshold before B2–B4 can map the
  basin around it.
- Dataset (fixed): `drosophila_cx_pi_task`.

## Why the agentic loop is primordial

The HP space here is high-dimensional; a single bad config lands in a
non-converging pocket routinely. **Any "column_dale fails" / "FC is
fundamentally underdetermined" claim in the final summary must be
qualified by the breadth of agentic search that produced it** — iter
count + axes covered + whether single-axis or joint sweeps were
exhausted. A non-converging single config is *unsearched*, not
*impossible*. The premature claim that "column_dale fails on FC + EPG"
would be exactly the kind of single-config mistake the agentic loop is
here to prevent: don't fall back on `edge_free` until B3 has searched
all the column_dale-compatible levers (aux losses, init, curriculum,
encoder/decoder shape).

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

Only training results validate or falsify.

### Causality rule (MANDATORY)

If a slot mutates more than one parameter, the effect cannot be attributed —
fatal experimental design error. **EXPLORATION**: slot 0 = parent control,
slots 1–9 each change exactly one axis. **ROBUSTNESS**: all 10 slots same
config (pipeline forces different seeds).

Do not stack two new knobs in one slot. Confirm each in its own slot at the
*current* parent first.

## Knowledge Base (in working memory)

### Established Principles

Promote a finding only if ALL of: (a) observed in ≥ 3 iterations, (b) low
CV across the 10 seeds, no catastrophic failures, (c) states a **causal**
relationship. Example: *"Under `wrec_param: edge_free`, `coeff_norm_floor`
has zero effect (≤ 0.001 drop in r_roll_1k across {0, 0.5, 1.0}) — the L1
sparsity from the readout pressure subsumes it."*

### Open Questions

Patterns observed 1-2 times, seed-dependent effects, contradictions.

### Falsified Hypotheses

State the original verbatim, the contradicting evidence (iter + metrics),
what was learned, and a revised hypothesis if applicable.

## File Structure

Under `<data_root>/log/Claude_exploration/LLM_drosophila_cx_pi_fc_epg2/`:

1. **`*_Claude_analysis.md`** — append-only full log.
2. **`*_Claude_memory.md`** — working memory; read + update every batch.
3. **`user_input.md`** — read every batch; ack pending items.

### Working Memory template

```markdown
# Working Memory: drosophila_cx_pi_fc_epg2

## Paper Summary (4 sentences, update at every block boundary)
- HP-causality: <single HP explaining most variance>
- Failure mode: <how edge_free / FC breaks when it breaks>
- Robustness: <CV across seeds of the winner>
- Surprise: <unexpected interaction>

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

**Step 1** — Read working memory + user input; identify current parent and
next-batch hypothesis.

**Step 2** — Analyse the previous batch's 10 slots; classify
stable-robust / stable / unstable / catastrophic.

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

**Step 5** — Edit the 10 config files for the next batch.

## Block Boundaries

At every block boundary: (1) update Paper Summary, (2) write one-paragraph
block summary, (3) promote consistent findings to Established Principles,
(4) move falsified hypotheses (with evidence), (5) winning slot becomes the
parent for B+1.

## Block plan (160 iter, 4 blocks × 40 iter, 10 slots × 4 batches)

The spine is **MIXED**: B1 chases the variant's central open question;
B2/B3 sweep HP families; B4 confirms robustness + free exploration.

### B1 — Column-Dale stabilisation (iter 1-40)

**Question**: Which HPs are load-bearing for crossing r_roll_1k ≥ 0.95
*while keeping* `wrec_param: column_dale`?

- B1.1 (iter 1-10): failure-mode deep-dive. All 10 slots
  `wrec_param: column_dale` at the parent recipe with different seeds;
  characterise the *shape* of the failure — collapse, drift,
  polarity-flip, or plateau? Per-epoch trajectory analysis is the
  primary diagnostic.
- B1.2 (iter 11-20): big single-axis sweeps around the parent — one
  axis per slot from {`lr_W_rec_schedule` scaling, `lr_W_ED`, `noise`,
  `coeff_tail_loss`, `grad_clip_W`, `coeff_W_L1`, `w_init_mode`,
  `w_init_scale`, `batch_size`}.
- B1.3 (iter 21-30): combine the two highest-leverage axes from B1.2
  into a 2D grid; pick promising joint regions while keeping each slot
  single-axis vs the *new* parent.
- B1.4 (iter 31-40): **ROBUSTNESS** of the best B1 slot (10 seeds).

Deliverable: a column_dale-compatible recipe that crosses the
r_roll_1k ≥ 0.95 threshold (or, if exhausted, a §Discussion paragraph
documenting which axes were searched and what the best column_dale
ceiling is).

### B2 — Aux losses and sparsity (iter 41-80)

**Question**: Does the connectome-prior infrastructure still help under
edge_free, or does the EPG-only readout already do that work?

Sweeps:
- `coeff_norm_floor` × `kappa_norm_floor`: {(0,0), (0.5,0.05), (1,0.1)}
- `coeff_cos_distance`: {0, 0.5, 1.0, 2.0}
- `coeff_tv_circular`: {0, 1e-3, 1e-2}
- `coeff_W_L1`: {0, 1e-6, 1e-5, 1e-4, 1e-3}
- `w_init_scale`: {1e-3, 1e-2, 5e-2, 1e-1}
- `w_init_mode`: {const, randn, w_con, zeros}

If any aux loss is necessary, §Methods cites it; if all are superfluous,
§Discussion notes that the readout constraint subsumes them.

### B3 — Architecture, encoder/decoder, curriculum (iter 81-120)

**Question**: How much of the FC's behaviour is encoder-decoder vs
recurrent?

Sweeps: `velocity_gate` {pen_4scalar, pen_only, none}, `input_proj` /
`output_proj` {matrix, mlp}, `hidden_dim` / `n_layers` (mlp only),
`MLP_activation`, `batch_size` {32, 64, 128}, `n_steps_schedule` shape,
`data_augmentation_loop`.

### B4 — Robustness confirmation + free exploration (iter 121-160)

- B4.1 ROBUSTNESS: best config from B1-B3, 10 different seeds.
- B4.2-B4.4 free exploration: combine the best findings; single-axis
  vs the new parent each slot.

## Available hyperparameters

All fields the agent may set per-slot. **One axis per slot.**

### Learning rates (three-group optimiser)

| Field                   | Parent       | Sweep                              | What                                                                |
| ----------------------- | ------------ | ---------------------------------- | ------------------------------------------------------------------- |
| `lr`                    | 2e-3         | {1e-3, 2e-3, 4e-3}                 | Biases group; fallback init for `lr_W_rec` / `lr_W_ED`.             |
| `lr_W_rec`              | unset        | {5e-4, 1e-3, 2e-3}                 | Initial lr for `S` (FC recurrent core). Sets schedule[0].           |
| `lr_W_ED`               | 5e-4         | {1e-4, 5e-4, 1e-3, 2e-3}           | Constant lr for `W_in`, `W_out`, velocity-gate scalars.             |
| `lr_W_rec_schedule`     | [2e-3 2e-3 1e-3 5e-4 5e-4] | scale by {0.25, 0.5, 1, 2, 4}      | Per-epoch trajectory of the `w_rec` group only.                     |
| `lr_W_ED_schedule`      | unset        | optional decay or ramp             | Per-epoch trajectory of the `w_ED` group.                           |

### Stabilisers

| Field                   | Parent       | Sweep                              | What                                                                |
| ----------------------- | ------------ | ---------------------------------- | ------------------------------------------------------------------- |
| `noise_recurrent_level` | 0.05         | {0, 0.01, 0.03, 0.05, 0.1}         | Gaussian noise on `h` at every Euler step.                          |
| `grad_clip_W`           | 2.5          | {0, 1, 2.5, 3, 5}                  | Max-norm gradient clip.                                             |
| `coeff_tail_loss`       | **0.05**     | {0, 0.02, 0.035, 0.05, 0.1}        | MSE weight on the rollout tail.                                     |

### Curriculum

| Field                    | Parent              | Sweep                                                  | What                                  |
| ------------------------ | ------------------- | ------------------------------------------------------ | ------------------------------------- |
| `n_steps_schedule`       | [100,200,300,400,500] | {[200×5], [500×5], [100…500], [50,100,200,300,500]} | Per-epoch BPTT horizon.               |
| `n_epochs`               | 5                   | {3, 5, 8} (== len(schedule))                           | Curriculum stages.                    |
| `data_augmentation_loop` | 1                   | {1, 2, 5}                                              | Train-set passes per epoch.           |
| `batch_size`             | 64                  | {32, 64, 128}                                          | Trials per gradient step.             |

### Connectome-prior aux losses

| Field                 | Parent | Sweep                  | What                                                          |
| --------------------- | ------ | ---------------------- | ------------------------------------------------------------- |
| `coeff_cos_distance`  | 0.0    | {0, 0.5, 1, 2}         | Per-block cosine alignment to W_con.                          |
| `coeff_norm_floor`    | **0.5**| {0, 0.5, 1}            | Soft floor on mean \|W\| per type-pair block.                 |
| `kappa_norm_floor`    | **0.05**| {0, 0.05, 0.1}        | Floor target for the norm-floor penalty.                      |
| `coeff_tv_circular`   | 0.0    | {0, 1e-3, 1e-2}        | Circular TV on EPG/PEN ring firing rates.                     |
| `coeff_W_L1`          | 0.0    | {0, 1e-6, 1e-5, 1e-4, 1e-3} | L1 on `S` (synaptic magnitude).                          |

### Initialisation

| Field           | Parent | Sweep                                  | What                                                    |
| --------------- | ------ | -------------------------------------- | ------------------------------------------------------- |
| `w_init_mode`   | const  | {const, randn, w_con, zeros}           | Init template for `S`.                                  |
| `w_init_scale`  | 0.01   | {1e-3, 1e-2, 5e-2, 1e-1}               | Scalar multiplier on `S` at init.                       |

### Architecture

| Field            | Parent        | Sweep                                   | What                                                  |
| ---------------- | ------------- | --------------------------------------- | ----------------------------------------------------- |
| `input_proj`     | matrix        | {matrix, mlp}                           | Encoder shape.                                        |
| `output_proj`    | matrix        | {matrix, mlp}                           | Decoder shape.                                        |
| `velocity_gate`  | pen_4scalar   | {pen_4scalar, pen_only, none}           | Anatomical gate on velocity column of `W_in`.         |
| `hidden_dim`     | 128 (mlp)     | {64, 128, 256}                          | MLP width (mlp only).                                 |
| `n_layers`       | 3 (mlp)       | {2, 3, 4}                               | MLP depth (mlp only).                                 |
| `MLP_activation` | relu          | {relu, tanh, leaky_relu, soft_relu}     | MLP nonlinearity.                                     |

## Mutation guardrails — DO NOT change

- `signal_model_name`, `aggr_type`, `lock_edge_signs`,
  `output_from_epg_only` (variant identity).
- **`wrec_param` is PINNED to `column_dale`.** Mutating to `edge_free`
  or `edge_magnitude` is forbidden — the scientific value of this
  exploration depends on staying Dale-faithful. Relaxing the
  parameterisation would trivially saturate the metric and invalidate
  every other finding.
- `simulation:` and `task.path_integration:` blocks.
- `dataset` (must stay `drosophila_cx_pi_task`).
- Seeds (pipeline-controlled).

## Final Summary

At exploration completion (after B4), write **two** outputs:

1. **Per-loop**: `<exploration_dir>/drosophila_cx_pi_fc_epg2_summary.md`.
2. **Shared**: append to
   `/home/node/.claude/projects/-workspace--devcontainer/memory/exploration_results.md`
   under `## drosophila_cx_pi_fc_epg2 — Key Discoveries (YYYY-MM-DD)`.

Eight bullets, **knowledge first, metric last**:

1. **Column-Dale recipe** — the HP combination that lets FC + EPG-only
   readout cross r_roll_1k ≥ 0.95 *under* `wrec_param: column_dale`,
   or, if exhausted, the documented ceiling and the axes searched.
   (§Methods + §Discussion.)
2. **HP causality** — the single HP explaining the most variance in
   r_roll_1k under column_dale, with explained fraction.
3. **Failure mode** — how the recipe breaks when it breaks; per-epoch
   trajectory shape; which metric flips first.
4. **Surprise** — unexpected HP interaction.
5. **Falsified hypothesis** — what we expected, what we got, what we
   learned.
6. **Aux-loss story** — whether `coeff_norm_floor` / `kappa_norm_floor`
   / `coeff_cos_distance` are necessary under column_dale or are
   superseded by the EPG-only readout's implicit constraint.
7. **Robustness numbers** — winner r_roll_1k mean ± std (10 seeds), CV%,
   catastrophic count.
8. **Best metric** (output, not goal) — best r_roll_1k + the recipe.
