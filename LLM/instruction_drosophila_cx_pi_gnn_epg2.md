# Drosophila CX — Path Integration (GNN, EPG-only readout) — Knowledge Exploration

## Goal

**The behavioural metric was reached on the first attempt.** In the
4-job EPG-only readout benchmark, the GNN converged at iter 25001 ep 5
(T=300) with `r_roll_1k = 0.99999`, rmse = 6.5°. No agentic loop was
needed; the parent recipe at
[config/drosophila_cx/drosophila_cx_pi_gnn_epg.yaml](../config/drosophila_cx/drosophila_cx_pi_gnn_epg.yaml)
already saturates the metric.

**This exploration is about understanding, not goal-finding.** Deliverables
for the paper:

- **Why the GNN converges so much faster** than the connectome-locked RNN
  or the FC under the same EPG-only readout. The hypothesis is that the
  learnable `f_θ` / `g_φ` MLPs let the optimiser bypass the
  Hulse-imposed sigmoid/linear-leak constraint and find a "shorter path"
  to the bump-attractor. Quantify which MLP component does the work.
- **Architecture sensitivity**: how do `embedding_dim`, MLP widths/depths,
  `g_phi_positive`, `coeff_f_theta_diff` shape the basin?
- **Failure modes**: when the GNN does fail, *how* does it fail (collapse
  to constant output, vanishing message, exploding `a`-embeddings)?
- **Seed robustness**: confirm the 0.9999 across 10 seeds for the paper.

## Starting Point

- Parent yaml: `config/drosophila_cx/drosophila_cx_pi_gnn_epg.yaml` (the
  paper-recipe-equivalent GNN config with EPG-only readout enabled).
- Iter 0 known floor: r_roll_1k ≈ 0.99999, rmse_roll ≈ 6.5° at T=300 ep 5.
- Dataset (fixed): `drosophila_cx_pi_task`.

## Why the agentic loop is primordial

The HP space here is high-dimensional; a single bad config lands in a
non-converging pocket routinely. **Any "X fails" / "Y is a fundamental
limit" claim in the final summary must be qualified by the breadth of
agentic search that produced it** — iter count + axes covered + whether
single-axis or joint sweeps were exhausted. Without the loop, a
non-converging single config is *unsearched*, not *impossible*. For the
GNN this matters most when probing knock-down ablations (e.g.
`embedding_dim=1`, `coeff_f_theta_diff=0`): "ablation X fails" is
load-bearing only if the rest of the HP basin was systematically
searched around X.

## Scientific Method

Strict **hypothesize → test → validate/falsify** cycle:

1. Write the hypothesis BEFORE running the slot.
2. Change EXACTLY ONE parameter per slot (causality rule).
3. Run 10 slots — 1 control + 9 single-axis mutations in EXPLORATION mode;
   all 10 identical (different seeds) in ROBUSTNESS mode.
4. Read `r_roll_1k`, `mse`, `fwhm_deg`, per-epoch trajectory,
   `collapse_detected`. Classify each slot:
   stable-robust / stable / unstable / catastrophic.
5. Revise hypotheses; promote findings per the KB thresholds below.

Only training results validate or falsify.

### Causality rule (MANDATORY)

If a slot mutates more than one parameter, the effect cannot be attributed —
fatal design error. **EXPLORATION**: slot 0 = parent control, slots 1–9
each change exactly one axis. **ROBUSTNESS**: all 10 slots same config
(pipeline forces different seeds).

Do not stack two new GNN axes (e.g. `embedding_dim` and `coeff_f_theta_diff`)
in one slot. Test each in its own slot at the *current* parent.

## Knowledge Base (in working memory)

### Established Principles

Promote a finding only if ALL of: (a) observed in ≥ 3 iterations, (b) low
CV across the 10 seeds, no catastrophic failures, (c) states a **causal**
relationship. Example: *"`embedding_dim ≥ 4` is necessary — below 4,
r_roll_1k stays < 0.8 across all 10 seeds; the latent must separate the
156 cell-type identities the edge function distinguishes."*

### Open Questions

Patterns observed 1-2 times, seed-dependent effects, contradictions.

### Falsified Hypotheses

State the original verbatim, the contradicting evidence (iter + metrics),
what was learned, and a revised hypothesis if applicable.

## File Structure

Under `<data_root>/log/Claude_exploration/LLM_drosophila_cx_pi_gnn_epg2/`:

1. **`*_Claude_analysis.md`** — append-only full log.
2. **`*_Claude_memory.md`** — working memory; read + update every batch.
3. **`user_input.md`** — read every batch; ack pending items.

### Working Memory template

```markdown
# Working Memory: drosophila_cx_pi_gnn_epg2

## Paper Summary (4 sentences, update at every block boundary)
- HP-causality: <which HP/MLP-component explains most variance>
- Failure mode: <how the GNN breaks when it breaks>
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

### B1 — Which MLP component does the work? (iter 1-40)

**Question**: The GNN converges 2× faster than the connectome-locked RNN
under EPG-only readout. Which MLP component (`f_θ` node update,
`g_φ` edge message, per-node embedding `a`) is doing the work?

- B1.1 (iter 1-10): **`embedding_dim` knock-down**. {1, 2, 4, 8, 16}.
  Below 4 should fail (the embedding must separate cell-type identities);
  above 4 should saturate. Locate the cliff.
- B1.2 (iter 11-20): **`g_phi_positive`** {true, false} ablation +
  `coeff_g_phi_diff` sweep {0, 5, 15, 50}. Probe what happens when the
  edge MLP is unconstrained vs Dale-conformant.
- B1.3 (iter 21-30): **`coeff_f_theta_diff`** sweep {0, 5, 15, 50, 100}
  + MLP depth (`n_layers` / `n_layers_update`) {1, 2, 3}. Probe what
  happens when the leak-monotonicity prior on f_θ is relaxed.
- B1.4 (iter 31-40): **ROBUSTNESS** at the iter-0 parent (10 seeds).

Deliverable: a per-component contribution analysis for §Discussion.
Quantify how much each MLP element (embedding / g_φ / f_θ) explains.

### B2 — Initialisation, sparsity, and `W` magnitude (iter 41-80)

**Question**: The GNN uses `w_init_mode: zeros` at init (recurrence is
feedforward until |W| crosses the bifurcation). How does this initial
inactivity shape convergence?

Sweeps:
- `w_init_mode`: {zeros, const, randn, w_con}
- `w_init_scale`: {1e-3, 1e-2, 5e-2, 1e-1}
- `coeff_W_L1`: {0, 1e-6, 1e-5, 1e-4, 1e-3}
- `coeff_norm_floor` / `kappa_norm_floor` (mostly inactive under
  edge_free GNN; verify or refute)
- `coeff_tv_circular`: {0, 1e-3, 1e-2}
- `coeff_cos_distance`: {0, 0.5, 1.0}

### B3 — Stabilisers, curriculum, and encoder/decoder (iter 81-120)

**Question**: How much of the GNN's fast convergence is in the
curriculum (T=10→1000 ramp) vs the architecture?

Sweeps:
- `n_steps_schedule`: {[10×5], [10,50,100,200,300], [100,200,300,400,500],
   [10,100,300,500,1000]} — the schedule shape was a primary axis for the
   GNN winner.
- `n_epochs` / `data_augmentation_loop`: {3, 5, 8} × {1, 2, 5}
- `noise_recurrent_level`: {0, 0.01, 0.025, 0.05}
- `grad_clip_W`: {0, 1, 2.5, 5}
- `coeff_tail_loss`: {0, 0.02, 0.05, 0.1}
- `velocity_gate`: {pen_4scalar, pen_only, none}
- `batch_size`: {8, 16, 32}
- `MLP_activation` (g_φ / f_θ): {tanh, relu, gelu, soft_relu}

### B4 — Robustness confirmation + free exploration (iter 121-160)

- B4.1 ROBUSTNESS: best config from B1-B3, 10 different seeds.
- B4.2-B4.4 free exploration: combine the best findings; single-axis
  vs the new parent each slot.

## Available hyperparameters

All fields the agent may set per-slot. **One axis per slot.**

### Learning rates (three-group optimiser)

| Field                   | Parent       | Sweep                              | What                                                                |
| ----------------------- | ------------ | ---------------------------------- | ------------------------------------------------------------------- |
| `lr`                    | 2e-3         | {1e-3, 2e-3, 4e-3}                 | Biases group; fallback init.                                        |
| `lr_W_rec`              | unset        | {5e-4, 1e-3, 2e-3}                 | Initial lr for the `w_rec` group (`W` + `a` + `g_phi.*` + `f_theta.*`). |
| `lr_W_ED`               | 5e-4         | {1e-4, 5e-4, 1e-3, 2e-3}           | Constant lr for `W_in`, `W_out`, velocity-gate scalars.             |
| `lr_W_rec_schedule`     | [2e-3 2e-3 2e-3 1e-3 5e-4 4e-4 3e-4 2e-4 5e-5 5e-5] | scale by {0.25, 0.5, 1, 2, 4} | Per-epoch trajectory of `w_rec`.                                |
| `lr_W_ED_schedule`      | unset        | optional decay or ramp             | Per-epoch trajectory of `w_ED`.                                     |

### GNN MLPs and embedding (the central axes of this variant)

| Field                   | Parent       | Sweep                              | What                                                                |
| ----------------------- | ------------ | ---------------------------------- | ------------------------------------------------------------------- |
| `embedding_dim`         | 4            | {1, 2, 4, 8, 16}                   | Per-neuron learnable embedding `a_i` dim (concatenated to node + edge MLP inputs). |
| `g_phi_positive`        | true         | {true, false}                      | When true, clip `g_phi` output to ≥0 (Dale's-law approximation).    |
| `coeff_g_phi_diff`      | 15           | {0, 5, 15, 50}                     | L2 penalty driving `g_phi` away from trivial constant.              |
| `coeff_f_theta_diff`    | 15           | {0, 5, 15, 50, 100}                | L2 penalty driving `f_theta` away from trivial constant + enforces leak monotonicity. |
| `hidden_dim` (g_phi)    | 64           | {32, 64, 128}                      | g_φ MLP width.                                                      |
| `n_layers` (g_phi)      | 2            | {1, 2, 3}                          | g_φ MLP depth.                                                      |
| `hidden_dim_update`     | 64           | {32, 64, 128}                      | f_θ MLP width.                                                      |
| `n_layers_update`       | 2            | {1, 2, 3}                          | f_θ MLP depth.                                                      |
| `MLP_activation`        | tanh         | {tanh, relu, gelu, soft_relu, leaky_relu} | g_φ / f_θ nonlinearity.                                       |

### Stabilisers

| Field                   | Parent       | Sweep                              | What                                                                |
| ----------------------- | ------------ | ---------------------------------- | ------------------------------------------------------------------- |
| `noise_recurrent_level` | 0.01         | {0, 0.01, 0.025, 0.05, 0.1}        | Gaussian noise on `h` at every Euler step.                          |
| `grad_clip_W`           | 1.0          | {0, 1, 2.5, 5}                     | Max-norm gradient clip.                                             |
| `coeff_tail_loss`       | **0.05**     | {0, 0.02, 0.035, 0.05, 0.1}        | MSE weight on the rollout tail.                                     |

### Curriculum

| Field                    | Parent              | Sweep                                                  | What                                  |
| ------------------------ | ------------------- | ------------------------------------------------------ | ------------------------------------- |
| `n_steps_schedule`       | [10,50,100,200,300] | {[10×5], [10,50,100,200,300], [100,200,300,400,500], [10,100,300,500,1000]} | Per-epoch BPTT horizon.    |
| `n_epochs`               | 5                   | {3, 5, 8}                                              | Curriculum stages.                    |
| `data_augmentation_loop` | 1                   | {1, 2, 5}                                              | Train-set passes per epoch.           |
| `batch_size`             | 16                  | {8, 16, 32}                                            | Trials per gradient step.             |

### Connectome-prior aux losses

| Field                 | Parent | Sweep                  | What                                                          |
| --------------------- | ------ | ---------------------- | ------------------------------------------------------------- |
| `coeff_cos_distance`  | 0.0    | {0, 0.5, 1, 2}         | Per-block cosine alignment to W_con.                          |
| `coeff_norm_floor`    | 0.0    | {0, 0.5, 1}            | Soft floor on mean \|W\| per type-pair block.                 |
| `kappa_norm_floor`    | 0.0    | {0, 0.05, 0.1}         | Floor target for the norm-floor penalty.                      |
| `coeff_tv_circular`   | 0.0    | {0, 1e-3, 1e-2}        | Circular TV on EPG/PEN ring firing rates.                     |
| `coeff_W_L1`          | 0.0    | {0, 1e-6, 1e-5, 1e-4, 1e-3} | L1 on per-edge `W` magnitude.                            |

### Initialisation

| Field           | Parent | Sweep                                  | What                                                    |
| --------------- | ------ | -------------------------------------- | ------------------------------------------------------- |
| `w_init_mode`   | **zeros** | {zeros, const, randn, w_con}        | Init template for the per-edge `W`. zeros = recurrence is feedforward until |W| crosses the bifurcation. |
| `w_init_scale`  | 0.01   | {1e-3, 1e-2, 5e-2, 1e-1}               | Scalar multiplier at init.                              |

### Encoder / decoder

| Field            | Parent      | Sweep                                   | What                                                  |
| ---------------- | ----------- | --------------------------------------- | ----------------------------------------------------- |
| `input_proj`     | matrix      | {matrix, mlp}                           | Encoder shape.                                        |
| `output_proj`    | matrix      | {matrix, mlp}                           | Decoder shape.                                        |
| `velocity_gate`  | pen_4scalar | {pen_4scalar, pen_only, none}           | Anatomical gate on velocity column of `W_in`.         |

## Mutation guardrails — DO NOT change

- `signal_model_name` (must stay `drosophila_cx_pi_gnn`).
- `aggr_type`, `lock_edge_signs`, `output_from_epg_only` (variant identity).
- `simulation:` and `task.path_integration:` blocks.
- `dataset` (must stay `drosophila_cx_pi_task`).
- Seeds (pipeline-controlled).

## Final Summary

At exploration completion (after B4), write **two** outputs:

1. **Per-loop**: `<exploration_dir>/drosophila_cx_pi_gnn_epg2_summary.md`.
2. **Shared**: append to
   `/home/node/.claude/projects/-workspace--devcontainer/memory/exploration_results.md`
   under `## drosophila_cx_pi_gnn_epg2 — Key Discoveries (YYYY-MM-DD)`.

Eight bullets, **knowledge first, metric last**:

1. **MLP-component contribution** — quantify how much of the
   convergence-speed advantage over the Known-ODE RNN comes from
   `embedding_dim`, `g_phi`, `f_theta` respectively. (§Discussion.)
2. **HP causality** — the single HP explaining the most variance in
   r_roll_1k under the GNN, with explained fraction.
3. **Failure mode** — how the GNN breaks when it breaks; per-epoch
   trajectory shape; which metric / MLP component flips first.
4. **Surprise** — unexpected interaction (e.g. `g_phi_positive: false`
   succeeding, large `embedding_dim` regressing, etc.).
5. **Falsified hypothesis** — what we expected, what we got, what we
   learned about the GNN's mechanism.
6. **`w_init_mode: zeros` story** — does the feedforward-until-bifurcation
   regime really do the work, or is it a red herring?
7. **Robustness numbers** — winner r_roll_1k mean ± std (10 seeds), CV%,
   catastrophic count.
8. **Best metric** (output, not goal) — best r_roll_1k + the recipe.
