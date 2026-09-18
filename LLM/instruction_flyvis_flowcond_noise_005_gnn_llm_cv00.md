# Conductance GNN on task-trained conductance data — flyvis optic flow, sigma = 0.05

## Goal

Fit a `flyvis_conductance` GNN to voltage traces generated **by a conductance model that was
itself trained on the optic-flow task**, and recover the generator's parameters from the
learned message. Nothing about the synapse is assumed: `W_ij` is a free per-edge weight, and
the driving force, the rectifier and the conductance all live inside one learned surface
`g_phi(v_j, a_j, v_i, a_i)`.

This dataset is not a teacher–student twin. Every earlier conductance result came from a
conductance model fitted to reproduce a *current-based* model's activity, which made the two
synapse families hard to separate because the data was built to be explainable by the current
form. Here the task, not another model, decided what the generator does — so a driving force
`(E_i - v_i)` that shows up in the readout is a property of the data.

**THE WORKING POINT IS `flyvis_flowcond_noise_005_gnn_llm_cv00`**, which is the `s25g1e5` arm
of the 22-arm sweep: `coeff_g_phi_silent: 25`, `coeff_f_theta_msg_gain: 1.0e-05`, and
**neither factor squared** (`w_squared: false`, `g_phi_positive: false`). It took the highest
`Wij_R2` peak of that sweep, 0.483, reproducibly across three separate launches.

**Primary metric: the PEAK of `Wij_R2` over the run, not its final value.** Gating metric:
`Eij_gate`. Tie-break: `Eij_pct_wrong_slope`. Ungated sanity metric: `msg_i_R2`. Read
"the gate" and "what the truth looks like now" below before trusting any recovery number.

---

## The model, exactly

`NeuralGNN` with `signal_model_name: flyvis_conductance`, in
`src/connectome_gnn/models/neural_gnn.py`:

```
in_features = [ v_j , a_j , v_i , a_i ]          # column 0 is ALWAYS v_j, in every family
g_phi_out   = g_phi(in_features)                 # 3-layer MLP, hidden 80, output 1
if g_phi_positive:  g_phi_out = g_phi_out ** 2   # FALSE here
W_edge      = W[edge] ** 2  if w_squared  else  W[edge]   # w_squared FALSE here
msg_i       = sum over incoming edges of  W_edge * g_phi_out
dv_i/dt     = f_theta(v_i, a_i, msg_i, excitation)
```

So the message is simply `msg = W * g_phi`, both factors free to take any sign.

### Where the sign of a synapse lives

The generator's message is `W_ij * relu(v_j) * (E_i - v_i)` with **`W_ij >= 0`**, so the
polarity of a synapse is carried entirely by the sign of `(E_i - v_i)`. The GNN has no such
factor, so the sign must live in `W`, in `g_phi`, or split between them. With both squaring
flags off nothing pins either, which is what lets `g_phi` represent the driving force — and
also what leaves the `W`/`g_phi` sign split unidentified.

**The readout therefore reports `|W|` for conductance runs**, because the generator's
conductance is non-negative and a signed `W` is not comparable to it. This is not a
convention you may change; `metrics.py` applies `np.abs` and the `Wij` scatter plots it.

---

## The template readout, and what it is measuring

Everything reported comes from the two-form template readout documented in
`docs/template_readout.tex`. The chain-based legacy readout is disabled and a fallback to it
is fatal (`require_template_readout`). Two fits:

**Per edge**, a three-column least squares on `[u, u*v_i, 1]` with `u = relu(v_j)` taken from
the generator's own activation:

```
msg_ij = W*u*(E - v_i) + C = (W E)*u + (-W)*(u v_i) + C
  ->  W = -b2,  E = -b1/b2,  C = b3
```

**Per postsynaptic neuron**, the update fit on `[1, v_i, msg, stim]` giving
`(alpha_0, alpha_1, alpha_2, alpha_3)`, from which `tau_i = -1/alpha_1`,
`k_i = tau_i*alpha_2`, `V_rest_i = tau_i*(alpha_0 + alpha_2*beta_i)`.

`W` is then reported in the generator's units as `W_corrected = k_i * W_fit`, one `k_i` per
postsynaptic neuron applied to all its incoming edges (median 19).

**The gauge is not the bottleneck and you should not spend a block on it.** Measured on the
300k checkpoint of the baseline: a split-half test that refits one scale per neuron on half
that neuron's edges and scores the other half gives R2 -0.32, against +0.29 for the readout's
own `k_i` on the same held-out edges. Substituting the generator's TRUE tau into `k_i` makes
`Wij_R2` worse, 0.214 against 0.358, because the model has learned a self-consistent
`(tau, W)` pair in its own units. The ceiling lives in the per-edge fit.

---

## THE GATE — why a recovery number here can be fiction

`E = -b1/b2` divides by the second coefficient. On an edge whose postsynaptic voltage barely
moved, `b2` is a small noisy number and `E` is its reciprocal; a handful of those produced
`|E|` in the thousands while the fit itself sat at R2 0.999. The readout gates on the t
statistic of `b2` and reports no reversal rather than a ratio of two small numbers.

- A slot whose final `Eij_gate` is below 0.9 has no valid `Eij_*` and no valid extracted `W`.
  Record `Wij_R2`, `msg_i_R2` and the trajectory metrics for it, mark the `Eij_*` numbers
  "gated out", and do not rank it on them.
- `msg_i_R2` is never gated out. It is how you tell a slot that is still learning from one
  that is stuck.

---

## NEVER WRITE A NUMBER YOU DID NOT READ FROM THIS RUN

Every number in your analysis entry must be read from a file this iteration wrote. Do not
carry a number forward from a previous block, do not round one from memory, and do not infer
one from a figure. If a metric is missing, write "not written" — that is information. A
fabricated number is worse than a gap because it survives into the comparison table and is
indistinguishable there from a measured one.

---

## What the truth looks like now

These are measured on this dataset, on the `nosq` family, and they bound what you should
expect. Quote them as context, never as this iteration's result.

| quantity | value | where |
| --- | --- | --- |
| best `Wij_R2` peak seen | 0.483 (`s25g1e5`), reproduced across three launches | 22-arm sweep |
| baseline `Wij_R2` peak | 0.399 | `cv00` |
| `Wij_R2` peaks at | iteration 16k–24k, then decays | baseline trajectory |
| `tau_R2` plateaus at | iteration 160k, around 0.87 | baseline trajectory |
| `V_rest_R2` plateaus at | iteration 120k, around 0.60 | baseline trajectory |
| `Eij_pct_wrong_slope` | 31.6% best, 40% baseline, 44% worst | 22-arm sweep |
| `rollout_r` | **0.990–0.996 on every non-diverged arm** | after the noise-free-twin fix |
| `msg_i_R2` | 0.94 with the gauge terms, 0.47 baseline, 0.11 under heavy W L1 | 22-arm sweep |

Four facts that decide how you spend slots:

1. **`rollout_r` is saturated and is NOT a ranking metric.** A checkpoint with `Wij_R2` 0.35
   rolls out at 0.994. Use it only as a divergence guard: `rollout_r < 0.9` means the model
   diverged and the slot is disqualified whatever else it reports.
2. **The two gauge terms are the only lever that has ever worked.** `coeff_g_phi_silent: 25`
   took the top peak at all three `coeff_f_theta_msg_gain` levels, cut wrong-slope from 40% to
   31%, and lifted `msg_i_R2` from 0.47 to 0.94. 25 was the TOP of that grid, so the optimum
   may be beyond it.
3. **`coeff_W_L1` and `coeff_W_L2` are dead.** Eight arms spanned peaks 0.302–0.408 around the
   baseline's 0.399 while `msg_i_R2` collapsed to 0.11–0.68. Do not spend a block on them.
4. **Late collapse is real and the final value lies about it.** `s5g1e3` peaked at 0.270 and
   ended at -0.082; `s5g1e5` peaked at 0.438 and ended at 0.242. Rank on the peak and
   disqualify on the ratio.

---

## Metrics — the exact names

Read these from the analysis log and from `tmp_training/<key>.log`. Each log's first line is a
`#` descriptor naming the readout that produced it; if that descriptor does not say
`readout=template`, stop and report it rather than reading the numbers.

| file | columns you must read |
| --- | --- |
| `tmp_training/Wij.log` | `Wij_R2` (outlier-free, the headline), `Wij_R2_all`, `Wij_gain`, `Wij_pearson`, `Wij_pct_outliers` |
| `tmp_training/Eij.log` | `Eij_R2`, `Eij_pct_wrong_slope`, `msg_form_r2_median`, `conductance_form_r2_median`, `current_form_r2_median` |
| `tmp_training/msg_i.log` | `msg_i_R2`, `msg_i_gain` |
| `tmp_training/tau.log` | `tau_R2` |
| `tmp_training/V_rest.log` | `V_rest_R2` |
| `tmp_training/rollout.log` | `rollout_r` (per-neuron Fisher-pooled), `rollout_r_pooled`, `rollout_n_diverged` |

`Wij_R2` is the outlier-trimmed number (threshold `|W_learned - W_true| > 1.0`) and
`Wij_R2_all` is not; quote `Wij_R2`. `rollout_r` is the per-neuron Pearson pooled in Fisher-z
space with diverged and flat neurons scored zero; `rollout_r_pooled` flattens all
(frame, neuron) pairs and rewards getting each cell's mean LEVEL right, so a large gap between
them means flat predictions. Report both; rank on neither.

The two form columns are a genuine diagnostic. `conductance_form_r2_median` has exceeded
`current_form_r2_median` at every iteration of every arm measured so far, by 0.007 to 0.038.
A LARGER gap is not better: across arms the relationship inverts, with the biggest gaps
(0.038–0.044) belonging to the arms with the worst wrong-slope. Without a silent anchor the
fit absorbs a message pedestal by tilting the driving force, which registers as `v_i`
dependence the current form cannot match.

---

## Explorable parameters

YOU MAY MODIFY ONLY THE PARAMETERS IN THIS TABLE.

| Parameter | Default | Suggested sweep | Notes |
| --- | --- | --- | --- |
| `coeff_g_phi_silent` | 25 | {10, 25, 50, 100, 200} | the silent anchor, the strongest known lever. 25 was the TOP of the previous grid and won there, so push past it |
| `g_phi_silent_range` | [-2.0, 0.0] | {[-2,0], [-3,-0.5], [-1,0], [-4,0]} | the presynaptic voltage window treated as silent |
| `coeff_f_theta_msg_gain` | 1.0e-05 | {1e-6, 1e-5, 1e-4, 1e-3} | drives `k_i -> 1`. Charged the gauge without needing tau |
| `lr_W` | 0.0009 | {3e-4, 6e-4, 9e-4, 2e-3} | learning rate on `W` |
| `lr` | 0.0018 | {6e-4, 1.2e-3, 1.8e-3, 4e-3} | learning rate on the g_phi and f_theta MLP weights |
| `lr_embedding` | 0.002325 | {1e-3, 2.3e-3, 5e-3} | learning rate on the per-neuron embeddings `a` |
| `w_init_scale` | 1.0 | {1, 5, 20, 60} | bound = `scale / sqrt(n_edges)`. Untested on this dataset and implicated by `Wij_gain` sitting at 0.42–0.61 |
| `w_init_mode` | `randn_scaled` | `randn_scaled`, `uniform_scaled`, `zeros` | lowercase |
| `batch_size` | 4 | {2, 4, 8, 16} | INTEGER. Interacts with `regul_batch_scaling: sqrt`; check that the regulariser/fit ratio held before reading a result |
| `hidden_dim` | 80 | {64, 80, 128} | g_phi and f_theta width |
| `embedding_dim` | 2 | {2, 4} | changing it changes `input_size` to `2 + 2*embedding_dim`; the group lasso's column slices index against that layout |
| `coeff_g_phi_weight_L1` | 0 | {0, 0.01, 0.05} | L1 on the g_phi MLP weights. **Values >= 0.1 collapse training at flyvis scale** |
| `coeff_g_phi_weight_L2` | 0 | {0, 1e-4, 1e-3} | L2 on the g_phi MLP weights |
| `coeff_f_theta_weight_L1` | 0.025 | {0, 0.025, 0.1} | L1 on the f_theta MLP weights |
| `coeff_f_theta_weight_L2` | 0.0005 | {0, 5e-4, 5e-3} | L2 on the f_theta MLP weights |
| `coeff_g_phi_input_group_L1` | 0 | {0, 0.25, 1} | group lasso over g_phi's input columns |
| `data_augmentation_loop` | 125 | 60–250 | retune only against the wall-clock target; do not hard-code a threshold |

**`regul_annealing_rate` MUST STAY 0.0.** The annealed coefficient is
`coeff * (1 - exp(-rate * epoch))`, exactly zero at epoch 0. Under any non-zero rate every
annealed coefficient you set would be identically zero through the first epoch — you would be
sweeping a number that never reaches the loss.

---

## Frozen — do not touch

- `coeff_W_L1`, `coeff_W_L2` — eight arms of the previous sweep established these do nothing
  for `Wij_R2` and damage `msg_i_R2`. Leave them at the parent's values.
- `integration_method`, `n_rollout_substeps` — the substep arms need 35–70 hours per run at
  this iteration budget and belong on a separate track.
- `coeff_g_phi_diff`, `coeff_g_phi_norm` — both 0 and they stay 0. `coeff_g_phi_diff` is a
  positive-monotonicity prior on `d g_phi / d v_j` that fights the driving force; at its
  inherited value of 375 it held `Wij_gain` at 0.075.
- **`g_phi_positive` and `w_squared`** — this exploration is defined by both being FALSE.
  Flipping either changes what `W` means and makes every earlier iteration incomparable.
- `simulation.seed`, `training.seed` — overwritten every batch with `iteration*1000 + slot`
  and `iteration*1000 + slot + 500`. Log the values the prompt reports.
- `training.n_epochs` — overwritten from `claude.n_epochs` every batch. Editing it is a silent
  no-op, so never dedicate a block to "training volume via n_epochs".
- `dataset` — must stay `flyvis_flowcond_noise_005_blank50_cv00` in every slot. Its noise-free
  twin `flyvis_flowcond_noise_free_blank50_cv00` must exist, because the rollout is scored
  against it.
- `simulation.*` — `n_neurons`, `n_edges`, `n_frames`, `delta_t`, `noise_model_level`.
- `input_size`, `signal_model_name`, `prediction`, `aggr_type`, `update_type`,
  `use_gt_edges: true`, `mlp_precision`, `torch_compile`.

---

## Block structure

`n_iter_block` is 24 and `n_parallel` is 8, so a block is three batches. The block number and
a `>>> BLOCK END <<<` marker are injected into your prompt.

| Block | Mode | Focus | Parameters | Ranges |
| --- | --- | --- | --- | --- |
| 1 | Robustness | Baseline variance | none — all 8 slots identical | Establish the CV of peak `Wij_R2`, `tau_R2`, `V_rest_R2`, `msg_i_R2` and the gate's trajectory. NOTHING LATER MEANS ANYTHING WITHOUT THIS: a later gain smaller than this spread is not a gain |
| 2 | Exploration | **The silent anchor** | `coeff_g_phi_silent`, `g_phi_silent_range` | silent {10, 25, 50, 100, 200}; range {[-2,0], [-3,-0.5], [-1,0], [-4,0]}. This is the highest-value block in the file — 25 won at the top of the previous grid, so the first question is whether more is better |
| 3 | Exploration | **The message gain** | `coeff_f_theta_msg_gain` | {1e-6, 1e-5, 1e-4, 1e-3} crossed with the winner of block 2. Report `Wij_gain` beside every result; the gain is what this coefficient is supposed to move |
| 4 | Exploration | **Learning rates** | `lr_W`, `lr`, `lr_embedding` | as in the parameter table. Record the `lr_W`/`lr` ratio — it, not either rate alone, is what moved earlier explorations |
| 5 | Exploration | **W initialisation** | `w_init_scale`, `w_init_mode` | scale {1, 5, 20, 60}, mode {randn_scaled, uniform_scaled, zeros}. `Wij_gain` sits at 0.42–0.61, i.e. the learned W is systematically small; this block asks whether the initialisation is why |
| 6 | Exploration | **Capacity and batch** | `hidden_dim`, `embedding_dim`, `batch_size`, `data_augmentation_loop` | hold the wall-clock target by moving DAL inversely with batch size; report achieved minutes per iteration beside every result |
| 7 | Combine + validate | Champion and its variance | any of the above, then none | First two batches consolidate the best of blocks 2–6, ONE change per slot. THE LAST BATCH IS A ROBUSTNESS TEST: all 8 slots at the champion with different seeds, to confirm the CV and that no seed is catastrophic |

**THE ORDER IS DELIBERATE AND IS NOT YOURS TO REORDER.** The silent anchor comes first
because it is the only term with a measured effect on this dataset, and every later block is
explored on top of whatever it settles.

---

## Acceptance

| Verdict | Rule |
| --- | --- |
| Stable-Robust | all 8 slots peak `Wij_R2` >= 0.50 and CV < 5% |
| Stable | mean peak `Wij_R2` >= 0.45, CV < 12% |
| Unstable | mean peak < 0.45 or CV >= 12% |
| Catastrophic | any slot peak `Wij_R2` < 0.20 — reject, do not pursue |

The thresholds are set against the 0.483 best and 0.399 baseline established on this dataset.
Revise them at a block boundary only if block 1's measured variance says they are wrong, and
say so explicitly in memory when you do.

Four rules specific to this experiment:

- **Rank on the peak.** `Wij_R2` peaks at iteration 16k–24k and decays. The final value is a
  different quantity and two arms with the same peak can end 0.4 apart.
- **Disqualify on `final / peak < 0.85`** as Disqualified-late-collapse. Record both numbers
  and the iteration of the peak for every slot.
- **Gate first.** A slot with `Eij_gate < 0.9` has no valid `Eij_*` or extracted `W`.
- **`rollout_r < 0.9` is a divergence, not a ranking.** Everything non-diverged sits at
  0.990–0.996, so this column separates broken from working and nothing finer.

---

## Iteration workflow

For each slot, in this order:

1. Read the `#` descriptor line of `Wij.log` and `Eij.log` first, and confirm it says
   `readout=template`. Then read the last row of `Eij.log` — it decides whether the `Eij_*`
   and extracted-`W` numbers mean anything.
2. Read the WHOLE of `Wij.log`, not only its last row: you need the peak, the iteration at
   which it occurred, and the final value.
3. Read the analysis log and extract every metric named in the Metrics table, then `msg_i.log`,
   `tau.log`, `V_rest.log` and `rollout.log`.
4. Compare peak `Wij_R2` against `msg_i_R2` and say which way they disagree, if they do.
   `Wij/comparison_*.png` and `msgi/msgi_*.png` show the two as scatters, and
   `function/g_phi/func_*.png` says whether the learned edge function has the sign-changing
   shape in `v_i` that `(E_i - v_i)` requires.
5. Write one entry per slot to the analysis log **and** to memory. The heading must be exactly
   `## Iter N: <short title>` — the resume mechanism parses that pattern and a different
   heading breaks `--resume`.
6. Edit all slot configs for the next batch under the causality rule in your prompt: slot 0 is
   the parent unchanged, every other slot changes exactly one parameter.

Entry template:

```markdown
## Iter N: <one-line hypothesis>
- Slot: S | seeds: sim=<value> train=<value>
- Changed: <parameter> <old> -> <new>   (slot 0: unchanged control)
- GATE Eij_gate: <value>  -> recovery numbers <valid | gated out>
- Wij_R2 PEAK: <value> at iter <value> | final <value> | final/peak <value>
- Wij_gain: <value>   Wij_pearson: <value>   Wij_pct_outliers: <value>
- Eij_R2: <value>   Eij_pct_wrong_slope: <value>   [gated out if Eij_gate < 0.9]
- cond_form_r2 / cur_form_r2: <value> / <value>   gap <value>
- msg_i_R2: <value>   msg_i_gain: <value>         [never gated]
- tau_R2: <value>   V_rest_R2: <value>
- rollout_r: <value>   rollout_r_pooled: <value>   n_diverged: <value>
- training_time_min: <value>
- Verdict: <Stable-Robust | Stable | Unstable | Catastrophic | Gated-out | Diverged | Disqualified-late-collapse>
- Reading: <two sentences: what moved, and whether the gate licensed the reading>
```

---

## Block boundaries

At `>>> BLOCK END <<<`:

1. Write a block summary into `## Previous Block Summaries`.
2. Promote findings that survived a robustness check to `### Established Principles`; move
   refuted ones to `### Falsified Hypotheses` with the evidence that killed them.
3. Update the comparison table with every metric column, including `Eij_gate` and final/peak.
4. Write the block's winner to
   `config/fly/flyvis_flowcond_noise_005_gnn_llm_winner.yaml` with a comment header giving the
   iteration and the headline metrics. `config/fly/` is the only config directory that exists.
5. State the next block's hypothesis in `## Current Block`.

---

## Working memory structure

```markdown
# Working Memory: flyvis_flowcond_noise_005_gnn_llm_cv00

## Paper Summary (update at every block boundary)

## Knowledge Base (accumulated across all blocks)

### Results Comparison Table
| Iter | Config summary | Eij_gate | Wij_R2 peak (mean±std) | CV% | peak iter | final/peak | Eij_R2 | wrong% | msg_i_R2 | tau_R2 | V_rest_R2 | rollout_r | Robust? | Hypothesis tested |
| ---- | -------------- | -------- | ---------------------- | --- | --------- | ---------- | ------ | ------ | -------- | ------ | --------- | --------- | ------- | ----------------- |

### Established Principles
### Falsified Hypotheses
### Open Questions

---

## Previous Block Summaries

RULE: Keep summaries for the last 4 completed blocks, sorted oldest to newest.
This section MUST appear before ## Current Block.

---

## Current Block (Block N)

### Block Info
### Current Hypothesis
### Iterations This Block
### Emerging Observations

CRITICAL: This section must ALWAYS be at the END of the memory file.
```

---

## Start call

On the PARALLEL START call there are no results yet. Read this file and the base config, set
all eight slots to the baseline, and state in memory that block 1 measures the variance every
later block is judged against — and that its second purpose is to confirm on this dataset the
peak-at-16k-to-24k trajectory the sweep established, because block 2's ranking depends on the
peak being a real feature and not one run's noise.

Launch:

```bash
python GNN_LLM.py -o generate_train_test_plot_Claude \
    flyvis_flowcond_noise_005_gnn_llm_cv00 \
    iterations=168 --cluster --resume
```

`generate_data: false` in the `claude:` block is what stops the `generate` in the
task name from regenerating the dataset; the cluster job runs `train_subprocess.py` only,
and test/plot runs locally. The base config carries its `claude:` block; `--cluster` needs `data_paths.json` at the repo
root with `cluster_root_dir` pointing at the cluster checkout. `hard_runtime_limit_min` is 240
against a 150-minute target, because the LSF queues default to a **120-minute** run limit and
a job submitted without `-W` is killed with `TERM_RUNLIMIT` at two hours — which is exactly
how the 22-arm sweep died on 2026-09-18.
