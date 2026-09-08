# Conductance GNN recovery — flyvis, sigma = 0.05

## Goal

Fit a `flyvis_conductance` GNN to voltage traces generated **by a conductance model**, and
recover the generator's parameters from the learned message. Nothing about the synapse is
assumed: `W_ij` is a free per-edge weight, and the driving force, the rectifier and the
conductance all live inside one learned surface `g_phi(v_j, a_j, v_i, a_i)`.

This is the first setting in which the GNN is asked to recover a driving force that
**actually exists in the data** — `(E_i - v_i)` is a real term here, not an artefact of
fitting a current-based teacher.

**Primary metric: `connectivity_R2`. Gating metric: `fit_r2_median` in
`tmp_training/gnn_conductance_fit.log`.** Read "the gate" below before trusting any
recovery number.

---

## The model, exactly

`NeuralGNN` with `signal_model_name: flyvis_conductance`, in
`src/connectome_gnn/models/neural_gnn.py`:

```
in_features = [ v_j , a_j , v_i , a_i ]          # column 0 is ALWAYS v_j, in every family
g_phi_out   = g_phi(in_features)                 # 3-layer MLP, hidden 80, output 1
if g_phi_positive:  g_phi_out = g_phi_out ** 2
W_edge      = W[edge] ** 2  if w_squared  else  W[edge]
msg_i       = sum over incoming edges of  W_edge * g_phi_out
dv_i/dt     = f_theta(v_i, a_i, msg_i, excitation)
```

`input_size` is `2 + 2*embedding_dim` = 6 with `embedding_dim: 2`. The prefix ordering
matters: `[v_j, a_j]` first, so column 0 means `v_j` in this family exactly as it does in
`flyvis_current`. The group lasso's column slices and the `coeff_g_phi_norm` anchor both
index against that layout.

### Where the sign of a synapse lives — and why the two squaring flags are exclusive

The generator's message is `W_ij * relu(v_j) * (E_i - v_i)` with `W_ij >= 0`, so the
polarity of a synapse is carried entirely by the sign of `(E_i - v_i)`.

The GNN has no such factor, so the sign must live in one of its two terms:

| `g_phi_positive` | `w_squared` | Sign carried by | Learned W is |
| --- | --- | --- | --- |
| `true` | `false` | `W` (signed) | a signed quantity — compare `abs(W)` to the truth, never `W` |
| `false` | `true` | `g_phi` (signed) | `W**2`, a non-negative conductance, directly comparable to `ode_params.W` |

**Setting both true raises in the constructor.** `W**2` and `g_phi**2` are each
non-negative, so every message would be non-negative and no inhibitory synapse could be
represented at all — the model would fit every dataset with one hand tied. The two flags
name the two places the sign can live, and exactly one must stay free.

This exploration runs the second row: `g_phi_positive: false`, `w_squared: true`, so
`get_model_W` returns `W**2` and every panel compares a conductance to a conductance.

Simulation constants, all FROZEN: 13,741 neurons, 1,736 input neurons, 65 cell types,
434,112 edges, 64,000 frames, `delta_t` 0.02, `noise_model_level` **0.05** (five
hundredths — the `noise_005` in the name is 0.05, not 0.005).

---

## THE GATE — why a recovery number here can be fiction

`W_ij` and `E_ij` enter the message as a **product**: `g * act(v_j) * (E - v_i)`. Scaling
the conductance up by c and shrinking the driving force by c leaves every message, and
every trajectory, unchanged. The data break the tie only through the `v_i`-dependence of
the driving force, and on flyvis `v_i` (mean +0.416, sd 0.629) modulates a driving force of
magnitude 14.7-24.4 by only 3-4%.

A conductance known-ODE checkpoint measured at 400k iterations had `g` 3.3x too large and
the driving force 0.39x too small — reciprocal to within 14% — while its free-run
trajectory matched at r = 0.95. The GNN sits in the same valley.

`E_ij` and `W_ij` are read back out of the learned message by
`extract_conductance_params_from_gnn`, which does **not** fit anything. Dividing the
message by `v_j` leaves a straight line in `v_i`:

```
msg_ij / v_j  =  W_ij * (E_i - v_i)        slope = -W_ij,  intercept = W_ij * E_i
```

so one least-squares line per edge gives `W_ij = -slope` and `E_ij = -intercept/slope`.
Samples below the 25th percentile of the positive `v_j` are dropped, which is what stops a
small `v_j` from manufacturing an enormous `W`.

**The straight line is also the test.** `fit_r2_median` is the median per-edge R2 of that
line. If it is not near 1 the learned message is not affine in `v_i`, the model has not
found the conductance form, and the `W` and `E` beside it describe nothing. On a model
trained on *current* data — where there is no `(E - v_i)` term to find — the extraction
correctly reports `fit_r2_median` 0.079 and a W recovery of -0.009. It declines to
manufacture structure that is not there, and so must you.

**Rule: do not report or act on `E_ij` or the extracted `W` while `fit_r2_median < 0.9`.**
Early in training it will sit near 0.15-0.5 and the E_ij RMSE will be enormous. That is the
gate working, not a failure. Say so in the entry and move on.

`w_r2_scaled` is the extracted `W` scored after dividing out the one global gain the GNN
cannot pin down (`W` and the amplitude of `g_phi` trade off exactly). Its slope is 1 by
construction, so **the slope is not a result** — only the R2 and the reported `w_scale`.

---

## Metrics — the exact names

**From the per-slot analysis log** (the `Metrics:` file named in your prompt):

| Key | Meaning | Target |
| --- | --- | --- |
| `connectivity_R2` | R2 of the corrected `W` against the truth | > 0.85 |
| `raw_W_R2` | same, before the g_phi correction | context only |
| `tau_R2`, `V_rest_R2` | tau and V_rest recovered **out of f_theta**, by fitting its local slope and offset per neuron | tau > 0.90, V_rest > 0.70 |
| `g_phi_functional_R2`, `f_theta_functional_R2` | shape agreement of the learned functions | > 0.90 |
| `onestep_pearson` | one-step-ahead correlation | > 0.99 |
| `rollout_pearson` | free-run rollout correlation | > 0.95 |
| `cluster_accuracy` | cell-type separability of the embedding | > 0.80 |
| `training_time_min` | wall clock | see the DAL rule in your prompt |

There is **no** `rollout_pearson_r`, no `test_R2`, no `conn_R2`, and no
`learning_rate_W_start`. A config key that does not exist is silently swallowed, so an
edit to an invented name costs you a whole slot with no error.

**`tau_R2` and `V_rest_R2` are real for this model.** Older instruction files claim a GNN
"absorbs tau and V_rest implicitly into f_theta" and that these read 0.00 or N/A. That is
false: `extract_f_theta_slopes` plus `derive_tau`/`derive_vrest` recover both, and a real
flyvis GNN slot logs `tau_R2: 0.9927`, `V_rest_R2: 0.5049`. Treat a tau collapse as a
genuine failure, never as expected behaviour.

**From the run directory**, `log/fly/<slot>/tmp_training/`. Name these paths exactly; do
not Glob:

| Path | What it carries |
| --- | --- |
| `metrics.log` | CSV: `iteration,connectivity_r2,vrest_r2_raw,tau_r2_raw,hidden_nnr_pearson,anchor_nnr_pearson,vrest_r2_clean,n_out_vrest,n_total_vrest,tau_r2_clean,n_out_tau,n_total_tau` |
| `gnn_conductance_fit.log` | CSV: `iteration,fit_r2_median,w_r2_scaled,w_scale` — **the gate** |
| `reversal_rmse.log` | CSV: `iteration,rmse,r2,slope,n_edges` — E_ij, valid only once the gate is passed |
| `g_phi_discard.log` | CSV: `iteration,cosine_to_keep,ratio_vi,ratio_ai,ratio_noise` — see the lasso block |
| `rollout_r.log` | CSV: `iteration,r,rmse,n_frames` |
| `Wij/raw_*.png`, `Wij/comparison_*.png`, `Wij/connectivity_*.png` | raw `W**2`, the g_phi-corrected `W*`, and the connectivity heatmap |
| `Eij/`, `tau/`, `vrest/`, `msgi/` | 2x2 recovery panels per quantity |
| `function/g_phi/`, `function/f_theta/` | learned function shapes against ground truth |
| `embedding/` | the learned per-neuron embedding |
| `traces/rollout_*.png` | stacked free-run traces |

`msgi/msgi_*.png` scores `msg_i`, the aggregated message on 10 fixed frames. It is the one
panel the degeneracy does not touch, and it read R2 +0.78 on a checkpoint whose `Wij`
panel read -10.5. **When those two disagree, the message panel describes what the
trajectory depends on.** Record both every iteration.

An empty folder is information: it means the quantity is not recovered for this model, not
that plotting failed.

---

## Explorable parameters

YOU MAY MODIFY ONLY THE PARAMETERS IN THIS TABLE.

| Parameter | Default | Suggested sweep | Notes |
| --- | --- | --- | --- |
| `lr_W` | 0.0009 | {3e-4, 6e-4, 9e-4, 2e-3} | learning rate on `W` |
| `lr` | 0.0018 | {6e-4, 1.2e-3, 1.8e-3, 4e-3} | learning rate on the g_phi and f_theta MLP weights |
| `lr_embedding` | 0.002325 | {1e-3, 2.3e-3, 5e-3} | learning rate on the per-neuron embeddings `a` |
| `w_init_scale` | 1.0 | {1, 5, 20, 60} | **the highest-value knob in this file — see block 2** |
| `w_init_mode` | `randn_scaled` | `randn_scaled`, `uniform_scaled`, `zeros` | lowercase. bound = `scale / sqrt(434,112)` |
| `coeff_g_phi_input_group_L1` | 0 | {0, 0.25, 1, 5} | group lasso over g_phi's input columns — see block 5 |
| `coeff_g_phi_diff` | 375 | {0, 375, 750} | positive-monotonicity prior on `d g_phi / d v_j`, i.e. `ReLU(-dg/dv)`. **Not** a "non-trivial activation" penalty |
| `coeff_g_phi_norm` | 0.45 | {0, 0.45, 0.9} | normalisation penalty on g_phi at saturation voltage |
| `coeff_g_phi_weight_L1` | 0 | {0, 0.01, 0.1} | L1 on the g_phi MLP weights. **Values >= 0.1 are known to collapse training at flyvis scale** |
| `coeff_g_phi_weight_L2` | 0 | {0, 1e-4, 1e-3} | L2 on the g_phi MLP weights |
| `coeff_f_theta_weight_L1` | 0.025 | {0, 0.025, 0.1} | L1 on the f_theta MLP weights |
| `coeff_f_theta_weight_L2` | 0.0005 | {0, 5e-4, 5e-3} | L2 on the f_theta MLP weights |
| `coeff_W_L1` | 7.5e-05 | {0, 1e-5, 7.5e-5, 3e-4} | L1 on `W`. With `w_squared` this acts on the square root of the conductance |
| `coeff_W_L2` | 7.5e-07 | {0, 7.5e-7, 1e-5} | L2 on `W`, same caveat |
| `batch_size` | 4 | {2, 4, 8} | INTEGER |
| `hidden_dim` | 80 | {64, 80, 128} | g_phi and f_theta width |
| `data_augmentation_loop` | 100 | 50-400 | retune against the time target your prompt states; do not hard-code a threshold |

**`regul_annealing_rate` MUST STAY 0.0.** The annealed coefficient is
`coeff * (1 - exp(-rate * epoch))`, which is **exactly zero at epoch 0**. Under the default
rate of 0.5 every regularisation coefficient you set would be identically zero through the
first epoch — you would be sweeping a number that never reaches the loss. Note that
`coeff_W_L1/L2`, both `g_phi_weight` terms, both `f_theta_weight` terms and
`coeff_g_phi_input_group_L1` are annealed, while `coeff_g_phi_diff` and `coeff_g_phi_norm`
are not. At rate 0 the distinction disappears, which is why it stays 0.

---

## Frozen — do not touch

- `simulation.seed`, `training.seed` — overwritten every batch with `iteration*1000 + slot`
  and `iteration*1000 + slot + 500`. Log the values the prompt reports.
- `training.n_epochs` — overwritten from `claude.n_epochs` every batch. Editing it is a
  silent no-op, so never dedicate a block to "training volume via n_epochs".
- `dataset` — must stay `flyvis_conductance_noise_005_blank50_cv00` in every slot.
- `simulation.*` — `n_neurons`, `n_edges`, `n_frames`, `delta_t`, `noise_model_level`.
- **`g_phi_positive` and `w_squared`** — this exploration is defined by
  `g_phi_positive: false`, `w_squared: true`. Flipping either changes what `W` means and
  makes every earlier iteration incomparable. If you want the other parameterisation, that
  is a separate exploration; say so in `user_input.md`.
- `input_size: 6`, `embedding_dim: 2`, `signal_model_name`, `prediction`, `aggr_type`,
  `update_type`, `use_gt_edges: true`, `mlp_precision: bf16`, `torch_compile: true`.

---

## Block structure

`n_iter_block` comes from the base config's `claude:` block; the block number and a
`>>> BLOCK END <<<` marker are injected into your prompt. Do not restate an iteration
count here.

| Block | Mode | Focus | Parameters to scan | Ranges |
| --- | --- | --- | --- | --- |
| 1 | Robustness | Baseline variance | none — all 4 slots identical | Establish CV of `connectivity_R2`, `tau_R2`, `V_rest_R2`, and the trajectory of `fit_r2_median` |
| 2 | Exploration | **W initialisation scale** | `w_init_scale`, `w_init_mode` | scale {1, 5, 20, 60}; mode {randn_scaled, uniform_scaled} |
| 3 | Exploration | Learning rates | `lr_W`, `lr`, `lr_embedding` | as in the parameter table; record the lr_W/lr ratio |
| 4 | Exploration | g_phi shape priors | `coeff_g_phi_diff`, `coeff_g_phi_norm` | diff {0, 375, 750}; norm {0, 0.45, 0.9} |
| 5 | Exploration | **Input group lasso** | `coeff_g_phi_input_group_L1` | {0, 0.25, 1, 5} |
| 6 | Exploration | MLP weight regularisation | `coeff_g_phi_weight_L1/L2`, `coeff_f_theta_weight_L1/L2` | keep g_phi_weight_L1 < 0.1 |
| 7 | Exploration | W regularisation | `coeff_W_L1`, `coeff_W_L2` | W_L1 {0, 1e-5, 3e-4}; W_L2 {0, 1e-5} |
| 8 | Exploration | Capacity and batch | `hidden_dim`, `batch_size`, `data_augmentation_loop` | hidden {64, 80, 128}; bs {2, 4, 8} |
| 9 | Exploration | Free — combine | any of the above | Consolidate the best of blocks 2-8 |
| 10 | Robustness | Final validation | none — all 4 slots at the champion | Confirm CV and no catastrophic seed |

**Slot 0 ratchets.** From block 2 onward, slot 0 is the best configuration so far, not the
original baseline.

Per-block notes:

- **Block 2 is first for a measured reason.** With `w_squared: true` and
  `w_init_mode: randn_scaled`, `W` starts near `1/sqrt(434,112)` ~ 0.0015, so the effective
  conductance `W**2` starts at ~2.3e-6 against a true median conductance of ~0.03 — five
  orders of magnitude low. The gradient is `d(W**2)/dW = 2W` ~ 0.003, so `W` barely moves:
  an observed run held `connectivity_R2` at exactly -0.1326, unchanged to six digits, from
  iteration 6,401 to 12,801. **A frozen `connectivity_R2` is the signature of this
  vanishing-gradient start.** `w_init_scale` around `sqrt(0.03) * sqrt(434112)` ~ 114 would
  put `W**2` at the true scale; sweep upward from 1 and watch whether `connectivity_R2`
  starts moving at all before optimising anything else.
- **Block 5 is the scientific question this dataset was made for.** The group lasso
  penalises each input column of g_phi's first layer jointly, so it can drop a whole input
  pathway. On *current*-generated data the true message is `W*relu(v_j)` with no
  postsynaptic dependence, and the lasso correctly drives the `v_i` and `a_i` columns to
  zero. Here `(E_i - v_i)` is real, so those inputs **are** needed and the same lasso must
  **not** discard them. Read `g_phi_discard.log`: `ratio_vi` and `ratio_ai` are the
  gradient of g_phi with respect to `v_i` and `a_i` divided by its gradient with respect to
  `v_j`, so **them falling toward zero IS the discard you are testing for**. A lasso
  strength that keeps them up here while still discarding them on current data would be a
  single specification serving both data families — that is the win condition.
  (These columns are all-NaN on `flyvis_current`, which has no `v_i`/`a_i` inputs at all.
  Here they are meaningful.)
- **Block 6**: `coeff_g_phi_weight_L1 >= 0.1` is known to collapse training at flyvis
  scale — the L1 gradient dominates the connectivity gradient and `W` goes to zero, showing
  as `connectivity_R2` near 0 and a spike at exactly -1.0 in the relative-error panel. Back
  off immediately if you see it.
- **Block 7**: with `w_squared` the L1 acts on the square root, so it is effectively an
  L1/2 penalty on the conductance. Expect it to bite harder than the same number would in a
  linear-W run.

---

## Acceptance

| Verdict | Rule |
| --- | --- |
| Stable-Robust | all 4 slots `connectivity_R2` >= 0.85 and CV < 3% |
| Stable | mean `connectivity_R2` >= 0.80, CV < 10% |
| Unstable | mean < 0.80 or CV >= 10% |
| Catastrophic | any slot `connectivity_R2` < 0.50 — reject, do not pursue |

Three rules specific to this experiment:

- **Gate first.** A slot whose final `fit_r2_median` is below 0.9 has no valid `E_ij` or
  extracted `W`. Record `connectivity_R2` and the trajectory metrics for it, mark the
  recovery numbers "gated out", and do not rank it on them.
- **Frozen-W check.** If `connectivity_R2` in `metrics.log` is identical to four decimals
  across three consecutive checkpoints, the initialisation is the problem, not the
  regularisation. Return to block 2 rather than continuing the current block.
- **Trajectory check.** Record the peak of `connectivity_r2` and `(final - peak) / peak`;
  treat `final / peak < 0.95` as disqualified.

---

## Iteration workflow

For each slot, in this order:

1. Read the analysis log for the slot and extract every metric named in the Metrics table.
2. Read the last row of `gnn_conductance_fit.log` **first** — it decides whether the
   recovery numbers mean anything. Then `metrics.log` (last row plus the peak of
   `connectivity_r2`), `reversal_rmse.log`, and `g_phi_discard.log`.
3. Look at `Wij/comparison_*.png` and `msgi/msgi_*.png` from the latest checkpoint, and
   `function/g_phi/func_*.png` to see whether the learned edge function has the
   sign-changing shape in `v_i` that `(E_i - v_i)` requires.
4. Write one entry per slot to the analysis log **and** to memory. The heading must be
   exactly `## Iter N: <short title>` — the resume mechanism parses that pattern and a
   different heading breaks `--resume`.
5. Edit all slot configs for the next batch under the causality rule in your prompt: slot 0
   is the parent unchanged, every other slot changes exactly one parameter.

Entry template:

```markdown
## Iter N: <one-line hypothesis>
- Slot: S | seeds: sim=<value> train=<value>
- Changed: <parameter> <old> -> <new>   (slot 0: unchanged control)
- GATE fit_r2_median: <value>  -> recovery numbers <valid | gated out>
- connectivity_R2: <value>  (peak <value> at iter <value>, final/peak <value>)
- w_r2_scaled: <value>  (w_scale <value>; slope is 1 by construction, not a result)
- E_ij: rmse <value>  r2 <value>       [omit if gated out]
- tau_R2: <value>   V_rest_R2: <value>
- onestep_pearson: <value>   rollout_pearson: <value>
- g_phi_discard: cosine_to_keep <value>  ratio_vi <value>  ratio_ai <value>
- msg_i panel R2: <value from msgi/*.png>
- training_time_min: <value>
- Verdict: <Stable-Robust | Stable | Unstable | Catastrophic | Gated-out | Frozen-W | Disqualified-late-collapse>
- Reading: <two sentences: what moved, and whether the gate licensed the reading>
```

---

## Block boundaries

At `>>> BLOCK END <<<`:

1. Write a block summary into `## Previous Block Summaries`.
2. Promote findings that survived a robustness check to `### Established Principles`; move
   refuted ones to `### Falsified Hypotheses` with the evidence that killed them.
3. Update the comparison table with every metric column, including `fit_r2_median`.
4. Write the block's winner to
   `config/fly/flyvis_conductance_noise_005_conductance_gnn_winner.yaml` with a comment
   header giving the iteration and the headline metrics. `config/fly/` is the only config
   directory that exists.
5. State the next block's hypothesis in `## Current Block`.

---

## Working memory structure

```markdown
# Working Memory: flyvis_conductance_noise_005_conductance_gnn_cv00

## Paper Summary (update at every block boundary)

## Knowledge Base (accumulated across all blocks)

### Results Comparison Table
| Iter | Config summary | fit_r2_median | conn_R2 (mean±std) | CV% | w_r2_scaled | E_ij r2 | tau_R2 | V_rest_R2 | rollout_pearson | ratio_vi | Robust? | Hypothesis tested |
| ---- | -------------- | ------------- | ------------------ | --- | ----------- | ------- | ------ | --------- | --------------- | -------- | ------- | ----------------- |

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

On the PARALLEL START call there are no results yet. Read this file and the base config,
set all four slots to the baseline, and state in memory that block 1 measures the variance
every later block is judged against — and that its second purpose is to see whether
`fit_r2_median` rises at all under the default initialisation, because block 2 exists to
fix it if not.

Launch:

```bash
python GNN_LLM.py -o generate_train_test_plot_Claude \
    flyvis_conductance_noise_005_conductance_gnn_cv00 \
    iterations=120 --cluster --resume
```

The base config must carry a `claude:` block; `--cluster` needs `data_paths.json` at the
repo root with `cluster_root_dir` pointing at the cluster checkout.
