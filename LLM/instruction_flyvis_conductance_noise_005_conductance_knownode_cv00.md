# Conductance known-ODE recovery — flyvis, sigma = 0.05

## Goal

Recover the four generating parameters of a conductance flyvis network from its voltage
traces: the per-edge conductance `W_ij`, the per-neuron reversal potentials `E_exc_i` and
`E_inh_i`, the membrane time constant `tau_i`, and the resting potential `V_rest_i`.

The model here is `flyvis_conductance_known_ode`, which assumes the generator's own
functional form. Nothing about the synapse is being learned as a free function — this is
parameter recovery, and every quantity has a true value on disk in `ode_params.pt`.

**This exploration optimises TWO objectives in order, not one. Stage 1 is the FIT:
`msg_i_R2`. Stage 2 is the IDENTIFICATION: `Wij_R2`, then `Eij_R2`, then
`|Wij_slope - 1|`.** Read "the two objectives" below before ranking anything — a W
recovery number on its own cannot steer this experiment, and the reason is measured, not
assumed.

---

## The model, exactly

`FlyvisConductanceKnownODE`, in `src/connectome_gnn/models/known_ode.py`:

```
g_ij      = W_ij ** 2                            # conductance, non-negative BY CONSTRUCTION
E_ij      = E_inh[dst]  if edge_is_inh  else  E_exc[dst]
msg_i     = sum over incoming edges of  g_ij * ReLU(v_src) * (E_ij - v_dst)
dv_i/dt   = ( -v_i + msg_i + I_i(t) + V_rest_i ) / softplus(raw_tau_i)
```

The activation is exactly ReLU — not "typically ReLU", and it is **not squared**. The
square is on `W`, so the stored parameter is the square root of the conductance and
`get_model_W` returns `W**2` for every comparison. The polarity of an edge comes from
`ode_params.edge_is_inh`, the presynaptic cell's Dale sign, **not** from `sign(W)`, which
is uniformly non-negative on this data.

Learned tensors and their sizes:

| Tensor | Shape | Count | Note |
| --- | --- | --- | --- |
| `model.W` | (n_edges, 1) | 434,112 | square root of the conductance |
| `model.raw_tau` | (n_neurons,) | 13,741 | `tau = softplus(raw_tau)` |
| `model.V_rest` | (n_neurons,) | 13,741 | |
| `model.E_exc`, `model.E_inh` | (n_neurons,) each | 27,482 | see below |

Simulation constants, all FROZEN: 13,741 neurons, 1,736 input neurons, 65 cell types,
434,112 edges, 64,000 frames, `delta_t` 0.02, `noise_model_level` **0.05** (that is
five hundredths — the `noise_005` in the name is 0.05, not 0.005).

---

## THE TWO OBJECTIVES — read this before ranking anything

**Why `Wij_R2` alone leaves you blind.** On the two runs that produced this file
it read **-1.17** and **-4.94**. An acceptance ladder whose best grade needs >= 0.85 and
whose reject grade is < 0.50 returns "reject" for every configuration you can reach, which
means it returns the same verdict for an improvement and for a regression. A metric that
cannot distinguish those is not a metric, and optimising against it is guessing.

Two numbers do have a working point and a gradient, and both are in the analysis log:

| Objective | Metric | Question it answers |
| --- | --- | --- |
| **1. FIT** | `msg_i_R2` | Does the model reproduce the generator's MESSAGE? |
| **2. IDENTIFICATION** | `Wij_R2` | Given the fit, is the SHAPE of the conductance right? |
| | `Eij_R2`, `Wij_slope` | How far along the degenerate valley does this run sit? |

**The Observed column is deliberately empty: block 1 measures it.** Every number this
file used to quote was taken on a SUPERSEDED dataset — the old bracketed twin, whose true
reversals were two constants, +24.77 and -14.47. This exploration runs on the rig-2 (ion)
twin, where the truth is different in kind and not only in value (see "What the truth
looks like now"). Do not carry any prior number across; establish the baseline in block 1
and judge everything against that.

**They are not the same objective and improving one does not imply improving the other.**
A model can pass every message perfectly and still sit anywhere along the valley: `msg_i`
is the product, and the product is invariant to the trade. So do NOT simply chase
`msg_i_R2` — a run with `msg_i_R2` 0.99 and `Wij_slope` 3.0 has recovered nothing about the
connectome.

**THE RANKING RULE, and it is lexicographic:**

1. **`msg_i_R2` is a floor, not a target.** A slot whose `msg_i_R2` falls more than 0.02
   below the block's control has BROKEN THE FIT. Reject it whatever it did to
   `Wij_R2` — a connectivity number computed from a model that no longer
   reproduces the message is measuring noise.
2. **Among slots that hold the floor, rank on `Wij_R2`**, then on `Eij_R2`, then on
   `|Wij_slope - 1|` shrinking. These three move together when the scale is genuinely
   being pinned, and that agreement is itself the evidence. `Wij_slope` is the gain: it
   is the slope of learned `W**2` against true `W`, so 1.0 means the conductance is on
   the right scale and 2.4 means it is 2.4x too large.
3. **Report `Wij_R2_all` beside `Wij_R2` in every entry.** `Wij_R2` is outlier-filtered
   at `recovery.W_outlier_thresh`, `Wij_R2_all` is not; the gap between them says how much
   of the recovery rests on a tail. Rank on the filtered one, report both.

`msg_i_R2` earns its place as the floor because it is the one number that stays meaningful
when everything else has collapsed: on the six conductance GNN runs it read **-0.04**,
which said "this model passes no message at all" at a moment when `Wij_R2` read a
respectable-looking -0.13 and the rollout correlation read 0.89.

---

## NEVER WRITE A NUMBER YOU DID NOT READ FROM THIS RUN

Every metric you record must come from a file **this run produced**: the per-slot analysis
log named in your prompt, or `log/fly/<slot>/results/metrics.txt`, or a CSV under
`log/fly/<slot>/tmp_training/`. If none of them is readable for a slot, the honest entry is
**no data** — write that, and say which files you checked.

This rule exists because it was broken. A previous run had all 8 slots fail on a missing
dataset, while `user_input.md` still carried a hand-harvested table from an EARLIER
exploration together with the instruction "copy them into the batch-1 entries, do not
attempt to re-derive them". The agent followed it, and a block-1 variance reference was
recorded — mean, sd and CV per metric, to four decimals — for eight runs that had never
executed. Nothing in the log said so.

Three consequences that bind you:

1. **A table in `user_input.md` is not a measurement.** Treat any number handed to you
   there as context, never as a result, and never copy one into the analysis log or the
   comparison table. If the user wants a historical baseline recorded, they can say so and
   you label it as such, with its provenance and its date.
2. **All slots failing is a pipeline fault, not a result.** Do not back-fill it, do not
   average around it, and do not advance the block plan as if the batch had run. Read
   `cluster_train_<slot>.err` in the exploration directory — that is where the training
   subprocess's traceback goes — report the exception verbatim in `user_input.md`, and hold
   the configs steady until metrics return.
3. **Numbers carried across a dataset change are wrong even when they are real.** The
   metric names AND the ground truth both changed with the extraction refactor and the
   rig-2 twin. A four-decimal number that predates either is not a stricter baseline, it is
   a false one.

---

## What the truth looks like now

The dataset is generated by the RIG-2 (ion) conductance twin, and its reversals are
PHYSIOLOGICAL rather than bracketed. That changes what `Eij_R2` is measuring, so read this
before interpreting it.

| | excitatory | inhibitory |
| --- | --- | --- |
| edges | 282,957 | 151,155 |
| true `E_ij` | **+10.368**, one value for every edge | **-5.872 … +2.178**, **60 distinct values** |
| of which positive | — | **4,819 edges (3.2%)** |

Three consequences.

1. **The excitatory row carries no information.** Every one of the 282,957 excitatory
   edges has the identical true `E`, so those points form a single vertical stripe in the
   `E_ij` scatter and contribute nothing to `Eij_R2` beyond an offset. All of the
   recoverable variance is in the 151,155 inhibitory edges. A slot that improves `Eij_R2`
   has improved the chloride reversals; nothing else was ever on the table.
2. **"Inhibitory" no longer means "negative reversal".** 3.2% of inhibitory edges have a
   true `E` above zero — depolarising chloride, which inhibits by shunting rather than by
   hyperpolarising. Any summary that averages a single number over "the inhibitory group"
   is averaging across edges whose driving force has opposite signs. Report the
   distribution, not a group mean.
3. **The reversal is a property of the POSTSYNAPTIC cell type**, one value per type, and
   the edge type only selects which of the two ions applies. So `E_ij` has 61 degrees of
   freedom in truth (1 cation + 60 chloride) while the model fits 27,482 of them. Watching
   the per-neuron estimates collapse onto their cell-type value is itself a result, and
   `tmp_training/Eij/` draws it.

---

## THE DEGENERACY — read this before optimising anything

`g_ij` and the driving force `(E_i - v_i)` enter the message **as a product**. Scaling the
conductance up by any constant c and shrinking the driving force by the same c leaves every
message, and therefore every trajectory, exactly unchanged. The data break that tie only
through the *v_i-dependence* of the driving force.

On flyvis that lever arm is tiny. Over the recording `v_i` has mean +0.416 and standard
deviation 0.629, while `|E - v_i|` is about 24.4 for an excitatory edge and 14.7 for an
inhibitory one. So `v_i` modulates the driving force by 3-4%, and the gradient signal that
separates `g` from `E` is a few percent of the message.

A trained checkpoint at 400k iterations measured:

| | excitatory (282,957 edges) | inhibitory (151,155 edges) |
| --- | --- | --- |
| E true -> learned | +24.77 -> +9.97 (x0.40) | -14.47 -> -7.00 (x0.48) |
| driving-force ratio | x0.392 | x0.492 |
| conductance g ratio | **x3.29** | **x2.67** |
| g * drive ratio | x1.14 | x1.18 |
| correlation with truth | g +0.68, g*drive +0.87 | g +0.56, g*drive +0.67 |

The two factors were wrong by almost exactly reciprocal amounts while the product — the
only thing the data constrain — was right to 14-18%, and the free-run trajectory matched
the truth at r = 0.95.

**Three consequences for how you run this exploration.**

1. A `Wij_R2` far below zero with `Wij_slope` near 2.4 is the signature of this
   degeneracy, not of a failed fit. Do not respond to it by raising `lr_W`.
2. `msg_i_R2` scores the aggregated message, which is the product that survives
   the trade. It has read R2 +0.78 on a checkpoint where the W recovery read -10.5.
   **When those two disagree, `msg_i_R2` is describing what the trajectory depends
   on and `Wij_R2` is describing a factorisation the data cannot resolve.** Both are
   keys in the analysis log, so record them side by side in every entry;
   `tmp_training/msgi/msgi_*.png` is the same quantity as a picture if you want to
   see the shape of the disagreement.
3. What breaks the tie is anything that pins the scale of E. `student_reversal_dim` would
   be the obvious lever — 2 free reversals instead of 27,482 — but you **cannot** set it:
   in recovery mode `_resolve_student_knobs` forces `RECOVERY_DEFAULTS` with
   `student_reversal_dim="per_neuron"` and REFUSES any `student_*` key a config tries to
   set. Reaching it needs a code change; post in `user_input.md` rather than trying.
   The levers you do have are `coeff_tau_L1/L2` and `coeff_V_rest_L1/L2`, which pin the
   scale from the other end of the same invariance.

---

## Metrics — the exact names

**WHICH FILE, WHICH NAMES — read this first.** Two writers score the same quantities
under two vocabularies, and the file you are told to read decides which one you see:

| file | written by | W recovery | gain | reversal | message |
| --- | --- | --- | --- | --- | --- |
| `results/metrics.txt` (**your source**) | `-o test_plot` (GNN_PlotFigure) | `connectivity_R2_scaled`, `W_corrected_R2`, `connectivity_pearson_r` | `w_scale` | `Eij_R2`, `Eij_slope`, `Eij_rmse`, `Eij_n_edges` | `msg_i_R2`, `msg_i_slope`, `msg_i_rmse`, `msg_i_n` |
| per-slot analysis log, in-training | trainer via `score_recovery` | `Wij_R2`, `Wij_R2_all`, `Wij_slope` | `Wij_slope` | `Eij_R2`, `Eij_slope`, `Eij_rmse`, `Eij_n` | `msg_i_R2`, `msg_i_slope`, `msg_i_rmse`, `msg_i_n` |

`connectivity_R2_scaled` is the scale-corrected W recovery and `w_scale` is the gain (learned
conductance = `w_scale` × true; 1.0 is perfect) — these are the `metrics.txt` spellings of what
the analysis log calls `Wij_R2` and `Wij_slope`. Rank on whichever file you read, using its
own names, and never write a name from one file into an entry sourced from the other. An
earlier version of this file said the legacy names were gone; that was true of the trainer's
log and false of `metrics.txt`, and the correction here is the measured one.


**From the per-slot analysis log** (the `Metrics:` file named in your prompt). Use these
spellings; nothing else exists:

Every quantity is scored through ONE entry point, `score_recovery`, which emits
`<key>_R2`, `<key>_slope`, `<key>_rmse` and `<key>_n` for each, plus `<key>_R2_all`,
`<key>_n_outliers` and `<key>_pct_outliers` where an outlier threshold applies, plus
`<key>_estimator` and `<key>_correction` naming how the number was produced. The keys
are `Wij`, `tau`, `V_rest`, `Eij`, `msg_i`.

| Key | Meaning | Target |
| --- | --- | --- |
| `Wij_R2` | R2 of learned `W**2` against the true conductance, outlier-filtered at `recovery.W_outlier_thresh` | **STAGE-2 OBJECTIVE** |
| `Wij_R2_all` | the same without outlier filtering | report beside `Wij_R2`; the gap says how much rests on a tail |
| `Wij_slope` | identity-line slope of that scatter — **this is the gain** | -> 1.0; 2.4 means the conductance is 2.4x too large |
| `Wij_rmse` | RMSE of the conductance | context only |
| `Wij_estimator` | how W was obtained; on this model always `direct` | provenance — flag anything else |
| `Wij_correction` | `W**2 (stored value is sqrt of the conductance)` | provenance |
| `tau_R2`, `tau_slope` | `softplus(raw_tau)` against `tau_i` | > 0.90 |
| `V_rest_R2`, `V_rest_slope` | `V_rest` against `V_i_rest` | > 0.85 |
| `Eij_R2` | R2 of the learned per-edge `E_ij` against the true one | the hard one, see the degeneracy section |
| `Eij_slope` | identity-line slope of the same scatter | 1.0 is perfect; well below 1 is the degenerate valley |
| `Eij_rmse` | RMSE of `E_ij`, in the dataset's voltage units | context; read it against the spread of the truth, not a fixed bar |
| `Eij_n` | edges E_ij was scored on | context only, expect 434,112 |
| `msg_i_R2` | R2 of the aggregated per-neuron message | **STAGE-1 FLOOR.** Must not fall > 0.02 below the block control |
| `msg_i_slope`, `msg_i_rmse` | slope and RMSE of the message scatter | slope -> 1.0; rmse context only |

**Renamed, not removed.** `connectivity_R2` → `connectivity_R2_scaled` is the headline in `metrics.txt`; `raw_W_R2` is not written by either writer. `w_scale` and `fit_r2_median` live in `metrics.txt` / `gnn_conductance_fit.log`; `Wij_*` and `Eij_gate` live in the trainer's analysis log. See the table above.

**THERE ARE NO "OBSERVED" NUMBERS IN THIS FILE, AND THAT IS DELIBERATE.** Every value
this file used to quote was measured on a superseded dataset and at 10 epochs, while this
exploration runs on the rig-2 (ion) twin at `claude.n_epochs` 3. Both changes move the
absolute scale, so a carried-over number would be worse than no number: it would look like
a target while being unreachable or already passed. Block 1 exists to establish the
baseline and its variance; rank every later slot against slot 0 of its OWN block, which
runs the same number of epochs on the same data, and never against a number quoted from
this file.

The one thing worth carrying from the earlier audit is the SHAPE of the epoch trade, not
its values: epochs 8->10 cost 1.6 h and changed no verdict, while 3 epochs captured about
three quarters of the total improvement for 30% of the cost. That is why `n_epochs` is 3.

`Eij_*` and `msg_i_*` are new. They used to exist only as figures and as
`tmp_training/Eij.log`, so earlier runs of this exploration have them
missing from the analysis log — an entry from before this change that omits
them is not a failed slot.

`Eij_*` appears **only on conductance-generated data**. On a current
generator there is no `(E - v_i)` term to recover, the keys are absent, and
that absence is information rather than a fault. `msg_i_*` appears on both.

There is **no** `rollout_pearson_r`, no `test_R2`, no `conn_R2` key. Do not look for them.

**From the run directory**, `log/fly/<slot>/tmp_training/`. Name these paths exactly; do
not Glob:

| Path | What it carries |
| --- | --- |
| `../results/metrics.txt` | **A SECOND COPY OF EVERY HEADLINE METRIC**, one `key: value` per line — `msg_i_R2`, `Wij_R2`, `Wij_slope`, `Wij_slope`, `Eij_R2`, `Wij_R2`. The same code writes both this and the analysis log. It is inside the run directory, so it stays reachable when the shared analysis-log directory is not. **If the analysis log is unreadable, read this and report normally — never mark an objective PENDING while a readable copy exists.** |
| `metrics.log` | CSV: `iteration,connectivity_r2,vrest_r2_raw,tau_r2_raw,hidden_nnr_pearson,anchor_nnr_pearson,vrest_r2_clean,n_out_vrest,n_total_vrest,tau_r2_clean,n_out_tau,n_total_tau` |
| `Eij.log` | CSV: `iteration,rmse,r2,slope,n_edges` — the E_ij TRAJECTORY. Its final row should agree with `Eij_rmse` / `Eij_R2` in the analysis log; read the log for the value and this file for how it got there |
| `rollout_r.log` | CSV: `iteration,r,rmse,n_frames` |
| `Wij/raw_*.png` | 2x2 recovery panel for `W**2` vs the true conductance |
| `Eij/Eij_*.png` | 2x2 recovery panel for the per-edge reversal |
| `tau/tau_*.png`, `vrest/vrest_*.png` | 2x2 panels for tau and V_rest |
| `msgi/msgi_*.png` | 2x2 panel for the aggregated message — the non-degenerate one |
| `traces/rollout_*.png` | stacked free-run traces, green truth over black rollout |

Reading the 2x2 panels: top-left is a scatter with `R2: <outlier-free> [<raw>]`, `slope:
<outlier-free> [<raw>]`, `N`, `outliers: <percent>` and `RMSE`. The outlier-free number is
the headline and is the one the progress bar prints. Top-right and bottom-left are the
absolute and relative error histograms with median, FWHM and how many points fell off the
axis. Bottom-right is `|error|` per cell type, sorted by the type's mean true value.

**An empty folder is information, not a failure.** `Eij/` and `msgi/` fill only on
conductance-generated data; on a current-generated dataset they stay empty because there
is no reversal to recover.

---

## Explorable parameters

YOU MAY MODIFY ONLY THE PARAMETERS IN THIS TABLE.

| Parameter | Default | Suggested sweep | Notes |
| --- | --- | --- | --- |
| `lr_W` | 0.0009 | {3e-4, 6e-4, 9e-4, 2e-3} | learning rate on `W`, the square root of the conductance |
| `lr` | 0.0018 | {6e-4, 1.2e-3, 1.8e-3, 4e-3} | learning rate on `raw_tau`, `V_rest`, `E_exc`, `E_inh` |
| `coeff_W_L1` | 7.5e-05 | {0, 1e-5, 7.5e-5, 3e-4} | L1 on `W`. NOTE it acts on the square root, so it is an L1/2 penalty on the conductance |
| `coeff_W_L2` | 7.5e-07 | {0, 7.5e-7, 1e-5} | L2 on `W`, same caveat |
| `coeff_tau_L1` | 0.0 | {0, 1e-6, 1e-5, 1e-4, 1e-3} | L1 on `raw_tau` (pre-softplus). One of the two levers against the degeneracy |
| `coeff_tau_L2` | 0.0 | {0, 1e-6, 1e-5, 1e-4, 1e-3} | L2 on `raw_tau` |
| `coeff_V_rest_L1` | 0.0 | {0, 1e-6, 1e-5, 1e-4} | L1 on `V_rest` |
| `coeff_V_rest_L2` | 0.0 | {0, 1e-6, 1e-5, 1e-4} | L2 on `V_rest` |
| `batch_size` | 4 | {2, 4, 8} | time windows per gradient step. INTEGER |
| `w_init_mode` | `randn_scaled` | `randn_scaled`, `zeros`, `uniform_scaled`, `randn` | lowercase. `w_con` also exists but is untested here |
| `w_init_scale` | 1.0 | {0.25, 0.5, 1.0, 2.0} | bound = `scale / sqrt(n_edges)`, i.e. divided by sqrt(434,112) ~ 659 — the TOTAL edge count, not a fan-in |
| `data_augmentation_loop` | 100 | 50-400 | retune against the time target your prompt states; do not hard-code a threshold |

**Coefficient scale, because it is not obvious.** The tau and V_rest penalties are summed
over 13,741 neurons each; `coeff_W_*` is summed over 434,112 edges. That is a factor of
about 32, so a nominal `coeff_tau_L2` bites roughly 32 times harder per parameter than the
same number on W. Sweep them on a log ladder, never linearly.

**`regul_annealing_rate` MUST STAY 0.0.** The annealed coefficient is
`coeff * (1 - exp(-rate * epoch))`, which is **exactly zero at epoch 0**. With the default
rate of 0.5 every regularisation coefficient you set would be identically zero for the
whole of the first epoch — a sweep of a number that never reaches the loss. At rate 0 the
full coefficient applies from the first iteration. This is the only supported regime here.

---

## Frozen — do not touch

- `simulation.seed`, `training.seed` — the pipeline overwrites both every batch with
  `iteration*1000 + slot` and `iteration*1000 + slot + 500`. Log the values it reports.
- `training.n_epochs` — the pipeline overwrites it from `claude.n_epochs` every batch. Any
  edit is silently reverted, so a "training volume" block that sweeps it is a wasted block.
- `dataset` — must stay `flyvis_conductance_ion_noise_005_blank50_cv00` in every slot.
- `simulation.*` — `n_neurons`, `n_edges`, `n_frames`, `delta_t`, `noise_model_level`.
- Every `student_*` key — `_resolve_student_knobs` refuses them in recovery mode and
  raises rather than silently accepting.
- `coeff_g_phi_diff`, `coeff_g_phi_norm`, `coeff_g_phi_weight_L1`, `coeff_g_phi_weight_L2`,
  `coeff_g_phi_input_group_L1`, `coeff_f_theta_weight_L1`, `coeff_f_theta_weight_L2`,
  `coeff_f_theta_msg_diff`, `embedding_dim`, `lr_embedding`. **The base config carries
  non-zero values for several of these and they are all dead here** — a known-ODE student
  has no `g_phi` and no `f_theta` MLP for them to act on. They are inherited clutter from
  the GNN template. Changing them does nothing; report it if you see them cited anywhere.
- `use_gt_edges: true`, `mlp_precision: bf16`, `torch_compile: true`.

If you believe a frozen parameter must move, write the case in `user_input.md` and wait.

---

## Block structure

`n_iter_block` comes from the base config's `claude:` block; block number and a
`>>> BLOCK END <<<` marker are injected into your prompt. Do not restate an iteration
count here — read it from the prompt.

| Block | Mode | Focus | Parameters to scan | Ranges |
| --- | --- | --- | --- | --- |
| 1 | Robustness | Baseline variance | none — all 8 slots identical | Establish the CV of **`msg_i_R2` and `Wij_R2` first**, then `Wij_slope`, `Eij_R2`, `tau_R2`, `V_rest_R2`. Those two CVs are what every later verdict is measured against |
| 2 | Exploration | Learning rates | `lr_W`, `lr` | lr_W {3e-4, 6e-4, 2e-3}; lr {6e-4, 1.8e-3, 4e-3}. Also record the lr_W/lr ratio |
| 3 | Exploration | W initialisation | `w_init_mode`, `w_init_scale` | init {randn_scaled, zeros, uniform_scaled}; scale {0.25, 0.5, 2.0} |
| 4 | Exploration | tau scale prior | `coeff_tau_L2` then `coeff_tau_L1` | {0, 1e-5, 1e-4, 1e-3}. **This is the degeneracy block** |
| 5 | Exploration | V_rest scale prior | `coeff_V_rest_L2`, `coeff_V_rest_L1` | {0, 1e-6, 1e-5, 1e-4} |
| 6 | Exploration | W regularisation | `coeff_W_L1`, `coeff_W_L2` | W_L1 {0, 1e-5, 3e-4}; W_L2 {0, 1e-5} |
| 7 | Exploration | Batch and training volume | `batch_size`, `data_augmentation_loop` | bs {2, 4, 8}; DAL against the prompt's time target |
| 8 | Exploration | Free — combine | any of the above | Consolidate the best of blocks 2-7 |
| 9 | Robustness | Final validation | none — all 8 slots at the champion | Confirm CV and no catastrophic seed |

**Slot 0 ratchets.** In blocks 2 onward, slot 0 is the best configuration found so far, not
the original baseline. The control must move with the evidence or you will spend nine
blocks comparing against a config you already know is beaten.

Per-block notes:

- **Block 1** exists because every later comparison is against this variance. Record the CV
  of BOTH objectives. If the CV of `Wij_R2` exceeds 10%, single-slot
  differences below that are noise and you must say so in every subsequent entry. The CV of
  `msg_i_R2` is what sets whether the 0.02 fit floor is meaningful — if the fit itself
  varies by more than 0.02 between identical slots, widen the floor to that CV and say so.
- **Block 4 is the point of the exploration.** The prediction is that an L2 on `raw_tau`
  introduces a preferred scale along the rescaling direction, so it should move `Wij_slope`
  toward 1.0 from its observed 2.39 and lift both `Wij_R2` and
  `Eij_R2`. **The success signature is stage-2 metrics rising while `msg_i_R2` does
  not move** — that is the scale being pinned rather than the message being refitted, and
  it is exactly the two-objective split this file is built on. If `msg_i_R2` falls as the
  coefficient rises, the prior is not pinning the scale, it is fighting the fit: back off.
  A flat `Wij_slope` and flat `Eij_R2` at every coefficient is the **null result**, and
  it is a real finding — it would say the identifiability limit, not the optimiser, is the
  bottleneck. Document it as such rather than sweeping harder.
- **Block 6**: `coeff_W_L1` acts on the square root of the conductance, so its effect on
  the conductance is not the L1 you would expect. A value that drove W to zero would show
  as `Wij_R2` collapsing toward 0 with the relative-error panel spiking at exactly
  -1.0. Back off immediately if you see it.

---

## Acceptance

Applied in this order. Stage 1 is a gate: a slot that fails it is never ranked on stage 2,
because a connectivity number read off a model that no longer reproduces the message is
measuring noise.

**Stage 1 — the fit gate, on `msg_i_R2`:**

| Verdict | Rule |
| --- | --- |
| Fit-Broken | `msg_i_R2` more than 0.02 below the block's slot-0 control — REJECT, do not rank, do not carry forward |
| Fit-Collapsed | `msg_i_R2` < 0.50 in absolute terms — the model has stopped passing the message; report it as a finding and return to the last configuration that held |
| Fit-OK | otherwise — proceed to stage 2 |

**Stage 2 — identification, on `Wij_R2` (thresholds set against the 0.343
and `Wij_slope` 2.39 this file was written from, NOT against 1.0):**

| Verdict | Rule |
| --- | --- |
| Strong | all 8 slots `Wij_R2` >= 0.60 AND `Wij_slope` within [0.7, 1.4], CV < 3% |
| Improving | mean `Wij_R2` >= 0.45 and above the block control by more than the block-1 CV |
| Flat | within the block-1 CV of the control — a real result if it holds across a block, not a failure |
| Regressed | mean `Wij_R2` below the control by more than the block-1 CV |

**There is no "Catastrophic" grade on `Wij_R2` any more, and its removal is the
point.** The old ladder rejected anything below 0.50, and every configuration reachable
here sits between -1.2 and -4.9, so it graded improvements and regressions identically.
Raw `Wij_R2` is recorded in every entry and is the number to quote; it is not a
ranking key.

Three extra rules specific to this experiment:

- **Both objectives, every entry.** State `msg_i_R2` and `Wij_R2`
  together, and say which of the two the slot moved. "Improved `Wij_R2`
  0.34 -> 0.41 at unchanged `msg_i_R2` 0.947" is a stage-2 win; "`msg_i_R2` 0.947 -> 0.83
  with `Wij_R2` 0.34 -> 0.52" is NOT a win, it is a broken fit whose
  connectivity number should not be believed.
- **`Wij_slope` is the direct read of progress.** It says how far along the degenerate valley
  the run sits, 1.0 being the truth. Watching it move 2.39 -> 1.8 -> 1.3 across a block is
  the clearest possible evidence that a prior is pinning the scale, and it is more
  legible than any R2 because it has physical meaning: 2.39 means the learned conductance
  is 2.39x the true one.

- **Trajectory check.** Read the `connectivity_r2` column of `metrics.log` and record both
  the iteration of its peak and `(final - peak) / peak`. Treat any slot with
  `final / peak < 0.95` as disqualified even if the final number looks acceptable — a late
  collapse is a real failure mode here.
- **Degeneracy check.** In every entry, record `Wij_R2`, `Eij_R2` and
  `msg_i_R2` together. A configuration that improves one of the first two while
  destroying the other has not improved recovery; it has moved along the degenerate
  valley, and `msg_i_R2` holding steady across that move is the proof — the message is
  the product the data constrain, so it does not care how the factorisation shifted.

---

## Iteration workflow

For each slot, in this order:

1. Read the analysis log for the slot and extract every metric named in the Metrics table.
   `Wij_R2`, `Eij_R2`/`Eij_slope`/`Eij_rmse` and
   `msg_i_R2`/`msg_i_slope` are all in there — one read, no figure-scraping and no
   parsing of training logs by hand.
2. Read `tmp_training/metrics.log` for the peak of `connectivity_r2` and how far the
   final value fell from it, and `tmp_training/Eij.log` for the shape of the
   E_ij trajectory. Both are about the trajectory; the final values came from step 1.
3. Compare `Wij_R2` against `msg_i_R2` and say which way they disagree, if they
   do. `tmp_training/Wij/raw_*.png` and `tmp_training/msgi/msgi_*.png` show the same two
   quantities as scatters if a number alone does not explain what happened.
4. Write one entry per slot to the analysis log **and** to the working memory. The heading
   must be exactly `## Iter N: <short title>` — the resume mechanism parses that pattern
   and a different heading will break `--resume`.
5. Edit all slot configs for the next batch under the causality rule stated in your prompt:
   slot 0 is the parent unchanged, every other slot changes exactly one parameter.

Entry template:

```markdown
## Iter N: <one-line hypothesis>
- Slot: S | seeds: sim=<value> train=<value>
- Changed: <parameter> <old> -> <new>   (slot 0: unchanged control)
- STAGE 1 msg_i_R2: <value>  (control <value>, delta <value>)  -> <Fit-OK | Fit-Broken | Fit-Collapsed>
- STAGE 2 connectivity_R2_scaled: <value>  w_scale: <value>  connectivity_pearson_r: <value>
- Eij_R2: <value>   Eij_slope: <value>   Eij_rmse: <value>
- connectivity_R2 (raw, quoted not ranked): <value>  (peak <value> at iter <value>, final/peak <value>)
- tau_R2: <value>   V_rest_R2: <value>
- onestep_pearson: <value>   rollout_pearson: <value>
- training_time_min: <value>
- Verdict: <Strong | Improving | Flat | Regressed | Fit-Broken | Fit-Collapsed | Disqualified-late-collapse>
- Reading: <two sentences: WHICH OF THE TWO OBJECTIVES moved, and whether w_scale moved toward 1.0>
```

---

## Block boundaries

At `>>> BLOCK END <<<`:

1. Write a block summary into `## Previous Block Summaries` in memory.
2. Promote any finding that survived a robustness check to `### Established Principles`;
   move any refuted one to `### Falsified Hypotheses` with the evidence that killed it.
3. Update the comparison table with every metric column, not just `Wij_R2`.
4. Write the block's winner to
   `config/fly/flyvis_conductance_noise_005_conductance_knownode_winner.yaml`, with a
   comment header giving the iteration, the four headline metrics and the E_ij RMSE.
   `config/fly/` is the only config directory that exists — do not write to
   `config/larva/`, `config/drosophila_cx/` or `config/zebrafish_oculomotor/`.
5. State the next block's hypothesis in `## Current Block`.

---

## Working memory structure

```markdown
# Working Memory: flyvis_conductance_noise_005_conductance_knownode_cv00

## Paper Summary (update at every block boundary)

## Knowledge Base (accumulated across all blocks)

### Results Comparison Table
| Iter | Config summary | msg_i_R2 (S1) | conn_R2_scaled (S2) | w_scale | CV% | Eij_R2 | Eij_rmse | conn_R2 raw | tau_R2 | V_rest_R2 | rollout_pearson | Verdict | Hypothesis tested |
| ---- | -------------- | ------------- | ------------------- | ------- | --- | ----------- | ------------- | ----------- | ------ | --------- | --------------- | ------- | ----------------- |

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
set all four slots to the baseline, and state in memory that block 1 is a robustness
measurement whose only purpose is to establish the variance every later block is judged
against.

Launch:

```bash
python GNN_LLM+.py -o generate_train_test_plot_Claude \
    flyvis_conductance_noise_005_conductance_knownode_cv00 \
    iterations=144 --cluster --node l4 --resume
```

144 iterations is the 9 blocks of this file at the base config's
`n_iter_block: 16`, run 8 slots at a time — 18 batches, two per block.

The base config must carry a `claude:` block; `--cluster` needs `data_paths.json` at the
repo root with `cluster_root_dir` pointing at the cluster checkout.
