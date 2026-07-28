# Conductance-based twin of the pretrained flyvis model, and a GNN that can fit it

Target branch: `neurips2026`

## Why

The pretrained flyvis networks deliver synaptic input as a **current**, so a
synapse's contribution depends only on the presynaptic voltage:

```
tau_i dv_i/dt = -(v_i - v_rest_i) + sum_j  s_ij alpha_ij N_ij relu(v_j)  + e_i
```

Real synapses open conductances towards a reversal potential, which makes the
contribution depend on the **postsynaptic** voltage too:

```
tau_i dv_i/dt = -(v_i - v_rest_i) + sum_j  G_ij relu(v_j) (E_ij - v_i)  + e_i
```

Two consequences. The model *shunts*: gain and membrane time constant are both
divided by `1 + sum_j G_ij relu(v_j)`, so both become input-dependent. And the
voltage is confined to the convex hull of the reversal potentials, so free
rollouts cannot diverge.

This adds that model as a ground-truth simulator, derived from the pretrained
current-based model rather than retrained, plus a GNN whose message function can
represent it. The point is a harder identifiability target: a message function of
`v_j` alone is structurally unable to express `(E - v_i)`, so this separates "the
GNN recovers the connectome" from "the GNN recovers the connectome *because the
messages happened to be presynaptic-only*".

## What the twin is

Same connectome, same parameter sharing, same number of free parameters as the
model it reparametrizes: **734** — one conductance scale per (presynaptic type,
postsynaptic type) pair, one time constant and one resting potential per cell
type. Only the synaptic term changes form. `G >= 0` always; a synapse's sign
lives in `E_rev`, not in `G`.

`FlyVisConductanceODEParams` subclasses `FlyVisODEParams`, so it inherits the
analysis interface (`has_tau`, `gt_vrest`, `derive_tau`, …) and keeps `W` as a
*derived* field — the small-signal weight `G (E - v_rest)`. Every existing
consumer of `ode_params.W` therefore keeps working untouched.

Reversal potentials sit a margin outside the voltage range the pretrained model
visits (default `(0.4, 1.0)` voltage spans, inhibitory then excitatory, mirroring
the inhibitory driving force being about half the excitatory one in real
neurons). That margin is the one knob: as it grows the twin degenerates
continuously back to the current-based model, since `G (E - v_i) -> s alpha N` as
`E -> infinity` with `G ~ 1/E`.

### How well it matches

Free rollouts from each model's own steady state, on held-out naturalistic clips:

| | extent 8 |
|---|---|
| unexplained synaptic-current variance | 0.27 % |
| voltage R², the fit's own held-out clips | 0.972 |
| voltage R², an independent clip sample | 0.960 |
| median total synaptic conductance G | 0.084 |

R² is flat across 1.5 s windows (0.97, 0.95, 0.98, 0.94, 0.95, 0.97, 0.94, 0.95),
so the residual is an error floor rather than compounding divergence. It is
genuinely shunting where it should be: `G ≈ 0.6` in Mi1 and Am, `0.4–0.5` in
L1/L2 — those cells run with gain and membrane time constant divided by 1.5 or
more — while the median across cell types is only ~0.08, so the effect is
concentrated in the strongly driven early pathway.

Two R² rows are quoted because they differ only in which clips were drawn; the
spread across clip samples (0.01–0.04) is larger than most differences this PR
discusses, so third decimals should not be read closely.

### One property to know about before interpreting results

Stage 2 constrains `G >= 0` while the sign of the driving force is fixed by the
connectome. Where a shared group's contribution is better explained with the
opposite polarity — or is collinear with another group onto the same cell type —
the non-negative least squares clamps it to zero. Measured at extent 8:

**~120 of 604 shared groups (20 %) are driven to zero, covering ~16 % of edges**
(128 / 17.8 % in the committed extent-8 derivation, which used fewer clips).

Those edges carry only **~3.9 %** of the teacher's total `|W|` (their median
`|W|` is ~260× smaller than the rest), which is why voltage R² is still 0.97 —
the fit prunes mostly weak synapses. But it means the twin is not merely "the
same network with conductances": it is also effectively sparser. This is
self-consistent for the metrics (`connectivity_r2` compares against the twin's
own `W`, which encodes the zeros), but **numbers from twin rollouts are not
directly comparable with published numbers from current-based rollouts.**

`conductance_pruning_stats` records this in every derivation's `meta.json`.

**More teacher data does not reduce it.** Sweeping the fitting set from 4 to 96
clips — 24x more data, 3 s to 73 s of teacher rollout, spanning many Sintel
scenes — leaves the pruning flat:

| clips | pruned groups | pruned edges | their share of \|W\| | held-out R² |
|---|---|---|---|---|
| 4 | 118 / 604 | 18.1 % | 3.62 % | 0.926 |
| 12 | 120 / 604 | 16.1 % | 4.05 % | 0.941 |
| 32 | 120 / 604 | 16.1 % | 3.82 % | 0.942 |
| 96 | 123 / 604 | 16.3 % | 3.89 % | 0.943 |

That rules out sampling noise. What *does* move it is the reversal margin, i.e.
how far the twin sits from the current-based model it degenerates to:

| margin | pruned groups | pruned edges | share of \|W\| | unexplained | R² |
|---|---|---|---|---|---|
| 0.5 | 140 / 604 | 21.2 % | 6.18 % | 0.0063 | 0.887 |
| 1.0 (default) | 120 / 604 | 16.1 % | 3.82 % | 0.0022 | 0.942 |
| 2.0 | 106 / 604 | 12.7 % | 2.97 % | 0.0006 | 0.977 |
| 4.0 | 92 / 604 | 11.2 % | 2.05 % | 0.0001 | 0.993 |
| 16.0 | 72 / 604 | 9.5 % | 0.90 % | 0.0000 | 0.9995 |

So most of the pruning is the driving-force misspecification: a constant weight
and a `(E - v_i)`-modulated one cannot match exactly, and the fit absorbs the
mismatch by zeroing columns that are collinear within a postsynaptic cell type.
As the twin degenerates to the teacher the pruning shrinks and the residual goes
to zero.

It does not vanish, though: ~72 groups (9.5 % of edges, 0.9 % of `|W|`) are still
zeroed at margin 16 where the fit is essentially exact. That floor is not caused
by the conductance reparametrization at all. Part of it is the teacher's own
structure — 31 groups (5 %) carry exactly zero weight in the pretrained model, so
zeroing them is correct — and the rest is ordinary NNLS sparsity under a collinear
design.

Raising `--ridge` is *not* a good lever: it does reduce the pruned group count
(120 → 50) but wrecks the fit (R² 0.94 → 0.64), because it pulls the solution
towards the closed-form prior, which is a poor model on its own.

Untested: stimulus *diversity* as opposed to quantity. The sweep above varies how
much Sintel the fit sees, not what kind; a DAVIS or mixed ensemble would drive
cell types into different relative activation regimes and could genuinely
decorrelate the design. The invariance across 24x more data spanning many scenes
suggests the effect is structural rather than a sampling artifact, but that is an
argument, not a measurement.

**If the pruning is a problem for your experiment, the margin is the knob** —
margin 2.0 buys R² 0.977 at 12.7 % pruned, margin 4.0 buys 0.993 at 11.2 %, at
the cost of a twin that shunts less (which is the whole point of the model, so
this is a real trade rather than a free win).

## Running the four experiments

flyvis is needed only to read the pretrained parameters and render Sintel — both
public API, so a stock `pip install flyvis` is enough. Every rollout, teacher and
twin, fitting and evaluation, runs on connectome-gnn's own integrators.

```bash
# extent 8  (13,741 neurons, 434,112 edges)
python GNN_Main.py -o generate_train_test_plot flyvis_twin_gnn_A_e8
python GNN_Main.py -o generate_train_test_plot flyvis_twin_gnn_cond_e8

# extent 15 (45,669 neurons, 1,513,231 edges — all 721 retinotopic columns)
python GNN_Main.py -o generate_train_test_plot flyvis_twin_gnn_A_e15
python GNN_Main.py -o generate_train_test_plot flyvis_twin_gnn_cond_e15
```

**The twin is derived on first use.** Nothing needs to be downloaded or
committed: if `graphs_data/fly/conductance_twin_extent{8,15}` is absent, data
generation derives it, logs that it is doing so, and saves it. Subsequent runs
reuse it. That keeps the submission branch small — `graphs_data/` is gitignored
here, so no twin is committed.

Derivation happens from the generator's *own* network and *own* stimulus dataset.
That is the part worth not changing: a twin is only valid for the connectome it
was derived on, and deriving from the network the data will actually be generated
with removes any chance of an extent or rendering mismatch. It costs a few minutes
once per extent (plus a one-off Sintel render if that extent has not been
rendered before).

To derive or re-derive by hand:

```bash
python -m connectome_gnn.generators.flyvis_conductance_fit \
    --extent 8 --n-concat 4 --out graphs_data/fly/conductance_twin_extent8
python -m connectome_gnn.generators.flyvis_conductance_fit \
    --out graphs_data/fly/conductance_twin_extent15   # extent from the checkpoint
```

Each output folder holds `ode_params.pt` (the tensors the simulator consumes),
`nodes.csv`, `edges.parquet`, `meta.json` with the fit diagnostics, and a README
with a runnable snippet. **These need no flyvis to simulate** — a test enforces
that by blocking the `flyvis` import and then stepping the ODE.

### Checking a twin

```bash
python -m connectome_gnn.generators.verify_flyvis_conductance \
    --extent 8 --params graphs_data/fly/conductance_twin_extent8 --n-concat 8
```

Reports two things. First, connectome-gnn's `FlyVisODE` against flyvis' own
`Network` on the current-based model: **1.2e-06 max, 6.3e-08 rms** (float32
rounding — the two associate the same expression differently). That agreement is
what licenses deriving the twin entirely inside this repo. Second, twin fidelity:
R², per cell type, resolved into time windows, plus the shunting distribution.

`plot_conductance_traces` draws the central neuron of all 65 cell types with both
variants overlaid, if you want to eyeball trace shape.

### How the data/model split works

```yaml
simulation:
  ground_truth_model: flyvis_conductance   # which ODE generates the data
  all_columns: false                       # extent 8; true gives extent 15
graph_model:
  signal_model_name: flyvis_A              # which GNN learns it
```

`ground_truth_model` defaults to `graph_model.signal_model_name`, so every
existing config behaves exactly as before.

| | experiments 1 & 3 | experiments 2 & 4 |
|---|---|---|
| config | `flyvis_twin_gnn_A_e{8,15}` | `flyvis_twin_gnn_cond_e{8,15}` |
| GNN | `flyvis_A` | `flyvis_conductance_factorized` |
| message | `W[e] · g_phi(v_j, a_j)²` | `(W[e] · g_phi(v_j, a_j))² · (e_psi(a_i, a_j) − v_i)` |
| sees `v_i`? | no | yes |
| can express `(E − v_i)`? | no | yes, explicitly |

`flyvis_conductance` (unfactorized, `msg = W[e] · g_phi(v_i, v_j, a_i, a_j)`,
`input_size: 6`) is also registered, as the least-assuming middle ground: it can
represent a driving force but is not told one exists.

Both conductance model types deliberately **do not square** the message.
`flyvis_A`/`flyvis_B` square `g_phi` to force positive messages; that would
forbid the sign change an inhibitory driving force needs. The factorized variant
puts positivity where it belongs, on the conductance factor.

## Identifiability metrics

`metrics.log` keeps its 12 columns and gains two: `conductance_r2,reversal_r2`.
`metrics.png` gains `R²_G` and `R²_E` curves and badge entries. Both are `nan`
for models that cannot factorize their message (`flyvis_A`, `flyvis_conductance`)
and for datasets with no conductance ground truth, so existing runs are
unaffected.

- **`connectivity_r2`** (unchanged) scores the learned per-edge `W` against
  `ode_params.W` = `G (E − v_rest)`, the small-signal weight — what a
  presynaptic-only message function can recover at best, and the fair common
  ground for all four runs.
- **`conductance_r2` / `reversal_r2`** score the factorized GNN's
  `edge_conductance()` against `conductance_params.pt` directly, asking the
  sharper question: did it recover the two factors, or only their product?

Both use `compute_r_squared_NSE`, matching `connectivity_r2`, `tau_r2` and
`vrest_r2` on this branch, so all of them are read on the same footing.

Two cautions. `reversal_r2` is a two-group separation, not a fit to a continuum:
`E` takes exactly two values, so it says whether the model told excitatory and
inhibitory edges apart. And a flat prediction reports `nan`, not `0.0` — an
untrained model has identical embeddings on every edge, so its learned reversal
potential is constant up to batched-matmul rounding, and fitting a line through
that noise produced an R² of anything at all (0.99999997 in one test run). `0.0`
would have read as "wrong" rather than "nothing to compare yet".

The prediction worth testing: experiments 1 and 3 are capped by the spread of
`(E − v_i)` across the operating range, and 2 and 4 are not.

## Compatibility

- `simulation.ground_truth_model` and `conductance_twin_path` are new, both
  defaulting to the previous behaviour.
- `ODEParamsBase.save/load` gain a `filename` argument, defaulting to
  `ode_params.pt`.
- `metrics.log` gained two trailing columns. All five write sites were widened —
  two go through the shared f-string, one is a hand-written `nan,0,0,nan,0,0`
  tail, two are NGP quick-refresh rows with their own layout — and a check
  confirms every row matches the 14-column header. Three would have gone ragged
  otherwise. Both readers (`plot_signal_loss`, `plot_metrics`) index positionally
  with bounds guards, so they tolerate the extra columns.
- `n_extra_null_edges` and `ablation_ratio` raise `NotImplementedError` under a
  conductance ground truth rather than being silently ignored: they rewrite
  `ode_params.W` before the ODE is built and so would never reach the simulated
  dynamics. Derive an ablated twin instead, or use `edge_removal_ratio`, which is
  applied after the rollout.
- The twin configs use `delta_t: 0.01`, not the `0.02` most fly configs use, and
  this is not cosmetic. dt enters the twin in exactly one place — the floor on
  `tau/(1+G)` that stops forward Euler overshooting — so the ODE itself is
  dt-free. Measured over a 1.5 s clip at extent 8, the fraction of neurons whose
  `tau/(1+G)` falls below the step:

  | dt | floor binds for |
  |---|---|
  | 0.02 | **75.5 %** |
  | 0.01 | 0.000 % |
  | 0.005 | 0.000 % |

  At 0.01 the floor never fires, so the twin carries no dt dependence at all. At
  0.02 it fires for three quarters of the population and the floored and exact
  schemes disagree by rms 0.169 against a voltage std of 0.59 — a different
  model, not a rounding difference. The shortest time constant in the connectome
  is 0.0193, which is why 0.02 lands on the wrong side.

  `integration="exponential"` removes the floor entirely: it returns the
  derivative reproducing exact relaxation over the step,
  `v(t+dt) = v_inf + (v - v_inf) exp(-dt/tau_eff)`, unconditionally stable at any
  step without distorting `tau_eff`. The default stays `euler_floored` so the
  numbers above stand; `floor_binding_fraction()` reports whether the distinction
  matters for a given run.

  One caveat on "dt-independent": both schemes converge first-order to the same
  limit, so neither is exact at finite dt, and the twin is fitted to reproduce
  *flyvis' forward-Euler trajectories*. The teacher's own discretization, not
  ours, sets the floor on how dt-free the object can be.

## Verified vs. not

Verified: the twin derivation end to end; integrator agreement against flyvis;
twin fidelity, drift and shunting on independent clip samples; simulation from
the exported files with `flyvis` unimportable; all four configs loading and
building their models; data generation through `graph_data_generator` with the
conductance ground truth; and auto-derivation firing on a missing twin and being
reused on the next run. 25 new tests; the suite is 184 passed / 6 failed, and
those 6 fail identically on untouched `neurips2026` (4 in `test_metrics.py`
reference the removed `compute_r_squared`, 2 in `test_utils.py::TestSortKey`).
Ruff error counts are unchanged from baseline on every file touched.

Not run: the four training runs, so no `connectivity_r2`, `conductance_r2` or
`reversal_r2` numbers are claimed here. Extent 15 is verified for the derivation
and config paths but its data generation has not been run end to end.
