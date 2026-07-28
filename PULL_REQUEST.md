# Conductance-based twin of the pretrained flyvis model

Target branch: `neurips2026`

## What the PR addresses

The pretrained flyvis networks deliver synaptic input as a current, so a
synapse's contribution depends only on the presynaptic voltage:

```
tau_i dv_i/dt = -(v_i - v_rest_i) + sum_j  s_ij alpha_ij N_ij relu(v_j)  + e_i
```

Biological synapses open conductances towards a reversal potential, which makes
the contribution depend on the postsynaptic voltage as well:

```
tau_i dv_i/dt = -(v_i - v_rest_i) + sum_j  G_ij relu(v_j) (E_ij - v_i)  + e_i
```

Gain and membrane time constant are then both divided by
`1 + sum_j G_ij relu(v_j)`, so both become input-dependent (shunting), and the
voltage is confined to the convex hull of the reversal potentials, which makes
free rollouts unconditionally stable.

This PR adds the conductance model as a ground-truth simulator, derived from the
pretrained current-based model rather than retrained, together with a GNN whose
message function can represent it. A message function of `v_j` alone cannot
express `(E - v_i)`, so the twin is a stricter identifiability target and
separates connectome recovery from the presynaptic-only structure of the message.

The twin retains the connectome, the parameter sharing and the parameter count of
the model it reparametrizes: 734 free parameters, being one conductance scale per
(presynaptic type, postsynaptic type) pair plus one time constant and one resting
potential per cell type. `G >= 0` throughout; synaptic sign resides in `E_rev`.

Fidelity to the pretrained model at extent 8, free rollouts from each model's own
steady state on held-out naturalistic clips:

| quantity | value |
|---|---|
| unexplained synaptic-current variance | 0.27 % |
| voltage R², fit's own held-out clips | 0.972 |
| voltage R², independent clip sample | 0.960 |
| R² per 0.8 s window | 0.97, 0.95, 0.98, 0.94, 0.95, 0.97, 0.94, 0.95 |
| median total synaptic conductance G | 0.084 |
| G in Mi1 / Am / L1 / L2 | 0.60 / 0.59 / 0.44 / 0.47 |

The flat window profile indicates an error floor rather than compounding
divergence. Shunting is concentrated in the strongly driven early pathway, where
gain and effective time constant are divided by 1.4–1.6, against a median across
cell types of 0.08.

## How to use it

flyvis is required only to read the pretrained parameters and render Sintel, both
public API. All rollouts, teacher and twin, fitting and evaluation, use
connectome-gnn's integrators.

### Experiments

```bash
# extent 8 (13,741 neurons, 434,112 edges)
python GNN_Main.py -o generate_train_test_plot flyvis_twin_gnn_A_e8
python GNN_Main.py -o generate_train_test_plot flyvis_twin_gnn_cond_e8

# extent 15 (45,669 neurons, 1,513,231 edges, all 721 retinotopic columns)
python GNN_Main.py -o generate_train_test_plot flyvis_twin_gnn_A_e15
python GNN_Main.py -o generate_train_test_plot flyvis_twin_gnn_cond_e15
```

| | `flyvis_twin_gnn_A_e{8,15}` | `flyvis_twin_gnn_cond_e{8,15}` |
|---|---|---|
| GNN | `flyvis_A` | `flyvis_conductance_factorized` |
| message | `W[e] · g_phi(v_j, a_j)²` | `(W[e] · g_phi(v_j, a_j))² · (e_psi(a_i, a_j) − v_i)` |
| takes `v_i` | no | yes |
| can express `(E − v_i)` | no | yes, explicitly |

`flyvis_conductance` is also registered, with
`msg = W[e] · g_phi(v_i, v_j, a_i, a_j)` and `input_size: 6`, admitting a driving
force without imposing one. Neither conductance model squares the message, since
squaring precludes the sign change an inhibitory driving force requires; the
factorized variant constrains positivity on the conductance factor instead.

### Obtaining a twin

The twin is derived on first use. If `graphs_data/fly/conductance_twin_extent{8,15}`
is absent, data generation derives it from the network and stimulus dataset of that
run, logs the derivation and saves it; later runs load it. `graphs_data/` is
gitignored, so no twin is committed. Derivation takes a few minutes per extent,
plus a one-off Sintel render for extents not yet rendered.

Manual derivation:

```bash
python -m connectome_gnn.generators.flyvis_conductance_fit \
    --extent 8 --n-concat 4 --out graphs_data/fly/conductance_twin_extent8
```

Each output folder holds `ode_params.pt`, `nodes.csv`, `edges.parquet`,
`meta.json` with the fit diagnostics, and a README. Simulating these requires no
flyvis; a test enforces this by blocking the import and stepping the ODE.

### Verification

```bash
python -m connectome_gnn.generators.verify_flyvis_conductance \
    --extent 8 --params graphs_data/fly/conductance_twin_extent8 --n-concat 8
```

This reports, first, connectome-gnn's `FlyVisODE` against flyvis' `Network` on the
current-based model at 1.2e-06 max and 6.3e-08 rms, i.e. float32 rounding, which
is the basis for deriving the twin inside this repository; and second, twin
fidelity by R², per cell type and per time window, with the shunting
distribution. `plot_conductance_traces` overlays the central neuron of all 65 cell
types for both models.

### Metrics

`metrics.log` gains two columns, `conductance_r2` and `reversal_r2`, and
`metrics.png` the corresponding curves. Both are `nan` for models that cannot
factorize their message and for datasets without a conductance ground truth.

- `connectivity_r2`, unchanged, scores learned `W` against `ode_params.W` =
  `G (E − v_rest)`, the small-signal weight. This is the common target across all
  four runs and the best a presynaptic-only message can attain.
- `conductance_r2` and `reversal_r2` score `edge_conductance()` against
  `conductance_params.pt`, separating recovery of the two factors from recovery of
  their product.

All three use `compute_r_squared_NSE`, as `tau_r2` and `vrest_r2` do. `E` takes
two values, so `reversal_r2` measures separation of excitatory from inhibitory
edges rather than fit to a continuum. Flat predictions return `nan`: an untrained
model has identical embeddings on every edge, making its reversal potential
constant up to rounding, and a linear fit through that noise yields an
uninformative value.

### Integration step

The twin configs set `delta_t: 0.01`. dt enters the model only through the floor
on `tau/(1+G)` that prevents forward-Euler overshoot; the ODE is otherwise
dt-free. Fraction of neurons whose `tau/(1+G)` falls below the step, over a 1.5 s
clip at extent 8:

| dt | floor active |
|---|---|
| 0.02 | 75.5 % |
| 0.01 | 0.000 % |
| 0.005 | 0.000 % |

The shortest time constant in the connectome is 0.0193. At 0.01 the floor never
activates and the simulated dynamics carry no dt dependence. At 0.02 it activates
for three quarters of the population, and the floored and exact schemes then
differ by rms 0.169 against a voltage standard deviation of 0.59.

`integration="exponential"` removes the floor, returning the derivative that
reproduces `v(t+dt) = v_inf + (v − v_inf) exp(−dt/tau_eff)`, unconditionally
stable at any step and reducing to `(v_inf − v)/tau_eff` as dt → 0. The default is
`euler_floored`; `floor_binding_fraction()` reports whether the distinction is
active. Both schemes are first-order consistent with the same ODE, and the twin is
fitted against flyvis' forward-Euler trajectories, so the teacher's discretization
bounds how dt-independent the twin can be.

### Reversal margin

Reversal potentials are placed a margin outside the voltage range the pretrained
model visits, as `(margin_inh, margin_exc)` in units of that range. The margin
controls how far the twin sits from the current-based model it degenerates to as
the margin grows, and thereby trades fidelity against shunting. Measured at
extent 8, 32 fitting clips, with margins scaled from the `(0.4, 1.0)` default:

| margin | E_inh | E_exc | (E_exc − v̄)/span | R² | pruned edges | median G | G in Mi1 | G in L1 |
|---|---|---|---|---|---|---|---|---|
| 0.5 | −5.77 | 11.40 | 1.09 | 0.908 | 22.5 % | 0.117 | 0.833 | 0.610 |
| 1.0 | −7.79 | 16.45 | 1.59 | 0.957 | 17.3 % | 0.082 | 0.579 | 0.381 |
| 2.0 | −11.83 | 26.55 | 2.59 | 0.985 | 13.7 % | 0.051 | 0.358 | 0.213 |
| 4.0 | −19.91 | 46.75 | 4.59 | 0.996 | 12.3 % | 0.030 | 0.199 | 0.114 |

In cortical and fly neurons the excitatory driving force is on the order of 50 mV
against a voltage excursion on the order of 20 mV, a ratio near 2.5, which
corresponds to margin 2.0 in the table. Margin 0.5 and 1.0 compress the driving
force below the physiological ratio; margin 4.0 exceeds it and leaves shunting
marginal, with `G` in Mi1 at 0.199, a gain modulation of only 1.2.

### Effective sparsification

Stage 2 constrains `G >= 0` while the sign of the driving force is fixed by the
connectome, so groups whose contribution would require the opposite polarity, or
which are collinear with another group onto the same postsynaptic cell type, are
clamped to zero. At extent 8 and the default margin, roughly 120 of 604 shared
groups (20 %) are zeroed, covering about 16 % of edges which carry 3.8 % of the
teacher's total `|W|`. The twin is therefore also effectively sparser than the
pretrained model. `connectivity_r2` compares against the twin's own `W`, which
encodes those zeros, so the metric is self-consistent, but values obtained on twin
rollouts are not directly comparable with values obtained on current-based
rollouts.

The effect does not depend on the amount of teacher data: from 4 to 96 fitting
clips, 3 s to 73 s of rollout, the pruned fraction stays within 16.1–18.1 % of
edges and 3.6–4.1 % of `|W|`, while held-out R² rises only from 0.926 to 0.943. It
depends instead on the margin, falling to 12.3 % at margin 4.0 where the fit is
essentially exact. A residual 9.5 % of edges is zeroed even at margin 16: 31
groups (5 %) carry exactly zero weight in the pretrained model, and the remainder
is NNLS sparsity under a collinear design. Raising `--ridge` lowers the pruned
group count from 120 to 50 but reduces R² from 0.94 to 0.64.

`conductance_pruning_stats` writes these figures into each derivation's
`meta.json`.

## How it is implemented

### Simulator

- `generators/flyvis_conductance_ode.py` — the ODE, aggregated with two
  `scatter_add` calls and no PyG dependency, with the two integration schemes.
- `generators/ode_params.py` — `FlyVisConductanceODEParams`, a subclass of
  `FlyVisODEParams` adding `G`, `E_rev` and `input_index` and retaining `W` as the
  derived small-signal weight, kept current by `refresh_effective_weights()`.
  Subclassing preserves the analysis interface (`has_tau`, `gt_vrest`,
  `derive_tau`) and every existing consumer of `ode_params.W`. `save` and `load`
  gain a `filename` argument so a twin can sit beside the current-based
  parameters.

### Derivation

`generators/flyvis_conductance_fit.py`, in three stages:

1. Closed form. Linearizing `(E - v_i)` about the mean postsynaptic voltage gives
   `G = alpha N / |E - mean_v|`, exact to first order and exact everywhere as the
   reversal potentials recede.
2. Convex. Along teacher trajectories the synaptic current is linear in `G`, and
   the shared conductances decouple across postsynaptic cell types, so the optimum
   follows from one non-negative least-squares problem per cell type, accumulated
   in double precision and solved by Cholesky factorization followed by NNLS. This
   stage accounts for nearly all of the fit.
3. Truncated backpropagation through time on free rollouts, correcting the drift
   that the teacher-forced stage 2 cannot observe. Conductances and time constants
   are stepped in log space. Gradients are enabled explicitly because the data
   generator disables them globally.

`ensure_conductance_twin()` loads or derives, using the caller's network and
stimulus dataset so that extent and rendering cannot diverge from the run. When no
dataset is supplied, as in a DAVIS run, one is built and the extent must be given
explicitly.

### Model

`models/flyvis_conductance_gnn.py` registers `flyvis_conductance` and
`flyvis_conductance_factorized`, following `NeuralGNN` in embedding handling, `W`
initialisation modes and node update. `edge_conductance()` exposes the learned
per-edge conductance and reversal potential of the factorized variant.

### Pipeline

`simulation.ground_truth_model` selects the data-generating ODE independently of
`graph_model.signal_model_name`, defaulting to the latter so existing configs are
unaffected. `simulation.conductance_twin_path` overrides the per-extent default
path. `n_extra_null_edges` and `ablation_ratio` raise `NotImplementedError` under a
conductance ground truth, because they rewrite `ode_params.W` before the ODE is
built and so would not reach the simulated dynamics; `edge_removal_ratio` is
applied after the rollout and is unaffected.

All five `metrics.log` write sites were widened to the 14-column header, including
one hand-written row and two NGP quick-refresh rows. Both readers index
positionally with bounds guards.

### Test and lint status

30 tests in `tests/test_flyvis_conductance.py`. The suite reports 189 passed and 6
failed; the same 6 fail on `neurips2026` without this branch, 4 in
`test_metrics.py` referring to the removed `compute_r_squared` and 2 in
`test_utils.py::TestSortKey`. `ruff check src/ tests/` reports 68 findings on both
this branch and `neurips2026`.

The four training runs have not been executed, so no `connectivity_r2`,
`conductance_r2` or `reversal_r2` values are reported. Extent 15 is verified for
derivation and config loading; its data generation has not been run end to end.
