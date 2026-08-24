# CLAUDE.md — connectome-gnn architecture rules

Neural-circuit GNN for the **inverse problem** (dynamics → connectome) across biomodels
(flyvis, drosophila_cx, larva, zebrafish). Two registries keyed by `signal_model_name`
drive everything:
- **model registry** — `@register_model` → `create_model` (`models/registry.py`): *what to build/train*.
- **ode-params registry** — `@register_ode_params` → `get_ode_params_class` (`generators/ode_params.py`): *how to score recovery* (the **grader**: `gt_tau`, `fit_g_phi_curves`, `gt_g_phi_func`, `effective_true_weights`).

## Where does each field live? (decide this for EVERY new field)

| Kind of thing | Lives in | Examples |
|---|---|---|
| **User choice / tunable value** — you'd set or sweep it | **config** (yaml ↔ `config.py` `BaseModel`) | `signal_model_name`, `dataset`, `coeff_*`, `lr_*`, `g_phi_positive`, `recurrent_activation`, `g_phi_norm_target`, outlier/loss thresholds |
| **Dispatch tag** — selects a code path, one of a fixed set | **model class attribute** | `MODEL_FAMILY` (`linear`/`gnn`/`mlp` recovery path), `FORWARD_KIND` (`rnn`/`mlp`/`eed`/`stimulus`/`gnn` forward signature) |
| **How the data was generated** — so a dataset/checkpoint is self-describing | **data artifact `ode_params.pt`** | `activation` (teacher nonlinearity), GT `tau_i`/`V_i_rest`/`edge_index`/`W`, `type_names`, provenance |

Mnemonic: a **class attr** answers *"which kind of model is this?"*; **config** answers
*"what values did the user pick?"*; **`ode_params.pt`** answers *"how was this data made?"*.

- **Never** put a tunable value on a class (no `g_phi_positive`, no coeffs as class attrs) — class attrs are *dispatch only*.
- **Never** hardcode a generation property (activation, GT curve shapes) in a plotter/grader — read it from `ode_params.pt` via the ode-params class.
- A field a user would tune → config. A fixed small set selecting behaviour → class tag. A fact about the generated data → `ode_params.pt`.

## Registries, not if/elif
Add a biomodel/model = **register a class**, keyed by `signal_model_name`. Do **not** add
`if 'foo' in signal_model_name` dispatch — resolve behaviour through the tags
(`models/utils.py: model_family()` / `forward_kind()`, which unwrap `torch.compile`).

## Self-describing data & reload safety
- Model state *derivable from the data* (e.g. `_edge_sign` = `sign(ode_params.W)`) is
  **re-derived on load** (`restore_edge_sign_lock`), **not** baked into the checkpoint
  (avoids a stale copy; single source of truth = the data). Guard with a **loud eval-time
  error** if it's required but unset.
- Shared recovery metric = `metrics.recovery_param_metrics` (computed once; figure/console/
  metrics.txt read from it so they can't drift).

## Outputs: never overwrite
- Generated data → `graphs_data/<dataset>/`; checkpoints/logs → `log/<dataset>/`.
- A new variant = a new `dataset`/config name (`_v1`/`_v2`) = new dirs. Never reuse a name
  with different semantics. Output root via `--output_root` / `GNN_OUTPUT_ROOT` — no new
  path-resolution branches/helpers.

## Conventions
- Recovery metrics: report **scale-sensitive NSE R²** *and* **scale-free structure**
  (Pearson r / z-scored R²). High structure + low NSE = "wiring recovered, under-scaled"
  (the W↔g_φ scale degeneracy), not "wiring wrong".
- Outlier thresholds: `DELTA_TAU=0.1`, `DELTA_VREST=0.2` (neurips.tex `eq:outlier_threshold`),
  named constants = single source of truth across figure/console/metrics.
- Plot colours: **green/black** = GT vs predicted; **red/blue** = two distinct sources
  (L/R, two cells); a single trace gets any neutral colour.
- Git: never stage `config/**/*.yaml` in routine commits; push with `--no-verify` (git-lfs
  absent in the devcontainer); branch before committing on the default branch.
