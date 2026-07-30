# CLAUDE.md — connectome-gnn architecture rules

## Python environment (run everything with this)
Use the **`neural-graph-linux`** conda env — it has torch + the full stack; the
devcontainer's default `python3` has **no torch**. Don't go searching for it:
- interpreter: `/workspace/.conda_envs/neural-graph-linux/bin/python`
- always `PYTHONPATH=src` and `GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData`
  (data/checkpoints live there; it's writable from the devcontainer).
- e.g. `GNN_OUTPUT_ROOT=… PYTHONPATH=src /workspace/.conda_envs/neural-graph-linux/bin/python GNN_Main.py -o generate <config>`.
- cluster training uses `bsub … "python GNN_Main.py -o … <config>"` (relative paths).

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
- Figure styling: **never** add titles to panels (`ax.set_title(...)`) or to the figure
  (`fig.suptitle(...)`) — titles belong in the LaTeX caption, not the image. Panel labels
  (`a`, `b`, `c`, …) use **bold** font: `ax.text(..., fontweight='bold')`.
- Visual consistency within a figure: every panel uses the **same** font size (axis labels,
  tick labels, panel label), same linewidth, same marker size, same tick length. Set these
  once at the top of the script via `plt.rcParams` (or a shared style helper) — do **not**
  pass per-panel overrides. Mismatched panel sizes/typography across a single figure are
  a regression.
- **One script = one figure.** Each script in `figures/<biomodel>/` produces exactly one
  final PNG/PDF using a single `plt.figure(...)` + `gridspec`/`subplots` layout.
  **No** post-hoc PNG montages, no Inkscape/PowerPoint compositing, no `PIL.Image.paste`
  of pre-rendered subfigures. If a figure has panels `a`–`j`, they are laid out by one
  gridspec in one script, so font sizes, axes, and spacing are guaranteed consistent.
- LaTeX writing: when editing `.tex` files (paper, supplement, captions), **never**
  mention code-level identifiers — no variable names, config filenames, function names,
  module paths, class names, or YAML keys. Describe the *concept* (e.g. "the recurrent
  time constant", not "`gt_tau`"; "the GNN recovery variant", not
  "`zebrafish_hd_si_gnn_ipn12_c3_gcamp.yaml`"). Override only if the user explicitly asks
  for the identifier in the text.
- Git: never stage `config/**/*.yaml` in routine commits; push with `--no-verify` (git-lfs
  absent in the devcontainer); branch before committing on the default branch.
