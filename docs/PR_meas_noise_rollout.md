# Two-form parameter readout, the cross-model known-ODE grid, and the experiment tables

Merges `repro/meas-noise-rollout` into `main`: 96 commits, 34 files of code and
tests plus 223 config and document files.

## What this adds

**A readout that says which family's equation a network learned.** The template
readout already fitted the generator's per-edge message form to the model's own
message. It now fits *both* families' forms to that same message — the
conductance form `b1·u + b2·(u·v_i) + b3` and the current form `W·u + C`, which
is the same design matrix with the driving-force column deleted — and reports
what the extra column buys. Three new keys in `results/metrics.txt`:
`conductance_form_r2_median`, `current_form_r2_median`, and
`driving_force_r2_gain_mean` / `_sd`.

R² alone cannot settle it, because the forms are nested and the fuller one ties
or wins by algebra. Two further measurements can:

- `conductance_form_E_absmedian` against `vi_abs_p99` — where the fitted
  reversal sits relative to the voltage the data actually reaches. A current GNN
  needs `|E|` **201×** outside that range to explain its own message with a
  driving force; a conductance GNN needs 1.4×.
- `template_alt_rollout_r` — the same message read as the *other* family, loaded
  into that family's known-ODE and rolled out on noise-free data.

`results/form_comparison.png` draws all of it (both R² distributions on `1 − R²`,
the test's own lines, and where the fitted reversal lands), `form_comparison.npz`
keeps the per-edge arrays, and `tools/compare_forms.py` runs the one comparison
whose null can be true: this run's gain distribution against a model that has no
driving force.

**The nested F test, reported honestly.** Per edge,
`F = (RSS_current − RSS_conductance) / (RSS_conductance/(n−3))`, the square of
the t statistic on `b2` the reversal gate already used. The run-level answer is
the share of edges rejecting, quoted at 3 standard errors and at a Bonferroni
threshold over every edge tested, always printed with the median frames per edge
and the warning that consecutive frames make both shares optimistic.

**The known-ODE cross-model grid**, 40 runs: each family's known-ODE fitted to
each family's data, five folds, two noise levels. Every cell predicts at one-step
r 1.00 and rollout r ≥ 0.99, mismatched included; only parameter recovery
separates them (W_ij 0.88 matched against −0.07 mismatched), and that needs the
true connectome. `learned_E_exc_median` / `learned_E_inh_median` are now written
for any known-ODE so that claim is checkable from `results/`.

**`docs/experiment_tables.pdf`**, built by `docs/make_experiment_tables.py` from
`docs/experiment_manifest.tsv` and each run's own `metrics.txt` via
`tools/collect_exp_rows.py` — no number in it is typed by hand. Fifteen tables
over 114 runs, each with a generated sentence saying whether the right form was
learned and a note naming which silence prior its rows trained under.

## Bugs fixed on the way

- `sort_key` could not parse the intra-epoch checkpoint name that
  `checkpoint_saves_per_epoch` writes. The sort happens in the plot phase, so
  **forty known-ODE jobs trained to completion and died on their own filenames**
  without writing a result.
- Only `FlyvisConductanceKnownODE` set `MODEL_FAMILY`; every other known-ODE fell
  through to `'gnn'` and was handed to the plotter that scatters `model.a`, which
  a model carrying W, tau and V_rest directly does not have.
- Closing the run's log file on NFS raised `OSError 116` after every figure and
  number had been written, so finished passes exited non-zero.
- The template rollout refused to run when a parameter read back 0.3% off — the
  known-ODE's storage transform, not a misassignment — costing six runs both of
  their rollouts.
- Two `arm_rc5` specs set a rollout curriculum without `recurrent_training`.
- The one-step Pearson r reached the terminal but never `metrics.txt`, so the
  tables had to be filled from cluster job logs.

## Tests

287 passed, 10 skipped. Thirteen tests used to fail in a full run and pass alone:
`flyvis/__init__.py` calls `torch.set_default_device(cuda)` at import, and the
first test to import it flipped the global default for everything after. An
autouse fixture in `tests/conftest.py` restores the CPU-only promise that
conftest's docstring already made.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
