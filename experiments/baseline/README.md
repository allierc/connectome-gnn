# The frozen baselines

Two files, never edited:

    gnn_current_baseline.yaml   GNN (ours), NeurIPS Table 1
    known_ode_baseline.yaml     Known ODE,  NeurIPS Table 1

Every experiment's specs are generated from one of these by override. Nothing
else in this repo should be called "the baseline".

## Where they come from

Verbatim copies of the published Table 1 configs, at the canonical grid point
`noise_005 / cv01`:

    GraphData/config/fly/flyvis_noise_005_blank50_unified_cv01.yaml
    GraphData/config/fly/flyvis_noise_005_blank50_known_ode_cv01.yaml

The runs behind Table 1 are archived under
`GraphData/log/fly/archive_4/flyvis_noise_*_blank50_{unified,known_ode}_cv0*`,
each with a `_commit` stamp. Training and evaluation ran at different commits
(train 56fbcb5, eval 6e05bc3), and `compute_r_squared` was redefined between
them, so a number re-measured today is not comparable to the printed table
without pinning both.

Two deliberate differences from the source files:

- The agentic loop's `claude:` block is removed from the Known-ODE config. It is
  loop bookkeeping (`n_parallel`, `node_name`, ...), not model configuration.
- `training.derivative_target` is set to `observed_fd`, the nominal target. Table 1 itself trained against the generator`s stored `y_list` -- that IS the bug -- so the baseline carries the corrected target and the bug survives only as an experiment arm.
  These configs predate that key, whose default is now `observed_fd`; loading
  them today without the line would silently train against a different target
  than the paper did, which is exactly the confusion the key exists to end.

## What an experiment overrides

The whole 3 model-noise x 5 fold grid is four fields. Verified by diffing the
published configs against each other -- nothing else varies:

    dataset                        flyvis_{noise}_blank50_{fold}
    simulation.noise_model_level   0.0 | 0.05 | 0.5
    simulation.seed                42 + fold
    training.seed                  1042 + fold

plus `config_file` and `description`, which every generated spec sets to its own
name.

## What is NOT the baseline

`flyvis_*_blank50_unified_fd_*` is the same config with the finite-difference
target, i.e. the post-fix rerun, not the published table. It is an arm, not a
baseline.
