Makes the conductance parameterisation `w_squared: true, g_phi_positive: false` actually trainable, and cleans up around it.

## The fixes

Last PR set the conductance parameterisation on all 93 configs but it collapsed: `Wij_gain` hit 0.000 within 12k iterations on 11 of 18 arms. Two causes, both the same mistake — treating the raw parameter as the effective weight.

- **`W_L2`** charged `model.W.norm(2)`, which under `w_squared` is a norm on `√conductance`, with the same vanishing `2W` in its gradient. Now on `W²`. (`W_L1` was fixed last PR; both are live in the sweep at 7.5e-05 and 7.5e-07, so both mattered.)
- **`W_sign`** is now **refused** rather than converted: it counts positive and negative weights per source neuron, and `W² ≥ 0` makes every edge read as excitatory, so it would report perfect Dale conformance while measuring nothing. Under this parameterisation the sign lives in `g_φ`.

Swept the rest: `grad_clip_W` clips the parameter being updated, so it is correct and unset everywhere here; `l1S` belongs to the CX task-RNN trainer. **The template readout is unaffected** — verified numerically, not assumed: it builds the message as `get_model_W(core) * g_φ`, and `get_model_W` returns `W²` under `w_squared`.

## Removals

The four NeurIPS misspecification knobs (`n_generation_substeps`, `finite_difference_target`, `adapt_g`, `adapt_tau_ms`) and everything they gated — the substep loop, the latent `adapt_c` state, the finite-difference target override, the `[misspec]` banner. Zero of 234 tracked configs used them, the `nr2_*` datasets are gone, and `graph_trainer` / `graph_tester` / `GNN_PlotFigure` never saw them. The integrator branch they left behind is flattened: pick the step, then add the noise, since the two are independent.

## Diagnostics

- `msg_form_r2_median` and the two-form medians now go into `tmp_training/Eij.log`. They come out of the same per-edge fit as `E_ij` and answer what its `R²` cannot — whether the message has the generator's shape — but were computed every checkpoint and thrown away, reachable only from a finished run's `metrics.txt`.
- **The `Eij` panel disagreed with its own log.** It passed no outlier threshold, so it printed an unfiltered `R²` of −4.5e7 while `Eij.log` reported −72 for the same quantity. Same threshold now, same `clean [all] (pct)` annotation `Wij` has always had, and the scatter is bounded to ±10 V — `E = −b₁/b₂` sends failed fits to 1e4 and autoscaling was compressing every real point onto a line.
- `E` and `msg_i` join the progress bar's `R²` format. `E` printed a bare RMSE, which has units and no ceiling; on these runs it reaches 2e4 while the same edges' `R²` is −72.

## Status

The parameterisation no longer collapses — `Wij_gain` climbs to 0.158 by 9.6k. Two things are open and **not** resolved here: `Wij_pearson` declines after ~6k (0.44 → 0.16) while the loss still improves, and half the edges fall out of the `E` filter. 17 sweep arms are running to bracket the regularisers.

Also unresolved, and worth knowing: the generator steps with exponential Euler while the GNN rollout uses forward Euler at the same Δt, so a perfectly-trained GNN would be unstable in rollout. The `finite_difference_target` flag removed above was one of the two possible fixes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
