---
number: 9
name: hidden_neurons
title: 20% hidden neurons, current against conductance lasso 25, one-step against
  20-step recurrent training, and five ways to fill the hidden traces
purpose: with 20% of the non-retina neurons unobserved, does 20-step recurrent training
  or a learned hidden-trace network (SIREN(x, y, t), NGP-T, with anchors or a warm-up)
  recover the circuit better than zero-silencing, for the current-form and the conductance
  lasso-25 model; one fold per approach for a landscape, five for the zero-silenced
  baseline
baseline: GraphData/config/fly/flyvis_noise_005_hidden_020_no_ngp_blank50_unified_cv0N.yaml
  (the paper's "20% hidden" row, Supp. Tab. 4)
specs_dir: experiments/specs/exp09/fly
task: train
queue: gpu_rtx6000
wall: '48:00'
n_cpus: 12
axes:
  hidden:
  - none
  - sirentxy
  - sirentxywu
  - ngpt
  - ngptanc
  - ngptwu
  - ngptancwu
  fold:
  - cv00
  - cv01
  - cv02
  - cv03
  - cv04
arms:
- id: cur1s
  label: current, one-step
  spec_pattern: flyvis_noise_005_hid20_{hidden}_cur1s_{fold}
  axes_only:
    hidden:
    - none
  differs_by: {}
- id: cur1s_inr
  label: current, one-step
  spec_pattern: flyvis_noise_005_hid20_{hidden}_cur1s_{fold}
  axes_only:
    hidden:
    - sirentxy
    - sirentxywu
    - ngpt
    - ngptanc
    - ngptwu
    - ngptancwu
    fold:
    - cv00
  differs_by: {}
- id: currc20
  label: current, recurrent 20
  spec_pattern: flyvis_noise_005_hid20_{hidden}_currc20_{fold}
  axes_only:
    hidden:
    - none
  differs_by:
    training.recurrent_training: true
- id: condl251s
  label: conductance lasso 25, one-step
  spec_pattern: flyvis_noise_005_hid20_{hidden}_condl251s_{fold}
  axes_only:
    hidden:
    - none
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
- id: condl251s_inr
  label: conductance lasso 25, one-step
  spec_pattern: flyvis_noise_005_hid20_{hidden}_condl251s_{fold}
  axes_only:
    hidden:
    - sirentxy
    - sirentxywu
    - ngpt
    - ngptanc
    - ngptwu
    - ngptancwu
    fold:
    - cv00
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
- id: condl25rc20
  label: conductance lasso 25, recurrent 20
  spec_pattern: flyvis_noise_005_hid20_{hidden}_condl25rc20_{fold}
  axes_only:
    hidden:
    - none
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
    training.recurrent_training: true
report:
  axis_labels:
    hidden:
      none: zero-silenced
      sirentxy: SIREN(x,y,t)
      sirentxywu: SIREN(x,y,t) + warm-up
      ngpt: NGP-T
      ngptanc: NGP-T + anchors
      ngptwu: NGP-T + warm-up
      ngptancwu: NGP-T + anchors + warm-up
  arm_labels:
    cur1s: current, one-step
    cur1s_inr: current, one-step
    currc20: current, recurrent 20
    condl251s: cond. lasso 25, one-step
    condl251s_inr: cond. lasso 25, one-step
    condl25rc20: cond. lasso 25, recurrent 20
  metric_columns:
    hidden_pearson_mean:
      header: hidden trace r (train)
      live: nnr_pearson
    hidden_rollout_pearson: hidden rollout r
    visible_rollout_pearson: visible rollout r
job_ids:
  flyvis_noise_005_hid20_none_cur1s_cv00: '154469613'
  flyvis_noise_005_hid20_none_cur1s_cv01: '154469614'
  flyvis_noise_005_hid20_none_cur1s_cv02: '154469615'
  flyvis_noise_005_hid20_none_cur1s_cv03: '154469616'
  flyvis_noise_005_hid20_none_cur1s_cv04: '154469617'
  flyvis_noise_005_hid20_sirentxy_cur1s_cv00: '154469631'
  flyvis_noise_005_hid20_sirentxywu_cur1s_cv00: '154469632'
  flyvis_noise_005_hid20_ngpt_cur1s_cv00: '154469633'
  flyvis_noise_005_hid20_ngptanc_cur1s_cv00: '154469634'
  flyvis_noise_005_hid20_ngptwu_cur1s_cv00: '154469635'
  flyvis_noise_005_hid20_ngptancwu_cur1s_cv00: '154469636'
  flyvis_noise_005_hid20_none_condl251s_cv00: '154469645'
  flyvis_noise_005_hid20_none_condl251s_cv01: '154469646'
  flyvis_noise_005_hid20_none_condl251s_cv02: '154469647'
  flyvis_noise_005_hid20_none_condl251s_cv03: '154469648'
  flyvis_noise_005_hid20_none_condl251s_cv04: '154469649'
  flyvis_noise_005_hid20_sirentxy_condl251s_cv00: '154469650'
  flyvis_noise_005_hid20_sirentxywu_condl251s_cv00: '154469651'
  flyvis_noise_005_hid20_ngpt_condl251s_cv00: '154469652'
  flyvis_noise_005_hid20_ngptanc_condl251s_cv00: '154469653'
  flyvis_noise_005_hid20_ngptwu_condl251s_cv00: '154469654'
  flyvis_noise_005_hid20_ngptancwu_condl251s_cv00: '154469655'
  flyvis_noise_005_hid20_none_currc20_cv00: '154469722'
  flyvis_noise_005_hid20_none_currc20_cv01: '154469723'
  flyvis_noise_005_hid20_none_currc20_cv02: '154469724'
  flyvis_noise_005_hid20_none_currc20_cv03: '154469726'
  flyvis_noise_005_hid20_none_currc20_cv04: '154469727'
  flyvis_noise_005_hid20_none_condl25rc20_cv00: '154469737'
  flyvis_noise_005_hid20_none_condl25rc20_cv01: '154469738'
  flyvis_noise_005_hid20_none_condl25rc20_cv02: '154469739'
  flyvis_noise_005_hid20_none_condl25rc20_cv03: '154469740'
  flyvis_noise_005_hid20_none_condl25rc20_cv04: '154469741'
analyse_job_ids:
  flyvis_noise_005_hid20_none_cur1s_cv00: '154477616'
  flyvis_noise_005_hid20_none_cur1s_cv01: '154477617'
  flyvis_noise_005_hid20_none_cur1s_cv02: '154477618'
  flyvis_noise_005_hid20_none_cur1s_cv03: '154477619'
  flyvis_noise_005_hid20_none_cur1s_cv04: '154477620'
  flyvis_noise_005_hid20_sirentxy_cur1s_cv00: '154477621'
  flyvis_noise_005_hid20_sirentxywu_cur1s_cv00: '154477622'
  flyvis_noise_005_hid20_ngpt_cur1s_cv00: '154477623'
  flyvis_noise_005_hid20_ngptanc_cur1s_cv00: '154477624'
  flyvis_noise_005_hid20_ngptwu_cur1s_cv00: '154477625'
  flyvis_noise_005_hid20_ngptancwu_cur1s_cv00: '154477626'
  flyvis_noise_005_hid20_none_currc20_cv00: '154477627'
  flyvis_noise_005_hid20_none_currc20_cv01: '154477628'
  flyvis_noise_005_hid20_none_currc20_cv02: '154477629'
  flyvis_noise_005_hid20_none_currc20_cv03: '154477630'
  flyvis_noise_005_hid20_none_currc20_cv04: '154477631'
  flyvis_noise_005_hid20_none_condl251s_cv00: '154477632'
  flyvis_noise_005_hid20_none_condl251s_cv01: '154477633'
  flyvis_noise_005_hid20_none_condl251s_cv02: '154477634'
  flyvis_noise_005_hid20_none_condl251s_cv03: '154477635'
  flyvis_noise_005_hid20_none_condl251s_cv04: '154477636'
  flyvis_noise_005_hid20_sirentxy_condl251s_cv00: '154477637'
  flyvis_noise_005_hid20_sirentxywu_condl251s_cv00: '154477638'
  flyvis_noise_005_hid20_ngpt_condl251s_cv00: '154477639'
  flyvis_noise_005_hid20_ngptanc_condl251s_cv00: '154477640'
  flyvis_noise_005_hid20_ngptwu_condl251s_cv00: '154477641'
  flyvis_noise_005_hid20_ngptancwu_condl251s_cv00: '154477642'
  flyvis_noise_005_hid20_none_condl25rc20_cv00: '154477643'
  flyvis_noise_005_hid20_none_condl25rc20_cv01: '154477644'
  flyvis_noise_005_hid20_none_condl25rc20_cv02: '154477645'
  flyvis_noise_005_hid20_none_condl25rc20_cv03: '154477646'
  flyvis_noise_005_hid20_none_condl25rc20_cv04: '154477647'
---

# Experiment 9 — hidden_neurons

**20% hidden neurons, current against conductance lasso 25, one-step against 20-step recurrent training, and five ways to fill the hidden traces**

**Purpose.** with 20% of the non-retina neurons unobserved, does 20-step recurrent training or a learned hidden-trace network (SIREN(x, y, t), NGP-T, with anchors or a warm-up) recover the circuit better than zero-silencing, for the current-form and the conductance lasso-25 model; one fold per approach for a landscape, five for the zero-silenced baseline

## Why

The paper's "20% hidden" row (Supp. Tab. 4) hides 20% of the non-retina
neurons and zero-silences them: their voltage enters the GNN as 0 at every step
and the loss uses the visible neurons only. It reads `R2_W` 0.52, rollout r
0.88, `R2_Vrest` 0.49 on the readout of the time. The paper also tried an NGP
hash grid to fill the hidden traces; "preliminary results recover part of the
lost W signal but do not help in parameter recovery", and its archived run
(`archive_4/flyvis_noise_005_hidden_020_ngp_blank50_unified_cv00`) has a hidden
trace r of -0.005, i.e. the grid did not recover the hidden voltages at all.

Two things are new here. Recurrent training at horizon 20 judges a hidden
trace by its effect on the visible neurons over 20 steps rather than one, which
is the most plausible way for the visible data to constrain it. And the
conductance lasso-25 model, which experiments 2-5 put level with the current
form on fully observed data (fixed readout, 2026-09-28).

## What the hidden approaches are

All are existing config options (`graph_model.inr_type_hidden` and training
keys); nothing is new code except the warm-up fix below.

| axis value | hidden voltages | keys |
|---|---|---|
| zero-silenced | 0 at every step | none |
| SIREN(x,y,t) | a SIREN of (x, y, t) sampled at each hidden neuron's position: one spatially smooth field | `inr_type_hidden: siren_txy` (defaults: 2,048 wide, 4 layers, omega 4,096) |
| NGP-T | a multi-resolution temporal hash grid, one output per hidden neuron | `inr_type_hidden: ngp_t`, grid of the paper's NGP spec (24 levels, 4 features, base 16, scale 1.4, decoder 512 x 3) |
| + anchors | 3,600 extra grid outputs trained directly on observed neurons' ground truth, weight 3,000 (the paper's NGP recipe) | `train_with_anchor_neurons`, `coeff_anchor_voltage: 3000`, `n_anchor: 3600` |
| + warm-up | zeros for the first 20% of the iterations, then the learned traces, with the GNN's learning rate damped 100x over a V of 5% either side of the switch | `warmup_inject_nnr_iter_frac: 0.2`, `warmup_inject_nnr_ramp_iter_frac: 0.05` |

The learned traces are trained only through the visible neurons' loss (and
the anchors'), at `lr_NNR_f` 0.0005 (the paper's NGP value). Anchors are not
available for SIREN(x,y,t): the code supports them only for the per-neuron
outputs of `siren_t` and `ngp_t`.

## What else differs

Every spec is the paper's zero-silenced spec for its fold with only these
changes, verified by parsing both:

- **conductance lasso 25**: `signal_model_name: flyvis_conductance`,
  `input_size: 6`, `coeff_g_phi_input_group_L1: 25` (as experiments 2-7);
- **recurrent 20**: `recurrent_training: true`, `n_epochs: 20`,
  `data_augmentation_loop: 50`, `rollout_horizon_schedule: 1..20`, experiment
  3's recipe. The one-step arms keep the paper's `n_epochs: 1`,
  `data_augmentation_loop: 500`, so the two trainings do not see the same
  number of iterations (the recurrent trainer also divides each epoch's
  iterations by its horizon);
- the hidden-approach keys above.

Hyperparameters of the GNN itself are the paper's zero-silenced ones in every
arm; the paper's NGP spec had also changed learning rates, batch size and
regularisation, which would confound the approach with the tuning.

## The warm-up was once per epoch

The trainer rebuilds the injection schedule at every epoch with a per-epoch
iteration count, so with `n_epochs: 20` the learned traces would have been
switched back to zeros at the start of every epoch. Fixed 2026-09-29
(`init_hidden_injection_schedule(..., first_epoch=...)`,
`tests/test_hidden_injection_warmup.py`): the warm-up happens in the first
epoch only. No earlier spec set a warm-up, so nothing earlier is affected.

## Also not usable

`coeff_hidden_voltage` would supervise the hidden neurons with their ground
truth (an oracle) and its code sits in a dead loop, so it has no effect; it is
not set here.

## The three hidden-trace columns

- **hidden trace r (train)**: Pearson r between the learned traces and the
  hidden neurons' true voltage on a random 64 neurons x 256 frames, from
  `tmp_training/nnr_pearson.log`. Written for NGP-T only: the helper that
  computes it (`_quick_ngp_pearson`) returns nothing for SIREN(x,y,t), so those
  rows stay blank until they land.
- **hidden rollout r** and **visible rollout r**: from `-o test_plot`, the
  test rollout's mean per-neuron Pearson r over the hidden and the visible
  neurons separately. For a zero-silenced run the hidden neurons are held at 0,
  so its hidden rollout r is not a trace recovery.

## First look, still training (2026-09-29, train split)

All 44 were running about 1 h after launch: one-step arms at 160,000-400,000
iterations, recurrent arms at 104,000-200,000 (recurrent epochs divide their
iterations by the horizon). On the train split so far `R2_W` is 0.60-0.65 for
the current form and 0.49-0.57 for the conductance lasso-25 model, with no
hidden approach clearly ahead of zero-silencing, and the NGP-T traces are not
tracking the hidden voltages (hidden trace r -0.05 to +0.05). Mid-run, train
split: not a result.

## Audit of the hidden-neuron code (2026-09-30)

Read-only audit of training, recurrent rollout, test rollout and metrics, on the
landed cv00 runs. Findings, ranked:

1. **BUG, reporting: the rollout-r drop 0.878 -> 0.81 is an artefact.** A
   constant trace has Pearson NaN and is dropped by the Fisher-z pool, so the
   zero-silenced (and dead-NGP) arms average 11,340 visible neurons while the
   learned-trace arms average 13,741 including the 2,401 hidden ones at r ~ 0.
   Visible-only, Fisher-z: none 0.877, NGP-T 0.884, NGP-T + anchors 0.884,
   NGP-T + anchors + warm-up 0.885, SIREN 0.876. `visible_rollout_pearson` uses
   an arithmetic mean instead (0.794 for the same neurons).
2. **`R2_W` over all edges is the right target** (Cedric, 2026-09-30): the aim
   is to recover the hidden traces and, through them, the 35% of edges (153,421)
   that touch a hidden neuron. Today those edges are not learned at all
   (visible -> hidden W gets no gradient), which is the failure, not the metric.
   As a diagnostic only, on visible -> visible edges: none 0.882, NGP-T + warm-up
   0.878, SIREN 0.879, NGP-T 0.862, NGP-T + anchors 0.849.
3. **BUG: the NGP-T + warm-up trace network died at injection.** Injection
   switched on at iteration 320,000; within ~700 iterations every unit of the
   last ReLU layer was dead (0 of 512), the output a constant bias. Nothing
   trains the grid during the warm-up without anchors and its features start at
   +-1e-4, so all units sit on the same side of zero for every t.
4. **BUG: the recurrent arms ignore anchors and the warm-up.** The anchor loss
   exists only in the one-step step; the dense rollout always injects and is
   never passed `injection_active`. Anchor r in the recurrent runs is ~0.
5. **Mislabelled: the numbers are not held-out.** With hidden neurons the tester
   evaluates on training frames 0-7,207 (the test split is other videos, where a
   trace network of absolute training time is meaningless by construction), and
   the dataset has no noise-free twin.
6. **DESIGN LIMIT, the main one: hidden voltages are not identifiable.** The only
   signal reaching a trace is through hidden -> visible edges; the traces act as
   a free input per neuron per frame that absorbs the visible neurons' noise.
   Learned traces are ~30x too large (median SD 6.3-6.9 V against 0.224 V), their
   sign is random (per-type mean r -0.67 to +0.66), the message they send has
   r ~ 0 with the true one, and hidden -> visible W correlates with the truth at
   -0.01. 99 of 2,401 hidden neurons have no visible target at all.
7. **DESIGN LIMIT: SIREN(x,y,t) cannot be type-specific.** The 13,741 neurons sit
   at 217 distinct positions, so every type in a column gets the same trace; at
   omega 4,096 the field is effectively white noise across columns. Its training
   Pearson is never logged (the batched call raises and is swallowed).

Checked and fine: hidden / anchor ids consistent everywhere and disjoint; hidden
voltages injected before the forward pass and excluded from the loss; no ground
truth leaks into the test rollout's initial state; gradients do reach the NGP.

## Meta-analysis and what to change

What the data say, once the two reporting bugs are corrected: **no hidden-trace
approach recovers the hidden neurons, and none improves the visible circuit**
(visible -> visible R2_W 0.85-0.88 against 0.882 zero-silenced; visible rollout
r 0.876-0.885 against 0.877), and the edges touching hidden neurons, which are
what `R2_W` over all edges is meant to reward, stay unlearned. Recurrent training at horizon 20 lowers `R2_W` in both
models (0.628 -> 0.571 current, 0.575 -> 0.491 conductance), but those rows use
the mixed-edge metric too and are to be re-read.

Suggested, in order:

1. Fix the rollout-r reporting (visible-only Fisher-z with its n; hidden traces
   scored by |r| and by the message they send). `R2_W` stays over all edges.
2. Make hidden voltages identifiable: restore a hidden self-consistency term
   ((trace(k+1) - trace(k))/dt against the GNN's own predicted dv/dt for the
   hidden neuron, which brings the 68,991 visible -> hidden edges and f_theta in),
   plus an amplitude bound.
3. A free-running recurrent option: integrate hidden neurons as states of the
   model instead of re-injecting them; the only way to get a held-out hidden
   score on the test videos.
4. Let traces share structure within a type (spatial NGP conditioned on the
   embedding a_i, anchors of the same types).
5. Robustness: non-ReLU decoder or larger grid init, train the NGP during the
   warm-up, flag a trace network whose output goes constant, wire anchors and
   warm-up into the recurrent path.

## The recurrent learned-trace runs were withdrawn (2026-09-30)

The 12 recurrent runs with learned traces (SIREN(x,y,t) and NGP-T variants on
cv00, both models) were killed before they finished and removed from this
record: the audit showed their hidden neurons were re-injected at every rollout
step, so recurrent training could not constrain them, and that the anchor loss
and the warm-up never reached the recurrent path. They are replaced by the
free-running runs (hidden neurons integrated by the model after the rollout's
first frame, `training.hidden_free_running`). The zero-silenced recurrent
baselines (5 folds per model) stay.

## Specs

44 runs in `experiments/specs/exp09/fly/`, named
`flyvis_noise_005_hid20_<hidden>_<model><training>_cv0N`: the zero-silenced
baseline on 5 folds x 4 (model, training) = 20, and the 6 learned-trace
variants on fold cv00 x 4 = 24. Queue `gpu_rtx6000` for all, wall 48 h
(experiment 3's recurrent runs took 22-34 h there).

<!-- STATUS:BEGIN -->

## Status

**32/32 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | hidden | n | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster | hidden trace r (train) | hidden rollout r | visible rollout r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cur1s | none | 5 | 0.983 ± 0.001 | 0.878 ± 0.002 |  |  | 0.628 ± 0.013 (0.1) | 0.935 ± 0.020 (4.2) | 0.703 ± 0.042 (25.7) | 0.514 ± 0.088 | 0.861 ± 0.027 (2.6) | 0.032 ± 0.011 | 0.351 ± 0.014 | 0.544 ± 0.025 |  |  | 0.787 ± 0.003 |
| cur1s_inr | sirentxy | 1 | 0.985 ± 0.000 | 0.808 ± 0.000 |  |  | 0.644 ± 0.000 (0.1) | 0.945 ± 0.000 (2.4) | 0.724 ± 0.000 (19.4) | 0.693 ± 0.000 | 0.873 ± 0.000 (2.3) | 0.014 ± 0.000 | 0.363 ± 0.000 | 0.533 ± 0.000 |  | 0.000 ± 0.000 | 0.783 ± 0.000 |
| cur1s_inr | sirentxywu | 1 | 0.981 ± 0.000 | 0.808 ± 0.000 |  |  | 0.619 ± 0.000 (0.1) | 0.794 ± 0.000 (5.2) | 0.665 ± 0.000 (26.9) | 0.439 ± 0.000 | 0.853 ± 0.000 (2.6) | 0.022 ± 0.000 | 0.355 ± 0.000 | 0.551 ± 0.000 |  | -0.001 ± 0.000 | 0.784 ± 0.000 |
| cur1s_inr | ngpt | 1 | 0.985 ± 0.000 | 0.819 ± 0.000 |  |  | 0.655 ± 0.000 (0.1) | 0.922 ± 0.000 (2.1) | 0.699 ± 0.000 (29.6) | 0.671 ± 0.000 | 0.880 ± 0.000 (2.5) | 0.012 ± 0.000 | 0.338 ± 0.000 | 0.567 ± 0.000 |  | 0.006 ± 0.000 | 0.794 ± 0.000 |
| cur1s_inr | ngptanc | 1 | 0.985 ± 0.000 | 0.819 ± 0.000 |  |  | 0.672 ± 0.000 (0.1) | 0.875 ± 0.000 (2.9) | 0.708 ± 0.000 (25.1) | 0.675 ± 0.000 | 0.891 ± 0.000 (2.4) | 0.010 ± 0.000 | 0.337 ± 0.000 | 0.582 ± 0.000 |  | 0.024 ± 0.000 | 0.794 ± 0.000 |
| cur1s_inr | ngptwu | 1 | 0.983 ± 0.000 | 0.878 ± 0.000 |  |  | 0.610 ± 0.000 (0.1) | 0.942 ± 0.000 (3.3) | 0.551 ± 0.000 (23.9) | 0.449 ± 0.000 | 0.875 ± 0.000 (2.5) | 0.021 ± 0.000 | 0.373 ± 0.000 | 0.529 ± 0.000 |  |  | 0.785 ± 0.000 |
| cur1s_inr | ngptancwu | 1 | 0.983 ± 0.000 | 0.822 ± 0.000 |  |  | 0.655 ± 0.000 (0.1) | 0.946 ± 0.000 (4.8) | 0.713 ± 0.000 (24.6) | 0.612 ± 0.000 | 0.872 ± 0.000 (2.4) | 0.021 ± 0.000 | 0.342 ± 0.000 | 0.557 ± 0.000 |  | 0.035 ± 0.000 | 0.797 ± 0.000 |
| currc20 | none | 5 | 0.971 ± 0.002 | 0.885 ± 0.003 |  |  | 0.571 ± 0.010 (0.2) | 0.899 ± 0.050 (3.1) | 0.652 ± 0.020 (29.1) | 0.535 ± 0.055 | 0.859 ± 0.013 (2.9) | 0.029 ± 0.009 | 0.564 ± 0.018 | 0.514 ± 0.015 |  |  | 0.794 ± 0.003 |
| condl251s | none | 5 | 0.982 ± 0.002 | 0.879 ± 0.002 |  |  | 0.575 ± 0.030 (0.2) | 0.924 ± 0.016 (4.6) | 0.718 ± 0.011 (27.5) | 0.707 ± 0.028 | 0.857 ± 0.026 (4.2) | 0.017 ± 0.010 | 0.388 ± 0.009 | 0.512 ± 0.026 |  |  | 0.788 ± 0.001 |
| condl251s_inr | sirentxy | 1 | 0.984 ± 0.000 | 0.808 ± 0.000 |  |  | 0.633 ± 0.000 (0.2) | 0.737 ± 0.000 (2.2) | 0.796 ± 0.000 (26.7) | 0.769 ± 0.000 | 0.876 ± 0.000 (3.4) | 0.006 ± 0.000 | 0.378 ± 0.000 | 0.543 ± 0.000 |  | 0.000 ± 0.000 | 0.783 ± 0.000 |
| condl251s_inr | sirentxywu | 1 | 0.984 ± 0.000 | 0.810 ± 0.000 |  |  | 0.658 ± 0.000 (0.1) | 0.921 ± 0.000 (3.6) | 0.781 ± 0.000 (23.4) | 0.776 ± 0.000 | 0.882 ± 0.000 (2.4) | 0.004 ± 0.000 | 0.360 ± 0.000 | 0.557 ± 0.000 |  | 0.000 ± 0.000 | 0.786 ± 0.000 |
| condl251s_inr | ngpt | 1 | 0.982 ± 0.000 | 0.819 ± 0.000 |  |  | 0.645 ± 0.000 (0.1) | 0.937 ± 0.000 (3.1) | 0.647 ± 0.000 (20.7) | 0.664 ± 0.000 | 0.888 ± 0.000 (2.6) | 0.007 ± 0.000 | 0.364 ± 0.000 | 0.514 ± 0.000 |  | -0.016 ± 0.000 | 0.797 ± 0.000 |
| condl251s_inr | ngptanc | 1 | 0.980 ± 0.000 | 0.820 ± 0.000 |  |  | 0.621 ± 0.000 (0.1) | 0.929 ± 0.000 (3.1) | 0.646 ± 0.000 (20.6) | 0.655 ± 0.000 | 0.874 ± 0.000 (2.8) | 0.006 ± 0.000 | 0.367 ± 0.000 | 0.531 ± 0.000 |  | -0.001 ± 0.000 | 0.797 ± 0.000 |
| condl251s_inr | ngptwu | 1 | 0.981 ± 0.000 | 0.878 ± 0.000 |  |  | 0.568 ± 0.000 (0.2) | 0.943 ± 0.000 (4.4) | 0.656 ± 0.000 (25.6) | 0.653 ± 0.000 | 0.845 ± 0.000 (4.5) | 0.004 ± 0.000 | 0.383 ± 0.000 | 0.517 ± 0.000 |  |  | 0.786 ± 0.000 |
| condl251s_inr | ngptancwu | 1 | 0.983 ± 0.000 | 0.822 ± 0.000 |  |  | 0.657 ± 0.000 (0.1) | 0.920 ± 0.000 (3.4) | 0.726 ± 0.000 (25.1) | 0.712 ± 0.000 | 0.880 ± 0.000 (2.4) | 0.004 ± 0.000 | 0.350 ± 0.000 | 0.549 ± 0.000 |  | 0.027 ± 0.000 | 0.797 ± 0.000 |
| condl25rc20 | none | 5 | 0.967 ± 0.004 | 0.885 ± 0.002 |  |  | 0.491 ± 0.008 (0.3) | 0.916 ± 0.019 (6.3) | 0.654 ± 0.017 (31.2) | 0.639 ± 0.009 | 0.841 ± 0.016 (5.9) | 0.024 ± 0.012 | 0.608 ± 0.016 | 0.465 ± 0.005 |  |  | 0.794 ± 0.002 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | hidden | iter | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster | hidden trace r (train) | hidden rollout r | visible rollout r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | | | | |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_noise_005_hid20_none_cur1s_cv00` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_cur1s_cv01` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_cur1s_cv02` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_cur1s_cv03` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_cur1s_cv04` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_sirentxy_cur1s_cv00` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_sirentxywu_cur1s_cv00` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_ngpt_cur1s_cv00` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_ngptanc_cur1s_cv00` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_ngptwu_cur1s_cv00` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_ngptancwu_cur1s_cv00` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_currc20_cv00` | landed | 575,233 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_currc20_cv01` | landed | 575,233 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_currc20_cv02` | landed | 575,233 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_currc20_cv03` | landed | 575,233 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_currc20_cv04` | landed | 575,233 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_condl251s_cv00` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_condl251s_cv01` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_condl251s_cv02` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_condl251s_cv03` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_condl251s_cv04` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_sirentxy_condl251s_cv00` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_sirentxywu_condl251s_cv00` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_ngpt_condl251s_cv00` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_ngptanc_condl251s_cv00` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_ngptwu_condl251s_cv00` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_ngptancwu_condl251s_cv00` | landed | 1,520,001 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_condl25rc20_cv00` | landed | 575,233 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_condl25rc20_cv01` | landed | 575,233 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_condl25rc20_cv02` | landed | 575,233 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_condl25rc20_cv03` | landed | 575,233 | `7c8a83e7abac` |  |
| `flyvis_noise_005_hid20_none_condl25rc20_cv04` | landed | 575,233 | `7c8a83e7abac` |  |

<!-- STATUS:END -->
