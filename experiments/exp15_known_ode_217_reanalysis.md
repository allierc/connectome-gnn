---
number: 15
name: known_ode_217_reanalysis
title: The published Known-ODE runs on Flyvis-217 re-analysed with the current test_plot
  pipeline (8 conditions x 5 folds, no retraining)
purpose: update Supplementary Table 7 (Known-ODE across degraded data) and the Known-ODE
  rows of Table 1 so that every number of the paper comes from one analysis code,
  the same one that scored the campaign's GNN runs and experiment 7's Known-ODE rows
baseline: GraphData/config/fly/<base>_blank50_known_ode_cv0N.yaml and the trained
  checkpoints of log/fly/archive_4/<base>_blank50_known_ode_cv0N (May 2026)
specs_dir: experiments/specs/exp15/fly
task: train
queue: gpu_rtx6000
wall: 04:00
n_cpus: 12
axes:
  condition:
  - flyvis_noise_free
  - flyvis_noise_005
  - flyvis_noise_05
  - flyvis_noise_005_010
  - flyvis_noise_005_020
  - flyvis_noise_005_null_edges_pc_400
  - flyvis_noise_005_removed_pc_20
  - flyvis_noise_005_removed_pc_50
  fold:
  - cv00
  - cv01
  - cv02
  - cv03
  - cv04
arms:
- id: kode
  label: Known ODE (published training)
  spec_pattern: '{condition}_blank50_kode217_{fold}'
  submit: false
  differs_by: {}
report:
  axis_labels:
    condition:
      flyvis_noise_free: noise-free
      flyvis_noise_005: low model noise
      flyvis_noise_05: high model noise
      flyvis_noise_005_010: meas. noise 0.1
      flyvis_noise_005_020: meas. noise 0.2
      flyvis_noise_005_null_edges_pc_400: +400% null edges
      flyvis_noise_005_removed_pc_20: -20% edges removed
      flyvis_noise_005_removed_pc_50: -50% edges removed
  arm_labels:
    kode: Known ODE
analyse_job_ids:
  flyvis_noise_free_blank50_kode217_cv00: '154580882'
  flyvis_noise_free_blank50_kode217_cv01: '154580986'
  flyvis_noise_free_blank50_kode217_cv02: '154580987'
  flyvis_noise_free_blank50_kode217_cv03: '154580988'
  flyvis_noise_free_blank50_kode217_cv04: '154580989'
  flyvis_noise_005_blank50_kode217_cv00: '154580990'
  flyvis_noise_005_blank50_kode217_cv01: '154580991'
  flyvis_noise_005_blank50_kode217_cv02: '154580992'
  flyvis_noise_005_blank50_kode217_cv03: '154580993'
  flyvis_noise_005_blank50_kode217_cv04: '154580994'
  flyvis_noise_05_blank50_kode217_cv00: '154580995'
  flyvis_noise_05_blank50_kode217_cv01: '154580996'
  flyvis_noise_05_blank50_kode217_cv02: '154580997'
  flyvis_noise_05_blank50_kode217_cv03: '154580998'
  flyvis_noise_05_blank50_kode217_cv04: '154580999'
  flyvis_noise_005_010_blank50_kode217_cv00: '154581000'
  flyvis_noise_005_010_blank50_kode217_cv01: '154581001'
  flyvis_noise_005_010_blank50_kode217_cv02: '154581002'
  flyvis_noise_005_010_blank50_kode217_cv03: '154581003'
  flyvis_noise_005_010_blank50_kode217_cv04: '154581004'
  flyvis_noise_005_020_blank50_kode217_cv00: '154581005'
  flyvis_noise_005_020_blank50_kode217_cv01: '154581006'
  flyvis_noise_005_020_blank50_kode217_cv02: '154581007'
  flyvis_noise_005_020_blank50_kode217_cv03: '154581008'
  flyvis_noise_005_020_blank50_kode217_cv04: '154581009'
  flyvis_noise_005_null_edges_pc_400_blank50_kode217_cv00: '154581010'
  flyvis_noise_005_null_edges_pc_400_blank50_kode217_cv01: '154581011'
  flyvis_noise_005_null_edges_pc_400_blank50_kode217_cv02: '154581012'
  flyvis_noise_005_null_edges_pc_400_blank50_kode217_cv03: '154581013'
  flyvis_noise_005_null_edges_pc_400_blank50_kode217_cv04: '154581014'
  flyvis_noise_005_removed_pc_20_blank50_kode217_cv00: '154581015'
  flyvis_noise_005_removed_pc_20_blank50_kode217_cv01: '154581016'
  flyvis_noise_005_removed_pc_20_blank50_kode217_cv02: '154581017'
  flyvis_noise_005_removed_pc_20_blank50_kode217_cv03: '154581018'
  flyvis_noise_005_removed_pc_20_blank50_kode217_cv04: '154581019'
  flyvis_noise_005_removed_pc_50_blank50_kode217_cv00: '154581020'
  flyvis_noise_005_removed_pc_50_blank50_kode217_cv01: '154581021'
  flyvis_noise_005_removed_pc_50_blank50_kode217_cv02: '154581022'
  flyvis_noise_005_removed_pc_50_blank50_kode217_cv03: '154581023'
  flyvis_noise_005_removed_pc_50_blank50_kode217_cv04: '154581024'
---

# Experiment 15 — known_ode_217_reanalysis

**The published Known-ODE runs on Flyvis-217 re-analysed with the current test_plot
pipeline (8 conditions x 5 folds, no retraining)**

**Purpose.** Supplementary Table 7 and the Known-ODE rows of Table 1 were scored in May
2026 by an earlier analysis code (W_corrected_R2, tau/V_rest no-outlier R2), while every
other number of the rebuilt paper comes from the current `-o test_plot`. The Known-ODE
is the generator's own equation, so its parameters are read directly (W, softplus(raw_tau),
V_rest) and no retraining is needed: the 40 published checkpoints are re-analysed.

## How

The checkpoints (`models/best_model_with_0_graphs_0.pt`, W / raw_tau / V_rest -- the
same parameterisation as the current `flyvis_known_ode` class) and their
`gt_weights.pt`, `training_edges.pt`, `xnorm.pt`, `ynorm.pt` were COPIED from
`log/fly/archive_4/<base>_blank50_known_ode_cv0N` into new run directories
`log/fly/<base>_blank50_kode217_cv0N`, with `_completed_train` recording the original
training commit; the archive is untouched. The specs are the published ones with only
`config_file` renamed. `tools/exp.py analyse 15` then runs `-o test_plot` on each.

## Specs

40 runs in `experiments/specs/exp15/fly/`, named `<condition>_blank50_kode217_cv0N`.
Analysis only (no training jobs); queue `gpu_rtx6000`.

<!-- STATUS:BEGIN -->

## Status

**40/40 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | condition | n | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kode | flyvis_noise_free | 5 | 1.000 ± 0.000 | 1.000 ± 0.000 |  |  | 0.965 ± 0.001 (0.0) | 0.998 ± 0.000 (1.9) | 0.897 ± 0.002 (1.7) |  | 0.993 ± 0.001 (0.0) |  |  | 0.922 ± 0.002 |
| kode | flyvis_noise_005 | 5 | 1.000 ± 0.000 | 1.000 ± 0.000 |  |  | 0.986 ± 0.000 (0.0) | 1.000 ± 0.000 (0.0) | 0.951 ± 0.001 (0.0) |  | 0.997 ± 0.000 (0.0) |  |  | 0.927 ± 0.005 |
| kode | flyvis_noise_05 | 5 | 1.000 ± 0.000 | 1.000 ± 0.000 |  |  | 1.000 ± 0.000 (0.0) | 1.000 ± 0.000 (0.0) | 1.000 ± 0.000 (0.0) |  | 1.000 ± 0.000 (0.0) |  |  | 0.928 ± 0.005 |
| kode | flyvis_noise_005_010 | 5 | 0.990 ± 0.000 | 0.965 ± 0.002 |  |  | 0.735 ± 0.001 (0.1) | 0.314 ± 0.004 (7.9) | 0.699 ± 0.001 (28.8) |  | 0.889 ± 0.016 (0.4) |  |  | 0.884 ± 0.013 |
| kode | flyvis_noise_005_020 | 5 | 0.963 ± 0.001 | 0.921 ± 0.003 |  |  | 0.475 ± 0.001 (0.2) | 0.059 ± 0.006 (9.7) | 0.562 ± 0.008 (41.8) |  | 0.817 ± 0.027 (0.9) |  |  | 0.865 ± 0.010 |
| kode | flyvis_noise_005_null_edges_pc_400 | 5 | 1.000 ± 0.000 | 1.000 ± 0.000 |  |  | 0.954 ± 0.000 (0.0) | 0.997 ± 0.000 (0.0) | 0.995 ± 0.000 (11.5) |  | 0.982 ± 0.003 (0.0) |  |  | 0.902 ± 0.011 |
| kode | flyvis_noise_005_removed_pc_20 | 5 | 0.997 ± 0.000 | 0.988 ± 0.001 |  |  | 0.839 ± 0.000 (0.0) | 0.989 ± 0.000 (0.3) | 0.787 ± 0.001 (14.1) |  | 0.890 ± 0.009 (1.2) |  |  | 0.711 ± 0.015 |
| kode | flyvis_noise_005_removed_pc_50 | 5 | 0.980 ± 0.000 | 0.939 ± 0.003 |  |  | 0.383 ± 0.002 (0.3) | 0.962 ± 0.001 (0.9) | 0.646 ± 0.003 (34.3) |  | 0.573 ± 0.035 (4.6) |  |  | 0.609 ± 0.018 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | condition | iter | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_noise_free_blank50_kode217_cv00` | landed |  | `614095c02d93` |  |
| `flyvis_noise_free_blank50_kode217_cv01` | landed |  | `unknown` |  |
| `flyvis_noise_free_blank50_kode217_cv02` | landed |  | `unknown` |  |
| `flyvis_noise_free_blank50_kode217_cv03` | landed |  | `unknown` |  |
| `flyvis_noise_free_blank50_kode217_cv04` | landed |  | `unknown` |  |
| `flyvis_noise_005_blank50_kode217_cv00` | landed |  | `d876a57fd871` |  |
| `flyvis_noise_005_blank50_kode217_cv01` | landed |  | `unknown` |  |
| `flyvis_noise_005_blank50_kode217_cv02` | landed |  | `unknown` |  |
| `flyvis_noise_005_blank50_kode217_cv03` | landed |  | `unknown` |  |
| `flyvis_noise_005_blank50_kode217_cv04` | landed |  | `unknown` |  |
| `flyvis_noise_05_blank50_kode217_cv00` | landed |  | `614095c02d93` |  |
| `flyvis_noise_05_blank50_kode217_cv01` | landed |  | `unknown` |  |
| `flyvis_noise_05_blank50_kode217_cv02` | landed |  | `unknown` |  |
| `flyvis_noise_05_blank50_kode217_cv03` | landed |  | `unknown` |  |
| `flyvis_noise_05_blank50_kode217_cv04` | landed |  | `unknown` |  |
| `flyvis_noise_005_010_blank50_kode217_cv00` | landed |  | `unknown` |  |
| `flyvis_noise_005_010_blank50_kode217_cv01` | landed |  | `unknown` |  |
| `flyvis_noise_005_010_blank50_kode217_cv02` | landed |  | `unknown` |  |
| `flyvis_noise_005_010_blank50_kode217_cv03` | landed |  | `unknown` |  |
| `flyvis_noise_005_010_blank50_kode217_cv04` | landed |  | `unknown` |  |
| `flyvis_noise_005_020_blank50_kode217_cv00` | landed |  | `unknown` |  |
| `flyvis_noise_005_020_blank50_kode217_cv01` | landed |  | `unknown` |  |
| `flyvis_noise_005_020_blank50_kode217_cv02` | landed |  | `unknown` |  |
| `flyvis_noise_005_020_blank50_kode217_cv03` | landed |  | `unknown` |  |
| `flyvis_noise_005_020_blank50_kode217_cv04` | landed |  | `unknown` |  |
| `flyvis_noise_005_null_edges_pc_400_blank50_kode217_cv00` | landed |  | `unknown` |  |
| `flyvis_noise_005_null_edges_pc_400_blank50_kode217_cv01` | landed |  | `unknown` |  |
| `flyvis_noise_005_null_edges_pc_400_blank50_kode217_cv02` | landed |  | `unknown` |  |
| `flyvis_noise_005_null_edges_pc_400_blank50_kode217_cv03` | landed |  | `unknown` |  |
| `flyvis_noise_005_null_edges_pc_400_blank50_kode217_cv04` | landed |  | `unknown` |  |
| `flyvis_noise_005_removed_pc_20_blank50_kode217_cv00` | landed |  | `unknown` |  |
| `flyvis_noise_005_removed_pc_20_blank50_kode217_cv01` | landed |  | `unknown` |  |
| `flyvis_noise_005_removed_pc_20_blank50_kode217_cv02` | landed |  | `unknown` |  |
| `flyvis_noise_005_removed_pc_20_blank50_kode217_cv03` | landed |  | `unknown` |  |
| `flyvis_noise_005_removed_pc_20_blank50_kode217_cv04` | landed |  | `unknown` |  |
| `flyvis_noise_005_removed_pc_50_blank50_kode217_cv00` | landed |  | `unknown` |  |
| `flyvis_noise_005_removed_pc_50_blank50_kode217_cv01` | landed |  | `unknown` |  |
| `flyvis_noise_005_removed_pc_50_blank50_kode217_cv02` | landed |  | `unknown` |  |
| `flyvis_noise_005_removed_pc_50_blank50_kode217_cv03` | landed |  | `unknown` |  |
| `flyvis_noise_005_removed_pc_50_blank50_kode217_cv04` | landed |  | `3f728a17eb04` |  |

<!-- STATUS:END -->

## Log

- 2026-10-08, first submission (LSF 154580882-154580921): all 40 jobs exited within
  30 s on a YAML parse error. The spec copy had replaced only the first line of the
  published two-line `description:`, leaving an orphan continuation line. Fixed in
  both copies (experiments/specs/exp15 and the staged GraphData/config/fly), all 80
  files re-parsed.
- cv00 of noise-free re-analysed locally as a check: one-step r 1.000, rollout r
  1.000, R2_W 0.965 over all 434,112 edges, R2_tau 0.998 (1.9% outliers beyond
  0.1 s), R2_V_rest 0.895 (1.7% outliers beyond 0.2), clustering accuracy 0.924. The
  published row read R2_V_rest 0.97 with 10.2% of neurons excluded as outliers by the
  May analysis; the current one excludes 1.7%, so its R2 keeps more of the misfit.
- The other 39 resubmitted (LSF 154580986-154581024, ids in the front matter).
- 2026-10-08, all 40 landed (5 folds per condition). Supplementary Table 7 and the
  Known-ODE rows of Table 1 are now generated from them by
  experiments/paper/scripts/make_tables.py. Against the published table: one-step
  and rollout r unchanged; R2_W within 0.05 except edges -50% (0.26 published, 0.38
  now, the template readout scoring only the fitted edges); R2_V_rest lower wherever
  the May analysis had excluded more neurons (noise-free 0.97 with 10.2% excluded,
  now 0.90 with 1.7%). The manuscript sentence claiming Known-ODE recovery R2 > 0.95
  is flagged red for the authors.
