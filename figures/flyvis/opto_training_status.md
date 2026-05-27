# Opto training status — 2026-05-11 (post test+plot wave)

Source: scan of `/groups/saalfeld/home/allierc/GraphData/log/fly/flyvis_noise_free_blank50_opto_*/`
Target: **1.6M iterations** (`tmp_training/total_iter.txt` = 1600000)
Sentinels: `_complete` = training done, `results_rollout.log` + `results/metrics.txt` = tested + plotted.

## Headline counts (over 195 opto log dirs)

| status | count |
|---|---|
| ✅ trained + tested + plotted | **174 / 195** |
| 🟡 partial training (480k iter, no `_complete`) | **21 / 195** |
| ⛔ dataset complete but no log dir | **10** (`L5_dc_05_cv01..04`, `Tm3_05_cv00..04` — Tm3_05 not in `CONDITIONS`) |

This is the post-wave state after the 46-job test+plot wave dispatched on 2026-05-11 09:14 — 34 fresh dc_05 test+plot runs + 12 collateral re-plots (caused by an earlier `--replot` mistake) all landed cleanly. The R² tables for every fully-done condition are in [../log/remote/cv_blank50_opto_summary.md](../log/remote/cv_blank50_opto_summary.md) (40 sections — baseline + 39 opto conditions).

## Fully done — 34 / 39 declared conditions

`TmY15_{05,heaviside,heaviside_05,dc_05}` · `Mi1_{05,heaviside,heaviside_05,dc_05}` ·
`Tm4_{05,heaviside,heaviside_05,dc_05}` · `Tm1_{05,heaviside,heaviside_05,dc_05}` ·
`Mi4_{05,heaviside,heaviside_05,dc_05}` · `TmY15+Mi1_{05,heaviside,heaviside_05,dc_05}` ·
`retina_{05,heaviside,heaviside_05}` · `retina_heaviside_var` ·
`L4_{05,heaviside,heaviside_05}` · `L5_{05,heaviside,heaviside_05}`

→ All 5 CV folds × 34 conditions = 170 folds fully tested + plotted, plus 4 partial-fold contributions to `retina_dc_05` = 174 total.

## Still owed — 5 partial-trained conditions (21 folds)

| condition | trained folds (`_complete`) | tested+plotted | notes |
|---|---|---|---|
| `retina_dc_05`        | 4/5 | 4/5 | cv04 stopped at 480k — needs retrain |
| `L4_dc_05`            | 0/5 | 0/5 | all 5 folds stopped at 480k |
| `Lawf2_05`            | 0/5 | 0/5 | all 5 folds stopped at 480k |
| `Lawf2_heaviside`     | 0/5 | 0/5 | all 5 folds stopped at 480k |
| `Lawf2_heaviside_05`  | 0/5 | 0/5 | all 5 folds stopped at 480k |

→ All 21 partial folds have intermediate `best_model_with_*.pt` checkpoints from periodic saves but no `_complete` sentinel — the outer-wave skip filter ([run_GNN_optogenetics.py:670-680](../run_GNN_optogenetics.py#L670-L680)) correctly excludes them from test+plot. To resume: `python run_GNN_optogenetics.py --retrain` after wiping their `models/` (otherwise `_is_trained` short-circuits).

## Datasets ready but not in CONDITIONS sweep

- `Lawf2_dc_05` — declared in `CONDITIONS` but no dataset generated (Lawf2 has zero outgoing edges → dc_05 redundant with the other Lawf2 waveforms).
- `L5_dc_05_cv01..04` — dataset only generated for cv00; cv00 still partial (no log dir). Re-run `run_generate_optogenetics.py` to fill.

## Wave history (last 24 h)

| time | event | outcome |
|---|---|---|
| 2026-05-10 ~21:08 | bsub 21 training jobs (Lawf2 × 15 + L4_dc_05 × 5 + retina_dc_05_cv04) | queued |
| 2026-05-11 00:08 | jobs start running on gpu_l4 | ~80k iter/h |
| 2026-05-11 ~05:00 | user `bkill` all 21 | stopped at iter ≈480k |
| 2026-05-11 ~08:30 | `python run_GNN_optogenetics.py --replot` (mistake) | submitted 11 collateral re-plot jobs before Ctrl-C |
| 2026-05-11 ~08:45 | `bkill 0` | 11 collateral jobs killed |
| 2026-05-11 ~08:50 | skip-filter fix landed in `_collect_trained_cfgs` ([commit 343e2f9](../run_GNN_optogenetics.py#L640-L708)) | partial + done folds now skipped at outer-wave level |
| 2026-05-11 09:14 | `python run_GNN_optogenetics.py` (default) | 46 jobs submitted (34 dc_05 + 12 collateral), all completed |
