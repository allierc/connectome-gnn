# Plan — Reviewer Vzfg (R2) experiments

Every `[pending]` in `reply_all.tex` maps to a stage below. Ordered by cost and
risk, cheapest and most certain first, so each stage can be written into the
reply as it lands rather than waiting on the batch.

## Where the work happens

The two worktrees have diverged and this matters for scheduling:

| worktree | branch | owns |
|---|---|---|
| `connectome-gnn-cx` | `feat/cx-observation` | misspecification battery (R1), `adapt_g`, drosophila CX |
| `connectome-gnn-ca` | `feat/calcium` | calcium observable + deconvolution, **optogenetics pipeline** |

Consequences: Q2 and Q3 run in `-ca`, Q1 and Q4 run in `-cx`. `-cx` has zero
opto configs and no `training.observable` flag; `-ca` has ~20 blank50 opto
configs and a `voltage|calcium` observable with Wiener/Tikhonov deconvolution.
Do not try to run calcium or opto from `-cx`.

## Assets confirmed present

- Q1 connectome: `papers/Code_NN/Code_NN/Data/Figure5/exported-traced-adjacencies-v1.2` ✅
- Q1 config: `config/drosophila_cx/archive/drosophila_cx_gt_edges_noise_free.yaml`
  (152 neurons, 6 types, `device: cpu`, 20 epochs; `drosophila_cx_known_ode` = oracle)
- Q4 checkpoints: `log/fly/flyvis_noise_free_blank50_{unified,known_ode}_cv00/models/best_model_with_0_graphs_0.pt` ✅
- Q3 opto configs (`-ca`): `flyvis_noise_free_blank50_opto_{L4,L5,Lawf2,Mi1,...}_{heaviside,dc,var}_05.yaml`
- Q2 machinery (`-ca`): `generators/gcamp_kernel.py`, `training.observable`,
  `simulation.calcium_type`, calcium noise, Wiener/Tikhonov deconvolution,
  observable-aware Known-ODE forward

## Blockers

- **MLP / EED checkpoints are not on this filesystem.**
  `fig_jacobian_l1_comparison.py` reads an inaccessible `kumarv4` path;
  `figures/_baseline_cache/` is empty. Affects the MLP row of Q4 and the MLP
  column of Q3. Either obtain the checkpoints, retrain (label as a rerun), or
  ship GNN + oracle and say so. **Decide before Stage 2** — it changes what Q3
  can claim.
- **152 vs 338 neurons.** `reply_all.tex` says 152, from the archived config;
  `figures/drosophila_cx/drosophila_cx_connectome_338` says 338. Stage 0
  resolves this.

---

## Stage 0 — CX smoke test (minutes, CPU, no cluster)

Gate for Stage 3. Confirms the archived config still runs, the hemibrain path
resolves, `W_corrected_R2` is emitted, and settles 152 vs 338.

```
cd /workspace/connectome-gnn-cx
GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData PYTHONPATH=src \
/workspace/.conda_envs/neural-graph-linux/bin/python GNN_Main.py \
  -o generate_train_test \
  config/drosophila_cx/archive/drosophila_cx_gt_edges_noise_free.yaml
```

Pass: `results/metrics.txt` contains `W_corrected_R2`. Fail: fix before
committing GPU time. **Do not skip — Stage 3's whole value rests on this.**

## Stage 1 — Q4 Jacobian (hours, no training)

No new runs. Uses the noise-free blank50 checkpoints above, per the spec in
`rebuttal_QA.md` §Q4.

- GT: `J_ij = W_ij·1[v_j>0]/tau_i`, `J_ii = -1/tau_i`, `S_ik = delta_ik/tau_i`
- Known-ODE: same, analytic in `W_hat`, `tau_hat`
- GNN: autograd chain rule, **not** finite differences (ReLU kinks)
- Same 200 held-out frames for every model; Supp. Fig. 18 panels and colour limits

Deliver: 6-panel figure + the 4-metric table. Off-support mass is the number
that matters; it is 0 by construction for GNN and oracle and is the claim
against the MLP.

## Stage 2 — Q3 OOD rollouts (no retraining, `-ca`)

Vary `I_i(t)` only, roll out fixed checkpoints. The opto pipeline already
implements the hardest condition the reviewer names.

| condition | source |
|---|---|
| naturalistic held-out (ID reference) | existing |
| full-field white noise | new stimulus, trivial |
| drifting gratings | new stimulus, trivial |
| sparse optogenetic drive | existing `*_opto_*_heaviside_05.yaml` |

Report rollout `r` and one-step `r` per model per condition. Prediction: ID
matched across models, gap opens off-manifold, largest under opto.

## Stage 3 — Q1 CX ring attractor (GPU, gated on Stage 0)

Four runs: {GNN, Known-ODE} × σ ∈ {0, 0.05}. σ=0 config exists; clone it with
`simulation.noise_model_level: 0.05` for the second. 152 neurons on CPU/1 GPU —
cheap enough to run alongside the R1 leftovers.

Fills the `[pending]` cell in the three-system coverage table. Prediction on
record in `reply_all.tex`: R²_W at σ=0 **below** Flyvis's 0.89, with a steeper
σ dependence.

## Stage 4 — Q2 calcium (GPU, `-ca`)

Most of this is built. Needed:

1. A blank50 flyvis config with `training.observable: calcium` + `simulation.calcium_type`
2. Confirm whether a Hill saturation exists; the `-cx` `models/gcamp.py` is
   kernel + resampling only. If absent, add it — it is the component that
   compresses the large excursions carrying the weight signal.
3. Runs: calcium raw, calcium deconvolved (Wiener/Tikhonov), voltage control

Note `1f816f7` sets `g_phi_positive=false` for calcium — check whether that
changes the extraction path before comparing R²_W against published voltage runs.

---

## Still running from R1

| run | job | fills |
|---|---|---|
| `nr2_joint_ga03_M1_cv00_unified` | 153174243 | joint mismatch, adaptation arm |
| `nr2_joint_ga03_M10_cv00_unified` | 153174245 | joint mismatch, both-at-once arm |
| `nr2_cad_K10_unified_cv00` | 153172779 | Δt_obs = 200 ms cell |

`finish_joint.py` has a false-completion bug: it treats a timed-out `bjobs` as
"job gone → done" while ignoring `complete_marker=False`, which is why the joint
rows are missing from `results_table.csv`. Fix before reusing it for Stages 3–4.

## Suggested order

Stage 0 now (minutes) → Stage 1 and Stage 2 in parallel (they share no
resources) → Stage 3 as soon as Stage 0 passes → Stage 4 last, largest scope.
Decide the MLP/EED checkpoint question before Stage 2 completes.
