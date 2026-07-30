# Research log — observation-supervision identifiability program

Running knowledge ledger for the program designed in `docs/IMPLEMENT_drosophila_observation.md`
(execution plan) and `docs/REFACTOR_drosophila_voltage.md` (design). Format per the
plan's §0 commitment: each entry records **(question, prediction, action, result,
interpretation, decision)**. Newest at the bottom. Engineering steps are logged as
*apparatus*; experiments as *findings*.

---

## 2026-05-29 — Apparatus: real-data ingestion + plan

- **Action.** Pulled Turner-Evans 2020 two-color calcium (figshare `12490274`, 1.16 GB),
  opened the devcontainer firewall (added `*.figshare.com` + AWS S3 us-east-1 CIDRs).
  Wrote `src/connectome_gnn/generators/real_calcium_twocolor.py`; converted **175 recordings**
  → connectome-aligned `NeuronTimeSeries` under `graphs_data/drosophila_cx/real_twocolor/`
  (EB 16-ROI / PB 18-ROI, 191,210 imaging frames, GT heading in `behavior.pt`).
- **Knowledge.** Real activity is **compartmental ROI**, not single-neuron (no bodyId);
  effective rank = n_roi, not 156. Raw fluorescence is not published (1 sample `.tif`).
  → motivates a learned/forward observation operator and the ROI (not per-neuron) loss for real data.
- **Decision.** Structured the program as a 4-branch identifiability ladder (Branch 0 voltage
  system-ID → 1 partial voltage → 2 real ROI → 3 hybrid), reframed around the scientific method
  (pre-registration, negative controls, decision tree). Committed plan + design + converter on
  branch `feat/cx-observation` (pushed).

## 2026-05-29 — Apparatus: Branch-0 sign constraint

- **Question.** Branch 0 trains a GNN on a teacher's voltage rollout given binary adjacency +
  sign. How should the connectome Dale sign be imposed?
- **Finding (code audit).** The task GNN's `lock_edge_signs` is a **hard** GT-locked constraint
  (`|W|·sign_GT`, `drosophila_cx_task_gnn.py:393`). The GNN-path `coeff_W_sign`/`dale_law` are
  **emergent** (each neuron picks its own sign), **not** connectome-locked — so the GT sign-lock
  was genuinely missing from `neural_gnn`.
- **Decision (user).** Use the **hard** GT-sign-lock (Eq 10), paired with the g_φ-monotonicity
  prior (Eq 11, already `coeff_g_phi_diff:1500` in the voltage config). Together they restore
  Dale's law per-edge and remove the ±-sign degeneracy; sign-lock alone is insufficient.
- **Action.** Added `graph_model.lock_edge_signs_from_connectome` (default False, byte-equal).
  Ported into `neural_gnn.py`: `_effective_edge_weights` (`|W|·sign_GT`) + `set_edge_sign_from_weights`;
  hook in `data_train_gnn` registers sign from `ode_params.W` after model build.

## Reproduce — Exp0 CLIs (generate voltage data from the RNN, then train the GNN)

Config: `config/drosophila_cx/drosophila_cx_obs_branch0_v1.yaml`
(`simulation.task_model_config_path` → the teacher RNN `drosophila_cx_pi_epg_no_tv_cv0`;
`graph_model.signal_model_name: drosophila_cx_voltage` → NeuralGNN; `lock_edge_signs_from_connectome: true`).
`GNN_OUTPUT_ROOT`/`--output_root` = `/groups/saalfeld/home/allierc/GraphData` (where `graphs_data/`, `log/`, `config/` live).

**1) Generate the voltage dataset by rolling the teacher RNN**
```bash
cd /workspace/connectome-gnn-cx          # cluster path: /groups/saalfeld/home/allierc/Graph/connectome-gnn-cx
GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData PYTHONPATH=src \
  python GNN_Main.py -o generate drosophila_cx/drosophila_cx_obs_branch0_v1 --force
# writes graphs_data/drosophila_cx/drosophila_cx_obs_branch0_v1/{x_list_train,x_list_test,y_list_*.zarr,ode_params.pt}
```

**2) Train the GNN on the voltage data (voltage-only recovery)**
```bash
# local (devcontainer GPU)
GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData PYTHONPATH=src \
  python GNN_Main.py -o train drosophila_cx/drosophila_cx_obs_branch0_v1 --force

# cluster a100 (single run) — runs from the synced /Graph checkout
ssh allierc@login1 "bash -l -c 'source /etc/profile.d/profile.lsf.sh; \
  cd /groups/saalfeld/home/allierc/Graph/connectome-gnn-cx && \
  bsub -n 4 -gpu \"num=1\" -q gpu_a100 -W 360 -o OUT -e ERR \
  bash -l -c \"conda run -n connectome-gnn python GNN_Main.py -o train \
    config/drosophila_cx/drosophila_cx_obs_branch0_v1.yaml \
    --output_root /groups/saalfeld/home/allierc/GraphData --force\"'"
```
(One-shot variants: `-o generate_train` or `-o generate_train_test_plot`.)
Recovery reads: `log/drosophila_cx/drosophila_cx_obs_branch0_v1/tmp_training/metrics.log` (col 2 = `connectivity_r2`)
and `tmp_training/matrix/comparison_*.png` (true W vs learned effective W*).

## 2026-05-29 — Experiment 0: setup + pre-registration

- **Apparatus.** Generated `drosophila_cx_obs_branch0_v1`: rolled the **no-TV teacher**
  `drosophila_cx_pi_epg_no_tv_cv0` (ckpt `..._graphs_9.pt`) → voltage `NeuronTimeSeries`
  + `ode_params.pt` (N=156, **E=10263**, density 0.424, **τ=0.1**, W_rec with **frac_neg=0.394** —
  the GT signs to recover). Config: NeuralGNN, `first_derivative`, `use_gt_edges`,
  `lock_edge_signs_from_connectome:true`, `coeff_g_phi_diff:1500`, `w_init_mode:uniform_scaled`.
- **Question.** Given full per-neuron voltage + binary topology + hard Dale sign, does the GNN
  recover the teacher's `(W_rec, τ, σ)`?
- **Pre-registered prediction.** `τ_eff = 0.1 ± 0.02 s`, sigmoid-ℓ2 `< 0.05`, `R²_W > 0.95`.
- **Negative controls (planned).** random teacher → `R²_W≈0`; shuffled support → collapse;
  sign-off (Eq 10) and g_φ-monotonicity-off (Eq 11) → ±-degeneracy degrades `R²_W`.
- **Decision rule.** If recovery passes → ceiling established, proceed to Exp1 (partial obs).
  If it fails → STOP and diagnose which parameter and why before any partial/real work.
- **Status.** First training run hit a buffer-registration bug (`_edge_sign` registered twice);
  fixed (register a `None` buffer slot once, assign later).

## 2026-05-29 — Exp0 debugging: sign-lock acts on dynamics, not on the reported W

- **Observation (user, `tmp_training/matrix`).** Learned-W matrix showed **wrong signs** vs GT W →
  "the hard sign constraint does not work."
- **Diagnosis.** The sign-lock IS applied in the dynamics: `_compute_messages` uses
  `_effective_edge_weights = |W|·sign_GT`. But `metrics.get_model_W` returned the **raw** `model.W`
  parameter (free sign — only its magnitude enters the message), so both the matrix plot and the
  `conn_R2` metric read the unconstrained parameter. The constraint was working; the *report* was wrong.
- **Fix.** Added a `neural_gnn.effective_W` property (`|W|·sign_GT` when locked, else raw `W`) and made
  `get_model_W` prefer it. Lock-off ⇒ `effective_W is self.W` ⇒ byte-equal for existing flyvis runs.
- **Knowledge.** Under the hard lock the recovered effective W has GT signs *by construction*; `conn_R2`
  now measures **magnitude** recovery (the real Exp0 question). Also learned (agent): `τ_R2`/`V_rest_R2`
  show NaN for `drosophila_cx_voltage` because no ODE-params class is registered for that signal model
  (`metrics.compute_dynamics_r2` falls through to `_DYNAMICS_R2_EMPTY`) → τ/σ recovery needs a separate
  probe (f_θ-slope → τ, g_φ → σ); **conn_R2 is the primary metric** for now.
- **Decision.** Re-run with the fixed reporting on the cluster (single run → a100; agentic loop → l4, per user).

## 2026-05-29 — Cluster wiring + Exp0 first result

- **Cluster gotcha.** First a100 submit (job 150080892) failed: it launched from `GraphCluster/connectome-gnn`,
  which is on `main` (56fbcb5) and lacks the new field → `extra_forbidden`. The **synced** checkout the
  devcontainer bind-mounts is `Graph/connectome-gnn-cx`; submitting `GNN_Main.py` from there (job 150080893,
  gpu_a100) picks up my code with no git push/pull. Launch via `ssh allierc@login1 … bsub -q gpu_a100 …`.
  LSF buffers `.out`/`.err` to completion → monitor the live `tmp_training/` instead.
- **Exp0 first result (job 150080893, epoch 1, iter 4000/80000).**
  - **Sign-lock CONFIRMED.** `tmp_training/matrix/comparison`: `true W` vs `learned effective W*` — all points
    in the correct quadrants (no sign flips). The `effective_W` fix reports the locked weight correctly.
  - **conn_R2 rising:** `−0.0006 → 0.24 → 0.26 → 0.275` (iters 1→4001). Magnitude still compressed (**slope 0.16**)
    → underestimated; likely `lr_W=5e-5`/`w_init_scale=0.01` too conservative for early training.
  - **τ_R2 / V_rest_R2 = NaN/garbage** (as predicted: no ODE-params class for `drosophila_cx_voltage`;
    `derive_tau`/`derive_vrest` from f_θ slopes are undefined here). **conn_R2 is the Exp0 metric**; τ/σ need a
    separate f_θ-slope / g_φ probe (a later apparatus item).
- **Interpretation.** The apparatus is correct end-to-end (sign-lock acts, recovery climbs). The *magnitude*
  recovery is the open question → the agentic loop should search `lr_W`, `w_init_scale`, `coeff_g_phi_diff`,
  `data_augmentation_loop`, `n_epochs`.

## 2026-05-29 — Exp0 default-HP ceiling (job 150080893 DONE, 20 epochs / 1.6M iters)

- **Result.** conn_R2 **plateaued at ≈0.24** (final `R²=0.236, slope=0.13`); magnitudes **collapse toward 0**
  (learned `|W*|` ≪ true `|W|`). Signs perfect (locked). So at the reference HPs the GNN does **not** recover
  the teacher's W magnitudes.
- **Pre-registered prediction was `R²_W>0.95` → NOT met at these HPs.**
- **CRITICAL methodological call (per `drosophila.tex` "agentic loop is primordial").** This is an
  **optimisation result, NOT a biological falsification.** A single HP setting giving `R²=0.24` is *unsearched*,
  not *impossible*. The collapse-to-0 signature points at optimisation (likely `lr_W=5e-5` / `w_init_scale=0.01`
  too small, and/or `g_φ²` absorbing scale while `coeff_g_phi_diff=1500` over-flattens) rather than an
  identifiability wall. **Do NOT record Exp0 as failed.** Decision: the hypothesis-driven agentic loop (l4) is
  now *required* before any conclusion — search `lr_W ∈ {5e-5..3e-3}`, `w_init_scale ∈ {0.01..0.3}`,
  `coeff_g_phi_diff ∈ {0..1500}`, `g_phi_positive`, `data_augmentation_loop`, `n_epochs`; primary score = conn_R2;
  slot 0 = this default config as control.

## 2026-05-29 — Exp0 reframed: structure IS recovered; it's a W↔g_φ scale degeneracy

- **Trigger (user).** "Learned W looks similar to GT up to a scalar." Verified on `best_model_..._graphs_9.pt`:
  - **Pearson r(effective_W, GT) = 0.881** (scale-free) — the **structure of W is recovered**.
  - best global scale **0.124** (learned `|W|` mean 0.023 vs GT 0.168, ~7× too small).
  - R² (scale-sensitive) = 0.226 = what conn_R2 reports; **R² after removing the single scalar = 0.679**.
- **Sign-lock metric check (user's other question).** Confirmed conn_R2 uses the sign-locked `effective_W`:
  `plot_training_linear:40` and `plot_training_flyvis:43` both pull `get_model_W(model)` → `effective_W`;
  the only raw-`model.W` reference is inside `get_model_W` itself. ✓
- **Interpretation.** Branch 0 is **not** a recovery failure. The message is `Ŵ·g_φ²`, so `Ŵ` and the g_φ scale
  trade off — the GNN absorbed the magnitude into g_φ and left `Ŵ` ~8× small. This is the **W↔g_φ scale
  degeneracy** (the flyvis paper pins it with a g_φ normalization anchor, neurips.tex eq. 244). The recovered
  *relative* connectivity is strong (r=0.88).
- **Decision.** Reframe the Exp0 metric + the loop: (i) primary score should be **scale-free** (Pearson r, or a
  scale-corrected R² — the `corrected=True` path already exists in `plot_weight_scatter`); (ii) the agentic loop's
  first hypothesis is the scale degeneracy → search `coeff_g_phi_norm` (anchor g_φ scale), `g_phi_positive`,
  and a g_φ-scale anchor, alongside `lr_W`/`w_init_scale`. **Knowledge gained:** topology+sign+voltage recovers
  W *structure* (r≈0.88) at default HPs; the open problem is fixing the global scale, not identifiability.

## 2026-05-29 — Paper: Appendix A (inverse-problem degeneracy) re-run on the no-TV model

- **Task (user).** Re-run `drosophila_nullspace.py` on the no-TV model and **put forward** the predecessor's
  single-fold **calibrated r = 0.98** in Appendix A (the text had buried it as "no longer holds under TV").
- **Run.** Script default `DATA_DIR`/ckpt = `drosophila_cx_pi_epg` (the no-TV base, `coeff_tv_circular=0`,
  = the predecessor). Output to `/tmp/nullspace_default_base` (did NOT clobber the paper figures). Result —
  per-$(i,\alpha,\mathrm{instance})$: **sum-preserving r=0.528, calibrated r=0.976**; per-$(i,\alpha)$:
  sum-preserving r=0.036, calibrated r=0.472. Matches `fig_sparsify_cx` (0.04 / 0.53 / 0.98). **Predecessor
  r=0.98 reproduced.**
- **Caveat (flagged to user).** Running on `no_tv_cv0` + my `drosophila_cx_obs_branch0_v1` voltage data gave
  calibrated r=0.20 — that dataset uses an opto-PEN input/OU scheme different from the standard
  `*_voltage_noise_free`, so it is not a clean comparison; the canonical no-TV-prior model+data is the base.
- **Edit.** Rewrote the Appendix-A paragraph (`drosophila.tex:1265`) to lead with the no-TV result
  ("near-fully saturable, calibrated r=0.98") consistent with `fig:sparsify_cx`, and demoted the TV
  degradation to a closing caveat. Numbers used: 0.53 / 0.98 (per-instance), 0.04 / 0.47 (per-type).

## 2026-05-29 — Appendix A switched to the latest no-TV model (no_tv_cv0 + branch0 data)

- **User decision.** Use `no_tv_cv0` + `drosophila_cx_obs_branch0_v1` (the latest no-TV results), not the base.
  Key difference: the two no-TV models behave differently — base saturates (calibrated instance r=0.98),
  `no_tv_cv0` does **not** (calibrated r=0.20). Reverted my earlier r=0.98 edit.
- **no_tv_cv0 collapse (rerun).** per-$(i,\alpha)$ [91.6% zeroed]: sum-preserving r=0.60, calibrated r=0.38;
  per-$(i,\alpha,\mathrm{instance})$ [50.2% zeroed]: sum-preserving r=0.63, **calibrated r=0.20**. So the unit
  null space is **not saturable by collapse** — the calibrated OLS fit is *worse* than the naïve sum here.
- **New analysis (user q: how far can we move across units and keep r>0.95).** Sum-zero perturbation *along*
  the unit null directions at amplitude $\lambda\cdot\mathrm{mean}(|W_{group}|)$, rollout 1000 frames, 5 draws:
  unit r = 0.97(λ=0.5) → 0.92(1) → 0.87(2); type r = 0.97(0.5) → 0.87(2). **r>0.95 holds up to λ≈0.5** (half
  the mean group weight). So the trained $\hat W^{rec}$ sits in a *bounded local null basin*: small per-clone
  re-weighting is dynamically silent, but the group cannot be collapsed onto one representative.
- **Figure.** Regenerated `fig_sparsify_cx` from no_tv_cv0: **fixed 4th-column panel labels** (now a–h across
  both rows, was a–f skipping col 4), **bigger fonts**, 4th column shows the calibrated collapse at r=0.20
  ("truncated partially"). Code fix in `drosophila_nullspace.py:plot_sparsify_figure`. Copied to
  `docs/figure/fig_sparsify_cx.png`.
- **Text.** Rewrote the Appendix-A paragraph + figure caption to the no_tv_cv0 story (not saturable; locally
  null up to λ≈0.5).
