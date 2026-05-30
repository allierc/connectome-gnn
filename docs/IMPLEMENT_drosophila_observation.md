# IMPLEMENT_drosophila_observation.md — a small research program on circuit identifiability

> Experiment + execution companion to `docs/REFACTOR_drosophila_voltage.md`. **This is a research program, not just an implementation.** Every step is designed to *yield knowledge* about when connectome topology + activity recover a circuit's mechanism. The engineering (the `observation_loss` apparatus) is subordinate; the byte-equality gates are *calibration controls*; each branch is a *pre-registered, falsifiable hypothesis test*. On approval, save to `docs/IMPLEMENT_drosophila_observation.md`.

## 0. The scientific question

**Under what observation conditions does connectome topology + activity identify a circuit's mechanism `(W_rec, τ, σ)`?** This continues `drosophila.tex` (task alone is insufficient — the implementation is a *manifold*) and `neurips.tex` (partial observation preserves `τ` but undermines `Ŵ`). We answer it with a **ladder of experiments** from full → partial → real observation, sharing one apparatus, with pre-registered metrics, negative controls, and a knowledge ledger.

**Method commitments (apply to every experiment):**
- **Pre-registration.** Metric + threshold + the decision each outcome triggers are fixed *before* the run (written in the config description and the ledger). No post-hoc thresholds.
- **Falsifiability.** Each hypothesis H1 states the outcome that would *refute* it, and runs a **negative control** that must behave as predicted for the result to count.
- **Agentic-loop provenance (project rule).** Every "X is not identifiable / X fails" claim must cite the agentic search breadth (coeff/lr/prior sweep) behind it. A single-config failure is *unsearched*, not *impossible*.
- **Knowledge ledger.** Each experiment appends `(question, prediction, result, interpretation, decision)` to a running `*_Claude_memory.md`, so the program accumulates and later sessions inherit it.
- **Reproducibility.** CV folds + seeds; CPU bit-exact for calibration, GPU within-noise for science.

---

## 1. Apparatus & calibration — so results are signal, not artifact

The instrument is `observation_loss` inside the cx trainer (build steps in §6). Before any result is trusted, calibrate — these gates are *controls on the apparatus itself*:

| Gate | Check | Knowledge it buys |
|---|---|---|
| **D′** | `coeff_observation=0` ⇒ byte-identical training | the instrument doesn't perturb the system when off — later signal is real |
| **A′** | calcium-extraction hash unchanged | the refactor introduced no observation-model artifact |
| **B′** | `cv0` teacher inference byte-identical | the ground-truth reference is stable |
| **C′** | `coeff=0` 1-epoch `metrics.log` matches baseline | training dynamics unchanged by the scaffolding |

A result only counts once its experiment's calibration gates are green.

---

## 2. The experimental ladder

### Experiment 0 — full-observation identifiability (the recovery ceiling)
*Apparatus: the **native** voltage-rollout trainer `data_train_gnn` + `neural_gnn` with the **ported Dale sign constraint**, trained on a voltage rollout of `drosophila_cx_pi_epg_no_tv_cv0` (§6 Phase 1a). No task, no `observation_loss` — Branch 0 is the existing GNN voltage-recovery loop given binary adjacency + sign.*
- **Q.** Given **full** per-neuron voltage + **binary** topology + **Dale signs**, does the GNN recover the teacher's `(W_rec, τ, σ)`?
- **H1.** The inverse problem is well-posed under full observation → recovers. **H0.** A structural degeneracy persists even with everything observed.
- **Prediction (pre-registered).** `τ_eff = τ_teacher ± 0.02`, sigmoid-ℓ2 `< 0.05`, `R²_W > 0.95`, across folds (panels k/l/m, `fig_evolution.py`).
- **Negative controls (falsifiers).** (i) *random/untrained teacher* → loss can still be minimised but `R²_W ≈ 0` (proves the metric measures *recovery*, not *fit*); (ii) *shuffled connectome support* → recovery collapses (proves topology does the work); (iii) *sign removed* (Eq 10 off) → quantifies how much Dale sign contributes; (iv) ***g_φ-monotonicity ablation*** — sign-lock (Eq 10) **without** the Eq 11 prior should reintroduce the ±-sign degeneracy and degrade `R²_W`; with it, recovery is clean. Tests the Eq 10+11 coupling mechanistically (this is the apparatus's own correctness claim).
- **Knowledge.** The **recovery ceiling**, and which of the three parameters is hardest even at 100%. **Decision:** if Exp0 fails, **STOP** — the premise is false; diagnose *which* parameter and *why* (degeneracy vs optimisation vs missing prior) before any partial/real work. If it passes, the ceiling normalises every degradation curve below.

### Experiment 1 — the identifiability frontier under partial observation
*Apparatus: Branch 1 (configs only). Teacher: `…_tv_cv0`.*
- **Q.** How does recovery degrade as the recorded fraction shrinks (100→10→1%)? Which parameter breaks first? Do priors move the frontier?
- **H1.** `τ` robust, `W` fragile (neurips prediction); the TV prior shifts the `W`-frontier; sign-locking is load-bearing.
- **Prediction.** Recovery curves `R²_W(f)`, `τ_eff(f)`, sigmoid-ℓ2`(f)`; a critical fraction `f*` where `R²_W` crosses 0.95; `±TV`, `±sign`, teacher-choice (`tv` vs `no_tv`) shift `f*`.
- **Controls.** random vs stratified neuron selection (does *which* neurons matter?); per-fold subset resampling (is the frontier robust to the specific neurons?).
- **Knowledge.** The **core result** — an identifiability frontier in (observation-fraction × prior) space; the observable-*cheap* vs observable-*expensive* mechanism components. A paper figure.

### Experiment 2 — does the recovered circuit survive real measurement?
*Apparatus: Branch 2 (§6 Phase 3), operator from §6 Phase 2/2N.*
- **Q.** With real ROI ΔF/F (no GT `W`), does topology + task + real activity yield a model that predicts *held-out real* activity, tracks real heading, and has `W` consistent with the connectome?
- **H1.** Yes. **H0.** Real data underdetermines or contradicts the connectome model.
- **Prediction.** Held-out ROI RMSE low; held-out-neuron/time prediction works; bump-vs-GT-heading circular-r high; learned `W` sign/structure connectome-consistent.
- **Controls.** shuffled ROI↔neuron assignment; mismatched recording; **operator ablation** `pooling` vs `ngp_cnn` (Exp 2N).
- **Knowledge.** The **sim-to-real gap**; whether the framework transfers off synthetic data.

### Experiment 2N — does observation-model fidelity matter?
*Apparatus: learned voltage→ROI (NGP+CNN, §6 Phase 2N).*
- **Q.** Does a *faithful* learned voltage→ROI (validated against the physical renderer and the real `.mat`) change the identifiability conclusion vs the coarse averaging `P`?
- **H1.** Higher fidelity → sharper recovery / better real fit; **H0.** the coarse `P` is "good enough" (morphology doesn't matter for ROI-level identifiability).
- **Prediction/measurement.** operator faithfulness (PSNR vs renderer; ROI corr vs real `.mat`); Δ in Exp-2 metrics, `pooling` vs `ngp_cnn`.
- **Knowledge.** Whether the **observation operator itself** is a confound — an ablation of the instrument.

### Experiment 3 — does real data *correct* (not merely confirm) the synthetic teacher?
*Apparatus: Branch 3 hybrid (§6 Phase 4).*
- **Q.** As the mix ratio ρ rises, does the recovered circuit move *toward biology*, closing gaps neither branch closes alone, without degrading heading?
- **H1 (the thesis).** Synthetic data converges to the task manifold; real ROI fine-tunes within it toward the biological circuit. **H0.** Real merely confirms (no movement) or fights the task (heading regresses).
- **Prediction.** Biological-consistency / recovery improves with ρ up to a point; heading non-regressed; hybrid > best single branch.
- **Controls.** ρ=0 (synthetic-only) and ρ=1 (real-only) endpoints; random-teacher hybrid.
- **Knowledge.** Whether **coarse-to-fine sim+real beats either alone** — the headline of the next paper.

---

## 3. Decision tree (how each result routes the next experiment)

- **Exp0 fails** → STOP. Diagnose the failing parameter + cause (degeneracy / optimisation / missing prior). No partial or real work until a ceiling exists.
- **Exp0 passes** → run **Exp1**. If `τ` robust & `W` fragile (as predicted) → **Exp1b**: which prior rescues `W` (TV? norm-floor? more neurons? sign?) — a targeted sub-search.
- **Exp1 frontier known** → run **Exp2**. If real fails where synthetic-partial *succeeded at matched fraction* → the gap is sim-to-real → **Exp2N**: does observation-model fidelity close it?
- **Exp2/2N done** → run **Exp3**. If hybrid > both endpoints → thesis confirmed; else report where/why (with agentic provenance).

---

## 4. What we will know at the end (the deliverable knowledge)

1. The **recovery ceiling** under full observation (Exp0) and which parameter is intrinsically hardest.
2. The **identifiability frontier** vs observation fraction and priors (Exp1) — observable-cheap vs observable-expensive mechanism components.
3. The **sim-to-real gap** (Exp2) and whether **observation-model fidelity** matters (Exp2N).
4. Whether **sim+real hybrid corrects** the synthetic teacher toward biology (Exp3).
5. For **every negative result**: the agentic search breadth that licenses it.

---

## 5. Reuse map (apparatus is mostly already in the repo)

**Branch 0 (native voltage path):** trainer `graph_trainer.py:data_train_gnn:124` (loads `ode_params` edge_index+W, recurrent_loss `1048-1125`) · model `models/neural_gnn.py` (edge weight `:243`, message `:629`, update/edge MLPs + embedding `a_i`) · **sign-lock + g_φ-monotonicity to port** `drosophila_cx_task_gnn.py:{141,393}` (`_edge_sign=sign(W_con)[src,dst]`, `_effective=|W|·sign`) + `loss_g_phi_diff` (Eq 11) · `ode_params.pt` carries edge_index + W(sign) · GT sign source `connconstr_data.py:226` (`mwrec=sign(Jf)`).
**Branches 1–3 (task path):** trainer `_data_train_drosophila_cx_task:1801` (loss `2080-2100`, loop `2055`, `eval_model` `2002`) · calcium model `graph_data_generator.py:2918-2929` · teacher load `fig_kinographs_const_omega.py:_load:54-81`, ckpts `log/.../drosophila_cx_pi_epg_{no_tv,tv}_cv0/models/*.pt` · `TaskTrials` `task_state.py` · converter `real_calcium_twocolor.py` + `graphs_data/.../real_twocolor/`.
**Shared:** recovery panels `fig_evolution.py` · NGP encoders `MultiResGrid_Network.py` + `data_train_INR` · renderer/extractor `fig_cx_anatomy_3d_voltage_anim.py:{123,158,459,429,767}` · anatomy `papers/janelia_cx/anatomy/cx_anatomy_test/` · registry `registry.py`.

## 6. Apparatus build (engineering, phased — each step tagged with the experiment it enables)

- **Phase 0 (calibration).** Add config fields (`coeff_observation,observation_target,coeff_task,teacher_*,real_data_root,mix_ratio,recorded_fraction,roi_operator`; defaults byte-equal). Capture goldens B′/C′. → enables gates D′/A′/B′/C′.
- **Phase 1a (Exp0 — voltage system ID via the *native* GNN trainer `data_train_gnn` + `neural_gnn`).**
  (i) *Dataset:* roll `no_tv_cv0` (load via `fig_kinographs_const_omega._load`) under OU → save voltage `NeuronTimeSeries` + `ode_params.pt` (edge_index + `W` carrying the Dale sign from `load_drosophila_cx_connectome`) → `drosophila_cx_obs_branch0_v1`.
  (ii) *Port the sign constraint into `neural_gnn.py` — two coupled pieces (drosophila.tex Eq 9–11):* **Eq 10 sign-locked Ŵ** (`_effective_edge_weights → |W|·sign_GT`; sign from `ode_params.W` via a new `set_edge_sign_from_ode_params`, applied at `_compute_messages:629`) **and Eq 11 the g_φ-monotonicity prior** (`∂g_φ/∂ĥ ≥ 0`, ported from `drosophila_cx_task_gnn.loss_g_phi_diff`). Together they restore Dale's law per-edge and remove the ±-sign degeneracy — **sign-lock alone is insufficient**. Hook `data_train_gnn` (`graph_trainer.py:~298`) to call `set_edge_sign_from_ode_params`; add config `graph_model.lock_edge_signs_from_connectome` and enable `coeff_g_phi_diff>0`.
  (iii) *Config* `drosophila_cx_obs_branch0_v1.yaml` (`signal_model_name: neural_gnn` cx-voltage variant, `lock_edge_signs_from_connectome:true`, `coeff_g_phi_diff>0`, `w_init_mode:w_con`, dataset = the rollout).
  (iv) recovery diagnostics on neural_gnn's update/edge fns (`τ_eff`, sigmoid-ℓ2, `R²_W`). → enables **Exp0**.
- **Phase 1b (Exp1 — task + sparse voltage via the *task* trainer).** `models/observation_loss.py` voltage path + `_select_neurons` + 3 guarded blocks in `_data_train_drosophila_cx_task` (`coeff_task*mse + … + obs`). Configs `…_voltage_v1` (task on, `tv`, fraction sweep + controls). → enables **Exp1**.
- **Phase 2 (Exp2 prep).** Extract `voltage_to_fluorescence`; add ROI path to `observation_loss` (pool-before-filter + causal `conv1d`, sequential oracle for tests). Tester `tests/test_observation_loss.py` (equivalence, gradcheck, pool-commute, masking, benchmark).
- **Phase 2N (Exp2N).** `voltage_image_ngp` (static per-pixel basis `B(p)∈R¹⁵⁶`, `image=einsum(B_grid,rate)`) + `roi_cnn` + offscreen render wrapper; train on rendered triples; validate vs renderer + real `.mat`; expose `roi_operator=ngp_cnn`; distil to a cheap per-frame head for the loop.
- **Phase 3 (Exp2).** `build_roi_pooling`/`load_real_pooled`/`real_batch`; real tap in the trainer; `…_roi_v1`, held-out split.
- **Phase 4 (Exp3).** `mix_ratio` wiring; `…_hybrid_v1` + CV folds; agentic-loop search over `coeff_observation/mix_ratio/recorded_fraction/calcium_tau`.

Each phase keeps `coeff_observation=0` byte-equal (gate D′), is independently revertible, and is `_v1`-isolated (the `cv0` teachers and `real_twocolor/` stay read-only).

## 7. Concrete first action
```bash
cd /workspace/connectome-gnn-cx && git checkout -b feat/cx-observation
# Phase 0: config fields + capture B′/C′ goldens BEFORE any trainer edit (the calibration controls).
# Then Exp0: Phase-1 voltage path → run drosophila_cx_obs_branch0_v1 → read τ_eff/sigmoid-ℓ2/R²_W.
# Pre-register the Exp0 prediction + decision in the ledger before reading the result.
```
First time a calibration gate fails after a change → that's the divergence; fix before interpreting any science.
