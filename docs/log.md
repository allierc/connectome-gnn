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
- **Decision.** Re-run with the fixed reporting on the cluster (single run → a100; agentic loop → l4,
  per user). SSH `allierc@login1` works; cluster uses a separate checkout
  (`GraphCluster/connectome-gnn`) → must push `feat/cx-observation` and pull it there. _Run pending._
