# REFACTOR_drosophila_voltage.md

**Headline: topology + task + activity -> identifiable circuit.** Add a single activity-observation loss term inside the existing task training so the next paper reads as a direct continuation of `drosophila.tex`. **No new teacher/student class hierarchy** — the "teacher" is simply another already-trained `DrosophilaCxTaskRNN` / `DrosophilaCxTaskGNN` instance, and targets are stored as `NeuronTimeSeries` (the existing on-disk format used by the data generators).

**Four branches.** This refactor is one new loss term (`observation_loss`) exercised in four configurations that share the same student (`DrosophilaCxTaskGNN`), the same hemibrain connectome (N=156), and — except Branch 0 — the same heading task. They form a difficulty ladder; do them in order.

| Branch | Supervision | Data | Task? | Question | Section |
|---|---|---|---|---|---|
| **0** | full per-neuron **voltage** rollout | frozen teacher RNN `drosophila_cx_pi_epg_no_tv_cv0` | no | given everything (full voltage + binary adjacency + Dale sign), does the GNN recover `W_rec`/`τ`/sigmoid? | C |
| **1** | per-neuron **voltage** + heading | frozen teacher `…_tv_cv0`, fresh OU | yes | does task + sparse voltage pin the implementation? | C–D, G–I |
| **2** | **ROI** ΔF/F + heading | real 2-photon (Turner-Evans 2020) | yes | does task + measured ROI activity transfer to real data? | D |
| **3** | voltage + ROI (mixed) | synthetic + real, oversampled | yes | does real data **correct** (vs confirm) the synthetic teacher? | D |

The branches are not symmetric: Branch 2 **cannot** supervise per-neuron voltage (the data is compartmental — ROI = neuropil wedge/glomerulus, not single-cell, §D), so it supervises `P · calcium(σ(h))` against measured ΔF/F. Branch 0 is the floor of the ladder: no task, full observation, the cleanest recovery test, and the cheapest to build (it reuses Branch 1's machinery with the task term off and a precomputed rollout). All four reduce to today's byte-equivalent training when the new coefficients are zero.

## A. Status quo

The drosophila CX pipeline is fully end-to-end direct supervision on heading-direction targets `(cos theta_hd, sin theta_hd)`. `DrosophilaCxTaskRNN` and `DrosophilaCxTaskGNN` (`src/connectome_gnn/models/drosophila_cx_task_rnn.py`, `drosophila_cx_task_gnn.py`) both consume the dict returned by `load_drosophila_cx_connectome(...)` (`connconstr_data.py:63-276`) and return `(y_hat, h_buf)` of shapes `(B,T,2)` and `(B,T,N)`. Loss assembly lives in `graph_trainer.py:2080-2100`: an `F.mse_loss(y_hat, y)` term plus regularisers (`coeff_cos_distance`, `coeff_norm_floor`, `coeff_tv_circular`, `coeff_W_L1`, `coeff_f_theta_diff`, `coeff_g_phi_diff`). The hidden state `h_buf` is never compared against anything — it only feeds priors. The CV harness in `cv_runner.py:211-350` rotates seeds per fold (`seed`, `seed+1000`) across `cv00..cv09`. The repo already defines `NeuronState` and `NeuronTimeSeries` (`neuron_state.py:36-178, 182-310`) as the canonical containers for per-frame and full-rollout circuit observables (`voltage`, `stimulus`, `calcium`, `fluorescence`, ...); the flyvis / cx ODE generators already round-trip simulation traces through `NeuronTimeSeries`.

## B. Goal

`drosophila.tex` shows that **topology + task is insufficient** for full identifiability (the GNN converges in heading but `f_theta` / `g_phi` / `W` are not uniquely pinned). The next claim: **topology + an activity signal closes the gap.** This is also the direct empirical follow-up to `neurips.tex` (flyvis), which states the open problems verbatim: "calcium/voltage imaging are nonlinear, lower-bandwidth proxies" and "partial observation … undermines Ŵ." We test this on a difficulty ladder (Branches 0→3), adding a single new loss term to the existing task training. **No `TeacherNet`, no separate distillation stage** — observation supervision is just another regulariser, exactly like `coeff_tv_circular` is today.

## C. Branch 0 — voltage-rollout system identification (do this first)

The simplest, most reuse-heavy rung: **no task, full voltage observation, topology + sign prior.** Train a `DrosophilaCxTaskGNN` to reproduce the **full voltage rollout** of a frozen teacher RNN — **`drosophila_cx_pi_epg_no_tv_cv0`** (`${GNN_OUTPUT_ROOT}/log/drosophila_cx/drosophila_cx_pi_epg_no_tv_cv0`) — given **only** the **binary adjacency** (connectome support) and the **Dale sign** per edge, and recover the teacher's `W_rec`, `τ`, and sigmoid signalling.

- **Why first.** It is the cleanest possible test: "given everything (full voltage + topology + signs), does the GNN recover the RNN?" If Branch 0 fails, the partial / ROI / task-only branches are hopeless; if it passes, it fixes the recovery ceiling. Strictly easier than §I's sanity check (which adds the task) and than Branch 1 (fresh OU + task).
- **The "new to GNN training" piece.** The GNN consumes **only** `edge_index = nonzero(W_con)` (binary support) and `sign(W_con)` (Dale sign), **never** the teacher's `W` magnitudes. Sign-locking already exists in `DrosophilaCxTaskGNN` (`lock_edge_signs=True → W_eff = |W|·sign_GT`); Branch 0 trains those magnitudes + `f_theta` (→ drift/`τ`) + `g_phi` (→ sigmoid) to fit voltage.
- **Data (reuse the generate→`NeuronTimeSeries`→zarr path).** Roll the frozen `no_tv_cv0` RNN forward once under OU (and constant-ω) stimulus; save `h_buf (T,N)` as the `voltage` field of a `NeuronTimeSeries`. Dataset `drosophila_cx_obs_branch0_v1`. No fresh generation per batch — Branch 0 fits a fixed rollout.
- **Loss.** The voltage branch of the one primitive (§D): `observation_loss(h_buf, teacher_voltage, lam, P=None, sim=None)` — full per-neuron MSE in firing-rate space, `recorded_fraction=1.0`, heading-task weight 0. Same primitive as Branch 1, with the teacher fixed to `no_tv_cv0` and the task term off.
- **Recovery diagnostics.** Identical to §I (panels k/l/m: `τ_eff`, sigmoid-ℓ2, `R²_W`), reused verbatim from `figures/drosophila_cx/fig_evolution.py`. **Pass:** `τ_eff` matches the teacher, sigmoid-ℓ2 < 0.05, `R²_W > 0.95` across folds.
- **Reuse scorecard.** Existing sign-locking + existing recovery diagnostics + the same `observation_loss` primitive. New code beyond the §D primitive = the teacher-rollout dump (a thin script) and one config.

## D. Reuse-first architecture: the single `observation_loss` primitive

The earlier draft of this doc proposed a `cx_observation_model.py` with skeleton-based projections, `voltage_metrics`, `loss_voltage_mse`, `loss_roi_mse`, and a sibling real-data trainer. **That is superseded.** Almost everything already exists; the only genuinely new thing is one observation-loss term and a reused projection.

**What already exists (reuse, do not rebuild):**

| Piece | Location | Reuse |
|---|---|---|
| Calcium forward model (`ca += (dt/τ)(−ca+act(v)); F=αca+β`) | `generators/graph_data_generator.py:2918-2929` | extract to a shared helper |
| Loss-assembly pattern (`term = model.loss_X(h_buf,λ) if λ>0 …; loss = mse+…`) | `models/graph_trainer.py:2080-2100` | add one term |
| `TaskTrials` (`stimulus`, `target=[cos,sin]`, `theta_hd`, `omega`) | `generators/task_state.py` | real loader builds one |
| Neuron→ROI assignment (`src_channel`, `src_roi`, `recorded_mask`) | converter `recorded.pt` (`real_calcium_twocolor.py:362-375`) | build `P` from it |
| Frozen teacher = a `DrosophilaCxTaskRNN`/GNN ckpt | `log/drosophila_cx/drosophila_cx_pi_epg_{no_tv,tv}_cv0` | `create_model` + `load_state_dict` (pattern `graph_data_generator.py:3268`) |
| Calcium config fields (`calcium_type/τ/α/β/activation`) | `config.py:417-441` | reuse as-is |
| Sign-locking (`W_eff=|W|·sign_GT`) | `drosophila_cx_task_gnn.py` | Branch 0/1 recovery |
| Recovery diagnostics (panels k/l/m) | `figures/drosophila_cx/fig_evolution.py` | §I, all branches |
| CV / agentic harness (`claude:` block, fold/seed rotation) | `models/cv_runner.py:211-350` | HP search (§J) |

**New code (deliberately tiny):**

1. **`voltage_to_fluorescence(v, ca_prev, sim)`** — extract `graph_data_generator.py:2918-2929` verbatim (same arithmetic order + four activations); rewire the generator to call it. Shared by generator and loss. (Branch 0/1 voltage path doesn't even need it.)
2. **`models/observation_loss.py` — one function** `observation_loss(h_buf, target, lam, *, sim=None, P=None, mask=None, sigma=None)`:
   - **voltage branch** (Branch 0/1): `P=None, sim=None` → masked MSE of `h_buf` vs teacher `h` over the recorded subset.
   - **ROI branch** (Branch 2/3): `sim, P, sigma` → `P · calcium(σ(h))` vs measured ΔF/F, masked.
   - `lam<=0` → exact `new_zeros(())` (byte-equal off).
3. **`build_roi_pooling(recorded, N=156) -> (P, keys)`** in `real_calcium_twocolor.py` (~10 lines) — sparse `(R,N)` averaging matrix from `src_channel/src_roi`; built once. `keys` fixes the row order shared with `roi_target`. (No skeletons in v1; arbor-weighted `P` is a named v2 refinement.)
4. **`load_real_pooled(root, device, recorded_fraction)`** + **`real_batch(real, B, T, device, gen)`** in `real_calcium_twocolor.py` — pool the 175 recordings, build `P` once, sample `B` fixed-T windows (reject windows past a recording's end; `roi_mask` zeroes NaN frames). Rebuild `stimulus` to match the synthetic encoding exactly (`u[:,:,0]=ω deg/s`, `u[:,0,1:3]=[cosθ0,sinθ0]`), `target=[cos,sin](heading)`.
5. **Config fields (`TrainingConfig`)**, defaults byte-equal: `coeff_observation=0.0`, `observation_target="none"` (`none|voltage|roi|hybrid`), `coeff_task=1.0` (Branch 0 sets `0.0`), `teacher_config=""`, `teacher_ckpt=""`, `real_data_root=""`, `mix_ratio=0.0`, `recorded_fraction=1.0`.

**Trainer integration — 3 guarded blocks, no fork of `_data_train_drosophila_cx_task`** (the 556-line function at `graph_trainer.py:1801`, also reused by the zebrafish/cortex trainers):

- **Setup** (~`graph_trainer.py:1872`): if `coeff_observation>0` and target ∈ {voltage,hybrid} → build the frozen teacher (`create_model` + `load_state_dict` + `eval` + `requires_grad_(False)`); if ∈ {roi,hybrid} → `load_real_pooled`. Branch 0 loads its fixed rollout dataset (or runs the teacher once).
- **Batch loop** (~`graph_trainer.py:2055`): choose synthetic vs real for this iter (`use_real = roi`, or `hybrid & N%K==0` from `mix_ratio`); `y_hat, h_buf = model(u)`; the heading MSE is `coeff_task * mse` (Branch 0 → 0).
- **Loss line** (`graph_trainer.py:2100`): `obs = observation_loss(...)` (voltage form under a `no_grad` teacher, or ROI form); `loss = coeff_task*mse + cosd + norm + tv + l1S + f_diff + g_diff + obs`. Call the teacher and `_sigma` through the **un-compiled** `eval_model` handle (pattern at `graph_trainer.py:2002`) to avoid `torch.compile` shape thrash.

This unifies the old §C/§D `loss_voltage_mse` and the old §J `loss_roi_mse` into the single `observation_loss`; Branches 0/1 are its voltage path, Branches 2/3 its ROI path.

## E. Performance: the bottleneck and the optimization (test-driven)

**Bottleneck:** the ROI branch's calcium recursion as a Python loop over T (≤1550) → O(T) graph depth, slow BPTT, T stored states. This is the only hot spot (the `(B,T,N)@(N,R)` projection with R≤18 is trivial; the voltage branch of Branch 0/1 has no recursion at all).

**Strategy — each step validated against the sequential loop as oracle:**

1. **Pool-before-filter (exact).** `s=σ(h)` per neuron `(B,T,N)` → `s_roi = P@s (B,T,R)` → run the calcium filter on **R≤18** channels, not N=156. Valid because the leaky filter is linear with a _shared_ τ and pooling is linear, so `P·filter(s)=filter(P·s)`. ~9× fewer filter channels.
2. **Conv, not loop.** The leaky integrator is an exponential IIR; replace the recursion with a depthwise **causal `conv1d`** using a truncated kernel `k[i]=α(1−α)^i`, `L=⌈log ε / log(1−α)⌉`. Fully parallel forward+backward, constant graph depth. FFT-conv fallback for long synthetic T.
3. Keep the sequential loop only as the correctness oracle (never in the train path).

ROI input chain: feed `σ(h)` with `calcium_activation=identity` so `calcium(σ(h))` = rate→Ca→F (the right observation model).

## F. Tester (explicit deliverable) — `tests/test_observation_loss.py` (CPU, fast)

- **Equivalence** (the optimization's correctness gate): conv/pool fast-path == sequential-reference within `atol`, over random shapes and τ values.
- **gradcheck**: `torch.autograd.gradcheck` (float64, tiny B,T,N,R) on `observation_loss` → differentiability + correct grads (voltage and ROI branches).
- **pool-commute**: `P·filter(s) == filter(P·s)`.
- **masking**: NaN frames / dropped ROIs → masked MSE correct, finite grads.
- **voltage branch**: identity projection == plain masked MSE to teacher `h`.
- **benchmark**: forward+backward wall-clock + peak memory at `(B=64, T=1550, N=156, R=18)`; assert fast-path under budget and ≥K× faster than the naive loop (GPU when available).
- **Gate A′** (generator byte-equality after the extraction): regenerate one `calcium_type=leaky` dataset pre/post, `sha256` the `fluorescence`/`y` zarr on all four activations — identical.

## G. Experimental matrix

Branch 0 teacher: **Known-ODE no-TV cv0** (`drosophila_cx_pi_epg_no_tv_cv0`). Branch 1 teacher: **Known-ODE +TV cv0** (`drosophila_cx_pi_epg_tv_cv0`, highest-epoch checkpoint picked by `_load`). Conditions (each × CV folds):

- **B0.** GNN, voltage 100%, no task, teacher = no-TV cv0 — the recovery floor (§C, §I criteria).
- **1.** GNN no-TV, voltage 1% — 2 neurons (1 EPG + 1 PEN), random per fold.
- **2.** GNN no-TV, voltage 10% — ~15 neurons stratified across EPG/PEN/Delta7/PEG.
- **3.** GNN no-TV, voltage 100% — full N=156 (task-on upper bound).
- **4.** GNN +TV, voltage 10% — TV + voltage composition.
- **5.** Known-ODE + voltage 10% — control: does the Known-ODE already saturate?
- **6.** GNN no-TV, voltage 10%, teacher = Known-ODE no-TV — does teacher choice matter?
- **R2/R3.** ROI 100% (Branch 2) and hybrid ρ ∈ {8:1, 4:1, 1:1, 1:4} (Branch 3).

Negative control (must appear in agentic-loop provenance per project rule): **GNN no-TV, voltage 10%, random teacher** (untrained ckpt) — expect the observation loss minimised, heading degraded.

## H. Identifiability metrics

Mirror `drosophila.tex` Section 4 (CV-mean tables) plus:

- **per-edge W_rec recovery R²** between student and teacher `J_effective` after sign-locking (`nullspace.py:620-640` style).
- **per-cell-type MI alignment** between `sigmoid(h_pred)` and `sigmoid(h_target)` aggregated by `neuron_types` → reuse `hd_mi_summary` axis labels.
- **embedding silhouette** on learned `a_i` (GNN only) clustered by ground-truth type.
- Existing: `r_roll_1k`, `bump_FWHM`, `chi^2` on cos/sin, four-classes total.
- **Branch 2/3 (no GT `W`)**: replace per-edge R² with ROI ΔF/F RMSE on held-out recordings, bump-phase vs GT heading (circular r), and held-out-neuron / held-out-time ROI prediction.

Tables in the next paper re-use the column layout of `fig_drosophila_cx_four_classes.py`.

## I. Recovery diagnostics & sanity check (Branch 0 and Branch 1)

The teacher (`no_tv_cv0` for Branch 0, `tv_cv0` for Branch 1) has three closed-form quantities the GNN must recover:

1. **Scalar leak `τ`** — teacher drift is `−ĥ_i/τ`. The GNN's `f_θ(ĥ_i, a_i, m=0)` should converge to a line of slope `−1/τ` for every neuron, irrespective of `a_i` (per-neuron `τ_i` collapses onto the scalar).
2. **Sigmoid firing-rate non-linearity** — the GNN's `g_φ(ĥ_j, a_j)` should converge to the sigmoid, irrespective of `a_j`.
3. **Per-edge weights `Ŵ_ij`** on the connectome support — the GNN's `Ŵ` should match the teacher's.

These are exactly the three quantities the current paper says are *not* recovered under task-only supervision. **In-training diagnostics** extend `fig_evolution.py` (do not build from scratch):

- **Panel k (drift / τ)**: learned `f_θ(ĥ_i, a_i, m=0)` in blue, reference `−ĥ/τ` in red; title annotates the linear-fit slope; summary `τ_eff = −1/mean(slope_i)`.
- **Panel l (signalling / σ)**: learned `g_φ(ĥ_j, a_j)` in blue, reference `σ(ĥ)` in red; title annotates ℓ2 distance.
- **New panel m (W recovery)**: scatter teacher `Ŵ_ij` (x) vs student `Ŵ_ij` (y) on the support; identity line; title annotates Pearson r / R².

Each is a one-shot scalar per epoch (`τ_eff`, sigmoid-ℓ2, `R²_W`) onto the trainer's `metrics.log`, so the agentic loop uses them as recovery scores alongside `r_roll_1k`. **Pass:** `τ_eff = 0.1 ± 0.02` s (or the teacher's `τ`), sigmoid-ℓ2 < 0.05, `R²_W > 0.95` across folds. If any fails, the supervision in that branch is *not* sufficient and the paper localises which piece (e.g. `g_φ` needs a monotonicity prior on top). If all pass, the partial-recording rows become the quantitative recovery curve.

## J. Agentic-loop optimization (reuse the existing scheme)

The repo's exploration harness (`claude:` config block + `cv_runner` fold/seed rotation, histories in `log/Claude_exploration/LLM_*`) drives Branches 0/2/3 the same way it tunes cx configs:

- **HP search:** `coeff_observation`, `mix_ratio`, `recorded_fraction`, `calcium_tau/alpha`, registration on/off → maximize the §H/§I identifiability metrics (Branch 0/1: `τ_eff`, `g_φ`-ℓ2, `R²_W`; Branch 2/3: ROI-RMSE, bump-vs-GT-heading circular-r, held-out prediction).
- New `_v1` configs slot in as fresh search targets with distinct dataset/log names; the loop already writes `*_Claude_memory.md`.

## K. Artifact safety, golden gates, and step sequence (zebrafish discipline)

**`_v1` artifact safety:** new configs only — `drosophila_cx_obs_branch0_v1` / `_voltage_v1` / `_roi_v1` / `_hybrid_v1` → new dataset/log dirs. The `cv0` teachers (`no_tv_cv0`, `tv_cv0`) and `real_twocolor/` are **read-only**; the trainer never writes there.

**Golden gates:** **A′** calcium-extraction hash (§F) · **B′** `cv0` inference golden (CPU constant-ω rollout, byte-identical after edits) · **C′** `coeff_observation=0` 1-epoch `metrics.log` matches pre-change · **D′** omitted-vs-explicit-default config-equivalence (byte-equal params + first-100-iter metrics).

**Step sequence (each individually shippable, `coeff=0` byte-equal):**

- **Step 0** — capture A′/B′/C′ baselines; tag the `cv0` source state.
- **Step 1 — Branch 0.** Add `models/observation_loss.py` (voltage path; no generator touch), the config fields, the 3 guarded blocks (teacher + voltage + `coeff_task` gate), the teacher-rollout dump, `drosophila_cx_obs_branch0_v1.yaml`. **Unblocks the rest.** Accept: D′ byte-equal at 0; with `coeff>0` and `coeff_task=0`, the §I recovery (`τ_eff`, sigmoid-ℓ2, `R²_W`) passes against `no_tv_cv0`.
- **Step 2 — Branch 1 + calcium extraction.** Turn the task back on (`coeff_task=1`), teacher = `tv_cv0`, run the §G recording-fraction sweep. Extract `voltage_to_fluorescence`; add the tester (§F). Accept: A′ identical hashes; tester green; recovery curve vs recorded-fraction.
- **Step 3 — Branch 2 (real ROI).** Add `build_roi_pooling` + `load_real_pooled` + `real_batch` + the fast pooled-conv path; `drosophila_cx_obs_roi_v1.yaml`. Accept: masked ROI RMSE drops on held-out recs; benchmark under budget; D′ still byte-equal at 0.
- **Step 4 — Branch 3 (hybrid).** Wire `mix_ratio`; `drosophila_cx_obs_hybrid_v1.yaml` + CV folds; launch the §J agentic search. Accept: `mix_ratio=0`→Step-2, `=1`→real-only; both losses track; recovery curve vs ρ.

**Deferred to v2 (named, not built now):** arbor-weighted `P` from skeletons (`fig_cx_anatomy_3d_voltage_anim.py:_extract_per_neuron_segments`/`_eb_wedge_polygons_3d`, validated against the single raw `.tif`); per-fly circular ROI registration. v1 leans on heading MSE (offset-invariant) + the voltage teacher for ring shape.

## L. Composition with the zebrafish circuit registry + verification

This refactor leans on the existing `NeuronState` / `NeuronTimeSeries` / `TaskTrials` triple both biomodels share, so no new abstractions are needed. It does **not** introduce `circuits/` / `tasks/` / `io_mappings/` on the drosophila side yet — deferred until a second drosophila circuit. The byte-equality contract (golden gate B in the zebrafish doc) holds for `coeff_observation=0`: a config omitting the new fields trains identically to the pre-refactor branch. Run gate D′ before merging.

**Verification (end-to-end):**

1. `tests/test_observation_loss.py` green (equivalence, gradcheck, pool-commute, masking, benchmark).
2. Gates A′–D′ pass.
3. Step-1 Branch 0: §I recovery passes against `no_tv_cv0` (`τ_eff`, sigmoid-ℓ2, `R²_W`).
4. Step-3 Branch 2: held-out masked ROI RMSE↓, decoded bump tracks GT `heading_rad`.
5. Agentic loop produces a recovery curve vs `recorded_fraction` (Branch 1) and `mix_ratio` (Branch 3).
