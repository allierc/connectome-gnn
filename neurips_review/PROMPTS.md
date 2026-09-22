# NeurIPS 2026 rebuttal — recovered source prompts

Verbatim user prompts that specified the rebuttal work, recovered from the Claude
Code session transcripts in `~/.claude/projects/-workspace--devcontainer/`.
Kept here because the chat UI does not survive a VS Code reload; this file does.

Math in the prose blocks is duplicated (`Δt/M` then `Δt/M` re-rendered) — that is a
copy-paste artifact from the original messages, preserved as-is.

## Session index

| session id | date | topic | resume with |
|---|---|---|---|
| `78bc7f6c-aac6-4cf4-8129-71710d26603e` | 2026-07-25 05:44–16:49 | **main**: misspecification battery (tests 1–3), cluster launch | `claude --resume 78bc7f6c-aac6-4cf4-8129-71710d26603e` |
| `0df48628-88ae-4fc6-8f78-86889cddfc9e` | 2026-07-25 06:11 | full-sample R² twins of Tabs. 1, 2, Supp. 4, 7 | `claude --resume 0df48628-88ae-4fc6-8f78-86889cddfc9e` |
| `2be8013c-7ad3-470e-81c0-a83af358266c` | 2026-07-25 05:33 | `rebuttal_noise_identifiability.tex` (Appx. C spectrum) | `claude --resume 2be8013c-7ad3-470e-81c0-a83af358266c` |
| `a1cf7c4d-5424-4a2b-89af-b08f9c87225d` | 2026-07-25 19:00 | Jacobian figure (Vzfg Q4) + overnight no-GPU batch | `claude --resume a1cf7c4d-5424-4a2b-89af-b08f9c87225d` |
| `456be999-fa7f-446c-94ca-e1a332f73b26` | 2026-07-26 06:25 | `rebuttal_QA.md` + `reply_R1.tex` drafting | `claude --resume 456be999-fa7f-446c-94ca-e1a332f73b26` |

Inputs referenced throughout: review [docs/neurips_review.txt](docs/neurips_review.txt),
paper [docs/neurips.tex](docs/neurips.tex).

---

## Prompt 1 — framing (78bc7f6c, 2026-07-25 05:44)

Im working on the review: /workspace/connectome-gnn-cx/docs/neurips_review.txt of our paper /workspace/connectome-gnn-cx/docs/neurips.tex

Im working on a reply of 2. The experimental setting is strongly model-matched

Although the GNN does not contain the exact scalar Flyvis equation, its hypothesis class incorporates much of the generator's structure:

    first-order dynamics;
    additive pairwise message aggregation;
    a supplied graph containing the true edges;
    direct observation of every neuron's voltage;
    access to the external visual input;
    matched simulation and inference time steps;
    monotonicity constraints;
    squared nonnegative message functions; and
    a 50% null-stimulus interval that helps expose resting dynamics.

Thus, the generator lies within the intended model family. The method is flexible with respect to the scalar nonlinearities, but it is not broadly agnostic to dynamical form, interaction order, coupling structure, observation process, or temporal discretization.

There is no experiment involving even mild model misspecification, such as a different observation cadence, conductance-like coupling, multiplicative interactions, latent variables outside the graph, or a generator whose update does not decompose into additive pairwise messages. Consequently, the experiments demonstrate recovery of a matched simulator, not general recovery of biological circuit mechanisms.

which sum up concernes of the three reviewers:
can you list the detailed concerns that are summed up

---

## Prompt 2 — THE launch prompt: misspecification experiments (78bc7f6c, 2026-07-25 05:51)

This is the one that created `neurips_review/` and defined tests 1–3.

I need to run some code in /workspace/connectome-gnn-cx
write these scripts in a separated folders neurips_review

here is my draft answer

We accept the substance and will rescope the claims. Two clarifications, then the new experiments.

Three listed items are properties of the preparation, not the method. A measured connectome, a known stimulus, and a blank-screen interval are all available in real experiments; the paper already tests violations of the first (−20%, −50% edges) and the second (SIREN). These do not constitute model match.

One item is inaccurate. No sign convention is imposed: g_phi^2 ≥ 0 factorizes magnitude from sign, and the sign of every interaction is carried by W_hat_ij ∈ R, unconstrained. Dale's law is not assumed. Relatedly, only the aggregation is additive — f_theta(v_i, a_i, m_i, I_i) is a general nonlinear function of self-state and drive.

Four are genuine: first-order dynamics, direct voltage observation, matched Δt, and the monotonicity priors. We test three:

Δt mismatch — simulate at Δt/M, M ∈ {1,2,5,10}, observe at Δt = 20 ms, so the discrete map is no longer shared (R1 Q1).
Monotonicity ablation — μ0 = 0, μ1 = 0, and unsquared g_phi, reporting R²_W for each.
Unobserved adaptation current — ċ_i = (v_i − c_i)/τ_a entering the generator with c_i never observed, violating both first-order-in-observables and the no-latent-variable assumption.

Conductance-like coupling and a calcium observation model require changes to the architecture and loss and are deferred to a revised version, where they will be the primary new evidence.

On simulator-only data: no dataset pairs synapse-resolution connectivity with dense activity in the same animal, which is the premise of the work rather than an oversight. We agree the title should not imply otherwise, and will scope the title, abstract, and contributions to model-matched in-silico feasibility — replacing "without assuming the form of the dynamical equation" with "agnostic to the scalar nonlinearities within a first-order, additively-aggregated class."

hence I need you to set up the new experiments:

```
# Rebuttal experiments: model-misspecification tests (AC §2, R1-W1/W2, PVbi ¶3)
# Base: Flyvis-217 (13,741 neurons, 434,112 edges), sigma=0.05, gamma=0,
# consensus hyperparameters (Supp. Tab. 6), same DAVIS clips / 50% null stimulus,
# single seed throughout (label as single-seed), 64,000 training frames,
# eval on 8,000 held-out frames.
# Report for every run: one-step r, rollout r, R2_W, SLOPE of W_hat vs W,
# R2_tau and R2_Vrest (inlier + full-sample + outlier %), cluster acc.
# Also run Known-ODE for tests 1 and 3 (oracle isolates inverse-problem
# degradation from GNN capacity).

## PRE-CHECK (zero compute, do first)
Grep configs for the mu_0 coefficient (coeff_f_theta_msg_diff or equivalent).
mu_0 is ABSENT from Supp. Tab. 6 and the noise-free agent log lists
f_theta_msg_diff=50 as NEUTRAL / not adopted. If mu_0 = 0 in all reported runs,
half of R1-W2's monotonicity objection is answered with no compute. Confirm and
report before running test 2.

## TEST 1 — Delta-t mismatch (satisfies R1 Q1). LAUNCH FIRST.
Goal: break the identity v(t+dt) - v(t) = dt * F(v(t)) so simulator and
inference no longer share the same discrete map.
Data gen, for M in {1, 2, 5, 10}:
  - integrate Eq. (1) with step h = dt/M, dt = 20 ms
  - hold TOTAL SIMULATED TIME fixed (64,000 * dt), not step count
  - subsample every M-th state -> 64,000 observed frames at 20 ms
  - scale process noise for constant per-unit-time variance: noise increment
    ~ N(0, h), i.e. per-step amplitude * sqrt(1/M) relative to M=1.
    CRITICAL: getting this wrong makes effective sigma vary with M and
    confounds the entire sweep. Verify: variance of v over the null-stimulus
    window must match across M for a single neuron.
Training: consensus hyperparameters unchanged. Target stays the OBSERVED finite
difference at 20 ms, y_i(t) = (v_i(t+dt) - v_i(t)) / dt. No correction, no
access to finer steps — that is the point.
Sanity gate: M=1 must reproduce the existing sigma=0.05 row (R2_W = 0.99).
If not, STOP — something else in the pipeline changed.
Loader check before launch: confirm nothing hardcodes dt = 0.02 when computing
y. If finer-step data are saved with h in the config, the target may silently be
computed at h instead of dt, defeating the experiment.
Expected: truncation bias (v(t+dt)-v(t))/dt = vdot + (dt/2)*vddot + O(dt^2)
should appear mainly as SLOPE != 1 with R2 largely preserved — recoverable
miscalibration, not loss of structure. Report slope prominently.
Cost: training cost is M-independent (64,000 frames either way); only simulation
scales. 4 GNN slots (~4.5 GPU-h each) + 4 Known-ODE (~0.2 each).

## TEST 2 — Monotonicity / message-form ablation (answers R1-W2). No new sim.
5 slots, all at sigma=0.05, consensus config as control:
  slot 0: control (unchanged)                  expect R2_W = 0.99
  slot 1: mu_0 = 0                             expect unchanged if already 0
  slot 2: mu_1 = 0 (750 -> 0)                  expect collapse (g_phi -> 0)
  slot 3: mu_2 = 0 (0.9 -> 0)                  expect scale drift in W_hat
  slot 4: g_phi unsquared (raw, not g_phi^2)   tests sign-factorization claim
Slot 4 requires re-extraction: Eq. (16) uses eta_j = slope of g_phi^2; with raw
g_phi it becomes the slope of g_phi itself. Update the extraction accordingly.
Report whichever way it lands. If mu_1 is load-bearing, frame it correctly:
mu_1 excludes the degenerate solution g_phi == 0, it is not a biological
assumption about synapses.
Cost: ~4.5 GPU-h each, ~22 GPU-h total.

## TEST 3 — Unobserved adaptation current (AC's latent-variable item).
Modified generator:
  tau_i * dv_i/dt = -v_i + Vrest_i + sum_j W_ij ReLU(v_j) - g_a * c_i + I_i
  dc_i/dt = (v_i - c_i) / tau_a
c_i is NEVER observed and NEVER given to the GNN or Known-ODE. This violates
first-order-in-observables AND introduces a latent variable outside the graph.
Grid: g_a in {0, 0.1, 0.3}; tau_a = 200 ms; c_i(0) = v_i(0); sigma = 0.05.
tau_a >> tau (~25 ms) is the point: a slow hidden variable the GNN cannot absorb
into f_theta(v_i, .), since c_i is not a function of instantaneous v_i.
Sanity gate: g_a = 0 must reproduce the existing sigma=0.05 row.
Expected: c_i acts as a slow per-neuron offset, so it should contaminate
V_rest first (already the weakest parameter), then tau, with W most robust
(weights are constrained by fast fluctuations c_i tracks too slowly to explain).
Report the pattern across g_a, not just aggregate degradation.
Cost: 3 GNN slots + 3 Known-ODE, ~1 batch.
```

note existing data is in /workspace/connectome-gnn-cx/graphs_data/fly
config used previoulsy are in /groups/saalfeld/home/allierc/GraphData/config/fly

the main script used to train is /workspace/connectome-gnn-cx/GNN_Main.py
but we will need to launch on the clusters on a100 nodes in parallel
there is code for that in /workspace/connectome-gnn-cx used by /workspace/connectome-gnn-cx/GNN_LLM.py

---

## Prompt 3 — Δt direction + observation-cadence sweep (78bc7f6c, 2026-07-25 12:07)

Settled that the generator is fine and the model is coarse, and added the
`Δt_obs ∈ {20, 40, 100, 200} ms` sweep at fixed `h = 2 ms`.

Reviewer asked for your direction. R1 Q1: "generate data with a finer integration step than the GNN uses." AC §2: "a different observation cadence." Simulate fine, infer coarse. Your spec is correct.

Why the reverse tests nothing. If you generate at 20 ms, the 20 ms map is the ground truth. A GNN stepping at Δt/M is then strictly more expressive than the generator — it can compose M substeps to reproduce the coarse map exactly, so there's no misspecification, only extra capacity. You'd also have no observations at the intermediate times to supervise or evaluate against.

The mismatch has to sit between the true dynamics and what the model can express. That requires the truth to be finer than the model's resolution, not coarser.

still we have to follow reviewer request

We are. R1 Q1, verbatim: "generate data with a finer integration step than the GNN uses."

    finer integration step → h = Δt/M ✓
    than the GNN uses → GNN steps at Δt = 20 ms ✓

Your spec is literally their sentence. Same for AC §2's "a different observation cadence" — you observe at 20 ms while the system evolves at 2 ms.

Where the ambiguity crept in is the label. Writing "M ∈ {1,2,5,10}" next to Δt/M can read as if M indexes what the GNN uses. It doesn't — it indexes the simulator's substepping. Worth stating the two rates explicitly in the response so no reviewer has to infer it:

    The generator is integrated at h = Δt/M with M ∈ {1,2,5,10} (i.e. down to 2 ms), while the GNN receives observations and predicts at Δt = 20 ms throughout.

That's one sentence and it removes any doubt about which direction the mismatch runs.

short answer we do what the reviewer asked ?

Yes. R1 Q1 says "generate data with a finer integration step than the GNN uses" — that is exactly your setup: simulate at Δt/M (down to 2 ms), GNN observes and predicts at 20 ms.

Only wording change worth making, so no one misreads which side M applies to:

    The generator is integrated at h = Δt/M, M ∈ {1,2,5,10}, while the GNN observes and predicts at Δt = 20 ms throughout.

so we do not have to retrain everything short answer

Correct. Nothing to retrain — your runs already implement what R1 asked for.

The only outstanding items on this test are the two I flagged: add M=2 (missing, and it's the informative point), and resolve whether the M=1 numbers (0.955 / 0.671) are full-sample or inlier, since that determines the baseline the sweep is read against.

simulate at Δt/M (down to 2 ms), GNN observes and predicts at 20 ms.
does this fullfill the reviewr demand ? or we also generate at 20 ms and GNN observe and predicts at 40 100 200 ms ? I think it is worth do train them

Yes, current design satisfies R1 Q1 — and the cadence sweep is worth adding

But don't generate at 20 ms. If the truth is the 20 ms Euler map, you're studying an integrator artifact rather than a dynamical system: at Δt/τ = 0.8 that map's decay factor is 0.20 vs. 0.45 for the exact flow. Any coarser observation is then measured against a reference that isn't biophysical.

Correct design — one generator, sweep the observation rate:

    h = 2 ms (fixed, near-continuous), Δt_obs ∈ {20, 40, 100, 200} ms
    Δt_obs/τ ∈ {0.8, 1.6, 4, 8}

This is the experimentally meaningful axis — imaging rate slower than the membrane time constant — and it's what AC §2's "different observation cadence" names. Your M=10 run is already the Δt_obs = 20 ms point, so you need three new conditions.

Note the overlap with an existing result. Supp. Tab. 4's "1/5 frames" row is close to Δt_obs = 100 ms and gives R²_W = 0.30 — with K=5 BPTT to bridge the gap. So expect steep degradation, and decide up front whether the new runs use one-step at Δt_obs (harsher, cleaner test of the mismatch) or BPTT (comparable to the existing row). Pick one and say which; mixing them makes the sweep unreadable.

Cost: 3 GNN + 3 oracle, one batch. Worth it — it turns two isolated points into a curve in Δt_obs/τ, which is the quantity a reviewer can map onto their own microscope.

procced with 3 GNN + 3 oracle, one batch

---

## Prompt 4 — full-sample R² spec (0df48628, 2026-07-25 06:11)

Im working on the review: /workspace/connectome-gnn-cx/docs/neurips_review.txt of our paper /workspace/connectome-gnn-cx/docs/neurips.tex

at many points they ask full-sample R2: running
do you understand what is meant ?

to me:

**Spec: full-sample parameter metrics**

Definition. Same statistic as Supp. Eq. 23, no inlier restriction:

    R²_full(θ) = 1 − [ (1/N) Σ_{i=1..N} (θ_i − θ̂_i)² ] / Var_i(θ_i),   N = all neurons

Negative values are expected and fine — report them. R²_W needs no change (already unfiltered).

One decision to make first. Appx. F.2 says neurons with degenerate slope s_i → 0 have τ̂_i clipped to a physical range. Under filtering this is invisible; unfiltered, the clip bounds set the value. So report two variants and state which the table uses:

(A) clipped — τ̂ as currently produced. Bounded, but the number depends on the clip range.
(B) unclipped — raw −1/s_i. Honest, may be catastrophically negative.

I'd table (A) and give (B) in one sentence, noting the clip range explicitly. Presenting only (A) without saying so would be the same class of omission the reviewer is objecting to.

Rows to cover:

| table | rows | params |
|---|---|---|
| Tab. 1 | 3 Known-ODE + 3 GNN | τ̂, V̂_rest |
| Tab. 2 | 4 Known-ODE + 4 GNN | same |
| Supp. Tab. 4 | 10 GNN degradations | same |
| Supp. Tab. 7 | 8 Known-ODE degradations | same |

Per fold, then mean ± SD across folds — same aggregation as existing entries.

Cell format: `0.61 (0.12) [40.8]` = inlier, full-sample, excluded %. Caption: *"inlier R² (full-sample R²) [excluded %]"*. Matches Supp. Fig. 15, which already does this correctly.

we must make twin of the scripts in /workspace/connectome-gnn-cx/figures to generate new results table:

can you write new scripts in /workspace/connectome-gnn-cx/figures for full sampl R2
do not touch existing script they belong to the submitted version

---

## Prompt 5 — Jacobian figure + overnight no-GPU batch (a1cf7c4d, 2026-07-25 19:19)

do we need to rune new experiments ? its gonna be night for me so we should launche them now if necessary :

**Jacobian task (Vzfg Q4)**

Supp. Fig. 18 currently shows the Recurrent MLP only. Vzfg asks for the identical figure for the GNN and Known-ODE — "Presumably, 4 should be very easy."

```
# TASK: Jacobian comparison for GNN and Known-ODE, in the Supp. Fig. 18 format
# No training. Uses trained sigma=0.05 checkpoints (GNN + Known-ODE) and the
# existing held-out trajectory.

## Quantity
J_ij(t) = d/dv_j [ dv_i/dt ]   and   S_ik(t) = d/dI_k [ dv_i/dt ]

Ground truth (Eq. 1):
  J_ij = W_ij * 1[v_j > 0] / tau_i      (j != i)
  J_ii = -1/tau_i
  S_ik = delta_ik / tau_i               (retinal neurons only)

Known-ODE: same expressions with W_hat, tau_hat. Analytic, no autograd needed.

GNN (Eq. 2), by chain rule through f_theta and g_phi:
  J_ij = (df_theta/dm) * W_hat_ij * 2*g_phi(v_j,a_j) * (dg_phi/dv_j)   (j != i)
  J_ii = df_theta/dv_i
  S_ik = (df_theta/dI) * delta_ik
Use torch.autograd.functional.jacobian or vmap+grad on the message/update MLPs.
Do NOT finite-difference — the ReLU kinks make it unstable.

## Evaluation point (state-dependent, must be identical across models)
Jacobians depend on v(t) through 1[v_j>0]. Evaluate all three models
(GT, Known-ODE, GNN) at the SAME set of ground-truth frames:
sample N=200 frames uniformly from the held-out 8,000-frame trajectory,
compute J at each, report the mean over frames. State N in the caption.

## Panels (match Supp. Fig. 18 exactly)
Top row, stimulus Jacobian: 1736 retinal neurons x 1736 stimulus inputs.
  (a) ground truth  (b) Known-ODE  (c) GNN
Bottom row, weight Jacobian: presynaptic {Mi1, T4c, TmY15} x
  postsynaptic {T4a, TmY18, T4c}, 217 columns each (same types as Fig. 18).
  (d) ground truth  (e) Known-ODE  (f) GNN
Colormap: symmetric-log, linthresh=0.1, one shared colorbar per row.
Identical vmin/vmax to Fig. 18 so the two figures can be read side by side.

## Quantitative metrics (this is the part Fig. 18 lacks — report as a table)
For each model, over the block shown AND over all edges:
  1. R2 between learned J entries and GT J entries (identity line)
  2. Pearson r, same
  3. off-support mass: sum|J_ij| over pairs (i,j) NOT in the connectome,
     divided by total sum|J_ij|.  GT = 0 by construction.
  4. sign agreement: fraction of on-support entries with matching sign
Include the Recurrent MLP as the third row of this table using the existing
Fig. 18 computation, so GNN / Known-ODE / MLP are directly comparable.

## Deliverables
- fig_jacobian_gnn_knownode.pdf (6 panels)
- metrics table, 3 models x 4 metrics
- flag if GNN off-support mass is non-zero (it should be ~0 by construction
  since J is supported on the prior graph; confirm numerically)
```

**Also launch tonight** — no GPU, highest value:

1. Full-sample R² for τ̂, V̂_rest across Tabs. 1, 2, Supp. 4, Supp. 7 — AC demand #2, spec sent earlier.
2. Excluded-neuron characterization for the four worst conditions (40.8%, 56.3%, 49.1%, 67.9%): per-cell-type breakdown, activity stats, Jaccard overlap with the Appx. C degenerate set.
3. Reconcile 13,697 vs 13,741 — likely zero-out-degree neurons. One line, answers R1-W9.
