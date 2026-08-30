# GraphCast ideas for the recurrent-training benchmark

Handoff note. Context: [`weekend_benchmark_results_2026_08_29.md`](weekend_benchmark_results_2026_08_29.md)
Task 1 found rollout does **not** beat one-step (best arm = plain t+1 at 46% of the updates;
`pushforward` −0.082, `last` −0.028, everything else within the 0.015 floor).

GraphCast **agrees with that result**, and its curriculum says why. Worth reading before the next
rollout grid.

---

## 1. Where the code is

| what | path |
|---|---|
| GraphCast/WeatherNext source | `papers/weathernext/weathernext/utils/` |
| interaction network (edge + node update) | `utils/typed_graph_net.py:369–546`, `590–654` |
| 16-layer processor, residuals | `utils/deep_gnn.py:242–289`, `316–326`, `371–400` |
| MLP + LayerNorm block | `utils/dense.py:56–140`, `183` |
| multi-mesh (union of face sets) | `utils/icosahedral_mesh.py:79–95`, `366–389` |
| paper supplement | `papers/graphcast_supp.pdf` — §3.5 p.21 (processor), §4.2 p.25 (loss), §4.3–4.5 pp.26–27 (curriculum) |
| my math-vs-code writeup | [`graphcast_processor_math_vs_code.pdf`](graphcast_processor_math_vs_code.tex) |

Our side: `models/graph_trainer.py` (train loop, `lr_scheduler` at :415, clip at :585),
`models/training_utils.py:844` (`build_lr_scheduler`), `models/recurrent_step.py:435–452` (rollout).

---

## 2. What GraphCast actually does (supplement §4.3–4.5, p.27)

Three phases, **311,000 updates total**:

| phase | updates | AR steps | learning rate |
|---|---|---|---|
| 1 | 1,000 | 1 | linear warmup 0 → 1e-3 |
| 2 | **299,000** | **1** | half-cosine decay **to 0** |
| 3 | 11,000 | 2 → 12, **+1 every 1,000** | **fixed 3e-7** |

So **96% of updates are one-step**, and rollout is a 3.5% tail fine-tune at a learning rate
**3,300× below peak**. Loss is on *every* step, uniformly averaged over lead times, with **full
BPTT through the whole unrolled sequence** (no truncation).

Optimizer: AdamW, **β2 = 0.95** (not 0.999), weight decay 0.1 on weight matrices, **global
grad-norm clip at 32**, batch 32 sampled with replacement.

---

## 3. How that maps onto Task 1

| our arm | result | GraphCast |
|---|---|---|
| `pushforward` (BPTT window 1) | **−0.082** | ✗ never truncates BPTT — agrees |
| `last` (endpoint only) | **−0.028** | ✗ scores every step — agrees |
| `discount` γ=0.5 | −0.002 | ✗ uniform over lead times — agrees |
| `uniform` | −0.006 | ✓ this is GraphCast's objective |
| `onestep` | best | ✓ 96% of its updates |

**Both large effects in Task 1 point the same way GraphCast does.** The two arms that clear the
resolution floor are exactly the two things GraphCast refuses to do. That is a genuine independent
confirmation and worth saying in the paper.

**The mismatch is the schedule, not the objective.** Our rollout arms ran K ramping 1→5 across the
whole run at **constant `lr: 0.0018`** (`config/fly/flyvis_noise_005_calib_rollout_l4.yaml:60,78`).
GraphCast runs rollout for 3.5% of updates at 3e-7 — roughly **6,000× lower LR than ours**. That is
a precise explanation for the observation in the plan doc: *"rollout reached a higher R²\_W (0.983)
and got there ten times faster — then declined steadily as the trajectory got longer, ending at
0.961."* A long-horizon phase at full LR is exactly what produces that shape.

---

## 4. What to try, ranked

**1. Anneal the LR to zero. Do this first, it is nearly free.**
The results doc reports *one-checkpoint collapses of 0.2–0.5 in `R²_W`* (0.8985 → 0.3637 → 0.9104
in three consecutive rows) and works around them with trailing medians. That is a symptom of never
annealing. `lr_scheduler: linear_warmup_cosine` already exists (`training_utils.py:844`) but every
weekend config uses the default `'none'` → constant LR. Decaying to 0 should make the **final
checkpoint usable** and retire the trailing-median rule.

**2. Rollout as a tail fine-tune, not as the objective.**
Bulk of the run at K=1 with cosine decay, then a short phase with K ramping +1 per N updates at a
learning rate 3–4 orders of magnitude below peak. This is the one arm Task 1 did not test, and it
is the arm GraphCast actually uses. Predicted outcome: keeps the fast accurate start *and* the
horizon robustness.

**3. Global gradient-norm clipping.**
We clip only `model.W`, only when `grad_clip_W > 0`, which defaults to `0.0`
(`config.py:1320`, `graph_trainer.py:585`). GraphCast clips **all** parameters at norm 32. Another
plausible cure for the checkpoint collapses, and it is one line.

**4. Normalise the target by the variance of the *increment*, not the state.**
GraphCast's `s_j` = per-variable **inverse variance of the time difference**
`Var[x_{t+1} − x_t]^{-1}` (supplement §4.2, p.26), so every target is unit-variance *as an
increment*. We currently use `fit_reduction: mean` with coefficients divided by 70,467. Per-neuron
`1/Var_t(v_{t+1} − v_t)` is the principled version and makes one loss weight transfer across
neurons with very different activity levels. Cheap to add, and it is the same
non-dimensionalisation argument as the `λ`-carries-units-of-time² point.

**5. AdamW with β2 = 0.95.**
Faster second-moment adaptation than the 0.999 default; standard for this model class. Low cost,
low confidence — try only if 1–3 leave residual instability.

**6. Monotone K ramp tied to update count, not to epochs.**
Ours is `rollout_horizon_schedule: [1..10]` per epoch. GraphCast increments every 1,000 updates.
Minor, but it decouples the horizon schedule from epoch length, which matters when arms differ in
update count (the `onestep_step_matched` problem).

---

## 5. What **not** to take from GraphCast

- **The 16 message-passing steps.** Depth there is a physical requirement: a 6-hour jump needs
  information to cross far more than one mesh hop. Message-passing depth should equal *synaptic
  hops per Δt*, which at flyvis's Δt = 0.02 s is about 1. If a larger Δt needs more reach, use
  **substeps**, not depth — substeps keep each step interpretable.
- **The learned edge latent.** GraphCast gives every edge a 512-d residually-updated state, which
  dissolves the distinction between "how strongly connected" and "what was just communicated".
  Our multiplicative `W_ij · g_φ(v_j, a_j)` is what makes `W` recoverable; do not give that up.
- **No per-node embedding.** GraphCast has none — weather grid points have no hidden identity.
  Neurons do (`a_i`), so that asymmetry is ours to keep.

---

## 6. Suggested next grid

Smallest design that tests the above, `flyvis_A` / cv00–cv04, matched updates:

| arm | schedule |
|---|---|
| `A` control | K=1, constant LR (current `onestep`) |
| `B` anneal | K=1, linear warmup + cosine to 0 |
| `C` anneal + clip | B + global grad-norm clip |
| `D` graphcast | C, then tail phase K: 2→6 at LR/1000 for the last ~3% of updates |
| `E` increment-norm | D + per-neuron inverse-variance target weighting |

Report the **final checkpoint** as well as the trailing median — if annealing works, the two should
agree, and that agreement is itself the result.
