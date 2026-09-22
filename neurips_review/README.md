# NeurIPS-2026 rebuttal — model-misspecification experiments (concern #2)

Addresses the meta-review's point 2 ("the experimental setting is strongly
model-matched") and R1-W1/W2, R1-Q1, PVbi-¶3. We show what happens when the
generator leaves the GNN's hypothesis class along three axes the reviewers named.

All runs are **single-seed (cv00)**, templated on the Flyvis-217 **blank50
consensus triplet** — the paper's σ=0.05 / 50%-null-stimulus reference:
`flyvis_noise_005_blank50_{gen,unified,known_ode}_cv00`.

## The three tests

| Test | Misspecification | Knob (in `_gen_` config only) | Runs |
|---|---|---|---|
| **1. Δt mismatch** (R1-Q1) | Simulate at `dt/M`, observe+train at `dt`=20 ms; the discrete map is no longer shared | `n_generation_substeps=M`, `finite_difference_target=true`, `M∈{1,2,5,10}` | 4 GNN + 4 Known-ODE |
| **2. Monotonicity ablation** (R1-W2) | Remove the monotonicity priors the reviewer says are hard-coded | `coeff_f_theta_msg_diff=0` (μ₀, already 0) / `coeff_g_phi_diff=0` (μ₁) | 3 GNN (`ctrl`,`mu0off`,`mu1off`) |
| **3. Unobserved adaptation** (AC latent-variable item) | Add a slow hidden current `-g_a·c_i`, `dc_i/dt=(v_i-c_i)/τ_a`, `c_i` never observed → latent variable outside the graph, violates first-order-in-observables | `adapt_g∈{0,0.1,0.3}`, `adapt_tau_ms=200` | 3 GNN + 3 Known-ODE |

We do **not** ablate the squared message `g_φ²`: it is the sign factorization
(magnitude via `g_φ²≥0`, sign carried by unconstrained `Ŵ∈ℝ`), not a monotonicity
prior — that point is answered analytically in the reply text, not by experiment.

Total: **17 cluster train+test+plot jobs** + **7 local dataset generations**
(Test 2 reuses `flyvis_noise_005_blank50_cv00`).

## Design decisions

- **Finite-difference target (Test 1).** The default trainer target is the stored
  *analytic* derivative `pde(x_t)`; if we kept it, finer substeps would just hand
  the GNN the exact true derivative (null experiment). So `_gen_` configs store
  `y=(v[t+Δt]−v[t])/Δt` — what a Δt observer can actually measure. Verified against
  real data: at σ=0.05 the finite-diff label correlates 0.965 with the analytic
  derivative (unbiased), so weight recovery (averaged over 64k frames) stays high
  while the M>1 curvature bias surfaces as **slope≠1** — report the slope.
- **Process-noise scaling.** M substeps of `h=Δt/M` each get noise `σ/√M`, so the
  per-observed-frame variance matches the base σ=0.05 across all M (verified by the
  smoke test).
- **Cluster decoupling.** Only generator code changed (locally). The 4 new
  `SimulationConfig` fields live **only** in the `_gen_` configs used locally; the
  cluster-side `_unified_`/`_known_ode_` training configs are clean of new keys, so
  the (possibly older) cluster checkout never parses them. Nothing to sync.
- **Known-ODE oracle** (`flyvis_known_ode`) trains on the same modified data, so it
  faces the same misspecification — isolating inverse-problem degradation from GNN
  capacity (Tests 1 & 3).

## Sanity gates (must hold)

- Test 1 `M=1` finite-diff: `x_list` **bit-identical** to base; `R²_W` ≈ base 0.99.
- Test 3 `g_a=0`: identical to base.
- Test 2 `ctrl` ≡ `mu0off` (μ₀ was already 0) — a same-seed reproducibility check;
  the real ablation is `mu1off`.

## How to run

```bash
cd /workspace/connectome-gnn-cx
ENV="GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData PYTHONPATH=src"
PY=/workspace/.conda_envs/neural-graph-linux/bin/python

# 1. emit all 24 YAMLs into GraphData/config/fly + manifest.json
$PY neurips_review/gen_configs.py

# 2. (optional) tiny-scale correctness check of the generator knobs
env $ENV $PY neurips_review/smoke_test.py

# 3. generate the 7 datasets locally on the 2 RTX A6000 GPUs
env $ENV $PY neurips_review/generate_local.py

# 4. submit the 17 train+test+plot jobs to gpu_a100, poll, collect
$PY neurips_review/launch_cluster.py            # --dry-run to preview bsub cmds

# 5. (re)collect metrics anytime
$PY neurips_review/collect_metrics.py           # -> results_table.csv
```

## Files

- `gen_configs.py` — emits configs + `manifest.json`
- `generate_local.py` — 2-GPU pool for the 7 datasets
- `launch_cluster.py` — ssh→bsub for the 17 jobs, polls to completion
- `collect_metrics.py` — parses `results/metrics.txt` + `results_test.log` → `results_table.csv`
- `smoke_test.py` — tiny-scale generator invariants
- `manifest.json` — machine-readable run list (written by `gen_configs.py`)

## Metrics reported (per run)

`R²_W`, weight **slope**, `W_structure_r`; `R²_τ` (full-sample **and** inlier +
outlier %); `R²_Vrest` (full + inlier + outlier %); one-step r; 8000-frame rollout
r; cluster accuracy. Full-sample τ/Vrest R² are emitted natively, covering the
reviewers' "report unfiltered metrics" ask for free.
