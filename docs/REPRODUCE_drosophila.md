# Reproducing the figures and results in `docs/drosophila.tex`

This manifest pins down everything needed to regenerate the main figures and
all numerical statements in `docs/drosophila.tex` from a clean checkout.
It is intentionally precise about commit ids, conda env, seeds, trained
checkpoint paths, and per-figure command lines, so the voltage-supervision
refactor branch (see `REFACTOR_drosophila_voltage.md`) can rerun this
pipeline and diff its outputs against the artefacts that ship with the
document.

---

## 0. Reference state

| Item | Value |
|---|---|
| Repo HEAD when this doc was written | `TBD-COMMIT-SHA` (`TBD-SHA7`) on `feat/janelia-cx`, committed `TBD-COMMIT-TIMESTAMP` |
| Tag to add when the branch is in a stable state | `git tag -a drosophila-tex-2026-05-29 -m "trained drosophila_cx 6-condition CV + drosophila.tex figures, frozen" TBD-COMMIT-SHA` (push with `git push origin drosophila-tex-2026-05-29 --no-verify`) |
| Active conda env (CLI prompt: `(neural-graph-linux)`) | `/workspace/.conda_envs/neural-graph-linux` |
| Activate from a fresh shell | `source /opt/conda/etc/profile.d/conda.sh && conda activate /workspace/.conda_envs/neural-graph-linux` |
| Repo root for every command below | `cd /workspace/connectome-gnn-cx` |
| GraphData root (logs + trained models + task data) | `GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData` (already set in `devcontainer.json`) |

> NB: the trained checkpoints and the generated task data live OUTSIDE the
> git repo, under `${GNN_OUTPUT_ROOT}`. Branch state alone is not enough
> to reproduce — the artefacts at the paths below have to be present too.

---

## 1. Cached input data (independent of the trained model)

### 1a. Cached hemibrain CX anatomy + connectome

Fetched once via the scripts below; the resulting CSVs / SWCs / OBJs ship in
the repo and never change unless re-fetched.

```
figures/drosophila_cx/drosophila_cx_anatomy/        # SWC skeletons + OBJ ROIs for 152 CX neurons (EPG / PEN_a / PEN_b / Delta7 / PEG)
figures/drosophila_cx/drosophila_cx_connectome/     # neurons.csv (152 rows) + connections.csv with hemibrain bodyIds
```

If those folders are present, **skip the fetch step**. To regenerate from
scratch (requires `NEUPRINT_APPLICATION_CREDENTIALS` env var and a host
that can reach `neuprint.janelia.org`, currently gated by
`.devcontainer/init-firewall.sh`):

```
python figures/drosophila_cx/fetch_drosophila_cx_anatomy.py
python figures/drosophila_cx/fetch_drosophila_cx_connectivity.py
```

### 1b. Generated task data (path-integration)

For training and for analyses that need a deterministic rollout, the
dataset `drosophila_cx/drosophila_cx_pi_task` (and its `_tv` variant) must
exist under `${GNN_OUTPUT_ROOT}/graphs_data/`. The seed that produced the
existing data is captured inside the `_complete` marker; regenerating with
the same `simulation.seed` and `task.path_integration.seed` yields
identical zarrs.

```
python GNN_Main.py -o generate drosophila/drosophila_cx_known_ode_notv    # base recipe (paper anchor)
python GNN_Main.py -o generate drosophila/drosophila_cx_known_ode_tv      # TV-regularised recipe
python GNN_Main.py -o generate drosophila/drosophila_cx_gnn_notv          # GNN base
python GNN_Main.py -o generate drosophila/drosophila_cx_gnn_tv            # GNN +TV
python GNN_Main.py -o generate drosophila/drosophila_cx_fc_notv           # fully connected ablation
python GNN_Main.py -o generate drosophila/drosophila_cx_frozen_wrec_notv  # frozen-W^rec control
```

---

## 2. Trained model

The figures load 6 CV banks (cv00..cv09) per architecture:

| | |
|---|---|
| Config name | `drosophila_cx_known_ode_notv` (paper anchor) |
| Yaml source | `config/drosophila_cx/drosophila_cx_known_ode_notv.yaml` |
| Log dir | `${GNN_OUTPUT_ROOT}/log/drosophila_cx/drosophila_cx_known_ode_notv_cv0/` (and `_cv1` .. `_cv9`) |
| Checkpoints | `models/best_model_with_0_graphs_{1..N}.pt` (one per epoch) |
| Default checkpoint used by figure scripts | highest-epoch glob, picked by `_load(...)` in `figures/drosophila_cx/*.py` |

The other five configs follow the same `<config>_cv{i}` layout:
`drosophila_cx_known_ode_tv_cv{0..9}`, `drosophila_cx_gnn_notv_cv{0..9}`,
`drosophila_cx_gnn_tv_cv{0..9}`, `drosophila_cx_fc_notv_cv{0..9}`,
`drosophila_cx_frozen_wrec_notv_cv{0..9}`.

Train from scratch (≈ 4 h per fold on l4, ≈ 40 h per architecture x 10 folds, run as 10 parallel bsub jobs):

```
python GNN_Main.py -o train_task drosophila/drosophila_cx_known_ode_notv
python GNN_Main.py -o train_task drosophila/drosophila_cx_known_ode_tv
python GNN_Main.py -o train_task drosophila/drosophila_cx_gnn_notv
python GNN_Main.py -o train_task drosophila/drosophila_cx_gnn_tv
python GNN_Main.py -o train_task drosophila/drosophila_cx_fc_notv
python GNN_Main.py -o train_task drosophila/drosophila_cx_frozen_wrec_notv
```

Cluster invocation uses relative paths (see project memory): `bsub ... "python GNN_Main.py -o train_task drosophila/<config>"`. Seeds (`training.seed`, `task.path_integration.seed`, `simulation.seed`) come from the yaml and are rotated per fold inside `cv_runner.py:262-272` (`seed=base+i`, `training_seed=base+1000+i`); do not change them between baseline and comparison runs.

---

## 3. Per-figure commands

Each command writes to `figures/drosophila_cx/<scriptname>.png`. Run from the
repo root with the env active. Figure 7 (`cx_four_classes`) is the
prerequisite for the per-class summary in the discussion section.

### Figure 1 — `fig_drosophila_cx_w_rec_comparison.png` (per-edge W_rec recovery, ground-truth vs. all 6 conditions)

```
python figures/drosophila_cx/fig_drosophila_cx_w_rec_comparison.py --condition all --fold mean
```

Reads cv00..cv09 ckpts for each condition; computes per-edge R^2 against `J_effective` from `load_drosophila_cx_connectome`. No external data.

### Figure 2 — `fig_drosophila_cx_function_dynamics.png` (learned f_theta / g_phi curves vs. sigmoid)

```
python figures/drosophila_cx/fig_drosophila_cx_function_dynamics.py --model drosophila_cx_gnn_notv --fold 0
```

GNN-only. Reads checkpoint, dumps `f_theta(h)` and `g_phi(h)` on a 1-D grid, overlays the ground-truth sigmoid leak.

### Figure 3 — `fig_drosophila_cx_kinographs.png` (EPG ring bump kinographs, all conditions)

```
python figures/drosophila_cx/fig_drosophila_cx_kinographs.py --omega_deg 60 --n_steps 1000
```

Pulls `snapshot_omega_deg=60` deterministic rollout per condition; uses `epg_glom_ix` for ring reordering.

### Figure 4 — `fig_drosophila_cx_traces.png` (cos/sin output traces vs. ground truth)

```
python figures/drosophila_cx/fig_drosophila_cx_traces.py --omega_pattern stops --n_steps 2000
```

Uses the stops-on path-integration regime (see `task.path_integration.stop_fraction`).

### Figure 5 — `fig_drosophila_cx_hd_mi_summary.png` (per-cell-type MI with heading)

```
python figures/drosophila_cx/fig_drosophila_cx_hd_mi_summary.py
```

Computes `MI(sigmoid(h_i), theta_hd)` per neuron, aggregated by `neuron_types` (EPG, PEN_a, PEN_b, Delta7, PEG). Reads cv0 by default for each condition; switch with `--fold`.

### Figure 6 — `fig_drosophila_cx_omega_mi_summary.png` (per-cell-type MI with omega)

```
python figures/drosophila_cx/fig_drosophila_cx_omega_mi_summary.py
```

Same layout as Figure 5 but targets the OU angular velocity `omega`. Reads `trials.omega` from the generated task zarr.

### Figure 7 — `fig_drosophila_cx_four_classes.png` (R/L/D/Z partition + chi^2)

```
python figures/drosophila_cx/fig_drosophila_cx_four_classes.py
```

Writes `fig_drosophila_cx_four_classes.csv` (consumed by the discussion-section per-architecture totals). Defaults: `--model drosophila_cx_known_ode_notv`, decision-tree thresholds via `--mi_q / --w_q / --omega_q` quantiles inside the script.

### Numerical claims block

Bulleted list of every number quoted in `drosophila.tex` that regenerates from these scripts:

- 46 EPG / 16 glomeruli ring (Figure 3, `n_epg` from `cx['n_epg']`).
- N = 152 neurons (`load_drosophila_cx_connectome`).
- 4 PEN subpopulations (`pen_subpop_ix.keys()` -> PENa_L/R, PENb_L/R).
- Per-architecture CV-mean `r_roll_1k`, `bump_FWHM`, `chi^2 (cos/sin)`, four-classes total (Figure 7 csv).
- TV vs no-TV deltas (Figures 1, 5, 7).

### Agentic-loop provenance

For every "X fails" claim in `drosophila.tex` (e.g. "fully connected RNN rarely converges", "frozen-W^rec control underperforms"), cite the agentic search breadth that produced it: configs swept, seeds tried, LR grid. The relevant logs live under `log/Claude_exploration/LLM_drosophila_cx/`. Negative results without a logged sweep do not belong here — see `project_drosophila_agentic_loop_role.md` in user memory.

---

## 4. Build the PDF

```
cd docs
pdflatex -interaction=nonstopmode drosophila.tex
pdflatex -interaction=nonstopmode drosophila.tex   # second pass for refs
```

Two passes are required because `cleveref` resolves cross-references on the second run. The expected page count for this revision of the doc is **TBD pages** (≈ TBD MB PDF).

---

## 5. Minimal regression check on a refactor branch

Before merging anything that touches `generators/`, `models/`, or the
connectome loader, confirm the three pillars from the design discussion:

**A. Task data (Layer 1) — connectome hash:**

```
python - <<'PY'
import hashlib, numpy as np
from connectome_gnn.generators.connconstr_data import load_drosophila_cx_connectome
cx = load_drosophila_cx_connectome("figures/drosophila_cx/drosophila_cx_connectome")
for k in ("J_effective", "neuron_types"):
    a = np.asarray(cx[k])
    print(k, a.shape, a.dtype, hashlib.sha256(a.tobytes()).hexdigest()[:16])
PY
```

**B. Inference (Layer 2) — checkpoint produces identical decoded HD:**

A deterministic constant-omega rollout from the trained checkpoint
(e.g. via `_deterministic_sweep_rollout(net, 1000, 60.0, 'cpu')` from
`drosophila_cx_eval.py`) should produce a `decoded_hd` array that is
bit-identical before and after a refactor on CPU. Stash the array as a
`.npz` baseline once; re-run and assert
`np.allclose(..., atol=0, rtol=0)` after the refactor.

**C. Training trajectory (Layer 3) — first 100 iters of metrics.log:**

Run a short `train_task` with the existing seeds and small
`n_trials_train`; capture the first 100 rows of `metrics.log`; diff after
the refactor. CPU bit-equal, GPU within numerical noise. Catches any
unintended init-order or RNG-order shift.

**D. Fold-mean tolerance** on `r_roll_1k`: across cv00..cv09 the mean must
stay within `|delta| < 0.01` of the pre-refactor value for each of the 6
architectures (read from `fig_drosophila_cx_four_classes.csv`). Total
four-classes count per architecture must match exactly.

**E. PDF page count** after `pdflatex` x 2: identical to the pre-refactor
value (TBD pages).

These checks are cheap (< 30 min total on cpu, modulo the cv mean which
requires the 60 trained ckpts to be cached) and they jointly say "the
trained checkpoint still belongs to the same scientific point" — which
is what you actually want from a refactor.

---

## 6. What this manifest does NOT cover

- matplotlib / font drift between machines — treat arrays as ground truth, not pixels.
- Animation and companion scripts not referenced by `drosophila.tex` (`fig_drosophila_cx_anim_*.py`, `fig_drosophila_cx_pca_ring.py`, `fig_drosophila_cx_readout_mi.py`); their commands are still in earlier git history if you need to regenerate any of them.
- Companion overlay figures (`fig_drosophila_cx_IPN12_companion.png` etc.) that are not referenced by the tex.
- The zebrafish circuit registry refactor (`REFACTOR_zebrafish_circuit_registry.md`) — drosophila stays on the legacy `_load_connectome` hook until a second CX circuit (FB / NO) is introduced.

---

## 7. LFS-vs-no-LFS handling

Mirrors the zebrafish manifest's **no-LFS** path: only small CSVs / SWCs /
OBJs ship in-repo (`figures/drosophila_cx/drosophila_cx_anatomy/`,
`figures/drosophila_cx/drosophila_cx_connectome/`). Everything heavy —
trained checkpoints (60 ckpts: 6 archs x 10 folds), generated zarrs, log
dirs — lives under `${GNN_OUTPUT_ROOT}` (NFS-mounted
`/groups/saalfeld/home/allierc/GraphData`). The manifest never invokes
`git lfs`; large artefacts are pinned by *path + content hash* (sha256[:16]
in Section 5A) and by the dataset's `_complete` marker rather than by an
LFS pointer.

Devcontainer-local push: `git-lfs` is not installed inside the dev
container, so `git push` fails the pre-push hook. Per project policy
(`feedback_git_lfs_push.md`), use `git push --no-verify` automatically
without asking — no LFS objects are produced by this manifest.

---

_Owner: this file should be updated whenever any of the 6 CV banks is
retrained, the figure-generating scripts change their default args, or
the tex includes a new figure. Treat it as the contract between the
analysis pipeline and the document._
