# Refactor plan: circuit / task / IO-mapping registries (zebrafish branch)

A self-contained design + execution document so that a future session
can pick up this refactor without re-deriving the discussion.

**Context**: a zebrafish HD model is currently trained on a single 731-cell
connectome pulled from neuprint-fish2. To add new cells (IPN12_a + IPN12_b
= +108 cells -> 839 total) or to investigate other tasks (optic flow,
visual reference, motor decoding), the current code path overwrites the
existing dataset / log directory and would clobber the known-good
trained checkpoint. The refactor below isolates three axes — circuit,
task, IO mapping — so new combinations live next to old ones, never on
top of them.

**Companion documents:**
- `docs/REPRODUCE_zebrafish.md` — manifest pinning the current good state
  (commit b6fa108, conda env, checkpoint paths, per-figure commands,
  regression-check recipes).
- `docs/zebrafish.tex` — the published companion that this refactor must
  not break.
- `docs/drosophila.tex` — the sister document; useful as a reference for
  the methods prose we want to preserve.

---

## 0. Re-evaluation after the `feat/cx-observation` registry work (2026-05-31)

Today's drosophila_cx refactor exercised exactly the registry pattern this plan
proposes, end-to-end, and surfaced concrete refinements. **The core approach
(registries by name; defaults preserve behaviour; fail-loud validation;
hand-run golden checks) is validated — proceed.** Adjust as below.

**Validated, no change needed:**
- *Registries by name, not class hierarchies* (§2). We added `MODEL_FAMILY` and
  `FORWARD_KIND` dispatch tags and extended the `ode_params` registry; the
  by-name + "omitted field ⇒ today's behaviour" pattern held across
  trainer/tester/plot. `circuit.name` / `task.name` / `io_mapping.name` as
  **config choices** (§4) is the right call.
- *Fail loudly at startup* (§8). Implemented as `NeuralGNN.effective_W` raising
  in eval mode when the sign-lock is on but the GT sign is unset. Use the same
  loud-guard style for the IOMapping↔Circuit↔Task compatibility asserts.
- *Hand-run golden checks, no pytest infra* (§6). Every commit today was verified
  by re-running `-o test_plot` and diffing `results/metrics.txt` byte-for-byte.
  That worked well; keep exactly this.

**Refinements (new, from today):**
1. **Generation-properties live with the DATA, not config or class.** The teacher
   firing-rate nonlinearity (`activation`, sigmoid) is now saved *into*
   `ode_params.pt` and read by the grader — not a config field, not hardcoded in
   a class. **Apply to Circuit:** persist `provenance` **and** the circuit's
   structural fields into the dataset artifact (extend `ode_params.pt`; do not
   invent a parallel store), so a checkpoint is self-describing about which
   circuit + activation produced it. This upgrades §2.1's `provenance` from
   in-memory to saved-with-data.
2. **The three-way "where does it live" rule** (crystallised today, see memory
   `feedback_class_attr_dispatch_only`): **config = user choices/values** (the
   three `.name` fields, coeffs, lr); **class attribute = dispatch tag only**
   (`MODEL_FAMILY`/`FORWARD_KIND` — never tunable values); **data artifact
   (`ode_params.pt`) = how the data was generated** (activation, provenance,
   `J_effective`, neuron types). Decide placement of every new field this way.
3. **Don't duplicate the data schema.** `ode_params` already persists
   `W`/`edge_index` (↔ `J_effective`), `neuron_types`, `type_names`, and now
   `activation`. The Circuit dataclass is the **in-memory builder** the generator
   consumes; the **persisted record** is `ode_params.pt`. Keep their fields
   aligned. If zebrafish voltage data is generated, add a
   `ZebrafishVoltageODEParams` peer carrying `activation` (mirror
   `DrosophilaCxVoltageODEParams`).
4. **Generalise the activation-save.** Only `_generate_voltage_from_cx_task_model`
   records `activation` today; the generic / `cortex` / zebrafish voltage
   generators still write `FlyVisODEParams` without it. Fold this into Step 1 (the
   circuit wrapper) — cheap, and keeps zebrafish self-describing.
   `ZebrafishHdTaskRNN(DrosophilaCxTaskRNN)` already exposes `recurrent_activation`
   (default sigmoid).
5. **Sign-lock + recovery are now generic — free for zebrafish.**
   `restore_edge_sign_lock`, the `MODEL_FAMILY` recovery dispatch, and
   `recovery_param_metrics` work for any signed-W GNN. The Circuit's sign-locked
   `J_effective` plugs straight in; no per-circuit recovery code needed.

**Branch strategy (supersedes §9–§10).** Continue the refactor **directly on
`feat/janelia-cx`** — the single working branch; no separate
`feat/circuit-registry`. First **merge today's `feat/cx-observation` work into
`feat/janelia-cx`** so it carries the `MODEL_FAMILY`/`FORWARD_KIND`/`activation`
foundation (both diverge from `bb68679`; `feat/janelia-cx` just predates today's
registry work). The published zebrafish.tex state is protected by a **git tag +
the out-of-git checkpoints** (§9), NOT by freezing the branch: tag the published
commit *before* merging, and never overwrite a `log/`/`graphs_data/` dir
(`_v1`/`_v2` naming) so the tagged source + untouched checkpoints stay
reproducible.

---

## 1. The three axes

| Axis | Owns | Today (one of each) | Tomorrow (many) |
|---|---|---|---|
| **Circuit** | which neurons + how they connect (a neuprint subset + Dale convention + spectral rescale) | 731-cell HD pool from fish2 | + IPN12 (839 cells), or different ROI carving, or RGC + IPN, etc. |
| **Task** | the (input, target) sequence distribution | swim-integration -> 3-ch in, 2-ch HD out | optic flow, visual-reference, motor decoding (different in/out shapes) |
| **IO mapping** | which neuron indices receive input / drive output | 4-scalar afferent gate on RIPN_{L,R}/ptIPN_{L,R}, ring-only readout | could be RGC -> IPN, motor neurons -> tail, etc. |

The three are independent: a circuit carries no task info, a task carries
no circuit info, and the IO mapping is the glue that binds (circuit,
task) at training time.

---

## 2. Recommended abstraction

**Registries by name, NOT deep class hierarchies.** Pattern matches what
the model registry already does (`signal_model_name: zebrafish_hd_si` ->
`create_model(...)`).

- Each circuit is registered under a stable name
  (e.g. `zebrafish_HD_731_v1`, `zebrafish_HD_IPN12_839_v1`) -> a build
  function returns a `Circuit` dataclass.
- Each task is registered under a name (`swim_integration`,
  `optic_flow`, ...) -> a build function returns a `TaskSpec` dataclass.
- Each IO mapping is registered under a name
  (`hd_ring_4scalar`, ...) -> a function that, given a circuit, returns
  input groups and output indices.

A heavy class hierarchy adds method-table complexity for no gain; if a
circuit later needs custom behaviour (e.g. a custom Dale rule), it can
be promoted from a function to a class without breaking callers.

### 2.1 Dataclass fields

**Circuit:**
- `N: int` — total neuron count
- `neuron_types: ndarray[int]` — per-neuron type id
- `type_names: list[str]`
- `J_effective: ndarray[N,N]` — signed, sign-locked, spectrally rescaled
- `soma_xyz: ndarray[N,3]` — used for ring ordering + figures
- `subpops: dict[str, ndarray[int]]` — named index sets
  (e.g. `RIPN_L`, `IPNd13B`, `ring`, `afferent`)
- `provenance: dict` — server URL, dataset, fetch date, type list, Dale
  config; so a checkpoint can name the exact circuit it was trained on
- Optional `skeleton_paths: dict[bodyId, str]` — useful for downstream
  rendering, not needed for training

**TaskSpec:**
- `n_input: int`, `n_output: int`, `dt: float`
- `sample_batch(B, T, seed) -> (u, y, aux)` where `aux` carries
  `theta_hd`, `is_stop`, etc.
- `description: str`

**IOMapping:**
- `input_groups: dict[str, ndarray[int]]` — which neuron indices receive
  which input channel (today: 4 sub-pops x 1 swim channel; future: e.g.
  RGC indices x many visual channels)
- `output_indices: ndarray[int]` — which neurons feed `W_out`
- Self-validation: assert `len(input_groups) == task.n_input`, etc.

The model class (e.g. `ZebrafishHdTaskRNN`) becomes thin: it accepts
`(circuit_name, io_mapping_name)` from config, looks them up, and builds
`W_in`, `W_out`, sign-locked `W_rec` from those. The architecture itself
(sigmoid neuron, leak tau) lives where it lives now.

---

## 3. Where things go in the source tree

```
src/connectome_gnn/
+- circuits/
|  +- __init__.py            # registry: register_circuit(name, build_fn), get_circuit(name)
|  +- _base.py               # Circuit dataclass
|  +- zebrafish_hd_731.py    # wraps the existing loader -> "zebrafish_HD_731_v1"
|  +- zebrafish_hd_ipn12_839.py   # new circuit -> "zebrafish_HD_IPN12_839_v1"
+- tasks/
|  +- __init__.py            # registry
|  +- _base.py               # TaskSpec dataclass
|  +- swim_integration.py    # migrated from generators/_generate_swim_integration_task
|  +- path_integration.py    # migrated from generators/_generate_path_integration_task
+- io_mappings/
   +- __init__.py            # registry
   +- _base.py               # IOMapping dataclass
   +- hd_ring_4scalar.py     # current pattern: 4-scalar afferent gate + ring readout
```

Model classes (`DrosophilaCxTaskRNN`, `ZebrafishHdTaskRNN`) are unchanged
in name; they grow `circuit_name` / `io_mapping_name` config fields with
defaults that preserve current behaviour.

---

## 4. Configuration

Current yaml:

```yaml
dataset: drosophila_cx/zebrafish_hd_si_task
graph_model:
  signal_model_name: zebrafish_hd_si
task:
  task_type: swim_integration
  swim_integration:
    dt: 0.01
    n_steps: 1000
    ...
```

Proposed yaml (after refactor):

```yaml
dataset: zebrafish/zebrafish_HD_731_v1__swim_integration
graph_model:
  signal_model_name: zebrafish_hd_si
circuit:
  name: zebrafish_HD_731_v1         # NEW; defaults to "zebrafish_HD_731_v1" if omitted
io_mapping:
  name: hd_ring_4scalar             # NEW; defaults to "hd_ring_4scalar"
task:
  name: swim_integration            # renamed from task_type
  swim_integration:
    dt: 0.01
    n_steps: 1000
    ...
```

Back-compat rule: if `circuit.name` and `io_mapping.name` are omitted,
the code path is identical to today's. Existing yamls keep loading.

---

## 5. Migration path (small steps, each individually safe)

### Step 1 — register the existing pool under a name (HIGHEST PRIORITY)

The smallest commit that unlocks adding IPN12 without overwrite risk.

- Add `src/connectome_gnn/circuits/_base.py` (dataclass + registry).
- Add `src/connectome_gnn/circuits/zebrafish_hd_731.py` which is just a
  thin wrapper around `load_zebrafish_hd_connectome()` in
  `src/connectome_gnn/generators/connconstr_data.py`. The wrapper
  registers under `zebrafish_HD_731_v1`.
- Make `ZebrafishHdTaskRNN.__init__` accept `circuit_name` (default
  `zebrafish_HD_731_v1`); when set, look up via the registry; when
  absent, fall through the current code path. Both paths must produce
  byte-identical model parameters (see test D below).
- No new yamls; existing yamls unchanged.

Expected scope: ~50-100 LOC. If it grows, you are also doing step 2 or
4 — split the PR.

**Acceptance:** all four golden checks in §6 pass.

### Step 2 — add the new circuit

- `src/connectome_gnn/circuits/zebrafish_hd_ipn12_839.py` registers
  `zebrafish_HD_IPN12_839_v1`. The build function fetches / loads the
  443 IPNd*+IPNds* + 200 RIPN* + 88 pt-IPN* + 51+55 IPN12_{a,b} cells
  (108 of which already cached at
  `figures/zebrafish/zebrafish_anatomy_IPN12/`).
- New yamls: `config/zebrafish/zebrafish_hd_si_ipn12_v1.yaml` (and
  optionally `_tv` / `_gnn` variants). Point `circuit.name:
  zebrafish_HD_IPN12_839_v1`. Distinct dataset + log paths so the old
  checkpoints stay untouched.
- Train. New checkpoint lands under
  `log/zebrafish/zebrafish_hd_si_ipn12_v1/`. The old
  `log/zebrafish/zebrafish_hd_si_dipn/` is untouched -> the zebrafish.tex
  pipeline continues to reproduce.

### Step 3 — migrate tasks

Only when you have a second task family (optic flow, visual reference,
motor decoding). Until then `tasks/swim_integration.py` can just be a
thin re-export of the existing `_generate_swim_integration_task`.

When the second task lands:
- `tasks/<name>.py` with `TaskSpec` dataclass + build function.
- Yaml gains `task.name`; the current `task.task_type` becomes a deprecated alias.
- Model classes may need a new family if (n_input, n_output) differs
  from swim_integration's (3, 2).

### Step 4 — extract IO mapping from the model class

The biggest refactor; defer as long as possible.

- Move the 4-scalar afferent gate construction (`v_RIPN_L/R`,
  `v_ptIPN_L/R` initialisation + the `W_in` mask) and the HD-ring readout
  slicing out of `ZebrafishHdTaskRNN.__init__` and into a function in
  `io_mappings/hd_ring_4scalar.py`. The model class consumes an
  `IOMapping` and builds `W_in`, `W_out` from it.
- Only worth doing when (a) you have >1 IO mapping in use, or (b) you
  want to add IPN12 cells to the input gate without forking the model
  class.
- An interim that costs nothing: give the model class accessor methods
  like `_get_input_groups(circuit)` so the structure can be overridden
  cleanly later without changing call sites.

---

## 6. Test gates (the "good refactor" contract)

Four checks, three short scripts. Run by hand on every refactor commit;
no pytest infra needed.

### A. Connectome hash (Layer 1, ~30 LOC)

```python
import hashlib, numpy as np
from connectome_gnn.generators.connconstr_data import load_zebrafish_hd_connectome
cx = load_zebrafish_hd_connectome("figures/zebrafish/zebrafish_connectome_HD")
for k in ("J_effective", "neuron_types"):
    a = np.asarray(cx[k])
    print(k, a.shape, a.dtype, hashlib.sha256(a.tobytes()).hexdigest()[:16])
```

Baseline values get committed to e.g.
`tests/golden/zebrafish_HD_731_v1.json`. After refactor, regenerate via
the new registry path and diff. Any drift = stop.

### B. Checkpoint inference golden (Layer 2)

Pick one deterministic rollout: constant-omega at omega_deg=90, n_steps=1000,
seed=0, on CPU. Load the trained checkpoint
(`best_model_with_0_graphs_5.pt`), run, save the `decoded_hd` trace +
final per-neuron voltages as `.npz`. Commit as
`tests/golden/inference_check_zebrafish_hd_si_dipn.npz`.

After refactor: re-run, assert
`np.allclose(new, baseline, atol=0, rtol=0)`. Strict equality (CPU,
deterministic ops). This is the SINGLE test that answers "did I break the
trained checkpoint?".

### C. Training-trajectory golden (Layer 3)

Run a fresh short training: `n_epochs: 1, snapshots_per_epoch: 1`, small
`n_trials_train`, CPU, fixed seeds. Capture `metrics.log`. Baseline once
before refactor.

After: re-run, diff column-by-column. Strict equality on CPU; on GPU only
the convergence (final `r_roll_1k`, `pi_acc`, `fwhm`) within ~2 sigma of
seed-to-seed variation.

### D. Config-equivalence (cheap, run continuously)

Two yamls, identical except one omits `circuit:` and the other sets
`circuit.name: zebrafish_HD_731_v1` explicitly. Load both, build both
models, assert all model parameters are byte-equal. Catches the "default
not honoured" bug.

---

## 7. Acceptance criteria for "Step 1 ships"

When ALL of these are true, Step 1 (the wrapper + default circuit name)
is safe to merge:

- [ ] Test A: connectome hash identical pre- vs post-refactor.
- [ ] Test B: checkpoint inference byte-identical on CPU.
- [ ] Test C: short training run matches old `metrics.log` for >=100 iters.
- [ ] Test D: model params from omitted-`circuit` yaml == explicit-default-`circuit` yaml.
- [ ] Manual: one full `train_task` epoch on the new branch lands `r_roll_1k`,
      `pi_acc`, `fwhm` within numerical noise of the published metrics
      (see `docs/REPRODUCE_zebrafish.md` for the reference numbers).
- [ ] `docs/REPRODUCE_zebrafish.md` still reproduces the 5 zebrafish.tex
      figures byte-identically (or within matplotlib font drift; eyeball OK).

If any of these fail, the refactor is an unintended experiment. Roll
back, find the divergence, retry.

---

## 8. Risks and trade-offs flagged

- **Yaml verbosity grows.** Future yamls have three name fields
  (`circuit.name`, `task.name`, `io_mapping.name`) instead of one
  (`signal_model_name`). Mitigation: defaults preserve current behaviour
  for unmodified yamls.
- **More places for misconfiguration.** Mitigation: validation in the
  model `__init__` — assert IOMapping is compatible with both Circuit
  (indices in range) and Task (input/output dims match). Fail loudly at
  startup, not 1000 iters into training.
- **Some (circuit, task, IO) triples are nonsensical** (HD circuit +
  optic-flow task with HD IO mapping). Don't enumerate the cartesian
  product. Pin known-good triples in concrete yamls and document.
- **GPU non-determinism** breaks bit-exact training tests. Use CPU for
  bit-equality, GPU only for "within noise".
- **`_v1` suffix discipline matters.** If you re-derive the same logical
  pool but with a stricter Dale config, that's `_v2` and the old `_v1`
  runs stay reproducible. Never reuse a circuit name with different
  semantics.

---

## 9. Artefact safety (the "don't overwrite" rule)

This is what protects the published zebrafish.tex results regardless of
what the source code does:

1. **Tag the current state when the drosophila branch is done.** From
   the README of `docs/REPRODUCE_zebrafish.md` §0:
   ```bash
   git tag -a zebrafish-tex-2026-05-29 \
     -m "trained dIPN HD model + zebrafish.tex figures, frozen" b6fa108
   git push origin zebrafish-tex-2026-05-29
   ```
   This makes the exact source code one `git checkout` away forever.
2. **Refactor on a NEW branch** (`feat/circuit-registry`, not on
   `feat/janelia-cx`). The four tests above must pass before merging back.
3. **Never reuse log dir / dataset names.** New circuits get new yaml
   names (`zebrafish_hd_si_ipn12_v1.yaml`) -> new log dirs -> new
   checkpoints. The old `log/zebrafish/zebrafish_hd_si_dipn/` is read-only
   by convention; if you want to retrain the SAME circuit with new
   hyperparams, that's a `_v2` variant with a distinct config and a
   distinct log dir.
4. **The trained `.pt` checkpoint lives outside Git.** A tag pins the
   source, but the artefact is at
   `${GNN_OUTPUT_ROOT}/log/zebrafish/zebrafish_hd_si_dipn/models/`. If
   that disk path goes away, the published results can be regenerated
   from the tagged source by re-running the training command in
   `docs/REPRODUCE_zebrafish.md` §2. Same seeds -> bit-identical
   checkpoints on CPU, within-noise on GPU.

---

## 10. Concrete first action when restarting

```bash
# 1. Make sure you're on the right baseline.
cd /workspace/connectome-gnn-cx
git fetch
git checkout feat/janelia-cx
git log --oneline -3   # should include b6fa108 or later

# 2. Tag the published state (if not done yet) - see §9.

# 3. Create the refactor branch.
git checkout -b feat/circuit-registry

# 4. Capture the four golden baselines BEFORE writing any new code.
mkdir -p tests/golden
# A) connectome hash
python -c "..."  >  tests/golden/zebrafish_HD_731_v1.json   # see §6.A
# B) checkpoint inference
python tests/scripts/capture_inference_baseline.py          # writes .npz
# C) short training run
python GNN_Main.py -o train_task zebrafish/zebrafish_hd_si_dipn_smoke
cp ${GNN_OUTPUT_ROOT}/log/zebrafish/zebrafish_hd_si_dipn_smoke/tmp_training/metrics.log \
   tests/golden/training_smoke_metrics.log

# 5. Now start the refactor (Step 1 in §5).
mkdir src/connectome_gnn/circuits
# ... see §3, §5, §2.
```

The first time the four tests fail after a code change, that's where the
divergence is. Don't move on until they pass again. The refactor is then
mechanical.

---

## 11. Glossary of names used above

| Name | Meaning |
|---|---|
| `zebrafish_HD_731_v1` | Current 731-cell HD circuit (443 IPNd*+IPNds*, 200 RIPN*, 88 pt-IPN*). Trained checkpoint exists. |
| `zebrafish_HD_IPN12_839_v1` | Proposed new circuit: 731 + 51 IPN12_a + 55 IPN12_b = 839. SWCs already cached at `figures/zebrafish/zebrafish_anatomy_IPN12/`. |
| `hd_ring_4scalar` | Current IO mapping: 4-scalar afferent gate on `RIPN_{L,R}` and `ptIPN_{L,R}`, readout from the 443-cell ring. |
| `swim_integration` | Current task: noisy angular impulses as input, $(\cos\theta, \sin\theta)$ as target. n_input=3, n_output=2. |
| `feat/janelia-cx` | Current working branch (multi-document, drosophila + zebrafish). Published zebrafish.tex points to commit b6fa108 on this branch. |
| `feat/circuit-registry` | Proposed refactor branch. Lives off `feat/janelia-cx` until Step 1 is green, then merges back. |

---

_Owner: replace the placeholder commands in §10 with real ones as Step 1
is implemented, then this document becomes the post-mortem rather than
the plan._
