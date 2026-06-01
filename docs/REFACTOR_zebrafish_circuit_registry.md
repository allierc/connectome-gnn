# Refactor: circuit / task / IO-mapping registries (zebrafish branch)

A self-contained design + execution + status document so that a future
session can pick up this refactor without re-deriving the discussion.

**Context**: a zebrafish HD model was originally trained on a single
731-cell connectome pulled from neuprint-fish2. To add new cells (IPN12_a
+ IPN12_b = +108 cells -> 839 total) or investigate other tasks (optic
flow, visual reference, motor decoding), the original code path would
overwrite existing dataset / log directories and clobber the known-good
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

## Implementation status (2026-06-01)

Working branch: `feat/cx-observation`. Published-state tag
`zebrafish-tex-frozen` (commit `b6fa108`) protects the original
331-cell checkpoint. Three commits landed against the plan:

| Commit  | Phase  | Scope                                                                                       |
| ------- | ------ | ------------------------------------------------------------------------------------------- |
| `855388e` | Step 0 | Standalone zebrafish RNN + GNN (no drosophila inheritance, fish-native vocab)               |
| `2a7bf57` | Step 1+2 | Circuit registry + IPN12-extended 839-cell circuit + two new yamls + new §Circuit variants  |
| `c8bce8d` | (docs)   | Refresh §Functional comparison with mid-epoch-3 numbers                                     |
| `ae892f6` | (docs)   | Initial §Functional comparison draft                                                        |

### What's done — Step 0 (unparent zebrafish models)

`ZebrafishHdTaskRNN` and `ZebrafishHdTaskGNN` are now standalone
`nn.Module` subclasses with zero runtime imports from the drosophila
tree. The dynamics, sign-locking, 4-scalar afferent gate, and dIPN-only
readout are duplicated verbatim from `DrosophilaCxTask{RNN,GNN}`, then
renamed to fish-native vocabulary:

| Renamed attribute                    | Why                                  |
| ------------------------------------ | ------------------------------------ |
| `n_epg` → `n_dipn`                   | The bump pool is the dIPN ring       |
| `epg_ix`/`epg_indices`/`epg_glom_ix` | Same; `epg_*` aliases retained for back-compat with `graph_trainer.py:2237,2265-2266` |
| `output_from_epg_only` → `output_from_dipn_only` | yaml-side flag already existed at `config.py:572` |
| `pen_subpop_ix` (keys `PENa_L/R`, `PENb_L/R`) → `afferent_subpop_ix` (keys `RIPN_L/R`, `ptIPN_L/R`) | Habenula afferents = RIPN, pretectum = pt-IPN per Petrucco 2023 |
| `v_pena_l/r` → `v_ripn_l/r`          | habenula gate scalars                |
| `v_penb_l/r` → `v_ptipn_l/r`         | pretectum gate scalars               |
| `_pen_ind_*` → `_afferent_ind_*`     | non-persistent indicator buffers     |

NO state-dict back-compat shim was added (user's explicit instruction):
`zebrafish-tex-frozen` checkpoints (with `v_pena_*`/`v_penb_*` keys) do
NOT load into the new classes. Verification path was instead "fresh
retraining via `python GNN_Main.py -o generate_train
zebrafish_hd_si_dipn_bis` with eyeball-compare to prior metrics" — the
user confirmed the first-iteration snapshot from the new run is
byte-identical to the prior nominal training (panel `h` distribution
match).

`load_zebrafish_hd_connectome` in `connconstr_data.py` was extended
additively to emit fish-native aliases (`n_dipn`, `dipn_ix`,
`afferent_subpop_ix`) alongside the legacy keys; the 731-cell tables
produce J_effective sha = `7d3f17f462a653c2` unchanged.

### What's done — Step 1 (named-circuit registry, default-omit byte-equal)

New file `src/connectome_gnn/generators/circuits.py` (398 LOC, no
sub-package — user prefers a single file for now):

- `Circuit` dataclass: `name`, `N`, `neuron_types`, `type_names`,
  `J_effective`, `soma_xyz`, `subpops` (named index sets), `bump_ring_ix`,
  `dale_signs`, `provenance` (with `J_effective_sha256` auto-set on
  registration). Method `as_loader_dict()` returns the canonical
  loader-output shape (both fish-native + fly-vocab keys) so model
  classes can consume a Circuit via the registry while reusing the
  legacy `cx[...]` access pattern.
- Registry: `register_circuit(name, build_fn)`, `get_circuit(name)`
  (lazy-built + cached), `list_circuits()`.
- `_discover_circuits()` lazily imports/registers all built-in circuits
  on first lookup.
- "How to add a new circuit" recipe is at the top of the module file
  (4-step contribution: cache CSVs → build fn + register → optional
  yaml → optional docs/<organism>.tex section).
- Built-in: `_register_zebrafish_hd_731()` wraps the existing loader
  pointed at `figures/zebrafish/zebrafish_connectome_HD/`.

`src/connectome_gnn/config.py`: added `CircuitConfig` (single `name:
Optional[str] = None` field) + `NeuralGraphConfig.circuit:
Optional[CircuitConfig] = None`. Default-omit = today's loader path,
byte-equivalent.

`src/connectome_gnn/models/zebrafish_hd_task_rnn.py` +
`zebrafish_hd_task_gnn.py`: dispatch `config.circuit.name` →
`get_circuit().as_loader_dict()`, falling through to the legacy
`load_zebrafish_hd_connectome(sim.connconstr_datapath)` when unset.

`src/connectome_gnn/generators/graph_data_generator.py`: when
`circuit.name` is set, `_generate_swim_integration_task` writes
`circuit_provenance.json` next to the train/test TaskTrials zarrs.
JSON-only (user's choice — repo had no precedent for task-only dataset
provenance; the `ode_params.pt` pattern doesn't fit a task-only
generator). Carries `circuit_name`, `N`, `J_effective_sha256`, `dt`,
`task_family`, `type_count`, `n_bump_cells`, `source_provenance`.

`config/zebrafish/zebrafish_hd_si_dipn_v1.yaml`: explicit registry-path
variant of `zebrafish_hd_si_dipn`. Smoke test:
`hashlib.sha256(...)[:16] = "42df333a79e1725f"` whether the yaml has
`circuit: {name: zebrafish_HD_731_v1}` or not (byte-equal params via
both paths).

### What's done — Step 2 (839-cell IPN12 extension)

`connconstr_data.py`: extended `_zhd_category`, `cat_order`,
`_ZHD_BUMP_PREFIXES`, `_ZHD_INH_PREFIXES` to recognise IPN12_a /
IPN12_b and slot them into the bump pool with inhibitory Dale flip.
Design choices (user-confirmed, see Step-2 discussion in chat history):
  - IPN12 cells JOIN the bump ring (n_bump grows 443 → 551)
  - IPN12 outgoing weights Dale-flipped to inhibitory (same as
    IPNd/IPNds, consistent with IPN GABAergic biology)
  - 4-scalar afferent gate untouched (IPN12 receives no ω input)
  - Loader stays in `connconstr_data.py`; circuit build function in
    `circuits.py` calls it

731-cell tables (no IPN12 rows present) produce identical J_effective
sha (`7d3f17f4`) after the extension — no behaviour change.

`figures/zebrafish/fetch_zebrafish_connectivity_HD_IPN12.py`: one-off
fetcher mirroring `fetch_zebrafish_connectivity_HD.py` with type filter
extended by `IPN12_a` + `IPN12_b`. Live-run verified (user ran with
the hardcoded JWT from `fetch_cx_anatomy.py:170`, level="noauth" token
works against fish2): 839 neurons (731 HD + 51 IPN12_a + 57 IPN12_b),
22,425 edges, every IPN12 cell has both in- and out-edges,
HD↔IPN12 traffic ~30% of total. Output at
`figures/zebrafish/zebrafish_connectome_HD_IPN12/` (CSV pair; NOT
git-tracked, derived data).

`circuits.py`: added `_register_zebrafish_hd_ipn12_839()` reading the
extended CSV pair. Builds N=839, n_bump=551, J_effective sha =
`c2f6609aac39f6d7`.

`config/zebrafish/zebrafish_hd_si_ipn12_v1.yaml`: training recipe of
`dipn_v1` pointed at `zebrafish_HD_IPN12_839_v1`. W_out shape (2,551),
708,385 trainable params (vs 538k for the 731-cell circuit).

`docs/zebrafish.tex`: two new subsections at the end of §Results:
  - §Circuit variants and the IPN12 extension (in commit `2a7bf57`):
    describes the two registered circuits, the IPN12 design choices,
    the fetcher + loader paths.
  - §Functional comparison: HD ring with vs without IPN12 (in `ae892f6`,
    refreshed in `c8bce8d`): preliminary mid-epoch-3 numbers showing
    the 839-cell run is ahead on every metric (loss 0.0027 vs 0.0052,
    rmse_roll 1.9° vs 3.7°, FWHM 31° vs 45°-saturated, all with
    matched hyperparameters and seeds). Gate-scalar trajectory analysis
    showing the 839-cell run releases the pt-IPN gate scalars toward
    zero (W^rec absorbs angular integration earlier) while the 731-cell
    run is still inflating them.

PDF rebuilt to 19 pages and pushed.

### What's done — verification

Steps 0/1/2 are running live as of commit `c8bce8d`:

- `bsub … python GNN_Main.py -o generate_train zebrafish_hd_si_dipn_v1`
  (job 150351976) — 538k params, mid-epoch-3, no errors.
- `bsub … python GNN_Main.py -o generate_train zebrafish_hd_si_ipn12_v1`
  (job 150351979) — 708k params, mid-epoch-3, converging faster than
  dipn_v1 on every metric.

Both write to distinct dataset and log directories:
  - `${GNN_OUTPUT_ROOT}/graphs_data/zebrafish/zebrafish_hd_si_task_v1/`
  - `${GNN_OUTPUT_ROOT}/graphs_data/zebrafish/zebrafish_hd_si_task_ipn12_v1/`
  - `${GNN_OUTPUT_ROOT}/log/zebrafish/zebrafish_hd_si_dipn_v1/`
  - `${GNN_OUTPUT_ROOT}/log/zebrafish/zebrafish_hd_si_ipn12_v1/`

The original `zebrafish_hd_si_dipn` checkpoint is untouched (the §9
artefact-safety rule).

---

## What remains

### Immediate (epoch 10 of both runs)

1. **Refresh `docs/zebrafish.tex` §Functional comparison** with the
   epoch-10 (final) numbers. Three `%% TODO:` markers in
   `docs/zebrafish.tex` flag the exact spots:
   ```bash
   cd /workspace/connectome-gnn-cx
   grep -n '%% TODO' docs/zebrafish.tex
   ```
   Numbers to refresh:
   - Convergence table (loss, pi_acc, r_roll_1k, rmse_roll, FWHM)
   - Gate-scalar table (read from `best_model_with_0_graphs_9.pt` for
     each run; see `tests/scripts/zebrafish_step0_baseline.py` for the
     state-dict-unwrap pattern):
     ```python
     import torch
     blob = torch.load(p, map_location='cpu', weights_only=True)
     sd = blob['model_state_dict']
     sd = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v
           for k,v in sd.items()}
     for k in ('v_ripn_l','v_ripn_r','v_ptipn_l','v_ptipn_r'):
         print(k, float(sd[k]))
     ```
   - Add (or remove) the qualitative claim about pt-IPN gate-scalar
     trajectory based on what the final values say.

2. **Add a Figure to §Functional comparison** showing the gate-scalar
   trajectories over training (read from `tmp_training/metrics.log` +
   the saved snapshot checkpoints). One panel per circuit, four lines
   per panel (`v_ripn_l/r`, `v_ptipn_l/r`). Source data is on disk;
   figure source lives at `figures/zebrafish/fig_zebrafish_gate_scalars.py`
   (TODO: create this).

3. **Per-cell-type MI fingerprint comparison** — adapt
   `figures/zebrafish/fig_zebrafish_four_classes.py` to run on both
   converged checkpoints. Specifically: do IPN12_a / IPN12_b cells fall
   in the R/L/D/Z classes the way IPNd cells do, or do they form a
   distinct cluster? The 108 new cells either tile the bump ring with
   distinct preferred angles (interesting, supports the "they extend
   the HD code" story) or cluster narrowly (suggests they encode
   something else — switch / context / state — and joining the bump
   pool was the wrong design choice).

### Step 3 — `tasks/` registry (deferred per §5)

Required only when a second task family lands. Today the only task
is `swim_integration`. When the second task arrives:

- Create `src/connectome_gnn/generators/tasks.py` (single-file pattern
  matching `circuits.py` per user preference).
- `TaskSpec` dataclass: `name`, `n_input`, `n_output`, `dt`, build
  function returning `(stimulus, target, aux)`.
- Migrate `_generate_swim_integration_task` and
  `_generate_path_integration_task` to register against this registry.
- Add `task.name` to the yaml (with `task.task_type` kept as a
  deprecated alias).
- Models that wire IO from a TaskSpec (e.g. via the future IO mapping
  registry) consume `task.n_input` / `task.n_output` to size their
  encoder/decoder.

Skipped now because there is no second task family on the roadmap.

### Step 4 — `io_mappings/` registry (deferred per §5)

Required only when (a) >1 IO mapping is in use, or (b) IPN12 cells
need to be added to the input gate without forking the model class.
Currently the 4-scalar afferent gate (RIPN_L/R, ptIPN_L/R) is the only
IO mapping the zebrafish models support and it lives inline in
`ZebrafishHdTask{RNN,GNN}.__init__`.

If/when extracted:

- Create `src/connectome_gnn/generators/io_mappings.py`.
- `IOMapping` dataclass: `input_groups` (dict[str, ndarray]),
  `output_indices`, self-validation (asserts `len(input_groups) ==
  task.n_input`, etc.).
- Move the `pen_4scalar` gate construction + the dIPN-only readout
  slicing out of the model's `__init__` and into a function in
  `io_mappings.py`.
- Add `io_mapping.name` to the yaml.

Skipped now — single IO mapping in production use, no second one on
the roadmap.

### Minor cleanups (anytime)

- Migrate `eval_model.epg_indices` / `eval_model.epg_glom_ix` in
  `graph_trainer.py:2237,2265-2266` to the fish-native names. Currently
  the standalone zebrafish models keep `epg_*` as Python-attr aliases
  for trainer compatibility; renaming requires touching the drosophila
  path too (since the trainer is shared). Small mechanical change once
  the comparison results are written up.
- Deprecate the `velocity_gate: pen_4scalar` yaml token in favour of
  `velocity_gate: afferent_4scalar`. The internal handler is already
  named `_afferent_*`; only the yaml-side token kept fly vocab for
  back-compat. One-cycle deprecation warning + remove.
- Promote the hardcoded JWT in `fetch_cx_anatomy.py:170` /
  `fetch_optic_lobe_anatomy.py:89` to a shared module-level constant
  or env-var-only path. The token is a "noauth" public-read JWT (no
  user credentials), but having it in three fetcher files (HD,
  cx_anatomy, optic_lobe_anatomy) violates DRY.

### Future circuits (template)

To add a third zebrafish circuit (e.g. `zebrafish_HD_visual_v1` with
RGC inputs):

1. Fetch the extra cell types via a new
   `fetch_zebrafish_connectivity_HD_<NAME>.py` (mirror the HD or
   HD_IPN12 fetcher).
2. Extend `_zhd_category` + `cat_order` + `_ZHD_*_PREFIXES` in
   `connconstr_data.py` to recognise the new cell types.
3. Add `_register_zebrafish_hd_<NAME>()` to `circuits.py` (call from
   `_discover_circuits`).
4. Write a new yaml `config/zebrafish/zebrafish_hd_si_<NAME>_v1.yaml`
   with `circuit: {name: zebrafish_HD_<NAME>_v1}` and a distinct
   dataset name.
5. Add a section to `docs/zebrafish.tex` documenting the new circuit
   + biological rationale.

The Step 0+1+2 path is now generic enough that this is mechanical.

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
