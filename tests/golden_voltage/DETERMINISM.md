# Determinism phase: data_generate_voltage golden harness

Question: can two runs of the BASE implementation (`BASE_SHA` = 496de14b)
produce byte-identical output, and under which conditions? The answer sets the
comparator policy in `manifest.py` and the conditions `driver.py` imposes on
every golden run. All numbers below were measured on 2026-09-25 with
`python scripts/golden_voltage.py determinism`; raw results are in
`<work>/determinism/det1/determinism.json`.

## Environment

| item | value |
|---|---|
| machine | Apple M4 Pro, 14 cores, macOS 26.6.2 |
| env | conda `flyvis-gnn-mac`: Python 3.12.13, torch 2.7.1, numpy 2.4.3, scipy 1.14.1, matplotlib 3.9.4, tensorstore 0.1.65, zarr 2.18.7, Pillow 12.1.1, datamate 1.0.0 |
| flyvis | editable checkout `~/Projects/flyvis` at 7c29aed (origin/main of lappalainenj/flyvis) |
| ffmpeg | 8.1 (Homebrew) |
| installed for the harness | `coverage` 7.16.1 (`pip install "coverage[toml]"` into flyvis-gnn-mac) |

## Conditions every golden run gets (driver.py, cell_runner.py)

* A fresh interpreter per run, `cwd` = the run directory, and an environment
  built from scratch: only PATH/HOME/USER/LANG/TMPDIR are inherited;
  `PYTHONHASHSEED=0`, all BLAS/OpenMP thread caps and `torch.set_num_threads`
  at 1, `MPLBACKEND=Agg`, `TZ=UTC`, `SOURCE_DATE_EPOCH=0`, `PYTHONNOUSERSITE=1`.
* `PYTHONPATH=<impl>/src:<impl>` and an assertion that `connectome_gnn` is
  imported from `<impl>` (the env's editable install points at
  `~/Projects/connectome-gnn/src`).
* **flyvis imported with MPS hidden** (`flyvis_pin.py`). On this Mac flyvis sets
  `flyvis.device = mps`; `BoxEye` then moves frames to MPS while its conv
  weights stay on the CPU, every DAVIS/Sintel sequence fails with "Input type
  (MPSFloatType) and weight type (torch.FloatTensor)", `RenderedDavis` logs and
  swallows the error, and dataset construction ends in "No sequences were
  successfully rendered". So `data_generate_voltage` with video input cannot run
  in flyvis-gnn-mac without this pin. It is a harness condition, identical for
  base and head; the generator's own `device` argument is unaffected.
* Its own `FLYVIS_ROOT_DIR`, cloned from a fixture template that already holds
  the bundled model 000, the extent-8 and extent-15 connectome caches, the
  NetworkView cache and every rendering the cells need (pre-rendered with the
  base code). `RenderedSintel` stores `sintel_path=<FLYVIS_ROOT_DIR>/SintelDataSet`
  in its config, so the cloned Sintel cache's `_meta.yaml` is rewritten to the
  run's root; otherwise every Sintel run misses the cache (see "cache" below).
* stdlib `random` seeded with 0xC0FFEE before the call (see "unseeded" below).

## Results

| experiment | cells | result |
|---|---|---|
| old x old, different output roots | 76 pairs covering all 74 cells of the full tier, incl. hybrid, Sintel, extent 15 (two cells were rerun after their definition changed) | **identical**: every file byte-for-byte (1981 zarr metadata/chunk files, 252 PNG, 83 `.pt`, 80 text, 5 MP4, 2 other), directory trees, normalised logs, state.json |
| threads 1 vs 4 | fast tier (incl. new F3) | identical |
| PYTHONHASHSEED 0 vs 1 | fast tier (incl. new F3) | identical |
| warm vs cold rendering cache | fast tier + sintel_only_noise + mixed_davis_stopiter | **differ in all 12** (mechanism below) |
| stdlib random not pre-seeded, 2 processes | sintel_null_edges, F6_edges_random, null_random_attempts | Sintel cell **differs** (null-edge `edge_index` in ode_params.pt, python RNG digest); both DAVIS cells identical |
| MPS x MPS | fast tier | **differ in all 10** |
| MPS x MPS with `torch.use_deterministic_algorithms(True)` | F1, F3, F6 | **differ in all 3** (no error raised) |
| CPU x MPS | fast tier | differ in all 10 (expected) |

### Cold vs warm rendering cache

On a cache miss `RenderedDavis.__init__` / `RenderedSintel.__init__` builds a
flyvis `BoxEye`, whose `_set_filter` constructs `nn.Conv2d(1, 1, 13)`. The
default Conv2d initialisation draws weight and bias from the torch global RNG,
and this happens inside `data_generate_voltage` after
`torch.random.manual_seed(sim.seed)`. With a warm cache datamate returns the
existing directory without running `__init__`, so nothing is drawn. The rendered
frames are the same either way; what changes is every later torch draw. In the
data: process and measurement noise, AR(1), mixed-mode noise frames,
`only_noise` stimuli, and the final torch RNG state (the only difference in F1,
F2, F6 and F8, which draw no torch noise). **This is a pre-existing
reproducibility defect: the first generation after a cache miss (a new machine
or a new FLYVIS_ROOT_DIR) cannot be reproduced by rerunning with the same
seed.** Policy: golden runs are always warm. Both implementations would shift the
same way, but the cold path is outside the golden scope.

### stdlib random

Confirms plan finding 2. `data_generate_voltage` never seeds stdlib `random`;
the DAVIS dataset does (`davis.py:919`, `random.seed(shuffle_seed)`) as a side
effect, so DAVIS null-edge draws are reproducible, while Sintel null-edge draws
are not reproducible across processes. The runner's pre-seed makes the Sintel
cells deterministic and does not change DAVIS cells, because the dataset
re-seeds.

### MPS

MPS runs are not reproducible. Voltages differ by 1 ulp (max |dv| 1e-6) from
the first integration step, on non-photoreceptor neurons only, and `y_list` by
up to 1.6e-4. This pattern fits a non-deterministic message
aggregation (`scatter_add_`), and `use_deterministic_algorithms(True)` does not
fix it on MPS. Under the PI rule (old x old on MPS must be byte-identical
before an MPS cell can be golden) **no MPS cell is golden; every golden cell
runs on CPU.** The stimulus-buffer alias (plan finding 3) is also CPU-only, since
`.to("mps")` copies, so CPU and MPS runs differ in F9 semantics, not only in
rounding.

## Comparator policy (manifest.POLICY)

| kind | comparator | why |
|---|---|---|
| zarr metadata and chunks, `.pt`, text, other | raw bytes | stable old x old everywhere; on mismatch zarr arrays are decoded (first differing index, max abs diff) and `.pt` files are loaded key by key, with storage sizes (plan finding 8) |
| PNG | raw bytes | 252/252 byte-identical old x old; the decoded RGBA digest is also recorded |
| MP4 | decoded: `ffmpeg -f framemd5` + stream parameters | bytes were also stable (5/5), but the container records the muxer version, which belongs to the environment rather than the implementation |
| `.generate_done` | bytes with `date:` dropped and `git_sha:` masked | only in the dispatch cell |
| directory tree | exact, including empty directories | `Fig/` must exist and be empty |
| state.json `compared` | exact | RNG digests (torch CPU, numpy, python), grad mode, rcParams digest, root and flyvis logger levels, default device, exception type and message (paths normalised) |
| stdout / stderr | exact after `normalize_log` | tqdm segments, timestamps and `module:lineno` / warning line numbers removed; run, implementation and work paths replaced; ANSI colours kept (they encode B37) |

## Facts measured here that change the plan's assumptions

* **No dataset access draws RNG in this configuration** (the harness self-test
  shows it; see mutants.py). DAVIS datasets are built with `augment=False`.
  Sintel's `__getitem__` jitter, noise and gamma draws are guarded by
  `if self.<x>_std`, all `None` here, and `n_rot=0, flip_axis=0` are passed
  explicitly. So plan finding 2's "Sintel augmentation draws on every access"
  is wrong for this flyvis commit, and the number and order of
  `stimulus_dataset[i]` accesses is not currently observable. It becomes
  observable as soon as any augmentation std is set.
* `truncate_max_frames` never reaches `RenderedDavis` (`MultiTaskDavis` does not
  pass `max_frames`), so it is a no-op on the DAVIS path.
* `CombinedVideoDataset` has no `arg_df`, so two `datavis_roots` fail at the
  train/test split with AttributeError (cell combined_roots).
* Every saved tensor (`ode_params.pt`, `weights_full.pt`, `edge_index_full.pt`,
  `kept_edge_indices.pt`, `ablation_mask.pt`) owns exactly its storage, so
  adding `.clone()` before a save is not observable. Saving a view of a larger
  storage is observable (mutant W_view_of_larger_storage).
* Forward Euler on the conductance twin diverges to NaN within the run
  (cell twin_forward_euler); the generator's comment predicts this.

## Quirks the chain must keep byte-identical (v2 candidates)

These stay exactly as they are in the refactored chain, documented in CAPS at
the top of the method that carries them. A later phase adds a `<method>_v2`
that fixes each one, with its own test. Phase 1 only records the list.

1. `torch.random.fork_rng(devices=device)` without `with` does nothing, so the
   caller sees the global RNG state that generation leaves behind (2242).
2. stdlib `random` is unseeded on the Sintel path, so null edges are not
   reproducible there (2539-2591).
3. `NeuronState.stimulus` aliases `net.stimulus.buffer` on CPU (2677). F9 pins
   this. On GPU and MPS `.to(device)` copies, so there is no alias. Per the PI
   decision, GPU-only semantics are acceptable for v2, CPU is unsupported for
   that path, and the aliasing itself is not solved.
4. erase checks `x_list_*` / `y_list_*` without `.zarr` (2237), so
   `y_list_*.zarr`, `noisy_y_list_*.zarr`, `weights_full.pt`,
   `noisy_test_data.ok`, `BRACKET_*.txt` and old `x_list_*` extra files
   survive. `_tile_train_zarrs` then tiles a stale `noisy_y_list_train.zarr`
   (cells dirty_erase, dirty_noerase_tile).
5. `noisy_y_list` = y + delta(eta)/dt, optionally smoothed, and never includes
   xi/dt (3925-3966).
6. Exponential Euler (`pde.step`) is applied to the current-based model too
   whenever `conductance_exponential_euler` is True (the default), so at
   sigma = 0, (v[t+1]-v[t])/dt differs from `y_list[t]` (3783-3786). Mutant
   store_finite_difference is caught in every fast cell.

Further candidates found in this phase (not in the PI list; for the PI to decide):

7. A rendering-cache miss consumes torch RNG after `manual_seed` (see above).
8. `truncate_max_frames` is a no-op on the DAVIS path.
9. `CombinedVideoDataset` lacks `arg_df`, so multiple `datavis_roots` cannot
   generate.
10. From reading the code, not tested: in mixed mode `davis_iter =
    iter(davis_dataset)` walks the whole DAVIS dataset, test videos included,
    while generating the train split.

## Phase 1 measurements (same machine, CPU, warm fixtures unless stated)

| run | result | wall |
|---|---|---|
| fast tier, base + head (20 runs, 5 jobs), from an empty work dir | 10/10 cells identical | 2 min 15 s (48 s fixture build + 86 s runs) |
| full tier, base + head (148 runs, 6 jobs) | 74/74 cells identical | 7 min 10 s |
| branch coverage of the base function, full tier (74 cells) | lines 783/815 = 96.07 %, arcs 255/282 = 90.43 %; all 32 lines and 27 arcs left are listed in JUSTIFIED_UNCOVERED.yaml (7 lines and 4 arcs env-gated flyvis_conductance, the rest dead code) | 5 min 8 s |
| same, fast tier only | lines 73.62 %, arcs 56.74 % | 1 min |
| mutant self-test (11 mutants, fast tier + 3 Sintel runs) | 7 caught, 4 equivalent, all as predicted | 9 min 33 s |

The coverage numbers are line and arc coverage as coverage.py measures them.
Conditional expressions (`a if c else b`) are not arcs, so B07, B44 and B47
are not measured. B07 and B47 are exercised in both directions
(all_columns_extent15 / long_vis_5100 against the rest); B44's CUDA arm is not,
since there is no CUDA device locally.

## Phase 2 (the refactor into generators/voltage/), measured 2026-09-26

* **RNG ledger check mode.** Every cell except F1_base runs with
  `CGNN_RNG_LEDGER_CHECK=1` (driver.py), so each stage's `draws=False` claim
  (generators/voltage/rng.py) is enforced in every golden run; F1_base runs
  the ledger as production does. `golden_voltage.py ledger <label>`
  summarises which stage advanced which stream.
* **Quirks 7-10 above, re-verified on the HEAD chain** (scratch runs in the
  golden environment):
  * 7: F3 with a cold rendering cache (one RenderedDavis created) differs from
    the warm run in voltage, noise, y_list, noisy_y_list, generation_log.txt and
    kinograph.png; the cold run's ledger shows `load_stimuli` advancing
    torch_cpu (warm: python only).
  * 8: truncate_max_frames 20 and 80 give the same 48 DAVIS sequences of 64
    frames from the same cached rendering.
  * 9: cell combined_roots raises AttributeError on `arg_df` (full tier).
  * 10: in F7_mixed the train split reads DAVIS items 0, 1 and 2 through the
    mixed iterator, all three from the two TEST videos
    (sequence_01_vid_00, sequence_04_vid_04). Train data contains test frames.
* All ten quirks are listed in `generators/voltage/__init__.py` and
  documented in CAPS at the top of the chain method that keeps them.
* **Head coverage** (`coverage --tier full --impl head`, 74 cells): lines
  1467/1467 and branches 234/234 of the voltage package plus the
  data_generate_voltage wrapper, outside 17 marked places under 9 IDs of
  JUSTIFIED_UNCOVERED_HEAD.yaml (the same dead / env-gated code as at base,
  plus the CUDA/MPS RNG snapshot and the ledger's raise); no stale marker.
  StageTracker misuse guards are covered by focused unit tests because the
  generic tracker lives outside the voltage package measured here.
* **Mutant self-test against HEAD** (the eleven mutants re-planted in the new
  modules): all as expected. init_calcium_after_materialize, equivalent in
  phase 1 (the data cannot show it), is now CAUGHT: it moves a torch draw into
  materialize_sequences, which is declared draws=False, and the ledger raises.
