# Refactor voltage-data generation into a mutable fluent pipeline

## Summary

This PR replaces the legacy voltage generator's 1,251-line entry point with a
68-line orchestration function. The new entry point exposes the workflow as 31
named fluent stage methods followed by `finish`: 32 ordered, reviewable calls
grouped into preprocess, dataset production, postprocess, evaluation, and
reporting. The implementation is split across 15 focused voltage modules.

`data_generate_voltage` remains the public entry point and returns `None` as
before. `VoltageGeneration` is mutable: each stage performs its work and returns
`self`. A reusable `StageTracker` enforces declared prerequisites, repetition,
failure policy, and completion requirements. It deliberately does not impose a
total order where stages are independent.

The legacy implementation is removed rather than retained beside the new
package. `graph_data_generator.py` now contains only the 68-line voltage wrapper;
the former `_run_ode_generation`, `_tile_train_zarrs`, and
`_compute_noisy_derivatives` helpers are gone. The file remains large because it
also owns separate task, spiking, task-model-rollout, and connectome-constrained
generators. Its remaining imports from `generators/voltage` are the wrapper's
pipeline dependency and intentional compatibility re-exports.

Outside localized import cleanup, unchanged functions in
`graph_data_generator.py` retain their upstream formatting, keeping the review
focused on the extracted voltage implementation.

Against current `origin/main`, the final branch changes 51 files
(+7,562/-1,799): 19 generator/tracker files (+3,720/-1,791), 24 test files
(+3,307/-4), one 221-line harness CLI, and seven CI/editor/documentation files
(+314/-4).

## Stage tracking

The reusable mechanism is independent of voltage generation:

```python
from connectome_gnn.stages import StageRule, StageTracker

tracker = StageTracker({
    "load": StageRule(),
    "build": StageRule(after={"load"}),
    "epoch": StageRule(after={"build"}, repeatable=True),
    "preview": StageRule(after={"build"}, required_for_finish=False),
})

with tracker.stage("load"):
    data = [1, 2, 3]
with tracker.stage("build"):
    model = {"n_items": len(data)}
for _ in range(2):
    with tracker.stage("epoch"):
        model["n_items"] += 1

tracker.finish()
```

The voltage pipeline adds its own RNG snapshot and ledger handling around this
generic reusable stage-tracker. If in future a second pipeline class (candidate:
training) reuses the same patterns, an abstract `FluentMutable` base class can
combine these chores.

## New entry point

The legacy voltage generator's 1,251-line entry point is replaced with this
68-line orchestration function, that provides clear modularity and order.

```python
def data_generate_voltage(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    device=None,
    save=True,
    compute_ranks=True,
):
    generator = VoltageGeneration.from_config(
        config,
        visualize=visualize,
        run_vizualized=run_vizualized,
        style=style,
        erase=erase,
        step=step,
        device=device,
        save=save,
        compute_ranks=compute_ranks,
    )

    # PREPROCESS
    generator = (
        generator.prepare_output()
        .seed()
        .log_banner()
        .make_folders()
        .build_network()
        .load_stimuli()
        .extract_ode_params()
        .add_null_edges()
        .ablate()
        .build_ode()
        .init_geometry()
        .steady_state()
        .init_state()
        .split_videos()
        .materialize_sequences()
        .plot_previews()
    )

    # PRODUCE DATASETS
    generator = (
        generator.integrate("train")
        .derive_noisy_targets("train")
        .tile_train()
        .reset_for_test()
        .integrate("test")
        .derive_noisy_targets("test")
    )

    # POSTPROCESS
    generator = generator.restore_grad().remove_edges().save_ground_truth()

    # EVALUATE
    generator = generator.load_train_split().check_bracket().compute_ranks().log_trace_window().measurement_snr()

    # REPORT
    generator.write_generation_log().render_figures().render_video().finish()
```

The stage methods remain in this same semantic order in `pipeline.py`, while
their declared rules capture only true dependencies (defined in
`VOLTAGE_STAGE_RULES` in `pipeline.py`). This keeps the
byte-sensitive legacy call order visible without confusing it with a claim that
every pair of stages is intrinsically ordered.

## Known quirks preserved for byte-identical output

These are documented, not fixed, in this PR. The current implementation links
show where each behavior is preserved; the legacy links show the corresponding
code in the pinned pre-refactor implementation.

1. **The RNG fork is never entered**, so generation changes the caller's RNG
   state: [current `seed`](https://github.com/allierc/connectome-gnn/blob/refactor/voltage-generator/src/connectome_gnn/generators/voltage/pipeline.py#L213-L228),
   [legacy lines 2242-2244](https://github.com/allierc/connectome-gnn/blob/496de14b15e64e25303a5a45a75b5e26fa61db71/src/connectome_gnn/generators/graph_data_generator.py#L2242-L2244).
2. **The global `random` stream is not seeded for null-edge sampling**:
   [current `add_null_edges`](https://github.com/allierc/connectome-gnn/blob/refactor/voltage-generator/src/connectome_gnn/generators/voltage/pipeline.py#L319-L332),
   [legacy lines 2537-2587](https://github.com/allierc/connectome-gnn/blob/496de14b15e64e25303a5a45a75b5e26fa61db71/src/connectome_gnn/generators/graph_data_generator.py#L2537-L2587).
3. **On CPU, `x.stimulus` aliases the network stimulus buffer**:
   [current `init_state`](https://github.com/allierc/connectome-gnn/blob/refactor/voltage-generator/src/connectome_gnn/generators/voltage/pipeline.py#L369-L385),
   [legacy lines 2673-2682](https://github.com/allierc/connectome-gnn/blob/496de14b15e64e25303a5a45a75b5e26fa61db71/src/connectome_gnn/generators/graph_data_generator.py#L2673-L2682).
4. **Erase misses some prior outputs, and a stale noisy target can be tiled**:
   [current `prepare_output`](https://github.com/allierc/connectome-gnn/blob/refactor/voltage-generator/src/connectome_gnn/generators/voltage/pipeline.py#L199-L211),
   [current `tile_train`](https://github.com/allierc/connectome-gnn/blob/refactor/voltage-generator/src/connectome_gnn/generators/voltage/pipeline.py#L471-L483),
   [legacy erase lines 2233-2240](https://github.com/allierc/connectome-gnn/blob/496de14b15e64e25303a5a45a75b5e26fa61db71/src/connectome_gnn/generators/graph_data_generator.py#L2233-L2240),
   [legacy tiling lines 3878-3922](https://github.com/allierc/connectome-gnn/blob/496de14b15e64e25303a5a45a75b5e26fa61db71/src/connectome_gnn/generators/graph_data_generator.py#L3878-L3922).
5. **Noisy derivative targets omit the process-noise increment**:
   [current `derive_noisy_targets`](https://github.com/allierc/connectome-gnn/blob/refactor/voltage-generator/src/connectome_gnn/generators/voltage/pipeline.py#L454-L469),
   [legacy lines 3925-3942](https://github.com/allierc/connectome-gnn/blob/496de14b15e64e25303a5a45a75b5e26fa61db71/src/connectome_gnn/generators/graph_data_generator.py#L3925-L3942).
6. **Exponential stepping is also used for current-based models**, so finite
   differences need not equal the stored drift:
   [current `integrate`](https://github.com/allierc/connectome-gnn/blob/refactor/voltage-generator/src/connectome_gnn/generators/voltage/pipeline.py#L412-L452),
   [legacy lines 3773-3786](https://github.com/allierc/connectome-gnn/blob/496de14b15e64e25303a5a45a75b5e26fa61db71/src/connectome_gnn/generators/graph_data_generator.py#L3773-L3786).
7. **A cold rendering cache can shift the later torch RNG stream**:
   [current `load_stimuli`](https://github.com/allierc/connectome-gnn/blob/refactor/voltage-generator/src/connectome_gnn/generators/voltage/pipeline.py#L290-L309),
   [legacy lines 2430-2475](https://github.com/allierc/connectome-gnn/blob/496de14b15e64e25303a5a45a75b5e26fa61db71/src/connectome_gnn/generators/graph_data_generator.py#L2430-L2475).
8. **The DAVIS path does not apply `truncate_max_frames`**:
   [current `load_stimuli`](https://github.com/allierc/connectome-gnn/blob/refactor/voltage-generator/src/connectome_gnn/generators/voltage/pipeline.py#L290-L309),
   [legacy lines 2445-2474](https://github.com/allierc/connectome-gnn/blob/496de14b15e64e25303a5a45a75b5e26fa61db71/src/connectome_gnn/generators/graph_data_generator.py#L2445-L2474).
9. **Multiple DAVIS roots fail because `CombinedVideoDataset` lacks `arg_df`**:
   [current `load_stimuli`](https://github.com/allierc/connectome-gnn/blob/refactor/voltage-generator/src/connectome_gnn/generators/voltage/pipeline.py#L290-L309),
   [current `split_videos`](https://github.com/allierc/connectome-gnn/blob/refactor/voltage-generator/src/connectome_gnn/generators/voltage/pipeline.py#L388-L398),
   [legacy construction lines 2476-2480](https://github.com/allierc/connectome-gnn/blob/496de14b15e64e25303a5a45a75b5e26fa61db71/src/connectome_gnn/generators/graph_data_generator.py#L2476-L2480),
   [legacy split lines 2685-2688](https://github.com/allierc/connectome-gnn/blob/496de14b15e64e25303a5a45a75b5e26fa61db71/src/connectome_gnn/generators/graph_data_generator.py#L2685-L2688).
10. **Mixed mode can use test DAVIS videos while generating the train split**:
    [current `integrate`](https://github.com/allierc/connectome-gnn/blob/refactor/voltage-generator/src/connectome_gnn/generators/voltage/pipeline.py#L412-L452),
    [legacy lines 3512-3525](https://github.com/allierc/connectome-gnn/blob/496de14b15e64e25303a5a45a75b5e26fa61db71/src/connectome_gnn/generators/graph_data_generator.py#L3512-L3525).

## Validation

- Full pre-mutable refactor golden suite: 74/74 byte-identical comparisons.
- Current mutable pipeline fast golden suite: 10/10 byte-identical comparisons.
- Full current HEAD coverage across 74 cells: 1,467/1,467 lines and 234/234
  branches outside 17 justified markers; no unexplained gaps, stale markers, or
  unused IDs.
- Focused tracker, pipeline, and golden-harness regression tests: 13/13 pass.
- Standard CI suites: tier1 67/67 and full non-golden pytest 420/420 pass.
- Mutation harness source checks: all 11 mutations still target exactly one
  intended location.
- The branch is rebased onto current `origin/main`; after the rebase, Ruff, all
    13 focused tests, and the 10-case byte-identity suite pass.

## Contributor notes

- VS Code recommends the Ruff extension and formats modified Python lines on
    save. The historical tree is not globally Ruff-formatted, so avoid unrelated
    whole-file formatting churn.
- CI enforces Ruff on every Python file added or modified by a push or pull
    request. Existing `origin/main` lint debt is outside that incremental gate.
- Treat `tests/golden_voltage/BASE_SHA` as a fixed behavioral reference. Do not
    advance it merely to make a mismatch pass; first determine and document why
    behavior changed.
- Run the ten-cell fast golden tier while editing this pipeline and the full
    74-cell tier before merge. A path-filtered `golden-voltage` pull-request
    workflow now runs the fast tier automatically.
- The ten documented quirks below are intentional compatibility behavior in
    this PR. Their fixes remain separate follow-up work.

## Review focus

- Whether the stage boundaries and declared prerequisites match the real data
  dependencies.
- Mutable state carried between train and test generation.
- RNG snapshot and ledger behavior around every stage.
- Whether the preserved quirks should remain follow-up work rather than expand
  this behavior-preserving PR.