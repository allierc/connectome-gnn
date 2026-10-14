# Voltage generator golden harness

This harness protects the behavior of `data_generate_voltage` while its implementation is refactored. It runs the pinned pre-refactor implementation from the commit in `BASE_SHA` and the current implementation in separate processes, then compares everything each run leaves behind.

It is stricter than an ordinary unit test. The manifest includes:

- every generated file and directory;
- byte-level Zarr, PyTorch, text, and PNG output;
- decoded MP4 frames and stream parameters;
- normalized stdout and stderr;
- global RNG state, PyTorch gradient mode, Matplotlib settings, logger levels, and expected exceptions.

The harness exists because this generator has observable behavior outside its return value: global RNG use, filesystem layout, logging, plots, videos, and partially written output on failure. A refactor can preserve numerical arrays while silently changing one of those contracts.

## Quick start

The harness needs Python 3.12, the test dependencies, the pinned `flyvis` revision, and `ffmpeg`. Its default work directory is `~/.cache/cgnn-golden-voltage`; override it with `CGNN_GOLDEN_WORK`.

```bash
# Build deterministic fixtures. This is cached after the first run.
python scripts/golden_voltage.py fixtures --tier fast

# Compare the pinned base and current implementation.
python scripts/golden_voltage.py run --tier fast --impl both

# Equivalent pytest entry point for the fast tier.
pytest -m golden tests/golden_voltage
```

Use the fast tier while changing the generator. Run the full tier before merge:

```bash
python scripts/golden_voltage.py run --tier full --impl both
CGNN_GOLDEN_TIER=full pytest -m golden tests/golden_voltage
```

The full tier contains 74 scenarios. The fast tier contains the ten cheapest representative scenarios.

## Additional checks

```bash
# Measure branch coverage of the current voltage package.
python scripts/golden_voltage.py coverage --tier full --impl head

# Verify that known behavioral mutations are caught as expected.
python scripts/golden_voltage.py mutants

# Re-run the determinism experiments used to define comparison policy.
python scripts/golden_voltage.py determinism

# Summarize which global RNG streams each stage advanced in a saved run.
python scripts/golden_voltage.py ledger <run-label>
```

See `DETERMINISM.md` for the controlled environment, comparator policy, measured limitations, and justified uncovered branches.

## Structure

- `BASE_SHA` pins the pre-refactor implementation. The remote tag
	`golden-voltage-base-20260925` makes that commit available to CI.
- `cells.py` declares the fast and full scenario matrix.
- `fixtures.py` builds deterministic DAVIS, FlyVis, conductance, and dirty-output fixtures.
- `cell_runner.py` executes one implementation and records its outputs and process state.
- `manifest.py` defines the comparison policy.
- `driver.py` prepares and runs cells in isolated subprocesses.
- `coverage_report.py` verifies that uncovered current code is explicitly justified.
- `mutants.py` plants known defects to test the harness itself.
- `test_golden_voltage.py` exposes the comparison through pytest.

## Maintenance rules

Do not update `BASE_SHA` merely to make a changed result pass. A mismatch means either the refactor changed behavior or the intended contract changed. In the latter case, document and review the behavior change first, then update the reference and determinism evidence deliberately.

Add or update cells when a changed code path is not exercised by the existing matrix. Keep fixtures immutable within a base/head pair: both implementations must observe the same files, paths, and cache state.

The GitHub workflow runs the fast tier automatically when a pull request changes
the generator, tracker, harness, or directly coupled files. Manual
`workflow_dispatch` remains available for either tier. Run the full tier locally
before merging substantial generator changes.
