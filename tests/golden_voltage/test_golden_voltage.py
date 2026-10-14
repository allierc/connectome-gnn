"""Golden byte-identity test: HEAD data_generate_voltage vs BASE_SHA.

Deselected by default (pyproject addopts ``-m "not golden"``). Run with

    pytest -m golden tests/golden_voltage                    # fast tier
    CGNN_GOLDEN_TIER=full pytest -m golden tests/golden_voltage

Needs flyvis, the bundled flyvis model and ffmpeg; builds its fixtures under
CGNN_GOLDEN_WORK (default ~/.cache/cgnn-golden-voltage) on first use. All cells
run up front, in parallel (CGNN_GOLDEN_JOBS, default 4), in one session
fixture; each test then compares one cell.
"""

from __future__ import annotations

import os
import time

import pytest

from . import cells as C
from . import driver as D
from . import fixtures as FX
from . import manifest as M

pytestmark = pytest.mark.golden

TIER = os.environ.get("CGNN_GOLDEN_TIER", "fast")
SELECTED = [c for c in C.select(TIER) if C.requirements_met(c)[0]]


@pytest.fixture(scope="session")
def golden_runs():
    pytest.importorskip("flyvis")
    pytest.importorskip("tensorstore")
    work = FX.work_root()
    base = D.ensure_base_worktree(work)
    D.prepare(SELECTED, work, base)
    label = "pytest-" + time.strftime("%Y%m%d-%H%M%S")
    specs = []
    for c in SELECTED:
        for tag, impl in (("base", base), ("head", D.REPO_ROOT)):
            specs.append(D.RunSpec(impl_root=impl, cell=c.name, out=work / "runs" / label / c.name / tag))
    results = D.run_many(specs, work, jobs=int(os.environ.get("CGNN_GOLDEN_JOBS", "4")), progress=None)
    return {(r.spec.cell, "base" if r.spec.impl_root == base else "head"): r for r in results}


@pytest.mark.parametrize("name", [c.name for c in SELECTED])
def test_cell(golden_runs, name):
    cell = C.CELLS_BY_NAME[name]
    rb, rh = golden_runs[(name, "base")], golden_runs[(name, "head")]
    assert rb.manifest is not None, f"base runner failed: {rb.error}"
    assert rh.manifest is not None, f"head runner failed: {rh.error}"
    unexpected = D.check_expectation(cell, rb.manifest)
    assert unexpected is None, f"base does not reach the cell's intended path: {unexpected}"
    diffs = M.compare(rb.manifest, rh.manifest, run_a=rb.spec.out, run_b=rh.spec.out)
    assert not diffs, f"{name}: head differs from base\n" + "\n".join(diffs)
