"""Phase S runner — executes the agent's staged tests and enforces the PASS gate.

Convention:
  staging/block_NN/<name>.py          — the staged mechanism function
  staging/block_NN/test_<name>.py     — a script (not a pytest fixture) that
                                         prints `PASS: <msg>` on success or
                                         `FAIL: <msg>` on failure and exits 0.

We run each test_*.py as a subprocess (fresh Python, no pytest dependency),
capture stdout+stderr, and look for a `PASS:` line. The test file is free to
use pytest internally; we only inspect its printed output.

Also exposes `load_full_voltage(...)` — a cached loader the staged tests can
use to get (v_clean, v_noisy) for the measurement-noise condition without
paying the zarr load cost per-test. Caches in /tmp; persists across tests
within one block.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import subprocess
import time
from typing import List, Optional

PASS_PREFIX = "PASS:"
FAIL_PREFIX = "FAIL:"


@dataclasses.dataclass
class TestResult:
    path: str
    passed: bool
    duration_sec: float
    pass_line: Optional[str]
    fail_line: Optional[str]
    stdout: str
    returncode: int

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        line = self.pass_line or self.fail_line or "(no PASS/FAIL marker)"
        return f"[{status}] {os.path.basename(self.path)}  {self.duration_sec:.1f}s — {line}"


@dataclasses.dataclass
class ScratchpadReport:
    block_number: int
    results: List[TestResult]
    any_passed: bool
    total_duration_sec: float

    def as_markdown(self) -> str:
        lines = [
            f"# Phase S — staging test report, block {self.block_number:02d}",
            "",
            f"Total tests: {len(self.results)} — any PASS: {self.any_passed} — "
            f"duration {self.total_duration_sec:.1f}s",
            "",
        ]
        for r in self.results:
            lines.append(f"- {r.summary()}")
            if not r.passed:
                tail = "\n    ".join(r.stdout.strip().splitlines()[-20:])
                lines.append(f"    ```\n    {tail}\n    ```")
        return "\n".join(lines) + "\n"


def _extract_marker(stdout: str, prefix: str) -> Optional[str]:
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith(prefix):
            return s
    return None


def run_tests(
    staging_block_dir: str,
    repo_root: str,
    block_number: int,
    per_test_timeout_sec: int = 600,
    python_exe: str = "python",
) -> ScratchpadReport:
    """Run every test_*.py in staging_block_dir (non-recursively).

    Each test is a subprocess with cwd=repo_root so the staged mechanism module
    can be imported via its fully-qualified package path.
    """
    os.makedirs(staging_block_dir, exist_ok=True)
    tests = sorted(glob.glob(os.path.join(staging_block_dir, "test_*.py")))

    results: List[TestResult] = []
    t0 = time.time()
    for path in tests:
        tt0 = time.time()
        try:
            proc = subprocess.run(
                [python_exe, path],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=per_test_timeout_sec,
            )
            stdout = (proc.stdout or "") + (proc.stderr or "")
            rc = proc.returncode
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or "") + (e.stderr or "")
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            stdout += f"\n[TIMEOUT after {per_test_timeout_sec}s]"
            rc = -1

        pass_line = _extract_marker(stdout, PASS_PREFIX)
        fail_line = _extract_marker(stdout, FAIL_PREFIX)
        passed = (pass_line is not None) and (rc == 0)
        results.append(TestResult(
            path=path,
            passed=passed,
            duration_sec=time.time() - tt0,
            pass_line=pass_line,
            fail_line=fail_line,
            stdout=stdout,
            returncode=rc,
        ))

    return ScratchpadReport(
        block_number=block_number,
        results=results,
        any_passed=any(r.passed for r in results),
        total_duration_sec=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Data cache for staged tests
# ---------------------------------------------------------------------------

_FULL_VOLTAGE_CACHE: dict = {}


def load_full_voltage(
    dataset: str = "fly/flyvis_noise_free",
    measurement_noise_level: float = 0.10,
    graphs_root: Optional[str] = None,
):
    """Load (v_clean, v_noisy) as torch tensors of shape (T, N) for Phase-S tests.

    Cached per (dataset, noise level) tuple for the lifetime of the Python
    process. Tests invoked by `run_tests()` each spawn a fresh interpreter, so
    the cache persists *within* one test process, not across them — that's
    fine: the zarr load is the slow part, and zarr's own mmap means the second
    read within the same process is instant.

    Returns torch tensors so staged tests can slice GPU-ready data directly.
    """
    key = (dataset, float(measurement_noise_level))
    if key in _FULL_VOLTAGE_CACHE:
        return _FULL_VOLTAGE_CACHE[key]

    import numpy as np
    import torch
    import zarr

    if graphs_root is None:
        # Prefer the connectome-gnn repo's own graphs_data, fall back to
        # flyvis-gnn's, which always has the zarr files for noise_free etc.
        candidates = [
            "/workspace/connectome-gnn/graphs_data",
        ]
        for c in candidates:
            if os.path.isdir(os.path.join(c, dataset, "x_list_train")):
                graphs_root = c
                break
        if graphs_root is None:
            raise FileNotFoundError(
                f"no graphs_data root contains '{dataset}/x_list_train'; "
                f"tried {candidates}"
            )

    v_path = os.path.join(graphs_root, dataset, "x_list_train", "voltage.zarr")
    v_arr = zarr.open_array(v_path, mode="r")
    v_clean_np = np.array(v_arr[:])  # (T, N)
    v_clean = torch.from_numpy(v_clean_np).float()

    # Synthetic measurement noise to match training pipeline:
    # y_noisy = y_clean + γ · ε where ε ~ N(0, 1) (per-sample).
    gen = torch.Generator().manual_seed(0)
    v_noisy = v_clean + measurement_noise_level * torch.randn(
        v_clean.shape, generator=gen
    )

    _FULL_VOLTAGE_CACHE[key] = (v_clean, v_noisy)
    return v_clean, v_noisy
