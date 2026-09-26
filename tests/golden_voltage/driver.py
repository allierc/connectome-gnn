"""Launch golden cells in fresh subprocesses and collect their manifests.

The environment is built here, before the child interpreter starts, because
PYTHONHASHSEED and the BLAS thread caps cannot be changed from inside it.
Nothing from the caller's environment leaks in except PATH/HOME/USER/LANG/
TMPDIR: in particular DATAVIS_ROOT, HYBRID_CONNECTOME_DIR and FLYVIS_ROOT_DIR
are set per cell or not at all.

Every run gets its own FLYVIS_ROOT_DIR, cloned from the fixture template
(APFS clone on macOS), so flyvis's caches (connectome, renderings, the
NetworkView ``__cache__``) are never shared between runs and never touch the
user's flyvis data.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import cells as C
from . import fixtures as FX
from . import manifest as M

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent


def base_sha() -> str:
    return (HARNESS_DIR / "BASE_SHA").read_text().split()[0]


def default_python() -> str:
    return os.environ.get("CGNN_GOLDEN_PYTHON", sys.executable)


def ensure_base_worktree(work: Path) -> Path:
    """A detached worktree of BASE_SHA under the work dir (created once)."""
    wt = Path(os.environ.get("CGNN_GOLDEN_BASE_ROOT", str(Path(work) / "base_worktree")))
    sha = base_sha()
    if (wt / ".git").exists():
        head = subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        if not head.startswith(sha[:7]) and not sha.startswith(head[:7]):
            raise RuntimeError(f"{wt} is at {head}, BASE_SHA is {sha}")
        return wt
    wt.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(wt), sha], check=True)
    return wt


@dataclass
class RunSpec:
    impl_root: Path
    cell: str
    out: Path
    device: str = "cpu"
    threads: int = 1
    hashseed: str = "0"
    preseed: bool = True
    cold_cache: bool = False
    coverage_data: Path | None = None
    deterministic: bool = False
    extra_env: dict = field(default_factory=dict)
    timeout: int = 3600


@dataclass
class RunResult:
    spec: RunSpec
    returncode: int
    wall_s: float
    manifest: dict | None
    error: str = ""


def _child_env(spec: RunSpec, cell: C.Cell, work: Path, flyvis_root: Path) -> dict:
    env = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "LANG", "TMPDIR") if k in os.environ}
    t = str(spec.threads)
    env.update({
        "PYTHONHASHSEED": spec.hashseed,
        "OMP_NUM_THREADS": t, "MKL_NUM_THREADS": t, "OPENBLAS_NUM_THREADS": t,
        "VECLIB_MAXIMUM_THREADS": t, "NUMEXPR_NUM_THREADS": t,
        "MPLBACKEND": "Agg", "SOURCE_DATE_EPOCH": "0", "TZ": "UTC", "TIMEZONE": "UTC",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "FLYVIS_ROOT_DIR": str(flyvis_root),
    })
    pp = [str(spec.impl_root / "src"), str(spec.impl_root)]
    for k, v in cell.env.items():
        v = C.resolve_env_value(v, work)
        if k == "PYTHONPATH_EXTRA":
            pp.append(v)
        else:
            env[k] = v
    if "hybrid" in cell.requires:
        env["HYBRID_CONNECTOME_DIR"] = str(C.hybrid_dir())
    if spec.device.startswith("cuda"):
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if cell.ledger:
        # generators/voltage/rng.py: enforce every draws=False claim and write the
        # per-stage RNG record next to the run (outside data/, so not in the manifest).
        # BASE ignores both variables.
        env["CGNN_RNG_LEDGER_CHECK"] = "1"
        env["CGNN_RNG_LEDGER_REPORT"] = str(Path(spec.out).resolve() / "rng_ledger.json")
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env.update(spec.extra_env)
    return env


def _rebase_rendering_meta(template: Path, flyvis_root: Path) -> None:
    """Point cached renderings' configs at this run's flyvis root.

    ``RenderedSintel``'s config contains ``sintel_path = <FLYVIS_ROOT_DIR>/SintelDataSet``,
    so a cache rendered under the template root never matches a run whose root
    is elsewhere, and every Sintel run would re-render for a minute. The
    SintelDataSet in both roots is the same symlink, so the cached data is
    valid; only the path in ``_meta.yaml`` is rewritten. DAVIS configs hold the
    fixture path, which is the same for every run, and are left unchanged.
    """
    rdir = flyvis_root / "renderings"
    if not rdir.exists():
        return
    old, new = str(template), str(flyvis_root)
    for meta in rdir.glob("*/_meta.yaml"):
        text = meta.read_text()
        if old in text:
            meta.unlink()                      # replace the cloned file rather than edit it in place
            meta.write_text(text.replace(old, new))


def run_one(spec: RunSpec, work: Path, python: str | None = None) -> RunResult:
    python = python or default_python()
    cell = C.CELLS_BY_NAME[spec.cell]
    out = Path(spec.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    flyvis_root = out / "flyvis_root"
    FX.clone_tree(FX.fixtures_dir(work) / "flyvis_template", flyvis_root,
                  exclude=("renderings",) if spec.cold_cache else ())
    _rebase_rendering_meta(FX.fixtures_dir(work) / "flyvis_template", flyvis_root)
    renders_before = sorted(p.name for p in (flyvis_root / "renderings").glob("*")) \
        if (flyvis_root / "renderings").exists() else []
    cmd = [python]
    if spec.coverage_data is not None:
        spec.coverage_data.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["-m", "coverage", "run", "--branch", "--parallel-mode",
                f"--data-file={spec.coverage_data}",
                f"--include={spec.impl_root / 'src' / 'connectome_gnn' / 'generators' / '*'}"]
    cmd += [str(HARNESS_DIR / "cell_runner.py"), "--impl-root", str(spec.impl_root), "--cell", spec.cell,
            "--out", str(out), "--work", str(work), "--device", spec.device, "--threads", str(spec.threads)]
    if not spec.preseed:
        cmd.append("--no-preseed")
    if spec.deterministic:
        cmd.append("--deterministic")
    env = _child_env(spec, cell, work, flyvis_root)
    t0 = time.time()
    with open(out / "stdout.txt", "w") as so, open(out / "stderr.txt", "w") as se:
        try:
            proc = subprocess.run(cmd, env=env, cwd=str(out), stdout=so, stderr=se, timeout=spec.timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    wall = time.time() - t0
    renders_after = sorted(p.name for p in (flyvis_root / "renderings").glob("*")) \
        if (flyvis_root / "renderings").exists() else []
    (out / "run_info.json").write_text(json.dumps({
        "cmd": cmd, "returncode": rc, "wall_s": round(wall, 2),
        "renderings_created": sorted(set(renders_after) - set(renders_before)),
        "env": {k: v for k, v in env.items() if k not in ("PATH",)},
    }, indent=1))
    if rc != 0 or not (out / "state.json").exists():
        tail = (out / "stderr.txt").read_text()[-3000:]
        return RunResult(spec, rc, wall, None, error=f"runner failed rc={rc}:\n{tail}")
    man = M.write_manifest(out, impl_root=str(spec.impl_root), work=str(work))
    # flyvis_root is large-ish and irrelevant once the manifest exists
    shutil.rmtree(flyvis_root, ignore_errors=True)
    return RunResult(spec, rc, wall, man)


def run_many(specs: list[RunSpec], work: Path, jobs: int = 4, python: str | None = None,
             progress=print) -> list[RunResult]:
    results = [None] * len(specs)
    with cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futs = {pool.submit(run_one, s, work, python): i for i, s in enumerate(specs)}
        for fut in cf.as_completed(futs):
            i = futs[fut]
            r = fut.result()
            results[i] = r
            if progress:
                status = "ok" if r.manifest else "RUNNER-FAIL"
                exc = (r.manifest or {}).get("state", {}) or {}
                exc = exc.get("exception")
                progress(f"  {r.spec.cell:32s} {Path(r.spec.impl_root).name:14s} {r.wall_s:7.1f}s {status}"
                         + (f"  raised {exc['type']}" if exc else ""))
    return results


def check_expectation(cell: C.Cell, manifest: dict) -> str | None:
    """None if the run ended the way the cell says it should, else a message."""
    import re
    exc = (manifest.get("state") or {}).get("exception")
    if cell.expect == "ok":
        return None if exc is None else f"expected success, got {exc['type']}: {exc['message'][:300]}"
    etype, regex = cell.expect
    if exc is None:
        return f"expected {etype}, got success"
    if etype != "*" and exc["type"] != etype:
        return f"expected {etype}, got {exc['type']}: {exc['message'][:300]}"
    if not re.search(regex, exc["message"]):
        return f"exception message does not match {regex!r}: {exc['message'][:300]}"
    return None


def prepare(cells: list[C.Cell], work: Path, base_root: Path, python: str | None = None) -> None:
    """Build every fixture the selected cells need (idempotent)."""
    python = python or default_python()
    FX.build_videos(work)            # before rendering_specs, which skips roots that do not exist
    specs, sintel, extents = [], set(), {8}
    for c in cells:
        for s in c.rendering_specs(work):
            if s not in specs:
                specs.append(s)
        if c.needs_sintel_rendering() and C.requirements_met(c)[0]:
            sintel.add(c.extent())
        if C.requirements_met(c)[0]:
            extents.add(c.extent())
    FX.ensure_fixtures(work, base_root, python, davis_specs=specs, sintel_extents=sorted(sintel),
                       extents=sorted(extents))
