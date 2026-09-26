"""Determinism phase: is byte identity achievable, and under which conditions?

Every experiment runs the BASE implementation twice under two conditions and
compares the manifests with the raw-bytes comparator for every kind (the
decoded digests are reported alongside, to see what a decoded comparator
would tolerate). Results go to ``<work>/determinism/<label>/determinism.json``;
DETERMINISM.md records the conclusions and the comparator policy they imply.

Experiments (``--experiments`` selects a subset):

  old_x_old      two fresh processes, different output roots, identical conditions
  threads        torch/BLAS threads 1 vs 4
  hashseed       PYTHONHASHSEED 0 vs 1
  cache          warm (pre-rendered) vs cold (empty) rendering cache
  unseeded       stdlib random NOT pre-seeded, two processes (expected: Sintel
                 null-edge cell differs, DAVIS null-edge cell does not)
  mps            device=mps, two processes (MPS old x old); plus MPS vs CPU
  mps_deterministic  as mps, with torch.use_deterministic_algorithms(True)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import cells as C
from . import driver as D
from . import fixtures as FX
from . import manifest as M

BYTES_ONLY = {k: "bytes" for k in M.POLICY}
DECODED = dict(BYTES_ONLY, png="decoded", mp4="decoded")

# old x old over the whole matrix is cheap enough locally; the other
# experiments use the fast tier plus the cells whose behaviour they probe.
EXPERIMENTS = ("old_x_old", "threads", "hashseed", "cache", "unseeded", "mps", "mps_deterministic")


def _cells(names):
    return [C.CELLS_BY_NAME[n] for n in names if C.requirements_met(C.CELLS_BY_NAME[n])[0]]


def _pairs(exp: str, full: bool, only=None) -> list[tuple[str, dict, dict]]:
    """(cell, kwargs of run A, kwargs of run B) per comparison; ``only`` restricts the cells."""
    fast = only or [c.name for c in C.CELLS if c.tier == "fast"]
    if exp == "old_x_old":
        names = only or ([c.name for c in C.CELLS] if full else fast)
        return [(n, {}, {}) for n in names]
    if exp == "threads":
        return [(n, {}, {"threads": 4}) for n in fast]
    if exp == "hashseed":
        return [(n, {}, {"hashseed": "1"}) for n in fast]
    if exp == "cache":
        return [(n, {}, {"cold_cache": True}) for n in fast + ["sintel_only_noise", "mixed_davis_stopiter"]]
    if exp == "unseeded":
        return [(n, {"preseed": False}, {"preseed": False})
                for n in ("sintel_null_edges", "F6_edges_random", "null_random_attempts")]
    if exp == "mps":
        return [(n, {"device": "mps"}, {"device": "mps"}) for n in fast] + \
               [(n, {}, {"device": "mps"}) for n in fast]
    if exp == "mps_deterministic":
        kw = {"device": "mps", "deterministic": True}
        return [(n, kw, kw) for n in ("F1_base", "F3_process_noise", "F6_edges_random")]
    raise KeyError(exp)


def run_experiment(exp: str, work: Path, base: Path, label: str, jobs: int, full: bool, only=None) -> dict:
    pairs = [(n, a, b) for n, a, b in _pairs(exp, full, only) if C.requirements_met(C.CELLS_BY_NAME[n])[0]]
    root = work / "determinism" / label / exp
    specs = []
    for i, (n, a, b) in enumerate(pairs):
        for side, kw in (("A", a), ("B", b)):
            specs.append(D.RunSpec(impl_root=base, cell=n, out=root / f"{i:02d}_{n}" / side, **kw))
    t0 = time.time()
    results = D.run_many(specs, work, jobs=jobs)
    report = {"experiment": exp, "wall_s": round(time.time() - t0, 1), "pairs": []}
    for i, (n, a, b) in enumerate(pairs):
        ra, rb = results[2 * i], results[2 * i + 1]
        entry = {"cell": n, "A": a, "B": b, "wall_A": round(ra.wall_s, 1), "wall_B": round(rb.wall_s, 1)}
        if ra.manifest is None or rb.manifest is None:
            entry["error"] = (ra.error or rb.error)[-2000:]
        else:
            entry["diffs_bytes"] = M.compare(ra.manifest, rb.manifest, policy=BYTES_ONLY,
                                             run_a=ra.spec.out, run_b=rb.spec.out)
            entry["diffs_decoded"] = M.compare(ra.manifest, rb.manifest, policy=DECODED,
                                               run_a=ra.spec.out, run_b=rb.spec.out, explain=False)
            kinds = {}
            for f, e in ra.manifest["files"].items():
                k = kinds.setdefault(e["kind"], {"n": 0, "bytes_equal": 0})
                k["n"] += 1
                eb = rb.manifest["files"].get(f)
                k["bytes_equal"] += bool(eb and eb["sha256"] == e["sha256"])
            entry["kinds"] = kinds
            entry["identical"] = not entry["diffs_bytes"]
        report["pairs"].append(entry)
    return report


def main(args) -> int:
    work = FX.work_root()
    base = D.ensure_base_worktree(work)
    exps = args.experiments or list(EXPERIMENTS)
    full = args.tier == "full"
    D.prepare(C.select("full" if full else "fast") + _cells(["sintel_only_noise", "sintel_null_edges",
                                                              "mixed_davis_stopiter", "null_random_attempts"])
              + _cells(args.cells or []), work, base)
    label = args.label or time.strftime("%Y%m%d-%H%M%S")
    out = work / "determinism" / label
    out.mkdir(parents=True, exist_ok=True)
    path = out / "determinism.json"
    reports = json.loads(path.read_text()) if path.exists() else {}
    for exp in exps:
        print(f"== {exp}")
        key = exp if not args.cells else f"{exp}[{','.join(args.cells)}]"
        reports[key] = run_experiment(exp, work, base, label, args.jobs, full, only=args.cells or None)
        exp = key
        path.write_text(json.dumps(reports, indent=1))
        for p in reports[exp]["pairs"]:
            tag = "IDENTICAL" if p.get("identical") else ("ERROR" if "error" in p else "DIFFER")
            print(f"  {tag:9s} {p['cell']:32s} A={p['A']} B={p['B']}")
            for d in (p.get("diffs_bytes") or [])[:6]:
                print("      " + d.replace("\n", "\n      "))
    print(f"wrote {path}")
    return 0
