"""Golden byte-identity harness for data_generate_voltage (CLI).

    # build fixtures, run the fast tier with base and head, compare
    python scripts/golden_voltage.py run --tier fast --impl both

    # only some cells, more workers
    python scripts/golden_voltage.py run --cells F1_base F7_mixed --jobs 6

    # determinism phase (old x old experiments), writes determinism.json
    python scripts/golden_voltage.py determinism

    # branch coverage of the BASE implementation over the cells
    python scripts/golden_voltage.py coverage --tier full

    # harness self-test: planted mutants must be caught by the fast tier
    python scripts/golden_voltage.py mutants

    # which stage advanced which RNG stream, over the HEAD runs of a label
    python scripts/golden_voltage.py ledger <label>

Environment:
  CGNN_GOLDEN_WORK       work dir (fixtures, base worktree, runs); default ~/.cache/cgnn-golden-voltage
  CGNN_GOLDEN_BASE_ROOT  an existing worktree of BASE_SHA to use instead of <work>/base_worktree
  CGNN_GOLDEN_PYTHON     interpreter for the cell subprocesses (default: this one)

The base implementation always runs from a separate worktree of
tests/golden_voltage/BASE_SHA; the head implementation is this checkout.
See tests/golden_voltage/DETERMINISM.md for what is compared and why.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from golden_voltage import cells as C  # noqa: E402
from golden_voltage import driver as D  # noqa: E402
from golden_voltage import fixtures as FX  # noqa: E402
from golden_voltage import manifest as M  # noqa: E402


def _select(args) -> list:
    cells = C.select(args.tier, args.cells or None)
    keep = []
    for c in cells:
        ok, why = C.requirements_met(c)
        if ok:
            keep.append(c)
        else:
            print(f"  skip {c.name}: {why}")
    return keep


def _head_root(args, work: Path, label: str) -> Path:
    """The HEAD implementation: this checkout, or (--snapshot) an APFS clone of its
    src/, GNN_PlotFigure.py and assets/ taken now, so the checkout can be edited
    while the cells run."""
    if not args.snapshot:
        return REPO_ROOT
    from golden_voltage import mutants
    return mutants.make_scratch(REPO_ROOT, work / "snapshots" / label)


def cmd_run(args) -> int:
    work = FX.work_root()
    base = D.ensure_base_worktree(work)
    cells = _select(args)
    t0 = time.time()
    D.prepare(cells, work, base)
    print(f"fixtures ready ({time.time() - t0:.0f}s)")
    label = args.label or time.strftime("%Y%m%d-%H%M%S")
    head = _head_root(args, work, label)
    impls = {"base": [base], "head": [head], "both": [base, head]}[args.impl]
    specs = []
    for c in cells:
        for impl in impls:
            tag = "base" if impl == base else "head"
            specs.append(D.RunSpec(impl_root=impl, cell=c.name, device=args.device, threads=args.threads,
                                   out=work / "runs" / label / c.name / tag))
    t1 = time.time()
    results = D.run_many(specs, work, jobs=args.jobs)
    print(f"ran {len(specs)} cell-runs in {time.time() - t1:.0f}s")
    failures = 0
    by_cell = {}
    for r in results:
        by_cell.setdefault(r.spec.cell, {})["base" if r.spec.impl_root == base else "head"] = r
    for c in cells:
        rs = by_cell[c.name]
        msgs = []
        for tag, r in rs.items():
            if r.manifest is None:
                msgs.append(f"{tag}: {r.error}")
                continue
            e = D.check_expectation(c, r.manifest)
            if e:
                msgs.append(f"{tag}: {e}")
        if "base" in rs and "head" in rs and rs["base"].manifest and rs["head"].manifest:
            msgs += M.compare(rs["base"].manifest, rs["head"].manifest,
                              run_a=rs["base"].spec.out, run_b=rs["head"].spec.out)
        status = "PASS" if not msgs else "FAIL"
        failures += bool(msgs)
        print(f"{status} {c.name}")
        for m in msgs:
            print("    " + m.replace("\n", "\n    "))
    print(f"{len(cells) - failures}/{len(cells)} cells pass; runs under {work / 'runs' / label}")
    return 1 if failures else 0


def cmd_ledger(args) -> int:
    """Per stage, which global RNG streams the HEAD runs of a label advanced (rng_ledger.json)."""
    import json
    root = FX.work_root() / "runs" / args.label
    reports = sorted(root.glob("*/head/rng_ledger.json"))
    if not reports:
        print(f"no rng_ledger.json under {root}/*/head")
        return 1
    order, stages = [], {}
    for rp in reports:
        cell = rp.parent.parent.name
        for e in json.loads(rp.read_text())["stages"]:
            st = stages.setdefault(e["stage"], {"declared": set(), "advanced": {}, "cells": 0, "raised": []})
            if e["stage"] not in order:
                order.append(e["stage"])
            st["declared"].add(e["declared_draws"])
            st["cells"] += 1
            if e["raised"]:
                st["raised"].append(cell)
            for s in e["advanced"]:
                st["advanced"].setdefault(s, []).append(cell)
    print(f"{len(reports)} reports under {root}")
    for name in order:
        st = stages[name]
        adv = ", ".join(f"{s} in {len(c)}/{st['cells']}" for s, c in sorted(st["advanced"].items())) or "none"
        decl = "/".join("draws" if d else "no-draw" for d in sorted(st["declared"]))
        print(f"  {name:28s} declared {decl:14s} advanced: {adv}" + (f"  raised in {st['raised']}" if st["raised"] else ""))
        if args.verbose:
            for s, c in sorted(st["advanced"].items()):
                print(f"      {s}: {sorted(c)}")
    return 0


def cmd_fixtures(args) -> int:
    work = FX.work_root()
    base = D.ensure_base_worktree(work)
    D.prepare(_select(args), work, base)
    print(f"fixtures ready under {FX.fixtures_dir(work)}")
    return 0


def cmd_determinism(args) -> int:
    from golden_voltage import determinism
    return determinism.main(args)


def cmd_coverage(args) -> int:
    from golden_voltage import coverage_report
    return coverage_report.main(args)


def cmd_mutants(args) -> int:
    from golden_voltage import mutants
    return mutants.main(args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--tier", default="fast", choices=["fast", "full"])
        p.add_argument("--cells", nargs="*")
        p.add_argument("--jobs", type=int, default=4)
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--label", default="")

    p = sub.add_parser("run")
    common(p)
    p.add_argument("--impl", default="both", choices=["base", "head", "both"])
    p.add_argument("--snapshot", action="store_true",
                   help="run HEAD from a clone of the checkout taken at start (edit freely meanwhile)")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("ledger", help="summarise the HEAD runs' RNG ledgers of a label")
    p.add_argument("label")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_ledger)
    p = sub.add_parser("fixtures")
    common(p)
    p.set_defaults(fn=cmd_fixtures)
    p = sub.add_parser("determinism")
    common(p)
    p.add_argument("--experiments", nargs="*", default=None)
    p.set_defaults(fn=cmd_determinism)
    p = sub.add_parser("coverage")
    common(p)
    p.add_argument("--reuse", action="store_true", help="only re-report existing coverage data")
    p.set_defaults(fn=cmd_coverage)
    p = sub.add_parser("mutants")
    common(p)
    p.add_argument("--mutants", nargs="*", default=None)
    p.set_defaults(fn=cmd_mutants)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
