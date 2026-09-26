"""Branch coverage of the BASE data_generate_voltage over the golden cells.

Runs every selected cell once with the base implementation under
``coverage run --branch`` (one data file per process), combines them, and
reports line and branch coverage restricted to the legacy function bodies in
the base file:

    data_generate_voltage          graph_data_generator.py:2214-3465
    _run_ode_generation            3467-3875
    _tile_train_zarrs              3878-3922
    _compute_noisy_derivatives     3925-3966

Line numbers are those of BASE_SHA and never move. Every uncovered line or
arc must be listed in JUSTIFIED_UNCOVERED.yaml with a reason; anything else
fails the report. Output: ``<work>/coverage/<label>/report.json`` and a text
summary on stdout.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from . import cells as C
from . import driver as D
from . import fixtures as FX

HARNESS_DIR = Path(__file__).resolve().parent
TARGET = "src/connectome_gnn/generators/graph_data_generator.py"
RANGES = ((2214, 3465), (3467, 3966))


def in_range(line: int) -> bool:
    return any(a <= line <= b for a, b in RANGES)


def load_justified() -> dict:
    import yaml
    doc = yaml.safe_load((HARNESS_DIR / "JUSTIFIED_UNCOVERED.yaml").read_text()) or {}
    lines, arcs = {}, {}
    for e in doc.get("lines", []) or []:
        for ln in _expand(e["lines"]):
            lines[ln] = e
    for e in doc.get("arcs", []) or []:
        a, b = (int(x) for x in str(e["arc"]).split("->"))
        arcs[(a, b)] = e
    return {"lines": lines, "arcs": arcs}


def _expand(spec) -> list[int]:
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def analyse(cov_json: Path, base_root: Path) -> dict:
    data = json.loads(cov_json.read_text())
    target = str((base_root / TARGET).resolve())
    fdata = None
    for path, fd in data["files"].items():
        if str(Path(path).resolve()) == target:
            fdata = fd
    if fdata is None:
        raise RuntimeError(f"{target} not in coverage data; files: {list(data['files'])[:5]}")
    executed = sorted(ln for ln in fdata["executed_lines"] if in_range(ln))
    missing = sorted(ln for ln in fdata["missing_lines"] if in_range(ln))
    ex_br = sorted(tuple(a) for a in fdata.get("executed_branches", []) if in_range(a[0]))
    mi_br = sorted(tuple(a) for a in fdata.get("missing_branches", []) if in_range(a[0]))
    return {"executed_lines": executed, "missing_lines": missing,
            "executed_branches": ex_br, "missing_branches": mi_br}


def report(an: dict, justified: dict, base_root: Path) -> tuple[dict, list[str]]:
    src = (base_root / TARGET).read_text().splitlines()
    n_lines = len(an["executed_lines"]) + len(an["missing_lines"])
    n_br = len(an["executed_branches"]) + len(an["missing_branches"])
    unjust_lines = [ln for ln in an["missing_lines"] if ln not in justified["lines"]]
    unjust_arcs = [a for a in an["missing_branches"] if tuple(a) not in justified["arcs"]]
    stale = sorted(set(justified["lines"]) - set(an["missing_lines"]))
    stale_arcs = sorted(set(justified["arcs"]) - {tuple(a) for a in an["missing_branches"]})
    summary = {
        "lines_total": n_lines, "lines_covered": len(an["executed_lines"]),
        "branches_total": n_br, "branches_covered": len(an["executed_branches"]),
        "line_pct": round(100 * len(an["executed_lines"]) / max(n_lines, 1), 2),
        "branch_pct": round(100 * len(an["executed_branches"]) / max(n_br, 1), 2),
        "missing_lines": an["missing_lines"], "missing_branches": [list(a) for a in an["missing_branches"]],
        "unjustified_lines": unjust_lines, "unjustified_arcs": [list(a) for a in unjust_arcs],
        "justified_but_covered_lines": stale, "justified_but_covered_arcs": [list(a) for a in stale_arcs],
    }
    text = [
        f"lines    {summary['lines_covered']}/{n_lines} = {summary['line_pct']}%",
        f"branches {summary['branches_covered']}/{n_br} = {summary['branch_pct']}%",
        f"uncovered lines: {len(an['missing_lines'])} ({len(unjust_lines)} unjustified)",
        f"uncovered arcs:  {len(an['missing_branches'])} ({len(unjust_arcs)} unjustified)",
    ]
    for ln in an["missing_lines"]:
        tag = "justified" if ln in justified["lines"] else "UNJUSTIFIED"
        text.append(f"  line {ln:5d} [{tag}] {src[ln - 1].strip()[:90]}")
    for a, b in an["missing_branches"]:
        tag = "justified" if (a, b) in justified["arcs"] else "UNJUSTIFIED"
        dst = src[b - 1].strip()[:50] if b > 0 else "<exit>"
        text.append(f"  arc {a:5d}->{b:<5d} [{tag}] {src[a - 1].strip()[:60]}  =>  {dst}")
    for ln in stale:
        text.append(f"  STALE justification: line {ln} is covered now")
    for a in stale_arcs:
        text.append(f"  STALE justification: arc {a[0]}->{a[1]} is covered now")
    return summary, text


def main(args) -> int:
    work = FX.work_root()
    base = D.ensure_base_worktree(work)
    label = args.label or "latest"
    out = work / "coverage" / label
    data_file = out / ".coverage"
    py = D.default_python()
    if not getattr(args, "reuse", False):
        cells = [c for c in C.select(args.tier, args.cells or None) if C.requirements_met(c)[0]]
        D.prepare(cells, work, base)
        if out.exists():
            import shutil
            shutil.rmtree(out)
        out.mkdir(parents=True)
        specs = [D.RunSpec(impl_root=base, cell=c.name, out=out / "runs" / c.name, coverage_data=data_file)
                 for c in cells]
        t0 = time.time()
        results = D.run_many(specs, work, jobs=args.jobs)
        print(f"ran {len(specs)} cells under coverage in {time.time() - t0:.0f}s")
        bad = [r for r in results if r.manifest is None]
        for r in bad:
            print(f"  RUNNER FAILED {r.spec.cell}: {r.error[-500:]}")
        subprocess.run([py, "-m", "coverage", "combine", f"--data-file={data_file}"], cwd=out, check=True,
                       capture_output=True)
    cov_json = out / "coverage.json"
    subprocess.run([py, "-m", "coverage", "json", f"--data-file={data_file}", "-o", str(cov_json),
                    f"--include={base / TARGET}"], cwd=out, check=True, capture_output=True)
    an = analyse(cov_json, base)
    summary, text = report(an, load_justified(), base)
    (out / "report.json").write_text(json.dumps(summary, indent=1))
    (out / "report.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    print(f"wrote {out / 'report.json'}")
    ok = not summary["unjustified_lines"] and not summary["unjustified_arcs"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(None))
