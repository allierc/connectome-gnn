"""Branch coverage of data_generate_voltage over the golden cells, for BASE or for HEAD.

Runs every selected cell once under ``coverage run --branch`` (one data file
per process), combines them, and reports line and branch coverage.

--impl base (the default; phase 1): the legacy function bodies in the base file

    data_generate_voltage          graph_data_generator.py:2214-3465
    _run_ode_generation            3467-3875
    _tile_train_zarrs              3878-3922
    _compute_noisy_derivatives     3925-3966

Line numbers are those of BASE_SHA and never move. Every uncovered line or
arc must be listed in JUSTIFIED_UNCOVERED.yaml with a reason.

--impl head (phase 2): every module of the new package
``src/connectome_gnn/generators/voltage/`` plus the ``data_generate_voltage``
wrapper in graph_data_generator.py. Head line numbers move with every edit,
so justifications live IN THE CODE, as a comment on the line they excuse:

    if cond:  # golden-uncovered: <ID>   the whole block is excluded (coverage's
                                          exclude rule: a block-opening line takes
                                          its block with it)
    if cond:  # golden-partial: <ID>     one arc of this branch is never taken

and every <ID> needs an entry (reason, reach) in JUSTIFIED_UNCOVERED_HEAD.yaml.
The report fails on an uncovered line or arc without a marker, on a marker
whose ID is not in the YAML, on a YAML ID no marker uses, and on a STALE
marker: an excluded block with an executed line in it, or a partial marker
whose branch took both arcs (measured on a second report without the markers).

Output: ``<work>/coverage/<label>/report.json`` and a text summary on stdout.
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


# ---------------------------------------------------------------------------
# HEAD: the voltage package, with in-code justification markers
# ---------------------------------------------------------------------------

HEAD_PACKAGE = "src/connectome_gnn/generators/voltage"
HEAD_WRAPPER = ("src/connectome_gnn/generators/graph_data_generator.py", "data_generate_voltage")
_MARK_RE = __import__("re").compile(r"# golden-(uncovered|partial): ([A-Za-z0-9_-]+)")


def head_targets(head_root: Path) -> dict:
    """{resolved path: None (whole file) or (first, last) line range} of the head report."""
    import ast
    out = {str(p.resolve()): None for p in sorted((head_root / HEAD_PACKAGE).glob("*.py"))}
    wp = head_root / HEAD_WRAPPER[0]
    for node in ast.parse(wp.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == HEAD_WRAPPER[1]:
            out[str(wp.resolve())] = (node.lineno, node.end_lineno)
    return out


def _json_report(py, data_file: Path, out_json: Path, include: list, rcfile: Path | None, cwd: Path) -> dict:
    cmd = [py, "-m", "coverage", "json", f"--data-file={data_file}", "-o", str(out_json),
           f"--include={','.join(include)}"]
    if rcfile is not None:
        cmd.append(f"--rcfile={rcfile}")
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)
    return json.loads(out_json.read_text())


def load_justified_head() -> dict:
    import yaml
    doc = yaml.safe_load((HARNESS_DIR / "JUSTIFIED_UNCOVERED_HEAD.yaml").read_text()) or {}
    return {str(k): v for k, v in (doc.get("ids") or {}).items()}


def _block_bodies(source: str) -> dict:
    """{line of a block-opening statement: (first, last) line of its body} (if/elif/for/while/with/try/def)."""
    import ast
    out = {}
    for node in ast.walk(ast.parse(source)):
        body = getattr(node, "body", None)
        if isinstance(node, ast.stmt) and isinstance(body, list) and body:
            out[node.lineno] = (body[0].lineno, body[-1].end_lineno)
    return out


def report_head(marked: dict, raw: dict, targets: dict, justified: dict) -> tuple[dict, list[str]]:
    def in_target(path, line):
        rng = targets.get(path)
        return rng is None or rng[0] <= line <= rng[1]

    def files(data):
        return {str(Path(p).resolve()): fd for p, fd in data["files"].items() if str(Path(p).resolve()) in targets}

    fm, fr = files(marked), files(raw)
    n_lines = n_cov = n_br = n_br_cov = 0
    unjust, markers, stale, text_rows = [], [], [], []
    for path in sorted(targets):
        src = Path(path).read_text().splitlines()
        fdm, fdr = fm.get(path), fr.get(path)
        rel = str(Path(path)).split("/src/")[-1]
        if fdm is None:
            unjust.append(f"{rel}: not imported by any cell")
            continue
        ex_l = [ln for ln in fdm["executed_lines"] if in_target(path, ln)]
        mi_l = [ln for ln in fdm["missing_lines"] if in_target(path, ln)]
        ex_b = [tuple(a) for a in fdm.get("executed_branches", []) if in_target(path, a[0])]
        mi_b = [tuple(a) for a in fdm.get("missing_branches", []) if in_target(path, a[0])]
        n_lines += len(ex_l) + len(mi_l)
        n_cov += len(ex_l)
        n_br += len(ex_b) + len(mi_b)
        n_br_cov += len(ex_b)
        for ln in mi_l:
            unjust.append(f"{rel}:{ln} line not covered: {src[ln - 1].strip()[:80]}")
        for a, b in mi_b:
            unjust.append(f"{rel}:{a}->{b} arc not covered: {src[a - 1].strip()[:80]}")
        # markers and staleness (the raw report has no exclusions)
        excluded = sorted(ln for ln in fdm.get("excluded_lines", []) if in_target(path, ln))
        raw_exec = set(fdr["executed_lines"]) if fdr else set()
        raw_missing_from = {a for a, _ in (fdr.get("missing_branches", []) if fdr else [])}
        bodies = _block_bodies("\n".join(src))
        for i, line in enumerate(src, start=1):
            m = _MARK_RE.search(line)
            if not m or not in_target(path, i):
                continue
            kind, mid = m.groups()
            markers.append({"file": rel, "line": i, "kind": kind, "id": mid})
            text_rows.append(f"  {kind:9s} {mid:22s} {rel}:{i}  {line.strip()[:70]}")
            if mid not in justified:
                unjust.append(f"{rel}:{i} marker id {mid!r} has no entry in JUSTIFIED_UNCOVERED_HEAD.yaml")
            if kind == "partial" and i not in raw_missing_from:
                stale.append(f"{rel}:{i} golden-partial {mid}: both arcs are taken now")
            if kind == "uncovered":
                body = bodies.get(i)
                if body is None:
                    unjust.append(f"{rel}:{i} golden-uncovered must mark a line that opens a block")
                    continue
                ran = sorted(ln for ln in raw_exec if body[0] <= ln <= body[1])
                if ran:
                    stale.append(f"{rel}:{i} golden-uncovered {mid}: lines {ran} of its block ran")
                if i not in excluded:
                    unjust.append(f"{rel}:{i} golden-uncovered marker was not excluded by coverage")
    used = {m["id"] for m in markers}
    unused = sorted(set(justified) - used)
    summary = {
        "lines_total": n_lines, "lines_covered": n_cov, "branches_total": n_br, "branches_covered": n_br_cov,
        "line_pct": round(100 * n_cov / max(n_lines, 1), 2), "branch_pct": round(100 * n_br_cov / max(n_br, 1), 2),
        "markers": markers, "unjustified": unjust, "stale": stale, "unused_ids": unused,
    }
    text = [
        f"HEAD voltage package + data_generate_voltage wrapper ({len(targets)} files)",
        f"lines    {n_cov}/{n_lines} = {summary['line_pct']}%   (marked blocks excluded)",
        f"branches {n_br_cov}/{n_br} = {summary['branch_pct']}%   (marked partial branches not counted)",
        f"markers: {len(markers)} ({len(used)} ids)",
        *text_rows,
        f"unjustified: {len(unjust)}", *[f"  {u}" for u in unjust],
        f"stale markers: {len(stale)}", *[f"  {u}" for u in stale],
        f"unused ids in JUSTIFIED_UNCOVERED_HEAD.yaml: {unused}",
    ]
    return summary, text


def main_head(args, work: Path) -> int:
    base = D.ensure_base_worktree(work)
    head = Path(args.head_root).resolve() if getattr(args, "head_root", "") else D.REPO_ROOT
    label = args.label or "latest-head"
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
        specs = [D.RunSpec(impl_root=head, cell=c.name, out=out / "runs" / c.name, coverage_data=data_file)
                 for c in cells]
        t0 = time.time()
        results = D.run_many(specs, work, jobs=args.jobs)
        print(f"ran {len(specs)} cells under coverage in {time.time() - t0:.0f}s")
        for r in results:
            if r.manifest is None:
                print(f"  RUNNER FAILED {r.spec.cell}: {r.error[-500:]}")
        subprocess.run([py, "-m", "coverage", "combine", f"--data-file={data_file}"], cwd=out, check=True,
                       capture_output=True)
    targets = head_targets(head)
    include = sorted({str(Path(t)) for t in targets})
    rc = out / "markers.coveragerc"
    # "[#]" rather than "#": configparser drops continuation lines that start with "#"
    rc.write_text("[report]\nexclude_also =\n    [#] golden-uncovered:\n"
                  "partial_branches =\n    [#] golden-partial:\n    pragma: no branch\n")
    marked = _json_report(py, data_file, out / "coverage_marked.json", include, rc, out)
    empty_rc = out / "empty.coveragerc"
    empty_rc.write_text("[report]\n")
    raw = _json_report(py, data_file, out / "coverage_raw.json", include, empty_rc, out)
    summary, text = report_head(marked, raw, targets, load_justified_head())
    (out / "report.json").write_text(json.dumps(summary, indent=1))
    (out / "report.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    print(f"wrote {out / 'report.json'}")
    ok = not summary["unjustified"] and not summary["stale"] and not summary["unused_ids"]
    return 0 if ok else 1


def main(args) -> int:
    work = FX.work_root()
    if getattr(args, "impl", "base") == "head":
        return main_head(args, work)
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
