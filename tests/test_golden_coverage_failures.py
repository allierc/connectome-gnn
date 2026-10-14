from types import SimpleNamespace

import pytest

from tests.golden_voltage import coverage_report


@pytest.mark.parametrize("impl", ["base", "head"])
def test_coverage_fails_before_report_when_runner_fails(monkeypatch, tmp_path, impl):
    cell = SimpleNamespace(name="broken")
    result = SimpleNamespace(
        manifest=None,
        spec=SimpleNamespace(cell=cell.name),
        error="runner failed",
    )
    args = SimpleNamespace(
        impl=impl,
        head_root="",
        label="failure",
        reuse=False,
        tier="fast",
        cells=None,
        jobs=1,
    )

    monkeypatch.setattr(coverage_report.FX, "work_root", lambda: tmp_path)
    monkeypatch.setattr(coverage_report.D, "ensure_base_worktree", lambda work: tmp_path / "base")
    monkeypatch.setattr(coverage_report.D, "prepare", lambda cells, work, base: None)
    monkeypatch.setattr(coverage_report.D, "run_many", lambda specs, work, jobs: [result])
    monkeypatch.setattr(coverage_report.C, "select", lambda tier, cells: [cell])
    monkeypatch.setattr(coverage_report.C, "requirements_met", lambda cell: (True, ""))
    monkeypatch.setattr(
        coverage_report.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("coverage report ran after a failed cell"),
    )

    assert coverage_report.main(args) == 1
