from types import SimpleNamespace

import pytest

from tests.golden_voltage.mutants import _require_manifests


def test_mutant_harness_rejects_runner_failure():
    result = SimpleNamespace(
        manifest=None,
        spec=SimpleNamespace(cell="broken"),
        error="syntax error",
    )

    with pytest.raises(RuntimeError, match="mutant example runner failures") as exc_info:
        _require_manifests([result], context="mutant example")
    assert "broken: syntax error" in str(exc_info.value)


def test_mutant_harness_accepts_valid_manifest_with_application_exception():
    result = SimpleNamespace(
        manifest={"state": {"exception": "ValueError"}},
        spec=SimpleNamespace(cell="expected-exception"),
        error="",
    )

    _require_manifests([result], context="mutant example")
