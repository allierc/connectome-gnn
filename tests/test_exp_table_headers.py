"""No column header in the experiment report names a message family by hand.

THE DEFECT THIS EXISTS TO CATCH. `tools/exp.py` says in its own docstring that
nothing in the report is typed -- every value comes from the run. The two
`fit roll r` columns broke that rule: they were labelled "cond. form" and
"curr. form" in `COLUMNS`, which is correct for a conductance model and exactly
backwards for a current one, because the tester writes `template_rollout_r` for
whichever family the MODEL belongs to and `template_alt_rollout_r` for the
other. Every exp01 table therefore read the two numbers the wrong way round --
1.00 against the conductance form and 0.62 against the current one -- and the
table looked like a result ("the conductance readout fits a current model
better") rather than like a swapped label.

The run already knew: `alt_form_family` is in every `results/metrics.txt`.

So the rule these tests enforce is narrow and checkable: a header that depends
on the run must be DERIVED from the run, and `_form_heads` is where that
derivation lives. A future column with the same property fails here the moment
someone types a family name into `COLUMNS` or `_PRETTY` instead.
"""
import importlib.util
import os

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "exp_tool", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "tools", "exp.py"))
exp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(exp)


# The words that name a message family. A header carrying one of these is
# asserting something about the model, and only the run may assert it.
_FAMILY_WORDS = ("cond", "curr", "conductance", "current")


def _runs_saying(*families):
    """Fake `runs(fm)` rows whose metrics report the given `alt_form_family`."""
    return [({"id": "a"}, {}, f"run{i}") for i in range(len(families))]


@pytest.fixture
def patched(monkeypatch):
    def use(*families):
        by_run = {f"run{i}": {"alt_form_family": f} if f else {}
                  for i, f in enumerate(families)}
        monkeypatch.setattr(exp, "metrics_of", lambda r: by_run.get(r) or None)
        return _runs_saying(*families)
    return use


def test_columns_do_not_name_a_family():
    """The static header set stays family-neutral."""
    for _key, head, _live in exp.COLUMNS:
        for word in _FAMILY_WORDS:
            assert word not in head.lower(), (
                f"header {head!r} names a message family; it must be derived "
                f"from the run's alt_form_family, not typed in COLUMNS")


def test_pretty_does_not_name_a_family():
    for head, tex in exp._PRETTY.items():
        for word in _FAMILY_WORDS:
            assert word not in tex.lower(), (
                f"_PRETTY[{head!r}] names a message family")


def test_a_current_model_reads_own_as_current(patched):
    """alt_form_family == conductance means the model's own form is current."""
    rs = patched(*["conductance"] * 5)
    own, alt = exp._form_heads(rs)
    assert "curr" in own and "cond" in alt


def test_a_conductance_model_reads_own_as_conductance(patched):
    rs = patched(*["current"] * 5)
    own, alt = exp._form_heads(rs)
    assert "cond" in own and "curr" in alt


def test_mixed_families_refuse_to_name_either(patched):
    """One table cannot label a column for two different model families."""
    assert exp._form_heads(patched("conductance", "current")) is None


def test_nothing_landed_refuses_too(patched):
    assert exp._form_heads(patched(None, None)) is None
