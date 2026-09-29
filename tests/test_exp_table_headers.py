"""The two fit-roll columns carry each number under the family it belongs to.

HISTORY. The columns were once typed "cond. form" / "curr. form" while the
tester wrote them by the MODEL's family, so every current-model table read the
two numbers backwards (exp01: 1.00 under conductance, 0.62 under current). They
were then renamed "own form" / "other form", which is true per row but means
current on a current model's row and conductance on a conductance model's --
ambiguous in any table that mixes the two (exp02). And underneath, the tester
itself crossed the constants whenever the model's family was not the data's: a
conductance GNN on flyvis rolled the CURRENT fit out in the CONDUCTANCE
generator (fixed in template_rollout._takes_alt_arrays, 2026-09-29).

THE RULE NOW. The headers name the template family, the same on every row, and
`_fit_roll_by_family` decides per run which of the tester's two rollouts goes
under which: by the generator each ran in (`<slot>_model`) and the fit it
carried (`<slot>_fit_family`, or for a run analysed before the fix the slot rule
rebuilt from `alt_form_family`). A rollout whose generator and fit disagree is
left out rather than shown under either name.
"""
import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "exp_tool", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "tools", "exp.py"))
exp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(exp)

CUR, COND = "flyvis_known_ode", "flyvis_conductance_known_ode"


def _m(own_gen, alt_gen, alt_form_family, fit_families=None):
    m = {"alt_form_family": alt_form_family,
         "template_rollout_model": own_gen, "template_rollout_r": "0.11",
         "template_rollout_pct_clamped": "70.0",
         "template_alt_rollout_model": alt_gen, "template_alt_rollout_r": "0.22",
         "template_alt_rollout_pct_clamped": "0.0"}
    if fit_families:
        m["template_rollout_fit_family"], m["template_alt_rollout_fit_family"] = fit_families
    return exp._fit_roll_by_family(m)


def test_a_current_model_on_current_data_reads_its_own_rollout_as_current():
    """The exp01 defect: the model's own rollout is the CURRENT form here."""
    m = _m(CUR, COND, "conductance")
    assert m["fitroll_current_r"] == "0.11"
    assert m["fitroll_conductance_r"] == "0.22"
    assert m["fitroll_current_pct_clamped"] == "70.0"


def test_a_crossed_rollout_written_before_the_fix_is_left_out():
    """A conductance model on current data, analysed before the pairing fix:
    each generator carried the other family's constants, so neither r is a
    number about the form its column would name."""
    m = _m(COND, CUR, "conductance")
    assert "fitroll_current_r" not in m and "fitroll_conductance_r" not in m


def test_the_same_model_after_the_fix_fills_both():
    m = _m(COND, CUR, "conductance", fit_families=("conductance", "current"))
    assert m["fitroll_conductance_r"] == "0.11"
    assert m["fitroll_current_r"] == "0.22"


def test_a_conductance_model_on_conductance_data_was_never_crossed():
    m = _m(COND, CUR, "current")
    assert m["fitroll_conductance_r"] == "0.11"
    assert m["fitroll_current_r"] == "0.22"


def test_a_run_without_a_template_rollout_gets_neither():
    m = exp._fit_roll_by_family({"Wij_R2": "0.9"})
    assert "fitroll_current_r" not in m and "fitroll_conductance_r" not in m


def test_the_columns_are_the_family_columns():
    keys = [k for k, _h, _l in exp.COLUMNS]
    assert "fitroll_current_r" in keys and "fitroll_conductance_r" in keys
    assert "template_rollout_r" not in keys and "template_alt_rollout_r" not in keys
