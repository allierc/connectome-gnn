"""`flyvis_general_form` is the general-form GNN (g_phi = MLP(v_j, a_j, v_i, a_i));
`flyvis_conductance` is its old name, kept so the archived run configs under log/fly load.
Every place that dispatches on the model name must treat the two identically."""
import pytest

from connectome_gnn.models.utils import GENERAL_FORM_GNN_NAMES, is_general_form_gnn


@pytest.mark.parametrize("name", ["flyvis_general_form", "flyvis_conductance"])
def test_both_names_are_the_general_form(name):
    assert is_general_form_gnn(name)


@pytest.mark.parametrize("name", ["flyvis_current", "flyvis_A", "drosophila_cx",
                                  "flyvis_conductance_known_ode", "flyvis_cond_known_ode", ""])
def test_other_names_are_not(name):
    assert not is_general_form_gnn(name)


def test_registries_agree():
    from connectome_gnn.generators.ode_params import get_ode_params_class
    from connectome_gnn.models.registry import _REGISTRY, _discover_models
    from connectome_gnn.template_rollout import _KNOWN_ODE_FOR
    _discover_models()
    new, old = GENERAL_FORM_GNN_NAMES
    assert _REGISTRY[new] is _REGISTRY[old]
    assert get_ode_params_class(new) is get_ode_params_class(old)
    assert _KNOWN_ODE_FOR[new] == _KNOWN_ODE_FOR[old]
