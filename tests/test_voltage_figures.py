from connectome_gnn.generators.voltage.figures import INDEX_TO_NAME as FIGURE_INDEX_TO_NAME
from connectome_gnn.metrics import INDEX_TO_NAME


def test_voltage_figures_use_canonical_neuron_type_names():
    assert FIGURE_INDEX_TO_NAME is INDEX_TO_NAME
