"""Tests for connectome_gnn.models.training_utils — config-driven logic."""
import pytest

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.training_utils import determine_load_fields

pytestmark = pytest.mark.tier2


class TestDetermineLoadFields:
    def test_base_fields(self, minimal_config):
        fields = determine_load_fields(minimal_config)
        assert "voltage" in fields
        assert "stimulus" in fields
        assert "neuron_type" in fields

    def test_visual_field_adds_pos(self, minimal_config_dict):
        minimal_config_dict["graph_model"]["field_type"] = "visual_NNR"
        config = NeuralGraphConfig(**minimal_config_dict)
        fields = determine_load_fields(config)
        assert "pos" in fields

    def test_calcium_adds_calcium(self, minimal_config_dict):
        d = minimal_config_dict.copy()
        d["simulation"] = {**d["simulation"], "calcium_type": "leaky"}
        config = NeuralGraphConfig(**d)
        fields = determine_load_fields(config)
        assert "calcium" in fields

    def test_noise_adds_noise(self, minimal_config_dict):
        d = minimal_config_dict.copy()
        d["simulation"] = {**d["simulation"], "measurement_noise_level": 0.05}
        config = NeuralGraphConfig(**d)
        fields = determine_load_fields(config)
        assert "noise" in fields

    def test_no_extra_fields_by_default(self, minimal_config):
        fields = determine_load_fields(minimal_config)
        assert "calcium" not in fields
        assert "noise" not in fields


class TestMetricsLogHeader:
    """metrics.log's raw tau / V_rest columns must be named as raw.

    Columns 2 and 3 are computed over EVERY neuron, so a handful with a
    near-zero fitted slope drive them to -30 or worse. They are not what
    metrics.png plots and not the paper convention -- the comparable values are
    vrest_r2_clean / tau_r2_clean at columns 6 and 9, with their outlier counts.
    Reading the obvious names made a healthy run look broken.

    Positions are asserted too: plot.py reads this file positionally
    (_f(parts, 2), _f(parts, 3), _f(parts, 6), _f(parts, 9)), so a rename must
    not reorder anything.
    """

    def _header(self, tmp_path):
        import os
        from connectome_gnn.models.training_utils import init_metrics_files
        os.makedirs(tmp_path / "tmp_training", exist_ok=True)
        init_metrics_files(str(tmp_path))
        with open(tmp_path / "tmp_training" / "metrics.log") as f:
            return f.readline().strip().split(",")

    def test_raw_columns_are_named_raw(self, tmp_path):
        cols = self._header(tmp_path)
        assert cols[2] == "vrest_r2_raw"
        assert cols[3] == "tau_r2_raw"

    def test_clean_columns_keep_their_positions(self, tmp_path):
        cols = self._header(tmp_path)
        assert cols[6] == "vrest_r2_clean"
        assert cols[9] == "tau_r2_clean"

    def test_full_layout_is_unchanged(self, tmp_path):
        assert self._header(tmp_path) == [
            "iteration", "connectivity_r2", "vrest_r2_raw", "tau_r2_raw",
            "hidden_nnr_pearson", "anchor_nnr_pearson",
            "vrest_r2_clean", "n_out_vrest", "n_total_vrest",
            "tau_r2_clean", "n_out_tau", "n_total_tau",
        ]
