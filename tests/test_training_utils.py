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


class TestMetricsFiles:
    """init_metrics_files no longer writes a positional metrics.log.

    The recovered-parameter trajectories live in tmp_training/<key>.log, one file
    per quantity with the metrics.txt column names (metrics.recovery_log_append);
    only the hidden-INR Pearson log, which is not a recovered parameter, is
    created here.
    """

    def test_only_nnr_pearson_log_is_created(self, tmp_path):
        import os
        from connectome_gnn.models.training_utils import init_metrics_files
        path = init_metrics_files(str(tmp_path))
        assert path == os.path.join(str(tmp_path), "tmp_training", "nnr_pearson.log")
        assert sorted(os.listdir(tmp_path / "tmp_training")) == ["nnr_pearson.log"]
        with open(path) as f:
            assert f.readline().strip().split(",") == [
                "iteration", "hidden_pearson_mean", "hidden_pearson_std",
                "anchor_pearson_mean", "anchor_pearson_std"]
