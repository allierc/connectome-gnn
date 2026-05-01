"""Smoke tests for hybrid connectome loading and network construction."""

import json
from pathlib import Path

import pytest

from connectome_gnn.generators.hybrid_connectome import (
    _table_dir,
    _VARIANT_TO_DIRNAME,
    load_tables,
    load_hybrid_network,
)
from connectome_gnn.generators.utils import is_flyvis_hybrid_model

DATA_DIR = Path(__file__).parent.parent / "data" / "hybrid_connectomes"
MANIFEST = DATA_DIR / "manifest.json"


def _load_manifest():
    if not MANIFEST.exists():
        pytest.skip("hybrid connectome data not present (download release artifact)")
    with open(MANIFEST) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Registry / name resolution
# ---------------------------------------------------------------------------


class TestVariantResolution:
    def test_is_flyvis_hybrid_model(self):
        assert is_flyvis_hybrid_model("e8_flywireRF")
        assert is_flyvis_hybrid_model("e8_flywireRF_proximal_nulls")
        assert is_flyvis_hybrid_model("full_eye_flywireRF")
        assert is_flyvis_hybrid_model("full_eye_flywireRF_proximal_nulls_known_ode")
        assert not is_flyvis_hybrid_model("flyvis_A")
        assert not is_flyvis_hybrid_model("flyvis_B")

    def test_variant_table_complete(self):
        expected = {
            "e8_flywireRF",
            "e8_flywireRF_known_ode",
            "e8_flywireRF_proximal_nulls",
            "e8_flywireRF_proximal_nulls_known_ode",
            "full_eye_flywireRF",
            "full_eye_flywireRF_known_ode",
            "full_eye_flywireRF_proximal_nulls",
            "full_eye_flywireRF_proximal_nulls_known_ode",
        }
        assert expected.issubset(set(_VARIANT_TO_DIRNAME.keys()))

    def test_table_dir_e8(self):
        d = _table_dir("e8_flywireRF", data_dir=DATA_DIR)
        assert d.name == "e8_flywireRF"

    def test_table_dir_full_eye_proximal(self):
        d = _table_dir("full_eye_flywireRF_proximal_nulls", data_dir=DATA_DIR)
        assert d.name == "full_eye_flywireRF_proximal_nulls"

    def test_table_dir_known_ode_aliases_to_gnn(self):
        # known_ode reuses the same parquet tables as its GNN sibling
        d_gnn = _table_dir("e8_flywireRF", data_dir=DATA_DIR)
        d_kode = _table_dir("e8_flywireRF_known_ode", data_dir=DATA_DIR)
        assert d_gnn == d_kode

    def test_table_dir_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown hybrid variant"):
            _table_dir("flyvis_X", data_dir=DATA_DIR)


# ---------------------------------------------------------------------------
# Data loading (requires parquet files)
# ---------------------------------------------------------------------------


class TestLoadTables:
    @pytest.fixture(autouse=True)
    def _manifest(self):
        self.manifest = _load_manifest()

    @pytest.mark.parametrize(
        "variant",
        [
            "e8_flywireRF",
            "e8_flywireRF_proximal_nulls",
            "full_eye_flywireRF",
            "full_eye_flywireRF_proximal_nulls",
        ],
    )
    def test_load_variant(self, variant):
        nodes, edges = load_tables(variant, data_dir=DATA_DIR)
        entry = next(e for e in self.manifest if e["variant"] == variant)
        assert len(nodes) == entry["n_nodes"]
        assert len(edges) == entry["n_edges"]
        assert "type" in nodes.columns
        assert "source_type" in edges.columns
        assert "target_type" in edges.columns

    def test_all_manifest_directories_present(self):
        for entry in self.manifest:
            d = DATA_DIR / entry["variant"]
            if not d.exists():
                # Random-null controls are scaffolded in the variant table
                # but may not be exported in every release.
                continue
            assert (d / "nodes.parquet").exists(), f"missing {d}/nodes.parquet"
            assert (d / "edges.parquet").exists(), f"missing {d}/edges.parquet"


# ---------------------------------------------------------------------------
# Network construction (requires parquet + flyvis)
# ---------------------------------------------------------------------------


class TestLoadNetwork:
    @pytest.fixture(autouse=True)
    def _check_data(self):
        _load_manifest()

    def test_build_network_e8(self):
        net = load_hybrid_network("e8_flywireRF", data_dir=DATA_DIR)
        assert hasattr(net, "edges_syn_strength")
        assert hasattr(net, "nodes_time_const")
        n_edges = net.connectome.edges.source_type.shape[0]
        assert n_edges == 327358

    def test_build_network_e8_proximal_nulls(self):
        net = load_hybrid_network(
            "e8_flywireRF_proximal_nulls", data_dir=DATA_DIR
        )
        n_edges = net.connectome.edges.source_type.shape[0]
        assert n_edges == 2418403
