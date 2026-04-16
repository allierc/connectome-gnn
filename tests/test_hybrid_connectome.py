"""Smoke tests for hybrid connectome loading and network construction."""

import json
from pathlib import Path

import pytest

from connectome_gnn.generators.hybrid_connectome import (
    _table_dir,
    _VARIANT_PREFIXES,
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
        assert is_flyvis_hybrid_model("flyvis_hybrid")
        assert is_flyvis_hybrid_model("flyvis_hybrid_flywireRF")
        assert is_flyvis_hybrid_model("flyvis_hybrid_zeroedge")
        assert is_flyvis_hybrid_model("flyvis_hybrid_flywireRF_zeroedge")
        assert not is_flyvis_hybrid_model("flyvis_A")
        assert not is_flyvis_hybrid_model("flyvis_B")

    def test_variant_prefixes_complete(self):
        expected = {
            "flyvis_hybrid",
            "flyvis_hybrid_flywireRF",
            "flyvis_hybrid_zeroedge",
            "flyvis_hybrid_flywireRF_zeroedge",
        }
        assert expected.issubset(set(_VARIANT_PREFIXES.keys()))

    def test_table_dir_non_zero(self):
        d = _table_dir("flyvis_hybrid", 8, 1, DATA_DIR)
        assert d.name == "flyvis_hybrid_e8"

    def test_table_dir_zero_edge(self):
        d = _table_dir("flyvis_hybrid_zeroedge", 15, 3, DATA_DIR)
        assert d.name == "flyvis_hybrid_zeroedge_e15_u3"

    def test_table_dir_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown hybrid variant"):
            _table_dir("flyvis_X", 8, 1, DATA_DIR)


# ---------------------------------------------------------------------------
# Data loading (requires parquet files)
# ---------------------------------------------------------------------------


class TestLoadTables:
    @pytest.fixture(autouse=True)
    def _manifest(self):
        self.manifest = _load_manifest()

    @pytest.mark.parametrize(
        "variant,extent,u",
        [
            ("flyvis_hybrid", 8, 1),
            ("flyvis_hybrid_flywireRF", 8, 1),
            ("flyvis_hybrid_zeroedge", 8, 1),
            ("flyvis_hybrid_flywireRF_zeroedge", 8, 1),
        ],
    )
    def test_load_e8_u1(self, variant, extent, u):
        nodes, edges = load_tables(variant, extent, u, DATA_DIR)
        entry = next(
            e for e in self.manifest
            if e["variant"] == variant and e["extent"] == extent
            and e.get("edge_uncertainty", 1) == u
        )
        assert len(nodes) == entry["n_nodes"]
        assert len(edges) == entry["n_edges"]
        assert "type" in nodes.columns
        assert "source_type" in edges.columns
        assert "target_type" in edges.columns

    def test_all_16_directories_present(self):
        for entry in self.manifest:
            u = entry.get("edge_uncertainty", 1)
            d = _table_dir(entry["variant"], entry["extent"], u, DATA_DIR)
            assert (d / "nodes.parquet").exists(), f"missing {d}/nodes.parquet"
            assert (d / "edges.parquet").exists(), f"missing {d}/edges.parquet"


# ---------------------------------------------------------------------------
# Network construction (requires parquet + flyvis)
# ---------------------------------------------------------------------------


class TestLoadNetwork:
    @pytest.fixture(autouse=True)
    def _check_data(self):
        _load_manifest()

    def test_build_network_v1_e8(self):
        net = load_hybrid_network("flyvis_hybrid", 8, 1, data_dir=DATA_DIR)
        assert hasattr(net, "edges_syn_strength")
        assert hasattr(net, "nodes_time_const")
        n_edges = net.connectome.edges.source_type.shape[0]
        assert n_edges == 451451

    def test_build_network_v5_e8_u2(self):
        net = load_hybrid_network(
            "flyvis_hybrid_flywireRF_zeroedge", 8, 2, data_dir=DATA_DIR
        )
        n_edges = net.connectome.edges.source_type.shape[0]
        assert n_edges == 639885
