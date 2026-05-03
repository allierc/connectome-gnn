"""Load pre-computed hybrid connectome tables and build a flyvis Network.

Usage::

    from connectome_gnn.generators.hybrid_connectome import load_hybrid_network

    net, orig_net = load_hybrid_network(
        signal_name="flyvis_hybrid_zeroedge",
        extent=8,
        edge_uncertainty=1,
        model="flow/0000/000",
    )
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datamate import Namespace
from flyvis.connectome import ConnectomeView
from flyvis.connectome.connectome import register_connectome, AVAILABLE_CONNECTOMES
from flyvis.network import Network
from flyvis.network.network_view import NetworkView
from flyvis.network.initialization import (
    Parameter,
    InitialDistribution,
    deepcopy_config,
    get_scatter_indices,
    symmetry_masks,
)
from flyvis.utils.class_utils import forward_subclass
from flyvis.utils.type_utils import byte_to_str

from connectome_gnn.log import get_logger

logger = get_logger(__name__)

# Default data directory: connectome-gnn/data/hybrid_connectomes/
_DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "hybrid_connectomes"


# ---------------------------------------------------------------------------
# In-memory connectome (replicated from flyrewire)
# ---------------------------------------------------------------------------

class _InMemoryDir:
    """Small array-backed table wrapper with flyvis Directory-like access."""

    def __init__(self, arrays):
        self._arrays = arrays
        for key, value in arrays.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return self._arrays[key]

    def __contains__(self, key):
        return key in self._arrays


class _InMemoryFlyvisConnectome:
    """Connectome-like object built from in-memory node/edge tables."""

    def __init__(self, nodes_df=None, edges_df=None, nodes_data=None, edges_data=None):
        if nodes_df is None and nodes_data is not None:
            nodes_df = pd.DataFrame(nodes_data)
        if edges_df is None and edges_data is not None:
            edges_df = pd.DataFrame(edges_data)
        if nodes_df is None or edges_df is None:
            raise ValueError("Provide nodes_df/edges_df or nodes_data/edges_data.")

        nodes = nodes_df.copy()
        edges = edges_df.copy()

        node_type_str = nodes["type"].astype(str)
        unique_types = pd.unique(node_type_str)

        self.unique_cell_types = np.asarray(unique_types, dtype="S")

        if "role" in nodes.columns:
            role_str = nodes["role"].astype(str)
        else:
            role_str = pd.Series("intermediate", index=nodes.index)

        input_types = pd.unique(node_type_str[role_str == "input"])
        output_types = pd.unique(node_type_str[role_str == "output"])
        inter_types = pd.unique(node_type_str[role_str == "intermediate"])

        self.input_cell_types = np.asarray(input_types, dtype="S")
        self.output_cell_types = np.asarray(output_types, dtype="S")
        self.intermediate_cell_types = np.asarray(inter_types, dtype="S")

        layout_rows = []
        for t in self.input_cell_types.astype(str):
            layout_rows.append((t, "retina"))
        for t in self.intermediate_cell_types.astype(str):
            layout_rows.append((t, "intermediate"))
        for t in self.output_cell_types.astype(str):
            layout_rows.append((t, "output"))
        self.layout = np.asarray(layout_rows, dtype="S")

        self.nodes = _InMemoryDir(
            {
                "index": nodes["index"].to_numpy(dtype=np.int64, copy=True),
                "type": node_type_str.to_numpy(dtype="S"),
                "u": nodes["u"].to_numpy(dtype=np.int32, copy=True),
                "v": nodes["v"].to_numpy(dtype=np.int32, copy=True),
                "role": role_str.to_numpy(dtype="S"),
            }
        )
        self.edges = _InMemoryDir(
            {
                "source_index": edges["source_index"].to_numpy(dtype=np.int64, copy=True),
                "target_index": edges["target_index"].to_numpy(dtype=np.int64, copy=True),
                "sign": edges["sign"].to_numpy(dtype=np.float32, copy=True),
                "n_syn": edges["n_syn"].to_numpy(dtype=np.float32, copy=True),
                "source_type": edges["source_type"].astype(str).to_numpy(dtype="S"),
                "target_type": edges["target_type"].astype(str).to_numpy(dtype="S"),
                "source_u": edges["source_u"].to_numpy(dtype=np.int32, copy=True),
                "target_u": edges["target_u"].to_numpy(dtype=np.int32, copy=True),
                "source_v": edges["source_v"].to_numpy(dtype=np.int32, copy=True),
                "target_v": edges["target_v"].to_numpy(dtype=np.int32, copy=True),
                "du": edges["du"].to_numpy(dtype=np.int32, copy=True),
                "dv": edges["dv"].to_numpy(dtype=np.int32, copy=True),
                "n_syn_certainty": edges["n_syn_certainty"].to_numpy(dtype=np.float32, copy=True),
            }
        )

        self.central_cells_index = np.int64(
            np.nonzero((self.nodes.u[:] == 0) & (self.nodes.v[:] == 0))[0],
        )

        layer_index = {}
        node_types_arr = self.nodes["type"][:]
        for cell_type in self.unique_cell_types[:]:
            node_indices = np.nonzero(node_types_arr == cell_type)[0]
            layer_index[cell_type.decode()] = np.int64(node_indices)
        self.nodes.layer_index = layer_index

    def __contains__(self, key):
        return hasattr(self, key)


class ZeroEdgeAwareSynapseCountScaling(Parameter):
    """SynapseCountScaling that zeros out zero-edge-only type pairs.

    Identical to flyvis's ``SynapseCountScaling`` except that type pairs
    whose edges *all* have ``n_syn == 0`` receive ``syn_strength = 0``
    instead of ``scale / 1e-6``.
    """

    @deepcopy_config
    def __init__(self, param_config, connectome):
        edges_dir = connectome.edges

        edges = pd.DataFrame({
            k: byte_to_str(edges_dir[k][:])
            for k in [*param_config.groupby, "n_syn"]
        })
        grouped_edges = edges.groupby(
            param_config.groupby, as_index=False, sort=False
        ).mean()

        scale = param_config.get("scale", 0.01)
        mean_n_syn = grouped_edges.n_syn.values

        # Zero-only type pairs: all edges have n_syn == 0 → group mean == 0.
        # Set their syn_strength to 0 so W is exactly 0.
        zero_mask = mean_n_syn == 0
        safe_mean = np.where(zero_mask, 1.0, mean_n_syn)  # avoid division by zero
        syn_strength = np.where(zero_mask, 0.0, scale / safe_mean)

        n_zeroed = int(zero_mask.sum())
        if n_zeroed > 0:
            logger.info(
                f"ZeroEdgeAwareSynapseCountScaling: zeroed syn_strength for "
                f"{n_zeroed}/{len(grouped_edges)} type pairs (all edges have n_syn=0)"
            )

        param_config.target_type = grouped_edges.target_type.values
        param_config.source_type = grouped_edges.source_type.values
        param_config.value = syn_strength

        self.indices = get_scatter_indices(edges, grouped_edges, param_config.groupby)
        self.parameter = forward_subclass(
            InitialDistribution, param_config, subclass_key="initial_dist"
        )
        self.keys = list(
            zip(
                param_config.source_type.tolist(),
                param_config.target_type.tolist(),
            )
        )
        self.symmetry_masks = symmetry_masks(
            param_config.get("symmetric", []), self.keys
        )


class HeterogeneousSynapseCount(Parameter):
    """Per-edge synapse counts without template averaging.

    Preserves the individual synapse count for every edge rather than
    grouping by (source_type, target_type, du, dv) and averaging.
    """

    @deepcopy_config
    def __init__(self, param_config, connectome):
        edges_dir = connectome.edges
        n_syn = np.asarray(edges_dir["n_syn"][:], dtype=np.float32)
        n_syn = np.maximum(n_syn, 1e-6)

        param_config.mean = np.log(n_syn).astype(np.float32)
        param_config.mode = "mean"

        self.parameter = forward_subclass(
            InitialDistribution, param_config, subclass_key="initial_dist"
        )
        self.indices = torch.arange(len(n_syn))
        self.keys = list(range(len(n_syn)))
        self.symmetry_masks = symmetry_masks(
            param_config.get("symmetric", []), self.keys
        )


# ---------------------------------------------------------------------------
# Directory naming convention
# ---------------------------------------------------------------------------

# Map signal_model_name → parquet variant directory name. The naming
# matches flyrewire's v2 hybrid_connectomes manifest: the eye-map axis
# (``e8_*`` vs ``full_eye_*``) is folded into the variant name, and
# zero-edge controls become ``_proximal_nulls`` / ``_random_nulls``.
# Known-ODE variants reuse the same parquet tables as their GNN sibling
# (``_known_ode`` only changes the model arch, not the connectome).
_VARIANT_TO_DIRNAME = {
    # e8 (truncated flyvis hex disk, FlyWire RFs)
    "e8_flywireRF": "e8_flywireRF",
    "e8_flywireRF_known_ode": "e8_flywireRF",
    "e8_flywireRF_mlp": "e8_flywireRF",
    "e8_flywireRF_eed": "e8_flywireRF",
    "e8_flywireRF_stimulus": "e8_flywireRF",
    "e8_flywireRF_proximal_nulls": "e8_flywireRF_proximal_nulls",
    "e8_flywireRF_proximal_nulls_known_ode": "e8_flywireRF_proximal_nulls",
    "e8_flywireRF_proximal_nulls_mlp": "e8_flywireRF_proximal_nulls",
    "e8_flywireRF_proximal_nulls_eed": "e8_flywireRF_proximal_nulls",
    "e8_flywireRF_proximal_nulls_stimulus": "e8_flywireRF_proximal_nulls",
    "e8_flywireRF_random_nulls": "e8_flywireRF_random_nulls",
    "e8_flywireRF_random_nulls_known_ode": "e8_flywireRF_random_nulls",
    "e8_flywireRF_typed_nulls": "e8_flywireRF_typed_nulls",
    "e8_flywireRF_typed_nulls_known_ode": "e8_flywireRF_typed_nulls",
    # full FlyWire eye (no extent applies)
    "full_eye_flywireRF": "full_eye_flywireRF",
    "full_eye_flywireRF_known_ode": "full_eye_flywireRF",
    "full_eye_flywireRF_mlp": "full_eye_flywireRF",
    "full_eye_flywireRF_eed": "full_eye_flywireRF",
    "full_eye_flywireRF_stimulus": "full_eye_flywireRF",
    "full_eye_flywireRF_proximal_nulls": "full_eye_flywireRF_proximal_nulls",
    "full_eye_flywireRF_proximal_nulls_known_ode": "full_eye_flywireRF_proximal_nulls",
    "full_eye_flywireRF_proximal_nulls_mlp": "full_eye_flywireRF_proximal_nulls",
    "full_eye_flywireRF_proximal_nulls_eed": "full_eye_flywireRF_proximal_nulls",
    "full_eye_flywireRF_proximal_nulls_stimulus": "full_eye_flywireRF_proximal_nulls",
    "full_eye_flywireRF_random_nulls": "full_eye_flywireRF_random_nulls",
    "full_eye_flywireRF_random_nulls_known_ode": "full_eye_flywireRF_random_nulls",
    "full_eye_flywireRF_typed_nulls": "full_eye_flywireRF_typed_nulls",
    "full_eye_flywireRF_typed_nulls_known_ode": "full_eye_flywireRF_typed_nulls",
}

# Back-compat alias kept for any external callers introspecting the table.
_VARIANT_PREFIXES = _VARIANT_TO_DIRNAME


def _table_dir(
    signal_name: str,
    extent: Optional[int] = None,
    edge_uncertainty: int = 1,
    data_dir: Optional[Path] = None,
) -> Path:
    """Resolve the directory containing nodes.parquet / edges.parquet.

    The ``extent`` and ``edge_uncertainty`` arguments are accepted for
    backwards compatibility but are ignored: the eye-map and null-edge
    configuration are now encoded directly in ``signal_name``.
    """
    del extent, edge_uncertainty  # encoded in signal_name now

    if data_dir is None:
        data_dir = Path(os.environ.get("HYBRID_CONNECTOME_DIR", str(_DEFAULT_DATA_DIR)))
    else:
        data_dir = Path(data_dir)

    dirname = _VARIANT_TO_DIRNAME.get(signal_name)
    if dirname is None:
        raise KeyError(
            f"Unknown hybrid variant '{signal_name}'. "
            f"Available: {sorted(_VARIANT_TO_DIRNAME.keys())}"
        )

    path = data_dir / dirname
    if not path.exists():
        raise FileNotFoundError(
            f"Hybrid connectome tables not found at {path}. "
            f"Run flyrewire/scripts/export_connectomes_v2.py to generate them."
        )
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_tables(
    signal_name: str,
    extent: Optional[int] = None,
    edge_uncertainty: int = 1,
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-computed nodes and edges DataFrames.

    Parameters
    ----------
    signal_name : str
        Model variant name (e.g. ``"e8_flywireRF_proximal_nulls"``).
    extent : int, optional
        Accepted for backwards compatibility; ignored. The eye-map axis is
        encoded in ``signal_name`` (``e8_*`` vs ``full_eye_*``).
    edge_uncertainty : int
        Accepted for backwards compatibility; ignored.
    data_dir : Path, optional
        Override data directory. Defaults to ``$HYBRID_CONNECTOME_DIR`` env
        var or ``connectome-gnn/data/hybrid_connectomes/``.

    Returns
    -------
    (nodes_df, edges_df)
        DataFrames matching the schema from ``_prepare_connectome_tables()``.
    """
    d = _table_dir(signal_name, extent, edge_uncertainty, data_dir)
    nodes_df = pd.read_parquet(d / "nodes.parquet")
    edges_df = pd.read_parquet(d / "edges.parquet")
    logger.info(
        f"loaded hybrid connectome: {d.name} "
        f"({len(nodes_df)} nodes, {len(edges_df)} edges)"
    )
    return nodes_df, edges_df


def load_hybrid_network(
    signal_name: str,
    extent: Optional[int] = None,
    edge_uncertainty: int = 1,
    model: Optional[str] = None,
    checkpoint: str = "best",
    data_dir: Optional[Path] = None,
) -> Network | Tuple[Network, Network]:
    """Load a hybrid connectome and build a flyvis Network.

    Replaces the ``FlyvisToFlywire(...).to_network()`` call with pre-computed
    tables.

    Parameters
    ----------
    signal_name : str
        Model variant name (see :data:`_VARIANT_TO_DIRNAME`).
    extent : int, optional
        Accepted for backwards compatibility; ignored. Encoded in
        ``signal_name``.
    edge_uncertainty : int
        Accepted for backwards compatibility; ignored.
    model : str, optional
        Flyvis model identifier (e.g. ``"flow/0000/000"``).  When provided,
        returns ``(hybrid_net, orig_net)`` with trained weights loaded.
    checkpoint : str
        Which checkpoint to load (default ``"best"``).
    data_dir : Path, optional
        Override data directory.

    Returns
    -------
    Network or (Network, Network)
    """
    nodes_df, edges_df = load_tables(signal_name, extent, edge_uncertainty, data_dir)

    # Register in-memory connectome type with flyvis
    connectome_type = "_InMemoryFlyvisConnectome"
    if connectome_type not in AVAILABLE_CONNECTOMES:
        register_connectome(_InMemoryFlyvisConnectome)

    connectome_cfg = {
        "type": connectome_type,
        "nodes_data": nodes_df.to_dict(orient="list"),
        "edges_data": edges_df.to_dict(orient="list"),
    }

    network_kwargs = {
        "connectome": connectome_cfg,
        "edge_config": Namespace(
            sign=Namespace(
                type="SynapseSign",
                initial_dist="Value",
                requires_grad=False,
                groupby=["source_type", "target_type"],
            ),
            syn_count=Namespace(
                type="HeterogeneousSynapseCount",
                initial_dist="Lognormal",
                mode="mean",
                requires_grad=False,
                std=1.0,
            ),
            syn_strength=Namespace(
                type="ZeroEdgeAwareSynapseCountScaling",
                initial_dist="Value",
                requires_grad=True,
                scale=0.01,
                clamp="non_negative",
                groupby=["source_type", "target_type"],
            ),
        ),
    }

    hybrid_net = Network(**network_kwargs)

    if model is None:
        return hybrid_net

    # Load trained flyvis model
    nnv = NetworkView(model)
    orig_config = nnv.dir.config.network.to_dict()
    orig_config["connectome"]["extent"] = extent
    orig_net = Network(**orig_config)
    chkpt_path = nnv.get_checkpoint(checkpoint)
    state = torch.load(chkpt_path, map_location="cpu", weights_only=False)
    trained_params = state.get("network", state)
    orig_net.load_state_dict(trained_params)

    # Hydrate hybrid with trained free parameters
    free_param_keys = ["edges_syn_strength", "nodes_time_const", "nodes_bias"]
    params = OrderedDict()
    for key in free_param_keys:
        if key in trained_params:
            params[key] = trained_params[key]

    # Handle shape mismatch for edges_syn_strength when cross-type zero-edges
    # introduce new (source_type, target_type) pairs not present in the original
    # flyvis checkpoint.  Map matching pairs and initialize new pairs with the
    # hybrid network's default values.
    hybrid_ss = hybrid_net.edges_syn_strength
    if "edges_syn_strength" in params and params["edges_syn_strength"].shape != hybrid_ss.shape:
        trained_ss = params["edges_syn_strength"]
        orig_keys = orig_net.edge_params.syn_strength.keys
        hybrid_keys = hybrid_net.edge_params.syn_strength.keys
        orig_key_to_idx = {k: i for i, k in enumerate(orig_keys)}

        expanded = hybrid_ss.detach().clone()  # start from hybrid defaults
        for j, hk in enumerate(hybrid_keys):
            if hk in orig_key_to_idx:
                expanded[j] = trained_ss[orig_key_to_idx[hk]]
        params["edges_syn_strength"] = expanded
        logger.info(
            f"expanded edges_syn_strength: {trained_ss.shape[0]} → {expanded.shape[0]} "
            f"({expanded.shape[0] - trained_ss.shape[0]} new type pairs from zero-edges)"
        )

    params["edges_syn_count"] = hybrid_net.edges_syn_count
    params["edges_sign"] = hybrid_net.edges_sign

    hybrid_net.load_state_dict(params)

    logger.info(
        f"hybrid network loaded: {len(nodes_df)} nodes, {len(edges_df)} edges"
    )
    return hybrid_net, orig_net
