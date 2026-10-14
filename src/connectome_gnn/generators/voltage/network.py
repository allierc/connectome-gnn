"""The generating network: a flyvis network (current or conductance dynamics) or a FlyWire hybrid.

``build_network`` returns a NetworkArtifact. Its ``net`` is a LINEAR resource:
``net.stimulus`` owns a buffer that ``add_input`` zeroes and refills in place,
and on CPU the neuron state's ``stimulus`` is a view of that buffer (see
QUIRKS in the package docstring), so the network is handed on, never copied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from connectome_gnn.generators.utils import is_flyvis_hybrid_model
from connectome_gnn.log import get_logger

logger = get_logger(__name__)

PHOTORECEPTOR_TYPES = {'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8'}
BOXEYE_KERNEL_SIZE = 13


@dataclass
class NetworkArtifact:
    """The network and what the stimulus datasets need from it."""

    net: Any                       # flyvis Network (or the hybrid's); LINEAR, see the module docstring
    orig_net: Any                  # hybrid only: what load_hybrid_network returns beside net; unused
    boxfilter: dict                # BoxEye kwargs every stimulus dataset renders with
    n_input_neurons_net: int       # photoreceptors counted from the node types (printed only)


def flyvis_preamble() -> None:
    """Imports and logger levels legacy sets before it builds any network.

    The imports run here, in legacy order, for their side effects (importing
    flyvis resets the root logger to INFO through basicConfig, ode_params
    registers its classes); later stages import what they use again, which is
    a lookup in sys.modules.
    """
    # flyvis.__init__ sets root logger to INFO via basicConfig — restore to WARNING
    import logging

    from flyvis import Network, NetworkView  # noqa: F401
    from flyvis.datasets.sintel import AugmentedSintel  # noqa: F401
    from flyvis.utils.config_utils import CONFIG_PATH, get_default_config  # noqa: F401

    from connectome_gnn.generators.flyvis_ode import (  # noqa: F401
        FlyVisODE,
        get_photoreceptor_positions_from_net,
        group_by_direction_and_function,
    )
    from connectome_gnn.generators.ode_params import FlyVisCurrentODEParams, get_ode_params_class  # noqa: F401
    from connectome_gnn.utils import setup_flyvis_model_path

    logging.getLogger().setLevel(logging.WARNING)
    setup_flyvis_model_path()

    # Initialize the flyvis network first (fast) so we can print actual network stats before
    # the slow stimulus rendering begins.
    logging.getLogger("flyvis.utils.logging_utils").setLevel(logging.ERROR)


def build_hybrid(spec) -> tuple:
    """(net, orig_net) of a FlyWire hybrid, from the pre-computed connectome tables."""
    from connectome_gnn.generators.hybrid_connectome import load_hybrid_network

    ns = spec.network
    signal_name = ns.signal_model_name
    edge_uncertainty = ns.edge_uncertainty
    flyvis_model_id = f"flow/{ns.ensemble_id}/{ns.model_id}"
    logger.info(f"loading hybrid network ({signal_name}, extent={ns.extent}, u={edge_uncertainty})...")
    net, orig_net = load_hybrid_network(
        signal_name=signal_name,
        extent=ns.extent,
        edge_uncertainty=edge_uncertainty,
        model=flyvis_model_id,
    )
    logger.info(
        f"hybrid network: {net.connectome.nodes.type[:].shape[0]} nodes, "
        f"{net.connectome.edges.source_index[:].shape[0]} edges"
    )
    return net, orig_net


def build_flyvis(spec):
    """A standard flyvis network with the trained parameters of flow/<ensemble>/<model>."""
    from flyvis import Network, NetworkView
    from flyvis.utils.config_utils import CONFIG_PATH, get_default_config

    ns = spec.network
    assert "flywire" not in ns.signal_model_name, "Should have taken the if-branch"
    config_net = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
    config_net.connectome.extent = ns.extent
    # ONE INDEX FOR EITHER FAMILY. The published models carry exactly one
    # checkpoint and it is the trained one, which is why this used to be a
    # hard-coded 0; a TASK-TRAINED run of ours has 65 and 0 is the untrained
    # network, so generating from it would silently produce data from a model
    # that never learned anything.
    _chkpt = ns.conductance_checkpoint_index
    if ns.ground_truth_model == "flyvis_conductance":
        # THE SAME TRANSPLANT, WITH THE CONDUCTANCE DYNAMICS. flyvis shares
        # every parameter by cell type and filter tap, so a state dict trained
        # at extent 15 loads into an extent-8 network unchanged -- that is what
        # the current-based path already relies on. All this branch adds is the
        # dynamics class and the two reversal parameters, so the shapes match
        # what the flow run saved.
        #
        # Importing the package is what REGISTERS them: flyvis resolves both by
        # class name against the live subclass tree and, finding neither, would
        # warn and fall back to a base class whose methods are `pass`.
        from datamate import Namespace

        import flyvis_conductance_optical_flow  # noqa: F401
        from flyvis_conductance_optical_flow.config import CONDUCTANCE_DYNAMICS

        config_net.dynamics.type = CONDUCTANCE_DYNAMICS
        for name, dim in (("E_exc_raw", "global"), ("E_inh_raw", "per_type")):
            config_net.node_config[name] = Namespace(
                type="ReversalPotential", groupby=["type"], initial_dist="Value",
                value=0.0, reversal_dim=dim, requires_grad=True)
        # NOT checkpoint 0: that is the untrained network. See
        # SimulationConfig.conductance_checkpoint_index.
    net = Network(**config_net)
    nnv = NetworkView(f"flow/{ns.ensemble_id}/{ns.model_id}")
    trained_net = nnv.init_network(checkpoint=_chkpt)
    net.load_state_dict(trained_net.state_dict())
    if ns.ground_truth_model == "flyvis_conductance":  # golden-uncovered: B10
        got = type(net.dynamics).__name__
        if got != CONDUCTANCE_DYNAMICS:
            raise RuntimeError(
                f"network.dynamics is {got}, not {CONDUCTANCE_DYNAMICS}; flyvis "
                "fell back to a base class whose methods are `pass`.")
        print(f"\033[96m  conductance generator: flow/{ns.ensemble_id}/"
              f"{ns.model_id} checkpoint {_chkpt}, {got}\033[0m", flush=True)
    return net


def print_network_stats(net) -> int:
    """Print node, photoreceptor and edge counts; return the photoreceptor count."""
    _node_types_str = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in net.connectome.nodes["type"][:]]
    n_input_neurons_net = int(np.sum([t in PHOTORECEPTOR_TYPES for t in _node_types_str]))
    print(f"  n_neurons:       {net.n_nodes}", flush=True)
    print(f"  n_input_neurons: {n_input_neurons_net}", flush=True)
    print(f"  n_edges:         {net.n_edges}", flush=True)
    return n_input_neurons_net


def attach_flywire_stimulus(net) -> dict:
    """Render at a standard hex disk and project each frame onto the FlyWire input columns.

    For FlyWire hybrids the stimulus is rendered on a *standard* hex disk
    large enough to contain every FlyWire input column, which lets flyvis's
    HexFlip/HexRotate augmentations apply unchanged (they require a regular
    lattice); ``net.stimulus.add_input`` is then wrapped so every frame is
    projected onto the actual FlyWire input columns. Returns the BoxEye kwargs.
    Draws torch RNG: building the standard BoxEye initialises a Conv2d.
    """
    from connectome_gnn.generators.flywire_eye import (
        standard_boxeye_and_flywire_index,
    )
    _be, _flywire_proj_idx, _be_extent = standard_boxeye_and_flywire_index(
        net, kernel_size=BOXEYE_KERNEL_SIZE,
    )
    boxfilter_arg = dict(extent=_be_extent, kernel_size=BOXEYE_KERNEL_SIZE)
    print(
        f"[stimulus] flywire_stimulus=True: rendering at standard "
        f"BoxEye(extent={_be_extent}) with {_be.hexals} hexals; "
        f"projecting to {_flywire_proj_idx.numel()} FlyWire columns",
        flush=True,
    )
    # Monkey-patch ``net.stimulus.add_input`` to project standard-hex
    # frames down to FlyWire columns. This way every existing call
    # site (including those inside the frame loop) works unchanged.
    _orig_add_input = net.stimulus.add_input

    def _patched_add_input(frame, *args, **kwargs):
        idx = _flywire_proj_idx.to(frame.device)
        return _orig_add_input(frame.index_select(-1, idx), *args, **kwargs)
    net.stimulus.add_input = _patched_add_input
    return boxfilter_arg


def build_network(spec) -> NetworkArtifact:
    """Build the network, switch autograd off, print its size, and pick the stimulus filter.

    Global side effects, as legacy: the root logger goes to WARNING and
    flyvis.utils.logging_utils to ERROR (flyvis_preamble), and
    ``torch.set_grad_enabled(False)`` stays in force until ``restore_grad``
    (or forever, if a later stage raises).
    """
    flyvis_preamble()
    if is_flyvis_hybrid_model(spec.network.signal_model_name):
        net, orig_net = build_hybrid(spec)
    else:
        net, orig_net = build_flyvis(spec), None
    torch.set_grad_enabled(False)
    n_input_neurons_net = print_network_stats(net)
    if spec.stimulus.flywire_stimulus:
        boxfilter = attach_flywire_stimulus(net)
    else:
        boxfilter = dict(extent=spec.network.extent, kernel_size=BOXEYE_KERNEL_SIZE)
    return NetworkArtifact(net=net, orig_net=orig_net, boxfilter=boxfilter,
                           n_input_neurons_net=n_input_neurons_net)
