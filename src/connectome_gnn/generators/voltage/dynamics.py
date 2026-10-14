"""The ground-truth dynamics: ODE parameters read from the network, and the FlyVisODE that integrates them."""

from __future__ import annotations

import torch

from connectome_gnn.log import get_logger

logger = get_logger(__name__)

GREEN, RESET = '\033[92m', '\033[0m'


def extract_ode_params(spec, net, device) -> tuple:
    """(ode_params, edge_index) of the model that GENERATES the data.

    WHICH synapse model generates the data is sim.ground_truth_model, NOT the
    model that will be trained on it -- see SimulationConfig.ground_truth_model
    for why those are separate axes. ``edge_index`` is ``ode_params.edge_index``
    moved to ``device``, which on the device it already lives on is the same
    tensor, not a copy.
    """
    from connectome_gnn.generators.ode_params import FlyVisCurrentODEParams

    ns = spec.network
    print(f"[DBG] extracting ODE params ({ns.ground_truth_model}) ...", flush=True)
    if ns.ground_truth_model == "conductance":
        from connectome_gnn.generators.ode_params import FlyVisConductanceODEParams

        # The connectome supplies the graph; the trained student supplies the
        # parameters. edge_index is a property of the connectome, not of the
        # fit, so it comes from the flyvis network either way.
        _edges = FlyVisCurrentODEParams.from_flyvis_network(net, device=device).edge_index
        ode_params = FlyVisConductanceODEParams.from_twin_checkpoint(
            ns.conductance_checkpoint, _edges, device=device)
        logger.info(
            f"conductance ground truth from {ns.conductance_checkpoint}: "
            f"E_inh={float(ode_params.E_inh[0]):+.3f} E_exc={float(ode_params.E_exc[0]):+.3f}, "
            f"{int(ode_params.edge_is_inh.sum())}/{ode_params.edge_is_inh.numel()} inhibitory edges")
    elif ns.ground_truth_model == "flyvis_conductance":  # golden-uncovered: B17
        # The parameters are already in the network -- `write_derived_params` has
        # materialised both reversals per neuron -- so unlike the twin path there is
        # no checkpoint to read and no square root to undo.
        from connectome_gnn.generators.ode_params import FlyVisConductanceODEParams

        ode_params = FlyVisConductanceODEParams.from_flyvis_network(net, device=device)
        logger.info(
            f"conductance ground truth from flow/{ns.ensemble_id}/{ns.model_id}: "
            f"E_exc {float(ode_params.E_exc.min()):+.3f}..{float(ode_params.E_exc.max()):+.3f}, "
            f"E_inh {float(ode_params.E_inh.min()):+.3f}..{float(ode_params.E_inh.max()):+.3f}, "
            f"{int(ode_params.edge_is_inh.sum())}/{ode_params.edge_is_inh.numel()} inhibitory edges")
    else:
        ode_params = FlyVisCurrentODEParams.from_flyvis_network(net, device=device)
    edge_index = ode_params.edge_index.to(device)
    print(f"[DBG] ODE params ready (edges={edge_index.shape[1]})", flush=True)
    return ode_params, edge_index


def build_ode(spec, ode_params, edge_index, device):
    """The FlyVisODE over ``ode_params`` (it keeps a reference, not a copy).

    RNG: a ``multiple_ReLU`` model whose ``params[0][0] <= 0`` draws its
    per-type gains with ``torch.randn`` here; every other model draws nothing.
    """
    from connectome_gnn.generators.flyvis_ode import FlyVisODE

    ns = spec.network
    pde = FlyVisODE(
        ode_params=ode_params,
        g_phi=torch.nn.functional.relu,
        params=ns.ode_params_list(),
        model_type=ns.signal_model_name,
        n_neuron_types=ns.n_neuron_types,
        device=device,
    )
    # Activity will be generated with the FULL connectivity below.
    # Edge removal (if any) is applied AFTER generation so that x_list/y_list
    # reflect the full-network dynamics. Only ode_params (the connectivity seen
    # by the GNN) is pruned, giving the GNN an incomplete adjacency matrix to
    # work with while the ground-truth activity it must predict is from the
    # full connectome. This tests whether the GNN can recover parameters
    # despite missing edges.
    print(f"{GREEN}[GENERATE] full connectivity: edge_index={edge_index.shape}  W={ode_params.W.shape}{RESET}")
    return pde
