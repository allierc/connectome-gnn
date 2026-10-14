"""Where the frame loop starts: neuron positions and types, the steady state, and the state x."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from connectome_gnn.neuron_state import NeuronState


@dataclass(frozen=True)
class Geometry:
    """Positions and types of the network's neurons (arrays: treat as read-only)."""

    X1: Any                     # (N, 2) float32 tensor, x = u + v/2, y = v sqrt(3)/2 for EVERY neuron
    x_coords: Any               # photoreceptors only (preview plots, tile stimuli)
    y_coords: Any
    u_coords: Any
    v_coords: Any
    node_types_int: Any         # (N,) index of each neuron's type in the sorted type names
    grouped_types: Any          # (N,) group_by_direction_and_function of each type


def init_geometry(net, device) -> Geometry:
    """Positions and type indices from ``net.connectome.nodes``."""
    from connectome_gnn.generators.flyvis_ode import (
        get_all_neuron_positions_from_net,
        get_photoreceptor_positions_from_net,
        group_by_direction_and_function,
    )

    # Per-neuron hexagonal Cartesian positions for the *full* network — every
    # node carries (u, v) in net.connectome.nodes, so we use the standard
    # x = u + 0.5*v, y = v*sqrt(3)/2 mapping for both photoreceptors and
    # non-retinal neurons. This replaces the previous behaviour where
    # non-retinal neurons received random equidistant positions; the spatial
    # NGP path (ngp_hidden_spatial=True) and any column-aware figure that
    # reads pos[hidden_ids] depend on this fix.
    x_coords_all, y_coords_all, _u_all, _v_all = get_all_neuron_positions_from_net(net)
    # Keep the photoreceptor-only arrays available for downstream code that
    # filters on input neurons (visual SIREN initialisation, etc.).
    x_coords, y_coords, u_coords, v_coords = get_photoreceptor_positions_from_net(net)

    node_types = np.array(net.connectome.nodes["type"])
    node_types_str = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in node_types]
    grouped_types = np.array([group_by_direction_and_function(t) for t in node_types_str])
    _, node_types_int = np.unique(node_types, return_inverse=True)

    X1 = torch.tensor(
        np.stack((x_coords_all, y_coords_all), axis=1),
        dtype=torch.float32, device=device,
    )
    return Geometry(X1=X1, x_coords=x_coords, y_coords=y_coords, u_coords=u_coords, v_coords=v_coords,
                    node_types_int=node_types_int, grouped_types=grouped_types)


def steady_state(spec, net, device):
    """The network's steady state under a uniform stimulus of ``steady_state_value`` (t_pre = 2 s).

    Touches ``net.stimulus``'s buffer (flyvis zeroes and fills it), which is why
    it is a stage of its own and not a pure function of the network.
    """
    state = net.steady_state(t_pre=2.0, dt=spec.delta_t, batch_size=1, value=spec.network.steady_state_value)
    return state.nodes.activity.squeeze().to(device)


def init_state(net, stimuli, geometry: Geometry, initial_state, device) -> NeuronState:
    """The NeuronState x the train split starts from.

    Order, as legacy: item(0) of the stimulus dataset is fed to
    ``net.stimulus``, then the initial calcium is drawn (torch.rand), then x is
    built. QUIRK: ``x.stimulus`` IS ``net.stimulus()`` -- on CPU a view of
    net.stimulus.buffer, not a copy -- and ``x.voltage`` is ``initial_state``
    itself; see ``VoltageGeneration.init_state``.
    """
    n_neurons = len(initial_state)
    sequences = stimuli.item(0)["lum"]
    frame = sequences[0][None, None]
    net.stimulus.add_input(frame)

    # init neuron state x

    _init_calcium = torch.rand(n_neurons, dtype=torch.float32, device=device)

    return NeuronState(
        index=torch.arange(n_neurons, dtype=torch.long, device=device),
        pos=geometry.X1,
        voltage=initial_state.to(device),
        stimulus=net.stimulus().squeeze().to(device),
        group_type=torch.tensor(geometry.grouped_types, dtype=torch.long, device=device),
        neuron_type=torch.tensor(geometry.node_types_int, dtype=torch.long, device=device),
        calcium=_init_calcium,
        fluorescence=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        noise=torch.zeros(n_neurons, dtype=torch.float32, device=device),
    )
