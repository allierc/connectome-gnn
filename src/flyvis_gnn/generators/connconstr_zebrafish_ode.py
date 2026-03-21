"""Zebrafish oculomotor linear integrator ODE.

Ref: Beiran & Litwin-Kumar (2023), Fig 5 — zebrafish oculomotor integrator
     papers/Code_NN/Code_NN/nn_fig5_zebrafish_teacher.py

ODE (simulate_series, line 172):
    r[i,:] = r[i-1,:] + dt * (W @ r[i-1,:] - r[i-1,:] + I[i-1] * v_in) / tau

Continuous form:
    dr/dt = (-r + W @ r + I(t) * v_in) / tau

This is a LINEAR ODE — no activation function. tau=1.0 fixed.
W is scaled so spectral radius = 0.9 (line 179).
"""

import torch
import torch.nn as nn

from flyvis_gnn.generators.ode_params import ZebrafishODEParams
from flyvis_gnn.neuron_state import NeuronState


class ZebrafishODE(nn.Module):
    """Linear integrator ODE for the zebrafish oculomotor system.

    Computes dv/dt = (-v + msg + stimulus * v_in) / tau
    where msg = sum_j W_j * v_j over incoming edges (no nonlinearity).

    Uses explicit scatter_add for message passing (no PyG dependency).
    """

    def __init__(self, ode_params=None, device=None):
        super().__init__()

        if isinstance(ode_params, dict):
            ode_params = ZebrafishODEParams(**ode_params)
        self.ode_params = ode_params
        self.device = device

        if self.ode_params is not None:
            self.ode_params.to(device)

    def _compute_messages(self, v, edge_index):
        """Compute per-edge messages and aggregate via scatter_add.

        Linear model: message = W_e * v_src (no activation).

        Ref: simulate_series line 172 — np.dot(W, r[i-1,:])
        This is the sparse equivalent of the dense W @ r.

        Args:
            v: (N, 1) firing rates
            edge_index: (2, E) source/destination indices

        Returns:
            msg: (N, 1) aggregated messages per node
        """
        src, dst = edge_index

        # Linear: no activation on source voltages
        # Ref: zebrafish model is purely linear (no nonlinearity mentioned in paper)
        edge_msg = self.ode_params.W[:, None] * v[src]

        msg = torch.zeros(v.shape[0], 1, device=self.device, dtype=v.dtype)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)

        return msg

    def forward(self, state: NeuronState, edge_index: torch.Tensor, **kwargs):
        """Compute dv/dt from neuron state and connectivity.

        Ref: simulate_series line 172:
            r[i,:] = r[i-1,:] + dt * (W @ r[i-1,:] - r[i-1,:] + I[i-1]*v_in) / tau

        Args:
            state: NeuronState with voltage, stimulus fields
            edge_index: (2, E) tensor of (src, dst) edge indices

        Returns:
            dv: (N, 1) voltage derivative
        """
        v = state.voltage.unsqueeze(-1)           # (N, 1)
        stim = state.stimulus.unsqueeze(-1)        # (N, 1) — I(t) broadcast to all neurons
        v_in = self.ode_params.v_in[:, None]       # (N, 1) — input direction vector
        tau = self.ode_params.tau

        msg = self._compute_messages(v, edge_index)

        # dr/dt = (-r + W @ r + I * v_in) / tau
        dv = (-v + msg + stim * v_in) / tau

        return dv
