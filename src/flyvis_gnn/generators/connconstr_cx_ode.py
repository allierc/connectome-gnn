"""Drosophila central complex ring attractor ODE.

Ref: Beiran & Litwin-Kumar (2023), Fig 5 — Drosophila adult central complex
     papers/Code_NN/Code_NN/nn_fig5_drosophilaCx_teacher.py

ODE (RNN.forward, line 187):
    h += alpha * (-h + exp(g) * softplus(h+b, beta) @ J^T + input @ W_in) * (1/tau)

Where:
    tau = 2.6 + 2.4 * tanh(tau_raw)   bounded in [0.2, 5.0]  (line 183)
    J = exp(wrec) * mwrec             log-space weights       (line 184)
    r = softplus(h + b, beta=5)       activation              (line 188, 105-106)
    gain = exp(g)                     per-neuron gain         (line 187)

Message from neuron j to neuron i:
    J[i,j] * exp(g_j) * softplus(h_j + b_j, beta=5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from flyvis_gnn.generators.ode_params import DrosophilaCxODEParams
from flyvis_gnn.neuron_state import NeuronState


class DrosophilaCxODE(nn.Module):
    """Ring attractor ODE for the Drosophila central complex.

    Computes dh/dt = alpha * (-h + msg + input) / tau
    where msg_i = sum_j J[i,j] * exp(g_j) * softplus(h_j + b_j, beta)
    and tau_i = 2.6 + 2.4 * tanh(tau_raw_i).

    Uses explicit scatter_add for message passing (no PyG dependency).
    """

    def __init__(self, ode_params=None, device=None):
        super().__init__()

        if isinstance(ode_params, dict):
            ode_params = DrosophilaCxODEParams(**ode_params)
        self.ode_params = ode_params
        self.device = device

        if self.ode_params is not None:
            self.ode_params.to(device)

    def _compute_messages(self, h, edge_index):
        """Compute per-edge messages and aggregate via scatter_add.

        Ref: RNN.forward line 187:
            exp(g.T) * r @ Jmat.t()
        where r = softplus(h + b, beta=5) and Jmat = exp(wrec) * mwrec.

        In sparse form: msg_i = sum_{edges to i} W_e * exp(g[src]) * softplus(h[src] + b[src])

        Args:
            h: (N, 1) hidden state
            edge_index: (2, E) source/destination indices

        Returns:
            msg: (N, 1) aggregated messages per node
        """
        src, dst = edge_index
        p = self.ode_params

        # Ref: line 188 — r = softplus(h + b, beta=5)
        # Ref: line 187 — exp(g) * r is the output of each neuron
        h_src = h[src]
        b_src = p.b[src, None]
        g_src = p.g[src, None]

        # Activation: Softplus with beta=5 (line 105-106)
        r_src = F.softplus(h_src + b_src, beta=p.beta)
        # Gain-weighted output: exp(g) * softplus(h+b)
        output_src = torch.exp(g_src) * r_src

        # Edge message: W_e * output_src
        edge_msg = p.W[:, None] * output_src

        msg = torch.zeros(h.shape[0], 1, device=self.device, dtype=h.dtype)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)

        return msg

    def forward(self, state: NeuronState, edge_index: torch.Tensor, **kwargs):
        """Compute dh/dt from neuron state and connectivity.

        Ref: RNN.forward line 187:
            h += alpha * (-h + exp(g)*softplus(h+b) @ J^T + input @ W_in) * (1/tau)

        Note: The input projection (input @ W_in) is handled externally via
        state.stimulus, which should already be the projected input for each neuron.

        Args:
            state: NeuronState with voltage (=h), stimulus fields
            edge_index: (2, E) tensor of (src, dst) edge indices

        Returns:
            dv: (N, 1) hidden state derivative
        """
        h = state.voltage.unsqueeze(-1)       # (N, 1)
        inp = state.stimulus.unsqueeze(-1)     # (N, 1) — already projected input
        p = self.ode_params

        msg = self._compute_messages(h, edge_index)

        # Ref: line 183 — tau = 2.6 + 2.4 * tanh(tau_raw), bounded [0.2, 5.0]
        tau = 2.6 + 2.4 * torch.tanh(p.tau_raw[:, None])

        # dh/dt = alpha * (-h + msg + input) / tau
        dv = p.alpha * (-h + msg + inp) / tau

        return dv
