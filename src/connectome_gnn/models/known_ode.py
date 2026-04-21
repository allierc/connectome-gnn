"""Known-ODE baseline — uses the exact ground-truth ODE structure per bio-model.

Each bio-model has a distinct activation function and parameter set:
  - Flyvis:     g_phi = ReLU,      dv/dt = (-v + msg + I + V_rest) / tau
  - Drosophila CX: g_phi = exp(g)*softplus(v+b, beta=5),  dv/dt = alpha*(-v + msg + I) / tau
  - Larva:      g_phi = g*softplus(v),  dv/dt = (-v + msg + I + bias) / tau  (two populations)
  - Zebrafish:  g_phi = identity,  dv/dt = (-v + msg + I) / tau  (tau=1 fixed)

All parameter sets (tau, V_rest/bias, W, gains) are directly learned.
No MLP, no embeddings — the activation function is the known ground-truth form.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from connectome_gnn.models.registry import register_model
from connectome_gnn.neuron_state import NeuronState


class KnownODEBase(nn.Module):
    """Base class for known-ODE baselines. Subclasses override _activation and _update."""

    # Known_ODE baselines never learn hidden-neuron INRs; the trainer tests
    # `model.NNR_hidden is not None` to decide between fill-in and zero-silencing.
    NNR_hidden = None

    def __init__(self, aggr_type='add', config=None, device=None):
        super().__init__()

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.model = model_config.signal_model_name
        self.calcium_type = simulation_config.calcium_type
        self.n_neurons = simulation_config.n_neurons
        self.n_input_neurons = simulation_config.n_input_neurons
        self.n_edges = simulation_config.n_edges
        self.n_extra_null_edges = simulation_config.n_extra_null_edges
        self.batch_size = train_config.batch_size
        self.update_type = model_config.update_type

        # Per-edge weights W (shared across all variants)
        n_w = self.n_edges + self.n_extra_null_edges
        w_init_mode = getattr(train_config, 'w_init_mode', 'zeros')
        if w_init_mode == 'zeros':
            W_init = torch.zeros(n_w, device=device, dtype=torch.float32)
        elif w_init_mode == 'randn_scaled':
            w_init_scale = getattr(train_config, 'w_init_scale', 1.0)
            W_init = torch.randn(n_w, device=device, dtype=torch.float32) * (w_init_scale / math.sqrt(n_w))
        elif w_init_mode == 'uniform_scaled':
            w_init_scale = getattr(train_config, 'w_init_scale', 1.0)
            bound = w_init_scale / math.sqrt(n_w)
            W_init = (torch.rand(n_w, device=device, dtype=torch.float32) * 2 - 1) * bound
        else:
            W_init = torch.randn(n_w, device=device, dtype=torch.float32)
        self.W = nn.Parameter(W_init[:, None], requires_grad=True)

    def get_learned_tau(self):
        """Return learned tau with the correct transform. Override in subclass."""
        return None

    def get_learned_vrest(self):
        """Return learned V_rest. Override in subclass."""
        return None

    def get_learned_gain(self):
        """Return learned gain. Override in subclass."""
        return None

    def get_learned_bias(self):
        """Return learned bias. Override in subclass."""
        return None

    def _activation(self, v):
        """Apply g_phi activation to source voltages. Override in subclass."""
        raise NotImplementedError

    def _compute_messages(self, v, edge_index):
        """msg_j = W_j * g_phi(v_j), aggregated via scatter_add."""
        src, dst = edge_index
        n_edges_batch = edge_index.shape[1]
        edge_W_idx = torch.arange(n_edges_batch, device=self.device) % (self.n_edges + self.n_extra_null_edges)

        activated = self._activation(v[src])
        edge_msg = self.W[edge_W_idx] * activated

        msg = torch.zeros(v.shape[0], 1, device=self.device, dtype=v.dtype)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)
        return msg

    def _update(self, v, msg, excitation, particle_id):
        """Compute dv/dt from v, aggregated messages, and excitation. Override in subclass."""
        raise NotImplementedError

    def forward(self, state: NeuronState, edge_index: torch.Tensor,
                data_id=[], k=[], return_all=False, **kwargs):
        self.data_id = data_id.squeeze().long().clone().detach() if hasattr(data_id, 'squeeze') else data_id

        v = state.observable(self.calcium_type)
        excitation = state.stimulus.unsqueeze(-1)
        particle_id = state.index.long()

        msg = self._compute_messages(v, edge_index)
        pred = self._update(v, msg, excitation, particle_id)

        if return_all:
            return pred, None, msg
        return pred


# ---------------------------------------------------------------------------
# Flyvis: g_phi = ReLU, dv/dt = (-v + msg + I + V_rest) / tau
# ---------------------------------------------------------------------------

@register_model("flyvis_known_ode")
class FlyvisKnownODE(KnownODEBase):

    def __init__(self, aggr_type='add', config=None, device=None):
        super().__init__(aggr_type=aggr_type, config=config, device=device)
        self.raw_tau = nn.Parameter(
            torch.zeros(self.n_neurons, device=device, dtype=torch.float32))
        self.V_rest = nn.Parameter(
            torch.zeros(self.n_neurons, device=device, dtype=torch.float32))

    def _activation(self, v):
        return F.relu(v)

    def get_learned_tau(self):
        return F.softplus(self.raw_tau).detach()

    def get_learned_vrest(self):
        return self.V_rest.detach()

    def _update(self, v, msg, excitation, particle_id):
        tau = F.softplus(self.raw_tau[particle_id]).unsqueeze(-1)
        v_rest = self.V_rest[particle_id].unsqueeze(-1)
        return (-v + msg + excitation + v_rest) / tau


# ---------------------------------------------------------------------------
# Drosophila CX: g_phi = exp(g)*softplus(v+b, beta=5)
#   dv/dt = alpha * (-v + msg + I) / tau
#   tau = 2.6 + 2.4 * tanh(tau_raw) -> bounded [0.2, 5.0]
# ---------------------------------------------------------------------------

@register_model("drosophila_cx_known_ode")
class DrosophilaCxKnownODE(KnownODEBase):

    def __init__(self, aggr_type='add', config=None, device=None):
        super().__init__(aggr_type=aggr_type, config=config, device=device)
        self.raw_tau = nn.Parameter(
            torch.zeros(self.n_neurons, device=device, dtype=torch.float32))
        self.g = nn.Parameter(
            torch.zeros(self.n_neurons, device=device, dtype=torch.float32))
        self.bias = nn.Parameter(
            torch.zeros(self.n_neurons, device=device, dtype=torch.float32))
        self.alpha = 1.0
        self.beta = 5.0

    def _activation(self, v):
        # v is (E, 1) from source neurons — need per-source g and b
        # This is called with v[src], so we need source indices
        # Override _compute_messages to pass source indices
        return F.softplus(v, beta=self.beta)

    def _compute_messages(self, v, edge_index):
        """CX-specific: msg_j = W_j * exp(g_j) * softplus(v_j + b_j, beta=5)."""
        src, dst = edge_index
        n_edges_batch = edge_index.shape[1]
        edge_W_idx = torch.arange(n_edges_batch, device=self.device) % (self.n_edges + self.n_extra_null_edges)

        src_mod = src % self.n_neurons
        gain = torch.exp(self.g[src_mod]).unsqueeze(-1)
        bias = self.bias[src_mod].unsqueeze(-1)
        activated = gain * F.softplus(v[src] + bias, beta=self.beta)
        edge_msg = self.W[edge_W_idx] * activated

        msg = torch.zeros(v.shape[0], 1, device=self.device, dtype=v.dtype)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)
        return msg

    def get_learned_tau(self):
        return (2.6 + 2.4 * torch.tanh(self.raw_tau)).detach()

    def get_learned_gain(self):
        return torch.exp(self.g).detach()

    def get_learned_bias(self):
        return self.bias.detach()

    def _update(self, v, msg, excitation, particle_id):
        # tau = 2.6 + 2.4 * tanh(tau_raw) -> bounded [0.2, 5.0]
        tau = (2.6 + 2.4 * torch.tanh(self.raw_tau[particle_id])).unsqueeze(-1)
        return self.alpha * (-v + msg + excitation) / tau


# ---------------------------------------------------------------------------
# Larva: two-population, g_phi = gain * softplus(v)
#   premotor: dv/dt = (-v + gp*softplus(v) @ Jpp + bp + stim) / taup
#   motor:    dv/dt = (-v + gm*softplus(v) @ Jpm + bm) / taum
# ---------------------------------------------------------------------------

@register_model("larva_known_ode")
class LarvaKnownODE(KnownODEBase):

    def __init__(self, aggr_type='add', config=None, device=None):
        super().__init__(aggr_type=aggr_type, config=config, device=device)
        self.raw_tau = nn.Parameter(
            torch.zeros(self.n_neurons, device=device, dtype=torch.float32))
        self.gain = nn.Parameter(
            torch.ones(self.n_neurons, device=device, dtype=torch.float32))
        self.bias = nn.Parameter(
            torch.zeros(self.n_neurons, device=device, dtype=torch.float32))

    def _compute_messages(self, v, edge_index):
        """Larva: msg_j = W_j * gain_j * softplus(v_j)."""
        src, dst = edge_index
        n_edges_batch = edge_index.shape[1]
        edge_W_idx = torch.arange(n_edges_batch, device=self.device) % (self.n_edges + self.n_extra_null_edges)

        src_mod = src % self.n_neurons
        g = self.gain[src_mod].unsqueeze(-1)
        activated = g * F.softplus(v[src])
        edge_msg = self.W[edge_W_idx] * activated

        msg = torch.zeros(v.shape[0], 1, device=self.device, dtype=v.dtype)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)
        return msg

    def _activation(self, v):
        return F.softplus(v)

    def get_learned_tau(self):
        return F.softplus(self.raw_tau).detach()

    def get_learned_gain(self):
        return self.gain.detach()

    def get_learned_bias(self):
        return self.bias.detach()

    def _update(self, v, msg, excitation, particle_id):
        tau = F.softplus(self.raw_tau[particle_id]).unsqueeze(-1)
        b = self.bias[particle_id].unsqueeze(-1)
        return (-v + msg + excitation + b) / tau


# ---------------------------------------------------------------------------
# Zebrafish oculomotor: g_phi = identity, dv/dt = (-v + msg + I) / tau
#   tau = 1 fixed, no nonlinearity
# ---------------------------------------------------------------------------

@register_model("zebrafish_oculomotor_known_ode", "zebrafish_known_ode")
class ZebrafishKnownODE(KnownODEBase):

    def _activation(self, v):
        return v  # identity — linear ODE

    def _update(self, v, msg, excitation, particle_id):
        return -v + msg + excitation  # tau=1 fixed, so dv/dt = -v + Wr + I
