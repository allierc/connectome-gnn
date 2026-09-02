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

        v = state.voltage.unsqueeze(-1)
        # Visual stimulus + optogenetic perturbation (when present) enter on
        # the same excitation channel; opto contributes +opto/tau to dv/dt.
        opto = state.optogenetics_stimulus if state.optogenetics_stimulus is not None else 0.0
        excitation = (state.stimulus + opto).unsqueeze(-1)
        particle_id = state.index.long()

        msg = self._compute_messages(v, edge_index)
        pred = self._update(v, msg, excitation, particle_id)

        if return_all:
            return pred, None, msg
        return pred


# ---------------------------------------------------------------------------
# Flyvis: g_phi = ReLU, dv/dt = (-v + msg + I + V_rest) / tau
# ---------------------------------------------------------------------------

@register_model(
    "flyvis_known_ode",
    "e8_flywireRF_known_ode",
    "e8_flywireRF_proximal_nulls_known_ode",
    "e8_flywireRF_random_nulls_known_ode",
    "full_eye_flywireRF_known_ode",
    "full_eye_flywireRF_proximal_nulls_known_ode",
    "full_eye_flywireRF_random_nulls_known_ode",
)
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
        idx = self._node_index(particle_id)
        tau = F.softplus(self.raw_tau[idx]).unsqueeze(-1)
        v_rest = self.V_rest[idx].unsqueeze(-1)
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


# ---------------------------------------------------------------------------
# Conductance flyvis: the STUDENT in a teacher-student distillation.
#
#   msg_ij  = W_ij^2 * relu(v_j) * (E_ij - v_i)
#   dv_i/dt = (-v_i + sum_j msg_ij + I_i + V_rest_i) / tau_i
#
# The teacher is the existing current-based flyvis data; the student is fitted on
# the derivative loss to recapitulate the same activity. It is NOT a parameter
# recovery -- the generator has no (E - v_i) term, so there is no conductance
# ground truth to recover, and a discrepancy in the learned constants is expected
# and unimportant. What has to hold is the ROLLOUT: once trained, this model
# becomes the ground truth for three new datasets (noise_free, 005, 05), so it is
# only useful if it can run free and stay on the teacher's trajectory.
#
# W ENTERS SQUARED, and that is the whole reason the sign works. The conductance
# must be non-negative so that the sign of the message comes from the driving
# force (E_ij - v_i) alone:
#     excitatory   E_exc > v  ->  msg > 0
#     inhibitory   E_inh < v  ->  msg < 0
# A signed W would multiply an inhibitory edge's negative driving force by a
# negative weight and produce an EXCITATORY message -- the sign can live in one
# factor or the other, not both. Squaring rather than softplus keeps it consistent
# with graph_model.g_phi_positive, which the GNN side already uses for the same
# reason.
#
# TWO REVERSAL POTENTIALS, one per presynaptic polarity, selected by the sign the
# connectome already carries: FlyVisODEParams.W is syn_strength * syn_count * sign,
# so sign(W_e) < 0 marks an inhibitory synapse. Under Dale's law that is a property
# of the presynaptic neuron, so the partition is anatomy rather than something to
# fit. Supplied by set_presynaptic_sign(); absent, every edge is treated as
# excitatory and E_inh never receives a gradient -- which is why the trainer must
# call it and why not calling it fails loudly rather than silently.
#
# REGISTERED AS `flyvis_cond_known_ode`, NOT `flyvis_conductance_known_ode`.
# Seven sites dispatch on `'flyvis_conductance' in signal_model_name`
# (metrics.py:602,705,828,897,1706; sparsify.py:755,837) and would all fire for a
# name containing that substring, building 6-column [v_i,v_j,a_i,a_j] g_phi
# features for a model that has no g_phi at all -- silent wrong features, not a
# crash, and only some of those sites guard on hasattr(model,'g_phi'). Dispatching
# on a model attribute instead is the real fix; until then the name sidesteps it.
# ---------------------------------------------------------------------------
@register_model("flyvis_cond_known_ode")
class FlyvisConductanceKnownODE(KnownODEBase):

    def __init__(self, aggr_type='add', config=None, device=None):
        super().__init__(aggr_type=aggr_type, config=config, device=device)
        tc = config.training
        self.cond_neuron_params = getattr(tc, "cond_neuron_params", "per_type")
        # PER TYPE BY DEFAULT, because the teacher's tau_i and V_i_rest hold exactly
        # 65 distinct values over 13,741 neurons -- one per cell type. Per-neuron is
        # 27,482 parameters describing 130. `type_index` is filled by
        # set_neuron_types(); until then the model is per-neuron so it is never
        # silently wrong, just larger.
        n_p = self.n_neurons
        self.register_buffer("type_index",
                             torch.arange(self.n_neurons, device=device, dtype=torch.long))
        if self.cond_neuron_params == "per_type":
            n_p = int(getattr(config.simulation, "n_neuron_types", 0) or self.n_neurons)
        self.raw_tau = nn.Parameter(
            torch.zeros(n_p, device=device, dtype=torch.float32),
            requires_grad=self.cond_neuron_params != "frozen")
        self.V_rest = nn.Parameter(
            torch.zeros(n_p, device=device, dtype=torch.float32),
            requires_grad=self.cond_neuron_params != "frozen")
        # Initialised straddling the voltage range so both driving forces start
        # with the right sign; flyvis voltages are O(1) about 0.
        self.cond_reversal_mode = getattr(tc, "cond_reversal_mode", "margin")
        self.delta_inh = float(getattr(tc, "cond_delta_inh", 0.4))
        self.delta_exc = float(getattr(tc, "cond_delta_exc", 1.0))
        _free = (getattr(tc, "cond_learn_reversal", True)
                 and self.cond_reversal_mode == "learned")
        self.E_exc = nn.Parameter(torch.tensor(1.0, device=device), requires_grad=_free)
        self.E_inh = nn.Parameter(torch.tensor(-1.0, device=device), requires_grad=_free)
        self._range_set = False
        self.W.requires_grad_(bool(getattr(tc, "cond_learn_edges", True)))
        n_w = self.n_edges + self.n_extra_null_edges
        self.register_buffer(
            "edge_is_inh", torch.zeros(n_w, dtype=torch.bool, device=device))
        self._sign_set = False

    def set_presynaptic_sign(self, w_signed):
        """Partition edges by presynaptic polarity from the connectome's own sign.

        w_signed: (E,) or (E,1) the ground-truth signed weights (ode_params.W).
        Only the SIGN is read -- no magnitude information reaches the model.
        """
        s = torch.as_tensor(w_signed).reshape(-1).to(self.edge_is_inh.device)
        n = min(s.numel(), self.edge_is_inh.numel())
        self.edge_is_inh[:n] = s[:n] < 0
        self._sign_set = True

    def _node_index(self, particle_id):
        """Neuron id -> parameter row. Identity per-neuron, cell type per-type."""
        return self.type_index[particle_id] if self.cond_neuron_params == "per_type" else particle_id

    def set_teacher_voltage_range(self, v_min, v_max):
        """Pin the reversals OUTSIDE the teacher's voltage range (cond_reversal_mode margin).

        E_exc = V_max + delta_exc * span, E_inh = V_min - delta_inh * span. Bracketing
        is then structural: V_i lies in [V_min, V_max] by definition, so (E_exc - V_i)
        > 0 and (E_inh - V_i) < 0 for every voltage the teacher ever visits, for any
        delta > 0. Nothing to penalise and nothing to check at runtime.
        """
        v_min, v_max = float(v_min), float(v_max)
        span = max(v_max - v_min, 1e-6)
        with torch.no_grad():
            self.E_exc.fill_(v_max + self.delta_exc * span)
            self.E_inh.fill_(v_min - self.delta_inh * span)
        self._range_set = True

    def set_neuron_types(self, type_list):
        """(N,) cell-type id per neuron, for cond_neuron_params: per_type."""
        t = torch.as_tensor(type_list).reshape(-1).long().to(self.type_index.device)
        self.type_index[: t.numel()] = t

    def set_teacher_neuron_params(self, tau, v_rest):
        """Pin tau/V_rest at the teacher's values (cond_neuron_params: frozen)."""
        with torch.no_grad():
            idx = self._node_index(torch.arange(self.n_neurons, device=self.W.device))
            tau_t = torch.as_tensor(tau).reshape(-1).to(self.raw_tau.device)
            vr_t = torch.as_tensor(v_rest).reshape(-1).to(self.V_rest.device)
            # softplus^-1 so softplus(raw_tau) reproduces tau exactly
            self.raw_tau.scatter_(0, idx, torch.log(torch.expm1(tau_t.clamp_min(1e-6))))
            self.V_rest.scatter_(0, idx, vr_t)

    def get_learned_tau(self):
        return F.softplus(self.raw_tau[self._node_index(
            torch.arange(self.n_neurons, device=self.raw_tau.device))]).detach()

    def get_learned_vrest(self):
        return self.V_rest[self._node_index(
            torch.arange(self.n_neurons, device=self.V_rest.device))].detach()

    def get_learned_conductance(self):
        """The non-negative conductance actually used, W^2, not the raw parameter."""
        return (self.W.detach() ** 2).squeeze(-1)

    def _activation(self, v):
        return F.relu(v)

    def _compute_messages(self, v, edge_index):
        """Overridden because the message needs v[dst].

        Every other KnownODEBase subclass reads v[src] only, so the base's
        `W * activation(v[src])` suffices for them. A conductance synapse's driving
        force is (E - v_i), i.e. it depends on the POSTsynaptic voltage, which is
        the one structural difference between this model and FlyvisKnownODE.
        """
        if self.cond_reversal_mode == "margin" and not self._range_set:
            raise RuntimeError(
                "flyvis_cond_known_ode: cond_reversal_mode 'margin' needs the teacher's "
                "voltage range; set_teacher_voltage_range() was never called, so the "
                "reversals would sit at their +-1 placeholders and could be crossed.")
        if not self._sign_set:
            raise RuntimeError(
                "flyvis_cond_known_ode: set_presynaptic_sign() was never called, so "
                "every edge would be treated as excitatory and E_inh would never "
                "receive a gradient. Pass ode_params.W to it after building the model.")
        src, dst = edge_index
        n_edges_batch = edge_index.shape[1]
        edge_W_idx = torch.arange(
            n_edges_batch, device=self.device) % (self.n_edges + self.n_extra_null_edges)

        g = self.W[edge_W_idx] ** 2                          # (E,1) conductance >= 0
        E = torch.where(self.edge_is_inh[edge_W_idx], self.E_inh, self.E_exc).unsqueeze(-1)
        edge_msg = g * self._activation(v[src]) * (E - v[dst])

        msg = torch.zeros(v.shape[0], 1, device=self.device, dtype=v.dtype)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)
        return msg

    def _update(self, v, msg, excitation, particle_id):
        idx = self._node_index(particle_id)
        tau = F.softplus(self.raw_tau[idx]).unsqueeze(-1)
        v_rest = self.V_rest[idx].unsqueeze(-1)
        return (-v + msg + excitation + v_rest) / tau
