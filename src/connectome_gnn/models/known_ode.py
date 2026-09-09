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
# connectome already carries: FlyVisCurrentODEParams.W is syn_strength * syn_count * sign,
# so sign(W_e) < 0 marks an inhibitory synapse. Under Dale's law that is a property
# of the presynaptic neuron, so the partition is anatomy rather than something to
# fit. Supplied by set_presynaptic_sign(); absent, every edge is treated as
# excitatory and E_inh never receives a gradient -- which is why the trainer must
# call it and why not calling it fails loudly rather than silently.
#
# NAMED `flyvis_conductance_known_ode`. It used to be `flyvis_cond_known_ode`
# purely to keep the substring `flyvis_conductance` out of it: seven sites
# dispatched on `'flyvis_conductance' in signal_model_name` and would all fire
# for this model, building 6-column [v_i,v_j,a_i,a_j] g_phi features for a
# model that has no g_phi at all -- silent wrong features, not a crash, and
# only some of them guard on hasattr(model,'g_phi').
#
# Those sites now use models.utils.is_conductance_gnn(), which excludes the
# known-ODE explicitly, so the dispatch carries the distinction instead of the
# name having to dodge it. `flyvis_cond_known_ode` remains a registry ALIAS for
# the archived runs.
# ---------------------------------------------------------------------------
@register_model(
    # `flyvis_cond_known_ode` kept as an ALIAS: the archived twin runs under
    # log/fly record it and are not in git.
    "flyvis_conductance_known_ode",
    "flyvis_cond_known_ode",
)
class FlyvisConductanceKnownODE(KnownODEBase):

    # Knobs that describe HOW TO GUESS a parameterisation the teacher cannot pin
    # down. They are meaningful only under distillation; on conductance-generated
    # data every one of them has an answer on disk instead. Named here so a spec
    # that sets one outside distillation is refused rather than ignored.
    STUDENT_ONLY_KEYS = ("student_reversal_mode", "student_reversal_dim",
                         "student_reversal_exc_global",
                         "student_E_exc_excursions_lo", "student_E_exc_excursions_hi",
                         "student_E_inh_excursions_lo", "student_E_inh_excursions_hi",
                         "student_neuron_params", "student_init",
                         "student_span_mode", "student_delta_inh",
                         "student_delta_exc", "student_learn_edges")

    # What the model uses when it is NOT distilling: the most general
    # parameterisation, assuming no structure at all. See _resolve_student_knobs.
    RECOVERY_DEFAULTS = dict(student_reversal_mode="learned",
                             student_reversal_dim="per_neuron",
                             # Both rows per neuron under recovery: the ion rig is a
                             # structural ASSUMPTION about which row varies, and on
                             # conductance data the true E is on disk, so asserting it
                             # could only hide a mismatch as a fit failure.
                             student_reversal_exc_global=False,
                             student_neuron_params="per_neuron",
                             student_init="default",
                             student_span_mode="extremes",
                             student_delta_inh=0.4,
                             student_delta_exc=1.0,
                             # Unused under 'learned', but _resolve_student_knobs
                             # returns every STUDENT_ONLY_KEY and __init__ reads them
                             # unconditionally, so they need values here too.
                             student_E_exc_excursions_lo=2.5,
                             student_E_exc_excursions_hi=2.5,
                             student_E_inh_excursions_lo=-1.5,
                             student_E_inh_excursions_hi=0.5,
                             student_learn_edges=True)

    @classmethod
    def _resolve_student_knobs(cls, tc):
        """Return the eight student_* settings this run should actually use.

        TWO REGIMES, AND THE CONFIG ONLY SPEAKS FOR ONE OF THEM.

        train_on_teacher True -- DISTILLATION. The teacher is current-based and has
        no (E - v_i) term, so E is unidentifiable: nothing in the data says whether
        the reversals are two numbers or 27,482, nor where they sit. The spec has to
        choose, and the six margin/learned x global/per_type/per_neuron runs are
        exactly that choice being swept. Read the knobs.

        train_on_teacher False -- RECOVERY on conductance-generated data. E, W, tau
        and V_rest all have true values in ode_params.pt, so there is nothing to
        choose and a knob could only assert structure the data already fixes -- and
        assert it WRONGLY the moment the generating student's granularity differs
        from the spec's, which would show up as a fit failure rather than as the
        misspecification it is. Use RECOVERY_DEFAULTS: the fully general
        parameterisation, E and tau/V_rest per neuron, reversals free, no margin
        pinning, no teacher closed-form init.

        Per-neuron under recovery is deliberately the LOOSEST option, not the one
        matching the truth. The generating student was margin/global, so the true E
        is two numbers; fitting 27,482 and watching them collapse onto two is a
        result the E_ij scatter shows directly, whereas telling the model 'global'
        would hand it the answer.

        A spec that sets a student_* key with train_on_teacher False raises: the
        alternative is silently overriding it, which is how a run ends up not being
        the experiment its config describes.
        """
        if getattr(tc, "train_on_teacher", False):
            return {k: getattr(tc, k) for k in cls.STUDENT_ONLY_KEYS}
        explicit = getattr(tc, "model_fields_set", set())
        stray = sorted(set(cls.STUDENT_ONLY_KEYS) & set(explicit))
        if stray:
            raise ValueError(
                f"{stray} set with train_on_teacher False. Those knobs choose a "
                "parameterisation for reversals the teacher cannot identify; on "
                "conductance-generated data the true values are in ode_params.pt "
                "and the model uses the general per-neuron form instead. Drop them "
                "from the spec, or set train_on_teacher True if this really is a "
                "distillation run.")
        return dict(cls.RECOVERY_DEFAULTS)

    def __init__(self, aggr_type='add', config=None, device=None):
        super().__init__(aggr_type=aggr_type, config=config, device=device)
        tc = config.training
        k = self._resolve_student_knobs(tc)
        self.student_neuron_params = k["student_neuron_params"]
        # PER TYPE UNDER DISTILLATION, because the teacher's tau_i and V_i_rest hold
        # exactly 65 distinct values over 13,741 neurons -- one per cell type.
        # Per-neuron is 27,482 parameters describing 130. `type_index` is filled by
        # set_neuron_types(); until then the model is per-neuron so it is never
        # silently wrong, just larger.
        n_p = self.n_neurons
        self.register_buffer("type_index",
                             torch.arange(self.n_neurons, device=device, dtype=torch.long))
        if self.student_neuron_params == "per_type":
            n_p = int(getattr(config.simulation, "n_neuron_types", 0) or self.n_neurons)
        self.raw_tau = nn.Parameter(
            torch.zeros(n_p, device=device, dtype=torch.float32),
            requires_grad=self.student_neuron_params != "frozen")
        self.V_rest = nn.Parameter(
            torch.zeros(n_p, device=device, dtype=torch.float32),
            requires_grad=self.student_neuron_params != "frozen")
        # Initialised straddling the voltage range so both driving forces start
        # with the right sign; flyvis voltages are O(1) about 0.
        self.student_reversal_mode = k["student_reversal_mode"]
        self.student_span_mode = k["student_span_mode"]
        self.student_init = k["student_init"]
        self.delta_inh = float(k["student_delta_inh"])
        self.delta_exc = float(k["student_delta_exc"])
        _free = self.student_reversal_mode == "learned"
        self.student_reversal_dim = k["student_reversal_dim"]
        n_rev = {"global": 1,
                 "per_type": int(getattr(config.simulation, "n_neuron_types", 0) or self.n_neurons),
                 "per_neuron": self.n_neurons}[self.student_reversal_dim]
        # THE TWO ROWS CAN HAVE DIFFERENT ROW COUNTS. student_reversal_exc_global
        # collapses the excitatory (cation) row to a single value and leaves the
        # inhibitory (chloride) row at student_reversal_dim's granularity -- see the
        # config comment for why that asymmetry is the biological one. Everything
        # downstream goes through _rev_index_exc / _rev_index_inh, never through a
        # shared row index, so the two shapes can never be confused for each other.
        self.student_reversal_exc_global = bool(k["student_reversal_exc_global"])
        n_rev_exc = 1 if self.student_reversal_exc_global else n_rev

        # PHYSIOLOGICAL MODE STORES E DERIVED, NOT FREE. E_exc/E_inh stop being
        # parameters and become BUFFERS holding the materialised reversals, while the
        # free variables are raw_E_exc/raw_E_inh, squashed into the configured band of
        # excursions from rest. Keeping the buffers named E_exc/E_inh is deliberate:
        # FlyVisConductanceODEParams.from_twin_checkpoint, compute_reversal_metrics and
        # extract_recovered_params all read `sd["E_exc"]` / `sd["E_inh"]` and neither
        # knows nor should know how the numbers were produced. `state_dict` below
        # refreshes them so a checkpoint is never stale.
        self.physiological = self.student_reversal_mode == "physiological"
        self.exc_lo = float(k["student_E_exc_excursions_lo"])
        self.exc_hi = float(k["student_E_exc_excursions_hi"])
        self.inh_lo = float(k["student_E_inh_excursions_lo"])
        self.inh_hi = float(k["student_E_inh_excursions_hi"])
        if self.physiological:
            if self.exc_hi < self.exc_lo or self.inh_hi < self.inh_lo:
                raise ValueError(
                    "student_E_*_excursions_hi must be >= its _lo; got exc "
                    f"[{self.exc_lo}, {self.exc_hi}] inh [{self.inh_lo}, {self.inh_hi}]")
            self.register_buffer("E_exc", torch.ones(n_rev_exc, device=device))
            self.register_buffer("E_inh", -torch.ones(n_rev, device=device))
            # requires_grad False when the band has zero width: a reversal biology pins
            # exactly, like the cation one at 0 mV, carries no free parameter at all
            # rather than a parameter the sigmoid then ignores.
            self.raw_E_exc = nn.Parameter(torch.zeros(n_rev_exc, device=device),
                                          requires_grad=self.exc_hi > self.exc_lo)
            self.raw_E_inh = nn.Parameter(torch.zeros(n_rev, device=device),
                                          requires_grad=self.inh_hi > self.inh_lo)
            # The anchor: rest per row, and the one excursion width both rows are
            # measured in. Buffers so `-o test` restores them without the setter.
            self.register_buffer("_rest_exc", torch.zeros(n_rev_exc, device=device))
            self.register_buffer("_rest_inh", torch.zeros(n_rev, device=device))
            self.register_buffer("_excursion", torch.ones(1, device=device))
        else:
            self.E_exc = nn.Parameter(torch.ones(n_rev_exc, device=device), requires_grad=_free)
            self.E_inh = nn.Parameter(-torch.ones(n_rev, device=device), requires_grad=_free)
        self.register_buffer("_range_set_b", torch.zeros(1, dtype=torch.bool, device=device))
        self._range_set = False
        self.W.requires_grad_(bool(k["student_learn_edges"]))
        # WHICH RECOVERY PATH SCORES THIS RUN, and it differs by regime, which is
        # why it is set per instance rather than as a class attribute.
        #
        # 'linear' means W, tau and V_rest are direct parameters that
        # plot_training_linear can read off and scatter against ode_params. That is
        # true of this model always -- but only MEANINGFUL under recovery, where the
        # dataset was made by a conductance model and ode_params.W really is the
        # conductance this one is learning.
        #
        # Under distillation the teacher is current-based: ode_params.W is a SIGNED
        # current weight and this model learns a non-negative conductance, so an R2
        # between them is a confident number about nothing. Leaving the default
        # 'gnn' there routes to plot_training_gnn, which early-returns NaN for a
        # model with no embedding and no g_phi -- the honest answer, and the one the
        # config comment on train_on_teacher already promises ("R2_W is meaningless
        # here ... the acceptance test is the ROLLOUT").
        self.MODEL_FAMILY = "linear" if not getattr(tc, "train_on_teacher", False) else "gnn"
        n_w = self.n_edges + self.n_extra_null_edges
        self.register_buffer(
            "edge_is_inh", torch.zeros(n_w, dtype=torch.bool, device=device))
        # BUFFERS, NOT PYTHON BOOLS, so load_state_dict restores them. edge_is_inh,
        # type_index, E_exc and E_inh are all saved in the checkpoint, but plain
        # attributes are not -- so at `-o test` the model came back fully configured
        # with both flags False and the guard raised. graph_tester builds the model
        # and loads a checkpoint; it never calls the setters, and it should not have
        # to, because everything they set is already in the state dict.
        # TWO REPRESENTATIONS OF THE SAME FACT, and both are needed. The BUFFER
        # persists through the checkpoint, so `-o test` gets a configured model
        # without calling the setters. The PYTHON BOOL is what forward() tests:
        # `bool(tensor)` inside a torch.compile'd forward is data-dependent control
        # flow and dynamo refuses it, which killed all six runs at iteration 0.
        # _load_from_state_dict below syncs the bool from the buffer.
        self.register_buffer("_sign_set_b", torch.zeros(1, dtype=torch.bool, device=device))
        self._sign_set = False

    def set_presynaptic_sign(self, w_signed):
        """Partition edges by presynaptic polarity.

        w_signed: (E,) or (E,1), either
          - the ground-truth SIGNED weights (ode_params.W on a current-generated
            dataset), from which only the SIGN is read -- no magnitude
            information reaches the model; or
          - a BOOL mask, already the polarity (ode_params.edge_is_inh on a
            conductance-generated dataset).

        The bool branch is not a convenience. On conductance-generated data
        ode_params.W is the CONDUCTANCE, which is non-negative by construction,
        so `W < 0` would mark every one of the 434,112 edges excitatory, E_inh
        would never receive a gradient, and the run would look healthy while
        fitting a purely excitatory network. The polarity there lives in
        edge_is_inh, which is a separate field precisely because the sign left W.
        """
        s = torch.as_tensor(w_signed).reshape(-1).to(self.edge_is_inh.device)
        n = min(s.numel(), self.edge_is_inh.numel())
        self.edge_is_inh[:n] = s[:n] if s.dtype == torch.bool else (s[:n] < 0)
        self._sign_set_b.fill_(True); self._sign_set = True

    def _node_index(self, particle_id):
        """Neuron id -> parameter row. Identity per-neuron, cell type per-type."""
        return self.type_index[particle_id] if self.student_neuron_params == "per_type" else particle_id

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Restore the python guard flags from their persisted buffers."""
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._sign_set = bool(self._sign_set_b)
        self._range_set = bool(self._range_set_b)

    def _rev_index(self, neuron_ids):
        """Postsynaptic neuron id -> reversal row. E belongs to the POSTsynaptic cell.

        MODULO n_neurons because _batch_frames replicates the graph B times with an
        offset, so dst spans [0, N*B) while type_index and the reversals are declared
        once per network. Without it a batched run indexes out of bounds and dies as
        a bare CUDA device-side assert with no line number.
        """
        ids = neuron_ids % self.n_neurons
        if self.student_reversal_dim == "global":
            return torch.zeros_like(ids)
        if self.student_reversal_dim == "per_type":
            return self.type_index[ids]
        return ids

    @staticmethod
    def _in_band(raw, lo, hi):
        """Squash a free variable into [lo, hi] excursions. Constant when lo == hi.

        A sigmoid rather than a clamp because the bound then holds at every gradient
        step by algebra instead of by projection, and because a clamped parameter
        sitting on its boundary receives zero gradient and never leaves it.
        """
        if hi <= lo:
            return torch.full_like(raw, lo)
        return lo + (hi - lo) * torch.sigmoid(raw)

    def _materialise_reversals(self):
        """(E_exc, E_inh) at their own row counts, DIFFERENTIABLE, in voltage units.

        Under 'margin' and 'learned' the parameters already are the reversals. Under
        'physiological' they are excursions from rest:

            E(r) = V_rest(r) + (lo + (hi - lo) * sigmoid(raw_r)) * excursion

        with `excursion` the teacher's voltage width at student_span_mode's
        granularity and `V_rest(r)` the teacher's resting potential reduced onto that
        row. Both come from set_physiological_anchor; see the config block for why
        multiples of the excursion are the only scale-free way to state a reversal on
        a model whose voltage carries no physical unit.
        """
        if not self.physiological:
            return self.E_exc, self.E_inh
        return (self._rest_exc + self._in_band(self.raw_E_exc, self.exc_lo, self.exc_hi)
                * self._excursion,
                self._rest_inh + self._in_band(self.raw_E_inh, self.inh_lo, self.inh_hi)
                * self._excursion)

    def set_physiological_anchor(self, v_rest, excursion):
        """Pin what 'rest' and 'one excursion' mean (student_reversal_mode physiological).

        v_rest: (N,) the TEACHER's resting potential per neuron, reduced here onto each
            row by mean, so a per-type chloride reversal is anchored on the mean rest of
            that cell type and a global cation reversal on the network's mean rest.
        excursion: scalar, the teacher's voltage width -- the central-99% width under
            student_span_mode p99, the raw range under 'extremes'. ONE number for both
            rows, because it is the network's operating scale and not a property of any
            cell: a near-silent cell type must not get a degenerate unit, which is
            exactly how the per-type margin rig drove Tm30's driving force to 0.006.
        """
        dev = self.E_exc.device
        vr = torch.as_tensor(v_rest, dtype=torch.float32, device=dev).reshape(-1)
        with torch.no_grad():
            self._excursion.fill_(max(float(excursion), 1e-3))
            for buf, idx_fn in ((self._rest_exc, self._rev_index_exc),
                                (self._rest_inh, self._rev_index_inh)):
                if vr.numel() != self.n_neurons:
                    buf.fill_(float(vr.mean()))
                    continue
                idx = idx_fn(torch.arange(self.n_neurons, device=dev))
                n = buf.numel()
                sums = torch.zeros(n, device=dev).index_add_(0, idx, vr)
                cnts = torch.zeros(n, device=dev).index_add_(0, idx, torch.ones_like(vr))
                buf.copy_(sums / cnts.clamp_min(1))
            exc, inh = self._materialise_reversals()
            self.E_exc.copy_(exc); self.E_inh.copy_(inh)
        self._range_set_b.fill_(True); self._range_set = True

    def state_dict(self, *args, **kwargs):
        """Refresh the E_exc/E_inh buffers before every save.

        Under 'physiological' they are derived from raw_E_* and only recomputed inside
        forward, so a checkpoint written between steps would otherwise carry whatever
        the last materialisation left. Every downstream reader --
        from_twin_checkpoint, compute_reversal_metrics, extract_recovered_params --
        goes through the state dict, so refreshing here covers all of them at once
        rather than asking each save site to remember.
        """
        if getattr(self, "physiological", False):
            with torch.no_grad():
                exc, inh = self._materialise_reversals()
                self.E_exc.copy_(exc); self.E_inh.copy_(inh)
        return super().state_dict(*args, **kwargs)

    def _rev_index_inh(self, neuron_ids):
        """Row of E_inh. Always at student_reversal_dim's granularity -- the chloride
        reversal is the one the ion rig leaves free to vary across cells."""
        return self._rev_index(neuron_ids)

    def _rev_index_exc(self, neuron_ids):
        """Row of E_exc. Row 0 for every neuron under student_reversal_exc_global.

        Separate from _rev_index_inh because under the ion rig E_exc holds ONE row
        while E_inh holds 65 or 13,741; indexing the first with the second's rows
        would be an out-of-bounds read on GPU, i.e. a bare device-side assert.
        """
        if self.student_reversal_exc_global:
            return torch.zeros_like(neuron_ids)
        return self._rev_index(neuron_ids)

    def set_teacher_voltage_range(self, v_min, v_max, v_lo=None, v_hi=None):
        """Pin the reversals OUTSIDE the teacher's voltage range (student_reversal_mode margin).

        E_exc = V_max + delta_exc * span, E_inh = V_min - delta_inh * span. Bracketing
        is then structural: V_i lies in [V_min, V_max] by definition, so (E_exc - V_i)
        > 0 and (E_inh - V_i) < 0 for every voltage the teacher ever visits, for any
        delta > 0. Nothing to penalise and nothing to check at runtime.
        """
        # Accepts scalars or (N,) per-neuron extremes; reduced to whatever
        # granularity student_reversal_dim asks for. 1e-3 span floor matching PR #46's
        # derive_conductance_twin: at 1e-6 a degenerate recording puts the reversals
        # a millionth outside the range, which brackets in principle but leaves no
        # usable driving force.
        dev = self.E_exc.device
        lo = torch.as_tensor(v_min, dtype=torch.float32, device=dev).reshape(-1)
        hi = torch.as_tensor(v_max, dtype=torch.float32, device=dev).reshape(-1)
        # BRACKET from lo/hi, MEASURE delta in v_lo..v_hi. Separating the two is the
        # whole of student_span_mode: the bracket must come from the extremes or the
        # sign guarantee fails, but delta's unit does not have to, and on a
        # heavy-tailed voltage the extremes make it ~3.4x too large.
        slo = lo if v_lo is None else torch.as_tensor(
            v_lo, dtype=torch.float32, device=dev).reshape(-1)
        shi = hi if v_hi is None else torch.as_tensor(
            v_hi, dtype=torch.float32, device=dev).reshape(-1)
        # ROW BY ROW, because the two rows need not have the same row count: under
        # student_reversal_exc_global the excitatory row is a single value while the
        # inhibitory one is per type or per neuron. Each row reduces the teacher's
        # extremes onto ITS OWN rows, so each brackets exactly the voltages the cells
        # sharing that reversal actually visit.
        with torch.no_grad():
            for param, idx_fn, delta, is_exc in (
                    (self.E_exc, self._rev_index_exc, self.delta_exc, True),
                    (self.E_inh, self._rev_index_inh, self.delta_inh, False)):
                if param.numel() == 1 or lo.numel() == 1:
                    span = (shi.max() - slo.min()).clamp_min(1e-3)
                    param.fill_(float(hi.max() + delta * span) if is_exc
                                else float(lo.min() - delta * span))
                    continue
                idx = idx_fn(torch.arange(lo.numel(), device=dev))
                n = param.numel()
                lo_r = torch.full((n,), float("inf"), device=dev).scatter_reduce(
                    0, idx, lo, reduce="amin", include_self=True)
                hi_r = torch.full((n,), float("-inf"), device=dev).scatter_reduce(
                    0, idx, hi, reduce="amax", include_self=True)
                slo_r = torch.full((n,), float("inf"), device=dev).scatter_reduce(
                    0, idx, slo, reduce="amin", include_self=True)
                shi_r = torch.full((n,), float("-inf"), device=dev).scatter_reduce(
                    0, idx, shi, reduce="amax", include_self=True)
                empty = ~torch.isfinite(lo_r)          # rows no neuron maps to
                lo_r[empty] = lo.min(); hi_r[empty] = hi.max()
                slo_r[empty] = slo.min(); shi_r[empty] = shi.max()
                span = (shi_r - slo_r).clamp_min(1e-3)
                param.copy_(hi_r + delta * span if is_exc else lo_r - delta * span)
        self._range_set_b.fill_(True); self._range_set = True

    def init_from_teacher(self, w_signed, edge_index, v_mean_per_neuron):
        """Stage-1 closed form: W^2 <- alpha_curr / (E - Vbar_ti). See student_init.

        Requires set_presynaptic_sign() and set_teacher_voltage_range()
        to have run, since it needs the per-edge polarity and the reversals. The
        quotient is positive by construction -- alpha_curr and (E - Vbar) carry the
        same sign -- so the sqrt is real without a clamp doing any work; the clamp is
        only there for an edge whose teacher weight is exactly zero.
        """
        dev = self.W.device
        w = torch.as_tensor(w_signed).reshape(-1).to(dev)
        dst = torch.as_tensor(edge_index[1]).reshape(-1).long().to(dev)
        vbar = torch.as_tensor(v_mean_per_neuron).reshape(-1).to(dev)
        n = min(w.numel(), self.W.shape[0])
        r_exc = self._rev_index_exc(dst[:n])
        r_inh = self._rev_index_inh(dst[:n])

        # VBAR AT E'S OWN GRANULARITY. The margin brackets whatever range E was built
        # from; a Vbar reduced differently can fall outside it and flip the sign of
        # (E - Vbar). Measured: per-neuron reversals against a per-CELL-TYPE Vbar gave
        # 512 of 434,112 edges a negative conductance. Reducing Vbar onto the same
        # rows removes the mismatch by construction -- and ONCE PER ROW, since the two
        # rows can now be at different granularities.
        def _vbar_rows(idx_fn, nrow):
            """Vbar averaged onto one reversal's rows, or None when that row is already
            per neuron and the per-neuron Vbar can be used directly."""
            if nrow == self.n_neurons or vbar.numel() != self.n_neurons:
                return None
            ridx = idx_fn(torch.arange(self.n_neurons, device=dev))
            sums = torch.zeros(nrow, device=dev).index_add_(0, ridx, vbar)
            cnts = torch.zeros(nrow, device=dev).index_add_(0, ridx, torch.ones_like(vbar))
            return sums / cnts.clamp_min(1)

        vbar_pn = vbar[dst[:n] % self.n_neurons]
        E_exc_row, E_inh_row = self._materialise_reversals()
        rows_exc = _vbar_rows(self._rev_index_exc, E_exc_row.numel())
        rows_inh = _vbar_rows(self._rev_index_inh, E_inh_row.numel())
        vbar_e = torch.where(self.edge_is_inh[:n],
                             vbar_pn if rows_inh is None else rows_inh[r_inh],
                             vbar_pn if rows_exc is None else rows_exc[r_exc])
        E = torch.where(self.edge_is_inh[:n], E_inh_row[r_inh], E_exc_row[r_exc])
        alpha = (w[:n] / (E - vbar_e))
        neg = int((alpha < 0).sum())
        if neg and not self.physiological:
            raise RuntimeError(
                f"student_init teacher_closed_form: {neg} of {n} edges gave a NEGATIVE "
                "conductance, which means E - Vbar does not carry the connectome sign "
                "on them -- the reversals are not bracketing the teacher's range. "
                "Check student_reversal_mode and the deltas.")
        if neg:
            # A NEGATIVE alpha IS EXPECTED HERE, AND IT IS THE MODEL MISMATCH ITSELF,
            # not a misconfiguration. A physiological chloride reversal sits INSIDE the
            # operating range, so a cell whose mean voltage is below its own E_Cl is
            # DEPOLARISED at rest by its chloride synapses -- real depolarising GABA.
            # The current-based teacher recorded those same synapses as inhibitory, and
            # no non-negative conductance can reproduce an inhibitory current through a
            # positive driving force. So the twin cannot match the teacher at rest on
            # these edges; it can only match it away from rest, and the derivative loss
            # is what has to settle that. Initialising on |alpha| matches the MAGNITUDE
            # of the teacher's current and lets the sign be whatever biology says.
            frac = 100.0 * neg / max(n, 1)
            print(f"\033[93mstudent_init teacher_closed_form: {neg} of {n} edges "
                  f"({frac:.2f}%) have (E - Vbar) against the connectome sign -- "
                  f"initialising on |alpha|. Expected under student_reversal_mode "
                  f"'physiological': these are the synapses whose reversal lies on the "
                  f"far side of the postsynaptic cell's mean voltage.\033[0m")
        with torch.no_grad():
            self.W[:n, 0] = alpha.abs().sqrt()

    def set_neuron_types(self, type_list):
        """(N,) cell-type id per neuron, for student_neuron_params: per_type."""
        t = torch.as_tensor(type_list).reshape(-1).long().to(self.type_index.device)
        self.type_index[: t.numel()] = t

    def set_teacher_neuron_params(self, tau, v_rest):
        """Pin tau/V_rest at the teacher's values (student_neuron_params: frozen)."""
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

    def get_learned_reversals(self):
        """(E_exc, E_inh), both MATERIALISED to (n_neurons,) whatever the granularity.

        The parameters are stored at whichever granularity
        `training.student_reversal_dim` asked for -- 1 row (global), one per
        cell type (per_type) or one per neuron (per_neuron), and the excitatory row
        collapsed to 1 on its own under `student_reversal_exc_global` -- and
        `forward` never expands them: `_rev_index_exc` / `_rev_index_inh` map a
        postsynaptic neuron id straight to its parameter row and index lazily.

        Comparison against ground truth needs the opposite: one value per neuron
        regardless of how few free parameters produced it, so that a global fit, an
        ion-rig fit and a per-neuron fit are all read on the same axis. Expanding
        each row through its own index is exactly that, and it is the same rule the
        generator applies on the ground-truth side
        (`FlyVisConductanceODEParams._per_neuron`, which broadcasts a scalar or
        indexes a 65-row array through `type_index`).

        THIS IS THE CONTRACT THE EXTRACTION RELIES ON. `extract_recovered_params`
        pairs `get_learned_reversal_per_edge` against
        `ode_params.reversal_per_edge()`, both (n_edges,), and
        `compute_reversal_metrics` pairs these two against `ode_params.E_exc/E_inh`,
        both (n_neurons,). Neither ever sees a row count, so a new granularity needs
        no change on the extraction side -- provided the expansion happens here.
        """
        ids = torch.arange(self.n_neurons, device=self.E_exc.device)
        E_exc, E_inh = self._materialise_reversals()
        return (E_exc[self._rev_index_exc(ids)].detach(),
                E_inh[self._rev_index_inh(ids)].detach())

    def get_learned_reversal_per_edge(self, edge_index=None):
        """(n_edges,) the reversal E_ij each edge drives toward, learned side.

        Same construction as the ground truth's
        `FlyVisConductanceODEParams.reversal_per_edge`: `edge_is_inh` is the
        PREsynaptic cell's Dale sign while E is indexed by the POSTsynaptic cell,
        so the destination row picks which of that neuron's two reversals applies.

        `edge_index` may be omitted only if the model was given one; pass the
        dataset's (2, n_edges) tensor. Only the first `n_edges + n_extra_null_edges`
        columns are used, so a batched edge_index (the graph replicated B times
        with a neuron-id offset) is handled by taking its first replica.
        """
        if edge_index is None:
            raise ValueError("get_learned_reversal_per_edge needs the dataset edge_index")
        n_w = self.n_edges + self.n_extra_null_edges
        dst = edge_index[1][:n_w]
        E_exc, E_inh = self.get_learned_reversals()
        return torch.where(self.edge_is_inh[:dst.numel()],
                           E_inh[dst % self.n_neurons], E_exc[dst % self.n_neurons])

    def _activation(self, v):
        return F.relu(v)

    def _compute_messages(self, v, edge_index):
        """Overridden because the message needs v[dst].

        Every other KnownODEBase subclass reads v[src] only, so the base's
        `W * activation(v[src])` suffices for them. A conductance synapse's driving
        force is (E - v_i), i.e. it depends on the POSTsynaptic voltage, which is
        the one structural difference between this model and FlyvisKnownODE.
        """
        if not self._range_set:
            raise RuntimeError(
                "flyvis_conductance_known_ode: set_teacher_voltage_range() was never called, so "
                "the reversals sit at their +-1 placeholders. Under 'margin' they would "
                "not bracket the teacher's range; under 'learned' the closed-form init "
                "would divide by an (E - Vbar) of the wrong sign. Both modes need it -- "
                "'learned' STARTS from the margin and fits from there.")
        if not self._sign_set:
            raise RuntimeError(
                "flyvis_conductance_known_ode: set_presynaptic_sign() was never called, so "
                "every edge would be treated as excitatory and E_inh would never "
                "receive a gradient. Pass ode_params.W to it after building the model.")
        src, dst = edge_index
        n_edges_batch = edge_index.shape[1]
        edge_W_idx = torch.arange(
            n_edges_batch, device=self.device) % (self.n_edges + self.n_extra_null_edges)

        g = self.W[edge_W_idx] ** 2                          # (E,1) conductance >= 0
        E_exc, E_inh = self._materialise_reversals()
        E = torch.where(self.edge_is_inh[edge_W_idx],
                        E_inh[self._rev_index_inh(dst)],
                        E_exc[self._rev_index_exc(dst)]).unsqueeze(-1)
        edge_msg = g * self._activation(v[src]) * (E - v[dst])

        msg = torch.zeros(v.shape[0], 1, device=self.device, dtype=v.dtype)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)
        return msg

    def _update(self, v, msg, excitation, particle_id):
        idx = self._node_index(particle_id)
        tau = F.softplus(self.raw_tau[idx]).unsqueeze(-1)
        v_rest = self.V_rest[idx].unsqueeze(-1)
        return (-v + msg + excitation + v_rest) / tau
