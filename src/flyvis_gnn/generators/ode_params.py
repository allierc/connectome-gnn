"""ODE parameter classes and registry for flyvis-gnn.

Maps config signal_model_name strings to ODE parameter dataclasses.
Each ODE_params_class knows how to construct itself from a source,
save/load to disk, and expose its fields by name.

Usage:
    @register_ode_params("flyvis_A", "flyvis_B")
    class FlyVisODEParams(ODEParamsBase):
        ...

    ODE_params_class = get_ode_params_class("flyvis_A")
    p = ODE_params_class.from_flyvis_network(net, device=device)
    p.save(folder)
    p = ODE_params_class.load(folder)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from typing import Any

import numpy as np
import torch

from flyvis_gnn.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ODE_PARAMS_REGISTRY: dict[str, type] = {}


def register_ode_params(*names: str):
    """Class decorator that registers an ODE params class under config names."""
    def decorator(cls):
        for name in names:
            if name in _ODE_PARAMS_REGISTRY:
                raise ValueError(
                    f"ODE params name '{name}' already registered to "
                    f"{_ODE_PARAMS_REGISTRY[name].__name__}"
                )
            _ODE_PARAMS_REGISTRY[name] = cls
        return cls
    return decorator


def get_ode_params_class(name: str) -> type:
    """Look up ODE params class by config signal_model_name."""
    if name not in _ODE_PARAMS_REGISTRY:
        available = sorted(_ODE_PARAMS_REGISTRY.keys())
        raise KeyError(
            f"Unknown ODE params '{name}'. Available: {available}"
        )
    return _ODE_PARAMS_REGISTRY[name]


def list_ode_params() -> list[str]:
    """Return sorted list of all registered ODE params names."""
    return sorted(_ODE_PARAMS_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class ODEParamsBase:
    """Base class for ODE parameter dataclasses.

    Provides to(), clone(), save(), load(), and dict-style access
    for backward compatibility (p["tau_i"] still works).
    """

    def __getitem__(self, key: str) -> Any:
        """Dict-style access for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        """Dict-style assignment for backward compatibility."""
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Support `"key" in params`."""
        return hasattr(self, key) and getattr(self, key) is not None

    def __iter__(self):
        """Iterate over field names (for `for key in params`)."""
        return iter(f.name for f in dc_fields(self))

    def to(self, device: torch.device) -> ODEParamsBase:
        """Move all tensor fields to device."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                setattr(self, f.name, val.to(device))
        return self

    def clone(self) -> ODEParamsBase:
        """Deep clone all tensor fields."""
        kwargs = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            kwargs[f.name] = val.clone() if isinstance(val, torch.Tensor) else val
        return self.__class__(**kwargs)

    def save(self, folder: str):
        """Save all fields as a single ode_params.pt dict."""
        os.makedirs(folder, exist_ok=True)
        state = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                state[f.name] = val.cpu()
            else:
                state[f.name] = val
        torch.save(state, os.path.join(folder, "ode_params.pt"))

    @classmethod
    def load(cls, folder: str, device: torch.device | str = "cpu"):
        """Load from ode_params.pt, or fall back to legacy individual .pt files."""
        unified_path = os.path.join(folder, "ode_params.pt")
        if os.path.exists(unified_path):
            state = torch.load(unified_path, map_location=device, weights_only=True)
            return cls(**state)
        return cls._load_legacy(folder, device)

    @classmethod
    def _load_legacy(cls, folder: str, device: torch.device | str = "cpu"):
        """Override in subclass to support legacy per-file loading."""
        raise FileNotFoundError(
            f"No ode_params.pt found at {folder} and no legacy loader defined "
            f"for {cls.__name__}"
        )


# ---------------------------------------------------------------------------
# FlyVis graded-voltage model params
# ---------------------------------------------------------------------------

@register_ode_params(
    "flyvis_A", "flyvis_B", "flyvis_C", "flyvis_D",
    "flyvis_A_multiple_ReLU", "flyvis_B_multiple_ReLU", "flyvis_C_multiple_ReLU",
    "flyvis_A_tanh", "flyvis_B_tanh", "flyvis_C_tanh",
    "flyvis_A_NULL", "flyvis_B_NULL", "flyvis_C_NULL",
    "flyvis_linear", "flyvis_linear_tanh",
)
@dataclass
class FlyVisODEParams(ODEParamsBase):
    """Parameters for the graded-voltage FlyVis ODE.

    Node params (indexed by neuron — one value per node in the graph):
        tau_i:     (N,) time constants
        V_i_rest:  (N,) resting potentials

    Edge params:
        edge_index: (2, E) source/destination indices
        w:          (E,) effective synaptic weights
    """
    tau_i: torch.Tensor = None       # (N,)
    V_i_rest: torch.Tensor = None    # (N,)
    edge_index: torch.Tensor = None  # (2, E)
    W: torch.Tensor = None           # (E,) effective synaptic weights

    @classmethod
    def from_flyvis_network(cls, net, device: torch.device | str = "cpu"):
        """Construct from a flyvis Network object."""
        params = net._param_api()
        tau_i = params.nodes.time_const
        V_i_rest = params.nodes.bias
        W = params.edges.syn_strength * params.edges.syn_count * params.edges.sign
        edge_index = torch.stack([
            torch.tensor(net.connectome.edges.source_index[:]),
            torch.tensor(net.connectome.edges.target_index[:]),
        ], dim=0)
        return cls(
            tau_i=tau_i.to(device),
            V_i_rest=V_i_rest.to(device),
            edge_index=edge_index.to(device),
            W=W.to(device),
        )

    @classmethod
    def _load_legacy(cls, folder: str, device: torch.device | str = "cpu"):
        """Load from legacy individual .pt files (taus.pt, V_i_rest.pt, etc.)."""
        def _load(name):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return torch.load(path, map_location=device, weights_only=True)
            return None

        tau_i = _load("taus.pt")
        V_i_rest = _load("V_i_rest.pt")
        W = _load("weights.pt")
        edge_index = _load("edge_index.pt")

        if tau_i is None and V_i_rest is None and W is None and edge_index is None:
            raise FileNotFoundError(
                f"No ode_params.pt or legacy .pt files found at {folder}"
            )

        logger.info(f"loaded legacy ODE params from {folder}")
        return cls(tau_i=tau_i, V_i_rest=V_i_rest, edge_index=edge_index, W=W)


# ---------------------------------------------------------------------------
# FlyVis AdEx spiking model params
# ---------------------------------------------------------------------------

# Default values from Zerlaut et al. 2018 (AutoMind ADEX_NEURON_DEFAULTS_ZERLAUT).
# Units: mV, pF, nS, pA, ms, Hz.  Stored as dimensionless floats in those units.
ADEX_DEFAULTS = dict(
    # Membrane
    C=200.0,             # pF  — membrane capacitance
    g_L=10.0,            # nS  — leak conductance
    v_rest=-65.0,        # mV  — resting (leak reversal) potential
    v_thresh=-50.0,      # mV  — spike initiation threshold (exp onset)
    delta_T=2.0,         # mV  — exponential nonlinearity sharpness
    v_cut=0.0,           # mV  — hard spike cutoff for detection
    v_reset=-65.0,       # mV  — post-spike reset voltage
    t_refrac=5.0,        # ms  — absolute refractory period
    # Adaptation
    a=4.0,               # nS  — subthreshold adaptation coupling
    b=20.0,              # pA  — spike-triggered adaptation increment
    tau_w=500.0,         # ms  — adaptation time constant
    # Synaptic (COBA)
    E_ge=0.0,            # mV  — excitatory reversal potential
    E_gi=-80.0,          # mV  — inhibitory reversal potential
    Q_ge=1.0,            # nS  — excitatory quantal conductance
    Q_gi=5.0,            # nS  — inhibitory quantal conductance
    tau_ge=5.0,          # ms  — excitatory conductance decay
    tau_gi=5.0,          # ms  — inhibitory conductance decay
    # Synaptic (CUBA) — no defaults from Zerlaut, set to 0 as placeholder
    J_exc=0.0,           # mV  — excitatory spike kick
    J_inh=0.0,           # mV  — inhibitory spike kick
    # External input
    I_bias=0.0,          # pA  — constant bias current
    stim_scale=1.0,      # pA per unit stimulus — converts visual input to current
    # Initial conditions
    v_0_mean=0.0,        # mV  — mean offset from v_rest for initial v
    v_0_std=4.0,         # mV  — std of initial v perturbation
)


@register_ode_params("flyvis_adex_coba", "flyvis_adex_cuba")
@dataclass
class FlyVisAdExODEParams(ODEParamsBase):
    """Parameters for the AdEx spiking FlyVis ODE.

    Per-neuron static params (indexed by neuron, one value per node):
        Membrane: C, g_L, v_rest, v_thresh, delta_T, v_cut, v_reset, t_refrac
        Adaptation: a, b, tau_w

    Per-neuron synaptic params:
        COBA: E_ge, E_gi, Q_ge, Q_gi, tau_ge, tau_gi
        CUBA: J_exc, J_inh

    Per-neuron external input:
        I_bias, stim_scale

    Network topology:
        edge_index: (2, E) source/destination indices
        is_excitatory: (N,) bool — True for excitatory neurons

    Synapse model selector:
        synapse_model: "COBA" or "CUBA"
    """
    # Membrane — (N,) per neuron
    C: torch.Tensor = None
    g_L: torch.Tensor = None
    v_rest: torch.Tensor = None
    v_thresh: torch.Tensor = None
    delta_T: torch.Tensor = None
    v_cut: torch.Tensor = None
    v_reset: torch.Tensor = None
    t_refrac: torch.Tensor = None

    # Adaptation — (N,)
    a: torch.Tensor = None
    b: torch.Tensor = None
    tau_w: torch.Tensor = None

    # Synaptic COBA — (N,)
    E_ge: torch.Tensor = None
    E_gi: torch.Tensor = None
    Q_ge: torch.Tensor = None
    Q_gi: torch.Tensor = None
    tau_ge: torch.Tensor = None
    tau_gi: torch.Tensor = None

    # Synaptic CUBA — (N,)
    J_exc: torch.Tensor = None
    J_inh: torch.Tensor = None

    # External input — (N,)
    I_bias: torch.Tensor = None
    stim_scale: torch.Tensor = None

    # Initial conditions (scalars, not per-neuron)
    v_0_mean: float = 0.0
    v_0_std: float = 4.0

    # Topology
    edge_index: torch.Tensor = None       # (2, E)
    is_excitatory: torch.Tensor = None    # (N,) bool

    # Synapse model selector
    synapse_model: str = "COBA"

    @classmethod
    def from_defaults(cls, n_neurons: int, is_excitatory: torch.Tensor,
                      edge_index: torch.Tensor, synapse_model: str = "COBA",
                      device: torch.device | str = "cpu",
                      overrides: dict | None = None) -> FlyVisAdExODEParams:
        """Construct from Zerlaut defaults with per-neuron expansion.

        Args:
            n_neurons: total number of neurons
            is_excitatory: (N,) bool tensor — True for excitatory neurons
            edge_index: (2, E) connectivity
            synapse_model: "COBA" or "CUBA"
            device: target device
            overrides: dict of param_name -> value to override defaults
        """
        d = {**ADEX_DEFAULTS}
        if overrides:
            d.update(overrides)

        def _expand(val):
            return torch.full((n_neurons,), val, dtype=torch.float32, device=device)

        return cls(
            C=_expand(d["C"]),
            g_L=_expand(d["g_L"]),
            v_rest=_expand(d["v_rest"]),
            v_thresh=_expand(d["v_thresh"]),
            delta_T=_expand(d["delta_T"]),
            v_cut=_expand(d["v_cut"]),
            v_reset=_expand(d["v_reset"]),
            t_refrac=_expand(d["t_refrac"]),
            a=_expand(d["a"]),
            b=_expand(d["b"]),
            tau_w=_expand(d["tau_w"]),
            E_ge=_expand(d["E_ge"]),
            E_gi=_expand(d["E_gi"]),
            Q_ge=_expand(d["Q_ge"]),
            Q_gi=_expand(d["Q_gi"]),
            tau_ge=_expand(d["tau_ge"]),
            tau_gi=_expand(d["tau_gi"]),
            J_exc=_expand(d["J_exc"]),
            J_inh=_expand(d["J_inh"]),
            I_bias=_expand(d["I_bias"]),
            stim_scale=_expand(d["stim_scale"]),
            v_0_mean=d["v_0_mean"],
            v_0_std=d["v_0_std"],
            edge_index=edge_index.to(device),
            is_excitatory=is_excitatory.to(device),
            synapse_model=synapse_model,
        )

    @classmethod
    def from_flyvis_network(cls, net, synapse_model: str = "COBA",
                            device: torch.device | str = "cpu",
                            overrides: dict | None = None) -> FlyVisAdExODEParams:
        """Construct from a flyvis Network, using Zerlaut defaults for AdEx params.

        E/I identity is inferred from the sign of synaptic weights:
        neurons with net positive outgoing weight are excitatory.
        """
        params = net._param_api()
        W = (params.edges.syn_strength * params.edges.syn_count * params.edges.sign).detach().to(device).float()
        src_raw = net.connectome.edges.source_index[:]
        dst_raw = net.connectome.edges.target_index[:]
        edge_index = torch.stack([
            torch.tensor(src_raw, dtype=torch.long, device=device) if not isinstance(src_raw, torch.Tensor) else src_raw.to(device).long(),
            torch.tensor(dst_raw, dtype=torch.long, device=device) if not isinstance(dst_raw, torch.Tensor) else dst_raw.to(device).long(),
        ], dim=0)

        n_neurons = len(params.nodes.time_const)
        src = edge_index[0]

        # Infer E/I from net outgoing weight sign per neuron
        sum_w = torch.zeros(n_neurons, device=device)
        sum_w.scatter_add_(0, src, W)
        is_excitatory = (sum_w >= 0)

        return cls.from_defaults(
            n_neurons=n_neurons,
            is_excitatory=is_excitatory,
            edge_index=edge_index,
            synapse_model=synapse_model,
            device=device,
            overrides=overrides,
        )


# ---------------------------------------------------------------------------
# FlyVis Hodgkin-Huxley model params
# ---------------------------------------------------------------------------

# Classic HH defaults (squid giant axon, Hodgkin & Huxley 1952).
# Units: mV, uF/cm^2, mS/cm^2, uA/cm^2, ms.
HH_DEFAULTS = dict(
    # Membrane capacitance
    C=1.0,               # uF/cm^2
    # Leak
    g_L=0.3,             # mS/cm^2
    E_L=-54.387,         # mV — leak reversal potential
    # Sodium
    g_Na=120.0,          # mS/cm^2
    E_Na=50.0,           # mV — sodium reversal potential
    # Potassium
    g_K=36.0,            # mS/cm^2
    E_K=-77.0,           # mV — potassium reversal potential
    # Synaptic coupling (continuous, voltage-dependent)
    syn_tau=5.0,         # ms — synaptic activation time constant
    syn_slope=5.0,       # mV — sigmoid slope for presynaptic activation
    syn_v_half=-45.0,    # mV — sigmoid midpoint (allows subthreshold transmission)
    # External input
    I_bias=3.0,          # uA/cm^2 — tonic drive (depolarises to ~-44mV, subthreshold)
    stim_scale=50.0,     # uA/cm^2 per unit stimulus
    # Weight scaling (flyvis connectome weights calibrated for graded model)
    w_scale=2.0,         # global multiplier on connectome W for HH dynamics
)


@register_ode_params("flyvis_hodgkin_huxley")
@dataclass
class FlyVisHodgkinHuxleyODEParams(ODEParamsBase):
    """Parameters for the Hodgkin-Huxley continuous spiking FlyVis ODE.

    Per-neuron membrane params (indexed by neuron, one value per node):
        C, g_L, E_L, g_Na, E_Na, g_K, E_K

    Per-neuron synaptic coupling (continuous, voltage-dependent):
        syn_tau, syn_slope, syn_v_half

    Per-neuron external input:
        I_bias, stim_scale

    Network topology:
        edge_index: (2, E) source/destination indices
        W: (E,) effective synaptic weights (from flyvis connectome)
    """
    # Membrane — (N,) per neuron
    C: torch.Tensor = None
    g_L: torch.Tensor = None
    E_L: torch.Tensor = None
    g_Na: torch.Tensor = None
    E_Na: torch.Tensor = None
    g_K: torch.Tensor = None
    E_K: torch.Tensor = None

    # Synaptic coupling — (N,)
    syn_tau: torch.Tensor = None
    syn_slope: torch.Tensor = None
    syn_v_half: torch.Tensor = None

    # External input — (N,)
    I_bias: torch.Tensor = None
    stim_scale: torch.Tensor = None

    # Topology
    edge_index: torch.Tensor = None  # (2, E)
    W: torch.Tensor = None           # (E,) effective synaptic weights

    @classmethod
    def from_defaults(cls, n_neurons: int, edge_index: torch.Tensor,
                      W: torch.Tensor,
                      device: torch.device | str = "cpu",
                      overrides: dict | None = None) -> FlyVisHodgkinHuxleyODEParams:
        """Construct from HH defaults with per-neuron expansion."""
        d = {**HH_DEFAULTS}
        if overrides:
            d.update(overrides)

        def _expand(val):
            return torch.full((n_neurons,), val, dtype=torch.float32, device=device)

        return cls(
            C=_expand(d["C"]),
            g_L=_expand(d["g_L"]),
            E_L=_expand(d["E_L"]),
            g_Na=_expand(d["g_Na"]),
            E_Na=_expand(d["E_Na"]),
            g_K=_expand(d["g_K"]),
            E_K=_expand(d["E_K"]),
            syn_tau=_expand(d["syn_tau"]),
            syn_slope=_expand(d["syn_slope"]),
            syn_v_half=_expand(d["syn_v_half"]),
            I_bias=_expand(d["I_bias"]),
            stim_scale=_expand(d["stim_scale"]),
            edge_index=edge_index.to(device),
            W=W.to(device),
        )

    @classmethod
    def from_flyvis_network(cls, net, device: torch.device | str = "cpu",
                            overrides: dict | None = None) -> FlyVisHodgkinHuxleyODEParams:
        """Construct from a flyvis Network.

        Per-type params derived from flyvis connectome:
            tau_i -> g_L = C / tau_i  (leak conductance from time constant)
            V_i_rest -> E_L           (leak reversal from resting potential)

        Na/K conductances use uniform squid-axon defaults.
        Synaptic weights W come from the connectome (continuous coupling).
        """
        params = net._param_api()
        W = (params.edges.syn_strength * params.edges.syn_count * params.edges.sign).detach().to(device).float()
        src_raw = net.connectome.edges.source_index[:]
        dst_raw = net.connectome.edges.target_index[:]
        edge_index = torch.stack([
            torch.tensor(src_raw, dtype=torch.long, device=device) if not isinstance(src_raw, torch.Tensor) else src_raw.to(device).long(),
            torch.tensor(dst_raw, dtype=torch.long, device=device) if not isinstance(dst_raw, torch.Tensor) else dst_raw.to(device).long(),
        ], dim=0)

        n_neurons = len(params.nodes.time_const)

        d = {**HH_DEFAULTS}
        if overrides:
            d.update(overrides)

        def _expand(val):
            return torch.full((n_neurons,), val, dtype=torch.float32, device=device)

        # Use uniform standard HH values for all neurons.
        # Previous approach derived g_L = C/tau_i from flyvis time constants,
        # but flyvis tau_i values are in arbitrary units (~0.01-0.1) which
        # produced g_L >> 0.3, overwhelming Na/K channels and preventing spikes.
        return cls(
            C=_expand(d["C"]),
            g_L=_expand(d["g_L"]),
            E_L=_expand(d["E_L"]),
            g_Na=_expand(d["g_Na"]),
            E_Na=_expand(d["E_Na"]),
            g_K=_expand(d["g_K"]),
            E_K=_expand(d["E_K"]),
            syn_tau=_expand(d["syn_tau"]),
            syn_slope=_expand(d["syn_slope"]),
            syn_v_half=_expand(d["syn_v_half"]),
            I_bias=_expand(d["I_bias"]),
            stim_scale=_expand(d["stim_scale"]),
            edge_index=edge_index,
            W=W * d["w_scale"],
        )


# ---------------------------------------------------------------------------
# Zebrafish oculomotor integrator (Beiran & Litwin-Kumar 2023, Fig 5)
# ---------------------------------------------------------------------------

@register_ode_params("zebrafish", "zebrafish_oculomotor")
@dataclass
class ZebrafishODEParams(ODEParamsBase):
    """Parameters for the zebrafish oculomotor linear integrator ODE.

    Ref: papers/Code_NN/Code_NN/nn_fig5_zebrafish_teacher.py
         simulate_series() line 172:
         r[i,:] = r[i-1,:] + dt*(W @ r[i-1,:] - r[i-1,:] + I[i-1]*v_in) / tau

    ODE: dr/dt = (-r + W @ r + I * v_in) / tau
    Linear (no nonlinearity). tau=1.0 fixed. dt=0.001.

    Edge params:
        edge_index: (2, E) source/destination indices
        W: (E,) sparse weights (from dense W scaled to spectral radius 0.9)

    Node params:
        v_in: (N,) input vector (eigenvector combination + noise)
        neuron_types: (N,) int cell type labels

    Scalars:
        tau: time constant (default 1.0)
        n_neurons: number of neurons
    """
    edge_index: torch.Tensor = None  # (2, E)
    W: torch.Tensor = None           # (E,)
    v_in: torch.Tensor = None        # (N,) input direction vector
    neuron_types: torch.Tensor = None  # (N,) int type labels
    tau: float = 1.0
    n_neurons: int = 0

    @classmethod
    def from_connectome(cls, datapath: str, device: torch.device | str = "cpu"):
        """Construct from Goldman lab MATLAB data.

        Ref: nn_fig5_zebrafish_teacher.py lines 64-179
        Uses load_zebrafish_connectome() from connconstr_data.py.
        """
        from flyvis_gnn.generators.connconstr_data import (
            dense_to_sparse, load_zebrafish_connectome,
        )

        data = load_zebrafish_connectome(datapath)
        edge_index, W_sparse = dense_to_sparse(data["W"])

        # Build integer type labels from cell_types
        cell_types = data["cell_types"]
        N = data["N"]
        unique_types = list(data["cell_type_names"])
        type_labels = np.zeros(N, dtype=np.int64)
        for i, name in enumerate(unique_types):
            mask = (cell_types == name).flatten()
            type_labels[mask] = i

        return cls(
            edge_index=edge_index.to(device),
            W=W_sparse.to(device),
            v_in=torch.tensor(data["v_in"], dtype=torch.float32, device=device),
            neuron_types=torch.tensor(type_labels, dtype=torch.long, device=device),
            tau=1.0,
            n_neurons=N,
        )

    @classmethod
    def from_pretrained(cls, datapath: str, device: torch.device | str = "cpu"):
        """Construct from pre-saved zebrafish.npz (output of teacher script).

        Ref: nn_fig5_zebrafish_teacher.py line 394
        """
        from flyvis_gnn.generators.connconstr_data import (
            dense_to_sparse, load_zebrafish_pretrained,
        )

        # zebrafish.npz may be in parent directory
        try:
            data = load_zebrafish_pretrained(datapath)
        except FileNotFoundError:
            parent = os.path.dirname(datapath)
            data = load_zebrafish_pretrained(parent)
        W_dense = data["W"]
        N = W_dense.shape[0]
        edge_index, W_sparse = dense_to_sparse(W_dense)

        return cls(
            edge_index=edge_index.to(device),
            W=W_sparse.to(device),
            v_in=torch.tensor(data["v_in"], dtype=torch.float32, device=device),
            neuron_types=torch.zeros(N, dtype=torch.long, device=device),
            tau=1.0,
            n_neurons=N,
        )

    def create_ode(self, device=None):
        from flyvis_gnn.generators.connconstr_zebrafish_ode import ZebrafishODE
        return ZebrafishODE(ode_params=self, device=device)

    def get_dt(self):
        return 0.001  # Ref: simulate_series line 166

    def get_n_neurons(self):
        return self.n_neurons

    def get_n_frames(self, sim):
        return 21000  # 3 pulse repeats (line 161-164)

    def generate_stimulus(self, n_frames, sim, device=None):
        """Returns per-neuron stimulus tensor (T, N)."""
        from flyvis_gnn.generators.connconstr_data import generate_zebrafish_stimulus
        I = generate_zebrafish_stimulus(n_frames)
        # Broadcast scalar stimulus to all neurons (ODE uses v_in internally)
        I_t = torch.tensor(I, dtype=torch.float32, device=device)
        return I_t.unsqueeze(1).expand(-1, self.n_neurons)  # (T, N)

    def init_state(self, voltage, datapath=None, device=None):
        pass  # zero init is fine

    def get_trial_length(self):
        return 0  # no trial structure


# ---------------------------------------------------------------------------
# Drosophila adult central complex ring attractor (Beiran & Litwin-Kumar 2023, Fig 5)
# ---------------------------------------------------------------------------

@register_ode_params("drosophila_cx")
@dataclass
class DrosophilaCxODEParams(ODEParamsBase):
    """Parameters for the Drosophila central complex ring attractor ODE.

    Ref: papers/Code_NN/Code_NN/nn_fig5_drosophilaCx_teacher.py
         RNN.forward() lines 171-191:
         h += alpha * (-h + exp(g) * softplus(h+b, beta) @ J^T + input) / tau
         tau = 2.6 + 2.4 * tanh(tau_raw)  →  bounded [0.2, 5.0]
         J_eff = exp(wrec) * mwrec  (line 184)
         Activation: Softplus(beta=5) (lines 105-106)

    Edge params:
        edge_index: (2, E)
        W: (E,) effective weights = exp(wrec_log) * sign(mwrec)

    Node params:
        g: (N,) log gain
        b: (N,) bias
        h0: (N,) initial hidden state
        tau_raw: (N,) raw time constant (bounded via tanh)
        neuron_types: (N,) int type labels

    Input/output weights:
        winp: (input_size, N) input projection
        wout: (N, output_size) output projection

    Scalars:
        alpha: learning rate for ODE integration (default 1.0)
        beta: softplus sharpness (default 5.0)
        noise_std: noise magnitude (default 0.0)
        n_neurons: total neuron count
    """
    edge_index: torch.Tensor = None    # (2, E)
    W: torch.Tensor = None             # (E,)
    g: torch.Tensor = None             # (N,) log gain
    b: torch.Tensor = None             # (N,) bias
    h0: torch.Tensor = None            # (N,) initial state
    tau_raw: torch.Tensor = None       # (N,) raw time constant
    neuron_types: torch.Tensor = None  # (N,) int type labels
    winp: torch.Tensor = None          # (input_size, N)
    wout: torch.Tensor = None          # (N, output_size)
    alpha: float = 1.0
    beta: float = 5.0
    noise_std: float = 0.0
    n_neurons: int = 0

    @classmethod
    def from_connectome(cls, datapath: str, device: torch.device | str = "cpu"):
        """Construct from hemibrain CSV data.

        Ref: nn_fig5_drosophilaCx_teacher.py lines 431-598
        Uses load_drosophila_cx_connectome() from connconstr_data.py.
        """
        from flyvis_gnn.generators.connconstr_data import (
            dense_to_sparse, load_drosophila_cx_connectome,
        )

        # Accept either parent dir or hemibrain subdir
        hemibrain_dir = datapath
        if not os.path.exists(os.path.join(datapath, "traced-neurons.csv")):
            hemibrain_dir = os.path.join(datapath, "exported-traced-adjacencies-v1.2")
        data = load_drosophila_cx_connectome(hemibrain_dir)
        N = data["N"]

        # J_effective = exp(wrec_log) * mwrec  (line 184)
        J_eff = data["J_effective"]
        edge_index, W_sparse = dense_to_sparse(J_eff)

        # Initialize node params to zeros (to be trained or loaded)
        # Ref: nn_fig5_drosophilaCx_teacher.py RNN.__init__ lines 97-115
        g = torch.zeros(N, dtype=torch.float32, device=device)
        b = torch.zeros(N, dtype=torch.float32, device=device)
        h0 = torch.zeros(N, dtype=torch.float32, device=device)
        tau_raw = torch.zeros(N, dtype=torch.float32, device=device)  # tanh(0)=0 → tau=2.6

        return cls(
            edge_index=edge_index.to(device),
            W=W_sparse.to(device),
            g=g,
            b=b,
            h0=h0,
            tau_raw=tau_raw,
            neuron_types=torch.tensor(data["neuron_types"], dtype=torch.long, device=device),
            winp=torch.tensor(data["winp"], dtype=torch.float32, device=device),
            wout=torch.tensor(data["wout"], dtype=torch.float32, device=device),
            alpha=1.0,
            beta=5.0,
            noise_std=0.0,
            n_neurons=N,
        )

    @classmethod
    def from_pretrained(cls, datapath: str, device: torch.device | str = "cpu"):
        """Construct from trained teacher params_netSimpleRing2_final.npz.

        Ref: nn_fig5_drosophilaCx_teacher.py lines 701-709
        Saved arrays:
          arr_0 = JJ = exp(wrec) * sign(mwrec)  (152, 152) effective connectivity
          arr_1 = gg = exp(g)  (152,) gains (already exponentiated)
          arr_2 = bb  (152,) biases
          arr_3 = hh0  (152,) initial hidden state
          arr_4 = wI  (48, 152) input weights
          arr_5 = wOut  (152, 49) output weights
          arr_6 = alpha_  scalar
          arr_7 = si_  (48, 152) input scaling
        """
        from flyvis_gnn.generators.connconstr_data import (
            dense_to_sparse, load_drosophila_cx_connectome,
        )

        # Prefer the .pt state dict (has taus), fallback to .npz
        pt_path = os.path.join(datapath, "netPopVec_Wrec_simplering.pt")
        npz_path = os.path.join(datapath, "params_netSimpleRing2_final.npz")

        if os.path.exists(pt_path):
            # Load full state dict — has all trained params including taus
            sd = torch.load(pt_path, map_location='cpu', weights_only=False)
            wrec_t = sd['wrec'].numpy()
            mwrec_t = sd['mwrec'].numpy()
            JJ = np.exp(wrec_t) * mwrec_t  # effective J (line 184)
            g = sd['g'].numpy().flatten()   # log gain (ODE uses exp(g))
            bb = sd['b'].numpy().flatten()
            hh0 = sd['h0'].numpy().flatten()
            tau_raw = sd['taus'].numpy().flatten()
            wI = sd['wi'].numpy()           # (48, N)
            si_ = sd['si'].numpy()          # (48, 1)
            wOut = sd['wout'].numpy()       # (N, 49)
            alpha_ = 0.2  # Ref: RNN.__init__ default alpha=0.2
            N = JJ.shape[0]
        elif os.path.exists(npz_path):
            AA = np.load(npz_path)
            JJ = AA['arr_0']       # effective J
            gg = AA['arr_1']       # exp(g), already exponentiated
            bb = AA['arr_2']       # bias
            hh0 = AA['arr_3']      # initial state
            wI = AA['arr_4']       # (48, N) input weights
            wOut = AA['arr_5']     # (N, 49) output weights
            alpha_ = float(AA['arr_6'])
            si_ = AA['arr_7']      # (48, N) or (48, 1) input scaling
            N = JJ.shape[0]
            g = np.log(np.maximum(gg, 1e-12))  # log it back
            tau_raw = np.zeros(N, dtype=np.float32)  # default
        else:
            raise FileNotFoundError(
                f"CX pretrained not found at {pt_path} or {npz_path}\n"
                "Run nn_fig5_drosophilaCx_teacher.py first."
            )

        edge_index, W_sparse = dense_to_sparse(JJ)

        # Load neuron type labels from hemibrain data
        hemibrain_dir = datapath
        if not os.path.exists(os.path.join(datapath, "traced-neurons.csv")):
            hemibrain_dir = os.path.join(datapath, "exported-traced-adjacencies-v1.2")
        try:
            cx_data = load_drosophila_cx_connectome(hemibrain_dir)
            neuron_types = torch.tensor(cx_data["neuron_types"], dtype=torch.long, device=device)
        except (FileNotFoundError, Exception):
            neuron_types = torch.zeros(N, dtype=torch.long, device=device)

        # Effective input weights: exp(si) * wI  (Ref: line 187)
        winp_effective = np.exp(si_) * wI

        return cls(
            edge_index=edge_index.to(device),
            W=W_sparse.to(device),
            g=torch.tensor(g.flatten(), dtype=torch.float32, device=device),
            b=torch.tensor(bb.flatten(), dtype=torch.float32, device=device),
            h0=torch.tensor(hh0.flatten(), dtype=torch.float32, device=device),
            tau_raw=torch.tensor(tau_raw.flatten(), dtype=torch.float32, device=device),
            neuron_types=neuron_types,
            winp=torch.tensor(winp_effective, dtype=torch.float32, device=device),
            wout=torch.tensor(wOut, dtype=torch.float32, device=device),
            alpha=alpha_,
            beta=5.0,
            noise_std=0.005,  # Ref: RNN.__init__ noise_std=0.005
            n_neurons=N,
        )

    def create_ode(self, device=None):
        from flyvis_gnn.generators.connconstr_cx_ode import DrosophilaCxODE
        return DrosophilaCxODE(ode_params=self, device=device)

    def get_dt(self):
        return 0.1  # Ref: teacher training dt

    def get_n_neurons(self):
        return self.n_neurons

    def get_n_frames(self, sim):
        n_trials = sim.connconstr_n_trials
        T_trial = 6.0
        return n_trials * int(T_trial / self.get_dt())

    def generate_stimulus(self, n_frames, sim, device=None):
        """Returns per-neuron stimulus tensor (T, N)."""
        from flyvis_gnn.generators.connconstr_data import (
            generate_cx_stimulus, load_drosophila_cx_connectome,
        )
        from flyvis_gnn.utils import to_numpy
        # Accept either parent dir or hemibrain subdir
        hemibrain_dir = sim.connconstr_datapath
        if not os.path.exists(os.path.join(hemibrain_dir, "traced-neurons.csv")):
            hemibrain_dir = os.path.join(hemibrain_dir, "exported-traced-adjacencies-v1.2")
        cx_data = load_drosophila_cx_connectome(hemibrain_dir)
        dt = self.get_dt()
        n_trials = sim.connconstr_n_trials
        T_trial = 6.0
        _, _, cx_inps, _ = generate_cx_stimulus(
            n_trials, T_trial, dt,
            cx_data["epg_ix"], cx_data["W_16to46"], cx_data["W_46to3"],
            seed=sim.seed,
        )
        cx_inps_flat = cx_inps.reshape(-1, 48)
        winp_np = to_numpy(self.winp)
        stim_projected = cx_inps_flat @ winp_np
        return torch.tensor(stim_projected, dtype=torch.float32, device=device)

    def init_state(self, voltage, datapath=None, device=None):
        if self.h0 is not None:
            voltage[:] = self.h0.clone()

    def get_trial_length(self):
        return int(6.0 / self.get_dt())


# ---------------------------------------------------------------------------
# Drosophila larva two-population model (Beiran & Litwin-Kumar 2023, Fig 5)
# ---------------------------------------------------------------------------

@register_ode_params("larva")
@dataclass
class LarvaODEParams(ODEParamsBase):
    """Parameters for the Drosophila larva two-population ODE.

    Ref: papers/Code_NN/Code_NN/Data/Figure5/setup.py forwardpass() lines 24-45

    Two populations:
      - Premotor (PMN, N neurons): up' = (1-dt/taup)*up + (dt/taup)*(gp*softplus(up) @ Jpp + bp + wsp @ stim)
      - Motor (MN, M neurons):     um' = (1-dt/taum)*um + (dt/taum)*(gm*softplus(up) @ Jpm + bm)

    Activation: Softplus (torch.nn.functional.softplus), NOT ReLU.
    Gains clamped to [0.5, 5.0] (setup.py line 49-51).

    For GNN compatibility, both populations are merged into a single graph:
      - Nodes 0..N-1 are premotor, N..N+M-1 are motor
      - Jpp edges: src ∈ [0,N), dst ∈ [0,N)
      - Jpm edges: src ∈ [0,N), dst ∈ [N,N+M)

    Edge params:
        edge_index: (2, E) combined Jpp + Jpm edges
        W: (E,) edge weights

    Node params:
        gp: (N,) premotor gain
        gm: (M,) motor gain
        bp: (N,) premotor bias
        bm: (M,) motor bias
        wsp: (S, N) stimulus-to-premotor weights
        neuron_types: (N+M,) int labels (0=premotor, 1=motor)

    Scalars:
        taup: premotor time constant
        taum: motor time constant
        n_premotor: N
        n_motor: M
        dt: integration time step
    """
    edge_index: torch.Tensor = None    # (2, E)
    W: torch.Tensor = None             # (E,)
    gp: torch.Tensor = None            # (N,)
    gm: torch.Tensor = None            # (M,)
    bp: torch.Tensor = None            # (N,)
    bm: torch.Tensor = None            # (M,)
    wsp: torch.Tensor = None           # (S, N) stimulus→premotor
    neuron_types: torch.Tensor = None  # (N+M,)
    taup: float = 1.0
    taum: float = 1.0
    n_premotor: int = 0
    n_motor: int = 0
    dt: float = 0.05  # Ref: setup.py line 224,227

    @classmethod
    def from_connectome(cls, datapath: str, device: torch.device | str = "cpu"):
        """Construct from larva h5 connectivity data.

        Ref: setup.py loadconns() lines 68-81
        Uses load_larva_connectome() from connconstr_data.py.
        """
        from flyvis_gnn.generators.connconstr_data import (
            dense_to_sparse, load_larva_connectome,
        )

        data = load_larva_connectome(datapath)
        N = data["N"]  # premotor
        M = data["M"]  # motor

        # Build combined graph: premotor [0..N-1], motor [N..N+M-1]
        # Jpp/Jpm are in [pre, post] format (setup.py does .T on h5 data).
        # dense_to_sparse expects [post, pre], so we transpose back.
        ei_pp, w_pp = dense_to_sparse(data["Jpp"].T)
        ei_pm, w_pm = dense_to_sparse(data["Jpm"].T)
        # Shift motor indices by N
        ei_pm[1] += N

        edge_index = torch.cat([ei_pp, ei_pm], dim=1)
        W = torch.cat([w_pp, w_pm])

        # Initialize gains and biases (to be trained or loaded from pretrained)
        gp = torch.ones(N, dtype=torch.float32, device=device)
        gm = torch.ones(M, dtype=torch.float32, device=device)
        bp = torch.zeros(N, dtype=torch.float32, device=device)
        bm = torch.zeros(M, dtype=torch.float32, device=device)

        # Type labels: 0=premotor, 1=motor
        neuron_types = torch.cat([
            torch.zeros(N, dtype=torch.long),
            torch.ones(M, dtype=torch.long),
        ]).to(device)

        return cls(
            edge_index=edge_index.to(device),
            W=W.to(device),
            gp=gp, gm=gm, bp=bp, bm=bm,
            wsp=torch.zeros(2, N, dtype=torch.float32, device=device),
            neuron_types=neuron_types,
            taup=1.0, taum=1.0,
            n_premotor=N, n_motor=M,
            dt=0.05,  # Ref: setup.py line 224,227
        )

    @classmethod
    def from_pretrained(cls, datapath: str, device: torch.device | str = "cpu"):
        """Construct from pre-trained ashokF_softplus.npz parameters.

        Ref: nn_fig5_plots_abc.py lines 31-41
        """
        from flyvis_gnn.generators.connconstr_data import (
            dense_to_sparse, load_larva_pretrained,
        )

        data = load_larva_pretrained(datapath)
        Jpp = data["Jpp"]
        Jpm = data["Jpm"]
        # Jpp: (N, N) premotor recurrent in [pre, post] format
        # Jpm: (N, M) premotor→motor in [pre, post] format
        N = Jpp.shape[0]  # premotor
        M = Jpm.shape[1]  # motor (columns = post = motor targets)

        # Pretrained Jpp/Jpm are in [pre, post] format (same convention as setup.py).
        # dense_to_sparse expects [post, pre], so transpose.
        ei_pp, w_pp = dense_to_sparse(Jpp.T)
        ei_pm, w_pm = dense_to_sparse(Jpm.T)
        ei_pm[1] += N

        edge_index = torch.cat([ei_pp, ei_pm], dim=1)
        W = torch.cat([w_pp, w_pm])

        neuron_types = torch.cat([
            torch.zeros(N, dtype=torch.long),
            torch.ones(M, dtype=torch.long),
        ]).to(device)

        return cls(
            edge_index=edge_index.to(device),
            W=W.to(device),
            gp=torch.tensor(data["gp"].flatten(), dtype=torch.float32, device=device),
            gm=torch.tensor(data["gm"].flatten(), dtype=torch.float32, device=device),
            bp=torch.tensor(data["bp"].flatten(), dtype=torch.float32, device=device),
            bm=torch.tensor(data["bm"].flatten(), dtype=torch.float32, device=device),
            wsp=torch.tensor(data["wsp"], dtype=torch.float32, device=device),
            neuron_types=neuron_types,
            taup=float(data["taup"].item() if hasattr(data["taup"], 'item') else data["taup"]),
            taum=float(data["taum"].item() if hasattr(data["taum"], 'item') else data["taum"]),
            n_premotor=N,
            n_motor=M,
            dt=0.05,  # Ref: setup.py line 224,227
        )

    def create_ode(self, device=None):
        from flyvis_gnn.generators.connconstr_larva_ode import LarvaODE
        return LarvaODE(ode_params=self, device=device)

    def get_dt(self):
        return self.dt

    def get_n_neurons(self):
        return self.n_premotor + self.n_motor

    def get_n_frames(self, sim):
        # Paper uses B=2 conditions × 120 frames each.
        # We concatenate both conditions and repeat n_repeats times.
        n_repeats = getattr(sim, 'connconstr_n_trials', 0)
        if n_repeats <= 0:
            n_repeats = 10  # default: 10 repeats → 2400 frames
        T_per_condition = int(6.0 / self.dt)  # 120 frames
        return 2 * T_per_condition * n_repeats  # B=2 conditions × repeats

    def generate_stimulus(self, n_frames, sim, device=None):
        """Returns per-neuron stimulus tensor (T, N_total).

        Concatenates both B=2 conditions (forward + backward) and repeats
        to produce enough training data.
        """
        from flyvis_gnn.generators.connconstr_data import (
            generate_larva_stimulus, load_larva_connectome,
        )
        conn_data = load_larva_connectome(sim.connconstr_datapath)
        mtarg, s_stim = generate_larva_stimulus(
            conn_data["mnorder"], B=2, S=2, dt=self.dt
        )
        # s_stim: (T_trial, B, S)
        T_trial = s_stim.shape[0]
        N_total = self.n_premotor + self.n_motor

        # Build one cycle: condition 0 then condition 1
        stim_cycle = torch.zeros(2 * T_trial, N_total, dtype=torch.float32, device=device)
        for bi in range(2):
            s_raw = torch.tensor(s_stim[:, bi, :], dtype=torch.float32, device=device)
            offset = bi * T_trial
            for t_idx in range(T_trial):
                stim_cycle[offset + t_idx, :self.n_premotor] = self.wsp.t() @ s_raw[t_idx]

        # Repeat cycle to fill n_frames
        n_cycle = stim_cycle.shape[0]
        n_repeats = (n_frames + n_cycle - 1) // n_cycle
        stim_all = stim_cycle.repeat(n_repeats, 1)[:n_frames]
        return stim_all

    def get_trial_length(self):
        # Reset at cycle boundary: 2 conditions × 120 frames = 240 frames per cycle
        return 2 * int(6.0 / self.dt)

    def init_state(self, voltage, datapath=None, device=None):
        try:
            from flyvis_gnn.generators.connconstr_data import load_larva_pretrained
            pretrained = load_larva_pretrained(datapath)
            if "p0" in pretrained and pretrained["p0"] is not None:
                p0 = pretrained["p0"]
                if p0.ndim == 3:
                    p0 = p0[0, 0, :]
                voltage[:self.n_premotor] = torch.tensor(
                    p0.flatten()[:self.n_premotor], dtype=torch.float32, device=device)
            if "m0" in pretrained and pretrained["m0"] is not None:
                m0 = pretrained["m0"]
                if m0.ndim == 3:
                    m0 = m0[0, 0, :]
                voltage[self.n_premotor:] = torch.tensor(
                    m0.flatten()[:self.n_motor], dtype=torch.float32, device=device)
        except (FileNotFoundError, KeyError):
            pass
