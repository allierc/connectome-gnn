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

from connectome_gnn.log import get_logger

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
                    f"ODE params name '{name}' already registered to {_ODE_PARAMS_REGISTRY[name].__name__}"
                )
            _ODE_PARAMS_REGISTRY[name] = cls
        return cls

    return decorator


def get_ode_params_class(name: str) -> type:
    """Look up ODE params class by config signal_model_name."""
    if name not in _ODE_PARAMS_REGISTRY:
        available = sorted(_ODE_PARAMS_REGISTRY.keys())
        raise KeyError(f"Unknown ODE params '{name}'. Available: {available}")
    return _ODE_PARAMS_REGISTRY[name]


def load_edge_index(folder: str, device: torch.device | str = "cpu") -> torch.Tensor:
    """Load edge_index from ode_params.pt, falling back to edge_index.pt."""
    ode_path = os.path.join(folder, "ode_params.pt")
    if os.path.exists(ode_path):
        state = torch.load(ode_path, map_location=device, weights_only=True)
        return state["edge_index"]
    path = os.path.join(folder, "edge_index.pt")
    if os.path.exists(path):
        return torch.load(path, map_location=device, weights_only=True)
    raise FileNotFoundError(f"No edge_index.pt or ode_params.pt in {folder}")


def load_weights(folder: str, device: torch.device | str = "cpu") -> torch.Tensor:
    """Load synaptic weights from ode_params.pt, falling back to weights.pt."""
    ode_path = os.path.join(folder, "ode_params.pt")
    if os.path.exists(ode_path):
        state = torch.load(ode_path, map_location=device, weights_only=True)
        return state["W"]
    path = os.path.join(folder, "weights.pt")
    if os.path.exists(path):
        return torch.load(path, map_location=device, weights_only=True)
    raise FileNotFoundError(f"No weights.pt or ode_params.pt in {folder}")


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
        raise FileNotFoundError(f"No ode_params.pt found at {folder} and no legacy loader defined for {cls.__name__}")

    # ------------------------------------------------------------------
    # Analysis interface — override in subclasses for model-specific
    # interpretation of learned f_theta and g_phi
    # ------------------------------------------------------------------

    def has_tau(self) -> bool:
        """Whether this model has per-neuron time constants to recover."""
        return False

    def has_vrest(self) -> bool:
        """Whether this model has per-neuron resting potentials to recover."""
        return False

    def has_gain(self) -> bool:
        """Whether this model has per-neuron gain parameters to recover."""
        return False

    def has_bias(self) -> bool:
        """Whether this model has per-neuron bias parameters to recover."""
        return False

    def gt_tau(self, n_neurons: int) -> np.ndarray | None:
        """Ground truth tau array (n_neurons,). None if not applicable."""
        return None

    def gt_vrest(self, n_neurons: int) -> np.ndarray | None:
        """Ground truth V_rest array (n_neurons,). None if not applicable."""
        return None

    def gt_gain(self, n_neurons: int) -> np.ndarray | None:
        """Ground truth gain array (n_neurons,). None if not applicable."""
        return None

    def gt_bias(self, n_neurons: int) -> np.ndarray | None:
        """Ground truth bias array (n_neurons,). None if not applicable."""
        return None

    def derive_tau(self, slopes_f_theta: np.ndarray, n_neurons: int) -> np.ndarray:
        """Derive time constants from f_theta slopes. Default: tau = 1/(-slope)."""
        slopes = slopes_f_theta[:n_neurons]
        derived = np.where(slopes != 0, 1.0 / -slopes, 1.0)
        return np.clip(derived, 0, 10)

    def derive_vrest(self, slopes_f_theta: np.ndarray, offsets_f_theta: np.ndarray, n_neurons: int) -> np.ndarray:
        """Derive resting potentials from f_theta slopes/offsets. Default: V = -offset/slope."""
        slopes = slopes_f_theta[:n_neurons]
        offsets = offsets_f_theta[:n_neurons]
        return np.where(slopes != 0, -offsets / slopes, 0.0)

    def gt_g_phi_func(self, v: np.ndarray) -> np.ndarray:
        """Ground truth g_phi(v) evaluated at points v. Shape: (n_neurons, n_pts) or (n_pts,).
        Override in subclass. Default: ReLU."""
        return np.maximum(v, 0.0)

    def g_phi_label(self) -> str:
        """Label for the ground truth g_phi in plots."""
        return r"$\mathrm{ReLU}(v_j)$"

    def f_theta_label(self) -> str:
        """Label for the ground truth f_theta in plots."""
        return r"$(-v_i + V^{rest}_i) / \tau_i$"

    def f_theta_param_names(self) -> list[str]:
        """Names of parameters recoverable from f_theta. Used for scatter plot labels."""
        return [r"$\tau_i$", r"$V^{rest}_i$"]

    # ------------------------------------------------------------------
    # g_phi curve fitting and W correction
    # ------------------------------------------------------------------

    def fit_g_phi_curves(self, v_ranges: np.ndarray, learned_curves: np.ndarray) -> dict:
        """Fit learned g_phi curves to extract model-specific parameters.

        Args:
            v_ranges: (N, n_pts) per-neuron voltage grids.
            learned_curves: (N, n_pts) learned g_phi output per neuron.

        Returns:
            dict with model-specific fitted params. Must include 'correction'
            key: (N,) array — per-neuron multiplicative factor absorbed into W.
            For ReLU models this is the slope; for softplus it's the gain.
        """
        # Default: linear slope (suitable for ReLU-like g_phi)
        from connectome_gnn.metrics import _vectorized_linear_fit

        slopes, offsets = _vectorized_linear_fit(v_ranges, learned_curves)
        slopes = np.asarray(slopes)
        slopes[np.abs(slopes) < 1e-8] = 1.0
        return {"correction": slopes, "slopes": slopes}

    def gt_g_phi_params(self, n_neurons: int) -> dict | None:
        """Ground truth g_phi parameters for R² comparison.
        Returns dict with same keys as fit_g_phi_curves (minus 'correction'),
        or None if no ground truth available."""
        return None

    def g_phi_param_names(self) -> list[str]:
        """Names of extractable g_phi parameters (for printing)."""
        return ["slope"]

    def effective_true_weights(self, gt_weights: np.ndarray, edges: np.ndarray, n_neurons: int) -> np.ndarray:
        """Adjust true weights to include the g_phi amplitude factor.

        For ReLU models (slope=1), returns gt_weights unchanged.
        For models where g_phi has a per-neuron gain entangled with W
        (e.g. CX softplus with exp(g)), returns W_true * gain[src].

        Args:
            gt_weights: (E,) true edge weights.
            edges: (2, E) edge indices.
            n_neurons: number of neurons.

        Returns:
            (E,) effective true weights for comparison with corrected W.
        """
        return gt_weights  # Default: no adjustment (ReLU slope ≈ 1)

    def clustering_features(self) -> list[str]:
        """Feature combinations for neuron type clustering analysis."""
        return ["a", "W"]

    def neuron_type_rmse_panels(self) -> list[str]:
        """Panel names for neuron type reconstruction plot. Each must match a key
        in the per-type RMSE dict returned by the analysis."""
        return ["weights"]


# ---------------------------------------------------------------------------
# FlyVis graded-voltage model params
# ---------------------------------------------------------------------------


@register_ode_params(
    "flyvis_A",
    "flyvis_B",
    "flyvis_C",
    "flyvis_D",
    "flyvis_A_multiple_ReLU",
    "flyvis_B_multiple_ReLU",
    "flyvis_C_multiple_ReLU",
    "flyvis_A_tanh",
    "flyvis_B_tanh",
    "flyvis_C_tanh",
    "flyvis_A_NULL",
    "flyvis_B_NULL",
    "flyvis_C_NULL",
    "flyvis_known_ode",
    "flyvis_hybrid",
    "flyvis_hybrid_flywireRF",
    "flyvis_hybrid_zeroedge",
    "flyvis_hybrid_flywireRF_zeroedge",
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

    tau_i: torch.Tensor = None  # (N,)
    V_i_rest: torch.Tensor = None  # (N,)
    edge_index: torch.Tensor = None  # (2, E)
    W: torch.Tensor = None  # (E,) effective synaptic weights

    @classmethod
    def from_flyvis_network(cls, net, device: torch.device | str = "cpu"):
        """Construct from a flyvis Network object."""
        params = net._param_api()
        tau_i = params.nodes.time_const
        V_i_rest = params.nodes.bias
        W = params.edges.syn_strength * params.edges.syn_count * params.edges.sign
        edge_index = torch.stack(
            [
                torch.tensor(net.connectome.edges.source_index[:]),
                torch.tensor(net.connectome.edges.target_index[:]),
            ],
            dim=0,
        )
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
            raise FileNotFoundError(f"No ode_params.pt or legacy .pt files found at {folder}")

        logger.info(f"loaded legacy ODE params from {folder}")
        return cls(tau_i=tau_i, V_i_rest=V_i_rest, edge_index=edge_index, W=W)

    # --- Analysis interface ---
    def has_tau(self):
        return True

    def has_vrest(self):
        return True

    def gt_tau(self, n_neurons):
        if self.tau_i is None:
            return None
        return self.tau_i[:n_neurons].cpu().numpy()

    def gt_vrest(self, n_neurons):
        if self.V_i_rest is None:
            return None
        return self.V_i_rest[:n_neurons].cpu().numpy()

    def gt_g_phi_func(self, v):
        return np.maximum(v, 0.0)  # ReLU

    def g_phi_label(self):
        return r"$\mathrm{ReLU}(v_j)$"

    def f_theta_label(self):
        return r"$(-v_i + V^{rest}_i) / \tau_i$"

    def gt_f_theta_func(self, v, n_neurons):
        """Ground truth f_theta(v) = (-v + V_rest) / tau per neuron. Shape: (n_neurons, n_pts)."""
        tau = self.gt_tau(n_neurons)
        vrest = self.gt_vrest(n_neurons)
        if tau is None or vrest is None:
            return None
        return (-v + vrest[:, None]) / tau[:, None]

    def f_theta_param_names(self):
        return [r"$\tau_i$", r"$V^{rest}_i$"]

    def clustering_features(self):
        return ["a", "τ", "V", "W", "(τ,V)", "(τ,V,W)", "(a,τ,V,W)"]

    def neuron_type_rmse_panels(self):
        return ["weights", "tau", "vrest"]


# ---------------------------------------------------------------------------
# FlyVis AdEx spiking model params
# ---------------------------------------------------------------------------

# Default values from Zerlaut et al. 2018 (AutoMind ADEX_NEURON_DEFAULTS_ZERLAUT).
# Units: mV, pF, nS, pA, ms, Hz.  Stored as dimensionless floats in those units.
ADEX_DEFAULTS = dict(
    # Membrane
    C=200.0,  # pF  — membrane capacitance
    g_L=10.0,  # nS  — leak conductance
    v_rest=-65.0,  # mV  — resting (leak reversal) potential
    v_thresh=-50.0,  # mV  — spike initiation threshold (exp onset)
    delta_T=2.0,  # mV  — exponential nonlinearity sharpness
    v_cut=0.0,  # mV  — hard spike cutoff for detection
    v_reset=-65.0,  # mV  — post-spike reset voltage
    t_refrac=5.0,  # ms  — absolute refractory period
    # Adaptation
    a=4.0,  # nS  — subthreshold adaptation coupling
    b=20.0,  # pA  — spike-triggered adaptation increment
    tau_w=500.0,  # ms  — adaptation time constant
    # Synaptic (COBA)
    E_ge=0.0,  # mV  — excitatory reversal potential
    E_gi=-80.0,  # mV  — inhibitory reversal potential
    Q_ge=1.0,  # nS  — excitatory quantal conductance
    Q_gi=5.0,  # nS  — inhibitory quantal conductance
    tau_ge=5.0,  # ms  — excitatory conductance decay
    tau_gi=5.0,  # ms  — inhibitory conductance decay
    # Synaptic (CUBA) — no defaults from Zerlaut, set to 0 as placeholder
    J_exc=0.0,  # mV  — excitatory spike kick
    J_inh=0.0,  # mV  — inhibitory spike kick
    # External input
    I_bias=0.0,  # pA  — constant bias current
    stim_scale=1.0,  # pA per unit stimulus — converts visual input to current
    # Initial conditions
    v_0_mean=0.0,  # mV  — mean offset from v_rest for initial v
    v_0_std=4.0,  # mV  — std of initial v perturbation
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
    edge_index: torch.Tensor = None  # (2, E)
    is_excitatory: torch.Tensor = None  # (N,) bool

    # Synapse model selector
    synapse_model: str = "COBA"

    @classmethod
    def from_defaults(
        cls,
        n_neurons: int,
        is_excitatory: torch.Tensor,
        edge_index: torch.Tensor,
        synapse_model: str = "COBA",
        device: torch.device | str = "cpu",
        overrides: dict | None = None,
    ) -> FlyVisAdExODEParams:
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
    def from_flyvis_network(
        cls, net, synapse_model: str = "COBA", device: torch.device | str = "cpu", overrides: dict | None = None
    ) -> FlyVisAdExODEParams:
        """Construct from a flyvis Network, using Zerlaut defaults for AdEx params.

        E/I identity is inferred from the sign of synaptic weights:
        neurons with net positive outgoing weight are excitatory.
        """
        params = net._param_api()
        W = (params.edges.syn_strength * params.edges.syn_count * params.edges.sign).detach().to(device).float()
        src_raw = net.connectome.edges.source_index[:]
        dst_raw = net.connectome.edges.target_index[:]
        edge_index = torch.stack(
            [
                torch.tensor(src_raw, dtype=torch.long, device=device)
                if not isinstance(src_raw, torch.Tensor)
                else src_raw.to(device).long(),
                torch.tensor(dst_raw, dtype=torch.long, device=device)
                if not isinstance(dst_raw, torch.Tensor)
                else dst_raw.to(device).long(),
            ],
            dim=0,
        )

        n_neurons = len(params.nodes.time_const)
        src = edge_index[0]

        # Infer E/I from net outgoing weight sign per neuron
        sum_w = torch.zeros(n_neurons, device=device)
        sum_w.scatter_add_(0, src, W)
        is_excitatory = sum_w >= 0

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
    C=1.0,  # uF/cm^2
    # Leak
    g_L=0.3,  # mS/cm^2
    E_L=-54.387,  # mV — leak reversal potential
    # Sodium
    g_Na=120.0,  # mS/cm^2
    E_Na=50.0,  # mV — sodium reversal potential
    # Potassium
    g_K=36.0,  # mS/cm^2
    E_K=-77.0,  # mV — potassium reversal potential
    # Synaptic coupling (continuous, voltage-dependent)
    syn_tau=5.0,  # ms — synaptic activation time constant
    syn_slope=5.0,  # mV — sigmoid slope for presynaptic activation
    syn_v_half=-45.0,  # mV — sigmoid midpoint (allows subthreshold transmission)
    # External input
    I_bias=3.0,  # uA/cm^2 — tonic drive (depolarises to ~-44mV, subthreshold)
    stim_scale=50.0,  # uA/cm^2 per unit stimulus
    # Weight scaling (flyvis connectome weights calibrated for graded model)
    w_scale=2.0,  # global multiplier on connectome W for HH dynamics
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
    W: torch.Tensor = None  # (E,) effective synaptic weights

    @classmethod
    def from_defaults(
        cls,
        n_neurons: int,
        edge_index: torch.Tensor,
        W: torch.Tensor,
        device: torch.device | str = "cpu",
        overrides: dict | None = None,
    ) -> FlyVisHodgkinHuxleyODEParams:
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
    def from_flyvis_network(
        cls, net, device: torch.device | str = "cpu", overrides: dict | None = None
    ) -> FlyVisHodgkinHuxleyODEParams:
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
        edge_index = torch.stack(
            [
                torch.tensor(src_raw, dtype=torch.long, device=device)
                if not isinstance(src_raw, torch.Tensor)
                else src_raw.to(device).long(),
                torch.tensor(dst_raw, dtype=torch.long, device=device)
                if not isinstance(dst_raw, torch.Tensor)
                else dst_raw.to(device).long(),
            ],
            dim=0,
        )

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


@register_ode_params("zebrafish", "zebrafish_oculomotor", "zebrafish_known_ode", "zebrafish_oculomotor_known_ode")
@dataclass
class ZebrafishODEParams(ODEParamsBase):
    """Parameters for the zebrafish oculomotor linear integrator ODE.

    The oculomotor integrator converts brief saccade velocity commands from
    the brainstem into persistent firing rate changes that encode eye position.
    This is a neural integrator: the network accumulates transient inputs
    into a sustained internal state.

    Stimulus: I(t) * v_in, where I(t) is a scalar velocity command (saccade
    signal from brainstem motor planning circuits) and v_in is the input
    direction vector — a specific combination of leading eigenvectors of W
    that excites the network along its integration axis. Neurons with large
    |v_in| are the primary recipients of the velocity command; others respond
    only through recurrent connectivity.

    Ref: papers/Code_NN/Code_NN/nn_fig5_zebrafish_teacher.py
         simulate_series() line 172:
         r[i,:] = r[i-1,:] + dt*(W @ r[i-1,:] - r[i-1,:] + I[i-1]*v_in) / tau
    Ref: Beiran & Litwin-Kumar (2023) Fig 5g — Goldman lab ConnMatrix

    ODE: dr/dt = (-r + W @ r + I * v_in) / tau
    Linear (no nonlinearity). tau=1.0 fixed. dt=0.001.

    Edge params:
        edge_index: (2, E) source/destination indices
        W: (E,) sparse weights (from dense W scaled to spectral radius 0.9)

    Node params:
        v_in: (N,) input direction vector — combination of leading
              eigenvectors of W that defines the integration axis
        neuron_types: (N,) int cell type labels (10 categories from
                      Goldman lab: integ, Ibnm, Ibni, MO, axlm, axl,
                      vest, abdm, abdi, vspns)

    Scalars:
        tau: time constant (default 1.0, fixed — not learned)
        n_neurons: number of neurons (609)
    """

    edge_index: torch.Tensor = None  # (2, E)
    W: torch.Tensor = None  # (E,)
    v_in: torch.Tensor = None  # (N,) input direction vector
    neuron_types: torch.Tensor = None  # (N,) int type labels
    type_names: list = None  # unique type name strings
    tau: float = 1.0
    n_neurons: int = 0

    @classmethod
    def from_connectome(cls, datapath: str, device: torch.device | str = "cpu"):
        """Construct from Goldman lab MATLAB data.

        Ref: nn_fig5_zebrafish_teacher.py lines 64-179
        Uses load_zebrafish_connectome() from connconstr_data.py.
        """
        from connectome_gnn.generators.connconstr_data import (
            dense_to_sparse,
            load_zebrafish_connectome,
        )

        data = load_zebrafish_connectome(datapath)
        edge_index, W_sparse = dense_to_sparse(data["W"])
        N = data["N"]

        return cls(
            edge_index=edge_index.to(device),
            W=W_sparse.to(device),
            v_in=torch.tensor(data["v_in"], dtype=torch.float32, device=device),
            neuron_types=torch.tensor(data["neuron_type_labels"], dtype=torch.long, device=device),
            type_names=list(data["cell_type_names"]),
            tau=1.0,
            n_neurons=N,
        )

    @classmethod
    def from_pretrained(cls, datapath: str, device: torch.device | str = "cpu"):
        """Construct from pre-saved zebrafish.npz (output of teacher script).

        Ref: nn_fig5_zebrafish_teacher.py line 394
        """
        from connectome_gnn.generators.connconstr_data import (
            dense_to_sparse,
            load_zebrafish_pretrained,
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

        # Load neuron type labels from Goldman MATLAB data
        try:
            from connectome_gnn.generators.connconstr_data import load_zebrafish_connectome

            goldman_dir = os.path.join(datapath, "goldman_data")
            if not os.path.isdir(goldman_dir):
                goldman_dir = os.path.join(os.path.dirname(datapath), "goldman_data")
            zf_data = load_zebrafish_connectome(goldman_dir)
            neuron_types = torch.tensor(zf_data["neuron_type_labels"], dtype=torch.long, device=device)
            type_names = list(zf_data["cell_type_names"])
        except Exception:
            neuron_types = torch.zeros(N, dtype=torch.long, device=device)
            type_names = None

        return cls(
            edge_index=edge_index.to(device),
            W=W_sparse.to(device),
            v_in=torch.tensor(data["v_in"], dtype=torch.float32, device=device),
            neuron_types=neuron_types,
            type_names=type_names,
            tau=1.0,
            n_neurons=N,
        )

    def create_ode(self, device=None):
        from connectome_gnn.generators.connconstr_zebrafish_ode import ZebrafishODE

        return ZebrafishODE(ode_params=self, device=device)

    def get_dt(self):
        return 0.001  # Ref: simulate_series line 166

    def get_n_neurons(self):
        return self.n_neurons

    def get_n_frames(self, sim):
        # Use sim.n_frames directly — stimulus is generated to fill this length
        return sim.n_frames

    def generate_stimulus(self, n_frames, sim, device=None):
        """Returns per-neuron stimulus tensor (T, N).

        The zebrafish oculomotor integrator receives inputs from multiple
        brainstem populations: saccade burst neurons (horizontal velocity
        commands), vestibular neurons (head rotation signals), and tonic
        neurons (position-related). These arrive along different directions
        in neural state space.

        We generate K=4 independent temporally-correlated signals, each
        projected along a different eigenvector direction of W:
          - Channel 0: primary integration axis (v_in, eigvecs 0-2) — saccade commands
          - Channel 1: eigenvector 3-5 — vestibular / secondary axis
          - Channel 2: eigenvector 6-9 — tonic modulation
          - Channel 3: random direction — broad neuromodulatory input

        This produces stimulus with rank ~4 instead of rank 1, giving the
        GNN richer dynamics to learn from while remaining biologically sound
        (multiple input pathways are well documented in zebrafish oculomotor
        literature: Aksay et al. 2007, Miri et al. 2011, Joshua & Bhatt 2023).

        Ref: paper uses single I(t)*v_in (simulate_series line 172).
        We extend to multiple input channels for richer training data.
        """
        from connectome_gnn.generators.connconstr_data import load_zebrafish_connectome
        import numpy as np

        rng = np.random.RandomState(sim.seed)
        N = self.n_neurons

        # Load eigenvectors of W for input direction design
        # Ref: nn_fig5_zebrafish_teacher.py lines 176-178
        datapath = sim.connconstr_datapath
        try:
            data = load_zebrafish_connectome(datapath)
            W_dense = data["W"]
            y_eig, v1 = np.linalg.eig(W_dense)
            sort_idx = np.flip(np.argsort(np.real(y_eig)))
            v1 = np.real(v1[:, sort_idx])
        except Exception:
            # Fallback: use v_in as sole direction
            v1 = np.zeros((N, 10))
            v1[:, 0] = self.v_in.cpu().numpy()

        # Build K=4 spatial input directions
        v_in_np = self.v_in.cpu().numpy()
        directions = np.zeros((N, 4))
        directions[:, 0] = v_in_np  # primary axis (eigvecs 0-2 + noise, from paper)
        directions[:, 1] = np.sum(v1[:, 3:6], axis=1)  # secondary axis
        directions[:, 2] = np.sum(v1[:, 6:10], axis=1)  # tertiary axis
        directions[:, 3] = rng.randn(N)  # broad neuromodulatory

        # Normalize each direction to match v_in magnitude
        v_in_norm = np.linalg.norm(v_in_np)
        for k in range(1, 4):
            dk_norm = np.linalg.norm(directions[:, k])
            if dk_norm > 1e-12:
                directions[:, k] *= v_in_norm / dk_norm

        # Generate K independent temporally-correlated signals.
        # Short correlation times (~100-300 frames) create fast transients
        # that the integrator must track, producing richer temporal dynamics.
        K = 4
        signal_taus = [200, 100, 300, 150]  # fast dynamics for rank
        amplitudes = [1.0, 0.7, 0.5, 0.3]  # more balanced across channels

        stim = np.zeros((n_frames, N), dtype=np.float32)
        for k in range(K):
            tau_k = signal_taus[k]
            filt = np.exp(-np.arange(tau_k * 3) / tau_k)
            filt /= filt.sum()
            noise = rng.randn(n_frames + len(filt))
            I_k = np.convolve(noise, filt, mode="full")[:n_frames]

            I_k *= amplitudes[k] * 800
            stim += I_k[:, None] * directions[None, :, k]  # (T,1) * (1,N)

        return torch.tensor(stim, dtype=torch.float32, device=device)

    def init_state(self, voltage, datapath=None, device=None):
        pass  # zero init is fine

    def get_trial_length(self):
        # Diff from paper repo: paper uses simulate_series() which processes one
        # pulse at a time (nn_fig5_zebrafish_teacher.py line 165). We generate
        # continuous trajectories — no trial resets.
        return 0

    # --- Analysis interface ---
    def has_tau(self):
        return False  # fixed tau=1

    def has_vrest(self):
        return False

    def gt_g_phi_func(self, v):
        return v  # identity — linear ODE, no activation

    def g_phi_label(self):
        return r"$v_j$ (identity)"

    def f_theta_label(self):
        return r"$-v_i / \tau$  ($\tau\!=\!1$)"

    def gt_f_theta_func(self, v, n_neurons):
        return -v * np.ones((n_neurons, 1))  # f(v) = -v/tau = -v

    def f_theta_param_names(self):
        return []  # no recoverable per-neuron params

    def clustering_features(self):
        return ["a", "W"]

    def neuron_type_rmse_panels(self):
        return ["weights"]


# ---------------------------------------------------------------------------
# Drosophila adult central complex ring attractor (Beiran & Litwin-Kumar 2023, Fig 5)
# ---------------------------------------------------------------------------


@register_ode_params("drosophila_cx", "drosophila_cx_rnn", "drosophila_cx_mlp", "drosophila_cx_known_ode")
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

    edge_index: torch.Tensor = None  # (2, E)
    W: torch.Tensor = None  # (E,)
    g: torch.Tensor = None  # (N,) log gain
    b: torch.Tensor = None  # (N,) bias
    h0: torch.Tensor = None  # (N,) initial state
    tau_raw: torch.Tensor = None  # (N,) raw time constant
    neuron_types: torch.Tensor = None  # (N,) int type labels
    winp: torch.Tensor = None  # (input_size, N)
    wout: torch.Tensor = None  # (N, output_size)
    type_names: list = None  # unique type name strings
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
        from connectome_gnn.generators.connconstr_data import (
            dense_to_sparse,
            load_drosophila_cx_connectome,
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
            type_names=data["type_names"],
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
        from connectome_gnn.generators.connconstr_data import (
            dense_to_sparse,
            load_drosophila_cx_connectome,
        )

        # Prefer the .pt state dict (has taus), fallback to .npz
        pt_path = os.path.join(datapath, "netPopVec_Wrec_simplering.pt")
        npz_path = os.path.join(datapath, "params_netSimpleRing2_final.npz")

        if os.path.exists(pt_path):
            # Load full state dict — has all trained params including taus
            sd = torch.load(pt_path, map_location="cpu", weights_only=False)
            wrec_t = sd["wrec"].numpy()
            mwrec_t = sd["mwrec"].numpy()
            JJ = np.exp(wrec_t) * mwrec_t  # effective J (line 184)
            g = sd["g"].numpy().flatten()  # log gain (ODE uses exp(g))
            bb = sd["b"].numpy().flatten()
            hh0 = sd["h0"].numpy().flatten()
            tau_raw = sd["taus"].numpy().flatten()
            wI = sd["wi"].numpy()  # (48, N)
            si_ = sd["si"].numpy()  # (48, 1)
            wOut = sd["wout"].numpy()  # (N, 49)
            alpha_ = 0.2  # Ref: RNN.__init__ default alpha=0.2
            N = JJ.shape[0]
        elif os.path.exists(npz_path):
            AA = np.load(npz_path)
            JJ = AA["arr_0"]  # effective J
            gg = AA["arr_1"]  # exp(g), already exponentiated
            bb = AA["arr_2"]  # bias
            hh0 = AA["arr_3"]  # initial state
            wI = AA["arr_4"]  # (48, N) input weights
            wOut = AA["arr_5"]  # (N, 49) output weights
            alpha_ = float(AA["arr_6"])
            si_ = AA["arr_7"]  # (48, N) or (48, 1) input scaling
            N = JJ.shape[0]
            g = np.log(np.maximum(gg, 1e-12))  # log it back
            tau_raw = np.zeros(N, dtype=np.float32)  # default
        else:
            raise FileNotFoundError(
                f"CX pretrained not found at {pt_path} or {npz_path}\nRun nn_fig5_drosophilaCx_teacher.py first."
            )

        edge_index, W_sparse = dense_to_sparse(JJ)

        # Load neuron type labels from hemibrain data
        hemibrain_dir = datapath
        if not os.path.exists(os.path.join(datapath, "traced-neurons.csv")):
            hemibrain_dir = os.path.join(datapath, "exported-traced-adjacencies-v1.2")
        try:
            cx_data = load_drosophila_cx_connectome(hemibrain_dir)
            neuron_types = torch.tensor(cx_data["neuron_types"], dtype=torch.long, device=device)
            type_names = cx_data["type_names"]
        except (FileNotFoundError, Exception):
            neuron_types = torch.zeros(N, dtype=torch.long, device=device)
            type_names = None

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
            type_names=type_names,
            winp=torch.tensor(winp_effective, dtype=torch.float32, device=device),
            wout=torch.tensor(wOut, dtype=torch.float32, device=device),
            alpha=alpha_,
            beta=5.0,
            noise_std=0.005,  # Ref: RNN.__init__ noise_std=0.005
            n_neurons=N,
        )

    def create_ode(self, device=None):
        from connectome_gnn.generators.connconstr_cx_ode import DrosophilaCxODE

        return DrosophilaCxODE(ode_params=self, device=device)

    def get_dt(self):
        return 0.1  # Ref: teacher training dt

    def get_n_neurons(self):
        return self.n_neurons

    def get_n_frames(self, sim):
        # Use sim.n_frames directly — stimulus is generated to fill this length
        return sim.n_frames

    def generate_stimulus(self, n_frames, sim, device=None):
        """Returns per-neuron stimulus tensor (T, N)."""
        from connectome_gnn.generators.connconstr_data import (
            generate_cx_stimulus,
            load_drosophila_cx_connectome,
        )
        from connectome_gnn.utils import to_numpy

        # Accept either parent dir or hemibrain subdir
        hemibrain_dir = sim.connconstr_datapath
        if not os.path.exists(os.path.join(hemibrain_dir, "traced-neurons.csv")):
            hemibrain_dir = os.path.join(hemibrain_dir, "exported-traced-adjacencies-v1.2")
        cx_data = load_drosophila_cx_connectome(hemibrain_dir)
        cx_inps = generate_cx_stimulus(
            n_frames,
            cx_data["epg_ix"],
            cx_data["W_16to46"],
            seed=sim.seed,
        )
        winp_np = to_numpy(self.winp)
        stim_projected = cx_inps @ winp_np
        # Scale to produce activity in [-10, 10] range (baseline gives ±4)
        return torch.tensor(2.5 * stim_projected, dtype=torch.float32, device=device)

    def init_state(self, voltage, datapath=None, device=None):
        if self.h0 is not None:
            voltage[:] = self.h0.clone()

    def get_trial_length(self):
        # Diff from paper repo: paper uses trial_len=60 (6s / dt=0.1) with state
        # reset at each trial boundary (nn_fig5_drosophilaCx_teacher.py generate_targets).
        # Trial resets are an artifact of the training procedure, not biologically
        # realistic — real neural circuits do not reset their state periodically.
        return 0

    # --- Analysis interface ---
    def has_tau(self):
        return True

    def has_vrest(self):
        return False

    def has_gain(self):
        return True

    def has_bias(self):
        return True

    def gt_tau(self, n_neurons):
        """CX tau = 2.6 + 2.4 * tanh(tau_raw), bounded [0.2, 5.0]."""
        if self.tau_raw is None:
            return None
        tau = 2.6 + 2.4 * np.tanh(self.tau_raw[:n_neurons].cpu().numpy())
        return tau

    def gt_vrest(self, n_neurons):
        return None  # CX ODE has no resting potential term

    def gt_gain(self, n_neurons):
        """CX gain = exp(g) per neuron."""
        if self.g is None:
            return None
        return np.exp(self.g[:n_neurons].cpu().numpy())

    def gt_bias(self, n_neurons):
        """CX bias b in softplus(v + b)."""
        if self.b is None:
            return None
        return self.b[:n_neurons].cpu().numpy()

    def derive_tau(self, slopes_f_theta, n_neurons):
        """CX f_theta slope = -alpha/tau → tau = alpha/(-slope)."""
        slopes = slopes_f_theta[:n_neurons]
        alpha = self.alpha
        derived = np.where(slopes != 0, alpha / -slopes, 1.0)
        return np.clip(derived, 0, 10)

    def gt_f_theta_func(self, v, n_neurons):
        """Ground truth f_theta(v) = -alpha * v / tau per neuron."""
        tau = self.gt_tau(n_neurons)
        if tau is None:
            return None
        return -self.alpha * v / tau[:, None]

    def gt_g_phi_func(self, v):
        """Ground truth g_phi(v) = exp(g) * softplus(v + b, beta=5) per neuron.
        Returns (n_neurons, n_pts) array.

        Args:
            v: (n_pts,) shared range or (N, n_pts) per-neuron ranges.
        """
        if self.g is None or self.b is None:
            return np.maximum(v, 0.0)  # fallback to ReLU
        g_np = self.g.cpu().numpy()
        b_np = self.b.cpu().numpy()
        beta = self.beta
        # softplus(x, beta) = (1/beta) * log(1 + exp(beta * x))
        if v.ndim == 1:
            x = v[None, :] + b_np[:, None]  # (N, n_pts)
        else:
            x = v + b_np[:, None]  # (N, n_pts) + (N, 1) = (N, n_pts)
        sp = np.where(beta * x > 20, x, np.log1p(np.exp(beta * x)) / beta)
        return np.exp(g_np[:, None]) * sp

    def g_phi_label(self):
        return r"$e^{g_j} \, \mathrm{softplus}(v_j + b_j, \beta\!=\!5)$"

    def f_theta_label(self):
        return r"$-\alpha \, v_i / \tau_i$"

    def f_theta_param_names(self):
        return [r"$\tau_i$"]

    def fit_g_phi_curves(self, v_ranges, learned_curves):
        """Fit learned g_phi to A * softplus(v + b, beta=5) per neuron.

        Gain A is entangled with W (the GNN can split amplitude arbitrarily
        between g_phi and W), so only bias b is a meaningful shape parameter.
        The correction factor is the gain A — used to form the effective weight
        W_eff = W_learned * A_j, compared against W_true * exp(g_true_j).

        Args:
            v_ranges: (N, n_pts) per-neuron voltage grids.
            learned_curves: (N, n_pts) learned g_phi output per neuron.

        Returns dict with:
            correction: (N,) gain A — multiplied INTO W to form effective weight
            gain: (N,) fitted amplitude (entangled with W, not reported as R²)
            bias: (N,) fitted bias b ≈ b_true (shape param, disentangled)
        """
        from scipy.optimize import curve_fit

        beta = self.beta
        n_neurons = learned_curves.shape[0]
        gains = np.ones(n_neurons)
        biases = np.zeros(n_neurons)

        def _softplus_model(v, A, b):
            x = beta * (v + b)
            return A * np.where(x > 20, v + b, np.log1p(np.exp(x)) / beta)

        for j in range(n_neurons):
            v_j = v_ranges[j]
            y = learned_curves[j]
            A0 = max(np.max(np.abs(y)), 0.1)
            try:
                popt, _ = curve_fit(_softplus_model, v_j, y, p0=[A0, 0.0], maxfev=2000)
                gains[j] = popt[0]
                biases[j] = popt[1]
            except (RuntimeError, ValueError):
                gains[j] = A0
                biases[j] = 0.0

        correction = gains.copy()
        correction[np.abs(correction) < 1e-8] = 1.0
        return {"correction": correction, "gain": gains, "bias": biases}

    def gt_g_phi_params(self, n_neurons):
        """Ground truth bias b for R² comparison.
        Gain is entangled with W — only bias is a meaningful shape parameter."""
        if self.b is None:
            return None
        b_np = self.b[:n_neurons].cpu().numpy()
        return {"bias": b_np}

    def g_phi_param_names(self):
        return ["bias"]

    def effective_true_weights(self, gt_weights, edges, n_neurons):
        """True effective weight = W_true * exp(g_src).

        The GNN's corrected_W = W_learned * A_src ≈ W_true * exp(g_src),
        so we compare against W_true * exp(g_src) rather than bare W_true.
        """
        if self.g is None:
            return gt_weights
        g_np = self.g[:n_neurons].cpu().numpy()
        src = edges[0]
        return gt_weights * np.exp(g_np[src])

    def clustering_features(self):
        return ["a", "τ", "W", "(a,τ,W)"]

    def neuron_type_rmse_panels(self):
        return ["weights", "tau"]


# ---------------------------------------------------------------------------
# Drosophila larva two-population model (Beiran & Litwin-Kumar 2023, Fig 5)
# ---------------------------------------------------------------------------


@register_ode_params("larva", "larva_known_ode")
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

    edge_index: torch.Tensor = None  # (2, E)
    W: torch.Tensor = None  # (E,)
    gp: torch.Tensor = None  # (N,)
    gm: torch.Tensor = None  # (M,)
    bp: torch.Tensor = None  # (N,)
    bm: torch.Tensor = None  # (M,)
    wsp: torch.Tensor = None  # (S, N) stimulus→premotor
    neuron_types: torch.Tensor = None  # (N+M,)
    type_names: list = None  # ["premotor", "motor"]
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
        from connectome_gnn.generators.connconstr_data import (
            dense_to_sparse,
            load_larva_connectome,
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
        neuron_types = torch.cat(
            [
                torch.zeros(N, dtype=torch.long),
                torch.ones(M, dtype=torch.long),
            ]
        ).to(device)

        return cls(
            edge_index=edge_index.to(device),
            W=W.to(device),
            gp=gp,
            gm=gm,
            bp=bp,
            bm=bm,
            wsp=torch.zeros(2, N, dtype=torch.float32, device=device),
            neuron_types=neuron_types,
            type_names=["premotor", "motor"],
            taup=1.0,
            taum=1.0,
            n_premotor=N,
            n_motor=M,
            dt=0.05,  # Ref: setup.py line 224,227
        )

    @classmethod
    def from_pretrained(cls, datapath: str, device: torch.device | str = "cpu"):
        """Construct from pre-trained ashokF_softplus.npz parameters.

        Ref: nn_fig5_plots_abc.py lines 31-41
        """
        from connectome_gnn.generators.connconstr_data import (
            dense_to_sparse,
            load_larva_pretrained,
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

        neuron_types = torch.cat(
            [
                torch.zeros(N, dtype=torch.long),
                torch.ones(M, dtype=torch.long),
            ]
        ).to(device)

        return cls(
            edge_index=edge_index.to(device),
            W=W.to(device),
            gp=torch.tensor(data["gp"].flatten(), dtype=torch.float32, device=device),
            gm=torch.tensor(data["gm"].flatten(), dtype=torch.float32, device=device),
            bp=torch.tensor(data["bp"].flatten(), dtype=torch.float32, device=device),
            bm=torch.tensor(data["bm"].flatten(), dtype=torch.float32, device=device),
            wsp=torch.tensor(data["wsp"], dtype=torch.float32, device=device),
            neuron_types=neuron_types,
            type_names=["premotor", "motor"],
            taup=float(data["taup"].item() if hasattr(data["taup"], "item") else data["taup"]),
            taum=float(data["taum"].item() if hasattr(data["taum"], "item") else data["taum"]),
            n_premotor=N,
            n_motor=M,
            dt=0.05,  # Ref: setup.py line 224,227
        )

    def create_ode(self, device=None):
        from connectome_gnn.generators.connconstr_larva_ode import LarvaODE

        return LarvaODE(ode_params=self, device=device)

    def get_dt(self):
        return self.dt

    def get_n_neurons(self):
        return self.n_premotor + self.n_motor

    def get_n_frames(self, sim):
        # Use sim.n_frames directly — stimulus is generated to fill this length
        return sim.n_frames

    def generate_stimulus(self, n_frames, sim, device=None):
        """Returns per-neuron stimulus tensor (T, N_total).

        Generates biologically realistic per-segment peristaltic stimulus
        for the Drosophila larva locomotor circuit.

        ===================================================================
        BIOLOGICAL BASIS — LITERATURE CONSENSUS
        ===================================================================

        1. AUTONOMOUS CPG WITH SENSORY MODULATION
           The larval VNC contains a CPG that generates fictive locomotion
           autonomously — rhythmic motor patterns persist in isolated nerve
           cords without sensory input or descending commands.
           - Pulver et al. 2015, J Neurophysiol: "Imaging fictive locomotor
             patterns in larval Drosophila." Isolated CNS produces three
             distinct motor patterns measured from Ca²⁺ signals.
           - Clark et al. 2018, Neural Development: "Neural circuits driving
             larval locomotion." CPG in thoracic/abdominal VNC segments
             generates forward and backward peristaltic waves.

        2. SEGMENTAL WAVE PROPAGATION
           Crawling is produced by peristaltic waves that propagate segment
           by segment. Excitatory premotor interneurons (CLI1, CLI2) are
           activated sequentially from posterior to anterior segments,
           with intersegmental delays of ~0.5-1s per segment.
           - Hasegawa et al. 2016, Sci Reports: CLI1/CLI2 "activated
             sequentially from posterior to anterior segments during
             peristalsis" and "directly activate motoneurons."
           - Fushiki et al. 2016, eLife: A27h premotor neurons drive
             segment-by-segment wave propagation via A27h→GDL feed-forward
             inhibition to the next anterior segment.

        3. INTERSEGMENTAL FEEDBACK
           Two pairs of direction-specific feedback interneurons (Ifb-Fwd,
           Ifb-Bwd) provide intersegmental excitation during forward vs
           backward crawling, targeting shared premotor interneurons.
           - Kohsaka et al. 2019, Nature Comms: "Regulation of forward and
             backward locomotion through intersegmental feedback circuits."
             Ifb-Fwd active only during forward crawling, Ifb-Bwd only
             during backward.

        4. DESCENDING DIRECTION COMMANDS
           The MDN (moonwalker descending neuron) switches locomotion
           direction. MDN activation induces backward crawling and inhibits
           forward crawling via the Pair1 interneuron, which inhibits A27h
           (the forward-specific premotor neuron).
           - Carreira-Rosario et al. 2018 (via Clark et al. 2018 review):
             MDN→Pair1→A27h circuit controls forward/backward switching.

        5. PROPRIOCEPTIVE MODULATION
           Class I md sensory neurons (vpda, ddaD, dmd1) provide segment-
           local proprioceptive feedback that modulates CPG timing. This
           feedback is phase-locked to the ongoing rhythm.
           - Vaadia et al. 2019, Frontiers: "The Drosophila Larval
             Locomotor Circuit." GDL interneurons receive proprioceptive
             input from Vpda stretch receptors.

        6. WHOLE-CNS IMAGING CONFIRMS LOW-DIMENSIONAL WAVE STRUCTURE
           Light-sheet calcium imaging of the entire larval CNS shows that
           activity during fictive locomotion is dominated by propagating
           waves — inherently low-dimensional (rank ~3-8 across 4 segments).
           - Lemon et al. 2015, Nature Comms: "Whole-central nervous system
             functional imaging in larval Drosophila." 20,000 volumes at
             5 Hz covering brain and VNC.

        7. PAPER'S ORIGINAL STIMULUS (Beiran & Litwin-Kumar 2025)
           The connconstr paper uses S=2 stimulus channels (forward/backward
           square pulses) projected via learned wsp (2×N) matrix. This is
           faithful to the binary descending command, but produces rank-2
           stimulus insufficient for GNN benchmarking.
           - Beiran & Litwin-Kumar 2025, Nature Neuroscience: "Prediction
             of neural activity in connectome-constrained recurrent networks."
             setup.py gentargets(): two square-pulse conditions.

        ===================================================================
        OUR STIMULUS DESIGN
        ===================================================================

        We extend the paper's 2-channel design to capture the segmental
        wave structure documented in the literature:

        1. Temporally correlated descending command — smooth stochastic
           drive (represents summed input from MDN, Pair1, and other
           descending neurons)
        2. Phase-shifted per segment — each of the 4 VNC segments (t3, a1,
           a2, a3) receives the command delayed by ~0.75s, mimicking the
           peristaltic wave propagation (Hasegawa et al. 2016, Fushiki 2016)
        3. Direction modulation — slowly alternating forward/backward
           reverses the segment phase order, as mediated by Ifb-Fwd/Ifb-Bwd
           (Kohsaka et al. 2019) and MDN/Pair1 switching
        4. Second independent command — captures parallel descending
           pathways (e.g. speed modulation via PMSIs, Kohsaka et al. 2014)
        5. Per-neuron wsp scaling — preserves the learned excitatory/
           inhibitory input structure from the teacher model

        Result: stimulus rank ≈ 3-5 (4 segments × 2 commands × direction),
        consistent with the low-dimensional but spatially structured
        traveling waves observed in whole-CNS imaging.

        Motor neurons receive no direct stimulus (driven by premotor→motor
        connections, matching the biological architecture).
        """
        import re

        rng = np.random.RandomState(sim.seed)
        N_total = self.n_premotor + self.n_motor
        N_pre = self.n_premotor

        # --- Step 0: Segmental organization ---
        # The larva VNC is organized into neuromeres (t3, a1, a2, a3), each
        # containing distinct sets of premotor interneurons.
        # Ref: Clark et al. 2018 Fig 1; Zarin et al. 2019 (TEM reconstruction
        # of all 60 MNs and 236 PMNs in segments A1-A2)
        seg_indices = self._get_segment_indices(sim.connconstr_datapath)
        n_segments = int(seg_indices.max()) + 1  # 4 segments: t3, a1, a2, a3

        # --- Step 1: Base descending command (temporally correlated noise) ---
        # Models the summed descending drive from brain to VNC.
        # The CPG is autonomous (Pulver et al. 2015) but modulated by
        # descending neurons including MDN and other command neurons
        # (Clark et al. 2018, Section "Higher-order control").
        # Correlation time ~3s matches the timescale of crawling bouts.
        signal_tau = 60  # frames (3s at dt=0.05)
        signal_filter = np.exp(-np.arange(signal_tau * 4) / signal_tau)
        signal_filter /= signal_filter.sum()

        noise = rng.randn(n_frames + len(signal_filter))
        base_signal = np.convolve(noise, signal_filter, mode="full")[:n_frames]

        # Slow amplitude envelope — models episodic locomotion bouts
        # Ref: Lemon et al. 2015 observed epochs of fictive locomotion
        # interspersed with quiescent periods in whole-CNS imaging.
        env_tau = 120  # frames (6s at dt=0.05)
        env_filter = np.exp(-np.arange(env_tau * 4) / env_tau)
        env_filter /= env_filter.sum()
        env_noise = rng.randn(n_frames + len(env_filter))
        envelope = np.convolve(env_noise, env_filter, mode="full")[:n_frames]
        envelope = (envelope - envelope.min()) / (envelope.max() - envelope.min() + 1e-12)
        base_signal *= envelope

        # --- Step 2: Phase-shifted per-segment signals ---
        # During crawling, peristaltic waves propagate segment-by-segment
        # with intersegmental delays of ~0.5-1s.
        # Ref: Pulver et al. 2015 — "linear scaling of intersegmental delay
        #      with cycle period appears to be a core feature of the larval
        #      locomotor CPG." Ca²⁺ waves travel posterior→anterior (forward)
        #      or anterior→posterior (backward).
        # Ref: Hasegawa et al. 2016 — CLI1/CLI2 premotor neurons "activated
        #      sequentially from posterior to anterior segments."
        # Ref: Fushiki et al. 2016 — A27h→GDL feed-forward inhibition creates
        #      segment-by-segment delay in wave propagation.
        segment_delay = 15  # frames (~0.75s at dt=0.05)

        per_segment = np.zeros((n_frames, n_segments), dtype=np.float32)
        for seg in range(n_segments):
            delay = seg * segment_delay
            per_segment[:, seg] = np.roll(base_signal, delay)
            if delay > 0:
                per_segment[:delay, seg] = 0

        # --- Step 3: Direction modulation (forward ↔ backward) ---
        # The MDN/Pair1 circuit switches locomotion direction. MDN activates
        # Pair1, which inhibits A27h to suppress forward crawling and enable
        # backward crawling. Direction reversal flips the segment phase order.
        # Ref: Kohsaka et al. 2019, Nature Comms — Ifb-Fwd and Ifb-Bwd are
        #      "differentially active during either forward or backward
        #      locomotion" and "commonly target a group of premotor
        #      interneurons."
        # Ref: Vaadia et al. 2019, Frontiers — MDN→Pair1→A27h circuit.
        # Correlation time ~10s: larvae alternate direction every few seconds.
        dir_tau = 200  # frames (10s at dt=0.05)
        dir_filter = np.exp(-np.arange(dir_tau * 4) / dir_tau)
        dir_filter /= dir_filter.sum()
        dir_noise = rng.randn(n_frames + len(dir_filter))
        direction = np.convolve(dir_noise, dir_filter, mode="full")[:n_frames]
        # Soft sign: +1 = forward (posterior→anterior), -1 = backward
        direction = np.tanh(direction * 2)

        # Forward: segments activated t3→a1→a2→a3 (anterior to posterior)
        # Backward: reversed order a3→a2→a1→t3
        per_segment_bwd = per_segment[:, ::-1].copy()

        # Smooth blend between forward and backward phase orders
        fwd_weight = (1 + direction[:, None]) / 2  # [0, 1]
        bwd_weight = 1 - fwd_weight
        per_segment_mixed = fwd_weight * per_segment + bwd_weight * per_segment_bwd

        # --- Step 4: Second independent descending command ---
        # Models a parallel descending pathway, e.g. speed modulation.
        # Ref: Kohsaka et al. 2014, Current Biology — PMSIs "regulate the
        #      speed of axial locomotion" by limiting motor burst duration,
        #      operating on a separate timescale from direction commands.
        # Also captures multi-modal sensory modulation that converges on
        # premotor neurons (mechanosensory, thermosensory pathways).
        # Ref: Vaadia et al. 2019 — "Basin neurons integrate nociceptive
        #      and mechanoreceptive inputs."
        noise2 = rng.randn(n_frames + len(signal_filter))
        base_signal2 = np.convolve(noise2, signal_filter, mode="full")[:n_frames]
        env_noise2 = rng.randn(n_frames + len(env_filter))
        envelope2 = np.convolve(env_noise2, env_filter, mode="full")[:n_frames]
        envelope2 = (envelope2 - envelope2.min()) / (envelope2.max() - envelope2.min() + 1e-12)
        base_signal2 *= envelope2

        # This second command also propagates as a traveling wave
        per_segment2 = np.zeros((n_frames, n_segments), dtype=np.float32)
        for seg in range(n_segments):
            delay = seg * segment_delay
            per_segment2[:, seg] = np.roll(base_signal2, delay)
            if delay > 0:
                per_segment2[:delay, seg] = 0

        # --- Step 5: Map per-segment signals to per-neuron via wsp ---
        # The learned wsp matrix (2, N_premotor) from Beiran & Litwin-Kumar
        # 2025 defines how each premotor neuron integrates the two command
        # channels. This preserves the excitatory/inhibitory input structure
        # that the teacher network learned.
        # Ref: setup.py line 33 — torch.matmul(s[t,:,:], wsp)
        wsp_np = self.wsp.cpu().numpy()  # (S=2, N_premotor)

        stim_premotor = np.zeros((n_frames, N_pre), dtype=np.float32)
        for ni in range(N_pre):
            seg = seg_indices[ni]
            # Channel 0 × direction-modulated segment wave
            # Channel 1 × second independent segment wave
            stim_premotor[:, ni] = wsp_np[0, ni] * per_segment_mixed[:, seg] + wsp_np[1, ni] * per_segment2[:, seg]

        # Motor neurons receive no direct stimulus — they are driven
        # exclusively by premotor→motor connections (Jpm).
        # Ref: setup.py line 32 — motor neuron eq has no stimulus term.
        stim_all = torch.zeros(n_frames, N_total, dtype=torch.float32, device=device)
        # Scale factor: paper's effective stimulus std ≈ 4.9 per neuron.
        # 5.0 matches paper training amplitude for meaningful dynamic rank.
        stim_all[:, :N_pre] = torch.tensor(5.0 * stim_premotor, dtype=torch.float32, device=device)

        return stim_all

    def _get_segment_indices(self, datapath):
        """Extract segment index per premotor neuron from h5 neuron names.

        Segments: t3=0, a1=1, a2=2, a3=3 (anterior to posterior).
        """
        import re

        try:
            from connectome_gnn.generators.connconstr_data import load_larva_connectome

            data = load_larva_connectome(datapath)
            pnames = [p.decode() if isinstance(p, bytes) else str(p) for p in data["pnames"]]
        except Exception:
            # Fallback: assign uniform segment index
            return np.zeros(self.n_premotor, dtype=int)

        seg_order = {"t3": 0, "a1": 1, "a2": 2, "a3": 3}
        indices = np.zeros(len(pnames), dtype=int)
        for i, name in enumerate(pnames):
            m = re.search(r"_([at]\d+)$", name)
            if m:
                indices[i] = seg_order.get(m.group(1), 0)
        return indices

    def get_trial_length(self):
        # Diff from paper repo: paper uses trial_len=240 (2 conditions × 6s / dt=0.05)
        # with state reset at each trial boundary (setup.py forwardpass).
        # Trial resets are an artifact of the training procedure, not biologically
        # realistic — real neural circuits do not reset their state periodically.
        return 0

    def init_state(self, voltage, datapath=None, device=None):
        try:
            from connectome_gnn.generators.connconstr_data import load_larva_pretrained

            pretrained = load_larva_pretrained(datapath)
            if "p0" in pretrained and pretrained["p0"] is not None:
                p0 = pretrained["p0"]
                if p0.ndim == 3:
                    p0 = p0[0, 0, :]
                voltage[: self.n_premotor] = torch.tensor(
                    p0.flatten()[: self.n_premotor], dtype=torch.float32, device=device
                )
            if "m0" in pretrained and pretrained["m0"] is not None:
                m0 = pretrained["m0"]
                if m0.ndim == 3:
                    m0 = m0[0, 0, :]
                voltage[self.n_premotor :] = torch.tensor(
                    m0.flatten()[: self.n_motor], dtype=torch.float32, device=device
                )
        except (FileNotFoundError, KeyError):
            pass

    # --- Analysis interface ---
    def has_tau(self):
        return True  # two distinct tau values (premotor/motor)

    def has_vrest(self):
        return False

    def has_gain(self):
        return True

    def has_bias(self):
        return True

    def gt_tau(self, n_neurons):
        """Premotor neurons have taup, motor neurons have taum."""
        tau = np.zeros(n_neurons)
        tau[: self.n_premotor] = self.taup
        tau[self.n_premotor : self.n_premotor + self.n_motor] = self.taum
        return tau[:n_neurons]

    def gt_gain(self, n_neurons):
        """Per-neuron gain: gp for premotor, gm for motor."""
        if self.gp is None or self.gm is None:
            return None
        gain = np.zeros(n_neurons)
        gain[: self.n_premotor] = self.gp[: self.n_premotor].cpu().numpy()
        gain[self.n_premotor : self.n_premotor + self.n_motor] = self.gm[: self.n_motor].cpu().numpy()
        return gain[:n_neurons]

    def gt_bias(self, n_neurons):
        """Per-neuron bias: bp for premotor, bm for motor."""
        if self.bp is None or self.bm is None:
            return None
        bias = np.zeros(n_neurons)
        bias[: self.n_premotor] = self.bp[: self.n_premotor].cpu().numpy()
        bias[self.n_premotor : self.n_premotor + self.n_motor] = self.bm[: self.n_motor].cpu().numpy()
        return bias[:n_neurons]

    def gt_g_phi_func(self, v):
        """Softplus activation (gain-modulated per neuron).
        Returns (N, n_pts) where N = n_premotor + n_motor.
        Premotor: gp_i * softplus(v), Motor: gm_i * softplus(v).

        Args:
            v: (n_pts,) shared range or (N, n_pts) per-neuron ranges.
        """
        sp = np.log1p(np.exp(v))  # same shape as v
        N = self.n_premotor + self.n_motor
        if self.gp is not None and self.gm is not None:
            gains = np.zeros(N)
            gains[: self.n_premotor] = self.gp.cpu().numpy()
            gains[self.n_premotor :] = self.gm.cpu().numpy()
            if sp.ndim == 1:
                return gains[:, None] * sp[None, :]  # (N, n_pts)
            else:
                return gains[:, None] * sp  # (N, 1) * (N, n_pts) = (N, n_pts)
        if sp.ndim == 1:
            return np.broadcast_to(sp, (N, len(v)))
        return sp

    def g_phi_label(self):
        return r"$g_i \, \mathrm{softplus}(v_j)$"

    def gt_f_theta_func(self, v, n_neurons):
        """Ground truth f_theta(v) = -v / tau per neuron. Shape: (n_neurons, n_pts)."""
        tau = self.gt_tau(n_neurons)
        if tau is None:
            return None
        return -v / tau[:, None]

    def f_theta_label(self):
        return r"$-v_i / \tau_i$"

    def f_theta_param_names(self):
        return [r"$\tau_i$"]

    def effective_true_weights(self, gt_weights, edges, n_neurons):
        """True effective weight = W_true * gain[dst].

        The larva ODE applies gain at the destination:
            dup/dt = (-up + gp_i * sum_j W[i,j] * softplus(v_j) + ...) / taup
            dum/dt = (-um + gm_i * sum_j W[i,j] * softplus(v_j) + ...) / taum

        The GNN can't separate destination gain from W, so the effective
        true weight for comparison is gp[dst] * W (premotor destinations)
        or gm[dst-N] * W (motor destinations).
        """
        if self.gp is None or self.gm is None:
            return gt_weights
        N = self.n_premotor
        gp_np = torch.clamp(self.gp, 0.5, 5.0).cpu().numpy()
        gm_np = torch.clamp(self.gm, 0.5, 5.0).cpu().numpy()
        # Build per-neuron gain array
        gains = np.zeros(n_neurons)
        gains[:N] = gp_np[: min(N, n_neurons)]
        gains[N:] = gm_np[: max(0, n_neurons - N)]
        dst = edges[1]
        return gt_weights * gains[dst]

    def clustering_features(self):
        return ["a", "τ", "W"]

    def neuron_type_rmse_panels(self):
        return ["weights", "tau"]
