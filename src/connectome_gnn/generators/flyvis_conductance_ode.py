"""Ground-truth ODE for the conductance-based flyvis twin.

The graded-voltage flyvis model (``flyvis_ode.FlyVisODE``) delivers synaptic
input as a *current*.  This module implements its conductance-based twin, in
which the same non-negative synaptic activation drives the voltage towards a
reversal potential:

    tau_i dv_i/dt = -(v_i - v_rest_i) + sum_j g_ij(v_j) (E_ij - v_i) + e_i
    g_ij(v_j)     = G_ij * relu(v_j)                              >= 0

with ``G`` the per-edge peak conductance and ``E`` the per-edge reversal
potential (excitatory or inhibitory, fixed by the connectome sign).  Writing
``G_tot_i = sum_j g_ij``, this is equivalently

    dv_i/dt   = (v_inf_i - v_i) / tau_eff_i
    v_inf_i   = (v_rest_i + e_i + sum_j g_ij E_ij) / (1 + G_tot_i)
    tau_eff_i = tau_i / (1 + G_tot_i)

so the model divisively normalizes its input, its membrane time constant
shortens with drive, and the voltage is confined to the convex hull of the
reversal potentials — free rollouts cannot diverge.

Like ``FlyVisODE`` this uses explicit scatter_add (no PyG) and has no flyvis
dependency: everything it needs lives in ``FlyVisConductanceODEParams``, which
can be built once from a flyvis network and then saved/loaded standalone.
"""

import torch
import torch.nn as nn

from connectome_gnn.generators.ode_params import FlyVisConductanceODEParams
from connectome_gnn.neuron_state import NeuronState


class FlyVisConductanceODE(nn.Module):
    """Ground-truth ODE for the conductance-based flyvis twin.

    args:
        ode_params: FlyVisConductanceODEParams (or a dict of its fields).
        g_phi: presynaptic release function, relu by default.
        delta_t: integration step of the caller.  The effective time constant
            is floored at this value so that forward Euler cannot overshoot the
            fixed point, matching flyvis' integration guard.  Leave at None to
            disable the floor.
        model_type: unused, accepted for signature parity with FlyVisODE.
    """

    def __init__(
        self,
        aggr_type="add",
        ode_params=None,
        params=[],
        g_phi=torch.nn.functional.relu,
        model_type=None,
        n_neuron_types=None,
        device=None,
        delta_t=None,
        integration="euler_floored",
    ):
        super().__init__()

        if integration not in ("euler_floored", "exponential"):
            raise ValueError(
                f"integration must be 'euler_floored' or 'exponential', got {integration!r}"
            )
        if isinstance(ode_params, dict):
            ode_params = FlyVisConductanceODEParams(**ode_params)
        self.ode_params = ode_params
        self.g_phi = g_phi
        self.model_type = model_type
        self.device = device
        self.delta_t = delta_t
        self.integration = integration

        if self.ode_params is not None:
            self.ode_params.to(device)

    def _compute_conductances(self, v, edge_index):
        """Aggregate synaptic conductance and conductance-weighted reversal drive.

        args:
            v: (N, 1) voltage
            edge_index: (2, E) source/destination indices

        returns:
            (G_tot, reversal_drive), each (N, 1)
        """
        src, dst = edge_index

        # per-edge conductance, non-negative because G >= 0 and g_phi >= 0
        g = self.ode_params.G[:, None] * self.g_phi(v[src])
        index = dst.unsqueeze(1).expand_as(g)

        total = torch.zeros(v.shape[0], g.shape[1], device=v.device, dtype=v.dtype)
        total.scatter_add_(0, index, g)

        drive = torch.zeros(v.shape[0], g.shape[1], device=v.device, dtype=v.dtype)
        drive.scatter_add_(0, index, g * self.ode_params.E_rev[:, None])

        return total, drive

    def forward(self, state: NeuronState, edge_index: torch.Tensor, has_field=False, data_id=[]):
        """Compute dv/dt from neuron state and connectivity.

        args:
            state: NeuronState with voltage and stimulus fields
            edge_index: (2, E) tensor of (src, dst) edge indices

        returns:
            dv: (N, 1) voltage derivative
        """
        v = state.voltage.unsqueeze(-1)
        v_rest = self.ode_params.V_i_rest[:, None]
        e = state.stimulus.unsqueeze(-1)
        tau = self.ode_params.tau_i[:, None]

        total, drive = self._compute_conductances(v, edge_index)

        leak = 1.0 + total
        v_inf = (v_rest + e + drive) / leak
        tau_eff = tau / leak

        if self.integration == "exponential" and self.delta_t is not None:
            # Exact relaxation over one step with the coefficients held fixed:
            #   v(t+dt) = v_inf + (v - v_inf) exp(-dt/tau_eff)
            # returned as the derivative that reproduces it under the caller's
            # forward-Euler update, v + dt*dv. Unconditionally stable for any dt
            # without distorting tau_eff, and it reduces to (v_inf - v)/tau_eff
            # as dt -> 0.
            decay = -torch.expm1(-self.delta_t / tau_eff)  # 1 - exp(-dt/tau_eff)
            return (v_inf - v) * decay / self.delta_t

        if self.delta_t is not None:
            # Floor tau_eff so a forward-Euler step cannot overshoot the fixed
            # point. This is a property of the integrator, not of the model: it
            # makes the simulated dynamics mildly dt-dependent wherever it binds.
            # Use integration="exponential" to avoid it entirely.
            tau_eff = torch.clamp(tau_eff, min=self.delta_t)

        return (v_inf - v) / tau_eff

    def floor_binding_fraction(self, state: NeuronState, edge_index: torch.Tensor) -> float:
        """Fraction of neurons whose tau/(1+G) currently sits below delta_t.

        Where this is zero the floored and exponential integrators agree, and the
        simulated dynamics carry no dt dependence beyond ordinary Euler error.
        """
        if self.delta_t is None:
            return 0.0
        v = state.voltage.unsqueeze(-1)
        total, _ = self._compute_conductances(v, edge_index)
        tau_eff = self.ode_params.tau_i[:, None] / (1.0 + total)
        return float((tau_eff < self.delta_t).float().mean())

    def func(self, u, type, function):
        """Isolated release / update functions, for comparison against a fitted GNN."""
        if function == 'phi':
            return self.g_phi(u)
        elif function == 'update':
            v_rest = self.ode_params.V_i_rest[type]
            tau = self.ode_params.tau_i[type]
            return (-u + v_rest) / tau
