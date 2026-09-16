"""Conductance-based synapses as a flyvis NetworkDynamics.

THE ONE STRUCTURAL DIFFERENCE from flyvis's current-based `PPNeuronIGRSynapses`
is that the message depends on the POSTSYNAPTIC voltage:

    current-based   msg_ij = (sign_ij * N_ij * s_ij) * relu(v_j)
    conductance     msg_ij = (N_ij * s_ij) * relu(v_j) * (E_ij - v_i)

with N_ij the synapse count, s_ij the synapse strength, v_j the presynaptic
voltage and v_i the postsynaptic one. The synapse's SIGN leaves the weight
entirely: `syn_strength` already carries `clamp: non_negative` in flyvis's own
config, so the product is a conductance that cannot go negative, and the sign of
the message is carried by the driving force (E_ij - v_i) alone --

    excitatory   E_exc > v_i  ->  msg > 0
    inhibitory   E_inh < v_i  ->  msg < 0

A signed weight AND a signed driving force would multiply to the wrong sign on
inhibitory edges; the sign can live in one factor or the other, never both.

WHICH REVERSAL AN EDGE DRIVES TOWARD is decided by `params.edges.sign`, the
connectome's own Dale sign, which flyvis fixes per (source type, target type)
group with requires_grad False. Under the transmitter-to-sign rule of Davis et
al. 2020 (eLife 50901) the cholinergic synapses open a nicotinic cation channel
while GABAergic, glutamatergic and histaminergic ones all open chloride channels
-- three transmitters, two ion species -- so two reversal rows suffice, and the
sign selects between them.

EXPANDING THE MESSAGE SHOWS WHAT IS ACTUALLY NEW. Writing the sum out,

    sum_j g_ij relu(v_j) (E_ij - v_i) = [sum_j g_ij relu(v_j) E_ij]
                                        - v_i [sum_j g_ij relu(v_j)]

the second term adds to the leak, so the effective membrane time constant
SHORTENS with drive. That is shunting, and it is the dynamical content of the
change.

SHUNTING STABILISES THE ODE AND DESTABILISES ITS EXPLICIT DISCRETISATION -- the
two are opposite statements and only the second one kills a run. flyvis
integrates with forward Euler at a FIXED dt of 20 ms (network.py:426,
`state + vel * dt`), so with

    G_i = sum_j g_ij relu(v_j)     the total synaptic conductance onto neuron i,
                                   in the same units as the unit leak conductance

one forward-Euler step multiplies the deviation of v_i by
[1 - (dt / tau_i) (1 + G_i)], which is a contraction only while

    (dt / tau_i) (1 + G_i) < 2.

The current-based model has no G_i inside that bracket: its factor is dt / tau_i,
which the `max(tau_i, dt)` floor below holds at or under 1, so it cannot diverge
whatever the weights do. The conductance model can, and did -- run flow/2000/000
died with a NaN at iteration 16,368 after 16,000 iterations of a perfectly flat
loss, and at its last checkpoint 2,690 of the 45,669 neurons were already past
the factor of 2 at a presynaptic relu(v_j) of 1, against a recorded activity
maximum of 2.7.

THE STEP IS THEREFORE EXPONENTIAL EULER, not forward Euler. Over one step the
equation is linear in v_i, so it can be solved exactly at frozen coefficients;
see `write_state_velocity` for the three lines that do it. It is stable for any
G_i >= 0, it has the same fixed point and the same continuous-time limit, and it
costs one `expm1` per neuron per step.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from flyvis.network.dynamics import NetworkDynamics
from flyvis.utils.tensor_utils import AutoDeref, RefTensor

__all__ = ["ConductanceSynapses"]


def in_band(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Squash a free variable into [lo, hi]; constant when lo >= hi.

    A sigmoid rather than a clamp, because the bound then holds at every gradient
    step by algebra instead of by projection, and because a clamped parameter
    sitting on its boundary receives zero gradient and never leaves it.

    A ZERO-WIDTH BAND PINS THE ROW at `lo`. That is how a reversal biology fixes
    -- the cation one, say -- carries no free parameter at all rather than a
    parameter the transform silently ignores.
    """
    if hi <= lo:
        return torch.full_like(raw, lo)
    return lo + (hi - lo) * torch.sigmoid(raw)


class ConductanceSynapses(NetworkDynamics):
    """Passive point neurons with conductance-based graded release synapses.

    Attributes:
        INTEGRATION: how one timestep is taken, printed in the training banner. It
            is an attribute of the CLASS rather than a string in the trainer so the
            banner cannot claim a step the code does not take: a checkout without
            this fix has no such attribute and the banner says forward Euler, which
            is the one line in a cluster log that answers "did the job pick up the
            new code".


    Args:
        activation: presynaptic nonlinearity, as for `PPNeuronIGRSynapses`.
        exc_band: (lo, hi) excursions from rest for the cation reversal.
        inh_band: (lo, hi) excursions from rest for the chloride reversal.
        excursion: the voltage scale one excursion means.

    The reversals are anchored on the postsynaptic cell's own resting potential:

        E(i) = bias(i) + (lo + (hi - lo) * sigmoid(raw(i))) * excursion

    `bias` is flyvis's own learned resting potential, so the anchor CO-ADAPTS with
    training rather than being a constant measured beforehand. This is the one
    place the model departs from the teacher-student trainer, which measured the
    teacher's recorded voltage range instead: training on optic flow from scratch
    there is no recording, and the voltage range is an outcome of training rather
    than an input to it.
    """

    INTEGRATION = "exponential Euler, stable at any total conductance"

    def __init__(
        self,
        activation: dict = None,
        exc_band: Sequence[float] = (1.5, 6.0),
        inh_band: Sequence[float] = (-6.0, -0.5),
        excursion: float = 1.0,
    ) -> None:
        super().__init__(activation=dict(activation or {"type": "relu"}))
        self.exc_band = (float(exc_band[0]), float(exc_band[1]))
        self.inh_band = (float(inh_band[0]), float(inh_band[1]))
        self.excursion = float(excursion)

    def write_derived_params(
        self, params: AutoDeref[str, AutoDeref[str, RefTensor]], **kwargs
    ) -> None:
        """Conductance per edge, and the two reversals per node.

        Called once per forward pass, which is why the reversals are materialised
        HERE and at node level: 45,669 sigmoids once, rather than 1,513,231 at
        every one of the unrolled timesteps. `_param_api` gathers any new key of
        `params.nodes` onto `params.sources` and `params.targets` immediately
        after this returns (network.py:313), so writing `E_exc`/`E_inh` as node
        quantities is what makes `params.targets.E_inh` exist at all.
        """
        # The current-based model computes sign * syn_count * syn_strength here.
        # Dropping the sign is what turns the weight into a conductance.
        params.edges.conductance = params.edges.syn_count * params.edges.syn_strength

        rest = params.nodes.bias
        params.nodes.E_exc = (
            rest + in_band(params.nodes.E_exc_raw, *self.exc_band) * self.excursion
        )
        params.nodes.E_inh = (
            rest + in_band(params.nodes.E_inh_raw, *self.inh_band) * self.excursion
        )

    def write_initial_state(
        self,
        state: AutoDeref[str, AutoDeref[str, RefTensor]],
        params: AutoDeref[str, AutoDeref[str, RefTensor]],
        **kwargs,
    ) -> None:
        """Start at rest, exactly as the current-based model does."""
        state.nodes.activity = params.nodes.bias

    def reversal_per_edge(
        self, params: AutoDeref[str, AutoDeref[str, RefTensor]]
    ) -> torch.Tensor:
        """(n_edges,) the reversal each edge drives toward.

        `sign` is the PRESYNAPTIC cell's transmitter polarity while E is indexed
        by the POSTSYNAPTIC cell, so this is not a property of either endpoint
        alone -- the edge chooses the ion, the receiving cell supplies its value.
        Shared with the analysis pass, which reports E per edge against the
        activity range of the cell receiving it.
        """
        return torch.where(
            params.edges.sign < 0, params.targets.E_inh, params.targets.E_exc
        )

    def write_state_velocity(
        self,
        vel: AutoDeref[str, AutoDeref[str, RefTensor]],
        state: AutoDeref[str, AutoDeref[str, RefTensor]],
        params: AutoDeref[str, AutoDeref[str, RefTensor]],
        target_sum: Callable,
        x_t: torch.Tensor,
        dt: float,
        **kwargs,
    ) -> None:
        """The exponential-Euler step, written as the velocity flyvis will scale by dt.

        SPLITTING THE MESSAGE IS WHAT MAKES THIS POSSIBLE. The chemical current

            sum_j g_ij relu(v_j) (E_ij - v_i)  =  S_i - v_i G_i

        is affine in the postsynaptic voltage v_i, with

            G_i = sum_j g_ij relu(v_j)         total conductance onto neuron i,
                                               in units of its leak conductance
            S_i = sum_j g_ij relu(v_j) E_ij    that conductance weighted by the
                                               reversal each edge drives toward

        so over one step, holding the presynaptic activity and the input fixed,

            tau_i dv_i/dt = a_i (v_inf_i - v_i),
            a_i = 1 + G_i,   v_inf_i = (bias_i + S_i + x_t_i) / a_i

        whose exact solution is v_i(t+dt) = v_inf_i + (v_i(t) - v_inf_i) e^(-z_i)
        with z_i = a_i dt / tau_i. flyvis's integrator only ever writes
        `state + vel * dt` (network.py:426), so returning

            vel_i = (v_inf_i - v_i) (1 - e^(-z_i)) / dt

        REPRODUCES THAT EXACT SOLUTION through it. No division by zero is possible:
        the conductance is non-negative by construction, so G_i >= 0 and a_i >= 1.
        As dt -> 0 the factor (1 - e^(-z))/dt -> a_i / tau_i and this is the
        forward-Euler velocity again, so the continuous model is unchanged -- only
        its discretisation is, and only in the direction of being correct for a
        step that forward Euler could not take at all.
        """
        # state.targets.activity is the POSTSYNAPTIC voltage seen per edge. Every
        # other NetworkDynamics subclass reads state.sources.activity only; this
        # one needs both ends -- here through the split above rather than through a
        # per-edge driving force, which is the whole of the structural difference.
        released = params.edges.conductance * self.activation(state.sources.activity)

        # ONE SCATTER FOR BOTH SUMS. `target_sum` scatters over the LAST dimension
        # and expands its index over every leading one (network.py:366), so stacking
        # the two edge quantities sums them in a single kernel over the 1,513,231
        # edges rather than two, at every one of the unrolled timesteps.
        summed = target_sum(
            torch.stack((released, released * self.reversal_per_edge(params)))
        )
        total_conductance, weighted_reversal = summed[0], summed[1]

        # The floor is flyvis's own, kept for the reason it was written -- tau is a
        # free parameter with no non-negativity clamp, so a step could otherwise
        # divide by zero or by a negative number. It is NO LONGER load-bearing for
        # stability: the run that exploded had a cell type at tau = 19.4 ms, under
        # the 20 ms floor, and exponential Euler is stable there anyway.
        tau = torch.max(
            params.nodes.time_const, params.nodes.time_const.new_tensor(dt)
        )
        a = 1.0 + total_conductance
        v_inf = (params.nodes.bias + weighted_reversal + x_t) / a
        # -expm1(-z) is 1 - e^(-z) computed without cancellation at small z, which
        # is the regime every weakly driven neuron sits in (z = a dt / tau is 0.4
        # at the initial tau of 50 ms with no synaptic drive at all).
        decayed = -torch.expm1(-a * dt / tau)
        vel.nodes.activity = (v_inf - state.nodes.activity) * decayed / dt

    def currents(
        self,
        state: AutoDeref[str, AutoDeref[str, RefTensor]],
        params: AutoDeref[str, AutoDeref[str, RefTensor]],
    ) -> torch.Tensor:
        """Per-edge chemical current, for `Network.current_response`."""
        return (
            params.edges.conductance
            * self.activation(state.sources.activity)
            * (self.reversal_per_edge(params) - state.targets.activity)
        )
