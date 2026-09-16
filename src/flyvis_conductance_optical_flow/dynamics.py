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
SHORTENS with drive. That is shunting, it is the dynamical content of the change,
and it makes the model more stable under strong input rather than less.
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
        """dv/dt with the driving force in the message."""
        # state.targets.activity is the POSTSYNAPTIC voltage seen per edge. Every
        # other NetworkDynamics subclass reads state.sources.activity only; this
        # one needs both ends, which is the whole of the structural difference.
        driving_force = self.reversal_per_edge(params) - state.targets.activity
        chemical_current = target_sum(
            params.edges.conductance
            * self.activation(state.sources.activity)
            * driving_force
        )
        vel.nodes.activity = (
            1
            / torch.max(
                params.nodes.time_const,
                params.nodes.time_const.new_tensor(dt),
            )
            * (
                -state.nodes.activity
                + params.nodes.bias
                + chemical_current
                + x_t
            )
        )

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
