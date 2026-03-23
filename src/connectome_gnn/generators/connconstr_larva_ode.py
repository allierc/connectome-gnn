"""Drosophila larva two-population ODE.

Ref: Beiran & Litwin-Kumar (2023), Fig 5 — Drosophila larva
     papers/Code_NN/Code_NN/Data/Figure5/setup.py forwardpass() lines 24-45

Two populations merged into a single graph:
  - Premotor (PMN, nodes 0..N-1):
      dup/dt = (-up + gp * msg_pp + bp + wsp @ stim) / taup
  - Motor (MN, nodes N..N+M-1):
      dum/dt = (-um + gm * msg_pm + bm) / taum

Where:
  msg_pp_i = sum_j Jpp[j→i] * softplus(up_j)   (premotor recurrent)
  msg_pm_i = sum_j Jpm[j→i] * softplus(up_j)   (premotor→motor)

Activation: Softplus (setup.py lines 32-33), NOT ReLU.
Gains clamped to [0.5, 5.0] (setup.py lines 49-51).

Differences from paper repo:
    - Paper uses trial-based training with state reset every 2 conditions × 6s.
      Trial resets are not biologically realistic — we generate continuous trajectories.
    - Paper uses dense J @ r; we use sparse scatter_add on a unified graph.
    - Paper treats premotor/motor as separate arrays; we merge into one graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from connectome_gnn.generators.ode_params import LarvaODEParams
from connectome_gnn.neuron_state import NeuronState


class LarvaODE(nn.Module):
    """Two-population ODE for the Drosophila larva motor system.

    Premotor neurons (0..N-1) receive recurrent input (Jpp) and stimulus.
    Motor neurons (N..N+M-1) receive feedforward input from premotor (Jpm).

    All edge sources are premotor neurons. The activation softplus(up) is
    applied at the source, and gain (gp/gm) is applied at the destination.

    Uses explicit scatter_add for message passing (no PyG dependency).
    """

    def __init__(self, ode_params=None, device=None):
        super().__init__()

        if isinstance(ode_params, dict):
            ode_params = LarvaODEParams(**ode_params)
        self.ode_params = ode_params
        self.device = device

        if self.ode_params is not None:
            self.ode_params.to(device)

    def _compute_messages(self, v, edge_index):
        """Compute per-edge messages and aggregate via scatter_add.

        Ref: setup.py forwardpass() lines 32-33:
            gp * softplus(up) @ Jpp   (premotor recurrent)
            gm * softplus(up) @ Jpm   (premotor→motor)

        In sparse form: msg_i = sum_{edges to i} W_e * softplus(v[src_e])
        All sources are premotor neurons (both Jpp and Jpm edges).

        Args:
            v: (N+M, 1) voltages (premotor + motor)
            edge_index: (2, E) source/destination indices

        Returns:
            msg: (N+M, 1) raw aggregated messages (before gain scaling)
        """
        src, dst = edge_index
        N_total = v.shape[0]

        # Activation: Softplus (setup.py line 32-33)
        # All source neurons are premotor, activation is on their voltage
        v_src = v[src]
        edge_msg = self.ode_params.W[:, None] * F.softplus(v_src)

        msg = torch.zeros(N_total, 1, device=self.device, dtype=v.dtype)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)

        return msg

    def forward(self, state: NeuronState, edge_index: torch.Tensor, **kwargs):
        """Compute dv/dt from neuron state and connectivity.

        Ref: setup.py forwardpass() lines 30-33:
            up = (1-dt/taup)*up + (dt/taup)*(gp * softplus(up) @ Jpp + bp + s @ wsp)
            um = (1-dt/taum)*um + (dt/taum)*(gm * softplus(up) @ Jpm + bm)

        Continuous form:
            dup/dt = (-up + gp * msg + bp + stim) / taup
            dum/dt = (-um + gm * msg + bm) / taum

        Args:
            state: NeuronState with voltage (up|um), stimulus fields
            edge_index: (2, E) tensor of (src, dst) edge indices

        Returns:
            dv: (N+M, 1) voltage derivative
        """
        v = state.voltage.unsqueeze(-1)       # (N+M, 1)
        stim = state.stimulus.unsqueeze(-1)    # (N+M, 1) — stimulus for premotor, 0 for motor
        p = self.ode_params
        N = p.n_premotor
        M = p.n_motor

        msg = self._compute_messages(v, edge_index)

        # Clamp gains to [0.5, 5.0] (setup.py lines 49-51)
        gp = torch.clamp(p.gp[:, None], 0.5, 5.0)
        gm = torch.clamp(p.gm[:, None], 0.5, 5.0)

        dv = torch.zeros_like(v)

        # Premotor: dup/dt = (-up + gp * msg + bp + stim) / taup
        dv[:N] = (-v[:N] + gp * msg[:N] + p.bp[:, None] + stim[:N]) / p.taup

        # Motor: dum/dt = (-um + gm * msg + bm) / taum
        dv[N:N+M] = (-v[N:N+M] + gm * msg[N:N+M] + p.bm[:, None]) / p.taum

        return dv
