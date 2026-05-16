"""Hybrid Hulse RNN registered for the path-integration task pipeline.

Architecture (Hulse Methods Eqs. 1, 9-11):

    τ * dh_j/dt = -h_j + Σ_k W_rec[j,k] σ(h_k) + Σ_l W_in[j,l] u_l + b_j
    y_hat_i     = Σ_j W_out[i,j] σ(h_j) + b_out_i,           i = 1, 2

with W_rec = |S| ⊙ W_con (sign and sparsity locked to the connectome at init;
only per-edge magnitudes |S| are learned). W_in and W_out are configurable as
either learnable matrices (Hulse default) or small MLPs reusing
`graph_model.hidden_dim` / `graph_model.n_layers`.

Buffer protocol matches `teachers.JaneliaCxRNN` (W_rec, W_con, _block_mask_i,
_ring_order_<name>, dt, n_units, neuron_types, type_names, epg_indices,
epg_glom_ix) so the helpers in `models.cx_eval` (path_integration_accuracy,
bump_fwhm, _save_training_snapshot, _deterministic_sweep_rollout) work on this
class without branching.

Registered names: "drosophila_cx_pi", "task_rnn" (canonical), "neural_task_gnn" (legacy alias).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from connectome_gnn.models.cx_eval import build_type_pair_blocks
from connectome_gnn.models.MLP import MLP
from connectome_gnn.models.registry import register_model


@register_model("drosophila_cx_pi", "cortex_delaygo", "task_rnn", "neural_task_gnn")
class TaskRNN(nn.Module):
    """Connectome-constrained CX RNN with configurable I/O projections."""

    def __init__(self, aggr_type: str = "add", config=None, device=None):
        super().__init__()

        self.device = device
        self.aggr_type = aggr_type

        sim = config.simulation
        gm = config.graph_model
        task = config.task
        pi = task.path_integration

        # --- Connectome (hard sign-lock at init, Hulse Eq. 9) -----------
        from connectome_gnn.generators.connconstr_data import (
            load_drosophila_cx_connectome,
        )
        include_er6 = bool(getattr(gm, "include_er6", True))
        cx = load_drosophila_cx_connectome(
            sim.connconstr_datapath, include_er6=include_er6
        )
        N = int(cx["N"])
        self.n_units = N
        self.n_input = 3
        self.n_output = 2

        W_con = torch.from_numpy(cx["J_effective"].astype(np.float32))
        self.register_buffer("W_con", W_con)
        self.register_buffer("W_con_sign", torch.sign(W_con))
        self.register_buffer("W_con_mask", (W_con != 0).to(torch.float32))

        # Trainable per-edge magnitude. dW_rec/dS = sign(W_con) is 0 at
        # absent connections, so sparsity is enforced for free.
        # w_init_mode: 'const' (default) | 'randn' | 'zeros'.
        # w_init_scale: scalar multiplier on the chosen template (default 0.01).
        tc = getattr(config, "training", None)
        w_init_mode = str(getattr(tc, "w_init_mode", "const")).lower()
        w_init_scale = float(getattr(tc, "w_init_scale", 0.01))
        if w_init_mode == "zeros":
            S_init = torch.zeros_like(self.W_con_mask)
        elif w_init_mode == "randn":
            S_init = torch.randn_like(self.W_con_mask) * w_init_scale * self.W_con_mask
        else:  # 'const'
            S_init = w_init_scale * self.W_con_mask
        self.S = nn.Parameter(S_init)

        # --- Type-pair masks for cos-distance / norm-floor regularisers --
        neuron_types = np.asarray(cx["neuron_types"]).astype(np.int64)
        type_names = list(cx["type_names"])
        type_pair_blocks = build_type_pair_blocks(
            neuron_types, type_names, cx["J_effective"].astype(np.float32)
        )
        self._block_names: list[str] = list(type_pair_blocks.keys())
        for i, name in enumerate(self._block_names):
            self.register_buffer(
                f"_block_mask_{i}", type_pair_blocks[name].to(torch.bool),
                persistent=False,
            )

        # --- Ring orderings for circular-TV regulariser ------------------
        n_epg = int(cx["n_epg"])
        epg_glom_ix = np.asarray(cx["epg_ix"], dtype=np.int64)
        self._ring_names: list[str] = []
        ring_assignments: dict = {}
        if "EPG" in type_names:
            epg_t = type_names.index("EPG")
            epg_idx = np.where(neuron_types == epg_t)[0]
            if epg_idx.size == epg_glom_ix.size:
                ring_assignments["EPG"] = (epg_idx, epg_glom_ix)
        pen_type_idx = [i for i, n in enumerate(type_names)
                        if "PEN" in n and "PEG" not in n]
        pen_idx_all: list[int] = []
        for t in pen_type_idx:
            pen_idx_all.extend(np.where(neuron_types == t)[0].tolist())
        if pen_idx_all:
            pen_idx_arr = np.array(sorted(pen_idx_all), dtype=np.int64)
            ring_assignments["PEN"] = (pen_idx_arr, np.arange(pen_idx_arr.size))
        for name, (idx, pos) in ring_assignments.items():
            sort = np.argsort(np.asarray(pos, dtype=np.int64), kind="stable")
            order = torch.from_numpy(np.asarray(idx, dtype=np.int64)[sort]).long()
            safe = name.replace("-", "_").replace(" ", "_")
            self.register_buffer(f"_ring_order_{safe}", order, persistent=False)
            self._ring_names.append(safe)

        # --- Configurable input projection W_in --------------------------
        # "matrix": learnable (N, n_input) Gaussian-init matrix (Hulse).
        # "mlp":    small MLP reusing graph_model.hidden_dim / n_layers.
        self.input_proj = getattr(gm, "input_proj", "matrix")
        if self.input_proj == "matrix":
            self.W_in = nn.Parameter(
                torch.randn(N, self.n_input, dtype=torch.float32) * (1.0 / 100.0)
            )
            self._W_in_mlp = None
        elif self.input_proj == "mlp":
            self.W_in = None
            self._W_in_mlp = MLP(
                input_size=self.n_input, output_size=N,
                nlayers=gm.n_layers, hidden_size=gm.hidden_dim,
                activation=gm.MLP_activation, device=device,
            )
        else:
            raise ValueError(f"input_proj must be 'matrix' or 'mlp', got {self.input_proj!r}")

        # Recurrent bias (Hulse: initialised to 1).
        self.b = nn.Parameter(torch.ones(N, dtype=torch.float32))

        # --- Configurable output projection W_out ------------------------
        self.output_proj = getattr(gm, "output_proj", "matrix")
        if self.output_proj == "matrix":
            self.W_out = nn.Parameter(torch.empty(self.n_output, N, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.W_out, a=math.sqrt(5))
            self.b_out = nn.Parameter(torch.zeros(self.n_output, dtype=torch.float32))
            self._W_out_mlp = None
        elif self.output_proj == "mlp":
            self.W_out = None
            self.b_out = None
            self._W_out_mlp = MLP(
                input_size=N, output_size=self.n_output,
                nlayers=gm.n_layers, hidden_size=gm.hidden_dim,
                activation=gm.MLP_activation, device=device,
            )
        else:
            raise ValueError(f"output_proj must be 'matrix' or 'mlp', got {self.output_proj!r}")

        # --- Dynamics constants (from task config) -----------------------
        self.tau = float(getattr(pi, "tau", 0.1))
        self.dt = float(pi.dt)

        # --- Stochastic regularisation during BPTT ----------------------
        # Flyvis injects `noise_recurrent_level * randn` at every recurrent
        # step (recurrent_step.py:_standard_recurrent_loss). Smooths the
        # long-T BPTT landscape and is one of the most effective stabilisers
        # for connectome-locked recurrent training. 0 = off (Hulse default).
        self.noise_recurrent_level = float(
            getattr(config.training, "noise_recurrent_level", 0.0)
        )

        # --- CX metadata exposed for cx_eval helpers ---------------------
        self.neuron_types = neuron_types
        self.type_names = type_names
        self.epg_indices = np.arange(n_epg, dtype=np.int64)
        self.epg_glom_ix = epg_glom_ix

        if device is not None:
            self.to(device)

    # --- Effective recurrent weight (Hulse Eq. 9) ----------------------

    @property
    def W_rec(self) -> torch.Tensor:
        """W_rec = |S| ⊙ W_con. W_con_sign is 0 wherever W_con is 0, so the
        sparsity mask is implicit."""
        return self.S.abs() * self.W_con_sign

    # --- Forward path ---------------------------------------------------

    def _project_in(self, u_t: torch.Tensor) -> torch.Tensor:
        """(B, n_input) -> (B, N)."""
        if self.input_proj == "matrix":
            return u_t @ self.W_in.t()
        return self._W_in_mlp(u_t)

    def _project_out(self, r: torch.Tensor) -> torch.Tensor:
        """(B, T, N) -> (B, T, n_output)."""
        if self.output_proj == "matrix":
            return r @ self.W_out.t() + self.b_out
        return self._W_out_mlp(r)

    def forward(
        self,
        u: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the network for T timesteps over a batch.

        Args:
            u: (B, T, n_input) input stream.
            h0: (B, N) initial subthreshold activity (zeros if None).

        Returns:
            y_hat: (B, T, n_output) readout.
            h_buf: (B, T, N)        subthreshold activity (for diagnostics
                                   and the circular-TV regulariser).
        """
        B, T, _ = u.shape
        N = self.n_units

        h = (torch.zeros(B, N, dtype=u.dtype, device=u.device)
             if h0 is None else h0)
        h_buf = torch.empty(B, T, N, dtype=u.dtype, device=u.device)

        W_rec_t = self.W_rec.t()
        dt_over_tau = self.dt / self.tau
        # Inject noise only during training (eval/snapshot stays deterministic).
        noise_lvl = (self.noise_recurrent_level
                     if (self.training and self.noise_recurrent_level > 0)
                     else 0.0)

        for t in range(T):
            r = torch.sigmoid(h)
            rec = r @ W_rec_t
            inp = self._project_in(u[:, t, :])
            h = h + dt_over_tau * (-h + rec + inp + self.b)
            if noise_lvl > 0:
                h = h + noise_lvl * torch.randn_like(h)
            h_buf[:, t, :] = h

        y_hat = self._project_out(torch.sigmoid(h_buf))
        return y_hat, h_buf

    # --- Regularisers (identical to teachers.JaneliaCxRNN) -------------

    def loss_cos_distance(self, lam: float = 1.0) -> torch.Tensor:
        """Hulse Eq. 10: per-(post-type, pre-type) block cosine-distance
        between W_rec and the connectome template W_con."""
        if not self._block_names or torch.all(self.W_con == 0):
            return self.W_rec.new_zeros(())
        total = self.W_rec.new_zeros(())
        eps = 1e-12
        for i in range(len(self._block_names)):
            mask = getattr(self, f"_block_mask_{i}")
            w_rec_b = self.W_rec[mask]
            w_con_b = self.W_con[mask]
            if w_con_b.abs().sum() < eps:
                continue
            num = (w_rec_b * w_con_b).sum()
            den = w_rec_b.norm() * w_con_b.norm() + eps
            total = total + (1.0 - num / den)
        return lam * total / max(len(self._block_names), 1)

    def loss_norm_floor(self, lam: float = 1.0, kappa: float = 0.05) -> torch.Tensor:
        """Hulse Eq. 11: soft lower bound on mean |W| per type-pair block."""
        if not self._block_names:
            return self.W_rec.new_zeros(())
        total = self.W_rec.new_zeros(())
        for i in range(len(self._block_names)):
            mask = getattr(self, f"_block_mask_{i}")
            mean_abs = self.W_rec[mask].abs().mean()
            slack = F.relu(kappa - mean_abs)
            total = total + slack.pow(2)
        return lam * total / max(len(self._block_names), 1)

    def loss_tv_circular(self, h_buf: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        """Circular total-variation penalty on EPG/PEN ring firing rates."""
        if not self._ring_names or lam == 0.0:
            return self.S.new_zeros(())
        r = torch.sigmoid(h_buf)
        total = self.S.new_zeros(())
        for name in self._ring_names:
            order = getattr(self, f"_ring_order_{name}")
            r_ring = r.index_select(-1, order)
            diffs = (torch.roll(r_ring, -1, dims=-1) - r_ring).abs()
            total = total + diffs.sum(dim=-1).mean()
        return lam * total / len(self._ring_names)
