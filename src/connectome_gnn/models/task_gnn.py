"""TaskGNN — hybrid of TaskRNN encoder/decoder + NeuralGNN recurrent core.

Architecture:

    Encoder  (W_in, optional velocity_gate)              ← TaskRNN
    Recurrent core (per timestep):                       ← NeuralGNN
        r        = σ(h)                                    (B, N)
        msg_e    = W_edge[e] · g_phi(r_src, a_src)^2       (B, E, 1)
        agg_j    = Σ_{e: dst(e)=j} msg_e                   (B, N, 1)
        rec_j    = f_theta(r_j, a_j, agg_j)                (B, N)
        τ · dh/dt = -h + rec + W_in·u + b
    Decoder  (W_out)                                     ← TaskRNN

Trains via the same `data_train_task_gnn` pipeline as TaskRNN — the trainer
only requires `forward(u) → (y_hat, h_buf)`, a `.S.abs()` for the L1 hook,
and the existing CX aux losses (`loss_cos_distance`, `loss_norm_floor`,
`loss_tv_circular`) which operate on the `W_rec` property surface.

Snapshot rendering and the GT-vs-learned scatter use a virtual `W_rec`
built by placing the GNN's per-edge `W` at the connectome edge positions
(diagonal masked). The f_theta non-linearity does NOT appear in this
surface — anatomy comparisons therefore capture per-edge gains only.
"""
import math
from typing import Optional

import torch
import torch.nn as nn

from connectome_gnn.models.MLP import MLP
from connectome_gnn.models.registry import register_model
from connectome_gnn.models.task_rnn import TaskRNN


@register_model("drosophila_cx_pi_gnn")
class TaskGNN(TaskRNN):
    """Drop-in for `drosophila_cx_pi`: same trainer, same encoder/decoder,
    GNN-driven recurrence (per-edge W, node embedding a, g_phi/f_theta MLPs).
    """

    def __init__(self, aggr_type: str = "add", config=None, device=None):
        # 1) Parent init wires the encoder, decoder, CX connectome buffers,
        #    type-pair masks, ring orderings, dt/τ, velocity_gate, etc.
        super().__init__(aggr_type=aggr_type, config=config, device=device)
        if self.W_param != "sign_locked":
            raise ValueError(
                "TaskGNN requires graph_model.W_param='sign_locked' (CX path)"
            )

        # 2) Drop the inherited per-edge magnitude `S` — it's replaced by the
        #    per-edge GNN weight `W`. `.S` is re-exposed as an alias to `.W`
        #    via __getattr__ so the trainer's `coeff_W_L1 · model.S.abs().sum()`
        #    hook continues to work.
        self._parameters.pop("S", None)

        N = self.n_units
        # 3) Edge index from W_con_mask (row=pre, col=post by TaskRNN
        #    convention, which is also the GNN src→dst convention).
        src, dst = self.W_con_mask.nonzero(as_tuple=True)
        edge_index = torch.stack([src, dst], dim=0).long().contiguous()
        self.register_buffer("_edge_index", edge_index, persistent=False)
        self.n_edges = int(edge_index.shape[1])

        # 4) GNN learnable components
        gm = config.graph_model
        emb_dim = int(getattr(gm, "embedding_dim", 2))
        hidden_dim = int(getattr(gm, "hidden_dim", 64))
        n_layers = int(getattr(gm, "n_layers", 3))
        hidden_dim_update = int(getattr(gm, "hidden_dim_update", hidden_dim))
        n_layers_update = int(getattr(gm, "n_layers_update", n_layers))
        act = str(getattr(gm, "MLP_activation", "relu"))
        self._g_phi_positive = bool(getattr(gm, "g_phi_positive", True))

        # g_phi: edge function — input (r_src, a_src), output scalar
        self.g_phi = MLP(
            input_size=1 + emb_dim, output_size=1,
            nlayers=n_layers, hidden_size=hidden_dim,
            activation=act, device=device,
        )
        # f_theta: node update — input (r, a, msg), output du/dt contribution
        self.f_theta = MLP(
            input_size=1 + emb_dim + 1, output_size=1,
            nlayers=n_layers_update, hidden_size=hidden_dim_update,
            activation=act, device=device,
        )

        # 5) Per-edge weight W (sign-free, no Dale lock)
        tc = config.training
        w_init_mode = str(getattr(tc, "w_init_mode", "zeros")).lower()
        w_init_scale = float(getattr(tc, "w_init_scale", 1.0))
        if w_init_mode == "zeros":
            W_init = torch.zeros(self.n_edges, 1, dtype=torch.float32)
        elif w_init_mode == "randn_scaled":
            W_init = torch.randn(self.n_edges, 1, dtype=torch.float32) * (
                w_init_scale / math.sqrt(max(1, self.n_edges))
            )
        elif w_init_mode == "uniform_scaled":
            bound = w_init_scale / math.sqrt(max(1, self.n_edges))
            W_init = (
                torch.rand(self.n_edges, 1, dtype=torch.float32) * 2.0 - 1.0
            ) * bound
        else:  # 'randn' or 'const' fallthrough
            W_init = torch.randn(self.n_edges, 1, dtype=torch.float32) * w_init_scale
        self.W = nn.Parameter(W_init)

        # 6) Per-node embedding a (ones init, matches NeuralGNN convention)
        self.a = nn.Parameter(torch.ones(N, emb_dim, dtype=torch.float32))

        # 7) Sign-lock toggle (shared with TaskRNN). When True, the effective
        # per-edge weight is `|W| · sign_GT`; magnitudes are learned, signs
        # come from the connectome (Dale-conformant). When False, W is
        # sign-free per edge — connectome topology is still enforced via
        # the edge_index restriction.
        # `self.lock_edge_signs` is already set by TaskRNN's __init__.
        if self.lock_edge_signs:
            # W_con / W_con_sign convention: row=pre, col=post — same as the
            # GNN's (src, dst). Index directly with `(src, dst)`.
            sign_e = self.W_con_sign[src, dst].to(torch.float32)   # (E,)
            self.register_buffer(
                "_edge_sign", sign_e.unsqueeze(-1), persistent=False,
            )

        if device is not None:
            self.to(device)

    def _effective_edge_weights(self) -> torch.Tensor:
        """Per-edge weight used in messages and in the W_rec surface.
        Shape: (E, 1).
            lock_edge_signs=True  → |W| · sign_GT (Dale-conformant).
            lock_edge_signs=False → W itself      (sign learned per edge).
        """
        if self.lock_edge_signs:
            return self.W.abs() * self._edge_sign
        return self.W

    def __getattr__(self, name: str):
        # Trainer reads `model.S` for `coeff_W_L1 · S.abs().sum()`. Route to W.
        if name == "S":
            return super().__getattr__("W")
        return super().__getattr__(name)

    # ------------------------------------------------------------------
    # Effective recurrent weight (linear surface only)
    # ------------------------------------------------------------------

    @property
    def W_rec(self) -> torch.Tensor:
        """Dense N×N built from the per-edge `W` placed at connectome edges,
        diagonal masked. Note: only the *linear* part of the GNN update
        appears here — `f_theta` does not. The cosine-distance / norm-floor
        regularisers and the GT-vs-learned scatter therefore compare
        per-edge gains, not the full recurrent operator.
        """
        N = self.n_units
        W_dense = self.W.new_zeros(N, N)
        src, dst = self._edge_index[0], self._edge_index[1]
        W_dense[src, dst] = self._effective_edge_weights().squeeze(-1)
        return W_dense * self._no_diag * self._image_mask

    # ------------------------------------------------------------------
    # GNN recurrent step
    # ------------------------------------------------------------------

    def _gnn_recurrent_drive(self, r: torch.Tensor) -> torch.Tensor:
        """(B, N) firing rates → (B, N) recurrent contribution to du/dt."""
        B, N = r.shape
        src = self._edge_index[0]
        dst = self._edge_index[1]

        # Per-edge features: (r_src, a_src).
        r_src = r[:, src].unsqueeze(-1)                       # (B, E, 1)
        a_src = self.a[src].unsqueeze(0).expand(B, -1, -1)    # (B, E, emb)
        edge_feat = torch.cat([r_src, a_src], dim=-1)
        g_out = self.g_phi(edge_feat)                         # (B, E, 1)
        if self._g_phi_positive:
            g_out = g_out ** 2
        # Per-edge weight (sign-locked or free) broadcasts over batch.
        edge_w = self._effective_edge_weights()               # (E, 1)
        msg = edge_w.unsqueeze(0) * g_out                     # (B, E, 1)

        # Scatter-add to destination nodes.
        agg = r.new_zeros(B, N, 1)
        agg.scatter_add_(1, dst.view(1, -1, 1).expand(B, -1, 1), msg)

        # f_theta on (r, a, msg).
        a_exp = self.a.unsqueeze(0).expand(B, -1, -1)         # (B, N, emb)
        feat = torch.cat([r.unsqueeze(-1), a_exp, agg], dim=-1)
        return self.f_theta(feat).squeeze(-1)                 # (B, N)

    # ------------------------------------------------------------------
    # Forward path (same shape contract as TaskRNN)
    # ------------------------------------------------------------------

    def forward(
        self,
        u: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = u.shape
        N = self.n_units
        h = u.new_zeros(B, N) if h0 is None else h0
        h_buf = u.new_empty(B, T, N)
        dt_over_tau = self.dt / self.tau
        noise_lvl = (
            self.noise_recurrent_level
            if (self.training and self.noise_recurrent_level > 0)
            else 0.0
        )
        for t in range(T):
            r = self._sigma(h)
            rec = self._gnn_recurrent_drive(r)
            inp = self._project_in(u[:, t, :])
            h = h + dt_over_tau * (-h + rec + inp + self.b)
            if noise_lvl > 0:
                h = h + noise_lvl * torch.randn_like(h)
            h_buf[:, t, :] = h
        y_hat = self._project_out(self._sigma(h_buf))
        return y_hat, h_buf
