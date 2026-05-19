"""CxTaskGNN — Drosophila CX path-integration GNN.

Hybrid of the TaskRNN encoder/decoder shell with a per-edge GNN
recurrent core. The dense `r @ W_rec` matmul of CxTaskRNN is replaced
by a NeuralGNN-style update:

    r        = σ(h)                                    (B, N)   — sigmoid wrap
    msg_e    = W_edge[e] · g_phi(v_src, a_src)²        (B, E, 1) — raw v=h
    agg_j    = Σ_{e: dst(e)=j} msg_e                   (B, N, 1)
    rec_j    = f_theta(v_j, a_j, agg_j)                (B, N)
    τ · dh/dt = -h + rec + W_in·u + b

Connectome topology is enforced via `edge_index = nonzeros(W_con)`.
Sign-lock toggle:
    lock_edge_signs=True  → effective per-edge weight = |W| · sign_GT
                            (Dale-conformant; magnitudes learned)
    lock_edge_signs=False → effective per-edge weight = W
                            (sign learned per edge; topology still fixed)

Snapshot rendering and the GT-vs-learned scatter use a *virtual* W_rec
built by placing the GNN's per-edge `W` at the connectome edge positions
(diagonal masked). The f_theta non-linearity does NOT appear in this
surface — anatomy comparisons therefore capture per-edge gains only.

Standalone class: no inheritance from CxTaskRNN. The encoder/decoder/
connectome setup is duplicated for clarity (one file = one full story).

Registered name: "drosophila_cx_pi_gnn".
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from connectome_gnn.models.cx_eval import build_type_pair_blocks
from connectome_gnn.models.MLP import MLP
from connectome_gnn.models.registry import register_model


_ACT_MAP = {
    "sigmoid": torch.sigmoid,
    "relu": F.relu,
    "tanh": torch.tanh,
    "softplus": F.softplus,
}


def _load_image_mask(path: str, N: int) -> torch.Tensor:
    """Load `path`, resize to (N, N), threshold at median → binary (N, N)."""
    from PIL import Image
    if not os.path.isfile(path):
        from connectome_gnn.utils import get_data_root
        cand = os.path.join(get_data_root(), path)
        if os.path.isfile(cand):
            path = cand
        else:
            raise FileNotFoundError(f"w_mask_image_path not found: {path}")
    img = Image.open(path).convert("L")
    img = img.resize((N, N), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    threshold = float(np.median(arr))
    mask_np = (arr > threshold).astype(np.float32)
    return torch.from_numpy(mask_np)


@register_model("drosophila_cx_pi_gnn")
class CxTaskGNN(nn.Module):
    """Sign-locked Drosophila CX path-integration GNN."""

    def __init__(self, aggr_type: str = "add", config=None, device=None):
        super().__init__()
        self.device = device
        self.aggr_type = aggr_type

        sim = config.simulation
        gm = config.graph_model
        task = config.task

        self.lock_edge_signs = bool(getattr(gm, "lock_edge_signs", True))

        # --- Load hemibrain CX connectome -------------------------------
        from connectome_gnn.generators.connconstr_data import (
            load_drosophila_cx_connectome,
        )
        cx = load_drosophila_cx_connectome(sim.connconstr_datapath)
        N = int(cx["N"])
        self.n_units = N
        self.n_input = 3
        self.n_output = 2

        W_con = torch.from_numpy(cx["J_effective"].astype(np.float32))
        self.register_buffer("W_con", W_con)
        self.register_buffer("W_con_sign", torch.sign(W_con))
        self.register_buffer("W_con_mask", (W_con != 0).to(torch.float32))

        # --- Edge index from W_con_mask (row=pre, col=post) ------------
        src, dst = self.W_con_mask.nonzero(as_tuple=True)
        edge_index = torch.stack([src, dst], dim=0).long().contiguous()
        self.register_buffer("_edge_index", edge_index, persistent=False)
        self.n_edges = int(edge_index.shape[1])

        # --- Per-edge sign buffer (used when lock_edge_signs=True) -----
        if self.lock_edge_signs:
            sign_e = self.W_con_sign[src, dst].to(torch.float32)
            self.register_buffer(
                "_edge_sign", sign_e.unsqueeze(-1), persistent=False,
            )

        # --- Type-pair masks for cos-distance / norm-floor regularisers
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

        # --- Ring orderings for circular-TV regulariser -----------------
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

        # --- Metadata for cx_eval helpers -------------------------------
        self.neuron_types = neuron_types
        self.type_names = type_names
        self.epg_indices = np.arange(n_epg, dtype=np.int64)
        self.epg_glom_ix = epg_glom_ix

        # --- Velocity gating (CX-only) ---------------------------------
        # `pen_only`: zero W_in[:, 0] outside PEN rows; per-unit weights free.
        # `pen_4scalar`: strict Hulse 2025 — 4 learnable scalars
        #                (L/R × PENa/PENb) broadcast onto subpopulations.
        # In either case, channels 1-2 (initial-bump cue) stay free for all rows.
        self.velocity_gate = str(getattr(gm, "velocity_gate", "none")).lower()
        if self.velocity_gate == "pen_only":
            mask = torch.zeros(N, self.n_input, dtype=torch.float32)
            mask[:, 1:] = 1.0
            if pen_idx_all:
                mask[torch.as_tensor(sorted(pen_idx_all), dtype=torch.long), 0] = 1.0
            self.register_buffer("_W_in_mask", mask, persistent=False)
        elif self.velocity_gate == "pen_4scalar":
            pen_subpop = cx.get("pen_subpop_ix", {})
            required = ("PENa_L", "PENa_R", "PENb_L", "PENb_R")
            missing = [k for k in required if k not in pen_subpop or len(pen_subpop[k]) == 0]
            if missing:
                raise ValueError(
                    f"velocity_gate='pen_4scalar' requires non-empty "
                    f"pen_subpop_ix for {required}; missing/empty: {missing}"
                )
            for key in required:
                ind = torch.zeros(N, dtype=torch.float32)
                ind[torch.as_tensor(pen_subpop[key], dtype=torch.long)] = 1.0
                self.register_buffer(f"_pen_ind_{key.lower()}", ind, persistent=False)
            self.v_pena_l = nn.Parameter(torch.tensor(0.01))
            self.v_pena_r = nn.Parameter(torch.tensor(-0.01))
            self.v_penb_l = nn.Parameter(torch.tensor(0.01))
            self.v_penb_r = nn.Parameter(torch.tensor(-0.01))
        elif self.velocity_gate != "none":
            raise ValueError(
                f"graph_model.velocity_gate must be 'none', 'pen_only', or "
                f"'pen_4scalar', got {self.velocity_gate!r}"
            )

        # --- Dynamics constants (from task.path_integration) ------------
        self.tau = float(getattr(task.path_integration, "tau", 0.1))
        self.dt = float(task.path_integration.dt)

        # --- Zero-diagonal mask -----------------------------------------
        self.register_buffer(
            "_no_diag", 1.0 - torch.eye(N, dtype=torch.float32),
            persistent=False,
        )

        # --- Optional image-derived binary mask on W_rec ----------------
        img_path = str(getattr(gm, "w_mask_image_path", "")).strip()
        if img_path:
            img_mask = _load_image_mask(img_path, N)
        else:
            img_mask = torch.ones(N, N, dtype=torch.float32)
        self.register_buffer("_image_mask", img_mask, persistent=False)

        # --- Encoder W_in (matrix or MLP) ------------------------------
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

        # --- Recurrent bias (initialised to 1) -------------------------
        self.b = nn.Parameter(torch.ones(N, dtype=torch.float32))

        # --- Decoder W_out (matrix or MLP) ------------------------------
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

        # --- Recurrent activation σ -------------------------------------
        act_name = str(getattr(gm, "recurrent_activation", "sigmoid")).lower()
        if act_name not in _ACT_MAP:
            raise ValueError(
                f"recurrent_activation must be one of {list(_ACT_MAP)}, "
                f"got {act_name!r}"
            )
        self.recurrent_activation_name = act_name
        self._sigma = _ACT_MAP[act_name]

        # --- Stochastic regularisation during BPTT ---------------------
        self.noise_recurrent_level = float(
            getattr(config.training, "noise_recurrent_level", 0.0)
        )

        # --- GNN-specific components -----------------------------------
        emb_dim = int(getattr(gm, "embedding_dim", 2))
        hidden_dim = int(getattr(gm, "hidden_dim", 64))
        n_layers = int(getattr(gm, "n_layers", 3))
        hidden_dim_update = int(getattr(gm, "hidden_dim_update", hidden_dim))
        n_layers_update = int(getattr(gm, "n_layers_update", n_layers))
        mlp_act = str(getattr(gm, "MLP_activation", "relu"))
        self._g_phi_positive = bool(getattr(gm, "g_phi_positive", True))

        # g_phi: edge function — input (v_src, a_src), output scalar.
        self.g_phi = MLP(
            input_size=1 + emb_dim, output_size=1,
            nlayers=n_layers, hidden_size=hidden_dim,
            activation=mlp_act, device=device,
        )
        # f_theta: node update — input (v, a, msg), output du/dt contribution.
        self.f_theta = MLP(
            input_size=1 + emb_dim + 1, output_size=1,
            nlayers=n_layers_update, hidden_size=hidden_dim_update,
            activation=mlp_act, device=device,
        )

        # Per-edge weight W (sign-free by default; sign is locked via the
        # _edge_sign buffer + _effective_edge_weights when lock_edge_signs).
        train_config = config.training
        w_init_mode = str(getattr(train_config, "w_init_mode", "zeros")).lower()
        w_init_scale = float(getattr(train_config, "w_init_scale", 1.0))
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

        # Per-node embedding a (ones init, matches NeuralGNN convention).
        self.a = nn.Parameter(torch.ones(N, emb_dim, dtype=torch.float32))

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # S alias for the trainer's L1 hook
    # ------------------------------------------------------------------

    @property
    def S(self) -> nn.Parameter:
        """Alias of `self.W` so the trainer's `coeff_W_L1 * model.S.abs().sum()`
        hook continues to work without architecture-aware branching."""
        return self.W

    # ------------------------------------------------------------------
    # Effective per-edge weight + dense W_rec view
    # ------------------------------------------------------------------

    def _effective_edge_weights(self) -> torch.Tensor:
        """Per-edge weight used in messages and in the W_rec surface.
        Shape: (E, 1).
            lock_edge_signs=True  → |W| · sign_GT  (Dale-conformant)
            lock_edge_signs=False → W              (sign learned per edge)
        """
        if self.lock_edge_signs:
            return self.W.abs() * self._edge_sign
        return self.W

    @property
    def W_rec(self) -> torch.Tensor:
        """Dense N×N built from the per-edge `W` placed at connectome edges,
        diagonal masked. Only the *linear* part of the GNN update appears
        here — `f_theta` does not. The cosine-distance / norm-floor
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

    def _gnn_recurrent_drive(self, v: torch.Tensor) -> torch.Tensor:
        """(B, N) subthreshold state v (= h) → (B, N) recurrent du/dt.

        Both g_phi and f_theta consume the raw state `v` directly, matching
        the NeuralGNN / data_train_gnn convention where the MLPs themselves
        are the nonlinearities — no sigmoid wrapping the GNN inputs. The
        sigmoid is reserved for the decoder (cos/sin readout from firing
        rates), preserving the TaskRNN output contract.
        """
        B, N = v.shape
        src = self._edge_index[0]
        dst = self._edge_index[1]

        # Per-edge features: (v_src, a_src).
        v_src = v[:, src].unsqueeze(-1)                       # (B, E, 1)
        a_src = self.a[src].unsqueeze(0).expand(B, -1, -1)    # (B, E, emb)
        edge_feat = torch.cat([v_src, a_src], dim=-1)
        g_out = self.g_phi(edge_feat)                         # (B, E, 1)
        if self._g_phi_positive:
            g_out = g_out ** 2
        # Per-edge weight broadcasts over batch.
        edge_w = self._effective_edge_weights()               # (E, 1)
        msg = edge_w.unsqueeze(0) * g_out                     # (B, E, 1)

        # Scatter-add to destination nodes.
        agg = v.new_zeros(B, N, 1)
        agg.scatter_add_(1, dst.view(1, -1, 1).expand(B, -1, 1), msg)

        # f_theta on (v, a, msg).
        a_exp = self.a.unsqueeze(0).expand(B, -1, -1)         # (B, N, emb)
        feat = torch.cat([v.unsqueeze(-1), a_exp, agg], dim=-1)
        return self.f_theta(feat).squeeze(-1)                 # (B, N)

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------

    def _project_in(self, u_t: torch.Tensor) -> torch.Tensor:
        """(B, n_input) -> (B, N)."""
        if self.input_proj == "matrix":
            W = self.W_in
            if self.velocity_gate == "pen_4scalar":
                v_col = (
                    self._pen_ind_pena_l * self.v_pena_l
                    + self._pen_ind_pena_r * self.v_pena_r
                    + self._pen_ind_penb_l * self.v_penb_l
                    + self._pen_ind_penb_r * self.v_penb_r
                )
                W = torch.cat([v_col.unsqueeze(1), W[:, 1:]], dim=1)
            else:
                mask = getattr(self, "_W_in_mask", None)
                if mask is not None:
                    W = W * mask
            return u_t @ W.t()
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
            h_buf: (B, T, N)        subthreshold activity.
        """
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
            # GNN core sees the raw subthreshold state (v ≡ h). f_theta is
            # responsible for the full -h + recurrent-drive term; no explicit
            # leak is added outside the MLP, matching the flyvis GNN convention
            # (dv/dt = f_theta(v, a, m, ...)).
            rec = self._gnn_recurrent_drive(h)
            inp = self._project_in(u[:, t, :])
            h = h + dt_over_tau * (rec + inp + self.b)
            if noise_lvl > 0:
                h = h + noise_lvl * torch.randn_like(h)
            h_buf[:, t, :] = h

        # Decoder reads firing rates so the cos/sin readout matches the
        # TaskRNN output contract.
        y_hat = self._project_out(self._sigma(h_buf))
        return y_hat, h_buf

    # ------------------------------------------------------------------
    # Regularisers (Hulse Eqs. 10-11 + circular TV)
    # ------------------------------------------------------------------

    def loss_cos_distance(self, lam: float = 1.0) -> torch.Tensor:
        """Hulse Eq. 10: per-(post-type, pre-type) block cosine-distance
        between W_rec and the connectome template W_con."""
        if not self._block_names:
            return self.W_rec.new_zeros(())
        if torch.all(self.W_con == 0):
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
            return h_buf.new_zeros(())
        r = self._sigma(h_buf)
        total = h_buf.new_zeros(())
        for name in self._ring_names:
            order = getattr(self, f"_ring_order_{name}")
            r_ring = r.index_select(-1, order)
            diffs = (torch.roll(r_ring, -1, dims=-1) - r_ring).abs()
            total = total + diffs.sum(dim=-1).mean()
        return lam * total / len(self._ring_names)
