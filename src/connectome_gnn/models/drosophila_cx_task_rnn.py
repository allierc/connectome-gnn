"""DrosophilaCxTaskRNN — sign-locked recurrent network for the Drosophila
CX path-integration task.

Architecture (Hulse 2025 Methods Eqs. 1, 9-11):

    τ * dh_j/dt = -h_j + Σ_k W_rec[j,k] σ(h_k) + Σ_l W_in[j,l] u_l + b_j
    y_hat_i     = Σ_j W_out[i,j] σ(h_j) + b_out_i

Recurrent matrix parameterisation via `graph_model.wrec_param`:
    "edge_magnitude" → W_rec = |S| ⊙ sign(W_con)        (Dale, sparsity locked)
    "edge_free"      → W_rec =  S  ⊙ mask(W_con)        (free sign per edge)
    "column_dale"    → W_rec = |S| ⊙ col_sign[None,:]   (dense N×N, column-Dale)
where col_sign[j] = sign(Σᵢ W_con[i, j]) is the dominant E/I identity of pre-
neuron j. Diagonal is always masked to zero.

Single-purpose class: hemibrain connectome loaded at init, n_input=3
(omega, cos(θ₀)·δ_t0, sin(θ₀)·δ_t0), n_output=2 (cos/sin heading readout).

Buffer protocol matches `teachers.JaneliaCxRNN`: W_rec, W_con, S,
_block_mask_i, _ring_order_<name>, dt, n_units, neuron_types,
type_names, epg_indices, epg_glom_ix — so the helpers in
`models.drosophila_cx_eval` (path_integration_accuracy, bump_fwhm,
_save_training_snapshot, _deterministic_sweep_rollout) work without
branching.

Registered name: "drosophila_cx_pi".
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from connectome_gnn.models.drosophila_cx_eval import build_type_pair_blocks
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


@register_model("drosophila_cx_pi")
class DrosophilaCxTaskRNN(nn.Module):
    """Sign-locked Drosophila CX path-integration RNN."""

    def __init__(self, aggr_type: str = "add", config=None, device=None):
        super().__init__()
        self.device = device
        self.aggr_type = aggr_type

        sim = config.simulation
        gm = config.graph_model
        task = config.task

        self.wrec_param = str(getattr(gm, "wrec_param", "edge_magnitude")).lower()
        if self.wrec_param not in ("edge_magnitude", "edge_free", "column_dale"):
            raise ValueError(
                f"graph_model.wrec_param must be 'edge_magnitude', 'edge_free' "
                f"or 'column_dale'; got {self.wrec_param!r}"
            )

        train_config = config.training
        w_init_mode = getattr(train_config, "w_init_mode", "const")

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

        # Per-pre-neuron sign for column_dale mode. W_con layout is
        # row=post, col=pre, so summing along dim=0 gives the net outgoing
        # weight of each pre-neuron j.
        if self.wrec_param == "column_dale":
            col_sign = torch.sign(W_con.sum(dim=0))
            if (col_sign == 0).any():
                zero_idx = torch.nonzero(col_sign == 0, as_tuple=True)[0].tolist()
                raise ValueError(
                    f"wrec_param='column_dale' requires every pre-neuron to "
                    f"have non-zero net outgoing weight in W_con; "
                    f"col_sign==0 at indices {zero_idx[:10]} (showing first 10)"
                )
            self.register_buffer("col_sign", col_sign)

        # --- Trainable matrix S -----------------------------------------
        # Shape (N, N) in all wrec_param modes. The W_rec property combines
        # S with the relevant sign/mask buffers (see W_rec docstring).
        if self.wrec_param == "column_dale":
            # Dense mode: random init at w_init_scale across all entries
            # (sparsity is not enforced; w_init_mode is ignored).
            w_init_scale = getattr(train_config, "w_init_scale", 0.01)
            S_init = torch.randn(N, N, dtype=W_con.dtype) * w_init_scale
        elif w_init_mode == "zeros":
            S_init = torch.zeros_like(self.W_con_mask)
        elif w_init_mode == "randn":
            w_init_scale = getattr(train_config, "w_init_scale", 0.01)
            S_init = torch.randn_like(self.W_con_mask) * w_init_scale * self.W_con_mask
        elif w_init_mode == "w_con":
            # Init S so the effective W_rec equals W_con exactly:
            #   "edge_magnitude" → |S|·sign(W_con) = W_con  ⇒  S = |W_con|
            #   "edge_free"      → S·mask         = W_con  ⇒  S = W_con
            if self.wrec_param == "edge_magnitude":
                S_init = W_con.abs().clone()
            else:
                S_init = W_con.clone()
        else:  # 'const'
            w_init_scale = getattr(train_config, "w_init_scale", 0.01)
            S_init = w_init_scale * self.W_con_mask
        self.S = nn.Parameter(S_init)

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

        # --- Metadata for drosophila_cx_eval helpers --------------------
        self.neuron_types = neuron_types
        self.type_names = type_names
        self.epg_indices = np.arange(n_epg, dtype=np.int64)
        self.epg_glom_ix = epg_glom_ix

        # --- Velocity gating (PEN-specific, CX-only) -------------------
        # `pen_only`: zero W_in[:, 0] outside PEN rows; per-unit weights free.
        # `pen_4scalar` (Hulse 2025 strict): 4 learnable scalars
        #               (L/R × PENa/PENb) broadcast onto subpopulations,
        #               signs initialised opposite for L vs R.
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
        # No self-connections, matching the GNN convention (edge_index
        # never includes self-loops).
        self.register_buffer(
            "_no_diag", 1.0 - torch.eye(N, dtype=torch.float32),
            persistent=False,
        )

        # --- Optional image-derived binary mask on W_rec ----------------
        # When set, dark pixels of the image become forbidden recurrent
        # connections. The image is resized to N×N and thresholded at its
        # median.
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

        # --- Recurrent bias (Hulse: initialised to 1) ------------------
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
        # Flyvis injects `noise_recurrent_level * randn` at every recurrent
        # step. Smooths the long-T BPTT landscape; 0 = off (Hulse default).
        self.noise_recurrent_level = float(
            getattr(config.training, "noise_recurrent_level", 0.0)
        )

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Effective recurrent weight
    # ------------------------------------------------------------------

    @property
    def W_rec(self) -> torch.Tensor:
        """Effective recurrent matrix, diagonal masked to zero.

        Layout: row i = postsynaptic, col j = presynaptic. Recurrent input
        is computed as `r @ W_rec.T` (Hulse/Beiran convention).

            "edge_magnitude" → W_rec = |S| ⊙ sign(W_con)
            "edge_free"      → W_rec =  S  ⊙ mask(W_con)
            "column_dale"    → W_rec = |S| ⊙ col_sign[None, :]
        """
        if self.wrec_param == "column_dale":
            W = self.S.abs() * self.col_sign.unsqueeze(0)
        elif self.wrec_param == "edge_magnitude":
            W = self.S.abs() * self.W_con_sign
        else:  # "edge_free"
            W = self.S * self.W_con_mask
        return W * self._no_diag * self._image_mask

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
            h_buf: (B, T, N)        subthreshold activity (for diagnostics
                                   and the circular-TV regulariser).
        """
        B, T, _ = u.shape
        N = self.n_units

        h = (torch.zeros(B, N, dtype=u.dtype, device=u.device)
             if h0 is None else h0)
        h_buf = torch.empty(B, T, N, dtype=u.dtype, device=u.device)

        # W_rec layout: row i = post, col j = pre, so the recurrent input is
        # rec[b, i] = sum_j W_rec[i, j] · r[b, j] = (r @ W_rec.T)[b, i].
        W_rec = self.W_rec
        dt_over_tau = self.dt / self.tau
        noise_lvl = (self.noise_recurrent_level
                     if (self.training and self.noise_recurrent_level > 0)
                     else 0.0)

        for t in range(T):
            r = self._sigma(h)
            # W_rec inherits J_effective's [post, pre] orientation from the
            # loader (Dale on cols = pre). The biologically-correct recurrent
            # input is `r @ W_rec.T` (matches Hulse/Beiran reference code:
            # `h += alpha * (-h + g · σ(h+b) @ J^T + I) / tau`).
            rec = r @ W_rec.T
            inp = self._project_in(u[:, t, :])
            h = h + dt_over_tau * (-h + rec + inp + self.b)
            if noise_lvl > 0:
                h = h + noise_lvl * torch.randn_like(h)
            h_buf[:, t, :] = h

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
