"""CortexTaskRNN — free-W recurrent network for Yang 2019 multitask
cognitive battery.

Architecture (Yang 2019):

    τ * dh_j/dt = -h_j + Σ_k W_rec[j,k] σ(h_k) + Σ_l W_in[j,l] u_l + b_j
    y_hat_i     = Σ_j W_out[i,j] σ(h_j) + b_out_i

`W_rec` is a plain (N, N) learnable Parameter. No biological prior, no
Dale, no sparsity, no connectome loading. `n_units`, `n_input`,
`n_output` are read directly from `graph_model`.

No CX-specific buffers (W_con, _block_mask_i, _ring_order_*,
neuron_types, EPG/PEN indices) — the Hulse regulariser methods
(`loss_cos_distance`, `loss_norm_floor`, `loss_tv_circular`) are kept
on the class but return zero, so the trainer's loss-computation
branch stays uniform across the CX and cortex pipelines.

Registered names: "cortex_all", "cortex_all_unique", "cortex_<rule>"
for the 20 Yang single-task variants.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


_CORTEX_TASKS = (
    "fdgo", "reactgo", "delaygo", "fdanti", "reactanti", "delayanti",
    "dm1", "dm2", "contextdm1", "contextdm2", "multidm",
    "delaydm1", "delaydm2", "contextdelaydm1", "contextdelaydm2", "multidelaydm",
    "dmsgo", "dmsnogo", "dmcgo", "dmcnogo",
)


@register_model(
    "cortex_all",
    "cortex_all_unique",
    *(f"cortex_{t}" for t in _CORTEX_TASKS),
)
class CortexTaskRNN(nn.Module):
    """Free-W cortex RNN. Plain (N, N) learnable recurrent matrix."""

    # Preserved for trainer log lines that read `model.W_param`. Cortex
    # always operates in free-W mode; the attribute exists for symmetry
    # with DrosophilaCxTaskRNN's API (which has no W_param attribute since it's
    # implicitly sign_locked).
    W_param: str = "free"

    def __init__(self, aggr_type: str = "add", config=None, device=None):
        super().__init__()
        self.device = device
        self.aggr_type = aggr_type

        gm = config.graph_model
        train_config = config.training
        w_init_mode = getattr(train_config, "w_init_mode", "const")

        # --- Dimensions read directly from graph_model -----------------
        N = int(getattr(gm, "n_units", 0))
        self.n_units = N
        self.n_input = int(getattr(gm, "n_input", 0))
        self.n_output = int(getattr(gm, "n_output", 0))
        if N <= 0 or self.n_input <= 0 or self.n_output <= 0:
            raise ValueError(
                "CortexTaskRNN requires graph_model.n_units, n_input, "
                f"n_output > 0; got n_units={N}, n_input={self.n_input}, "
                f"n_output={self.n_output}"
            )

        # --- Free recurrent matrix (N, N) ------------------------------
        # Parameter name kept as `_W_rec_free` for state_dict compatibility
        # with checkpoints saved by the original TaskRNN class.
        if w_init_mode == "zeros":
            W_init = torch.zeros(N, N, dtype=torch.float32)
        elif w_init_mode == "randn":
            w_init_scale = getattr(train_config, "w_init_scale", 1.0)
            W_init = torch.randn(N, N, dtype=torch.float32) * w_init_scale
        elif w_init_mode == "uniform_scaled":
            w_init_scale = getattr(train_config, "w_init_scale", 1.0)
            bound = w_init_scale / math.sqrt(N)
            W_init = (torch.rand(N, N, dtype=torch.float32) * 2.0 - 1.0) * bound
        else:  # 'randn_scaled', 'const', or unknown → Yang-style edge-of-chaos
            w_init_scale = getattr(train_config, "w_init_scale", 1.0)
            W_init = torch.randn(N, N, dtype=torch.float32) * (w_init_scale / math.sqrt(N))
        self._W_rec_free = nn.Parameter(W_init)

        # --- CX-side compat surface ------------------------------------
        # drosophila_cx_eval helpers and the trainer's loss branch read these as
        # "is there a connectome?" sentinels; empty lists mean "no
        # type-pair regularisation / no ring TV".
        self._block_names: list[str] = []
        self._ring_names: list[str] = []

        # --- Dynamics constants (cortex reads from graph_model) --------
        self.tau = float(getattr(gm, "tau", 0.1))
        self.dt = float(getattr(gm, "dt", 0.02))

        # --- Zero-diagonal mask -----------------------------------------
        # No self-connections, matching the GNN convention (edge_index
        # never includes self-loops).
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
        # No velocity gating: cortex has no PEN subpopulations to gate.
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

        # --- Readout input: σ(h) (firing rate, default) or raw h (Yang) ---
        self.readout_uses_sigma = bool(
            getattr(gm, "readout_uses_sigma", True)
        )

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Effective recurrent weight
    # ------------------------------------------------------------------

    @property
    def W_rec(self) -> torch.Tensor:
        """Effective recurrent matrix, diagonal masked to zero.

        Convention: W_rec[j, i] = weight from presynaptic neuron j onto
        postsynaptic neuron i, matching the GNN's (src=pre, dst=post)
        edge_index layout. Just the learnable parameter with the
        zero-diagonal and image masks applied.
        """
        return self._W_rec_free * self._no_diag * self._image_mask

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------

    def _project_in(self, u_t: torch.Tensor) -> torch.Tensor:
        """(B, n_input) -> (B, N). No velocity gating."""
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
            h_buf: (B, T, N)        subthreshold activity.
        """
        B, T, _ = u.shape
        N = self.n_units

        h = (torch.zeros(B, N, dtype=u.dtype, device=u.device)
             if h0 is None else h0)
        h_buf = torch.empty(B, T, N, dtype=u.dtype, device=u.device)

        # W_rec layout: row j = presynaptic, col i = postsynaptic so
        # rec[b, i] = sum_j r[b, j] · W_rec[j, i] = (r @ W_rec)[b, i].
        W_rec = self.W_rec
        dt_over_tau = self.dt / self.tau
        noise_lvl = (self.noise_recurrent_level
                     if (self.training and self.noise_recurrent_level > 0)
                     else 0.0)

        for t in range(T):
            r = self._sigma(h)
            rec = r @ W_rec
            inp = self._project_in(u[:, t, :])
            h = h + dt_over_tau * (-h + rec + inp + self.b)
            if noise_lvl > 0:
                h = h + noise_lvl * torch.randn_like(h)
            h_buf[:, t, :] = h

        readout_input = self._sigma(h_buf) if self.readout_uses_sigma else h_buf
        y_hat = self._project_out(readout_input)
        return y_hat, h_buf

    # ------------------------------------------------------------------
    # Regulariser hooks (no-ops in cortex mode)
    # ------------------------------------------------------------------
    # The path-integration trainer calls these unconditionally; returning
    # zero keeps the trainer code path uniform between CX and cortex
    # without architecture-aware branching.

    def loss_cos_distance(self, lam: float = 1.0) -> torch.Tensor:
        return self.W_rec.new_zeros(())

    def loss_norm_floor(self, lam: float = 1.0, kappa: float = 0.05) -> torch.Tensor:
        return self.W_rec.new_zeros(())

    def loss_tv_circular(self, h_buf: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        return h_buf.new_zeros(())
