"""TaskRNN — Hulse-style recurrent network for task training.

Architecture (Hulse Methods Eqs. 1, 9-11; Yang 2019 multitask):

    τ * dh_j/dt = -h_j + Σ_k W_rec[j,k] σ(h_k) + Σ_l W_in[j,l] u_l + b_j
    y_hat_i     = Σ_j W_out[i,j] σ(h_j) + b_out_i

Two parameterisations of the recurrent matrix, selected by
`graph_model.W_param`:

  - "sign_locked"  (CX / Hulse path-integration default):
                   W_rec = |S| ⊙ W_con. Sign and sparsity are locked to
                   the connectome at init; only per-edge magnitudes |S|
                   are learned. Requires `task.path_integration` block.
  - "free"         (cortex / Yang multitask):
                   W_rec is a plain (N, N) learnable Parameter. No
                   biological prior, no Dale, no sparsity. Reads
                   `n_units`, `n_input`, `n_output` directly from
                   `graph_model`.

σ is configurable via `graph_model.recurrent_activation`:
  sigmoid (Hulse default) | relu | tanh | softplus.

W_in / W_out are independently configurable as either learnable matrices
(Hulse default) or small MLPs reusing `graph_model.hidden_dim` /
`graph_model.n_layers`.

Buffer protocol matches `teachers.JaneliaCxRNN` (W_rec, W_con,
_block_mask_i, _ring_order_<name>, dt, n_units, neuron_types, type_names,
epg_indices, epg_glom_ix) so the helpers in `models.cx_eval`
(path_integration_accuracy, bump_fwhm, _save_training_snapshot,
_deterministic_sweep_rollout) work on this class **in sign_locked mode**
without branching. In free mode those CX-specific buffers are absent;
cortex helpers should use `models.cortex_eval` instead.

Registered names: "drosophila_cx_pi", "cortex_delaygo", "task_rnn"
(canonical), "neural_task_gnn" (legacy alias).
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
    """Load `path`, resize to (N, N), threshold at median → binary (N, N).

    Light pixels become 1 (connection allowed), dark pixels become 0
    (connection forbidden). Used as a structural prior on W_rec.
    """
    from PIL import Image
    if not os.path.isfile(path):
        # try to resolve via the repo's config helper (some users pass a
        # path relative to the repo data root)
        from connectome_gnn.utils import get_data_root
        cand = os.path.join(get_data_root(), path)
        if os.path.isfile(cand):
            path = cand
        else:
            raise FileNotFoundError(f"w_mask_image_path not found: {path}")
    img = Image.open(path).convert("L")  # grayscale 0..255
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
    "drosophila_cx_pi",
    "task_rnn",
    "neural_task_gnn",
    "cortex_all",
    "cortex_all_unique",
    *(f"cortex_{t}" for t in _CORTEX_TASKS),
)
class TaskRNN(nn.Module):
    """Configurable recurrent network for task training (CX & cortex)."""

    def __init__(self, aggr_type: str = "add", config=None, device=None):
        super().__init__()

        self.device = device
        self.aggr_type = aggr_type

        sim = config.simulation
        gm = config.graph_model
        task = config.task

        W_param = str(getattr(gm, "W_param", "sign_locked")).lower()
        self.W_param = W_param
        # Honour the unified sign-lock toggle (default True = current
        # behaviour). Only meaningful in CX (sign_locked) mode; in the
        # free-W branch this is read but ignored by `W_rec`.
        self.lock_edge_signs = bool(getattr(gm, "lock_edge_signs", True))

        tc = getattr(config, "training", None)
        w_init_mode = str(getattr(tc, "w_init_mode", "const")).lower()
        # Default scale: 0.01 for sign_locked (Hulse), 1.0 for free (Yang).
        default_scale = 0.01 if W_param == "sign_locked" else 1.0
        w_init_scale = float(getattr(tc, "w_init_scale", default_scale))

        if W_param == "sign_locked":
            self._init_cx_branch(sim, gm, task, w_init_mode, w_init_scale)
        elif W_param == "free":
            self._init_free_branch(gm, w_init_mode, w_init_scale)
        else:
            raise ValueError(
                f"graph_model.W_param must be 'sign_locked' or 'free', got {W_param!r}"
            )

        N = self.n_units
        # Zero-diagonal mask: TaskRNN's effective W_rec has no self-connections,
        # matching the GNN convention (edge_index never includes self-loops).
        self.register_buffer(
            "_no_diag", 1.0 - torch.eye(N, dtype=torch.float32),
            persistent=False,
        )
        # Optional image-derived binary mask on W_rec. When unset, this is a
        # ones matrix (no effect); when set, dark pixels of the image become
        # forbidden recurrent connections. The image is resized to N×N and
        # thresholded at its median.
        img_path = str(getattr(gm, "w_mask_image_path", "")).strip()
        if img_path:
            img_mask = _load_image_mask(img_path, N)
        else:
            img_mask = torch.ones(N, N, dtype=torch.float32)
        self.register_buffer("_image_mask", img_mask, persistent=False)

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

        # --- Recurrent activation σ (configurable) -----------------------
        act_name = str(getattr(gm, "recurrent_activation", "sigmoid")).lower()
        if act_name not in _ACT_MAP:
            raise ValueError(
                f"recurrent_activation must be one of {list(_ACT_MAP)}, "
                f"got {act_name!r}"
            )
        self.recurrent_activation_name = act_name
        self._sigma = _ACT_MAP[act_name]

        # --- Stochastic regularisation during BPTT ----------------------
        # Flyvis injects `noise_recurrent_level * randn` at every recurrent
        # step (recurrent_step.py:_standard_recurrent_loss). Smooths the
        # long-T BPTT landscape and is one of the most effective stabilisers
        # for connectome-locked recurrent training. 0 = off (Hulse default).
        self.noise_recurrent_level = float(
            getattr(config.training, "noise_recurrent_level", 0.0)
        )

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Mode-specific init helpers
    # ------------------------------------------------------------------

    def _init_cx_branch(self, sim, gm, task, w_init_mode: str, w_init_scale: float) -> None:
        """Sign-locked W_rec = |S| ⊙ W_con. Loads CX connectome, builds CX
        regulariser buffers (type-pair masks, ring orderings)."""
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
        if w_init_mode == "zeros":
            S_init = torch.zeros_like(self.W_con_mask)
        elif w_init_mode == "randn":
            S_init = torch.randn_like(self.W_con_mask) * w_init_scale * self.W_con_mask
        else:  # 'const'
            S_init = w_init_scale * self.W_con_mask
        self.S = nn.Parameter(S_init)
        self._W_rec_free = None  # marker: use S-based W_rec property

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

        # --- CX metadata exposed for cx_eval helpers ---------------------
        self.neuron_types = neuron_types
        self.type_names = type_names
        self.epg_indices = np.arange(n_epg, dtype=np.int64)
        self.epg_glom_ix = epg_glom_ix

        # --- Optional velocity-channel anatomical gate -------------------
        # `velocity_gate: pen_only`   — zero W_in[:, 0] outside PENa/PENb
        #                                rows; per-unit weights stay free.
        # `velocity_gate: pen_4scalar` — strict Hulse 2025: 4 learnable
        #                                scalars (L/R × PENa/PENb) broadcast
        #                                onto their subpopulations, signs
        #                                initialised opposite for L vs R.
        # In either case, channels 1-2 (initial-bump cue) stay free for
        # all rows.
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
            # One-hot indicator buffers: row k is 1 iff unit k belongs to
            # the subpop. Lets us write v_col = Σ_subpop indicator · scalar
            # without in-place ops (autograd-friendly).
            for key in required:
                ind = torch.zeros(N, dtype=torch.float32)
                ind[torch.as_tensor(pen_subpop[key], dtype=torch.long)] = 1.0
                self.register_buffer(f"_pen_ind_{key.lower()}", ind, persistent=False)
            # 4 velocity scalars; init opposite signs for L/R so the
            # symmetry-breaking starts in the right direction.
            self.v_pena_l = nn.Parameter(torch.tensor(0.01))
            self.v_pena_r = nn.Parameter(torch.tensor(-0.01))
            self.v_penb_l = nn.Parameter(torch.tensor(0.01))
            self.v_penb_r = nn.Parameter(torch.tensor(-0.01))
        elif self.velocity_gate != "none":
            raise ValueError(
                f"graph_model.velocity_gate must be 'none', 'pen_only', or "
                f"'pen_4scalar', got {self.velocity_gate!r}"
            )

        # --- Dynamics constants (from task.path_integration for CX) -----
        pi = task.path_integration
        self.tau = float(getattr(pi, "tau", 0.1))
        self.dt = float(pi.dt)

    def _init_free_branch(self, gm, w_init_mode: str, w_init_scale: float) -> None:
        """Fully-learnable W_rec (N, N). Reads n_units/n_input/n_output from
        graph_model; no biological prior, no CX-specific buffers."""
        N = int(getattr(gm, "n_units", 0))
        self.n_units = N
        self.n_input = int(getattr(gm, "n_input", 0))
        self.n_output = int(getattr(gm, "n_output", 0))
        if N <= 0 or self.n_input <= 0 or self.n_output <= 0:
            raise ValueError(
                "W_param='free' requires graph_model.n_units, n_input, "
                f"n_output > 0; got n_units={N}, n_input={self.n_input}, "
                f"n_output={self.n_output}"
            )

        if w_init_mode == "zeros":
            W_init = torch.zeros(N, N, dtype=torch.float32)
        elif w_init_mode == "randn":
            W_init = torch.randn(N, N, dtype=torch.float32) * w_init_scale
        elif w_init_mode == "uniform_scaled":
            bound = w_init_scale / math.sqrt(N)
            W_init = (torch.rand(N, N, dtype=torch.float32) * 2.0 - 1.0) * bound
        else:  # 'randn_scaled', 'const', or unknown → Yang-style edge-of-chaos
            W_init = torch.randn(N, N, dtype=torch.float32) * (w_init_scale / math.sqrt(N))
        self._W_rec_free = nn.Parameter(W_init)

        # S, CX buffers, and CX metadata are not present in free mode.
        self.S = None
        self._block_names = []
        self._ring_names = []

        # Dynamics constants from graph_model (defaults match Hulse).
        self.tau = float(getattr(gm, "tau", 0.1))
        self.dt = float(getattr(gm, "dt", 0.02))

    # ------------------------------------------------------------------
    # Effective recurrent weight
    # ------------------------------------------------------------------

    @property
    def W_rec(self) -> torch.Tensor:
        """Effective recurrent matrix. Diagonal is always masked to zero.
        Convention: W_rec[j, i] = weight from presynaptic neuron j onto
        postsynaptic neuron i, matching the GNN's (src=pre, dst=post)
        edge_index layout.

        CX (sign_locked architecture):
            lock_edge_signs=True  → W_rec = |S| ⊙ W_con_sign  (Dale-conformant)
            lock_edge_signs=False → W_rec = S ⊙ W_con_mask    (free sign per edge,
                                                              topology still fixed)
        Free architecture:
            W_rec is the learnable Parameter directly.
        """
        if self.W_param == "sign_locked":
            if getattr(self, "lock_edge_signs", True):
                W = self.S.abs() * self.W_con_sign
            else:
                W = self.S * self.W_con_mask
        else:
            W = self._W_rec_free
        return W * self._no_diag * self._image_mask

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------

    def _project_in(self, u_t: torch.Tensor) -> torch.Tensor:
        """(B, n_input) -> (B, N)."""
        if self.input_proj == "matrix":
            W = self.W_in
            if getattr(self, "velocity_gate", "none") == "pen_4scalar":
                # Build the velocity column from the 4 PEN subpop scalars;
                # keep the cue columns (1, 2) of self.W_in as-is.
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

        # W_rec layout: row j = presynaptic, col i = postsynaptic so
        # rec[b, i] = sum_j r[b, j] · W_rec[j, i] = (r @ W_rec)[b, i].
        # This matches the GNN's edge_index convention (src=pre, dst=post),
        # so a learned (j -> i) edge weight in the GNN maps directly to
        # W_rec[j, i] here without any transpose. Compute the property once
        # before the time loop (it materialises the masked diagonal).
        W_rec = self.W_rec
        dt_over_tau = self.dt / self.tau
        # Inject noise only during training (eval/snapshot stays deterministic).
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

        y_hat = self._project_out(self._sigma(h_buf))
        return y_hat, h_buf

    # ------------------------------------------------------------------
    # Regularisers (CX-only; return 0 in free mode)
    # ------------------------------------------------------------------

    def loss_cos_distance(self, lam: float = 1.0) -> torch.Tensor:
        """Hulse Eq. 10: per-(post-type, pre-type) block cosine-distance
        between W_rec and the connectome template W_con. Returns 0 in
        free mode (no W_con)."""
        if not self._block_names or self.W_param != "sign_locked":
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
        """Hulse Eq. 11: soft lower bound on mean |W| per type-pair block.
        Returns 0 in free mode (no type-pair blocks)."""
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
        """Circular total-variation penalty on EPG/PEN ring firing rates.
        Returns 0 in free mode (no ring assignments)."""
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
