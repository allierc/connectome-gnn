"""ZebrafishHdTaskGNN — sign-locked GNN for the larval-zebrafish dIPN
heading-direction ring (companion of :mod:`zebrafish_hd_task_rnn`).

Standalone class, no inheritance and no runtime imports from the drosophila
tree. Hybrid of the TaskRNN encoder/decoder shell with a per-edge GNN
recurrent core. The dense `r @ W_rec` matmul of :class:`ZebrafishHdTaskRNN`
is replaced by a NeuralGNN-style update:

    msg_e    = W_edge[e] · g_phi(v_src, a_src)²   (B, E, 1) — raw v=h
    agg_j    = Σ_{e: dst(e)=j} msg_e              (B, N, 1)
    rec_j    = f_theta(v_j, a_j, agg_j)           (B, N)
    τ · dh/dt = rec + W_in·u

Connectome topology is enforced via `edge_index = nonzeros(W_con)`.
Sign-lock toggle:
    lock_edge_signs=True  → effective per-edge weight = |W| · sign_GT
                            (Dale-conformant; magnitudes learned)
    lock_edge_signs=False → effective per-edge weight = W
                            (sign learned per edge; topology still fixed)

Vocabulary is fish-native: `n_dipn`, `dipn_ix`, `output_from_dipn_only`,
`afferent_subpop_ix` (keyed `RIPN_L/R`, `ptIPN_L/R`), and the four
trainable afferent-gate scalars `v_ripn_l/r`, `v_ptipn_l/r`. Display
labels: "r1π / dIPN" / "RIPN / pt-IPN".

Checkpoints from the inheritance era (`v_pena_*` / `v_penb_*` keys in
state-dict) are NOT loadable by this class — Step 0 retrains from
scratch and validates by comparing fresh metrics to the previous
results. See `docs/REFACTOR_zebrafish_circuit_registry.md` Step 0
acceptance.

Registered name: ``zebrafish_hd_si_gnn``.
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


def _build_type_pair_blocks(
    neuron_types: np.ndarray, type_names: list, W_con: np.ndarray,
) -> "dict[str, torch.Tensor]":
    """(post-type → pre-type) boolean masks over (N, N) for blocks that
    carry at least one non-zero W_con entry."""
    blocks: dict = {}
    nt = np.asarray(neuron_types).astype(np.int64)
    unique = sorted(set(nt.tolist()))
    for q in unique:
        post_mask = nt == q
        for p in unique:
            pre_mask = nt == p
            block = np.outer(post_mask, pre_mask)
            if block.sum() == 0:
                continue
            sub = W_con[block]
            if np.abs(sub).sum() < 1e-12:
                continue
            tp_name = f"{type_names[int(p)]}->{type_names[int(q)]}"
            blocks[tp_name] = torch.from_numpy(block.astype(np.bool_))
    return blocks


@register_model("zebrafish_hd_si_gnn")
class ZebrafishHdTaskGNN(nn.Module):
    """Sign-locked larval-zebrafish dIPN heading-direction GNN."""

    bump_label: str = "r1π / dIPN"
    afferent_label: str = "RIPN / pt-IPN"

    def __init__(self, aggr_type: str = "add", config=None, device=None):
        super().__init__()
        self.device = device
        self.aggr_type = aggr_type

        sim = config.simulation
        gm = config.graph_model
        task = config.task

        self.lock_edge_signs = bool(getattr(gm, "lock_edge_signs", True))

        # --- Load fish HD connectome ------------------------------------
        # Either the named-circuit registry (``config.circuit.name``) or
        # the legacy loader path. Both produce the same canonical dict
        # shape consumed below. See zebrafish_hd_task_rnn.py for the
        # rationale.
        circuit_cfg = getattr(config, "circuit", None)
        if circuit_cfg is not None and getattr(circuit_cfg, "name", None):
            from connectome_gnn.generators.circuits import get_circuit
            cx = get_circuit(circuit_cfg.name).as_loader_dict()
        else:
            from connectome_gnn.generators.connconstr_data import (
                load_zebrafish_hd_connectome,
            )
            cx = load_zebrafish_hd_connectome(sim.connconstr_datapath)
        N = int(cx["N"])
        self.n_units = N
        self.n_input = 3
        self.n_output = 2

        W_con = torch.from_numpy(cx["J_effective"].astype(np.float32))
        self.register_buffer("W_con", W_con)
        self.register_buffer("W_con_sign", torch.sign(W_con))
        self.register_buffer("W_con_mask", (W_con != 0).to(torch.float32))

        # --- Edge index from W_con_mask (row=post, col=pre) -------------
        # _edge_index[0] = post (row of W_con); _edge_index[1] = pre (col).
        src_post, dst_pre = self.W_con_mask.nonzero(as_tuple=True)
        edge_index = torch.stack([src_post, dst_pre], dim=0).long().contiguous()
        self.register_buffer("_edge_index", edge_index, persistent=False)
        self.n_edges = int(edge_index.shape[1])

        # --- Per-edge sign buffer (used when lock_edge_signs=True) ------
        if self.lock_edge_signs:
            sign_e = self.W_con_sign[src_post, dst_pre].to(torch.float32)
            self.register_buffer(
                "_edge_sign", sign_e.unsqueeze(-1), persistent=False,
            )

        # --- Type-pair masks for cos-distance / norm-floor regularisers
        neuron_types = np.asarray(cx["neuron_types"]).astype(np.int64)
        type_names = list(cx["type_names"])
        type_pair_blocks = _build_type_pair_blocks(
            neuron_types, type_names, cx["J_effective"].astype(np.float32)
        )
        self._block_names: list[str] = list(type_pair_blocks.keys())
        for i, name in enumerate(self._block_names):
            self.register_buffer(
                f"_block_mask_{i}", type_pair_blocks[name].to(torch.bool),
                persistent=False,
            )

        # --- Ring orderings for circular-TV regulariser -----------------
        n_dipn = int(cx.get("n_dipn", cx["n_epg"]))
        dipn_glom_ix = np.asarray(cx.get("dipn_ix", cx["epg_ix"]), dtype=np.int64)
        self._ring_names: list[str] = []
        if n_dipn > 0 and dipn_glom_ix.size == n_dipn:
            dipn_idx_arr = np.arange(n_dipn, dtype=np.int64)
            sort = np.argsort(dipn_glom_ix, kind="stable")
            order = torch.from_numpy(dipn_idx_arr[sort]).long()
            self.register_buffer("_ring_order_dIPN", order, persistent=False)
            self._ring_names.append("dIPN")

        # --- Metadata for downstream eval / plot helpers ----------------
        # epg_indices / epg_glom_ix kept as aliases because the multi-
        # species trainer (graph_trainer.py:2237, 2265-2266) and plot_cx
        # still read those names. A follow-up commit can migrate them.
        self.neuron_types = neuron_types
        self.type_names = type_names
        self.dipn_indices = np.arange(n_dipn, dtype=np.int64)
        self.dipn_glom_ix = dipn_glom_ix
        self.epg_indices = self.dipn_indices      # back-compat alias
        self.epg_glom_ix = self.dipn_glom_ix      # back-compat alias

        # --- Velocity gating (4-scalar afferent gate) -------------------
        # ``pen_4scalar``: 4 learnable scalars driving the
        # angular-velocity channel. For zebrafish:
        #     v_ripn_l/r   ← habenula → IPN afferents
        #     v_ptipn_l/r  ← pretectum → IPN afferents
        # Channels 1-2 (initial-bump cue) stay free for all rows.
        self.velocity_gate = str(getattr(gm, "velocity_gate", "none")).lower()
        if self.velocity_gate == "pen_only":
            mask = torch.zeros(N, self.n_input, dtype=torch.float32)
            mask[:, 1:] = 1.0
            self.register_buffer("_W_in_mask", mask, persistent=False)
        elif self.velocity_gate == "pen_4scalar":
            afferent = cx.get("afferent_subpop_ix", None)
            if afferent is None:
                pen = cx.get("pen_subpop_ix", {})
                afferent = {
                    "RIPN_L":  pen.get("PENa_L", np.array([], dtype=np.int64)),
                    "RIPN_R":  pen.get("PENa_R", np.array([], dtype=np.int64)),
                    "ptIPN_L": pen.get("PENb_L", np.array([], dtype=np.int64)),
                    "ptIPN_R": pen.get("PENb_R", np.array([], dtype=np.int64)),
                }
            required = ("RIPN_L", "RIPN_R", "ptIPN_L", "ptIPN_R")
            missing = [k for k in required
                        if k not in afferent or len(afferent[k]) == 0]
            if missing:
                raise ValueError(
                    f"velocity_gate='pen_4scalar' requires non-empty "
                    f"afferent_subpop_ix for {required}; missing/empty: {missing}"
                )
            for key in required:
                ind = torch.zeros(N, dtype=torch.float32)
                ind[torch.as_tensor(afferent[key], dtype=torch.long)] = 1.0
                self.register_buffer(
                    f"_afferent_ind_{key.lower()}", ind, persistent=False,
                )
            self.v_ripn_l  = nn.Parameter(torch.tensor(0.01))
            self.v_ripn_r  = nn.Parameter(torch.tensor(-0.01))
            self.v_ptipn_l = nn.Parameter(torch.tensor(0.01))
            self.v_ptipn_r = nn.Parameter(torch.tensor(-0.01))
        elif self.velocity_gate != "none":
            raise ValueError(
                f"graph_model.velocity_gate must be 'none', 'pen_only', or "
                f"'pen_4scalar', got {self.velocity_gate!r}"
            )

        # --- Dynamics constants -----------------------------------------
        if task.task_type != "swim_integration":
            raise ValueError(
                f"ZebrafishHdTaskGNN requires task_type='swim_integration'; "
                f"got {task.task_type!r}"
            )
        task_block = task.swim_integration
        self.tau = float(getattr(task_block, "tau", 0.1))
        self.dt = float(task_block.dt)

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

        # --- Decoder W_out (matrix or MLP) ------------------------------
        self.output_from_dipn_only = bool(getattr(gm, "output_from_dipn_only", False))
        self._readout_dim = int(n_dipn) if self.output_from_dipn_only else N
        self.output_proj = getattr(gm, "output_proj", "matrix")
        if self.output_proj == "matrix":
            self.W_out = nn.Parameter(
                torch.empty(self.n_output, self._readout_dim, dtype=torch.float32)
            )
            nn.init.kaiming_uniform_(self.W_out, a=math.sqrt(5))
            self.b_out = nn.Parameter(torch.zeros(self.n_output, dtype=torch.float32))
            self._W_out_mlp = None
        elif self.output_proj == "mlp":
            self.W_out = None
            self.b_out = None
            self._W_out_mlp = MLP(
                input_size=self._readout_dim, output_size=self.n_output,
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
        # Shrink last-layer weights of both MLPs by 10× so initial recurrent
        # drive is small (forward stability).
        with torch.no_grad():
            self.g_phi.layers[-1].weight.mul_(0.1)
            self.f_theta.layers[-1].weight.mul_(0.1)

        # Per-edge weight W (sign-free; sign locked via _edge_sign buffer
        # + _effective_edge_weights when lock_edge_signs).
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
        elif w_init_mode == "w_con":
            w_con_edges = W_con[src_post, dst_pre].to(torch.float32)
            if self.lock_edge_signs:
                w_con_edges = w_con_edges.abs()
            W_init = w_con_edges.unsqueeze(-1).contiguous()
        else:  # 'randn' or 'const' fallthrough
            W_init = torch.randn(self.n_edges, 1, dtype=torch.float32) * w_init_scale
        self.W = nn.Parameter(W_init)

        # Per-node embedding a (ones init).
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
        diagonal masked. Only the linear part of the GNN update appears
        here — `f_theta` does not. Used by the cos-distance / norm-floor
        regularisers and the GT-vs-learned scatter (per-edge gains only).
        """
        N = self.n_units
        W_dense = self.W.new_zeros(N, N)
        src_post, dst_pre = self._edge_index[0], self._edge_index[1]
        W_dense[src_post, dst_pre] = self._effective_edge_weights().squeeze(-1)
        return W_dense * self._no_diag * self._image_mask

    # ------------------------------------------------------------------
    # GNN recurrent step
    # ------------------------------------------------------------------

    def _gnn_recurrent_drive(self, v: torch.Tensor) -> torch.Tensor:
        """(B, N) subthreshold state v (= h) → (B, N) recurrent du/dt.

        Both g_phi and f_theta consume the raw state `v` directly. The
        sigmoid is reserved for the decoder (cos/sin readout from firing
        rates), preserving the TaskRNN output contract.

        Edge convention: `_edge_index` is built from `W_con_mask.nonzero()`
        where `W_con` is [post, pre]. So
        `_edge_index[0] = post` and `_edge_index[1] = pre`. For
        biologically-correct GNN message flow (pre → post) we use:
          src = _edge_index[1]  (pre — message source)
          dst = _edge_index[0]  (post — accumulation target)
        Matches Beiran/Hulse's `@ J^T` convention.
        """
        B, N = v.shape
        src = self._edge_index[1]   # pre  (col of W_con)
        dst = self._edge_index[0]   # post (row of W_con)

        v_src = v[:, src].unsqueeze(-1)                       # (B, E, 1)
        a_src = self.a[src].unsqueeze(0).expand(B, -1, -1)    # (B, E, emb)
        edge_feat = torch.cat([v_src, a_src], dim=-1)
        g_out = self.g_phi(edge_feat)                         # (B, E, 1)
        if self._g_phi_positive:
            g_out = g_out ** 2
        edge_w = self._effective_edge_weights()               # (E, 1)
        msg = edge_w.unsqueeze(0) * g_out                     # (B, E, 1)

        agg = v.new_zeros(B, N, 1)
        agg.scatter_add_(1, dst.view(1, -1, 1).expand(B, -1, 1), msg)

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
                    self._afferent_ind_ripn_l  * self.v_ripn_l
                    + self._afferent_ind_ripn_r  * self.v_ripn_r
                    + self._afferent_ind_ptipn_l * self.v_ptipn_l
                    + self._afferent_ind_ptipn_r * self.v_ptipn_r
                )
                W = torch.cat([v_col.unsqueeze(1), W[:, 1:]], dim=1)
            else:
                mask = getattr(self, "_W_in_mask", None)
                if mask is not None:
                    W = W * mask
            return u_t @ W.t()
        return self._W_in_mlp(u_t)

    def _project_out(self, r: torch.Tensor) -> torch.Tensor:
        """(B, T, N) -> (B, T, n_output).

        With ``output_from_dipn_only=True`` the decoder sees only the first
        ``n_dipn`` columns of ``r`` (the dIPN block).
        """
        if self.output_from_dipn_only:
            r = r[..., : self._readout_dim]
        if self.output_proj == "matrix":
            return r @ self.W_out.t() + self.b_out
        return self._W_out_mlp(r)

    def forward(
        self,
        u: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Run the network for T timesteps over a batch.

        Args:
            u: (B, T, n_input) input stream.
            h0: (B, N) initial subthreshold activity (zeros if None).

        Returns:
            y_hat: (B, T, n_output) readout.
            h_buf: (B, T, N) subthreshold activity.
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
            # leak outside the MLP (flyvis GNN convention).
            rec = self._gnn_recurrent_drive(h)
            inp = self._project_in(u[:, t, :])
            h = h + dt_over_tau * (rec + inp)
            if noise_lvl > 0:
                h = h + noise_lvl * torch.randn_like(h)
            # NaN guard — kicks in only when f_theta hasn't yet learned a
            # leak. Clamp is invisible in healthy training where |h| stays
            # well below the bound.
            h = h.clamp(-50.0, 50.0)
            h_buf[:, t, :] = h

        y_hat = self._project_out(self._sigma(h_buf))
        return y_hat, h_buf

    # ------------------------------------------------------------------
    # Regularisers
    # ------------------------------------------------------------------

    def loss_cos_distance(self, lam: float = 1.0) -> torch.Tensor:
        """Per-(post-type, pre-type) block cosine-distance between W_rec
        and W_con (Hulse Eq. 10)."""
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
        """Soft lower bound on mean |W| per type-pair block (Hulse Eq. 11)."""
        if not self._block_names:
            return self.W_rec.new_zeros(())
        total = self.W_rec.new_zeros(())
        for i in range(len(self._block_names)):
            mask = getattr(self, f"_block_mask_{i}")
            mean_abs = self.W_rec[mask].abs().mean()
            slack = F.relu(kappa - mean_abs)
            total = total + slack.pow(2)
        return lam * total / max(len(self._block_names), 1)

    def loss_f_theta_diff(self, h_buf: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        """Negative-monotonicity prior on ∂f_θ/∂h.

        Penalises positive slope of f_θ w.r.t. its state input (v ≡ h,
        column 0 of the f_θ feature vector). Forces f_θ to learn the leak
        term that was dropped from the explicit forward, which is the
        stabiliser that prevents runaway integration.
        """
        if lam <= 0 or not hasattr(self, "f_theta"):
            return h_buf.new_zeros(())
        h_last = h_buf[:, -1, :].detach()                        # (B, N)
        B, N = h_last.shape
        a_exp = self.a.unsqueeze(0).expand(B, -1, -1)            # (B, N, emb)
        agg = h_last.new_zeros(B, N, 1)
        feat = torch.cat(
            [h_last.unsqueeze(-1), a_exp, agg], dim=-1
        )                                                         # (B, N, 1+emb+1)
        dv = 0.05 * h_last.abs().max().clamp(min=1e-6)
        feat_next = feat.clone()
        feat_next[..., 0] = feat_next[..., 0] + dv
        f0 = self.f_theta(feat)
        f1 = self.f_theta(feat_next)
        return lam * F.relu(f1 - f0).norm(2)

    def loss_g_phi_diff(self, h_buf: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        """Positive-monotonicity prior on ∂g_φ/∂v.

        Penalises NEGATIVE slope of g_φ w.r.t. its presynaptic-state input
        (v ≡ h). Forces g_φ to be monotonically non-decreasing in v — the
        Dale-conformant prior that 'more presynaptic activity → larger
        message magnitude in the same sign as the edge'. Most useful when
        `g_phi_positive=False`.
        """
        if lam <= 0 or not hasattr(self, "g_phi"):
            return h_buf.new_zeros(())
        h_last = h_buf[:, -1, :].detach()                        # (B, N)
        B, N = h_last.shape
        a_exp = self.a.unsqueeze(0).expand(B, -1, -1)            # (B, N, emb)
        feat = torch.cat([h_last.unsqueeze(-1), a_exp], dim=-1)  # (B, N, 1+emb)
        dv = 0.05 * h_last.abs().max().clamp(min=1e-6)
        feat_next = feat.clone()
        feat_next[..., 0] = feat_next[..., 0] + dv
        g0 = self.g_phi(feat)
        g1 = self.g_phi(feat_next)
        if self._g_phi_positive:
            g0, g1 = g0 ** 2, g1 ** 2
        return lam * F.relu(g0 - g1).norm(2)

    def loss_tv_circular(self, h_buf: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        """Circular total-variation penalty on dIPN ring firing rates."""
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
