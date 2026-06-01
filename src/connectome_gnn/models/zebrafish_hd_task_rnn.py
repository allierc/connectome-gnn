"""ZebrafishHdTaskRNN — sign-locked recurrent network for the larval-zebrafish
dIPN heading-direction ring.

Standalone class, no inheritance and no runtime imports from the drosophila
tree. The dynamics are identical:

    τ · dh_j/dt = -h_j + Σ_k W_rec[j,k] σ(h_k) + Σ_l W_in[j,l] u_l + b_j
    y_hat_i     = Σ_j W_out[i,j] σ(h_j) + b_out_i

with sign-locked W_rec parameterisation (`graph_model.wrec_param`):
    "edge_magnitude" → W_rec = |S| ⊙ sign(W_con)
    "edge_free"      → W_rec =  S  ⊙ mask(W_con)
    "column_dale"    → W_rec = |S| ⊙ col_sign[None, :]

Vocabulary is fish-native: `n_dipn` (number of IPNd*/IPNds* HD cells, 443 in
the 731-cell subset), `dipn_ix` (per-cell ring-bin index along the
mediolateral axis), `output_from_dipn_only` (decode only from the dIPN
block), `afferent_subpop_ix` (keyed `RIPN_L/R`, `ptIPN_L/R`). The
four trainable afferent-gate scalars are `v_ripn_l/r` (habenula → IPN
gain) and `v_ptipn_l/r` (pretectum → IPN gain).

Checkpoints from the inheritance era (`v_pena_*` / `v_penb_*` keys in
state-dict) are NOT loadable by this class — Step 0 retrains from
scratch and validates by comparing fresh metrics to the previous
results. See `docs/REFACTOR_zebrafish_circuit_registry.md` Step 0
acceptance.

Registered name: ``zebrafish_hd_si`` (HD = heading direction,
SI = swim integration).
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
    carry at least one non-zero W_con entry. Used by the cos-distance and
    norm-floor regularisers."""
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


@register_model("zebrafish_hd_si")
class ZebrafishHdTaskRNN(nn.Module):
    """Sign-locked larval-zebrafish dIPN heading-direction RNN."""

    bump_label: str = "r1π / dIPN"
    afferent_label: str = "RIPN / pt-IPN"

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

        # --- Load fish HD connectome ------------------------------------
        # Two paths produce the same canonical dict shape (N, J_effective,
        # neuron_types, type_names, n_dipn/n_epg, dipn_ix/epg_ix,
        # afferent_subpop_ix/pen_subpop_ix, dale_signs, …):
        #
        #   1. ``config.circuit.name`` set → resolve via the named-circuit
        #      registry (``connectome_gnn.generators.circuits``). The
        #      Circuit dataclass is the in-memory builder; ``as_loader_dict``
        #      flattens it back to the legacy access pattern below.
        #   2. ``config.circuit`` absent / name empty → fall through to the
        #      legacy ``load_zebrafish_hd_connectome(sim.connconstr_datapath)``
        #      so existing yamls keep loading byte-equivalently.
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

        # Per-pre-neuron sign for column_dale mode. Cells whose net is
        # zero (orphans whose outgoing partners lie outside the modelled
        # 731-cell subset — ~8 cells in the dIPN pool) fall back to
        # ``cx["dale_signs"]``.
        if self.wrec_param == "column_dale":
            col_sign = torch.sign(W_con.sum(dim=0))
            if (col_sign == 0).any() and "dale_signs" in cx:
                dale = torch.as_tensor(cx["dale_signs"],
                                        dtype=col_sign.dtype,
                                        device=col_sign.device)
                col_sign = torch.where(col_sign == 0, dale, col_sign)
            if (col_sign == 0).any():
                zero_idx = torch.nonzero(col_sign == 0, as_tuple=True)[0].tolist()
                raise ValueError(
                    f"wrec_param='column_dale' requires every pre-neuron to "
                    f"have non-zero net outgoing weight in W_con (or a Dale "
                    f"prior via cx['dale_signs']); col_sign==0 at indices "
                    f"{zero_idx[:10]} (showing first 10)"
                )
            self.register_buffer("col_sign", col_sign)

        # --- Trainable matrix S -----------------------------------------
        if self.wrec_param == "column_dale":
            w_init_scale = getattr(train_config, "w_init_scale", 0.01)
            S_init = torch.randn(N, N, dtype=W_con.dtype) * w_init_scale
        elif w_init_mode == "zeros":
            S_init = torch.zeros_like(self.W_con_mask)
        elif w_init_mode == "randn":
            w_init_scale = getattr(train_config, "w_init_scale", 0.01)
            S_init = torch.randn_like(self.W_con_mask) * w_init_scale * self.W_con_mask
        elif w_init_mode == "w_con":
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
        # The fish loader puts the dIPN cells (IPNd* + IPNds*) into the
        # first ``n_dipn`` rows and assigns each one a ring-bin index in
        # ``dipn_ix`` (mediolateral soma-X discretisation). Set up a single
        # ring order under the name "dIPN" for the TV regulariser.
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
        # ``dipn_indices`` / ``dipn_glom_ix`` are the fish-native primary
        # names; ``epg_*`` aliases are kept because the trainer
        # (graph_trainer.py:2237, 2265-2266) and plot_cx.py still read
        # those keys. A follow-up commit can migrate those call sites.
        self.neuron_types = neuron_types
        self.type_names = type_names
        self.dipn_indices = np.arange(n_dipn, dtype=np.int64)
        self.dipn_glom_ix = dipn_glom_ix
        self.epg_indices = self.dipn_indices      # back-compat alias
        self.epg_glom_ix = self.dipn_glom_ix      # back-compat alias

        # --- Velocity gating (4-scalar afferent gate) -------------------
        # ``pen_4scalar`` (Hulse 2025 strict): 4 learnable scalars driving
        # the angular-velocity channel. For zebrafish:
        #     v_ripn_l/r   ← habenula → IPN afferents (PENa in fly code)
        #     v_ptipn_l/r  ← pretectum → IPN afferents (PENb in fly code)
        # The yaml token stays ``pen_4scalar`` for back-compat with the
        # ``zebrafish-tex-frozen`` configs.
        # Channels 1-2 (initial-bump cue) stay free for all rows.
        self.velocity_gate = str(getattr(gm, "velocity_gate", "none")).lower()
        if self.velocity_gate == "pen_only":
            mask = torch.zeros(N, self.n_input, dtype=torch.float32)
            mask[:, 1:] = 1.0
            self.register_buffer("_W_in_mask", mask, persistent=False)
        elif self.velocity_gate == "pen_4scalar":
            # Prefer the fish-native key ``afferent_subpop_ix`` when the
            # loader emits it; fall back to the legacy ``pen_subpop_ix``
            # mapping (PENa = RIPN, PENb = pt-IPN per the loader docstring).
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
                f"ZebrafishHdTaskRNN requires task_type='swim_integration'; "
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

        # --- Recurrent bias --------------------------------------------
        self.b = nn.Parameter(torch.ones(N, dtype=torch.float32))

        # --- Decoder W_out (matrix or MLP) ------------------------------
        # When ``output_from_dipn_only=True`` the decoder reads only from
        # the first ``n_dipn`` neurons (the dIPN block, indices
        # 0..n_dipn-1). Zebrafish analog of Hulse's frozen wout[0:46,:]
        # convention.
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

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Effective recurrent weight
    # ------------------------------------------------------------------

    @property
    def W_rec(self) -> torch.Tensor:
        """Effective recurrent matrix, diagonal masked to zero.

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
        ``n_dipn`` columns of ``r`` (the dIPN block, indices 0..n_dipn-1).
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
            h_buf: (B, T, N) subthreshold activity (for diagnostics and
                   the circular-TV regulariser).
        """
        B, T, _ = u.shape
        N = self.n_units

        h = (torch.zeros(B, N, dtype=u.dtype, device=u.device)
             if h0 is None else h0)
        h_buf = torch.empty(B, T, N, dtype=u.dtype, device=u.device)

        # W_rec layout: row i = post, col j = pre, so the recurrent input
        # is rec[b, i] = sum_j W_rec[i, j] · r[b, j] = (r @ W_rec.T)[b, i].
        W_rec = self.W_rec
        dt_over_tau = self.dt / self.tau
        noise_lvl = (self.noise_recurrent_level
                     if (self.training and self.noise_recurrent_level > 0)
                     else 0.0)

        for t in range(T):
            r = self._sigma(h)
            rec = r @ W_rec.T
            inp = self._project_in(u[:, t, :])
            h = h + dt_over_tau * (-h + rec + inp + self.b)
            if noise_lvl > 0:
                h = h + noise_lvl * torch.randn_like(h)
            h_buf[:, t, :] = h

        y_hat = self._project_out(self._sigma(h_buf))
        return y_hat, h_buf

    # ------------------------------------------------------------------
    # Regularisers
    # ------------------------------------------------------------------

    def loss_cos_distance(self, lam: float = 1.0) -> torch.Tensor:
        """Per-(post-type, pre-type) block cosine-distance between W_rec
        and the connectome template W_con (Hulse Eq. 10)."""
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
