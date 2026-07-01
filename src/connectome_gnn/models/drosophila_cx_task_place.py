"""DrosophilaCxTaskPlace — heading + distance + place-cell model.

Two coupled networks, trained jointly end-to-end:

  * Net1 = the sign-locked CX path-integration RNN (``DrosophilaCxTaskRNN``),
    supervised on heading + distance exactly as the ``both`` task. This class
    *subclasses* it so the trainer sees the same ``S`` / ``W_rec`` / ``b`` /
    ``loss_*`` / ``neuron_types`` / ``epg_*`` attributes (param-group naming,
    regularisers and the heading snapshot all keep working unchanged).

  * Net2 = a second, learnable recurrent network that reads Net1's full
    338-cell state each step and emits an allocentric place code. It has
    ``net2_n_interneurons`` interneurons + ``place_grid**2`` place cells on a
    *synthetic* sign-locked Dale connectome ``W2`` (10 %% sparse, 60/40 E/I)
    generated at data-generation time and loaded from the dataset. Only the
    per-edge magnitudes of ``W2`` are learned; an ``encoder`` MLP injects
    Net1's state into the interneurons and a ``decoder`` MLP reads the place
    cells out to ``K`` place-field logits.

Output is ``y_hat = concat(net1 heading+distance [B,T,3], place logits
[B,T,K])``. The place loss/score (KL on the normalised place distribution +
auxiliary population-vector position decode, plus a [0,1] cosine-similarity
score) are encapsulated here so the trainer only adds a small dispatch.

Registered name: ``drosophila_cx_pi_place``.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from connectome_gnn.models.drosophila_cx_task_rnn import DrosophilaCxTaskRNN
from connectome_gnn.models.MLP import MLP
from connectome_gnn.models.registry import register_model
from connectome_gnn.utils import graphs_data_path


@register_model("drosophila_cx_pi_place")
class DrosophilaCxTaskPlace(DrosophilaCxTaskRNN):

    bump_label = "EPG"
    afferent_label = "PEN"

    def __init__(self, aggr_type: str = "add", config=None, device=None):
        # Build Net1 as the heading+distance ("both") CX RNN: force its
        # task_targets so n_input=4 / n_output=3 regardless of the place
        # config's task_targets.
        net1_cfg = config.model_copy(deep=True)
        net1_cfg.training.task_targets = ["rotation", "translation"]
        super().__init__(aggr_type=aggr_type, config=net1_cfg, device=device)
        self._n_head_out = int(self.n_output)          # 3 = [cosθ, sinθ, d]

        si = config.task.swim_integration
        gm = config.graph_model
        n_inter = int(getattr(si, "net2_n_interneurons", 200))
        grid = int(getattr(si, "place_grid", 20))
        K = grid * grid
        N2 = n_inter + K
        self.net2_n_inter = n_inter
        self.net2_n_place = K
        self.net2_N = N2

        # --- synthetic sign-locked E/I recurrent connectome W2 -------------
        # Built deterministically from the config params (net2_sparsity /
        # net2_ei_ratio / net2_seed) rather than loaded from the dataset, so
        # sweeping sparsity / E/I in the config changes W2 without needing a
        # new dataset. For the base params this reproduces the dataset's saved
        # net2_Wcon.npz exactly (same seed → same matrix).
        from connectome_gnn.generators.graph_data_generator import (
            _generate_net2_connectivity,
        )
        W2_np, n_types, ei, type_names = _generate_net2_connectivity(
            n_inter, K,
            float(getattr(si, "net2_sparsity", 0.10)),
            float(getattr(si, "net2_ei_ratio", 0.60)),
            int(getattr(si, "net2_seed", 700000)))
        W2 = torch.from_numpy(W2_np.astype(np.float32))
        self.register_buffer("W2_con", W2)
        self.register_buffer("W2_con_sign", torch.sign(W2))
        # per-neuron metadata (functional type + Dale sign) for plots/tests
        self.net2_neuron_types = np.asarray(n_types).astype(np.int64)
        self.net2_type_names = [str(s) for s in list(type_names)]
        self.net2_ei = np.asarray(ei).astype(np.int64)
        # sign-locked: W2_rec = |S2| ⊙ sign(W2_con); init so W2_rec == W2_con
        self.S2 = nn.Parameter(W2.abs().clone())
        self.register_buffer("net2_place_idx",
                             torch.arange(n_inter, N2, dtype=torch.long))

        # --- encoder MLP (free, dense): Net1 state → interneuron drive -------
        # No decoder: the K place cells ARE the place-field code. Their output
        # layer is a plain tanh (∈(-1,1)); cell k is trained (MSE) to fire its
        # Gaussian place field g_k directly, so the code isn't scrambled by a
        # learnable readout and the anchored seed shows straight through.
        self.net2_encoder = MLP(
            input_size=self.n_units, output_size=n_inter,
            nlayers=int(gm.n_layers), hidden_size=int(gm.hidden_dim),
            activation=gm.MLP_activation, device=device)

        # --- Net2 dynamics constants --------------------------------------
        self.net2_dt = float(self.dt)
        self.net2_tau = float(getattr(si, "net2_tau_s", self.tau))

        # --- place / grid geometry (centres + σ) for loss / decode ---------
        # grid_mode: target is a toroidal grid-cell code (target_kind=
        # "grid_cells") — toroidal Gaussian targets + circular position decode;
        # otherwise the bounded place-cell code.
        self.grid_mode = str(getattr(si, "target_kind", "")).lower() == "grid_cells"
        geo_file = "grid_geometry.npz" if self.grid_mode else "place_geometry.npz"
        geo = np.load(os.path.join(graphs_data_path(config.dataset), geo_file))
        self.register_buffer(
            "place_centers", torch.from_numpy(geo["centers"].astype(np.float32)))
        self.place_sigma = float(geo["sigma"])
        if self.grid_mode:
            self.grid_period = float(geo["period"])
            self.arena_half = self.grid_period       # torus extent for plots
        else:
            self.grid_period = None
            self.arena_half = float(geo["arena_half"])

        # Path-integration anchor: seed the place-cell population at t=0 with
        # the Gaussian code of the (given) start position, then integrate.
        self.place_anchor = bool(getattr(config.training, "place_anchor", False))

        # total output width = heading+distance (3) + place logits (K)
        self.n_output = self._n_head_out + K
        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    @property
    def W2_rec(self) -> torch.Tensor:
        """Sign-locked Net2 recurrent matrix |S2| ⊙ sign(W2_con)."""
        return self.S2.abs() * self.W2_con_sign

    def place_field(self, xy: torch.Tensor) -> torch.Tensor:
        """Raw Gaussian place-field activations g_k ∈ [0,1] (peak 1 at the
        cell's centre) evaluated at position ``xy`` (...,2) → (...,K). This is
        BOTH the anchor seed (at the start) and the tanh-output MSE target
        (every frame). Toroidal wrap in grid_mode."""
        d = xy.unsqueeze(-2) - self.place_centers          # (...,K,2)
        if self.grid_mode:
            L = self.grid_period
            d = d - L * torch.round(d / L)
        d2 = d.pow(2).sum(-1)
        return torch.exp(-d2 / (2.0 * self.place_sigma * self.place_sigma))

    # back-compat alias (start-position bump == place field at pos0)
    def _place_bump(self, pos0: torch.Tensor) -> torch.Tensor:
        return self.place_field(pos0)

    def forward(self, u, h0=None, pos0=None):
        # Net1 (CX) rollout → heading+distance readout y1 and full state h1.
        y1, h1 = super().forward(u, h0=h0)        # y1 (B,T,3), h1 (B,T,N=338)
        r1 = self._sigma(h1)                       # full Net1 state, (B,T,338)
        B, T, _ = r1.shape
        drive = self.net2_encoder(r1)              # (B,T,n_inter) — all t at once
        W2T = self.W2_rec.t()
        a = self.net2_dt / self.net2_tau
        h2 = r1.new_zeros(B, self.net2_N)
        # PI anchor: seed the place cells so their tanh OUTPUT at t=0 equals the
        # start-position Gaussian, i.e. h2_place = atanh(g(pos0)).
        if self.place_anchor and pos0 is not None:
            g0 = self.place_field(pos0.to(r1)).clamp(max=0.999)  # (B,K)
            h2 = h2.index_copy(1, self.net2_place_idx, torch.atanh(g0).to(r1.dtype))
        place = r1.new_empty(B, T, self.net2_n_place)
        pad_K = (0, self.net2_n_place)             # right-pad interneuron drive
        for t in range(T):
            r2 = self._sigma(h2)
            rec2 = r2 @ W2T
            inp2 = F.pad(drive[:, t], pad_K)        # (B,N2): drive→interneurons
            h2 = h2 + a * (-h2 + rec2 + inp2)
            # tanh output layer: the place cells directly emit their field ∈(-1,1)
            place[:, t] = torch.tanh(h2[:, self.net2_place_idx])   # (B,K)
        # h1 (Net1 state) is returned as h_buf so the trainer's circular-TV
        # regulariser and the heading snapshot operate on the compass.
        return torch.cat([y1, place], dim=-1), h1

    # ------------------------------------------------------------------
    # Place-cell target / loss / score (encapsulated; trainer just dispatches)
    # ------------------------------------------------------------------
    def place_targets(self, xy: torch.Tensor) -> torch.Tensor:
        """Normalised Gaussian target distribution q (B,T,K), Σ_k q=1.
        Place: Euclidean Gaussian. Grid: toroidal (wrapped per-axis distance
        on the torus of period λ)."""
        d = xy.unsqueeze(-2) - self.place_centers          # (B,T,K,2)
        if self.grid_mode:
            L = self.grid_period
            d = d - L * torch.round(d / L)                 # wrap to [-L/2, L/2)
        d2 = d.pow(2).sum(-1)
        p = torch.exp(-d2 / (2.0 * self.place_sigma * self.place_sigma))
        return p / (p.sum(-1, keepdim=True) + 1e-9)

    def decode_position(self, p: torch.Tensor) -> torch.Tensor:
        """Decode (x,y) from the code p (...,K). Place: linear population
        vector Σ_k p_k c_k. Grid: per-axis circular decode on the torus."""
        if not self.grid_mode:
            return p @ self.place_centers
        L = self.grid_period
        ang = (2.0 * math.pi / L) * self.place_centers     # (K,2) centre phases
        mc = p @ torch.cos(ang)                            # (...,2)
        ms = p @ torch.sin(ang)
        return torch.atan2(ms, mc) * (L / (2.0 * math.pi))  # → (-L/2, L/2]

    def decode_position_anchored(self, p: torch.Tensor,
                                 pos0: torch.Tensor) -> torch.Tensor:
        """Anchored path-integration readout: the trial start ``pos0`` is given
        (the PI anchor), so absolute position = pos0 + the network's decoded
        *relative* displacement (decode(t) − decode(t=0)). This pins t=0 to the
        true start and removes the constant per-trial offset that the raw place
        decode carries (the random Net2 tracks displacement well but never
        anchors absolute position). ``p`` (...,T,K), ``pos0`` (...,2)."""
        xy = self.decode_position(p)                       # (...,T,2)
        d = xy - xy[..., :1, :]                            # relative displacement
        if self.grid_mode:
            L = self.grid_period
            d = d - L * torch.round(d / L)
        return pos0.unsqueeze(-2) + d

    def _pos_sqerr(self, xy_dec, xy_true):
        """Squared position error (...): Euclidean (place) or toroidal (grid)."""
        d = xy_dec - xy_true
        if self.grid_mode:
            L = self.grid_period
            d = d - L * torch.round(d / L)
        return d.pow(2).sum(-1)

    def place_prob(self, place: torch.Tensor) -> torch.Tensor:
        """Non-negative, normalised population code from the tanh field output
        (...,K), for population-vector position decoding."""
        p = F.relu(place)
        return p / (p.sum(-1, keepdim=True) + 1e-9)

    def place_per_frame_loss(self, y_hat, y_true,
                             coeff_place: float = 1.0, coeff_pos: float = 1.0,
                             coeff_consistency: float = 0.0,
                             warmup: int = 10):
        """Per-frame place loss (B,T) + a [0,1] place score + position RMSE.

        ``y_hat`` = [cosθ, sinθ, d, place(K)] with place = the tanh place-cell
        field ∈(-1,1); ``y_true`` = [cosθ, sinθ, d, x, y]. Loss = heading+
        distance MSE + coeff_place·MSE(place, g) where g_k = Gaussian place
        field at (x,y) (the cells are trained to BE their fields). Position is
        read out as a population vector over the positive activations (used for
        the score, the consistency term and the reported RMSE). Score = mean
        cosine similarity between the (rectified) field and the target g (∈[0,1]).
        """
        K = self.net2_n_place
        head_mse = (y_hat[..., :3] - y_true[..., :3]).pow(2).mean(-1)   # (B,T)
        place = y_hat[..., 3:3 + K]                        # tanh field (B,T,K)
        xy_true = y_true[..., 3:5]
        g = self.place_field(xy_true)                      # raw Gaussian (B,T,K)
        field_mse = (place - g).pow(2).mean(-1)            # (B,T)
        pf = head_mse + coeff_place * field_mse
        # population-vector position readout (rectified field → distribution),
        # anchored to the given start so absolute position is pinned at t=0.
        p = self.place_prob(place)                         # (B,T,K)
        if self.place_anchor:
            xy_dec = self.decode_position_anchored(p, y_true[:, 0, 3:5])  # (B,T,2)
        else:
            xy_dec = self.decode_position(p)               # (B,T,2)
        pos_se = self._pos_sqerr(xy_dec, xy_true)          # (B,T)
        if coeff_consistency > 0.0:
            # Net1↔Net2 integrator consistency: per step the magnitude of
            # Net2's decoded position change must equal Net1's decoded
            # forward-distance change (both = |v_fwd|·dt). On the torus the
            # position delta is wrapped before taking its magnitude.
            d_hat = y_hat[..., 2]
            dxy = xy_dec[:, 1:] - xy_dec[:, :-1]                  # (B,T-1,2)
            if self.grid_mode:
                L = self.grid_period
                dxy = dxy - L * torch.round(dxy / L)
            sp2 = dxy.norm(dim=-1)                                # (B,T-1)
            sp1 = (d_hat[:, 1:] - d_hat[:, :-1]).abs()            # (B,T-1)
            cons = F.pad((sp2 - sp1).pow(2), (1, 0))             # (B,T)
            pf = pf + coeff_consistency * cons
        with torch.no_grad():
            pp = F.relu(place[:, warmup:]); gg = g[:, warmup:]
            cos = (pp * gg).sum(-1) / (pp.norm(dim=-1) * gg.norm(dim=-1) + 1e-9)
            score = cos.mean()
            pos_rmse = pos_se[:, warmup:].mean().sqrt()
        return pf, score, pos_rmse
