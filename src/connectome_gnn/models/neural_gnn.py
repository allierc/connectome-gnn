import math

import numpy as np
import torch
import torch.nn as nn

from connectome_gnn.models.MLP import MLP
from connectome_gnn.models.registry import register_model
from connectome_gnn.models.Siren_Network import Siren
from connectome_gnn.models.MultiResGrid_Network import (
    MultiResTemporalGrid,
    MultiResSpatioTemporalGrid,
)
from connectome_gnn.neuron_state import NeuronState


@register_model(
    "flyvis_A",
    "flyvis_A_tanh",
    "flyvis_A_multiple_ReLU",
    "flyvis_A_NULL",
    "flyvis_B",
    "flyvis_C",
    "flyvis_D",
    "flyvis_hybrid",
    "flyvis_hybrid_zeroedge",
    "e8_flywireRF",
    "e8_flywireRF_proximal_nulls",
    "e8_flywireRF_random_nulls",
    "full_eye_flywireRF",
    "full_eye_flywireRF_proximal_nulls",
    "full_eye_flywireRF_random_nulls",
    "drosophila_cx",
    "drosophila_cx_voltage",  # voltage-recovery flow (FlyVisODEParams schema on disk)
    "larva",
    "zebrafish",
    "zebrafish_oculomotor",
    "cortex",
)
class NeuralGNN(nn.Module):
    """GNN for neural signal dynamics with per-edge W.

    Equations:
        msg_j = W[edge] * g_phi(v_j, a_j)^2   (g_phi_positive=True)
        msg_j = W[edge] * g_phi(v_i, v_j, a_i, a_j)^2   (variant B)
        du/dt = f_theta(v, a, sum(msg), excitation)

    Uses explicit scatter_add for message passing (no PyG dependency).
    """

    # Connectivity-recovery family (single source of truth for the trainer /
    # tester / plot_synaptic dispatch). 'gnn': W/tau/V_rest live inside the
    # learned f_theta/g_phi MLPs and must be extracted (slope fit + g_phi
    # correction). See models.recovery / metrics.recovered_connectivity.
    MODEL_FAMILY = "gnn"

    PARAMS_DOC = {
        "model_name": "NeuralGNN",
        "description": "GNN for neural signal dynamics with per-edge W: "
        "du/dt = f_theta(v, a, sum(msg), excitation), msg_j = W[edge] * g_phi(v_j, a_j)",
        "key_differences_from_SignalPropagation": {
            "W_shape": "1D per-edge vector W[n_edges + n_extra_null_edges, 1] instead of dense N×N matrix",
            "visual_input": "Supports visual field input (DAVIS/calcium) via excitation channel",
            "calcium": "Can use calcium concentration instead of voltage as observable",
            "g_phi_positive": "When True, g_phi output is squared to enforce positive edge messages",
        },
        "equations": {
            "message_flyvis_A": "msg_j = W[edge_idx] * g_phi(v_j, a_j)^2   (g_phi_positive=True)",
            "message_flyvis_B": "msg_j = W[edge_idx] * g_phi(v_i, v_j, a_i, a_j)^2",
            "update": "du/dt = f_theta(v, a, sum(msg), excitation)",
        },
        "graph_model_config": {
            "description": "Parameters in the graph_model: section of the YAML config.",
            "g_phi": {
                "description": "Edge message function — computes per-edge features, multiplied by W[edge]",
                "input_size": {
                    "flyvis_A": "input_size = 1 + embedding_dim  (v_j, a_j)",
                    "flyvis_B": "input_size = 2 + 2*embedding_dim  (v_i, v_j, a_i, a_j)",
                },
                "output_size": "1 (scalar edge message)",
                "hidden_dim": {"description": "Hidden layer width", "typical_range": [32, 128], "default": 64},
                "n_layers": {"description": "Number of MLP layers", "typical_range": [2, 5], "default": 3},
            },
            "f_theta": {
                "description": "Node update function — computes du/dt from voltage + embedding + messages + excitation",
                "input_size_update": "1 + embedding_dim + output_size + 1  (v, a, msg, excitation)",
                "output_size": "1 (du/dt scalar)",
                "hidden_dim_update": {"description": "Hidden layer width", "typical_range": [32, 128], "default": 64},
                "n_layers_update": {"description": "Number of MLP layers", "typical_range": [2, 5], "default": 3},
            },
            "embedding": {
                "embedding_dim": {
                    "description": "Dimension of learnable node embedding a_i",
                    "typical_range": [1, 8],
                    "default": 2,
                },
            },
            "g_phi_positive": {
                "description": "If True, square g_phi output to enforce positive messages",
                "default": True,
            },
            "field_type": {
                "description": "Visual field type — determines visual input reconstruction model (e.g. 'visual_NNR')"
            },
            "MLP_activation": {"description": "Activation function for MLPs", "default": "tanh"},
        },
        "simulation_params": {
            "description": "Parameters in the simulation: section of the YAML config",
            "n_neurons": {"description": "Total number of neurons in the connectome"},
            "n_input_neurons": {"description": "Number of input (photoreceptor) neurons"},
            "n_neuron_types": {"description": "Number of neuron cell types"},
            "n_edges": {"description": "Number of synaptic connections in the connectome"},
            "n_extra_null_edges": {"description": "Additional null edges for capacity (default 0)"},
            "n_frames": {"description": "Number of simulation time frames"},
            "visual_input_type": {"description": "Type of visual stimulus (e.g. 'DAVIS')"},
            "noise_model_level": {"description": "Noise level added to observations", "typical_range": [0.0, 0.1]},
            "calcium_type": {
                "description": "If not 'none', use calcium concentration instead of voltage as observable"
            },
        },
        "training_params": {
            "description": "Parameters in the training: section that affect model architecture or loss",
            "tunable": [
                {
                    "name": "lr_W",
                    "description": "Learning rate for per-edge connectivity W",
                    "typical_range": [1e-4, 5e-2],
                },
                {"name": "lr", "description": "Learning rate for MLPs", "typical_range": [1e-4, 5e-3]},
                {
                    "name": "lr_embedding",
                    "description": "Learning rate for embeddings a",
                    "typical_range": [1e-4, 5e-3],
                },
                {"name": "coeff_W_L1", "description": "L1 sparsity penalty on W", "typical_range": [1e-6, 1e-3]},
                {
                    "name": "coeff_g_phi_diff",
                    "description": "Regularizer: g_phi output variance penalty",
                    "typical_range": [0, 500],
                },
                {
                    "name": "coeff_g_phi_norm",
                    "description": "Regularizer: edge weight norm penalty",
                    "typical_range": [0, 10],
                },
                {
                    "name": "coeff_g_phi_weight_L1",
                    "description": "L1 penalty on g_phi weights",
                    "typical_range": [0, 10],
                },
                {
                    "name": "coeff_f_theta_weight_L1",
                    "description": "L1 penalty on f_theta weights",
                    "typical_range": [0, 10],
                },
                {
                    "name": "coeff_f_theta_weight_L2",
                    "description": "L2 penalty on f_theta weights",
                    "typical_range": [0, 0.01],
                },
                {"name": "batch_size", "description": "Number of time frames per batch", "typical_range": [1, 4]},
                {"name": "data_augmentation_loop", "description": "Number of augmentation iterations per epoch", "typical_range": [10, 50]},
                {"name": "w_init_mode", "description": "W initialization: 'zeros' (default), 'randn', 'randn_scaled', or 'uniform_scaled'"},
                {"name": "w_init_scale", "description": "Scale factor for randn_scaled/uniform_scaled init (bound = scale/sqrt(n_edges))", "default": 1.0},
            ],
        },
    }

    def __init__(self, aggr_type="add", config=None, device=None):
        super().__init__()

        simulation_config = config.simulation
        model_config = config.graph_model

        self.device = device
        self.aggr_type = aggr_type
        self.model = model_config.signal_model_name
        self.dimension = simulation_config.dimension
        self.embedding_dim = model_config.embedding_dim
        self.n_neurons = simulation_config.n_neurons
        self.n_input_neurons = simulation_config.n_input_neurons
        self.n_dataset = config.training.n_runs
        self.n_frames = simulation_config.n_frames
        self.field_type = model_config.field_type
        self.embedding_trial = config.training.embedding_trial
        self.multi_connectivity = config.training.multi_connectivity
        self.calcium_type = simulation_config.calcium_type
        self.MLP_activation = config.graph_model.MLP_activation

        self.training_time_window = config.training.time_window

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers

        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.input_size_update = model_config.input_size_update

        self.n_edges = simulation_config.n_edges
        self.n_extra_null_edges = simulation_config.n_extra_null_edges
        self.g_phi_positive = model_config.g_phi_positive

        self.batch_size = config.training.batch_size
        self.update_type = model_config.update_type

        self.g_phi = MLP(
            input_size=self.input_size,
            output_size=self.output_size,
            nlayers=self.n_layers,
            hidden_size=self.hidden_dim,
            activation=self.MLP_activation,
            device=self.device,
        )

        self.f_theta = MLP(
            input_size=self.input_size_update,
            output_size=self.output_size,
            nlayers=self.n_layers_update,
            hidden_size=self.hidden_dim_update,
            activation=self.MLP_activation,
            device=self.device,
        )

        self.a = nn.Parameter(
            torch.tensor(
                np.ones((int(self.n_neurons), self.embedding_dim)),
                device=self.device,
                requires_grad=True,
                dtype=torch.float32,
            )
        )

        train_config = config.training
        n_w = self.n_edges + self.n_extra_null_edges
        w_init_mode = getattr(train_config, "w_init_mode", "zeros")
        if w_init_mode == "zeros":
            W_init = torch.zeros(n_w, device=self.device, dtype=torch.float32)
        elif w_init_mode == "randn_scaled":
            w_init_scale = getattr(train_config, "w_init_scale", 1.0)
            W_init = torch.randn(n_w, device=self.device, dtype=torch.float32) * (w_init_scale / math.sqrt(n_w))
        elif w_init_mode == 'uniform_scaled':
            w_init_scale = getattr(train_config, 'w_init_scale', 1.0)
            bound = w_init_scale / math.sqrt(n_w)
            W_init = (torch.rand(n_w, device=self.device, dtype=torch.float32) * 2 - 1) * bound
        else:  # 'randn'
            W_init = torch.randn(n_w, device=self.device, dtype=torch.float32)
        self.W = nn.Parameter(W_init[:, None], requires_grad=True)

        # Branch-0 hard Eq-10 sign-lock: when enabled, the effective per-edge
        # weight in the message is |W|·sign_GT (the GT sign is registered by the
        # trainer from ode_params.W via set_edge_sign_from_weights). Only the
        # magnitudes are learned. Default off keeps every existing neural_gnn
        # config byte-equivalent (the free-sign learned W path).
        self.lock_edge_signs_from_connectome = bool(
            getattr(model_config, "lock_edge_signs_from_connectome", False)
        )
        # registered as a buffer slot (None) so the trainer can assign the GT
        # sign tensor later without a re-registration clash.
        self.register_buffer("_edge_sign", None, persistent=False)

        if "visual" in model_config.field_type:
            if "instantNGP" in model_config.field_type:
                # to be implemented
                pass
            else:
                print("use NNR for visual field reconstruction")
                self.NNR_f = Siren(
                    in_features=model_config.input_size_nnr_f,
                    out_features=model_config.output_size_nnr_f,
                    hidden_features=model_config.hidden_dim_nnr_f,
                    hidden_layers=model_config.n_layers_nnr_f,
                    first_omega_0=model_config.omega_f,
                    hidden_omega_0=model_config.omega_f,
                    outermost_linear=model_config.outermost_linear_nnr_f,
                )
                self.NNR_f.to(self.device)

                # Match training normalization (graph_trainer.py divides by raw period).
                # Previous code divided by 2*pi here — revert if needed:
                self.NNR_f_xy_period = model_config.nnr_f_xy_period / (2 * np.pi)
                self.NNR_f_T_period = model_config.nnr_f_T_period / (2 * np.pi)
                # self.NNR_f_xy_period = model_config.nnr_f_xy_period
                # self.NNR_f_T_period = model_config.nnr_f_T_period

        # Hidden-neuron INR: learns voltages of silenced neurons jointly with GNN.
        # Built in __init__ (like NNR_f) using hidden_neuron_fraction from model_config.
        # "siren_t"   : SIREN(t) -> (n_hidden,)  — independent signal per neuron
        # "siren_txy" : SIREN(x,y,t) -> scalar   — spatially-correlated field
        # "ngp_t"     : MultiResTemporalGrid(t) -> (n_hidden,)  — local, no waterbed
        # "none"      : zero-silencing, no INR
        # When train_with_anchor_neurons is enabled, the NNR_hidden output is extended by
        # n_anchor extra slots used only for direct supervision against observed GT voltages.
        # The first n_hidden slots remain the "hidden" outputs; slots [n_hidden:n_hidden+n_anchor]
        # are the "anchor" outputs.
        self.NNR_hidden = None
        self._inr_hidden_type = getattr(model_config, 'inr_type_hidden', 'none')
        self.NNR_hidden_T_period = getattr(model_config, 'nnr_hidden_T_period', 64000.0) / (2 * np.pi)
        self.NNR_hidden_xy_period = getattr(model_config, 'nnr_f_xy_period', 1.0) / (2 * np.pi)
        self.NNR_hidden_n_frames = float(simulation_config.n_frames)  # for NGP [0,1] normalisation
        # Spatio-temporal NGP defaults — only become meaningful if the
        # ngp_t branch below is selected with ngp_hidden_spatial=True.
        self._ngp_spatial_enabled = False
        self._ngp_xy_period = 1.0
        self._ngp_pos_norm = None
        _hidden_frac = getattr(model_config, 'hidden_neuron_fraction', 0.0)
        _train_anchor = bool(getattr(config.training, 'train_with_anchor_neurons', False))
        _n_anchor_cfg = int(getattr(config.training, 'n_anchor', 0))
        self.n_hidden = 0
        self.n_anchor = 0
        if self._inr_hidden_type in ('siren_t', 'siren_txy') and _hidden_frac > 0.0:
            n_non_retina = simulation_config.n_neurons - simulation_config.n_input_neurons
            n_hidden = int(n_non_retina * _hidden_frac)
            self.n_hidden = n_hidden
            # Anchor only supported for siren_t (per-neuron output). siren_txy uses scalar field.
            if _train_anchor and self._inr_hidden_type == 'siren_t':
                self.n_anchor = _n_anchor_cfg if _n_anchor_cfg > 0 else n_hidden
            in_features = 1 if self._inr_hidden_type == 'siren_t' else 3
            out_features = (n_hidden + self.n_anchor) if self._inr_hidden_type == 'siren_t' else 1
            self.NNR_hidden = Siren(
                in_features=in_features,
                out_features=out_features,
                hidden_features=getattr(model_config, 'hidden_dim_nnr_hidden', 2048),
                hidden_layers=getattr(model_config, 'n_layers_nnr_hidden', 4),
                first_omega_0=getattr(model_config, 'omega_hidden', 4096.0),
                hidden_omega_0=getattr(model_config, 'omega_hidden', 4096.0),
                outermost_linear=getattr(model_config, 'outermost_linear_nnr_hidden', True),
            )
            self.NNR_hidden.to(self.device)
        elif self._inr_hidden_type == 'ngp_t' and _hidden_frac > 0.0:
            n_non_retina = simulation_config.n_neurons - simulation_config.n_input_neurons
            n_hidden = int(n_non_retina * _hidden_frac)
            self.n_hidden = n_hidden
            if _train_anchor:
                self.n_anchor = _n_anchor_cfg if _n_anchor_cfg > 0 else n_hidden
            self._ngp_spatial_enabled = bool(getattr(model_config, 'ngp_hidden_spatial', False))
            self._ngp_xy_period = float(getattr(model_config, 'ngp_hidden_xy_period', 1.0))
            self._ngp_pos_norm = None  # populated lazily from state.pos on first call
            if self._ngp_spatial_enabled:
                # Per-neuron query: scalar output per (t, pos, a) triple.
                # Output slot role (hidden vs anchor) is encoded by which
                # neuron ids we query, not by an output index — so n_output=1.
                # The decoder also consumes the GNN's learned embedding
                # self.a[ids] (a_dim=embedding_dim) so the same per-neuron
                # latent that drives g_phi / f_theta is available to the
                # NGP non-linearly. Cell type itself is NOT fed in: only
                # the learned a_i, which the GNN co-trains via g_phi for
                # hidden neurons through visible postsynaptic neighbours.
                self.NNR_hidden = MultiResSpatioTemporalGrid(
                    n_levels=getattr(model_config, 'ngp_hidden_n_levels', 16),
                    n_features_per_level=getattr(model_config, 'ngp_hidden_n_features_per_level', 4),
                    base_resolution=getattr(model_config, 'ngp_hidden_base_resolution', 16),
                    per_level_scale=getattr(model_config, 'ngp_hidden_per_level_scale', 1.4),
                    spatial_n_levels=getattr(model_config, 'ngp_hidden_spatial_n_levels', 6),
                    spatial_n_features_per_level=getattr(model_config, 'ngp_hidden_spatial_n_features_per_level', 4),
                    spatial_base_resolution=getattr(model_config, 'ngp_hidden_spatial_base_resolution', 4),
                    spatial_per_level_scale=getattr(model_config, 'ngp_hidden_spatial_per_level_scale', 1.5),
                    n_output=1,
                    mlp_width=getattr(model_config, 'ngp_hidden_mlp_width', 256),
                    mlp_layers=getattr(model_config, 'ngp_hidden_mlp_layers', 2),
                    a_dim=int(model_config.embedding_dim),
                )
            else:
                self.NNR_hidden = MultiResTemporalGrid(
                    n_levels=getattr(model_config, 'ngp_hidden_n_levels', 24),
                    n_features_per_level=getattr(model_config, 'ngp_hidden_n_features_per_level', 4),
                    base_resolution=getattr(model_config, 'ngp_hidden_base_resolution', 16),
                    per_level_scale=getattr(model_config, 'ngp_hidden_per_level_scale', 1.4),
                    n_output=n_hidden + self.n_anchor,
                    mlp_width=getattr(model_config, 'ngp_hidden_mlp_width', 512),
                    mlp_layers=getattr(model_config, 'ngp_hidden_mlp_layers', 4),
                )
            self.NNR_hidden.to(self.device)

        # Optional factorized output path. Parallel low-rank path that mixes a
        # per-neuron identity factor (model.a or a dedicated embedding) with the
        # NGP's pre-head time features, added to the shared decoder output.
        # Only ngp_t supports this currently (requires return_features from the
        # temporal grid). rank=0 disables.
        self.ngp_factorized_rank = int(getattr(model_config, 'ngp_factorized_rank', 0))
        self._ngp_factorized = (
            self.NNR_hidden is not None
            and self._inr_hidden_type == 'ngp_t'
            and self.ngp_factorized_rank > 0
        )
        self.ngp_time_proj = None
        self.ngp_emb_proj = None
        self.ngp_emb = None
        self._ngp_use_a = False
        if self._ngp_factorized:
            _rank = self.ngp_factorized_rank
            _mlp_w = int(getattr(model_config, 'ngp_hidden_mlp_width', 512))
            self.ngp_time_proj = nn.Linear(_mlp_w, _rank, bias=False).to(self.device)
            if bool(getattr(model_config, 'ngp_factorized_from_a', True)):
                self.ngp_emb_proj = nn.Linear(
                    int(getattr(model_config, 'embedding_dim', 2)),
                    _rank, bias=False,
                ).to(self.device)
                self._ngp_use_a = True
            else:
                self.ngp_emb = nn.Parameter(
                    torch.randn(int(simulation_config.n_neurons), _rank,
                                device=self.device) * 0.01
                )
                self._ngp_use_a = False

    def _ngp_emb_lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """(N,) neuron indices → (N, rank) factorized embedding."""
        if self._ngp_use_a:
            return self.ngp_emb_proj(self.a[ids])
        return self.ngp_emb[ids]

    def _ngp_cache_pos(self, state: NeuronState):
        """Populate self._ngp_pos_norm from state.pos on first call."""
        if self._ngp_pos_norm is None:
            self._ngp_pos_norm = (
                state.pos[:, :self.dimension] / self._ngp_xy_period
            ).detach().to(self.device)

    def _ngp_query_spatial(self, k_tensor: torch.Tensor,
                           ids: torch.Tensor) -> torch.Tensor:
        """Spatial+temporal NGP query at B frames × M neurons.

        k_tensor: (B,) integer frame indices.
        ids:      (M,) neuron ids whose positions index into self._ngp_pos_norm.
        Returns:  (B, M) tensor of predicted voltages.

        Requires self._ngp_pos_norm to have been populated by a prior
        forward_hidden(state, ...) call.
        """
        if self._ngp_pos_norm is None:
            raise RuntimeError(
                "forward_hidden(state, ...) must be called once before batched "
                "spatial NGP calls so the model can cache neuron positions."
            )
        B = int(k_tensor.shape[0])
        M = int(ids.shape[0])
        t_norm = (k_tensor.to(device=self.device, dtype=torch.float32)
                  / self.NNR_hidden_n_frames)                       # (B,)
        pos_m = self._ngp_pos_norm[ids.to(self._ngp_pos_norm.device)]  # (M, 2)
        t_flat = t_norm.repeat_interleave(M).unsqueeze(1)            # (B*M, 1)
        pos_flat = pos_m.repeat(B, 1)                                # (B*M, 2)
        # Per-query GNN embedding self.a[ids] — the same learned latent
        # that g_phi/f_theta consume. Tiled across frames so each (k, id)
        # query carries its identity factor non-linearly into the decoder.
        a_m = self.a[ids.to(self.a.device)]                          # (M, embedding_dim)
        a_flat = a_m.repeat(B, 1)                                    # (B*M, embedding_dim)
        if self._ngp_factorized:
            out, feat = self.NNR_hidden(t_flat, pos_flat, a=a_flat,
                                         return_features=True)
            out = out.squeeze(-1)                                    # (B*M,)
            time_lr = self.ngp_time_proj(feat)                       # (B*M, rank)
            emb_per_neuron = self._ngp_emb_lookup(ids)               # (M, rank)
            emb_flat = emb_per_neuron.repeat(B, 1)                   # (B*M, rank)
            out = out + (emb_flat * time_lr).sum(dim=-1)
        else:
            out = self.NNR_hidden(t_flat, pos_flat, a=a_flat).squeeze(-1)  # (B*M,)
        return out.reshape(B, M)

    def forward_hidden(self, state: NeuronState, k: int, hidden_ids: torch.Tensor) -> torch.Tensor:
        """Predict voltages for hidden neurons at frame k via the hidden SIREN.

        Returns:
            (n_hidden,) tensor — can be assigned directly to x.voltage[hidden_ids].
            Gradients flow back into NNR_hidden so it is trained jointly with the GNN.
        """
        if self.NNR_hidden is None:
            raise RuntimeError("forward_hidden called but NNR_hidden is not initialised")

        if self._inr_hidden_type == 'siren_txy':
            # One SIREN query per hidden neuron: input (x_i, y_i, t)
            pos_h = state.pos[hidden_ids, :self.dimension]                      # (n_hidden, 2)
            t_vec = torch.full((len(hidden_ids), 1), float(k) / self.NNR_hidden_T_period,
                               device=self.device, dtype=torch.float32)
            in_feats = torch.cat([pos_h / self.NNR_hidden_xy_period, t_vec], dim=1)  # (n_hidden, 3)
            return self.NNR_hidden(in_feats).squeeze(-1)                        # (n_hidden,)
        elif self._inr_hidden_type == 'ngp_t':
            if self._ngp_spatial_enabled:
                self._ngp_cache_pos(state)
                k_t = torch.tensor([int(k)], device=self.device, dtype=torch.long)
                return self._ngp_query_spatial(k_t, hidden_ids).squeeze(0)      # (n_hidden,)
            # MultiResTemporalGrid: t normalized to [0, 1]
            t_in = torch.full((1, 1), float(k) / self.NNR_hidden_n_frames,
                              device=self.device, dtype=torch.float32)          # (1, 1)
            if self._ngp_factorized:
                shared, feat = self.NNR_hidden(t_in, return_features=True)      # (1, n_tot), (1, mlp_w)
                out = shared.squeeze(0)[:self.n_hidden]                         # (n_hidden,)
                time_lr = self.ngp_time_proj(feat).squeeze(0)                   # (rank,)
                emb_lr = self._ngp_emb_lookup(hidden_ids)                       # (n_hidden, rank)
                return out + (emb_lr * time_lr).sum(dim=-1)                     # (n_hidden,)
            out = self.NNR_hidden(t_in).squeeze(0)                              # (n_hidden [+ n_anchor],)
            if self.n_anchor > 0:
                out = out[:self.n_hidden]
            return out
        else:  # siren_t
            t_in = torch.full((1, 1), float(k) / self.NNR_hidden_T_period,
                              device=self.device, dtype=torch.float32)          # (1, 1)
            out = self.NNR_hidden(t_in).squeeze(0)                              # (n_hidden [+ n_anchor],)
            if self.n_anchor > 0:
                out = out[:self.n_hidden]
            return out

    def forward_anchor(self, k: int, anchor_ids: torch.Tensor = None) -> torch.Tensor:
        """Predict voltages for anchor neurons at frame k — only the anchor output slots.

        Returns (n_anchor,) tensor. Used exclusively for direct voltage supervision;
        anchor predictions are NOT injected into x.voltage (the GNN already sees the
        observed voltage of anchor neurons through the normal visible path).

        When the factorized head is enabled (ngp_factorized_rank > 0), `anchor_ids`
        must be provided so the per-neuron identity factor can be looked up.
        """
        if self.NNR_hidden is None or self.n_anchor == 0:
            raise RuntimeError("forward_anchor called but anchor outputs are not enabled")
        if self._inr_hidden_type == 'ngp_t':
            if self._ngp_spatial_enabled:
                if anchor_ids is None:
                    raise RuntimeError("forward_anchor requires anchor_ids when ngp_hidden_spatial=True")
                k_t = torch.tensor([int(k)], device=self.device, dtype=torch.long)
                return self._ngp_query_spatial(k_t, anchor_ids).squeeze(0)       # (n_anchor,)
            t_in = torch.full((1, 1), float(k) / self.NNR_hidden_n_frames,
                              device=self.device, dtype=torch.float32)
            if self._ngp_factorized:
                if anchor_ids is None:
                    raise RuntimeError("forward_anchor requires anchor_ids when ngp_factorized_rank > 0")
                shared, feat = self.NNR_hidden(t_in, return_features=True)
                out = shared.squeeze(0)[self.n_hidden:]                          # (n_anchor,)
                time_lr = self.ngp_time_proj(feat).squeeze(0)                    # (rank,)
                emb_lr = self._ngp_emb_lookup(anchor_ids)                        # (n_anchor, rank)
                return out + (emb_lr * time_lr).sum(dim=-1)
            return self.NNR_hidden(t_in).squeeze(0)[self.n_hidden:]
        elif self._inr_hidden_type == 'siren_t':
            t_in = torch.full((1, 1), float(k) / self.NNR_hidden_T_period,
                              device=self.device, dtype=torch.float32)
            return self.NNR_hidden(t_in).squeeze(0)[self.n_hidden:]
        else:
            raise RuntimeError(f"anchor outputs not supported for inr_type_hidden={self._inr_hidden_type}")

    def forward_anchor_batched(self, k_tensor: torch.Tensor,
                                 anchor_ids: torch.Tensor = None) -> torch.Tensor:
        """Batched forward_anchor: predict anchor voltages at B frame indices in one call.

        k_tensor:  (B,) integer tensor of frame indices.
        anchor_ids: (n_anchor,) neuron ids; required when ngp_factorized_rank > 0.
        Returns: (B, n_anchor) tensor.
        """
        if self.NNR_hidden is None or self.n_anchor == 0:
            raise RuntimeError("forward_anchor_batched called but anchor outputs are not enabled")
        if self._inr_hidden_type == 'ngp_t' and self._ngp_spatial_enabled:
            if anchor_ids is None:
                raise RuntimeError("forward_anchor_batched requires anchor_ids when ngp_hidden_spatial=True")
            return self._ngp_query_spatial(k_tensor, anchor_ids)               # (B, n_anchor)
        if self._inr_hidden_type == 'ngp_t':
            denom = self.NNR_hidden_n_frames
        elif self._inr_hidden_type == 'siren_t':
            denom = self.NNR_hidden_T_period
        else:
            raise RuntimeError(f"anchor outputs not supported for inr_type_hidden={self._inr_hidden_type}")
        t_in = (k_tensor.to(device=self.device, dtype=torch.float32) / denom).unsqueeze(-1)  # (B, 1)
        if self._ngp_factorized:
            if anchor_ids is None:
                raise RuntimeError("forward_anchor_batched requires anchor_ids when ngp_factorized_rank > 0")
            shared, feat = self.NNR_hidden(t_in, return_features=True)
            out = shared[:, self.n_hidden:]                                    # (B, n_anchor)
            time_lr = self.ngp_time_proj(feat)                                 # (B, rank)
            emb_lr = self._ngp_emb_lookup(anchor_ids)                          # (n_anchor, rank)
            return out + time_lr @ emb_lr.T                                    # (B, n_anchor)
        return self.NNR_hidden(t_in)[:, self.n_hidden:]

    def forward_hidden_batched(self, k_tensor: torch.Tensor,
                                 hidden_ids: torch.Tensor = None) -> torch.Tensor:
        """Batched forward_hidden for ngp_t / siren_t: predict hidden voltages at B frame indices.

        k_tensor:   (B,) integer tensor of frame indices.
        hidden_ids: (n_hidden,) neuron ids; required when ngp_factorized_rank > 0.
        Returns: (B, n_hidden) tensor. Only supported for time-only INRs (ngp_t, siren_t)
        where the output is a single vector per time step, independent of neuron position.
        """
        if self.NNR_hidden is None:
            raise RuntimeError("forward_hidden_batched called but NNR_hidden is not initialised")
        if self._inr_hidden_type == 'ngp_t' and self._ngp_spatial_enabled:
            if hidden_ids is None:
                raise RuntimeError("forward_hidden_batched requires hidden_ids when ngp_hidden_spatial=True")
            return self._ngp_query_spatial(k_tensor, hidden_ids)               # (B, n_hidden)
        if self._inr_hidden_type == 'ngp_t':
            denom = self.NNR_hidden_n_frames
        elif self._inr_hidden_type == 'siren_t':
            denom = self.NNR_hidden_T_period
        else:
            raise RuntimeError(f"forward_hidden_batched not supported for inr_type_hidden={self._inr_hidden_type}")
        t_in = (k_tensor.to(device=self.device, dtype=torch.float32) / denom).unsqueeze(-1)  # (B, 1)
        if self._ngp_factorized:
            if hidden_ids is None:
                raise RuntimeError("forward_hidden_batched requires hidden_ids when ngp_factorized_rank > 0")
            shared, feat = self.NNR_hidden(t_in, return_features=True)
            out = shared[:, :self.n_hidden]                                    # (B, n_hidden)
            time_lr = self.ngp_time_proj(feat)                                 # (B, rank)
            emb_lr = self._ngp_emb_lookup(hidden_ids)                          # (n_hidden, rank)
            return out + time_lr @ emb_lr.T                                    # (B, n_hidden)
        return self.NNR_hidden(t_in)[:, :self.n_hidden]

    def forward_visual(self, state: NeuronState, k):
        """Reconstruct visual field from neuron positions and time step k."""
        if "instantNGP" in self.field_type:
            # to be implemented
            pass
        else:
            kk = torch.full((state.n_neurons, 1), float(k), device=self.device, dtype=torch.float32)
            in_features = torch.cat(
                (state.pos[:, : self.dimension] / self.NNR_f_xy_period, kk / self.NNR_f_T_period), dim=1
            )
            reconstructed_field = self.NNR_f(in_features[: self.n_input_neurons]) ** 2

        return reconstructed_field

    def _compute_messages(self, v, embedding, edge_index):
        """Compute per-edge messages and aggregate via scatter_add.

        args:
            v: (N, 1) observable (voltage or calcium)
            embedding: (N, embedding_dim) node embeddings
            edge_index: (2, E) source/destination indices

        returns:
            msg: (N, 1) aggregated messages per node
        """
        src, dst = edge_index

        # compute edge-to-W indices (supports batched edge_index)
        n_edges_batch = edge_index.shape[1]
        edge_W_idx = torch.arange(n_edges_batch, device=self.device) % (self.n_edges + self.n_extra_null_edges)

        # build per-edge features (ensure 2D even when embedding_dim=1)
        emb = embedding if embedding.dim() == 2 else embedding.unsqueeze(-1)
        if self.model == "flyvis_B":
            in_features = torch.cat([v[dst], v[src], emb[dst], emb[src]], dim=1)
        else:
            in_features = torch.cat([v[src], emb[src]], dim=1)

        # edge function
        g_phi_out = self.g_phi(in_features)
        if self.g_phi_positive:
            g_phi_out = g_phi_out**2

        # weight by per-edge W (|W|·sign_GT when the hard sign-lock is on)
        edge_msg = self._effective_edge_weights(edge_W_idx) * g_phi_out  # (E, 1)

        # aggregate: scatter_add messages to destination nodes
        msg = torch.zeros(v.shape[0], edge_msg.shape[1], device=self.device, dtype=v.dtype)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)

        return msg

    def _effective_edge_weights(self, edge_W_idx: torch.Tensor) -> torch.Tensor:
        """Per-edge weight used in messages.

        Hard Eq-10 sign-lock: ``|W|·sign_GT`` when
        ``lock_edge_signs_from_connectome`` is on and the GT sign buffer is set;
        otherwise the free-sign learned ``W`` (unchanged legacy behaviour).
        """
        W_e = self.W[edge_W_idx]
        if self.lock_edge_signs_from_connectome and self._edge_sign is not None:
            return W_e.abs() * self._edge_sign[edge_W_idx]
        return W_e

    @property
    def effective_W(self) -> torch.Tensor:
        """Full per-edge weight actually used in the dynamics — `|W|·sign_GT`
        under the hard sign-lock, else the raw `W`. Metrics and plots should
        read THIS (via get_model_W), not the raw parameter, whose sign is free
        because only its magnitude enters the message."""
        if self.lock_edge_signs_from_connectome and self._edge_sign is not None:
            return self.W.abs() * self._edge_sign
        return self.W

    def set_edge_sign_from_weights(self, gt_weights: torch.Tensor) -> None:
        """Register per-edge GT sign for the hard Eq-10 sign-lock (|W|·sign_GT).

        ``gt_weights``: (n_edges,) or (n_edges, 1) GT effective weights with the
        sign embedded (e.g. ``ode_params.W``), in the SAME edge order as the
        model's ``W`` (i.e. ``use_gt_edges=True`` so ``edges == ode_params
        .edge_index``). Any null edges get sign +1.
        """
        w = gt_weights.detach().reshape(-1).to(self.device)
        sign = torch.sign(w).to(torch.float32)
        n_w = self.n_edges + self.n_extra_null_edges
        if sign.numel() < n_w:
            pad = torch.ones(n_w - sign.numel(), device=self.device, dtype=torch.float32)
            sign = torch.cat([sign, pad], dim=0)
        elif sign.numel() > n_w:
            sign = sign[:n_w]
        self._edge_sign = sign[:, None]  # assign into the pre-registered buffer slot

    def forward(self, state: NeuronState, edge_index: torch.Tensor, data_id=[], k=[], return_all=False, **kwargs):
        """Forward pass: compute du/dt from neuron state and connectivity.

        args:
            state: NeuronState with voltage, stimulus, index fields
            edge_index: (2, E) tensor of (src, dst) edge indices
            data_id: dataset ID tensor
            k: time step (for visual field reconstruction)
            return_all: if True, return (pred, in_features, msg)

        returns:
            pred: (N, 1) predicted du/dt
        """
        self.data_id = data_id.squeeze().long().clone().detach()

        v = state.observable(self.calcium_type)
        # Sum optogenetic perturbation (when present) into the same excitation channel
        # as visual stimulus. Keeps f_theta input dim unchanged → existing checkpoints
        # remain loadable; opto contributes zero when state.optogenetics_stimulus is None.
        opto = state.optogenetics_stimulus if state.optogenetics_stimulus is not None else 0.0
        excitation = (state.stimulus + opto).unsqueeze(-1)
        particle_id = state.index.long()
        embedding = self.a[particle_id]
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(-1)

        msg = self._compute_messages(v, embedding, edge_index)

        in_features = torch.cat([v, embedding, msg, excitation], dim=1)
        pred = self.f_theta(in_features)

        if return_all:
            return pred, in_features, msg
        else:
            return pred
