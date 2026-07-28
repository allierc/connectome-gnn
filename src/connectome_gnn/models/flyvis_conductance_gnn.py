"""GNN for inferring the conductance-based flyvis twin.

In the current-based model the synaptic message depends only on the presynaptic
voltage, so ``FlyVisGNN`` (flyvis_A) feeds only ``v_j`` to the edge function. The
conductance-based twin instead delivers

    msg_ij = G_ij * relu(v_j) * (E_ij - v_i)

which depends on the *postsynaptic* voltage as well, and changes sign as ``v_i``
crosses the reversal potential. A message function of ``v_j`` alone cannot
represent it, and squaring the edge function — as flyvis_A/B do to keep messages
positive — forbids the sign change that inhibitory synapses need. This module
therefore provides two model types that take ``v_i`` as an input to ``g_phi``:

``flyvis_conductance``
    msg_ij = W[edge] * g_phi(v_i, v_j, a_i, a_j),  unconstrained in sign.
    The least assuming option: it can represent the driving force but is not
    told that one exists.

``flyvis_conductance_factorized``
    msg_ij = (W[edge] * g_phi(v_j, a_j))^2 * (e_psi(a_i, a_j) - v_i)
    Bakes the physics in: a non-negative conductance times an explicit driving
    force towards a learned per-edge reversal potential. Harder to fit but
    directly interpretable — ``e_psi`` recovers E and ``(W g_phi)^2`` recovers G.

Both share ``FlyVisGNN``'s node update, ``du/dt = f_theta(v, a, sum(msg), e)``,
and its per-edge W, so the connectivity readout used elsewhere in the repo
applies unchanged.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from connectome_gnn.models.MLP import MLP
from connectome_gnn.models.registry import register_model
from connectome_gnn.neuron_state import NeuronState


@register_model("flyvis_conductance", "flyvis_conductance_factorized")
class FlyVisConductanceGNN(nn.Module):
    """GNN whose edge function sees the postsynaptic voltage.

    Equations:
        flyvis_conductance:
            msg_j = W[edge] * g_phi(v_i, v_j, a_i, a_j)
        flyvis_conductance_factorized:
            msg_j = (W[edge] * g_phi(v_j, a_j))^2 * (e_psi(a_i, a_j) - v_i)
        both:
            du/dt = f_theta(v, a, sum(msg), excitation)

    Uses explicit scatter_add for message passing (no PyG dependency).
    """

    PARAMS_DOC = {
        "model_name": "FlyVisConductanceGNN",
        "description": "GNN for the conductance-based FlyVis twin: the edge function "
                       "takes the postsynaptic voltage v_i, so it can represent the "
                       "driving force (E - v_i) that scales every synaptic current.",
        "key_differences_from_FlyVisGNN": {
            "g_phi_inputs": "includes v_i (postsynaptic voltage), not only v_j",
            "sign": "messages are not squared; inhibitory synapses need the driving "
                    "force to be able to change sign",
            "factorized_variant": "splits the message into a non-negative conductance "
                                  "and an explicit (E - v_i) driving force, so the "
                                  "learned reversal potential is readable off e_psi",
        },
        "equations": {
            "message_flyvis_conductance": "msg_j = W[edge] * g_phi(v_i, v_j, a_i, a_j)",
            "message_flyvis_conductance_factorized":
                "msg_j = (W[edge] * g_phi(v_j, a_j))^2 * (e_psi(a_i, a_j) - v_i)",
            "update": "du/dt = f_theta(v, a, sum(msg), excitation)",
        },
        "graph_model_config": {
            "description": "Parameters in the graph_model: section of the YAML config.",
            "g_phi (MLP0)": {
                "description": "Edge message function",
                "input_size": {
                    "flyvis_conductance": "2 + 2*embedding_dim  (v_i, v_j, a_i, a_j)",
                    "flyvis_conductance_factorized": "1 + embedding_dim  (v_j, a_j)",
                },
                "output_size": "1",
            },
            "e_psi (MLP2, factorized only)": {
                "description": "Per-edge reversal potential from the two embeddings",
                "input_size": "2*embedding_dim  (a_i, a_j); override with "
                              "input_size_reversal",
                "hidden_dim": "hidden_dim_reversal, defaults to hidden_dim",
                "n_layers": "n_layers_reversal, defaults to n_layers",
            },
            "f_theta (MLP1)": {
                "description": "Node update; identical to FlyVisGNN",
                "input_size_update": "1 + embedding_dim + output_size + 1",
            },
            "embedding": {"embedding_dim": {"default": 2}},
            "g_phi_positive": {
                "description": "Ignored: squaring the message would forbid the sign "
                               "change of the driving force. The factorized variant "
                               "enforces positivity where it belongs, on the "
                               "conductance factor.",
            },
        },
        "training_params": {
            "description": "Same knobs as FlyVisGNN; W is the per-edge peak "
                           "conductance rather than a signed weight, so it is "
                           "expected to be one-signed after training.",
        },
    }

    def __init__(self, aggr_type='add', config=None, device=None):
        super().__init__()

        simulation_config = config.simulation
        model_config = config.graph_model

        self.device = device
        self.aggr_type = aggr_type
        self.model = model_config.signal_model_name
        self.factorized = 'factorized' in self.model
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

        self.batch_size = config.training.batch_size
        self.update_type = model_config.update_type

        expected = (
            1 + self.embedding_dim if self.factorized
            else 2 + 2 * self.embedding_dim
        )
        if self.input_size != expected:
            raise ValueError(
                f"{self.model} needs graph_model.input_size = {expected} "
                f"(got {self.input_size}); the edge function takes "
                + ("(v_j, a_j)" if self.factorized else "(v_i, v_j, a_i, a_j)")
            )

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

        # per-edge reversal potential, only for the factorized variant
        self.e_psi = None
        if self.factorized:
            self.e_psi = MLP(
                input_size=getattr(
                    model_config, 'input_size_reversal', 2 * self.embedding_dim
                ),
                output_size=1,
                nlayers=getattr(model_config, 'n_layers_reversal', self.n_layers),
                hidden_size=getattr(model_config, 'hidden_dim_reversal', self.hidden_dim),
                activation=self.MLP_activation,
                device=self.device,
            )

        self.a = nn.Parameter(
            torch.tensor(
                np.ones((int(self.n_neurons), self.embedding_dim)),
                device=self.device,
                requires_grad=True, dtype=torch.float32))

        train_config = config.training
        n_w = self.n_edges + self.n_extra_null_edges
        w_init_mode = getattr(train_config, 'w_init_mode', 'zeros')
        if w_init_mode == 'zeros':
            W_init = torch.zeros(n_w, device=self.device, dtype=torch.float32)
        elif w_init_mode == 'randn_scaled':
            w_init_scale = getattr(train_config, 'w_init_scale', 1.0)
            W_init = torch.randn(n_w, device=self.device, dtype=torch.float32) * (
                w_init_scale / math.sqrt(n_w))
        elif w_init_mode == 'uniform_scaled':
            w_init_scale = getattr(train_config, 'w_init_scale', 1.0)
            bound = w_init_scale / math.sqrt(n_w)
            W_init = (torch.rand(n_w, device=self.device, dtype=torch.float32) * 2 - 1) * bound
        else:  # 'randn'
            W_init = torch.randn(n_w, device=self.device, dtype=torch.float32)
        self.W = nn.Parameter(W_init[:, None], requires_grad=True)

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

        n_edges_batch = edge_index.shape[1]
        edge_W_idx = torch.arange(n_edges_batch, device=self.device) % (
            self.n_edges + self.n_extra_null_edges)

        # keep 2D even when embedding_dim == 1
        embedding = embedding if embedding.dim() == 2 else embedding.unsqueeze(-1)

        if self.factorized:
            # non-negative conductance from the presynaptic side ...
            conductance = (
                self.W[edge_W_idx] * self.g_phi(
                    torch.cat([v[src], embedding[src]], dim=1))
            ) ** 2
            # ... times the driving force towards a learned reversal potential
            reversal = self.e_psi(torch.cat([embedding[dst], embedding[src]], dim=1))
            edge_msg = conductance * (reversal - v[dst])
        else:
            in_features = torch.cat(
                [v[dst], v[src], embedding[dst], embedding[src]], dim=1)
            # deliberately unsquared: the driving force changes sign with v_i
            edge_msg = self.W[edge_W_idx] * self.g_phi(in_features)

        msg = torch.zeros(v.shape[0], edge_msg.shape[1], device=self.device, dtype=v.dtype)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)

        return msg

    def forward(self, state: NeuronState, edge_index: torch.Tensor, data_id=[], k=[],
                return_all=False, **kwargs):
        """Forward pass: compute du/dt from neuron state and connectivity.

        args:
            state: NeuronState with voltage, stimulus, index fields
            edge_index: (2, E) tensor of (src, dst) edge indices
            data_id: dataset ID tensor
            return_all: if True, return (pred, in_features, msg)

        returns:
            pred: (N, 1) predicted du/dt
        """
        if len(data_id):
            self.data_id = data_id.squeeze().long().clone().detach()

        v = state.observable(self.calcium_type)
        excitation = state.stimulus.unsqueeze(-1)
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

    def edge_conductance(self, v, embedding, edge_index):
        """Per-edge conductance and reversal potential learned by the factorized model.

        args:
            v: (N, 1) observable
            embedding: (N, embedding_dim) node embeddings
            edge_index: (2, E) source/destination indices

        returns:
            (conductance, reversal), each (E, 1)

        raises:
            NotImplementedError: for the unfactorized variant, where conductance
                and driving force are not separately identifiable.
        """
        if not self.factorized:
            raise NotImplementedError(
                "conductance and driving force are entangled in "
                f"{self.model}; use flyvis_conductance_factorized to read them off"
            )
        src, dst = edge_index
        edge_W_idx = torch.arange(edge_index.shape[1], device=self.device) % (
            self.n_edges + self.n_extra_null_edges)
        conductance = (
            self.W[edge_W_idx] * self.g_phi(torch.cat([v[src], embedding[src]], dim=1))
        ) ** 2
        reversal = self.e_psi(torch.cat([embedding[dst], embedding[src]], dim=1))
        return conductance, reversal
