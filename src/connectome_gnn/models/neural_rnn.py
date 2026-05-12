"""RNN baseline model — GRU-based recurrent network for neural dynamics prediction.

A global (non-graph-structured) RNN that takes all neuron voltages as input.
This baseline shows the advantage of the GNN's explicit graph structure.

Architecture:
    h_t = GRU(h_{t-1}, [v_t, I_t])
    dv/dt = Linear(h_t)

The model does NOT use edge topology — it processes the full neuron state
as a flat vector. Per-edge W is learned separately for connectivity comparison.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from connectome_gnn.models.registry import register_model


@register_model(
    "flyvis_rnn",
    "drosophila_cx_rnn", "larva_rnn", "zebrafish_oculomotor_rnn",
)
class NeuralRNN(nn.Module):
    """GRU-based RNN baseline for neural dynamics prediction.

    Forward interface matches the sequential training path in graph_trainer.py:
        pred = model(x, h=None, return_all=False)
        pred, h = model(x, h=h, return_all=True)

    Also exposes self.W and self.a for compatibility with plotting/analysis code.
    """

    def __init__(self, aggr_type='add', config=None, device=None):
        super().__init__()

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.model = model_config.signal_model_name
        self.calcium_type = simulation_config.calcium_type
        self.observable = train_config.observable
        self.n_neurons = simulation_config.n_neurons
        self.n_input_neurons = simulation_config.n_input_neurons
        self.n_edges = simulation_config.n_edges
        self.n_extra_null_edges = simulation_config.n_extra_null_edges
        self.batch_size = train_config.batch_size
        self.update_type = model_config.update_type
        self.embedding_dim = model_config.embedding_dim

        hidden_dim = model_config.hidden_dim
        n_gru_layers = model_config.n_layers

        # Input: all neuron voltages + stimulus for input neurons
        # x tensor layout: (n_neurons, features) where features include voltage and stimulus
        # We flatten voltage (n_neurons) + stimulus (n_input_neurons) into one vector
        gru_input_size = self.n_neurons + self.n_input_neurons

        # GRU layers
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
        ).to(device)

        # Output: predict dv/dt for all neurons
        self.fc = nn.Linear(hidden_dim, self.n_neurons).to(device)

        # Per-edge W for connectivity comparison (not used in forward pass)
        # The RNN learns connectivity implicitly in GRU weights;
        # this explicit W is compared against ground truth for benchmarking.
        n_w = self.n_edges + self.n_extra_null_edges
        w_init_mode = getattr(train_config, 'w_init_mode', 'zeros')
        if w_init_mode == 'zeros':
            W_init = torch.zeros(n_w, device=device, dtype=torch.float32)
        elif w_init_mode == 'randn_scaled':
            w_init_scale = getattr(train_config, 'w_init_scale', 1.0)
            W_init = torch.randn(n_w, device=device, dtype=torch.float32) * (w_init_scale / math.sqrt(n_w))
        elif w_init_mode == 'uniform_scaled':
            w_init_scale = getattr(train_config, 'w_init_scale', 1.0)
            bound = w_init_scale / math.sqrt(n_w)
            W_init = (torch.rand(n_w, device=device, dtype=torch.float32) * 2 - 1) * bound
        else:
            W_init = torch.randn(n_w, device=device, dtype=torch.float32)
        self.W = nn.Parameter(W_init[:, None], requires_grad=True)

        # Embedding for compatibility with clustering/plotting code
        self.a = nn.Parameter(
            torch.randn(self.n_neurons, self.embedding_dim,
                        device=device, dtype=torch.float32),
            requires_grad=True,
        )

    def forward(self, x, h=None, c=None, data_id=[], k=[], return_all=False, **kwargs):
        """Forward pass: predict dv/dt from current neuron state.

        Args:
            x: State tensor (n_neurons, features). Features layout depends on
               calcium_type — voltage is at column 3 (no calcium) or 7 (with calcium),
               stimulus at column 4.
            h: GRU hidden state (n_layers, 1, hidden_dim) or None.
            return_all: If True, return (prediction, hidden_state).

        Returns:
            dv_dt: (n_neurons, 1) predicted derivative.
            h: Updated hidden state (only if return_all=True).
        """
        # Extract observable column from packed state tensor.
        # Layout: [..., voltage(3), stimulus(4), ..., calcium(7), ...].
        if self.observable == "calcium":
            v = x[:, 7:8]
        else:
            v = x[:, 3:4]

        stimulus = x[:self.n_input_neurons, 4:5]

        # Flatten: (n_neurons + n_input_neurons,)
        inp = torch.cat([v.flatten(), stimulus.flatten()])

        # Reshape for GRU: (batch=1, seq=1, features)
        inp = inp.unsqueeze(0).unsqueeze(0)

        # GRU forward
        out, h_next = self.gru(inp, h)

        # Predict dv/dt for all neurons
        dv_dt = self.fc(out.squeeze(0)).view(-1, 1)

        if return_all:
            return dv_dt, h_next
        return dv_dt
