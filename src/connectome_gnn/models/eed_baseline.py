"""EED baseline — encode-evolve-decode with MLPWithSkips sub-networks.

Reproduces the EED architecture from the reference checkpoint
(checkpoint_1step_20251211_d5fdaa4/19d5ee) with hardcoded hyperparameters.
All sub-networks use MLPWithSkips (skip connections from input to each layer).
"""

import torch
import torch.nn as nn

from connectome_gnn.models.registry import register_model
from connectome_gnn.neuron_state import NeuronState


class MLPWithSkips(nn.Module):
    """MLP where each hidden layer receives a direct linear projection of the input.

    The input projection (without activation) is concatenated with the previous
    layer's output before the linear + activation step.  The output layer also
    gets a skip projection.
    """

    def __init__(self, input_dim, output_dim, hidden_units, num_hidden_layers, activation="ReLU"):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers

        if num_hidden_layers == 0:
            self.output_layer = nn.Linear(input_dim, output_dim)
            return

        # Linear projections from input to each hidden layer (no activation)
        self.input_projections = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.input_projections.append(nn.Linear(input_dim, hidden_units))

        # Main hidden layers
        self.linear_layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(num_hidden_layers):
            if i == 0:
                layer_input_dim = input_dim + hidden_units
            else:
                layer_input_dim = hidden_units + hidden_units
            self.linear_layers.append(nn.Linear(layer_input_dim, hidden_units))
            self.activations.append(getattr(nn, activation)())

        # Output layer also gets concatenated input projection
        self.input_projection_final = nn.Linear(input_dim, hidden_units)
        self.output_layer = nn.Linear(hidden_units + hidden_units, output_dim)

    def forward(self, x):
        if self.num_hidden_layers == 0:
            return self.output_layer(x)

        original_input = x
        hidden = x

        for i in range(self.num_hidden_layers):
            input_proj = self.input_projections[i](original_input)
            concat = torch.cat([hidden, input_proj], dim=-1)
            hidden = self.linear_layers[i](concat)
            hidden = self.activations[i](hidden)

        input_proj_final = self.input_projection_final(original_input)
        concat_final = torch.cat([hidden, input_proj_final], dim=-1)
        return self.output_layer(concat_final)


@register_model("flyvis_eed")
class EEDBaseline(nn.Module):
    """Encode-Evolve-Decode baseline with hardcoded reference hyperparameters.

    Architecture (all MLPWithSkips):
        encoder:          n_neurons → 256,  1 hidden of 256, ReLU
        decoder:          256 → n_neurons,  1 hidden of 256, ReLU
        stimulus_encoder: n_input_neurons → 64, 3 hidden of 64, ReLU
        evolver:          256+64=320 → 256,  1 hidden of 256, ReLU
                          zero-init final layer, residual connection

    Returns dv/dt so the existing Euler-step trainer works unchanged.
    """

    LATENT_DIMS = 256
    HIDDEN_UNITS = 256
    STIM_LATENT = 64
    STIM_HIDDEN_LAYERS = 3

    def __init__(self, aggr_type='add', config=None, device=None):
        super().__init__()

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.model = model_config.signal_model_name
        self.n_neurons = simulation_config.n_neurons
        self.n_input_neurons = simulation_config.n_input_neurons
        self.n_edges = simulation_config.n_edges
        self.n_extra_null_edges = simulation_config.n_extra_null_edges
        self.batch_size = train_config.batch_size
        self.update_type = model_config.update_type
        self.calcium_type = simulation_config.calcium_type
        self.dt = simulation_config.delta_t

        # Sub-networks
        self.encoder = MLPWithSkips(
            self.n_neurons, self.LATENT_DIMS,
            self.HIDDEN_UNITS, num_hidden_layers=1,
        )
        self.decoder = MLPWithSkips(
            self.LATENT_DIMS, self.n_neurons,
            self.HIDDEN_UNITS, num_hidden_layers=1,
        )
        self.stimulus_encoder = MLPWithSkips(
            self.n_input_neurons, self.STIM_LATENT,
            self.STIM_LATENT, num_hidden_layers=self.STIM_HIDDEN_LAYERS,
        )
        self.evolver = MLPWithSkips(
            self.LATENT_DIMS + self.STIM_LATENT, self.LATENT_DIMS,
            self.HIDDEN_UNITS, num_hidden_layers=1,
        )

        # Zero-init evolver's output layer so residual starts as identity
        nn.init.zeros_(self.evolver.output_layer.weight)
        nn.init.zeros_(self.evolver.output_layer.bias)

        # Move to device
        if device is not None:
            self.to(device)

        # Dummy W parameter for compatibility with regularizer/trainer
        n_w = self.n_edges + self.n_extra_null_edges
        self.W = nn.Parameter(torch.zeros(max(n_w, 1), 1, device=device), requires_grad=False)

    def _mlp_forward(self, x):
        """Forward pass: x is (B, n_neurons + n_input_neurons), returns dv/dt (B, n_neurons)."""
        v = x[:, :self.n_neurons]
        stim = x[:, self.n_neurons:]

        z = self.encoder(v)
        s = self.stimulus_encoder(stim)
        z_next = z + self.evolver(torch.cat([z, s], dim=1))
        x_next = self.decoder(z_next)

        return (x_next - v) / self.dt

    def predict_dvdt(self, v, stim):
        """Model-agnostic interface: (v, stim) → dvdt.

        Args:
            v: (B, n_neurons) or (n_neurons,) voltage tensor
            stim: (B, n_input_neurons) or (n_input_neurons,) stimulus tensor
        Returns:
            dvdt with same batch shape as v
        """
        mlp_input = torch.cat([v, stim], dim=-1)
        if mlp_input.dim() == 1:
            mlp_input = mlp_input.unsqueeze(0)
            return self._mlp_forward(mlp_input).squeeze(0)
        return self._mlp_forward(mlp_input)

    def forward(self, state: NeuronState, edge_index: torch.Tensor = None,
                data_id=[], k=[], return_all=False, **kwargs):
        """Compute dv/dt from neuron state. Same interface as MLPBaseline."""
        self.data_id = data_id.squeeze().long().clone().detach() if hasattr(data_id, 'squeeze') else data_id

        v = state.observable(self.calcium_type)    # (N, 1)
        stim = state.stimulus.unsqueeze(-1)        # (N, 1)

        n_total = v.shape[0]
        batch_size = n_total // self.n_neurons
        v_batched = v.view(batch_size, self.n_neurons)
        stim_batched = stim.view(batch_size, self.n_neurons)[:, :self.n_input_neurons]

        mlp_input = torch.cat([v_batched, stim_batched], dim=1)
        mlp_output = self._mlp_forward(mlp_input)

        pred = mlp_output.view(n_total, 1)

        if return_all:
            return pred, None, None
        return pred
