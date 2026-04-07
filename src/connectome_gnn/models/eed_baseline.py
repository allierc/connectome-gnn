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
    """Encode-Evolve-Decode model with configurable architecture.

    All sub-networks use MLPWithSkips. Evolver has residual connection
    and zero-initialized final layer. Returns dv/dt so the existing
    Euler-step test/plot pipeline works unchanged.

    Architecture is configured via graph_model fields — see EED mapping
    comment in config.py GraphModelConfig.
    """

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

        # Read EED architecture from config
        latent_dims = model_config.latent_dim
        n_layers_enc_dec = model_config.n_layers_encoder
        n_layers_evolver = model_config.n_layers_evolver
        stim_latent_dims = model_config.stim_latent_dims
        hidden_dim_stim = model_config.hidden_dim_stim_encoder
        n_layers_stim = model_config.n_layers_stim_encoder

        # Sub-networks (all use latent_dims as hidden dim, except stimulus encoder)
        self.encoder = MLPWithSkips(
            self.n_neurons, latent_dims,
            latent_dims, num_hidden_layers=n_layers_enc_dec,
        )
        self.decoder = MLPWithSkips(
            latent_dims, self.n_neurons,
            latent_dims, num_hidden_layers=n_layers_enc_dec,
        )
        self.stimulus_encoder = MLPWithSkips(
            self.n_input_neurons, stim_latent_dims,
            hidden_dim_stim, num_hidden_layers=n_layers_stim,
        )
        self.evolver = MLPWithSkips(
            latent_dims + stim_latent_dims, latent_dims,
            latent_dims, num_hidden_layers=n_layers_evolver,
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

    def predict_dvdt(self, v, stim):
        """Model-agnostic interface: (v, stim) → dvdt.

        Args:
            v: (B, n_neurons) or (n_neurons,) voltage tensor
            stim: (B, n_input_neurons) or (n_input_neurons,) stimulus tensor
        Returns:
            dvdt with same batch shape as v
        """
        squeeze = v.dim() == 1
        if squeeze:
            v = v.unsqueeze(0)
            stim = stim.unsqueeze(0)
        z = self.encoder(v)
        s = self.stimulus_encoder(stim)
        z_next = z + self.evolver(torch.cat([z, s], dim=1))
        x_pred = self.decoder(z_next)
        dvdt = (x_pred - v) / self.dt
        return dvdt.squeeze(0) if squeeze else dvdt

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

        pred = self.predict_dvdt(v_batched, stim_batched).view(n_total, 1)

        if return_all:
            return pred, None, None
        return pred
