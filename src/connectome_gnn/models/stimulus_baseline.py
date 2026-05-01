"""Stimulus-only baseline — predicts voltage from past stimulus frames alone.

No dependence on past voltage/activity. Serves as a lower-bound baseline:
any model that cannot beat this is not learning dynamics beyond what the
stimulus trajectory already encodes.

Architecture:
    v(t) = Predictor(flatten(StimEncoder(stim[t-tw+1 : t+1])))
    stimulus_encoder: MLP(n_input_neurons -> stim_latent_dims)  [per-frame]
    predictor:        MLP(tw * stim_latent_dims -> n_neurons)
"""

import torch
import torch.nn as nn

from connectome_gnn.models.registry import register_model


def _build_mlp(input_dim, output_dim, hidden_dim, n_layers, device=None):
    """Build a vanilla MLP: Linear+ReLU hidden layers, linear output."""
    layers = [nn.Linear(input_dim, hidden_dim, device=device), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim, device=device), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, output_dim, device=device))
    return nn.Sequential(*layers)


@register_model(
    "flyvis_stimulus",
    "e8_flywireRF_stimulus", "e8_flywireRF_proximal_nulls_stimulus",
)
class StimulusBaseline(nn.Module):
    """Stimulus-only baseline: v(t) = Predictor(flatten(StimEncoder(stim[t-tw:t]))).

    Input:  stimulus context window (B, tw, n_input_neurons)
    Output: predicted voltage       (B, n_neurons)
    """

    def __init__(self, aggr_type='add', config=None, device=None):
        super().__init__()

        sim = config.simulation
        mc = config.graph_model
        tc = config.training

        self.device = device
        self.n_neurons = sim.n_neurons
        self.n_input_neurons = sim.n_input_neurons
        self.tw = tc.time_window
        self.stim_latent_dims = mc.stim_latent_dims

        # Stimulus encoder: per-frame MLP
        self.stimulus_encoder = _build_mlp(
            input_dim=self.n_input_neurons,
            output_dim=mc.stim_latent_dims,
            hidden_dim=mc.hidden_dim_stim_encoder,
            n_layers=mc.n_layers_stim_encoder,
            device=device,
        )

        # Predictor: flattened encoded context -> voltage
        self.predictor = _build_mlp(
            input_dim=self.tw * mc.stim_latent_dims,
            output_dim=self.n_neurons,
            hidden_dim=mc.hidden_dim,
            n_layers=mc.n_layers,
            device=device,
        )

        # Dummy W for compatibility with regularizer/trainer
        self.W = nn.Parameter(torch.zeros(1, 1, device=device), requires_grad=False)

    def predict_voltage(self, stim_context):
        """Predict voltage from a stimulus context window.

        Args:
            stim_context: (B, tw, n_input_neurons)
        Returns:
            v_pred: (B, n_neurons)
        """
        B, tw, stim_dim = stim_context.shape
        # Encode each frame independently
        z = self.stimulus_encoder(stim_context.reshape(B * tw, stim_dim))  # (B*tw, stim_latent_dims)
        # Flatten all encoded frames
        z = z.reshape(B, tw * self.stim_latent_dims)
        return self.predictor(z)
