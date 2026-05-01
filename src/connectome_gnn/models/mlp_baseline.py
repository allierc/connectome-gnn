"""MLP baseline model — flat MLP mapping [v; stimulus] → dv/dt.

No graph structure, no latent space. Pure black-box baseline.
Stimulus is sliced to input neurons only (EED-aligned).
Effective connectivity is extracted post-hoc via Jacobian dF/dv.
"""

import torch
import torch.nn as nn

from connectome_gnn.models.registry import register_model
from connectome_gnn.neuron_state import NeuronState


@register_model(
    "drosophila_cx_mlp", "larva_mlp", "zebrafish_oculomotor_mlp",
    "flyvis_mlp",
    "e8_flywireRF_mlp", "e8_flywireRF_proximal_nulls_mlp",
)
class MLPBaseline(nn.Module):
    """Flat MLP baseline: dv/dt = MLP([v; stim]).

    Input:  concatenation of all neuron voltages and input-neuron stimuli
            (n_neurons + n_input_neurons,)
    Output: dv/dt for all neurons  (n_neurons,)

    No graph, no edges, no per-edge W. Connectivity is extracted
    post-hoc via the Jacobian dF/dv.
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

        input_size = self.n_neurons + self.n_input_neurons  # [v; stim_input_neurons]
        output_size = self.n_neurons
        hidden_dim = model_config.hidden_dim
        n_layers = model_config.n_layers

        # Optional: linear skip connection at each hidden layer
        self.add_skip_layers = getattr(model_config, 'add_skip_layers', False)
        if self.add_skip_layers:
            if n_layers < 2:
                raise NotImplementedError("Skip only works with 2 layers min")
            self.skip_layers = nn.ModuleList()
            self.skip_layers.append(nn.Linear(input_size, hidden_dim, device=device))
            for _ in range(n_layers - 2):
                self.skip_layers.append(nn.Linear(hidden_dim*2, hidden_dim, device=device))

        # x ->   Mx.    ]-|.     |--> M xx
        #.  |             |-> xx-|
        #.  |-> phi(Mx) ]-|      |--> phi(M xx)
        # MLP hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_dim, device=device))
        for _ in range(n_layers - 2):
            # with skip connections we have a concatenation
            hidden_dim2 = hidden_dim + int(self.add_skip_layers)*hidden_dim
            self.hidden_layers.append(nn.Linear(hidden_dim2, hidden_dim, device=device))
        _activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'soft_relu': nn.Softplus(),
        }
        self.activation = _activations[model_config.MLP_activation]

        # Output layer
        final_input_dim = hidden_dim
        if self.add_skip_layers and n_layers >= 2:
            final_input_dim = hidden_dim * 2
        self.final_layer = nn.Linear(final_input_dim, output_size, device=device)
        if model_config.zero_init_output:
            nn.init.zeros_(self.final_layer.weight)
            nn.init.zeros_(self.final_layer.bias)

        # Optional residual: linear projection from input to hidden dim,
        # skip from first hidden output to last hidden output
        self.add_residual = getattr(model_config, 'add_residual', False)
        if self.add_residual:
            self.residual_proj = nn.Linear(input_size, hidden_dim, device=device)

        # Optional learnable per-neuron diagonal term: dv_i/dt = alpha_i * v_i + MLP(v, stim)_i
        self.add_diagonal = getattr(model_config, 'add_diagonal', False)
        if self.add_diagonal:
            self.alpha = nn.Parameter(torch.zeros(self.n_neurons, device=device))

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
        mlp_input = torch.cat([v, stim], dim=-1)
        squeezed = mlp_input.dim() == 1
        if squeezed:
            mlp_input = mlp_input.unsqueeze(0)

        h = mlp_input
        if self.add_skip_layers and len(self.hidden_layers) >= 1:
            for layer, skip_layer in zip(self.hidden_layers, self.skip_layers):
                h1 = skip_layer(h)
                h2 = self.activation(layer(h))
                h = torch.cat([h1, h2], dim=-1)
        else:
            for i, layer in enumerate(self.hidden_layers):
                h = self.activation(layer(h))
                if self.add_residual and i == 0:
                    h = h + self.residual_proj(mlp_input)
                    h_first = h
            if self.add_residual and len(self.hidden_layers) > 1:
                h = h + h_first

        out = self.final_layer(h)
        if self.add_diagonal:
            out = out + self.alpha * v
        return out.squeeze(0) if squeezed else out

    def forward(self, state: NeuronState, edge_index: torch.Tensor = None,
                data_id=[], k=[], return_all=False, **kwargs):
        """Compute dv/dt from neuron state.

        Ignores edge_index — no graph structure used.
        Stimulus is sliced to input neurons only (EED-aligned).
        """
        self.data_id = data_id.squeeze().long().clone().detach() if hasattr(data_id, 'squeeze') else data_id

        v = state.observable(self.calcium_type)    # (N, 1)
        stim = state.stimulus.unsqueeze(-1)        # (N, 1)

        # Reshape for batch: group neurons by batch item
        n_total = v.shape[0]
        batch_size = n_total // self.n_neurons
        v_batched = v.view(batch_size, self.n_neurons)                            # (B, n_neurons)
        stim_batched = stim.view(batch_size, self.n_neurons)[:, :self.n_input_neurons]  # (B, n_input_neurons)

        pred = self.predict_dvdt(v_batched, stim_batched).view(n_total, 1)  # (N, 1)

        if return_all:
            return pred, None, None
        return pred

    def compute_jacobian(self, state: NeuronState):
        """Compute effective connectivity Jacobian dF/dv.

        Returns:
            J: (n_neurons, n_neurons) tensor where J[i,j] = d(dv_i/dt) / dv_j
        """
        v = state.observable(self.calcium_type).detach()    # (N, 1)
        stim = state.stimulus.detach()                      # (N,)

        # Single sample (no batch), slice stimulus to input neurons
        v_flat = v[:self.n_neurons].squeeze(-1)                       # (n_neurons,)
        stim_flat = stim[:self.n_input_neurons]                       # (n_input_neurons,)

        v_input = v_flat.clone().requires_grad_(True)
        mlp_output = self.predict_dvdt(v_input, stim_flat)      # (n_neurons,)

        J = torch.zeros(self.n_neurons, self.n_neurons, device=self.device)
        for i in range(self.n_neurons):
            if v_input.grad is not None:
                v_input.grad.zero_()
            mlp_output[i].backward(retain_graph=True)
            J[i] = v_input.grad.clone()

        return J

    def compute_jacobian_batched(self, x_ts, n_samples=100, seed=0):
        """Compute mean Jacobian over multiple frames for robust W estimation.

        Args:
            x_ts: NeuronTimeSeries with .frame(k) method
            n_samples: number of frames to average over
            seed: for reproducible frame selection

        Returns:
            J_mean: (n_neurons, n_neurons) mean Jacobian
        """
        n_frames = x_ts.voltage.shape[0]
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)
        frame_indices = torch.randint(0, n_frames, (n_samples,), generator=rng, device='cpu')

        J_sum = torch.zeros(self.n_neurons, self.n_neurons, device=self.device)
        for k in frame_indices:
            state = x_ts.frame(k.item())
            J_sum += self.compute_jacobian(state)

        return J_sum / n_samples
