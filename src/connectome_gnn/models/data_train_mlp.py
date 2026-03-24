"""MLP baseline trainer — EED-aligned training loop.

Trains the MLP baseline model with:
- dv/dt = MLP([v; stim]), integrated as x_{t+1} = x_t + dt * MLP(...)
- MSE loss on predicted dv/dt vs finite-difference target (x_{t+1} - x_t) / dt
- Random batch sampling of time frames
- Adam optimizer, TF32 precision

This mirrors the training loop in NeuralGraph/LatentEvolution/latent.py,
adapted for the flat MLP (no encoder/decoder, no latent space).
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from connectome_gnn.log import get_logger
from connectome_gnn.models.training_utils import build_model, load_flyvis_data, determine_load_fields
from connectome_gnn.utils import create_log_dir

_logger = get_logger(__name__)


# --- Compiled training step ---

def _compute_loss(model, x_t, stim_t, x_target, dt):
    """Compute MSE loss for one-step prediction.

    x_pred = x_t + dt * MLP([x_t; stim_t])
    loss = MSE(x_pred, x_{t+1})
    """
    mlp_input = torch.cat([x_t, stim_t], dim=1)  # (B, n_neurons + n_input_neurons)
    dvdt = model._mlp_forward(mlp_input)          # (B, n_neurons)
    x_pred = x_t + dt * dvdt
    return F.mse_loss(x_pred, x_target)


_compute_loss_compiled = torch.compile(
    _compute_loss, fullgraph=True, mode="reduce-overhead"
)


# --- Training loop ---

def data_train_mlp(config, erase, best_model, device, log_file=None):
    """Train MLP baseline with EED-aligned training loop.

    Args:
        config: NeuralGraphConfig
        erase: if True, overwrite existing log directory
        best_model: checkpoint identifier (or None)
        device: torch device
        log_file: optional open file handle for logging
    """
    sim = config.simulation
    tc = config.training

    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)

    # TF32 precision for faster matmuls on Ampere+ GPUs
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    log_dir, logger = create_log_dir(config, erase)

    # --- Load data ---
    load_fields = determine_load_fields(config)
    x_ts, _y_ts, type_list = load_flyvis_data(
        config.dataset, split='train', fields=load_fields, device=device,
    )

    n_neurons = x_ts.n_neurons
    n_frames = x_ts.n_frames
    n_input_neurons = sim.n_input_neurons
    config.simulation.n_neurons = n_neurons
    sim.n_frames = n_frames
    _logger.info(f'dataset: {n_frames} frames, {n_neurons} neurons, {n_input_neurons} input neurons')

    # Precompute voltage and stimulus tensors on device
    voltage = x_ts.voltage                        # (T, N)
    stimulus = x_ts.stimulus[:, :n_input_neurons]  # (T, n_input_neurons)

    # --- Build model ---
    checkpoint_path = None
    if tc.pretrained_model != '':
        checkpoint_path = tc.pretrained_model
    model, start_epoch = build_model(config, device, checkpoint_path=checkpoint_path)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'total parameters: {n_params:,}')

    # --- Optimizer (Adam, single LR — following EED) ---
    optimizer = torch.optim.Adam(model.parameters(), lr=tc.lr)

    # --- Training parameters ---
    batch_size = tc.batch_size
    data_passes_per_epoch = tc.data_augmentation_loop
    n_epochs = tc.n_epochs

    # Valid frame range: need t and t+1, so max index is n_frames - 2
    max_frame = n_frames - 2
    batches_per_epoch = int(max_frame * data_passes_per_epoch / batch_size)

    _logger.info(f'batch_size: {batch_size}, data_passes_per_epoch: {data_passes_per_epoch}')
    _logger.info(f'batches_per_epoch: {batches_per_epoch}, n_epochs: {n_epochs}')

    net_path = os.path.join(log_dir, 'models')
    os.makedirs(net_path, exist_ok=True)

    dt = torch.tensor(sim.delta_t, device=device)

    # Constant model baseline: MSE(x_t, x_{t+1}) — predicting no change
    with torch.no_grad():
        constant_model_loss = F.mse_loss(voltage[:-1], voltage[1:]).item()
    _logger.info(f'constant model baseline MSE: {constant_model_loss:.4e}')

    # Select compiled vs uncompiled loss function
    compute_loss = _compute_loss_compiled if device.type == "cuda" else _compute_loss

    # --- Training loop ---
    model.train()
    training_start = time.time()
    best_loss = float('inf')

    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        n_batches = 0

        pbar = trange(batches_per_epoch, ncols=120, desc=f'epoch {epoch+1}/{n_epochs}')
        for _ in pbar:
            t_indices = torch.randint(0, max_frame + 1, (batch_size,), device=device)

            x_t = voltage[t_indices]           # (B, N)
            stim_t = stimulus[t_indices]       # (B, n_input_neurons)
            x_target = voltage[t_indices + 1]  # (B, N)

            optimizer.zero_grad()
            loss = compute_loss(model, x_t, stim_t, x_target, dt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if n_batches % 100 == 0:
                pbar.set_postfix_str(f'loss={epoch_loss / n_batches:.4e}')

        mean_loss = epoch_loss / max(n_batches, 1)
        epoch_duration = time.time() - epoch_start
        total_elapsed = time.time() - training_start

        _logger.info(
            f'epoch {epoch+1}/{n_epochs} | '
            f'loss: {mean_loss:.4e} | '
            f'duration: {epoch_duration:.1f}s (total: {total_elapsed:.1f}s)'
        )

        # Save best model
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), os.path.join(net_path, 'best_model.pt'))
            _logger.info(f'  saved best model (loss={best_loss:.4e})')

        # Save periodic checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': mean_loss,
        }, os.path.join(net_path, 'latest_checkpoint.pt'))

    total_time = time.time() - training_start
    _logger.info(f'training complete: {n_epochs} epochs in {total_time:.1f}s, best loss: {best_loss:.4e}')
    _logger.info(f'constant model baseline: {constant_model_loss:.4e}')
