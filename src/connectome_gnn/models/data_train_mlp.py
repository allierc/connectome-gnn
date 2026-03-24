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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from connectome_gnn.log import get_logger
from connectome_gnn.models.training_utils import build_model, load_flyvis_data, determine_load_fields
from connectome_gnn.utils import create_log_dir

_logger = get_logger(__name__)


# --- Rollout evaluation ---

def rollout_mse(model, voltage, stimulus, dt, n_windows=10, window_len=1000):
    """Compute mean rollout MSE over evenly-spaced windows.

    Divides the data into n_windows of window_len frames each.
    For each window, starts from the true initial state and rolls out
    autoregressively, feeding in the true stimulus at each step.

    Also computes the constant model baseline: x_{t+1} = x_0 (predict
    initial state forever). This grows as the true trajectory drifts.

    Returns:
        model_mse: (window_len,) array of model MSE averaged over windows
        constant_mse: (window_len,) array of constant-model MSE averaged over windows
    """
    n_frames = voltage.shape[0]
    total_needed = n_windows * window_len
    if total_needed > n_frames:
        window_len = n_frames // n_windows
    starts = torch.linspace(0, n_frames - window_len, n_windows).long()

    model_mse_sum = torch.zeros(window_len, device=voltage.device)
    const_mse_sum = torch.zeros(window_len, device=voltage.device)

    for s in starts:
        v = voltage[s]  # (N,) — initial state
        v0 = v          # constant model prediction
        for t in range(window_len):
            frame_idx = s + t
            stim_t = stimulus[frame_idx]  # (n_input_neurons,)

            mlp_input = torch.cat([v, stim_t], dim=0).unsqueeze(0)  # (1, N + n_input)
            dvdt = model._mlp_forward(mlp_input).squeeze(0)         # (N,)
            v = v + dt * dvdt

            target = voltage[frame_idx + 1]
            model_mse_sum[t] += F.mse_loss(v, target).item()
            const_mse_sum[t] += F.mse_loss(v0, target).item()

    return (
        (model_mse_sum / n_windows).cpu().numpy(),
        (const_mse_sum / n_windows).cpu().numpy(),
    )


def plot_rollout_mse(model_mse, constant_mse, epoch, log_dir, dt_value):
    """Save rollout MSE vs time step plot."""
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(len(model_mse))
    time_axis = steps * dt_value
    ax.plot(time_axis, model_mse, linewidth=1, label='MLP')
    ax.plot(time_axis, constant_mse, color='gray', linestyle='--', linewidth=0.8, label='constant (x=x₀)')
    ax.legend()
    ax.set_xlabel('rollout time')
    ax.set_ylabel('MSE')
    ax.set_title(f'Validation rollout MSE — epoch {epoch+1}')
    ax.set_yscale('log')
    fig.tight_layout()

    plot_dir = os.path.join(log_dir, 'tmp_training')
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, f'rollout_mse_epoch_{epoch+1:03d}.png'), dpi=100)
    plt.close(fig)


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

    # --- Data split ---
    all_voltage = x_ts.voltage                          # (T, N)
    all_stimulus = x_ts.stimulus[:, :n_input_neurons]   # (T, n_input_neurons)

    train_start = tc.train_start
    train_end = tc.train_end if tc.train_end > 0 else n_frames

    voltage = all_voltage[train_start:train_end]          # (T_train, N)
    stimulus = all_stimulus[train_start:train_end]        # (T_train, n_input_neurons)
    n_train_frames = voltage.shape[0]
    _logger.info(f'train split: frames [{train_start}, {train_end}) = {n_train_frames} frames')

    has_val = tc.val_start > 0 or tc.val_end > 0
    if has_val:
        val_voltage = all_voltage[tc.val_start:tc.val_end]
        val_stimulus = all_stimulus[tc.val_start:tc.val_end]
        n_val_frames = val_voltage.shape[0]
        _logger.info(f'val split: frames [{tc.val_start}, {tc.val_end}) = {n_val_frames} frames')

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

    # Valid frame range: need t and t+1, so max index is n_train_frames - 2
    max_frame = n_train_frames - 2
    batches_per_epoch = int(max_frame * data_passes_per_epoch / batch_size)

    _logger.info(f'batch_size: {batch_size}, data_passes_per_epoch: {data_passes_per_epoch}')
    _logger.info(f'batches_per_epoch: {batches_per_epoch}, n_epochs: {n_epochs}')

    net_path = os.path.join(log_dir, 'models')
    os.makedirs(net_path, exist_ok=True)

    dt = torch.tensor(sim.delta_t, device=device)

    # Constant model baseline: MSE(x_t, x_{t+1}) — predicting no change
    with torch.no_grad():
        constant_model_loss = F.mse_loss(voltage[:-1], voltage[1:]).item()
        if has_val:
            constant_val_loss = F.mse_loss(val_voltage[:-1], val_voltage[1:]).item()
    _logger.info(f'constant model baseline MSE: {constant_model_loss:.4e}')
    if has_val:
        _logger.info(f'constant model baseline val MSE: {constant_val_loss:.4e}')

    # Select compiled vs uncompiled loss function
    compute_loss = _compute_loss_compiled if device.type == "cuda" else _compute_loss

    # --- Training loop ---
    model.train()
    training_start = time.time()
    best_val_loss = float('inf')

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

        # --- Validation ---
        val_str = ''
        val_loss = None
        if has_val:
            model.eval()
            with torch.no_grad():
                # 1-step validation loss
                val_max = n_val_frames - 2
                val_indices = torch.randint(0, val_max + 1, (batch_size,), device=device)
                val_loss = _compute_loss(
                    model,
                    val_voltage[val_indices],
                    val_stimulus[val_indices],
                    val_voltage[val_indices + 1],
                    dt,
                ).item()

                # Rollout evaluation on validation data
                model_rollout_mse, const_rollout_mse = rollout_mse(
                    model, val_voltage, val_stimulus, dt,
                )
            model.train()

            val_str = f' | val: {val_loss:.4e}'
            plot_rollout_mse(model_rollout_mse, const_rollout_mse, epoch, log_dir, sim.delta_t)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(net_path, 'best_model.pt'))
                _logger.info(f'  saved best model (val_loss={best_val_loss:.4e})')
        else:
            # No validation — save best by train loss
            if mean_loss < best_val_loss:
                best_val_loss = mean_loss
                torch.save(model.state_dict(), os.path.join(net_path, 'best_model.pt'))
                _logger.info(f'  saved best model (train_loss={best_val_loss:.4e})')

        _logger.info(
            f'epoch {epoch+1}/{n_epochs} | '
            f'train: {mean_loss:.4e}{val_str} | '
            f'duration: {epoch_duration:.1f}s (total: {total_elapsed:.1f}s)'
        )

        # Save periodic checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': mean_loss,
            'val_loss': val_loss if has_val else mean_loss,
        }, os.path.join(net_path, 'latest_checkpoint.pt'))

    total_time = time.time() - training_start
    _logger.info(f'training complete: {n_epochs} epochs in {total_time:.1f}s, best loss: {best_val_loss:.4e}')
    _logger.info(f'constant model baseline: {constant_model_loss:.4e}')
