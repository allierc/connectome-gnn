"""Training loop for the stimulus-only baseline model.

Predicts voltage at time t from tw frames of stimulus ending at t (inclusive):
stim[t-tw+1 : t+1]. No dependence on past voltage/activity — each prediction
is independent.
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


def _gather_stim_context(stimulus_unfolded, t_indices, tw):
    """Gather stimulus context windows for a batch of time indices.

    For target v(t), returns stim[t-tw+1 : t+1] (tw frames ending at t inclusive).

    Args:
        stimulus_unfolded: (T - tw + 1, tw, n_input) from stimulus.unfold(0, tw, 1).transpose(1,2)
        t_indices: (B,) time indices (each >= tw - 1)
        tw: time window size (scalar tensor on same device)

    Returns:
        stim_context: (B, tw, n_input)
    """
    return stimulus_unfolded[t_indices - tw + 1]


def _compute_loss(model, voltage, stimulus_unfolded, t_indices, tw):
    """Compute MSE loss for a batch of time indices."""
    stim_ctx = _gather_stim_context(stimulus_unfolded, t_indices, tw)
    pred = model.predict_voltage(stim_ctx)
    target = voltage[t_indices]
    return F.mse_loss(pred, target)



def data_train_stimulus(config, erase, best_model, device, log_file=None):
    """Train the stimulus-only baseline model.

    Args:
        config: NeuralGraphConfig
        erase: if True, overwrite existing log directory
        best_model: checkpoint identifier (or None)
        device: torch device
        log_file: optional open file handle for logging
    """
    sim = config.simulation
    tc = config.training
    tw_int = tc.time_window

    if isinstance(device, str):
        device = torch.device(device)

    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    log_dir, logger = create_log_dir(config, erase)

    load_fields = determine_load_fields(config)
    x_ts, _y_ts, type_list = load_flyvis_data(
        config.dataset, split='train', fields=load_fields,
    )
    x_ts = x_ts.to(device)

    n_neurons = x_ts.n_neurons
    n_frames = x_ts.n_frames
    n_input_neurons = sim.n_input_neurons
    config.simulation.n_neurons = n_neurons
    sim.n_frames = n_frames
    _logger.info(f'dataset: {n_frames} frames, {n_neurons} neurons, {n_input_neurons} input neurons')
    _logger.info(f'stimulus time window: tw={tw_int}')

    all_voltage = x_ts.voltage                          # (T, N)
    all_stimulus = x_ts.stimulus[:, :n_input_neurons]   # (T, n_input_neurons)

    train_start = tc.train_start
    train_end = tc.train_end if tc.train_end > 0 else n_frames

    voltage = all_voltage[train_start:train_end]
    stimulus = all_stimulus[train_start:train_end]
    n_train_frames = voltage.shape[0]
    _logger.info(f'train split: frames [{train_start}, {train_end}) = {n_train_frames} frames')

    assert tw_int > 0, f'time_window must be > 0 for stimulus baseline, got {tw_int}'
    assert tw_int < n_train_frames, f'time_window ({tw_int}) >= n_train_frames ({n_train_frames})'

    # Scalar tensor on device — avoids CPU-GPU sync in torch.compile'd code
    tw = torch.tensor(tw_int, device=device)

    # Pre-compute unfolded stimulus view for efficient context gathering
    # stimulus.unfold(0, tw_int, 1) -> (T - tw_int + 1, n_input, tw_int), transpose to (T - tw_int + 1, tw_int, n_input)
    stimulus_unfolded = stimulus.unfold(0, tw_int, 1).transpose(1, 2).contiguous()

    checkpoint_path = None
    if tc.pretrained_model != '':
        checkpoint_path = tc.pretrained_model
    model, start_epoch = build_model(config, device, checkpoint_path=checkpoint_path)
    assert hasattr(model, 'predict_voltage'), (
        f"{type(model).__name__} must implement predict_voltage(stim_context) "
        "to be used with data_train_stimulus"
    )

    # Compile per-call so CUDA Graph pools can be freed between CV folds
    _compute_loss_compiled = torch.compile(
        _compute_loss, fullgraph=True, mode="reduce-overhead"
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'total parameters: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=tc.lr)

    batch_size = tc.batch_size
    data_passes_per_epoch = tc.data_augmentation_loop
    n_epochs = tc.n_epochs

    # Valid frame range: (tw_int - 1) <= t < n_train_frames
    # With context window stim[t-tw+1 : t+1], need t-tw+1 >= 0 i.e. t >= tw-1
    max_frame = n_train_frames - 1
    min_frame = tw_int - 1
    batches_per_epoch = int((n_train_frames - min_frame) * data_passes_per_epoch / batch_size)

    _logger.info(f'batch_size: {batch_size}, data_passes_per_epoch: {data_passes_per_epoch}')
    _logger.info(f'batches_per_epoch: {batches_per_epoch}, n_epochs: {n_epochs}')

    net_path = os.path.join(log_dir, 'models')
    os.makedirs(net_path, exist_ok=True)

    # Constant model baseline (computed on CPU to avoid large GPU intermediate)
    with torch.no_grad():
        v_cpu = voltage.cpu()
        constant_model_rmse = float(np.sqrt(F.mse_loss(v_cpu[:-1], v_cpu[1:]).item()))
        del v_cpu
    _logger.info(f'constant model baseline RMSE: {constant_model_rmse:.4e}')

    # --- Training loop ---
    model.train()
    training_start = time.time()
    best_epoch = 0

    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_loss = torch.zeros((), device=device)
        n_batches = 0

        pbar = trange(batches_per_epoch, ncols=120, desc=f'epoch {epoch+1}/{n_epochs}')
        for _ in pbar:
            t_indices = torch.randint(min_frame, max_frame + 1, (batch_size,), device=device)

            optimizer.zero_grad()
            loss = _compute_loss_compiled(
                model, voltage, stimulus_unfolded, t_indices, tw,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()
            n_batches += 1

            if n_batches % 100 == 0:
                pbar.set_postfix_str(f'loss={epoch_loss / n_batches:.4e}')

        mean_loss = epoch_loss.item() / max(n_batches, 1)
        epoch_duration = time.time() - epoch_start
        total_elapsed = time.time() - training_start

        # Save model
        best_epoch = epoch + 1
        torch.save(
            {'model_state_dict': model.state_dict()},
            os.path.join(net_path, f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt'),
        )

        _logger.info(
            f'epoch {epoch+1}/{n_epochs} | '
            f'train: {mean_loss:.4e} | '
            f'duration: {epoch_duration:.1f}s (total: {total_elapsed:.1f}s)'
        )

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': mean_loss,
        }, os.path.join(net_path, 'latest_checkpoint.pt'))

    total_time = time.time() - training_start
    _logger.info(f'training complete: {n_epochs=} in {total_time=:.1f}s')
    _logger.info(f'constant model baseline RMSE: {constant_model_rmse:.4e}')

    if log_file:
        log_file.write('\n--- Training stimulus baseline results ---\n')
        log_file.write(f'train_best_epoch: {best_epoch}\n')
        log_file.write(f'train_constant_baseline_rmse: {constant_model_rmse:.4e}\n')
