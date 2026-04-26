"""Data-driven training loop for models with a predict_dvdt interface.

Trains any model implementing predict_dvdt(v, stim) → dvdt using:
- Euler integration: x_{t+1} = x_t + dt * predict_dvdt(x_t, stim_t)
- MSE loss against ground-truth next state
- Random batch sampling of time frames, optional multi-step rollout
- Adam optimizer, TF32 precision
"""

import os
import signal
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

_VAL_ROLLOUT_LEN = 8000
_VAL_BLOCK_SIZE = 64


def _rollout_block(model, v, stim_block, target_block, dt):
    """Roll out model for _VAL_BLOCK_SIZE steps.

    Args:
        v: (N,) current state
        stim_block: (_VAL_BLOCK_SIZE, n_input)
        target_block: (_VAL_BLOCK_SIZE, N) ground-truth next states

    Returns:
        v: (N,) state after the block
        out_mses: (_VAL_BLOCK_SIZE,) per-step MSE
    """
    out_mses = torch.empty(_VAL_BLOCK_SIZE, device=v.device)
    for t in range(_VAL_BLOCK_SIZE):
        dvdt = model.predict_dvdt(v, stim_block[t])
        v = v + dt * dvdt
        out_mses[t] = F.mse_loss(v, target_block[t])
    return v, out_mses


def _rollout_block_latent(model, z, stim_block, target_block):
    """Roll out EED for _VAL_BLOCK_SIZE steps in PURE LATENT SPACE.

    No re-encoding: evolves z autoregressively, decodes each step only
    to score against ground truth.

    Args:
        z: (1, latent_dim) current latent state
        stim_block: (_VAL_BLOCK_SIZE, n_input)
        target_block: (_VAL_BLOCK_SIZE, N) ground-truth next states

    Returns:
        z: (1, latent_dim) latent state after the block
        out_mses: (_VAL_BLOCK_SIZE,) per-step MSE
    """
    out_mses = torch.empty(_VAL_BLOCK_SIZE, device=z.device)
    for t in range(_VAL_BLOCK_SIZE):
        stim_z = model.stimulus_encoder(stim_block[t].unsqueeze(0))
        z = z + model.evolver(torch.cat([z, stim_z], dim=-1))
        v_pred = model.decoder(z).squeeze(0)
        out_mses[t] = F.mse_loss(v_pred, target_block[t])
    return z, out_mses


def val_rollout_latent(model, voltage, stimulus, val_start_idx, dt):
    """Single-start PURE-LATENT rollout validation on training data.

    Encodes the initial voltage once, then chains the evolver in latent
    space for _VAL_ROLLOUT_LEN steps. Decodes each step only for the MSE
    score (decoded prediction does NOT feed back into the encoder).

    `dt` is unused (kept for signature parity with `val_rollout`).

    Returns:
        mse_curve: (_VAL_ROLLOUT_LEN,) numpy array of per-step MSE
        div_time: first step index where MSE > 1, or _VAL_ROLLOUT_LEN if never
        rollout_rmse: RMSE = sqrt(mean MSE) over steps [0, div_time)
    """
    del dt  # unused; latent rollout does not Euler-integrate in activity
    _rollout_block_latent_compiled = torch.compile(_rollout_block_latent, mode="default")

    all_mse = torch.empty(_VAL_ROLLOUT_LEN, device=voltage.device)
    z = model.encoder(voltage[val_start_idx].unsqueeze(0))  # (1, latent_dim)

    n_blocks = _VAL_ROLLOUT_LEN // _VAL_BLOCK_SIZE
    for b in range(n_blocks):
        t0 = val_start_idx + b * _VAL_BLOCK_SIZE
        stim_block = stimulus[t0:t0 + _VAL_BLOCK_SIZE]
        target_block = voltage[t0 + 1:t0 + _VAL_BLOCK_SIZE + 1]
        z, block_mse = _rollout_block_latent_compiled(model, z, stim_block, target_block)
        all_mse[b * _VAL_BLOCK_SIZE:(b + 1) * _VAL_BLOCK_SIZE] = block_mse

    mse_np = all_mse.cpu().numpy()
    # NaN/Inf during latent drift counts as divergence
    mse_np = np.nan_to_num(mse_np, nan=np.inf, posinf=np.inf, neginf=np.inf)
    above = np.where(mse_np > 1.0)[0]
    div_time = int(above[0]) if len(above) > 0 else _VAL_ROLLOUT_LEN
    rollout_rmse_val = float(np.sqrt(mse_np[:max(div_time, 1)].mean()))
    return mse_np, div_time, rollout_rmse_val


def val_rollout(model, voltage, stimulus, val_start_idx, dt):
    """Single-start rollout validation on training data.

    Runs _VAL_ROLLOUT_LEN steps from val_start_idx using compiled _VAL_BLOCK_SIZE-step blocks.

    Returns:
        mse_curve: (_VAL_ROLLOUT_LEN,) numpy array of per-step MSE
        div_time: first step index where MSE > 1, or _VAL_ROLLOUT_LEN if never
        rollout_rmse: RMSE = sqrt(mean MSE) over steps [0, div_time)
    """
    _rollout_block_compiled = torch.compile(_rollout_block, mode="default")

    all_mse = torch.empty(_VAL_ROLLOUT_LEN, device=voltage.device)
    v = voltage[val_start_idx].clone()

    n_blocks = _VAL_ROLLOUT_LEN // _VAL_BLOCK_SIZE
    for b in range(n_blocks):
        t0 = val_start_idx + b * _VAL_BLOCK_SIZE
        stim_block = stimulus[t0:t0 + _VAL_BLOCK_SIZE]
        target_block = voltage[t0 + 1:t0 + _VAL_BLOCK_SIZE + 1]
        v, block_mse = _rollout_block_compiled(model, v, stim_block, target_block, dt)
        all_mse[b * _VAL_BLOCK_SIZE:(b + 1) * _VAL_BLOCK_SIZE] = block_mse

    mse_np = all_mse.cpu().numpy()
    above = np.where(mse_np > 1.0)[0]
    div_time = int(above[0]) if len(above) > 0 else _VAL_ROLLOUT_LEN
    rollout_rmse_val = float(np.sqrt(mse_np[:max(div_time, 1)].mean()))
    return mse_np, div_time, rollout_rmse_val


def plot_rollout_mse(mse_curve, div_time, epoch, log_dir):
    """Save rollout RMSE vs time step plot, marking divergence time."""
    rmse_curve = np.sqrt(mse_curve)
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(len(rmse_curve))
    ax.plot(steps, rmse_curve, linewidth=1, label='model')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='RMSE=1')
    if div_time < _VAL_ROLLOUT_LEN:
        ax.axvline(div_time, color='red', linestyle=':', linewidth=0.8,
                   label=f'div_time={div_time}')
    ax.legend()
    ax.set_xlabel('Rollout Time Steps')
    ax.set_ylabel('RMSE')
    ax.set_title(f'Validation rollout RMSE — epoch {epoch+1} | div_time={div_time}')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1.0)
    fig.tight_layout()

    plot_dir = os.path.join(log_dir, 'tmp_training')
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, f'rollout_rmse_epoch_{epoch+1:03d}.png'), dpi=100)
    plt.close(fig)


def _compute_loss_multistep(model, voltage, stimulus, t_indices, dt, rollout_steps):
    """Compute MSE loss averaged over a multi-step rollout.

    Unrolls for rollout_steps steps from t_indices, accumulating MSE at each step.
    Backprop through the full rollout penalizes error compounding.
    """
    x = voltage[t_indices]  # (B, N)
    loss = torch.zeros((), device=x.device)
    for k in range(rollout_steps):
        stim_k = stimulus[t_indices + k]              # (B, n_input)
        dvdt = model.predict_dvdt(x, stim_k)
        x = x + dt * dvdt
        target = voltage[t_indices + k + 1]           # (B, N)
        loss = loss + F.mse_loss(x, target)
    return loss / rollout_steps


def data_train_rollout(config, erase, best_model, device, log_file=None):
    """Train a predict_dvdt-compatible model with Euler integration.

    Args:
        config: NeuralGraphConfig
        erase: if True, overwrite existing log directory
        best_model: checkpoint identifier (or None)
        device: torch device
        log_file: optional open file handle for logging
    """
    sim = config.simulation
    tc = config.training

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

    all_voltage = x_ts.voltage                          # (T, N)
    all_stimulus = x_ts.stimulus[:, :n_input_neurons]   # (T, n_input_neurons)

    train_start = tc.train_start
    train_end = tc.train_end if tc.train_end > 0 else n_frames

    voltage = all_voltage[train_start:train_end]          # (T_train, N)
    stimulus = all_stimulus[train_start:train_end]        # (T_train, n_input_neurons)
    n_train_frames = voltage.shape[0]
    _logger.info(f'train split: frames [{train_start}, {train_end}) = {n_train_frames} frames')

    # Fixed validation start: one random point, held constant for the whole run.
    # Need _VAL_ROLLOUT_LEN steps of look-ahead room.
    max_val_start = n_train_frames - _VAL_ROLLOUT_LEN - 1
    val_start_idx = int(torch.randint(0, max_val_start + 1, (1,)).item())
    _logger.info(f'val_start_idx: {val_start_idx} (fixed for this run)')

    checkpoint_path = None
    if tc.pretrained_model != '':
        checkpoint_path = tc.pretrained_model
    model, start_epoch = build_model(config, device, checkpoint_path=checkpoint_path)
    assert hasattr(model, 'predict_dvdt'), (
        f"{type(model).__name__} must implement predict_dvdt(v, stim) → dvdt "
        "to be used with data_train_rollout"
    )

    # Compile per-call so CUDA Graph pools can be freed between CV folds
    _compute_loss_multistep_compiled = torch.compile(
        _compute_loss_multistep, fullgraph=True, mode="reduce-overhead"
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'total parameters: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=tc.lr)

    batch_size = tc.batch_size
    data_passes_per_epoch = tc.data_augmentation_loop
    n_epochs = tc.n_epochs
    rollout_train_steps = tc.rollout_train_steps

    # Valid frame range: need t through t+rollout_train_steps
    max_frame = n_train_frames - rollout_train_steps - 1
    batches_per_epoch = int(max_frame * data_passes_per_epoch / batch_size)

    _logger.info(f'batch_size: {batch_size}, data_passes_per_epoch: {data_passes_per_epoch}')
    _logger.info(f'batches_per_epoch: {batches_per_epoch}, n_epochs: {n_epochs}')
    _logger.info(f'rollout_train_steps: {rollout_train_steps}')

    net_path = os.path.join(log_dir, 'models')
    os.makedirs(net_path, exist_ok=True)

    dt = torch.tensor(sim.delta_t, device=device)

    # Constant model baseline (computed on CPU to avoid large GPU intermediate)
    with torch.no_grad():
        v_cpu = voltage.cpu()
        constant_model_rmse = float(np.sqrt(F.mse_loss(v_cpu[:-1], v_cpu[1:]).item()))
        del v_cpu
    _logger.info(f'constant model baseline RMSE: {constant_model_rmse:.4e}')

    # --- Profiler setup ---
    prof = None
    _prof_stop_after = 0
    _global_step = 0
    if tc.profiling:
        trace_dir = os.path.join(log_dir, 'profiler')
        os.makedirs(trace_dir, exist_ok=True)

        _PROF_WAIT, _PROF_WARMUP, _PROF_ACTIVE = 20, 5, 10

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=_PROF_WAIT, warmup=_PROF_WARMUP, active=_PROF_ACTIVE, repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir, use_gzip=True),
            record_shapes=True,
            with_stack=True,
        )
        prof.start()
        _prof_stop_after = _PROF_WAIT + _PROF_WARMUP + _PROF_ACTIVE
        _logger.info(f'profiler started; trace will be written to {trace_dir}')

    # --- Training loop ---
    _sigusr2_received = False
    def _handle_sigusr2(signum, frame):
        nonlocal _sigusr2_received
        _sigusr2_received = True
        _logger.info('SIGUSR2 received — will stop training after current batch')
    signal.signal(signal.SIGUSR2, _handle_sigusr2)

    model.train()
    training_start = time.time()
    best_epoch = 0

    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_loss = torch.zeros((), device=device)
        n_batches = 0

        pbar = trange(batches_per_epoch, ncols=120, desc=f'epoch {epoch+1}/{n_epochs}')
        for _ in pbar:
            t_indices = torch.randint(0, max_frame + 1, (batch_size,), device=device)

            optimizer.zero_grad()
            loss = _compute_loss_multistep_compiled(
                model, voltage, stimulus, t_indices, dt, rollout_train_steps,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()
            n_batches += 1

            if _sigusr2_received:
                break

            if prof is not None:
                _global_step += 1
                prof.step()
                if _global_step == _prof_stop_after:
                    prof.stop()
                    prof = None
                    _logger.info('profiler stopped; trace written')

            if n_batches % 100 == 0:
                pbar.set_postfix_str(f'loss={epoch_loss / n_batches:.4e}')

        mean_loss = epoch_loss.item() / max(n_batches, 1)
        epoch_duration = time.time() - epoch_start
        total_elapsed = time.time() - training_start

        if _sigusr2_received:
            _logger.info(f'training interrupted at epoch {epoch+1}/{n_epochs}, batch {n_batches}/{batches_per_epoch}')
            best_epoch = epoch + 1
            break

        # --- Validation ---
        val_start_t = time.time()
        model.eval()
        with torch.no_grad():
            mse_curve, div_time, mean_rollout_rmse = val_rollout(
                model, voltage, stimulus, val_start_idx, dt,
            )
        model.train()
        val_duration = time.time() - val_start_t

        val_str = f' | div_time={div_time} rollout_rmse={mean_rollout_rmse:.4e} ({val_duration:.1f}s)'
        plot_rollout_mse(mse_curve, div_time, epoch, log_dir)

        # Save current model as "best" (always overwritten — last epoch wins)
        best_epoch = epoch + 1
        torch.save(
            {'model_state_dict': model.state_dict()},
            os.path.join(net_path, f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt'),
        )

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
            'rollout_rmse': mean_rollout_rmse,
            'div_time': div_time,
        }, os.path.join(net_path, 'latest_checkpoint.pt'))

    total_time = time.time() - training_start

    if _sigusr2_received:
        _logger.info(f'training interrupted: {best_epoch}/{n_epochs} epochs in {total_time:.1f}s')
        with open(os.path.join(log_dir, '_interrupted'), 'w') as f:
            f.write(f'SIGUSR2 at epoch {best_epoch}/{n_epochs}, batch {n_batches}/{batches_per_epoch}, training_time={total_time:.1f}s\n')
    else:
        _logger.info(f'training complete: {n_epochs=} in {total_time=:.1f}s, {div_time=:,d}, {mean_rollout_rmse=:.3e}')

    _logger.info(f'constant model baseline RMSE: {constant_model_rmse:.4e}')

    if log_file:
        log_file.write('\n--- Training rollout results (computed on training data) ---\n')
        log_file.write(f'train_div_time: {div_time}\n')
        log_file.write(f'train_rollout_rmse: {mean_rollout_rmse:.4e}\n')
        log_file.write(f'train_best_epoch: {best_epoch}\n')
        log_file.write(f'train_constant_baseline_rmse: {constant_model_rmse:.4e}\n')
