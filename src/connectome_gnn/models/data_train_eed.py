"""EED (Encode-Evolve-Decode) native training loop.

Trains an EED model using two losses:
- Reconstruction loss: MSE(decoder(encoder(x_t)), x_t)
- Evolution loss:      MSE(decoder(evolver(encoder(x_t), stim_encoder(stim_t))), x_{t+1})

Validation uses predict_dvdt + Euler rollout (mathematically equivalent)
so the test/plot pipeline works unchanged.
"""

import os
import signal
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from connectome_gnn.log import get_logger
from connectome_gnn.models.data_train_rollout import val_rollout_latent, plot_rollout_mse
from connectome_gnn.models.training_utils import build_model, load_flyvis_data, determine_load_fields
from connectome_gnn.utils import create_log_dir

_logger = get_logger(__name__)


def compute_eed_loss(model, voltage, stimulus, t_indices, dt, rollout_steps):
    """Compute reconstruction + multi-step pure-latent rollout loss for a batch.

    Reconstruction loss at t=0: MSE(decoder(encoder(x_t)), x_t).
    Rollout loss at t=1..rollout_steps: chain evolver in latent space, decode each
    step, accumulate MSE against true voltage. This matches the test pipeline
    (graph_tester.py) which encodes once and never re-encodes.

    Args:
        model: EEDBaseline with encoder, decoder, stimulus_encoder, evolver
        voltage: (T, N) full voltage tensor
        stimulus: (T, n_input) full stimulus tensor
        t_indices: (B,) random time indices
        dt: scalar time step (unused; kept for signature compatibility)
        rollout_steps: number of latent-evolver steps to unroll

    Returns:
        total_loss, recon_loss, evolve_loss
    """
    x_t = voltage[t_indices]           # (B, N)

    z = model.encoder(x_t)
    x_recon = model.decoder(z)
    recon_loss = F.mse_loss(x_recon, x_t)

    evolve_loss = torch.zeros((), device=x_t.device)
    for k in range(rollout_steps):
        stim_k = stimulus[t_indices + k]
        z_stim = model.stimulus_encoder(stim_k)
        z = z + model.evolver(torch.cat([z, z_stim], dim=1))
        x_pred = model.decoder(z)
        target = voltage[t_indices + k + 1]
        evolve_loss = evolve_loss + F.mse_loss(x_pred, target)
    evolve_loss = evolve_loss / rollout_steps

    total_loss = recon_loss + evolve_loss
    return total_loss, recon_loss, evolve_loss


def data_train_eed(config, erase, best_model, device, log_file=None):
    """Train an EED model with native encode-evolve-decode losses.

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

    voltage = all_voltage[train_start:train_end]
    stimulus = all_stimulus[train_start:train_end]
    n_train_frames = voltage.shape[0]
    _logger.info(f'train split: frames [{train_start}, {train_end}) = {n_train_frames} frames')

    # Validation start for rollout (same logic as data_train_rollout)
    from connectome_gnn.models.data_train_rollout import _VAL_ROLLOUT_LEN
    max_val_start = n_train_frames - _VAL_ROLLOUT_LEN - 1
    val_start_idx = int(torch.randint(0, max_val_start + 1, (1,)).item())
    _logger.info(f'val_start_idx: {val_start_idx} (fixed for this run)')

    checkpoint_path = None
    if tc.pretrained_model != '':
        checkpoint_path = tc.pretrained_model
    model, start_epoch = build_model(config, device, checkpoint_path=checkpoint_path)
    assert hasattr(model, 'encoder') and hasattr(model, 'evolver'), (
        f"{type(model).__name__} must have encoder/decoder/evolver/stimulus_encoder "
        "sub-networks to be used with data_train_eed"
    )

    # Compile per-call so CUDA Graph pools can be freed between CV folds
    compute_eed_loss_compiled = torch.compile(
        compute_eed_loss, fullgraph=True, mode="reduce-overhead"
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'total parameters: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=tc.lr)

    batch_size = tc.batch_size
    data_passes_per_epoch = tc.data_augmentation_loop
    n_epochs = tc.n_epochs

    rollout_steps = tc.rollout_train_steps
    # Valid frame range: need t through t+rollout_steps
    max_frame = n_train_frames - rollout_steps - 1
    batches_per_epoch = int(max_frame * data_passes_per_epoch / batch_size)

    _logger.info(f'batch_size: {batch_size}, data_passes_per_epoch: {data_passes_per_epoch}')
    _logger.info(f'batches_per_epoch: {batches_per_epoch}, n_epochs: {n_epochs}')

    net_path = os.path.join(log_dir, 'models')
    os.makedirs(net_path, exist_ok=True)

    dt = torch.tensor(sim.delta_t, device=device)

    # Constant model baseline (computed on CPU to avoid large GPU intermediate)
    with torch.no_grad():
        v_cpu = voltage.cpu()
        constant_model_rmse = float(np.sqrt(F.mse_loss(v_cpu[:-1], v_cpu[1:]).item()))
        del v_cpu
    _logger.info(f'constant model baseline RMSE: {constant_model_rmse:.4e}')

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
        epoch_total = torch.zeros((), device=device)
        epoch_recon = torch.zeros((), device=device)
        epoch_evolve = torch.zeros((), device=device)
        n_batches = 0

        pbar = trange(batches_per_epoch, ncols=120, desc=f'epoch {epoch+1}/{n_epochs}')
        for _ in pbar:
            t_indices = torch.randint(0, max_frame + 1, (batch_size,), device=device)

            optimizer.zero_grad()
            total_loss, recon_loss, evolve_loss = compute_eed_loss_compiled(
                model, voltage, stimulus, t_indices, dt, rollout_steps,
            )
            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.detach()
            epoch_recon += recon_loss.detach()
            epoch_evolve += evolve_loss.detach()
            n_batches += 1

            if _sigusr2_received:
                break

            if n_batches % 100 == 0:
                pbar.set_postfix_str(
                    f'total={epoch_total / n_batches:.4e} '
                    f'recon={epoch_recon / n_batches:.4e} '
                    f'evolve={epoch_evolve / n_batches:.4e}'
                )

        mean_total = epoch_total.item() / max(n_batches, 1)
        mean_recon = epoch_recon.item() / max(n_batches, 1)
        mean_evolve = epoch_evolve.item() / max(n_batches, 1)
        epoch_duration = time.time() - epoch_start
        total_elapsed = time.time() - training_start

        if _sigusr2_received:
            _logger.info(f'training interrupted at epoch {epoch+1}/{n_epochs}, batch {n_batches}/{batches_per_epoch}')
            best_epoch = epoch + 1
            break

        # --- Validation rollout (PURE LATENT: encode once, chain evolver, decode) ---
        val_start_t = time.time()
        model.eval()
        with torch.no_grad():
            mse_curve, div_time, mean_rollout_rmse = val_rollout_latent(
                model, voltage, stimulus, val_start_idx, dt,
            )
        model.train()
        val_duration = time.time() - val_start_t

        plot_rollout_mse(mse_curve, div_time, epoch, log_dir)

        # Save model
        best_epoch = epoch + 1
        torch.save(
            {'model_state_dict': model.state_dict()},
            os.path.join(net_path, f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt'),
        )

        _logger.info(
            f'epoch {epoch+1}/{n_epochs} | '
            f'total: {mean_total:.4e} recon: {mean_recon:.4e} evolve: {mean_evolve:.4e} | '
            f'div_time={div_time} rollout_rmse={mean_rollout_rmse:.4e} ({val_duration:.1f}s) | '
            f'duration: {epoch_duration:.1f}s (total: {total_elapsed:.1f}s)'
        )

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': mean_total,
            'recon_loss': mean_recon,
            'evolve_loss': mean_evolve,
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
        log_file.write('\n--- Training EED results (computed on training data) ---\n')
        log_file.write(f'train_div_time: {div_time}\n')
        log_file.write(f'train_rollout_rmse: {mean_rollout_rmse:.4e}\n')
        log_file.write(f'train_best_epoch: {best_epoch}\n')
        log_file.write(f'train_constant_baseline_rmse: {constant_model_rmse:.4e}\n')
        log_file.write(f'train_final_recon_loss: {mean_recon:.4e}\n')
        log_file.write(f'train_final_evolve_loss: {mean_evolve:.4e}\n')
