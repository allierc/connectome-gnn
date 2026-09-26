"""Post-processing of written splits: noisy derivative targets and train tiling.

Moved verbatim from ``graph_data_generator`` (refactor phase 2, step 5);
``graph_data_generator._tile_train_zarrs`` and ``_compute_noisy_derivatives``
stay importable.
"""

import numpy as np

from connectome_gnn.log import get_logger
from connectome_gnn.utils import graphs_data_path

logger = get_logger(__name__)


def _tile_train_zarrs(config, factor: int, save_calcium: bool):
    """Tile the unique train block `factor` times across every dynamic field.

    After data_generate_fly_voltage simulates n_frames // factor unique frames
    and finalises the writers, this helper re-opens each per-field store
    (voltage, stimulus, noise, calcium/fluorescence, y_clean, noisy_y) and
    writes (factor - 1) more copies, yielding a length-(unique_n * factor)
    dataset of the short trajectory repeated end-to-end.
    """
    from pathlib import Path

    import tensorstore as ts

    base = Path(graphs_data_path(config.dataset, "x_list_train"))
    fields = ['voltage', 'stimulus', 'noise']
    if save_calcium:
        fields += ['calcium', 'fluorescence']

    for name in fields:
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(base / f'{name}.zarr')},
        }
        store = ts.open(spec).result()
        unique = store.read().result()
        unique_n, N = unique.shape
        new_n = unique_n * factor
        store = store.resize(exclusive_max=[new_n, N]).result()
        for k in range(1, factor):
            store[unique_n * k: unique_n * (k + 1)].write(unique).result()

    # 3-D arrays: y_list_train and (if present) noisy_y_list_train
    for tag in ("y_list_train", "noisy_y_list_train"):
        path = Path(graphs_data_path(config.dataset, tag) + ".zarr")
        if not path.exists():
            continue
        spec = {'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': str(path)}}
        store = ts.open(spec).result()
        unique = store.read().result()
        unique_n, N, F = unique.shape
        new_n = unique_n * factor
        store = store.resize(exclusive_max=[new_n, N, F]).result()
        for k in range(1, factor):
            store[unique_n * k: unique_n * (k + 1)].write(unique).result()

    logger.info(f"tiled train zarrs ×{factor}: unique_n={unique_n} → total={unique_n * factor}")


def _compute_noisy_derivatives(config, sim, n_neurons, split="train"):
    """Compute noisy derivatives from saved clean derivatives and noise.

    noisy_y[t] = y_clean[t] + (noise[t+1] - noise[t]) / dt
    Last frame uses clean derivative (no future noise available).
    """
    from connectome_gnn.utils import graphs_data_path
    from connectome_gnn.zarr_io import ZarrArrayWriter, load_raw_array, load_simulation_data

    y_clean = load_raw_array(graphs_data_path(config.dataset, f"y_list_{split}"))  # (T, N, 1)
    noise_ts = load_simulation_data(graphs_data_path(config.dataset, f"x_list_{split}"), fields=["noise"])
    noise = noise_ts.noise.numpy()  # (T, N)

    # Compute noise derivative: (noise[t+1] - noise[t]) / dt
    noise_diff = np.zeros_like(noise)
    noise_diff[:-1] = (noise[1:] - noise[:-1]) / sim.delta_t  # last frame: 0

    noisy_y = y_clean + noise_diff[:, :, np.newaxis]  # broadcast to (T, N, 1)

    # Temporal smoothing of noisy derivatives (reduces derivative noise by sqrt(window))
    window = sim.derivative_smoothing_window
    if window > 1:
        from scipy.ndimage import uniform_filter1d

        # Apply centered moving average along time axis (axis=0)
        # mode='nearest' pads boundaries with edge values
        noisy_y = uniform_filter1d(noisy_y, size=window, axis=0, mode="nearest")
        logger.debug(f"  applied derivative smoothing: window={window} (noise reduction ~{1 / np.sqrt(window):.2f}x)")

    noisy_y_writer = ZarrArrayWriter(
        path=graphs_data_path(config.dataset, f"noisy_y_list_{split}"),
        n_neurons=n_neurons,
        n_features=1,
        time_chunks=2000,
    )
    for t in range(noisy_y.shape[0]):
        noisy_y_writer.append(noisy_y[t])
    noisy_y_writer.finalize()
    logger.info(
        f"computed noisy derivatives for {split}: {noisy_y.shape[0]} frames "
        f"(measurement_noise_level={sim.measurement_noise_level})"
    )
