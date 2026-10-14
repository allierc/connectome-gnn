"""Post-processing of written splits: noisy derivative targets and train tiling.

Both read back what the frame loop wrote and write zarr stores; neither
draws RNG.
"""

from pathlib import Path

import numpy as np

from connectome_gnn.log import get_logger
from connectome_gnn.zarr_io import ZarrArrayWriter, load_raw_array, load_simulation_data

logger = get_logger(__name__)


def tile_train(store, factor: int, save_calcium) -> None:
    """Tile the unique train block `factor` times across every dynamic field.

    After the train split simulated n_frames // factor unique frames and
    finalised its writers, this re-opens each per-field store (voltage,
    stimulus, noise, calcium/fluorescence, y_clean, noisy_y) and writes
    (factor - 1) more copies, yielding a length-(unique_n * factor) dataset of
    the short trajectory repeated end-to-end.

    QUIRK: noisy_y_list_train.zarr is tiled WHENEVER IT EXISTS, including a
    stale one that the legacy erase left behind from an earlier noisy run
    (golden cell dirty_noerase_tile); see ``VoltageGeneration.tile_train``.
    """
    import tensorstore as ts

    base = Path(store.path("x_list_train"))
    fields = ['voltage', 'stimulus', 'noise']
    if save_calcium:
        fields += ['calcium', 'fluorescence']

    for name in fields:
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(base / f'{name}.zarr')},
        }
        zstore = ts.open(spec).result()
        unique = zstore.read().result()
        unique_n, N = unique.shape
        new_n = unique_n * factor
        zstore = zstore.resize(exclusive_max=[new_n, N]).result()
        for k in range(1, factor):
            zstore[unique_n * k: unique_n * (k + 1)].write(unique).result()

    # 3-D arrays: y_list_train and (if present) noisy_y_list_train
    for tag in ("y_list_train", "noisy_y_list_train"):
        path = Path(store.path(tag) + ".zarr")
        if not path.exists():
            continue
        spec = {'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': str(path)}}
        zstore = ts.open(spec).result()
        unique = zstore.read().result()
        unique_n, N, F = unique.shape
        new_n = unique_n * factor
        zstore = zstore.resize(exclusive_max=[new_n, N, F]).result()
        for k in range(1, factor):
            zstore[unique_n * k: unique_n * (k + 1)].write(unique).result()

    logger.info(f"tiled train zarrs ×{factor}: unique_n={unique_n} → total={unique_n * factor}")


def noisy_derivatives(spec, store, n_neurons: int, split: str = "train") -> None:
    """Write noisy_y_list_<split>.zarr from the saved clean derivatives and measurement noise.

    noisy_y[t] = y_clean[t] + (noise[t+1] - noise[t]) / dt, optionally smoothed
    over ``derivative_smoothing_window`` frames; the last frame uses the clean
    derivative (no future noise available). QUIRK: it carries eta but NOT the
    process noise xi/dt; see ``VoltageGeneration.derive_noisy_targets``.
    """
    y_clean = load_raw_array(store.path(f"y_list_{split}"))  # (T, N, 1)
    noise_ts = load_simulation_data(store.path(f"x_list_{split}"), fields=["noise"])
    noise = noise_ts.noise.numpy()  # (T, N)

    # Compute noise derivative: (noise[t+1] - noise[t]) / dt
    noise_diff = np.zeros_like(noise)
    noise_diff[:-1] = (noise[1:] - noise[:-1]) / spec.delta_t  # last frame: 0

    noisy_y = y_clean + noise_diff[:, :, np.newaxis]  # broadcast to (T, N, 1)

    # Temporal smoothing of noisy derivatives (reduces derivative noise by sqrt(window))
    window = spec.derivative_smoothing_window
    if window > 1:
        from scipy.ndimage import uniform_filter1d

        # Apply centered moving average along time axis (axis=0)
        # mode='nearest' pads boundaries with edge values
        noisy_y = uniform_filter1d(noisy_y, size=window, axis=0, mode="nearest")
        logger.debug(f"  applied derivative smoothing: window={window} (noise reduction ~{1 / np.sqrt(window):.2f}x)")

    noisy_y_writer = ZarrArrayWriter(
        path=store.path(f"noisy_y_list_{split}"),
        n_neurons=n_neurons,
        n_features=1,
        time_chunks=2000,
    )
    for t in range(noisy_y.shape[0]):
        noisy_y_writer.append(noisy_y[t])
    noisy_y_writer.finalize()
    # The level printed is the TRAIN one for either split, as legacy printed it.
    logger.info(
        f"computed noisy derivatives for {split}: {noisy_y.shape[0]} frames "
        f"(measurement_noise_level={spec.noise_for(split).gate_measurement_std})"
    )
