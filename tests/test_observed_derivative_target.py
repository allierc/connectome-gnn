"""The derivative target must be the finite difference of the OBSERVED voltage.

Guards the twin of the process-noise bug fixed in f88290e: the model is fed
v[t] + eta_t (measurement noise, added per frame in run_nominal_train_step) but
was scored against the finite difference of the clean v[t], an oracle target no
recording can supply.
"""
import numpy as np
import pytest
import torch

from connectome_gnn.models.training_utils import observed_derivative_target
from connectome_gnn.neuron_state import NeuronTimeSeries

pytestmark = pytest.mark.tier3

T, N = 64, 8
DELTA_T = 0.01
GAMMA = 0.1  # measurement_noise_level, in the same units as voltage


@pytest.fixture
def ts():
    """Voltage and a fixed measurement-noise realisation, both (T, N)."""
    rng = np.random.default_rng(0)
    voltage = rng.standard_normal((T, N)).astype(np.float32).cumsum(axis=0) * 0.1
    noise = (rng.standard_normal((T, N)).astype(np.float32) * GAMMA)
    return NeuronTimeSeries(
        voltage=torch.from_numpy(voltage),
        noise=torch.from_numpy(noise),
    )


def test_target_differences_the_observed_voltage(ts):
    y = observed_derivative_target(ts, GAMMA, DELTA_T)

    observed = (ts.voltage + ts.noise).numpy()
    expected = (observed[1:] - observed[:-1]) / DELTA_T

    assert y.shape == (T, N, 1)
    np.testing.assert_allclose(y[:-1, :, 0], expected, rtol=1e-5, atol=1e-5)


def test_last_frame_repeats_its_predecessor(ts):
    y = observed_derivative_target(ts, GAMMA, DELTA_T)
    np.testing.assert_array_equal(y[-1], y[-2])


def test_noise_free_path_is_unchanged(ts):
    """measurement_noise_level == 0 must difference the clean voltage exactly,
    so every noise-free run reproduces bit-for-bit."""
    y = observed_derivative_target(ts, 0.0, DELTA_T)

    clean = ts.voltage.numpy()
    expected = (clean[1:] - clean[:-1]) / DELTA_T

    np.testing.assert_allclose(y[:-1, :, 0], expected, rtol=1e-6, atol=1e-6)


def test_missing_noise_field_raises_rather_than_falling_back(ts):
    """A dataset whose noise.zarr was never written loads with .noise = None,
    and from_zarr_v3 does not complain. Silently differencing the clean voltage
    there would put the oracle target straight back, so it must raise."""
    bare = NeuronTimeSeries(voltage=ts.voltage)

    with pytest.raises(AssertionError, match="noise.zarr is missing"):
        observed_derivative_target(bare, GAMMA, DELTA_T)


def test_missing_noise_field_is_fine_when_noise_free(ts):
    """The same dataset is legitimate at measurement_noise_level 0."""
    bare = NeuronTimeSeries(voltage=ts.voltage)
    np.testing.assert_array_equal(
        observed_derivative_target(bare, 0.0, DELTA_T),
        observed_derivative_target(ts, 0.0, DELTA_T),
    )


def test_measurement_noise_actually_enters_the_target(ts):
    """The regression itself: with gamma > 0 the target must NOT equal the
    clean-voltage difference. Expected gap is std(d(eta)/dt) = gamma*sqrt(2)/dt."""
    noisy = observed_derivative_target(ts, GAMMA, DELTA_T)
    clean = observed_derivative_target(ts, 0.0, DELTA_T)

    gap = (noisy - clean)[:-1].std()
    assert gap == pytest.approx(GAMMA * np.sqrt(2) / DELTA_T, rel=0.2)
