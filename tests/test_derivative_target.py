"""training.derivative_target = clean_fd reproduces the pre-#58 target.

The choice exists so that a before/after comparison can be two arms of one
five-fold at ONE commit, rather than two checkouts of the repo. That is only
worth anything if the flag reproduces the old behaviour exactly, which is what
these tests pin: ON must equal differencing the clean voltage, OFF must equal
differencing the observed one, and on a noise-free dataset the two must be
bit-identical so the flag cannot silently change an experiment it does not apply
to.
"""
import numpy as np
import pytest
import torch

from connectome_gnn.models.training_utils import observed_derivative_target

DELTA_T = 0.01
GAMMA = 0.1
T, N = 40, 7


class _TS:
    """The two fields observed_derivative_target reads."""

    def __init__(self, voltage, noise):
        self.voltage = voltage
        self.noise = noise


@pytest.fixture
def ts():
    g = torch.Generator().manual_seed(0)
    voltage = torch.randn(T, N, generator=g)
    noise = torch.randn(T, N, generator=g) * GAMMA
    return _TS(voltage, noise)


def _flagged(ts, clean, gamma=GAMMA):
    """What init_training_data computes for a given flag value."""
    return observed_derivative_target(ts, 0.0 if clean else gamma, DELTA_T)


def test_on_differences_the_clean_voltage(ts):
    v = ts.voltage.numpy()
    want = np.zeros_like(v)
    want[:-1] = (v[1:] - v[:-1]) / DELTA_T
    want[-1] = want[-2]
    np.testing.assert_allclose(_flagged(ts, clean=True), want[..., None], rtol=0, atol=0)


def test_off_differences_the_observed_voltage(ts):
    v = (ts.voltage + ts.noise).numpy()
    want = np.zeros_like(v)
    want[:-1] = (v[1:] - v[:-1]) / DELTA_T
    want[-1] = want[-2]
    np.testing.assert_allclose(_flagged(ts, clean=False), want[..., None], rtol=0, atol=0)


def test_the_flag_changes_the_target_by_the_noise_derivative(ts):
    """The gap is d(eta)/dt, whose scale is what makes the bug matter.

    At gamma 0.1 and dt 0.01 its standard deviation is 0.1*sqrt(2)/0.01 = 14.1
    in the same units as dv/dt. A flag whose effect were negligible would not be
    worth a five-fold.
    """
    gap = (_flagged(ts, clean=False) - _flagged(ts, clean=True)).ravel()[:-N]
    expected_sd = GAMMA * np.sqrt(2.0) / DELTA_T
    assert 0.5 * expected_sd < gap.std() < 2.0 * expected_sd


def test_no_op_without_measurement_noise(ts):
    """On a noise-free dataset the flag must not change a single value."""
    np.testing.assert_array_equal(_flagged(ts, clean=True, gamma=0.0),
                                  _flagged(ts, clean=False, gamma=0.0))


def test_config_defaults_to_the_fix():
    from connectome_gnn.config import TrainingConfig
    assert TrainingConfig().derivative_target == "observed_fd"


def test_config_rejects_an_unknown_target():
    """A typo must not silently fall through to the default and print as a fix."""
    import pydantic
    from connectome_gnn.config import TrainingConfig
    with pytest.raises(pydantic.ValidationError):
        TrainingConfig(derivative_target="clean")
