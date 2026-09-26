"""training.derivative_target picks the nominal target or reproduces the bug.

Two values, because the target has been wrong in exactly one way: `y_list` is
the generator's own stored derivative, f(v), and the integrator adds the process
noise xi to the STATE after that right-hand side was evaluated. So any finite
difference of the trajectory carries xi/dt and f(v) does not, and training
against y_list asks the model to predict a trajectory the data does not follow.

The flag exists so a before/after pair is two arms of one sweep at ONE commit
rather than two checkouts. That is worth something only if it reproduces the old
behaviour exactly, which is what these tests pin.
"""
import numpy as np
import pytest
import torch

from connectome_gnn.models.training_utils import observed_derivative_target

DELTA_T = 0.02
GAMMA = 0.1
T, N = 60, 9


class _TS:
    """The two fields observed_derivative_target reads."""

    def __init__(self, voltage, noise):
        self.voltage = voltage
        self.noise = noise


def _trajectory(xi_level, seed=0):
    """A trajectory built the way the generator builds one.

    v[t+1] = v[t] + dt*f(v[t]) + xi[t], with f a fixed linear leak so that the
    analytic right-hand side is known exactly and `y_list` can be constructed
    the way the generator stores it: evaluated BEFORE xi is added.
    """
    rng = np.random.default_rng(seed)
    v = np.zeros((T, N))
    f = np.zeros((T, N))
    v[0] = rng.normal(size=N)
    for t in range(T - 1):
        f[t] = -3.0 * v[t] + 0.5            # the generator's own dv/dt
        xi = rng.normal(size=N) * xi_level
        v[t + 1] = v[t] + DELTA_T * f[t] + xi
    f[-1] = -3.0 * v[-1] + 0.5
    eta = rng.normal(size=(T, N)) * GAMMA
    return v, f, eta


@pytest.fixture
def ts():
    g = torch.Generator().manual_seed(0)
    voltage = torch.randn(T, N, generator=g)
    noise = torch.randn(T, N, generator=g) * GAMMA
    return _TS(voltage, noise)


def test_observed_fd_differences_the_observed_voltage(ts):
    v = (ts.voltage + ts.noise).numpy()
    want = np.zeros_like(v)
    want[:-1] = (v[1:] - v[:-1]) / DELTA_T
    want[-1] = want[-2]
    np.testing.assert_allclose(observed_derivative_target(ts, GAMMA, DELTA_T),
                               want[..., None], rtol=0, atol=0)


def test_observed_fd_carries_the_measurement_noise_derivative(ts):
    """d(eta)/dt has standard deviation gamma*sqrt(2)/dt -- here 7.07."""
    noisy = observed_derivative_target(ts, GAMMA, DELTA_T)
    clean = observed_derivative_target(ts, 0.0, DELTA_T)
    gap = (noisy - clean).ravel()[:-N]
    expected = GAMMA * np.sqrt(2.0) / DELTA_T
    assert 0.5 * expected < gap.std() < 2.0 * expected


@pytest.mark.parametrize("xi_level", [0.0, 0.05, 0.5])
def test_the_bug_is_the_missing_process_noise(xi_level):
    """y_list differs from the trajectory's own derivative by exactly xi/dt.

    This is the whole experiment in one assertion: the gap is zero when the
    generator added no process noise and grows in proportion to it, which is why
    the sweep varies noise_model_level rather than the measurement noise.
    """
    v, f, _eta = _trajectory(xi_level)
    fd = np.zeros_like(v)
    fd[:-1] = (v[1:] - v[:-1]) / DELTA_T
    fd[-1] = fd[-2]
    gap = (fd - f)[:-1]                     # clean finite difference minus y_list
    if xi_level == 0.0:
        np.testing.assert_allclose(gap, 0.0, atol=1e-12)
    else:
        assert np.isclose(gap.std(), xi_level / DELTA_T, rtol=0.25)


def test_config_defaults_to_the_nominal_target():
    from connectome_gnn.config import TrainingConfig
    assert TrainingConfig().derivative_target == "observed_fd"


def test_config_rejects_an_unknown_target():
    """A typo must not fall through to the default and print as a fix."""
    import pydantic
    from connectome_gnn.config import TrainingConfig
    with pytest.raises(pydantic.ValidationError):
        TrainingConfig(derivative_target="analytic")
