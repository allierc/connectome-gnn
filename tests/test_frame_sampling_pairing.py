"""Arms of one comparison must sample start frames from the same range.

`get_training_frame_sampling` computes last_frame = n_frames - 4 - target_offset.
The rollout curriculum passes target_offset = max(K); the one-step path derives it
from time_step. So a K=1..5 curriculum drew from 63,991 while its own t+1 control
drew from 63,995 — the two were never RNG-paired, and any Task-1 difference
smaller than the run-to-run floor was unattributable.

The invariant that should hold, and did not: **a K=1 recurrent run is one-step
training**, so it must sample identically to `recurrent_training: false`.

`training.frame_target_offset` pins the range for every arm of a comparison.
"""
import pytest

from connectome_gnn.config import SimulationConfig, TrainingConfig
from connectome_gnn.models.training_utils import get_training_frame_sampling

pytestmark = pytest.mark.tier2

# Built in-memory rather than read from config/fly: those specs are gitignored
# experiment configs that come and go with each grid, and this invariant is a
# property of the sampler, not of any one spec.
N_FRAMES = 64000

ONESTEP = dict(recurrent_training=False)
ROLLOUT = dict(recurrent_training=True, rollout_horizon_schedule=[1, 2, 3, 4, 5])


class _Cfg:
    def __init__(self, **train):
        self.simulation = SimulationConfig(
            params=[[1.0, 1.0, 1.0, 1.0]], n_frames=N_FRAMES, delta_t=0.02,
            n_neurons=16, n_edges=32, n_input_neurons=2, n_neuron_types=2)
        self.training = TrainingConfig(n_epochs=5, batch_size=4, time_step=1, **train)


def _cfg(arm):
    return _Cfg(**arm)


def _range(cfg, target_offset=None):
    return get_training_frame_sampling(
        cfg.simulation, cfg.training, target_offset=target_offset).frame_range


class TestK1EqualsOneStep:
    def test_k1_curriculum_samples_like_one_step(self):
        """A curriculum whose only horizon is 1 IS one-step training."""
        one = _cfg(ONESTEP)
        roll = _cfg(ROLLOUT)
        roll.training.rollout_horizon_schedule = [1] * roll.training.n_epochs
        assert _range(roll, target_offset=1) == _range(one)

    def test_longer_curriculum_shrinks_the_range_without_a_pin(self):
        """Documents the defect: this is why onestep and the rollout arms differed."""
        one, roll = _cfg(ONESTEP), _cfg(ROLLOUT)
        assert _range(roll, target_offset=5) < _range(one)
        assert _range(one) - _range(roll, target_offset=5) == 4


class TestPin:
    def test_pin_equalises_every_arm(self):
        one, roll = _cfg(ONESTEP), _cfg(ROLLOUT)
        one.training.frame_target_offset = 5
        roll.training.frame_target_offset = 5
        assert _range(one) == _range(roll, target_offset=5)

    def test_pin_overrides_the_callers_argument(self):
        """The rollout path passes target_offset=max(K); an explicit pin must win,
        otherwise the one-step arm could never be made to match it."""
        roll = _cfg(ROLLOUT)
        roll.training.frame_target_offset = 5
        assert _range(roll, target_offset=1) == _range(roll, target_offset=5)

    def test_zero_pin_is_the_previous_behaviour(self):
        one, roll = _cfg(ONESTEP), _cfg(ROLLOUT)
        assert one.training.frame_target_offset == 0
        assert _range(one) == 63995
        assert _range(roll, target_offset=5) == 63991

    def test_pin_is_monotone_in_offset(self):
        one = _cfg(ONESTEP)
        prev = None
        for off in (1, 2, 5, 10):
            one.training.frame_target_offset = off
            r = _range(one)
            if prev is not None:
                assert r < prev
            prev = r
