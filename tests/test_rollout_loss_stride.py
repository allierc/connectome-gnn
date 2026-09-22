"""training.rollout_loss_stride scores only the observed steps of the rollout.

WHAT IT REPLACES. `time_step: 5` expressed "one frame in five is observed" by
DECIMATING THE DATASET, which also deepened the rollout and moved the target --
three effects from one number, and a result could not be attributed to any of
them. Here the data is untouched and the model still integrates every
intermediate frame; only the supervision is sparse, which is the honest
statement of partial temporal sampling.
"""
import pytest

from connectome_gnn.models.recurrent_step import _rollout_step_weights


def scored(weights):
    """The 1-indexed steps that carry weight."""
    return [s + 1 for s, w in enumerate(weights) if w > 0.0]


def test_stride_five_over_twenty_scores_the_four_observed_steps():
    assert scored(_rollout_step_weights("uniform", 20, 0.9, loss_stride=5)) == [5, 10, 15, 20]


def test_stride_zero_scores_every_step():
    assert scored(_rollout_step_weights("uniform", 20, 0.9, loss_stride=0)) == list(range(1, 21))


def test_default_is_dense():
    """Omitting the argument must not change the existing objective."""
    assert (_rollout_step_weights("uniform", 7, 0.9)
            == _rollout_step_weights("uniform", 7, 0.9, loss_stride=0))


@pytest.mark.parametrize("weighting", ["uniform", "discount", "linear_decay"])
def test_the_mask_is_applied_after_the_weighting(weighting):
    """The two knobs stay orthogonal: a scored step keeps its dense weight.

    So "one frame in five, discounted" is expressible, rather than the stride
    silently flattening the discount it is combined with.
    """
    dense = _rollout_step_weights(weighting, 20, 0.9)
    strided = _rollout_step_weights(weighting, 20, 0.9, loss_stride=5)
    for s in (4, 9, 14, 19):                      # 0-indexed steps 5, 10, 15, 20
        assert strided[s] == dense[s]
    assert sum(1 for w in strided if w > 0.0) == 4


def test_a_horizon_shorter_than_the_stride_scores_nothing():
    """The failure this exists to make visible: a zero loss that looks like training."""
    assert scored(_rollout_step_weights("uniform", 3, 0.9, loss_stride=5)) == []


def test_k_one_is_still_one_step_training():
    assert _rollout_step_weights("uniform", 1, 0.9, loss_stride=0) == [1.0]


def test_config_defaults_to_dense():
    from connectome_gnn.config import TrainingConfig
    assert TrainingConfig().rollout_loss_stride == 0
