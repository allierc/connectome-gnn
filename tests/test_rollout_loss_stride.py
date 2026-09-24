"""training.rollout_loss_stride scores only the observed steps of the rollout.

WHAT IT REPLACES. `time_step: 5` expressed "one frame in five is observed" by
DECIMATING THE DATASET, which also deepened the rollout and moved the target --
three effects from one number, and a result could not be attributed to any of
them. Here the data is untouched and the model still integrates every
intermediate frame; only the supervision is sparse, which is the honest
statement of partial temporal sampling.

THE TEST CLASS THAT WAS MISSING. Until 2026-09-24 every test in this file
called `_rollout_step_weights` and asserted on a list of floats. Nothing tied
the mask to the FRAME the scored step consumes, so an off-by-one lived here
undetected through thirty cluster runs: the mask read `(s + 1) % m == 0`, which
selects s = m-1, 2m-1, ..., while `_dense_rollout_loss` runs step s with the
state at frame k+s. With m = 5 the loss landed on k+4, k+9, k+14, k+19 -- and a
1-in-5 recording anchored at k observes k, k+5, k+10, k+15, k+20, so not one
scored frame was an observed one. `test_scored_frames_are_the_observed_frames`
is the assertion that would have caught it, and it is written against the
observation grid rather than against the mask, so it cannot drift with it.
"""
import pytest

from connectome_gnn.models.recurrent_step import _rollout_step_weights


def scored(weights):
    """The 0-INDEXED steps that carry weight.

    0-indexed because that is what `_dense_rollout_loss` iterates and what
    indexes the target: at step s the state is frame k+s and the target is
    y_ts[k+s]. The old helper returned s+1, which is the convention the
    off-by-one hid behind.
    """
    return [s for s, w in enumerate(weights) if w > 0.0]


def observed_frames(k_span, stride):
    """Offsets from k that a 1-in-`stride` recording anchored at k actually has."""
    return list(range(0, k_span, stride))


# --------------------------------------------------------------------------- #
#  the frames, which is the thing that matters                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("horizon,stride", [(6, 5), (21, 5), (11, 5), (7, 3), (4, 2)])
def test_scored_frames_are_the_observed_frames(horizon, stride):
    """Every scored step sits on the observation grid, and none is missed.

    This is the whole point of the knob, and it is stated against the grid a
    recording would have rather than against the mask's own arithmetic.
    """
    got = scored(_rollout_step_weights("uniform", horizon, 0.9, loss_stride=stride))
    assert got == observed_frames(horizon, stride)


def test_step_zero_is_always_scored():
    """Step 0 is the un-integrated observed frame k: the only zero-drift anchor.

    Dropping it was half the 2026-09-24 defect -- it left the objective with no
    term evaluated at an exact observation, so nothing pinned the prediction's
    scale and the group lasso was free to shrink the message to zero.
    """
    for stride in (2, 3, 5, 10):
        assert 0 in scored(_rollout_step_weights("uniform", 20, 0.9, loss_stride=stride))


def test_no_scored_frame_is_unobserved():
    """The failure mode by name: scoring a frame the regime cannot supply."""
    got = scored(_rollout_step_weights("uniform", 21, 0.9, loss_stride=5))
    unobserved = set(range(21)) - set(observed_frames(21, 5))
    assert not (set(got) & unobserved)


def test_a_horizon_scores_floor_plus_one_steps():
    """K steps and stride m give floor((K-1)/m) + 1 scored terms."""
    for horizon in range(1, 30):
        n = len(scored(_rollout_step_weights("uniform", horizon, 0.9, loss_stride=5)))
        assert n == (horizon - 1) // 5 + 1


# --------------------------------------------------------------------------- #
#  the weights themselves                                                      #
# --------------------------------------------------------------------------- #
def test_stride_zero_scores_every_step():
    assert scored(_rollout_step_weights("uniform", 20, 0.9, loss_stride=0)) == list(range(20))


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
    dense = _rollout_step_weights(weighting, 21, 0.9)
    strided = _rollout_step_weights(weighting, 21, 0.9, loss_stride=5)
    for s in (0, 5, 10, 15, 20):
        assert strided[s] == dense[s]
    assert sum(1 for w in strided if w > 0.0) == 5


def test_k_one_is_still_one_step_training():
    assert _rollout_step_weights("uniform", 1, 0.9, loss_stride=0) == [1.0]
    # and with a stride, because step 0 is scored
    assert _rollout_step_weights("uniform", 1, 0.9, loss_stride=5) == [1.0]


def test_config_defaults_to_dense():
    from connectome_gnn.config import TrainingConfig
    assert TrainingConfig().rollout_loss_stride == 0


# --------------------------------------------------------------------------- #
#  the guard                                                                   #
# --------------------------------------------------------------------------- #
def test_a_horizon_that_scores_only_the_anchor_is_one_step_training():
    """K <= m reaches step 0 and no further, so the rollout does unscored work.

    graph_trainer refuses such a schedule; this records WHY, at the level of the
    weights, so the two cannot drift apart.
    """
    for horizon in (1, 3, 5):
        assert scored(_rollout_step_weights("uniform", horizon, 0.9, loss_stride=5)) == [0]
    assert scored(_rollout_step_weights("uniform", 6, 0.9, loss_stride=5)) == [0, 5]
