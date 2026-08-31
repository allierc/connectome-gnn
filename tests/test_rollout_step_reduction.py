"""The rollout loss has TWO reductions; they must stay independent.

    loss = REDUCE_s  w_s * fit_reduction( pred_s - y_{k+s} )   s = 0 .. K-1

`fit_reduction` collapses ONE step's residual over n_visible * batch_size, and is
shared with one-step training -- it is the one that couples to batch_size, which
`regul_batch_scaling` compensates. `rollout_step_reduction` collapses the K steps
and has nothing to do with batch_size. Conflating them is what produced the
rs_* specs that carried every coeff_* divided by 70467.
"""
import pytest

from connectome_gnn.config import TrainingConfig
from connectome_gnn.models.recurrent_step import _rollout_step_weights

pytestmark = pytest.mark.tier2


class TestConfigSurface:
    def test_default_is_mean(self):
        assert TrainingConfig(n_epochs=1, batch_size=4).rollout_step_reduction == "mean"

    def test_sum_is_accepted(self):
        assert TrainingConfig(n_epochs=1, batch_size=4,
                              rollout_step_reduction="sum").rollout_step_reduction == "sum"

    def test_unknown_is_rejected(self):
        with pytest.raises(ValueError):
            TrainingConfig(n_epochs=1, batch_size=4, rollout_step_reduction="average")

    @pytest.mark.parametrize("step_red", ["mean", "sum"])
    @pytest.mark.parametrize("fit_red,scaling", [("norm2", "sqrt"), ("norm2", "none"),
                                                 ("mean", "none")])
    def test_the_two_reductions_are_independent(self, step_red, fit_red, scaling):
        """Every valid (fit_reduction, regul_batch_scaling) pair takes either K-step
        reduction: the horizon knob must not be constrained by the batch knob."""
        t = TrainingConfig(n_epochs=1, batch_size=4, fit_reduction=fit_red,
                           regul_batch_scaling=scaling, rollout_step_reduction=step_red)
        assert t.rollout_step_reduction == step_red
        assert t.fit_reduction == fit_red

    def test_step_reduction_does_not_trip_the_batch_validator(self):
        """Only fit_reduction 'mean' + regul_batch_scaling double-corrects."""
        with pytest.raises(ValueError, match="double-corrects"):
            TrainingConfig(n_epochs=1, batch_size=4, fit_reduction="mean",
                           regul_batch_scaling="sqrt", rollout_step_reduction="mean")


class TestStepWeights:
    @pytest.mark.parametrize("w", ["uniform", "discount", "linear_decay", "last"])
    def test_K1_is_a_single_unit_weight(self, w):
        """The K=1 == one-step equality has to hold under every weighting, so at
        K=1 'mean' (divide by sum w = 1) and 'sum' must coincide."""
        assert _rollout_step_weights(w, 1, 0.9) == [1.0]

    def test_uniform_sum_over_mean_is_exactly_K(self):
        """With uniform weights the two K-step reductions differ by the factor K --
        which is why 'sum' is the honest name for "no normalisation" rather than
        something to emulate by hand-scaling the weights."""
        for K in (1, 2, 5, 10):
            w = _rollout_step_weights("uniform", K, 0.9)
            assert sum(w) == pytest.approx(float(K))

    def test_discount_and_last_normalise_to_one_under_mean(self):
        K, gamma = 5, 0.9
        for name in ("uniform", "discount", "linear_decay", "last"):
            w = _rollout_step_weights(name, K, gamma)
            assert sum(w) > 0.0
            assert all(x >= 0.0 for x in w)
            normalised = [x / sum(w) for x in w]
            assert sum(normalised) == pytest.approx(1.0)

    def test_last_scores_only_the_endpoint(self):
        assert _rollout_step_weights("last", 4, 0.9) == [0.0, 0.0, 0.0, 1.0]

    def test_discount_is_gamma_to_the_step(self):
        assert _rollout_step_weights("discount", 3, 0.5) == [1.0, 0.5, 0.25]

    def test_unknown_weighting_raises(self):
        with pytest.raises(ValueError, match="unknown rollout_step_weighting"):
            _rollout_step_weights("cosine", 3, 0.9)
