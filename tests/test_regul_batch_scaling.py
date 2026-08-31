"""Regularisation strength must not depend on batch_size when asked not to.

The fit term is one norm2 over the whole batched graph -- ||r||_2 over
n_visible * batch_size elements -- so it grows as sqrt(batch_size). The
regularisers are parameter norms computed once per iteration, independent of B.
At fixed coeff_* the regulariser/fit ratio therefore falls as 1/sqrt(B): batch_size
is silently a regularisation hyperparameter, and coeff_* cannot be copied between
configs with different batch sizes. Three of the NeurIPS paper's configs sit at
B=1 or B=16 rather than 4.

`regul_batch_scaling: sqrt` multiplies the coefficients by sqrt(B) so the ratio is
constant. Default 'none' reproduces the historical behaviour exactly.
"""
import pytest

from connectome_gnn.config import (GraphModelConfig, NeuralGraphConfig, PlottingConfig,
                                   SimulationConfig, TrainingConfig)
from connectome_gnn.models.regularizer import LossRegularizer

pytestmark = pytest.mark.tier2

COEFFS = dict(coeff_W_L1=1.5e-4, coeff_g_phi_diff=750.0, coeff_g_phi_norm=0.9,
              coeff_g_phi_weight_L1=0.28, regul_annealing_rate=0.0)


def _reg(batch_size, scaling):
    t = TrainingConfig(n_epochs=1, batch_size=batch_size,
                       regul_batch_scaling=scaling, **COEFFS)
    mc = GraphModelConfig(signal_model_name="flyvis_A", prediction="first_derivative",
                          input_size=3, n_layers=2, hidden_dim=16, output_size=1,
                          input_size_update=5, n_layers_update=2, hidden_dim_update=16,
                          output_size_update=1, aggr_type="add", embedding_dim=2,
                          update_type="generic")
    sim = SimulationConfig(params=[[1.0, 1.0, 1.0, 1.0]], n_frames=1000, delta_t=0.02,
                           n_neurons=16, n_edges=32, n_input_neurons=2, n_neuron_types=2)
    NeuralGraphConfig(dataset="unit-test", simulation=sim, training=t,
                      graph_model=mc, plotting=PlottingConfig())
    return LossRegularizer(train_config=t, model_config=mc, activity_column=3,
                           plot_frequency=1, n_neurons=16, trainer_type="flyvis",
                           n_neuron_types=0)


class TestDefaultIsHistorical:
    def test_none_is_the_default(self):
        assert TrainingConfig(n_epochs=1, batch_size=4).regul_batch_scaling == "none"

    @pytest.mark.parametrize("B", [1, 4, 16])
    def test_none_leaves_coefficients_untouched(self, B):
        c = _reg(B, "none")._coeffs
        assert c["W_L1"] == pytest.approx(COEFFS["coeff_W_L1"])
        assert c["g_phi_diff"] == pytest.approx(COEFFS["coeff_g_phi_diff"])
        assert c["g_phi_norm"] == pytest.approx(COEFFS["coeff_g_phi_norm"])


class TestSqrtScaling:
    @pytest.mark.parametrize("B", [1, 2, 4, 8, 16])
    def test_coefficients_scale_as_sqrt_batch(self, B):
        c = _reg(B, "sqrt")._coeffs
        s = B ** 0.5
        assert c["W_L1"] == pytest.approx(COEFFS["coeff_W_L1"] * s)
        assert c["g_phi_diff"] == pytest.approx(COEFFS["coeff_g_phi_diff"] * s)
        assert c["g_phi_norm"] == pytest.approx(COEFFS["coeff_g_phi_norm"] * s)

    def test_ratio_to_the_fit_term_is_batch_invariant(self):
        """The point of the flag: regulariser / fit is constant in B.

        The fit is ||r||_2 over n*B elements, i.e. proportional to sqrt(B) for a
        fixed per-element residual. Under 'sqrt' the coefficients carry the same
        sqrt(B), so the ratio cancels.
        """
        ratios = []
        for B in (1, 2, 4, 8, 16):
            regul = _reg(B, "sqrt")._coeffs["g_phi_diff"]
            fit = B ** 0.5                       # ||r||_2 scaling, per-element fixed
            ratios.append(regul / fit)
        assert max(ratios) == pytest.approx(min(ratios), rel=1e-12)

    def test_none_ratio_is_NOT_batch_invariant(self):
        """Documents the coupling the flag exists to remove."""
        ratios = [_reg(B, "none")._coeffs["g_phi_diff"] / (B ** 0.5)
                  for B in (1, 4, 16)]
        assert ratios[0] == pytest.approx(4 * ratios[2])   # B=1 is 4x B=16

    def test_scaling_survives_annealing(self):
        """Annealed and non-annealed coefficients must both carry the factor."""
        t = TrainingConfig(n_epochs=5, batch_size=4, regul_batch_scaling="sqrt",
                           regul_annealing_rate=0.5, coeff_W_L1=1.0, coeff_g_phi_diff=1.0)
        mc = GraphModelConfig(signal_model_name="flyvis_A", prediction="first_derivative",
                              input_size=3, n_layers=2, hidden_dim=16, output_size=1,
                              input_size_update=5, n_layers_update=2, hidden_dim_update=16,
                              output_size_update=1, aggr_type="add", embedding_dim=2,
                              update_type="generic")
        r = LossRegularizer(train_config=t, model_config=mc, activity_column=3,
                            plot_frequency=1, n_neurons=16, trainer_type="flyvis",
                            n_neuron_types=0)
        r.set_epoch(3)
        # g_phi_diff is not annealed -> exactly sqrt(4) = 2
        assert r._coeffs["g_phi_diff"] == pytest.approx(2.0)
        # W_L1 is annealed -> anneal(1.0) * 2, and must be strictly between 0 and 2
        assert 0.0 < r._coeffs["W_L1"] < 2.0


class TestMutualExclusionWithMean:
    """'sqrt' and fit_reduction 'mean' fix the same coupling; both is a 2x error."""

    @pytest.mark.parametrize("scaling,reduction", [
        ("none", "norm2"), ("none", "mean"), ("sqrt", "norm2"),
    ])
    def test_valid_combinations_load(self, scaling, reduction):
        TrainingConfig(n_epochs=1, batch_size=4,
                       regul_batch_scaling=scaling, fit_reduction=reduction)

    def test_sqrt_with_mean_is_rejected_at_load(self):
        with pytest.raises(ValueError, match="double-corrects"):
            TrainingConfig(n_epochs=1, batch_size=4,
                           regul_batch_scaling="sqrt", fit_reduction="mean")

    def test_the_error_names_the_over_correction(self):
        """The message must say by how much, so the fix is obvious."""
        with pytest.raises(ValueError) as e:
            TrainingConfig(n_epochs=1, batch_size=16,
                           regul_batch_scaling="sqrt", fit_reduction="mean")
        assert "4.00" in str(e.value)      # sqrt(16)
