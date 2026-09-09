"""Contract tests for the single circuit-parameter extraction entry point."""

import numpy as np
import pytest

from connectome_gnn.metrics import (
    RecoveredParams,
    _pair,
    resolve_W_estimator,
    score_recovery,
)


class _Cfg:
    class recovery:
        W_mode = "auto"
        W_outlier_thresh = 1.0
        tau_outlier_thresh = 0.1
        V_rest_outlier_thresh = 0.2
        gate_fit_r2 = 0.9


class _Model:
    def __init__(self, family):
        self.MODEL_FAMILY = family


class _Ode:
    def __init__(self, conductance):
        self.E_exc = np.zeros(3) if conductance else None


@pytest.mark.parametrize("family,conductance,expected", [
    ("linear", False, "direct"),
    ("linear", True, "direct"),
    ("mlp", False, "jacobian"),
    ("gnn", False, "gain_corrected"),
    # A GNN on conductance data takes the line fit: it matches the generative
    # form, where the gain correction collapses a postsynaptic quantity onto the
    # presynaptic neuron.
    ("gnn", True, "edge_line_fit"),
])
def test_auto_resolves_on_family_and_data(family, conductance, expected):
    assert resolve_W_estimator(_Model(family), _Ode(conductance), _Cfg) == expected


def test_explicit_mode_overrides_auto():
    class Pinned(_Cfg):
        class recovery(_Cfg.recovery):
            W_mode = "direct"
    assert resolve_W_estimator(_Model("gnn"), _Ode(True), Pinned) == "direct"


def test_absent_quantity_is_absent_not_zeros():
    """The ladder this replaces fell through to np.zeros(n_neurons), which reached
    the analysis log looking exactly like a measurement of zero."""
    rec = RecoveredParams()
    assert rec.get("tau") is None
    assert "tau" not in rec
    assert not any(k.startswith("tau_") for k in score_recovery(rec, _Cfg))


def test_gated_out_quantity_reads_as_absent():
    rec = RecoveredParams(pairs={"W": (np.arange(5.), np.arange(5.))},
                          valid={"W": False})
    assert rec.get("W") is None
    assert not any(k.startswith("Wij_") for k in score_recovery(rec, _Cfg))


def test_pair_drops_non_finite_and_refuses_degenerate():
    gt, learned = _pair([1., 2., np.nan, 4.], [1., 2., 3., 4.])
    assert gt.size == learned.size == 3
    assert _pair([1.], [1.]) is None
    assert _pair(None, [1., 2.]) is None


def test_headline_is_outlier_free_and_all_is_not():
    """The bug this fixes: the trainer reported the filtered number and test_plot
    the unfiltered one, under similar names, differing by a factor of 30."""
    gt = np.zeros(100)
    learned = np.zeros(100)
    learned[0] = 50.0                      # one edge far outside the 1.0 band
    out = score_recovery(
        RecoveredParams(pairs={"W": (gt + np.arange(100) * 0.01,
                                     learned + np.arange(100) * 0.01)},
                        estimator={"W": "direct"}), _Cfg)
    assert out["Wij_n_outliers"] == 1
    assert out["Wij_R2"] > out["Wij_R2_all"]
    assert out["Wij_estimator"] == "direct"


def test_no_threshold_quantities_emit_no_outlier_keys():
    """E_ij and msg_i have no published tolerance band, so inventing one would
    decide by fiat which edges count."""
    out = score_recovery(
        RecoveredParams(pairs={"msg_i": (np.arange(10.), np.arange(10.))}), _Cfg)
    assert "msg_i_R2" in out
    assert "msg_i_n_outliers" not in out
    assert "msg_i_R2_all" not in out
