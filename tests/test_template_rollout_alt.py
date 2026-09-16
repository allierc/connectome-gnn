"""The alternative-form view the second rollout is built from.

The rollout that answers "could the OTHER family's constants have produced this
trajectory" is only as trustworthy as the swap that builds it: put the other
form's W on the wrong edges and the rollout fails for a reason that has nothing
to do with the families.
"""

import numpy as np

from connectome_gnn.template_rollout import _alt_view, other_known_ode_name


class _Rec:
    def __init__(self):
        self.pairs = {"W": (np.arange(4.0), np.array([1.0, 2.0, 3.0, 4.0])),
                      "tau": (np.ones(3), np.ones(3))}
        self.diagnostics = {"_W_learned_full": np.array([1.0, 2.0, 3.0, 4.0]),
                            "_tmpl_E_full": np.array([-5.0, -5.0, 10.0, 10.0]),
                            "_W_alt_full": np.array([9.0, np.nan, 7.0, 6.0]),
                            "_E_alt_full": np.array([np.nan] * 4)}
        self.estimator, self.correction, self.valid = {}, {}, {}

    def get(self, k):
        return self.pairs.get(k)


class _Cfg:
    class graph_model:
        signal_model_name = "flyvis_conductance"


def test_the_alt_view_swaps_only_what_the_family_changes():
    rec = _Rec()
    alt = _alt_view(rec)
    assert np.array_equal(alt.diagnostics["_W_learned_full"],
                          rec.diagnostics["_W_alt_full"], equal_nan=True)
    # tau is the update fit's, which does not depend on which form the per-edge
    # message was read with, so it must come through untouched.
    assert alt.pairs["tau"] is rec.pairs["tau"]
    # The caller's rec keeps its own W: the view must not write back, or the
    # first rollout's checkpoint would be rebuilt from the second one's numbers.
    assert np.array_equal(rec.diagnostics["_W_learned_full"],
                          np.array([1.0, 2.0, 3.0, 4.0]))


def test_the_alt_pair_drops_the_edges_the_alt_fit_did_not_fit():
    alt = _alt_view(_Rec())
    gt, learned = alt.pairs["W"]
    assert learned.size == 3 and np.all(np.isfinite(learned))
    assert gt.size == learned.size


def test_the_other_family_is_the_other_known_ode():
    assert other_known_ode_name(_Cfg()) == "flyvis_known_ode"

    class _Cur(_Cfg):
        class graph_model:
            signal_model_name = "flyvis_current"
    assert other_known_ode_name(_Cur()) == "flyvis_conductance_known_ode"
