"""The nested test between the two families, and what it refuses to claim."""
import numpy as np
from connectome_gnn.metrics import form_comparison_stats


def _diag(n=2000, gain=0.05, t=8.0, seed=0):
    r = np.random.default_rng(seed)
    cur = np.clip(1 - np.abs(r.normal(0, 0.01, n)), 0, 1)
    return {"_form_cur_r2_full": cur,
            "_form_cond_r2_full": np.minimum(cur + gain, 1.0),
            "_form_t_b2_full": np.full(n, t),
            "_form_n_used_full": np.full(n, 400.0)}


def test_the_bonferroni_threshold_grows_with_the_number_of_edges():
    """A claim about the run is a claim across every test it ran.

    t >= 3 is the readout's per-edge gate; asserting the families differ ON THIS
    RUN means surviving one threshold applied 434,112 times, which is a larger
    number, and it has to move when the edge count does.
    """
    small, _ = form_comparison_stats(_diag(n=100))
    big, _ = form_comparison_stats(_diag(n=400000))
    assert big["t_bonferroni"] > small["t_bonferroni"] > 3.0


def test_a_form_that_buys_nothing_rejects_nowhere():
    stats, lines = form_comparison_stats(_diag(gain=0.0, t=0.5))
    assert stats["gain_q50"] == 0.0
    assert stats["pct_t3"] == 0.0 and stats["pct_bonferroni"] == 0.0


def test_the_caveat_travels_with_the_number():
    """The share of rejecting edges assumes independent frames; they are
    consecutive samples of one trajectory, so the line that reports the share
    has to report what it rests on."""
    _, lines = form_comparison_stats(_diag())
    text = " ".join(lines)
    assert "frames per edge" in text and "optimistic" in text
