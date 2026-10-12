"""The score mask (config.recovery.score_mask): scoring restricted to kept edges and
neurons, the readout itself untouched. Small synthetic case, CPU, no data files."""
import numpy as np
import pytest

from connectome_gnn.config import RecoveryConfig
from connectome_gnn.metrics import (RecoveredParams, _pair, cluster_recovery,
                                    cluster_recovery_variants, load_score_mask,
                                    restrict_recovered, score_recovery)


def _rec(seed=0, n=200, n_edges=1500):
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=n_edges)
    w_learned = w_true + 0.1 * rng.normal(size=n_edges)
    w_learned[rng.choice(n_edges, 300, replace=False)] = np.nan   # unfitted edges
    w_learned[:40] += 5.0                                           # a few outliers
    tau_true = 0.05 + 0.1 * rng.random(n)
    tau_learned = tau_true + 0.01 * rng.normal(size=n)
    tau_learned[5] = np.nan
    v_true = rng.normal(size=n)
    v_learned = v_true + 0.05 * rng.normal(size=n)
    msg_true = rng.normal(size=3 * n)
    rec = RecoveredParams()
    rec.pairs["W"] = _pair(w_true, w_learned)
    rec.pairs["tau"] = _pair(tau_true, tau_learned)
    rec.pairs["V_rest"] = _pair(v_true, v_learned)
    rec.pairs["V_rest_uncorrected"] = (v_true, v_learned - 0.1)     # a plain tuple, full length
    rec.pairs["msg_i"] = _pair(msg_true, msg_true + 0.1)
    rec.estimator["W"] = "template_fit"
    rec.diagnostics.update(readout="template", tmpl_pct_unfitted=20.0,
                           _W_learned_full=w_learned)
    return rec, dict(w_true=w_true, w_learned=w_learned, tau_true=tau_true,
                     tau_learned=tau_learned, v_true=v_true, v_learned=v_learned)


def test_pair_is_a_tuple_that_knows_its_positions():
    gt, learned = np.arange(6.0), np.array([0.0, np.nan, 2.0, 3.0, np.inf, 5.0])
    p = _pair(gt, learned)
    g, l = p
    assert isinstance(p, tuple) and len(p) == 2
    np.testing.assert_array_equal(p.idx, [0, 2, 3, 5])
    assert p.n_full == 6
    np.testing.assert_array_equal(g, gt[p.idx])


def test_no_mask_scores_bit_for_bit():
    rec, _ = _rec()
    a = score_recovery(rec)
    b = score_recovery(restrict_recovered(rec))
    for k, v in b.items():
        assert a[k] == v or (v != v and a[k] != a[k]), k


def test_mask_scores_exactly_the_kept_finite_entries():
    rec, d = _rec()
    rng = np.random.default_rng(1)
    keep_e = rng.random(d["w_true"].size) > 0.4
    keep_n = rng.random(d["tau_true"].size) > 0.2
    out = score_recovery(restrict_recovered(rec, keep_edges=keep_e, keep_neurons=keep_n))

    def ref(gt, learned, keep, key):
        ok = keep & np.isfinite(gt) & np.isfinite(learned)
        r = RecoveredParams()
        r.pairs[key] = (gt[ok], learned[ok])
        return score_recovery(r)

    w = ref(d["w_true"], d["w_learned"], keep_e, "W")
    t = ref(d["tau_true"], d["tau_learned"], keep_n, "tau")
    v = ref(d["v_true"], d["v_learned"], keep_n, "V_rest")
    for k in ("Wij_R2", "Wij_R2_all", "Wij_slope", "Wij_n", "Wij_n_outliers", "Wij_pct_outliers"):
        assert out[k] == w[k], k
    for k in ("tau_R2", "tau_n", "tau_pct_outliers"):
        assert out[k] == t[k], k
    for k in ("V_rest_R2", "V_rest_n"):
        assert out[k] == v[k], k
    assert out["Wij_n"] == int((keep_e & np.isfinite(d["w_learned"])).sum())
    # msg_i is not a per-edge or per-neuron quantity: left as it is
    assert out["msg_i_R2"] == score_recovery(rec)["msg_i_R2"]
    # the readout's scalar diagnostics describe every edge: not carried over
    assert "tmpl_pct_unfitted" not in out and out["readout"] == "template"


def test_a_mask_of_the_wrong_length_raises():
    rec, d = _rec()
    with pytest.raises(ValueError):
        restrict_recovered(rec, keep_edges=np.ones(d["w_true"].size + 1, bool))


def test_load_score_mask_picks_the_lattice_by_length(tmp_path):
    p = tmp_path / "mask.npz"
    np.savez(p, keep_edges_217=np.r_[True, False, True], keep_edges_721=np.ones(5, bool),
             keep_neurons_217=np.r_[True, False], readme="x")
    ke, kn = load_score_mask(str(p), 3, 2)
    np.testing.assert_array_equal(ke, [True, False, True])
    np.testing.assert_array_equal(kn, [True, False])
    with pytest.raises(ValueError):
        load_score_mask(str(p), 4, 2)          # no keep_edges* of length 4
    with pytest.raises(FileNotFoundError):
        load_score_mask(str(tmp_path / "none.npz"), 3, 2)


def _toy(seed=0, n=240, n_types=8, n_edges=2400):
    rng = np.random.default_rng(seed)
    types = np.repeat(np.arange(n_types), n // n_types)
    edges = np.stack([rng.integers(0, n, n_edges), rng.integers(0, n, n_edges)])
    W = rng.normal(size=n_edges)
    tau = 0.05 + 0.03 * types + 0.002 * rng.normal(size=n)
    V = 0.1 * types + 0.01 * rng.normal(size=n)
    return types, edges, W, tau, V


def test_cluster_recovery_mask():
    types, edges, W, tau, V = _toy()
    n = len(types)
    plain = cluster_recovery(types, edges, W, n, learned_tau=tau, learned_vrest=V,
                             n_components=16, return_features=True)
    none = cluster_recovery(types, edges, W, n, learned_tau=tau, learned_vrest=V,
                            n_components=16, return_features=True, keep_edges=None,
                            keep_neurons=None)
    np.testing.assert_array_equal(plain["_X"], none["_X"])
    assert plain["clustering_accuracy"] == none["clustering_accuracy"]

    rng = np.random.default_rng(2)
    ke, kn = rng.random(edges.shape[1]) > 0.5, rng.random(n) > 0.25
    m = cluster_recovery(types, edges, W, n, learned_tau=tau, learned_vrest=V,
                         n_components=16, return_features=True, keep_edges=ke, keep_neurons=kn)
    # the same as clustering the kept neurons on the stack built from the kept edges
    ref = cluster_recovery(types, edges[:, ke], W[ke], n, learned_tau=tau, learned_vrest=V,
                           n_components=16, return_features=True)
    assert m["_X"].shape[0] == kn.sum()
    np.testing.assert_array_equal(m["_X"], ref["_X"][kn])
    v = cluster_recovery_variants(types, edges, n, learned_W=W, learned_tau=tau, learned_vrest=V,
                                  gt_W=W, gt_tau=tau, gt_vrest=V, n_components=16,
                                  keep_edges=ke, keep_neurons=kn)
    assert v["clustering_params_accuracy"] == m["clustering_accuracy"]


def test_config_default_is_none():
    assert RecoveryConfig().score_mask is None
    assert RecoveryConfig(score_mask="flyvis/silent_mask_000_epe5.3.npz").score_mask.endswith(".npz")
