"""metrics.cluster_recovery_variants: the clusterability figure's comparison,
scored in the plot pass under clustering_<variant>_<stat>."""
import numpy as np

from connectome_gnn.metrics import CLUSTERING_VARIANTS, cluster_recovery, cluster_recovery_variants


def _toy(seed=0, n=240, n_types=8, n_edges=2400):
    rng = np.random.default_rng(seed)
    types = np.repeat(np.arange(n_types), n // n_types)
    edges = np.stack([rng.integers(0, n, n_edges), rng.integers(0, n, n_edges)])
    W = rng.normal(size=n_edges)
    tau = 0.05 + 0.03 * types + 0.002 * rng.normal(size=n)
    V = 0.1 * types + 0.01 * rng.normal(size=n)
    emb = np.column_stack([types, -types]).astype(float) + 0.01 * rng.normal(size=(n, 2))
    return types, edges, W, tau, V, emb


def test_each_variant_is_cluster_recovery_on_its_subset():
    types, edges, W, tau, V, emb = _toy()
    out = cluster_recovery_variants(types, edges, len(types), learned_W=W, learned_tau=tau,
                                    learned_vrest=V, embedding=emb, gt_W=W, gt_tau=tau,
                                    gt_vrest=V, n_components=16)
    assert {k.split("_")[1] for k in out} == set(CLUSTERING_VARIANTS)
    emb_only = cluster_recovery(types, edges, None, len(types), embedding=emb, n_components=16)
    assert out["clustering_emb_accuracy"] == emb_only["clustering_accuracy"]
    assert out["clustering_emb_n_features"] == 2
    params = cluster_recovery(types, edges, W, len(types), learned_tau=tau, learned_vrest=V,
                              n_components=16)
    assert out["clustering_params_accuracy"] == params["clustering_accuracy"]
    assert out["clustering_params_n_features"] == 10          # tau, V_rest, 8 W statistics


def test_a_variant_without_inputs_is_left_out():
    types, edges, W, tau, V, emb = _toy()
    out = cluster_recovery_variants(types, edges, len(types), learned_W=W, learned_tau=tau,
                                    n_components=16)
    assert all(k.startswith("clustering_params_") for k in out)
