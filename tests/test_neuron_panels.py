"""neuron_panels: the true forms follow the generator family, and the figure draws."""
import numpy as np, torch, os, tempfile, pytest
from connectome_gnn.neuron_panels import true_forms, plot_neuron_panels, symbolic_forms


class _OP:
    def __init__(self, conductance):
        n, E = 6, 12
        self.W = torch.rand(E) if conductance else torch.randn(E)
        self.edge_index = torch.stack([torch.randint(0, n, (E,)), torch.zeros(E, dtype=torch.long)])
        self.edge_is_inh = torch.rand(E) > 0.5
        if conductance:
            self.E_exc = torch.full((n,), 10.37); self.E_inh = torch.full((n,), -5.25)
    def gt_tau(self, n): return np.full(n, 0.09)
    def gt_vrest(self, n): return np.full(n, 2.91)
    def gt_g_phi_func(self, v): return np.maximum(v, 0.0)
    def reversal_per_edge(self):
        return torch.where(self.edge_is_inh, self.E_inh[self.edge_index[1]], self.E_exc[self.edge_index[1]])


def test_conductance_form_carries_the_reversal():
    f = true_forms(_OP(True), 0, [0, 1], 6)
    assert "relu(v_j)" in f["edges"][0] and "- v_i" in f["edges"][0]
    assert f["conductance"] and "10.37" in f["edges"][0] or "-5.25" in f["edges"][0]
    assert "tau = 0.0900" in f["update"]


def test_current_form_has_no_driving_force():
    f = true_forms(_OP(False), 0, [0, 1], 6)
    assert "relu(v_j)" in f["edges"][0] and "v_i" not in f["edges"][0]
    assert not f["conductance"]


def test_symbolic_is_skipped_when_disabled():
    class C: sr_enabled = False
    out = symbolic_forms({}, C())
    assert out["update"] is None and out["edges"] == {}


def test_figure_is_written_with_and_without_a_rollout(tmp_path):
    n_t, n_e = 50, 3
    rng = np.random.default_rng(0)
    op = _OP(True)
    g = {"frames": np.arange(n_t), "v_i": rng.normal(size=n_t), "stim": rng.normal(size=n_t),
         "msg_model": rng.normal(size=n_t), "pred": rng.normal(size=n_t),
         "m_true": rng.normal(size=(n_e, n_t)), "m_model": rng.normal(size=(n_e, n_t)),
         "edge_ids": np.arange(n_e), "src": np.arange(n_e),
         "forms": true_forms(op, 0, np.arange(n_e), 6),
         "W_model": np.ones(n_e), "v_j": rng.normal(size=(n_e, n_t))}
    sr = {"update": "(2.91 - v_i + msg + stim) * 11.1", "update_note": None,
          "edges": {0: "0.5 * relu(v_j)"}, "edge_notes": {}}
    for roll in (None, (rng.normal(size=n_t), rng.normal(size=n_t))):
        p = plot_neuron_panels(g, sr, 0, str(tmp_path), rollout=roll)
        assert os.path.getsize(p) > 10000


def test_panels_never_mix_two_trajectories():
    """If the test split cannot be matched to the rollout bundle, the free run is
    dropped rather than drawn from a different split than the other panels."""
    from connectome_gnn.neuron_panels import _test_split

    class _X:
        n_frames = 7208
    class _Data:
        x_ts = _X()
    # no bundle -> nothing to align against
    x, ok = _test_split(None, _Data(), None)
    assert x is _Data.x_ts and ok is False
    # a bundle whose length cannot be matched -> falls back, not aligned
    x, ok = _test_split(None, _Data(), {"activity_true": np.zeros((3, 999))})
    assert ok is False
