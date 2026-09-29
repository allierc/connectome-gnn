"""The template readout streamed over edges: same numbers, bounded memory.

extract_template_params used to hold (edges x frames) arrays for every edge at
once and ran 80-95 GB cards out of memory on the 1.3-9.6 M-edge FlyWire graphs
(experiment 6). It now sums a block of edges at a time. These tests pin the
three things that change could break: the two global quantiles it now computes
without the full array, independence from the block size, and the second pass
adding each short edge's OWN frames (it used to add another edge's).
"""
import numpy as np
import pytest
import torch

import connectome_gnn.metrics as metrics
from connectome_gnn.metrics import _pooled_quantile, extract_template_params
from test_template_recovery import _OP, _XTS, E_EXC, E_INH, K, N, _Cfg, _Model


# --------------------------------------------------------------------------- #
# _pooled_quantile == np.quantile on the array it stands for, to dtype precision
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("q", [0.0, 0.5, 0.99, 1.0, 0.3141])
def test_pooled_quantile_is_numpys_on_the_expanded_array(dtype, positive, q):
    rng = np.random.default_rng(7)
    # Ties on purpose: rounding to 0.25 V makes many equal values, which is
    # where an off-by-one in the cumulative weights would show.
    values = (np.round(rng.normal(size=(9, 13)) * 4) / 4).astype(dtype)
    weights = rng.integers(0, 5, size=13)            # zeros included
    expanded = np.concatenate([np.repeat(values[:, n], weights[n])
                               for n in range(13)])
    if positive:
        expanded = expanded[expanded > 0]
    got = _pooled_quantile(values, weights, q, positive=positive)
    expected = float(np.quantile(expanded, q))
    assert got == pytest.approx(expected, abs=64 * np.finfo(dtype).eps)


def test_pooled_quantile_percentile_form_matches_np_percentile():
    rng = np.random.default_rng(3)
    values = np.abs(rng.normal(size=(40, 25))).astype(np.float32)
    weights = rng.integers(0, 9, size=25)
    expanded = np.concatenate([np.repeat(values[:, n], weights[n]) for n in range(25)])
    assert (_pooled_quantile(values, weights, 99, percentile=True)
            == float(np.percentile(expanded, 99)))


def test_pooled_quantile_of_nothing_is_none():
    assert _pooled_quantile(np.zeros((3, 4)), np.ones(4, int), 0.5, positive=True) is None
    assert _pooled_quantile(np.ones((3, 4)), np.zeros(4, int), 0.5) is None


# --------------------------------------------------------------------------- #
# the streamed readout
# --------------------------------------------------------------------------- #
def _big_fixture(seed=0, n_e=40, T=400):
    torch.manual_seed(seed)
    src = torch.randint(0, N, (n_e,))
    dst = torch.randint(0, N, (n_e,))
    edges = torch.stack([src, dst])
    w_true = torch.rand(n_e) + 0.2
    e_per_neuron = torch.where(torch.arange(N) % 2 == 0,
                               torch.tensor(E_EXC), torch.tensor(E_INH))
    model = _Model(w_true / K, e_per_neuron)
    x_ts = _XTS(torch.randn(T, N) * 2.0)
    return _OP(w_true, edges, e_per_neuron), model, edges, x_ts, w_true


class _CfgUniform(_Cfg):
    class recovery:
        template_frame_choice = "uniform"
        template_second_pass_frames = 64


def _readout(model, op, cfg, edges, x_ts, **kw):
    return extract_template_params(model, op, config=cfg, edges=edges, x_ts=x_ts,
                                   device="cpu", n_neurons=N, gauge_tau="true", **kw)


def _same(a, b):
    """Every pair and every numeric diagnostic of two readouts, bit for bit."""
    assert a.pairs.keys() == b.pairs.keys()
    for k in a.pairs:
        pa, pb = a.pairs[k], b.pairs[k]
        if pa is None or pb is None:
            assert pa is pb, k
            continue
        for x, y in zip(pa, pb):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=k)
    for k, v in a.diagnostics.items():
        if isinstance(v, (int, float, np.ndarray, np.floating)):
            np.testing.assert_array_equal(np.asarray(v), np.asarray(b.diagnostics[k]),
                                          err_msg=k)


@pytest.mark.parametrize("cfg", [_Cfg, _CfgUniform])
def test_the_block_size_does_not_move_a_single_number(monkeypatch, cfg):
    """Every sum is per edge, so cutting the edges into blocks of 7 (edge, frame)
    pairs -- one edge per block -- or of a million gives the same readout."""
    op, model, edges, x_ts, _ = _big_fixture()
    kw = dict(n_frames=32, min_points=12)
    monkeypatch.setattr(metrics, "TEMPLATE_BLOCK_PAIRS", 1 << 20)
    whole = _readout(model, op, cfg(), edges, x_ts, **kw)
    monkeypatch.setattr(metrics, "TEMPLATE_BLOCK_PAIRS", 7)
    blocked = _readout(model, op, cfg(), edges, x_ts, **kw)
    _same(whole, blocked)


def test_the_second_pass_adds_each_short_edges_own_frames():
    """The message IS the template, so any edge fitted on its own frames comes
    back exact. At 16 first-pass frames and a 12-row minimum nearly every edge is
    short; the second pass must rescue them WITH THEIR OWN data. Before the fix
    each chunk re-permuted the edges and a short edge was summed with another
    edge's frames: W survived here only because the message was scaled by the
    short edge's own W, while E came back as the OTHER edge's sender's reversal
    (8 of 26 fitted edges off by more than 0.01, against 0 of 40 now)."""
    op, model, edges, x_ts, w_true = _big_fixture()
    rec = _readout(model, op, _CfgUniform(), edges, x_ts, n_frames=16, min_points=12)
    assert rec.diagnostics["tmpl_pct_unfitted_first_pass"] > 50.0
    assert rec.diagnostics["tmpl_pct_rescued_second_pass"] > 25.0
    gt_w, learned_w = rec.get("W")
    ok = np.isfinite(learned_w)
    assert ok.mean() > 0.75
    np.testing.assert_allclose(learned_w[ok], gt_w[ok], rtol=1e-3)
    gt_e, learned_e = rec.pairs["E_ij"]
    ok_e = np.isfinite(learned_e)
    assert ok_e.mean() > 0.75
    np.testing.assert_allclose(learned_e[ok_e], gt_e[ok_e], atol=1e-3)
