"""g_phi's input layout must mean the same thing in every model family.

flyvis_conductance's g_phi input was reordered from the interleaved

    [v_i, v_j, a_i, a_j]

to a PREFIX-EXTENSION of the non-conductance layout

    [v_j, a_j, v_i, a_i]        vs       [v_j, a_j]

so that columns 0..emb_dim are [v_j, a_j] in BOTH families. That invariant is what
makes a hardcoded column index safe. It was not safe before: the coeff_g_phi_norm
prior pins `in_features[:, 0] = 2*xnorm`, which meant v_j on flyvis_A (the intended
presynaptic anchor) but v_i on flyvis_conductance -- and since the probe leaves v_j
varying across neurons while demanding a constant output, it required g_phi to be
FLAT IN v_j, directly opposing coeff_g_phi_diff which requires dg/dv_j > 0.

These tests pin the invariant, the permutation-equivalence of the reorder, and the
column semantics that every consumer (regularizer groups, discard metrics, grad
ratios) depends on.
"""
import pytest
import torch

from connectome_gnn.metrics import g_phi_column_layout
from connectome_gnn.models.MLP import MLP
from connectome_gnn.utils import legacy_g_phi_column_permutation

pytestmark = pytest.mark.tier2

EMB = 2
N, E = 6, 10


@pytest.fixture(autouse=True)
def _cpu():
    torch.set_default_device("cpu")
    yield
    torch.set_default_device("cpu")


def _voltages_and_embeddings():
    v = torch.arange(1.0, N + 1).unsqueeze(1) * 10.0          # distinct per neuron
    emb = torch.arange(1.0, N * EMB + 1).reshape(N, EMB)
    src = torch.tensor([0, 1, 2, 3])                           # presynaptic  j
    dst = torch.tensor([2, 3, 4, 5])                           # postsynaptic i
    return v, emb, src, dst


def _current(v, emb, src, dst):
    """Exactly neural_gnn.NeuralGNN.message's conductance branch."""
    return torch.cat([v[src], emb[src], v[dst], emb[dst]], dim=1)


def _legacy(v, emb, src, dst):
    return torch.cat([v[dst], v[src], emb[dst], emb[src]], dim=1)


class TestLayoutInvariant:
    def test_column_zero_is_vj_in_both_families(self):
        v, emb, src, dst = _voltages_and_embeddings()
        cond = _current(v, emb, src, dst)
        flyA = torch.cat([v[src], emb[src]], dim=1)
        torch.testing.assert_close(cond[:, 0], v[src].squeeze(1))
        torch.testing.assert_close(flyA[:, 0], v[src].squeeze(1))

    def test_conductance_is_a_prefix_extension(self):
        """The non-conductance layout must be a literal prefix of the conductance one."""
        v, emb, src, dst = _voltages_and_embeddings()
        cond = _current(v, emb, src, dst)
        flyA = torch.cat([v[src], emb[src]], dim=1)
        torch.testing.assert_close(cond[:, : 1 + EMB], flyA)

    def test_slices_match_construction(self):
        """g_phi_column_layout's slices must name the columns the forward pass built."""
        v, emb, src, dst = _voltages_and_embeddings()
        cond = _current(v, emb, src, dst)

        class Fake(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.model = "flyvis_conductance"
                s.g_phi = MLP(input_size=2 + 2 * EMB, output_size=1, nlayers=2,
                              hidden_size=8, device="cpu")
                s.a = torch.zeros(N, EMB)

        layout, n_noise = g_phi_column_layout(Fake(), EMB)
        assert n_noise == 0
        torch.testing.assert_close(cond[:, layout["vj"]], v[src])
        torch.testing.assert_close(cond[:, layout["vi"]], v[dst])
        torch.testing.assert_close(cond[:, layout["aj"]], emb[src])
        torch.testing.assert_close(cond[:, layout["ai"]], emb[dst])


class TestLegacyPermutation:
    def test_permutation_maps_legacy_to_current(self):
        v, emb, src, dst = _voltages_and_embeddings()
        idx = legacy_g_phi_column_permutation(2 + 2 * EMB, EMB)
        torch.testing.assert_close(_legacy(v, emb, src, dst)[:, idx],
                                   _current(v, emb, src, dst))

    def test_permutation_is_a_bijection(self):
        for n_noise in (0, 2):
            idx = legacy_g_phi_column_permutation(2 + 2 * EMB + n_noise, EMB)
            assert sorted(idx) == list(range(2 + 2 * EMB + n_noise))

    def test_noise_columns_stay_trailing(self):
        idx = legacy_g_phi_column_permutation(2 + 2 * EMB + 2, EMB)
        base = 2 + 2 * EMB
        assert idx[base:] == [base, base + 1]

    def test_migrated_checkpoint_computes_identically(self):
        """A legacy g_phi + permuted first-layer weights must be bit-identical.

        This is what makes the reorder a pure relabelling: an old checkpoint stays
        usable, and no trained function is lost.
        """
        torch.manual_seed(0)
        width = 2 + 2 * EMB
        g_phi = MLP(input_size=width, output_size=1, nlayers=3, hidden_size=16, device="cpu")

        v, emb, src, dst = _voltages_and_embeddings()
        legacy_in = _legacy(v, emb, src, dst)
        current_in = _current(v, emb, src, dst)

        out_legacy = g_phi(legacy_in)                       # old code path

        idx = legacy_g_phi_column_permutation(width, EMB)
        with torch.no_grad():
            g_phi.layers[0].weight.copy_(g_phi.layers[0].weight[:, idx])
        out_current = g_phi(current_in)                     # new code path, migrated weights

        # Exact in real arithmetic — it is a relabelling of the same sum. Not
        # bit-identical in fp32 only because permuting the columns permutes the
        # accumulation order of the first layer's dot product (~1e-7 relative).
        torch.testing.assert_close(out_current, out_legacy, rtol=1e-5, atol=1e-6)

    def test_migrated_checkpoint_is_exact_in_float64(self):
        """The same check in float64: the reorder loses nothing but rounding."""
        torch.manual_seed(0)
        width = 2 + 2 * EMB
        g_phi = MLP(input_size=width, output_size=1, nlayers=3, hidden_size=16,
                    device="cpu").double()
        v, emb, src, dst = _voltages_and_embeddings()
        out_legacy = g_phi(_legacy(v, emb, src, dst).double())
        idx = legacy_g_phi_column_permutation(width, EMB)
        with torch.no_grad():
            g_phi.layers[0].weight.copy_(g_phi.layers[0].weight[:, idx])
        out_current = g_phi(_current(v, emb, src, dst).double())
        torch.testing.assert_close(out_current, out_legacy, rtol=1e-12, atol=1e-12)

    def test_rejects_too_narrow(self):
        with pytest.raises(ValueError):
            legacy_g_phi_column_permutation(1 + EMB, EMB)   # flyvis_A width


class TestNormAnchorNowTargetsVj:
    """The bug this reorder exists to kill.

    coeff_g_phi_norm pins in_features[:, 0]. Under the old layout that column was
    v_i on conductance while v_j kept varying, so the constant-output target forced
    g_phi flat in v_j. Under the new layout column 0 is v_j in both families, so the
    prior anchors the intended quantity and the remaining variation is across
    embeddings -- which is what a scale anchor should normalise over.
    """

    def test_pinning_column_zero_pins_vj_and_leaves_embeddings_varying(self):
        v, emb, src, dst = _voltages_and_embeddings()
        probe = _current(v, emb, src, dst).clone()
        probe[:, 0] = 2.0                                    # regularizer.py's anchor
        assert probe[:, 0].unique().numel() == 1, "v_j must be pinned"
        assert probe[:, 1:1 + EMB].unique().numel() > 1, "a_j must still vary"

    def test_legacy_layout_would_have_left_vj_varying(self):
        """Documents the old failure mode so a future reorder cannot silently restore it."""
        v, emb, src, dst = _voltages_and_embeddings()
        probe = _legacy(v, emb, src, dst).clone()
        probe[:, 0] = 2.0
        # column 1 was v_j and stayed varying -> constant-output target => flat in v_j
        assert probe[:, 1].unique().numel() > 1

    def test_anchor_must_pin_every_voltage_column(self):
        """The reorder alone is NOT sufficient; the anchor must pin v_i too.

        Pinning one voltage column while the other keeps per-neuron data values
        turns a gauge fix into "g_phi is flat along the unpinned axis", asserted
        over every scored row. Pre-reorder that axis was v_j (unsatisfiable for
        the true relu(v_j)); post-reorder it would be v_i -- satisfiable, but it
        is the discard hypothesis under test, so it would manufacture the result.
        Only pinning both leaves the model free along both axes.
        """
        v, emb, src, dst = _voltages_and_embeddings()

        class Fake(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.model = "flyvis_conductance"
                s.g_phi = MLP(input_size=2 + 2 * EMB, output_size=1, nlayers=2,
                              hidden_size=8, device="cpu")
                s.a = torch.zeros(N, EMB)

        layout, _ = g_phi_column_layout(Fake(), EMB)
        probe = _current(v, emb, src, dst).clone()
        for key in ("vj", "vi"):
            probe[:, layout[key]] = 2.0

        assert probe[:, layout["vj"]].unique().numel() == 1, "v_j must be pinned"
        assert probe[:, layout["vi"]].unique().numel() == 1, "v_i must be pinned too"
        # only the embeddings may still vary — that is what a scale anchor
        # legitimately normalises over
        assert probe[:, layout["aj"]].unique().numel() > 1
        assert probe[:, layout["ai"]].unique().numel() > 1
