"""extract_template_params: the generator's form, fitted back out of a model.

A model whose per-edge message IS W * relu(v_j) * (E - v_i) must come back with
that W and that E exactly, gauge included -- if the closed-form fit cannot
recover the form it was handed, nothing it says about a trained GNN is worth
reading.
"""
import numpy as np
import pytest
import torch

from connectome_gnn.metrics import extract_template_params, score_recovery

N, EMB, T = 6, 2, 60
E_INH, E_EXC = -5.25, 10.37   # the two reversals, one per presynaptic neuron
TAU = 0.09             # seconds, the generator's membrane time constant
V_REST = 2.91          # the voltage the generator relaxes to
K = 0.5                # the gauge: one unit of model message is K of a true one
DFDMSG = K / TAU       # so that k = tau * dftheta_dmsg = K
G_MODEL = 1.0          # the weight the model's update puts on its own message
T_MODEL = DFDMSG / G_MODEL          # so that T * G is the derivative above


class _GPhi(torch.nn.Module):
    """relu(v_j) * (E_j - v_i) from the conductance layout [v_j, a_j, v_i, a_i].

    The reversal rides on the PRESYNAPTIC embedding, as it does in the generator:
    the transmitter the sending cell releases picks the channel, so every edge
    out of neuron j shares one E and the fit has two distinct values to recover
    rather than one constant.
    """
    input_width = 2 + 2 * EMB

    def forward(self, x):
        v_j, v_i, e_j = x[:, 0], x[:, 1 + EMB], x[:, 1]
        return (torch.relu(v_j) * (e_j - v_i)).unsqueeze(1)


class _FTheta(torch.nn.Module):
    """The update the template fits: T * ((V - v) + G * msg), from [v, a, msg, stim]."""
    def forward(self, x):
        v, msg = x[:, 0:1], x[:, 1 + EMB:2 + EMB]
        return T_MODEL * ((V_REST - v) + G_MODEL * msg)


class _Model(torch.nn.Module):
    def __init__(self, w_model, e_per_neuron):
        super().__init__()
        self.a = torch.stack([e_per_neuron, torch.zeros(N)], dim=1)
        self.W = w_model
        self.w_squared = False
        self.g_phi = _GPhi()
        self.f_theta = _FTheta()

    def forward(self, st, edges, data_id=None, return_all=False):
        """A miniature of the real thing: message per edge, summed onto the
        postsynaptic neuron, then the update read off it."""
        v = (st[:, :1] if st.ndim == 2 else st.reshape(-1, 1)).float()
        src, dst = edges[0], edges[1]
        cols = torch.cat([v[src], self.a[src], v[dst], self.a[dst]], dim=1)
        m_e = self.W[:, None] * self.g_phi(cols)
        msg = torch.zeros(N, 1).index_add_(0, dst, m_e)
        feats = torch.cat([v, self.a, msg, torch.zeros(N, 1)], dim=1)
        return self.f_theta(feats), feats, msg


class _OP:
    def __init__(self, w_true, edges, e_per_neuron):
        self.W = w_true
        self.edge_index = edges
        self.e_per_neuron = e_per_neuron
        self.E_exc = torch.full((N,), E_EXC)
        self.E_inh = torch.full((N,), E_INH)
        self.edge_is_inh = e_per_neuron[edges[0]] < 0

    def gt_tau(self, n):
        return np.full(n, TAU)

    def gt_vrest(self, n):
        return np.full(n, V_REST)

    def has_tau(self):
        return True

    def has_vrest(self):
        return True

    def derive_tau(self, slopes, n):
        """The base ode_params inversion: tau = 1 / -slope, clipped."""
        sl = np.asarray(slopes)[:n]
        return np.clip(np.where(sl != 0, 1.0 / -sl, 1.0), 0, 10)

    def derive_vrest(self, slopes, offsets, n):
        sl, off = np.asarray(slopes)[:n], np.asarray(offsets)[:n]
        return np.where(sl != 0, -off / sl, 0.0)

    def gt_g_phi_func(self, v):
        return np.maximum(v, 0.0)

    def reversal_per_edge(self):
        return self.e_per_neuron[self.edge_index[0]]

    def effective_true_weights(self, w, edges, n):
        return np.asarray(w).ravel()


class _XTS:
    def __init__(self, voltage):
        self.voltage = voltage
        self.n_frames = voltage.shape[0]

    def frame(self, k):
        return self.voltage[k].reshape(-1, 1)


class _Cfg:
    class graph_model:
        signal_model_name = "flyvis_conductance"
        g_phi_positive = False
        embedding_dim = EMB


def _fixture(seed=0):
    torch.manual_seed(seed)
    n_e = 10
    src = torch.randint(0, N, (n_e,))
    dst = torch.randint(0, N, (n_e,))
    edges = torch.stack([src, dst])
    w_true = torch.rand(n_e) + 0.2
    e_per_neuron = torch.where(torch.arange(N) % 2 == 0,
                               torch.tensor(E_EXC), torch.tensor(E_INH))
    # The model carries the same synapses in its own gauge: K times larger a
    # message, which f_theta divides back out. Recovering w_true means undoing
    # exactly that.
    model = _Model(w_true / K, e_per_neuron)
    x_ts = _XTS(torch.randn(T, N) * 2.0)
    return _Cfg(), _OP(w_true, edges, e_per_neuron), model, edges, x_ts, w_true


def test_template_recovers_the_conductance_and_the_reversal():
    cfg, op, model, edges, x_ts, w_true = _fixture()
    rec = extract_template_params(model, op, config=cfg, edges=edges, x_ts=x_ts,
                                  device="cpu", n_neurons=N, n_frames=T,
                                  gauge_tau="true", min_points=4)
    gt_w, learned_w = rec.get("W")
    assert np.allclose(learned_w, gt_w, rtol=1e-3), (gt_w, learned_w)
    gt_e, learned_e = rec.pairs["E_ij"]
    assert np.allclose(learned_e, gt_e, atol=1e-3)
    assert rec.diagnostics["msg_form_r2_median"] > 0.999
    assert abs(rec.diagnostics["tmpl_k_median"] - K) < 1e-5


def test_scores_carry_the_one_vocabulary():
    cfg, op, model, edges, x_ts, _ = _fixture()
    scored = score_recovery(extract_template_params(
        model, op, config=cfg, edges=edges, x_ts=x_ts, device="cpu", n_neurons=N,
        n_frames=T, gauge_tau="true", min_points=4), cfg)
    assert scored["Wij_R2"] > 0.999 and abs(scored["Wij_gain"] - 1.0) < 1e-3
    assert scored["Eij_R2"] > 0.999
    assert scored["Wij_estimator"] == "template_fit"


def test_an_ungauged_model_is_off_by_exactly_the_gauge():
    """With the gauge dropped the fitted W is K times too large -- the whole
    point of scaling by tau * dftheta_dmsg, and the failure mode a global gain
    hides by fitting one factor after the fact."""
    cfg, op, model, edges, x_ts, w_true = _fixture()
    rec = extract_template_params(model, op, config=cfg, edges=edges, x_ts=x_ts,
                                  device="cpu", n_neurons=N, n_frames=T,
                                  gauge_tau="true", min_points=4)
    _gt, learned = rec.get("W")
    raw = learned / rec.diagnostics["tmpl_k_median"]
    assert np.allclose(raw, np.asarray(w_true) / K, rtol=1e-3)


def test_the_update_template_gives_back_vrest_and_the_gauge():
    """The four columns that supply k_i also supply V_rest and tau, and the
    fitted T * G must agree with autograd's df_theta/dmsg -- they are the same
    number whenever the update is affine in the message, which is what makes the
    closed-form fit a stand-in for the PySR search."""
    cfg, op, model, edges, x_ts, _ = _fixture()
    rec = extract_template_params(model, op, config=cfg, edges=edges, x_ts=x_ts,
                                  device="cpu", n_neurons=N, n_frames=T,
                                  gauge_tau="true", min_points=4)
    _gt_v, learned_v = rec.pairs["V_rest"]
    assert np.allclose(learned_v, V_REST, atol=1e-3)
    _gt_t, learned_t = rec.pairs["tau"]
    assert np.allclose(learned_t, 1.0 / T_MODEL, rtol=1e-3)
    assert rec.diagnostics["update_form_r2_median"] > 0.999
    assert abs(rec.diagnostics["tmpl_dfdmsg_over_autograd"] - 1.0) < 1e-4
    assert abs(rec.diagnostics["tmpl_G_median"] - G_MODEL) < 1e-3


def test_a_pedestal_in_the_message_is_read_as_an_offset_not_as_conductance():
    """A model whose every edge adds a constant to its message must come back
    with the same W and E as one that does not, and with that constant reported.
    Without the third column the pedestal is fitted by tilting the driving force,
    which moves the reversal -- the failure this term exists to prevent."""
    cfg, op, model, edges, x_ts, w_true = _fixture()
    pedestal = 0.35

    class _Shifted(_GPhi):
        def forward(self, x):
            return super().forward(x) + pedestal

    model.g_phi = _Shifted()
    rec = extract_template_params(model, op, config=cfg, edges=edges, x_ts=x_ts,
                                  device="cpu", n_neurons=N, n_frames=T,
                                  gauge_tau="true", min_points=4)
    gt_w, learned_w = rec.get("W")
    assert np.allclose(learned_w, gt_w, rtol=2e-2), (gt_w, learned_w)
    _gt_e, learned_e = rec.pairs["E_ij"]
    assert np.allclose(learned_e, _gt_e, atol=2e-2)
    # The message the model sends is W_model * (g_phi + pedestal), so the
    # constant each edge adds is W_model * pedestal, and the readout reports it
    # in the generator's units, i.e. times the gauge K.
    expect = float(np.median(np.abs(K * to_np(model.W) * pedestal)))
    assert abs(rec.diagnostics["tmpl_offset_median"] - expect) / expect < 0.05


def to_np(t):
    return t.detach().numpy().ravel()


def test_both_families_forms_are_fitted_to_the_same_message():
    """A conductance message needs the driving-force column; a current one does not.

    The comparison the two medians exist for. `current_form_r2_median` fits
    W*relu(v_j) + C -- the conductance form with the u*v_i column deleted -- to
    the same per-edge message, so a model whose message the CURRENT form already
    explains has not demonstrated a learned driving force however well the
    conductance form fits it. Here the model really is conductance, so the
    deleted column has to cost R2.
    """
    cfg, op, model, edges, x_ts, _ = _fixture()
    rec = extract_template_params(model, op, config=cfg, edges=edges, x_ts=x_ts,
                                  device="cpu", n_neurons=N, n_frames=T,
                                  gauge_tau="true", min_points=4)
    d = rec.diagnostics
    assert d["conductance_form_r2_median"] > 0.999
    assert d["current_form_r2_median"] < 0.99
    assert d["driving_force_r2_gain_mean"] > 0.001
    # The reversal it needs sits where the generator put it, a few units from
    # the voltages the cells take -- not hundreds, which is the signature of a
    # driving-force column fitted to a message that has none.
    assert d["conductance_form_E_over_vi"] < 10.0


def test_a_current_message_costs_nothing_to_drop_the_driving_force():
    """The same model with relu(v_j) alone: both forms fit, the gain is ~0.

    The nested pair means the conductance form can never fit WORSE, so the test
    of 'is there a driving force here' is whether the extra column BUYS
    anything, not whether the fuller form fits.
    """
    cfg, op, model, edges, x_ts, _ = _fixture()

    class _CurrentGPhi(_GPhi):
        def forward(self, x):
            return torch.relu(x[:, 0]).unsqueeze(1)

    model.g_phi = _CurrentGPhi()
    rec = extract_template_params(model, op, config=cfg, edges=edges, x_ts=x_ts,
                                  device="cpu", n_neurons=N, n_frames=T,
                                  gauge_tau="true", min_points=4)
    d = rec.diagnostics
    assert d["current_form_r2_median"] > 0.999
    assert abs(d["driving_force_r2_gain_mean"]) < 1e-6
    # R2 CANNOT TELL THEM APART HERE -- the conductance form contains the
    # current one, so it ties -- and the reversal is what gives it away: to
    # explain a message with no driving force the fit pushes E far outside
    # every voltage in the data.
    assert d["conductance_form_r2_median"] > 0.999
    assert d["conductance_form_E_over_vi"] > 20.0


def test_a_current_form_component_biases_E_and_leaves_W_alone():
    """THE ONE ERROR THE READOUT CANNOT CORRECT, and the reason E carries an
    se beside it.

    The template fits [u, u*v_i, 1] with u = act(v_j). A component of the
    message proportional to u with no driving force in it, D*u, is collinear
    with the W*E term: both live in the first column. So it lands entirely in
    b1 and shifts E = -b1/b2 by D/W, while b2 = -W is untouched.

    This is not a gauge and no correction removes it -- b1 is one coefficient
    and cannot be split into W*E and D from a single edge's message.
    """
    rng = np.random.default_rng(0)
    W, E, D = 0.4, -3.0, 0.05
    u = np.abs(rng.normal(1.0, 0.4, 4000))
    vi = rng.normal(0.0, 1.0, 4000)

    def fit(y):
        A = np.column_stack([u, u * vi, np.ones_like(u)])
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        return -b[1], -b[0] / b[1]          # W_fit, E_fit

    W0, E0 = fit(W * u * (E - vi))
    W1, E1 = fit(W * u * (E - vi) + D * u)
    assert W0 == pytest.approx(W, rel=1e-9) and E0 == pytest.approx(E, rel=1e-9)
    # W survives the contamination; E moves by exactly D/W.
    assert W1 == pytest.approx(W, rel=1e-9)
    assert E1 == pytest.approx(E + D / W, rel=1e-9)
    assert abs(E1 - E) > 0.1


def test_the_gauge_cancels_in_E_but_not_in_W():
    """Why the document says E needs no correction: msg_hat = msg / k + C
    divides b1 and b2 by the same k, and E is their ratio. W is not a ratio and
    must be multiplied back by k."""
    rng = np.random.default_rng(1)
    W, E, k, C = 0.4, -3.0, 7.0, 1.3
    u = np.abs(rng.normal(1.0, 0.4, 4000))
    vi = rng.normal(0.0, 1.0, 4000)
    A = np.column_stack([u, u * vi, np.ones_like(u)])
    b, *_ = np.linalg.lstsq(A, (W * u * (E - vi)) / k + C, rcond=None)
    assert -b[0] / b[1] == pytest.approx(E, rel=1e-9)       # E: k cancelled
    assert -b[1] == pytest.approx(W / k, rel=1e-9)          # W: still in 1/k units
    assert k * -b[1] == pytest.approx(W, rel=1e-9)          # ... until multiplied back


def test_the_W_constraint_belongs_to_the_conductance_form_only():
    """The two generators put the sign in different places:

        conductance   W * relu(v_j) * (E - v_i),  W >= 0, sign in (E - v_i)
        current       W * relu(v_j),              W SIGNED

    so W >= 0 is part of the conductance equation and would be a falsehood
    imposed on the current one. The readout constrains the three-column
    conductance fit and leaves the two-column current fit free.
    """
    rng = np.random.default_rng(0)
    u = np.abs(rng.normal(1.0, 0.4, 4000))
    vi = rng.normal(0.0, 1.0, 4000)

    # A message that is genuinely CURRENT-form with an inhibitory (negative) W.
    y = -0.4 * u
    A2 = np.column_stack([u, np.ones_like(u)])
    c, *_ = np.linalg.lstsq(A2, y, rcond=None)
    assert c[0] == pytest.approx(-0.4, rel=1e-9)      # free: recovers the sign

    # The same message read through the CONSTRAINED conductance form. Its free
    # fit wants W < 0, so the constraint binds, W is clamped to 0 and the
    # solution collapses to the current form -- which is the precise sense in
    # which that edge shows no driving force.
    A3 = np.column_stack([u, u * vi, np.ones_like(u)])
    b, *_ = np.linalg.lstsq(A3, y, rcond=None)
    W_free = -b[1]
    binds = not (b[1] < 0)
    assert W_free <= 1e-9 or binds        # free W is not positive here
    W_nn = 0.0 if binds else W_free
    assert W_nn == pytest.approx(0.0, abs=1e-9)


def test_constrained_fit_equals_the_free_one_when_W_is_positive():
    """Where the constraint is slack the two coincide, so constraining costs
    nothing on an edge that really is conductance-shaped."""
    rng = np.random.default_rng(1)
    u = np.abs(rng.normal(1.0, 0.4, 4000))
    vi = rng.normal(0.0, 1.0, 4000)
    y = 0.4 * u * (-3.0 - vi)                          # W = +0.4 > 0
    A3 = np.column_stack([u, u * vi, np.ones_like(u)])
    b, *_ = np.linalg.lstsq(A3, y, rcond=None)
    assert b[1] < 0                                    # constraint slack
    assert -b[1] == pytest.approx(0.4, rel=1e-9)
    assert -b[0] / b[1] == pytest.approx(-3.0, rel=1e-9)
