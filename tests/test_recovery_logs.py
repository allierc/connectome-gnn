"""The one vocabulary: per-quantity training logs, metrics.txt lines, gain convention.

tmp_training/<key>.log, results/metrics.txt and the analysis log carry the same
`<key>_<stat>` names, produced by score_recovery and written by
recovery_log_append / write_recovery_metrics. These tests pin that contract.
"""
import os

import numpy as np
import pytest

from connectome_gnn.metrics import (
    RECOVERY_KEYS, RecoveredParams, metrics_lines, r2_up_to_scale,
    recovery_log_append, recovery_log_columns, score_recovery,
    training_log_append, training_log_last, training_log_read,
    write_recovery_metrics,
)


def _rec(n=200, seed=0):
    rng = np.random.default_rng(seed)
    rec = RecoveredParams()
    gt = rng.normal(size=n)
    rec.pairs["W"] = (gt, 2.0 * gt + 0.01 * rng.normal(size=n))   # learned = 2 x true
    rec.estimator["W"] = "direct"
    tau = rng.uniform(0.05, 0.3, size=n)
    rec.pairs["tau"] = (tau, tau + 0.001 * rng.normal(size=n))
    rec.pairs["V_rest"] = (rng.uniform(0, 1, size=n), rng.uniform(0, 1, size=n))
    e = rng.normal(size=n)
    rec.pairs["E_ij"] = (e, e)
    m = rng.normal(size=n)
    rec.pairs["msg_i"] = (m, 0.5 * m)
    return rec


def test_gain_convention_is_learned_equals_gain_times_true():
    rec = _rec()
    s = score_recovery(rec)
    assert s["Wij_gain"] == pytest.approx(2.0, abs=0.01)          # not 0.5
    assert s["msg_i_gain"] == pytest.approx(0.5, abs=1e-6)
    assert s["Wij_R2_scaled"] > 0.99 and s["msg_i_R2_scaled"] > 0.999
    f = r2_up_to_scale(rec.pairs["W"][0], rec.pairs["W"][1])
    assert f["gain"] == s["Wij_gain"] and f["scale"] == f["gain"]


def test_every_quantity_carries_rel_err_and_w_carries_structure():
    s = score_recovery(_rec())
    for key in ("Wij", "tau", "V_rest", "Eij", "msg_i"):
        assert f"{key}_rel_err_median" in s and f"{key}_rel_err_iqr" in s
    assert s["Wij_pearson"] == pytest.approx(1.0, abs=1e-3)
    assert s["Wij_zscored_R2"] == pytest.approx(1.0, abs=1e-3)
    assert "tau_gain" not in s and "Eij_pearson" not in s


def test_recovery_logs_one_file_per_quantity_same_names_as_metrics_txt(tmp_path):
    rec = _rec()
    s = score_recovery(rec)
    recovery_log_append(str(tmp_path), 17, s)
    recovery_log_append(str(tmp_path), 34, s)
    for key in ("Wij", "tau", "V_rest", "Eij", "msg_i"):
        d = training_log_read(str(tmp_path), key)
        assert d is not None and list(d["iteration"]) == [17, 34]
        for col in recovery_log_columns(key):
            assert col in d, col
        assert d[f"{key}_R2"][-1] == pytest.approx(s[f"{key}_R2"], abs=1e-6)
    assert training_log_read(str(tmp_path), "gain") is None      # absent quantity, no file
    # a column the quantity does not have is nan, not missing
    assert np.isnan(training_log_read(str(tmp_path), "tau")["tau_R2_scaled"]).all() \
        if "tau_R2_scaled" in training_log_read(str(tmp_path), "tau") else True
    lines = metrics_lines(s)
    names = {ln.split(":")[0] for ln in lines}
    for key in ("Wij", "tau", "V_rest", "Eij", "msg_i"):
        for col in recovery_log_columns(key):
            if col in s:
                assert col in names
    assert "Wij_estimator" in names


def test_write_recovery_metrics_writes_metrics_txt_and_log(tmp_path):
    s = score_recovery(_rec())
    s["clustering_accuracy"] = 0.42
    log = tmp_path / "analysis.log"
    with open(log, "w") as lf:
        write_recovery_metrics(s, str(tmp_path), log_file=lf)
    txt = open(tmp_path / "results" / "metrics.txt").read()
    assert "Wij_R2:" in txt and "Wij_gain:" in txt and "clustering_accuracy: 0.42" in txt
    assert open(log).read() == txt
    # readable back the way the LLM pipeline and cv_runner read it
    kv = dict(ln.split(": ", 1) for ln in txt.strip().split("\n"))
    assert float(kv["Wij_gain"]) == pytest.approx(s["Wij_gain"], abs=1e-6)
    assert kv["Wij_estimator"] == "direct"


def test_training_log_round_trip_and_last_row(tmp_path):
    training_log_append(str(tmp_path), "rollout", 1, {"rollout_r": 0.5, "rollout_n_diverged": 3})
    training_log_append(str(tmp_path), "rollout", 2, {"rollout_r": float("nan"), "rollout_n_diverged": 0})
    d = training_log_read(str(tmp_path), "rollout")
    assert d["rollout_r"][0] == 0.5 and np.isnan(d["rollout_r"][1])
    last = training_log_last(str(tmp_path), "rollout")
    assert last["iteration"] == 2 and last["rollout_n_diverged"] == 0
    header = open(tmp_path / "tmp_training" / "rollout.log").readline().strip()
    assert header == "iteration,rollout_r,rollout_n_diverged"


def test_no_legacy_names_anywhere_in_the_catalogue():
    s = score_recovery(_rec())
    legacy = {"connectivity_R2", "W_corrected_R2", "W_corrected_no_outliers_R2", "w_scale",
              "W_structure_r", "W_zscored_R2", "tau_no_outliers_R2", "Eij_n_edges", "raw_W_R2"}
    assert not legacy & set(s)
    assert set(RECOVERY_KEYS) >= {"Wij", "tau", "V_rest", "Eij", "msg_i"}


def test_sort_key_orders_intra_epoch_checkpoints():
    """The plot phase sorts checkpoints, so a name it cannot parse kills a run
    AFTER its training finished -- which is how forty known-ODE jobs trained to
    completion and wrote no results."""
    from connectome_gnn.utils import sort_key
    names = ["best_model_with_0_graphs_1_136532.pt",
             "best_model_with_0_graphs_0_68266.pt",
             "best_model_with_0_graphs_0.pt"]
    assert [sort_key(n) for n in sorted(names, key=sort_key)] == \
        [(0, 0, 0), (0, 0, 68266), (0, 1, 136532)]


def test_known_ode_models_route_to_the_linear_plotter():
    """Their W, tau and V_rest are direct parameters; the gnn plotter reads an
    embedding they do not have, and the failure surfaces only after training."""
    from connectome_gnn.models.known_ode import KnownODEBase, FlyvisKnownODE
    assert KnownODEBase.MODEL_FAMILY == "linear"
    assert FlyvisKnownODE.MODEL_FAMILY == "linear"


def _cfg(readout="template"):
    return type("C", (), {"recovery": type("R", (), {"readout": readout})()})()


def test_every_key_log_names_the_readout_that_wrote_it(tmp_path):
    """A `<key>.log` is the trajectory a sweep is read on, and `Wij_R2` means a
    different quantity under each readout -- per-edge least squares in the
    generator's units under the template, the model's raw weight times a measured
    gain under the chain. The descriptor is a `#` line so readers skip it."""
    from connectome_gnn.metrics import recovery_log_append, training_log_read
    scored = {"readout": "template", "Wij_R2": 0.95,
              "Wij_estimator": "template_fit", "Wij_correction": "per edge"}
    recovery_log_append(str(tmp_path), 1000, scored)
    recovery_log_append(str(tmp_path), 2000, scored)
    head = open(tmp_path / "tmp_training" / "Wij.log").readline()
    assert head.startswith("# readout=template")
    assert "Wij_estimator=template_fit" in head
    # Written once, and invisible to the reader.
    assert training_log_read(str(tmp_path), "Wij")["iteration"].tolist() == [1000, 2000]


def test_metrics_txt_leads_with_the_readout():
    """Above the first `Wij_*` line, because it qualifies every line under it."""
    from connectome_gnn.metrics import metrics_lines
    assert metrics_lines({"readout": "template", "Wij_R2": 0.9})[0] == "readout: template"


def test_template_readout_failure_is_fatal_unless_the_chain_was_requested():
    """The fallback used to keep the chain's numbers under the template's column
    names, mid-file, behind a logger.warning. A run may not change estimator
    halfway through; only `recovery.readout: chain` may choose the chain."""
    from connectome_gnn.metrics import ReadoutError, require_template_readout
    with pytest.raises(ReadoutError):
        require_template_readout(_cfg("template"), ValueError("boom"), "checkpoint 1")
    require_template_readout(_cfg("chain"), ValueError("boom"), "checkpoint 1")


def test_w_squared_init_preserves_the_effective_weight(tmp_path):
    """THE INIT IS A STATEMENT ABOUT THE EFFECTIVE WEIGHT. Under w_squared the
    SQUARE must land where w_init_mode intended; initialising the parameter
    itself there and squaring it puts the conductance six orders too low and
    cripples dL/dW = dL/dmsg * g_phi * 2W, so zero becomes absorbing."""
    import math
    import torch
    n_w = 434112
    torch.manual_seed(0)
    base = torch.randn(n_w) * (1.0 / math.sqrt(n_w))
    eff_plain = base.abs().median()
    eff_squared = (base.abs().sqrt() ** 2).median()
    assert eff_squared == pytest.approx(float(eff_plain), rel=1e-5)
    # ... and the gradient factor 2W is no longer negligible.
    assert 2 * base.abs().sqrt().median() > 20 * 2 * base.abs().median()


def test_w_L1_penalises_the_conductance_not_its_root():
    """An L1 on the raw parameter under w_squared is an L1 on sqrt(conductance),
    and it pushes W to exactly where dL/dW = 2W vanishes."""
    import torch

    class _M:
        w_squared = True
        W = torch.tensor([[0.3], [-0.4]])

    m = _M()
    w = m.W ** 2 if m.w_squared else m.W
    assert float(w.norm(1)) == pytest.approx(0.09 + 0.16)
    assert float(m.W.norm(1)) == pytest.approx(0.7)      # what it used to charge


def test_W_L2_penalises_the_conductance_not_its_root():
    """Same bug as W_L1, same fix: under w_squared the conductance is W**2, so a
    norm on the raw parameter is a norm on its root and its gradient carries the
    same vanishing 2W factor. Both coefficients are live in the gauge grid
    (7.5e-05 and 7.5e-07), so both were pulling W into the absorbing state."""
    import torch
    W = torch.tensor([[0.3], [-0.4]])
    assert float((W ** 2).norm(2)) == pytest.approx(0.183576, abs=1e-6)  # ||0.09, 0.16||
    assert float(W.norm(2)) == pytest.approx(0.5)                        # what it charged


def test_W_sign_is_refused_under_w_squared():
    """A sign-consistency penalty on a non-negative quantity is vacuous: it would
    report perfect Dale conformance while measuring nothing. Under w_squared the
    sign lives in g_phi, so the term has to be refused rather than applied."""
    import inspect

    from connectome_gnn.models import regularizer
    src = inspect.getsource(regularizer)
    i = src.index("coeff_W_sign is non-zero on a w_squared model")
    guard = src.rindex("if getattr(model, 'w_squared', False):", 0, i)
    assert "raise ValueError" in src[guard:i + 200]


def test_eij_log_carries_the_per_edge_form_diagnostics(tmp_path):
    """msg_form_r2 and the two-form medians come out of the SAME per-edge fit as
    E_ij and answer what its R2 cannot -- whether the message has the
    generator's shape, and whether the driving force does any work. They were
    computed every checkpoint and thrown away, reachable only from metrics.txt
    and so only for a run that finished."""
    from connectome_gnn.metrics import (
        recovery_log_append, recovery_log_columns, training_log_read)
    cols = recovery_log_columns("Eij")
    for c in ("msg_form_r2_median", "conductance_form_r2_median",
              "current_form_r2_median"):
        assert c in cols, c
    # Global keys, so NOT prefixed -- unlike every <key>_<stat> beside them.
    assert "Eij_msg_form_r2_median" not in cols
    scored = {"Eij_R2": 0.7, "msg_form_r2_median": 0.94,
              "conductance_form_r2_median": 0.94, "current_form_r2_median": 0.61}
    recovery_log_append(str(tmp_path), 1000, scored)
    d = training_log_read(str(tmp_path), "Eij")
    assert d["msg_form_r2_median"][-1] == pytest.approx(0.94)
    assert d["current_form_r2_median"][-1] == pytest.approx(0.61)


def test_other_quantities_keep_their_columns():
    """_EXTRA_GLOBAL is per key: adding Eij diagnostics must not shift Wij."""
    from connectome_gnn.metrics import recovery_log_columns
    assert recovery_log_columns("Wij")[-1] == "Wij_R2_uncorrected"
    assert all(c.startswith("tau_") for c in recovery_log_columns("tau"))


def test_euler_and_multi_substeps_are_the_two_integrators():
    """The generator uses exponential Euler; forward Euler at the same delta_t
    contracts only while z = (delta_t/tau)(1+G) < 2 and the network reaches 4.4.
    Sub-stepping divides z by M. M=2 is NOT enough (z=2.21); M=3 clears it."""
    import torch

    from connectome_gnn.models.substep import integrate_frame, substep_report

    class _X:
        def __init__(self):
            self.voltage = torch.ones(3)

    k = 4.2 / 0.019                       # a/tau at the stiffest neuron
    fwd = lambda st: (-k * st.voltage).unsqueeze(-1)  # noqa: E731

    x = _X()
    for _ in range(6):
        integrate_frame(x, fwd, 0.020, "euler")
    assert float(x.voltage.abs().max()) > 100          # diverges

    x = _X()
    for _ in range(6):
        integrate_frame(x, fwd, 0.020, "multi_substeps", 5)
    assert float(x.voltage.abs().max()) < 1e-6         # contracts

    assert "DIVERGES" in substep_report("euler", 5, 0.020)
    assert "contracts" in substep_report("multi_substeps", 5, 0.020)
    assert "DIVERGES" in substep_report("multi_substeps", 2, 0.020)


def test_multi_substeps_at_M1_is_euler():
    """A config saying multi_substeps with M=1 is saying two things; it takes the
    Euler path rather than erroring, and the stability line reports z at M=1."""
    import torch

    from connectome_gnn.models.substep import integrate_frame

    class _X:
        def __init__(self):
            self.voltage = torch.tensor([1.0])

    fwd = lambda st: torch.tensor([[2.0]])  # noqa: E731
    a, b = _X(), _X()
    integrate_frame(a, fwd, 0.020, "euler")
    integrate_frame(b, fwd, 0.020, "multi_substeps", 1)
    assert float(a.voltage[0]) == pytest.approx(float(b.voltage[0]))


def test_unknown_integration_method_raises():
    from connectome_gnn.models.substep import integrate_frame
    with pytest.raises(ValueError, match="unknown integration_method"):
        integrate_frame(None, None, 0.02, "rk4")


def test_abs_W_is_conductance_only():
    """The conductance generator's W_ij is a conductance, non-negative by
    construction, so the scatter and its R2 report |W|: a sign-flipped edge of
    the right size is a SIGN error and must not read as a size error. The
    current generator's W carries the synapse's sign, so there |W| would discard
    the polarity the readout exists to recover. The correction string says which
    was applied."""
    import inspect

    from connectome_gnn import metrics
    src = inspect.getsource(metrics)
    i = src.index("W_learned = np.abs(W_learned)")
    guard = src.rindex("if cond:", 0, i)
    assert i - guard < 40, "np.abs must sit under the conductance guard"
    assert "|W| reported" in src and "the current generator's W carries the sign" in src
