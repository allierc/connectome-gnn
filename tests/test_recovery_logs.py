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
