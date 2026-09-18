"""The bjobs poll must ride out an SSH outage instead of ending the campaign.

On 2026-09-18 the forwarded SSH agent died mid-poll during the LLM+HPO run and
the loop raised on the first bad `bjobs`, ending a campaign whose batch had
ALREADY trained -- the cluster jobs were unaffected and kept running, but the
orchestrator was gone. A poll failure says nothing about the jobs, so it is
retried with backoff over a budget longer than a laptop sleeping or a VPN
reconnect, and the give-up message says to restart with --resume.
"""
import subprocess as sp

import pytest

from connectome_gnn.LLM import cluster as C

pytestmark = pytest.mark.tier2


def _res(rc, out="", err=""):
    return sp.CompletedProcess(args="", returncode=rc, stdout=out, stderr=err)


def test_retries_then_succeeds(monkeypatch, tmp_path):
    calls = {"n": 0}
    fail = "ssh_askpass: exec(/usr/bin/ssh-askpass): No such file or directory"

    def fake_run(cmd, **kw):
        calls["n"] += 1
        if calls["n"] <= 3:
            return _res(255, "", fail)
        return _res(0, "JOBID USER STAT\n999 allierc DONE\n")

    monkeypatch.setattr(C.subprocess, "run", fake_run)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    d = tmp_path / "slot0"; (d / "tmp_training").mkdir(parents=True)
    r = C.wait_for_cluster_jobs_with_metrics({0: "999"}, {0: str(d)},
                                             poll_interval=0, metrics_interval=10**9)
    assert r == {0: True}, r
    assert calls["n"] == 4, calls


def test_gives_up_after_the_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(C.subprocess, "run", lambda cmd, **kw: _res(255, "", "down"))
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    d = tmp_path / "slot0"; (d / "tmp_training").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="--resume"):
        C.wait_for_cluster_jobs_with_metrics({0: "999"}, {0: str(d)},
                                             poll_interval=0, metrics_interval=10**9)
