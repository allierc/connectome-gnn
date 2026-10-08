import pytest

from connectome_gnn.stages import (
    StageFinishError,
    StagePrerequisiteError,
    StageRepeatError,
    StageRule,
    StageStateError,
    StageTracker,
)


def test_partial_prerequisites_do_not_order_independent_stages():
    tracker = StageTracker(
        {
            "load": StageRule(),
            "build": StageRule(after={"load"}),
            "preview": StageRule(after={"load"}, required_for_finish=False),
        }
    )

    with pytest.raises(StagePrerequisiteError, match="load"):
        with tracker.stage("build"):
            pass

    with tracker.stage("load"):
        pass
    with tracker.stage("build"):
        pass
    tracker.finish()

    assert tracker.completed == frozenset({"load", "build"})
    assert tracker.finished


def test_repeatable_rule_allows_only_declared_repetition():
    tracker = StageTracker(
        {
            "setup": StageRule(),
            "epoch": StageRule(after={"setup"}, repeatable=True),
        }
    )

    with tracker.stage("setup"):
        pass
    with pytest.raises(StageRepeatError):
        with tracker.stage("setup"):
            pass
    with tracker.stage("epoch"):
        pass
    with tracker.stage("epoch"):
        pass


def test_failure_poisoning_preserves_failed_stage():
    tracker = StageTracker({"write": StageRule()})

    with pytest.raises(ValueError, match="disk"):
        with tracker.stage("write"):
            raise ValueError("disk")

    assert tracker.failed == "write"
    with pytest.raises(StageStateError, match="failed in stage write"):
        with tracker.stage("write"):
            pass


def test_failure_policy_can_allow_recovery():
    tracker = StageTracker({"diagnostic": StageRule()}, poison_on_failure=False)

    with pytest.raises(ValueError):
        with tracker.stage("diagnostic"):
            raise ValueError
    with tracker.stage("diagnostic"):
        pass

    assert tracker.failed is None
    assert tracker.completed == frozenset({"diagnostic"})


def test_finish_lists_missing_required_stages():
    tracker = StageTracker(
        {
            "required": StageRule(),
            "optional": StageRule(required_for_finish=False),
        }
    )

    with pytest.raises(StageFinishError, match="required"):
        tracker.finish()
