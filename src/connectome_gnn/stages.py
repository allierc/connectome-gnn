"""Reusable stage rules for stateful workflows.

``StageTracker`` checks only the dependencies declared by its owner; it does
not impose a total order. Stages are single-use unless marked repeatable, and
optional stages need not run before ``finish()``. For example::

    from connectome_gnn.stages import StageRule, StageTracker

    tracker = StageTracker({
        "load": StageRule(),
        "build": StageRule(after={"load"}),
        "epoch": StageRule(after={"build"}, repeatable=True),
        "preview": StageRule(after={"build"}, required_for_finish=False),
    })

    with tracker.stage("load"):
        data = [1, 2, 3]
    with tracker.stage("build"):
        model = {"n_items": len(data)}
    for _ in range(2):
        with tracker.stage("epoch"):
            model["n_items"] += 1

    tracker.finish()
    assert tracker.completed == frozenset({"load", "build", "epoch"})
    assert tracker.finished

Entering ``build`` before ``load`` raises ``StagePrerequisiteError``. By
default, an exception inside a stage poisons the tracker because a stateful
owner may be partially mutated; owners with recoverable stages can construct
the tracker with ``poison_on_failure=False``.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping, Set
from contextlib import contextmanager
from dataclasses import dataclass, field


class StageError(RuntimeError):
    """Base class for invalid stage transitions."""


class UnknownStageError(StageError):
    """A stage has no rule in this workflow."""


class StagePrerequisiteError(StageError):
    """A stage was entered before its prerequisites completed."""


class StageStateError(StageError):
    """The tracker cannot enter a stage in its current state."""


class StageRepeatError(StageStateError):
    """A non-repeatable stage was entered more than once."""


class StageFinishError(StageError):
    """A workflow was finished before its required stages completed."""


@dataclass(frozen=True)
class StageRule:
    """The prerequisites and completion policy for one named stage."""

    after: Set[str] = frozenset()
    repeatable: bool = False
    required_for_finish: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "after", frozenset(self.after))


@dataclass
class StageTracker:
    """Track partial ordering, repetition, failure, and workflow completion.

    ``poison_on_failure`` is selected by the owning workflow. Stateful builders
    normally poison because a failed stage may have partially mutated them;
    workflows with recoverable diagnostics may allow another stage afterward.
    """

    rules: Mapping[str, StageRule]
    poison_on_failure: bool = True
    _completed: set[str] = field(default_factory=set, init=False, repr=False)
    _running: str | None = field(default=None, init=False, repr=False)
    _failed: str | None = field(default=None, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.rules = dict(self.rules)
        known = set(self.rules)
        for name, rule in self.rules.items():
            unknown = rule.after - known
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"{name}: unknown prerequisites: {names}")

    @property
    def completed(self) -> frozenset[str]:
        return frozenset(self._completed)

    @property
    def failed(self) -> str | None:
        return self._failed

    @property
    def finished(self) -> bool:
        return self._finished

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Validate and record one stage, marking the workflow failed on error."""
        self._start(name)
        try:
            yield
        except BaseException:
            self._running = None
            if self.poison_on_failure:
                self._failed = name
            raise
        else:
            self._running = None
            self._completed.add(name)

    def finish(self) -> None:
        """Mark the workflow finished after every required stage completed."""
        self._ensure_ready("finish")
        missing = {
            name for name, rule in self.rules.items() if rule.required_for_finish and name not in self._completed
        }
        if missing:
            names = ", ".join(sorted(missing))
            raise StageFinishError(f"finish: required stages not completed: {names}")
        self._finished = True

    def _start(self, name: str) -> None:
        self._ensure_ready(name)
        try:
            rule = self.rules[name]
        except KeyError as exc:
            raise UnknownStageError(f"unknown stage: {name}") from exc
        if name in self._completed and not rule.repeatable:
            raise StageRepeatError(f"{name}: stage is not repeatable")
        missing = rule.after - self._completed
        if missing:
            names = ", ".join(sorted(missing))
            raise StagePrerequisiteError(f"{name}: prerequisites not completed: {names}")
        self._running = name

    def _ensure_ready(self, action: str) -> None:
        if self._finished:
            raise StageStateError(f"{action}: workflow is finished")
        if self._failed is not None:
            raise StageStateError(f"{action}: workflow failed in stage {self._failed}")
        if self._running is not None:
            raise StageStateError(f"{action}: stage {self._running} is already running")
