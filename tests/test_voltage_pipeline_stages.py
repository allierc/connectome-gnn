from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from connectome_gnn.generators.voltage.pipeline import VOLTAGE_STAGE_RULES, StaleStageError, VoltageGeneration
from connectome_gnn.generators.voltage.rng import RngLedger, RngSnapshot
from connectome_gnn.stages import StagePrerequisiteError, StageRule, StageTracker

pytestmark = pytest.mark.tier2


def test_stage_mutates_and_returns_same_generation():
    store = Mock()
    generator = VoltageGeneration(
        spec=cast(Any, SimpleNamespace(device=None)),
        store=store,
        ledger=RngLedger(),
        rng=RngSnapshot.capture(),
        stages=StageTracker({"make_folders": StageRule()}),
    )

    result = generator.make_folders()

    assert result is generator
    assert generator.stages.completed == frozenset({"make_folders"})
    store.make_folders.assert_called_once_with()

    torch.manual_seed(123456)
    rng_before_rejected_call = torch.get_rng_state().clone()
    with pytest.raises(StaleStageError, match="not repeatable"):
        generator.make_folders()
    assert torch.equal(torch.get_rng_state(), rng_before_rejected_call)


def test_train_integration_requires_output_folders():
    tracker = StageTracker(VOLTAGE_STAGE_RULES)
    for name in (
        "seed",
        "build_network",
        "load_stimuli",
        "extract_ode_params",
        "add_null_edges",
        "ablate",
        "build_ode",
        "init_geometry",
        "steady_state",
        "init_state",
        "split_videos",
        "materialize_sequences",
    ):
        with tracker.stage(name):
            pass

    with pytest.raises(StagePrerequisiteError, match="make_folders"):
        with tracker.stage("integrate_train"):
            pass

    with tracker.stage("prepare_output"):
        pass
    with tracker.stage("make_folders"):
        pass
    with tracker.stage("integrate_train"):
        pass
