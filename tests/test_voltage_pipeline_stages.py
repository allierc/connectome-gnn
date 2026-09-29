from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from connectome_gnn.generators.voltage.pipeline import StaleStageError, VoltageGeneration
from connectome_gnn.generators.voltage.rng import RngLedger, RngSnapshot
from connectome_gnn.stages import StageRule, StageTracker

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
