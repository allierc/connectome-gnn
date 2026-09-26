"""Global RNG snapshots, and the per-stage ledger of which stage draws from which stream.

Voltage generation draws from four GLOBAL streams and never from a generator
of its own: the torch CPU generator (plus the device generator when it runs
on CUDA or MPS), numpy's legacy global ``RandomState``, and stdlib
``random``. Some of the draws happen inside third-party code (flyvis network
init, the BoxEye conv of a cold rendering cache, ``torch.svd_lowrank``,
``plot_activity_traces``), so the order of the stages is part of the output:
move a stage and every later draw shifts.

RngSnapshot
    The state of all streams at one point, as a value. ``restore`` puts it
    back exactly (the cached Box-Muller / gauss_next values included), so a
    stage that restores its parent's snapshot before it runs draws the same
    numbers however the chain above it was built.

RngLedger
    ``with ledger.stage(name, draws=...)`` around a stage records which
    streams the stage advanced. ``draws=False`` is a claim: in check mode
    (``CGNN_RNG_LEDGER_CHECK=1``, set by the golden harness) a stage that
    advances any stream while claiming not to raises RngLedgerError.
    ``draws=True`` means MAY draw; it is never asserted. With
    ``CGNN_RNG_LEDGER_REPORT=<path>`` the per-stage record is written there as
    JSON after every stage, so a run that raises still leaves its report. The
    record is kept in memory either way (two snapshots per stage, well under a
    millisecond), so production and the harness run the same code path.

No stage may create its own ``torch.Generator`` or seed anything legacy did
not seed: that would change the streams. Per-stage ``SeedSequence``
generators are a follow-up that changes the output (a new golden reference).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

ENV_CHECK = "CGNN_RNG_LEDGER_CHECK"
ENV_REPORT = "CGNN_RNG_LEDGER_REPORT"

STREAMS = ("torch_cpu", "torch_device", "numpy", "python")


class RngLedgerError(RuntimeError):
    """A stage declared ``draws=False`` but advanced a global RNG stream."""


def _device_kind(device) -> str | None:
    """'cuda' / 'mps' when the run draws from that device's generator, else None."""
    dev = str(device) if device is not None else "cpu"
    if dev.startswith("cuda") and torch.cuda.is_available():  # golden-uncovered: env-gated-device
        return "cuda"
    if dev.startswith("mps") and torch.backends.mps.is_available():  # golden-uncovered: env-gated-device
        return "mps"
    return None


@dataclass(frozen=True)
class RngSnapshot:
    """The state of every global RNG stream generation draws from."""

    torch_cpu: bytes
    torch_device: bytes | None           # the run's CUDA / MPS generator, None on CPU
    numpy: tuple                         # np.random.get_state() with the key array as bytes
    python: tuple                        # random.getstate()

    @classmethod
    def capture(cls, device=None) -> "RngSnapshot":
        kind = _device_kind(device)
        if kind == "cuda":  # golden-uncovered: env-gated-device
            dev_state = torch.cuda.get_rng_state(device).numpy().tobytes()
        elif kind == "mps":  # golden-uncovered: env-gated-device
            dev_state = torch.mps.get_rng_state().numpy().tobytes()
        else:
            dev_state = None
        name, keys, pos, has_gauss, cached = np.random.get_state()
        return cls(
            torch_cpu=torch.get_rng_state().numpy().tobytes(),
            torch_device=dev_state,
            numpy=(name, keys.tobytes(), pos, has_gauss, cached),
            python=random.getstate(),
        )

    def restore(self, device=None) -> None:
        """Set every global stream to this snapshot."""
        torch.set_rng_state(torch.frombuffer(bytearray(self.torch_cpu), dtype=torch.uint8))
        kind = _device_kind(device)
        if kind is not None and self.torch_device is not None:  # golden-uncovered: env-gated-device
            state = torch.frombuffer(bytearray(self.torch_device), dtype=torch.uint8)
            if kind == "cuda":
                torch.cuda.set_rng_state(state, device)
            else:
                torch.mps.set_rng_state(state)
        name, keys, pos, has_gauss, cached = self.numpy
        np.random.set_state((name, np.frombuffer(keys, dtype=np.uint32).copy(), pos, has_gauss, cached))
        random.setstate(self.python)

    def digests(self) -> dict:
        """sha256 per stream (None for an absent device stream)."""
        def sha(b):
            return hashlib.sha256(b).hexdigest()
        return {
            "torch_cpu": sha(self.torch_cpu),
            "torch_device": sha(self.torch_device) if self.torch_device is not None else None,
            "numpy": sha(repr(self.numpy).encode()),
            "python": sha(repr(self.python).encode()),
        }


class RngLedger:
    """Per-stage record of RNG consumption (see the module docstring)."""

    def __init__(self, device=None):
        self.device = device
        self.check = os.environ.get(ENV_CHECK, "") == "1"
        self.report_path = os.environ.get(ENV_REPORT) or None
        self.entries: list[dict] = []
        self._open = None                    # (name, draws, snapshot before, t0)

    @contextmanager
    def stage(self, name: str, *, draws: bool):
        self.begin(name, draws=draws)
        try:
            yield
        except BaseException:
            self.abort()
            raise
        self.end()

    def begin(self, name: str, *, draws: bool) -> None:
        """Open stage ``name``, closing the open one first (sequential form of ``stage``)."""
        self.end()
        self._open = (name, draws, RngSnapshot.capture(self.device), time.perf_counter())

    def end(self) -> None:
        """Close the open stage, if any; in check mode, enforce its draws=False claim."""
        self._close(raised=False)

    def abort(self) -> None:
        """Close the open stage as raised (its claim is not enforced)."""
        self._close(raised=True)

    def _close(self, *, raised: bool) -> None:
        if self._open is None:
            return
        name, draws, before, t0 = self._open
        self._open = None
        b, a = before.digests(), RngSnapshot.capture(self.device).digests()
        advanced = [s for s in STREAMS if b[s] != a[s]]
        self.entries.append({"stage": name, "declared_draws": bool(draws), "advanced": advanced,
                             "wall_s": round(time.perf_counter() - t0, 4), "raised": raised})
        self._write()
        if self.check and not draws and advanced and not raised:  # golden-uncovered: ledger-violation
            raise RngLedgerError(
                f"stage {name!r} is declared draws=False but advanced {advanced}; "
                "either the stage draws RNG (declare draws=True) or code was moved across a draw")

    def _write(self) -> None:
        if self.report_path is None:
            return
        with open(self.report_path, "w") as f:
            json.dump({"check": self.check, "stages": self.entries}, f, indent=1)
