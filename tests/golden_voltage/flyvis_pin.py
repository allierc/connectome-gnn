"""Import flyvis with its module-level ``flyvis.device`` pinned.

``flyvis/__init__.py`` picks ``cuda`` if available, else ``mps``, else ``cpu``,
and binds it both as ``flyvis.device`` (read at call time, e.g. by
``BoxEye.__call__`` and ``chkpt_utils``) and as default arguments (bound when
flyvis's modules are imported). On a Mac with MPS this breaks DAVIS and Sintel
rendering in the flyvis-gnn-mac env: ``BoxEye`` moves the frames to MPS while
its conv weights stay on the CPU, every sequence fails with "Input type
(MPSFloatType) and weight type (torch.FloatTensor) should be the same", the
error is logged and swallowed by ``RenderedDavis``, and the dataset build ends
in "No sequences were successfully rendered".

The harness therefore imports flyvis with MPS hidden, for base and head alike,
before anything else imports it. The generator's own tensors still go to the
``device`` argument of ``data_generate_voltage`` (cpu or mps); only flyvis's
internal default device is pinned. On a CUDA host the pin is not applied
(flyvis then also calls ``torch.set_default_device(cuda)``).
"""

from __future__ import annotations

import sys


def import_flyvis_pinned(pin: str = "cpu"):
    import torch
    if "flyvis" in sys.modules:
        import flyvis
        return flyvis
    if pin == "cpu" and not torch.cuda.is_available():
        original = torch.backends.mps.is_available
        torch.backends.mps.is_available = lambda: False
        try:
            import flyvis
        finally:
            torch.backends.mps.is_available = original
        assert str(flyvis.device) == "cpu", flyvis.device
        return flyvis
    import flyvis
    return flyvis


# The same pin as a code prefix for `python -c` fixture subprocesses.
PIN_SNIPPET = (
    "import torch as _t\n"
    "_o = _t.backends.mps.is_available\n"
    "_t.backends.mps.is_available = lambda: False\n"
    "import flyvis as _fv\n"
    "_t.backends.mps.is_available = _o\n"
    "assert str(_fv.device) == 'cpu' or _t.cuda.is_available()\n"
)
