"""Run ONE golden cell with ONE implementation, in this (fresh) process.

    python tests/golden_voltage/cell_runner.py --impl-root <worktree> --cell F1_base \
        --out <run dir> [--device cpu] [--threads 1] [--no-preseed]

Do not call this directly; ``driver.run_one`` builds the
environment (PYTHONHASHSEED, thread caps, FLYVIS_ROOT_DIR, PYTHONPATH) that
must be in place BEFORE the interpreter starts.

What this process does, in order:
  1. puts ``<impl>/src`` and ``<impl>`` (for ``GNN_PlotFigure``) first on
     sys.path and ASSERTS that ``connectome_gnn`` comes from ``<impl>``: the
     env's editable install points at ~/Projects/connectome-gnn/src, which is
     not necessarily the implementation under test;
  2. ``set_data_root(<out>/data)``; copies the dirty template if asked;
  3. seeds stdlib ``random`` with 0xC0FFEE (identical for base and head):
     data_generate_voltage never seeds it, and the Sintel path never reaches
     the DAVIS dataset's ``random.seed``, so without this the null-edge draws
     are nondeterministic across processes;
  4. builds the config from base_config.yaml + the cell's overrides;
  5. applies the cell's monkeypatches (identical for both implementations);
  6. calls the entry point and records the exception, if any;
  7. writes state.json: RNG digests, grad mode, rcParams digest, logger levels,
     exception; plus informational fields that are never compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent


def _setup_path(impl_root: Path) -> None:
    here = str(HARNESS_DIR)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    sys.path.insert(0, str(impl_root))
    sys.path.insert(0, str(impl_root / "src"))
    sys.path.append(str(HARNESS_DIR.parent))      # for `import golden_voltage`


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rng_digests(device) -> dict:
    import random

    import numpy as np
    import torch
    out = {"torch_cpu": _sha(torch.get_rng_state().numpy().tobytes())}
    name, keys, pos, has_gauss, cached = np.random.get_state()
    out["numpy"] = _sha(name.encode() + keys.tobytes() + repr((pos, has_gauss, cached)).encode())
    out["python"] = _sha(repr(random.getstate()).encode())
    dev = str(device or "cpu")
    if dev.startswith("cuda") and torch.cuda.is_available():
        out["torch_device"] = _sha(b"".join(s.numpy().tobytes() for s in torch.cuda.get_rng_state_all()))
    elif dev.startswith("mps") and torch.backends.mps.is_available():
        out["torch_device"] = _sha(torch.mps.get_rng_state().numpy().tobytes())
    return out


def global_state(device) -> dict:
    import logging

    import matplotlib
    import torch
    rc = sorted((k, repr(v)) for k, v in matplotlib.rcParams.items())
    return {
        "rng": rng_digests(device),
        "grad_enabled": bool(torch.is_grad_enabled()),
        "rcparams_sha": _sha(repr(rc).encode()),
        "root_logger_level": logging.getLogger().level,
        "flyvis_logging_utils_level": logging.getLogger("flyvis.utils.logging_utils").level,
        "default_device": str(torch.get_default_device()),
    }


# ---------------------------------------------------------------------------
# monkeypatches: same object for base and head, applied everywhere it is bound
# ---------------------------------------------------------------------------

def _patch_everywhere(module_name: str, attr: str, make_replacement) -> None:
    """Replace ``module.attr`` and every ``connectome_gnn.*`` binding of the same object.

    Works whether the implementation looks the name up at call time (lazy
    ``from X import f`` inside the function) or bound it at import time, and
    whichever module the refactor moves the caller into.
    """
    import importlib
    mod = importlib.import_module(module_name)
    original = getattr(mod, attr)
    replacement = make_replacement(original)
    # import the generator package so its top-level bindings exist before the sweep
    importlib.import_module("connectome_gnn.generators.graph_data_generator")
    for name, m in list(sys.modules.items()):
        if m is None or not (name.startswith("connectome_gnn") or name == "GNN_PlotFigure"):
            continue
        if getattr(m, attr, None) is original:
            setattr(m, attr, replacement)


def _raiser(msg):
    def make(original):
        def f(*a, **k):
            raise RuntimeError(msg)
        return f
    return make


def _raise_on_filename(suffix, msg):
    def make(original):
        def f(path, *a, **k):
            if str(path).endswith(suffix):
                raise RuntimeError(msg)
            return original(path, *a, **k)
        return f
    return make


PATCHES = {
    "build_neighbor_graph_raises": lambda: _patch_everywhere(
        "connectome_gnn.generators.utils", "build_neighbor_graph", _raiser("golden: build_neighbor_graph disabled")),
    "report_dataset_reversals_raises": lambda: _patch_everywhere(
        "connectome_gnn.plot", "report_dataset_reversals", _raiser("golden: reversal report disabled")),
    "per_type_traces_raise": lambda: _patch_everywhere(
        "connectome_gnn.models.teacher_eval", "save_trace_figure",
        _raise_on_filename("activity_all.png", "golden: per-type traces disabled")),
}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class SimProxy:
    """config.simulation plus attributes SimulationConfig cannot hold.

    SimulationConfig has extra="ignore", so ``save_calcium`` and
    ``n_frames_test`` (read by the generator only through getattr) can never
    be set from YAML; this proxy is the only way to reach those branches.
    """

    def __init__(self, sim, extra):
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_extra", dict(extra))

    def __getattr__(self, name):
        extra = object.__getattribute__(self, "_extra")
        if name in extra:
            return extra[name]
        return getattr(object.__getattribute__(self, "_sim"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_sim"), name, value)


class ConfigProxy:
    def __init__(self, config, sim_proxy):
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_sim", sim_proxy)

    def __getattr__(self, name):
        if name == "simulation":
            return object.__getattribute__(self, "_sim")
        return getattr(object.__getattribute__(self, "_config"), name)


def build_config(cell, work: Path):
    import yaml

    from connectome_gnn.config import NeuralGraphConfig
    from golden_voltage.cells import resolve_env_value
    from golden_voltage.fixtures import davis_root

    raw = yaml.safe_load((HARNESS_DIR / "base_config.yaml").read_text())
    raw["dataset"] = cell.dataset
    raw["graph_model"]["signal_model_name"] = cell.model
    raw["simulation"]["datavis_roots"] = [davis_root(work, r) for r in cell.roots]
    for k, v in cell.sim.items():
        raw["simulation"][k] = resolve_env_value(v, work)
    config = NeuralGraphConfig(**raw)
    if cell.phantom:
        return ConfigProxy(config, SimProxy(config.simulation, cell.phantom))
    return config


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl-root", required=True, type=Path)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-preseed", action="store_true")
    ap.add_argument("--flyvis-device", default="cpu", help="pin flyvis.device (see flyvis_pin.py)")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms(True) (CUDA tier)")
    args = ap.parse_args(argv)

    impl_root = args.impl_root.resolve()
    out = args.out.resolve()
    _setup_path(impl_root)

    from golden_voltage.cells import CELLS_BY_NAME
    cell = CELLS_BY_NAME[args.cell]

    t0 = time.time()
    import random

    import torch
    torch.set_num_threads(args.threads)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    from golden_voltage.flyvis_pin import import_flyvis_pinned
    fv = import_flyvis_pinned(args.flyvis_device)

    import connectome_gnn
    got = Path(connectome_gnn.__file__).resolve()
    assert str(got).startswith(str(impl_root) + os.sep), (
        f"connectome_gnn imported from {got}, not from the implementation under test {impl_root}")
    from connectome_gnn.utils import set_data_root

    data_root = out / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    set_data_root(str(data_root))

    if cell.dirty:
        from golden_voltage.fixtures import clone_tree, fixtures_dir
        clone_tree(fixtures_dir(args.work) / "dirty", data_root / "graphs_data" / cell.dataset)

    config = build_config(cell, args.work)
    for p in cell.patches:
        PATCHES[p]()

    if not args.no_preseed:
        random.seed(0xC0FFEE)

    exc = None
    t_call = time.time()
    try:
        if cell.entry == "voltage":
            from connectome_gnn.generators.graph_data_generator import data_generate_voltage
            data_generate_voltage(config, device=args.device, **cell.call_args())
        elif cell.entry == "dispatch":
            from connectome_gnn.generators.graph_data_generator import data_generate
            data_generate(config, device=args.device, **cell.call_args())
        else:
            raise ValueError(cell.entry)
    except BaseException as e:   # noqa: BLE001 -- the exception IS the result for raise cells
        exc = {"type": type(e).__name__, "message": str(e)}
        (out / "traceback.txt").write_text(traceback.format_exc())
    t_end = time.time()

    sys.stdout.flush()
    sys.stderr.flush()
    state = {
        "compared": dict(global_state(args.device), exception=exc),
        "info": {
            "cell": cell.name,
            "impl_root": str(impl_root),
            "connectome_gnn_file": str(got),
            "device": args.device,
            "flyvis_device": str(fv.device),
            "threads": args.threads,
            "preseed": not args.no_preseed,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "wall_import_s": round(t_call - t0, 3),
            "wall_call_s": round(t_end - t_call, 3),
        },
    }
    (out / "state.json").write_text(json.dumps(state, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
