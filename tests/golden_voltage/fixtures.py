"""Deterministic fixtures for the data_generate_voltage golden harness.

Everything here is built ONCE into a work directory and then only read. The
old and the new implementation must see the same fixture bytes and the same
directory entries, because two of them leak into the outputs:

  * ``RenderedDavis`` walks the video root with ``Path.iterdir()``, so the
    sequence numbering in the rendering cache follows the filesystem's order.
  * ``datavis_roots`` is echoed verbatim into ``generation_log.txt``, so the
    absolute fixture path is part of the golden bytes.

A fixture is therefore never rebuilt between a base run and a head run. Each
piece has its own completion marker, so an interrupted build restarts cleanly.

Layout under ``<work>/fixtures``::

    davis_a/JPEGImages/480p/vid_00..05   six 64-frame videos
                           /short_00..01 two 20-frame videos (skip_short_videos)
    davis_b/...                          three 64-frame videos (combined roots)
    davis_c/...                          two 64-frame videos (small mixed-mode pool)
    davis_single/...                     one video (empty train split)
    davis_short/...                      two 20-frame videos (mixed-mode davis_iter exhaustion)
    flyvis_template/                     FLYVIS_ROOT_DIR template: bundled model,
                                         connectome caches, rendering caches
    net/twin_safe.pt, twin_cross.pt      conductance-twin checkpoints
    net/kept_mask.pt                     precomputed kept_edge_indices
    net/graph_stats.json                 edge count, out-degree facts used by cells
    dirty/                               stale dataset directory (erase semantics)
    stub_conductance/                    fake flyvis_conductance_optical_flow package

Only numpy, cv2 and the standard library are imported at module level; the
pieces that need flyvis or the base implementation run in a subprocess.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

FIXTURE_VERSION = 1

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent

VIDEO_ROOTS = {
    # name: [(video_name, n_frames, seed), ...]
    "davis_a": [(f"vid_{k:02d}", 64, 100 + k) for k in range(6)]
    + [(f"short_{k:02d}", 20, 200 + k) for k in range(2)],
    "davis_b": [(f"bvid_{k:02d}", 64, 300 + k) for k in range(3)],
    "davis_c": [(f"cvid_{k:02d}", 64, 400 + k) for k in range(2)],
    "davis_single": [("solo_00", 64, 500)],
    "davis_short": [(f"svid_{k:02d}", 20, 600 + k) for k in range(2)],
}
FRAME_HW = (96, 160)


def work_root() -> Path:
    """Where fixtures, base-run caches and scratch runs live (never in the repo)."""
    env = os.environ.get("CGNN_GOLDEN_WORK")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".cache" / "cgnn-golden-voltage").resolve()


def fixtures_dir(work: Path) -> Path:
    return Path(work) / "fixtures"


def davis_root(work: Path, name: str) -> str:
    """The value a cell puts into ``simulation.datavis_roots``."""
    return str(fixtures_dir(work) / name)


def _marker(path: Path) -> Path:
    return path.parent / f".{path.name}.complete"


def _done(path: Path) -> bool:
    m = _marker(path)
    return m.exists() and m.read_text().strip() == str(FIXTURE_VERSION)


def _mark(path: Path) -> None:
    _marker(path).write_text(f"{FIXTURE_VERSION}\n")


# ---------------------------------------------------------------------------
# synthetic videos
# ---------------------------------------------------------------------------

def _video_frames(seed: int, n_frames: int, hw=FRAME_HW) -> np.ndarray:
    """(T, H, W, 3) uint8: a drifting grating plus three moving Gaussian blobs."""
    rs = np.random.RandomState(seed)
    h, w = hw
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    theta = rs.uniform(0, np.pi)
    freq = rs.uniform(0.03, 0.12)
    speed = rs.uniform(0.1, 0.6)
    tint = rs.uniform(0.7, 1.0, size=3)
    blobs = [
        (rs.uniform(0, h), rs.uniform(0, w), rs.uniform(-2, 2), rs.uniform(-2, 2),
         rs.uniform(6, 20), rs.uniform(-0.45, 0.45))
        for _ in range(3)
    ]
    proj = xx * np.cos(theta) + yy * np.sin(theta)
    frames = np.empty((n_frames, h, w, 3), dtype=np.uint8)
    for t in range(n_frames):
        img = 0.5 + 0.3 * np.sin(2 * np.pi * freq * proj - speed * t)
        for cy, cx, vy, vx, r, a in blobs:
            img += a * np.exp(-((yy - cy - vy * t) ** 2 + (xx - cx - vx * t) ** 2) / (2 * r * r))
        img = np.clip(img, 0, 1)
        frames[t] = np.round(255 * img[..., None] * tint[None, None, :]).astype(np.uint8)
    return frames


def build_videos(work: Path) -> None:
    import cv2

    fx = fixtures_dir(work)
    for root_name, videos in VIDEO_ROOTS.items():
        root = fx / root_name
        if _done(root):
            continue
        if root.exists():
            shutil.rmtree(root)
        base = root / "JPEGImages" / "480p"
        for vname, n_frames, seed in videos:
            vdir = base / vname
            vdir.mkdir(parents=True)
            for t, rgb in enumerate(_video_frames(seed, n_frames)):
                bgr = np.ascontiguousarray(rgb[..., ::-1])
                ok = cv2.imwrite(str(vdir / f"{t:05d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                assert ok, vdir
        _mark(root)


# ---------------------------------------------------------------------------
# subprocess helper (flyvis / base implementation)
# ---------------------------------------------------------------------------

def _run_py(code: str, *, env_extra: dict, python: str, cwd: Path, pythonpath: list[str] | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "USER", "LANG", "TMPDIR", "LC_ALL")}
    env.update({
        "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg", "TZ": "UTC",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    })
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env.update(env_extra)
    cwd.mkdir(parents=True, exist_ok=True)
    from .flyvis_pin import PIN_SNIPPET
    proc = subprocess.run([python, "-c", PIN_SNIPPET + code], env=env, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fixture subprocess failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
    return proc.stdout


# ---------------------------------------------------------------------------
# flyvis root template
# ---------------------------------------------------------------------------

_BUILD_CONNECTOME = textwrap.dedent("""
    import sys
    from flyvis import Network
    from flyvis.utils.config_utils import CONFIG_PATH, get_default_config
    for extent in [int(e) for e in sys.argv[1:]] if len(sys.argv) > 1 else EXTENTS:
        cfg = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
        cfg.connectome.extent = extent
        net = Network(**cfg)
        print(extent, net.n_nodes, net.n_edges, flush=True)
""")


def build_flyvis_template(work: Path, base_root: Path, python: str, extents=(8,)) -> Path:
    """FLYVIS_ROOT_DIR template: bundled model 000, connectome caches, Sintel link.

    The model comes from the BASE worktree's ``assets/flyvis_model`` -- the file
    ``setup_flyvis_model_path`` would install -- so ``setup_flyvis_model_path``
    finds it present and never copies (its log line would otherwise appear in
    the first run only).
    """
    tpl = fixtures_dir(work) / "flyvis_template"
    tpl.mkdir(parents=True, exist_ok=True)
    model_dst = tpl / "results" / "flow" / "0000" / "000"
    if not (model_dst / "best_chkpt").exists():
        src = Path(base_root) / "assets" / "flyvis_model" / "flow" / "0000" / "000"
        assert (src / "best_chkpt").exists(), f"bundled flyvis model missing at {src}"
        model_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, model_dst)
    sintel_link = tpl / "SintelDataSet"
    if not sintel_link.exists():
        real = Path.home() / "Projects" / "flyvis" / "data" / "SintelDataSet"
        if real.exists():
            sintel_link.symlink_to(real)
    for extent in extents:
        m = tpl / f"connectome_e{extent}"
        if _done(m):
            continue
        code = _BUILD_CONNECTOME.replace("EXTENTS", repr([extent]))
        out = _run_py(code, env_extra={"FLYVIS_ROOT_DIR": str(tpl)}, python=python, cwd=work / "tmp")
        m.write_text(out)
        _mark(m)
    # NetworkView builds the connectome of the MODEL's own config (extent 15)
    # before init_network; without it in the template every run spends ~10 s
    # building it (5 s of that in datamate's polling sleeps).
    m = tpl / "networkview_000"
    if not _done(m):
        code = ("from flyvis import NetworkView\n"
                "NetworkView('flow/0000/000').init_network(checkpoint=0)\nprint('ok')\n")
        m.write_text(_run_py(code, env_extra={"FLYVIS_ROOT_DIR": str(tpl)}, python=python, cwd=work / "tmp"))
        _mark(m)
    return tpl


def clone_tree(src: Path, dst: Path, *, exclude=()) -> None:
    """Copy a directory, using APFS clones on macOS (constant time, copy-on-write)."""
    dst.mkdir(parents=True, exist_ok=True)
    for child in sorted(Path(src).iterdir()):
        if child.name in exclude or child.name.startswith(".") and child.name.endswith(".complete"):
            continue
        target = dst / child.name
        if child.is_symlink():
            target.symlink_to(os.readlink(child))
        elif sys.platform == "darwin":
            subprocess.run(["cp", "-cR", str(child), str(target)], check=True)
        elif child.is_dir():
            shutil.copytree(child, target, symlinks=True)
        else:
            shutil.copy2(child, target)


# ---------------------------------------------------------------------------
# rendering caches (pre-rendered with the BASE implementation's davis.py)
# ---------------------------------------------------------------------------

# Mirrors graph_data_generator.py:2445-2468 (the RenderedDavis part of it:
# boxfilter, vertical_splits, n_frames, max_frames, center_crop_fraction,
# skip_short_videos, davis_path). The generator's dataset lookup hits this
# cache only when these match; driver.py records any cold render in run_info.json.
_RENDER_DAVIS = textwrap.dedent("""
    import json, os, sys
    specs = json.loads(sys.argv[1])
    from connectome_gnn.generators.davis import AugmentedVideoDataset
    for s in specs:
        video_config = {
            "n_frames": 50, "max_frames": s["max_frames"], "flip_axes": [0, 1],
            "n_rotations": [0, 90, 180, 270], "temporal_split": False, "dt": 0.02,
            "boxfilter": dict(extent=s["extent"], kernel_size=13), "vertical_splits": 1,
            "center_crop_fraction": 0.6, "augment": False, "unittest": False,
            "skip_short_videos": s["skip_short"], "shuffle_sequences": True, "shuffle_seed": 0,
        }
        ds = AugmentedVideoDataset(root_dir=os.path.join(s["root"], "JPEGImages/480p"), **video_config)
        print(json.dumps(s), len(ds), flush=True)
""")

# Mirrors graph_data_generator.py:2488-2500.
_RENDER_SINTEL = textwrap.dedent("""
    import json, sys
    from flyvis.datasets.sintel import AugmentedSintel
    for extent in json.loads(sys.argv[1]):
        ds = AugmentedSintel(n_frames=19, flip_axes=[0, 1], n_rotations=[0, 1, 2, 3, 4, 5],
                             temporal_split=True, dt=0.02, interpolate=True,
                             boxfilter=dict(extent=extent, kernel_size=13),
                             vertical_splits=3, center_crop_fraction=0.7)
        print(extent, len(ds), flush=True)
""")


def render_specs_key(specs) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(sorted(specs, key=json.dumps), sort_keys=True).encode()).hexdigest()[:16]


def build_renderings(work: Path, base_root: Path, python: str, davis_specs: list[dict], sintel_extents=()) -> None:
    tpl = fixtures_dir(work) / "flyvis_template"
    done_file = tpl / "renderings_done.json"
    done = json.loads(done_file.read_text()) if done_file.exists() else {"davis": [], "sintel": []}
    todo = [s for s in davis_specs if s not in done["davis"]]
    env = {"FLYVIS_ROOT_DIR": str(tpl)}
    pp = [str(Path(base_root) / "src"), str(base_root)]
    if todo:
        code = _RENDER_DAVIS.replace("sys.argv[1]", repr(json.dumps(todo)))
        _run_py(code, env_extra=env, python=python, cwd=work / "tmp", pythonpath=pp)
        done["davis"] += todo
        done_file.write_text(json.dumps(done, indent=1, sort_keys=True))
    todo_s = [e for e in sintel_extents if e not in done["sintel"]]
    if todo_s:
        code = _RENDER_SINTEL.replace("sys.argv[1]", repr(json.dumps(todo_s)))
        _run_py(code, env_extra=env, python=python, cwd=work / "tmp", pythonpath=pp)
        done["sintel"] += todo_s
        done_file.write_text(json.dumps(done, indent=1, sort_keys=True))


# ---------------------------------------------------------------------------
# network-derived fixtures (flyvis only; no connectome_gnn import)
# ---------------------------------------------------------------------------

_NET_FIXTURES = textwrap.dedent("""
    import json, sys
    from pathlib import Path
    import numpy as np
    import torch
    from flyvis import Network, NetworkView
    from flyvis.utils.config_utils import CONFIG_PATH, get_default_config
    out = Path(sys.argv[1])
    cfg = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
    cfg.connectome.extent = 8
    net = Network(**cfg)
    net.load_state_dict(NetworkView("flow/0000/000").init_network(checkpoint=0).state_dict())
    p = net._param_api()
    tau = p.nodes.time_const.detach().float()
    v_rest = p.nodes.bias.detach().float()
    w = (p.edges.syn_strength * p.edges.syn_count * p.edges.sign).detach().float()
    src = torch.tensor(net.connectome.edges.source_index[:]).long()
    types = np.array(net.connectome.nodes["type"])
    _, type_index = np.unique(types, return_inverse=True)
    type_index = torch.tensor(type_index).long()
    n_types = int(type_index.max()) + 1
    def per_type(v):
        return torch.stack([v[type_index == t].mean() for t in range(n_types)])
    tau_t = per_type(tau)
    raw_tau = tau_t + torch.log(-torch.expm1(-tau_t))        # softplus^-1
    vrest_t = per_type(v_rest)
    base = dict(W=w.abs().sqrt()[:, None], raw_tau=raw_tau, V_rest=vrest_t,
                type_index=type_index, edge_is_inh=(w < 0))
    safe = dict(base, E_exc=torch.tensor([5.0]), E_inh=torch.full((n_types,), -5.0))
    cross = dict(base, E_exc=torch.tensor([5.0]), E_inh=vrest_t + 0.5)
    torch.save({"model_state_dict": safe}, out / "twin_safe.pt")
    torch.save({"model_state_dict": cross}, out / "twin_cross.pt")
    n_edges = int(src.numel())
    rs = np.random.RandomState(1234)
    kept = np.sort(rs.choice(n_edges, int(0.7 * n_edges), replace=False))
    torch.save(torch.tensor(kept), out / "kept_mask.pt")
    deg = np.bincount(src.numpy(), minlength=net.n_nodes)
    stats = dict(n_nodes=int(net.n_nodes), n_edges=n_edges, n_types=n_types,
                 first_zero_outdegree=int(np.argmax(deg == 0)) if (deg == 0).any() else -1,
                 n_zero_outdegree=int((deg == 0).sum()),
                 outdeg_first_300=deg[:300].tolist(),
                 min_outdeg_nonzero=int(deg[deg > 0].min()))
    (out / "graph_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "outdeg_first_300"}))
""")


def build_net_fixtures(work: Path, python: str) -> Path:
    out = fixtures_dir(work) / "net"
    if _done(out):
        return out
    out.mkdir(parents=True, exist_ok=True)
    tpl = fixtures_dir(work) / "flyvis_template"
    scratch = work / "tmp" / "net_fixture_root"
    if scratch.exists():
        shutil.rmtree(scratch)
    clone_tree(tpl, scratch, exclude=("renderings",))   # NetworkView writes __cache__
    code = _NET_FIXTURES.replace("sys.argv[1]", repr(str(out)))
    _run_py(code, env_extra={"FLYVIS_ROOT_DIR": str(scratch)}, python=python, cwd=work / "tmp")
    shutil.rmtree(scratch)
    _mark(out)
    return out


def graph_stats(work: Path) -> dict:
    return json.loads((fixtures_dir(work) / "net" / "graph_stats.json").read_text())


# ---------------------------------------------------------------------------
# dirty dataset directory
# ---------------------------------------------------------------------------

def _write_stale_zarr(path: Path, shape) -> None:
    import tensorstore as ts
    spec = {
        "driver": "zarr", "kvstore": {"driver": "file", "path": str(path)},
        "metadata": {"dtype": "<f4", "shape": list(shape), "chunks": list(shape),
                     "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}},
        "create": True, "delete_existing": True,
    }
    store = ts.open(spec).result()
    store.write(np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) / 7.0).result()


def build_dirty(work: Path) -> Path:
    """A dataset directory as an earlier (different) run would have left it.

    The generator's erase looks for ``y_list_*`` WITHOUT the ``.zarr`` suffix,
    so none of the stale ``*.zarr`` below are removed by ``erase=True``; the
    cells that copy this directory pin exactly which stale files survive.
    """
    d = fixtures_dir(work) / "dirty"
    if _done(d):
        return d
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    _write_stale_zarr(d / "y_list_train.zarr", (10, 5, 1))
    _write_stale_zarr(d / "noisy_y_list_train.zarr", (10, 5, 1))
    _write_stale_zarr(d / "noisy_y_list_test.zarr", (10, 5, 1))
    (d / "x_list_train").mkdir()
    (d / "x_list_train" / "stale_marker.txt").write_text("stale x_list_train\n")
    _write_stale_zarr(d / "x_list_train" / "voltage.zarr", (10, 5))
    (d / "x_list_test").mkdir()
    (d / "x_list_test" / "stale_marker.txt").write_text("stale x_list_test\n")
    (d / "Fig").mkdir()
    (d / "Fig" / "Fig_0_000123.png").write_bytes(b"\x89PNG stale\n")
    (d / "noisy_test_data.ok").write_text("")
    (d / "BRACKET_VIOLATION.txt").write_text("stale bracket violation\n")
    import torch
    torch.save(torch.arange(5, dtype=torch.float32), d / "weights_full.pt")
    _mark(d)
    return d


# ---------------------------------------------------------------------------
# stub for flyvis_conductance_optical_flow (environment-gated branch B09/B10)
# ---------------------------------------------------------------------------

def build_conductance_stub(work: Path) -> Path:
    """A package that makes ``import flyvis_conductance_optical_flow`` succeed.

    ``CONDUCTANCE_DYNAMICS`` names a class that is NOT registered with flyvis,
    so the generator reaches its dynamics-class check with flyvis's fallback
    base class and raises (B10). The real package is only on the cluster.
    """
    d = fixtures_dir(work) / "stub_conductance"
    pkg = d / "flyvis_conductance_optical_flow"
    if _done(d):
        return d
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text('"""golden-harness stub; not the real package."""\n')
    (pkg / "config.py").write_text('CONDUCTANCE_DYNAMICS = "GoldenStubConductanceDynamics"\n')
    _mark(d)
    return d


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def ensure_fixtures(work: Path, base_root: Path, python: str, *, davis_specs=(), sintel_extents=(),
                    extents=(8,)) -> None:
    work = Path(work)
    (work / "tmp").mkdir(parents=True, exist_ok=True)
    build_videos(work)
    build_flyvis_template(work, base_root, python, extents=tuple(sorted(set(extents) | {8})))
    build_net_fixtures(work, python)
    build_dirty(work)
    build_conductance_stub(work)
    build_renderings(work, base_root, python, list(davis_specs), sintel_extents=tuple(sintel_extents))
