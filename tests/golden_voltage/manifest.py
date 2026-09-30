"""Manifests of a golden run, and the comparators between two of them.

A manifest records everything a ``data_generate_voltage`` call left behind:

  files   every file under ``<run>/data`` (the whole graphs_data tree): sha256
          of the bytes, plus a decoded digest for PNG (RGBA pixels) and MP4
          (``ffmpeg -f framemd5`` + stream parameters)
  dirs    every directory, including empty ones (``Fig/`` must exist and be empty)
  state   the ``compared`` block of state.json (RNG digests, grad mode,
          rcParams digest, logger levels, exception)
  logs    stdout / stderr after normalisation (``normalize_log``)

Comparison policy (``POLICY``) was set by the determinism phase, see
DETERMINISM.md. Every kind is compared; ``policy`` only chooses between the
raw-bytes digest and the decoded digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

# Set from the determinism phase (DETERMINISM.md, section "Comparator policy").
POLICY = {
    "zarr": "bytes",    # .zarray / .zattrs / chunk files
    "pt": "bytes",      # torch.save zip archives
    "txt": "bytes",     # generation_log.txt, BRACKET_*.txt, .ok markers, .generate_done (masked)
    "png": "bytes",
    "mp4": "decoded",   # container bytes can carry an encoder string; frames are what matter
    "other": "bytes",
}


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kind_of(rel: str) -> str:
    parts = rel.split("/")
    if any(p.endswith(".zarr") for p in parts[:-1]):
        return "zarr"
    suffix = os.path.splitext(rel)[1].lower()
    return {".pt": "pt", ".txt": "txt", ".ok": "txt", ".png": "png", ".mp4": "mp4",
            ".csv": "txt", ".json": "txt", ".yaml": "txt"}.get(suffix, "other")


def _png_decoded(p: Path) -> str | None:
    try:
        import numpy as np
        from PIL import Image
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGBA"))
        return hashlib.sha256(repr(arr.shape).encode() + arr.tobytes()).hexdigest()
    except Exception:   # a stale fixture PNG is not a real PNG
        return None


def _mp4_decoded(p: Path) -> str | None:
    try:
        fm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-f", "framemd5", "-"],
                            capture_output=True, text=True, check=True).stdout
        frames = "\n".join(line for line in fm.splitlines() if not line.startswith("#"))
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(p)],
                               capture_output=True, text=True, check=True).stdout
        keep = ("codec_name", "width", "height", "pix_fmt", "r_frame_rate", "nb_frames", "duration_ts",
                "time_base", "profile", "level")
        streams = [{k: s.get(k) for k in keep} for s in json.loads(probe)["streams"]]
        return hashlib.sha256((frames + json.dumps(streams, sort_keys=True)).encode()).hexdigest()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# log normalisation
# ---------------------------------------------------------------------------

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_TQDM = re.compile(r"(\d+%\|)|(it/s\])|(s/it\])|(\[\d\d:\d\d<)")
_TS = re.compile(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] ")
_MODLINE = re.compile(r"^(\[<TS>\] [A-Za-z_][\w]*):\d+ ")
_WARNLOC = re.compile(r"^(\S+\.py):\d+: (\w*Warning)")


def normalize_log(text: str, *, replacements: dict[str, str]) -> str:
    """Make a captured stream comparable across runs and implementations.

    Removed: tqdm progress segments (rates, ETAs, ``\\r`` redraws), log
    timestamps, the line number in ``module:lineno`` log prefixes and in
    warning locations (a refactor moves code, not behaviour). Replaced:
    absolute run / implementation / work paths by placeholders. ANSI colour
    codes are KEPT: they encode the removal-colour branch (B37).
    """
    out = []
    for raw_line in text.replace("\r\n", "\n").split("\n"):
        segs = [s for s in raw_line.split("\r") if s and not _TQDM.search(s)]
        if not segs:
            if raw_line == "":
                out.append("")
            continue
        line = segs[-1]
        for old, new in replacements.items():
            if old:
                line = line.replace(old, new)
        line = _TS.sub("[<TS>] ", line)
        line = _MODLINE.sub(r"\1:<L> ", line)
        line = _WARNLOC.sub(r"\1:<L>: \2", line)
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


def log_replacements(run_dir: Path, impl_root: str, work: str) -> dict[str, str]:
    run_dir = str(Path(run_dir).resolve())
    reps = {}
    for p, tag in ((run_dir, "<OUT>"), (str(Path(impl_root).resolve()), "<IMPL>"),
                   (str(Path(work).resolve()), "<WORK>")):
        reps[p] = tag
        if p.startswith("/private/"):
            reps[p[len("/private"):]] = tag      # macOS /tmp and /var are symlinks into /private
    # longest first so a nested path is replaced before its parent
    return dict(sorted(reps.items(), key=lambda kv: -len(kv[0])))


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def _mask_generate_done(p: Path) -> str:
    lines = [ln for ln in p.read_text().splitlines() if not ln.startswith("date:")]
    lines = [("git_sha: <MASKED>" if ln.startswith("git_sha:") else ln) for ln in lines]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def build_manifest(run_dir: Path, *, impl_root: str, work: str) -> dict:
    run_dir = Path(run_dir)
    data = run_dir / "data"
    files, dirs = {}, []
    for root, dnames, fnames in os.walk(data):
        dnames.sort()
        rel_root = os.path.relpath(root, data)
        dirs.append(rel_root)
        for fn in sorted(fnames):
            p = Path(root) / fn
            rel = os.path.normpath(os.path.join(rel_root, fn))
            k = kind_of(rel)
            entry = {"kind": k, "size": p.stat().st_size, "sha256": _sha_file(p)}
            if fn == ".generate_done":
                entry["masked"] = _mask_generate_done(p)
            if k == "png":
                entry["decoded"] = _png_decoded(p)
            elif k == "mp4":
                entry["decoded"] = _mp4_decoded(p)
            files[rel] = entry
    state = json.loads((run_dir / "state.json").read_text()) if (run_dir / "state.json").exists() else None
    reps = log_replacements(run_dir, impl_root, work)
    if state and state["compared"].get("exception"):
        msg = state["compared"]["exception"]["message"]
        for old, new in reps.items():
            msg = msg.replace(old, new)
        state["compared"]["exception"]["message"] = msg
    logs = {}
    for stream in ("stdout", "stderr"):
        p = run_dir / f"{stream}.txt"
        text = p.read_text(errors="replace") if p.exists() else ""
        norm = normalize_log(text, replacements=reps)
        (run_dir / f"{stream}.normalized.txt").write_text(norm)
        logs[stream] = hashlib.sha256(norm.encode()).hexdigest()
    return {
        "files": files,
        "dirs": sorted(dirs),
        "state": state["compared"] if state else None,
        "info": state["info"] if state else None,
        "logs": logs,
    }


def write_manifest(run_dir: Path, *, impl_root: str, work: str) -> dict:
    m = build_manifest(run_dir, impl_root=impl_root, work=work)
    (Path(run_dir) / "manifest.json").write_text(json.dumps(m, indent=1, sort_keys=True))
    return m


def load_manifest(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "manifest.json").read_text())


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def _digest(entry: dict, policy: dict) -> str:
    if "masked" in entry:
        return entry["masked"]
    mode = policy.get(entry["kind"], "bytes")
    if mode == "decoded" and entry.get("decoded"):
        return entry["decoded"]
    return entry["sha256"]


def compare(a: dict, b: dict, *, policy: dict = POLICY, run_a: Path | None = None,
            run_b: Path | None = None, explain: bool = True) -> list[str]:
    """Differences between two manifests, as human-readable lines ([] = identical).

    With ``run_a`` / ``run_b`` given, a zarr or .pt mismatch is decoded and the
    first differing element reported.
    """
    diffs = []
    da, db = set(a["dirs"]), set(b["dirs"])
    for d in sorted(da - db):
        diffs.append(f"[A] dir only in A: {d}")
    for d in sorted(db - da):
        diffs.append(f"[A] dir only in B: {d}")
    fa, fb = a["files"], b["files"]
    for f in sorted(set(fa) - set(fb)):
        diffs.append(f"[A] file only in A: {f}")
    for f in sorted(set(fb) - set(fa)):
        diffs.append(f"[A] file only in B: {f}")
    explained_zarr = set()
    for f in sorted(set(fa) & set(fb)):
        ea, eb = fa[f], fb[f]
        if _digest(ea, policy) == _digest(eb, policy):
            continue
        cat = "B" if ea["kind"] in ("png", "mp4") else "A"
        mode = policy.get(ea["kind"], "bytes")
        diffs.append(f"[{cat}] {mode} differ: {f} (size {ea['size']} vs {eb['size']})")
        if explain and run_a and run_b:
            detail = None
            if ea["kind"] == "zarr":
                zroot = f.split(".zarr/")[0] + ".zarr"
                if zroot not in explained_zarr:
                    explained_zarr.add(zroot)
                    detail = explain_zarr(Path(run_a) / "data" / zroot, Path(run_b) / "data" / zroot)
            elif ea["kind"] == "pt":
                detail = explain_pt(Path(run_a) / "data" / f, Path(run_b) / "data" / f)
            if detail:
                diffs.append("      " + detail)
    sa, sb = a.get("state") or {}, b.get("state") or {}
    for k in sorted(set(sa) | set(sb)):
        if sa.get(k) != sb.get(k):
            cat = "D" if k == "exception" else "C"
            diffs.append(f"[{cat}] state.{k}: {sa.get(k)!r} vs {sb.get(k)!r}")
    for stream in ("stdout", "stderr"):
        if a["logs"].get(stream) != b["logs"].get(stream):
            line = ""
            if run_a and run_b:
                line = first_log_diff(Path(run_a) / f"{stream}.normalized.txt",
                                      Path(run_b) / f"{stream}.normalized.txt")
            diffs.append(f"[E] {stream} differs after normalisation{line}")
    return diffs


def first_log_diff(pa: Path, pb: Path) -> str:
    la = pa.read_text().splitlines() if pa.exists() else []
    lb = pb.read_text().splitlines() if pb.exists() else []
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return f": line {i + 1}\n      A: {x[:200]!r}\n      B: {y[:200]!r}"
    return f": {len(la)} vs {len(lb)} lines"


def explain_zarr(pa: Path, pb: Path) -> str:
    try:
        import numpy as np
        import tensorstore as ts
        arrs = []
        for p in (pa, pb):
            spec = {"driver": "zarr", "kvstore": {"driver": "file", "path": str(p)}}
            arrs.append(ts.open(spec).result().read().result())
        x, y = arrs
        if x.shape != y.shape:
            return f"decoded shape {x.shape} vs {y.shape}"
        ne = np.argwhere(~((x == y) | (np.isnan(x) & np.isnan(y))))
        if len(ne) == 0:
            return "decoded arrays EQUAL (bytes differ: metadata or compression)"
        first = tuple(int(i) for i in ne[0])
        return (f"decoded: {len(ne)} elements differ; first at {first} "
                f"({x[first]!r} vs {y[first]!r}); max|d|={float(np.nanmax(np.abs(x - y))):.3e}")
    except Exception as e:
        return f"(zarr decode failed: {type(e).__name__}: {e})"


def explain_pt(pa: Path, pb: Path) -> str:
    try:
        import torch

        def flat(obj, prefix=""):
            if isinstance(obj, dict):
                for k in sorted(obj, key=str):
                    yield from flat(obj[k], f"{prefix}{k}.")
            else:
                yield prefix.rstrip("."), obj

        xa = dict(flat(torch.load(pa, weights_only=False, map_location="cpu")))
        xb = dict(flat(torch.load(pb, weights_only=False, map_location="cpu")))
        if set(xa) != set(xb):
            return f".pt keys differ: {sorted(set(xa) ^ set(xb))}"
        out = []
        for k in xa:
            va, vb = xa[k], xb[k]
            if torch.is_tensor(va) and torch.is_tensor(vb):
                sa, sb = va.untyped_storage().size(), vb.untyped_storage().size()
                if sa != sb:
                    out.append(f"{k or '<root>'}: storage bytes {sa} vs {sb} (view saved with its whole storage?)")
                if va.shape != vb.shape or va.dtype != vb.dtype or not torch.equal(va, vb):
                    out.append(f"{k or '<root>'}: values/shape/dtype differ")
            elif va != vb:
                out.append(f"{k}: {va!r} vs {vb!r}")
        return "; ".join(out) or ".pt decoded EQUAL (zip bytes differ)"
    except Exception as e:
        return f"(pt decode failed: {type(e).__name__}: {e})"
