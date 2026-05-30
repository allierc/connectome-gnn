"""Convert Turner-Evans et al. 2020 two-color calcium imaging into the
connectome-aligned ``NeuronTimeSeries`` / zarr format used by the
``drosophila_cx`` pipeline.

Source dataset
--------------
Turner-Evans, Jensen, et al. (2020) *The Neuroanatomical Ultrastructure and
Function of a Biological Ring Attractor*, Neuron. figshare collection
``10.25378/janelia.c.5179721``; two-color item ``10.25378/janelia.12490274``
(``TwoColor.zip``). Each ``.mat`` holds, for one trial:

* ``GROIaveMax`` / ``RROIaveMax`` ``(n_roi, T_img)`` -- green / red channel
  F/F0 per ROI per imaging frame.  ``n_roi`` is 16 for EB recordings (EB
  wedges) and 18 for PB recordings (PB glomeruli).
* ``positionDat`` -- ball-tracker behaviour: ``OffsetRot`` (heading, degrees),
  ``OffsetFor`` / ``OffsetLat``, sampled ~120 Hz; ``tFrameGrab`` (imaging
  frame-grab sync pulses) and ``t`` (behaviour clock, seconds).

The two imaged cell types are named by the containing folder, e.g.
``GreenEPGsRedDelta7s`` -> green channel = EPG, red channel = Delta7.

Output
------
For each trial, a folder under
``graphs_data/drosophila_cx/real_twocolor/<recording>/`` containing:

* ``x_list/``        -- ``NeuronTimeSeries`` zarr, N=156 (the CX connectome),
                        ``fluorescence`` = F/F0 and ``calcium`` = dF/F placed at
                        the recorded neurons, NaN elsewhere; ``voltage`` mirrors
                        ``fluorescence`` so generic observable readers work.
* ``behavior.pt``    -- heading (rad), forward, lateral, angular velocity and
                        per-frame times, all resampled onto the imaging grid.
* ``recorded.pt``    -- ``recorded_mask`` (N,), ``neuron_ix`` (the recorded
                        subset), and the per-neuron ROI source.
* ``meta.json``      -- provenance (source path + md5), region, channel->type
                        map, ROI counts, mapping notes.
* ``Fig/overview.png`` -- kinographs + bump-vs-heading + mapping diagnostic.

ROI -> neuron mapping
---------------------
EPG (46 neurons, reordered to ring topology by the loader) maps to its source
ROIs *exactly* via ``epg_ix`` for 16-wedge EB recordings, and via a documented
18->16 PB-glomerulus fold for PB recordings.  For the partner types
(Delta7 / PEG / PEN) the loader does **not** reorder neurons into glomerular
order, so this script assigns them by even glomerular tiling in connectome
order -- an approximation, flagged here and in ``meta.json``.  The EPG bump
(the canonical heading readout) is always exactly mapped; partner-channel
placement is a starting point the domain expert can refine via
``--pb-drop`` / a custom mapping.

Usage
-----
    PYTHONPATH=src python -m connectome_gnn.generators.real_calcium_twocolor \
        --src   /groups/.../graphs_data/TwoColor \
        --out   /groups/.../graphs_data/drosophila_cx/real_twocolor \
        --hemibrain papers/Code_NN/Code_NN/Data/Figure5/exported-traced-adjacencies-v1.2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import scipy.io as sio  # noqa: E402

from connectome_gnn.neuron_state import NeuronState
from connectome_gnn.zarr_io import ZarrSimulationWriterV3
from connectome_gnn.generators.connconstr_data import load_drosophila_cx_connectome


# --------------------------------------------------------------------------- #
# Folder name -> (green channel cell type, red channel cell type).
# Values are connectome ``type_names`` strings (see load_drosophila_cx_connectome:
#   ['EPG', 'EPGt', 'PEN_a(PEN1)', 'PEN_b(PEN2)', 'Delta7', 'PEG', 'ER6']).
# --------------------------------------------------------------------------- #
GROUP_CHANNELS: dict[str, tuple[str, str]] = {
    "GreenEPGsRedDelta7s": ("EPG", "Delta7"),
    "RedEPGsGreenDelta7s": ("Delta7", "EPG"),
    "GreenPEGsRedPEN2s":   ("PEG", "PEN_b(PEN2)"),
    "RedPEGsGreenPEN2s":   ("PEN_b(PEN2)", "PEG"),
}


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def detect_region(matpath: Path, n_roi: int) -> str:
    """EB / PB from the path, falling back to ROI count (16->EB, 18->PB)."""
    parts = {p.upper() for p in matpath.parts}
    if "EB" in parts:
        return "EB"
    if "PB" in parts:
        return "PB"
    return {16: "EB", 18: "PB"}.get(n_roi, "unknown")


def group_of(matpath: Path) -> str:
    for part in matpath.parts:
        if part in GROUP_CHANNELS:
            return part
    raise ValueError(f"no known two-color group folder in {matpath}")


# --------------------------------------------------------------------------- #
# Time alignment
# --------------------------------------------------------------------------- #
def frame_times_from_grab(t_frame_grab: np.ndarray, n_img: int) -> np.ndarray:
    """Per-imaging-frame timestamps from the sub-frame grab pulses.

    ``tFrameGrab`` carries ``ppf`` sync pulses per imaging frame (14 for the
    750-frame PB recordings, 10 for the 1550-frame EB recordings); we average
    each group of ``ppf`` pulses to get one timestamp per frame.  If the length
    is not an integer multiple we fall back to a uniform grid.
    """
    tfg = np.asarray(t_frame_grab, dtype=float).ravel()
    ppf = len(tfg) // n_img
    if ppf >= 1 and ppf * n_img <= len(tfg):
        return tfg[: ppf * n_img].reshape(n_img, ppf).mean(axis=1)
    return np.linspace(tfg[0], tfg[-1], n_img)


def _norm(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float).ravel()
    span = t.max() - t.min()
    return (t - t.min()) / (span + 1e-12)


def resample_linear(x: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Interp on *normalised* time (frame-grab and behaviour clocks have
    different units but cover the same trial, recorded simultaneously)."""
    return np.interp(_norm(t_dst), _norm(t_src), np.asarray(x, dtype=float).ravel())


def resample_circular(angle_deg: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Circular-safe resampling of a heading angle. Returns radians in (-pi, pi]."""
    a = np.deg2rad(np.asarray(angle_deg, dtype=float).ravel())
    ui = np.interp(_norm(t_dst), _norm(t_src), np.unwrap(a))
    return np.angle(np.exp(1j * ui))


# --------------------------------------------------------------------------- #
# ROI -> neuron mapping
# --------------------------------------------------------------------------- #
def assign_type_rois(
    n_neurons: int,
    n_roi: int,
    *,
    is_epg: bool = False,
    epg_ix: list[int] | None = None,
    pb_drop: tuple[int, int] = (0, 9),
) -> np.ndarray:
    """ROI index feeding each of a cell type's ``n_neurons`` connectome neurons.

    EPG uses the exact 16-bin ``epg_ix``; for 18-ROI PB recordings the 18 PB
    glomeruli are folded to 16 by dropping the two outermost (``pb_drop``,
    default the L9/R9 ends) before applying ``epg_ix``.  All other types are
    tiled evenly across ``n_roi`` in connectome order (approximate -- see module
    docstring).
    """
    if is_epg and epg_ix is not None and len(epg_ix) == n_neurons:
        if n_roi == 16:
            return np.asarray(epg_ix, dtype=np.int64)
        if n_roi == 18:
            keep = [i for i in range(18) if i not in pb_drop]  # 16 glomeruli
            if len(keep) == 16:
                return np.asarray([keep[g] for g in epg_ix], dtype=np.int64)
    # even tiling: neuron k -> roi floor(k * n_roi / n_neurons)
    return (np.arange(n_neurons) * n_roi // max(n_neurons, 1)).astype(np.int64)


def build_mapping(
    cx: dict,
    green_type: str,
    red_type: str,
    n_roi_green: int,
    n_roi_red: int,
    pb_drop: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-neuron (src_channel, src_roi, recorded_mask).

    src_channel: 0 = green, 1 = red, -1 = not recorded.
    src_roi:     ROI index within that channel, or -1.
    """
    N = int(cx["N"])
    nt = np.asarray(cx["neuron_types"])
    names = list(cx["type_names"])
    epg_ix = cx["epg_ix"]

    src_channel = np.full(N, -1, dtype=np.int64)
    src_roi = np.full(N, -1, dtype=np.int64)

    for channel, tname, n_roi in [(0, green_type, n_roi_green), (1, red_type, n_roi_red)]:
        if tname not in names:
            raise ValueError(f"cell type {tname!r} not in connectome types {names}")
        tid = names.index(tname)
        idx = np.where(nt == tid)[0]  # connectome-order neurons of this type
        rois = assign_type_rois(
            len(idx), n_roi,
            is_epg=(tname == "EPG"), epg_ix=epg_ix, pb_drop=pb_drop,
        )
        src_channel[idx] = channel
        src_roi[idx] = rois

    recorded = src_channel >= 0
    return src_channel, src_roi, recorded


def neuron_layout(cx: dict) -> np.ndarray:
    """A simple 2D layout for plotting: EPG on the ring by glomerulus, other
    types on concentric rings."""
    N = int(cx["N"])
    nt = np.asarray(cx["neuron_types"])
    names = list(cx["type_names"])
    epg_ix = cx["epg_ix"]
    pos = np.zeros((N, 2), dtype=np.float32)
    epg_id = names.index("EPG")
    epg_idx = np.where(nt == epg_id)[0]
    for k, j in enumerate(epg_idx):
        ang = epg_ix[k] / 16.0 * 2 * np.pi
        pos[j] = [np.cos(ang), np.sin(ang)]
    for tid, _ in enumerate(names):
        if tid == epg_id:
            continue
        idx = np.where(nt == tid)[0]
        r = 1.3 + 0.22 * tid
        for k, j in enumerate(idx):
            ang = k / max(len(idx), 1) * 2 * np.pi
            pos[j] = [r * np.cos(ang), r * np.sin(ang)]
    return pos


# --------------------------------------------------------------------------- #
# Per-recording conversion
# --------------------------------------------------------------------------- #
def parse_mat(matpath: Path, roi_stat: str) -> dict | None:
    try:
        m = sio.loadmat(str(matpath), struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        warnings.warn(f"skipping v7.3/HDF5 .mat (needs h5py path): {matpath}")
        return None
    gkey, rkey = f"G{roi_stat}", f"R{roi_stat}"  # e.g. GROIaveMax
    if gkey not in m or rkey not in m or "positionDat" not in m:
        warnings.warn(f"skipping {matpath}: missing {gkey}/{rkey}/positionDat")
        return None
    g = np.asarray(m[gkey], dtype=np.float32)
    r = np.asarray(m[rkey], dtype=np.float32)
    pd = m["positionDat"]
    return {
        "green": g, "red": r,
        "t": np.asarray(pd.t, dtype=float).ravel(),
        "OffsetRot": np.asarray(pd.OffsetRot, dtype=float).ravel(),
        "OffsetFor": np.asarray(pd.OffsetFor, dtype=float).ravel(),
        "OffsetLat": np.asarray(pd.OffsetLat, dtype=float).ravel(),
        "tFrameGrab": np.asarray(pd.tFrameGrab, dtype=float).ravel(),
        "fullpath": str(getattr(m, "fullpath", m.get("fullpath", ""))),
    }


def dff(F: np.ndarray, q: float = 10.0) -> np.ndarray:
    """dF/F with a per-ROI baseline = q-th percentile over time. F is (n_roi, T)."""
    base = np.percentile(F, q, axis=1, keepdims=True)
    return (F - base) / (np.abs(base) + 1e-6)


def convert_recording(
    matpath: Path,
    out_root: Path,
    cx: dict,
    pos: np.ndarray,
    *,
    roi_stat: str,
    pb_drop: tuple[int, int],
) -> dict | None:
    rec = parse_mat(matpath, roi_stat)
    if rec is None:
        return None

    group = group_of(matpath)
    green_type, red_type = GROUP_CHANNELS[group]
    green, red = rec["green"], rec["red"]          # (n_roi, T_img)
    n_roi_g, T = green.shape
    n_roi_r, T_r = red.shape
    if T_r != T:
        T = min(T, T_r)
        green, red = green[:, :T], red[:, :T]
    region = detect_region(matpath, n_roi_g)

    # ---- ROI -> neuron mapping --------------------------------------------
    N = int(cx["N"])
    src_channel, src_roi, recorded = build_mapping(
        cx, green_type, red_type, n_roi_g, n_roi_r, pb_drop
    )

    # ---- place channel F/F0 on the 156-neuron axis (NaN where unrecorded) --
    F = np.full((T, N), np.nan, dtype=np.float32)
    dff_full = np.full((T, N), np.nan, dtype=np.float32)
    green_dff, red_dff = dff(green), dff(red)
    gmask = src_channel == 0
    rmask = src_channel == 1
    F[:, gmask] = green[src_roi[gmask], :].T
    F[:, rmask] = red[src_roi[rmask], :].T
    dff_full[:, gmask] = green_dff[src_roi[gmask], :].T
    dff_full[:, rmask] = red_dff[src_roi[rmask], :].T

    # ---- behaviour resampled onto the imaging-frame grid ------------------
    frame_t = frame_times_from_grab(rec["tFrameGrab"], T)
    heading = resample_circular(rec["OffsetRot"], rec["t"], frame_t)  # rad
    forward = resample_linear(rec["OffsetFor"], rec["t"], frame_t)
    lateral = resample_linear(rec["OffsetLat"], rec["t"], frame_t)
    ang_vel = np.gradient(np.unwrap(heading))  # rad / frame

    # ---- recording name / output folder -----------------------------------
    stem = matpath.stem
    date = matpath.parent.name
    name = f"{group}__{date}__{stem}"
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "Fig").mkdir(exist_ok=True)

    # ---- write NeuronTimeSeries zarr (x_list) -----------------------------
    neuron_type = torch.as_tensor(np.asarray(cx["neuron_types"]), dtype=torch.long)
    group_type = torch.zeros(N, dtype=torch.long)
    pos_t = torch.as_tensor(pos, dtype=torch.float32)
    index = torch.arange(N, dtype=torch.long)
    zeros = torch.zeros(N, dtype=torch.float32)

    writer = ZarrSimulationWriterV3(
        path=out_dir / "x_list", n_neurons=N, time_chunks=2000, save_calcium=True,
    )
    for t in range(T):
        f_t = torch.as_tensor(F[t], dtype=torch.float32)
        writer.append_state(NeuronState(
            index=index, pos=pos_t, group_type=group_type, neuron_type=neuron_type,
            voltage=f_t,                                   # observable mirrors F/F0
            stimulus=zeros,                                # visual stim unknown
            calcium=torch.as_tensor(dff_full[t], dtype=torch.float32),
            fluorescence=f_t,
            noise=zeros,
        ))
    n_written = writer.finalize()

    # ---- behaviour + recorded mask + metadata -----------------------------
    neuron_ix = np.where(recorded)[0]
    torch.save({
        "heading_rad": torch.as_tensor(heading, dtype=torch.float32),
        "forward": torch.as_tensor(forward, dtype=torch.float32),
        "lateral": torch.as_tensor(lateral, dtype=torch.float32),
        "ang_vel": torch.as_tensor(ang_vel, dtype=torch.float32),
        "frame_time": torch.as_tensor(frame_t, dtype=torch.float32),
    }, out_dir / "behavior.pt")
    torch.save({
        "recorded_mask": torch.as_tensor(recorded, dtype=torch.bool),
        "neuron_ix": torch.as_tensor(neuron_ix, dtype=torch.long),
        "src_channel": torch.as_tensor(src_channel, dtype=torch.long),
        "src_roi": torch.as_tensor(src_roi, dtype=torch.long),
        "green_type": green_type, "red_type": red_type,
    }, out_dir / "recorded.pt")

    meta = {
        "source_mat": str(matpath),
        "source_md5": md5_of(matpath),
        "device_fullpath": rec["fullpath"],
        "group": group, "region": region,
        "green_type": green_type, "red_type": red_type,
        "n_roi_green": int(n_roi_g), "n_roi_red": int(n_roi_r),
        "n_frames": int(n_written), "n_neurons": int(N),
        "n_recorded": int(neuron_ix.size),
        "roi_stat": roi_stat, "pb_drop": list(pb_drop),
        "condition": "Dark" if "Dark" in stem else ("Stripe" if "Stripe" in stem else "Other"),
        "mapping_note": (
            "EPG mapped exactly via epg_ix (16-wedge EB; 18->16 PB fold). "
            "Partner type tiled evenly in connectome order (approximate)."
        ),
        "fields": {
            "fluorescence": "F/F0 (ROIaveMax) at recorded neurons, NaN elsewhere",
            "calcium": "dF/F (per-ROI 10th-pct baseline)",
            "voltage": "mirror of fluorescence (observable convenience)",
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    _plot_overview(out_dir / "Fig" / "overview.png", green, red, heading, frame_t,
                   src_channel, src_roi, recorded, cx, name, green_type, red_type, region)

    return meta


# --------------------------------------------------------------------------- #
# Diagnostics plot
# --------------------------------------------------------------------------- #
def _pva_phase(channel_roi: np.ndarray) -> np.ndarray:
    """Population-vector-average phase (rad) over ROIs treated as a ring."""
    n_roi = channel_roi.shape[0]
    ang = np.arange(n_roi) / n_roi * 2 * np.pi
    z = (np.exp(1j * ang)[:, None] * np.maximum(channel_roi, 0)).sum(axis=0)
    return np.angle(z)


def _plot_overview(path, green, red, heading, frame_t, src_channel, src_roi,
                   recorded, cx, name, green_type, red_type, region):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{name}\n[{region}] green={green_type}  red={red_type}", fontsize=10)

    # kinographs
    for a, data, ttl in [(ax[0, 0], green, f"green: {green_type}"),
                         (ax[0, 1], red, f"red: {red_type}")]:
        im = a.imshow(data, aspect="auto", origin="lower", cmap="viridis",
                      extent=[0, data.shape[1], 0, data.shape[0]])
        a.set_title(ttl); a.set_xlabel("imaging frame"); a.set_ylabel("ROI")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, label="F/F0")
    # heading overlaid on the green kinograph in ROI coordinate
    hd_roi = (heading + np.pi) / (2 * np.pi) * green.shape[0]
    ax[0, 0].plot(np.arange(len(hd_roi)), hd_roi, "r.", ms=1.0, alpha=0.5, label="heading")
    ax[0, 0].legend(loc="upper right", fontsize=7)

    # bump (EPG-channel PVA) vs heading
    # pick whichever channel is EPG; else use green
    epg_channel = green if green_type == "EPG" else (red if red_type == "EPG" else green)
    bump = _pva_phase(epg_channel)
    ax[1, 0].plot(np.rad2deg(heading), ".", ms=1.5, label="heading", color="k")
    ax[1, 0].plot(np.rad2deg(bump), ".", ms=1.5, label="bump (PVA)", color="tab:green")
    ax[1, 0].set_title("bump phase vs heading"); ax[1, 0].set_xlabel("imaging frame")
    ax[1, 0].set_ylabel("angle (deg)"); ax[1, 0].legend(fontsize=7)

    # mapping diagnostic: neuron index -> source ROI, coloured by channel
    N = len(src_channel)
    col = np.where(src_channel == 0, 0.0, np.where(src_channel == 1, 1.0, np.nan))
    a = ax[1, 1]
    for ch, c, lab in [(0, "tab:green", f"green={green_type}"),
                       (1, "tab:red", f"red={red_type}")]:
        msk = src_channel == ch
        a.scatter(np.arange(N)[msk], src_roi[msk], s=10, c=c, label=lab)
    a.set_title(f"ROI->neuron mapping (recorded={int(recorded.sum())}/{N})")
    a.set_xlabel("connectome neuron index"); a.set_ylabel("source ROI"); a.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=110)
    plt.close(fig)


MAPPING_README = """\
# real_twocolor — Turner-Evans 2020 two-color calcium, connectome-aligned

Each subfolder is one imaging trial converted to the `drosophila_cx`
`NeuronTimeSeries` format (N=156).

## Per-recording files
- `x_list/`     zarr NeuronTimeSeries. `fluorescence` = F/F0, `calcium` = dF/F,
                `voltage` = mirror of `fluorescence`, all NaN at non-recorded
                neurons. `neuron_type` = CX connectome types.
- `behavior.pt` heading_rad, forward, lateral, ang_vel, frame_time (imaging grid).
- `recorded.pt` recorded_mask (N,), neuron_ix (recorded subset), src_channel,
                src_roi. **Use `neuron_ix` as the voltage-supervision recorded
                subset** (REFACTOR_drosophila_voltage.md, `voltage_ix`).
- `meta.json`   provenance (source path + md5), region, channel->type, mapping.

## Mapping caveats (read before trusting partner-channel placement)
- **EPG** is mapped exactly via `epg_ix` (16 EB wedges; 18 PB glomeruli folded
  to 16 by dropping the outermost pair, `--pb-drop`).
- **Delta7 / PEG / PEN** are NOT reordered to glomerular order by the connectome
  loader, so they are tiled evenly across ROIs in connectome index order. This
  is an approximation — validate against `Fig/overview.png` and refine if needed.
- Behaviour is resampled on *normalised* time (imaging and ball clocks share the
  trial window); fine sub-frame alignment is not reconstructed.
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="root of extracted TwoColor/ tree")
    p.add_argument("--out", required=True, help="output root (…/drosophila_cx/real_twocolor)")
    p.add_argument("--hemibrain", required=True,
                   help="hemibrain CSV dir (…/exported-traced-adjacencies-v1.2)")
    p.add_argument("--roi-stat", default="ROIaveMax", choices=["ROIaveMax", "ROIaveMean"])
    p.add_argument("--pb-drop", type=int, nargs=2, default=(0, 9),
                   help="PB glomerulus indices dropped in the 18->16 EPG fold")
    p.add_argument("--groups", nargs="*", default=None,
                   help="restrict to these group folders (default: all known)")
    p.add_argument("--limit", type=int, default=0, help="convert at most N recordings (0=all)")
    p.add_argument("--dry-run", action="store_true", help="list recordings, do not write")
    args = p.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    cx = load_drosophila_cx_connectome(args.hemibrain)
    pos = neuron_layout(cx)

    groups = args.groups or list(GROUP_CHANNELS)
    mats = sorted(
        m for g in groups for m in (src / g).rglob("*.mat")
    )
    if args.limit:
        mats = mats[: args.limit]

    print(f"found {len(mats)} .mat recordings under {src}")
    if args.dry_run:
        for m in mats:
            print("  ", m.relative_to(src))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "mapping_README.md").write_text(MAPPING_README)

    rows = []
    for i, m in enumerate(mats):
        try:
            meta = convert_recording(
                m, out, cx, pos,
                roi_stat=args.roi_stat, pb_drop=tuple(args.pb_drop),
            )
        except Exception as e:  # keep going; report at the end
            warnings.warn(f"FAILED {m}: {e!r}")
            meta = None
        tag = "ok" if meta else "skip"
        print(f"[{i + 1}/{len(mats)}] {tag}: {m.relative_to(src)}")
        if meta:
            rows.append(meta)

    if rows:
        cols = ["group", "region", "green_type", "red_type", "n_roi_green",
                "n_roi_red", "n_frames", "n_recorded", "condition",
                "source_md5", "source_mat"]
        with open(out / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print(f"\nconverted {len(rows)}/{len(mats)} -> {out}")
    print(f"manifest: {out / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
