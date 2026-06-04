"""Build a calcium-observation training dataset from a real ZAPBench sequence.

Companion to the synthetic ``swim_integration`` generator
(``graph_data_generator._generate_swim_integration_task``): instead of sampling
synthetic angular-velocity impulses, this slices the REAL recorded ZAPBench
stimulus (e.g. the "Rotations" 45 deg/s grating block) into the same 10 s,
1000-frame trials the curriculum trains on, and attaches the per-neuron recorded
ΔF/F as an extra ``calcium`` field next to the standard TaskTrials fields.

Why a separate dataset (not a flag on the swim generator):
  * dataset A = synthetic swim_integration  -> heading task only (unchanged)
  * dataset B = THIS                          -> heading task + real calcium

The on-disk layout is byte-format-consistent with the swim dataset
(``stimulus/target/theta_hd/is_stop/omega`` zarrs via ``ZarrTaskTrialsWriter``)
plus one extra ``calcium.zarr`` written in the identical blosc/zstd format —
the same way the swim generator writes its extra ``swim_label.zarr``. A
``calcium_meta.npz`` sidecar carries the per-observed-neuron bodyId/type/side
and the global normalisation stats so the trainer can map the 481 observed
columns onto model-neuron indices (by bodyId) at train time.

No warmup: each window starts fresh and carries its initial heading as the
``(cosθ0, sinθ0)`` cue on frame 0, exactly like a synthetic trial, so the model
anchors its bump from the cue and integrates the recorded ω.

The real recorded ΔF/F is sampled at the imaging rate (~1.09 Hz, 0.915 s); it is
linearly interpolated up to the model grid (dt=0.01) with edge-clamped padding,
then z-scored per neuron over the WHOLE block (global, not per trial). When the
model is trained, a small MLP learns the residual model-calcium -> observed-ΔF/F
gain/offset, so this global z-score only needs to be stable, not exact.

Usage
-----
    python -m connectome_gnn.generators.make_calcium_dataset \
        --sequence rotation \
        --connectome figures/zebrafish/zebrafish_connectome_HD_IPN12 \
        --out $GNN_OUTPUT_ROOT/graphs_data/zebrafish/zebrafish_hd_si_calcium_rotation_ipn12_v1 \
        --n-trials-train 2000 --n-trials-test 200

Then inspect the printed summary and the ``*.png`` figures written next to the
``train/`` and ``test/`` zarr folders.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# repo root: src/connectome_gnn/generators/make_calcium_dataset.py -> up 4
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_CONNECTOME = os.path.join(_REPO, "figures", "zebrafish",
                                   "zebrafish_connectome_HD_IPN12")
_DEFAULT_FISHFUNCEM = os.path.join(_REPO, "papers", "fishFuncEM", "data")


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------
def load_sequence_heading(sequence, connectome_dir, fishfuncem_data, model_dt,
                          src_dt=0.915):
    """Return (theta_hr_deg, frames, theta_frame_deg) for a ZAPBench sequence.

    theta_hr_deg  — heading (deg) on the MODEL grid (model_dt).
    frames        — imaging-frame indices of the block (into the full ΔF/F).
    theta_frame   — heading (deg) at the imaging frames.

    Rotation is read from the cached ``functional/rotation_heading.npz`` when
    present (offline); otherwise rebuilt via zapbench_stimulus.rotation_headings
    (needs fishfuncem + a one-off 288 MB GCS read).
    """
    if sequence != "rotation":
        raise NotImplementedError(
            f"sequence={sequence!r} not implemented yet. Only 'rotation' is "
            "wired up (cache at functional/rotation_heading.npz). To add e.g. "
            "'turning', mirror zapbench_stimulus.rotation_headings for that "
            "onset block and drop a <sequence>_heading.npz cache here.")

    cache = os.path.join(connectome_dir, "functional", "rotation_heading.npz")
    if os.path.isfile(cache):
        z = np.load(cache)
        if abs(float(z["model_dt"]) - model_dt) < 1e-9:
            print(f"[heading] loaded cache {cache}")
            return (z["theta_hr"].astype(np.float64),
                    z["frames"].astype(np.int64),
                    z["theta_frame"].astype(np.float64))
        print(f"[heading] cache model_dt={float(z['model_dt'])} != {model_dt}; "
              "rebuilding")

    from connectome_gnn.generators.zapbench_stimulus import rotation_headings
    theta_hr, frames, theta_frame = rotation_headings(
        fishfuncem_data, connectome_dir, model_dt, src_dt=src_dt)
    return (np.asarray(theta_hr, np.float64), np.asarray(frames, np.int64),
            np.asarray(theta_frame, np.float64))


def load_calcium_traces(connectome_dir):
    """Load per-neuron recorded ΔF/F + identity from circuit_functional_traces.npz.

    Returns dict: traces (T_full, R), bodyId (R,), zapbenchId (R,),
    type (R,), side (R,).
    """
    p = os.path.join(connectome_dir, "functional", "circuit_functional_traces.npz")
    z = np.load(p, allow_pickle=True)
    traces = np.asarray(z["traces"], np.float32)            # (T_full, R)
    print(f"[calcium] {p}: traces {traces.shape} "
          f"({traces.shape[1]} observed neurons)")
    return dict(traces=traces,
                bodyId=np.asarray(z["bodyId"], np.int64),
                zapbenchId=np.asarray(z["zapbenchId"], np.int64),
                type=np.asarray(z["type"]).astype(str),
                side=np.asarray(z["side"]).astype(str))


# ---------------------------------------------------------------------------
# Full-block model-grid construction
# ---------------------------------------------------------------------------
def build_full_block(theta_hr_deg, frames, traces, model_dt, src_dt):
    """Assemble model-grid signals for the whole sequence block.

    Returns dict of float32 arrays over T_model steps:
      omega       (T_model,)        deg/s   = d(heading)/dt
      heading_rad (T_model,)        rad
      target      (T_model, 2)      [cos, sin](heading)
      is_stop     (T_model,)        1.0 where |omega| ~ 0
      calcium     (T_model, R)      per-neuron z-scored ΔF/F (interp to model grid)
      ca_mean, ca_std (R,)          per-neuron norm stats (over the block)
    """
    T_model = theta_hr_deg.shape[0]
    t_model = np.arange(T_model) * model_dt                 # [0, dur)

    heading_deg = theta_hr_deg
    heading_rad = np.deg2rad(heading_deg).astype(np.float32)
    omega = np.gradient(heading_deg, model_dt).astype(np.float32)   # deg/s
    target = np.stack([np.cos(heading_rad), np.sin(heading_rad)], -1).astype(np.float32)
    is_stop = (np.abs(omega) < 1e-3).astype(np.float32)

    # Real ΔF/F at the imaging frames of this block -> interp up to model grid.
    ca_img = traces[frames]                                 # (n_fr, R)
    n_fr, R = ca_img.shape
    t_img = np.arange(n_fr) * src_dt                        # imaging-frame times
    ca_model = np.empty((T_model, R), np.float32)
    for r in range(R):                                      # np.interp edge-clamps -> padding
        ca_model[:, r] = np.interp(t_model, t_img, ca_img[:, r])

    # Global (per-neuron, over the whole block) z-score — NOT per trial.
    ca_mean = ca_model.mean(0)
    ca_std = ca_model.std(0)
    ca_std_safe = np.where(ca_std < 1e-6, 1.0, ca_std)
    ca_norm = ((ca_model - ca_mean) / ca_std_safe).astype(np.float32)

    print(f"[block] T_model={T_model} ({T_model * model_dt:.1f}s), "
          f"imaging frames={n_fr}, |omega| max={np.abs(omega).max():.1f} deg/s, "
          f"stop fraction={is_stop.mean():.2f}")
    return dict(omega=omega, heading_rad=heading_rad, target=target,
                is_stop=is_stop, calcium=ca_norm,
                ca_mean=ca_mean.astype(np.float32),
                ca_std=ca_std.astype(np.float32))


# ---------------------------------------------------------------------------
# Window sampling (time-disjoint train/test)
# ---------------------------------------------------------------------------
def sample_starts(T_model, n_steps, n_train, n_test, seed, test_frac=0.15):
    """Sample window start indices for train/test.

    The FIRST ``n_tile = T_model // n_steps`` train windows are DETERMINISTIC,
    consecutive, non-overlapping tiles ``[0, n_steps, 2·n_steps, …]`` that pave
    the whole block end-to-end. Rolling the trained model on these tiles and
    concatenating the per-trial voltage reconstructs the full ~600 s recording
    (this is what the ``-o test`` calcium-reconstruction panel does — see
    ``graph_tester.data_test_path_integration_task`` section (e)). Because they
    pave the whole block they intentionally span the late (test) region too:
    they are a reconstruction probe, NOT held-out data.

    The remaining ``n_train − n_tile`` train windows, and ALL test windows, keep
    the original time-disjoint random sampling: the block is split in time at
    (1 − test_frac); the random train windows start in the early region, test
    windows in the late region, so overlapping 10 s random windows never leak
    across the split.

    Returns ``(train_starts, test_starts, n_tile)``; ``train_starts[:n_tile]``
    are the consecutive block tiles, in time order.
    """
    last_start = T_model - n_steps
    if last_start <= 0:
        raise ValueError(f"block ({T_model}) shorter than n_steps ({n_steps})")
    rng = np.random.default_rng(seed)

    # First n_tile train windows: consecutive tiles paving [0, n_tile·n_steps).
    n_tile = int(min(T_model // n_steps, n_train))
    tile_starts = np.arange(n_tile, dtype=np.int64) * n_steps

    split = int(round(last_start * (1.0 - test_frac)))
    n_rand_train = max(n_train - n_tile, 0)
    rand_train = rng.integers(0, split + 1, size=n_rand_train, dtype=np.int64)
    train = np.concatenate([tile_starts, rand_train])
    # test starts must not let a window reach back into train territory either;
    # they live entirely in [split+n_steps, last_start] when room allows.
    test_lo = min(split + n_steps, last_start)
    test = rng.integers(test_lo, last_start + 1, size=n_test, dtype=np.int64)
    print(f"[windows] first {n_tile} train = consecutive tiles paving "
          f"[0,{n_tile * n_steps}) (full-block reconstruction set); "
          f"split @ step {split}/{last_start}; "
          f"{n_rand_train} random train starts in [0,{split}], "
          f"test starts in [{test_lo},{last_start}]")
    return train, test, n_tile


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def write_split(split_dir, starts, block, n_steps, model_dt, calcium_chunk=1000):
    """Write one split (train/ or test/) in the swim-dataset zarr format + calcium.zarr."""
    import tensorstore as ts

    from connectome_gnn.task_state import TaskState
    from connectome_gnn.zarr_io import ZarrTaskTrialsWriter

    import torch

    omega = block["omega"]
    heading_rad = block["heading_rad"]
    target = block["target"]
    is_stop = block["is_stop"]
    calcium = block["calcium"]
    R = calcium.shape[1]
    B = len(starts)

    # --- standard TaskTrials fields via the streaming writer ---------------
    writer = ZarrTaskTrialsWriter(split_dir, chunk_trials=calcium_chunk)
    ca_windows = np.empty((B, n_steps, R), np.float32)
    for b, s in enumerate(starts):
        sl = slice(int(s), int(s) + n_steps)
        stim = np.zeros((n_steps, 3), np.float32)
        stim[:, 0] = omega[sl]                              # ch0 = ω(deg/s)
        stim[0, 1] = np.cos(heading_rad[int(s)])            # cue: cosθ0 @ t=0
        stim[0, 2] = np.sin(heading_rad[int(s)])            # cue: sinθ0 @ t=0
        st = TaskState(
            task_family="swim_integration", n_input=3, n_output=2, dt=model_dt,
            stimulus=torch.from_numpy(stim),
            target=torch.from_numpy(target[sl].copy()),
            theta_hd=torch.from_numpy(heading_rad[sl].copy()),
            is_stop=torch.from_numpy(is_stop[sl].copy()),
            omega=torch.from_numpy(omega[sl].copy()),
        )
        writer.append_trial(st)
        ca_windows[b] = calcium[sl]
    n_written = writer.finalize()

    # --- extra calcium.zarr in the identical blosc/zstd format -------------
    ca_path = os.path.join(split_dir, "calcium.zarr")
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": ca_path},
        "metadata": {
            "dtype": "<f4",
            "shape": [B, n_steps, R],
            "chunks": [min(calcium_chunk, B), n_steps, R],
            "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2},
        },
        "create": True, "delete_existing": True,
    }
    store = ts.open(spec).result()
    store[:].write(ca_windows).result()

    # --- augment meta.json with calcium bookkeeping ------------------------
    meta_path = os.path.join(split_dir, "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    meta["calcium"] = {"field": "calcium.zarr", "n_obs": int(R),
                       "shape": [int(B), int(n_steps), int(R)],
                       "norm": "per_neuron_zscore_global"}
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    return n_written, ca_windows


def write_calcium_meta(out_dir, ca, block, sequence, model_dt, src_dt,
                       n_block_tiles=0, n_steps=0):
    """Sidecar identifying the observed-neuron columns + norm stats.

    ``n_block_tiles`` / ``n_steps`` record that the first ``n_block_tiles`` train
    trials are consecutive ``n_steps``-frame tiles that pave the whole block, so
    ``-o test`` can reassemble them into the full ~600 s reconstruction.
    """
    np.savez(os.path.join(out_dir, "calcium_meta.npz"),
             bodyId=ca["bodyId"], zapbenchId=ca["zapbenchId"],
             type=ca["type"], side=ca["side"],
             ca_mean=block["ca_mean"], ca_std=block["ca_std"],
             sequence=np.array(sequence), model_dt=np.float32(model_dt),
             src_dt=np.float32(src_dt),
             n_block_tiles=np.int64(n_block_tiles),
             n_steps=np.int64(n_steps),
             block_T_model=np.int64(block["calcium"].shape[0]))


def write_mapping(out_dir, ca, circuit_name, connectome_dir):
    """Save calcium_mapping.pt: observed-column <-> model-neuron index map +
    the rastermap row ordering used to compare real vs learned kinographs.

    The ``calcium`` columns (the 481 observed neurons, in calcium.zarr order)
    only partly overlap the model: ``model_index[c]`` is the model-neuron index
    for observed column c (``-1`` if that bodyId is not in the circuit), and
    ``shared_mask`` flags the observed columns the trainer can supervise.

    ``kino_*`` give the bump-pool comparison row set in the EXACT order the
    functional panels plot — reusing the panel's ``build_rows`` /
    ``sort_rows_rastermap`` so real and learned kinographs line up row-for-row.
    """
    import sys

    import torch

    from connectome_gnn.generators.circuits import get_circuit

    # observed-column -> model-neuron index (by bodyId)
    c = get_circuit(circuit_name)
    body_ids = np.asarray(c.body_ids, dtype=np.int64)
    idx_of = {int(b): i for i, b in enumerate(body_ids)}
    obs_body = ca["bodyId"]
    model_index = np.array([idx_of.get(int(b), -1) for b in obs_body], np.int64)
    shared_mask = model_index >= 0
    obs_pos = {int(b): i for i, b in enumerate(obs_body)}      # bodyId -> obs column

    # rastermap-ordered bump-pool rows, straight from the panel (single source)
    fig_dir = os.path.join(_REPO, "figures", "zebrafish")
    if fig_dir not in sys.path:
        sys.path.insert(0, fig_dir)
    import zebrafish_functional_traces_panel as panel        # noqa: E402

    rows, _ = panel.build_rows(connectome_dir, circuit_name)
    rows = panel.sort_rows_rastermap(rows)
    kino_obs_index = np.array(
        [obs_pos.get(int(b), -1) for b in rows["bodyId"].to_numpy()], np.int64)

    mapping = {
        "circuit_name": circuit_name,
        "n_model": int(body_ids.shape[0]),
        # full observed-column -> model map (calcium.zarr column order)
        "obs_bodyId": torch.from_numpy(obs_body.astype(np.int64)),
        "obs_zapbenchId": torch.from_numpy(ca["zapbenchId"].astype(np.int64)),
        "obs_type": list(ca["type"]),
        "obs_side": list(ca["side"]),
        "model_index": torch.from_numpy(model_index),
        "shared_mask": torch.from_numpy(shared_mask),
        # bump-pool kinograph comparison rows, in plot (rastermap) order
        "kino_order": "rastermap",
        "kino_bodyId": torch.from_numpy(rows["bodyId"].to_numpy().astype(np.int64)),
        "kino_model_index": torch.from_numpy(rows["model_index"].to_numpy().astype(np.int64)),
        "kino_zapbenchId": torch.from_numpy(rows["zapbenchId"].to_numpy().astype(np.int64)),
        "kino_obs_index": torch.from_numpy(kino_obs_index),     # -1 where unobserved
        "kino_matched": torch.from_numpy(rows["matched"].to_numpy().astype(bool)),
        "kino_type": list(rows["type"].astype(str)),
        "kino_side": list(rows["side"].astype(str)),
    }
    path = os.path.join(out_dir, "calcium_mapping.pt")
    torch.save(mapping, path)
    n_kino_obs = int((kino_obs_index >= 0).sum())
    print(f"[mapping] {path}: {int(shared_mask.sum())}/{len(obs_body)} observed "
          f"columns map into model '{circuit_name}' (N={body_ids.shape[0]}); "
          f"kinograph rows={len(rows)} ({n_kino_obs} observed bump-pool neurons "
          f"shared for real-vs-learned compare)")
    return mapping


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _wrap_deg(a_deg):
    """Wrap heading to [-180, 180); NaN at the wrap discontinuities so the
    plotted trace doesn't draw vertical jump lines."""
    w = (((np.asarray(a_deg, np.float64) + 180.0) % 360.0) - 180.0)
    if w.size > 1:
        w[1:][np.abs(np.diff(w)) > 180.0] = np.nan
    return w


def plot_block_kinograph(out_png, block, model_dt, sequence, max_rows=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ca = block["calcium"]                                   # (T, R)
    T, R = ca.shape
    rows = np.linspace(0, R - 1, min(R, max_rows)).astype(int)
    t = np.arange(T) * model_dt
    fig, axs = plt.subplots(3, 1, figsize=(11, 6), height_ratios=[3, 1, 1],
                            sharex=True)
    axs[0].imshow(ca[:, rows].T, aspect="auto", cmap="viridis",
                  extent=[t[0], t[-1], R, 0], vmin=-2, vmax=4,
                  interpolation="nearest")
    axs[0].set_ylabel(f"observed neurons (n={R})")
    axs[0].set_title(f"REAL ΔF/F (z-scored, interp to model grid) — {sequence}")
    axs[1].plot(t, block["omega"], "k", lw=0.6)
    axs[1].set_ylabel("ω (deg/s)")
    hd = _wrap_deg(np.rad2deg(block["heading_rad"]))        # [-180, 180]
    axs[2].plot(t, hd, "g", lw=0.6)
    axs[2].set_ylim(-180, 180)
    axs[2].set_ylabel("heading (deg)")
    axs[2].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"[fig] {out_png}")


def plot_example_trials(out_png, ca_windows, split_dir, model_dt, n_show=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import zarr

    stim = np.asarray(zarr.open(os.path.join(split_dir, "stimulus.zarr"), "r"))
    tgt = np.asarray(zarr.open(os.path.join(split_dir, "target.zarr"), "r"))
    B, T, R = ca_windows.shape
    n_show = min(n_show, B)
    t = np.arange(T) * model_dt
    fig, axs = plt.subplots(3, n_show, figsize=(3.2 * n_show, 6),
                            sharex=True)
    if n_show == 1:
        axs = axs[:, None]
    for j in range(n_show):
        axs[0, j].imshow(ca_windows[j].T, aspect="auto", cmap="viridis",
                         extent=[t[0], t[-1], R, 0], vmin=-2, vmax=4,
                         interpolation="nearest")
        axs[0, j].set_title(f"trial {j}")
        axs[1, j].plot(t, stim[j, :, 0], "k", lw=0.7)
        hd = np.rad2deg(np.arctan2(tgt[j, :, 1], tgt[j, :, 0]))
        axs[2, j].plot(t, hd, "g", lw=0.7)
        axs[2, j].set_xlabel("time (s)")
    axs[0, 0].set_ylabel(f"calcium (n={R})")
    axs[1, 0].set_ylabel("ω (deg/s)")
    axs[2, 0].set_ylabel("heading (deg)")
    fig.suptitle("example calcium-observation trials (10 s windows)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"[fig] {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", default="rotation",
                    help="ZAPBench stimulus block (only 'rotation' wired up)")
    ap.add_argument("--connectome", default=_DEFAULT_CONNECTOME,
                    help="circuit dir holding functional/{circuit_functional_traces,rotation_heading}.npz")
    ap.add_argument("--circuit", default="zebrafish_HD_IPN12_839_v1",
                    help="registered circuit name for the obs->model index mapping")
    ap.add_argument("--fishfuncem", default=_DEFAULT_FISHFUNCEM,
                    help="fishFuncEM data dir (only needed if the heading cache is missing)")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--n-trials-train", type=int, default=2000)
    ap.add_argument("--n-trials-test", type=int, default=200)
    ap.add_argument("--n-steps", type=int, default=1000, help="frames per trial (1000 = 10 s)")
    ap.add_argument("--dt", type=float, default=0.01, help="model timestep (s)")
    ap.add_argument("--src-dt", type=float, default=0.915, help="imaging timestep (s)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"=== calcium dataset: sequence={args.sequence} -> {args.out} ===")

    theta_hr, frames, theta_frame = load_sequence_heading(
        args.sequence, args.connectome, args.fishfuncem, args.dt, args.src_dt)
    ca = load_calcium_traces(args.connectome)
    block = build_full_block(theta_hr, frames, ca["traces"], args.dt, args.src_dt)

    train_starts, test_starts, n_tile = sample_starts(
        block["calcium"].shape[0], args.n_steps,
        args.n_trials_train, args.n_trials_test, args.seed)

    n_tr, ca_tr = write_split(os.path.join(args.out, "train"), train_starts,
                              block, args.n_steps, args.dt)
    n_te, ca_te = write_split(os.path.join(args.out, "test"), test_starts,
                              block, args.n_steps, args.dt)
    write_calcium_meta(args.out, ca, block, args.sequence, args.dt, args.src_dt,
                       n_block_tiles=n_tile, n_steps=args.n_steps)
    write_mapping(args.out, ca, args.circuit, args.connectome)

    plot_block_kinograph(os.path.join(args.out, f"calcium_block_{args.sequence}.png"),
                         block, args.dt, args.sequence)
    plot_example_trials(os.path.join(args.out, "calcium_traces_train.png"),
                        ca_tr, os.path.join(args.out, "train"), args.dt)

    print(f"=== done: train={n_tr} (first {n_tile} = consecutive block tiles) "
          f"test={n_te} trials, {ca['traces'].shape[1]} observed neurons, "
          f"n_steps={args.n_steps} ===")


if __name__ == "__main__":
    main()
