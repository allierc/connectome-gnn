"""Reproduce the ZAPBench panel-A whole-brain ΔF/F kinograph with our code,
and mark the rows (neurons) our HD/IPN12 circuit uses with red ticks on the
right.

Full df/F matrix (7879 time steps x 71721 neurons, raw ΔF/F 0..0.5, viridis),
9 task blocks labelled GAIN/DOTS/FLASH/TAXIS/TURNING/POSITION/OPEN LOOP/
ROTATION/DARK with white separators, a 5-min scale bar, and red ticks at the
zapbenchId rows of the 481 matched neurons.

Rows are in raw zapbenchId order (= the trace-zarr column order our pipeline
indexes), so a red tick at row z means "we pull neuron zapbenchId=z". The
paper sorts rows by rastermap (cosmetic); we keep raw order so the ticks land
at the actual indices we use.

  /workspace/.conda_envs/neural-graph-linux/bin/python \
      figures/zebrafish/zapbench_full_kinograph.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(_REPO, "papers", "fishFuncEM"))

_GD = "/groups/saalfeld/home/allierc/GraphData/graphs_data/zebrafish"
DFF_NPY = os.path.join(_GD, "zapbench_dff_full.npy")
SORT_NPY = os.path.join(_GD, "zapbench_rastermap_sorting.npy")
SORT_URL = ("https://storage.googleapis.com/zapbench-release/volumes/20240930/"
            "traces_rastermap_sorted/sorting.json")
ZARR_URL = ("gs://ahrens-connectomics-eda3e36871e30f4c0b73531c3a0f6a90/"
            "alex_data/traces/20220404_emf3/xm-97797086.ckpt-332.240317/"
            "segmentation_filtered/df_over_f_clipped/")
TASKS = ["GAIN", "DOTS", "FLASH", "TAXIS", "TURNING",
         "POSITION", "OPEN LOOP", "ROTATION", "DARK"]
FRAME_SEC = 0.915


def _load_sorting():
    """Official rastermap permutation (cached). perm[i] = original neuron at
    sorted row i."""
    if os.path.isfile(SORT_NPY):
        return np.load(SORT_NPY)
    import json
    import urllib.request
    perm = np.asarray(json.loads(urllib.request.urlopen(SORT_URL).read()),
                      dtype=np.int64)
    np.save(SORT_NPY, perm)
    return perm


def _load_dff(dff_path, perm=None, disp_rows=6000):
    """Return (img (n_disp, T), N_full). Loads the cached memmap and block-
    means neurons down to ~disp_rows for a low-memory display. If `perm` is
    given (rastermap order), neurons are gathered in that order before the
    block-mean (each block read is a few columns, so memory stays small)."""
    mm = np.load(dff_path, mmap_mode="r")            # (T, N)
    T, N = mm.shape
    order = perm if perm is not None else np.arange(N)
    fac = max(1, N // disp_rows)
    n_disp = N // fac
    img = np.empty((n_disp, T), dtype=np.float32)
    for r in range(n_disp):
        cols = order[r * fac:(r + 1) * fac]
        img[r] = np.asarray(mm[:, cols]).mean(1)
    return img, N


def _load_dff_cols(dff_path, cols):
    """Return (len(cols), T): one full-resolution row per requested neuron
    column (zapbenchId), in the given order. Used for the bump-only kinograph
    where there are few enough rows (≈300) to skip the block-mean."""
    mm = np.load(dff_path, mmap_mode="r")            # (T, N)
    img = np.asarray(mm[:, np.asarray(cols, dtype=np.int64)]).T.astype(np.float32)
    return img, mm.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dff", default=DFF_NPY)
    ap.add_argument("--raw-order", action="store_true",
                    help="keep raw zapbenchId order (default: rastermap sorted)")
    ap.add_argument("--bump-only", action="store_true",
                    help="second kinograph: show ONLY the 300 mapped bump "
                         "neurons at full resolution (no whole-brain rows, no "
                         "red ticks), rastermap-ordered")
    ap.add_argument("--zscore", action="store_true",
                    help="per-neuron z-score the rows instead of raw ΔF/F "
                         "(0..0.5); recommended with --bump-only")
    args = ap.parse_args()
    out = os.path.join(_REPO, "figures", "zebrafish",
                       "zapbench_bump_kinograph.png" if args.bump_only
                       else "zapbench_full_kinograph.png")
    # matched rows — restrict to the bump pool (dIPN ring + IPN12), the same
    # neuron set the functional panel plots, so the red ticks mark exactly the
    # bump neurons our HD circuit uses (not the afferents).
    m = pd.read_csv(os.path.join(
        _REPO, "figures", "zebrafish", "zebrafish_connectome_HD_IPN12",
        "functional", "bodyid_zapbench_map.csv"))

    def _is_bump(t):
        t = str(t)
        return (t.startswith("IPN12") or t.startswith("IPNd")
                or t.startswith("IPNds"))

    bump = m[m["matched"] & m["type"].map(_is_bump)]
    zb = bump["zapbenchId"].astype(int).to_numpy()

    # task-block boundaries (imaging frames)
    from fishfuncem import FishFunctional
    ff = FishFunctional.from_data_dir(os.path.join(_REPO, "papers",
                                                   "fishFuncEM", "data"))
    onsets = np.asarray(ff.onsets_frames).astype(int)

    perm = None if args.raw_order else _load_sorting()
    if args.bump_only:
        # second kinograph: only the 300 mapped bump neurons, ordered by the
        # same rastermap permutation, loaded at full resolution (one row each).
        if perm is not None:
            inv = np.empty(perm.size, dtype=np.int64); inv[perm] = np.arange(perm.size)
            zb = zb[np.argsort(inv[zb])]
        else:
            zb = np.sort(zb)
        print(f"[load] loading ΔF/F for {len(zb)} bump neurons "
              f"({'raw' if perm is None else 'rastermap'} order) ...", flush=True)
        img, N = _load_dff_cols(args.dff, zb)           # (n_bump, T)
    else:
        print(f"[load] loading ΔF/F matrix (cached memmap, "
              f"{'raw' if perm is None else 'rastermap-sorted'} order) ...", flush=True)
        img, N = _load_dff(args.dff, perm=perm)         # (n_disp, T), N full
        if perm is not None:                            # zapbenchId -> sorted row
            inv = np.empty(N, dtype=np.int64); inv[perm] = np.arange(N)
            zb = inv[zb]
    T = img.shape[1]
    n_rows = img.shape[0]
    if args.zscore:
        mu = img.mean(1, keepdims=True); sd = img.std(1, keepdims=True)
        img = (img - mu) / np.where(sd > 1e-6, sd, 1.0)
        vmin = float(np.percentile(img, 2)); vmax = float(np.percentile(img, 99.5))
    else:
        vmin, vmax = 0.0, 0.5
    print(f"[load] display {img.shape}; N_full={N}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(17, 9))
    ax.imshow(img, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
              extent=[0, T, n_rows, 0], interpolation="nearest")

    # task separators + labels
    bounds = list(onsets) + [T]
    for b in bounds[1:-1]:
        ax.axvline(b, color="white", lw=1.2)
    for i, name in enumerate(TASKS):
        c = 0.5 * (bounds[i] + bounds[i + 1])
        ax.text(c, -n_rows * 0.012, name, color="black", fontsize=11,
                ha="center", va="bottom", fontweight="bold")

    # red ticks on the right for the rows we use — whole-brain view only; the
    # bump-only kinograph IS those neurons, so no ticks are needed.
    if not args.bump_only:
        x0, x1 = T * 1.002, T * 1.02
        for z in zb:
            ax.plot([x0, x1], [z, z], color="red", lw=0.4, solid_capstyle="butt",
                    clip_on=False)
        ax.text(T * 1.025, n_rows * 0.5, f"{len(zb)} HD/IPN12 neurons",
                color="red", fontsize=11, rotation=90, va="center", ha="left")

    # 5-min scale bar (bottom-left)
    bar = 5 * 60 / FRAME_SEC
    ax.plot([0.01 * T, 0.01 * T + bar], [n_rows * 1.02, n_rows * 1.02],
            color="black", lw=3, clip_on=False)
    ax.text(0.01 * T, n_rows * 1.04, "5 min", fontsize=10, va="top")

    ax.set_xlim(0, T * 1.03)
    lut = "z-score" if args.zscore else "ΔF/F"
    ax.set_ylabel(f"{n_rows:,} HD/IPN12 bump neurons" if args.bump_only
                  else f"{n_rows:,} neurons", fontsize=13)
    ax.set_xlabel(f"{T:,} time steps", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_visible(False)            # drop the black bounding box
    fig.suptitle(
        f"ZAPBench {lut} — {n_rows} mapped HD/IPN12 bump neurons "
        f"(rastermap order)" if args.bump_only else
        "ZAPBench whole-brain ΔF/F (our pipeline) — "
        "red ticks = rows used by our HD/IPN12 circuit",
        fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.05, right=0.93)
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
