#!/usr/bin/env python
"""Build a deconvolved-calcium dataset: calcium (+ measurement noise) -> voltage.

The two-step calcium protocol (experiment 12): the source dataset carries the
simulated voltage and the calcium it produces through the GCaMP6f kernel
(`x_list_*/calcium.zarr`, C = K * v); this adds white measurement noise at a
given SNR, deconvolves it back to voltage (generators/calcium_deconvolution.py,
lambda chosen blind by default), and writes a sibling dataset in which
`voltage.zarr` IS THE ESTIMATE, so every trainer and tester reads it unchanged,
and `voltage_true.zarr` is the simulated voltage, which the tester scores the
rollout against. Everything else is symlinked from the source.

    python tools/build_calcium_dataset.py --snr inf 40 30 20 [--lam auto|<float>]

`--observe calcium` (experiment 13) skips the deconvolution: `voltage.zarr` is
the noisy calcium itself, which the latent-calcium trainer reads as its
recording (training.observable: calcium), in `<source>_cal_<snr>`. The noise is
drawn with the same seed and in the same order as the deconvolved dataset's,
so experiments 12 and 13 see the same measurements.

Measurement noise: one SD = frac * SD(C) over all neurons, frac = 10^(-snr/20)
(40 dB -> 0.01, 30 dB -> 0.032, 20 dB -> 0.1), the earlier builder's convention.
The deconvolution quality against the truth (r of v and of its derivative) is
measured on a subset and written to the dataset's `calcium_deconvolution.txt`
-- for the record only; nothing is chosen from it.
"""
import argparse
import os
import sys

import numpy as np
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from connectome_gnn.generators.calcium_deconvolution import (  # noqa: E402
    choose_lambda, wiener_deconvolve)
from connectome_gnn.models.gcamp import create_gcamp  # noqa: E402

ROOT = f"{os.environ['GNN_OUTPUT_ROOT']}/graphs_data/fly"
DT = 0.02
KERNEL = dict(tau_rise=0.075, tau_decay=0.4, length_s=2.4)   # the source's kernel


def _link(src, dst):
    if not os.path.lexists(dst):
        os.symlink(src, dst)


def _write_like(src_arr, path, data):
    z = zarr.open(path, mode="w", shape=data.shape, chunks=src_arr.chunks,
                  dtype="float32", compressor=src_arr.compressor)
    for s in range(0, data.shape[0], src_arr.chunks[0]):
        z[s:s + src_arr.chunks[0]] = data[s:s + src_arr.chunks[0]].astype(np.float32)


def _r_cols(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0) + 1e-30)


def build(source, snr, lam_mode, seed=0, n_sub=500, sub_frames=16000, observe="voltage"):
    tag = "inf" if snr == "inf" else f"{int(snr)}db"
    kind = "cal" if observe == "calcium" else "ca"
    name = f"{source.replace('_kernel', '')}_{kind}_{tag}"
    out = os.path.join(ROOT, name)
    os.makedirs(out, exist_ok=True)
    frac = 0.0 if snr == "inf" else 10.0 ** (-float(snr) / 20.0)
    k = create_gcamp("double_exp", **KERNEL).kernel(DT).numpy().astype(np.float64)
    src_dir = os.path.join(ROOT, source)
    for f in os.listdir(src_dir):
        if not f.startswith("x_list_"):
            _link(os.path.join(src_dir, f), os.path.join(out, f))
    rng = np.random.default_rng(seed)
    lam, log = None, [f"source {source}, kernel {KERNEL}, dt {DT}, snr {snr} dB (frac {frac:.4f})"]
    for split in ("x_list_train", "x_list_test"):
        s_dir, o_dir = os.path.join(src_dir, split), os.path.join(out, split)
        if not os.path.isdir(s_dir):
            continue
        os.makedirs(o_dir, exist_ok=True)
        for f in os.listdir(s_dir):
            if f not in ("voltage.zarr",):
                _link(os.path.join(s_dir, f), os.path.join(o_dir, f))
        _link(os.path.join(s_dir, "voltage.zarr"), os.path.join(o_dir, "voltage_true.zarr"))
        c_arr = zarr.open(os.path.join(s_dir, "calcium.zarr"), mode="r")
        v_arr = zarr.open(os.path.join(s_dir, "voltage.zarr"), mode="r")
        C = np.asarray(c_arr[:], dtype=np.float64)
        if frac > 0:                       # one SD for all neurons: frac x SD of all calcium
            C = C + rng.normal(size=C.shape) * (frac * C.std())
        T, N = C.shape
        sub = np.sort(rng.choice(N, min(n_sub, N), replace=False))
        if observe == "calcium":                         # the recording is the state
            _write_like(c_arr, os.path.join(o_dir, "voltage.zarr"), C)
            log.append(f"{split}: {T} frames x {N} neurons; voltage.zarr = calcium + noise SD "
                       f"{frac * np.asarray(c_arr[:]).std():.3g} (frac {frac:.4f} x SD of all calcium)")
            continue
        if lam is None:                                  # chosen on the train split, kept for test
            if lam_mode == "auto":
                lam, sd = choose_lambda(C[:sub_frames, sub], k)
                log.append(f"lambda blind (discrepancy, noise SD from the autocovariance, "
                           f"{len(sub)} traces x {sub_frames} frames): {lam:.3g}; noise SD "
                           f"estimated {sd:.3g}, added {frac * np.asarray(c_arr[:sub_frames]).std():.3g}")
            else:
                lam = float(lam_mode)
                log.append(f"lambda fixed = {lam:.3g}")
        est = np.empty((T, N), dtype=np.float32)
        for s in range(0, N, 1000):
            est[:, s:s + 1000] = wiener_deconvolve(C[:, s:s + 1000], k, lam)
        _write_like(c_arr, os.path.join(o_dir, "voltage.zarr"), est)
        vt = np.asarray(v_arr[:, sub], dtype=np.float64)
        e = est[:, sub].astype(np.float64)
        rv, rd = _r_cols(e, vt), _r_cols(np.diff(e, axis=0), np.diff(vt, axis=0))
        log.append(f"{split}: {T} frames x {N} neurons; against the truth on {len(sub)} neurons: "
                   f"r(v) median {np.median(rv):.4f}, r(vdot) median {np.median(rd):.4f}")
    open(os.path.join(out, "calcium_deconvolution.txt"), "w").write("\n".join(log) + "\n")
    print(f"{name}:\n  " + "\n  ".join(log))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="flyvis_unified_blank50_kernel")
    ap.add_argument("--snr", nargs="+", default=["inf", "40", "30", "20"])
    ap.add_argument("--lam", default="auto")
    ap.add_argument("--observe", choices=["voltage", "calcium"], default="voltage")
    a = ap.parse_args()
    for snr in a.snr:
        build(a.source, snr, a.lam, observe=a.observe)


if __name__ == "__main__":
    main()
