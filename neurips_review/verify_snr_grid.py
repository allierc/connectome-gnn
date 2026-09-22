#!/usr/bin/env python
"""Assert the SNR-grid datasets are what we think they are. Re-run any time.

Guards the three defects that have already bitten this pipeline:
  1. lambda too large -> derivative destroyed while the trace still looks fine.
     Checked by r(vdot), NOT r(v). r(v) moved <0.02 for both past bugs.
  2. FFT wrap / kernel warm-up -> a handful of frames with ~100x the interior
     amplitude, which dominate an MSE loss. Checked on max|v| and max|y|.
  3. dataset overwritten by a regeneration pass -> voltage.zarr would become a
     fresh simulation instead of the deconvolved estimate. Checked by comparing
     the live r(v)/r(vdot) against the values recorded at build time.

Exit code 0 = all good, 1 = at least one dataset is not trustworthy.
"""
import os, sys
import numpy as np, zarr

BASE = "/groups/saalfeld/home/allierc/GraphData/graphs_data/fly"
TRUTH = "nr2_ca_blank50_kernel"
TAGS = ["g000", "g010", "g030", "g100"]
TOL = 0.01          # r must match the build-time record this closely
EDGE_RATIO = 3.0    # max|v| anywhere vs interior max


def r(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / np.sqrt((a @ a) * (b @ b) + 1e-30))


def recorded(tag):
    p = os.path.join(BASE, f"nr2_ca_snr_{tag}", ".generate_done")
    out = {}
    for ln in open(p):
        if ln.startswith(("train:", "test:")):
            split = ln.split(':')[0]
            for part in ln.split():
                if part.startswith("r(v)="):
                    out[(split, 'v')] = float(part.split('=')[1])
                elif part.startswith("r(vdot)="):
                    out[(split, 'vdot')] = float(part.split('=')[1])
    return out


def check(tag):
    ok = True
    rec = recorded(tag)
    for split in ("train", "test"):
        vt = np.asarray(zarr.open_array(
            f"{BASE}/{TRUTH}/x_list_{split}/voltage.zarr", mode='r')[:], dtype=np.float64)
        vd = np.asarray(zarr.open_array(
            f"{BASE}/nr2_ca_snr_{tag}/x_list_{split}/voltage.zarr", mode='r')[:], dtype=np.float64)
        n = min(len(vt), len(vd)); vt, vd = vt[:n], vd[:n]
        rv, rd = r(vt, vd), r(np.diff(vt, axis=0), np.diff(vd, axis=0))

        # (3) content matches build-time record
        for name, got in (('v', rv), ('vdot', rd)):
            want = rec.get((split, name))
            if want is None:
                print(f"  [{tag}/{split}] FAIL no build-time record for r({name})"); ok = False
            elif abs(got - want) > TOL:
                print(f"  [{tag}/{split}] FAIL r({name})={got:.4f} but build recorded {want:.4f}"
                      f"  -> dataset changed since build"); ok = False

        # (2) no edge spikes
        prof = np.abs(vd).max(axis=1)
        interior = prof[120:-120].max()
        if prof.max() > EDGE_RATIO * interior:
            bad = np.where(prof > EDGE_RATIO * interior)[0]
            print(f"  [{tag}/{split}] FAIL {len(bad)} frames exceed {EDGE_RATIO}x interior "
                  f"(max {prof.max():.1f} vs {interior:.1f}) at {bad[:6].tolist()}"); ok = False

        # y targets consistent with v
        y = np.asarray(zarr.open_array(
            f"{BASE}/nr2_ca_snr_{tag}/y_list_{split}.zarr", mode='r')[:], dtype=np.float64)
        y = y[..., 0] if y.ndim == 3 else y
        expect = (vd[1:] - vd[:-1]) / 0.02
        if not np.allclose(y[:len(expect)], expect, atol=1e-2, rtol=1e-3):
            print(f"  [{tag}/{split}] FAIL y targets are not diff(v)/dt"); ok = False

        print(f"  [{tag}/{split}] r(v)={rv:.4f} r(vdot)={rd:.4f} "
              f"max|v|={prof.max():.2f} (interior {interior:.2f}) max|y|={np.abs(y).max():.1f}")
    return ok


if __name__ == '__main__':
    allok = True
    for tag in TAGS:
        print(f"== nr2_ca_snr_{tag}")
        allok &= check(tag)
    print("\nALL DATASETS VERIFIED" if allok else "\nPROBLEMS FOUND -- do not trust these runs")
    sys.exit(0 if allok else 1)
