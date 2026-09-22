#!/usr/bin/env python
"""Pre-training data-integrity checks: confirm each generated dataset carries the
intended misspecification (before we trust ~8h of cluster training)."""
import numpy as np
import tensorstore as ts

ROOT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/fly"
DT = 0.02


def z(ds, sub):
    p = f"{ROOT}/{ds}/{sub}"
    return np.asarray(ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": p}}).result().read().result())


def volt(ds):
    return z(ds, "x_list_train/voltage.zarr")


def yy(ds):
    return z(ds, "y_list_train.zarr").reshape(-1, 13741)


base = "flyvis_noise_005_blank50_cv00"   # existing analytic-target reference
T = 6000
vb = volt(base)[:T]
yb = yy(base)[:T]

print("== Test 1 (Δt mismatch): finite-diff target stored, curvature grows with M ==")
for M in [1, 2, 5, 10]:
    ds = f"nr2_dt_M{M}_cv00"
    v = volt(ds)[:T]
    y = yy(ds)[:T]
    fd = (v[1:] - v[:-1]) / DT
    m = min(len(fd), len(y) - 1)
    err = np.max(np.abs(y[:m] - fd[:m]))                      # y == observed finite diff?
    # curvature bias proxy: slope of finite-diff target vs the M=1 (=analytic) target,
    # matched on the SAME base analytic derivative via per-neuron regression of y on yb.
    slope = np.polyfit(yb[:m].ravel(), y[:m].ravel(), 1)[0]
    dvb = np.max(np.abs(v[:T] - vb[:T]))                       # trajectory divergence from base
    print(f"  M={M:2d}: y==finiteDiff (max err {err:.2e}) | slope(y vs base-analytic)={slope:.3f} "
          f"| max|v−base|={dvb:.2e}")

print("\n== Test 3 (unobserved adaptation): ga=0 reproduces base, ga>0 diverges (grows in time) ==")
v00 = volt("nr2_adapt_ga00_cv00")[:T]
y00 = yy("nr2_adapt_ga00_cv00")[:T]
print(f"  ga=0.0: max|v−base|={np.max(np.abs(v00 - vb)):.2e} (reproduce base)  "
      f"max|y−base_analytic|={np.max(np.abs(y00[:len(yb)] - yb[:len(y00)])):.2e} (analytic, NOT finite-diff)")
for tag, ga in [("01", 0.1), ("03", 0.3)]:
    v = volt(f"nr2_adapt_ga{tag}_cv00")[:T]
    d = np.abs(v - v00)
    early = np.mean(d[:200]); late = np.mean(d[-200:])
    print(f"  ga={ga}: mean|v−ga0| early={early:.3e} late={late:.3e} "
          f"({'grows' if late > early * 1.3 else 'flat'}: slow current accumulates)")
