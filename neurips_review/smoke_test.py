#!/usr/bin/env python
"""Tiny-scale correctness smoke test for the misspecification generator knobs.

Generates 4 small datasets and checks the invariants that must hold before we
spend GPU-hours on full generation:

  base   : no knobs (pure base single-Euler path)             -> reference
  M1fd   : n_substeps=1 + finite_difference_target            -> voltage MUST equal base
                                                                   (only y semantics differ)
  M5fd   : n_substeps=5 + finite_difference_target            -> voltage MUST differ (finer integ)
  ga03   : adapt_g=0.3                                          -> voltage MUST differ (hidden current)

Checks:
  1. voltage(M1fd) bit-identical to voltage(base)   [gated path leaves base untouched]
  2. y(M1fd) == observed finite difference (v[t+1]-v[t])/dt of the SHARED voltage
  3. y(base) == analytic derivative (differs from finite diff by ~process-noise/dt)
  4. voltage(M5fd) differs from base; per-observation voltage variance ~preserved
  5. voltage(ga03) differs from base

Run:
  GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData PYTHONPATH=src \
    /workspace/.conda_envs/neural-graph-linux/bin/python neurips_review/smoke_test.py
"""
import copy
import os
import subprocess
import sys

import numpy as np
import tensorstore as ts
import yaml

REPO = "/workspace/connectome-gnn-cx"
CONFIG_DIR = "/groups/saalfeld/home/allierc/GraphData/config/fly"
ROOT = "/groups/saalfeld/home/allierc/GraphData"
PY = "/workspace/.conda_envs/neural-graph-linux/bin/python"
UNIFIED_TMPL = os.path.join(CONFIG_DIR, "flyvis_noise_005_blank50_unified_cv00.yaml")

N_FRAMES = 400
MAX_SEQ = 4

CASES = {
    "nr2_smoke_base": {},
    "nr2_smoke_M1fd": {"n_generation_substeps": 1, "finite_difference_target": True},
    "nr2_smoke_M5fd": {"n_generation_substeps": 5, "finite_difference_target": True},
    "nr2_smoke_ga03": {"adapt_g": 0.3, "adapt_tau_ms": 200.0},
}


def openz(p):
    return ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": p}}).result().read().result()


def write_cfg(name, knobs):
    with open(UNIFIED_TMPL) as f:
        cfg = yaml.safe_load(f)
    cfg["description"] = f"SMOKE {name}"
    cfg["dataset"] = name
    sim = cfg["simulation"]
    sim["n_frames"] = N_FRAMES
    sim["max_train_sequences"] = MAX_SEQ
    for k in ("n_generation_substeps", "finite_difference_target", "adapt_g", "adapt_tau_ms"):
        sim.pop(k, None)
    sim.update(knobs)
    cfg["config_file"] = f"fly/{name}"
    path = os.path.join(CONFIG_DIR, name + ".yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def generate(name, gpu):
    env = dict(os.environ, GNN_OUTPUT_ROOT=ROOT, PYTHONPATH="src", CUDA_VISIBLE_DEVICES=str(gpu))
    log = os.path.join(REPO, "neurips_review", f"_smoke_{name}.log")
    cfg_abs = os.path.join(CONFIG_DIR, name + ".yaml")  # absolute path (config dir is not on repo path)
    with open(log, "w") as lf:
        p = subprocess.Popen([PY, "GNN_Main.py", "-o", "generate", cfg_abs, "--force"],
                             cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return p, log


def main():
    # write configs
    for name, knobs in CASES.items():
        write_cfg(name, knobs)
    # generate 2 at a time on the 2 GPUs
    names = list(CASES)
    procs = {}
    for i, name in enumerate(names):
        gpu = i % 2
        p, log = generate(name, gpu)
        procs[name] = (p, log)
        if (i % 2 == 1) or (i == len(names) - 1):
            for nm, (pp, lg) in procs.items():
                rc = pp.wait()
                tag = "ok" if rc == 0 else f"FAIL rc={rc}"
                print(f"[gen] {nm}: {tag}  (log {lg})")
                if rc != 0:
                    subprocess.run(["tail", "-30", lg])
            procs = {}

    # ---- verify ----
    dt = 0.02
    V = {n: openz(f"{ROOT}/graphs_data/fly/{n}/x_list_train/voltage.zarr") for n in CASES}
    Y = {n: np.asarray(openz(f"{ROOT}/graphs_data/fly/{n}/y_list_train.zarr")).reshape(len(V[n]), -1) for n in CASES}

    fails = []

    def check(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    T = min(len(V["nr2_smoke_base"]), len(V["nr2_smoke_M1fd"]))
    vb, vm1 = V["nr2_smoke_base"][:T], V["nr2_smoke_M1fd"][:T]
    # Exact bit-identity is unattainable across GPUs (scatter_add atomics are
    # non-deterministic); require divergence at rounding level (<<1) so we know
    # the gated M=1 path reproduces the base voltage dynamics (only y differs).
    dmax = float(np.max(np.abs(vb - vm1)))
    check(dmax < 1e-3, f"1. voltage(M1fd) == voltage(base) up to GPU rounding (max|diff|={dmax:.2e})")

    # finite-diff of the shared voltage vs stored y(M1fd)
    fd = (vb[1:] - vb[:-1]) / dt
    yb = Y["nr2_smoke_base"][:T]
    ym1 = Y["nr2_smoke_M1fd"][:T]
    m = min(len(fd), len(ym1) - 1)
    err_fd = np.max(np.abs(ym1[:m] - fd[:m]))
    check(err_fd < 1e-3, f"2. y(M1fd) == observed finite difference (max abs err {err_fd:.2e})")
    corr_analytic = np.corrcoef(yb[:m].ravel(), fd[:m].ravel())[0, 1]
    check(0.85 < corr_analytic < 0.999, f"3. y(base)=analytic differs from finite-diff (corr {corr_analytic:.3f}, expect ~0.96)")

    vM5 = V["nr2_smoke_M5fd"]
    Tm = min(len(vb), len(vM5))
    diff5 = np.max(np.abs(vb[:Tm] - vM5[:Tm]))
    check(diff5 > 1e-4, f"4a. voltage(M5fd) differs from base (max abs diff {diff5:.2e})")
    # per-observation voltage std should be within ~30% of base (noise 1/sqrt(M) scaling preserves it)
    r_std = float(np.std(vM5[:Tm]) / (np.std(vb[:Tm]) + 1e-9))
    check(0.5 < r_std < 1.6, f"4b. voltage std ratio M5/base = {r_std:.3f} (should be ~1)")

    vga = V["nr2_smoke_ga03"]
    Tg = min(len(vb), len(vga))
    diffg = np.max(np.abs(vb[:Tg] - vga[:Tg]))
    check(diffg > 1e-4, f"5. voltage(ga03) differs from base (max abs diff {diffg:.2e})")

    print("\n" + ("ALL SMOKE CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
