#!/usr/bin/env python
"""Wait for the two joint-misspecification datasets, then submit + collect.

Runs unattended: polls the local generation until both datasets are complete,
submits the 4 cluster jobs (2 GNN + 2 Known-ODE) and waits for them, then runs
collect_metrics.py so results_table.csv picks up the `joint` rows.

  GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData PYTHONPATH=src \
    python neurips_review/finish_joint.py
"""
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/groups/saalfeld/home/allierc/GraphData"
PY = "/workspace/.conda_envs/neural-graph-linux/bin/python"
MAX_GEN_HOURS = 8.0


def ready(ds):
    d = f"{ROOT}/graphs_data/fly/{ds}"
    return (os.path.exists(f"{d}/ode_params.pt")
            and os.path.exists(f"{d}/x_list_train/voltage.zarr/.zarray"))


def main():
    man = json.load(open(os.path.join(HERE, "manifest_joint.json")))
    datasets = man["datasets"]
    t0 = time.time()
    while not all(ready(d) for d in datasets):
        if time.time() - t0 > MAX_GEN_HOURS * 3600:
            print("TIMEOUT waiting for generation:",
                  {d: ready(d) for d in datasets}, flush=True)
            return 1
        print(f"[wait] {int((time.time()-t0)/60)} min: "
              + ", ".join(f"{d}={'ok' if ready(d) else '...'}" for d in datasets), flush=True)
        time.sleep(300)

    print("both datasets ready; submitting cluster jobs", flush=True)
    return subprocess.run(
        [PY, os.path.join(HERE, "launch_cluster.py"),
         "--manifest", "manifest_joint.json", "--max-hours", "14"],
        cwd=HERE).returncode


if __name__ == "__main__":
    raise SystemExit(main())
