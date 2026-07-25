#!/usr/bin/env python
"""Generate the 7 misspecification datasets LOCALLY on the 2 RTX A6000 GPUs.

Only the `_gen_` configs carry the misspecification knobs; they run through the
(locally-modified) generator. Output goes to the shared GraphData filesystem so
the cluster can train on it afterwards. Test 2 needs no generation (reuses the
existing flyvis_noise_005_blank50_cv00 data).

  GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData PYTHONPATH=src \
    /workspace/.conda_envs/neural-graph-linux/bin/python neurips_review/generate_local.py
"""
import json
import os
import subprocess
import time

REPO = "/workspace/connectome-gnn-cx"
ROOT = "/groups/saalfeld/home/allierc/GraphData"
CONFIG_DIR = os.path.join(ROOT, "config", "fly")
PY = "/workspace/.conda_envs/neural-graph-linux/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
N_GPUS = 2


def load_manifest():
    with open(os.path.join(HERE, "manifest.json")) as f:
        return json.load(f)


def dataset_ready(dataset):
    d = f"{ROOT}/graphs_data/fly/{dataset}"
    # ode_params.pt is written last; require it so a dataset isn't declared ready mid-write
    return (os.path.exists(f"{d}/ode_params.pt")
            and os.path.exists(f"{d}/x_list_train/voltage.zarr/.zarray"))


def launch(gen_config, gpu):
    env = dict(os.environ, GNN_OUTPUT_ROOT=ROOT, PYTHONPATH="src", CUDA_VISIBLE_DEVICES=str(gpu))
    cfg_abs = os.path.join(CONFIG_DIR, gen_config + ".yaml")
    log = os.path.join(HERE, f"_gen_{gen_config}.log")
    lf = open(log, "w")
    p = subprocess.Popen([PY, "GNN_Main.py", "-o", "generate", cfg_abs, "--force"],
                         cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return {"proc": p, "log": log, "lf": lf, "gpu": gpu, "config": gen_config}


def main():
    man = load_manifest()
    jobs = list(man["gen"])  # [{config, dataset}, ...]
    # longest first so the 2 GPUs stay balanced (M10 dominates)
    order = {"M10": 0, "M5": 1, "M2": 2, "M1": 3}
    jobs.sort(key=lambda j: order.get(next((k for k in order if k in j["config"]), ""), 9))

    pending = list(jobs)
    running = {}  # gpu -> job
    results = {}
    print(f"generating {len(pending)} datasets on {N_GPUS} GPUs")
    while pending or running:
        # fill free GPUs
        for gpu in range(N_GPUS):
            if gpu not in running and pending:
                j = pending.pop(0)
                if dataset_ready(j["dataset"]):
                    print(f"[skip] {j['dataset']} already present")
                    results[j["config"]] = 0
                    continue
                running[gpu] = launch(j["config"], gpu)
                print(f"[gpu{gpu}] START {j['config']} -> {j['dataset']}")
        # poll
        for gpu, job in list(running.items()):
            rc = job["proc"].poll()
            if rc is not None:
                job["lf"].close()
                ds = next(d["dataset"] for d in jobs if d["config"] == job["config"])
                ok = rc == 0 and dataset_ready(ds)
                results[job["config"]] = 0 if ok else (rc or 1)
                print(f"[gpu{gpu}] DONE  {job['config']}: {'ok' if ok else 'FAIL'} (rc={rc})  log {job['log']}")
                if not ok:
                    subprocess.run(["tail", "-25", job["log"]])
                del running[gpu]
        time.sleep(10)

    n_fail = sum(1 for v in results.values() if v != 0)
    print(f"\nGENERATION COMPLETE: {len(results) - n_fail}/{len(results)} ok, {n_fail} failed")
    return n_fail


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
