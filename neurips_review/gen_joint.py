#!/usr/bin/env python
"""Joint-misspecification 2x2 (R1 W1: "their joint effect is never quantified").

Two axes, finite-difference target throughout so all four cells share one target
definition:

                     g_a = 0                 g_a = 0.3
  h = 20 ms   nr2_dt_M1_cv00   (have)   nr2_joint_ga03_M1_cv00   (new)
  h =  2 ms   nr2_dt_M10_cv00  (have)   nr2_joint_ga03_M10_cv00  (new)

The existing nr2_adapt_ga03 run is NOT a valid single-axis control here: it was
generated with the analytic target and no substeps, so the M1/g_a=0.3 cell has to
be regenerated under the finite-difference target.

Emits 2 gen configs + 4 train configs and appends them to manifest_joint.json.
Same conventions as gen_configs.py: misspec knobs live only in `_gen_` configs.
"""
import copy
import json
import os

import yaml

CONFIG_DIR = "/groups/saalfeld/home/allierc/GraphData/config/fly"
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifest_joint.json")

UNIFIED_TMPL = os.path.join(CONFIG_DIR, "flyvis_noise_005_blank50_unified_cv00.yaml")
KNOWNODE_TMPL = os.path.join(CONFIG_DIR, "flyvis_noise_005_blank50_known_ode_cv00.yaml")

MISSPEC = ("n_generation_substeps", "finite_difference_target", "adapt_g", "adapt_tau_ms")
CELLS = [("M1", 1), ("M10", 10)]
GA = 0.3
TAU_A = 200.0


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _dump(cfg, name):
    path = os.path.join(CONFIG_DIR, name + ".yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return path


def _strip(sim):
    for k in MISSPEC:
        sim.pop(k, None)


def main():
    unified = _load(UNIFIED_TMPL)
    knownode = _load(KNOWNODE_TMPL)
    man = {"gen": [], "train": [], "datasets": [], "config_dir": CONFIG_DIR}

    for tag, M in CELLS:
        ds = f"nr2_joint_ga{int(GA*10):02d}_{tag}_cv00"
        descr = (f"Joint misspecification: adaptation g_a={GA} (tau_a={TAU_A:.0f}ms, latent c_i) "
                 f"AND generator step h=dt/{M}, observed+trained at dt=20ms, finite-diff target")

        gen = copy.deepcopy(unified)
        gen["description"] = descr + " | LOCAL generation (single-seed cv00)"
        gen["dataset"] = ds
        _strip(gen["simulation"])
        gen["simulation"].update({
            "n_generation_substeps": M,
            "finite_difference_target": True,
            "adapt_g": GA,
            "adapt_tau_ms": TAU_A,
        })
        gen["config_file"] = f"fly/{ds}_gen"
        _dump(gen, f"{ds}_gen")

        for tmpl, suffix, model in ((unified, "unified", "gnn"), (knownode, "known_ode", "known_ode")):
            cfg = copy.deepcopy(tmpl)
            cfg["description"] = descr + f" | {model} (single-seed cv00)"
            cfg["dataset"] = ds
            _strip(cfg["simulation"])  # cluster parser never sees the new keys
            cfg["config_file"] = f"fly/{ds}_{suffix}"
            _dump(cfg, f"{ds}_{suffix}")
            man["train"].append({"config": f"{ds}_{suffix}", "dataset": ds,
                                 "model": model, "test": "joint"})

        man["gen"].append({"config": f"{ds}_gen", "dataset": ds})
        man["datasets"].append(ds)

    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=2)
    print(f"emitted {len(man['gen'])} gen + {len(man['train'])} train configs")
    print("datasets:", man["datasets"])
    print("manifest ->", MANIFEST)


if __name__ == "__main__":
    main()
