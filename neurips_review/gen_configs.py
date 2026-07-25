#!/usr/bin/env python
"""NeurIPS-2026 rebuttal (concern #2: "strongly model-matched") — config emitter.

Emits every YAML for the three model-misspecification experiments into the
shared, cluster-visible config dir, all templated on the Flyvis-217 blank50
consensus triplet (the paper's sigma=0.05 / 50%-null-stimulus reference):

  base gen   : flyvis_noise_005_blank50_gen_cv00.yaml     (sim block, blank50)
  base GNN   : flyvis_noise_005_blank50_unified_cv00.yaml (Supp. Tab. 6 consensus HPs)
  base oracle: flyvis_noise_005_blank50_known_ode_cv00.yaml (Known-ODE, flyvis_known_ode)

Design (see neurips_review/README.md):
  * The 4 misspecification knobs (n_generation_substeps, finite_difference_target,
    adapt_g, adapt_tau_ms) live ONLY in the local `_gen_` configs used for local
    data generation. The cluster-side `_unified_`/`_known_ode_` training configs
    are byte-clean of new keys, so the (possibly older) cluster checkout never
    parses them. Training just reads the pre-generated data.
  * Single seed throughout (cv00). Everything is labelled single-seed.

Experiments:
  Test 1  Delta-t mismatch      M in {1,2,5,10}  -> substeps + finite-diff target
  Test 2  Monotonicity ablation {ctrl, mu0off, mu1off}  (reuses base blank50 data)
  Test 3  Unobserved adaptation g_a in {0.0,0.1,0.3}    -> hidden -g_a*c current

Writes a manifest neurips_review/manifest.json consumed by generate_local.py /
launch_cluster.py / collect_metrics.py.
"""
import copy
import json
import os

import yaml

CONFIG_DIR = "/groups/saalfeld/home/allierc/GraphData/config/fly"
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifest.json")

UNIFIED_TMPL = os.path.join(CONFIG_DIR, "flyvis_noise_005_blank50_unified_cv00.yaml")
KNOWNODE_TMPL = os.path.join(CONFIG_DIR, "flyvis_noise_005_blank50_known_ode_cv00.yaml")
BASE_DATASET = "flyvis_noise_005_blank50_cv00"  # existing; reused by Test 2

PREFIX = "nr2"  # neurips rebuttal, concern #2

TEST1_M = [1, 2, 5, 10]
TEST3_GA = [("00", 0.0), ("01", 0.1), ("03", 0.3)]
TEST2 = [
    ("ctrl", {}),                              # consensus as-is (mu0 already 0, mu1=750)
    ("mu0off", {"coeff_f_theta_msg_diff": 0}), # explicit mu0=0 (already the default -> == ctrl)
    ("mu1off", {"coeff_g_phi_diff": 0}),       # remove g_phi monotonicity prior
]


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _dump(cfg, name):
    path = os.path.join(CONFIG_DIR, name + ".yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return path


def _strip_misspec(sim):
    for k in ("n_generation_substeps", "finite_difference_target", "adapt_g", "adapt_tau_ms"):
        sim.pop(k, None)


def make_gen(base_unified, name, dataset, sim_updates, descr):
    """Local generation config: unified template + misspec knobs in the sim block."""
    cfg = copy.deepcopy(base_unified)
    cfg["description"] = descr
    cfg["dataset"] = dataset
    sim = cfg["simulation"]
    _strip_misspec(sim)
    sim.update(sim_updates)  # inject the misspec knobs
    cfg["config_file"] = f"fly/{name}"
    return _dump(cfg, name)


def make_train(base_tmpl, name, dataset, train_updates, descr):
    """Cluster training config: clean of any misspec keys."""
    cfg = copy.deepcopy(base_tmpl)
    cfg["description"] = descr
    cfg["dataset"] = dataset
    _strip_misspec(cfg["simulation"])  # ensure no new keys reach the cluster parser
    if train_updates:
        cfg["training"].update(train_updates)
    cfg["config_file"] = f"fly/{name}"
    return _dump(cfg, name)


def main():
    unified = _load(UNIFIED_TMPL)
    knownode = _load(KNOWNODE_TMPL)

    manifest = {"gen": [], "train": [], "datasets": [], "config_dir": CONFIG_DIR}

    def add_condition(dataset, sim_updates, tag_descr):
        """Emit gen + unified + known_ode for a single generated dataset."""
        gen_name = f"{dataset}_gen"
        uni_name = f"{dataset}_unified"
        ko_name = f"{dataset}_known_ode"
        make_gen(unified, gen_name, dataset, sim_updates, f"{tag_descr} | LOCAL generation (single-seed cv00)")
        make_train(unified, uni_name, dataset, None, f"{tag_descr} | GNN consensus (single-seed cv00)")
        make_train(knownode, ko_name, dataset, None, f"{tag_descr} | Known-ODE oracle (single-seed cv00)")
        manifest["gen"].append({"config": gen_name, "dataset": dataset})
        manifest["datasets"].append(dataset)
        manifest["train"].append({"config": uni_name, "dataset": dataset, "model": "gnn", "test": "dt" if "dt_" in dataset else "adapt"})
        manifest["train"].append({"config": ko_name, "dataset": dataset, "model": "known_ode", "test": "dt" if "dt_" in dataset else "adapt"})

    # ---- Test 1: Delta-t mismatch ----
    for M in TEST1_M:
        ds = f"{PREFIX}_dt_M{M}_cv00"
        add_condition(
            ds,
            {"n_generation_substeps": M, "finite_difference_target": True},
            f"Test1 Delta-t mismatch M={M} (simulate at dt/{M}, observe+train at dt=20ms, finite-diff target)",
        )

    # ---- Test 3: Unobserved adaptation current ----
    for tag, ga in TEST3_GA:
        ds = f"{PREFIX}_adapt_ga{tag}_cv00"
        add_condition(
            ds,
            {"adapt_g": ga, "adapt_tau_ms": 200.0},
            f"Test3 unobserved adaptation g_a={ga}, tau_a=200ms (latent c_i outside the graph)",
        )

    # ---- Test 2: Monotonicity ablation (reuse base blank50 data; GNN only) ----
    for tag, upd in TEST2:
        name = f"{PREFIX}_mono_{tag}_cv00"
        make_train(
            unified, name, BASE_DATASET, upd,
            f"Test2 monotonicity ablation [{tag}] on {BASE_DATASET} (single-seed cv00)",
        )
        manifest["train"].append({"config": name, "dataset": BASE_DATASET, "model": "gnn", "test": "mono"})

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"emitted {len(manifest['gen'])} gen configs, {len(manifest['train'])} train configs")
    print(f"datasets to generate locally: {manifest['datasets']}")
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
