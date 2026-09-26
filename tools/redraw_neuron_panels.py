#!/usr/bin/env python
"""Redraw neuron_panels for a finished-or-running run from its latest checkpoint."""
import glob
import os
import re
import sys

import torch

sys.path.insert(0, "src")
from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.registry import create_model
from connectome_gnn.models.training_utils import init_training_data
from connectome_gnn.neuron_panels import analyse_neurons
from connectome_gnn.utils import migrate_state_dict, set_data_root

set_data_root(os.environ["GNN_OUTPUT_ROOT"])
LOG = f"{os.environ['GNN_OUTPUT_ROOT']}/log/fly"
dev = "cuda" if torch.cuda.is_available() else "cpu"
for run in sys.argv[1:]:
    cfg = NeuralGraphConfig.from_yaml(f"config/fly/{run}.yaml")
    cks = sorted(glob.glob(f"{LOG}/{run}/models/*graphs_0_*.pt"),
                 key=lambda f: int(re.findall(r"_(\d+)\.pt$", f)[0]))
    if not cks:
        print(f"{run}: no checkpoint"); continue
    it = int(re.findall(r"_(\d+)\.pt$", cks[-1])[0])
    sd = torch.load(cks[-1], map_location=dev, weights_only=False); migrate_state_dict(sd)
    cfg.simulation.n_edges = sd["model_state_dict"]["W"].shape[0]
    cfg.simulation.n_extra_null_edges = 0
    model = create_model(cfg.graph_model.signal_model_name,
                         aggr_type=cfg.graph_model.aggr_type, config=cfg, device=dev)
    model.load_state_dict(sd["model_state_dict"], strict=False); model.eval()
    cfg.dataset = "fly/" + cfg.dataset
    log_dir = os.path.join(LOG, run)
    import logging as _lg
    data = init_training_data(cfg, dev, log_dir, _lg.getLogger("repanel"))
    model.edges = data.edges
    out = os.path.join(log_dir, "tmp_training", "neuron_panels")
    analyse_neurons(cfg, model, data, log_dir, device=dev, out_dir=out,
                    tag=f"{it:08d}", sr_enabled=False, use_rollout=False, quiet=True)
    print(f"{run}: redrew neuron2895 at iter {it} -> {out}")
