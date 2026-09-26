#!/usr/bin/env python
"""Redraw neuron_panels from a PRESERVED run, into a directory of your choosing.

tools/redraw_neuron_panels.py writes back into the run it read, which is exactly
what must not happen to a preserved checkpoint: the models under
log/fly/_preserved/ are the ones the figures in the presentation and the tables
in docs/ were measured from. This one takes the log directory, the checkpoint and
the neuron ids explicitly and never writes inside the run.

    python tools/redraw_panels_preserved.py \
        --log-dir <preserved run> --out /tmp/panels --neurons 2895 1234 ...

With no --neurons it picks one neuron per distinct cell type among those whose
in-degree lies in [--min-deg, --max-deg], which keeps every figure roughly the
same height and stops a hub with 200 synapses from producing a 300-inch page.
"""
import argparse
import glob
import logging
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
from connectome_gnn.models.registry import create_model  # noqa: E402
from connectome_gnn.models.training_utils import init_training_data  # noqa: E402
from connectome_gnn.neuron_panels import analyse_neurons  # noqa: E402
from connectome_gnn.utils import migrate_state_dict, set_data_root, to_numpy  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--log-dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--neurons", type=int, nargs="*", default=None)
p.add_argument("--n-types", type=int, default=4, help="neurons to pick when none given")
p.add_argument("--min-deg", type=int, default=8)
p.add_argument("--max-deg", type=int, default=14)
p.add_argument("--checkpoint", default=None, help="default: the highest iteration")
p.add_argument("--config", default=None, help="default: <log-dir>/config.yaml")
p.add_argument("--data-root", default=os.environ["GNN_OUTPUT_ROOT"])
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

set_data_root(args.data_root)
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg = NeuralGraphConfig.from_yaml(args.config or os.path.join(args.log_dir, "config.yaml"))
def _iter_of(path):
    """The iteration in a checkpoint name, or 0 for the unsuffixed final one.

    Runs from before the per-epoch checkpointing write a single
    best_model_with_0_graphs_0.pt with no iteration in it; globbing only for the
    suffixed form found no checkpoint at all for every one of them.
    """
    m = re.findall(r"_(\d+)\.pt$", os.path.basename(path))
    return int(m[0]) if m else 0


ck = args.checkpoint or sorted(
    glob.glob(os.path.join(args.log_dir, "models", "best_model_with_0_graphs_0*.pt")),
    key=_iter_of)[-1]
it = _iter_of(ck)

sd = torch.load(ck, map_location=dev, weights_only=False)
migrate_state_dict(sd)
cfg.simulation.n_edges = sd["model_state_dict"]["W"].shape[0]
cfg.simulation.n_extra_null_edges = 0
model = create_model(cfg.graph_model.signal_model_name,
                     aggr_type=cfg.graph_model.aggr_type, config=cfg, device=dev)
model.load_state_dict(sd["model_state_dict"], strict=False)
model.eval()

if not cfg.dataset.startswith("fly/"):
    cfg.dataset = "fly/" + cfg.dataset
data = init_training_data(cfg, dev, args.log_dir, logging.getLogger("repanel"))
model.edges = data.edges

neurons = args.neurons
if not neurons:
    # ONE PER TYPE, and only among the types that actually receive in the band.
    # Picking by index alone gave four neurons of the same lamina type twice
    # over; the panels are worth showing precisely because the types differ.
    e = to_numpy(data.edges).reshape(2, -1)
    deg = np.bincount(e[1].astype(int), minlength=int(data.n_neurons))
    types = np.asarray(to_numpy(data.type_list), dtype=float).ravel().astype(int)
    ok = np.where((deg >= args.min_deg) & (deg <= args.max_deg))[0]
    rng = np.random.default_rng(args.seed)
    seen, neurons = set(), []
    for i in rng.permutation(ok):
        t = int(types[i])
        if t in seen:
            continue
        seen.add(t)
        neurons.append(int(i))
        if len(neurons) >= args.n_types:
            break

from connectome_gnn.metrics import INDEX_TO_NAME  # noqa: E402
types = np.asarray(to_numpy(data.type_list), dtype=float).ravel().astype(int)
print(f"checkpoint iter {it}")
for n in neurons:
    print(f"  neuron {n}  type {INDEX_TO_NAME.get(int(types[n]), '?')}")

cfg.analysis = cfg.analysis.model_copy(update={"neurons": list(neurons)})
os.makedirs(args.out, exist_ok=True)
paths = analyse_neurons(cfg, model, data, args.log_dir, device=dev, out_dir=args.out,
                        tag=f"{it:08d}", sr_enabled=False, use_rollout=False, quiet=False)
for q in paths:
    print("wrote", q)
