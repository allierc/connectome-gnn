"""Load one run -- its config, its training data and its best checkpoint -- for the tools.

    from run_loader import load_run
    cfg, data, model, log_dir = load_run(config_name, device)

Shared by tools/eij_by_weight.py, gain_probe.py and wij_scatter.py (it lived in
tools/pysr_recovery.py, the chain-versus-template comparison removed with the
gain-correction chain on 2026-10-13).
"""
import glob
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from connectome_gnn.models.registry import create_model                  # noqa: E402
from connectome_gnn.models.training_utils import init_training_data      # noqa: E402
from connectome_gnn.models.utils import load_run_config                  # noqa: E402
from connectome_gnn.utils import log_path, migrate_state_dict            # noqa: E402


def load_run(config_name, device):
    """The config, the data and the trained model of one run."""
    cfg, _yaml = load_run_config(config_name, False, "plot")
    log_dir = log_path(cfg.config_file)
    data = init_training_data(cfg, device, log_dir, logging.getLogger(__name__))
    model = create_model(cfg.graph_model.signal_model_name,
                         aggr_type=cfg.graph_model.aggr_type, config=cfg,
                         device=device).to(device)
    ck = sorted(glob.glob(os.path.join(log_dir, "models", "best_model_with_*_graphs_*.pt")))
    if not ck:
        raise FileNotFoundError(f"no checkpoint in {log_dir}/models")
    sd = torch.load(ck[-1], map_location=device, weights_only=False)
    # torch.compile prefixes every key with _orig_mod.; without this migration
    # load_state_dict(strict=False) accepts the checkpoint and loads NOTHING,
    # and every number below would describe an untrained model.
    migrate_state_dict(sd)
    _missing, unexpected = model.load_state_dict(sd["model_state_dict"], strict=False)
    n_loaded = len(sd["model_state_dict"]) - len(unexpected)
    if n_loaded == 0:
        raise RuntimeError(f"checkpoint {ck[-1]} loaded 0 tensors")
    model.eval()
    return cfg, data, model, log_dir
