"""Redraw results/neuron<N>_panels.png for a finished run, nothing else.

WHY NOT `-o test_plot`. The panels are one figure out of a pass that also runs a
rollout, four scatters, the clustering and the embedding -- half an hour for a
picture that takes three minutes. This loads the checkpoint, calls
`analyse_neurons`, and stops, which is what a panel-only fix needs.

Usage:
    python tools/rebuild_neuron_panels.py <config> [--device cuda:0]
"""

import argparse
import glob
import logging
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="config name, e.g. flyvis_current_noise_005_current_cv02")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    from connectome_gnn.models.utils import load_run_config
    from connectome_gnn.utils import log_path, migrate_state_dict
    from connectome_gnn.models.training_utils import init_training_data
    from connectome_gnn.models.registry import create_model
    from connectome_gnn.neuron_panels import analyse_neurons
    from connectome_gnn import pysr_env

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    config, _ = load_run_config(args.config, False, "plot")
    if not (getattr(config, "analysis", None) and config.analysis.neurons):
        print(f"{args.config}: no analysis.neurons, nothing to draw")
        return
    log_dir = log_path(config.config_file)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = init_training_data(config, device, log_dir, logger)

    model = create_model(config.graph_model.signal_model_name,
                         aggr_type=config.graph_model.aggr_type,
                         config=config, device=device).to(device)
    ckpts = sorted(glob.glob(os.path.join(log_dir, "models",
                                          "best_model_with_*_graphs_*.pt")))
    if not ckpts:
        sys.exit(f"no checkpoint in {log_dir}/models")
    state = torch.load(ckpts[-1], map_location=device, weights_only=False)
    migrate_state_dict(state)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    written = analyse_neurons(config, model, data, log_dir, device=device, logger=logger)
    for p in written:
        print(f"wrote {p}")
    # Same rule as the plot pass: one README when PySR could not run, and none
    # when it could.
    note = pysr_env.write_status(log_dir)
    if note:
        print(f"PySR did not run -> {note}")


if __name__ == "__main__":
    main()
