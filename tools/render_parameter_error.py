"""Re-draw results/parameter_error.png for a run that has already been plotted.

WHY A SEPARATE ENTRY POINT. The figure is written by `-o plot`, at the end of a
pass that also runs the rollout, the scatters, the clustering and the embedding
-- tens of minutes and a 1.3 GB rollout bundle for one panel. This does the only
part the panel needs: load the checkpoint, run the same extraction the pass
scores (`extract_recovered_params`, then the template readout on top of it), and
hand the resulting pairs to `_plot_parameter_error`.

IT DOES NOT WRITE metrics.txt. The file already in results/ was written by the
plot pass and carries lines this script cannot produce -- rollout_r, rollout_rmse
and the clustering_* block come from passes that are not run here -- so
rewriting it would silently delete them. The numbers annotated on the panels are
read from the extraction this script performs, which is the same estimator
metrics.txt reports, so the figure and the file agree without either writing the
other.

Usage:
    python tools/render_parameter_error.py fly/flyvis_current_noise_free_current_cv04
    python tools/render_parameter_error.py <config> --device cuda:1
"""

import argparse
import importlib.util
import logging
import os
import sys

import torch


def _load_plotfigure():
    """GNN_PlotFigure as a module: it is a script at the repo root, not a package."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "gnn_plotfigure", os.path.join(root, "GNN_PlotFigure.py"))
    mod = importlib.util.module_from_spec(spec)
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = _argv
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="config name, e.g. fly/flyvis_current_noise_free_current_cv04")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    from connectome_gnn.models.utils import load_run_config
    from connectome_gnn.utils import log_path
    from connectome_gnn.models.training_utils import init_training_data
    from connectome_gnn.models.registry import create_model
    from connectome_gnn.utils import migrate_state_dict
    from connectome_gnn.metrics import (extract_recovered_params,
                                        extract_template_params,
                                        template_readout_enabled,
                                        score_recovery)
    import glob

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    config, _yaml = load_run_config(args.config, False, "plot")
    log_dir = log_path(config.config_file)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"run     : {log_dir}")
    print(f"device  : {device}")

    data = init_training_data(config, device, log_dir, logger)
    model = create_model(config.graph_model.signal_model_name,
                         aggr_type=config.graph_model.aggr_type,
                         config=config, device=device).to(device)
    ckpts = sorted(glob.glob(os.path.join(log_dir, "models", "best_model_with_*_graphs_*.pt")))
    if not ckpts:
        sys.exit(f"no checkpoint in {log_dir}/models -- nothing to extract")
    state = torch.load(ckpts[-1], map_location=device, weights_only=False)
    migrate_state_dict(state)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    print(f"weights : {os.path.basename(ckpts[-1])}")

    n_neurons = int(data.n_neurons)
    rec = extract_recovered_params(model, data.ode_params, config, edges=data.edges,
                                   x_ts=data.x_ts, device=device, n_neurons=n_neurons)
    if template_readout_enabled(config, model):
        # The same precedence the plot pass uses: the template readout supersedes
        # the correction chain for W, E_ij, tau and V_rest and extends its object.
        rec = extract_template_params(model, data.ode_params, config=config,
                                      edges=data.edges, x_ts=data.x_ts, device=device,
                                      n_neurons=n_neurons, base=rec)
    scored = score_recovery(rec, config)

    gpf = _load_plotfigure()
    out = gpf._plot_parameter_error(rec, scored, log_dir)
    if out is None:
        sys.exit("no quantity had a (true, learned) pair -- nothing drawn")
    print(f"wrote   : {out}")
    for key in ("Wij", "tau", "V_rest"):
        med, iqr = scored.get(f"{key}_rel_err_median"), scored.get(f"{key}_rel_err_iqr")
        if med is not None:
            print(f"  {key:7s} rel err median {100 * med:6.2f}%   IQR {100 * iqr:6.2f}%"
                  if iqr is not None else
                  f"  {key:7s} rel err median {100 * med:6.2f}%")


if __name__ == "__main__":
    main()
