"""Clustering accuracy on the template readout's parameters against the gain-correction chain's.

    PYTHONPATH=src python tools/clustering_template_vs_chain.py <run> [<run> ...] [--device cuda:0]

The plot pass clusters the cell types (metrics.cluster_recovery: a 100-component Gaussian
mixture on the embedding a_i, tau_i, V_rest_i and eight statistics of the learned W around
each neuron) on the CHAIN's tau, V_rest and corrected W, while W, tau and V_rest are reported
from the template readout. Per run this prints:

  chain_metrics     clustering_accuracy as results/metrics.txt has it (chain features);
  chain_refit       the same mixture refitted on results/extras/clustering_features.npz
                    X_full, the stack that number was computed from: checks the refit
                    reproduces it, so the template number below differs by its features only
                    (null where the run's plot pass predates that file);
  template          the mixture on the template readout's W (per edge, unfitted edges 0 as
                    cluster_recovery reads nan), tau and offset-corrected V_rest, plus the
                    same embedding; n_features says which quantities entered (one not
                    finite on every neuron is left out of the stack).

Writes results/extras/clustering_template_vs_chain.json in each run folder.
"""
import argparse
import json
import os

import numpy as np
import torch

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.generators.ode_params import load_ode_params_for_run
from connectome_gnn.metrics import cluster_recovery, extract_template_params
from connectome_gnn.models.registry import create_model
from connectome_gnn.sparsify import clustering_gmm
from connectome_gnn.utils import migrate_state_dict, set_data_root, sort_key
from connectome_gnn.zarr_io import load_simulation_data

ROOT = "/groups/saalfeld/home/allierc/GraphData"


def load(run, device):
    import glob
    log_dir = f"{ROOT}/log/fly/{run}"
    config = NeuralGraphConfig.from_yaml(f"{log_dir}/config.yaml")
    if not config.dataset.startswith("fly/"):
        config.dataset = "fly/" + config.dataset
    x_ts = load_simulation_data(f"{ROOT}/graphs_data/{config.dataset}/x_list_train",
                                fields=["index", "voltage", "stimulus", "neuron_type", "group_type"])
    edges = torch.load(f"{log_dir}/training_edges.pt", map_location=device, weights_only=False)
    ckpt = sorted(glob.glob(f"{log_dir}/models/best_model_with_*"), key=sort_key)[-1]
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    migrate_state_dict(sd)
    config.simulation.n_edges = sd["model_state_dict"]["W"].shape[0]
    config.simulation.n_extra_null_edges = 0
    model = create_model(config.graph_model.signal_model_name, aggr_type=config.graph_model.aggr_type,
                         config=config, device=device)
    model.load_state_dict(sd["model_state_dict"], strict=False)
    model.eval()
    return log_dir, config, x_ts, edges, model


def chain_numbers(log_dir, type_ids):
    acc = None
    for line in open(f"{log_dir}/results/metrics.txt"):
        if line.startswith("clustering_accuracy:"):
            acc = float(line.split(":")[1])
    path = f"{log_dir}/results/extras/clustering_features.npz"
    if not os.path.exists(path):       # plot passes older than b18972ef saved no stack
        return acc, None, None
    X = np.load(path)["X_full"]
    n = X.shape[0]
    refit = clustering_gmm(X, type_ids[:n], n_components=min(100, n - 1))["accuracy"]
    return acc, float(refit), int(X.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    set_data_root(ROOT)
    dev = torch.device(args.device)
    for run in args.runs:
        log_dir, config, x_ts, edges, model = load(run, dev)
        n = int(x_ts.n_neurons)
        ode_params = load_ode_params_for_run(config, device="cpu")
        type_ids = x_ts.neuron_type.cpu().numpy().astype(int).ravel()
        rec = extract_template_params(model, ode_params, config=config, edges=edges, x_ts=x_ts,
                                      device=dev, n_neurons=n)
        W = rec.diagnostics["_W_learned_full"]
        tau = rec.pairs["tau"][1] if rec.pairs.get("tau") is not None else None
        vrest = rec.pairs["V_rest"][1] if rec.pairs.get("V_rest") is not None else None
        emb = getattr(model, "_orig_mod", model).a[:n].detach().cpu().numpy()
        cl = cluster_recovery(type_ids, edges.cpu().numpy(), W, n, embedding=emb,
                              learned_tau=tau, learned_vrest=vrest,
                              n_components=min(100, n - 1))
        acc_m, acc_refit, nf_chain = chain_numbers(log_dir, type_ids)
        out = {"run": run, "chain_metrics": acc_m, "chain_refit": acc_refit,
               "chain_n_features": nf_chain,
               "template": cl["clustering_accuracy"], "template_n_features": cl["clustering_n_features"],
               "template_tau_finite": bool(tau is not None and np.isfinite(tau).all()),
               "template_vrest_finite": bool(vrest is not None and np.isfinite(vrest).all()),
               "template_W_fitted_frac": float(np.isfinite(W).mean())}
        print(json.dumps(out), flush=True)
        json.dump(out, open(f"{log_dir}/results/extras/clustering_template_vs_chain.json", "w"), indent=1)


if __name__ == "__main__":
    main()
