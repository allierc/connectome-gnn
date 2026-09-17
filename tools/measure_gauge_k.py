"""k_i = tau_i * df/dmsg_i on real frames. The gauge is fixed when k_i = 1.

This is the quantity coeff_f_theta_msg_gain exists to drive to 1, measured the
way the readout measures it -- autograd on frames the network actually visits,
not a synthetic probe.
"""
import sys, os, glob, re, numpy as np, torch
sys.path.insert(0, "src")
from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.registry import create_model
from connectome_gnn.utils import migrate_state_dict
from connectome_gnn.metrics import _template_gauge
from connectome_gnn.generators.ode_params import load_ode_params_for_run
from connectome_gnn.zarr_io import load_simulation_data
from connectome_gnn.utils import graphs_data_path, set_data_root

set_data_root("/groups/saalfeld/home/allierc/GraphData")
LOG = "/groups/saalfeld/home/allierc/GraphData/log/fly"
dev = "cuda" if torch.cuda.is_available() else "cpu"
import json
OUT = "docs/gauge_k.json"
cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
print(f"{'run':30s} {'iter':>8s} {'k median':>9s} {'k IQR':>16s} {'%|k-1|<0.2':>11s}")
for run in sys.argv[1:]:
    cfg = NeuralGraphConfig.from_yaml(f"config/fly/{run}.yaml")
    import glob, re
    cks = sorted(glob.glob(f"{LOG}/{run}/models/best_model_with_0_graphs_0_*.pt"),
                 key=lambda f: int(re.findall(r"_(\d+)\.pt$", f)[0]))
    if not cks:
        print(f"{run[-30:]:30s} {'no ckpt':>8s}"); continue
    it = int(re.findall(r"_(\d+)\.pt$", cks[-1])[0])
    sd = torch.load(cks[-1], map_location=dev, weights_only=False); migrate_state_dict(sd)
    cfg.simulation.n_edges = sd["model_state_dict"]["W"].shape[0]
    cfg.simulation.n_extra_null_edges = 0
    model = create_model(cfg.graph_model.signal_model_name,
                         aggr_type=cfg.graph_model.aggr_type, config=cfg, device=dev)
    model.load_state_dict(sd["model_state_dict"], strict=False); model.eval()
    cfg.dataset = "fly/" + cfg.dataset if not cfg.dataset.startswith("fly/") else cfg.dataset
    op = load_ode_params_for_run(cfg, device=dev)
    x_ts = load_simulation_data(
        graphs_data_path(cfg.dataset, "x_list_train")).truncate_frames(2000)
    edges = torch.load(f"{LOG}/{run}/training_edges.pt",
                       map_location=dev, weights_only=False)
    model.edges = edges
    N = cfg.simulation.n_neurons
    # gauge_tau="model": no ground truth anywhere, k collapses to G.
    k, dfdmsg, tau, _ = _template_gauge(model, cfg, op, edges, x_ts, N, dev,
                                        n_frames=8, seed=0, gauge_tau="model", T=None, G=None)
    k = k[np.isfinite(k)]
    q1, q3 = np.percentile(k, [25, 75])
    frac = 100.0 * np.mean(np.abs(k - 1) < 0.2)
    cache[run] = dict(iteration=it, k_median=float(np.median(k)),
                      k_q1=float(q1), k_q3=float(q3), pct_near_one=float(frac))
    json.dump(cache, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"{run[-30:]:30s} {it:8d} {np.median(k):9.3f} [{q1:6.3f},{q3:6.3f}] {frac:10.1f}%")
