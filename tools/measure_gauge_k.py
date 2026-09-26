"""k_i = tau_i * df/dmsg_i on real frames. The gauge is fixed when k_i = 1.

This is the quantity coeff_f_theta_msg_gain exists to drive to 1, measured the
way the readout measures it -- autograd on frames the network actually visits,
not a synthetic probe.
"""
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
from connectome_gnn.generators.ode_params import load_ode_params_for_run
from connectome_gnn.metrics import _template_gauge
from connectome_gnn.models.registry import create_model
from connectome_gnn.utils import graphs_data_path, migrate_state_dict, set_data_root
from connectome_gnn.zarr_io import load_simulation_data

set_data_root(os.environ["GNN_OUTPUT_ROOT"])
LOG = f"{os.environ['GNN_OUTPUT_ROOT']}/log/fly"
dev = "cuda" if torch.cuda.is_available() else "cpu"
import json

OUT = "docs/gauge_k.json"
cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
print(f"{'run':30s} {'iter':>8s} {'k median':>9s} {'k IQR':>16s} {'%|k-1|<0.2':>11s}")
for run in sys.argv[1:]:
    # THE RUN'S OWN CONFIG WHEN THERE IS NO SPEC FILE. The agentic loop writes
    # its arms straight into the log directory and never leaves a yaml under
    # config/fly, so keying only on the spec made the whole campaign unmeasurable
    # here -- which is exactly the set k is wanted for.
    # THE DATA ROOT'S SPEC FOLDER AS A FALLBACK. The agentic loop writes its
    # arms into GraphData/config/fly and never into the repo, so keying only on
    # the repo made the whole campaign unmeasurable here -- which is exactly the
    # set k is wanted for.
    _cands = [f"config/fly/{run}.yaml",
              f"{os.environ['GNN_OUTPUT_ROOT']}/config/fly/{run}.yaml",
              f"{LOG}/{run}/config.yaml"]
    _spec = next((c for c in _cands if os.path.exists(c)), None)
    if _spec is None:
        print(f"{run[-30:]:30s} {'no config':>8s}"); continue
    cfg = NeuralGraphConfig.from_yaml(_spec)
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
    # THE CHECKPOINT'S OWN MLP WIDTH AND DEPTH, not the spec's. The agentic loop
    # rewrites one yaml per arm in place, so a spec edited by a later block (the
    # 256-wide capacity block) no longer describes the 80-wide checkpoint sitting
    # beside it, and load_state_dict then kills the whole sweep at that arm.
    _rebuild = False
    for _pref, _hd, _nl_at in (("g_phi", "hidden_dim", "n_layers"),
                               ("f_theta", "hidden_dim_update", "n_layers_update")):
        _w = sd["model_state_dict"].get(f"{_pref}.layers.0.weight")
        if _w is None:
            continue
        _nl = 1 + max(int(k.split(".")[2]) for k in sd["model_state_dict"]
                      if k.startswith(f"{_pref}.layers.") and k.endswith(".weight"))
        if (int(_w.shape[0]) != getattr(cfg.graph_model, _hd)
                or _nl != getattr(cfg.graph_model, _nl_at)):
            print(f"{run[-30:]:30s} {it:8d}   {_pref} spec "
                  f"{getattr(cfg.graph_model, _hd)}x{getattr(cfg.graph_model, _nl_at)}"
                  f" -> checkpoint {int(_w.shape[0])}x{_nl}")
            setattr(cfg.graph_model, _hd, int(_w.shape[0]))
            setattr(cfg.graph_model, _nl_at, _nl)
            _rebuild = True
    if _rebuild:
        model = create_model(cfg.graph_model.signal_model_name,
                             aggr_type=cfg.graph_model.aggr_type, config=cfg, device=dev)
    model.load_state_dict(sd["model_state_dict"], strict=False); model.eval()
    # k = tau * df/dmsg is only defined where there IS an f_theta. A known-ODE
    # run integrates the generator's own update, so it has no learned gauge to
    # measure and must say so rather than crash on a None feature block.
    if getattr(getattr(model, "_orig_mod", model), "f_theta", None) is None:
        print(f"{run[-30:]:30s} {it:8d}   no f_theta (known ODE)"); continue
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
