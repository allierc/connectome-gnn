"""Score one run's synapses twice: the correction chain, and the template fit.

    python tools/pysr_recovery.py <config> [<config> ...] [--frames 64] [--gauge-tau model|true]

For each run this loads the best checkpoint and computes W_ij and E_ij two ways:

  metrics.txt        `extract_recovered_params`, the estimator the run already
                     used -- the per-edge line through msg/v_j, or the g_phi /
                     f_theta correction chain, depending on the family
  metrics_pysr.txt   `extract_template_params`, the generator's own closed form
                     msg_ij = W*act(v_j)*(E - v_i) fitted per edge with act GIVEN,
                     then W scaled into the generator's units per neuron by
                     k_i = tau_i * dftheta_dmsg_i

Both write the same `<key>_<stat>` names, so the two files diff line by line and
the table printed here is that diff. What to look for: `Wij_gain` near 1.0 says
the per-neuron gauge did its job where a single global factor could not, and
`Eij_gate` (the median per-edge fit R2) says whether the form fits at all -- a
high gain error under a high gate means the scale is wrong, a low gate means the
model's message is not a conductance in the first place and neither W nor E from
it means anything.
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from connectome_gnn.metrics import (extract_recovered_params,            # noqa: E402
                                    extract_template_params, metrics_lines,
                                    score_recovery, write_recovery_metrics)
from connectome_gnn.models.registry import create_model                  # noqa: E402
from connectome_gnn.models.training_utils import init_training_data      # noqa: E402
from connectome_gnn.models.utils import load_run_config                  # noqa: E402
from connectome_gnn.utils import log_path, migrate_state_dict            # noqa: E402

# The keys worth putting side by side. Both readouts emit every `<key>_<stat>`;
# these are the ones that say whether the synapses came back, rather than how
# many outliers were dropped on the way.
COMPARE = ("Wij_R2", "Wij_R2_scaled", "Wij_gain", "Wij_pearson", "Wij_slope",
           "Wij_rel_err_median", "Wij_n",
           "Eij_R2", "Eij_slope", "Eij_rel_err_median", "Eij_gate",
           "Eij_pct_wrong_slope", "Eij_n")


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


def score_both(cfg, data, model, device, n_frames, gauge_tau):
    """(chain, template) scored dicts for one run, on the same frames."""
    edges = data.edges.to(device)
    extra = {} if n_frames is None else {"n_frames": n_frames}
    chain = score_recovery(
        extract_recovered_params(model, data.ode_params, config=cfg, edges=edges,
                                 x_ts=data.x_ts, device=device,
                                 n_neurons=int(data.n_neurons),
                                 need=("W", "E_ij")), cfg)
    tmpl = score_recovery(
        extract_template_params(model, data.ode_params, config=cfg, edges=edges,
                                x_ts=data.x_ts, device=device,
                                n_neurons=int(data.n_neurons),
                                gauge_tau=gauge_tau, **extra), cfg)
    return chain, tmpl


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{float(v):+.4f}" if abs(float(v)) < 1e4 else f"{float(v):.4g}"


def report(name, chain, tmpl):
    print(f"\n=== {name}")
    print(f"{'key':<22}{'chain':>12}{'template':>12}")
    for k in COMPARE:
        if k not in chain and k not in tmpl:
            continue
        print(f"{k:<22}{_fmt(chain.get(k)):>12}{_fmt(tmpl.get(k)):>12}")
    for k in ("tmpl_fit_r2_median", "tmpl_k_median", "tmpl_dfdmsg_median",
              "tmpl_dfdmsg_absmedian", "tmpl_pct_unfitted",
              "tmpl_pct_E_unidentified", "tmpl_pct_W_from_slope",
              "tmpl_n_used_median", "tmpl_vj_floor"):
        if k in tmpl:
            print(f"{k:<22}{'':>12}{_fmt(tmpl[k]):>12}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    # None, not a number: the extractor's own default is the tested one, and a
    # tool default silently overriding it is how the 256-frame fix was measured
    # at 64 frames and looked like it had not taken effect.
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--gauge-tau", choices=("model", "true"), default="model")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    for name in args.configs:
        try:
            cfg, data, model, log_dir = load_run(name, args.device)
            chain, tmpl = score_both(cfg, data, model, args.device,
                                     args.frames, args.gauge_tau)
            report(name, chain, tmpl)
            if not args.no_write:
                write_recovery_metrics(tmpl, log_dir, name="metrics_pysr.txt")
                print(f"  -> {os.path.join(log_dir, 'results', 'metrics_pysr.txt')}")
        except Exception as exc:
            print(f"\n=== {name}\n  failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
