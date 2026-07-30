"""Precision-horizon metrics for the swim-integration models (CLI wrapper).

Rollout Pearson saturates (~0.99 for every connectome-locked integrator over a
10 s trial), so it cannot discriminate models. Instead we drive a long
*naturalistic* rollout (Ornstein--Uhlenbeck angular velocity ω(t) and forward
speed v_fwd(t)) and measure how the error grows. The metric itself lives in
``connectome_gnn.models.bump_attractor_eval.precision_horizon_metrics`` (the
same code the path-integration test, ``GNN_Main.py -o test``, now emits):

  * tau_theta (s)    -- heading precision horizon: time the decoded heading
                        stays within ``thr_deg`` (default 15 deg) of truth.
  * |g_theta-1|      -- heading integration-gain error (the ω-independent
                        driver of heading drift).
  * tau_d (s)        -- displacement precision horizon; meaningful only for the
                        LEAKY (bounded) integrators (a cumulative target leaves
                        its trained range over a 60 s rollout, so its horizon
                        measures extrapolation, not integration quality).

Usage:
    python figures/zebrafish/precision_horizon.py --runs <run_basename> ... \
        --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from fig_zebrafish_mi_partition import _load_run  # noqa: E402
from connectome_gnn.models.bump_attractor_eval import (  # noqa: E402
    precision_horizon_metrics,
)
from connectome_gnn.utils import (  # noqa: E402
    load_data_root_from_json, set_data_root,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True,
                   help="trained-run directory basenames under log/zebrafish/")
    p.add_argument("--data_root",
                   default="/groups/saalfeld/home/allierc/GraphData")
    p.add_argument("--device", default="cuda")
    p.add_argument("--thr_deg", type=float, default=15.0)
    p.add_argument("--rel_d", type=float, default=0.20)
    p.add_argument("--n_seed", type=int, default=4)
    args = p.parse_args()
    try:
        set_data_root(load_data_root_from_json())
    except FileNotFoundError:
        pass
    dev = torch.device(args.device)
    for rn in args.runs:
        net, _ = _load_run(
            os.path.join(args.data_root, "log", "zebrafish", rn), dev)
        m = precision_horizon_metrics(
            net, device=dev, thr_deg=args.thr_deg, rel_d=args.rel_d,
            leaky=("leaky" in rn), n_seed=args.n_seed)
        td = f"{m['tau_d_s']:.1f}s" if m["tau_d_s"] is not None else "--"
        print(f"{rn:50s}  tau_theta={m['tau_theta_s']:5.1f}s  "
              f"|g-1|={m['heading_gain_err']:.3f}  tau_d={td}")


if __name__ == "__main__":
    main()
