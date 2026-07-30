"""Paper figure: GNN dashboard --- companion of the RNN evolution figure for
the message-passing GNN variant (zebrafish_hd_si_gnn).

Thin entry point: the data-loading probe (``load_evolution_data``, re-exported
here as ``_load_model_and_rollouts``) and the panel layout
(``plot_cx.plot_gnn_dashboard``) are both organism-agnostic and shared with the
Drosophila CX companion (``figures/drosophila_cx/fig_gnn_dashboard.py``). Rows
1-2 mirror the RNN evolution figure (Fig. 4 a-h); row 3 adds the three
GNN-specific panels --- the per-neuron embedding a_i, the learned per-edge
signalling function g_phi(a_j, v_j), and the node update f_theta(a_i, v_i).

Usage:
    python figures/zebrafish/fig_gnn_dashboard.py \\
        --run_dir /groups/.../log/zebrafish/zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation \\
        --out_dir figures/zebrafish/
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
# Sibling shim: the shared loader is re-exported by fig_evolution; _despine
# (open-axes styling) is picked up by plot_gnn_dashboard from this dir.
sys.path.insert(0, _HERE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir", required=True,
        help="GNN training-run directory (with config.yaml, models/).")
    p.add_argument(
        "--out_dir", default=_HERE,
        help="output directory (default: figures/zebrafish/).")
    p.add_argument("--snapshot_n_steps", type=int, default=1500)
    p.add_argument("--snapshot_omega_deg", type=float, default=60.0)
    p.add_argument("--trial_seed", type=int, default=None)
    p.add_argument("--trial_idx", type=int, default=None)
    args = p.parse_args()

    from connectome_gnn.plot_cx import plot_gnn_dashboard
    from fig_evolution import _load_model_and_rollouts

    data = _load_model_and_rollouts(
        args.run_dir,
        snapshot_n_steps=args.snapshot_n_steps,
        snapshot_omega_deg=args.snapshot_omega_deg,
        trial_seed=args.trial_seed,
        trial_idx=args.trial_idx,
    )
    net = data["net"]
    if not all(hasattr(net, n) for n in ("a", "g_phi", "f_theta")):
        raise SystemExit(
            f"[fig_gnn_dashboard] {args.run_dir} is not a GNN run "
            f"(model has no a/g_phi/f_theta); use fig_evolution.py instead.")
    print(f"[fig_gnn_dashboard] loaded {data['checkpoint']}")
    out_path = os.path.join(
        args.out_dir,
        f"fig_gnn_dashboard_{os.path.basename(os.path.abspath(args.run_dir))}.png",
    )
    plot_gnn_dashboard(data, out_path)
    print(f"[fig_gnn_dashboard] wrote {out_path}")


if __name__ == "__main__":
    main()
