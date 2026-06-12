"""Paper figure: Drosophila CX GNN dashboard --- companion of the RNN
evolution figure (``fig_evolution.py``) for the message-passing GNN variant
(``drosophila_cx_pi_gnn`` / the ``gnn_rotation`` configs).

Thin entry point, the exact mirror of ``figures/zebrafish/fig_gnn_dashboard.py``:
the data-loading probe (``load_evolution_data``, re-exported by the drosophila
``fig_evolution`` as ``_load_model_and_rollouts``) and the panel layout
(``connectome_gnn.plot_cx.plot_gnn_dashboard``) are both organism-agnostic and
shared with the zebrafish dashboard --- they key off the run's
``neuron_types`` / ``type_names`` and the model's ``a`` / ``g_phi`` /
``f_theta``, so the fly circuit (338 cells, EPG ring) renders through the same
builder as the fish.

  Rows 1-2 (a-h): GT connectome W_con, learned per-edge gain W_rec, per-edge
                  weight scatter, all-neuron kinograph, phase-sorted (EPG-ring)
                  kinograph, decoded-vs-true heading tracking, one held-out
                  test trial, integration-gain scatter --- mirrors Fig. 4 a-h.
  Row 3 (i-k):    GNN-specific --- per-neuron embedding a_i, learned per-edge
                  signalling g_phi(a_j, v_j), node update f_theta(a_i, v_i).

Usage:
    python figures/drosophila_cx/fig_gnn_dashboard.py \\
        --run_dir /groups/.../log/drosophila_cx/gnn_rotation \\
        --out_dir figures/drosophila_cx/
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
# Sibling shim: this dir for fig_evolution (the shared-loader re-export); the
# zebrafish figures dir so plot_gnn_dashboard finds _despine (open-axes style).
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "zebrafish"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir", required=True,
        help="GNN training-run directory (with config.yaml, models/), e.g. "
             ".../log/drosophila_cx/gnn_rotation.")
    p.add_argument(
        "--out_dir", default=_HERE,
        help="output directory (default: figures/drosophila_cx/).")
    # Fly heading drifts fast; a shorter constant-omega snapshot keeps the
    # ring legible. Override to match the zebrafish defaults if desired.
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
