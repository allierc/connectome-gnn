"""Generate the two IPN12 training-evolution figures for zebrafish.tex:

  Figure 3  ->  fig_evolution_zebrafish_hd_si_ipn12_v1.png  (IPN12 inhibitory)
  Figure 4  ->  fig_evolution_zebrafish_hd_si_ipn12_v2.png  (IPN12 excitatory)

Both use the same panel layout as Figure 3 of docs/drosophila.tex
(figures/drosophila_cx/fig_evolution.py): GT vs learned recurrent
matrix, all-neuron / afferent / dIPN-ring kinographs, constant-omega and
swim-test rollouts, per-cell-type subthreshold-state violins, and the
integration-gain sweep. This is a thin wrapper over the shared loader /
renderer in figures/zebrafish/fig_evolution.py so both panels are
produced (and kept in sync) from one command.

Usage:
    python figures/zebrafish/fig_evolution_ipn12.py
    python figures/zebrafish/fig_evolution_ipn12.py \\
        --data_root /groups/saalfeld/home/allierc/GraphData \\
        --out_dir figures/zebrafish/
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from fig_evolution import _load_model_and_rollouts  # noqa: E402


# (run-dir basename, paper-figure label) for the two Dale-sign variants.
VARIANTS = [
    ("zebrafish_hd_si_ipn12_v1", "Figure 3 (IPN12 inhibitory)"),
    ("zebrafish_hd_si_ipn12_v2", "Figure 4 (IPN12 excitatory)"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data_root",
        default="/groups/saalfeld/home/allierc/GraphData",
        help="GraphData root; run dirs are <data_root>/log/zebrafish/<run>/.")
    p.add_argument("--out_dir", default=HERE)
    p.add_argument("--snapshot_n_steps", type=int, default=1500)
    p.add_argument("--snapshot_omega_deg", type=float, default=60.0)
    p.add_argument("--trial_seed", type=int, default=None)
    p.add_argument("--trial_idx", type=int, default=None)
    args = p.parse_args()

    from connectome_gnn.plot_cx import plot_cx_evolution

    os.makedirs(args.out_dir, exist_ok=True)
    for run_name, label in VARIANTS:
        run_dir = os.path.join(args.data_root, "log", "zebrafish", run_name)
        if not os.path.isdir(run_dir):
            print(f"[fig_evolution_ipn12] SKIP {label}: missing {run_dir}")
            continue
        data = _load_model_and_rollouts(
            run_dir,
            snapshot_n_steps=args.snapshot_n_steps,
            snapshot_omega_deg=args.snapshot_omega_deg,
            trial_seed=args.trial_seed,
            trial_idx=args.trial_idx,
        )
        out_path = os.path.join(args.out_dir, f"fig_evolution_{run_name}.png")
        plot_cx_evolution(data, out_path, run_dir=run_dir)
        print(f"[fig_evolution_ipn12] {label}: wrote {out_path} "
              f"(from {os.path.basename(data['checkpoint'])})")


if __name__ == "__main__":
    main()
