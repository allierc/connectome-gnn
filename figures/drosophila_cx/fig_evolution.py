"""Paper figure: drosophila_cx_pi training evolution (4 x 2 panels).

Thin CLI shim. All panel rendering lives in
``connectome_gnn.plot_cx.plot_cx_evolution`` (the public entry point),
which is also imported by the training-time snapshot helper in
``connectome_gnn.models.bump_attractor_eval._save_training_snapshot``.
That centralisation removed the importlib hack that used to load this
file via a hard-coded path; the bulk of the panel code (formerly here,
~700 lines of ``_panel_*`` helpers + ``build_figure``) now lives in
``plot_cx`` alongside ``plot_cx_matrix``, ``plot_cx_training_snapshot``
etc.

What this file still does:

    * ``_load_model_and_rollouts``: load a run directory's checkpoint,
      run the deterministic constant-ω rollout + the integration-gain
      sweep + pick an OU test trial. Produces the ``data`` dict that
      ``plot_cx_evolution`` consumes.
    * ``main``: argparse, iterate over run dirs, write one PNG each.

Usage:
    python figures/drosophila_cx/fig_evolution.py \
        --run_dir /groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi \
        --out_dir figures/drosophila_cx/
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))


# Evolution-figure data loader is shared via connectome_gnn.plot_cx
# (load_evolution_data); re-export the historical name for back-compat.
from connectome_gnn.plot_cx import (  # noqa: E402
    load_evolution_data as _load_model_and_rollouts,
    plot_evolution,
)



# --- CLI -----------------------------------------------------------------


DEFAULT_RUN_DIRS = [
    "/groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi",
    "/groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi_frozen_Wrec",
    "/groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi_fc",
    "/groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi_gnn",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir", action="append", default=None,
        help="training-run directory (with config.yaml, models/, ...). "
             "May be passed multiple times to generate one figure per run.")
    p.add_argument(
        "--out_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="output directory. Each figure is written as "
             "fig_evolution_<run_basename>.png.")
    p.add_argument("--snapshot_n_steps", type=int, default=1500)
    p.add_argument("--snapshot_omega_deg", type=float, default=60.0)
    p.add_argument("--trial_seed", type=int, default=None,
                    help="seed picking the OU test trial "
                         "(default: config.training.seed + 17)")
    p.add_argument("--trial_idx", type=int, default=None,
                    help="explicit test-trial index (overrides --trial_seed).")
    args = p.parse_args()

    
    run_dirs = args.run_dir or DEFAULT_RUN_DIRS
    os.makedirs(args.out_dir, exist_ok=True)
    for run_dir in run_dirs:
        try:
            data = _load_model_and_rollouts(
                run_dir,
                snapshot_n_steps=args.snapshot_n_steps,
                snapshot_omega_deg=args.snapshot_omega_deg,
                trial_seed=args.trial_seed,
                trial_idx=args.trial_idx,
            )
        except Exception as exc:
            print(f"[fig_evolution] SKIP {run_dir}: {exc}")
            continue
        print(f"[fig_evolution] loaded {data['checkpoint']}")
        out_path = os.path.join(
            args.out_dir,
            f"fig_evolution_{os.path.basename(os.path.abspath(run_dir))}.png",
        )
        plot_evolution(data, out_path, run_dir=run_dir, n_rows=4)


if __name__ == "__main__":
    main()
