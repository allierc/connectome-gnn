"""Paper figure: zebrafish_hd_si training evolution (4 x 2 / 4 x 3 panels).

Companion of figures/drosophila_cx/fig_evolution.py. Same panel layout
(a–h in two-row mode, a–l with extras in three-row mode). All rendering
is shared via ``connectome_gnn.plot_cx.plot_cx_evolution``; this file is
just the CLI / data-loading shim that:

  * loads a zebrafish run directory (``<data_root>/log/zebrafish/<run>/``),
  * runs the same deterministic constant-ω rollout used by the
    drosophila companion (the model integrates ω regardless of whether
    ω came from an OU stream or a swim-impulse boxcar, so a
    constant-ω probe is still meaningful for the bump trajectory),
  * picks one swim-integration test trial,
  * passes the species-specific axis labels (``r1π / dIPN`` for the
    bump cells, ``RIPN / pt-IPN`` for the afferents) through to the
    figure builder.

Usage:
    python figures/zebrafish/fig_evolution.py \\
        --run_dir /groups/saalfeld/home/allierc/GraphData/log/zebrafish/zebrafish_hd_si_ipn12_v1_cv0 \\
        --out_dir figures/zebrafish/
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


# The evolution-figure data loader now lives in the shared module
# connectome_gnn.plot_cx (load_evolution_data) so the fly + fish entry
# scripts share one loader + one plotter. Re-exported under the historical
# name for callers doing ``from fig_evolution import _load_model_and_rollouts``
# (e.g. fig_evolution_ipn12_artr_pt1.py).
from connectome_gnn.plot_cx import (  # noqa: E402
    load_evolution_data as _load_model_and_rollouts,
    plot_evolution,
)


# --- CLI -----------------------------------------------------------------


DEFAULT_RUN_DIRS = [
    "/groups/saalfeld/home/allierc/GraphData/log/zebrafish/zebrafish_hd_si_ipn12_v1_cv0",
    "/groups/saalfeld/home/allierc/GraphData/log/zebrafish/zebrafish_hd_si_gnn_ipn12_v1_cv0",
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
                    help="seed picking the swim-integration test trial "
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
        # n_rows=4: 4×5 macro layout — rows 0-1 hold the 4-column
        # panels a-h, rows 2-3 hold the 5-column test rows i, j (random
        # held-out trials, constant-input deterministic sweeps). One
        # matplotlib figure, no PNG montage.
        plot_evolution(data, out_path, run_dir=run_dir, n_rows=4)


if __name__ == "__main__":
    main()
