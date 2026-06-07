"""Training-evolution figures for every ARTR / pt-IPN1 production variant.

The refined 839-cell circuit of Figure 1c (angular drive omega routed
exclusively to ARTR, forward-swim v_fwd to pt-IPN1 and optionally to
motor_efferent) is trained as a family of sub-tasks projected from one
on-disk swim-integration superset (see ``method:swim_task_modes``):

    selfmotion_rotation           — rotation only        (n_in=3, n_out=2)
    selfmotion_translation        — translation only     (n_in=1, n_out=1, perfect d)
    selfmotion_translation_leaky  — translation only     (n_in=1, n_out=1, leaky d)
    selfmotion_both               — rotation + scalar d  (n_in=4, n_out=3, perfect)
    selfmotion_both_leaky         — rotation + scalar d  (n_in=4, n_out=3, leaky)
    position_2d                   — rotation + (x, y)    (n_in=4, n_out=4, perfect)
    position_2d_leaky             — rotation + (x, y)    (n_in=4, n_out=4, leaky)

This wrapper iterates the shared loader / renderer in
``figures/zebrafish/fig_evolution.py`` over each run dir present and
writes one ``fig_evolution_<run-name>.png`` per model.

Usage:
    python figures/zebrafish/fig_evolution_ipn12_artr_pt1.py
    python figures/zebrafish/fig_evolution_ipn12_artr_pt1.py \\
        --data_root /groups/saalfeld/home/allierc/GraphData \\
        --out_dir figures/zebrafish/
    python figures/zebrafish/fig_evolution_ipn12_artr_pt1.py \\
        --only selfmotion_rotation position_2d
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


PREFIX = "zebrafish_hd_si_ipn12_artr_pt1_"
VARIANTS = [
    # (suffix, short label used in console messages)
    ("selfmotion_rotation",           "rotation only"),
    ("selfmotion_translation",        "translation only (perfect d)"),
    ("selfmotion_translation_leaky",  "translation only (leaky d)"),
    ("selfmotion_both",               "rotation + scalar d (perfect)"),
    ("selfmotion_both_leaky",         "rotation + scalar d (leaky)"),
    ("position_2d",                   "rotation + 2D path (perfect)"),
    ("position_2d_leaky",             "rotation + 2D path (leaky)"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data_root",
        default="/groups/saalfeld/home/allierc/GraphData",
        help="GraphData root; run dirs are <data_root>/log/zebrafish/<run>/.")
    p.add_argument("--out_dir", default=HERE)
    p.add_argument("--snapshot_n_steps", type=int, default=1000)
    p.add_argument("--snapshot_omega_deg", type=float, default=60.0)
    p.add_argument("--trial_seed", type=int, default=None)
    p.add_argument("--trial_idx", type=int, default=None)
    p.add_argument("--only", nargs="+", default=None,
                   help="render only these suffixes (e.g. selfmotion_rotation)")
    args = p.parse_args()

    from connectome_gnn.plot_cx import plot_cx_evolution

    os.makedirs(args.out_dir, exist_ok=True)
    wanted = set(args.only) if args.only else None
    for suffix, label in VARIANTS:
        if wanted is not None and suffix not in wanted:
            continue
        run_name = PREFIX + suffix
        run_dir = os.path.join(args.data_root, "log", "zebrafish", run_name)
        if not os.path.isdir(run_dir):
            print(f"[fig_evolution_ipn12_artr_pt1] SKIP {label}: "
                  f"missing {run_dir}")
            continue
        try:
            data = _load_model_and_rollouts(
                run_dir,
                snapshot_n_steps=args.snapshot_n_steps,
                snapshot_omega_deg=args.snapshot_omega_deg,
                trial_seed=args.trial_seed,
                trial_idx=args.trial_idx,
            )
            out_path = os.path.join(args.out_dir,
                                     f"fig_evolution_{run_name}.png")
            # n_rows=4: 4×5 macro layout — rows 0-1 hold the 4-column
            # panels a-h, rows 2-3 hold the 5-column test rows i, j.
            # One matplotlib figure, no PNG montage.
            plot_cx_evolution(data, out_path, run_dir=run_dir, n_rows=4)
            print(f"[fig_evolution_ipn12_artr_pt1] {label}: wrote {out_path} "
                  f"(from {os.path.basename(data['checkpoint'])})")
        except Exception as e:
            print(f"[fig_evolution_ipn12_artr_pt1] FAIL {label}: "
                  f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
