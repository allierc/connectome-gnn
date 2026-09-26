#!/usr/bin/env python
"""Write the railed-neuron share of each template rollout into metrics.txt.

WHY THIS EXISTS. `template_rollout_r` can be a finite number produced entirely
by the +-100 V divergence clamp in graph_tester: experiment 2's lasso-100 arm
reported r = 0.112-0.120 on ten independent runs because 67-71% of its 13,741
neurons were pinned on that rail. The tester now counts the clamp as it fires
and writes `template_rollout_pct_clamped`, but every run that landed before
2026-09-23 predates that field, and re-running `-o test_plot` on 35 runs to
recover one number is not worth an hour of GPU each.

The saved per-neuron RMSE arrays are enough. A neuron whose rollout RMSE is at
the rail against a signal whose 99th-percentile |v| is about 3.4 V spent
essentially the whole trajectory there.

NOT THE SAME QUANTITY AS THE TESTER'S, and named so. The tester counts
neuron-FRAMES at the clamp over all frames; this counts NEURONS whose whole-
trajectory RMSE sits at the rail. They answer the same question -- is this r a
measurement or a rail -- and they are not interchangeable, so the report prefers
the tester's when a run has it.

    python tools/backfill_clamp_share.py [--dry-run]
"""
import argparse
import glob
import os
import re

import numpy as np

LOG_ROOT = f"{os.environ['GNN_OUTPUT_ROOT']}/log/fly"
# The clamp graph_tester applies, and how close to it counts as sitting on it.
V_CLAMP = 100.0
RAIL = 0.95 * V_CLAMP
KEY = {"template": "template_rollout_pct_neurons_railed",
       "template_alt": "template_alt_rollout_pct_neurons_railed"}


def shares(run_dir):
    """{metric key: percentage of neurons at the rail} for the run's rollouts."""
    out = {}
    for tag, key in KEY.items():
        hits = glob.glob(os.path.join(
            run_dir, f"results_rollout_on_*_{tag}_rmse.npy"))
        # `template` would otherwise match `template_alt`'s file too.
        if tag == "template":
            hits = [h for h in hits if "_template_alt_" not in h]
        if not hits:
            continue
        r = np.load(hits[0])
        if r.size:
            out[key] = 100.0 * float(np.mean(r > RAIL))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    n_written = 0
    for m in sorted(glob.glob(os.path.join(LOG_ROOT, "*", "results", "metrics.txt"))):
        run_dir = os.path.dirname(os.path.dirname(m))
        text = open(m).read()
        # Idempotent, and never overrides the tester's own in-loop count.
        add = {k: v for k, v in shares(run_dir).items()
               if not re.search(rf"^{k}:", text, re.M)}
        if not add:
            continue
        line = "".join(f"{k}: {v:.2f}\n" for k, v in sorted(add.items()))
        print(f"{os.path.basename(run_dir)}: "
              + "  ".join(f"{k.split('_pct')[0]} {v:.1f}%" for k, v in sorted(add.items())))
        if not args.dry_run:
            with open(m, "a") as f:
                f.write(line)
        n_written += 1
    print(f"{n_written} runs {'would be' if args.dry_run else ''} updated")


if __name__ == "__main__":
    main()
