#!/usr/bin/env python
"""Recover the template-rollout r, RMSE and clamp share that runs on their own
noise-free dataset never wrote into metrics.txt.

WHY. template_rollout.run looked for results_rollout_on_<dataset>_<mode>.log,
but the tester only adds `_on_<dataset>` when the test dataset differs from the
training one. A noise-free run is its own noise-free twin, so its log is
results_rollout_<mode>.log: the rollout ran, the log is on disk, and the
parser found nothing -- so every noise-free row showed a blank fit-roll r.
template_rollout.py now uses the tester's rule; this reads the logs the
already-landed runs wrote, so none of them has to be analysed again.

Only fills keys that are absent, never overwrites; only touches runs whose
metrics.txt names a template rollout model and lacks its r.

    python tools/backfill_template_rollout.py [--dry-run]
"""
import argparse
import glob
import os
import re

from connectome_gnn.template_rollout import _parse_rollout_log

LOG_ROOT = f"{os.environ['GNN_OUTPUT_ROOT']}/log/fly"
PREFIXES = (("template_rollout", "template"), ("template_alt_rollout", "template_alt"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    n = 0
    for m in sorted(glob.glob(os.path.join(LOG_ROOT, "*", "results", "metrics.txt"))):
        run = os.path.dirname(os.path.dirname(m))
        text = open(m).read()
        add = {}
        for prefix, mode in PREFIXES:
            if not re.search(rf"^{prefix}_model:", text, re.M):
                continue
            if re.search(rf"^{prefix}_r:", text, re.M):
                continue
            got = _parse_rollout_log(os.path.join(run, f"results_rollout_{mode}.log"))
            for key, val in (("r", got.get("r")), ("rmse", got.get("rmse")),
                             ("pct_clamped", got.get("pct_clamped"))):
                if val is not None:
                    add[f"{prefix}_{key}"] = val
        if not add:
            continue
        print(os.path.basename(run), " ".join(f"{k.replace('template_', '')}={v:g}" for k, v in add.items()))
        if not a.dry_run:
            with open(m, "a") as f:
                f.write("".join(f"{k}: {v}\n" for k, v in add.items()))
        n += 1
    print(f"{n} runs {'would be ' if a.dry_run else ''}updated")


if __name__ == "__main__":
    main()
