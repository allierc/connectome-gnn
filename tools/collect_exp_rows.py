"""Read every run named in docs/experiment_manifest.tsv and write the table TSV.

WHY A FILE AND NOT A SHELL ONE-LINER. The tables are rebuilt after every batch of
plot jobs, and each rebuild has to read the same keys out of the same
metrics.txt, or two rows of one table stop being the same statistic. The
manifest holds only what a run cannot tell you about itself -- which table it
belongs in, what the row is called -- and every number comes from the run.

Usage:
    python tools/collect_exp_rows.py [manifest.tsv] > /tmp/exp_rows.tsv

Manifest columns (tab separated): section, block, label, config, status.
Output columns, one row per run:
    section block label config status Wij tau V_rest Eij rollout msg onestep
    cluster msg_form update_form conductance_form current_form gain_mean gain_sd
    E_over_vi rollout_own rollout_alt
with each recovery field written `clean|all|pct outliers`, the three numbers
metrics.txt keeps apart.
"""

import os
import re
import sys

LOG_ROOT = os.environ.get("GNN_LOG_ROOT",
                          "/groups/saalfeld/home/allierc/GraphData/log/fly")

# The two template-fit medians were renamed on 2026-09-15; runs plotted before
# that still write the old keys, so both are accepted and the new name is what
# the table gets. See tools/extraction_gate.py for the full rename map.
MEDIANS = {"msg_form": ("msg_form_r2_median", "tmpl_fit_r2_median"),
           "update_form": ("update_form_r2_median", "tmpl_update_r2_median"),
           # Both families' forms fitted to the SAME message, added 2026-09-15.
           # Runs plotted before that carry neither and print one fewer number.
           "conductance_form": ("conductance_form_r2_median",),
           "current_form": ("current_form_r2_median",),
           # What the family test reads: how much the driving-force column buys,
           # how far outside the data's voltage range the reversal it needs
           # sits, and how each reconstruction rolls out.
           "gain_mean": ("driving_force_r2_gain_mean",),
           "gain_sd": ("driving_force_r2_gain_sd",),
           "E_over_vi": ("conductance_form_E_over_vi",),
           "roll_own": ("template_rollout_r",),
           "roll_alt": ("template_alt_rollout_r",)}


def read_metrics(config):
    path = os.path.join(LOG_ROOT, config, "results", "metrics.txt")
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def num(m, *keys):
    """First key present that parses as a float, else ''."""
    for k in keys:
        v = m.get(k)
        if v in (None, ""):
            continue
        try:
            return f"{float(v):.6f}"
        except ValueError:
            continue
    return ""


def triple(m, q):
    """`clean|all|pct` for one quantity, as the tables print it."""
    return "|".join([num(m, f"{q}_R2"), num(m, f"{q}_R2_all"),
                     num(m, f"{q}_pct_outliers")])


_PEARSON = re.compile(r"Pearson r:\s*([-\d.eE+]+)")


def one_step_r(config):
    """The one-step Pearson r, from the run's own results_test.log.

    `-o test` writes that file next to the run; the plot pass mirrors the
    ROLLOUT r into metrics.txt but the one-step r only reached the terminal
    until 2026-09-15, so older runs are read here from the log rather than from
    metrics.txt. metrics.txt wins when it carries the key.
    """
    m = read_metrics(config)
    if m.get("one_step_r"):
        return num(m, "one_step_r")
    path = os.path.join(LOG_ROOT, config, "results_test.log")
    if not os.path.exists(path):
        return ""
    hit = _PEARSON.search(open(path, errors="replace").read())
    return hit.group(1) if hit else ""


def main():
    manifest = sys.argv[1] if len(sys.argv) > 1 else "docs/experiment_manifest.tsv"
    for line in open(manifest):
        f = line.rstrip("\n").split("\t")
        if len(f) < 4 or not f[0].strip():
            continue
        sec, block, label, config = f[:4]
        status = f[4] if len(f) > 4 else ""
        m = read_metrics(config)
        row = [sec, block, label, config, status,
               triple(m, "Wij"), triple(m, "tau"), triple(m, "V_rest"),
               triple(m, "Eij"), num(m, "rollout_r"), triple(m, "msg_i"),
               one_step_r(config), num(m, "clustering_accuracy"),
               num(m, *MEDIANS["msg_form"]), num(m, *MEDIANS["update_form"]),
               num(m, *MEDIANS["conductance_form"]), num(m, *MEDIANS["current_form"]),
               num(m, *MEDIANS["gain_mean"]), num(m, *MEDIANS["gain_sd"]),
               num(m, *MEDIANS["E_over_vi"]), num(m, *MEDIANS["roll_own"]),
               num(m, *MEDIANS["roll_alt"])]
        print("\t".join(row))


if __name__ == "__main__":
    main()
