#!/usr/bin/env python
"""Old readout against new, per arm, written into each experiment's markdown.

WHY THIS EXISTS. On 2026-09-28 every landed run of experiments 0-5 was
re-analysed (`exp analyse --redo pre_readout_fix`) after the template readout's
second-pass fix and its switch to one uniform draw of 1,024 frames (commit
9ef188e6, experiment 8). Each run kept its earlier results/metrics.txt as
superseded/pre_readout_fix/metrics.txt. This reads both and writes, between
READOUT_FIX markers in the experiment's markdown, the fold mean of each quantity
the readout moves, old -> new, so the change is derived rather than typed.

    python tools/readout_fix_compare.py [N ...]      # default: experiments 0-5
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp as E  # noqa: E402

TAG = "pre_readout_fix"
KEYS = [("Wij_R2", "R2_W", 3), ("V_rest_R2", "R2_Vrest", 3),
        ("fitroll_current_r", "fit roll r current form", 3),
        ("fitroll_conductance_r", "fit roll r conductance form", 3),
        ("clustering_accuracy", "cluster", 3), ("tmpl_pct_fitted", "edges fitted %", 1)]
_B, _E = "<!-- READOUT_FIX:BEGIN -->", "<!-- READOUT_FIX:END -->"


def _read(path):
    """metrics.txt as numbers, with the fit-roll columns named by family."""
    raw = {}
    for line in open(path):
        m = re.match(r"^(\w+): (\S+)", line)
        if m:
            raw[m.group(1)] = m.group(2)
    out = {}
    for k, v in E._fit_roll_by_family(raw).items():
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def block(fm):
    groups = {}
    for arm, pt, run in E.runs(fm):
        old = os.path.join(E.LOG_ROOT, run, "superseded", TAG, "metrics.txt")
        new = os.path.join(E.LOG_ROOT, run, "results", "metrics.txt")
        if not (os.path.isfile(old) and os.path.isfile(new)):
            continue
        cell = tuple(E._axis_label(fm, k, v) for k, v in pt.items() if k != "fold")
        groups.setdefault((arm["id"], cell), []).append((_read(old), _read(new)))
    if not groups:
        return None
    axes = [k for k in fm["axes"] if k != "fold"]
    # The columns the status table reads off each arm's differs_by (the lasso),
    # so two arms that share a label are still told apart.
    extra = E._arm_columns(fm)
    L = [_B, "", "## Re-analysed on the fixed readout (2026-09-28)", "",
         "Every landed run was re-analysed after the second-pass fix and the switch to "
         "one uniform draw of 1,024 frames (commit 9ef188e6, experiment 8); the earlier "
         "`metrics.txt` is kept as `superseded/pre_readout_fix/`. Fold means, "
         "old -> new, from `tools/readout_fix_compare.py`. The status table below is "
         "the new readout.", "",
         "| " + " | ".join(["arm"] + [h for h, _ in extra] + axes + ["n"]
                           + [h for _, h, _ in KEYS]) + " |",
         "|" + "---|" * (2 + len(extra) + len(axes) + len(KEYS))]
    for (arm_id, cell), prs in groups.items():
        cells = []
        for key, _h, nd in KEYS:
            both = [(o[key], n[key]) for o, n in prs if key in o and key in n]
            if not both:
                cells.append("")
                continue
            mo = sum(b[0] for b in both) / len(both)
            mn = sum(b[1] for b in both) / len(both)
            cells.append(f"{mo:.{nd}f} -> {mn:.{nd}f}")
        L.append("| " + " | ".join([E._arm_label(fm, arm_id)]
                                   + [E._arm_value(fm["arms"], arm_id, k) for _, k in extra]
                                   + [*cell, str(len(prs)), *cells]) + " |")
    L += ["", _E]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("numbers", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    a = ap.parse_args()
    for n in a.numbers:
        path = E.exp_path(n)
        fm, _ = E.load(path)
        b = block(fm)
        if b is None:
            print(f"experiment {n}: nothing re-analysed")
            continue
        text = open(path).read()
        if _B in text:
            text = re.sub(re.escape(_B) + r".*?" + re.escape(_E), b, text, flags=re.S)
        else:
            text = text.replace(E._BEGIN, b + "\n\n" + E._BEGIN, 1)
        open(path, "w").write(text)
        print(f"experiment {n}: {len(b.splitlines()) - 10} rows")


if __name__ == "__main__":
    main()
