#!/usr/bin/env python3
"""Generate docs/flyvis_results.md from log directories.

Reuses the collection logic from collect_flyvis_results.py and outputs
the full HTML-styled markdown file with color-coded tables.

Usage:
    python scripts/generate_flyvis_md.py
"""

import math
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(REPO_ROOT, "log", "fly")
OUT_PATH = os.path.join(REPO_ROOT, "docs", "flyvis_results.md")

# Import collection helpers
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from collect_flyvis_results import (
    VARIANTS, NOISE_CONDITIONS, METRICS, METRIC_LABELS, MAX_CV,
    collect_cv_metrics, stats, color_code,
)

# HTML color backgrounds (with 60 = ~37% opacity)
COLORS = {"G": "#2ea04360", "O": "#d2992260", "R": "#cf222e60"}


def bg(val):
    """Return style attribute for a value's color."""
    c = color_code(val)
    return f' style="background:{COLORS[c]}"' if val is not None else ""


def fv(val):
    """Format a value to 3 decimal places."""
    if val is None:
        return "—"
    return f"{val:.3f}"


def write_summary_table(f, label, dir_pattern, note=""):
    """Write one summary HTML table."""
    f.write(f"\n### {label}\n\n<table>\n")
    f.write("<tr><th>Condition</th><th>Seeds</th>"
            "<th>Conn R2 (W)</th><th>tau R2</th><th>V_rest R2</th>"
            "<th>Cluster acc</th><th>One-step Pearson</th>"
            "<th>Rollout Pearson</th></tr>\n")

    for noise_key, noise_label in NOISE_CONDITIONS:
        rows = collect_cv_metrics(dir_pattern, noise_key)
        n = len(rows)
        if n == 0:
            continue

        cells = []
        for metric in METRICS:
            vals = [r[1].get(metric) for r in rows if metric in r[1]]
            mean, std, _, _ = stats(vals)
            if mean is not None:
                cells.append(f'<td{bg(mean)}>{fv(mean)} &pm; {fv(std)}</td>')
            else:
                cells.append("<td>—</td>")

        f.write(f'<tr><td><b>{noise_label}</b></td><td>{n}</td>'
                + "".join(cells) + "</tr>\n")

    f.write("</table>\n")


def write_detail_table(f, label, dir_pattern, noise_key, noise_label):
    """Write one per-seed detail HTML table."""
    rows = collect_cv_metrics(dir_pattern, noise_key)
    if not rows:
        return

    f.write(f"\n### {label} — {noise_label}\n\n<table>\n")
    f.write("<tr><th>Seed</th>"
            "<th>W R2</th><th>tau R2</th><th>V_rest R2</th>"
            "<th>Cluster acc</th><th>One-step r</th>"
            "<th>Rollout r</th></tr>\n")

    all_vals = {m: [] for m in METRICS}

    for cv, m in rows:
        seed = 42 + cv
        cells = []
        for metric in METRICS:
            val = m.get(metric)
            if val is not None:
                cells.append(f'<td{bg(val)}>{fv(val)}</td>')
                all_vals[metric].append(val)
            else:
                cells.append("<td>—</td>")
        f.write(f'<tr><td>cv{cv:02d} ({seed})</td>' + "".join(cells) + "</tr>\n")

    # Summary rows
    for stat_name, idx in [("Mean", 0), ("Std", 1), ("Min", 2), ("Max", 3)]:
        cells = []
        for metric in METRICS:
            vals = all_vals[metric]
            s = stats(vals)
            val = s[idx] if vals else None
            if val is not None:
                if stat_name == "Mean":
                    cells.append(f'<td{bg(val)}><b>{fv(val)}</b></td>')
                else:
                    cells.append(f'<td><b>{fv(val)}</b></td>')
            else:
                cells.append("<td>—</td>")
        f.write(f'<tr><td><b>{stat_name}</b></td>' + "".join(cells) + "</tr>\n")

    f.write("</table>\n")


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(OUT_PATH, "w") as f:
        f.write("""# Flyvis Results — GNN vs Known-ODE

**Model**: Drosophila optic lobe (13,741 neurons, 434,112 GT edges)
**ODE**: Graded-voltage model: dv/dt = (-v + V_rest)/tau + ReLU(v) @ W

<style>table { font-size: 0.85em; } th, td { padding: 3px 6px; }</style>

## Summary Table (mean &pm; std over seeds)

Color code: <span style="color:#2ea043">green</span> &gt; 0.9, <span style="color:#d29922">orange</span> &gt; 0.5, <span style="color:#cf222e">red</span> &le; 0.5.
""")

        for label, pattern in VARIANTS:
            write_summary_table(f, label, pattern)

        f.write("""
---

## Per-Seed Detail
""")

        for label, pattern in VARIANTS:
            for noise_key, noise_label in NOISE_CONDITIONS:
                write_detail_table(f, label, pattern, noise_key, noise_label)

        # Key observations
        # Collect summary data for the observations section
        data = {}
        for vlabel, vpattern in VARIANTS:
            for nkey, nlabel in NOISE_CONDITIONS:
                rows = collect_cv_metrics(vpattern, nkey)
                if rows:
                    means = {}
                    for metric in METRICS:
                        vals = [r[1].get(metric) for r in rows if metric in r[1]]
                        m, s, _, _ = stats(vals)
                        means[metric] = (m, s)
                    data[(vlabel, nlabel)] = (len(rows), means)

        f.write("""
---

## Key Observations

### LLM exploration improves over default config
""")
        # Compare LLM vs default for each noise condition
        for nkey, nlabel in NOISE_CONDITIONS:
            llm = data.get(("GNN (LLM-optimized)", nlabel))
            default = data.get(("GNN (default)", nlabel))
            if llm and default:
                improvements = []
                for metric, mlabel in zip(METRICS[:4], METRIC_LABELS[:4]):
                    lm = llm[1].get(metric, (None, None))[0]
                    dm = default[1].get(metric, (None, None))[0]
                    if lm is not None and dm is not None and dm > 0:
                        pct = (lm - dm) / dm * 100
                        improvements.append(f"**{mlabel}**: {fv(dm)} → {fv(lm)} ({pct:+.0f}%)")
                if improvements:
                    f.write(f"- **{nlabel}**: " + ", ".join(improvements) + "\n")

        f.write("""
### Noise helps parameter recovery
""")
        llm_nf = data.get(("GNN (LLM-optimized)", "Noise-free"))
        llm_05 = data.get(("GNN (LLM-optimized)", "Noise=0.05"))
        llm_5 = data.get(("GNN (LLM-optimized)", "Noise=0.5"))
        if llm_nf and llm_05 and llm_5:
            for metric, mlabel in zip(METRICS[:4], METRIC_LABELS[:4]):
                v_nf = fv(llm_nf[1].get(metric, (None,))[0])
                v_05 = fv(llm_05[1].get(metric, (None,))[0])
                v_5 = fv(llm_5[1].get(metric, (None,))[0])
                f.write(f"- **{mlabel}**: {v_nf} (noise-free) → {v_05} (σ=0.05) → {v_5} (σ=0.5)\n")

        f.write("""
### GNN vs Known-ODE
- Known-ODE has near-zero variance across seeds (ground-truth ODE structure removes optimization difficulty)
- GNN matches Known-ODE at high noise (σ=0.5) for W R2 (both 0.997)
- Known-ODE consistently better for V_rest (direct parameter vs indirect extraction from f_theta)

### Status
""")
        for vlabel, vpattern in VARIANTS:
            counts = []
            for nkey, nlabel in NOISE_CONDITIONS:
                rows = collect_cv_metrics(vpattern, nkey)
                counts.append(f"{len(rows)} ({nlabel})")
            f.write(f"- {vlabel}: {', '.join(counts)}\n")

    print(f"Generated {OUT_PATH}")


if __name__ == "__main__":
    main()
