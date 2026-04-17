"""Phase V — automatic KEEP/REVERT decision on a block's code change.

Triple-check against a pre-block baseline:
    (i)   mean improves by DELTA_MIN on the primary metric, AND
    (ii)  at least FRACTION_BETTER of seeds strictly better than pre-block
          median, AND
    (iii) no seed worse than pre-block min minus CATASTROPHE_MARGIN.

Pure function, no I/O. Unit-testable. Primary metric defaults to W R² ('W_R2')
because the measurement-noise objective is connectivity recovery; the
secondary metrics are still checked for the no-catastrophe clause on each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Dict, List, Literal, Sequence

Decision = Literal["KEEP", "REVERT"]

PRIMARY_METRIC = "W_R2"
DELTA_MIN = 0.005
FRACTION_BETTER = 0.75
CATASTROPHE_MARGIN = 0.02

# Metrics we also check for "no catastrophe", even if they aren't the primary
# optimisation target. Primary is always W_R2.
GUARDED_METRICS = ("W_R2", "tau_R2", "V_rest_R2", "clustering_accuracy")


@dataclass
class VerdictReport:
    decision: Decision
    reason: str
    primary_metric: str
    pre_mean: float
    post_mean: float
    delta: float
    fraction_better: float
    worst_post: float
    catastrophe_floor: float
    checks: Dict[str, bool] = field(default_factory=dict)

    def as_markdown(self) -> str:
        lines = [
            f"# Verdict: {self.decision}",
            "",
            f"**Reason**: {self.reason}",
            "",
            f"- primary metric: `{self.primary_metric}`",
            f"- pre-block mean: {self.pre_mean:.4f}",
            f"- post-block mean: {self.post_mean:.4f}  (Δ = {self.delta:+.4f})",
            f"- fraction of seeds strictly better than pre-median: {self.fraction_better:.2f}",
            f"- worst post-block seed: {self.worst_post:.4f}  (catastrophe floor = {self.catastrophe_floor:.4f})",
            "",
            "## Check breakdown",
        ]
        for k, v in self.checks.items():
            lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
        return "\n".join(lines) + "\n"


def decide(
    pre: Dict[str, Sequence[float]],
    post: Dict[str, Sequence[float]],
    primary_metric: str = PRIMARY_METRIC,
    delta_min: float = DELTA_MIN,
    fraction_better: float = FRACTION_BETTER,
    catastrophe_margin: float = CATASTROPHE_MARGIN,
) -> VerdictReport:
    """Decide KEEP or REVERT for a block, given per-seed metric lists.

    `pre` and `post` are dicts mapping metric name → list of seed-level values.
    """

    pre_vals = list(pre.get(primary_metric, []))
    post_vals = list(post.get(primary_metric, []))

    if len(post_vals) < 3:
        return VerdictReport(
            decision="REVERT",
            reason=(
                f"insufficient seeds in post-block: got {len(post_vals)}, "
                f"need ≥ 3 for a causal verdict"
            ),
            primary_metric=primary_metric,
            pre_mean=float("nan"),
            post_mean=float("nan"),
            delta=float("nan"),
            fraction_better=0.0,
            worst_post=float("nan"),
            catastrophe_floor=float("nan"),
        )
    if len(pre_vals) == 0:
        return VerdictReport(
            decision="KEEP",
            reason="no pre-block baseline — first code block, keeping by default",
            primary_metric=primary_metric,
            pre_mean=float("nan"),
            post_mean=mean(post_vals),
            delta=float("nan"),
            fraction_better=1.0,
            worst_post=min(post_vals),
            catastrophe_floor=float("nan"),
        )

    pre_mean = mean(pre_vals)
    pre_median = median(pre_vals)
    pre_min = min(pre_vals)
    post_mean = mean(post_vals)
    delta = post_mean - pre_mean
    frac_better = sum(1 for v in post_vals if v > pre_median) / len(post_vals)
    worst_post = min(post_vals)
    catastrophe_floor = pre_min - catastrophe_margin

    # Per-metric no-catastrophe check
    catastrophes = []
    for m in GUARDED_METRICS:
        pv = list(pre.get(m, []))
        qv = list(post.get(m, []))
        if not pv or not qv:
            continue
        floor = min(pv) - catastrophe_margin
        worst = min(qv)
        if worst < floor:
            catastrophes.append(f"{m}: worst={worst:.4f} < floor={floor:.4f}")

    check_delta = delta >= delta_min
    check_frac = frac_better >= fraction_better
    check_cata = len(catastrophes) == 0

    checks = {
        f"Δ {primary_metric} ≥ {delta_min}": check_delta,
        f"≥ {fraction_better:.0%} seeds > pre-median": check_frac,
        "no catastrophe on any guarded metric": check_cata,
    }

    if check_delta and check_frac and check_cata:
        reason = (
            f"Δ {primary_metric} = {delta:+.4f} (≥ {delta_min}), "
            f"{frac_better:.0%} seeds better than pre-median, no catastrophe"
        )
        decision: Decision = "KEEP"
    else:
        parts = []
        if not check_delta:
            parts.append(f"Δ {primary_metric} = {delta:+.4f} (need ≥ {delta_min})")
        if not check_frac:
            parts.append(
                f"only {frac_better:.0%} seeds better than pre-median "
                f"(need ≥ {fraction_better:.0%})"
            )
        if not check_cata:
            parts.append("catastrophe(s): " + "; ".join(catastrophes))
        reason = "; ".join(parts)
        decision = "REVERT"

    return VerdictReport(
        decision=decision,
        reason=reason,
        primary_metric=primary_metric,
        pre_mean=pre_mean,
        post_mean=post_mean,
        delta=delta,
        fraction_better=frac_better,
        worst_post=worst_post,
        catastrophe_floor=catastrophe_floor,
        checks=checks,
    )


def collect_metrics_from_run_dirs(run_dirs: Sequence[str]) -> Dict[str, List[float]]:
    """Parse metrics.txt files produced by data_plot into a per-metric seed list.

    Reads `<run_dir>/results/metrics.txt` for each run and extracts
    W_corrected_R2, tau_R2, V_rest_R2, clustering_accuracy, and rollout/one-step
    pearson (if present). Primary returned key is W_R2 (alias of W_corrected_R2).
    """
    import os

    wanted = {
        "W_corrected_R2": "W_R2",
        "tau_R2": "tau_R2",
        "V_rest_R2": "V_rest_R2",
        "clustering_accuracy": "clustering_accuracy",
    }
    out: Dict[str, List[float]] = {v: [] for v in wanted.values()}

    for run_dir in run_dirs:
        path = os.path.join(run_dir, "results", "metrics.txt")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip()
                if k not in wanted:
                    continue
                try:
                    out[wanted[k]].append(float(v.strip()))
                except ValueError:
                    pass
    return out
