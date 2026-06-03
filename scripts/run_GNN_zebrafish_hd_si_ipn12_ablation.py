"""Per-cell-type ablation runner for zebrafish_hd_si_ipn12_v1.

Companion to scripts/run_GNN_zebrafish_hd_si_ipn12_stats.py. Where the stats
runner re-seeds the SAME circuit 10x for variance, this runner holds the seed
FIXED and swaps the circuit, knocking out one cell type at a time, so the
delta in the swim-integration metrics is attributable to that type alone.

Mechanism (no data regeneration):
  * The connectome is rebuilt from `circuit.name` at train time
    (zebrafish_hd_task_rnn.py: get_circuit(...).as_loader_dict()), and the
    task dataset (stimulus / target / theta_hd) is circuit-agnostic — so we
    just point each config at a different *registered* circuit and reuse the
    shared dataset (zebrafish_hd_si_task_ipn12_v1).
  * For each of the 33 cell types, circuits.py registers a connectivity-
    lesion variant `zebrafish_HD_IPN12_839_v1_ablate_<token>`: N=839 and all
    ordering / subpops / decoder dims are identical to v1, but that type's
    rows AND columns in J_effective are zeroed. Since the RNN gates the
    trainable recurrent weight by `W_con_mask = (W_con != 0)`, the knockout
    holds through training.

We emit one config per ablated type (+ an `ablate_none` baseline = unmodified
v1 at the same seed, for a matched control), submit them all in parallel on
l4, stream graph_trainer metrics every 300 s, and finally write a summary
table ranking ablations by their swim-integration r_roll_1k (most damaging
first, with the delta vs the baseline).

Output layout (rooted at GNN_OUTPUT_ROOT or --output_root):
  config/zebrafish/zebrafish_hd_si_ipn12_v1_ablate_<token>.yaml
  log/zebrafish/zebrafish_hd_si_ipn12_v1_ablate_<token>/...
  log/zebrafish/zebrafish_ipn12_ablation_runner/   (cluster scratch + summary)
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from connectome_gnn.LLM.cluster import (
    submit_cluster_job,
    wait_for_cluster_jobs_with_metrics,
    _read_latest_training_metrics,
)
from connectome_gnn.generators.circuits import (
    _IPN12_ABLATION_TYPES as ABLATION_TYPES,
    ablation_type_token,
)


BASE = "zebrafish_hd_si_ipn12_v1"          # base config (config/zebrafish/<BASE>.yaml)
BASE_CIRCUIT = "zebrafish_HD_IPN12_839_v1"  # its registered circuit
BIOMODEL = "zebrafish"
BASELINE_TOKEN = "none"                     # ablate_none == unmodified v1 control


def _repo_root() -> str:
    return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def _output_root(explicit: str | None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    env = os.environ.get("GNN_OUTPUT_ROOT")
    if env:
        return os.path.abspath(env)
    raise SystemExit(
        "no output root: pass --output_root or set GNN_OUTPUT_ROOT "
        "(e.g. /groups/saalfeld/home/allierc/GraphData)"
    )


def _base_yaml_path() -> str:
    return os.path.join(_repo_root(), "config", BIOMODEL, f"{BASE}.yaml")


def _variants(with_baseline: bool) -> list[tuple[str, str, str]]:
    """Return the (token, circuit_name, config_tag) triples to run.

    config_tag is the dotted log/config stem `<BASE>_ablate_<token>`.
    The baseline keeps the unmodified v1 circuit at the same seed.
    """
    out = []
    if with_baseline:
        out.append((BASELINE_TOKEN, BASE_CIRCUIT,
                    f"{BASE}_ablate_{BASELINE_TOKEN}"))
    for t in ABLATION_TYPES:
        tok = ablation_type_token(t)
        out.append((tok, f"{BASE_CIRCUIT}_ablate_{tok}",
                    f"{BASE}_ablate_{tok}"))
    return out


def emit_ablation_yamls(output_root: str, with_baseline: bool) -> list[tuple]:
    """Write one config per variant; only `circuit.name` differs from the
    base v1 config (dataset + seed left untouched -> shared data, matched
    seed). Returns the list of (token, circuit_name, config_tag) triples."""
    src_path = _base_yaml_path()
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"base yaml not found: {src_path}")
    with open(src_path) as f:
        base_cfg = yaml.safe_load(f)

    out_dir = os.path.join(output_root, "config", BIOMODEL)
    os.makedirs(out_dir, exist_ok=True)

    variants = _variants(with_baseline)
    for _tok, circuit_name, tag in variants:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("circuit", {})["name"] = circuit_name
        # dataset / training.seed deliberately untouched: shared task data,
        # fixed seed so the only varying factor is the lesioned circuit.
        with open(os.path.join(out_dir, f"{tag}.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[emit] wrote {len(variants)} ablation yamls under {out_dir} "
          f"(circuit.name swapped per type; dataset/seed unchanged"
          f"{', incl. ablate_none baseline' if with_baseline else ''})")
    return variants


def submit_training(variants, output_root, *, node_name, conda_env, device,
                    hard_runtime_limit_min, erase):
    """One bsub training job per variant. Returns (job_ids, log_dirs, tags)
    keyed by slot id."""
    yaml_dir = os.path.join(output_root, "config", BIOMODEL)
    runner_dir = os.path.join(output_root, "log", BIOMODEL,
                              "zebrafish_ipn12_ablation_runner")
    os.makedirs(runner_dir, exist_ok=True)

    job_ids: dict[int, str] = {}
    log_dirs: dict[int, str] = {}
    tags: dict[int, str] = {}
    for slot, (_tok, _circuit, tag) in enumerate(variants):
        config_path = os.path.join(yaml_dir, f"{tag}.yaml")
        analysis_log = os.path.join(runner_dir, f"slot_{slot:02d}_analysis.log")
        jid = submit_cluster_job(
            slot=slot,
            config_path=config_path,
            analysis_log_path=analysis_log,
            config_file_field=f"{BIOMODEL}/{tag}",
            log_dir=runner_dir,
            erase=erase,
            node_name=node_name,
            conda_env=conda_env,
            n_cpus=2,
            device=device,
            output_root=output_root,
            hard_runtime_limit_min=hard_runtime_limit_min,
        )
        if jid is None:
            raise RuntimeError(f"failed to submit {tag} (slot {slot}) "
                               f"to {node_name}")
        job_ids[slot] = jid
        log_dirs[slot] = os.path.join(output_root, "log", BIOMODEL, tag)
        tags[slot] = tag
    return job_ids, log_dirs, tags


def emit_ablation_summary(output_root, variants) -> str:
    """Read each variant's final r_roll_1k from tmp_training/metrics.log and
    write a markdown table ranked most-damaging-first, with the delta vs the
    `ablate_none` baseline (if present)."""
    runner_dir = os.path.join(output_root, "log", BIOMODEL,
                              "zebrafish_ipn12_ablation_runner")
    rows = []  # (token, tag, r_roll_1k or None)
    for _tok, _circuit, tag in variants:
        log_dir = os.path.join(output_root, "log", BIOMODEL, tag)
        tm = _read_latest_training_metrics(log_dir)
        r = tm.get("r_roll_1k") if tm else None
        rows.append((_tok, tag, r))

    baseline = next((r for tok, _t, r in rows
                     if tok == BASELINE_TOKEN and r is not None), None)

    have = [(tok, tag, r) for tok, tag, r in rows if r is not None]
    have.sort(key=lambda x: x[2])  # ascending: most damaging first

    lines = [
        f"# {BASE} — per-cell-type ablation summary",
        "",
        f"- variants with metrics: {len(have)}/{len(rows)}",
        (f"- baseline (ablate_none) r_roll_1k: {baseline:.4f}"
         if baseline is not None else "- baseline (ablate_none): n/a"),
        "",
        "| rank | ablated type | r_roll_1k | Δ vs baseline |",
        "|------|--------------|-----------|---------------|",
    ]
    for i, (tok, _tag, r) in enumerate(have, 1):
        d = f"{r - baseline:+.4f}" if baseline is not None else "-"
        label = "— (baseline)" if tok == BASELINE_TOKEN else tok
        lines.append(f"| {i} | {label} | {r:.4f} | {d} |")
    for tok, _tag, r in rows:
        if r is None:
            lines.append(f"| - | {tok} | (no metrics) | - |")

    path = os.path.join(runner_dir, f"{BASE}_ablation_summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[summary] wrote {path}")
    return path


def run_test_plot(variants, output_root, *, python_exec=None):
    """Run ``GNN_Main.py -o test_plot`` locally for each (already-trained)
    variant. test/plot is a local step (the cluster job trains only), so this
    just loops. Each run writes ``<log_dir>/results_path_integration.log`` and
    ``results/{test_random_trials,test_deterministic_sweep,test_integration_gain}.png``
    --- the trial-based pi_acc + per-trial Pearson + the integration-gain scan.

    We pass the config PATH (not the dotted ``zebrafish/<tag>`` name) so the log
    dir resolves to ``log/zebrafish/<tag>`` instead of the doubled
    ``log/zebrafish/zebrafish/<tag>``."""
    import subprocess
    repo = _repo_root()
    py = python_exec or sys.executable
    env = os.environ.copy()
    env["GNN_OUTPUT_ROOT"] = output_root
    n = len(variants)
    for slot, (_tok, _circuit, tag) in enumerate(variants):
        cfg = os.path.join(output_root, "config", BIOMODEL, f"{tag}.yaml")
        if not os.path.isfile(cfg):
            cfg = os.path.join(repo, "config", BIOMODEL, f"{tag}.yaml")
        if not os.path.isfile(cfg):
            print(f"  [skip {slot + 1}/{n}] {tag}: no config yaml")
            continue
        print(f"[test_plot {slot + 1}/{n}] {tag}", flush=True)
        r = subprocess.run([py, "GNN_Main.py", "-o", "test_plot", cfg,
                            "--output_root", output_root], cwd=repo, env=env)
        if r.returncode != 0:
            print(f"  [warn] {tag} test exited {r.returncode}")


def collect_test_results(variants, output_root) -> str:
    """Parse each variant's results_path_integration.log into a comparison CSV
    + ranked table. Pulls the trial-based metrics (full-split pi_acc, mean
    per-trial Pearson) and the integration-gain linearity (mean gain and mean
    |gain-1| over the omega scan) --- all computed on the task trials / sweeps,
    not the single curriculum-horizon rollout logged during training."""
    import csv

    import numpy as np

    def _parse(path):
        """results_path_integration.log is a sectioned CSV: a full_test_pi_acc
        line, then `# Random test trials` (trial_idx,rmse_deg,pearson),
        `# Deterministic sweeps`, and `# Integration gain`
        (omega,slope,gain,fit_r2)."""
        pi = np.nan
        tr_rmse, tr_r, gains, r2 = [], [], [], []
        sec = None
        for ln in open(path):
            ln = ln.strip()
            if ln.startswith("full_test_pi_acc"):
                pi = float(ln.rsplit(":", 1)[1]); continue
            if ln.startswith("# Random test trials"):
                sec = "trials"; continue
            if ln.startswith("# Deterministic"):
                sec = "sweep"; continue
            if ln.startswith("# Integration gain"):
                sec = "gain"; continue
            if not ln or ln[0].isalpha():        # blank or CSV header row
                continue
            p = ln.split(",")
            try:
                if sec == "trials" and len(p) >= 3:
                    tr_rmse.append(float(p[1])); tr_r.append(float(p[2]))
                elif sec == "gain" and len(p) >= 4:
                    gains.append(float(p[2])); r2.append(float(p[3]))
            except ValueError:
                pass
        return pi, tr_rmse, tr_r, gains, r2

    rows = []
    for _tok, _c, tag in variants:
        log = os.path.join(output_root, "log", BIOMODEL, tag,
                           "results_path_integration.log")
        if not os.path.isfile(log):
            continue
        pi, tr_rmse, tr_r, gains, r2 = _parse(log)
        g = np.array(gains) if gains else np.array([np.nan])
        rows.append(dict(
            tok=_tok, pi=pi,
            trial_rmse=float(np.nanmean(tr_rmse)) if tr_rmse else np.nan,
            trial_r=float(np.nanmean(tr_r)) if tr_r else np.nan,
            mean_gain=float(np.nanmean(g)),
            gain_dev=float(np.nanmean(np.abs(g - 1.0))),
            fit_r2=float(np.nanmean(r2)) if r2 else np.nan))
    runner_dir = os.path.join(output_root, "log", BIOMODEL,
                              "zebrafish_ipn12_ablation_runner")
    os.makedirs(runner_dir, exist_ok=True)
    out_csv = os.path.join(runner_dir, "test_summary.csv")
    rows.sort(key=lambda d: (np.nan_to_num(d["trial_rmse"], nan=-1)), reverse=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token", "pi_acc", "trial_rmse_deg", "trial_pearson",
                    "mean_gain", "mean_abs_gain_minus_1", "gain_fit_r2"])
        for r in rows:
            w.writerow([r["tok"], f"{r['pi']:.4f}", f"{r['trial_rmse']:.3f}",
                        f"{r['trial_r']:.4f}", f"{r['mean_gain']:.3f}",
                        f"{r['gain_dev']:.3f}", f"{r['fit_r2']:.4f}"])
    print(f"\n[collect] {len(rows)} runs -> {out_csv}")
    print(f"{'token':<14}{'pi_acc':>9}{'trial_rmse':>11}{'trial_r':>9}"
          f"{'gain':>7}{'|g-1|':>7}")
    for r in rows:
        print(f"{r['tok']:<14}{r['pi']:>9.4f}{r['trial_rmse']:>11.3f}"
              f"{r['trial_r']:>9.4f}{r['mean_gain']:>7.3f}{r['gain_dev']:>7.3f}")
    return out_csv


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['train', 'test_plot'], default='train',
                   help="train (default): emit + submit ablation training. "
                        "test_plot: run GNN_Main -o test_plot locally on each "
                        "already-trained variant and collect a comparison table "
                        "(trial pi_acc, per-trial r, integration-gain linearity).")
    p.add_argument('--cluster', choices=['l4', 'a100', 'h100'], default='l4')
    p.add_argument('--no-baseline', dest='baseline', action='store_false',
                   help='skip the ablate_none (unmodified v1) control run')
    p.add_argument('--only', default=None,
                   help='comma-separated cell types to ablate (default: all 33). '
                        'e.g. --only IPN12_a,IPN12_b,pt-IPN1')
    p.add_argument('--skip-emit', action='store_true')
    p.add_argument('--skip-train', action='store_true',
                   help='emit yamls only, do not submit')
    p.add_argument('--no-erase', dest='erase', action='store_false')
    p.add_argument('--no-wait', dest='wait', action='store_false')
    p.add_argument('--output_root', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--hard-runtime-min', type=int, default=600)
    p.add_argument('--metrics-interval', type=int, default=300)
    p.add_argument('--poll-interval', type=int, default=60)
    p.add_argument('--conda-env', default='connectome-gnn')
    args = p.parse_args()

    out = _output_root(args.output_root)

    variants = _variants(args.baseline)
    if args.only:
        wanted = {ablation_type_token(t.strip()) for t in args.only.split(',')}
        variants = [v for v in variants
                    if v[0] in wanted or v[0] == BASELINE_TOKEN and args.baseline]

    print(f"\n=== zebrafish ipn12 per-cell-type ablation: {len(variants)} "
          f"variants ({args.mode}) (output_root={out}) ===")

    if args.mode == 'test_plot':
        run_test_plot(variants, out)
        collect_test_results(variants, out)
        return

    if not args.skip_emit:
        # Re-emit only the selected variants (emit writes all; filter when --only).
        if args.only:
            # emit just the selected subset
            src = _base_yaml_path()
            with open(src) as f:
                base_cfg = yaml.safe_load(f)
            out_dir = os.path.join(out, "config", BIOMODEL)
            os.makedirs(out_dir, exist_ok=True)
            for _tok, circuit_name, tag in variants:
                cfg = copy.deepcopy(base_cfg)
                cfg.setdefault("circuit", {})["name"] = circuit_name
                with open(os.path.join(out_dir, f"{tag}.yaml"), "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"[emit] wrote {len(variants)} ablation yamls (subset) "
                  f"under {out_dir}")
        else:
            emit_ablation_yamls(out, args.baseline)

    if args.skip_train:
        print("[train] skipped (--skip-train)")
        return

    job_ids, log_dirs, _tags = submit_training(
        variants, out,
        node_name=args.cluster,
        conda_env=args.conda_env,
        device=args.device,
        hard_runtime_limit_min=args.hard_runtime_min,
        erase=args.erase,
    )
    print(f"[submitted] {len(job_ids)} jobs to gpu_{args.cluster}")

    if args.wait:
        wait_for_cluster_jobs_with_metrics(
            job_ids,
            log_dirs=log_dirs,
            poll_interval=args.poll_interval,
            metrics_interval=args.metrics_interval,
        )
        emit_ablation_summary(out, variants)
    else:
        print(f"[submitted] {len(job_ids)} jobs; not waiting (--no-wait)")


if __name__ == '__main__':
    main()


# python scripts/run_GNN_zebrafish_hd_si_ipn12_ablation.py
# python scripts/run_GNN_zebrafish_hd_si_ipn12_ablation.py --only IPN12_a,IPN12_b --no-wait
# python scripts/run_GNN_zebrafish_hd_si_ipn12_ablation.py --skip-train   # emit yamls only
