#!/usr/bin/env python
"""Regression gate for the circuit-parameter extraction refactor.

WHAT IT GUARDS. Learned W, tau, V_rest, E_ij and msg_i are today extracted in
five different ways depending on the model and the data, and the R2 that scores
them is computed in five different places -- including inside `plot_training_gnn`,
a PLOTTING function whose return value is the trainer's headline
`connectivity_r2`. Unifying that touches every number the project publishes, so
the refactor is only safe if the numbers do not move.

HOW. Run `GNN_Main.py -o test_plot` over a fixed set of checkpoints before the
refactor and after it, and require every key in each run's
`results/metrics.txt` to agree to two decimals. Going through test_plot rather
than calling the extractor directly is deliberate: the entanglement being
removed lives in GNN_PlotFigure, so a unit test on the extractor alone would
not see it.

THE RUNS cover one estimator path each, which is the property that matters --
a gate of six current-data GNN runs would exercise one code path six times.

Usage:
    python tools/extraction_gate.py submit                # launch test_plot for every run
    python tools/extraction_gate.py collect before        # harvest -> tools/extraction_gate/before.json
    python tools/extraction_gate.py compare before after  # 2-decimal diff
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from connectome_gnn.LLM.cluster import _bsub_over_ssh  # noqa: E402

DATA_ROOT = '/groups/saalfeld/home/allierc/GraphData'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extraction_gate')

# (config name, what estimator path it exercises). One row per path, plus a
# second seed where a paired fold exists, so a change that is really seed noise
# is distinguishable from a change in the code.
RUNS = [
    ('flyvis_current_noise_005_current_cv01',                  'g_phi_corrected'),
    ('flyvis_current_noise_005_current_cv02',                  'g_phi_corrected (2nd seed)'),
    ('flyvis_current_noise_free_current_cv01',                 'g_phi_corrected, sigma=0'),
    ('flyvis_current_noise_free_current_cv02',                 'g_phi_corrected, sigma=0 (2nd seed)'),
    ('flyvis_current_noise_005_current_rc_uniform_cv00',       'g_phi_corrected, rollout curriculum'),
    ('flyvis_current_noise_005_current_rc_uniform_cv01',       'g_phi_corrected, rollout curriculum (2nd seed)'),
    ('flyvis_conductance_noise_005_conductance_knownode_cv00', 'direct (W**2), + Eij and msg_i'),
    ('flyvis_conductance_noise_005_conductance_lasso_0_cv00',  'edge_line_fit, gated out (fit_r2_median 0.13)'),
]


def log_dir(cfg: str) -> str:
    return os.path.join(DATA_ROOT, 'log', 'fly', cfg)


def submit(queue: str = 'l4') -> None:
    """One test_plot job per run, through the same submitter the LLM loop uses."""
    for cfg, path in RUNS:
        d = log_dir(cfg)
        if not os.path.isdir(os.path.join(d, 'models')):
            print(f"\033[91mSKIP {cfg}: no models/ directory\033[0m")
            continue
        jid, queue_label, res = _bsub_over_ssh(
            f"python GNN_Main.py -o test_plot {cfg}",
            conda_env='connectome-gnn', node_name=queue, n_cpus=8, device='cuda',
            hard_runtime_limit_min=240,
            stdout_path=f"{d}/bsub_gate_%J.out",
            stderr_path=f"{d}/bsub_gate_%J.err",
            job_name=f"gate_{cfg}")
        if jid:
            print(f"\033[92m  {jid}  {queue_label}  {cfg}\033[0m   [{path}]")
        else:
            print(f"\033[91m  FAILED   {cfg}\033[0m\n    {res.stdout.strip()}\n    {res.stderr.strip()}")


def _read_metrics(path: str) -> dict:
    """metrics.txt is `key: value` per line. Values that are not numbers are kept
    as strings so a provenance field like `Wij_estimator` survives the round trip."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            if ':' not in line:
                continue
            k, _, v = line.partition(':')
            v = v.strip()
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v
    return out


def collect(label: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    snapshot = {}
    for cfg, path in RUNS:
        m = _read_metrics(os.path.join(log_dir(cfg), 'results', 'metrics.txt'))
        snapshot[cfg] = {'estimator_path': path, 'metrics': m}
        status = f"{len(m)} keys" if m else "\033[91mNO metrics.txt\033[0m"
        print(f"  {cfg:<56s} {status}")
    dst = os.path.join(OUT_DIR, f"{label}.json")
    with open(dst, 'w') as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    print(f"\nwrote {dst}")


def compare(a_label: str, b_label: str, decimals: int = 2) -> int:
    """Two-decimal comparison. Returns a shell exit code: 0 when nothing moved."""
    with open(os.path.join(OUT_DIR, f"{a_label}.json")) as f:
        a = json.load(f)
    with open(os.path.join(OUT_DIR, f"{b_label}.json")) as f:
        b = json.load(f)

    n_moved = n_added = n_removed = 0
    for cfg, _ in RUNS:
        ma = a.get(cfg, {}).get('metrics', {})
        mb = b.get(cfg, {}).get('metrics', {})
        moved, added, removed = [], sorted(set(mb) - set(ma)), sorted(set(ma) - set(mb))
        for k in sorted(set(ma) & set(mb)):
            va, vb = ma[k], mb[k]
            if isinstance(va, float) and isinstance(vb, float):
                if round(va, decimals) != round(vb, decimals):
                    moved.append((k, va, vb))
            elif va != vb:
                moved.append((k, va, vb))
        n_moved += len(moved); n_added += len(added); n_removed += len(removed)
        if not (moved or added or removed):
            print(f"\033[92m  OK      {cfg}  ({len(ma)} keys identical to {decimals} dp)\033[0m")
            continue
        print(f"\033[93m  CHANGED {cfg}\033[0m")
        for k, va, vb in moved:
            print(f"      {k:<34s} {va} -> {vb}")
        for k in added:
            print(f"      \033[92m+ {k:<32s} {mb[k]}\033[0m")
        for k in removed:
            print(f"      \033[91m- {k:<32s} {ma[k]}\033[0m")

    print(f"\n{n_moved} moved, {n_added} added, {n_removed} removed")
    if n_moved:
        print("A moved key is a REGRESSION unless it is one you can name the bug for.")
    return 1 if n_moved else 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd == 'submit':
        submit(*(sys.argv[2:3] or []))
    elif cmd == 'collect':
        collect(sys.argv[2] if len(sys.argv) > 2 else 'before')
    elif cmd == 'compare':
        sys.exit(compare(sys.argv[2], sys.argv[3],
                         int(sys.argv[4]) if len(sys.argv) > 4 else 2))
    else:
        print(__doc__)
        sys.exit(2)
