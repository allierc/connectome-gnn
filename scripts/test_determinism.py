"""Is flyvis GNN training bitwise reproducible? Measure it, then fix it.

The question
-----------
Two runs of the paper's sigma=0 config -- same commit, same YAML, same seeds,
same read-only datasets -- disagreed on R^2_W at iteration 32,001 by up to
0.07. If identical inputs do not give identical outputs, no re-run can confirm
or refute a published number, and every "the result moved" observation is
uninterpretable.

The cause
---------
NeuralGNN aggregates messages with

    msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)

``scatter_add_`` on CUDA accumulates through atomics. 434,112 edges land on
13,741 nodes, so each node sums ~32 floats in whatever order the scheduler
produces, and float addition is not associative. Measured directly at flyvis
scale, 8 identical calls give 8 distinct answers, max |delta| 5.7e-06. That is
negligible per step and decisive over 1.6M steps.

``torch.use_deterministic_algorithms(True)`` substitutes a sort-based scatter
with a fixed reduction order: 8/8 identical, max |delta| 0. It costs ~26x on
that op (13.7 -> 361 us) which is a sliver of a ~10 ms iteration.

What this script does
---------------------
Runs the real trainer N times per arm on one GPU and compares the runs within
each arm, byte for byte:

    arm 'off'  training.deterministic = False   -- expect runs to diverge
    arm 'on'   training.deterministic = True    -- expect runs to be identical

Comparison is on tmp_training/metrics.log (every R^2 checkpoint of the run) and
on the SHA-256 of every saved checkpoint tensor. Both must match for a pair to
count as reproducible; metrics.log alone would miss a divergence that has not
yet reached the reported digits.

Short runs by construction: data_augmentation_loop is overridden to make
Niter = n_frames * DAL // batch_size * 0.2 land near --n-iter, since the R^2
checkpoint cadence is derived from the UNCAPPED Niter (graph_trainer ~line 610)
and so max_iterations_per_epoch would silently suppress every row.

Everything is written under a dedicated ``dettest`` config/log name. The
published and repro artifacts are never opened for writing.

    # default: 2 runs per arm, both arms, GPU 0
    python scripts/test_determinism.py

    # just prove the problem exists, without the fix
    python scripts/test_determinism.py --arms off --runs 3

    # cost of determinism at this scale
    python scripts/test_determinism.py --arms on --runs 1 --time
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_CFG_DEFAULT = 'flyvis_noise_free_blank50_unified_cv00'   # published, READ ONLY

_G, _O, _R, _DIM = '\033[92m', '\033[38;5;208m', '\033[0m', '\033[2m'


def emit_config(output_root, arm, n_iter, batch_size, no_compile, source_cfg):
    """Write the dettest YAML for one arm, derived from source_cfg."""
    test_stem = f'{source_cfg}_dettest'
    src = os.path.join(output_root, 'config', 'fly', f'{source_cfg}.yaml')
    assert os.path.isfile(src), f'source config missing: {src}'
    with open(src) as f:
        cfg = yaml.safe_load(f)

    name = f'{test_stem}_{arm}'
    assert name != source_cfg
    # Niter = n_frames * DAL // batch_size * 0.2  (graph_trainer ~line 605).
    # Solve for the DAL that lands nearest the requested iteration count.
    n_frames = cfg['simulation']['n_frames']
    dal = max(1, round(n_iter * batch_size / (0.2 * n_frames)))
    cfg['training']['data_augmentation_loop'] = dal
    cfg['training']['batch_size'] = batch_size
    cfg['training']['deterministic'] = (arm == 'on')
    if no_compile:
        cfg['training']['torch_compile'] = False
    cfg['config_file'] = f'fly/{name}'
    cfg['description'] = (f'Determinism probe ({arm}). Derived from '
                          f'{source_cfg}.yaml, DAL={dal} for a short run.')
    dst = os.path.join(output_root, 'config', 'fly', f'{name}.yaml')
    with open(dst, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    niter = int(n_frames * dal // batch_size * 0.2)
    return dst, name, niter


def run_once(cfg_path, name, output_root, gpu):
    """One training run. Returns (seconds, metrics_log_text, checkpoint_hash)."""
    log_dir = os.path.join(output_root, 'log', 'fly', name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, 'train_subprocess.py', '--config', cfg_path,
         '--device', 'cuda', '--config_file', f'fly/{name}',
         '--output_root', output_root, '--erase'],
        cwd=_REPO_ROOT, env=env, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(f'{_O}  run FAILED (rc={r.returncode}){_R}')
        print('\n'.join((r.stderr or r.stdout).splitlines()[-25:]))
        return dt, None, None

    metrics_path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    metrics = open(metrics_path).read() if os.path.isfile(metrics_path) else None

    # Hash every saved tensor, not just the metrics: two runs can agree to 6
    # decimals in the log while their weights have already parted.
    import glob
    import torch
    h = hashlib.sha256()
    ckpts = sorted(glob.glob(os.path.join(log_dir, 'models', '*.pt')))
    for c in ckpts:
        sd = torch.load(c, map_location='cpu', weights_only=False)
        sd = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd
        for k in sorted(sd):
            v = sd[k]
            if hasattr(v, 'detach'):
                h.update(k.encode())
                h.update(v.detach().cpu().numpy().tobytes())
    return dt, metrics, (h.hexdigest() if ckpts else None)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arms', nargs='+', choices=['off', 'on'],
                   default=['off', 'on'],
                   help="'off' = training as it is today, 'on' = with "
                        'training.deterministic. Default: both.')
    p.add_argument('--runs', type=int, default=2,
                   help='Repeats per arm (>=2 to compare). Default 2.')
    p.add_argument('--n-iter', dest='n_iter', type=int, default=6400,
                   help='Approximate training iterations per run; DAL is '
                        'solved to hit it. Default 6400 (~2 min).')
    p.add_argument('--source-config', dest='source_cfg',
                   default=SOURCE_CFG_DEFAULT,
                   help='config/fly/<name>.yaml to derive the dettest arms '
                        f'from (read-only). Default: {SOURCE_CFG_DEFAULT}.')
    p.add_argument('--batch-size', dest='batch_size', type=int, default=4,
                   help='As published. Default 4.')
    p.add_argument('--gpu', type=int, default=0,
                   help='Local GPU index. Both runs of a pair MUST share a '
                        'GPU — different silicon is a different question.')
    p.add_argument('--no-compile', dest='no_compile', action='store_true',
                   help='Disable torch.compile (faster startup; tests the '
                        'eager path only).')
    p.add_argument('--output_root', default=None)
    p.add_argument('--time', action='store_true',
                   help='Report wall-clock per run.')
    args = p.parse_args()

    output_root = (args.output_root or os.environ.get('GNN_OUTPUT_ROOT')
                   or yaml.safe_load(
                       open(os.path.join(_REPO_ROOT, 'data_paths.json'))
                   ).get('cluster_data_dir'))
    assert output_root and os.path.isdir(output_root), output_root
    assert args.runs >= 1

    print('=' * 74)
    print('Determinism probe — same config, same seed, same GPU, N runs')
    print(f'  source config : {args.source_cfg} (read-only)')
    print(f'  arms          : {args.arms}   runs/arm: {args.runs}')
    print(f'  target iters  : ~{args.n_iter:,}   batch {args.batch_size}   '
          f'GPU {args.gpu}')
    print('=' * 74)

    results = {}
    for arm in args.arms:
        cfg_path, name, niter = emit_config(
            output_root, arm, args.n_iter, args.batch_size, args.no_compile,
            args.source_cfg)
        print(f'\n--- arm {arm}  (deterministic={arm == "on"}, '
              f'Niter={niter:,})\n    {cfg_path}')
        runs = []
        for k in range(args.runs):
            dt, metrics, chash = run_once(cfg_path, name, output_root, args.gpu)
            tag = f'{dt:6.1f}s' if args.time else ''
            n_rows = (metrics.count('\n') - 1) if metrics else 0
            print(f'    run {k}: {tag}  metrics rows={n_rows}  '
                  f'ckpt sha={(chash or "none")[:12]}')
            runs.append((metrics, chash))
        results[arm] = runs

    print(f'\n{"=" * 74}\nverdict\n{"=" * 74}')
    for arm, runs in results.items():
        if len(runs) < 2:
            print(f'  arm {arm}: only 1 run, nothing to compare')
            continue
        ref_m, ref_h = runs[0]
        # A crashed run yields (None, None). Two identical crashes would
        # otherwise compare equal and be reported as reproducible, which is the
        # most dangerous possible false positive here.
        if any(m is None for m, _ in runs):
            n_bad = sum(m is None for m, _ in runs)
            print(f'  arm {arm:<4} {_O}{n_bad}/{len(runs)} run(s) produced no '
                  f'metrics — training failed, verdict withheld{_R}')
            continue
        same_m = all(m == ref_m for m, _ in runs[1:])
        same_h = all(h == ref_h for _, h in runs[1:])
        ok = same_m and same_h
        col = _G if (ok == (arm == 'on')) else _O
        print(f'  arm {arm:<4} metrics identical: {str(same_m):<5}   '
              f'checkpoints identical: {str(same_h):<5}   '
              f'{col}{"REPRODUCIBLE" if ok else "NOT reproducible"}{_R}')
        if not ok and ref_m:
            # Show the first checkpoint where the runs part company.
            rows = [m.strip().splitlines() for m, _ in runs if m]
            for line_i in range(1, min(len(r) for r in rows)):
                vals = [r[line_i] for r in rows]
                if len(set(vals)) > 1:
                    print(f'{_DIM}    first divergence, metrics.log line '
                          f'{line_i}:{_R}')
                    for j, v in enumerate(vals):
                        print(f'      run {j}: {v}')
                    break


if __name__ == '__main__':
    main()
