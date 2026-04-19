"""
Emit the 8 TeX rows of tab:cv_per_condition from per-config cv_summary.txt.

Each row is read from the latest block of
    <DATA_ROOT>/log/fly/<config>/results/cv_summary.txt

Columns in the table:
    condition | noise sigma | noise gamma | edges
              | one-step r | rollout r
              | W R^2 | tau R^2 | V_rest R^2 | cluster acc.

Prediction columns (one-step r / rollout r) come from yt_one_step_r /
yt_rollout_r (DAVIS-trained model tested on held-out YouTube-VOS folds).
Parameter recovery columns come from W_corrected_R2 / tau_R2 / V_rest_R2 /
clustering_accuracy in the same block.

Output:
    <DATA_ROOT>/log/fly/cv_per_condition_rows.tex
    (also printed to stdout)

Usage:
    python scripts/emit_conditions_table_rows.py \\
        --output_root /groups/saalfeld/home/allierc/GraphData
"""

import argparse
import os
import re
import sys

import numpy as np


# Resolve repo root from this script's location (robust local + cluster).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (os.path.join(REPO_ROOT, 'src'), REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


GOOD_THRESHOLD = 0.9


# Condition -> (TeX label, config name, noise_sigma, noise_gamma, edges)
CONDITIONS = [
    ('noise-free',            'flyvis_noise_free_winner',                 '0',    '0',   '434\\,112'),
    ('low intrinsic noise',   'flyvis_noise_005_winner',                  '0.05', '0',   '434\\,112'),
    ('high intrinsic noise',  'flyvis_noise_05_winner',                   '0.5',  '0',   '434\\,112'),
    ('low meas. noise',       'flyvis_noise_005_010_winner',              '0.05', '0.1', '434\\,112'),
    ('$+400\\%$ null edges',  'flyvis_noise_005_null_edges_pc_400',       '0.05', '0',   '2\\,170\\,560'),
    ('$-20\\%$ edges removed','flyvis_noise_005_removed_pc_20_winner',    '0.05', '0',   '347\\,000'),
    ('$1/5$ frames',          'flyvis_noise_005_stride_5_winner',         '0.05', '0',   '434\\,112'),
    ('$10\\%$ hidden',        'flyvis_noise_005_hidden_010_ngp_winner',   '0.05', '0',   '434\\,112'),
]


def fmt(mean, sd):
    if np.isnan(mean):
        return '$\\cdot$'
    body = f"${mean:.2f}{{\\pm}}{sd:.2f}$"
    return f"\\good{{{body}}}" if mean > GOOD_THRESHOLD else body


def parse_last_cv_block(summary_path):
    """Return dict metric_name -> (mean, sd) from the LAST block of cv_summary.txt."""
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path) as f:
        txt = f.read()
    # Each block ends with a stats table; grab the last one's lines.
    blocks = txt.split('\nCV log:')
    block = 'CV log:' + blocks[-1]
    stats = {}
    # Pattern: "<name>  mean   sd   cv%  min  max" after the Metric header.
    in_stats = False
    for ln in block.splitlines():
        if ln.strip().startswith('Metric'):
            in_stats = True
            continue
        if not in_stats:
            continue
        m = re.match(r'\s*(\w+)\s+([-\d.]+|\S)\s+([-\d.]+|\S)\s+', ln)
        if m:
            name = m.group(1)
            mean_s, sd_s = m.group(2), m.group(3)
            try:
                stats[name] = (float(mean_s), float(sd_s))
            except ValueError:
                stats[name] = (float('nan'), float('nan'))
    return stats


def emit_row(label, config, noise_sig, noise_gam, edges_str, stats):
    def get(key):
        return stats.get(key, (float('nan'), float('nan')))
    one_m, one_s   = get('yt_one_step_r')
    roll_m, roll_s = get('yt_rollout_r')
    W_m, W_s       = get('W_corrected_R2')
    tau_m, tau_s   = get('tau_R2')
    V_m, V_s       = get('V_rest_R2')
    cl_m, cl_s     = get('clustering_accuracy')
    return (
        f'{label:<22} & ${noise_sig}$ & ${noise_gam}$ & ${edges_str}$\n'
        f'  & {fmt(one_m, one_s)} & {fmt(roll_m, roll_s)}\n'
        f'  & {fmt(W_m, W_s)} & {fmt(tau_m, tau_s)} & {fmt(V_m, V_s)} & {fmt(cl_m, cl_s)} \\\\'
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_root',
                   default='/groups/saalfeld/home/allierc/GraphData')
    p.add_argument('--pre_folder', default='fly')
    args = p.parse_args()

    rows = []
    for label, cfg, nsig, ngam, edges in CONDITIONS:
        summary = os.path.join(args.output_root, 'log', args.pre_folder,
                               cfg, 'results', 'cv_summary.txt')
        stats = parse_last_cv_block(summary)
        if not stats:
            print(f'WARN: no cv_summary.txt for {cfg} ({summary})',
                  file=sys.stderr)
        rows.append(emit_row(label, cfg, nsig, ngam, edges, stats))

    out_dir = os.path.join(args.output_root, 'log')
    os.makedirs(out_dir, exist_ok=True)
    out_tex = os.path.join(out_dir, 'cv_per_condition_rows.tex')
    with open(out_tex, 'w') as f:
        f.write('% --- tab:cv_per_condition — generated rows ---\n')
        for r in rows:
            f.write(r + '\n')
        f.write('% --------------------------------------------\n')

    print('% --- tab:cv_per_condition — generated rows ---')
    for r in rows:
        print(r)
    print('% --------------------------------------------')
    print(f'\nwrote {out_tex}')


if __name__ == '__main__':
    main()
