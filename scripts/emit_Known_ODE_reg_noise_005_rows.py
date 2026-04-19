"""
Emit the TeX row for the Known-ODE-reg low-intrinsic-noise condition, from
  <DATA_ROOT>/log/fly/flyvis_noise_005_known_ode_reg_winner/results/cv_summary.txt

Prediction columns (one-step r / rollout r) come from yt_one_step_r /
yt_rollout_r; parameter recovery columns come from the matching Phase 3
fold-avg entries.

Output:
    <DATA_ROOT>/log/cv_known_ode_reg_noise_005_rows.tex
    (also printed to stdout)
"""

import argparse
import os
import re
import sys

import numpy as np


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (os.path.join(REPO_ROOT, 'src'), REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


GOOD_THRESHOLD = 0.9

LABEL     = 'low intrinsic noise'
CONFIG    = 'flyvis_noise_005_known_ode_reg_winner'
NOISE_SIG = '0.05'
NOISE_GAM = '0'
EDGES     = '434\\,112'
OUT_BASENAME = 'cv_known_ode_reg_noise_005_rows.tex'


def fmt(mean, sd):
    if np.isnan(mean):
        return '$\\cdot$'
    body = f"${mean:.2f}{{\\pm}}{sd:.2f}$"
    return f"\\good{{{body}}}" if mean > GOOD_THRESHOLD else body


def parse_last_cv_block(summary_path):
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path) as f:
        txt = f.read()
    blocks = txt.split('\nCV log:')
    block = 'CV log:' + blocks[-1]
    stats = {}
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


def emit_row(label, noise_sig, noise_gam, edges_str, stats):
    def get(k):
        return stats.get(k, (float('nan'), float('nan')))
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

    summary = os.path.join(args.output_root, 'log', args.pre_folder,
                           CONFIG, 'results', 'cv_summary.txt')
    stats = parse_last_cv_block(summary)
    if not stats:
        print(f'WARN: no cv_summary.txt at {summary}', file=sys.stderr)
    row = emit_row(LABEL, NOISE_SIG, NOISE_GAM, EDGES, stats)

    out_dir = os.path.join(args.output_root, 'log')
    os.makedirs(out_dir, exist_ok=True)
    out_tex = os.path.join(out_dir, OUT_BASENAME)
    with open(out_tex, 'w') as f:
        f.write(f'% --- row for {CONFIG} ---\n')
        f.write(row + '\n')
        f.write('% -------------------------\n')

    print(f'% --- row for {CONFIG} ---')
    print(row)
    print('% -------------------------')
    print(f'\nwrote {out_tex}')


if __name__ == '__main__':
    main()
