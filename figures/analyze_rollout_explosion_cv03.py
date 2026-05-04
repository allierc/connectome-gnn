"""Analyze why flyvis_noise_free_blank50_unified_cv03 rollout explodes.

cv03 is the outlier in the 5-fold CV (table from run_GNN_unified_blank50.py):
    cv00..02,04 : Roll r ≈ 1.000, Roll RMSE ≈ 0.006-0.019
    cv03        : Roll r = 0.482, Roll RMSE = 57.23 — predictions clamp at ±100.

This script reads ``rollout_bundle.npz`` from each fold's results dir and
characterises the explosion in cv03 by:

  1. Onset timing — first frame where any neuron crosses |pred|>10 / |pred|=100.
  2. Per-frame number of clamped neurons (saturated at ±100).
  3. Cell-type susceptibility — fraction of each type that ever clamps.
  4. Pre-explosion residuals — neuron-level Pearson r over the pre-clamp window
     (frames 0..onset-1) to identify which neurons were already drifting before
     the clamp kicks in (i.e. the seeds of the runaway).
  5. Trace samples — early-exploding neurons (first to clamp) and late /
     non-exploding neurons, plotted side by side with cv00 traces for the same
     neuron indices as a healthy baseline.

Output: a multi-panel PNG and a printed text summary.

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/analyze_rollout_explosion_cv03.py

Output
------
    figures/analyze_rollout_explosion_cv03.png
    figures/analyze_rollout_explosion_cv03.txt
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Data root      : /groups/saalfeld/home/allierc/GraphData
# Configs        : <DATA_ROOT>/config/fly/flyvis_noise_free_blank50_unified_cv{00..04}.yaml
# Trained models : <DATA_ROOT>/log/fly/flyvis_noise_free_blank50_unified_cv{00..04}/models/best_model_with_0_graphs_0.pt
# Eval logs      : <DATA_ROOT>/log/fly/flyvis_noise_free_blank50_unified_cv{00..04}/results/rollout_bundle.npz
# Output         : figures/analyze_rollout_explosion_cv03.png
#                  figures/analyze_rollout_explosion_cv03.txt
# ─────────────────────────────────────────────────────────────────────────────

import collections
import os
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = '/groups/saalfeld/home/allierc/GraphData'
REPO_ROOT = '/workspace/connectome-gnn'
OUT_BASE  = os.path.join(REPO_ROOT, 'figures', 'analyze_rollout_explosion_cv03')

CV_FOLDS = ['cv00', 'cv01', 'cv02', 'cv03', 'cv04']
TARGET   = 'cv03'
HEALTHY  = 'cv00'

CLAMP_VALUE = 100.0
CLAMP_TOL   = 1e-2          # |pred| >= 99.99 counts as clamped
EARLY_THRESH = 10.0         # |pred| > 10 = "started diverging"


# ───────────────────────────────────────── helpers ──────────────────────────
def bundle_path(cv):
    return f'{DATA_ROOT}/log/fly/flyvis_noise_free_blank50_unified_{cv}/results/rollout_bundle.npz'


def load_bundle(cv):
    p = bundle_path(cv)
    if not os.path.isfile(p):
        sys.exit(f'missing rollout bundle: {p}')
    b = np.load(p, allow_pickle=True)
    return {
        'pred':       np.asarray(b['activity_pred']),
        'true':       np.asarray(b['activity_true']),
        'stim':       np.asarray(b['stimulus']),
        'type_ids':   np.asarray(b['type_ids']).astype(int),
        'type_names': list(b['type_names']),
    }


def per_neuron_pearson(pred, true):
    """Vectorised Pearson r per row of pred vs true. NaN-safe for zero-var rows."""
    pred = pred.astype(np.float64); true = true.astype(np.float64)
    pm = pred.mean(axis=1, keepdims=True); tm = true.mean(axis=1, keepdims=True)
    pd_ = pred - pm; td_ = true - tm
    num = (pd_ * td_).sum(axis=1)
    den = np.sqrt((pd_ ** 2).sum(axis=1) * (td_ ** 2).sum(axis=1))
    out = np.zeros(pred.shape[0], dtype=np.float64)
    nz = den > 1e-12
    out[nz] = num[nz] / den[nz]
    return out


# ───────────────────────────────────── 1. cross-fold summary ────────────────
def cross_fold_summary():
    rows = []
    for cv in CV_FOLDS:
        b = load_bundle(cv)
        pred = b['pred']; true = b['true']
        clamped = np.abs(pred) >= CLAMP_VALUE - CLAMP_TOL
        pearson = per_neuron_pearson(pred, true)
        rows.append({
            'cv': cv,
            'pred_min': pred.min(), 'pred_max': pred.max(),
            'pred_mean': pred.mean(), 'pred_std': pred.std(),
            'frac_clamped_any': clamped.any(axis=1).mean(),
            'first_clamp_frame': (int(np.argmax(clamped.any(axis=0)))
                                  if clamped.any() else -1),
            'mean_pearson': pearson.mean(),
            'median_pearson': np.median(pearson),
        })
    return rows


# ───────────────────────────────────── 2. cv03 onset analysis ───────────────
def onset_analysis(b):
    pred = b['pred']
    abs_pred = np.abs(pred)
    n_neurons, n_frames = pred.shape

    # max |pred| across neurons at each frame (the explosion envelope)
    max_per_frame = abs_pred.max(axis=0)
    n_clamp_per_frame = (abs_pred >= CLAMP_VALUE - CLAMP_TOL).sum(axis=0)
    n_diverging_per_frame = (abs_pred > EARLY_THRESH).sum(axis=0)

    # First time any neuron crossed each threshold
    crossings = {}
    for label, thresh in [('|pred|>1', 1.0),
                          ('|pred|>10', EARLY_THRESH),
                          ('|pred|>50', 50.0),
                          ('|pred|=100', CLAMP_VALUE - CLAMP_TOL)]:
        hits = abs_pred > thresh
        any_hit_per_frame = hits.any(axis=0)
        crossings[label] = (int(np.argmax(any_hit_per_frame))
                            if any_hit_per_frame.any() else -1)

    # First neuron to clamp, and its cell type
    clamped = abs_pred >= CLAMP_VALUE - CLAMP_TOL
    first_clamp_per_neuron = np.full(n_neurons, n_frames, dtype=np.int64)
    has_clamp = clamped.any(axis=1)
    first_clamp_per_neuron[has_clamp] = np.argmax(clamped[has_clamp], axis=1)

    return {
        'max_per_frame':       max_per_frame,
        'n_clamp_per_frame':   n_clamp_per_frame,
        'n_diverging_per_frame': n_diverging_per_frame,
        'crossings':           crossings,
        'first_clamp_per_neuron': first_clamp_per_neuron,
        'has_clamp':           has_clamp,
    }


# ───────────────────────────────── 3. per-cell-type breakdown ───────────────
def per_celltype_breakdown(b, has_clamp):
    tids = b['type_ids']; tnames = b['type_names']
    out = []
    for tid, name in enumerate(tnames):
        mask = tids == tid
        n_total = mask.sum()
        if n_total == 0:
            continue
        n_clamp = has_clamp[mask].sum()
        out.append({'type': name, 'n_total': int(n_total),
                    'n_clamp': int(n_clamp),
                    'frac_clamp': n_clamp / n_total})
    out.sort(key=lambda r: (-r['frac_clamp'], -r['n_clamp']))
    return out


# ───────────────────────────────── 4. pre-clamp drift ───────────────────────
def pre_clamp_drift(b, onset_frame):
    """Per-neuron Pearson r over frames [0, onset_frame).

    onset_frame = first frame where ANY neuron clamps. Window before that
    is the only stretch where the model is still in its valid regime; if a
    neuron's pre-window r is already low, it was diverging before the clamp.
    """
    pre_pred = b['pred'][:, :onset_frame]
    pre_true = b['true'][:, :onset_frame]
    return per_neuron_pearson(pre_pred, pre_true)


# ───────────────────────────────────── plotting ─────────────────────────────
def plot_summary(b03, b00, info03, ct_rows, pre_r03, out_png):
    pred03 = b03['pred']; true03 = b03['true']
    pred00 = b00['pred']; true00 = b00['true']
    tnames = b03['type_names']; tids = b03['type_ids']
    n_frames = pred03.shape[1]

    fig = plt.figure(figsize=(16, 14), dpi=120)
    gs = mgs.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.30,
                      left=0.06, right=0.98, top=0.95, bottom=0.05)

    # A — explosion envelope: max |pred| per frame
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(np.maximum(info03['max_per_frame'], 1e-3), color='k', lw=0.8)
    for label, frame in info03['crossings'].items():
        if frame > 0:
            ax.axvline(frame, ls='--', lw=0.6, alpha=0.6)
            ax.text(frame, ax.get_ylim()[1] * 0.5, f' {label}@{frame}',
                    fontsize=7, rotation=90, va='top')
    ax.set_xlabel('frame'); ax.set_ylabel('max |pred| across neurons')
    ax.set_title('A. cv03 explosion envelope (log scale)', fontsize=10)

    # B — # of clamped / diverging neurons per frame
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(info03['n_diverging_per_frame'], color='tab:orange',
            lw=0.8, label='|pred|>10')
    ax.plot(info03['n_clamp_per_frame'], color='tab:red',
            lw=0.8, label='|pred|=100 (clamped)')
    ax.set_xlabel('frame'); ax.set_ylabel('# neurons')
    ax.set_title('B. divergence spread over time', fontsize=10)
    ax.legend(fontsize=7)

    # C — histogram of first-clamp frames for clamped neurons
    ax = fig.add_subplot(gs[0, 2])
    fc = info03['first_clamp_per_neuron'][info03['has_clamp']]
    ax.hist(fc, bins=80, color='tab:red', alpha=0.7)
    ax.set_xlabel('first frame |pred|=100'); ax.set_ylabel('# neurons')
    ax.set_title(f'C. first clamp time '
                 f'({info03["has_clamp"].sum()} neurons clamp)', fontsize=10)

    # D — per cell-type fraction clamped (top 25)
    ax = fig.add_subplot(gs[1, :])
    top = ct_rows[:25]
    names = [r['type'] for r in top]
    fracs = [r['frac_clamp'] for r in top]
    counts = [f"{r['n_clamp']}/{r['n_total']}" for r in top]
    ax.bar(range(len(top)), fracs, color='tab:red', alpha=0.8)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(names, rotation=60, ha='right', fontsize=8)
    for i, c in enumerate(counts):
        ax.text(i, fracs[i] + 0.02, c, ha='center', fontsize=6)
    ax.set_ylim(0, 1.15); ax.set_ylabel('fraction clamped')
    ax.set_title('D. per cell-type fraction that ever clamps '
                 '(top 25 most affected)', fontsize=10)

    # E — pre-clamp Pearson r distribution
    onset = info03['crossings']['|pred|=100']
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(pre_r03, bins=80, color='steelblue', alpha=0.8)
    ax.axvline(np.median(pre_r03), color='k', ls='--', lw=0.8,
               label=f'median={np.median(pre_r03):.3f}')
    ax.set_xlabel(f'per-neuron pearson r over frames [0, {onset})')
    ax.set_ylabel('# neurons')
    ax.set_title(f'E. pre-clamp drift '
                 f'(window: 0–{onset}; window is the only valid regime)',
                 fontsize=10)
    ax.legend(fontsize=7)

    # F — pre-clamp r vs first-clamp time scatter
    ax = fig.add_subplot(gs[2, 1])
    fc_full = info03['first_clamp_per_neuron'].astype(float)
    fc_full[~info03['has_clamp']] = np.nan
    ax.scatter(pre_r03, fc_full, s=3, alpha=0.3, color='tab:red')
    ax.set_xlabel('pre-clamp pearson r'); ax.set_ylabel('first clamp frame')
    ax.set_title('F. early-r vs explosion latency', fontsize=10)

    # G — example traces: 3 first-clampers + 3 never-clampers
    early_idx = np.argsort(info03['first_clamp_per_neuron'])[:3]
    if (~info03['has_clamp']).any():
        late_idx = np.where(~info03['has_clamp'])[0]
        late_pick = late_idx[np.linspace(0, len(late_idx) - 1, 3).astype(int)]
    else:
        late_pick = np.argsort(info03['first_clamp_per_neuron'])[-3:]
    pick = list(early_idx) + list(late_pick)
    titles = [f'EARLY-clamp #{i} ({tnames[tids[i]]})' for i in early_idx] + \
             [f'NON-clamp #{i} ({tnames[tids[i]]})' for i in late_pick]

    for k, (idx, title) in enumerate(zip(pick, titles)):
        row = 3
        col = k % 3
        if k < 3:
            ax = fig.add_subplot(gs[row, col])
        else:
            ax = fig.add_subplot(gs[row, col])
            # second row — overwrite if more than 3 picks
        ax.plot(true03[idx], color='tab:green', lw=0.7, label='gt', alpha=0.8)
        ax.plot(pred03[idx], color='k', lw=0.5, label='cv03 pred')
        ax.plot(pred00[idx], color='tab:blue', lw=0.5, alpha=0.6,
                label='cv00 pred (healthy)')
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('frame'); ax.set_ylabel('voltage')
        if k == 0:
            ax.legend(fontsize=6)

    fig.suptitle('cv03 rollout explosion analysis '
                 '(flyvis_noise_free_blank50_unified)', fontsize=12)
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_png}')


# ───────────────────────────────────── text report ──────────────────────────
def write_report(rows, info03, ct_rows, pre_r03, out_txt, b03):
    lines = []
    lines.append('=== Cross-fold summary ===')
    lines.append(f"{'cv':<6}{'pred_min':>10}{'pred_max':>10}{'pred_mean':>11}"
                 f"{'pred_std':>10}{'frac_clamp':>12}{'first_clamp':>13}"
                 f"{'mean_r':>10}{'med_r':>10}")
    for r in rows:
        lines.append(f"{r['cv']:<6}{r['pred_min']:>10.3f}{r['pred_max']:>10.3f}"
                     f"{r['pred_mean']:>11.3f}{r['pred_std']:>10.3f}"
                     f"{r['frac_clamped_any']:>12.3f}"
                     f"{r['first_clamp_frame']:>13d}"
                     f"{r['mean_pearson']:>10.3f}{r['median_pearson']:>10.3f}")

    lines.append('')
    lines.append('=== cv03 onset timing ===')
    for k, v in info03['crossings'].items():
        lines.append(f'  first frame with {k:<12s}: {v}')
    lines.append(f"  total neurons that ever clamp: "
                 f"{info03['has_clamp'].sum()}/{len(info03['has_clamp'])}")

    lines.append('')
    lines.append('=== Per cell-type explosion (top 25) ===')
    lines.append(f"{'type':<14}{'n_clamp':>10}{'n_total':>10}{'frac':>10}")
    for r in ct_rows[:25]:
        lines.append(f"{r['type']:<14}{r['n_clamp']:>10}{r['n_total']:>10}"
                     f"{r['frac_clamp']:>10.3f}")

    lines.append('')
    lines.append('=== Cell types that NEVER clamp ===')
    safe = [r for r in ct_rows if r['n_clamp'] == 0]
    for r in safe:
        lines.append(f"  {r['type']:<14}({r['n_total']} cells)")

    lines.append('')
    onset = info03['crossings']['|pred|=100']
    lines.append(f'=== Pre-clamp Pearson r (window 0-{onset}) ===')
    lines.append(f'  mean   = {pre_r03.mean():.4f}')
    lines.append(f'  median = {np.median(pre_r03):.4f}')
    lines.append(f'  q10    = {np.quantile(pre_r03, 0.10):.4f}')
    lines.append(f'  q90    = {np.quantile(pre_r03, 0.90):.4f}')

    # Per-cell-type pre-clamp r — useful to see whether bad types track explosion
    lines.append('')
    lines.append('=== Pre-clamp r by cell type (top 15 worst-tracking types) ===')
    tids = b03['type_ids']; tnames = b03['type_names']
    type_rs = []
    for tid, name in enumerate(tnames):
        mask = tids == tid
        if mask.sum() == 0:
            continue
        type_rs.append((name, float(np.mean(pre_r03[mask])), int(mask.sum())))
    type_rs.sort(key=lambda x: x[1])
    lines.append(f"{'type':<14}{'mean_r':>10}{'n':>8}")
    for name, r, n in type_rs[:15]:
        lines.append(f"{name:<14}{r:>10.4f}{n:>8d}")

    txt = '\n'.join(lines)
    with open(out_txt, 'w') as f:
        f.write(txt + '\n')
    print(txt)
    print(f'\nwrote {out_txt}')


# ───────────────────────────────────────── main ─────────────────────────────
def main():
    print('loading bundles for cross-fold summary...')
    rows = cross_fold_summary()

    print(f'\nloading {TARGET} (broken) and {HEALTHY} (healthy baseline)...')
    b03 = load_bundle(TARGET)
    b00 = load_bundle(HEALTHY)

    print('analysing cv03 onset...')
    info03 = onset_analysis(b03)

    print('per-cell-type breakdown...')
    ct_rows = per_celltype_breakdown(b03, info03['has_clamp'])

    onset = info03['crossings']['|pred|=100']
    if onset <= 0:
        sys.exit('no clamping detected in cv03 — re-check the bundle')
    print(f'pre-clamp Pearson r over frames [0, {onset})...')
    pre_r03 = pre_clamp_drift(b03, onset)

    out_png = OUT_BASE + '.png'
    out_txt = OUT_BASE + '.txt'
    plot_summary(b03, b00, info03, ct_rows, pre_r03, out_png)
    write_report(rows, info03, ct_rows, pre_r03, out_txt, b03)


if __name__ == '__main__':
    main()
