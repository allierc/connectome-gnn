"""Who gets excluded by the residual filter? (NeurIPS-2026 rebuttal, AC demand #2)

The submitted tables report tau/V_rest R^2 over the *inlier* set left after the
residual filter (neurips.tex eq:outlier_threshold, delta_tau=0.1 /
delta_Vrest=0.2). Reviewers ask what the excluded neurons are. This script
characterises them for the worst conditions, along three axes:

  1. cell type      -- per-type exclusion rate + enrichment vs the global rate,
                       and each type's share of the excluded set
  2. activity       -- excluded vs kept: mean voltage, voltage SD, and the
                       fraction of frames with v>0 (the ReLU-active fraction;
                       a neuron that is never active carries no drive, so its
                       V_rest / tau are weakly constrained)
  3. identifiability-- Jaccard overlap with the Appx. C degenerate set
                       {i : d_i > r}, d_i = in-degree, r = effective rank of the
                       population activity at 99% cumulative variance
                       (Appx. C procedure: H in R^{8000 x 982}, r=45 for
                       Flyvis-217). Reported against the Jaccard expected for a
                       RANDOM set of the same size -- a bare Jaccard is
                       uninterpretable.

Plus cross-fold consistency: is the same neuron excluded in every fold, or is
exclusion fold-specific noise?

NO training, NO GPU. Everything is read from artifacts already on disk:
    log/fly/<config>_cv<NN>/results/learned_ode_params.pt   (tau_i, V_i_rest)
    graphs_data/fly/<dataset>/                              (GT params, activity)

Run:
    GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData PYTHONPATH=src \
      /workspace/.conda_envs/neural-graph-linux/bin/python \
      neurips_review/excluded_neurons.py
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, 'src'))

from connectome_gnn.generators.ode_params import get_ode_params_class
from connectome_gnn.models.training_utils import load_flyvis_data
from connectome_gnn.utils import graphs_data_path, set_data_root

# Filter thresholds. GNN_PlotFigure.py currently declares DELTA_TAU=0.1,
# DELTA_VREST=0.2, and neurips.tex eq:outlier_threshold states the same. BUT the
# Known-ODE runs on disk were plotted BEFORE commit 64fd6b9 ("unify outlier
# thresholds") and their recorded V_rest_n_outliers reproduces only at
# delta_Vrest=0.1, while the GNN runs reproduce at 0.2. So the threshold is
# *detected per run* by matching the recorded count, and reported explicitly --
# assuming 0.2 everywhere would characterise a different set than the tables do.
TAU_THRESH = 0.1
VREST_THRESH_CANDIDATES = (0.2, 0.1)
VREST_THRESH_DEFAULT = 0.2

# Appx. C degenerate set: {i : d_i > APPX_C_RANK}, d_i = in-degree. The rank is
# FIXED at the Appx. C value rather than re-estimated per condition: the
# effective rank of the *measured* activity is inflated by measurement noise
# (gamma=0.2 gives r~650, not 45), but the degenerate set is a property of the
# circuit and stimulus, not of the observation noise. The per-condition rank is
# still computed and reported as a diagnostic.
APPX_C_RANK = 45
N_FRAMES_SVD = 8000
N_NEURONS_SVD = 982
VAR_THRESH = 0.99

# Activity stats are computed on a strided subsample (the full voltage tensor is
# 64000 x N float32 = 3.5 GB at N=13741, 13 GB at N=50412).
ACTIVITY_STRIDE = 16

N_FOLDS = 5

# The conditions the reviewers flagged, by their V_rest excluded %.
# 49.1% is ambiguous -- it occurs for BOTH the GNN mid-meas-noise row (Supp.
# Tab. 4) and the Known-ODE low-meas-noise row (Supp. Tab. 7); both are included.
CONDITIONS = [
    dict(key='fulleye_prox_gnn', excl_pct=40.8, table='Tab. 2',
         label='FlyWire full eye, proximal nulls (GNN)',
         config='full_eye_flywireRF_proximal_nulls_noise_005_blank50_flywire'),
    dict(key='stride5_gnn', excl_pct=56.3, table='Supp. Tab. 4',
         label='1/5 frames (GNN)',
         config='flyvis_noise_005_stride_5_blank50_unified'),
    dict(key='midmeas_gnn', excl_pct=49.1, table='Supp. Tab. 4',
         label='mid meas. noise gamma=0.2 (GNN)',
         config='flyvis_noise_005_020_blank50_unified'),
    dict(key='lowmeas_ko', excl_pct=49.1, table='Supp. Tab. 7',
         label='low meas. noise gamma=0.1 (Known-ODE)',
         config='flyvis_noise_005_010_blank50_known_ode'),
    dict(key='midmeas_ko', excl_pct=67.9, table='Supp. Tab. 7',
         label='mid meas. noise gamma=0.2 (Known-ODE)',
         config='flyvis_noise_005_020_blank50_known_ode'),
]


# ---------------------------------------------------------------------------
# Paths / loading
# ---------------------------------------------------------------------------
def _log_dir(root, config, fold):
    return os.path.join(root, 'log', 'fly', f'{config}_cv{fold:02d}')


def _resolve_dataset(root, config, fold):
    """Dataset name for a fold: log/config.yaml if present, else config/fly/."""
    cfg_name = f'{config}_cv{fold:02d}'
    for path in (os.path.join(_log_dir(root, config, fold), 'config.yaml'),
                 os.path.join(root, 'config', 'fly', f'{cfg_name}.yaml')):
        if os.path.isfile(path):
            with open(path) as f:
                d = yaml.safe_load(f)
            if d and d.get('dataset'):
                return d['dataset']
    raise FileNotFoundError(f'cannot resolve dataset for {cfg_name}')


def _load_learned(root, config, fold):
    path = os.path.join(_log_dir(root, config, fold),
                        'results', 'learned_ode_params.pt')
    d = torch.load(path, map_location='cpu', weights_only=False)
    return (d['tau_i'].float().numpy().ravel(),
            d['V_i_rest'].float().numpy().ravel())


def _load_gt(dataset):
    OdeCls = get_ode_params_class('flyvis_known_ode')
    ode = OdeCls.load(graphs_data_path(f'fly/{dataset}'), device='cpu')
    return (ode.tau_i.float().numpy().ravel(),
            ode.V_i_rest.float().numpy().ravel(),
            ode.edge_index.cpu().numpy())


# ---------------------------------------------------------------------------
# Cell-type names
# ---------------------------------------------------------------------------
_TYPE_NAME_CACHE = {}


def _type_names(n_neurons, all_columns):
    """Per-neuron flyvis cell-type strings; falls back to type_<idx> labels."""
    key = (n_neurons, bool(all_columns))
    if key in _TYPE_NAME_CACHE:
        return _TYPE_NAME_CACHE[key]
    names = None
    try:
        from flyvis import Network
        from flyvis.utils.config_utils import CONFIG_PATH, get_default_config
        cfg = get_default_config(overrides=[],
                                path=f'{CONFIG_PATH}/network/network.yaml')
        cfg.connectome.extent = 15 if all_columns else 8
        net = Network(**cfg)
        raw = np.array(net.connectome.nodes['type'])
        cand = np.array([t.decode() if isinstance(t, bytes) else str(t)
                         for t in raw])
        if len(cand) == n_neurons:
            names = cand
    except Exception as exc:                                # noqa: BLE001
        print(f'    [warn] flyvis cell-type names unavailable ({exc.__class__.__name__});'
              ' using numeric type labels')
    _TYPE_NAME_CACHE[key] = names
    return names


# ---------------------------------------------------------------------------
# Per-fold analysis
# ---------------------------------------------------------------------------
def _effective_rank(voltage, n_frames, n_neurons_sub, var_thresh, rng):
    """Appx. C: effective rank of subsampled population activity at `var_thresh`."""
    T, N = voltage.shape
    f_idx = np.linspace(0, T - 1, min(n_frames, T)).astype(np.int64)
    n_sub = min(n_neurons_sub, N)
    n_idx = rng.choice(N, size=n_sub, replace=False)
    n_idx.sort()
    H = voltage[f_idx][:, n_idx].numpy().astype(np.float64)
    H = H - H.mean(axis=0, keepdims=True)
    s = np.linalg.svd(H, compute_uv=False)
    ev = s ** 2
    if ev.sum() <= 0:
        return 0
    cum = np.cumsum(ev) / ev.sum()
    return int(np.searchsorted(cum, var_thresh) + 1)


def _activity_stats(voltage, stride):
    """Per-neuron (mean v, SD v, fraction of frames with v>0) on a subsample."""
    v = voltage[::stride]
    return (v.mean(axis=0).numpy(),
            v.std(axis=0).numpy(),
            (v > 0).float().mean(axis=0).numpy())


def _jaccard(a_mask, b_mask):
    inter = int(np.logical_and(a_mask, b_mask).sum())
    union = int(np.logical_or(a_mask, b_mask).sum())
    return (inter / union) if union else float('nan'), inter


def _expected_jaccard(n_a, n_b, n_total):
    """Jaccard expected if the two sets of the same sizes were independent."""
    if not n_total:
        return float('nan')
    exp_inter = n_a * n_b / n_total
    union = n_a + n_b - exp_inter
    return (exp_inter / union) if union else float('nan'), exp_inter


def _recorded_metrics(root, config, fold):
    path = os.path.join(_log_dir(root, config, fold), 'results', 'metrics.txt')
    out = {}
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                if ':' in line:
                    k, v = line.split(':', 1)
                    try:
                        out[k.strip()] = float(v)
                    except ValueError:
                        pass
    return out


def _detect_vrest_thresh(residual, recorded_n):
    """Pick the delta whose exclusion count reproduces the recorded one."""
    if recorded_n is None:
        return VREST_THRESH_DEFAULT, False
    for th in VREST_THRESH_CANDIDATES:
        if int((residual > th).sum()) == int(recorded_n):
            return th, True
    return VREST_THRESH_DEFAULT, False


def _analyse_fold(root, cond, fold, rng):
    dataset = _resolve_dataset(root, cond['config'], fold)
    tau_hat, vrest_hat = _load_learned(root, cond['config'], fold)
    tau_gt, vrest_gt, edge_index = _load_gt(dataset)
    N = tau_gt.shape[0]

    x_ts, _, type_list = load_flyvis_data(
        dataset_name=f'fly/{dataset}', split='train',
        fields=['voltage', 'neuron_type'])
    voltage = x_ts.voltage
    types = type_list.numpy().ravel().astype(np.int64)

    rec = _recorded_metrics(root, cond['config'], fold)
    v_res = np.abs(vrest_hat - vrest_gt)
    v_thresh, matched = _detect_vrest_thresh(v_res, rec.get('V_rest_n_outliers'))
    v_out = v_res > v_thresh
    t_out = np.abs(tau_hat - tau_gt) > TAU_THRESH

    # exclusion % at BOTH thresholds, so the reply can state a consistent number
    excl_at = {th: float((v_res > th).mean()) for th in VREST_THRESH_CANDIDATES}

    in_deg = np.bincount(edge_index[1], minlength=N)
    rank = _effective_rank(voltage, N_FRAMES_SVD, N_NEURONS_SVD, VAR_THRESH, rng)
    degen = in_deg > APPX_C_RANK

    mean_v, sd_v, act_frac = _activity_stats(voltage, ACTIVITY_STRIDE)

    jac, inter = _jaccard(v_out, degen)
    exp_jac, exp_inter = _expected_jaccard(int(v_out.sum()), int(degen.sum()), N)

    return dict(
        dataset=dataset, n_neurons=N, rank=rank,
        v_thresh=v_thresh, thresh_matched=matched, excl_at=excl_at,
        v_out=v_out, t_out=t_out, degen=degen, types=types,
        n_v_out=int(v_out.sum()), n_t_out=int(t_out.sum()),
        n_degen=int(degen.sum()),
        jaccard=jac, jaccard_expected=exp_jac,
        overlap=inter, overlap_expected=exp_inter,
        mean_v=mean_v, sd_v=sd_v, act_frac=act_frac,
        in_deg=in_deg,
    )


# ---------------------------------------------------------------------------
# Aggregation across folds
# ---------------------------------------------------------------------------
def _med(x):
    return float(np.median(x)) if len(x) else float('nan')


def _aggregate(cond, folds, type_names):
    N = folds[0]['n_neurons']
    n_f = len(folds)

    excl_count = np.zeros(N, dtype=np.int64)
    for f in folds:
        excl_count += f['v_out'].astype(np.int64)

    # --- activity: excluded vs kept, per fold then averaged over folds ---
    act = {k: {'out': [], 'keep': []} for k in ('mean_v', 'sd_v', 'act_frac')}
    for f in folds:
        m = f['v_out']
        for k in act:
            act[k]['out'].append(_med(f[k][m]))
            act[k]['keep'].append(_med(f[k][~m]))
    act_summary = {
        k: dict(excluded=float(np.mean(v['out'])),
                kept=float(np.mean(v['keep'])))
        for k, v in act.items()
    }

    # --- per-cell-type exclusion rate, averaged over folds ---
    types = folds[0]['types']
    uniq = np.unique(types)
    global_rate = float(np.mean([f['n_v_out'] / f['n_neurons'] for f in folds]))
    per_type = []
    for t in uniq:
        idx = types == t
        n_t = int(idx.sum())
        rates = [float(f['v_out'][idx].mean()) for f in folds]
        shares = [float(f['v_out'][idx].sum()) / max(1, f['n_v_out'])
                  for f in folds]
        rate = float(np.mean(rates))
        name = (type_names[np.where(idx)[0][0]]
                if type_names is not None else f'type_{int(t)}')
        per_type.append(dict(
            type=name, type_idx=int(t), n_neurons=n_t,
            excl_rate=rate, excl_rate_sd=float(np.std(rates)),
            enrichment=(rate / global_rate) if global_rate else float('nan'),
            share_of_excluded=float(np.mean(shares)),
            median_act_frac=_med(folds[0]['act_frac'][idx]),
            median_in_deg=float(np.median(folds[0]['in_deg'][idx])),
        ))
    per_type.sort(key=lambda d: d['excl_rate'], reverse=True)

    return dict(
        key=cond['key'], label=cond['label'], table=cond['table'],
        config=cond['config'], reported_excl_pct=cond['excl_pct'],
        n_neurons=N, n_folds=n_f,
        dataset=folds[0]['dataset'],
        vrest_thresh=float(np.mean([f['v_thresh'] for f in folds])),
        thresh_reproduces_paper=bool(all(f['thresh_matched'] for f in folds)),
        excl_pct_at_010=100.0 * float(np.mean([f['excl_at'][0.1] for f in folds])),
        excl_pct_at_020=100.0 * float(np.mean([f['excl_at'][0.2] for f in folds])),
        effective_rank=float(np.mean([f['rank'] for f in folds])),
        excl_pct_vrest=100.0 * float(np.mean([f['n_v_out'] / f['n_neurons']
                                              for f in folds])),
        excl_pct_vrest_sd=100.0 * float(np.std([f['n_v_out'] / f['n_neurons']
                                                for f in folds])),
        excl_pct_tau=100.0 * float(np.mean([f['n_t_out'] / f['n_neurons']
                                            for f in folds])),
        degen_pct=100.0 * float(np.mean([f['n_degen'] / f['n_neurons']
                                         for f in folds])),
        jaccard=float(np.mean([f['jaccard'] for f in folds])),
        jaccard_expected=float(np.mean([f['jaccard_expected'] for f in folds])),
        overlap_enrichment=float(np.mean([f['overlap'] / f['overlap_expected']
                                          for f in folds
                                          if f['overlap_expected'] > 0])),
        # cross-fold consistency of exclusion
        excluded_ever=int((excl_count > 0).sum()),
        excluded_always=int((excl_count == n_f).sum()),
        excluded_majority=int((excl_count >= max(1, n_f - 1)).sum()),
        consistency=(float((excl_count == n_f).sum())
                     / max(1, int((excl_count > 0).sum()))),
        activity=act_summary,
        per_type=per_type,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_summary(r):
    print(f"\n  {r['label']}   [{r['table']}]")
    print(f"    config     {r['config']}  ({r['n_folds']} folds, N={r['n_neurons']})")
    ok = 'reproduces paper' if r['thresh_reproduces_paper'] else 'NO MATCH to paper'
    print(f"    excluded   V_rest {r['excl_pct_vrest']:.1f}+-{r['excl_pct_vrest_sd']:.1f}%"
          f"  (paper reports {r['reported_excl_pct']:.1f}%)"
          f"   tau {r['excl_pct_tau']:.1f}%")
    print(f"    threshold  delta_Vrest={r['vrest_thresh']:.2f} [{ok}]"
          f"   excl at 0.1={r['excl_pct_at_010']:.1f}%"
          f"  at 0.2={r['excl_pct_at_020']:.1f}%")
    a = r['activity']
    print(f"    activity   excluded vs kept (median over neurons, mean over folds)")
    print(f"                 mean v      {a['mean_v']['excluded']:+.4f}  vs {a['mean_v']['kept']:+.4f}")
    print(f"                 SD v        {a['sd_v']['excluded']:.4f}  vs {a['sd_v']['kept']:.4f}")
    print(f"                 frac v>0    {a['act_frac']['excluded']:.3f}  vs {a['act_frac']['kept']:.3f}")
    print(f"    degeneracy eff. rank r={r['effective_rank']:.0f}"
          f"  -> degenerate set {r['degen_pct']:.1f}% of neurons")
    print(f"                 Jaccard {r['jaccard']:.3f}"
          f"  (random baseline {r['jaccard_expected']:.3f},"
          f" overlap enrichment {r['overlap_enrichment']:.2f}x)")
    print(f"    consistency {r['excluded_always']} / {r['excluded_ever']} neurons"
          f" excluded in ALL {r['n_folds']} folds ({100*r['consistency']:.0f}%)")
    print(f"    top excluded types:")
    for d in r['per_type'][:6]:
        print(f"                 {d['type']:<10} n={d['n_neurons']:<6}"
              f" rate={100*d['excl_rate']:5.1f}%"
              f" enrich={d['enrichment']:4.2f}x"
              f" share={100*d['share_of_excluded']:4.1f}%"
              f" frac_v>0={d['median_act_frac']:.3f}")


def _write_csvs(out_dir, results):
    import csv
    p1 = os.path.join(out_dir, 'excluded_neurons_summary.csv')
    with open(p1, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key', 'label', 'table', 'config', 'n_neurons', 'n_folds',
                    'excl_pct_vrest', 'excl_pct_vrest_sd', 'reported_excl_pct',
                    'vrest_thresh', 'thresh_reproduces_paper',
                    'excl_pct_at_010', 'excl_pct_at_020',
                    'excl_pct_tau', 'effective_rank', 'degen_pct',
                    'jaccard', 'jaccard_random', 'overlap_enrichment',
                    'excluded_ever', 'excluded_always', 'consistency',
                    'med_mean_v_excl', 'med_mean_v_kept',
                    'med_sd_v_excl', 'med_sd_v_kept',
                    'med_actfrac_excl', 'med_actfrac_kept'])
        for r in results:
            a = r['activity']
            w.writerow([r['key'], r['label'], r['table'], r['config'],
                        r['n_neurons'], r['n_folds'],
                        f"{r['excl_pct_vrest']:.2f}", f"{r['excl_pct_vrest_sd']:.2f}",
                        f"{r['reported_excl_pct']:.1f}",
                        f"{r['vrest_thresh']:.2f}", r['thresh_reproduces_paper'],
                        f"{r['excl_pct_at_010']:.2f}", f"{r['excl_pct_at_020']:.2f}",
                        f"{r['excl_pct_tau']:.2f}",
                        f"{r['effective_rank']:.0f}", f"{r['degen_pct']:.2f}",
                        f"{r['jaccard']:.4f}", f"{r['jaccard_expected']:.4f}",
                        f"{r['overlap_enrichment']:.3f}",
                        r['excluded_ever'], r['excluded_always'],
                        f"{r['consistency']:.3f}",
                        f"{a['mean_v']['excluded']:.5f}", f"{a['mean_v']['kept']:.5f}",
                        f"{a['sd_v']['excluded']:.5f}", f"{a['sd_v']['kept']:.5f}",
                        f"{a['act_frac']['excluded']:.4f}", f"{a['act_frac']['kept']:.4f}"])
    p2 = os.path.join(out_dir, 'excluded_neurons_by_type.csv')
    with open(p2, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key', 'label', 'type', 'type_idx', 'n_neurons',
                    'excl_rate', 'excl_rate_sd', 'enrichment',
                    'share_of_excluded', 'median_act_frac', 'median_in_deg'])
        for r in results:
            for d in r['per_type']:
                w.writerow([r['key'], r['label'], d['type'], d['type_idx'],
                            d['n_neurons'], f"{d['excl_rate']:.4f}",
                            f"{d['excl_rate_sd']:.4f}", f"{d['enrichment']:.3f}",
                            f"{d['share_of_excluded']:.4f}",
                            f"{d['median_act_frac']:.4f}",
                            f"{d['median_in_deg']:.1f}"])
    return p1, p2


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output_root',
                   default=os.environ.get('GNN_OUTPUT_ROOT'))
    p.add_argument('--only', default=None,
                   help='comma-separated condition keys to run')
    p.add_argument('--n_folds', type=int, default=N_FOLDS)
    args = p.parse_args()

    root = args.output_root
    assert root and os.path.isdir(root), f'bad output_root: {root!r}'
    set_data_root(root)

    conds = CONDITIONS
    if args.only:
        keep = set(args.only.split(','))
        conds = [c for c in conds if c['key'] in keep]

    print('=' * 78)
    print('excluded-neuron characterisation  (rebuttal, AC demand #2)')
    print(f'  root {root}')
    print(f'  filters: |tau_hat - tau| > {TAU_THRESH};'
          f' delta_Vrest detected per run from {VREST_THRESH_CANDIDATES}'
          f' (default {VREST_THRESH_DEFAULT})')
    print('=' * 78)

    results = []
    for cond in conds:
        print(f"\n[{cond['key']}] {cond['label']}")
        folds = []
        for fold in range(args.n_folds):
            d = _log_dir(root, cond['config'], fold)
            if not os.path.isfile(os.path.join(d, 'results',
                                               'learned_ode_params.pt')):
                print(f'    cv{fold:02d}: missing learned_ode_params.pt -- skip')
                continue
            rng = np.random.default_rng(1234 + fold)
            try:
                folds.append(_analyse_fold(root, cond, fold, rng))
                print(f"    cv{fold:02d}: ok  ({folds[-1]['dataset']},"
                      f" N={folds[-1]['n_neurons']},"
                      f" V_rest excl {100*folds[-1]['n_v_out']/folds[-1]['n_neurons']:.1f}%,"
                      f" r={folds[-1]['rank']})")
            except Exception as exc:                        # noqa: BLE001
                print(f'    cv{fold:02d}: FAILED {exc.__class__.__name__}: {exc}')
        if not folds:
            print('    no usable folds -- skipping condition')
            continue
        names = _type_names(folds[0]['n_neurons'],
                            all_columns='full_eye' in cond['config'])
        r = _aggregate(cond, folds, names)
        _print_summary(r)
        results.append(r)

    if results:
        p1, p2 = _write_csvs(_HERE, results)
        print(f'\nwrote {p1}')
        print(f'wrote {p2}')
        slim = [{k: v for k, v in r.items() if k != 'per_type'} for r in results]
        with open(os.path.join(_HERE, 'excluded_neurons.json'), 'w') as f:
            json.dump(slim, f, indent=2)
    print('\ndone.')


if __name__ == '__main__':
    main()
