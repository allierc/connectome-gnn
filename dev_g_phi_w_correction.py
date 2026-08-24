"""Standalone experiment: derive the per-neuron g_phi correction factor
eta_j from real (edge, frame) local gradients d(g_phi)/d(vj) instead of the
current single global 1D-sweep affine fit, and test several data-only
filters for which points to average before reducing to one eta_j per
presynaptic neuron.

No filter here ever looks at ground-truth weights — only at properties of
the local gradient itself (sign, region of vj, magnitude), so a filter that
"wins" is winning on physically-motivated grounds, not by fitting to the
answer.

Baseline (top-left panel, unchanged): compute_all_corrected_weights's
current method (global 1D-sweep affine fit of g_phi^2 over each neuron's
own activity range).

Usage:
    /workspace/.conda_envs/neural-graph-linux/bin/python dev_g_phi_w_correction.py \
        --log_dir /groups/saalfeld/home/allierc/GraphData/log/fly/flyvis_noise_005_conductance_cv00
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from connectome_gnn.metrics import (
    W_OUTLIER_THRESH,
    compute_activity_stats,
    compute_all_corrected_weights,
    compute_corrected_weights,
    compute_g_phi_edge_grad,
    compute_grad_msg,
    extract_f_theta_slopes,
    get_model_W,
    sample_g_phi_vi_vj_observed,
)
from connectome_gnn.plot import plot_weight_scatter
from connectome_gnn.utils import set_data_root, to_numpy

from dev_g_phi_vi_vj import load_run


def _group_reduce(j_ids, values, n_neurons, mask=None, weights=None, reducer='mean', pct=(10, 90)):
    """Reduce (E, F) per-edge-per-frame values to one scalar per presynaptic
    neuron j, honoring an optional boolean mask and optional per-point
    weights. Neurons with no surviving points fall back to 1.0 (matches
    extract_g_phi_slopes's convention for degenerate/invalid neurons).

    pct: (lo, hi) percentile band kept by reducer='trimmed_mean' — the rest
    is treated as outliers and dropped before averaging.
    """
    j_flat = np.repeat(j_ids, values.shape[1])       # (E*F,)
    v_flat = values.ravel()
    m_flat = mask.ravel() if mask is not None else np.ones_like(v_flat, dtype=bool)
    w_flat = weights.ravel()[m_flat] if weights is not None else None

    eta = np.ones(n_neurons, dtype=np.float32)
    j_masked = j_flat[m_flat]
    v_masked = v_flat[m_flat]

    if reducer == 'mean':
        sums = np.bincount(j_masked, weights=v_masked, minlength=n_neurons)
        counts = np.bincount(j_masked, minlength=n_neurons)
        has_data = counts > 0
        eta[has_data] = (sums[has_data] / counts[has_data]).astype(np.float32)
        return eta

    # median / trimmed_mean / weighted need per-group access to the raw
    # values (not just a sum) — sort once by j, then split into contiguous
    # per-neuron runs (O(N log N)) instead of an O(N * n_neurons) `==` scan.
    order = np.argsort(j_masked, kind='stable')
    j_sorted = j_masked[order]
    v_sorted = v_masked[order]
    w_sorted = w_flat[order] if w_flat is not None else None

    unique_js, start_idx = np.unique(j_sorted, return_index=True)
    v_groups = np.split(v_sorted, start_idx[1:])
    w_groups = np.split(w_sorted, start_idx[1:]) if w_sorted is not None else [None] * len(unique_js)

    for j, vals, wts in zip(unique_js, v_groups, w_groups):
        if reducer == 'median':
            eta[j] = np.median(vals)
        elif reducer == 'trimmed_mean':
            lo, hi = np.percentile(vals, pct)
            trimmed = vals[(vals >= lo) & (vals <= hi)]
            eta[j] = trimmed.mean() if trimmed.size else vals.mean()
        elif reducer == 'weighted':
            eta[j] = np.average(vals, weights=wts) if wts.sum() > 0 else vals.mean()

    return eta


def build_filters(raw, n_neurons):
    """Data-only filter/aggregation variants. Each returns a (name, eta_j) pair.

    Filter axes (orthogonal, combined where meaningful):
      region:    all vj | vj>0 (active/ReLU region) | far-from-kink (|vj| top 50%)
      slope:     all | positive-slope-only (monotonicity prior, mu_1 in training)
      aggregate: mean | median | trimmed_mean (10-90pct) | weighted by |g_phi|
    """
    j_ids, vj, grad = raw['j_ids'], raw['vj'], raw['grad_vj']
    g_phi_abs = np.abs(raw['g_phi'])

    pos_vj = vj > 0
    pos_slope = grad > 0
    # top-50% |vj| per edge (row-wise), i.e. away from the vj=0 kink
    far_thresh = np.median(np.abs(vj), axis=1, keepdims=True)
    far_kink = np.abs(vj) >= far_thresh

    variants = [
        ('autograd mean, all',                  None,                          None,          'mean'),
        ('autograd median, all',                None,                          None,          'median'),
        ('autograd mean, vj>0',                 pos_vj,                        None,          'mean'),
        ('autograd median, vj>0',               pos_vj,                        None,          'median'),
        ('autograd mean, slope>0',              pos_slope,                     None,          'mean'),
        ('autograd median, slope>0',            pos_slope,                     None,          'median'),
        ('autograd mean, vj>0 & slope>0',       pos_vj & pos_slope,            None,          'mean'),
        ('autograd median, vj>0 & slope>0',     pos_vj & pos_slope,            None,          'median'),
        ('autograd trimmed_mean, all',          None,                          None,          'trimmed_mean'),
        ('autograd trimmed_mean, vj>0',         pos_vj,                        None,          'trimmed_mean'),
        ('autograd weighted|g_phi|, all',       None,                          g_phi_abs,     'weighted'),
        ('autograd weighted|g_phi|, vj>0',      pos_vj,                        g_phi_abs,     'weighted'),
        ('autograd mean, far-from-kink',        far_kink,                      None,          'mean'),
        ('autograd mean, far-kink & slope>0',   far_kink & pos_slope,          None,          'mean'),
        ('autograd mean, vj>0 & far-kink & slope>0',
                                                 pos_vj & far_kink & pos_slope, None,          'mean'),
    ]

    results = []
    for name, mask, weights, reducer in variants:
        eta = _group_reduce(j_ids, grad, n_neurons, mask=mask, weights=weights, reducer=reducer)
        results.append((name, eta))
    return results


def compute_g_phi_fd_slope(model, config, edges, x_ts, n_frames=16, seed=0):
    """Per-edge local slope d(g_phi)/d(vj), estimated by finite differences
    on REAL (vi, vj, g_phi) samples from sample_g_phi_vi_vj_observed —
    instead of the analytic autograd gradient in compute_g_phi_edge_grad.

    For each edge, sort its own sampled frames by vj and difference
    consecutive points: slope = d(g_phi) / d(vj) between neighbors. Output
    shape/keys match compute_g_phi_edge_grad so the same build_filters /
    _group_reduce / montage code downstream is unchanged.
    """
    res = sample_g_phi_vi_vj_observed(model, config, edges, x_ts,
                                      n_edges=edges.shape[1], n_frames=n_frames, seed=seed)
    vj, vi, g_phi = res['vj'], res['vi'], res['g_phi']

    order = np.argsort(vj, axis=1)
    vj_s = np.take_along_axis(vj, order, axis=1)
    vi_s = np.take_along_axis(vi, order, axis=1)
    g_s = np.take_along_axis(g_phi, order, axis=1)

    d_vj = np.diff(vj_s, axis=1)
    d_g = np.diff(g_s, axis=1)
    eps = 1e-6
    slope = d_g / np.where(np.abs(d_vj) < eps, eps, d_vj)

    return {
        'j_ids': res['edge_ij'][:, 1].astype(np.int64),   # j — presynaptic
        'i_ids': res['edge_ij'][:, 0].astype(np.int64),   # i — postsynaptic
        'vj': 0.5 * (vj_s[:, :-1] + vj_s[:, 1:]),
        'vi': 0.5 * (vi_s[:, :-1] + vi_s[:, 1:]),
        'g_phi': 0.5 * (g_s[:, :-1] + g_s[:, 1:]),
        'grad_vj': slope,
    }


def build_filters_observed(raw, n_neurons):
    """Filters for the finite-difference (sample_g_phi_vi_vj_observed) slope
    source. Emphasizes a percentile-outlier-trim sweep: points where the
    local slope deviates strongly from the dominant vj-driven trend are the
    candidate vi-influenced outliers, so trimming them before averaging is a
    data-only (no GT) way to suppress vi's effect on eta_j.
    """
    j_ids, vj, grad = raw['j_ids'], raw['vj'], raw['grad_vj']
    g_phi_abs = np.abs(raw['g_phi'])

    pos_vj = vj > 0
    pos_slope = grad > 0

    variants = [
        ('observed mean, all',                    None,               None,       'mean',        None),
        ('observed median, all',                  None,               None,       'median',      None),
        ('observed trimmed[25,75], all',          None,               None,       'trimmed_mean', (25, 75)),
        ('observed trimmed[10,90], all',          None,               None,       'trimmed_mean', (10, 90)),
        ('observed trimmed[5,95], all',           None,               None,       'trimmed_mean', (5, 95)),
        ('observed trimmed[1,99], all',           None,               None,       'trimmed_mean', (1, 99)),
        ('observed mean, vj>0',                   pos_vj,             None,       'mean',        None),
        ('observed trimmed[25,75], vj>0',         pos_vj,             None,       'trimmed_mean', (25, 75)),
        ('observed trimmed[10,90], vj>0',         pos_vj,             None,       'trimmed_mean', (10, 90)),
        ('observed trimmed[5,95], vj>0',          pos_vj,             None,       'trimmed_mean', (5, 95)),
        ('observed mean, slope>0',                pos_slope,          None,       'mean',        None),
        ('observed trimmed[10,90], slope>0',      pos_slope,          None,       'trimmed_mean', (10, 90)),
        ('observed weighted|g_phi|, all',         None,               g_phi_abs,  'weighted',    None),
        ('observed weighted|g_phi|, vj>0',        pos_vj,             g_phi_abs,  'weighted',    None),
        ('observed trimmed[10,90], vj>0 & slope>0',
                                                   pos_vj & pos_slope, None,       'trimmed_mean', (10, 90)),
    ]

    results = []
    for name, mask, weights, reducer, pct in variants:
        kwargs = {'pct': pct} if pct is not None else {}
        eta = _group_reduce(j_ids, grad, n_neurons, mask=mask, weights=weights, reducer=reducer, **kwargs)
        results.append((name, eta))
    return results


def compute_shared_postsynaptic_gain(model, config, edges, x_ts, n_neurons, device,
                                      n_grad_frames=8, seed=0):
    """slopes_f_theta (s_i) and grad_msg (kappa_i) — shared across every
    g_phi-slope filter variant; only eta_j changes between panels.
    """
    mu_activity, sigma_activity = compute_activity_stats(x_ts, device)
    slopes_f_theta, _ = extract_f_theta_slopes(model, config, n_neurons, mu_activity, sigma_activity, device)
    slopes_f_theta_t = torch.tensor(slopes_f_theta, dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed)
    frame_idx = rng.choice(x_ts.n_frames, size=min(n_grad_frames, x_ts.n_frames), replace=False)
    data_id = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)

    was_training = model.training
    model.eval()
    grad_list = []
    for k in frame_idx:
        state = x_ts.frame(int(k)).to(device)
        with torch.no_grad():
            _, in_features, _ = model(state, edges, data_id=data_id, return_all=True)
        grad_list.append(compute_grad_msg(model, in_features, config))
    if was_training:
        model.train()
    grad_msg = torch.median(torch.stack(grad_list, dim=0), dim=0).values

    return slopes_f_theta_t, grad_msg


def main(args):
    set_data_root(args.data_root)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config, model, edges, x_ts = load_run(args.log_dir, args.config_name, device)
    n_neurons = model.a.shape[0]

    gt_weights = torch.load(os.path.join(args.log_dir, 'gt_weights.pt'),
                            map_location=device, weights_only=False)

    from connectome_gnn.generators.ode_params import get_ode_params_class, FlyVisODEParams
    try:
        OdeParamsCls = get_ode_params_class(config.graph_model.signal_model_name)
    except KeyError:
        OdeParamsCls = FlyVisODEParams
    from connectome_gnn.utils import graphs_data_path
    ode_params_path = graphs_data_path(config.dataset, 'ode_params.pt')
    ode_params = (OdeParamsCls.load(graphs_data_path(config.dataset), device='cpu')
                  if os.path.exists(ode_params_path) else OdeParamsCls())

    gt_w_full = ode_params.effective_true_weights(to_numpy(gt_weights), to_numpy(edges), n_neurons)

    # --- Baseline: current method (unchanged) ---------------------------
    print('computing baseline (current method)...')
    corrected_W_baseline, _, _, _, _ = compute_all_corrected_weights(
        model, config, edges, x_ts, device, ode_params=ode_params)
    baseline_np = to_numpy(corrected_W_baseline.squeeze())

    # --- Shared postsynaptic gain (kappa_i * from f_theta) ---------------
    print('computing shared postsynaptic gain (f_theta side)...')
    slopes_f_theta_t, grad_msg = compute_shared_postsynaptic_gain(
        model, config, edges, x_ts, n_neurons, device, n_grad_frames=8, seed=args.seed)

    # --- Raw per-edge, per-frame local g_phi slope ------------------------
    if args.source == 'autograd':
        print('computing per-edge local d(g_phi)/d(vj) via autograd...')
        raw = compute_g_phi_edge_grad(model, config, edges, x_ts, n_frames=args.n_frames, seed=args.seed)
        filters = build_filters(raw, n_neurons)
    else:
        print('computing per-edge local d(g_phi)/d(vj) via finite differences on observed pairs...')
        raw = compute_g_phi_fd_slope(model, config, edges, x_ts, n_frames=args.n_frames, seed=args.seed)
        filters = build_filters_observed(raw, n_neurons)

    panels = [('current method (1D sweep)', baseline_np)]
    for name, eta_j in filters:
        eta_t = torch.tensor(eta_j, dtype=torch.float32, device=device)
        corrected_W = compute_corrected_weights(model, edges, slopes_f_theta_t, eta_t, grad_msg)
        panels.append((name, to_numpy(corrected_W.squeeze())))

    # --- Montage ------------------------------------------------------
    n_grid = 4
    fig, axes = plt.subplots(n_grid, n_grid, figsize=(6 * n_grid, 6 * n_grid), facecolor='white')
    axes = axes.ravel()
    print(f'{"panel":40s} R2')
    for idx, (name, learned_w) in enumerate(panels[:16]):
        ax = axes[idx]
        r2, slope = plot_weight_scatter(ax, gt_weights=gt_w_full, learned_weights=learned_w,
                                        corrected=True, outlier_threshold=W_OUTLIER_THRESH)
        ax.set_title(name, fontsize=13)
        print(f'{name:40s} {r2:.4f}')
    for idx in range(len(panels), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    out_name = f'weights_comparison_corrected_montage_{args.source}.png'
    out_path = os.path.join(args.log_dir, 'results', out_name)
    plt.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str,
                        default='/groups/saalfeld/home/allierc/GraphData/log/fly/flyvis_noise_005_conductance_cv00')
    parser.add_argument('--config_name', type=str, default='flyvis_noise_005_conductance_cv00')
    parser.add_argument('--data_root', type=str, default='/groups/saalfeld/home/allierc/GraphData')
    parser.add_argument('--source', type=str, default='autograd', choices=['autograd', 'observed'],
                        help='autograd = compute_g_phi_edge_grad (analytic); '
                             'observed = sample_g_phi_vi_vj_observed + finite differences')
    parser.add_argument('--n_frames', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
