"""Pure metrics computation — no matplotlib dependency.

Contains the connectivity R² pipeline (slope correction, grad_msg,
corrected weights) and derived quantities (tau, V_rest).

Used by:
    - plot.py (re-exports for backward compatibility)
    - GNN_PlotFigure.py (post-training analysis)
    - graph_trainer.py (training-time monitoring)
    - sparsify.py (pruning)
"""
from typing import Optional

import numpy as np
import torch
from scipy.optimize import curve_fit

from connectome_gnn.fitting_models import linear_model
from connectome_gnn.utils import graphs_data_path, to_numpy

# ------------------------------------------------------------------ #
#  Neuron type constants
# ------------------------------------------------------------------ #

INDEX_TO_NAME: dict[int, str] = {
    0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)',
    5: 'L1', 6: 'L2', 7: 'L3', 8: 'L4', 9: 'L5',
    10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi10', 14: 'Mi11',
    15: 'Mi12', 16: 'Mi13', 17: 'Mi14', 18: 'Mi15', 19: 'Mi2',
    20: 'Mi3', 21: 'Mi4', 22: 'Mi9', 23: 'R1', 24: 'R2',
    25: 'R3', 26: 'R4', 27: 'R5', 28: 'R6', 29: 'R7', 30: 'R8',
    31: 'T1', 32: 'T2', 33: 'T2a', 34: 'T3', 35: 'T4a',
    36: 'T4b', 37: 'T4c', 38: 'T4d', 39: 'T5a', 40: 'T5b',
    41: 'T5c', 42: 'T5d', 43: 'Tm1', 44: 'Tm16', 45: 'Tm2',
    46: 'Tm20', 47: 'Tm28', 48: 'Tm3', 49: 'Tm30', 50: 'Tm4',
    51: 'Tm5Y', 52: 'Tm5a', 53: 'Tm5b', 54: 'Tm5c', 55: 'Tm9',
    56: 'TmY10', 57: 'TmY13', 58: 'TmY14', 59: 'TmY15',
    60: 'TmY18', 61: 'TmY3', 62: 'TmY4', 63: 'TmY5a', 64: 'TmY9',
}

ANATOMICAL_ORDER: list[Optional[int]] = [
    None, 23, 24, 25, 26, 27, 28, 29, 30,
    5, 6, 7, 8, 9, 10, 11, 12,
    19, 20, 21, 22,
    13, 14, 15, 16, 17, 18,
    43, 45, 48, 50, 44, 46, 47, 49, 51, 52, 53, 54, 55,
    61, 62, 63, 56, 57, 58, 59, 60, 64,
    1, 2, 4, 3,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    0,
]


NAME_TO_INDEX: dict[str, int] = {v: k for k, v in INDEX_TO_NAME.items()}


def _load_identifiability_lists() -> tuple[list[str], list[str]]:
    """Derive IDENTIFIABLE_TYPES and NO_OUTGOING_TYPES from the canonical JSON.

    Source: figures/structural_nullspace_table.json (produced by
    src/connectome_gnn/models/structural_nullspace_table.py — this is the
    authoritative analytical artifact for the opto-recovery experiment,
    since it also provides the per-type null_dim ranking).

    IDENTIFIABLE: cell types with no degenerate (k>=2) groups
                  ⇒ weights recoverable from naturalistic dynamics alone
                  ⇒ negative controls for opto experiment.
    NO_OUTGOING:  cell types that never appear as presynaptic.

    Returns ([], []) if the JSON is missing.
    """
    import json
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(repo_root, 'figures', 'structural_nullspace_table.json'),
        os.path.join(repo_root, 'scripts', 'structural_nullspace_table.json'),  # legacy
    ]
    json_path = next((p for p in candidates if os.path.exists(p)), None)
    if json_path is None:
        return [], []
    data = json.load(open(json_path))
    identifiable = sorted(
        data.get('identifiable_type_names', []),
        key=lambda n: NAME_TO_INDEX.get(n, 999),
    )
    no_outgoing = sorted(
        data.get('no_outgoing_type_names', []),
        key=lambda n: NAME_TO_INDEX.get(n, 999),
    )
    return identifiable, no_outgoing


IDENTIFIABLE_TYPES, NO_OUTGOING_TYPES = _load_identifiability_lists()

# Hierarchical groups — mirror of group_by_direction_and_function in
# generators/flyvis_ode.py. Kept for the Gal4-driver-line UX (pan-cell-type,
# all-columns drive). NB: group-only targeting cannot break the columnar
# sum-zero kernel — always pair with column_distinct=True in OptoTargetSpec.
GROUP_NAMES: dict[int, str] = {
    0: 'photoreceptors_outer',  # R1-R6
    1: 'photoreceptors_inner',  # R7-R8
    2: 'lamina_monopolar',      # L1-L5
    3: 'lamina_interneurons',   # Am, C2, C3
    4: 'medulla_Mi_early',      # Mi1-Mi4
    5: 'medulla_Mi_mid',        # Mi9-Mi12
    6: 'medulla_Mi_late',       # Mi13-Mi15
    7: 'medulla_Tm_early',      # Tm1-Tm4
    8: 'medulla_Tm5',           # Tm5*
    9: 'medulla_Tm_mid',        # Tm9, Tm16, Tm20
    10: 'medulla_Tm_late',      # Tm28, Tm30
    11: 'medulla_TmY',          # TmY*
    12: 'T4a', 13: 'T4b', 14: 'T4c', 15: 'T4d',
    16: 'T5_OFF',               # T5a-T5d
    17: 'tangential',           # T1, T2, T2a, T3
    18: 'wide_field_Lawf',
    19: 'other_CT1',
}

GROUP_AGGREGATES: dict[str, list[str]] = {
    'all_photoreceptors':  ['photoreceptors_outer', 'photoreceptors_inner'],
    'all_lamina':          ['lamina_monopolar', 'lamina_interneurons'],
    'all_medulla':         ['medulla_Mi_early', 'medulla_Mi_mid', 'medulla_Mi_late',
                            'medulla_Tm_early', 'medulla_Tm5', 'medulla_Tm_mid',
                            'medulla_Tm_late', 'medulla_TmY'],
    'all_T4':              ['T4a', 'T4b', 'T4c', 'T4d'],
    'all_T5':              ['T5_OFF'],
    'all_T4_T5':           ['T4a', 'T4b', 'T4c', 'T4d', 'T5_OFF'],
}


def _build_group_to_types() -> dict[str, list[str]]:
    """Invert group_by_direction_and_function over all known cell type names."""
    from connectome_gnn.generators.flyvis_ode import group_by_direction_and_function
    out: dict[str, list[str]] = {name: [] for name in GROUP_NAMES.values()}
    for name in INDEX_TO_NAME.values():
        gid = group_by_direction_and_function(name)
        out[GROUP_NAMES[gid]].append(name)
    return out


_GROUP_TO_TYPES: Optional[dict[str, list[str]]] = None


def _group_table() -> dict[str, list[str]]:
    global _GROUP_TO_TYPES
    if _GROUP_TO_TYPES is None:
        _GROUP_TO_TYPES = _build_group_to_types()
    return _GROUP_TO_TYPES


def group_to_neuron_types(group_name: str) -> list[str]:
    """Resolve a group name (or aggregate) to a flat list of cell type names."""
    table = _group_table()
    if group_name in table:
        return list(table[group_name])
    if group_name in GROUP_AGGREGATES:
        out: list[str] = []
        for sub in GROUP_AGGREGATES[group_name]:
            out.extend(table[sub])
        return out
    raise KeyError(
        f"unknown group '{group_name}' — known groups: "
        f"{sorted(set(GROUP_NAMES.values()) | set(GROUP_AGGREGATES.keys()))}"
    )


def name_to_neuron_ids(neuron_type: torch.Tensor, names: list[str]) -> torch.Tensor:
    """Bool mask over neurons whose integer type ∈ {NAME_TO_INDEX[n] for n in names}."""
    type_ids = torch.tensor(
        [NAME_TO_INDEX[n] for n in names],
        device=neuron_type.device, dtype=neuron_type.dtype,
    )
    return torch.isin(neuron_type, type_ids)


def neuron_type_names(neuron_type: torch.Tensor) -> list[str]:
    """Per-neuron type-name list, length N. Inverse of name_to_neuron_ids."""
    ids = neuron_type.detach().cpu().tolist()
    return [INDEX_TO_NAME[i] for i in ids]


def neuron_column_ids(pos: torch.Tensor) -> torch.Tensor:
    """(N,) long: which retinotopic column each neuron belongs to.

    Columns are unique (x, y) photoreceptor positions; every non-photoreceptor
    neuron inherits its column from its retinotopic neighborhood. The mapping
    is constructed by finding distinct positions and assigning an index per
    distinct position. extent=8 → 217 columns, extent=15 → 721.
    """
    pos_cpu = pos.detach().cpu()
    keys = (pos_cpu * 1e4).round().long()
    flat = keys[:, 0] * 100000 + keys[:, 1]
    uniq, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    return inverse.to(pos.device, dtype=torch.long)


def summarize_targets(state, mask: torch.Tensor) -> dict[str, tuple[int, int, float]]:
    """{type_name: (n_targeted, n_total_of_type, fraction)} for every type with
    nonzero targeting. Used to log opto coverage at generation time."""
    out: dict[str, tuple[int, int, float]] = {}
    nt = state.neuron_type
    for type_id in torch.unique(nt[mask]).tolist():
        name = INDEX_TO_NAME[int(type_id)]
        n_total = int((nt == type_id).sum())
        n_target = int(((nt == type_id) & mask).sum())
        out[name] = (n_target, n_total, n_target / n_total if n_total else 0.0)
    return out


def fingerprint_dataset(state) -> str:
    """Stable sha256 over (n_neurons, neuron_type bytes). Used by
    OptoTargetSpec.dataset_fingerprint to guard explicit_indices targets
    against silent ID drift across connectome variants."""
    import hashlib
    nt = state.neuron_type.detach().cpu().to(torch.int32).numpy().tobytes()
    h = hashlib.sha256()
    h.update(int(state.n_neurons).to_bytes(8, 'little'))
    h.update(nt)
    return h.hexdigest()


def load_nullspace_ranking(
    json_path: str = "figures/structural_nullspace_table.json",
    metric: str = "null_dim",
) -> list[tuple[str, float, float]]:
    """Load structural-nullspace artifact and return (name, score, lambda_max)
    sorted descending by `metric`. `lambda_max` defaults to NaN if absent
    (older JSONs predate the lambda_max instrumentation pass).

    Source: produced by src/connectome_gnn/models/structural_nullspace_table.py;
    written to figures/structural_nullspace_table.json.
    """
    import json
    import math
    import os
    candidates = [json_path]
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates.append(os.path.join(repo_root, json_path))
    # legacy location
    candidates.append(os.path.join(repo_root, "scripts", "structural_nullspace_table.json"))
    chosen = next((p for p in candidates if os.path.exists(p)), None)
    if chosen is None:
        raise FileNotFoundError(
            f"nullspace JSON not found at {json_path}; run "
            f"connectome_gnn/models/structural_nullspace_table.py to produce it."
        )
    data = json.load(open(chosen))
    type_results = data.get('type_results', {})
    out: list[tuple[str, float, float]] = []
    for entry in type_results.values():
        name = entry['name']
        nd = float(entry.get('null_dim', 0.0))
        lam = float(entry.get('lambda_max', math.nan))
        if metric == 'null_dim':
            score = nd
        elif metric == 'leverage':
            n_c = float(entry.get('n_neurons_of_type', entry.get('n_c', 1.0)) or 1.0)
            r2 = entry.get('conn_r2', entry.get('R2_W', 1.0))
            r2 = float(r2) if r2 is not None else 1.0
            score = (nd / max(n_c, 1.0)) * (1.0 - r2)
        else:
            raise ValueError(f"unknown metric '{metric}' (expected 'null_dim' or 'leverage')")
        out.append((name, score, lam))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def validate_registry() -> None:
    """Internal-consistency check, called on import. Raises AssertionError on drift."""
    from connectome_gnn.generators.flyvis_ode import group_by_direction_and_function
    assert len(INDEX_TO_NAME) == 65, f"INDEX_TO_NAME has {len(INDEX_TO_NAME)} entries, expected 65"
    assert sorted(INDEX_TO_NAME.keys()) == list(range(65)), "INDEX_TO_NAME ids not contiguous 0..64"
    for i, n in INDEX_TO_NAME.items():
        assert NAME_TO_INDEX[n] == i, f"NAME_TO_INDEX inverse mismatch at {n}"
    for n in IDENTIFIABLE_TYPES + NO_OUTGOING_TYPES:
        assert n in NAME_TO_INDEX, f"unknown type name in registry constants: {n}"
    assert set(IDENTIFIABLE_TYPES).isdisjoint(set(NO_OUTGOING_TYPES))
    assert len(GROUP_NAMES) == 20
    table = _group_table()
    assert set(table.keys()) == set(GROUP_NAMES.values())
    for agg, subs in GROUP_AGGREGATES.items():
        for s in subs:
            assert s in table, f"GROUP_AGGREGATES[{agg!r}] references unknown group {s!r}"
    # round trip: every type name resolves into exactly one group, and that
    # group's expansion contains the type
    for name in INDEX_TO_NAME.values():
        gid = group_by_direction_and_function(name)
        gname = GROUP_NAMES[gid]
        assert name in table[gname], f"round-trip failed for {name} (group {gname})"


validate_registry()


# ------------------------------------------------------------------ #
#  Weight extraction
# ------------------------------------------------------------------ #

def get_model_W(model) -> torch.Tensor:
    """Get the weight matrix from a model, handling low-rank factorization."""
    # Prefer the effective weight (|W|·sign_GT under the hard sign-lock); when
    # the lock is off this equals the raw W, so existing models are unaffected.
    if hasattr(model, 'effective_W'):
        return model.effective_W
    if hasattr(model, 'W'):
        return model.W
    elif hasattr(model, 'WL') and hasattr(model, 'WR'):
        return model.WL @ model.WR
    else:
        raise AttributeError("Model has neither 'W' nor 'WL'/'WR' attributes")


# ------------------------------------------------------------------ #
#  R² computation
# ------------------------------------------------------------------ #


def compute_r_squared_lin_fit(true: np.ndarray, learned: np.ndarray) -> tuple[float, float]:
    """Compute R² and linear fit slope between true and learned arrays."""
    lin_fit, _ = curve_fit(linear_model, true, learned)
    residuals = learned - linear_model(true, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((learned - np.mean(learned)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return r_squared, lin_fit[0]


def compute_r_squared_identity_line(true: np.ndarray, learned: np.ndarray) -> tuple[float, float]:
    """Identity-line R² plus calibration-fit slope (diagnostic).

    R² = 1 − var(true − learned) / var(true) — measures how close `learned`
    is to `true` on the same scale (no fit, ideal when slope=1, intercept=0).
    Slope is the `a` of `true ≈ a·learned + b` from np.polyfit, returned for
    diagnostic use (plot annotations, metrics.txt, JSON summaries) so callers
    that consume the second tuple element still see the real number rather
    than a hardcoded constant. Returns (nan, nan) on failure."""
    try:
        var_true = float(np.var(true))
        var_unexpl = float(np.var(true - learned))
        r_squared = 1.0 - var_unexpl / var_true if var_true > 0 else float('nan')
        try:
            slope = float(np.polyfit(true, learned, 1)[0])
        except Exception:
            slope = float('nan')
        return r_squared, slope
    except Exception:
        return float('nan'), float('nan')
    

def compute_r_squared_NSE(true: np.ndarray, learned: np.ndarray) -> tuple[float, float]:
    """Identity-line R² (Nash-Sutcliffe efficiency) and calibration slope.

    R² = 1 - mean((true - learned)²) / var(true)
    Penalizes both noise and scale/bias errors. Range: (-inf, 1].
    Slope from learned ≈ a·true + b diagnoses scale miscalibration when R² is low.
    """
    try:
        a = np.asarray(true).ravel()
        # Effectively-constant GT -> identity-line R² is undefined and would
        # explode to a huge negative number on tiny float-level variation
        # (e.g. a tau that is 0.1 up to ~1e-6 noise, var≈1e-12). Return NaN so
        # callers show 'N/A' + a MAE instead (see is_degenerate_gt/recovery_mae).
        scale = max(float(np.mean(np.abs(a))), 1e-12)
        if a.size < 2 or float(np.std(a)) / scale < 1e-4:
            return float('nan'), float('nan')
        var_true = float(np.var(true))
        mse = float(np.mean((true - learned) ** 2))
        r_squared = 1.0 - mse / var_true
        slope = float(np.polyfit(true, learned, 1)[0])
        return r_squared, slope
    except Exception:
        return float('nan'), float('nan')


def compute_r_squared_filtered(true: np.ndarray, learned: np.ndarray, outlier_threshold: float = 5.0) -> tuple[float, float, np.ndarray]:
    """Compute identity-line R² with outlier removal + diagnostic slope.

    Removes points where |learned - true| > outlier_threshold, then computes
    R² and slope on the inliers via compute_r_squared (slope from polyfit).

    Returns:
        r_squared: float.
        slope: float (np.polyfit on inliers).
        inlier_mask: (N,) bool array — True for inliers.
    """
    residuals = learned - true
    mask = np.abs(residuals) <= outlier_threshold
    true_in = true[mask]
    learned_in = learned[mask]

    r_squared, slope = compute_r_squared_NSE(true_in, learned_in)
    return r_squared, slope, mask


def is_degenerate_gt(true: np.ndarray, rel_eps: float = 1e-4) -> bool:
    """True when the GT is effectively constant, so the identity-line (NSE) R²
    is undefined (``1 - rss/var`` with ``var≈0``) and explodes to a huge negative
    number on tiny float-level variation. Uses a *relative* test (coefficient of
    variation ``std/|mean| < rel_eps``) so it is scale-robust — e.g. a tau that is
    0.1 up to ~1e-6 float noise (var≈1e-12) is correctly flagged constant. For
    such a parameter report a MAE instead of a garbage R²."""
    a = np.asarray(true).ravel()
    if a.size < 2:
        return True
    scale = max(float(np.mean(np.abs(a))), 1e-12)
    return float(np.std(a)) / scale < rel_eps


def recovery_mae(true: np.ndarray, learned: np.ndarray) -> float:
    """Mean absolute error |true - learned| — the fallback metric for a recovered
    parameter whose GT is constant (R² undefined)."""
    a = np.asarray(true).ravel()
    b = np.asarray(learned).ravel()
    n = min(a.size, b.size)
    return float(np.mean(np.abs(a[:n] - b[:n]))) if n else float('nan')


def r2_scatter_text(true: np.ndarray, learned: np.ndarray, clean_r2: float = None,
                    label: str = 'R²', n: int = None) -> str:
    """Annotation text for a recovery scatter (tau, V_rest, ...).

    Normal: ``'R²: 0.83\\nslope: 1.02'`` (or ``'R²: clean (all)\\nslope'`` when
    ``clean_r2`` is given). When the GT has ~no variance the R² is undefined, so
    it shows ``'R²: N/A (const GT)\\nMAE: 0.012'`` instead. Optional ``n`` appends
    a sample-count line."""
    tail = '' if n is None else f'\nN: {n}'
    if is_degenerate_gt(true):
        return f'{label}: N/A (const GT)\nMAE: {recovery_mae(true, learned):.3g}{tail}'
    r2, slope = compute_r_squared_NSE(true, learned)
    if clean_r2 is not None:
        return f'{label}: {clean_r2:.2f} ({r2:.2f})\nslope: {slope:.2f}{tail}'
    return f'{label}: {r2:.2f}\nslope: {slope:.2f}{tail}'


def fmt_r2_bar(val) -> str:
    """Progress-bar value: ``'N/A'`` when the R² is undefined (None/NaN, e.g. a
    constant-GT parameter), else 3-dp."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 'N/A'
    return f'{val:.3f}'


def recovery_param_metrics(gt: np.ndarray, learned: np.ndarray, outlier_thresh: float) -> dict:
    """All recovery metrics for one per-neuron parameter (tau, V_rest), computed
    ONCE so the scatter, console line and metrics.txt can't disagree.

    Outlier rule: ``|learned - true| > outlier_thresh`` (the neurips.tex
    eq:outlier_threshold band, delta_tau=0.1 / delta_Vrest=0.2).

    Returns a dict with: ``r2``/``slope`` (full identity-line NSE),
    ``r2_clean``/``slope_clean`` (inliers only; NaN if <2 inliers),
    ``n_outliers``/``n_total``/``pct_outliers``, ``outlier_mask``/``inlier_mask``,
    ``degenerate`` (bool, GT ~constant -> R² undefined), ``mae`` (fallback metric),
    and ``rel_err_median``/``rel_err_iqr`` (|learned-true|/max(|true|,1e-6))."""
    gt = np.asarray(gt).ravel()
    learned = np.asarray(learned).ravel()
    n = min(gt.size, learned.size)
    gt, learned = gt[:n], learned[:n]
    r2, slope = compute_r_squared_NSE(gt, learned)
    out_mask = np.abs(learned - gt) > outlier_thresh
    in_mask = ~out_mask
    n_out = int(out_mask.sum())
    n_tot = int(gt.size)
    if int(in_mask.sum()) >= 2:
        r2_clean, slope_clean = compute_r_squared_NSE(gt[in_mask], learned[in_mask])
    else:
        r2_clean, slope_clean = float('nan'), float('nan')
    if n_tot:
        rel = np.abs(learned - gt) / np.maximum(np.abs(gt), 1e-6)
        rel_med = float(np.median(rel))
        q1, q3 = np.percentile(rel, [25.0, 75.0])
        rel_iqr = float(q3 - q1)
    else:
        rel_med = rel_iqr = float('nan')
    return dict(
        r2=r2, slope=slope, r2_clean=r2_clean, slope_clean=slope_clean,
        n_outliers=n_out, n_total=n_tot,
        pct_outliers=(100.0 * n_out / n_tot) if n_tot else 0.0,
        outlier_mask=out_mask, inlier_mask=in_mask,
        degenerate=is_degenerate_gt(gt), mae=recovery_mae(gt, learned),
        rel_err_median=rel_med, rel_err_iqr=rel_iqr,
    )


# ------------------------------------------------------------------ #
#  Vectorized helpers
# ------------------------------------------------------------------ #

def _vectorized_linspace(starts: np.ndarray, ends: np.ndarray, n_pts: int, device: torch.device) -> torch.Tensor:
    """Create (N, n_pts) tensor where row n spans [starts[n], ends[n]].

    Instead of calling torch.linspace N times, we parameterize with
    t in [0, 1] and broadcast:  rr[n, i] = start[n] + t[i] * (end[n] - start[n])
    """
    t = torch.linspace(0, 1, n_pts, device=device)                   # (n_pts,)
    starts_t = torch.as_tensor(starts, dtype=torch.float32, device=device)  # (N,)
    ends_t = torch.as_tensor(ends, dtype=torch.float32, device=device)      # (N,)
    return starts_t[:, None] + t[None, :] * (ends_t - starts_t)[:, None]    # (N, n_pts)


def _batched_mlp_eval(mlp, model_a, rr, build_features_fn,
                      device, chunk_size=2000, post_fn=None):
    """Evaluate an MLP for all neurons at once, in chunks.

    Instead of N individual forward passes on (1000, D) inputs, we
    stack all neurons into (N*1000, D) and run one pass per chunk.

    Args:
        mlp: nn.Module — the MLP to evaluate (model.g_phi or model.f_theta).
        model_a: (N, emb_dim) embedding tensor.
        rr: (N, n_pts) tensor of input values per neuron.
        build_features_fn: callable(rr_flat, emb_flat) -> (chunk*n_pts, D)
            Builds the MLP input features from flattened rr and embeddings.
        device: torch device.
        chunk_size: number of neurons per chunk (limits GPU memory).
        post_fn: optional callable applied to MLP output (e.g. lambda x: x**2).

    Returns:
        (N, n_pts) tensor of MLP outputs.
    """
    N, n_pts = rr.shape
    emb_dim = model_a.shape[1]
    results = []

    for i in range(0, N, chunk_size):
        chunk_rr = rr[i:i + chunk_size]                        # (C, n_pts)
        chunk_a = model_a[i:i + chunk_size]                     # (C, emb_dim)
        C = chunk_rr.shape[0]

        # Flatten: repeat each neuron's values n_pts times
        rr_flat = chunk_rr.reshape(-1, 1)                       # (C*n_pts, 1)
        emb_flat = chunk_a[:, None, :].expand(-1, n_pts, -1)    # (C, n_pts, emb_dim)
        emb_flat = emb_flat.reshape(-1, emb_dim)                 # (C*n_pts, emb_dim)

        in_features = build_features_fn(rr_flat, emb_flat)       # (C*n_pts, D)

        with torch.no_grad():
            out = mlp(in_features.float())                       # (C*n_pts, 1)
            if post_fn is not None:
                out = post_fn(out)

        results.append(out.squeeze(-1).reshape(C, n_pts))        # (C, n_pts)

    return torch.cat(results, dim=0)                              # (N, n_pts)


def _vectorized_linear_fit(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized least-squares linear regression across rows.

    Fits y[n] = slope[n] * x[n] + offset[n] for all N rows in parallel,
    replacing N individual scipy.curve_fit calls.

    Uses the closed-form solution:
        slope  = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
        offset = (Σy − slope·Σx) / n

    Args:
        x: (N, n_pts) numpy array or tensor.
        y: (N, n_pts) numpy array or tensor.

    Returns:
        slopes: (N,) numpy array.
        offsets: (N,) numpy array.
    """
    if isinstance(x, torch.Tensor):
        x = to_numpy(x)
    if isinstance(y, torch.Tensor):
        y = to_numpy(y)

    n_pts = x.shape[1]
    sx = x.sum(axis=1)
    sy = y.sum(axis=1)
    sxy = (x * y).sum(axis=1)
    sxx = (x * x).sum(axis=1)

    denom = n_pts * sxx - sx * sx
    # Guard against degenerate cases (constant x)
    safe = np.abs(denom) > 1e-12
    slopes = np.where(safe, (n_pts * sxy - sx * sy) / np.where(safe, denom, 1.0), 0.0)
    offsets = np.where(safe, (sy - slopes * sx) / n_pts, 0.0)

    return slopes, offsets


# ------------------------------------------------------------------ #
#  Feature-building helpers for the two MLPs
# ------------------------------------------------------------------ #

def _build_g_phi_features(rr_flat, emb_flat, signal_model_name):
    """Build input features for g_phi MLP."""
    if 'flyvis_B' in signal_model_name:
        return torch.cat([rr_flat * 0, rr_flat, emb_flat, emb_flat], dim=1)
    else:
        return torch.cat([rr_flat, emb_flat], dim=1)


def _build_f_theta_features(rr_flat, emb_flat):
    """Build input features for f_theta MLP: (v, embedding, msg=0, exc=0)."""
    zeros = torch.zeros_like(rr_flat)
    return torch.cat([rr_flat, emb_flat, zeros, zeros], dim=1)


# ------------------------------------------------------------------ #
#  Activity statistics
# ------------------------------------------------------------------ #

def compute_activity_stats(x_ts, device: Optional[torch.device] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-neuron mean and std of voltage activity.

    Args:
        x_ts: NeuronTimeSeries (voltage field is (T, N) tensor).
        device: optional device override.

    Returns:
        mu_activity: (N,) tensor of per-neuron mean voltage.
        sigma_activity: (N,) tensor of per-neuron std voltage.
    """
    voltage = x_ts.voltage  # (T, N), already on device if x_ts was moved
    if device is None or voltage.device == device:
        return voltage.mean(dim=0), voltage.std(dim=0)
    # Avoid OOM when (T, N) is large (e.g. e15 hybrids: T~64k, N~50k → ~12 GiB)
    # AND the target device is full (e.g. CUDA Graphs in GNN training reserve
    # ~38 GiB during plot_training_flyvis). Reduce on CPU in float64 with small
    # chunks; only the small (N,) result tensors are moved to device at the end.
    T, N = voltage.shape
    # ~256 MiB float32 chunks
    chunk = max(1, int(2.5e8 // (4 * max(N, 1))))
    s1 = torch.zeros(N, dtype=torch.float64, device='cpu')
    s2 = torch.zeros(N, dtype=torch.float64, device='cpu')
    for i in range(0, T, chunk):
        v = voltage[i:i + chunk]
        if v.is_cuda:
            v = v.cpu()
        v = v.to(torch.float64)
        s1.add_(v.sum(dim=0))
        s2.add_((v * v).sum(dim=0))
        del v
    mu = (s1 / T).to(torch.float32).to(device)
    var = (s2 / T - (s1 / T) ** 2).clamp_min_(0.0)
    sigma = var.sqrt().to(torch.float32).to(device)
    return mu, sigma


# ------------------------------------------------------------------ #
#  Slope extraction
# ------------------------------------------------------------------ #

def evaluate_g_phi_curves(model, config, n_neurons, mu_activity, sigma_activity, device):
    """Evaluate learned g_phi curves over each neuron's activity range.

    Returns:
        v_ranges: (N, n_pts) numpy array of voltage grids per neuron.
        curves: (N, n_pts) numpy array of g_phi outputs.
        valid: (N,) bool array — neurons with positive activity range.
    """
    signal_model_name = config.graph_model.signal_model_name
    g_phi_positive = config.graph_model.g_phi_positive
    n_pts = 1000

    mu = to_numpy(mu_activity).astype(np.float32) if torch.is_tensor(mu_activity) else np.asarray(mu_activity, dtype=np.float32)
    sigma = to_numpy(sigma_activity).astype(np.float32) if torch.is_tensor(sigma_activity) else np.asarray(sigma_activity, dtype=np.float32)

    valid = (mu + sigma) > 0
    starts = np.maximum(mu - 2 * sigma, 0.0)
    ends = mu + 2 * sigma
    starts[~valid] = 0.0
    ends[~valid] = 1.0

    rr = _vectorized_linspace(starts, ends, n_pts, device)  # (N, n_pts)

    post_fn = (lambda x: x ** 2) if g_phi_positive else None
    build_fn = lambda rr_f, emb_f: _build_g_phi_features(rr_f, emb_f, signal_model_name)

    func = _batched_mlp_eval(model.g_phi, model.a[:n_neurons], rr,
                             build_fn, device, post_fn=post_fn)  # (N, n_pts)

    return to_numpy(rr), to_numpy(func), valid


def extract_g_phi_slopes(model, config, n_neurons, mu_activity, sigma_activity, device):
    """Extract linear slope of g_phi for each neuron j (vectorized).

    Returns:
        slopes: (n_neurons,) numpy array of g_phi slopes.
    """
    v_ranges, curves, valid = evaluate_g_phi_curves(
        model, config, n_neurons, mu_activity, sigma_activity, device)

    rr_t = torch.tensor(v_ranges, dtype=torch.float32)
    func_t = torch.tensor(curves, dtype=torch.float32)
    slopes, _ = _vectorized_linear_fit(rr_t, func_t)

    slopes[~valid] = 1.0
    return slopes


def extract_f_theta_slopes(model, config, n_neurons, mu_activity, sigma_activity, device):
    """Extract linear slope and offset of f_theta for each neuron i (vectorized).

    Evaluates f_theta(a_i, v_i, msg=0, exc=0) over each neuron's activity range
    in one batched forward pass, then fits all slopes/offsets with vectorized regression.

    Returns:
        slopes: (n_neurons,) numpy array — slope relates to 1/tau.
        offsets: (n_neurons,) numpy array — offset relates to V_rest.
    """
    n_pts = 1000
    mu = to_numpy(mu_activity).astype(np.float32) if torch.is_tensor(mu_activity) else np.asarray(mu_activity, dtype=np.float32)
    sigma = to_numpy(sigma_activity).astype(np.float32) if torch.is_tensor(sigma_activity) else np.asarray(sigma_activity, dtype=np.float32)

    starts = mu - 2 * sigma
    ends = mu + 2 * sigma

    rr = _vectorized_linspace(starts, ends, n_pts, device)  # (N, n_pts)

    func = _batched_mlp_eval(model.f_theta, model.a[:n_neurons], rr,
                             lambda rr_f, emb_f: _build_f_theta_features(rr_f, emb_f),
                             device)  # (N, n_pts)

    slopes, offsets = _vectorized_linear_fit(rr, func)

    return slopes, offsets


# ------------------------------------------------------------------ #
#  Derived quantities from f_theta slopes
# ------------------------------------------------------------------ #

def derive_tau(slopes_f_theta: np.ndarray, n_neurons: int) -> np.ndarray:
    """Convert f_theta slopes to learned tau: tau = 1/(-slope), clipped to [0,1].

    Args:
        slopes_f_theta: (N,) numpy array of f_theta slopes.
        n_neurons: number of neurons to use.

    Returns:
        learned_tau: (n_neurons,) numpy array.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        learned_tau = np.where(slopes_f_theta != 0, 1.0 / -slopes_f_theta, 1.0)[:n_neurons]
    return np.clip(learned_tau, 0, 1)


def derive_vrest(slopes_f_theta: np.ndarray, offsets_f_theta: np.ndarray, n_neurons: int) -> np.ndarray:
    """Convert f_theta slopes/offsets to learned V_rest: V_rest = -offset/slope.

    Args:
        slopes_f_theta: (N,) numpy array of f_theta slopes.
        offsets_f_theta: (N,) numpy array of f_theta offsets.
        n_neurons: number of neurons to use.

    Returns:
        learned_V_rest: (n_neurons,) numpy array.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(slopes_f_theta != 0, -offsets_f_theta / slopes_f_theta, 1.0)[:n_neurons]


def _torch_linear_fit(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable least-squares linear regression in pure torch.

    Same closed-form OLS as _vectorized_linear_fit, but operates on
    torch tensors with gradient tracking preserved through y.

    Args:
        x: (N, n_pts) tensor (no grad needed — voltage grid points).
        y: (N, n_pts) tensor (grad flows through here from f_theta).

    Returns:
        slopes: (N,) tensor.
        offsets: (N,) tensor.
    """
    n_pts = x.shape[1]
    sx = x.sum(dim=1)
    sy = y.sum(dim=1)
    sxy = (x * y).sum(dim=1)
    sxx = (x * x).sum(dim=1)

    denom = n_pts * sxx - sx * sx
    slopes = (n_pts * sxy - sx * sy) / (denom + 1e-12)
    offsets = (sy - slopes * sx) / n_pts

    return slopes, offsets


def compute_f_theta_linearity_loss(model, n_neurons: int, mu: np.ndarray, sigma: np.ndarray, device: torch.device, n_pts: int = 200) -> torch.Tensor:
    """Unsupervised f_theta linearity loss.

    Evaluates f_theta WITH gradient tracking, fits a differentiable OLS
    line through the outputs, and penalizes the residual (non-linear
    component). No ground-truth V_rest is needed.

    Physical motivation: the true neuron dynamics are leaky integrators
    (dv/dt = -(v - V_rest)/tau), which is linear in v. Penalizing
    f_theta's deviation from linearity is an inductive bias toward the
    correct physics, constraining the space of solutions so that
    V_rest = -offset/slope is more uniquely determined.

    Gradients flow through f_theta parameters only:
    - model.a (embeddings) is detached
    - rr (voltage grid) is constructed from cached data stats (no grad)

    Args:
        model: NeuralGNN model with f_theta and a attributes.
        n_neurons: Number of neurons.
        mu: (N,) numpy array — per-neuron mean voltage.
        sigma: (N,) numpy array — per-neuron std voltage.
        device: Torch device.
        n_pts: Number of voltage grid points (default 200).

    Returns:
        Scalar mean-squared residual loss with gradient through f_theta.
    """
    starts = mu - 2 * sigma
    ends = mu + 2 * sigma

    rr = _vectorized_linspace(starts, ends, n_pts, device)  # (N, n_pts), no grad

    # Evaluate f_theta WITHOUT no_grad — gradient flows through f_theta weights
    emb_dim = model.a.shape[1]
    rr_flat = rr.reshape(-1, 1)                                          # (N*n_pts, 1)
    a_detached = model.a[:n_neurons].detach()                             # block grad to embeddings
    emb_flat = a_detached[:, None, :].expand(-1, n_pts, -1).reshape(-1, emb_dim)  # (N*n_pts, emb_dim)

    in_features = _build_f_theta_features(rr_flat, emb_flat)             # (N*n_pts, D)
    out = model.f_theta(in_features.float())                             # (N*n_pts, 1)
    func = out.squeeze(-1).reshape(n_neurons, n_pts)                     # (N, n_pts)

    # Differentiable OLS: fit a line through f_theta outputs
    slopes, offsets = _torch_linear_fit(rr, func)

    # Linear prediction: what f_theta WOULD output if it were perfectly linear
    linear_pred = slopes[:, None] * rr + offsets[:, None]                # (N, n_pts)

    # Residual: the non-linear component of f_theta
    residual = func - linear_pred                                        # (N, n_pts)

    # Mean squared residual across all neurons and points
    loss = (residual ** 2).mean()

    return loss


def compute_f_theta_centering_loss(
    model,
    n_neurons: int,
    mu: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Unsupervised f_theta centering loss — anchors V_rest toward mean voltage.

    Evaluates f_theta at (v=μ_i, a_i, msg=0, exc=0) for each neuron and
    penalizes the output magnitude. If f_theta is approximately linear
    (dv/dt ≈ -(v - V_rest)/tau), then f_theta(μ) = (V_rest - μ)/tau.
    Penalizing this pulls V_rest toward μ (the observed mean voltage),
    providing an unsupervised anchor for the zero-crossing location.

    Unlike the linearity loss (which constrains f_theta's *shape*),
    this constrains f_theta's *location* — where the zero-crossing falls.

    Cost: N f_theta evaluations (trivial — no voltage grid needed).

    Args:
        model: NeuralGNN model with f_theta and a attributes.
        n_neurons: Number of neurons.
        mu: (N,) numpy array — per-neuron mean voltage.
        device: Torch device.

    Returns:
        Scalar MSE loss with gradient through f_theta.
    """
    mu_t = torch.tensor(mu[:n_neurons], dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)

    emb = model.a[:n_neurons].detach()                # (N, emb_dim) — block grad to embeddings
    zeros = torch.zeros(n_neurons, 1, device=device)  # msg=0, exc=0

    in_features = torch.cat([mu_t, emb, zeros, zeros], dim=1)  # (N, 1+emb_dim+1+1)
    out = model.f_theta(in_features.float())                    # (N, 1)

    # MSE: penalize f_theta output at mean voltage
    loss = (out ** 2).mean()

    return loss


# ------------------------------------------------------------------ #
#  Dynamics R² (V_rest and tau)
# ------------------------------------------------------------------ #

# Outlier thresholds — must match GNN_PlotFigure.py (lines 422 and 539) so
# the live training metrics agree with the post-training data_plot summary.
TAU_OUTLIER_THRESH = 0.1
VREST_OUTLIER_THRESH = 0.2


def _r2_with_outliers(gt, learned, thresh):
    """Return (r2_all, r2_clean, n_out, n_total) for one parameter array.

    Outlier mask: |learned - gt| > thresh (same rule as data_plot).
    r2_all   : R² over every neuron.
    r2_clean : R² over inliers only (NaN if <2 inliers).
    """
    gt_arr  = np.asarray(gt).ravel()
    lrn_arr = np.asarray(learned).ravel()
    n_total = int(gt_arr.size)
    if n_total == 0:
        return 0.0, float('nan'), 0, 0

    try:
        r2_all, _ = compute_r_squared_NSE(gt_arr, lrn_arr)
    except Exception:
        r2_all = 0.0

    out_mask = np.abs(lrn_arr - gt_arr) > thresh
    n_out = int(out_mask.sum())
    inl_mask = ~out_mask
    if int(inl_mask.sum()) >= 2:
        try:
            r2_clean, _ = compute_r_squared_NSE(gt_arr[inl_mask], lrn_arr[inl_mask])
        except Exception:
            r2_clean = float('nan')
    else:
        r2_clean = float('nan')
    return float(r2_all), float(r2_clean), n_out, n_total


_DYNAMICS_R2_EMPTY = {
    'vrest_r2': 0.0, 'vrest_r2_clean': float('nan'),
    'n_out_vrest': 0, 'n_total_vrest': 0,
    'tau_r2':   0.0, 'tau_r2_clean':   float('nan'),
    'n_out_tau':   0, 'n_total_tau':   0,
}


def compute_dynamics_r2(model, x_ts, config, device, n_neurons):
    """Compute V_rest R² and tau R² during training (lightweight, no plots).

    Uses the ODE params analysis interface: gt_tau(), gt_vrest(), derive_tau(),
    derive_vrest(). Returns the empty-metric dict for models that don't have
    these params.

    Returns:
        dict with keys:
            vrest_r2       : R² over all neurons
            vrest_r2_clean : R² over inliers (|learned-gt| <= VREST_OUTLIER_THRESH)
            n_out_vrest    : number of V_rest outliers
            n_total_vrest  : total neurons evaluated for V_rest
            tau_r2         : R² over all neurons
            tau_r2_clean   : R² over inliers (|learned-gt| <= TAU_OUTLIER_THRESH)
            n_out_tau      : number of tau outliers
            n_total_tau    : total neurons evaluated for tau
    """
    from connectome_gnn.generators.ode_params import (
        FlyVisODEParams, get_ode_params_class,
    )
    signal_model = config.graph_model.signal_model_name
    try:
        OdeParamsCls = get_ode_params_class(signal_model)
    except KeyError:
        OdeParamsCls = FlyVisODEParams
    try:
        ode_params = OdeParamsCls.load(graphs_data_path(config.dataset), device=device)
    except FileNotFoundError:
        return dict(_DYNAMICS_R2_EMPTY)
    except TypeError:
        # On-disk schema mismatch (e.g. signal_model=drosophila_cx maps to
        # DrosophilaCxODEParams but the file holds FlyVisODEParams from the
        # voltage-recovery generator). Fall back to FlyVisODEParams.
        try:
            ode_params = FlyVisODEParams.load(
                graphs_data_path(config.dataset), device=device
            )
        except (FileNotFoundError, TypeError):
            return dict(_DYNAMICS_R2_EMPTY)

    mu, sigma = compute_activity_stats(x_ts, device)
    slopes, offsets = extract_f_theta_slopes(model, config, n_neurons, mu, sigma, device)

    out = dict(_DYNAMICS_R2_EMPTY)

    if ode_params.has_tau():
        gt_tau = ode_params.gt_tau(n_neurons)
        if gt_tau is not None:
            learned_tau = ode_params.derive_tau(slopes, n_neurons)
            tr2, tr2c, ntout, ntot = _r2_with_outliers(gt_tau, learned_tau, TAU_OUTLIER_THRESH)
            out['tau_r2']        = tr2
            out['tau_r2_clean']  = tr2c
            out['n_out_tau']     = ntout
            out['n_total_tau']   = ntot

    if ode_params.has_vrest():
        gt_vrest = ode_params.gt_vrest(n_neurons)
        if gt_vrest is not None:
            learned_vrest = ode_params.derive_vrest(slopes, offsets, n_neurons)
            vr2, vr2c, nvout, nvtot = _r2_with_outliers(gt_vrest, learned_vrest, VREST_OUTLIER_THRESH)
            out['vrest_r2']        = vr2
            out['vrest_r2_clean']  = vr2c
            out['n_out_vrest']     = nvout
            out['n_total_vrest']   = nvtot

    return out


def compute_dynamics_r2_linear(model, config, device, n_neurons):
    """Compute V_rest R² and tau R² for LinearODE (direct parameter comparison).

    Unlike GNN models where tau and V_rest must be extracted from f_theta
    slopes, the linear model exposes them as direct learnable parameters.

    Returns:
        (dynamics_dict, conn_r2): the same dict layout as compute_dynamics_r2
        plus a separate conn_r2 float.
    """
    import torch.nn.functional as F

    from connectome_gnn.generators.ode_params import get_ode_params_class, FlyVisODEParams
    signal_model = config.graph_model.signal_model_name
    try:
        OdeParamsCls = get_ode_params_class(signal_model)
    except KeyError:
        OdeParamsCls = FlyVisODEParams
    ode_params = OdeParamsCls.load(graphs_data_path(config.dataset), device=device)
    gt_weights = to_numpy(ode_params.W)
    learned_W = to_numpy(get_model_W(model).squeeze())

    out = dict(_DYNAMICS_R2_EMPTY)
    conn_r2 = 0.0

    # tau and V_rest only exist for FlyVis models
    if hasattr(ode_params, 'V_i_rest') and ode_params.V_i_rest is not None:
        try:
            learned_vrest = to_numpy(model.V_rest[:n_neurons].detach())
            gt_vrest = to_numpy(ode_params.V_i_rest[:n_neurons])
            vr2, vr2c, nvout, nvtot = _r2_with_outliers(gt_vrest, learned_vrest, VREST_OUTLIER_THRESH)
            out['vrest_r2']        = vr2
            out['vrest_r2_clean']  = vr2c
            out['n_out_vrest']     = nvout
            out['n_total_vrest']   = nvtot
        except Exception:
            pass
    if hasattr(ode_params, 'tau_i') and ode_params.tau_i is not None:
        try:
            learned_tau = to_numpy(F.softplus(model.raw_tau[:n_neurons]).detach())
            gt_tau = to_numpy(ode_params.tau_i[:n_neurons])
            tr2, tr2c, ntout, ntot = _r2_with_outliers(gt_tau, learned_tau, TAU_OUTLIER_THRESH)
            out['tau_r2']        = tr2
            out['tau_r2_clean']  = tr2c
            out['n_out_tau']     = ntout
            out['n_total_tau']   = ntot
        except Exception:
            pass
    try:
        conn_r2, _ = compute_r_squared_NSE(gt_weights, learned_W)
    except Exception:
        pass

    return out, conn_r2


# ------------------------------------------------------------------ #
#  Jacobian-based connectivity R2 for MLP baseline
# ------------------------------------------------------------------ #

def compute_jacobian_connectivity_r2(model, x_ts, ode_params, n_neurons, device,
                                     n_samples=100, seed=0):
    """Compute connectivity R2 by comparing Jacobian dF/dv to GT weight matrix.

    The MLP baseline has no explicit W. We extract the effective connectivity
    from the Jacobian dF/dv averaged over multiple frames, then compare to
    the GT weight matrix (dense, n_neurons x n_neurons).

    Returns:
        conn_r2: float R² value
    """
    import numpy as np

    model.eval()
    with torch.no_grad():
        pass  # just to set eval mode
    # Need gradients for Jacobian computation
    J_mean = model.compute_jacobian_batched(x_ts, n_samples=n_samples, seed=seed)
    model.train()

    # Build GT dense weight matrix
    ei = to_numpy(ode_params.edge_index)
    gt_W = to_numpy(ode_params.W)
    W_dense_gt = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    W_dense_gt[ei[0], ei[1]] = gt_W

    J_np = to_numpy(J_mean)

    try:
        conn_r2, _ = compute_r_squared_NSE(W_dense_gt.flatten(), J_np.flatten())
    except Exception:
        conn_r2 = 0.0

    return conn_r2


# ------------------------------------------------------------------ #
#  Gradient of f_theta w.r.t. msg
# ------------------------------------------------------------------ #

def compute_grad_msg(model, in_features, config):
    """Compute d(f_theta)/d(msg) for each neuron from a forward-pass in_features.

    Args:
        model: NeuralGNN model.
        in_features: (N, D) tensor from model(..., return_all=True).
            Layout: [v(1), embedding(E), msg(1), excitation(1)].
        config: config object with graph_model.embedding_dim.

    Returns:
        grad_msg: (N,) tensor of gradients.
    """
    emb_dim = config.graph_model.embedding_dim
    v = in_features[:, 0:1].clone().detach()
    embedding = in_features[:, 1:1 + emb_dim].clone().detach()
    msg = in_features[:, 1 + emb_dim:2 + emb_dim].clone().detach()
    excitation = in_features[:, 2 + emb_dim:3 + emb_dim].clone().detach()

    msg.requires_grad_(True)
    in_features_grad = torch.cat([v, embedding, msg, excitation], dim=1)
    out = model.f_theta(in_features_grad)

    grad = torch.autograd.grad(
        outputs=out,
        inputs=msg,
        grad_outputs=torch.ones_like(out),
        retain_graph=False,
        create_graph=False,
    )[0]

    return grad.squeeze().detach()


# ------------------------------------------------------------------ #
#  Corrected weights
# ------------------------------------------------------------------ #

def compute_corrected_weights(model, edges, slopes_f_theta, slopes_g_phi, grad_msg):
    """Compute corrected W_ij from raw W, slopes, and grad_msg.

    Formula:
        corrected_W_ij = -W_ij / slope_phi[i] * grad_msg[i] * slope_edge[j]

    Args:
        model: model with .W, .n_edges, .n_extra_null_edges attributes.
        edges: (2, E) edge index tensor.
        slopes_f_theta: (N,) array/tensor of f_theta slopes per neuron.
        slopes_g_phi: (N,) array/tensor of g_phi slopes per neuron.
        grad_msg: (N,) tensor of d(f_theta)/d(msg) per neuron.

    Returns:
        corrected_W: (E, 1) tensor of corrected weights.
    """
    device = get_model_W(model).device

    # Convert to tensors if needed
    if not isinstance(slopes_f_theta, torch.Tensor):
        slopes_f_theta = torch.tensor(slopes_f_theta, dtype=torch.float32, device=device)
    if not isinstance(slopes_g_phi, torch.Tensor):
        slopes_g_phi = torch.tensor(slopes_g_phi, dtype=torch.float32, device=device)

    n_w = model.n_edges + model.n_extra_null_edges

    # Map edges to neuron indices (handles batched edges via modulo)
    target_neuron_ids = edges[1, :] % n_w   # i — post-synaptic
    prior_neuron_ids = edges[0, :] % n_w    # j — pre-synaptic

    slopes_phi_per_edge = slopes_f_theta[target_neuron_ids]     # (E,)
    slopes_edge_per_edge = slopes_g_phi[prior_neuron_ids]    # (E,)
    grad_msg_per_edge = grad_msg[target_neuron_ids]             # (E,)

    W = get_model_W(model)  # (E, 1)

    corrected_W = (-W
                   / slopes_phi_per_edge[:, None]
                   * grad_msg_per_edge.unsqueeze(1)
                   * slopes_edge_per_edge.unsqueeze(1))

    # Sanitize: division by near-zero slopes can produce inf/nan
    corrected_W = torch.nan_to_num(corrected_W, nan=0.0, posinf=0.0, neginf=0.0)

    return corrected_W


def compute_all_corrected_weights(model, config, edges, x_ts, device,
                                   n_grad_frames=8, ode_params=None):
    """High-level: compute corrected W from model state and training data.

    Uses model-specific g_phi fitting via ode_params to extract the per-neuron
    correction factor (ReLU slope for flyvis, softplus gain for CX, etc.).

    Args:
        model: NeuralGNN model.
        config: full config object.
        edges: (2, E) edge index tensor.
        x_ts: NeuronTimeSeries (training data).
        device: torch device.
        n_grad_frames: number of frames to sample for grad_msg (default 8).
        ode_params: ODEParamsBase instance for model-specific g_phi fitting.

    Returns:
        corrected_W: (E, 1) tensor of corrected weights.
        slopes_f_theta: (N,) numpy array.
        g_phi_correction: (N,) numpy array — per-neuron factor used for W correction.
        offsets_f_theta: (N,) numpy array.
        g_phi_fitted: dict — all fitted g_phi params (model-specific).
    """
    n_neurons = model.a.shape[0]

    # 1. Activity statistics
    mu_activity, sigma_activity = compute_activity_stats(x_ts, device)

    # 2. Evaluate g_phi curves and fit model-specific parameters
    v_ranges, g_phi_curves, valid = evaluate_g_phi_curves(
        model, config, n_neurons, mu_activity, sigma_activity, device)

    if ode_params is not None:
        g_phi_fitted = ode_params.fit_g_phi_curves(v_ranges, g_phi_curves)
    else:
        # Fallback: linear slope (legacy behavior)
        rr_t = torch.tensor(v_ranges, dtype=torch.float32)
        func_t = torch.tensor(g_phi_curves, dtype=torch.float32)
        slopes, _ = _vectorized_linear_fit(rr_t, func_t)
        slopes[~valid] = 1.0
        g_phi_fitted = {'correction': slopes, 'slopes': slopes}

    g_phi_correction = g_phi_fitted['correction']
    g_phi_correction[~valid] = 1.0

    # 3. f_theta slopes
    slopes_f_theta, offsets_f_theta = extract_f_theta_slopes(
        model, config, n_neurons, mu_activity, sigma_activity, device)

    # 4. Compute grad_msg over multiple frames and take median
    n_frames = x_ts.voltage.shape[0]
    frame_indices = np.linspace(n_frames // 10, n_frames - 100, n_grad_frames, dtype=int)
    data_id = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)

    was_training = model.training
    model.eval()

    edges = edges.to(device)

    grad_list = []
    for k in frame_indices:
        state = x_ts.frame(int(k)).to(device)
        with torch.no_grad():
            _, in_features, _ = model(state, edges, data_id=data_id, return_all=True)
        grad_k = compute_grad_msg(model, in_features, config)
        grad_list.append(grad_k)

    if was_training:
        model.train()

    grad_msg = torch.stack(grad_list).median(dim=0).values  # (N,)

    # 5. Corrected weights using model-specific g_phi correction
    corrected_W = compute_corrected_weights(
        model, edges, slopes_f_theta, g_phi_correction, grad_msg)

    return corrected_W, slopes_f_theta, g_phi_correction, offsets_f_theta, g_phi_fitted
