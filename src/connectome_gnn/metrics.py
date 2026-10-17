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

import os
import numpy as np
import torch
from scipy.optimize import curve_fit

from connectome_gnn.fitting_models import linear_model
from connectome_gnn.models.utils import pad_g_phi_input, is_general_form_gnn
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
    """The model's weights IN THE SAME UNITS AS ode_params.W, for comparison."""
    # checked first: under w_squared the message uses W**2, so W**2 (not the raw
    # root) is what compares to ode_params.W; the sign-lock is moot for a square.
    if getattr(model, 'w_squared', False):
        return model.W.detach() ** 2
    # Prefer the effective weight (|W|·sign_GT under the hard sign-lock); when
    # the lock is off this equals the raw W, so existing models are unaffected.
    if hasattr(model, 'effective_W'):
        return model.effective_W
    # the conductance student stores sqrt(g) as W; ode_params.W holds g itself.
    if hasattr(model, 'get_learned_conductance'):
        return model.get_learned_conductance()
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
    

def _r2_slope_identity(true: np.ndarray, learned: np.ndarray) -> tuple[float, float]:
    """Identity-line R² (Nash-Sutcliffe efficiency) and calibration slope.

    R² = 1 - mean((true - learned)²) / var(true)
    Penalizes both noise and scale/bias errors. Range: (-inf, 1].
    Slope from learned ≈ a·true + b diagnoses scale miscalibration when R² is low.

    Private — recovery_param_metrics() is the single public entry point for
    R² anywhere in this codebase; nothing outside this module should call
    this directly.
    """
    try:
        a = np.asarray(true).ravel()
        # constant GT: the R² is undefined and explodes on float noise; NaN makes
        # callers show 'N/A' + a MAE (see is_degenerate_gt).
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
    _m = recovery_param_metrics(true, learned)
    r2, slope = _m['r2'], _m['slope']
    if clean_r2 is not None:
        return f'{label}: {clean_r2:.2f} ({r2:.2f})\nslope: {slope:.2f}{tail}'
    return f'{label}: {r2:.2f}\nslope: {slope:.2f}{tail}'


def fmt_r2_bar(val) -> str:
    """Progress-bar value: ``'N/A'`` when the R² is undefined (None/NaN, e.g. a
    constant-GT parameter), else 3-dp."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 'N/A'
    return f'{val:.3f}'


def recovery_param_metrics(gt: np.ndarray, learned: np.ndarray, outlier_thresh: float = None) -> dict:
    """All recovery metrics for a parameter (W, tau, V_rest, ...) — the single
    public entry point for R² anywhere in this codebase, computed ONCE so the
    scatter, console line and metrics.txt can't disagree.

    Outlier rule: ``|learned - true| > outlier_thresh`` (the neurips.tex
    eq:outlier_threshold band, delta_tau=0.1 / delta_Vrest=0.2 / delta_W=1.0).
    With ``outlier_thresh=None`` (default) nothing is filtered — ``r2_clean``
    equals ``r2`` — for callers that just want the plain full-sample R²/slope.

    Returns a dict with: ``r2``/``slope`` (full identity-line NSE),
    ``r2_clean``/``slope_clean`` (inliers only; NaN if <2 inliers),
    ``n_outliers``/``n_total``/``pct_outliers``, ``outlier_mask``/``inlier_mask``,
    ``degenerate`` (bool, GT ~constant -> R² undefined), ``mae`` (fallback metric),
    and ``rel_err_median``/``rel_err_iqr`` (|learned-true|/max(|true|,1e-6))."""
    gt = np.asarray(gt).ravel()
    learned = np.asarray(learned).ravel()
    n = min(gt.size, learned.size)
    gt, learned = gt[:n], learned[:n]
    r2, slope = _r2_slope_identity(gt, learned)
    if outlier_thresh is None:
        out_mask = np.zeros(gt.shape, dtype=bool)
    else:
        out_mask = np.abs(learned - gt) > outlier_thresh
    in_mask = ~out_mask
    n_out = int(out_mask.sum())
    n_tot = int(gt.size)
    if int(in_mask.sum()) >= 2:
        r2_clean, slope_clean = _r2_slope_identity(gt[in_mask], learned[in_mask])
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
                      device, chunk_size=2000, post_fn=None, model_a_i=None):
    """Evaluate an MLP for all neurons at once, in chunks.

    Instead of N individual forward passes on (1000, D) inputs, we
    stack all neurons into (N*1000, D) and run one pass per chunk.

    Args:
        mlp: nn.Module — the MLP to evaluate (model.g_phi or model.f_theta).
        model_a: (N, emb_dim) embedding tensor.
        rr: (N, n_pts) tensor of input values per neuron.
        build_features_fn: callable(rr_flat, emb_flat) -> (chunk*n_pts, D), or
            callable(rr_flat, emb_flat, emb_i_flat) -> (chunk*n_pts, D) when
            model_a_i is given. Builds the MLP input features.
        device: torch device.
        chunk_size: number of neurons per chunk (limits GPU memory).
        post_fn: optional callable applied to MLP output (e.g. lambda x: x**2).
        model_a_i: optional (N, emb_dim) second per-row embedding tensor
            (e.g. a real postsynaptic partner embedding), chunked/expanded
            the same way as model_a and passed to build_features_fn as a
            third argument. None (default) preserves the 2-arg call.

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

        if model_a_i is not None:
            chunk_a_i = model_a_i[i:i + chunk_size]                       # (C, emb_dim)
            emb_i_flat = chunk_a_i[:, None, :].expand(-1, n_pts, -1)
            emb_i_flat = emb_i_flat.reshape(-1, emb_dim)                  # (C*n_pts, emb_dim)
            in_features = build_features_fn(rr_flat, emb_flat, emb_i_flat)
        else:
            in_features = build_features_fn(rr_flat, emb_flat)   # (C*n_pts, D)

        with torch.no_grad():
            # no-op unless the noise-probe control widened this MLP's input
            out = mlp(pad_g_phi_input(in_features.float(), mlp))  # (C*n_pts, 1)
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

def _build_g_phi_features(rr_flat, emb_flat, signal_model_name, emb_i_flat=None):
    """Build input features for g_phi MLP.

    rr_flat / emb_flat: swept presynaptic voltage vj and embedding aj.
    emb_i_flat: postsynaptic embedding ai — required for flyvis_general_form, where
        g_phi(vi=0, vj, ai, aj) depends on both endpoints. Should be a real
        partner embedding (see `_avg_postsynaptic_embedding`), not aj again.
    """
    if is_general_form_gnn(signal_model_name):
        if emb_i_flat is None:
            raise ValueError(
                "_build_g_phi_features: flyvis_general_form requires emb_i_flat (postsynaptic embedding)"
            )
        return torch.cat([rr_flat, emb_flat, rr_flat * 0, emb_i_flat], dim=1)
    else:
        return torch.cat([rr_flat, emb_flat], dim=1)


def _avg_postsynaptic_embedding(model_a, edges, n_neurons):
    """Per-presynaptic-neuron average embedding of its real postsynaptic partners.

    ai_avg[j] = mean(model_a[dst]) over edges with src == j. Used to build a
    representative ai for g_phi(vi=0, vj, ai, aj) when g_phi depends on both
    endpoints (flyvis_general_form) — real partners, not aj=ai. Neurons with no outgoing
    edges fall back to their own embedding.
    """
    src, dst = edges[0], edges[1]
    emb_dim = model_a.shape[1]
    sum_emb = torch.zeros(n_neurons, emb_dim, device=model_a.device, dtype=model_a.dtype)
    count = torch.zeros(n_neurons, device=model_a.device, dtype=model_a.dtype)
    sum_emb.index_add_(0, src, model_a[dst])
    count.index_add_(0, src, torch.ones(src.shape[0], device=model_a.device, dtype=model_a.dtype))
    has_out = (count > 0).unsqueeze(1)
    return torch.where(has_out, sum_emb / count.clamp_min(1).unsqueeze(1), model_a[:n_neurons])


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
    # reduce on CPU in float64 chunks: (T, N) can be ~12 GiB and the target
    # device may be full; only the (N,) results are moved to `device`.
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

def eval_g_phi_over_domain(model, config, n_neurons, rr, device, edges=None):
    """Evaluate g_phi over a caller-supplied per-neuron voltage domain.

    Shared core of `evaluate_g_phi_curves` — factored out so callers that
    need a non-default domain (e.g. a wider range for plotting, vs. the
    activity-range domain used for slope fitting) can reuse the same
    flyvis_general_form-aware feature construction instead of duplicating it.

    Args:
        rr: (N, n_pts) tensor of per-neuron voltage values to sweep (vj).

    For flyvis_general_form, g_phi depends on both endpoints (vi, vj, ai, aj);
    ai is built as each neuron's real average postsynaptic-partner embedding
    via `edges` (see `_avg_postsynaptic_embedding`), not a self-pair ai=aj.
    `edges` is required in that case.

    Returns:
        (N, n_pts) tensor of g_phi outputs.
    """
    signal_model_name = config.graph_model.signal_model_name
    g_phi_positive = config.graph_model.g_phi_positive
    post_fn = (lambda x: x ** 2) if g_phi_positive else None
    model_a = model.a[:n_neurons]

    if is_general_form_gnn(signal_model_name):
        if edges is None:
            raise ValueError(
                "eval_g_phi_over_domain: flyvis_general_form requires `edges` to build the postsynaptic embedding ai"
            )
        model_a_i = _avg_postsynaptic_embedding(model_a, edges, n_neurons)
        build_fn = lambda rr_f, emb_f, emb_i_f: _build_g_phi_features(
            rr_f, emb_f, signal_model_name, emb_i_flat=emb_i_f)
    else:
        model_a_i = None
        build_fn = lambda rr_f, emb_f: _build_g_phi_features(rr_f, emb_f, signal_model_name)

    return _batched_mlp_eval(model.g_phi, model_a, rr,
                             build_fn, device, post_fn=post_fn, model_a_i=model_a_i)  # (N, n_pts)


def evaluate_g_phi_curves(model, config, n_neurons, mu_activity, sigma_activity, device, edges=None):
    """Evaluate learned g_phi curves over each neuron's activity range.

    See `eval_g_phi_over_domain` for the flyvis_general_form / `edges` note.

    Returns:
        v_ranges: (N, n_pts) numpy array of voltage grids per neuron.
        curves: (N, n_pts) numpy array of g_phi outputs.
        valid: (N,) bool array — neurons with positive activity range.
    """
    n_pts = 1000

    mu = to_numpy(mu_activity).astype(np.float32) if torch.is_tensor(mu_activity) else np.asarray(mu_activity, dtype=np.float32)
    sigma = to_numpy(sigma_activity).astype(np.float32) if torch.is_tensor(sigma_activity) else np.asarray(sigma_activity, dtype=np.float32)

    valid = (mu + sigma) > 0
    starts = np.maximum(mu - 2 * sigma, 0.0)
    ends = mu + 2 * sigma
    starts[~valid] = 0.0
    ends[~valid] = 1.0

    rr = _vectorized_linspace(starts, ends, n_pts, device)  # (N, n_pts)
    func = eval_g_phi_over_domain(model, config, n_neurons, rr, device, edges=edges)

    return to_numpy(rr), to_numpy(func), valid


# rows per g_phi call in _g_phi_on_edges (~300 MB activation at hidden 80);
# a memory knob only, the result does not depend on it.
G_PHI_EVAL_CHUNK = 1_000_000

# (edge, frame) pairs per block of the streamed template readout (~2 GB GPU,
# ~7 GB host per block); a memory knob only, every sum is per edge.
TEMPLATE_BLOCK_PAIRS = 1 << 26


def _pooled_quantile(values, weights, q, positive=False, percentile=False):
    """np.quantile(x, q) -- or np.percentile(x, q) -- of the array x holding each
    values[f, n] weights[n] times, without building x.

    Exact, including numpy's rounding: the two order statistics the linear
    method interpolates between are read off the cumulative weights, q is given
    the dtype numpy would give it (a Python float takes a float array's dtype;
    a percentile is divided by 100 in that dtype), the virtual index is computed
    in that dtype, and the interpolation itself is handed back to np.quantile on
    the two values. `positive` keeps only values > 0 first, as `u[u > 0]` does.
    None when nothing is left; nan when a kept value is nan, as numpy returns.

    Args:
        values: (n_frames, n_cells) float array.
        weights: (n_cells,) non-negative integer counts.
        q: a fraction, or with `percentile` a percentage.
    """
    v = np.asarray(values)
    if percentile:
        q = np.asanyarray(np.true_divide(q, v.dtype.type(100)))
    elif isinstance(q, (int, float)) and v.dtype.kind == "f":
        q = np.asanyarray(q, dtype=v.dtype)
    else:
        q = np.asanyarray(q)
    w = np.broadcast_to(np.asarray(weights, dtype=np.int64), v.shape).ravel()
    v = v.ravel()
    keep = w > 0
    if positive:
        keep &= v > 0
    v, w = v[keep], w[keep]
    n = int(w.sum())
    if n == 0:
        return None
    if np.isnan(v).any():
        return float("nan")
    order = np.argsort(v, kind="stable")
    sv, cw = v[order], np.cumsum(w[order])
    virt = (n - 1) * q
    if virt >= n - 1:
        lo = hi = n - 1
    elif virt < 0:
        lo = hi = 0
    else:
        lo = int(np.floor(virt))
        hi = lo + 1
    pair = np.array([sv[np.searchsorted(cw, lo, side="right")],
                     sv[np.searchsorted(cw, hi, side="right")]], dtype=sv.dtype)
    # A 0-d array, not a Python float, so np.quantile keeps its dtype.
    gamma = np.asanyarray(virt - np.floor(virt), dtype=virt.dtype)
    return float(np.quantile(pair, gamma))


def sample_g_phi_vi_vj_observed(model, config, edges, x_ts, n_edges=16, n_frames=2000,
                                seed=0, frame_idx=None):
    """Evaluate learned g_phi at REAL, co-occurring (vi(t), vj(t)) pairs for a
    sample of real edges — not an independent (vi, vj) grid.

    Tests whether g_phi depends on vi where real activity lives: connected
    neurons are correlated, so the visited (vi, vj) region is not the full grid.
    Uses the real (ai, aj) of each sampled edge.

    Args:
        edges: (2, E) edge index tensor — src (presynaptic, j) / dst
            (postsynaptic, i).
        x_ts: NeuronTimeSeries with the real voltage recording (T, N).
        n_edges: number of real edges to sample (one panel each).
        n_frames: number of time frames to subsample per edge (real
            observed (vi, vj) pairs — subsampled for compute, not every
            frame).
        seed: RNG seed for edge/frame sampling (reproducible panel
            selection).

    Returns:
        dict with:
            edge_ij: (n_edges, 2) int array of (i, j) sampled — i=post, j=pre.
            vi / vj: (n_edges, n_frames) numpy arrays — observed voltage
                pairs, same time indices for both (co-occurring).
            g_phi: (n_edges, n_frames) numpy array — g_phi at those pairs.
    """
    device = model.a.device
    sel, frame_idx = _observed_plan(int(edges.shape[1]), n_edges, int(x_ts.n_frames),
                                    n_frames, seed, frame_idx)
    src = edges[0, sel].to(device)   # j — presynaptic
    dst = edges[1, sel].to(device)   # i — postsynaptic
    vi, vj, g_phi_vals = _g_phi_on_edges(model, config, src, dst,
                                         _frame_voltages(x_ts, frame_idx, device))

    return {
        'edge_ij': np.stack([to_numpy(dst), to_numpy(src)], axis=1),
        # indices into the caller's edge array: rng.choice permutes even a full
        # draw, so pair results with ode_params.W through this, not by (i, j).
        'edge_idx': np.asarray(sel).astype(np.int64),
        'vi': to_numpy(vi),
        'vj': to_numpy(vj),
        'g_phi': to_numpy(g_phi_vals),
    }


def _observed_plan(n_edges_total, n_edges, n_total_frames, n_frames, seed,
                   frame_idx=None):
    """The edges and frames `sample_g_phi_vi_vj_observed` samples, and nothing else.

    One RNG stream: edges first, then frames (unless given). The streamed
    template readout draws through here too, so it reads the same pairs.
    """
    rng = np.random.default_rng(seed)
    sel = rng.choice(n_edges_total, size=min(n_edges, n_edges_total), replace=False)
    # a caller may pass frame_idx, e.g. to cover a rarely-active presynaptic
    # cell that a uniform draw would miss.
    if frame_idx is None:
        frame_idx = rng.choice(n_total_frames, size=min(n_frames, n_total_frames),
                               replace=False)
    return sel, np.asarray(frame_idx).astype(np.int64)


def _frame_voltages(x_ts, frame_idx, device):
    """(n_frames, N) voltages at `frame_idx`, on `device`.

    Indexed before moving: the whole (T, N) recording can exceed 10 GB.
    """
    vol = x_ts.voltage
    fi = torch.as_tensor(np.asarray(frame_idx), device=vol.device).long()
    return vol[fi].to(device)


def _g_phi_on_edges(model, config, src, dst, volt_f):
    """g_phi at the observed (v_i, v_j) of the edges src -> dst, over the frames of volt_f.

    Returns (vi, vj, g_phi), each an (n_e, n_frames) tensor; g_phi squared if
    `g_phi_positive`, as it enters the message.
    """
    signal_model_name = config.graph_model.signal_model_name
    g_phi_positive = config.graph_model.g_phi_positive
    emb_dim = model.a.shape[1]
    n_e, n_frames = int(src.numel()), int(volt_f.shape[0])
    vi = volt_f[:, dst].T.contiguous()                   # (n_e, n_frames)
    vj = volt_f[:, src].T.contiguous()                   # (n_e, n_frames)

    ai_flat = model.a[dst].unsqueeze(1).expand(-1, n_frames, -1).reshape(-1, emb_dim)
    aj_flat = model.a[src].unsqueeze(1).expand(-1, n_frames, -1).reshape(-1, emb_dim)
    vi_flat = vi.reshape(-1, 1)
    vj_flat = vj.reshape(-1, 1)

    # chunked: callers may pass every edge (~14M rows at flyvis scale, OOM on a
    # 24 GB card in one call); rows are independent, so the result is bit-identical.
    with torch.no_grad():
        outs = []
        for _lo in range(0, vj_flat.shape[0], G_PHI_EVAL_CHUNK):
            _hi = _lo + G_PHI_EVAL_CHUNK
            if is_general_form_gnn(signal_model_name):
                _in = torch.cat([vj_flat[_lo:_hi], aj_flat[_lo:_hi],
                                 vi_flat[_lo:_hi], ai_flat[_lo:_hi]], dim=1)
            else:
                _in = torch.cat([vj_flat[_lo:_hi], aj_flat[_lo:_hi]], dim=1)
            _o = model.g_phi(pad_g_phi_input(_in.float(), model))
            if g_phi_positive:
                _o = _o ** 2
            outs.append(_o)
        out = torch.cat(outs, dim=0)

    return vi, vj, out.reshape(n_e, n_frames)


def r2_up_to_scale(true, learned):
    """R2 of `learned` against `true` after dividing out ONE global gain.

    For quantities fixed only up to a common factor (a GNN's W_ij and msg_i,
    which trade off against the g_phi amplitude exactly).

    The gain convention used by every `<key>_gain` this module emits:

        gain = <true, learned> / <true, true>        so that  learned ~= gain * true

    (1.0 perfect, 2.0 = learned twice too large); the R2 is the identity-line R2
    of learned / gain against true. The slope is then 1 by construction, so none
    is returned.

    Returns dict with r2, gain, n. `scale` is kept as an alias of `gain`.
    """
    true = np.asarray(true).ravel().astype(np.float64)
    learned = np.asarray(learned).ravel().astype(np.float64)
    n = min(true.size, learned.size)
    true, learned = true[:n], learned[:n]
    ok = np.isfinite(true) & np.isfinite(learned)
    true, learned = true[ok], learned[ok]
    denom = float(true @ true)
    nan = float('nan')
    if true.size < 2 or denom <= 0:
        return {'r2': nan, 'gain': nan, 'scale': nan, 'n': int(true.size)}
    gain = float(true @ learned) / denom
    if abs(gain) < 1e-300:
        return {'r2': nan, 'gain': gain, 'scale': gain, 'n': int(true.size)}
    r2, _ = _r2_slope_identity(true, learned / gain)
    return {'r2': r2, 'gain': gain, 'scale': gain, 'n': int(true.size)}


def g_phi_first_layer_discard_score(model, emb_dim):
    """L1 mass on g_phi's [vi, ai] first-layer input columns (the ones the
    true flyvis_conductance generative model ReLU(vj) doesn't need), as a
    fraction of the total L1 mass across all 4 input groups [vi, vj, ai, aj].
    0 = filter has fully dropped vi/ai; 0.5 = no preference; matches a dot
    product of the per-group L1 norms against the discard mask [1,0,1,0],
    normalized. Pure weight inspection, no forward pass or data needed.

    No-op (returns nan) unless g_phi's first layer has the flyvis_general_form
    2+2*emb_dim input width -- matched structurally, not by model name.
    """
    first_layer = model.g_phi.layers[0]
    W0 = first_layer.weight.detach()
    if W0.shape[1] != 2 + 2 * emb_dim:
        return float('nan')
    n_vj = W0[:, 0:1].abs().sum().item()
    n_aj = W0[:, 1:1 + emb_dim].abs().sum().item()
    n_vi = W0[:, 1 + emb_dim:2 + emb_dim].abs().sum().item()
    n_ai = W0[:, 2 + emb_dim:2 + 2 * emb_dim].abs().sum().item()
    total = n_vi + n_vj + n_ai + n_aj
    return (n_vi + n_ai) / total if total > 0 else float('nan')


def g_phi_column_layout(model, emb_dim):
    """Column slices of g_phi's first-layer weight, by model family.

    g_phi's input is built in two shapes (neural_gnn.py NeuralGNN.message):

        flyvis_general_form : [v_j, a_j, v_i, a_i] + noise   width 2 + 2*emb + n
        everything else    : [v_j, a_j]           + noise   width 1 + emb   + n

    Columns 0..emb_dim are [v_j, a_j] in both families, so hardcoded column
    indices (e.g. the coeff_g_phi_norm anchor at column 0) mean the same thing.

    Dispatched on `model.model`, the attribute the forward pass branches on.
    Returns (layout, n_noise), layout mapping 'vi'/'vj'/'ai'/'aj'/'noise' to
    slices ('vi'/'ai' None on the non-conductance layout); (None, 0) if the
    width matches neither layout.
    """
    width = model.g_phi.layers[0].weight.shape[1]
    if is_general_form_gnn(getattr(model, "model", None) or ""):
        base = 2 + 2 * emb_dim
        if width < base:
            return None, 0
        layout = {'vj': slice(0, 1), 'aj': slice(1, 1 + emb_dim),
                  'vi': slice(1 + emb_dim, 2 + emb_dim), 'ai': slice(2 + emb_dim, base)}
    else:
        base = 1 + emb_dim
        if width < base:
            return None, 0
        layout = {'vi': None, 'vj': slice(0, 1), 'ai': None, 'aj': slice(1, base)}
    layout['noise'] = slice(base, width) if width > base else None
    return layout, width - base


def g_phi_first_layer_cosine_to_keep(model, emb_dim):
    """Cosine similarity between g_phi's first-layer per-group L2 norms
    [vi, vj, ai, aj] (the SAME quantity the group-lasso regularizer itself
    penalizes -- see the group_norm term in regularizer.py) and the target
    direction [0, 1, 0, 1] (only vj, aj should matter for the true
    ReLU(vj) generative model).

    Scale-invariant, and indifferent to how vj/aj split the kept mass.
    1 = fully aligned (vi=ai=0); 0 = all mass on vi/ai.

    Defined for both g_phi layouts (see g_phi_column_layout): the discard side is
    {vi, ai, noise} on flyvis_general_form and {noise} alone on flyvis_A.

    No-op (returns nan) if g_phi's first layer matches neither layout.
    """
    layout, _ = g_phi_column_layout(model, emb_dim)
    if layout is None:
        return float('nan')
    W0 = model.g_phi.layers[0].weight.detach()

    def gnorm(key):
        sl = layout[key]
        return W0[:, sl].norm(2).item() if sl is not None else 0.0

    n_vj, n_aj = gnorm('vj'), gnorm('aj')
    v_norm = sum(gnorm(k) ** 2 for k in ('vi', 'vj', 'ai', 'aj', 'noise')) ** 0.5
    if v_norm <= 0:
        return float('nan')
    return (n_vj + n_aj) / (v_norm * (2 ** 0.5))


def compute_g_phi_grad_ratios(model, config, edges, x_ts, n_frames=16, seed=0):
    """Functional companion to g_phi_first_layer_discard_score: ratios of
    |d(g_phi)/d(vi)| and the embedding-gradient norm |d(g_phi)/d(ai)| against
    |d(g_phi)/d(vj)|, averaged over real (edge, frame) pairs via autograd.
    Trust this over the weight-level score when they disagree: a later layer
    can route around a large first-layer weight.

    Returns (ratio_vi, ratio_ai, ratio_noise). ratio_noise is nan unless
    n_g_phi_noise_inputs > 0; the noise columns carry no information, so any
    non-zero value is misassigned credit. ratio_vi / ratio_ai are nan on the
    flyvis_A layout (no vi/ai inputs). All nan if the layout matches neither
    family (see g_phi_column_layout).
    """
    device = model.a.device
    emb_dim = model.a.shape[1]
    layout, n_noise = g_phi_column_layout(model, emb_dim)
    if layout is None:
        return float('nan'), float('nan'), float('nan')
    has_post = layout['vi'] is not None

    src = edges[0].to(device)
    dst = edges[1].to(device)

    rng = np.random.default_rng(seed)
    n_frames = min(n_frames, x_ts.n_frames)
    frame_idx = rng.choice(x_ts.n_frames, size=n_frames, replace=False)

    voltage = x_ts.voltage.to(device)
    ai_fixed = model.a[dst]
    aj_fixed = model.a[src]
    g_phi_positive = config.graph_model.g_phi_positive

    abs_grad_vi, abs_grad_vj, grad_ai_norm, grad_noise_norm = [], [], [], []
    for k in frame_idx:
        vj_k = voltage[k, src].clone().detach().requires_grad_(True)
        vi_k = voltage[k, dst].clone().detach().requires_grad_(True)
        ai_k = ai_fixed.clone().detach().requires_grad_(True)

        # Both layouts start [v_j, a_j]; conductance appends [v_i, a_i]. wrt is
        # ordered vj, (vi, ai), (noise) so grads[0] is ALWAYS d/dvj -- the shared
        # denominator -- regardless of family.
        if has_post:
            parts = [vj_k.unsqueeze(1), aj_fixed, vi_k.unsqueeze(1), ai_k]
            wrt = [vj_k, vi_k, ai_k]
        else:
            # flyvis_A layout: g_phi never sees v_i or a_i, so there is nothing
            # post-synaptic to differentiate.
            parts = [vj_k.unsqueeze(1), aj_fixed]
            wrt = [vj_k]
        if n_noise:
            # same distribution the model draws internally, so the gradient is
            # measured at a representative point of the noise input
            noise_k = torch.randn(src.shape[0], n_noise, device=device).requires_grad_(True)
            parts.append(noise_k)
            wrt.append(noise_k)

        in_features = torch.cat(parts, dim=1)
        out = model.g_phi(pad_g_phi_input(in_features.float(), model))
        if g_phi_positive:
            out = out ** 2

        grads = torch.autograd.grad(out.sum(), wrt, retain_graph=False, create_graph=False)
        abs_grad_vj.append(grads[0].detach().abs().mean().item())
        if has_post:
            abs_grad_vi.append(grads[1].detach().abs().mean().item())
            grad_ai_norm.append(grads[2].detach().norm(dim=1).mean().item())
        if n_noise:
            grad_noise_norm.append(grads[-1].detach().norm(dim=1).mean().item())

    m_vj = float(np.mean(abs_grad_vj))
    m_vi = float(np.mean(abs_grad_vi)) if abs_grad_vi else float('nan')
    m_ai = float(np.mean(grad_ai_norm)) if grad_ai_norm else float('nan')
    m_noise = float(np.mean(grad_noise_norm)) if grad_noise_norm else float('nan')
    if m_vj <= 0:
        return float('nan'), float('nan'), float('nan')
    return m_vi / m_vj, m_ai / m_vj, m_noise / m_vj


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

    slopes, offsets = _torch_linear_fit(rr, func)
    linear_pred = slopes[:, None] * rr + offsets[:, None]                # (N, n_pts)
    residual = func - linear_pred                                        # (N, n_pts)
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

    loss = (out ** 2).mean()

    return loss


# ------------------------------------------------------------------ #
#  Dynamics R² (V_rest and tau)
# ------------------------------------------------------------------ #

# Outlier thresholds — the single source of truth for all three recovered
# parameters. GNN_PlotFigure.py imports these rather than redefining them,
# so the live training metrics agree with the post-training data_plot summary.
TAU_OUTLIER_THRESH = 0.1
VREST_OUTLIER_THRESH = 0.2
W_OUTLIER_THRESH = 1.0


_DYNAMICS_R2_EMPTY = {
    'n_out_conn': 0, 'n_total_conn': 0,
    'vrest_r2': 0.0, 'vrest_r2_clean': float('nan'),
    'n_out_vrest': 0, 'n_total_vrest': 0,
    'tau_r2':   0.0, 'tau_r2_clean':   float('nan'),
    'n_out_tau':   0, 'n_total_tau':   0,
    # the arrays the R2s were computed from, so a recovery panel matches
    # metrics.log; None when the model or dataset has no such parameter.
    'tau_true': None, 'tau_learned': None,
    'vrest_true': None, 'vrest_learned': None,
}


def compute_dynamics_r2(model, x_ts, config, device, n_neurons, edges=None):
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

    tau and V_rest come from extract_recovered_params, scored by
    recovery_param_metrics.

    """
    from connectome_gnn.generators.ode_params import load_ode_params_for_run
    try:
        ode_params = load_ode_params_for_run(config, device=device)
    except (FileNotFoundError, TypeError):
        return dict(_DYNAMICS_R2_EMPTY)

    # `edges` is required on a GNN: its tau and V_rest come from the template
    # readout's update fit, which runs the model's own forward pass.
    rec = extract_recovered_params(model, ode_params, config, edges=edges, x_ts=x_ts,
                                   device=device, n_neurons=n_neurons,
                                   need=("tau", "V_rest"))
    return _dynamics_dict_from(rec, config)


def _dynamics_dict_from(rec, config):
    """Render a RecoveredParams into the trainer's dynamics dict.

    Keep its shape: plot.py reads the metrics.log columns by position.
    """
    out = dict(_DYNAMICS_R2_EMPTY)
    for quantity, prefix in (("tau", "tau"), ("V_rest", "vrest")):
        pair = rec.get(quantity)
        if pair is None:
            continue
        gt, learned = pair
        m = recovery_param_metrics(gt, learned, _thresh_for(quantity, config))
        out[f'{prefix}_r2'] = m['r2']
        out[f'{prefix}_r2_clean'] = m['r2_clean']
        out[f'n_out_{prefix}'] = m['n_outliers']
        out[f'n_total_{prefix}'] = m['n_total']
        out[f'{prefix}_true'] = gt
        out[f'{prefix}_learned'] = learned
    return out


def compute_dynamics_r2_linear(model, config, device, n_neurons):
    """Compute V_rest R² and tau R² for LinearODE (direct parameter comparison).

    Unlike GNN models where tau and V_rest must be extracted from f_theta
    slopes, the linear model exposes them as direct learnable parameters.

    Returns:
        (dynamics_dict, conn_r2): the same dict layout as compute_dynamics_r2
        plus a separate conn_r2 float. The dict carries the tau / V_rest arrays.
    """
    from connectome_gnn.generators.ode_params import load_ode_params_for_run
    ode_params = load_ode_params_for_run(config, device=device)

    rec = extract_recovered_params(model, ode_params, config, edges=None,
                                   device=device, n_neurons=n_neurons,
                                   need=("W", "tau", "V_rest"))
    out = _dynamics_dict_from(rec, config)

    conn_r2 = 0.0
    w_pair = rec.get("W")
    if w_pair is not None:
        # outlier-filtered, matching plot_training_linear's scatter annotation.
        _cm = recovery_param_metrics(w_pair[0], w_pair[1], _thresh_for("W", config))
        conn_r2 = _cm['r2_clean']
        out['n_out_conn'], out['n_total_conn'] = _cm['n_outliers'], _cm['n_total']

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
        conn_r2 = recovery_param_metrics(W_dense_gt.flatten(), J_np.flatten())['r2']
    except Exception:
        conn_r2 = 0.0

    return conn_r2


# ------------------------------------------------------------------ #
#  Reversal potential recovery (conductance datasets only)
# ------------------------------------------------------------------ #

# frames the msg_i panel is evaluated on, evenly spaced and fixed, so two
# checkpoints differ only because the model moved.
MSG_N_FRAMES = 10


def compute_msg_i_recovery(model, ode_params, x_ts, edges, device,
                           n_frames=MSG_N_FRAMES, scale=None):
    """The aggregated per-neuron message msg_i, true vs learned.

    msg_i is the only channel by which connectivity reaches dv/dt, and the
    W_ij / E_ij trade-off inside g_ij * act(v_j) * (E_ij - v_i) cancels in it.

    The true side is rebuilt from ode_params (the generator's own message):

        edge_msg = W_ij * gt_g_phi_func(v_j)          the activation the
                                                      generator used, ReLU on
                                                      flyvis, whatever
                                                      `activation` says on cx
        edge_msg *= (E_ij - v_i)                      conductance datasets only;
                                                      a current generator has no
                                                      such term
        msg_i     = sum of edge_msg over incoming edges

    The learned side is the model's own msg from forward(..., return_all=True),
    carried into generator units by `scale` (the update fit's gauge) when given,
    else through f_theta on a GNN (see _msg_through_f_theta).

    Returns (true, learned) flattened over frames, each (n_frames * n_neurons,),
    or None when there is no ground-truth W.
    """
    if getattr(ode_params, 'W', None) is None:
        return None

    T = int(x_ts.n_frames)
    frame_idx = np.linspace(0, T - 1, n_frames).astype(int)

    ei = edges.to(device)
    src, dst = ei[0], ei[1]
    W = torch.as_tensor(to_numpy(ode_params.W).ravel(), dtype=torch.float32,
                        device=device)
    E_edge = None
    if getattr(ode_params, 'E_exc', None) is not None:
        E_edge = ode_params.reversal_per_edge().to(device).float().ravel()

    n_neurons = int(x_ts.voltage.shape[1])
    data_id = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)

    was_training = model.training
    model.eval()
    # the GNN's leak tau is f_theta's slope fitted over the activity range (as for
    # tau_R2); a per-frame central difference blows up where f_theta is flat in v.
    tau_fit = None
    if _msg_i_through_f_theta(model):
        try:
            mu, sigma = compute_activity_stats(x_ts, device)
            slopes, _ = extract_f_theta_slopes(model, None, n_neurons, mu, sigma, device)
            slopes = torch.as_tensor(np.asarray(slopes, dtype=np.float32), device=device)
            tau_fit = torch.where(slopes < -1e-6, -1.0 / slopes,
                                  torch.full_like(slopes, float("nan")))
            tau_fit = torch.where(tau_fit <= 10.0, tau_fit, torch.full_like(tau_fit, float("nan")))
        except Exception as exc:
            print(f"\033[93mmsg_i: leak-slope fit failed ({type(exc).__name__}: {exc}); "
                  "falling back to the local derivative\033[0m")
            tau_fit = None
    true_all, learned_all = [], []
    with torch.no_grad():
        for k in frame_idx:
            state = x_ts.frame(int(k)).to(device)
            v = state.voltage.float().ravel()

            act = torch.as_tensor(
                np.asarray(ode_params.gt_g_phi_func(to_numpy(v[src]))),
                dtype=torch.float32, device=device).ravel()
            edge_msg = W[:act.numel()] * act
            if E_edge is not None:
                edge_msg = edge_msg * (E_edge[:act.numel()] - v[dst][:act.numel()])
            msg_true = torch.zeros(n_neurons, device=device)
            msg_true.scatter_add_(0, dst[:edge_msg.numel()], edge_msg)

            pred, in_features, msg_learned = model(state, ei, data_id=data_id,
                                                   return_all=True)
            if scale is not None:
                # update-template route: the model's aggregate times the update
                # fit's gauge k_i = tau_i*T_i*G_i; no f_theta derivative needed.
                _sc = torch.as_tensor(np.asarray(scale, dtype=np.float32),
                                      device=msg_learned.device).ravel()
                msg_learned = msg_learned.ravel()[:n_neurons] * _sc[:n_neurons]
            elif _msg_i_through_f_theta(model):
                msg_learned = _msg_through_f_theta(model, pred, in_features, n_neurons,
                                                   tau=tau_fit)
            true_all.append(to_numpy(msg_true).ravel())
            learned_all.append(to_numpy(msg_learned).ravel()[:n_neurons])
    if was_training:
        model.train()

    return np.concatenate(true_all), np.concatenate(learned_all)


def _msg_i_through_f_theta(model) -> bool:
    """True for a model whose message only reaches the trajectory through a
    learned update f_theta(v, a, msg, exc) -- the GNN. A known-ODE's msg is the
    physical message already."""
    core = getattr(model, "_orig_mod", model)
    return hasattr(core, "f_theta") and hasattr(core, "_run_mlp") and \
        getattr(core, "a", None) is not None


def msg_i_estimator(model) -> str:
    return "through_f_theta" if _msg_i_through_f_theta(model) else "forward"


def _msg_through_f_theta(model, pred, in_features, n_neurons, tau=None):
    """The message a GNN really passes, read through its own update in voltage
    units:

        msg_eff_i = tau_i * [ f_theta(a_i, v_i, msg_i, exc_i) - f_theta(a_i, v_i, 0, exc_i) ]
        tau_i     = -1 / (d f_theta / d v_i)      (central difference at this frame)

    The raw aggregate carries a per-neuron gain gamma_i that f_theta undoes and
    the trajectory never sees; the difference above is gamma-free, and a perfect
    model returns the generator's msg exactly.

    The baseline is taken at the actual v_i, so V_rest is not needed. Neurons
    whose leak slope is not negative read nan and are dropped.

    `tau` (n_neurons,), when given, is the per-neuron leak time constant fitted
    over the activity range (the one tau_R2 reads), clipped like derive_tau;
    the per-frame central difference is the fallback and is fragile wherever
    f_theta is locally flat in v.
    """
    core = getattr(model, "_orig_mod", model)
    emb_dim = int(core.a.shape[1])
    msg_col = 1 + emb_dim
    with torch.no_grad():
        x0 = in_features.clone()
        x0[:, msg_col] = 0.0
        f0 = core._run_mlp(core.f_theta, x0)
        v = in_features[:, 0]
        delta = 1e-2 * max(float(v.std()), 1e-3)
        xp = in_features.clone(); xp[:, 0] = v + delta
        xm = in_features.clone(); xm[:, 0] = v - delta
        dfdv = (core._run_mlp(core.f_theta, xp) - core._run_mlp(core.f_theta, xm)) / (2 * delta)
        d = (pred - f0).ravel()[:n_neurons]
        if tau is None:
            dfdv = dfdv.ravel()[:n_neurons]
            tau = torch.where(dfdv < -1e-6, -1.0 / dfdv, torch.full_like(dfdv, float("nan")))
        return d * tau.to(d.device, d.dtype)[:n_neurons]


def compute_reversal_metrics(model, ode_params, config=None, edges=None, x_ts=None):
    """Recovery of the per-edge reversal potential E_ij, true vs learned.

    Conductance-generated data only; returns None on current data (no E_ij), so
    the caller draws no panel. Both sides are compared per edge,
    `where(edge_is_inh, E_inh[dst], E_exc[dst])`, whatever granularity each stores.

    Returns a dict with:
      rmse    root-mean-square error of learned minus true E_ij, in the same
              VOLTAGE UNITS as the dataset's voltages (flyvis voltages are O(1)
              about 0, and the margin/global reversals sit near +24.8 / -14.5)
      r2      coefficient of determination of learned E_ij against true E_ij
      slope   slope of the least-squares fit of learned on true
      n_edges how many edges entered the comparison
      true, learned   the two (n_edges,) numpy arrays, for the scatter

    Only for models holding E as a parameter (`get_learned_reversal_per_edge`);
    a GNN's E_ij comes from :func:`extract_template_params`, so this returns None
    for it, as it does when either side has no reversals.
    """
    import numpy as np

    gt_rev = getattr(ode_params, "reversal_per_edge", None)
    if gt_rev is None or getattr(ode_params, "E_exc", None) is None:
        return None
    # Under torch.compile the parameters live on the wrapped module.
    core = getattr(model, "_orig_mod", model)
    # a GNN holds no reversal parameter (see extract_template_params).
    if not hasattr(core, "get_learned_reversal_per_edge"):
        return None

    with torch.no_grad():
        true = to_numpy(gt_rev()).ravel()
        learned = to_numpy(
            core.get_learned_reversal_per_edge(ode_params.edge_index)).ravel()
        # per-neuron arrays too (one E_exc / E_inh per postsynaptic cell) for
        # plot_reversal_scatter; the RMSE is per edge, weighted by in-degree.
        l_exc, l_inh = core.get_learned_reversals()
        per_neuron = dict(
            true_exc=to_numpy(ode_params.E_exc).ravel(),
            true_inh=to_numpy(ode_params.E_inh).ravel(),
            learned_exc=to_numpy(l_exc).ravel(),
            learned_inh=to_numpy(l_inh).ravel(),
            type_index=(to_numpy(core.type_index).ravel()
                        if getattr(core, "type_index", None) is not None else None),
        )
        # untargeted neurons keep their initial reversals; flagged, not filtered.
        _dst = to_numpy(ode_params.edge_index[1]).ravel()
        # per-edge cell type = the postsynaptic cell's, which E belongs to.
        if per_neuron["type_index"] is not None:
            _ti = per_neuron["type_index"]
            per_neuron["edge_type"] = _ti[_dst % _ti.size]
        _targeted = np.zeros(per_neuron["true_exc"].size, dtype=bool)
        _targeted[np.unique(_dst) % _targeted.size] = True
        per_neuron["targeted"] = _targeted

    n = int(min(true.size, learned.size))
    true, learned = true[:n], learned[:n]
    ok = np.isfinite(true) & np.isfinite(learned)
    if ok.sum() < 2:
        return None
    true, learned = true[ok], learned[ok]

    m = recovery_param_metrics(true, learned)
    return {
        "rmse": float(np.sqrt(np.mean((learned - true) ** 2))),
        "r2": float(m["r2"]),
        "slope": float(m["slope"]),
        "n_edges": int(true.size),
        "true": true,
        "learned": learned,
        **per_neuron,
    }


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
#  Calcium / time-series spectral comparison
# ------------------------------------------------------------------ #
# used by the zebrafish calcium figures (power spectra, best-match scoring).


def fft_power_spectrum(
    x: np.ndarray,
    dt: float = 1.0,
    *,
    axis: int = -1,
    detrend: bool = True,
    window: str = "hann",
    one_sided: bool = True,
):
    """Per-trace FFT power spectrum.

    Parameters
    ----------
    x : np.ndarray
        Time series. Frequencies are computed along ``axis``.
    dt : float
        Sample interval in seconds. ``1/dt`` is the sampling rate.
    axis : int
        Time axis. Default ``-1``.
    detrend : bool
        Subtract the per-trace mean before the FFT so the DC bin doesn't
        dominate (and the per-frame baseline drift in ΔF/F doesn't leak
        into the lowest few bins).
    window : ``"hann"`` | ``"hamming"`` | ``None``
        Optional tapering window. ``"hann"`` is sensible for irregularly
        sampled ZAPBench-style ΔF/F (no implicit periodicity assumption).
    one_sided : bool
        Keep only non-negative frequencies (``rfft`` semantics).

    Returns
    -------
    freqs : np.ndarray, shape ``(F,)``
        Frequency bin centres in Hz.
    power : np.ndarray, same shape as ``x`` except the time axis is
        replaced by ``F``. Single-sided power $|X(f)|^2$ (the windowed,
        detrended FFT magnitude squared); not normalised — this is the
        raw spectrum so callers can pick their own normalisation.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.moveaxis(x, axis, -1)
    T = int(x.shape[-1])
    if detrend:
        x = x - x.mean(axis=-1, keepdims=True)
    if window == "hann":
        w = np.hanning(T)
    elif window == "hamming":
        w = np.hamming(T)
    else:
        w = None
    if w is not None:
        x = x * w
    if one_sided:
        X = np.fft.rfft(x, n=T, axis=-1)
        freqs = np.fft.rfftfreq(T, d=dt)
    else:
        X = np.fft.fft(x, n=T, axis=-1)
        freqs = np.fft.fftfreq(T, d=dt)
    power = (X.real ** 2 + X.imag ** 2)
    power = np.moveaxis(power, -1, axis)
    return freqs, power


def _normalise_pdf(p: np.ndarray, axis: int = -1, eps: float = 1e-12):
    """Normalise the spectrum to a probability mass function along
    ``axis`` so spectral-distance metrics that assume a PMF are
    well-defined.

    Used internally by :func:`spectrum_distance`. Zero or negative
    inputs (numerical noise) are clipped before normalisation.
    """
    p = np.maximum(p, 0.0)
    s = p.sum(axis=axis, keepdims=True)
    return p / (s + eps)


def spectrum_distance(
    p: np.ndarray,
    q: np.ndarray,
    *,
    metric: str = "l2",
    axis: int = -1,
):
    """Distance between two power spectra ``p`` and ``q``.

    Both inputs should be the power output of :func:`fft_power_spectrum`
    (any non-negative array works). The arrays must have the same shape
    along ``axis`` (the frequency axis); other axes broadcast.

    Parameters
    ----------
    metric : ``"l2"`` | ``"l1"`` | ``"cosine"`` | ``"jsd"``
        - ``"l2"``: Euclidean distance over a log-1+ transform of the
          unit-sum spectrum (matches what eye balls do — low-frequency
          differences dominate without small numerical bins exploding).
        - ``"l1"``: total-variation distance after unit-sum
          normalisation. Cheap, bounded in ``[0, 2]``.
        - ``"cosine"``: ``1 - dot(p, q) / (||p|| ||q||)``. Insensitive
          to absolute amplitude — good for comparing shape.
        - ``"jsd"``: Jensen–Shannon divergence on the unit-sum
          normalised spectra (symmetric, bounded ``[0, ln 2]``).

    Returns
    -------
    np.ndarray
        Distance scalar per non-axis index. Broadcasts over the
        non-frequency axes of ``p`` and ``q``.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if metric == "cosine":
        num = (p * q).sum(axis=axis)
        denom = (np.sqrt((p * p).sum(axis=axis))
                 * np.sqrt((q * q).sum(axis=axis)))
        return 1.0 - num / (denom + 1e-12)
    pn = _normalise_pdf(p, axis=axis)
    qn = _normalise_pdf(q, axis=axis)
    if metric == "l1":
        return np.abs(pn - qn).sum(axis=axis)
    if metric == "l2":
        return float(np.linalg.norm(
            np.log1p(pn) - np.log1p(qn), axis=axis,
        )) if pn.ndim == 1 else np.linalg.norm(
            np.log1p(pn) - np.log1p(qn), axis=axis,
        )
    if metric == "jsd":
        m = 0.5 * (pn + qn)
        def _kl(a, b):
            mask = a > 0
            r = np.zeros_like(a)
            r[mask] = a[mask] * (np.log(a[mask]) - np.log(b[mask] + 1e-12))
            return r.sum(axis=axis)
        return 0.5 * _kl(pn, m) + 0.5 * _kl(qn, m)
    raise ValueError(
        f"spectrum_distance: unknown metric {metric!r}; expected one of "
        f"'l2', 'l1', 'cosine', 'jsd'.")


def best_observed_match(
    obs: np.ndarray,
    model: np.ndarray,
    dt: float,
    *,
    metric: str = "cosine",
    band_hz: tuple | None = None,
    return_scores: bool = False,
):
    """For every modelled trace, find the observed trace whose power
    spectrum is the closest match.

    Used to search the ~70 k observed-neuron pool from a recording for
    the cells that best explain each modelled cell, regardless of which
    481 neurons were anatomy-matched at training time.

    Parameters
    ----------
    obs : np.ndarray, shape ``(N_obs, T)``
        Observed traces (ΔF/F or voltage), one per row.
    model : np.ndarray, shape ``(N_model, T)``
        Modelled traces, one per row. Must have the same ``T`` as
        ``obs`` (resample or trim before calling).
    dt : float
        Sample interval in seconds. Forwarded to
        :func:`fft_power_spectrum`.
    metric : str
        Spectral-distance metric (see :func:`spectrum_distance`).
        ``"cosine"`` is shape-only; ``"l2"`` weights low-frequency
        differences more.
    band_hz : tuple ``(f_lo, f_hi)`` or None
        Restrict the comparison to a frequency band. ``None`` uses the
        full spectrum.
    return_scores : bool
        When ``True``, also return the full ``(N_model, N_obs)``
        distance matrix.

    Returns
    -------
    best_idx : np.ndarray, shape ``(N_model,)``
        For each modelled trace, the index into ``obs`` of the
        best-matching observed trace.
    best_score : np.ndarray, shape ``(N_model,)``
        The matching distance (lower = better fit).
    scores : np.ndarray, shape ``(N_model, N_obs)``
        Only when ``return_scores=True``: the full distance matrix.
    """
    obs = np.asarray(obs, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    if obs.shape[-1] != model.shape[-1]:
        raise ValueError(
            f"best_observed_match: time axis mismatch obs.T={obs.shape[-1]} "
            f"vs model.T={model.shape[-1]} — resample first.")
    freqs, p_obs = fft_power_spectrum(obs, dt=dt, axis=-1)
    _, p_mod = fft_power_spectrum(model, dt=dt, axis=-1)
    if band_hz is not None:
        f_lo, f_hi = band_hz
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        p_obs = p_obs[..., mask]
        p_mod = p_mod[..., mask]
    # pairwise (N_model, N_obs, F), chunked over N_obs to bound memory.
    n_model = p_mod.shape[0]
    n_obs = p_obs.shape[0]
    scores = np.empty((n_model, n_obs), dtype=np.float64)
    CHUNK = max(1, int(2**24 // max(p_obs.shape[-1], 1)))
    for j0 in range(0, n_obs, CHUNK):
        j1 = min(j0 + CHUNK, n_obs)
        scores[:, j0:j1] = spectrum_distance(
            p_mod[:, None, :], p_obs[None, j0:j1, :],
            metric=metric, axis=-1,
        )
    best_idx = np.argmin(scores, axis=1)
    best_score = scores[np.arange(n_model), best_idx]
    if return_scores:
        return best_idx, best_score, scores
    return best_idx, best_score


# ------------------------------------------------------------------ #
#  Recovered circuit parameters — the single extraction entry point
# ------------------------------------------------------------------ #
#
# `extract_recovered_params` returns (ground truth, learned) pairs; `score_recovery`
# turns them into numbers via recovery_param_metrics; panels only read the pairs.
#
# Estimators, one per model, chosen by resolve_W_estimator:
#
#   direct          W, tau, V_rest (and E_ij on conductance data) are named
#                   parameters. Known-ODE / linear. W**2 under w_squared, since
#                   there the stored value is sqrt(conductance); tau =
#                   softplus(raw_tau); E_ij = where(edge_is_inh, E_inh[dst],
#                   E_exc[dst]) through get_learned_reversal_per_edge().
#
#   template        A GNN (any model with a g_phi). The generator's own closed form
#                   fitted to the model's message per edge, W_ij * act(v_j) *
#                   (E_ij - v_i) + C_ij on conductance data and W_ij * act(v_j) +
#                   C_ij on current data, and its update per neuron, a0 + a1 v_i +
#                   a2 msg_i + a3 stim_i, which gives tau, V_rest and the gauge k_i
#                   that carries W into the generator's units. See the block comment
#                   above extract_template_params.
#
#   jacobian        MLP baseline, no explicit W. Effective connectivity from
#                   dF/dv averaged over frames.
#
# msg_i has one path for every family (see compute_msg_i_recovery).
#
# A quantity the model does not have is absent from the pairs, never zeros.

from dataclasses import dataclass, field  # noqa: E402


RECOVERED_QUANTITIES = ("W", "tau", "V_rest", "E_ij", "msg_i", "gain", "bias")


@dataclass
class RecoveredParams:
    """The (ground truth, learned) pairs for every quantity a run can recover.

    `pairs` holds `{quantity: (gt, learned)}` as 1-D float arrays of equal length;
    a quantity the model or the dataset does not have is simply absent. Panels and
    metrics both read from here, which is what stops them describing different
    numbers.
    """

    pairs: dict = field(default_factory=dict)
    estimator: dict = field(default_factory=dict)   # 'W' -> 'template'
    correction: dict = field(default_factory=dict)  # 'W' -> the factors divided out
    valid: dict = field(default_factory=dict)       # False when gated out
    diagnostics: dict = field(default_factory=dict)  # fit_r2_median, w_scale, ...

    def get(self, quantity):
        """The (gt, learned) pair, or None when absent or gated out."""
        if not self.valid.get(quantity, True):
            return None
        return self.pairs.get(quantity)

    def __contains__(self, quantity):
        return self.get(quantity) is not None


def _is_conductance_data(ode_params) -> bool:
    """True when the generator had a reversal potential. Asked of the data, not
    the model: the same model class is trained on both kinds of data."""
    return getattr(ode_params, "E_exc", None) is not None


def resolve_W_estimator(model, ode_params, config) -> str:
    """Which readout this run uses, read off the model alone.

    "template" for a GNN with a g_phi (:func:`extract_template_params`), "direct"
    for every model whose W, tau and V_rest are named parameters (the known-ODE,
    also when it is trained on a teacher and tagged "gnn"), "jacobian" for the MLP
    baseline.
    """
    from connectome_gnn.models.utils import model_family

    family = model_family(model)
    if family == "mlp":
        return "jacobian"
    if family == "gnn" and hasattr(getattr(model, "_orig_mod", model), "g_phi"):
        return "template"
    return "direct"


class _Pair(tuple):
    """(gt, learned) as a plain tuple, plus `idx` (the entries' positions in the
    trimmed full arrays) and `n_full` (that length), which `restrict_recovered`
    needs to apply a mask after non-finite entries were dropped."""

    def __new__(cls, items, idx=None, n_full=None):
        t = super().__new__(cls, items)
        t.idx, t.n_full = idx, n_full
        return t


def _pair(gt, learned):
    """Trim two arrays to a common length and drop non-finite entries.

    Returns None when fewer than 2 entries survive (same as an absent quantity)."""
    if gt is None or learned is None:
        return None
    # inputs may be numpy, tensors or lists.
    def _np(a):
        return np.asarray(to_numpy(a) if torch.is_tensor(a) else a).ravel().astype(float)

    gt, learned = _np(gt), _np(learned)
    n = min(gt.size, learned.size)
    if n < 2:
        return None
    gt, learned = gt[:n], learned[:n]
    ok = np.isfinite(gt) & np.isfinite(learned)
    if ok.sum() < 2:
        return None
    return _Pair((gt[ok], learned[ok]), idx=np.flatnonzero(ok), n_full=n)


# --------------------------------------------------------------------------- #
#  Scoring on a subset: the score mask (config.recovery.score_mask)
#
#  The mask picks the edges and neurons a score is about (e.g. those carrying
#  signal on the optic-flow task). It restricts scoring only; the readout still
#  runs on the whole network.
# --------------------------------------------------------------------------- #

# domain of each pair; msg_i and unlisted quantities are not restricted.
EDGE_QUANTITIES = ("W", "W_uncorrected", "E_ij")
NEURON_QUANTITIES = ("tau", "V_rest", "V_rest_uncorrected", "gain", "bias")


def resolve_score_mask_path(path):
    """An absolute path as given; a relative one under the data root (the
    folder holding graphs_data/ and log/, GNN_OUTPUT_ROOT when set)."""
    if os.path.isabs(path):
        return path
    from connectome_gnn.utils import get_data_root
    return os.path.join(get_data_root(), path)


def load_score_mask(path, n_edges, n_neurons):
    """(keep_edges, keep_neurons): bool arrays of the run's edge and neuron
    counts, True = scored. Read from the .npz at `path` (see
    resolve_score_mask_path): the key `keep_edges` / `keep_neurons`, or else the
    one key starting with that name whose length is the run's count -- so one
    file can carry the same mask on several lattices (keep_edges_217,
    keep_edges_721) and each run picks its own. Either may be absent (None);
    a length that matches nothing raises rather than scoring the wrong set."""
    full = resolve_score_mask_path(path)
    if not os.path.isfile(full):
        raise FileNotFoundError(f"recovery.score_mask: no file {full}")
    z = np.load(full, allow_pickle=False)

    def _pick(stem, n):
        keys = [k for k in z.files if k == stem or k.startswith(stem + "_")]
        if not keys:
            return None
        fits = [k for k in keys if z[k].ndim == 1 and z[k].shape[0] == n]
        if len(fits) != 1:
            raise ValueError(f"recovery.score_mask {full}: {len(fits)} of the keys "
                             f"{keys} have length {n}, expected exactly one")
        return np.asarray(z[fits[0]]).astype(bool)

    keep_e, keep_n = _pick("keep_edges", int(n_edges)), _pick("keep_neurons", int(n_neurons))
    if keep_e is None and keep_n is None:
        raise ValueError(f"recovery.score_mask {full}: no keep_edges* / keep_neurons* key")
    return keep_e, keep_n


def restrict_recovered(rec, keep_edges=None, keep_neurons=None):
    """A copy of `rec` whose per-edge pairs keep only the edges with
    keep_edges True and whose per-neuron pairs keep only the neurons with
    keep_neurons True (None = that domain unrestricted).

    Scalar diagnostics (medians over every edge) are dropped, since they describe
    the unmasked network; the full-length underscore arrays and `readout` stay.
    A pair whose length is not the domain's raises rather than score unmasked.
    """

    out = RecoveredParams(pairs={}, estimator=dict(rec.estimator),
                          correction=dict(rec.correction), valid=dict(rec.valid),
                          diagnostics={k: v for k, v in rec.diagnostics.items()
                                       if k.startswith("_") or k == "readout"})
    for q, pair in rec.pairs.items():
        keep = (keep_edges if q in EDGE_QUANTITIES
                else keep_neurons if q in NEURON_QUANTITIES else None)
        if pair is None or keep is None:
            out.pairs[q] = pair
            continue
        gt, learned = np.asarray(pair[0]), np.asarray(pair[1])
        idx, n_full = getattr(pair, "idx", None), getattr(pair, "n_full", None)
        if idx is None:
            # a plain tuple: positional only when it covers the whole domain
            idx, n_full = np.arange(gt.size), gt.size
        if n_full != keep.size:
            raise ValueError(f"score mask: {q} has {n_full} entries, the mask {keep.size}")
        sel = keep[idx]
        out.pairs[q] = (_Pair((gt[sel], learned[sel]), idx=idx[sel], n_full=n_full)
                        if sel.sum() >= 2 else None)
    return out


def _extract_direct(rec, model, ode_params, edges, n_neurons):
    """Known-ODE and linear models: every quantity is a named parameter."""
    core = getattr(model, "_orig_mod", model)

    w_learned = get_model_W(core)
    w_true = getattr(ode_params, "W", None)
    if w_true is not None and w_learned is not None:
        rec.pairs["W"] = _pair(w_true, w_learned)
        rec.estimator["W"] = "direct"
        rec.diagnostics["_W_learned_full"] = np.asarray(to_numpy(w_learned)).ravel()
        rec.correction["W"] = ("W**2 (stored value is sqrt of the conductance)"
                               if getattr(core, "w_squared", False) else "none")

    if hasattr(core, "get_learned_tau") and ode_params.has_tau():
        rec.pairs["tau"] = _pair(ode_params.gt_tau(n_neurons),
                                 core.get_learned_tau()[:n_neurons])
        rec.estimator["tau"] = "direct"
    if hasattr(core, "get_learned_vrest") and ode_params.has_vrest():
        rec.pairs["V_rest"] = _pair(ode_params.gt_vrest(n_neurons),
                                    core.get_learned_vrest()[:n_neurons])
        rec.estimator["V_rest"] = "direct"

    for name, getter in (("gain", "get_learned_gain"), ("bias", "get_learned_bias")):
        if not hasattr(core, getter):
            continue
        learned = getattr(core, getter)()
        gt = getattr(ode_params, f"gt_{name}")(n_neurons)
        if learned is not None and gt is not None:
            rec.pairs[name] = _pair(gt, learned[:n_neurons])
            rec.estimator[name] = "direct"

    # edges is required here and only here: the per-edge reversal is assembled
    # from the per-neuron E_exc/E_inh through the edge index.
    if (edges is not None and _is_conductance_data(ode_params)
            and hasattr(core, "get_learned_reversal_per_edge")):
        rec.pairs["E_ij"] = _pair(ode_params.reversal_per_edge(),
                                  core.get_learned_reversal_per_edge(edges))
        rec.estimator["E_ij"] = "direct"
        rec.correction["E_ij"] = "where(edge_is_inh, E_inh[dst], E_exc[dst])"


def extract_recovered_params(model, ode_params, config=None, edges=None, x_ts=None,
                             device=None, n_neurons=None,
                             need=RECOVERED_QUANTITIES) -> RecoveredParams:
    """Dispatcher: read the learned circuit parameters out of a trained model, paired
    with their ground truth, by the route resolve_W_estimator picks from the model.

      "template"   a GNN with a g_phi -> extract_template_params (edges and x_ts
                   required). The generator's family and activation come from
                   `ode_params`, not from an argument.
      "direct"     a known-ODE: W, tau, V_rest (and E_ij) are its named parameters.
      "jacobian"   the MLP baseline: only labelled here; its dense effective
                   connectivity is computed by its own caller.

    Returns a RecoveredParams of (gt, learned) arrays, not metrics; score_recovery
    turns them into the `<key>_<stat>` lines (its docstring lists them). A quantity
    that cannot be computed is absent, never zero-filled. `need` restricts the direct
    route's msg_i pass; the template readout computes everything.
    """
    if ode_params is None:
        return RecoveredParams()
    if n_neurons is None:
        n_neurons = int(getattr(model, "a", np.zeros((0, 0))).shape[0]) or None

    estimator = resolve_W_estimator(model, ode_params, config)
    # GNN: the template readout, uncaught -- there is no other estimator to fall
    # back to; the callers turn a failure into a ReadoutError.
    if estimator == "template":
        if edges is None or x_ts is None:
            raise ValueError("extract_recovered_params: a GNN is read by the template "
                             "readout, which needs `edges` and `x_ts`")
        return extract_template_params(model, ode_params, config=config, edges=edges,
                                       x_ts=x_ts, device=device, n_neurons=n_neurons)

    rec = RecoveredParams()
    # the route, written to metrics.txt as `readout:` (the template sets its own)
    rec.diagnostics["readout"] = estimator
    try:
        if estimator == "direct":
            _extract_direct(rec, model, ode_params, edges, n_neurons)
        else:
            # jacobian: computed by the MLP baseline's own caller (a dense n x n matrix)
            rec.estimator["W"] = estimator
    except Exception as exc:
        # caught (this runs in the training loop) but recorded and printed
        rec.diagnostics["extraction_error"] = f"{type(exc).__name__}: {exc}"
        print(f"\033[91mextract_recovered_params({estimator}) failed: "
              f"{type(exc).__name__}: {exc}\033[0m")

    # msg_i: the model's per-neuron message against the generator's, on either family
    if "msg_i" in need and x_ts is not None:
        try:
            out = compute_msg_i_recovery(model, ode_params, x_ts, edges, device)
        except Exception as exc:
            rec.diagnostics["msg_i_error"] = f"{type(exc).__name__}: {exc}"
            print(f"\033[91mmsg_i extraction failed: {type(exc).__name__}: {exc}\033[0m")
            out = None
        if out is None:
            rec.diagnostics.setdefault(
                "msg_i_error", "compute_msg_i_recovery returned None (no ground-truth W)")
            print("\033[93mmsg_i: compute_msg_i_recovery returned None\033[0m")
        else:
            pair = _pair(out[0], out[1])
            if pair is None:
                _t, _l = np.asarray(out[0]).ravel(), np.asarray(out[1]).ravel()
                rec.diagnostics["msg_i_error"] = (
                    f"empty pair: true n={_t.size} finite={int(np.isfinite(_t).sum())}, "
                    f"learned n={_l.size} finite={int(np.isfinite(_l).sum())}")
                print(f"\033[93mmsg_i: {rec.diagnostics['msg_i_error']}\033[0m")
            else:
                rec.pairs["msg_i"] = pair
                rec.estimator["msg_i"] = msg_i_estimator(model)
                rec.correction["msg_i"] = (
                    "tau_i * [f_theta(v_i, msg_i) - f_theta(v_i, 0)], tau_i = -1/dftheta_dv"
                    if rec.estimator["msg_i"] == "through_f_theta" else "model forward, raw aggregate")
    elif "msg_i" in need:
        print("\033[93mmsg_i skipped: x_ts is None\033[0m")

    rec.pairs = {k: v for k, v in rec.pairs.items() if v is not None}
    return rec


def _thresh_for(quantity, config):
    r = getattr(config, "recovery", None)
    return {
        "W": getattr(r, "W_outlier_thresh", W_OUTLIER_THRESH),
        "tau": getattr(r, "tau_outlier_thresh", TAU_OUTLIER_THRESH),
        "V_rest": getattr(r, "V_rest_outlier_thresh", VREST_OUTLIER_THRESH),
        "E_ij": getattr(r, "Eij_outlier_thresh", 5.0),
        "msg_i": getattr(r, "msg_i_outlier_thresh", 1.0),
    }.get(quantity)          # None -> no filtering


# --------------------------------------------------------------------------- #
#  The template readout: the generator's closed form, fitted per edge
#
#  The model's own per-edge message is fitted to the form the generator used,
#
#       msg_ij  =  W_ij * act(v_j) * (E_ij - v_i)  +  C_ij
#
#  with the presynaptic shape `act` given (ode_params.gt_g_phi_func, relu on
#  flyvis) rather than searched for as PySR would, which makes the fit linear:
#
#       msg_ij  =  (W_ij * E_ij) * u  +  (-W_ij) * (u * v_i)  +  C_ij,   u = act(v_j)
#
#  a closed-form least squares per edge on the three columns [u, u * v_i, 1] on
#  conductance data; current data has no driving force, so the form is
#  W_ij * u + C_ij on the two columns [u, 1]. C_ij is the per-edge offset g_phi
#  emits at silent input, where the generator's message is exactly zero.
#
#  The gauge is per neuron. A GNN's message has no absolute scale (scale every
#  message by c and f_theta can divide it back out), so the fitted W_ij is off
#  by a per-neuron factor. The generator adds msg/tau to dv/dt and the model
#  adds df_theta/dmsg * msg, so
#
#       k_i  =  tau_i * (d f_theta / d msg)_i
#
#  converts neuron i's model message into the generator's units,
#  W_ij_generator ~= k_i * W_ij_fitted; one global gain (`Wij_gain`) cannot
#  absorb it. E_ij needs no gauge: it is the root of the message in v_i, which
#  a rescaling does not move.
# --------------------------------------------------------------------------- #

def _update_template_fit(model, config, edges, x_ts, n_neurons, device,
                         n_frames=64, seed=0):
    """The model's own update, read as the generator's, per neuron.

    Fits

        dv_i/dt  =  T_i * ( (V_i - v_i) + G_i * msg_i + f(stim_i) )

    which, with f linear, is a four-column least squares per neuron:

        pred_i  =  a0 + a1 * v_i + a2 * msg_i + a3 * stim_i
        T_i = -a1          the model's leak rate, so tau_i = 1 / T_i
        V_i = a0 / T_i     the voltage it relaxes to
        G_i = a2 / T_i     the weight it puts on its own message

    A constant inside f(stim) is indistinguishable from V_i, so V is biased on
    stimulated neurons (photoreceptors on flyvis); T and G are not.

    Returns (T, V, G, r2, a1, a0), each (N,): r2 is the fraction of the model's
    dv/dt the four columns explain (low = update not affine in msg); a1, a0 are
    the raw slope and offset that ode_params.derive_tau / derive_vrest invert.
    """
    core = getattr(model, "_orig_mod", model)
    emb_dim = int(core.a.shape[1])
    rng = np.random.default_rng(seed)
    n_total = int(x_ts.n_frames)
    idx = rng.choice(n_total, size=min(n_frames, n_total), replace=False)
    did = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)

    # normal equations per neuron, accumulated over streamed frames.
    XtX = torch.zeros(n_neurons, 4, 4, dtype=torch.float64, device=device)
    Xty = torch.zeros(n_neurons, 4, dtype=torch.float64, device=device)
    Syy = torch.zeros(n_neurons, dtype=torch.float64, device=device)
    Sy = torch.zeros(n_neurons, dtype=torch.float64, device=device)
    n_obs = 0
    with torch.no_grad():
        for k in idx:
            st = x_ts.frame(int(k)).to(device)
            pred, feats, _msg = core(st, edges, data_id=did, return_all=True)
            v = feats[:n_neurons, 0].double()
            m = feats[:n_neurons, 1 + emb_dim].double()
            stim = feats[:n_neurons, 2 + emb_dim].double()
            y = pred.reshape(-1)[:n_neurons].double()
            X = torch.stack([torch.ones_like(v), v, m, stim], dim=1)   # (N, 4)
            XtX += X.unsqueeze(2) * X.unsqueeze(1)
            Xty += X * y.unsqueeze(1)
            Syy += y * y
            Sy += y
            n_obs += 1

    # 1e-12 ridge: a never-stimulated neuron has a singular stim column, which
    # would otherwise poison T and V too.
    eye = torch.eye(4, dtype=torch.float64, device=device) * 1e-12
    beta = torch.linalg.solve(XtX + eye, Xty.unsqueeze(2)).squeeze(2)   # (N, 4)
    ss_tot = Syy - Sy ** 2 / max(n_obs, 1)
    ss_res = Syy - (beta * Xty).sum(1)
    r2 = to_numpy(torch.where(ss_tot > 0, 1.0 - ss_res / ss_tot,
                              torch.full_like(ss_tot, float("nan")))).astype(np.float64)

    b = to_numpy(beta).astype(np.float64)
    T = -b[:, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        V = np.where(T != 0, b[:, 0] / T, np.nan)
        G = np.where(T != 0, b[:, 2] / T, np.nan)
    # raw a1, a0 for the family-specific, clipped derive_tau / derive_vrest; a
    # bare 1/T blows up where T collapses.
    return T, V, G, r2, b[:, 1], b[:, 0]


def _template_gauge(model, config, ode_params, edges, x_ts, n_neurons, device,
                    n_frames=8, seed=0, gauge_tau="model", T=None, G=None,
                    slope_f=None):
    """Per neuron, the factor carrying its model message into the generator's units.

    k_i = tau_i * (d f_theta / d msg)_i, the derivative being T_i * G_i from the
    update fit when given, else autograd on f_theta; they agree when the update
    is affine in msg (the gap is reported as tmpl_dfdmsg_resid).

    `gauge_tau`: "model" uses the model's own tau (blind; W then also carries
    tau's per-neuron error); "true" uses the generator's tau, putting W in the
    units of ode_params.W (not a recovery number).

    Returns (k, dfdmsg, tau, autograd_dfdmsg), each (N,).
    """
    core = getattr(model, "_orig_mod", model)
    rng = np.random.default_rng(seed)
    n_total = int(x_ts.n_frames)
    idx = rng.choice(n_total, size=min(n_frames, n_total), replace=False)
    did = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)
    grads = []
    with torch.no_grad():
        frames = [x_ts.frame(int(k)).to(device) for k in idx]
    for st in frames:
        with torch.no_grad():
            _pred, feats, _msg = core(st, edges, data_id=did, return_all=True)
        # compute_grad_msg rebuilds its own graph, so the forward can run no_grad.
        grads.append(np.asarray(to_numpy(compute_grad_msg(core, feats, config))
                                ).ravel()[:n_neurons])
    autograd_dfdmsg = np.mean(np.stack(grads), axis=0).astype(np.float64)
    dfdmsg = autograd_dfdmsg if (T is None or G is None) else T * G

    if gauge_tau == "true":
        tau = np.asarray(ode_params.gt_tau(n_neurons), dtype=np.float64)
    elif slope_f is not None:
        # the reported (clipped) tau, not a raw 1/T: tau scales the neuron's whole
        # row of conductances, so an unclipped tail corrupts it.
        tau = np.asarray(ode_params.derive_tau(np.asarray(slope_f), n_neurons),
                         dtype=np.float64)
    elif T is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            tau = np.where(T != 0, 1.0 / T, np.nan)
    else:
        mu, sigma = compute_activity_stats(x_ts, device)
        slopes, _offsets = extract_f_theta_slopes(core, config, n_neurons, mu, sigma, device)
        tau = np.asarray(ode_params.derive_tau(np.asarray(slopes), n_neurons),
                         dtype=np.float64)
    return dfdmsg * tau, dfdmsg, tau, autograd_dfdmsg


class ReadoutError(RuntimeError):
    """The template readout failed.

    Its own class so broad `except Exception` handlers around diagnostics can let
    it through: there is no other estimator to fall back to.
    """


def extract_template_params(model, ode_params, config=None, edges=None, x_ts=None,
                            device=None, n_neurons=None, n_frames=1024, seed=0,
                            vj_quantile=0.5, min_points=8, gauge_tau="model",
                            gauge_frames=8, w_from="pooled_E",
                            t_slope=3.0, update_frames=64) -> RecoveredParams:
    """The template readout: a GNN's W, E_ij, tau, V_rest and msg_i, read out by fitting
    the generator's own form to the model's message (per edge) and update (per neuron).

    The GNN route of :func:`extract_recovered_params`, which calls it; see the block
    comment above for the equation and the gauge. Returns a `RecoveredParams` that
    :func:`score_recovery` turns into the metrics.txt lines.

    Args:
        n_frames: real frames per edge for the fit, one uniform random draw
            (config.recovery.template_n_frames overrides it when set). Edges
            whose sender rarely clears the v_j floor stay unmeasured at any size.
        vj_quantile: floor on the presynaptic drive, as a quantile of the
            POSITIVE activations. Frames below it carry almost no drive, so they
            say little about W or E while still weighting the least squares.
        min_points: an edge needs this many surviving frames, and a non-singular
            normal matrix, or its entry is nan.
        gauge_tau: "model" (default, blind: a recovery metric cannot use the tau
            it recovers) or "true" (the generator's tau, which W_true's units
            assume; separates the template's error from tau's, not a recovery
            number). See :func:`_template_gauge`.
        w_from: "pooled_E" reads W as (W*E) / E_pooled, "slope" as the raw
            -b2 of the fit. Conductance data only; see the pooling comment.
        t_slope: how many standard errors the driving-force coefficient must
            clear before the edge reports a reversal at all.
        update_frames: frames for the per-neuron update fit that yields tau,
            V_rest and the gauge. Capped by n_frames.
    """
    # ---- 1. inputs -----------------------------------------------------------------------------
    core = getattr(model, "_orig_mod", model)                    # the model, unwrapped from torch.compile
    n_neurons = int(core.a.shape[0]) if n_neurons is None else int(n_neurons)   # from the embedding a
    device = core.a.device if device is None else device         # where the model is evaluated
    cond = _is_conductance_data(ode_params)                      # True: conductance-generated data (E_exc)
    rec = RecoveredParams()                                      # the result: pairs, estimators, gates
    rec.diagnostics["readout"] = "template"                      # "readout: template" in metrics.txt

    _rc = getattr(config, "recovery", None)                      # the config's recovery: section
    _nf = getattr(_rc, "template_n_frames", None)                # recovery.template_n_frames, if set
    n_frames = int(_nf) if _nf else n_frames                     # real frames drawn for the per-edge fit
    frame_filter = bool(getattr(_rc, "template_frame_filter", True))   # recovery.template_frame_filter
    E_all = int(edges.shape[1])                                  # number of edges
    n_total = int(x_ts.n_frames)                                 # frames in the training trajectory
    _ends = to_numpy(edges).reshape(2, -1).astype(np.int64)      # edge list, (2, E)
    src_all, dst_all = _ends[0], _ends[1]                        # presynaptic j, postsynaptic i
    W_all = to_numpy(get_model_W(core)).ravel().astype(np.float64)   # learned W per edge (W**2 if w_squared)

    # ---- 2. helpers: the per-edge least-squares sums -------------------------------------------
    def _gt(v_):
        return np.asarray(ode_params.gt_g_phi_func(v_), dtype=np.float64).reshape(v_.shape)

    # columns [u, u*v_i, 1] with u = act(v_j), target the model's message
    # W_model*g_phi; the 1-column is the offset C_ij g_phi emits at silent input.
    def _sums(u_, vi_, msg_, floor_):
        """The ten per-edge sums (over frames) the normal equations are built from;
        sums add across samples, so frames and edges can be streamed."""
        keep_ = np.ones(u_.shape, dtype=bool) if floor_ is None else u_ > max(floor_, 1e-6)
        y_ = np.where(keep_, msg_, 0.0)
        a1 = np.where(keep_, u_, 0.0)
        a2 = np.where(keep_, u_ * vi_, 0.0)
        a3 = keep_.astype(np.float64)
        return dict(n=keep_.sum(axis=1).astype(np.float64),
                    S11=(a1 * a1).sum(1), S12=(a1 * a2).sum(1), S22=(a2 * a2).sum(1),
                    S13=(a1 * a3).sum(1), S23=(a2 * a3).sum(1),
                    S1y=(a1 * y_).sum(1), S2y=(a2 * y_).sum(1),
                    S3y=(a3 * y_).sum(1), Syy=(y_ * y_).sum(1))

    _SUM_KEYS = ("n", "S11", "S12", "S22", "S13", "S23", "S1y", "S2y", "S3y", "Syy")

    def _block_sums(order, volt_f, floor_):
        """`_sums` for the edges `order` (positions in the caller's edge array),
        in that order, one block of edges at a time."""
        step = max(1, TEMPLATE_BLOCK_PAIRS // max(int(volt_f.shape[0]), 1))
        parts = []
        for lo in range(0, len(order), step):
            idx = order[lo:lo + step]
            vi_t, vj_t, g_t = _g_phi_on_edges(
                core, config, torch.as_tensor(src_all[idx], device=device),
                torch.as_tensor(dst_all[idx], device=device), volt_f)
            vj_b = to_numpy(vj_t).astype(np.float64)
            parts.append(_sums(_gt(vj_b), to_numpy(vi_t).astype(np.float64),
                               W_all[idx][:, None] * to_numpy(g_t).astype(np.float64),
                               floor_))
            del vi_t, vj_t, g_t
        if not parts:
            return {k_: np.zeros(0) for k_ in _SUM_KEYS}
        return {k_: np.concatenate([p_[k_] for p_ in parts]) for k_ in _SUM_KEYS}

    # ---- 3. frames and the activity filter -----------------------------------------------------
    # one draw of n_frames frames shared by every edge; the filter threshold
    # (floor) is the median positive act(v_j) over (edge, frame) pairs
    # (out-degree weighted); |v_i|'s 99th percentile is the voltage scale.
    _n_cells = int(x_ts.voltage.shape[1])
    out_deg = np.bincount(src_all, minlength=_n_cells)
    in_deg = np.bincount(dst_all, minlength=_n_cells)

    sel, _fidx = _observed_plan(E_all, E_all, n_total, n_frames, seed)
    eid = np.asarray(sel).astype(np.int64)
    n_e = int(eid.size)
    i_ids = dst_all[eid]
    _vf = _frame_voltages(x_ts, _fidx, device)                  # (F, N)
    _vf_np = to_numpy(_vf)

    floor = _pooled_quantile(_gt(_vf_np.astype(np.float64)), out_deg, vj_quantile,
                             positive=True)
    floor = 0.0 if floor is None else floor
    _vi_scale = _pooled_quantile(np.abs(_vf_np), in_deg, 99, percentile=True)
    _vi_scale = float("nan") if _vi_scale is None else _vi_scale

    acc = _block_sums(eid, _vf, floor if frame_filter else None)   # the slow step: g_phi on every edge
    n_used = acc["n"]
    S11, S12, S22 = acc["S11"], acc["S12"], acc["S22"]
    S13, S23, S33 = acc["S13"], acc["S23"], acc["n"]
    S1y, S2y, Syy = acc["S1y"], acc["S2y"], acc["Syy"]
    S3y = acc["S3y"]

    # ---- 4. per-edge fit of the model's own family ---------------------------------------------
    def _solve(mats, rhs, good):
        """Per-edge least squares from stacked normal equations, nan where singular."""
        out = np.full(rhs.shape, np.nan)
        if good.any():
            out[good] = np.linalg.solve(mats[good], rhs[good][:, :, None])[:, :, 0]
        return out

    with np.errstate(divide='ignore', invalid='ignore'):
        if cond:
            # conductance data: W*u*(E - v_i) + C, three columns -> b1 = W*E, b2 = -W, b3 = C
            _M = np.stack([np.stack([S11, S12, S13], -1),
                           np.stack([S12, S22, S23], -1),
                           np.stack([S13, S23, S33], -1)], -2)      # (E, 3, 3)
            _r = np.stack([S1y, S2y, S3y], -1)
            _det = np.linalg.det(_M)
            ok = (n_used >= min_points) & np.isfinite(_det) & (np.abs(_det) > 1e-18)
            _b = _solve(_M, _r, ok)
            b1, b2, b3 = _b[:, 0], _b[:, 1], _b[:, 2]              # W*E, -W, offset
            W_fit = -b2
            ss_res = Syy - b1 * S1y - b2 * S2y - b3 * S3y
            # E only where b2 clears t_slope standard errors (else it is noise)
            sigma2 = np.where(n_used > 3, ss_res / np.maximum(n_used - 3, 1), np.nan)
            _v11 = np.full(n_e, np.nan)
            _v12 = np.full(n_e, np.nan)
            _v22 = np.full(n_e, np.nan)
            if ok.any():
                _inv = np.linalg.inv(_M[ok])
                _v11[ok], _v12[ok], _v22[ok] = _inv[:, 0, 0], _inv[:, 0, 1], _inv[:, 1, 1]
            se_b2 = np.sqrt(np.maximum(sigma2 * _v22, 0.0))
            t_b2 = np.where(se_b2 > 0, np.abs(b2) / se_b2, np.nan)
            E_fit = np.where((np.abs(W_fit) > 1e-12) & (t_b2 >= t_slope),
                             b1 / W_fit, np.nan)
            # the W >= 0 constrained solution, and se(E) by the delta method:
            #   var(E) = sigma^2/b2^2 * [V11 + (b1/b2)^2 V22 - 2 (b1/b2) V12]
            _binds = ~(b2 < 0)                      # free fit wanted W <= 0
            W_nn = np.where(_binds, 0.0, -b2)
            E_nn = np.where(_binds, np.nan, b1 / np.where(_binds, 1.0, -b2))
            _ratio = np.where(np.abs(b2) > 1e-12, b1 / b2, np.nan)
            se_E = np.sqrt(np.maximum(
                sigma2 / np.maximum(b2 ** 2, 1e-24)
                * (_v11 + _ratio ** 2 * _v22 - 2.0 * _ratio * _v12), 0.0))
            se_E = np.where(np.isfinite(E_fit), se_E, np.nan)
        else:
            # current data: W*u + C, two columns -> b1 = W, b3 = C (b2 = 0, kept for shared code)
            _den = S11 * S33 - S13 ** 2
            ok = (n_used >= min_points) & (_den > 0)
            b1 = np.where(ok, (S33 * S1y - S13 * S3y) / _den, np.nan)
            b3 = np.where(ok, (S11 * S3y - S13 * S1y) / _den, np.nan)
            b2 = np.zeros(n_e)
            W_fit, E_fit = b1, np.full(n_e, np.nan)
            se_E = np.full(n_e, np.nan)
            W_nn, E_nn = b1, np.full(n_e, np.nan)
            _binds = np.zeros(n_e, dtype=bool)
            ss_res = Syy - b1 * S1y - b3 * S3y
        # per-edge fit R2, uncentred (against zero): no drive must mean no message
        fit_r2 = np.where(ok & (Syy > 0), 1.0 - ss_res / Syy, np.nan)

        # ---- 5. cross-check: both families' forms on the same message ------------------------------
        # current form [u, 1] (on current data it repeats step 4)
        _den2 = S11 * S33 - S13 ** 2
        ok2 = (n_used >= min_points) & (_den2 > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            c1 = np.where(ok2, (S33 * S1y - S13 * S3y) / _den2, np.nan)
            c3 = np.where(ok2, (S11 * S3y - S13 * S1y) / _den2, np.nan)
        cur_r2 = np.where(ok2 & (Syy > 0), 1.0 - (Syy - c1 * S1y - c3 * S3y) / Syy,
                          np.nan)
        # constrained (W >= 0) conductance form: the free fit, or the current form where it binds
        cond_nn_r2 = np.where(_binds, cur_r2, fit_r2)
        if cond:
            cond_r2 = fit_r2
            cb1, cb2 = b1, b2
            t_cond = t_b2
        else:
            # conductance form [u, u*v_i, 1] on current data; t_cond = |slope in v_i| / its SE
            _M3 = np.stack([np.stack([S11, S12, S13], -1),
                            np.stack([S12, S22, S23], -1),
                            np.stack([S13, S23, S33], -1)], -2)
            _r3 = np.stack([S1y, S2y, S3y], -1)
            _det3 = np.linalg.det(_M3)
            ok3 = (n_used >= min_points) & np.isfinite(_det3) & (np.abs(_det3) > 1e-18)
            _b3 = _solve(_M3, _r3, ok3)
            cond_r2 = np.where(
                ok3 & (Syy > 0),
                1.0 - (Syy - _b3[:, 0] * S1y - _b3[:, 1] * S2y - _b3[:, 2] * S3y) / Syy,
                np.nan)
            cb1, cb2 = _b3[:, 0], _b3[:, 1]
            _ss3 = Syy - _b3[:, 0] * S1y - _b3[:, 1] * S2y - _b3[:, 2] * S3y
            _s2 = np.where(n_used > 3, _ss3 / np.maximum(n_used - 3, 1), np.nan)
            _v = np.full(n_e, np.nan)
            if ok3.any():
                _v[ok3] = np.linalg.inv(_M3[ok3])[:, 1, 1]
            _se = np.sqrt(np.maximum(_s2 * _v, 0.0))
            t_cond = np.where(_se > 0, np.abs(cb2) / _se, np.nan)

        # the reversal potential E (the postsynaptic voltage where the synaptic current
        # W*relu(v_j)*(E - v_i) changes sign) the conductance form needs, E = -cb1/cb2;
        # on current data it lands far outside the visited voltages
        with np.errstate(divide='ignore', invalid='ignore'):
            E_cond_form = np.where(cb2 != 0, -cb1 / cb2, np.inf * np.sign(cb1))

    # ---- 6. conductance data only: E pooled per neuron, W = (W*E) / E_pooled -------------------
    # one reversal per neuron and sender sign in the generator, so the median of
    # each sign; W from the well-determined b1 rather than the noisier b2
    E_pooled = np.full(n_e, np.nan)
    if cond:
        good = np.isfinite(E_fit)
        for sign_mask in (E_fit > 0, E_fit < 0):
            m = good & sign_mask
            if not m.any():
                continue
            sel_idx = np.where(m)[0]
            order = np.argsort(i_ids[sel_idx], kind='stable')
            ids_s, vals_s = i_ids[sel_idx][order], E_fit[sel_idx][order]
            uniq, start = np.unique(ids_s, return_index=True)
            med = {int(i): float(np.median(v))
                   for i, v in zip(uniq, np.split(vals_s, start[1:]))}
            E_pooled[sel_idx] = np.array([med[int(i)] for i in i_ids[sel_idx]])
    with np.errstate(divide='ignore', invalid='ignore'):
        W_pooled = np.where(np.abs(E_pooled) > 1e-9, b1 / E_pooled, np.nan)
    if cond and w_from == "pooled_E":
        pct_slope = float(100.0 * np.mean(~np.isfinite(W_pooled) & np.isfinite(W_fit)))
        W_used = np.where(np.isfinite(W_pooled), W_pooled, W_fit)
    else:
        pct_slope = 100.0 if cond else float("nan")
        W_used = W_fit

    # ---- 7. per-neuron update fit and the gauge: W into the generator's units ------------------
    # f_theta ~ a0 + a1 v + a2 msg + a3 I per neuron -> T, V, G, a1 (slope_f), a0 (offset_f)
    T, V_fit, G_fit, update_r2, slope_f, offset_f = _update_template_fit(
        core, config, edges, x_ts, n_neurons, device,
        n_frames=min(n_frames, update_frames), seed=seed)
    k, dfdmsg, tau_used, autograd_dfdmsg = _template_gauge(
        core, config, ode_params, edges, x_ts, n_neurons, device,
        n_frames=gauge_frames, seed=seed, gauge_tau=gauge_tau, T=T, G=G_fit,
        slope_f=slope_f)
    W_learned = k[i_ids] * W_used                                # k_i = tau_i * d f_theta / d msg_i

    # ---- 8. the W pair and its gate ------------------------------------------------------------
    # back into the caller's edge order (eid was a permutation), aligned with ode_params.W
    def _scatter(v, fill=np.nan):
        out = np.full(int(edges.shape[1]), fill, dtype=np.float64)
        out[eid] = v
        return out

    W_learned, E_learned = _scatter(W_learned), _scatter(E_fit)
    fit_r2_full, slope_full = _scatter(fit_r2), _scatter(b2)
    gt_W = np.asarray(ode_params.effective_true_weights(
        to_numpy(ode_params.W), to_numpy(edges), n_neurons))

    gate = getattr(getattr(config, "recovery", None), "gate_fit_r2", 0.9)
    r2_med = float(np.nanmedian(fit_r2_full)) if np.isfinite(fit_r2_full).any() else float("nan")
    # |W| on conductance data (the generator's W >= 0); signed on current data
    if cond:
        W_learned = np.abs(W_learned)
    rec.pairs["W"] = _pair(gt_W, W_learned)
    rec.estimator["W"] = "template_fit"
    rec.correction["W"] = (
        f"msg_ij = W*act(v_j)*(E - v_i) per edge; "
        f"W_ij scaled by k_i = tau_i * dftheta_dmsg_i ({gauge_tau} tau)"
        + ("; |W| reported, the generator's conductance being non-negative"
           if cond else "; signed, the current generator's W carries the sign"))
    rec.diagnostics["_W_learned_full"] = W_learned
    rec.valid["W"] = r2_med >= gate
    # ---- 9. fit diagnostics (metrics.txt) ------------------------------------------------------
    rec.diagnostics["msg_form_r2_median"] = r2_med
    # fitted edges: at least min_points frames kept by the filter (every sampled frame when the filter is off)
    _n_fitted = int((n_used >= min_points).sum())
    rec.diagnostics["tmpl_n_edges"] = int(n_e)
    rec.diagnostics["tmpl_n_edges_fitted"] = _n_fitted
    rec.diagnostics["tmpl_pct_fitted"] = float(100.0 * _n_fitted / max(n_e, 1))
    if cond:
        _n_E = int(np.isfinite(E_fit).sum())
        rec.diagnostics["tmpl_n_E_identified"] = _n_E
        rec.diagnostics["tmpl_pct_E_identified"] = float(100.0 * _n_E / max(n_e, 1))
    rec.diagnostics["current_form_r2_median"] = (
        float(np.nanmedian(cur_r2)) if np.isfinite(cur_r2).any() else float("nan"))
    rec.diagnostics["conductance_nn_form_r2_median"] = (
        float(np.nanmedian(cond_nn_r2)) if np.isfinite(cond_nn_r2).any()
        else float("nan"))
    rec.diagnostics["Wij_pct_W_negative"] = (
        float(100.0 * np.mean(_binds[np.isfinite(b2)]))
        if np.isfinite(b2).any() else float("nan"))
    rec.diagnostics["_W_nn_full"] = _scatter(W_nn)
    rec.diagnostics["_E_nn_full"] = _scatter(E_nn)
    rec.diagnostics["conductance_form_r2_median"] = (
        float(np.nanmedian(cond_r2)) if np.isfinite(cond_r2).any() else float("nan"))
    # driving-force R2 gain per edge (conductance minus current form), mean and SD
    _gain = cond_r2 - cur_r2
    _gf = _gain[np.isfinite(_gain)]
    rec.diagnostics["driving_force_r2_gain_mean"] = (
        float(np.mean(_gf)) if _gf.size else float("nan"))
    rec.diagnostics["driving_force_r2_gain_sd"] = (
        float(np.std(_gf, ddof=1)) if _gf.size > 1 else float("nan"))
    # the other family's constants, gauged by the same k_i, for the alternate-form rollout
    if cond:
        _W_alt = k[i_ids] * c1                      # current form: msg = W*u + C
        _E_alt = np.full(n_e, np.nan)
        rec.diagnostics["alt_form_family"] = "current"
    else:
        _W_alt = k[i_ids] * (-cb2)                  # conductance form: b2 = -W
        _E_alt = E_cond_form
        rec.diagnostics["alt_form_family"] = "conductance"
    # per-edge arrays (underscored keys stay out of metrics.txt)
    rec.diagnostics["_form_cond_r2_full"] = _scatter(cond_r2)
    rec.diagnostics["_form_cur_r2_full"] = _scatter(cur_r2)
    rec.diagnostics["_form_t_b2_full"] = _scatter(t_cond)
    rec.diagnostics["_form_E_cond_full"] = _scatter(E_cond_form)
    rec.diagnostics["_form_n_used_full"] = _scatter(n_used)
    rec.diagnostics["_W_alt_full"] = _scatter(_W_alt)
    rec.diagnostics["_E_alt_full"] = _scatter(_E_alt)

    _E_CAP = 1e6
    _Ec = np.abs(E_cond_form[~np.isnan(E_cond_form)])
    _E_med = float(np.median(np.minimum(_Ec, _E_CAP * max(_vi_scale, 1e-12)))) \
        if _Ec.size else float("nan")
    rec.diagnostics["conductance_form_E_absmedian"] = _E_med
    rec.diagnostics["vi_abs_p99"] = _vi_scale
    rec.diagnostics["conductance_form_E_over_vi"] = (
        float(_E_med / _vi_scale)
        if np.isfinite(_E_med) and np.isfinite(_vi_scale) and _vi_scale > 0
        else float("nan"))
    rec.diagnostics["tmpl_frames_used"] = int(len(_fidx))
    # ---- 10. conductance data only: the E_ij pair ----------------------------------------------
    if cond:
        rec.diagnostics["msg_form_r2_median"] = r2_med
        # share of edges whose message rises with v_i, which no driving force does
        rec.diagnostics["Eij_pct_wrong_slope"] = (
            float(100.0 * np.mean(slope_full[np.isfinite(slope_full)] > 0))
            if np.isfinite(slope_full).any() else float("nan"))
        _E_true = np.asarray(to_numpy(ode_params.reversal_per_edge())).ravel()
        rec.pairs["E_ij"] = _pair(_E_true, E_learned)
        rec.estimator["E_ij"] = "template_fit"
        # no gauge: k_i cancels in E = -b1/b2
        rec.correction["E_ij"] = (
            f"E_ij = -b1/b2 from the three-column fit [act(v_j), act(v_j)*v_i, 1]; "
            f"the gauge k_i cancels in the ratio so no scale correction applies; "
            f"kept where |b2| clears {t_slope} standard errors")
        rec.valid["E_ij"] = r2_med >= gate
        # |E - E_true| over se(E): ~1 noise-limited, >> 1 a term the form lacks
        _se_full = _scatter(se_E)
        rec.diagnostics["_Eij_se_full"] = _se_full
        _fin = np.isfinite(_se_full) & np.isfinite(E_learned) & np.isfinite(_E_true)
        rec.diagnostics["Eij_se_median"] = (
            float(np.median(_se_full[_fin])) if _fin.any() else float("nan"))
        if _fin.any():
            _obs = np.abs(E_learned[_fin] - _E_true[_fin])
            rec.diagnostics["Eij_abs_err_median"] = float(np.median(_obs))
            _den = np.median(_se_full[_fin])
            rec.diagnostics["Eij_err_over_se"] = (
                float(np.median(_obs) / _den) if _den > 0 else float("nan"))
    # ---- 11. tau, V_rest and msg_i from the update fit -----------------------------------------
    # tau = -1/a1 (clipped), V_rest = -a0/a1, via the family's derive_tau / derive_vrest
    if ode_params.has_tau():
        rec.pairs["tau"] = _pair(ode_params.gt_tau(n_neurons),
                                 ode_params.derive_tau(slope_f, n_neurons))
        rec.estimator["tau"] = "update_template"
        rec.correction["tau"] = ("ode_params.derive_tau of a1, from "
                                 "pred ~ a0 + a1*v + a2*msg + a3*stim on real frames")
    if ode_params.has_vrest():
        rec.pairs["V_rest"] = _pair(ode_params.gt_vrest(n_neurons),
                                    ode_params.derive_vrest(slope_f, offset_f, n_neurons))
        rec.estimator["V_rest"] = "update_template"
        rec.correction["V_rest"] = ("ode_params.derive_vrest of (a1, a0); a0 is "
                                    "confounded with a constant inside f(stim)")
    # msg_i: the model's aggregate scaled by the same k_i as W
    if x_ts is not None:
        try:
            _mt = compute_msg_i_recovery(model, ode_params, x_ts, edges, device,
                                         scale=k)
            if _mt is not None:
                _mp = _pair(_mt[0], _mt[1])
                if _mp is not None:
                    rec.pairs["msg_i"] = _mp
                    rec.estimator["msg_i"] = "update_template"
                    rec.correction["msg_i"] = (
                        "k_i * msg_hat_i, the model's own aggregate scaled by "
                        "the update fit's gauge k_i = tau_i * T_i * G_i")
        except Exception as _exc:
            rec.diagnostics["msg_i_template_error"] = f"{type(_exc).__name__}: {_exc}"

    # ---- 12. update diagnostics, the offset, and V_rest corrected ------------------------------
    rec.diagnostics["update_form_r2_median"] = float(np.nanmedian(update_r2))
    rec.diagnostics["tmpl_G_median"] = float(np.nanmedian(G_fit))
    # T*G over autograd's df_theta/dmsg: 1 when the update is affine in msg
    with np.errstate(divide='ignore', invalid='ignore'):
        _ratio = np.where(autograd_dfdmsg != 0, (T * G_fit) / autograd_dfdmsg, np.nan)
    rec.diagnostics["tmpl_dfdmsg_over_autograd"] = float(np.nanmedian(_ratio))
    rec.diagnostics["tmpl_k_median"] = float(np.nanmedian(k))
    rec.diagnostics["_tmpl_k_full"] = np.asarray(k, dtype=np.float64)
    rec.diagnostics["tmpl_dfdmsg_median"] = float(np.nanmedian(dfdmsg))
    rec.diagnostics["tmpl_dfdmsg_absmedian"] = float(np.nanmedian(np.abs(dfdmsg)))
    rec.diagnostics["tmpl_pct_W_from_slope"] = pct_slope
    rec.diagnostics["tmpl_n_used_median"] = float(np.nanmedian(n_used))
    rec.diagnostics["tmpl_pct_unfitted"] = float(100.0 * np.mean(~np.isfinite(W_learned)))
    rec.diagnostics["tmpl_vj_floor"] = floor if frame_filter else 0.0
    rec.diagnostics["tmpl_frame_filter"] = float(frame_filter)
    # the offset k_i*C_ij per edge and summed over each neuron's fitted incoming edges
    _off_scaled = k[i_ids] * b3
    _off_full = _scatter(_off_scaled)
    _per_neuron = np.zeros(n_neurons)
    _good_off = np.isfinite(_off_scaled)
    if _good_off.any():
        np.add.at(_per_neuron, i_ids[_good_off], _off_scaled[_good_off])
    rec.diagnostics["tmpl_offset_median"] = float(np.nanmedian(np.abs(_off_full)))
    rec.diagnostics["tmpl_offset_per_neuron_median"] = float(np.median(_per_neuron))
    rec.diagnostics["tmpl_offset_per_neuron_absmedian"] = float(np.median(np.abs(_per_neuron)))
    rec.diagnostics["_tmpl_offset"] = _off_full
    rec.diagnostics["_tmpl_offset_per_neuron"] = _per_neuron
    # the reported V_rest = -a0/a1 + the neuron's incoming offset (the update hands a
    # message constant straight to V_rest); `is not None`: the key can hold None
    if rec.pairs.get("V_rest") is not None:
        _vt, _vl = rec.pairs["V_rest"]
        if _vl.size == n_neurons:
            rec.diagnostics["_V_rest_offset_corrected"] = _vl + _per_neuron
            rec.pairs["V_rest_uncorrected"] = (_vt, _vl)
            rec.pairs["V_rest"] = _pair(_vt, _vl + _per_neuron)
            rec.correction["V_rest"] = (
                "ode_params.derive_vrest of (a1, a0), plus the neuron's own "
                "incoming message offset k_i*beta_i, which the update otherwise "
                "hands straight to the resting level")
    rec.diagnostics["tmpl_pct_E_unidentified"] = (
        float(100.0 * np.mean(~np.isfinite(E_fit))) if cond else float("nan"))
    rec.diagnostics["_tmpl_E_full"] = E_learned
    rec.diagnostics["_tmpl_W_fit"] = _scatter(W_used)
    rec.diagnostics["_tmpl_fit_r2"] = fit_r2_full
    rec.diagnostics["_tmpl_k"] = k
    rec.diagnostics["_tmpl_tau"] = tau_used
    return rec


def score_Pysr_recovery(model, ode_params, config=None, edges=None, x_ts=None,
                        device=None, n_neurons=None, **kw) -> dict:
    """:func:`score_recovery` of :func:`extract_template_params`: same keys,
    statistics and outlier rule."""
    rec = extract_template_params(model, ode_params, config=config, edges=edges,
                                  x_ts=x_ts, device=device, n_neurons=n_neurons, **kw)
    return score_recovery(rec, config)


# The emitted key for each quantity. Wij/Eij/msg_i are symbol-first; tau and
# V_rest keep the spelling every consumer already reads.
_KEY = {"W": "Wij", "tau": "tau", "V_rest": "V_rest",
        "E_ij": "Eij", "msg_i": "msg_i", "gain": "gain", "bias": "bias"}


def score_recovery(rec: RecoveredParams, config=None) -> dict:
    """Score every recovered quantity through the one R2 entry point.

    The outlier threshold is decided here (:func:`_thresh_for`), not per call
    site: the headline `<q>_R2` is outlier-free, `<q>_R2_all` is not.

    ----------------------------------------------------------------------------
    Every metric this produces (the pairs come from extract_recovered_params)
    ----------------------------------------------------------------------------
    These keys reach `results/metrics.txt`, the per-slot analysis log and the LLM
    exploration instructions. The names are a contract: the instruction files
    rank slots on these spellings, so renaming one silently breaks an agentic loop.

    Quantities, and the key each is emitted under (`_KEY`):

        quantity   key       what it is
        W          Wij       per-edge synaptic weight. On conductance data this is
                             the CONDUCTANCE (non-negative); on current data it is
                             the signed current weight.
        tau        tau       per-neuron membrane time constant.
        V_rest     V_rest    per-neuron resting potential.
        E_ij       Eij       per-edge reversal potential, i.e. the voltage each
                             edge's driving force (E_ij - v_i) points toward.
                             Conductance-generated data only; absent elsewhere,
                             which is information, not a failure.
        msg_i      msg_i     the aggregated per-neuron message, sum over incoming
                             edges of W_ij * act(v_j) * (E_ij - v_i). The one
                             quantity the W/E degeneracy does not touch.
        gain       gain      per-neuron output gain, where the model has one.
        bias       bias      per-neuron output bias, where the model has one.

    For each of the above:

        <key>_R2          coefficient of determination of learned against true, on
                          the IDENTITY line (not a free-slope fit), so a scale
                          error costs R2 rather than being absorbed. Outlier-
                          filtered for W, tau and V_rest; unfiltered for Eij and
                          msg_i, which have no published tolerance band.
        <key>_slope       identity-line slope of the same scatter. 1.0 is perfect.
                          For W this is the gain: 2.4 means the learned conductance
                          is 2.4x the true one. It is the third ranking key of the
                          known-ODE exploration.
        <key>_rmse        root-mean-square error, in that quantity's own units.
                          Read it against the spread of the truth, not a fixed bar:
                          the reversals of one twin span 8 voltage units and those
                          of another span 40.
        <key>_n           how many elements entered the comparison. Context, but a
                          drop in it means pairs were trimmed or non-finite.

    Only where an outlier threshold applies (W, tau, V_rest -- see
    :func:`_thresh_for`; None for Eij and msg_i):

        <key>_R2_all       the same R2 without outlier filtering. The gap between
                           it and <key>_R2 says how much of the recovery rests on
                           a tail. Rank on the filtered one, report both.
        <key>_n_outliers   how many elements the threshold removed.
        <key>_pct_outliers the same as a percentage of <key>_n.

    Provenance, emitted whenever the extractor recorded it. Strings, not numbers:
    the same key can come from different estimators on different runs, so check
    them before comparing across runs:

        <key>_estimator   which path produced the number:
                            direct         a named parameter, read straight off.
                            template_fit   W / E_ij from the template readout's
                                           per-edge fit (extract_template_params).
                            update_template  tau / V_rest / msg_i from its
                                           per-neuron update fit.
                            edge_line_fit  per-edge line fit of msg_ij/v_j against
                                           v_i; W is -slope and E is -intercept/
                                           slope, so the two cannot disagree.
                            gain_corrected the GNN's W after dividing out the
                                           g_phi and f_theta gains.
                            jacobian       the MLP baseline's effective
                                           connectivity, a dense n x n matrix.
                            f_theta_slope  tau and V_rest read out of f_theta's
                                           local linearisation.
                            forward        msg_i, a known-ODE's own message.
                            through_f_theta  msg_i of a GNN, read through its update:
                                           tau_i * [f_theta(v_i, msg_i) -
                                           f_theta(v_i, 0)], gauge-free.
        <key>_correction  the exact algebra that was applied, e.g.
                          `W**2 (stored value is sqrt of the conductance)` or
                          `where(edge_is_inh, E_inh[dst], E_exc[dst])`.

    Always, per quantity:

        <key>_rel_err_median  median of |learned - true| / max(|true|, 1e-6) over
                              the full sample, and
        <key>_rel_err_iqr     its interquartile range. Median and IQR, never mean
                              and SD, which the heavy tails inflate.

    Only for W and msg_i, the two a GNN pins down up to a gain:

        <key>_gain        learned ~= gain * true, the least-squares factor through
                          the origin with the truth as regressor (see
                          :func:`r2_up_to_scale`). 1.0 is perfect; 2.0 means twice
                          too large. The reciprocal <true,learned>/<learned,learned>
                          is not used anywhere.
        <key>_R2_scaled   the identity-line R2 once that gain is divided out. The
                          number to steer by while <key>_R2 is deeply negative.

    Only for W, scale ignored altogether (over edges whose true weight is non-zero):

        Wij_pearson       Pearson r of learned against true. High with a low
                          Wij_R2 reads "wiring recovered, scale not".
        Wij_zscored_R2    identity-line R2 of the two z-scored vectors.

    Keys that are not per-quantity:

        Wij_R2_uncorrected  W scored before the gain correction. Says how much of
                            the recovery the correction is responsible for. Only
                            on the `gain_corrected` path -- "uncorrected" means
                            before the correction, not before outlier filtering.
        V_rest_R2_uncorrected  V_rest before the message offset is added back
                            (template readout).
        msg_form_r2_median  median per-edge R2 of the message's fitted form
                            (the template fit; `edge_line_fit` on that path). A
                            precondition, not a detail: below roughly 0.9 the
                            message is not affine in v_i, the model has not found
                            the conductance form, and the W and E beside it
                            describe nothing. GNN paths only.
        Eij_pct_wrong_slope percentage of edges whose per-edge line RISES with
                            v_i, which no conductance does. Zero on a run in the
                            conductance form; on a gated run it separates a noisy
                            fit from a systematically wrong sign. GNN paths only.
        extraction_error    present only when this function caught an exception,
                            carrying `TypeName: message`. Its presence means every
                            quantity below it is missing because the extractor
                            broke, not because the model lacks them.

    A quantity gated out emits nothing: `rec.valid[q]` False makes `rec.get(q)`
    return None, so its whole family is skipped. Absence is the signal.

    Where the numbers land, spelled identically in all three places:

        tmp_training/<key>.log   one CSV per quantity (Wij.log, tau.log, V_rest.log,
                                 Eij.log, msg_i.log), header `iteration,<key>_R2,
                                 <key>_R2_all,...` in :func:`recovery_log_columns`
                                 order; one row per training checkpoint. Read by
                                 name with :func:`training_log_read`. cluster.log
                                 and rollout.log follow the same layout for the
                                 two numbers that are not recovered parameters.
        results/metrics.txt      `key: value`, written once by `-o test_plot`
                                 through :func:`write_recovery_metrics`.
        the per-slot analysis log the LLM exploration reads: the same lines.

    Renamed on 2026-09-11 (tools/extraction_gate.py carries the map for logs
    written before): `connectivity_R2` / `W_corrected_no_outliers_R2` -> `Wij_R2`;
    `W_corrected_R2` -> `Wij_R2_all`; `connectivity_R2_scaled` -> `Wij_R2_scaled`;
    `w_scale` -> `Wij_gain`; `connectivity_pearson_r` / `W_structure_r` ->
    `Wij_pearson`; `W_zscored_R2` -> `Wij_zscored_R2`; `raw_W_R2` ->
    `Wij_R2_uncorrected`; test_plot's old `tau_R2` (unfiltered) -> `tau_R2_all`
    and `tau_no_outliers_R2` -> `tau_R2`, likewise V_rest; `Eij_n_edges` ->
    `Eij_n`; the trainer's `connectivity_r2` / `vrest_r2_clean` / `tau_r2_clean`
    columns -> `Wij_R2` / `V_rest_R2` / `tau_R2` in their own files;
    `msgi_r2.log`'s `r2_scaled` / `scale` -> `msg_i_R2_scaled` / `msg_i_gain`;
    `gnn_conductance_fit.log`'s `fit_r2_median` -> `msg_form_r2_median`.
    """
    out = {}
    for quantity, key in _KEY.items():
        pair = rec.get(quantity)
        if pair is None:
            continue
        gt, learned = pair
        m = recovery_param_metrics(gt, learned, _thresh_for(quantity, config))
        clean = _thresh_for(quantity, config) is not None
        out[f"{key}_R2"] = float(m["r2_clean"] if clean else m["r2"])
        out[f"{key}_slope"] = float(m["slope_clean"] if clean else m["slope"])
        out[f"{key}_rmse"] = float(np.sqrt(np.mean((learned - gt) ** 2)))
        out[f"{key}_n"] = int(m["n_total"])
        if clean:
            out[f"{key}_R2_all"] = float(m["r2"])
            out[f"{key}_n_outliers"] = int(m["n_outliers"])
            out[f"{key}_pct_outliers"] = float(m["pct_outliers"])
        # |learned - true| / max(|true|, 1e-6) over the full sample: median and
        # interquartile range, never mean +- SD, which the heavy tails inflate.
        out[f"{key}_rel_err_median"] = float(m["rel_err_median"])
        out[f"{key}_rel_err_iqr"] = float(m["rel_err_iqr"])
        if quantity in rec.estimator:
            out[f"{key}_estimator"] = rec.estimator[quantity]
        if quantity in rec.correction:
            out[f"{key}_correction"] = rec.correction[quantity]

    # "uncorrected" = before the gain correction (not before outlier filtering);
    # scored on R2 alone.
    unc = rec.pairs.get("W_uncorrected")
    if unc is not None:
        out["Wij_R2_uncorrected"] = float(
            recovery_param_metrics(unc[0], unc[1],
                                   _thresh_for("W", config))["r2_clean"])

    # V_rest_R2 is the offset-corrected level; this is the raw one, so the gap
    # says how much of the V_rest error is the message offset beta_i.
    _vunc = rec.pairs.get("V_rest_uncorrected")
    if _vunc is not None:
        out["V_rest_R2_uncorrected"] = float(
            recovery_param_metrics(_vunc[0], _vunc[1],
                                   _thresh_for("V_rest", config))["r2_clean"])

    # scale-free companions (learned ~= gain * true, see r2_up_to_scale).
    for quantity in ("W", "msg_i"):
        pair = rec.get(quantity)
        if pair is None:
            continue
        key = _KEY[quantity]
        s_ = r2_up_to_scale(pair[0], pair[1])
        out[f"{key}_R2_scaled"] = float(s_["r2"])
        out[f"{key}_gain"] = float(s_["gain"])
    # W structure ignoring scale, over edges whose true weight is non-zero.
    w = rec.get("W")
    if w is not None:
        gt, learned = (np.asarray(w[0]).ravel().astype(np.float64),
                       np.asarray(w[1]).ravel().astype(np.float64))
        nz = gt != 0
        gt, learned = gt[nz], learned[nz]
        if gt.size > 1 and gt.std() > 0 and learned.std() > 0:
            out["Wij_pearson"] = float(np.corrcoef(gt, learned)[0, 1])
            gz = (gt - gt.mean()) / (gt.std() + 1e-12)
            lz = (learned - learned.mean()) / (learned.std() + 1e-12)
            out["Wij_zscored_R2"] = float(recovery_param_metrics(gz, lz)["r2"])
        else:
            out["Wij_pearson"] = float("nan")
            out["Wij_zscored_R2"] = float("nan")

    # underscore-prefixed diagnostics are arrays and never reach the key-value log.
    for k, v in rec.diagnostics.items():
        if not k.startswith("_"):
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
#  One vocabulary, one writer
#
#  Every recovered-quantity number is a `<key>_<stat>` from score_recovery,
#  spelled the same in tmp_training/<key>.log (one row per checkpoint),
#  results/metrics.txt and the per-slot analysis log (see score_recovery).

# --------------------------------------------------------------------------- #

RECOVERY_KEYS = tuple(_KEY.values())

# Column order of tmp_training/<key>.log, and the order results/metrics.txt lists
# them in. Shared stats first, then what only some quantities have.
_COMMON_STATS = ("R2", "R2_all", "slope", "rmse", "n", "n_outliers", "pct_outliers",
                 "rel_err_median", "rel_err_iqr")
_EXTRA_STATS = {
    "Wij":   ("R2_scaled", "gain", "pearson", "zscored_R2", "R2_uncorrected"),
    "Eij":   ("pct_wrong_slope",),
    "msg_i": ("R2_scaled", "gain"),
    "V_rest": ("R2_uncorrected",),
}

# diagnostics that are not `<key>_<stat>`, logged with the quantity they describe;
# all three come from the per-edge fit that produces E_ij:
#
#   msg_form_r2_median          median per-edge R2 of the model's own family
#   conductance_form_r2_median  the same message under W*relu(v_j)*(E - v_i)
#   current_form_r2_median      ... and under W*relu(v_j) (no driving force)
_EXTRA_GLOBAL = {
    "Eij": ("msg_form_r2_median", "conductance_form_r2_median",
            "current_form_r2_median"),
}


def recovery_log_columns(key):
    """The columns of tmp_training/<key>.log, in file order.

    `<key>_<stat>` for the per-quantity statistics, then any `_EXTRA_GLOBAL`
    diagnostics under their own names -- those are keyed globally in `scored`,
    not per quantity, so they are NOT prefixed.
    """
    return (tuple(f"{key}_{st}" for st in _COMMON_STATS + _EXTRA_STATS.get(key, ()))
            + _EXTRA_GLOBAL.get(key, ()))


def _fmt_metric(v):
    if v is None:
        return "nan"
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "nan" if not np.isfinite(v) else f"{float(v):.6f}"
    return str(v)


def training_log_append(log_dir, name, iteration, row, note=None):
    """Append one row to tmp_training/<name>.log, writing the header first.

    `row` is an ordered {column: value}; the header is `iteration,` + its keys.
    Readers parse by NAME (training_log_read), never by position, so a column
    can be added without shifting anyone.

    `note` is a provenance descriptor (estimator, correction) written once as a
    `#` line above the header, since a column means a different quantity under a
    different readout; `training_log_read` skips `#` lines.
    """
    tmp = os.path.join(log_dir, "tmp_training")
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, f"{name}.log")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f:
            if note:
                f.write(f"# {note}\n")
            f.write("iteration," + ",".join(row.keys()) + "\n")
    with open(path, "a") as f:
        f.write(f"{int(iteration)}," + ",".join(_fmt_metric(v) for v in row.values()) + "\n")


def training_log_read(log_dir, name):
    """Read tmp_training/<name>.log into {column: np.ndarray} by header name.

    None when the file is missing or has no data rows. Unparseable cells read
    as NaN; a row shorter than the header is padded with NaN.
    """
    path = os.path.join(log_dir, "tmp_training", f"{name}.log")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if len(lines) < 2 or not lines[0].startswith("iteration"):
        return None
    cols = lines[0].split(",")
    rows = []
    for ln in lines[1:]:
        parts = ln.split(",")
        vals = []
        for i in range(len(cols)):
            try:
                vals.append(float(parts[i]) if i < len(parts) else np.nan)
            except ValueError:
                vals.append(np.nan)
        rows.append(vals)
    arr = np.asarray(rows, dtype=np.float64)
    out = {c: arr[:, i] for i, c in enumerate(cols)}
    out["iteration"] = out["iteration"].astype(np.int64)
    return out


def training_log_last(log_dir, name):
    """The last row of tmp_training/<name>.log as {column: float}, or None."""
    d = training_log_read(log_dir, name)
    if d is None:
        return None
    return {c: (int(v[-1]) if c == "iteration" else float(v[-1])) for c, v in d.items()}


def recovery_log_append(log_dir, iteration, scored):
    """One row into tmp_training/<key>.log for every quantity `scored` carries.

    `scored` is score_recovery's dict. A quantity is present when its `<key>_R2`
    is; a column the quantity does not have (Wij_gain on a known-ODE, whose W is
    read directly) is written as nan, so every file has a fixed header.
    """
    for key in RECOVERY_KEYS:
        # a gated-out quantity has no <key>_R2 but still gets a row (all nan
        # except the gate, msg_form_r2_median), which says why.
        present = (f"{key}_R2" in scored
                   or (key == "Eij" and "msg_form_r2_median" in scored)
                   or (key == "Wij" and "Wij_estimator" in scored))
        if not present:
            continue
        cols = recovery_log_columns(key)
        # the metrics.txt descriptor, written once (the readout cannot change
        # mid-run).
        _note = "; ".join(
            f"{c}={scored[c]}" for c in ("readout", f"{key}_estimator",
                                         f"{key}_correction") if c in scored)
        training_log_append(log_dir, key, iteration,
                            {c: scored.get(c) for c in cols}, note=_note or None)


def metrics_lines(scored):
    """`key: value` lines for results/metrics.txt and the analysis log.

    Per quantity, the numeric columns in file order, then the estimator and the
    correction strings. Keys score_recovery emits that belong to no quantity
    (a diagnostic such as extraction_error) come last, unchanged.
    """
    lines, seen = [], set()
    # `readout:` first, since it qualifies every `<key>_*` line below it.
    if "readout" in scored:
        lines.append(f"readout: {_fmt_metric(scored['readout'])}")
        seen.add("readout")
    for key in RECOVERY_KEYS:
        if f"{key}_R2" not in scored:
            continue
        for c in recovery_log_columns(key) + (f"{key}_estimator", f"{key}_correction"):
            if c in scored:
                lines.append(f"{c}: {_fmt_metric(scored[c])}")
                seen.add(c)
    for k, v in scored.items():
        if k not in seen and not isinstance(v, np.ndarray):
            lines.append(f"{k}: {_fmt_metric(v)}")
    return lines


def write_recovery_metrics(scored, log_dir, log_file=None, logger=None,
                           name="metrics.txt"):
    """The only writer of recovered-parameter metrics for `-o test_plot`.

    Appends metrics_lines(scored) to results/<name>, to the analysis log
    `log_file` when given, and to `logger`. `name` lets a second readout emit
    the same keys into its own file, so the two diff line by line.
    """
    lines = metrics_lines(scored)
    if not lines:
        return
    path = os.path.join(log_dir, "results", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as mf:
        mf.write("\n".join(lines) + "\n")
    if log_file is not None:
        log_file.write("\n".join(lines) + "\n")
    if logger is not None:
        for ln in lines:
            logger.info(ln)


def _connectivity_stats(w, src, dst, n):
    """Per-neuron mean/std/min/max of in-weights and out-weights, (8, n)."""
    w = np.asarray(w, dtype=np.float64).ravel()
    in_count = np.bincount(dst, minlength=n).astype(np.float64)
    out_count = np.bincount(src, minlength=n).astype(np.float64)
    in_sum = np.bincount(dst, weights=w, minlength=n)
    out_sum = np.bincount(src, weights=w, minlength=n)
    in_sq = np.bincount(dst, weights=w ** 2, minlength=n)
    out_sq = np.bincount(src, weights=w ** 2, minlength=n)
    safe_in = np.where(in_count > 0, in_count, 1)
    safe_out = np.where(out_count > 0, out_count, 1)
    in_mean = in_sum / safe_in
    out_mean = out_sum / safe_out
    in_std = np.sqrt(np.maximum(in_sq / safe_in - in_mean ** 2, 0))
    out_std = np.sqrt(np.maximum(out_sq / safe_out - out_mean ** 2, 0))
    in_max = np.full(n, -np.inf); np.maximum.at(in_max, dst, w)
    in_min = np.full(n, np.inf); np.minimum.at(in_min, dst, w)
    out_max = np.full(n, -np.inf); np.maximum.at(out_max, src, w)
    out_min = np.full(n, np.inf); np.minimum.at(out_min, src, w)
    for arr, c in [(in_mean, in_count), (in_std, in_count),
                   (in_min, in_count), (in_max, in_count),
                   (out_mean, out_count), (out_std, out_count),
                   (out_min, out_count), (out_max, out_count)]:
        arr[c == 0] = 0
    return np.column_stack([in_mean, in_std, out_mean, out_std,
                            in_min, in_max, out_min, out_max])


def cluster_recovery(type_list, edges, learned_W, n_neurons, embedding=None,
                     learned_tau=None, learned_vrest=None, n_components=None,
                     return_features=False, keep_edges=None, keep_neurons=None):
    """Cell-type clustering accuracy from what the model learned -- ONE function
    for the trainer's cluster.log and test_plot's `clustering_accuracy`.

    Features per neuron, in this order and only when present: the learned
    embedding a_i (GNN only), tau_i, V_rest_i, then eight statistics of the
    learned weights around the neuron (mean/std/min/max of incoming and of
    outgoing). A Gaussian mixture with n_components = min(100, n_neurons - 1)
    is fitted on the standardised stack and its components matched to the true
    types (sparsify.clustering_gmm). A quantity that is absent, not per-neuron,
    or not finite everywhere is left out of the stack rather than fed as NaN.

    Returns {clustering_accuracy, clustering_ari, clustering_nmi, clustering_n_components,
    clustering_n_features} (+ `_X`, the feature stack, when return_features), or
    None when there is nothing to cluster.

    keep_edges / keep_neurons (bool, None = all): the score mask. The weight
    statistics are taken over the kept edges only, and only the kept neurons are
    clustered and scored (their rows of the stack, their true types); a neuron
    left with no kept edge reads its statistics as 0, as one with no edge does.
    """
    from connectome_gnn.sparsify import clustering_gmm

    def _host(x):
        # inputs may be CUDA tensors, which np.asarray cannot read.
        if x is None:
            return None
        return to_numpy(x) if hasattr(x, "detach") else np.asarray(x)

    type_list, edges, learned_W = _host(type_list), _host(edges), _host(learned_W)
    embedding, learned_tau, learned_vrest = (_host(embedding), _host(learned_tau),
                                             _host(learned_vrest))
    n = int(n_neurons)
    feats = []
    for arr in (embedding, learned_tau, learned_vrest):
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr.reshape(n, -1) if arr.size % n == 0 and arr.shape[0] == n else None
        if arr is not None and np.isfinite(arr).all():
            feats.append(arr)
    if learned_W is not None and edges is not None:
        e = np.asarray(edges)
        w = np.asarray(learned_W, dtype=np.float64).ravel()
        # unfitted edges are nan; read them as zero here (features only) so one
        # bad edge does not blank the feature.
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        m = min(w.size, e.shape[1])
        if m > 0:
            if keep_edges is None:
                feats.append(_connectivity_stats(w[:m], e[0, :m], e[1, :m], n))
            else:
                ke = np.asarray(keep_edges, bool)[:m]
                feats.append(_connectivity_stats(w[:m][ke], e[0, :m][ke], e[1, :m][ke], n))
    if not feats:
        return None
    X = np.column_stack(feats)
    labels = np.asarray(type_list).ravel()[:n]
    if keep_neurons is not None:
        kn = np.asarray(keep_neurons, bool)[:n]
        X, labels = X[kn], labels[kn]
    X = X[:, np.isfinite(X).all(axis=0)]      # drop any column a nan still reached
    if X.shape[1] == 0:
        return None
    if n_components is None:
        n_components = min(100, X.shape[0] - 1)
    res = clustering_gmm(X, labels, n_components=n_components)
    out = {
        "clustering_accuracy": float(res["accuracy"]),
        "clustering_ari": float(res["ari"]),
        "clustering_nmi": float(res["nmi"]),
        "clustering_n_components": int(n_components),
        "clustering_n_features": int(X.shape[1]),
    }
    if return_features:
        out["_X"] = X          # underscore: never reaches a log or metrics.txt
    return out


# clustering variants for the paper's clusterability figure, each
# cluster_recovery on a subset of the features, keys clustering_<variant>_<stat>:
#   gt      the true tau, V_rest and statistics of the true W -- the ceiling
#   params  the learned tau, V_rest and W, no embedding
#   emb     the learned embedding a_i alone
CLUSTERING_VARIANTS = ("gt", "params", "emb")


def cluster_recovery_variants(type_list, edges, n_neurons, learned_W=None, learned_tau=None,
                              learned_vrest=None, embedding=None, gt_W=None, gt_tau=None,
                              gt_vrest=None, n_components=None, return_features=False,
                              keep_edges=None, keep_neurons=None):
    """{clustering_<variant>_<stat>: value} for each CLUSTERING_VARIANTS entry
    whose inputs are given; a variant with nothing to cluster is left out. With
    return_features, also `_X_<variant>`: the stack that variant was clustered on
    (underscore keys never reach metrics.txt)."""
    inputs = {
        "gt": dict(learned_W=gt_W, learned_tau=gt_tau, learned_vrest=gt_vrest),
        "params": dict(learned_W=learned_W, learned_tau=learned_tau, learned_vrest=learned_vrest),
        "emb": dict(learned_W=None, embedding=embedding),
    }
    out = {}
    for variant in CLUSTERING_VARIANTS:
        kw = inputs[variant]
        if all(v is None for v in kw.values()):
            continue
        res = cluster_recovery(type_list, edges, kw.get("learned_W"), n_neurons,
                               embedding=kw.get("embedding"), learned_tau=kw.get("learned_tau"),
                               learned_vrest=kw.get("learned_vrest"), n_components=n_components,
                               return_features=return_features, keep_edges=keep_edges,
                               keep_neurons=keep_neurons)
        if res is None:
            continue
        if return_features:
            out[f"_X_{variant}"] = res.pop("_X")
        for k, v in res.items():
            out[f"clustering_{variant}_{k[len('clustering_'):]}"] = v
    return out


# --------------------------------------------------------------------------- #
#  Are the two forms distinguishable on this run
# --------------------------------------------------------------------------- #

def form_comparison_stats(diagnostics, n_edges_reported=None):
    """The nested-model test between the two families, edge by edge.

    The two fits are paired and nested (current = conductance minus the u*v_i
    column), so a two-sample test between the R2 distributions is meaningless;
    the nested pair defines an F test per edge:

        F = (RSS_current - RSS_conductance) / (RSS_conductance / (n - 3))

    with 1 and n-3 degrees of freedom, n the frames the edge was fitted on. In R2
    terms F = (R2_cond - R2_cur) / ((1 - R2_cond) / (n - 3)), the square of the
    readout's t statistic on b2; the run-level answer is the share of edges that
    reject. Consecutive frames are correlated, so the shares are optimistic.

    Returns (stats dict, list of terminal lines).
    """
    cond = np.asarray(diagnostics.get("_form_cond_r2_full", []), dtype=np.float64)
    cur = np.asarray(diagnostics.get("_form_cur_r2_full", []), dtype=np.float64)
    t = np.asarray(diagnostics.get("_form_t_b2_full", []), dtype=np.float64)
    n_used = np.asarray(diagnostics.get("_form_n_used_full", []), dtype=np.float64)
    if cond.size == 0 or cur.size == 0:
        return {}, []

    ok = np.isfinite(cond) & np.isfinite(cur)
    gain = np.where(ok, cond - cur, np.nan)
    g = gain[np.isfinite(gain)]
    n_e = int(n_edges_reported or ok.sum())
    tt = t[np.isfinite(t)]

    # t >= 3 is the readout's per-edge reversal gate; the Bonferroni threshold
    # caps the chance of any false rejection across all edges at 5%.

    from scipy.stats import norm
    _t_bonf = float(norm.isf(0.025 / max(n_e, 1)))
    stats = {
        "n_edges": n_e,
        "gain_mean": float(np.mean(g)) if g.size else float("nan"),
        "gain_sd": float(np.std(g, ddof=1)) if g.size > 1 else float("nan"),
        "gain_q10": float(np.quantile(g, 0.10)) if g.size else float("nan"),
        "gain_q50": float(np.quantile(g, 0.50)) if g.size else float("nan"),
        "gain_q90": float(np.quantile(g, 0.90)) if g.size else float("nan"),
        "cond_median": float(np.nanmedian(cond)) if np.isfinite(cond).any() else float("nan"),
        "cur_median": float(np.nanmedian(cur)) if np.isfinite(cur).any() else float("nan"),
        "t_median": float(np.median(tt)) if tt.size else float("nan"),
        "pct_t3": float(100.0 * np.mean(tt >= 3.0)) if tt.size else float("nan"),
        "t_bonferroni": _t_bonf,
        "pct_bonferroni": float(100.0 * np.mean(tt >= _t_bonf)) if tt.size else float("nan"),
        "frames_median": float(np.nanmedian(n_used)) if n_used.size else float("nan"),
    }
    f = stats
    lines = [
        f"are the two forms distinguishable on this run?  "
        f"(nested F test per edge, 1 and n-3 d.o.f.)",
        f"   median per-edge R²: conductance {f['cond_median']:.4f}   "
        f"current {f['cur_median']:.4f}",
        f"   gain quantiles [10 / 50 / 90]: {f['gain_q10']:+.4f} / "
        f"{f['gain_q50']:+.4f} / {f['gain_q90']:+.4f}   "
        f"(mean {f['gain_mean']:+.4f} ± {f['gain_sd']:.4f})",
        f"   edges where the driving force clears 3 s.e.: {f['pct_t3']:.1f}%   "
        f"and Bonferroni t ≥ {f['t_bonferroni']:.2f} over {f['n_edges']:,} tests: "
        f"{f['pct_bonferroni']:.1f}%",
        f"   median |t| on the driving-force slope {f['t_median']:.1f}, on a median "
        f"{f['frames_median']:.0f} frames per edge — consecutive frames, so these "
        f"shares are optimistic",
    ]
    return stats, lines
