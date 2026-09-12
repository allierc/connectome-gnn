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
from connectome_gnn.models.utils import pad_g_phi_input, is_conductance_gnn
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
    # w_squared FIRST, because it is a statement about how W enters the ODE and
    # therefore about what W MEANS, which overrides any convention about its
    # sign. Under it the message uses W**2, so W**2 is the learned conductance
    # and the raw parameter is its square root; scattering the root against
    # ode_params.W would report the mismatch as a recovery failure. The sign-lock
    # below is moot here anyway -- a square has no sign to lock.
    if getattr(model, 'w_squared', False):
        return model.W.detach() ** 2
    # Prefer the effective weight (|W|·sign_GT under the hard sign-lock); when
    # the lock is off this equals the raw W, so existing models are unaffected.
    if hasattr(model, 'effective_W'):
        return model.effective_W
    # THE CONDUCTANCE STUDENT STORES THE SQUARE ROOT. Its `W` parameter enters the
    # ODE as W**2, so the conductance is non-negative by construction, and
    # ode_params.W on a conductance-generated dataset holds that SQUARE. Returning
    # the raw parameter here would scatter sqrt(g) against g and report the
    # mismatch as a recovery failure.
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
    emb_i_flat: postsynaptic embedding ai — required for flyvis_conductance, where
        g_phi(vi=0, vj, ai, aj) depends on both endpoints. Should be a real
        partner embedding (see `_avg_postsynaptic_embedding`), not aj again.
    """
    if is_conductance_gnn(signal_model_name):
        if emb_i_flat is None:
            raise ValueError(
                "_build_g_phi_features: flyvis_conductance requires emb_i_flat (postsynaptic embedding)"
            )
        return torch.cat([rr_flat, emb_flat, rr_flat * 0, emb_i_flat], dim=1)
    else:
        return torch.cat([rr_flat, emb_flat], dim=1)


def _avg_postsynaptic_embedding(model_a, edges, n_neurons):
    """Per-presynaptic-neuron average embedding of its real postsynaptic partners.

    ai_avg[j] = mean(model_a[dst]) over edges with src == j. Used to build a
    representative ai for g_phi(vi=0, vj, ai, aj) when g_phi depends on both
    endpoints (flyvis_conductance) — real partners, not aj=ai. Neurons with no outgoing
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

def eval_g_phi_over_domain(model, config, n_neurons, rr, device, edges=None):
    """Evaluate g_phi over a caller-supplied per-neuron voltage domain.

    Shared core of `evaluate_g_phi_curves` — factored out so callers that
    need a non-default domain (e.g. a wider range for plotting, vs. the
    activity-range domain used for slope fitting) can reuse the same
    flyvis_conductance-aware feature construction instead of duplicating it.

    Args:
        rr: (N, n_pts) tensor of per-neuron voltage values to sweep (vj).

    For flyvis_conductance, g_phi depends on both endpoints (vi, vj, ai, aj);
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

    if is_conductance_gnn(signal_model_name):
        if edges is None:
            raise ValueError(
                "eval_g_phi_over_domain: flyvis_conductance requires `edges` to build the postsynaptic embedding ai"
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

    See `eval_g_phi_over_domain` for the flyvis_conductance / `edges` note.

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


# Rows per g_phi evaluation in sample_g_phi_vi_vj_observed. 1e6 rows keeps the
# [rows, hidden] activation near 300 MB at hidden 80, so the conductance
# W-correction runs on a 24 GB l4 as well as an a100. Purely a memory knob --
# the result does not depend on it.
G_PHI_EVAL_CHUNK = 1_000_000


def sample_g_phi_vi_vj_observed(model, config, edges, x_ts, n_edges=16, n_frames=2000, seed=0):
    """Evaluate learned g_phi at REAL, co-occurring (vi(t), vj(t)) pairs for a
    sample of real edges — not an independent (vi, vj) grid.

    Diagnostic for whether flyvis_conductance's MLP(ai, aj, vi, vj) actually
    learned a vi-dependent (conductance-like, e.g. ReLU(vi-vj-threshold))
    function that matters where real activity lives, or collapsed to a
    vj-only function matching the true flyvis synapse ReLU(vj). Connected
    neurons i, j are typically correlated, so the region of the (vi, vj)
    plane real activity actually visits is not the full independent-grid
    rectangle — a grid answers "does g_phi depend on vi anywhere," this
    answers "does it depend on vi where the network actually operates,"
    which is the one that matters for the correction/interpretation.

    Real (ai, aj) per sampled edge, not synthetic — consistent with
    `_avg_postsynaptic_embedding`'s "use the real graph" principle.

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
    signal_model_name = config.graph_model.signal_model_name
    g_phi_positive = config.graph_model.g_phi_positive
    device = model.a.device
    emb_dim = model.a.shape[1]

    rng = np.random.default_rng(seed)
    n_edges_total = edges.shape[1]
    sel = rng.choice(n_edges_total, size=min(n_edges, n_edges_total), replace=False)
    src = edges[0, sel].to(device)   # j — presynaptic
    dst = edges[1, sel].to(device)   # i — postsynaptic
    n_e = len(sel)

    n_frames = min(n_frames, x_ts.n_frames)
    frame_idx = torch.from_numpy(rng.choice(x_ts.n_frames, size=n_frames, replace=False)).to(device).long()

    voltage = x_ts.voltage.to(device)                    # (T, N)
    vi = voltage[frame_idx][:, dst].T.contiguous()        # (n_e, n_frames)
    vj = voltage[frame_idx][:, src].T.contiguous()        # (n_e, n_frames)

    ai_flat = model.a[dst].unsqueeze(1).expand(-1, n_frames, -1).reshape(-1, emb_dim)
    aj_flat = model.a[src].unsqueeze(1).expand(-1, n_frames, -1).reshape(-1, emb_dim)
    vi_flat = vi.reshape(-1, 1)
    vj_flat = vj.reshape(-1, 1)

    # CHUNKED, because the caller passes EVERY edge: compute_g_phi_fd_slope asks
    # for n_edges = edges.shape[1], so at flyvis scale this is 434,112 x 32 =
    # 13.9M rows. One call means a [13.9M, hidden] activation inside g_phi --
    # 4.14 GiB at hidden 80 -- which fits a 40-80 GB a100 and OOMs a 24 GB l4
    # every time, at the first plot_training_gnn after the 16k checkpoint.
    #
    # Bit-identical: rows are independent, nothing is reduced across them, so
    # splitting and concatenating returns the same tensor. Cost is unchanged --
    # same FLOPs, same traffic, ~14 extra kernel launches against 13.9M rows --
    # which matters because this runs at every connectivity checkpoint.
    with torch.no_grad():
        outs = []
        for _lo in range(0, vj_flat.shape[0], G_PHI_EVAL_CHUNK):
            _hi = _lo + G_PHI_EVAL_CHUNK
            if is_conductance_gnn(signal_model_name):
                _in = torch.cat([vj_flat[_lo:_hi], aj_flat[_lo:_hi],
                                 vi_flat[_lo:_hi], ai_flat[_lo:_hi]], dim=1)
            else:
                _in = torch.cat([vj_flat[_lo:_hi], aj_flat[_lo:_hi]], dim=1)
            _o = model.g_phi(pad_g_phi_input(_in.float(), model))
            if g_phi_positive:
                _o = _o ** 2
            outs.append(_o)
        out = torch.cat(outs, dim=0)

    g_phi_vals = out.reshape(n_e, n_frames)

    return {
        'edge_ij': np.stack([to_numpy(dst), to_numpy(src)], axis=1),
        'vi': to_numpy(vi),
        'vj': to_numpy(vj),
        'g_phi': to_numpy(g_phi_vals),
    }


def compute_g_phi_edge_grad(model, config, edges, x_ts, n_frames=8, seed=0):
    """Per-(edge, real frame) local gradient d(g_phi)/d(vj) at the actual
    (vi, vj, ai, aj) the network visits — the per-edge, per-frame analogue of
    `evaluate_g_phi_curves`'s single global 1D-sweep slope.

    Mirrors `compute_grad_msg`'s real-frame sampling (autograd at sampled
    real frames, not a synthetic domain), but for g_phi's own presynaptic
    sensitivity, over ALL real edges at once per frame (cheap — one forward
    through the small g_phi MLP per frame).

    Args:
        edges: (2, E) edge index — src (presynaptic, j) / dst (postsynaptic, i).
        x_ts: NeuronTimeSeries with the real voltage recording.
        n_frames: number of real frames to sample (autograd call each).
        seed: RNG seed for frame sampling (reproducible).

    Returns dict, each (E, n_frames) except j_ids/i_ids which are (E,):
        j_ids, i_ids: int arrays — presynaptic / postsynaptic neuron per edge.
        vj, vi: real voltage at the sampled frames.
        g_phi: g_phi output at (vi, vj, ai, aj) — squared if g_phi_positive,
            matching what actually enters the message.
        grad_vj: d(g_phi)/d(vj) at that point (post-squaring if applicable).
    """
    signal_model_name = config.graph_model.signal_model_name
    g_phi_positive = config.graph_model.g_phi_positive
    device = model.a.device

    src = edges[0].to(device)   # j — presynaptic
    dst = edges[1].to(device)   # i — postsynaptic
    n_edges = edges.shape[1]

    rng = np.random.default_rng(seed)
    n_frames = min(n_frames, x_ts.n_frames)
    frame_idx = rng.choice(x_ts.n_frames, size=n_frames, replace=False)

    voltage = x_ts.voltage.to(device)   # (T, N)
    ai = model.a[dst]
    aj = model.a[src]

    vj_all = torch.zeros(n_edges, n_frames, device=device)
    vi_all = torch.zeros(n_edges, n_frames, device=device)
    g_phi_all = torch.zeros(n_edges, n_frames, device=device)
    grad_all = torch.zeros(n_edges, n_frames, device=device)

    for f_idx, k in enumerate(frame_idx):
        vj_k = voltage[k, src].clone().detach().requires_grad_(True)   # (E,)
        vi_k = voltage[k, dst].clone().detach()

        if is_conductance_gnn(signal_model_name):
            in_features = torch.cat([vj_k.unsqueeze(1), aj, vi_k.unsqueeze(1), ai], dim=1)
        else:
            in_features = torch.cat([vj_k.unsqueeze(1), aj], dim=1)

        out = model.g_phi(pad_g_phi_input(in_features.float(), model))
        if g_phi_positive:
            out = out ** 2
        grad = torch.autograd.grad(out.sum(), vj_k, retain_graph=False, create_graph=False)[0]

        vj_all[:, f_idx] = vj_k.detach()
        vi_all[:, f_idx] = vi_k
        g_phi_all[:, f_idx] = out.detach().squeeze(-1)
        grad_all[:, f_idx] = grad.detach()

    return {
        'j_ids': to_numpy(src),
        'i_ids': to_numpy(dst),
        'vj': to_numpy(vj_all),
        'vi': to_numpy(vi_all),
        'g_phi': to_numpy(g_phi_all),
        'grad_vj': to_numpy(grad_all),
    }


def compute_g_phi_fd_slope(model, config, edges, x_ts, n_frames=32, seed=0):
    """Per-edge local slope d(g_phi)/d(vj), estimated by finite differences
    on real, co-occurring (vi, vj, g_phi) samples from
    `sample_g_phi_vi_vj_observed` — the empirical counterpart to
    `compute_g_phi_edge_grad`'s analytic autograd gradient.

    For each edge, sorts its own sampled frames by vj and differences
    consecutive points: slope = d(g_phi) / d(vj) between neighbors. Since
    vi varies (uncontrolled) between the two differenced points, this is
    noisier than the autograd version — large outliers appear whenever two
    neighbors happen to be close in vj — so callers should filter/trim
    before averaging (see `compute_g_phi_correction_conductance`).

    Returns dict, each (E, n_frames-1) except j_ids/i_ids which are (E,):
        j_ids, i_ids: int arrays — presynaptic / postsynaptic neuron per edge.
        vj, vi, g_phi: midpoint values of each differenced pair.
        grad_vj: the finite-difference slope.
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


def compute_g_phi_correction_conductance(model, config, edges, x_ts, n_neurons, n_frames=32, seed=0):
    """Per-neuron g_phi correction factor eta_j for flyvis_conductance,
    replacing the single global 1D-sweep affine fit (`evaluate_g_phi_curves`)
    used by other flyvis variants — g_phi depends on both endpoints here, so
    a synthetic vi=0 sweep with an averaged ai isn't representative.

    Method (selected by a three-way comparison against the 1D-sweep and an
    autograd-gradient method — see dev_g_phi_w_correction.py): finite-
    difference d(g_phi)/d(vj) on real (edge, frame) pairs
    (`compute_g_phi_fd_slope`), restricted to the physically active region
    (vj>0) and to slopes consistent with the monotonicity prior (slope>0),
    then a 10-90 percentile-trimmed mean per presynaptic neuron j — trimming
    rejects the large local-slope outliers introduced by vi's uncontrolled
    variation between the two finite-difference points. Validated against the
    1D-sweep baseline and an independently derived autograd-gradient method.

    Returns:
        (n_neurons,) numpy array — one correction factor per neuron;
        neurons with no surviving (edge, frame) samples default to 1.0.
    """
    raw = compute_g_phi_fd_slope(model, config, edges, x_ts, n_frames=n_frames, seed=seed)
    j_ids, vj, grad = raw['j_ids'], raw['vj'], raw['grad_vj']

    active_mask = (vj > 0) & (grad > 0)
    j_flat = np.repeat(j_ids, vj.shape[1])[active_mask.ravel()]
    grad_flat = grad.ravel()[active_mask.ravel()]

    eta = np.ones(n_neurons, dtype=np.float32)
    order = np.argsort(j_flat, kind='stable')
    j_sorted = j_flat[order]
    grad_sorted = grad_flat[order]
    unique_js, start_idx = np.unique(j_sorted, return_index=True)
    groups = np.split(grad_sorted, start_idx[1:])

    for j, vals in zip(unique_js, groups):
        lo, hi = np.percentile(vals, [10, 90])
        trimmed = vals[(vals >= lo) & (vals <= hi)]
        eta[j] = trimmed.mean() if trimmed.size else vals.mean()

    return eta


def extract_conductance_params_from_gnn(model, config, edges, x_ts, n_frames=64,
                                        vj_quantile=0.25, min_points=8, seed=0):
    """Read W_ij and E_ij back out of a trained conductance GNN, per edge.

    THE PROBLEM. The known-ODE student stores W and E as named parameters, so
    :func:`compute_reversal_metrics` just reads them. The GNN stores a per-edge
    weight and an MLP, and its message is

        msg_ij  =  W_gnn_ij * g_phi(v_j, a_j, v_i, a_i)^2

    with nothing labelled "conductance" or "reversal potential" anywhere in it.
    Both quantities are still in there, because the generator this is fitting is

        msg_ij  =  W_ij * relu(v_j) * (E_i - v_i)

    and they can be read out without any optimisation at all.

    THE METHOD, and why it is a division rather than a fit. Divide the message by
    v_j -- the same normalisation the current-data extraction applies through
    `compute_g_phi_correction_conductance`, and for the same reason: it stops the
    presynaptic drive from being absorbed into W and producing enormous W on the
    edges whose v_j happened to be small. What is left,

        y_ij(t)  =  msg_ij(t) / v_j(t)  =  W_ij * (E_i - v_i(t))

    is a STRAIGHT LINE IN v_i with slope -W_ij and intercept W_ij * E_i. So one
    ordinary least-squares line per edge, over the real co-occurring (v_i, v_j)
    the network actually visits, gives

        W_ij = -slope            E_ij = intercept / W_ij = -intercept / slope

    THE STRAIGHT LINE IS ALSO THE TEST. `fit_r2` is the R2 of that line per edge.
    If it is not near 1 the learned message is not affine in v_i, the model has
    not found the conductance form, and the W and E read off it describe nothing.
    Check it before reporting either.

    WHAT IS AND IS NOT IDENTIFIABLE. W_gnn and the amplitude of g_phi^2 trade off
    against one common scalar -- doubling one and halving the other leaves every
    message unchanged -- so W_ij comes out UP TO A SINGLE GLOBAL FACTOR and its
    absolute size means nothing. E_ij is a RATIO of the intercept to the slope, so
    that factor cancels and E is recovered absolutely. Compare W with
    :func:`r2_up_to_scale`, which fits the one factor and reports R2 with the
    slope pinned to 1 by construction; compare E directly.

    Only samples with v_j above `vj_quantile` OF THE POSITIVE v_j VALUES enter the
    fit. Below that the division amplifies noise without adding information, and
    v_j <= 0 carries none at all -- the true drive is relu(v_j), so those frames
    say nothing about W_ij or E_ij.

    Args:
        n_frames: real frames sampled per edge.
        vj_quantile: quantile OF THE POSITIVE SAMPLED v_j used as the floor.
        min_points: an edge needs this many surviving samples, with a non-zero
            spread in v_i, or its entry is nan.

    Returns a dict of (E,) arrays unless noted:
        W          extracted conductance, up to one global scale factor
        E          extracted per-edge reversal potential, in voltage units
        fit_r2     R2 of the per-edge straight line -- the validity check
        n_used     samples that survived the v_j floor, per edge
        E_pooled   per-edge reversal after pooling each postsynaptic neuron's
                   incoming edges by SIGN of E, median within group: the
                   generator gives every edge onto neuron i the same E_exc[i] or
                   E_inh[i], so pooling is the estimator that uses that, and the
                   split by sign is what keeps excitatory and inhibitory apart
        E_exc, E_inh   (N,) the two pooled values per postsynaptic neuron, nan
                   where that neuron received no edge of that sign
        vj_floor   the floor actually used, in voltage units
    """
    res = sample_g_phi_vi_vj_observed(model, config, edges, x_ts,
                                      n_edges=edges.shape[1], n_frames=n_frames,
                                      seed=seed)
    vi, vj, g_sq = res['vi'], res['vj'], res['g_phi']      # each (E, n_frames)
    i_ids = res['edge_ij'][:, 0].astype(np.int64)          # postsynaptic

    W_gnn = to_numpy(get_model_W(model)).ravel()[:vi.shape[0]].astype(np.float64)

    pos = vj[vj > 0]
    vj_floor = float(np.quantile(pos, vj_quantile)) if pos.size else 0.0
    keep = vj > max(vj_floor, 1e-6)

    # y = msg / v_j = W_ij * (E_i - v_i); masked entries contribute 0 to every sum.
    y = np.where(keep, W_gnn[:, None] * g_sq / np.where(keep, vj, 1.0), 0.0)
    x = np.where(keep, vi, 0.0)

    n = keep.sum(axis=1).astype(np.float64)
    Sx, Sy = x.sum(axis=1), y.sum(axis=1)
    Sxx, Syy, Sxy = (x * x).sum(axis=1), (y * y).sum(axis=1), (x * y).sum(axis=1)

    den_x = n * Sxx - Sx ** 2
    den_y = n * Syy - Sy ** 2
    cov = n * Sxy - Sx * Sy
    ok = (n >= min_points) & (den_x > 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        slope = np.where(ok, cov / den_x, np.nan)
        intercept = np.where(ok, (Sy - slope * Sx) / n, np.nan)
        fit_r2 = np.where(ok & (den_y > 0), cov ** 2 / (den_x * den_y), np.nan)
        W = -slope
        E = np.where(W != 0, -intercept / slope, np.nan)

    # Pool by (postsynaptic neuron, sign of E). The generator hands every edge
    # onto neuron i one of exactly two reversals, so the per-edge estimates of
    # each sign are repeated measurements of the same number; the median is the
    # estimator that says so, and it is taken separately per sign because mixing
    # +24.8 with -14.5 would average to a value neither of them takes.
    n_neurons = model.a.shape[0]
    E_exc = np.full(n_neurons, np.nan)
    E_inh = np.full(n_neurons, np.nan)
    good = np.isfinite(E) & np.isfinite(fit_r2)
    for arr, sign_mask in ((E_exc, E > 0), (E_inh, E < 0)):
        m = good & sign_mask
        if not m.any():
            continue
        order = np.argsort(i_ids[m], kind='stable')
        ids_s, vals_s = i_ids[m][order], E[m][order]
        uniq, start = np.unique(ids_s, return_index=True)
        for _i, vals in zip(uniq, np.split(vals_s, start[1:])):
            arr[_i] = np.median(vals)
    E_pooled = np.where(E < 0, E_inh[i_ids], E_exc[i_ids])

    # Edges whose fitted line RISES with v_i: the message grows as the
    # postsynaptic cell depolarises, which no conductance does (the driving
    # force E - v_i can only fall). Not a sign to repair -- W enters squared and
    # the sign is g_phi's -- but the share of edges where the model has not
    # found the conductance form, beside the fit R2 that says how far it is.
    pct_wrong_slope = float(100.0 * np.mean(slope[ok] > 0)) if ok.any() else float('nan')
    return {'W': W, 'E': E, 'fit_r2': fit_r2, 'n_used': n, 'pct_wrong_slope': pct_wrong_slope,
            'E_pooled': E_pooled, 'E_exc': E_exc, 'E_inh': E_inh,
            'vj_floor': vj_floor}


def r2_up_to_scale(true, learned):
    """R2 of `learned` against `true` after dividing out ONE global gain.

    For a quantity the model can only pin down up to a common factor -- the GNN's
    W_ij, where W and the amplitude of g_phi trade off exactly; msg_i on a GNN,
    which carries the same g_phi gain -- the identity-line R2 charges that factor
    as error and says nothing about whether the shape is right.

    THE ONE GAIN CONVENTION, used by every `<key>_gain` this module emits:

        gain = <true, learned> / <true, true>        so that  learned ~= gain * true

    i.e. the least-squares factor through the origin with `true` as the regressor,
    1.0 being perfect, 2.0 meaning the learned values are twice too large. The R2
    is then the identity-line R2 of learned / gain against true. (The reciprocal
    fit, c = <true, learned> / <learned, learned> with c * learned against true,
    gives a slightly different R2 and a gain that reads backwards; it was the
    convention here before 2026-09-11 while test_plot's `w_scale` and the
    trainer's msg_i scale used this one -- that split is gone.)

    ONE free parameter over however many samples there are, and the slope
    afterwards is 1 BY CONSTRUCTION, so no slope is returned.

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

    No-op (returns nan) unless g_phi's first layer has the flyvis_conductance
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

        flyvis_conductance : [v_j, a_j, v_i, a_i] + noise   width 2 + 2*emb + n
        everything else    : [v_j, a_j]           + noise   width 1 + emb   + n

    The conductance layout is a PREFIX-EXTENSION of the other one: columns
    0..emb_dim are [v_j, a_j] in BOTH families. That is what makes every
    hardcoded column index (notably the coeff_g_phi_norm anchor at column 0)
    mean the same physical quantity everywhere.

    Dispatched on `model.model`, which is the exact attribute the forward pass
    branches on — a metric that guessed the layout independently could silently
    read a_j's columns as a_i's. Returns (layout, n_noise) where layout maps
    'vi'/'vj'/'ai'/'aj'/'noise' to slices; 'vi' and 'ai' are None for the
    non-conductance layout, which has no post-synaptic inputs to discard.

    Returns (None, 0) if the width does not match either layout, so callers
    stay no-ops on models this was never designed for.
    """
    width = model.g_phi.layers[0].weight.shape[1]
    if getattr(model, "model", None) == "flyvis_conductance":
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

    Unlike g_phi_first_layer_discard_score's linear fraction, this is
    scale-invariant and doesn't penalize vj/aj for splitting the "good"
    mass unevenly between them -- only vi/ai carrying any weight at all
    pulls it down. 1 = fully aligned (vi=ai=0); 0 = all mass on vi/ai.

    Defined for both g_phi layouts (see g_phi_column_layout). On
    flyvis_conductance the discard side is {vi, ai, noise}; on flyvis_A, whose
    g_phi already takes only (vj, aj), it is {noise} alone -- which is exactly
    the positive control: flyvis_A is the correctly-specified family, so any
    weight it keeps on the noise columns is a failure of credit assignment that
    cannot be blamed on the extra inputs being redundant.

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
    The weight-level score can disagree with this (a later layer can route
    around a large first-layer weight) -- this is the one to trust when they
    do; see dev_g_phi_regularization_comparison.py for the comparison that
    established this.

    Returns (ratio_vi, ratio_ai, ratio_noise). ratio_noise is nan unless the
    noise-probe control is on (n_g_phi_noise_inputs > 0); it is the headline
    number for that experiment, since the noise columns are uninformative BY
    CONSTRUCTION, so any non-zero value is credit that should not have been
    assigned. Compare it against ratio_vi: noise -> 0 while vi stays up means vi
    is retained for a reason (it is redundant with f_theta's own vi input), not
    because credit assignment failed.

    On the flyvis_A layout g_phi has no vi/ai inputs at all, so ratio_vi and
    ratio_ai are nan there by construction and ratio_noise is the whole result --
    that is the point of running the probe on the correctly-specified family.

    All nan if g_phi's input layout matches neither family (see
    g_phi_column_layout).
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


def extract_g_phi_slopes(model, config, n_neurons, mu_activity, sigma_activity, device, edges=None):
    """Extract linear slope of g_phi for each neuron j (vectorized).

    Returns:
        slopes: (n_neurons,) numpy array of g_phi slopes.
    """
    v_ranges, curves, valid = evaluate_g_phi_curves(
        model, config, n_neurons, mu_activity, sigma_activity, device, edges=edges)

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
    # THE ARRAYS THE R2s ABOVE WERE COMPUTED FROM. A caller that wants the
    # recovery panel would otherwise have to re-derive tau and V_rest out of
    # f_theta -- a whole extract_f_theta_slopes pass over every neuron -- and,
    # worse, could then draw a panel describing different numbers from the ones
    # in metrics.log. None when the model or the dataset has no such parameter.
    'tau_true': None, 'tau_learned': None,
    'vrest_true': None, 'vrest_learned': None,
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

    THE FAMILY FORK IS GONE. This used to branch on hasattr(model,
    "get_learned_tau") and had a sister, compute_dynamics_r2_linear, that differed
    ONLY in that branch before computing an identical R2. Both now delegate:
    extract_recovered_params decides how to obtain tau and V_rest, and
    recovery_param_metrics scores them. The dict shape is unchanged, so callers
    (graph_trainer, plot_dynamics_recovery) are untouched.
    """
    from connectome_gnn.generators.ode_params import load_ode_params_for_run
    try:
        ode_params = load_ode_params_for_run(config, device=device)
    except (FileNotFoundError, TypeError):
        return dict(_DYNAMICS_R2_EMPTY)

    rec = extract_recovered_params(model, ode_params, config, edges=None, x_ts=x_ts,
                                   device=device, n_neurons=n_neurons,
                                   need=("tau", "V_rest"))
    return _dynamics_dict_from(rec, config)


def _dynamics_dict_from(rec, config):
    """Render a RecoveredParams into the legacy dynamics dict.

    Kept as a shim so the extraction refactor does not have to move the trainer's
    metrics.log column layout at the same time -- plot.py reads that file by
    POSITIONAL index, so a shape change there breaks every reader after it.
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
        plus a separate conn_r2 float.

    NOW A THIN WRAPPER over the same extractor compute_dynamics_r2 uses. It used
    to reach into model.V_rest and F.softplus(model.raw_tau) directly rather than
    calling the accessors the class exposes, and -- the reason plot.py grew its own
    copy of that reach-in -- it returned the R2s WITHOUT the arrays, so a caller
    that wanted the tau panel had no choice but to re-derive them. The arrays come
    back now, which is what lets the panels stop computing anything.
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
        # Outlier-filtered, matching plot_training_linear's scatter annotation and
        # the GNN training-time convention. Before the filter was added this was
        # the full-sample R2 of the raw weights, which silently disagreed with the
        # plot printed beside it.
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

def _reversal_metrics_from_gnn(core, ode_params, config, edges, x_ts):
    """The GNN branch of :func:`compute_reversal_metrics` -- same dict, read out
    of the learned message rather than off a parameter.

    Returns None when the extraction produced too few usable edges, which is the
    same signal the parameter path uses: no panel, no bar entry, rather than a
    number nothing stands behind.
    """
    import numpy as np

    ext = extract_conductance_params_from_gnn(core, config, edges, x_ts)
    true = to_numpy(ode_params.reversal_per_edge()).ravel()
    learned = ext['E_pooled']

    n = int(min(true.size, learned.size))
    true, learned = true[:n], learned[:n]
    ok = np.isfinite(true) & np.isfinite(learned)
    if ok.sum() < 2:
        return None
    true_ok, learned_ok = true[ok], learned[ok]

    m = recovery_param_metrics(true_ok, learned_ok)
    w_true = to_numpy(ode_params.W).ravel()[:ext['W'].size]
    w_fit = r2_up_to_scale(w_true, ext['W'])

    # The per-neuron reversals the granularity-adaptive panel draws. The
    # extraction already pools each postsynaptic neuron's incoming edges by the
    # sign of E, so E_exc/E_inh are the same two numbers per cell the known-ODE
    # student stores -- just estimated rather than parameterised.
    type_index = (to_numpy(core.type_index).ravel()
                  if getattr(core, "type_index", None) is not None else None)
    _dst = to_numpy(ode_params.edge_index[1]).ravel()
    per_neuron = dict(
        true_exc=to_numpy(ode_params.E_exc).ravel(),
        true_inh=to_numpy(ode_params.E_inh).ravel(),
        learned_exc=ext['E_exc'],
        learned_inh=ext['E_inh'],
        type_index=type_index,
    )
    if type_index is not None:
        per_neuron["edge_type"] = type_index[_dst % type_index.size]
    _targeted = np.zeros(per_neuron["true_exc"].size, dtype=bool)
    _targeted[np.unique(_dst) % _targeted.size] = True
    per_neuron["targeted"] = _targeted

    return {
        "rmse": float(np.sqrt(np.mean((learned_ok - true_ok) ** 2))),
        "r2": float(m["r2"]),
        "slope": float(m["slope"]),
        "n_edges": int(true_ok.size),
        "true": true_ok,
        "learned": learned_ok,
        "fit_r2_median": float(np.nanmedian(ext['fit_r2'])),
        "pct_wrong_slope": ext.get("pct_wrong_slope", float("nan")),
        "w_r2_scaled": float(w_fit['r2']),
        "w_scale": float(w_fit['scale']),
        **per_neuron,
    }


# How many frames the msg_i panel is evaluated on. THE SAME FRAMES EVERY TIME --
# evenly spaced over the recording by linspace, not sampled -- because the point
# of the panel is to be flipped through: two consecutive checkpoints must differ
# because the MODEL moved, not because the frames did.
MSG_N_FRAMES = 10


def compute_msg_i_recovery(model, ode_params, x_ts, edges, device,
                           n_frames=MSG_N_FRAMES):
    """The aggregated per-neuron message msg_i, true vs learned.

    WHY THE AGGREGATED MESSAGE AND NOT THE EDGE ONE. msg_i is what f_theta
    consumes and the only channel through which connectivity reaches dv/dt, so it
    is the quantity whose recovery the trajectory actually depends on. It is also
    the one place the conductance/driving-force degeneracy stops mattering: the
    message is g_ij * act(v_j) * (E_i - v_i), g and the driving force trade off
    inside it, and msg_i is the product that survives that trade -- a model can
    have W_ij wrong by 3x and E_ij wrong by 1/3 and still land msg_i on the
    identity line, which is exactly what a trajectory correlation of 0.95 beside
    an R2 of -10 on W_ij means.

    THE TRUE SIDE IS REBUILT FROM ode_params, not read from the dataset, so it is
    the generator's own message rather than anything the trainer derived:

        edge_msg = W_ij * gt_g_phi_func(v_j)          the activation the
                                                      generator used, ReLU on
                                                      flyvis, whatever
                                                      `activation` says on cx
        edge_msg *= (E_ij - v_i)                      conductance datasets only;
                                                      a current generator has no
                                                      such term
        msg_i     = sum of edge_msg over incoming edges

    THE LEARNED SIDE IS THE MODEL'S OWN msg, via forward(..., return_all=True),
    which every family implements -- NeuralGNN and KnownODEBase alike. Nothing is
    re-derived, so the panel shows the message the model is really passing rather
    than a reconstruction that could differ from it.

    Returns (true, learned, groups) flattened over frames, each
    (n_frames * n_neurons,), with `groups` the postsynaptic cell type tiled to
    match -- or None when there is no ground-truth W to build the true side from.
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
            if _msg_i_through_f_theta(model):
                msg_learned = _msg_through_f_theta(model, pred, in_features, n_neurons)
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


def _msg_through_f_theta(model, pred, in_features, n_neurons):
    """The message a GNN really passes, read through its own update in voltage
    units:

        msg_eff_i = tau_i * [ f_theta(a_i, v_i, msg_i, exc_i) - f_theta(a_i, v_i, 0, exc_i) ]
        tau_i     = -1 / (d f_theta / d v_i)      (central difference at this frame)

    WHY NOT THE RAW AGGREGATE. What `return_all` hands back is the sum of g_phi,
    and the model is free to scale it by any factor gamma_i that f_theta then
    undoes: msg' = gamma_i * msg, f'(v, msg') = f(v, msg' / gamma_i). The raw R2
    charges the model for gamma_i, which never reaches the trajectory, and one
    global factor (msg_i_R2_scaled) cannot remove a per-neuron one. The
    difference above is gamma-free, and dividing by the model's own leak slope
    (not the true tau) keeps it the model's number: on the generator's
    tau dv/dt = -(v - V_rest) + msg + stim, linear in msg, a perfect model returns
    msg exactly, whatever the g_phi / f_theta split.

    The no-message baseline is taken at the ACTUAL v_i, not at v_i = 0, so no
    linearity in v is assumed and V_rest is not needed. Neurons whose leak slope
    is not negative (f_theta not yet a leak there) read nan and are dropped.
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
        dfdv = dfdv.ravel()[:n_neurons]
        tau = torch.where(dfdv < -1e-6, -1.0 / dfdv, torch.full_like(dfdv, float("nan")))
        return d * tau


def compute_reversal_metrics(model, ode_params, config=None, edges=None, x_ts=None):
    """Recovery of the per-edge reversal potential E_ij, true vs learned.

    ONLY DEFINED ON CONDUCTANCE-GENERATED DATA. The current-based generator has
    no (E - v_i) term at all, so there is no true E_ij to compare against and
    this returns None -- which is the signal for the caller to draw no panel and
    print no bar entry, rather than a zero that reads like a measurement.

    Both sides are reduced to ONE VALUE PER EDGE before comparing, because that
    is the only representation the two share. The ground truth stores E per
    postsynaptic neuron (n_neurons,) after the generator expanded whatever
    granularity the student was fitted at; the model stores it at its own
    granularity (1, n_neuron_types or n_neurons rows) and expands on demand. The
    per-edge form is `where(edge_is_inh, E_inh[dst], E_exc[dst])` on both sides,
    so a global fit and a per-neuron fit land on the same axis.

    Returns a dict with:
      rmse    root-mean-square error of learned minus true E_ij, in the same
              VOLTAGE UNITS as the dataset's voltages (flyvis voltages are O(1)
              about 0, and the margin/global reversals sit near +24.8 / -14.5)
      r2      coefficient of determination of learned E_ij against true E_ij
      slope   slope of the least-squares fit of learned on true
      n_edges how many edges entered the comparison
      true, learned   the two (n_edges,) numpy arrays, for the scatter

    TWO WAYS OF GETTING `learned`, chosen by what the model is. The known-ODE
    student holds E as a named parameter and `get_learned_reversal_per_edge`
    reads it. THE GNN HOLDS NO SUCH PARAMETER -- its message is
    `W_gnn * g_phi(v_j, a_j, v_i, a_i)^2` and E is implicit in the shape of the
    MLP -- so when `config`, `edges` and `x_ts` are supplied it falls back to
    :func:`extract_conductance_params_from_gnn`, which reads E out as the
    zero-crossing in v_i of the message divided by v_j. That path adds two keys:

      fit_r2_median  median per-edge R2 of the straight line the extraction
                     assumes. THIS IS THE PRECONDITION, not a detail: below
                     roughly 0.9 the message is not affine in v_i, the GNN has
                     not found the conductance form, and the E and W beside it
                     describe nothing.
      w_r2_scaled / w_scale   R2 of the extracted W against the true W after
                     dividing out the one global gain the GNN cannot pin down
                     (see :func:`r2_up_to_scale`), and the gain that was divided
                     out. The slope after that is 1 by construction, so it is not
                     reported.

    Returns None when either side has no reversals.
    """
    import numpy as np

    gt_rev = getattr(ode_params, "reversal_per_edge", None)
    if gt_rev is None or getattr(ode_params, "E_exc", None) is None:
        return None
    # Under torch.compile the parameters live on the wrapped module.
    core = getattr(model, "_orig_mod", model)
    if not hasattr(core, "get_learned_reversal_per_edge"):
        if config is None or edges is None or x_ts is None:
            return None
        return _reversal_metrics_from_gnn(core, ode_params, config, edges, x_ts)

    with torch.no_grad():
        true = to_numpy(gt_rev()).ravel()
        learned = to_numpy(
            core.get_learned_reversal_per_edge(ode_params.edge_index)).ravel()
        # PER-NEURON arrays alongside the per-edge ones. The per-edge form is what
        # the RMSE is defined on -- E_ij is an edge quantity -- but it weights each
        # neuron by its in-degree and hides how many neurons sit behind a stripe.
        # The per-neuron form is what the reversals actually ARE, one E_exc and one
        # E_inh per postsynaptic cell, and it is what the granularity-adaptive
        # panel in plot_reversal_scatter draws.
        l_exc, l_inh = core.get_learned_reversals()
        per_neuron = dict(
            true_exc=to_numpy(ode_params.E_exc).ravel(),
            true_inh=to_numpy(ode_params.E_inh).ravel(),
            learned_exc=to_numpy(l_exc).ravel(),
            learned_inh=to_numpy(l_inh).ravel(),
            type_index=(to_numpy(core.type_index).ravel()
                        if getattr(core, "type_index", None) is not None else None),
        )
        # Neurons no edge ever targets get no gradient on their reversals, so their
        # learned value is still the initialisation. Recorded, not filtered: a
        # violin that silently drops them would misreport how much of the
        # parameterisation the data actually constrains.
        _dst = to_numpy(ode_params.edge_index[1]).ravel()
        # Per-EDGE cell type for the violin panel: the POSTSYNAPTIC cell, since
        # E belongs to the neuron the driving force (E - v_i) acts on.
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
    edges = edges.to(device)

    # 1. Activity statistics
    mu_activity, sigma_activity = compute_activity_stats(x_ts, device)

    # 2. g_phi correction factor per presynaptic neuron j
    if is_conductance_gnn(config.graph_model.signal_model_name):
        # g_phi depends on both endpoints here (vi, vj, ai, aj) — a
        # synthetic vi=0 domain sweep isn't representative, so this uses
        # real (edge, frame) local slopes instead. See
        # compute_g_phi_correction_conductance for the method and the
        # comparison that selected it.
        slopes = compute_g_phi_correction_conductance(model, config, edges, x_ts, n_neurons)
        g_phi_fitted = {'correction': slopes, 'slopes': slopes}
    else:
        v_ranges, g_phi_curves, valid = evaluate_g_phi_curves(
            model, config, n_neurons, mu_activity, sigma_activity, device, edges=edges)

        if ode_params is not None:
            g_phi_fitted = ode_params.fit_g_phi_curves(v_ranges, g_phi_curves)
        else:
            # Fallback: linear slope (legacy behavior)
            rr_t = torch.tensor(v_ranges, dtype=torch.float32)
            func_t = torch.tensor(g_phi_curves, dtype=torch.float32)
            slopes, _ = _vectorized_linear_fit(rr_t, func_t)
            slopes[~valid] = 1.0
            g_phi_fitted = {'correction': slopes, 'slopes': slopes}

        g_phi_fitted['correction'][~valid] = 1.0

    g_phi_correction = g_phi_fitted['correction']

    # 3. f_theta slopes
    slopes_f_theta, offsets_f_theta = extract_f_theta_slopes(
        model, config, n_neurons, mu_activity, sigma_activity, device)

    # 4. Compute grad_msg over multiple frames and take median
    n_frames = x_ts.voltage.shape[0]
    frame_indices = np.linspace(n_frames // 10, n_frames - 100, n_grad_frames, dtype=int)
    data_id = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)

    was_training = model.training
    model.eval()

    grad_list = []
    for k in frame_indices:
        state = x_ts.frame(int(k)).to(device)
        with torch.no_grad():
            _, in_features, _ = model(state, edges, data_id=data_id, return_all=True)
        grad_k = compute_grad_msg(model, in_features, config)
        grad_list.append(grad_k)

    if was_training:
        model.train()

    # Median on CPU: torch.median on CUDA also returns tie-break indices, which
    # have no deterministic implementation, so under
    # use_deterministic_algorithms(True) the GPU call raises. Only .values is
    # used here and the values themselves are well-defined, so the reduction
    # moves to CPU rather than being excused with warn_only. This is a
    # diagnostic path (weight correction for the R^2 checkpoint), a few frames
    # x n_neurons, so the transfer is not on the training hot path -- but it
    # does feed W_corrected_R2, which is why it must not be left nondeterministic.
    grad_msg = torch.stack(grad_list).cpu().median(dim=0).values.to(device)  # (N,)

    # 5. Corrected weights using model-specific g_phi correction
    corrected_W = compute_corrected_weights(
        model, edges, slopes_f_theta, g_phi_correction, grad_msg)

    return corrected_W, slopes_f_theta, g_phi_correction, offsets_f_theta, g_phi_fitted


# ------------------------------------------------------------------ #
#  Calcium / time-series spectral comparison
# ------------------------------------------------------------------ #
# Used by:
#   - figures/zebrafish/fig_zebrafish_calcium_baseline.py — power-
#     spectrum panel on real vs.\ modelled ΔF/F.
#   - figures/zebrafish/best_match_to_model.py (and analogs) — score
#     every observed neuron against every modelled neuron by spectral
#     distance (and/or correlation) to find best-matching candidates
#     from the ~70 k-cell recording.
# Implemented as a thin numpy layer so it stays cheap and reusable.

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
    # Vectorised pairwise: tile to (N_model, N_obs, F) — cheap when
    # F is small (~rfft of a few-min trace) but heavy when N_obs is
    # ~70 k. Chunk over N_obs to bound memory.
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
# WHY THIS EXISTS. `recovery_param_metrics` is already the single entry point for
# R2 everywhere. The EXTRACTION side -- producing the `learned` array that gets
# scored -- had five independent implementations that disagreed, and the trainer's
# headline connectivity_r2 arrived as a return value of `plot_training_gnn`, which
# got it from `plot_recovery_panels`: the 2x2 panel drawer. Metric and figure were
# the same call, so `test_plot` re-derived the number independently and the two
# could differ. On one known-ODE checkpoint they did, by a factor of 30 (-1.22
# outlier-free against -34.35 unfiltered), because nothing forced the two paths to
# filter the same quantity the same way.
#
# `extract_recovered_params` returns the (ground truth, learned) PAIRS, not
# metrics. `score_recovery` turns those into numbers via recovery_param_metrics.
# Panels become consumers of the arrays rather than producers of the metric.
#
#
# ESTIMATORS. What "the learned W" means depends on where W sits in the model.
#
#   direct          W is a named parameter. Known-ODE / linear. W**2 under
#                   w_squared, since there the stored value is sqrt(conductance).
#
#   gain_corrected  W reaches dv/dt through two learned functions, so the gains
#                   they apply must be divided out:
#
#                       msg_i    = sum_j W_ij * g_phi(v_j, ...)
#                       dv_i/dt  = f_theta(v_i, a_i, msg_i, ...)
#
#                       corrected_W_ij = -W_ij / slope_f_theta[i]
#                                               * grad_msg[i]
#                                               * slope_g_phi[j]
#
#                   slope_g_phi[j]   = d g_phi / d v_j at the presynaptic neuron;
#                   grad_msg[i]      = d f_theta / d msg_i;
#                   slope_f_theta[i] = d f_theta / d v_i, the leak, ~ -1/tau_i.
#                   The two f_theta terms form a RATIO, so tau cancels -- which is
#                   why a W error under this estimator does not track tau_i
#                   (measured on a conductance known-ODE: corr = +0.014).
#
#   edge_line_fit   Conductance data. The generator's message is affine in v_i,
#                   so dividing it by v_j leaves a straight line per edge:
#
#                       msg_ij / v_j = W_ij * (E_i - v_i)
#                       slope = -W_ij,   intercept = W_ij * E_i
#
#                   Gated: the median per-edge R2 of that line is the test that
#                   the message has the conductance form at all. Below
#                   recovery.gate_fit_r2 the W and E beside it describe nothing.
#
#   jacobian        MLP baseline, no explicit W. Effective connectivity from
#                   dF/dv averaged over frames.
#
#
# tau AND V_rest. Two estimators, and which one ran has never been recorded.
#
#   direct          Named parameters. tau = softplus(raw_tau), V_rest = V_rest.
#
#   f_theta_slope   A GNN buries both inside f_theta, but the local linearisation
#                   dv_i/dt ~ slope_i * v_i + offset_i recovers them:
#
#                       tau_i    = -1 / slope_i          (clipped to [0, 1])
#                       V_rest_i = -offset_i / slope_i
#
#                   slope_i and offset_i come from extract_f_theta_slopes. The
#                   clip and the slope==0 guard are why a degenerate fit returns
#                   tau = 1.0 rather than an infinity.
#
#
# E_ij. Conductance-generated data only. None elsewhere, which is information --
# a current generator has no reversal potential to recover -- not a failure.
#
#   direct          get_learned_reversal_per_edge(): a named parameter,
#                   where(edge_is_inh, E_inh[dst], E_exc[dst]).
#
#   edge_line_fit   The same line as the W estimator above, read from the other
#                   end: E_ij = -intercept / slope. W and E come from ONE fit and
#                   so cannot disagree about it.
#
#
# msg_i. One path, no estimator, no family branch -- the reason it is the model
# the other five are being made to look like. Compares the model's own message,
# via forward(..., return_all=True), against the generator's rebuilt from
# ode_params. The one quantity the conductance degeneracy does not touch: W and E
# trade off INSIDE the message, so msg_i scores the product the trajectory
# depends on rather than a factorisation the data cannot resolve.
#
#
# NONE, NEVER ZEROS. A quantity the model does not have is absent. The ladder this
# replaces fell through to np.zeros(n_neurons), which reached the analysis log
# looking exactly like a measurement of zero.

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
    estimator: dict = field(default_factory=dict)   # 'W' -> 'gain_corrected'
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
    """True when the generator had a reversal potential. Asked of the DATA, not
    the model: the same conductance known-ODE is trained on a current-generated
    teacher in the distillation runs and on conductance data in the recovery runs,
    so its name cannot answer."""
    return getattr(ode_params, "E_exc", None) is not None


def resolve_W_estimator(model, ode_params, config) -> str:
    """Which W estimator this run uses. `auto` reads the model's family tag and
    the dataset's generator; an explicit config value overrides."""
    from connectome_gnn.models.utils import model_family

    mode = getattr(getattr(config, "recovery", None), "W_mode", "auto")
    mode = getattr(mode, "value", mode)
    if mode != "auto":
        return mode

    family = model_family(model)
    if family == "mlp":
        return "jacobian"
    if family in ("linear", "known_ode"):
        return "direct"
    # A GNN has no named W in the units of ode_params.W. On conductance data the
    # per-edge line fit matches the generative form exactly and carries its own
    # goodness-of-fit; on current data there is no (E - v_i) term to fit, so the
    # gain correction is the only option.
    return "edge_line_fit" if _is_conductance_data(ode_params) else "gain_corrected"


def _pair(gt, learned):
    """Trim two arrays to a common length and drop non-finite entries.

    Returns None rather than an empty pair when nothing survives, so an absent
    quantity and a quantity that produced only NaN are indistinguishable to the
    caller -- both mean "no measurement", which is the honest reading."""
    if gt is None or learned is None:
        return None
    # Both sides arrive in whatever the source uses: ode_params.gt_tau() returns
    # numpy, a model accessor returns a tensor, a test passes a list.
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
    return gt[ok], learned[ok]


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
    """Every learned circuit parameter of a trained model, paired with its truth.

    Returns arrays, not metrics -- see the block comment above for the estimators
    and the equation each one inverts. Pass `need` to skip work: the f_theta slope
    fit runs over every neuron and msg_i costs ten forward passes, so a caller that
    wants only W should say so.

    Every quantity is best-effort. One that cannot be computed is absent from
    `pairs`, never a zero-filled array standing in for a measurement.

    ----------------------------------------------------------------------------
    EVERY METRIC THIS PRODUCES, AND WHAT EACH ONE MEANS
    ----------------------------------------------------------------------------
    This function returns ARRAYS; :func:`score_recovery` turns them into the
    key-value metrics that reach `results/metrics.txt`, the per-slot analysis log
    and the LLM exploration instructions. The catalogue is written here because
    this is where a reader arrives asking "what can I measure", and because the
    names are a contract: the instruction files rank slots on these spellings, so
    renaming one silently breaks an agentic loop that has no way to notice.

    FIVE QUANTITIES, and the key each is emitted under (`_KEY`):

        quantity   key       what it is
        W          Wij       per-edge synaptic weight. On conductance data this is
                             the CONDUCTANCE (non-negative); on current data it is
                             the signed current weight.
        tau        tau       per-neuron membrane time constant.
        V_rest     V_rest    per-neuron resting potential.
        E_ij       Eij       per-edge reversal potential, i.e. the voltage each
                             edge's driving force (E_ij - v_i) points toward.
                             CONDUCTANCE-GENERATED DATA ONLY -- absent elsewhere,
                             which is information, not a failure.
        msg_i      msg_i     the aggregated per-neuron message, sum over incoming
                             edges of W_ij * act(v_j) * (E_ij - v_i). The one
                             quantity the W/E degeneracy does not touch.
        gain       gain      per-neuron output gain, where the model has one.
        bias       bias      per-neuron output bias, where the model has one.

    FOR EACH of the above, `score_recovery` emits:

        <key>_R2          coefficient of determination of learned against true, on
                          the IDENTITY line (not a free-slope fit), so a scale
                          error costs R2 rather than being absorbed. Outlier-
                          filtered for W, tau and V_rest; unfiltered for Eij and
                          msg_i, which have no published tolerance band.
        <key>_slope       identity-line slope of the same scatter. 1.0 is perfect.
                          FOR W THIS IS THE GAIN: 2.4 means the learned conductance
                          is 2.4x the true one. It is the third ranking key of the
                          known-ODE exploration.
        <key>_rmse        root-mean-square error, in that quantity's own units.
                          Read it against the SPREAD OF THE TRUTH, not a fixed bar:
                          the reversals of one twin span 8 voltage units and those
                          of another span 40.
        <key>_n           how many elements entered the comparison. Context, but a
                          drop in it means pairs were trimmed or non-finite.

    AND ONLY WHERE AN OUTLIER THRESHOLD APPLIES (W, tau, V_rest -- see
    :func:`_thresh_for`; None for Eij and msg_i):

        <key>_R2_all       the same R2 WITHOUT outlier filtering. The gap between
                           it and <key>_R2 says how much of the recovery rests on
                           a tail. Rank on the filtered one, report both.
        <key>_n_outliers   how many elements the threshold removed.
        <key>_pct_outliers the same as a percentage of <key>_n.

    PROVENANCE, emitted whenever the extractor recorded it. These are STRINGS, not
    numbers, and they exist because the same key can be produced by different
    estimators on different runs -- a comparison across runs that does not check
    them may be comparing two different measurements:

        <key>_estimator   which path produced the number:
                            direct         a named parameter, read straight off.
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

    ALWAYS, per quantity:

        <key>_rel_err_median  median of |learned - true| / max(|true|, 1e-6) over
                              the full sample, and
        <key>_rel_err_iqr     its interquartile range. Median and IQR, never mean
                              and SD, which the heavy tails inflate.

    ONLY FOR W AND msg_i, the two a GNN pins down up to a gain:

        <key>_gain        learned ~= gain * true, the least-squares factor through
                          the origin with the truth as regressor (see
                          :func:`r2_up_to_scale`). 1.0 is perfect; 2.0 means twice
                          too large. THE ONE CONVENTION -- the reciprocal
                          <true,learned>/<learned,learned> is not used anywhere.
        <key>_R2_scaled   the identity-line R2 once that gain is divided out. The
                          number to steer by while <key>_R2 is deeply negative.

    ONLY FOR W, scale ignored altogether (over edges whose true weight is non-zero):

        Wij_pearson       Pearson r of learned against true. High with a low
                          Wij_R2 reads "wiring recovered, scale not".
        Wij_zscored_R2    identity-line R2 of the two z-scored vectors.

    THREE KEYS THAT ARE NOT PER-QUANTITY:

        Wij_R2_uncorrected  W scored BEFORE the gain correction. Says how much of
                            the recovery the correction is responsible for. Only
                            on the `gain_corrected` path -- note "uncorrected"
                            means before the CORRECTION, not before outlier
                            filtering, two senses the old `raw_W_R2` conflated.
        Eij_gate            median per-edge R2 of the straight line the
                            `edge_line_fit` extraction assumes. THIS IS A
                            PRECONDITION, not a detail: below roughly 0.9 the
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
                            broke, NOT because the model lacks them.

    A QUANTITY GATED OUT EMITS NOTHING AT ALL. `rec.valid[q]` False makes
    `rec.get(q)` return None, so `score_recovery` skips the whole family rather
    than reporting a number nothing stands behind. Absence is the signal.

    WHERE THE NUMBERS LAND, spelled identically in all three places:

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

    RENAMED ON 2026-09-11 (tools/extraction_gate.py carries the map for logs
    written before): `connectivity_R2` / `W_corrected_no_outliers_R2` -> `Wij_R2`;
    `W_corrected_R2` -> `Wij_R2_all`; `connectivity_R2_scaled` -> `Wij_R2_scaled`;
    `w_scale` -> `Wij_gain`; `connectivity_pearson_r` / `W_structure_r` ->
    `Wij_pearson`; `W_zscored_R2` -> `Wij_zscored_R2`; `raw_W_R2` ->
    `Wij_R2_uncorrected`; test_plot's old `tau_R2` (unfiltered) -> `tau_R2_all`
    and `tau_no_outliers_R2` -> `tau_R2`, likewise V_rest; `Eij_n_edges` ->
    `Eij_n`; the trainer's `connectivity_r2` / `vrest_r2_clean` / `tau_r2_clean`
    columns -> `Wij_R2` / `V_rest_R2` / `tau_R2` in their own files;
    `msgi_r2.log`'s `r2_scaled` / `scale` -> `msg_i_R2_scaled` / `msg_i_gain`;
    `gnn_conductance_fit.log`'s `fit_r2_median` -> `Eij_gate`.
    """
    rec = RecoveredParams()
    if ode_params is None:
        return rec
    if n_neurons is None:
        n_neurons = int(getattr(model, "a", np.zeros((0, 0))).shape[0]) or None

    estimator = resolve_W_estimator(model, ode_params, config)
    try:
        if estimator == "direct":
            _extract_direct(rec, model, ode_params, edges, n_neurons)
        elif estimator in ("gain_corrected", "edge_line_fit"):
            _extract_gnn(rec, model, ode_params, config, edges, x_ts, device,
                         n_neurons, estimator, need)
        else:
            # jacobian: the MLP baseline's effective connectivity. Left to its
            # existing caller rather than moved, because it compares a dense
            # n x n matrix and not a per-edge vector like every other estimator.
            rec.estimator["W"] = estimator
    except Exception as exc:
        # Caught rather than raised because this runs inside the training loop and
        # a failed diagnostic must not kill a run -- but NOT silently: an aborted
        # extraction drops every quantity at once, which reads downstream as "this
        # model has no tau" rather than "the extractor broke".
        rec.diagnostics["extraction_error"] = f"{type(exc).__name__}: {exc}"
        print(f"\033[91mextract_recovered_params({estimator}) failed: "
              f"{type(exc).__name__}: {exc}\033[0m")

    # msg_i ON BOTH DATA FAMILIES: the true message on current data is simply
    # W_ij * act(v_j) with no driving-force factor, and compute_msg_i_recovery
    # builds that itself (None only when there is no ground-truth W). It is the
    # one recovery number that needs no estimator choice, so a current run wants
    # it beside Wij_R2 exactly as a conductance run does.
    if "msg_i" in need and x_ts is not None:
        try:
            out = compute_msg_i_recovery(model, ode_params, x_ts, edges, device)
        except Exception as exc:
            # Loud, like the family extractors above: a silent None here read
            # downstream as "this model has no msg_i".
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
    }.get(quantity)          # None -> no filtering, which is right for E_ij/msg_i


# The emitted key for each quantity. Wij/Eij/msg_i are symbol-first; tau and
# V_rest keep the spelling every consumer already reads.
_KEY = {"W": "Wij", "tau": "tau", "V_rest": "V_rest",
        "E_ij": "Eij", "msg_i": "msg_i", "gain": "gain", "bias": "bias"}


def score_recovery(rec: RecoveredParams, config=None) -> dict:
    """Score every recovered quantity through the one R2 entry point.

    THE EXTRACTOR OWNS THE OUTLIER THRESHOLD, which is the point: the headline
    `<q>_R2` is outlier-free and `<q>_R2_all` is not, decided here rather than at
    each call site. Previously the trainer reported the filtered number and
    test_plot the unfiltered one under similar names.
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

    # The uncorrected parameter is scored on R2 alone: it exists to say how much
    # of the recovery the gain correction is responsible for, not as a second
    # headline. Note this is "before the correction", NOT "before outlier
    # filtering" -- the two senses that raw_W_R2 and connectivity_full_sample_R2
    # used to share the word "raw" for.
    unc = rec.pairs.get("W_uncorrected")
    if unc is not None:
        out["Wij_R2_uncorrected"] = float(
            recovery_param_metrics(unc[0], unc[1],
                                   _thresh_for("W", config))["r2_clean"])

    # SCALE-FREE COMPANIONS for the two quantities a GNN pins down only up to a
    # gain. `<key>_gain` follows the one convention (learned ~= gain * true, see
    # r2_up_to_scale); `<key>_R2_scaled` is the R2 once it is divided out.
    for quantity in ("W", "msg_i"):
        pair = rec.get(quantity)
        if pair is None:
            continue
        key = _KEY[quantity]
        s_ = r2_up_to_scale(pair[0], pair[1])
        out[f"{key}_R2_scaled"] = float(s_["r2"])
        out[f"{key}_gain"] = float(s_["gain"])
    # W STRUCTURE, ignoring scale altogether: Pearson r over the edges whose true
    # weight is non-zero, and the identity-line R2 of the two z-scored vectors.
    # High Wij_pearson with a low Wij_R2 reads as "wiring recovered, scale not";
    # low Wij_pearson as "wiring wrong". These were W_structure_r / W_zscored_R2.
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

    # Underscore-prefixed diagnostics are intermediates shared between quantities
    # (the f_theta slopes, the g_phi correction) and are numpy arrays; they stay
    # on the object for the caller and never reach the key-value log.
    for k, v in rec.diagnostics.items():
        if not k.startswith("_"):
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
#  ONE VOCABULARY, ONE WRITER
#
#  Every number a run reports about a recovered quantity is a `<key>_<stat>` from
#  score_recovery, and it is spelled the same in the three places it lands:
#
#    tmp_training/<key>.log   one CSV per quantity, one row per checkpoint, the
#                             TRAJECTORY (Wij.log, tau.log, V_rest.log, Eij.log,
#                             msg_i.log; plus cluster.log and rollout.log, which
#                             are not recovered parameters but follow the layout)
#    results/metrics.txt      `key: value`, the FINAL numbers `-o test_plot` writes
#    the per-slot analysis log the LLM exploration reads (same lines)
#
#  Before 2026-09-11 the trainer wrote a positional-column metrics.log with its
#  own column names (connectivity_r2, vrest_r2_clean, ...), test_plot wrote
#  results/metrics.txt with a third set (W_corrected_no_outliers_R2, tau_R2 for
#  the UNFILTERED number, ...), and readers guessed. The gate in
#  tools/extraction_gate.py carries the rename map for logs written before.
# --------------------------------------------------------------------------- #

RECOVERY_KEYS = tuple(_KEY.values())

# Column order of tmp_training/<key>.log, and the order results/metrics.txt lists
# them in. Shared stats first, then what only some quantities have.
_COMMON_STATS = ("R2", "R2_all", "slope", "rmse", "n", "n_outliers", "pct_outliers",
                 "rel_err_median", "rel_err_iqr")
_EXTRA_STATS = {
    "Wij":   ("R2_scaled", "gain", "pearson", "zscored_R2", "R2_uncorrected"),
    "Eij":   ("gate", "pct_wrong_slope"),
    "msg_i": ("R2_scaled", "gain"),
}


def recovery_log_columns(key):
    """The `<key>_<stat>` columns of tmp_training/<key>.log, in file order."""
    return tuple(f"{key}_{st}" for st in _COMMON_STATS + _EXTRA_STATS.get(key, ()))


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


def training_log_append(log_dir, name, iteration, row):
    """Append one row to tmp_training/<name>.log, writing the header first.

    `row` is an ordered {column: value}; the header is `iteration,` + its keys.
    Readers parse by NAME (training_log_read), never by position, so a column
    can be added without shifting anyone.
    """
    tmp = os.path.join(log_dir, "tmp_training")
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, f"{name}.log")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f:
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
        # A gated-out quantity (a conductance GNN below the line-fit gate) has
        # no <key>_R2 but still has something to record: the gate itself for
        # E_ij, and for W the fact that the estimator ran. Those rows are all
        # nan except the gate, which is the number that says WHY.
        present = (f"{key}_R2" in scored
                   or (key == "Eij" and "Eij_gate" in scored)
                   or (key == "Wij" and "Wij_estimator" in scored))
        if not present:
            continue
        cols = recovery_log_columns(key)
        training_log_append(log_dir, key, iteration,
                            {c: scored.get(c) for c in cols})


def metrics_lines(scored):
    """`key: value` lines for results/metrics.txt and the analysis log.

    Per quantity, the numeric columns in file order, then the estimator and the
    correction strings. Keys score_recovery emits that belong to no quantity
    (a diagnostic such as extraction_error) come last, unchanged.
    """
    lines, seen = [], set()
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


def write_recovery_metrics(scored, log_dir, log_file=None, logger=None):
    """THE writer of recovered-parameter metrics for `-o test_plot`.

    Appends metrics_lines(scored) to results/metrics.txt, to the analysis log
    `log_file` when given, and to `logger`. Nothing else writes a `<key>_<stat>`
    line to either file.
    """
    lines = metrics_lines(scored)
    if not lines:
        return
    path = os.path.join(log_dir, "results", "metrics.txt")
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
                     return_features=False):
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
    """
    from connectome_gnn.sparsify import clustering_gmm

    def _host(x):
        # Callers hand over tensors on the GPU (type_list, edges, model.a) as
        # readily as arrays; np.asarray cannot read a CUDA tensor.
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
        # A per-edge line fit leaves nan where the fit failed; a mixture model
        # cannot take nan, and one bad edge must not blank the whole feature.
        # Unmeasured is read as zero conductance here, for the features only.
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        m = min(w.size, e.shape[1])
        if m > 0:
            feats.append(_connectivity_stats(w[:m], e[0, :m], e[1, :m], n))
    if not feats:
        return None
    X = np.column_stack(feats)
    X = X[:, np.isfinite(X).all(axis=0)]      # drop any column a nan still reached
    if X.shape[1] == 0:
        return None
    if n_components is None:
        n_components = min(100, n - 1)
    res = clustering_gmm(X, np.asarray(type_list).ravel()[:n], n_components=n_components)
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


def _extract_gnn(rec, model, ode_params, config, edges, x_ts, device, n_neurons,
                 estimator, need):
    """A GNN keeps no parameter in the units of ode_params.W, so every quantity
    here is inferred. See the block comment above for the two W estimators and the
    f_theta-slope inversion that yields tau and V_rest."""
    core = getattr(model, "_orig_mod", model)
    gate = getattr(getattr(config, "recovery", None), "gate_fit_r2", 0.9)

    # Only when W is actually wanted: effective_true_weights needs the edge index,
    # and a caller asking for tau alone (the trainer's per-checkpoint dynamics
    # pass) legitimately has none to give.
    want_W = "W" in need and edges is not None
    gt_W = None
    if want_W and getattr(ode_params, "W", None) is not None:
        gt_W = np.asarray(ode_params.effective_true_weights(
            to_numpy(ode_params.W), to_numpy(edges), n_neurons))

    if want_W and estimator == "edge_line_fit":
        ext = extract_conductance_params_from_gnn(core, config, edges, x_ts)
        fit_r2 = float(np.nanmedian(ext["fit_r2"]))
        rec.diagnostics["Eij_gate"] = fit_r2
        rec.diagnostics["Eij_pct_wrong_slope"] = ext.get("pct_wrong_slope", float("nan"))
        # Below the gate the message is not affine in v_i, so the W and E the
        # line produced describe nothing. Recorded as invalid rather than
        # dropped: "we measured and it failed" is not "we did not measure".
        ok = fit_r2 >= gate
        rec.pairs["W"] = _pair(gt_W, ext["W"])
        rec.estimator["W"] = "edge_line_fit"
        rec.diagnostics["_W_learned_full"] = np.asarray(to_numpy(ext["W"])).ravel()
        rec.correction["W"] = "msg_ij / v_j = W_ij * (E_i - v_i); W_ij = -slope"
        rec.valid["W"] = ok
        if "E_ij" in need and _is_conductance_data(ode_params):
            rec.pairs["E_ij"] = _pair(ode_params.reversal_per_edge(), ext["E"])
            rec.estimator["E_ij"] = "edge_line_fit"
            rec.correction["E_ij"] = "E_ij = -intercept / slope, the same line as W"
            rec.valid["E_ij"] = ok

    elif want_W and estimator == "gain_corrected":
        corrected_W, slopes_f, g_phi_corr, offsets_f, _ = compute_all_corrected_weights(
            core, config, edges, x_ts, device, ode_params=ode_params)
        rec.pairs["W"] = _pair(gt_W, to_numpy(corrected_W).squeeze())
        rec.estimator["W"] = "gain_corrected"
        rec.diagnostics["_W_learned_full"] = to_numpy(corrected_W).squeeze().ravel()
        rec.correction["W"] = "g_phi[j] * dftheta_dmsg[i] / dftheta_dv[i]"
        # The uncorrected parameter, for the comparison that says how much of the
        # recovery the correction is responsible for.
        rec.pairs["W_uncorrected"] = _pair(gt_W, to_numpy(get_model_W(core)).squeeze())
        rec.diagnostics["_slopes_f_theta"] = slopes_f
        rec.diagnostics["_offsets_f_theta"] = offsets_f
        rec.diagnostics["_g_phi_correction"] = g_phi_corr

    # tau and V_rest come out of the SAME f_theta linearisation, so they are
    # derived together from one slope fit rather than two.
    if ("tau" in need or "V_rest" in need) and x_ts is not None:
        slopes = rec.diagnostics.get("_slopes_f_theta")
        offsets = rec.diagnostics.get("_offsets_f_theta")
        if slopes is None:
            mu, sigma = compute_activity_stats(x_ts, device)
            slopes, offsets = extract_f_theta_slopes(core, config, n_neurons,
                                                    mu, sigma, device)
        # ode_params.derive_tau, NOT the module-level derive_tau: the inversion is
        # family-specific and the two disagree. CX's f_theta slope is -alpha/tau,
        # so it divides by alpha and clips to [0, 10], where the module-level
        # helper assumes -1/slope and clips to [0, 1]. Using the wrong one is
        # silent -- it returns a plausible number.
        if "tau" in need and ode_params.has_tau():
            rec.pairs["tau"] = _pair(ode_params.gt_tau(n_neurons),
                                     ode_params.derive_tau(np.asarray(slopes), n_neurons))
            rec.estimator["tau"] = "f_theta_slope"
            rec.correction["tau"] = "tau_i from dftheta_dv[i], per ode_params.derive_tau"
        if "V_rest" in need and ode_params.has_vrest():
            rec.pairs["V_rest"] = _pair(
                ode_params.gt_vrest(n_neurons),
                ode_params.derive_vrest(np.asarray(slopes), np.asarray(offsets), n_neurons))
            rec.estimator["V_rest"] = "f_theta_slope"
            rec.correction["V_rest"] = "V_rest_i = -offset_i / dftheta_dv[i]"

    # E_ij under gain_corrected still comes from the line fit, which is the only
    # way to read a reversal out of a GNN; compute_reversal_metrics runs it.
    if ("E_ij" in need and "E_ij" not in rec.pairs
            and _is_conductance_data(ode_params)):
        _rev = compute_reversal_metrics(core, ode_params, config=config,
                                        edges=edges, x_ts=x_ts)
        if _rev is not None:
            rec.pairs["E_ij"] = _pair(_rev["true"], _rev["learned"])
            rec.estimator["E_ij"] = "edge_line_fit"
            if "fit_r2_median" in _rev:
                rec.diagnostics["Eij_gate"] = _rev["fit_r2_median"]
                rec.diagnostics["Eij_pct_wrong_slope"] = _rev.get("pct_wrong_slope", float("nan"))
                rec.valid["E_ij"] = _rev["fit_r2_median"] >= gate
