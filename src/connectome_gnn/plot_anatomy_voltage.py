"""Anatomy-voltage snapshot rendering for the `-o test --anatomy_voltage`
pipeline.

Two organism-specific render functions consume:
  - ``h_traj``: (T, N) subthreshold trajectory from one test trial
  - ``circuit``: a registered ``Circuit`` (with ``body_ids`` +
    ``provenance['anatomy_dir']`` populated)
  - ``plot_cfg``: ``PlottingConfig`` with the ``anatomy_voltage_*`` fields
  - ``out_path``: where to write the PNG

Behaviour: project every SWC skeleton in the circuit's anatomy dir to 2D
(elev, azim from the plotting config), and overlay a green tint whose
alpha follows the per-neuron z-score at frame ``anatomy_voltage_frame_idx``
of the supplied ``h_traj``. The function is a no-op when ``h_traj`` is
empty or when the circuit lacks ``body_ids``.

The companion stand-alone scripts
(``figures/zebrafish/fig_zebrafish_anatomy_3d_voltage_anim.py`` and
``figures/drosophila_cx/fig_cx_anatomy_3d_voltage_anim.py``) remain the
authoritative source for full frame-sequence animations + ROI overlays
+ swim/heading traces; this module is the single-PNG shortcut routed
through ``data_test``.
"""
from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np

# Light imports only at module load; matplotlib + navis are deferred to
# the render functions so test runs that don't request --anatomy_voltage
# don't pay the import cost.


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _project_2d(xyz: np.ndarray, elev: float, azim: float) -> np.ndarray:
    """Project (M, 3) world coords to (M, 2) screen coords matching the
    matplotlib mplot3d convention for (elev, azim) in degrees. Verbatim
    from ``figures/zebrafish/fig_zebrafish_anatomy_3d_HD._project_2d``."""
    e = np.deg2rad(elev)
    a = np.deg2rad(azim)
    ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    R = np.array([[-sa,        ca,         0.0],
                  [-ca * se, -sa * se,     ce]])
    return xyz @ R.T


def _zscore_per_neuron(h_traj: np.ndarray, warmup: int = 0) -> np.ndarray:
    """(T, N) -> (T, N) per-neuron z-score using mean + std over
    [warmup, T). Returns zeros for any neuron whose std collapses
    (e.g. silent units).

    Matches the standalone script's LUT computation in
    ``fig_zebrafish_anatomy_3d_voltage_anim.main`` (lines 1465-1469),
    which uses ``warmup=0`` — mean/std over the FULL rollout. Earlier
    versions of this helper used ``warmup=50`` to exclude the bump-
    formation transient, but that made the test-pipeline output
    visibly inconsistent with the standalone (more cells lit, because
    the steady-state baseline is tighter without the transient
    contributing to mu/sd)."""
    h = np.asarray(h_traj, dtype=np.float32)
    if h.ndim != 2:
        raise ValueError(f"h_traj must be (T, N), got {h.shape}")
    T = h.shape[0]
    if T <= warmup:
        mu = h.mean(axis=0, keepdims=True)
        sd = h.std(axis=0, keepdims=True) + 1e-6
    else:
        mu = h[warmup:].mean(axis=0, keepdims=True)
        sd = h[warmup:].std(axis=0, keepdims=True) + 1e-6
    return (h - mu) / sd


def _build_bodyid_index(anatomy_dir: str) -> "dict[int, tuple[str, str]]":
    """Map bodyId -> (absolute SWC path, type string) by reading
    ``<anatomy_dir>/index.csv`` if present; otherwise by walking
    ``<anatomy_dir>/skeletons/`` and parsing filenames of the form
    ``<safe_type>__<bodyId>.swc``."""
    import pandas as pd

    idx_csv = os.path.join(anatomy_dir, "index.csv")
    out: "dict[int, tuple[str, str]]" = {}
    if os.path.isfile(idx_csv):
        df = pd.read_csv(idx_csv)
        for _, row in df.iterrows():
            bid = int(row["bodyId"])
            swc = str(row["swc"])
            t = str(row.get("type", ""))
            path = swc if os.path.isabs(swc) else os.path.join(anatomy_dir, swc)
            out[bid] = (path, t)
        return out

    skel_dir = os.path.join(anatomy_dir, "skeletons")
    if not os.path.isdir(skel_dir):
        return out
    for fname in os.listdir(skel_dir):
        if not fname.endswith(".swc"):
            continue
        stem = fname[:-4]
        safe_t, _, bid_str = stem.rpartition("__")
        try:
            bid = int(bid_str)
        except ValueError:
            continue
        out[bid] = (os.path.join(skel_dir, fname), safe_t)
    return out


def _load_skeletons_for_circuit(circuit, downsample: int = 10):
    """Load SWC skeletons for each neuron in ``circuit.body_ids`` order.

    Looks for skeletons under ``circuit.provenance['anatomy_dir']`` and,
    if the circuit declares extras, ``provenance['anatomy_extra_dirs']``
    (e.g. the IPN12 cache). Missing skeletons become ``None`` entries
    in the returned list, so the model→skeleton alignment is preserved
    by index even if some neurons lack anatomy.

    Returns
    -------
    neurons : list[navis.TreeNeuron | None]
        Length N, one entry per circuit neuron in model order.
    type_names : list[str]
        Per-neuron canonical type string from the loader (zebrafish:
        ``IPNd13B`` etc; fly: ``EPG``/``PEN``/...).
    has_skel : np.ndarray  (N,) bool
        True where a skeleton was found.
    """
    import navis

    if circuit.body_ids is None:
        raise ValueError(
            f"Circuit {circuit.name!r} has no body_ids; cannot map model "
            f"indices to SWC skeletons. The build function needs to "
            f"populate body_ids from cx['bodyId']."
        )

    anatomy_dir = circuit.provenance.get("anatomy_dir")
    extra_dirs = list(circuit.provenance.get("anatomy_extra_dirs", []) or [])
    if not anatomy_dir:
        raise ValueError(
            f"Circuit {circuit.name!r} has no provenance['anatomy_dir']."
        )

    # Build a unified bodyId -> (swc_path, type_str) map across all dirs.
    bid_map: "dict[int, tuple[str, str]]" = {}
    for d in [anatomy_dir, *extra_dirs]:
        if not os.path.isdir(d):
            continue
        for bid, val in _build_bodyid_index(d).items():
            bid_map.setdefault(bid, val)

    neurons: list = []
    type_names: list = []
    has_skel = np.zeros(len(circuit.body_ids), dtype=bool)

    # Per-model-index canonical type name (from the circuit's typing).
    canonical_types = [str(t) for t in circuit.type_names]
    for i, bid in enumerate(circuit.body_ids):
        canon_t = canonical_types[int(circuit.neuron_types[i])]
        entry = bid_map.get(int(bid))
        if entry is None:
            neurons.append(None)
            type_names.append(canon_t)
            continue
        swc_path, _ = entry
        if not os.path.isfile(swc_path):
            neurons.append(None)
            type_names.append(canon_t)
            continue
        n = navis.read_swc(swc_path)
        if downsample and downsample > 1:
            n = navis.downsample_neuron(n, downsampling_factor=downsample,
                                         preserve_nodes=None)
        neurons.append(n)
        type_names.append(canon_t)
        has_skel[i] = True

    return neurons, type_names, has_skel


def _extract_segments(neurons):
    """Flatten the list of TreeNeurons into per-neuron (E_i, 2, 3) arrays.
    Returns (seg_arrays, seg_owner, all_segs)."""
    seg_arrays = []
    for n in neurons:
        if n is None:
            seg_arrays.append(np.zeros((0, 2, 3), dtype=np.float32))
            continue
        nodes = n.nodes
        child = nodes[nodes.parent_id != -1]
        if len(child) == 0:
            seg_arrays.append(np.zeros((0, 2, 3), dtype=np.float32))
            continue
        parent_xyz = nodes.set_index("node_id").loc[
            child.parent_id.values, ["x", "y", "z"]
        ].values
        child_xyz = child[["x", "y", "z"]].values
        seg_arrays.append(np.stack([parent_xyz, child_xyz], axis=1)
                          .astype(np.float32))
    counts = np.array([len(s) for s in seg_arrays])
    seg_owner = np.repeat(np.arange(len(neurons)), counts)
    all_segs = (np.concatenate(seg_arrays, axis=0) if seg_arrays
                else np.zeros((0, 2, 3), dtype=np.float32))
    return seg_arrays, seg_owner, all_segs


def _resolve_frame_idx(frame_idx: int, T: int) -> int:
    """Resolve a possibly-negative ``frame_idx`` into [0, T)."""
    if frame_idx < 0:
        frame_idx = T + frame_idx
    return max(0, min(int(frame_idx), T - 1))


def _type_keep_mask(
    type_names_per_neuron: list, whitelist: list, has_skel: np.ndarray,
) -> np.ndarray:
    """Bool (N,) mask: True where the neuron's canonical type is in the
    whitelist AND a skeleton is available. Empty whitelist = all neurons
    with skeletons."""
    n = len(type_names_per_neuron)
    keep_type = np.ones(n, dtype=bool) if not whitelist else np.array(
        [t in whitelist for t in type_names_per_neuron], dtype=bool,
    )
    return keep_type & has_skel


# ---------------------------------------------------------------------------
# Animal-icon silhouettes — minimal ports of the standalone scripts'
# _draw_fish_icon / _draw_fly_icon helpers. Drawn in an inset axes
# (top-right corner of the figure) when
# ``plot_cfg.anatomy_voltage_show_icon`` is True.
# ---------------------------------------------------------------------------

# Larval-zebrafish dorsal silhouette, nose at -x at theta=0 (verbatim
# from figures/zebrafish/fig_zebrafish_anatomy_3d_voltage_anim.py).
_FISH_SILHOUETTE_X = np.array([
    -1.20, -0.95, -0.50, -0.30, -0.10,
     0.30,  0.80,  1.30,  1.40,  1.30,
     0.80,  0.30, -0.10, -0.30, -0.50, -0.95,
], dtype=np.float32)
_FISH_SILHOUETTE_Y = np.array([
     0.00,  0.30,  0.35,  0.55,  0.30,
     0.20,  0.10,  0.40,  0.00, -0.40,
    -0.10, -0.20, -0.30, -0.55, -0.35, -0.30,
], dtype=np.float32)
_FISH_EYES_X = np.array([-0.85, -0.85], dtype=np.float32)
_FISH_EYES_Y = np.array([ 0.20, -0.20], dtype=np.float32)
_FISH_WAIST_X = 0.30
_FISH_TAIL_TIP_X = 1.40


def _fish_tail_swish(frame_t, phase_per_frame=0.18, amp=0.25):
    """Per-vertex y offsets for the fish silhouette so the caudal fin
    sweeps side-to-side over time. Vertices forward of the waist (x <
    waist) are unaffected. Verbatim from the standalone script."""
    if frame_t is None:
        return np.zeros_like(_FISH_SILHOUETTE_Y)
    span = _FISH_TAIL_TIP_X - _FISH_WAIST_X
    t_norm = np.clip((_FISH_SILHOUETTE_X - _FISH_WAIST_X) / span, 0.0, 1.0)
    phase = phase_per_frame * float(frame_t)
    return amp * (t_norm ** 1.5) * float(np.sin(phase))

# Drosophila dorsal silhouette, nose at +x at theta=0 (verbatim from
# figures/drosophila_cx/fig_cx_anatomy_3d_voltage_anim.py).
_FLY_BODY_TOP = np.array([
    (-1.50, 0.00), (-1.25, 0.18), (-1.00, 0.32), (-0.70, 0.46),
    (-0.35, 0.48), (-0.10, 0.42), ( 0.15, 0.52), ( 0.40, 0.58),
    ( 0.65, 0.50), ( 0.80, 0.30), ( 0.95, 0.32), ( 1.15, 0.34),
    ( 1.30, 0.26), ( 1.45, 0.10), ( 1.50, 0.00),
], dtype=np.float32)
_FLY_BODY_X = np.concatenate([_FLY_BODY_TOP[:, 0],
                               _FLY_BODY_TOP[::-1, 0][1:-1]])
_FLY_BODY_Y = np.concatenate([_FLY_BODY_TOP[:, 1],
                               -_FLY_BODY_TOP[::-1, 1][1:-1]])
_FLY_WING_L_X = np.array([0.45, 0.25, -0.10, -0.55, -1.05, -1.45, -1.55,
                          -1.30, -0.80, -0.30, 0.15, 0.40, 0.45],
                          dtype=np.float32)
_FLY_WING_L_Y = np.array([0.50, 0.70,  0.95,  1.20,  1.40,  1.40,  1.15,
                          0.95,  0.75,  0.60, 0.52, 0.50, 0.50],
                          dtype=np.float32)
_FLY_EYES_X = np.array([1.18, 1.18], dtype=np.float32)
_FLY_EYES_Y = np.array([0.26, -0.26], dtype=np.float32)


def _wing_flap_scale(frame_t, period: int = 80, burst_len: int = 22) -> float:
    """Y-scale factor (0.30..1.0) for fly wings so they flutter in bursts.
    Wings rest fully spread (1.0) most of the time; during the first
    ``burst_len`` frames of each ``period``-frame cycle they oscillate
    through four wing beats. Verbatim from the standalone."""
    if frame_t is None:
        return 1.0
    cycle = int(frame_t) % period
    if cycle >= burst_len:
        return 1.0
    phase = (cycle / max(burst_len - 1, 1)) * 2.0 * np.pi * 4.0
    return 0.30 + 0.70 * (0.5 + 0.5 * float(np.cos(phase)))


def _draw_animal_icon(ax, organism: str, theta_rad: float,
                       body_color="white", eye_color=(0.30, 0.30, 0.30),
                       wing_alpha: float = 0.30,
                       frame_t: "int|None" = None) -> None:
    """Draw a small animal silhouette in ``ax``, rotated by ``theta_rad``.

    ``frame_t`` (optional) drives the caudal-fin swish for fish (and could
    drive the wing flap for fly — currently spread-only). Mirrors the
    standalone scripts' ``frame_t`` argument."""
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)

    c, s = float(np.cos(theta_rad)), float(np.sin(theta_rad))

    def _rot(x, y):
        return c * x - s * y, s * x + c * y

    if organism == "zebrafish":
        # Apply tail swish to the body Y coords before rotation.
        sy = _FISH_SILHOUETTE_Y + _fish_tail_swish(frame_t)
        bx, by = _rot(_FISH_SILHOUETTE_X, sy)
        ax.fill(bx, by, color=body_color, edgecolor="none", linewidth=0,
                zorder=2)
        ex, ey = _rot(_FISH_EYES_X, _FISH_EYES_Y)
        ax.plot(ex, ey, linestyle="", marker="o", markersize=2.0,
                color=eye_color, markeredgewidth=0, zorder=3)
    else:
        # drosophila: wings (with periodic flap), then body, then eyes.
        flap = _wing_flap_scale(frame_t)
        wy_left = _FLY_WING_L_Y * flap
        for wy_src in (wy_left, -wy_left):
            wx, wy = _rot(_FLY_WING_L_X, wy_src)
            ax.fill(wx, wy, color=body_color, alpha=wing_alpha,
                    edgecolor="none", linewidth=0, zorder=2)
        bx, by = _rot(_FLY_BODY_X, _FLY_BODY_Y)
        ax.fill(bx, by, color=body_color, edgecolor="none", linewidth=0,
                zorder=3)
        ex, ey = _rot(_FLY_EYES_X, _FLY_EYES_Y)
        ax.plot(ex, ey, linestyle="", marker="o", markersize=2.0,
                color=eye_color, markeredgewidth=0, zorder=4)


# ---------------------------------------------------------------------------
# Two-phase render: heavy one-time scene prep + cheap per-frame paint.
# ---------------------------------------------------------------------------

def _prepare_projection(circuit, plot_cfg, organism_label: str) -> dict:
    """Type-independent one-time work: load every SWC, project to 2D,
    compute the per-segment owner array, and the static camera frame.

    This is the heavy part (navis SWC loads + projection of ~80k
    segments). It does NOT depend on the type whitelist, so a multi-group
    render (one folder per type list) can reuse a single projection and
    only recompute the cheap per-group keep mask via ``_scene_for_types``.
    """
    elev = float(plot_cfg.anatomy_voltage_elev)
    azim = float(plot_cfg.anatomy_voltage_azim)
    downsample = int(plot_cfg.anatomy_voltage_downsample)
    bg = str(plot_cfg.anatomy_voltage_bg).lower()

    neurons, type_names_per, has_skel = _load_skeletons_for_circuit(
        circuit, downsample=downsample,
    )
    _, seg_owner, all_segs = _extract_segments(neurons)

    if all_segs.size == 0:
        return {
            "empty": True, "bg": bg, "circuit": circuit,
            "organism_label": organism_label,
        }

    segs2d_full = _project_2d(
        all_segs.reshape(-1, 3), elev, azim,
    ).reshape(-1, 2, 2)

    # Static camera box (data range). Use the FULL set of segments so
    # that the camera frames the whole anatomy — the base skeleton (which
    # is everyone) sets the limits, not the whitelisted subset.
    xy = segs2d_full.reshape(-1, 2)
    x_lo, y_lo = xy.min(axis=0)
    x_hi, y_hi = xy.max(axis=0)
    pad_x = 0.04 * (x_hi - x_lo + 1e-6)
    pad_y = 0.04 * (y_hi - y_lo + 1e-6)

    return {
        "empty": False,
        "segs2d_full": segs2d_full,
        "seg_owner": seg_owner,
        "type_names_per": type_names_per,
        "has_skel": has_skel,
        "n_total_neurons": int(circuit.N),
        "xlim": (float(x_lo - pad_x), float(x_hi + pad_x)),
        "ylim": (float(y_lo - pad_y), float(y_hi + pad_y)),
        "bg": bg,
        "circuit": circuit,
        "organism_label": organism_label,
    }


def _scene_for_types(proj: dict, types: list) -> dict:
    """Cheap per-group step: given a reusable projection from
    ``_prepare_projection`` and a type whitelist, compute the lit-segment
    subset + a scene dict ready for ``_render_frame_from_scene``."""
    if proj.get("empty"):
        return dict(proj)
    keep = _type_keep_mask(
        proj["type_names_per"], list(types or []), proj["has_skel"],
    )
    seg_owner = proj["seg_owner"]
    segs2d_full = proj["segs2d_full"]
    keep_owner = keep[seg_owner]
    return {
        "empty": False,
        "segs2d_full": segs2d_full,      # all neurons, for the base skeleton
        "segs2d_lit": segs2d_full[keep_owner],   # whitelist, for green overlay
        "seg_owner_kept": seg_owner[keep_owner],
        "n_lit_neurons": int(keep.sum()),
        "n_total_neurons": proj["n_total_neurons"],
        "xlim": proj["xlim"],
        "ylim": proj["ylim"],
        "bg": proj["bg"],
        "circuit": proj["circuit"],
        "organism_label": proj["organism_label"],
    }


def _prepare_scene(circuit, plot_cfg, organism_label: str) -> dict:
    """One-time work: load every SWC, project to 2D, compute per-segment
    owner array, derive the static camera frame. Result is reused across
    all frames in an animation pass; this is the speedup that takes the
    per-frame time from O(seconds) to O(ms).

    Thin wrapper over ``_prepare_projection`` + ``_scene_for_types`` using
    the single whitelist in ``plot_cfg.anatomy_voltage_types`` (preserves
    the original single-group callers)."""
    proj = _prepare_projection(circuit, plot_cfg, organism_label)
    return _scene_for_types(proj, list(plot_cfg.anatomy_voltage_types))


def _render_frame_from_scene(
    scene: dict, z_t: np.ndarray, frame_idx: int, T: int,
    plot_cfg, out_path: str,
    theta_hd_rad: float = 0.0,
    neurons_label: Optional[str] = None,
) -> None:
    """Per-frame paint. Uses the prepared ``scene`` so the only work here
    is alpha lookup + LineCollection + savefig. Mirrors the standalone
    ``_paint_panel`` strategy: single-color base for ALL segments, then
    a green overlay for ONLY the lit segments (filter ``alpha > 0.02``)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    bg = scene["bg"]
    circuit = scene["circuit"]
    organism_label = scene["organism_label"]

    # Figure + manually-placed axes for pixel-fixed output. dpi set
    # on the figure AND passed to savefig so the test pipeline's
    # ``default_style.apply_globally`` (which sets savefig.dpi=200)
    # doesn't sneak in a different resolution.
    fig = plt.figure(figsize=(7.0, 7.0), dpi=150)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.94])
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    if scene["empty"]:
        ax.text(0.5, 0.5, f"no skeletons for {circuit.name!r}",
                ha="center", va="center", color="red",
                transform=ax.transAxes)
        ax.set_axis_off()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    segs2d_full = scene["segs2d_full"]
    segs2d_lit = scene["segs2d_lit"]
    seg_owner_kept = scene["seg_owner_kept"]
    z_lo = float(plot_cfg.anatomy_voltage_z_lo)
    z_hi = float(plot_cfg.anatomy_voltage_z_hi)
    alpha_global = float(plot_cfg.anatomy_voltage_alpha)

    # Optional dark-grey base skeleton (gated on
    # ``plot_cfg.anatomy_voltage_show_base``). Painted from the FULL
    # segment set — not the whitelist — so the green-highlighted
    # sub-population is always seen against the complete anatomy context.
    # The standalone disables this on its dorsal panel
    # (``show_base=False``) so ROI pixel-sampling isn't biased by static
    # ink; the test-pipeline default is ON because there's no ROI
    # sampling here.
    if bool(getattr(plot_cfg, "anatomy_voltage_show_base", True)):
        base_color = (0.45, 0.45, 0.45) if bg == "black" else (0.40, 0.40, 0.40)
        ax.add_collection(LineCollection(
            segs2d_full, colors=[base_color], linewidths=0.35, alpha=0.05,
        ))

    # Green overlay — only segments whose owner neuron passes the type
    # whitelist AND has alpha > 0.02 at this frame.
    denom = max(1e-6, z_hi - z_lo)
    per_neuron_a = np.clip((z_t - z_lo) / denom, 0.0, 1.0) * alpha_global
    seg_alpha = per_neuron_a[seg_owner_kept]
    lit = seg_alpha > 0.02
    if lit.any():
        rgba = np.tile(np.array([0.0, 1.0, 0.3, 1.0], dtype=np.float32),
                       (int(lit.sum()), 1))
        rgba[:, 3] = seg_alpha[lit]
        ax.add_collection(LineCollection(
            segs2d_lit[lit], colors=rgba, linewidths=0.45,
        ))

    # Static camera.
    ax.set_xlim(scene["xlim"])
    ax.set_ylim(scene["ylim"])
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Small top-left text label — frame counter + head direction (wrapped
    # to [-180, 180]) + the neuron-type whitelist (or 'all' when empty).
    # Smaller font so it doesn't dominate the panel.
    import math
    hd_deg = math.degrees(theta_hd_rad)
    hd_deg = ((hd_deg + 180.0) % 360.0) - 180.0
    if neurons_label is not None:
        neurons_str = neurons_label
    else:
        whitelist = list(getattr(plot_cfg, "anatomy_voltage_types", []) or [])
        neurons_str = "all" if not whitelist else ",".join(whitelist)
    txt_color = "white" if bg == "black" else "black"
    ax.text(
        0.02, 0.97,
        f"frame {frame_idx:5d}/{T-1}  HD {hd_deg:+5.0f}°  neurons: {neurons_str}",
        color=txt_color, fontsize=7, family="monospace",
        ha="left", va="top", transform=ax.transAxes,
    )

    # Optional animal icon — inset axes ANCHORED INSIDE the main axes
    # (top-right corner of the data plot), oriented at the current
    # heading. Anchoring to ``ax`` instead of ``fig`` keeps the icon
    # locked to the visible plot region even when matplotlib's axes box
    # shifts. For zebrafish the caudal fin swishes with ``frame_t``;
    # for drosophila the wings flap in periodic bursts. Gated on
    # ``plot_cfg.anatomy_voltage_show_icon``.
    if bool(getattr(plot_cfg, "anatomy_voltage_show_icon", False)):
        organism = ("zebrafish" if organism_label.startswith("zebrafish")
                    else "drosophila")
        # 12% × 12% square in the top-right corner of the main axes
        # (axes-fraction coords). At figsize=(7, 7) × dpi=150 this is
        # ≈ 125 px square, roughly the same on-screen size as the
        # standalone's 1.7" icon on its 10"×14.6" panel.
        icon_ax = ax.inset_axes([0.88, 0.88, 0.12, 0.12])
        _draw_animal_icon(icon_ax, organism, theta_hd_rad,
                          body_color=txt_color,
                          eye_color=(0.30, 0.30, 0.30),
                          frame_t=frame_idx)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Force exact 1050×1050 px output: re-assert figsize + dpi right
    # before savefig (the test pipeline's preloaded figure_style sets
    # savefig.dpi=200 and figure.dpi=100 globally, and some upstream
    # plot helper may have nudged savefig.bbox to "tight"). Pass an
    # explicit Bbox so matplotlib can't auto-trim to content.
    from matplotlib.transforms import Bbox
    fig.set_size_inches(7.0, 7.0)
    fig.set_dpi(150)
    fig.savefig(
        out_path, dpi=150, facecolor=fig.get_facecolor(),
        bbox_inches=Bbox([[0, 0], [7.0, 7.0]]), pad_inches=0,
    )
    plt.close(fig)


def _render_snapshot(
    h_traj: np.ndarray,
    circuit,
    plot_cfg,
    out_path: str,
    organism_label: str,
) -> None:
    """One-shot render: prepares the scene + paints a single frame.
    Used by ``render_zebrafish_anatomy_voltage`` /
    ``render_drosophila_anatomy_voltage`` when called outside the
    multi-frame loop. The multi-frame entry point
    ``run_anatomy_voltage_test`` calls ``_prepare_scene`` once and
    ``_render_frame_from_scene`` per frame for a ~7x speedup."""
    T, N = h_traj.shape
    if N != circuit.N:
        raise ValueError(
            f"h_traj has N={N} but circuit {circuit.name!r} has N={circuit.N}; "
            f"the rollout shape must match the circuit."
        )
    z = _zscore_per_neuron(h_traj)
    frame_idx = _resolve_frame_idx(int(plot_cfg.anatomy_voltage_frame_idx), T)
    scene = _prepare_scene(circuit, plot_cfg, organism_label)
    _render_frame_from_scene(
        scene, z[frame_idx], frame_idx, T, plot_cfg, out_path,
        theta_hd_rad=0.0,
    )


# ---------------------------------------------------------------------------
# Per-organism entry points
# ---------------------------------------------------------------------------

def render_zebrafish_anatomy_voltage(
    h_traj: np.ndarray, circuit, plot_cfg, out_path: str,
) -> None:
    """Render the 3D-anatomy + voltage snapshot for a zebrafish HD circuit.

    Assumes ``circuit`` is one of the registered ``zebrafish_HD_*_v*``
    circuits with ``body_ids`` and ``provenance['anatomy_dir']`` set.
    Per-yaml defaults (dorsal view, z_hi=15) live in
    ``plotting.anatomy_voltage_*``.
    """
    _render_snapshot(h_traj, circuit, plot_cfg, out_path,
                     organism_label="zebrafish")


def render_drosophila_anatomy_voltage(
    h_traj: np.ndarray, circuit, plot_cfg, out_path: str,
) -> None:
    """Render the 3D-anatomy + voltage snapshot for a drosophila CX
    circuit.

    Assumes ``circuit`` is a ``drosophila_cx_*_v*`` Circuit with
    ``body_ids`` + ``provenance['anatomy_dir']`` set (Step-1 work added
    these for zebrafish; the drosophila equivalent is a future commit).
    Per-yaml defaults (elev=-7.6, azim=86.6) live in
    ``plotting.anatomy_voltage_*``.
    """
    _render_snapshot(h_traj, circuit, plot_cfg, out_path,
                     organism_label="drosophila CX")


# ---------------------------------------------------------------------------
# Rollout helpers — extracted from the standalone scripts so the test
# pipeline can run the same deterministic probes (const-omega, swim
# stream, single-direction impulse train, OU velocity) without
# re-implementing the stimulus construction in graph_tester.
# ---------------------------------------------------------------------------

def _const_omega_stimulus(n_steps: int, dt: float, omega_deg: float,
                           theta0_rad: float) -> np.ndarray:
    """Build a (1, T, 3) constant-omega stimulus tensor (numpy)."""
    import math
    T = int(n_steps)
    u = np.zeros((1, T, 3), dtype=np.float32)
    u[0, :, 0] = float(omega_deg)
    u[0, 0, 1] = math.cos(float(theta0_rad))
    u[0, 0, 2] = math.sin(float(theta0_rad))
    return u


def _swim_stochastic_stimulus(n_steps: int, dt: float, seed: int = 0,
                               swim_rate_hz: float = 0.5,
                               swim_duration_s: float = 0.3,
                               phase_impulse_mean_rad: float = 0.785,
                               phase_impulse_std_rad: float = 0.40,
                               backward_phase_mean_rad: float = 3.14,
                               backward_phase_std_rad: float = 0.30,
                               left_fraction: float = 0.40,
                               right_fraction: float = 0.40,
                               forward_fraction: float = 0.15,
                               backward_fraction: float = 0.05) -> np.ndarray:
    """Build a (1, T, 3) stochastic swim-integration stimulus.

    Mirrors ``_build_swim_batch`` in
    ``figures/zebrafish/fig_zebrafish_anatomy_3d_voltage_anim.py``.
    Returns only the stimulus channel since the test pipeline doesn't
    consume the per-event display traces from that script.
    """
    import math
    T = int(n_steps)
    L = max(1, int(round(swim_duration_s / dt)))
    rng = np.random.default_rng(seed)
    p_swim = swim_rate_hz * dt

    cdf = np.cumsum([left_fraction, right_fraction,
                     forward_fraction, backward_fraction])
    onset = rng.uniform(size=T) < p_swim
    u_type = rng.uniform(size=T)
    cat = np.digitize(u_type, cdf[:-1]) + 1  # 1=L, 2=R, 3=F, 4=B

    sigma_log_LR = phase_impulse_std_rad / max(phase_impulse_mean_rad, 1e-6)
    sigma_log_B = backward_phase_std_rad / max(backward_phase_mean_rad, 1e-6)
    mag_LR = rng.lognormal(
        mean=math.log(max(phase_impulse_mean_rad, 1e-6)),
        sigma=sigma_log_LR, size=T).astype(np.float32)
    mag_B = rng.lognormal(
        mean=math.log(max(backward_phase_mean_rad, 1e-6)),
        sigma=sigma_log_B, size=T).astype(np.float32)

    delta_theta = np.zeros(T, dtype=np.float32)
    m_left = (cat == 1) & onset
    m_right = (cat == 2) & onset
    m_back = (cat == 4) & onset
    delta_theta[m_left] = +mag_LR[m_left]
    delta_theta[m_right] = -mag_LR[m_right]
    bw_sign = np.where(rng.uniform(size=T) < 0.5, +1.0, -1.0)
    delta_theta[m_back] = (bw_sign[m_back] * mag_B[m_back]).astype(np.float32)

    omega_rad = np.zeros(T, dtype=np.float32)
    for k in range(L):
        omega_rad[k:] += delta_theta[:T - k] / (L * dt)
    omega_deg = np.rad2deg(omega_rad).astype(np.float32)

    theta0 = float(rng.uniform(0, 2 * math.pi))
    u = np.zeros((1, T, 3), dtype=np.float32)
    u[0, :, 0] = omega_deg
    u[0, 0, 1] = math.cos(theta0)
    u[0, 0, 2] = math.sin(theta0)
    return u


def _single_impulse_stimulus(n_steps: int, dt: float, direction: str,
                              interval_s: float, magnitude_rad: float,
                              t_event_s: float, theta0_rad: float,
                              swim_duration_s: float = 0.3) -> np.ndarray:
    """Periodic single-direction (L or R) swim impulse stimulus.

    Mirrors ``_build_single_impulse_batch`` in
    ``figures/zebrafish/fig_zebrafish_anatomy_3d_voltage_anim.py``.
    Returns the (1, T, 3) stimulus tensor only.
    """
    import math
    T = int(n_steps)
    L = max(1, int(round(swim_duration_s / dt)))
    k0 = min(max(0, int(round(t_event_s / dt))), T - 1)
    sign = +1.0 if direction.upper() == "L" else -1.0
    mag = float(abs(magnitude_rad))
    if interval_s and interval_s > 0:
        step_k = max(1, int(round(interval_s / dt)))
        event_ks = np.arange(k0, T, step_k, dtype=np.int64)
    else:
        event_ks = np.array([k0], dtype=np.int64)

    delta_theta = np.zeros(T, dtype=np.float32)
    delta_theta[event_ks] = sign * mag
    omega_rad = np.zeros(T, dtype=np.float32)
    for k in range(L):
        omega_rad[k:] += delta_theta[:T - k] / (L * dt)
    omega_deg = np.rad2deg(omega_rad).astype(np.float32)

    u = np.zeros((1, T, 3), dtype=np.float32)
    u[0, :, 0] = omega_deg
    u[0, 0, 1] = math.cos(float(theta0_rad))
    u[0, 0, 2] = math.sin(float(theta0_rad))
    return u


def _build_stimulus_from_plot_cfg(plot_cfg, dt: float):
    """Single entry point that picks the right stimulus builder based on
    ``plot_cfg.anatomy_voltage_pattern`` and returns ``(u, label, extra)``:
    a (1, T, 3) stimulus, a caption string, and an optional dict of
    pattern-specific metadata (None for the synthetic patterns; for
    ``zapbench_rotation`` it carries warm_steps / sample_steps / theta_frame /
    n_frames so a caller can subsample the rollout at imaging frames)."""
    pat = str(getattr(plot_cfg, "anatomy_voltage_pattern", "const")).lower()
    n_steps = int(plot_cfg.anatomy_voltage_n_steps)
    extra = None
    if pat == "zapbench_rotation":
        from connectome_gnn.generators.zapbench_stimulus import (
            zapbench_rotation_stimulus)
        drive = zapbench_rotation_stimulus(
            dt,
            warmup_s=float(getattr(plot_cfg, "anatomy_voltage_warmup_s", 10.0)),
            connectome_dir=(getattr(plot_cfg,
                                    "anatomy_voltage_zapbench_connectome", "")
                            or None),
            fishfuncem_data=(getattr(
                plot_cfg, "anatomy_voltage_zapbench_fishfuncem_data", "")
                or None))
        u, label = drive["u"], drive["label"]
        extra = {k: drive[k] for k in
                 ("warm_steps", "sample_steps", "theta_frame", "n_frames")}
        return u, label, extra
    if pat == "const":
        u = _const_omega_stimulus(
            n_steps, dt,
            float(plot_cfg.anatomy_voltage_omega_deg),
            float(plot_cfg.anatomy_voltage_theta0_rad),
        )
        label = f"const ω={plot_cfg.anatomy_voltage_omega_deg:.0f}°/s"
    elif pat == "swim":
        u = _swim_stochastic_stimulus(
            n_steps, dt,
            seed=int(plot_cfg.anatomy_voltage_seed),
            swim_rate_hz=1.0 / max(
                float(plot_cfg.anatomy_voltage_swim_interval_s), 1e-6,
            ),
        )
        label = (f"swim seed={plot_cfg.anatomy_voltage_seed} "
                 f"rate={1.0/max(plot_cfg.anatomy_voltage_swim_interval_s,1e-6):.2f}Hz")
    elif pat in ("swim_left", "swim_right"):
        direction = "L" if pat == "swim_left" else "R"
        u = _single_impulse_stimulus(
            n_steps, dt, direction,
            interval_s=float(plot_cfg.anatomy_voltage_swim_interval_s),
            magnitude_rad=float(plot_cfg.anatomy_voltage_swim_magnitude_rad),
            t_event_s=float(plot_cfg.anatomy_voltage_swim_t_event_s),
            theta0_rad=float(plot_cfg.anatomy_voltage_theta0_rad),
        )
        label = (f"{pat} Δt={plot_cfg.anatomy_voltage_swim_interval_s:g}s "
                 f"|Δθ|={plot_cfg.anatomy_voltage_swim_magnitude_rad:.3f}rad")
    elif pat == "ou":
        # Drosophila Hulse-style natural OU velocity stream. The
        # generator returns a full TaskTrials; we only need the stimulus.
        from connectome_gnn.generators.utils import generate_path_integration_batch
        rng = np.random.default_rng(int(plot_cfg.anatomy_voltage_seed))
        batch = generate_path_integration_batch(
            batch_size=1, n_steps=n_steps, dt=float(dt),
            device="cpu", rng=rng,
        )
        u = batch.stimulus.numpy()
        label = f"OU seed={plot_cfg.anatomy_voltage_seed}"
    else:
        raise ValueError(
            f"unknown anatomy_voltage_pattern={pat!r}; expected one of "
            f"const / swim / swim_left / swim_right / ou / zapbench_rotation"
        )
    return u, label, extra


def run_task_rollout(model, plot_cfg, device):
    """Run the chosen probe rollout on ``model`` and return
    ``(h_traj, y_hat, theta_hd, label, extra)``.

    ``h_traj`` is the (T, N) hidden-state (voltage) trajectory — the series to
    convolve with a GCaMP kernel for calcium. ``y_hat`` is the model's (T, 2)
    readout (cos/sin heading). ``theta_hd`` is integrated directly from the
    stimulus's ω channel (channel 0, deg/s) with θ₀ recovered from channels 1-2
    (cos θ₀·δ_{t=0}, sin θ₀·δ_{t=0}). ``extra`` is the pattern metadata from
    :func:`_build_stimulus_from_plot_cfg` (None except for zapbench_rotation).

    ``model`` must expose a ``dt`` attribute and a ``forward(u) -> (y_hat,
    h_buf)`` signature matching the standalone TaskRNN/TaskGNN classes."""
    import math
    import torch
    dt = float(model.dt)
    u_np, label, extra = _build_stimulus_from_plot_cfg(plot_cfg, dt)
    u = torch.from_numpy(u_np).to(device)
    with torch.no_grad():
        y_hat, h_buf = model(u)
    omega_rad = np.deg2rad(u_np[0, :, 0].astype(np.float64))
    theta0_rad = float(math.atan2(u_np[0, 0, 2], u_np[0, 0, 1]))
    theta_hd = theta0_rad + np.cumsum(omega_rad) * dt
    return (h_buf[0].detach().cpu().numpy(),
            y_hat[0].detach().cpu().numpy(),
            theta_hd.astype(np.float32), label, extra)


def run_probe_rollout(model, plot_cfg, device) -> "tuple[np.ndarray, np.ndarray, str]":
    """Thin wrapper over :func:`run_task_rollout` for the anatomy renderer:
    returns ``(h_traj, theta_hd, label)`` (drops the readout + extras)."""
    h_traj, _y, theta_hd, label, _extra = run_task_rollout(model, plot_cfg, device)
    return h_traj, theta_hd, label


# ---------------------------------------------------------------------------
# Dispatcher used by data_test_path_integration_task
# ---------------------------------------------------------------------------

def render_anatomy_voltage(
    h_traj: np.ndarray,
    circuit,
    plot_cfg,
    out_path: str,
    *,
    organism: Optional[str] = None,
) -> None:
    """Dispatch on ``organism`` (or infer from circuit.name) and call the
    matching per-organism render. ``organism`` is one of ``"zebrafish"``
    or ``"drosophila"``."""
    if organism is None:
        if "zebrafish" in circuit.name.lower():
            organism = "zebrafish"
        elif "drosophila" in circuit.name.lower() or "cx" in circuit.name.lower():
            organism = "drosophila"
        else:
            raise ValueError(
                f"Cannot infer organism from circuit name {circuit.name!r}; "
                f"pass organism='zebrafish' or 'drosophila' explicitly."
            )
    if organism == "zebrafish":
        return render_zebrafish_anatomy_voltage(h_traj, circuit, plot_cfg, out_path)
    if organism == "drosophila":
        return render_drosophila_anatomy_voltage(h_traj, circuit, plot_cfg, out_path)
    raise ValueError(f"Unknown organism: {organism!r}")


# ---------------------------------------------------------------------------
# Output helpers: types -> folder name, content crop, frames -> mp4
# ---------------------------------------------------------------------------

def types_folder_name(types) -> str:
    """Filesystem-safe folder name for a type whitelist:
    ``["IPN12_a", "IPN12_b"] -> "IPN12_a_IPN12_b"``; empty -> ``"all"``."""
    parts = [str(t).strip() for t in (types or []) if str(t).strip()]
    if not parts:
        return "all"
    # Keep underscores: the type names themselves contain them (IPN12_a)
    # and "_" is also the join separator -> ["IPN12_a","IPN12_b"] ->
    # "IPN12_a_IPN12_b". Only non [A-Za-z0-9_.+-] chars are replaced.
    safe = [re.sub(r"[^A-Za-z0-9_.+-]", "-", p) for p in parts]
    return "_".join(safe)


def _ffmpeg_exe() -> str:
    """Path to an ffmpeg binary: prefer the pip-bundled imageio-ffmpeg
    (always present in our env), else the system ``ffmpeg``."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _content_bbox(frame_paths, bg: str = "black",
                  sample: int = 60, pad_frac: float = 0.01):
    """Union bounding box (left, upper, right, lower) of the non-background
    content across a sample of frames, padded and rounded to even side
    lengths (h264 needs even W/H). The dark-grey base skeleton is in every
    frame, so this box is stable across frames and across type groups."""
    from PIL import Image
    paths = list(frame_paths)
    if not paths:
        return None
    black_bg = str(bg).lower() == "black"
    step = max(1, len(paths) // max(1, sample))
    xlo = ylo = 1 << 30
    xhi = yhi = -1
    H = W = 0
    for p in paths[::step]:
        a = np.asarray(Image.open(p).convert("RGB"))
        H, W = a.shape[:2]
        mask = (a.max(2) > 12) if black_bg else (a.min(2) < 243)
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        xlo = min(xlo, int(xs.min())); xhi = max(xhi, int(xs.max()))
        ylo = min(ylo, int(ys.min())); yhi = max(yhi, int(ys.max()))
    if xhi < 0:
        return (0, 0, W, H)
    px = int(pad_frac * (xhi - xlo + 1))
    py = int(pad_frac * (yhi - ylo + 1))
    xlo = max(0, xlo - px); ylo = max(0, ylo - py)
    xhi = min(W - 1, xhi + px); yhi = min(H - 1, yhi + py)
    w = (xhi - xlo + 1) & ~1   # force even
    h = (yhi - ylo + 1) & ~1
    return (xlo, ylo, xlo + w, ylo + h)


def crop_frames_inplace(frame_paths, box) -> None:
    """Crop each PNG to ``box`` (left, upper, right, lower), overwriting it.
    No-op when ``box`` is None."""
    if box is None:
        return
    from PIL import Image
    for p in frame_paths:
        im = Image.open(p)
        im.crop(box).save(p)


def frames_to_mp4(frame_paths, mp4_path: str, fps: int = 20) -> Optional[str]:
    """Encode a list of (equal-size) PNG frames into an H.264 mp4 by piping
    raw RGB into ffmpeg. Returns the mp4 path, or None if no frames."""
    import subprocess
    from PIL import Image
    paths = list(frame_paths)
    if not paths:
        return None
    a0 = np.asarray(Image.open(paths[0]).convert("RGB"))
    H, W = a0.shape[:2]
    os.makedirs(os.path.dirname(mp4_path) or ".", exist_ok=True)
    cmd = [
        _ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(int(fps)), "-i", "pipe:", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-movflags", "+faststart", mp4_path,
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for p in paths:
            a = np.asarray(Image.open(p).convert("RGB"))
            if a.shape[:2] != (H, W):  # safety: re-fit stray sizes
                a = np.asarray(Image.open(p).convert("RGB").resize((W, H)))
            proc.stdin.write(np.ascontiguousarray(a).tobytes())
        proc.stdin.close()
        rc = proc.wait()
    except (OSError, FileNotFoundError) as e:
        # ffmpeg missing / unwritable target: the frames are already on
        # disk, so warn and continue rather than abort the whole render.
        print(f"  [anatomy_voltage] mp4 encode skipped ({e}); frames kept "
              f"under {os.path.dirname(mp4_path)}")
        return None
    return mp4_path if rc == 0 else None


# ---------------------------------------------------------------------------
# Companion "sliding kinograph" movie — neuron x time heatmap with a vertical
# time-cursor that slides in lock-step with the anatomy frames.
# ---------------------------------------------------------------------------

def render_kinograph_movie(
    z: np.ndarray,
    frame_ks: np.ndarray,
    row_mask: np.ndarray,
    theta_hd: np.ndarray,
    out_dir: str,
    mp4_path: str,
    plot_cfg,
    *,
    dt: float,
    fps: int = 20,
    label: str = "",
    neurons_label: str = "all",
    max_rows: int = 400,
) -> "Optional[str]":
    """Render a sliding-cursor kinograph movie of the probe rollout.

    ``z`` is the (T, N) per-neuron z-scored voltage (same array the anatomy
    render paints). ``row_mask`` is a (N,) bool selecting which neurons
    become kinograph rows (the group whitelist). Rows are sorted by
    time-of-peak so a travelling bump reads as a diagonal. One frame is
    written per entry of ``frame_ks`` (the SAME strided indices the anatomy
    movie uses), each showing the full heatmap plus a vertical white cursor
    at the current time and the head-direction trace overlaid — so the kino
    movie and the anatomy movie play frame-for-frame in sync.

    Returns the mp4 path (or last frame path if encoding is unavailable).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, N = z.shape
    rows = np.where(row_mask)[0]
    if rows.size == 0:
        rows = np.arange(N)
    # Keep the panel readable: cap row count by uniform subsampling.
    if rows.size > max_rows:
        rows = rows[np.linspace(0, rows.size - 1, max_rows).astype(np.int64)]
    kino = z[:, rows].T                      # (n_rows, T)
    # Sort rows by time-of-peak so a travelling bump is a clean diagonal.
    order = np.argsort(np.argmax(kino, axis=1))
    kino = kino[order]
    n_rows = kino.shape[0]

    vmax = float(np.percentile(np.abs(kino), 99)) or 1.0
    t_s = np.arange(T) * float(dt)
    # HD trace wrapped to [-180, 180]°, rescaled into the row range so it
    # overlays the heatmap without a second axis.
    hd_deg = np.degrees(np.asarray(theta_hd, dtype=np.float64))
    hd_deg = ((hd_deg + 180.0) % 360.0) - 180.0
    hd_row = (hd_deg + 180.0) / 360.0 * (n_rows - 1)

    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(out_dir):
        if fn.startswith("frame_") and fn.endswith(".png"):
            try: os.remove(os.path.join(out_dir, fn))
            except OSError: pass

    from tqdm import tqdm
    frame_paths: list = []
    extent = [t_s[0], t_s[-1], 0, n_rows]
    pbar = tqdm(list(enumerate(frame_ks)), desc="[kinograph]",
                unit="frame", ncols=150, leave=True)
    for ix, k in pbar:
        k = int(k)
        fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=120)
        ax.imshow(kino, aspect="auto", origin="lower", cmap="inferno",
                  vmin=-vmax, vmax=vmax, extent=extent, interpolation="nearest")
        ax.plot(t_s, hd_row, color="#39ff7a", lw=0.8, alpha=0.7)
        ax.axvline(t_s[k], color="white", lw=1.5)
        ax.set_xlim(t_s[0], t_s[-1])
        ax.set_ylim(0, n_rows)
        ax.set_xlabel("time (s)", fontsize=9)
        ax.set_ylabel(f"neuron (sorted by peak)  [{neurons_label}]", fontsize=9)
        ax.set_title(f"{label}   t={t_s[k]:.2f}s   frame {ix}/{len(frame_ks) - 1}",
                     fontsize=9)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"frame_{ix:04d}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        frame_paths.append(out_path)
        pbar.set_postfix_str(f"frame_t={k}/{T - 1}")
    pbar.close()

    if len(frame_paths) > 1:
        mp4 = frames_to_mp4(frame_paths, mp4_path, fps=fps)
        if mp4:
            print(f"  [kinograph] wrote movie: {mp4}")
            return mp4
    return frame_paths[-1] if frame_paths else None


# ---------------------------------------------------------------------------
# Test-pipeline entry point — one call from data_test_path_integration_task
# ---------------------------------------------------------------------------

def run_anatomy_voltage_test(
    model,
    circuit,
    plot_cfg,
    log_dir: str,
    *,
    device=None,
    organism: Optional[str] = None,
    type_groups: "Optional[list]" = None,
    crop: bool = True,
    make_movie: bool = True,
    fps: Optional[int] = None,
) -> "Optional[str]":
    """Run the probe rollout chosen by ``plot_cfg.anatomy_voltage_pattern``
    on ``model`` and render either a single PNG snapshot
    (``stride <= 0``) or a frame sequence (``stride > 0``). All
    ``anatomy_voltage_*`` knobs come from ``plot_cfg``; the caller (the
    test pipeline) provides ``log_dir`` so we can place the output under
    ``<log_dir>/tmp_recons/``.

    Output layout (one sub-folder + one mp4 per type group)::

        <log_dir>/tmp_recons/<folder>/frame_0000.png ...
        <log_dir>/tmp_recons/<folder>.mp4

    where ``<folder>`` is the type whitelist joined by ``_`` (e.g.
    ``IPN12_a_IPN12_b``) or ``all`` when the whitelist is empty
    (:func:`types_folder_name`). Frames are cropped to the anatomy's
    content box (``crop``) and assembled into an mp4 (``make_movie``).

    ``type_groups`` lets one rollout drive several whitelists: a list
    where each element is itself a list of type names (``[]`` = all). When
    None, the single whitelist in ``plot_cfg.anatomy_voltage_types`` is
    used. The heavy SWC load + 2D projection is shared across groups.

    Returns the last mp4 path (or the last frame path if movies are off),
    and None on early skip / failure.
    """
    if circuit is None or circuit.body_ids is None:
        return None
    if organism is None:
        organism = ("zebrafish" if "zebrafish" in circuit.name.lower()
                    else "drosophila")
    organism_label = "zebrafish" if organism == "zebrafish" else "drosophila CX"

    if type_groups is None:
        type_groups = [list(plot_cfg.anatomy_voltage_types)]
    if fps is None:
        fps = int(getattr(plot_cfg, "anatomy_voltage_fps", 20) or 20)

    # 1. Probe rollout under the chosen pattern (shared across groups).
    h_traj, theta_hd, label = run_probe_rollout(model, plot_cfg, device)
    T, N = h_traj.shape
    if N != circuit.N:
        raise ValueError(
            f"h_traj N={N} != circuit.N={circuit.N}; the rollout shape "
            f"must match the circuit."
        )
    z = _zscore_per_neuron(h_traj)

    stride = int(getattr(plot_cfg, "anatomy_voltage_stride", 0) or 0)
    tmp_recons = os.path.join(log_dir, "tmp_recons")
    os.makedirs(tmp_recons, exist_ok=True)

    if stride <= 0:
        frame_ks = np.array(
            [_resolve_frame_idx(int(plot_cfg.anatomy_voltage_frame_idx), T)],
            dtype=np.int64,
        )
    else:
        frame_ks = np.arange(stride - 1, T, stride, dtype=np.int64)

    # 2. HEAVY one-time, type-independent scene prep (SWCs, projection,
    # segment-owner map, static camera box). Reused across every group so
    # an N-group render pays the ~10-30 s SWC load + projection only once.
    import time as _time
    from tqdm import tqdm
    _t0 = _time.time()
    proj = _prepare_projection(circuit, plot_cfg, organism_label)
    print(f"  [anatomy_voltage:{organism}] projection ready in "
          f"{_time.time() - _t0:.1f}s "
          f"({len(proj.get('segs2d_full', [])):,} segments, "
          f"{len(type_groups)} group(s))")

    last_out: Optional[str] = None
    for grp in type_groups:
        types = list(grp or [])
        folder = types_folder_name(types)
        neurons_label = "all" if not types else ",".join(types)
        out_dir = os.path.join(tmp_recons, folder)
        os.makedirs(out_dir, exist_ok=True)
        # Clean stale frames from a prior run of this group.
        for fn in os.listdir(out_dir):
            if fn.startswith("frame_") and fn.endswith(".png"):
                try: os.remove(os.path.join(out_dir, fn))
                except OSError: pass

        scene = _scene_for_types(proj, types)
        print(f"  [anatomy_voltage:{organism}] group '{folder}': "
              f"whitelist {scene.get('n_lit_neurons', 0)}/{circuit.N} neurons")

        # Companion sliding kinograph, synced to the SAME frame indices so it
        # plays in lock-step with the anatomy movie. Rendered FIRST because
        # it's the cheap part (~seconds) and shouldn't be lost if the slow
        # anatomy frame loop below is interrupted.
        if bool(getattr(plot_cfg, "anatomy_voltage_kinograph", False)):
            row_mask = _type_keep_mask(
                proj["type_names_per"], types, proj["has_skel"],
            )
            kino_out = render_kinograph_movie(
                z, frame_ks, row_mask, theta_hd,
                os.path.join(tmp_recons, f"{folder}_kino"),
                os.path.join(tmp_recons, f"{folder}_kino.mp4"),
                plot_cfg, dt=float(model.dt), fps=fps,
                label=label, neurons_label=neurons_label,
            )
            if kino_out is not None:
                last_out = kino_out

        frame_paths: list = []
        crop_box = None   # fixed from the first frame; see note below
        pbar = tqdm(
            list(enumerate(frame_ks)),
            desc=f"[anatomy_voltage:{organism}] {folder}",
            unit="frame", ncols=150, leave=True,
        )
        for ix, k in pbar:
            out_path = os.path.join(out_dir, f"frame_{ix:04d}.png")
            _render_frame_from_scene(
                scene, z[int(k)], int(k), T, plot_cfg, out_path,
                theta_hd_rad=float(theta_hd[int(k)]),
                neurons_label=neurons_label,
            )
            # 3. Crop away the blank margins as each frame is written, so the
            # folder is tight even mid-run (and the user sees cropped frames
            # while it renders). The dark-grey base skeleton (all neurons,
            # drawn every frame) fixes the content box, so the first frame
            # defines it for the whole group; the green overlay is always a
            # subset of the base and can never fall outside it.
            if crop:
                if crop_box is None:
                    crop_box = _content_bbox([out_path], bg=proj.get("bg", "black"))
                crop_frames_inplace([out_path], crop_box)
            frame_paths.append(out_path)
            pbar.set_postfix_str(f"frame_t={int(k)}/{T-1}")
        pbar.close()

        # 4. Assemble the mp4 next to the folder: tmp_recons/<folder>.mp4.
        last_out = frame_paths[-1] if frame_paths else None
        if make_movie and len(frame_paths) > 1:
            mp4_path = os.path.join(tmp_recons, f"{folder}.mp4")
            mp4 = frames_to_mp4(frame_paths, mp4_path, fps=fps)
            if mp4:
                print(f"  [anatomy_voltage:{organism}] wrote movie: {mp4}")
                last_out = mp4
    return last_out
