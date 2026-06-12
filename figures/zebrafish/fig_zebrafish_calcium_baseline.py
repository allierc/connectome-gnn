"""Figure 12 — recorded vs task-only learned calcium (one script, one figure).

Every panel (a–e) is rendered natively in a single matplotlib figure from
the companion ``.npz`` files dumped by ``fig_functional_panel.py``. There
is no PNG read-back, no compositing of pre-rendered subfigures: panels a,
b, c, d (kinographs + heading strip) and e (power-spectrum overlay) all
share the same font sizes, line widths, and spine style.

    panel a (real,    full block): kinograph + ω + true/decoded HD
    panel b (model,   full block): same layout for the model rollout
    panel c (real,    first 100 s zoom): same as a, x-limit cropped
    panel d (model,   first 100 s zoom): same as b, x-limit cropped
    panel e (full block): per-neuron power spectrum, real vs model
                          (median ± IQR, log–log)

Usage:
    # regenerates the .npz files first, then builds the figure
    python figures/zebrafish/fig_zebrafish_calcium_baseline.py
    # reuse existing .npz (skip the rollout)
    python figures/zebrafish/fig_zebrafish_calcium_baseline.py --skip-generate
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpecFromSubplotSpec

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_PANEL_PY = os.path.join(_HERE, "fig_functional_panel.py")
_DEFAULT_CONFIG = "zebrafish/zebrafish_hd_si_ipn_917_v1_selfmotion_rotation"
_DEFAULT_CIRCUIT = "zebrafish_HD_IPN_917_artr_pt1"
# The recorded-cell matching (functional/bodyid_zapbench_map.csv +
# circuit_functional_traces.npz) is bodyId-keyed, so the IPN12 copy is
# reused for the 917 circuit — build_rows intersects it with the 917
# body_ids (414 of the 481 matched cells survive).
_DEFAULT_CONNECTOME = os.path.join(_HERE, "zebrafish_connectome_HD_IPN12")
_DEFAULT_OUT = os.path.join(_HERE, "fig_zebrafish_calcium_baseline.png")
_PAIRS_T_S = 200.0   # time window (s) for the example-pairs panel (e)

# Shared typography — every panel reads from this so a/b/c/d match e.
FS_LABEL  = 16
FS_TICK   = 13
FS_LEGEND = 13
FS_PANEL  = 16
LW_TRACE  = 1.4
LW_SPEC   = 2.0
GT_COLOR   = "#4daf4a"
PRED_COLOR = "black"


def _compute_sort_by_bodyId(config_name, calcium_mapping_pt,
                              n_steps=1500, omega_deg_per_s=60.0):
    """For each observed bodyId, return (partition_label, preferred_phase).

    Runs a single constant-ω rollout on the trained model (the same
    probe used by Fig. 4 panel e), computes the per-cell preferred
    heading phase φ_i = arg(Σ (h_i - <h_i>) · e^{iθ(t)}), and bridges
    via calcium_mapping.pt (bodyId ↔ model_index). Result: a dict
    bodyId -> (partition_label, φ). Cached so the function is cheap to
    call several times in the same Python process.
    """
    cache_key = (config_name, n_steps, omega_deg_per_s)
    cache = getattr(_compute_sort_by_bodyId, "_cache", {})
    if cache_key in cache:
        return cache[cache_key]
    import torch
    from connectome_gnn.config import NeuralGraphConfig
    from connectome_gnn.models.registry import create_model
    from connectome_gnn.models.bump_attractor_eval import (
        _deterministic_sweep_rollout,
    )
    from connectome_gnn.plot_cx import (
        _preferred_phase, _hd_partition_of,
    )
    from connectome_gnn.utils import (
        config_path, log_path, migrate_state_dict, set_data_root,
    )
    import glob
    set_data_root(os.environ.get(
        "GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"))
    cfg_path = config_path(f"{config_name}.yaml")
    cfg = NeuralGraphConfig.from_yaml(cfg_path)
    cfg.config_file = config_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(cfg.graph_model.signal_model_name,
                          aggr_type=cfg.graph_model.aggr_type,
                          config=cfg, device=device).to(device)
    log_dir = log_path(config_name)
    ck = max(glob.glob(f"{log_dir}/models/best_model_with_*.pt"),
              key=os.path.getmtime)
    sd = torch.load(ck, map_location=device, weights_only=False)
    migrate_state_dict(sd)
    model.load_state_dict(sd["model_state_dict"], strict=False)
    model.eval()
    ro = _deterministic_sweep_rollout(
        model, n_steps=n_steps, omega_deg_per_s=omega_deg_per_s,
        device=device,
    )
    h_traj = np.asarray(ro["h"])                   # (T, N)
    theta = np.asarray(ro["true_theta"])           # (T,)
    phi_per_model_ix = _preferred_phase(h_traj, theta)  # (N,)
    # Cell types per model index, for the partition label.
    nt = np.asarray(model.neuron_types).astype(int)
    type_names = list(model.type_names)
    partition_per_model_ix = [
        _hd_partition_of(type_names[int(t)]) for t in nt
    ]
    cm = torch.load(calcium_mapping_pt, map_location="cpu",
                      weights_only=False)
    obs_body = np.asarray(cm["obs_bodyId"], dtype=np.int64)
    obs_model_ix = np.asarray(cm["model_index"], dtype=np.int64)
    out = {}
    for b, mi in zip(obs_body, obs_model_ix):
        if 0 <= int(mi) < phi_per_model_ix.size:
            out[int(b)] = (
                partition_per_model_ix[int(mi)],
                float(phi_per_model_ix[int(mi)]),
            )
    cache[cache_key] = out
    setattr(_compute_sort_by_bodyId, "_cache", cache)
    print(f"[sort] computed partition+φ from constant-ω rollout on "
          f"{os.path.basename(ck)}; "
          f"{len(out)}/{len(obs_body)} bodyIds resolved")
    return out


def _run_panel(extra_args, out_dir, *, connectome=None, circuit=None):
    cmd = [sys.executable, _PANEL_PY, "--no-title", "--out", out_dir]
    if connectome is not None:
        cmd += ["--connectome", connectome]
    if circuit is not None:
        cmd += ["--circuit", circuit]
    cmd += [*extra_args]
    print("[gen]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=_REPO)


def _wrap_deg(rad):
    """Wrap heading to (-180, 180] degrees, like the original render."""
    d = np.degrees(rad)
    return (d + 180.0) % 360.0 - 180.0


def _kinograph_block(fig, outer_ss, npz, label, *, t_window=None,
                      row_order=None, row_partition=None):
    """Render one (kinograph + ω + HD) block into the given outer subplotspec.

    Reads kino / omega / theta / decoded / t_sec from `npz`. Uses the
    same height ratios as the original `panel.render(show_swim_panels=
    False)` but with the manuscript fonts/spines (no bounding box,
    consistent fontsize across all panels).
    """
    kino    = np.asarray(npz["kino"])
    t       = np.asarray(npz["t_sec"])
    omega   = np.asarray(npz["omega"])
    theta   = np.asarray(npz["theta"])
    decoded = np.asarray(npz["decoded"])
    # Optional row permutation (partition-primary, preferred-phase
    # secondary — same sort key as Figure 4 panel e). The caller
    # computes ``row_order`` and ``row_partition`` once from a
    # constant-ω rollout on the trained model so both the recorded
    # and predicted kinographs share identical rows.
    if row_order is not None and len(row_order) == kino.shape[0]:
        kino = kino[np.asarray(row_order, np.int64)]
        if row_partition is not None:
            row_partition = np.asarray(
                row_partition, dtype=object)[np.asarray(row_order,
                                                         np.int64)]

    x_lo, x_hi_full = float(t[0]), float(t[-1])
    x_hi = x_hi_full if t_window is None else min(x_hi_full,
                                                  x_lo + float(t_window))
    # decimate the line panels (kino keeps full res)
    s = max(1, len(t) // 4000)
    td = t[::s]

    # hspace bumped from 0.10 → 0.40 so the trace panels (ω, HD) sit
    # cleanly below the kinograph instead of overlapping into its
    # bottom row.
    gs = GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer_ss,
        height_ratios=[8.4, 1.0, 1.2], hspace=0.40)

    # ── kinograph ────────────────────────────────────────────────────
    ax_k = fig.add_subplot(gs[0])
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("black")
    finite = kino[np.isfinite(kino)]
    vmin = float(np.percentile(finite, 2.0))  if finite.size else -1.0
    vmax = float(np.percentile(finite, 99.5)) if finite.size else  4.0
    ax_k.imshow(kino, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[x_lo, x_hi_full, kino.shape[0], 0],
                interpolation="nearest")
    ax_k.set_xlim(x_lo, x_hi)
    ax_k.tick_params(axis="x", labelbottom=False, length=3)
    if row_partition is not None:
        # Partition group names at each block centre (Fig. 4 panel e).
        from connectome_gnn.plot_cx import (
            _HD_PARTITION_ORDER, _hd_draw_partition_boundaries,
        )
        key = {k: i for i, k in enumerate(_HD_PARTITION_ORDER)}
        rank = np.array([key.get(p, len(_HD_PARTITION_ORDER))
                         for p in row_partition])
        changes = np.where(np.diff(rank) != 0)[0] + 0.5
        bnds = np.concatenate([[0], changes + 0.5, [rank.size]])
        centres = (bnds[:-1] + bnds[1:]) / 2 - 0.5
        labels = [row_partition[int(round(c))] for c in centres]
        ax_k.set_yticks(centres)
        ax_k.set_yticklabels(labels, fontsize=FS_TICK)
        _hd_draw_partition_boundaries(ax_k, row_partition,
                                       axis="y", color="w",
                                       lw=0.6, alpha=0.7)
    else:
        ax_k.set_yticks([])
        ax_k.set_ylabel("obs. neurons",
                        fontsize=FS_LABEL, labelpad=6)
    for sp in ax_k.spines.values():
        sp.set_visible(False)
    # panel letter — pushed left of the ylabel so it does not overlap
    # the rotated "bump-pool neuron (n=...)" text.
    ax_k.text(-0.07, 1.04, label, transform=ax_k.transAxes,
              ha="right", va="bottom",
              fontsize=FS_PANEL, fontweight="bold")

    # ── ω ───────────────────────────────────────────────────────────
    ax_o = fig.add_subplot(gs[1])
    ax_o.plot(td, omega[::s], color=GT_COLOR, lw=LW_TRACE)
    ax_o.axhline(0, color="0.35", lw=0.4)
    oa = max(np.abs(omega).max(), 1.0)
    ax_o.set_ylim(-oa * 1.15, oa * 1.15)
    ax_o.set_xlim(x_lo, x_hi)
    ax_o.set_ylabel(r"$\omega$ (°/s)", fontsize=FS_LABEL, labelpad=6)
    ax_o.tick_params(axis="y", labelsize=FS_TICK, length=3)
    ax_o.tick_params(axis="x", labelbottom=False, length=3)
    for sp in ax_o.spines.values():
        sp.set_visible(False)

    # ── HD (true vs decoded) ────────────────────────────────────────
    ax_h = fig.add_subplot(gs[2])
    ax_h.plot(td, _wrap_deg(theta[::s]),   color=GT_COLOR,   lw=LW_TRACE,
              label="true")
    ax_h.plot(td, _wrap_deg(decoded[::s]), color=PRED_COLOR, lw=LW_TRACE * 0.85,
              label="decoded")
    ax_h.set_ylim(-180, 180)
    ax_h.set_yticks([-180, 0, 180])
    ax_h.set_xlim(x_lo, x_hi)
    ax_h.set_xlabel("time (s)", fontsize=FS_LABEL)
    ax_h.set_ylabel("HD (°)", fontsize=FS_LABEL, labelpad=6)
    ax_h.tick_params(axis="both", labelsize=FS_TICK, length=3)
    for sp in ax_h.spines.values():
        sp.set_visible(False)
    ax_h.legend(fontsize=FS_LEGEND, frameon=False, loc="upper right",
                ncol=2, handlelength=1.4)


def _spectrum_panel(ax, freqs, p_real, p_modl, n_real, n_modl, label):
    """Median + IQR power-spectrum overlay (panel e)."""
    f = freqs[1:]
    p_r = p_real[:, 1:]
    p_m = p_modl[:, 1:]
    for population, color, lbl, n in [
        (p_r, GT_COLOR,   "recorded", n_real),
        (p_m, PRED_COLOR, "modelled", n_modl),
    ]:
        med = np.median(population, axis=0)
        q25 = np.percentile(population, 25, axis=0)
        q75 = np.percentile(population, 75, axis=0)
        ax.fill_between(f, q25, q75, color=color, alpha=0.18, lw=0)
        ax.plot(f, med, color=color, lw=LW_SPEC,
                label=f"{lbl}  (n={n})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz)", fontsize=FS_LABEL)
    ax.set_ylabel(r"$|X(f)|^{2}$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK, length=3)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="upper right")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-0.07, 1.04, label, transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=FS_PANEL, fontweight="bold")


def _compute_spectrum(real_npz, model_npz):
    """Return (freqs, p_real, p_modl, n_real, n_modl) from the two npz files."""
    from connectome_gnn.metrics import fft_power_spectrum
    real = np.load(real_npz); modl = np.load(model_npz)
    T = min(real["kino"].shape[-1], modl["kino"].shape[-1])

    def _clean(k):
        k = k[:, :T]
        return k[np.isfinite(k).all(axis=1)]

    real_k = _clean(real["kino"])
    modl_k = _clean(modl["kino"])
    dt = float(real["dt_sec"])
    freqs, p_real = fft_power_spectrum(real_k, dt=dt)
    _,    p_modl = fft_power_spectrum(modl_k, dt=dt)
    return freqs, p_real, p_modl, int(real_k.shape[0]), int(modl_k.shape[0])


def _pair_panel(ax, real_npz, model_npz, t_window_s, label):
    """For every modelled bump-pool neuron, find the recorded neuron
    (within the 300-cell anatomy-matched pool) whose power spectrum is
    closest in cosine distance, and overlay the 15 best/median/worst
    pairs on the first `t_window_s` seconds.
    """
    from connectome_gnn.metrics import best_observed_match
    real = np.load(real_npz); modl = np.load(model_npz)
    T = min(real["kino"].shape[-1], modl["kino"].shape[-1])
    real_k = real["kino"][:, :T]
    modl_k = modl["kino"][:, :T]
    real_k = real_k[np.isfinite(real_k).all(axis=1)]
    modl_k = modl_k[np.isfinite(modl_k).all(axis=1)]
    dt = float(real["dt_sec"])
    t_sec = np.asarray(real.get("t_sec"))[:T] if "t_sec" in real.files \
        else np.arange(T) * dt
    t_sec = t_sec[:T]
    n_show = int(np.searchsorted(t_sec, t_window_s, side="right"))
    n_show = max(n_show, 1)

    best_idx, best_score = best_observed_match(
        real_k, modl_k, dt=dt, metric="cosine")
    order = np.argsort(best_score)
    n_each = min(5, modl_k.shape[0] // 3)
    picks = np.concatenate([
        order[:n_each],
        order[len(order)//2 - n_each//2: len(order)//2 + n_each - n_each//2],
        order[-n_each:],
    ])
    spacing = 4.5
    for k, mi in enumerate(picks):
        oi = int(best_idx[mi])
        offset = -k * spacing
        m = modl_k[mi, :n_show]; o = real_k[oi, :n_show]
        ax.plot(t_sec[:n_show], m + offset, color="black", lw=0.9)
        ax.plot(t_sec[:n_show], o + offset, color="#4daf4a", lw=0.9)
    # group labels on the right edge
    for j, name in enumerate(("best", "median", "worst")):
        ymid = -(j * n_each + (n_each - 1) / 2.0) * spacing
        ax.text(t_sec[n_show - 1] * 1.01, ymid, name,
                ha="left", va="center", fontsize=FS_TICK, color="0.4")
    ax.set_yticks([])
    ax.set_xlim(float(t_sec[0]), float(t_sec[n_show - 1]))
    ax.set_xlabel("time (s)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK, length=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-0.04, 1.02, label, transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=FS_PANEL, fontweight="bold")


def _zc(M):
    """Per-cell (row) z-score over time."""
    M = np.asarray(M, np.float64)
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    return (M - mu) / np.where(sd > 1e-6, sd, 1.0)


def _afferent_panels(real_npz, model_full_p):
    """Build the three afferent-class kinographs (recorded + model), each
    sorted by per-cell preferred-heading angle φ_i = arg(Σ ΔF/F_i·e^{iθ})
    computed from the recorded ΔF/F (same sort Fig. 15 uses for the bump
    pool), with the SAME row order applied to the matched model rows.

    Returns ``[(name, real_kino_sorted, model_kino_sorted), ...]`` or [] if
    the companions lack the afferent arrays.
    """
    if "aff_names" not in getattr(real_npz, "files", []):
        return []
    if not os.path.isfile(model_full_p):
        print(f"[aff] model full-calcium dump missing: {model_full_p}")
        return []
    from connectome_gnn.plot_cx import _preferred_phase
    full = np.load(model_full_p, allow_pickle=True)
    cal = np.asarray(full["calcium"])                       # (N, T)
    bids = np.asarray(full["body_ids"], np.int64)
    row_of = {int(b): i for i, b in enumerate(bids)}
    th_rad = np.deg2rad(np.asarray(real_npz["theta"], np.float64))
    out = []
    for nm in [str(x) for x in real_npz["aff_names"]]:
        r = np.asarray(real_npz[f"aff_{nm}"], np.float64)   # (n, T) recorded
        cb = np.asarray(real_npz[f"affbid_{nm}"], np.int64)
        T = min(r.shape[1], cal.shape[1])
        m_rows, keep = [], []
        for k, b in enumerate(cb):
            ri = row_of.get(int(b))
            if ri is not None:
                m_rows.append(cal[ri, :T]); keep.append(k)
        if not m_rows:
            continue
        keep = np.asarray(keep, int)
        rz = _zc(r[keep, :T])
        mz = _zc(np.asarray(m_rows, np.float64))
        phi = _preferred_phase(rz.T, th_rad[:T])
        order = np.argsort(phi, kind="stable")
        out.append((nm, rz[order], mz[order]))
    return out


def _kino_only(fig, ss, kino, label, ylabel):
    """A single kinograph strip (no ω / HD traces), name-labelled."""
    ax = fig.add_subplot(ss)
    finite = kino[np.isfinite(kino)]
    vmin = float(np.percentile(finite, 2.0))  if finite.size else -1.0
    vmax = float(np.percentile(finite, 99.5)) if finite.size else  4.0
    ax.imshow(kino, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
              extent=[0, kino.shape[1], kino.shape[0], 0],
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(ylabel, fontsize=FS_TICK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-0.07, 1.04, label, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=FS_PANEL, fontweight="bold")


def _render_like_fig16(real_npz, model_npz, model_full_p, out_path,
                        classes=("ARTR",)):
    """Render Figure 13 in the Figure-16 reconstruction layout.

    Stacks, top→bottom: the recorded ω drive; for the bump-pool and each
    requested afferent ``classes`` a (recorded ΔF/F, model continuous)
    kinograph pair with per-panel z-MSE / SSIM; and the true-vs-decoded
    heading at the bottom. Same renderer as Fig. 16
    (``plot_calcium_reconstruction``); the task-only 917 model is driven by
    the recorded ZAPBench Rotations stimulus over the full block. Default
    ``classes=("ARTR",)`` drops pt-IPN1 and motor_efferent.
    """
    from connectome_gnn.plot_cx import plot_calcium_reconstruction

    dt = float(real_npz["dt_sec"])
    real_pool = np.asarray(real_npz["kino"], np.float64)     # (K, T)
    model_pool = np.asarray(model_npz["kino"], np.float64)   # (K, T)
    T = min(real_pool.shape[1], model_pool.shape[1])
    groups = [{
        "name": "bump-pool",
        "real": real_pool[:, :T].T,                          # (T, K)
        "stitch": None,
        "continuous": model_pool[:, :T].T,
    }]
    # Afferent classes (recorded + matched model rows, preferred-heading
    # sorted), filtered to `classes`.
    for nm, rz, mz in _afferent_panels(real_npz, model_full_p):
        if nm not in classes:
            continue
        Ta = min(rz.shape[1], mz.shape[1], T)
        groups.append({
            "name": nm,
            "real": rz[:, :Ta].T,
            "stitch": None,
            "continuous": mz[:, :Ta].T,
        })

    omega = np.asarray(model_npz["omega"], np.float64)
    theta_deg = np.degrees(np.asarray(model_npz["theta"], np.float64))
    dec_deg = np.degrees(np.asarray(model_npz["decoded"], np.float64))
    hd = {"continuous": {"true": theta_deg, "pred": dec_deg}}

    plot_calcium_reconstruction(
        groups, dt, out_path, omega=omega, hd=hd,
        show_stitch=False, trial_s=None)
    names = ", ".join(g["name"] for g in groups)
    print(f"[fig] wrote {out_path} (Fig-16 layout; groups: {names})")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=_DEFAULT_CONFIG,
                   help="model config for panels b/d (default: task-only baseline)")
    p.add_argument("--circuit", default=_DEFAULT_CIRCUIT,
                   help="circuit registry name for recorded-cell row matching "
                        "(must match the model's circuit)")
    p.add_argument("--connectome", default=_DEFAULT_CONNECTOME,
                   help="connectome dir holding functional/ (bodyId↔zapbench "
                        "map + recorded traces); reused across circuits")
    p.add_argument("--gcamp", default="gcamp7f")
    p.add_argument("--out", default=_DEFAULT_OUT, help="output PNG path")
    p.add_argument("--like-fig16", action="store_true",
                   help="render in the Fig-16 reconstruction layout "
                        "(bump-pool + ARTR only) instead of the a–i baseline")
    p.add_argument("--panel-dir", default=None,
                   help="dir for the kinograph .npz companions "
                        "(default: alongside --out)")
    p.add_argument("--skip-generate", action="store_true",
                   help="reuse already-generated .npz files in --panel-dir")
    p.add_argument("--zoom-s", type=float, default=100.0,
                   help="seconds for the right-hand zoom column "
                        "(first T s); pass <=0 to drop the zoom column")
    args = p.parse_args()

    panel_dir = args.panel_dir or os.path.dirname(os.path.abspath(args.out))
    os.makedirs(panel_dir, exist_ok=True)
    stem = args.config.rsplit("/", 1)[-1]
    zoom = args.zoom_s if args.zoom_s and args.zoom_s > 0 else None
    real_npz_p = os.path.join(panel_dir, "functional_panel_real_rotation.npz")
    model_npz_p = os.path.join(
        panel_dir, f"functional_panel_{stem}_{args.gcamp}.npz")

    if not args.skip_generate:
        # The kinograph PNGs are no longer consumed by this script, but
        # fig_functional_panel.py still writes the companion .npz that
        # we read below. Drive it once for real, once for model. Both use
        # the SAME circuit so panels a/b share the matched row order.
        _run_panel(["--sequence", "rotation"], panel_dir,
                   connectome=args.connectome, circuit=args.circuit)
        _run_panel(["--config", args.config, "--gcamp", args.gcamp], panel_dir,
                   connectome=args.connectome, circuit=args.circuit)

    for path in (real_npz_p, model_npz_p):
        if not os.path.isfile(path):
            sys.exit(f"missing companion: {path}\n"
                     f"  rerun without --skip-generate to regenerate it")

    real_npz = np.load(real_npz_p)
    model_npz = np.load(model_npz_p)
    model_full_p = model_npz_p.replace(".npz", "_full.npz")

    # Fig-16 reconstruction layout (bump-pool + ARTR only): reuse the same
    # renderer as Figure 16, then stop.
    if args.like_fig16:
        _render_like_fig16(real_npz, model_npz, model_full_p, args.out)
        return

    # ── afferent kinographs (ARTR / pt-IPN1 / motor_efferent) ────────
    # Recorded + matched-model rows per class, each sorted by preferred-
    # heading angle (Fig. 15 sort). Built from the recorded afferent
    # arrays in the real npz and the full per-cell model calcium dump.
    aff = _afferent_panels(real_npz, model_full_p)

    # ── outer layout ────────────────────────────────────────────────
    # a (real ring, full block) | b (model ring) | then per afferent
    # class a recorded + a model strip | c (power-spectrum overlay).
    # The bump-pool kinographs stay in the official ZAPBench rastermap
    # row order (the partition + preferred-phase sort wiring is kept in
    # _kinograph_block + _compute_sort_by_bodyId for future use).
    ratios = [1.0, 1.0] + [0.46] * (2 * len(aff)) + [0.42]
    fig = plt.figure(figsize=(20, 4.6 * sum(ratios)))
    outer = fig.add_gridspec(
        len(ratios), 1, height_ratios=ratios, hspace=0.32,
        left=0.085, right=0.985, top=0.985, bottom=0.035,
    )
    _kinograph_block(fig, outer[0, 0], real_npz,  "a")
    _kinograph_block(fig, outer[1, 0], model_npz, "b")
    _letters = "cdefghijkl"
    _ri = 2
    for _k, (_nm, _r, _m) in enumerate(aff):
        _kino_only(fig, outer[_ri, 0], _r, _letters[2 * _k],
                   f"{_nm}\nreal ΔF/F (n={_r.shape[0]})")
        _kino_only(fig, outer[_ri + 1, 0], _m, _letters[2 * _k + 1],
                   f"{_nm}\nmodel")
        _ri += 2

    freqs, p_real, p_modl, n_real, n_modl = _compute_spectrum(
        real_npz_p, model_npz_p)
    ax_e = fig.add_subplot(outer[_ri, 0])
    _spectrum_panel(ax_e, freqs, p_real, p_modl, n_real, n_modl,
                    _letters[2 * len(aff)])

    from _despine import open_axes
    open_axes(fig)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
