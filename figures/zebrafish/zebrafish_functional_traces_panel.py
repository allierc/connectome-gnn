"""Side-by-side functional-trace panels: REAL zapbench ΔF/F vs RNN-model
calcium, for the zebrafish HD/IPN12 circuit, styled like
figures/zebrafish/fig_zebrafish_anatomy_3d_voltage_anim.py (black bg,
green/white HD traces, blue/red L/R + grey/orange F/B swim ticks).

Two PNGs, both laid out identically so they read as a pair:
  * <out>/functional_panel_real.png   — recorded ΔF/F during omr_turning
  * <out>/functional_panel_model.png  — v1 voltage -> gcamp7f kernel -> calcium
                                         during a swim_integration rollout

Kinograph rows = the circuit's bump-pool neurons (dIPN ring + IPN12),
ordered by hemisphere then ring bin, z-scored per neuron. The SAME row set
is used for both panels, so they align by neuron identity; in the REAL panel
rows whose circuit neuron has no functional match are left BLANK (black),
which also visualises coverage.

HD panel: true heading (green) vs predicted (white). Model prediction is the
trained readout (arctan2 of y_hat); the real "prediction" is a population-
vector decode of the matched dIPN ring (connectome ring-bin order — labelled
exploratory), and the real "true" heading is the omr stimulus turn direction
integrated to an implied heading (nominal rate).

Run (env with fishfuncem + torch + the trained model):
  /workspace/.conda_envs/neural-graph-linux/bin/python \
      figures/zebrafish/zebrafish_functional_traces_panel.py
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "papers", "fishFuncEM"))

GREEN = (0.0, 0.85, 0.4)
LR_RED, LR_BLUE = "#ff4444", "#4488ff"      # right=red(+), left=blue(-)
FB_GREY, FB_ORANGE = "#888888", "#ff9922"   # forward=grey(+), backward=orange(-)
N_RING_BINS = 16
OMR_TASK = 4
OMR_LABELS = {-2: "leftward", -1: "backward", 0: "no_movement",
              3: "forward", 4: "rightward"}
GCAMP7F = dict(tau_rise=0.150, tau_decay=1.200, length_seconds=7.2)

# official zapbench rastermap permutation (same cache the whole-brain
# kinograph uses) — lets the real panel order its rows exactly like
# figures/zebrafish/zapbench_full_kinograph.py instead of by hemisphere/ring bin.
_GD = "/groups/saalfeld/home/allierc/GraphData/graphs_data/zebrafish"
SORT_NPY = os.path.join(_GD, "zapbench_rastermap_sorting.npy")
SORT_URL = ("https://storage.googleapis.com/zapbench-release/volumes/20240930/"
            "traces_rastermap_sorted/sorting.json")


def _load_rastermap_sorting():
    """perm[i] = original neuron at sorted row i (cached)."""
    if os.path.isfile(SORT_NPY):
        return np.load(SORT_NPY)
    import json
    import urllib.request
    perm = np.asarray(json.loads(urllib.request.urlopen(SORT_URL).read()),
                      dtype=np.int64)
    np.save(SORT_NPY, perm)
    return perm


def sort_rows_rastermap(rows):
    """Reorder rows by the official zapbench rastermap permutation, so the panel
    kinograph matches the whole-brain kinograph (figures/zebrafish/zapbench_full_kinograph
    .py) and the hemisphere (L/R) grouping is dropped. Unmatched rows (no
    zapbenchId) sort to the bottom, keeping their blanks contiguous."""
    perm = _load_rastermap_sorting()
    inv = np.empty(perm.size, dtype=np.int64)
    inv[perm] = np.arange(perm.size)
    zid = rows["zapbenchId"].to_numpy()
    pos = np.where(zid >= 0, inv[np.clip(zid, 0, perm.size - 1)],
                   perm.size + np.arange(len(rows)))
    return (rows.assign(_rm=pos).sort_values("_rm")
            .drop(columns="_rm").reset_index(drop=True))


# --------------------------------------------------------------------------- #
#  small shared helpers
# --------------------------------------------------------------------------- #
def _group_of(t):
    t = str(t)
    if t.startswith("IPN12"):
        return "IPN12"
    if t.startswith("IPNd") or t.startswith("IPNds"):
        return "dIPN_ring"
    if t.startswith("RIPN") or t.startswith("pt-IPN"):
        return "afferent"
    return "other"


def _wrap_hd_deg(a_rad):
    w = (((np.rad2deg(a_rad) + 180.0) % 360.0) - 180.0).astype(np.float32)
    if w.size > 1:
        w[1:][np.abs(np.diff(w)) > 180.0] = np.nan
    return w


def _zscore_rows(M):
    mu = np.nanmean(M, axis=1, keepdims=True)
    sd = np.nanstd(M, axis=1, keepdims=True)
    return (M - mu) / np.where(sd > 1e-6, sd, 1.0)


def _zscore_global(M):
    """Single global z-score over all finite entries (one scale for the whole
    kinograph), so neurons keep their relative amplitudes. Blanks stay NaN."""
    mu = np.nanmean(M)
    sd = np.nanstd(M)
    return (M - mu) / (sd if sd > 1e-6 else 1.0)


def _gcamp_model(tau_rise=None, tau_decay=None, length_s=None):
    """The shared voltage->calcium model (connectome_gnn.models.gcamp), built
    with the panel's kernel time constants. Single source of truth for the
    convolution; defaults reproduce the GCaMP7f figures."""
    from connectome_gnn.models.gcamp import create_gcamp
    return create_gcamp(
        "double_exp",
        tau_rise=GCAMP7F["tau_rise"] if tau_rise is None else tau_rise,
        tau_decay=GCAMP7F["tau_decay"] if tau_decay is None else tau_decay,
        length_s=length_s)


def gcamp7f_kernel(dt, tau_rise=None, tau_decay=None, length_s=None):
    return _gcamp_model(tau_rise, tau_decay, length_s).kernel(dt).cpu().numpy()


def apply_gcamp(V, dt, tau_rise=None, tau_decay=None, length_s=None):
    """Causal convolution of (T, N) voltage with the GCaMP kernel."""
    g = _gcamp_model(tau_rise, tau_decay, length_s)
    return g(np.asarray(V, dtype=np.float32), dt_in=dt)


# --------------------------------------------------------------------------- #
#  row set: circuit bump-pool neurons (shared by both panels)
# --------------------------------------------------------------------------- #
def build_rows(connectome_dir, circuit_name):
    from connectome_gnn.generators.circuits import get_circuit
    c = get_circuit(circuit_name)
    body_ids = np.asarray(c.body_ids, dtype=np.int64)
    idx_of = {int(b): i for i, b in enumerate(body_ids)}
    n_bump = int(len(c.subpops["bump"]))
    ring = np.asarray(c.bump_ring_ix, dtype=np.int64)

    m = pd.read_csv(os.path.join(connectome_dir, "functional",
                                 "bodyid_zapbench_map.csv"))
    m["group"] = m["type"].map(_group_of)
    bump = m[m["group"].isin(["dIPN_ring", "IPN12"])].copy()

    rows = []
    for _, r in bump.iterrows():
        b = int(r["bodyId"])
        mi = idx_of.get(b)
        if mi is None or mi >= n_bump:
            continue
        rows.append(dict(bodyId=b, type=str(r["type"]), group=r["group"],
                         side=str(r["side"]), ring_bin=int(ring[mi]),
                         model_index=mi,
                         matched=bool(r["matched"]),
                         zapbenchId=(int(r["zapbenchId"])
                                     if not pd.isna(r["zapbenchId"]) else -1)))
    rows = pd.DataFrame(rows)
    side_rank = {"left": 0, "right": 1}
    rows["side_rank"] = rows["side"].map(lambda s: side_rank.get(s, 2))
    rows = rows.sort_values(["side_rank", "ring_bin", "type"]).reset_index(drop=True)
    return rows, c


def ring_pva(activity, ring_bins):
    """Population-vector heading (rad) per frame from (T, n) activity and
    per-neuron ring bin. Weights = ReLU of the per-neuron z-scored trace."""
    ang = 2 * np.pi * np.asarray(ring_bins) / N_RING_BINS
    w = np.clip(activity, 0, None)
    cx = (w * np.cos(ang)[None, :]).sum(1)
    sx = (w * np.sin(ang)[None, :]).sum(1)
    return np.arctan2(sx, cx)


def mlp_decode_hd(X, theta, train_frac=0.7, seed=0):
    """Regularised circular HD decoder: StandardScaler -> PCA -> RidgeCV
    mapping per-frame neural activity X (T, n) -> (cos θ, sin θ), fit on the
    first `train_frac` of frames and run over all frames. Returns
    (decoded_theta_rad, n_train, test_circ_r).

    A flexible MLP on ~300 features with <1k training frames just memorises
    the train block (train r→1, test r→0). Reducing to a few PCA components
    + ridge generalises (train r ≈ test r); the honest test r is modest,
    which itself reflects that the 45°/s rotation is undersampled at the
    1.09 Hz imaging rate.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    X = np.nan_to_num(np.asarray(X, dtype=np.float32))
    T = X.shape[0]
    n_tr = max(2, int(T * train_frac))
    y = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    k = int(min(10, X.shape[1], n_tr - 1))
    model = make_pipeline(StandardScaler(), PCA(n_components=k),
                          RidgeCV(alphas=np.logspace(-1, 4, 20)))
    model.fit(X[:n_tr], y[:n_tr])
    pred = model.predict(X)
    theta_hat = np.arctan2(pred[:, 1], pred[:, 0])

    def _ccorr(a, b):
        a = a - np.angle(np.mean(np.exp(1j * a)))
        b = b - np.angle(np.mean(np.exp(1j * b)))
        return float(np.sum(np.sin(a) * np.sin(b)) /
                     np.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2) + 1e-9))
    test_r = _ccorr(theta_hat[n_tr:], theta[n_tr:]) if T - n_tr > 5 else float("nan")
    return theta_hat, n_tr, test_r


# --------------------------------------------------------------------------- #
#  omr_turning stimulus, shared by REAL and MODEL panels
# --------------------------------------------------------------------------- #
def omr_block_stim(fishfuncem_data, trace_len=None):
    """Return (stim_per_frame, n_frames) for the omr_turning block."""
    from fishfuncem import FishFunctional
    ff = FishFunctional.from_data_dir(fishfuncem_data)
    on = int(np.asarray(ff.onsets_frames)[OMR_TASK])
    off = int(np.asarray(ff.offsets_frames)[OMR_TASK])
    if trace_len is not None:
        off = min(off, trace_len)
    ts = np.arange(on + 1, off)
    stim = np.rint(np.asarray(ff.stim_info_frames)[ts]).astype(int)
    return stim, ts


def stim_to_drive(stim, dt, nominal_omega):
    """Map omr per-frame labels -> (omega, turn_lr, swim_fb) on a `dt` grid.

    leftward (-2)  -> +ω (CCW, blue L tick);  rightward (4) -> -ω (red R tick).
    forward (3) grey, backward (-1) orange F/B ticks; neither rotates heading.
    Returns arrays on the model/real grid (one entry per timestep), the true
    heading, and the time axis. `dt` is the grid step; for the real panel
    pass 0.915, for the model pass model.dt (the labels are then repeated to
    fill each 0.915 s imaging frame).
    """
    n = len(stim)
    omega_f = np.zeros(n, np.float32)
    omega_f[stim == -2] = +nominal_omega
    omega_f[stim == 4] = -nominal_omega
    lr_f = np.zeros(n, np.float32)
    lr_f[stim == -2] = -nominal_omega          # left -> blue (negative)
    lr_f[stim == 4] = +nominal_omega           # right -> red (positive)
    fb_f = np.zeros(n, np.float32)
    fb_f[stim == 3] = +1.0; fb_f[stim == -1] = -1.0

    rep = max(1, int(round(0.915 / dt)))       # imaging-frame -> grid steps
    omega = np.repeat(omega_f, rep)
    turn_lr = np.repeat(lr_f, rep)
    swim_fb = np.repeat(fb_f, rep)
    theta = np.cumsum(np.deg2rad(omega)) * dt
    t_sec = np.arange(len(omega)) * dt
    return dict(omega=omega, turn_lr=turn_lr, swim_fb=swim_fb,
                theta=theta, t_sec=t_sec, rep=rep)


# --------------------------------------------------------------------------- #
#  Rotations task — real heading from stimParam3 (the 8. Rotations sequence).
#  Moved to the package so the same probe is a first-class anatomy_voltage
#  pattern (zapbench_rotation); re-imported here so this script is unchanged.
# --------------------------------------------------------------------------- #
from connectome_gnn.generators.zapbench_stimulus import (  # noqa: E402
    ROT_TASK, rotation_headings, heading_to_drive)


# --------------------------------------------------------------------------- #
#  MODEL panel data — driven by the SAME omr stimulus as the real panel
# --------------------------------------------------------------------------- #
def model_panel(rows, log_dir, drive, device, model_circuit=None, warmup_s=10.0,
                config_path=None, sample_steps=None, display=None,
                gcamp=None):
    import torch
    from connectome_gnn.config import NeuralGraphConfig
    from connectome_gnn.models.registry import create_model
    from connectome_gnn.generators.circuits import get_circuit
    from connectome_gnn.utils import migrate_state_dict, set_data_root
    set_data_root(os.environ.get("GNN_OUTPUT_ROOT",
                                 "/groups/saalfeld/home/allierc/GraphData"))

    cfg_path = config_path or os.path.join(log_dir, "config.yaml")
    cfg = NeuralGraphConfig.from_yaml(cfg_path)
    model = create_model(cfg.graph_model.signal_model_name,
                         aggr_type=cfg.graph_model.aggr_type,
                         config=cfg, device=device).to(device)
    ck = max(glob.glob(f"{log_dir}/models/best_model_with_*.pt"),
             key=os.path.getmtime)
    sd = torch.load(ck, map_location=device, weights_only=False)
    migrate_state_dict(sd)
    model.load_state_dict(sd["model_state_dict"], strict=False)
    model.eval()
    dt = float(model.dt)

    # Resolve which circuit gives this model's neuron order (and bodyIds):
    # the config's named circuit if set, else the caller-supplied fallback
    # (the 731-cell dipn/gnn_dipn configs use the legacy loader == HD_731_v1).
    cc = getattr(getattr(cfg, "circuit", None), "name", None) or model_circuit
    if cc is None:
        raise ValueError("no circuit.name in config; pass --model-circuit")
    bids = np.asarray(get_circuit(cc).body_ids, dtype=np.int64)
    idx_of = {int(b): i for i, b in enumerate(bids)}
    print(f"[model] {os.path.basename(log_dir)}: {os.path.basename(ck)}  dt={dt}  "
          f"N={model.n_units}  circuit={cc}  driving omr ({len(drive['omega'])} steps)")

    omega = drive["omega"]
    T = len(omega)
    # Prepend a zero-ω warmup so the network (and the GCaMP convolution)
    # settle from the zero initial state; we discard it afterwards so the
    # displayed model starts already-settled and stays time-aligned with the
    # real panel (no startup transient dominating the global z-score).
    warm = max(0, int(round(warmup_s / dt)))
    omega_full = np.concatenate([np.zeros(warm, np.float32), omega])
    Tf = len(omega_full)
    u = np.zeros((1, Tf, 3), np.float32)
    u[0, :, 0] = omega_full; u[0, 0, 1] = 1.0  # theta0 = 0
    with torch.no_grad():
        y_hat, h = model(torch.from_numpy(u).to(device))
    h_full = h[0].cpu().numpy()                 # (warm+T, N) voltage
    decoded = np.arctan2(y_hat[0, warm:, 1].cpu().numpy(),
                         y_hat[0, warm:, 0].cpu().numpy())
    g = gcamp or {}
    calcium = apply_gcamp(h_full, dt, g.get("tau_rise"), g.get("tau_decay"),
                          g.get("length_s"))[warm:]   # convolve full, skip warmup
    K = gcamp7f_kernel(dt, g.get("tau_rise"), g.get("tau_decay"), g.get("length_s"))
    print(f"[model] warmup-skipped first {warmup_s}s ({warm} steps); GCaMP "
          f"tau_rise={g.get('tau_rise', GCAMP7F['tau_rise'])}s "
          f"tau_decay={g.get('tau_decay', GCAMP7F['tau_decay'])}s "
          f"support={len(K) * dt:.1f}s")

    # For the rotation task the stimulus (45°/s) is faster than the 1.09 Hz
    # imaging, so the model is run at the true rate then SAMPLED at imaging
    # frames — exactly as the real ΔF/F is sampled — and the display fields
    # come from `display` (frame grid). Without sample_steps (omr) keep the
    # native model grid.
    if sample_steps is not None:
        fidx = np.clip(np.asarray(sample_steps, dtype=np.int64), 0,
                       calcium.shape[0] - 1)
        calcium = calcium[fidx]
        decoded = decoded[fidx]
        drive = display if display is not None else drive

    # Fill the fixed reference rows by bodyId; rows the model lacks (e.g. the
    # IPN12 cells for the 731-cell models) stay BLANK, exactly like the real
    # panel's unmatched rows.
    kino = np.full((len(rows), calcium.shape[0]), np.nan, dtype=np.float32)
    n_filled = 0
    for ri, b in enumerate(rows["bodyId"].to_numpy()):
        mi = idx_of.get(int(b))
        if mi is not None and mi < calcium.shape[1]:
            kino[ri] = calcium[:, mi]; n_filled += 1
    print(f"[model] filled {n_filled}/{len(rows)} reference rows")
    kino = _zscore_global(kino)
    return dict(kino=kino, t_sec=drive["t_sec"], omega=drive["omega"],
                theta=drive["theta"], decoded=decoded,
                turn_lr=drive["turn_lr"], swim_fb=drive["swim_fb"],
                pred_label="readout decode", omega_label="ω (°/s)")


# --------------------------------------------------------------------------- #
#  REAL panel data
# --------------------------------------------------------------------------- #
def real_panel(rows, connectome_dir, ts, drive, decoder="mlp"):
    z = np.load(os.path.join(connectome_dir, "functional",
                             "circuit_functional_traces.npz"), allow_pickle=True)
    traces = z["traces"]                         # (T_full, n_matched) z-scored
    col_of = {int(b): i for i, b in enumerate(z["bodyId"].astype(np.int64))}

    # kinograph: row set with blanks where unmatched.
    kino = np.full((len(rows), len(ts)), np.nan, dtype=np.float32)
    for ri, r in rows.iterrows():
        if r["matched"] and int(r["bodyId"]) in col_of:
            kino[ri] = traces[ts, col_of[int(r["bodyId"])]]
    kino = _zscore_global(kino)

    # predicted heading from the real bump-pool activity.
    mmask = rows["matched"].to_numpy()
    midx = np.nonzero(mmask)[0]
    split_t = None
    if decoder == "mlp":
        pred, n_tr, test_r = mlp_decode_hd(kino[midx].T, drive["theta"])
        split_t = float(drive["t_sec"][n_tr])
        pred_label = f"PCA+Ridge decode (test r={test_r:.2f}, n={len(midx)})"
    else:
        pred = ring_pva(np.nan_to_num(kino[midx].T),
                        rows["ring_bin"].to_numpy()[midx])
        pred_label = "ring PVA (exploratory)"
    # `drive` is built on the 0.915 s imaging grid (one entry per frame), so
    # omega / theta / ticks are identical (up to sampling) to the model.
    return dict(kino=kino, t_sec=drive["t_sec"], omega=drive["omega"],
                theta=drive["theta"], decoded=pred,
                turn_lr=drive["turn_lr"], swim_fb=drive["swim_fb"],
                pred_label=pred_label, omega_label="ω (nominal)",
                split_t=split_t)


# --------------------------------------------------------------------------- #
#  rendering (shared layout)
# --------------------------------------------------------------------------- #
def render(d, rows, out_png, title, bg="black", cmap_name="black_green",
           show_partition=True, show_swim_panels=True, t_window=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    txt = "white" if bg == "black" else "black"
    dim = "0.35"
    t = d["t_sec"]; x_lo, x_hi_full = float(t[0]), float(t[-1])
    # Optional display crop to the first `t_window` seconds (a zoom). The
    # kinograph extent and the line data are unchanged; only the x-limits of
    # every panel are tightened, so the same arrays render as a zoom-in.
    x_hi = x_hi_full if t_window is None else min(x_hi_full, x_lo + float(t_window))

    fig = plt.figure(figsize=(15, 9), facecolor=bg)
    # drop the L/R + F/B swim-tick panels when show_swim_panels is False
    # (HD becomes the bottom axis with the time label).
    ratios = [4.2, 1.0, 1.2, 0.8, 0.8] if show_swim_panels else [4.2, 1.0, 1.2]
    gs = GridSpec(len(ratios), 1, figure=fig, height_ratios=ratios, hspace=0.12)

    def _style(ax, ylabel, bottom=False):
        ax.set_facecolor(bg)
        ax.set_ylabel(ylabel, color=txt, fontsize=11, labelpad=3)
        ax.tick_params(axis="y", colors=txt, labelsize=9, length=3)
        ax.tick_params(axis="x", colors=txt, labelsize=9, length=3,
                       labelbottom=bottom)
        ax.spines[:].set_visible(False)
        ax.set_xlim(x_lo, x_hi)

    # ── kinograph (z-scored), blanks black ──────────────────────────────
    ax = fig.add_subplot(gs[0])
    if cmap_name == "viridis":
        cmap = matplotlib.colormaps["viridis"].copy()
    else:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "black_green", ["black", "#00331a", "#00e63a", "#b6ffb6"])
    cmap.set_bad("black")
    finite = d["kino"][np.isfinite(d["kino"])]
    vmin = float(np.percentile(finite, 2)) if finite.size else -1.0
    vmax = float(np.percentile(finite, 99.5)) if finite.size else 4.0
    ax.imshow(d["kino"], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
              extent=[x_lo, x_hi_full, len(rows), 0], interpolation="nearest")
    # hemisphere boundary + side bar (only when rows are grouped by hemisphere;
    # the rastermap order mixes L/R, so the partition is meaningless there)
    if show_partition:
        n_left = int((rows["side"] == "left").sum())
        ax.axhline(n_left, color=txt, lw=0.6, alpha=0.5)
        ax.text(x_lo, n_left * 0.5, " L", color=LR_BLUE, fontsize=11,
                va="center", ha="left", fontweight="bold")
        ax.text(x_lo, n_left + (len(rows) - n_left) * 0.5, " R", color=LR_RED,
                fontsize=11, va="center", ha="left", fontweight="bold")
    n_match = int(rows["matched"].sum())
    ax.set_ylabel(f"bump-pool neuron  (n={len(rows)}, {n_match} mapped)",
                  color=txt, fontsize=11)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=txt, labelbottom=False, length=3)
    ax.set_facecolor(bg)
    ax.spines[:].set_visible(False)
    ax.set_xlim(x_lo, x_hi)
    ax.set_title(title, color=txt, fontsize=13, pad=8)

    # decimation stride for the line/tick panels (kino keeps full res).
    s = max(1, len(t) // 4000)
    td = t[::s]

    # ── ω panel ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1])
    ax.plot(td, d["omega"][::s], color=GREEN, lw=1.0)
    ax.axhline(0, color=dim, lw=0.3)
    oa = max(np.abs(d["omega"]).max(), 1.0); ax.set_ylim(-oa * 1.15, oa * 1.15)
    _style(ax, d["omega_label"])

    # ── HD panel (true green, predicted white) ──────────────────────────
    ax = fig.add_subplot(gs[2])
    ax.plot(td, _wrap_hd_deg(d["theta"][::s]), color=GREEN, lw=1.1, label="true")
    ax.plot(td, _wrap_hd_deg(d["decoded"][::s]), color=txt, lw=0.9,
            label=d["pred_label"])
    # train/test split marker for the learned decoder: everything to the
    # right of the line is held-out (the honest part of the fit).
    if d.get("split_t") is not None:
        ax.axvline(d["split_t"], color="0.55", lw=0.8, ls="--")
        ax.text(d["split_t"], 150, " held-out →", color="0.7", fontsize=7,
                ha="left", va="top")
    ax.set_ylim(-180, 180); ax.set_yticks([-180, 0, 180])
    _style(ax, "HD (°)", bottom=not show_swim_panels)
    if not show_swim_panels:
        ax.set_xlabel("time (s)", color=txt, fontsize=11)
    ax.legend(fontsize=7, loc="upper right", facecolor=bg, edgecolor=dim,
              labelcolor=txt, framealpha=0.4)

    if show_swim_panels:
        # ── L/R ticks (sustained omr epochs -> coloured bands) ──────────
        ax = fig.add_subplot(gs[3])
        lr = d["turn_lr"]
        ax.fill_between(t, 0, lr, where=lr > 0, color=LR_RED, step="mid", linewidth=0)
        ax.fill_between(t, 0, lr, where=lr < 0, color=LR_BLUE, step="mid", linewidth=0)
        ax.axhline(0, color=dim, lw=0.3)
        la = max(np.abs(lr).max(), 1.0); ax.set_ylim(-la * 1.2, la * 1.2)
        _style(ax, "L / R")

        # ── F/B ticks ───────────────────────────────────────────────────
        ax = fig.add_subplot(gs[4])
        fb = d["swim_fb"]
        ax.fill_between(t, 0, fb, where=fb > 0, color=FB_GREY, step="mid", linewidth=0)
        ax.fill_between(t, 0, fb, where=fb < 0, color=FB_ORANGE, step="mid", linewidth=0)
        ax.axhline(0, color=dim, lw=0.3); ax.set_ylim(-1.5, 1.5)
        _style(ax, "F / B", bottom=True)
        ax.set_xlabel("time (s)", color=txt, fontsize=11)

    fig.savefig(out_png, dpi=130, facecolor=bg)
    plt.close(fig)
    print(f"[plot] wrote {out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--connectome",
                   default=os.path.join(_REPO_ROOT, "figures", "zebrafish",
                                        "zebrafish_connectome_HD_IPN12"))
    p.add_argument("--circuit", default="zebrafish_HD_IPN12_839_v1")
    p.add_argument("--log-dir",
                   default="/groups/saalfeld/home/allierc/GraphData/log/"
                           "zebrafish/zebrafish_hd_si_ipn12_v1")
    p.add_argument("--fishfuncem-data",
                   default=os.path.join(_REPO_ROOT, "papers", "fishFuncEM", "data"))
    p.add_argument("--nominal-omega", type=float, default=30.0,
                   help="deg/s turn rate assigned to omr leftward/rightward "
                        "epochs (drives both the true-HD trace and the model)")
    p.add_argument("--model-dt", type=float, default=0.01,
                   help="model integration step (must match the trained model)")
    p.add_argument("--warmup-s", type=float, default=10.0,
                   help="model warmup seconds to run then discard (skip the "
                        "startup transient before the displayed window)")
    p.add_argument("--gcamp-tau-rise", type=float, default=GCAMP7F["tau_rise"],
                   help="GCaMP kernel rise time constant (s)")
    p.add_argument("--gcamp-tau-decay", type=float, default=GCAMP7F["tau_decay"],
                   help="GCaMP kernel decay time constant (s)")
    p.add_argument("--gcamp-length", type=float, default=None,
                   help="GCaMP kernel support (s); default max(7.2, 6*tau_decay)")
    p.add_argument("--model-circuit", default=None,
                   help="circuit name giving the model's neuron order/bodyIds "
                        "when the config has no circuit.name (e.g. "
                        "zebrafish_HD_IPN12_839_v1)")
    p.add_argument("--config-path", default=None,
                   help="explicit config yaml for the model (when the log dir "
                        "has no config.yaml, e.g. cv runs)")
    p.add_argument("--tag", default=None,
                   help="suffix for the model PNG: functional_panel_model_<tag>.png")
    p.add_argument("--title", default=None, help="override model panel title")
    p.add_argument("--real-decoder", choices=["mlp", "pva"], default="mlp",
                   help="how to decode HD from the real traces (default: a "
                        "small trained MLP; pva = connectome ring PVA)")
    p.add_argument("--stimulus", choices=["omr", "rotation"], default="rotation",
                   help="which task block to compare on: 'rotation' uses the "
                        "real 45°/s grating heading (stimParam3); 'omr' uses "
                        "the directional omr_turning labels + nominal ω")
    p.add_argument("--kino-sort", choices=["side", "rastermap"],
                   default="rastermap",
                   help="real-panel kinograph row order: 'rastermap' = the same "
                        "whole-brain order as figures/zebrafish/zapbench_full_kinograph.py "
                        "(drops the L/R partition, viridis LUT); 'side' = group "
                        "by hemisphere then ring bin (black-green LUT)")
    p.add_argument("--out", default=None)
    p.add_argument("--which", choices=["both", "real", "model"], default="both")
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out or os.path.join(_REPO_ROOT, "figures", "zebrafish")
    os.makedirs(out_dir, exist_ok=True)

    rows, circuit = build_rows(args.connectome, args.circuit)
    print(f"[rows] {len(rows)} bump-pool neurons, {int(rows['matched'].sum())} mapped")

    # Build the shared stimulus drive. 'rotation' uses the real 45°/s grating
    # heading (stimParam3, the "8. Rotations" sequence); 'omr' uses the
    # directional omr_turning labels with a nominal ω.
    z = np.load(os.path.join(args.connectome, "functional",
                             "circuit_functional_traces.npz"), allow_pickle=True)
    sample_steps = None
    display_drive = None
    if args.stimulus == "rotation":
        theta_hr, ts, theta_frame = rotation_headings(
            args.fishfuncem_data, args.connectome, args.model_dt)
        keep = ts < z["traces"].shape[0]
        ts = ts[keep]; theta_frame = theta_frame[: len(ts)]
        drive_real = heading_to_drive(theta_frame, dt=0.915, src_dt=0.915)
        drive_model = heading_to_drive(theta_hr, dt=args.model_dt,
                                       src_dt=args.model_dt)   # true 45°/s
        display_drive = drive_real                              # frame grid
        sample_steps = np.round(np.arange(len(ts)) * 0.915 / args.model_dt
                                ).astype(np.int64)
        real_title = "REAL zapbench ΔF/F — Rotations (45°/s grating, imaging-sampled)"
        block = "Rotations"
        print(f"[stim] Rotations: {len(ts)} frames "
              f"(~{len(ts) * 0.915 / 60:.1f} min)")
    else:
        stim, ts = omr_block_stim(args.fishfuncem_data,
                                  trace_len=z["traces"].shape[0])
        drive_real = stim_to_drive(stim, dt=0.915, nominal_omega=args.nominal_omega)
        drive_model = stim_to_drive(stim, dt=args.model_dt,
                                    nominal_omega=args.nominal_omega)
        real_title = "REAL zapbench ΔF/F — omr_turning (dIPN ring + IPN12)"
        block = "omr"
        print(f"[stim] omr_turning: {len(ts)} frames "
              f"(~{len(ts) * 0.915 / 60:.1f} min), ω={args.nominal_omega}°/s")

    # Order the kinograph rows like the whole-brain kinograph (rastermap) so the
    # real and model panels read together and compare row-for-row; this drops
    # the hemisphere grouping, so render with viridis + no L/R partition. The
    # rastermap order can only place neurons with a zapbenchId, so keep just the
    # 300 mapped bump neurons (the same set the real panel shows) instead of
    # piling the unmatched rows up as a black block at the bottom.
    rmap = args.kino_sort == "rastermap"
    panel_rows = (sort_rows_rastermap(rows[rows["matched"]].reset_index(drop=True))
                  if rmap else rows)
    kino_kw = dict(cmap_name="viridis" if rmap else "black_green",
                   show_partition=not rmap)

    if args.which in ("real", "both"):
        d = real_panel(panel_rows, args.connectome, ts, drive_real,
                       decoder=args.real_decoder)
        # white background, black predicted trace, no L/R + F/B swim panels
        render(d, panel_rows, os.path.join(out_dir, "functional_panel_real.png"),
               real_title, bg="white", show_swim_panels=False, **kino_kw)
    if args.which in ("model", "both"):
        # Same 300-neuron rastermap row set: the model's GCaMP rows are filled by
        # bodyId; neurons the model lacks (e.g. IPN12 in the 731-cell dipn runs)
        # stay BLANK (black), so the kinograph also reads as a coverage map.
        d = model_panel(panel_rows, args.log_dir, drive_model, device,
                        model_circuit=args.model_circuit, warmup_s=args.warmup_s,
                        config_path=args.config_path,
                        sample_steps=sample_steps, display=display_drive,
                        gcamp=dict(tau_rise=args.gcamp_tau_rise,
                                   tau_decay=args.gcamp_tau_decay,
                                   length_s=args.gcamp_length))
        suffix = f"_{args.tag}" if args.tag else ""
        title = args.title or (
            f"model → GCaMP7f calcium — SAME {block} stimulus "
            f"({args.tag or os.path.basename(args.log_dir)})")
        render(d, panel_rows, os.path.join(out_dir, f"functional_panel_model{suffix}.png"),
               title, **kino_kw)


if __name__ == "__main__":
    main()
