"""ARTR L/R alternation: emergent intrinsic oscillator, or inherited drive?

The anterior rhombencephalic turning region (ARTR / HBO; Dunn et al. 2016) is
defined biologically by a *slow antiphase L/R oscillation* (~10-20 s) that is
**intrinsic**: it persists with no turning input ("dark-state" persistence) and
its period is set by ARTR's own mutual-inhibition time constants, NOT by the
stimulus. In this model ARTR (RIPN01/02/03_a/03_b, split by side) receives the
angular drive omega through W_in and recurrent feedback from the IPN ring, so
seeing it "oscillate" during a swim trial is ambiguous: the omega swim command
is itself an alternating square wave, and the heading bump rotates at |omega|/360
Hz, so a fast ARTR oscillation can be *inherited* from either the drive or the
ring without ARTR being an oscillator at all.

This script runs the three discriminating probes and renders one diagnostic
figure + prints a verdict:

  (1) Constant-omega frequency sweep.  Measure ARTR's oscillation frequency
      f_ARTR at each constant omega and compare to the bump-rotation frequency
      f_bump = |omega|/360 Hz.
        - f_ARTR tracks f_bump (scales with omega)         => RING FEEDBACK (inherited)
        - f_ARTR flat / invariant to omega, amp sustained  => INTRINSIC (emergent)
        - amplitude collapses to 0                          => purely driven, no oscillator
  (2) Square-wave omega drive.  Shows the antiphase ARTR_L/ARTR_R alternation as
      it actually appears in a swim trial, and which frequency dominates.
  (3) Zero-input free-run from a perturbation (omega = 0).  The decisive test:
      an intrinsic oscillator keeps oscillating with no drive; a driven /
      feedback system decays to a fixed point.

Usage:
    cd /workspace/connectome-gnn-cx
    python figures/zebrafish/fig_zebrafish_artr_oscillation.py \
        --run zebrafish_hd_si_ipn12_artr_pt1_selfmotion_rotation_gcamp
    # --run accepts a bare run name (searched under <data_root>/log/zebrafish/
    # and .../zebrafish/archive/) or an absolute run-dir path.

Reusable API:
    from fig_zebrafish_artr_oscillation import load_artr_model, const_omega_sweep, free_run
    net, info = load_artr_model("…_selfmotion_rotation_gcamp")
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Repo root on sys.path and as cwd: the zebrafish circuit loader reads the
# connectome from the *relative* path figures/zebrafish/zebrafish_connectome_*/,
# so the model build only succeeds when cwd is the repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

DEFAULT_DATA_ROOT = "/groups/saalfeld/home/allierc/GraphData"
L_COLOR, R_COLOR = "#c0392b", "#2c6fbb"   # red = ARTR_L, blue = ARTR_R (L/R convention)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _resolve_run_dir(run: str, data_root: str) -> str:
    if os.path.isabs(run) and os.path.isdir(run):
        return run
    for cand in (
        os.path.join(data_root, "log", "zebrafish", run),
        os.path.join(data_root, "log", "zebrafish", "archive", run),
        run,
    ):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"run dir not found for {run!r} under {data_root}")


def load_artr_model(run: str, data_root: str = DEFAULT_DATA_ROOT,
                    device: str | None = None,
                    sign_constrain_gate: bool = False):
    """Build the trained ARTR model from a run name / path.

    Returns (net, info) where info has the ARTR_L/ARTR_R index tensors, dt,
    tau and the learned opponent-gate scalars. Bypasses the dataset/group
    machinery (no graphs_data needed — only the local connectome + checkpoint).

    ``sign_constrain_gate``: the model now defaults this lock to True
    (``_sgn_l=-softplus``, ``_sgn_r=+softplus`` forces the two hemispheres to
    opposite signs). These ARTR checkpoints were trained BEFORE the lock, with
    *signed* gate scalars (the task-only run even learned both-negative scalars,
    impossible under the lock). Loading them with the lock on reinterprets the
    stored values as pre-softplus and forces both models to perfect antiphase.
    Default False restores the as-trained behaviour; pass True only for
    checkpoints actually trained with the lock.
    """
    os.chdir(REPO)
    from connectome_gnn.config import NeuralGraphConfig
    from connectome_gnn.models.registry import create_model
    from connectome_gnn.utils import set_data_root

    run_dir = _resolve_run_dir(run, data_root)
    set_data_root(data_root)
    cfg = NeuralGraphConfig.from_yaml(os.path.join(run_dir, "config.yaml"))
    if not cfg.dataset.startswith("zebrafish/"):
        cfg.dataset = "zebrafish/" + cfg.dataset
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = create_model(cfg.graph_model.signal_model_name,
                       aggr_type=cfg.graph_model.aggr_type,
                       config=cfg, device=dev)
    # Match how the checkpoint was trained (see docstring): the stored gate
    # scalars are signed, so the sign-lock must be OFF for the forward to use
    # them as-is. Affects only _sgn_l/_sgn_r at use time, not the loaded params.
    if hasattr(net, "sign_constrain_gate"):
        net.sign_constrain_gate = bool(sign_constrain_gate)
    ckpts = glob.glob(os.path.join(run_dir, "models", "best_model_with_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {run_dir}/models/")
    ckpts.sort(key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1)))
    sd = torch.load(ckpts[-1], map_location=dev, weights_only=False)["model_state_dict"]
    # legacy fly-gate names -> zebrafish names (back-compat with old ckpts)
    for o, n in {"v_pena_l": "v_ripn_l", "v_pena_r": "v_ripn_r",
                 "v_penb_l": "v_ptipn_l", "v_penb_r": "v_ptipn_r"}.items():
        if o in sd and n not in sd:
            sd[n] = sd.pop(o)
    net.load_state_dict(sd, strict=False)
    net.eval()

    if not (hasattr(net, "_afferent_ind_artr_l") and
            hasattr(net, "_afferent_ind_artr_r")):
        raise RuntimeError(
            f"{run} has no ARTR sub-population buffers — needs a "
            f"velocity_gate='pen_artr_ptipn1[_propriocep]' circuit.")
    info = dict(
        run=os.path.basename(run_dir),
        checkpoint=os.path.basename(ckpts[-1]),
        iL=(net._afferent_ind_artr_l > 0).nonzero(as_tuple=True)[0],
        iR=(net._afferent_ind_artr_r > 0).nonzero(as_tuple=True)[0],
        dt=float(net.dt), tau=float(net.tau),
        # effective (sign-locked) gate scalars when sign_constrain_gate is on
        v_artr_l=float(net._sgn_l(net.v_artr_l)) if hasattr(net, "_sgn_l")
                 else float(net.v_artr_l),
        v_artr_r=float(net._sgn_r(net.v_artr_r)) if hasattr(net, "_sgn_r")
                 else float(net.v_artr_r),
        device=dev,
    )
    return net, info


# --------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------
def _peak_freq_mhz(x, dt, warm=1500):
    """Dominant frequency (mHz) and std-amplitude of x after dropping warm-up."""
    x = np.asarray(x)[warm:]
    x = x - x.mean()
    if x.std() < 1e-7:
        return 0.0, 0.0
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fr = np.fft.rfftfreq(len(x), d=dt)
    p[0] = 0.0
    k = int(np.argmax(p))
    return fr[k] * 1e3, float(x.std())


@torch.no_grad()
def _artr_traces(net, info, ro_h):
    h = np.asarray(ro_h)
    h = h[0] if h.ndim == 3 else h
    L = h[:, info["iL"].cpu().numpy()].mean(1)
    R = h[:, info["iR"].cpu().numpy()].mean(1)
    return L, R


def const_omega_sweep(net, info, omegas, n_steps=6000):
    """Probe (1): f_ARTR and amplitude vs constant omega, with f_bump reference."""
    from connectome_gnn.models.bump_attractor_eval import _deterministic_sweep_rollout
    dt = info["dt"]
    rows = []
    for om in omegas:
        ro = _deterministic_sweep_rollout(net, n_steps=n_steps,
                                          omega_deg_per_s=float(om),
                                          device=info["device"])
        L, R = _artr_traces(net, info, ro["h"])
        fL, aL = _peak_freq_mhz(L, dt)
        rows.append(dict(omega=float(om), f_artr=fL, amp=aL,
                         f_bump=abs(om) / 360.0 * 1e3))
    return rows


@torch.no_grad()
def square_wave_drive(net, info, omega_amp=45.0, half_s=40.0, total_s=120.0):
    """Probe (2): antiphase ARTR L/R under an alternating swim command."""
    dt = info["dt"]
    T = int(total_s / dt)
    half = int(half_s / dt)
    om = np.zeros(T, np.float32)
    for b in range(0, T, half):
        om[b:b + half] = omega_amp if (b // half) % 2 == 0 else -omega_amp
    u = np.zeros((1, T, 3), np.float32)
    u[0, :, 0] = om
    u[0, 0, 1] = 1.0   # heading cue at t=0 only
    _, h = net(torch.from_numpy(u).to(info["device"]))
    L, R = _artr_traces(net, info, h.cpu().numpy())
    t = np.arange(T) * dt
    return t, om, L, R


@torch.no_grad()
def free_run(net, info, total_s=80.0, eps=1.0, seed=1):
    """Probe (3): zero-input free-run from a perturbation (the intrinsic test)."""
    dt, tau = info["dt"], info["tau"]
    T = int(total_s / dt)
    Wrec = net.W_rec
    u = torch.zeros(1, 3, device=info["device"])     # omega = 0, no cue
    h = (eps * torch.randn(1, net.n_units,
                           generator=torch.Generator().manual_seed(seed))
         ).to(info["device"])
    L = np.empty(T); R = np.empty(T)
    dot = dt / tau
    iL, iR = info["iL"], info["iR"]
    for k in range(T):
        r = net._sigma(h)
        h = h + dot * (-h + r @ Wrec.T + net._project_in(u) + net.b)
        s = net._sigma(h)
        L[k] = s[0, iL].mean().item()
        R[k] = s[0, iR].mean().item()
    t = np.arange(T) * dt
    return t, L, R


def classify(sweep_rows, free_amp):
    """Verdict string from the probes."""
    if free_amp > 0.2:
        return ("INTRINSIC (emergent): ARTR self-oscillates with no drive "
                f"(free-run amp={free_amp:.2f}).")
    # period tracking the bump across omega => inherited via ring feedback
    osc = [r for r in sweep_rows if r["amp"] > 0.05]
    tracks = (len(osc) >= 2 and
              np.corrcoef([r["omega"] for r in osc],
                          [r["f_artr"] for r in osc])[0, 1] > 0.5)
    note = ("the ARTR period tracks the bump period 360/|omega|" if tracks
            else "ARTR settles to a fixed offset")
    return ("INHERITED (not emergent): no self-sustained oscillation "
            f"(free-run amp={free_amp:.3f}); {note}. The swim-trial "
            "alternation follows the omega command + bump feedback.")


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def make_figure(net, info, out_path, omegas=None):
    if omegas is None:
        omegas = [5., 10., 15., 22.5, 30., 45., 60., 90., 135.]
    sweep = const_omega_sweep(net, info, omegas)
    t_sq, om_sq, L_sq, R_sq = square_wave_drive(net, info)
    t_fr, L_fr, R_fr = free_run(net, info)
    f_fr, a_fr = _peak_freq_mhz(L_fr - R_fr, info["dt"], warm=int(20 / info["dt"]))
    verdict = classify(sweep, a_fr)

    om = np.array([r["omega"] for r in sweep])
    f_artr = np.array([r["f_artr"] for r in sweep])
    amp = np.array([r["amp"] for r in sweep])
    f_bump = np.array([r["f_bump"] for r in sweep])
    osc = amp > 0.05

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    LF, TF, LET = 13, 11, 18

    # (a) period sweep: ARTR period vs bump period reference (seconds)
    a = ax[0, 0]
    artr_T = 1e3 / np.maximum(f_artr, 1e-9)          # ms->s: 1000/mHz
    bump_T = 360.0 / np.maximum(np.abs(om), 1e-9)    # s per bump revolution
    a.plot(om[osc], bump_T[osc], "k--s", mfc="none", ms=6, lw=1.5,
           label=r"bump period $360/|\omega|$")
    a.plot(om[osc], artr_T[osc], "o-", color="#7b2d8e", ms=7,
           label="measured ARTR period")
    a.set_xlabel(r"constant $\omega$  (deg/s)", fontsize=LF)
    a.set_ylabel("ARTR oscillation period (s)", fontsize=LF)
    if (~osc).any():
        x0 = (om[osc].max() + om[~osc].min()) / 2 if osc.any() else om[~osc].min()
        a.axvspan(x0, om.max() * 1.05, color="0.85", alpha=0.6, lw=0)
        a.set_xlim(om.min() - 5, om.max() + 5)
        ylo, yhi = a.get_ylim()
        a.text((x0 + om.max()) / 2, yhi * 0.92, "settled\n(no oscillation)",
               ha="center", va="top", fontsize=TF - 1, color="0.4")
    a.legend(fontsize=TF - 1, frameon=False, loc="upper center")
    a.tick_params(labelsize=TF)

    # (b) amplitude vs omega -> collapse = not self-sustained
    a = ax[0, 1]
    a.plot(om, amp, "o-", color="#7b2d8e", ms=7)
    a.axhline(0, color="grey", lw=0.8)
    a.set_xlabel(r"constant $\omega$  (deg/s)", fontsize=LF)
    a.set_ylabel(r"ARTR oscillation amplitude (std of $\bar h_L$)", fontsize=LF)
    a.tick_params(labelsize=TF)

    # (c) square-wave drive: antiphase L/R following the drive
    a = ax[1, 0]
    a2 = a.twinx()
    a2.plot(t_sq, om_sq, color="0.6", lw=1.2)
    a2.set_ylabel(r"$\omega$ drive (deg/s)", fontsize=LF, color="0.5")
    a2.tick_params(labelsize=TF, colors="0.5")
    a.plot(t_sq, L_sq, color=L_COLOR, lw=1.4, label="ARTR$_L$")
    a.plot(t_sq, R_sq, color=R_COLOR, lw=1.4, label="ARTR$_R$")
    a.set_xlabel("time (s)", fontsize=LF)
    a.set_ylabel(r"ARTR $\bar h$", fontsize=LF)
    a.legend(fontsize=TF, frameon=False, loc="upper right")
    a.tick_params(labelsize=TF)
    a.set_zorder(a2.get_zorder() + 1); a.patch.set_visible(False)

    # (d) zero-input free-run: decay to fixed point = not intrinsic
    a = ax[1, 1]
    a.plot(t_fr, L_fr, color=L_COLOR, lw=1.4, label="ARTR$_L$")
    a.plot(t_fr, R_fr, color=R_COLOR, lw=1.4, label="ARTR$_R$")
    a.set_xlabel("time (s)  —  $\\omega=0$, free-run from perturbation", fontsize=LF)
    a.set_ylabel(r"ARTR $\bar h$", fontsize=LF)
    a.legend(fontsize=TF, frameon=False, loc="upper right")
    a.tick_params(labelsize=TF)

    for letter, axx in zip("abcd", ax.ravel()):
        axx.text(-0.12, 1.04, letter, transform=axx.transAxes,
                 fontsize=LET, fontweight="bold", ha="left", va="bottom")

    # No on-figure title (paper convention: panel letters only, no titles).
    # The verdict is returned + printed for the caption, not drawn on the figure.
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return verdict, sweep, a_fr


# --------------------------------------------------------------------------
# Real-vs-model ARTR kinograph
# --------------------------------------------------------------------------
REAL_NPZ_DEFAULT = os.path.join(HERE, "functional_panel_real_rotation.npz")
MODEL_FULL_NPZ_DEFAULT = os.path.join(
    HERE, "functional_panel_zebrafish_hd_si_ipn12_artr_pt1_"
          "selfmotion_rotation_gcamp7f_full.npz")


def _zscore_rows(A):
    A = np.asarray(A, np.float32)
    m = A.mean(1, keepdims=True)
    s = A.std(1, keepdims=True)
    return (A - m) / np.where(s < 1e-6, 1.0, s)


def kinograph_data(run="zebrafish_hd_si_ipn12_artr_pt1_selfmotion_rotation_gcamp",
                   real_npz=REAL_NPZ_DEFAULT, model_full_npz=MODEL_FULL_NPZ_DEFAULT,
                   data_root=DEFAULT_DATA_ROOT, gcamp="gcamp7f"):
    """Match the recorded ARTR neurons to their modelled counterparts and
    group them by hemisphere. Returns real/model kinographs (n_match x T),
    side labels, omega(t), and the L/R population means + correlations.

    The model side is computed by driving the trained network with the
    *recorded* omega through its velocity gate (the way the model is actually
    used), then GCaMP-convolving the voltage to match the real DeltaF/F domain.
    (We do NOT read the stored anatomy_voltage calcium dump: that rollout drives
    the net differently and does not route omega through the ARTR gate.)"""
    import torch
    from connectome_gnn.models.gcamp import create_gcamp
    R = np.load(real_npz, allow_pickle=True)
    M = np.load(model_full_npz, allow_pickle=True)
    real, rbid, omega = R["aff_ARTR"], R["affbid_ARTR"], R["omega"]
    t_sec = R["t_sec"]; dt_sec = float(R["dt_sec"])
    mbid = M["body_ids"]                       # circuit neuron ordering only

    net, info = load_artr_model(run, data_root)
    dt = info["dt"]
    # drive the recorded omega through the gate (upsample imaging grid -> model dt)
    reps = max(1, int(round(dt_sec / dt)))
    om_hi = np.repeat(omega.astype(np.float32), reps)
    u = np.zeros((1, len(om_hi), 3), np.float32)
    u[0, :, 0] = om_hi; u[0, 0, 1] = 1.0       # heading cue at t=0 only
    with torch.no_grad():
        _, h = net(torch.from_numpy(u).to(info["device"]))
    ca = create_gcamp(gcamp)(h[0], dt_in=dt).cpu().numpy()    # (T_hi, N)
    ca = ca[::reps][:len(omega)]                              # (T_img, N)
    volt = h[0].cpu().numpy()[::reps][:len(omega)]            # raw voltage, NO gcamp

    idx = {int(b): i for i, b in enumerate(mbid)}
    keep = [i for i, b in enumerate(rbid) if int(b) in idx]
    real = real[np.array(keep)]
    colidx = np.array([idx[int(rbid[i])] for i in keep])
    model = ca[:, colidx].T                                   # (n_keep, T) GCaMP-convolved
    model_ng = volt[:, colidx].T                              # (n_keep, T) raw voltage (no GCaMP)
    # hemisphere from the trained model's ARTR_L/R buffers
    Lb = {int(b) for b in mbid[info["iL"].cpu().numpy()]}
    side = np.array(["L" if int(rbid[i]) in Lb else "R" for i in keep])
    # per-neuron spectral centroid (Hz) on the recorded data; cells with a
    # centroid above ~0.06 Hz (period < ~16 s) carry the fast component.
    def _centroid(x):
        x = x - x.mean()
        if x.std() < 1e-9:
            return 0.0
        p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        fr = np.fft.rfftfreq(len(x), d=dt_sec); p[0] = 0.0
        return float((fr * p).sum() / max(p.sum(), 1e-12))
    cen = np.array([_centroid(real[r]) for r in range(real.shape[0])])
    fast = cen > 0.06
    # order each hemisphere so the slow cells flank the central L/R divider and
    # the few fast cells are grouped at the OUTER edge (a labelled sub-band),
    # rather than forming a high-frequency stripe at the interface.
    order = []
    for s in ("L", "R"):
        rows = np.where(side == s)[0]
        ref = real[rows].mean(0)
        corr = np.array([np.corrcoef(real[r], ref)[0, 1] for r in rows])
        clean = rows[~fast[rows]][np.argsort(-corr[~fast[rows]])]
        fst = rows[fast[rows]]
        fst = fst[np.argsort(-cen[fst])]
        order.extend((list(fst) + list(clean)) if s == "L"
                     else (list(clean) + list(fst)))   # fast -> outer edge
    order = np.array(order)
    nLfast = int(fast[np.where(side == "L")[0]].sum())
    nRfast = int(fast[np.where(side == "R")[0]].sum())
    real, model, model_ng, side = real[order], model[order], model_ng[order], side[order]
    cen = cen[order]
    fp = 1.0 / cen[cen > 0.06] if (cen > 0.06).any() else np.array([np.inf])
    nL = int((side == "L").sum())

    def means(A):
        L = A[side == "L"].mean(0); Rr = A[side == "R"].mean(0)
        return L, Rr, float(np.corrcoef(L, Rr)[0, 1])
    rL, rR, r_real = means(real)
    mL, mR, r_model = means(model)
    mLn, mRn, r_model_ng = means(model_ng)
    return dict(real=real, model=model, model_ng=model_ng, side=side, nL=nL,
                omega=np.asarray(omega), t_sec=np.asarray(t_sec),
                real_L=rL, real_R=rR, r_real=r_real,
                model_L=mL, model_R=mR, r_model=r_model,
                model_ng_L=mLn, model_ng_R=mRn, r_model_ng=r_model_ng,
                n_match=real.shape[0], nLfast=nLfast, nRfast=nRfast,
                fast_pmin=float(np.min(fp)), fast_pmax=float(np.max(fp)))


def make_kinograph_figure(out_path, **kw):
    d = kinograph_data(**kw)
    t = d["t_sec"]; ext = [t[0], t[-1], 0, d["n_match"]]
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[0.5, 2, 2, 1.4]))
    LF, TF, LET = 12, 10, 16

    ax[0].plot(t, d["omega"], color="0.35", lw=1.2)
    ax[0].set_ylabel(r"$\omega$ (°/s)", fontsize=LF); ax[0].tick_params(labelsize=TF)

    for a, key, lab in ((ax[1], "real", f"real $\\Delta F/F$  (n={d['n_match']})"),
                        (ax[2], "model", "model ARTR")):
        a.imshow(_zscore_rows(d[key]), aspect="auto", origin="lower",
                 extent=ext, cmap="viridis", vmin=-2, vmax=2, interpolation="nearest")
        a.axhline(d["nL"], color="w", lw=1.2)             # L | R divider
        a.set_ylabel(lab, fontsize=LF)
        a.text(0.004, d["nL"] / 2, "L", color="w", fontsize=TF, va="center",
               transform=a.get_yaxis_transform())
        a.text(0.004, (d["nL"] + d["n_match"]) / 2, "R", color="w", fontsize=TF,
               va="center", transform=a.get_yaxis_transform())
        a.tick_params(labelsize=TF)

    a = ax[3]
    a.plot(t, d["real_L"] / max(np.abs(d["real_L"]).max(), 1e-9), color=L_COLOR,
           lw=1.3, label=f"real (r$_{{LR}}$={d['r_real']:+.2f})")
    a.plot(t, d["real_R"] / max(np.abs(d["real_R"]).max(), 1e-9), color=R_COLOR, lw=1.3)
    a.plot(t, d["model_L"] / max(np.abs(d["model_L"]).max(), 1e-9), color=L_COLOR,
           lw=1.1, ls="--", label=f"model (r$_{{LR}}$={d['r_model']:+.2f})")
    a.plot(t, d["model_R"] / max(np.abs(d["model_R"]).max(), 1e-9), color=R_COLOR,
           lw=1.1, ls="--")
    a.set_ylabel("L/R pop. mean\n(norm.)", fontsize=LF)
    a.set_xlabel("time (s)", fontsize=LF)
    a.legend(fontsize=TF - 1, frameon=False, ncol=2, loc="upper right")
    a.tick_params(labelsize=TF)

    for letter, axx in zip("abcd", ax):
        axx.text(-0.075, 1.02, letter, transform=axx.transAxes, fontsize=LET,
                 fontweight="bold", ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[artr_kino] real r(L,R)={d['r_real']:+.3f}  model r(L,R)={d['r_model']:+.3f}"
          f"  ({d['n_match']} matched ARTR neurons, {d['nL']} L / {d['n_match']-d['nL']} R)")
    return d


def make_kinograph_compare_figure(out_path, **kw):
    """ARTR kinograph comparing the model WITH GCaMP (voltage convolved to DeltaF/F)
    against the model WITHOUT GCaMP (raw voltage), both vs the recorded DeltaF/F.
    Shows what the calcium-indicator forward model contributes (slow temporal
    low-pass) on top of the underlying voltage."""
    d = kinograph_data(**kw)
    t = d["t_sec"]; ext = [t[0], t[-1], 0, d["n_match"]]
    fig, ax = plt.subplots(5, 1, figsize=(11, 11), sharex=True,
                           gridspec_kw=dict(height_ratios=[0.5, 2, 2, 2, 1.4]))
    LF, TF, LET = 12, 10, 16

    ax[0].plot(t, d["omega"], color="0.35", lw=1.2)
    ax[0].set_ylabel(r"$\omega$ (°/s)", fontsize=LF); ax[0].tick_params(labelsize=TF)

    for a, key, lab in ((ax[1], "real", f"real $\\Delta F/F$  (n={d['n_match']})"),
                        (ax[2], "model", "model + GCaMP"),
                        (ax[3], "model_ng", "model, no GCaMP\n(raw voltage)")):
        a.imshow(_zscore_rows(d[key]), aspect="auto", origin="lower",
                 extent=ext, cmap="viridis", vmin=-2, vmax=2, interpolation="nearest")
        a.axhline(d["nL"], color="w", lw=1.2)
        a.set_ylabel(lab, fontsize=LF)
        a.text(0.004, d["nL"] / 2, "L", color="w", fontsize=TF, va="center",
               transform=a.get_yaxis_transform())
        a.text(0.004, (d["nL"] + d["n_match"]) / 2, "R", color="w", fontsize=TF,
               va="center", transform=a.get_yaxis_transform())
        a.tick_params(labelsize=TF)

    a = ax[4]
    nrm = lambda v: v / max(np.abs(v).max(), 1e-9)
    a.plot(t, nrm(d["real_L"]), color=L_COLOR, lw=1.3,
           label=f"real (r$_{{LR}}$={d['r_real']:+.2f})")
    a.plot(t, nrm(d["real_R"]), color=R_COLOR, lw=1.3)
    a.plot(t, nrm(d["model_L"]), color=L_COLOR, lw=1.1, ls="--",
           label=f"+GCaMP (r$_{{LR}}$={d['r_model']:+.2f})")
    a.plot(t, nrm(d["model_R"]), color=R_COLOR, lw=1.1, ls="--")
    a.plot(t, nrm(d["model_ng_L"]), color=L_COLOR, lw=1.0, ls=":",
           label=f"no GCaMP (r$_{{LR}}$={d['r_model_ng']:+.2f})")
    a.plot(t, nrm(d["model_ng_R"]), color=R_COLOR, lw=1.0, ls=":")
    a.set_ylabel("L/R pop. mean\n(norm.)", fontsize=LF)
    a.set_xlabel("time (s)", fontsize=LF)
    a.legend(fontsize=TF - 1, frameon=False, ncol=3, loc="upper right")
    a.tick_params(labelsize=TF)

    for letter, axx in zip("abcde", ax):
        axx.text(-0.075, 1.02, letter, transform=axx.transAxes, fontsize=LET,
                 fontweight="bold", ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[artr_kino_cmp] real r={d['r_real']:+.3f}  +GCaMP r={d['r_model']:+.3f}  "
          f"no-GCaMP r={d['r_model_ng']:+.3f}")
    return d


def make_kinograph_train_compare_figure(out_path, run_obs=None, run_noobs=None,
                                        data_root=DEFAULT_DATA_ROOT, gcamp="gcamp7f"):
    """ARTR kinograph comparing the model trained WITH the calcium-observation
    loss (the ``_gcamp`` run) against the one trained WITHOUT it (task-only),
    both vs the recorded DeltaF/F. Both model sides are computed identically
    (recorded omega -> velocity gate -> GCaMP convolution); only the trained
    checkpoint differs. Demonstrates that the antiphase L/R opponency is
    recruited by the observation loss, not by the heading task alone."""
    run_obs = run_obs or "zebrafish_hd_si_ipn12_artr_pt1_selfmotion_rotation_gcamp"
    run_noobs = run_noobs or (run_obs[:-6] if run_obs.endswith("_gcamp") else run_obs)
    d_obs = kinograph_data(run=run_obs, data_root=data_root, gcamp=gcamp)
    d_no = kinograph_data(run=run_noobs, data_root=data_root, gcamp=gcamp)
    # both share the recorded data + neuron order (deterministic from `real`)
    d = d_obs
    t = d["t_sec"]; ext = [t[0], t[-1], 0, d["n_match"]]
    fig, ax = plt.subplots(7, 1, figsize=(11, 13), sharex=True,
                           gridspec_kw=dict(
                               height_ratios=[0.5, 2, 2, 2, 1.1, 1.1, 1.1]))
    LF, TF, LET = 12, 10, 16

    ax[0].plot(t, d["omega"], color="0.35", lw=1.2)
    ax[0].set_ylabel(r"$\omega$ (°/s)", fontsize=LF); ax[0].tick_params(labelsize=TF)

    panels = ((ax[1], d["real"], f"real $\\Delta F/F$  (n={d['n_match']})",
               d["r_real"]),
              (ax[2], d_obs["model"], "model\nWITH obs.\\ loss", d_obs["r_model"]),
              (ax[3], d_no["model"], "model\nWITHOUT obs.\\ loss", d_no["r_model"]))
    for a, arr, lab, rlr in panels:
        a.imshow(_zscore_rows(arr), aspect="auto", origin="lower",
                 extent=ext, cmap="viridis", vmin=-2, vmax=2, interpolation="nearest")
        a.axhline(d["nL"], color="w", lw=1.2)
        a.set_ylabel(lab, fontsize=LF)
        a.text(0.004, d["nL"] / 2, "L", color="w", fontsize=TF, va="center",
               transform=a.get_yaxis_transform())
        a.text(0.004, (d["nL"] + d["n_match"]) / 2, "R", color="w", fontsize=TF,
               va="center", transform=a.get_yaxis_transform())
        a.text(0.99, 0.92, f"$r_{{LR}}={rlr:+.2f}$", color="w", fontsize=TF,
               ha="right", va="top", transform=a.transAxes)
        a.tick_params(labelsize=TF)

    # mark the 'fast' sub-band (cells with a fast component) at the outer edge
    # of the L block on the recorded panel; same rows across all three panels.
    nf = d.get("nLfast", 0)
    if nf:
        tr = ax[1].get_yaxis_transform()
        ax[1].plot([-0.018, -0.018], [0.3, nf - 0.3], color="#e8820e", lw=4,
                   transform=tr, clip_on=False, solid_capstyle="butt")
        ax[1].text(-0.052, nf / 2,
                   f"fast\n{d['fast_pmin']:.0f}–{d['fast_pmax']:.0f}\\,s",
                   rotation=90, color="#e8820e", fontsize=TF - 2,
                   ha="center", va="center", transform=tr)

    # L/R population means, one panel per condition (real / with obs. / without)
    nrm = lambda v: v / max(np.abs(v).max(), 1e-9)
    mean_specs = (
        (ax[4], d["real_L"], d["real_R"], d["r_real"], "real\nL/R mean"),
        (ax[5], d_obs["model_L"], d_obs["model_R"], d_obs["r_model"],
         "with obs.\nL/R mean"),
        (ax[6], d_no["model_L"], d_no["model_R"], d_no["r_model"],
         "without obs.\nL/R mean"),
    )
    for a, Lv, Rv, rlr, lab in mean_specs:
        a.plot(t, nrm(Lv), color=L_COLOR, lw=1.3, label="ARTR$_L$")
        a.plot(t, nrm(Rv), color=R_COLOR, lw=1.3, label="ARTR$_R$")
        a.axhline(0, color="0.8", lw=0.6, zorder=0)
        a.set_ylabel(lab, fontsize=LF)
        a.text(0.99, 0.93, f"$r_{{LR}}={rlr:+.2f}$", ha="right", va="top",
               transform=a.transAxes, fontsize=TF)
        a.set_ylim(-1.15, 1.15)
        a.tick_params(labelsize=TF)
    ax[4].legend(fontsize=TF - 1, frameon=False, ncol=2, loc="lower left")
    ax[6].set_xlabel("time (s)", fontsize=LF)

    for letter, axx in zip("abcdefg", ax):
        axx.text(-0.075, 1.02, letter, transform=axx.transAxes, fontsize=LET,
                 fontweight="bold", ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[artr_kino_train] real r={d['r_real']:+.3f}  "
          f"WITH-obs r={d_obs['r_model']:+.3f}  WITHOUT-obs r={d_no['r_model']:+.3f}")
    return d_obs, d_no


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run",
                   default="zebrafish_hd_si_ipn12_artr_pt1_selfmotion_rotation_gcamp",
                   help="run name (searched under log/zebrafish[/archive]) or abs path")
    p.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--run_noobs", default=None,
                   help="task-only run (trained WITHOUT the calcium-observation "
                        "loss) for the kinograph comparison; default = --run with "
                        "the trailing '_gcamp' stripped")
    p.add_argument("--kinograph", action="store_true",
                   help="render the ARTR kinograph comparing real vs model trained "
                        "WITH obs. loss vs model trained WITHOUT obs. loss")
    p.add_argument("--kinograph_single", action="store_true",
                   help="render the real-vs-(single)-model ARTR kinograph only")
    p.add_argument("--kinograph_compare", action="store_true",
                   help="render the kinograph comparing the model WITH GCaMP vs "
                        "WITHOUT GCaMP (raw voltage) against the recorded DeltaF/F")
    args = p.parse_args()

    if args.kinograph:
        out = args.out or os.path.join(HERE, "fig_zebrafish_artr_kinograph.png")
        make_kinograph_train_compare_figure(
            out, run_obs=args.run, run_noobs=args.run_noobs,
            data_root=args.data_root)
        print(f"[artr_kino] wrote {out}")
        return

    if args.kinograph_single:
        out = args.out or os.path.join(HERE, "fig_zebrafish_artr_kinograph_single.png")
        make_kinograph_figure(out, run=args.run, data_root=args.data_root)
        print(f"[artr_kino] wrote {out}")
        return

    if args.kinograph_compare:
        out = args.out or os.path.join(HERE, "fig_zebrafish_artr_kinograph_gcamp_compare.png")
        make_kinograph_compare_figure(out, run=args.run, data_root=args.data_root)
        print(f"[artr_kino_cmp] wrote {out}")
        return

    net, info = load_artr_model(args.run, args.data_root, args.device)
    print(f"[artr_osc] loaded {info['run']} ({info['checkpoint']}) "
          f"dt={info['dt']:.3f} tau={info['tau']:.3f}  "
          f"ARTR_L={info['iL'].numel()} ARTR_R={info['iR'].numel()}  "
          f"gate v_artr_l={info['v_artr_l']:+.4f} v_artr_r={info['v_artr_r']:+.4f}")
    out = args.out or os.path.join(HERE, f"fig_zebrafish_artr_oscillation_{info['run']}.png")
    verdict, sweep, a_fr = make_figure(net, info, out)
    print("\n  omega(deg/s)  ARTR_period(s)  amp     bump_period(s)")
    for r in sweep:
        artr_T = 1e3 / r["f_artr"] if r["f_artr"] > 0 else float("inf")
        bump_T = 360.0 / abs(r["omega"]) if r["omega"] else float("inf")
        ap = "settled" if r["amp"] <= 0.05 else f"{artr_T:8.1f}"
        print(f"  {r['omega']:8.2f}    {ap:>10}    {r['amp']:6.3f}  {bump_T:10.1f}")
    print(f"\n  free-run (omega=0) amplitude = {a_fr:.4f}")
    print(f"\n  VERDICT: {verdict}")
    print(f"\n[artr_osc] wrote {out}")


if __name__ == "__main__":
    main()
