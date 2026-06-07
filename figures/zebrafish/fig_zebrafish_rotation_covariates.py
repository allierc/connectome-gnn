"""Figure: all locally-available recorded data along the ZAPBench *rotation* block.

Companion to the calcium-reconstruction figure (the paper's Fig. 12, produced by
``graph_tester`` / ``plot_cx.plot_calcium_reconstruction``). That figure shows the
real bump-pool ΔF/F (non-stationary) above the learned rollout (periodic,
deterministic). The obvious question it raises — *why is the real activity not
periodic when the angular-velocity drive is?* — motivates this figure: it lays
out, on a single shared time axis over the rotation block, every recorded /
derived covariate we have on disk, so the hidden non-periodic structure (slow
drift, behaviour) is visible next to the periodic stimulus.

Rotation block
--------------
Imaging frames ``[6623, 7279)`` (656 frames ≈ 600 s, 45°/s grating), the same
slice ``make_calcium_dataset`` / ``zapbench_stimulus.rotation_headings`` use.
Everything is plotted at the imaging-frame rate (src_dt≈0.915 s) except ω, which
is drawn on the fine model grid (dt=0.01) it is defined on; both span 0–600 s.

Panels (top→bottom)
-------------------
  1. ω angular-velocity drive (°/s)      — PERIODIC stimulus (the model input).
  2. heading (°, wrapped)                — integral of ω.
  3. stimulus direction (±1)             — stim_info_frames, the L/R grating code
                                           (PERIODIC; sanity that it tracks ω).
  4. swim behaviour — forward            — bilateral tail-EMG swim amplitude
  5. swim behaviour — turning            —  / L-R asymmetry (see PROVENANCE below).
  6. bump-pool real ΔF/F (recorded)      — n≈300 rastermap-ordered observed
                                           neurons, identical row set + style as
                                           Fig. 12's real panel.

PROVENANCE of forward/turning (panels 4-5) — RESOLVED
-----------------------------------------------------
``data/functional/forward_turning_from_neurons1201.npz`` (keys ``forward``,
``turning``; one value per imaging frame, 7870 frames) is the fish's BEHAVIOURAL
MOTOR OUTPUT derived from the two tail-muscle EMG electrodes — NOT a neural
decode (my earlier "neuron-decoded" label was wrong). The method is the
``fishfuncem.functional.FishEphys`` swim pipeline:
  * ephys ch0 = muscular activity LEFT, ch1 = RIGHT (EPHYS_CHANNELS, FishEphys.py)
  * per channel: outlier-clip → 40 ms windowed-variance EMG envelope →
    swim-bout detection → drift-removal → sliding-window norm →
    swim_power (rising-exp weight) and turn_power (decaying-exp weight).
  * ``_derive_behavior``: amplitude = (swim_power_L + swim_power_R)/2  → FORWARD
    (bilateral swim thrust); direction = [turn_power_L, turn_power_R] → TURNING
    (left/right asymmetry).
The exact script that wrote THIS npz (combining/renaming to forward/turning and
resampling 6 kHz → imaging frames, likely via cell_ephys_index — hence the
"from_neurons" = on-the-neuron-frame-grid tag) is NOT in the repo; it was
committed as a data artifact on the upstream ``kperks`` branch
(github.com/ahrens-fish-lab/fishFuncEM). Consumed once in
``notebooks/04_functional_analysis.py`` as ``swim`` ("decoded swim signals").

Because these are INDEPENDENT behaviour (motor output, not read off the calcium),
they are a legitimate covariate for the real ΔF/F's non-periodic structure.
Empirically over the rotation block: ``turning`` tracks the grating
(corr with ω ≈ 0.76 — optomotor following, largely PERIODIC here); ``forward`` is
a slow non-periodic drift, uncorrelated with |ω| (lag-1 autocorr ≈ 0.97).

  6'. raw tail-EMG ch0 (left) + ch1 (right) — per-frame RMS of the two motor
      channels (the SOURCE of panels 4-5). Only drawn when the raw ephys file is
      present (see ``_find_ephys_file``); otherwise a placeholder panel says so.

The raw ephys lives in the ~290 MB ``…_emf3.10chFlt`` /
``stimuli_and_ephys.10chFlt`` file (GCS / cloned-repo ``data/``), not committed
here. Drop it under ``papers/fishFuncEM/data/`` (or set ``$FISH_EPHYS_FILE``) and
the two EMG panels populate automatically — aligned to the imaging frames via the
``markers`` map in ``onsets_processed.npz``. Channels 0/1 are the left/right
tail-muscle electrodes; their windowed-variance envelope is exactly what the
FishEphys swim pipeline turns into the forward/turning of panels 4-5.

Usage
-----
    python figures/zebrafish/fig_zebrafish_rotation_covariates.py
writes ``figures/zebrafish/fig_zebrafish_rotation_covariates.png``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (os.path.join(_REPO, "src"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

CONNECTOME = os.path.join(_REPO, "figures", "zebrafish",
                          "zebrafish_connectome_HD_IPN12")
CIRCUIT = "zebrafish_HD_IPN12_839_v1"
FUNC = os.path.join(CONNECTOME, "functional")
FISHFUNC = os.path.join(_REPO, "papers", "fishFuncEM", "data", "functional")
FISHDATA = os.path.join(_REPO, "papers", "fishFuncEM", "data")
SRC_DT = 0.915                              # imaging-frame timestep (s)
N_EPHYS_CH = 10                             # channels in the .10chFlt binary
EPHYS_L, EPHYS_R = 0, 1                     # ch0 = left tail-EMG, ch1 = right

# Fig. 12 real-ΔF/F panel style (plot_cx.plot_calcium_reconstruction).
VIRIDIS_VMIN, VIRIDIS_VMAX = -2.0, 3.0


def _find_ephys_file():
    """Locate the raw 10-channel ephys binary, or None if absent.

    Searches ``$FISH_EPHYS_FILE`` first, then any ``*.10chFlt`` under
    ``papers/fishFuncEM/data`` (the cloned-repo / GCS download location).
    """
    import glob
    env = os.environ.get("FISH_EPHYS_FILE")
    if env and os.path.isfile(env):
        return env
    hits = sorted(glob.glob(os.path.join(FISHDATA, "**", "*.10chFlt"),
                            recursive=True))
    return hits[0] if hits else None


def _load_ephys_per_frame_rms(path, markers, f0, f1):
    """Per-imaging-frame RMS of the two tail-EMG channels over [f0, f1).

    ``markers[f]`` holds the 6 kHz ephys sample indices of imaging frame f's
    72 z-planes (from ``onsets_processed.npz``); we take each frame's ephys
    window ``[markers[f][0], markers[f+1][0])`` and reduce the raw EMG to a
    single RMS amplitude — an envelope on the SAME 656-frame grid as every
    other panel. Returns (emg_L, emg_R), each shape (f1-f0,).
    """
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % N_EPHYS_CH:
        raise ValueError(f"{path}: not divisible by {N_EPHYS_CH} channels")
    raw = raw.reshape((-1, N_EPHYS_CH)).T            # (10, T_ephys)
    chL, chR = raw[EPHYS_L], raw[EPHYS_R]

    def rms_per_frame(ch):
        out = np.full(f1 - f0, np.nan, np.float64)
        for i, f in enumerate(range(f0, f1)):
            a = int(markers[f][0])
            b = int(markers[f + 1][0]) if f + 1 < len(markers) else int(markers[f][-1])
            if b > a:
                seg = ch[a:b].astype(np.float64)
                out[i] = np.sqrt(np.mean(seg * seg))
        return out

    return rms_per_frame(chL), rms_per_frame(chR)


def _wrap_deg(a_deg):
    """Wrap to [-180,180); NaN at the wrap jumps so no vertical join lines."""
    w = ((np.asarray(a_deg, np.float64) + 180.0) % 360.0) - 180.0
    if w.size > 1:
        w[1:][np.abs(np.diff(w)) > 180.0] = np.nan
    return w


def _zscore_cols(M):
    """Per-neuron (column) z-score over time, matching the Fig. 12 kinograph."""
    mu = M.mean(0, keepdims=True)
    sd = M.std(0, keepdims=True)
    return (M - mu) / np.where(sd > 1e-6, sd, 1.0)


def load_rotation_block():
    """Assemble every locally-available covariate over the rotation block.

    Returns a dict with the frame slice, time axes, and each panel array.
    """
    # --- rotation frames + heading/ω on the model grid -----------------------
    head = np.load(os.path.join(FUNC, "rotation_heading.npz"))
    frames = head["frames"].astype(np.int64)            # (656,) imaging frames
    model_dt = float(head["model_dt"])
    theta_hr_deg = head["theta_hr"].astype(np.float64)  # cumulative heading (deg)
    theta_frame_deg = head["theta_frame"].astype(np.float64)  # at imaging frames
    omega = np.gradient(theta_hr_deg, model_dt)         # deg/s on model grid
    t_model = np.arange(theta_hr_deg.shape[0]) * model_dt

    f0, f1 = int(frames[0]), int(frames[-1]) + 1        # [6623, 7279)
    n_fr = f1 - f0
    t_frame = np.arange(n_fr) * SRC_DT                  # imaging-frame seconds

    # --- stimulus direction (unified stim code), rotation slice --------------
    stim = np.load(os.path.join(FISHFUNC, "stim_info_frames.npz"))["stim_info_frames"]
    stim_rot = stim[f0:f1].astype(np.float32)           # ±1 L/R grating

    # --- neuron-decoded behaviour, rotation slice ----------------------------
    beh = np.load(os.path.join(FISHFUNC, "forward_turning_from_neurons1201.npz"))
    forward_rot = beh["forward"][f0:f1].astype(np.float32)
    turning_rot = beh["turning"][f0:f1].astype(np.float32)

    # --- bump-pool real ΔF/F (same rows/order as Fig. 12's real panel) -------
    import zebrafish_functional_traces_panel as panel
    rows, _ = panel.build_rows(CONNECTOME, CIRCUIT)
    rows = panel.sort_rows_rastermap(rows)
    rows = rows[rows["matched"]].reset_index(drop=True)   # observed bump neurons

    npz = np.load(os.path.join(FUNC, "circuit_functional_traces.npz"),
                  allow_pickle=True)
    obs_body = npz["bodyId"].astype(np.int64)
    obs_pos = {int(b): i for i, b in enumerate(obs_body)}
    obs_ix = np.array([obs_pos[int(b)] for b in rows["bodyId"].to_numpy()],
                      np.int64)
    ca = np.asarray(npz["traces"], np.float32)[f0:f1][:, obs_ix]   # (656, n_bump)

    # --- raw tail-EMG channels (ch0=L, ch1=R), if the .10chFlt is present ----
    # markers[f] = 6 kHz ephys sample indices of frame f (onsets_processed.npz).
    emg_L = emg_R = None
    ephys_path = _find_ephys_file()
    if ephys_path is not None:
        od = np.load(os.path.join(FISHFUNC, "onsets_processed.npz"),
                     allow_pickle=True)["onsets_dict"].item()
        emg_L, emg_R = _load_ephys_per_frame_rms(ephys_path, od["markers"], f0, f1)
        print(f"[ephys] loaded {ephys_path}: per-frame EMG RMS ch0(L)/ch1(R)")
    else:
        print("[ephys] no .10chFlt found (set $FISH_EPHYS_FILE or drop it under "
              "papers/fishFuncEM/data/) — EMG panels show a placeholder")

    return dict(frames=(f0, f1), n_fr=n_fr,
                t_frame=t_frame, t_model=t_model,
                omega=omega, heading_deg=theta_frame_deg,
                stim_dir=stim_rot, forward=forward_rot, turning=turning_rot,
                emg_L=emg_L, emg_R=emg_R, ephys_path=ephys_path,
                calcium=ca, n_bump=ca.shape[1])


def make_figure(out_png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load_rotation_block()
    f0, f1 = d["frames"]
    t = d["t_frame"]
    have_emg = d["emg_L"] is not None

    # ω | heading | stim | forward | turning | EMG ch0(L) | EMG ch1(R) | kinograph
    ratios = [1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 2.8]
    fig, axs = plt.subplots(len(ratios), 1, sharex=True,
                            figsize=(13, 0.95 * sum(ratios)),
                            gridspec_kw=dict(height_ratios=ratios),
                            squeeze=True)

    # 1) ω drive (model grid) — periodic
    axs[0].plot(d["t_model"], d["omega"], color="0.2", lw=0.6)
    axs[0].axhline(0, color="0.7", lw=0.4)
    axs[0].set_ylabel("ω (°/s)")
    axs[0].set_title("angular-velocity drive (model input — PERIODIC)")

    # 2) heading (wrapped)
    axs[1].plot(t, _wrap_deg(d["heading_deg"]), color="0.2", lw=0.7)
    axs[1].set_ylim(-185, 185); axs[1].set_yticks([-180, 0, 180])
    axs[1].set_ylabel("heading (°)")
    axs[1].set_title("heading (∫ω)")

    # 3) stimulus direction (±1)
    axs[2].plot(t, d["stim_dir"], color="0.4", lw=0.7, drawstyle="steps-mid")
    axs[2].set_ylabel("stim dir")
    axs[2].set_title("stimulus direction code (stim_info — PERIODIC)")

    # 4) swim behaviour — forward (bilateral tail-EMG amplitude); slow drift.
    # Neutral purple: red/blue are reserved below for the two L/R raw channels.
    axs[3].plot(t, d["forward"], color="tab:purple", lw=0.7)
    axs[3].axhline(0, color="0.8", lw=0.4)
    axs[3].set_ylabel("forward (a.u.)")
    axs[3].set_title("swim behaviour — forward (bilateral tail-EMG amplitude): "
                     "slow drift, ~indep. of ω")

    # 5) swim behaviour — turning (L/R tail-EMG asymmetry); tracks ω (r≈0.76)
    axs[4].plot(t, d["turning"], color="tab:olive", lw=0.7)
    axs[4].axhline(0, color="0.8", lw=0.4)
    axs[4].set_ylabel("turning (a.u.)")
    axs[4].set_title("swim behaviour — turning (L/R tail-EMG asymmetry): "
                     "tracks ω, r≈0.76 (optomotor following)")

    # 6-7) raw tail-EMG per-frame RMS — the SOURCE of panels 4-5.
    #      ch0 = left (red), ch1 = right (blue) — two distinct sources.
    for ax, key, lab, col in ((axs[5], "emg_L", "ch0 left", "tab:red"),
                              (axs[6], "emg_R", "ch1 right", "tab:blue")):
        if have_emg:
            ax.plot(t, d[key], color=col, lw=0.6)
            ax.set_ylabel(f"EMG {lab}\nRMS (a.u.)")
            ax.set_title(f"raw tail-EMG {lab} (per-frame RMS — independent motor signal)")
        else:
            ax.text(0.5, 0.5, f"raw tail-EMG {lab}: .10chFlt not found\n"
                    "(set $FISH_EPHYS_FILE or drop it under papers/fishFuncEM/data/)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="0.4")
            ax.set_ylabel(f"EMG {lab}")
            ax.set_yticks([])

    # 8) bump-pool real ΔF/F kinograph — non-periodic (matches Fig. 12 real panel)
    n = d["n_bump"]
    axs[7].imshow(_zscore_cols(d["calcium"]).T, aspect="auto", cmap="viridis",
                  vmin=VIRIDIS_VMIN, vmax=VIRIDIS_VMAX,
                  extent=[t[0], t[-1], n, 0], interpolation="nearest")
    axs[7].set_ylabel(f"bump-pool\nneuron (n={n})")
    axs[7].set_title("bump-pool: real ΔF/F (recorded — NON-periodic)")
    axs[7].set_xlabel("time (s)")

    for ax in axs:
        ax.set_xlim(t[0], t[-1])

    # Figure title removed; manuscript caption carries the descriptive
    # text instead (CLAUDE.md style rule).
    fig.tight_layout()

    if out_png is None:
        out_png = os.path.join(_HERE, "fig_zebrafish_rotation_covariates.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[fig] wrote {out_png}")
    return out_png


if __name__ == "__main__":
    make_figure()
