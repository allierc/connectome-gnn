"""ZAPBench Rotations-block stimulus for zebrafish HD probe rollouts.

The "8. Rotations" sequence is the real ±45°/s grating heading recorded in
stimParam3 (channel 6 of stimuli_and_ephys.10chFlt). This module turns that
heading into the `(1, T, 3)` model stimulus the zebrafish TaskRNN/TaskGNN
consumes, so the same rotation probe used by the functional panels is available
as a first-class `anatomy_voltage_pattern` (see plot_anatomy_voltage) and to
figures/zebrafish/fig_functional_panel.py — no rollout code duplicated.

fishfuncem is imported lazily so importing this module never requires it; only
(re)building the heading cache does (or a 288 MB GCS read, cached afterwards).
"""
from __future__ import annotations

import os

import numpy as np

# repo root: src/connectome_gnn/generators/zapbench_stimulus.py -> up 4
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     "..", "..", ".."))
_DEFAULT_CONNECTOME = os.path.join(_REPO, "figures", "zebrafish",
                                   "zebrafish_connectome_HD_IPN12")
_DEFAULT_FISHFUNCEM = os.path.join(_REPO, "papers", "fishFuncEM", "data")

ROT_TASK = 7                                   # 0-based onset index (trial_index 8)
_ROT_OBJ = "volumes/20240930/stimuli_raw/stimuli_and_ephys.10chFlt"
_ROT_BUCKET = "zapbench-release"


def _import_fishfunctional():
    """Prefer the pip-installed fishfuncem; fall back to the papers/fishFuncEM
    checkout if it isn't installed in this env, so the rotation probe works from
    any entry point (GNN_Main test runs, not just the figure scripts)."""
    try:
        from fishfuncem import FishFunctional
    except ModuleNotFoundError:
        import sys
        ff_dir = os.path.join(_REPO, "papers", "fishFuncEM")
        if os.path.isdir(ff_dir) and ff_dir not in sys.path:
            sys.path.insert(0, ff_dir)
        from fishfuncem import FishFunctional
    return FishFunctional


def rotation_headings(fishfuncem_data, connectome_dir, model_dt, src_dt=0.915):
    """Real heading (deg) for the Rotations block from stimParam3 (ch6),
    unwrapped at the full 6 kHz so the true 45°/s rotation is preserved.

    Returns (theta_hr, frames, theta_frame):
      theta_hr    — heading on the MODEL grid (model_dt); drives the model at
                    the true ±45°/s so its bump rotates correctly.
      frames      — imaging-frame indices of the block (for ΔF/F slicing).
      theta_frame — heading at imaging-frame times (1.09 Hz); used for display
                    and the MLP target. Sampling the true 45°/s rotation at
                    1.09 Hz aliases — this is the unavoidable imaging limit, so
                    the model output is sampled the SAME way for a fair compare.
    Cached so the 288 MB GCS read happens once."""
    cache = os.path.join(connectome_dir, "functional", "rotation_heading.npz")
    FishFunctional = _import_fishfunctional()
    ff = FishFunctional.from_data_dir(fishfuncem_data)
    f0 = int(np.asarray(ff.onsets_frames)[ROT_TASK])
    f1 = int(np.asarray(ff.offsets_frames)[ROT_TASK])
    frames = np.arange(f0, f1)
    n_fr = len(frames)
    dur = n_fr * src_dt
    if os.path.isfile(cache):
        z = np.load(cache)
        if z["frames"].shape == frames.shape and abs(float(z["model_dt"]) - model_dt) < 1e-9:
            return z["theta_hr"], frames, z["theta_frame"]
    on = np.load(os.path.join(fishfuncem_data, "functional",
                              "onsets_processed.npz"), allow_pickle=True
                 )["onsets_dict"].item()
    oe = np.asarray(on["onsets_ephys"]).astype(np.int64)
    s0, s1 = int(oe[ROT_TASK]), int(oe[ROT_TASK + 1])
    NCH = 10
    import urllib.request
    req = urllib.request.Request(
        f"https://storage.googleapis.com/{_ROT_BUCKET}/{_ROT_OBJ}",
        headers={"Range": f"bytes={s0 * NCH * 4}-{s1 * NCH * 4 - 1}"})
    a = np.frombuffer(urllib.request.urlopen(req).read(),
                      dtype="<f4").reshape(-1, NCH)
    theta_full = np.unwrap(a[:, 6], period=90.0)        # true heading @ 6 kHz
    src_t = np.linspace(0.0, dur, len(theta_full))
    theta_hr = np.interp(np.arange(0.0, dur, model_dt), src_t, theta_full)
    theta_frame = np.interp(np.arange(n_fr) * src_dt, src_t, theta_full)
    np.savez(cache, theta_hr=theta_hr.astype(np.float32),
             theta_frame=theta_frame.astype(np.float32),
             frames=frames, model_dt=np.float32(model_dt))
    print(f"[rotation] heading: {n_fr} frames, true 45°/s, "
          f"|path|={np.abs(np.diff(theta_full)).sum()/360:.0f} turns over "
          f"{dur/60:.1f} min (imaging-undersampled for display)")
    return theta_hr.astype(np.float32), frames, theta_frame.astype(np.float32)


def heading_to_drive(theta_deg, dt, src_dt=0.915):
    """Build a drive (omega, theta, zero ticks, t_sec) on a `dt` grid from a
    per-source-frame heading (deg). Rotation has no swim events, so the L/R and
    F/B tick channels are zero."""
    n_src = len(theta_deg)
    T_src = (n_src - 1) * src_dt
    t = np.arange(0.0, T_src, dt) if dt < src_dt else np.arange(n_src) * src_dt
    th = np.interp(t, np.arange(n_src) * src_dt, theta_deg)   # deg
    theta_rad = np.deg2rad(th)
    omega = np.gradient(th, dt)                                # deg/s
    return dict(omega=omega.astype(np.float32),
                turn_lr=np.zeros_like(omega, np.float32),
                swim_fb=np.zeros_like(omega, np.float32),
                theta=theta_rad.astype(np.float32),
                t_sec=t.astype(np.float32), rep=1)


def zapbench_rotation_stimulus(model_dt, warmup_s=10.0, connectome_dir=None,
                               fishfuncem_data=None, src_dt=0.915):
    """Build the ZAPBench Rotations model stimulus `(1, warm+T, 3)`.

    Returns a dict:
      u            — (1, warm+T, 3) float32; channel 0 = ω(°/s) on the model
                     grid (warmup zeros prepended), channels 1/2 = (cosθ0, sinθ0)
                     on frame 0 only. θ0 = 0 (the bump starts at heading 0 and
                     the recorded ω drives the rotation), matching the panels.
      warm_steps   — number of prepended warmup steps (discard after rollout).
      sample_steps — indices into the POST-warmup model grid at the 1.09 Hz
                     imaging frames (use to subsample calcium for display).
      theta_frame  — true heading (deg) at the imaging frames (display / target).
      n_frames     — number of imaging frames in the block.
      label        — caption string.
    """
    connectome_dir = connectome_dir or _DEFAULT_CONNECTOME
    fishfuncem_data = fishfuncem_data or _DEFAULT_FISHFUNCEM
    theta_hr, frames, theta_frame = rotation_headings(
        fishfuncem_data, connectome_dir, model_dt, src_dt=src_dt)

    drive = heading_to_drive(theta_hr, dt=model_dt, src_dt=model_dt)
    omega = drive["omega"]                              # °/s on model grid
    T = len(omega)
    warm = max(0, int(round(warmup_s / model_dt)))
    omega_full = np.concatenate([np.zeros(warm, np.float32), omega])
    u = np.zeros((1, warm + T, 3), np.float32)
    u[0, :, 0] = omega_full
    u[0, 0, 1] = 1.0                                   # θ0 = 0  (cos0=1, sin0=0)

    n_frames = len(theta_frame)
    sample_steps = np.clip(
        np.round(np.arange(n_frames) * src_dt / model_dt).astype(np.int64),
        0, T - 1)
    label = (f"zapbench_rotation: {n_frames} frames "
             f"(~{n_frames * src_dt / 60:.1f} min, 45°/s grating), warmup {warmup_s:g}s")
    return dict(u=u, warm_steps=warm, sample_steps=sample_steps,
                theta_frame=theta_frame.astype(np.float32),
                n_frames=n_frames, label=label)
