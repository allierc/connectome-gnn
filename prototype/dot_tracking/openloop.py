"""Open-loop tracking: the controller sees target VELOCITY, never position.

The closed-loop app (`app.py`) hands each controller the retinal error
`e = target - gaze`, so it can always correct. This script removes that. The
controller is given

    * the target's velocity v(t), and
    * the initial position, which is the centre for every trial,

and nothing else. It must therefore reconstruct position by INTEGRATING
velocity, with no reference to check itself against. This is the
configuration the oculomotor note describes: the circuit's job is
`position = integral of velocity`, and the interesting quantity is how long
that integral stays usable before drift carries the target out of the field
of view.

The prediction is that tracking works "up to a point". This script measures
where that point is, as a survival time:

    t_lose = the first time |target - gaze| exceeds FOV (0.6 grid units)

The failure modes of an integrator are separated, because each leaves a
different signature in how the error grows:

    perfect   exact integration           error stays at numerical zero
    gain      integrates k*v, k != 1      error grows with DISPLACEMENT, and
                                          is capped by the arena
    leaky     dg/dt = -g/tau + v          error saturates; gaze sags to centre

The two growth laws — bounded-linear and saturating — are the point. They are
distinguishable from a single error trace, so a real circuit's drift can be
classified rather than merely measured.

Usage::

    python openloop.py                       # browser GUI, like app.py
    python openloop.py --sweep               # batch: table + openloop.png
    python openloop.py --sweep --n-seeds 40 --duration 30
"""
from __future__ import annotations

import argparse
import os
import sys

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory import generate                       # noqa: E402
from followers import JOY_FULL_SCALE                  # noqa: E402

FOV = 0.6                    # field-of-view radius, grid units
# Display-only magnification of the stick. The open-loop command is just the
# target's own velocity, so a slow target asks for only 0.15 of the 1.6 u/s
# full scale; some magnification is needed for the course to be legible, but
# 6x overstated it and the stick swung to the gate on ordinary motion. 2.5x
# keeps a fast target near half deflection. Saturation is still judged on the
# UNSCALED command, so this never changes what counts as saturated.
JOY_VIEW_GAIN = 2.5

OPEN_LOOP = {}
PARAMS = {}


def register(name, params=()):
    def deco(fn):
        OPEN_LOOP[name] = fn
        PARAMS[name] = list(params)
        return fn
    return deco


def _knob(name, label, lo, hi, default, step):
    return {"name": name, "label": label, "min": lo, "max": hi,
            "default": default, "step": step}


def defaults(name):
    return {p["name"]: p["default"] for p in PARAMS.get(name, [])}


def _integrate(vx, vy, dt, x0, y0, leak=None):
    """Gaze from a commanded velocity, starting at (x0, y0).

    Returns (gx, gy, ux, uy): the integrator state and the command that drove
    it. The command is what the JOYSTICK holds — it is deliberately not the
    gaze velocity, because for a leaky integrator those differ: dg/dt =
    -g/tau + u, so the leak term moves the gaze without anyone touching the
    stick. Reporting dg/dt as the stick would blame the hand for the
    integrator's own imperfection.

    The stick saturates at JOY_FULL_SCALE exactly as in the closed-loop app,
    so a controller cannot cheat by commanding an arbitrarily large velocity.
    `leak` is the integrator time constant in seconds; None = perfect.
    """
    n = len(vx)
    gx = np.empty(n); gy = np.empty(n)
    ux = np.clip(vx, -JOY_FULL_SCALE, JOY_FULL_SCALE)
    uy = np.clip(vy, -JOY_FULL_SCALE, JOY_FULL_SCALE)
    cx, cy = float(x0), float(y0)
    a = 0.0 if leak is None else dt / float(leak)
    for k in range(n):
        gx[k], gy[k] = cx, cy
        cx += ux[k] * dt - a * cx
        cy += uy[k] * dt - a * cy
    return gx, gy, ux, uy


@register("perfect")
def perfect(t, vx, vy, x0, y0, dt, rng, **kw):
    """Exact integration — the upper bound. Any residual error is the
    discretisation of the integral, not a property of the controller."""
    return _integrate(vx, vy, dt, x0, y0)


@register("gain", [_knob("k", "velocity gain k", 0.0, 2.0, 0.5, 0.01)])
def gain(t, vx, vy, x0, y0, dt, rng, k=0.5, **kw):
    """Integrates k*v with k != 1: a miscalibrated velocity-to-position gain.

    Integration is linear, so the error is exactly (1-k) times the
    DISPLACEMENT from the start — not the path length. In a bounded arena
    that displacement cannot exceed the arena, which makes a gain error
    SELF-LIMITING: measured here, the target gets a median 1.16 grid units
    from the centre, so |1-k| must exceed 0.52 before the dot can ever leave
    the field of view. The default is deliberately that extreme; a realistic
    5% miscalibration is invisible in this geometry.

    The corollary matters for the circuit: in a bounded workspace, gain
    calibration is not the thing that loses the target. Leak and noise are.
    """
    return _integrate(k * vx, k * vy, dt, x0, y0)


@register("leaky", [_knob("tau", "integrator time constant tau (s)", 0.1, 32.0, 2.0, 0.1)])
def leaky(t, vx, vy, x0, y0, dt, rng, tau=2.0, **kw):
    """Leaky integrator, dg/dt = -g/tau + v — the biological one.

    The leak pulls gaze back toward the centre, so a sustained excursion is
    under-represented and the error SATURATES rather than growing without
    bound. tau is the integrator time constant the oculomotor circuit is
    supposed to make long; tau -> infinity recovers `perfect`.
    """
    return _integrate(vx, vy, dt, x0, y0, leak=tau)


# --------------------------------------------------------------------------
# the eye plant (identified in Plexus/prototype/eye/fit_plant.py)
# --------------------------------------------------------------------------
PLANT_NPZ = {"h": "/workspace/Plexus/prototype/eye/plant.npz",
             "v": "/workspace/Plexus/prototype/eye/plant_v.npz"}
DEG_PER_UNIT = 15.0          # grid units -> degrees of eccentricity
BOUND = 0.95                 # the arena half-width the target is clamped to


def deg_per_unit(scale, variant=None):
    """Grid units -> degrees. `auto` shrinks the world until the WHOLE arena
    is inside the chosen eye's reachable travel.

    Without it the target routinely sits outside the eye's range and the eye
    saturates against its own mechanics — a workspace failure that looks like
    a tracking failure. Eye A reaches only +3.4 deg of abduction, so a world
    scaled for eye C asks it for something no controller can deliver."""
    if scale != "auto":
        return float(scale)
    p = load_plants().get(variant)
    if p is None:
        return DEG_PER_UNIT
    # The world must fit inside the smaller reach of the TWO axes. Sizing it
    # from the horizontal alone put the vertical target outside the eye's
    # travel 40% of the time on eye C (60% on D) — a target it cannot look
    # at, which reads as a controller that cannot track.
    reach = []
    for ax in ("h", "v"):
        c = p[ax]["coef"]
        f = lambda u, c=c: sum(ci * u ** (i + 1) for i, ci in enumerate(c))
        reach.append(min(abs(f(-1.0)), abs(f(+1.0))))
    return float(min(reach) / BOUND)
PLANTS = {}
EYE_SCORES = {}
# A-E labels, in the order the eyes were made. The raw variant names carry
# their build history, which is useful in the archive and unreadable in a
# selector.
EYE_LABEL = {
    "eye_probe_c_a": "A", "eye_probe_baseline_fixmat": "B",
    "eye_p3a_length": "C", "eye_p3b_pulley": "D", "eye_p3c_drive": "E",
    "ideal_linear": "ideal",
}


def load_plants():
    """Every identified variant, plus an `ideal` one with a straight static
    curve. The ideal exists so the interface can separate two very different
    failures: the eye's DYNAMICS (lag and ring, which a controller can learn
    to invert) from its measured STATIC CURVE (which is non-monotone near
    zero and therefore cannot be inverted at all)."""
    if PLANTS:
        return PLANTS
    try:
        V = {ax: json.loads(str(np.load(p, allow_pickle=False)["variants"]))
             for ax, p in PLANT_NPZ.items()}
    except Exception as e:
        print(f"[plant] {type(e).__name__}: {e} — plant panel disabled")
        return PLANTS
    for k in sorted(set(V["h"]) & set(V["v"])):
        ent = {}
        for ax in ("h", "v"):
            v = V[ax][k]
            if int(v["order"]) != 2:
                break
            wn, zeta = np.exp(np.asarray(v["theta"], float))
            ent[ax] = dict(coef=np.asarray(v["coef"], float), wn=float(wn),
                           zeta=float(zeta), rms=float(v["rms"]))
        if len(ent) == 2:
            PLANTS[k] = ent
    # Same mechanics, same travel, monotone gain — the control that separates
    # the eye's dynamics from its static curve.
    ref = PLANTS.get("eye_p3a_length")
    if ref is not None:
        PLANTS["ideal_linear"] = {
            ax: dict(coef=np.array([float(sum(ref[ax]["coef"])), 0.0, 0.0]),
                     wn=ref[ax]["wn"], zeta=ref[ax]["zeta"], rms=float("nan"))
            for ax in ("h", "v")}
    return PLANTS


CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def eye_label(variant):
    """`C (0.31 deg)` — the eye and how well its own controller does on it.

    The score is the trained network's mean gaze error, so it answers the
    question a selector should answer: with a controller that has learned
    THIS eye, how well can the target be held?"""
    lab = EYE_LABEL.get(variant, variant)
    if not EYE_SCORES:
        p = os.path.join(CKPT_DIR, "")
        for f in sorted(os.listdir(CKPT_DIR)) if os.path.isdir(CKPT_DIR) else []:
            if f.startswith("eye_report_") and f.endswith(".json"):
                try:
                    with open(os.path.join(CKPT_DIR, f)) as fh:
                        EYE_SCORES[f[len("eye_report_"):-len(".json")]] = json.load(fh)
                except Exception:
                    pass
    sc = EYE_SCORES.get(variant)
    if sc and "gaze_err_mean_deg" in sc:
        return f"{lab}  ({sc['gaze_err_mean_deg']:.2f}\u00b0)"
    return f"{lab}  (untrained)"


def plant_static(coef, u):
    return sum(c * u ** (k + 1) for k, c in enumerate(coef))


def plant_run_states(name, u_cmd, dt, axis="h"):
    """As plant_run, but also returns the plant's velocity state.

    A second-order plant has TWO states, position and velocity, and the
    velocity is the one you cannot see in the world panel — it is what makes
    the eye overshoot. Exposing it is the difference between watching the
    eye and watching the mechanics."""
    p = load_plants()[name][axis]
    f = plant_static(p["coef"], np.clip(u_cmd, -1.0, 1.0))
    w2, twz = p["wn"] ** 2, 2.0 * p["zeta"] * p["wn"]
    g = np.zeros_like(f); vv = np.zeros_like(f); v = 0.0; y = 0.0
    for k in range(len(f)):
        g[k] = y; vv[k] = v
        v += dt * (w2 * (f[k] - y) - twz * v)
        y += dt * v
    return g, vv


def eye_states(base, variant, vx, vy, dt):
    """Every state of the coupled system at every timestep, for the monitor:
    the velocity input, the recurrent rates, the four motor pools, and the
    two plant states per axis."""
    import torch
    key = ("eye", base)
    if key not in _ML_CACHE:
        _eye_predict(base, variant, None, vx[:1], vy[:1], 0.0, 0.0, dt)
    m = _ML_CACHE[key]
    with torch.no_grad():
        u = torch.tensor(np.stack([vx, vy], -1)[None], dtype=torch.float32)
        alpha = (m.dt / m.log_tau.exp()).clamp(1e-4, 1.0)
        h = torch.zeros(1, m.W.shape[0])
        drive = m.enc(u)
        rates = []
        for t in range(u.shape[1]):
            r = torch.tanh(h)
            rates.append(r[0].clone())
            h = h + alpha * (-h + r @ m.W.T + drive[:, t])
        R = torch.stack(rates, 0).numpy()
        pools = torch.nn.functional.softplus(
            m.mot(torch.tensor(R)[None]))[0].numpy()
    return R, pools


def plant_run(name, u_cmd, dt, axis="h"):
    """Signed command in [-1,1] -> gaze in degrees, on one axis."""
    p = load_plants()[name][axis]
    f = plant_static(p["coef"], np.clip(u_cmd, -1.0, 1.0))
    w2, twz = p["wn"] ** 2, 2.0 * p["zeta"] * p["wn"]
    g = np.zeros_like(f); v = 0.0; y = 0.0
    for k in range(len(f)):
        g[k] = y
        v += dt * (w2 * (f[k] - y) - twz * v)
        y += dt * v
    return g


def plant_curve(name, n=201, axis="h"):
    """Phi sampled over the command range, plus where its slope is negative —
    the band in which the plant cannot be inverted."""
    p = load_plants()[name][axis]
    u = np.linspace(-1.0, 1.0, n)
    f = plant_static(p["coef"], u)
    d = np.gradient(f, u)
    return u.tolist(), f.tolist(), (d < 0).tolist()


# --------------------------------------------------------------------------
# trained models (learn.py) exposed as controllers
# --------------------------------------------------------------------------
_ML_CACHE = {}
SCORES = {}


def _load_scores():
    """Read learn.py's report.json so the selector can show each controller's
    test score. Scored on one shared test split — hand-written and learned
    controllers alike — so the numbers are comparable."""
    p = os.path.join(CKPT_DIR, "report.json")
    if os.path.isfile(p):
        try:
            with open(p) as f:
                SCORES.update(json.load(f))
        except Exception:
            pass
    return SCORES


def _ml_predict(name, t, vx, vy, x0, y0, dt, rng, **kw):
    """Run a trained checkpoint. The network predicts POSITION directly, so
    the equivalent stick command is the derivative of its own output — unlike
    the analytic controllers, it has no separately identifiable input to an
    internal integrator."""
    import torch
    if name not in _ML_CACHE:
        import learn
        blob = torch.load(os.path.join(CKPT_DIR, f"{name}.pt"),
                          map_location="cpu", weights_only=False)
        m = learn.make(name, float(blob.get("dt", dt)))
        m.load_state_dict(blob["state_dict"])
        m.eval()
        _ML_CACHE[name] = m
    m = _ML_CACHE[name]
    with torch.no_grad():
        u = torch.tensor(np.stack([vx, vy], -1)[None], dtype=torch.float32)
        p = m(u)[0].numpy()
    gx = p[:, 0] + (x0 - p[0, 0])          # anchor on the known start
    gy = p[:, 1] + (y0 - p[0, 1])
    ux = np.clip(np.gradient(gx, dt), -JOY_FULL_SCALE, JOY_FULL_SCALE)
    uy = np.clip(np.gradient(gy, dt), -JOY_FULL_SCALE, JOY_FULL_SCALE)
    return gx, gy, ux, uy


EYE_CONTROLLER = {}          # plant variant -> controller trained for it


def _eye_predict(base, variant, t, vx, vy, x0, y0, dt):
    """A network trained through one eye, driving that eye.

    learn_eye.py trains a HORIZONTAL model: one velocity channel in, a
    push-pull muscle command out. Two consequences are handled here rather
    than hidden.

    The command, not a position, is what the network emits — so the
    controller's own trace is the command read through the plant's STATIC
    map, Phi(cmd). The blue trace adds the second-order mechanics on top, and
    the distance between them is exactly what the dynamics cost.

    The network is two-dimensional: two velocity channels in, four motor
    pools out (LR, MR, SR, IR), one push-pull command per axis, each driving
    its own plant. The earlier version was horizontal-only and reused the
    same network for the vertical with a gain-ratio correction, which clipped
    15% of samples and read on screen as an eye lag it never had.
    """
    import torch
    key = ("eye", base)
    if key not in _ML_CACHE:
        import learn_eye
        blob = torch.load(os.path.join(CKPT_DIR, f"{base}.pt"),
                          map_location="cpu", weights_only=False)
        pls, _ = learn_eye.load_plants(variant, float(blob.get("dt", dt)))
        m = learn_eye.CTRNNEye(float(blob.get("dt", dt)), pls)
        m.load_state_dict(blob["state_dict"])
        m.eval()
        _ML_CACHE[key] = m
    m = _ML_CACHE[key]
    P = load_plants()[variant]
    with torch.no_grad():
        u = torch.tensor(np.stack([vx, vy], -1)[None], dtype=torch.float32)
        _, mm = m(u)
    cmd_h = (mm[0, :, 0] - mm[0, :, 1]).numpy()      # LR - MR
    cmd_v = (mm[0, :, 2] - mm[0, :, 3]).numpy()      # SR - IR
    ux = np.clip(cmd_h, -JOY_FULL_SCALE, JOY_FULL_SCALE)
    uy = np.clip(cmd_v, -JOY_FULL_SCALE, JOY_FULL_SCALE)
    # red = the command through the static map only
    dpu = deg_per_unit("auto", variant)
    gx = plant_static(P["h"]["coef"], np.clip(cmd_h, -1, 1)) / dpu
    gy = plant_static(P["v"]["coef"], np.clip(cmd_v, -1, 1)) / dpu
    return gx, gy, ux, uy


def register_trained():
    """One controller per checkpoint in models/, named `ml:<model>`.

    Checkpoints called ``ctrnn_eye_<variant>`` are also indexed by the eye
    they were trained through, because a controller that has learned one
    plant's inverse is not the controller for another. The UI uses that index
    to pair a selected eye with its own network instead of reusing one."""
    if not os.path.isdir(CKPT_DIR):
        return
    for f in sorted(os.listdir(CKPT_DIR)):
        if not f.endswith(".pt"):
            continue
        base = f[:-3]
        if base.startswith("ctrnn_eye_"):
            var = base[len("ctrnn_eye_"):]
            if var in load_plants():
                OPEN_LOOP[f"eye:{var}"] = (
                    lambda t, vx, vy, x0, y0, dt, rng, _v=var, _b=base, **kw:
                    _eye_predict(_b, _v, t, vx, vy, x0, y0, dt))
                PARAMS[f"eye:{var}"] = []
                EYE_CONTROLLER[var] = f"eye:{var}"
            continue
        OPEN_LOOP[f"ml:{base}"] = (
            lambda t, vx, vy, x0, y0, dt, rng, _b=base, **kw:
            _ml_predict(_b, t, vx, vy, x0, y0, dt, rng, **kw))
        PARAMS[f"ml:{base}"] = []


def label_for(name):
    """`gru (0.009)` — the controller and its mean |drift| on the test set.

    Controllers trained through an eye are named after the eye they inverted,
    not after the checkpoint file: `ctrnn_eye_eye_p3a_length` reads as
    `ctrnn C`. The doubled "eye" is an artefact of the filename and carries
    no information."""
    key = name[3:] if name.startswith("ml:") else name
    if name.startswith("eye:"):
        var = name[4:]
        sc = EYE_SCORES.get(var) or {}
        e = sc.get("gaze_err_mean_deg")
        return (f"ctrnn {EYE_LABEL.get(var, var)}"
                + (f" ({e:.2f}\u00b0)" if e is not None else ""))
    if key.startswith("ctrnn_eye_"):
        var = key[len("ctrnn_eye_"):]
        lab = EYE_LABEL.get(var, var)
        sc = EYE_SCORES.get(var) or {}
        e = sc.get("gaze_err_mean_deg")
        return f"ctrnn {lab}" + (f" ({e:.2f}\u00b0)" if e is not None else "")
    s = SCORES.get(key)
    if not s:
        return name
    return f"{key} ({s['err_mean']:.3f})"


def run_trial(name, tr, rng, **kw):
    """Run one controller on one trajectory; return error and survival time."""
    t = np.asarray(tr["t"]); x = np.asarray(tr["x"]); y = np.asarray(tr["y"])
    dt = float(tr["settings"]["dt"])
    vx = np.gradient(x, dt); vy = np.gradient(y, dt)
    gx, gy, ux, uy = OPEN_LOOP[name](t, vx, vy, x[0], y[0], dt, rng, **kw)
    err = np.hypot(x - gx, y - gy)
    lost = np.flatnonzero(err > FOV)
    return {
        "err": err,
        "t_lose": float(t[lost[0]]) if lost.size else np.inf,
        "err_end": float(err[-1]),
        "path_len": float(tr["path_len"]),
    }


def sweep(conditions, n_seeds, duration, dt, base_seed=0):
    """Every controller on every condition, n_seeds trajectories each."""
    out = {}
    for cond in conditions:
        for name in OPEN_LOOP:
            errs, tl = [], []
            for s in range(n_seeds):
                tr = generate(start="center", duration=duration, dt=dt,
                              seed=base_seed + s, **cond)
                r = run_trial(name, tr, np.random.default_rng(9000 + s),
                              **defaults(name))
                errs.append(r["err"]); tl.append(r["t_lose"])
            out[(tuple(cond.items()), name)] = {
                "err": np.array(errs), "t_lose": np.array(tl)}
    return out


def sweep_tau(taus, speeds, n_seeds, duration, dt, base_seed=0):
    """Median survival time against the integrator time constant.

    This is the design curve the oculomotor note wants: how long must tau be
    for the circuit to hold a target of a given speed inside the fovea for a
    given time?
    """
    out = {}
    for sp in speeds:
        med = []
        for tau in taus:
            tl = []
            for s in range(n_seeds):
                tr = generate(start="center", speed=sp, angle="low",
                              shape="curve", duration=duration, dt=dt,
                              seed=base_seed + s)
                r = run_trial("leaky", tr, np.random.default_rng(9000 + s),
                              tau=tau)
                tl.append(min(r["t_lose"], duration))
            med.append(float(np.median(tl)))
        out[sp] = med
    return out


def example_trial(cond, name, duration, dt, seed=0, **kw):
    """One trial kept in full, so the figure can show the actual paths rather
    than only their error statistics."""
    tr = generate(start="center", duration=duration, dt=dt, seed=seed, **cond)
    x = np.asarray(tr["x"]); y = np.asarray(tr["y"])
    vx = np.gradient(x, dt); vy = np.gradient(y, dt)
    gx, gy, _, _ = OPEN_LOOP[name](np.asarray(tr["t"]), vx, vy, x[0], y[0],
                                   dt, np.random.default_rng(9000 + seed), **kw)
    return {"t": np.asarray(tr["t"]), "x": x, "y": y, "gx": gx, "gy": gy,
            "name": name}


def figure(res, conditions, n_seeds, duration, dt, out_path, tau_sweep=None,
           taus=None, example=None):
    names = list(OPEN_LOOP)
    col = {"perfect": "#111111", "gain": "#cf222e", "leaky": "#1f6feb"}
    ref = tuple(conditions[0].items())
    t = np.arange(int(round(duration / dt))) * dt

    fig = plt.figure(figsize=(12.5, 17.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.15, 1.15],
                          hspace=0.34, wspace=0.24)
    ax = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                   [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
    axE = fig.add_subplot(gs[2, 0])          # target path
    axF = fig.add_subplot(gs[3, 0])          # computed path, below it
    axG = fig.add_subplot(gs[2, 1])          # x(t), both
    axH = fig.add_subplot(gs[3, 1])          # y(t), both
    for a in list(ax.ravel()) + [axE, axF, axG, axH]:
        a.spines[["top", "right"]].set_visible(False)

    # (a) error growth, median + IQR, on the reference condition
    for nm in names:
        e = res[(ref, nm)]["err"]
        med = np.median(e, axis=0)
        ax[0, 0].fill_between(t, np.percentile(e, 25, axis=0),
                              np.percentile(e, 75, axis=0),
                              color=col[nm], alpha=0.15, lw=0)
        ax[0, 0].plot(t, med, color=col[nm], lw=1.8, label=nm)
    ax[0, 0].axhline(FOV, color="0.4", ls="--", lw=1)
    ax[0, 0].text(duration * 0.99, FOV * 1.04, "field of view", ha="right",
                  va="bottom", fontsize=8, color="0.4")
    ax[0, 0].set_xlabel("time (s)"); ax[0, 0].set_ylabel("|error| (grid units)")
    ax[0, 0].set_ylim(0, min(2.0, FOV * 3)); ax[0, 0].legend(frameon=False,
                                                             fontsize=9)

    # (b) same, log-log: the growth LAW is the slope
    for nm in names:
        med = np.median(res[(ref, nm)]["err"], axis=0)
        ax[0, 1].loglog(t[1:], np.maximum(med[1:], 1e-6), color=col[nm], lw=1.8)
    for slope, lab, off in ((1.0, "linear (gain)", 0.30),):
        ax[0, 1].loglog(t[1:], off * (t[1:] / t[-1]) ** slope, color="0.6",
                        ls=":", lw=1)
        ax[0, 1].text(t[-1], off, f"  {lab}", fontsize=8, color="0.5",
                      va="center")
    ax[0, 1].axhline(FOV, color="0.4", ls="--", lw=1)
    ax[0, 1].set_xlabel("time (s)"); ax[0, 1].set_ylabel("median |error|")

    # (c) survival: fraction of trials still inside the field of view
    for nm in names:
        tl = res[(ref, nm)]["t_lose"]
        surv = [(tl > tt).mean() for tt in t]
        ax[1, 0].plot(t, surv, color=col[nm], lw=1.8, label=nm)
    ax[1, 0].set_xlabel("time (s)")
    ax[1, 0].set_ylabel("fraction still tracking")
    ax[1, 0].set_ylim(-0.03, 1.03)

    # (d) the design curve: survival vs the integrator time constant
    spcol = {"slow": "#2ea043", "middle": "#1f6feb", "fast": "#cf222e"}
    for sp, med in (tau_sweep or {}).items():
        ax[1, 1].semilogx(taus, med, "o-", color=spcol.get(sp, "0.3"),
                          lw=1.8, ms=4, label=f"{sp} target")
    ax[1, 1].axhline(duration, color="0.4", ls="--", lw=1)
    ax[1, 1].text(taus[-1], duration, "never lost ", ha="right", va="bottom",
                  fontsize=8, color="0.4")
    ax[1, 1].set_xlabel("integrator time constant tau (s)")
    ax[1, 1].set_ylabel("time to leave the field of view (s)")
    ax[1, 1].legend(frameon=False, fontsize=9)

    # (e)-(f) the paths themselves: the target above, and below it the
    # trajectory the controller actually reconstructed by integrating
    # velocity. Same axes on both, so the drift is a visual difference and
    # not a rescaling.
    ex = example
    for a, (px, py), c, lab in (
            (axE, (ex["x"], ex["y"]), "#111111", "target"),
            (axF, (ex["gx"], ex["gy"]), "#cf222e",
             f"computed ({ex['name']}, open loop)")):
        a.plot(px, py, color=c, lw=1.6)
        a.plot(px[0], py[0], "o", color=c, ms=7, mfc="none", mew=1.5)
        a.plot(px[-1], py[-1], "o", color=c, ms=6)
        a.set_xlim(-1.05, 1.05); a.set_ylim(-1.05, 1.05)
        a.set_aspect("equal")
        a.set_xlabel("x"); a.set_ylabel("y")
        a.text(0.02, 0.98, lab, transform=a.transAxes, va="top", ha="left",
               fontsize=10, color=c)
    axE.text(0.98, 0.02, "open circle = start (centre)", transform=axE.transAxes,
             va="bottom", ha="right", fontsize=8, color="0.5")

    # (g)-(h) the same pair against time, one axis each: where the two curves
    # separate is the moment the integral stopped being usable.
    for a, tgt, cmp_, lb in ((axG, ex["x"], ex["gx"], "x"),
                             (axH, ex["y"], ex["gy"], "y")):
        a.plot(ex["t"], tgt, color="#111111", lw=1.5, label="target")
        a.plot(ex["t"], cmp_, color="#cf222e", lw=1.5, label="computed")
        a.set_xlabel("time (s)"); a.set_ylabel(lb)
        a.set_ylim(-1.05, 1.05)
    axG.legend(frameon=False, fontsize=9, ncol=2)

    for a, L in zip(list(ax.ravel()) + [axE, axF, axG, axH], "abcdefgh"):
        a.text(-0.09, 1.04, L, transform=a.transAxes, fontsize=15,
               fontweight="bold", va="bottom", ha="left")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"[fig] wrote {out_path}")


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>open-loop tracking</title>
<style>
  :root { --fg:#fff; --bg:#000; --dim:#fff; --red:#e5484d; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45
         -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
         -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1080px; margin:0 auto; padding:26px 22px 40px; }
  h1 { font-size:15px; font-weight:600; letter-spacing:.14em;
       text-transform:uppercase; margin:0 0 6px; }
  .sub { font-size:12px; color:var(--dim); margin:0 0 22px; }
  .controls { display:flex; flex-wrap:wrap; gap:22px; margin-bottom:18px; }
  .group { display:flex; flex-direction:column; gap:7px; }
  .label { font-size:10px; letter-spacing:.16em; text-transform:uppercase;
           color:var(--dim); }
  .seg { display:flex; }
  .seg button { background:var(--bg); color:var(--fg); border:1px solid var(--fg);
                border-right-width:0; padding:6px 13px; font:inherit;
                font-size:12px; cursor:pointer; }
  .seg button:last-child { border-right-width:1px; }
  .seg button[aria-pressed="true"] { background:var(--fg); color:var(--bg); }
  .knobs { display:flex; flex-wrap:wrap; gap:26px; margin:0 0 22px;
           padding:14px 16px; border:1px solid #333; }
  .knob { display:flex; flex-direction:column; gap:5px; min-width:260px; }
  .knob .kl { font-size:11px; color:var(--dim); display:flex;
              justify-content:space-between; gap:12px; }
  .knob .kl b { color:var(--fg); font-weight:600;
                font-variant-numeric:tabular-nums; }
  .knob .ends { display:flex; justify-content:space-between; font-size:9px;
                color:var(--fg); font-variant-numeric:tabular-nums; }
  input[type=range] { -webkit-appearance:none; appearance:none; width:100%;
                      height:1px; background:var(--fg); outline:none; margin:6px 0; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance:none;
    appearance:none; width:13px; height:13px; background:var(--fg);
    border:1px solid var(--fg); cursor:pointer; border-radius:0; }
  input[type=range]::-moz-range-thumb { width:13px; height:13px;
    background:var(--fg); border:1px solid var(--fg); cursor:pointer;
    border-radius:0; }
  .row { display:flex; gap:22px; align-items:flex-start; flex-wrap:wrap; }
  .panel { display:flex; flex-direction:column; gap:8px; }
  canvas { display:block; background:var(--bg); border:1px solid var(--fg); }
  .cap { font-size:10px; letter-spacing:.16em; text-transform:uppercase;
         color:var(--dim); }
  .cap i { color:var(--red); font-style:normal; }
  .stats { font-size:12px; color:var(--dim); margin-top:16px;
           font-variant-numeric:tabular-nums; }
  .stats b { color:var(--fg); font-weight:600; }
  .stats .lost { color:var(--red); }
</style></head><body><div class="wrap">
<h1>open loop &mdash; velocity only</h1>
<p class="sub">The controller is given the target's velocity and its starting
position at the centre. It never sees where the target actually is, so it has
to integrate &mdash; and nothing tells it when the integral has gone wrong.
The number after each controller is its mean |drift| on the shared test set:
lower is better, and it is the same measurement for the hand-written and the
learned ones.</p>
<div class="controls" id="controls"></div>
<div class="knobs" id="knobs"></div>
<div class="row">
  <div class="panel"><canvas id="world" width="620" height="620"></canvas>
    <div class="cap" id="worldcap">world &mdash; target, and <i>computed</i></div></div>
  <div class="panel"><canvas id="eye" width="360" height="360"></canvas>
    <div class="cap" id="eyecap">the eye, seen from in front</div></div>
</div>
<div class="row" style="margin-top:20px">
  <div class="panel"><canvas id="states" width="620" height="250"></canvas>
    <div class="cap">signal path &mdash; input, circuit, motor pools, eye state</div></div>
</div>
<div class="stats" id="stats"></div>
</div><script>
const SPEC=__SPEC__, CTRL_A=__CTRL_A__, CTRL_M=__CTRL_M__;
const PARAMS=__PARAMS__, FOV=__FOV__;
const JOYGAIN=__JOYGAIN__, LABELS=__LABELS__;
// Short trials on a slow target by default: with a leaky integrator the
// drift is the thing to watch, and a slow target is both easier to follow by
// eye and — because the leak is a high-pass — lost sooner.
const DURATIONS=["4","8","16","30"];
const WORLDS=["auto","3","7","15"];
const PLANTS=__PLANTS__, EYECTRL=__EYECTRL__, PLANTLAB=__PLANTLAB__;
const sel={start:"center",shape:"curve",motion:"stop_and_go",speed:"slow",
           angle:"low",controller:"eye:eye_p3a_length",duration:"8",
           plant:"eye_p3a_length",world:"auto"};
// opened on eye C, so the controller must be the one trained for it
const knob={}; let TR=null,k=0,timer=null,pending=null;

const C=document.getElementById("controls"),K=document.getElementById("knobs");
// Declared before ANY group() call: group() pushes into them, and a `const`
// referenced before its declaration is a ReferenceError, not undefined — the
// whole script dies and every row after the failure silently never renders.
const CTRLBTN=[], PLANTBTN=[];
function group(name,opts,onpick,key){
  key=key||name;
  const g=document.createElement("div"); g.className="group";
  const l=document.createElement("div"); l.className="label"; l.textContent=name;
  const s=document.createElement("div"); s.className="seg";
  opts.forEach(o=>{
    const b=document.createElement("button");
    b.textContent=(name==="world" ? (o==="auto"?"auto":o+"\u00b0/unit")
                  : name==="duration" ? o+" s"
                  : key==="controller" ? (LABELS[o]||o)
                  : name==="plant" ? (o==="none" ? "none" : (PLANTLAB[o]||o))
                  : o.replace(/_/g," "));
    b.setAttribute("aria-pressed", sel[key]===o);
    b.onclick=()=>{ sel[key]=o;
      // Controller selection is shared across the analytic and learned
      // groups, so clear both rather than only this one.
      const pool=(key==="controller")?CTRLBTN:[...s.children];
      pool.forEach(c=>c.setAttribute("aria-pressed",c===b));
      // choosing an eye-blind controller drops the eye
      if(key==="controller" && !o.startsWith("eye:") && sel.plant!=="none")
        clearPlant();
      if(onpick) onpick(o); load(); };
    if(key==="controller") CTRLBTN.push(b);
    if(name==="plant") PLANTBTN.push(b);
    s.appendChild(b);
  });
  g.append(l,s); C.appendChild(g); return s;
}
Object.entries(SPEC).forEach(([n,v])=>group(n,v));
group("duration",DURATIONS);
group("world",WORLDS);
// The two rows are ONE exclusive choice, not a combination. An eye button
// selects that eye together with the network trained through it; a
// controller button clears the eye, because none of those controllers has
// ever seen one. Allowing the cross product produced the misleading pairing
// — an eye-blind controller driving an eye — that read as a regression.
group("plant",["none"].concat(PLANTS), p=>{
  const c = EYECTRL[p];
  if(p!=="none" && c){ sel.controller=c; markCtrl(c); }
  if(p==="none" && (sel.controller||"").startsWith("eye:")){
    sel.controller="perfect"; markCtrl("perfect"); }
});
function markCtrl(n){
  CTRLBTN.forEach(b=>b.setAttribute("aria-pressed",(LABELS[n]||n)===b.textContent));
}
function clearPlant(){
  sel.plant="none";
  PLANTBTN.forEach(b=>b.setAttribute("aria-pressed", b.textContent==="none"));
}
// Two groups, one shared selection: the analytic controllers are lesion
// models with hand-set defects, the learned ones are fitted. Mixing them in
// one strip invited reading the analytic rows as competitors, which they are
// not — fitted, gain and leak collapse onto `perfect`.
group("analytic",CTRL_A,buildKnobs,"controller");
group("learned",CTRL_M,buildKnobs,"controller");
const gx0=document.createElement("div"); gx0.className="group";
gx0.innerHTML='<div class="label">&nbsp;</div>';
const sx0=document.createElement("div"); sx0.className="seg";
const nb=document.createElement("button"); nb.textContent="new seed";
nb.onclick=()=>load(true); sx0.appendChild(nb); gx0.appendChild(sx0);
C.appendChild(gx0);

function buildKnobs(name){
  K.innerHTML=""; const ps=PARAMS[name]||[];
  if(!ps.length){ K.innerHTML='<div class="label">no parameters &mdash; exact integration</div>';
                  return; }
  ps.forEach(p=>{
    if(knob[p.name]===undefined) knob[p.name]=p.default;
    const d=document.createElement("div"); d.className="knob";
    const lab=document.createElement("div"); lab.className="kl";
    const val=document.createElement("b");
    val.textContent=(+knob[p.name]).toFixed(3);
    const nm=document.createElement("span"); nm.textContent=p.label;
    lab.append(nm,val);
    const r=document.createElement("input"); r.type="range";
    r.min=p.min; r.max=p.max; r.step=p.step; r.value=knob[p.name];
    const ends=document.createElement("div"); ends.className="ends";
    ends.innerHTML=`<span>${p.min}</span><span>${p.max}</span>`;
    r.oninput=()=>{ knob[p.name]=+r.value; val.textContent=(+r.value).toFixed(3);
                    clearTimeout(pending); pending=setTimeout(()=>load(),110); };
    d.append(lab,r,ends); K.appendChild(d);
  });
}
buildKnobs(sel.controller);

const W=document.getElementById("world").getContext("2d");
const EY=document.getElementById("eye").getContext("2d");
const ST=document.getElementById("states").getContext("2d");
const TRAIL=110;
function toPx(v,size){ return (v+1)/2*(size-2)+1; }
// With an eye attached the world panel is drawn in DEGREES over a fixed
// window, not in arena units. That is the difference the world selector was
// missing: rescaling the world then shrinks the trajectory inside a box of
// constant size, instead of rescaling the box with it and cancelling out.
const DEGSPAN=18;                       // half-window, degrees
function degPx(v,size){ return (v/DEGSPAN+1)/2*(size-2)+1; }
function wPx(v,size){ return TR.deg_per_unit
  ? degPx(v*TR.deg_per_unit,size) : toPx(v,size); }

function drawWorld(){
  const n=620; W.fillStyle="#000"; W.fillRect(0,0,n,n);
  W.strokeStyle="#1c1c1c"; W.lineWidth=1;
  for(let i=1;i<4;i++){ const p=Math.round(i*n/4)+.5;
    W.beginPath(); W.moveTo(p,0); W.lineTo(p,n); W.moveTo(0,p); W.lineTo(n,p); W.stroke(); }
  if(TR.deg_per_unit){
    const d=TR.deg_per_unit;
    // ruler in degrees, fixed window
    W.fillStyle="#7a7a7a"; W.font="9px sans-serif"; W.textAlign="center";
    for(let g=-15;g<=15;g+=5){
      const px=degPx(g,n);
      W.fillText(g+"\u00b0", px, n-4);
      W.strokeStyle="#2a2a2a"; W.beginPath();
      W.moveTo(px,n-14); W.lineTo(px,n-10); W.stroke();
    }
    // the world the target lives in, at the current scale
    const wx=degPx(0.95*d,n)-degPx(0,n);
    W.strokeStyle="#4b5563"; W.lineWidth=1.2;
    W.strokeRect(degPx(0,n)-wx, degPx(0,n)-wx, 2*wx, 2*wx);
    W.textAlign="left"; W.fillStyle="#6b7280";
    W.fillText("world "+d.toFixed(1)+"\u00b0/unit  (\u00b1"
               +(0.95*d).toFixed(1)+"\u00b0)", 5, 12);
    // what the eye can reach — when the target leaves this, no controller
    // can follow it and the failure is mechanical, not one of control
    if(TR.reach_h){
      W.strokeStyle="#8a6d1f"; W.lineWidth=1.2; W.setLineDash([5,4]);
      W.strokeRect(degPx(-TR.reach_h,n), degPx(-TR.reach_v,n),
                   degPx(TR.reach_h,n)-degPx(-TR.reach_h,n),
                   degPx(TR.reach_v,n)-degPx(-TR.reach_v,n));
      W.setLineDash([]); W.fillStyle="#8a6d1f";
      W.fillText("eye reach \u00b1"+TR.reach_h.toFixed(1)+"\u00b0",
                 degPx(-TR.reach_h,n)+4, degPx(-TR.reach_v,n)+11);
    }
  }
  W.lineJoin="round"; W.lineCap="round";
  const track=(ax,ay,dim,bright,lwd,lwb)=>{
    W.strokeStyle=dim; W.lineWidth=lwd; W.beginPath();
    for(let i=0;i<=k;i++){ const px=wPx(ax[i],n),py=wPx(-ay[i],n);
      i?W.lineTo(px,py):W.moveTo(px,py); } W.stroke();
    W.strokeStyle=bright; W.lineWidth=lwb; W.beginPath();
    const s0=Math.max(0,k-TRAIL);
    for(let i=s0;i<=k;i++){ const px=wPx(ax[i],n),py=wPx(-ay[i],n);
      i===s0?W.moveTo(px,py):W.lineTo(px,py); } W.stroke();
  };
  track(TR.x,TR.y,"#5a5a5a","#d8d8d8",2,3.5);
  track(TR.gx,TR.gy,"#7a2226","#e5484d",2,3.5);
  if(TR.gaze_grid_h) track(TR.gaze_grid_h,TR.gaze_grid_v,"#1d3f5e","#4da3ff",1.5,2.5);
  // the two current positions, and the drift between them
  const tx=wPx(TR.x[k],n),ty=wPx(-TR.y[k],n);
  const cx=wPx(TR.gx[k],n),cy=wPx(-TR.gy[k],n);
  W.strokeStyle="#666"; W.lineWidth=1; W.setLineDash([3,3]);
  W.beginPath(); W.moveTo(tx,ty); W.lineTo(cx,cy); W.stroke(); W.setLineDash([]);
  W.fillStyle="#fff"; W.beginPath(); W.arc(tx,ty,7,0,7); W.fill();
  W.fillStyle="#e5484d"; W.beginPath(); W.arc(cx,cy,7,0,7); W.fill();
  if(TR.gaze_grid_h){
    const ex_=wPx(TR.gaze_grid_h[k],n), ey_=wPx(-TR.gaze_grid_v[k],n);
    W.fillStyle="#4da3ff"; W.beginPath(); W.arc(ex_,ey_,6,0,7); W.fill();
  }
  W.strokeStyle="#e5484d"; W.lineWidth=1.2;
  W.beginPath(); W.arc(cx,cy,15,0,7); W.stroke();
}

function drawEye(){
  const n=360,c=n/2,R=118;
  EY.fillStyle="#000"; EY.fillRect(0,0,n,n);
  if(!TR.plant){ EY.fillStyle="#666"; EY.font="12px sans-serif";
    EY.fillText("no eye selected",14,24); return; }
  // Gaze in degrees maps to a displacement of the pupil across the face of
  // the globe: the eye rotates, so the pupil sweeps a circle of radius R and
  // its projection is R*sin(angle). Drawn at the eye's OWN scale, not the
  // world's, so the panel shows what the mechanics did rather than how close
  // that was to the target.
  const gh=TR.gaze_deg[k], gv=(TR.gaze_grid_v[k]||0)*TR.deg_per_unit;
  const px=c+Math.sin(gh*Math.PI/180)*R, py=c-Math.sin(gv*Math.PI/180)*R;
  // sclera
  EY.fillStyle="#eef2f6"; EY.strokeStyle="#fff"; EY.lineWidth=1.5;
  EY.beginPath(); EY.arc(c,c,R,0,7); EY.fill(); EY.stroke();
  // where the target would put it, for comparison
  const th=TR.target_deg[k], tv=TR.y[k]*TR.deg_per_unit;
  const tx=c+Math.sin(th*Math.PI/180)*R, ty=c-Math.sin(tv*Math.PI/180)*R;
  EY.strokeStyle="#9aa4b2"; EY.lineWidth=1.5; EY.setLineDash([3,3]);
  EY.beginPath(); EY.arc(tx,ty,26,0,7); EY.stroke(); EY.setLineDash([]);
  // iris + the black pupil
  EY.fillStyle="#7c93ad"; EY.beginPath(); EY.arc(px,py,40,0,7); EY.fill();
  EY.fillStyle="#000"; EY.beginPath(); EY.arc(px,py,19,0,7); EY.fill();
  EY.fillStyle="#ffffff"; EY.globalAlpha=0.55;
  EY.beginPath(); EY.arc(px-7,py-8,5,0,7); EY.fill(); EY.globalAlpha=1;
  // axes, so left/right and up/down are readable
  EY.strokeStyle="#333"; EY.lineWidth=1;
  EY.beginPath(); EY.moveTo(c-R,c); EY.lineTo(c+R,c);
  EY.moveTo(c,c-R); EY.lineTo(c,c+R); EY.stroke();
  EY.fillStyle="#8a8a8a"; EY.font="11px sans-serif";
  EY.fillText("h "+gh.toFixed(1)+"\u00b0",10,20);
  EY.fillText("v "+gv.toFixed(1)+"\u00b0",10,36);
  EY.fillStyle="#9aa4b2"; EY.fillText("dashed = where the target is",10,n-12);
}

// Pixel-art state monitor. Every cell is one scalar at this instant, drawn
// as a hard-edged block: blue for negative, warm for positive, black at
// zero. Deliberately unsmoothed — the point is to read individual units, not
// a surface.
function cmap(v,lo,hi){
  const t=Math.max(-1,Math.min(1,(v-(lo+hi)/2)/((hi-lo)/2||1)));
  if(t>=0) return `rgb(${Math.round(20+235*t)},${Math.round(20+150*t)},${Math.round(30+20*t)})`;
  return `rgb(${Math.round(20-10*t)},${Math.round(20+90*-t)},${Math.round(30+225*-t)})`;
}
function cells(x0,y0,vals,cols,px,lo,hi,gap){
  gap=gap||2;
  vals.forEach((v,i)=>{
    const cx=x0+(i%cols)*(px+gap), cy=y0+Math.floor(i/cols)*(px+gap);
    ST.fillStyle=cmap(v,lo,hi); ST.fillRect(cx,cy,px,px);
  });
}
function lab(x,y,t,col){ ST.fillStyle=col||"#8a8a8a"; ST.font="10px monospace";
  ST.fillText(t,x,y); }

function drawStates(){
  const W=620,H=250; ST.fillStyle="#000"; ST.fillRect(0,0,W,H);
  if(!TR.rates){ lab(14,24,"select an eye to see the circuit state","#666"); return; }
  const r=TR.rates[k], p=TR.pools[k];
  const vx=(TR.x[k]-(TR.x[k-1]!==undefined?TR.x[k-1]:TR.x[k]))/TR.settings.dt;
  const vy=(TR.y[k]-(TR.y[k-1]!==undefined?TR.y[k-1]:TR.y[k]))/TR.settings.dt;

  // 1. input: two cells
  lab(14,20,"INPUT  v"); cells(14,28,[vx,vy],1,22,-1,1);
  lab(14,90,"vx"); lab(14,102,"vy");

  // 2. the recurrent core: 64 units as an 8x8 block of rates
  lab(96,20,"ctRNN  tanh(h)   64 units");
  cells(96,28,r,8,14,-0.6,0.6);
  lab(96,158,"each cell one neuron's rate");

  // 3. motor pools: four cells, non-negative
  lab(300,20,"MOTOR POOLS  >= 0");
  cells(300,28,p,1,26,0,1.2);
  ["LR","MR","SR","IR"].forEach((n,i)=>lab(332,46+i*28,n+"  "+p[i].toFixed(2),"#c9c9c9"));

  // 4. the commands and the eye's two states per axis
  lab(430,20,"EYE  state");
  const gh=TR.gaze_deg[k], gv=(TR.gaze_grid_v[k]||0)*TR.deg_per_unit;
  const rows=[["cmd h",TR.u_cmd[k],-1,1],["cmd v",TR.jy?TR.jy[k]:0,-1,1],
              ["gaze h",gh/15,-1,1],["gaze v",gv/15,-1,1],
              ["vel h",TR.vel_h[k]/20,-1,1],["vel v",TR.vel_v[k]/20,-1,1]];
  rows.forEach((rw,i)=>{
    cells(430,28+i*28,[rw[1]],1,22,rw[2],rw[3]);
    lab(458,45+i*28,rw[0],"#c9c9c9");
  });
  const vals=[TR.u_cmd[k],0,gh,gv,TR.vel_h[k],TR.vel_v[k]];
  [0,2,3,4,5].forEach(i=>lab(516,45+i*28,
    (i>=4? vals[i].toFixed(1)+"°/s" : i>=2? vals[i].toFixed(1)+"°" : vals[i].toFixed(2)),"#7a7a7a"));

  // flow arrows
  ST.strokeStyle="#444"; ST.lineWidth=1;
  [[62,60,92,60],[236,60,296,60],[352,60,426,60]].forEach(a=>{
    ST.beginPath(); ST.moveTo(a[0],a[1]); ST.lineTo(a[2],a[3]); ST.stroke();
    ST.beginPath(); ST.moveTo(a[2],a[3]); ST.lineTo(a[2]-5,a[3]-4);
    ST.lineTo(a[2]-5,a[3]+4); ST.closePath(); ST.fillStyle="#444"; ST.fill();
  });
  lab(14,H-10,"blue negative   black zero   warm positive","#555");
}

function stats(){
  const inside=TR.err.filter(e=>e<=FOV).length/TR.err.length*100;
  // With an eye attached the red trace is where the EYE points, not what the
  // controller computed. Every controller in these rows was trained without
  // an eye, so it has no inverse model of one and pays a lag it cannot see.
  // Saying so stops that reading as a controller regression.
  const uncomp = TR.plant
    ? '<br><span style="color:#e5a23c">red is the controller, blue is the eye '
      + 'driven by it. This controller was trained without an eye, so the '
      + 'blue lag is the plant, not a control failure. The number on the eye '
      + 'button is what a controller trained THROUGH that eye achieves.</span>'
    : "";
  const lost = TR.t_lose===null
    ? 'never lost within '+TR.settings.duration.toFixed(0)+'s'
    : '<span class="lost">lost after <b>'+TR.t_lose.toFixed(1)+' s</b></span>';
  document.getElementById("stats").innerHTML=
    lost+` &nbsp;&middot;&nbsp; mean |drift| <b>${TR.err_mean.toFixed(3)}</b>`+
    ` &nbsp; final <b>${TR.err_end.toFixed(3)}</b>`+
    ` &nbsp;&middot;&nbsp; inside the field of view <b>${inside.toFixed(1)}%</b>`+
    ` of the time &nbsp;&middot;&nbsp; seed <b>${TR.settings.seed}</b>`;
}

function frame(){ drawWorld(); drawEye(); drawStates(); stats();
  document.getElementById("worldcap").innerHTML = TR.plant
    ? 'world &mdash; target, <i>computed</i>, and <b style="color:#4da3ff">where the eye points</b>'
    : 'world &mdash; target, and <i>computed</i>';
  k=(k+1)%TR.t.length; }

async function load(newseed){
  const q=new URLSearchParams(sel);
  Object.entries(knob).forEach(([n,v])=>q.set(n,v));
  if(!newseed && TR) q.set("seed", TR.settings.seed);
  const keep=(!newseed&&TR)?k:0;
  const r=await (await fetch("/api/trace?"+q)).json();
  if(r.error){ document.getElementById("stats").textContent=r.error; return; }
  TR=r; k=Math.min(keep,TR.t.length-1);
  if(timer) clearInterval(timer);
  timer=setInterval(frame, TR.settings.dt*1000);
}
load(true);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            from trajectory import SPEC
            page = (PAGE.replace("__SPEC__", json.dumps(SPEC))
                        .replace("__CTRL_A__", json.dumps(
                            [n for n in OPEN_LOOP
                             if not n.startswith(("ml:", "eye:"))]))
                        .replace("__CTRL_M__", json.dumps(
                            # per-eye controllers are NOT listed here: the
                            # plant row already selects the eye and its own
                            # network together, so a second set of A-E buttons
                            # would be the same choice twice.
                            [n for n in OPEN_LOOP if n.startswith("ml:")]
                            + [n for n in OPEN_LOOP if n.startswith("eye:")]))
                        .replace("__LABELS__", json.dumps(
                            {n: label_for(n) for n in OPEN_LOOP}))
                        .replace("__PARAMS__", json.dumps(PARAMS))
                        .replace("__FOV__", str(FOV))
                        .replace("__PLANTS__", json.dumps(
                            sorted(load_plants(),
                                   key=lambda v: EYE_LABEL.get(v, v))))
                        .replace("__PLANTLAB__", json.dumps(
                            {v: eye_label(v) for v in load_plants()}))
                        .replace("__EYECTRL__", json.dumps(EYE_CONTROLLER))
                        .replace("__JOYGAIN__", str(JOY_VIEW_GAIN)))
            return self._send(page, "text/html; charset=utf-8")
        if u.path == "/api/trace":
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                tr = generate(
                    start=q.get("start", "center"),
                    shape=q.get("shape", "curve"),
                    motion=q.get("motion", "continue"),
                    speed=q.get("speed", "middle"),
                    angle=q.get("angle", "low"),
                    duration=float(q.get("duration", 8.0)),
                    seed=int(q["seed"]) if q.get("seed") else None,
                )
                name = q.get("controller", "leaky")
                kn = {p["name"]: float(q[p["name"]])
                      for p in PARAMS.get(name, []) if p["name"] in q}
                x = np.asarray(tr["x"]); y = np.asarray(tr["y"])
                dt = float(tr["settings"]["dt"])
                vx = np.gradient(x, dt); vy = np.gradient(y, dt)
                gx, gy, ux, uy = OPEN_LOOP[name](
                    np.asarray(tr["t"]), vx, vy, x[0], y[0], dt,
                    np.random.default_rng(tr["settings"]["seed"]), **kn)
                jx = np.clip(ux / JOY_FULL_SCALE, -1.0, 1.0)
                jy = np.clip(uy / JOY_FULL_SCALE, -1.0, 1.0)
                ex, ey = x - gx, y - gy
                err = np.hypot(ex, ey)
                lost = np.flatnonzero(err > FOV)
                tr.update({
                    "gx": gx.tolist(), "gy": gy.tolist(),
                    "ex": ex.tolist(), "ey": ey.tolist(), "err": err.tolist(),
                    "err_mean": float(err.mean()),
                    "err_end": float(err[-1]),
                    "t_lose": (float(tr["t"][int(lost[0])])
                               if lost.size else None),
                    "jx": jx.tolist(), "jy": jy.tolist(),
                    "joy_sat": float(np.mean((np.abs(jx) >= 1.0)
                                             | (np.abs(jy) >= 1.0))),
                    "joy_full_scale": JOY_FULL_SCALE,
                    "controller": name, "params": kn,
                })
                pname = q.get("plant", "none")
                if pname != "none" and pname in load_plants():
                    # Phi maps command -> POSITION, so the muscle command is
                    # the controller's position estimate, not its velocity
                    # stick. gx is in grid units and the command axis is
                    # [-1, 1], so the two coincide once gx is clipped: full
                    # deflection asks the eye for its full travel.
                    _dt = float(tr["settings"]["dt"])
                    # The MUSCLE COMMAND is what the controller returns in
                    # ux/uy. gx is a position — for an eye-trained network it
                    # is already the command passed through Phi — so feeding
                    # gx back into the plant applies Phi TWICE and the eye
                    # overshoots by the gain, visibly and from the first
                    # movement rather than after a stop. Measured on eye C:
                    # 1.049 deg that way against 0.048 with the real command.
                    u_cmd = np.clip(ux, -1.0, 1.0)
                    v_cmd = np.clip(uy, -1.0, 1.0)
                    gaze_h, vel_h = plant_run_states(pname, u_cmd, _dt, "h")
                    gaze_v, vel_v = plant_run_states(pname, v_cmd, _dt, "v")
                    st = None
                    if name.startswith("eye:"):
                        try:
                            R, pools = eye_states(
                                "ctrnn_eye_" + name[4:], name[4:],
                                np.gradient(x, _dt), np.gradient(y, _dt), _dt)
                            st = {"rates": np.round(R, 3).tolist(),
                                  "pools": np.round(pools, 3).tolist()}
                        except Exception as e:
                            print(f"[states] {type(e).__name__}: {e}")
                    dpu = deg_per_unit(q.get("world", "auto"), pname)
                    tgt = np.asarray(tr["x"]) * dpu
                    pu, pf, pneg = plant_curve(pname, axis="h")
                    # The world / retina / drift / x(t) / y(t) panels keep
                    # showing WHAT THE CONTROLLER COMPUTED. None of the
                    # controllers selectable here was trained through an eye,
                    # so routing their output through one and calling the
                    # result "the controller" confuses a plant lag with a
                    # control failure — which is exactly how it read. The
                    # eye's response lives in the plant row instead, where it
                    # is labelled as the eye.
                    gaze_grid_h = gaze_h / dpu
                    gaze_grid_v = gaze_v / dpu
                    tr.update({
                        "gaze_grid_h": gaze_grid_h.tolist(),
                        "gaze_grid_v": gaze_grid_v.tolist(),
                        "plant": pname, "gaze_deg": gaze_h.tolist(),
                        "target_deg": tgt.tolist(),
                        "gaze_err": np.abs(gaze_h - tgt).tolist(),
                        "gaze_err_mean": float(np.abs(gaze_h - tgt).mean()),
                        "phi_u": pu, "phi_f": pf, "phi_neg": pneg,
                        "phi_neg_frac": float(np.mean(pneg)),
                        "u_cmd": u_cmd.tolist(),
                        "deg_per_unit": dpu,
                        "vel_h": vel_h.tolist(), "vel_v": vel_v.tolist(),
                        "rates": (st or {}).get("rates"),
                        "pools": (st or {}).get("pools"),
                        "reach_h": float(min(abs(plant_static(
                            load_plants()[pname]["h"]["coef"], -1.0)),
                            abs(plant_static(
                                load_plants()[pname]["h"]["coef"], 1.0)))),
                        "reach_v": float(min(abs(plant_static(
                            load_plants()[pname]["v"]["coef"], -1.0)),
                            abs(plant_static(
                                load_plants()[pname]["v"]["coef"], 1.0)))),
                    })
            except Exception as e:
                return self._send(json.dumps({"error": f"{type(e).__name__}: {e}"}),
                                  "application/json")
            return self._send(json.dumps(tr), "application/json")
        self.send_error(404)


def serve(host, port):
    register_trained()
    _load_scores()
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"open-loop tracking -> http://localhost:{port}   (ctrl-c to stop)")
    print("  controllers: " + ", ".join(label_for(n) for n in OPEN_LOOP))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


def main():
    global JOY_VIEW_GAIN
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", action="store_true",
                   help="run the batch sweep and write the figure instead of "
                        "serving the GUI")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--joy-view-gain", type=float, default=JOY_VIEW_GAIN,
                   help="display magnification of the stick deflection")
    p.add_argument("--n-seeds", type=int, default=24)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "openloop.png"))
    a = p.parse_args()
    JOY_VIEW_GAIN = a.joy_view_gain
    if not a.sweep:
        return serve(a.host, a.port)

    conditions = [
        {"speed": "slow",   "angle": "low",   "shape": "curve"},
        {"speed": "middle", "angle": "low",   "shape": "curve"},
        {"speed": "fast",   "angle": "low",   "shape": "curve"},
        {"speed": "middle", "angle": "sharp", "shape": "curve"},
        {"speed": "fast",   "angle": "sharp", "shape": "curve"},
    ]
    print(f"open-loop sweep: {len(OPEN_LOOP)} controllers x "
          f"{len(conditions)} conditions x {a.n_seeds} seeds, "
          f"{a.duration:.0f}s trials, all starting at the centre")
    res = sweep(conditions, a.n_seeds, a.duration, a.dt)
    taus = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    tsw = sweep_tau(taus, ["slow", "middle", "fast"], a.n_seeds, a.duration,
                    a.dt)

    hdr = f"{'condition':26s}" + "".join(f"{n:>12s}" for n in OPEN_LOOP)
    print("\nmedian time to leave the field of view (s), inf = never\n" + hdr)
    print("-" * len(hdr))
    for c in conditions:
        row = f"{' '.join(c.values()):26s}"
        for nm in OPEN_LOOP:
            tl = res[(tuple(c.items()), nm)]["t_lose"]
            m = np.median(tl)
            row += f"{'never' if np.isinf(m) else f'{m:.1f}':>12s}"
        print(row)
    print("\nparameters:", {k: defaults(k) for k in OPEN_LOOP if PARAMS[k]})
    print("\nmedian survival (s) against integrator tau, low-angle curve")
    print(f"{'tau (s)':>9s}" + "".join(f"{t:>8g}" for t in taus))
    for sp, med in tsw.items():
        print(f"{sp:>9s}" + "".join(
            f"{'never' if m >= a.duration else f'{m:.1f}':>8s}" for m in med))
    ex = example_trial(conditions[1], "leaky", a.duration, a.dt, seed=0,
                       **defaults("leaky"))
    figure(res, conditions, a.n_seeds, a.duration, a.dt, a.out,
           tau_sweep=tsw, taus=taus, example=ex)


if __name__ == "__main__":
    main()
