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
    c = p["h"]["coef"]
    f = lambda u: sum(ci * u ** (i + 1) for i, ci in enumerate(c))
    reach = min(abs(f(-1.0)), abs(f(+1.0)))       # the smaller half-travel
    return float(reach / BOUND)
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
        OPEN_LOOP[f"ml:{base}"] = (
            lambda t, vx, vy, x0, y0, dt, rng, _b=base, **kw:
            _ml_predict(_b, t, vx, vy, x0, y0, dt, rng, **kw))
        PARAMS[f"ml:{base}"] = []
        if base.startswith("ctrnn_eye_"):
            EYE_CONTROLLER[base[len("ctrnn_eye_"):]] = f"ml:{base}"


def label_for(name):
    """`gru (0.009)` — the controller and its mean |drift| on the test set.

    Controllers trained through an eye are named after the eye they inverted,
    not after the checkpoint file: `ctrnn_eye_eye_p3a_length` reads as
    `ctrnn C`. The doubled "eye" is an artefact of the filename and carries
    no information."""
    key = name[3:] if name.startswith("ml:") else name
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
  <div class="panel"><canvas id="world" width="430" height="430"></canvas>
    <div class="cap" id="worldcap">world &mdash; target, and <i>computed</i></div></div>
  <div class="panel"><canvas id="retina" width="286" height="286"></canvas>
    <div class="cap">where the target is, if you believe the integral</div>
    <canvas id="strip" width="286" height="86"></canvas>
    <div class="cap">|drift| over time</div></div>
  <div class="panel"><canvas id="joy" width="240" height="240"></canvas>
    <div class="cap">joystick &mdash; command into the integrator (&times;__JOYGAIN__)</div></div>
</div>
<div class="row" style="margin-top:22px">
  <div class="panel"><canvas id="xy" width="470" height="190"></canvas>
    <div class="cap">x(t) &mdash; target vs <i>computed</i></div></div>
  <div class="panel"><canvas id="yt" width="470" height="190"></canvas>
    <div class="cap">y(t) &mdash; target vs <i>computed</i></div></div>
</div>
<div class="row" id="plantrow" style="margin-top:22px; display:none">
  <div class="panel"><canvas id="phi" width="300" height="250"></canvas>
    <div class="cap">&Phi; &mdash; command to gaze (red = slope &lt; 0)</div></div>
  <div class="panel"><canvas id="eye" width="250" height="250"></canvas>
    <div class="cap">the eye</div></div>
  <div class="panel"><canvas id="gz" width="600" height="250"></canvas>
    <div class="cap">gaze vs target eccentricity (deg)</div></div>
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
const sel={start:"center",shape:"curve",motion:"continue",speed:"slow",
           angle:"low",controller:"leaky",duration:"8",plant:"none",
           world:"auto"};
const knob={}; let TR=null,k=0,timer=null,pending=null;

const C=document.getElementById("controls"),K=document.getElementById("knobs");
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
      if(onpick) onpick(o); load(); };
    if(key==="controller") CTRLBTN.push(b);
    s.appendChild(b);
  });
  g.append(l,s); C.appendChild(g); return s;
}
Object.entries(SPEC).forEach(([n,v])=>group(n,v));
group("duration",DURATIONS);
group("world",WORLDS);
group("plant",["none"].concat(PLANTS), p=>{
  // A controller trained through one eye has learned THAT eye's inverse.
  // Selecting an eye therefore selects its own network, when one exists.
  const c=EYECTRL[p];
  if(c && OPENLOOP_HAS(c)){ sel.controller=c;
    CTRLBTN.forEach(b=>b.setAttribute("aria-pressed",
      (LABELS[c]||c)===b.textContent));
    buildKnobs(c); }
});
function OPENLOOP_HAS(n){ return CTRLBTN.some(b=>(LABELS[n]||n)===b.textContent); }
// Two groups, one shared selection: the analytic controllers are lesion
// models with hand-set defects, the learned ones are fitted. Mixing them in
// one strip invited reading the analytic rows as competitors, which they are
// not — fitted, gain and leak collapse onto `perfect`.
const CTRLBTN=[];
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
const R=document.getElementById("retina").getContext("2d");
const S=document.getElementById("strip").getContext("2d");
const X=document.getElementById("xy").getContext("2d");
const Y=document.getElementById("yt").getContext("2d");
const J=document.getElementById("joy").getContext("2d");
const PH=document.getElementById("phi").getContext("2d");
const EY=document.getElementById("eye").getContext("2d");
const GZ=document.getElementById("gz").getContext("2d");
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
  const n=430; W.fillStyle="#000"; W.fillRect(0,0,n,n);
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
  // the two current positions, and the drift between them
  const tx=wPx(TR.x[k],n),ty=wPx(-TR.y[k],n);
  const cx=wPx(TR.gx[k],n),cy=wPx(-TR.gy[k],n);
  W.strokeStyle="#666"; W.lineWidth=1; W.setLineDash([3,3]);
  W.beginPath(); W.moveTo(tx,ty); W.lineTo(cx,cy); W.stroke(); W.setLineDash([]);
  W.fillStyle="#fff"; W.beginPath(); W.arc(tx,ty,7,0,7); W.fill();
  W.fillStyle="#e5484d"; W.beginPath(); W.arc(cx,cy,7,0,7); W.fill();
  W.strokeStyle="#e5484d"; W.lineWidth=1.2;
  W.beginPath(); W.arc(cx,cy,15,0,7); W.stroke();
}

function drawRetina(){
  const n=286,c=n/2; R.fillStyle="#000"; R.fillRect(0,0,n,n);
  R.strokeStyle="#242424"; R.lineWidth=1;
  [0.33,0.66,1.0].forEach(f=>{ R.beginPath(); R.arc(c,c,f*(c-2),0,7); R.stroke(); });
  R.strokeStyle="#e5484d"; R.lineWidth=1.2;
  R.beginPath(); R.moveTo(c-10,c); R.lineTo(c+10,c);
  R.moveTo(c,c-10); R.lineTo(c,c+10); R.stroke();
  const px=c+TR.ex[k]/FOV*(c-2), py=c-TR.ey[k]/FOV*(c-2);
  const inside=TR.err[k]<=FOV;
  R.fillStyle=inside?"#fff":"#777";
  R.beginPath(); R.arc(Math.max(5,Math.min(n-5,px)),Math.max(5,Math.min(n-5,py)),
                       inside?8:5,0,7); R.fill();
  if(!inside){ R.fillStyle="#e5484d"; R.font="600 11px sans-serif";
    R.fillText("LOST — drift exceeds the field of view",8,n-9); }
}

function drawStrip(){
  const w=286,h=86; S.fillStyle="#000"; S.fillRect(0,0,w,h);
  const mx=Math.max(FOV*1.2,Math.max(...TR.err));
  S.strokeStyle="#555"; const yf=h-2-(FOV/mx)*(h-8);
  S.setLineDash([3,3]); S.beginPath(); S.moveTo(0,yf); S.lineTo(w,yf); S.stroke();
  S.setLineDash([]);
  S.strokeStyle="#e5484d"; S.lineWidth=1.6; S.beginPath();
  for(let i=0;i<TR.err.length;i++){
    const px=i/(TR.err.length-1)*w, py=h-2-(TR.err[i]/mx)*(h-8);
    i?S.lineTo(px,py):S.moveTo(px,py); } S.stroke();
  if(TR.t_lose!==null){
    const lx=TR.t_lose/TR.settings.duration*w;
    S.strokeStyle="#e5484d"; S.lineWidth=1; S.setLineDash([2,2]);
    S.beginPath(); S.moveTo(lx,0); S.lineTo(lx,h); S.stroke(); S.setLineDash([]);
    S.fillStyle="#fff"; S.font="10px sans-serif";
    S.fillText(TR.t_lose.toFixed(1)+"s",Math.min(lx+4,w-34),11);
  }
  S.strokeStyle="#888"; S.lineWidth=1; const cx=k/(TR.err.length-1)*w;
  S.beginPath(); S.moveTo(cx,0); S.lineTo(cx,h); S.stroke();
}

function axis(ctx,tgt,cmp_){
  const w=470,h=190; ctx.fillStyle="#000"; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle="#242424"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(0,h/2); ctx.lineTo(w,h/2); ctx.stroke();
  const yy=v=>h/2-v*(h/2-6);
  const line=(a,col,lw)=>{ ctx.strokeStyle=col; ctx.lineWidth=lw; ctx.beginPath();
    for(let i=0;i<=k;i++){ const px=i/(a.length-1)*w;
      i?ctx.lineTo(px,yy(a[i])):ctx.moveTo(px,yy(a[i])); } ctx.stroke(); };
  line(tgt,"#d8d8d8",1.6); line(cmp_,"#e5484d",1.6);
}

function drawJoy(){
  const n=240,c=n/2,r=c-15;
  J.fillStyle="#000"; J.fillRect(0,0,n,n);
  J.fillStyle="#fff"; J.fillRect(c-r,c-r,2*r,2*r);
  J.strokeStyle="#c9c9c9"; J.lineWidth=1;
  J.beginPath(); J.moveTo(c-r,c); J.lineTo(c+r,c);
  J.moveTo(c,c-r); J.lineTo(c,c+r); J.stroke();
  // View gain only. The command is the velocity fed INTO the integrator, not
  // the gaze velocity: for a leaky integrator the leak moves the gaze with
  // the stick untouched, and blaming that on the hand would be wrong.
  const jx=Math.max(-1,Math.min(1,TR.jx[k]*JOYGAIN));
  const jy=Math.max(-1,Math.min(1,TR.jy[k]*JOYGAIN));
  const px=c+jx*r, py=c-jy*r;
  J.strokeStyle="#e5484d"; J.lineWidth=6; J.lineCap="round";
  J.beginPath(); J.moveTo(c,c); J.lineTo(px,py); J.stroke();
  J.fillStyle="#e5484d"; J.beginPath(); J.arc(px,py,12,0,7); J.fill();
  J.fillStyle="#222"; J.beginPath(); J.arc(c,c,3.5,0,7); J.fill();
  const sat=Math.abs(TR.jx[k])>=1||Math.abs(TR.jy[k])>=1;
  const sp=Math.hypot(TR.jx[k],TR.jy[k])*TR.joy_full_scale;
  J.fillStyle="#111"; J.font="600 11px sans-serif";
  J.fillText(sp.toFixed(2)+" u/s"+(sat?"   SATURATED":""), c-r+8, c+r-9);
  if(sat){ J.strokeStyle="#e5484d"; J.lineWidth=2;
           J.strokeRect(c-r+1,c-r+1,2*r-2,2*r-2); }
}

function drawPhi(){
  const w=300,h=250,pad=30; PH.fillStyle="#000"; PH.fillRect(0,0,w,h);
  const F=TR.phi_f, U=TR.phi_u, NEG=TR.phi_neg;
  const fmax=Math.max(...F.map(Math.abs))*1.1;
  const X=u=>pad+(u+1)/2*(w-pad-8), Y=f=>h/2-f/fmax*(h/2-pad*0.6);
  PH.strokeStyle="#333"; PH.beginPath();
  PH.moveTo(pad,h/2); PH.lineTo(w-8,h/2); PH.moveTo(X(0),8); PH.lineTo(X(0),h-8);
  PH.stroke();
  // the curve, red wherever the slope is negative -> not invertible there
  for(let i=1;i<U.length;i++){
    PH.strokeStyle = (NEG[i]||NEG[i-1]) ? "#e5484d" : "#fff";
    PH.lineWidth = (NEG[i]||NEG[i-1]) ? 3.0 : 1.8;
    PH.beginPath(); PH.moveTo(X(U[i-1]),Y(F[i-1])); PH.lineTo(X(U[i]),Y(F[i])); PH.stroke();
  }
  const u=Math.max(-1,Math.min(1,TR.u_cmd[k]));
  let j=0; for(let i=1;i<U.length;i++) if(Math.abs(U[i]-u)<Math.abs(U[j]-u)) j=i;
  PH.fillStyle = NEG[j] ? "#e5484d" : "#4da3ff";
  PH.beginPath(); PH.arc(X(u),Y(F[j]),6,0,7); PH.fill();
  PH.fillStyle="#fff"; PH.font="10px sans-serif";
  PH.fillText("u",w-16,h/2-6); PH.fillText("deg",X(0)+5,14);
  if(TR.phi_neg_frac>0){ PH.fillStyle="#e5484d";
    PH.fillText("not invertible over "+(TR.phi_neg_frac*100).toFixed(0)+"%",pad,h-8); }
}

function drawEye(){
  const n=250,c=n/2,R=72; EY.fillStyle="#000"; EY.fillRect(0,0,n,n);
  const g=TR.gaze_deg[k], t=TR.target_deg[k];
  // target eccentricity, as a mark on the orbit
  const ta=t*Math.PI/180;
  EY.strokeStyle="#888"; EY.lineWidth=1;
  EY.beginPath(); EY.arc(c,c,R+26,0,7); EY.stroke();
  EY.fillStyle="#fff";
  EY.beginPath(); EY.arc(c+Math.sin(ta)*(R+26),c-Math.cos(ta)*(R+26),5,0,7); EY.fill();
  // globe, rotated by the gaze the plant actually produced
  const ga=g*Math.PI/180;
  EY.fillStyle="#1b2129"; EY.strokeStyle="#fff"; EY.lineWidth=1.4;
  EY.beginPath(); EY.arc(c,c,R,0,7); EY.fill(); EY.stroke();
  const px=c+Math.sin(ga)*R*0.62, py=c-Math.cos(ga)*R*0.62;
  EY.fillStyle="#cfd8e3"; EY.beginPath(); EY.arc(px,py,R*0.34,0,7); EY.fill();
  EY.fillStyle="#111"; EY.beginPath(); EY.arc(px,py,R*0.14,0,7); EY.fill();
  EY.strokeStyle="#4da3ff"; EY.lineWidth=2;
  EY.beginPath(); EY.moveTo(c,c); EY.lineTo(c+Math.sin(ga)*R,c-Math.cos(ga)*R); EY.stroke();
  EY.fillStyle="#fff"; EY.font="11px sans-serif";
  EY.fillText("gaze "+g.toFixed(1)+"\u00b0",8,16);
  EY.fillStyle="#888"; EY.fillText("target "+t.toFixed(1)+"\u00b0",8,32);
}

function drawGaze(){
  const w=600,h=250,pad=26; GZ.fillStyle="#000"; GZ.fillRect(0,0,w,h);
  const A=TR.target_deg, B=TR.gaze_deg;
  const m=Math.max(...A.map(Math.abs),...B.map(Math.abs))*1.1||1;
  const Y=v=>h/2-v/m*(h/2-pad*0.5);
  GZ.strokeStyle="#333"; GZ.beginPath(); GZ.moveTo(0,h/2); GZ.lineTo(w,h/2); GZ.stroke();
  const line=(a,col,lw)=>{ GZ.strokeStyle=col; GZ.lineWidth=lw; GZ.beginPath();
    for(let i=0;i<=k;i++){ const px=i/(a.length-1)*w;
      i?GZ.lineTo(px,Y(a[i])):GZ.moveTo(px,Y(a[i])); } GZ.stroke(); };
  line(A,"#d8d8d8",1.8); line(B,"#4da3ff",1.8);
  GZ.fillStyle="#fff"; GZ.font="11px sans-serif";
  GZ.fillText("target",8,16); GZ.fillStyle="#4da3ff"; GZ.fillText("gaze (after the eye)",58,16);
  GZ.fillStyle="#888";
  GZ.fillText("mean |error| "+TR.gaze_err_mean.toFixed(2)+"\u00b0"
    +"   world "+TR.deg_per_unit.toFixed(1)+"\u00b0/unit",8,h-8);
}

function stats(){
  const inside=TR.err.filter(e=>e<=FOV).length/TR.err.length*100;
  const lost = TR.t_lose===null
    ? 'never lost within '+TR.settings.duration.toFixed(0)+'s'
    : '<span class="lost">lost after <b>'+TR.t_lose.toFixed(1)+' s</b></span>';
  document.getElementById("stats").innerHTML=
    lost+` &nbsp;&middot;&nbsp; mean |drift| <b>${TR.err_mean.toFixed(3)}</b>`+
    ` &nbsp; final <b>${TR.err_end.toFixed(3)}</b>`+
    ` &nbsp;&middot;&nbsp; inside the field of view <b>${inside.toFixed(1)}%</b>`+
    ` of the time &nbsp;&middot;&nbsp; seed <b>${TR.settings.seed}</b>`;
}

function frame(){ drawWorld(); drawRetina(); drawStrip(); drawJoy();
  axis(X,TR.x,TR.gx); axis(Y,TR.y,TR.gy); stats();
  const on = !!TR.plant;
  document.getElementById("worldcap").innerHTML = on
    ? 'world &mdash; target, and <i>where the eye points</i>'
    : 'world &mdash; target, and <i>computed</i>';
  document.getElementById("plantrow").style.display = on ? "flex" : "none";
  if(on){ drawPhi(); drawEye(); drawGaze(); }
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
                            [n for n in OPEN_LOOP if not n.startswith("ml:")]))
                        .replace("__CTRL_M__", json.dumps(
                            # per-eye controllers are NOT listed here: the
                            # plant row already selects the eye and its own
                            # network together, so a second set of A-E buttons
                            # would be the same choice twice.
                            [n for n in OPEN_LOOP if n.startswith("ml:")
                             and not n.startswith("ml:ctrnn_eye_")]))
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
                    u_cmd = np.clip(gx, -1.0, 1.0)
                    v_cmd = np.clip(gy, -1.0, 1.0)
                    gaze_h = plant_run(pname, u_cmd, _dt, "h")
                    gaze_v = plant_run(pname, v_cmd, _dt, "v")
                    dpu = deg_per_unit(q.get("world", "auto"), pname)
                    tgt = np.asarray(tr["x"]) * dpu
                    pu, pf, pneg = plant_curve(pname, axis="h")
                    # THE RED TRAJECTORY IS NOW THE EYE, not the integrator.
                    # Everything downstream — world track, retina, drift strip,
                    # x(t)/y(t) — reads gx/gy, so writing the gaze back into
                    # them makes every panel show where the eye actually
                    # points rather than where the controller believes it is.
                    gx = gaze_h / dpu
                    gy = gaze_v / dpu
                    ex, ey = x - gx, y - gy
                    err = np.hypot(ex, ey)
                    lost = np.flatnonzero(err > FOV)
                    tr.update({
                        "gx": gx.tolist(), "gy": gy.tolist(),
                        "ex": ex.tolist(), "ey": ey.tolist(),
                        "err": err.tolist(),
                        "err_mean": float(err.mean()),
                        "err_end": float(err[-1]),
                        "t_lose": (float(tr["t"][int(lost[0])])
                                   if lost.size else None),
                        "plant": pname, "gaze_deg": gaze_h.tolist(),
                        "target_deg": tgt.tolist(),
                        "gaze_err": np.abs(gaze_h - tgt).tolist(),
                        "gaze_err_mean": float(np.abs(gaze_h - tgt).mean()),
                        "phi_u": pu, "phi_f": pf, "phi_neg": pneg,
                        "phi_neg_frac": float(np.mean(pneg)),
                        "u_cmd": u_cmd.tolist(),
                        "deg_per_unit": dpu,
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
