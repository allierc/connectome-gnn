"""Gaze controllers — the "joystick" half of the dot-tracking prototype.

A follower consumes the target trace and produces a gaze trajectory. The
question the retina inset asks is whether the dot can be held at the CENTRE
of the field of view, so the quantity of interest is the retinal error

    e(t) = target(t) - gaze(t)

which is what an oculomotor circuit actually sees and must drive to zero.

THE PLANT IS AN INTEGRATOR. The joystick commands gaze *velocity*, so gaze
position is the integral of the stick. That single fact sets the character of
every controller here: proportional feedback alone already closes the loop
with first-order dynamics, `dg/dt = Kp * e`, whose time constant is `1/Kp`.
There is no need for an explicit low-pass anywhere — the plant supplies it.

Each follower declares its knobs in PARAMS with a low/high range, and the web
UI builds a slider per knob from that declaration. To add a controller, write
the function, register it with its parameter spec, and it appears in the
selector; nothing else needs editing.
"""
from __future__ import annotations

import numpy as np

FOLLOWERS = {}
PARAMS = {}

# Joystick full-scale deflection, in grid units per second. The stick is a
# RATE control: its direction is the direction the gaze is commanded to move
# and its magnitude is the commanded speed, so deflection = gaze velocity
# divided by this constant, clipped to the unit box. 1.6 units/s is a little
# under twice the "fast" target speed (0.90 units/s), so a saturated stick
# can still catch up with the quickest dot but not instantly.
JOY_FULL_SCALE = 1.6


def register(name, params=()):
    def deco(fn):
        FOLLOWERS[name] = fn
        PARAMS[name] = list(params)
        return fn
    return deco


def _knob(name, label, lo, hi, default, step):
    return {"name": name, "label": label, "min": lo, "max": hi,
            "default": default, "step": step}


@register("fixed")
def fixed(t, x, y, **kw):
    """Gaze pinned at the origin — the do-nothing baseline.

    The retinal error IS the target, so this sets the scale every other
    controller is measured against. A follower that cannot beat `fixed` is
    not tracking.
    """
    return np.zeros_like(x), np.zeros_like(y)


@register("pid", [
    _knob("kp", "P — gain on error", 0.0, 20.0, 5.5, 0.1),
    _knob("ki", "I — gain on accumulated error", 0.0, 30.0, 0.0, 0.5),
    _knob("kd", "D — gain on error rate", 0.0, 2.0, 0.0, 0.02),
])
def pid(t, x, y, kp=5.5, ki=0.0, kd=0.0, **kw):
    """Textbook PID on the retinal error, output = stick deflection.

        v = Kp e + Ki integral(e) + Kd de/dt ,   dg/dt = v

    With Ki = Kd = 0 this is pure proportional feedback and the loop is a
    first-order lag of time constant 1/Kp — so Kp = 5.5 is a 0.18 s pursuit
    latency. Raising Kp shortens it until the one-sample delay of the
    discrete loop starts to ring.

    I integrates away the standing error that P alone leaves during
    constant-velocity pursuit, at the cost of overshoot after a stop.
    D anticipates, and amplifies the corner of every sharp turn.

    The integral is frozen whenever the stick is against its gate
    (clamping anti-windup); without that, a long saturated stretch charges
    the integrator and the gaze sails past the target when it is released.
    """
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0 / 60.0
    n = len(x)
    gx = np.empty(n); gy = np.empty(n)
    cx = cy = 0.0
    ix = iy = 0.0
    epx = epy = 0.0
    for k in range(n):
        gx[k], gy[k] = cx, cy
        ex, ey = x[k] - cx, y[k] - cy
        dx = (ex - epx) / dt if k else 0.0
        dy = (ey - epy) / dt if k else 0.0
        vx = kp * ex + ki * ix + kd * dx
        vy = kp * ey + ki * iy + kd * dy
        sx = np.clip(vx, -JOY_FULL_SCALE, JOY_FULL_SCALE)
        sy = np.clip(vy, -JOY_FULL_SCALE, JOY_FULL_SCALE)
        if sx == vx:                      # anti-windup: only integrate when
            ix += ex * dt                 # the stick has authority left
        if sy == vy:
            iy += ey * dt
        cx += sx * dt
        cy += sy * dt
        epx, epy = ex, ey
    return gx, gy


@register("pursuit", [
    _knob("kp", "P — gain on error", 0.0, 20.0, 4.0, 0.1),
    _knob("kff", "feedforward on target velocity", 0.0, 1.5, 0.9, 0.02),
    _knob("delay_ms", "sensorimotor delay (ms)", 0.0, 250.0, 80.0, 5.0),
])
def pursuit(t, x, y, kp=4.0, kff=0.9, delay_ms=80.0, **kw):
    """Smooth pursuit: velocity feedforward plus proportional feedback,
    both acting on delayed sensory information.

        v(t) = Kff * target_velocity(t - D) + Kp * e(t - D)

    This is the biologically shaped one. Pure feedback cannot track a moving
    target without a standing error, because the error is what generates the
    command — the eye must fall behind to keep moving. Real smooth pursuit
    solves it with a velocity-matching feedforward term, so Kff near 1 keeps
    up with a constant-velocity target and Kp only mops up the residual.

    The delay is the reason this is not trivially solved: every biological
    loop is 60-130 ms behind the world, so the controller steers by where
    the target WAS. Raising delay_ms at fixed gain is what turns smooth
    pursuit into oscillation, which is worth seeing on the error strip.
    """
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0 / 60.0
    n = len(x)
    lag = max(0, int(round((delay_ms / 1000.0) / dt)))
    vx_t = np.gradient(x, dt)
    vy_t = np.gradient(y, dt)
    gx = np.empty(n); gy = np.empty(n)
    cx = cy = 0.0
    for k in range(n):
        gx[k], gy[k] = cx, cy
        j = k - lag
        if j < 0:                          # before the first sensory sample
            continue                       # arrives, no command is possible
        ex, ey = x[j] - gx[j], y[j] - gy[j]
        vx = kff * vx_t[j] + kp * ex
        vy = kff * vy_t[j] + kp * ey
        cx += np.clip(vx, -JOY_FULL_SCALE, JOY_FULL_SCALE) * dt
        cy += np.clip(vy, -JOY_FULL_SCALE, JOY_FULL_SCALE) * dt
    return gx, gy


def joystick(t, gx, gy):
    """The stick deflection that would have produced this gaze.

    Rate control: deflection direction is the direction of commanded gaze
    motion, deflection magnitude is the commanded speed as a fraction of
    JOY_FULL_SCALE. Derived from the gaze velocity rather than asked of each
    follower, so any controller — including ones not written yet — gets a
    joystick trace for free. Values are clipped to the square [-1, 1]^2,
    which is the physical stick gate; a clipped sample means the controller
    is demanding more speed than the stick can deliver.
    """
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0 / 60.0
    jx = np.clip(np.gradient(gx, dt) / JOY_FULL_SCALE, -1.0, 1.0)
    jy = np.clip(np.gradient(gy, dt) / JOY_FULL_SCALE, -1.0, 1.0)
    sat = float(np.mean((np.abs(jx) >= 1.0) | (np.abs(jy) >= 1.0)))
    return jx, jy, sat


def apply(name, t, x, y, **kw):
    """Run a follower and return gaze, retinal error, joystick and summaries.

    Unknown keyword arguments are dropped rather than raising, so the web UI
    can post the union of every controller's knobs without knowing which one
    is selected.
    """
    if name not in FOLLOWERS:
        raise ValueError(f"follower {name!r} not in {sorted(FOLLOWERS)}")
    t, x, y = map(np.asarray, (t, x, y))
    allowed = {p["name"] for p in PARAMS.get(name, [])}
    kw = {k: v for k, v in kw.items() if k in allowed}
    gx, gy = FOLLOWERS[name](t, x, y, **kw)
    ex, ey = x - gx, y - gy
    r = np.hypot(ex, ey)
    jx, jy, sat = joystick(t, gx, gy)
    return {
        "gx": gx.tolist(), "gy": gy.tolist(),
        "ex": ex.tolist(), "ey": ey.tolist(), "err": r.tolist(),
        "jx": jx.tolist(), "jy": jy.tolist(),
        "err_mean": float(r.mean()), "err_max": float(r.max()),
        "err_p95": float(np.percentile(r, 95)),
        "joy_sat": sat, "joy_full_scale": JOY_FULL_SCALE,
        "follower": name, "params": kw,
    }
