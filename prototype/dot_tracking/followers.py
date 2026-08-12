"""Gaze controllers — the "joystick" half of the dot-tracking prototype.

A follower consumes the target trace and produces a gaze trajectory. The
question the inset viz asks is whether the dot can be held at the CENTRE of
the field of view, so the quantity of interest is not the gaze itself but the
retinal error

    e(t) = target(t) - gaze(t)

which is what an oculomotor circuit actually sees and what it must drive to
zero. A follower that lags produces a persistent e; one that oscillates
produces a ringing e; a perfect one produces none.

Every follower is a function (t, x, y) -> (gx, gy) registered in FOLLOWERS.
They run causally: gaze at step k may use the target up to and including k,
never beyond. Two are provided as reference points, deliberately trivial:

    fixed   gaze never moves — the do-nothing baseline. The error IS the
            target, so the inset shows the raw stimulus and sets the scale
            every other follower is measured against.
    lag     first-order pursuit, gaze accelerates toward the target with a
            time constant tau. One knob, no prediction, no saccades: the
            simplest thing that could work, and the one to beat.

The real algorithms go here next; the plumbing already carries them.
"""
from __future__ import annotations

import numpy as np

FOLLOWERS = {}

# Joystick full-scale deflection, in grid units per second. The stick is a
# RATE control: its direction is the direction the gaze is commanded to move
# and its magnitude is the commanded speed, so deflection = gaze velocity
# divided by this constant, clipped to the unit box. 1.6 units/s is a little
# under twice the "fast" target speed (0.90 units/s), so a saturated stick
# can still catch up with the quickest dot but not instantly.
JOY_FULL_SCALE = 1.6


def register(name):
    def deco(fn):
        FOLLOWERS[name] = fn
        return fn
    return deco


@register("fixed")
def fixed(t, x, y, **kw):
    """Gaze pinned at the origin."""
    return np.zeros_like(x), np.zeros_like(y)


@register("lag")
def lag(t, x, y, tau=0.18, **kw):
    """First-order low-pass pursuit: dg/dt = (target - g) / tau.

    tau is the pursuit latency in seconds. Larger tau lags further behind and
    leaves a bigger standing error during constant-velocity travel — the
    classic pursuit-gain shortfall.
    """
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0 / 60.0
    a = min(1.0, dt / max(float(tau), 1e-6))
    gx, gy = np.empty_like(x), np.empty_like(y)
    cx = cy = 0.0
    for k in range(len(x)):
        cx += a * (x[k] - cx)
        cy += a * (y[k] - cy)
        gx[k], gy[k] = cx, cy
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
    vx = np.gradient(gx, dt)
    vy = np.gradient(gy, dt)
    jx = np.clip(vx / JOY_FULL_SCALE, -1.0, 1.0)
    jy = np.clip(vy / JOY_FULL_SCALE, -1.0, 1.0)
    sat = float(np.mean((np.abs(jx) >= 1.0) | (np.abs(jy) >= 1.0)))
    return jx, jy, sat


def apply(name, t, x, y, **kw):
    """Run a follower and return gaze, retinal error, joystick and summaries."""
    if name not in FOLLOWERS:
        raise ValueError(f"follower {name!r} not in {sorted(FOLLOWERS)}")
    t, x, y = map(np.asarray, (t, x, y))
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
        "follower": name,
    }
