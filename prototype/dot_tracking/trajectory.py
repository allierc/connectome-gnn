"""Target-dot trajectories on the [-1, 1]^2 grid.

The stimulus half of the dot-tracking prototype: a generator of pursuit
targets whose character is set by four independent switches. It has no web
and no plotting dependency, so the follower work can import it directly.

    shape   segment | curve         piecewise-linear vs Catmull-Rom smoothed
    motion  continue | stop_and_go  constant travel vs move/pause alternation
    speed   slow | middle | fast    0.15 | 0.40 | 0.90 grid-units per second
    angle   low | sharp             5-35 deg vs 80-160 deg turns at waypoints

The four are orthogonal by construction: `shape` and `angle` decide the PATH
(a curve in space), `speed` and `motion` decide the SCHEDULE along it (how
arc length is consumed over time). Generation therefore runs in two stages —
lay down the path, then walk it — which is also why the dot moves at a
genuinely constant speed rather than at constant parameter increment, the
usual artefact of spline animation.

Geometry: positions are clamped to [-0.95, 0.95] by reflecting the heading
off the wall, so the dot stays visible without the path piling up in a
corner.

Usage::

    python trajectory.py --shape curve --angle sharp --speed fast --json out.json
    from trajectory import generate, SPEC
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# --- the four switches ------------------------------------------------------
SPEC = {
    "shape":  ["segment", "curve"],
    "motion": ["continue", "stop_and_go"],
    "speed":  ["slow", "middle", "fast"],
    "angle":  ["low", "sharp"],
}

SPEED_UPS = {"slow": 0.15, "middle": 0.40, "fast": 0.90}   # grid units / s
TURN_DEG = {"low": (5.0, 35.0), "sharp": (80.0, 160.0)}
STEP_LEN = {"low": 0.45, "sharp": 0.30}   # waypoint spacing, grid units
BOUND = 0.95

# stop_and_go: alternating move / pause, each drawn from these ranges (s),
# with a raised-cosine ramp so the dot does not step discontinuously.
GO_S = (0.45, 1.10)
STOP_S = (0.30, 0.80)
RAMP_S = 0.12


def _waypoints(rng, angle, need_len):
    """Random walk of headings, reflected at the walls, until the polyline is
    at least `need_len` long. Returns (M, 2)."""
    lo, hi = np.deg2rad(TURN_DEG[angle])
    step = STEP_LEN[angle]
    p = rng.uniform(-0.55, 0.55, size=2)
    heading = rng.uniform(0.0, 2.0 * np.pi)
    pts, acc = [p.copy()], 0.0
    while acc < need_len:
        heading += rng.uniform(lo, hi) * rng.choice([-1.0, 1.0])
        for _ in range(12):
            nxt = p + step * np.array([np.cos(heading), np.sin(heading)])
            if np.all(np.abs(nxt) <= BOUND):
                break
            # Reflect off whichever wall was crossed; if both, reflect twice.
            if abs(nxt[0]) > BOUND:
                heading = np.pi - heading
            if abs(nxt[1]) > BOUND:
                heading = -heading
        else:
            # Cornered: aim back at the origin and take the step regardless.
            heading = np.arctan2(-p[1], -p[0])
            nxt = p + step * np.array([np.cos(heading), np.sin(heading)])
        nxt = np.clip(nxt, -BOUND, BOUND)
        acc += float(np.linalg.norm(nxt - p))
        p = nxt
        pts.append(p.copy())
    return np.asarray(pts)


def _catmull_rom(pts, per_seg=24):
    """Centripetal-ish Catmull-Rom through the waypoints (uniform knots are
    enough here; the walk already spaces points evenly)."""
    P = np.vstack([pts[0], pts, pts[-1]])          # duplicate the endpoints
    out = []
    t = np.linspace(0.0, 1.0, per_seg, endpoint=False)[:, None]
    for i in range(len(P) - 3):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        out.append(0.5 * ((2 * p1)
                          + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2][None, :])
    return np.clip(np.vstack(out), -BOUND, BOUND)


def _speed_profile(rng, motion, v, n, dt):
    """Instantaneous speed per sample. `continue` is flat; `stop_and_go`
    alternates go/stop with raised-cosine ramps of RAMP_S seconds."""
    if motion == "continue":
        return np.full(n, v)
    gate = np.zeros(n)
    i, on = 0, True
    while i < n:
        span = rng.uniform(*(GO_S if on else STOP_S))
        j = min(n, i + max(1, int(round(span / dt))))
        gate[i:j] = 1.0 if on else 0.0
        i, on = j, not on
    k = max(3, int(round(RAMP_S / dt)))
    win = 0.5 * (1.0 - np.cos(np.linspace(0.0, 2.0 * np.pi, k)))
    win /= win.sum()
    return v * np.convolve(gate, win, mode="same")


def generate(shape="curve", motion="continue", speed="middle", angle="low",
             duration=20.0, dt=1.0 / 60.0, seed=None):
    """Return a dict with t, x, y, speed and the settings that produced them."""
    for k, v in (("shape", shape), ("motion", motion),
                 ("speed", speed), ("angle", angle)):
        if v not in SPEC[k]:
            raise ValueError(f"{k}={v!r} not in {SPEC[k]}")
    seed = int(np.random.SeedSequence().entropy % 2**31) if seed is None \
        else int(seed)
    rng = np.random.default_rng(seed)

    n = int(round(duration / dt))
    v = SPEED_UPS[speed]
    sp = _speed_profile(rng, motion, v, n, dt)
    need = float(sp.sum() * dt) * 1.15 + 1.0      # margin for spline stretch

    path = _waypoints(rng, angle, need)
    if shape == "curve":
        path = _catmull_rom(path)

    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s_path = np.concatenate([[0.0], np.cumsum(seg)])
    s_t = np.concatenate([[0.0], np.cumsum(sp[:-1] * dt)])
    s_t = np.clip(s_t, 0.0, s_path[-1])           # never run off the end

    x = np.interp(s_t, s_path, path[:, 0])
    y = np.interp(s_t, s_path, path[:, 1])
    return {
        "t": (np.arange(n) * dt).tolist(),
        "x": x.tolist(),
        "y": y.tolist(),
        "speed": sp.tolist(),
        "settings": {"shape": shape, "motion": motion, "speed": speed,
                     "angle": angle, "duration": duration, "dt": dt,
                     "seed": seed},
        "path_len": float(s_path[-1]),
        "n_waypoints": int(len(path)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, opts in SPEC.items():
        p.add_argument(f"--{k}", default=opts[0], choices=opts)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--json", default=None, help="write the trace here")
    a = p.parse_args()
    tr = generate(a.shape, a.motion, a.speed, a.angle, a.duration, a.dt, a.seed)
    print(f"{tr['settings']}\n  samples={len(tr['t'])}  "
          f"path_len={tr['path_len']:.2f}  waypoints={tr['n_waypoints']}  "
          f"x in [{min(tr['x']):.2f},{max(tr['x']):.2f}]  "
          f"y in [{min(tr['y']):.2f},{max(tr['y']):.2f}]")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(tr, f)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
