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

Four failure modes of an integrator are separated, because each leaves a
different signature in how the error grows:

    perfect   exact integration           error stays at numerical zero
    gain      integrates k*v, k != 1      error grows LINEARLY with path length
    leaky     dg/dt = -g/tau + v          error saturates; gaze sags to centre
    noisy     integrates v + white noise  error grows as sqrt(t): a random walk

Those three growth laws — linear, saturating, square-root — are the point.
They are distinguishable from a single error trace, so a real circuit's drift
can be classified rather than merely measured.

Usage::

    python openloop.py                       # sweep + figure + table
    python openloop.py --n-seeds 40 --duration 30
    python openloop.py --out /tmp/openloop.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory import generate                       # noqa: E402
from followers import JOY_FULL_SCALE                  # noqa: E402

FOV = 0.6                    # field-of-view radius, grid units

OPEN_LOOP = {}
PARAMS = {}


def register(name, params=()):
    def deco(fn):
        OPEN_LOOP[name] = fn
        PARAMS[name] = dict(params)
        return fn
    return deco


def _integrate(vx, vy, dt, x0, y0, leak=None):
    """Gaze from a commanded velocity, starting at (x0, y0).

    The stick saturates at JOY_FULL_SCALE exactly as in the closed-loop app,
    so a controller cannot cheat by commanding an arbitrarily large velocity.
    `leak` is the integrator time constant in seconds; None = perfect.
    """
    n = len(vx)
    gx = np.empty(n); gy = np.empty(n)
    cx, cy = float(x0), float(y0)
    a = 0.0 if leak is None else dt / float(leak)
    for k in range(n):
        gx[k], gy[k] = cx, cy
        ux = np.clip(vx[k], -JOY_FULL_SCALE, JOY_FULL_SCALE)
        uy = np.clip(vy[k], -JOY_FULL_SCALE, JOY_FULL_SCALE)
        cx += ux * dt - a * cx
        cy += uy * dt - a * cy
    return gx, gy


@register("perfect")
def perfect(t, vx, vy, x0, y0, dt, rng, **kw):
    """Exact integration — the upper bound. Any residual error is the
    discretisation of the integral, not a property of the controller."""
    return _integrate(vx, vy, dt, x0, y0)


@register("gain", {"k": 0.5})
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


@register("leaky", {"tau": 2.0})
def leaky(t, vx, vy, x0, y0, dt, rng, tau=2.0, **kw):
    """Leaky integrator, dg/dt = -g/tau + v — the biological one.

    The leak pulls gaze back toward the centre, so a sustained excursion is
    under-represented and the error SATURATES rather than growing without
    bound. tau is the integrator time constant the oculomotor circuit is
    supposed to make long; tau -> infinity recovers `perfect`.
    """
    return _integrate(vx, vy, dt, x0, y0, leak=tau)


@register("noisy", {"sigma": 0.15})
def noisy(t, vx, vy, x0, y0, dt, rng, sigma=0.15, **kw):
    """Integrates v plus white velocity noise, scaled so that sigma is the
    drift in GRID UNITS PER SQRT(SECOND).

    The per-sample noise is sigma/sqrt(dt), which makes the integrated random
    walk independent of the sampling rate and gives the drift a closed form:
    its standard deviation after T seconds is exactly sigma*sqrt(T), per axis.
    So sigma = 0.15 puts the typical drift at the 0.6 field-of-view radius
    after 16 s. Parameterising it any other way makes the answer depend on dt,
    which is a property of the simulation and not of the circuit.

    Growth as sqrt(t) is slower than a gain error at first but unbounded, and
    it is the only one of the four with zero mean error — the drift has no
    systematic direction, so averaging across trials hides it entirely.
    """
    nz = rng.normal(0.0, sigma / np.sqrt(dt), size=(2, len(vx)))
    return _integrate(vx + nz[0], vy + nz[1], dt, x0, y0)


def run_trial(name, tr, rng, **kw):
    """Run one controller on one trajectory; return error and survival time."""
    t = np.asarray(tr["t"]); x = np.asarray(tr["x"]); y = np.asarray(tr["y"])
    dt = float(tr["settings"]["dt"])
    vx = np.gradient(x, dt); vy = np.gradient(y, dt)
    gx, gy = OPEN_LOOP[name](t, vx, vy, x[0], y[0], dt, rng, **kw)
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
                              **PARAMS[name])
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
    gx, gy = OPEN_LOOP[name](np.asarray(tr["t"]), vx, vy, x[0], y[0], dt,
                             np.random.default_rng(9000 + seed), **kw)
    return {"t": np.asarray(tr["t"]), "x": x, "y": y, "gx": gx, "gy": gy,
            "name": name}


def figure(res, conditions, n_seeds, duration, dt, out_path, tau_sweep=None,
           taus=None, example=None):
    names = list(OPEN_LOOP)
    col = {"perfect": "#111111", "gain": "#cf222e",
           "leaky": "#1f6feb", "noisy": "#d29922"}
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
    for slope, lab, off in ((1.0, "linear (gain)", 0.30),
                            (0.5, "sqrt t (noise)", 0.06)):
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


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-seeds", type=int, default=24)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "openloop.png"))
    a = p.parse_args()

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
    print("\nparameters:", {k: v for k, v in PARAMS.items() if v})
    print("\nmedian survival (s) against integrator tau, low-angle curve")
    print(f"{'tau (s)':>9s}" + "".join(f"{t:>8g}" for t in taus))
    for sp, med in tsw.items():
        print(f"{sp:>9s}" + "".join(
            f"{'never' if m >= a.duration else f'{m:.1f}':>8s}" for m in med))
    ex = example_trial(conditions[1], "leaky", a.duration, a.dt, seed=0,
                       **PARAMS["leaky"])
    figure(res, conditions, a.n_seeds, a.duration, a.dt, a.out,
           tau_sweep=tsw, taus=taus, example=ex)


if __name__ == "__main__":
    main()
