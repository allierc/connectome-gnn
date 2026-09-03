"""Target-dot trajectories on the [-1, 1]^2 grid.

The stimulus half of the dot-tracking prototype: a generator of pursuit
targets whose character is set by four independent switches. It has no web
and no plotting dependency, so the follower work can import it directly.

    shape   segment | curve         piecewise-linear vs Catmull-Rom smoothed
    motion  continue | stop_and_go  constant travel vs move/pause alternation
    speed   slow | middle | fast    0.15 | 0.40 | 0.90 grid-units per second
    angle   low | sharp             5-35 deg vs 80-160 deg turns at waypoints
    axis    plane | horizontal      the full grid vs the horizontal line y=0

The four are orthogonal by construction: `shape` and `angle` decide the PATH
(a curve in space), `speed` and `motion` decide the SCHEDULE along it (how
arc length is consumed over time). Generation therefore runs in two stages —
lay down the path, then walk it — which is also why the dot moves at a
genuinely constant speed rather than at constant parameter increment, the
usual artefact of spline animation.

Geometry: positions are clamped to [-0.95, 0.95] by reflecting the heading
off the wall, so the dot stays visible without the path piling up in a
corner.

`axis=horizontal` confines the dot to y = 0. The heading still walks the full
circle; only its x-component is stepped, and the resulting line is then walked
at the declared speed — so `speed` keeps its meaning exactly (a horizontal
'fast' dot really does travel 0.90 units/s along x, whereas merely dropping
the y column of a plane 'fast' trajectory would leave it moving at
0.90·|cos θ|, pausing every time the plane dot went vertical). The other
three switches survive with their meaning intact but narrowed:

    angle   unchanged in meaning -- the heading still walks the full circle;
            only its x-component is stepped. `low` drifts, giving long runs
            that reverse at the wall or when the heading rolls through
            vertical; `sharp` changes the step's sign and length often, so
            the dot wanders rather than oscillating in place.
    shape   still the reversal's shape: `segment` reverses instantly (a
            velocity step), `curve` decelerates through the turnaround.
    motion  unchanged; the schedule does not know about dimension.

Use it when the effector being driven has one horizontal degree of freedom --
the 285-cell oculomotor pool reaches LR and MR only, so a vertical stimulus
component there is an input no output can answer.

Building a CORPUS. A single trajectory is one call to `generate`. A training
corpus is a yaml under `config/zebrafish/` naming the condition grid, the
per-split trial counts and the seed offsets that keep the splits disjoint:

    python trajectory.py --corpus ../../config/zebrafish/dot_corpus_2d.yaml
    python trajectory.py --corpus ../../config/zebrafish/dot_corpus_1d.yaml

Both write to `$GNN_OUTPUT_ROOT/graphs_data/zebrafish/<name>/`: one npz per
split, plus `<name>.png` (the condition grid, one panel per condition) and
`<name>.mp4` (a sample of trials animated, so the stimulus can be watched
rather than inferred from a static plot). Nothing is written into the repo.

Usage::

    python trajectory.py --shape curve --angle sharp --speed fast --json out.json
    python trajectory.py --corpus <spec.yaml> [--force] [--no-viz]
    from trajectory import generate, build_corpus, SPEC
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# --- the four switches ------------------------------------------------------
SPEC = {
    "start":  ["center", "random"],
    "shape":  ["segment", "curve"],
    "motion": ["continue", "stop_and_go"],
    "speed":  ["slow", "middle", "fast"],
    "angle":  ["low", "sharp"],
    "axis":   ["plane", "horizontal"],
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


MIN_COS = 0.15   # headings within 8.6 deg of vertical contribute no waypoint


def _waypoints_h(rng, angle, need_len, start="random"):
    """The `axis=horizontal` walk: the plane walk's HORIZONTAL COMPONENT,
    re-walked as a path on the line y = 0. Returns (M, 2), second column zero,
    so every caller downstream stays two-dimensional and unchanged.

    The heading still random-walks over the full circle exactly as
    `_waypoints` does -- only the step taken is its x-component,
    `step * cos(heading)`. Collapsing the heading to +-x instead would make
    `angle=sharp` a flip-flop between two positions (a 160 deg turn and a 180
    deg turn are the same reversal once the direction is binary), which is a
    square wave, not a pursuit target. Keeping cos(heading) preserves what
    `angle` means: `low` drifts, so the dot makes long runs and reverses only
    at the wall or when the heading rolls through vertical; `sharp` jumps
    80-160 deg per waypoint, so the step changes sign often AND varies in
    length, and the dot wanders instead of oscillating in place.

    Headings within `MIN_COS` of vertical are skipped rather than emitted as
    a near-duplicate waypoint -- in the plane those samples are vertical
    travel, and on the line they are nothing at all.

    Note what this costs: the horizontal path length is shorter than the plane
    path it came from, so the dot covers less ground per waypoint. The SPEED
    is unaffected -- `generate` walks whatever path this returns at the
    declared units/s -- but the turn rate per second is lower than the plane
    corpus at the same `angle`."""
    lo, hi = np.deg2rad(TURN_DEG[angle])
    step = STEP_LEN[angle]
    x = 0.0 if start == "center" else float(rng.uniform(-0.55, 0.55))
    heading = float(rng.uniform(0.0, 2.0 * np.pi))
    xs, acc = [x], 0.0
    for _ in range(200000):
        if acc >= need_len:
            break
        heading += float(rng.uniform(lo, hi)) * float(rng.choice([-1.0, 1.0]))
        c = float(np.cos(heading))
        if abs(c) < MIN_COS:
            continue
        nxt = x + step * c
        if abs(nxt) > BOUND:                       # reflect off the wall
            heading = np.pi - heading
            c = float(np.cos(heading))
            nxt = x + step * c
        nxt = float(np.clip(nxt, -BOUND, BOUND))
        acc += abs(nxt - x)
        x = nxt
        xs.append(x)
    return np.stack([np.asarray(xs), np.zeros(len(xs))], axis=-1)


def _waypoints(rng, angle, need_len, start="random"):
    """Random walk of headings, reflected at the walls, until the polyline is
    at least `need_len` long. Returns (M, 2).

    ``start='center'`` pins the first waypoint at the origin, which is what an
    open-loop experiment needs: the controller is handed the initial position
    and must integrate from there, so every trial has to begin at a known,
    identical place for the drift to be comparable across seeds."""
    lo, hi = np.deg2rad(TURN_DEG[angle])
    step = STEP_LEN[angle]
    p = (np.zeros(2) if start == "center"
         else rng.uniform(-0.55, 0.55, size=2))
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
             duration=20.0, dt=1.0 / 60.0, seed=None, start="random",
             axis="plane", speed_scale=1.0):
    """Return a dict with t, x, y, speed and the settings that produced them."""
    for k, v in (("shape", shape), ("motion", motion), ("speed", speed),
                 ("angle", angle), ("start", start), ("axis", axis)):
        if v not in SPEC[k]:
            raise ValueError(f"{k}={v!r} not in {SPEC[k]}")
    seed = int(np.random.SeedSequence().entropy % 2**31) if seed is None \
        else int(seed)
    rng = np.random.default_rng(seed)

    n = int(round(duration / dt))
    v = SPEED_UPS[speed] * float(speed_scale)
    sp = _speed_profile(rng, motion, v, n, dt)
    need = float(sp.sum() * dt) * 1.15 + 1.0      # margin for spline stretch

    path = (_waypoints_h(rng, angle, need, start=start) if axis == "horizontal"
            else _waypoints(rng, angle, need, start=start))
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
                     "angle": angle, "axis": axis, "start": start,
                     "speed_scale": speed_scale, "duration": duration,
                     "dt": dt, "seed": seed},
        "path_len": float(s_path[-1]),
        "n_waypoints": int(len(path)),
    }


# --------------------------------------------------------------------------
# corpus -- a whole training dataset, declared by a yaml
# --------------------------------------------------------------------------
def output_root():
    """Where generated data goes. `$GNN_OUTPUT_ROOT`, else data_paths.json's
    cluster_data_dir -- the repo's one resolution order, not a new one."""
    root = os.environ.get("GNN_OUTPUT_ROOT")
    if root:
        return root
    import sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "src"))
    from connectome_gnn.utils import load_data_root_from_json
    return load_data_root_from_json()


def load_corpus_spec(path):
    """Read a corpus yaml and expand its condition grid.

    Returns the spec dict with an extra `condition_list` key: the cartesian
    product of `conditions`, in the yaml's own key order, so a run is
    reproducible from the file alone."""
    import itertools
    import yaml
    with open(path) as f:
        spec = yaml.safe_load(f)
    for k in ("name", "axis", "duration", "dt", "conditions", "splits"):
        if k not in spec:
            raise ValueError(f"{path}: missing required key {k!r}")
    keys = list(spec["conditions"])
    for k in keys:
        bad = [v for v in spec["conditions"][k] if v not in SPEC[k]]
        if bad:
            raise ValueError(f"{path}: conditions.{k} has {bad}, not in {SPEC[k]}")
    spec["condition_list"] = [dict(zip(keys, combo)) for combo in
                              itertools.product(*(spec["conditions"][k] for k in keys))]
    spec["_path"] = path
    return spec


def corpus_dir(spec, root=None):
    return os.path.join(root or output_root(), "graphs_data", "zebrafish",
                        spec["name"])


def build_corpus(spec, root=None, force=False, viz=True):
    """Write one npz per split (plus the png/mp4) and return {split: path}.

    Every trial is `generate(**condition, seed=seed0 + ci*100000 + k)`, so the
    seed offsets in the yaml are what keep splits disjoint -- a trial can only
    appear in two splits if two `seed0`s are within 100000*n_conditions of each
    other, which the shipped specs are not."""
    out = corpus_dir(spec, root)
    os.makedirs(out, exist_ok=True)
    conds = spec["condition_list"]
    duration, dt = float(spec["duration"]), float(spec["dt"])
    start = spec.get("start", "center")
    axis = spec["axis"]
    # `speed_scale` multiplies SPEED_UPS. It exists so a corpus can be paired
    # with a WIDER eye without also making the task faster: the corpus is in
    # grid units and the trainer maps the arena onto the eye's own reach, so a
    # plant with 1.8x the span traverses 1.8x the degrees per second on the
    # same corpus. Setting speed_scale = 1/1.8 there holds deg/s fixed, and
    # eye_G vs eye_GL then isolates span instead of confounding it with speed.
    sscale = float(spec.get("speed_scale", 1.0))

    paths = {}
    for split, s in spec["splits"].items():
        path = os.path.join(out, f"{split}.npz")
        n_per = int(s["n_per_cond"])
        if os.path.isfile(path) and not force:
            z = np.load(path, allow_pickle=False)
            if (z["u"].shape[0] == n_per * len(conds)
                    and abs(float(z["duration"]) - duration) < 1e-9
                    and str(z["axis"]) == axis
                    and abs(float(z.get("speed_scale", 1.0)) - sscale) < 1e-9):
                print(f"[corpus] reuse {path}  u{z['u'].shape}")
                paths[split] = path
                continue
        U, Y, C = [], [], []
        for ci, cond in enumerate(conds):
            for k in range(n_per):
                tr = generate(duration=duration, dt=dt, start=start, axis=axis,
                              speed_scale=sscale,
                              seed=int(s["seed0"]) + ci * 100000 + k, **cond)
                x = np.asarray(tr["x"], np.float32)
                y = np.asarray(tr["y"], np.float32)
                U.append(np.stack([np.gradient(x, dt), np.gradient(y, dt)], -1))
                Y.append(np.stack([x, y], -1))
                C.append(ci)
        U = np.asarray(U, np.float32)
        Y = np.asarray(Y, np.float32)
        np.savez_compressed(path, u=U, y=Y, cond=np.asarray(C, np.int64),
                            dt=dt, duration=duration, axis=axis,
                            speed_scale=sscale, conditions=json.dumps(conds))
        print(f"[corpus] {path}  u{U.shape} y{Y.shape}  {U.nbytes / 1e6:.0f} MB")
        paths[split] = path

    if viz:
        plot_corpus(spec, root)
        animate_corpus(spec, root)
    return paths


# --------------------------------------------------------------------------
# visualisation -- one static grid, one movie
# --------------------------------------------------------------------------
def _load_split_npz(spec, split, root=None):
    z = np.load(os.path.join(corpus_dir(spec, root), f"{split}.npz"),
                allow_pickle=False)
    return z["u"], z["y"], z["cond"], float(z["dt"])


def plot_corpus(spec, root=None, split="train", n_per_panel=6):
    """One panel per condition, `n_per_panel` trials drawn in each.

    For `axis: horizontal` the panel is x against TIME, not x against y -- a
    y that is identically zero plots as a flat line that says nothing, and
    what actually distinguishes the conditions there is the temporal profile.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    U, Y, C, dt = _load_split_npz(spec, split, root)
    conds = spec["condition_list"]
    horiz = spec["axis"] == "horizontal"
    ncol = len(spec["conditions"]["speed"]) * len(spec["conditions"]["angle"])
    nrow = int(np.ceil(len(conds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.5 * nrow),
                             facecolor="black")
    axes = np.atleast_1d(axes).ravel()
    t = np.arange(Y.shape[1]) * dt
    for ci, cond in enumerate(conds):
        ax = axes[ci]
        idx = np.where(C == ci)[0][:n_per_panel]
        for j, i in enumerate(idx):
            col = plt.cm.viridis(j / max(1, len(idx) - 1))
            if horiz:
                ax.plot(t, Y[i, :, 0], color=col, lw=0.8, alpha=0.9)
            else:
                ax.plot(Y[i, :, 0], Y[i, :, 1], color=col, lw=0.8, alpha=0.9)
        ax.set_facecolor("black")
        if horiz:
            ax.set_xlim(0, t[-1]); ax.set_ylim(-1, 1)
        else:
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.3")
        ax.text(0.03, 0.97, "/".join(str(cond[k]) for k in spec["conditions"]),
                transform=ax.transAxes, va="top", ha="left", color="white",
                fontsize=8)
    for ax in axes[len(conds):]:
        ax.axis("off")
    fig.suptitle("", color="white")
    fig.text(0.01, 0.99, f"{spec['name']}  ({spec['axis']}, {split} split, "
                         f"{n_per_panel} of {int((C == 0).sum())} trials/panel, "
                         f"{'x vs t' if horiz else 'x vs y'})",
             color="white", fontsize=11, va="top", ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = os.path.join(corpus_dir(spec, root), f"{spec['name']}.png")
    fig.savefig(png, dpi=140, facecolor="black")
    plt.close(fig)
    print(f"[corpus] {png}")
    return png


def animate_corpus(spec, root=None, split="train"):
    """A movie of a few trials, one per condition, drawn as a moving dot with
    a fading trail -- the thing a static path plot cannot show is the
    SCHEDULE, which is exactly what `motion` and `speed` control."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = \
            imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass                                   # a system ffmpeg may still exist

    viz = spec.get("viz", {})
    n_show = int(viz.get("n_trials_mp4", 12))
    fps = int(viz.get("fps", 30))
    trail_s = float(viz.get("trail_s", 1.5))

    U, Y, C, dt = _load_split_npz(spec, split, root)
    conds = spec["condition_list"]
    pick = [int(np.where(C == ci)[0][0])
            for ci in np.linspace(0, len(conds) - 1, n_show).astype(int)]
    horiz = spec["axis"] == "horizontal"
    T = Y.shape[1]
    trail = max(2, int(round(trail_s / dt)))

    ncol = int(np.ceil(np.sqrt(n_show)))
    nrow = int(np.ceil(n_show / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow),
                             facecolor="black")
    axes = np.atleast_1d(axes).ravel()
    t = np.arange(T) * dt
    dots, tails = [], []
    for j, i in enumerate(pick):
        ax = axes[j]
        ax.set_facecolor("black")
        if horiz:
            ax.set_xlim(0, t[-1]); ax.set_ylim(-1, 1)
        else:
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.3")
        ax.text(0.03, 0.97, "/".join(str(conds[int(C[i])][k])
                                     for k in spec["conditions"]),
                transform=ax.transAxes, va="top", ha="left", color="white",
                fontsize=7)
        (tl,) = ax.plot([], [], color="0.65", lw=1.0)
        (dt_,) = ax.plot([], [], "o", color="#ff4d4d", ms=6)
        tails.append(tl); dots.append(dt_)
    for ax in axes[n_show:]:
        ax.axis("off")
    fig.tight_layout()

    def frame(f):
        lo = max(0, f - trail)
        for j, i in enumerate(pick):
            if horiz:
                tails[j].set_data(t[lo:f + 1], Y[i, lo:f + 1, 0])
                dots[j].set_data([t[f]], [Y[i, f, 0]])
            else:
                tails[j].set_data(Y[i, lo:f + 1, 0], Y[i, lo:f + 1, 1])
                dots[j].set_data([Y[i, f, 0]], [Y[i, f, 1]])
        return tails + dots

    step = max(1, int(round((1.0 / fps) / dt)))
    ani = animation.FuncAnimation(fig, frame, frames=range(0, T, step),
                                  blit=True, interval=1000 / fps)
    mp4 = os.path.join(corpus_dir(spec, root), f"{spec['name']}.mp4")
    ani.save(mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2400),
             savefig_kwargs={"facecolor": "black"})
    plt.close(fig)
    print(f"[corpus] {mp4}")
    return mp4


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, opts in SPEC.items():
        p.add_argument(f"--{k}", default=opts[0], choices=opts)
    p.add_argument("--speed-scale", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--json", default=None, help="write the trace here")
    p.add_argument("--corpus", default=None,
                   help="build a whole dataset from this corpus yaml instead "
                        "of printing one trajectory")
    p.add_argument("--output-root", default=None,
                   help="overrides $GNN_OUTPUT_ROOT for --corpus")
    p.add_argument("--force", action="store_true",
                   help="rebuild --corpus splits even if the npz already match")
    p.add_argument("--no-viz", action="store_true",
                   help="skip the png and the mp4")
    a = p.parse_args()

    if a.corpus:
        spec = load_corpus_spec(a.corpus)
        print(f"[corpus] {spec['name']}: axis={spec['axis']}  "
              f"{len(spec['condition_list'])} conditions x "
              f"{ {k: v['n_per_cond'] for k, v in spec['splits'].items()} }  "
              f"duration={spec['duration']}s dt={spec['dt']:.6f}")
        build_corpus(spec, root=a.output_root, force=a.force,
                     viz=not a.no_viz)
        return

    tr = generate(a.shape, a.motion, a.speed, a.angle, a.duration, a.dt,
                  a.seed, a.start, a.axis, a.speed_scale)
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
