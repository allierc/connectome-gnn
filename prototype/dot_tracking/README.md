# prototype/dot_tracking — following a dot with a joystick

A scoped-down stand-in for the oculomotor problem: a target dot moves on a
`[-1, 1]` grid, a controller drives gaze to keep it centred, and the question
is how well that works as the target gets harder. It exists to sharpen the
problem statement before any of it is wired to a connectome.

```bash
python prototype/dot_tracking/app.py        # -> http://localhost:8000
python prototype/dot_tracking/app.py --port 8080
```

Standard library plus numpy. No Flask, no npm. VS Code forwards the port
automatically in a devcontainer; otherwise open the URL by hand.

## The three panels

| panel | shows | reads as |
|---|---|---|
| **world** | the `[-1, 1]` grid, the dot, the travelled path, the gaze marker | where things are |
| **retina** | the dot in gaze-centred coordinates, crosshair at the fovea | **can we hold the dot at the centre of the field of view?** |
| **joystick** | a red stick in a white gate, top view | what the controller is asking for |

The retina inset is the one that answers the question. If the dot sits under
the crosshair the target is foveated; if it drifts to the rim it is not; if it
leaves the `FOV = 0.6` circle it greys out and is labelled outside the field
of view. The strip below plots `|error|` against time with the FOV radius
dashed, so a transient miss during a sharp turn is distinguishable from a
standing lag during steady pursuit.

## The joystick is a rate control

Deflection **direction** is the direction the gaze is commanded to move;
deflection **magnitude** is the commanded speed, as a fraction of
`JOY_FULL_SCALE = 1.6` grid units per second. So the stick is the gaze
velocity, normalised and clipped to its square gate.

Deriving it from the gaze velocity rather than asking each controller for it
means any follower — including ones not written yet — gets a joystick trace
for free. When the stick hits the gate the box outlines in red and the sample
counts toward `stick saturated`: the controller is demanding more speed than
the stick can deliver, which is a different failure from simply aiming badly.

The alternative convention — stick deflection as absolute gaze *position* —
is a different plant with different failure modes, and would be worth trying
if rate control turns out not to match the biology.

## Trajectories

Four independent switches, in `trajectory.py`:

| switch | options | effect |
|---|---|---|
| `shape` | `segment`, `curve` | piecewise-linear waypoints, or Catmull-Rom through them |
| `motion` | `continue`, `stop_and_go` | constant travel, or move/pause alternation with raised-cosine ramps |
| `speed` | `slow`, `middle`, `fast` | 0.15, 0.40, 0.90 grid units per second |
| `angle` | `low`, `sharp` | 5–35° or 80–160° turns at the waypoints |

`shape` and `angle` decide the **path**; `speed` and `motion` decide the
**schedule** along it. Generation is therefore two-stage — lay down the path,
then walk it at the requested speed — which is why the dot travels at a
genuinely constant speed rather than at constant spline-parameter increment,
the usual artefact of animating a spline directly.

Headings reflect off the walls at `|x|, |y| = 0.95`, so the dot stays visible
without the path piling into a corner.

## Followers

In `followers.py`, one function each, registered by name:

- `fixed` — gaze pinned at the origin. The do-nothing baseline: the retinal
  error *is* the target, which sets the scale everything else is measured
  against.
- `lag` — first-order pursuit, `dg/dt = (target - g) / tau`, `tau = 0.18 s`.
  One knob, no prediction, no saccades: the simplest thing that could work.

On a fast, sharp-angle, curved target these separate cleanly — mean `|error|`
0.10 for `lag` against 0.62 for `fixed`, a factor of 6 — and `lag` saturates
the stick on 0.2 % of samples, at the sharpest turns only.

Add a controller by writing a function and decorating it with
`@register("name")`; it appears in the web selector automatically.

## Using it headlessly

`trajectory.py` and `followers.py` have no web dependency:

```python
from trajectory import generate
from followers import apply
tr = generate(shape="curve", angle="sharp", speed="fast", seed=1)
res = apply("lag", tr["t"], tr["x"], tr["y"])
print(res["err_mean"], res["joy_sat"])
```

```bash
python prototype/dot_tracking/trajectory.py --shape curve --angle sharp --json trace.json
```

## Where this is going

The controller is the part to replace. The intended sequence is: hand-written
baselines (here), then an optimised controller, then the 285-cell oculomotor
circuit of `config/zebrafish/zebrafish_om_intg_285_v1.yaml` driving the stick,
with the retinal error as its afferent input — which is what AF5 carries — and
`AMN`/`AIN` rates as the horizontal command.
