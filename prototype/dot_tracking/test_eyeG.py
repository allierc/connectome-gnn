#!/usr/bin/env python
"""test_eyeG -- one movie: the target, the circuit, and eye G moving.

    python test_eyeG.py --tag eyeG_light
    python test_eyeG.py --tag eyeG_deep --duration 8 --fps 30

A fixed two-part test sequence, so two runs are comparable frame by frame:

    0 - T s     continue,     middle speed      smooth pursuit
    T - 2T s    stop_and_go,  middle speed      the same speed, chopped into
                                                move / pause, which is what makes
                                                an integrator's leak visible

The two halves are joined in VELOCITY, not position, and the target position is the
integral of the join — so the target never jumps, and the circuit's state carries
across the transition rather than being reset. That transition is the interesting
frame of the movie: the same controller meeting a regime it was trained on but has
not been in for eight seconds.

THREE PANELS

  left    the world, in degrees. White is the target. Red is the COMMAND, meaning
          the equilibrium the muscles are asking for -- where the eye would settle
          if the drive froze -- and blue is the GAZE, where the eye actually is.
          Red on blue does not mean the eye is doing nothing: a second-order eye has
          unity DC gain, so it does not shrink a slow target, it delays it by
          2 zeta / omega_n. See section 5.1 of the oculomotor note.

  middle  the circuit at work, in the pixel layout of the web interface: the two
          velocity inputs, the recurrent rates as a square, and the non-negative
          motor drives -- four synergies or six muscles, whichever eye is loaded.

  right   eye G itself. The geometry is the real scanned one, straight out of
          `archive/eye_G/baseline_curves.npz`: the globe shell and the six straps as
          the Blender model placed them. THE MPM IS NOT IN THE LOOP -- it cannot be,
          that is why the reduced model exists -- so the globe is rotated rigidly by
          the model's (theta, phi, psi) and each strap follows it in proportion to
          its arc length, anchored at the orbit and carried at the insertion. Muscle
          brightness is the drive the circuit is sending. It is a picture of the
          model's output on the real anatomy, not a simulation of the tissue.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
# A leftover DISPLAY from the host/VS Code (e.g. DISPLAY=:3) makes VTK attempt a
# real X connection it has no authority for -- off_screen=True still renders fine
# either way (it falls back to a software context), but Xlib prints "Authorization
# required..." to stderr before that fallback kicks in. Clearing DISPLAY here, before
# vtk/pyvista is imported anywhere (including inside render_surface_vtk), skips the
# real-X attempt entirely and removes that line at the source; the VTK-internal
# "bad X server connection" WARN is a separate log channel, silenced below once vtk
# is actually importable. Neither call changes what gets rendered.
os.environ.pop("DISPLAY", None)
try:
    import vtk as _vtk
    _vtk.vtkObject.GlobalWarningDisplayOff()
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS_EYE = "/workspace/Plexus/prototype/eye"
sys.path.insert(0, HERE)
sys.path.insert(0, PLEXUS_EYE)
from trajectory import generate, SPEED_UPS               # noqa: E402
import train_eyeG as TG                                  # noqa: E402

EYE_DIR = TG.EYE_DIR
MUS_COLOR = {"LR": (0.30, 0.55, 0.95), "SR": (0.85, 0.25, 0.25),
             "MR": (0.90, 0.75, 0.25), "IR": (0.35, 0.75, 0.45),
             "SO": (0.60, 0.40, 0.80), "IO": (0.90, 0.55, 0.25)}
BG = "#000000"
# One type scale for every panel: axis name / tick or category label /
# everything else. Before this the same kind of label was 13pt in the world
# panel and 8.5pt in the circuit panel, which read as two figures side by side.
FS_AXIS, FS_TICK, FS_NOTE = 11.0, 9.0, 9.0
# Figure 1c's E/I key (scripts/plot_oculomotor_connectome._EI_COLOR): BLUE
# excitatory, RED inhibitory. Figure 1c is a paper figure on white and uses
# RdBu, whose centre is white; on this movie's black background that would
# hand the panel to the 94% of pairs with no synapse and invert the figure.
# So: the same two endpoint colours, centred on black instead.
EI_E, EI_I = "#1f4fd8", "#d81b26"
from matplotlib.colors import LinearSegmentedColormap as _LSC
EI_CMAP = _LSC.from_list("ei_black",
                         [EI_I, "#3a0a0e", "#000000", "#0a1440", EI_E])
# NaN = not applicable (a cell with no path to a muscle, a not-yet-filled
# kinograph column) must read as background, not as the zero colour.
EI_CMAP.set_bad(BG)


# ---------------------------------------------------------------------------
# the test sequence
# ---------------------------------------------------------------------------
# Four regimes back to back: fast first, because a controller that survives fast
# smooth pursuit and fast stop-and-go and then meets the slower pair has been asked
# the hard question first, and the two halves are directly comparable.
# Circles FIRST, and centred on the origin: a target whose direction turns at a
# constant rate and never reverses is the cleanest probe there is, and putting it
# before the corpus regimes means it is seen from a rested state rather than after
# 32 s of accumulated drift.
CIRCLE_DEG = 5.0

# Saccades: the target jumps left/right at a fixed rate and holds between jumps.
# Six rates spanning the eye's own corner frequency, which for eye G is
# wn/2pi = 0.9 to 1.2 Hz -- so the slow end is well inside the regime where the eye
# is a pure delay and the fast end is past its resonance, and the sequence walks
# the controller across that boundary.
# Geometric from 0.5 to 4 Hz. 0.25 was dropped as uninformative -- two seconds
# between jumps is long enough for any of these controllers -- and the ratio is
# constant so the six rates sample the eye's corner frequency evenly in log f.
SACCADE_HZ = [0.5, 0.75, 1.15, 1.75, 2.65, 4.0]
SACCADE_DEG = 5.0

PHASES = [("circle_cw", "middle"), ("circle_ccw", "middle"),
          ("continue", "fast"), ("stop_and_go", "fast"),
          ("continue", "middle"), ("stop_and_go", "middle")]


def _saccade(duration, dt, scale, hz, amp_deg=SACCADE_DEG, rise=0.06):
    """Left/right saccades at `hz`, as velocity.

    A saccade is a step in angle, so in velocity it is a brief boxcar: the target
    is carried across in `rise` seconds and then held. Amplitude is in DEGREES and
    converted through the horizontal scale, so every frequency covers the same
    angular excursion and only the rate changes.

    The first jump goes to $+A/2$ and the rest alternate about the centre, so the
    train has zero mean and the target does not walk off across the phase.
    """
    n = int(round(duration / dt))
    nr = max(1, int(round(rise / dt)))
    v = np.zeros((n, 2))
    A = amp_deg / scale[0]                       # grid units
    period = int(round(1.0 / (hz * dt)))
    here, k, first = -A / 2, 0, True
    while k + nr < n:
        target = A / 2 if here < 0 else -A / 2
        if first:
            v[k:k + nr, 0] = (A / 2) / rise      # centre -> +A/2
            here, first = A / 2, False
        else:
            v[k:k + nr, 0] = (target - here) / rise
            here = target
        k += max(period, nr + 1)
    # Every phase must end where it began. Concatenating phases joins VELOCITIES,
    # so a phase with an odd number of jumps leaves a net displacement and the
    # target walks a half-amplitude further off centre at each change of rate --
    # which is what the first version did, ending 17 deg out.
    if abs(here) > 1e-9 and n - nr - 1 > 0:
        v[n - nr - 1:n - 1, 0] = -here / rise
    return v


def _circle(duration, dt, scale, ccw, radius_deg=CIRCLE_DEG, laps=1.0, lead=0.75):
    """A circle of `radius_deg` IN GAZE ANGLE, as velocity, centred on the origin.

    Not one of `trajectory.SPEC`'s shapes, and deliberately so: the corpus is
    piecewise-straight or spline, and a circle is the one target whose direction
    turns at a constant rate and never reverses. Running it both ways separates a
    controller that integrates velocity from one that has learned this corpus's turn
    statistics.

    The radius is given in DEGREES and converted per axis, so it is a true circle in
    the space the error is measured in rather than in grid units -- on an eye whose
    horizontal and vertical reach differ, a circle in grid units is an ellipse in
    gaze. The first `lead` seconds run radially out from the centre to the circle's
    start: the harness integrates from the origin, so a circle begun there would be
    tangent to it and would swing out to twice its radius, which does not fit. Only
    the FIRST circle gets that lead -- `laps` is a whole number, so each circle ends
    where it began and the next one continues from there; giving the second a lead of
    its own would step it out by another radius and leave the arena.
    """
    n = int(round(duration / dt))
    nl = int(round(lead / dt))
    rx, ry = radius_deg / scale[0], radius_deg / scale[1]        # grid units per axis
    v = np.zeros((n, 2))
    if nl:
        v[:nl] = (rx / lead, 0.0)                                # out to (rx, 0)
    w = 2 * np.pi * laps / max(duration - lead, 1e-6)
    sgn = 1.0 if ccw else -1.0
    th = sgn * w * (np.arange(n - nl) * dt)
    v[nl:] = np.stack([-rx * w * sgn * np.sin(th),
                       ry * w * sgn * np.cos(th)], -1)
    return v


def saccade_sequence(duration, dt, scale):
    """Six saccade rates back to back, one phase each. No steering and no seed
    search: every phase is centred on the origin by construction, so the target
    cannot wander out of the arena the way the corpus regimes can."""
    segs = [_saccade(duration, dt, scale, f) for f in SACCADE_HZ]
    v = np.concatenate(segs, 0).astype(np.float32)
    n0 = len(segs[0])
    return v, (np.cumsum(v, 0) * dt).astype(np.float32), \
        [n0 * (i + 1) for i in range(len(SACCADE_HZ) - 1)]


def sequence(duration, dt, scale, seed=0, bound=0.95, tries=200):
    """The four regimes of PHASES, joined in VELOCITY.

    Joining velocities rather than positions means the target never jumps at a
    boundary and the circuit's state carries across, so each transition is a real
    test: the same controller meeting a regime it has not been in for `duration`.

    Retries the seed until the integrated target stays inside the arena, rather than
    clipping it: clipping would distort the velocity, which is the only thing the
    circuit is given, and a distorted input is not the regime we mean to test.
    """
    for k in range(tries):
        segs, n0 = [], None
        for i, (mo, sp) in enumerate(PHASES):
            if mo.startswith("circle"):
                first = not any(PHASES[j][0].startswith("circle")
                                for j in range(i))
                segs.append(_circle(duration, dt, scale, ccw=mo.endswith("ccw"),
                                    lead=0.75 if first else 0.0))
            else:
                q = generate(shape="curve", motion=mo, speed=sp, angle="low",
                             duration=duration, dt=dt, seed=seed + 1000 * k + i,
                             start="center")
                segs.append(np.stack([np.gradient(np.asarray(q["x"]), dt),
                                      np.gradient(np.asarray(q["y"]), dt)], -1))
            n0 = len(segs[0])
        # Six phases of concatenated velocity is 48 s of walk and it leaves the arena
        # every time. Each phase after the first is ROTATED so its net displacement
        # points back toward the centre: a rotation preserves speed, turn rate and
        # every other statistic of the regime exactly -- it changes only the heading --
        # whereas clipping the position would distort the velocity, which is the one
        # thing the circuit is given.
        # Rotating a phase so its NET displacement points home fixes where it ends
        # and not where it goes: a phase can bulge to 0.9 units on the way. So the
        # angle is chosen by search -- 72 rotations, keep the one whose whole path
        # stays closest in. Rotation is still an isometry, so speed, turn rate and
        # the stop/go timing are untouched whichever angle wins; only the heading
        # changes, and only the heading was ever the problem.
        pos = np.zeros(2)
        for i, seg in enumerate(segs):
            if i and not PHASES[i][0].startswith("circle"):
                best, best_r = None, np.inf
                for a in np.linspace(0, 2 * np.pi, 72, endpoint=False):
                    c_, s_ = np.cos(a), np.sin(a)
                    cand = seg @ np.array([[c_, s_], [-s_, c_]])
                    reach = np.abs(pos + np.cumsum(cand, 0) * dt).max()
                    if reach < best_r:
                        best, best_r = cand, reach
                segs[i] = best
            pos = pos + segs[i].sum(0) * dt
        v = np.concatenate(segs, 0).astype(np.float32)
        xy = np.cumsum(v, 0) * dt
        if np.abs(xy).max() <= bound:
            return v, xy.astype(np.float32), [n0 * (i + 1)
                                              for i in range(len(PHASES) - 1)]
    raise SystemExit(f"[seq] no seed within {tries} kept the target inside "
                     f"+-{bound}; lower --duration or raise the bound")


# ---------------------------------------------------------------------------
# eye G geometry, for the right-hand panel
# ---------------------------------------------------------------------------
def load_geometry(eye_dir=EYE_DIR, stride=1):
    """Rest geometry of eye G, plus the tissue label that makes rotation visible.

    A globe drawn as a uniform cloud of shell points is ROTATIONALLY INVARIANT to
    look at: spin it and nothing changes, which is why the first version of this
    panel appeared frozen. `tissue` labels the cornea/iris patch separately (1482 of
    13347 points), and drawing that patch in its own colour gives the globe a pupil
    to move -- the same reason a real eye's rotation is visible.
    """
    z = np.load(os.path.join(eye_dir, "baseline_curves.npz"), allow_pickle=True)
    g = dict(shell=np.asarray(z["shell"][0], np.float64)[::stride],
             tissue=np.asarray(z["tissue"], int)[::stride],
             mus=np.asarray(z["mus_pos"][0], np.float64)[::stride],
             parent=np.asarray(z["mus_parent"], int)[::stride],
             s=np.asarray(z["mus_s"], np.float64)[::stride],
             centre=np.asarray(z["centre"][0], np.float64))
    g["mus_vm"] = np.asarray(z["mus_vm"][0], np.float64)[::stride]
    g["s"] = (g["s"] - g["s"].min()) / max(1e-9, float(np.ptp(g["s"])))
    # the smallest tissue class on the shell is the cornea/iris cap
    lab, cnt = np.unique(g["tissue"], return_counts=True)
    g["cornea"] = g["tissue"] == lab[np.argmin(np.where(cnt > 10, cnt, 1 << 30))]
    return g


def rot(theta, phi, psi):
    """(horizontal, vertical, torsion) in degrees -> 3x3, in the plant's frame:
    +x caudal, +y dorsal, +z the optic axis. Horizontal is about the dorsal axis,
    vertical about the caudal one, torsion about the optic axis."""
    t, p, s = np.radians([theta, phi, psi])
    Ry = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(s), -np.sin(s), 0], [np.sin(s), np.cos(s), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class _RotSeq:
    """Frame k of a rigidly-turned point set, computed on demand.

    `SurfaceScene` indexes its capture as `cap["shell"][k]`, so a capture can be
    anything that answers to that. Materialising 480 frames of 13 347 shell points
    would be 150 MB for no reason; this returns the rotation when asked.
    """

    def __init__(self, rest, centre, ang, frac=None):
        self.base = np.asarray(rest, float) - centre
        self.c, self.ang, self.frac = centre, np.asarray(ang, float), frac

    def __len__(self):
        return len(self.ang)

    def __getitem__(self, k):
        turned = self.base @ rot(*self.ang[int(k)]).T
        if self.frac is None:
            return turned + self.c
        return self.base + self.frac * (turned - self.base) + self.c


class _Const:
    def __init__(self, v, n):
        self.v, self.n = np.asarray(v), n

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        return self.v


class SurfaceView:
    """The right panel drawn by the Plexus surface renderer itself.

    `render_surface_vtk.SurfaceScene` skins the Blender meshes to the run's particles
    -- translucent globe, solid lens, six smooth straps, a grey arrow at the rest
    optic axis and a yellow one carried by the gaze. That is the render that made
    `archive/eye_G/pairs_surface.png`, and reproducing it here rather than
    approximating it means importing it, not rewriting it.

    What it is given is a synthetic capture: eye G's rest particles turned rigidly by
    the model's (theta, phi, psi), straps following by arc length. So the SURFACES and
    the lighting are the real renderer's; only the motion is the reduced model's, and
    the honest version -- the MPM run on the controller's own command -- is `--compose`.
    """

    def __init__(self, geo, angles, cmd, act, dt, size=(620, 620), az=25.0,
                 hud=True):
        import render_surface_vtk as RS
        c, T = geo["centre"], len(angles)
        pad = np.zeros((T, 6))
        pad[:, :act.shape[1]] = act[:, :6]
        cap = {"shell": _RotSeq(geo["shell"], c, angles),
               "mus_pos": _RotSeq(geo["mus"], c, angles, frac=geo["s"][:, None]),
               "centre": _Const(c, T), "tissue": geo["tissue"],
               "mus_parent": geo["parent"], "act": pad,
               "gaze": np.asarray(angles), "target": np.asarray(cmd),
               "frame": np.arange(T)}
        self.scene = RS.SurfaceScene(cap, side="R", size=size, globe_alpha=0.20)
        if not hud:
            # `SurfaceScene._text` draws the frame/gaze HUD and the activation
            # legend with pyvista's add_text, i.e. in VTK's own bitmap font at
            # its own sizes, while every other label in this figure is
            # matplotlib's. Rather than port two panels of axes, ticks and
            # colormaps to VTK for the sake of one font, suppress those two
            # actors and redraw the same text in matplotlib. Patched here and
            # not in Plexus: render_surface_vtk is another repo's renderer and
            # this is a preference of THIS figure, not a fix to it.
            _add = self.scene.p.add_text

            def _skip_named(*args, **kw):
                if kw.get("name") in ("hud", "legend"):
                    return None
                return _add(*args, **kw)
            self.scene.p.add_text = _skip_named
        self.az, self.dt = az, dt
        self.angles, self.cmd, self.act = np.asarray(angles), np.asarray(cmd), pad

    def frame(self, k):
        return np.asarray(self.scene.frame(int(k), self.az, self.dt))

    def close(self):
        self.scene.close()


class EyeView:
    """Eye G's own geometry, moved by the model. Three ways to draw it:

        vtk         reconstructed SURFACES -- translucent globe, solid cornea cap,
                    six solid straps. The default, and the one that reads.
        mpm         the material points themselves, cornea picked out
        mpm_stress  the same points coloured by von Mises stress at rest

    The surfaces are reconstructed ONCE, at rest, and then their vertices are
    rotated: the globe turns rigidly and every strap follows by its arc length,
    anchored at the orbit and carried at the insertion. Nothing is re-extracted per
    frame, so a frame costs one matrix multiply and a screenshot.
    """

    MODES = ("surface", "vtk", "mpm", "mpm_stress")

    def __init__(self, geo, mode="vtk", size=(560, 560), muscles=TG.MUSCLES):
        import pyvista as pv
        pv.OFF_SCREEN = True
        self.pv, self.geo, self.mode, self.muscles = pv, geo, mode, muscles
        self.pl = pv.Plotter(off_screen=True, window_size=list(size))
        self.pl.set_background("black")
        self.parts = []                       # (mesh, rest points, arc fraction)

        def add(points, **kw):
            if mode == "vtk":
                m = pv.PolyData(points).reconstruct_surface(nbr_sz=14, sample_spacing=None)
                self.pl.add_mesh(m, smooth_shading=True, **kw)
            else:
                m = pv.PolyData(points)
                self.pl.add_mesh(m, point_size=kw.pop("point_size", 4.0),
                                 render_points_as_spheres=True, **kw)
            return m

        sh, cor, c = geo["shell"], geo["cornea"], geo["centre"]
        m = add(sh[~cor], color=(0.55, 0.57, 0.62), opacity=0.30 if mode == "vtk" else 0.25,
                point_size=3.0)
        self.parts.append((m, sh[~cor] - c, None))
        m = add(sh[cor], color=(0.92, 0.94, 0.98), opacity=1.0, point_size=5.0)
        self.parts.append((m, sh[cor] - c, None))          # the cornea: the landmark
        self.mus_actors = []
        for k, key in enumerate(muscles):
            sel = geo["parent"] == k
            if mode == "mpm_stress":
                pd = pv.PolyData(geo["mus"][sel])
                pd["vm"] = geo["mus_vm"][sel]
                self.pl.add_mesh(pd, scalars="vm", cmap="inferno", point_size=4.5,
                                 render_points_as_spheres=True, show_scalar_bar=False)
                mm = pd
            else:
                mm = add(geo["mus"][sel], color=MUS_COLOR[key], opacity=0.95,
                         point_size=4.5)
            self.parts.append((mm, geo["mus"][sel] - c, geo["s"][sel][:, None]))
            self.mus_actors.append(list(self.pl.renderer.actors)[-1])
        # A 5-degree rotation of a globe is barely visible; the optic axis drawn as
        # an arrow is not, and it is what the Plexus renderer draws for the same reason.
        r = float(np.linalg.norm(sh - c, axis=1).max())
        self.gaze_rest = np.array([0.0, 0.0, 1.0]) * r * 1.55
        self.arrow = pv.Arrow(start=c, direction=self.gaze_rest,
                              scale=float(np.linalg.norm(self.gaze_rest)),
                              tip_length=0.22, tip_radius=0.055, shaft_radius=0.018)
        self.arrow_rest = self.arrow.points - c
        # grey arrow FIXED at the rest optic axis, yellow arrow carried by the gaze:
        # the pair is what makes a five-degree rotation readable, and it is the
        # convention `render_surface_vtk` already uses in the Plexus renders.
        ref = pv.Arrow(start=c, direction=self.gaze_rest,
                       scale=float(np.linalg.norm(self.gaze_rest)),
                       tip_length=0.22, tip_radius=0.045, shaft_radius=0.012)
        self.pl.add_mesh(ref, color=(0.55, 0.55, 0.58), opacity=0.55)
        self.pl.add_mesh(self.arrow, color=(0.95, 0.80, 0.25))
        self.parts.append((self.arrow, self.arrow_rest, None))
        self.pl.camera_position = "xy"
        self.pl.camera.azimuth = 25
        self.pl.camera.elevation = 12
        self.pl.camera.zoom(1.45)
        self.pl.screenshot(return_img=True)

    def frame(self, angles, drive):
        R = rot(*angles); c = self.geo["centre"]
        for mesh, rest, frac in self.parts:
            turned = rest @ R.T
            mesh.points = (rest + frac * (turned - rest) + c) if frac is not None \
                else turned + c
        if self.mode != "mpm_stress":
            for k, name in enumerate(self.mus_actors):
                a = float(np.clip(drive[k] if k < len(drive) else 0.0, 0, 1))
                self.pl.renderer.actors[name].prop.opacity = 0.35 + 0.6 * a
        self.pl.render()          # screenshot() alone does NOT pick up new points:
        return np.asarray(self.pl.screenshot(return_img=True))   # this is the movie

    def close(self):
        self.pl.close()


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------
def build_figure(reach, hidden, n_act, act_names, img0, title,
                 conn=None, conn_blocks=None, cmd_trail=True):
    """The three panels. `conn` swaps the middle one's content.

    Without it (the free ctRNN) the middle panel is the web interface's pixel
    layout: inputs, the recurrent rates as a square, the motor drives. There
    is nothing else to show -- a free (hidden, hidden) matrix has no structure
    worth a picture.

    With it (a connectome-constrained circuit) the same panel shows the
    CONNECTIVITY, `conn` being an (N, N) signed post-x-pre array already
    scaled for display, with the live rates as a strip beneath it. That is the
    substantive difference between the two models, so it is the thing the
    middle panel should carry: for the zebrafish circuit the matrix is the
    measured connectome and the square of rates is a reshape with no meaning
    (285 is not a square, and neighbouring cells in it are not neighbours).
    `conn_blocks` is [(name, first_row), ...] for the cell-type boundaries.
    """
    fig = plt.figure(figsize=(18.5, 7.8) if conn is not None else (16.0, 5.6),
                     facecolor=BG)
    # The middle panel is wider when it carries the connectome: it has a
    # 285x285 matrix plus three neuron-group vectors to fit, where the free
    # ctRNN's version is one small square.
    if conn is not None:
        gs = fig.add_gridspec(2, 3, width_ratios=[0.92, 1.45, 0.92],
                              height_ratios=[1.0, 0.62], wspace=0.11,
                              hspace=0.11, left=0.05, right=0.985,
                              top=0.975, bottom=0.075)
    else:
        gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.75, 1.0], wspace=0.16,
                              left=0.05, right=0.985, top=0.965, bottom=0.10)
    ax_w, ax_c, ax_e = (fig.add_subplot(gs[0, k]) for k in range(3))
    if conn is not None:
        # The world panel is `aspect="equal"` on a symmetric range, so its
        # square is sized by the axes HEIGHT and fills the row -- which left it
        # visibly larger than the matrix beside it, and the two read as
        # different scales rather than as one figure. Narrowing the column
        # cannot fix that (it only adds side padding), so shrink the axes box
        # itself until the square matches the matrix's drawn height.
        _p = ax_w.get_position()
        _k = 0.66
        ax_w.set_position([_p.x0, _p.y0 + _p.height * (1 - _k) / 2,
                           _p.width, _p.height * _k])
    for ax in (ax_w, ax_c, ax_e):
        ax.set_facecolor(BG)

    # --- world -----------------------------------------------------------
    lim_h, lim_v = reach[0] * 1.15, reach[1] * 1.15
    ax_w.set_xlim(-lim_h, lim_h); ax_w.set_ylim(-lim_v, lim_v)
    ax_w.set_aspect("equal")
    ax_w.axhline(0, color="#555", lw=0.8); ax_w.axvline(0, color="#555", lw=0.8)
    ax_w.add_patch(plt.Rectangle((-reach[0], -reach[1]), 2 * reach[0], 2 * reach[1],
                                 fill=False, ec="#ffffff", ls=":", lw=1.0,
                                 alpha=0.45))
    _ = lambda *_a, **_k: None
    _(-reach[0], reach[1] * 1.03, f"eye reach  ±{reach[0]:.1f}° h / "
              f"±{reach[1]:.1f}° v", color="#ffffff", fontsize=10, va="bottom")
    for s in ax_w.spines.values():
        s.set_color("#444")
    ax_w.tick_params(colors="#ffffff", labelsize=FS_TICK)
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    step = 2.0 if lim_h < 12 else 5.0
    ax_w.xaxis.set_major_locator(MultipleLocator(step))
    ax_w.yaxis.set_major_locator(MultipleLocator(step))
    deg = FuncFormatter(lambda t, _: f"{t:.0f}°")
    ax_w.xaxis.set_major_formatter(deg); ax_w.yaxis.set_major_formatter(deg)
    ax_w.set_xlabel("θ  horizontal gaze (deg)", color="#ffffff", fontsize=FS_AXIS)
    ax_w.set_ylabel("φ  vertical gaze (deg)", color="#ffffff", fontsize=FS_AXIS)
    # The right panel looks AT the eye, so its left is the eye's right. Mirroring
    # both axes here makes a rightward excursion move rightward in both panels;
    # only the display is flipped, never the data.
    ax_w.invert_xaxis(); ax_w.invert_yaxis()
    trail_t, = ax_w.plot([], [], "-", color="#ffffff", lw=1.0, alpha=0.45)
    # red is the command, which is context for the two traces that matter; it is
    # drawn back so the white target and the blue gaze read first
    # The command's TRAIL is context, not a result, and on a horizontal task it
    # simply doubles the gaze trace. `cmd_trail=False` keeps the red dot -- the
    # instantaneous equilibrium the muscles are asking for, which is worth
    # seeing lead the gaze -- and drops the history behind it.
    trail_c, = ax_w.plot([], [], "-", color="#e05a4a", lw=1.0,
                         alpha=0.35 if cmd_trail else 0.0)
    trail_g, = ax_w.plot([], [], "-", color="#4da3ff", lw=1.6, alpha=0.95)
    # the target is a RING, not a disc: the gaze dot sits inside it when the
    # controller is working, and a filled marker would simply hide the thing the
    # movie exists to show.
    dot_t, = ax_w.plot([], [], "o", mfc="none", mec="#ffffff", mew=1.8, ms=15)
    dot_c, = ax_w.plot([], [], "o", color="#e05a4a", ms=6, mec="none", alpha=0.5)
    dot_g, = ax_w.plot([], [], "o", color="#4da3ff", ms=9, mec="none")

    # --- circuit ---------------------------------------------------------
    # Left to right, in the direction the signal travels, so this panel reads the
    # same way as the world panel beside it and as Figure 1e of the note.
    side = int(np.ceil(np.sqrt(hidden)))
    ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1); ax_c.axis("off")
    _in_ext = (0.015, 0.088, 0.50, 0.74) if conn is not None \
        else (0.015, 0.10, 0.36, 0.64)
    im_in = ax_c.imshow(np.zeros((2, 1)),
                        cmap=EI_CMAP if conn is not None else "viridis",
                        vmin=-1, vmax=1, extent=_in_ext, aspect="auto", zorder=3)
    im_out = ax_c.imshow(np.zeros((n_act, 1)), cmap="viridis", vmin=0, vmax=1,
                         extent=(0.90, 0.985, 0.28, 0.72), aspect="auto", zorder=3)
    im_rate = None
    if conn is None:
        im_rec = ax_c.imshow(np.zeros((side, side)), cmap="viridis", vmin=-1, vmax=1,
                             extent=(0.235, 0.765, 0.235, 0.765), aspect="auto", zorder=3)
        ax_c.text(0.5, 0.185, f"{hidden} rates   $r=\\tanh v$", color="#ffffff",
                  fontsize=FS_NOTE, ha="center", va="top")
        mid_l, mid_r = 0.235, 0.765
    else:
        n = conn["n"]
        n_in, n_intg = conn["n_in"], conn["n_intg"]
        n_out_cells = n - n_in - n_intg
        mid_l, mid_r, lo, hi = 0.320, 0.712, 0.300, 0.880
        cmap_ei = EI_CMAP
        # The matrix, in Figure 1c's own terms: post x pre, sign taken from the
        # PRESYNAPTIC column's Dale assignment, blue excitatory and red
        # inhibitory. What differs is the intensity -- Figure 1c shows the
        # static synapse weight, this shows |W_ij r_j|, the MESSAGE actually
        # crossing that synapse this frame, so a still of the movie is the
        # circuit's traffic rather than its wiring.
        im_rec = ax_c.imshow(np.zeros((n, n)), cmap=cmap_ei, vmin=-1, vmax=1,
                             extent=(mid_l, mid_r, lo, hi), aspect="auto",
                             zorder=3, interpolation="nearest")
        # Rules on the block BOUNDARIES, labels at the block CENTRES. A label
        # on the boundary names the row where a type starts, which reads as if
        # it belonged to the type above it; centred, it names the band it
        # actually spans.
        _blocks = list(conn_blocks or [])
        _edges = [r0 for _, r0 in _blocks] + [n]
        last_y = np.inf
        for i, (nm, r0) in enumerate(_blocks):
            y = hi - (hi - lo) * r0 / n
            x = mid_l + (mid_r - mid_l) * r0 / n
            if i:                                   # no rule on the outer edge
                ax_c.plot([mid_l, mid_r], [y, y], color="#777", lw=0.4,
                          alpha=0.45, zorder=4)
                ax_c.plot([x, x], [lo, hi], color="#777", lw=0.4, alpha=0.45,
                          zorder=4)
            y_c = hi - (hi - lo) * 0.5 * (r0 + _edges[i + 1]) / n
            if last_y - y_c > 0.026:
                ax_c.text(mid_l - 0.008, y_c, nm, color="#ccc", fontsize=FS_TICK,
                          ha="right", va="center")
                last_y = y_c
        ax_c.add_patch(plt.Rectangle((mid_l, lo), mid_r - mid_l, hi - lo,
                                     fill=False, ec="#777", lw=0.6, zorder=5))

        # THE THREE GROUPS, EACH BESIDE ITS OWN ROWS. A rate is produced at the
        # cell's ROW -- it is the postsynaptic sum that drives it -- so a vector
        # drawn level with its rows can be read straight across into the matrix.
        # The ordering is afferent -> recurrent -> output, so the three blocks
        # are contiguous and together they cover all n rows exactly once: AF5 on
        # the left where the velocity arrives, then the integrator and the motor
        # cells stacked on the right in the order the signal reaches them.
        def _rows(a0, a1):
            return hi - (hi - lo) * a1 / n, hi - (hi - lo) * a0 / n

        y_in0, y_in1 = _rows(0, n_in)
        y_intg0, y_intg1 = _rows(n_in, n_in + n_intg)
        y_out0, y_out1 = _rows(n_in + n_intg, n)
        y_in_c = 0.5 * (y_in0 + y_in1)
        y_intg_c = 0.5 * (y_intg0 + y_intg1)
        y_out_c = 0.5 * (y_out0 + y_out1)

        def _col(x0, x1, k, y0, y1):
            return ax_c.imshow(np.zeros((k, 1)), cmap=cmap_ei, vmin=-1, vmax=1,
                               extent=(x0, x1, y0, y1), aspect="auto",
                               zorder=3, interpolation="nearest")
        im_cin = _col(0.150, 0.190, n_in, y_in0, y_in1)
        im_intg = _col(0.727, 0.767, n_intg, y_intg0, y_intg1)
        im_cout = _col(0.727, 0.767, n_out_cells, y_out0, y_out1)
        for x0, x1, y0, y1 in ((0.150, 0.190, y_in0, y_in1),
                               (0.727, 0.767, y_intg0, y_intg1),
                               (0.727, 0.767, y_out0, y_out1)):
            ax_c.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                         fill=False, ec="#777", lw=0.5, zorder=5))
        ax_c.text(0.170, y_in1 + 0.010, f"AF5 ({n_in})", color="#ffffff",
                  fontsize=FS_NOTE, ha="center", va="bottom")
        ax_c.text(0.775, y_intg_c, f"INTG ({n_intg})", color="#ffffff",
                  fontsize=FS_NOTE, ha="left", va="center")
        # below its own column: to its right is where the arrow to the
        # muscle drives runs, and the label sat on top of it.
        ax_c.text(0.747, y_out0 - 0.010, f"AMN/AIN ({n_out_cells})",
                  color="#ffffff", fontsize=FS_NOTE, ha="center", va="top")
        ax_c.text(mid_l - 0.175, 0.5 * (lo + hi), "postsynaptic",
                  color="#ddd", fontsize=FS_AXIS, ha="center", va="center",
                  rotation=90)
        ax_c.text(0.5, lo - 0.022, "presynaptic", color="#ddd",
                  fontsize=FS_AXIS, ha="center", va="top")
        # velocity -> AF5 -> circuit -> AMN/AIN -> drives, each arrow level
        # with the blocks it joins
        for x0, x1, yy in ((0.126, 0.146, y_in_c),
                           (mid_r + 0.004, 0.723, y_out_c),
                           (0.771, 0.854, y_out_c)):
            ax_c.annotate("", xy=(x1, yy), xytext=(x0, yy),
                          arrowprops=dict(arrowstyle="-|>", color="#ffffff",
                                          lw=1.1))
        _yc_in, _yc_out = y_in_c, y_out_c
    if conn is None:
        for x0, x1, y in ((0.10, mid_l, 0.5), (mid_r, 0.90, 0.5)):
            ax_c.annotate("", xy=(x1, y), xytext=(x0, y),
                          arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=1.3))
        ax_c.text(0.058, 0.30, r"$(\dot x,\dot y)$", color="#ffffff",
                  fontsize=FS_NOTE, ha="center", va="top")
        ax_c.text(0.943, 0.235, f"$m=[\\hat W^{{out}}r]_+$", color="#ffffff",
                  fontsize=FS_NOTE, ha="center", va="top")
        out_lo, out_hi = 0.28, 0.72
    else:
        # centre the velocity pair on AF5's rows and the drives on AMN/AIN's,
        # so all five stages of the flow line up with what they feed.
        im_in.set_extent((0.082, 0.122, _yc_in - 0.055, _yc_in + 0.055))
        out_lo, out_hi = _yc_out - 0.105, _yc_out + 0.105
        im_out.set_extent((0.858, 0.898, out_lo, out_hi))
        for _x0, _x1, _y0, _y1 in ((0.082, 0.122, _yc_in - 0.055, _yc_in + 0.055),
                                   (0.858, 0.898, out_lo, out_hi)):
            ax_c.add_patch(plt.Rectangle((_x0, _y0), _x1 - _x0, _y1 - _y0,
                                         fill=False, ec="#777", lw=0.5,
                                         zorder=7))
        ax_c.text(0.102, _yc_in + 0.062, r"$(\dot x,\dot y)$", color="#ffffff",
                  fontsize=FS_NOTE, ha="center", va="bottom")
    for k, nm in enumerate(act_names):
        ax_c.text(0.907 if conn is not None else 0.995,
                  out_hi - (out_hi - out_lo) * (k + 0.5) / n_act, nm,
                  color="#ffffff", fontsize=FS_NOTE, ha="left", va="center")

    # --- kinograph -------------------------------------------------------
    # Every cell's rate against time, under the matrix whose rows they are.
    # The matrix shows one instant; this shows the history that produced it,
    # which is the only place a leak or a drift in the integrator is visible
    # at all. Row order is the matrix's, so a band here is the same population
    # as the band above it.
    im_kino = None
    if conn is not None:
        ax_k = fig.add_subplot(gs[1, :])
        ax_k.set_facecolor(BG)
        im_kino = ax_k.imshow(np.zeros((conn["n"], conn["kino_w"])),
                              cmap=EI_CMAP, vmin=-1, vmax=1, aspect="auto",
                              interpolation="nearest", origin="upper")
        ax_k.set_yticks([]); ax_k.set_xticks([])
        for sp_ in ax_k.spines.values():
            sp_.set_color("#777"); sp_.set_linewidth(0.5)
        for lbl, a0, a1 in (("AF5", 0, conn["n_in"]),
                            ("INTG", conn["n_in"], conn["n_in"] + conn["n_intg"]),
                            ("AMN/AIN", conn["n_in"] + conn["n_intg"], conn["n"])):
            if a0:
                ax_k.axhline(a0, color="#777", lw=0.5, alpha=0.6)
            ax_k.text(-0.004, 1.0 - (a0 + a1) / (2 * conn["n"]), lbl,
                      transform=ax_k.transAxes, color="#ccc", fontsize=FS_TICK,
                      ha="right", va="center")
        ax_k.set_xlabel("time  (the window slides once the trace reaches the "
                        "right edge)", color="#ddd", fontsize=FS_AXIS,
                        labelpad=4)
        ax_k.set_ylabel(f"all {conn['n']} rates", color="#ddd",
                        fontsize=FS_AXIS, labelpad=34)

    # --- eye -------------------------------------------------------------
    ax_e.axis("off")
    im_eye = ax_e.imshow(img0)
    txt_ang = ax_e.text(0.02, 0.02, "", transform=ax_e.transAxes, color="#ffffff",
                        fontsize=FS_NOTE, va="bottom")
    # Replaces SurfaceScene's own upper-left HUD when SurfaceView(hud=False)
    # suppresses it, so the whole figure is one font at one set of sizes.
    txt_hud = ax_e.text(0.02, 0.98, "", transform=ax_e.transAxes,
                        color="#ffffff", fontsize=FS_NOTE, va="top", ha="left",
                        linespacing=1.35)
    # the regime, boxed in the world panel's top right, where the eye is not
    phase = ax_w.text(0.975, 0.965, "", transform=ax_w.transAxes, color="#ffffff",
                      fontsize=FS_NOTE, ha="right", va="top")
    art = dict(trail_t=trail_t, trail_c=trail_c, trail_g=trail_g, dot_t=dot_t,
               dot_c=dot_c, dot_g=dot_g, im_in=im_in, im_rec=im_rec,
               im_out=im_out, im_eye=im_eye, txt_ang=txt_ang,
               txt_hud=txt_hud, phase=phase, im_kino=im_kino,
               side=side, hidden=hidden, im_rate=im_rate,
               im_cin=locals().get("im_cin"), im_cout=locals().get("im_cout"),
               im_intg=locals().get("im_intg"))
    return fig, art


def emit_spec(m, dt, tag, eye_dir=EYE_DIR, sim_dt=0.003, stride=3, base="pairs_long"):
    """Hand the controller's drive trace back to the REAL MPM plant.

    The right-hand panel above rotates eye G's rest geometry rigidly, which is honest
    but is not the tissue moving. This writes what the Plexus session needs to run the
    actual plant on the same command: the six-muscle trace resampled from the
    controller's 1/60 s to the simulator's own timestep, and a spec that plays it back
    verbatim through `muscle_probe [playback]`.

    Composing the result back in is `--compose`, so the two halves of the movie stay
    frame-aligned: their render is the right panel, everything else is drawn here.
    """
    import yaml
    n_sim = int(round(len(m) * dt / sim_dt))
    t_ctrl = np.arange(len(m)) * dt
    t_sim = np.arange(n_sim) * sim_dt
    trace = np.stack([np.interp(t_sim, t_ctrl, m[:, k]) for k in range(m.shape[1])], -1)
    trace = np.clip(trace, 0.0, 1.0)
    npy = os.path.join(eye_dir, f"ctrl_{tag}_act.npy")
    np.save(npy, trace.astype(np.float32))

    spec = yaml.safe_load(open(os.path.join(eye_dir, f"{base}_spec.yaml")))
    spec["general"]["name"] = f"eye_G_ctrl_{tag}"
    spec["general"]["n_frames"] = n_sim
    ops = spec.get("ops", spec.get("operators", []))
    for o in ops:
        if o.get("op") == "muscle_probe":
            o.clear()
            o.update({"op": "muscle_probe", "implementation": "playback",
                      "at": "muscle", "trace": os.path.basename(npy),
                      "tonic": 0.14, "tau": 0.02})
    out = os.path.join(eye_dir, f"ctrl_{tag}_spec.yaml")
    yaml.safe_dump(spec, open(out, "w"), sort_keys=False)
    print(f"[spec] wrote {npy}   ({trace.shape[0]} frames at dt {sim_dt})")
    print(f"[spec] wrote {out}")
    print("[spec] run it in the Plexus session, then compose:\n"
          f"    cd /workspace/Plexus/prototype/eye\n"
          f"    python run_eye_G.py --spec {os.path.basename(out)} "
          f"--out archive/eye_G --label ctrl_{tag} --render surface --stride {stride}\n"
          f"    # then, back here:\n"
          f"    python test_eyeG.py --tag {tag} --compose "
          f"{eye_dir}/ctrl_{tag}.mp4")
    return npy, out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="eyeG_light")
    p.add_argument("--eye-dir", default=EYE_DIR)
    p.add_argument("--duration", type=float, default=8.0, help="seconds PER half")
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trail", type=float, default=1.5, help="trail length, seconds")
    p.add_argument("--out", default=None)
    p.add_argument("--saccade", action="store_true",
                   help="six L/R saccade rates instead of the corpus regimes")
    p.add_argument("--emit-spec", action="store_true",
                   help="write the drive trace and a playback spec for the real MPM")
    p.add_argument("--compose", default=None, metavar="MP4",
                   help="use this rendered movie as the right panel, frame-aligned")
    p.add_argument("--render", default="surface", choices=EyeView.MODES,
                   help="how the right panel is drawn")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    out = a.out or os.path.join(
        TG.MODELS, f"{a.tag}_saccade.mp4" if a.saccade else f"{a.tag}_test.mp4")

    ck_path = os.path.join(TG.MODELS, f"{a.tag}.pt")
    if not os.path.isfile(ck_path):
        sys.exit(f"[test] no checkpoint {ck_path}. Train one first:\n"
                 f"    python train_eyeG.py --fit "
                 f"{'deep' if 'deep' in a.tag else 'light'}")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    spec = {k: (np.asarray(v) if isinstance(v, list) else v)
            for k, v in ck["eye"].items()}
    # An empty (0, 2) array round-trips through tolist() as [], which numpy reads
    # back as shape (0,) and the buffer shapes then disagree. Restore the width.
    for k, w in (("pairs", 2), ("pair_coef", 3)):
        if k in spec:
            spec[k] = np.asarray(spec[k], float if w == 3 else int).reshape(-1, w)
    eye = TG.EyeG(spec, a.dt)
    model = TG.CTRNNEyeG(eye, hidden=ck["hidden"], dt=a.dt)
    model.load_state_dict(ck["state"]); model.eval()
    reach, _, _ = eye.reach_deg()
    scale = np.asarray(ck["scale"], np.float32)
    names = list(ck["act_names"])
    print(f"[test] {a.tag}: {eye.kind} eye, {eye.n_act} drives ({', '.join(names)}), "
          f"reach {reach[0]:.1f}/{reach[1]:.1f} deg")

    # --- run -------------------------------------------------------------
    if a.saccade:
        v, xy, cuts = saccade_sequence(a.duration, a.dt, scale)
        labels = [f"saccades  {f:g} Hz" for f in SACCADE_HZ]
    else:
        v, xy, cuts = sequence(a.duration, a.dt, scale, seed=a.seed)
        labels = [(f"circle {'ccw' if mo.endswith('ccw') else 'cw'}  {sp}"
                   if mo.startswith("circle") else f"{mo.replace('_', '-')}  {sp}")
                  for mo, sp in PHASES]
    bounds = [0] + list(cuts) + [len(v)]
    tgt = xy * scale                                        # target, in degrees
    with torch.no_grad():
        u = torch.tensor(v[None])
        x, m, R = model(u, want_states=True)
        cmd = eye.equilibrium(m)                            # the red trace
    x = x[0].numpy(); m = m[0].numpy(); R = R[0].numpy(); cmd = cmd[0].numpy()
    err = np.linalg.norm(x[:, :2] - tgt, axis=-1)
    per = [float(err[bounds[i]:bounds[i + 1]].mean()) for i in range(len(labels))]
    print(f"[test] |err| mean {err.mean():.2f} deg   mean |torsion| "
          f"{np.abs(x[:, 2]).mean():.2f} deg")
    for lb, e in zip(labels, per):
        print(f"         {lb:22s} {e:.3f} deg")

    if a.emit_spec:
        emit_spec(m, a.dt, a.tag, a.eye_dir)
        return

    # --- render ----------------------------------------------------------
    ext = None
    if a.compose:
        import imageio.v2 as _iio
        ext = [f for f in _iio.get_reader(a.compose)]
        print(f"[compose] {len(ext)} rendered frames from {a.compose} against "
              f"{len(x)} controller steps")
        view = None
        img0 = ext[0]
    elif a.render == "surface":
        geo = load_geometry(a.eye_dir)
        view = SurfaceView(geo, x, cmd, m, a.dt)
        img0 = view.frame(0)
    else:
        geo = load_geometry(a.eye_dir)
        view = EyeView(geo, mode=a.render)
        img0 = view.frame(x[0], m[0])
    title = (f"{a.tag} — eye G, {eye.kind} characterisation — "
             f"continue then stop-and-go, middle speed — mean |err| {err.mean():.2f}°")
    fig, art = build_figure(reach, ck["hidden"], eye.n_act, names, img0, title)

    step = max(1, int(round(1.0 / (a.fps * a.dt))))
    keep = int(a.trail / a.dt)
    import imageio.v2 as iio
    w = iio.get_writer(out, fps=a.fps, codec="libx264", quality=8,
                       macro_block_size=1)
    side, hidden = art["side"], art["hidden"]
    pad = np.zeros(side * side - hidden, np.float32)
    frames = range(0, len(x), step)
    for k in tqdm(frames, desc=f"[render] {os.path.basename(out)}", unit="frame", ncols=100):
        lo = max(0, k - keep)
        art["trail_t"].set_data(tgt[lo:k + 1, 0], tgt[lo:k + 1, 1])
        art["trail_c"].set_data(cmd[lo:k + 1, 0], cmd[lo:k + 1, 1])
        art["trail_g"].set_data(x[lo:k + 1, 0], x[lo:k + 1, 1])
        art["dot_t"].set_data([tgt[k, 0]], [tgt[k, 1]])
        art["dot_c"].set_data([cmd[k, 0]], [cmd[k, 1]])
        art["dot_g"].set_data([x[k, 0]], [x[k, 1]])
        art["im_in"].set_data(np.clip(v[k][:, None] / 1.5, -1, 1))
        art["im_rec"].set_data(np.concatenate([R[k], pad]).reshape(side, side))
        art["im_out"].set_data(np.clip(m[k][:, None], 0, 1))
        art["im_eye"].set_data(
            ext[min(int(k * len(ext) / len(x)), len(ext) - 1)] if ext is not None
            else (view.frame(k) if isinstance(view, SurfaceView)
                  else view.frame(x[k], m[k])))
        if not isinstance(view, SurfaceView):
            art["txt_ang"].set_text(f"θ {x[k,0]:+6.2f}   φ {x[k,1]:+6.2f}   "
                                    f"ψ {x[k,2]:+6.2f}  deg")
        art["phase"].set_text(labels[max(0, int(np.searchsorted(cuts, k, "right")))])
        fig.canvas.draw()
        w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    w.close(); plt.close(fig)
    if view is not None:
        view.close()
    # keep the traces: the movie is the only record otherwise, and a figure of
    # these curves cannot be made from an mp4
    np.savez_compressed(out.replace(".mp4", ".npz"),
                        t=np.arange(len(x)) * a.dt, target=tgt, command=cmd,
                        gaze=x, drives=m, cuts=np.asarray(cuts),
                        labels=np.array(labels, dtype=object),
                        reach=np.asarray(reach), scale=scale, dt=a.dt)
    rep = out.replace(".mp4", ".json")
    json.dump({"tag": a.tag, "kind": eye.kind, "n_act": eye.n_act,
               "err_mean_deg": float(err.mean()),
               "err_by_phase_deg": dict(zip(labels, per)),
               "mean_abs_torsion_deg": float(np.abs(x[:, 2]).mean()),
               "reach_deg": [float(r) for r in reach],
               "duration_per_half_s": a.duration, "seed": a.seed},
              open(rep, "w"), indent=2)
    print(f"[test] wrote {out}\n       wrote {rep}")


if __name__ == "__main__":
    main()
