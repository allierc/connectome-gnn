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

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trajectory import generate                          # noqa: E402
import train_eyeG as TG                                  # noqa: E402

EYE_DIR = TG.EYE_DIR
MUS_COLOR = {"LR": (0.30, 0.55, 0.95), "SR": (0.85, 0.25, 0.25),
             "MR": (0.90, 0.75, 0.25), "IR": (0.35, 0.75, 0.45),
             "SO": (0.60, 0.40, 0.80), "IO": (0.90, 0.55, 0.25)}
BG = "#000000"


# ---------------------------------------------------------------------------
# the test sequence
# ---------------------------------------------------------------------------
def sequence(duration, dt, seed=0, bound=0.95, tries=40):
    """continue then stop_and_go, both middle speed, joined in velocity.

    Retries the seed until the integrated target stays inside the arena, rather than
    clipping it: clipping would distort the velocity, which is the only thing the
    circuit is given, and a distorted input is not the regime we mean to test.
    """
    for k in range(tries):
        parts = [generate(shape="curve", motion=m, speed="middle", angle="low",
                          duration=duration, dt=dt, seed=seed + 1000 * k + i,
                          start="center")
                 for i, m in enumerate(("continue", "stop_and_go"))]
        v = np.concatenate([np.stack([np.gradient(np.asarray(p["x"]), dt),
                                      np.gradient(np.asarray(p["y"]), dt)], -1)
                            for p in parts], 0).astype(np.float32)
        xy = np.cumsum(v, 0) * dt
        if np.abs(xy).max() <= bound:
            n = len(parts[0]["x"])
            return v, xy.astype(np.float32), n
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

    MODES = ("vtk", "mpm", "mpm_stress")

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
        return np.asarray(self.pl.screenshot(return_img=True))

    def close(self):
        self.pl.close()


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------
def build_figure(reach, hidden, n_act, act_names, img0, title):
    fig = plt.figure(figsize=(16.0, 5.6), facecolor=BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.75, 1.0], wspace=0.16,
                          left=0.04, right=0.985, top=0.90, bottom=0.08)
    ax_w, ax_c, ax_e = (fig.add_subplot(gs[0, k]) for k in range(3))
    for ax in (ax_w, ax_c, ax_e):
        ax.set_facecolor(BG)

    # --- world -----------------------------------------------------------
    lim_h, lim_v = reach[0] * 1.15, reach[1] * 1.15
    ax_w.set_xlim(-lim_h, lim_h); ax_w.set_ylim(-lim_v, lim_v)
    ax_w.set_aspect("equal")
    ax_w.axhline(0, color="#222", lw=0.8); ax_w.axvline(0, color="#222", lw=0.8)
    ax_w.add_patch(plt.Rectangle((-reach[0], -reach[1]), 2 * reach[0], 2 * reach[1],
                                 fill=False, ec="#5a5a2a", ls=":", lw=1.0))
    ax_w.text(-reach[0], reach[1] * 1.03, f"eye reach  ±{reach[0]:.1f}° h / "
              f"±{reach[1]:.1f}° v", color="#7a7a4a", fontsize=8, va="bottom")
    for s in ax_w.spines.values():
        s.set_color("#333")
    ax_w.tick_params(colors="#666", labelsize=8)
    ax_w.set_xlabel("θ  horizontal gaze (deg)", color="#999", fontsize=9)
    ax_w.set_ylabel("φ  vertical gaze (deg)", color="#999", fontsize=9)
    trail_t, = ax_w.plot([], [], "-", color="#ffffff", lw=1.0, alpha=0.45)
    trail_c, = ax_w.plot([], [], "-", color="#e05a4a", lw=1.2, alpha=0.75)
    trail_g, = ax_w.plot([], [], "-", color="#4da3ff", lw=1.6, alpha=0.95)
    # the target is a RING, not a disc: the gaze dot sits inside it when the
    # controller is working, and a filled marker would simply hide the thing the
    # movie exists to show.
    dot_t, = ax_w.plot([], [], "o", mfc="none", mec="#ffffff", mew=1.8, ms=15)
    dot_c, = ax_w.plot([], [], "o", color="#e05a4a", ms=7, mec="none")
    dot_g, = ax_w.plot([], [], "o", color="#4da3ff", ms=9, mec="none")

    # --- circuit ---------------------------------------------------------
    side = int(np.ceil(np.sqrt(hidden)))
    ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1); ax_c.axis("off")
    ax_c.text(0.5, 1.005, "the circuit", color="#999", fontsize=10,
              ha="center", va="top")
    im_in = ax_c.imshow(np.zeros((1, 2)), cmap="coolwarm", vmin=-1, vmax=1,
                        extent=(0.30, 0.70, 0.80, 0.90), aspect="auto", zorder=3)
    im_rec = ax_c.imshow(np.zeros((side, side)), cmap="coolwarm", vmin=-1, vmax=1,
                         extent=(0.13, 0.87, 0.22, 0.72), aspect="auto", zorder=3)
    im_out = ax_c.imshow(np.zeros((1, n_act)), cmap="magma", vmin=0, vmax=1,
                         extent=(0.13, 0.87, 0.07, 0.15), aspect="auto", zorder=3)
    for y, s in ((0.915, r"input  $(\dot x,\dot y)$"),
                 (0.755, f"recurrent  {hidden} rates  $r=\\tanh v$"),
                 (0.185, f"output  {n_act} drives  $m=[\\hat W^{{out}}r]_+$")):
        ax_c.text(0.5, y, s, color="#888", fontsize=8.5, ha="center", va="bottom")
    for k, nm in enumerate(act_names):
        ax_c.text(0.13 + 0.74 * (k + 0.5) / n_act, 0.045, nm, color="#888",
                  fontsize=7.5, ha="center", va="top")

    # --- eye -------------------------------------------------------------
    ax_e.axis("off")
    im_eye = ax_e.imshow(img0)
    txt_ang = ax_e.text(0.02, 0.02, "", transform=ax_e.transAxes, color="#bbb",
                        fontsize=9, va="bottom", family="monospace")
    sup = fig.text(0.5, 0.965, title, color="#ddd", fontsize=11, ha="center",
                   va="top")
    phase = fig.text(0.04, 0.965, "", color="#e0c060", fontsize=10, ha="left",
                     va="top")
    art = dict(trail_t=trail_t, trail_c=trail_c, trail_g=trail_g, dot_t=dot_t,
               dot_c=dot_c, dot_g=dot_g, im_in=im_in, im_rec=im_rec,
               im_out=im_out, im_eye=im_eye, txt_ang=txt_ang, phase=phase,
               side=side, hidden=hidden)
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
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trail", type=float, default=1.5, help="trail length, seconds")
    p.add_argument("--out", default=None)
    p.add_argument("--emit-spec", action="store_true",
                   help="write the drive trace and a playback spec for the real MPM")
    p.add_argument("--compose", default=None, metavar="MP4",
                   help="use this rendered movie as the right panel, frame-aligned")
    p.add_argument("--render", default="vtk", choices=EyeView.MODES,
                   help="how the right panel is drawn")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    out = a.out or os.path.join(TG.MODELS, f"{a.tag}_test.mp4")

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
    v, xy, n_split = sequence(a.duration, a.dt, seed=a.seed)
    tgt = xy * scale                                        # target, in degrees
    with torch.no_grad():
        u = torch.tensor(v[None])
        x, m, R = model(u, want_states=True)
        cmd = eye.equilibrium(m)                            # the red trace
    x = x[0].numpy(); m = m[0].numpy(); R = R[0].numpy(); cmd = cmd[0].numpy()
    err = np.linalg.norm(x[:, :2] - tgt, axis=-1)
    print(f"[test] |err| mean {err.mean():.2f} deg   "
          f"first half {err[:n_split].mean():.2f}   second {err[n_split:].mean():.2f}   "
          f"mean |torsion| {np.abs(x[:, 2]).mean():.2f} deg")

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
    for k in range(0, len(x), step):
        lo = max(0, k - keep)
        art["trail_t"].set_data(tgt[lo:k + 1, 0], tgt[lo:k + 1, 1])
        art["trail_c"].set_data(cmd[lo:k + 1, 0], cmd[lo:k + 1, 1])
        art["trail_g"].set_data(x[lo:k + 1, 0], x[lo:k + 1, 1])
        art["dot_t"].set_data([tgt[k, 0]], [tgt[k, 1]])
        art["dot_c"].set_data([cmd[k, 0]], [cmd[k, 1]])
        art["dot_g"].set_data([x[k, 0]], [x[k, 1]])
        art["im_in"].set_data(np.clip(v[k][None] / 1.5, -1, 1))
        art["im_rec"].set_data(np.concatenate([R[k], pad]).reshape(side, side))
        art["im_out"].set_data(np.clip(m[k][None], 0, 1))
        art["im_eye"].set_data(ext[min(int(k * len(ext) / len(x)), len(ext) - 1)]
                               if ext is not None else view.frame(x[k], m[k]))
        art["txt_ang"].set_text(f"θ {x[k,0]:+6.2f}   φ {x[k,1]:+6.2f}   "
                                f"ψ {x[k,2]:+6.2f}  deg")
        art["phase"].set_text("continue" if k < n_split else "stop-and-go")
        fig.canvas.draw()
        w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    w.close(); plt.close(fig)
    if view is not None:
        view.close()
    rep = out.replace(".mp4", ".json")
    json.dump({"tag": a.tag, "kind": eye.kind, "n_act": eye.n_act,
               "err_mean_deg": float(err.mean()),
               "err_continue_deg": float(err[:n_split].mean()),
               "err_stop_and_go_deg": float(err[n_split:].mean()),
               "mean_abs_torsion_deg": float(np.abs(x[:, 2]).mean()),
               "reach_deg": [float(r) for r in reach],
               "duration_per_half_s": a.duration, "seed": a.seed},
              open(rep, "w"), indent=2)
    print(f"[test] wrote {out}\n       wrote {rep}")


if __name__ == "__main__":
    main()
