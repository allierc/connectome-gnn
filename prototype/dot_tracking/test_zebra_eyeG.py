#!/usr/bin/env python
"""test_zebra_eyeG -- the twin of test_eyeG's movie, for the connectome circuit.

    python test_zebra_eyeG.py --spec ../../config/zebrafish/zebrafish_om_intg_285_nominal.yaml
    python test_zebra_eyeG.py --tag zebraEyeG            # a models/<tag>.pt run
    python test_zebra_eyeG.py --spec <spec> --saccade    # the six saccade rates

Writes `<run>/results/<name>_test.mp4`, the exact three-panel movie
`test_eyeG.py --tag eyeG` writes for the free ctRNN: world / circuit / eye G.
EVERY PIXEL COMES FROM test_eyeG -- `build_figure`, `SurfaceView`,
`load_geometry`, `_saccade` and the writer loop are imported and called, not
reimplemented, so the two movies are comparable frame for frame and a change
to the renderer lands in both at once.

WHAT DIFFERS, AND WHY EACH DIFFERENCE IS FORCED

  the controller   `ZebrafishCircuitRNN` (285 real cells, Dale-signed, masked
                   to 5013 measured synapses) instead of a free (64, 64)
                   matrix. The middle panel's "recurrent rates" square is
                   therefore 17x17 = 289 with 4 pad cells, not 8x8.

  the stimulus     HORIZONTAL, via `sequence_h` below. test_eyeG's `sequence`
                   opens with two CIRCLE phases and walks a 2-D curve; this
                   pool reaches LR and MR only, so a vertical component is an
                   input no output can answer and the movie would be showing
                   the circuit failing at a task it was never given. The
                   phases here are the horizontal regimes it was trained on
                   plus a saccade train, which is the classic probe of an
                   integrator's leak.

  the drives       two of six are ever nonzero (LR, MR). The other four are
                   drawn at exactly zero rather than softplus(0) -- see
                   train_zebra_eyeG's docstring for why that is the honest
                   rendering.

The red trace is still the COMMAND (where the eye would settle if the drive
froze) and blue still the GAZE; red sitting on blue is unity DC gain, not an
idle eye. Section 5.1 of the oculomotor note.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_eyeG as TE                                       # noqa: E402
import train_eyeG as TG                                      # noqa: E402
import train_zebra_eyeG as TZ                                # noqa: E402
from trajectory import generate                              # noqa: E402
from zebrafish_circuit import load_oculomotor_circuit        # noqa: E402

# The horizontal counterpart of test_eyeG.PHASES. Six phases, so the movie has
# the same shape and the same per-phase table; no circles, because a circle is
# half vertical and this pool has no vertical output.
PHASES_H = [("continue", "slow"), ("continue", "middle"), ("continue", "fast"),
            ("stop_and_go", "middle"), ("stop_and_go", "fast"),
            ("saccade", "1.15")]


def sequence_h(duration, dt, scale, seed=0, bound=0.95, tries=200):
    """PHASES_H joined in VELOCITY, on the line y = 0.

    Same contract as test_eyeG.sequence -- velocities are concatenated so the
    target never jumps and the circuit's state carries across a boundary --
    but the arena has to be kept a different way, and the difference is forced
    by the dimension rather than chosen.

    `sequence` keeps a 2-D target inside by ROTATING each phase toward the
    centre, searching 72 angles. A rotation is an isometry, so speed, turn
    rate and stop/go timing survive it exactly and only the heading changes.
    On a line the only isometry left is a REFLECTION -- two candidates, not
    72 -- and two is not enough: six 8 s phases are 48 s of walk, and at
    `fast` (0.90 units/s) that is 43 units of travel on an arena 1.9 wide, so
    a purely sign-chosen concatenation drifts out every time. That is what the
    global-retry version did, 200 seeds deep.

    So the search moves inside the loop: each phase is drawn against many
    SEEDS and both signs, and the first candidate whose integrated path stays
    inside the arena FROM WHERE THE PREVIOUS PHASE LEFT OFF is kept. Every
    candidate is a genuine sample of that regime at that speed -- re-drawing a
    seed changes which walk you get, never what a walk of that kind looks like
    -- so this keeps test_eyeG's rule of retrying rather than clipping. The
    ranking falls back to the least-bad candidate only if nothing fits, which
    is reported rather than silently accepted.
    """
    rng_seeds = range(seed, seed + tries)
    segs, pos = [], 0.0
    for i, (mo, sp) in enumerate(PHASES_H):
        if mo == "saccade":
            # already zero-net by construction, and horizontal-only
            seg = TE._saccade(duration, dt, scale, float(sp))
            segs.append(seg)
            pos = pos + seg[:, 0].sum() * dt
            continue
        best, best_r = None, np.inf
        for s in rng_seeds:
            q = generate(shape="curve", motion=mo, speed=sp, angle="low",
                         duration=duration, dt=dt, axis="horizontal",
                         seed=s + 1000 * i, start="center")
            base = np.stack([np.gradient(np.asarray(q["x"]), dt),
                             np.gradient(np.asarray(q["y"]), dt)], -1)
            for cand in (base, -base):
                reach = np.abs(pos + np.cumsum(cand[:, 0]) * dt).max()
                if reach < best_r:
                    best, best_r = cand, reach
            if best_r <= bound:
                break
        if best_r > bound:
            print(f"[seq] phase {i} ({mo} {sp}) reaches {best_r:.2f} > {bound} "
                  f"after {tries} seeds; keeping the least-bad draw")
        segs.append(best)
        pos = pos + best[:, 0].sum() * dt
    n0 = len(segs[0])
    v = np.concatenate(segs, 0).astype(np.float32)
    xy = np.cumsum(v, 0) * dt
    return v, xy.astype(np.float32), [n0 * (i + 1)
                                      for i in range(len(PHASES_H) - 1)]


def resolve_run(a):
    """Find the checkpoint, its circuit config and where the movie goes.

    Two layouts, because both exist: a `--spec` run writes the repo's
    log/<biomodel>/<name>/{models/best.pt,results/} tree, and a bare `--tag`
    run writes prototype/dot_tracking/models/<tag>.pt beside the others.
    """
    if a.spec:
        import yaml
        with open(a.spec) as f:
            spec = yaml.safe_load(f)
        here = os.path.dirname(os.path.abspath(a.spec))
        rd = TZ.run_dir(spec, a.out_root)
        return (os.path.join(rd, "models", "best.pt"),
                os.path.join(here, spec["circuit_config"]),
                os.path.join(rd, "results"), spec["name"])
    return (os.path.join(TG.MODELS, f"{a.tag}.pt"), a.config,
            TG.MODELS, a.tag)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default=None, help="nominal run spec")
    p.add_argument("--out-root", default=None)
    p.add_argument("--tag", default="zebraEyeG",
                   help="used only without --spec")
    p.add_argument("--config", default=TZ.DEFAULT_CONFIG)
    p.add_argument("--pkl", default=TZ.DEFAULT_PKL)
    p.add_argument("--eye-dir", default=TG.EYE_DIR)
    p.add_argument("--duration", type=float, default=8.0, help="seconds PER phase")
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trail", type=float, default=1.5)
    p.add_argument("--out", default=None)
    p.add_argument("--saccade", action="store_true",
                   help="six L/R saccade rates instead of the horizontal regimes")
    p.add_argument("--render", default="surface", choices=TE.EyeView.MODES)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    ck_path, cfg_path, out_dir, name = resolve_run(a)
    if not os.path.isfile(ck_path):
        sys.exit(f"[test] no checkpoint {ck_path}. Train one first:\n"
                 f"    python train_zebra_eyeG.py"
                 + (f" --spec {a.spec}" if a.spec else ""))
    os.makedirs(out_dir, exist_ok=True)
    out = a.out or os.path.join(
        out_dir, f"{name}_saccade.mp4" if a.saccade else f"{name}_test.mp4")

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    eye_spec = {k: (np.asarray(v) if isinstance(v, list) else v)
                for k, v in ck["eye"].items()}
    for k, w in (("pairs", 2), ("pair_coef", 3)):
        if k in eye_spec:
            eye_spec[k] = np.asarray(eye_spec[k],
                                     float if w == 3 else int).reshape(-1, w)
    eye = TG.EyeG(eye_spec, a.dt)

    circuit = load_oculomotor_circuit(cfg_path, a.pkl,
                                      eye_side=ck.get("eye_side"))
    model = TZ.ZebrafishCircuitRNN(circuit, tau0=ck.get("tau0", 0.1), dt=a.dt)
    model.eye = eye
    model.load_state_dict(ck["state"])
    model.eval()

    reach, _, _ = eye.reach_deg()
    scale = np.asarray(ck["scale"], np.float32)
    names = list(eye.act_names)
    print(f"[test] {name}: {circuit['cfg'].circuit.name}, {model.N} cells, "
          f"{ck.get('eye_side')} eye, readout "
          + ", ".join(f"{k} {len(v)}" for k, v in sorted(circuit["output_idx"].items())))

    # --- run --------------------------------------------------------------
    if a.saccade:
        v, xy, cuts = TE.saccade_sequence(a.duration, a.dt, scale)
        labels = [f"saccades  {f:g} Hz" for f in TE.SACCADE_HZ]
    else:
        v, xy, cuts = sequence_h(a.duration, a.dt, scale, seed=a.seed)
        labels = [(f"saccades  {sp} Hz" if mo == "saccade"
                   else f"{mo.replace('_', '-')}  {sp}") for mo, sp in PHASES_H]
    bounds = [0] + list(cuts) + [len(v)]
    tgt = xy * scale
    with torch.no_grad():
        x, m, R = model(torch.tensor(v[None]), want_states=True)
        cmd = eye.equilibrium(m)
    x = x[0].numpy(); m = m[0].numpy(); R = R[0].numpy(); cmd = cmd[0].numpy()

    # theta only: phi is unreachable from this pool, so a 2-D error would be
    # dominated by an axis the controller was never asked about.
    err = np.abs(x[:, 0] - tgt[:, 0])
    print(f"[test] |err_theta| mean {err.mean():.3f} deg   mean |torsion| "
          f"{np.abs(x[:, 2]).mean():.2f} deg   |phi| {np.abs(x[:, 1]).mean():.2f} deg")
    per = [float(err[bounds[i]:bounds[i + 1]].mean()) for i in range(len(labels))]
    for lb, e in zip(labels, per):
        print(f"         {lb:22s} {e:.3f} deg")

    # --- render: test_eyeG's own figure and views, unchanged ---------------
    geo = TE.load_geometry(a.eye_dir)
    if a.render == "surface":
        view = TE.SurfaceView(geo, x, cmd, m, a.dt)
        img0 = view.frame(0)
    else:
        view = TE.EyeView(geo, mode=a.render)
        img0 = view.frame(x[0], m[0])
    title = (f"{name} — 285-cell oculomotor circuit ({ck.get('eye_side')} eye), "
             f"eye G — horizontal regimes — mean |err| {err.mean():.3f}°")
    fig, art = TE.build_figure(reach, model.N, eye.n_act, names, img0, title)

    step = max(1, int(round(1.0 / (a.fps * a.dt))))
    keep = int(a.trail / a.dt)
    import imageio.v2 as iio
    from tqdm import tqdm
    w = iio.get_writer(out, fps=a.fps, codec="libx264", quality=8,
                       macro_block_size=1)
    side, hidden = art["side"], art["hidden"]
    pad = np.zeros(side * side - hidden, np.float32)
    for k in tqdm(range(0, len(x), step), desc=f"[render] {os.path.basename(out)}",
                  unit="frame", ncols=100):
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
        art["im_eye"].set_data(view.frame(k) if a.render == "surface"
                               else view.frame(x[k], m[k]))
        fig.canvas.draw()
        w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    w.close()
    print(f"[test] wrote {out}")

    rep = os.path.join(out_dir, f"{name}_test.json")
    json.dump({"name": name, "checkpoint": ck_path, "eye_side": ck.get("eye_side"),
               "n_cells": int(model.N), "saccade": bool(a.saccade),
               "err_theta_mean_deg": float(err.mean()),
               "mean_abs_torsion_deg": float(np.abs(x[:, 2]).mean()),
               "mean_abs_phi_deg": float(np.abs(x[:, 1]).mean()),
               "per_phase": dict(zip(labels, per)), "movie": out},
              open(rep, "w"), indent=2)
    print(f"[test] wrote {rep}")


if __name__ == "__main__":
    main()
