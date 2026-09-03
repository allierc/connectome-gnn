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
                   matrix.

  the middle panel FIGURE 1c OF THE NOTE, MADE LIVE, and wider than
                   test_eyeG's because it has more to carry. The 285x285
                   matrix keeps Figure 1c's key exactly -- post x pre, BLUE
                   excitatory and RED inhibitory, the sign taken from the
                   presynaptic column's Dale assignment -- and changes only
                   what sets the brightness: |W_ij r_j|, the message crossing
                   that synapse this frame, instead of the static synapse
                   weight. Sign from the connectome, weight from training,
                   brightness from the traffic.

                   Around it, the three neuron groups as vectors of live
                   rates on that same blue/red scale: AF5 as a column beside
                   its own rows on the left, AMN/AIN as a column beside
                   theirs on the right, and the INTG integrator as a row
                   along the bottom in the matrix's column order. So the
                   panel reads left to right in the direction the signal
                   travels -- velocity, afferent cells, the circuit, motor
                   cells, muscle drive -- with every value on one key.

                   test_eyeG draws a square of rates there instead, because a
                   free (hidden, hidden) matrix has no structure worth a
                   picture. Here the matrix IS the result, and a 17x17
                   reshape of 285 cells would be worse than uninformative:
                   285 is not a square, and neighbours in the reshape are not
                   neighbours in the circuit.

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
    p.add_argument("--preview", type=float, default=None, metavar="FRAC",
                   help="render ONE frame at this fraction of the run (0..1) "
                        "to <results>/<name>_preview.png and stop. The movie "
                        "takes ~4 min; a layout change needs one frame to "
                        "check, so iterate with this and render the mp4 once "
                        "the panel is right.")
    p.add_argument("--phi-zero", action="store_true",
                   help="DISPLAY ONLY: draw the gaze and rotate the eye with "
                        "phi forced to 0, so the movie reads as pure "
                        "left-right. The model is untouched -- nothing is "
                        "retrained, no loss changes -- and the true mean |phi| "
                        "is still printed and still written to the json, so "
                        "the number this hides stays on the record. Use it to "
                        "look at horizontal tracking without the vertical "
                        "offset in the way; do not use it to claim the circuit "
                        "holds phi at zero. It does not: phi is reachable over "
                        "5.13 deg from LR+MR alone, with a hard floor of 2.3 "
                        "deg at one end of the gaze range, so making it "
                        "emergent means putting phi in the LOSS "
                        "(train_zebra_eyeG --track-phi), not zeroing it here.")
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
    phi_true = float(np.abs(x[:, 1]).mean())
    if a.phi_zero:
        x = x.copy(); cmd = cmd.copy()
        x[:, 1] = 0.0; cmd[:, 1] = 0.0
        print(f"[test] --phi-zero: drawing phi as 0. The model's actual mean "
              f"|phi| is {phi_true:.2f} deg and is unchanged.")

    # theta only: phi is unreachable from this pool, so a 2-D error would be
    # dominated by an axis the controller was never asked about.
    err = np.abs(x[:, 0] - tgt[:, 0])
    print(f"[test] |err_theta| mean {err.mean():.3f} deg   mean |torsion| "
          f"{np.abs(x[:, 2]).mean():.2f} deg   |phi| {phi_true:.2f} deg"
          + ("  (drawn as 0)" if a.phi_zero else ""))
    per = [float(err[bounds[i]:bounds[i + 1]].mean()) for i in range(len(labels))]
    for lb, e in zip(labels, per):
        print(f"         {lb:22s} {e:.3f} deg")

    # --- render: test_eyeG's own figure and views, unchanged ---------------
    geo = TE.load_geometry(a.eye_dir)
    if a.render == "surface":
        view = TE.SurfaceView(geo, x, cmd, m, a.dt, hud=False)
        img0 = view.frame(0)
    else:
        view = TE.EyeView(geo, mode=a.render)
        img0 = view.frame(x[0], m[0])
    # THE MESSAGE PASSING, not the wiring. Figure 1c of the note shows this
    # matrix with its intensity set by the static synapse weight; here the
    # intensity is |W_ij r_j| -- what actually crosses that synapse this frame
    # -- while the COLOUR stays Figure 1c's, the Dale sign of the presynaptic
    # column: blue excitatory, red inhibitory. Sign from the connectome, weight
    # from training, brightness from the traffic.
    #
    # The sign is deliberately the column's and not the message's. r_j = tanh v
    # is signed, so sign(W_ij r_j) flickers with every rate zero-crossing and
    # the E/I band structure -- the one thing this panel exists to make
    # falsifiable by eye, per _panel_signed_matrix's docstring -- would strobe
    # away. Brightness carries the dynamics; colour carries the anatomy.
    with torch.no_grad():
        What = model.W_hat().numpy()
    absW = np.abs(What)
    col_sign = circuit["sign_col"].astype(np.float32)

    # One fixed scale for the whole movie, so two frames are comparable. Taken
    # from the 99.5th percentile of the messages that actually occur over the
    # run rather than from the weights, which would leave the panel dark: a
    # strong synapse onto a silent cell carries nothing.
    _s = R[::max(1, len(R) // 120)]
    _mag = (absW[None] * np.abs(_s)[:, None, :]).reshape(-1)
    msg_lim = float(np.percentile(_mag[_mag > 0], 99.5)) or 1.0

    # log1p, as Figure 1c does -- the messages span decades -- and the same
    # 2x2 grey_dilation, without which a 285x285 matrix drawn at movie
    # resolution loses most of its 5013 synapses to pixel rounding.
    from scipy.ndimage import grey_dilation
    _knee = np.log1p(9.0)

    def conn_frame(k):
        mag = np.log1p(9.0 * absW * np.abs(R[k])[None, :] / msg_lim) / _knee
        mag = grey_dilation(mag, size=(2, 2))
        return np.clip(mag, 0.0, 1.0) * col_sign[None, :]

    names_arr = circuit["names"]
    role_of = {s.name: (s.role or "recurrent")
               for s in circuit["cfg"].circuit.cell_types}
    roles = np.array([role_of[str(t)] for t in names_arr])
    idx_in = np.where(roles == "afferent")[0]
    idx_intg = np.where(roles == "recurrent")[0]
    idx_out = np.where(roles == "output")[0]
    # Of the 127 output-role cells, only the ones this eye's readout uses are
    # driven: LR from AMN-L (47), MR from AIN-R (19). The other 61 are in the
    # circuit and have rates, but no path to a muscle. The column keeps all 127
    # rows so it stays aligned with the matrix, and the unused cells are drawn
    # as NaN -- i.e. as background -- so the mask itself is visible rather than
    # implied.
    idx_readout = np.sort(np.concatenate(
        [np.asarray(v, int) for v in circuit["output_idx"].values()]))
    out_keep = np.isin(idx_out, idx_readout)
    print(f"[test] readout mask: {int(out_keep.sum())} of {idx_out.size} "
          f"output-role cells drive a muscle")
    # The ordering is afferent -> recurrent -> output by construction, so the
    # three groups are contiguous blocks and a vector drawn beside the matrix
    # lines up with its own rows.
    assert idx_in.max() < idx_intg.min() and idx_intg.max() < idx_out.min()
    # Kinograph window: KINO_S seconds of history. The trace fills left to
    # right and the window then slides, so the newest sample is always at the
    # right edge and the x axis is a fixed span of time rather than the whole
    # run squeezed into a panel.
    KINO_S = 12.0
    kino_w = int(round(KINO_S / a.dt))
    kino_full = np.clip(R.T.astype(np.float32), -1.0, 1.0)   # (n, T)
    # Same readout mask as the output column: the 61 output-role cells with no
    # path to a muscle are blanked, so the kinograph and the vector beside the
    # matrix agree about which cells are actually driving this eye.
    _dead = idx_out[~out_keep]
    kino_full[_dead, :] = np.nan
    # AMN and AIN banded separately -- one drives LR from the left hemisphere,
    # the other MR from the right.
    kino_blocks = [("AF5", 0), ("INTG", int(idx_in.size))]
    for nm in ("AMN", "AIN"):
        rows = np.where(names_arr == nm)[0]
        if rows.size:
            kino_blocks.append((nm, int(rows.min())))
    kino_blocks.sort(key=lambda t: t[1])
    conn_meta = dict(n=int(model.N), n_in=int(idx_in.size),
                     n_intg=int(idx_intg.size), kino_w=kino_w,
                     kino_blocks=kino_blocks)

    def kino_frame(k):
        if k + 1 >= kino_w:                       # sliding
            return kino_full[:, k + 1 - kino_w:k + 1]
        buf = np.full((model.N, kino_w), np.nan, np.float32)
        buf[:, :k + 1] = kino_full[:, :k + 1]     # filling, left-aligned
        return buf
    blocks, seen = [], None
    for r, nm in enumerate(names_arr):
        if nm != seen:
            blocks.append((str(nm), r))
            seen = nm
    title = (f"{name} — 285-cell oculomotor circuit ({ck.get('eye_side')} eye), "
             f"eye G — horizontal regimes — mean |err| {err.mean():.3f}°"
             + (f"   [φ drawn as 0; true mean |φ| {phi_true:.2f}°]"
                if a.phi_zero else ""))
    fig, art = TE.build_figure(reach, model.N, eye.n_act, names, img0, title,
                               conn=conn_meta, conn_blocks=blocks,
                               cmd_trail=False)

    step = max(1, int(round(1.0 / (a.fps * a.dt))))
    keep = int(a.trail / a.dt)
    side, hidden = art["side"], art["hidden"]
    pad = np.zeros(side * side - hidden, np.float32)

    def draw(k):
        lo = max(0, k - keep)
        art["trail_t"].set_data(tgt[lo:k + 1, 0], tgt[lo:k + 1, 1])
        art["trail_c"].set_data(cmd[lo:k + 1, 0], cmd[lo:k + 1, 1])
        art["trail_g"].set_data(x[lo:k + 1, 0], x[lo:k + 1, 1])
        art["dot_t"].set_data([tgt[k, 0]], [tgt[k, 1]])
        art["dot_c"].set_data([cmd[k, 0]], [cmd[k, 1]])
        art["dot_g"].set_data([x[k, 0]], [x[k, 1]])
        art["im_in"].set_data(np.clip(v[k][:, None] / 1.5, -1, 1))
        if art["im_cin"] is not None:
            art["im_rec"].set_data(conn_frame(k))
            art["im_cin"].set_data(R[k][idx_in][:, None])
            _o = R[k][idx_out].astype(np.float32).copy()
            _o[~out_keep] = np.nan
            art["im_cout"].set_data(_o[:, None])
            art["im_intg"].set_data(R[k][idx_intg][:, None])
            art["im_kino"].set_data(kino_frame(k))
        else:
            art["im_rec"].set_data(np.concatenate([R[k], pad]).reshape(side, side))
        art["im_out"].set_data(np.clip(m[k][:, None], 0, 1))
        art["im_eye"].set_data(view.frame(k) if a.render == "surface"
                               else view.frame(x[k], m[k]))
        # SurfaceView paints its own angle overlay; setting txt_ang too would
        # print the numbers twice. Same rule as test_eyeG's writer loop.
        # SurfaceScene's VTK HUD is suppressed (hud=False), so the same
        # content is drawn here in the figure's own font.
        art["txt_hud"].set_text(
            f"frame {k}   t = {k * a.dt:5.2f} s\n"
            f"command   h {cmd[k,0]:+5.1f}   v {cmd[k,1]:+5.1f}   "
            f"\u03c8 {cmd[k,2]:+5.1f}\n"
            f"gaze      h {x[k,0]:+5.1f}   v {x[k,1]:+5.1f}   "
            f"\u03c8 {x[k,2]:+5.1f}")
        # two rows of three: one line of six ran off the panel and lost IO
        art["txt_ang"].set_text(
            "activation   " + "  ".join(f"{nm} {m[k, j]:.2f}"
                                        for j, nm in enumerate(names[:3]))
            + "\n             " + "  ".join(f"{nm} {m[k, j + 3]:.2f}"
                                             for j, nm in enumerate(names[3:])))
        art["phase"].set_text(labels[max(0, int(np.searchsorted(cuts, k, "right")))])
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3]

    if a.preview is not None:
        k = int(np.clip(a.preview, 0.0, 1.0) * (len(x) - 1))
        png = os.path.join(out_dir, f"{name}_preview.png")
        import imageio.v2 as iio
        iio.imwrite(png, draw(k))
        print(f"[test] preview frame {k} of {len(x)} -> {png}")
        return

    import imageio.v2 as iio
    from tqdm import tqdm
    w = iio.get_writer(out, fps=a.fps, codec="libx264", quality=8,
                       macro_block_size=1)
    for k in tqdm(range(0, len(x), step), desc=f"[render] {os.path.basename(out)}",
                  unit="frame", ncols=100):
        w.append_data(draw(k))
    w.close()
    print(f"[test] wrote {out}")

    rep = os.path.join(out_dir, f"{name}_test.json")
    json.dump({"name": name, "checkpoint": ck_path, "eye_side": ck.get("eye_side"),
               "n_cells": int(model.N), "saccade": bool(a.saccade),
               "err_theta_mean_deg": float(err.mean()),
               "mean_abs_torsion_deg": float(np.abs(x[:, 2]).mean()),
               "mean_abs_phi_deg": phi_true,
               "phi_zero_in_movie": bool(a.phi_zero),
               "per_phase": dict(zip(labels, per)), "movie": out},
              open(rep, "w"), indent=2)
    print(f"[test] wrote {rep}")


if __name__ == "__main__":
    main()
