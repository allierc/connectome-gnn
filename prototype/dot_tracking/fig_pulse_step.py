#!/usr/bin/env python
"""fig_pulse_step -- did the controller learn the eye's inverse, or only to track?

    python fig_pulse_step.py                       # both eyes, writes the figure
    python fig_pulse_step.py --tags eyeG_deep

The trained command swings hard whenever the target reverses -- 3.9x the ordinary
command-to-gaze gap at the sharpest 3 % of turns -- while the gaze itself barely
degrades. The claim under test is that this is not instability but the plant's
INVERSE, and specifically Robinson's pulse-step: a pulse to drive the eye against
its viscosity, then a step to hold it there.

That claim is falsifiable, because the inverse has a closed form. The eye is

    xdd + C xd + K x = K x_inf                                            (1)

so the command that would place the gaze exactly on a desired trajectory x* is

    x_inf* = x* + K^-1 C x*d + K^-1 x*dd                                  (2)

-- position, plus a velocity term scaled by K^-1 C, plus an acceleration term
scaled by K^-1. A step in x* makes the velocity term a pulse: that IS the pulse-step,
falling out of the algebra rather than being designed in.

Four tests, in increasing strength:

  1. STEP PROBE. Drive the network with a velocity boxcar, so the target steps and
     holds. It has never seen a step -- the corpus is splines at three speeds -- and
     if the command shows a pulse followed by a step, the shape is not memorised.

  2. THE COEFFICIENTS. Regress the network's actual command on the TARGET and its
     first two derivatives, freely:  cmd ~ A0 x* + A1 x*d + A2 x*dd. If the network
     implements (2) then A0 = I, A1 = K^-1 C and A2 = K^-1, and these are matrices
     it was never told. Regressing on the ACHIEVED gaze instead would be circular --
     (1) makes that an identity -- which is why the regressor is the target.

  3. PLANT SPECIFICITY. Two eyes with different C and K. If the pulse were a habit of
     the architecture or of the corpus, its gain would be the same in both; if it is
     the inverse, each network's A1 tracks its OWN K^-1 C.

  4. NECESSITY. Drive each eye with the pulse-free command x* and compare the gaze
     it produces against the network's. If the pulse-free gaze tracks just as well,
     the pulse is decoration; if it lags, the loss had no choice.

Nothing in the architecture supplies x*d or x*dd. The readout is a static map of the
rates, so the velocity term has to be built from the input -- which IS the target
velocity, scaled -- and the position and acceleration terms have to be integrated and
differentiated out of it by the recurrent dynamics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_eyeG as TG                                      # noqa: E402
import test_eyeG as TE                                       # noqa: E402

BG = "#000000"
FG = "#ffffff"


def load(tag):
    ck = torch.load(os.path.join(TG.MODELS, f"{tag}.pt"), map_location="cpu",
                    weights_only=False)
    spec = {k: (np.asarray(v) if isinstance(v, list) else v)
            for k, v in ck["eye"].items()}
    for k, w in (("pairs", 2), ("pair_coef", 3)):
        if k in spec:
            spec[k] = np.asarray(spec[k], float if w == 3 else int).reshape(-1, w)
    eye = TG.EyeG(spec, ck["dt"])
    mdl = TG.CTRNNEyeG(eye, hidden=ck["hidden"], dt=ck["dt"])
    mdl.load_state_dict(ck["state"]); mdl.eval()
    return mdl, eye, np.asarray(ck["scale"], np.float32), ck["dt"]


def roll(mdl, eye, v):
    with torch.no_grad():
        x, m, R = mdl(torch.tensor(np.asarray(v, np.float32)[None]), want_states=True)
        cmd = eye.equilibrium(m)
    return [a[0].numpy() for a in (x, m, R, cmd)]


def drive_eye(eye, x_inf, dt):
    """Push a prescribed equilibrium trajectory through the eye's mechanics -- the
    forward half of (1), used to ask what a pulse-free command would actually do."""
    with torch.no_grad():
        return TG.rollout(torch.tensor(np.asarray(x_inf, np.float32))[None],
                          eye.K, eye.C, dt, eye.Minv)[0].numpy()


def d1(a, dt):
    return np.gradient(a, dt, axis=0)


def step_probe(mdl, eye, scale, dt, amp_deg=4.0, t_on=1.0, ramp=0.10, T=4.0):
    """A target that steps and holds -- a regime the corpus never contains."""
    n = int(T / dt)
    v = np.zeros((n, 2), np.float32)
    k0, kr = int(t_on / dt), max(1, int(ramp / dt))
    v[k0:k0 + kr, 0] = (amp_deg / scale[0]) / ramp        # boxcar in velocity
    x, m, R, cmd = roll(mdl, eye, v)
    tgt = np.cumsum(v, 0) * dt * scale
    return np.arange(n) * dt, tgt, cmd, x


def inverse_fit(cmd, tgt3, dt, eye):
    """Test 2: regress the command on the target and its first two derivatives."""
    X = np.concatenate([tgt3, d1(tgt3, dt), d1(d1(tgt3, dt), dt),
                        np.ones((len(tgt3), 1))], 1)          # (T, 10)
    B, *_ = np.linalg.lstsq(X, cmd, rcond=None)               # (10, 3)
    pred = X @ B
    r2 = 1 - ((pred - cmd) ** 2).sum() / ((cmd - cmd.mean(0)) ** 2).sum()
    K = eye.K.detach().numpy().astype(float)
    C = eye.C.detach().numpy().astype(float)
    Ki = np.linalg.inv(K)
    return dict(A0=B[0:3].T, A1=B[3:6].T, A2=B[6:9].T, r2=float(r2),
                A1_hat=Ki @ C, A2_hat=Ki)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", default=["eyeG_deep", "eyeG_light"])
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(TG.MODELS, "fig_pulse_step.png"))
    a = p.parse_args()

    res, report = {}, {}
    for tag in a.tags:
        mdl, eye, scale, dt = load(tag)
        # --- test 1: the step probe --------------------------------------
        t_s, tgt_s, cmd_s, x_s = step_probe(mdl, eye, scale, dt)
        # --- the corpus sequence, for tests 2 and 4 -----------------------
        v, xy, cuts = TE.sequence(a.duration, dt, scale, seed=a.seed)
        tgt = xy * scale
        x, m, R, cmd = roll(mdl, eye, v)
        tgt3 = np.concatenate([tgt, np.zeros((len(tgt), 1))], 1)
        fit = inverse_fit(cmd, tgt3, dt, eye)
        # --- test 4: what the pulse-free command would do -----------------
        gz_free = drive_eye(eye, tgt3, dt)
        e_net = np.linalg.norm(x[:, :2] - tgt, axis=1)
        e_free = np.linalg.norm(gz_free[:, :2] - tgt, axis=1)
        # U-turns, as in the audit
        hd = np.arctan2(v[:, 1], v[:, 0])
        turn = np.abs((np.diff(hd, prepend=hd[0]) + np.pi) % (2 * np.pi) - np.pi) / dt
        sp = np.linalg.norm(v, axis=1)
        mov = sp > 0.05 * sp.max()
        U = np.convolve((turn > np.percentile(turn[mov], 97)) & mov,
                        np.ones(9), "same") > 0
        res[tag] = dict(t_s=t_s, tgt_s=tgt_s, cmd_s=cmd_s, x_s=x_s, tgt=tgt, cmd=cmd,
                        x=x, U=U, e_net=e_net, e_free=e_free, fit=fit, dt=dt)
        report[tag] = dict(
            inverse_r2=fit["r2"],
            A1_measured=fit["A1"][:2, :2].tolist(), A1_analytic=fit["A1_hat"][:2, :2].tolist(),
            A2_measured=fit["A2"][:2, :2].tolist(), A2_analytic=fit["A2_hat"][:2, :2].tolist(),
            pulse_gain_s=float(np.trace(fit["A1"][:2, :2]) / 2),
            pulse_gain_analytic_s=float(np.trace(fit["A1_hat"][:2, :2]) / 2),
            err_network_deg=float(e_net.mean()),
            err_pulse_free_deg=float(e_free.mean()),
            err_network_uturn_deg=float(e_net[U].mean()),
            err_pulse_free_uturn_deg=float(e_free[U].mean()))
        f = report[tag]
        print(f"\n=== {tag} ===")
        print(f"  inverse fit  cmd ~ A0 x* + A1 x*' + A2 x*''      R2 = {fit['r2']:.4f}")
        print(f"  pulse gain   measured {f['pulse_gain_s']*1e3:7.1f} ms   "
              f"analytic K^-1C {f['pulse_gain_analytic_s']*1e3:7.1f} ms")
        print(f"  tracking     network {f['err_network_deg']:.3f} deg   "
              f"pulse-free {f['err_pulse_free_deg']:.3f} deg   "
              f"(U-turns {f['err_network_uturn_deg']:.3f} vs "
              f"{f['err_pulse_free_uturn_deg']:.3f})")

    # ----------------------------------------------------------------- figure
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 8.4), facecolor=BG)
    for x_ in ax.flat:
        x_.set_facecolor(BG)
        for s in x_.spines.values():
            s.set_color("#444")
        x_.tick_params(colors=FG, labelsize=10)
        x_.xaxis.label.set_color(FG); x_.yaxis.label.set_color(FG)

    tag0 = a.tags[0]
    r = res[tag0]
    # (a) the step probe
    A = ax[0, 0]
    A.plot(r["t_s"], r["tgt_s"][:, 0], "-", color=FG, lw=1.6, label="target  $\\theta^\\star$")
    A.plot(r["t_s"], r["cmd_s"][:, 0], "-", color="#e05a4a", lw=1.6,
           label="command  $\\Phi(m)$")
    A.plot(r["t_s"], r["x_s"][:, 0], "-", color="#4da3ff", lw=2.0, label="gaze  $\\theta$")
    A.set_xlabel("time (s)"); A.set_ylabel("horizontal (deg)")
    A.legend(frameon=False, labelcolor=FG, fontsize=9, loc="lower right")
    A.text(0.02, 0.97, "a   step probe — a regime the corpus never contains",
           transform=A.transAxes, color=FG, fontsize=10, va="top")

    # (b) command against the closed-form inverse, on the corpus sequence
    B = ax[0, 1]
    tgt3 = np.concatenate([r["tgt"], np.zeros((len(r["tgt"]), 1))], 1)
    Ki = r["fit"]["A2_hat"]; KiC = r["fit"]["A1_hat"]
    pred = (tgt3 + d1(tgt3, r["dt"]) @ KiC.T + d1(d1(tgt3, r["dt"]), r["dt"]) @ Ki.T)
    B.plot(pred[:, 0], r["cmd"][:, 0], ".", ms=1.4, color="#4da3ff", alpha=0.5)
    lim = np.percentile(np.abs(np.concatenate([pred[:, 0], r["cmd"][:, 0]])), 99.5)
    B.plot([-lim, lim], [-lim, lim], "-", color="#888", lw=1.0)
    rr = 1 - ((pred[:, :2] - r["cmd"][:, :2]) ** 2).sum() / \
        ((r["cmd"][:, :2] - r["cmd"][:, :2].mean(0)) ** 2).sum()
    B.set_xlabel("closed-form inverse  $x^\\star+K^{-1}C\\dot x^\\star+K^{-1}\\ddot x^\\star$ (deg)")
    B.set_ylabel("network's command (deg)")
    B.text(0.02, 0.97, f"b   no free parameters — $R^2$ = {rr:.3f}",
           transform=B.transAxes, color=FG, fontsize=10, va="top")

    # (c) the pulse gain of each network against its own plant
    C = ax[1, 0]
    tags = list(res)
    meas = [report[t]["pulse_gain_s"] * 1e3 for t in tags]
    anal = [report[t]["pulse_gain_analytic_s"] * 1e3 for t in tags]
    xi = np.arange(len(tags))
    C.bar(xi - 0.19, meas, 0.36, color="#e05a4a", label="measured  $A_1$")
    C.bar(xi + 0.19, anal, 0.36, color="#4da3ff", label="analytic  $K^{-1}C$")
    C.set_xticks(xi); C.set_xticklabels(tags, color=FG)
    C.set_ylabel("pulse gain (ms)")
    C.legend(frameon=False, labelcolor=FG, fontsize=9)
    C.text(0.02, 0.97, "c   each network tracks its own plant, not a habit",
           transform=C.transAxes, color=FG, fontsize=10, va="top")

    # (d) necessity
    D = ax[1, 1]
    w = 0.36
    for i, t in enumerate(tags):
        f = report[t]
        D.bar(i - 0.19, f["err_network_deg"], w, color="#4da3ff",
              label="network's command" if i == 0 else None)
        D.bar(i + 0.19, f["err_pulse_free_deg"], w, color="#e0a04a",
              label="pulse-free  $x^\\star$ alone" if i == 0 else None)
    D.set_xticks(np.arange(len(tags))); D.set_xticklabels(tags, color=FG)
    D.set_ylabel("mean |gaze $-$ target| (deg)")
    D.set_yscale("log")
    D.legend(frameon=False, labelcolor=FG, fontsize=9)
    D.text(0.02, 0.97, "d   drop the pulse and the eye lags",
           transform=D.transAxes, color=FG, fontsize=10, va="top")

    fig.tight_layout()
    fig.savefig(a.out, dpi=170, facecolor=BG)
    json.dump(report, open(a.out.replace(".png", ".json"), "w"), indent=2)
    print(f"\nwrote {a.out}\nwrote {a.out.replace('.png', '.json')}")


if __name__ == "__main__":
    main()
