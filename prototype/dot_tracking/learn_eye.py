"""Train the ctrnn THROUGH the eye plant, instead of straight to position.

The training data does not change. Same velocity input, same position target.
What changes is what sits between the network and the loss:

    before:  v -> [ctrnn] -> W_out -> position                 (loss here)
    now:     v -> [ctrnn] -> LR,MR >= 0 -> cmd -> [eye] -> gaze  (loss here)

Three things follow from that, and none of them is a data change.

1. THE READOUT BECOMES A MUSCLE COMMAND. The decoder no longer emits a
   position; it emits two non-negative drives, lateral and medial rectus, and
   the signed command is their difference. Non-negativity is not decoration —
   motor pools cannot fire negatively, and forcing the push-pull here is what
   makes `LR - MR` a mechanical fact rather than a fitted convention.

2. THE TASK BECOMES ONE-DIMENSIONAL. Only the horizontal muscle pair was
   identified, so only horizontal gaze is available. The 2-D dot task is
   projected onto its x axis. Fitting the vertical plant from the SR/IR
   probes in the same archive would restore the second dimension.

3. THE NETWORK MUST NOW INVERT THE PLANT. The eye lags and rings — at
   zeta ~ 0.31, wn ~ 10 rad/s it overshoots and oscillates at ~1.5 Hz. To put
   gaze where the target is, the command has to lead it. So the network is no
   longer learning integration alone; it is learning integration composed
   with an inverse model of its own body. That is the interesting part, and
   it is why this is worth doing rather than bolting the plant on at test
   time.

Which eye. `eye_p3a_length` is used rather than the best-fitting variant:
`baseline_fixmat` fits to 0.41 deg but reaches only +2.9 deg one way against
-10.3 the other, so most of the task would be outside its workspace.
`p3a_length` is nearly symmetric (-15.2 / +15.9 deg) and still fits to
0.62 deg. Best-fitting is not the same as best-suited.

CAVEAT, inherited from the identification: the static curve `f` rests on two
plateaus per variant, so its shape between the endpoints is extrapolation.
The dynamics are sound; the gain is provisional until a staircase protocol
is run. Results here are therefore about whether the network CAN invert a
plant, not about this eye's exact numbers.

Usage::

    python learn_eye.py                       # train through the plant
    python learn_eye.py --no-plant            # same task, no plant (control)
    python learn_eye.py --variant eye_p3b_pulley
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import learn                                          # noqa: E402

PLANT_NPZ = "/workspace/Plexus/prototype/eye/plant.npz"
DEFAULT_VARIANT = "eye_p3a_length"
DEG_PER_UNIT = 15.0     # grid units -> degrees; +-0.95 grid stays in range


class EyePlant(nn.Module):
    """Frozen Hammerstein plant: signed command -> horizontal gaze (deg).

        f      = sum_k coef[k] * u^(k+1)                  static, measured
        y'' + 2 zeta wn y' + wn^2 y = wn^2 f              linear, fitted

    Differentiable and cheap: one polynomial and a two-state linear recursion
    per step. Parameters are buffers, not parameters — the body is measured,
    not learned, and letting the optimiser retune it would defeat the point.
    """

    def __init__(self, coef, wn, zeta, dt):
        super().__init__()
        self.register_buffer("coef", torch.as_tensor(coef, dtype=torch.float32))
        self.wn, self.zeta, self.dt = float(wn), float(zeta), float(dt)

    def static(self, u):
        return sum(c * u ** (k + 1) for k, c in enumerate(self.coef))

    def forward(self, cmd):                      # cmd: (B, T)
        f = self.static(cmd.clamp(-1.0, 1.0))
        B, T = cmd.shape
        y = torch.zeros(B, device=cmd.device, dtype=cmd.dtype)
        v = torch.zeros_like(y)
        out = []
        w2, twz, dt = self.wn ** 2, 2 * self.zeta * self.wn, self.dt
        for t in range(T):
            out.append(y)
            v = v + dt * (w2 * (f[:, t] - y) - twz * v)
            y = y + dt * v
        return torch.stack(out, 1)


class CTRNNEye(nn.Module):
    """ctrnn core, push-pull motor readout, then the eye.

    The core is identical to `learn.CTRNN` — same equation, same zero-init
    recurrent matrix, same learned per-neuron tau — so any difference in
    result is attributable to the plant and not to the network.
    """

    def __init__(self, dt, plant=None, hidden=64, tau0=0.5):
        super().__init__()
        self.dt = dt
        self.plant = plant
        self.log_tau = nn.Parameter(torch.full((hidden,), float(np.log(tau0))))
        self.W = nn.Parameter(torch.zeros(hidden, hidden))
        self.enc = nn.Linear(1, hidden)
        self.mot = nn.Linear(hidden, 2)          # -> LR, MR drives
        with torch.no_grad():
            self.enc.weight.mul_(0.5)

    def forward(self, u):                        # u: (B, T, 1) velocity
        alpha = (self.dt / self.log_tau.exp()).clamp(1e-4, 1.0)
        B, T, _ = u.shape
        h = torch.zeros(B, self.W.shape[0], device=u.device, dtype=u.dtype)
        drive = self.enc(u)
        rates = []
        for t in range(T):
            r = torch.tanh(h)
            rates.append(r)
            h = h + alpha * (-h + r @ self.W.T + drive[:, t])
        R = torch.stack(rates, 1)
        m = nn.functional.softplus(self.mot(R))   # motor pools are >= 0
        cmd = m[..., 0] - m[..., 1]               # push-pull, LR - MR
        if self.plant is None:
            return cmd * DEG_PER_UNIT, m          # control: command is gaze
        return self.plant(cmd), m


def load_plant(variant, dt):
    z = np.load(PLANT_NPZ, allow_pickle=False)
    V = json.loads(str(z["variants"]))
    v = V[variant]
    if int(v["order"]) != 2:
        raise SystemExit(f"{variant} was fitted at order {v['order']}")
    wn, zeta = np.exp(np.asarray(v["theta"]))
    f = lambda u: sum(c * u ** (i + 1) for i, c in enumerate(v["coef"]))
    print(f"[plant] {variant}: wn={wn:.2f} rad/s  zeta={zeta:.3f}  "
          f"range {f(-1.0):+.1f}..{f(+1.0):+.1f} deg  (fit RMS {v['rms']:.2f})")
    return EyePlant(v["coef"], wn, zeta, dt)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default=DEFAULT_VARIANT)
    p.add_argument("--no-plant", action="store_true",
                   help="control: same 1-D task, no plant in the loop")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    a = p.parse_args()
    dev = torch.device(a.device)

    # --- data: UNCHANGED, only projected onto the horizontal axis ---------
    sp = {}
    for nm, n, s0 in (("train", 150, 0), ("val", 25, 5_000_000),
                      ("test", 40, 9_000_000)):
        u, y, c = learn.load_split(nm, n, a.duration, a.dt, s0)
        sp[nm] = (torch.as_tensor(u[..., :1]).to(dev),            # vx only
                  torch.as_tensor(y[..., 0] * DEG_PER_UNIT).to(dev))
    (Utr, Ytr), (Uva, Yva), (Ute, Yte) = sp["train"], sp["val"], sp["test"]
    print(f"[data] horizontal only, target scaled x{DEG_PER_UNIT:g} deg/unit; "
          f"|target| max {float(Ytr.abs().max()):.1f} deg")

    plant = None if a.no_plant else load_plant(a.variant, a.dt).to(dev)
    torch.manual_seed(0)
    model = CTRNNEye(a.dt, plant).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    T = Utr.shape[1]
    sched = [max(60, int(T * f)) for f in (0.25, 0.5, 0.75, 1.0)]
    best, best_state = np.inf, None
    for ep in range(a.epochs):
        model.train()
        h = sched[min(3, int(4 * ep / max(a.epochs - 1, 1)))]
        perm = torch.randperm(Utr.shape[0], device=dev)
        for i in range(0, len(perm), a.batch):
            ix = perm[i:i + a.batch]
            pred, _ = model(Utr[ix, :h])
            loss = ((pred - Ytr[ix, :h]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            va = float(((model(Uva)[0] - Yva) ** 2).mean())
        if va < best:
            best, best_state = va, {k: v.detach().clone()
                                    for k, v in model.state_dict().items()}
        if ep % max(1, a.epochs // 6) == 0 or ep == a.epochs - 1:
            print(f"  ep {ep:3d}  horizon {h:3d}  val MSE {va:9.4f} deg^2")
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred, m = model(Ute)
        err = (pred - Yte).abs()
        cmd = (m[..., 0] - m[..., 1])
    tag = "no plant" if a.no_plant else a.variant
    print(f"\n[{tag}]  test mean |gaze error| {float(err.mean()):.3f} deg"
          f"   p95 {float(err.flatten().kthvalue(int(0.95*err.numel()))[0]):.3f}"
          f"   as a fraction of the +-{DEG_PER_UNIT:.0f} deg workspace: "
          f"{float(err.mean())/DEG_PER_UNIT*100:.1f}%")
    print(f"   command |LR-MR| mean {float(cmd.abs().mean()):.3f} "
          f"max {float(cmd.abs().max()):.3f}   saturated "
          f"{float((cmd.abs() >= 1).float().mean())*100:.1f}% of samples")
    print(f"   motor pools: LR mean {float(m[...,0].mean()):.3f}, "
          f"MR mean {float(m[...,1].mean()):.3f}  (both >= 0 by construction)")
    out = os.path.join(learn.CKPT, f"ctrnn_eye{'_noplant' if a.no_plant else ''}.pt")
    torch.save({"state_dict": model.state_dict(), "variant": tag,
                "dt": a.dt, "deg_per_unit": DEG_PER_UNIT}, out)
    print(f"   saved {out}")


if __name__ == "__main__":
    main()
