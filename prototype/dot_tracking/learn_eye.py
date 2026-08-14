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

PLANT_NPZ = {"h": "/workspace/Plexus/prototype/eye/plant.npz",
             "v": "/workspace/Plexus/prototype/eye/plant_v.npz"}
DEFAULT_VARIANT = "eye_p3a_length"
BOUND = 0.95            # the arena half-width the target is clamped to


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

    def __init__(self, dt, plants=None, hidden=64, tau0=0.5):
        super().__init__()
        self.dt = dt
        self.plants = plants                     # {"h": EyePlant, "v": EyePlant}
        self.log_tau = nn.Parameter(torch.full((hidden,), float(np.log(tau0))))
        self.W = nn.Parameter(torch.zeros(hidden, hidden))
        self.enc = nn.Linear(2, hidden)          # (vx, vy)
        self.mot = nn.Linear(hidden, 4)          # -> LR, MR, SR, IR
        with torch.no_grad():
            self.enc.weight.mul_(0.5)

    def forward(self, u):                        # u: (B, T, 2) velocity
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
        m = nn.functional.softplus(self.mot(R))   # four pools, all >= 0
        cmd_h = m[..., 0] - m[..., 1]             # LR - MR
        cmd_v = m[..., 2] - m[..., 3]             # SR - IR
        if self.plants is None:
            return torch.stack([cmd_h, cmd_v], -1), m
        return torch.stack([self.plants["h"](cmd_h),
                            self.plants["v"](cmd_v)], -1), m


def _variant_entry(V, variant):
    if variant == "ideal_linear":
        # Same mechanics and same full-scale travel as p3a_length, but a
        # MONOTONE static curve. This is the control that decides whether the
        # residual error is Phi's non-invertibility or something a trained
        # controller simply has not learned yet: only Phi differs.
        ref = V["eye_p3a_length"]
        span = float(sum(ref["coef"]))
        return dict(coef=[span, 0.0], order=2, theta=ref["theta"],
                    rms=float("nan"))
    return V[variant]


def load_plants(variant, dt):
    """One plant per axis. The horizontal pair is LR/MR, the vertical SR/IR,
    and they are genuinely different — for eye C the travel is 15.0 deg
    horizontally against 9.1 vertically — so a single plant cannot serve
    both."""
    out, reach = {}, {}
    for ax, path in PLANT_NPZ.items():
        V = json.loads(str(np.load(path, allow_pickle=False)["variants"]))
        v = _variant_entry(V, variant)
        if int(v["order"]) != 2:
            raise SystemExit(f"{variant}/{ax} was fitted at order {v['order']}")
        wn, zeta = np.exp(np.asarray(v["theta"]))
        f = lambda u, c=v["coef"]: sum(ci * u ** (i + 1) for i, ci in enumerate(c))
        out[ax] = EyePlant(v["coef"], wn, zeta, dt)
        reach[ax] = min(abs(f(-1.0)), abs(f(1.0)))
        print(f"[plant] {variant}/{ax}: wn={wn:.2f} zeta={zeta:.3f}  "
              f"range {f(-1.0):+.1f}..{f(+1.0):+.1f} deg  reach {reach[ax]:.1f}")
    # ANISOTROPIC: each axis scaled to its OWN reach. A square world has to
    # take the smaller of the two, which for eye C throws away 40% of the
    # horizontal range to accommodate the vertical. The eye's travel is not
    # square — a laterally-placed eye moves far more horizontally than
    # vertically — so neither should the task be.
    dpu = {ax: reach[ax] / BOUND for ax in reach}
    print(f"[world] {dpu['h']:.2f} deg/unit horizontal (+-{dpu['h']*BOUND:.1f} deg), "
          f"{dpu['v']:.2f} vertical (+-{dpu['v']*BOUND:.1f} deg)")
    return out, dpu


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

    # --- data: UNCHANGED, now used on both axes ---------------------------
    plants, dpu = ((None, {"h": 15.0, "v": 15.0}) if a.no_plant
                   else load_plants(a.variant, a.dt))
    scale = np.array([dpu["h"], dpu["v"]], dtype=np.float32)
    if plants is not None:
        plants = {k: v.to(dev) for k, v in plants.items()}
    sp = {}
    for nm, n, s0 in (("train", 150, 0), ("val", 25, 5_000_000),
                      ("test", 40, 9_000_000)):
        u, y, c = learn.load_split(nm, n, a.duration, a.dt, s0)
        sp[nm] = (torch.as_tensor(u).to(dev),                     # (vx, vy)
                  torch.as_tensor(y * scale).to(dev))            # (x, y) deg
    (Utr, Ytr), (Uva, Yva), (Ute, Yte) = sp["train"], sp["val"], sp["test"]
    print(f"[data] anisotropic world: h x{dpu['h']:.2f}, v x{dpu['v']:.2f} deg/unit; "
          f"|target| max h {float(Ytr[...,0].abs().max()):.1f}, "
          f"v {float(Ytr[...,1].abs().max()):.1f} deg")
    torch.manual_seed(0)
    model = CTRNNEye(a.dt, plants).to(dev)
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
        err = torch.linalg.norm(pred - Yte, dim=-1)      # 2-D radial error
        err_h = (pred[..., 0] - Yte[..., 0]).abs()
        err_v = (pred[..., 1] - Yte[..., 1]).abs()
        cmd = torch.stack([m[..., 0] - m[..., 1], m[..., 2] - m[..., 3]], -1)
    tag = "no plant" if a.no_plant else a.variant
    json.dump({"variant": tag, "deg_per_unit": float(dpu["h"]),
               "deg_per_unit_h": float(dpu["h"]),
               "deg_per_unit_v": float(dpu["v"]),
               "gaze_err_mean_deg": float(err.mean()),
               "gaze_err_h_deg": float(err_h.mean()),
               "gaze_err_v_deg": float(err_v.mean()),
               "gaze_err_p95_deg": float(err.flatten().kthvalue(
                   int(0.95 * err.numel()))[0]),
               "cmd_abs_mean": float(cmd.abs().mean()),
               "cmd_saturated_frac": float((cmd.abs() >= 1).float().mean()),
               },
              open(os.path.join(learn.CKPT, "eye_report_"
                                + ("noplant" if a.no_plant else a.variant)
                                + ".json"), "w"), indent=2)
    print(f"\n[{tag}]  2-D gaze error {float(err.mean()):.3f} deg"
          f"   (h {float(err_h.mean()):.3f}, v {float(err_v.mean()):.3f})"
          f"   world +-{dpu['h']*BOUND:.1f}/{dpu['v']*BOUND:.1f} deg")
    print(f"   commands saturated {float((cmd.abs() >= 1).float().mean())*100:.1f}%"
          f" of samples   |cmd| max {float(cmd.abs().max()):.3f}")
    print("   motor pools (all >= 0): "
          + "  ".join(f"{n} {float(m[..., i].mean()):.3f}"
                      for i, n in enumerate(("LR", "MR", "SR", "IR"))))
    # One checkpoint per eye. A controller trained for one plant is not the
    # controller for another — it has learned that plant's inverse — so the
    # UI must pair each eye with its own network rather than reuse one.
    out = os.path.join(learn.CKPT, "ctrnn_eye_"
                       + ("noplant" if a.no_plant else a.variant) + ".pt")
    torch.save({"state_dict": model.state_dict(), "variant": tag,
                "dt": a.dt, "deg_per_unit_h": float(dpu["h"]),
                "deg_per_unit_v": float(dpu["v"])}, out)
    print(f"   saved {out}")


if __name__ == "__main__":
    main()
