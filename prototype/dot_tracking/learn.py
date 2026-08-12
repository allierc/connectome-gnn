"""Learn the open-loop integrator: dataset, models, training, evaluation.

Stage 1 of the plan in ``docs/oculomotor_circuit_and_task.md`` §4. These
networks are NOT the model — they are the calibration. They answer what an
unconstrained learner can do on this task, so that the connectome-constrained
circuit trained later can be compared against a ceiling rather than against
nothing.

The task is supervised, dense, and open loop: input is the target's velocity
`(vx, vy)`, output is its position `(x, y)`, and the correct answer is the
integral of the input. The environment never reacts to the network, so this
is sequence-to-sequence regression trained by BPTT — no reinforcement
anywhere.

Architecture is deliberately the shape the biology will take: a small linear
ENCODER, a recurrent CORE, a small linear DECODER. Only the core changes
between models, so the comparison isolates the memory mechanism.

    intlin   learned leaky integrator: g <- a*g + B u.  Six parameters, and
             the exactly-right model class. Its fitted `a` is a direct read of
             the time constant the task demands.
    mlp      feedforward over a sliding window of the last W velocity samples.
             HAS NO STATE, so it can only integrate what fits in its window —
             included precisely because it must fail, which is what makes the
             failure of memoryless models measurable rather than assumed.
    rnn      vanilla discrete tanh RNN. Included as a negative result: it
             cannot learn this task, and the reason is optimisation rather
             than capacity.
    ctrnn    continuous-time rate network, tau dh/dt = -h + W r + W_in u —
             the same equation as the connectome model, minus the sign-lock
             and the anatomy. THE stand-in for stage 1.
    gru      gated core: the strong baseline, and an upper bound nobody
             claims is biological.

Usage::

    python learn.py --build                 # write the dataset only
    python learn.py                         # build if needed, train all four
    python learn.py --models rnn gru --epochs 60
    python learn.py --eval                  # re-score existing checkpoints
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory import SPEC, generate                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CKPT = os.path.join(HERE, "models")
FOV = 0.6

# Every combination of the four trajectory switches, always centre-started.
# 2 shapes x 2 motions x 3 speeds x 2 angles = 24 conditions; the dataset is
# balanced across them so no regime is over-represented in the loss.
CONDITIONS = [{"shape": sh, "motion": mo, "speed": sp, "angle": an}
              for sh in SPEC["shape"] for mo in SPEC["motion"]
              for sp in SPEC["speed"] for an in SPEC["angle"]]


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def build_dataset(n_per_cond, duration, dt, seed0, path):
    """(N, T, 2) velocities and (N, T, 2) positions, balanced over conditions.

    Seeds are disjoint across splits by construction (`seed0` offsets each),
    so no trajectory is ever shared between train and test — the trajectories
    are deterministic in their seed, and reusing one would be leakage.
    """
    U, Y, C = [], [], []
    for ci, cond in enumerate(CONDITIONS):
        for k in range(n_per_cond):
            tr = generate(start="center", duration=duration, dt=dt,
                          seed=seed0 + ci * 100000 + k, **cond)
            x = np.asarray(tr["x"], np.float32)
            y = np.asarray(tr["y"], np.float32)
            U.append(np.stack([np.gradient(x, dt), np.gradient(y, dt)], -1))
            Y.append(np.stack([x, y], -1))
            C.append(ci)
    U = np.asarray(U, np.float32); Y = np.asarray(Y, np.float32)
    C = np.asarray(C, np.int64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, u=U, y=Y, cond=C, dt=dt, duration=duration,
                        conditions=json.dumps(CONDITIONS))
    print(f"[data] {path}  u{U.shape} y{Y.shape}  "
          f"{U.nbytes / 1e6:.0f} MB in memory")
    return U, Y, C


def load_split(name, n_per_cond, duration, dt, seed0, force=False):
    path = os.path.join(DATA, f"{name}.npz")
    if os.path.isfile(path) and not force:
        z = np.load(path, allow_pickle=False)
        if (z["u"].shape[0] == n_per_cond * len(CONDITIONS)
                and abs(float(z["duration"]) - duration) < 1e-9):
            print(f"[data] reuse {path}  u{z['u'].shape}")
            return z["u"], z["y"], z["cond"]
    return build_dataset(n_per_cond, duration, dt, seed0, path)


# --------------------------------------------------------------------------
# models — linear encoder, swappable core, linear decoder
# --------------------------------------------------------------------------
class IntLin(nn.Module):
    """g <- a*g + B u ;  out = C g.  The right model class, six parameters.

    `a` is stored as a logit and squashed, so the integrator can approach but
    not exceed a = 1: a perfect integrator is reachable in the limit, an
    unstable one is not. The fitted time constant is tau = -dt / ln(a).
    """
    kind = "recurrent"

    def __init__(self, dt, **kw):
        super().__init__()
        self.dt = dt
        self.a_logit = nn.Parameter(torch.tensor(4.0))
        self.B = nn.Linear(2, 2, bias=False)
        self.C = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.B.weight.copy_(torch.eye(2) * dt)
            self.C.weight.copy_(torch.eye(2))

    def forward(self, u):
        a = torch.sigmoid(self.a_logit)
        g = torch.zeros(u.shape[0], 2, device=u.device, dtype=u.dtype)
        out = []
        Bu = self.B(u)
        for t in range(u.shape[1]):
            out.append(self.C(g))
            g = a * g + Bu[:, t]
        return torch.stack(out, 1)

    def tau(self):
        a = float(torch.sigmoid(self.a_logit))
        return float("inf") if a >= 1.0 else -self.dt / np.log(max(a, 1e-12))


class Windowed(nn.Module):
    """MLP over the last `win` velocity samples. Deliberately memoryless.

    Position is the integral of everything since t=0, so a fixed window can
    at best reconstruct the last `win*dt` seconds of displacement. This model
    is here to make that ceiling measurable instead of assumed.
    """
    kind = "windowed"

    def __init__(self, dt, hidden=128, win=30, **kw):
        super().__init__()
        self.win = win
        self.net = nn.Sequential(nn.Linear(2 * win, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, u):
        B, T, _ = u.shape
        pad = torch.zeros(B, self.win - 1, 2, device=u.device, dtype=u.dtype)
        z = torch.cat([pad, u], 1).unfold(1, self.win, 1)      # (B,T,2,win)
        return self.net(z.reshape(B, T, -1))


class Recurrent(nn.Module):
    """Linear encoder -> recurrent core -> linear decoder.

    The shape the connectome model will take: the encoder is the analogue of
    the input gate onto the AF5 afferents, the decoder of the readout off
    AMN/AIN. Only `core` differs between the rnn and gru variants, so any
    difference between them is attributable to the memory mechanism and not
    to the interface.
    """
    kind = "recurrent"

    def __init__(self, dt, hidden=64, cell="rnn", **kw):
        super().__init__()
        self.enc = nn.Linear(2, hidden)
        self.core = (nn.RNN(hidden, hidden, batch_first=True,
                            nonlinearity="tanh") if cell == "rnn"
                     else nn.GRU(hidden, hidden, batch_first=True))
        self.dec = nn.Linear(hidden, 2)
        if cell == "rnn":
            # Identity recurrent init (IRNN, Le et al. 2015) with small input
            # weights. With the default random W_hh a tanh RNN cannot learn
            # this task at all — it plateaus at MSE 0.113 after 250 epochs,
            # because the gradient of an 8 s integral vanishes through a
            # contracting recurrent map. Starting AT the identity makes each
            # unit a perfect integrator before training, so the optimiser
            # only has to learn the encoder, the decoder and the departures
            # from identity. This is an optimisation fix, not extra capacity.
            with torch.no_grad():
                self.core.weight_hh_l0.copy_(torch.eye(hidden))
                self.core.weight_ih_l0.mul_(0.1)
                self.core.bias_hh_l0.zero_(); self.core.bias_ih_l0.zero_()
                self.enc.weight.mul_(0.1)

    def forward(self, u):
        h, _ = self.core(self.enc(u))
        return self.dec(h)


class CTRNN(nn.Module):
    """Continuous-time rate network, the form the connectome model takes:

        tau dh/dt = -h + W r + W_in u ,   r = tanh(h) ,   out = D r

    discretised as h <- h + (dt/tau)(-h + W r + W_in u). This is the same
    equation as `zebrafish_hd_si`, minus the sign-lock and the connectome —
    so it is the honest stand-in for stage 1, and swapping W for the 285-cell
    matrix is the only change stage 2 needs.

    It is also far better conditioned than a discrete tanh RNN. With dt/tau
    small the update is near-identity, so the gradient of a long integral
    neither vanishes nor explodes at initialisation; W starts at zero, which
    makes the network a bank of leaky integrators of the input before any
    training happens. tau is learned per neuron (log-parameterised, so it
    stays positive) because the time constant is the quantity of interest.
    """
    kind = "recurrent"

    def __init__(self, dt, hidden=64, tau0=0.5, **kw):
        super().__init__()
        self.dt = dt
        self.log_tau = nn.Parameter(torch.full((hidden,), float(np.log(tau0))))
        self.W = nn.Parameter(torch.zeros(hidden, hidden))
        self.enc = nn.Linear(2, hidden)
        self.dec = nn.Linear(hidden, 2)
        with torch.no_grad():
            self.enc.weight.mul_(0.5)

    def forward(self, u):
        alpha = (self.dt / self.log_tau.exp()).clamp(1e-4, 1.0)
        B, T, _ = u.shape
        h = torch.zeros(B, self.W.shape[0], device=u.device, dtype=u.dtype)
        drive = self.enc(u)
        out = []
        for t in range(T):
            r = torch.tanh(h)
            out.append(self.dec(r))
            h = h + alpha * (-h + r @ self.W.T + drive[:, t])
        return torch.stack(out, 1)

    def tau(self):
        return float(self.log_tau.exp().median())


def make(name, dt):
    if name == "intlin":
        return IntLin(dt)
    if name == "mlp":
        return Windowed(dt)
    if name == "rnn":
        return Recurrent(dt, cell="rnn")
    if name == "gru":
        return Recurrent(dt, cell="gru")
    if name == "ctrnn":
        return CTRNN(dt)
    raise ValueError(f"unknown model {name!r}")


MODELS = ["intlin", "mlp", "rnn", "ctrnn", "gru"]


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train(name, Utr, Ytr, Uva, Yva, dt, epochs, batch, lr, device,
          curriculum=True):
    """BPTT with a horizon curriculum.

    Short horizons first: the gradient of a long integral through a recurrent
    core is ill-conditioned at initialisation, and letting the model first
    learn to integrate over 1 s makes the 8 s case reachable. This is the same
    device the heading-direction configs use (`n_steps_schedule`).
    """
    torch.manual_seed(0)
    model = make(name, dt).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    T = Utr.shape[1]
    n = Utr.shape[0]
    sched = ([max(60, int(T * f)) for f in (0.25, 0.5, 0.75, 1.0)]
             if curriculum else [T])
    best, best_state = np.inf, None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        h = sched[min(len(sched) - 1, int(len(sched) * ep / max(epochs - 1, 1)))]
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            ix = perm[i:i + batch]
            u, y = Utr[ix, :h], Ytr[ix, :h]
            pred = model(u)
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(ix)
        model.eval()
        with torch.no_grad():
            va = float(((model(Uva) - Yva) ** 2).mean())
        if va < best:
            best = va
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if ep % max(1, epochs // 8) == 0 or ep == epochs - 1:
            print(f"  [{name}] ep {ep:3d}  horizon {h:4d}  "
                  f"train {tot / n:.5f}  val {va:.5f}")
    model.load_state_dict(best_state)
    print(f"  [{name}] best val MSE {best:.5f}   ({time.time() - t0:.0f}s)")
    return model, best


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, U, Y, C, dt, device):
    """Test error plus the two quantities that matter operationally: how far
    the reconstruction drifts, and how long it stays inside the fovea."""
    model.eval()
    pred = torch.cat([model(U[i:i + 512]) for i in range(0, len(U), 512)])
    err = torch.linalg.norm(pred - Y, dim=-1)                  # (N, T)
    lost = err > FOV
    T = err.shape[1]
    first = torch.where(lost.any(1), lost.float().argmax(1), torch.tensor(T))
    t_lose = first.float().cpu().numpy() * dt
    never = (~lost.any(1)).float().mean().item()
    out = {
        "mse": float(((pred - Y) ** 2).mean()),
        "err_mean": float(err.mean()),
        "err_final": float(err[:, -1].mean()),
        "frac_never_lost": never,
        "t_lose_median": float(np.median(t_lose)),
        "per_condition": {},
    }
    cc = C.cpu().numpy()
    for ci, cond in enumerate(CONDITIONS):
        m = cc == ci
        if m.any():
            out["per_condition"][" ".join(cond.values())] = {
                "err_mean": float(err[m].mean()),
                "frac_never_lost": float((~lost[m].any(1)).float().mean()),
            }
    return out


@torch.no_grad()
def tau_probe(model, dt, device, drive_s=2.0, hold_s=20.0, v=0.4):
    """Hold-and-decay: drive at constant velocity, then stop, and measure how
    the reported position decays.

    This is the quantity §4.2 warns cannot be read off a short training trial.
    Driving and then removing the input exposes the integrator's own time
    constant directly, whatever horizon the model was trained on. Returns the
    time for the held position to fall to 1/e of its value at input offset;
    `inf` means it never does within `hold_s`.
    """
    nd, nh = int(drive_s / dt), int(hold_s / dt)
    u = torch.zeros(1, nd + nh, 2, device=device)
    u[0, :nd, 0] = v
    p = model(u)[0, :, 0].cpu().numpy()
    p0 = p[nd - 1]
    if abs(p0) < 1e-6:
        return {"tau_e": 0.0, "hold_gain": 0.0}
    tail = p[nd:] / p0
    below = np.flatnonzero(tail < 1.0 / np.e)
    return {"tau_e": float(below[0] * dt) if below.size else float("inf"),
            "hold_gain": float(p0 / (v * drive_s))}


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--n-train", type=int, default=150,
                   help="trajectories per condition (x24 conditions)")
    p.add_argument("--n-val", type=int, default=25)
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--build", action="store_true", help="build data and stop")
    p.add_argument("--eval", action="store_true",
                   help="score existing checkpoints without training")
    p.add_argument("--rebuild", action="store_true")
    a = p.parse_args()

    os.makedirs(DATA, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
    splits = {}
    for nm, n, s0 in (("train", a.n_train, 0), ("val", a.n_val, 5_000_000),
                      ("test", a.n_test, 9_000_000)):
        splits[nm] = load_split(nm, n, a.duration, a.dt, s0, force=a.rebuild)
    if a.build:
        return

    dev = torch.device(a.device)
    tens = {k: tuple(torch.as_tensor(x).to(dev) for x in v)
            for k, v in splits.items()}
    (Utr, Ytr, _), (Uva, Yva, _), (Ute, Yte, Cte) = (
        tens["train"], tens["val"], tens["test"])
    print(f"[setup] device={dev}  train={tuple(Utr.shape)}  "
          f"test={tuple(Ute.shape)}  {len(CONDITIONS)} conditions")

    report = {}
    for name in a.models:
        path = os.path.join(CKPT, f"{name}.pt")
        if a.eval:
            if not os.path.isfile(path):
                print(f"  [{name}] no checkpoint, skipped")
                continue
            blob = torch.load(path, map_location=dev, weights_only=False)
            model = make(name, a.dt).to(dev)
            model.load_state_dict(blob["state_dict"])
        else:
            print(f"\n[train] {name}")
            model, _ = train(name, Utr, Ytr, Uva, Yva, a.dt, a.epochs,
                             a.batch, a.lr, dev)
            torch.save({"state_dict": model.state_dict(), "model": name,
                        "dt": a.dt, "duration": a.duration}, path)
            print(f"  [{name}] saved {path}")
        m = evaluate(model, Ute, Yte, Cte, a.dt, dev)
        m.update(tau_probe(model, a.dt, dev))
        if hasattr(model, "tau"):
            m["tau_fitted"] = model.tau()
        report[name] = m

    # Score the analytic controllers on the SAME test split, so the numbers
    # in the UI are comparable across hand-written and learned solutions
    # rather than each being measured on its own trajectories.
    import openloop as OL
    Ute_np = Ute.cpu().numpy(); Yte_np = Yte.cpu().numpy()
    for nm in OL.OPEN_LOOP:
        errs, lost_any, tl = [], [], []
        for i in range(len(Ute_np)):
            x, y = Yte_np[i, :, 0], Yte_np[i, :, 1]
            gx, gy, _, _ = OL.OPEN_LOOP[nm](
                np.arange(len(x)) * a.dt, Ute_np[i, :, 0], Ute_np[i, :, 1],
                x[0], y[0], a.dt, np.random.default_rng(i), **OL.defaults(nm))
            e = np.hypot(x - gx, y - gy)
            errs.append(e.mean()); L = np.flatnonzero(e > FOV)
            lost_any.append(L.size > 0)
            tl.append(L[0] * a.dt if L.size else len(x) * a.dt)
        report[nm] = {"mse": float("nan"), "err_mean": float(np.mean(errs)),
                      "err_final": float("nan"),
                      "frac_never_lost": float(1.0 - np.mean(lost_any)),
                      "t_lose_median": float(np.median(tl)),
                      "tau_e": float("nan"), "analytic": True}

    print("\n" + "=" * 78)
    print(f"{'model':8s}{'test MSE':>11s}{'mean |err|':>12s}"
          f"{'never lost':>12s}{'t_lose med':>12s}{'tau_e (s)':>12s}")
    print("-" * 78)
    for k, m in report.items():
        te = m["tau_e"]
        mse = m["mse"]
        print(f"{k:8s}{'--' if np.isnan(mse) else f'{mse:.5f}':>11s}"
              f"{m['err_mean']:12.4f}"
              f"{m['frac_never_lost'] * 100:11.1f}%"
              f"{m['t_lose_median']:12.2f}"
              f"{'inf' if np.isinf(te) else f'{te:.2f}':>12s}")
    with open(os.path.join(CKPT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {os.path.join(CKPT, 'report.json')}")


if __name__ == "__main__":
    main()
