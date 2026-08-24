#!/usr/bin/env python
"""train_zebra_eyeG -- the REAL connectome-constrained circuit, through eye G.

    python train_zebra_eyeG.py
    python train_zebra_eyeG.py --epochs 250 --tag long

train_eyeG.py's CTRNNEyeG is explicitly the free prototype (section 5.2 of the
oculomotor note: "what the eye does coupled to a ctRNN, NOT the zebrafish
circuit") -- a (hidden, hidden) matrix with no anatomy in it at all. This script
is the other half: the 285-cell pool of `config/zebrafish/zebrafish_om_intg_285_v1
.yaml`, sign-locked to the real connectome (eq:sign-lock), through the SAME frozen
eye. Everything about the eye -- EyeG, fit_eye, rollout, eq:mechanics -- is
imported from train_eyeG.py unchanged; only the circuit changes.

WHERE THE CIRCUIT COMES FROM. `zebrafish_circuit.load_oculomotor_circuit` runs
the exact same pickle load, cell selection and Dale-sign assignment as
`scripts/plot_oculomotor_connectome.py --config` -- the script that draws
Figure 1 of the note. Same pickle, same yaml, same sort order, same sign
convention: what this trains is provably the circuit the figure shows, not a
re-derivation of it. See `zebrafish_circuit.py`'s own docstring for what that
script is (and, importantly, is NOT: a registered, GNN_Main.py-trainable
circuit -- the connectome CSVs, registry entry and AF5-in/AMN+AIN-out task
binding a "proper" run would need don't exist yet; see
`docs/HOWTO_add_oculomotor_circuit.md`).

ONLY TWO OF SIX MUSCLES ARE REACHABLE FROM THIS POOL. AMN -> LR, AIN -> MR;
SR/IR/SO/IO have no cell type in these 285 cells (they are driven by OMN,
which section 1.6 of the note leaves out). The other four muscle channels are
therefore held at EXACTLY zero drive -- not softplus(0), which would fake a
nonzero tonic contraction that does not exist in this circuit -- so vertical
gaze and torsion get no direct drive at all from this controller. That
degradation is the actual scientific content of this script, not a bug.

ONLY THE MAGNITUDES ARE LEARNED (eq:sign-lock): `S` is a free (285, 285)
parameter, masked to the measured synapses (never a new edge) and initialised
from the connectome's own weight (`training.w_init_mode: w_con` in the yaml),
but its SIGN is fixed per presynaptic cell type from the Dale assignment.
Nothing here can flip a measured excitatory cell to inhibitory.
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
import train_eyeG as TG                                  # noqa: E402
from zebrafish_circuit import (                           # noqa: E402
    load_oculomotor_circuit, DEFAULT_CONFIG, DEFAULT_PKL, MUSCLES,
)

MODELS = TG.MODELS


# ---------------------------------------------------------------------------
# the circuit
# ---------------------------------------------------------------------------
class ZebrafishCircuitRNN(nn.Module):
    """The connectome-constrained circuit of the note's section 4 (steps 1-5),
    eq:input-map / eq:circuit / eq:sign-lock / eq:output-map, in place of
    CTRNNEyeG's free (hidden, hidden) matrix.

        I(t)        = W_in p_dot(t)                    W_in zero outside AF5
        tau_i dv_i  = -v_i + sum_j W_hat_ij r_j + I_i   r_j = tanh(v_j)
        W_hat_ij    = |S_ij| sign(W_con_ij)             S masked to real synapses
        m(t)        = [W_out r(t)]_+                    only AMN->LR, AIN->MR

    `circuit` is a `zebrafish_circuit.load_oculomotor_circuit()` dict: the
    real 285-cell pool, not a placeholder. Every buffer below (`sign_col`,
    `support`, `afferent_mask`, the two output masks) is fixed by the
    connectome and the yaml's cell_types; only `S`, `Win` (on AF5 rows only)
    and `Wout` (on AMN/AIN columns only) are `nn.Parameter`s.
    """

    def __init__(self, circuit, tau0=0.1, dt=1 / 60):
        super().__init__()
        self.dt = dt
        self.names = circuit["names"]
        N = len(self.names)
        self.N = N

        self.log_tau = nn.Parameter(torch.full((N,), float(np.log(tau0))))
        self.register_buffer("sign_col", torch.as_tensor(circuit["sign_col"]))
        self.register_buffer("support", torch.as_tensor(circuit["support"],
                                                         dtype=torch.float32))
        # S initialised from the connectome's own weight (training.w_init_mode:
        # w_con in the yaml) -- the magnitude starts AT the measured synapse
        # strength and is free to move away from it; the sign never does.
        self.S = nn.Parameter(torch.as_tensor(circuit["W_mag"]).clone())

        afferent = torch.as_tensor(circuit["afferent"], dtype=torch.float32)
        self.register_buffer("afferent_mask", afferent)
        self.Win = nn.Parameter(torch.randn(N, 2) * 0.1 * afferent[:, None])

        # Wout is (2, N): row 0 -> LR (from AMN cells only), row 1 -> MR (from
        # AIN cells only). No cross terms -- an AMN cell cannot feed MR.
        eff = circuit["effector_col"]                    # {"AMN": 0 (LR), "AIN": 2 (MR)}
        out_idx = circuit["output_idx"]                  # {"AMN": rows, "AIN": rows}
        row_of_muscle = {0: 0, 2: 1}                      # MUSCLES index -> Wout row
        out_mask = np.zeros((2, N), np.float32)
        self.muscle_row = {}                              # Wout row -> index into MUSCLES(6)
        for name, muscle_idx in eff.items():
            if muscle_idx not in row_of_muscle:
                continue                                   # only LR/MR are wired below
            row = row_of_muscle[muscle_idx]
            out_mask[row, out_idx[name]] = 1.0
            self.muscle_row[row] = muscle_idx
        self.register_buffer("out_mask", torch.as_tensor(out_mask))
        self.Wout = nn.Parameter(torch.randn(2, N) * 0.1 * torch.as_tensor(out_mask))

    @property
    def n_act(self):
        return 6                                           # EyeG always takes six

    def W_hat(self):
        """eq:sign-lock: |S| on the measured synapses only, signed by the
        presynaptic cell's Dale assignment."""
        return (self.S.abs() * self.support) * self.sign_col[None, :]

    def forward(self, pdot, want_states=False):
        """pdot = (x_dot, y_dot), eq:input-vector, the only input. Returns
        (u, m) -- u the gaze, m the six muscle drives (four of them always
        zero) -- or (u, m, R) with the firing rates too if `want_states`."""
        alpha = (self.dt / self.log_tau.exp()).clamp(1e-4, 1.0)   # dt / tau_i
        B, T, _ = pdot.shape
        v = torch.zeros(B, self.N, device=pdot.device, dtype=pdot.dtype)  # v_i(0)=0
        Win_eff = self.Win * self.afferent_mask[:, None]
        I = pdot @ Win_eff.T                              # (B, T, N), eq:input-map
        What = self.W_hat()
        rates = []
        for t in range(T):
            r = torch.tanh(v)                              # rho = tanh, eq:circuit
            rates.append(r)
            v = v + alpha * (-v + r @ What.T + I[:, t])     # eq:euler-circuit
        R = torch.stack(rates, 1)                          # (B, T, N)
        Wout_eff = self.Wout * self.out_mask
        drive = nn.functional.softplus(R @ Wout_eff.T)      # (B, T, 2): [LR, MR]
        m = torch.zeros(B, T, 6, device=pdot.device, dtype=pdot.dtype)
        for row, muscle_idx in self.muscle_row.items():
            m[..., muscle_idx] = drive[..., row]
        u = self.eye(m)                                     # set in main(); (B,T,3) deg
        return (u, m, R) if want_states else (u, m)


# ---------------------------------------------------------------------------
# training -- identical to train_eyeG.main() except the controller
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eye-dir", default=TG.EYE_DIR)
    p.add_argument("--eye-fit", default=TG.EYE_FIT_PATH,
                   help="cached eye fit; loaded if present, else fitted and written")
    p.add_argument("--refit", action="store_true",
                   help="ignore --eye-fit even if present and refit from --eye-dir")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="circuit yaml with circuit.cell_types")
    p.add_argument("--pkl", default=DEFAULT_PKL,
                   help="the oculomotor connectome pickle")
    p.add_argument("--tag", default="zebraEyeG", help="checkpoint name")
    p.add_argument("--tau0", type=float, default=0.1, help="initial time constant, s")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--lam-psi", type=float, default=0.05,
                   help="torsion penalty; Donders' law in its simplest form")
    p.add_argument("--track-phi", action="store_true",
                   help="also supervise vertical gaze (phi). Off by default: "
                        "with only LR/MR reachable this pool cannot drive phi at "
                        "all, so phi just sits at whatever g(m) implies for "
                        "horizontal-only drives -- there is no gradient signal "
                        "that reduces a phi error, only one that fights the "
                        "theta objective for no benefit. Flip this on once "
                        "OMN (SR/IR/SO/IO) joins the pool and phi is actually "
                        "reachable; nothing else below needs to change.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    tag = a.tag
    dev = torch.device(a.device)
    os.makedirs(MODELS, exist_ok=True)

    # --- the eye (unchanged from train_eyeG.py) ---------------------------
    spec = TG.fit_eye(a.eye_dir, cache=a.eye_fit, refit=a.refit)
    eye = TG.EyeG(spec, a.dt).to(dev)
    reach, pos, neg = eye.reach_deg()
    print(f"[eye] {eye.n_act} drives: {', '.join(eye.act_names)}")
    print(f"[eye] reach  h {reach[0]:.1f}  v {reach[1]:.1f}  torsion {reach[2]:.1f} deg"
          f"   (+{pos[0]:.1f}/-{neg[0]:.1f} h, +{pos[1]:.1f}/-{neg[1]:.1f} v)")
    span_h, span_v = pos[0] + neg[0], pos[1] + neg[1]
    if span_h < TG.GATE_H or span_v < TG.GATE_V:
        print(f"[gate] WARNING: span {span_h:.1f} deg h / {span_v:.1f} deg v against "
              f"the {TG.GATE_H:.0f}/{TG.GATE_V:.0f} the task needs.")

    # --- the circuit --------------------------------------------------------
    circuit = load_oculomotor_circuit(a.config, a.pkl)
    n_e = int((circuit["sign_col"] > 0).sum())
    print(f"[circuit] {circuit['cfg'].circuit.name}: {len(circuit['names'])} cells, "
          f"{int(circuit['support'].sum())} edges, {n_e}E/{len(circuit['names']) - n_e}I, "
          f"{int(circuit['afferent'].sum())} afferent (AF5)")
    reachable = sorted(v for v in circuit["effector_col"].values() if v in (0, 2))
    print(f"[circuit] muscles reachable: {[MUSCLES[i] for i in reachable]}; "
          f"NOT reachable (held at zero): "
          f"{[m for i, m in enumerate(MUSCLES) if i not in reachable]}")

    # One tracked angle (theta) by default, matching what this pool can actually
    # drive; see --track-phi's help for why supervising phi buys nothing today.
    n_track = 2 if a.track_phi else 1
    print(f"[task] tracking {'(theta, phi)' if n_track == 2 else 'theta only'} "
          f"({n_track} of 2 angles); torsion always penalised toward zero")

    # --- data: the shared corpus, scaled to this eye's own reach ---------
    scale = np.array([reach[0], reach[1]], np.float32) / TG.BOUND
    print(f"[data] anisotropic world: h x{scale[0]:.2f}, v x{scale[1]:.2f} deg/unit "
          f"(+-{reach[0]:.1f} / +-{reach[1]:.1f} deg)")
    sp = {}
    for nm, n, s0 in (("train", 150, 0), ("val", 25, 5_000_000),
                      ("test", 40, 9_000_000)):
        pdot, star, _ = TG.learn.load_split(nm, n, a.duration, a.dt, s0)
        sp[nm] = (torch.as_tensor(pdot).to(dev),
                  torch.as_tensor(star * scale).to(dev))
    (Pdot_tr, Star_tr), (Pdot_va, Star_va), (Pdot_te, Star_te) = \
        sp["train"], sp["val"], sp["test"]

    torch.manual_seed(0)
    model = ZebrafishCircuitRNN(circuit, tau0=a.tau0, dt=a.dt).to(dev)
    model.eye = eye
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_out_cells = sum(idx.size for idx in circuit["output_idx"].values())
    print(f"[circuit] {n_params} trainable parameters "
          f"(S masked to {int(circuit['support'].sum())} synapses, "
          f"Win on {int(circuit['afferent'].sum())} afferent rows, "
          f"Wout on {n_out_cells} output cells)")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    T = Pdot_tr.shape[1]
    sched = [max(60, int(T * f)) for f in (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)]

    def evaluate(Pdot, Star):
        model.eval()
        with torch.no_grad():
            errs = []
            for i in range(0, Pdot.shape[0], a.batch):
                u, _ = model(Pdot[i:i + a.batch])
                errs.append((u[..., :n_track] - Star[i:i + a.batch, :, :n_track])
                            .norm(dim=-1))
            return float(torch.cat(errs).mean())

    def untracked_phi_drift(Pdot):
        """Diagnostic only, never in the loss: how far phi wanders when nothing
        is asking it to go anywhere. Visible so it isn't just silently ignored."""
        if n_track == 2:
            return 0.0
        model.eval()
        with torch.no_grad():
            u, _ = model(Pdot[:64])
            return float(u[..., 1].abs().mean())

    best, best_state = np.inf, None
    for ep in range(a.epochs):
        model.train()
        h = sched[min(3, int(4 * ep / max(a.epochs - 1, 1)))]
        perm = torch.randperm(Pdot_tr.shape[0], device=dev)
        tot = 0.0
        for i in range(0, len(perm), a.batch):
            j = perm[i:i + a.batch]
            u, _ = model(Pdot_tr[j, :h])
            # eq:loss / eq:loss-again, restricted to the n_track reachable angles
            loss = ((u[..., :n_track] - Star_tr[j, :h, :n_track]) ** 2).mean() \
                + a.lam_psi * (u[..., 2] ** 2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss.detach()) * len(j)
        sch.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            e = evaluate(Pdot_va, Star_va)
            print(f"  ep {ep:3d}  horizon {h:3d}  train {tot / len(perm):.5f}  "
                  f"val |err| {TG._err_color(e)} deg")
            if e < best:
                best, best_state = e, {k: v.detach().clone()
                                       for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(Pdot_te, Star_te)
    with torch.no_grad():
        u, m = model(Pdot_te[:64])
        psi = float(u[..., 2].abs().mean())
        occ = float((m > 1e-3).float().mean())
    phi_drift = untracked_phi_drift(Pdot_te)
    print(f"[done] test |err| {TG._err_color(test)} deg   mean |torsion| {psi:.2f} deg   "
          f"drives active {occ * 100:.0f}% of the time")
    if n_track == 1:
        print(f"[done] phi (untracked, not in the loss) drifts to "
              f"{phi_drift:.2f} deg mean |phi| -- expected: nothing asked it not to.")

    ck = os.path.join(MODELS, f"{tag}.pt")
    torch.save({"state": model.state_dict(),
                "eye": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in spec.items()},
                "circuit_config": a.config, "circuit_pkl": a.pkl,
                "dt": a.dt, "scale": scale.tolist(), "tau0": a.tau0}, ck)
    rep = os.path.join(MODELS, f"{tag}.json")
    json.dump({"tag": tag, "n_cells": len(circuit["names"]),
               "n_track": n_track, "track_phi": bool(a.track_phi),
               "gaze_err_mean_deg": test, "val_err_deg": best,
               "mean_abs_torsion_deg": psi,
               "untracked_phi_drift_deg": phi_drift if n_track == 1 else None,
               "deg_per_unit_h": float(scale[0]), "deg_per_unit_v": float(scale[1]),
               "span_h_deg": float(span_h), "span_v_deg": float(span_v),
               "epochs": a.epochs, "lam_psi": a.lam_psi,
               "muscles_reachable": [MUSCLES[i] for i in reachable]},
              open(rep, "w"), indent=2)
    print(f"[done] wrote {ck}\n       wrote {rep}")


if __name__ == "__main__":
    main()
