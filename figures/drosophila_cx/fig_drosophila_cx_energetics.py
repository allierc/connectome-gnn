"""Energetics of the heading code: a von-Neumann three-tier ladder.

For a trained head-direction run (Drosophila central complex by default,
or larval zebrafish via --species) we already measure the *information*
the recurrent pool carries about heading (and translation). This script
adds the *energy* side and places the circuit on the ladder von Neumann
drew in his Silliman lectures (Yale 1956): the thermodynamic minimum, the
metabolic cost of the neurons, and the cost of the silicon model that
emulates them.

Three energies, one information (I_pop, in bits, the decoder lower bound
of the whole recurrent pool about heading):

  Tier 1  Landauer floor   E_min  = I_pop * kT ln2     per heading update
                           per-bit floor = kT ln2  (independent of I_pop)
  Tier 2  Biological        P_bio  = N_rec * <rate> * E_spike
  Tier 3  Silicon model     P_si   = FLOP/s * J_per_FLOP   (the RNN itself)
                           J_per_FLOP amortized from the inference GPU:
                           gpu_watts / gpu_tflops_eff (A100, batched)

and the gap factors between tiers (von Neumann's 10^11 and 2x10^5,
re-derived from a connectome-constrained model that actually runs).

The leak makes the erasure rate explicit: a leaky integrator with state
time-constant tau overwrites its estimate at rate 1/tau, so the mandatory
dissipation is (bits erased / s) * kT ln2 = (I_pop / tau) * kT ln2.

We also quantify the "reliability tax" von Neumann *guessed* was behind
the gap: the recurrent pool spends N_rec neurons to hold I_pop bits that
N_eff = I_pop / <I_single> neurons would suffice for if non-redundant.
The redundancy factor R = N_rec * <I_single> / I_pop is read straight off
the per-neuron and pooled MI we already compute.

Usage:
    python figures/drosophila_cx/fig_drosophila_cx_energetics.py \
        --species drosophila_cx --run drosophila_cx_rotation_distance --device cpu
    # larval zebrafish companion:
    python figures/drosophila_cx/fig_drosophila_cx_energetics.py \
        --species zebrafish \
        --run zebrafish_hd_si_ipn_917_v1_selfmotion_both --device cpu
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

import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.abspath(os.path.join(HERE, ".."))   # the figures/ root
sys.path.insert(0, os.path.join(FIGROOT, "..", "src"))
sys.path.insert(0, os.path.join(FIGROOT, "zebrafish"))  # _despine lives here

from connectome_gnn.utils import (  # noqa: E402
    load_data_root_from_json, set_data_root, graphs_data_path,
)
from connectome_gnn.zarr_io import load_raw_array  # noqa: E402

# Per-species MI-partition module: heading bits, recurrent-pool mask, rollout
# accumulator and the decoder lower bound all come from the SAME estimators the
# paper's MI figures use, so the bits here are the same currency. Paths are
# anchored to the figures/ root so this script runs from any location.
_MI_MODULES = {
    "zebrafish": os.path.join(FIGROOT, "zebrafish", "fig_zebrafish_mi_partition.py"),
    "drosophila_cx": os.path.join(
        FIGROOT, "drosophila_cx", "fig_drosophila_cx_mi_partition.py"),
}


def _load_mi_helpers(species):
    """Import the species-specific MI-partition module and return its helpers.
    Both modules expose an identical helper interface by construction."""
    path = os.path.abspath(_MI_MODULES[species])
    spec = importlib.util.spec_from_file_location(f"_mi_{species}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = ("_load_run", "_accumulate_hidden", "_circular_mi",
             "_group_mi_bits", "_bin_theta_dec", "_disp_decode_label",
             "_is_recurrent", "_PROFILE_BY_TARGET", "_RECOGNISED")
    return {n: getattr(mod, n) for n in names}

# ----------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------
K_B = 1.380649e-23          # Boltzmann constant [J/K]
T_BODY = 301.0              # larval zebrafish rearing temp ~28 C [K]
KT_LN2 = K_B * T_BODY * np.log(2.0)          # Landauer cost per bit [J]
E_ATP = 8.3e-20             # in-vivo ATP hydrolysis, ~50 kJ/mol [J]


def _population_mi(helpers, run_basename, data_root, n_trials, device, log_subdir):
    """Roll out the trained run and return everything the energetics need:
    per-neuron heading MI, pooled heading & translation MI (decoder lower
    bound), the recurrent firing rates, and a size-resolved pooling curve.
    ``helpers`` is the dict of species-specific MI helpers."""
    _load_run = helpers["_load_run"]
    _accumulate_hidden = helpers["_accumulate_hidden"]
    _circular_mi = helpers["_circular_mi"]
    _group_mi_bits = helpers["_group_mi_bits"]
    _bin_theta_dec = helpers["_bin_theta_dec"]
    _disp_decode_label = helpers["_disp_decode_label"]
    _is_recurrent = helpers["_is_recurrent"]
    _PROFILE_BY_TARGET = helpers["_PROFILE_BY_TARGET"]
    _RECOGNISED = helpers["_RECOGNISED"]

    run_dir = os.path.join(data_root, "log", log_subdir, run_basename)
    net, config = _load_run(run_dir, device)

    root = graphs_data_path(config.dataset)
    u_test = load_raw_array(f"{root}/test/stimulus.zarr")
    y_test = load_raw_array(f"{root}/test/target.zarr")
    task_raw = list(getattr(config.training, "task_targets", None) or [])
    task_key = tuple(t for t in _RECOGNISED if t in task_raw)
    if u_test.shape[-1] >= 4 and task_key:
        key = (int(y_test.shape[-1]), task_key)
        if key in _PROFILE_BY_TARGET:
            in_cols, out_cols = _PROFILE_BY_TARGET[key]
            u_test = u_test[..., in_cols]
            y_test = y_test[..., out_cols]

    H, theta, d, pos2d = _accumulate_hidden(net, u_test, y_test, device, n_trials)

    # recurrent pool mask (the GABAergic substrate; afferents excluded)
    type_names = list(net.type_names)
    nt = np.asarray(net.neuron_types).astype(int)
    is_rec = np.array([_is_recurrent(type_names[int(t)]) for t in nt])
    rec_ix = np.where(is_rec)[0]
    Hrec = H[:, rec_ix]                              # (T, N_rec)
    N_rec = rec_ix.size

    # firing rates: the model's own readout nonlinearity on the hidden state
    with torch.no_grad():
        rates = net._sigma(torch.from_numpy(Hrec)).cpu().numpy()  # (T, N_rec) in (0,1)
    mean_norm_rate = float(np.mean(rates))           # mean normalised rate

    # per-neuron heading MI (plug-in, same estimator as the paper)
    I_single = np.array([_circular_mi(Hrec[:, k], theta) for k in range(N_rec)])

    # pooled decoder lower bound about heading & translation (whole pool)
    rng = np.random.default_rng(0)
    Tn = Hrec.shape[0]
    MAXS = 9000
    sel = rng.choice(Tn, MAXS, replace=False) if Tn > MAXS else np.arange(Tn)
    Xs = Hrec[sel]
    yb_theta = _bin_theta_dec(theta)[sel]
    yb_disp = _disp_decode_label(d, pos2d)[sel]
    I_pop_theta = _group_mi_bits(Xs, yb_theta)
    I_pop_disp = _group_mi_bits(Xs, yb_disp)

    # size-resolved pooling curve I(N): the redundancy / reliability story
    sizes = [s for s in (1, 2, 5, 10, 20, 50, 100, 200, 400, N_rec)
             if s <= N_rec]
    if sizes[-1] != N_rec:
        sizes.append(N_rec)
    pool_curve = []
    for s in sizes:
        reps = 1 if s >= N_rec else 5          # average a few random subsets
        vals = []
        for r in range(reps):
            cols = (np.arange(N_rec) if s >= N_rec
                    else rng.choice(N_rec, s, replace=False))
            vals.append(_group_mi_bits(Xs[:, cols], yb_theta))
        pool_curve.append((s, float(np.mean(vals))))

    return dict(
        N_rec=N_rec, mean_norm_rate=mean_norm_rate,
        I_single=I_single, I_pop_theta=I_pop_theta, I_pop_disp=I_pop_disp,
        pool_curve=pool_curve,
        tau=float(net.tau), dt=float(net.dt),
    )


def _measure_gpu_watts(n_rec, dt, gpu_batch, seconds, device, sigma=torch.sigmoid):
    """Sample ``nvidia-smi`` power.draw around a batched rollout of the dense
    recurrent operator and return net active board power *per circuit* [W].

    The headline silicon cost is amortized: ``gpu_batch`` independent circuits
    run concurrently to saturate the board, so the per-circuit power is
    (run - idle) / gpu_batch. The measured op is the dominant N x N recurrent
    matmul (the leak + nonlinearity the paper's FLOP count also approximates).
    Returns (per_circuit_W, board_run_W, board_idle_W, n_eff_batch) or None if
    CUDA / power telemetry is unavailable."""
    import subprocess
    import threading
    import time

    if device.type != "cuda" or not torch.cuda.is_available():
        print("[measure_gpu] no CUDA device; skipping measurement")
        return None

    def _draw():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2)
            return float(out.stdout.strip().splitlines()[0])
        except Exception:
            return None

    if _draw() is None:
        print("[measure_gpu] power.draw unavailable on this GPU; skipping")
        return None

    def _sample(stop, bucket):
        while not stop.is_set():
            w = _draw()
            if w is not None:
                bucket.append(w)
            time.sleep(0.05)

    # idle baseline (no GPU work)
    torch.cuda.synchronize()
    idle = []
    stop = threading.Event()
    th = threading.Thread(target=_sample, args=(stop, idle)); th.start()
    time.sleep(seconds)
    stop.set(); th.join()

    # busy loop: B circuits x (N x N) recurrent step, paced as fast as the device
    # will go (the amortized ceiling); state carried across steps like a rollout.
    W = torch.randn(n_rec, n_rec, device=device) / np.sqrt(n_rec)
    x = torch.randn(gpu_batch, n_rec, device=device)
    run = []
    stop = threading.Event()
    th = threading.Thread(target=_sample, args=(stop, run)); th.start()
    t_end = time.time() + seconds
    with torch.no_grad():
        while time.time() < t_end:
            for _ in range(50):
                x = sigma(x @ W)
            torch.cuda.synchronize()
    stop.set(); th.join()

    if len(run) < 3 or len(idle) < 3:
        print("[measure_gpu] too few power samples; skipping")
        return None
    board_run = float(np.median(run))
    board_idle = float(np.median(idle))
    per_circuit = max(board_run - board_idle, 0.0) / gpu_batch
    print(f"[measure_gpu] board run={board_run:.1f} W idle={board_idle:.1f} W "
          f"batch={gpu_batch} -> per-circuit {per_circuit:.3e} W")
    return per_circuit, board_run, board_idle, gpu_batch


def _energetics(m, r_max_hz, atp_per_spike, j_per_flop, gpu_tflops_eff,
                silicon_mode="flop", p_si_watts=None):
    """Assemble the three-tier ladder from the measured (info, rate, N)."""
    tau = m["tau"]
    I_pop = m["I_pop_theta"]                          # bits of heading
    N_rec = m["N_rec"]

    # --- Tier 1: Landauer floor (erasure rate set by the state leak 1/tau) ---
    erase_rate = I_pop / tau                          # bits erased per second
    P_landauer = erase_rate * KT_LN2                  # [W]
    E_landauer_update = I_pop * KT_LN2                # per state refresh (tau) [J]
    eta_landauer = KT_LN2                             # [J/bit] (floor)

    # --- Tier 2: biological metabolic cost --------------------------------
    rate_hz = r_max_hz * m["mean_norm_rate"]          # spikes/s/neuron
    E_spike = atp_per_spike * E_ATP                   # [J/spike]
    P_bio = N_rec * rate_hz * E_spike                 # [W]
    E_bio_update = P_bio * tau                         # per refresh [J]
    eta_bio = E_bio_update / I_pop                     # [J/bit]

    # --- Tier 3: silicon model (the RNN forward pass) ---------------------
    # Reported as a range, most-honest to most-idealized, all per bit:
    #   flop : 2 N^2 / dt * J_per_FLOP  -- accelerator lower bound (useful FLOPs
    #          only; j_per_flop is gpu_watts / sustained TFLOP/s, A100-amortized)
    #   meas : measured net board/wall power per circuit (run - idle, amortized)
    # The dense recurrent operator W_rec (N x N) dominates at ~2 N^2 FLOP/step.
    flop_per_step = 2.0 * N_rec ** 2
    steps_per_s = 1.0 / m["dt"]
    flop_demand = flop_per_step * steps_per_s          # useful FLOP/s, one circuit
    P_si_flop = flop_demand * j_per_flop               # [W] accelerator lower bound
    eta_si_flop = (P_si_flop * tau) / I_pop            # [J/bit]
    # fraction of the GPU a single circuit fills: the amortized FLOP power is
    # only realisable when ~1/util_single such circuits are batched to saturate
    # it; an unbatched run pays the full device TDP for these few FLOPs.
    util_single = flop_demand / (gpu_tflops_eff * 1e12)
    n_parallel = 1.0 / util_single if util_single > 0 else float("nan")

    # measured estimate (board or wall, per circuit), if supplied / sampled
    P_si_meas = p_si_watts
    eta_si_meas = (P_si_meas * tau) / I_pop if P_si_meas is not None else None

    # headline silicon power: chosen mode, falling back to the FLOP bound
    if silicon_mode != "flop" and P_si_meas is not None:
        P_si, eta_si = P_si_meas, eta_si_meas
    else:
        P_si, eta_si = P_si_flop, eta_si_flop
    E_si_update = P_si * tau                            # per refresh [J]

    # --- reliability tax (redundancy of the population code) --------------
    I_single_mean = float(np.mean(m["I_single"]))
    N_eff = I_pop / I_single_mean if I_single_mean > 0 else float("nan")
    redundancy = N_rec * I_single_mean / I_pop if I_pop > 0 else float("nan")

    return dict(
        I_pop=I_pop, I_pop_disp=m["I_pop_disp"], N_rec=N_rec, tau=tau,
        rate_hz=rate_hz,
        P_landauer=P_landauer, P_bio=P_bio, P_si=P_si,
        E_landauer_update=E_landauer_update,
        E_bio_update=E_bio_update, E_si_update=E_si_update,
        eta_landauer=eta_landauer, eta_bio=eta_bio, eta_si=eta_si,
        gap_bio_landauer=eta_bio / eta_landauer,
        gap_si_landauer=eta_si / eta_landauer,
        gap_si_bio=eta_si / eta_bio,
        silicon_mode=silicon_mode,
        eta_si_flop=eta_si_flop, eta_si_meas=eta_si_meas,
        P_si_flop=P_si_flop, P_si_meas=P_si_meas,
        j_per_flop=j_per_flop, util_single=util_single, n_parallel=n_parallel,
        I_single_mean=I_single_mean, N_eff=N_eff, redundancy=redundancy,
    )


def _eng(x):
    """1.4x10^-16 style string."""
    if x == 0 or not np.isfinite(x):
        return f"{x:g}"
    e = int(np.floor(np.log10(abs(x))))
    mant = x / 10 ** e
    return f"{mant:.2f}e{e:+d}"


def _floor_pow10_tex(x):
    """Floor a multiplicative factor to its order of magnitude as mathtext,
    e.g. 2.33e11 -> '$\\times 10^{11}$' (matches von Neumann's '~10^11')."""
    if x <= 0 or not np.isfinite(x):
        return ""
    return f"$\\times 10^{{{int(np.floor(np.log10(x)))}}}$"


def _report(e, args):
    L = []
    L.append("=" * 66)
    L.append(f"  ENERGETICS OF THE HEADING CODE  ({args.run})")
    L.append("=" * 66)
    L.append(f"  recurrent pool      N_rec = {e['N_rec']}")
    L.append(f"  state time const    tau   = {e['tau']*1e3:.0f} ms")
    L.append(f"  heading info        I_pop = {e['I_pop']:.2f} bits  "
             f"(translation {e['I_pop_disp']:.2f} bits)")
    L.append(f"  mean firing rate          = {e['rate_hz']:.2f} Hz "
             f"(r_max={args.r_max_hz:g} Hz)")
    L.append("-" * 66)
    L.append("  TIER                power [W]      energy/update [J]   J / bit")
    L.append(f"  1 Landauer floor    {_eng(e['P_landauer'])}     "
             f"{_eng(e['E_landauer_update'])}        {_eng(e['eta_landauer'])}")
    L.append(f"  2 biological        {_eng(e['P_bio'])}     "
             f"{_eng(e['E_bio_update'])}        {_eng(e['eta_bio'])}")
    L.append(f"  3 silicon model     {_eng(e['P_si'])}     "
             f"{_eng(e['E_si_update'])}        {_eng(e['eta_si'])}")
    L.append("-" * 66)
    L.append(f"  gap  biology / Landauer floor   = {_eng(e['gap_bio_landauer'])}"
             "   (von Neumann: ~1e11)")
    L.append(f"  gap  silicon / biology          = {_eng(e['gap_si_bio'])}"
             "   (von Neumann: ~2e5)")
    L.append(f"  gap  silicon / Landauer floor   = {_eng(e['gap_si_landauer'])}")
    L.append("-" * 66)
    L.append(f"  SILICON ESTIMATE RANGE (headline mode: {e['silicon_mode']})")
    L.append(f"  (i)  FLOP lower bound       eta_si = {_eng(e['eta_si_flop'])} J/bit  "
             f"(A100 {args.gpu_watts:.0f} W / {args.gpu_tflops_eff:g} TFLOP/s "
             f"-> {_eng(e['j_per_flop'])} J/FLOP)")
    if e["eta_si_meas"] is not None:
        L.append(f"  (ii) measured ({e['silicon_mode']:>11}) eta_si = "
                 f"{_eng(e['eta_si_meas'])} J/bit  "
                 f"(P_si = {_eng(e['P_si_meas'])} W per circuit)")
    else:
        L.append("  (ii) measured GPU/wall      not supplied "
                 "(pass --p_si_watts or --measure_gpu)")
    L.append(f"  single-circuit GPU util     = {_eng(e['util_single'])}  "
             f"(~{_eng(e['n_parallel'])} circuits to fill the card)")
    L.append("-" * 66)
    L.append("  RELIABILITY TAX (von Neumann's conjecture, measured)")
    L.append(f"  mean per-neuron heading MI      = {e['I_single_mean']:.3f} bits")
    L.append(f"  non-redundant neurons needed    N_eff = {e['N_eff']:.1f}")
    L.append(f"  redundancy factor  N_rec*I1/Ipop = {e['redundancy']:.1f}x")
    L.append("=" * 66)
    return "\n".join(L)


def _figure(m, e, out_png):
    LF, TF, LET = 13, 11, 15
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # (a) energy-per-bit ladder ------------------------------------------
    ax = axes[0]
    tiers = ["Landauer\nfloor", "biology", "silicon\nmodel"]
    etas = [e["eta_landauer"], e["eta_bio"], e["eta_si"]]
    cols = ["#777777", "#d29922", "#6a51a3"]
    xs = np.arange(3)
    ax.bar(xs, etas, color=cols, width=0.62, edgecolor="none", log=True)
    ax.axhline(e["eta_landauer"], color="#777777", ls="--", lw=0.9, zorder=0)
    # silicon range: FLOP lower bound -> measured device/wall (when available)
    if e.get("eta_si_meas") is not None:
        lo, hi = sorted((e["eta_si_flop"], e["eta_si_meas"]))
        ax.plot([2, 2], [lo, hi], color="k", lw=1.3, zorder=7)
        for yv in (lo, hi):
            ax.plot([1.86, 2.14], [yv, yv], color="k", lw=1.3, zorder=7)
    for x, v in zip(xs, etas):
        ax.text(x, v * 1.6, _eng(v), ha="center", va="bottom",
                fontsize=9.5)
    ax.set_xticks(xs); ax.set_xticklabels(tiers, fontsize=TF)
    ax.set_ylabel("energy per bit of heading  [J / bit]", fontsize=LF)
    ymin = e["eta_landauer"] / 5
    ymax = e["eta_si"] * 60
    ax.set_ylim(ymin, ymax)
    ax.tick_params(labelsize=TF)
    ax.text(-0.02, 1.04, "a", transform=ax.transAxes, fontsize=LET,
            fontweight="bold")
    # gap annotations between adjacent bars (geometric midpoints)
    _bbox = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
    ax.text(0.5, np.sqrt(e["eta_landauer"] * e["eta_bio"]),
            _floor_pow10_tex(e["gap_bio_landauer"]), ha="center",
            va="center", fontsize=8.5, color="0.35", bbox=_bbox, zorder=6)
    ax.text(1.5, np.sqrt(e["eta_bio"] * e["eta_si"]),
            _floor_pow10_tex(e["gap_si_bio"]), ha="center",
            va="center", fontsize=8.5, color="0.35", bbox=_bbox, zorder=6)

    # (b) pooling curve: redundancy / reliability tax --------------------
    ax = axes[1]
    sizes = np.array([s for s, _ in m["pool_curve"]], dtype=float)
    vals = np.array([v for _, v in m["pool_curve"]], dtype=float)
    ax.plot(sizes, vals, "o-", color="#0072b2", ms=5, lw=1.6,
            label="pooled  $I(\\theta)$")
    # the non-redundant expectation: N * mean single-cell MI, capped at ceiling
    ax.plot(sizes, np.minimum(sizes * e["I_single_mean"], e["I_pop"] * 1.05),
            "--", color="0.6", lw=1.2, label="non-redundant  $N\\,\\bar I_1$")
    ax.axhline(e["I_pop"], color="k", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("neurons pooled  $N$", fontsize=LF)
    ax.set_ylabel("heading information  [bits]", fontsize=LF)
    ax.set_ylim(0, max(e["I_pop"] * 1.25, 1.0))
    ax.tick_params(labelsize=TF)
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    ax.text(-0.02, 1.04, "b", transform=ax.transAxes, fontsize=LET,
            fontweight="bold")
    ax.text(0.04, 0.92,
            f"redundancy  {e['redundancy']:.0f}$\\times$\n"
            f"$N_{{\\rm eff}}\\approx{e['N_eff']:.0f}$ of {e['N_rec']}",
            transform=ax.transAxes, fontsize=10, va="top")

    plt.tight_layout()
    try:
        from _despine import open_axes
        open_axes(fig)
    except Exception:
        pass
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--species", default="drosophila_cx",
                   choices=sorted(_MI_MODULES),
                   help="which MI-partition helpers / log subdir to use")
    p.add_argument("--run", default="drosophila_cx_rotation_distance")
    p.add_argument("--data_root",
                   default="/groups/saalfeld/home/allierc/GraphData")
    p.add_argument("--n_trials", type=int, default=16)
    p.add_argument("--device", default="cpu")
    # biophysical / hardware constants (cited defaults, all overridable)
    p.add_argument("--r_max_hz", type=float, default=30.0,
                   help="peak firing rate that sigmoid=1 maps to [Hz]")
    p.add_argument("--atp_per_spike", type=float, default=1e8,
                   help="ATP molecules hydrolysed per spike (small vertebrate neuron)")
    # Silicon tier reported as a range, most-honest to most-idealized:
    #   flop          : 2 N^2 / dt * J_per_FLOP        (accelerator lower bound)
    #   measured_gpu  : P_si = (run - idle) board power, amortized per circuit
    #   wall          : P_si = (run - idle) wall-plug power, amortized per circuit
    # --silicon_mode picks which one drives the headline gaps and the figure.
    p.add_argument("--silicon_mode", default="flop",
                   choices=("flop", "measured_gpu", "wall"),
                   help="which silicon estimate is the headline (default: flop)")
    # FLOP lower-bound inputs: amortized energy/FLOP = gpu_watts / gpu_tflops_eff
    p.add_argument("--gpu_watts", type=float, default=400.0,
                   help="inference GPU board power [W] (A100 SXM4 ~400, PCIe ~250)")
    p.add_argument("--gpu_tflops_eff", type=float, default=10.0,
                   help="throughput the GPU sustains when batched [TFLOP/s] "
                        "(A100 FP32 peak 19.5, ~50%% realised on small matmuls)")
    p.add_argument("--j_per_flop", type=float, default=None,
                   help="override amortized energy per FLOP [J]; default is "
                        "gpu_watts / gpu_tflops_eff")
    # Measured-power inputs: net active power (run minus idle), amortized to ONE
    # circuit. Either supply it directly, or let --measure_gpu sample nvidia-smi
    # around a batched rollout of the recurrent operator on a GPU node.
    p.add_argument("--p_si_watts", type=float, default=None,
                   help="measured net silicon power per circuit [W] "
                        "(run minus idle, amortized over the batch)")
    p.add_argument("--measure_gpu", action="store_true",
                   help="sample nvidia-smi power.draw around a batched recurrent "
                        "rollout to fill in --p_si_watts (needs a CUDA device)")
    p.add_argument("--gpu_batch", type=int, default=4096,
                   help="circuits run concurrently when measuring, to amortize "
                        "the board over a saturating workload")
    p.add_argument("--measure_seconds", type=float, default=4.0,
                   help="busy-loop / idle sampling window for --measure_gpu [s]")
    p.add_argument("--out", default=None,
                   help="output PNG (defaults next to the species MI figure)")
    args = p.parse_args()

    if args.out is None:
        outdir = os.path.dirname(os.path.abspath(_MI_MODULES[args.species]))
        args.out = os.path.join(outdir, f"fig_{args.species}_energetics.png")

    try:
        set_data_root(load_data_root_from_json())
    except FileNotFoundError:
        pass

    helpers = _load_mi_helpers(args.species)
    device = torch.device(args.device)
    m = _population_mi(helpers, args.run, args.data_root, args.n_trials, device,
                       log_subdir=args.species)
    # amortized device energy/FLOP: GPU wall power / sustained throughput
    j_per_flop = (args.j_per_flop if args.j_per_flop is not None
                  else args.gpu_watts / (args.gpu_tflops_eff * 1e12))

    # measured silicon power (per circuit): explicit flag wins; else sample the
    # GPU if asked. Mode falls back to the FLOP bound if no measurement exists.
    p_si_watts = args.p_si_watts
    if p_si_watts is None and args.measure_gpu:
        meas = _measure_gpu_watts(m["N_rec"], m["dt"], args.gpu_batch,
                                  args.measure_seconds, device)
        if meas is not None:
            p_si_watts = meas[0]
    if args.silicon_mode != "flop" and p_si_watts is None:
        print(f"[fig_energetics] --silicon_mode {args.silicon_mode} requested but "
              "no measured power; reporting FLOP lower bound as headline.")

    e = _energetics(m, args.r_max_hz, args.atp_per_spike, j_per_flop,
                    args.gpu_tflops_eff, silicon_mode=args.silicon_mode,
                    p_si_watts=p_si_watts)

    print(_report(e, args))
    _figure(m, e, args.out)
    print(f"\n[fig_energetics] wrote {args.out}")

    # persist the numbers for the paper / reproducibility
    dump = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in e.items()}
    dump["constants"] = dict(K_B=K_B, T_BODY=T_BODY, KT_LN2=KT_LN2,
                             E_ATP=E_ATP, r_max_hz=args.r_max_hz,
                             atp_per_spike=args.atp_per_spike,
                             gpu_watts=args.gpu_watts,
                             gpu_tflops_eff=args.gpu_tflops_eff,
                             j_per_flop=j_per_flop)
    dump["pool_curve"] = m["pool_curve"]
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(dump, f, indent=2)
    print(f"[fig_energetics] wrote {os.path.splitext(args.out)[0]}.json")


if __name__ == "__main__":
    main()
