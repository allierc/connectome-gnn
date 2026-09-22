#!/usr/bin/env python
"""Build the measurement-noise (SNR) grid for Vzfg Q2.

One dataset per measurement-noise level: take the noiseless GCaMP6f calcium
trace, add i.i.d. Gaussian noise at gamma = frac * SD(calcium), deconvolve back
to voltage with lambda TUNED FOR THAT NOISE LEVEL, and write the estimate where
the trainer expects voltage. Process noise is fixed at sigma = 0.05 (it is baked
into the trajectory and is not an axis here).

Lambda must be retuned per level: the derivative-Tikhonov prior trades derivative
fidelity against noise amplification, so a lambda that is right at gamma = 0.03 is
catastrophic at gamma = 0 (it cost ~70% of the derivative amplitude, which is what
invalidated the first deconvolution arm).

Usage:
    python build_snr_grid.py            # build datasets + configs
    python build_snr_grid.py --dry-run  # print the plan only
"""
import argparse, os, shutil, sys
import numpy as np, zarr

sys.path.insert(0, '/workspace/connectome-gnn-ca/src')
from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.generators.gcamp_kernel import build_kernel_from_config
from connectome_gnn.models.calcium_deconvolution import wiener_deconvolve
from connectome_gnn.utils import graphs_data_path, set_data_root

ROOT = "/groups/saalfeld/home/allierc/GraphData"
SRC_CFG = os.environ.get('GRID_SRC_CFG',
    "/workspace/connectome-gnn-ca/config/fly/nr2_ca_calcium_unified.yaml")
BASE_CFG = {
    "unified":   "/workspace/connectome-gnn-ca/config/fly/nr2_ca_deconv_kernel_unified.yaml",
    "known_ode": "/workspace/connectome-gnn-ca/config/fly/nr2_ca_deconv_kernel_known_ode.yaml",
}
CONFIG_OUT = os.path.join(ROOT, "config", "fly")
SRC_DS = os.environ.get('GRID_SRC_DS', "fly/nr2_ca_blank50_kernel")
PREFIX = os.environ.get('GRID_PREFIX', "nr2_ca_snr")   # dataset/config name stem
CHUNK = 2000
SEED = 1234
# gamma as a fraction of SD(calcium). 0 is the arithmetic control (exactly
# invertible); 0.01-0.10 spans ~36 dB down to ~16 dB, which brackets real
# two-photon GCaMP.
FRACS = [0.0, 0.01, 0.03, 0.10]
LAM_GRID = [1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
WARMUP = 4          # frames 0..3 are kernel warm-up: incomplete convolution


def _open(p):
    return np.asarray(zarr.open_array(p, mode='r')[:], dtype=np.float32)


def tune_lambda(ca, vt, kernel, rng, n_probe=250):
    """Pick lambda by per-neuron Pearson on vdot, which is what W is read from."""
    L = len(kernel)
    idx = np.sort(rng.choice(ca.shape[1], min(n_probe, ca.shape[1]), replace=False))
    T = min(32000, ca.shape[0])
    c, v = ca[:T, idx], vt[:T, idx]
    m = slice(L, T - L)
    dv = np.diff(v[m], axis=0)
    best = (None, -9.0)
    for lam in LAM_GRID:
        est = wiener_deconvolve(c, kernel, lam=lam, regularizer='derivative')
        de = np.diff(est[m], axis=0)
        a = dv.ravel().astype(np.float64); b = de.ravel().astype(np.float64)
        a = a - a.mean(); b = b - b.mean()
        r = float((a @ b) / np.sqrt((a @ a) * (b @ b) + 1e-30))
        if r > best[1]:
            best = (lam, r)
    return best


def build(frac, dry):
    set_data_root(ROOT)
    cfg = NeuralGraphConfig.from_yaml(SRC_CFG)
    sim = cfg.simulation
    kernel = build_kernel_from_config(sim, device='cpu').numpy().astype(np.float32)
    dt = float(sim.delta_t)
    tag = f"g{int(round(frac*1000)):03d}"
    dst_name = f"fly/{PREFIX}_{tag}"
    src_dir = graphs_data_path(SRC_DS)
    dst_dir = graphs_data_path(dst_name)

    rng = np.random.default_rng(SEED)
    ca0 = _open(os.path.join(src_dir, 'x_list_train', 'calcium.zarr'))
    vt0 = _open(os.path.join(src_dir, 'x_list_train', 'voltage.zarr'))
    sd = float(ca0.std())
    gamma = frac * sd
    noisy = ca0 + rng.standard_normal(ca0.shape).astype(np.float32) * gamma if gamma > 0 else ca0
    lam, r_probe = tune_lambda(noisy, vt0, kernel, np.random.default_rng(SEED))
    snr = float('inf') if gamma == 0 else 20*np.log10(sd/gamma)
    print(f"[{tag}] frac={frac} gamma={gamma:.5f} SNR={snr:.1f} dB -> lambda={lam:.0e} (probe r(vdot)={r_probe:.4f})",
          flush=True)
    if dry:
        return dst_name, None

    os.makedirs(dst_dir, exist_ok=True)
    info = {}
    for split in ('train', 'test'):
        sx = os.path.join(src_dir, f'x_list_{split}')
        dx = os.path.join(dst_dir, f'x_list_{split}')
        os.makedirs(dx, exist_ok=True)
        ca = _open(os.path.join(sx, 'calcium.zarr'))
        truth = _open(os.path.join(sx, 'voltage.zarr'))
        r_ = np.random.default_rng(SEED + (0 if split == 'train' else 1))
        if gamma > 0:
            ca = ca + r_.standard_normal(ca.shape).astype(np.float32) * gamma
        # Reflect-pad one kernel length. The circular FFT wraps the end of the
        # record onto its start, which at small lambda puts a ~100x spike on the
        # final frame; padding moves that wrap into material we discard. Needed
        # in addition to the warm-up repair below -- padding fixes the tail, the
        # repair fixes the head, and omitting either cost r(vdot) 0.99 -> 0.24.
        L = len(kernel)
        padded = np.concatenate([ca[1:L + 1][::-1], ca, ca[-L - 1:-1][::-1]])
        est_pad = np.empty_like(padded)
        for a in range(0, ca.shape[1], CHUNK):
            b = min(a + CHUNK, ca.shape[1])
            est_pad[:, a:b] = wiener_deconvolve(padded[:, a:b], kernel, lam=lam,
                                                regularizer='derivative')
        est = est_pad[L:L + ca.shape[0]].copy()
        # Frames 0..WARMUP-1 are kernel warm-up: the calcium there is an
        # incomplete convolution, so v is unrecoverable. Hold the first valid one.
        est[:WARMUP] = est[WARMUP]
        zarr.save_array(os.path.join(dx, 'voltage.zarr'), est.astype(np.float32))
        for nm in ('stimulus', 'neuron_type', 'group_type', 'pos', 'noise'):
            s = os.path.join(sx, f'{nm}.zarr')
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(dx, f'{nm}.zarr'), dirs_exist_ok=True)
        y = np.zeros_like(est)
        y[:-1] = (est[1:] - est[:-1]) / dt
        y[-1] = y[-2]
        zarr.save_array(os.path.join(dst_dir, f'y_list_{split}.zarr'),
                        y.astype(np.float32)[:, :, None])

        def pr(x, z):
            x = x.ravel().astype(np.float64); z = z.ravel().astype(np.float64)
            x = x - x.mean(); z = z - z.mean()
            return float((x @ z) / np.sqrt((x @ x) * (z @ z) + 1e-30))
        info[split] = (pr(truth, est), pr(np.diff(truth, axis=0), np.diff(est, axis=0)))
        print(f"  [{tag}] {split}: r(v)={info[split][0]:.4f} r(vdot)={info[split][1]:.4f}", flush=True)

    for f in ('ode_params.pt', 'generation_log.txt'):
        s = os.path.join(src_dir, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst_dir, f))
    with open(os.path.join(dst_dir, '.generate_done'), 'w') as fh:
        fh.write(f"snr grid: frac={frac} gamma={gamma:.6f} SNR={snr:.2f}dB lambda={lam}\n")
        for k, v in info.items():
            fh.write(f"{k}: r(v)={v[0]:.4f} r(vdot)={v[1]:.4f}\n")

    cfg_names = []
    os.makedirs(CONFIG_OUT, exist_ok=True)
    for arm, base_path in BASE_CFG.items():
        label = 'GNN' if arm == 'unified' else 'Known-ODE'
        out = []
        for ln in open(base_path).read().splitlines():
            if ln.startswith('description:'):
                out.append(f'description: "Vzfg Q2 SNR grid: deconvolved calcium, '
                           f'gamma={frac:g}xSD ({snr:.0f} dB), lambda={lam:g}, {label}"')
            elif ln.startswith('dataset:'):
                out.append(f'dataset: {dst_name}')
            else:
                out.append(ln)
        cfg_name = f"{PREFIX}_{tag}_{arm}"
        with open(os.path.join(CONFIG_OUT, cfg_name + '.yaml'), 'w') as fh:
            fh.write('\n'.join(out) + '\n')
        print(f"  [{tag}] wrote config {cfg_name}.yaml", flush=True)
        cfg_names.append(cfg_name)
    with open(os.path.join(dst_dir, '.grid_meta'), 'w') as fh:
        fh.write(f"frac={frac}\ngamma={gamma:.6f}\nsnr_db={snr}\nlambda={lam}\n")
        for k, v in info.items():
            fh.write(f"{k}_r_v={v[0]:.4f}\n{k}_r_vdot={v[1]:.4f}\n")
    return dst_name, cfg_names


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    names = [build(f, a.dry_run) for f in FRACS]
    flat=[c for n in names if n[1] for c in n[1]]
    print("\nconfigs:", flat)
