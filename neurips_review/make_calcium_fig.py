#!/usr/bin/env python
"""Figure for calcium_note: observable chain + rollout, best case (no meas. noise).

A  ground-truth voltage v                     (green)
B  GCaMP6f calcium C = K * v, same y-scale    (red)
C  deconvolved voltage over truth             (black on green)
D  derivative dv/dt, true vs deconvolved      (zoomed: it is dense at dt=20 ms)
E  rollout, Known-ODE fitted on the deconvolved observable

Colour convention follows _plot_three_panel in calcium_deconvolution.py:
green = ground truth, black = model/recovered, red = a distinct-source signal.
"""
import argparse, os, sys
import numpy as np, zarr
import matplotlib
matplotlib.use('Agg')
RC = '/workspace/connectome-gnn/figures/janne.matplotlibrc'
if os.path.isfile(RC):
    matplotlib.rc_file(RC)
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, '/workspace/connectome-gnn-ca/src')
from connectome_gnn.generators.gcamp_kernel import select_reference_neurons

BASE = "/groups/saalfeld/home/allierc/GraphData/graphs_data/fly"
LOG = "/groups/saalfeld/home/allierc/GraphData/log/fly"
ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='g000', help='SNR grid tag: g000 g010 g030 g100')
ARGS = ap.parse_args()
TAG = ARGS.tag
FRAC = {'g000': 0.0, 'g010': 0.01, 'g030': 0.03, 'g100': 0.10}[TAG]
SNR_DB = {'g000': 'no measurement noise', 'g010': '40 dB', 'g030': '30 dB', 'g100': '20 dB'}[TAG]
GRID_SEED = 1234                 # must match SEED in build_snr_grid.py
TRUTH, DECONV = "nr2_ca_blank50_kernel", f"nr2_ca_snr_{TAG}"
DT = 0.02
SPLIT = 'test'                  # everything on one split so panels are comparable
START, END = 500, 1500          # 10-30 s, past the kernel warm-up
ZOOM = 100                      # frames for the derivative panel (2 s)
C_GT, C_HAT, C_CA = '#2ca02c', 'black', '#d62728'
FS_L, FS_T = 9, 7
SHRINK = 0.65
N_CELLS = 8


def load(ds, name, split=SPLIT):
    return np.asarray(zarr.open_array(f"{BASE}/{ds}/x_list_{split}/{name}.zarr",
                                      mode='r')[START:END], dtype=np.float32)


nt = torch.tensor(np.asarray(zarr.open_array(
    f"{BASE}/{TRUTH}/x_list_{SPLIT}/neuron_type.zarr", mode='r')[:], dtype=np.int64))
idx, labels = select_reference_neurons(nt)
idx, labels = idx[:N_CELLS], labels[:N_CELLS]

v = load(TRUTH, 'voltage')[:, idx]
ca_clean_full = np.asarray(zarr.open_array(
    f"{BASE}/{TRUTH}/x_list_{SPLIT}/calcium.zarr", mode='r')[:], dtype=np.float32)
gamma = FRAC * float(ca_clean_full.std())
if gamma > 0:
    # reproduce exactly the noise the grid builder added for this split
    rng_ = np.random.default_rng(GRID_SEED + (0 if SPLIT == 'train' else 1))
    ca_full = ca_clean_full + rng_.standard_normal(
        ca_clean_full.shape).astype(np.float32) * gamma
else:
    ca_full = ca_clean_full
ca = ca_full[START:END][:, idx]
vh = load(DECONV, 'voltage')[:, idx]
dv = np.diff(v, axis=0) / DT
dvh = np.diff(vh, axis=0) / DT

rb = np.load(f"{LOG}/nr2_ca_snr_{TAG}_known_ode/results/rollout_bundle.npz",
             allow_pickle=True)
# NB activity_true is the observable the model was fitted on -- verified to be
# the deconvolved trace at r = 1.000000, NOT the ground-truth voltage.
robs = rb['activity_true'][idx, START:END].T
rp = rb['activity_pred'][idx, START:END].T

t = (np.arange(END - START) + START) * DT
tz = t[:ZOOM]


def draw(ax, series, letter, time, step, legend=True):
    n = series[0][0].shape[1]
    for i in range(n):
        y0 = (n - 1 - i) * step
        for data, col, lw, lab in series:
            ax.plot(time, SHRINK * (data[:len(time), i] - data[:, i].mean()) + y0,
                    lw=lw, color=col, alpha=0.95, label=lab if i == 0 else None)
        ax.text(time[0] - (time[-1] - time[0]) * 0.015, y0, labels[i],
                fontsize=FS_T, va='center', ha='right')
    ax.text(-0.055, 1.0, letter, transform=ax.transAxes, va='bottom',
            ha='left', fontsize=FS_L + 3, fontweight='bold')
    ax.set_ylim([-step, (n - 1) * step + 1.7 * step])
    ax.set_yticks([])
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    if legend and len(series) > 1:
        ax.legend(loc='upper right', fontsize=FS_T, frameon=False, ncol=2)


# shared scale for A/B/C so the attenuation in B is visible, not normalised away
step_v = 3.0 * SHRINK * max(a[:, i].std() for a in (v, ca, vh) for i in range(N_CELLS))
step_d = 3.0 * SHRINK * max(a[:ZOOM, i].std() for a in (dv, dvh) for i in range(N_CELLS))
step_r = 3.0 * SHRINK * max(a[:, i].std() for a in (robs, rp) for i in range(N_CELLS))

fig, axes = plt.subplots(5, 1, figsize=(7.0, 10.4))
draw(axes[0], [(v, C_GT, 1.1, 'voltage $v$')], 'A', t, step_v)
_ca_lab = 'calcium $C = K * v$' if FRAC == 0 else (
    rf'calcium $C + \gamma\epsilon$, {SNR_DB}')
draw(axes[1], [(ca, C_CA, 1.0, _ca_lab)], 'B', t, step_v, legend=True)
draw(axes[2], [(v, C_GT, 1.2, 'true $v$'), (vh, C_HAT, 0.6, r'deconvolved $\hat v$')],
     'C', t, step_v)
draw(axes[3], [(dv, C_GT, 1.2, r'true $\dot v$'), (dvh, C_HAT, 0.6, r'deconvolved $\dot{\hat v}$')],
     'D', tz, step_d)
draw(axes[4], [(robs, C_GT, 1.2, r'observable $\hat v$'), (rp, C_HAT, 0.6, 'rollout')],
     'E', t, step_r)
for a in axes:
    a.tick_params(labelsize=FS_T)
for a in axes[:3]:
    a.set_xticklabels([])
axes[3].set_xlabel('time (s)   [$2$ s zoom]', fontsize=FS_L)
axes[4].set_xlabel('time (s)', fontsize=FS_L)
fig.tight_layout(h_pad=0.7)
out = f'/workspace/connectome-gnn-cx/neurips_review/calcium_fig_{TAG}.pdf'
fig.savefig(out, dpi=200, bbox_inches='tight')
fig.savefig(out.replace('.pdf', '.png'), dpi=160, bbox_inches='tight')


def r(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / np.sqrt((a @ a) * (b @ b)))


print(f"tag={TAG}  gamma/SD={FRAC}  ({SNR_DB})  gamma={gamma:.5f}")
print(f"cells: {labels}")
print(f"C  r(v, vhat)       = {r(v, vh):.4f}")
print(f"D  r(vdot, vdothat) = {r(dv, dvh):.4f}")
print(f"   SD: vdot={dv.std():.3f}  Cdot={np.diff(ca,axis=0).std()/DT:.3f}"
      f"  ratio={np.diff(ca,axis=0).std()/np.diff(v,axis=0).std():.4f}")
print(f"E  r(observable, rollout) = {r(robs, rp):.4f}   [green = deconvolved obs, not truth]")
print(f"   r(true v, rollout)     = {r(v, rp):.4f}")
print("wrote", out)
