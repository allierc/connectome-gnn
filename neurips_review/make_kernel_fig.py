#!/usr/bin/env python
"""GCaMP6f kernel and its transfer function -- why B -> C is possible.

The kernel is GIVEN: built from the same config fields the generator used
(tau_rise, tau_decay, length, dt), so the inverse knows K exactly. Panel B shows
the attenuation |K(f)| that makes the calcium trace look smooth, and the point
that it never reaches zero: the high-frequency content is scaled down, not
destroyed, so dividing it back out is exact in the absence of noise.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
RC = '/workspace/connectome-gnn/figures/janne.matplotlibrc'
if os.path.isfile(RC):
    matplotlib.rc_file(RC)
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/connectome-gnn-ca/src')
from connectome_gnn.generators.gcamp_kernel import build_kernel_from_config
from connectome_gnn.config import NeuralGraphConfig

cfg = NeuralGraphConfig.from_yaml(
    '/workspace/connectome-gnn-ca/config/fly/nr2_ca_calcium_unified.yaml')
sim = cfg.simulation
k = build_kernel_from_config(sim, device='cpu').numpy().astype(np.float64)
dt = float(sim.calcium_kernel_dt_seconds)
L = len(k)

nfft = 8192
K = np.fft.rfft(k, n=nfft)
f = np.fft.rfftfreq(nfft, d=dt)
mag = np.abs(K)
nyq = 1.0 / (2 * dt)

FS_L, FS_T = 9, 7
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))

ax[0].plot(np.arange(L) * dt, k, color='#d62728', lw=1.4)
ax[0].set_xlabel('time (s)', fontsize=FS_L)
ax[0].set_ylabel('$K(t)$', fontsize=FS_L)
ax[0].text(-0.16, 1.0, 'A', transform=ax[0].transAxes, va='bottom', ha='left',
           fontsize=FS_L + 3, fontweight='bold')
ax[0].text(0.97, 0.9, rf'$\tau_r={sim.calcium_kernel_tau_rise*1000:.0f}$ ms'
                      f'\n' rf'$\tau_d={sim.calcium_kernel_tau_decay*1000:.0f}$ ms',
           transform=ax[0].transAxes, ha='right', va='top', fontsize=FS_T)

ax[1].semilogy(f, mag / mag.max(), color='#d62728', lw=1.4)
ax[1].axhline(mag.min() / mag.max(), color='0.5', ls=':', lw=0.9)
ax[1].set_xlabel('frequency (Hz)', fontsize=FS_L)
ax[1].set_ylabel(r'$|K(f)|\,/\,|K|_{\max}$', fontsize=FS_L)
ax[1].set_xlim(0, nyq)
ax[1].text(-0.18, 1.0, 'B', transform=ax[1].transAxes, va='bottom', ha='left',
           fontsize=FS_L + 3, fontweight='bold')
ax[1].text(0.97, 0.93, f'min $= {mag.min()/mag.max():.1e}$\nno zeros in band',
           transform=ax[1].transAxes, ha='right', va='top', fontsize=FS_T)
for a in ax:
    a.tick_params(labelsize=FS_T)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)

fig.tight_layout()
out = '/workspace/connectome-gnn-cx/neurips_review/kernel_fig.pdf'
fig.savefig(out, dpi=200, bbox_inches='tight')
fig.savefig(out.replace('.pdf', '.png'), dpi=160, bbox_inches='tight')
print(f"L={L} samples ({L*dt:.2f}s)  sum={k.sum():.4f}  peak at {k.argmax()*dt:.2f}s")
print(f"|K| min/max = {mag.min()/mag.max():.4e}  -> worst-case amplification {mag.max()/mag.min():.0f}x")
print(f"float32 relative precision ~1e-7 -> deconvolved error floor ~{1e-7*mag.max()/mag.min():.1e}")
print("wrote", out)
