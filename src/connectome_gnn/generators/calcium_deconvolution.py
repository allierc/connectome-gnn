"""Calcium -> voltage: Tikhonov deconvolution with a blind choice of lambda.

THE FORWARD MODEL. A calcium indicator reports C = K * v + e: the voltage
convolved with a causal, unit-area kernel K (GCaMP6f here: difference of
exponentials, rise 75 ms, decay 400 ms, 2.4 s support; models/gcamp.py), plus
white measurement noise e.

THE INVERSE. In Fourier space, with a smoothness penalty on v,

    V(w) = conj(K(w)) F(w) / ( |K(w)|^2 + lambda * max|K|^2 * |1 - e^{-iw}|^4 ),

the Tikhonov (Wiener-like) estimate that penalises the second difference of v.
Two details decide whether it works (calcium_note.tex, 2026-07): the trace is
reflect-padded by one kernel length before the FFT (without it r(vdot) fell
from 0.97 to 0.24: the circular convolution wraps the end of the trace onto its
start), and the first frames, where the causal kernel has not yet seen enough
input, are held at the first reliable value.

BLIND LAMBDA. The earlier builder chose lambda by maximising r(vdot) against the
TRUE vdot -- an oracle. Here the measurement-noise SD is estimated per trace
from the autocovariance (white noise sits only at lag 0, so extrapolating lags
1-4 back to lag 0 gives the signal's variance, and the excess at lag 0 is the
noise's), and lambda is the largest value on a grid whose residual
||K * v_hat - F|| stays within that noise (Morozov's discrepancy principle). On
250 flyvis neurons it matched the oracle's r(vdot): 0.965 against 0.997 with no
measurement noise, 0.377 against 0.377 at 40 dB; GCV and a high-band noise
estimate failed with no noise because the dynamics carry high-frequency power.
"""
import numpy as np

# The first frames are held at this one: the kernel's rise needs a few samples.
N_WARMUP = 4
# A floor for the noise estimate when there is none: the discrepancy principle
# then asks for the smallest lambda on the grid.
_NOISE_FLOOR = 1e-9


def _penalty_spectrum(n):
    w = 2.0 * np.pi * np.fft.rfftfreq(n)
    return np.abs(1.0 - np.exp(-1j * w)) ** 4


def wiener_deconvolve(calcium, kernel, lam):
    """Tikhonov deconvolution of `calcium` (T, N) or (T,) by `kernel` (L,).

    `lam` is a scalar or one value per trace (N,). Returns the voltage estimate,
    same shape as `calcium`.
    """
    c = np.asarray(calcium, dtype=np.float64)
    squeeze = c.ndim == 1
    if squeeze:
        c = c[:, None]
    T = c.shape[0]
    k = np.asarray(kernel, dtype=np.float64).ravel()
    pad = min(len(k), T - 1)
    cp = np.concatenate([c[pad:0:-1], c, c[-2:-pad - 2:-1]], axis=0)
    n = cp.shape[0]
    K = np.fft.rfft(k, n=n)
    P = _penalty_spectrum(n)
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    denom = (np.abs(K) ** 2)[:, None] + lam[None, :] * (np.abs(K) ** 2).max() * P[:, None]
    V = np.conj(K)[:, None] * np.fft.rfft(cp, axis=0) / denom
    est = np.fft.irfft(V, n=n, axis=0)[pad:pad + T]
    est[:N_WARMUP] = est[N_WARMUP]
    return est[:, 0] if squeeze else est


def estimate_noise_sd(calcium):
    """Measurement-noise SD from the autocovariance, pooled over traces.

    The autocovariance g of signal + white noise is the signal's at every lag
    but 0, where the noise variance adds. The signal's lag-0 value is the cubic
    through lags 1..3 evaluated at 0, 3 g1 - 3 g2 + g3; the excess of g0 over it
    is the noise variance. (A straight line through lags 1-3 overshoots lag 0 for
    a smooth trace like calcium and returned 0; a quadratic in the lag undershot
    it and read noise where there was none.)
    """
    x = np.asarray(calcium, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    x = x - x.mean(0)
    T = x.shape[0]
    g = [(x[:T - l] * x[l:]).mean() for l in range(4)]
    return float(np.sqrt(max(g[0] - (3 * g[1] - 3 * g[2] + g[3]), 0.0)))


def choose_lambda(calcium, kernel, grid=None, noise_sd=None):
    """One blind lambda for all traces, by the discrepancy principle.

    The largest lambda on `grid` whose mean squared residual |(1 - h) F|^2 (in
    Fourier space, h the filter's transfer function) does not exceed the
    estimated noise variance; the smallest grid value when none does. Returns
    (lambda, noise SD).
    """
    c = np.asarray(calcium, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, None]
    grid = (np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0])
            if grid is None else np.asarray(grid, dtype=np.float64))
    sd = estimate_noise_sd(c) if noise_sd is None else float(noise_sd)
    k = np.asarray(kernel, dtype=np.float64).ravel()
    T = c.shape[0]
    pad = min(len(k), T - 1)
    cp = np.concatenate([c[pad:0:-1], c, c[-2:-pad - 2:-1]], axis=0)
    n = cp.shape[0]
    F2 = np.abs(np.fft.rfft(cp, axis=0)) ** 2
    K2 = np.abs(np.fft.rfft(k, n=n)) ** 2
    P = _penalty_spectrum(n)
    wt = np.ones(len(P)); wt[1:-1] = 2.0              # rfft holds half the spectrum
    chosen = grid[0]
    for lam in grid:                                   # ascending
        h = K2 / (K2 + lam * K2.max() * P)
        res = (((1.0 - h) ** 2 * wt)[:, None] * F2).sum(0).mean() / n / n
        if res <= sd ** 2:
            chosen = lam
    return float(chosen), sd
