"""GCaMP calcium-indicator registry for connectome-gnn.

Maps indicator names (e.g. 'gcamp7f', 'gcamp6s', 'gcamp8m') to differentiable
voltage->calcium observation models, mirroring models/registry.py.

A GCaMP model turns a neural voltage / firing-rate series into a calcium
(dF/F-like) series in a single call:

    calcium = gcamp(voltage, dt_in, dt_out)

It (1) convolves `voltage` along time with a causal calcium kernel sampled at
the input timestep `dt_in`, then (2) optionally resamples from `dt_in` to the
output timestep `dt_out` by anti-aliased area averaging (each output sample is
the mean of the calcium over its [t, t+dt_out) bin, fractional overlaps
included). With `dt_out=None` it just returns the convolution at `dt_in`.

The kernel is the difference of two exponentials (rise/decay time constants),
normalised to unit area so calcium preserves the mean of the input. The forward
is a plain torch op, so it is differentiable w.r.t. `voltage` and runs on GPU;
it accepts a torch.Tensor or a numpy array and returns the same type.

Shapes: voltage is (T, N) or (B, T, N) — time is axis -2, neurons axis -1.

Usage:
    from connectome_gnn.models.gcamp import create_gcamp, list_gcamp
    gcamp = create_gcamp("gcamp7f")
    calcium = gcamp(voltage, dt_in=0.01, dt_out=0.915)   # (..., T_out, N)

    # custom kernel
    gcamp = create_gcamp("double_exp", tau_rise=0.2, tau_decay=2.5)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

_REGISTRY: dict[str, type] = {}


def register_gcamp(*names: str):
    """Class decorator that registers a GCaMP model under one or more names."""
    def decorator(cls):
        for name in names:
            if name in _REGISTRY:
                raise ValueError(
                    f"GCaMP name '{name}' already registered to "
                    f"{_REGISTRY[name].__name__}")
            _REGISTRY[name] = cls
        return cls
    return decorator


def create_gcamp(name: str, **kwargs):
    """Look up a GCaMP model by name and instantiate it."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown GCaMP '{name}'. Available: {list_gcamp()}")
    return _REGISTRY[name](**kwargs)


def list_gcamp() -> list[str]:
    """Return the sorted list of registered GCaMP names."""
    return sorted(_REGISTRY.keys())


# --------------------------------------------------------------------------- #
#  base class
# --------------------------------------------------------------------------- #
class GCaMPKernel(nn.Module):
    """Causal calcium-indicator observation model: voltage -> calcium.

    Subclasses set `tau_rise` / `tau_decay` (seconds) or override `impulse(t)`
    for a different kernel shape. `length_s` is the kernel support; if None it
    defaults to max(7.2, 6 * tau_decay) so a slow kernel is not truncated.
    """

    tau_rise: float = 0.15
    tau_decay: float = 1.2
    length_s: float | None = None

    def __init__(self, tau_rise: float | None = None,
                 tau_decay: float | None = None,
                 length_s: float | None = None):
        super().__init__()
        if tau_rise is not None:
            self.tau_rise = float(tau_rise)
        if tau_decay is not None:
            self.tau_decay = float(tau_decay)
        if length_s is not None:
            self.length_s = float(length_s)

    # -- kernel shape ------------------------------------------------------- #
    def impulse(self, t: torch.Tensor) -> torch.Tensor:
        """Continuous-time impulse response at times t>=0 (seconds), unnormalised.
        Difference of exponentials; >=0 for tau_decay > tau_rise."""
        return torch.exp(-t / self.tau_decay) - torch.exp(-t / self.tau_rise)

    def support_s(self) -> float:
        return (self.length_s if self.length_s is not None
                else max(7.2, 6.0 * self.tau_decay))

    def kernel(self, dt: float, device=None, dtype=torch.float32) -> torch.Tensor:
        """Causal, unit-area kernel sampled at step `dt` (1-D tensor)."""
        n = max(2, int(math.ceil(self.support_s() / dt)))
        t = torch.arange(n, device=device, dtype=dtype) * dt
        k = self.impulse(t)
        s = k.sum()
        return k / s if s.abs() > 0 else k

    # -- forward ------------------------------------------------------------ #
    def forward(self, voltage, dt_in: float, dt_out: float | None = None):
        """voltage (T, N) or (B, T, N) -> calcium, same rank.

        Convolve along time at `dt_in`, then area-average resample to `dt_out`
        (no resample if `dt_out` is None or ~= dt_in)."""
        is_np = isinstance(voltage, np.ndarray)
        x = torch.as_tensor(voltage) if is_np else voltage
        if not torch.is_floating_point(x):
            x = x.float()
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)                      # (1, T, N)
        if x.dim() != 3:
            raise ValueError(f"voltage must be (T,N) or (B,T,N), got {tuple(x.shape)}")

        ca = self._convolve(x, dt_in)              # (B, T, N) at dt_in
        if dt_out is not None and abs(dt_out - dt_in) > 1e-9 * max(dt_in, 1.0):
            ca = self._area_resample(ca, dt_in, dt_out)

        if squeeze:
            ca = ca.squeeze(0)
        return ca.cpu().numpy() if is_np else ca

    # -- internals ---------------------------------------------------------- #
    def _convolve(self, x: torch.Tensor, dt_in: float) -> torch.Tensor:
        """Causal convolution of (B, T, N) with the kernel along time, via FFT
        (the kernel is one-sided, so the first T samples of the full convolution
        are exactly the causal response y[t] = sum_{j>=0} k[j] x[t-j])."""
        T = x.shape[1]
        k = self.kernel(dt_in, device=x.device, dtype=x.dtype)
        L = k.numel()
        n = T + L - 1
        Xf = torch.fft.rfft(x, n=n, dim=1)
        Kf = torch.fft.rfft(k, n=n).view(1, -1, 1)
        y = torch.fft.irfft(Xf * Kf, n=n, dim=1)
        return y[:, :T, :].contiguous()

    @staticmethod
    def _area_resample(ca: torch.Tensor, dt_in: float, dt_out: float) -> torch.Tensor:
        """Anti-aliased resample of (B, T, N) from dt_in to dt_out: each output
        sample is the mean of the sample-and-hold signal over its dt_out bin."""
        B, T, N = ca.shape
        r = dt_out / dt_in                          # input samples per output bin
        T_out = max(1, int(math.floor(T / r)))
        # cumulative integral (in sample units) of the sample-and-hold signal,
        # so the area over a fractional interval is an interpolation of C.
        C = torch.zeros(B, T + 1, N, device=ca.device, dtype=ca.dtype)
        C[:, 1:, :] = torch.cumsum(ca, dim=1)
        edges = torch.arange(T_out + 1, device=ca.device, dtype=ca.dtype) * r
        lo = torch.clamp(edges.floor().long(), 0, T)
        frac = (edges - lo).view(1, -1, 1)
        hi = torch.clamp(lo + 1, 0, T)
        C_lo = C.index_select(1, lo)
        C_hi = C.index_select(1, hi)
        C_edge = C_lo + (C_hi - C_lo) * frac        # interp cumulative at bin edges
        return (C_edge[:, 1:, :] - C_edge[:, :-1, :]) / r

    def extra_repr(self) -> str:
        return (f"tau_rise={self.tau_rise}, tau_decay={self.tau_decay}, "
                f"support_s={self.support_s():.1f}")


# --------------------------------------------------------------------------- #
#  registered presets
#
#  Nominal double-exponential approximations of the indicators' rise/decay
#  kinetics (seconds). These are coarse single-compartment fits, not exact
#  biophysical models — tune per recording / temperature if it matters.
#  gcamp7f (0.15 / 1.2) is the canonical default used by the functional panels.
# --------------------------------------------------------------------------- #
@register_gcamp("gcamp6f")
class GCaMP6f(GCaMPKernel):
    tau_rise, tau_decay = 0.07, 0.40


@register_gcamp("gcamp6m")
class GCaMP6m(GCaMPKernel):
    tau_rise, tau_decay = 0.10, 1.00


@register_gcamp("gcamp6s")
class GCaMP6s(GCaMPKernel):
    tau_rise, tau_decay = 0.18, 1.50


@register_gcamp("gcamp7f")
class GCaMP7f(GCaMPKernel):
    tau_rise, tau_decay = 0.15, 1.20


@register_gcamp("gcamp8f")
class GCaMP8f(GCaMPKernel):
    tau_rise, tau_decay = 0.02, 0.15


@register_gcamp("gcamp8m")
class GCaMP8m(GCaMPKernel):
    tau_rise, tau_decay = 0.05, 0.35


@register_gcamp("double_exp")
class DoubleExpGCaMP(GCaMPKernel):
    """Generic difference-of-exponentials; pass tau_rise / tau_decay explicitly."""


if __name__ == "__main__":
    # quick self-check: unit area, causality, and resample preserves the mean.
    g = create_gcamp("gcamp7f")
    print("registered:", list_gcamp())
    print(g)
    v = torch.zeros(2000, 3)
    v[500:] = 1.0                                   # step input
    ca = g(v, dt_in=0.01)                           # convolve only
    k = g.kernel(0.01)
    print(f"kernel len={k.numel()} sum={k.sum():.6f} (should be 1)")
    print(f"calcium before step (t<5s) max={ca[:500].abs().max():.4f} (causal -> ~0)")
    print(f"calcium settles to ~1: {ca[-1].mean():.4f}")
    ca_ds = g(v, dt_in=0.01, dt_out=0.915)          # convolve + resample
    print(f"resampled shape {tuple(ca_ds.shape)}; "
          f"mean preserved: {ca.mean():.4f} vs {ca_ds.mean():.4f}")
