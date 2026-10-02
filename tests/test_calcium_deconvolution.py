"""Calcium -> voltage deconvolution: exact without noise, blind lambda sane."""
import numpy as np

from connectome_gnn.generators.calcium_deconvolution import (
    choose_lambda, estimate_noise_sd, wiener_deconvolve)
from connectome_gnn.models.gcamp import create_gcamp


def _kernel():
    return create_gcamp("double_exp", tau_rise=0.075, tau_decay=0.4,
                        length_s=2.4).kernel(0.02).numpy().astype(np.float64)


def _signal(T=6000, N=4, seed=0):
    rng = np.random.default_rng(seed)
    v = np.cumsum(rng.normal(size=(T, N)), 0)
    v = np.stack([np.convolve(v[:, j], np.ones(25) / 25, "same") for j in range(N)], 1)
    k = _kernel()
    c = np.stack([np.convolve(v[:, j], k)[:T] for j in range(N)], 1)
    return v, c, k


def test_the_kernel_has_unit_area():
    assert abs(_kernel().sum() - 1.0) < 1e-6


def test_noise_free_calcium_deconvolves_to_the_voltage():
    v, c, k = _signal()
    est = wiener_deconvolve(c, k, 1e-8)
    inner = slice(200, -200)
    r = [np.corrcoef(est[inner, j], v[inner, j])[0, 1] for j in range(v.shape[1])]
    assert min(r) > 0.999


def test_the_noise_estimate_finds_white_noise():
    v, c, k = _signal(T=20000, N=3, seed=1)
    sd = 0.05 * c.std()
    noisy = c + np.random.default_rng(2).normal(size=c.shape) * sd
    assert abs(estimate_noise_sd(noisy) / sd - 1.0) < 0.15
    assert estimate_noise_sd(c) < 0.05 * sd


def test_blind_lambda_grows_with_noise():
    v, c, k = _signal(T=8000, N=3, seed=3)
    rng = np.random.default_rng(4)
    lam_lo, _ = choose_lambda(c + rng.normal(size=c.shape) * 0.001 * c.std(), k)
    lam_hi, _ = choose_lambda(c + rng.normal(size=c.shape) * 0.1 * c.std(), k)
    assert lam_hi > lam_lo
