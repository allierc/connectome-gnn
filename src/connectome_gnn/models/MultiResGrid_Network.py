"""Multi-resolution 1-D temporal feature grid (pure PyTorch, no tinycudann).

InstantNGP-style architecture for 1-D time input.  Each level is an
nn.Embedding grid with linear interpolation — local in time, no waterbed
problem.  Comparable quality to tinycudann HashGrid for 1-D inputs with
the advantage of zero external dependencies.

Usage:
    model = MultiResTemporalGrid(
        n_levels=24, n_features_per_level=4,
        base_resolution=16, per_level_scale=1.4,
        n_output=1000, mlp_width=512, mlp_layers=4,
    )
    t = torch.rand(96, 1)   # normalized in [0, 1]
    y = model(t)             # (96, 1000)
"""

import torch
import torch.nn as nn


class MultiResTemporalGrid(nn.Module):
    """Multi-resolution 1-D temporal feature grid + MLP.

    Maps t ∈ [0, 1] to n_output values using:
      1. Multi-resolution grid encoding (24 levels × n_features, linear interp)
      2. PyTorch MLP to expand to desired output dims

    Locality: each time sample reads only 2 neighbouring grid cells per
    level — updating one frame does not corrupt others (no waterbed).

    Args:
        n_levels: number of grid levels (default 24)
        n_features_per_level: learnable features per grid cell (default 4)
        base_resolution: coarsest grid resolution (default 16)
        per_level_scale: resolution multiplier per level (default 1.4)
        n_output: output dimension (number of neurons)
        mlp_width: hidden width of the MLP decoder
        mlp_layers: number of hidden layers in the MLP decoder
    """

    def __init__(
        self,
        n_levels: int = 24,
        n_features_per_level: int = 4,
        base_resolution: int = 16,
        per_level_scale: float = 1.4,
        n_output: int = 1000,
        mlp_width: int = 512,
        mlp_layers: int = 4,
    ):
        super().__init__()

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level

        self.grids = nn.ModuleList()
        self.resolutions: list[int] = []
        res = float(base_resolution)
        for _ in range(n_levels):
            r = max(2, int(res))
            emb = nn.Embedding(r + 1, n_features_per_level)
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)
            self.grids.append(emb)
            self.resolutions.append(r)
            res *= per_level_scale

        n_enc = n_levels * n_features_per_level
        # Split MLP into a shared body (grid encoding -> (B, mlp_width)) and a
        # head (mlp_width -> n_output). Consumers can request the body's
        # output via forward(..., return_features=True) to compose additional
        # low-rank / factorized paths without re-running the grid lookup.
        body_layers: list[nn.Module] = [nn.Linear(n_enc, mlp_width), nn.ReLU()]
        for _ in range(mlp_layers - 1):
            body_layers += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        self.mlp_body = nn.Sequential(*body_layers)
        self.head = nn.Linear(mlp_width, n_output)

    def forward(self, t: torch.Tensor, return_features: bool = False):
        """t: (B, 1) normalized in [0, 1]  →  (B, n_output)

        If return_features=True, also returns the pre-head features of shape
        (B, mlp_width) as the second element of a tuple. Those features are
        the shared time representation used by the factorized head in
        NeuralGNN.forward_{hidden,anchor}[_batched].
        """
        t = t.squeeze(1)   # (B,)
        features = []
        for emb, res in zip(self.grids, self.resolutions):
            pos = t * res
            i0  = pos.long().clamp(0, res - 1)
            i1  = (i0 + 1).clamp(0, res)
            w1  = (pos - pos.floor()).unsqueeze(1)
            w0  = 1.0 - w1
            features.append(w0 * emb(i0) + w1 * emb(i1))
        feat = self.mlp_body(torch.cat(features, dim=1))  # (B, mlp_width)
        out = self.head(feat)                              # (B, n_output)
        if return_features:
            return out, feat
        return out
