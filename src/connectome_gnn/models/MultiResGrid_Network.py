"""Multi-resolution feature grids (pure PyTorch, no tinycudann).

InstantNGP-style 1-D and 2-D feature grids.  Each level is an
nn.Embedding grid with linear / bilinear interpolation — local in input
space, no waterbed problem.  Comparable quality to tinycudann HashGrid
for low-dimensional inputs with the advantage of zero external
dependencies.

Public classes:
    MultiResTemporalGrid
        1-D temporal grid + decoder MLP. Maps t in [0, 1] to (B, n_output).
    MultiResHexGrid2D
        2-D positional grid (no decoder). Maps (u, v) in [0, 1]^2 to
        per-query feature vector of size n_levels * n_features_per_level.
        Used as the spatial branch of MultiResSpatioTemporalGrid.
    MultiResSpatioTemporalGrid
        Wraps a 1-D temporal grid and a 2-D positional grid; the
        concatenated features are decoded by a shared MLP. Per-query
        evaluation: forward(t, pos) -> (B, n_output) with the spatial
        features looked up at each neuron position.
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


class MultiResHexGrid2D(nn.Module):
    """Multi-resolution 2-D feature grid with bilinear interpolation.

    Each level holds an nn.Embedding of size (R+1)*(R+1) x n_features
    laid out in row-major (i_u * (R+1) + i_v). A query at (u, v) in
    [0, 1]^2 reads the four neighbouring cells and bilinearly
    interpolates their feature vectors. Concatenating across levels
    gives a (n_levels * n_features_per_level)-dim feature per query.

    Args:
        n_levels: number of grid levels.
        n_features_per_level: learnable features per cell.
        base_resolution: coarsest grid resolution.
        per_level_scale: resolution multiplier per level.
    """

    def __init__(
        self,
        n_levels: int = 6,
        n_features_per_level: int = 4,
        base_resolution: int = 4,
        per_level_scale: float = 1.5,
    ):
        super().__init__()

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level

        self.grids = nn.ModuleList()
        self.resolutions: list[int] = []
        res = float(base_resolution)
        for _ in range(n_levels):
            r = max(2, int(res))
            emb = nn.Embedding((r + 1) * (r + 1), n_features_per_level)
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)
            self.grids.append(emb)
            self.resolutions.append(r)
            res *= per_level_scale

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """pos: (B, 2) in [0, 1]^2 -> (B, n_levels * n_features_per_level)."""
        u = pos[:, 0]
        v = pos[:, 1]
        features = []
        for emb, res in zip(self.grids, self.resolutions):
            stride = res + 1
            pu = u * res
            pv = v * res
            iu0 = pu.long().clamp(0, res - 1)
            iv0 = pv.long().clamp(0, res - 1)
            iu1 = (iu0 + 1).clamp(0, res)
            iv1 = (iv0 + 1).clamp(0, res)
            wu = (pu - pu.floor())
            wv = (pv - pv.floor())
            f00 = emb(iu0 * stride + iv0)
            f01 = emb(iu0 * stride + iv1)
            f10 = emb(iu1 * stride + iv0)
            f11 = emb(iu1 * stride + iv1)
            wu = wu.unsqueeze(1)
            wv = wv.unsqueeze(1)
            feat = ((1 - wu) * (1 - wv) * f00
                    + (1 - wu) * wv * f01
                    + wu * (1 - wv) * f10
                    + wu * wv * f11)
            features.append(feat)
        return torch.cat(features, dim=1)


class MultiResSpatioTemporalGrid(nn.Module):
    """1-D temporal grid + 2-D spatial grid + shared decoder MLP.

    Each query is per-(t, pos): forward(t, pos) returns (B, n_output)
    where B is the batch dimension (number of neurons or
    time x neuron pairs). The spatial features are looked up at every
    query position, so neighbouring positions in (u, v) share grid
    cells and the decoder borrows across them.

    Args:
        See MultiResTemporalGrid for the temporal half and
        MultiResHexGrid2D for the spatial half.
    """

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 4,
        base_resolution: int = 16,
        per_level_scale: float = 1.4,
        spatial_n_levels: int = 6,
        spatial_n_features_per_level: int = 4,
        spatial_base_resolution: int = 4,
        spatial_per_level_scale: float = 1.5,
        n_output: int = 1,
        mlp_width: int = 256,
        mlp_layers: int = 2,
        a_dim: int = 0,
    ):
        super().__init__()

        self.temporal = nn.ModuleList()
        self.t_resolutions: list[int] = []
        res = float(base_resolution)
        for _ in range(n_levels):
            r = max(2, int(res))
            emb = nn.Embedding(r + 1, n_features_per_level)
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)
            self.temporal.append(emb)
            self.t_resolutions.append(r)
            res *= per_level_scale

        self.spatial = MultiResHexGrid2D(
            n_levels=spatial_n_levels,
            n_features_per_level=spatial_n_features_per_level,
            base_resolution=spatial_base_resolution,
            per_level_scale=spatial_per_level_scale,
        )

        n_t = n_levels * n_features_per_level
        n_s = spatial_n_levels * spatial_n_features_per_level
        # When a_dim > 0 the decoder also consumes a per-query embedding
        # (typically the GNN's self.a[ids]) so cell-type / per-neuron
        # identity is available non-linearly throughout the decoder MLP,
        # not just as an additive low-rank correction.
        self.a_dim = int(a_dim)
        body_layers: list[nn.Module] = [
            nn.Linear(n_t + n_s + self.a_dim, mlp_width),
            nn.ReLU(),
        ]
        for _ in range(mlp_layers - 1):
            body_layers += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        self.mlp_body = nn.Sequential(*body_layers)
        self.head = nn.Linear(mlp_width, n_output)

    def _temporal_features(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B, 1) in [0, 1] -> (B, n_t)."""
        t = t.squeeze(1)
        feats = []
        for emb, res in zip(self.temporal, self.t_resolutions):
            pos = t * res
            i0 = pos.long().clamp(0, res - 1)
            i1 = (i0 + 1).clamp(0, res)
            w1 = (pos - pos.floor()).unsqueeze(1)
            w0 = 1.0 - w1
            feats.append(w0 * emb(i0) + w1 * emb(i1))
        return torch.cat(feats, dim=1)

    def forward(self, t: torch.Tensor, pos: torch.Tensor,
                a: torch.Tensor = None,
                return_features: bool = False):
        """Per-query forward.

        Two broadcast modes are supported for ``t`` and ``pos``:
          * t shape (B, 1) and pos shape (B, 2): one (t, pos) pair per row.
          * t shape (1, 1) and pos shape (B, 2): single time, B positions.
            The temporal feature is broadcast across B.

        ``a`` (optional, required when self.a_dim > 0) is the per-query
        identity embedding. Shape (B, a_dim) — broadcast against t/pos
        the same way (singleton dim 0 is expanded).

        Returns:
            out: (B, n_output)
            (out, feat) if return_features=True with feat shape (B, mlp_width).
        """
        t_feat = self._temporal_features(t)                       # (Bt, n_t)
        s_feat = self.spatial(pos)                                # (Bp, n_s)
        if t_feat.shape[0] == 1 and s_feat.shape[0] != 1:
            t_feat = t_feat.expand(s_feat.shape[0], -1)
        elif s_feat.shape[0] == 1 and t_feat.shape[0] != 1:
            s_feat = s_feat.expand(t_feat.shape[0], -1)
        feats = [t_feat, s_feat]
        if self.a_dim > 0:
            if a is None:
                raise RuntimeError(
                    "MultiResSpatioTemporalGrid built with a_dim>0 requires "
                    "an `a` tensor at forward()."
                )
            B_target = t_feat.shape[0]
            if a.shape[0] == 1 and B_target != 1:
                a = a.expand(B_target, -1)
            elif a.shape[0] != B_target:
                raise RuntimeError(
                    f"a has batch dim {a.shape[0]} but expected {B_target} "
                    f"to match t/pos."
                )
            feats.append(a)
        feat = self.mlp_body(torch.cat(feats, dim=1))             # (B, mlp_width)
        out = self.head(feat)                                     # (B, n_output)
        if return_features:
            return out, feat
        return out
