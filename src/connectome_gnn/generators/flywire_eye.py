"""Custom BoxEye that samples at the FlyWire column lattice.

The hybrid (flyvis + FlyWire) connectomes built by ``flyrewire`` carry an
input column lattice that does *not* match flyvis's regular hex disk:
the FlyWire columns are a slightly larger, irregular set
(~796 columns at e15 vs flyvis's 721). Feeding such a network with the
default ``BoxEye(extent=15)`` raises::

    RuntimeError: input has shape [1, 1, 1, 721]
                  but buffer has shape [1, 1, n_nodes]

because ``Stimulus.add_input`` requires
``x.shape[-1] == input_index.shape[-1] == M`` (one luminance per input
column).  The fix is to sample the rendered video at the FlyWire column
positions instead of flyvis's hex disk: same box convolution, different
receptor lattice.

This module provides:

* :class:`FlyWireBoxEye` — drop-in replacement for ``BoxEye`` that takes
  arbitrary pre-computed pixel positions.
* :func:`boxeye_from_network` — derive a ``FlyWireBoxEye`` whose
  receptors match the input columns of a flyvis ``Network`` instance,
  in the order expected by its ``Stimulus.input_index``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from flyvis.datasets.rendering import BoxEye


class FlyWireBoxEye(BoxEye):
    """``BoxEye`` whose receptors live on arbitrary pixel positions.

    Bypasses :meth:`BoxEye._receptor_centers` (which assumes a regular
    hex disk parameterized by ``extent``) and accepts pre-computed
    ``(y, x)`` integer pixel coordinates.  Reuses the parent's box-filter
    convolution and :meth:`hex_render` unchanged.
    """

    def __init__(
        self,
        receptor_centers_yx,
        kernel_size: int = 13,
        flywire_uv: Optional[Tuple[Tuple[int, int], ...]] = None,
        extent: Optional[int] = None,
    ):
        # Skip BoxEye.__init__: we must not let it call
        # self._receptor_centers() with a non-existent self.extent.
        # ``extent`` is informational only — kept so callers (e.g.
        # MultiTaskDavis) that read ``boxfilter['extent']`` or
        # ``boxfilter.extent`` for unrelated bookkeeping (HexRotate
        # construction) get a usable int. The actual sampling lattice
        # is determined entirely by ``receptor_centers_yx``.
        self.extent = extent
        self.kernel_size = kernel_size
        self.flywire_uv = flywire_uv  # for debugging / inspection only

        raw = torch.as_tensor(receptor_centers_yx, dtype=torch.long)
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError(
                "receptor_centers_yx must have shape (M, 2); "
                f"got {tuple(raw.shape)}"
            )

        # Center receptors around the geometric midpoint so that the
        # parent's hex_render formula `c = centers + [H//2, W//2]`
        # lands in the valid pixel range [0, H) x [0, W) regardless of
        # whether the original (u, v) lattice is symmetric around 0.
        ymin, xmin = raw.min(dim=0).values.tolist()
        ymax, xmax = raw.max(dim=0).values.tolist()
        offset = torch.tensor(
            [(ymin + ymax) // 2, (xmin + xmax) // 2], dtype=torch.long
        )
        self.receptor_centers = raw - offset
        self.hexals = int(self.receptor_centers.shape[0])
        self.min_frame_size = (
            self.receptor_centers.max(dim=0).values
            - self.receptor_centers.min(dim=0).values
            + 1
        )

        self._set_filter()

        pad = (self.kernel_size - 1) / 2
        self.pad = (
            int(np.ceil(pad)),
            int(np.floor(pad)),
            int(np.ceil(pad)),
            int(np.floor(pad)),
        )

    def _receptor_centers(self):  # pragma: no cover - parent contract
        # The parent's ``illustrate`` method calls this; keep it sensible
        # by yielding the stored receptor positions.
        for y, x in self.receptor_centers.tolist():
            yield (float(y), float(x))

    # --- dict-style access for callers that treat ``boxfilter`` as a config
    # --- dict (e.g. flyvis MultiTaskDavis: ``boxfilter['extent']``).
    def __getitem__(self, key):
        if key == "extent":
            return self.extent
        if key == "kernel_size":
            return self.kernel_size
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _input_columns_uv(net) -> List[Tuple[int, int]]:
    """Return the ``(u, v)`` of each input column in ``net``, in the
    order expected by :class:`flyvis.network.Stimulus.input_index`.

    Verifies that all input cell types (typically ``R1..R8``) share the
    same column ordering — this invariant is required for
    ``Stimulus.add_input`` to broadcast correctly across the
    receptor-type dimension of ``input_index``.
    """
    nodes = net.connectome.nodes
    types = np.array([
        t.decode("utf-8") if isinstance(t, bytes) else str(t)
        for t in nodes.type[:]
    ])
    u = np.asarray(nodes.u[:], dtype=np.int64)
    v = np.asarray(nodes.v[:], dtype=np.int64)
    input_types = [
        t.decode("utf-8") if isinstance(t, bytes) else str(t)
        for t in net.connectome.input_cell_types[:]
    ]
    if not input_types:
        raise ValueError("network has no input_cell_types")

    canon = input_types[0]
    canon_idx = np.where(types == canon)[0]
    cols_uv = list(zip(u[canon_idx].tolist(), v[canon_idx].tolist()))

    for t in input_types[1:]:
        idx = np.where(types == t)[0]
        other = list(zip(u[idx].tolist(), v[idx].tolist()))
        if other != cols_uv:
            raise AssertionError(
                f"input cell type {t!r} has a different column ordering "
                f"({len(other)} cols) than canonical input type "
                f"{canon!r} ({len(cols_uv)} cols); "
                f"Stimulus.add_input would broadcast incorrectly."
            )
    return cols_uv


def boxeye_from_network(
    net, kernel_size: int = 13, extent: Optional[int] = None,
) -> FlyWireBoxEye:
    """Build a :class:`FlyWireBoxEye` matching ``net``'s input columns.

    Pixel positions follow the standard flyvis ``BoxEye`` formula
    ``y = d * (u + v/2), x = d * v`` with ``d = kernel_size``, applied
    to each FlyWire column ``(u, v)`` carried by the network.

    ``extent`` is informational (used only for downstream bookkeeping,
    e.g. HexRotate construction by MultiTaskDavis). If omitted, it is
    inferred from the maximum hex-axial radius of the column lattice.
    """
    cols_uv = _input_columns_uv(net)
    d = kernel_size
    centers = [(d * (u + v / 2.0), d * v) for (u, v) in cols_uv]
    centers_yx = torch.tensor(centers, dtype=torch.long)
    if extent is None:
        extent = int(
            max(
                max(abs(u), abs(v), abs(u + v)) for (u, v) in cols_uv
            )
        )
    return FlyWireBoxEye(
        centers_yx,
        kernel_size=kernel_size,
        flywire_uv=tuple(cols_uv),
        extent=extent,
    )


def standard_boxeye_and_flywire_index(
    net, kernel_size: int = 13, margin: int = 0,
) -> Tuple[BoxEye, torch.LongTensor, int]:
    """Build a standard :class:`BoxEye` whose hex disk contains every
    FlyWire input column of ``net``, plus an index that maps standard
    hex output to FlyWire-column input.

    Rationale
    ---------
    flyvis's hex-symmetry augmentations (``HexFlip``, ``HexRotate``)
    permute receptor indices on the *regular* hex disk. They are not
    valid on the irregular FlyWire column lattice. By rendering at the
    standard hex disk with extent ``N`` ≥ max FlyWire (|u|,|v|,|u+v|),
    the augmentations remain mathematically valid (exact image-space
    symmetries), and we recover the FlyWire-column input afterwards
    with one ``index_select`` per call to ``Stimulus.add_input``.

    Returns
    -------
    boxeye : ``BoxEye``
        Standard regular-hex BoxEye covering all FlyWire columns.
    flywire_to_hex_idx : ``torch.LongTensor`` of shape ``(M,)``
        For each FlyWire input column, the index of the matching cell
        in ``boxeye`` (in the order produced by
        :meth:`BoxEye._receptor_centers`, which is also the order of
        the rendered hex output's last dimension).
    extent : int
        The hex-disk extent ``N`` used to build the BoxEye.
    """
    cols_uv = _input_columns_uv(net)
    n = max(max(abs(u), abs(v), abs(u + v)) for (u, v) in cols_uv) + int(margin)

    boxeye = BoxEye(extent=n, kernel_size=kernel_size)

    # Build (u, v) -> hex index map by replaying BoxEye._receptor_centers
    # iteration order. This is the same order BoxEye uses to fill its
    # ``receptor_centers`` and therefore the order along the last axis
    # of ``hex_render`` output.
    uv_to_hex: dict = {}
    i = 0
    for u in range(-n, n + 1):
        v_min = max(-n, -n - u)
        v_max = min(n, n - u)
        for v in range(v_min, v_max + 1):
            uv_to_hex[(u, v)] = i
            i += 1
    assert i == boxeye.hexals, (i, boxeye.hexals)

    missing = [uv for uv in cols_uv if uv not in uv_to_hex]
    if missing:
        raise AssertionError(
            f"{len(missing)} FlyWire columns lie outside the standard "
            f"hex disk of extent {n}; first few: {missing[:5]}"
        )
    idx = torch.tensor([uv_to_hex[uv] for uv in cols_uv], dtype=torch.long)
    return boxeye, idx, n


def squeezed_boxeye_and_flywire_index(
    net,
    kernel_size: int = 13,
    target_extent: Optional[int] = None,
) -> Tuple[BoxEye, torch.LongTensor, int]:
    """Like :func:`standard_boxeye_and_flywire_index`, but rendered at
    a *smaller* hex disk by relabelling FlyWire columns onto the closest
    free cell of a tighter standard hex lattice.

    Motivation
    ----------
    The FlyWire input lattice has axial radius up to 30 (~796 columns),
    forcing ``BoxEye(extent=30)`` and a 2.56× upsample of typical 480p
    sources. The resulting receptor-pair correlations decay much more
    slowly with hex distance than at extent=15 — adjacent receptors
    bin overlapping, interpolated source pixels.

    With this variant we keep the FlyWire connectivity unchanged but
    place the FlyWire columns on the smallest standard hex disk that
    can hold them all (extent ``N`` such that ``3N²+3N+1 ≥ M``,
    e.g. ``N=16`` for ``M=796``). Each FlyWire column is mapped to its
    nearest free cell in the standard disk (greedy, processed
    centre-out, using flyvis ``hex_to_pixel`` coordinates after the
    FlyWire cloud is recentred and rescaled to the standard disk's
    bounding circle). The geometry is no longer literal — it is the
    "squeeze-into-shape" convention of
    :meth:`flyrewire.FlyRewire.plot_coregistration_hulls` — but the
    per-step decorrelation matches that of canonical extent=15
    rendering.

    Returns the same triple ``(boxeye, flywire_to_hex_idx, extent)``
    as :func:`standard_boxeye_and_flywire_index`.
    """
    cols_uv = _input_columns_uv(net)
    M = len(cols_uv)

    def _hexals(N: int) -> int:
        return 3 * N * N + 3 * N + 1

    if target_extent is None:
        N = 1
        while _hexals(N) < M:
            N += 1
    else:
        N = int(target_extent)
        if _hexals(N) < M:
            raise ValueError(
                f"target_extent={N} has only {_hexals(N)} hexals, "
                f"cannot hold {M} FlyWire columns."
            )

    boxeye = BoxEye(extent=N, kernel_size=kernel_size)

    # Standard hex disk, in BoxEye._receptor_centers iteration order.
    std_uv: List[Tuple[int, int]] = []
    for u in range(-N, N + 1):
        v_min = max(-N, -N - u)
        v_max = min(N, N - u)
        for v in range(v_min, v_max + 1):
            std_uv.append((u, v))
    assert len(std_uv) == boxeye.hexals, (len(std_uv), boxeye.hexals)

    # Cartesian (flyvis hex_to_pixel) for both lattices.
    sq3 = float(np.sqrt(3.0))

    def _xy(uv_arr: np.ndarray) -> np.ndarray:
        return np.stack(
            [sq3 * (uv_arr[:, 0] + 0.5 * uv_arr[:, 1]), 1.5 * uv_arr[:, 1]],
            axis=1,
        )

    std_xy = _xy(np.asarray(std_uv, dtype=float))

    fw_arr = np.asarray(cols_uv, dtype=float)
    fw_arr = fw_arr - fw_arr.mean(axis=0, keepdims=True)
    fw_xy = _xy(fw_arr)
    fw_max_r = float(np.sqrt((fw_xy ** 2).sum(axis=1)).max())
    std_max_r = float(np.sqrt((std_xy ** 2).sum(axis=1)).max())
    if fw_max_r > 0:
        fw_xy *= std_max_r / fw_max_r

    # Greedy nearest-free-cell assignment, processed centre-out so the
    # innermost FlyWire columns claim the centre cells first and
    # peripheral ones absorb whatever leftover ring positions remain.
    order = np.argsort((fw_xy ** 2).sum(axis=1))
    D = ((fw_xy[:, None, :] - std_xy[None, :, :]) ** 2).sum(axis=-1)
    used = np.zeros(len(std_uv), dtype=bool)
    assignment = np.full(M, -1, dtype=np.int64)
    for i in order:
        for j in np.argsort(D[i]):
            if not used[j]:
                used[j] = True
                assignment[i] = int(j)
                break
    if (assignment < 0).any():
        raise RuntimeError("squeezed assignment failed (unfilled slots).")

    idx = torch.from_numpy(assignment).to(torch.long)
    return boxeye, idx, N
