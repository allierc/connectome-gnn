"""Shared figure style for the zebrafish figures.

``open_axes(fig)`` drops the top and right spines from every line/scatter panel
of a figure, leaving only the x and y axis. Panels that contain an image
(``imshow`` kinographs, connectivity matrices) or a colorbar keep their full
frame. Call it just before ``fig.savefig`` so the whole figure follows the
open-axes convention used across the paper.
"""
from __future__ import annotations


def open_axes(fig):
    for ax in fig.axes:
        if getattr(ax, "images", None):                 # imshow panels keep their frame
            continue
        if getattr(ax, "_colorbar", None) is not None:  # colorbars keep their frame
            continue
        spines = getattr(ax, "spines", {})
        for sp in ("top", "right"):
            if sp in spines:
                spines[sp].set_visible(False)
    return fig
