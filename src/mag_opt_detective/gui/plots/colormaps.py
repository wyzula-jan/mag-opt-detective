"""pyqtgraph colour maps built from the shared stops in :mod:`mag_opt_detective.core.colormaps`."""

from __future__ import annotations

from functools import cache

import numpy as np
import pyqtgraph as pg

from mag_opt_detective.core import colormaps as core

names = core.names


@cache
def colormap(name: str) -> pg.ColorMap:
    """The colour map *name* as a pyqtgraph ColorMap (do not modify it)."""
    table = core.stops(name)
    pos = np.array([p for p, _ in table])
    color = np.array([c for _, c in table], dtype=np.ubyte)
    return pg.ColorMap(pos, color, name=name)


@cache
def _lut(name: str, n: int) -> np.ndarray:
    table = core.lut(name, n)
    table.flags.writeable = False
    return table


def lookup_table(name: str, n: int = 256) -> np.ndarray:
    """Read-only ``uint8`` lookup table ``(n, 3)`` for ImageItem.setLookupTable."""
    return _lut(name, n)
