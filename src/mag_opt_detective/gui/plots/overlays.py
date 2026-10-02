"""Overlay layers: named groups of curves and markers drawn on top of a plot."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg

Curve = tuple[np.ndarray, np.ndarray]

_BASE_Z = 10.0  # above the image (z = 0), below the crosshair


class OverlayLayer:
    """Curves (optionally with a wider shadow below them) and scatter markers.

    Items never take part in auto-ranging, and later layers are drawn above earlier ones.
    """

    def __init__(self, name: str, plot: pg.PlotItem, z: float):
        self.name = name
        self._plot = plot
        self._z = z
        self._visible = True
        self._curves: list[pg.PlotDataItem] = []
        self._shadows: list[pg.PlotDataItem] = []
        self._scatters: list[pg.ScatterPlotItem] = []
        self._n_curves = 0
        self._n_shadows = 0
        self._n_scatters = 0

    def _add(self, item: pg.GraphicsObject, dz: float) -> None:
        item.setZValue(self._z + dz)
        self._plot.addItem(item, ignoreBounds=True)

    def _sync(self) -> None:
        for items, n in (
            (self._curves, self._n_curves),
            (self._shadows, self._n_shadows),
            (self._scatters, self._n_scatters),
        ):
            for i, item in enumerate(items):
                item.setVisible(self._visible and i < n)

    def set_curves(self, curves: Sequence[Curve], pen, shadow_pen=None) -> None:
        """Replace the curves; *shadow_pen* draws an outline below each one."""
        for i, (x, y) in enumerate(curves):
            if i == len(self._curves):
                self._curves.append(pg.PlotDataItem())
                self._add(self._curves[i], 0.2)
            self._curves[i].setData(x, y)
            self._curves[i].setPen(pen)
            if shadow_pen is not None:
                if i == len(self._shadows):
                    self._shadows.append(pg.PlotDataItem())
                    self._add(self._shadows[i], 0.1)
                self._shadows[i].setData(x, y)
                self._shadows[i].setPen(shadow_pen)
        self._n_curves = len(curves)
        self._n_shadows = len(curves) if shadow_pen is not None else 0
        self._sync()

    def set_points(self, x, y, symbol="o", size=8, pen=None, brush=None) -> None:
        """Replace all markers by one group; *pen* and *brush* may be lists (one per point)."""
        self._n_scatters = 0
        self.add_points(x, y, symbol=symbol, size=size, pen=pen, brush=brush)

    def add_points(self, x, y, symbol="o", size=8, pen=None, brush=None) -> None:
        """Add a group of markers above the existing ones."""
        i = self._n_scatters
        if i == len(self._scatters):
            self._scatters.append(pg.ScatterPlotItem())
            self._add(self._scatters[i], 0.3 + 0.01 * i)
        scatter = self._scatters[i]
        scatter.setData(
            x=np.asarray(x, dtype=float),
            y=np.asarray(y, dtype=float),
            symbol=symbol,
            size=size,
            pen=pen if pen is not None else pg.mkPen("w"),
            brush=brush,
        )
        self._n_scatters = i + 1
        self._sync()

    def clear(self) -> None:
        """Remove all curves and markers (the items are kept for reuse)."""
        self._n_curves = self._n_shadows = self._n_scatters = 0
        for scatter in self._scatters:
            scatter.clear()
        self._sync()

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        self._sync()

    def is_visible(self) -> bool:
        return self._visible

    def curve_data(self) -> list[Curve]:
        """(x, y) of the curves currently shown."""
        return [c.getData() for c in self._curves[: self._n_curves] if c.isVisible()]

    def point_data(self) -> list[Curve]:
        """(x, y) of each marker group currently shown."""
        return [s.getData() for s in self._scatters[: self._n_scatters] if s.isVisible()]

    def items(self) -> list[pg.GraphicsObject]:
        """All graphics items of the layer (shown or not)."""
        return [*self._shadows, *self._curves, *self._scatters]


class OverlayMixin:
    """Registry of overlay layers on ``self.plot``, in creation order."""

    plot: pg.PlotItem

    def _init_overlays(self) -> None:
        self._layers: dict[str, OverlayLayer] = {}

    def layer(self, name: str) -> OverlayLayer:
        """The layer *name*, created above all existing layers on first use."""
        layer = self._layers.get(name)
        if layer is None:
            layer = OverlayLayer(name, self.plot, _BASE_Z + len(self._layers))
            self._layers[name] = layer
        return layer

    def layers(self) -> list[OverlayLayer]:
        """All layers, bottom first."""
        return list(self._layers.values())
