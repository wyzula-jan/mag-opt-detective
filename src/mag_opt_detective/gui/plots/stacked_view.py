"""All spectra of a map on top of each other, shifted by a constant offset."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor

from mag_opt_detective.core.colormaps import lut
from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.core.units import axis_label
from mag_opt_detective.gui.plots.base import PlotView, Range, set_range

# part of the colour map used to colour traces by field: skips the ends, which can
# disappear into the background (black in magma, white in grey)
_FIELD_COLOURS = (0.1, 0.82)
_DEFAULT_TRACE_CMAP = "viridis"


class StackedPlot(PlotView):
    """All spectra of a map on top of each other, shifted by a constant offset.

    By default every field gets a trace in its own hue; see :meth:`set_trace_options`.
    Emits :attr:`cursorMoved` ``(energy, y, None)``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._curves: list[pg.PlotDataItem] = []
        self._shown = np.array([], dtype=int)  # field index of each curve
        self._fmap: FieldMap | None = None
        self._offset = 0.0
        self._every = 1
        self._by_field = False
        self._trace_cmap: str | None = None
        self.plot.setLabel("bottom", "Energy")
        self.plot.setLabel("left", "Intensity (a.u.)")
        self.plot.setClipToView(True)
        self.plot.setDownsampling(auto=True, mode="peak")
        self.apply_theme(self._colors)

    def _cursor_text(self, x: float, y: float) -> str:
        return f"E = {x:.3f}    I = {y:.4g}"

    def set_trace_options(
        self, every: int = 1, color_by_field: bool = False, cmap: str | None = None
    ) -> None:
        """Show every *every*-th spectrum; colour them by field with *cmap* (viridis if None)."""
        if every < 1:
            raise ValueError("every must be at least 1")
        self._every = int(every)
        self._by_field = bool(color_by_field)
        self._trace_cmap = cmap
        if self._fmap is not None:
            self._draw()

    def set_map(
        self, fmap: FieldMap, offset: float, y_range: Range = None, x_range: Range = None
    ) -> None:
        self.clear_map()
        self._fmap = fmap
        self._offset = float(offset)
        self.plot.setLabel("bottom", axis_label(fmap.unit))
        self._draw()
        set_range(self.plot, x_range, y_range)

    def _pens(self, fmap: FieldMap, shown: np.ndarray) -> list:
        n = fmap.field.size
        if not self._by_field:
            return [pg.intColor(int(j), hues=max(n, 1)) for j in shown]
        table = lut(self._trace_cmap or _DEFAULT_TRACE_CMAP)
        field = fmap.field
        span = float(field.max() - field.min()) if n else 0.0
        t = (field[shown] - field.min()) / span if span > 0 else np.zeros(shown.size)
        lo, hi = _FIELD_COLOURS
        index = np.rint((lo + (hi - lo) * t) * (len(table) - 1)).astype(int)
        return [pg.mkPen(QColor(*(int(c) for c in table[i]))) for i in index]

    def _draw(self) -> None:
        fmap = self._fmap
        self._remove_curves()
        if fmap is None:
            return
        self._shown = np.arange(0, fmap.field.size, self._every)
        for k, (j, pen) in enumerate(zip(self._shown, self._pens(fmap, self._shown), strict=True)):
            curve = self.plot.plot(fmap.energy, fmap.values[:, j] + k * self._offset, pen=pen)
            self._curves.append(curve)

    def _remove_curves(self) -> None:
        for curve in self._curves:
            self.plot.removeItem(curve)
        self._curves.clear()
        self._shown = np.array([], dtype=int)

    def clear_map(self) -> None:
        self._remove_curves()
        self._fmap = None

    # ------------------------------------------------------------------ traces
    def shown_fields(self) -> np.ndarray:
        """Field indices of the traces shown, bottom first."""
        return self._shown.copy()

    def _traces_at(self, energy: float) -> np.ndarray | None:
        """Plotted y of every shown trace at *energy* (NaN where missing), None outside."""
        fmap = self._fmap
        if fmap is None or fmap.energy.size == 0 or self._shown.size == 0:
            return None
        axis = fmap.energy
        values = fmap.values[:, self._shown]
        if axis[0] > axis[-1]:
            axis, values = axis[::-1], values[::-1]
        if not axis[0] <= energy <= axis[-1]:
            return None
        i = int(np.clip(np.searchsorted(axis, energy, side="right") - 1, 0, axis.size - 1))
        if i == axis.size - 1 or axis[i + 1] == axis[i]:
            row = values[i]
        else:
            w = (energy - axis[i]) / (axis[i + 1] - axis[i])
            row = (1 - w) * values[i] + w * values[i + 1]
        return row + np.arange(self._shown.size) * self._offset

    def trace_y(self, j: int, energy: float) -> float | None:
        """Plotted y of field index *j* at *energy* (with its offset), None if not shown."""
        position = np.flatnonzero(self._shown == j)
        if position.size == 0:
            return None
        row = self._traces_at(energy)
        if row is None or not np.isfinite(row[position[0]]):
            return None
        return float(row[position[0]])

    def trace_at(self, x: float, y: float) -> int | None:
        """Field index of the shown trace nearest to (*x* = energy, *y*), or None."""
        row = self._traces_at(x)
        if row is None or not np.isfinite(row).any():
            return None
        return int(self._shown[np.nanargmin(np.abs(row - y))])
