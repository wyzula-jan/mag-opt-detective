"""Colour map of intensity over field and energy, with its colour scale."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPalette
from PySide6.QtWidgets import QVBoxLayout, QWidget

from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.core.units import axis_label
from mag_opt_detective.gui.plots.base import (
    AxisCells,
    PartPainter,
    PlotView,
    Range,
    map_cells,
    robust_levels,
    set_range,
)
from mag_opt_detective.gui.plots.colors import PlotColors
from mag_opt_detective.gui.plots.colorscale import ColorScale, HistogramScale, make_scale


class ColorMapPlot(PlotView):
    """Intensity as a function of field (x) and energy (y) with a colour scale beside it.

    The scale sits in :attr:`scale_container`, a separate widget right of the plot.
    Emits :attr:`pointClicked` ``(field, energy)`` on a left click inside the plot,
    :attr:`levelsEdited` when the user changes the levels on the scale and
    :attr:`cursorMoved` ``(field, energy, value or None)``.
    """

    pointClicked = Signal(float, float)
    levelsEdited = Signal(float, float)  # the user changed the levels on the scale

    def __init__(self, parent=None, scale_style: str = "histogram"):
        super().__init__(parent)
        self._unit = ""
        self._cmap = ""
        self._fmap: FieldMap | None = None
        self._cells: tuple[AxisCells, AxisCells] | None = None  # (field, energy) as drawn
        self._auto_levels = True
        self._margins = (0, 0)
        self.plot.setLabel("bottom", "Magnetic Field (T)")
        self.plot.setLabel("left", "Energy")
        self.image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        self.layer("points")
        self.layer("models")

        self.scale_container = QWidget()
        self.scale_container.setAutoFillBackground(True)  # the plot background
        self._scale_layout = QVBoxLayout(self.scale_container)
        self._scale_layout.setContentsMargins(0, 0, 0, 0)
        self._row.addWidget(self.scale_container)
        self._scale: ColorScale = make_scale(scale_style)
        self._install_scale(self._scale)

        self.apply_theme(self._colors)
        self.plot.vb.sigResized.connect(self._align_scale)
        self.plot.scene().sigMouseClicked.connect(self._on_click)

    # ------------------------------------------------------------------ colour scale
    @property
    def scale(self) -> ColorScale:
        """The colour scale in use."""
        return self._scale

    @property
    def hist(self) -> pg.HistogramLUTItem | None:
        """The HistogramLUTItem while the histogram style is active, else None."""
        return self._scale.hist if isinstance(self._scale, HistogramScale) else None

    def scale_style(self) -> str:
        return self._scale.style

    def set_scale_style(self, style: str) -> None:
        """Swap the colour scale ("histogram" or "bar"), keeping levels and colour map."""
        if style == self._scale.style:
            return
        new = make_scale(style)
        old = self._scale
        lo, hi = old.levels()
        old.levelsEdited.disconnect(self._on_scale_edited)
        old.detach()
        self._scale_layout.removeWidget(old.widget)
        old.widget.hide()
        old.widget.deleteLater()
        new.set_colormap(old.colormap())
        new.set_levels(lo, hi, auto_range=self._auto_levels)
        self._scale = new
        self._install_scale(new)

    def _install_scale(self, scale: ColorScale) -> None:
        scale.apply_theme(self._colors)
        scale.attach(self.image)
        scale.levelsEdited.connect(self._on_scale_edited)
        self._scale_layout.addWidget(scale.widget)
        self._align_scale()

    def set_scale_visible(self, visible: bool) -> None:
        """Show or hide the colour scale (exports follow)."""
        self.scale_container.setVisible(visible)

    def scale_visible(self) -> bool:
        return not self.scale_container.isHidden()

    def _align_scale(self, *_args) -> None:
        plot = self.plot.sceneBoundingRect()
        data = self.plot.vb.sceneBoundingRect()
        self._margins = self._scale.margins(plot, data, self.view.height())
        self._scale_layout.setContentsMargins(0, self._margins[0], 0, self._margins[1])

    def _on_scale_edited(self, lo: float, hi: float) -> None:
        if self._fmap is None:
            return
        self._auto_levels = False
        self.levelsEdited.emit(lo, hi)

    def levels(self) -> tuple[float, float]:
        return self._scale.levels()

    def set_levels(self, lo: float, hi: float) -> None:
        """Set the colour levels without emitting :attr:`levelsEdited`."""
        self._auto_levels = False
        self._scale.set_levels(lo, hi)
        self.image.setLevels((lo, hi))

    def colormap(self) -> str:
        return self._scale.colormap()

    def set_colormap(self, name: str) -> None:
        self._scale.set_colormap(name)
        self._cmap = name

    def apply_theme(self, colors: PlotColors) -> None:
        """Plot, scale and the well behind missing data (NaN) from *colors*."""
        super().apply_theme(colors)
        self.plot.vb.setBackgroundColor(colors.q("well"))
        palette = self.scale_container.palette()
        palette.setColor(QPalette.ColorRole.Window, colors.q("background"))
        self.scale_container.setPalette(palette)
        self._scale.apply_theme(colors)

    # ------------------------------------------------------------------ overlays
    def set_model_curves(self, field: np.ndarray, lines: np.ndarray | None) -> None:
        """Draw model energies ``lines[i, j]`` at ``field[j]``; None removes them."""
        layer = self.layer("models")
        if lines is None or len(lines) == 0:
            layer.clear()
            return
        pen = pg.mkPen((255, 255, 255, 210), width=1.5, style=Qt.PenStyle.DashLine)
        layer.set_curves([(field, line) for line in lines], pen)

    def model_curve_data(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """(field, energy) of the visible model curves."""
        return self.layer("models").curve_data()

    def set_points(
        self,
        current: tuple[np.ndarray, np.ndarray] | None,
        others: list[tuple[np.ndarray, np.ndarray, object]] | None = None,
    ) -> None:
        """Mark picked points: *current* curve as crosses, *others* as coloured circles."""
        layer = self.layer("points")
        layer.clear()
        if others:
            x = np.concatenate([np.asarray(b, dtype=float) for b, _, _ in others])
            y = np.concatenate([np.asarray(e, dtype=float) for _, e, _ in others])
            pens = [pg.mkPen(color, width=2) for b, _, color in others for _ in range(len(b))]
            layer.add_points(x, y, symbol="o", size=8, pen=pens, brush=None)
        if current is not None:
            pen = pg.mkPen("r", width=2)
            layer.add_points(current[0], current[1], symbol="x", size=11, pen=pen, brush=None)

    # ------------------------------------------------------------------ data
    def value_at(self, b: float, energy: float) -> float | None:
        """Map value drawn at (b, energy), or None outside the image."""
        fmap, cells = self._fmap, self._cells
        if fmap is None or cells is None:
            return None
        col, row = cells[0].sample_at(b), cells[1].sample_at(energy)
        if col is None or row is None:
            return None
        return float(fmap.values[row, col])

    def drawn_cells(self) -> tuple[AxisCells, AxisCells] | None:
        """How the map is drawn: the (field, energy) cells, or None without a map."""
        return self._cells

    def _cursor_value(self, x: float, y: float) -> float | None:
        return self.value_at(x, y)

    def _cursor_text(self, x: float, y: float) -> str:
        text = f"B = {x:.3f} T    E = {y:.3f} {self._unit}"
        value = self.value_at(x, y)
        return text if value is None else f"{text}    value = {value:.5g}"

    def set_map(
        self,
        fmap: FieldMap,
        levels: Range = None,
        cmap: str = "magma",
        x_range: Range = None,
        y_range: Range = None,
    ) -> None:
        self._show(fmap, levels, cmap)
        if x_range is None:
            x_range = (float(fmap.field.min()), float(fmap.field.max()))
        set_range(self.plot, x_range, y_range)

    def _show(self, fmap: FieldMap, levels: Range, cmap: str) -> None:
        self._fmap = fmap
        self._unit = str(fmap.unit)
        self.plot.setLabel("left", axis_label(fmap.unit))
        scale = self._scale
        if cmap != self._cmap:
            scale.set_colormap(cmap)
            self._cmap = cmap
        self._auto_levels = levels is None
        lo, hi = robust_levels(fmap.values) if levels is None else levels
        x, y = self._cells = map_cells(fmap.field, fmap.energy)
        values = fmap.values  # an uneven axis is drawn on a finer, even one (see axis_cells)
        if y.index is not None:
            values = values[y.index]
        if x.index is not None:
            values = values[:, x.index]
        with scale.quiet():  # the histogram re-reads the levels from the new image
            self.image.setImage(values, autoLevels=False, levels=(lo, hi))
            self.image.setRect(QRectF(x.start, y.start, x.span, y.span))
            scale.set_levels(lo, hi, auto_range=self._auto_levels)

    def clear_map(self) -> None:
        self._fmap = None
        self._cells = None
        self.image.clear()
        self.set_points(None)

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self.image.image is None:
            return
        pos = event.scenePos()
        if not self.plot.vb.sceneBoundingRect().contains(pos):
            return
        point = self.plot.vb.mapSceneToView(pos)
        self.pointClicked.emit(point.x(), point.y())

    # ------------------------------------------------------------------ export
    def _export_parts(self, height: float) -> list[tuple[float, PartPainter]]:
        parts = super()._export_parts(height)
        if not self.scale_visible():
            return parts
        scale = self._scale
        top, bottom = self._margins

        def paint(painter: QPainter, rect: QRectF, resolution: float) -> None:
            inner = rect.adjusted(0, top, 0, -bottom)
            if inner.height() > 0:
                scale.render(painter, inner, resolution)

        return [*parts, (float(scale.widget.width()), paint)]
