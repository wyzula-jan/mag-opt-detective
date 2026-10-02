"""pyqtgraph views: field/energy colour map and stacked spectra."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph import exporters
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtSvg import QSvgGenerator

from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.core.units import axis_label

Range = tuple[float, float] | None

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".svg")


def robust_levels(values: np.ndarray) -> tuple[float, float]:
    """Colour levels from the 1st/99th percentile (ignores NaN and outliers)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return float(lo), float(hi)


def pixel_rect(field: np.ndarray, energy: np.ndarray) -> QRectF:
    """Image rectangle with pixel centres placed on the field/energy grid."""

    def span(axis: np.ndarray) -> tuple[float, float]:
        step = (axis[-1] - axis[0]) / (axis.size - 1) if axis.size > 1 else 1.0
        return axis[0] - step / 2, (axis[-1] - axis[0]) + step

    x0, width = span(field)
    y0, height = span(energy)
    return QRectF(x0, y0, width, height)


def _set_range(plot: pg.PlotItem, x_range: Range, y_range: Range) -> None:
    if x_range is None:
        plot.enableAutoRange(x=True)
    else:
        plot.setXRange(*x_range, padding=0)
    if y_range is None:
        plot.enableAutoRange(y=True)
    else:
        plot.setYRange(*y_range, padding=0)


class _CrosshairMixin:
    """Crosshair and a cursor read-out label on top of a plot."""

    def _init_crosshair(self, plot: pg.PlotItem, label: pg.LabelItem) -> None:
        self._xh_plot = plot
        self._xh_label = label
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((200, 200, 200, 150)))
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((200, 200, 200, 150)))
        plot.addItem(self._vline, ignoreBounds=True)
        plot.addItem(self._hline, ignoreBounds=True)
        self._proxy = pg.SignalProxy(
            plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )

    def _cursor_text(self, x: float, y: float) -> str:
        raise NotImplementedError

    def _on_mouse_moved(self, event) -> None:
        pos = event[0]
        if not self._xh_plot.vb.sceneBoundingRect().contains(pos):
            return
        point = self._xh_plot.vb.mapSceneToView(pos)
        self._vline.setPos(point.x())
        self._hline.setPos(point.y())
        self._xh_label.setText(self._cursor_text(point.x(), point.y()))

    def _export_svg(self, path: Path) -> None:
        # pyqtgraph 0.14's SVGExporter cannot parse the path data Qt 6.11 writes,
        # so render the scene through QSvgGenerator (lines stay vectors).
        source = self.ci.sceneBoundingRect()
        target = QRectF(0, 0, source.width(), source.height())
        generator = QSvgGenerator()
        generator.setFileName(str(path))
        generator.setSize(target.size().toSize())
        generator.setViewBox(target)
        generator.setTitle("mag-opt-detective plot")
        painter = QPainter(generator)
        try:
            self.scene().render(painter, target, source)
        finally:
            painter.end()

    def export_image(self, path: str | Path, scale: float = 2.0) -> None:
        """Save the whole view (plot, colour bar, labels) as PNG or SVG, without crosshair."""
        suffix = Path(path).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image type {suffix or '(none)'}: use .png or .svg")
        label = self._xh_label.text
        self._vline.hide()
        self._hline.hide()
        self._xh_label.setText("")
        try:
            if suffix == ".svg":
                self._export_svg(Path(path))
            else:
                exporter = exporters.ImageExporter(self.ci)
                width = int(self.ci.boundingRect().width() * scale)
                exporter.parameters()["width"] = max(width, 100)
                exporter.export(str(path))
        finally:
            self._vline.show()
            self._hline.show()
            self._xh_label.setText(label)


class ColorMapPlot(_CrosshairMixin, pg.GraphicsLayoutWidget):
    """Intensity as a function of field (x) and energy (y) with a histogram/LUT.

    Emits :attr:`pointClicked` ``(field, energy)`` on a left click inside the plot.
    """

    pointClicked = Signal(float, float)
    levelsEdited = Signal(float, float)  # the user dragged the histogram levels

    def __init__(self, parent=None):
        super().__init__(parent)
        self._unit = ""
        self._cmap = ""
        self._fmap: FieldMap | None = None
        self._rect = QRectF()
        self._updating = False
        self.label = self.addLabel("", row=0, col=0, colspan=2, justify="left")
        self.plot = self.addPlot(row=1, col=0)
        self.plot.setLabel("bottom", "Magnetic Field (T)")
        self.plot.setLabel("left", "Energy")
        self.image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.image)
        self.hist.sigLevelChangeFinished.connect(self._on_levels_changed)
        self.addItem(self.hist, row=1, col=1)

        self.current_points = pg.ScatterPlotItem(
            symbol="x", size=11, pen=pg.mkPen("r", width=2), brush=None
        )
        self.all_points = pg.ScatterPlotItem(symbol="o", size=8)
        self.plot.addItem(self.all_points)
        self.plot.addItem(self.current_points)

        self._model_curves: list[pg.PlotDataItem] = []

        self._init_crosshair(self.plot, self.label)
        self.plot.scene().sigMouseClicked.connect(self._on_click)

    def set_model_curves(self, field: np.ndarray, lines: np.ndarray | None) -> None:
        """Draw model energies ``lines[i, j]`` at ``field[j]``; None removes them."""
        n = 0 if lines is None else len(lines)
        pen = pg.mkPen((255, 255, 255, 210), width=1.5, style=Qt.PenStyle.DashLine)
        while len(self._model_curves) < n:
            curve = pg.PlotDataItem(pen=pen)
            self.plot.addItem(curve, ignoreBounds=True)
            self._model_curves.append(curve)
        for i, curve in enumerate(self._model_curves):
            if i < n:
                curve.setData(field, lines[i])
                curve.show()
            else:
                curve.hide()

    def model_curve_data(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """(field, energy) of the visible model curves."""
        return [c.getData() for c in self._model_curves if c.isVisible()]

    def value_at(self, b: float, energy: float) -> float | None:
        """Map value of the pixel under (b, energy), or None outside the image."""
        fmap = self._fmap
        if fmap is None or not self._rect.contains(QPointF(b, energy)):
            return None
        row = int(np.abs(fmap.energy - energy).argmin())
        col = int(np.abs(fmap.field - b).argmin())
        return float(fmap.values[row, col])

    def _cursor_text(self, x: float, y: float) -> str:
        text = f"B = {x:.3f} T    E = {y:.3f} {self._unit}"
        value = self.value_at(x, y)
        return text if value is None else f"{text}    value = {value:.5g}"

    def _on_levels_changed(self, *_args) -> None:
        if self._updating or self._fmap is None:
            return
        lo, hi = self.hist.getLevels()
        self.levelsEdited.emit(float(lo), float(hi))

    def set_map(
        self,
        fmap: FieldMap,
        levels: Range = None,
        cmap: str = "magma",
        x_range: Range = None,
        y_range: Range = None,
    ) -> None:
        self._updating = True
        try:
            self._show(fmap, levels, cmap)
        finally:
            self._updating = False
        if x_range is None:
            x_range = (float(fmap.field.min()), float(fmap.field.max()))
        _set_range(self.plot, x_range, y_range)

    def _show(self, fmap: FieldMap, levels: Range, cmap: str) -> None:
        self._fmap = fmap
        self._unit = str(fmap.unit)
        self.plot.setLabel("left", axis_label(fmap.unit))
        if cmap != self._cmap:
            self.hist.gradient.loadPreset(cmap)
            self._cmap = cmap
        auto = levels is None
        lo, hi = robust_levels(fmap.values) if auto else levels
        self.image.setImage(fmap.values, autoLevels=False, levels=(lo, hi))
        self._rect = pixel_rect(fmap.field, fmap.energy)
        self.image.setRect(self._rect)
        self.hist.setLevels(lo, hi)
        if auto:
            self.hist.autoHistogramRange()
        else:
            pad = 0.1 * abs(hi - lo)
            self.hist.setHistogramRange(lo - pad, hi + pad)

    def clear_map(self) -> None:
        self._fmap = None
        self.image.clear()
        self.set_points(None)

    def set_points(
        self,
        current: tuple[np.ndarray, np.ndarray] | None,
        others: list[tuple[np.ndarray, np.ndarray, object]] | None = None,
    ) -> None:
        """Mark picked points: *current* curve as crosses, *others* as coloured circles."""
        if current is None:
            self.current_points.clear()
        else:
            self.current_points.setData(x=current[0], y=current[1])
        self.all_points.clear()
        for b, e, color in others or []:
            self.all_points.addPoints(x=b, y=e, pen=pg.mkPen(color, width=2), brush=None)

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self.image.image is None:
            return
        pos = event.scenePos()
        if not self.plot.vb.sceneBoundingRect().contains(pos):
            return
        point = self.plot.vb.mapSceneToView(pos)
        self.pointClicked.emit(point.x(), point.y())


class StackedPlot(_CrosshairMixin, pg.GraphicsLayoutWidget):
    """All spectra of a map on top of each other, shifted by a constant offset."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._curves: list[pg.PlotDataItem] = []
        self.label = self.addLabel("", row=0, col=0, justify="left")
        self.plot = self.addPlot(row=1, col=0)
        self.plot.setLabel("bottom", "Energy")
        self.plot.setLabel("left", "Intensity (a.u.)")
        self.plot.setClipToView(True)
        self.plot.setDownsampling(auto=True, mode="peak")
        self._init_crosshair(self.plot, self.label)

    def _cursor_text(self, x: float, y: float) -> str:
        return f"E = {x:.3f}    I = {y:.4g}"

    def set_map(
        self, fmap: FieldMap, offset: float, y_range: Range = None, x_range: Range = None
    ) -> None:
        self.clear_map()
        self.plot.setLabel("bottom", axis_label(fmap.unit))
        n = fmap.field.size
        for j in range(n):
            curve = self.plot.plot(
                fmap.energy, fmap.values[:, j] + j * offset, pen=pg.intColor(j, hues=max(n, 1))
            )
            self._curves.append(curve)
        _set_range(self.plot, x_range, y_range)

    def clear_map(self) -> None:
        for curve in self._curves:
            self.plot.removeItem(curve)
        self._curves.clear()
