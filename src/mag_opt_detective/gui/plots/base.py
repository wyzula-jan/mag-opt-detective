"""Shared parts of the plot views: helpers, crosshair, theme and image export."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph import exporters
from PySide6.QtCore import QRectF, Signal
from PySide6.QtGui import QImage, QPainter, QPicture
from PySide6.QtSvg import QSvgGenerator
from PySide6.QtWidgets import QHBoxLayout, QWidget

from mag_opt_detective.core.spectra import cell_edges
from mag_opt_detective.gui.plots.colors import PlotColors
from mag_opt_detective.gui.plots.overlays import OverlayMixin

Range = tuple[float, float] | None
# paints one part of an exported image into a rectangle; the float is the pixel scale
PartPainter = Callable[[QPainter, QRectF, float], None]

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".svg")

UNIFORM_TOL = 1e-3  # of a step: an axis this close to an even grid is drawn as one
MAX_DRAWN_PIXELS = 8_000_000  # of a map on an uneven grid, which is drawn on a finer one


def robust_levels(values: np.ndarray) -> tuple[float, float]:
    """Colour levels from the 1st/99th percentile (ignores NaN and outliers)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return float(lo), float(hi)


@dataclass(frozen=True)
class AxisCells:
    """How one axis of a map is drawn: *size* equal cells of width *step* from *start*.

    Cell k shows sample ``index[k]``; with *index* None the axis is uniform and cell k shows
    sample k.
    """

    start: float
    step: float
    size: int
    index: np.ndarray | None = None

    @property
    def span(self) -> float:
        return self.size * self.step

    def sample_at(self, x: float) -> int | None:
        """The sample drawn at *x*, or None outside the cells."""
        if not math.isfinite(x):
            return None
        k = math.floor((x - self.start) / self.step)
        if not 0 <= k < self.size:
            return None
        return k if self.index is None else int(self.index[k])


def _is_uniform(axis: np.ndarray) -> bool:
    """Rising, and every sample within UNIFORM_TOL of a step of its place on an even grid."""
    step = (axis[-1] - axis[0]) / (axis.size - 1)
    if not step > 0:
        return False
    even = axis[0] + step * np.arange(axis.size)
    return bool(np.abs(axis - even).max() <= UNIFORM_TOL * step)


def axis_cells(axis: np.ndarray, max_cells: int = 1 << 30) -> AxisCells:
    """Cells that draw the samples of *axis* where they are (any order and spacing).

    A uniform axis gets one cell per sample, centred on it. Otherwise every sample covers
    the span of its :func:`~mag_opt_detective.core.spectra.cell_edges` (midpoints between
    neighbours, as in the journal figure), drawn with cells of half the smallest step, so
    that edges on a regular lattice (missing files, maps merged by field) fall on cell edges.
    The cells get wider when there would be more than *max_cells* (at least one per sample).
    """
    axis = np.asarray(axis, dtype=float)
    if axis.size == 1:
        return AxisCells(float(axis[0]) - 0.5, 1.0, 1)
    if _is_uniform(axis):
        step = float(axis[-1] - axis[0]) / (axis.size - 1)
        return AxisCells(float(axis[0]) - step / 2, step, axis.size)
    values, first = np.unique(axis, return_index=True)
    if values.size == 1:
        return AxisCells(float(values[0]) - 0.5, 1.0, 1, first[:1])
    edges = cell_edges(values)
    width = float(edges[-1] - edges[0])
    step = float(np.diff(values).min()) / 2
    size = math.ceil(width / step - 1e-6)
    limit = max(max_cells, 2 * values.size)
    if size > limit:
        size, step = limit, width / limit
    centres = edges[0] + (np.arange(size) + 0.5) * step
    k = np.clip(np.searchsorted(edges, centres, side="right") - 1, 0, values.size - 1)
    return AxisCells(float(edges[0]), step, size, first[k])


def map_cells(field: np.ndarray, energy: np.ndarray) -> tuple[AxisCells, AxisCells]:
    """(field, energy) cells of a drawn map, at most about MAX_DRAWN_PIXELS in all."""
    y = axis_cells(energy, MAX_DRAWN_PIXELS // (2 * max(field.size, 1)))
    x = axis_cells(field, MAX_DRAWN_PIXELS // y.size)
    return x, y


def pixel_rect(field: np.ndarray, energy: np.ndarray) -> QRectF:
    """Rectangle of the image of a map with the samples on their field and energy."""
    x, y = map_cells(np.asarray(field, dtype=float), np.asarray(energy, dtype=float))
    return QRectF(x.start, y.start, x.span, y.span)


def set_range(plot: pg.PlotItem, x_range: Range, y_range: Range) -> None:
    """Fixed view ranges, or auto-range for the axes given as None."""
    if x_range is None:
        plot.enableAutoRange(x=True)
    else:
        plot.setXRange(*x_range, padding=0)
    if y_range is None:
        plot.enableAutoRange(y=True)
    else:
        plot.setYRange(*y_range, padding=0)


def paint_item(
    painter: QPainter, item: pg.GraphicsObject, target: QRectF, resolution: float = 1.0
) -> None:
    """Render a pyqtgraph item and its children into *target*, as pyqtgraph's exporters do."""
    source = item.sceneBoundingRect()
    if source.isEmpty():
        return
    opts = {"antialias": True, "background": None, "painter": painter}
    exporter = exporters.Exporter(item)
    exporter.setExportMode(True, {**opts, "resolutionScale": resolution})
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        item.scene().render(painter, target, source)
    finally:
        exporter.setExportMode(False)


class PlotView(OverlayMixin, QWidget):
    """A pyqtgraph plot with a cursor read-out label, crosshair, overlay layers and theme.

    Emits :attr:`cursorMoved` ``(x, y, value)`` when the mouse moves over the data area.
    """

    cursorMoved = Signal(float, float, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._colors = PlotColors()
        self.view = pg.GraphicsLayoutWidget()
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(0)
        self._row.addWidget(self.view, stretch=1)
        self.label = self.view.addLabel("", row=0, col=0, justify="left")
        self.plot = self.view.addPlot(row=1, col=0)
        self._init_overlays()
        self._vline = pg.InfiniteLine(angle=90, movable=False)
        self._hline = pg.InfiniteLine(angle=0, movable=False)
        for line in (self._vline, self._hline):
            line.setZValue(1000)
            self.plot.addItem(line, ignoreBounds=True)
        self._proxy = pg.SignalProxy(
            self.plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )

    # ------------------------------------------------------------------ cursor
    def _cursor_text(self, x: float, y: float) -> str:
        raise NotImplementedError

    def _cursor_value(self, x: float, y: float) -> object:
        return None

    def crosshair(self) -> tuple[pg.InfiniteLine, pg.InfiniteLine]:
        """The cursor's vertical and horizontal lines (hidden while an image is exported)."""
        return self._vline, self._hline

    def _on_mouse_moved(self, event) -> None:
        pos = event[0]
        vb = self.plot.vb
        if not vb.sceneBoundingRect().contains(pos):
            return
        point = vb.mapSceneToView(pos)
        x, y = point.x(), point.y()
        self._vline.setPos(x)
        self._hline.setPos(y)
        self.label.setText(self._cursor_text(x, y))
        self.cursorMoved.emit(x, y, self._cursor_value(x, y))

    # ------------------------------------------------------------------ theme
    def colors(self) -> PlotColors:
        return self._colors

    def apply_theme(self, colors: PlotColors) -> None:
        """Backgrounds, axes, labels and crosshair from *colors*."""
        self._colors = colors
        self.view.setBackground(colors.q("background"))
        fg = colors.q("foreground")
        for name in ("left", "bottom", "right", "top"):
            axis = self.plot.getAxis(name)
            axis.setPen(fg)
            axis.setTextPen(fg)
        self.label.setAttr("color", fg)
        self.label.setText(self.label.text)
        pen = pg.mkPen(colors.q("crosshair"))
        self._vline.setPen(pen)
        self._hline.setPen(pen)

    # ------------------------------------------------------------------ export
    def _export_parts(self, height: float) -> list[tuple[float, PartPainter]]:
        """(width, painter) of the parts placed left to right in an exported image."""
        source = self.view.ci.sceneBoundingRect()

        def paint(painter: QPainter, rect: QRectF, resolution: float) -> None:
            paint_item(painter, self.view.ci, rect, resolution)

        return [(source.width(), paint)]

    @contextmanager
    def _exporting(self) -> Iterator[None]:
        text = self.label.text
        self._vline.hide()
        self._hline.hide()
        self.label.setText("")
        try:
            yield
        finally:
            self._vline.show()
            self._hline.show()
            self.label.setText(text)

    def _paint_parts(self, painter: QPainter, parts, width: float, height: float, scale: float):
        painter.fillRect(QRectF(0, 0, width, height), self._colors.q("background"))
        x = 0.0
        for part_width, paint in parts:
            paint(painter, QRectF(x, 0, part_width, height), scale)
            x += part_width

    def export_image(self, path: str | Path, scale: float = 2.0) -> None:
        """Save the view (plot, colour scale, labels) as PNG or SVG, without crosshair."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image type {suffix or '(none)'}: use .png or .svg")
        with self._exporting():
            height = self.view.ci.sceneBoundingRect().height()
            parts = self._export_parts(height)
            width = sum(w for w, _ in parts)
            if suffix == ".svg":
                self._export_svg(path, parts, width, height)
            else:
                self._export_raster(path, parts, width, height, scale)

    def _export_svg(self, path: Path, parts, width: float, height: float) -> None:
        # pyqtgraph 0.14's SVGExporter cannot parse the path data Qt 6.11 writes,
        # so render through QSvgGenerator (lines stay vectors).
        target = QRectF(0, 0, width, height)
        generator = QSvgGenerator(QSvgGenerator.SvgVersion.Svg11)  # 1.1 keeps clipping
        generator.setFileName(str(path))
        generator.setSize(target.size().toSize())
        generator.setViewBox(target)
        generator.setTitle("mag-opt-detective plot")
        # pyqtgraph caches axes as QPictures, which QPainter rescales to the device DPI
        generator.setResolution(QPicture().logicalDpiX())
        painter = QPainter(generator)
        try:
            self._paint_parts(painter, parts, width, height, 1.0)
        finally:
            painter.end()

    def _export_raster(self, path: Path, parts, width: float, height: float, scale: float):
        pixels = max(int(width * scale), 100)
        factor = pixels / width if width > 0 else 1.0
        image = QImage(pixels, max(int(height * factor), 1), QImage.Format.Format_ARGB32)
        image.fill(self._colors.q("background"))
        painter = QPainter(image)
        try:
            painter.scale(factor, factor)
            self._paint_parts(painter, parts, width, height, factor)
        finally:
            painter.end()
        if not image.save(str(path)):
            raise OSError(f"could not write {path}")
