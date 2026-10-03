"""Colour scales shown beside a map: pyqtgraph's histogram or a slim colour bar."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager, suppress

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontInfo,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from mag_opt_detective.core import colormaps as core
from mag_opt_detective.gui.plots.base import paint_item
from mag_opt_detective.gui.plots.colormaps import colormap, lookup_table
from mag_opt_detective.gui.plots.colors import PlotColors

Levels = tuple[float, float]


class ColorScale(QObject):
    """Shows and edits the levels and colour map of an attached ImageItem.

    :attr:`levelsEdited` fires only when the user changes the levels, never for
    :meth:`set_levels`. Each scale lives in its own :attr:`widget`.
    """

    levelsEdited = Signal(float, float)
    style = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cmap = "grey"
        self._quiet = 0

    @property
    def widget(self) -> QWidget:
        raise NotImplementedError

    @contextmanager
    def quiet(self) -> Iterator[None]:
        """Treat level changes made inside the block as programmatic (no signal)."""
        self._quiet += 1
        try:
            yield
        finally:
            self._quiet -= 1

    def _user_edited(self, lo: float, hi: float) -> None:
        if not self._quiet:
            self.levelsEdited.emit(float(lo), float(hi))

    def colormap(self) -> str:
        return self._cmap

    def set_colormap(self, name: str) -> None:
        raise NotImplementedError

    def set_levels(self, lo: float, hi: float, auto_range: bool = False) -> None:
        """Set the levels without emitting; *auto_range*: they were taken from the data."""
        raise NotImplementedError

    def levels(self) -> Levels:
        raise NotImplementedError

    def image(self) -> pg.ImageItem | None:
        """The attached image, if any."""
        raise NotImplementedError

    def attach(self, image: pg.ImageItem) -> None:
        """Drive the levels and lookup table of *image*, keeping the current levels."""
        raise NotImplementedError

    def detach(self) -> None:
        """Stop driving the image."""
        raise NotImplementedError

    def apply_theme(self, colors: PlotColors) -> None:
        raise NotImplementedError

    def margins(self, plot: QRectF, data: QRectF, height: float) -> tuple[int, int]:
        """Top and bottom margins that align the scale with a plot of *height*.

        *plot* is the plot item with its axes and *data* the data area, in view pixels.
        """
        return round(plot.top()), max(0, round(height - plot.bottom()))

    def render(self, painter: QPainter, rect: QRectF, resolution: float = 1.0) -> None:
        """Paint the scale into *rect* for an exported image."""
        raise NotImplementedError


def _alpha(color: QColor, alpha: float) -> QColor:
    color = QColor(color)
    color.setAlphaF(alpha)
    return color


class HistogramScale(ColorScale):
    """pyqtgraph's HistogramLUTItem (histogram, level region and gradient)."""

    style = "histogram"
    WIDTH = 116

    def __init__(self, parent=None):
        super().__init__(parent)
        self._view = pg.GraphicsLayoutWidget()
        self._view.ci.setContentsMargins(0, 0, 9, 0)
        self._view.setFixedWidth(self.WIDTH)
        self.hist = pg.HistogramLUTItem()
        self._view.addItem(self.hist)
        self.hist.sigLevelChangeFinished.connect(self._on_level_change_finished)
        self.set_colormap(self._cmap)

    @property
    def widget(self) -> QWidget:
        return self._view

    def _on_level_change_finished(self, *_args) -> None:
        self._user_edited(*self.levels())

    def set_colormap(self, name: str) -> None:
        self._cmap = name
        gradient = self.hist.gradient
        gradient.setColorMap(colormap(name))
        gradient.showTicks(False)  # the stops come from the colour map, not from editing

    def set_levels(self, lo: float, hi: float, auto_range: bool = False) -> None:
        with self.quiet():
            self.hist.setLevels(lo, hi)
        if auto_range:
            self.hist.autoHistogramRange()
        else:
            pad = 0.1 * abs(hi - lo)
            self.hist.setHistogramRange(lo - pad, hi + pad)

    def levels(self) -> Levels:
        lo, hi = self.hist.getLevels()
        return float(lo), float(hi)

    def image(self) -> pg.ImageItem | None:
        return self.hist.imageItem()

    def attach(self, image: pg.ImageItem) -> None:
        self.detach()
        lo, hi = self.levels()
        with self.quiet():
            self.hist.setImageItem(image)  # also auto-levels from the image ...
            self.hist.setLevels(lo, hi)  # ... so put the levels back

    def detach(self) -> None:
        hist = self.hist
        image = hist.imageItem()
        if image is None:
            return
        # undo HistogramLUTItem.setImageItem: it connects sigImageChanged and installs
        # its own bound getLookupTable as the image's LUT
        with suppress(RuntimeError, TypeError):
            image.sigImageChanged.disconnect(hist.imageChanged)
        if getattr(image.lut, "__self__", None) is hist:
            trivial = hist.gradient.isLookupTrivial()
            image.setLookupTable(None if trivial else hist.gradient.getLookupTable(256))
        hist.imageItem = lambda: None  # same dead weakref as a fresh item

    def apply_theme(self, colors: PlotColors) -> None:
        """Background and axis, a grey histogram and accent level handles (as the inspector's
        histogram) instead of pyqtgraph's blue fill and olive lines."""
        self._view.setBackground(colors.q("background"))
        fg = colors.q("foreground")
        self.hist.axis.setPen(fg)
        self.hist.axis.setTextPen(fg)
        self.hist.plot.setPen(pg.mkPen(_alpha(fg, 0.55), width=1))
        self.hist.plot.setBrush(pg.mkBrush(_alpha(fg, 0.3)))
        accent = colors.q("accent")
        region = self.hist.region
        region.setBrush(pg.mkBrush(_alpha(accent, 0.12)))
        region.setHoverBrush(pg.mkBrush(_alpha(accent, 0.2)))
        for line in region.lines:
            line.setPen(pg.mkPen(accent, width=1.5))
            line.setHoverPen(pg.mkPen(accent, width=3))
        self.hist.update()  # the lines to the gradient take the handles' pen

    def render(self, painter: QPainter, rect: QRectF, resolution: float = 1.0) -> None:
        paint_item(painter, self._view.ci, rect, resolution)


def _nice_step(span: float, n: int) -> float:
    raw = span / max(1, n)
    power = 10.0 ** math.floor(math.log10(raw))
    f = raw / power
    return (1 if f < 1.5 else 2 if f < 3 else 5 if f < 7 else 10) * power


def _ticks(lo: float, hi: float, n: int) -> tuple[list[float], float]:
    if not hi > lo or not math.isfinite(hi - lo):
        return [], 1.0
    step = _nice_step(hi - lo, n)
    first = math.ceil(lo / step)
    values = [(first + k) * step for k in range(int((hi - lo) / step) + 2)]
    return [0.0 if abs(v) < step * 1e-9 else v for v in values if v <= hi + step * 1e-9], step


def _tick_label(value: float, step: float) -> str:
    if step < 1e-4 or abs(value) >= 1e5:
        return f"{value:.3g}"
    decimals = min(max(-math.floor(math.log10(step) + 1e-9), 0), 6)
    return f"{value:.{decimals}f}"


class ColorBar(QWidget):
    """Slim vertical colour bar with two draggable level handles.

    Drag a handle to move one level, drag elsewhere to shift both, and scroll to scale
    the span. The value axis stays frozen during a drag so the handle follows the mouse.
    """

    levelsChanging = Signal(float, float)  # live, while dragging or scrolling
    levelsChosen = Signal(float, float)  # once per edit: on release / after scrolling

    PAD = 8  # vertical room for the end labels and handles
    BAR = 12
    LEFT = 7
    GAP = 9  # bar to labels, clear of the handles
    GRAB = 7  # pixels around a handle that pick it up
    MARGIN = 0.25  # value axis = levels +- this fraction of their span
    WHEEL_DELAY_MS = 300

    def __init__(self, parent=None):
        super().__init__(parent)
        self._levels: Levels = (0.0, 1.0)
        self._stops = core.stops("grey")
        self._colors = PlotColors()
        self._frozen: Levels | None = None
        self._drag: tuple[str, float, Levels] | None = None
        self._wheel_timer = QTimer(self)
        self._wheel_timer.setSingleShot(True)
        self._wheel_timer.setInterval(self.WHEEL_DELAY_MS)
        self._wheel_timer.timeout.connect(self._finish_wheel)
        self.setMouseTracking(True)
        self.setMinimumHeight(4 * self.PAD)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setFixedWidth(self.preferred_width())

    def preferred_width(self) -> int:
        metrics = self.fontMetrics()
        label = max(metrics.horizontalAdvance(s) for s in ("-0.00000", "-1.23e-05"))
        return self.LEFT + self.BAR + self.GAP + label + 4

    # ------------------------------------------------------------------ state
    def levels(self) -> Levels:
        return self._levels

    def set_levels(self, lo: float, hi: float) -> None:
        """Programmatic levels; cancels a drag or a pending scroll edit."""
        self._wheel_timer.stop()
        self._drag = None
        self._frozen = None
        self._levels = (float(lo), float(hi))
        self.update()

    def set_colormap(self, name: str) -> None:
        self._stops = core.stops(name)
        self.update()

    def apply_theme(self, colors: PlotColors) -> None:
        self._colors = colors
        self.update()

    def value_range(self) -> Levels:
        """Values at the bottom and top of the bar."""
        if self._frozen is not None:
            return self._frozen
        lo, hi = self._levels
        pad = self.MARGIN * ((hi - lo) or 1.0)
        return lo - pad, hi + pad

    def bar_rect(self, rect: QRectF | None = None) -> QRectF:
        rect = QRectF(self.rect()) if rect is None else rect
        height = max(rect.height() - 2 * self.PAD, 1.0)
        return QRectF(rect.left() + self.LEFT, rect.top() + self.PAD, self.BAR, height)

    def y_for(self, value: float, rect: QRectF | None = None) -> float:
        bar = self.bar_rect(rect)
        vmin, vmax = self.value_range()
        return bar.bottom() - (value - vmin) / (vmax - vmin) * bar.height()

    def value_at(self, y: float, rect: QRectF | None = None) -> float:
        bar = self.bar_rect(rect)
        vmin, vmax = self.value_range()
        return vmin + (bar.bottom() - y) / bar.height() * (vmax - vmin)

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), self._colors.q("background"))
            self.paint_bar(painter, QRectF(self.rect()))
        finally:
            painter.end()

    def paint_bar(self, painter: QPainter, rect: QRectF) -> None:
        """Paint the bar, its value axis and the handles into *rect* (no background)."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont(self.font())
        font.setPixelSize(QFontInfo(self.font()).pixelSize())  # exports match the screen
        painter.setFont(font)
        bar = self.bar_rect(rect)
        lo, hi = self._levels
        y_lo, y_hi = self.y_for(lo, rect), self.y_for(hi, rect)
        if abs(y_lo - y_hi) < 1e-6:
            painter.fillRect(bar, QColor(*self._stops[-1][1]))
        else:
            gradient = QLinearGradient(QPointF(0, y_lo), QPointF(0, y_hi))  # pads outside
            for pos, rgb in self._stops:
                gradient.setColorAt(pos, QColor(*rgb))
            painter.fillRect(bar, QBrush(gradient))

        fg = self._colors.q("foreground")
        frame = QColor(fg)
        frame.setAlphaF(frame.alphaF() * 0.6)
        painter.setPen(QPen(frame, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(bar.adjusted(0.5, 0.5, -0.5, -0.5))

        vmin, vmax = self.value_range()
        values, step = _ticks(vmin, vmax, max(3, int(bar.height() / 60)))
        metrics = painter.fontMetrics()
        half = metrics.ascent() / 2 - 1
        for value in values:
            y = self.y_for(value, rect)
            if y < bar.top() + 2 or y > bar.bottom() - 2:
                continue
            painter.setPen(QPen(frame, 1))
            painter.drawLine(QPointF(bar.right(), y), QPointF(bar.right() + 4, y))
            painter.setPen(QPen(fg))
            painter.drawText(QPointF(bar.right() + self.GAP, y + half), _tick_label(value, step))

        painter.setPen(QPen(self._colors.q("accent"), 2))
        painter.setBrush(self._colors.q("background"))
        for value in (lo, hi):
            y = self.y_for(value, rect)
            handle = QRectF(bar.left() - 5, y - 3.5, bar.width() + 10, 7)
            painter.drawRoundedRect(handle, 2, 2)
        painter.restore()

    # ------------------------------------------------------------------ mouse
    def _hit(self, y: float) -> str:
        lo, hi = self._levels
        d_hi, d_lo = abs(y - self.y_for(hi)), abs(y - self.y_for(lo))
        if min(d_hi, d_lo) > self.GRAB:
            return "mid"
        return "hi" if d_hi <= d_lo else "lo"

    def _cursor_for(self, kind: str) -> Qt.CursorShape:
        if kind == "mid":
            return Qt.CursorShape.ClosedHandCursor if self._drag else Qt.CursorShape.OpenHandCursor
        return Qt.CursorShape.SizeVerCursor

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if self._wheel_timer.isActive():
            self._finish_wheel()
        y = event.position().y()
        kind = self._hit(y)
        self._frozen = self.value_range()
        self._drag = (kind, self.value_at(y), self._levels)
        self.setCursor(self._cursor_for(kind))
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        y = event.position().y()
        if self._drag is None:
            self.setCursor(self._cursor_for(self._hit(y)))
            return
        kind, start, (lo, hi) = self._drag
        value = self.value_at(y)
        eps = 0.02 * (hi - lo)
        if kind == "hi":
            levels = (lo, max(value, lo + eps))
        elif kind == "lo":
            levels = (min(value, hi - eps), hi)
        else:
            levels = (lo + value - start, hi + value - start)
        if levels != self._levels:
            self._levels = levels
            self.update()
            self.levelsChanging.emit(*levels)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag is None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        start_levels = self._drag[2]
        self._drag = None
        self._frozen = None
        self.setCursor(self._cursor_for(self._hit(event.position().y())))
        self.update()
        if self._levels != start_levels:
            self.levelsChosen.emit(*self._levels)
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if not delta or self._drag is not None:
            event.ignore()
            return
        factor = math.exp(-max(-120, min(120, delta)) * 0.002)  # up: narrower span
        lo, hi = self._levels
        centre, half = (lo + hi) / 2, (hi - lo) / 2 * factor
        self._levels = (centre - half, centre + half)
        self.update()
        self.levelsChanging.emit(*self._levels)
        self._wheel_timer.start()
        event.accept()

    def _finish_wheel(self) -> None:
        self._wheel_timer.stop()
        self.levelsChosen.emit(*self._levels)


class BarScale(ColorScale):
    """Our slim colour bar (:class:`ColorBar`)."""

    style = "bar"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: pg.ImageItem | None = None
        self._lut: np.ndarray = lookup_table(self._cmap)
        self.bar = ColorBar()
        self.bar.levelsChanging.connect(self._on_levels_changing)
        self.bar.levelsChosen.connect(self._user_edited)
        self.set_colormap(self._cmap)

    @property
    def widget(self) -> QWidget:
        return self.bar

    def _on_levels_changing(self, lo: float, hi: float) -> None:
        if self._image is not None:
            self._image.setLevels((lo, hi))

    def set_colormap(self, name: str) -> None:
        self._cmap = name
        self._lut = lookup_table(name)
        self.bar.set_colormap(name)
        if self._image is not None:
            self._image.setLookupTable(self._lut)

    def set_levels(self, lo: float, hi: float, auto_range: bool = False) -> None:
        self.bar.set_levels(lo, hi)
        if self._image is not None:
            self._image.setLevels((lo, hi))

    def levels(self) -> Levels:
        return self.bar.levels()

    def image(self) -> pg.ImageItem | None:
        return self._image

    def attach(self, image: pg.ImageItem) -> None:
        self.detach()
        self._image = image
        image.setLookupTable(self._lut)
        image.setLevels(self.levels())

    def detach(self) -> None:
        self._image = None

    def apply_theme(self, colors: PlotColors) -> None:
        self.bar.apply_theme(colors)

    def margins(self, plot: QRectF, data: QRectF, height: float) -> tuple[int, int]:
        pad = ColorBar.PAD
        return max(0, round(data.top()) - pad), max(0, round(height - data.bottom()) - pad)

    def render(self, painter: QPainter, rect: QRectF, resolution: float = 1.0) -> None:
        self.bar.paint_bar(painter, rect)


SCALES: dict[str, type[ColorScale]] = {"histogram": HistogramScale, "bar": BarScale}
SCALE_STYLES = tuple(SCALES)


def make_scale(style: str) -> ColorScale:
    """A new colour scale of *style* ("histogram" or "bar")."""
    try:
        return SCALES[style]()
    except KeyError:
        raise ValueError(f"unknown colour scale style {style!r}") from None
