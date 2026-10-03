"""The baseline region on a plot: a band along the energy axis, dragged by its edges or body.

:class:`BaselineRegion` is a :class:`pyqtgraph.LinearRegionItem` that never turns inside out:
its edges stay at least a minimum width apart and inside the limits (the data, or the energy
window), and dragging the band keeps its width. :attr:`~BaselineRegion.edited` and
:attr:`~BaselineRegion.editFinished` report the user's drags, rounded to a power of ten below a
screen pixel; :meth:`~BaselineRegion.set_region` (the typed fields) emits nothing. A grip in the
middle of each edge, a stronger fill and edge under the mouse and the cursors show what can be
dragged; :meth:`~BaselineRegion.setMovable` turns all that off (the band is then only shown).
Over a colour map the edges have a dark shadow, as the processing guides, so they read on any
colours; over spectra a halo in the background colour keeps them clear of the traces, and the
fill (a separate item, :attr:`~BaselineRegion.fill`) lies below the traces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pyqtgraph as pg
import shiboken6
from PySide6.QtCore import QLineF, QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsItem

HORIZONTAL, VERTICAL = "horizontal", "vertical"  # energy on the y axis (map) or the x axis
LABEL = "Baseline region"
GRIP = 9.0  # px: diameter of the grip in the middle of each edge
HALO = 1.6  # px: the shadow around a grip
LABEL_MARGIN = 3.0  # px: room a label inside a band needs on each side


@dataclass(frozen=True)
class RegionStyle:
    """Colours of a region: its edges and fill (normal and under the mouse), the label, and a
    pen drawn below the edges and grips (a shadow or halo; None for none)."""

    pen: QPen
    hover_pen: QPen
    brush: QBrush
    hover_brush: QBrush
    label: QColor
    label_fill: QBrush
    shadow: QPen | None = None


def data_style() -> RegionStyle:
    """White over a colour map, with a dark shadow (data colours, as the processing guides)."""
    return RegionStyle(
        pen=pg.mkPen((255, 255, 255, 215), width=1.4),
        hover_pen=pg.mkPen((255, 255, 255, 255), width=2.4),
        brush=pg.mkBrush(255, 255, 255, 40),
        hover_brush=pg.mkBrush(255, 255, 255, 70),
        label=QColor(255, 255, 255, 240),
        label_fill=pg.mkBrush(20, 16, 24, 140),
        shadow=pg.mkPen((0, 0, 0, 110), width=3.4),
    )


def theme_style(accent: QColor, background: QColor) -> RegionStyle:
    """The theme's accent over a plot's *background* (stacked spectra)."""

    def tint(alpha: float) -> QColor:
        color = QColor(accent)
        color.setAlphaF(alpha)
        return color

    fill = QColor(background)
    fill.setAlphaF(0.7)
    return RegionStyle(
        pen=pg.mkPen(tint(0.9), width=1.4),
        hover_pen=pg.mkPen(accent, width=2.4),
        brush=pg.mkBrush(tint(0.1)),
        hover_brush=pg.mkBrush(tint(0.2)),
        label=QColor(accent),
        label_fill=pg.mkBrush(fill),
        shadow=pg.mkPen(background, width=4.0),
    )


class BaselineRegion(pg.LinearRegionItem):
    """A band over an energy range (*orientation* HORIZONTAL: energy on the y axis) that the
    user drags by its edges or body, labelled *label*; see the module docstring. Put it on a
    plot with :meth:`add_to`."""

    edited = Signal(float, float)  # the user drags the region: (lo, hi)
    editFinished = Signal(float, float)  # the user let go of it

    def __init__(self, orientation: str, style: RegionStyle, label: str = LABEL):
        super().__init__(values=(0.0, 1.0), orientation=orientation, swapMode=None)
        self._horizontal = orientation == HORIZONTAL
        self._limits: tuple[float | None, float | None] = (None, None)
        self._min_width = 0.0
        self._quiet = False  # set_region: not the user
        self._placing = False
        self._drag: tuple[float, float, float] | None = None  # region and press of a band drag
        self.fill = pg.LinearRegionItem(orientation=orientation, movable=False, pen=pg.mkPen(None))
        for line in self.fill.lines:
            line.hide()
        for line in self.lines:
            line.addMarker("o", position=0.5, size=GRIP)  # the grip (it also widens the hit area)
            line.sigPositionChanged.connect(self._keep_apart)
        self._set_cursors(True)
        # above the upper edge at the left of the map; right of the lower edge at the top of
        # the stacked plot (the label follows its edge and stays in view)
        edge, position, anchor = (
            (self.lines[1], 0.01, (0, 1)) if self._horizontal else (self.lines[0], 0.98, (0, 0))
        )
        self.label = pg.InfLineLabel(edge, label, position=position, anchors=[anchor, anchor])
        self._style = style
        self.set_style(style)
        self.sigRegionChanged.connect(self._follow)
        self.sigRegionChanged.connect(self._on_changed)
        self.sigRegionChangeFinished.connect(self._on_finished)

    def add_to(self, plot: pg.PlotItem, z: float, fill_z: float) -> None:
        """Put the band on *plot* (outside its auto-range): the edges, grips and label at *z*,
        the fill at *fill_z* (e.g. below spectra, so they keep their colours)."""
        self.setZValue(z)
        self.fill.setZValue(fill_z)
        plot.addItem(self.fill, ignoreBounds=True)
        plot.addItem(self, ignoreBounds=True)

    # ------------------------------------------------------------------ appearance
    def set_style(self, style: RegionStyle) -> None:
        self._style = style
        for line in self.lines:
            line.setPen(style.pen)
            line.setHoverPen(style.hover_pen)
        self.setBrush(pg.mkBrush(None))  # the fill item paints the band
        self.setHoverBrush(pg.mkBrush(None))
        self.fill.setBrush(style.brush)
        self.fill.setHoverBrush(style.hover_brush)
        self._show_hover()
        self.label.setColor(style.label)
        self.label.fill = style.label_fill
        self.label.update()
        self.update()

    def setMouseHover(self, hover: bool) -> None:
        super().setMouseHover(hover)
        self._show_hover()

    def _show_hover(self) -> None:
        fill = self.fill
        fill.currentBrush = fill.hoverBrush if self.mouseHovering else fill.brush
        fill.update()

    def setMovable(self, m: bool = True) -> None:
        """Let the user drag the band and its edges, or only show it (no hover, no cursors:
        drags go to the plot, e.g. to zoom)."""
        super().setMovable(m)
        if not hasattr(self, "_horizontal"):  # called while pyqtgraph builds the item
            return
        self._set_cursors(m)
        if not m:
            self.moving = False
            for item in (self, *self.lines):
                item.setMouseHover(False)

    def _set_cursors(self, movable: bool) -> None:
        shape = Qt.CursorShape
        edge = shape.SizeVerCursor if self._horizontal else shape.SizeHorCursor
        cursors = ((self.lines[0], edge), (self.lines[1], edge), (self, shape.SizeAllCursor))
        for item, cursor in cursors:
            if movable:
                item.setCursor(cursor)
            else:
                item.unsetCursor()

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemVisibleHasChanged:
            fill = getattr(self, "fill", None)
            if fill is not None and shiboken6.isValid(fill):
                fill.setVisible(bool(value))
        return result

    def viewTransformChanged(self) -> None:
        super().viewTransformChanged()
        self._fit_label()

    def _follow(self, _item=None) -> None:
        self.fill.setRegion(self.getRegion())
        self._fit_label()

    def _fit_label(self) -> None:
        """A label inside a vertical band shows only while it fits in it."""
        label, view = getattr(self, "label", None), self.getViewBox()
        if self._horizontal or label is None or not isinstance(view, pg.ViewBox):
            return
        pixel = view.viewPixelSize()[0]
        lo, hi = self.getRegion()
        width = (hi - lo) / pixel if math.isfinite(pixel) and pixel > 0 else math.inf
        fits = width >= label.boundingRect().width() + 2 * LABEL_MARGIN
        if fits != label.isVisibleTo(self):
            label.setVisible(fits)

    def paint(self, p, *args) -> None:
        super().paint(p, *args)  # the fill
        shadow = self._style.shadow
        if shadow is None:
            return
        rect = self.boundingRect()
        edges = [
            QLineF(rect.left(), v, rect.right(), v)
            if self._horizontal
            else QLineF(v, rect.top(), v, rect.bottom())
            for v in self.getRegion()
        ]
        p.setPen(shadow)
        for edge in edges:
            p.drawLine(edge)
        transform = p.transform()  # the grips' halos, in pixels
        p.resetTransform()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(shadow.color())
        radius = GRIP / 2 + HALO
        for edge in edges:
            p.drawEllipse(transform.map(edge.center()), radius, radius)

    # ------------------------------------------------------------------ limits and region
    def set_limits(self, lo: float | None, hi: float | None, min_width: float = 0.0) -> None:
        """Keep the edges inside [*lo*, *hi*] (None: open) and *min_width* apart."""
        self._limits = (lo, hi)
        self._min_width = max(float(min_width), 0.0)
        self.set_region(*self.getRegion())

    def limits(self) -> tuple[float | None, float | None]:
        return self._limits

    def region(self) -> tuple[float, float]:
        return self.getRegion()

    def set_region(self, lo: float, hi: float) -> None:
        """Show the region (lo, hi), kept inside the limits, without reporting an edit."""
        self._quiet = True
        try:
            self._place(float(lo), float(hi))
        finally:
            self._quiet = False

    def setRegion(self, rgn) -> None:
        """Move the edges to *rgn* as a drag would: :attr:`edited` and :attr:`editFinished`
        report it (see :meth:`set_region` for a move that reports nothing)."""
        if tuple(rgn) == self.getRegion():
            return
        self._place(float(rgn[0]), float(rgn[1]))
        self.sigRegionChangeFinished.emit(self)

    def is_dragging(self) -> bool:
        """The user is dragging the band or one of its edges."""
        return bool(self.moving or any(line.moving for line in self.lines))

    def step(self) -> float:
        """Dragged values are rounded to this: the power of ten below a screen pixel (0, no
        rounding, while the plot is not on screen)."""
        view, widget = self.getViewBox(), self.getViewWidget()
        if not isinstance(view, pg.ViewBox) or widget is None or not widget.isVisible():
            return 0.0
        pixel = view.viewPixelSize()[1 if self._horizontal else 0]
        if not (math.isfinite(pixel) and pixel > 0):
            return 0.0
        return 10.0 ** math.floor(math.log10(pixel))

    def _place(self, lo: float, hi: float) -> None:
        """Put the edges at *lo* and *hi* (then kept apart and inside the limits)."""
        old = self.getRegion()
        self.blockLineSignal = True
        self._placing = True
        try:
            for line, value in zip(self.lines, (lo, hi), strict=True):
                line.setBounds((None, None))
                line.setValue(value)
            self._apply_bounds()
        finally:
            self._placing = False
            self.blockLineSignal = False
        if self.getRegion() != old:
            self.prepareGeometryChange()
            self.sigRegionChanged.emit(self)

    def _keep_apart(self, *_args) -> None:
        """An edge moved (dragged): the bounds of both follow."""
        if self._placing:
            return
        self._placing = True
        try:
            self._apply_bounds()
        finally:
            self._placing = False

    def _apply_bounds(self) -> None:
        """Each edge may move up to the other one (less the minimum width) and to the limits."""
        lower, upper = self.lines
        lo, hi = self._limits
        width = self._min_width
        top = upper.value() - width
        lower.setBounds((lo if lo is None else min(lo, top), top))
        bottom = lower.value() + width
        upper.setBounds((bottom, hi if hi is None else max(hi, bottom)))

    # ------------------------------------------------------------------ the user's drags
    def _rounded(self) -> tuple[float, float]:
        """The region rounded to :meth:`step`, unless that would break a limit or the width."""
        lo, hi = self.getRegion()
        step = self.step()
        if step <= 0:
            return lo, hi
        digits = -round(math.log10(step))
        a, b = round(lo, digits), round(hi, digits)
        low, high = self._limits
        a = a if low is None else max(a, low)
        b = b if high is None else min(b, high)
        return (a, b) if b - a >= self._min_width and a < b else (lo, hi)

    def _on_changed(self, _item) -> None:
        if self._quiet:
            return
        # what is dragged stays highlighted (pyqtgraph drops the hover state during a drag)
        if self.moving:
            self.setMouseHover(True)
        for line in self.lines:
            if line.moving or self.moving:
                line.setMouseHover(True)
        self.edited.emit(*self._rounded())

    def _on_finished(self, _item) -> None:
        if self._quiet:
            return
        for item in (self, *self.lines):  # the next mouse move highlights what is under it
            item.setMouseHover(False)
        self.editFinished.emit(*self._rounded())

    def _axis(self, point: QPointF) -> float:
        return point.y() if self._horizontal else point.x()

    def mouseDragEvent(self, ev) -> None:
        """Drag the band: both edges move together, keeping the width, until one meets a limit."""
        if not self.movable or ev.button() != Qt.MouseButton.LeftButton:
            return
        ev.accept()
        if ev.isStart():
            self._drag = (*self.getRegion(), self._axis(ev.buttonDownPos()))
            self.moving = True
        if not self.moving or self._drag is None:
            return
        lo, hi, start = self._drag
        shift = self._axis(ev.pos()) - start
        low, high = self._limits
        if high is not None:
            shift = min(shift, high - hi)
        if low is not None:
            shift = max(shift, low - lo)
        self._place(lo + shift, hi + shift)
        if ev.isFinish():
            self.moving = False
            self._drag = None
            self.sigRegionChangeFinished.emit(self)

    def mouseClickEvent(self, ev) -> None:
        """A right click during a band drag puts the band back."""
        if self.moving and self._drag is not None and ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            lo, hi, _start = self._drag
            self.moving = False
            self._drag = None
            self._place(lo, hi)
            self.sigRegionChangeFinished.emit(self)
