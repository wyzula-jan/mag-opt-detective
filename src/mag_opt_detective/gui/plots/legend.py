"""A legend inside a plot: a small panel listing what the plot draws, which the user can drag.

Each :class:`LegendEntry` is a row with a sample (a marker or a short line, drawn with the pens
of the plot) and a label. The panel lives in the data area of a plot but keeps its place on
screen while the view pans, zooms or is resized: its place is kept as fractions of the room
left of it and above it, so a legend in a corner stays in that corner. It is drawn above the
overlay layers and below the auto-pick region (z 20). A press on it moves it and reaches no
plot tool (it takes the mouse from pyqtgraph), except where a region drawn above it is.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, QSizeF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPalette,
    QPen,
    QTransform,
)
from PySide6.QtWidgets import QApplication, QGraphicsItem, QGraphicsObject

Z = 19.5  # above the overlay layers (z 10, 11, ...), below the auto-pick region (z 20)
MARGIN = 8.0  # px between the legend and the edges of the data area
PAD_X, PAD_Y = 8.0, 5.0
SAMPLE = 22.0  # width of a line sample (markers are centred in it)
GAP = 6.0  # between a sample and its label
GROUP_GAP = 7.0  # between groups of entries, with a hairline in the middle
FONT_PX = 11
MAX_TEXT = 220.0  # px of a label (longer ones are elided)
RADIUS = 6.0


@dataclass(frozen=True, eq=False)
class LegendEntry:
    """A row of the legend.

    *pens* draw the sample in turn (an outline first); a *marker* is a circle of *size* px,
    filled with *brush*, otherwise the sample is a short line. *detail* follows the label in a
    muted colour; *emphasis* sets the label bold (the current curve). A new *group* starts after
    a hairline.
    """

    label: str
    pens: tuple[QPen, ...]
    brush: QBrush | QColor | None = None
    marker: bool = False
    size: float = 9.0
    detail: str = ""
    emphasis: bool = False
    group: str = ""

    @property
    def text(self) -> str:
        """The row as read: the label and its detail."""
        return f"{self.label} · {self.detail}" if self.detail else self.label


@dataclass(frozen=True)
class LegendColors:
    """The panel (translucent), its hairline border, the text and the detail text."""

    panel: QColor
    border: QColor
    text: QColor
    muted: QColor


def _part_of_roi(item: QGraphicsItem) -> bool:
    """*item* is an ROI (e.g. the auto-pick region) or one of its handles or edges."""
    while item is not None:
        if isinstance(item, pg.ROI):
            return True
        item = item.parentItem()
    return False


class PlotLegend(QGraphicsObject):
    """A draggable legend in the data area of *plot* (a ``pg.PlotItem``).

    It shows while it is switched on (:meth:`set_shown`) and has entries (:meth:`set_entries`).
    :attr:`moved` fires when the user has dragged it, :attr:`shownChanged` when it is switched
    on or off. Bound to the settings, it keeps ``{"shown", "x", "y"}`` (x, y: its place as
    fractions of the free room, 0 = left / top, 1 = right / bottom).
    """

    moved = Signal()
    shownChanged = Signal(bool)

    def __init__(self, plot: pg.PlotItem):
        super().__init__()
        self._vb: pg.ViewBox = plot.vb
        self._entries: list[LegendEntry] = []
        self._shown = False
        self._fx, self._fy = 0.0, 0.0  # top left
        self._size = (0.0, 0.0)
        self._row = 0.0  # height of a row
        self._rows: list[tuple[float, LegendEntry]] = []  # y of each row's centre
        self._breaks: list[float] = []  # y of the hairlines between groups
        self._drag: tuple[QPointF, QPointF] | None = None  # press (view box) and top left then
        palette, role = QApplication.palette(), QPalette.ColorRole
        roles = (role.Base, role.Mid, role.Text, role.PlaceholderText)  # until set_colors()
        self._colors = LegendColors(*(palette.color(r) for r in roles))
        self._font = QFont(QApplication.font())
        self._font.setPixelSize(FONT_PX)
        self._bold = QFont(self._font)
        self._bold.setWeight(QFont.Weight.DemiBold)
        self.setZValue(Z)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Drag to move the legend")
        self.setParentItem(self._vb.childGroup)  # drawn with the data, never in its bounds
        self._vb.sigTransformChanged.connect(self.place)
        self._vb.sigResized.connect(self.place)
        self._sync()

    # ------------------------------------------------------------------ content
    def entries(self) -> list[LegendEntry]:
        return list(self._entries)

    def texts(self) -> list[str]:
        """The rows as read, top first."""
        return [entry.text for entry in self._entries]

    def set_entries(self, entries: Sequence[LegendEntry]) -> None:
        """Show *entries*, top first (the legend hides while there are none)."""
        self._entries = list(entries)
        self._layout()
        self._sync()

    def set_colors(self, colors: LegendColors) -> None:
        self._colors = colors
        self.update()

    def colors(self) -> LegendColors:
        return self._colors

    # ------------------------------------------------------------------ shown and place
    def is_shown(self) -> bool:
        """Whether the legend is switched on (it is visible only with entries)."""
        return self._shown

    def set_shown(self, shown: bool) -> None:
        shown = bool(shown)
        if shown != self._shown:
            self._shown = shown
            self._sync()
            self.shownChanged.emit(shown)

    def position(self) -> tuple[float, float]:
        """Its place as fractions of the free room (0, 0: top left; 1, 1: bottom right)."""
        return self._fx, self._fy

    def set_position(self, x: float, y: float) -> None:
        self._fx, self._fy = (min(max(float(v), 0.0), 1.0) for v in (x, y))
        self.place()

    def rect_in_view(self) -> QRectF:
        """Where the legend is drawn, in pixels of the view box."""
        return QRectF(self._top_left(), QSizeF(*self._size))

    def _room(self) -> tuple[QRectF, float, float]:
        """The data area within the margins, and the room left for the legend in x and y."""
        area = self._vb.rect().adjusted(MARGIN, MARGIN, -MARGIN, -MARGIN)
        width, height = self._size
        return area, max(area.width() - width, 0.0), max(area.height() - height, 0.0)

    def _top_left(self) -> QPointF:
        area, free_x, free_y = self._room()
        return QPointF(area.left() + self._fx * free_x, area.top() + self._fy * free_y)

    def place(self, *_args) -> None:
        """Put the legend at its place in the view (after a pan, zoom, resize or new size)."""
        group = self._vb.childTransform()  # data -> view box pixels, brought up to date
        sx, sy = group.m11(), group.m22()
        if not (sx and sy and math.isfinite(sx) and math.isfinite(sy)):
            return  # no range yet
        # one unit of the legend is one pixel, upright, whatever the scale of the data
        self.setTransform(QTransform.fromScale(1.0 / sx, 1.0 / sy))
        self.setPos(self._vb.mapToView(self._top_left()))

    def _sync(self) -> None:
        self.setVisible(self._shown and bool(self._entries))
        self.place()

    # ------------------------------------------------------------------ settings protocol
    def settings_value(self) -> str:
        return json.dumps({"shown": self._shown, "x": self._fx, "y": self._fy})

    def set_settings_value(self, value) -> bool:
        try:
            data = json.loads(value) if isinstance(value, str) else value
        except ValueError:
            return False
        if not isinstance(data, dict) or not isinstance(data.get("shown"), bool):
            return False
        place = [data.get("x"), data.get("y")]
        if not all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
            for v in place
        ):
            return False
        self.set_position(*place)
        self.set_shown(data["shown"])
        return True

    # ------------------------------------------------------------------ drawing
    def _layout(self) -> None:
        metrics, bold = QFontMetricsF(self._font), QFontMetricsF(self._bold)
        sizes = [entry.size for entry in self._entries]
        row = self._row = max(metrics.height(), *sizes) + 4.0 if sizes else 0.0
        text_width, y = 0.0, PAD_Y
        self._rows, self._breaks = [], []
        group = self._entries[0].group if self._entries else ""
        for entry in self._entries:
            if entry.group != group:
                self._breaks.append(y + GROUP_GAP / 2)
                y += GROUP_GAP
                group = entry.group
            self._rows.append((y + row / 2, entry))
            y += row
            width = (bold if entry.emphasis else metrics).horizontalAdvance(entry.label)
            if entry.detail:
                width += metrics.horizontalAdvance(f" · {entry.detail}")
            text_width = max(text_width, min(width, MAX_TEXT))
        self.prepareGeometryChange()
        if self._entries:
            self._size = (PAD_X + SAMPLE + GAP + math.ceil(text_width) + PAD_X, y + PAD_Y)
        else:
            self._size = (0.0, 0.0)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, *self._size)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if not self._rows:
            return
        colors = self._colors
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self._size
        painter.setPen(QPen(colors.border, 1.0))
        painter.setBrush(colors.panel)
        painter.drawRoundedRect(QRectF(0.5, 0.5, width - 1.0, height - 1.0), RADIUS, RADIUS)
        painter.setPen(QPen(colors.border, 1.0))
        for y in self._breaks:
            painter.drawLine(QPointF(PAD_X, y), QPointF(width - PAD_X, y))
        x_text = PAD_X + SAMPLE + GAP
        room = width - PAD_X - x_text
        flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        normal, bold = QFontMetricsF(self._font), QFontMetricsF(self._bold)
        for y, entry in self._rows:
            self._paint_sample(painter, entry, QPointF(PAD_X + SAMPLE / 2, y))
            metrics = bold if entry.emphasis else normal
            box = QRectF(x_text, y - self._row / 2, room, self._row)
            label = metrics.elidedText(entry.label, Qt.TextElideMode.ElideRight, room)
            painter.setFont(self._bold if entry.emphasis else self._font)
            painter.setPen(colors.text)
            painter.drawText(box, flags, label)
            if entry.detail:
                used = metrics.horizontalAdvance(label)
                detail = normal.elidedText(
                    f" · {entry.detail}", Qt.TextElideMode.ElideRight, max(room - used, 0.0)
                )
                painter.setFont(self._font)
                painter.setPen(colors.muted)
                painter.drawText(box.adjusted(used, 0, 0, 0), flags, detail)

    @staticmethod
    def _paint_sample(painter: QPainter, entry: LegendEntry, centre: QPointF) -> None:
        if entry.marker:
            radius = entry.size / 2
            painter.setBrush(entry.brush if entry.brush is not None else Qt.BrushStyle.NoBrush)
            for pen in entry.pens:
                painter.setPen(pen)
                painter.drawEllipse(centre, radius, radius)
            return
        half = QPointF(SAMPLE / 2, 0.0)
        for pen in entry.pens:
            painter.setPen(pen)
            painter.drawLine(centre - half, centre + half)

    # ------------------------------------------------------------------ dragging
    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._under_region(event.scenePos()):
            event.ignore()  # the region above the legend takes it (through pyqtgraph)
            return
        self._drag = (self._vb.mapFromScene(event.scenePos()), self._top_left())
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None:
            return
        press, top_left = self._drag
        moved = top_left + self._vb.mapFromScene(event.scenePos()) - press
        area, free_x, free_y = self._room()
        x = (moved.x() - area.left()) / free_x if free_x else 0.0
        y = (moved.y() - area.top()) / free_y if free_y else 0.0
        self.set_position(x, y)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.moved.emit()

    def _under_region(self, scene_pos: QPointF) -> bool:
        """An ROI lies above the legend at *scene_pos*."""
        for item in self.scene().items(scene_pos):  # topmost first
            if item is self:
                return False
            if _part_of_roi(item):
                return True
        return False
