"""Picked points on the plots: the curve colours, the markers and the hint shown while picking.

Curve *i* of ``controller.curve_names()`` gets ``CURVE_COLORS[i % 6]`` everywhere: on the map,
on the stacked plot and in the chips of the Points panel. The current curve is drawn filled
and the others as open rings; a dark outline keeps every colour readable on all colour maps
and on the light and dark plot backgrounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from mag_opt_detective.core.units import from_cm1
from mag_opt_detective.gui.plots import OverlayLayer, StackedPlot
from mag_opt_detective.gui.theme import current_tokens

CURVE_COLORS = ("#ffffff", "#5ad1ff", "#a6e35f", "#ffd166", "#ff86d9", "#ff9b6b")
OUTLINE = QColor(10, 6, 14, 217)
SHOW_CURRENT, SHOW_ALL, SHOW_NONE = "current", "all", "hidden"
SHOW_MODES = (  # value, label, tooltip
    (SHOW_CURRENT, "Current", "Markers of the current curve only"),
    (SHOW_ALL, "All", "Markers of all curves (the current one filled)"),
    (SHOW_NONE, "Hidden", "No markers"),
)
MAP_SIZE, STACKED_SIZE = 9.0, 8.0  # marker diameters in pixels


def curve_color(index: int) -> QColor:
    """Colour of the curve at *index* in ``controller.curve_names()``."""
    return QColor(CURVE_COLORS[index % len(CURVE_COLORS)])


@dataclass(frozen=True, eq=False)
class MarkerSet:
    """Markers of one curve in plot coordinates."""

    x: np.ndarray
    y: np.ndarray
    color: QColor
    current: bool


def shown_curves(controller, mode: str) -> list[tuple[str, QColor, bool]]:
    """``(name, colour, current)`` of the curves with points that *mode* shows, current last."""
    table = controller.points
    if mode == SHOW_NONE or table is None:
        return []
    shown = [
        (name, curve_color(i), name == controller.curve)
        for i, name in enumerate(controller.curve_names())
        if name in table.names and (mode == SHOW_ALL or name == controller.curve)
    ]
    return sorted(shown, key=lambda item: item[2])


def draw_markers(layer: OverlayLayer, sets: list[MarkerSet], size: float) -> None:
    """Open rings for the other curves, then the current curve filled (the last group)."""
    layer.clear()
    others = [s for s in sets if not s.current]
    if others:
        x = np.concatenate([s.x for s in others])
        y = np.concatenate([s.y for s in others])
        layer.add_points(x, y, size=size, pen=pg.mkPen(OUTLINE, width=4), brush=None)
        pens = [pg.mkPen(s.color, width=2) for s in others for _ in range(s.x.size)]
        layer.add_points(x, y, size=size, pen=pens, brush=None)
    for s in sets:
        if s.current:
            pen = pg.mkPen(OUTLINE, width=1.5)
            layer.add_points(s.x, s.y, size=size + 1, pen=pen, brush=pg.mkBrush(s.color))


def map_markers(controller, mode: str) -> list[MarkerSet]:
    """Markers on the map: (field, energy in the display unit)."""
    sets = []
    for name, color, current in shown_curves(controller, mode):
        b, e = controller.points.points(name)
        sets.append(MarkerSet(b, from_cm1(e, controller.unit), color, current))
    return sets


def field_index(field: np.ndarray, b: float) -> int | None:
    """Index of the value of *field* that *b* lies on (within half the smallest step)."""
    if field.size == 0:
        return None
    j = int(np.abs(field - b).argmin())
    steps = np.diff(np.unique(field))
    tolerance = steps.min() / 2 if steps.size else 1e-9 * max(1.0, abs(float(b)))
    return j if abs(field[j] - b) <= tolerance else None


def stacked_field(controller) -> np.ndarray | None:
    """Field of each trace index of the stacked plot, None without one.

    That is the field of the map shown, which for R(B)/R(B-ΔB) starts at the second field
    (derivatives keep the field axis).
    """
    if controller.result is None:
        return None
    try:
        return controller.result.base(controller.selection.kind).field
    except ValueError:  # R(B)/R(B-ΔB) of a single field
        return None


def stacked_markers(controller, stacked: StackedPlot, mode: str) -> list[MarkerSet]:
    """Markers on the stacked plot: each point on its field's trace (energy, trace y).

    Points on traces that are not shown (every n-th spectrum) or outside them are left out.
    """
    field = stacked_field(controller)
    if field is None:
        return []
    sets = []
    for name, color, current in shown_curves(controller, mode):
        b, e = controller.points.points(name)
        x, y = [], []
        for bk, ek in zip(b, from_cm1(e, controller.unit), strict=True):
            j = field_index(field, bk)
            yk = None if j is None else stacked.trace_y(j, float(ek))
            if yk is not None:
                x.append(float(ek))
                y.append(yk)
        sets.append(MarkerSet(np.array(x), np.array(y), color, current))
    return sets


class PickHint(QWidget):
    """A chip at the top centre of *host* while picking; clicks pass through it.

    While *avoid* (the error bar, a child of *host*) is shown, the chip sits below it.
    """

    MARGIN = 10

    def __init__(self, host: QWidget, avoid: QWidget | None = None):
        super().__init__(host)
        self._host = host
        self._avoid = avoid
        self._text = ""
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        host.installEventFilter(self)
        if avoid is not None:
            avoid.installEventFilter(self)
        self.hide()

    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text
        self.setAccessibleName(text)
        self.place()
        self.update()

    def sizeHint(self) -> QSize:
        metrics = self.fontMetrics()
        return QSize(metrics.horizontalAdvance(self._text) + 24, metrics.height() + 12)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kinds = (QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show, QEvent.Type.Hide)
        if event.type() in kinds:
            self.place()
        return False

    def place(self) -> None:
        area = self._host.rect()
        hint = self.sizeHint()
        width = max(40, min(hint.width(), area.width() - 2 * self.MARGIN))
        top = self.MARGIN
        avoid = self._avoid
        if avoid is not None and avoid.isVisible() and avoid.parentWidget() is self._host:
            top = avoid.geometry().bottom() + 8
        self.setGeometry((area.width() - width) // 2, top, width, hint.height())
        self.raise_()

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(tokens["accent"], 1))
        painter.setBrush(tokens["win"])
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 7, 7)
        painter.setPen(tokens["fg"])
        rect = self.rect().adjusted(12, 0, -12, 0)
        text = self.fontMetrics().elidedText(self._text, Qt.TextElideMode.ElideRight, rect.width())
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.end()
