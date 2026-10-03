"""The region of the Auto-pick tool's Detect mode: a box, a rotated box, an ellipse or a polygon
drawn on the map, which stays there to be moved and reshaped (pyqtgraph ROIs).

The map's axes have different units (tesla, cm^-1), so a region lives in a *frame*: a parent
item scaled so that one of its units is one screen pixel when the region is drawn. A rotated
box or an ellipse then looks and turns as on paper, and a unit switch only rescales the
frame's energy axis. :meth:`Region.polygon` gives the outline in plot coordinates (field on x,
energy in the display unit on y); the search builds its mask from that, on the map's own grid.
The outline is drawn by the region itself (a dashed accent line on a halo, as the plain box),
the ROI only draws its handles.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import ClassVar

import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ROI import Handle, MouseDragHandler
from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem

from mag_opt_detective.core.units import CM1_PER_UNIT, Unit

RECT, ROTATED, ELLIPSE, POLYGON = "rect", "rotated", "ellipse", "polygon"
SHAPES = (  # value, label, tooltip
    (RECT, "Box", "Drag a box around the lines; drag its corners to resize it"),
    (
        ROTATED,
        "Rotated",
        "Drag a box around the lines, then turn it with its round handle "
        "(along a sloping line, for example)",
    ),
    (ELLIPSE, "Ellipse", "Drag an ellipse around the lines; its round handle turns it"),
    (
        POLYGON,
        "Polygon",
        "Draw a loop around the lines with the mouse held down; it becomes a polygon whose "
        "corners you can drag (a click on an edge adds a corner, a right-click on one "
        "removes it)",
    ),
)
NOUNS = {RECT: "box", ROTATED: "box", ELLIPSE: "ellipse", POLYGON: "polygon"}
ELLIPSE_POINTS = 120  # of the outline (and the mask) of an ellipse
LASSO_TOLERANCE = 2.5  # px: a drawn loop is simplified to corners this close to it
MAX_CORNERS = 40  # of a drawn polygon (the tolerance grows until it has no more)
MIN_AREA = 50.0  # px^2: a smaller loop is no region
HANDLE_RADIUS = 7.0  # px
Z = 20.0  # the region and its handles: above the overlay layers, to be grabbed
OUTLINE_Z = 9.7  # its outline: below the overlay layers, above the image

Colors = Callable[[], tuple[QColor, QColor]]  # accent and halo


# ---------------------------------------------------------------------- geometry
def simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """The corners of a drawn line (Ramer-Douglas-Peucker): every point is within *tolerance*
    of the line through the corners kept."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        a, b = points[first], points[last]
        inner = points[first + 1 : last]
        ab = b - a
        length = math.hypot(*ab)
        if length == 0:
            distance = np.hypot(*(inner - a).T)
        else:
            distance = np.abs(ab[0] * (inner[:, 1] - a[1]) - ab[1] * (inner[:, 0] - a[0])) / length
        k = int(distance.argmax())
        if distance[k] > tolerance:
            middle = first + 1 + k
            keep[middle] = True
            stack += [(first, middle), (middle, last)]
    return points[keep]


def loop_corners(points: np.ndarray) -> np.ndarray | None:
    """The corners of a polygon from a loop drawn in pixels, or None if it encloses too little
    (the tolerance grows until there are at most :data:`MAX_CORNERS`)."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    tolerance = LASSO_TOLERANCE
    corners = simplify(points, tolerance)
    while len(corners) > MAX_CORNERS:
        tolerance *= 1.5
        corners = simplify(points, tolerance)
    if len(corners) > 3 and np.hypot(*(corners[-1] - corners[0])) <= tolerance:
        corners = corners[:-1]  # the loop was closed by hand
    x, y = corners.T
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return corners if len(corners) >= 3 and area >= MIN_AREA else None


def ellipse_points(width: float, height: float, n: int = ELLIPSE_POINTS) -> np.ndarray:
    """Points around the ellipse in the box ``(0, 0, width, height)``."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([width / 2 * (1 + np.cos(t)), height / 2 * (1 + np.sin(t))])


def rubber_band(shape: str, points: Sequence[QPointF]) -> np.ndarray:
    """The outline of a drag in progress (plot coordinates): the box or the ellipse between its
    first and last point, or the loop drawn so far."""
    xy = np.array([(p.x(), p.y()) for p in points], dtype=float).reshape(-1, 2)
    if shape == POLYGON:
        return xy
    (x0, y0), (x1, y1) = xy[0], xy[-1]
    if shape == ELLIPSE:
        return ellipse_points(x1 - x0, y1 - y0) + np.array([x0, y0])
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _path(polygon: np.ndarray) -> QPainterPath:
    path = QPainterPath()
    if len(polygon):
        path.addPolygon(QPolygonF([QPointF(x, y) for x, y in polygon]))
        path.closeSubpath()
    return path


# ---------------------------------------------------------------------- items
class Frame(pg.ItemGroup):
    """The parent of a region: one unit is a pixel (as drawn) and energies follow the unit."""

    def __init__(self):
        super().__init__()
        self.setZValue(Z)
        self._scale = (1.0, 1.0)  # field (T) and energy (cm^-1) per unit
        self._unit = Unit.CM1

    def set_scale(self, field: float, energy_cm1: float, unit: Unit) -> None:
        self._scale = (field, energy_cm1)
        self.set_unit(unit)

    def set_unit(self, unit: Unit) -> None:
        self._unit = Unit(unit)
        field, energy_cm1 = self._scale
        self.setTransform(QTransform.fromScale(field, energy_cm1 / CM1_PER_UNIT[self._unit]))


class RegionHandle(Handle):
    """A handle in the tool's colours: filled with the accent on a halo, inverted under the
    mouse or while dragged. Scale and corner handles are squares (diamonds mark the chosen
    line), turning handles round."""

    types: ClassVar[dict] = {**Handle.types, "s": (4, math.pi / 4)}

    def __init__(self, typ: str, parent: pg.ROI, colors: Colors):
        super().__init__(HANDLE_RADIUS, typ=typ, parent=parent)
        self.colors = colors
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paint(self, p: QPainter, opt, widget) -> None:
        accent, halo = self.colors()
        hot = self.currentPen is self.hoverPen
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(accent if hot else halo, 1.6)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(QBrush(halo if hot else accent))
        p.drawPath(QTransform.fromScale(0.78, 0.78).map(self.shape()))  # inside the bounds


class _Styled:
    """Handles drawn by :class:`RegionHandle`; the ROI draws no outline (the region does)."""

    colors: Colors

    def addHandle(self, info, index=None):
        if info.get("item") is None:
            info["item"] = RegionHandle(info["type"], self, self.colors)
        return super().addHandle(info, index=index)


def _no_pen() -> QPen:
    return QPen(Qt.PenStyle.NoPen)


class BoxROI(_Styled, pg.ROI):
    """A box with a scale handle on each corner (the opposite corner stays); *rotatable* adds
    a round handle on the right edge that turns it about its centre."""

    def __init__(self, pos, size, colors: Colors, rotatable: bool):
        self.colors = colors
        super().__init__(pos, size, pen=_no_pen(), hoverPen=_no_pen(), rotatable=rotatable)
        for corner in ((0, 0), (1, 0), (1, 1), (0, 1)):
            self.addScaleHandle(corner, (1 - corner[0], 1 - corner[1]))
        if rotatable:
            self.addRotateHandle((1, 0.5), (0.5, 0.5))


class EllipseRegionROI(_Styled, pg.EllipseROI):
    """An ellipse with pyqtgraph's handles: scale (square) and turn (round)."""

    def __init__(self, pos, size, colors: Colors):
        self.colors = colors
        super().__init__(pos, size, pen=_no_pen(), hoverPen=_no_pen())


class PolygonROI(_Styled, pg.PolyLineROI):
    """A closed polygon with a handle on each corner; a drag on an edge moves it all."""

    def __init__(self, corners, colors: Colors):
        self.colors = colors
        super().__init__(corners, closed=True, pen=_no_pen(), hoverPen=_no_pen())

    def addSegment(self, h1, h2, index=None):
        super().addSegment(h1, h2, index=index)
        segment = self.segments[-1 if index is None else index]
        segment.mouseDragHandler = MouseDragHandler(self)  # an edge takes drags, not moves


# ---------------------------------------------------------------------- the region
class Region(QObject):
    """The Detect region on a map plot: its ROI (in a :class:`Frame`) and outline.

    :attr:`changed` fires on every step of an edit, :attr:`finished` when an edit ends (a
    drag let go, a corner added or removed); :attr:`editing` is True in between.
    """

    changed = Signal()
    finished = Signal()

    def __init__(self, plot, colors: Colors, parent: QObject | None = None):
        super().__init__(parent)
        self.plot = plot  # a ColorMapPlot
        self.colors = colors
        self.shape: str | None = None
        self.roi: pg.ROI | None = None
        self.editing = False
        self.frame = Frame()
        plot.plot.addItem(self.frame, ignoreBounds=True)
        self.outline = (QGraphicsPathItem(), QGraphicsPathItem())  # halo, dashed line
        for z, item in enumerate(self.outline):
            item.setZValue(OUTLINE_Z + 0.05 * z)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            item.hide()
            plot.plot.addItem(item, ignoreBounds=True)

    # --- making and dropping ------------------------------------------------------------
    def create(self, shape: str, points: Sequence[QPointF], unit: Unit) -> bool:
        """Put a region of *shape* where a drag went through *points* (plot coordinates): the
        box or ellipse between the first and last point, or the polygon around them; False
        (and the region stays as it was) if a polygon would enclose too little."""
        field_px, energy_px = self.pixel_size()
        local = np.array([(p.x() / field_px, p.y() / energy_px) for p in points])
        if shape == POLYGON:
            corners = loop_corners(local)
            if corners is None:
                return False
            roi = PolygonROI([tuple(c) for c in corners], self.colors)
        else:
            low = np.minimum(local[0], local[-1])
            size = np.maximum(np.abs(local[-1] - local[0]), 1.0)
            if shape == ELLIPSE:
                roi = EllipseRegionROI(tuple(low), tuple(size), self.colors)
            else:
                roi = BoxROI(tuple(low), tuple(size), self.colors, rotatable=shape == ROTATED)
        self.clear()
        self.frame.set_scale(field_px, energy_px * CM1_PER_UNIT[Unit(unit)], unit)
        self.shape, self.roi = shape, roi
        roi.setCursor(Qt.CursorShape.SizeAllCursor)
        roi.setParentItem(self.frame)
        roi.sigRegionChangeStarted.connect(self._on_started)
        roi.sigRegionChanged.connect(self._on_changed)
        roi.sigRegionChangeFinished.connect(self._on_finished)
        self.show_outline()
        return True

    def clear(self) -> None:
        """Drop the region (and hide its outline)."""
        roi, self.roi, self.shape, self.editing = self.roi, None, None, False
        if roi is not None:
            for signal in (
                roi.sigRegionChangeStarted,
                roi.sigRegionChanged,
                roi.sigRegionChangeFinished,
            ):
                signal.disconnect()
            roi.setParentItem(None)
            scene = roi.scene()
            if scene is not None:
                scene.removeItem(roi)
        self.hide_outline()

    def pixel_size(self) -> tuple[float, float]:
        """Field and energy (display unit) per screen pixel, or a fair guess before the plot
        is laid out."""
        vb = self.plot.plot.vb
        sizes = vb.viewPixelSize()
        if all(math.isfinite(s) and s > 0 for s in sizes):
            return float(sizes[0]), float(sizes[1])
        (x0, x1), (y0, y1) = vb.viewRange()
        return (abs(x1 - x0) or 1.0) / 600.0, (abs(y1 - y0) or 1.0) / 400.0

    # --- reading ------------------------------------------------------------------------
    def polygon(self) -> np.ndarray | None:
        """The outline in plot coordinates, ``(n, 2)`` of (field, energy); None without one."""
        roi = self.roi
        if roi is None:
            return None
        if self.shape == POLYGON:
            local = [h.pos() for h in roi.getHandles()]
        else:
            width, height = roi.size()
            if self.shape == ELLIPSE:
                local = [QPointF(x, y) for x, y in ellipse_points(width, height)]
            else:
                local = [QPointF(0, 0), QPointF(width, 0), QPointF(width, height)]
                local.append(QPointF(0, height))
        mapped = self.frame.mapToParent(roi.mapToParent(QPolygonF(local)))
        return np.array([(p.x(), p.y()) for p in mapped], dtype=float).reshape(-1, 2)

    def is_box(self) -> bool:
        """The region is an upright box (searched as a box, without a mask)."""
        return self.shape == RECT

    def owns(self, item: QGraphicsItem) -> bool:
        """*item* is the region or one of its handles or edges (a press there edits it)."""
        roi = self.roi
        return roi is not None and (item is roi or roi.isAncestorOf(item))

    # --- showing ------------------------------------------------------------------------
    def set_unit(self, unit: Unit) -> None:
        """Keep the region where it is on the map when the energies change unit."""
        self.frame.set_unit(unit)
        self.show_outline()

    def show_outline(self, polygon: np.ndarray | None = None, dragging: bool = False) -> None:
        """The dashed outline of the region (or of *polygon*, a drag in progress), filled
        lightly while it is drawn or edited."""
        if polygon is None:
            polygon = self.polygon()
            dragging = dragging or self.editing
        if polygon is None or len(polygon) < 2:
            self.hide_outline()
            return
        accent, halo = self.colors()
        shadow, line = self.outline
        dark = QPen(halo, 2.5)
        dashed = QPen(accent, 1.2, Qt.PenStyle.DashLine)
        for pen in (dark, dashed):
            pen.setCosmetic(True)
        fill = QColor(accent)
        fill.setAlphaF(0.12 if dragging else 0.0)
        shadow.setPen(dark)
        shadow.setBrush(QBrush(fill))
        line.setPen(dashed)
        line.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        path = _path(polygon)
        for item in self.outline:
            item.setPath(path)
            item.show()
        if self.roi is not None:
            for handle in self.roi.getHandles():
                handle.update()  # the theme's colours

    def hide_outline(self) -> None:
        for item in self.outline:
            item.hide()

    def outline_visible(self) -> bool:
        return self.outline[1].isVisible()

    def outline_path(self) -> QPainterPath:
        return self.outline[1].path()

    # --- edits --------------------------------------------------------------------------
    def _on_started(self, _roi=None) -> None:
        self.editing = True
        self.show_outline()

    def _on_changed(self, _roi=None) -> None:
        self.show_outline()
        self.changed.emit()

    def _on_finished(self, _roi=None) -> None:
        self.editing = False
        self.show_outline()
        self.finished.emit()
