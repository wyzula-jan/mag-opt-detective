"""Horizontal slider with two handles over a float extent."""

from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFocusEvent, QKeyEvent, QMouseEvent, QPainter, QPalette, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from mag_opt_detective.gui.theme import mix

LO, HI, BOTH = 0, 1, 2
_STEPS = {  # in fractions of the extent
    Qt.Key.Key_Left: -0.01,
    Qt.Key.Key_Down: -0.01,
    Qt.Key.Key_Right: 0.01,
    Qt.Key.Key_Up: 0.01,
    Qt.Key.Key_PageDown: -0.1,
    Qt.Key.Key_PageUp: 0.1,
    Qt.Key.Key_Home: -math.inf,
    Qt.Key.Key_End: math.inf,
}
TIP = "Drag a handle to move one end, or the bar between them to move the range (Shift+arrow keys)"


class RangeSlider(QWidget):
    """Two handles (low, high) over the float extent ``[a, b]``.

    Mouse: press on or near a handle drags it; press on the bar between the handles drags
    both and keeps their distance, until one end reaches the extent; press on the groove
    moves the nearest handle there. The handles win where they overlap the bar, and the
    whole bar when its free part is under :attr:`MIN_GRAB` pixels. Keyboard: Tab / Shift+Tab
    pick the handle (then leave the widget), arrows move it by 1 % of the extent, Page
    Up/Down by 10 %, Home/End jump to the ends; with Shift the same keys move the range.
    ``valuesChanged`` fires on every user change, ``editingFinished`` when the mouse button
    or key is released. :meth:`set_values` is programmatic and silent; values outside the
    extent are kept and drawn at the ends.

    Moving the range keeps its width, so the minimum span never applies to it. It moves only
    while an end has room inside the extent: on a range that fills the extent (as on Auto) a
    press between the handles moves the nearest handle, as on the groove. Muted is a colour
    only: every drag and key works, so a RangeControl on Auto switches to Fixed. Handles
    pinned at an end (min span apart) still move away from it together, by the bar when it
    is wide enough to grab, or by Shift+arrows.
    """

    valuesChanged = Signal(float, float)
    editingFinished = Signal()

    HANDLE = 15.0  # handle diameter
    GROOVE = 4.0  # groove height
    BAND = 10.0  # height of the hover band around the bar
    MARGIN = 3.0  # room for the focus ring
    MIN_GRAB = 6.0  # the bar's smallest grabbable part between the handles' reach (px)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._a, self._b = 0.0, 1.0
        self._lo, self._hi = 0.0, 1.0
        self._min_span: float | None = None
        self._active = LO
        self._drag: int | None = None
        self._grab = 0.0  # handle drags: press x minus the handle's x
        self._origin = (0.0, 0.0, 0.0)  # range drags: value under the press, lo, hi
        self._hover_x: float | None = None
        self._hot = False  # the bar is under the mouse and can be dragged
        self._cursor: Qt.CursorShape | None = None
        self._key_changed = False
        self._muted = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setToolTip(TIP)
        self._describe()

    # --- values ------------------------------------------------------------------------
    def extent(self) -> tuple[float, float]:
        return self._a, self._b

    def set_extent(self, a: float, b: float) -> None:
        a, b = sorted((float(a), float(b)))
        self._a, self._b = a, b
        self._refresh_hover()
        self.update()

    def values(self) -> tuple[float, float]:
        return self._lo, self._hi

    def set_values(self, lo: float, hi: float) -> None:
        """Set both values without emitting a signal."""
        self._lo, self._hi = sorted((float(lo), float(hi)))
        self._describe()
        self._refresh_hover()
        self.update()

    def set_min_span(self, span: float | None) -> None:
        """Smallest allowed ``hi - lo`` (None: 1 % of the extent)."""
        self._min_span = None if span is None else max(0.0, float(span))

    def min_span(self) -> float:
        return self._min_span if self._min_span is not None else 0.01 * self._width()

    def set_muted(self, muted: bool) -> None:
        """Draw the range in a neutral colour (used for the Auto state)."""
        self._muted = bool(muted)
        self.update()

    def is_muted(self) -> bool:
        return self._muted

    def active_handle(self) -> int:
        """The handle moved by the keyboard: 0 (low) or 1 (high)."""
        return self._active

    def set_active_handle(self, handle: int) -> None:
        self._active = HI if handle else LO
        self.update()

    def x_for(self, value: float) -> float:
        """The x of *value* in widget coordinates (at the ends for values outside)."""
        x0, x1 = self._track()
        f = min(max((value - self._a) / self._width(), 0.0), 1.0)
        return x0 + f * (x1 - x0)

    def part_at(self, x: float) -> int | None:
        """What a press at *x* grabs: a handle (0 low, 1 high), the range (2) or None."""
        handle = self._handle_at(x)
        if handle is not None:
            return handle
        x_lo, x_hi = self.x_for(self._lo), self.x_for(self._hi)
        if not x_lo < x < x_hi:
            return None
        if x_hi - x_lo - 2 * self._reach() < self.MIN_GRAB:  # too narrow: the handles win
            return self._handle_at(x, reach=math.inf)
        return BOTH if self._lo > self._a or self._hi < self._b else None

    def _width(self) -> float:
        return (self._b - self._a) or 1.0

    def _span(self) -> float:
        """The minimum span, at most the whole extent."""
        return min(self.min_span(), self._b - self._a)

    def _move(self, handle: int, value: float) -> bool:
        """Move *handle* to *value* (kept in the extent and min span apart); True if changed."""
        a, b, span = self._a, self._b, self._span()
        value = min(max(value, a), b)
        lo, hi = self._lo, self._hi
        if handle == LO:
            lo = min(value, hi - span)
            if lo < a:  # the other handle sits at the low end: make room inside the extent
                lo, hi = a, max(hi, a + span)
        else:
            hi = max(value, lo + span)
            if hi > b:
                lo, hi = min(lo, b - span), b
        return self._apply(lo, hi)

    def _shift(self, lo: float, hi: float, delta: float) -> bool:
        """Put the range at ``(lo, hi) + delta``, its width kept, stopped where an end reaches
        the extent (an end already outside does not move further out); True if changed."""
        down, up = min(self._a - lo, 0.0), max(self._b - hi, 0.0)  # the room on either side
        if delta <= down:
            lo, hi = (self._a if down else lo), hi + down
        elif delta >= up:
            lo, hi = lo + up, (self._b if up else hi)
        else:
            lo, hi = lo + delta, hi + delta
        return self._apply(lo, hi)

    def _apply(self, lo: float, hi: float) -> bool:
        """Store a user change and emit it; False (and silent) if nothing changed."""
        if (lo, hi) == (self._lo, self._hi):
            return False
        self._lo, self._hi = lo, hi
        self._describe()
        self.update()
        self.valuesChanged.emit(lo, hi)
        return True

    def _describe(self) -> None:
        self.setAccessibleDescription(
            f"from {self._lo:.6g} to {self._hi:.6g}; Shift+arrow keys move the range"
        )

    # --- geometry ----------------------------------------------------------------------
    def sizeHint(self) -> QSize:
        return QSize(160, int(self.HANDLE + 2 * self.MARGIN + 1))

    def minimumSizeHint(self) -> QSize:
        return QSize(int(2 * self.HANDLE + 2 * self.MARGIN), self.sizeHint().height())

    def _track(self) -> tuple[float, float]:
        left = self.MARGIN + self.HANDLE / 2
        return left, max(left, self.width() - left)

    def _value_at(self, x: float) -> float:
        x0, x1 = self._track()
        f = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
        return self._a + min(max(f, 0.0), 1.0) * self._width()

    def _reach(self) -> float:
        """How far from a handle's centre a press still grabs it."""
        return self.HANDLE / 2 + 2

    def _handle_at(self, x: float, reach: float | None = None) -> int | None:
        x_lo, x_hi = self.x_for(self._lo), self.x_for(self._hi)
        d_lo, d_hi = abs(x - x_lo), abs(x - x_hi)
        reach = self._reach() if reach is None else reach
        if min(d_lo, d_hi) > reach:
            return None
        nearer_lo = d_lo < d_hi or (d_lo == d_hi and x <= x_lo)  # a tie: the side of the press
        handle, other, d_other = (LO, HI, d_hi) if nearer_lo else (HI, LO, d_lo)
        if self._pinned(handle) and d_other <= reach:  # take the one that can move
            return other
        return handle

    def _pinned(self, handle: int) -> bool:
        """True if *handle* can move neither way: at an end, with the other one span away."""
        if self._hi - self._lo > self._span() * (1 + 1e-9):
            return False
        return self._lo <= self._a if handle == LO else self._hi >= self._b

    # --- mouse -------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        x = event.position().x()
        part = self.part_at(x)
        if part == BOTH:
            self._origin = (self._value_at(x), self._lo, self._hi)
        elif part is None:
            value = self._value_at(x)
            part = LO if abs(value - self._lo) <= abs(value - self._hi) else HI
            self._grab = 0.0
            self._active = part
            self._move(part, value)
        else:
            self._grab = x - self.x_for(self._lo if part == LO else self._hi)
            self._active = part
        self._drag = part
        self._refresh_hover()
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x = event.position().x()
        if self._drag is None:
            self._hover_x = x
            self._refresh_hover()
            super().mouseMoveEvent(event)
            return
        if self._drag == BOTH:
            start, lo, hi = self._origin
            self._shift(lo, hi, self._value_at(x) - start)
        else:
            self._move(self._drag, self._value_at(x - self._grab))
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag is None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        self._drag = None
        inside = self.rect().contains(event.position().toPoint())
        self._hover_x = event.position().x() if inside else None
        self._refresh_hover()
        self.update()
        event.accept()
        self.editingFinished.emit()

    def leaveEvent(self, event: QEvent) -> None:
        self._hover_x = None
        self._refresh_hover()
        super().leaveEvent(event)

    def _refresh_hover(self) -> None:
        """Hover state and cursor: an open hand over a bar that can move, closed while
        moving it."""
        if self._drag == BOTH:
            hot, shape = True, Qt.CursorShape.ClosedHandCursor
        else:
            hot = self._drag is None and self._hover_x is not None
            hot = hot and self.part_at(self._hover_x) == BOTH
            shape = Qt.CursorShape.OpenHandCursor if hot else None
        if hot != self._hot:
            self._hot = hot
            self.update()
        if shape != self._cursor:
            self._cursor = shape
            if shape is None:
                self.unsetCursor()
            else:
                self.setCursor(shape)

    # --- keyboard ----------------------------------------------------------------------
    def focusNextPrevChild(self, next: bool) -> bool:
        if self.hasFocus():
            if next and self._active == LO:
                self.set_active_handle(HI)
                return True
            if not next and self._active == HI:
                self.set_active_handle(LO)
                return True
        return super().focusNextPrevChild(next)

    def focusInEvent(self, event: QFocusEvent) -> None:
        if event.reason() == Qt.FocusReason.TabFocusReason:
            self._active = LO
        elif event.reason() == Qt.FocusReason.BacktabFocusReason:
            self._active = HI
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        step = _STEPS.get(event.key())
        if step is None:
            super().keyPressEvent(event)
            return
        step *= self._width()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            changed = self._shift(self._lo, self._hi, step)
        else:
            current = self._lo if self._active == LO else self._hi
            changed = self._move(self._active, current + step)
        if changed:
            self._key_changed = True
        event.accept()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() in _STEPS and not event.isAutoRepeat():
            if self._key_changed:
                self._key_changed = False
                self.editingFinished.emit()
            event.accept()
            return
        super().keyReleaseEvent(event)

    # --- painting ----------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        pal = self.palette()
        if not self.isEnabled():
            group = QPalette.ColorGroup.Disabled
        elif self.isActiveWindow():
            group = QPalette.ColorGroup.Active
        else:
            group = QPalette.ColorGroup.Inactive
        accent = pal.color(group, QPalette.ColorRole.Highlight)
        neutral = pal.color(group, QPalette.ColorRole.PlaceholderText)
        fill = neutral if self._muted or not self.isEnabled() else accent
        hot = self._hot and self.isEnabled()

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cy = self.height() / 2
        width = self.width() - 2 * self.MARGIN
        groove = QRectF(self.MARGIN, cy - self.GROOVE / 2, width, self.GROOVE)
        p.setPen(QPen(pal.color(group, QPalette.ColorRole.Midlight), 1))
        p.setBrush(pal.color(group, QPalette.ColorRole.AlternateBase))
        p.drawRoundedRect(groove.adjusted(0.5, 0.5, -0.5, -0.5), 2, 2)

        x_lo, x_hi = self.x_for(self._lo), self.x_for(self._hi)
        p.setPen(Qt.PenStyle.NoPen)
        bar = fill
        if hot:  # a soft band and a stronger bar: the range can be grabbed
            band = QColor(fill)
            band.setAlphaF(0.24 if self._drag == BOTH else 0.16)
            p.setBrush(band)
            radius = self.BAND / 2
            p.drawRoundedRect(QRectF(x_lo, cy - radius, x_hi - x_lo, self.BAND), radius, radius)
            bar = mix(fill, pal.color(group, QPalette.ColorRole.Text), 0.8)
        p.setBrush(bar)
        p.drawRoundedRect(QRectF(x_lo, cy - self.GROOVE / 2, x_hi - x_lo, self.GROOVE), 2, 2)

        radius = self.HANDLE / 2
        handles = [(LO, x_lo), (HI, x_hi)]
        if self._active == LO:
            handles.reverse()  # the active handle on top
        for handle, x in handles:
            centre = QPointF(x, cy)
            if self.hasFocus() and handle == self._active and self._drag != BOTH:
                ring = QColor(accent)
                ring.setAlphaF(0.45)
                p.setPen(QPen(ring, 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(centre, radius + 1.5, radius + 1.5)
            p.setPen(QPen(fill, 2))
            p.setBrush(pal.color(group, QPalette.ColorRole.Base))
            p.drawEllipse(centre, radius - 1, radius - 1)
        p.end()
