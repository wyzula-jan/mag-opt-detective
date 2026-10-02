"""Horizontal slider with two handles over a float extent."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFocusEvent, QKeyEvent, QMouseEvent, QPainter, QPalette, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

LO, HI = 0, 1
_MOVE_KEYS = {
    Qt.Key.Key_Left,
    Qt.Key.Key_Right,
    Qt.Key.Key_Up,
    Qt.Key.Key_Down,
    Qt.Key.Key_PageUp,
    Qt.Key.Key_PageDown,
    Qt.Key.Key_Home,
    Qt.Key.Key_End,
}


class RangeSlider(QWidget):
    """Two handles (low, high) over the float extent ``[a, b]``.

    Mouse: press on or near a handle drags it; press on the groove moves the nearest handle
    there. Keyboard: Tab / Shift+Tab pick the handle (then leave the widget), arrows move it by
    1 % of the extent (Shift: 10 %), Page Up/Down by 10 %, Home/End jump to the ends.
    ``valuesChanged`` fires on every user change, ``editingFinished`` when the mouse button or
    key is released. :meth:`set_values` is programmatic and silent; values outside the extent
    are kept and drawn at the ends.
    """

    valuesChanged = Signal(float, float)
    editingFinished = Signal()

    HANDLE = 15.0  # handle diameter
    GROOVE = 4.0  # groove height
    MARGIN = 3.0  # room for the focus ring

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._a, self._b = 0.0, 1.0
        self._lo, self._hi = 0.0, 1.0
        self._min_span: float | None = None
        self._active = LO
        self._drag: int | None = None
        self._grab = 0.0
        self._key_changed = False
        self._muted = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._describe()

    # --- values ------------------------------------------------------------------------
    def extent(self) -> tuple[float, float]:
        return self._a, self._b

    def set_extent(self, a: float, b: float) -> None:
        a, b = sorted((float(a), float(b)))
        self._a, self._b = a, b
        self.update()

    def values(self) -> tuple[float, float]:
        return self._lo, self._hi

    def set_values(self, lo: float, hi: float) -> None:
        """Set both values without emitting a signal."""
        self._lo, self._hi = sorted((float(lo), float(hi)))
        self._describe()
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
        if (lo, hi) == (self._lo, self._hi):
            return False
        self._lo, self._hi = lo, hi
        self._describe()
        self.update()
        self.valuesChanged.emit(lo, hi)
        return True

    def _describe(self) -> None:
        self.setAccessibleDescription(f"from {self._lo:.6g} to {self._hi:.6g}")

    # --- geometry ----------------------------------------------------------------------
    def sizeHint(self) -> QSize:
        return QSize(160, int(self.HANDLE + 2 * self.MARGIN + 1))

    def minimumSizeHint(self) -> QSize:
        return QSize(int(2 * self.HANDLE + 2 * self.MARGIN), self.sizeHint().height())

    def _track(self) -> tuple[float, float]:
        left = self.MARGIN + self.HANDLE / 2
        return left, max(left, self.width() - left)

    def _x_for(self, value: float) -> float:
        x0, x1 = self._track()
        f = min(max((value - self._a) / self._width(), 0.0), 1.0)
        return x0 + f * (x1 - x0)

    def _value_at(self, x: float) -> float:
        x0, x1 = self._track()
        f = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
        return self._a + min(max(f, 0.0), 1.0) * self._width()

    def _handle_at(self, x: float) -> int | None:
        x_lo, x_hi = self._x_for(self._lo), self._x_for(self._hi)
        d_lo, d_hi = abs(x - x_lo), abs(x - x_hi)
        reach = self.HANDLE / 2 + 2
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
        handle = self._handle_at(x)
        if handle is None:
            value = self._value_at(x)
            handle = LO if abs(value - self._lo) <= abs(value - self._hi) else HI
            self._grab = 0.0
            self._active = handle
            self._move(handle, value)
        else:
            self._grab = x - self._x_for(self._lo if handle == LO else self._hi)
            self._active = handle
        self._drag = handle
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is None:
            super().mouseMoveEvent(event)
            return
        self._move(self._drag, self._value_at(event.position().x() - self._grab))
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag is None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        self._drag = None
        event.accept()
        self.editingFinished.emit()

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
        key = event.key()
        if key not in _MOVE_KEYS:
            super().keyPressEvent(event)
            return
        big = 0.1 * self._width()
        step = (
            big if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 0.01 * self._width()
        )
        current = self._lo if self._active == LO else self._hi
        target = {
            Qt.Key.Key_Left: current - step,
            Qt.Key.Key_Down: current - step,
            Qt.Key.Key_Right: current + step,
            Qt.Key.Key_Up: current + step,
            Qt.Key.Key_PageDown: current - big,
            Qt.Key.Key_PageUp: current + big,
            Qt.Key.Key_Home: self._a,
            Qt.Key.Key_End: self._b,
        }[key]
        if self._move(self._active, target):
            self._key_changed = True
        event.accept()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() in _MOVE_KEYS and not event.isAutoRepeat():
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

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cy = self.height() / 2
        width = self.width() - 2 * self.MARGIN
        groove = QRectF(self.MARGIN, cy - self.GROOVE / 2, width, self.GROOVE)
        p.setPen(QPen(pal.color(group, QPalette.ColorRole.Midlight), 1))
        p.setBrush(pal.color(group, QPalette.ColorRole.AlternateBase))
        p.drawRoundedRect(groove.adjusted(0.5, 0.5, -0.5, -0.5), 2, 2)

        x_lo, x_hi = self._x_for(self._lo), self._x_for(self._hi)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(QRectF(x_lo, cy - self.GROOVE / 2, x_hi - x_lo, self.GROOVE), 2, 2)

        radius = self.HANDLE / 2
        handles = [(LO, x_lo), (HI, x_hi)]
        if self._active == LO:
            handles.reverse()  # the active handle on top
        for handle, x in handles:
            centre = QPointF(x, cy)
            if self.hasFocus() and handle == self._active:
                ring = QColor(accent)
                ring.setAlphaF(0.45)
                p.setPen(QPen(ring, 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(centre, radius + 1.5, radius + 1.5)
            p.setPen(QPen(fill, 2))
            p.setBrush(pal.color(group, QPalette.ColorRole.Base))
            p.drawEllipse(centre, radius - 1, radius - 1)
        p.end()
