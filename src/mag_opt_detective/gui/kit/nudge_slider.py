"""One-handle slider over a float value: an absolute range or a spring-back relative jog."""

from __future__ import annotations

import json
import math

from PySide6.QtCore import QAbstractAnimation, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QActionGroup,
    QColor,
    QContextMenuEvent,
    QFocusEvent,
    QFont,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import QMenu, QSizePolicy, QWidget

from mag_opt_detective.gui.kit._common import json_object, make_animation, to_float

RANGE, RELATIVE = "range", "relative"
MODES = (RANGE, RELATIVE)
SPANS = (1.0, 10.0, 50.0)  # % offered for the relative mode
DEFAULT_SPAN = 10.0
_KEYS = {  # steps: one fine step, or a coarse one (ten fine steps)
    Qt.Key.Key_Left: -1,
    Qt.Key.Key_Down: -1,
    Qt.Key.Key_Right: 1,
    Qt.Key.Key_Up: 1,
    Qt.Key.Key_PageDown: -10,
    Qt.Key.Key_PageUp: 10,
    Qt.Key.Key_Home: -math.inf,
    Qt.Key.Key_End: math.inf,
}


def jog_fraction(offset: float) -> float:
    """The share of the span that a handle *offset* (-1 … 1 of the half track) away from the
    centre gives: half as steep near the centre (fine control), the whole span at the ends."""
    offset = min(max(offset, -1.0), 1.0)
    return 0.5 * offset * (1.0 + offset * offset)


def mode_text(mode: str, span: float) -> str:
    """``Range`` or ``Relative ±10 %``."""
    return "Range" if mode == RANGE else f"Relative ±{span:g} %"


def _quantize(value: float, resolution: float) -> float:
    """*value* rounded to the power of ten at or below *resolution*."""
    if not (math.isfinite(resolution) and resolution > 0 and math.isfinite(value)):
        return value
    digits = -math.floor(math.log10(resolution))
    return float(round(value, digits))


class NudgeSlider(QWidget):
    """A slider with one handle and two modes.

    **Range** (absolute): the handle maps :meth:`set_range`'s ``[lo, hi]``; a press on the
    groove jumps there. Values outside the range are kept and drawn at its ends.

    **Relative** (jog): the handle rests at the centre tick. Dragging it away changes the value
    continuously, by up to ±:meth:`span` % of the value at the press when the handle reaches an
    end (:func:`jog_fraction`: gentle near the centre). A press anywhere grabs the handle where
    it is (no jump), so a click alone changes nothing. On release the handle springs back to the
    centre without changing the value, and the next drag works around the new value. The span
    is taken of ``max(|value|, floor)`` (:meth:`set_floor`), and of 1 for a zero value without
    a floor, so a value never gets stuck at 0. The change so far is shown on the groove while
    dragging (``+3.2 %``).

    Keyboard, both modes: arrows nudge by a fine step, Shift+arrows (or Page Up/Down) by ten.
    Range: 1 % and 10 % of the range (not beyond it), Home/End jump to its ends. Relative: a
    tenth of the span and the whole span of the current value. :meth:`set_bounds` limits the
    value in both modes and :meth:`set_step` makes it a multiple of a step (whole numbers).

    ``valueChanged`` fires on every user change (drag, keys), ``editingFinished`` when the
    mouse button or key is released after one. Setters are silent. The context menu offers the
    modes (:meth:`mode_menu`); a choice there emits ``modeChanged(mode, span)``. The mode and
    span are its settings (JSON).
    """

    valueChanged = Signal(float)
    editingFinished = Signal()
    modeChanged = Signal(str, float)

    HANDLE = 15.0  # handle diameter
    GROOVE = 4.0  # groove height
    MARGIN = 3.0  # room for the focus ring
    TICK = 10.0  # height of the centre tick (relative mode; half way out: half as high)
    SPRING_MS = 160

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._value = 0.0
        self._lo, self._hi = 0.0, 1.0
        self._minimum, self._maximum = -math.inf, math.inf
        self._mode = RANGE
        self._span = DEFAULT_SPAN
        self._modes = MODES
        self._floor = 0.0
        self._step: float | None = None
        self._drag = False
        self._grab = 0.0  # press x minus the handle's x
        self._origin = 0.0  # relative: the value at the press
        self._offset = 0.0  # relative: the handle from the centre, -1 … 1 of the half track
        self._changed = False
        self._key_changed = False
        self.duration_ms = self.SPRING_MS
        self._spring = make_animation(self, self.SPRING_MS)
        self._spring.valueChanged.connect(self._on_spring)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._describe()

    # --- values ------------------------------------------------------------------------
    def value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        """Show *value* without emitting (a relative drag goes on around it)."""
        value = float(value)
        if self._drag and self._mode == RELATIVE:
            self._origin = value - self._jog(self._origin, self._offset)
        self._value = value
        self._describe()
        self.update()

    def range(self) -> tuple[float, float]:
        return self._lo, self._hi

    def set_range(self, lo: float, hi: float) -> None:
        """The values the handle spans in the range mode."""
        self._lo, self._hi = sorted((float(lo), float(hi)))
        self.update()

    def bounds(self) -> tuple[float, float]:
        return self._minimum, self._maximum

    def set_bounds(self, minimum: float | None = None, maximum: float | None = None) -> None:
        """Hard limits of the value in both modes (None: open)."""
        self._minimum = -math.inf if minimum is None else float(minimum)
        self._maximum = math.inf if maximum is None else float(maximum)

    def set_step(self, step: float | None) -> None:
        """Values are multiples of *step* (1 for whole numbers); None: rounded to the travel."""
        self._step = None if step is None else float(step)

    def set_floor(self, floor: float) -> None:
        """The smallest magnitude the relative span is taken of (a typical scale)."""
        self._floor = abs(float(floor))

    def floor(self) -> float:
        return self._floor

    # --- mode --------------------------------------------------------------------------
    def mode(self) -> str:
        return self._mode

    def span(self) -> float:
        """The relative span in % of the value."""
        return self._span

    def modes(self) -> tuple[str, ...]:
        return self._modes

    def set_modes(self, modes) -> None:
        """The modes offered (e.g. only the range for a whole number)."""
        modes = tuple(m for m in MODES if m in modes)
        if not modes:
            raise ValueError("at least one mode")
        self._modes = modes
        if self._mode not in modes:
            self.set_mode(modes[0])

    def set_mode(self, mode: str, span: float | None = None) -> None:
        """Switch to *mode* (a relative *span* in %) without emitting; a mode that is not
        offered is ignored."""
        if mode not in MODES:
            raise ValueError(f"unknown slider mode {mode!r}")
        if span is not None:
            if not (math.isfinite(span) and span > 0):
                raise ValueError(f"invalid span {span!r}")
            self._span = float(span)
        if mode not in self._modes:
            return
        if mode != self._mode:
            self._end_drag()
            self._mode = mode
        self._describe()
        self.update()

    def mode_menu(self) -> QMenu:
        """The modes as a menu (also the context menu); a choice emits ``modeChanged``."""
        menu = QMenu(self)
        group = QActionGroup(menu)
        choices = [(RANGE, self._span)] + [(RELATIVE, span) for span in SPANS]
        for mode, span in choices:
            action = menu.addAction(mode_text(mode, span))
            action.setCheckable(True)
            action.setActionGroup(group)
            action.setChecked(mode == self._mode and (mode == RANGE or span == self._span))
            action.setEnabled(mode in self._modes)
            action.triggered.connect(lambda _on=False, m=mode, s=span: self._choose(m, s))
        return menu

    def _choose(self, mode: str, span: float) -> None:
        if (mode, span) == (self._mode, self._span) or mode not in self._modes:
            return
        self.set_mode(mode, span if mode == RELATIVE else None)
        self.modeChanged.emit(self._mode, self._span)

    # --- relative mode -------------------------------------------------------------------
    def span_amount(self, origin: float | None = None) -> float:
        """The change at a relative end for a drag starting at *origin* (default: now)."""
        origin = self._value if origin is None else origin
        base = max(abs(origin), self._floor) or 1.0
        return base * self._span / 100.0

    def handle_offset(self) -> float:
        """Where the handle is in the relative mode: -1 … 1 of the half track (0 at rest)."""
        return self._offset

    def is_dragging(self) -> bool:
        return self._drag

    def is_springing(self) -> bool:
        return self._spring.state() == QAbstractAnimation.State.Running

    def change_text(self) -> str:
        """The change of the running relative drag: ``+3.2 %`` (absolute from 0)."""
        if self._origin:
            return f"{(self._value - self._origin) / abs(self._origin) * 100:+.1f} %"
        return f"{self._value:+.4g}"

    def _jog(self, origin: float, offset: float) -> float:
        return jog_fraction(offset) * self.span_amount(origin)

    def _on_spring(self, value) -> None:
        self._offset = float(value)
        self.update()

    def _spring_back(self) -> None:
        if self._offset == 0.0 or self.duration_ms <= 0:
            self._offset = 0.0
            self.update()
            return
        self._spring.stop()
        self._spring.setDuration(self.duration_ms)
        self._spring.setStartValue(self._offset)
        self._spring.setEndValue(0.0)
        self._spring.start()

    def _end_drag(self) -> None:
        self._spring.stop()
        self._drag = False
        self._offset = 0.0

    # --- changes -------------------------------------------------------------------------
    def _clean(self, value: float, resolution: float) -> float:
        """*value* kept in the bounds and rounded (to the step, else to *resolution*)."""
        if self._step:
            value = round(value / self._step) * self._step
        else:
            value = _quantize(value, resolution)
        return min(max(value, self._minimum), self._maximum)

    def _apply(self, value: float) -> bool:
        """Store a user change and emit it; False (silent) if nothing changed."""
        if value == self._value or not math.isfinite(value):
            return False
        self._value = value
        self._describe()
        self.update()
        self.valueChanged.emit(value)
        return True

    def _range_resolution(self) -> float:
        return (self._hi - self._lo) / 1000.0

    def _describe(self) -> None:
        if self._mode == RANGE:
            how = "drag along the range"
        else:
            how = (
                f"drag away from the centre to change the value by up to ±{self._span:g} %; "
                "the handle springs back when released"
            )
        self.setAccessibleDescription(
            f"{mode_text(self._mode, self._span)}: {how}. Arrow keys nudge, Shift+arrows "
            f"more. Value {self._value:.6g}"
        )

    # --- geometry ----------------------------------------------------------------------
    def sizeHint(self) -> QSize:
        return QSize(160, int(self.HANDLE + 2 * self.MARGIN + 1))

    def minimumSizeHint(self) -> QSize:
        return QSize(int(2 * self.HANDLE + 2 * self.MARGIN), self.sizeHint().height())

    def track(self) -> tuple[float, float]:
        """The x of the handle's centre at the two ends."""
        left = self.MARGIN + self.HANDLE / 2
        return left, max(left, self.width() - left)

    def centre_x(self) -> float:
        x0, x1 = self.track()
        return (x0 + x1) / 2

    def x_for(self, value: float) -> float:
        """The x of *value* in the range mode (at the ends for values outside)."""
        x0, x1 = self.track()
        width = (self._hi - self._lo) or 1.0
        f = min(max((value - self._lo) / width, 0.0), 1.0)
        return x0 + f * (x1 - x0)

    def handle_x(self) -> float:
        if self._mode == RELATIVE:
            x0, x1 = self.track()
            return self.centre_x() + self._offset * (x1 - x0) / 2
        return self.x_for(self._value)

    def _value_at(self, x: float) -> float:
        x0, x1 = self.track()
        f = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
        return self._lo + min(max(f, 0.0), 1.0) * (self._hi - self._lo)

    # --- mouse -------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self._spring.stop()
        self._offset = 0.0
        x = event.position().x()
        self._drag = True
        self._changed = False
        if self._mode == RELATIVE:
            self._origin = self._value
            self._grab = x - self.centre_x()
        elif abs(x - self.handle_x()) <= self.HANDLE / 2 + 2:
            self._grab = x - self.handle_x()
        else:  # the groove: the handle jumps there
            self._grab = 0.0
            value = self._clean(self._value_at(x), self._range_resolution())
            self._changed = self._apply(value)
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._drag:
            super().mouseMoveEvent(event)
            return
        x = event.position().x() - self._grab
        if self._mode == RELATIVE:
            x0, x1 = self.track()
            half = (x1 - x0) / 2
            self._offset = min(max((x - self.centre_x()) / half, -1.0), 1.0) if half else 0.0
            amount = self.span_amount(self._origin)
            value = self._clean(self._origin + jog_fraction(self._offset) * amount, amount / 1000.0)
            self.update()
        else:
            value = self._clean(self._value_at(x), self._range_resolution())
        if self._apply(value):
            self._changed = True
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self._drag or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        self._drag = False
        if self._mode == RELATIVE:
            self._spring_back()
        self.update()
        event.accept()
        if self._changed:
            self._changed = False
            self.editingFinished.emit()

    def wheelEvent(self, event) -> None:
        event.ignore()  # the inspector scrolls

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        if len(self._modes) < 2:
            event.ignore()
            return
        self.mode_menu().exec(event.globalPos())
        event.accept()

    # --- keyboard ----------------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent) -> None:
        steps = _KEYS.get(event.key())
        if steps is None:
            super().keyPressEvent(event)
            return
        event.accept()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier and abs(steps) == 1:
            steps *= 10
        if self._mode == RANGE:
            if math.isinf(steps):
                value = self._lo if steps < 0 else self._hi
            else:
                fine = self._step or (self._hi - self._lo) / 100.0
                value = self._value + steps * fine
                # not further out than the range (a value typed outside comes back in)
                value = min(max(value, min(self._lo, self._value)), max(self._hi, self._value))
            value = self._clean(value, self._range_resolution())
        else:
            if math.isinf(steps):
                return
            amount = self.span_amount()
            value = self._clean(self._value + steps * amount / 10.0, amount / 1000.0)
        if self._apply(value):
            self._key_changed = True

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() in _KEYS and not event.isAutoRepeat():
            if self._key_changed:
                self._key_changed = False
                self.editingFinished.emit()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusInEvent(self, event: QFocusEvent) -> None:
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.update()

    # --- settings protocol ---------------------------------------------------------------
    def settings_value(self) -> str:
        return json.dumps({"mode": self._mode, "span": self._span})

    def set_settings_value(self, value) -> bool:
        data = json_object(value)
        if data is None or data.get("mode") not in self._modes:
            return False
        span = to_float(data.get("span", self._span))
        if span is None or span <= 0:
            return False
        self.set_mode(data["mode"], span)
        return True

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
        fill = accent if self.isEnabled() else neutral

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cy = self.height() / 2
        groove = QRectF(self.MARGIN, cy - self.GROOVE / 2, self.width() - 2 * self.MARGIN, 4)
        p.setPen(QPen(pal.color(group, QPalette.ColorRole.Midlight), 1))
        p.setBrush(pal.color(group, QPalette.ColorRole.AlternateBase))
        p.drawRoundedRect(groove.adjusted(0.5, 0.5, -0.5, -0.5), 2, 2)

        x = self.handle_x()
        if self._mode == RELATIVE:  # a scale: the centre tick and shorter ones half way out
            centre = self.centre_x()
            half = (self.track()[1] - self.track()[0]) / 2
            p.setPen(QPen(pal.color(group, QPalette.ColorRole.Mid), 1.5))
            for at, size in ((0.0, self.TICK), (-0.5, self.TICK / 2), (0.5, self.TICK / 2)):
                tick_x = centre + at * half
                p.drawLine(QPointF(tick_x, cy - size / 2), QPointF(tick_x, cy + size / 2))
            start = centre
        else:
            start = self.track()[0] - self.HANDLE / 2
        if abs(x - start) > 0.5:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(fill)
            left, right = sorted((start, x))
            p.drawRoundedRect(QRectF(left, cy - self.GROOVE / 2, right - left, 4), 2, 2)
        if self._mode == RELATIVE and self._drag and self._value != self._origin:
            self._paint_hint(p, group, x)

        radius = self.HANDLE / 2
        centre_point = QPointF(x, cy)
        if self.hasFocus():
            ring = QColor(accent)
            ring.setAlphaF(0.45)
            p.setPen(QPen(ring, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(centre_point, radius + 1.5, radius + 1.5)
        p.setPen(QPen(fill, 2))
        p.setBrush(pal.color(group, QPalette.ColorRole.Base))
        p.drawEllipse(centre_point, radius - 1, radius - 1)
        p.end()

    def _paint_hint(self, p: QPainter, group: QPalette.ColorGroup, handle_x: float) -> None:
        """The change so far on a small pill over the half of the groove away from the handle."""
        pal = self.palette()
        font = QFont(self.font())
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() * 0.8)
        p.setFont(font)
        text = self.change_text()
        width = p.fontMetrics().horizontalAdvance(text) + 10
        height = min(self.height() - 2.0, p.fontMetrics().height() + 2.0)
        x0, x1 = self.track()
        centre = self.centre_x()
        middle = (x0 + centre) / 2 if handle_x >= centre else (centre + x1) / 2
        left = min(max(middle - width / 2, 0.0), self.width() - width)
        box = QRectF(left, (self.height() - height) / 2, width, height)
        p.setPen(QPen(pal.color(group, QPalette.ColorRole.Mid), 1))
        p.setBrush(pal.color(group, QPalette.ColorRole.Base))
        p.drawRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5), height / 2, height / 2)
        p.setPen(pal.color(group, QPalette.ColorRole.Text))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
