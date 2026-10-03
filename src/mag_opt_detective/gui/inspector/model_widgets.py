"""Small widgets of the model cards: number fields, parameter rows (caption, slider and field on
shared columns), the slider mode, the expression editor, the colour swatch and compact grids.
They are painted from the theme tokens (light and dark)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import (
    QActionGroup,
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QIcon,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QTextCharFormat,
    QTextCursor,
    QTextFormat,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.gui import icons
from mag_opt_detective.gui.inspector.model_state import COLOR_NAMES, MODEL_COLORS
from mag_opt_detective.gui.kit.nudge_slider import (
    DEFAULT_SPAN,
    RANGE,
    RELATIVE,
    SPANS,
    NudgeSlider,
    mode_text,
)
from mag_opt_detective.gui.panels.common import ElidedLabel, UnitField, mono_font, scaled_font
from mag_opt_detective.gui.theme import current_tokens
from mag_opt_detective.gui.widgets import parse_float

SHADOW = QColor(10, 6, 14, 150)  # data colour: the dark shadow of curves and dots
UNIT_SCALE = 0.88  # the unit inside a number field (as UnitField draws it)
FIELD_HEIGHT = 24  # number fields and the controls beside them (the kit's compact height)
FIELD_PADDING = 18  # a number field's margins, spacing and border around number and unit


def muted_label(text: str, factor: float = 0.88) -> QLabel:
    """A small muted label (column headers, field captions)."""
    label = QLabel(text)
    label.setProperty("kit", "muted")
    label.setFont(scaled_font(label, factor))
    return label


def tool_button(icon: str, tooltip: str, color: str = "muted") -> QToolButton:
    """A small icon-only button in the plot-toolbar style."""
    button = QToolButton()
    button.setProperty("kit", "tool")
    button.setToolTip(tooltip)
    button.setAccessibleName(tooltip)
    button.setIconSize(QSize(14, 14))
    button.setFixedSize(24, 24)
    icons.set_icon(button, icon, color)
    return button


class IconButton(QToolButton):
    """A small icon button painted as the kit tool buttons but without their padding, so the
    icon keeps its size in a tight row (the card header in a narrow inspector)."""

    def __init__(self, icon: str, tooltip: str, width: int, icon_size: int = 13):
        super().__init__()
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setFixedSize(width, 20)
        self.setIconSize(QSize(icon_size, icon_size))
        icons.set_icon(self, icon, "muted")

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        hot = self.underMouse() and self.isEnabled()
        if hot or self.hasFocus():
            painter.setPen(QPen(tokens["accent"], 1) if self.hasFocus() else Qt.PenStyle.NoPen)
            painter.setBrush(tokens["hover"] if hot else Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 5, 5)
        mode = QIcon.Mode.Normal if self.isEnabled() else QIcon.Mode.Disabled
        self.icon().paint(painter, self.rect(), Qt.AlignmentFlag.AlignCenter, mode)
        painter.end()

    def enterEvent(self, event) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.update()
        super().leaveEvent(event)


def compact(field: UnitField) -> UnitField:
    """Let a field box shrink with a narrow inspector (cells of the model tables)."""
    field.edit.setMinimumWidth(14)
    field.layout().setContentsMargins(4, 2, 5, 2)
    return field


class NumberField(UnitField):
    """A number (C locale, scientific notation allowed), right-aligned in a rounded box with its
    unit.

    ``valueEdited`` fires while the user types a valid number (None for an empty *optional*
    field); invalid text (or a number outside *minimum* .. *maximum*) marks the box.
    :meth:`set_value` never emits.
    """

    valueEdited = Signal(object)

    def __init__(
        self,
        unit: str = "",
        *,
        name: str = "",
        optional: bool = False,
        minimum: float | None = None,
        maximum: float | None = None,
        integer: bool = False,
        digits: int = 6,
        placeholder: str = "",
    ):
        edit = QLineEdit()
        super().__init__(edit, unit)
        compact(self)
        self.setFixedHeight(FIELD_HEIGHT)
        edit.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._optional = optional
        self._minimum = minimum
        self._maximum = maximum
        self._integer = integer
        self._digits = digits
        self._value: float | None = None
        edit.setPlaceholderText(placeholder)
        edit.setAccessibleName(name)
        edit.setToolTip(name)
        edit.textChanged.connect(self._on_text)
        edit.editingFinished.connect(self._tidy)

    def value(self) -> float | None:
        return self._value

    def text(self) -> str:
        return self.edit.text()

    def set_value(self, value: float | None) -> None:
        """Show *value* without emitting (None or an infinite bound shows empty).

        Text that already reads as *value* stays as typed (e.g. "1." while typing "1.5").
        """
        valid = value is not None and math.isfinite(value)
        self._value = float(value) if valid else None
        ok, shown = self._parse(self.edit.text())
        if not (ok and shown == self._value):
            self._show(self._value)
        self.set_invalid(False)

    def _show(self, value: float | None) -> None:
        text = "" if value is None else self._format(value)
        if text != self.edit.text():
            with QSignalBlocker(self.edit):
                self.edit.setText(text)
                if not self.edit.hasFocus():  # a number too long for the box shows its start
                    self.edit.home(False)

    def _format(self, value: float) -> str:
        return str(round(value)) if self._integer else f"{value:.{self._digits}g}"

    def _parse(self, text: str) -> tuple[bool, float | None]:
        if not text.strip():
            return self._optional, None
        value = parse_float(text)
        if value is None or (self._integer and not value.is_integer()):
            return False, None
        if self._minimum is not None and value < self._minimum:
            return False, None
        if self._maximum is not None and value > self._maximum:
            return False, None
        return True, value

    def _on_text(self, text: str) -> None:
        ok, value = self._parse(text)
        self.set_invalid(not ok)
        if ok:
            self._value = value
            self.valueEdited.emit(value)

    def _tidy(self) -> None:
        if not self.is_invalid():
            self._show(self._value)


# ---------------------------------------------------------------------- parameter rows
LINE_GAP = 2  # caption line to slider (rows too narrow for one line)
CAPTION_SCALE = 0.94
SYMBOLS = ("Δ₁₂", "E₀")  # the widest captions of the built-in models (symbols)
WIDEST_NUMBERS = (("-0000.00", "cm⁻¹"), ("0.0000", "10⁵ m/s"))  # (number, unit) to fit
VELOCITY_UNIT = "10⁵ m/s"
SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def subscript(*numbers: int) -> str:
    """``₁₂`` for (1, 2); ``₁,₁₀`` once a number has two digits."""
    texts = [str(n).translate(SUBSCRIPTS) for n in numbers]
    return ("," if any(n >= 10 for n in numbers) else "").join(texts)


@dataclass(frozen=True)
class Columns:
    """The columns every parameter row shares: caption, slider and number field (px)."""

    label: int
    value: int
    gap: int = 6
    slider: int = 56  # the narrowest slider of a one-line row

    def wide_width(self) -> int:
        """The narrowest row that holds caption, slider and field on one line."""
        return self.label + self.slider + self.value + 2 * self.gap


_COLUMNS: dict[str, Columns] = {}


def columns(widget: QWidget) -> Columns:
    """The shared columns for *widget*'s font: the number field fits an energy in any unit and
    the velocity with its unit, the caption column the symbols of the built-in parameters
    (longer names are elided). Caption, slider and field fit one line of the narrowest
    inspector."""
    key = widget.font().key()
    if key not in _COLUMNS:
        caption = QFontMetrics(scaled_font(widget, CAPTION_SCALE))
        mono = QFontMetrics(mono_font())
        unit = QFontMetrics(scaled_font(widget, UNIT_SCALE))
        room = max(mono.horizontalAdvance(n) + unit.horizontalAdvance(u) for n, u in WIDEST_NUMBERS)
        label = max(caption.horizontalAdvance(text) for text in SYMBOLS) + 4
        _COLUMNS[key] = Columns(label, room + FIELD_PADDING)
    return _COLUMNS[key]


def symmetric_range(half: float):
    """A slider range of ``[-half, half]``, widened for a value beyond it (signed values)."""

    def range_for(value: float) -> tuple[float, float]:
        end = max(half, nice_ceil(1.5 * abs(value))) if abs(value) > half else half
        return -end, end

    return range_for


def default_range(value: float) -> tuple[float, float]:
    """A slider range around *value*: from 0 to about twice it (-1 … 1 for 0)."""
    if value == 0 or not math.isfinite(value):
        return -1.0, 1.0
    end = nice_ceil(2 * abs(value))
    return (0.0, end) if value > 0 else (-end, 0.0)


def fixed_range(lo: float, hi: float):
    """A slider range of ``[lo, hi]``, widened for a value beyond it."""

    def range_for(value: float) -> tuple[float, float]:
        if value > hi:
            return lo, nice_ceil(1.5 * value)
        if value < lo:
            return (-nice_ceil(1.5 * abs(value)) if value < 0 else 0.0), hi
        return lo, hi

    return range_for


def nice_ceil(value: float) -> float:
    """The next 1, 2 or 5 times a power of ten at or above *value* (> 0)."""
    if not (math.isfinite(value) and value > 0):
        return 1.0
    power = 10.0 ** math.floor(math.log10(value))
    for step in (1, 2, 5, 10):
        if step * power >= value * (1 - 1e-12):
            return float(f"{step * power:.12g}")
    return 10 * power


class SliderMode(QObject):
    """The mode every model slider uses: Range, or Relative with a span in %. ``changed`` fires
    when it is switched (:meth:`menu`, or a slider's context menu). Its setting is a key:
    ``range``, ``relative-1``, ``relative-10`` or ``relative-50``."""

    changed = Signal(str, float)
    KEYS: ClassVar[dict[str, tuple[str, float]]] = {
        "range": (RANGE, 0.0),
        **{f"relative-{s:g}": (RELATIVE, s) for s in SPANS},
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode, self.span = RELATIVE, DEFAULT_SPAN

    def key(self) -> str:
        return "range" if self.mode == RANGE else f"relative-{self.span:g}"

    def set(self, mode: str, span: float) -> None:
        """Switch to *mode* (the range mode keeps the last relative span)."""
        span = self.span if mode == RANGE else float(span)
        if (mode, span) != (self.mode, self.span):
            self.mode, self.span = mode, span
            self.changed.emit(mode, span)

    def menu(self, parent: QWidget) -> QMenu:
        """The modes as checkable entries; a choice switches every slider."""
        menu = QMenu(parent)
        group = QActionGroup(menu)
        for key, (mode, span) in self.KEYS.items():
            action = menu.addAction(mode_text(mode, span))
            action.setCheckable(True)
            action.setActionGroup(group)
            action.setChecked(key == self.key())
            action.triggered.connect(lambda _on=False, m=mode, s=span: self.set(m, s))
        return menu

    def apply_to(self, slider: NudgeSlider) -> None:
        """Keep *slider* in this mode (and let its context menu switch every slider)."""
        slider.set_mode(self.mode, self.span)
        self.changed.connect(slider.set_mode)
        slider.modeChanged.connect(self.set)

    # settings protocol
    def settings_value(self) -> str:
        return self.key()

    def set_settings_value(self, value) -> bool:
        if value not in self.KEYS:
            return False
        self.set(*self.KEYS[value])
        return True


class ParamRow(QWidget):
    """One parameter on the shared :func:`columns`: caption (a symbol; *name* is its tooltip
    and the accessible name), slider and number field.

    The three sit on one line; only in a row narrower than :meth:`Columns.wide_width` do the
    caption and the field share the first line and the slider take a second, so nothing is
    squeezed. The field is right-aligned with its unit inside. The slider changes the value
    live (``valueEdited(value, True)`` while dragging or for each key press) and commits it
    once on release (``(value, False)``); typing in the field emits ``(value, False)``.

    In the range mode the slider spans *range_for(value)* (re-derived when a value falls
    outside it or the unit changes). The relative span is taken of at least *floor*, or of
    *floor_share* of the size of the last value set or typed (at 0: of that size), so a value
    can be dragged across 0.
    *nonnegative*: the slider never goes below 0 (the field still takes what *minimum*
    allows). :meth:`set_value` is silent.
    """

    valueEdited = Signal(float, bool)

    def __init__(
        self,
        caption: str,
        unit: str = "",
        *,
        name: str = "",
        digits: int = 6,
        integer: bool = False,
        minimum: float | None = None,
        maximum: float | None = None,
        mode: SliderMode | None = None,
        range_for=default_range,
        floor: float = 0.0,
        floor_share: float | None = None,
        nonnegative: bool = False,
        mono: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        name = name or caption
        self._range_for = range_for
        self._range_stale = True
        self._floor_share = floor_share
        self._scale = 1.0  # the size of the value (floor_share)
        self._label_width: int | None = None  # a caption column of its own (set_label_width)
        self._wide: bool | None = None
        self.caption = ElidedLabel(caption)
        self.caption.setProperty("kit", "muted")
        self.caption.setFont(mono_font(0.88) if mono else scaled_font(self, CAPTION_SCALE))
        self.caption.setToolTip(name)
        self.caption.setAccessibleName(name)
        self.field = NumberField(
            unit,
            name=f"{name} value",
            minimum=minimum,
            maximum=maximum,
            integer=integer,
            digits=digits,
        )
        self.caption.setBuddy(self.field.edit)
        self.slider = NudgeSlider()
        self.slider.setAccessibleName(name)
        self.slider.setToolTip(name)
        low = 0.0 if nonnegative and (minimum is None or minimum < 0) else minimum
        self.slider.set_bounds(low, maximum)
        self.slider.set_floor(floor)
        if integer:
            self.slider.set_step(1.0)
            self.slider.set_modes((RANGE,))
        elif mode is not None:
            mode.apply_to(self.slider)
        for child in (self.caption, self.field, self.slider):
            child.setParent(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.editingFinished.connect(
            lambda: self.valueEdited.emit(self.slider.value(), False)
        )
        self.field.valueEdited.connect(self._on_field)
        self._order_tabs()

    # --- values --------------------------------------------------------------------------
    def value(self) -> float | None:
        return self.field.value()

    def set_value(self, value: float) -> None:
        """Show *value* in the field and on the slider without emitting."""
        self.field.set_value(value)
        self.slider.set_value(value)
        self._fit_range(value)
        self._share_floor(value)

    def set_unit(self, text: str) -> None:
        if text != self.field.unit_label.text():
            self._range_stale = True
        self.field.set_unit(text)

    def set_floor(self, floor: float) -> None:
        self.slider.set_floor(floor)

    def set_label_width(self, width: int | None) -> None:
        """A caption column of *width* (e.g. the widest name of a card) instead of the shared
        symbol column; it gives way down to the symbol column so the slider keeps its least
        width. None: the shared column."""
        self._label_width = width
        self._place()

    def caption_width(self) -> int:
        """The caption column at the row's width (one line)."""
        c = columns(self)
        if self._label_width is None:
            return c.label
        room = self.width() - c.value - 2 * c.gap - c.slider
        return max(c.label, min(self._label_width, room))

    def set_range_for(self, range_for) -> None:
        self._range_for = range_for
        self._range_stale = True

    def _fit_range(self, value: float) -> None:
        lo, hi = self.slider.range()
        if self._range_stale or not lo <= value <= hi:
            self.slider.set_range(*self._range_for(value))
            self._range_stale = False

    def _share_floor(self, value: float) -> None:
        """The relative floor: *floor_share* of the value's scale (its last non-zero size),
        the whole scale at 0."""
        if self._floor_share is None or not math.isfinite(value):
            return
        if value:
            self._scale = abs(value)
        self.slider.set_floor(self._floor_share * self._scale if value else self._scale)

    def _on_slider(self, value: float) -> None:
        self.field.set_value(value)
        self.valueEdited.emit(value, True)  # committed by editingFinished

    def _on_field(self, value) -> None:
        if value is None:
            return
        self.slider.set_value(value)
        self._fit_range(value)
        self._share_floor(value)
        self.valueEdited.emit(float(value), False)

    # --- layout --------------------------------------------------------------------------
    def is_wide(self) -> bool:
        """Whether caption, slider and field share one line."""
        return self.width() >= columns(self).wide_width()

    def _height(self, wide: bool) -> int:
        return FIELD_HEIGHT if wide else FIELD_HEIGHT + LINE_GAP + self.slider.sizeHint().height()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._height(width >= columns(self).wide_width())

    def sizeHint(self) -> QSize:
        c = columns(self)
        return QSize(c.wide_width(), self._height(True))

    def minimumSizeHint(self) -> QSize:
        c = columns(self)
        return QSize(self.caption.minimumSizeHint().width() + c.gap + c.value, self._height(True))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place()

    def _place(self) -> None:
        c = columns(self)
        width = self.width()
        wide = self.is_wide()
        self.field.setGeometry(width - c.value, 0, c.value, FIELD_HEIGHT)
        slider_height = self.slider.sizeHint().height()
        if wide:
            label = self.caption_width()
            self.caption.setGeometry(0, 0, label, FIELD_HEIGHT)
            left = label + c.gap
            top = (FIELD_HEIGHT - slider_height) // 2
            self.slider.setGeometry(left, top, width - c.value - c.gap - left, slider_height)
        else:
            self.caption.setGeometry(0, 0, width - c.value - c.gap, FIELD_HEIGHT)
            self.slider.setGeometry(0, FIELD_HEIGHT + LINE_GAP, width, slider_height)
        if wide != self._wide:
            self._wide = wide
            self._order_tabs()
            self.updateGeometry()

    def _order_tabs(self) -> None:
        """Tab follows what is drawn: slider then field on one line, field then slider on two
        (in Qt's chain and in the child order, which the main window's Tab walks)."""
        if self._wide:
            QWidget.setTabOrder(self.slider, self.field.edit)
            self.field.raise_()
        else:
            QWidget.setTabOrder(self.field.edit, self.slider)
            self.slider.raise_()


class CodeBox(QWidget):
    """A monospaced multi-line editor in a rounded box; :meth:`mark` highlights an error line
    and underlines its column. ``textChanged`` fires on every edit."""

    textChanged = Signal(str)

    def __init__(self, placeholder: str = "", lines: int = 4):
        super().__init__()
        self.edit = QPlainTextEdit()
        self.edit.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self.edit.setPlaceholderText(placeholder)
        self.edit.setFont(mono_font())
        self.edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.edit.setTabChangesFocus(True)
        self.edit.setAccessibleName("Expression, one branch per line")
        palette = self.edit.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor(0, 0, 0, 0))
        self.edit.setPalette(palette)
        self.edit.viewport().setAutoFillBackground(False)
        height = self.edit.fontMetrics().lineSpacing() * lines + 14
        self.edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.edit.setFixedHeight(height)
        self.edit.setMinimumWidth(40)
        self._invalid = False
        self._error: tuple[int, int] | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.addWidget(self.edit)
        self.edit.installEventFilter(self)
        self.edit.textChanged.connect(lambda: self.textChanged.emit(self.text()))

    def text(self) -> str:
        return self.edit.toPlainText()

    def set_text(self, text: str) -> None:
        """Show *text* without emitting (the cursor keeps its place when it is unchanged)."""
        if text != self.text():
            with QSignalBlocker(self.edit):
                self.edit.setPlainText(text)

    def mark(self, line: int | None, column: int | None = None) -> None:
        """Highlight 1-based *line* and underline *column* (None clears the mark)."""
        self._invalid = line is not None
        self._error = (line, column or 1) if line is not None else None
        self._apply_mark()
        self.update()

    def is_invalid(self) -> bool:
        return self._invalid

    def _apply_mark(self) -> None:
        selections = []
        if self._error is not None:
            tokens = current_tokens()
            line, column = self._error
            block = self.edit.document().findBlockByNumber(line - 1)
            if block.isValid():
                whole = QTextEdit.ExtraSelection()
                whole.format.setBackground(tokens["err-soft"])
                whole.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
                whole.cursor = QTextCursor(block)
                selections.append(whole)
                char = QTextEdit.ExtraSelection()
                char.format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)
                char.format.setUnderlineColor(tokens["err"])
                cursor = QTextCursor(block)
                length = max(1, block.length() - 1)
                cursor.movePosition(
                    QTextCursor.MoveOperation.Right,
                    QTextCursor.MoveMode.MoveAnchor,
                    min(column - 1, length - 1),
                )
                cursor.movePosition(
                    QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, 1
                )
                char.cursor = cursor
                selections.append(char)
        self.edit.setExtraSelections(selections)

    def eventFilter(self, watched, event) -> bool:
        if event.type() in (QEvent.Type.FocusIn, QEvent.Type.FocusOut):
            self.update()
        return False

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._apply_mark()

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        painter.setBrush(tokens["surface"])
        if self._invalid:
            border = tokens["err"]
        elif self.edit.hasFocus():
            border = tokens["accent"]
        else:
            border = tokens["line-strong"]
        if self._invalid or self.edit.hasFocus():
            ring = QColor(border)
            ring.setAlphaF(0.25)
            painter.setPen(QPen(ring, 3))
            painter.drawRoundedRect(rect, 5, 5)
        painter.setPen(QPen(border, 1))
        painter.drawRoundedRect(rect, 5, 5)
        painter.end()


def paint_line_sample(painter: QPainter, rect: QRectF, color: str) -> None:
    """A short model curve as the map draws it: dashed in *color* over its dark shadow, so
    light colours show in both themes."""
    middle = rect.center().y()
    shadow = QPen(SHADOW, rect.height())
    shadow.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(shadow)
    inset = rect.height() / 2
    painter.drawLine(QPointF(rect.left() + inset, middle), QPointF(rect.right() - inset, middle))
    pen = QPen(QColor(color), 2)
    pen.setDashPattern([2.0, 1.5])
    painter.setPen(pen)
    painter.drawLine(QPointF(rect.left() + 2, middle), QPointF(rect.right() - 2, middle))


def color_icon(color: str, selected: bool = False, size: int = 16) -> QIcon:
    """The line sample of *color* (menu entries of the colour swatch); *selected* rings it."""
    icon = QIcon()
    for scale in (1.0, 2.0):
        pixmap = QPixmap(round(size * scale), round(size * scale))
        pixmap.setDevicePixelRatio(scale)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if selected:
            painter.setPen(QPen(current_tokens()["accent"], 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(0.75, 0.75, size - 1.5, size - 1.5), 4, 4)
        paint_line_sample(painter, QRectF(2.5, size / 2 - 3, size - 5, 6), color)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


class ColorSwatch(QAbstractButton):
    """The model colour as a short curve (as on the map); a click offers the other colours."""

    colorChosen = Signal(str)
    WIDTH, HEIGHT = 20, 22

    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = MODEL_COLORS[0]
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip("Colour of the curves")
        self.clicked.connect(self._choose)

    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        self._color = color
        name = dict(zip(MODEL_COLORS, COLOR_NAMES, strict=True)).get(color, color)
        self.setAccessibleName(f"Colour: {name}")
        self.update()

    def menu(self) -> QMenu:
        menu = QMenu(self)
        for color, name in zip(MODEL_COLORS, COLOR_NAMES, strict=True):
            chosen = color == self._color
            action = menu.addAction(color_icon(color, chosen), name)
            action.setCheckable(True)
            action.setChecked(chosen)
            if chosen:  # the style may not draw the check next to an icon
                font = QFont(menu.font())
                font.setWeight(QFont.Weight.DemiBold)
                action.setFont(font)
            action.triggered.connect(lambda _checked=False, c=color: self.colorChosen.emit(c))
        return menu

    def _choose(self) -> None:
        self.menu().exec(self.mapToGlobal(self.rect().bottomLeft()))

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.underMouse() or self.hasFocus():
            painter.setPen(QPen(tokens["accent"] if self.hasFocus() else tokens["line-strong"], 1))
            painter.setBrush(tokens["hover"])
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 5, 5)
        paint_line_sample(painter, QRectF(2, self.HEIGHT / 2 - 3, self.WIDTH - 4, 6), self._color)
        painter.end()

    def enterEvent(self, event) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.update()
        super().leaveEvent(event)


class CurveDot(QWidget):
    """A small dot in a picked curve's colour (rows of the fit mapping)."""

    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(12, 12)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(SHADOW, 1))
        painter.setBrush(self._color)
        painter.drawEllipse(QRectF(1, 1, 10, 10))
        painter.end()


class TableBox(QWidget):
    """A rounded box (line border) holding a grid of cells (``grid``) or, with *rows*, a
    column of row widgets (``rows``), as the file tables."""

    def __init__(self, parent=None, rows: bool = False):
        super().__init__(parent)
        if rows:
            self.rows = QVBoxLayout(self)
            self.rows.setContentsMargins(6, 6, 6, 6)
            self.rows.setSpacing(6)
        else:
            self.grid = QGridLayout(self)
            self.grid.setContentsMargins(6, 4, 4, 4)
            self.grid.setHorizontalSpacing(4)
            self.grid.setVerticalSpacing(3)

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(tokens["line"], 1))
        painter.setBrush(tokens["win"])
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
        painter.end()


def mono_label(text: str = "") -> QLabel:
    label = QLabel(text)
    font = QFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
    base = label.font().pointSizeF()
    if base > 0:
        font.setPointSizeF(base * 0.92)
    label.setFont(font)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def row_layout(*widgets: QWidget, spacing: int = 6) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(spacing)
    for widget in widgets:
        if widget is None:
            row.addStretch(1)
        else:
            row.addWidget(widget)
    return row
