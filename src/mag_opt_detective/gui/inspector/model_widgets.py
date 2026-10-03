"""Small widgets of the model cards: number fields, slider rows, the expression editor, the
colour swatch and compact grids. They are painted from the theme tokens (light and dark)."""

from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QRectF, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
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
    QSlider,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.gui import icons
from mag_opt_detective.gui.inspector.model_state import COLOR_NAMES, MODEL_COLORS
from mag_opt_detective.gui.panels.common import UnitField, mono_font, scaled_font
from mag_opt_detective.gui.theme import current_tokens
from mag_opt_detective.gui.widgets import parse_float

SHADOW = QColor(10, 6, 14, 150)  # outline of the colour dots, as the curve chips


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


def compact(field: UnitField) -> UnitField:
    """Let a field box shrink with a narrow inspector (cells of the model tables)."""
    field.edit.setMinimumWidth(14)
    field.layout().setContentsMargins(4, 2, 5, 2)
    return field


class NumberField(UnitField):
    """A number (C locale, scientific notation allowed) in a rounded box with its unit.

    ``valueEdited`` fires while the user types a valid number (None for an empty *optional*
    field); invalid text marks the box. :meth:`set_value` never emits.
    """

    valueEdited = Signal(object)

    def __init__(
        self,
        unit: str = "",
        *,
        name: str = "",
        optional: bool = False,
        minimum: float | None = None,
        integer: bool = False,
        digits: int = 6,
        placeholder: str = "",
    ):
        edit = QLineEdit()
        super().__init__(edit, unit)
        compact(self)
        self._optional = optional
        self._minimum = minimum
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
        """Show *value* without emitting (None or an infinite bound shows empty)."""
        valid = value is not None and math.isfinite(value)
        self._value = float(value) if valid else None
        text = "" if not valid else self._format(value)
        if text != self.edit.text():
            with QSignalBlocker(self.edit):
                self.edit.setText(text)
        self.set_invalid(False)

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
        return True, value

    def _on_text(self, text: str) -> None:
        ok, value = self._parse(text)
        self.set_invalid(not ok)
        if ok:
            self._value = value
            self.valueEdited.emit(value)

    def _tidy(self) -> None:
        if not self.is_invalid():
            self.set_value(self._value)


class _Slider(QSlider):
    """A slider that the mouse wheel turns only while it has the focus (the inspector
    scrolls)."""

    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class SliderField(QWidget):
    """A caption, then a slider for quick live changes and a number field (as in the mockup).

    The slider spans ``[lo, hi]``; the field also takes values beyond it (the slider then sits
    at its end). ``valueChanged`` fires on user edits only.
    """

    valueChanged = Signal(float)
    STEPS = 1000

    def __init__(
        self,
        caption: str,
        lo: float,
        hi: float,
        unit: str = "",
        *,
        integer: bool = False,
        minimum: float | None = None,
        digits: int = 6,
    ):
        super().__init__()
        self._lo, self._hi = lo, hi
        self._integer = integer
        self._value = lo
        self.caption = muted_label(caption, 0.94)
        self.slider = _Slider(Qt.Orientation.Horizontal)
        self.slider.setAccessibleName(caption)
        self.slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.field = NumberField(
            unit, name=f"{caption} value", minimum=minimum, integer=integer, digits=digits
        )
        self.field.setFixedWidth(92)
        self.slider.setMinimumWidth(40)
        self._set_slider_range()
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(3)
        grid.addWidget(self.caption, 0, 0, 1, 2)
        grid.addWidget(self.slider, 1, 0)
        grid.addWidget(self.field, 1, 1)
        grid.setColumnStretch(0, 1)
        self.slider.valueChanged.connect(self._on_slider)
        self.field.valueEdited.connect(self._on_field)

    def value(self) -> float:
        return self._value

    def set_unit(self, unit: str) -> None:
        self.field.set_unit(unit)

    def set_range(self, lo: float, hi: float) -> None:
        """The slider's span (the value is kept)."""
        self._lo, self._hi = lo, hi
        self._set_slider_range()
        self._place_slider()

    def set_value(self, value: float) -> None:
        """Show *value* without emitting."""
        self._value = float(value)
        self.field.set_value(self._value)
        self._place_slider()

    def _set_slider_range(self) -> None:
        with QSignalBlocker(self.slider):
            if self._integer:
                self.slider.setRange(round(self._lo), round(self._hi))
            else:
                self.slider.setRange(0, self.STEPS)

    def _place_slider(self) -> None:
        with QSignalBlocker(self.slider):
            if self._integer:
                self.slider.setValue(round(self._value))
            elif self._hi > self._lo:
                t = (self._value - self._lo) / (self._hi - self._lo)
                self.slider.setValue(round(min(max(t, 0.0), 1.0) * self.STEPS))

    def _on_slider(self, position: int) -> None:
        if self._integer:
            value = float(position)
        else:
            value = self._lo + (self._hi - self._lo) * position / self.STEPS
            step = (self._hi - self._lo) / self.STEPS
            value = round(value / step) * step if step > 0 else value
            value = float(f"{value:.4g}")
        self._value = value
        self.field.set_value(value)
        self.valueChanged.emit(value)

    def _on_field(self, value) -> None:
        if value is None:
            return
        self._value = float(value)
        self._place_slider()
        self.valueChanged.emit(self._value)


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


CHIP = QColor(34, 26, 40)  # the dark chip a model's dashed line is shown on (as on a map)


def paint_line_chip(painter: QPainter, rect: QRectF, color: str, border: QColor) -> None:
    """A dark rounded chip with a dashed line in *color*: how the model's curves look."""
    painter.setPen(QPen(border, 1))
    painter.setBrush(CHIP)
    painter.drawRoundedRect(rect, 3, 3)
    pen = QPen(QColor(color), 2)
    pen.setDashPattern([2.0, 1.5])
    painter.setPen(pen)
    middle = rect.center().y()
    painter.drawLine(rect.left() + 3, middle, rect.right() - 3, middle)


def color_icon(color: str, size: int = 16) -> QIcon:
    """The line chip of *color* (menu entries of the colour swatch)."""
    icon = QIcon()
    for scale in (1.0, 2.0):
        pixmap = QPixmap(round(size * scale), round(size * scale))
        pixmap.setDevicePixelRatio(scale)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_line_chip(painter, QRectF(0.5, 3.5, size - 1, size - 7), color, SHADOW)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


class ColorSwatch(QAbstractButton):
    """The model colour as a dashed line on a dark chip; a click offers the other colours."""

    colorChosen = Signal(str)
    WIDTH, HEIGHT = 26, 22

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
            action = menu.addAction(color_icon(color), name)
            action.setCheckable(True)
            action.setChecked(color == self._color)
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
        chip = QRectF(4.5, 6.5, self.WIDTH - 9, self.HEIGHT - 13)
        paint_line_chip(painter, chip, self._color, tokens["line-strong"])
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
