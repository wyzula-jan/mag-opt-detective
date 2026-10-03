"""Small widgets shared by the rail panels, painted from the theme tokens (light and dark).

Section labels, notes with a status icon, link buttons, cards, drop zones, tags, number fields
with a unit, switch rows and spin boxes that ignore the wheel unless focused.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.gui import icons
from mag_opt_detective.gui.kit import Switch
from mag_opt_detective.gui.theme import current_tokens

LEVEL_ICONS = {"ok": "check", "warn": "triangle-alert", "err": "triangle-alert", "info": "info"}
LEVEL_COLOURS = {"ok": "ok", "warn": "warn", "err": "err", "info": "muted", "muted": "muted"}


def scaled_font(widget: QWidget, factor: float, bold: bool = False) -> QFont:
    font = QFont(widget.font())
    if font.pointSizeF() > 0:
        font.setPointSizeF(font.pointSizeF() * factor)
    font.setBold(bold)
    return font


def mono_font(factor: float = 0.92) -> QFont:
    """The system's fixed-pitch font, a little smaller than the UI font."""
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    base = QFont().pointSizeF()
    if base > 0:
        font.setPointSizeF(base * factor)
    return font


def section_label(text: str) -> QLabel:
    """A small uppercase heading (FIELD VALUES, ZERO FIELD, ...)."""
    label = QLabel(text.upper())
    label.setProperty("kit", "muted")
    font = scaled_font(label, 0.82, bold=True)
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 107)
    label.setFont(font)
    return label


def hint(text: str = "") -> QLabel:
    """A muted, wrapped explanation."""
    label = QLabel(text)
    label.setProperty("kit", "muted")
    label.setWordWrap(True)
    label.setFont(scaled_font(label, 0.94))
    return label


def dropped_paths(mime) -> list[str]:
    """Local files of a drop; a dropped folder gives its (visible) files."""
    files: list[str] = []
    if not mime.hasUrls():
        return files
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        path = Path(url.toLocalFile())
        if path.is_dir():
            files.extend(
                str(p) for p in sorted(path.iterdir()) if p.is_file() and not p.name.startswith(".")
            )
        elif path.is_file():
            files.append(str(path))
    return files


class WidthWatcher(QObject):
    """Calls *on_width(width)* whenever *widget* is resized (e.g. to shorten labels)."""

    def __init__(self, widget: QWidget, on_width: Callable[[int], object]):
        super().__init__(widget)
        self._on_width = on_width
        widget.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Resize:
            self._on_width(event.size().width())
        return False


def fit_segments(control, labels: dict[str, tuple[str, str]], width: int) -> None:
    """Use the long label of every option of *control* if they fit in *width*, else the short."""
    for value, (long, _short) in labels.items():
        control.button(value).setText(long)
    if control.sizeHint().width() > width:
        for value, (_long, short) in labels.items():
            control.button(value).setText(short)


def content_width(page: QWidget) -> int:
    """Width left for a panel's content in *page* (margins and a scroll bar taken off)."""
    scrollbar = page.style().pixelMetric(page.style().PixelMetric.PM_ScrollBarExtent)
    return page.width() - 28 - scrollbar


class FileDrops(QObject):
    """Makes *widget* accept dropped files (or folders) and hands their paths to *on_drop*."""

    def __init__(self, widget: QWidget, on_drop: Callable[[list[str]], object], hover=None):
        super().__init__(widget)
        self._on_drop = on_drop
        self._hover = hover
        widget.setAcceptDrops(True)
        widget.installEventFilter(self)

    def _set_hover(self, hover: bool) -> None:
        if self._hover is not None:
            self._hover(hover)

    def eventFilter(self, watched, event) -> bool:
        kind = event.type()
        if kind in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            if event.mimeData().hasUrls():
                event.setDropAction(Qt.DropAction.CopyAction)
                event.accept()
                self._set_hover(True)
            else:
                event.ignore()
            return True
        if kind == QEvent.Type.DragLeave:
            self._set_hover(False)
            return True
        if kind == QEvent.Type.Drop:
            self._set_hover(False)
            files = dropped_paths(event.mimeData())
            if not files:
                event.ignore()
                return True
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            self._on_drop(files)
            return True
        return False


class ElidedLabel(QLabel):
    """A one-line label that elides its text (at the right by default) instead of growing."""

    def __init__(self, text: str = "", parent=None, mode=Qt.TextElideMode.ElideRight):
        super().__init__(text, parent)
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self) -> QSize:
        return QSize(20, super().minimumSizeHint().height())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.setFont(self.font())
        rect = self.contentsRect()
        text = self.fontMetrics().elidedText(self.text(), self._mode, rect.width())
        painter.drawText(rect, int(self.alignment() | Qt.AlignmentFlag.AlignVCenter), text)
        painter.end()


class _TokenText(QLabel):
    """A wrapped label painted in a theme token's colour (read at paint time, so it follows
    theme switches whatever order the palette and the style sheet are applied in)."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.token = "muted"
        self.setWordWrap(True)

    def paintEvent(self, event) -> None:
        palette = QPalette(self.palette())
        palette.setColor(QPalette.ColorRole.WindowText, current_tokens()[self.token])
        painter = QPainter(self)
        flags = int(self.alignment()) | int(Qt.TextFlag.TextWordWrap)
        self.style().drawItemText(
            painter,
            self.contentsRect(),
            flags,
            palette,
            self.isEnabled(),
            self.text(),
            QPalette.ColorRole.WindowText,
        )
        painter.end()


class Note(QWidget):
    """A line of text with a status icon; levels ``muted`` (no icon), ``info``, ``ok``,
    ``warn`` and ``err``."""

    def __init__(self, text: str = "", level: str = "muted", parent=None):
        super().__init__(parent)
        self._level = level
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(14, 16)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label = _TokenText(text)
        self.text_label.setFont(scaled_font(self.text_label, 0.94))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.text_label, 1)
        self._apply()

    def text(self) -> str:
        return self.text_label.text()

    def level(self) -> str:
        return self._level

    def set_text(self, text: str, level: str = "muted") -> None:
        self.text_label.setText(text)
        if level != self._level:
            self._level = level
            self._apply()

    def _apply(self) -> None:
        self.text_label.token = LEVEL_COLOURS[self._level]
        self.text_label.update()
        name = LEVEL_ICONS.get(self._level)
        self.icon_label.setVisible(name is not None)
        if name is not None:
            ratio = self.devicePixelRatioF()
            color = LEVEL_COLOURS[self._level]
            self.icon_label.setPixmap(icons.pixmap(name, 13, color, scale=ratio))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self._apply()


class LinkButton(QToolButton):
    """A text button drawn as a link in the accent colour (Add…, Clear, …)."""

    def __init__(self, text: str, tooltip: str = "", parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip or text.rstrip("…"))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAutoRaise(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFont(scaled_font(self, 0.94))

    def sizeHint(self) -> QSize:
        metrics = self.fontMetrics()
        return QSize(metrics.horizontalAdvance(self.text()) + 10, metrics.height() + 6)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self.isEnabled() and (self.underMouse() or self.isDown()):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(tokens["accent-soft"])
            painter.drawRoundedRect(rect, 4, 4)
        if self.hasFocus():
            painter.setPen(QPen(tokens["accent"], 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(tokens["accent"] if self.isEnabled() else tokens["faint"])
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()


class Card(QWidget):
    """A rounded surface with a line border; holds a vertical layout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(1, 1, 1, 1)
        self.box.setSpacing(0)

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(tokens["line"], 1))
        painter.setBrush(tokens["surface"])
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
        painter.end()


class Divider(QWidget):
    """A one-pixel line in the theme's line colour (between rows of a card)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), current_tokens()["line"])
        painter.end()


class DropZone(QWidget):
    """A dashed box that takes dropped files or folders; a click emits :attr:`clicked`."""

    clicked = Signal()
    filesDropped = Signal(list)

    def __init__(self, text: str, icon: str | None = "upload", parent=None):
        super().__init__(parent)
        self._hover = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setAccessibleName(text)
        self.icon_name = icon
        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setVisible(icon is not None)
        self.text_label = hint(text)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)
        FileDrops(self, self.filesDropped.emit, hover=self._set_hover)
        self._update_icon()

    def set_text(self, text: str) -> None:
        self.text_label.setText(text)
        self.setAccessibleName(text)

    def _set_hover(self, hover: bool) -> None:
        self._hover = hover
        self.update()

    def _update_icon(self) -> None:
        if self.icon_name is not None:
            ratio = self.devicePixelRatioF()
            self.icon_label.setPixmap(icons.pixmap(self.icon_name, 16, "muted", scale=ratio))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._update_icon()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.clicked.emit()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        hot = self._hover or self.hasFocus()
        pen = QPen(tokens["accent"] if hot else tokens["line-strong"], 1.5)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(tokens["accent-soft"] if self._hover else Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 8, 8)
        painter.end()


class Tag(QLabel):
    """A small label on a sunken pill (Before, After, ...)."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setProperty("kit", "muted")
        self.setFont(scaled_font(self, 0.82, bold=True))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setContentsMargins(6, 1, 6, 1)

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(tokens["line"], 1))
        painter.setBrush(tokens["sunken"])
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)
        painter.end()
        super().paintEvent(event)


class UnitField(QWidget):
    """A frameless number field and its unit inside one rounded box (as in the mockup).

    The box shows focus in the accent colour and an invalid value in the error colour.
    """

    def __init__(self, edit: QLineEdit, unit: str = "", parent=None):
        super().__init__(parent)
        self.edit = edit
        self._invalid = False
        edit.setFrame(False)
        edit.setFont(mono_font())
        edit.setMinimumWidth(30)
        edit.setAutoFillBackground(False)
        palette = edit.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor(0, 0, 0, 0))
        edit.setPalette(palette)
        self.unit_label = QLabel(unit)
        self.unit_label.setProperty("kit", "muted")
        self.unit_label.setFont(scaled_font(self.unit_label, 0.88))
        self.unit_label.setVisible(bool(unit))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 7, 2)
        layout.setSpacing(3)
        layout.addWidget(edit, 1)
        layout.addWidget(self.unit_label)
        edit.installEventFilter(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_unit(self, text: str) -> None:
        self.unit_label.setText(text)
        self.unit_label.setVisible(bool(text))

    def set_invalid(self, invalid: bool) -> None:
        if invalid != self._invalid:
            self._invalid = invalid
            self.edit.setProperty("invalid", invalid)
            self.update()

    def is_invalid(self) -> bool:
        return self._invalid

    def eventFilter(self, watched, event) -> bool:
        if event.type() in (QEvent.Type.FocusIn, QEvent.Type.FocusOut):
            self.update()
        return False

    def mousePressEvent(self, event) -> None:
        self.edit.setFocus(Qt.FocusReason.MouseFocusReason)
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        painter.setBrush(tokens["surface"] if self.isEnabled() else tokens["sunken"])
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


class SwitchRow(QWidget):
    """A bold title and a short description left of a :class:`Switch`; a click toggles it."""

    def __init__(self, title: str, description: str = "", parent=None):
        super().__init__(parent)
        self.switch = Switch()
        self.switch.setAccessibleName(title)
        self.title_label = QLabel(title)
        self.title_label.setFont(scaled_font(self.title_label, 0.98, bold=True))
        self.description_label = hint(description)
        self.description_label.setVisible(bool(description))
        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(1)
        text.addWidget(self.title_label)
        text.addWidget(self.description_label)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(text, 1)
        layout.addWidget(self.switch, 0, Qt.AlignmentFlag.AlignTop)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.switch.isEnabled()
            and self.rect().contains(event.position().toPoint())
        ):
            self.switch.toggle()
        super().mouseReleaseEvent(event)


class SpinBox(QSpinBox):
    """A whole-number field drawn as the kit's number fields (``kit="field"``: a rounded box,
    no arrows; the arrow keys still step it). It ignores the mouse wheel unless it has the
    focus (panels scroll)."""

    def __init__(self, minimum: int, maximum: int, value: int, step: int = 1, parent=None):
        super().__init__(parent)
        self.setProperty("kit", "field")
        self.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.setFont(mono_font())
        self.setRange(minimum, maximum)
        self.setSingleStep(step)
        self.setValue(value)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumWidth(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class CheckBox(QCheckBox):
    """A tick box as the mockup's ``.cb``: filled with the accent colour and a tick when
    checked, a line box when not; text, if any, follows it."""

    SIZE = 15
    GAP = 6

    def sizeHint(self) -> QSize:
        size = self.SIZE + 2
        if not self.text():
            return QSize(size, size)
        metrics = self.fontMetrics()
        width = size + self.GAP + metrics.horizontalAdvance(self.text())
        return QSize(width, max(size, metrics.height()))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def hitButton(self, pos) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        top = (self.height() - self.SIZE) / 2
        box = QRectF(1.5, top + 0.5, self.SIZE - 1, self.SIZE - 1)
        checked, enabled = self.isChecked(), self.isEnabled()
        if checked:
            fill = tokens["accent"] if enabled else tokens["line-strong"]
            border = fill
        else:
            fill = tokens["surface"] if enabled else tokens["sunken"]
            border = tokens["line-strong"] if enabled else tokens["line"]
        if self.hasFocus():
            ring = QColor(tokens["accent"])
            ring.setAlphaF(0.35)
            painter.setPen(QPen(ring, 3))
            painter.drawRoundedRect(box, 3, 3)
        painter.setPen(QPen(border, 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(box, 3, 3)
        if checked:
            pen = QPen(tokens["accent-fg"], 1.8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            left, bottom = box.left(), box.bottom()
            painter.drawPolyline(
                [
                    QPointF(left + 3.5, box.center().y() + 0.5),
                    QPointF(left + 6.2, bottom - 3.6),
                    QPointF(box.right() - 3.2, box.top() + 4.0),
                ]
            )
        if self.text():
            painter.setPen(tokens["fg"] if enabled else tokens["faint"])
            text_rect = self.rect().adjusted(self.SIZE + 2 + self.GAP, 0, 0, 0)
            painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignVCenter), self.text())
        painter.end()


def small_button(text: str, icon: str | None = None, tooltip: str = "") -> QToolButton:
    """A compact text button (Plot, Save current map, ...) with an optional icon."""
    button = QToolButton()
    button.setProperty("kit", "button")
    button.setText(text)
    button.setToolTip(tooltip)
    button.setFont(scaled_font(button, 0.94))
    if icon is not None:
        icons.set_icon(button, icon)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    else:
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    button.setIconSize(QSize(14, 14))
    return button


def labelled(text: str, widget: QWidget) -> QWidget:
    """*widget* with a small muted label above it."""
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(3)
    label = QLabel(text)
    label.setProperty("kit", "muted")
    label.setFont(scaled_font(label, 0.88))
    label.setBuddy(widget if not isinstance(widget, UnitField) else widget.edit)
    layout.addWidget(label)
    layout.addWidget(widget)
    return box


def clear_layout(layout) -> None:
    """Remove every item of *layout*; its widgets disappear at once and are deleted later."""
    while layout.count():
        widget = layout.takeAt(0).widget()
        if widget is not None:
            widget.hide()
            widget.deleteLater()


def panel_layout(widget: QWidget, spacing: int = 16) -> QVBoxLayout:
    """The body layout of a rail panel (margins as in the mockup)."""
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(14, 12, 14, 16)
    layout.setSpacing(spacing)
    return layout


def block(*widgets: QWidget, spacing: int = 7) -> QWidget:
    """A group of widgets stacked closely (one topic of a panel)."""
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for widget in widgets:
        layout.addWidget(widget)
    return box
