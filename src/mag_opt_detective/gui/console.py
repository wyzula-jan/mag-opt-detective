"""The log: a console in a drawer below the plot and the Log button in the status bar."""

from __future__ import annotations

import html
import logging
import time
from collections import deque

from PySide6.QtCore import QObject, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QFontDatabase, QIcon, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QStyle,
    QStyleOptionToolButton,
    QStylePainter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.gui import icons
from mag_opt_detective.gui.theme import current_tokens
from mag_opt_detective.gui.widgets import Separator

logger = logging.getLogger("mag_opt_detective")

MAX_LINES = 5000


def level_token(level: int) -> str:
    """Colour token of a log line: errors in the error colour, warnings in the warning one."""
    if level >= logging.ERROR:
        return "err"
    return "warn" if level >= logging.WARNING else "fg"


class ConsoleWidget(QPlainTextEdit):
    """The log lines: the time dimmed, errors and warnings in their colours (as the mockup's
    ``.log``). :meth:`recolor` paints them again in the current theme."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(MAX_LINES)
        self.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self._lines: deque[tuple[str, str, int]] = deque(maxlen=MAX_LINES)

    def append_line(self, stamp: str, text: str, level: int = logging.INFO) -> None:
        self._lines.append((stamp, text, level))
        self.appendHtml(self._html(stamp, text, level))

    def recolor(self) -> None:
        self.clear()
        for line in self._lines:
            self.appendHtml(self._html(*line))

    @staticmethod
    def _html(stamp: str, text: str, level: int) -> str:
        tokens = current_tokens()
        return (
            f"<span style='white-space:pre-wrap'>"
            f"<span style='color:{tokens['faint'].name()}'>{stamp}</span>  "
            f"<span style='color:{tokens[level_token(level)].name()}'>{html.escape(text)}</span>"
            "</span>"
        )


class _Emitter(QObject):
    message = Signal(str, int)  # text, level number
    line = Signal(str, str, int)  # time, text, level number


class QtLogHandler(logging.Handler):
    """Forwards log records to a :class:`ConsoleWidget` through a queued Qt signal.

    :attr:`errors` counts the records of level ERROR and above.
    """

    def __init__(self, console: ConsoleWidget):
        super().__init__()
        self.errors = 0
        self._emitter = _Emitter()
        self.message = self._emitter.message
        self._emitter.line.connect(console.append_line)
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
            text = self.format(record)
            if record.levelno >= logging.WARNING:  # also said in words, for copied lines
                text = f"{text}  [{record.levelname.lower()}]"
            if record.levelno >= logging.ERROR:
                self.errors += 1
            self._emitter.line.emit(stamp, text, record.levelno)
            self._emitter.message.emit(f"{stamp}  {text}", record.levelno)
        except RuntimeError:  # console already deleted
            pass


class Badge(QWidget):
    """A small pill with a number in the error colour; hidden at zero."""

    HEIGHT = 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self._count = 0
        self.setFixedHeight(self.HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()

    def count(self) -> int:
        return self._count

    def set_count(self, count: int) -> None:
        self._count = count
        self.setFixedWidth(self.sizeHint().width())
        self.setVisible(count > 0)
        self.update()

    def _text(self) -> str:
        return str(self._count) if self._count < 100 else "99+"

    def sizeHint(self) -> QSize:
        width = max(self.HEIGHT, self.fontMetrics().horizontalAdvance(self._text()) + 8)
        return QSize(width, self.HEIGHT)

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        rect = QRectF(self.rect())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(tokens["err"])
        painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(tokens["surface"])
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._text())
        painter.end()


class LogButton(QToolButton):
    """Status-bar button that opens the log; a pill after its text counts the unseen errors
    (inside the button, as the mockup's ``.st-log .n``)."""

    BADGE_GAP = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("kit", "tool")
        self.setCheckable(True)
        self.setText("Log")
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setToolTip("Show or hide the log")
        icons.set_icon(self, "terminal", "muted", on_color="accent")
        self.badge = Badge(self)

    def unseen(self) -> int:
        return self.badge.count()

    def set_unseen(self, errors: int) -> None:
        self.badge.set_count(errors)
        self.setAccessibleName(f"Log, {errors} new errors" if errors else "Log")
        self.updateGeometry()
        self._place_badge()

    def _room(self) -> int:
        """Width kept after the text for the pill (0 without one)."""
        return 0 if self.badge.isHidden() else self.badge.width() + self.BADGE_GAP

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(hint.width() + self._room(), hint.height())

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        """The button as usual, its icon and text moved left of the pill."""
        option = QStyleOptionToolButton()
        self.initStyleOption(option)
        painter = QStylePainter(self)
        room = self._room()
        if not room:
            painter.drawComplexControl(QStyle.ComplexControl.CC_ToolButton, option)
            return
        label = QStyleOptionToolButton(option)
        option.text, option.icon = "", QIcon()
        painter.drawComplexControl(QStyle.ComplexControl.CC_ToolButton, option)
        label.rect = option.rect.adjusted(0, 0, -room, 0)
        painter.drawControl(QStyle.ControlElement.CE_ToolButtonLabel, label)

    def _place_badge(self) -> None:
        badge = self.badge
        x = self.width() - badge.width() - self.BADGE_GAP
        badge.move(x, (self.height() - badge.height()) // 2)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_badge()


class LogDrawer(QWidget):
    """Header (title, close button) and the console."""

    def __init__(self, parent=None):
        super().__init__(parent)
        head = QWidget()
        row = QHBoxLayout(head)
        row.setContentsMargins(12, 4, 6, 4)
        icon = QLabel()
        icon.setPixmap(icons.pixmap("terminal", 14, "muted", self.devicePixelRatioF()))
        title = QLabel("Log")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        self.close_button = QToolButton()
        self.close_button.setProperty("kit", "tool")
        self.close_button.setToolTip("Close the log")
        self.close_button.setAccessibleName("Close log")
        icons.set_icon(self.close_button, "x", "muted")
        row.addWidget(icon)
        row.addWidget(title, stretch=1)
        row.addWidget(self.close_button)
        self.console = ConsoleWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(head)
        layout.addWidget(Separator())
        layout.addWidget(self.console, stretch=1)


def install(window) -> None:
    """Put the console into the log drawer and the Log button into the status bar."""
    drawer = LogDrawer()
    window.log_panel.content().layout().addWidget(drawer)
    window.log_drawer = drawer
    window.console = drawer.console
    handler = QtLogHandler(drawer.console)
    window.log_handler = handler
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    button = LogButton()
    window.log_button = button
    window.statusBar().addPermanentWidget(button)
    window.themeChanged.connect(drawer.console.recolor)
    seen = [0]

    def sync(open_: bool) -> None:
        if open_:
            seen[0] = handler.errors
        button.setChecked(open_)
        button.set_unseen(0 if open_ else handler.errors - seen[0])

    button.toggled.connect(lambda checked: window.log_panel.set_open(checked))
    drawer.close_button.clicked.connect(lambda: window.log_panel.set_open(False))
    window.log_panel.openChanged.connect(sync)
    handler.message.connect(lambda _text, _level: sync(window.log_panel.is_open()))
    sync(window.log_panel.is_open())
