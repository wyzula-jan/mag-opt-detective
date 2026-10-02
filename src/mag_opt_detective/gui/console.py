"""The log: a console in a drawer below the plot and the Log button in the status bar."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QFontDatabase, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.gui import icons
from mag_opt_detective.gui.theme import current_tokens
from mag_opt_detective.gui.widgets import Separator

logger = logging.getLogger("mag_opt_detective")


class ConsoleWidget(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.setFrameShape(QPlainTextEdit.Shape.NoFrame)


class _Emitter(QObject):
    message = Signal(str, int)  # text, level number


class QtLogHandler(logging.Handler):
    """Forwards log records to a :class:`ConsoleWidget` through a queued Qt signal.

    :attr:`errors` counts the records of level ERROR and above.
    """

    def __init__(self, console: ConsoleWidget):
        super().__init__()
        self.errors = 0
        self._emitter = _Emitter()
        self.message = self._emitter.message
        self._emitter.message.connect(lambda text, _level: console.appendPlainText(text))
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
            if record.levelno >= logging.WARNING:
                text = f"{text}  [{record.levelname.lower()}]"
            if record.levelno >= logging.ERROR:
                self.errors += 1
            self._emitter.message.emit(text, record.levelno)
        except RuntimeError:  # console already deleted
            pass


class Badge(QWidget):
    """A small pill with a number in the error colour; hidden at zero."""

    HEIGHT = 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self._count = 0
        self.hide()

    def count(self) -> int:
        return self._count

    def set_count(self, count: int) -> None:
        self._count = count
        self.setVisible(count > 0)
        self.updateGeometry()
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


class LogButton(QWidget):
    """Status-bar button that opens the log, with a badge counting the unseen errors."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.button = QToolButton()
        self.button.setProperty("kit", "tool")
        self.button.setCheckable(True)
        self.button.setText("Log")
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setToolTip("Show or hide the log")
        icons.set_icon(self.button, "terminal", "muted", on_color="accent")
        self.badge = Badge()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 4, 0)
        row.setSpacing(2)
        row.addWidget(self.button)
        row.addWidget(self.badge)
        self.toggled = self.button.toggled

    def unseen(self) -> int:
        return self.badge.count()

    def set_unseen(self, errors: int) -> None:
        self.badge.set_count(errors)
        self.button.setAccessibleName(f"Log, {errors} new errors" if errors else "Log")

    def isChecked(self) -> bool:
        return self.button.isChecked()

    def setChecked(self, checked: bool) -> None:
        self.button.setChecked(checked)


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
