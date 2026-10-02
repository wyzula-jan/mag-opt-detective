"""Console panel that shows the application log."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QPlainTextEdit


class ConsoleWidget(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))


class _Emitter(QObject):
    message = Signal(str)


class QtLogHandler(logging.Handler):
    """Forwards log records to a :class:`ConsoleWidget` through a queued Qt signal."""

    def __init__(self, console: ConsoleWidget):
        super().__init__()
        self._emitter = _Emitter()
        self._emitter.message.connect(console.appendPlainText)
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
            if record.levelno >= logging.WARNING:
                text = f"{text}  [{record.levelname.lower()}]"
            self._emitter.message.emit(text)
        except RuntimeError:  # console already deleted
            pass
