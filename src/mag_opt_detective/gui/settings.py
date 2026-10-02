"""Remembering the user's choices between sessions (QSettings)."""

from __future__ import annotations

import logging

from PySide6.QtCore import QByteArray, QSettings
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QAbstractButton, QComboBox, QLineEdit, QSpinBox, QWidget

logger = logging.getLogger(__name__)

ORGANIZATION = "mag-opt-detective"
APPLICATION = "mag-opt-detective"
PREFIX = "v1"  # bump when stored keys change meaning


def default_settings() -> QSettings:
    """The user's settings store (registry, plist or ini file depending on the OS)."""
    return QSettings(ORGANIZATION, APPLICATION)


def _read(widget: QWidget):
    if isinstance(widget, QAbstractButton):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QSpinBox):
        return widget.value()
    if isinstance(widget, QLineEdit):
        return widget.text()
    raise TypeError(f"cannot store {type(widget).__name__}")


def _apply(widget: QWidget, value) -> bool:
    """Set *value* on *widget*; returns False (and changes nothing) if it is not valid."""
    if isinstance(widget, QAbstractButton):
        if isinstance(value, str):
            value = value.lower() == "true"
        if not isinstance(value, bool):
            return False
        # radio buttons: only setting True is meaningful, the group unchecks the others
        if value or not widget.autoExclusive():
            widget.setChecked(value)
        return True
    if isinstance(widget, QComboBox):
        index = widget.findText(str(value))
        if index < 0:
            return False
        widget.setCurrentIndex(index)
        return True
    if isinstance(widget, QSpinBox):
        try:
            number = int(value)
        except (TypeError, ValueError):
            return False
        if not widget.minimum() <= number <= widget.maximum():
            return False
        widget.setValue(number)
        return True
    if isinstance(widget, QLineEdit):
        text = str(value)
        validator = widget.validator()
        if text and validator is not None:
            state, *_ = validator.validate(text, 0)
            if state != QValidator.State.Acceptable:
                return False
        widget.setText(text)
        return True
    return False


class Persistence:
    """Saves and restores registered widgets under ``v1/<key>``."""

    def __init__(self, settings: QSettings):
        self.settings = settings
        self._widgets: dict[str, QWidget] = {}
        self._defaults: dict[str, object] = {}

    def bind(self, key: str, widget: QWidget) -> None:
        self._widgets[key] = widget
        self._defaults[key] = _read(widget)

    def save(self) -> None:
        for key, widget in self._widgets.items():
            self.settings.setValue(f"{PREFIX}/{key}", _read(widget))
        self.settings.sync()

    def restore(self) -> None:
        for key, widget in self._widgets.items():
            value = self.settings.value(f"{PREFIX}/{key}")
            if value is not None and not _apply(widget, value):
                logger.debug("Ignoring invalid stored setting %s=%r", key, value)

    def reset(self) -> None:
        """Forget everything stored and go back to the built-in defaults."""
        self.settings.remove(PREFIX)
        self.settings.sync()
        for key, widget in self._widgets.items():
            _apply(widget, self._defaults[key])

    # raw values (window geometry, splitter state, last folder) ----------------
    def set_value(self, key: str, value) -> None:
        self.settings.setValue(f"{PREFIX}/{key}", value)

    def value(self, key: str, default=None):
        return self.settings.value(f"{PREFIX}/{key}", default)

    def bytes_value(self, key: str) -> QByteArray | None:
        value = self.value(key)
        return value if isinstance(value, QByteArray) and not value.isEmpty() else None
