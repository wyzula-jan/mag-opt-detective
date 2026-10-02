"""Segmented control: exclusive options shown as one group of buttons."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QSizePolicy, QToolButton, QWidget

from mag_opt_detective.gui import icons

SIZES = ("md", "sm", "xs")


class SegmentedControl(QWidget):
    """Exclusive checkable tool buttons styled as one segmented group.

    The first option added is selected. ``valueChanged`` fires whenever the selection changes,
    programmatically or by the user. Only the selected segment is in the Tab order; arrow keys
    (and Home/End) move the selection. Settings protocol: the value string.
    """

    valueChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None, *, size: str = "md", expand: bool = False):
        super().__init__(parent)
        if size not in SIZES:
            raise ValueError(f"size must be one of {SIZES}")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("kit", "segmented")
        self.setProperty("kitSize", size)
        self._expand = expand
        self._buttons: dict[str, QToolButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.buttonToggled.connect(self._on_toggled)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        horizontal = QSizePolicy.Policy.Expanding if expand else QSizePolicy.Policy.Fixed
        self.setSizePolicy(horizontal, QSizePolicy.Policy.Fixed)

    # --- options -----------------------------------------------------------------------
    def add_option(
        self, value: str, text: str, tooltip: str | None = None, icon: str | None = None
    ) -> QToolButton:
        """Append an option; *icon* is an icon name from ``gui.icons``."""
        if value in self._buttons:
            raise ValueError(f"duplicate option {value!r}")
        button = QToolButton(self)
        button.setProperty("kit", "segment")
        button.setCheckable(True)
        button.setText(text)
        if tooltip:
            button.setToolTip(tooltip)
        if icon:
            icons.set_icon(button, icon)
            style = Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            button.setToolButtonStyle(style if text else Qt.ToolButtonStyle.ToolButtonIconOnly)
        else:
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.setAccessibleName(text or tooltip or value)
        if self._expand:
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        button.installEventFilter(self)
        self._buttons[value] = button
        self._group.addButton(button)
        self.layout().addWidget(button)
        if len(self._buttons) == 1:
            button.setChecked(True)
        self._update_focus()
        return button

    def options(self) -> list[str]:
        return list(self._buttons)

    def button(self, value: str) -> QToolButton:
        return self._buttons[value]

    def set_option_enabled(self, value: str, enabled: bool) -> None:
        self._buttons[value].setEnabled(enabled)
        self._update_focus()

    # --- value -------------------------------------------------------------------------
    def value(self) -> str:
        for value, button in self._buttons.items():
            if button.isChecked():
                return value
        return ""

    def set_value(self, value: str) -> None:
        if value not in self._buttons:
            raise ValueError(f"unknown option {value!r}")
        self._buttons[value].setChecked(True)

    def _on_toggled(self, button: QToolButton, checked: bool) -> None:
        if not checked:
            return
        had_focus = any(b.hasFocus() for b in self._buttons.values())
        self._update_focus()
        if had_focus:
            button.setFocus(Qt.FocusReason.OtherFocusReason)
        self.valueChanged.emit(self.value())

    def _update_focus(self) -> None:
        """Roving focus: only the selected (or first enabled) segment takes Tab focus."""
        enabled = [b for b in self._buttons.values() if b.isEnabled()]
        checked = [b for b in enabled if b.isChecked()]
        target = (checked or enabled or [None])[0]
        for button in self._buttons.values():
            policy = Qt.FocusPolicy.StrongFocus if button is target else Qt.FocusPolicy.NoFocus
            button.setFocusPolicy(policy)
        self.setFocusProxy(target)

    # --- keyboard ----------------------------------------------------------------------
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        is_key = event.type() == QEvent.Type.KeyPress
        if is_key and watched in self._buttons.values() and self._on_key(watched, event):
            return True
        return super().eventFilter(watched, event)

    def _on_key(self, button: QToolButton, event: QKeyEvent) -> bool:
        enabled = [b for b in self._buttons.values() if b.isEnabled()]
        if not enabled:
            return False
        index = enabled.index(button) if button in enabled else 0
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            index = max(0, index - 1)
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            index = min(len(enabled) - 1, index + 1)
        elif key == Qt.Key.Key_Home:
            index = 0
        elif key == Qt.Key.Key_End:
            index = len(enabled) - 1
        else:
            return False
        enabled[index].setChecked(True)
        enabled[index].setFocus(Qt.FocusReason.OtherFocusReason)
        return True

    # --- settings protocol -------------------------------------------------------------
    def settings_value(self) -> str:
        return self.value()

    def set_settings_value(self, value) -> bool:
        if not isinstance(value, str) or value not in self._buttons:
            return False
        self.set_value(value)
        return True
