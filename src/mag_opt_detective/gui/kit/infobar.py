"""Non-modal message bar (error, warning or info)."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QIcon, QKeyEvent, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.gui import icons
from mag_opt_detective.gui.kit._common import set_style_property

LEVELS: dict[str, tuple[str, str]] = {  # level: (icon, colour token)
    "error": ("triangle-alert", "err"),
    "warning": ("triangle-alert", "warn"),
    "info": ("info", "accent"),
}


class _IconView(QWidget):
    """A fixed-size icon, painted at the pixel ratio of whatever screen it is on."""

    def __init__(self, size: int):
        super().__init__()
        self.setFixedSize(size, size)
        self._icon = QIcon()

    def icon(self) -> QIcon:
        return self._icon

    def set_icon(self, icon: QIcon) -> None:
        self._icon = icon
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        self._icon.paint(painter, self.rect())
        painter.end()


class InfoBar(QFrame):
    """Icon, bold title, text, an optional action button and a close button.

    Hidden until :meth:`show_message`. Colours come from the theme stylesheet
    (``QFrame[kit="infobar"][level=...]``). ``closed`` fires whenever a shown bar is dismissed:
    by its close button, Escape, its action button or :meth:`dismiss`.
    """

    closed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("kit", "infobar")
        self.setProperty("level", "error")
        self._level = "error"
        self._action: Callable[[], object] | None = None

        self.icon_view = _IconView(16)
        self.title_label = QLabel()
        self.title_label.setTextFormat(Qt.TextFormat.PlainText)
        self.title_label.setWordWrap(True)
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)
        self.text_label = QLabel()
        self.text_label.setProperty("kit", "muted")
        self.text_label.setTextFormat(Qt.TextFormat.PlainText)
        self.text_label.setWordWrap(True)
        self.action_button = QPushButton()
        self.action_button.setProperty("kit", "button")
        self.action_button.setVisible(False)
        self.action_button.clicked.connect(self._on_action)
        self.close_button = QToolButton()
        self.close_button.setProperty("kit", "tool")
        self.close_button.setAccessibleName("Dismiss")
        self.close_button.setToolTip("Dismiss")
        icons.set_icon(self.close_button, "x", "muted")
        self.close_button.clicked.connect(self.dismiss)

        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(2)
        texts.addWidget(self.title_label)
        texts.addWidget(self.text_label)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 8, 8)
        layout.setSpacing(10)
        top = Qt.AlignmentFlag.AlignTop
        layout.addWidget(self.icon_view, 0, top)
        layout.addLayout(texts, 1)
        layout.addWidget(self.action_button, 0, top)
        layout.addWidget(self.close_button, 0, top)
        self._update_icon()
        self.hide()

    def show_message(
        self,
        level: str,
        title: str,
        text: str = "",
        action_text: str | None = None,
        action: Callable[[], object] | None = None,
    ) -> None:
        """Show a message; *action* runs (after the bar closes) when its button is clicked."""
        if level not in LEVELS:
            raise ValueError(f"level must be one of {tuple(LEVELS)}")
        self._level = level
        set_style_property(self, "level", level)
        self._update_icon()
        self.title_label.setText(title)
        self.text_label.setText(text)
        self.text_label.setVisible(bool(text))
        self.action_button.setText(action_text or "")
        self.action_button.setVisible(bool(action_text))
        self._action = action
        self.setAccessibleName(title)
        self.setAccessibleDescription(text)
        self.show()

    def level(self) -> str:
        return self._level

    def dismiss(self) -> None:
        if self.isHidden():
            return
        self.hide()
        self.closed.emit()

    def _on_action(self) -> None:
        action = self._action
        self.dismiss()
        if action is not None:
            action()

    def _update_icon(self) -> None:
        name, token = LEVELS[self._level]
        self.icon_view.set_icon(icons.icon(name, token))

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.PaletteChange:
            self._update_icon()
        super().changeEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.dismiss()
            event.accept()
            return
        super().keyPressEvent(event)
