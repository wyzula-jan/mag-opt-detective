"""What an empty area shows instead of its content: icon, bold title, a line, an action."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from mag_opt_detective.gui import icons
from mag_opt_detective.gui.kit.button import SmallButton
from mag_opt_detective.gui.theme import current_tokens

ICON_SIZE = 30
TEXT_WIDTH = 320  # about 38 characters, as in the mockup


class EmptyState(QWidget):
    """A centred message on the plot background (``.empty-state`` of the mockup).

    :meth:`show_message` sets the icon, the title, the explanation and an optional action
    button; the widget paints its own background, so it hides whatever it is placed over.
    """

    def __init__(self, parent: QWidget | None = None, background: str = "plot-bg"):
        super().__init__(parent)
        self._background = background
        self._icon = "layers"
        self._action: Callable[[], object] | None = None
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(ICON_SIZE, ICON_SIZE)
        self.title_label = QLabel()
        self.title_label.setTextFormat(Qt.TextFormat.PlainText)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(True)
        font = self.title_label.font()
        font.setBold(True)
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() * 1.08)
        self.title_label.setFont(font)
        self.text_label = QLabel()
        self.text_label.setProperty("kit", "muted")
        self.text_label.setTextFormat(Qt.TextFormat.PlainText)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setWordWrap(True)
        self.action_button = SmallButton("")  # the mockup's .btn.sm
        self.action_button.setVisible(False)
        self.action_button.clicked.connect(self._on_action)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(8)
        layout.addStretch(1)
        center = Qt.AlignmentFlag.AlignHCenter
        layout.addWidget(self.icon_label, 0, center)
        layout.addWidget(self.title_label, 0, center)
        layout.addWidget(self.text_label, 0, center)
        layout.addSpacing(2)
        layout.addWidget(self.action_button, 0, center)
        layout.addStretch(1)
        self._update_icon()

    def show_message(
        self,
        icon: str,
        title: str,
        text: str = "",
        action_text: str | None = None,
        action: Callable[[], object] | None = None,
    ) -> None:
        """Show *title* and *text*; the button (when *action_text* is given) runs *action*."""
        self._icon = icon
        self._update_icon()
        self.title_label.setText(title)
        self.text_label.setText(text)
        self.text_label.setVisible(bool(text))
        self.action_button.setText(action_text or "")
        self.action_button.setVisible(bool(action_text))
        self._action = action
        self._fit_labels()
        self.setAccessibleName(title)
        self.setAccessibleDescription(text)

    def title(self) -> str:
        return self.title_label.text()

    def text(self) -> str:
        return self.text_label.text()

    def _on_action(self) -> None:
        if self._action is not None:
            self._action()

    def _update_icon(self) -> None:
        ratio = self.devicePixelRatioF()
        self.icon_label.setPixmap(icons.pixmap(self._icon, ICON_SIZE, "faint", scale=ratio))

    def sizeHint(self) -> QSize:
        return QSize(TEXT_WIDTH + 48, 200)

    def _fit_labels(self) -> None:
        """Wrapped labels in a centring layout get their narrow hint: size them here."""
        margins = self.layout().contentsMargins()
        width = max(80, min(TEXT_WIDTH, self.width() - margins.left() - margins.right()))
        for label in (self.title_label, self.text_label):
            label.setFixedSize(width, label.heightForWidth(width))

    def resizeEvent(self, event) -> None:
        self._fit_labels()
        super().resizeEvent(event)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.PaletteChange:
            self._update_icon()
        super().changeEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), current_tokens()[self._background])
        painter.end()
