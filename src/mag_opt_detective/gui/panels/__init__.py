"""Panels of the left rail: Sample, Reference, Processing, Library and Points.

Each module exposes ``install(window)``: it builds its panel, adds it to the rail with
``window.add_panel``, wires its handlers to ``window.controller`` and binds its own settings
keys with ``window.persistence``.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from mag_opt_detective.gui.widgets import Separator


class _FitWidth(QObject):
    """Keeps *content* no wider than the scroll area's viewport: the panel never scrolls
    sideways, so one wide widget (a long file name) cannot push the others out of view."""

    def __init__(self, scroll: QScrollArea, content: QWidget):
        super().__init__(scroll)
        self._content = content
        scroll.viewport().installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Resize:
            self._content.setMaximumWidth(max(1, event.size().width()))
        return False


class PanelPage(QWidget):
    """A rail panel: a header (title and subtitle) above its scrolling content, which is
    never wider than the panel."""

    def __init__(self, title: str, subtitle: str, content: QWidget, parent=None):
        super().__init__(parent)
        self.content = content
        head = QWidget()
        head_layout = QVBoxLayout(head)
        head_layout.setContentsMargins(14, 12, 14, 10)
        head_layout.setSpacing(2)
        self.title = QLabel(title)
        font = self.title.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.08)
        self.title.setFont(font)
        self.subtitle = QLabel(subtitle)
        self.subtitle.setProperty("kit", "muted")
        self.subtitle.setWordWrap(True)
        head_layout.addWidget(self.title)
        head_layout.addWidget(self.subtitle)
        self.subtitle.setVisible(bool(subtitle))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        self.scroll = scroll
        _FitWidth(scroll, content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(head)
        layout.addWidget(Separator())
        layout.addWidget(scroll, stretch=1)

    def set_subtitle(self, text: str) -> None:
        self.subtitle.setText(text)
        self.subtitle.setVisible(bool(text))
