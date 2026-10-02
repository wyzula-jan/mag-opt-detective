"""Collapsible section: a header button and a body whose height animates."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QPalette
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from mag_opt_detective.gui import icons
from mag_opt_detective.gui.kit._common import DURATION_MS, QWIDGETSIZE_MAX, make_animation, to_bool


class _Reveal(QWidget):
    """Holds the body. While animating, the body keeps its natural height and is clipped."""

    def __init__(self, body: QWidget, parent: QWidget):
        super().__init__(parent)
        self._body = body
        body.setParent(self)
        self._animating = False

    def set_animating(self, animating: bool) -> None:
        self._animating = animating
        self.updateGeometry()
        self._place()

    def natural_height(self, width: int | None = None) -> int:
        width = self.width() if width is None else width
        if self._body.hasHeightForWidth():
            height = self._body.heightForWidth(width)
            if height >= 0:
                return height
        return self._body.sizeHint().height()

    def sizeHint(self) -> QSize:
        return self._body.sizeHint()

    def minimumSizeHint(self) -> QSize:
        hint = self._body.minimumSizeHint()
        return QSize(hint.width(), 0) if self._animating else hint

    def hasHeightForWidth(self) -> bool:
        return self._body.hasHeightForWidth()

    def heightForWidth(self, width: int) -> int:
        return self._body.heightForWidth(width)

    def _place(self) -> None:
        if self._animating:
            height = max(self.height(), self.natural_height())
            self._body.setGeometry(0, 0, self.width(), height)
        else:
            self._body.setGeometry(self.rect())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place()

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.LayoutRequest:  # the body's size hints changed
            self.updateGeometry()
            self._place()
        return super().event(event)


class _Header(QWidget):
    """Header row; a click anywhere on it toggles the section."""

    def __init__(self, section: CollapsibleSection):
        super().__init__(section)
        self._section = section

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self._section.set_expanded(not self._section.is_expanded())
            event.accept()
            return
        super().mouseReleaseEvent(event)


class CollapsibleSection(QWidget):
    """A section with a header (chevron, bold title, optional trailing widget) and a body.

    Put content into :meth:`body_layout`. ``toggled`` fires on every change of
    :meth:`is_expanded`. A collapsed body is hidden, so it leaves the Tab order.
    Settings protocol: bool (expanded).
    """

    toggled = Signal(bool)
    duration_ms = DURATION_MS  # animation length; tests may lengthen it per instance

    def __init__(
        self,
        title: str = "",
        parent: QWidget | None = None,
        *,
        expanded: bool = True,
        trailing: QWidget | None = None,
        separator: bool = True,
    ):
        super().__init__(parent)
        self._expanded = bool(expanded)
        self._separator = separator

        self.header = _Header(self)
        self.button = QToolButton(self.header)
        self.button.setProperty("kit", "section-header")
        self.button.setCheckable(True)
        self.button.setChecked(self._expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        font = self.button.font()
        font.setBold(True)
        self.button.setFont(font)
        self.button.clicked.connect(self.set_expanded)
        self._trailing: QWidget | None = None
        self._header_layout = QHBoxLayout(self.header)
        self._header_layout.setContentsMargins(10, 6, 14, 6)
        self._header_layout.setSpacing(6)
        self._header_layout.addWidget(self.button)
        self._header_layout.addStretch(1)

        self.body = QWidget()
        self._body_layout = QVBoxLayout(self.body)
        self._body_layout.setContentsMargins(14, 0, 14, 14)
        self._body_layout.setSpacing(14)
        self._reveal = _Reveal(self.body, self)
        self._reveal.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._reveal.setVisible(self._expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self._reveal)

        self._animation = make_animation(self)
        self._animation.valueChanged.connect(lambda v: self._reveal.setMaximumHeight(int(v)))
        self._animation.finished.connect(self._finish)
        self.set_title(title)
        if trailing is not None:
            self.set_trailing(trailing)
        self._update_chevron()

    # --- content -----------------------------------------------------------------------
    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def set_title(self, title: str) -> None:
        self.button.setText(title)
        self.button.setAccessibleName(title)

    def title(self) -> str:
        return self.button.text()

    def set_trailing(self, widget: QWidget | None) -> None:
        """A small widget at the end of the header (e.g. a sub-label); None removes it."""
        if self._trailing is not None:
            self._header_layout.removeWidget(self._trailing)
            self._trailing.setParent(None)
        self._trailing = widget
        if widget is not None:
            self._header_layout.addWidget(widget)

    def trailing(self) -> QWidget | None:
        return self._trailing

    # --- expanded state ----------------------------------------------------------------
    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool, animate: bool = True) -> None:
        expanded = bool(expanded)
        if expanded == self._expanded:
            return
        self._expanded = expanded
        with QSignalBlocker(self.button):
            self.button.setChecked(expanded)
        self._update_chevron()
        self._animation.stop()
        if animate and self.isVisible():
            start = self._reveal.height() if self._reveal.isVisible() else 0
            end = self._reveal.natural_height(self.width()) if expanded else 0
            self._reveal.setMaximumHeight(start)
            self._reveal.set_animating(True)
            self._reveal.setVisible(True)
            self._animation.setStartValue(start)
            self._animation.setEndValue(end)
            self._animation.setDuration(self.duration_ms)
            self._animation.start()
        else:
            self._finish()
        self.toggled.emit(expanded)

    def toggle(self, animate: bool = True) -> None:
        self.set_expanded(not self._expanded, animate)

    def is_animating(self) -> bool:
        return self._animation.state() == self._animation.State.Running

    def _finish(self) -> None:
        self._reveal.set_animating(False)
        self._reveal.setMaximumHeight(QWIDGETSIZE_MAX)
        self._reveal.setVisible(self._expanded)

    def _update_chevron(self) -> None:
        icons.set_icon(self.button, "chevron-down" if self._expanded else "chevron-right")

    def paintEvent(self, event) -> None:
        if not self._separator:
            return
        p = QPainter(self)
        p.setPen(self.palette().color(QPalette.ColorRole.Midlight))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        p.end()

    # --- settings protocol -------------------------------------------------------------
    def settings_value(self) -> bool:
        return self._expanded

    def set_settings_value(self, value) -> bool:
        expanded = to_bool(value)
        if expanded is None:
            return False
        self.set_expanded(expanded, animate=False)
        return True
