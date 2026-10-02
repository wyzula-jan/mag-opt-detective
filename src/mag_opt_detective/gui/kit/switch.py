"""Toggle switch (a painted QCheckBox)."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QPen
from PySide6.QtWidgets import QCheckBox, QWidget


class Switch(QCheckBox):
    """A checkbox drawn as a sliding switch; the optional text is drawn to its right.

    Mouse, Space and the settings store treat it exactly like a QCheckBox.
    """

    TRACK_W, TRACK_H = 30, 18
    KNOB = 14
    MARGIN = 2  # room for the focus ring
    SPACING = 8

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setTristate(False)

    def sizeHint(self) -> QSize:
        width = self.TRACK_W + 2 * self.MARGIN
        if self.text():
            width += self.SPACING + self.fontMetrics().horizontalAdvance(self.text())
        height = max(self.TRACK_H + 2 * self.MARGIN, self.fontMetrics().height())
        return QSize(width, height)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def hitButton(self, pos: QPoint) -> bool:
        return self.rect().contains(pos)

    def _track(self) -> QRectF:
        top = (self.height() - self.TRACK_H) / 2
        return QRectF(self.MARGIN, top, self.TRACK_W, self.TRACK_H)

    def paintEvent(self, event) -> None:
        pal = self.palette()
        group = QPalette.ColorGroup.Active if self.isEnabled() else QPalette.ColorGroup.Disabled
        on = self.isChecked()
        track = self._track()
        radius = self.TRACK_H / 2

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.isEnabled():
            p.setOpacity(0.45)
        accent = pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Highlight)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(accent if on else pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Mid))
        p.drawRoundedRect(track, radius, radius)

        inset = (self.TRACK_H - self.KNOB) / 2
        x = track.right() - inset - self.KNOB if on else track.left() + inset
        p.setBrush(pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Base))
        p.drawEllipse(QRectF(x, track.top() + inset, self.KNOB, self.KNOB))

        if self.hasFocus():
            ring = QColor(accent)
            ring.setAlphaF(0.45)
            p.setPen(QPen(ring, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(track.adjusted(-1.5, -1.5, 1.5, 1.5), radius + 1.5, radius + 1.5)

        if self.text():
            p.setOpacity(1.0)
            p.setPen(pal.color(group, QPalette.ColorRole.WindowText))
            left = track.right() + self.SPACING
            text_rect = QRectF(left, 0, self.width() - left, self.height())
            p.drawText(
                text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text()
            )
        p.end()
